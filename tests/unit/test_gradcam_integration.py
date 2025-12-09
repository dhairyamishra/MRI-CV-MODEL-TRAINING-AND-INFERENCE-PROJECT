"""
PHASE 1.1.11: Grad-CAM Integration Tests - Real Module Coverage

Tests the actual Grad-CAM implementation in src/eval/grad_cam.py
to increase coverage from 16% to >70%.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
import cv2
import tempfile
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eval.grad_cam import GradCAM


class SimpleCNN(nn.Module):
    """Simple CNN for testing Grad-CAM."""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_cam_target_layer(self):
        """Get the last convolutional layer for Grad-CAM."""
        return self.features[-3]  # Last Conv2d layer


class TestGradCAMInitialization:
    """Test GradCAM class initialization."""
    
    def test_gradcam_initialization(self):
        """Test GradCAM initializes correctly."""
        model = SimpleCNN()
        target_layer = model.get_cam_target_layer()
        
        grad_cam = GradCAM(model, target_layer)
        
        assert grad_cam.model is model
        assert grad_cam.target_layer is target_layer
        assert grad_cam.gradients is None
        assert grad_cam.activations is None
    
    def test_hooks_registration(self):
        """Test forward and backward hooks are registered."""
        model = SimpleCNN()
        target_layer = model.get_cam_target_layer()
        
        # Count hooks before
        forward_hooks_before = len(target_layer._forward_hooks)
        backward_hooks_before = len(target_layer._backward_hooks)
        
        grad_cam = GradCAM(model, target_layer)
        
        # Count hooks after
        forward_hooks_after = len(target_layer._forward_hooks)
        backward_hooks_after = len(target_layer._backward_hooks)
        
        assert forward_hooks_after > forward_hooks_before
        assert backward_hooks_after > backward_hooks_before


class TestGradCAMGeneration:
    """Test Grad-CAM heatmap generation."""
    
    def test_generate_cam_basic(self):
        """Test basic CAM generation."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Create dummy input
        input_tensor = torch.randn(1, 1, 64, 64)
        
        # Generate CAM
        cam = grad_cam.generate_cam(input_tensor)
        
        # Validate output
        assert isinstance(cam, np.ndarray)
        assert cam.ndim == 2  # Should be 2D heatmap
        assert np.all((cam >= 0) & (cam <= 1))  # Normalized to [0, 1]
    
    def test_generate_cam_with_target_class(self):
        """Test CAM generation with specified target class."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        input_tensor = torch.randn(1, 1, 64, 64)
        
        # Generate CAM for class 0
        cam_class0 = grad_cam.generate_cam(input_tensor, target_class=0)
        
        # Generate CAM for class 1
        cam_class1 = grad_cam.generate_cam(input_tensor, target_class=1)
        
        # CAMs should be different for different classes
        assert not np.array_equal(cam_class0, cam_class1)
    
    def test_generate_cam_without_target_class(self):
        """Test CAM generation uses predicted class when target_class=None."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        input_tensor = torch.randn(1, 1, 64, 64)
        
        # Generate CAM without specifying target class
        cam = grad_cam.generate_cam(input_tensor, target_class=None)
        
        assert isinstance(cam, np.ndarray)
        assert cam.shape[0] > 0
        assert cam.shape[1] > 0
    
    def test_cam_normalization(self):
        """Test CAM is properly normalized to [0, 1]."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        input_tensor = torch.randn(1, 1, 64, 64)
        cam = grad_cam.generate_cam(input_tensor)
        
        # Check normalization
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0
        # Should have some variation (not all zeros or all ones)
        assert cam.std() > 0.01
    
    def test_cam_spatial_dimensions(self):
        """Test CAM spatial dimensions match feature map size."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        input_tensor = torch.randn(1, 1, 64, 64)
        cam = grad_cam.generate_cam(input_tensor)
        
        # CAM should have spatial dimensions matching the target layer output
        # The target layer is after first MaxPool2d(2), so 64x64 -> 32x32
        # (before the second MaxPool2d)
        assert cam.shape[0] == 32
        assert cam.shape[1] == 32


class TestGradCAMOverlay:
    """Test Grad-CAM overlay generation."""
    
    def test_generate_overlay_basic(self):
        """Test basic overlay generation."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Create dummy input and CAM
        input_image = np.random.rand(64, 64)
        cam = np.random.rand(16, 16)
        
        # Generate overlay
        overlay = grad_cam.generate_overlay(input_image, cam)
        
        # Validate output
        assert isinstance(overlay, np.ndarray)
        assert overlay.ndim == 3  # Should be RGB
        assert overlay.shape[2] == 3  # 3 color channels
        assert overlay.shape[0] == 64  # Height matches input
        assert overlay.shape[1] == 64  # Width matches input
    
    def test_generate_overlay_with_3d_input(self):
        """Test overlay generation with 3D input image."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Create 3D input (H, W, 1)
        input_image = np.random.rand(64, 64, 1)
        cam = np.random.rand(16, 16)
        
        # Generate overlay
        overlay = grad_cam.generate_overlay(input_image, cam)
        
        # Should still produce valid RGB overlay
        assert overlay.shape == (64, 64, 3)
    
    def test_overlay_alpha_blending(self):
        """Test different alpha values produce different blending."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        input_image = np.random.rand(64, 64)
        cam = np.random.rand(16, 16)
        
        # Generate overlays with different alpha values
        overlay_alpha_0 = grad_cam.generate_overlay(input_image, cam, alpha=0.0)
        overlay_alpha_05 = grad_cam.generate_overlay(input_image, cam, alpha=0.5)
        overlay_alpha_1 = grad_cam.generate_overlay(input_image, cam, alpha=1.0)
        
        # Different alpha values should produce different results
        assert not np.array_equal(overlay_alpha_0, overlay_alpha_05)
        assert not np.array_equal(overlay_alpha_05, overlay_alpha_1)
    
    def test_overlay_colormap_options(self):
        """Test different colormap options."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        input_image = np.random.rand(64, 64)
        cam = np.random.rand(16, 16)
        
        # Test different colormaps
        colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_VIRIDIS]
        
        overlays = []
        for colormap in colormaps:
            overlay = grad_cam.generate_overlay(input_image, cam, colormap=colormap)
            overlays.append(overlay)
            assert overlay.shape == (64, 64, 3)
        
        # Different colormaps should produce different results
        assert not np.array_equal(overlays[0], overlays[1])
        assert not np.array_equal(overlays[1], overlays[2])
    
    def test_overlay_preserves_image_structure(self):
        """Test overlay preserves underlying image structure."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Create input with clear structure
        input_image = np.zeros((64, 64))
        input_image[20:40, 20:40] = 1.0  # Bright square
        
        cam = np.random.rand(16, 16)
        
        # Generate overlay with low alpha (mostly original image)
        overlay = grad_cam.generate_overlay(input_image, cam, alpha=0.2)
        
        # Overlay should still show the bright square region
        square_region = overlay[20:40, 20:40, :]
        background_region = overlay[0:10, 0:10, :]
        
        # Square region should be brighter than background
        assert np.mean(square_region) > np.mean(background_region)


class TestGradCAMHooks:
    """Test Grad-CAM hook mechanisms."""
    
    def test_activation_hook_saves_activations(self):
        """Test forward hook saves activations correctly."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Initially None
        assert grad_cam.activations is None
        
        # Forward pass
        input_tensor = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Activations should be saved
        assert grad_cam.activations is not None
        assert isinstance(grad_cam.activations, torch.Tensor)
    
    def test_gradient_hook_saves_gradients(self):
        """Test backward hook saves gradients correctly."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Initially None
        assert grad_cam.gradients is None
        
        # Forward and backward pass
        input_tensor = torch.randn(1, 1, 64, 64, requires_grad=True)
        output = model(input_tensor)
        loss = output[0, 0]
        loss.backward()
        
        # Gradients should be saved
        assert grad_cam.gradients is not None
        assert isinstance(grad_cam.gradients, torch.Tensor)


class TestGradCAMEdgeCases:
    """Test Grad-CAM edge cases and error handling."""
    
    def test_cam_with_zero_gradients(self):
        """Test CAM generation when gradients are zero."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Create input that might produce zero gradients
        input_tensor = torch.zeros(1, 1, 64, 64)
        
        # Should still generate valid CAM (might be uniform)
        cam = grad_cam.generate_cam(input_tensor)
        
        assert isinstance(cam, np.ndarray)
        assert cam.shape[0] > 0
        assert np.all((cam >= 0) & (cam <= 1))
    
    def test_cam_with_small_input(self):
        """Test CAM generation with small input size."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Small input
        input_tensor = torch.randn(1, 1, 32, 32)
        
        cam = grad_cam.generate_cam(input_tensor)
        
        assert isinstance(cam, np.ndarray)
        assert cam.ndim == 2
    
    def test_overlay_with_uniform_cam(self):
        """Test overlay generation with uniform CAM."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        input_image = np.random.rand(64, 64)
        cam = np.ones((16, 16)) * 0.5  # Uniform CAM
        
        overlay = grad_cam.generate_overlay(input_image, cam)
        
        assert overlay.shape == (64, 64, 3)
        assert np.all(np.isfinite(overlay))
    
    def test_overlay_with_extreme_values(self):
        """Test overlay with extreme input values."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Extreme input values
        input_image = np.random.rand(64, 64) * 1000  # Very large values
        cam = np.random.rand(16, 16)
        
        overlay = grad_cam.generate_overlay(input_image, cam)
        
        # Should still produce valid overlay with normalization
        assert overlay.shape == (64, 64, 3)
        assert np.all(overlay >= 0)
        assert np.all(overlay <= 255)


class TestGradCAMBatchProcessing:
    """Test Grad-CAM with batch processing scenarios."""
    
    def test_multiple_cam_generations(self):
        """Test generating multiple CAMs sequentially."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Generate multiple CAMs
        cams = []
        for _ in range(5):
            input_tensor = torch.randn(1, 1, 64, 64)
            cam = grad_cam.generate_cam(input_tensor)
            cams.append(cam)
        
        # All should be valid
        assert len(cams) == 5
        for cam in cams:
            assert isinstance(cam, np.ndarray)
            assert cam.ndim == 2
    
    def test_cam_consistency(self):
        """Test CAM is consistent for same input."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Same input
        input_tensor = torch.randn(1, 1, 64, 64)
        
        # Generate CAM twice
        cam1 = grad_cam.generate_cam(input_tensor.clone())
        cam2 = grad_cam.generate_cam(input_tensor.clone())
        
        # Should be very similar (allowing for numerical precision)
        assert np.allclose(cam1, cam2, rtol=1e-5, atol=1e-7)


class TestGradCAMVisualizationQuality:
    """Test visualization quality metrics."""
    
    def test_cam_has_variation(self):
        """Test CAM has spatial variation (not uniform)."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Create input with structure
        input_tensor = torch.randn(1, 1, 64, 64)
        cam = grad_cam.generate_cam(input_tensor)
        
        # CAM should have some variation
        assert cam.std() > 0.01  # Not uniform
    
    def test_overlay_maintains_uint8_range(self):
        """Test overlay values are in valid uint8 range."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        input_image = np.random.rand(64, 64)
        cam = np.random.rand(16, 16)
        
        overlay = grad_cam.generate_overlay(input_image, cam)
        
        # Should be in [0, 255] range for uint8
        assert np.all(overlay >= 0)
        assert np.all(overlay <= 255)
    
    def test_overlay_has_color_information(self):
        """Test overlay has color information from heatmap."""
        model = SimpleCNN()
        model.eval()
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        input_image = np.random.rand(64, 64)
        cam = np.random.rand(16, 16)
        
        overlay = grad_cam.generate_overlay(input_image, cam)
        
        # RGB channels should have different values (colorful)
        r_channel = overlay[:, :, 0]
        g_channel = overlay[:, :, 1]
        b_channel = overlay[:, :, 2]
        
        # Not all channels should be identical (grayscale)
        assert not (np.array_equal(r_channel, g_channel) and 
                   np.array_equal(g_channel, b_channel))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
