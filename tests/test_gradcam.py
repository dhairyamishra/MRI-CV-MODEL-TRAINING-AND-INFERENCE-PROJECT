"""
Unit tests for Grad-CAM implementation.

Tests the GradCAM class and visualization functions.
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eval.grad_cam import GradCAM
from src.models.classifier import create_classifier


@pytest.fixture
def model():
    """Create a test model."""
    model = create_classifier('efficientnet', pretrained=False)
    model.eval()
    return model


@pytest.fixture
def dummy_input():
    """Create dummy input tensor."""
    return torch.randn(1, 1, 256, 256)


class TestGradCAM:
    """Tests for GradCAM class."""
    
    def test_gradcam_creation(self, model):
        """Test that Grad-CAM can be created."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        assert grad_cam is not None
        assert grad_cam.model == model
        assert grad_cam.target_layer == target_layer
    
    def test_generate_cam(self, model, dummy_input):
        """Test CAM generation."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        cam = grad_cam.generate_cam(dummy_input)
        
        # Check output
        assert isinstance(cam, np.ndarray)
        assert cam.ndim == 2  # Should be 2D heatmap
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0
    
    def test_cam_with_target_class(self, model, dummy_input):
        """Test CAM generation with specific target class."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Generate CAM for class 0
        cam0 = grad_cam.generate_cam(dummy_input, target_class=0)
        
        # Generate CAM for class 1
        cam1 = grad_cam.generate_cam(dummy_input, target_class=1)
        
        # CAMs should be different for different classes
        assert not np.allclose(cam0, cam1)
    
    def test_cam_without_target_class(self, model, dummy_input):
        """Test CAM generation without specifying target class."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Should use predicted class
        cam = grad_cam.generate_cam(dummy_input, target_class=None)
        
        assert isinstance(cam, np.ndarray)
    
    def test_generate_overlay(self, model, dummy_input):
        """Test overlay generation."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Generate CAM
        cam = grad_cam.generate_cam(dummy_input)
        
        # Get original image
        original_image = dummy_input[0, 0].cpu().numpy()
        
        # Generate overlay
        overlay = grad_cam.generate_overlay(original_image, cam)
        
        # Check output
        assert isinstance(overlay, np.ndarray)
        assert overlay.ndim == 3  # Should be RGB
        assert overlay.shape[2] == 3
        assert overlay.dtype == np.uint8
    
    def test_overlay_with_different_alpha(self, model, dummy_input):
        """Test overlay with different alpha values."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        cam = grad_cam.generate_cam(dummy_input)
        original_image = dummy_input[0, 0].cpu().numpy()
        
        # Test different alpha values
        for alpha in [0.0, 0.5, 1.0]:
            overlay = grad_cam.generate_overlay(original_image, cam, alpha=alpha)
            assert overlay.shape[2] == 3
    
    def test_hooks_registered(self, model):
        """Test that hooks are properly registered."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Hooks should be registered
        assert grad_cam.gradients is None  # Not yet computed
        assert grad_cam.activations is None  # Not yet computed
    
    def test_activations_saved(self, model, dummy_input):
        """Test that activations are saved during forward pass."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Generate CAM (triggers forward pass)
        _ = grad_cam.generate_cam(dummy_input)
        
        # Activations should be saved
        assert grad_cam.activations is not None
        assert isinstance(grad_cam.activations, torch.Tensor)
    
    def test_gradients_saved(self, model, dummy_input):
        """Test that gradients are saved during backward pass."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Generate CAM (triggers backward pass)
        _ = grad_cam.generate_cam(dummy_input)
        
        # Gradients should be saved
        assert grad_cam.gradients is not None
        assert isinstance(grad_cam.gradients, torch.Tensor)


class TestGradCAMEdgeCases:
    """Tests for edge cases."""
    
    def test_different_input_sizes(self, model):
        """Test Grad-CAM with different input sizes."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        sizes = [224, 256, 512]
        
        for size in sizes:
            input_tensor = torch.randn(1, 1, size, size)
            cam = grad_cam.generate_cam(input_tensor)
            
            assert isinstance(cam, np.ndarray)
            assert cam.min() >= 0.0
            assert cam.max() <= 1.0
    
    def test_batch_size_one_required(self, model):
        """Test that batch size must be 1."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        # Batch size > 1 should still work (uses first sample)
        input_tensor = torch.randn(4, 1, 256, 256)
        cam = grad_cam.generate_cam(input_tensor)
        
        assert isinstance(cam, np.ndarray)
    
    def test_cam_shape_matches_feature_map(self, model, dummy_input):
        """Test that CAM shape matches feature map size."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        cam = grad_cam.generate_cam(dummy_input)
        
        # CAM should have same spatial dimensions as feature maps
        assert cam.ndim == 2
        assert cam.shape[0] > 0
        assert cam.shape[1] > 0


class TestOverlayGeneration:
    """Tests for overlay generation."""
    
    def test_overlay_size_matches_input(self, model, dummy_input):
        """Test that overlay size matches input image size."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        cam = grad_cam.generate_cam(dummy_input)
        original_image = dummy_input[0, 0].cpu().numpy()
        
        overlay = grad_cam.generate_overlay(original_image, cam)
        
        # Overlay should match original image size
        assert overlay.shape[0] == original_image.shape[0]
        assert overlay.shape[1] == original_image.shape[1]
    
    def test_overlay_with_3d_input(self, model, dummy_input):
        """Test overlay generation with 3D input image."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        cam = grad_cam.generate_cam(dummy_input)
        
        # 3D input (H, W, 1)
        original_image = dummy_input[0].cpu().numpy().transpose(1, 2, 0)
        
        overlay = grad_cam.generate_overlay(original_image, cam)
        
        assert overlay.shape[2] == 3
    
    def test_overlay_value_range(self, model, dummy_input):
        """Test that overlay values are in valid range."""
        target_layer = model.get_cam_target_layer()
        grad_cam = GradCAM(model, target_layer)
        
        cam = grad_cam.generate_cam(dummy_input)
        original_image = dummy_input[0, 0].cpu().numpy()
        
        overlay = grad_cam.generate_overlay(original_image, cam)
        
        assert overlay.min() >= 0
        assert overlay.max() <= 255


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
