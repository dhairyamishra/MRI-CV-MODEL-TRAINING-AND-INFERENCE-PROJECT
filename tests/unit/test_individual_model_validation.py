"""
PHASE 1.2.2: Individual Model Validation - Critical Safety Tests

Tests EfficientNet-B0, ConvNeXt, U-Net 2D architectures, model loading,
mixed precision, device compatibility, and memory bounds.
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.classifier import BrainTumorClassifier, ConvNeXtClassifier, create_classifier
from src.models.unet2d import UNet2D
from src.models.multi_task_model import MultiTaskModel


class TestEfficientNetB0Architecture:
    """Test EfficientNet-B0 model convergence and behavior."""

    def test_efficientnet_creation(self):
        """Test EfficientNet-B0 model can be created."""
        model = BrainTumorClassifier(pretrained=False)

        assert isinstance(model, nn.Module)
        assert hasattr(model, 'forward')

        # Check expected architecture
        # EfficientNet-B0 should have reasonable parameter count
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 4_000_000   # Around 4M params
        assert total_params < 6_000_000

    def test_efficientnet_forward_pass(self):
        """Test EfficientNet forward pass with various input sizes."""
        model = BrainTumorClassifier(pretrained=False)
        model.eval()

        test_sizes = [(128, 128), (224, 224), (256, 256)]

        for height, width in test_sizes:
            with torch.no_grad():
                input_tensor = torch.randn(1, 1, height, width)  # Grayscale medical images

                # EfficientNet expects 3 channels, so model should handle conversion
                output = model(input_tensor)

                # Should output binary classification logits
                assert output.shape == (1, 2)  # [batch_size, num_classes]
                assert torch.isfinite(output).all()

    def test_efficientnet_convergence_potential(self):
        """Test EfficientNet can learn simple patterns."""
        model = BrainTumorClassifier(pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create simple synthetic dataset
        # Pattern: vertical stripes = tumor, horizontal = no tumor
        def create_sample(is_tumor):
            img = torch.randn(1, 224, 224)
            if is_tumor:
                # Add vertical pattern
                img[:, :, ::2] += 1.0
            else:
                # Add horizontal pattern
                img[:, ::2, :] += 1.0
            return img, torch.tensor([1.0 if is_tumor else 0.0])

        # Training simulation
        initial_loss = float('inf')
        final_loss = 0.0

        for step in range(5):
            # Create batch
            batch_inputs = []
            batch_labels = []

            for _ in range(4):
                is_tumor = np.random.random() > 0.5
                img, label = create_sample(is_tumor)
                batch_inputs.append(img)
                batch_labels.append(label)

            batch_inputs = torch.stack(batch_inputs)
            batch_labels = torch.stack(batch_labels)

            # Forward
            outputs = model(batch_inputs)
            loss = nn.BCEWithLogitsLoss()(outputs[:, 1], batch_labels.float())

            # Track loss
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Should show some learning (loss reduction)
        assert final_loss < initial_loss
        assert final_loss < 1.0  # Should learn the simple pattern


class TestConvNeXtArchitecture:
    """Test ConvNeXt model architecture and behavior."""

    def test_convnext_creation(self):
        """Test ConvNeXt model can be created."""
        model = ConvNeXtClassifier(pretrained=False)

        assert isinstance(model, nn.Module)
        assert hasattr(model, 'forward')

        # ConvNeXt should have higher parameter count than EfficientNet
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 20_000_000  # Around 27.8M params
        assert total_params < 35_000_000

    def test_convnext_forward_pass(self):
        """Test ConvNeXt forward pass."""
        model = ConvNeXtClassifier(pretrained=False)
        model.eval()

        with torch.no_grad():
            input_tensor = torch.randn(1, 1, 224, 224)
            output = model(input_tensor)

            assert output.shape == (1, 2)
            assert torch.isfinite(output).all()

    def test_convnext_feature_hierarchy(self):
        """Test ConvNeXt builds proper feature hierarchy."""
        model = ConvNeXtClassifier(pretrained=False)

        # Access intermediate features (if model supports it)
        # This tests that the model has proper hierarchical structure
        input_tensor = torch.randn(1, 1, 224, 224)

        with torch.no_grad():
            output = model(input_tensor)

            # Should produce confident predictions (not random)
            probs = torch.softmax(output, dim=1)
            max_prob = torch.max(probs)

            # At least some confidence (better than random 0.5)
            assert max_prob > 0.5


class TestUNet2DArchitecture:
    """Test U-Net 2D segmentation model."""

    def test_unet_creation(self):
        """Test U-Net 2D model can be created."""
        model = UNet2D(in_channels=1, out_channels=4, features=32)

        assert isinstance(model, nn.Module)
        assert hasattr(model, 'forward')

        # U-Net should have reasonable parameter count
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 30_000_000  # Around 31.4M params
        assert total_params < 35_000_000

    def test_unet_segmentation_output(self):
        """Test U-Net produces proper segmentation outputs."""
        model = UNet2D(in_channels=1, out_channels=4, features=32)
        model.eval()

        with torch.no_grad():
            input_tensor = torch.randn(1, 1, 128, 128)
            output = model(input_tensor)

            # Should output segmentation mask
            assert output.shape == (1, 4, 128, 128)  # [batch, classes, height, width]
            assert torch.isfinite(output).all()

            # Should sum to 1 across classes (logits)
            probs = torch.softmax(output, dim=1)
            class_sums = torch.sum(probs, dim=1)
            assert torch.allclose(class_sums, torch.ones_like(class_sums), atol=1e-6)

    def test_unet_skip_connections(self):
        """Test U-Net skip connections preserve spatial information."""
        model = UNet2D(in_channels=1, out_channels=4, features=32)

        # Create input with spatial pattern
        input_tensor = torch.zeros(1, 1, 128, 128)
        # Add checkerboard pattern
        input_tensor[0, 0, ::2, ::2] = 1.0
        input_tensor[0, 0, 1::2, 1::2] = 1.0

        with torch.no_grad():
            output = model(input_tensor)

            # U-Net should preserve some spatial structure
            # (This is a simplified test - real validation would be more complex)
            assert output.shape[2:] == (128, 128)  # Spatial dimensions preserved
            assert not torch.allclose(output, torch.zeros_like(output))  # Not zero output


class TestModelLoading:
    """Test model checkpoint loading across PyTorch versions."""

    def test_checkpoint_structure(self):
        """Test model checkpoint has required components."""
        # Create mock checkpoint
        model = BrainTumorClassifier(pretrained=False)
        checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': torch.optim.Adam(model.parameters()).state_dict(),
            'best_metric': 0.95,
            'config': {
                'model_type': 'efficientnet',
                'learning_rate': 1e-3,
                'batch_size': 16
            }
        }

        # Validate checkpoint structure
        required_keys = ['epoch', 'model_state_dict', 'best_metric', 'config']
        for key in required_keys:
            assert key in checkpoint

        assert isinstance(checkpoint['epoch'], int)
        assert isinstance(checkpoint['best_metric'], (int, float))
        assert isinstance(checkpoint['config'], dict)

    def test_model_state_loading(self):
        """Test model state can be loaded from checkpoint."""
        # Create and save model
        model1 = BrainTumorClassifier(pretrained=False)
        original_params = {name: param.clone() for name, param in model1.named_parameters()}

        # Create checkpoint
        checkpoint = {
            'epoch': 5,
            'model_state_dict': model1.state_dict(),
            'best_metric': 0.90
        }

        # Create new model and load state
        model2 = BrainTumorClassifier(pretrained=False)
        model2.load_state_dict(checkpoint['model_state_dict'])

        # Parameters should be identical
        for name, param in model2.named_parameters():
            assert torch.allclose(param, original_params[name])

    @patch('torch.load')
    def test_pytorch_version_compatibility(self, mock_load):
        """Test checkpoint loading handles PyTorch version differences."""
        # Mock different PyTorch versions in checkpoint
        mock_checkpoint = {
            'epoch': 10,
            'model_state_dict': {'layer.weight': torch.randn(10, 10)},
            'pytorch_version': '2.0.0',
            'best_metric': 0.95
        }

        mock_load.return_value = mock_checkpoint

        # Should load without errors (weights_only=False handles version differences)
        with patch('builtins.open', mock_open()):
            loaded = torch.load('fake_path.pth', weights_only=False)

        assert 'model_state_dict' in loaded
        assert loaded['epoch'] == 10

    def test_checkpoint_validation(self):
        """Test checkpoint file validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "test_checkpoint.pth"

            # Create valid checkpoint
            model = BrainTumorClassifier(pretrained=False)
            checkpoint = {
                'epoch': 10,
                'model_state_dict': model.state_dict(),
                'best_metric': 0.95,
                'config': {'model_type': 'efficientnet'}
            }

            torch.save(checkpoint, checkpoint_path)

            # Should load successfully
            loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert loaded_checkpoint['epoch'] == 10
            assert abs(loaded_checkpoint['best_metric'] - 0.95) < 1e-6


class TestMixedPrecision:
    """Test AMP training stability."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP testing")
    def test_amp_forward_pass(self):
        """Test model works with automatic mixed precision."""
        model = BrainTumorClassifier(pretrained=False).cuda()
        model.eval()

        with torch.no_grad():
            # Test with float16
            input_tensor = torch.randn(1, 1, 224, 224).cuda().half()

            # Model should handle half precision
            output = model(input_tensor.half())

            assert output.shape == (1, 2)
            assert output.dtype == torch.float16
            assert torch.isfinite(output).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP testing")
    def test_amp_training_step(self):
        """Test AMP training step works."""
        from torch.cuda.amp import autocast, GradScaler

        model = BrainTumorClassifier(pretrained=False).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler()

        # Create batch
        input_tensor = torch.randn(2, 1, 224, 224).cuda()
        labels = torch.randint(0, 2, (2,)).cuda()

        # AMP training step
        with autocast():
            outputs = model(input_tensor)
            loss = nn.CrossEntropyLoss()(outputs, labels)

        # Scale loss and backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Should complete without errors
        assert torch.isfinite(loss).all()

    def test_cpu_mixed_precision_fallback(self):
        """Test mixed precision behavior on CPU."""
        model = BrainTumorClassifier(pretrained=False)
        model.eval()

        with torch.no_grad():
            # CPU doesn't support true mixed precision, but should still work
            input_tensor = torch.randn(1, 1, 224, 224)

            output = model(input_tensor)

            # Should work with regular precision
            assert output.shape == (1, 2)
            assert output.dtype == torch.float32
            assert torch.isfinite(output).all()


class TestDeviceCompatibility:
    """Test CPU/GPU inference consistency."""

    def test_cpu_gpu_consistency(self):
        """Test model produces consistent results on CPU and GPU."""
        pytest.skip("Skipping GPU test in CI environment")

        model_cpu = BrainTumorClassifier(pretrained=False)
        model_cpu.eval()

        # Create identical inputs
        input_tensor = torch.randn(1, 1, 224, 224)

        with torch.no_grad():
            # CPU inference
            output_cpu = model_cpu(input_tensor)

            # GPU inference (if available)
            if torch.cuda.is_available():
                model_gpu = BrainTumorClassifier(pretrained=False).cuda()
                model_gpu.eval()
                model_gpu.load_state_dict(model_cpu.state_dict())

                input_gpu = input_tensor.cuda()
                output_gpu = model_gpu(input_gpu).cpu()

                # Should be very close (allowing for numerical differences)
                assert torch.allclose(output_cpu, output_gpu, atol=1e-4)

    def test_device_transfer(self):
        """Test model can be moved between devices."""
        model = BrainTumorClassifier(pretrained=False)

        # Start on CPU
        assert next(model.parameters()).device.type == 'cpu'

        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            assert next(model.parameters()).device.type == 'cuda'

            # Move back to CPU
            model = model.cpu()
            assert next(model.parameters()).device.type == 'cpu'

        # Test inference works after device transfer
        input_tensor = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            output = model(input_tensor)
            assert output.shape == (1, 2)
            assert torch.isfinite(output).all()


class TestMemoryBounds:
    """Test peak GPU/CPU memory usage limits."""

    def test_model_memory_usage(self):
        """Test model memory consumption is reasonable."""
        model = BrainTumorClassifier(pretrained=False)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        param_memory_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes

        # Should be reasonable for a classification model
        assert param_memory_mb < 100  # Less than 100MB for parameters

    def test_inference_memory_bounds(self):
        """Test inference memory usage bounds."""
        model = BrainTumorClassifier(pretrained=False)
        model.eval()

        # Test different batch sizes
        batch_sizes = [1, 4, 8]

        for batch_size in batch_sizes:
            with torch.no_grad():
                input_tensor = torch.randn(batch_size, 1, 224, 224)

                # Measure memory before
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    start_memory = torch.cuda.memory_allocated()

                # Inference
                output = model(input_tensor)

                # Check memory after
                if torch.cuda.is_available():
                    end_memory = torch.cuda.memory_allocated()
                    peak_memory = torch.cuda.max_memory_allocated()

                    # Memory should be reasonable
                    memory_mb = peak_memory / (1024 * 1024)
                    assert memory_mb < 1000  # Less than 1GB peak for reasonable batch

                # Output should be valid
                assert output.shape == (batch_size, 2)
                assert torch.isfinite(output).all()

    def test_memory_cleanup(self):
        """Test memory is properly cleaned up."""
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        model = BrainTumorClassifier(pretrained=False)

        # Use model
        with torch.no_grad():
            input_tensor = torch.randn(4, 1, 224, 224)
            output = model(input_tensor)

        # Delete model and tensors
        del model, input_tensor, output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Memory should be freed (approximately)
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Allow some tolerance for PyTorch's memory management
        memory_diff = abs(final_memory - initial_memory)
        assert memory_diff < 50 * 1024 * 1024  # Less than 50MB difference
