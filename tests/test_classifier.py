"""
Unit tests for classifier models.

Tests the BrainTumorClassifier and ConvNeXtClassifier implementations.
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.classifier import (
    BrainTumorClassifier,
    ConvNeXtClassifier,
    create_classifier
)


class TestBrainTumorClassifier:
    """Tests for BrainTumorClassifier."""
    
    def test_model_creation(self):
        """Test that model can be created."""
        model = BrainTumorClassifier(pretrained=False)
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        model = BrainTumorClassifier(pretrained=False)
        model.eval()
        
        # Create dummy input (batch_size=4, channels=1, height=256, width=256)
        x = torch.randn(4, 1, 256, 256)
        
        with torch.no_grad():
            output = model(x)
        
        # Check output shape (batch_size=4, num_classes=2)
        assert output.shape == (4, 2)
    
    def test_single_channel_input(self):
        """Test that model accepts single-channel input."""
        model = BrainTumorClassifier(pretrained=False)
        model.eval()
        
        x = torch.randn(1, 1, 256, 256)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 2)
    
    def test_feature_extraction(self):
        """Test feature extraction for Grad-CAM."""
        model = BrainTumorClassifier(pretrained=False)
        model.eval()
        
        x = torch.randn(2, 1, 256, 256)
        
        with torch.no_grad():
            features = model.extract_features(x)
        
        # Features should be 4D tensor
        assert len(features.shape) == 4
        assert features.shape[0] == 2  # Batch size
    
    def test_parameter_count(self):
        """Test parameter counting methods."""
        model = BrainTumorClassifier(pretrained=False)
        
        total_params = model.get_num_total_params()
        trainable_params = model.get_num_trainable_params()
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    def test_freeze_unfreeze(self):
        """Test freezing and unfreezing backbone."""
        model = BrainTumorClassifier(pretrained=False)
        
        # Initially all params should be trainable
        initial_trainable = model.get_num_trainable_params()
        
        # Freeze backbone
        model.freeze_backbone()
        frozen_trainable = model.get_num_trainable_params()
        
        assert frozen_trainable < initial_trainable
        
        # Unfreeze backbone
        model.unfreeze_backbone()
        unfrozen_trainable = model.get_num_trainable_params()
        
        assert unfrozen_trainable == initial_trainable
    
    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        model = BrainTumorClassifier(pretrained=False)
        model.eval()
        
        sizes = [224, 256, 512]
        
        for size in sizes:
            x = torch.randn(1, 1, size, size)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, 2)
    
    def test_dropout_rate(self):
        """Test model creation with different dropout rates."""
        for dropout in [0.0, 0.3, 0.5]:
            model = BrainTumorClassifier(pretrained=False, dropout=dropout)
            assert model.dropout_rate == dropout
    
    def test_cam_target_layer(self):
        """Test getting CAM target layer."""
        model = BrainTumorClassifier(pretrained=False)
        target_layer = model.get_cam_target_layer()
        
        assert target_layer is not None
        assert isinstance(target_layer, nn.Module)


class TestConvNeXtClassifier:
    """Tests for ConvNeXtClassifier."""
    
    def test_model_creation(self):
        """Test that ConvNeXt model can be created."""
        model = ConvNeXtClassifier(pretrained=False)
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        model = ConvNeXtClassifier(pretrained=False)
        model.eval()
        
        x = torch.randn(2, 1, 256, 256)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 2)
    
    def test_cam_target_layer(self):
        """Test getting CAM target layer."""
        model = ConvNeXtClassifier(pretrained=False)
        target_layer = model.get_cam_target_layer()
        
        assert target_layer is not None


class TestCreateClassifier:
    """Tests for create_classifier factory function."""
    
    def test_create_efficientnet(self):
        """Test creating EfficientNet model."""
        model = create_classifier('efficientnet', pretrained=False)
        assert isinstance(model, BrainTumorClassifier)
    
    def test_create_convnext(self):
        """Test creating ConvNeXt model."""
        model = create_classifier('convnext', pretrained=False)
        assert isinstance(model, ConvNeXtClassifier)
    
    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError):
            create_classifier('invalid_model', pretrained=False)
    
    def test_custom_parameters(self):
        """Test creating model with custom parameters."""
        model = create_classifier(
            'efficientnet',
            pretrained=False,
            num_classes=3,
            dropout=0.5
        )
        
        assert model.num_classes == 3
        assert model.dropout_rate == 0.5


class TestModelGradients:
    """Tests for gradient flow."""
    
    def test_backward_pass(self):
        """Test that gradients flow correctly."""
        model = BrainTumorClassifier(pretrained=False)
        model.train()
        
        x = torch.randn(2, 1, 256, 256)
        target = torch.tensor([0, 1])
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_frozen_backbone_no_gradients(self):
        """Test that frozen backbone doesn't get gradients."""
        model = BrainTumorClassifier(pretrained=False)
        model.freeze_backbone()
        model.train()
        
        x = torch.randn(2, 1, 256, 256)
        target = torch.tensor([0, 1])
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Check that backbone parameters don't have gradients
        for param in model.features.parameters():
            assert param.grad is None or torch.all(param.grad == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
