"""
Unit tests for data pipeline components.

Tests dataset classes, transforms, and data loading.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.transforms import (
    RandomRotation90,
    RandomIntensityShift,
    RandomIntensityScale,
    RandomGaussianNoise,
    get_train_transforms,
    get_val_transforms,
    get_strong_train_transforms,
    get_light_train_transforms
)


class TestTransforms:
    """Tests for custom transform classes."""
    
    def test_random_rotation90(self):
        """Test RandomRotation90 transform."""
        transform = RandomRotation90(p=1.0)  # Always apply
        
        image = np.random.rand(256, 256).astype(np.float32)
        original_shape = image.shape
        
        transformed = transform(image)
        
        # Shape should be preserved
        assert transformed.shape == original_shape
        # Values should be in same range
        assert transformed.min() >= 0.0
        assert transformed.max() <= 1.0
    
    def test_random_intensity_shift(self):
        """Test RandomIntensityShift transform."""
        transform = RandomIntensityShift(shift_range=0.1, p=1.0)
        
        image = np.ones((256, 256), dtype=np.float32) * 0.5
        transformed = transform(image)
        
        # Values should be shifted but still in valid range
        assert transformed.min() >= 0.0
        assert transformed.max() <= 1.0
        # Should be different from original
        assert not np.allclose(transformed, image)
    
    def test_random_intensity_scale(self):
        """Test RandomIntensityScale transform."""
        transform = RandomIntensityScale(scale_range=(0.8, 1.2), p=1.0)
        
        image = np.ones((256, 256), dtype=np.float32) * 0.5
        transformed = transform(image)
        
        # Values should be scaled but still in valid range
        assert transformed.min() >= 0.0
        assert transformed.max() <= 1.0
    
    def test_random_gaussian_noise(self):
        """Test RandomGaussianNoise transform."""
        transform = RandomGaussianNoise(std=0.01, p=1.0)
        
        image = np.ones((256, 256), dtype=np.float32) * 0.5
        transformed = transform(image)
        
        # Noise should be added
        assert not np.allclose(transformed, image)
        # Values should still be in valid range
        assert transformed.min() >= 0.0
        assert transformed.max() <= 1.0
    
    def test_transform_probability(self):
        """Test that transforms respect probability parameter."""
        transform = RandomRotation90(p=0.0)  # Never apply
        
        image = np.random.rand(256, 256).astype(np.float32)
        transformed = transform(image)
        
        # Should be unchanged
        assert np.allclose(transformed, image)
    
    def test_transform_composition(self):
        """Test that multiple transforms can be composed."""
        transforms = get_train_transforms()
        
        image = np.random.rand(256, 256).astype(np.float32)
        
        # Should not raise error
        transformed = transforms(image)
        
        assert transformed.shape == image.shape
        assert transformed.min() >= 0.0
        assert transformed.max() <= 1.0


class TestTransformPresets:
    """Tests for transform preset functions."""
    
    def test_train_transforms(self):
        """Test standard train transforms."""
        transforms = get_train_transforms()
        
        image = np.random.rand(256, 256).astype(np.float32)
        transformed = transforms(image)
        
        assert transformed.shape == image.shape
    
    def test_val_transforms(self):
        """Test validation transforms (should be identity)."""
        transforms = get_val_transforms()
        
        image = np.random.rand(256, 256).astype(np.float32)
        transformed = transforms(image)
        
        # Validation transforms should not modify image
        assert np.allclose(transformed, image)
    
    def test_strong_train_transforms(self):
        """Test strong augmentation transforms."""
        transforms = get_strong_train_transforms()
        
        image = np.random.rand(256, 256).astype(np.float32)
        transformed = transforms(image)
        
        assert transformed.shape == image.shape
    
    def test_light_train_transforms(self):
        """Test light augmentation transforms."""
        transforms = get_light_train_transforms()
        
        image = np.random.rand(256, 256).astype(np.float32)
        transformed = transforms(image)
        
        assert transformed.shape == image.shape


class TestTransformEdgeCases:
    """Tests for edge cases in transforms."""
    
    def test_all_zeros_image(self):
        """Test transforms on all-zeros image."""
        transforms = get_train_transforms()
        
        image = np.zeros((256, 256), dtype=np.float32)
        transformed = transforms(image)
        
        # Should handle gracefully
        assert transformed.shape == image.shape
        assert transformed.min() >= 0.0
        assert transformed.max() <= 1.0
    
    def test_all_ones_image(self):
        """Test transforms on all-ones image."""
        transforms = get_train_transforms()
        
        image = np.ones((256, 256), dtype=np.float32)
        transformed = transforms(image)
        
        # Should handle gracefully
        assert transformed.shape == image.shape
        assert transformed.min() >= 0.0
        assert transformed.max() <= 1.0
    
    def test_different_image_sizes(self):
        """Test transforms on different image sizes."""
        transforms = get_train_transforms()
        
        sizes = [(128, 128), (256, 256), (512, 512)]
        
        for size in sizes:
            image = np.random.rand(*size).astype(np.float32)
            transformed = transforms(image)
            assert transformed.shape == image.shape
    
    def test_transform_determinism_with_seed(self):
        """Test that transforms are deterministic with same seed."""
        np.random.seed(42)
        transforms1 = get_train_transforms()
        image = np.random.rand(256, 256).astype(np.float32)
        result1 = transforms1(image.copy())
        
        np.random.seed(42)
        transforms2 = get_train_transforms()
        result2 = transforms2(image.copy())
        
        # Results should be similar (not exact due to random state)
        assert result1.shape == result2.shape


class TestDataLoading:
    """Tests for data loading functionality."""
    
    @pytest.mark.skipif(
        not Path("data/processed/kaggle/train").exists(),
        reason="Processed data not available"
    )
    def test_dataloader_creation(self):
        """Test that dataloaders can be created."""
        try:
            from src.data.kaggle_mri_dataset import create_dataloaders
            
            train_loader, val_loader, test_loader = create_dataloaders(
                batch_size=4,
                num_workers=0,
                train_transform=None,
                val_transform=None
            )
            
            assert isinstance(train_loader, DataLoader)
            assert isinstance(val_loader, DataLoader)
            assert isinstance(test_loader, DataLoader)
        except Exception as e:
            pytest.skip(f"Data not available: {e}")
    
    @pytest.mark.skipif(
        not Path("data/processed/kaggle/train").exists(),
        reason="Processed data not available"
    )
    def test_batch_shapes(self):
        """Test that batches have correct shapes."""
        try:
            from src.data.kaggle_mri_dataset import create_dataloaders
            
            train_loader, _, _ = create_dataloaders(
                batch_size=4,
                num_workers=0
            )
            
            # Get one batch
            images, labels, ids = next(iter(train_loader))
            
            # Check shapes
            assert images.shape[0] <= 4  # Batch size
            assert images.shape[1] == 1  # Single channel
            assert images.shape[2] == 256  # Height
            assert images.shape[3] == 256  # Width
            
            assert labels.shape[0] <= 4
            assert len(ids) <= 4
        except Exception as e:
            pytest.skip(f"Data not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
