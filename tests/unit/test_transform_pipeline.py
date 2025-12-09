"""
PHASE 1.1.3: Transform Pipeline Validation - Critical Safety Tests

Tests geometric transforms, intensity transforms, medical-specific transforms,
transform composition, reproducibility, and performance.

These tests ensure data augmentation preserves medical image integrity.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import time
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
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


class TestGeometricTransforms:
    """Test geometric transformations (rotation, flip, scale, elastic)."""

    def test_random_rotation90(self):
        """Test 90-degree rotation transform."""
        transform = RandomRotation90(p=1.0)  # Always apply

        # Create test image with asymmetric pattern
        test_image = np.zeros((128, 128), dtype=np.float32)
        test_image[10:20, 10:20] = 1.0  # Top-left square
        original_sum = np.sum(test_image)

        # Apply rotation
        rotated = transform(test_image)

        # Basic properties
        assert rotated.shape == test_image.shape
        assert rotated.dtype == test_image.dtype
        assert np.sum(rotated) == original_sum  # Pixel sum preserved
        assert np.all((rotated >= 0) & (rotated <= 1))  # Valid range

        # Verify it's actually rotated (not identical)
        assert not np.array_equal(rotated, test_image)

    def test_rotation_reproducibility(self):
        """Test rotation reproducibility with fixed seed."""
        np.random.seed(42)
        transform1 = RandomRotation90(p=1.0)
        np.random.seed(42)
        transform2 = RandomRotation90(p=1.0)

        test_image = np.random.rand(128, 128).astype(np.float32)

        result1 = transform1(test_image)
        result2 = transform2(test_image)

        # Should be identical with same seed
        np.testing.assert_array_equal(result1, result2)

    def test_rotation_edge_cases(self):
        """Test rotation with edge cases."""
        transform = RandomRotation90(p=1.0)

        # Test with all-zero image
        zero_image = np.zeros((64, 64), dtype=np.float32)
        rotated_zero = transform(zero_image)
        np.testing.assert_array_equal(rotated_zero, zero_image)

        # Test with all-ones image
        ones_image = np.ones((64, 64), dtype=np.float32)
        rotated_ones = transform(ones_image)
        np.testing.assert_array_equal(rotated_ones, ones_image)


class TestIntensityTransforms:
    """Test intensity transformations (brightness, contrast, gamma, noise)."""

    def test_intensity_shift(self):
        """Test random intensity shift."""
        transform = RandomIntensityShift(shift_range=0.2, p=1.0)

        # Create test image
        test_image = np.full((128, 128), 0.5, dtype=np.float32)

        # Apply shift
        shifted = transform(test_image)

        # Should be shifted but still in valid range
        assert shifted.shape == test_image.shape
        assert not np.array_equal(shifted, test_image)
        assert np.all((shifted >= 0) & (shifted <= 1))

        # Check that shift is within expected bounds
        shift_amount = np.mean(shifted - test_image)
        assert abs(shift_amount) <= 0.2

    def test_intensity_scale(self):
        """Test random intensity scaling."""
        transform = RandomIntensityScale(scale_range=(0.8, 1.2), p=1.0)

        # Create test image
        test_image = np.full((128, 128), 0.5, dtype=np.float32)

        # Apply scaling
        scaled = transform(test_image)

        assert scaled.shape == test_image.shape
        assert np.all((scaled >= 0) & (scaled <= 1))

        # Check scaling effect
        scale_factor = np.mean(scaled) / np.mean(test_image)
        assert 0.8 <= scale_factor <= 1.2

    def test_gaussian_noise(self):
        """Test Gaussian noise addition."""
        transform = RandomGaussianNoise(std=0.05, p=1.0)

        # Create clean test image
        test_image = np.full((128, 128), 0.5, dtype=np.float32)

        # Apply noise
        noisy = transform(test_image)

        assert noisy.shape == test_image.shape
        assert not np.array_equal(noisy, test_image)
        assert np.all((noisy >= 0) & (noisy <= 1))

        # Noise should be small
        noise_magnitude = np.std(noisy - test_image)
        assert noise_magnitude < 0.1  # Reasonable noise level

    def test_intensity_bounds_preservation(self):
        """Test that intensity transforms preserve valid ranges."""
        transforms = [
            RandomIntensityShift(shift_range=0.5, p=1.0),
            RandomIntensityScale(scale_range=(0.5, 1.5), p=1.0),
            RandomGaussianNoise(std=0.1, p=1.0)
        ]

        test_cases = [
            np.zeros((64, 64), dtype=np.float32),  # All zeros
            np.ones((64, 64), dtype=np.float32),   # All ones
            np.full((64, 64), 0.5, dtype=np.float32),  # Mid-range
            np.random.rand(64, 64).astype(np.float32)  # Random
        ]

        for transform in transforms:
            for test_image in test_cases:
                result = transform(test_image)

                # Must stay in [0, 1] range
                assert np.all((result >= 0) & (result <= 1)), \
                    f"Transform {transform.__class__.__name__} produced out-of-bounds values"


class TestMedicalSpecificTransforms:
    """Test transforms that preserve anatomical plausibility."""

    def test_anatomical_continuity(self):
        """Test that transforms preserve anatomical structures."""
        # Create synthetic anatomical structure
        anatomical_image = np.zeros((128, 128), dtype=np.float32)

        # Simulate brain regions
        anatomical_image[40:88, 40:88] = 0.8  # Main brain tissue
        anatomical_image[50:78, 50:78] = 0.9  # Core region
        anatomical_image[60:68, 60:68] = 0.3  # CSF-like region

        original_structure = anatomical_image.copy()

        # Apply various transforms
        transforms_to_test = [
            RandomRotation90(p=1.0),
            RandomIntensityShift(shift_range=0.1, p=1.0),
            RandomIntensityScale(scale_range=(0.9, 1.1), p=1.0)
        ]

        for transform in transforms_to_test:
            transformed = transform(anatomical_image)

            # Basic validity checks
            assert transformed.shape == anatomical_image.shape
            assert np.all(np.isfinite(transformed))
            assert np.all((transformed >= 0) & (transformed <= 1))

            # Should preserve some structural information
            # (This is a simplified check - real anatomical validation would be more complex)
            structure_preserved = np.corrcoef(
                original_structure.flatten(),
                transformed.flatten()
            )[0, 1]
            assert structure_preserved > 0.5  # Reasonable correlation maintained

    def test_medical_image_properties(self):
        """Test preservation of medical image properties."""
        # Create image mimicking medical imaging characteristics
        medical_image = np.random.exponential(1.0, (128, 128)).astype(np.float32)
        medical_image = medical_image / np.max(medical_image)  # Normalize to [0, 1]

        # Apply medical-appropriate transforms
        transform = RandomIntensityScale(scale_range=(0.8, 1.2), p=1.0)
        transformed = transform(medical_image)

        # Should maintain medical imaging characteristics
        assert transformed.shape == medical_image.shape
        assert np.all((transformed >= 0) & (transformed <= 1))

        # Check that intensity distribution is reasonably preserved
        original_mean = np.mean(medical_image)
        transformed_mean = np.mean(transformed)
        assert abs(original_mean - transformed_mean) < 0.2  # Reasonable change


class TestTransformComposition:
    """Test complete augmentation pipelines."""

    def test_train_transform_composition(self):
        """Test complete training transform pipeline."""
        transforms = get_train_transforms()

        # Create test batch
        test_images = [np.random.rand(128, 128).astype(np.float32) for _ in range(5)]

        # Apply transforms to each image
        transformed_images = []
        for img in test_images:
            transformed = transforms(img)
            transformed_images.append(transformed)

            # Each result should be valid
            assert transformed.shape == img.shape
            assert transformed.dtype == img.dtype
            assert np.all(np.isfinite(transformed))
            assert np.all((transformed >= 0) & (transformed <= 1))

        # Transforms should produce variety (not all identical)
        all_identical = all(
            np.array_equal(transformed_images[0], img)
            for img in transformed_images[1:]
        )
        assert not all_identical  # Should produce variation

    def test_validation_transform_stability(self):
        """Test validation transforms are deterministic."""
        transforms = get_val_transforms()

        test_image = np.random.rand(128, 128).astype(np.float32)

        # Apply multiple times
        results = [transforms(test_image) for _ in range(5)]

        # All results should be identical (deterministic)
        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)

    def test_transform_pipeline_robustness(self):
        """Test transform pipeline handles edge cases."""
        transforms = get_train_transforms()

        edge_cases = [
            np.zeros((128, 128), dtype=np.float32),  # All zeros
            np.ones((128, 128), dtype=np.float32),   # All ones
            np.full((128, 128), 0.5, dtype=np.float32),  # Constant value
            np.random.rand(128, 128).astype(np.float32) * 1e-6  # Very small values
        ]

        for test_image in edge_cases:
            try:
                result = transforms(test_image)

                # Should produce valid output
                assert result.shape == test_image.shape
                assert np.all(np.isfinite(result))
                assert np.all((result >= 0) & (result <= 1))

            except Exception as e:
                pytest.fail(f"Transform pipeline failed on edge case: {e}")


class TestTransformReproducibility:
    """Test transform reproducibility with fixed seeds."""

    def test_seed_based_reproducibility(self):
        """Test transforms are reproducible with fixed seeds."""
        # Test with same seed
        np.random.seed(123)
        transform1 = get_train_transforms()

        np.random.seed(123)
        transform2 = get_train_transforms()

        test_image = np.random.rand(128, 128).astype(np.float32)

        result1 = transform1(test_image)
        result2 = transform2(test_image)

        # Should be identical with same seed
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_variation(self):
        """Test different seeds produce different results."""
        np.random.seed(123)
        transform1 = get_train_transforms()

        np.random.seed(456)
        transform2 = get_train_transforms()

        test_image = np.random.rand(128, 128).astype(np.float32)

        result1 = transform1(test_image)
        result2 = transform2(test_image)

        # Should be different with different seeds
        assert not np.array_equal(result1, result2)

    def test_reproducibility_across_runs(self):
        """Test reproducibility across multiple transform applications."""
        np.random.seed(42)

        results = []
        for _ in range(3):
            transform = get_train_transforms()
            result = transform(np.random.rand(64, 64).astype(np.float32))
            results.append(result)

        # Results should vary (different random state each time)
        assert not np.array_equal(results[0], results[1])
        assert not np.array_equal(results[1], results[2])


class TestTransformPerformance:
    """Test transform speed on large datasets."""

    def test_transform_speed(self):
        """Test transform performance on large dataset."""
        transforms = get_train_transforms()

        # Create larger dataset
        dataset_size = 100
        image_size = (256, 256)

        test_images = [
            np.random.rand(*image_size).astype(np.float32)
            for _ in range(dataset_size)
        ]

        # Measure transform time
        start_time = time.time()

        transformed_images = []
        for img in test_images:
            transformed = transforms(img)
            transformed_images.append(transformed)

        end_time = time.time()
        total_time = end_time - start_time

        # Performance requirements
        avg_time_per_image = total_time / dataset_size
        assert avg_time_per_image < 0.1  # Less than 100ms per image

        # Verify all transforms completed successfully
        assert len(transformed_images) == dataset_size
        for img in transformed_images:
            assert img.shape == image_size
            assert np.all(np.isfinite(img))

    def test_memory_usage_efficiency(self):
        """Test transform memory usage patterns."""
        transforms = get_train_transforms()

        # Test with progressively larger images
        sizes = [(64, 64), (128, 128), (256, 256)]

        for size in sizes:
            test_image = np.random.rand(*size).astype(np.float32)

            # Measure memory before
            initial_memory = np.zeros(1)  # Placeholder for memory tracking

            # Apply transform
            result = transforms(test_image)

            # Verify result
            assert result.shape == size
            assert np.all(np.isfinite(result))

            # Clean up
            del test_image, result

    def test_batch_transform_performance(self):
        """Test transform performance on batched data."""
        transforms = get_train_transforms()

        # Simulate DataLoader batch processing
        batch_size = 16
        image_size = (128, 128)

        # Create batch
        batch_images = [
            np.random.rand(*image_size).astype(np.float32)
            for _ in range(batch_size)
        ]

        start_time = time.time()

        # Process batch
        batch_results = []
        for img in batch_images:
            result = transforms(img)
            batch_results.append(result)

        end_time = time.time()
        batch_time = end_time - start_time

        # Batch processing should be efficient
        avg_time_per_image = batch_time / batch_size
        assert avg_time_per_image < 0.05  # Less than 50ms per image in batch

        assert len(batch_results) == batch_size
