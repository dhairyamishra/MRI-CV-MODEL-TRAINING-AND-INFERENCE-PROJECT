"""
Unit tests for inference predictor.

Tests the ClassifierPredictor implementation.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
from PIL import Image
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predict import ClassifierPredictor
from src.models.classifier import create_classifier


@pytest.fixture(scope="module")
def dummy_checkpoint(tmp_path_factory):
    """Create a dummy checkpoint file that persists for all tests in module."""
    tmp_path = tmp_path_factory.mktemp("checkpoints")
    model = create_classifier('efficientnet', pretrained=False)
    
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'best_metric': 0.95,
        'config': {}
    }
    
    checkpoint_path = tmp_path / "test_model.pth"
    torch.save(checkpoint, checkpoint_path)
    
    return str(checkpoint_path)


@pytest.fixture
def dummy_image():
    """Create a dummy grayscale image."""
    return np.random.rand(256, 256).astype(np.float32)


@pytest.fixture
def dummy_rgb_image():
    """Create a dummy RGB image."""
    return np.random.rand(256, 256, 3).astype(np.float32)


class TestClassifierPredictor:
    """Tests for ClassifierPredictor."""
    
    def test_predictor_creation(self, dummy_checkpoint):
        """Test that predictor can be created."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            assert predictor is not None
            assert predictor.device == torch.device('cpu')
        except Exception as e:
            assert False, f"Failed to create predictor: {e}"
    
    def test_device_auto_selection(self, dummy_checkpoint):
        """Test automatic device selection."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device=None)
            assert predictor.device in [torch.device('cuda'), torch.device('cpu')]
        except Exception as e:
            assert False, f"Failed to select device: {e}"
    
    def test_preprocess_grayscale_image(self, dummy_checkpoint, dummy_image):
        """Test preprocessing of grayscale image."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            tensor = predictor.preprocess_image(dummy_image)
            # Check shape (1, 1, 256, 256)
            assert tensor.shape == (1, 1, 256, 256)
            assert tensor.dtype == torch.float32
            assert tensor.min() >= 0.0
            assert tensor.max() <= 1.0
        except Exception as e:
            assert False, f"Failed to preprocess image: {e}"
    
    def test_preprocess_rgb_image(self, dummy_checkpoint, dummy_rgb_image):
        """Test preprocessing of RGB image."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            tensor = predictor.preprocess_image(dummy_rgb_image)
            # Should convert to grayscale
            assert tensor.shape == (1, 1, 256, 256)
        except Exception as e:
            assert False, f"Failed to preprocess image: {e}"
    
    def test_preprocess_different_sizes(self, dummy_checkpoint):
        """Test preprocessing images of different sizes."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            sizes = [(128, 128), (512, 512), (300, 400)]
            for size in sizes:
                image = np.random.rand(*size).astype(np.float32)
                tensor = predictor.preprocess_image(image, target_size=256)
                assert tensor.shape == (1, 1, 256, 256)
        except Exception as e:
            assert False, f"Failed to preprocess image: {e}"
    
    def test_predict_single_image(self, dummy_checkpoint, dummy_image):
        """Test prediction on single image."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            result = predictor.predict(dummy_image)
            # Check result structure
            assert 'predicted_class' in result
            assert 'predicted_label' in result
            assert 'confidence' in result
            assert 'probabilities' in result
            # Check types
            assert isinstance(result['predicted_class'], int)
            assert isinstance(result['predicted_label'], str)
            assert isinstance(result['confidence'], float)
            assert isinstance(result['probabilities'], dict)
            # Check values
            assert result['predicted_class'] in [0, 1]
            assert result['predicted_label'] in ['No Tumor', 'Tumor']
            assert 0.0 <= result['confidence'] <= 1.0
            # Check probabilities sum to 1
            prob_sum = sum(result['probabilities'].values())
            assert abs(prob_sum - 1.0) < 1e-5
        except Exception as e:
            assert False, f"Failed to predict image: {e}"
    
    def test_predict_without_probabilities(self, dummy_checkpoint, dummy_image):
        """Test prediction without returning probabilities."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            result = predictor.predict(dummy_image, return_probabilities=False)
            assert 'probabilities' not in result
        except Exception as e:
            assert False, f"Failed to predict image: {e}"
    
    def test_predict_batch(self, dummy_checkpoint):
        """Test batch prediction."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            # Create batch of images
            images = [np.random.rand(256, 256).astype(np.float32) for _ in range(5)]
            results = predictor.predict_batch(images)
            assert len(results) == 5
            for result in results:
                assert 'predicted_class' in result
                assert 'predicted_label' in result
                assert 'confidence' in result
        except Exception as e:
            assert False, f"Failed to predict batch: {e}"
    
    def test_predict_from_path(self, dummy_checkpoint, tmp_path):
        """Test prediction from image file path."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            # Create temporary image file
            image = Image.fromarray((np.random.rand(256, 256) * 255).astype(np.uint8))
            image_path = tmp_path / "test_image.png"
            image.save(image_path)
            result = predictor.predict_from_path(str(image_path))
            assert 'predicted_class' in result
            assert 'predicted_label' in result
        except Exception as e:
            assert False, f"Failed to predict from path: {e}"
    
    def test_class_names(self, dummy_checkpoint):
        """Test that class names are correct."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            assert predictor.class_names == ['No Tumor', 'Tumor']
        except Exception as e:
            assert False, f"Failed to get class names: {e}"
    
    def test_prediction_consistency(self, dummy_checkpoint, dummy_image):
        """Test that predictions are consistent for same input."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            result1 = predictor.predict(dummy_image)
            result2 = predictor.predict(dummy_image)
            assert result1['predicted_class'] == result2['predicted_class']
            assert abs(result1['confidence'] - result2['confidence']) < 1e-5
        except Exception as e:
            assert False, f"Failed to predict image: {e}"


class TestPredictorEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_uint8_image(self, dummy_checkpoint):
        """Test prediction with uint8 image (0-255 range)."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            image = (np.random.rand(256, 256) * 255).astype(np.uint8)
            result = predictor.predict(image)
            assert 'predicted_class' in result
        except Exception as e:
            assert False, f"Failed to predict image: {e}"
    
    def test_very_small_image(self, dummy_checkpoint):
        """Test prediction with very small image."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            image = np.random.rand(32, 32).astype(np.float32)
            result = predictor.predict(image)
            assert 'predicted_class' in result
        except Exception as e:
            assert False, f"Failed to predict image: {e}"
    
    def test_very_large_image(self, dummy_checkpoint):
        """Test prediction with very large image."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            image = np.random.rand(1024, 1024).astype(np.float32)
            result = predictor.predict(image)
            assert 'predicted_class' in result
        except Exception as e:
            assert False, f"Failed to predict image: {e}"
    
    def test_empty_batch(self, dummy_checkpoint):
        """Test prediction with empty batch."""
        try:
            predictor = ClassifierPredictor(dummy_checkpoint, device='cpu')
            results = predictor.predict_batch([])
            assert len(results) == 0
        except Exception as e:
            assert False, f"Failed to predict batch: {e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
