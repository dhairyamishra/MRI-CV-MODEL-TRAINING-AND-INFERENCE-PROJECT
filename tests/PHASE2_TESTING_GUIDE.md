# Phase 2 Testing Guide - Data & Training Pipelines

**Target:** Increase coverage from 17% to 25%  
**Focus Areas:** Data pipeline (11%) and Training pipeline (8%)  
**Estimated Tests:** ~60 new tests  
**Timeline:** 1-2 days

---

## 🎯 Priority Targets

### 1. Data Pipeline Tests (High Priority)

#### Module: `src/data/brats2d_dataset.py` (Current: 17%)
**Target Coverage:** 60%+  
**Estimated Tests:** 12-15 tests

**Test Ideas:**
```python
class TestBraTS2DDataset:
    """Test BraTS 2D dataset loading and iteration."""
    
    def test_dataset_initialization(self):
        """Test dataset initializes with correct parameters."""
        # Test with valid data directory
        # Verify __len__ returns correct count
        # Check metadata is loaded
    
    def test_dataset_getitem(self):
        """Test __getitem__ returns correct format."""
        # Verify returns (image, label) tuple
        # Check image shape and dtype
        # Validate label is in correct range
    
    def test_dataset_with_transforms(self):
        """Test dataset applies transforms correctly."""
        # Create dataset with transforms
        # Verify transforms are applied
        # Check output shape after transforms
    
    def test_dataset_empty_directory(self):
        """Test dataset handles empty directory."""
        # Should raise error or return empty dataset
    
    def test_dataset_statistics(self):
        """Test dataset statistics computation."""
        # Verify mean, std calculation
        # Check class distribution
    
    def test_dataset_caching(self):
        """Test dataset caching mechanism."""
        # Load same item twice
        # Verify caching improves speed
```

#### Module: `src/data/preprocess_brats_2d.py` (Current: 16%)
**Target Coverage:** 50%+  
**Estimated Tests:** 10-12 tests

**Test Ideas:**
```python
class TestBraTSPreprocessing:
    """Test BraTS preprocessing pipeline."""
    
    def test_slice_extraction(self):
        """Test 3D to 2D slice extraction."""
        # Create mock 3D volume
        # Extract slices
        # Verify slice count and dimensions
    
    def test_normalization_methods(self):
        """Test different normalization methods."""
        # Test z-score normalization
        # Test min-max normalization
        # Test percentile normalization
    
    def test_empty_slice_filtering(self):
        """Test filtering of empty slices."""
        # Create volume with empty slices
        # Apply filtering
        # Verify empty slices removed
    
    def test_modality_handling(self):
        """Test handling of different modalities."""
        # Test T1, T2, FLAIR, T1ce
        # Verify correct modality selection
```

#### Module: `src/data/split_brats.py` (Current: 11%)
**Target Coverage:** 60%+  
**Estimated Tests:** 8-10 tests

**Test Ideas:**
```python
class TestPatientLevelSplitting:
    """Test patient-level data splitting."""
    
    def test_split_ratios(self):
        """Test split produces correct ratios."""
        # Split with 70/15/15
        # Verify actual ratios match
    
    def test_no_patient_leakage(self):
        """Test no patient appears in multiple splits."""
        # Verify train/val/test have no overlap
        # Check patient IDs are unique per split
    
    def test_reproducibility(self):
        """Test splitting is reproducible with seed."""
        # Split twice with same seed
        # Verify identical results
    
    def test_minimum_samples_per_split(self):
        """Test each split has minimum samples."""
        # Verify no empty splits
        # Check minimum sample count
```

---

### 2. Training Pipeline Tests (High Priority)

#### Module: `src/training/train_cls.py` (Current: 15%)
**Target Coverage:** 40%+  
**Estimated Tests:** 10-12 tests

**Test Ideas:**
```python
class TestClassificationTraining:
    """Test classification training pipeline."""
    
    def test_training_step(self):
        """Test single training step."""
        # Create mock model and data
        # Run one training step
        # Verify loss is computed
        # Check gradients exist
    
    def test_validation_step(self):
        """Test validation step."""
        # Run validation
        # Verify no gradients computed
        # Check metrics are logged
    
    def test_checkpoint_saving(self):
        """Test checkpoint saving logic."""
        # Train for few steps
        # Verify checkpoint saved
        # Check checkpoint contains required keys
    
    def test_early_stopping(self):
        """Test early stopping mechanism."""
        # Simulate no improvement
        # Verify training stops
    
    def test_learning_rate_scheduling(self):
        """Test LR scheduler updates."""
        # Train with scheduler
        # Verify LR changes over epochs
```

#### Module: `src/training/losses.py` (Current: 35%)
**Target Coverage:** 70%+  
**Estimated Tests:** 12-15 tests

**Test Ideas:**
```python
class TestSegmentationLosses:
    """Test segmentation loss functions."""
    
    def test_dice_loss_perfect_prediction(self):
        """Test Dice loss with perfect prediction."""
        # pred = target
        # Loss should be ~0
    
    def test_dice_loss_worst_prediction(self):
        """Test Dice loss with worst prediction."""
        # pred = 1 - target
        # Loss should be ~1
    
    def test_dice_loss_gradient_flow(self):
        """Test Dice loss allows gradient flow."""
        # Compute loss
        # Backward pass
        # Verify gradients exist
    
    def test_focal_loss_class_imbalance(self):
        """Test Focal loss handles class imbalance."""
        # Create imbalanced data
        # Compute focal loss
        # Verify focuses on hard examples
    
    def test_combined_loss(self):
        """Test combined loss (Dice + BCE)."""
        # Compute both losses
        # Verify weighted combination
```

---

## 🛠️ Test Template

Use this template for creating new test files:

```python
"""
PHASE 2: [Module Name] Integration Tests

Tests the actual [module] implementation to increase coverage
from [X]% to [Y]%+.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.[module_path] import [functions_to_test]


class Test[FeatureName]:
    """Test [feature] functionality."""
    
    def test_[feature]_basic(self):
        """Test basic [feature] operation."""
        # Arrange
        input_data = create_test_data()
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result is not None
        assert result.shape == expected_shape
        assert result.dtype == expected_dtype
    
    def test_[feature]_edge_case(self):
        """Test [feature] with edge case."""
        # Test with boundary values
        # Verify correct handling
    
    def test_[feature]_error_handling(self):
        """Test [feature] error handling."""
        # Test with invalid input
        # Verify appropriate error raised


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 📋 Testing Checklist

For each module, ensure tests cover:

### Functionality
- [ ] Basic operation with valid inputs
- [ ] Different parameter combinations
- [ ] Return value format and types
- [ ] Side effects (file creation, logging, etc.)

### Edge Cases
- [ ] Empty inputs
- [ ] Single element inputs
- [ ] Very large inputs
- [ ] Boundary values
- [ ] None/null values

### Error Handling
- [ ] Invalid input types
- [ ] Out-of-range values
- [ ] Missing required parameters
- [ ] Corrupted data

### Integration
- [ ] Works with other components
- [ ] Correct data flow
- [ ] Proper cleanup

---

## 🚀 Quick Commands

### Run Tests for Specific Module
```bash
# Data pipeline tests
python -m pytest tests/unit/test_data_pipeline_integration.py -v

# Training pipeline tests
python -m pytest tests/unit/test_training_pipeline_integration.py -v

# With coverage for specific module
python -m pytest tests/unit/test_data_pipeline_integration.py --cov=src/data --cov-report=term-missing
```

### Generate Coverage Report
```bash
# HTML report
python -m pytest tests/ --cov=src --cov-report=html

# Terminal report with missing lines
python -m pytest tests/ --cov=src --cov-report=term-missing:skip-covered
```

### Run Tests in Parallel (faster)
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run with 4 workers
python -m pytest tests/ -n 4
```

---

## 📊 Progress Tracking

### Phase 2 Goals

| Module | Current | Target | Tests Needed | Status |
|--------|---------|--------|--------------|--------|
| `brats2d_dataset.py` | 17% | 60% | 12-15 | ⏳ Pending |
| `preprocess_brats_2d.py` | 16% | 50% | 10-12 | ⏳ Pending |
| `split_brats.py` | 11% | 60% | 8-10 | ⏳ Pending |
| `train_cls.py` | 15% | 40% | 10-12 | ⏳ Pending |
| `losses.py` | 35% | 70% | 12-15 | ⏳ Pending |
| `multi_source_dataset.py` | 12% | 50% | 8-10 | ⏳ Pending |

**Total Tests to Add:** ~60 tests  
**Estimated Coverage Gain:** +8-10%

---

## 💡 Tips for Efficient Testing

### 1. Start with Happy Path
- Write tests for normal, expected usage first
- Verify basic functionality works
- Then add edge cases and error handling

### 2. Use Fixtures
```python
@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    # Setup
    dataset = create_test_dataset()
    yield dataset
    # Teardown
    cleanup_test_data()

def test_with_fixture(sample_dataset):
    """Test using fixture."""
    assert len(sample_dataset) > 0
```

### 3. Parametrize Tests
```python
@pytest.mark.parametrize("input_val,expected", [
    (0, 0),
    (1, 1),
    (5, 25),
])
def test_square(input_val, expected):
    """Test square function with multiple inputs."""
    assert square(input_val) == expected
```

### 4. Mock External Dependencies
```python
@patch('src.data.download_data')
def test_preprocessing(mock_download):
    """Test preprocessing without actual download."""
    mock_download.return_value = mock_data
    result = preprocess_pipeline()
    assert result is not None
```

---

## 🎯 Success Criteria

Phase 2 is complete when:
- ✅ 60+ new tests added
- ✅ All new tests passing
- ✅ Overall coverage ≥ 25%
- ✅ Data pipeline coverage ≥ 40%
- ✅ Training pipeline coverage ≥ 30%
- ✅ No flaky tests
- ✅ Test execution time < 60 seconds

---

**Ready to start Phase 2!** 🚀

Choose a module from the priority list and begin with the test template above.
