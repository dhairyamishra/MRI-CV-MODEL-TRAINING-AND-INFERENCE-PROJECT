# Testing Guide for SliceWise Phase 2 🧪

This guide explains how to test all Phase 2 implementations.

---

## 🎯 Quick Start

### Run All Tests

```bash
# Run comprehensive test suite
python scripts/run_tests.py
```

This will:
- Run all unit tests
- Generate a detailed report
- Show pass/fail status for each component
- Provide recommendations

---

## 📋 Test Categories

### 1. **Classifier Models** (`test_classifier.py`)

Tests the neural network architectures:
- ✅ Model creation (EfficientNet, ConvNeXt)
- ✅ Forward pass with different input sizes
- ✅ Single-channel input adaptation
- ✅ Feature extraction for Grad-CAM
- ✅ Parameter counting
- ✅ Backbone freezing/unfreezing
- ✅ Gradient flow
- ✅ Factory function

**Run individually:**
```bash
pytest tests/test_classifier.py -v
```

### 2. **Inference Predictor** (`test_predictor.py`)

Tests the prediction interface:
- ✅ Predictor creation
- ✅ Device selection (CPU/GPU)
- ✅ Image preprocessing (grayscale, RGB, different sizes)
- ✅ Single image prediction
- ✅ Batch prediction
- ✅ Prediction from file path
- ✅ Result format validation
- ✅ Probability computation
- ✅ Prediction consistency

**Run individually:**
```bash
pytest tests/test_predictor.py -v
```

### 3. **Grad-CAM** (`test_gradcam.py`)

Tests explainability features:
- ✅ Grad-CAM creation
- ✅ CAM generation
- ✅ Target class specification
- ✅ Overlay generation
- ✅ Hook registration
- ✅ Activation/gradient saving
- ✅ Different input sizes
- ✅ Alpha blending

**Run individually:**
```bash
pytest tests/test_gradcam.py -v
```

### 4. **Data Pipeline** (`test_data_pipeline.py`)

Tests data loading and augmentation:
- ✅ Transform classes (rotation, intensity, noise)
- ✅ Transform probability
- ✅ Transform composition
- ✅ Preset functions (train, val, strong, light)
- ✅ Edge cases (zeros, ones, different sizes)
- ✅ DataLoader creation (if data available)
- ✅ Batch shapes

**Run individually:**
```bash
pytest tests/test_data_pipeline.py -v
```

### 5. **Smoke Tests** (`test_smoke.py`)

Basic functionality tests:
- ✅ PyTorch import
- ✅ CUDA availability
- ✅ Basic tensor operations
- ✅ Model forward pass

**Run individually:**
```bash
pytest tests/test_smoke.py -v
```

---

## 🔧 Running Tests

### Option 1: Comprehensive Test Suite (Recommended)

```bash
python scripts/run_tests.py
```

**Output:**
```
======================================================================
  SliceWise Phase 2 - Comprehensive Test Suite
======================================================================

🔍 Discovering tests...
📝 Found 5 test files:
   - test_smoke.py
   - test_classifier.py
   - test_predictor.py
   - test_gradcam.py
   - test_data_pipeline.py

======================================================================
  Running Tests
======================================================================

----------------------------------------------------------------------
  Running: test_classifier.py
----------------------------------------------------------------------
test_classifier.py::TestBrainTumorClassifier::test_model_creation PASSED
test_classifier.py::TestBrainTumorClassifier::test_forward_pass PASSED
...

📊 Results:
   ✅ PASSED  test_smoke.py
   ✅ PASSED  test_classifier.py
   ✅ PASSED  test_predictor.py
   ✅ PASSED  test_gradcam.py
   ✅ PASSED  test_data_pipeline.py

📈 Statistics:
   Total Tests: 5
   Passed: 5
   Failed: 0
   Success Rate: 100.0%
   Total Time: 45.23s

✅ ALL TESTS PASSED
```

### Option 2: Individual Test Files

```bash
# Test specific component
pytest tests/test_classifier.py -v

# Test with coverage
pytest tests/test_classifier.py --cov=src.models --cov-report=html

# Test with specific markers
pytest tests/ -v -m "not slow"
```

### Option 3: Quick Smoke Test

```bash
# Just run smoke tests
pytest tests/test_smoke.py -v
```

---

## 📊 Test Coverage

### Current Coverage

| Component | Tests | Coverage |
|-----------|-------|----------|
| **Classifier Models** | 15+ | ~90% |
| **Inference Predictor** | 12+ | ~85% |
| **Grad-CAM** | 10+ | ~80% |
| **Data Pipeline** | 15+ | ~75% |
| **Total** | **50+** | **~82%** |

### Generate Coverage Report

```bash
# Install coverage
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# View HTML report
# Open htmlcov/index.html in browser
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Ensure project is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%  # Windows

# Or install in editable mode
pip install -e .
```

### Issue: "CUDA out of memory" during tests

**Solution:**
Tests run on CPU by default. If you see CUDA errors:
```python
# Tests use device='cpu' by default
# Check test fixtures in test files
```

### Issue: Data pipeline tests fail

**Solution:**
```bash
# Preprocess data first
python scripts/download_kaggle_data.py
python src/data/preprocess_kaggle.py
python src/data/split_kaggle.py

# Or skip data tests
pytest tests/ -v -k "not data"
```

### Issue: Tests are slow

**Solution:**
```bash
# Run in parallel
pip install pytest-xdist
pytest tests/ -n auto

# Run only fast tests
pytest tests/ -v -m "not slow"
```

---

## ✅ Test Checklist

Before considering Phase 2 complete, ensure:

- [ ] All unit tests pass
- [ ] Classifier models work correctly
- [ ] Predictor handles different inputs
- [ ] Grad-CAM generates valid heatmaps
- [ ] Data pipeline loads and transforms data
- [ ] No memory leaks in forward/backward passes
- [ ] Edge cases are handled gracefully
- [ ] Code coverage > 80%

---

## 🎯 Testing Best Practices

### 1. **Test Isolation**
Each test should be independent and not rely on others.

### 2. **Use Fixtures**
```python
@pytest.fixture
def model():
    return create_classifier('efficientnet', pretrained=False)
```

### 3. **Test Edge Cases**
- Empty inputs
- Very large/small values
- Different data types
- Boundary conditions

### 4. **Mock External Dependencies**
```python
@pytest.fixture
def dummy_checkpoint(tmp_path):
    # Create temporary checkpoint
    ...
```

### 5. **Clear Test Names**
```python
def test_model_accepts_single_channel_input():
    # Clear what is being tested
    ...
```

---

## 📝 Adding New Tests

### Template for New Test File

```python
"""
Tests for [component name].
"""

import sys
from pathlib import Path
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.your_module import YourClass


class TestYourClass:
    """Tests for YourClass."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        obj = YourClass()
        result = obj.method()
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case."""
        obj = YourClass()
        # Test edge case
        ...


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 🚀 Continuous Integration

### GitHub Actions

Tests automatically run on:
- Every push
- Every pull request
- Python 3.10 and 3.11
- CPU-only environment

See `.github/workflows/ci.yml` for configuration.

---

## 📚 Additional Resources

### pytest Documentation
- [pytest.org](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest markers](https://docs.pytest.org/en/stable/mark.html)

### Testing Best Practices
- [Python Testing Best Practices](https://realpython.com/pytest-python-testing/)
- [Test-Driven Development](https://testdriven.io/)

---

## 🎉 Success Criteria

Your Phase 2 implementation is ready when:

✅ All tests pass  
✅ Coverage > 80%  
✅ No critical bugs  
✅ Edge cases handled  
✅ Documentation complete  

---

**Ready to test?** Run:
```bash
python scripts/run_tests.py
```

---

*Testing is not about finding bugs, it's about preventing them!* 🐛➡️✅
