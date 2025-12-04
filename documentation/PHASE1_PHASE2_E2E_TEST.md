# Phase 1 & Phase 2 End-to-End Test Guide

## Overview

The **Phase 1 & Phase 2 End-to-End Test** (`test_phase1_phase2_full_e2e.py`) is a comprehensive test suite that validates the entire SliceWise MRI Brain Tumor Detection pipeline from raw data acquisition to production-ready inference.

This test ensures that all components work together seamlessly and that the system is production-ready.

---

## What Does This Test Cover?

### **Phase 1: Data Acquisition & Preprocessing** (6 Tests)

1. **Data Download** - Validates Kaggle dataset download
2. **Data Preprocessing** - Tests JPG → .npz conversion with normalization
3. **Data Splitting** - Verifies stratified train/val/test splits (70/15/15)
4. **Dataset Loading** - Tests PyTorch dataset class and data loading
5. **Transform Pipeline** - Validates augmentation transforms (train/val/strong/light)
6. **DataLoader Creation** - Tests batch loading and data pipeline

### **Phase 2: Classification MVP** (7 Tests)

7. **Model Creation** - Tests EfficientNet-B0 and ConvNeXt architectures
8. **Mini Training** - Runs 2-epoch training to verify training loop
9. **Model Evaluation** - Computes accuracy and AUC metrics
10. **Grad-CAM Generation** - Tests explainability visualization
11. **Inference Pipeline** - Validates single and batch prediction
12. **API Endpoints** - Tests FastAPI backend (if running)
13. **Full Integration** - End-to-end test from data to prediction with Grad-CAM

**Total: 13 comprehensive tests**

---

## Prerequisites

### 1. Environment Setup

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### 2. Kaggle API Credentials (Optional)

If you want to test data download (not required if data already exists):

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save `kaggle.json` to:
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
4. Set permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

### 3. Existing Data (Recommended)

If you already have the Kaggle Brain MRI dataset downloaded and preprocessed, the test will skip the download step and use existing data. This is the **recommended** approach for faster testing.

---

## How to Run

### **Option 1: Quick Test (Recommended)**

Use existing data (skips download):

```bash
python scripts/test_phase1_phase2_full_e2e.py
```

This assumes you have:
- `data/raw/kaggle_brain_mri/` with yes/no folders
- `data/processed/kaggle/` with preprocessed .npz files

### **Option 2: Full Test with Data Download**

Download data from Kaggle (requires API credentials):

```bash
python scripts/test_phase1_phase2_full_e2e.py --download-data
```

⚠️ **Warning**: This will download ~50MB of data and may take several minutes.

### **Option 3: Isolated Test with Temporary Directory**

Test in a temporary directory (cleanup after):

```bash
python scripts/test_phase1_phase2_full_e2e.py --use-temp-dir --download-data
```

This is useful for CI/CD or when you don't want to affect your existing data.

---

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--download-data` | Download data from Kaggle (requires API credentials) |
| `--use-temp-dir` | Use temporary directory for testing (cleanup after) |
| `-h, --help` | Show help message |

---

## Expected Output

### **Successful Run Example**

```
================================================================================
  COMPREHENSIVE PHASE 1 & PHASE 2 END-TO-END TEST SUITE
================================================================================
  Timestamp: 2025-12-03 23:36:05
  Device: cuda
  Data Directory: c:\...\data\processed\kaggle
================================================================================

================================================================================
  PHASE 1: Data Acquisition & Preprocessing
================================================================================

--------------------------------------------------------------------------------
  Phase 1 Test 1: Data Download
--------------------------------------------------------------------------------
✅ Data Download: Skipped (data exists): 154 tumor, 91 no tumor

--------------------------------------------------------------------------------
  Phase 1 Test 2: Data Preprocessing
--------------------------------------------------------------------------------
ℹ️  Found 245 existing .npz files
✅ Data Preprocessing: Using existing 245 preprocessed files

--------------------------------------------------------------------------------
  Phase 1 Test 3: Data Splitting
--------------------------------------------------------------------------------
✅ Data Splitting: Using existing splits: 171 train, 37 val, 37 test

--------------------------------------------------------------------------------
  Phase 1 Test 4: Dataset Loading
--------------------------------------------------------------------------------
✅ Dataset Loading: Loaded 171 samples, 108 tumor, 63 no tumor

--------------------------------------------------------------------------------
  Phase 1 Test 5: Transform Pipeline
--------------------------------------------------------------------------------
✅ Transform Pipeline: Tested 4 transform presets: train, val, strong, light

--------------------------------------------------------------------------------
  Phase 1 Test 6: DataLoader Creation
--------------------------------------------------------------------------------
✅ DataLoader Creation: Created loaders: 171 train, 37 val, 37 test

================================================================================
  PHASE 2: Classification MVP
================================================================================

--------------------------------------------------------------------------------
  Phase 2 Test 1: Model Creation
--------------------------------------------------------------------------------
✅ Model Creation: EfficientNet: 4,011,391 params, ConvNeXt: 27,821,314 params

--------------------------------------------------------------------------------
  Phase 2 Test 2: Mini Training Run
--------------------------------------------------------------------------------
  Epoch 1/2: Loss=0.6891, Acc=60.00%
  Epoch 2/2: Loss=0.6523, Acc=65.00%
✅ Mini Training: Trained 2 epochs, final acc=65.00%

--------------------------------------------------------------------------------
  Phase 2 Test 3: Model Evaluation
--------------------------------------------------------------------------------
✅ Model Evaluation: Acc=62.50%, AUC=0.687

--------------------------------------------------------------------------------
  Phase 2 Test 4: Grad-CAM Generation
--------------------------------------------------------------------------------
✅ Grad-CAM Generation: Generated (256, 256) heatmap with overlay

--------------------------------------------------------------------------------
  Phase 2 Test 5: Inference Pipeline
--------------------------------------------------------------------------------
✅ Inference Pipeline: Single: tumor (87.34%), Batch: 3 images

--------------------------------------------------------------------------------
  Phase 2 Test 6: API Endpoints
--------------------------------------------------------------------------------
❌ API Endpoints: Backend not running (optional test)

--------------------------------------------------------------------------------
  Phase 2 Test 7: Full Integration
--------------------------------------------------------------------------------
✅ Full Integration: ✓ Correct prediction with 89.23% confidence, Grad-CAM generated

================================================================================
  TEST SUMMARY
================================================================================

📊 PHASE 1 Results:
   Total Tests: 6
   Passed: 6
   Failed: 0
   Success Rate: 100.0%

📊 PHASE 2 Results:
   Total Tests: 7
   Passed: 6
   Failed: 1
   Success Rate: 85.7%

📊 OVERALL Results:
   Total Tests: 13
   Passed: 12
   Failed: 1
   Success Rate: 92.3%
   Time: 45.23s

📄 Results saved to: phase1_phase2_full_e2e_results.json

================================================================================
⚠️  1 test(s) failed. Review errors above.
   Phase 2: 1 failed
================================================================================
```

### **Test Results JSON**

Results are saved to `phase1_phase2_full_e2e_results.json`:

```json
{
  "timestamp": "2025-12-03 23:36:05",
  "phase1_tests": {
    "Data Download": {
      "status": "PASSED",
      "message": "Skipped (data exists): 154 tumor, 91 no tumor"
    },
    "Data Preprocessing": {
      "status": "PASSED",
      "message": "Using existing 245 preprocessed files"
    },
    ...
  },
  "phase2_tests": {
    "Model Creation": {
      "status": "PASSED",
      "message": "EfficientNet: 4,011,391 params, ConvNeXt: 27,821,314 params"
    },
    ...
  },
  "overall_status": "FAILED",
  "summary": {
    "phase1": {
      "total": 6,
      "passed": 6,
      "failed": 0,
      "success_rate": 100.0
    },
    "phase2": {
      "total": 7,
      "passed": 6,
      "failed": 1,
      "success_rate": 85.7
    },
    "overall": {
      "total": 13,
      "passed": 12,
      "failed": 1,
      "success_rate": 92.3,
      "elapsed_time": 45.23
    }
  }
}
```

---

## Troubleshooting

### **Test 1: Data Download Fails**

**Error**: `Data not found. Set skip_download=False to download.`

**Solution**:
1. Run with `--download-data` flag
2. Ensure Kaggle API credentials are configured
3. Or manually download data to `data/raw/kaggle_brain_mri/`

---

### **Test 2: Data Preprocessing Fails**

**Error**: `No .npz files created`

**Solution**:
1. Check that raw data exists in `data/raw/kaggle_brain_mri/yes/` and `no/`
2. Ensure you have write permissions to `data/processed/kaggle/`
3. Run preprocessing manually: `python src/data/preprocess_kaggle.py`

---

### **Test 3: Data Splitting Fails**

**Error**: `No training files`

**Solution**:
1. Ensure preprocessing completed successfully
2. Check that `data/processed/kaggle/` contains .npz files
3. Run splitting manually: `python src/data/split_kaggle.py`

---

### **Test 4-6: Phase 1 Dataset/Transform Failures**

**Error**: `Dataset is empty` or `Transform failed`

**Solution**:
1. Verify data splits exist in `data/processed/kaggle/train/`, `val/`, `test/`
2. Check file permissions
3. Ensure PyTorch and dependencies are installed correctly

---

### **Test 7: Model Creation Fails**

**Error**: `Wrong output shape` or `CUDA out of memory`

**Solution**:
1. If CUDA OOM, test will automatically fall back to CPU
2. Ensure PyTorch is installed with CUDA support (if using GPU)
3. Check model architecture compatibility

---

### **Test 8: Mini Training Fails**

**Error**: Training loop crashes

**Solution**:
1. Check CUDA/CPU compatibility
2. Reduce batch size if memory issues
3. Ensure dataloaders are working (Tests 4-6 passed)

---

### **Test 9-11: Evaluation/Grad-CAM/Inference Failures**

**Error**: Various runtime errors

**Solution**:
1. Ensure training completed (Test 8 passed)
2. Check checkpoint was saved to `checkpoints/cls/full_e2e_test_model.pth`
3. Verify model is in eval mode

---

### **Test 12: API Endpoints Fail**

**Error**: `Backend not running (optional test)`

**Solution**: This is **optional**. To test API:
1. Start backend: `python scripts/run_backend.py`
2. Ensure it's running on `http://localhost:8000`
3. Re-run the test

**Note**: API test failure does NOT indicate a critical issue.

---

### **Test 13: Full Integration Fails**

**Error**: Integration test crashes

**Solution**:
1. Ensure all previous tests passed (especially 1-11)
2. Check that checkpoint exists
3. Verify Grad-CAM layer is accessible

---

## Performance Benchmarks

### **Expected Timing** (CPU)

| Phase | Tests | Expected Time |
|-------|-------|---------------|
| Phase 1 | 6 | 5-10 seconds |
| Phase 2 | 7 | 30-60 seconds |
| **Total** | **13** | **35-70 seconds** |

### **Expected Timing** (GPU)

| Phase | Tests | Expected Time |
|-------|-------|---------------|
| Phase 1 | 6 | 5-10 seconds |
| Phase 2 | 7 | 15-30 seconds |
| **Total** | **13** | **20-40 seconds** |

**Note**: First run with data download may take 5-10 minutes.

---

## CI/CD Integration

### **GitHub Actions Example**

```yaml
name: Phase 1 & 2 E2E Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run E2E Test
        run: python scripts/test_phase1_phase2_full_e2e.py --use-temp-dir --download-data
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
```

---

## What to Do After Running

### **All Tests Pass** ✅

Congratulations! Your Phase 1 and Phase 2 implementation is production-ready.

**Next Steps**:
1. Review the results JSON for detailed metrics
2. Proceed to Phase 3 (U-Net Segmentation)
3. Deploy the API and frontend for production use

### **Some Tests Fail** ⚠️

1. Review the error messages in the console output
2. Check the troubleshooting section above
3. Fix the failing components
4. Re-run the test

### **Many Tests Fail** ❌

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Verify data exists in the correct directories
3. Check file permissions
4. Review the Phase 0, 1, and 2 documentation
5. Run individual components manually to isolate the issue

---

## Related Documentation

- **Phase 0**: [`PHASE0_COMPLETE.md`](PHASE0_COMPLETE.md) - Project setup
- **Phase 1**: Data acquisition and preprocessing (see `FULL-PLAN.md`)
- **Phase 2**: [`PHASE2_COMPLETE.md`](PHASE2_COMPLETE.md) - Classification MVP
- **Quick Start**: [`PHASE2_QUICKSTART.md`](PHASE2_QUICKSTART.md) - 5-minute guide
- **Full Plan**: [`FULL-PLAN.md`](FULL-PLAN.md) - Complete project roadmap

---

## Support

If you encounter issues not covered in this guide:

1. Check the main `README.md` for general setup instructions
2. Review the individual component documentation
3. Ensure all prerequisites are met
4. Check that your Python version is 3.10 or 3.11

---

## Summary

The Phase 1 & Phase 2 E2E test is a **comprehensive validation** of your entire pipeline. It ensures:

- ✅ Data can be acquired and preprocessed correctly
- ✅ Datasets and transforms work as expected
- ✅ Models can be created and trained
- ✅ Evaluation and explainability features work
- ✅ Inference pipeline is production-ready
- ✅ API integration is functional (optional)

**Run this test regularly** to ensure your system remains production-ready as you make changes!
