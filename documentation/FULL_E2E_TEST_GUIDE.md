# Full End-to-End Test Guide
## SliceWise: Phase 1 → Phase 6 Validation

This guide explains how to run the comprehensive end-to-end test that validates your entire SliceWise pipeline.

---

## 📋 What This Test Covers

The full E2E test validates **all 6 phases** of the SliceWise project:

### **Phase 1: Data Acquisition & Preprocessing**
- ✓ Kaggle dataset availability and loading
- ✓ BraTS dataset availability (optional)
- ✓ Dataset class functionality
- ✓ DataLoader creation and batching

### **Phase 2: Classification Pipeline**
- ✓ Classifier model creation (EfficientNet-B0)
- ✓ Forward pass and gradient flow
- ✓ Trained model checkpoint loading
- ✓ Inference with ClassifierPredictor
- ✓ Grad-CAM generation

### **Phase 3: Segmentation Pipeline**
- ✓ U-Net model creation
- ✓ Forward pass and architecture
- ✓ Trained segmentation checkpoint
- ✓ Inference with SegmentationPredictor
- ✓ Post-processing pipeline

### **Phase 4: Calibration & Uncertainty**
- ✓ Temperature scaling for calibration
- ✓ MC Dropout uncertainty estimation
- ✓ Test-Time Augmentation (TTA)
- ✓ Ensemble predictor (MC + TTA)

### **Phase 5: Metrics & Patient-Level**
- ✓ Dice coefficient and IoU computation
- ✓ Patient-level aggregation
- ✓ Volume estimation

### **Phase 6: API & Integration**
- ✓ Backend API health check
- ✓ `/model/info` endpoint
- ✓ `/classify` endpoint
- ✓ `/segment` endpoint

---

## 🚀 Quick Start

### **Option 1: Full Test (Recommended)**

```bash
# Make sure backend is running first
python scripts/run_demo_backend.py

# In another terminal, run the full test
python scripts/test_full_e2e_phase1_to_phase6.py
```

### **Option 2: Quick Mode (Faster)**

```bash
python scripts/test_full_e2e_phase1_to_phase6.py --quick
```

### **Option 3: Without API Tests**

If you don't want to start the backend:

```bash
python scripts/test_full_e2e_phase1_to_phase6.py
# API tests will be skipped with warnings
```

---

## 📊 Expected Output

The test will display color-coded output:

```
================================================================================
                SliceWise Full E2E Test: Phase 1 → Phase 6
================================================================================

ℹ Device: cuda
ℹ Quick mode: False
ℹ Skip training: False

================================================================================
                    PHASE 1: Data Acquisition & Preprocessing
================================================================================

▶ Testing: Kaggle dataset availability
✓ Kaggle train set found: 171 files

▶ Testing: Kaggle dataset loading
✓ Dataset loaded: 171 samples
✓ Sample shape: torch.Size([1, 256, 256]), label: 1

...

================================================================================
                              TEST SUMMARY
================================================================================

Overall Results:
  Total Tests:    25
  Passed:         23
  Failed:         0
  Warnings:       2
  Pass Rate:      100.0%

Phase Breakdown:
  ✓ Phase 1: 4/4 (100%)
  ✓ Phase 2: 4/4 (100%)
  ✓ Phase 3: 4/4 (100%)
  ✓ Phase 4: 4/4 (100%)
  ✓ Phase 5: 2/2 (100%)
  ✓ Phase 6: 3/3 (100%)

Results saved to: full_e2e_test_results.json

================================================================================
                          ALL TESTS PASSED! 🎉
================================================================================
```

---

## 📁 Test Results

Results are automatically saved to:
```
full_e2e_test_results.json
```

This JSON file contains:
- Start and end timestamps
- Detailed results for each phase
- Individual test outcomes
- Pass/fail statistics
- Warnings and errors

---

## ⚙️ Prerequisites

### **Required:**
1. **Kaggle dataset** preprocessed and split:
   ```
   data/processed/kaggle/train/
   data/processed/kaggle/val/
   data/processed/kaggle/test/
   ```

2. **Trained classifier** checkpoint:
   ```
   checkpoints/cls/best_model.pth
   ```

3. **Trained segmentation** checkpoint:
   ```
   checkpoints/seg/best_model.pth
   ```

### **Optional:**
4. **BraTS dataset** (for additional validation):
   ```
   data/processed/brats2d/train/
   ```

5. **Calibration checkpoint** (for Phase 4):
   ```
   checkpoints/cls/temperature_scaler.pth
   ```

6. **Backend API running** (for Phase 6):
   ```bash
   python scripts/run_demo_backend.py
   ```

---

## 🔧 Command-Line Options

```bash
python scripts/test_full_e2e_phase1_to_phase6.py [OPTIONS]

Options:
  --quick              Use smaller batch sizes for faster testing
  --skip-training      Skip training-related tests (not implemented yet)
  -h, --help          Show help message
```

---

## 🐛 Troubleshooting

### **Issue: "Kaggle dataset not found"**
**Solution:** Run preprocessing first:
```bash
python scripts/download_kaggle_data.py
python src/data/preprocess_kaggle.py
python src/data/split_kaggle.py
```

### **Issue: "No trained checkpoint found"**
**Solution:** Train the models:
```bash
# Train classifier
python scripts/train_classifier.py

# Train segmentation
python scripts/train_segmentation.py
```

### **Issue: "Backend API not running"**
**Solution:** Start the backend in a separate terminal:
```bash
python scripts/run_demo_backend.py
```
Then re-run the test.

### **Issue: "CUDA out of memory"**
**Solution:** Use quick mode or run on CPU:
```bash
# Quick mode
python scripts/test_full_e2e_phase1_to_phase6.py --quick

# Or set CUDA_VISIBLE_DEVICES=""
CUDA_VISIBLE_DEVICES="" python scripts/test_full_e2e_phase1_to_phase6.py
```

---

## 📈 Understanding Test Results

### **Color Codes:**
- 🟢 **Green (✓)**: Test passed
- 🔴 **Red (✗)**: Test failed
- 🟡 **Yellow (⚠)**: Warning (optional component missing)
- 🔵 **Blue (▶)**: Test in progress
- 🔷 **Cyan (ℹ)**: Information

### **Test Categories:**

| Category | What It Tests | Critical? |
|----------|---------------|-----------|
| **Data Loading** | Dataset files exist and can be loaded | ✅ Yes |
| **Model Creation** | Models can be instantiated | ✅ Yes |
| **Forward Pass** | Models can process inputs | ✅ Yes |
| **Checkpoints** | Trained models can be loaded | ⚠️ Optional |
| **Inference** | Predictions work correctly | ✅ Yes |
| **API Endpoints** | Backend responds correctly | ⚠️ Optional |

---

## 🎯 Success Criteria

### **Minimum for Success:**
- ✅ All Phase 1 tests pass (data loading)
- ✅ All Phase 2 model creation tests pass
- ✅ All Phase 3 model creation tests pass
- ✅ At least one inference test passes

### **Full Success:**
- ✅ All tests pass (100% pass rate)
- ✅ No warnings
- ✅ All API endpoints respond correctly

### **Acceptable with Warnings:**
- ✅ Core tests pass
- ⚠️ Optional components missing (BraTS, calibration, API)
- ⚠️ Pass rate > 80%

---

## 📝 Next Steps After Testing

### **If All Tests Pass:**
1. ✅ Your pipeline is fully functional!
2. ✅ Ready for Phase 7 (Documentation)
3. ✅ Can proceed with production deployment

### **If Some Tests Fail:**
1. Check the error messages in the output
2. Review the `full_e2e_test_results.json` file
3. Fix the failing components
4. Re-run the test

### **If Warnings Appear:**
1. Review which optional components are missing
2. Decide if you need them for your use case
3. Optionally train/prepare missing components

---

## 🔄 Continuous Integration

You can integrate this test into your CI/CD pipeline:

```yaml
# .github/workflows/e2e-test.yml
name: Full E2E Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run E2E Test
        run: python scripts/test_full_e2e_phase1_to_phase6.py --quick
```

---

## 📚 Related Documentation

- **PHASE1_PHASE2_E2E_TEST.md** - Original Phase 1-2 test
- **PHASE6_QUICKSTART.md** - Quick start for demo app
- **PHASE6_COMPLETE.md** - Complete Phase 6 documentation
- **FULL-PLAN.md** - Overall project plan

---

## 💡 Tips

1. **Run regularly**: Test after major changes to catch regressions early
2. **Use quick mode**: For rapid iteration during development
3. **Check JSON output**: For detailed debugging information
4. **Start backend first**: For complete Phase 6 testing
5. **Monitor GPU memory**: Use `nvidia-smi` to check usage

---

## 🎉 Example Success Output

```
================================================================================
                          ALL TESTS PASSED! 🎉
================================================================================

✓ Phase 1: Data Acquisition & Preprocessing (4/4)
✓ Phase 2: Classification Pipeline (4/4)
✓ Phase 3: Segmentation Pipeline (4/4)
✓ Phase 4: Calibration & Uncertainty (4/4)
✓ Phase 5: Metrics & Patient-Level (2/2)
✓ Phase 6: API & Integration (3/3)

Total: 21/21 tests passed (100%)
Time: 45.3 seconds

Your SliceWise pipeline is fully functional! 🚀
```

---

**Last Updated:** 2024-12-04  
**Version:** 1.0  
**Author:** SliceWise Team
