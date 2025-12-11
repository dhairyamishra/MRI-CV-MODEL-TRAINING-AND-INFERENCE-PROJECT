# Bug Fix: Missing Kaggle Preprocessing in Pipeline

**Date**: December 10, 2025  
**Issue**: Users getting "No samples found in provided directories" error when running evaluation  
**Root Cause**: `run_full_pipeline.py` was missing the Kaggle preprocessing step

---

## Problem Description

When users ran:
```bash
python scripts/run_full_pipeline.py --mode full --training-mode quick
```

The pipeline would fail during evaluation with:
```
Loaded 0 samples from data/processed/kaggle/test
ValueError: No samples found in provided directories
```

## Root Cause Analysis

The `run_full_pipeline.py` script had a **critical bug** in the `run_data_preprocessing()` function:

### Expected Data Pipeline Flow:
1. **Download** → `data/raw/kaggle_brain_mri/{yes,no}/*.jpg`
2. **Preprocess** → `data/processed/kaggle/*.npz` (convert JPG to NPZ)
3. **Split** → `data/processed/kaggle/{train,val,test}/*.npz`

### Actual Buggy Flow:
1. ✅ **Download** → `data/raw/kaggle_brain_mri/{yes,no}/*.jpg` (worked)
2. ❌ **Preprocess** → SKIPPED! Only preprocessed BraTS data
3. ❌ **Split** → Failed because no `.npz` files existed to split

### The Bug (Line 272-311 in `run_full_pipeline.py`):

```python
def run_data_preprocessing(self) -> bool:
    # ... BraTS preprocessing code ...
    
    if not self._run_command(cmd, "preprocess_brats", timeout=timeout):
        return False
    
    return True  # ❌ BUG: Returns without preprocessing Kaggle data!
```

The function only preprocessed BraTS data and completely skipped the Kaggle dataset preprocessing step.

## Files Fixed

### 1. `scripts/run_full_pipeline.py` (Multiple fixes)

#### Fix 1: Path Mismatch in Download Check (Line 244)

**Bug:** The script checked for `data/raw/kaggle/` but the download script creates `data/raw/kaggle_brain_mri/`

**Before:**
```python
kaggle_dir = self.project_root / "data" / "raw" / "kaggle"  # ❌ Wrong path
```

**After:**
```python
kaggle_dir = self.project_root / "data" / "raw" / "kaggle_brain_mri"  # ✅ Correct path
```

**Impact:** The check would never detect existing Kaggle data, causing unnecessary re-downloads or skipping the prompt entirely.

#### Fix 2: Missing Kaggle Preprocessing Step (Lines 312-322)

**Added missing Kaggle preprocessing step:**

```python
# Preprocess Kaggle data (JPG → NPZ conversion)
self._print_info("Preprocessing Kaggle data (JPG→NPZ conversion, normalization)...")
if not self._run_command(
    ["python", "src/data/preprocess_kaggle.py",
     "--raw-dir", "data/raw/kaggle_brain_mri",
     "--processed-dir", "data/processed/kaggle",
     "--target-size", "256", "256"],
    "preprocess_kaggle",
    timeout=300  # 5 minutes should be enough for 245 images
):
    return False
```

### 2. `README.md` (Lines 175-211)

**Fixed incorrect script paths in documentation:**

| Section | Old Path (Incorrect) | New Path (Correct) |
|---------|---------------------|-------------------|
| Kaggle Download | `scripts/download_kaggle_data.py` | `scripts/data/collection/download_kaggle_data.py` |
| Kaggle Split | `src/data/split_kaggle.py` | `scripts/data/splitting/split_kaggle_data.py` |
| BraTS Download | `scripts/download_brats_data.py` | `scripts/data/collection/download_brats_data.py` |
| BraTS Preprocess | `scripts/preprocess_all_brats.py` | `scripts/data/preprocessing/preprocess_all_brats.py` |
| BraTS Split | `src/data/split_brats.py` | `scripts/data/splitting/split_brats_data.py` |

## Verification

After the fix, the complete data pipeline now works correctly:

```bash
# Clean slate test
rm -rf data/raw/kaggle_brain_mri data/processed/kaggle

# Run pipeline
python scripts/run_full_pipeline.py --mode full --training-mode quick
```

**Expected Output:**
```
[Step 1/6] Data Download
  ✓ Downloading Kaggle brain MRI dataset...
  
[Step 2/6] Data Preprocessing
  ✓ Preprocessing BraTS data (3D→2D conversion, normalization)...
  ✓ Preprocessing Kaggle data (JPG→NPZ conversion, normalization)...  # ← NEW!
  
[Step 3/6] Data Splitting
  ✓ Splitting Kaggle data (patient-level, 70/15/15)...
  
[Step 4/6] Multi-Task Training
  ...
```

**Result:**
- `data/processed/kaggle/train/` → 171 samples
- `data/processed/kaggle/val/` → 37 samples
- `data/processed/kaggle/test/` → 37 samples

## Impact

This bug affected **all users** who tried to run the full pipeline from scratch. The bug was introduced when the scripts were reorganized into subdirectories (`data/collection/`, `data/preprocessing/`, `data/splitting/`) but the pipeline controller wasn't updated to include the Kaggle preprocessing step.

## Prevention

To prevent similar issues in the future:

1. **Integration Tests**: Add E2E tests that run the full pipeline on a small dataset
2. **Data Validation**: Add checks in each step to verify expected data exists
3. **Better Error Messages**: Update error messages to guide users on missing steps
4. **Documentation Review**: Keep README in sync with actual script locations

## Related Issues

- Users reporting "No samples found" errors
- Evaluation scripts failing immediately
- Manual workaround: Running preprocessing steps individually

---

**Status**: ✅ Fixed and tested  
**Affected Versions**: All versions before this fix  
**Fix Version**: December 10, 2025
