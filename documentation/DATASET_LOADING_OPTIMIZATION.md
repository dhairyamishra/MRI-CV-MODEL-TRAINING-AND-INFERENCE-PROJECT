# Dataset Loading Optimization for Quick Mode

**Date**: December 10, 2025  
**Solution**: Use subset of BraTS dataset in quick mode for faster data loading

---

## Problem

When running `python scripts/run_full_pipeline.py --mode full --training-mode quick`, the data loading was slow because:
- BraTS has 988 patients total
- Even in quick mode, it would try to process many patients
- Loading and preprocessing takes significant time

---

## Solution: Use Dataset Subsets

Instead of skipping BraTS entirely, we now use **intelligent subsets** based on training mode:

| Mode | BraTS Patients | Percentage | Purpose |
|------|----------------|------------|---------|
| **Quick** | 100 | ~10% | Fast testing, CI/CD |
| **Baseline** | 300 | ~30% | Development, experiments |
| **Production** | 988 | 100% | Full training |

---

## Implementation

### File: `scripts/run_full_pipeline.py` (Lines 280-290)

```python
# Determine number of patients based on training mode
# Quick mode: Use ~10% of BraTS dataset for faster loading
if self.args.training_mode == "quick":
    num_patients = 100  # ~10% of 988 patients
    self._print_info(f"Quick mode: Processing {num_patients} patients (~10% of dataset for faster loading)")
elif self.args.training_mode == "baseline":
    num_patients = 300  # ~30% for baseline
    self._print_info(f"Baseline mode: Processing {num_patients} patients (~30% of dataset)")
else:  # production
    num_patients = None  # Process all 988 patients
    self._print_info("Production mode: Processing ALL patients (988)")
```

The `--num-patients` parameter is then passed to the preprocessing script:
```python
cmd.extend(["--num-patients", str(num_patients)])
```

---

## Benefits

### Quick Mode (100 patients):
- ✅ **Faster preprocessing**: ~10-15 minutes vs 2-4 hours
- ✅ **Faster training**: Less data to load per epoch
- ✅ **Still representative**: 100 patients is enough for testing
- ✅ **Full pipeline**: Tests all 3 stages (seg warmup, cls head, joint)
- ✅ **Both datasets**: Uses BraTS (100) + Kaggle (245)

### Baseline Mode (300 patients):
- ✅ **Better results**: More data than quick mode
- ✅ **Reasonable time**: ~1-2 hours preprocessing
- ✅ **Good for development**: Balance between speed and quality

### Production Mode (988 patients):
- ✅ **Best results**: Full dataset
- ✅ **Publication-ready**: Complete training
- ✅ **Long but thorough**: 2-4 hours preprocessing

---

## Expected Timeline

### Quick Mode (~30-45 minutes total):
```
Step 1: Download BraTS + Kaggle          → 10-30 min (one-time)
Step 2: Preprocess 100 BraTS + Kaggle    → 10-15 min
Step 3: Split data                        → <1 min
Step 4: Multi-task training (3 stages)    → 10-15 min
Step 5: Evaluation                        → 2-5 min
Step 6: Demo launch                       → instant
```

### Baseline Mode (~2-4 hours total):
```
Step 1: Download BraTS + Kaggle          → 10-30 min (one-time)
Step 2: Preprocess 300 BraTS + Kaggle    → 30-60 min
Step 3: Split data                        → <1 min
Step 4: Multi-task training (3 stages)    → 1-2 hours
Step 5: Evaluation                        → 10-15 min
Step 6: Demo launch                       → instant
```

### Production Mode (~8-12 hours total):
```
Step 1: Download BraTS + Kaggle          → 10-30 min (one-time)
Step 2: Preprocess 988 BraTS + Kaggle    → 2-4 hours
Step 3: Split data                        → <1 min
Step 4: Multi-task training (3 stages)    → 4-6 hours
Step 5: Evaluation                        → 30-60 min
Step 6: Demo launch                       → instant
```

---

## Usage

```bash
# Quick mode - 100 BraTS patients (~10%)
python scripts/run_full_pipeline.py --mode full --training-mode quick

# Baseline mode - 300 BraTS patients (~30%)
python scripts/run_full_pipeline.py --mode full --training-mode baseline

# Production mode - 988 BraTS patients (100%)
python scripts/run_full_pipeline.py --mode full --training-mode production
```

---

## Manual Override

If you want to customize the number of patients:

```bash
# Preprocess specific number of patients
python scripts/data/preprocessing/preprocess_all_brats.py --num-patients 50

# Or skip preprocessing and use existing data
python scripts/run_full_pipeline.py --mode full --skip-preprocessing
```

---

## Why This Approach is Better

### ❌ Previous Approach (Rejected):
- Skip BraTS entirely in quick mode
- Train classification-only
- Miss testing segmentation pipeline
- Not representative of full system

### ✅ Current Approach (Implemented):
- Use 10% of BraTS in quick mode
- Train full multi-task model
- Test complete pipeline
- Representative subset
- Much faster than full dataset

---

## Performance Comparison

| Metric | Quick (100) | Baseline (300) | Production (988) |
|--------|-------------|----------------|------------------|
| **Preprocessing Time** | 10-15 min | 30-60 min | 2-4 hours |
| **Training Time** | 10-15 min | 1-2 hours | 4-6 hours |
| **Total Time** | 30-45 min | 2-4 hours | 8-12 hours |
| **BraTS Slices** | ~5,000 | ~15,000 | ~50,000 |
| **Kaggle Images** | 245 | 245 | 245 |
| **Model Type** | Multi-task | Multi-task | Multi-task |
| **Use Case** | Testing, CI/CD | Development | Production |

---

## Technical Details

### BraTS Dataset Structure:
- Total patients: 988
- Average slices per patient: ~50-60
- Total slices (approx): ~50,000

### Quick Mode (100 patients):
- Slices: ~5,000-6,000
- Enough for: Testing all pipeline stages
- Training time: Reasonable for quick iteration

### Baseline Mode (300 patients):
- Slices: ~15,000-18,000
- Enough for: Good model performance
- Training time: Acceptable for development

### Production Mode (988 patients):
- Slices: ~50,000
- Enough for: Best model performance
- Training time: Long but necessary for publication

---

**Status**: ✅ Implemented and optimized  
**Impact**: 10x faster quick mode while maintaining full pipeline testing
