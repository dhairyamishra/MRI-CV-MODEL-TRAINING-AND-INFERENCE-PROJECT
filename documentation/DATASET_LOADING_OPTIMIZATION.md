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

Instead of skipping BraTS entirely, we now use **intelligent subsets** based on training mode.

The system **dynamically calculates** the number of patients based on the actual dataset size:

| Mode | Percentage | Calculation | Purpose |
|------|------------|-------------|---------|
| **Quick** | 5% | `max(2, int(total * 0.05))` | Fast testing, CI/CD |
| **Baseline** | 30% | `max(50, int(total * 0.30))` | Development, experiments |
| **Production** | 100% | All patients | Full training |

### Examples:
- **500 patients available**: Quick=25, Baseline=150, Production=500
- **988 patients available**: Quick=49, Baseline=296, Production=988
- **100 patients available**: Quick=5, Baseline=50, Production=100
- **40 patients available**: Quick=2 (minimum), Baseline=50, Production=40

---

## Implementation

### File: `scripts/run_full_pipeline.py` (Lines 280-302)

```python
# Dynamically determine number of patients based on actual dataset size
brats_raw_dir = self.project_root / "data" / "raw" / "brats2020"

# Count available BraTS patient folders
total_patients = 0
if brats_raw_dir.exists():
    patient_folders = list(brats_raw_dir.glob("BraTS*"))
    total_patients = len(patient_folders)
    self._print_info(f"Found {total_patients} BraTS patient folders")
else:
    self._print_warning(f"BraTS directory not found: {brats_raw_dir}")
    total_patients = 500  # Fallback estimate

# Calculate percentages based on actual dataset size
if self.args.training_mode == "quick":
    num_patients = max(2, int(total_patients * 0.05))  # 5% with minimum of 2
    self._print_info(f"Quick mode: Processing {num_patients} patients (~5% of {total_patients} for faster loading)")
elif self.args.training_mode == "baseline":
    num_patients = max(50, int(total_patients * 0.30))  # 30% with minimum of 50
    self._print_info(f"Baseline mode: Processing {num_patients} patients (~30% of {total_patients})")
else:  # production
    num_patients = None  # Process all patients
    self._print_info(f"Production mode: Processing ALL {total_patients} patients")

The `--num-patients` parameter is then passed to the preprocessing script:
```python
cmd.extend(["--num-patients", str(num_patients)])
```

---

## Benefits

### Quick Mode (5% = 24 patients for 496 total, or 49 for 988 total):
- ✅ **Ultra-fast preprocessing**: ~2-3 minutes vs 2-4 hours
- ✅ **Faster training**: Minimal data to load per epoch
- ✅ **Still representative**: 24-49 patients is enough for pipeline testing
- ✅ **Full pipeline**: Tests all 3 stages (seg warmup, cls head, joint)
- ✅ **Both datasets**: Uses BraTS (5%) + Kaggle (245)
- ✅ **Dynamic scaling**: Adapts to any dataset size automatically

### Baseline Mode (30% = 148 patients for 496 total, or 296 for 988 total):
- ✅ **Better results**: More data than quick mode
- ✅ **Reasonable time**: ~30-60 minutes preprocessing
- ✅ **Good for development**: Balance between speed and quality
- ✅ **Dynamic scaling**: Automatically calculates 30% of available patients

### Production Mode (100% = all available patients):
- ✅ **Best results**: Full dataset
- ✅ **Publication-ready**: Complete training
- ✅ **Long but thorough**: 2-4 hours preprocessing
- ✅ **Adapts to dataset**: Works with any BraTS version (2020, 2021, custom)

---

## Expected Timeline

### Quick Mode (~10-15 minutes total) ⚡:
```
Step 1: Download BraTS + Kaggle          → 10-30 min (one-time, skipped if exists)
Step 2: Preprocess 5% BraTS + Kaggle     → 2-3 min (24 patients for 496 total)
Step 3: Split data                        → <1 min
Step 4: Multi-task training (3 stages)    → 5-10 min
Step 5: Evaluation                        → 2-5 min
Step 6: Demo launch                       → instant
```

### Baseline Mode (~2-4 hours total):
```
Step 1: Download BraTS + Kaggle          → 10-30 min (one-time, skipped if exists)
Step 2: Preprocess 30% BraTS + Kaggle    → 30-60 min (148 patients for 496 total)
Step 3: Split data                        → <1 min
Step 4: Multi-task training (3 stages)    → 1-2 hours
Step 5: Evaluation                        → 10-15 min
Step 6: Demo launch                       → instant
```

### Production Mode (~8-12 hours total):
```
Step 1: Download BraTS + Kaggle          → 10-30 min (one-time, skipped if exists)
Step 2: Preprocess 100% BraTS + Kaggle   → 2-4 hours (496 or 988 patients)
Step 3: Split data                        → <1 min
Step 4: Multi-task training (3 stages)    → 4-6 hours
Step 5: Evaluation                        → 30-60 min
Step 6: Demo launch                       → instant
```

---

## Usage

```bash
# Quick mode - 5% of BraTS patients (min 2, e.g., 24 for 496 total)
python scripts/run_full_pipeline.py --mode full --training-mode quick

# Baseline mode - 30% of BraTS patients (min 50, e.g., 148 for 496 total)
python scripts/run_full_pipeline.py --mode full --training-mode baseline

# Production mode - 100% of BraTS patients (all available)
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
- Use 5% of BraTS in quick mode (with minimum of 2 patients)
- Dynamic scaling based on actual dataset size
- Train full multi-task model
- Test complete pipeline
- Representative subset
- 2x faster than previous 10% approach

---

## Performance Comparison

| Metric | Quick (5%) | Baseline (30%) | Production (100%) |
|--------|------------|----------------|-------------------|
| **BraTS Patients** | 24 (496 total) | 148 (496 total) | 496 (all) |
| **Preprocessing Time** | 2-3 min | 30-60 min | 2-4 hours |
| **Training Time** | 5-10 min | 1-2 hours | 4-6 hours |
| **Total Time** | 10-15 min | 2-4 hours | 8-12 hours |
| **BraTS Slices** | ~1,200 | ~7,400 | ~24,800 |
| **Kaggle Images** | 245 | 245 | 245 |
| **Model Type** | Multi-task | Multi-task | Multi-task |
| **Use Case** | Testing, CI/CD | Development | Production |
| **Dynamic Scaling** | ✅ Yes | ✅ Yes | ✅ Yes |

---

## Technical Details

### BraTS Dataset Structure:
- Total patients: 496 (typical) or 988 (full BraTS 2020)
- Average slices per patient: ~50-60
- Total slices (approx): ~24,800 (496) or ~50,000 (988)

### Quick Mode (5% with minimum 2 patients):
- Patients: 24 (for 496 total) or 49 (for 988 total)
- Slices: ~1,200-2,400
- Enough for: Testing all pipeline stages
- Training time: Ultra-fast for quick iteration (5-10 min)
- **2x faster** than previous 10% approach

### Baseline Mode (30% with minimum 50 patients):
- Patients: 148 (for 496 total) or 296 (for 988 total)
- Slices: ~7,400-15,000
- Enough for: Good model performance
- Training time: Acceptable for development (1-2 hours)

### Production Mode (100% of available patients):
- Patients: All available (496, 988, or custom)
- Slices: ~24,800-50,000+
- Enough for: Best model performance
- Training time: Long but necessary for publication (4-6 hours)
- **Adapts to any dataset size automatically**

---

**Status**: ✅ Implemented and optimized (December 10, 2025)  
**Impact**: 2x faster quick mode (5% vs 10%) while maintaining full pipeline testing  
**Key Feature**: Dynamic scaling adapts to any dataset size automatically
