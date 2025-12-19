# Multi-Task Model Debug & Fix Plan

**Date:** December 18, 2025  
**Status:** Analysis Complete → Ready for Implementation  
**Priority:** CRITICAL - Multi-task classifier is inaccurate due to data pipeline issues

---

## Executive Summary

The standalone tumor classifier achieves **94.59% accuracy** with focused Grad-CAM attention on lesions, while the multi-task model shows poor classification performance with diffuse, non-specific Grad-CAM activations across skull boundaries and background regions.

**Root Cause:** Normalization mismatch and missing preprocessing transforms during multi-task training create inconsistent input distributions that prevent the model from learning tumor-specific features.

---

## Problem Statement

### Observed Symptoms

1. **Classification Performance Gap**
   - Standalone classifier: 94.59% accuracy, 99.38% ROC-AUC
   - Multi-task classifier: Significantly lower accuracy (exact metrics TBD)

2. **Grad-CAM Visualization Differences**
   - **Standalone model:** Tight, focused red blob over tumor region (correct localization)
   - **Multi-task model:** Diffuse activations with multiple hotspots on edges, corners, and skull boundaries (spurious features)

3. **Model Behavior**
   - Multi-task model appears to rely on dataset artifacts, global intensity patterns, and border effects rather than tumor morphology

---

## Root Cause Analysis

### 1. Normalization Mismatch (CRITICAL)

**The Problem:**
- **Kaggle dataset (standalone):** Preprocessed with min-max normalization to [0, 1] range
  - File: `src/data/preprocess_kaggle.py`
  - Line 34: `img = img.astype(np.float32) / 255.0`
  
- **BraTS dataset (multi-task):** Preprocessed with z-score normalization (mean=0, std=1)
  - File: `src/data/preprocess_brats_2d.py`
  - Lines 34-41: Z-score normalization with mean/std calculation
  
- **Multi-task training:** Uses mismatched Kaggle data
  - Configs point to: `data/processed/kaggle/` (min-max normalized)
  - Should point to: `data/processed/kaggle_unified/` (z-score normalized)
  - Files affected:
    - `configs/stages/stage2_cls_head.yaml` (lines 17-18)
    - `configs/stages/stage3_joint.yaml` (lines 17-18)

**Impact:**
- Multi-task encoder sees mixed distributions in single batches:
  - BraTS samples: z-score normalized (includes negative values, ~[-3, +3] range)
  - Kaggle samples: min-max normalized ([0, 1] range)
- Model cannot learn consistent feature representations
- Leads to reliance on global intensity biases and dataset-specific artifacts

**Evidence:**
- Unified preprocessing script exists but is unused: `src/data/preprocess_kaggle_unified.py`
- Multi-task predictor uses z-score at inference (line 234), creating additional train-test mismatch
- Diffuse Grad-CAMs indicate model learning spurious correlations

---

### 2. Missing Preprocessing Transforms (HIGH PRIORITY)

**The Problem:**
- **Standalone classifier:** Uses comprehensive transforms
  - File: `src/training/train_cls.py`
  - Lines 181-194: Applies `get_train_transforms()` and `get_val_transforms()`
  - Includes: `SkullBoundaryMask`, flips, rotations, intensity augmentations
  
- **Multi-task training:** No transforms applied
  - File: `src/training/train_multitask_cls_head.py`
  - Line 259: `transform=None` (with TODO comment)
  - File: `src/training/train_multitask_joint.py`
  - Similar `transform=None` usage

**Impact:**
- Skull boundaries and background artifacts remain in multi-task training data
- Model learns to attend to borders, corners, and non-tumor regions
- Grad-CAM shows activations on skull edges instead of tumor
- Reduced generalization due to lack of augmentation

**Evidence:**
- `src/data/transforms.py` defines `SkullBoundaryMask` (lines 14-114) and augmentations
- Standalone model's focused Grad-CAMs demonstrate benefit of skull masking
- Multi-task Grad-CAMs show edge/corner activations consistent with unmasked borders

---

### 3. Dataset Mixing Not Implemented (MEDIUM PRIORITY)

**The Problem:**
- **BraTS preprocessing:** Saves only tumor slices by default
  - File: `src/data/preprocess_brats_2d.py`
  - Line 181: `if not save_all_slices and not has_tumor(...): continue`
  - Default: `save_all_slices=False` (line 244)
  - Result: BraTS contributes almost exclusively class-1 (tumor) samples
  
- **Dataset imbalance:** BraTS (thousands of tumor slices) overwhelms Kaggle (~245 total images)
  
- **Config vs Code gap:** Mixing ratios specified but not enforced
  - Config: `configs/stages/stage3_joint.yaml` (lines 52-54)
    - `brats_ratio: 0.3`
    - `kaggle_ratio: 0.7`
  - Code: `src/training/train_multitask_joint.py`
    - Simply concatenates datasets and shuffles
    - No per-batch ratio enforcement or weighted sampling

**Impact:**
- Severe class imbalance toward tumor class
- Classification head becomes biased toward predicting class-1
- Kaggle negative samples underrepresented in training
- Model may learn "always predict tumor" shortcut

**Evidence:**
- `src/data/multi_source_dataset.py` (lines 129-130): BraTS labels derived from mask presence
- No sampler or batch composition logic in training scripts
- Config fields exist but are unused by code

---

### 4. Loss Weighting Not Wired (LOW PRIORITY)

**The Problem:**
- **Config specification:** Stage 3 defines loss weights
  - File: `configs/stages/stage3_joint.yaml` (lines 34-43)
  - Nested structure: `loss.seg_loss.name`, `loss.cls_loss.name`, `loss.lambda_cls: 0.5`
  
- **Code expectation:** Different flat structure
  - File: `src/training/multi_task_losses.py`
  - Function `create_multi_task_loss()` expects flat keys like `seg_loss_type`, `cls_loss_type`
  - Mismatch causes fallback to defaults (lambda_cls=1.0 instead of 0.5)

**Impact:**
- Classification loss not properly weighted relative to segmentation
- May contribute to suboptimal multi-task balance
- Less critical than normalization/transforms but reduces fine-tuning control

---

### 5. Architecture Differences (EXPECTED, NOT PRIMARY CAUSE)

**Observations:**
- Standalone: EfficientNet/ConvNeXt with ImageNet pretraining
- Multi-task: U-Net encoder, randomly initialized, warm-started on segmentation only
- Multi-task classification head: Small hidden dim (64 for `multitask_medium`)

**Assessment:**
- These differences are expected and acceptable
- Not the primary cause of poor performance
- Data pipeline issues must be fixed first before considering architecture changes

---

## Detailed Fix Plan

### Pre-Phase 1: Grad-CAM Sanity Check (10 minutes)

**Objective:** Verify Grad-CAM implementation is correct before fixing training pipeline

Before investing time in training fixes, confirm your Grad-CAM visualization is actually showing what you think it's showing. A misconfigured Grad-CAM can mislead your entire debugging process.

#### Checklist: Grad-CAM Configuration Verification

**File to check:** `src/inference/multi_task_predictor.py` (lines 417-469: `predict_with_gradcam()`)

- [ ] **Backprop from classification logits only**
  - Verify: Grad-CAM backward pass uses `output['cls']` (classification logits), NOT:
    - ❌ Segmentation logits (`output['seg']`)
    - ❌ Combined loss value
    - ❌ Dictionary output
  - Location: Line 442 in `ClassificationWrapper.forward()` - should return `output['cls']`
  - Why: Backprop from wrong head will show wrong attention patterns

- [ ] **Target layer is encoder bottleneck**
  - Verify: Hook attached to `self.model.encoder.down_blocks[-1]` (line 448)
  - NOT attached to:
    - ❌ Skip connection layers (will highlight edges/boundaries)
    - ❌ Decoder layers (will show segmentation features)
    - ❌ Early encoder layers (too low-level)
  - Why: Skip connections naturally highlight boundaries; decoder focuses on segmentation

- [ ] **Model in eval() mode during CAM generation**
  - Verify: `self.model.eval()` called before Grad-CAM (line 458)
  - Check: Model switched back to eval after `self.model.train()` (line 456 → 458)
  - Why: Dropout/BatchNorm differences between train/eval add noise to activations

- [ ] **Gradients zeroed before backward**
  - Verify: `model.zero_grad(set_to_none=True)` called in GradCAM class
  - File: `src/eval/grad_cam.py` - check `generate_cam()` method
  - Why: Accumulated gradients from previous samples contaminate CAM

#### Quick Verification Script

Create `scripts/verify_gradcam_config.py`:

```python
"""Verify Grad-CAM is configured correctly."""
import torch
import numpy as np
from src.inference.multi_task_predictor import MultiTaskPredictor

# Load model
predictor = MultiTaskPredictor(
    checkpoint_path="checkpoints/multitask_joint/best_model.pth"
)

# Check 1: Wrapper returns classification logits only
print("✓ Check 1: Classification wrapper")
wrapper = predictor._gradcam.model if predictor._gradcam else None
if wrapper is None:
    # Initialize Grad-CAM with realistic raw image values
    # ⚠️ CRITICAL: Use raw grayscale range [0, 255], not [0, 1]
    dummy_image = (np.random.rand(256, 256) * 255.0).astype(np.float32)
    _ = predictor.predict_with_gradcam(dummy_image)
    wrapper = predictor._gradcam.model

dummy_input = torch.randn(1, 1, 256, 256)
output = wrapper(dummy_input)
assert isinstance(output, torch.Tensor), "❌ Wrapper returns dict, not tensor!"
assert output.shape == (1, 2), f"❌ Wrong shape: {output.shape}, expected (1, 2)"
print(f"  Output shape: {output.shape} ✓")

# Check 2: Target layer is encoder bottleneck
print("\n✓ Check 2: Target layer")
target_layer = predictor._gradcam.target_layer
print(f"  Target: {target_layer}")
assert 'encoder' in str(target_layer), "❌ Target not in encoder!"
assert 'down_blocks' in str(target_layer), "❌ Target not bottleneck!"
print("  Encoder bottleneck confirmed ✓")

# Check 3: Model mode during CAM
print("\n✓ Check 3: Model eval mode")
print(f"  Model training mode: {predictor.model.training}")
assert not predictor.model.training, "❌ Model in training mode!"
print("  Model in eval() mode ✓")

# Check 4: Gradient zeroing
print("\n✓ Check 4: Gradient handling")
print("  Check src/eval/grad_cam.py for zero_grad() calls")
print("  Manual verification required")

print("\n✅ All automated checks passed!")
print("⚠️  Manually verify zero_grad() in src/eval/grad_cam.py")
```

Run verification:
```bash
python scripts/verify_gradcam_config.py
```

#### Expected Output
```
✓ Check 1: Classification wrapper
  Output shape: (1, 2) ✓

✓ Check 2: Target layer
  Target: DownBlock(...)
  Encoder bottleneck confirmed ✓

✓ Check 3: Model eval mode
  Model training mode: False
  Model in eval() mode ✓

✓ Check 4: Gradient handling
  Check src/eval/grad_cam.py for zero_grad() calls
  Manual verification required

✅ All automated checks passed!
⚠️  Manually verify zero_grad() in src/eval/grad_cam.py
```

#### If Checks Fail

**Problem:** Wrapper returns dict instead of tensor
- **Fix:** Update `ClassificationWrapper.forward()` to return only `output['cls']`

**Problem:** Target layer is in decoder or skip connections
- **Fix:** Change target to `self.model.encoder.down_blocks[-1]` (bottleneck)

**Problem:** Model in training mode
- **Fix:** Add `self.model.eval()` before `generate_cam()` call

**Problem:** No gradient zeroing
- **Fix:** Add `self.model.zero_grad(set_to_none=True)` in `GradCAM.generate_cam()`

#### Why This Matters

If Grad-CAM is misconfigured:
- You might be visualizing segmentation attention (decoder) instead of classification attention
- Skip connections naturally highlight edges → false alarm about "border artifacts"
- Training mode dropout adds noise → diffuse activations that aren't real
- Accumulated gradients → inconsistent CAMs across samples

**This 10-minute check prevents wasting hours "fixing training" while visualizing the wrong thing.**

---

### Phase 1: Normalization Unification (CRITICAL - Do First)

**Objective:** Ensure both BraTS and Kaggle use z-score normalization for multi-task training

#### Step 1.1: Generate Unified Kaggle Dataset
```bash
# Run unified preprocessing with z-score normalization
python src/data/preprocess_kaggle_unified.py \
    --input data/raw/kaggle_brain_mri \
    --output data/processed/kaggle_unified \
    --target-size 256 256 \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
```

**Expected Output:**
- `data/processed/kaggle_unified/train/` (~171 samples)
- `data/processed/kaggle_unified/val/` (~37 samples)
- `data/processed/kaggle_unified/test/` (~37 samples)
- All images z-score normalized (mean≈0, std≈1)

**Verification:**
```python
# Quick sanity check
import numpy as np
data = np.load('data/processed/kaggle_unified/train/kaggle_0000.npz')
image = data['image']
print(f"Mean: {image.mean():.4f}, Std: {image.std():.4f}")
print(f"Range: [{image.min():.4f}, {image.max():.4f}]")
# Expected: Mean≈0, Std≈1, Range includes negatives
```

⚠️ **CRITICAL: Don't rely on per-image stats alone!**

Z-score per-image guarantees mean≈0/std≈1, but distribution shapes can still differ due to:
- Clipping artifacts
- Background masking differences
- Resizing interpolation effects

**Add histogram check** to verify distribution alignment:
```python
# scripts/verify_distribution_alignment.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_batch(data_dir, n=50):
    """Load a batch of images."""
    files = list(Path(data_dir).glob("*.npz"))[:n]
    images = []
    for f in files:
        data = np.load(f)
        images.append(data['image'].flatten())
    return np.concatenate(images)

# Load batches
brats_pixels = load_batch("data/processed/brats2d_full/train")
kaggle_pixels = load_batch("data/processed/kaggle_unified/train")

# Plot histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(brats_pixels, bins=100, alpha=0.7, label='BraTS', density=True)
axes[0].hist(kaggle_pixels, bins=100, alpha=0.7, label='Kaggle', density=True)
axes[0].set_xlabel('Pixel Intensity')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_title('Pixel Intensity Distributions')

# Q-Q plot
from scipy import stats
stats.probplot(brats_pixels[:10000], dist="norm", plot=axes[1])
axes[1].set_title('BraTS Q-Q Plot (Normal)')

plt.tight_layout()
plt.savefig('outputs/distribution_alignment.png', dpi=150)
print("✓ Saved distribution comparison to outputs/distribution_alignment.png")
print("\nVisually inspect: distributions should overlap closely")
```

**Success criteria:**
- Histograms overlap significantly (similar shape)
- Both roughly Gaussian-like (Q-Q plot linear)
- No obvious bimodal or clipped distributions

#### Step 1.2: Update Stage 2 Config
**File:** `configs/stages/stage2_cls_head.yaml`

**Changes:**
```yaml
# Line 17-18: Update Kaggle paths
data:
  brats_train_dir: "data/processed/brats2d_full/train"
  brats_val_dir: "data/processed/brats2d_full/val"
  kaggle_train_dir: "data/processed/kaggle_unified/train"  # Changed from kaggle/train
  kaggle_val_dir: "data/processed/kaggle_unified/val"      # Changed from kaggle/val
```

#### Step 1.3: Update Stage 3 Config
**File:** `configs/stages/stage3_joint.yaml`

**Changes:**
```yaml
# Line 17-18: Update Kaggle paths
data:
  brats_train_dir: "data/processed/brats2d_full/train"
  brats_val_dir: "data/processed/brats2d_full/val"
  kaggle_train_dir: "data/processed/kaggle_unified/train"  # Changed from kaggle/train
  kaggle_val_dir: "data/processed/kaggle_unified/val"      # Changed from kaggle/val
```

#### Step 1.4: Verify Inference Preprocessing Matches Training

⚠️ **CRITICAL: Verify end-to-end preprocessing consistency**

**The Problem:**
If your predictor does z-score at inference but training used min-max Kaggle, that's train-test mismatch.

**Verification Steps:**

1. **Check predictor preprocessing:**
   - File: `src/inference/multi_task_predictor.py`
   - Line 234: `preprocess_image(..., normalize_method='z_score')`
   - Confirm it uses z-score (mean/std normalization)

2. **Trace full pipeline:**
```python
# scripts/verify_end_to_end_preprocessing.py
import numpy as np
import torch
import cv2
from src.inference.multi_task_predictor import MultiTaskPredictor
from src.data.multi_source_dataset import MultiSourceDataset

# Load a training sample (already z-score normalized)
train_dataset = MultiSourceDataset(
    kaggle_dir="data/processed/kaggle_unified/train",
    transform=None  # No transforms for this test
)
sample = train_dataset[0]
train_image = sample['image'].numpy()  # (1, H, W)

print("Training pipeline (from .npz):")
print(f"  Mean: {train_image.mean():.4f}")
print(f"  Std: {train_image.std():.4f}")
print(f"  Range: [{train_image.min():.4f}, {train_image.max():.4f}]")

# ⚠️ CRITICAL FIX: Load TRULY RAW image (not pre-normalized .npz)
# Option 1: Load from original JPG
raw_jpg_path = "data/raw/kaggle_brain_mri/yes/Y1.jpg"  # Example
raw_image = cv2.imread(raw_jpg_path, cv2.IMREAD_GRAYSCALE)
raw_image = cv2.resize(raw_image, (256, 256))
raw_image = raw_image.astype(np.float32)

print("\nRaw image (before any normalization):")
print(f"  Range: [{raw_image.min():.1f}, {raw_image.max():.1f}]")

# Preprocess through predictor (should apply z-score)
predictor = MultiTaskPredictor(
    checkpoint_path="checkpoints/multitask_joint/best_model.pth"
)
tensor, _ = predictor.preprocess_image(raw_image, normalize_method='z_score')
infer_image = tensor.numpy()  # (1, 1, H, W)

print("\nInference pipeline (predictor z-score):")
print(f"  Mean: {infer_image.mean():.4f}")
print(f"  Std: {infer_image.std():.4f}")
print(f"  Range: [{infer_image.min():.4f}, {infer_image.max():.4f}]")

# Compare distributions (not exact values - different source images)
print("\n=== Distribution Comparison ===")
print(f"Training mean: {train_image.mean():.4f}, Inference mean: {infer_image.mean():.4f}")
print(f"Training std:  {train_image.std():.4f}, Inference std:  {infer_image.std():.4f}")

if abs(train_image.mean()) < 0.1 and abs(infer_image.mean()) < 0.1:
    if abs(train_image.std() - 1.0) < 0.2 and abs(infer_image.std() - 1.0) < 0.2:
        print("✅ Both pipelines use z-score normalization")
    else:
        print("⚠️  Std deviation mismatch")
else:
    print("❌ MISMATCH: One pipeline not using z-score!")

# Option 2: Direct comparison using same .npz file
print("\n=== Direct Comparison (same file) ===")
npz_image = np.load("data/processed/kaggle_unified/train/kaggle_0000.npz")['image']
print(f"From .npz: mean={npz_image.mean():.4f}, std={npz_image.std():.4f}")
print("This should match training pipeline exactly.")
print("Inference pipeline should produce similar stats when given raw input.")
```

**Success Criteria:**
- Training .npz samples: mean ≈0, std ≈1 (per-image or per-slice)
- Inference-preprocessed raw inputs: mean ≈0, std ≈1
- Both distributions roughly Gaussian (histogram shapes similar)
- No "mean absolute difference" check needed - different source images!

⚠️ **Note:** MAE < 1e-5 only valid if comparing SAME image through both pipelines. Since training uses .npz (pre-normalized) and inference uses raw JPG, we compare distribution statistics, not pixel values.

#### Step 1.5: Verify Dataset Statistics
Create a diagnostic script to confirm both datasets have matching distributions:

```python
# scripts/verify_multitask_normalization.py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def check_dataset_stats(data_dir, name):
    files = list(Path(data_dir).glob("*.npz"))
    means, stds, mins, maxs = [], [], [], []
    
    for f in tqdm(files[:50], desc=f"Checking {name}"):  # Sample 50 files
        data = np.load(f)
        img = data['image']
        means.append(img.mean())
        stds.append(img.std())
        mins.append(img.min())
        maxs.append(img.max())
    
    print(f"\n{name} Statistics:")
    print(f"  Mean: {np.mean(means):.4f} ± {np.std(means):.4f}")
    print(f"  Std:  {np.mean(stds):.4f} ± {np.std(stds):.4f}")
    print(f"  Range: [{np.mean(mins):.4f}, {np.mean(maxs):.4f}]")

check_dataset_stats("data/processed/brats2d_full/train", "BraTS")
check_dataset_stats("data/processed/kaggle_unified/train", "Kaggle Unified")
```

**Expected Output:**
- Both datasets: Mean≈0, Std≈1, Range includes negatives
- Similar statistical distributions
- Run histogram check (Step 1.1) for distribution shape alignment

---

### Phase 2: Add Preprocessing Transforms (HIGH PRIORITY)

**Objective:** Apply skull masking and augmentations to multi-task training

#### Step 2.1: Create Paired Transform Wrapper (CRITICAL FIX)
**File:** `src/data/transforms.py` (add new class)

⚠️ **BUG RISK: Random transforms must sample randomness ONCE and apply to both image and mask**

Calling `transform(image)` then `transform(mask)` with random transforms will sample randomness **twice**, causing misalignment (e.g., image flipped left, mask flipped right).

**Correct Implementation:**

```python
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random

class PairedTransform:
    """
    Transform wrapper that applies same random parameters to both image and mask.
    
    Samples randomness once per transform, then applies to both tensors.
    """
    
    def __init__(self, 
                 hflip_p=0.5, 
                 vflip_p=0.5, 
                 rotation_p=0.5,
                 intensity_shift_p=0.5,
                 intensity_scale_p=0.5,
                 noise_p=0.3,
                 apply_skull_mask=True):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotation_p = rotation_p
        self.intensity_shift_p = intensity_shift_p
        self.intensity_scale_p = intensity_scale_p
        self.noise_p = noise_p
        self.apply_skull_mask = apply_skull_mask
        
        if apply_skull_mask:
            self.skull_mask = SkullBoundaryMask(threshold_percentile=1.0, kernel_size=5)
    
    def __call__(self, image, mask=None):
        """
        Apply paired transforms.
        
        Args:
            image: torch.Tensor (C, H, W)
            mask: torch.Tensor (H, W) or (1, H, W) or None
        
        Returns:
            Transformed image and mask (or just image if mask is None)
        """
        # ⚠️ CRITICAL: Ensure mask has channel dim for TF operations
        mask_was_2d = False
        if mask is not None and mask.ndim == 2:
            mask = mask.unsqueeze(0)  # (H, W) → (1, H, W)
            mask_was_2d = True
        
        # Skull masking (image only, before geometric transforms)
        if self.apply_skull_mask:
            image = self.skull_mask(image)
        
        # Geometric transforms (sample once, apply to both)
        if random.random() < self.hflip_p:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)
                mask = (mask > 0).to(mask.dtype)  # Keep discrete (handles 0/255 or int types)
        
        if random.random() < self.vflip_p:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)
                mask = (mask > 0).to(mask.dtype)  # Keep discrete (handles 0/255 or int types)
        
        if random.random() < self.rotation_p:
            # Sample rotation angle once
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                # ⚠️ CRITICAL: Use correct interpolation for each
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                if mask is not None:
                    # Masks MUST use nearest-neighbor to avoid soft/aliased edges
                    mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
                    # Ensure mask stays discrete after transform
                    mask = (mask > 0).to(mask.dtype)  # Binarize (handles 0/255 or int types)
        
        # Intensity transforms (image only)
        # ⚠️ CRITICAL: No clamping to [0,1] - data is z-score normalized!
        if random.random() < self.intensity_shift_p:
            shift = random.uniform(-0.1, 0.1)
            image = image + shift  # No clamp - z-score can have negatives
        
        if random.random() < self.intensity_scale_p:
            scale = random.uniform(0.9, 1.1)
            image = image * scale  # No clamp - preserve z-score range
        
        if random.random() < self.noise_p:
            # Increased noise std for z-score space (σ≈1)
            noise = torch.randn_like(image) * 0.1  # 0.1 instead of 0.01
            image = image + noise  # No clamp
        
        # ⚠️ CRITICAL: Restore original mask shape if needed
        if mask is not None:
            if mask_was_2d and mask.shape[0] == 1:
                mask = mask.squeeze(0)  # (1, H, W) → (H, W)
            return image, mask
        else:
            return image


def get_multitask_train_transforms():
    """Get transforms for multi-task training (handles both image and mask)."""
    return PairedTransform(
        hflip_p=0.5,
        vflip_p=0.5,
        rotation_p=0.5,
        intensity_shift_p=0.5,
        intensity_scale_p=0.5,
        noise_p=0.3,
        apply_skull_mask=True
    )


def get_multitask_val_transforms():
    """Get transforms for multi-task validation (skull masking only)."""
    return PairedTransform(
        hflip_p=0.0,
        vflip_p=0.0,
        rotation_p=0.0,
        intensity_shift_p=0.0,
        intensity_scale_p=0.0,
        noise_p=0.0,
        apply_skull_mask=True
    )
```

⚠️ **Skull Masking Caution for BraTS:**

Applying `SkullBoundaryMask` is helpful for Kaggle, but be careful with BraTS:
- If skull masking interacts differently with BraTS intensity profiles, it can become another domain cue
- The model might learn "masked = Kaggle, unmasked = BraTS" instead of tumor features

**Recommendation: Source-Aware Skull Masking**

**Option A: Conditional masking in transform**
```python
class PairedTransform:
    def __init__(self, ..., apply_skull_mask=True, skull_mask_sources=['kaggle']):
        self.apply_skull_mask = apply_skull_mask
        self.skull_mask_sources = skull_mask_sources  # Which sources to mask
        if apply_skull_mask:
            self.skull_mask = SkullBoundaryMask(threshold_percentile=1.0, kernel_size=5)
    
    def __call__(self, image, mask=None, source='kaggle'):
        # Skull masking (source-aware)
        if self.apply_skull_mask and source in self.skull_mask_sources:
            image = self.skull_mask(image)
        # ... rest of transforms
```

**Option B: Separate transforms per source**
```python
# In training script
kaggle_transform = PairedTransform(apply_skull_mask=True)   # Mask ON for Kaggle
brats_transform = PairedTransform(apply_skull_mask=False)   # Mask OFF for BraTS

# In dataset __getitem__
if source == 'kaggle':
    image, mask = kaggle_transform(image, mask)
else:
    image, mask = brats_transform(image, mask)
```

**Validation:**
- After Phase 2, generate per-source Grad-CAMs (BraTS-only and Kaggle-only batches)
- Verify both show lesion-focused attention, not dataset artifacts
- If BraTS CAMs still show edge artifacts, skull masking isn't the issue

#### Step 2.2: Update Multi-Source Dataset
**File:** `src/data/multi_source_dataset.py`

⚠️ **CRITICAL: Pass `source` to transform for source-aware masking compatibility**

**Changes at lines 152-161:**
```python
# Apply transforms if provided
if self.transform is not None:
    # Check if transform accepts 'source' parameter (for source-aware masking)
    import inspect
    transform_sig = inspect.signature(self.transform.__call__)
    accepts_source = 'source' in transform_sig.parameters
    
    if output['has_mask']:
        # Transform both image and mask (for BraTS)
        if accepts_source:
            image, mask = self.transform(image, output['mask'], source=source)
        else:
            image, mask = self.transform(image, output['mask'])
        output['mask'] = mask
    else:
        # Transform image only (for Kaggle - no mask)
        if accepts_source:
            image = self.transform(image, mask=None, source=source)
        else:
            image = self.transform(image, mask=None)

output['image'] = image
```

**Why This Matters:**
- **Without source passing:** Source-aware masking silently uses default ('kaggle'), potentially masking everything
- **With inspection:** Works with both simple `PairedTransform` and source-aware version
- **Backward compatible:** Doesn't break if you don't use source-aware masking

**Alternative (Simpler, No Inspection):**
```python
# Apply transforms if provided
if self.transform is not None:
    if output['has_mask']:
        # Try source-aware first, fall back to simple
        try:
            image, mask = self.transform(image, output['mask'], source=source)
        except TypeError:
            image, mask = self.transform(image, output['mask'])
        output['mask'] = mask
    else:
        try:
            image = self.transform(image, mask=None, source=source)
        except TypeError:
            image = self.transform(image, mask=None)

output['image'] = image
```

#### Step 2.3: Update Stage 2 Training Script
**File:** `src/training/train_multitask_cls_head.py`

**Changes at lines 256-265:**
```python
# Create datasets
print("\n=== Loading Datasets ===")

# Import transforms
from src.data.transforms import get_multitask_train_transforms, get_multitask_val_transforms

train_dataset = MultiSourceDataset(
    brats_dir=config['data'].get('brats_train_dir'),
    kaggle_dir=config['data'].get('kaggle_train_dir'),
    transform=get_multitask_train_transforms(),  # Changed from None
)
val_dataset = MultiSourceDataset(
    brats_dir=config['data'].get('brats_val_dir'),
    kaggle_dir=config['data'].get('kaggle_val_dir'),
    transform=get_multitask_val_transforms(),  # Changed from None
)
```

#### Step 2.4: Update Stage 3 Training Script
**File:** `src/training/train_multitask_joint.py`

Apply similar changes as Step 2.3 to add transforms.

---

### Phase 3: Dataset Mixing & Class Balance (CRITICAL - Elevate Priority)

**Objective:** Fix severe class imbalance caused by BraTS tumor-only slices

⚠️ **THE BIG PROBLEM: BraTS Tumor-Only Bias**

If you saved only tumor slices from BraTS (`save_all_slices=False`), your mixed dataset is:
- **BraTS:** Almost all class-1 (tumor)
- **Kaggle:** Both classes but tiny (~245 images)

**Impact:**
- Classifier learns "always tumor" bias, OR
- Learns "BraTS vs Kaggle" artifacts that correlate with label
- This can dominate classification behavior more than normalization!

**Two Clean Options (Pick One Early):**

#### Option A: Stage 2 with Kaggle-Only (RECOMMENDED for quick diagnostic)

**Why:** Encoder is frozen in Stage 2 anyway. Train classification head on clean, balanced Kaggle first.

**Benefits:**
- Confirms head can learn lesion cues without dataset artifacts
- Fast iteration (small dataset)
- Clean signal for Grad-CAM validation
- Add BraTS classification later (Stage 3) once negatives exist

**Implementation:**

⚠️ **CRITICAL: Ensure MultiSourceDataset handles `brats_dir=None`**

Before using this config, verify `src/data/multi_source_dataset.py` has guards:
```python
# In MultiSourceDataset.__init__
if brats_dir is not None:
    brats_path = Path(brats_dir)
    if brats_path.exists():
        # ... load BraTS files
```

**Option A: Use MultiSourceDataset with null BraTS** (if guards exist)
```yaml
# configs/stages/stage2_cls_head.yaml
data:
  brats_train_dir: null  # Disable BraTS for Stage 2
  brats_val_dir: null
  kaggle_train_dir: "data/processed/kaggle_unified/train"
  kaggle_val_dir: "data/processed/kaggle_unified/val"
```

**Option B: Use KaggleOnlyDataset directly** (cleaner diagnostic)
```python
# In train_multitask_cls_head.py
from src.data.multi_source_dataset import KaggleOnlyDataset

train_dataset = KaggleOnlyDataset(
    data_dir=config['data']['kaggle_train_dir'],
    transform=get_multitask_train_transforms()
)
val_dataset = KaggleOnlyDataset(
    data_dir=config['data']['kaggle_val_dir'],
    transform=get_multitask_val_transforms()
)
```

**Recommendation:** Use Option B for Stage 2 Kaggle-only diagnostic - it's cleaner and guaranteed to work.

**Expected Result:**
- Val accuracy > 90% on Kaggle
- Grad-CAMs show lesion focus
- Proves classification head architecture is sound

#### Option B: Regenerate BraTS with Negatives + Controlled Sampling

**Step 1: Regenerate BraTS with all slices**
```bash
python src/data/preprocess_brats_2d.py \
    --input data/raw/brats2020/MICCAI_BraTS2020_TrainingData \
    --output data/processed/brats2d_full_with_negatives \
    --save-all-slices \
    --min-tumor-pixels 10
```

**Step 2: Aggressive negative subsampling**

⚠️ **CRITICAL: Preserve patient-level splits to avoid data leakage!**

Don't dump all slices into training naively—you'll drown in negatives. Also, subsample **per split** to maintain train/val/test separation.

```python
# src/data/subsample_brats_negatives.py
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

def subsample_negatives_per_split(input_dir, output_dir, negatives_per_patient=5):
    """
    Keep all tumor slices, subsample negatives.
    Operates on each split (train/val/test) separately to prevent leakage.
    
    Args:
        input_dir: BraTS directory with train/val/test subdirs
        output_dir: Output directory (will create train/val/test)
        negatives_per_patient: Max negatives per patient
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Process each split separately
    for split in ['train', 'val', 'test']:
        split_input = input_path / split
        split_output = output_path / split
        
        if not split_input.exists():
            print(f"⚠️  Split {split} not found, skipping")
            continue
        
        split_output.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== Processing {split} split ===")
        
        # Group by patient
        from collections import defaultdict
        patient_files = defaultdict(list)
        
        for f in split_input.glob("*.npz"):
            patient_id = f.stem.split('_slice_')[0]
            patient_files[patient_id].append(f)
        
        print(f"Found {len(patient_files)} patients in {split}")
        
        kept_tumor = 0
        kept_no_tumor = 0
        
        for patient_id, files in tqdm(patient_files.items(), desc=f"Subsampling {split}"):
            tumor_files = []
            no_tumor_files = []
            
            for f in files:
                data = np.load(f, allow_pickle=True)
                mask = data['mask']
                if np.sum(mask > 0) >= 10:
                    tumor_files.append(f)
                else:
                    no_tumor_files.append(f)
            
            # Keep all tumor slices
            for f in tumor_files:
                shutil.copy(f, split_output / f.name)
                kept_tumor += 1
            
            # Subsample negatives
            if len(no_tumor_files) > negatives_per_patient:
                no_tumor_files = np.random.choice(
                    no_tumor_files, 
                    negatives_per_patient, 
                    replace=False
                )
            
            for f in no_tumor_files:
                shutil.copy(f, split_output / f.name)
                kept_no_tumor += 1
        
        print(f"  {split}: Kept {kept_tumor} tumor, {kept_no_tumor} no-tumor")
        print(f"  Ratio: {kept_tumor/(kept_tumor+kept_no_tumor):.2%} tumor")

if __name__ == "__main__":
    subsample_negatives_per_split(
        "data/processed/brats2d_full_with_negatives",
        "data/processed/brats2d_balanced",
        negatives_per_patient=5
    )
```

**Alternative: Subsample before splitting**
```bash
# Option 1: Subsample per split (recommended - preserves existing splits)
python src/data/subsample_brats_negatives.py

# Option 2: Subsample flat, then split at patient-level
python src/data/preprocess_brats_2d.py --save-all-slices
python src/data/subsample_brats_negatives.py  # On flat directory
python src/data/split_brats.py --input data/processed/brats2d_balanced
```

**Step 3: Update configs to use balanced BraTS**
```yaml
data:
  brats_train_dir: "data/processed/brats2d_balanced/train"
  brats_val_dir: "data/processed/brats2d_balanced/val"
```

---

#### Sampler Implementation (If Using Mixed Dataset)

#### Step 3.1: Weighted Random Sampler (If Using Mixed Dataset)
**File:** `src/training/train_multitask_joint.py`

⚠️ **CRITICAL: Precompute metadata to avoid slow startup**

Don't compute sampler weights by repeatedly loading `.npz` files—this makes training startup painfully slow.

**Correct Implementation:**

```python
from torch.utils.data import WeightedRandomSampler
import numpy as np

def create_balanced_sampler(dataset):
    """
    Create sampler to balance BraTS and Kaggle samples.
    
    Precomputes source and label metadata during dataset init.
    """
    # Precompute metadata once (do this in dataset __init__)
    if not hasattr(dataset, '_sample_metadata'):
        print("Precomputing sample metadata for sampler...")
        dataset._sample_metadata = []
        for sample_info in dataset.samples:
            data = np.load(sample_info['path'], allow_pickle=True)
            source = sample_info['source']
            
            if source == 'brats':
                mask = data['mask']
                label = 1 if np.sum(mask > 0) >= 10 else 0
            else:
                label = int(data['label'])
            
            dataset._sample_metadata.append({
                'source': source,
                'label': label
            })
    
    # Count samples per source
    n_brats = sum(1 for m in dataset._sample_metadata if m['source'] == 'brats')
    n_kaggle = sum(1 for m in dataset._sample_metadata if m['source'] == 'kaggle')
    
    # Assign weights: higher weight to underrepresented source
    weights = []
    for meta in dataset._sample_metadata:
        if meta['source'] == 'brats':
            weights.append(1.0 / n_brats)
        else:
            weights.append(1.0 / n_kaggle)
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)

# Usage in dataloader creation:
train_sampler = create_balanced_sampler(train_dataset)
train_loader = DataLoader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    sampler=train_sampler,  # Use sampler instead of shuffle
    num_workers=0,
    pin_memory=False,
    collate_fn=custom_collate_fn,
)
```

**Option B: Dual DataLoader with Manual Mixing** (More control)
```python
from src.data.multi_source_dataset import BraTSOnlyDataset, KaggleOnlyDataset

# Create separate datasets
brats_train = BraTSOnlyDataset(
    data_dir=config['data']['brats_train_dir'],
    transform=get_multitask_train_transforms(),
)
kaggle_train = KaggleOnlyDataset(
    data_dir=config['data']['kaggle_train_dir'],
    transform=get_multitask_train_transforms(),
)

# Create separate loaders
brats_loader = DataLoader(brats_train, batch_size=int(batch_size * 0.3), shuffle=True)
kaggle_loader = DataLoader(kaggle_train, batch_size=int(batch_size * 0.7), shuffle=True)

# Training loop: manually combine batches
brats_iter = iter(brats_loader)
kaggle_iter = iter(kaggle_loader)

for epoch in range(num_epochs):
    for _ in range(min(len(brats_loader), len(kaggle_loader))):
        # Get batches from each source
        try:
            brats_batch = next(brats_iter)
        except StopIteration:
            brats_iter = iter(brats_loader)
            brats_batch = next(brats_iter)
        
        try:
            kaggle_batch = next(kaggle_iter)
        except StopIteration:
            kaggle_iter = iter(kaggle_loader)
            kaggle_batch = next(kaggle_iter)
        
        # Combine batches
        combined_batch = {
            'image': torch.cat([brats_batch['image'], kaggle_batch['image']], dim=0),
            'cls': torch.cat([brats_batch['cls'], kaggle_batch['cls']], dim=0),
            # ... combine other fields
        }
        
        # Train on combined batch
        # ...
```

**Recommendation:** Start with unified metadata approach below (combines source + class balancing in one pass).

#### Step 3.2: Unified Metadata Precompute (Source + Class Balance)

⚠️ **CRITICAL: Do metadata loading ONCE, not per-sampler function**

```python
# Add to MultiSourceDataset.__init__ (one-time precompute)
def _precompute_metadata(self):
    """Precompute source and label metadata for fast sampler creation."""
    import pickle
    
    # Try to load cached metadata
    cache_file = Path(self.cache_dir) / "metadata_cache.pkl" if hasattr(self, 'cache_dir') else None
    if cache_file and cache_file.exists():
        print(f"Loading cached metadata from {cache_file}")
        with open(cache_file, 'rb') as f:
            self._sample_metadata = pickle.load(f)
        print(f"  Loaded metadata for {len(self._sample_metadata)} samples")
        return
    
    print("Precomputing sample metadata for balanced sampling...")
    self._sample_metadata = []
    
    for sample_info in self.samples:
        data = np.load(sample_info['path'], allow_pickle=True)
        source = sample_info['source']
        
        if source == 'brats':
            mask = data['mask']
            label = 1 if np.sum(mask > 0) >= self.min_tumor_pixels else 0
        else:
            label = int(data['label'])
        
        self._sample_metadata.append({
            'source': source,
            'label': label
        })
    
    print(f"  Precomputed metadata for {len(self._sample_metadata)} samples")
    
    # Cache to disk for future runs
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(self._sample_metadata, f)
        print(f"  Cached metadata to {cache_file}")

# Call in __init__ after loading samples
if len(self.samples) > 0:
    # Set cache directory (e.g., same as data dir)
    if hasattr(self, 'brats_dir') and self.brats_dir:
        self.cache_dir = Path(self.brats_dir).parent / '.cache'
    elif hasattr(self, 'kaggle_dir') and self.kaggle_dir:
        self.cache_dir = Path(self.kaggle_dir).parent / '.cache'
    else:
        self.cache_dir = None
    
    self._precompute_metadata()


# Unified sampler creation (uses precomputed metadata)
def create_balanced_sampler(dataset, balance_source=True, balance_class=True):
    """
    Create sampler balancing source and/or class.
    
    Uses precomputed metadata - no file I/O!
    """
    if not hasattr(dataset, '_sample_metadata'):
        raise ValueError("Dataset must have _sample_metadata. Call _precompute_metadata() first.")
    
    from collections import Counter
    
    # Count frequencies
    sources = [m['source'] for m in dataset._sample_metadata]
    labels = [m['label'] for m in dataset._sample_metadata]
    
    source_counts = Counter(sources)
    class_counts = Counter(labels)
    
    print(f"\nDataset composition:")
    print(f"  Sources: {dict(source_counts)}")
    print(f"  Classes: {dict(class_counts)}")
    
    # Compute weights
    weights = []
    for meta in dataset._sample_metadata:
        weight = 1.0
        
        if balance_source:
            weight *= (1.0 / source_counts[meta['source']])
        
        if balance_class:
            weight *= (1.0 / class_counts[meta['label']])
        
        weights.append(weight)
    
    # Normalize
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    print(f"  Sampler: balance_source={balance_source}, balance_class={balance_class}")
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)
```

---

### Phase 4: Wire Loss Weights from Config (LOW PRIORITY)

**Objective:** Ensure `lambda_cls` and `lambda_seg` flow from YAML to loss function

⚠️ **Add Runtime Guardrail: Print Resolved Loss Config**

To prevent silent fallback to defaults, always print the resolved loss configuration at training start:

```python
# In training script, after creating loss
print("\n=== Loss Configuration ===")
print(f"  Segmentation loss: {seg_loss_type}")
print(f"  Classification loss: {cls_loss_type}")
print(f"  Lambda seg: {lambda_seg}")
print(f"  Lambda cls: {lambda_cls}")
print(f"  Dice weight: {dice_weight}")
print(f"  BCE weight: {bce_weight}")
print("=" * 50)
```

This ensures you never silently fall back to defaults again.

#### Step 4.1: Flatten Config Structure
**File:** `configs/stages/stage3_joint.yaml`

**Option A: Flatten YAML (simpler - RECOMMENDED)**
```yaml
# Replace lines 34-43 with flat structure:
loss:
  seg_loss_type: "dice_bce"
  seg_dice_weight: 0.6
  seg_bce_weight: 0.4
  cls_loss_type: "cross_entropy"
  cls_label_smoothing: 0.1
  lambda_seg: 1.0
  lambda_cls: 0.5
```

**Option B: Update Code to Parse Nested Config (more flexible)**
**File:** `src/training/train_multitask_joint.py`

```python
# Parse nested loss config
loss_config = config['loss']

# Extract segmentation loss params
seg_loss_config = {
    'seg_loss_type': loss_config['seg_loss']['name'],
    'dice_weight': loss_config['seg_loss'].get('dice_weight', 0.5),
    'bce_weight': loss_config['seg_loss'].get('bce_weight', 0.5),
}

# Extract classification loss params
cls_loss_config = {
    'cls_loss_type': loss_config['cls_loss']['name'],
    'label_smoothing': loss_config['cls_loss'].get('label_smoothing', 0.0),
}

# Extract lambda weights
lambda_seg = loss_config.get('lambda_seg', 1.0)
lambda_cls = loss_config.get('lambda_cls', 1.0)

# Create loss with proper params
from src.training.multi_task_losses import create_multi_task_loss

criterion = create_multi_task_loss(
    seg_loss_type=seg_loss_config['seg_loss_type'],
    cls_loss_type=cls_loss_config['cls_loss_type'],
    lambda_seg=lambda_seg,
    lambda_cls=lambda_cls,
    dice_weight=seg_loss_config['dice_weight'],
    bce_weight=seg_loss_config['bce_weight'],
    label_smoothing=cls_loss_config['label_smoothing'],
)
```

**Recommendation:** Use Option A (flatten YAML) for immediate fix, then refactor to Option B later for better config organization.

**Add runtime print** (see above) to verify loss weights are loaded correctly.

---

## Validation & Testing Plan

### Quick Validation (2-5 epochs, ~10 minutes)

**Purpose:** Verify fixes work before full training

#### Test 1: Normalization Fix Validation
```bash
# After Phase 1 changes
python scripts/verify_multitask_normalization.py

# Expected: Both datasets show mean≈0, std≈1
```

#### Test 2: Quick Stage 2 Run
```bash
# Train classification head for 5 epochs
python src/training/train_multitask_cls_head.py \
    --config configs/stages/stage2_cls_head.yaml \
    --encoder-init checkpoints/multitask_seg_warmup/best_model.pth \
    --checkpoint-dir checkpoints/multitask_cls_head_fixed

# Monitor validation accuracy - should improve significantly
```

**Success Criteria:**
- Val accuracy > 85% after 5 epochs (vs previous lower performance)
- Training loss decreases smoothly
- No NaN or divergence

#### Test 3: Grad-CAM Comparison
```bash
# Generate Grad-CAMs from fixed model
python scripts/generate_multitask_gradcam.py \
    --checkpoint checkpoints/multitask_cls_head_fixed/best_model.pth \
    --output-dir outputs/gradcam_fixed \
    --num-samples 10
```

**Success Criteria:**
- Grad-CAMs show more focused attention on tumor regions
- Reduced activation on skull boundaries and corners
- Visual similarity to standalone classifier CAMs

#### Test 4: Dataset Statistics
```python
# Check class balance in training batches
from src.data.multi_source_dataset import MultiSourceDataset

dataset = MultiSourceDataset(
    brats_dir="data/processed/brats2d_full/train",
    kaggle_dir="data/processed/kaggle_unified/train",
)

stats = dataset.get_statistics()
print(stats)

# Expected output:
# - brats_ratio: ~0.3-0.4 (if using sampler)
# - tumor_ratio: ~0.5-0.6 (balanced)
# - kaggle_ratio: ~0.6-0.7
```

---

### Full Training & Evaluation

#### Stage 2: Classification Head Training
```bash
# Full training with fixed pipeline
python src/training/train_multitask_cls_head.py \
    --config configs/stages/stage2_cls_head.yaml \
    --encoder-init checkpoints/multitask_seg_warmup/best_model.pth \
    --checkpoint-dir checkpoints/multitask_cls_head_v2

# Expected: 30-50 epochs, ~1-2 hours
```

**Target Metrics:**
- Val Accuracy: > 90%
- Val ROC-AUC: > 0.92
- Comparable to standalone classifier

#### Stage 3: Joint Fine-Tuning
```bash
# Joint training with all fixes
python src/training/train_multitask_joint.py \
    --config configs/stages/stage3_joint.yaml \
    --init-checkpoint checkpoints/multitask_cls_head_v2/best_model.pth \
    --checkpoint-dir checkpoints/multitask_joint_v2

# Expected: 50 epochs, ~2-3 hours
```

**Target Metrics:**
- Val Accuracy: > 91%
- Val Dice: > 0.75
- Combined metric: > 0.83

#### Comprehensive Evaluation
```bash
# Evaluate on test set
python src/eval/evaluate_multitask.py \
    --checkpoint checkpoints/multitask_joint_v2/best_model.pth \
    --brats-test-dir data/processed/brats2d_full/test \
    --kaggle-test-dir data/processed/kaggle_unified/test \
    --output-dir results/multitask_v2

# Compare to standalone models
python scripts/compare_all_phases.py \
    --seg-checkpoint checkpoints/seg2d/best_model.pth \
    --cls-checkpoint checkpoints/classifier/best_model.pth \
    --multitask-checkpoint checkpoints/multitask_joint_v2/best_model.pth \
    --output-dir results/phase_comparison_v2
```

**Success Criteria:**
- Multi-task classification accuracy within 2% of standalone
- Grad-CAMs show focused tumor attention
- Segmentation Dice maintained or improved
- Combined metric > 0.83

---

## Expected Improvements

### Before Fixes (Current State)
- Multi-task classification: Poor accuracy, diffuse Grad-CAMs
- Model relies on spurious features (borders, global intensity)
- Inconsistent input distributions
- Class imbalance issues

### After Phase 1 (Normalization Fix)
- **Expected improvement:** +15-25% classification accuracy
- Consistent z-score normalization across all data
- Model can learn tumor-specific features
- Grad-CAMs should become more focused

### After Phase 2 (Transforms Added)
- **Expected improvement:** +5-10% additional accuracy
- Skull boundary artifacts removed
- Better generalization from augmentations
- Grad-CAMs localize to tumor regions

### After Phase 3 (Dataset Mixing)
- **Expected improvement:** +3-5% accuracy, better balance
- Reduced class bias
- More robust predictions on negative samples
- Improved calibration

### Final Target Performance
- Classification Accuracy: **90-93%** (vs 94.59% standalone)
- ROC-AUC: **0.92-0.95**
- Segmentation Dice: **0.75-0.78**
- Combined Metric: **0.83-0.86**
- Grad-CAM: Focused on tumor, minimal border activation

---

## Implementation Priority Guide

### Must-Fix (Correctness Issues)
These will cause training failures or silent bugs:

1. ✅ **Z-score normalization unification** (Phase 1) - Train/test mismatch
2. ✅ **No clamping to [0,1] in transforms** - Destroys z-score distributions
3. ✅ **InterpolationMode import fix** - Will cause import error
4. ✅ **Rotation interpolation** (BILINEAR for image, NEAREST for mask) - Prevents aliased masks
5. ✅ **BraTS split preservation** - Prevents data leakage
6. ✅ **Config merge workflow** (stages → merge → final) - Ensures configs are used correctly
7. ✅ **Dataset → Transform source passing** - Prevents silent default behavior in source-aware masking
   - Use inspection or try/except to handle both simple and source-aware transforms
   - Without this, source-aware masking silently uses default ('kaggle')

### Strongly Recommended (Prevents Subtle Bugs)
Cheap to implement, avoids edge cases:

1. ✅ **Mask binarization after transforms** - Use `mask > 0` instead of `mask > 0.5`
   - Handles 0/255 or int mask types safely
   - Tiny change, big safety improvement
2. ✅ **Mask shape safety guards** - Ensure (1, H, W) for TF operations
   - `unsqueeze(0)` at start if 2D, `squeeze(0)` at end
   - Prevents "shape mismatch" errors mid-training
   - Works across torchvision versions
3. ✅ **Metadata disk caching** - Speeds up reruns (matters for large BraTS)
4. ✅ **Per-source validation metrics** - Reveals dataset shortcuts immediately
5. ✅ **Runtime loss config print** - Prevents silent fallback to defaults

### Conditional (Only If Using That Feature)

1. **Source-aware skull masking** - Only if you want different masking per dataset
   - **If NOT using:** Apply same masking to both (simpler, still works)
   - **If using:** Must pass `source` argument through dataset → transform
   - **Default behavior:** If you don't implement source-passing, transform will use default (mask everything or nothing)

2. **Grad-CAM verification script** - Only affects 10-min sanity check
   - **If script fails:** You'll debug verification, not training
   - **Doesn't affect:** Actual training or model performance
   - **Critical fix applied:** Dummy image now uses raw grayscale range [0, 255]
   - **Why it matters:** Predictor expects raw input and normalizes internally
   - **Old bug:** `np.random.rand(256,256)` creates [0,1] values → wrong input distribution
   - **Fixed:** `np.random.rand(256,256) * 255.0` creates [0,255] values → correct

### Optional (Nice-to-Have)

1. **Histogram distribution checks** - Adds confidence but not required
2. **Kaggle-only Stage 2 diagnostic** - Isolates variables but can skip if confident
3. **End-to-end preprocessing verification** - Good for paranoia, not strictly needed

---

## Recommended Execution Sequence (Bulletproof Approach)

**Goal:** Isolate variables and get to clean "signal" faster

This sequence is optimized based on feedback to minimize risk and maximize learning at each step.

---

### Step 0: Pre-Flight Check (10 minutes)
- [ ] **Grad-CAM Sanity Check:** Run verification script
- [ ] Confirm wrapper returns classification logits only
- [ ] Confirm target layer is encoder bottleneck
- [ ] Confirm model in eval() mode
- [ ] Confirm gradients zeroed

**Win Condition:** Grad-CAMs are correctly configured

---

### Step 1: Phase 1 - Normalization Unification (30 minutes)
- [ ] **1.1:** Generate unified Kaggle z-score dataset
- [ ] **1.2:** Update Stage 2 config to use `kaggle_unified`
- [ ] **1.3:** Run histogram check (distribution alignment)
- [ ] **1.4:** Verify end-to-end preprocessing (training == inference)
- [ ] **1.5:** Verify dataset statistics

**Win Condition:** Both datasets z-score normalized, distributions aligned, train==inference

---

### Step 2: Stage 2 Quick Run with Kaggle-Only (20 minutes)

**Why:** Encoder frozen anyway. Train head on clean, balanced Kaggle first.

- [ ] **2.1:** Edit `configs/stages/stage2_cls_head.yaml` (disable BraTS)
- [ ] **2.2:** Run `python scripts/utils/merge_configs.py --all` to regenerate final configs
- [ ] **2.3:** Train using `configs/final/stage2_cls_head.yaml` for 5-10 epochs
- [ ] **2.4:** Check val accuracy (target: > 85%)
- [ ] **2.5:** Generate Grad-CAMs and per-source validation report

**Win Condition:** 
- Val accuracy > 85% on Kaggle
- Grad-CAMs show lesion focus (not borders)
- Proves classification head can learn tumor cues

**If this fails:** Problem is NOT dataset mixing—investigate architecture or Kaggle data quality

**Per-Source Validation Report:**
```python
# Add to validation loop in train_multitask_cls_head.py
kaggle_preds, kaggle_labels = [], []
brats_preds, brats_labels = [], []

for batch in val_loader:
    # ... forward pass ...
    for i, source in enumerate(batch['source']):
        if source == 'kaggle':
            kaggle_preds.append(pred[i])
            kaggle_labels.append(label[i])
        else:
            brats_preds.append(pred[i])
            brats_labels.append(label[i])

# Compute per-source metrics
from sklearn.metrics import accuracy_score, roc_auc_score

if len(kaggle_preds) > 0:
    kaggle_acc = accuracy_score(kaggle_labels, kaggle_preds)
    kaggle_auc = roc_auc_score(kaggle_labels, kaggle_probs)
    print(f"\nKaggle Val: Acc={kaggle_acc:.4f}, AUC={kaggle_auc:.4f}")

if len(brats_preds) > 0:
    brats_acc = accuracy_score(brats_labels, brats_preds)
    brats_auc = roc_auc_score(brats_labels, brats_probs)
    print(f"BraTS Val:  Acc={brats_acc:.4f}, AUC={brats_auc:.4f}")

# If metrics differ significantly, model learned dataset artifacts!
if abs(kaggle_acc - brats_acc) > 0.1:
    print("⚠️  WARNING: Large accuracy gap between sources!")
    print("    Model may be learning BraTS-vs-Kaggle shortcuts.")
```

---

### Step 3: Add BraTS Classification (Choose Path)

**Path A: Use existing BraTS (if negatives exist)**
- [ ] Re-enable BraTS in Stage 2 config
- [ ] Implement WeightedRandomSampler (precompute metadata)
- [ ] Run Stage 2 with mixed dataset
- [ ] Verify Grad-CAMs on both sources

**Path B: Regenerate BraTS with negatives (if tumor-only)**
- [ ] Run `preprocess_brats_2d.py --save-all-slices`
- [ ] Run subsampling script (5 negatives per patient)
- [ ] Update config to use balanced BraTS
- [ ] Implement sampler and run Stage 2

**Win Condition:**
- Val accuracy maintained or improved
- Grad-CAMs lesion-focused on both BraTS and Kaggle
- No "dataset artifact" learning

---

### Step 4: Add Paired Transforms (1 hour)
- [ ] **4.1:** Implement `PairedTransform` class
- [ ] **4.2:** Update `MultiSourceDataset` to use paired transforms
- [ ] **4.3:** Update Stage 2/3 training scripts
- [ ] **4.4:** Run quick Stage 2 test (5 epochs)
- [ ] **4.5:** Generate per-source Grad-CAMs

**Win Condition:**
- Transforms don't break mask alignment
- Grad-CAMs remain lesion-focused
- Slight accuracy improvement from augmentation

**Caution:** Check if skull masking creates BraTS/Kaggle domain cue

---

### Step 5: Tune Mixing + Loss Weights (30 minutes)
- [ ] **5.1:** Flatten loss config YAML
- [ ] **5.2:** Add runtime loss config print
- [ ] **5.3:** Adjust `lambda_cls` if needed (start with 0.5)
- [ ] **5.4:** Run Stage 3 quick test

**Win Condition:**
- Loss weights correctly loaded (verified by print)
- Combined metric improves

---

### Step 6: Full Training (4-6 hours)
- [ ] **6.1:** Full Stage 2 (30-50 epochs)
- [ ] **6.2:** Full Stage 3 (50 epochs)
- [ ] **6.3:** Comprehensive evaluation
- [ ] **6.4:** Phase comparison report

**Win Condition:**
- Classification accuracy > 90%
- ROC-AUC > 0.92
- Segmentation Dice > 0.75
- Combined metric > 0.83
- Grad-CAMs lesion-focused

---

## Original Implementation Timeline (For Reference)

### Pre-Flight Check (Day 1 - Start Here) - Grad-CAM Verification
- [ ] **Pre-Phase 1:** Run Grad-CAM sanity check script (10 min)
- [ ] **Verify:** Classification wrapper returns logits only
- [ ] **Verify:** Target layer is encoder bottleneck
- [ ] **Verify:** Model in eval() mode
- [ ] **Verify:** Gradients zeroed before backward

**Total Time:** ~10 minutes  
**Expected Result:** Confidence that Grad-CAMs are showing real attention patterns, not artifacts

### Immediate (Day 1) - Critical Fixes
- [ ] **Phase 1.1:** Generate unified Kaggle dataset (15 min)
- [ ] **Phase 1.2-1.3:** Update Stage 2/3 configs in `configs/stages/` (5 min)
- [ ] **Phase 1.3b:** Run `python scripts/utils/merge_configs.py --all` (1 min)
- [ ] **Phase 1.4-1.5:** Verify normalization and inference consistency (15 min)
- [ ] **Quick Test:** 5-epoch Stage 2 Kaggle-only using `configs/final/stage2_*.yaml` (15 min)
- [ ] **Validation:** Check metrics, per-source report, and Grad-CAMs (10 min)

**Total Time:** ~1 hour  
**Expected Result:** Significant accuracy improvement visible, lesion-focused CAMs

**Config Usage Note:**
- Edit: `configs/stages/stage2_cls_head.yaml`, `stage3_joint.yaml`
- Merge: `python scripts/utils/merge_configs.py --all`
- Train with: `configs/final/stage2_cls_head.yaml`, `configs/final/stage3_joint.yaml`

### Short-term (Day 1-2) - High Priority
- [ ] **Phase 2.1:** Create segmentation transform wrapper (30 min)
- [ ] **Phase 2.2:** Update multi-source dataset (15 min)
- [ ] **Phase 2.3-2.4:** Add transforms to training scripts (20 min)
- [ ] **Quick Test:** 5-epoch Stage 2 run with transforms (15 min)
- [ ] **Validation:** Compare Grad-CAMs (10 min)

**Total Time:** ~1.5 hours  
**Expected Result:** More focused Grad-CAMs, better generalization

### Medium-term (Day 2-3) - Full Training
- [ ] **Full Stage 2:** Train classification head (1-2 hours)
- [ ] **Full Stage 3:** Joint fine-tuning (2-3 hours)
- [ ] **Evaluation:** Comprehensive test set evaluation (30 min)
- [ ] **Comparison:** Generate phase comparison report (20 min)

**Total Time:** ~4-6 hours  
**Expected Result:** Production-ready multi-task model

### Optional (Day 3-4) - Refinements
- [ ] **Phase 3:** Implement dataset mixing (1-2 hours)
- [ ] **Phase 4:** Wire loss weights (30 min)
- [ ] **Retrain:** Full pipeline with all optimizations (4-6 hours)
- [ ] **Final Eval:** Complete evaluation suite (1 hour)

**Total Time:** ~7-10 hours  
**Expected Result:** Optimal multi-task performance

---

## Risk Mitigation

### Risk 1: Unified Kaggle Dataset Generation Fails
**Mitigation:**
- Verify raw Kaggle data exists: `data/raw/kaggle_brain_mri/yes/` and `no/`
- Check preprocessing script for errors
- Fallback: Manually normalize existing Kaggle data with z-score

### Risk 2: Transforms Break Mask Alignment
**Mitigation:**
- Test transform wrapper on single sample first
- Visualize transformed image-mask pairs
- Ensure geometric transforms apply identically to both

### Risk 3: Dataset Mixing Causes Training Instability
**Mitigation:**
- Start with simple WeightedRandomSampler
- Monitor loss curves for divergence
- Adjust sampling weights if needed
- Fallback: Use original concatenation with balanced Kaggle data

### Risk 4: Performance Doesn't Improve After Phase 1
**Mitigation:**
- Double-check normalization statistics
- Verify configs point to correct directories
- Test inference pipeline normalization
- Consider additional debugging with per-source metrics

---

## Success Metrics

### Quantitative Metrics
- [ ] Classification accuracy > 90%
- [ ] ROC-AUC > 0.92
- [ ] Segmentation Dice > 0.75
- [ ] Combined metric > 0.83
- [ ] Within 3% of standalone classifier accuracy

### Qualitative Metrics
- [ ] Grad-CAMs show focused tumor attention
- [ ] Minimal activation on skull boundaries
- [ ] Consistent predictions across similar samples
- [ ] Improved calibration (ECE < 0.05)

### Code Quality Metrics
- [ ] All configs point to correct data paths
- [ ] Transforms applied consistently
- [ ] Dataset mixing implemented and tested
- [ ] Loss weights wired from config
- [ ] Documentation updated

---

## Rollback Plan

If fixes cause regressions or training failures:

1. **Immediate Rollback:**
   - Revert config changes: `git checkout configs/stages/`
   - Use original Kaggle data path
   - Remove transforms: set `transform=None`

2. **Partial Rollback:**
   - Keep normalization fix (Phase 1)
   - Remove transforms (Phase 2) if causing issues
   - Skip dataset mixing (Phase 3) if unstable

3. **Debugging Steps:**
   - Check data loading with `scripts/debug_multitask_data.py`
   - Verify batch composition
   - Monitor loss curves for NaN or divergence
   - Test on small subset first

---

## Documentation Updates

After successful implementation:

1. **Update README.md:**
   - Document unified preprocessing requirement
   - Add normalization consistency note
   - Update multi-task training instructions

2. **Update QUICKSTART COMMAND LIST.md:**
   - Add unified Kaggle preprocessing step
   - Update config paths
   - Add validation commands

3. **Create MULTITASK_FIX_SUMMARY.md:**
   - Document root causes
   - Explain fixes applied
   - Show before/after metrics
   - Include Grad-CAM comparisons

4. **Update Training Configs:**
   - Add comments explaining normalization requirement
   - Document transform usage
   - Explain dataset mixing strategy

---

## Next Steps

### Immediate Actions (Start Here)
1. Run `python src/data/preprocess_kaggle_unified.py` to generate z-score normalized Kaggle data
2. Update `configs/stages/stage2_cls_head.yaml` and `stage3_joint.yaml` with new paths
3. Run verification script to confirm normalization consistency
4. Execute 5-epoch quick test of Stage 2
5. Generate and compare Grad-CAMs

### Follow-up Actions
1. Implement transform wrapper for segmentation tasks
2. Add transforms to multi-task training scripts
3. Run full Stage 2 and Stage 3 training
4. Evaluate on test set and compare to standalone models
5. Document results and update codebase

### Optional Enhancements
1. Implement dataset mixing with WeightedRandomSampler
2. Wire loss weights from config
3. Retrain with all optimizations
4. Generate comprehensive evaluation report

---

## Conclusion

The multi-task model's poor classification performance is primarily caused by **normalization mismatch** between BraTS (z-score) and Kaggle (min-max) datasets, compounded by **missing preprocessing transforms** (skull masking, augmentations) and **dataset imbalance**.

The fix is straightforward and high-impact:
1. **Use unified z-score Kaggle data** (15 min setup, +15-25% accuracy expected)
2. **Add skull masking and augmentations** (1 hour implementation, +5-10% accuracy expected)
3. **Optionally implement dataset mixing** (2 hours, +3-5% accuracy expected)

With these changes, the multi-task model should achieve **90-93% classification accuracy** with focused Grad-CAM attention on tumors, making it clinically viable while maintaining strong segmentation performance (Dice > 0.75).

**Priority:** Start with Phase 1 (normalization) immediately—this single fix will likely resolve the majority of the performance gap.

---

**Document Version:** 1.0  
**Last Updated:** December 18, 2025  
**Author:** Cascade AI Assistant  
**Status:** Ready for Implementation
