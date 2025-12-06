# Phase 2.1 Quick Test Guide

**Goal**: Test the segmentation warm-up training pipeline with a quick 5-epoch run on 10 patients.

---

## 📋 Prerequisites

- ✅ Phase 1 complete (multi-task model architecture)
- ✅ BraTS 2D data preprocessed (10 patients, ~569 slices)
- ✅ CUDA-capable GPU (recommended)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Split the Data

First, we need to split the BraTS data into train/val/test sets:

```bash
python scripts/split_brats_data.py
```

**Expected Output:**
```
================================================================================
BraTS 2D Data Splitting
================================================================================

Input directory: data/processed/brats2d
Output directory: data/processed/brats2d
Split ratios: Train=0.7, Val=0.15, Test=0.15
Random seed: 42

Found 569 .npz files

Splitting dataset...
  10 patients total
  Train: 7 patients (417 slices)
  Val: 1 patients (45 slices)
  Test: 2 patients (107 slices)

================================================================================
✓ Data splitting complete!
================================================================================

Data split into:
  - data/processed/brats2d/train/
  - data/processed/brats2d/val/
  - data/processed/brats2d/test/

You can now proceed with training!
```

---

### Step 2: Run Quick Training Test

Run the segmentation warm-up training with the quick test config:

```bash
python scripts/train_multitask_seg_warmup.py --config configs/multitask_seg_warmup_quick_test.yaml
```

**Expected Output:**
```
================================================================================
Phase 2.1: Segmentation Warm-Up Training
================================================================================

Config: configs/multitask_seg_warmup_quick_test.yaml
Checkpoint directory: checkpoints/multitask_seg_warmup_test

Using device: cuda

=== Loading Datasets ===
Loaded 417 BraTS samples from data/processed/brats2d/train
Loaded 45 BraTS samples from data/processed/brats2d/val
Train samples: 417
Val samples: 45

=== Creating Multi-Task Model ===
Total parameters: 7,938,049
Trainable parameters: 7,938,049

=== Starting Training ===
Training for 5 epochs
Batch size: 4
Learning rate: 0.001
Mixed precision: True

============================================================
Epoch 1/5
============================================================
Training: 100%|████████████| 105/105 [00:15<00:00, 6.8it/s, loss=0.4523, dice=0.6234, iou=0.4521]
Validation: 100%|████████████| 12/12 [00:02<00:00, 5.2it/s, loss=0.3891, dice=0.6892, iou=0.5234]

Train - Loss: 0.4523, Dice: 0.6234, IoU: 0.4521
Val   - Loss: 0.3891, Dice: 0.6892, IoU: 0.5234
✓ New best model! Dice: 0.6892
Saved checkpoint to checkpoints/multitask_seg_warmup_test/best_model.pth

... (epochs 2-5) ...

=== Training Complete ===
Best validation Dice: 0.7306
Checkpoints saved to: checkpoints/multitask_seg_warmup_test

Next: Use this checkpoint to initialize encoder for Phase 2.2 (Classification Head Training)
```

**Training Time**: ~2-3 minutes (5 epochs on 10 patients)

---

### Step 3: Verify Checkpoint

Check that the checkpoint was saved correctly:

```bash
# Windows
dir checkpoints\multitask_seg_warmup_test

# Expected files:
# - best_model.pth
# - last_model.pth
```

---

## 📊 Expected Results

### Quick Test (5 epochs, 10 patients):
- **Train Dice**: ~0.70-0.85
- **Val Dice**: ~0.65-0.75
- **Training Time**: ~2-3 minutes
- **Checkpoint Size**: ~120 MB

### What This Validates:
✅ Multi-task model architecture works  
✅ BraTS2DSliceDataset loads correctly  
✅ Training loop runs without errors  
✅ Mixed precision (AMP) works  
✅ Checkpoints save correctly  
✅ Encoder can be extracted for Phase 2.2  

---

## 🔍 Troubleshooting

### Issue: "No .npz files found"
**Solution**: Run the data splitting script first:
```bash
python scripts/split_brats_data.py
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in config:
```yaml
training:
  batch_size: 2  # Reduce from 4
```

### Issue: "ModuleNotFoundError"
**Solution**: Make sure you're in the project root:
```bash
cd c:\--DPM-MAIN-DIR--\windsurf_projects\MRI-CV-MODEL-TRAINING-AND-INFERENCE-PROJECT
python scripts/train_multitask_seg_warmup.py --config configs/multitask_seg_warmup_quick_test.yaml
```

### Issue: Training is too slow
**Solution**: This is expected for the first epoch (JIT compilation). Subsequent epochs should be faster.

---

## 📈 Next Steps After Successful Test

Once the quick test passes:

### Option A: Run Full Segmentation Warm-Up (50 epochs)
```bash
python scripts/train_multitask_seg_warmup.py --config configs/multitask_seg_warmup.yaml
```
- **Training Time**: ~20-30 minutes
- **Expected Val Dice**: ~0.75-0.80

### Option B: Proceed to Phase 2.2 (Classification Head Training)
1. Create config for Phase 2.2
2. Download/prepare Kaggle dataset
3. Train classification head with frozen encoder

### Option C: Proceed to Phase 2.3 (Joint Fine-Tuning)
1. Create config for Phase 2.3
2. Implement alternating batch training
3. Fine-tune entire model

---

## 📝 What We're Testing

### Phase 2.1 Components:
1. **MultiTaskModel** in segmentation-only mode
   - Forward pass: `model(x, do_seg=True, do_cls=False)`
   - Returns: `{'seg': logits, 'cls': None, 'features': [...]}`

2. **BraTS2DSliceDataset**
   - Loads .npz files with image + mask
   - Returns: `{'image': tensor, 'mask': tensor}`

3. **Training Loop**
   - Mixed precision (AMP)
   - Gradient clipping
   - Early stopping
   - Checkpoint management

4. **Loss Function**
   - Dice + BCE combined loss
   - Handles binary segmentation

---

## 🎯 Success Criteria

✅ **Training completes without errors**  
✅ **Validation Dice > 0.65** (for 5 epochs)  
✅ **Checkpoint files created** (best_model.pth, last_model.pth)  
✅ **Training time < 5 minutes** (for 5 epochs)  
✅ **GPU memory usage < 8GB**  

If all criteria pass, you're ready to proceed with the full training pipeline!

---

## 📚 Related Documentation

- `PHASE1_COMPLETE.md` - Multi-task model architecture
- `configs/multitask_seg_warmup.yaml` - Full training config (50 epochs)
- `src/models/multi_task_model.py` - Model implementation
- `src/training/train_multitask_seg_warmup.py` - Training script

---

**Last Updated**: December 6, 2025  
**Status**: Ready for testing ✅
