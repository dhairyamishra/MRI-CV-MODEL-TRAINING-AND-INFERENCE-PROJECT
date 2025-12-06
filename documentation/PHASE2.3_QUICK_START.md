# Phase 2.3: Joint Fine-Tuning - Quick Start Guide

**Goal**: Train both segmentation and classification tasks jointly with an unfrozen encoder to achieve the best performance on both tasks.

---

## 🎯 What is Phase 2.3?

Phase 2.3 is the **final training stage** where we:
1. ✅ **Unfreeze the encoder** (all 2.0M parameters trainable)
2. ✅ **Train both tasks together** (segmentation + classification)
3. ✅ **Use combined loss** (L_seg + λ_cls * L_cls)
4. ✅ **Apply differential learning rates** (encoder: 1e-4, decoder/cls_head: 3e-4)

This allows the encoder to learn features that are optimal for **both** tasks simultaneously!

---

## 📊 What We Have So Far

| Phase | Task | Encoder | Best Metric | Status |
|-------|------|---------|-------------|--------|
| 2.1 | Segmentation | Trainable | Dice: 0.7120 | ✅ Complete |
| 2.2 | Classification | **Frozen** | Acc: 0.8365 | ✅ Complete |
| 2.3 | **Both** | **Trainable** | TBD | ⏳ Ready to run |

**Phase 2.3 Goal**: Improve both tasks by joint training!

---

## 🚀 Quick Start (2 Minutes)

### Step 1: Verify Prerequisites

Make sure Phase 2.2 completed successfully:

```bash
# Check if Phase 2.2 checkpoint exists
ls checkpoints/multitask_cls_head/best_model.pth
```

**Expected output**: File should exist with ~8-10 MB size

### Step 2: Run Phase 2.3 Training

```bash
python scripts/train_multitask_joint.py \
    --config configs/multitask_joint_quick_test.yaml \
    --init-from checkpoints/multitask_cls_head/best_model.pth
```

**Training time**: ~2-3 minutes (10 epochs)

---

## 📋 What Happens During Training

### 1. **Initialization**
- Loads Phase 2.2 checkpoint (encoder + decoder + cls_head)
- **Unfreezes all parameters** (2.0M trainable!)
- Sets up differential learning rates

### 2. **Training Loop** (Each Epoch)

For each batch:
- **If BraTS sample** (has mask):
  - Compute segmentation loss (Dice + BCE)
  - Compute classification loss (CrossEntropy)
  - Total loss = L_seg + λ_cls * L_cls
  
- **If Kaggle sample** (no mask):
  - Compute classification loss only
  - Total loss = λ_cls * L_cls

### 3. **Metrics Tracked**
- **Segmentation**: Dice score (on BraTS samples)
- **Classification**: Accuracy (on all samples)
- **Combined metric**: (Dice + Acc) / 2

---

## 🔧 Key Configuration Parameters

From `configs/multitask_joint_quick_test.yaml`:

```yaml
training:
  epochs: 10
  batch_size: 8
  
  # Differential learning rates
  encoder_lr: 0.0001      # 1e-4 (lower for fine-tuning)
  decoder_cls_lr: 0.0003  # 3e-4 (higher for task heads)

loss:
  seg_loss_type: "dice_bce"  # Combined Dice + BCE
  cls_loss_type: "ce"        # CrossEntropy
  lambda_cls: 1.0            # Weight for classification loss
```

**Why differential learning rates?**
- Encoder already learned good features in Phase 2.1 & 2.2
- Task heads (decoder + cls_head) need more adaptation
- Lower encoder LR prevents catastrophic forgetting

---

## 📊 Expected Results

### Quick Test (10 epochs)

| Metric | Phase 2.1 | Phase 2.2 | Phase 2.3 (Expected) |
|--------|-----------|-----------|----------------------|
| **Segmentation Dice** | 0.7120 | N/A | **0.72-0.75** ⬆️ |
| **Classification Acc** | N/A | 0.8365 | **0.85-0.88** ⬆️ |
| **Combined Metric** | - | - | **0.78-0.82** |

**Key Insight**: Joint training should improve **both** tasks!

### Full Training (20-30 epochs)

For production, increase epochs in config:
```yaml
training:
  epochs: 30  # Better convergence
```

Expected improvements:
- Segmentation Dice: **0.75-0.78**
- Classification Acc: **0.88-0.92**

---

## 🎯 Understanding the Output

### Training Progress

```
============================================================
Epoch 1/10
============================================================
Training: 100%|████| 74/74 [00:02<00:00, loss=0.4521, seg=0.2841, cls=0.1680, acc=0.8200]
Validation: 100%|██| 13/13 [00:00<00:00, loss=0.3892, dice=0.7234, acc=0.8571]

Train - Loss: 0.4521, Dice: 0.7012, Acc: 0.8200
Val   - Loss: 0.3892, Dice: 0.7234, Acc: 0.8571
✓ Saved best model (metric: 0.7903)
```

**Metrics explained**:
- `loss`: Combined loss (seg + cls)
- `seg`: Segmentation loss (Dice + BCE)
- `cls`: Classification loss (CrossEntropy)
- `dice`: Dice score for segmentation
- `acc`: Classification accuracy
- `metric`: Combined metric = (Dice + Acc) / 2

---

## 🔍 Troubleshooting

### Issue 1: "Initialization checkpoint not found"

**Error**:
```
❌ Error: Initialization checkpoint not found: checkpoints/multitask_cls_head/best_model.pth
```

**Solution**: Run Phase 2.2 first:
```bash
python scripts/train_multitask_cls_head.py \
    --config configs/multitask_cls_head_quick_test.yaml \
    --encoder-init checkpoints/multitask_seg_warmup/best_model.pth
```

### Issue 2: Out of Memory (OOM)

**Solution**: Reduce batch size in config:
```yaml
training:
  batch_size: 4  # Reduce from 8
```

### Issue 3: Training too slow

**Solution**: Disable mixed precision (if GPU doesn't support it):
```yaml
training:
  mixed_precision: false
```

---

## 📈 Monitoring Training

### 1. Watch Loss Curves

Both losses should decrease:
- **Segmentation loss**: Should stay around 0.25-0.35
- **Classification loss**: Should decrease to 0.15-0.25
- **Total loss**: Should decrease steadily

### 2. Check Metrics

- **Dice score**: Should improve from Phase 2.1 baseline
- **Accuracy**: Should maintain or improve from Phase 2.2
- **Combined metric**: Should increase over epochs

### 3. Look for Overfitting

- If train metrics >> val metrics: Reduce learning rates or add regularization
- If both plateau: Training converged successfully!

---

## 🎉 After Training Completes

### 1. Check Checkpoints

```bash
ls checkpoints/multitask_joint/
# Should see:
# - best_model.pth (best combined metric)
# - last_model.pth (final epoch)
```

### 2. Compare with Baselines

| Model | Seg Dice | Cls Acc | Combined |
|-------|----------|---------|----------|
| Phase 2.1 (Seg only) | 0.7120 | - | - |
| Phase 2.2 (Cls only) | - | 0.8365 | - |
| **Phase 2.3 (Joint)** | **TBD** | **TBD** | **TBD** |

### 3. Next Steps

1. **Evaluate on test set**:
   ```bash
   python scripts/evaluate_multitask.py \
       --checkpoint checkpoints/multitask_joint/best_model.pth \
       --test-dir data/processed/brats2d/test
   ```

2. **Generate visualizations**:
   - Segmentation masks
   - Grad-CAM heatmaps
   - Confusion matrices

3. **Create evaluation report**:
   - Compare all three phases
   - Analyze where joint training helps most
   - Document findings

---

## 🔬 Advanced: Understanding the Loss Function

### Combined Loss Formula

For **BraTS samples** (have masks):
```
L_total = L_seg + λ_cls * L_cls
L_seg = 0.5 * Dice_loss + 0.5 * BCE_loss
L_cls = CrossEntropy_loss
```

For **Kaggle samples** (no masks):
```
L_total = λ_cls * L_cls
L_cls = CrossEntropy_loss
```

### Why This Works

1. **Shared encoder** learns features useful for both tasks
2. **Segmentation** provides spatial understanding
3. **Classification** provides high-level semantic understanding
4. **Joint training** encourages encoder to learn features that help both!

---

## 📚 Key Files Created

### Training Code (3 files)
- `src/training/multi_task_losses.py` (239 lines) - Combined loss functions
- `src/training/train_multitask_joint.py` (488 lines) - Training script
- `scripts/train_multitask_joint.py` (148 lines) - Launcher script

### Configuration (1 file)
- `configs/multitask_joint_quick_test.yaml` (52 lines) - Training config

### Documentation (1 file)
- `documentation/PHASE2.3_QUICK_START.md` (This file!)

**Total new code**: ~927 lines across 5 files

---

## 🎯 Success Criteria

Phase 2.3 is successful if:
- ✅ Training completes without errors
- ✅ Segmentation Dice ≥ Phase 2.1 baseline (0.7120)
- ✅ Classification Acc ≥ Phase 2.2 baseline (0.8365)
- ✅ At least one metric shows improvement
- ✅ Combined metric > 0.75

---

## 🚀 Ready to Run!

Execute this command to start Phase 2.3:

```bash
python scripts/train_multitask_joint.py \
    --config configs/multitask_joint_quick_test.yaml \
    --init-from checkpoints/multitask_cls_head/best_model.pth
```

**Expected time**: ~2-3 minutes  
**Expected result**: Improved performance on both tasks! 🎉

---

**Questions or issues?** Check the troubleshooting section or review the training logs for detailed error messages.
