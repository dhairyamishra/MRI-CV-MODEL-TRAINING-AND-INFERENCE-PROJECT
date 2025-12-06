# Phase 2.2 Quick Start Guide

**Goal**: Train the classification head with a frozen encoder using BraTS + Kaggle data.

---

## 📋 Prerequisites

- ✅ Phase 2.1 complete (encoder initialized from segmentation)
- ✅ Checkpoint: `checkpoints/multitask_seg_warmup/best_model.pth`
- ✅ BraTS data split into train/val/test
- ✅ Kaggle data preprocessed (~253 files)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Split Kaggle Data

Split the Kaggle dataset into train/val/test with stratification:

```bash
python scripts/split_kaggle_data.py
```

**Expected Output:**
```
======================================================================
Kaggle Dataset Splitting (Stratified)
======================================================================

Configuration:
  Input:  data/processed/kaggle
  Output: data/processed/kaggle
  Split ratios: 70% train / 15% val / 15% test
  Random seed: 42

Grouping files by class...
  No tumor: ~95 files
  Yes tumor: ~158 files
  Total: ~253 files

Split sizes:
  Train: ~177 files (66 no, 111 yes)
  Val:   ~38 files (14 no, 24 yes)
  Test:  ~38 files (15 no, 23 yes)

======================================================================
✓ Splitting Complete!
======================================================================
```

---

### Step 2: Run Classification Head Training

Train the classification head with the frozen encoder:

```bash
python scripts/train_multitask_cls_head.py \
  --config configs/multitask_cls_head_quick_test.yaml \
  --encoder-init checkpoints/multitask_seg_warmup/best_model.pth
```

**What This Does:**
1. ✅ Loads the multi-task model
2. ✅ Initializes encoder from Phase 2.1 checkpoint
3. ✅ **Freezes the encoder** (no gradient updates)
4. ✅ Trains only the classification head (~263K parameters)
5. ✅ Uses BraTS + Kaggle data (derived labels + real labels)

**Expected Output:**
```
================================================================================
Phase 2.2: Classification Head Training
================================================================================

Config: configs/multitask_cls_head_quick_test.yaml
Encoder init: checkpoints/multitask_seg_warmup/best_model.pth

Using device: cuda

=== Loading Encoder from checkpoints/multitask_seg_warmup/best_model.pth ===
✓ Loaded encoder and decoder from Phase 2.1 checkpoint

=== Freezing Encoder ===
Total parameters: 2,014,019
Trainable parameters: 263,170
Frozen parameters: 1,750,849

=== Loading Datasets ===
Loaded 417 BraTS samples
Loaded 177 Kaggle samples
Train samples: 594 (BraTS + Kaggle)
Val samples: 83

=== Starting Training ===
Training for 10 epochs
Batch size: 8
Learning rate: 0.001

============================================================
Epoch 1/10
============================================================
Training: 100%|████████| 75/75 [00:05<00:00, 14.2it/s, loss=0.6234, acc=0.6500]
Validation: 100%|████████| 11/11 [00:00<00:00, 22.1it/s, loss=0.5891, acc=0.7012]

Train - Loss: 0.6234, Acc: 0.6500
Val   - Loss: 0.5891, Acc: 0.7012
✓ New best model! Acc: 0.7012

... (epochs 2-10) ...

=== Training Complete ===
Best validation Accuracy: 0.8554
Checkpoints saved to: checkpoints/multitask_cls_head_test

Next: Use this checkpoint for Phase 2.3 (Joint Fine-Tuning)
```

**Training Time**: ~1-2 minutes (10 epochs)

---

### Step 3: Verify Results

Check that the checkpoint was saved:

```bash
dir checkpoints\multitask_cls_head_test
```

Expected files:
- `best_model.pth` (~8 MB)
- `last_model.pth` (~8 MB)

---

## 📊 Expected Results

### Quick Test (10 epochs):
- **Train Acc**: ~0.75-0.85
- **Val Acc**: ~0.70-0.80
- **Training Time**: ~1-2 minutes
- **Trainable Params**: 263,170 (only classification head!)
- **Frozen Params**: 1,750,849 (encoder + decoder)

### What This Validates:
✅ Encoder loads from Phase 2.1 checkpoint  
✅ Encoder freezing works correctly  
✅ MultiSourceDataset loads BraTS + Kaggle  
✅ Classification head trains on both datasets  
✅ Model converges with frozen encoder  
✅ Ready for Phase 2.3 joint fine-tuning  

---

## 🔍 Key Differences from Phase 2.1

| Aspect | Phase 2.1 | Phase 2.2 |
|--------|-----------|-----------|
| **Task** | Segmentation only | Classification only |
| **Dataset** | BraTS only | BraTS + Kaggle |
| **Trainable** | Encoder + Decoder | Classification head only |
| **Parameters** | 2.0M trainable | 263K trainable (87% frozen!) |
| **Forward Pass** | `do_seg=True, do_cls=False` | `do_seg=False, do_cls=True` |
| **Loss** | Dice + BCE | CrossEntropy |
| **Output** | Segmentation mask | Class probabilities |

---

## 🔧 Troubleshooting

### Issue: "No Kaggle data found"
**Solution**: Run the Kaggle data splitting script first:
```bash
python scripts/split_kaggle_data.py
```

### Issue: "Encoder checkpoint not found"
**Solution**: Make sure Phase 2.1 completed successfully:
```bash
dir checkpoints\multitask_seg_warmup\best_model.pth
```

### Issue: "Model architecture mismatch"
**Solution**: Ensure Phase 2.2 config matches Phase 2.1:
- `base_filters: 32` (same as Phase 2.1)
- `depth: 3` (same as Phase 2.1)

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in config:
```yaml
training:
  batch_size: 4  # Reduce from 8
```

---

## 📈 Next Steps After Successful Test

### Option A: Run Full Classification Head Training (30 epochs)
```bash
python scripts/train_multitask_cls_head.py \
  --config configs/multitask_cls_head.yaml \
  --encoder-init checkpoints/multitask_seg_warmup/best_model.pth
```
- **Training Time**: ~5-10 minutes
- **Expected Val Acc**: ~0.85-0.90

### Option B: Proceed to Phase 2.3 (Joint Fine-Tuning)
1. Create Phase 2.3 config
2. Implement alternating batch strategy
3. Unfreeze encoder and train both tasks together

---

## 📝 What We're Training

### Phase 2.2 Components:

1. **MultiTaskModel** in classification-only mode
   - Forward pass: `model(x, do_seg=False, do_cls=True)`
   - Returns: `{'seg': None, 'cls': logits, 'features': [...]}`

2. **Frozen Encoder**
   - Weights loaded from Phase 2.1
   - `requires_grad = False` for all encoder parameters
   - Provides good features from segmentation task

3. **Trainable Classification Head**
   - Global average pooling + MLP
   - Only 263K parameters (0.8% of total model)
   - Learns to classify using segmentation features

4. **MultiSourceDataset**
   - Combines BraTS + Kaggle data
   - BraTS: Derived labels from masks (has tumor if mask > 0)
   - Kaggle: Real classification labels

---

## 🎯 Success Criteria

✅ **Training completes without errors**  
✅ **Validation Acc > 0.70** (for 10 epochs)  
✅ **Only 263K parameters trainable** (encoder frozen)  
✅ **Checkpoint files created**  
✅ **Training time < 3 minutes**  

If all criteria pass, you're ready for Phase 2.3! 🚀

---

## 📚 Related Documentation

- `PHASE2_QUICK_TEST_GUIDE.md` - Phase 2.1 segmentation warm-up
- `PHASE1_COMPLETE.md` - Multi-task model architecture
- `src/models/multi_task_model.py` - Model implementation
- `src/training/train_multitask_cls_head.py` - Training script

---

**Last Updated**: December 6, 2025  
**Status**: Ready for testing ✅
