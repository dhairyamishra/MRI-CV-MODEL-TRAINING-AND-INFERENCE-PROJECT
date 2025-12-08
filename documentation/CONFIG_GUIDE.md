# Configuration Parameters Guide

Complete reference for all configuration parameters used in SliceWise Multi-Task Brain Tumor Detection training.

---

## Table of Contents

1. [Experiment Metadata](#1-experiment-metadata)
2. [Paths & Directories](#2-paths--directories)
3. [Model Architecture](#3-model-architecture)
4. [Loss Functions](#4-loss-functions)
5. [Optimizer](#5-optimizer)
6. [Learning Rate Scheduler](#6-learning-rate-scheduler)
7. [Training Hyperparameters](#7-training-hyperparameters)
8. [Early Stopping](#8-early-stopping)
9. [Checkpoint Saving](#9-checkpoint-saving)
10. [Data Augmentation](#10-data-augmentation)
11. [Dataset Configuration](#11-dataset-configuration)
12. [Weights & Biases Logging](#12-weights--biases-logging)
13. [Performance Optimization](#13-performance-optimization)

---

## 1. Experiment Metadata

Descriptive information about the training run.

```yaml
experiment:
  name: "multitask_seg_warmup_production"
  description: "Phase 2.1 PRODUCTION - Segmentation warm-up"
  tags: ["multitask", "segmentation", "warmup"]
  notes: "Additional notes about this experiment"
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Unique identifier for the experiment |
| `description` | string | Human-readable description |
| `tags` | list | Keywords for filtering/searching experiments |
| `notes` | string | Additional context or observations |

**Purpose:** Helps organize and track experiments, especially when using W&B.

---

## 2. Paths & Directories

File system locations for data, checkpoints, and outputs.

```yaml
paths:
  train_dir: "data/processed/brats2d_full/train"
  val_dir: "data/processed/brats2d_full/val"
  test_dir: "data/processed/brats2d_full/test"
  checkpoint_dir: "checkpoints/multitask_seg_warmup_production"
  log_dir: "logs/multitask_seg_warmup_production"
  output_dir: "outputs/multitask_seg_warmup_production"
```

| Parameter | Description | Example |
|-----------|-------------|---------|
| `train_dir` | Training data directory | `data/processed/brats2d_full/train` |
| `val_dir` | Validation data directory | `data/processed/brats2d_full/val` |
| `test_dir` | Test data directory | `data/processed/brats2d_full/test` |
| `checkpoint_dir` | Where to save model checkpoints | `checkpoints/multitask_seg_warmup_production` |
| `log_dir` | Training logs location | `logs/multitask_seg_warmup_production` |
| `output_dir` | Evaluation outputs (metrics, plots) | `outputs/multitask_seg_warmup_production` |

**For Stage 2 & 3:** Also includes `kaggle_train_dir`, `kaggle_val_dir`, `kaggle_test_dir` for classification data.

---

## 3. Model Architecture

Defines the neural network structure.

```yaml
model:
  name: "multitask"
  in_channels: 1
  out_channels: 1          # For segmentation
  seg_out_channels: 1      # Alternative name
  num_classes: 2           # For classification
  cls_num_classes: 2       # Alternative name
  base_filters: 64
  depth: 4
  cls_hidden_dim: 256
  cls_dropout: 0.5
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | "multitask" | Model type identifier |
| `in_channels` | int | 1 | Input channels (1 for grayscale MRI) |
| `out_channels` | int | 1 | Segmentation output channels (1 for binary) |
| `seg_out_channels` | int | 1 | Same as `out_channels` (alternative name) |
| `num_classes` | int | 2 | Classification classes (2 for tumor/no-tumor) |
| `cls_num_classes` | int | 2 | Same as `num_classes` (alternative name) |
| `base_filters` | int | 64 | Starting number of filters in encoder |
| `depth` | int | 4 | Number of encoder/decoder blocks |
| `cls_hidden_dim` | int | 256 | Hidden layer size in classification head |
| `cls_dropout` | float | 0.5 | Dropout rate in classification head |

### Understanding `base_filters` and `depth`:

**`base_filters`:** Controls model capacity
- 32: Lightweight (~8M parameters)
- 64: Standard (~31M parameters) ✅ **Recommended**
- 128: Heavy (~120M parameters)

**`depth`:** Controls network depth
- 3: Shallow (faster, less capacity)
- 4: Standard ✅ **Recommended**
- 5: Deep (slower, more capacity)

**Filter progression with `base_filters=64`, `depth=4`:**
- Encoder: 64 → 128 → 256 → 512
- Decoder: 512 → 256 → 128 → 64

---

## 4. Loss Functions

Defines how the model's error is calculated.

### Stage 1: Segmentation Loss

```yaml
loss:
  name: "dice_bce"
  dice_weight: 0.6
  bce_weight: 0.4
  smooth: 1.0
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `name` | string | - | Loss function type |
| `dice_weight` | float | 0-1 | Weight for Dice loss component |
| `bce_weight` | float | 0-1 | Weight for Binary Cross-Entropy |
| `smooth` | float | >0 | Smoothing factor to prevent division by zero |

**Loss Types:**
- `dice`: Dice coefficient loss (good for segmentation)
- `bce`: Binary Cross-Entropy (pixel-wise loss)
- `dice_bce`: Combined (recommended) ✅

**Formula:** `L_total = dice_weight * L_dice + bce_weight * L_bce`

### Stage 2: Classification Loss

```yaml
loss:
  name: "cross_entropy"
  label_smoothing: 0.1
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `name` | string | - | "cross_entropy" for classification |
| `label_smoothing` | float | 0-0.3 | Prevents overconfidence (0.1 recommended) |

**Label Smoothing:** Converts hard labels (0, 1) to soft labels (0.1, 0.9)
- Helps prevent overfitting
- Improves calibration
- Typical range: 0.05-0.15

### Stage 3: Multi-Task Loss

```yaml
loss:
  seg_loss:
    name: "dice_bce"
    dice_weight: 0.6
    bce_weight: 0.4
  cls_loss:
    name: "cross_entropy"
    label_smoothing: 0.1
  lambda_seg: 1.0
  lambda_cls: 0.5
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `lambda_seg` | float | Weight for segmentation loss |
| `lambda_cls` | float | Weight for classification loss |

**Formula:** `L_total = lambda_seg * L_seg + lambda_cls * L_cls`

**Tuning Tips:**
- Equal weights (1.0, 1.0): Balanced
- Higher seg (1.0, 0.5): Prioritize segmentation ✅
- Higher cls (0.5, 1.0): Prioritize classification

---

## 5. Optimizer

Controls how model weights are updated.

```yaml
optimizer:
  name: "adamw"
  lr: 0.0003
  weight_decay: 0.0001
  betas: [0.9, 0.999]
  eps: 1.0e-8
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `name` | string | - | Optimizer type (adam, adamw, sgd) |
| `lr` | float | 1e-5 to 1e-2 | Learning rate (step size) |
| `weight_decay` | float | 0-0.01 | L2 regularization strength |
| `betas` | list | [0.9, 0.999] | Momentum parameters for Adam |
| `eps` | float | 1e-8 | Numerical stability constant |

### Learning Rate Guidelines:

| Stage | LR | Rationale |
|-------|-----|-----------|
| Stage 1 (Seg Warmup) | 3e-4 | Training from scratch |
| Stage 2 (Cls Head) | 1e-3 | New head, frozen encoder |
| Stage 3 (Joint) | 1e-4 (encoder), 3e-4 (heads) | Fine-tuning |

**Optimizer Types:**
- `adam`: Standard adaptive optimizer
- `adamw`: Adam with decoupled weight decay ✅ **Recommended**
- `sgd`: Stochastic Gradient Descent (simpler, sometimes better)

**Weight Decay:**
- 0: No regularization
- 1e-4: Light regularization ✅ **Recommended**
- 1e-3: Strong regularization

---

## 6. Learning Rate Scheduler

Adjusts learning rate during training.

```yaml
scheduler:
  name: "cosine"
  T_max: 1000
  eta_min: 1.0e-7
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Scheduler type |
| `T_max` | int | Number of epochs for one cosine cycle |
| `eta_min` | float | Minimum learning rate |

### Scheduler Types:

**Cosine Annealing** ✅ **Recommended**
```
LR starts high, smoothly decreases to eta_min over T_max epochs
```

**Step LR**
```yaml
scheduler:
  name: "step"
  step_size: 30
  gamma: 0.1
```
Reduces LR by `gamma` every `step_size` epochs.

**ReduceLROnPlateau**
```yaml
scheduler:
  name: "plateau"
  patience: 10
  factor: 0.5
```
Reduces LR when metric plateaus.

### Cosine Annealing Visualization:
```
LR
 |
 |●
 | ●
 |  ●
 |   ●
 |    ●●
 |      ●●
 |        ●●●
 |           ●●●●●●●●●●●●●
 +-------------------------> Epochs
 0                      T_max
```

---

## 7. Training Hyperparameters

Core training settings.

```yaml
training:
  epochs: 1000
  batch_size: 32
  num_workers: 0
  pin_memory: true
  use_amp: true
  grad_clip: 1.0
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `epochs` | int | 1-10000 | Maximum training epochs |
| `batch_size` | int | 1-128 | Samples per batch |
| `num_workers` | int | 0-16 | Data loading threads |
| `pin_memory` | bool | - | Faster GPU transfer (set true) |
| `use_amp` | bool | - | Automatic Mixed Precision |
| `grad_clip` | float | 0.5-5.0 | Gradient clipping threshold |

### Batch Size Guidelines:

| GPU VRAM | Recommended Batch Size |
|----------|------------------------|
| 6 GB | 8-16 |
| 8 GB | 16-24 |
| 10-12 GB | 24-32 ✅ |
| 16+ GB | 32-64 |

**Too large:** Out of memory (OOM) error
**Too small:** Unstable training, slower convergence

### Mixed Precision Training (`use_amp`):

**Benefits:**
- ✅ 2x faster training
- ✅ 50% less GPU memory
- ✅ Same accuracy

**How it works:** Uses FP16 (half precision) for most operations, FP32 for critical parts.

### Gradient Clipping (`grad_clip`):

Prevents exploding gradients by capping gradient norm.

```
If ||gradient|| > grad_clip:
    gradient = gradient * (grad_clip / ||gradient||)
```

**Values:**
- 0.5: Aggressive clipping
- 1.0: Standard ✅ **Recommended**
- 5.0: Gentle clipping

### Stage 3: Differential Learning Rates

```yaml
training:
  encoder_lr: 0.0001
  decoder_cls_lr: 0.0003
  weight_decay: 0.0001
```

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `encoder_lr` | LR for pre-trained encoder | 1e-4 (lower) |
| `decoder_cls_lr` | LR for task heads | 3e-4 (higher) |

**Why different LRs?**
- Encoder is already trained → small updates
- Task heads are new → larger updates

---

## 8. Early Stopping

Automatically stops training when validation metric plateaus.

```yaml
early_stopping:
  enabled: true
  patience: 50
  min_delta: 0.0005
  monitor: "val_dice"
  mode: "max"
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled` | bool | Enable/disable early stopping |
| `patience` | int | Epochs to wait before stopping |
| `min_delta` | float | Minimum improvement to count |
| `monitor` | string | Metric to track |
| `mode` | string | "max" (higher is better) or "min" (lower is better) |

### Monitored Metrics:

| Stage | Monitor | Mode | Description |
|-------|---------|------|-------------|
| Stage 1 | `val_dice` | max | Validation Dice score |
| Stage 2 | `val_acc` | max | Validation accuracy |
| Stage 3 | `val_combined` | max | (dice + acc) / 2 |

### How It Works:

```
Epoch 1: val_dice = 0.70 (best so far, save checkpoint)
Epoch 2: val_dice = 0.72 (improved by 0.02 > min_delta, save)
Epoch 3: val_dice = 0.71 (decreased, counter = 1)
Epoch 4: val_dice = 0.71 (no improvement, counter = 2)
...
Epoch 52: val_dice = 0.72 (counter = 50, STOP!)
```

### Patience Guidelines:

| Training Mode | Patience | Rationale |
|---------------|----------|-----------|
| Quick Test | 10 | Fast iteration |
| Baseline | 20 | Moderate |
| Production | 50 ✅ | Patient, thorough |

---

## 9. Checkpoint Saving

Controls how and when models are saved.

```yaml
save_best: true
save_last: true
save_frequency: 10
keep_last_n: 5
save_optimizer: true
save_scheduler: true
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `save_best` | bool | Save best model based on validation metric |
| `save_last` | bool | Save most recent checkpoint |
| `save_frequency` | int | Save every N epochs |
| `keep_last_n` | int | Keep only last N periodic checkpoints |
| `save_optimizer` | bool | Include optimizer state (for resuming) |
| `save_scheduler` | bool | Include scheduler state (for resuming) |

### Checkpoint Files:

```
checkpoints/multitask_seg_warmup_production/
├── best_model.pth          ← Best validation score (ALWAYS KEPT)
├── last_model.pth          ← Most recent epoch (ALWAYS KEPT)
├── checkpoint_epoch_010.pth
├── checkpoint_epoch_020.pth
├── checkpoint_epoch_030.pth
├── checkpoint_epoch_040.pth
└── checkpoint_epoch_050.pth  (older ones deleted if > keep_last_n)
```

### What's Saved in a Checkpoint:

```python
checkpoint = {
    'epoch': 42,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),  # if save_optimizer=true
    'scheduler_state_dict': scheduler.state_dict(),  # if save_scheduler=true
    'val_dice': 0.7845,
    'train_loss': 0.1234,
    # ... other metrics
}
```

### Disk Space:

| Checkpoint Type | Size | Count |
|-----------------|------|-------|
| Model only | ~120 MB | 1 (best) + 1 (last) + keep_last_n |
| With optimizer | ~500 MB | Same |

**Example:** `save_frequency=10`, `keep_last_n=5`, 100 epochs
- Disk usage: ~3.5 GB (best + last + 5 periodic)

---

## 10. Data Augmentation

Random transformations applied to training data.

```yaml
augmentation:
  train:
    enabled: true
    random_flip_h: 0.5
    random_flip_v: 0.5
    random_rotate: 20
    random_scale: 0.15
    elastic_deform: true
    elastic_alpha: 50
    elastic_sigma: 5
    gaussian_noise: 0.02
    gaussian_blur: 0.3
    brightness: 0.2
    contrast: 0.2
    gamma: 0.2
  val:
    enabled: false
```

### Geometric Augmentations:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `random_flip_h` | float | 0-1 | Horizontal flip probability |
| `random_flip_v` | float | 0-1 | Vertical flip probability |
| `random_rotate` | int | 0-180 | Max rotation degrees (±) |
| `random_scale` | float | 0-0.5 | Max scale factor (±) |

**Example:** `random_rotate: 20`
- Image randomly rotated between -20° and +20°

### Elastic Deformation:

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `elastic_deform` | Enable elastic deformation | true |
| `elastic_alpha` | Deformation strength | 50 |
| `elastic_sigma` | Deformation smoothness | 5 |

**Purpose:** Simulates natural tissue deformation in MRI scans.

### Intensity Augmentations:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `gaussian_noise` | float | 0-0.1 | Noise standard deviation |
| `gaussian_blur` | float | 0-1 | Blur probability |
| `brightness` | float | 0-0.5 | Brightness adjustment range (±) |
| `contrast` | float | 0-0.5 | Contrast adjustment range (±) |
| `gamma` | float | 0-0.5 | Gamma correction range (±) |

**Example:** `brightness: 0.2`
- Brightness randomly adjusted by ±20%

### Augmentation Strength by Stage:

| Stage | Strength | Rationale |
|-------|----------|-----------|
| Stage 1 | Aggressive | Learning robust features |
| Stage 2 | Moderate | Classification is more sensitive |
| Stage 3 | Moderate | Balancing both tasks |

### Validation Augmentation:

```yaml
val:
  enabled: false  # ALWAYS false for validation!
```

**Why?** Validation should be deterministic for fair comparison.

---

## 11. Dataset Configuration

(Stage 3 only) Controls how BraTS and Kaggle datasets are mixed.

```yaml
dataset:
  mix_datasets: true
  brats_ratio: 0.3
  kaggle_ratio: 0.7
  sampling: "balanced"
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `mix_datasets` | bool | Enable dataset mixing |
| `brats_ratio` | float | Proportion of BraTS samples per batch |
| `kaggle_ratio` | float | Proportion of Kaggle samples per batch |
| `sampling` | string | "balanced" or "weighted" |

### How Mixing Works:

**Batch size = 32, brats_ratio = 0.3:**
- ~10 samples from BraTS (with segmentation masks)
- ~22 samples from Kaggle (classification only)

**Why mix?**
- BraTS: Has segmentation labels
- Kaggle: Has classification labels only
- Mixed batches train both tasks simultaneously

---

## 12. Weights & Biases Logging

Integration with W&B for experiment tracking.

```yaml
wandb:
  enabled: true
  project: "slicewise-multitask-production"
  entity: null
  name: "seg_warmup_production"
  tags: ["production", "stage1"]
  notes: "Production training with full dataset"
  log_gradients: true
  log_weights: false
  log_frequency: 100
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled` | bool | Enable W&B logging |
| `project` | string | W&B project name |
| `entity` | string | W&B username/team (null = default) |
| `name` | string | Run name |
| `tags` | list | Tags for filtering runs |
| `notes` | string | Additional context |
| `log_gradients` | bool | Log gradient norms |
| `log_weights` | bool | Log weight histograms (large!) |
| `log_frequency` | int | Log every N batches |

### What Gets Logged:

**Every Epoch:**
- Train/val loss
- Train/val metrics (dice, acc, etc.)
- Learning rate
- Epoch time

**Every `log_frequency` Batches:**
- Batch loss
- Gradient norms (if `log_gradients=true`)

**Once:**
- Config parameters
- Model architecture
- System info (GPU, CPU, etc.)

### Viewing Results:

1. Go to https://wandb.ai
2. Navigate to your project
3. Compare runs, view charts, download data

---

## 13. Performance Optimization

Advanced settings for speed and efficiency.

```yaml
performance:
  gradient_accumulation_steps: 1
  compile_model: false
  cudnn_benchmark: true
  cudnn_deterministic: false
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `gradient_accumulation_steps` | int | Accumulate gradients over N batches |
| `compile_model` | bool | Use PyTorch 2.0 compilation |
| `cudnn_benchmark` | bool | Auto-tune cuDNN kernels |
| `cudnn_deterministic` | bool | Force deterministic operations |

### Gradient Accumulation:

**Problem:** GPU memory limited, can't use large batch size

**Solution:** Accumulate gradients over multiple small batches

```yaml
batch_size: 8
gradient_accumulation_steps: 4
# Effective batch size = 8 * 4 = 32
```

**How it works:**
1. Forward pass on batch 1 (size 8)
2. Backward pass (accumulate gradients)
3. Forward pass on batch 2 (size 8)
4. Backward pass (accumulate gradients)
5. ... repeat 4 times
6. Update weights (effective batch size = 32)

### Model Compilation (PyTorch 2.0+):

```yaml
compile_model: true
```

**Benefits:**
- 20-50% faster training
- Optimizes model graph

**Requirements:**
- PyTorch 2.0+
- May have compatibility issues

### cuDNN Settings:

| Setting | Speed | Determinism |
|---------|-------|-------------|
| `benchmark=true, deterministic=false` | ✅ Fastest | ❌ Non-deterministic |
| `benchmark=false, deterministic=true` | ❌ Slower | ✅ Reproducible |

**Recommendation:** Use `benchmark=true` for production (faster)

---

## Quick Reference Tables

### Stage-Specific Configs:

| Parameter | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| **Epochs** | 1000 | 1000 | 1000 |
| **Batch Size** | 32 | 32 | 32 |
| **Learning Rate** | 3e-4 | 1e-3 | 1e-4 (enc), 3e-4 (heads) |
| **Loss** | Dice+BCE | CrossEntropy | Multi-task |
| **Monitor** | val_dice | val_acc | val_combined |
| **Patience** | 50 | 50 | 50 |

### Common Pitfalls:

| Issue | Symptom | Solution |
|-------|---------|----------|
| OOM Error | "CUDA out of memory" | Reduce `batch_size` |
| Slow Training | <1 it/s | Enable `use_amp`, increase `num_workers` |
| Overfitting | Train loss ↓, Val loss ↑ | Increase augmentation, weight_decay |
| Underfitting | Both losses high | Increase `base_filters`, `epochs` |
| Unstable Training | Loss oscillates wildly | Reduce `lr`, enable `grad_clip` |

---

## Example Configs

### Minimal Config (Quick Test):

```yaml
experiment:
  name: "quick_test"

model:
  base_filters: 32
  depth: 3

training:
  epochs: 10
  batch_size: 8

optimizer:
  lr: 0.001

early_stopping:
  patience: 5
```

### Production Config (Full Training):

```yaml
experiment:
  name: "production_run"

model:
  base_filters: 64
  depth: 4

training:
  epochs: 1000
  batch_size: 32
  use_amp: true

optimizer:
  lr: 0.0003
  weight_decay: 0.0001

scheduler:
  name: "cosine"
  T_max: 1000

early_stopping:
  patience: 50

augmentation:
  train:
    enabled: true
    random_rotate: 20
    elastic_deform: true
```

---

## Tuning Tips

### 1. Start Small, Scale Up
```
Quick test (10 epochs) → Baseline (100 epochs) → Production (1000 epochs)
```

### 2. Learning Rate Finder
```
Try: 1e-5, 1e-4, 1e-3, 1e-2
Pick: Fastest convergence without instability
```

### 3. Batch Size vs. Learning Rate
```
If you double batch_size, consider increasing lr by √2
```

### 4. Augmentation Strength
```
Too weak: Overfitting
Too strong: Underfitting (model can't learn)
Sweet spot: Train acc slightly > Val acc
```

### 5. Early Stopping Patience
```
Short training (<50 epochs): patience = 10
Medium training (50-200 epochs): patience = 20
Long training (>200 epochs): patience = 50
```

---

## Troubleshooting

### Training Not Starting

**Check:**
1. Data paths exist: `ls data/processed/brats2d_full/train`
2. GPU available: `nvidia-smi`
3. Config syntax: YAML indentation correct

### Training Crashes

**Common Causes:**
1. OOM: Reduce `batch_size`
2. NaN loss: Reduce `lr`, enable `grad_clip`
3. Data loading: Set `num_workers=0` (Windows)

### Poor Performance

**Checklist:**
1. ✅ Sufficient epochs (early stopping patience)
2. ✅ Appropriate learning rate
3. ✅ Augmentation enabled
4. ✅ Model capacity (base_filters, depth)
5. ✅ Class balance (check dataset distribution)

---

## Additional Resources

- [PyTorch Optimizer Docs](https://pytorch.org/docs/stable/optim.html)
- [Learning Rate Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [W&B Documentation](https://docs.wandb.ai/)

---

**Last Updated:** December 2025  
**Version:** 1.0  
**Author:** SliceWise Development Team
