# SliceWise Production Training Guide

**Complete guide for training highly optimized models with 100+ epochs**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Training Configurations](#training-configurations)
4. [Step-by-Step Training](#step-by-step-training)
5. [Monitoring Training](#monitoring-training)
6. [Evaluation & Visualization](#evaluation--visualization)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)

---

## Overview

This guide covers the complete production training pipeline for SliceWise, including:

- **Classification Model**: EfficientNet-B0 for binary brain tumor detection
- **Segmentation Model**: U-Net 2D for precise tumor segmentation
- **100+ Epochs**: Highly optimized training for maximum performance
- **Full Pipeline**: Data download → Preprocessing → Training → Evaluation → Visualization

### What You'll Get

After completing this guide, you'll have:

✅ Fully trained models with checkpoints  
✅ Comprehensive evaluation metrics  
✅ Visualization of predictions (Grad-CAM, segmentation overlays)  
✅ Calibrated models for reliable confidence scores  
✅ Training curves and performance analysis  
✅ Production-ready models for deployment  

---

## Quick Start

### 🚀 Train Everything (Recommended)

Train both classification and segmentation models with full pipeline:

```bash
# Train both models with 100 epochs each
python scripts/train_production.py --task both --epochs 100

# Quick test (20 epochs for classification, 10 for segmentation)
python scripts/train_production.py --task both --quick-test
```

### 🎯 Train Individual Models

**Classification Only:**
```bash
python scripts/train_production.py --task classification --epochs 100
```

**Segmentation Only:**
```bash
python scripts/train_production.py --task segmentation --epochs 100
```

### ⚡ Skip Steps (If Data Already Exists)

```bash
# Skip data download
python scripts/train_production.py --task classification --skip-data

# Skip evaluation and visualization
python scripts/train_production.py --task classification --skip-eval --skip-viz
```

---

## Training Configurations

### Classification Configuration

**File:** `configs/config_cls_production.yaml`

**Key Features:**
- **Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Epochs**: 100
- **Batch Size**: 32
- **Optimizer**: AdamW (lr=0.0003, weight_decay=0.01)
- **Scheduler**: Cosine annealing with warmup
- **Loss**: Cross-entropy with label smoothing (0.1)
- **Augmentation**: Strong (rotation, flip, brightness, contrast)
- **Mixed Precision**: Enabled (AMP)
- **Early Stopping**: 20 epochs patience

**Expected Performance:**
- Train Accuracy: ~95-98%
- Val Accuracy: ~92-95%
- ROC-AUC: ~0.95-0.98
- Training Time: ~2-4 hours (GPU), ~12-24 hours (CPU)

### Segmentation Configuration

**File:** `configs/seg2d_production.yaml`

**Key Features:**
- **Model**: U-Net 2D (31.4M parameters)
- **Epochs**: 100
- **Batch Size**: 16 (effective 32 with gradient accumulation)
- **Optimizer**: AdamW (lr=0.0005, weight_decay=0.0001)
- **Scheduler**: Cosine annealing with warmup
- **Loss**: Dice + BCE (0.6:0.4 ratio)
- **Augmentation**: Strong (rotation, flip, elastic deformation, noise)
- **Mixed Precision**: Enabled (AMP)
- **Early Stopping**: 25 epochs patience

**Expected Performance:**
- Train Dice: ~0.85-0.90
- Val Dice: ~0.75-0.85
- IoU: ~0.65-0.75
- Training Time: ~4-8 hours (GPU), ~24-48 hours (CPU)

---

## Step-by-Step Training

### Step 1: Environment Setup

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

**Required:**
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU training)
- 16GB+ RAM
- 50GB+ disk space

### Step 2: Data Preparation

The training script automatically handles data download and preprocessing, but you can run these steps manually:

**Download Kaggle Data (Classification):**
```bash
python scripts/download_kaggle_data.py
python scripts/preprocess_kaggle.py
```

**Download BraTS Data (Segmentation):**
```bash
python scripts/download_brats_data.py
python scripts/preprocess_all_brats.py
```

**Expected Data Structure:**
```
data/
├── processed/
│   ├── kaggle/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── brats2d/
│       ├── train/
│       ├── val/
│       └── test/
└── raw/
    ├── kaggle/
    └── brats/
```

### Step 3: Configure Weights & Biases (Optional)

For experiment tracking:

```bash
# Login to W&B
wandb login

# Or disable W&B
# Edit config file: logging.use_wandb: false
```

### Step 4: Start Training

**Option A: Unified Pipeline (Recommended)**

```bash
# Train classification model
python scripts/train_production.py --task classification --epochs 100

# Monitor in another terminal
python scripts/monitor_training.py --task classification
```

**Option B: Manual Training**

```bash
# Classification
python scripts/train_classifier.py --config configs/config_cls_production.yaml

# Segmentation
python scripts/train_segmentation.py --config configs/seg2d_production.yaml
```

### Step 5: Monitor Training Progress

**Real-time Monitoring:**
```bash
# Live plots with auto-refresh
python scripts/monitor_training.py --task classification --refresh 5
```

**Weights & Biases Dashboard:**
- Visit: https://wandb.ai/your-username/slicewise-classification-production
- View: Loss curves, metrics, learning rate, system stats

**TensorBoard (Alternative):**
```bash
tensorboard --logdir runs/cls_production
```

### Step 6: Evaluation

After training completes, evaluate the model:

```bash
# Classification
python scripts/evaluate_classifier.py \
    --checkpoint checkpoints/cls_production/best_model.pth \
    --output-dir outputs/classification_production/evaluation

# Segmentation
python scripts/evaluate_segmentation.py \
    --checkpoint checkpoints/seg_production/best_model.pth \
    --output-dir outputs/seg_production/evaluation
```

### Step 7: Generate Visualizations

**Grad-CAM (Classification):**
```bash
python scripts/generate_gradcam.py \
    --num-samples 50 \
    --output-dir visualizations/classification_production/gradcam
```

**Segmentation Overlays:**
```bash
python scripts/visualize_segmentation_results.py \
    --num-samples 50 \
    --output-dir visualizations/seg_production/predictions
```

### Step 8: Model Calibration

Improve confidence scores:

```bash
python scripts/calibrate_classifier.py
```

---

## Monitoring Training

### Real-Time Monitoring Script

The `monitor_training.py` script provides live visualization:

**Features:**
- 📊 Loss curves (train & validation)
- 📈 Metric curves (ROC-AUC / Dice)
- 📉 Learning rate schedule
- 📋 Current statistics
- 🔄 Auto-refresh every N seconds

**Usage:**
```bash
# Classification
python scripts/monitor_training.py --task classification --refresh 5

# Segmentation
python scripts/monitor_training.py --task segmentation --refresh 10

# Single snapshot (no live updates)
python scripts/monitor_training.py --task classification --no-live
```

### What to Watch For

**Good Signs:**
- ✅ Smooth loss decrease
- ✅ Val loss following train loss closely
- ✅ Metrics steadily improving
- ✅ No sudden spikes or divergence

**Warning Signs:**
- ⚠️ Val loss increasing while train loss decreases (overfitting)
- ⚠️ Loss not decreasing after many epochs (learning rate too low)
- ⚠️ Loss exploding (learning rate too high, gradient issues)
- ⚠️ Metrics plateauing early (need more data/augmentation)

### Checkpoints

Models are automatically saved:

**Classification:**
```
checkpoints/cls_production/
├── best_model.pth          # Best validation ROC-AUC
├── last_model.pth          # Most recent epoch
├── epoch_005.pth           # Saved every 5 epochs
├── epoch_010.pth
└── ...
```

**Segmentation:**
```
checkpoints/seg_production/
├── best_model.pth          # Best validation Dice
├── last_model.pth          # Most recent epoch
├── epoch_005.pth
└── ...
```

---

## Evaluation & Visualization

### Comprehensive Evaluation

**Classification Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix
- Per-class metrics
- Calibration (ECE, Brier score)

**Segmentation Metrics:**
- Dice Coefficient
- IoU (Intersection over Union)
- Precision, Recall, F1
- Specificity
- Hausdorff Distance
- Boundary F-measure

### Visualization Outputs

**Classification:**
- `visualizations/classification_production/`
  - `gradcam/` - Grad-CAM heatmaps showing model attention
  - `confusion_matrix.png` - Confusion matrix
  - `roc_curve.png` - ROC curve
  - `pr_curve.png` - Precision-Recall curve
  - `calibration_plot.png` - Reliability diagram

**Segmentation:**
- `visualizations/seg_production/`
  - `predictions/` - Segmentation overlays
  - `dice_distribution.png` - Dice score distribution
  - `examples/` - Best/worst predictions

---

## Best Practices

### 🎯 Training Tips

1. **Start with Quick Test**
   ```bash
   python scripts/train_production.py --task classification --quick-test
   ```
   Verify everything works before committing to 100 epochs.

2. **Monitor GPU Usage**
   ```bash
   watch -n 1 nvidia-smi
   ```
   Ensure GPU is being utilized (>80% utilization is good).

3. **Use Mixed Precision**
   Already enabled in production configs. Speeds up training by 2-3x.

4. **Enable Gradient Accumulation**
   If running out of GPU memory, increase `gradient_accumulation_steps` in config.

5. **Save Checkpoints Frequently**
   Default: every 5 epochs. Adjust `save_frequency` in config.

### 📊 Hyperparameter Tuning

**If validation loss plateaus early:**
- Increase learning rate (e.g., 0.0003 → 0.0005)
- Reduce weight decay
- Add more augmentation

**If overfitting (val loss > train loss):**
- Increase weight decay (e.g., 0.01 → 0.05)
- Add dropout (increase from 0.3 → 0.5)
- Reduce model complexity
- Add more data augmentation

**If underfitting (both losses high):**
- Increase model capacity
- Reduce regularization
- Train longer
- Check data quality

### 💾 Resource Management

**GPU Memory:**
- Classification: ~4-6 GB
- Segmentation: ~8-12 GB
- Reduce batch size if OOM errors occur

**Disk Space:**
- Data: ~30 GB
- Checkpoints: ~5-10 GB
- Logs: ~1-2 GB
- Total: ~50 GB recommended

**Training Time Estimates:**

| Task | GPU (RTX 3090) | GPU (GTX 1080) | CPU |
|------|----------------|----------------|-----|
| Classification (100 epochs) | 2-3 hours | 4-6 hours | 24-36 hours |
| Segmentation (100 epochs) | 4-6 hours | 8-12 hours | 48-72 hours |

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```yaml
# In config file, reduce batch size
training:
  batch_size: 16  # Try 8, 4, or even 2
  
# Or enable gradient accumulation
training:
  gradient_accumulation_steps: 4  # Effective batch = batch_size * 4
```

#### 2. Loss is NaN

**Error:** Loss becomes NaN during training

**Solutions:**
- Reduce learning rate (e.g., 0.0003 → 0.0001)
- Enable gradient clipping (already enabled: `grad_clip: 1.0`)
- Check for corrupted data
- Disable mixed precision temporarily: `use_amp: false`

#### 3. Training Too Slow

**Solutions:**
- Enable mixed precision: `use_amp: true`
- Increase `num_workers` for data loading
- Enable `pin_memory: true`
- Use `persistent_workers: true`
- Reduce augmentation complexity

#### 4. Validation Loss Not Improving

**Solutions:**
- Increase learning rate
- Reduce early stopping patience
- Check if data is properly shuffled
- Verify train/val split is correct

#### 5. Model Not Learning

**Symptoms:** Loss stays constant

**Solutions:**
- Check learning rate (might be too low)
- Verify data labels are correct
- Check if model is frozen: `freeze_backbone: false`
- Increase warmup epochs

### Debug Mode

Enable verbose logging:

```python
# In training script, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Advanced Topics

### Resume Training from Checkpoint

```bash
python scripts/train_production.py \
    --task classification \
    --resume checkpoints/cls_production/epoch_050.pth
```

### Custom Configuration

Create a custom config:

```bash
# Copy production config
cp configs/config_cls_production.yaml configs/my_custom_config.yaml

# Edit as needed
nano configs/my_custom_config.yaml

# Train with custom config
python scripts/train_classifier.py --config configs/my_custom_config.yaml
```

### Multi-GPU Training

```yaml
# In config file
hardware:
  device: "cuda"
  gpu_ids: [0, 1, 2, 3]  # Use multiple GPUs
```

### Experiment Tracking

**W&B Sweeps for Hyperparameter Search:**

```yaml
# sweep.yaml
program: scripts/train_classifier.py
method: bayes
metric:
  name: val_roc_auc
  goal: maximize
parameters:
  lr:
    min: 0.0001
    max: 0.001
  weight_decay:
    values: [0.001, 0.01, 0.1]
  dropout:
    values: [0.2, 0.3, 0.4, 0.5]
```

```bash
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

### Transfer Learning

Fine-tune on your own dataset:

```yaml
# In config file
model:
  pretrained: true
  freeze_backbone: true  # Freeze encoder initially
  
training:
  epochs: 20  # First phase: train classifier only
  
# Then unfreeze and fine-tune
model:
  freeze_backbone: false
  
training:
  epochs: 50  # Second phase: fine-tune entire model
  optimizer:
    lr: 0.00001  # Lower learning rate for fine-tuning
```

---

## Summary

You now have everything needed to train production-quality models:

✅ **Optimized Configs**: 100+ epochs with advanced techniques  
✅ **Automated Pipeline**: One command to train everything  
✅ **Real-time Monitoring**: Live plots and metrics  
✅ **Comprehensive Evaluation**: Metrics, visualizations, calibration  
✅ **Best Practices**: Tips for optimal performance  
✅ **Troubleshooting**: Solutions to common issues  

### Next Steps

1. **Start Training:**
   ```bash
   python scripts/train_production.py --task both --epochs 100
   ```

2. **Monitor Progress:**
   ```bash
   python scripts/monitor_training.py --task classification
   ```

3. **Evaluate Results:**
   - Check W&B dashboard
   - View visualizations in `visualizations/`
   - Review metrics in `outputs/`

4. **Deploy Models:**
   ```bash
   python scripts/run_demo.py
   ```

---

## Support

**Issues?** Check the troubleshooting section or open an issue on GitHub.

**Questions?** Refer to the [FULL-PLAN.md](FULL-PLAN.md) for project architecture.

**Want to contribute?** See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

**Happy Training! 🚀**
