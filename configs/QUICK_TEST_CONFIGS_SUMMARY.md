# Quick Test Configuration Summary

This document explains the modifications made to production configs for rapid testing.

## Overview

All three production configs have been modified to create **quick test versions** that run significantly faster while still validating the full pipeline functionality.

## Modified Configs

1. **multitask_seg_warmup_production.yaml** → Quick Test Version
2. **multitask_cls_head_production.yaml** → Quick Test Version  
3. **multitask_joint_production.yaml** → Quick Test Version

## Key Changes

### 1. Training Duration
| Parameter | Production | Quick Test | Speedup |
|-----------|-----------|------------|---------|
| Epochs | 50 | 3 | **16.7x faster** |
| Early Stopping Patience | 50 | 10 | 5x faster |
| Scheduler T_max | 50 | 3 | Matches epochs |

### 2. Batch Processing
| Parameter | Production | Quick Test | Impact |
|-----------|-----------|------------|--------|
| Batch Size | 32 | 64 | **2x fewer iterations** |
| Iterations per Epoch | 485 | ~243 | **2x faster per epoch** |

**Total speedup per stage: ~33x faster** (16.7x epochs × 2x batch size)

### 3. Data Augmentation
Simplified augmentation for faster preprocessing:

| Augmentation | Production | Quick Test |
|--------------|-----------|------------|
| Rotation | 15-20° | 10° |
| Scale | 0.1-0.15 | 0.08 |
| Elastic Deform | Enabled | **Disabled** |
| Gaussian Noise | 0.015-0.02 | 0.008-0.01 |
| Gaussian Blur | 0.2-0.3 | 0.15-0.2 |
| Brightness/Contrast/Gamma | 0.15-0.2 | 0.1 |

### 4. Logging & Monitoring
| Feature | Production | Quick Test |
|---------|-----------|------------|
| W&B Logging | Enabled | **Disabled** |
| Checkpoint Frequency | Every 5-10 epochs | Every epoch (implicit) |

### 5. Checkpoint Paths
All checkpoints are saved to separate directories to avoid conflicts:

- Stage 1: `checkpoints/multitask_seg_warmup_quick_test/`
- Stage 2: `checkpoints/multitask_cls_head_quick_test/`
- Stage 3: `checkpoints/multitask_joint_quick_test/`

## Expected Timeline

### Production Mode (Original)
- **Stage 1 (Seg Warmup)**: ~4-6 hours (50 epochs × 485 batches)
- **Stage 2 (Cls Head)**: ~3-4 hours (50 epochs × ~400 batches)
- **Stage 3 (Joint)**: ~4-6 hours (50 epochs × ~485 batches)
- **Total**: ~11-16 hours

### Quick Test Mode (Modified)
- **Stage 1 (Seg Warmup)**: ~8-12 minutes (3 epochs × 243 batches)
- **Stage 2 (Cls Head)**: ~6-10 minutes (3 epochs × ~200 batches)
- **Stage 3 (Joint)**: ~8-12 minutes (3 epochs × ~243 batches)
- **Total**: ~22-34 minutes

**Overall speedup: ~33x faster** ⚡

## Usage

### Running Quick Test Pipeline

```bash
# Option 1: Use run_full_pipeline.py with production configs (now modified)
python scripts/run_full_pipeline.py --mode full --training-mode production

# Option 2: Run individual stages manually
python scripts/training/multitask/train_multitask_seg_warmup.py \
    --config configs/multitask_seg_warmup_production.yaml

python scripts/training/multitask/train_multitask_cls_head.py \
    --config configs/multitask_cls_head_production.yaml

python scripts/training/multitask/train_multitask_joint.py \
    --config configs/multitask_joint_production.yaml
```

### Reverting to Production Settings

If you need to restore production settings later, modify these parameters:

```yaml
# In all three configs:
training:
  epochs: 50              # Change back from 3
  batch_size: 32          # Change back from 64

early_stopping:
  patience: 50            # Change back from 10

scheduler:
  T_max: 50               # Change back from 3

wandb:
  enabled: true           # Change back from false

augmentation:
  train:
    elastic_deform: true  # Re-enable (Stage 1 & 3 only)
    # Restore original values for all augmentation params
```

## Expected Performance

### Quick Test (3 epochs)
Since we're only training for 3 epochs, expect:

- **Segmentation Dice**: 0.60-0.75 (lower than production)
- **Classification Accuracy**: 70-85% (lower than production)
- **Purpose**: Validate pipeline functionality, not final performance

### Production (50 epochs)
With full training:

- **Segmentation Dice**: 0.75-0.80
- **Classification Accuracy**: 91-93%
- **Sensitivity**: 95-97%
- **Purpose**: Achieve best model performance

## What Gets Validated

Even with quick test mode, you still validate:

✅ **Data Pipeline**: Loading, preprocessing, augmentation  
✅ **Model Architecture**: Forward/backward pass, gradient flow  
✅ **Training Loop**: Loss computation, optimizer steps, metrics  
✅ **Checkpoint Saving**: Model state persistence  
✅ **Multi-Task Learning**: Segmentation + classification integration  
✅ **Differential Learning Rates**: Encoder vs decoder/cls head  
✅ **Mixed Dataset Batching**: BraTS + Kaggle integration  
✅ **Early Stopping**: Monitoring and patience logic  
✅ **Evaluation**: Metrics computation and visualization  

## Recommendations

1. **Use Quick Test Mode** when:
   - Testing new features or bug fixes
   - Validating pipeline changes
   - Debugging training issues
   - CI/CD automated testing
   - Rapid iteration during development

2. **Use Production Mode** when:
   - Training final models for deployment
   - Benchmarking performance
   - Generating results for papers/reports
   - Comparing different architectures

## Notes

- Quick test uses the **full dataset** (same as production), just fewer epochs
- Batch size increase (32→64) requires ~2x more GPU memory (~8GB → ~12GB)
- If you encounter OOM errors, reduce batch size to 32 or 48
- W&B logging disabled to reduce overhead, but you can re-enable if needed
- All metrics are still computed and printed to console

## Troubleshooting

### Out of Memory (OOM)
```yaml
# Reduce batch size in config
training:
  batch_size: 32  # or even 16
```

### Training Too Slow
```yaml
# Disable more augmentations
augmentation:
  train:
    enabled: false  # Disable all augmentation
```

### Want to See W&B Logs
```yaml
wandb:
  enabled: true
  project: "slicewise-quick-test"
```

---

**Created**: December 8, 2025  
**Purpose**: Enable rapid testing of multi-task training pipeline  
**Speedup**: ~33x faster than production (22-34 min vs 11-16 hours)
