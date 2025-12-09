# Configuration System - Hierarchical Config Architecture

This directory contains the hierarchical configuration system for the SliceWise Multi-Task Brain Tumor Detection project. The system eliminates duplication and ensures consistency across all training configurations.

## 📁 Directory Structure

```
configs/
├── base/                      # Base configurations (shared foundation)
│   ├── common.yaml           # Common settings (seed, device, logging)
│   ├── model_architectures.yaml  # Model architecture presets
│   ├── training_defaults.yaml    # Training hyperparameters
│   ├── augmentation_presets.yaml # Augmentation strategies
│   └── platform_overrides.yaml   # Platform-specific settings
│
├── stages/                    # Stage-specific configurations
│   ├── stage1_seg_warmup.yaml    # Segmentation warm-up
│   ├── stage2_cls_head.yaml      # Classification head training
│   └── stage3_joint.yaml         # Joint fine-tuning
│
├── modes/                     # Mode-specific overrides
│   ├── quick_test.yaml       # Quick test (3 epochs, minimal aug)
│   ├── baseline.yaml         # Baseline (50 epochs, moderate aug)
│   └── production.yaml       # Production (100 epochs, aggressive aug)
│
├── final/                     # Generated configs (gitignored)
│   ├── stage1_quick.yaml     # Auto-generated from base+stage+mode
│   ├── stage1_baseline.yaml
│   ├── stage1_production.yaml
│   ├── stage2_quick.yaml
│   ├── stage2_baseline.yaml
│   ├── stage2_production.yaml
│   ├── stage3_quick.yaml
│   ├── stage3_baseline.yaml
│   └── stage3_production.yaml
│
└── multitask-model/           # Legacy configs (deprecated, kept for reference)
    └── ... (8 old config files)
```

## 🎯 How It Works

### 1. Hierarchical Merging
Configs are merged in this order (later overrides earlier):
```
base/common.yaml
  ↓
base/training_defaults.yaml
  ↓
stages/stageN_*.yaml
  ↓
modes/MODE.yaml
  ↓
final/stageN_MODE.yaml (generated)
```

### 2. Reference Resolution
The merger tool automatically resolves references:

**Architecture References:**
```yaml
# In stage config
model:
  architecture: "multitask_medium"

# Expands to (from base/model_architectures.yaml)
model:
  in_channels: 1
  seg_out_channels: 1
  cls_num_classes: 2
  base_filters: 32
  depth: 4
  cls_hidden_dim: 256
  cls_dropout: 0.5
```

**Augmentation Presets:**
```yaml
# In mode config
augmentation:
  preset: "moderate"

# Expands to (from base/augmentation_presets.yaml)
augmentation:
  train:
    enabled: true
    random_flip_h: 0.5
    random_flip_v: 0.5
    random_rotate: 15
    # ... (full augmentation config)
```

## 🚀 Usage

### Generate Configs

**Generate all 9 configs:**
```bash
python scripts/utils/merge_configs.py --all
```

**Generate single config:**
```bash
python scripts/utils/merge_configs.py --stage 1 --mode quick
python scripts/utils/merge_configs.py --stage 2 --mode production
```

**Custom output directory:**
```bash
python scripts/utils/merge_configs.py --all --output-dir configs/generated
```

### Use in Training

The generated configs are automatically used by training scripts:

```bash
# Quick test (uses configs/final/stage1_quick.yaml)
python scripts/run_full_pipeline.py --mode full --training-mode quick

# Baseline (uses configs/final/stage2_baseline.yaml)
python scripts/training/multitask/train_multitask_cls_head.py

# Production (uses configs/final/stage3_production.yaml)
python scripts/run_full_pipeline.py --mode full --training-mode production
```

## 📝 Configuration Levels

### Base Configs (Shared Foundation)

#### `base/common.yaml`
- Seed: 42
- Device: "cuda"
- CuDNN settings
- Logging format
- Checkpoint defaults

#### `base/model_architectures.yaml`
Three architecture presets:
- **multitask_small**: 16 base filters, depth 3, 128 hidden dim
- **multitask_medium**: 32 base filters, depth 4, 256 hidden dim (default)
- **multitask_large**: 64 base filters, depth 5, 512 hidden dim

#### `base/training_defaults.yaml`
- num_workers: 0 (Windows compatible)
- use_amp: true (mixed precision)
- grad_clip: 1.0
- Optimizer: AdamW (lr, weight_decay, betas)
- Scheduler: Cosine annealing
- Early stopping defaults

#### `base/augmentation_presets.yaml`
Three augmentation strategies:
- **minimal**: Basic flips/rotations (quick testing)
- **moderate**: Standard augmentations (baseline)
- **aggressive**: Heavy augmentations (production)

### Stage Configs (Task-Specific)

#### `stages/stage1_seg_warmup.yaml`
- **Purpose**: Initialize encoder with segmentation task
- **Data**: BraTS only
- **Loss**: Dice + BCE (0.6/0.4 ratio)
- **Monitor**: val_dice (maximize)

#### `stages/stage2_cls_head.yaml`
- **Purpose**: Train classification head with frozen encoder
- **Data**: BraTS + Kaggle
- **Loss**: Cross-entropy with label smoothing (0.1)
- **Init**: Load encoder from Stage 1
- **Monitor**: val_acc (maximize)

#### `stages/stage3_joint.yaml`
- **Purpose**: Fine-tune all components jointly
- **Data**: BraTS + Kaggle (30%/70% mix)
- **Loss**: Multi-task (seg + cls, λ=1.0/0.5)
- **Optimizer**: Differential LR (encoder=1e-4, heads=3e-4)
- **Init**: Load from Stage 2
- **Monitor**: val_combined (maximize)

### Mode Configs (Training Intensity)

#### `modes/quick_test.yaml`
- **Epochs**: 3
- **Batch size**: 64
- **Augmentation**: minimal
- **Early stopping**: disabled
- **W&B**: disabled
- **Time**: ~30 minutes

#### `modes/baseline.yaml`
- **Epochs**: 50
- **Batch size**: 16
- **Augmentation**: moderate
- **Early stopping**: patience 15
- **W&B**: enabled
- **Time**: ~2-4 hours

#### `modes/production.yaml`
- **Epochs**: 100
- **Batch size**: 32
- **Augmentation**: aggressive
- **Early stopping**: patience 20
- **W&B**: enabled (with gradient logging)
- **Time**: ~8-12 hours

## 🔧 Modifying Configs

### To Change a Common Setting
Edit `base/common.yaml` or `base/training_defaults.yaml`, then regenerate:
```bash
python scripts/utils/merge_configs.py --all
```

### To Add a New Architecture
Add to `base/model_architectures.yaml`:
```yaml
architectures:
  multitask_xlarge:
    in_channels: 1
    seg_out_channels: 1
    cls_num_classes: 2
    base_filters: 128
    depth: 6
    cls_hidden_dim: 1024
    cls_dropout: 0.5
```

Then reference it in stage configs:
```yaml
model:
  architecture: "multitask_xlarge"
```

### To Add a New Training Mode
Create `modes/custom.yaml`:
```yaml
mode: "custom"
training:
  epochs: 75
  batch_size: 24
augmentation:
  preset: "moderate"
wandb:
  enabled: true
```

Then generate:
```bash
python scripts/utils/merge_configs.py --stage 1 --mode custom
```

## 📊 Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total config lines** | ~1,100 | ~400 base + 9 generated | **64% reduction** |
| **Duplication** | 70-90% | 0% | **Eliminated** |
| **Files to edit** | 8 files | 1 file | **87% less work** |
| **Consistency errors** | High risk | Zero risk | **Guaranteed** |
| **New mode creation** | Copy 3 files | Create 1 file | **67% faster** |

## 🔍 Metadata Tracking

Each generated config includes metadata for tracking:
```yaml
_metadata:
  stage: 1
  mode: quick
  generated_at: '2025-12-08T17:25:53.129832'
  generated_from:
    - base/common.yaml
    - base/training_defaults.yaml
    - stages/stage1_seg_warmup.yaml
    - modes/quick_test.yaml
```

## 🚨 Important Notes

1. **Never edit `configs/final/` directly** - These are auto-generated
2. **Always regenerate after base changes** - Run `merge_configs.py --all`
3. **CI/CD auto-generates** - Configs are built in GitHub Actions
4. **Legacy configs deprecated** - Use `configs/final/` instead of `configs/multitask-model/`

## 📚 Related Documentation

- [CONFIG_REFACTORING_PLAN.md](../docs/CONFIG_REFACTORING_PLAN.md) - Full refactoring plan
- [PIPELINE_CONTROLLER_GUIDE.md](../documentation/PIPELINE_CONTROLLER_GUIDE.md) - Pipeline usage
- [README.md](../README.md) - Main project documentation

## 🤝 Contributing

When adding new features that require config changes:
1. Update the appropriate base/stage/mode config
2. Regenerate all configs: `python scripts/utils/merge_configs.py --all`
3. Test with at least one training mode
4. Commit base configs (not generated ones)
