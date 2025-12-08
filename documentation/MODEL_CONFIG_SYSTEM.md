# Model Configuration Management System

**Purpose**: Prevent architecture mismatches between training and inference by storing model architecture parameters alongside checkpoints.

---

## 🎯 Problem Solved

Previously, model architecture parameters (`base_filters`, `depth`) were hardcoded in inference scripts, leading to mismatches when loading checkpoints trained with different architectures.

**Before** ❌:
```python
# Training: base_filters=64, depth=4
# Inference: base_filters=32, depth=3  ← MISMATCH!
predictor = MultiTaskPredictor(checkpoint_path, base_filters=32, depth=3)
# RuntimeError: size mismatch...
```

**After** ✅:
```python
# Automatically loads correct architecture from model_config.json
predictor = MultiTaskPredictor(checkpoint_path)  # No parameters needed!
```

---

## 📁 File Structure

```
checkpoints/
└── multitask_joint/
    ├── best_model.pth           # Model weights
    ├── model_config.json        # Architecture configuration ← NEW!
    └── last_model.pth
```

---

## 🔧 Usage

### 1. **Generate Config Files for Existing Checkpoints**

Run this once to create `model_config.json` for all your trained models:

```bash
python scripts/generate_model_configs.py
```

**Output**:
```
================================================================================
Generating Model Configuration Files
================================================================================

[OK] Generated config for: multitask_seg_warmup
  Location: checkpoints/multitask_seg_warmup/model_config.json
  Architecture: base_filters=64, depth=4

[OK] Generated config for: multitask_cls_head
  Location: checkpoints/multitask_cls_head/model_config.json
  Architecture: base_filters=64, depth=4

[OK] Generated config for: multitask_joint
  Location: checkpoints/multitask_joint/model_config.json
  Architecture: base_filters=64, depth=4

================================================================================
✅ Generated 3 configuration files
================================================================================
```

---

### 2. **During Training** (Future Models)

Save model with config:

```python
from src.models.model_config import ModelConfig, save_model_with_config

# Create config
config = ModelConfig(
    base_filters=64,
    depth=4,
    in_channels=1,
    seg_out_channels=1,
    cls_num_classes=2,
    description="Multi-task model for brain tumor detection",
    trained_on="BraTS 2020 + Kaggle Brain MRI"
)

# Save model + config together
save_model_with_config(
    model=model,
    checkpoint_path=checkpoint_dir / "best_model.pth",
    config=config,
    optimizer=optimizer,
    epoch=epoch,
    metrics={"dice": 0.88, "accuracy": 0.90}
)
```

This creates:
- `checkpoints/multitask_joint/best_model.pth`
- `checkpoints/multitask_joint/model_config.json` ← Automatically!

---

### 3. **During Inference** (Automatic)

The predictor now automatically loads the config:

```python
from src.inference.multi_task_predictor import MultiTaskPredictor

# Option 1: Auto-detect architecture (RECOMMENDED)
predictor = MultiTaskPredictor(
    checkpoint_path="checkpoints/multitask_joint/best_model.pth"
)
# [OK] Loaded model config from: checkpoints/multitask_joint/model_config.json
# Using architecture: base_filters=64, depth=4

# Option 2: Override if needed
predictor = MultiTaskPredictor(
    checkpoint_path="checkpoints/multitask_joint/best_model.pth",
    base_filters=32,  # Override
    depth=3           # Override
)
```

---

## 📋 Model Config Format

**`model_config.json`**:
```json
{
  "base_filters": 64,
  "depth": 4,
  "in_channels": 1,
  "seg_out_channels": 1,
  "cls_num_classes": 2,
  "model_type": "multitask",
  "version": "1.0",
  "description": "Multi-task model - Joint fine-tuning (FINAL)",
  "trained_on": "BraTS 2020 + Kaggle Brain MRI (both tasks)",
  "training_config": "configs/multitask_joint_quick_test.yaml"
}
```

---

## 🔄 Migration Guide

### For Existing Code

**Before**:
```python
# Old way - hardcoded parameters
predictor = MultiTaskPredictor(
    checkpoint_path="checkpoints/multitask_joint/best_model.pth",
    base_filters=64,
    depth=4
)
```

**After**:
```python
# New way - auto-detect
predictor = MultiTaskPredictor(
    checkpoint_path="checkpoints/multitask_joint/best_model.pth"
)
```

### Files to Update

1. ✅ `scripts/test_multitask_e2e.py` - Already updated
2. ✅ `app/backend/main_v2.py` - Already updated
3. ✅ `scripts/test_backend_startup.py` - Already updated

**Simply remove the `base_filters` and `depth` parameters!**

---

## 🎯 Benefits

1. **No More Mismatches**: Architecture is always correct
2. **Self-Documenting**: Config file shows model details
3. **Version Tracking**: Know what config was used for training
4. **Easy Debugging**: Clear error messages if config missing
5. **Future-Proof**: New models automatically include config

---

## 🔍 API Reference

### `ModelConfig` Class

```python
from src.models.model_config import ModelConfig

# Create config
config = ModelConfig(
    base_filters=64,
    depth=4,
    in_channels=1,
    seg_out_channels=1,
    cls_num_classes=2
)

# Save to file
config.save("checkpoints/my_model/model_config.json")

# Load from file
config = ModelConfig.load("checkpoints/my_model/model_config.json")

# Load from checkpoint directory
config = ModelConfig.from_checkpoint_dir("checkpoints/my_model")

# Get model parameters
params = config.get_model_params()
# {'base_filters': 64, 'depth': 4, ...}
```

### Helper Functions

```python
from src.models.model_config import save_model_with_config, load_model_with_config

# Save model + config
save_model_with_config(
    model=model,
    checkpoint_path="checkpoints/my_model/best.pth",
    config=config,
    optimizer=optimizer,
    epoch=10,
    metrics={"accuracy": 0.95}
)

# Load model + config
model, config, checkpoint = load_model_with_config(
    checkpoint_path="checkpoints/my_model/best.pth",
    model_factory_fn=create_multi_task_model,
    device='cuda'
)
```

---

## ⚠️ Troubleshooting

### Config File Not Found

If `model_config.json` doesn't exist:

```
⚠ Model config not found at checkpoints/multitask_joint/model_config.json
  Using default configuration (base_filters=64, depth=4)
```

**Solution**: Run `python scripts/generate_model_configs.py`

### Architecture Mismatch

If you get size mismatch errors:

1. Check if `model_config.json` exists
2. Verify the config matches your checkpoint
3. Regenerate config if needed

---

## 📚 Related Files

- **Core Module**: `src/models/model_config.py`
- **Generator Script**: `scripts/generate_model_configs.py`
- **Predictor**: `src/inference/multi_task_predictor.py` (auto-loads config)
- **Training Scripts**: Will be updated to save config automatically

---

## ✅ Checklist

- [x] Create `ModelConfig` class
- [x] Create config generator script
- [x] Update `MultiTaskPredictor` to auto-load config
- [x] Generate configs for existing checkpoints
- [x] Update all inference scripts
- [x] Document the system

---

**Status**: ✅ **COMPLETE** - All existing checkpoints now have config files, and all inference code auto-detects architecture!

**Next**: Run `python scripts/generate_model_configs.py` to create config files for your trained models.
