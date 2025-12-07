# Full Pipeline Controller Guide

**Complete End-to-End Multi-Task Training Pipeline**

This guide explains how to use the `run_full_pipeline.py` controller script to orchestrate the complete training and deployment pipeline for the SliceWise Multi-Task Brain Tumor Detection model.

---

## 🎯 Overview

The pipeline controller automates the entire workflow:

1. **Data Download** - BraTS 2020 (~15GB) + Kaggle (~500MB)
2. **Data Preprocessing** - 3D→2D conversion, normalization, filtering
3. **Data Splitting** - Patient-level train/val/test splits (70/15/15)
4. **Multi-Task Training** - 3-stage training strategy:
   - Stage 1: Segmentation warm-up (encoder + decoder)
   - Stage 2: Classification head training (frozen encoder)
   - Stage 3: Joint fine-tuning (all components)
5. **Comprehensive Evaluation** - Metrics, Grad-CAM, phase comparison
6. **Demo Application** - FastAPI backend + Streamlit frontend

---

## 🚀 Quick Start

### Option 1: Full Pipeline (Recommended for First Run)

```bash
# Quick test (10 patients, 5 epochs, ~30 minutes)
python scripts/run_full_pipeline.py --mode full --training-mode quick

# Baseline training (100 patients, 50 epochs, ~2-4 hours)
python scripts/run_full_pipeline.py --mode full --training-mode baseline

# Production training (988 patients, 100 epochs, ~8-12 hours)
python scripts/run_full_pipeline.py --mode full --training-mode production
```

### Option 2: Skip Data Download (If Already Downloaded)

```bash
python scripts/run_full_pipeline.py --mode full --training-mode baseline --skip-download
```

### Option 3: Partial Pipeline

```bash
# Only data preparation
python scripts/run_full_pipeline.py --mode data-only --training-mode production

# Only training and evaluation (requires prepared data)
python scripts/run_full_pipeline.py --mode train-eval --training-mode baseline

# Only demo application (requires trained model)
python scripts/run_full_pipeline.py --mode demo
```

---

## 📋 Pipeline Modes

### `--mode` Options

| Mode | Description | Steps | Use Case |
|------|-------------|-------|----------|
| `full` | Complete pipeline | 1-6 | First-time setup, full training |
| `data-only` | Data preparation only | 1-3 | Prepare data for later training |
| `train-eval` | Training + evaluation | 4-5 | Re-train with different hyperparameters |
| `demo` | Demo application only | 6 | Test trained model |

### `--training-mode` Options

| Mode | Patients | Epochs | Duration | Use Case |
|------|----------|--------|----------|----------|
| `quick` | 10 | 5 | ~30 min | Quick test, debugging |
| `baseline` | 100 | 50 | ~2-4 hrs | Baseline experiments |
| `production` | 988 (all) | 100 | ~8-12 hrs | Final production model |

---

## 📊 Expected Timeline

### Quick Mode (~30 minutes total)
- Data Download: 5-10 min (if not cached)
- Preprocessing: 2-3 min (10 patients)
- Splitting: <1 min
- Training Stage 1: 5-8 min
- Training Stage 2: 2-4 min
- Training Stage 3: 5-8 min
- Evaluation: 2-3 min
- Demo: Launches immediately

### Baseline Mode (~2-4 hours total)
- Data Download: 10-30 min (if not cached)
- Preprocessing: 15-30 min (100 patients)
- Splitting: 1-2 min
- Training Stage 1: 45-90 min
- Training Stage 2: 20-40 min
- Training Stage 3: 45-90 min
- Evaluation: 10-15 min
- Demo: Launches immediately

### Production Mode (~8-12 hours total)
- Data Download: 10-30 min (if not cached)
- Preprocessing: 2-4 hours (988 patients)
- Splitting: 5-10 min
- Training Stage 1: 4-6 hours
- Training Stage 2: 2-3 hours
- Training Stage 3: 4-6 hours
- Evaluation: 30-60 min
- Demo: Launches immediately

---

## 🔧 Configuration Files

The controller automatically selects the appropriate config files:

### Quick Mode
- `configs/multitask_seg_warmup_quick_test.yaml` (5 epochs)
- `configs/multitask_cls_head_quick_test.yaml` (5 epochs)
- `configs/multitask_joint_quick_test.yaml` (5 epochs)

### Baseline Mode
- `configs/multitask_seg_warmup.yaml` (50 epochs)
- `configs/multitask_cls_head_quick_test.yaml` (5 epochs)
- `configs/multitask_joint_quick_test.yaml` (10 epochs)

### Production Mode
- `configs/multitask_seg_warmup_production.yaml` (100 epochs)
- `configs/multitask_cls_head_production.yaml` (50 epochs)
- `configs/multitask_joint_production.yaml` (50 epochs)

---

## 📁 Output Structure

After running the pipeline, you'll have:

```
project_root/
├── data/
│   ├── raw/
│   │   ├── brats2020/          # Downloaded BraTS data
│   │   └── kaggle/             # Downloaded Kaggle data
│   └── processed/
│       ├── brats2d_full/       # Preprocessed BraTS (2D slices)
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── kaggle/             # Split Kaggle data
│           ├── train/
│           ├── val/
│           └── test/
├── checkpoints/
│   ├── multitask_seg_warmup/   # Stage 1 checkpoints
│   ├── multitask_cls_head/     # Stage 2 checkpoints
│   └── multitask_joint/        # Stage 3 checkpoints (FINAL MODEL)
├── results/
│   ├── multitask_evaluation/   # Test set metrics
│   └── phase_comparison/       # Phase comparison results
├── visualizations/
│   └── multitask_gradcam/      # Grad-CAM visualizations
├── logs/
│   ├── multitask_seg_warmup/   # Training logs
│   ├── multitask_cls_head/
│   └── multitask_joint/
└── pipeline_results.json       # Pipeline execution summary
```

---

## 🎓 Training Strategy

### Stage 1: Segmentation Warm-Up
- **Goal**: Initialize encoder with good features
- **Dataset**: BraTS (with segmentation masks)
- **Frozen**: None
- **Trainable**: Encoder + Segmentation Decoder
- **Loss**: Dice + BCE
- **Duration**: Longest stage (~40% of total training time)

### Stage 2: Classification Head Training
- **Goal**: Train classification head on Kaggle data
- **Dataset**: Kaggle (tumor/no-tumor labels)
- **Frozen**: Encoder + Segmentation Decoder
- **Trainable**: Classification Head only
- **Loss**: Cross-Entropy
- **Duration**: Shortest stage (~20% of total training time)

### Stage 3: Joint Fine-Tuning
- **Goal**: Fine-tune all components together
- **Dataset**: BraTS + Kaggle (mixed)
- **Frozen**: None
- **Trainable**: All components (differential LR)
- **Loss**: Combined (Dice + BCE + Cross-Entropy)
- **Duration**: Medium stage (~40% of total training time)

---

## 📈 Expected Performance

### Quick Mode (10 patients, 5 epochs)
- **Classification Accuracy**: ~75-85%
- **Segmentation Dice**: ~0.60-0.70
- **Purpose**: Smoke test, debugging

### Baseline Mode (100 patients, 50 epochs)
- **Classification Accuracy**: ~85-90%
- **Segmentation Dice**: ~0.70-0.75
- **Purpose**: Baseline experiments, hyperparameter tuning

### Production Mode (988 patients, 100 epochs)
- **Classification Accuracy**: ~91-93%
- **Classification Sensitivity**: ~95-97%
- **Segmentation Dice**: ~0.75-0.80
- **Purpose**: Final production model

---

## 🔍 Monitoring Progress

### Real-Time Monitoring

The controller prints colored progress updates:
- 🔵 **Blue**: Info messages
- 🟢 **Green**: Success messages
- 🟡 **Yellow**: Warnings
- 🔴 **Red**: Errors

### Weights & Biases (W&B)

If W&B is enabled in configs, you can monitor training in real-time:

```bash
# Login to W&B (first time only)
wandb login

# View training dashboard
# Go to: https://wandb.ai/<your-username>/slicewise-multitask-production
```

### Pipeline Results

After completion, check `pipeline_results.json`:

```json
{
  "start_time": "2025-12-07T10:00:00",
  "end_time": "2025-12-07T14:30:00",
  "mode": "full",
  "training_mode": "baseline",
  "success": true,
  "total_duration_seconds": 16200,
  "steps": {
    "download_brats": {"status": "success", "duration_seconds": 1200},
    "download_kaggle": {"status": "success", "duration_seconds": 180},
    "preprocess_brats": {"status": "success", "duration_seconds": 1800},
    "split_brats": {"status": "success", "duration_seconds": 30},
    "split_kaggle": {"status": "success", "duration_seconds": 15},
    "train_stage1_seg_warmup": {"status": "success", "duration_seconds": 5400},
    "train_stage2_cls_head": {"status": "success", "duration_seconds": 2400},
    "train_stage3_joint": {"status": "success", "duration_seconds": 5400},
    "evaluate_multitask": {"status": "success", "duration_seconds": 600},
    "generate_gradcam": {"status": "success", "duration_seconds": 300},
    "compare_phases": {"status": "success", "duration_seconds": 900}
  }
}
```

---

## 🛠️ Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size in config files

```yaml
# Edit configs/multitask_*_production.yaml
training:
  batch_size: 8  # Reduce from 16
```

### Issue: Data Download Fails

**Solution**: Check Kaggle API credentials

```bash
# Ensure ~/.kaggle/kaggle.json exists with valid credentials
cat ~/.kaggle/kaggle.json
```

### Issue: Training Timeout

**Solution**: Increase timeout or use smaller dataset

```bash
# Use baseline mode instead of production
python scripts/run_full_pipeline.py --mode full --training-mode baseline
```

### Issue: CUDA Out of Memory

**Solution**: Use mixed precision or smaller model

```yaml
# Configs already use AMP, but you can also reduce model size
model:
  base_filters: 32  # Reduce from 64
  depth: 3          # Reduce from 4
```

### Issue: Demo Won't Launch

**Solution**: Check if model checkpoint exists

```bash
# Verify checkpoint exists
ls checkpoints/multitask_joint/best_model.pth

# If missing, run training first
python scripts/run_full_pipeline.py --mode train-eval --training-mode quick
```

---

## 🎯 Best Practices

### 1. Start Small, Scale Up

```bash
# First: Quick test to verify everything works
python scripts/run_full_pipeline.py --mode full --training-mode quick

# Then: Baseline for experiments
python scripts/run_full_pipeline.py --mode full --training-mode baseline --skip-download

# Finally: Production for final model
python scripts/run_full_pipeline.py --mode full --training-mode production --skip-download
```

### 2. Use Checkpoints Wisely

- Keep Stage 1 checkpoint for reuse
- Experiment with Stage 2/3 hyperparameters
- Save best models separately

### 3. Monitor GPU Usage

```bash
# In another terminal, monitor GPU
watch -n 1 nvidia-smi
```

### 4. Use tmux/screen for Long Runs

```bash
# Start tmux session
tmux new -s training

# Run pipeline
python scripts/run_full_pipeline.py --mode full --training-mode production

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

---

## 📚 Additional Resources

- **Full Documentation**: `documentation/CONSOLIDATED_DOCUMENTATION.md`
- **Scripts Reference**: `SCRIPTS_REFERENCE.md`
- **Multi-Task Architecture**: `documentation/PHASE1_COMPLETE.md`
- **Training Details**: `documentation/MULTITASK_EVALUATION_REPORT.md`
- **Demo Guide**: `documentation/PHASE6_COMPLETE.md`

---

## 🤝 Support

If you encounter issues:

1. Check `pipeline_results.json` for error details
2. Review logs in `logs/multitask_*/`
3. Check W&B dashboard for training curves
4. Verify data integrity with `scripts/debug/debug_multitask_data.py`

---

**Happy Training! 🚀**

*SliceWise Team - December 7, 2025*
