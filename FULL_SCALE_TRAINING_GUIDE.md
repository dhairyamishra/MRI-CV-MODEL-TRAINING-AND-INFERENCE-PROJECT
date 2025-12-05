# 🚀 Full-Scale BraTS Training Guide

Complete guide for training U-Net segmentation on the full BraTS dataset (988 patients).

---

## 📋 Quick Start

### Option 1: Full Production Run (Recommended)
```bash
# Complete pipeline: preprocess → train → evaluate
# Estimated time: 6-12 hours
python scripts/full_scale_brats_e2e.py
```

### Option 2: Quick Test (100 Patients)
```bash
# Test the pipeline with subset
# Estimated time: 2-3 hours
python scripts/full_scale_brats_e2e.py --num-patients 100
```

### Option 3: Skip Preprocessing (If Already Done)
```bash
# Only train and evaluate
python scripts/full_scale_brats_e2e.py --skip-preprocessing
```

---

## 🎯 What the Pipeline Does

### Step 1: Check Data ✅
- Verifies BraTS 2020 dataset exists in `data/raw/brats2020/`
- Counts patient directories
- Checks sample patient structure

### Step 2: Preprocess 🔄
- Converts 3D NIfTI volumes → 2D slices
- Extracts FLAIR modality
- Normalizes with z-score
- Filters empty slices (min 100 tumor pixels)
- Saves to `data/processed/brats2d_full/`

**Output:**
- ~25,000 slices from 988 patients
- Each slice: 256×256 .npz file
- Time: 2-4 hours

### Step 3: Split Data 📊
- Creates patient-level splits (prevents data leakage)
- 70% train / 15% val / 15% test
- Stratified by tumor presence

**Output:**
- Train: ~690 patients (~17,500 slices)
- Val: ~150 patients (~3,750 slices)
- Test: ~150 patients (~3,750 slices)

### Step 4: Train Model 🏋️
- U-Net 2D (31M parameters)
- 100 epochs with early stopping
- Mixed precision (AMP)
- W&B logging
- Checkpoint management

**Configuration:**
- Loss: Dice + BCE (60/40 weight)
- Optimizer: AdamW (lr=0.0005)
- Scheduler: Cosine annealing
- Batch size: 16 (effective 32 with gradient accumulation)
- Augmentation: Strong (flips, rotations, elastic, noise)

**Output:**
- Best model: `checkpoints/seg_production/best_model.pth`
- W&B dashboard: `slicewise-segmentation-production`
- Time: 4-8 hours (depends on GPU)

### Step 5: Evaluate 📈
- Tests on held-out test set
- Computes comprehensive metrics
- Generates visualizations

**Metrics:**
- Dice coefficient
- IoU (Jaccard)
- Precision, Recall, F1
- Specificity
- Per-slice and aggregate statistics

**Output:**
- `outputs/seg_full_scale_evaluation/evaluation_results.json`
- `outputs/seg_full_scale_evaluation/metrics_distribution.png`
- `outputs/seg_full_scale_evaluation/visualizations/` (50 samples)

### Step 6: Save Results 💾
- Saves pipeline execution summary
- Tracks timing for each step
- Generates final report

**Output:**
- `full_scale_brats_e2e_results.json`

---

## 📊 Expected Results

### Current Baseline (30 patients)
- Train Dice: 0.92
- Val Dice: 0.71
- Test Dice: 0.69 ± 0.24

### Expected with Full Dataset (988 patients)
- Train Dice: 0.90-0.93
- Val Dice: 0.78-0.82 ✅ (meets 0.78 target!)
- Test Dice: 0.76-0.80
- Specificity: 0.995+

**Why Better?**
- 33× more training data
- Better generalization
- More diverse tumor types
- Reduced overfitting

---

## 🔧 Advanced Options

### Custom Number of Patients
```bash
# Train on 200 patients (faster iteration)
python scripts/full_scale_brats_e2e.py --num-patients 200

# Train on 500 patients (good balance)
python scripts/full_scale_brats_e2e.py --num-patients 500
```

### Skip Specific Steps
```bash
# Only evaluate (model already trained)
python scripts/full_scale_brats_e2e.py \
    --skip-preprocessing \
    --skip-training

# Only train (data already preprocessed)
python scripts/full_scale_brats_e2e.py \
    --skip-preprocessing
```

### Custom Configuration
```bash
# Use custom config file
python scripts/full_scale_brats_e2e.py \
    --config configs/my_custom_config.yaml
```

### Custom Paths
```bash
# Use different directories
python scripts/full_scale_brats_e2e.py \
    --brats-raw-dir /path/to/brats \
    --brats-processed-dir /path/to/output
```

---

## 📁 Output Files

```
Project Root/
├── data/
│   └── processed/
│       └── brats2d_full/
│           ├── train/          # ~17,500 slices
│           ├── val/            # ~3,750 slices
│           └── test/           # ~3,750 slices
│
├── checkpoints/
│   └── seg_production/
│       ├── best_model.pth      # Best model (highest val Dice)
│       ├── last_model.pth      # Final model
│       └── checkpoint_epoch_*.pth  # Periodic checkpoints
│
├── outputs/
│   └── seg_full_scale_evaluation/
│       ├── evaluation_results.json
│       ├── metrics_distribution.png
│       └── visualizations/
│           ├── sample_0001.png
│           ├── sample_0002.png
│           └── ...
│
├── logs/
│   └── seg_production/
│       └── training.log
│
└── full_scale_brats_e2e_results.json  # Pipeline summary
```

---

## ⏱️ Time Estimates

| Dataset Size | Preprocessing | Training | Evaluation | Total |
|--------------|---------------|----------|------------|-------|
| 100 patients | 15-20 min | 1.5-2 hrs | 5 min | 2-3 hrs |
| 200 patients | 30-40 min | 2-3 hrs | 8 min | 3-4 hrs |
| 500 patients | 1-2 hrs | 3-5 hrs | 12 min | 4-7 hrs |
| 988 patients | 2-4 hrs | 4-8 hrs | 20 min | 6-12 hrs |

**Factors:**
- CPU speed (preprocessing)
- GPU type (training)
- Disk I/O speed
- Number of workers

---

## 💻 Hardware Requirements

### Minimum
- CPU: 4+ cores
- RAM: 16GB
- GPU: 6GB VRAM (GTX 1060, RTX 2060)
- Disk: 50GB free space

### Recommended
- CPU: 8+ cores
- RAM: 32GB
- GPU: 8GB+ VRAM (RTX 3070, A100)
- Disk: 100GB free space (SSD preferred)

### Optimal
- CPU: 16+ cores
- RAM: 64GB
- GPU: 16GB+ VRAM (A100, V100)
- Disk: 200GB SSD

---

## 🐛 Troubleshooting

### Issue: Out of Memory (GPU)
```bash
# Reduce batch size in config
# Edit configs/seg2d_production.yaml:
training:
  batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 4  # Increase to maintain effective batch
```

### Issue: Out of Memory (CPU during preprocessing)
```bash
# Process in smaller batches
python scripts/full_scale_brats_e2e.py --num-patients 100
# Then run again with --skip-preprocessing for next batch
```

### Issue: Slow Preprocessing
```bash
# Check disk I/O
# Move data to SSD if possible
# Reduce num_workers if CPU-bound
```

### Issue: Training Stalled
```bash
# Check W&B dashboard for metrics
# Verify GPU utilization: nvidia-smi
# Check logs: logs/seg_production/training.log
```

---

## 📊 Monitoring Training

### Weights & Biases
```bash
# View training progress
# Open: https://wandb.ai/YOUR_USERNAME/slicewise-segmentation-production
```

**Key Metrics to Watch:**
- `train/dice` - Should increase to 0.90+
- `val/dice` - Should increase to 0.78+
- `train/loss` - Should decrease to <0.1
- `val/loss` - Should decrease to <0.2
- `lr` - Should decay from 0.0005 to ~1e-7

### TensorBoard (Alternative)
```bash
# Start TensorBoard
tensorboard --logdir runs/seg_production

# Open browser: http://localhost:6006
```

---

## ✅ Success Criteria

### Training Complete When:
- ✅ Val Dice ≥ 0.78 (target met!)
- ✅ Early stopping triggered (no improvement for 25 epochs)
- ✅ 100 epochs completed
- ✅ Checkpoint saved

### Evaluation Complete When:
- ✅ Test Dice ≥ 0.75
- ✅ Specificity ≥ 0.99
- ✅ 50 visualizations generated
- ✅ Metrics distribution looks reasonable

---

## 🎯 Next Steps After Training

### 1. Analyze Results
```bash
# View evaluation results
cat outputs/seg_full_scale_evaluation/evaluation_results.json

# View visualizations
open outputs/seg_full_scale_evaluation/visualizations/
```

### 2. Test in UI
```bash
# Start demo application
python scripts/run_demo.py

# Open browser: http://localhost:8501
# Go to Segmentation tab
# Upload test images
```

### 3. Generate More Visualizations
```bash
# Generate 100 visualizations
python src/eval/eval_seg2d.py \
    --checkpoint checkpoints/seg_production/best_model.pth \
    --data-dir data/processed/brats2d_full/test \
    --max-visualizations 100
```

### 4. Patient-Level Evaluation
```bash
# Evaluate at patient level
python src/eval/patient_level_eval.py \
    --checkpoint checkpoints/seg_production/best_model.pth \
    --data-dir data/processed/brats2d_full/test
```

### 5. Profile Inference Speed
```bash
# Measure latency and throughput
python src/eval/profile_inference.py \
    --checkpoint checkpoints/seg_production/best_model.pth
```

---

## 📝 Example Output

```
[2025-12-04 19:30:00] [INFO] ================================================================================
[2025-12-04 19:30:00] [INFO] FULL-SCALE BRATS E2E PIPELINE
[2025-12-04 19:30:00] [INFO] ================================================================================
[2025-12-04 19:30:00] [INFO] Processing: ALL 988 patients (full production run)

[2025-12-04 19:30:01] [SUCCESS] ✓ Found 988 patient directories
[2025-12-04 19:32:15] [SUCCESS] ✓ Completed: Preprocessing (120.5 minutes)
[2025-12-04 19:33:02] [SUCCESS] ✓ Completed: Creating patient-level splits (0.8 minutes)
[2025-12-04 23:45:30] [SUCCESS] ✓ Completed: Training U-Net (252.5 minutes)
[2025-12-04 23:58:12] [SUCCESS] ✓ Completed: Evaluating model on test set (12.7 minutes)

================================================================================
PIPELINE SUMMARY
================================================================================
✓ check_data         : success    (N/A)
✓ preprocess         : success    (120.5 minutes)
✓ split              : success    (0.8 minutes)
✓ train              : success    (252.5 minutes)
✓ evaluate           : success    (12.7 minutes)

Total Time: 6.43 hours
================================================================================
```

---

## 🎓 Tips for Best Results

### 1. Monitor Training Closely
- Check W&B dashboard every 10 epochs
- Look for overfitting (train >> val metrics)
- Adjust learning rate if needed

### 2. Use Early Stopping
- Don't force 100 epochs if converged
- Patience=25 is reasonable
- Best model is saved automatically

### 3. Experiment with Hyperparameters
- Try different loss weights (Dice vs BCE)
- Adjust augmentation strength
- Tune learning rate

### 4. Save Intermediate Checkpoints
- Keep last 5 checkpoints
- Useful if best model overfits
- Can resume from any checkpoint

### 5. Validate on Diverse Cases
- Check performance on small tumors
- Check performance on large tumors
- Check performance on edge cases

---

## 🚀 Ready to Run!

```bash
# Start the full-scale training pipeline
python scripts/full_scale_brats_e2e.py

# Or quick test first
python scripts/full_scale_brats_e2e.py --num-patients 100
```

**Good luck! This should get you to Dice ≥ 0.78!** 🎯
