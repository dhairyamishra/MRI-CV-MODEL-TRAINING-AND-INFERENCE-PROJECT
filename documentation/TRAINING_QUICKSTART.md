# 🚀 SliceWise Production Training - Quick Reference

**One-page guide to train highly optimized models**

---

## ⚡ Quick Start Commands

### Train Everything (Recommended)
```bash
# Full production training (100 epochs each)
python scripts/train_production.py --task both --epochs 100

# Quick test (10-20 epochs)
python scripts/train_production.py --task both --quick-test
```

### Train Individual Models
```bash
# Classification only (100 epochs)
python scripts/train_production.py --task classification --epochs 100

# Segmentation only (100 epochs)
python scripts/train_production.py --task segmentation --epochs 100
```

### Monitor Training (Run in separate terminal)
```bash
# Real-time monitoring with live plots
python scripts/monitor_training.py --task classification --refresh 5
python scripts/monitor_training.py --task segmentation --refresh 5
```

---

## 📊 What You'll Get

After training completes:

| Output | Location | Description |
|--------|----------|-------------|
| **Checkpoints** | `checkpoints/cls_production/` | Best & periodic model weights |
| **Logs** | `logs/classification_production/` | Training logs |
| **Metrics** | `outputs/classification_production/` | Evaluation results |
| **Visualizations** | `visualizations/classification_production/` | Grad-CAM, plots |
| **W&B Dashboard** | https://wandb.ai | Live training curves |

---

## 🎯 Expected Performance

### Classification (EfficientNet-B0)
- **Accuracy**: 92-95% (validation)
- **ROC-AUC**: 0.95-0.98
- **Training Time**: 2-4 hours (GPU), 12-24 hours (CPU)
- **GPU Memory**: ~4-6 GB

### Segmentation (U-Net 2D)
- **Dice Score**: 0.75-0.85 (validation)
- **IoU**: 0.65-0.75
- **Training Time**: 4-8 hours (GPU), 24-48 hours (CPU)
- **GPU Memory**: ~8-12 GB

---

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `configs/config_cls_production.yaml` | Classification training (100 epochs) |
| `configs/seg2d_production.yaml` | Segmentation training (100 epochs) |
| `configs/config_cls.yaml` | Classification quick test (50 epochs) |
| `configs/seg2d_baseline.yaml` | Segmentation quick test (10 epochs) |

---

## 📈 Key Features

### Classification Config
- ✅ EfficientNet-B0 (pretrained)
- ✅ AdamW optimizer (lr=0.0003)
- ✅ Cosine annealing + warmup
- ✅ Label smoothing (0.1)
- ✅ Strong augmentation
- ✅ Mixed precision (AMP)
- ✅ Early stopping (20 epochs)

### Segmentation Config
- ✅ U-Net 2D (31.4M params)
- ✅ AdamW optimizer (lr=0.0005)
- ✅ Dice + BCE loss (0.6:0.4)
- ✅ Gradient accumulation (2x)
- ✅ Elastic deformation
- ✅ Mixed precision (AMP)
- ✅ Early stopping (25 epochs)

---

## 🛠️ Common Options

### Skip Steps
```bash
# Skip data download (if already downloaded)
python scripts/train_production.py --task classification --skip-data

# Skip evaluation
python scripts/train_production.py --task classification --skip-eval

# Skip visualizations
python scripts/train_production.py --task classification --skip-viz

# Skip calibration
python scripts/train_production.py --task classification --skip-calibration
```

### Custom Epochs
```bash
# Train for 50 epochs instead of 100
python scripts/train_production.py --task classification --epochs 50

# Train for 200 epochs
python scripts/train_production.py --task segmentation --epochs 200
```

### Resume Training
```bash
# Resume from checkpoint
python scripts/train_production.py \
    --task classification \
    --resume checkpoints/cls_production/epoch_050.pth
```

---

## 📊 Monitoring Options

### Real-time Plots
```bash
# Live monitoring (auto-refresh every 5 seconds)
python scripts/monitor_training.py --task classification --refresh 5

# Single snapshot (no live updates)
python scripts/monitor_training.py --task classification --no-live
```

### Weights & Biases
```bash
# Login to W&B (first time only)
wandb login

# View dashboard
# Visit: https://wandb.ai/your-username/slicewise-classification-production
```

### TensorBoard
```bash
# Launch TensorBoard
tensorboard --logdir runs/cls_production

# Open browser to: http://localhost:6006
```

### GPU Monitoring
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

---

## 🎯 Post-Training Commands

### Evaluation
```bash
# Evaluate classification model
python scripts/evaluate_classifier.py \
    --checkpoint checkpoints/cls_production/best_model.pth \
    --output-dir outputs/classification_production/evaluation

# Evaluate segmentation model
python scripts/evaluate_segmentation.py \
    --checkpoint checkpoints/seg_production/best_model.pth \
    --output-dir outputs/seg_production/evaluation
```

### Generate Visualizations
```bash
# Grad-CAM for classification
python scripts/generate_gradcam.py \
    --num-samples 50 \
    --output-dir visualizations/classification_production/gradcam

# Segmentation overlays
python scripts/visualize_segmentation_results.py \
    --num-samples 50 \
    --output-dir visualizations/seg_production/predictions
```

### Model Calibration
```bash
# Calibrate classifier for better confidence scores
python scripts/calibrate_classifier.py

# View calibration results
python scripts/view_calibration_results.py
```

### Test Demo Application
```bash
# Launch full demo (backend + frontend)
python scripts/run_demo.py

# Or run separately
python scripts/run_demo_backend.py  # Terminal 1
python scripts/run_demo_frontend.py # Terminal 2
```

---

## ⚠️ Troubleshooting

### CUDA Out of Memory
```yaml
# Edit config file: reduce batch size
training:
  batch_size: 8  # or 4, or 2
  gradient_accumulation_steps: 4  # Maintain effective batch size
```

### Loss is NaN
- Reduce learning rate in config: `lr: 0.0001`
- Disable mixed precision: `use_amp: false`

### Training Too Slow
- Ensure GPU is being used: `nvidia-smi`
- Increase `num_workers` in config
- Enable `persistent_workers: true`

### Model Not Learning
- Check learning rate (might be too low)
- Verify data is loaded correctly
- Check `freeze_backbone: false` in config

---

## 📁 Directory Structure

```
MRI-CV-MODEL-TRAINING-AND-INFERENCE-PROJECT/
├── configs/
│   ├── config_cls_production.yaml    ← Classification config
│   └── seg2d_production.yaml         ← Segmentation config
├── scripts/
│   ├── train_production.py           ← Main training script
│   ├── monitor_training.py           ← Real-time monitoring
│   ├── evaluate_classifier.py        ← Evaluation
│   └── generate_gradcam.py           ← Visualizations
├── checkpoints/
│   ├── cls_production/               ← Classification checkpoints
│   └── seg_production/               ← Segmentation checkpoints
├── outputs/
│   ├── classification_production/    ← Metrics & results
│   └── seg_production/
├── visualizations/
│   ├── classification_production/    ← Grad-CAM, plots
│   └── seg_production/
└── logs/
    ├── classification_production/    ← Training logs
    └── seg_production/
```

---

## 📚 Documentation

- **Full Guide**: [documentation/PRODUCTION_TRAINING_GUIDE.md](documentation/PRODUCTION_TRAINING_GUIDE.md)
- **Project Plan**: [documentation/FULL-PLAN.md](documentation/FULL-PLAN.md)
- **Phase 6 Demo**: [documentation/PHASE6_COMPLETE.md](documentation/PHASE6_COMPLETE.md)

---

## ✅ Training Checklist

Before starting training:

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Check GPU availability: `nvidia-smi`
- [ ] Login to W&B (optional): `wandb login`
- [ ] Verify disk space: ~50 GB free
- [ ] Review config files in `configs/`

To start training:

- [ ] Run: `python scripts/train_production.py --task both --epochs 100`
- [ ] Monitor: `python scripts/monitor_training.py --task classification`
- [ ] Check W&B dashboard for live metrics
- [ ] Wait for training to complete (4-12 hours)

After training:

- [ ] Evaluate models
- [ ] Generate visualizations
- [ ] Calibrate models
- [ ] Test with demo app
- [ ] Review metrics and plots

---

## 🎓 Pro Tips

1. **Start with quick test** to verify everything works:
   ```bash
   python scripts/train_production.py --task classification --quick-test
   ```

2. **Monitor GPU usage** to ensure full utilization:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Use tmux/screen** for long training sessions:
   ```bash
   tmux new -s training
   python scripts/train_production.py --task both --epochs 100
   # Detach: Ctrl+B, then D
   # Reattach: tmux attach -t training
   ```

4. **Save W&B API key** to avoid re-login:
   ```bash
   export WANDB_API_KEY=your_key_here
   ```

5. **Check training progress** from anywhere with W&B mobile app

---

## 🚀 Ready to Train?

```bash
# Start training now!
python scripts/train_production.py --task both --epochs 100
```

**Good luck! 🎉**

For detailed information, see [PRODUCTION_TRAINING_GUIDE.md](documentation/PRODUCTION_TRAINING_GUIDE.md)
