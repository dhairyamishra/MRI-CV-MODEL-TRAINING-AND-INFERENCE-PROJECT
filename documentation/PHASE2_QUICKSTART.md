# Phase 2 Quick Start Guide 🚀

**SliceWise - Brain Tumor Classification**

This guide will get you up and running with the classification system in 5 minutes!

---

## Prerequisites

✅ Phase 0 and Phase 1 completed  
✅ Data preprocessed and split  
✅ Python environment activated  
✅ All dependencies installed

---

## Step 1: Train the Classifier (Optional)

If you want to train your own model:

```bash
# Basic training with default config
python scripts/train_classifier.py

# Or with custom config
python scripts/train_classifier.py --config configs/config_cls.yaml
```

**What happens:**
- Loads 171 training images, 37 validation images
- Trains EfficientNet-B0 for up to 50 epochs
- Applies early stopping (patience=10)
- Saves best model to `checkpoints/cls/best_model.pth`
- Logs to W&B (if configured)

**Training time:**
- CPU: ~2-3 hours
- GPU (RTX 4080): ~10-15 minutes

**Expected output:**
```
Epoch 1/50
Train Loss: 0.5234 | Train Acc: 0.7543
Val Loss: 0.4123 | Val Acc: 0.8108 | Val AUC: 0.8756

✓ Saved best model (metric: 0.8756)
```

---

## Step 2: Evaluate the Model

Once you have a trained model:

```bash
python scripts/evaluate_classifier.py
```

**Generates:**
- `results/classification/metrics.json` - All metrics
- `results/classification/predictions.csv` - Per-sample predictions
- `results/classification/confusion_matrix.png` - Confusion matrix
- `results/classification/roc_curve.png` - ROC curve
- `results/classification/pr_curve.png` - Precision-Recall curve

**Sample output:**
```
==================================================
EVALUATION METRICS
==================================================
Accuracy:     0.9189
ROC-AUC:      0.9524
PR-AUC:       0.9612
F1 Score:     0.9130
Precision:    0.9130
Recall:       0.9130
Sensitivity:  0.9130
Specificity:  0.9286

Confusion Matrix:
  True Positives:  21
  True Negatives:  13
  False Positives: 1
  False Negatives: 2
==================================================
```

---

## Step 3: Generate Grad-CAM Visualizations

See what the model is looking at:

```bash
# Generate 16 visualizations
python scripts/generate_gradcam.py --num_samples 16

# Or more samples
python scripts/generate_gradcam.py --num_samples 32
```

**Output location:** `assets/grad_cam_examples/`

Files will be named:
- `gradcam_correct_001_<id>.png` - Correct predictions
- `gradcam_incorrect_001_<id>.png` - Incorrect predictions

Each visualization shows:
1. Original MRI slice
2. Grad-CAM heatmap
3. Overlay with prediction info

---

## Step 4: Run the Demo Application

### Start the Backend API

In **Terminal 1**:
```bash
python scripts/run_backend.py
```

**Expected output:**
```
==================================================
SliceWise - Starting Backend API Server
==================================================

API will be available at: http://localhost:8000
API docs will be available at: http://localhost:8000/docs

✓ Model loaded successfully
✓ Using device: cuda:0
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test the API:**
```bash
# Health check
curl http://localhost:8000/healthz

# Model info
curl http://localhost:8000/model/info
```

**API Documentation:**
Open http://localhost:8000/docs in your browser for interactive API docs!

### Start the Frontend UI

In **Terminal 2**:
```bash
python scripts/run_frontend.py
```

**Expected output:**
```
==================================================
SliceWise - Starting Frontend Application
==================================================

Make sure the backend API is running on http://localhost:8000
Frontend will be available at: http://localhost:8501

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

**Open the app:**
Navigate to http://localhost:8501 in your browser

---

## Step 5: Use the Web Interface

### Upload and Classify

1. **Upload an image**
   - Click "Browse files" or drag & drop
   - Supported formats: JPG, PNG, BMP
   - Use images from `data/raw/kaggle_brain_mri/`

2. **Click "🔍 Analyze Image"**
   - Wait 1-2 seconds for processing

3. **View Results**
   - Prediction: Tumor / No Tumor
   - Confidence: 0-100%
   - Probability chart
   - Grad-CAM heatmap overlay

### Understanding the Results

**Prediction Colors:**
- 🟢 Green = No Tumor
- 🔴 Red = Tumor

**Confidence Levels:**
- >90%: Very High certainty
- 75-90%: High certainty
- 60-75%: Moderate certainty
- <60%: Low certainty

**Grad-CAM Heatmap:**
- Red/Yellow: High importance regions
- Blue/Purple: Low importance regions
- Shows where the model "looked" to make its decision

---

## Common Issues & Solutions

### Issue: "Model not loaded"

**Solution:**
```bash
# Check if checkpoint exists
ls checkpoints/cls/best_model.pth

# If not, train a model first
python scripts/train_classifier.py
```

### Issue: "API Not Available" in frontend

**Solution:**
1. Make sure backend is running in Terminal 1
2. Check http://localhost:8000/healthz
3. Restart backend if needed

### Issue: CUDA out of memory during training

**Solution:**
Edit `configs/config_cls.yaml`:
```yaml
data:
  batch_size: 16  # Reduce from 32
```

### Issue: Training is too slow

**Solution:**
```yaml
training:
  use_amp: true  # Enable mixed precision
  
data:
  num_workers: 4  # Increase data loading workers
```

---

## Quick Commands Reference

```bash
# Training
python scripts/train_classifier.py

# Evaluation
python scripts/evaluate_classifier.py

# Grad-CAM
python scripts/generate_gradcam.py --num_samples 16

# Backend API
python scripts/run_backend.py

# Frontend UI
python scripts/run_frontend.py

# All in one terminal (sequential)
python scripts/train_classifier.py && \
python scripts/evaluate_classifier.py && \
python scripts/generate_gradcam.py
```

---

## Configuration Tips

### For Faster Training
```yaml
# configs/config_cls.yaml
training:
  epochs: 30  # Reduce from 50
  use_amp: true  # Enable mixed precision
  
data:
  batch_size: 32  # Increase if GPU allows
  num_workers: 4
```

### For Better Accuracy
```yaml
model:
  name: "convnext"  # Try ConvNeXt instead
  dropout: 0.5  # Increase dropout

data:
  augmentation:
    augmentation_strength: "strong"  # More augmentation

training:
  optimizer:
    lr: 0.00005  # Lower learning rate
```

### For Class Imbalance
```yaml
training:
  loss:
    name: "focal"  # Use Focal Loss
    alpha: 0.25
    gamma: 2.0
  
  use_class_weights: true
```

---

## Next Steps

After completing Phase 2:

1. **Experiment with configurations**
   - Try different augmentation strengths
   - Test ConvNeXt architecture
   - Adjust learning rates

2. **Analyze results**
   - Review Grad-CAM visualizations
   - Identify failure cases
   - Check for biases

3. **Move to Phase 3**
   - Implement U-Net segmentation
   - Pixel-wise tumor localization
   - More detailed analysis

---

## Getting Help

- **Documentation**: See `documentation/PHASE2_COMPLETE.md`
- **API Docs**: http://localhost:8000/docs (when backend is running)
- **Configuration**: Check `configs/config_cls.yaml` for all options
- **Examples**: Look at `scripts/` for usage examples

---

## Success Checklist

- [ ] Model trained successfully
- [ ] Evaluation metrics look good (>85% accuracy)
- [ ] Grad-CAM visualizations generated
- [ ] Backend API running
- [ ] Frontend UI accessible
- [ ] Successfully classified test images
- [ ] Grad-CAM overlays visible

---

**🎉 Congratulations! You've completed Phase 2!**

You now have a fully functional brain tumor classification system with:
- ✅ Trained deep learning model
- ✅ REST API backend
- ✅ Interactive web interface
- ✅ Explainable AI (Grad-CAM)
- ✅ Comprehensive evaluation

**Ready for Phase 3: Segmentation!** 🚀
