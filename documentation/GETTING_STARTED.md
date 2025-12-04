# 🚀 Getting Started with SliceWise

**Welcome to SliceWise!** This guide will help you get up and running in minutes.

---

## 📋 Prerequisites

Before you begin, ensure you have:

- ✅ Python 3.10 or 3.11 installed
- ✅ Git installed
- ✅ (Optional) CUDA-capable GPU for faster training
- ✅ (Optional) Kaggle account for dataset download

---

## ⚡ Quick Setup (5 minutes)

### Step 1: Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd MRI-CV-MODEL-TRAINING-AND-INFERENCE-PROJECT

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/verify_setup.py
```

**Expected output:**
```
✓ Python version: 3.11.x
✓ PyTorch installed
✓ CUDA available
✓ All dependencies installed
✓ All modules importable
```

---

### Step 2: Download and Prepare Data

```bash
# Setup Kaggle API (one-time)
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/ (or %USERPROFILE%\.kaggle\ on Windows)

# Download dataset
python scripts/download_kaggle_data.py

# Preprocess images
python src/data/preprocess_kaggle.py

# Create train/val/test splits
python src/data/split_kaggle.py
```

**Result:** 171 train / 37 val / 37 test images ready!

---

### Step 3: Train Your First Model (Optional)

```bash
# Train with default settings
python scripts/train_classifier.py

# This will:
# - Train EfficientNet-B0 for up to 50 epochs
# - Apply early stopping (patience=10)
# - Save best model to checkpoints/cls/best_model.pth
# - Log metrics (if W&B configured)
```

**Training time:**
- GPU (RTX 4080): ~10-15 minutes
- CPU: ~2-3 hours

---

### Step 4: Run the Demo Application

**Terminal 1 - Start Backend:**
```bash
python scripts/run_backend.py
```

Wait for: `✓ Model loaded successfully`

**Terminal 2 - Start Frontend:**
```bash
python scripts/run_frontend.py
```

**Open in browser:** http://localhost:8501

---

## 🎯 What Can You Do Now?

### 1. **Upload and Classify MRI Images**
- Drag and drop any MRI image
- Get instant predictions (Tumor/No Tumor)
- View confidence scores
- See Grad-CAM explanations

### 2. **Evaluate Model Performance**
```bash
python scripts/evaluate_classifier.py
```
Generates metrics, ROC curves, confusion matrix

### 3. **Generate Grad-CAM Visualizations**
```bash
python scripts/generate_gradcam.py --num_samples 16
```
Creates explainability heatmaps

### 4. **Experiment with Configurations**
Edit `configs/config_cls.yaml` to:
- Try different architectures (EfficientNet vs ConvNeXt)
- Adjust learning rates
- Change augmentation strength
- Modify training parameters

---

## 📚 Next Steps

### Learn More
- **[PHASE2_QUICKSTART.md](PHASE2_QUICKSTART.md)** - Detailed Phase 2 guide
- **[PHASE2_COMPLETE.md](documentation/PHASE2_COMPLETE.md)** - Complete technical docs
- **[README.md](README.md)** - Project overview

### Explore the Code
```
src/
├── models/        # Neural network architectures
├── training/      # Training loops and utilities
├── eval/          # Evaluation and Grad-CAM
├── inference/     # Prediction interface
└── data/          # Dataset classes and transforms

app/
├── backend/       # FastAPI REST API
└── frontend/      # Streamlit web UI
```

### Try Advanced Features
1. **Custom Training:**
   ```python
   from src.training.train_cls import ClassificationTrainer
   
   trainer = ClassificationTrainer('configs/config_cls.yaml')
   trainer.train()
   ```

2. **Programmatic Inference:**
   ```python
   from src.inference.predict import ClassifierPredictor
   
   predictor = ClassifierPredictor('checkpoints/cls/best_model.pth')
   result = predictor.predict(image_array)
   ```

3. **API Integration:**
   ```python
   import requests
   
   files = {'file': open('mri_image.jpg', 'rb')}
   response = requests.post('http://localhost:8000/classify_slice', files=files)
   print(response.json())
   ```

---

## 🐛 Troubleshooting

### Issue: "Model not loaded" in frontend
**Solution:** Train a model first or check that `checkpoints/cls/best_model.pth` exists

### Issue: "API Not Available"
**Solution:** Make sure backend is running in Terminal 1

### Issue: CUDA out of memory
**Solution:** Reduce batch size in `configs/config_cls.yaml`:
```yaml
data:
  batch_size: 16  # Reduce from 32
```

### Issue: Slow training
**Solution:** Enable mixed precision:
```yaml
training:
  use_amp: true
```

---

## 🎓 Learning Resources

### Understanding the Model
- **EfficientNet Paper:** [arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
- **Grad-CAM Paper:** [arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)

### Medical AI Ethics
- Always include medical disclaimers
- Never use for clinical diagnosis
- Understand model limitations
- Consider data privacy

### FastAPI & Streamlit
- **FastAPI Docs:** [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)

---

## 🤝 Getting Help

### Documentation
- Check `documentation/` folder for detailed guides
- Review `PHASE2_QUICKSTART.md` for common tasks
- Read code comments and docstrings

### API Documentation
- Start backend: `python scripts/run_backend.py`
- Visit: http://localhost:8000/docs
- Interactive API testing available

### Common Commands
```bash
# Verify setup
python scripts/verify_setup.py

# Train model
python scripts/train_classifier.py

# Evaluate model
python scripts/evaluate_classifier.py

# Generate Grad-CAM
python scripts/generate_gradcam.py

# Run backend
python scripts/run_backend.py

# Run frontend
python scripts/run_frontend.py
```

---

## ✅ Success Checklist

- [ ] Environment setup complete
- [ ] Dependencies installed
- [ ] Data downloaded and preprocessed
- [ ] Model trained (or using pretrained)
- [ ] Backend API running
- [ ] Frontend UI accessible
- [ ] Successfully classified test images
- [ ] Viewed Grad-CAM visualizations

---

## 🎉 You're All Set!

Congratulations! You now have a fully functional brain tumor classification system.

**What's next?**
- Experiment with different configurations
- Try your own MRI images
- Explore the codebase
- Move to Phase 3 (Segmentation)

**Need help?** Check the documentation or open an issue.

---

**Built with ❤️ for advancing medical AI research**

*Remember: This is for research and educational purposes only. Not for clinical use.*
