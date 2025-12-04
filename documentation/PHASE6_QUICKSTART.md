# Phase 6 Quick Start Guide

**Get the SliceWise demo running in 5 minutes!**

---

## ⚡ Quick Start (3 Commands)

```bash
# 1. Ensure you have trained models (or use test models)
# Skip if you already have checkpoints/cls/best_model.pth and checkpoints/seg/best_model.pth

# 2. Start the demo
python scripts/run_demo.py

# 3. Open your browser
# http://localhost:8501
```

That's it! 🎉

---

## 📋 Prerequisites

### Required
- Python 3.10+
- PyTorch 2.0+ with CUDA (or CPU)
- All dependencies installed: `pip install -r requirements.txt`

### Model Checkpoints
You need at least one of these:
- `checkpoints/cls/best_model.pth` - Classifier (for classification features)
- `checkpoints/seg/best_model.pth` - Segmentation (for segmentation features)

**Don't have models?** Train them first:
```bash
# Train classifier (~5 minutes on GPU)
python scripts/train_classifier.py

# Train segmentation (~10 minutes on GPU for 10 patients)
python scripts/train_segmentation.py
```

Or use the Phase 1 & 2 E2E test to generate a test model:
```bash
python scripts/test_phase1_phase2_full_e2e.py
```

---

## 🚀 Running the Demo

### Option 1: All-in-One (Recommended)

```bash
python scripts/run_demo.py
```

This starts both backend and frontend together. Access at:
- **Frontend:** http://localhost:8501
- **Backend API:** http://localhost:8000/docs

### Option 2: Separate Terminals

**Terminal 1 - Backend:**
```bash
python scripts/run_demo_backend.py
```

**Terminal 2 - Frontend:**
```bash
python scripts/run_demo_frontend.py
```

### Option 3: Custom Ports

```bash
python scripts/run_demo.py --backend-port 9000 --frontend-port 9501
```

---

## 🎯 Using the Application

### Tab 1: Classification 🔍

1. Click "Browse files" or drag-and-drop an MRI image
2. Check "Generate Grad-CAM" for explainability
3. Check "Show Calibrated Probabilities" if you have calibration
4. Click "🔍 Classify"
5. View results: prediction, confidence, Grad-CAM heatmap

**Try it with:**
- Any brain MRI slice (JPG, PNG, BMP)
- Images from `data/processed/kaggle/test/`

### Tab 2: Segmentation 🎨

1. Upload an MRI slice
2. Adjust "Probability Threshold" (default: 0.5)
3. Set "Min Tumor Area" to filter small regions
4. Optional: Enable "Estimate Uncertainty" for MC Dropout + TTA
5. Click "🎨 Segment"
6. View: original, mask, probability map, overlay

**Advanced:**
- Enable uncertainty for epistemic/aleatoric estimates
- Adjust MC iterations (higher = more accurate but slower)
- Toggle TTA for test-time augmentation

### Tab 3: Batch Processing 📦

1. Select mode: "Classification" or "Segmentation"
2. Upload multiple images (up to 100)
3. Preview uploaded images
4. Click "🚀 Process Batch"
5. Review summary statistics
6. Download results as CSV

**Use cases:**
- Screen multiple patients quickly
- Batch analysis for research
- Export results for further analysis

### Tab 4: Patient Analysis 👤

1. Enter patient ID (e.g., "PATIENT_001")
2. Upload a stack of MRI slices (10-50 images)
3. Set threshold, min area, and slice thickness
4. Click "🔬 Analyze Patient"
5. Review:
   - Tumor volume estimation
   - Affected slice ratio
   - Slice-by-slice results
   - Distribution plots
6. Download CSV and JSON reports

**Best for:**
- Patient-level tumor detection
- Volume estimation
- Longitudinal analysis

---

## 🔧 Troubleshooting

### Backend won't start

**Error:** `Model checkpoint not found`
```bash
# Train a model first
python scripts/train_classifier.py
```

**Error:** `Port 8000 already in use`
```bash
# Use a different port
python scripts/run_demo_backend.py --port 8001
```

### Frontend shows "API Not Available"

1. Check if backend is running: http://localhost:8000/healthz
2. Start backend: `python scripts/run_demo_backend.py`
3. Wait 3-5 seconds for models to load

### Predictions are slow

- **Classification:** Should be <100ms on GPU
- **Segmentation:** Should be <200ms on GPU
- **Uncertainty:** Can take 1-2 seconds (MC Dropout + TTA)

**Solutions:**
- Ensure CUDA is available: `torch.cuda.is_available()`
- Reduce MC iterations in uncertainty estimation
- Disable TTA for faster results

### Out of memory

```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""
python scripts/run_demo.py
```

Or reduce batch size in the code.

---

## 📊 Testing the Demo

### Quick Smoke Test

```bash
# 1. Start demo
python scripts/run_demo.py

# 2. In another terminal, test API
curl http://localhost:8000/healthz

# 3. Open browser and upload a test image
# Use any image from data/processed/kaggle/test/
```

### Full Feature Test

1. **Classification:**
   - Upload image → Enable Grad-CAM → Classify
   - Check both raw and calibrated probabilities

2. **Segmentation:**
   - Upload image → Enable uncertainty → Segment
   - Verify all visualizations appear

3. **Batch:**
   - Upload 5 images → Process batch → Download CSV
   - Verify CSV contains all results

4. **Patient:**
   - Upload 10 slices → Analyze → Download JSON
   - Verify volume estimation is calculated

---

## 🎓 Example Workflows

### Workflow 1: Quick Classification

```bash
# Start demo
python scripts/run_demo.py

# In browser (http://localhost:8501):
# 1. Go to "Classification" tab
# 2. Upload: data/processed/kaggle/test/Y1.jpg
# 3. Enable Grad-CAM
# 4. Click "Classify"
# 5. Review: Should show "Tumor" with high confidence
```

### Workflow 2: Segmentation with Uncertainty

```bash
# In browser:
# 1. Go to "Segmentation" tab
# 2. Upload an MRI slice
# 3. Check "Estimate Uncertainty"
# 4. Set MC iterations: 10
# 5. Enable TTA
# 6. Click "Segment"
# 7. Review uncertainty map
```

### Workflow 3: Batch Screening

```bash
# In browser:
# 1. Go to "Batch Processing" tab
# 2. Select "Classification"
# 3. Upload all images from data/processed/kaggle/test/
# 4. Click "Process Batch"
# 5. Download CSV
# 6. Open in Excel/Python for analysis
```

---

## 📝 API Usage (Advanced)

### Python Client Example

```python
import requests

# Classification
with open('mri_slice.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify',
        files={'file': f},
        params={'return_gradcam': True}
    )
    result = response.json()
    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.2%}")

# Segmentation
with open('mri_slice.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/segment',
        files={'file': f},
        params={'threshold': 0.5, 'min_area': 50}
    )
    result = response.json()
    print(f"Tumor detected: {result['has_tumor']}")
    print(f"Area: {result['tumor_area_pixels']} pixels")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/healthz

# Model info
curl http://localhost:8000/model/info

# Classify image
curl -X POST http://localhost:8000/classify \
  -F "file=@test.png" \
  -F "return_gradcam=true"

# Segment image
curl -X POST http://localhost:8000/segment \
  -F "file=@test.png" \
  -F "threshold=0.5" \
  -F "min_area=50"
```

---

## 🎯 Next Steps

After running the demo:

1. **Explore Features:**
   - Try all 4 tabs
   - Test with different images
   - Experiment with parameters

2. **Review Documentation:**
   - [PHASE6_COMPLETE.md](PHASE6_COMPLETE.md) - Full documentation
   - [FULL-PLAN.md](FULL-PLAN.md) - Project roadmap
   - API Docs: http://localhost:8000/docs

3. **Train Better Models:**
   - Use full BraTS dataset (988 patients)
   - Tune hyperparameters
   - Try different architectures

4. **Customize:**
   - Modify UI in `app/frontend/app_v2.py`
   - Add endpoints in `app/backend/main_v2.py`
   - Integrate with your own data

---

## 📞 Getting Help

**Common Issues:**
- Models not loading → Check `checkpoints/` directory
- API not responding → Restart backend
- Slow predictions → Check GPU availability
- Frontend errors → Check browser console

**Resources:**
- API Documentation: http://localhost:8000/docs
- Full Guide: [PHASE6_COMPLETE.md](PHASE6_COMPLETE.md)
- Project Plan: [FULL-PLAN.md](FULL-PLAN.md)

---

## ✅ Checklist

Before running the demo, ensure:

- [ ] Python 3.10+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] At least one model checkpoint exists
- [ ] Ports 8000 and 8501 are available
- [ ] CUDA available (optional, but recommended)

---

**Ready to go? Run:** `python scripts/run_demo.py` 🚀

**Phase 6 Quick Start** | SliceWise MRI Brain Tumor Detection
