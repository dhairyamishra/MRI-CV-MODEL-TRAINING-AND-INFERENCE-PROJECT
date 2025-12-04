# Phase 3 Quick Start: U-Net Segmentation

**Get started with brain tumor segmentation in 5 minutes!**

---

## 🚀 Quick Setup

### **Prerequisites**
- Python 3.10+
- CUDA-capable GPU (recommended)
- BraTS 2020 dataset downloaded

---

## ⚡ 5-Minute Quick Start

### **1. Preprocess Data (2 min)**

```bash
# Process 10 patients for quick testing
python src/data/preprocess_brats_2d.py \
    --input data/raw/brats2020/MICCAI_BraTS2020_TrainingData \
    --output data/processed/brats2d \
    --modality flair \
    --num-patients 10
```

**Output:** ~569 slices from 10 patients

---

### **2. Split Data (10 sec)**

```bash
python src/data/split_brats.py --input data/processed/brats2d
```

**Output:**
- Train: 417 slices (7 patients)
- Val: 45 slices (1 patient)
- Test: 107 slices (2 patients)

---

### **3. Train Model (2 min)**

```bash
python scripts/train_segmentation.py
```

**Expected Results:**
- Train Dice: ~0.86
- Val Dice: ~0.74
- 10 epochs, ~10 minutes

---

### **4. Evaluate Model (30 sec)**

```bash
python src/eval/eval_seg2d.py --checkpoint checkpoints/seg/best_model.pth --data-dir data/processed/brats2d/val --output-dir outputs/seg/evaluation
```

**Outputs:**
- `evaluation_results.json` - Metrics
- `metrics_distribution.png` - Plots
- `visualizations/` - 20 overlay images

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Dice | 0.708 ± 0.182 |
| IoU | 0.573 ± 0.177 |
| Precision | 0.768 ± 0.213 |
| Recall | 0.676 ± 0.189 |
| Specificity | 0.998 ± 0.002 |

---

## 🔧 Common Commands

### **Inference on Single Image**

```python
from src.inference.infer_seg2d import SegmentationPredictor
import numpy as np

# Load model
predictor = SegmentationPredictor('checkpoints/seg/best_model.pth')

# Load image (256x256 numpy array)
image = np.load('path/to/image.npy')

# Predict
result = predictor.predict_slice(image)
mask = result['mask']
prob = result['prob']
```

---

### **Post-Processing**

```python
from src.inference.postprocess import postprocess_mask

# Apply post-processing
clean_mask, stats = postprocess_mask(
    prob_map,
    threshold=0.5,
    min_object_size=100,
    fill_holes_size=500,
    morphology_op='close'
)

print(f"Removed {stats['initial_pixels'] - stats['final_pixels']} noise pixels")
```

---

### **Batch Inference**

```python
from src.inference.infer_seg2d import SegmentationPredictor

predictor = SegmentationPredictor('checkpoints/seg/best_model.pth')

# Predict on batch (B, 256, 256)
images = np.random.randn(10, 256, 256)
result = predictor.predict_batch(images)
masks = result['masks']  # (10, 256, 256)
```

---

## 🎯 Full Dataset Training

### **Process All 988 Patients**

```bash
# This takes 2-4 hours
python src/data/preprocess_brats_2d.py \
    --input data/raw/brats2020/MICCAI_BraTS2020_TrainingData \
    --output data/processed/brats2d_full \
    --modality flair \
    --num-patients 988

# Split
python src/data/split_brats.py --input data/processed/brats2d_full

# Train (6-12 hours)
python scripts/train_segmentation.py
```

---

## 📁 Output Structure

```
outputs/seg/evaluation/
├── evaluation_results.json          # Detailed metrics
├── metrics_distribution.png         # Histogram plots
└── visualizations/
    ├── sample_0000.png              # Overlay: Input | GT | Pred | TP/FP/FN
    ├── sample_0001.png
    └── ...
```

---

## 🐛 Troubleshooting

### **CUDA Out of Memory**
```yaml
# In configs/seg2d_baseline.yaml
training:
  batch_size: 8  # Reduce from 16
```

### **Training Too Slow**
```yaml
training:
  use_amp: true  # Enable mixed precision
  num_workers: 4  # Increase data loading workers
```

### **Model Not Improving**
```yaml
optimizer:
  lr: 0.0001  # Reduce learning rate

scheduler:
  name: "plateau"  # Use adaptive scheduler
```

---

## 📚 Next Steps

1. **Review visualizations** in `outputs/seg/evaluation/visualizations/`
2. **Check W&B dashboard** for training curves
3. **Try different loss functions** (edit `configs/seg2d_baseline.yaml`)
4. **Scale to full dataset** (988 patients)
5. **Experiment with post-processing** parameters

---

## 🔗 Related Documentation

- [PHASE3_COMPLETE.md](PHASE3_COMPLETE.md) - Full documentation
- [FULL-PLAN.md](FULL-PLAN.md) - Project roadmap
- [BRATS_DATASET_GUIDE.md](BRATS_DATASET_GUIDE.md) - Dataset details

---

**Happy Segmenting! 🧠✨**
