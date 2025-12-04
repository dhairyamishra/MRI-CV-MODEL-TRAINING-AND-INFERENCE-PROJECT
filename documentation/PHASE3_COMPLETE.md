# Phase 3 Complete: U-Net Segmentation Pipeline

**SliceWise - Brain Tumor Segmentation**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Components Implemented](#components-implemented)
3. [Architecture Details](#architecture-details)
4. [Training Results](#training-results)
5. [Evaluation Metrics](#evaluation-metrics)
6. [File Structure](#file-structure)
7. [Usage Guide](#usage-guide)
8. [Next Steps](#next-steps)

---

## 🎯 Overview

Phase 3 implements a complete **2D U-Net segmentation pipeline** for brain tumor detection on BraTS 2020 dataset. The pipeline includes data preprocessing, model training, inference, post-processing, and comprehensive evaluation.

**Key Achievements:**
- ✅ Full U-Net architecture (31.4M parameters)
- ✅ 5 loss functions (Dice, BCE, Dice+BCE, Tversky, Focal)
- ✅ Complete training pipeline with W&B logging
- ✅ Inference utilities with batch support
- ✅ Post-processing tools (thresholding, morphology, connected components)
- ✅ Comprehensive evaluation with visualizations
- ✅ Baseline model trained (Dice: 0.708 on validation)

---

## 🏗️ Components Implemented

### **1. U-Net 2D Architecture**
**File:** `src/models/unet2d.py` (352 lines)

**Features:**
- Configurable depth (3-5 encoder/decoder blocks)
- Configurable base filters (32-128)
- Skip connections for precise localization
- Bilinear or transposed convolution upsampling
- Binary and multi-class segmentation support
- Dropout support for regularization

**Architecture:**
```
Input (1, 256, 256)
    ↓
Encoder (4 blocks with MaxPool)
    ↓
Bottleneck
    ↓
Decoder (4 blocks with Upsample + Skip Connections)
    ↓
Output (1, 256, 256)
```

**Parameters:** 31,383,681 (standard config: 64 base filters, depth 4)

---

### **2. Segmentation Loss Functions**
**File:** `src/training/losses.py` (396 lines)

**Implemented Losses:**

| Loss | Use Case | Formula |
|------|----------|---------|
| **Dice Loss** | Primary metric | `1 - (2*TP) / (2*TP + FP + FN)` |
| **BCE Loss** | Pixel-wise classification | Binary Cross-Entropy |
| **Dice+BCE** | Best of both worlds | `α*Dice + β*BCE` |
| **Tversky Loss** | Handle class imbalance | Configurable α, β for FP/FN |
| **Focal Loss** | Focus on hard examples | `-(1-p)^γ * log(p)` |

**Factory Function:**
```python
from src.training.losses import get_loss_function

criterion = get_loss_function('dice_bce', dice_weight=0.5, bce_weight=0.5)
```

---

### **3. BraTS 2D Preprocessing**
**File:** `src/data/preprocess_brats_2d.py` (452 lines)

**Features:**
- 3D NIfTI → 2D slice extraction
- Multiple modalities (FLAIR, T1, T1ce, T2)
- Multiple normalization methods (z-score, min-max, percentile)
- Empty slice filtering (configurable threshold)
- Metadata preservation (patient ID, slice index, tumor info)
- Progress tracking with tqdm

**Output Format:**
```python
# Each .npz file contains:
{
    'image': np.ndarray,      # (1, 256, 256) float32
    'mask': np.ndarray,       # (1, 256, 256) uint8
    'patient_id': str,
    'slice_idx': int,
    'modality': str,
    'has_tumor': bool,
    'tumor_pixels': int,
    'pixdim': tuple
}
```

---

### **4. BraTS 2D Dataset Class**
**File:** `src/data/brats2d_dataset.py` (234 lines)

**Features:**
- PyTorch Dataset for 2D slices
- Optional transform support
- Metadata access
- Statistics computation
- DataLoader creation helper

**Usage:**
```python
from src.data.brats2d_dataset import BraTS2DSliceDataset, create_dataloaders

dataset = BraTS2DSliceDataset('data/processed/brats2d/train')
train_loader, val_loader = create_dataloaders(
    train_dir='data/processed/brats2d/train',
    val_dir='data/processed/brats2d/val',
    batch_size=16
)
```

---

### **5. Patient-Level Splitting**
**File:** `src/data/split_brats.py` (245 lines)

**Features:**
- Patient-level splitting (prevents data leakage)
- Configurable ratios (default: 70/15/15)
- Random seed for reproducibility
- Progress tracking

**Usage:**
```bash
python src/data/split_brats.py --input data/processed/brats2d
```

---

### **6. Training Pipeline**
**File:** `src/training/train_seg2d.py` (462 lines)

**Features:**
- Mixed precision training (AMP)
- Multiple optimizers (Adam, AdamW, SGD)
- Learning rate scheduling (Cosine, Step, Plateau)
- Early stopping
- Gradient clipping
- W&B logging
- Checkpoint management

**Configuration:** `configs/seg2d_baseline.yaml` (143 lines)

**Training Command:**
```bash
python scripts/train_segmentation.py
# or
python src/training/train_seg2d.py --config configs/seg2d_baseline.yaml
```

---

### **7. Inference Utilities**
**File:** `src/inference/infer_seg2d.py` (329 lines)

**Features:**
- `SegmentationPredictor` class
- Single slice prediction
- Batch prediction
- DataLoader prediction
- Returns probability maps and binary masks

**Usage:**
```python
from src.inference.infer_seg2d import SegmentationPredictor

predictor = SegmentationPredictor('checkpoints/seg/best_model.pth')
result = predictor.predict_slice(image)
mask = result['mask']
prob = result['prob']
```

---

### **8. Post-Processing Functions**
**File:** `src/inference/postprocess.py` (301 lines)

**Features:**
- Thresholding (fixed or Otsu)
- Remove small objects
- Fill holes
- Keep largest component
- Morphological operations (open, close, dilate, erode)
- Complete pipeline with statistics

**Usage:**
```python
from src.inference.postprocess import postprocess_mask

mask, stats = postprocess_mask(
    prob_map,
    threshold=0.5,
    min_object_size=100,
    fill_holes_size=500,
    morphology_op='close'
)
```

---

### **9. Evaluation Script**
**File:** `src/eval/eval_seg2d.py` (378 lines)

**Features:**
- Comprehensive metrics (Dice, IoU, Precision, Recall, F1, Specificity)
- Visualization overlays (TP=Green, FP=Red, FN=Blue)
- Metrics distribution plots
- JSON results export
- Per-sample metrics

**Usage:**
```bash
python src/eval/eval_seg2d.py \
    --checkpoint checkpoints/seg/best_model.pth \
    --data-dir data/processed/brats2d/val \
    --output-dir outputs/seg/evaluation
```

---

## 📊 Training Results

### **Baseline Model (10 Patients, 10 Epochs)**

**Dataset:**
- Train: 417 slices (7 patients)
- Val: 45 slices (1 patient)
- Test: 107 slices (2 patients)

**Training Configuration:**
- Model: U-Net (31.4M params)
- Loss: Dice + BCE (50/50 weight)
- Optimizer: Adam (lr=0.001)
- Scheduler: Cosine Annealing
- Batch Size: 16
- Mixed Precision: Enabled
- Early Stopping: 15 epochs patience

**Final Metrics:**

| Split | Dice | IoU | Loss |
|-------|------|-----|------|
| Train | 0.860 | 0.771 | 0.166 |
| Val | 0.743 | 0.597 | 0.271 |

**W&B Run:** [View on W&B](https://wandb.ai/dhairya28m-nyu/slicewise-segmentation)

---

## 📈 Evaluation Metrics

### **Validation Set Performance (45 slices)**

| Metric | Mean ± Std | Min | Max | Median |
|--------|------------|-----|-----|--------|
| **Dice** | 0.708 ± 0.182 | - | - | - |
| **IoU** | 0.573 ± 0.177 | - | - | - |
| **Precision** | 0.768 ± 0.213 | - | - | - |
| **Recall** | 0.676 ± 0.189 | - | - | - |
| **F1** | 0.708 ± 0.182 | - | - | - |
| **Specificity** | 0.998 ± 0.002 | - | - | - |

**Interpretation:**
- ✅ **Dice 0.708**: Good overlap (70.8%) for baseline
- ✅ **High Specificity (99.8%)**: Very few false positives
- ✅ **Moderate Recall (67.6%)**: Catches most tumor regions
- ⚠️ **Standard deviation ~0.18**: Some variability across slices

**Outputs:**
- `evaluation_results.json` - Detailed per-sample metrics
- `metrics_distribution.png` - Histogram plots
- `visualizations/` - 20 overlay images

---

## 📁 File Structure

```
Phase 3 Components:
├── src/
│   ├── models/
│   │   └── unet2d.py                    # U-Net architecture (352 lines)
│   ├── training/
│   │   ├── losses.py                    # 5 loss functions (396 lines)
│   │   └── train_seg2d.py               # Training pipeline (462 lines)
│   ├── data/
│   │   ├── preprocess_brats_2d.py       # 3D→2D preprocessing (452 lines)
│   │   ├── brats2d_dataset.py           # PyTorch dataset (234 lines)
│   │   └── split_brats.py               # Patient-level split (245 lines)
│   ├── inference/
│   │   ├── infer_seg2d.py               # Inference utilities (329 lines)
│   │   └── postprocess.py               # Post-processing (301 lines)
│   └── eval/
│       └── eval_seg2d.py                # Evaluation script (378 lines)
├── configs/
│   └── seg2d_baseline.yaml              # Training config (143 lines)
├── scripts/
│   └── train_segmentation.py            # Helper script (21 lines)
├── checkpoints/seg/
│   ├── best_model.pth                   # Best checkpoint (Dice: 0.743)
│   └── last_model.pth                   # Final checkpoint
└── outputs/seg/evaluation/
    ├── evaluation_results.json          # Detailed metrics
    ├── metrics_distribution.png         # Distribution plots
    └── visualizations/                  # 20 overlay images
```

**Total Lines of Code:** ~3,300 lines

---

## 🚀 Usage Guide

### **1. Preprocess BraTS Data**

```bash
# Process 10 patients (quick test)
python src/data/preprocess_brats_2d.py \
    --input data/raw/brats2020/MICCAI_BraTS2020_TrainingData \
    --output data/processed/brats2d \
    --modality flair \
    --num-patients 10

# Process all 988 patients (full dataset)
python src/data/preprocess_brats_2d.py \
    --input data/raw/brats2020/MICCAI_BraTS2020_TrainingData \
    --output data/processed/brats2d_full \
    --modality flair \
    --num-patients 988
```

### **2. Split Data**

```bash
python src/data/split_brats.py --input data/processed/brats2d
```

### **3. Train Model**

```bash
python scripts/train_segmentation.py
# or with custom config
python src/training/train_seg2d.py --config configs/seg2d_baseline.yaml
```

### **4. Evaluate Model**

```bash
python src/eval/eval_seg2d.py \
    --checkpoint checkpoints/seg/best_model.pth \
    --data-dir data/processed/brats2d/val \
    --output-dir outputs/seg/evaluation
```

### **5. Run Inference**

```python
from src.inference.infer_seg2d import SegmentationPredictor
from src.inference.postprocess import postprocess_mask

# Load model
predictor = SegmentationPredictor('checkpoints/seg/best_model.pth')

# Predict
result = predictor.predict_slice(image)
prob_map = result['prob']

# Post-process
mask, stats = postprocess_mask(
    prob_map,
    threshold=0.5,
    min_object_size=100,
    fill_holes_size=500
)
```

---

## 🎯 Next Steps

### **Immediate (Phase 3 Completion):**
1. ✅ Create helper scripts (`scripts/evaluate_segmentation.py`)
2. ✅ Document Phase 3 (this document)
3. ⏳ Process all 988 patients
4. ⏳ Train on full dataset (50-100 epochs)
5. ⏳ Final evaluation on test set

### **Phase 4 (Calibration & Uncertainty):**
- Temperature scaling for calibration
- MC Dropout for uncertainty estimation
- Test-time augmentation (TTA)

### **Phase 5 (Ablations & Evaluation):**
- Multi-modal experiments (FLAIR + T1 + T2)
- Different loss functions comparison
- Augmentation ablation
- Patient-level aggregation

### **Phase 6 (Demo Application):**
- Integrate segmentation into FastAPI backend
- Add segmentation to Streamlit frontend
- Create unified classification + segmentation UI

---

## 📊 Performance Comparison

| Model | Parameters | Dice (Val) | IoU (Val) | Training Time |
|-------|------------|------------|-----------|---------------|
| U-Net Baseline | 31.4M | 0.708 | 0.573 | ~10 min (10 epochs) |
| U-Net Full* | 31.4M | TBD | TBD | ~6-12 hours (988 patients) |

*To be trained on full dataset

---

## 🔬 Technical Details

### **Model Architecture:**
- **Encoder:** 4 downsampling blocks (MaxPool + DoubleConv)
- **Bottleneck:** DoubleConv at lowest resolution
- **Decoder:** 4 upsampling blocks (Upsample + Concatenate + DoubleConv)
- **Output:** 1×1 conv for binary segmentation

### **Training Techniques:**
- Mixed precision (AMP) for faster training
- Gradient clipping (max norm: 1.0)
- Cosine annealing LR schedule
- Early stopping (patience: 15)
- W&B logging for experiment tracking

### **Data Augmentation:**
- Random horizontal/vertical flips
- Random rotations (±15°)
- Random scaling (±10%)
- Gaussian noise (σ=0.01)

---

## 📝 Key Learnings

1. **Patient-level splitting is crucial** to prevent data leakage
2. **Dice + BCE loss** works well for medical segmentation
3. **High specificity (99.8%)** indicates conservative predictions
4. **Baseline Dice 0.708** is reasonable for 10 patients
5. **Standard deviation ~0.18** suggests need for more data

---

## 🙏 Acknowledgments

- **BraTS 2020 Dataset:** Medical Image Computing and Computer Assisted Intervention (MICCAI)
- **U-Net Architecture:** Ronneberger et al., 2015
- **Libraries:** PyTorch, MONAI, scikit-image, OpenCV

---

## 📚 References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
2. Menze, B. H., et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). IEEE TMI.
3. Bakas, S., et al. (2018). Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation. Frontiers in Oncology.

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-04  
**Status:** Phase 3 Complete - Baseline Model Trained ✅
