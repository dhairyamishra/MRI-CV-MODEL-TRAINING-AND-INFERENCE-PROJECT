# BraTS Dataset Guide

## Overview

The **Brain Tumor Segmentation (BraTS) Challenge** dataset is the gold standard for brain tumor segmentation research. It contains multi-modal MRI scans with expert-annotated tumor segmentation masks.

This guide will help you download and prepare the BraTS dataset for Phase 3 of the SliceWise project.

---

## Dataset Information

### **What is BraTS?**

BraTS (Brain Tumor Segmentation) is an annual challenge that provides:
- **Multi-parametric MRI scans** (T1, T1ce, T2, FLAIR)
- **Expert-annotated segmentation masks** with multiple tumor regions
- **Pre-processed and skull-stripped** volumes
- **Standardized format** (NIfTI files)

### **Dataset Statistics**

| Version | Training Cases | Validation Cases | Test Cases |
|---------|---------------|------------------|------------|
| BraTS 2020 | 369 | 125 | 166 |
| BraTS 2021 | 1,251 | 219 | 530 |
| BraTS 2023 | 1,470+ | - | - |

### **Tumor Regions**

BraTS provides segmentation for multiple tumor sub-regions:

| Label | Region | Description |
|-------|--------|-------------|
| 0 | Background | Normal brain tissue |
| 1 | NCR/NET | Necrotic and Non-Enhancing Tumor core |
| 2 | ED | Peritumoral Edema |
| 4 | ET | Enhancing Tumor |

**Combined Regions:**
- **Whole Tumor (WT)**: Labels 1, 2, 4 (all tumor regions)
- **Tumor Core (TC)**: Labels 1, 4 (excludes edema)
- **Enhancing Tumor (ET)**: Label 4 only

---

## Download Options

### **Option 1: Kaggle (Easiest)**

BraTS datasets are available on Kaggle for easy download.

#### **BraTS 2020**
```bash
# Install kaggle API
pip install kaggle

# Download BraTS 2020
kaggle datasets download -d awsaf49/brats20-dataset-training-validation

# Or use our script
python scripts/download_brats_data.py --version 2020 --output data/raw/brats2020
```

**Kaggle Dataset Links:**
- BraTS 2020: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
- BraTS 2021: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

#### **BraTS 2021**
```bash
kaggle datasets download -d dschettler8845/brats-2021-task1
```

### **Option 2: Official BraTS Website**

For the latest datasets and to participate in the challenge:

1. **Register**: https://www.synapse.org/#!Synapse:syn25829067/wiki/
2. **Accept Terms**: Review and accept the data use agreement
3. **Download**: Access the dataset through Synapse

**Note**: Official downloads require registration and may take 1-2 days for approval.

### **Option 3: Medical Segmentation Decathlon**

Alternative source with similar data:

- Website: http://medicaldecathlon.com/
- Task 1: Brain Tumors (484 volumes)
- Format: NIfTI (.nii.gz)

---

## Dataset Structure

### **Raw BraTS Format**

```
data/raw/brats2020/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii.gz    # FLAIR modality
│   ├── BraTS20_Training_001_t1.nii.gz       # T1 modality
│   ├── BraTS20_Training_001_t1ce.nii.gz     # T1 contrast-enhanced
│   ├── BraTS20_Training_001_t2.nii.gz       # T2 modality
│   └── BraTS20_Training_001_seg.nii.gz      # Segmentation mask
├── BraTS20_Training_002/
│   └── ...
└── BraTS20_Training_369/
    └── ...
```

### **File Naming Convention**

- `*_flair.nii.gz`: FLAIR (Fluid Attenuated Inversion Recovery)
- `*_t1.nii.gz`: T1-weighted
- `*_t1ce.nii.gz`: T1 contrast-enhanced
- `*_t2.nii.gz`: T2-weighted
- `*_seg.nii.gz`: Ground truth segmentation mask

### **Volume Dimensions**

All volumes are standardized to:
- **Shape**: (240, 240, 155) - (Height, Width, Depth)
- **Spacing**: 1mm × 1mm × 1mm isotropic
- **Format**: NIfTI (.nii.gz)
- **Orientation**: RAS (Right-Anterior-Superior)

---

## Preprocessing Pipeline

For Phase 3, we'll convert 3D volumes to 2D slices:

### **Step 1: Download BraTS Data**

```bash
python scripts/download_brats_data.py --version 2020
```

### **Step 2: Extract 2D Slices**

```bash
python src/data/preprocess_brats_2d.py \
    --input data/raw/brats2020 \
    --output data/processed/brats2d \
    --modality flair \
    --min-tumor-pixels 100
```

**Parameters:**
- `--modality`: Which MRI modality to use (flair, t1, t1ce, t2)
- `--min-tumor-pixels`: Skip slices with fewer tumor pixels
- `--target-size`: Resize slices (default: 256×256)

### **Step 3: Create Train/Val/Test Splits**

```bash
python src/data/split_brats.py \
    --input data/processed/brats2d \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### **Processed Data Structure**

```
data/processed/brats2d/
├── train/
│   ├── BraTS20_001_slice_075.npz
│   ├── BraTS20_001_slice_076.npz
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Each `.npz` file contains:
- `image`: (1, H, W) - Single modality MRI slice
- `mask`: (1, H, W) - Binary or multi-class segmentation mask
- `metadata`: Dictionary with patient ID, slice index, tumor regions, etc.

---

## Modality Selection

### **Which Modality to Use?**

For **Phase 3 baseline**, we recommend starting with **FLAIR**:

| Modality | Best For | Pros | Cons |
|----------|----------|------|------|
| **FLAIR** | Edema detection | High contrast for edema | May miss enhancing tumor |
| **T1ce** | Enhancing tumor | Shows blood-brain barrier breakdown | Requires contrast agent |
| **T2** | Overall tumor | Good general contrast | Less specific |
| **T1** | Anatomical reference | Good brain structure | Poor tumor contrast |

**Recommendation**: Start with FLAIR, then experiment with multi-modal fusion later.

---

## Data Augmentation

For segmentation, use augmentations that preserve spatial relationships:

```python
# Recommended augmentations
- Random rotation (90°, 180°, 270°)
- Random horizontal/vertical flip
- Random intensity shift/scale
- Elastic deformation (careful with masks)
- Random crop/zoom

# Avoid
- Color jitter (grayscale images)
- Cutout/mixup (breaks spatial structure)
```

---

## Evaluation Metrics

BraTS uses these standard metrics:

1. **Dice Coefficient** (primary metric)
   - Measures overlap between prediction and ground truth
   - Range: 0 (no overlap) to 1 (perfect overlap)

2. **Hausdorff Distance (95th percentile)**
   - Measures boundary accuracy
   - Lower is better

3. **Sensitivity (Recall)**
   - True positive rate
   - Important for medical applications

4. **Specificity**
   - True negative rate
   - Avoid false alarms

---

## Quick Start Commands

### **Complete Setup (Recommended)**

```bash
# 1. Download BraTS 2020 from Kaggle
python scripts/download_brats_data.py --version 2020

# 2. Preprocess to 2D slices (FLAIR modality)
python src/data/preprocess_brats_2d.py \
    --input data/raw/brats2020 \
    --output data/processed/brats2d \
    --modality flair

# 3. Create splits
python src/data/split_brats.py \
    --input data/processed/brats2d

# 4. Verify data
python scripts/visualize_brats_data.py
```

### **Alternative: Use Subset for Testing**

```bash
# Process only first 10 patients for quick testing
python src/data/preprocess_brats_2d.py \
    --input data/raw/brats2020 \
    --output data/processed/brats2d \
    --modality flair \
    --max-patients 10
```

---

## Data Storage Requirements

| Dataset | Raw Size | Processed 2D | Total |
|---------|----------|--------------|-------|
| BraTS 2020 (369 patients) | ~20 GB | ~5 GB | ~25 GB |
| BraTS 2021 (1,251 patients) | ~70 GB | ~15 GB | ~85 GB |

**Recommendation**: Start with BraTS 2020 for development, scale to 2021 for final model.

---

## Troubleshooting

### **Issue: Kaggle API not configured**

```bash
# Set up Kaggle credentials
# 1. Go to https://www.kaggle.com/account
# 2. Create API token
# 3. Save kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### **Issue: Out of memory during preprocessing**

```bash
# Process in smaller batches
python src/data/preprocess_brats_2d.py \
    --batch-size 5 \
    --max-patients 50
```

### **Issue: NIfTI files not loading**

```bash
# Install required packages
pip install nibabel SimpleITK
```

---

## Next Steps

After setting up the BraTS dataset:

1. ✅ **Data downloaded and preprocessed**
2. 🔄 **Implement U-Net architecture** → `src/models/unet2d.py`
3. 🔄 **Create segmentation training script** → `src/training/train_seg2d.py`
4. 🔄 **Implement evaluation metrics** → `src/eval/eval_seg2d.py`
5. 🔄 **Train baseline model**
6. 🔄 **Evaluate and visualize results**

---

## References

- **BraTS Challenge**: https://www.med.upenn.edu/cbica/brats/
- **Paper**: Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE TMI 2015
- **Kaggle**: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
- **Medical Decathlon**: http://medicaldecathlon.com/

---

## License & Citation

BraTS data is provided for research purposes only. If you use this dataset, please cite:

```bibtex
@article{menze2015multimodal,
  title={The multimodal brain tumor image segmentation benchmark (BRATS)},
  author={Menze, Bjoern H and Jakab, Andras and Bauer, Stefan and others},
  journal={IEEE transactions on medical imaging},
  volume={34},
  number={10},
  pages={1993--2024},
  year={2015}
}
```

**Important**: Always include medical disclaimers when using this data for research or demonstrations.
