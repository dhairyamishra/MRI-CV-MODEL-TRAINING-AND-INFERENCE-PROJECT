# Dataset Comparison: Kaggle vs BraTS

This document compares the two datasets used in the SliceWise MRI Brain Tumor Detection project.

---

## Overview

| Feature | Kaggle Brain MRI | BraTS 2020 |
|---------|------------------|------------|
| **Source** | Kaggle (navoneel/brain-mri-images-for-brain-tumor-detection) | Medical Segmentation Decathlon / BraTS Challenge |
| **Task** | Binary Classification | Semantic Segmentation |
| **Format** | 2D JPG images | 3D NIfTI volumes (.nii.gz) |
| **Modalities** | Single (unknown modality) | 4 modalities (FLAIR, T1, T1ce, T2) |
| **Labels** | Image-level (yes/no tumor) | Pixel-level segmentation masks |
| **Size** | ~250 images | 369 training patients (3D volumes) |
| **Resolution** | Variable (typically ~512x512) | 240×240×155 voxels |
| **Preprocessing** | Min-max normalization [0,1] | Z-score normalization (mean=0, std=1) |

---

## 1. Kaggle Brain MRI Dataset

### Characteristics
- **Purpose**: Binary classification (tumor present vs. no tumor)
- **Data Type**: 2D grayscale images (JPG format)
- **Classes**: 
  - `yes/` - Images with brain tumors
  - `no/` - Images without brain tumors
- **Typical Size**: ~512×512 pixels (variable)
- **Color Space**: Grayscale (single channel)

### Preprocessing Pipeline
1. Load JPG as grayscale
2. Normalize to [0, 1] range (min-max)
3. Resize to 256×256
4. Save as `.npz` with metadata

### Metadata Structure
```json
{
  "image_id": "Y1",
  "class": "yes",
  "label": 1,
  "original_size": [512, 512],
  "target_size": [256, 256],
  "source": "kaggle_brain_mri"
}
```

### Use Cases
- ✅ **Classification**: Train models to detect tumor presence
- ✅ **Transfer Learning**: Pre-train on simple binary task
- ✅ **Quick Prototyping**: Small dataset, fast iteration
- ❌ **Segmentation**: No pixel-level annotations
- ❌ **Volume Analysis**: Only 2D slices, no 3D context

### Strengths
- Simple and easy to work with
- Good for initial prototyping
- Fast training due to small size
- Clear binary labels

### Limitations
- Small dataset (~250 images)
- No segmentation masks
- Unknown MRI modality
- Variable image quality
- No patient-level information
- Potential class imbalance

---

## 2. BraTS 2020 Dataset

### Characteristics
- **Purpose**: Semantic segmentation of brain tumors
- **Data Type**: 3D NIfTI volumes (medical imaging standard)
- **Modalities**: 
  - **FLAIR**: Fluid-attenuated inversion recovery
  - **T1**: T1-weighted
  - **T1ce**: T1-weighted with contrast enhancement
  - **T2**: T2-weighted
- **Volume Size**: 240×240×155 voxels per modality
- **Segmentation Labels**:
  - 0: Background
  - 1: Necrotic/non-enhancing tumor core
  - 2: Peritumoral edema
  - 4: GD-enhancing tumor

### Preprocessing Pipeline (3D → 2D)
1. Load 3D NIfTI volume
2. Extract 2D slices along depth axis
3. Apply z-score normalization (mean=0, std=1)
4. Convert multi-class mask to binary (tumor vs. background)
5. Filter empty slices (optional)
6. Resize to 256×256
7. Save as `.npz` with metadata

### Metadata Structure
```json
{
  "patient_id": "BraTS20_Training_001",
  "slice_idx": 75,
  "modality": "flair",
  "original_shape": [240, 240, 155],
  "has_tumor": true,
  "tumor_pixels": 1523,
  "normalize_method": "zscore",
  "pixdim": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}
```

### Use Cases
- ✅ **Segmentation**: Train U-Net for precise tumor delineation
- ✅ **Multi-modal Learning**: Leverage 4 MRI modalities
- ✅ **Volume Estimation**: Calculate tumor volume in mm³
- ✅ **Patient-level Analysis**: Group slices by patient
- ✅ **Clinical Research**: Medical-grade annotations
- ⚠️ **Classification**: Can derive labels from segmentation masks

### Strengths
- Large dataset (369 patients = thousands of slices)
- Pixel-level annotations (ground truth masks)
- Multiple MRI modalities
- Medical-grade quality
- Patient-level organization
- 3D spatial context available
- Standardized format (NIfTI)

### Limitations
- More complex preprocessing required
- Larger storage requirements
- Class imbalance (most pixels are background)
- Requires medical imaging libraries (nibabel)
- Slower to process than Kaggle dataset

---

## Key Differences

### 1. **Task Type**
- **Kaggle**: Image-level classification (yes/no)
- **BraTS**: Pixel-level segmentation (where is the tumor?)

### 2. **Annotation Granularity**
- **Kaggle**: Binary label per image
- **BraTS**: Segmentation mask per slice + patient ID

### 3. **Data Complexity**
- **Kaggle**: Simple 2D images
- **BraTS**: 3D volumes with 4 modalities

### 4. **Normalization**
- **Kaggle**: Min-max [0, 1] - preserves relative intensities
- **BraTS**: Z-score (μ=0, σ=1) - standardizes distribution

### 5. **Clinical Relevance**
- **Kaggle**: Educational/prototyping
- **BraTS**: Research-grade medical data

### 6. **Model Requirements**
- **Kaggle**: Simple CNN (EfficientNet, ResNet)
- **BraTS**: U-Net, attention mechanisms, multi-modal fusion

---

## Preprocessing Comparison

### Kaggle Pipeline
```python
# Load JPG → Normalize [0,1] → Resize → Save
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32) / 255.0  # Min-max
image = cv2.resize(image, (256, 256))
```

### BraTS Pipeline
```python
# Load NIfTI → Extract slice → Z-score → Resize → Save
volume = nib.load(path).get_fdata()
slice_2d = volume[:, :, slice_idx]
slice_2d = (slice_2d - mean) / std  # Z-score
slice_2d = resize(slice_2d, (256, 256))
```

---

## When to Use Each Dataset

### Use Kaggle Dataset When:
- 🎯 Learning classification basics
- 🎯 Quick prototyping and experimentation
- 🎯 Limited computational resources
- 🎯 Building baseline models
- 🎯 Teaching/educational purposes

### Use BraTS Dataset When:
- 🎯 Training segmentation models
- 🎯 Precise tumor localization needed
- 🎯 Clinical research applications
- 🎯 Multi-modal learning experiments
- 🎯 Volume estimation required
- 🎯 Patient-level analysis needed

---

## Combined Workflow (Our Approach)

In this project, we use **both datasets**:

1. **Phase 1-2**: Train classifier on **Kaggle** dataset
   - Fast iteration
   - Establish baseline
   - Validate pipeline

2. **Phase 3-5**: Train segmentation on **BraTS** dataset
   - Precise tumor delineation
   - Volume estimation
   - Clinical-grade results

3. **Phase 6**: Deploy both models in unified API
   - Classification endpoint (Kaggle-trained)
   - Segmentation endpoint (BraTS-trained)
   - Best of both worlds!

---

## Export and Compare

Use the provided script to visualize differences:

```bash
# Export 10 examples from each dataset
python scripts/export_dataset_examples.py

# Export more samples
python scripts/export_dataset_examples.py --num-samples 20

# Custom distribution
python scripts/export_dataset_examples.py \
    --kaggle-with-tumor 8 \
    --kaggle-without-tumor 2 \
    --brats-with-tumor 8 \
    --brats-without-tumor 2
```

### Output Structure
```
data/dataset_examples/
├── kaggle/
│   ├── kaggle_000/
│   │   ├── image.png
│   │   └── metadata.json
│   ├── kaggle_001/
│   └── ...
├── brats/
│   ├── brats_000/
│   │   ├── image.png
│   │   ├── mask.png
│   │   ├── overlay.png
│   │   └── metadata.json
│   ├── brats_001/
│   └── ...
├── dataset_comparison.png  # Side-by-side visualization
└── export_summary.json     # Statistics
```

---

## Summary

| Aspect | Kaggle | BraTS |
|--------|--------|-------|
| **Complexity** | ⭐ Simple | ⭐⭐⭐ Complex |
| **Annotation Quality** | ⭐⭐ Basic | ⭐⭐⭐⭐⭐ Medical-grade |
| **Dataset Size** | ⭐⭐ Small (~250) | ⭐⭐⭐⭐ Large (369 patients) |
| **Clinical Relevance** | ⭐⭐ Educational | ⭐⭐⭐⭐⭐ Research-grade |
| **Preprocessing** | ⭐ Easy | ⭐⭐⭐ Moderate |
| **Training Speed** | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐ Moderate |

**Both datasets are valuable** - Kaggle for quick classification experiments, BraTS for production-grade segmentation!
