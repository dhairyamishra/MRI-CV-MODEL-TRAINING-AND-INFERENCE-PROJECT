# Data Documentation

This document describes the datasets used in the SliceWise project, how to access them, and their organization.

## Datasets

### 1. BraTS (Brain Tumor Segmentation Challenge)

**Description:** Multi-institutional dataset of brain MRI scans with expert-annotated tumor segmentations.

**Versions:**
- BraTS 2020
- BraTS 2021

**Modalities:**
- T1-weighted (T1)
- T1-weighted with contrast enhancement (T1ce)
- T2-weighted (T2)
- Fluid Attenuated Inversion Recovery (FLAIR)

**Annotations:**
- Whole tumor (WT)
- Tumor core (TC)
- Enhancing tumor (ET)

**Access:**
1. Register at: https://www.med.upenn.edu/cbica/brats2020/registration.html
2. Download from TCIA (The Cancer Imaging Archive)
3. Alternative: Kaggle (search "BraTS 2020")

**License:** Creative Commons Attribution-NonCommercial 4.0 International License

**Citation:**
```
@article{menze2015multimodal,
  title={The multimodal brain tumor image segmentation benchmark (BRATS)},
  author={Menze, Bjoern H and Jakab, Andras and Bauer, Stefan and others},
  journal={IEEE transactions on medical imaging},
  volume={34},
  number={10},
  pages={1993--2024},
  year={2015},
  publisher={IEEE}
}
```

### 2. Kaggle Brain MRI Images for Brain Tumor Detection

**Description:** Binary classification dataset (tumor present vs. no tumor).

**Dataset Handle:** `navoneel/brain-mri-images-for-brain-tumor-detection`

**Access:**
```bash
# Using kagglehub
import kagglehub
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")

# Or using Kaggle CLI
kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
```

**License:** Check Kaggle dataset page for specific license

**Size:** ~250 images

## Data Organization

### Raw Data Structure

```
data/
├── raw/
│   ├── brats2020/
│   │   ├── BraTS20_Training_001/
│   │   │   ├── BraTS20_Training_001_flair.nii.gz
│   │   │   ├── BraTS20_Training_001_t1.nii.gz
│   │   │   ├── BraTS20_Training_001_t1ce.nii.gz
│   │   │   ├── BraTS20_Training_001_t2.nii.gz
│   │   │   └── BraTS20_Training_001_seg.nii.gz
│   │   └── ...
│   ├── brats2021/
│   │   └── ...
│   └── kaggle_brain_mri/
│       ├── yes/
│       │   ├── Y1.jpg
│       │   └── ...
│       └── no/
│           ├── N1.jpg
│           └── ...
```

### Processed Data Structure

After preprocessing, data is organized as:

```
data/
├── processed/
│   ├── brats2d/
│   │   ├── train/
│   │   │   ├── BraTS20_001_slice_050.npz
│   │   │   └── ...
│   │   ├── val/
│   │   │   └── ...
│   │   └── test/
│   │       └── ...
│   └── kaggle/
│       ├── train/
│       │   ├── Y1.npz
│       │   └── ...
│       ├── val/
│       │   └── ...
│       └── test/
│           └── ...
```

### NPZ File Format

Each `.npz` file contains:
- `image`: numpy array of shape `(C, H, W)` - normalized image data
- `mask`: numpy array of shape `(1, H, W)` - binary segmentation mask (if available)
- `metadata`: dictionary with:
  - `patient_id`: patient identifier
  - `slice_idx`: slice index in original volume
  - `modality`: imaging modality (e.g., "FLAIR")
  - `spacing`: voxel spacing (if available)
  - `original_shape`: original image dimensions

## Data Splits

Patient-level splits are stored in `splits/`:
- `train_patients.csv` - 70% of patients
- `val_patients.csv` - 15% of patients
- `test_patients.csv` - 15% of patients

**Important:** Splits are done at the patient level to prevent data leakage (slices from the same patient are never split across train/val/test).

## Preprocessing Pipeline

See `src/data/preprocess_brats_2d.py` for BraTS preprocessing:
1. Load NIfTI volumes with nibabel
2. Select modality (default: FLAIR)
3. Normalize intensities (z-score or min-max)
4. Extract 2D slices
5. Filter empty slices (optional)
6. Save as `.npz` files

## Ethical Considerations

⚠️ **IMPORTANT:**
- All data must be de-identified before use
- This project is for **research and educational purposes only**
- NOT approved for clinical use or medical diagnosis
- Follow all dataset licenses and usage restrictions
- Cite original dataset authors in any publications

## Data Download Scripts

See `scripts/download_data.py` for automated data download (requires Kaggle API credentials).

## Storage Requirements

Approximate storage needs:
- BraTS 2020 (raw): ~50 GB
- BraTS 2020 (processed 2D slices): ~10 GB
- Kaggle Brain MRI: ~100 MB

For HPC: Store data in `/scratch/$USER/` for better I/O performance.
