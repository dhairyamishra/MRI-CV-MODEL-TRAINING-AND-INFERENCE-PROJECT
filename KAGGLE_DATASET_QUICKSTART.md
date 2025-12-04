# Kaggle Brain MRI Dataset - Quick Start Guide

This guide will help you download and preprocess the Kaggle Brain MRI dataset for training.

## Prerequisites

1. **Kaggle API credentials** configured
2. **Python environment** with dependencies installed

### Setup Kaggle API

```bash
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Save kaggle.json to the appropriate location:

# Linux/Mac
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

## Step-by-Step Workflow

### Step 1: Download Dataset

```bash
python scripts/download_kaggle_data.py
```

**Output:**
- Downloads to: `data/raw/kaggle_brain_mri/`
- Structure:
  ```
  data/raw/kaggle_brain_mri/
  ├── yes/  (tumor images)
  └── no/   (no tumor images)
  ```

### Step 2: Preprocess to .npz Format

```bash
python src/data/preprocess_kaggle.py
```

**What it does:**
- Loads JPG images
- Converts to grayscale
- Resizes to 256×256
- Normalizes to [0, 1] range
- Saves as compressed .npz files

**Output:**
- Location: `data/processed/kaggle/`
- Format: `{class}_{image_id}.npz`
- Each file contains:
  - `image`: (1, 256, 256) numpy array
  - `label`: 0 or 1
  - `metadata`: dict with image info

### Step 3: Create Train/Val/Test Splits

```bash
python src/data/split_kaggle.py
```

**What it does:**
- Stratified split (maintains class balance)
- Default: 70% train, 15% val, 15% test
- Copies files to split directories

**Output:**
```
data/processed/kaggle/
├── train/  (70% of data)
├── val/    (15% of data)
└── test/   (15% of data)
```

### Step 4: Verify Data

```python
# Test the dataset
python src/data/kaggle_mri_dataset.py

# Or in Python/Jupyter:
from src.data.kaggle_mri_dataset import KaggleBrainMRIDataset

dataset = KaggleBrainMRIDataset("data/processed/kaggle/train")
print(f"Dataset size: {len(dataset)}")
print(f"Class distribution: {dataset.get_class_distribution()}")

# Get a sample
image, label = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label: {label}")
```

## Usage in Training

```python
from src.data.kaggle_mri_dataset import create_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_dir="data/processed/kaggle/train",
    val_dir="data/processed/kaggle/val",
    test_dir="data/processed/kaggle/test",
    batch_size=32,
    num_workers=4,
    train_transform=get_train_transforms(),
    val_transform=get_val_transforms(),
)

# Use in training loop
for images, labels in train_loader:
    # images: (batch_size, 1, 256, 256)
    # labels: (batch_size,)
    pass
```

## Customization

### Change Image Size

```bash
python src/data/preprocess_kaggle.py --target-size 512 512
```

### Change Split Ratios

```bash
python src/data/split_kaggle.py \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### Use Different Augmentation

```python
from src.data.transforms import (
    get_train_transforms,
    get_strong_train_transforms,
    get_light_train_transforms,
)

# Standard augmentation
train_transform = get_train_transforms()

# Stronger augmentation
train_transform = get_strong_train_transforms()

# Lighter augmentation
train_transform = get_light_train_transforms()
```

## Troubleshooting

### "No .npz files found"
- Make sure you ran preprocessing before splitting
- Check that `data/processed/kaggle/` contains .npz files

### "Kaggle API credentials not found"
- Verify `~/.kaggle/kaggle.json` exists
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`

### "Import errors"
- Install dependencies: `pip install -e ".[dev]"`
- Make sure you're in the project root directory

## Expected Results

After completing all steps:

- **Total images**: ~250
- **Train set**: ~175 images
- **Val set**: ~38 images
- **Test set**: ~37 images
- **Class balance**: Maintained across all splits
- **Image size**: 256×256 (single channel)
- **Value range**: [0.0, 1.0]

## Next Steps

1. ✅ Data is ready for training
2. Move to **Phase 2**: Classification MVP
3. Train a classifier on this dataset
4. Evaluate with Grad-CAM visualizations

## Files Created

| Script | Purpose |
|--------|---------|
| `scripts/download_kaggle_data.py` | Download dataset from Kaggle |
| `src/data/preprocess_kaggle.py` | Convert JPG to .npz format |
| `src/data/split_kaggle.py` | Create train/val/test splits |
| `src/data/kaggle_mri_dataset.py` | PyTorch Dataset class |
| `src/data/transforms.py` | Data augmentation |

---

**Time to complete**: ~10 minutes (depending on download speed)  
**Disk space required**: ~100 MB
