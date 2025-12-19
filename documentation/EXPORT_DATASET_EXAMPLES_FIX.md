# Export Dataset Examples Fix - Load from RAW Data

## ✅ FIXED: Now Loading from RAW Directories

### **Problem:**
The script was loading from **preprocessed** `.npz` files instead of **raw** source data:
- ❌ `data/processed/kaggle/` (already normalized .npz files)
- ❌ `data/processed/brats2d/` (already sliced .npz files)

### **Solution:**
Updated to load from **RAW** source directories:
- ✅ `data/raw/kaggle_brain_mri/` (original JPG files)
- ✅ `data/raw/brats2020/` (original NIfTI volumes)

---

## 📁 Updated Paths

### **Kaggle Dataset:**
```python
# OLD (preprocessed)
kaggle_dir = "data/processed/kaggle"
# Loaded: yes_*.npz, no_*.npz from train/val/test subdirs

# NEW (raw)
kaggle_dir = "data/raw/kaggle_brain_mri"
# Loads: *.jpg from yes/ and no/ subdirs
```

### **BraTS Dataset:**
```python
# OLD (preprocessed)
brats_dir = "data/processed/brats2d"
# Loaded: *.npz slices from train/val/test subdirs

# NEW (raw)
brats_dir = "data/raw/brats2020"
# Loads: NIfTI volumes from BraTS*/ patient directories
# Extracts middle slices with/without tumors
```

---

## 🔧 What the Script Does Now

### **Kaggle Processing:**
1. Finds JPG files in `yes/` and `no/` folders
2. Loads raw JPG images
3. Resizes to 256×256
4. Normalizes to [0, 1] range
5. Saves as PNG examples with metadata

### **BraTS Processing:**
1. Finds patient directories (`BraTS*`)
2. Loads FLAIR and segmentation NIfTI volumes
3. Extracts middle slices (tries offsets: 0, ±5, ±10)
4. Separates tumor vs no-tumor slices
5. Resizes to 256×256
6. Saves as PNG examples with metadata

---

## 🎯 Usage

### **Default (10 samples each):**
```bash
python scripts/data/preprocessing/export_dataset_examples.py
```

### **Custom counts:**
```bash
python scripts/data/preprocessing/export_dataset_examples.py \
    --kaggle-with-tumor 5 \
    --kaggle-without-tumor 5 \
    --brats-with-tumor 5 \
    --brats-without-tumor 5
```

---

## 📊 Output Structure

```
data/dataset_examples/
├── kaggle/
│   ├── yes_tumor/
│   │   ├── sample_000/
│   │   │   ├── image.png
│   │   │   └── metadata.json
│   │   ├── sample_001/
│   │   └── ...
│   └── no_tumor/
│       ├── sample_000/
│       └── ...
├── brats/
│   ├── yes_tumor/
│   │   ├── sample_000/
│   │   │   ├── image.png
│   │   │   ├── mask.png
│   │   │   ├── overlay.png
│   │   │   └── metadata.json
│   │   └── ...
│   └── no_tumor/
│       └── ...
├── dataset_comparison.png
└── export_summary.json
```

---

## ✅ Benefits

1. **Shows true raw data** - What the data looks like before any processing
2. **Validates preprocessing** - Can compare raw vs preprocessed
3. **Dataset exploration** - Easy to visualize both datasets
4. **Debugging** - Helps identify preprocessing issues
5. **Documentation** - Examples for papers/presentations

---

## 🚀 Status

✅ **COMPLETE** - Script now loads from RAW directories
✅ Kaggle: Loads JPG files from `data/raw/kaggle_brain_mri/`
✅ BraTS: Loads NIfTI volumes from `data/raw/brats2020/`
✅ Generates comparison visualizations
✅ Saves metadata for each sample

**Ready to use!**
