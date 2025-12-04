# SliceWise - MRI Brain Tumor Detection & Segmentation

> **A production-ready deep learning pipeline for brain tumor classification and segmentation from MRI images**

## 🎯 Project Status

- ✅ **Phase 0**: Project Scaffolding & Environment (Complete)
- ✅ **Phase 1**: Data Acquisition & 2D Preprocessing (Complete - Kaggle dataset)
- ✅ **Phase 2**: Classification MVP (Complete - EfficientNet + API + UI)
- 🚧 **Phase 3**: Segmentation Pipeline (Next)

## Overview

SliceWise is a comprehensive medical imaging project that implements deep learning models for:
1. **Binary Classification**: Detecting presence of brain tumors in MRI scans
2. **Segmentation**: Precise tumor boundary delineation (BraTS dataset)
3. **Calibrated Uncertainty**: Reliable confidence estimates for clinical decision support

The project features a modular, production-ready architecture with:
- Clean data preprocessing pipelines
- PyTorch dataset classes with augmentation
- FastAPI backend for inference
- Streamlit frontend for visualization
- Comprehensive testing and CI/CD

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or 3.11
- CUDA-capable GPU (optional, but recommended)
- Kaggle API credentials (for dataset download)

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd slicewise

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Verify setup
python scripts/verify_setup.py
```

### Data Pipeline (Kaggle Dataset)

```bash
# 1. Setup Kaggle API (one-time)
# Download kaggle.json from https://www.kaggle.com/account
mkdir ~/.kaggle  # Windows: %USERPROFILE%\.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. Download dataset (245 images)
python scripts/download_kaggle_data.py

# 3. Preprocess to .npz format
python src/data/preprocess_kaggle.py

# 4. Create train/val/test splits
python src/data/split_kaggle.py

# Done! Dataset ready at data/processed/kaggle/{train,val,test}/
```

**Result**: 171 train / 37 val / 37 test images, stratified by class

## 📊 Dataset

### Kaggle Brain MRI Dataset
- **Source**: [navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Total Images**: 245
- **Classes**: 
  - Tumor (yes): 154 images (62.9%)
  - No tumor (no): 91 images (37.1%)
- **Format**: Preprocessed to 256×256 single-channel .npz files
- **Splits**: 70% train / 15% val / 15% test (stratified)

### BraTS Dataset (Optional)
- **Purpose**: 3D segmentation with expert annotations
- **Status**: Documentation complete, preprocessing pending
- See `documentation/DATA_README.md` for access instructions

{{ ... }}

## 🏗️ Project Structure

```
slicewise/
├── src/                          # Source code
│   ├── data/                     # Data pipeline
│   │   ├── kaggle_mri_dataset.py # PyTorch dataset
│   │   ├── preprocess_kaggle.py  # Preprocessing
│   │   ├── split_kaggle.py       # Train/val/test splits
│   │   └── transforms.py         # Augmentation
│   ├── models/                   # Model architectures (Phase 2)
│   ├── training/                 # Training loops (Phase 2)
│   ├── eval/                     # Evaluation & metrics (Phase 4)
│   └── inference/                # Inference pipeline (Phase 3)
├── scripts/                      # Utility scripts
│   ├── download_kaggle_data.py   # Dataset download
│   ├── smoke_test.py             # Basic functionality test
│   └── verify_setup.py           # Setup verification
├── configs/                      # Configuration files
│   ├── hpc.yaml                  # HPC environment
│   └── local.yaml                # Local development
├── tests/                        # Unit tests
├── documentation/                # Project documentation
├── jupyter_notebooks/            # Analysis notebooks
│   └── MRI-Brain-Tumor-Detecor.ipynb  # Original notebook
└── data/                         # Data directory (gitignored)
    ├── raw/                      # Raw datasets
    └── processed/                # Preprocessed .npz files
```

## 🔬 Data Pipeline Features

### Unified .npz Format
All preprocessed data uses a consistent format:
```python
{
    'image': np.ndarray,      # Shape: (1, H, W), range: [0, 1]
    'label': int,             # 0 or 1 for classification
    'mask': np.ndarray,       # Shape: (1, H, W) for segmentation
    'metadata': dict,         # Image ID, source, original size, etc.
}
```

### Data Augmentation
- Random rotations (90°, 180°, 270°)
- Random horizontal/vertical flips
- Intensity shifts and scaling
- Gaussian noise
- Three presets: standard, strong, light

### Usage Example
```python
from src.data.kaggle_mri_dataset import create_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=32,
    num_workers=4,
    train_transform=get_train_transforms(),
    val_transform=get_val_transforms(),
)

# Use in training
for images, labels in train_loader:
    # images: (batch_size, 1, 256, 256)
    # labels: (batch_size,)
    pass
```

## 🧪 Testing & Verification

```bash
# Run full verification suite
python scripts/verify_setup.py

# Run smoke test (creates fake MRI, runs tiny U-Net)
python scripts/smoke_test.py

# Run unit tests
pytest tests/

# Code formatting
black src/ tests/
isort src/ tests/
ruff check src/ tests/
```

## 📚 Documentation

### Getting Started
- **[SETUP.md](documentation/SETUP.md)** - Detailed installation guide
- **[QUICKSTART.md](documentation/QUICKSTART.md)** - Quick start guide
- **[PHASE2_QUICKSTART.md](PHASE2_QUICKSTART.md)** - ⭐ Phase 2 quick start (NEW!)

### Phase Summaries
- **[PHASE0_COMPLETE.md](documentation/PHASE0_COMPLETE.md)** - Phase 0 summary
- **[PHASE1_PROGRESS.md](documentation/PHASE1_PROGRESS.md)** - Phase 1 summary
- **[PHASE2_COMPLETE.md](documentation/PHASE2_COMPLETE.md)** - ⭐ Phase 2 summary (NEW!)

### Technical Documentation
- **[DATA_README.md](documentation/DATA_README.md)** - Dataset documentation
- **[PROJECT_STRUCTURE.md](documentation/PROJECT_STRUCTURE.md)** - Codebase organization
- **[FULL-PLAN.md](documentation/FULL-PLAN.md)** - Complete 8-phase roadmap
- **[KAGGLE_DATASET_QUICKSTART.md](KAGGLE_DATASET_QUICKSTART.md)** - Dataset pipeline guide

{{ ... }}

## 🎯 Roadmap

### ✅ Completed
- [x] Phase 0: Project scaffolding, dependencies, CI/CD
- [x] Phase 1: Kaggle dataset pipeline (download, preprocess, split)
- [x] Data augmentation and transforms
- [x] PyTorch dataset classes
- [x] Comprehensive documentation

### ✅ Completed
- [x] Phase 2: Classification MVP
  - [x] EfficientNet-B0 and ConvNeXt classifiers
  - [x] Training loop with W&B logging, mixed precision, early stopping
  - [x] Grad-CAM for explainability
  - [x] Comprehensive evaluation metrics (ROC-AUC, PR-AUC, etc.)
  - [x] FastAPI backend with 5 endpoints
  - [x] Beautiful Streamlit frontend UI
  - [x] Helper scripts for all workflows

### 🚧 In Progress
- [ ] Phase 3: Segmentation pipeline (U-Net)

### 📋 Planned
- [ ] Phase 4: Calibration & uncertainty
- [ ] Phase 5: Ablation studies
- [ ] Phase 7: Documentation & LaTeX report
- [ ] Phase 8: Packaging & deployment

See [FULL-PLAN.md](documentation/FULL-PLAN.md) for detailed roadmap.

{{ ... }}

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Follow code style (black, isort, ruff)
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

MIT License with medical disclaimer. See [LICENSE](LICENSE) for details.

**⚠️ IMPORTANT**: This software is for research and educational purposes only. It is NOT a medical device and should NOT be used for clinical diagnosis or treatment decisions.

## 🙏 Acknowledgments

- Dataset: Navoneel Chakrabarty (Kaggle)
- BraTS Challenge organizers
- PyTorch, MONAI, and FastAPI communities

## 📞 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for advancing medical AI research**
