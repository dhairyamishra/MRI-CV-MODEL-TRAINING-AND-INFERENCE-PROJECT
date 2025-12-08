# SliceWise - MRI Brain Tumor Detection & Segmentation

> **A production-ready deep learning pipeline for brain tumor classification and segmentation from MRI images with unified multi-task architecture**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)

## 🎯 Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** | ✅ Complete | Project Scaffolding & Environment |
| **Phase 1** | ✅ Complete | Data Acquisition & 2D Preprocessing |
| **Phase 2** | ✅ Complete | Classification MVP (EfficientNet + API) |
| **Phase 3** | ✅ Complete | U-Net Segmentation Pipeline |
| **Phase 4** | ✅ Complete | Calibration & Uncertainty Estimation |
| **Phase 5** | ✅ Complete | Metrics & Patient-Level Evaluation |
| **Phase 6** | ✅ Complete | Demo Application (API + UI) |
| **Multi-Task** | ✅ Complete | Unified Architecture (Classification + Segmentation) |
| **Frontend Refactor** | ✅ Complete | Modular UI Architecture (87% Code Reduction) |
| **Phase 7** | 🚧 In Progress | Documentation & LaTeX Write-up |
| **Phase 8** | 📋 Planned | Packaging & Deployment |

**Progress: 90% Complete (7/8 phases + Multi-Task + Frontend) • ~18,700+ lines of code • 21 organized scripts**

## 🌟 Overview

SliceWise is a comprehensive medical imaging project that implements state-of-the-art deep learning models for:

1. **🔍 Binary Classification**: Detecting presence of brain tumors in MRI scans
   - Multi-task unified encoder (shared with segmentation)
   - Grad-CAM explainability for interpretable predictions
   - Temperature-scaled calibration for reliable confidence estimates
   - **Accuracy: 91.3%, Sensitivity: 97.1%, ROC-AUC: 91.8%**

2. **🎯 Tumor Segmentation**: Precise tumor boundary delineation
   - U-Net 2D architecture with shared encoder
   - Multiple loss functions (Dice, BCE, Focal, Tversky)
   - MC Dropout and Test-Time Augmentation for uncertainty estimation
   - **Dice Score: 76.5% ± 14.0%, IoU: 64.0%**

3. **🚀 Multi-Task Architecture**: Unified model for both tasks
   - **Single forward pass** for classification + segmentation
   - **31.7M parameters** (9.4% reduction vs separate models)
   - **~40% faster inference** with conditional segmentation
   - **Shared encoder** learns optimal features for both tasks

4. **📊 Patient-Level Analysis**: Clinical decision support
   - Patient-level tumor detection and volume estimation
   - Comprehensive metrics (Dice, IoU, Sensitivity, Specificity)
   - Uncertainty quantification for risk assessment

### Key Features

- 🏗️ **Production-Ready Architecture**: Modular, tested, and documented
- 🚀 **FastAPI Backend**: 12 comprehensive REST endpoints
- 🎨 **Streamlit Frontend**: Refactored modular UI (15 files, 87% complexity reduction)
- 🧪 **Comprehensive Testing**: Full E2E test suite with 100% pass rate
- 📈 **Experiment Tracking**: W&B integration for training monitoring
- 🔧 **Flexible Configuration**: YAML-based configs for all components
- ⚡ **High Performance**: 2,500+ images/sec throughput, <1ms latency
- 🎯 **Educational**: Extensive documentation and code comments
- 📦 **Organized Scripts**: 21 scripts organized by functionality

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or 3.12
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM
- Kaggle API credentials (for dataset download)
- **Node.js and npm** (for PM2 process manager - recommended for demo)

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd MRI-CV-MODEL-TRAINING-AND-INFERENCE-PROJECT

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Install PM2 for demo process management (recommended)
npm install -g pm2

# 5. Verify setup
python scripts/verify_setup.py
```

### 🎮 Full Pipeline Controller (RECOMMENDED)

The easiest way to train and deploy the complete multi-task model:

```bash
# Quick test (10 patients, 5 epochs, ~30 minutes)
python scripts/run_full_pipeline.py --mode full --training-mode quick

# Baseline training (100 patients, 50 epochs, ~2-4 hours)
python scripts/run_full_pipeline.py --mode full --training-mode baseline

# Production training (988 patients, 100 epochs, ~8-12 hours)
python scripts/run_full_pipeline.py --mode full --training-mode production
```

**What it does:**
1. ✅ Downloads BraTS 2020 + Kaggle datasets
2. ✅ Preprocesses and splits data (patient-level)
3. ✅ Trains multi-task model (3 stages: seg warmup → cls head → joint)
4. ✅ Evaluates on test set with comprehensive metrics
5. ✅ Launches demo application (FastAPI + Streamlit)

See `PIPELINE_CONTROLLER_GUIDE.md` for full documentation.

### 🎬 Run the Demo Application (Pre-trained Model)

If you already have a trained model:

```bash
# Start both backend and frontend
python scripts/demo/run_multitask_demo.py

# Or start them separately:
python scripts/demo/run_demo_backend.py  # Backend on http://localhost:8000
python scripts/run_demo_frontend.py # Frontend on http://localhost:8501
```

Then open your browser to **http://localhost:8501** and explore:
- 🔍 **Classification Tab**: Upload MRI, get tumor predictions with Grad-CAM
- 🎯 **Segmentation Tab**: Precise tumor boundary detection with uncertainty
- 📦 **Batch Processing**: Process multiple images at once
- 👤 **Patient Analysis**: Analyze patient stacks with volume estimation

### 📊 Dataset Setup

#### Kaggle Brain MRI Dataset (Quick Start)

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

#### BraTS Dataset (Advanced - For Segmentation)

```bash
# 1. Download BraTS 2020 dataset (988 patients, ~80GB)
python scripts/download_brats_data.py

# 2. Preprocess 3D volumes to 2D slices
python scripts/preprocess_all_brats.py

# 3. Create patient-level splits (prevents data leakage)
python src/data/split_brats.py

# Done! Dataset ready at data/processed/brats2d/{train,val,test}/
```

See `documentation/BRATS_DATASET_GUIDE.md` for detailed instructions.

## 🏗️ Project Structure

```
MRI-CV-MODEL-TRAINING-AND-INFERENCE-PROJECT/
├── src/                              # Source code (~11,800+ lines)
│   ├── data/                         # Data pipeline
│   │   ├── kaggle_mri_dataset.py     # Kaggle dataset class
│   │   ├── brats2d_dataset.py        # BraTS 2D dataset class
│   │   ├── preprocess_kaggle.py      # Kaggle preprocessing
│   │   ├── preprocess_brats_2d.py    # BraTS 3D→2D extraction
│   │   ├── split_kaggle.py           # Kaggle train/val/test split
│   │   ├── split_brats.py            # BraTS patient-level split
│   │   └── transforms.py             # Augmentation pipeline
│   ├── models/                       # Model architectures
│   │   ├── classifier.py             # EfficientNet-B0 & ConvNeXt
│   │   └── unet2d.py                 # U-Net 2D (31.4M params)
│   ├── training/                     # Training pipelines
│   │   ├── train_cls.py              # Classifier training
│   │   ├── train_seg2d.py            # Segmentation training
│   │   └── losses.py                 # Loss functions (Dice, Focal, etc.)
│   ├── eval/                         # Evaluation & metrics
│   │   ├── eval_cls.py               # Classifier evaluation
│   │   ├── eval_seg2d.py             # Segmentation evaluation
│   │   ├── calibration.py            # Temperature scaling
│   │   ├── metrics.py                # Comprehensive metrics
│   │   ├── patient_level_eval.py     # Patient-level analysis
│   │   ├── profile_inference.py      # Performance profiling
│   │   └── grad_cam.py               # Grad-CAM explainability
│   └── inference/                    # Inference pipeline
│       ├── predict.py                # Classifier predictor
│       ├── infer_seg2d.py            # Segmentation predictor
│       ├── uncertainty.py            # MC Dropout + TTA
│       └── postprocess.py            # Post-processing utilities
├── app/                              # Demo application
│   ├── backend/                      # FastAPI backend
│   │   ├── main.py                   # Original API (Phase 2)
│   │   └── main_v2.py                # Enhanced API (Phase 6, 12 endpoints)
│   └── frontend/                     # Streamlit frontend
│       ├── app.py                    # Original UI (Phase 2)
│       └── app_v2.py                 # Enhanced UI (Phase 6, 4 tabs)
├── scripts/                          # Utility scripts
│   ├── download_kaggle_data.py       # Kaggle dataset download
│   ├── download_brats_data.py        # BraTS dataset download
│   ├── train_classifier.py           # Train classifier
│   ├── train_segmentation.py         # Train segmentation
│   ├── evaluate_classifier.py        # Evaluate classifier
│   ├── evaluate_segmentation.py      # Evaluate segmentation
│   ├── calibrate_classifier.py       # Calibrate classifier
│   ├── generate_gradcam.py           # Generate Grad-CAM
│   ├── run_demo.py                   # Run full demo
│   ├── run_demo_backend.py           # Run backend only
│   ├── run_demo_frontend.py          # Run frontend only
│   └── test_full_e2e_phase1_to_phase6.py  # Full E2E test suite
├── configs/                          # Configuration files
│   ├── config_cls.yaml               # Classifier config
│   ├── seg2d_baseline.yaml           # Segmentation config
│   ├── hpc.yaml                      # HPC environment
│   └── local.yaml                    # Local development
├── tests/                            # Unit tests
│   ├── test_classifier.py            # Classifier tests
│   ├── test_data_pipeline.py         # Data pipeline tests
│   ├── test_gradcam.py               # Grad-CAM tests
│   └── test_segmentation.py          # Segmentation tests
├── documentation/                    # Comprehensive documentation
│   ├── FULL-PLAN.md                  # Complete 8-phase roadmap
│   └── FULL_E2E_TEST_GUIDE.md        # E2E testing guide
├── jupyter_notebooks/                # Analysis notebooks
│   └── MRI-Brain-Tumor-Detecor.ipynb # Original exploration
├── outputs/                          # Training outputs
│   ├── calibration/                  # Calibration results
│   └── seg/                          # Segmentation results
└── data/                             # Data directory (gitignored)
    ├── raw/                          # Raw datasets
    │   ├── kaggle_brain_mri/         # Kaggle dataset
    │   └── brats2020/                # BraTS dataset
    └── processed/                    # Preprocessed .npz files
        ├── kaggle/                   # Kaggle processed
        └── brats2d/                  # BraTS 2D slices
```

## 📊 Datasets

### Kaggle Brain MRI Dataset
- **Source**: [navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Total Images**: 245
- **Classes**: 
  - Tumor (yes): 154 images (62.9%)
  - No tumor (no): 91 images (37.1%)
- **Format**: Preprocessed to 256×256 single-channel .npz files
- **Splits**: 70% train / 15% val / 15% test (stratified)
- **Use Case**: Binary classification

### BraTS Dataset
- **Source**: [Brain Tumor Segmentation Challenge](https://www.med.upenn.edu/cbica/brats2020/)
- **Total Patients**: 988 (369 training, 125 validation, 494 testing)
- **Modalities**: FLAIR, T1, T1ce, T2
- **Annotations**: Expert-labeled tumor segmentations (3 classes)
- **Format**: 3D NIfTI volumes → 2D slices (.npz)
- **Use Case**: Tumor segmentation with precise boundaries

## 🎓 Training Models

### Train Classifier

```bash
# Train EfficientNet-B0 on Kaggle dataset
python scripts/train_classifier.py \
    --config configs/config_cls.yaml \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch_size 32

# Train with W&B logging
python scripts/train_classifier.py \
    --config configs/config_cls.yaml \
    --wandb_project slicewise \
    --wandb_run_name efficientnet_experiment_1
```

**Features**:
- Mixed precision training (AMP)
- Early stopping with patience
- Multiple optimizers (Adam, AdamW, SGD)
- Multiple schedulers (Cosine, Step, Plateau)
- Class weight balancing
- Gradient clipping
- Checkpoint management

### Train Segmentation Model

```bash
# Train U-Net on BraTS dataset
python scripts/train_segmentation.py \
    --config configs/seg2d_baseline.yaml \
    --epochs 100 \
    --batch_size 16

# Train with custom loss function
python scripts/train_segmentation.py \
    --config configs/seg2d_baseline.yaml \
    --loss dice_bce \
    --learning_rate 1e-4
```

**Features**:
- Multiple loss functions (Dice, BCE, Focal, Tversky, Dice+BCE)
- Patient-level data splitting (no leakage)
- Empty slice filtering
- Multiple normalization methods
- W&B logging with visualizations

### Calibrate Classifier

```bash
# Calibrate classifier for better confidence estimates
python scripts/calibrate_classifier.py \
    --checkpoint outputs/classifier/best_model.pth \
    --config configs/config_cls.yaml \
    --output_dir outputs/calibration/

# Results: ECE reduction from 0.0461 → 0.0147 (68.2% improvement)
```

## 🔌 API Endpoints

The FastAPI backend (`app/backend/main_v2.py`) provides 12 comprehensive endpoints:

### Health & Info
- `GET /healthz` - Health check
- `GET /model/info` - Model information (classifier, segmentation, uncertainty)

### Classification
- `POST /classify` - Basic classification with confidence
- `POST /classify/gradcam` - Classification with Grad-CAM visualization
- `POST /classify/batch` - Batch classification (up to 100 images)

### Segmentation
- `POST /segment` - Basic segmentation with binary mask
- `POST /segment/uncertainty` - Segmentation with MC Dropout + TTA uncertainty
- `POST /segment/batch` - Batch segmentation (up to 50 images)

### Patient-Level Analysis
- `POST /patient/analyze_stack` - Analyze patient MRI stack with volume estimation

### Example Usage

```python
import requests
import numpy as np
from PIL import Image

# Load MRI image
image = np.array(Image.open("mri_scan.png").convert("L"))

# Classify with Grad-CAM
response = requests.post(
    "http://localhost:8000/classify/gradcam",
    json={"image": image.tolist()}
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Calibrated Confidence: {result['calibrated_confidence']:.2%}")

# Segment with uncertainty
response = requests.post(
    "http://localhost:8000/segment/uncertainty",
    json={
        "image": image.tolist(),
        "n_mc_samples": 10,
        "use_tta": True
    }
)
result = response.json()
print(f"Mean Dice: {result['mean_dice']:.3f}")
print(f"Epistemic Uncertainty: {result['epistemic_uncertainty']:.3f}")
print(f"Aleatoric Uncertainty: {result['aleatoric_uncertainty']:.3f}")
```

## 🧪 Testing

### Run Full E2E Test Suite

```bash
# Test all phases (1-6) with comprehensive validation
python scripts/test_full_e2e_phase1_to_phase6.py

# Expected output:
# [OK] Phase 1: Data pipeline (4/4 tests)
# [OK] Phase 2: Classification (5/5 tests)
# [OK] Phase 3: Segmentation (5/5 tests)
# [OK] Phase 4: Calibration & Uncertainty (4/4 tests)
# [OK] Phase 5: Metrics & Patient-Level (3/3 tests)
# [OK] Phase 6: API Integration (4/4 tests)
# Total: 25/25 tests passing (100%)
```

### Run Unit Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_classifier.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/ app/ scripts/
isort src/ tests/ app/ scripts/

# Lint code
ruff check src/ tests/ app/ scripts/

# Type checking
mypy src/
```

## 📈 Performance Metrics

### Classification Results (Kaggle Dataset)
- **Accuracy**: 94.6%
- **ROC-AUC**: 0.95+
- **PR-AUC**: 0.97+
- **Sensitivity**: 96.3%
- **Specificity**: 91.7%
- **ECE (before calibration)**: 0.0461
- **ECE (after calibration)**: 0.0147 (68.2% reduction)

### Segmentation Results (BraTS Dataset - Baseline)
- **Train Dice**: 0.860 ± 0.045
- **Val Dice**: 0.743 ± 0.089
- **Test Dice**: 0.708 ± 0.182
- **IoU**: 0.597
- **Specificity**: 0.998 (very conservative)

### Inference Performance
- **Throughput**: 2,551 images/sec (256×256, batch=32)
- **Latency**: 0.4ms per image (p50)
- **GPU Memory**: ~2.5GB peak usage
- **Classification**: ~50ms per image
- **Segmentation**: ~80ms per image
- **Uncertainty (MC+TTA)**: ~800ms per image

## 🎯 Roadmap

### ✅ Completed Phases

- [x] **Phase 0**: Project scaffolding, dependencies, CI/CD
- [x] **Phase 1**: Data acquisition & preprocessing (Kaggle + BraTS)
- [x] **Phase 2**: Classification MVP (EfficientNet + API + UI)
- [x] **Phase 3**: U-Net segmentation pipeline
- [x] **Phase 4**: Calibration & uncertainty estimation
- [x] **Phase 5**: Comprehensive metrics & patient-level evaluation
- [x] **Phase 6**: Demo application with 12 API endpoints & multi-tab UI
- [x] **Multi-Task Integration**: Unified architecture with 3-stage training
  - Stage 1: Segmentation warm-up (15.7M params)
  - Stage 2: Classification head training (263K params)
  - Stage 3: Joint fine-tuning (31.7M params total)
  - **Results**: 91.3% accuracy, 97.1% sensitivity, 76.5% Dice
  - **Benefits**: 9.4% fewer parameters, ~40% faster inference
- [x] **Frontend Refactor**: Modular UI architecture (87% code reduction)

### 🚧 In Progress

- [ ] **Phase 7**: Documentation & LaTeX write-up
  - [x] Update README with multi-task features
  - [x] Create SCRIPTS_REFERENCE.md with all 21 scripts
  - [x] Reorganize scripts by functionality
  - [ ] Write LaTeX report with methodology and results
  - [ ] Create presentation slides

### 📋 Planned

- [ ] **Phase 8**: Packaging & deployment
  - [ ] Docker containerization
  - [ ] Cloud deployment (AWS/GCP/Azure)
  - [ ] Model versioning and registry
  - [ ] Production monitoring and logging

See [FULL-PLAN.md](documentation/FULL-PLAN.md) for detailed roadmap.

## 📚 Documentation

### Quick Reference
- **[SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md)** - Complete reference for all 21 scripts with options and descriptions
- **[scripts/README.md](scripts/README.md)** - Scripts organization guide with workflows and troubleshooting
- **[FULL-PLAN.md](documentation/FULL-PLAN.md)** - Complete 8-phase roadmap with detailed checklists
- **[CONSOLIDATED_DOCUMENTATION.md](documentation/CONSOLIDATED_DOCUMENTATION.md)** - All phase documentation in one place
- **[MULTITASK_EVALUATION_REPORT.md](documentation/MULTITASK_EVALUATION_REPORT.md)** - Multi-task architecture analysis and results

### Technical Documentation
- **Data Pipeline**: See `src/data/` module docstrings
- **Model Architectures**: See `src/models/` module docstrings
- **Training**: See `src/training/` module docstrings
- **Evaluation**: See `src/eval/` module docstrings
- **API**: See `app/backend/main_v2.py` docstrings

## 📊 Project Statistics

- **Total Lines of Code**: ~18,700+
- **Number of Files**: 50+
- **Test Coverage**: 100% E2E coverage
- **Documentation**: 2,000+ lines
- **Phases Complete**: 7/8 (87.5%)

---

**Built with ❤️ for advancing medical AI research**

*SliceWise - Empowering clinicians with AI-powered brain tumor detection*
