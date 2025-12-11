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

**Progress: 90% Complete (7/8 phases + Multi-Task + Frontend + Config System + Automated Pipeline) • ~20,000+ lines of code • 70+ files • 25+ organized scripts**

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
- 🚀 **FastAPI Backend**: 11 comprehensive REST endpoints
- 🎨 **Streamlit Frontend**: Refactored modular UI (15 files, 87% complexity reduction)
- 🧪 **Comprehensive Testing**: Full E2E test suite with 100% pass rate
- 📈 **Experiment Tracking**: W&B integration for training monitoring
- 🔧 **Flexible Configuration**: YAML-based configs for all components
- ⚡ **High Performance**: 2,500+ images/sec throughput, <1ms latency
- 🎯 **Educational**: Extensive documentation and code comments
- 📦 **Organized Scripts**: 25+ scripts organized by functionality
- 🎮 **Automated Pipeline**: ONE command from data to demo (6 steps)
- 🎯 **Smart Prompts**: Only 4 conditional Y/N prompts (if data exists)
- 📊 **Dynamic Scaling**: Auto-calculates dataset percentages (5%/30%/100%)

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or 3.12
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM
- Kaggle API credentials (for dataset download)
- **Node.js and npm** (for PM2 process manager - **highly recommended** for demo on Windows)

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

# 4. Install PM2 for demo process management (highly recommended for Windows)
npm install -g pm2

# 5. Generate training configs (hierarchical system)
python scripts/utils/merge_configs.py --all

# 6. Verify setup
python scripts/verify_setup.py
```

### 🎮 Full Pipeline Controller (RECOMMENDED) ⚡

The easiest way to train and deploy the complete multi-task model - **ONE COMMAND**:

```bash
# Quick test (5% data = 24 patients, 2 epochs, ~10-15 minutes) ⚡
python scripts/run_full_pipeline.py --mode full --training-mode quick

# Baseline training (30% data = 148 patients, 50 epochs, ~2-4 hours)
python scripts/run_full_pipeline.py --mode full --training-mode baseline

# Production training (100% data = 496 patients, 100 epochs, ~8-12 hours)
python scripts/run_full_pipeline.py --mode full --training-mode production
```

**What it does (6 automated stages):**
1. ✅ **Data Download**: BraTS 2020 + Kaggle datasets (with smart Y/N prompts if exists)
2. ✅ **Data Preprocessing**: 3D→2D extraction, normalization, filtering (dynamic scaling)
3. ✅ **Data Splitting**: Patient-level 70/15/15 split (prevents leakage)
4. ✅ **Multi-Task Training**: 3-stage training (seg warmup → cls head → joint)
5. ✅ **Comprehensive Evaluation**: Metrics, Grad-CAM, phase comparison
6. ✅ **Demo Launch**: FastAPI + Streamlit with PM2 process management

**Smart User Interaction** (only if data exists):
- Re-download BraTS? (y/N)
- Re-download Kaggle? (y/N)
- Re-preprocess BraTS? (y/N)
- Re-preprocess Kaggle? (y/N)

**Everything else**: Fully automated!

**Expected Performance:**
- **Quick Mode**: Acc ~75-85%, Dice ~0.60-0.70 (**~10-15 min** - 2x faster!) ⚡
- **Baseline Mode**: Acc ~85-90%, Dice ~0.70-0.75 (~2-4 hours)
- **Production Mode**: Acc ~91-93%, Sensitivity ~95-97%, Dice ~0.75-0.80 (~8-12 hours)

**Dynamic Dataset Scaling**:
- Quick: 5% of available patients (min 2)
- Baseline: 30% of available patients (min 50)
- Production: 100% of available patients

See `scripts/README.md` and `scripts/FULL_PIPELINE_SUMMARY.md` for full documentation.

### 🎬 Run the Demo Application (Pre-trained Model)

If you already have a trained model:

#### Option 1: PM2 Process Manager (Recommended for Windows)

```bash
# Start both backend and frontend with PM2
python scripts/demo/run_demo_pm2.py

# Or use PM2 directly
pm2 start configs/pm2-ecosystem/ecosystem.config.js

# Manage processes
pm2 status              # View status
pm2 logs                # View logs
pm2 monit               # Interactive monitoring
pm2 stop all            # Stop demo
pm2 delete all          # Stop and remove
```

**PM2 Benefits:**
- ✅ Auto-restart on crash
- ✅ Centralized logging (`logs/` directory)
- ✅ Background execution
- ✅ Windows subprocess compatibility
- ✅ Easy monitoring and management

#### Option 2: Manual Launch (2 Terminals)

```bash
# Terminal 1: Backend
python scripts/demo/run_demo_backend.py  # http://localhost:8000

# Terminal 2: Frontend
python scripts/demo/run_demo_frontend.py # http://localhost:8501
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
python scripts/data/collection/download_kaggle_data.py

# 3. Preprocess to .npz format
python src/data/preprocess_kaggle.py

# 4. Create train/val/test splits
python scripts/data/splitting/split_kaggle_data.py

# Done! Dataset ready at data/processed/kaggle/{train,val,test}/
```

**Result**: 171 train / 37 val / 37 test images, stratified by class

#### BraTS Dataset (Advanced - For Segmentation)

```bash
# 1. Download BraTS 2020 dataset (988 patients, ~80GB)
python scripts/data/collection/download_brats_data.py

# 2. Preprocess 3D volumes to 2D slices
python scripts/data/preprocessing/preprocess_all_brats.py

# 3. Create patient-level splits (prevents data leakage)
python scripts/data/splitting/split_brats_data.py

# Done! Dataset ready at data/processed/brats2d/{train,val,test}/
```

See `documentation/BRATS_DATASET_GUIDE.md` for detailed instructions.

## 🏗️ Project Structure

```
MRI-CV-MODEL-TRAINING-AND-INFERENCE-PROJECT/
├── src/                              # Source code (~13,000+ lines)
│   ├── data/                         # Data pipeline (12 files)
│   │   ├── kaggle_mri_dataset.py     # Kaggle dataset class
│   │   ├── brats2d_dataset.py        # BraTS 2D dataset class
│   │   ├── brats_classification_dataset.py  # BraTS classification dataset
│   │   ├── preprocess_kaggle.py      # Kaggle preprocessing
│   │   ├── preprocess_brats_2d.py    # BraTS 3D→2D extraction
│   │   ├── split_kaggle.py           # Kaggle train/val/test split
│   │   ├── split_brats.py            # BraTS patient-level split
│   │   └── transforms.py             # Augmentation pipeline
│   ├── models/                       # Model architectures (8 files)
│   │   ├── classifier.py             # EfficientNet-B0 & ConvNeXt
│   │   ├── unet2d.py                 # U-Net 2D (31.4M params)
│   │   ├── multi_task_model.py       # Unified multi-task architecture
│   │   ├── unet_encoder.py           # Shared encoder (15.7M params)
│   │   ├── unet_decoder.py           # Segmentation decoder (15.7M params)
│   │   └── classification_head.py    # Classification head (263K params)
│   ├── training/                     # Training pipelines (8 files)
│   │   ├── train_cls.py              # Classifier training
│   │   ├── train_seg2d.py            # Segmentation training
│   │   ├── train_multitask_seg_warmup.py   # Stage 1: Seg warmup
│   │   ├── train_multitask_cls_head.py     # Stage 2: Cls head
│   │   ├── train_multitask_joint.py        # Stage 3: Joint fine-tuning
│   │   ├── losses.py                 # Loss functions (Dice, Focal, etc.)
│   │   └── multi_task_losses.py      # Combined loss functions
│   ├── eval/                         # Evaluation & metrics (8 files)
│   │   ├── eval_cls.py               # Classifier evaluation
│   │   ├── eval_seg2d.py             # Segmentation evaluation
│   │   ├── calibration.py            # Temperature scaling
│   │   ├── metrics.py                # Comprehensive metrics
│   │   ├── patient_level_eval.py     # Patient-level analysis
│   │   ├── profile_inference.py      # Performance profiling
│   │   └── grad_cam.py               # Grad-CAM explainability
│   └── inference/                    # Inference pipeline (6 files)
│       ├── predict.py                # Classifier predictor
│       ├── infer_seg2d.py            # Segmentation predictor
│       ├── multi_task_predictor.py   # Multi-task predictor
│       ├── uncertainty.py            # MC Dropout + TTA
│       └── postprocess.py            # Post-processing utilities
├── app/                              # Demo application
│   ├── backend/                      # FastAPI backend
│   │   ├── main.py                   # Legacy API (Phase 2)
│   │   └── main_v2.py                # Production API (12 endpoints, 5 routers)
│   └── frontend/                     # Streamlit frontend (modular)
│       ├── app.py                    # Main orchestrator (151 lines)
│       ├── app_v2.py                 # Legacy monolithic (1,187 lines)
│       ├── components/               # UI components (8 files)
│       │   ├── header.py             # App header
│       │   ├── sidebar.py            # System status
│       │   ├── multitask_tab.py      # Multi-task UI
│       │   ├── classification_tab.py # Classification UI
│       │   ├── segmentation_tab.py   # Segmentation UI
│       │   ├── batch_tab.py          # Batch processing
│       │   └── patient_tab.py        # Patient analysis
│       ├── config/                   # Configuration
│       │   └── settings.py           # Centralized settings
│       ├── styles/                   # CSS styling
│       │   ├── theme.css             # Theme variables
│       │   └── main.css              # Component styles
│       └── utils/                    # Utilities (3 files)
│           ├── api_client.py         # API communication
│           ├── image_utils.py        # Image processing
│           └── validators.py         # Input validation
├── scripts/                          # Utility scripts (25+ files)
│   ├── run_full_pipeline.py          # 🎮 Full pipeline controller
│   ├── data/                         # Data management
│   │   ├── collection/               # Dataset download scripts
│   │   ├── preprocessing/            # Preprocessing scripts
│   │   └── splitting/                # Data splitting scripts
│   ├── demo/                         # Demo launchers (5 files)
│   │   ├── run_demo_pm2.py           # PM2-based launcher (recommended)
│   │   ├── run_demo_backend.py       # Backend launcher
│   │   ├── run_demo_frontend.py      # Frontend launcher
│   │   └── manual_demo_*.py          # Manual demo scripts
│   ├── evaluation/                   # Evaluation scripts
│   │   ├── multitask/                # Multi-task evaluation
│   │   └── testing/                  # Testing scripts
│   ├── training/                     # Training launchers
│   └── utils/                        # Utility tools
│       └── merge_configs.py          # Config merger (hierarchical system)
├── configs/                          # 🔧 Hierarchical Configuration System
│   ├── base/                         # Base configs (5 files)
│   │   ├── common.yaml               # Common settings
│   │   ├── model_architectures.yaml  # Model presets
│   │   ├── training_defaults.yaml    # Training defaults
│   │   ├── augmentation_presets.yaml # Augmentation presets
│   │   └── platform_overrides.yaml   # Platform-specific settings
│   ├── stages/                       # Training stages (3 files)
│   │   ├── stage1_seg_warmup.yaml    # Stage 1: Segmentation warmup
│   │   ├── stage2_cls_head.yaml      # Stage 2: Classification head
│   │   └── stage3_joint.yaml         # Stage 3: Joint fine-tuning
│   ├── modes/                        # Training modes (3 files)
│   │   ├── quick_test.yaml           # Quick test (5% data, 2 epochs)
│   │   ├── baseline.yaml             # Baseline (30% data, 50 epochs)
│   │   └── production.yaml           # Production (100% data, 100 epochs)
│   ├── final/                        # 🤖 Auto-generated configs (9 files, gitignored)
│   │   └── stage{1,2,3}_{quick,baseline,production}.yaml
│   ├── pm2-ecosystem/                # PM2 process management
│   │   └── ecosystem.config.js       # PM2 configuration
│   └── README.md                     # Config system documentation
├── tests/                            # Unit tests (8 files)
│   ├── test_classifier.py            # Classifier tests
│   ├── test_data_pipeline.py         # Data pipeline tests
│   ├── test_config_generation.py     # Config system tests (27 tests)
│   ├── test_gradcam.py               # Grad-CAM tests
│   └── test_segmentation.py          # Segmentation tests
├── documentation/                    # Comprehensive documentation (15+ files)
│   ├── FULL-PLAN.md                  # Complete 8-phase roadmap
│   ├── PIPELINE_CONTROLLER_GUIDE.md  # Pipeline controller guide
│   ├── PM2_DEMO_GUIDE.md             # PM2 usage guide
│   ├── CONFIG_GUIDE.md               # Config system guide
│   ├── FRONTEND_REFACTORING.md       # Frontend refactoring summary
│   └── MULTITASK_EVALUATION_REPORT.md # Multi-task analysis
├── jupyter_notebooks/                # Analysis notebooks
│   └── MRI-Brain-Tumor-Detecor.ipynb # Original exploration
├── logs/                             # PM2 logs (gitignored)
│   ├── backend-*.log                 # Backend logs
│   └── frontend-*.log                # Frontend logs
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

## ⚙️ Configuration System

### Hierarchical Configuration Architecture

SliceWise uses a **hierarchical configuration system** that eliminates 70-90% duplication across training configs:

```bash
# Generate all 9 training configs (stage1-3 × quick/baseline/production)
python scripts/utils/merge_configs.py --all

# Generate single config
python scripts/utils/merge_configs.py --stage 1 --mode quick

# Validate config generation
pytest tests/test_config_generation.py -v  # 27 tests, 100% pass rate
```

**Architecture:**
```
base/common.yaml (dataset paths, device settings)
  ↓ (deep merge)
base/training_defaults.yaml (optimizer, scheduler, loss)
  ↓ (deep merge)
stages/stageN_*.yaml (stage-specific settings)
  ↓ (deep merge)
modes/MODE.yaml (quick/baseline/production overrides)
  ↓ (resolve references)
final/stageN_MODE.yaml (auto-generated, ready to use)
```

**Benefits:**
- ✅ **64% reduction** in config lines (1,100 → 365 base)
- ✅ **100% duplication eliminated** (was 70-90%)
- ✅ **87% less work** to change parameters (1 file vs 8)
- ✅ **Guaranteed consistency** (auto-generated)
- ✅ **Reference resolution** for model architectures and augmentation presets

**Example References:**
```yaml
# In your config
model:
  architecture: "multitask_medium"  # Expands to full model params

augmentation:
  preset: "moderate"  # Expands to full augmentation config
```

See `configs/README.md` and `documentation/CONFIG_GUIDE.md` for full documentation.

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

The FastAPI backend is organized into **5 modular routers** with **11 comprehensive endpoints**:

### Health & Info (`routers/health.py`)
- `GET /` - API information and endpoint overview
- `GET /healthz` - Health check with model status
- `GET /model/info` - Detailed model information and capabilities

### Classification (`routers/classification.py`)
- `POST /classify` - Single image classification with confidence
- `POST /classify/gradcam` - Classification with Grad-CAM visualization
- `POST /classify/batch` - Batch classification (up to 100 images)

### Segmentation (`routers/segmentation.py`)
- `POST /segment` - Single image segmentation with binary mask
- `POST /segment/uncertainty` - Segmentation with MC Dropout + TTA uncertainty
- `POST /segment/batch` - Batch segmentation (up to 100 images)

### Multi-Task (`routers/multitask.py`)
- `POST /predict_multitask` - Unified classification + conditional segmentation

### Patient Analysis (`routers/patient.py`)
- `POST /patient/analyze_stack` - Patient-level analysis with volume estimation

### Architecture Benefits
- **Service Layer**: Business logic separated from HTTP concerns
- **Dependency Injection**: Clean, testable code with proper DI
- **Modular Design**: Each router handles specific functionality
- **Error Handling**: Centralized exception handling middleware
- **Validation**: Comprehensive input validation with Pydantic models

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
- [x] **Config System Refactor**: Hierarchical configuration (64% reduction, 100% duplication eliminated)
- [x] **PM2 Integration**: Process management for reliable demo deployment
- [x] **Automated Pipeline Controller**: ONE command full pipeline with smart prompts and dynamic scaling
  - 6-step automation (download → preprocess → split → train → evaluate → demo)
  - Smart Y/N prompts (only for existing data)
  - Dynamic dataset scaling (5%/30%/100% based on mode)
  - Quick mode optimized to 5% (min 2 patients) - 2x faster!

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
- **[scripts/README.md](scripts/README.md)** - Complete scripts reference with automated pipeline guide (updated Dec 10, 2025)
- **[scripts/FULL_PIPELINE_SUMMARY.md](scripts/FULL_PIPELINE_SUMMARY.md)** - Quick pipeline reference with examples
- **[SCRIPTS_REFERENCE.md](SCRIPTS_REFERENCE.md)** - Detailed reference for all 25+ scripts
- **[FULL-PLAN.md](documentation/FULL-PLAN.md)** - Complete 8-phase roadmap with detailed checklists
- **[CONSOLIDATED_DOCUMENTATION.md](documentation/CONSOLIDATED_DOCUMENTATION.md)** - All phase documentation in one place
- **[MULTITASK_EVALUATION_REPORT.md](documentation/MULTITASK_EVALUATION_REPORT.md)** - Multi-task architecture analysis and results
- **[documentation/DATASET_LOADING_OPTIMIZATION.md](documentation/DATASET_LOADING_OPTIMIZATION.md)** - Dynamic dataset scaling guide

### Configuration & Deployment
- **[configs/README.md](configs/README.md)** - Hierarchical config system documentation (400+ lines)
- **[documentation/CONFIG_GUIDE.md](documentation/CONFIG_GUIDE.md)** - Complete config refactoring summary
- **[documentation/PM2_DEMO_GUIDE.md](documentation/PM2_DEMO_GUIDE.md)** - PM2 process management guide (488 lines)
- **[documentation/PM2_SETUP_SUMMARY.md](documentation/PM2_SETUP_SUMMARY.md)** - Quick PM2 reference

### Frontend Architecture
- **[app/frontend/README.md](app/frontend/README.md)** - Modular frontend architecture guide (492 lines)
- **[documentation/FRONTEND_REFACTORING.md](documentation/FRONTEND_REFACTORING.md)** - Frontend refactoring summary (461 lines)

### Technical Documentation
- **Data Pipeline**: See `src/data/` module docstrings
- **Model Architectures**: See `src/models/` module docstrings
- **Training**: See `src/training/` module docstrings
- **Evaluation**: See `src/eval/` module docstrings
- **API**: See `app/backend/main_v2.py` docstrings

## 📊 Project Statistics

- **Total Lines of Code**: ~20,000+
- **Number of Files**: 70+
- **Scripts**: 25+ organized scripts
- **Test Coverage**: 100% E2E coverage (25/25 tests passing)
- **Config Tests**: 27 unit tests (100% pass rate)
- **Documentation**: 5,000+ lines across 15+ files
- **Phases Complete**: 7/8 (87.5%)
- **Major Refactors**: 3 (Frontend, Config System, PM2 Integration)
- **Automation**: 1 command, 6 steps, 4 conditional prompts
- **Quick Mode**: 2x faster (5% data, ~10-15 min)

**Recent Updates (December 10, 2025)**:
- ⚡ Quick mode optimized to 5% dataset (min 2 patients) - 2x faster!
- 🎯 Smart Y/N prompts (only for existing data)
- 📊 Dynamic dataset scaling (adapts to any dataset size)
- 🔧 Individual download/preprocessing controls
- 📝 Comprehensive documentation updates

---

**Built with ❤️ for advancing medical AI research**

*SliceWise - Empowering clinicians with AI-powered brain tumor detection*
