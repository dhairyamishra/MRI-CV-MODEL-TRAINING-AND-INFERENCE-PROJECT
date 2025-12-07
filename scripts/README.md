# Scripts Directory Organization

## Overview

This directory contains all executable scripts for the **SliceWise MRI Brain Tumor Detection** project. Scripts are organized by functionality and workflow stage to improve maintainability and discoverability.

## Directory Structure

```
scripts/
├── data/                    # Data pipeline scripts
│   ├── collection/         # Raw data acquisition
│   ├── preprocessing/      # Data transformation and preparation
│   └── splitting/          # Train/validation/test splitting
├── training/               # Model training scripts
│   ├── multitask/          # Multi-task learning pipelines
│   └── utils/              # Training utilities
├── evaluation/             # Model evaluation and testing
│   ├── multitask/          # Multi-task model evaluation
│   └── testing/            # End-to-end and component testing
├── demo/                   # Demo and application launchers
└── debug/                  # Debugging and diagnostic tools
```

## 📊 Scripts by Category

### 🔍 Data Pipeline Scripts

#### `data/collection/`
**Purpose**: Download and acquire raw datasets for training and testing.

| Script | Description | Datasets | Size | Runtime |
|--------|-------------|----------|------|---------|
| `download_brats_data.py` | Downloads BraTS 2020 dataset | 988 patients, 4 modalities | ~15GB | 10-30 min |
| `download_kaggle_data.py` | Downloads Kaggle brain tumor dataset | 3,000+ images | ~500MB | 2-5 min |

**Usage**:
```bash
# Download BraTS dataset
python scripts/data/collection/download_brats_data.py

# Download Kaggle dataset
python scripts/data/collection/download_kaggle_data.py
```

#### `data/preprocessing/`
**Purpose**: Convert raw data into training-ready format.

| Script | Description | Input | Output | Runtime |
|--------|-------------|-------|--------|---------|
| `preprocess_all_brats.py` | 3D→2D conversion, normalization | BraTS NIfTI files | 2D slices + metadata | 5-15 min |
| `export_dataset_examples.py` | Create visualization samples | Processed data | PNG examples + JSON | 2-5 min |

**Usage**:
```bash
# Preprocess BraTS data (988 patients)
python scripts/data/preprocessing/preprocess_all_brats.py

# Export dataset examples for visualization
python scripts/data/preprocessing/export_dataset_examples.py
```

#### `data/splitting/`
**Purpose**: Create patient-level train/validation/test splits to prevent data leakage.

| Script | Description | Strategy | Ratio |
|--------|-------------|----------|-------|
| `split_brats_data.py` | Patient-level splitting for BraTS | 70/15/15 | 988 patients |
| `split_kaggle_data.py` | Patient-level splitting for Kaggle | 70/15/15 | 3,000+ images |

**Usage**:
```bash
# Split BraTS dataset
python scripts/data/splitting/split_brats_data.py

# Split Kaggle dataset
python scripts/data/splitting/split_kaggle_data.py
```

### 🏋️ Training Scripts

#### `training/multitask/`
**Purpose**: Train the unified multi-task model (classification + segmentation).

| Script | Description | Stage | Parameters | Runtime |
|--------|-------------|-------|------------|---------|
| `train_multitask_seg_warmup.py` | Warm up segmentation decoder | 1/3 | 15.7M params | 10-20 min |
| `train_multitask_cls_head.py` | Train classification head | 2/3 | 263K params | 5-15 min |
| `train_multitask_joint.py` | Joint fine-tuning | 3/3 | 31.7M params | 15-30 min |

**3-Stage Training Pipeline**:
```bash
# Stage 1: Segmentation warm-up (freeze encoder, train decoder)
python scripts/training/multitask/train_multitask_seg_warmup.py

# Stage 2: Classification head training (freeze encoder/decoder, train cls head)
python scripts/training/multitask/train_multitask_cls_head.py

# Stage 3: Joint fine-tuning (unfreeze all, differential learning rates)
python scripts/training/multitask/train_multitask_joint.py
```

**Key Features**:
- **Shared Encoder**: 49.5% of parameters (15.7M)
- **Differential LR**: Encoder=1e-4, Decoder/Cls=3e-4
- **Mixed Precision**: AMP training
- **Early Stopping**: Validation monitoring

#### `training/utils/`
| Script | Description | Output |
|--------|-------------|--------|
| `generate_model_configs.py` | Generate YAML configs for training | `configs/multitask_*.yaml` |

### 📈 Evaluation Scripts

#### `evaluation/multitask/`
**Purpose**: Evaluate trained multi-task models and generate insights.

| Script | Description | Metrics | Output |
|--------|-------------|---------|--------|
| `evaluate_multitask.py` | Comprehensive evaluation | Dice, IoU, Acc, ROC-AUC | `results/multitask_evaluation/` |
| `generate_multitask_gradcam.py` | Grad-CAM visualizations | Heatmaps + overlays | `visualizations/multitask_gradcam/` |
| `compare_all_phases.py` | Phase comparison analysis | Statistical tests | `results/phase_comparison.json` |

**Usage**:
```bash
# Full evaluation suite
python scripts/evaluation/multitask/evaluate_multitask.py

# Generate Grad-CAM visualizations
python scripts/evaluation/multitask/generate_multitask_gradcam.py --num-samples 50

# Compare all training phases
python scripts/evaluation/multitask/compare_all_phases.py
```

#### `evaluation/testing/`
**Purpose**: Test scripts for validation and debugging.

| Script | Description | Scope | Runtime |
|--------|-------------|-------|---------|
| `test_multitask_e2e.py` | End-to-end pipeline test | Full integration | 30-60 sec |
| `test_backend_startup.py` | API backend validation | Backend only | 5-10 sec |
| `test_brain_crop.py` | Brain cropping validation | Preprocessing | 10-20 sec |

**Usage**:
```bash
# Run full E2E test suite
python scripts/evaluation/testing/test_multitask_e2e.py

# Test backend startup
python scripts/evaluation/testing/test_backend_startup.py

# Test brain cropping
python scripts/evaluation/testing/test_brain_crop.py
```

### 🎬 Demo Scripts

**Purpose**: Launch interactive demos and applications.

| Script | Description | Components | URLs |
|--------|-------------|------------|-------|
| `run_multitask_demo.py` | **Current Multi-Task Demo** | Backend + Frontend | http://localhost:8501 |
| `run_demo.py` | Legacy Phase 6 Demo | Separate models | http://localhost:8501 |
| `run_demo_backend.py` | Backend only launcher | API server | http://localhost:8000 |
| `run_demo_frontend.py` | Frontend only launcher | Streamlit UI | http://localhost:8501 |

**Quick Start - Multi-Task Demo**:
```bash
# Launch unified demo (recommended)
python scripts/demo/run_multitask_demo.py

# Features:
# - Multi-Task tab: Classification + Segmentation
# - Conditional segmentation (only if tumor prob >= 30%)
# - Grad-CAM visualization
# - Performance metrics
# - Clinical recommendations
```

**Legacy Demo (Phase 6)**:
```bash
# Separate model demo
python scripts/demo/run_demo.py
```

### 🔧 Debug Scripts

| Script | Description | Use Case |
|--------|-------------|----------|
| `debug_multitask_data.py` | Data pipeline debugging | Dataset validation, batch inspection |

## 🚀 Quick Start Workflows

### **Complete Pipeline (New Users)**
```bash
# 1. Data acquisition
python scripts/data/collection/download_brats_data.py
python scripts/data/collection/download_kaggle_data.py

# 2. Data preparation
python scripts/data/preprocessing/preprocess_all_brats.py
python scripts/data/splitting/split_brats_data.py
python scripts/data/splitting/split_kaggle_data.py

# 3. Training (3 stages)
python scripts/training/multitask/train_multitask_seg_warmup.py
python scripts/training/multitask/train_multitask_cls_head.py
python scripts/training/multitask/train_multitask_joint.py

# 4. Evaluation
python scripts/evaluation/multitask/evaluate_multitask.py

# 5. Testing
python scripts/evaluation/testing/test_multitask_e2e.py

# 6. Demo
python scripts/demo/run_multitask_demo.py
```

### **Resume Training (Existing Data)**
```bash
# Jump to training
python scripts/training/multitask/train_multitask_joint.py --resume

# Quick evaluation
python scripts/evaluation/multitask/evaluate_multitask.py
```

### **Demo Only (Pre-trained Models)**
```bash
# Launch demo with pre-trained models
python scripts/demo/run_multitask_demo.py
```

## 📋 Prerequisites

### **System Requirements**
- **Python**: 3.11+ (tested on 3.13)
- **CUDA**: 11.8+ (recommended for training)
- **RAM**: 16GB+ (32GB+ recommended)
- **Storage**: 50GB+ free space

### **Dependencies**
Install all requirements:
```bash
pip install -r requirements.txt
```

### **Data Requirements**
- **BraTS 2020**: ~15GB (download via scripts)
- **Kaggle Dataset**: ~500MB (download via scripts)
- **Pre-trained Models**: Available in `checkpoints/` directory

### **Model Checkpoints**
Expected checkpoint locations:
```
checkpoints/
├── multitask_seg_warmup/best_model.pth
├── multitask_cls_head/best_model.pth
└── multitask_joint/best_model.pth
```

## 🎯 Key Features

### **Multi-Task Architecture**
- **Unified Model**: Single forward pass for classification + segmentation
- **Shared Encoder**: EfficientNet-B0 backbone (15.7M parameters)
- **Conditional Logic**: Segmentation only when tumor probability ≥ 30%
- **Performance**: ~40% faster inference, 9.4% fewer parameters

### **Clinical Performance**
- **Classification**: 91.3% accuracy, 97.1% sensitivity
- **Segmentation**: 76.5% Dice, 64.0% IoU
- **Combined Metric**: 83.9% (harmonic mean)

### **Production Ready**
- **Mixed Precision**: Automatic mixed precision training
- **Early Stopping**: Validation monitoring with patience
- **Checkpointing**: Best model saving + resume capability
- **Logging**: Weights & Biases integration
- **Error Handling**: Robust exception handling

## 🔧 Configuration

### **Training Configs**
Located in `configs/` directory:
- `config_multitask_seg_warmup.yaml`
- `config_multitask_cls_head.yaml`
- `config_multitask_joint.yaml`

### **Environment Variables**
```bash
# Optional: Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Optional: Weights & Biases
export WANDB_API_KEY=your_key_here
```

## 🐛 Troubleshooting

### **Common Issues**

**"Checkpoint not found"**
```bash
# Check if training completed
ls -la checkpoints/multitask_joint/

# Re-run training if missing
python scripts/training/multitask/train_multitask_joint.py
```

**"CUDA out of memory"**
```bash
# Reduce batch size in config
vim configs/config_multitask_joint.yaml
# Change: batch_size: 16 -> batch_size: 8
```

**"Port already in use"**
```bash
# Kill existing processes
pkill -f streamlit
pkill -f uvicorn

# Or use different ports
python scripts/demo/run_multitask_demo.py --frontend-port 8502 --backend-port 8001
```

**"Import errors"**
```bash
# Ensure you're in project root
cd /path/to/project

# Install dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### **Debug Mode**
```bash
# Run with verbose logging
python scripts/evaluation/testing/test_multitask_e2e.py --verbose

# Debug data pipeline
python scripts/debug/debug_multitask_data.py
```

## 📖 Additional Resources

- **Main Documentation**: `documentation/CONSOLIDATED_DOCUMENTATION.md`
- **Training Results**: `results/multitask_evaluation/`
- **Model Checkpoints**: `checkpoints/multitask_joint/`
- **Visualizations**: `visualizations/multitask_gradcam/`
- **API Documentation**: http://localhost:8000/docs (when running)

## 🤝 Contributing

When adding new scripts:
1. **Follow naming convention**: `snake_case.py`
2. **Add docstrings**: Comprehensive function/class documentation
3. **Include usage examples**: In script headers
4. **Update this README**: Add new script to appropriate section
5. **Test thoroughly**: Run relevant evaluation scripts

---

**Last Updated**: December 7, 2025
**Project Phase**: Multi-Task Integration Complete
**Total Scripts**: 21 organized scripts
