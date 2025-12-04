# SliceWise Project Structure

This document describes the organization of the SliceWise codebase.

## Directory Layout

```
slicewise/
├── .github/                    # GitHub-specific files
│   └── workflows/
│       └── ci.yml             # CI/CD pipeline configuration
│
├── app/                       # Application code (to be created)
│   ├── backend/               # FastAPI backend
│   │   ├── main.py           # API entry point
│   │   ├── routers/          # API route handlers
│   │   └── utils/            # Backend utilities
│   └── frontend/             # Frontend application
│       ├── app.py            # Streamlit/Gradio app
│       └── components/       # UI components
│
├── assets/                    # Static assets and outputs (to be created)
│   ├── grad_cam_examples/    # Grad-CAM visualizations
│   ├── smoke_test/           # Smoke test outputs
│   └── figures/              # Figures for documentation
│
├── configs/                   # Configuration files
│   ├── hpc.yaml              # HPC environment config
│   ├── local.yaml            # Local development config
│   ├── config_cls.yaml       # Classification training config (to be created)
│   └── seg2d_baseline.yaml   # Segmentation training config (to be created)
│
├── data/                      # Data directory (gitignored, to be created)
│   ├── raw/                  # Raw datasets
│   │   ├── brats2020/
│   │   ├── brats2021/
│   │   └── kaggle_brain_mri/
│   └── processed/            # Preprocessed data
│       ├── brats2d/
│       └── kaggle/
│
├── documentation/             # Project documentation
│   ├── DATA_README.md        # Dataset documentation
│   ├── FEATURE_MAP.md        # Feature mapping
│   ├── FULL-PLAN.md          # Complete project roadmap
│   ├── PHASE0_COMPLETE.md    # Phase 0 completion summary
│   ├── PROJECT_STRUCTURE.md  # This file
│   ├── QUICKSTART.md         # Quick start guide
│   └── SETUP.md              # Installation and setup instructions
│
├── jupyter_notebooks/         # Jupyter notebooks
│   ├── MRI-Brain-Tumor-Detecor.ipynb  # Original notebook
│   ├── 01_visualize_brats_slices.ipynb (to be created)
│   ├── 02_eda.ipynb (to be created)
│   └── ablation_summary.ipynb (to be created)
│
├── scripts/                   # Utility scripts
│   ├── smoke_test.py         # Smoke test
│   ├── download_data.py      # Data download script (to be created)
│   ├── run_backend.sh        # Start backend server (to be created)
│   └── run_frontend.sh       # Start frontend app (to be created)
│
├── splits/                    # Train/val/test splits (to be created)
│   ├── train_patients.csv
│   ├── val_patients.csv
│   └── test_patients.csv
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── README.md             # Source code module documentation
│   │
│   ├── data/                 # Data processing (to be created)
│   │   ├── __init__.py
│   │   ├── brats2d_dataset.py        # BraTS 2D dataset
│   │   ├── kaggle_mri_dataset.py     # Kaggle dataset
│   │   ├── preprocess_brats_2d.py    # BraTS preprocessing
│   │   ├── split_patients.py         # Patient splitting
│   │   └── transforms.py             # Data augmentations
│   │
│   ├── models/               # Model architectures (to be created)
│   │   ├── __init__.py
│   │   ├── classifier.py     # Classification models
│   │   ├── unet2d.py        # 2D U-Net
│   │   ├── unetpp.py        # U-Net++ (optional)
│   │   └── deeplabv3.py     # DeepLabv3+ (optional)
│   │
│   ├── training/             # Training logic (to be created)
│   │   ├── __init__.py
│   │   ├── train_cls.py     # Classification training
│   │   ├── train_seg2d.py   # Segmentation training
│   │   ├── losses.py        # Loss functions
│   │   └── trainer.py       # Base trainer class
│   │
│   ├── eval/                 # Evaluation and metrics (to be created)
│   │   ├── __init__.py
│   │   ├── eval_cls.py      # Classification evaluation
│   │   ├── eval_seg2d.py    # Segmentation evaluation
│   │   ├── metrics.py       # Metric implementations
│   │   ├── calibration.py   # Calibration utilities
│   │   ├── grad_cam.py      # Grad-CAM implementation
│   │   ├── patient_level_eval.py  # Patient-level metrics
│   │   ├── run_ablations.py       # Ablation studies
│   │   └── profile_inference.py   # Latency profiling
│   │
│   └── inference/            # Inference utilities (to be created)
│       ├── __init__.py
│       ├── infer_seg2d.py   # Segmentation inference
│       └── postprocess.py   # Post-processing
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_smoke.py         # Basic smoke tests
│   ├── test_datasets.py      # (to be created)
│   ├── test_models.py        # (to be created)
│   ├── test_losses.py        # (to be created)
│   └── test_inference.py     # (to be created)
│
├── .gitignore                # Git ignore patterns
├── .pre-commit-config.yaml   # Pre-commit hooks
├── LICENSE                   # MIT License
├── pyproject.toml            # Project metadata and config
├── README.md                 # Main documentation
├── requirements.txt          # Python dependencies
└── setup.py                  # Setup script
```

## Module Descriptions

### `src/data/`
Data loading, preprocessing, and augmentation:
- **Dataset classes**: PyTorch datasets for BraTS and Kaggle data
- **Preprocessing**: Convert 3D volumes to 2D slices
- **Transforms**: Augmentation pipelines
- **Splitting**: Patient-level train/val/test splits

### `src/models/`
Neural network architectures:
- **Classifier**: EfficientNet/ConvNeXt for binary classification
- **U-Net**: 2D U-Net for segmentation
- **Advanced models**: U-Net++, DeepLabv3+ (optional)

### `src/training/`
Training loops and loss functions:
- **Training scripts**: End-to-end training pipelines
- **Losses**: Dice, BCE, Tversky, combined losses
- **Trainer**: Base class with common training logic

### `src/eval/`
Evaluation, metrics, and analysis:
- **Metrics**: Dice, IoU, ROC-AUC, calibration metrics
- **Grad-CAM**: Explainability visualizations
- **Calibration**: Temperature scaling
- **Ablations**: Systematic experiments
- **Profiling**: Latency and memory benchmarks

### `src/inference/`
Inference and post-processing:
- **Inference**: Run models on new data
- **Post-processing**: Thresholding, connected components, hole filling

### `app/`
Demo application:
- **Backend**: FastAPI REST API
- **Frontend**: Streamlit/Gradio UI

### `configs/`
YAML configuration files for different environments and experiments.

### `scripts/`
Utility scripts for common tasks (data download, testing, running apps).

### `jupyter_notebooks/`
Jupyter notebooks for exploration, visualization, and analysis.
- Contains the original `MRI-Brain-Tumor-Detecor.ipynb`
- Will contain new notebooks for data visualization and experiments

### `documentation/`
All project documentation files:
- **DATA_README.md**: Dataset access and organization
- **FEATURE_MAP.md**: Feature mapping and planning
- **FULL-PLAN.md**: Complete project roadmap (all 8 phases)
- **PHASE0_COMPLETE.md**: Phase 0 completion summary
- **PROJECT_STRUCTURE.md**: This file - codebase organization
- **QUICKSTART.md**: Quick start guide
- **SETUP.md**: Detailed installation instructions

### `tests/`
Unit tests for all modules (pytest).

## Key Files

- **`pyproject.toml`**: Modern Python packaging configuration
- **`requirements.txt`**: Dependency list
- **`setup.py`**: Backward-compatible setup script
- **`.pre-commit-config.yaml`**: Code quality hooks
- **`.github/workflows/ci.yml`**: CI/CD pipeline
- **`LICENSE`**: MIT License with medical disclaimer

## Development Workflow

1. **Setup**: Install dependencies with `pip install -e ".[dev]"`
2. **Pre-commit**: Install hooks with `pre-commit install`
3. **Testing**: Run tests with `pytest tests/`
4. **Formatting**: Auto-format with `black src/ tests/`
5. **Linting**: Check code with `ruff check src/ tests/`

## Adding New Components

### New Dataset
1. Create dataset class in `src/data/`
2. Add preprocessing script if needed
3. Update `documentation/DATA_README.md`

### New Model
1. Implement in `src/models/`
2. Add unit test in `tests/test_models.py`
3. Create training config in `configs/`

### New Metric
1. Implement in `src/eval/metrics.py`
2. Add unit test in `tests/test_metrics.py`
3. Integrate into evaluation scripts

### New Documentation
1. Add markdown files to `documentation/`
2. Update this PROJECT_STRUCTURE.md if needed
3. Link from main README.md

## Notes

- All data files are gitignored to prevent accidental commits
- Model weights (`.pth`, `.pt`) are gitignored
- Use configs for all hyperparameters (no hardcoding)
- Follow black/isort formatting (enforced by pre-commit)
- Write docstrings for all public functions/classes
- All documentation is centralized in `documentation/` folder
- Original notebook preserved in `jupyter_notebooks/`
