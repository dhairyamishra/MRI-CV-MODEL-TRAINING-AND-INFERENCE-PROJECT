# Source Code Organization

This directory contains the core SliceWise implementation.

## Modules

### `data/`
Data loading, preprocessing, and augmentation pipelines.

**Key files:**
- `brats2d_dataset.py` - PyTorch dataset for BraTS 2D slices
- `kaggle_mri_dataset.py` - PyTorch dataset for Kaggle Brain MRI
- `preprocess_brats_2d.py` - Convert 3D BraTS volumes to 2D slices
- `split_patients.py` - Create train/val/test splits at patient level
- `transforms.py` - Data augmentation transforms

### `models/`
Neural network architectures for classification and segmentation.

**Key files:**
- `classifier.py` - Binary classification models (EfficientNet, ConvNeXt)
- `unet2d.py` - 2D U-Net for segmentation
- `unetpp.py` - U-Net++ architecture (optional)
- `deeplabv3.py` - DeepLabv3+ architecture (optional)

### `training/`
Training loops, loss functions, and optimization.

**Key files:**
- `train_cls.py` - Classification training script
- `train_seg2d.py` - Segmentation training script
- `losses.py` - Loss functions (Dice, BCE, Tversky, etc.)
- `trainer.py` - Base trainer class with common logic

### `eval/`
Evaluation, metrics, calibration, and analysis.

**Key files:**
- `eval_cls.py` - Classification evaluation
- `eval_seg2d.py` - Segmentation evaluation
- `metrics.py` - Metric implementations (Dice, IoU, ROC-AUC, etc.)
- `calibration.py` - Temperature scaling and calibration
- `grad_cam.py` - Grad-CAM for explainability
- `patient_level_eval.py` - Patient-level aggregation
- `run_ablations.py` - Ablation study runner
- `profile_inference.py` - Latency and memory profiling

### `inference/`
Inference pipelines and post-processing.

**Key files:**
- `infer_seg2d.py` - Segmentation inference
- `postprocess.py` - Post-processing (thresholding, connected components)

## Usage

All modules are designed to be imported and used programmatically:

```python
from src.data import BraTS2DSliceDataset
from src.models import UNet2D
from src.training import DiceLoss
from src.inference import predict_slice
```

Or run scripts directly:

```bash
python -m src.training.train_seg2d --config configs/seg2d_baseline.yaml
python -m src.eval.eval_seg2d --checkpoint checkpoints/best_model.pth
```

## Development

When adding new functionality:
1. Add implementation in appropriate module
2. Add unit tests in `tests/`
3. Update this README if adding new files
4. Follow code style (black, isort, ruff)
5. Add docstrings for all public functions/classes

For more details, see:
- `documentation/PROJECT_STRUCTURE.md` - Complete project organization
- `documentation/SETUP.md` - Development environment setup
- `documentation/QUICKSTART.md` - Quick start guide
