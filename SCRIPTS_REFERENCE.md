# Scripts Reference Guide

**SliceWise MRI Brain Tumor Detection - Complete Scripts Reference**

All scripts must be run from the project root directory. Format: `python scripts/<path>/<script>.py [options]`

---

## 📦 Data Collection

- `python scripts/data/collection/download_brats_data.py [--year {2020,2021}] [--output DIR]` - Downloads BraTS dataset from Kaggle (988 patients, ~15GB, 10-30 min)
- `python scripts/data/collection/download_kaggle_data.py [--output DIR]` - Downloads Kaggle brain MRI dataset (3000+ images, ~500MB, 2-5 min)

## 🔄 Data Preprocessing

- `python scripts/data/preprocessing/preprocess_all_brats.py [--input DIR] [--output DIR] [--num-patients N] [--modality {flair,t1,t1ce,t2}] [--normalization {zscore,minmax,percentile}] [--min-tumor-pixels N] [--no-split]` - Converts 3D BraTS NIfTI to 2D slices with normalization and filtering (5-15 min for 100 patients, 2-4 hours for all 988)
- `python scripts/data/preprocessing/export_dataset_examples.py [--kaggle-dir DIR] [--brats-dir DIR] [--output-dir DIR] [--num-samples N] [--tumor-only] [--no-tumor-only] [--save-comparison]` - Exports dataset samples as PNG images with metadata JSON for visualization (2-5 min)

## ✂️ Data Splitting

- `python scripts/data/splitting/split_brats_data.py [--input DIR] [--output DIR] [--train-ratio FLOAT] [--val-ratio FLOAT] [--test-ratio FLOAT] [--seed INT] [--min-tumor-slices N]` - Creates patient-level train/val/test splits for BraTS (default 70/15/15, prevents data leakage)
- `python scripts/data/splitting/split_kaggle_data.py [--input DIR] [--output DIR] [--train-ratio FLOAT] [--val-ratio FLOAT] [--test-ratio FLOAT] [--seed INT]` - Creates patient-level train/val/test splits for Kaggle dataset (default 70/15/15)

## 🏋️ Training - Multi-Task Pipeline

- `python scripts/training/multitask/train_multitask_seg_warmup.py [--config PATH] [--checkpoint-dir DIR] [--resume PATH] [--epochs N] [--batch-size N] [--lr FLOAT]` - Stage 1/3: Warm up segmentation decoder with frozen encoder (15.7M params, 10-20 min)
- `python scripts/training/multitask/train_multitask_cls_head.py --encoder-init PATH [--config PATH] [--checkpoint-dir DIR] [--resume PATH]` - Stage 2/3: Train classification head with frozen encoder/decoder (263K params, 5-15 min, requires Phase 1 checkpoint)
- `python scripts/training/multitask/train_multitask_joint.py --init-from PATH [--config PATH] [--checkpoint-dir DIR] [--resume PATH]` - Stage 3/3: Joint fine-tuning with differential learning rates (31.7M params, 15-30 min, requires Phase 2 checkpoint)

## 🛠️ Training Utilities

- `python scripts/training/utils/generate_model_configs.py` - Generates YAML configuration files for multi-task training (outputs to configs/multitask_*.yaml)

## 📊 Evaluation - Multi-Task

- `python scripts/evaluation/multitask/evaluate_multitask.py [--checkpoint PATH] [--config PATH] [--output-dir DIR] [--batch-size N] [--device {cuda,cpu}]` - Comprehensive evaluation of multi-task model with Dice, IoU, Accuracy, ROC-AUC metrics (outputs to results/multitask_evaluation/)
- `python scripts/evaluation/multitask/generate_multitask_gradcam.py [--checkpoint PATH] [--config PATH] [--output-dir DIR] [--num-samples N] [--device {cuda,cpu}]` - Generates Grad-CAM heatmap visualizations for interpretability (outputs to visualizations/multitask_gradcam/)
- `python scripts/evaluation/multitask/compare_all_phases.py [--seg-warmup PATH] [--cls-head PATH] [--joint PATH] [--config PATH] [--output-dir DIR] [--batch-size N] [--device {cuda,cpu}]` - Statistical comparison of all 3 training phases with significance tests (outputs to results/phase_comparison.json)

## 🧪 Testing & Validation

- `python scripts/evaluation/testing/test_multitask_e2e.py [--verbose]` - End-to-end integration test of full multi-task pipeline (30-60 sec, validates data→training→inference)
- `python scripts/evaluation/testing/test_backend_startup.py` - Validates API backend imports and model loading (5-10 sec, checks FastAPI server readiness)
- `python scripts/evaluation/testing/test_brain_crop.py` - Tests brain region cropping preprocessing transform (10-20 sec, validates BrainRegionCrop functionality)

## 🎬 Demo Applications

- `python scripts/demo/run_multitask_demo.py` - **RECOMMENDED** Launches unified multi-task demo with backend + frontend (FastAPI + Streamlit, opens browser to http://localhost:8501)
- `python scripts/demo/run_demo.py [--backend-port PORT] [--frontend-port PORT] [--no-check]` - Legacy Phase 6 demo with separate classification/segmentation models (backend + frontend)
- `python scripts/demo/run_demo_backend.py [--port PORT] [--host HOST] [--reload] [--no-check]` - Launches FastAPI backend only (default http://localhost:8000, API docs at /docs)
- `python scripts/demo/run_demo_frontend.py [--port PORT] [--backend-url URL] [--no-check]` - Launches Streamlit frontend only (default http://localhost:8501, requires backend running)

## 🔧 Debug & Diagnostics

- `python scripts/debug/debug_multitask_data.py` - Debugs multi-task dataset loading, validates batch collation, mixed source handling (BraTS + Kaggle), mask availability, and data shapes

---

## 📝 Quick Reference by Workflow

### Complete Training Pipeline
```bash
# 1. Download data
python scripts/data/collection/download_brats_data.py
python scripts/data/collection/download_kaggle_data.py

# 2. Preprocess and split
python scripts/data/preprocessing/preprocess_all_brats.py --num-patients 100
python scripts/data/splitting/split_brats_data.py
python scripts/data/splitting/split_kaggle_data.py

# 3. Train (3 stages)
python scripts/training/multitask/train_multitask_seg_warmup.py
python scripts/training/multitask/train_multitask_cls_head.py --encoder-init checkpoints/multitask_seg_warmup/best_model.pth
python scripts/training/multitask/train_multitask_joint.py --init-from checkpoints/multitask_cls_head/best_model.pth

# 4. Evaluate
python scripts/evaluation/multitask/evaluate_multitask.py
python scripts/evaluation/multitask/generate_multitask_gradcam.py --num-samples 50

# 5. Test
python scripts/evaluation/testing/test_multitask_e2e.py

# 6. Demo
python scripts/demo/run_multitask_demo.py
```

### Quick Test (Pre-trained Models)
```bash
python scripts/demo/run_multitask_demo.py
```

### Debug Data Issues
```bash
python scripts/debug/debug_multitask_data.py
python scripts/evaluation/testing/test_brain_crop.py
```

---

## 📋 Script Categories Summary

| Category | Scripts | Purpose |
|----------|---------|---------|
| **Data Collection** | 2 | Download BraTS and Kaggle datasets |
| **Data Preprocessing** | 2 | Convert 3D→2D, normalize, export examples |
| **Data Splitting** | 2 | Patient-level train/val/test splits |
| **Training** | 3 | Multi-task 3-stage training pipeline |
| **Training Utils** | 1 | Config generation |
| **Evaluation** | 3 | Metrics, Grad-CAM, phase comparison |
| **Testing** | 3 | E2E tests, backend validation, preprocessing tests |
| **Demo** | 4 | Interactive web applications |
| **Debug** | 1 | Dataset debugging and validation |
| **TOTAL** | **21** | Complete ML pipeline |

---

## ⚠️ Important Notes

1. **Always run from project root**: All scripts expect to be run from the project root directory
2. **Path depth**: Scripts use `.parent.parent.parent` for 3-level deep paths, `.parent.parent` for 2-level deep
3. **Dependencies**: Ensure `pip install -r requirements.txt` is run first
4. **CUDA**: Training scripts will use GPU if available, CPU otherwise
5. **Checkpoints**: Expected in `checkpoints/multitask_{seg_warmup,cls_head,joint}/best_model.pth`
6. **Data**: Expected in `data/raw/` and `data/processed/` directories
7. **Configs**: YAML configs in `configs/` directory (multitask_*.yaml)

---

**Last Updated**: December 7, 2025  
**Total Scripts**: 21  
**Project**: SliceWise MRI Brain Tumor Detection
