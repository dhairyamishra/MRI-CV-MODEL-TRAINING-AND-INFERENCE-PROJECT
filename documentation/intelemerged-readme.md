Yes, I understand and I’m ready.

You’ve set the **root-level README as the source of truth**, and we’ll **merge all of the detailed docs** into a **single, de-duplicated, logically ordered main README** while still keeping the per-module docs (`SRC_*.md`, `SCRIPTS_*.md`, `APP_*.md`, etc.) as deeper references.

Below is a **proposed new `README.md`** that:

* Uses your current root README as the backbone.
* Pulls in/aligns content from:

  * `SRC_ARCHITECTURE_AND_IMPLEMENTATION.md`
  * `DATA_ARCHITECTURE_AND_MANAGEMENT.md`
  * `SCRIPTS_ARCHITECTURE_AND_USAGE.md`
  * `CONFIG_SYSTEM_ARCHITECTURE.md`
  * `APP_ARCHITECTURE_AND_FUNCTIONALITY.md`
  * `VISUALIZATIONS_GUIDE.md`
  * `WANDB_INTEGRATION_GUIDE.md`
  * `LOGGING_SYSTEM_GUIDE.md`
  * `TESTING_SUITE_AND_VALIDATION.md`
  * `ARCHIVES_REPOSITORY_GUIDE.md`

I’ve **removed cross-doc duplication** and instead link to the detailed docs from the main README.

---

````markdown
# SliceWise - MRI Brain Tumor Detection & Segmentation

> **A production-ready deep learning pipeline for brain tumor classification and segmentation from MRI images with a unified multi-task architecture**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)

---

## 🎯 Project Status

| Phase                 | Status      | Description                                      |
|-----------------------|------------|--------------------------------------------------|
| **Phase 0**           | ✅ Complete | Project scaffolding & environment                |
| **Phase 1**           | ✅ Complete | Data acquisition & 2D preprocessing              |
| **Phase 2**           | ✅ Complete | Classification MVP (EfficientNet + API)          |
| **Phase 3**           | ✅ Complete | U-Net segmentation pipeline                      |
| **Phase 4**           | ✅ Complete | Calibration & uncertainty estimation             |
| **Phase 5**           | ✅ Complete | Metrics & patient-level evaluation               |
| **Phase 6**           | ✅ Complete | Demo application (FastAPI backend + Streamlit UI)|
| **Multi-Task**        | ✅ Complete | Unified architecture (classification + segmentation) |
| **Frontend Refactor** | ✅ Complete | Modular UI architecture (significant code cut)   |
| **Phase 7**           | 🚧 In Progress | Documentation & LaTeX write-up                |
| **Phase 8**           | 📋 Planned | Packaging & deployment                           |

**Progress:** ~90% complete • ~20,000+ LOC • 70+ files • 25+ organized scripts

---

## 🌟 High-Level Overview

SliceWise is a **medical imaging** project focused on **MRI brain tumor detection**, implementing:

1. **🔍 Binary Classification**  
   - Tumor vs. no-tumor classification on 2D slices  
   - Shared encoder with segmentation  
   - Grad-CAM explainability & calibrated confidences  

2. **🎯 Tumor Segmentation**  
   - 2D U-Net with shared encoder  
   - Multiple loss options (Dice, BCE, Focal, Tversky)  
   - Uncertainty estimation (MC Dropout, Test-Time Augmentation)  

3. **🚀 Multi-Task Architecture**  
   - Single model, shared encoder + dual heads  
   - Single forward pass for classification + segmentation  
   - Parameter-efficient (~9–10% fewer params vs separate models)  
   - ~40% faster inference via conditional segmentation  

4. **📊 Patient-Level Analysis**  
   - Patient-level metrics & tumor volume estimation  
   - Clinical-style reports and visualizations  

5. **🧪 Production-Ready Stack**  
   - Modular **src/** architecture (data, models, training, eval, inference)  
   - Hierarchical **YAML config** system  
   - **FastAPI** backend, **Streamlit** frontend  
   - **W&B** experiment tracking, rich logging & visualizations  
   - Full testing suite (+ PM2-based demo runner)

For deep-dive docs, see `documentation/` (e.g. `SRC_ARCHITECTURE_AND_IMPLEMENTATION.md`, `APP_ARCHITECTURE_AND_FUNCTIONALITY.md`, etc.).

---

## 📁 Repository Structure (Condensed)

```bash
MRI-CV-MODEL-TRAINING-AND-INFERENCE-PROJECT/
├── app/                     # FastAPI backend + Streamlit frontend
│   ├── backend/             # REST API, services, models, settings
│   └── frontend/            # Modular Streamlit UI
├── src/                     # Core ML implementation (Python package)
│   ├── data/                # Datasets, transforms, dataloaders
│   ├── models/              # U-Net, multi-task model, classifier head
│   ├── training/            # Training loops, schedulers, losses
│   ├── eval/                # Metrics, evaluation utilities
│   └── inference/           # Deployed inference helpers & predictors
├── scripts/                 # Orchestrated scripts (data, train, eval, demo)
│   ├── data/                # Download, preprocess, split, export examples
│   ├── training/            # Multitask 3-stage training scripts
│   ├── evaluation/          # Metrics, Grad-CAM, segmentation analysis
│   ├── demo/                # PM2 demo runner & helpers
│   └── utils/               # Config merger, helpers, etc.
├── configs/
│   ├── base/                # Common training, model, augmentation defaults
│   ├── stages/              # Stage 1–3 configs (seg warmup, cls, joint)
│   ├── modes/               # quick / baseline / production modes
│   ├── final/               # Auto-generated merged configs (gitignored)
│   └── pm2-ecosystem/       # PM2 process ecosystem config
├── tests/                   # Unit + integration + config tests
├── documentation/           # This project’s detailed architecture docs
├── data/                    # (gitignored) raw/ + processed/ folders
├── checkpoints/             # (gitignored) trained model weights
├── outputs/                 # (gitignored) eval + visualization outputs
├── logs/                    # (gitignored) backend/frontend/PM2 logs
├── wandb/                   # (gitignored) W&B run directories
└── README.md                # You are here
````

For full tree & explanations, see:

* `SRC_ARCHITECTURE_AND_IMPLEMENTATION.md`
* `SCRIPTS_ARCHITECTURE_AND_USAGE.md`
* `CONFIG_SYSTEM_ARCHITECTURE.md`
* `APP_ARCHITECTURE_AND_FUNCTIONALITY.md`
* `DATA_ARCHITECTURE_AND_MANAGEMENT.md`

---

## 🚀 Quick Start

### 1. Prerequisites

* Python **3.11** or **3.12**
* CUDA-capable GPU (optional, but recommended)
* 8GB+ RAM
* **Kaggle API** configured (`~/.kaggle/kaggle.json`) for Kaggle dataset
* **Node.js + npm** (for PM2-based demo, especially on Windows)

### 2. Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd MRI-CV-MODEL-TRAINING-AND-INFERENCE-PROJECT

# 2. Create and activate virtualenv
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 3. Install project in editable mode (with dev extras)
pip install -e ".[dev]"

# 4. (Optional but recommended) Install PM2 globally
npm install -g pm2

# 5. Generate hierarchical training configs
python scripts/utils/merge_configs.py --all
```

### 3. Sanity Check (Smoke Tests)

```bash
# Quick environment validation (PyTorch, basic UNet, src package, etc.)
pytest tests/test_smoke.py -v
```

Expected: all tests in `test_smoke.py` pass.

---

## 📊 Datasets & Data Pipeline

### 1. Datasets Used

* **BraTS 2020 / 2021**:

  * 3D multimodal MRI volumes (FLAIR, T1, T1ce, T2) with segmentation masks
  * Used for **segmentation** and indirectly for classification through slice labels

* **Kaggle Brain MRI Dataset (Navoneel)**:

  * 2D JPEG slices, tumor / no-tumor labels
  * Used for **classification** and mixed into the **multi-task** pipeline

See `DATA_ARCHITECTURE_AND_MANAGEMENT.md` for full dataset statistics, schema, and rationale.

### 2. Data Download

```bash
# BraTS (2020 or 2021)
python scripts/data/collection/download_brats_data.py --version 2020
# or
python scripts/data/collection/download_brats_data.py --version 2021

# Kaggle classification dataset
python scripts/data/collection/download_kaggle_data.py
```

Output (examples):

```bash
data/raw/
├── brats2020/
│   └── BraTS2020_TrainingData/...
└── kaggle_brain_mri/
    ├── yes/
    └── no/
```

### 3. Preprocessing: 3D → 2D Multi-Task Slices

```bash
# Main BraTS 3D → 2D conversion + quality filtering
python scripts/data/preprocessing/preprocess_all_brats.py

# (Optional) Process specific patients only
python scripts/data/preprocessing/preprocess_all_brats.py --patient-ids 001 002 003
```

This pipeline performs:

* Brain extraction
* Registration to FLAIR
* Slice extraction & filtering
* Z-score normalization
* Metadata creation (patient IDs, quality metrics)

Kaggle preprocessing & unified multi-task datasets are also handled in the same system – see `DATA_ARCHITECTURE_AND_MANAGEMENT.md` + `SCRIPTS_ARCHITECTURE_AND_USAGE.md`.

### 4. Patient-Level Splits (Leakage-Safe)

```bash
# BraTS splits (70/15/15, patient-level)
python scripts/data/splitting/split_brats_data.py

# Kaggle splits (stratified class ratios)
python scripts/data/splitting/split_kaggle_data.py
```

Splits operate at **patient level**, not random slices, to avoid data leakage.

---

## 🧠 Core Architecture (src/)

The `src/` package contains the model, data, training, evaluation, and inference code in a clean 5-module architecture:

* `src/data/` – dataset classes, transforms, dataloader factories
* `src/models/` – U-Net encoder/decoder, multi-task model, classification head
* `src/training/` – training loops, schedulers, optimizer logic, losses
* `src/eval/` – metrics, evaluation, calibration
* `src/inference/` – predictors used by the backend & CLI tools

Key concept: **shared encoder** + **dual heads (segmentation + classification)** for multi-task learning. See `SRC_ARCHITECTURE_AND_IMPLEMENTATION.md` for diagrams and parameter breakdown.

---

## ⚙️ Hierarchical Configuration System (configs/)

All experiments and deployments are controlled via YAML configs:

* **Base configs**: global defaults (training, architectures, augmentation, platform, logging)
* **Stage configs**: Stage 1 (seg warmup), Stage 2 (cls head), Stage 3 (joint)
* **Mode configs**: quick / baseline / production
* **Final configs**: auto-generated combinations per (stage, mode)

### Generate All Configs

```bash
python scripts/utils/merge_configs.py --all
```

### Example Usage

```bash
# Quick mode full pipeline
python scripts/run_full_pipeline.py --mode full --training-mode quick

# Production training
python scripts/run_full_pipeline.py --mode full --training-mode production
```

See `CONFIG_SYSTEM_ARCHITECTURE.md` for:

* Deep-merge algorithm
* Reference expansion (`architecture: multitask_medium`, `augmentation.preset: moderate`)
* Expected performance per mode (quick vs baseline vs production)
* PM2 integration & config testing guidance

---

## 🏋️ Training & Evaluation Workflows

### 1. End-to-End Pipeline (Recommended Entry Point)

```bash
# End-to-end pipeline: data → train (stages 1–3) → eval → visualizations
python scripts/run_full_pipeline.py --mode full --training-mode quick
```

Modes:

* `--training-mode quick` – sanity runs, short experiments
* `--training-mode baseline` – standard research
* `--training-mode production` – full, publication-quality runs

### 2. 3-Stage Multi-Task Training (Manual)

If you want fine-grained control:

```bash
# Stage 1 – segmentation warmup
python scripts/training/multitask/train_multitask_seg_warmup.py \
  --config configs/final/stage1_quick.yaml

# Stage 2 – classification head
python scripts/training/multitask/train_multitask_cls_head.py \
  --config configs/final/stage2_quick.yaml

# Stage 3 – joint fine-tuning
python scripts/training/multitask/train_multitask_joint.py \
  --config configs/final/stage3_quick.yaml
```

### 3. Evaluation & Visualizations

```bash
# Multi-task model evaluation
python scripts/evaluation/multitask/eval_multitask_model.py \
  --checkpoint checkpoints/multitask_joint/best_model.pth

# Grad-CAM generation
python scripts/evaluation/multitask/generate_multitask_gradcam.py \
  --checkpoint checkpoints/multitask_joint/best_model.pth \
  --num-samples 50 \
  --output-dir visualizations/multitask_gradcam
```

Outputs:

* Quantitative metrics (Dice, IoU, AUC, etc.)
* Grad-CAM heatmaps (classification + segmentation)
* 4-panel segmentation comparison (input / GT / prediction / error overlay)

See `VISUALIZATIONS_GUIDE.md` for formats, interpretation guidelines, and clinical checklists.

---

## 🧪 Testing & Validation

The project includes a **multi-layered testing strategy**:

* **Smoke tests**: environment, PyTorch, CNN sanity checks
* **Config tests**: 27 tests validating hierarchical config generation
* **End-to-end tests**: pipeline from data → model → inference → API
* **Backend tests**: FastAPI startup and health endpoints

### Common Commands

```bash
# Environment smoke tests
pytest tests/test_smoke.py -v

# Config system validation (27 tests)
pytest tests/test_config_generation.py -v

# E2E multi-task pipeline test
python scripts/evaluation/testing/test_multitask_e2e.py

# Backend startup test (healthz, endpoints)
python scripts/evaluation/testing/test_backend_startup.py
```

See `TESTING_SUITE_AND_VALIDATION.md` for details and rationales.

---

## 🔧 Application Layer (FastAPI + Streamlit)

The deployed application lives inside `app/`:

* **Backend** (`app/backend/`) – FastAPI service:

  * Lifespan-based startup/shutdown
  * Routers for health, classification, segmentation, multitask, patient-level analysis
  * Services for model loading, image preprocessing, inference orchestration
  * Pydantic models for request/response schemas

* **Frontend** (`app/frontend/`) – Streamlit UI:

  * Upload MRI slices
  * Run classification / segmentation / multi-task predictions
  * Visualize Grad-CAM, segmentations, error overlays
  * Present metrics & patient-level summaries

For architecture diagrams and endpoint lists, see `APP_ARCHITECTURE_AND_FUNCTIONALITY.md`.

---

## 📦 Demo & PM2-Based Deployment

For a local demo with robust process management (especially on Windows):

```bash
# Start full demo (backend + frontend under PM2)
python scripts/demo/run_demo_pm2.py

# Inspect processes
pm2 status

# View logs
pm2 logs

# Stop everything
pm2 stop all
```

PM2 plus the config ecosystem provides:

* Auto-restarts on crash
* Centralized logging in `logs/`
* Memory-based restart thresholds
* Cross-platform behavior (Win / Linux / macOS)

See `CONFIG_SYSTEM_ARCHITECTURE.md` + `SCRIPTS_ARCHITECTURE_AND_USAGE.md` for PM2 details.

---

## 📈 Experiment Tracking (Weights & Biases)

SliceWise integrates **Weights & Biases (W&B)** for:

* Real-time metric logging during training
* Hyperparameter management & comparison
* Artifact storage (checkpoints, plots, Grad-CAM, etc.)
* System metrics (GPU utilization, throughput)

Basic flow:

1. Enable W&B in config (`configs/modes/*.yaml`)
2. Run training scripts (they auto-initialize W&B if enabled)
3. Inspect runs via W&B web UI (metrics, comparisons, charts)

See `WANDB_INTEGRATION_GUIDE.md` for setup, run structures, and dashboard hints.

---

## 📜 Logging System

Logs are written to the `logs/` directory, typically via PM2:

* `backend-error.log`, `backend-out.log`, `backend-combined.log`
* `frontend-error.log`, `frontend-out.log`, `frontend-combined.log`

You can:

* Tail logs with `pm2 logs`
* Archive logs for long-term analysis
* Parse logs for performance metrics, error rates, and usage patterns

See `LOGGING_SYSTEM_GUIDE.md` for examples of shell one-liners for log analytics.

---

## 🏛️ Archives & Legacy Code

All deprecated code related to the **pre-multi-task architecture** is stored in:

```bash
archives/
├── app/          # Monolithic legacy backend/frontend
├── configs/      # Old config structure
└── scripts/      # Phase 1–5 legacy scripts
```

**Important:**

* This code is **not maintained** and may not run with the current environment.
* Use only for historical comparison and research.
* For all new experiments and production work, use `src/`, `scripts/`, `configs/`, and `app/`.

See `ARCHIVES_REPOSITORY_GUIDE.md` for context on the evolution from Phases 1–5 → Phase 6 multi-task architecture.

---

## 🤝 Contributing / Extending

If you want to extend SliceWise, common entry points are:

* New datasets → `src/data/` + preprocessing scripts
* New models → `src/models/` + config hooks in `configs/base/model_architectures.yaml`
* New losses/metrics → `src/training/losses.py`, `src/eval/metrics.py`
* New visualizations → `src/inference/visualization_utils.py` + `VISUALIZATIONS_GUIDE.md`
* New API endpoints → `app/backend/routers/` + `services/`

General workflow:

1. Add/modify code in `src/` or `app/`
2. Update configs in `configs/base/` & generate final configs
3. Add tests in `tests/` where relevant
4. Run smoke + config + relevant E2E tests
5. Update documentation under `documentation/` if needed

---

## 📄 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

```

---

If you’d like, next step I can:

- Trim / expand specific sections (e.g., make Quick Start even shorter for recruiters, move some technical stuff into separate docs), or  
- Generate **an additional “Developer-Only README”** focused just on internal workflows (training, evaluation, deployment) while keeping this one as the polished public-facing overview.
```
