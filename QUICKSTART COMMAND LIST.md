Here’s the execution-guide README you asked for — focused only on **“what to run, in what order”**, from blank machine → running app.
(Commands are based on your consolidated main README and merged docs. )

---

````markdown
# SliceWise – End-to-End Execution Guide 

**Last Updated**: December 19, 2025

This document is a **step-by-step runbook** to go from:

> **Fresh clone → data ready → configs → training → evaluation → tests → running the app**

For explanations and architecture details, see the main `README.md`. This guide is just **“do this, see that”**.

## Recent Enhancements (December 19, 2025)

- **Brain Masking for Grad-CAM**: Eliminates red background artifacts, focuses on brain tissue only
- **Robust Kaggle Preprocessing**: 97.1% quality pass rate with advanced morphological operations
- **Skull Boundary Detection**: Automatic mask inversion detection for correct segmentation
- **Segmentation Fixes**: Z-score normalization and proper visualization in UI
- **API Improvements**: Fixed cv2 import, consistent preprocessing across all endpoints

---

## 0. One-Time System Setup

**Goal:** Have Python, Node, and basic tools ready.

1. **Install prerequisites (if not already installed)**  
   - Python **3.11 or 3.12**  
   - (Optional) CUDA drivers + GPU  
   - Node.js + npm (for PM2 demo)  
   - Git  
   - Kaggle account + API token (`kaggle.json`)

2. **Check versions (optional sanity)**
   ```bash
   python --version
   pip --version
   node --version
   npm --version
   git --version
````

---

## 1. Clone Repo & Create Environment

**Goal:** Have the project and a clean Python environment.

1. **Clone the repo**

   ```bash
   git clone <repository-url>
   cd MRI-CV-MODEL-TRAINING-AND-INFERENCE-PROJECT
   ```

2. **Create and activate virtualenv**

   ```bash
   python -m venv venv

   # Linux/macOS
   source venv/bin/activate

   # Windows (PowerShell)
   # .\venv\Scripts\Activate.ps1
   ```

3. **Install package (dev mode)**

   ```bash
   pip install -e ".[dev]"
   ```

4. **(Optional) Install PM2 globally for demo**

   ```bash
   npm install -g pm2
   ```

5. **(Optional) Quick setup verification**

   ```bash
   python scripts/verify_setup.py
   ```

   **Expected result:** Script finishes without errors and prints basic environment + dependency info.

---

## 2. Configure Kaggle & Data Folders

**Goal:** Make sure datasets can be downloaded + stored correctly.

1. **Configure Kaggle API**

   ```bash
   # Put kaggle.json in the right place

   # Linux/macOS:
   mkdir -p ~/.kaggle
   cp /path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json

   # Windows (PowerShell):
   # mkdir $env:USERPROFILE\.kaggle
   # copy .\kaggle.json $env:USERPROFILE\.kaggle\
   ```

2. **Check data directory exists**

   ```bash
   ls data
   ```

   **Expected result:** `data/` exists (empty or with `raw/` / `processed/`). If not, create:

   ```bash
   mkdir -p data/raw data/processed
   ```

---

## 3. Download Datasets

**Goal:** Get Kaggle + BraTS data onto disk.

> ⚠ You can **skip BraTS** if you only want a small classification demo. For full multi-task pipeline, run **both**.

### 3.1 Kaggle Brain MRI Dataset (Classification)

```bash
python scripts/data/collection/download_kaggle_data.py
```

**Expected result:**

* Creates `data/raw/kaggle_brain_mri/`
* Contains `yes/` and `no/` image folders.

---

### 3.2 BraTS Dataset (Segmentation & Multi-Task)

```bash
python scripts/data/collection/download_brats_data.py --version 2020
# or
python scripts/data/collection/download_brats_data.py --version 2021
```

**Expected result:**

* Creates `data/raw/brats2020/` or `data/raw/brats2021/`
* Contains the official BraTS training data folders.

---

## 4. Preprocess Data (3D → 2D Slices, Normalization, etc.)

**Goal:** Convert raw datasets into unified, model-ready 2D format.

### 4.1 BraTS 3D → 2D Preprocessing

```bash
python scripts/data/preprocessing/preprocess_all_brats.py
```

**Expected result:**

* Creates `data/processed/brats2d/` with `.npz` 2D slices
* Writes metadata (patient IDs, quality metrics) under `data/processed/...`

> Optional: restrict to specific patients (debugging only)
>
> ```bash
> python scripts/data/preprocessing/preprocess_all_brats.py --patient-ids 001 002 003
> ```

---

### 4.2 Kaggle Preprocessing (Classification)

If not already processed by the unified pipeline:

```bash
python src/data/preprocess_kaggle.py
```

**Expected result:**

* Creates `data/processed/kaggle/` with normalized `.npz` files
* Applies **robust brain masking** (97.1% quality pass rate)
* Uses z-score normalization (mean=0, std=1) for consistent model input
* Logs summary statistics (counts, shape, normalization, quality metrics) to console.

**Key Features:**
- Advanced morphological operations (closing, flood fill, convex hull)
- Automatic quality checks (3-95% area, 75% max border)
- Eliminates skull rings and background artifacts
- Foreground-only normalization (background=0)

---

## 5. Create Train/Val/Test Splits

**Goal:** Leakage-safe splits at **patient level** (not per slice).

### 5.1 BraTS Splits

```bash
python scripts/data/splitting/split_brats_data.py
```

**Expected result:**

* Creates split definition files, e.g. under
  `data/processed/brats2d/splits/{train,val,test}.csv`
* Each row corresponds to a **patient**, not a single slice.

---

### 5.2 Kaggle Splits

```bash
python scripts/data/splitting/split_kaggle_data.py
```

**Expected result:**

* Creates train / val / test splits, stratified by class, e.g.
  `data/processed/kaggle/splits/{train,val,test}.csv`

---

## 6. Generate Training Configs (Hierarchical Config System)

**Goal:** Build all final YAML configs that training scripts will use.

```bash
python scripts/utils/merge_configs.py --all
```

**Expected result:**

* `configs/final/` is populated with something like:

  * `stage1_quick.yaml`, `stage2_quick.yaml`, `stage3_quick.yaml`
  * `stage1_baseline.yaml`, ... etc.
* No errors in console.

> (Optional) Test config generation:
>
> ```bash
> pytest tests/test_config_generation.py -v
> ```

---

## 7. Run the Training Pipeline

You have **two main options**:

---

### 7A. Single-Command Full Pipeline (Recommended)

**Goal:** Data checks → multi-stage training → evaluation → visualizations in one go.

```bash
# Quick, fast sanity run
python scripts/run_full_pipeline.py --mode full --training-mode quick

# Baseline (longer, better results)
python scripts/run_full_pipeline.py --mode full --training-mode baseline

# Production (full dataset, long run)
python scripts/run_full_pipeline.py --mode full --training-mode production
```

**Expected result (per mode):**

* Progress logs for:

  * Data checks
  * Stage 1, 2, 3 training
  * Evaluation + visualizations
* New checkpoints under `checkpoints/`
* New metrics & plots under `outputs/` or `visualizations/`

---

### 7B. Manual 3-Stage Multi-Task Training

**Goal:** Full control over each training stage.

1. **Stage 1 – Segmentation warm-up**

   ```bash
   python scripts/training/multitask/train_multitask_seg_warmup.py \
     --config configs/final/stage1_quick.yaml
   ```

   **Expected result:**

   * Segmentation-only model checkpoint in `checkpoints/multitask_stage1/`

2. **Stage 2 – Classification head**

   ```bash
   python scripts/training/multitask/train_multitask_cls_head.py \
     --config configs/final/stage2_quick.yaml
   ```

   **Expected result:**

   * Classification head trained, checkpoint in `checkpoints/multitask_stage2/`

3. **Stage 3 – Joint fine-tuning**

   ```bash
   python scripts/training/multitask/train_multitask_joint.py \
     --config configs/final/stage3_quick.yaml
   ```

   **Expected result:**

   * Final multi-task checkpoint (shared encoder + dual heads) in
     `checkpoints/multitask_joint/` (e.g. `best_model.pth`)

> Swap `*_quick.yaml` for `*_baseline.yaml` or `*_production.yaml` as needed.

---

## 8. Run Evaluation & Visualizations

**Goal:** Get metrics + nice visual outputs (Grad-CAM, segmentation overlays, etc.).

1. **Quantitative multi-task evaluation**

   ```bash
   python scripts/evaluation/multitask/eval_multitask_model.py \
     --checkpoint checkpoints/multitask_joint/best_model.pth
   ```

   **Expected result:**

   * Prints Dice, IoU, AUC, etc. to console
   * Writes metrics JSON/CSV into `outputs/` (e.g., `outputs/multitask_eval/`)

2. **Grad-CAM visualizations**

   ```bash
   python scripts/evaluation/multitask/generate_multitask_gradcam.py \
     --checkpoint checkpoints/multitask_joint/best_model.pth \
     --num-samples 50 \
     --output-dir visualizations/multitask_gradcam
   ```

   **Expected result:**

   * Image files under `visualizations/multitask_gradcam/` showing:

     * Input MRI
     * Classification Grad-CAM **with brain masking** (no background artifacts)
     * Segmentation mask / overlays with skull boundary detection
     * Error maps

   **Key Features:**
   - Brain masking applied before and after Grad-CAM generation
   - Eliminates red/hot activations in black background
   - Focuses attention on clinically relevant brain tissue
   - Improved overlay blending (cv2.addWeighted, 50% alpha)
   - Cleaner, more professional visualizations

3. **(Optional) Inference profiling**

   ```bash
   python scripts/evaluation/profile_inference.py
   ```

   **Expected result:**

   * Prints FPS / latency / memory usage stats.

---

## 9. Run Tests

**Goal:** Make sure everything works end-to-end and configs are valid.

1. **Smoke tests (env + core components)**

   ```bash
   pytest tests/test_smoke.py -v
   ```

2. **Config generation tests**

   ```bash
   pytest tests/test_config_generation.py -v
   ```

3. **End-to-end multi-task pipeline test**

   ```bash
   python scripts/evaluation/testing/test_multitask_e2e.py
   ```

4. **Backend startup / API tests**

   ```bash
   python scripts/evaluation/testing/test_backend_startup.py
   ```

**Expected result:**
All tests complete without failures, basic endpoints respond.

---

## 10. Run the Application (API + UI)

You again have **two options**: PM2 (recommended) or manual.

---

### 10A. PM2 Demo (Recommended)

**Goal:** Start backend + frontend with automatic restarts & logging.

```bash
# One-liner helper
python scripts/demo/run_demo_pm2.py
```

**Expected result:**

* PM2 starts at least two processes (backend + frontend)
* Logs are written under `logs/`
* Backend typically at `http://localhost:8000`
* Frontend typically at `http://localhost:8501`

**Useful PM2 commands:**

```bash
pm2 status      # Show running processes
pm2 logs        # Tail logs for all apps
pm2 monit       # Live resource view
pm2 stop all    # Stop all demo processes
pm2 delete all  # Stop + remove from PM2
```

---

### 10B. Manual Launch (Two Terminals)

**Terminal 1 – Backend**

```bash
# From project root, with venv active
python scripts/demo/run_demo_backend.py
# or directly:
# uvicorn app.backend.main_v2:app --host 0.0.0.0 --port 8000
```

**Terminal 2 – Frontend**

```bash
# From project root, with venv active
python scripts/demo/run_demo_frontend.py
# or directly:
# streamlit run app/frontend/app.py --server.port 8501
```

**Expected result:**

* Visit `http://localhost:8501`

  * Upload MRI images
  * Run classification, segmentation, and multi-task predictions
  * View **brain-masked Grad-CAMs** (no background artifacts)
  * View segmentation overlays with **skull boundary detection**
  * View patient-level summaries with volume estimation

**UI Features:**
- ✅ Grad-CAM with automatic brain masking
- ✅ Segmentation with z-score normalization (correct predictions)
- ✅ Skull boundary detection for Kaggle images
- ✅ Uncertainty estimation (MC Dropout + TTA)
- ✅ Professional overlay visualizations

---

## 11. Shut Down & Cleanup (Optional)

1. **Stop app**

   * PM2:

     ```bash
     pm2 stop all
     ```
   * Manual:

     * Ctrl+C in both terminal windows

2. **Deactivate virtualenv**

   ```bash
   deactivate
   ```

3. **(Optional) Remove large data/checkpoints for space**

   ```bash
   rm -rf data/raw/ data/processed/ checkpoints/ outputs/ visualizations/ logs/
   ```

---

## 12. Minimal “Happy Path” Checklist ✅

If you want a **quick sanity recipe**, this is the shortest path:

```bash
# 1) Clone + env
git clone <repository-url>
cd MRI-CV-MODEL-TRAINING-AND-INFERENCE-PROJECT
python -m venv venv
source venv/bin/activate         # or .\venv\Scripts\Activate.ps1 on Windows
pip install -e ".[dev]"
npm install -g pm2               # optional but recommended

# 2) Data
python scripts/data/collection/download_kaggle_data.py
python scripts/data/collection/download_brats_data.py --version 2020
python scripts/data/preprocessing/preprocess_all_brats.py
python scripts/data/splitting/split_brats_data.py
python scripts/data/splitting/split_kaggle_data.py

# 3) Configs
python scripts/utils/merge_configs.py --all

# 4) Train (quick mode)
python scripts/run_full_pipeline.py --mode full --training-mode quick

# 5) Tests (sanity)
pytest tests/test_smoke.py -v
pytest tests/test_config_generation.py -v

# 6) Demo
python scripts/demo/run_demo_pm2.py
# open http://localhost:8501
```

You now have the **end-to-end pipeline + app** running from a fresh clone.

---

## 13. Key Technical Enhancements (December 2025)

### Brain Masking for Grad-CAM
- **Problem**: Red/hot activations in black background (misleading)
- **Solution**: Apply brain mask before and after Grad-CAM generation
- **Result**: Clean visualizations focused on brain tissue only
- **Files**: `src/inference/multi_task_predictor.py`, `app/backend/services/multitask_service.py`

### Robust Kaggle Preprocessing
- **Problem**: Skull rings, black blobs, inconsistent masks
- **Solution**: Advanced morphological pipeline (closing, flood fill, convex hull)
- **Result**: 97.1% quality pass rate (238/245 images)
- **Files**: `src/data/brain_mask.py`, `src/data/preprocess_kaggle.py`

### Segmentation Preprocessing Fix
- **Problem**: Model showing "Not Detected" (0.0%) in UI
- **Solution**: Added z-score normalization to all segmentation endpoints
- **Result**: Model predictions now match evaluation results (96.5% probability)
- **Files**: `app/backend/main_v2.py`, `app/backend/services/segmentation_service.py`

### Skull Boundary Detection
- **Problem**: Inverted segmentation masks on Kaggle images
- **Solution**: Automatic detection and correction of mask inversion
- **Result**: Correct tumor predictions with proper boundary detection
- **Files**: `src/inference/multi_task_predictor.py`

---

**For detailed technical documentation, see:**
- `documentation/GRADCAM_BRAIN_MASKING_FIX.md`
- `documentation/ROBUST_BRAIN_MASKING_IMPLEMENTATION.md`
- `documentation/SKULL_BOUNDARY_DETECTION_FIX_SUMMARY.md`
- `BUGFIX_SEGMENTATION_ENDPOINT.md`
