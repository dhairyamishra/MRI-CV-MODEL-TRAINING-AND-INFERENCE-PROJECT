# SliceWise — PROJECT_PLAN.md

High-level phases for the SliceWise project. Check items off as you go.

---

## Phase 0 — Project Scaffolding & Environment

- [x] **Create repo + base structure**
  - [x] Initialize Git repo (`slicewise/`).
  - [x] Create folders:
    - [x] `src/data/`
    - [x] `src/models/`
    - [x] `src/training/`
    - [x] `src/eval/`
    - [x] `src/inference/`
    - [x] `app/backend/` (FastAPI)
    - [x] `app/frontend/` (Streamlit/Gradio/React)
    - [x] `configs/`
    - [x] `jupyter_notebooks/` 
    - [x] `assets/`
    - [x] `documentation/` 
  - [x] Add base files:
    - [x] `pyproject.toml` or `setup.cfg`/`setup.py`
    - [x] `requirements.txt` (or `environment.yml`)
    - [x] `README.md`
    - [x] `LICENSE`
    - [x] `.gitignore`
    - [x] `.pre-commit-config.yaml` (optional)

- [x] **Set up Python + dependencies**
  - [x] Choose Python version (e.g., 3.10/3.11).
  - [x] Add core dependencies:
    - [x] `torch`, `torchvision`
    - [x] `monai`
    - [x] `numpy`, `scipy`, `pandas`
    - [x] `scikit-image`
    - [x] `pydicom`, `nibabel`
    - [x] `matplotlib`, `seaborn`
    - [x] `wandb` or `mlflow`
    - [x] `tqdm`, `omegaconf` or `pyyaml`
    - [x] `fastapi`, `uvicorn`
    - [x] `streamlit` or `gradio` (or React stack)
    - [x] `albumentations` (optional)
  - [x] Add dev dependencies:
    - [x] `pytest`, `pytest-cov`
    - [x] `black`, `isort`
    - [x] `flake8` or `ruff`

- [x] **Basic CI / sanity checks**
  - [x] Add GitHub Actions (or similar) workflow:
    - [x] Install dependencies (CPU-only).
    - [x] Run unit tests on small stubs.
  - [x] Add a "smoke test" script:
    - [x] Load fake slice, pass through tiny U-Net, save fake mask image.

- [x] **Compute + storage setup**
  - [x] Request NYU HPC access (A100/T4/L4).
  - [x] Create config files:
    - [x] `configs/hpc.yaml` (paths, batch sizes, num_workers, etc.)
    - [x] `configs/local.yaml`
  - [x] Decide and document dataset locations:
    - [x] e.g., `/scratch/$USER/datasets/brats2020`
    - [x] e.g., `/scratch/$USER/datasets/kaggle_brain_mri`

---

## Phase 1 — Data Acquisition & 2D Preprocessing

- [x] **Download + organize datasets**
  - [ ] BraTS 2020–2021:
    - [x] Document how to request/download (TCIA/Kaggle).
    - [x] Create `DATA_README.md` explaining access and licenses.
  - [x] Kaggle Brain MRI (yes/no):
    - [x] Download dataset.
    - [x] Store under `data/raw/kaggle_brain_mri/`.

- [x] **Define unified data layout**
  - [x] Decide on processed structure, e.g.:
    - [x] `data/processed/brats2d/{split}/{patient_id}_{slice_idx}.npz`
    - [x] `data/processed/kaggle/{split}/{id}.npz`
  - [x] Ensure `.npz` contains `image`, `mask` (if any), and metadata.

- [ ] **Implement BraTS 3D → 2D slice extraction**
  - [ ] Create `src/data/preprocess_brats_2d.py`:
    - [ ] Load NIfTI volumes with `nibabel`.
    - [ ] Select modalities (start: FLAIR only).
    - [ ] Normalize intensities (per-volume z-score or [0,1] with clipping).
    - [ ] Ensure correct alignment between images and masks.
    - [ ] Filter out trivially empty slices if needed.
    - [ ] Save slices as `.npz` with:
      - [ ] `image`: `(C,H,W)`
      - [ ] `mask`: `(1,H,W)`
      - [ ] metadata: `patient_id`, `slice_idx`, spacing, etc.

- [ ] **Implement patient-level train/val/test split**
  - [ ] `src/data/split_patients.py`:
    - [ ] Read list of patient IDs.
    - [ ] Randomly assign to train/val/test (e.g., 70/15/15) with fixed seed.
    - [ ] Save:
      - [ ] `splits/train_patients.csv`
      - [ ] `splits/val_patients.csv`
      - [ ] `splits/test_patients.csv`
  - [ ] Use these splits when generating processed slices.

- [x] **Implement PyTorch dataset classes**
  - [ ] `src/data/brats2d_dataset.py`:
    - [ ] Implement `BraTS2DSliceDataset` returning `image`, `mask`, IDs.
  - [x] `src/data/kaggle_mri_dataset.py`:
    - [x] Implement `KaggleBrainMRIDataset` returning `image`, `label`, ID.

- [x] **Define augmentations / transforms**
  - [x] Create `src/data/transforms.py`:
    - [x] Train transforms:
      - [x] Random rotations/flips.
      - [x] Intensity shifts/scaling.
      - [x] Optional elastic deformations.
    - [x] Val/test transforms:
      - [x] Resize/center-crop to target size (e.g., 256×256 or 320×320).
    - [x] Ensure masks use nearest-neighbor interpolation.

- [ ] **Sanity checks on preprocessed data**
  - [ ] Notebook `jupyter_notebooks/01_visualize_brats_slices.ipynb`:
    - [ ] Randomly sample slices.
    - [ ] Plot image + mask overlay.
    - [ ] Verify orientation, alignment, intensities.

---

## Phase 2 — Classification MVP (Kaggle Yes/No)

- [ ] **Implement classifier model**
  - [ ] `src/models/classifier.py`:
    - [ ] Wrap EfficientNet-B0 or ConvNeXt-Tiny.
    - [ ] Adapt for single-channel input.
    - [ ] Output 2-class logits.

- [ ] **Training loop for classification**
  - [ ] `src/training/train_cls.py`:
    - [ ] Load config `configs/config_cls.yaml`.
    - [ ] Build train/val DataLoaders.
    - [ ] Use `CrossEntropyLoss`, Adam/AdamW, optional scheduler.
    - [ ] Log loss and ROC-AUC to W&B/MLflow.
    - [ ] Save checkpoints `checkpoints/cls/` with best val metric.
    - [ ] (Optional) early stopping.

- [ ] **Basic evaluation + Grad-CAM**
  - [ ] `src/eval/eval_cls.py`:
    - [ ] Compute accuracy, ROC-AUC, PR-AUC.
    - [ ] Save ROC/PR plots.
  - [ ] `src/eval/grad_cam.py`:
    - [ ] Implement Grad-CAM for final conv layer.
    - [ ] Generate Grad-CAM overlays for sample slices.
    - [ ] Save visuals to `assets/grad_cam_examples/`.

- [ ] **Demo wiring for classification**
  - [ ] Minimal FastAPI endpoint `/classify_slice`:
    - [ ] Accept uploaded image.
    - [ ] Preprocess, run classifier, return probabilities.
  - [ ] Simple Streamlit/Gradio page:
    - [ ] Upload slice.
    - [ ] Display original + Grad-CAM overlay + probability.

---

## Phase 3 — Baseline 2D Segmentation Pipeline (U-Net)

- [ ] **Implement U-Net architecture**
  - [ ] `src/models/unet2d.py`:
    - [ ] Implement configurable 2D U-Net, or wrap MONAI’s UNet.
    - [ ] Parameters: `in_channels`, `out_channels`, `base_filters`, `depth`.

- [ ] **Implement loss functions**
  - [ ] `src/training/losses.py`:
    - [ ] Dice loss.
    - [ ] BCE with logits.
    - [ ] Combined Dice + BCE.
    - [ ] Tversky loss (configurable α, β).
  - [ ] Make loss type selectable via YAML config.

- [ ] **Segmentation training script**
  - [ ] `src/training/train_seg2d.py`:
    - [ ] Config `configs/seg2d_baseline.yaml`:
      - [ ] Model params, loss, optimizer, lr, batch size, epochs, augmentations.
    - [ ] Train on `BraTS2DSliceDataset` (train/val).
    - [ ] Log train/val loss, Dice, IoU to W&B/MLflow.
    - [ ] Save best checkpoint to `checkpoints/seg/`.
    - [ ] Periodically log example predictions with overlays.

- [ ] **Segmentation inference utility**
  - [ ] `src/inference/infer_seg2d.py`:
    - [ ] Implement `predict_slice(image)`:
      - [ ] Returns probability map and binary mask.
    - [ ] Optional batch/stack inference function.

- [ ] **Post-processing functions**
  - [ ] `src/inference/postprocess.py`:
    - [ ] Thresholding (configurable; Otsu fallback option).
    - [ ] Connected components:
      - [ ] Remove tiny blobs (min area).
      - [ ] Fill small holes.
      - [ ] Optionally keep largest component.
    - [ ] Return cleaned mask.

- [ ] **Visualization + quick evaluation**
  - [ ] `src/eval/eval_seg2d.py`:
    - [ ] Iterate over validation set.
    - [ ] Compute Dice and IoU per slice.
    - [ ] Save summary metrics.
    - [ ] Save example overlays: image + GT mask + predicted mask.

---

## Phase 4 — Calibration & Uncertainty

- [ ] **Classifier calibration (temperature scaling)**
  - [ ] `src/eval/calibration.py`:
    - [ ] Given logits + labels (val):
      - [ ] Optimize scalar temperature `T` by NLL.
      - [ ] Compute ECE and Brier score before/after scaling.
      - [ ] Produce reliability diagrams.
  - [ ] Update inference to apply calibrated `T` by default.

- [ ] **Segmentation confidence & uncertainty**
  - [ ] Extend `infer_seg2d.py`:
    - [ ] Always preserve full probability map.
  - [ ] Implement MC Dropout or TTA:
    - [ ] Enable dropout at test time.
    - [ ] Run N stochastic passes per slice.
    - [ ] Compute:
      - [ ] Mean probability map.
      - [ ] Pixel-wise variance map (uncertainty).
  - [ ] Save and visualize uncertainty maps in notebooks.

---

## Phase 5 — Ablations & Evaluation Suite

- [ ] **Metrics implementation**
  - [ ] `src/eval/metrics.py`:
    - [ ] Dice and IoU.
    - [ ] Boundary F-measure (thin contour bands).
    - [ ] Classification metrics:
      - [ ] Accuracy, ROC-AUC, PR-AUC.
      - [ ] Sensitivity, specificity at tuned thresholds.
    - [ ] Calibration metrics:
      - [ ] ECE, Brier score.

- [ ] **Patient-level aggregation**
  - [ ] `src/eval/patient_level_eval.py`:
    - [ ] Group slices by `patient_id`.
    - [ ] Define patient-level decision:
      - [ ] Tumor present if any slice exceeds prob + min area thresholds.
    - [ ] If slice thickness available:
      - [ ] Approximate tumor volume = Σ(area × thickness).
    - [ ] Compute patient-level sensitivity, specificity.
    - [ ] Save per-patient CSV summaries.

- [ ] **Ablation runner**
  - [ ] `src/eval/run_ablations.py`:
    - [ ] Define configs varying:
      - [ ] Input modalities (FLAIR vs multi-modal).
      - [ ] Loss function (Dice+BCE vs Tversky).
      - [ ] Augmentation strength (light vs strong).
      - [ ] Post-processing on vs off.
    - [ ] For each config:
      - [ ] Train or fine-tune model (possibly on subset).
      - [ ] Evaluate and record metrics.
    - [ ] Notebook `jupyter_notebooks/ablation_summary.ipynb`:
      - [ ] Aggregate results into tables and plots.

- [ ] **Efficiency / latency profiling**
  - [ ] `src/eval/profile_inference.py`:
    - [ ] Benchmark N slices on target GPU.
    - [ ] Report p50/p95 latency and peak GPU memory.
    - [ ] Compare different input resolutions.
    - [ ] Save results to CSV.

---

## Phase 6 — Demo Application (API + UI)

- [ ] **FastAPI backend**
  - [ ] `app/backend/main.py`:
    - [ ] Startup:
      - [ ] Load calibrated classifier weights.
      - [ ] Load best segmentation model weights.
      - [ ] Load config for thresholds and post-processing.
    - [ ] Endpoints:
      - [ ] `POST /classify_slice`:
        - [ ] Accept image upload.
        - [ ] Preprocess, run classifier, return probabilities + class.
      - [ ] `POST /segment_slice`:
        - [ ] Accept single slice (or small stack).
        - [ ] Run segmentation + post-processing.
        - [ ] Return probability map summary, mask (PNG or RLE), metrics.
      - [ ] `POST /segment_stack`:
        - [ ] Accept zipped stack / NIfTI / DICOM series.
        - [ ] Convert to slices, process, aggregate patient-level decision + volume.

- [ ] **Demo UI (Streamlit/Gradio)**
  - [ ] `app/frontend/app.py`:
    - [ ] Sidebar controls:
      - [ ] Model selection (baseline vs best).
      - [ ] Threshold sliders (probability, min area).
      - [ ] Toggle uncertainty overlay.
    - [ ] Main view:
      - [ ] File uploader for:
        - [ ] Single slice.
        - [ ] Short stack.
      - [ ] Slice viewer (slider to scroll slices in stack).
      - [ ] Display:
        - [ ] Original image.
        - [ ] Mask overlay (adjustable opacity).
        - [ ] Probability/uncertainty map overlays.
        - [ ] Classifier outputs + Grad-CAM overlays.
      - [ ] Download buttons:
        - [ ] Overlay images (`mask.png`, etc.).
        - [ ] Per-slice CSV of predictions.
    - [ ] Add clear disclaimers:
      - [ ] “Research use only — not a medical device.”
      - [ ] “Upload de-identified images only.”

- [ ] **Hook frontend to backend**
  - [ ] Decide architecture:
    - [ ] Directly import inference code, or
    - [ ] Call FastAPI endpoints via HTTP.
  - [ ] Implement integration.
  - [ ] Handle loading states and errors gracefully.

- [ ] **UX polish**
  - [ ] Add loading spinners during inference.
  - [ ] Add tooltips for overlays and metrics.
  - [ ] Add simple color legend (e.g., red = high tumor probability).

---

## Phase 7 — Documentation, LaTeX Write-up & Artifacts

- [ ] **Repo documentation**
  - [ ] Update `README.md` with:
    - [ ] Project overview and goals.
    - [ ] Environment setup and installation.
    - [ ] Data acquisition instructions (BraTS, Kaggle).
    - [ ] Steps for:
      - [ ] Preprocessing.
      - [ ] Training.
      - [ ] Evaluation.
      - [ ] Running the demo app.
    - [ ] Ethical and non-diagnostic usage disclaimer.

- [x] **In-code documentation**
  - [ ] Add docstrings for:
    - [ ] Datasets.
    - [ ] Models.
    - [ ] Training scripts.
    - [ ] Evaluation scripts.
    - [ ] Inference and API endpoints.
  - [x] Create `src/README.md` describing module layout.

- [ ] **LaTeX report**
  - [ ] Create `report/` directory:
    - [ ] `report.tex` (extend proposal into full paper).
    - [ ] `bibliography.bib`.
  - [ ] Generate and include:
    - [ ] Example segmentation overlays (success & failure cases).
    - [ ] Grad-CAM examples.
    - [ ] Calibration plots (reliability diagram, etc.).
    - [ ] Ablation tables and plots.
    - [ ] Efficiency/latency tables.
  - [ ] Sections to finalize:
    - [ ] Methods (data, models, training details).
    - [ ] Experiments (evaluation setup and metrics).
    - [ ] Results and ablations.
    - [ ] Limitations, ethics, and future work.
    - [ ] Team member contributions.

- [ ] **Ethics & privacy documentation**
  - [ ] Create `ETHICS.md`:
    - [ ] Data de-identification expectations.
    - [ ] Non-medical-use disclaimer.
    - [ ] Dataset licenses and citation requirements.
    - [ ] Safety and limitations statement.

---

## Phase 8 — Packaging, Reproducibility & Final Polish

- [ ] **Reproducible configs**
  - [ ] Ensure all experiments use saved YAML configs under `configs/experiments/`.
  - [ ] For best model:
    - [ ] Document W&B/MLflow run IDs.
    - [ ] Create `scripts/reproduce_best_model.sh` that:
      - [ ] Runs preprocessing (if needed).
      - [ ] Trains best model from scratch.
      - [ ] Evaluates on test set.
      - [ ] Saves final checkpoint in `models/best_seg2d.pth`.

- [ ] **Model versioning & model card**
  - [ ] Save:
    - [ ] Baseline U-Net.
    - [ ] Best UNet++ or DeepLabv3+ model (if implemented).
    - [ ] Calibrated classifier.
  - [ ] Create `MODEL_CARD.md`:
    - [ ] Training data description.
    - [ ] Evaluation metrics.
    - [ ] Intended use and limitations.
    - [ ] Ethical note and disclaimers.

- [ ] **Lightweight tests**
  - [ ] Create small synthetic dataset:
    - [ ] 16×16 images with simple geometric “tumors”.
  - [ ] Add tests:
    - [ ] Dataset loading test.
    - [ ] Classifier and segmenter forward-pass tests.
    - [ ] Inference pipeline test (slice → mask).
    - [ ] Post-processing test (min area, hole filling).
  - [ ] Ensure tests run in CI.

- [ ] **App deployment plan**
  - [ ] Simple local run scripts:
    - [ ] `scripts/run_backend.sh`
    - [ ] `scripts/run_frontend.sh` or `make run-app`
  - [ ] (Optional) Dockerization:
    - [ ] Dockerfile for CPU-only demo.
    - [ ] Instructions for running container with GPU on HPC/cloud.

- [ ] **Final cleanup**
  - [ ] Remove dead code, unused notebooks, and temporary artifacts.
  - [ ] Ensure `.gitignore` excludes:
    - [ ] Checkpoints.
    - [ ] Raw datasets.
    - [ ] Large log files.
  - [ ] Tag repo release (e.g., `v1.0-slicewise-demo`).
  - [ ] Verify:
    - [ ] End-to-end run from README is correct.
    - [ ] Demo runs smoothly on a fresh environment.

---
