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
  - [x] BraTS 2020–2021:
    - [x] Document how to request/download (TCIA/Kaggle).
    - [x] Create `DATA_README.md` explaining access and licenses.
    - [x] Downloaded BraTS 2020 dataset (988 patients, ~80 GB)
    - [x] Created `scripts/download_brats_data.py` for automated download
    - [x] Created `scripts/verify_brats_structure.py` for structure verification
    - [x] Created `scripts/diagnose_brats_data.py` for diagnostics
    - [x] Created comprehensive `documentation/BRATS_DATASET_GUIDE.md` (359 lines)
  - [x] Kaggle Brain MRI (yes/no):
    - [x] Download dataset.
    - [x] Store under `data/raw/kaggle_brain_mri/`.
    - [x] Created `scripts/download_kaggle_data.py` for automated download
    - [x] Verified download (245 images: 154 tumor, 91 no tumor)

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

- [x] **Implement patient-level train/val/test split**
  - [x] `src/data/split_kaggle.py`:
    - [x] Read list of image files.
    - [x] Randomly assign to train/val/test (70/15/15) with fixed seed.
    - [x] Save splits to directories:
      - [x] `data/processed/kaggle/train/` (171 files)
      - [x] `data/processed/kaggle/val/` (37 files)
      - [x] `data/processed/kaggle/test/` (37 files)
    - [x] Stratified splitting maintains class balance
  - [x] Use these splits when generating processed slices.

- [x] **Implement PyTorch dataset classes**
  - [ ] `src/data/brats2d_dataset.py`:
    - [ ] Implement `BraTS2DSliceDataset` returning `image`, `mask`, IDs.
  - [x] `src/data/kaggle_mri_dataset.py`:
    - [x] Implement `KaggleBrainMRIDataset` returning `image`, `label`, ID.
    - [x] Added `get_class_distribution()` method
    - [x] Added `get_sample_metadata()` method
    - [x] Created `create_dataloaders()` helper function

- [x] **Define augmentations / transforms**
  - [x] Create `src/data/transforms.py`:
    - [x] Train transforms:
      - [x] Random rotations/flips (90°, 180°, 270°).
      - [x] Intensity shifts/scaling.
      - [x] Optional Gaussian noise (elastic deformations alternative).
    - [x] Val/test transforms:
      - [x] No augmentation (images already preprocessed to 256×256).
    - [x] Ensure masks use nearest-neighbor interpolation.
    - [x] Created three presets: standard, strong, light augmentation

- [ ] **Sanity checks on preprocessed data**
  - [ ] Notebook `jupyter_notebooks/01_visualize_brats_slices.ipynb`:
    - [ ] Randomly sample slices.
    - [ ] Plot image + mask overlay.
    - [ ] Verify orientation, alignment, intensities.

---

## Phase 2 — Classification MVP (Kaggle Yes/No) 

- [x] **Implement classifier model**
  - [x] `src/models/classifier.py`:
    - [x] Wrap EfficientNet-B0 and ConvNeXt-Tiny.
    - [x] Adapt for single-channel input with weight averaging.
    - [x] Output 2-class logits.
    - [x] Built-in Grad-CAM support.

- [x] **Training loop for classification**
  - [x] `src/training/train_cls.py`:
    - [x] Load config `configs/config_cls.yaml`.
    - [x] Build train/val DataLoaders with augmentation.
    - [x] CrossEntropyLoss + Focal Loss options.
    - [x] Adam/AdamW/SGD optimizers.
    - [x] Cosine/Step/Plateau schedulers.
    - [x] Log to W&B with comprehensive metrics.
    - [x] Save checkpoints to `checkpoints/cls/`.
    - [x] Early stopping with configurable patience.
    - [x] Mixed precision training (AMP).
    - [x] Gradient clipping.

- [x] **Comprehensive evaluation + Grad-CAM**
  - [x] `src/eval/eval_cls.py`:
    - [x] Compute accuracy, ROC-AUC, PR-AUC, F1, precision, recall.
    - [x] Save ROC/PR/confusion matrix plots.
    - [x] Export predictions CSV and metrics JSON.
  - [x] `src/eval/grad_cam.py`:
    - [x] Full Grad-CAM implementation.
    - [x] Generate overlays with OpenCV.
    - [x] Batch visualization support.
    - [x] Save to `assets/grad_cam_examples/`.

- [x] **Production-ready API + UI**
  - [x] FastAPI backend (`app/backend/main.py`):
    - [x] `/healthz` - Health check
    - [x] `/model/info` - Model information
    - [x] `/classify_slice` - Single image classification
    - [x] `/classify_batch` - Batch classification
    - [x] `/classify_with_gradcam` - Classification + Grad-CAM
  - [x] Streamlit frontend (`app/frontend/app.py`):
    - [x] Beautiful UI with medical disclaimers.
    - [x] Drag-and-drop upload.
    - [x] Real-time predictions.
    - [x] Grad-CAM visualization.
    - [x] Probability charts.
    - [x] Interpretation guidance.

- [x] **Helper scripts**
  - [x] `scripts/train_classifier.py`
  - [x] `scripts/evaluate_classifier.py`
  - [x] `scripts/generate_gradcam.py`
  - [x] `scripts/run_backend.py`
  - [x] `scripts/run_frontend.py`

---

## Phase 3 — Baseline 2D Segmentation Pipeline (U-Net) 

- [x] **Implement U-Net architecture**
  - [x] `src/models/unet2d.py`:
    - [x] Implement configurable 2D U-Net (352 lines)
    - [x] Parameters: `in_channels`, `out_channels`, `base_filters`, `depth`
    - [x] Encoder-decoder with skip connections
    - [x] Bilinear or transposed conv upsampling
    - [x] Binary and multi-class segmentation support
    - [x] 31.4M parameters (standard config: 64 base filters, depth 4)
    - [x] Tested: forward pass, gradient flow, multiple input sizes
    - [x] Factory function `create_unet()` for easy instantiation

- [x] **Implement loss functions**
  - [x] `src/training/losses.py` (396 lines):
    - [x] Dice loss
    - [x] BCE with logits
    - [x] Combined Dice + BCE
    - [x] Tversky loss (configurable α, β for FP/FN weighting)
    - [x] Focal loss (focuses on hard examples)
    - [x] Factory function `get_loss_function()` for easy selection
    - [x] All losses tested with backward pass
  - [ ] Make loss type selectable via YAML config

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
