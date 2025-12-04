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

- [x] **Define unified data layout**
  - [x] Decide on processed structure:
    - [x] `data/processed/brats2d/{split}/{patient_id}_{slice_idx}.npz`
    - [x] `data/processed/kaggle/{split}/{id}.npz`
  - [x] Ensure `.npz` contains `image`, `mask` (if any), and metadata.
  - [x] Created `src/data/preprocess_kaggle.py` for Kaggle preprocessing
  - [x] Preprocessed all 245 images to .npz format (256×256, normalized)

- [x] **Implement BraTS 3D → 2D slice extraction**
  - [x] Create `src/data/preprocess_brats_2d.py` (452 lines):
    - [x] Load NIfTI volumes with `nibabel`.
    - [x] Select modalities (FLAIR, T1, T1ce, T2).
    - [x] Normalize intensities (z-score, min-max, percentile methods).
    - [x] Ensure correct alignment between images and masks.
    - [x] Filter out trivially empty slices (configurable threshold).
    - [x] Save slices as `.npz` with:
      - [x] `image`: `(1,H,W)` - normalized float32
      - [x] `mask`: `(1,H,W)` - binary uint8
      - [x] metadata: `patient_id`, `slice_idx`, modality, tumor info, pixdim
    - [x] Tested with 10 patients (569 slices extracted)

- [x] **Implement patient-level train/val/test split**
  - [x] `src/data/split_kaggle.py`:
    - [x] Read list of image files.
    - [x] Randomly assign to train/val/test (70/15/15) with fixed seed.
    - [x] Save splits to directories:
      - [x] `data/processed/kaggle/train/` (171 files)
      - [x] `data/processed/kaggle/val/` (37 files)
      - [x] `data/processed/kaggle/test/` (37 files)
    - [x] Stratified splitting maintains class balance
  - [x] `src/data/split_brats.py` (245 lines):
    - [x] Patient-level splitting to avoid data leakage
    - [x] Configurable train/val/test ratios (default: 70/15/15)
    - [x] Random seed for reproducibility
    - [x] Tested: 7 train / 1 val / 2 test patients

- [x] **Implement PyTorch dataset classes**
  - [x] `src/data/brats2d_dataset.py` (234 lines):
    - [x] Implement `BraTS2DSliceDataset` returning `image`, `mask`, metadata
    - [x] Support for optional transforms
    - [x] `get_statistics()` method for dataset analysis
    - [x] `get_sample_metadata()` for individual samples
    - [x] `create_dataloaders()` helper function
    - [x] Tested: Successfully loads 569 slices from 10 patients
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
  - [x] `src/models/unet2d.py` (352 lines):
    - [x] Implement configurable 2D U-Net
    - [x] Parameters: `in_channels`, `out_channels`, `base_filters`, `depth`
    - [x] Encoder-decoder with skip connections
    - [x] Bilinear or transposed conv upsampling
    - [x] Binary and multi-class segmentation support
    - [x] 31.4M parameters (standard config: 64 base filters, depth 4)
    - [x] Tested: forward pass, gradient flow, multiple input sizes
    - [x] Factory function `create_unet()` for easy instantiation

- [x] **Implement loss functions**
  - [x] `src/training/losses.py` (396 lines):
    - [x] Dice loss (primary segmentation metric)
    - [x] BCE with logits (pixel-wise classification)
    - [x] Combined Dice + BCE (best of both worlds)
    - [x] Tversky loss (configurable α, β for FP/FN weighting)
    - [x] Focal loss (focuses on hard examples)
    - [x] Factory function `get_loss_function()` for easy selection
    - [x] All losses tested with backward pass
  - [x] Loss type selectable via YAML config

- [x] **Segmentation training script**
  - [x] `src/training/train_seg2d.py` (462 lines):
    - [x] Load config from `configs/seg2d_baseline.yaml`
    - [x] Train on `BraTS2DSliceDataset` (train/val)
    - [x] Calculate Dice and IoU metrics per batch
    - [x] Log train/val loss, Dice, IoU to W&B
    - [x] Save best checkpoint to `checkpoints/seg/`
    - [x] Mixed precision training (AMP)
    - [x] Gradient clipping
    - [x] Early stopping with configurable patience
    - [x] Learning rate scheduling (Cosine, Step, Plateau)
    - [x] Multiple optimizers (Adam, AdamW, SGD)
  - [x] `configs/seg2d_baseline.yaml` (143 lines):
    - [x] Model params (U-Net config)
    - [x] Loss function selection and parameters
    - [x] Optimizer and scheduler settings
    - [x] Training hyperparameters (batch size, epochs, AMP)
    - [x] Early stopping configuration
    - [x] Checkpoint management
    - [x] W&B logging settings
  - [x] `scripts/train_segmentation.py` (21 lines):
    - [x] Simple wrapper script for easy execution
  - [x] **Baseline model trained (10 patients): Train Dice 0.860, Val Dice 0.743**

- [x] **Segmentation inference utility**
  - [x] `src/inference/infer_seg2d.py` (329 lines):
    - [x] Implement `SegmentationPredictor` class
    - [x] `predict_slice(image)` - Returns probability map and binary mask
    - [x] `predict_batch()` - Batch inference
    - [x] `predict_dataloader()` - Full dataset inference
    - [x] Convenience functions for easy usage

- [x] **Post-processing functions**
  - [x] `src/inference/postprocess.py` (301 lines):
    - [x] Thresholding (fixed and Otsu methods)
    - [x] Connected components analysis
    - [x] Remove tiny blobs (min area filtering)
    - [x] Fill small holes
    - [x] Keep largest component option
    - [x] Morphological operations (open, close, dilate, erode)
    - [x] Complete pipeline with statistics

- [x] **Visualization + comprehensive evaluation**
  - [x] `src/eval/eval_seg2d.py` (378 lines):
    - [x] Iterate over validation/test set
    - [x] Compute Dice, IoU, Precision, Recall, F1, Specificity per slice
    - [x] Save summary metrics (mean, std, min, max, median)
    - [x] Save example overlays: Input | GT | Pred | TP/FP/FN
    - [x] Generate metrics distribution plots
    - [x] Export detailed JSON results
  - [x] **Evaluation results: Dice 0.708 ± 0.182 (45 val slices)**

- [x] **Helper scripts**
  - [x] `scripts/evaluate_segmentation.py` (199 lines):
    - [x] Easy evaluation with sensible defaults
    - [x] Automatic split detection (train/val/test)
    - [x] Error checking and helpful messages
  - [x] `scripts/preprocess_all_brats.py` (304 lines):
    - [x] Automated preprocessing pipeline
    - [x] Progress tracking and time estimation
    - [x] Confirmation prompts

- [x] **Documentation**
  - [x] `PHASE3_COMPLETE.md` (494 lines):
    - [x] Complete component overview
    - [x] Training results and metrics
    - [x] Usage guide with examples
  - [x] `PHASE3_QUICKSTART.md` (221 lines):
    - [x] 5-minute quick start
    - [x] Common commands
    - [x] Troubleshooting guide

---

## Phase 4 — Calibration & Uncertainty 

- [x] **Classifier calibration (temperature scaling)**
  - [x] `src/eval/calibration.py` (349 lines):
    - [x] Given logits + labels (val):
      - [x] Optimize scalar temperature `T` by NLL.
      - [x] Compute ECE and Brier score before/after scaling.
      - [x] Produce reliability diagrams.
  - [x] `scripts/calibrate_classifier.py` (201 lines):
    - [x] Helper script for easy calibration
    - [x] Generates before/after reliability diagrams
    - [x] Saves calibrated temperature scaler
  - [x] **Tested on real classifier: 68.2% ECE reduction (0.0461 → 0.0147)**

- [x] **Segmentation confidence & uncertainty**
  - [x] `src/inference/uncertainty.py` (372 lines):
    - [x] MC Dropout implementation
      - [x] Enable dropout at test time
      - [x] Run N stochastic passes per slice
      - [x] Compute mean probability map and pixel-wise variance
    - [x] Test-Time Augmentation (TTA)
      - [x] 6 augmentations (original, hflip, vflip, rot90, rot180, rot270)
      - [x] Average predictions across augmentations
    - [x] Ensemble predictor (MC Dropout + TTA)
      - [x] Separates epistemic vs aleatoric uncertainty
  - [x] Fully tested with synthetic data

---

## Phase 5 — Ablations & Evaluation Suite 

- [x] **Metrics implementation**
  - [x] `src/eval/metrics.py` (404 lines):
    - [x] Segmentation metrics:
      - [x] Dice coefficient and IoU
      - [x] Boundary F-measure (thin contour bands)
      - [x] Pixel accuracy
      - [x] Sensitivity and specificity
    - [x] Classification metrics:
      - [x] Accuracy, ROC-AUC, PR-AUC
      - [x] Sensitivity, specificity at tuned thresholds
      - [x] Optimal threshold finding (F1, Youden's J, Accuracy)
    - [x] Comprehensive evaluation function
    - [x] Fully tested with synthetic data

- [x] **Patient-level aggregation**
  - [x] `src/eval/patient_level_eval.py` (334 lines):
    - [x] Group slices by `patient_id`
    - [x] Define patient-level decision:
      - [x] Tumor present if any slice exceeds prob + min area thresholds
    - [x] Volume estimation:
      - [x] Approximate tumor volume = Σ(area × thickness)
    - [x] Compute patient-level sensitivity, specificity
    - [x] Save per-patient CSV summaries
    - [x] Tested with 3 synthetic patients

- [x] **Ablation runner**
  - [x] `src/eval/run_ablations.py`:
    - [x] Define configs varying:
      - [x] Input modalities (FLAIR vs multi-modal)
      - [x] Loss function (Dice+BCE vs Tversky)
      - [x] Augmentation strength (light vs strong)
      - [x] Post-processing on vs off
    - [x] For each config:
      - [x] Train or fine-tune model (possibly on subset)
      - [x] Evaluate and record metrics
    - [x] Notebook `jupyter_notebooks/ablation_summary.ipynb`:
      - [x] Aggregate results into tables and plots

- [x] **Efficiency / latency profiling**
  - [x] `src/eval/profile_inference.py` (289 lines):
    - [x] Benchmark N slices on target GPU
    - [x] Report p50/p95/p99 latency and peak GPU memory
    - [x] Compare different input resolutions (128×128, 256×256, 512×512)
    - [x] Compare different batch sizes (1, 4, 8)
    - [x] Measure throughput (images/second)
    - [x] Save results to CSV
    - [x] Tested: 256×256 @ 2,551 imgs/sec, 0.4ms latency

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
