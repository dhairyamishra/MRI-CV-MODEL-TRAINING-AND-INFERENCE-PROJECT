📋 Complete Implementation Task List
I've created a 28-task checklist organized by phase. Here's the detailed breakdown:

PHASE 0: Data Standardization (4 tasks)
Goal: Make BraTS and Kaggle datasets produce identical tensor formats

[x] Define input specification - Critical decision point
Choose: 1×256×256 (single modality - FLAIR) or 4×256×256 (multi-modal)
Recommendation: Start with 1×256×256 for simplicity
[x] Create Kaggle preprocessing pipeline
New file: src/data/preprocess_kaggle_unified.py
Resize all images to 256×256
Apply z-score normalization (matching BraTS)
Save as .npz files with metadata
[x] Implement MultiSourceDataset
New file: src/data/multi_source_dataset.py
Returns dict: {"image": tensor, "mask": tensor/None, "cls": int, "source": str}
Handles both BraTS (with masks) and Kaggle (mask=None)
[x] Create unified dataloader factory
Function: create_multi_task_dataloaders()
Support Option A (alternating batches) and Option B (mixed batches)
PHASE 1: Model Refactoring (5 tasks)
Goal: Split U-Net into modular encoder + decoder + classification head

[] Refactor UNet2D → UNetEncoder
Modify: src/models/unet2d.py
Extract encoder path (downsampling blocks)
Return feature list: [x1, x2, x3, x4, bottleneck]
[] Create UNetDecoder
New class in src/models/unet2d.py
Takes feature list from encoder
Upsampling path with skip connections
Outputs segmentation logits
[] Implement ClassificationHead
New file: src/models/classification_head.py
Global average pooling on bottleneck features
Small MLP (e.g., 512 → 256 → 2)
Dropout for regularization
[] Create MultiTaskModel
New file: src/models/multi_task_model.py
Wraps encoder + decoder + cls_head
Forward pass with do_seg and do_cls flags
Returns dict: {"seg": logits, "cls": logits}
[] Add Grad-CAM support
Hook into encoder's bottleneck layer
Ensure compatibility with existing src/eval/grad_cam.py
PHASE 2: Training Strategy (8 tasks)
Goal: Staged curriculum learning

Stage 2.1: Segmentation Warm-up
[] Create segmentation-only training script
New file: scripts/train_multitask_stage1_seg.py
Use existing train_seg2d.py as template
Train only on BraTS with Dice+BCE loss
[] Run baseline training
Train for 20-30 epochs
Target: Dice > 0.70
Save checkpoint: checkpoints/multitask/stage1_seg_warmup.pth
Stage 2.2: Classification Head Training
[] Create classification head training script
New file: scripts/train_multitask_stage2_cls.py
Load stage 1 checkpoint, freeze encoder
Train cls_head on BraTS + Kaggle
[] Run classification training
Use weighted BCE or Focal loss
Train for 10-15 epochs
Save checkpoint: checkpoints/multitask/stage2_cls_head.pth
Stage 2.3: Joint Fine-tuning
[] Implement alternating batch training
New file: scripts/train_multitask_stage3_joint.py
Alternate between BraTS batches (both tasks) and Kaggle batches (cls only)
[] Implement combined loss function
New file: src/training/multi_task_losses.py
L_total = L_seg + λ_cls * L_cls for BraTS
L_total = λ_cls * L_cls for Kaggle
Start with λ_cls = 1.0
[] Add differential learning rates
Encoder: 1e-4
Decoder + cls_head: 3e-4
Use PyTorch parameter groups
[] Run joint fine-tuning
Load stage 2 checkpoint
Unfreeze encoder (or just last block)
Train for 15-20 epochs
Save final checkpoint: checkpoints/multitask/stage3_joint_final.pth
PHASE 3: Evaluation (4 tasks)
Goal: Validate that multi-task learning helps

[] Create segmentation comparison script
New file: scripts/evaluate_multitask_segmentation.py
Compare baseline (stage 1) vs multi-task (stage 3)
Metrics: Dice, IoU, boundary F-measure
[] Create classification evaluation script
New file: scripts/evaluate_multitask_classification.py
Test on BraTS (derived labels) and Kaggle test set
Metrics: ROC-AUC, PR-AUC, sensitivity, specificity
[] Generate Grad-CAM visualizations
Modify existing scripts/generate_gradcam.py
Support multi-task model
Visualize both BraTS and Kaggle samples
[] Create comparison report
New file: documentation/MULTITASK_EVALUATION_REPORT.md
Tables comparing all metrics
Visualizations (Grad-CAM overlays, confusion matrices)
Ablation study results
PHASE 4: Integration (4 tasks)
Goal: Deploy multi-task model in production app

[] Create unified inference wrapper
New file: src/inference/multi_task_predictor.py
Single forward pass returns both tumor_prob and mask
Handle preprocessing (z-score normalization)
[] Update FastAPI backend
Modify: app/backend/main_v2.py
Replace separate models with multi-task model
New endpoint: /predict_multitask (returns both outputs)
[] Update Streamlit UI
Modify: app/frontend/app_v2.py
Conditional display logic:
If tumor_prob < 0.3: Show "No tumor detected"
If tumor_prob ≥ 0.3: Show segmentation + Grad-CAM
[] Create model config file
New file: configs/multi_task_model_config.yaml
Store: modality, input_size, normalization params, thresholds
PHASE 5: Stretch Goals (3 tasks - Optional)
Goal: Advanced features for future iterations

🔮 Multi-modal support
4-channel encoder for BraTS (FLAIR, T1, T1ce, T2)
Modality projection layer for Kaggle (1→4 channels)
🔮 Domain adaptation
Style augmentation (blur, noise, contrast)
Reduce distribution shift between BraTS and Kaggle
🔮 Uncertainty estimation
Integrate existing MC-dropout from src/inference/uncertainty.py
Show uncertainty maps for both tasks
🗂️ New Files to Create
Data (3 files)
src/data/preprocess_kaggle_unified.py - Kaggle preprocessing
src/data/multi_source_dataset.py - Unified dataset class
src/data/dataloader_factory.py - Multi-task dataloader creation
Models (2 files)
src/models/classification_head.py - Classification head module
src/models/multi_task_model.py - Main multi-task wrapper
Training (2 files)
src/training/multi_task_losses.py - Combined loss functions
src/training/train_multitask.py - Unified training loop
Scripts (6 files)
scripts/train_multitask_stage1_seg.py - Stage 1 training
scripts/train_multitask_stage2_cls.py - Stage 2 training
scripts/train_multitask_stage3_joint.py - Stage 3 training
scripts/evaluate_multitask_segmentation.py - Seg evaluation
scripts/evaluate_multitask_classification.py - Cls evaluation
scripts/run_multitask_pipeline.py - End-to-end orchestrator
Inference (1 file)
src/inference/multi_task_predictor.py - Unified inference
Config (1 file)
configs/config_multitask.yaml - Multi-task training config
Documentation (1 file)
documentation/MULTITASK_EVALUATION_REPORT.md - Results report
🚀 Recommended Execution Order
Start with Phase 0 (Data standardization) - Foundation for everything
Then Phase 1 (Model refactoring) - Can test each component independently
Phase 2.1 (Seg warm-up) - Quick validation that refactored model works
Phase 2.2 (Cls head) - Verify classification on both datasets
Phase 2.3 (Joint training) - The main experiment
Phase 3 (Evaluation) - Measure success
Phase 4 (Integration) - Deploy to production
Phase 5 (Optional) - Future improvements
📊 Expected Outcomes
Success Metrics:

Segmentation: Dice ≥ 0.70 (maintain baseline performance)
Classification on BraTS: ROC-AUC ≥ 0.95
Classification on Kaggle: ROC-AUC ≥ 0.90
Grad-CAM: Activations align with tumor regions
Key Benefits:

Single model inference (faster, more efficient)
Better classification from segmentation features
Unified codebase (easier maintenance)
Richer interpretability (seg masks + Grad-CAM)