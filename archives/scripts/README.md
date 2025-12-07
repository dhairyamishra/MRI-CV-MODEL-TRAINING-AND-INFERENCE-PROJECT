# Archived Legacy Scripts

**Archive Date:** 2025-12-07 05:35:45  
**Reason:** Transition to Phase 6 Multi-Task Architecture  
**Total Scripts Archived:** 13

---

## 📦 What's Archived Here

These scripts were used in Phase 1-5 of the SliceWise project when we had **separate models** for classification and segmentation. They have been replaced by the **unified multi-task architecture** in Phase 6.

### Phase 1-5 (Individual Models)
- Separate classifier and segmentation models
- Individual training pipelines
- Separate evaluation and calibration

### Phase 6 (Multi-Task Architecture) - Current
- Unified encoder shared between tasks
- 3-stage training pipeline (seg warmup → cls head → joint fine-tuning)
- 40% faster inference, 9.4% fewer parameters
- 91.3% accuracy, 97.1% sensitivity

---

## 📁 Archive Structure

### phase1-5_training/ (6 scripts)
Legacy training scripts for individual models:
- `train_classifier.py`
- `train_segmentation.py`
- `train_classifier_brats.py`
- `train_brats_e2e.py`
- `train_production.py`
- `train_controller.py`

### phase1-5_evaluation/ (3 scripts)
Legacy evaluation scripts for individual models:
- `evaluate_classifier.py`
- `evaluate_segmentation.py`
- `generate_gradcam.py`

### phase1-5_calibration/ (2 scripts)
Legacy calibration scripts:
- `calibrate_classifier.py`
- `view_calibration_results.py`

### phase1-5_demo/ (2 scripts)
Legacy demo and testing scripts:
- `run_demo_with_production_models.py`
- `test_full_e2e_phase1_to_phase6.py`

---

## 🚀 Current System (Phase 6)

For current multi-task system, use:

### Training
```bash
python scripts/train_multitask_seg_warmup.py
python scripts/train_multitask_cls_head.py
python scripts/train_multitask_joint.py
```

### Testing
```bash
python scripts/test_multitask_e2e.py
```

### Demo
```bash
python scripts/run_multitask_demo.py
```

---

## 📚 Documentation

- **Current System:** `documentation/MULTITASK_LEARNING_COMPLETE.md`
- **Scripts Analysis:** `scripts/SCRIPTS_ANALYSIS.md`
- **Consolidated Docs:** `documentation/CONSOLIDATED_DOCUMENTATION.md`

---

**Note:** These archived scripts are kept for historical reference and research purposes. They are not maintained and may not work with the current codebase.
