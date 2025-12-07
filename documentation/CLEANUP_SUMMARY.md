# SliceWise MRI Brain Tumor Detection - REPO CLEANUP GUIDE
## Repository Analysis & Cleanup Summary

**Date:** December 6, 2025  
**Project Status:** Phase 6 Complete (Multi-Task Architecture)  
**Current Size:** ~13,500 lines, 50+ files  
**Target Size After Cleanup:** ~8,000 lines (40% reduction)

---

## 📊 Executive Summary

The SliceWise project has evolved significantly from individual classification/segmentation models (Phases 1-5) to a unified **multi-task architecture** (Phase 6). This evolution enables:

- 🚀 **40% faster inference** (single forward pass)
- 💾 **9.4% fewer parameters** (2.0M vs 2.2M)  
- 🎯 **Better performance** (91.3% accuracy, 97.1% sensitivity)
- 🏗️ **Simplified maintenance** (unified vs separate pipelines)

**Cleanup Opportunity:** ~40% of current code is legacy from earlier research phases and can be safely removed while preserving all production functionality.

---

## 🎯 Current Active System (Phase 6)

### Core Components (Keep These)
- **Backend API:** `app/backend/main_v2.py` (12 endpoints)
- **Frontend UI:** `app/frontend/app_v2.py` (5 tabs)
- **Multi-Task Model:** `src/models/multi_task_model.py` + components
- **Training Pipeline:** Multi-task scripts (3-stage training)
- **Demo Launchers:** `scripts/run_demo*.py`

### Production Features
- ✅ Multi-task inference (classification + conditional segmentation)
- ✅ Grad-CAM explainability
- ✅ Uncertainty estimation (MC Dropout + TTA)
- ✅ Patient-level analysis & volume estimation
- ✅ Batch processing with CSV/JSON export
- ✅ Production API with error handling

---

## 🗑️ Legacy Components to Remove

### Phase 1-5 Research Code (~5,000+ lines)
- **Old Backend:** `app/backend/main.py` (Phase 2)
- **Old Frontend:** `app/frontend/app.py` (Phase 2) 
- **Separate Models:** `src/models/classifier.py`, `src/models/unet2d.py`
- **Legacy Training:** `scripts/train_classifier.py`, `scripts/train_segmentation.py`
- **Old Configs:** Most `.yaml` files in `configs/` (17 files)
- **Research Scripts:** 20+ evaluation/utility scripts
- **Legacy Docs:** Phase 1-5 documentation (~2,000+ lines)

### Data Pipeline Legacy
- **Old Preprocessing:** Separate Kaggle/BraTS preprocessing (superseded by unified)
- **Legacy Datasets:** Old dataset classes and transforms

---

## ❓ Components Needing Verification

### Potentially Keep (Verify Usage)
- **Individual Predictors:** `src/inference/predict.py`, `src/inference/infer_seg2d.py`
- **Calibration Scripts:** `scripts/calibrate_classifier.py`
- **Legacy Evals:** Some evaluation scripts if still referenced
- **Model Configs:** `src/models/model_config.py` (check dependencies)

### Archive for Reference
- **Research Notebooks:** `jupyter_notebooks/`
- **Legacy Documentation:** Phase 1-5 docs (zip for archive)

---

## 📋 Cleanup Phases

### Phase 1: Safe Removals (Immediate)
- Remove old backend/frontend (`main.py`, `app.py`)
- Remove separate classifier/unet models
- Remove legacy training scripts
- Remove unused config files

### Phase 2: Verification Required  
- Check dependencies on individual predictors
- Verify model_config.py usage
- Test all endpoints still work

### Phase 3: Documentation Cleanup
- Consolidate docs to Phase 6 focus
- Archive research documentation
- Update README and guides

---

## 🎯 Expected Benefits

- **Maintenance:** 60% fewer files to maintain
- **Complexity:** Single unified architecture vs 3 separate pipelines
- **Performance:** Faster, more efficient multi-task model
- **Clarity:** Focus on production system, not research history

---

## ⚠️ Important Notes

- **Backup First:** Create git branch before cleanup
- **Test Thoroughly:** Run full E2E tests after each phase
- **Keep Archives:** Zip legacy code for potential future reference
- **Update Docs:** README and guides need Phase 6 focus

---

## 📁 File Structure After Cleanup

```
SliceWise/
├── app/                          # Production API + UI
│   ├── backend/main_v2.py        # ✅ Keep
│   └── frontend/app_v2.py        # ✅ Keep
├── src/
│   ├── models/                   # Multi-task components only
│   │   ├── multi_task_model.py   # ✅ Keep
│   │   ├── unet_encoder.py       # ✅ Keep  
│   │   ├── unet_decoder.py       # ✅ Keep
│   │   └── classification_head.py# ✅ Keep
│   ├── inference/
│   │   ├── multi_task_predictor.py # ✅ Keep
│   │   └── uncertainty.py        # ✅ Keep
│   └── training/multi_task_losses.py # ✅ Keep
├── scripts/                      # Multi-task training + demo only
│   ├── train_multitask_*.py      # ✅ Keep (3 files)
│   ├── run_demo*.py              # ✅ Keep (3 files)
│   └── run_multitask_demo.py     # ✅ Keep
├── configs/                      # Multi-task configs only
│   └── multitask_*.yaml          # ✅ Keep (3 files)
├── documentation/                # Phase 6 + essentials
│   ├── PHASE6_*.md               # ✅ Keep
│   ├── MULTITASK_*.md            # ✅ Keep
│   └── CLEANUP_SUMMARY.md        # ✅ New
└── tests/                        # Current tests only
    └── test_*.py                 # ✅ Keep (verify all work)
```

---

## 🚀 Next Steps

1. **Create backup branch:** `git checkout -b cleanup-backup`
2. **Phase 1 cleanup:** Remove obviously legacy files
3. **Test thoroughly:** Run `python scripts/test_full_e2e_phase1_to_phase6.py`
4. **Phase 2 verification:** Check dependencies on uncertain components
5. **Documentation update:** Consolidate to Phase 6 focus
6. **Final testing:** Ensure demo works perfectly

---

**Total Impact:** ~40% codebase reduction while maintaining full functionality.
**Risk Level:** Low (legacy code well-archived, modern system tested).
**Time Estimate:** 2-4 hours for complete cleanup.

*SliceWise v2 - Streamlined for Production Excellence* 🚀
