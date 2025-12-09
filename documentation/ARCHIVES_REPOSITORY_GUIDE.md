# SliceWise Archives - Legacy Code Repository

**Version:** Archive 1.0 (Phase 1-5 → Phase 6 Migration)  
**Date:** December 8, 2025  
**Status:** 🏛️ Historical Archive (Deprecated Code)  

---

## 🎯 Executive Summary

The `archives/` directory serves as a **historical repository** for deprecated code and configurations from earlier phases of the SliceWise project. This archive was created during the **major architectural transition** from Phase 1-5 (individual models) to Phase 6 (unified multi-task architecture).

**Key Purpose:**
- **Historical Preservation**: Maintain codebase evolution record
- **Research Reference**: Enable comparison studies between architectures
- **Migration Documentation**: Track what changed and why
- **Educational Value**: Demonstrate architectural evolution

**⚠️ Important Notice:**
- **Code is NOT maintained** and may not work with current dependencies
- **For current development**, use the active codebase in `src/`, `scripts/`, etc.
- **Educational/research purposes only** - not for production use

---

## 🏗️ Archive Structure & Contents

### Directory Organization

```
archives/
├── app/                          # 🖥️ Legacy Application Code
│   ├── backend/main_v2.py       # Monolithic API (986 lines → 160 lines modular)
│   └── frontend/app_v2.py       # Monolithic UI (1,187 lines → 151 lines modular)
│
├── configs/                      # ⚙️ Legacy Configuration System
│   └── multitask-model/         # Old individual configs (before hierarchical system)
│       ├── multi_task_production.yaml
│       ├── multitask_cls_head_production.yaml
│       ├── multitask_cls_head_quick_test.yaml
│       ├── multitask_joint_production.yaml
│       ├── multitask_joint_quick_test.yaml
│       ├── multitask_seg_warmup.yaml
│       ├── multitask_seg_warmup_production.yaml
│       └── multitask_seg_warmup_quick_test.yaml
│
└── scripts/                      # 🔧 Legacy Scripts (Phase 1-5)
    ├── README.md                # Archive documentation
    ├── cleanup_legacy_scripts.py # Archiving automation script
    ├── phase1-5_training/       # Individual model training
    ├── phase1-5_evaluation/     # Individual model evaluation
    ├── phase1-5_calibration/    # Individual model calibration
    └── phase1-5_demo/           # Individual model demos
```

---

## 📚 Historical Context - Phase Evolution

### Phase 1-5: Individual Models Architecture
**Legacy Approach (Archived):**
- **Separate Models**: Individual classifier and segmentation models
- **Individual Training**: Separate training pipelines for each task
- **Manual Coordination**: No unified inference or evaluation
- **Maintenance Burden**: Duplication across similar components

### Phase 6: Multi-Task Architecture (Current)
**Modern Approach:**
- **Unified Model**: Single multi-task model with shared encoder
- **3-Stage Training**: Progressive optimization (warmup → head → joint)
- **40% Faster Inference**: Single forward pass for both tasks
- **9.4% Fewer Parameters**: Efficient parameter sharing

### Why the Archive Exists

**Architectural Evolution:**
```python
# Phase 1-5: Separate Models
classifier = Classifier()        # 4M parameters
segmentation = UNet2D()         # 31M parameters
# Total: 35M parameters, 2 forward passes

# Phase 6: Multi-Task Model
multitask = MultiTaskModel()    # 31.7M parameters (9.4% reduction)
# Total: 31.7M parameters, 1 forward pass, 40% faster
```

---

## 📱 Legacy Application Code (`archives/app/`)

### Backend: `main_v2.py` (37,368 bytes)
**Status:** Monolithic API implementation
**Replaced by:** `app/backend/main.py` (160 lines, modular)

**What it contained:**
- **Single-file API**: All endpoints in one massive file
- **Manual routing**: Hardcoded endpoint definitions
- **Tight coupling**: Business logic mixed with HTTP handling
- **Limited scalability**: Difficult to maintain and extend

**Migration impact:**
- **84% code reduction**: 986 lines → 160 lines modular
- **Service layer**: Separated business logic from HTTP concerns
- **Dependency injection**: Clean, testable architecture
- **Error handling**: Centralized exception management

### Frontend: `app_v2.py` (52,085 bytes)
**Status:** Monolithic Streamlit application
**Replaced by:** `app/frontend/app.py` + 14 component files

**What it contained:**
- **Single-file UI**: All interface logic in one file
- **Embedded CSS**: Styling mixed with Python code
- **Tight coupling**: UI logic mixed with business logic
- **Maintenance burden**: Difficult to modify and extend

**Migration impact:**
- **87% code reduction**: 1,187 lines → 151 lines main file
- **Modular components**: Separated concerns by functionality
- **External CSS**: Professional styling with theme support
- **Component reusability**: Clean, maintainable architecture

---

## ⚙️ Legacy Configuration System (`archives/configs/`)

### Individual Config Files (Pre-Hierarchical System)

**Status:** Manual configuration management
**Replaced by:** `configs/base/`, `configs/stages/`, `configs/modes/`, `configs/final/`

**Archived configs:**
- `multi_task_production.yaml` - Production training settings
- `multitask_cls_head_*.yaml` - Classification head training
- `multitask_joint_*.yaml` - Joint fine-tuning
- `multitask_seg_warmup_*.yaml` - Segmentation warmup

### Configuration Evolution

**Phase 1-5 Problems:**
- **Massive duplication**: 70-90% identical content across configs
- **Maintenance nightmare**: Change required editing 8+ files
- **Inconsistency risk**: Easy to introduce errors
- **Scalability barrier**: Adding new configs required extensive copying

**Phase 6 Solution:**
- **Hierarchical system**: Base → Stage → Mode → Final
- **64% reduction**: 1,100 lines → 365 base configs
- **87% less work**: Change global settings in one place
- **Automated consistency**: Generated configs guarantee uniformity

---

## 🔧 Legacy Scripts Archive (`archives/scripts/`)

### Archive Organization by Phase

#### `phase1-5_training/` (6 scripts)
**Legacy training scripts for individual models:**

| Script | Purpose | Status |
|--------|---------|---------|
| `train_classifier.py` | Classification model training | Superseded |
| `train_segmentation.py` | Segmentation model training | Superseded |
| `train_classifier_brats.py` | BraTS classification training | Superseded |
| `train_brats_e2e.py` | End-to-end BraTS training | Superseded |
| `train_production.py` | Production training pipeline | Superseded |
| `train_controller.py` | Training orchestration | Superseded |

**Replaced by:** `scripts/training/multitask/` (3-stage pipeline)

#### `phase1-5_evaluation/` (3 scripts)
**Legacy evaluation scripts for individual models:**

| Script | Purpose | Status |
|--------|---------|---------|
| `evaluate_classifier.py` | Classification evaluation | Superseded |
| `evaluate_segmentation.py` | Segmentation evaluation | Superseded |
| `generate_gradcam.py` | Grad-CAM visualization | Superseded |

**Replaced by:** `scripts/evaluation/multitask/` (unified evaluation)

#### `phase1-5_calibration/` (2 scripts)
**Legacy calibration scripts:**

| Script | Purpose | Status |
|--------|---------|---------|
| `calibrate_classifier.py` | Temperature scaling | Superseded |
| `view_calibration_results.py` | Calibration visualization | Superseded |

**Replaced by:** Integrated calibration in multi-task evaluation

#### `phase1-5_demo/` (2 scripts)
**Legacy demo and testing scripts:**

| Script | Purpose | Status |
|--------|---------|---------|
| `run_demo_with_production_models.py` | Production demo | Superseded |
| `test_full_e2e_phase1_to_phase6.py` | E2E testing | Superseded |

**Replaced by:** `scripts/demo/` (PM2-managed multi-task demos)

### Archiving Process

**Automated Cleanup Script:** `cleanup_legacy_scripts.py`

**What it did:**
1. **Identified legacy scripts** based on functionality analysis
2. **Created archive structure** organized by phase and function
3. **Moved deprecated code** to `archives/scripts/phase1-5_*`
4. **Updated documentation** to reflect current system
5. **Preserved git history** for research purposes

---

## 📊 Performance & Architecture Comparison

### Code Metrics Evolution

| Metric | Phase 1-5 (Archived) | Phase 6 (Current) | Improvement |
|--------|---------------------|-------------------|-------------|
| **Backend LOC** | 986 lines | 160 lines | **84% reduction** |
| **Frontend LOC** | 1,187 lines | 151 lines | **87% reduction** |
| **Config LOC** | 1,100 lines | 365 lines | **64% reduction** |
| **Inference Speed** | 2 passes | 1 pass | **40% faster** |
| **Parameters** | 35M | 31.7M | **9.4% fewer** |
| **Accuracy** | ~85% | 91.3% | **6.8% improvement** |

### Architecture Benefits

**Multi-Task vs Individual Models:**
- **Unified Encoder**: Learned features optimal for both tasks
- **Parameter Sharing**: Reduced memory footprint
- **Single Forward Pass**: Improved inference efficiency
- **Task Interaction**: Classification benefits from segmentation features
- **Maintenance**: Single codebase vs multiple separate systems

---

## 🔍 Research & Educational Value

### Architectural Evolution Study

**Why preserve this archive:**
- **Educational**: Demonstrate software architecture evolution
- **Research**: Enable comparison studies between approaches
- **Documentation**: Track design decisions and trade-offs
- **Reference**: Historical context for future improvements

### Code Archaeology

**What researchers can learn:**
- **Monolithic → Modular**: Transition from single-file to component architecture
- **Individual → Unified**: Evolution from separate to multi-task models
- **Manual → Automated**: Shift from manual to hierarchical configuration
- **Maintenance → Scalability**: Journey from maintenance burden to sustainable development

### Migration Case Study

**Lessons learned:**
- **Incremental refactoring**: Gradual architectural improvement
- **Testing importance**: Comprehensive testing enabled safe transitions
- **Documentation value**: Clear documentation facilitated smooth migration
- **Performance benefits**: Architectural changes delivered measurable improvements

---

## 🚨 Usage Warnings & Best Practices

### ⚠️ Critical Warnings

**Do NOT use archived code for:**
- **Production systems**: Code is unmaintained and may have security issues
- **New development**: Use current Phase 6 architecture instead
- **Medical applications**: Archived code lacks current safety validations
- **Research without review**: Always verify code correctness

### ✅ Appropriate Uses

**Safe uses of archived code:**
- **Historical research**: Study architectural evolution
- **Educational purposes**: Teach software engineering concepts
- **Comparative studies**: Benchmark different approaches
- **Documentation examples**: Illustrate evolution of design patterns

### Access Guidelines

**For researchers accessing archived code:**
1. **Review current system first**: Understand Phase 6 architecture
2. **Document purpose**: Clearly state why archived code is needed
3. **Verify compatibility**: Check if code works with current environment
4. **Cite appropriately**: Reference both archived and current versions
5. **Report issues**: Document any problems found in archived code

---

## 🔄 Future Archive Management

### Planned Evolution

**Archive lifecycle:**
- **Active preservation**: Maintain for research and educational use
- **Periodic review**: Assess continued relevance and value
- **Potential migration**: Move to separate historical repository if needed
- **Documentation updates**: Keep archive documentation current

### Potential Additions

**Future archival candidates:**
- **Phase 6 legacy**: When Phase 7 architecture is developed
- **Experimental branches**: Failed experiments with lessons learned
- **Alternative approaches**: Different architectural explorations
- **Performance studies**: Comprehensive benchmarking data

---

## 📚 Related Documentation

- **[MULTITASK_LEARNING_COMPLETE.md](documentation/MULTITASK_LEARNING_COMPLETE.md)** - Phase 6 architecture
- **[SCRIPTS_ARCHITECTURE_AND_USAGE.md](documentation/SCRIPTS_ARCHITECTURE_AND_USAGE.md)** - Current scripts
- **[CONFIG_SYSTEM_ARCHITECTURE.md](documentation/CONFIG_SYSTEM_ARCHITECTURE.md)** - Current configuration
- **[APP_ARCHITECTURE_AND_FUNCTIONALITY.md](documentation/APP_ARCHITECTURE_AND_FUNCTIONALITY.md)** - Current application

---

*Archives preserved for historical research and educational purposes. For current development, use the active Phase 6 codebase.*
