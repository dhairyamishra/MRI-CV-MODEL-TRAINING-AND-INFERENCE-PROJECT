# 🎨 Frontend Modular Refactoring Plan

**Project:** SliceWise MRI Brain Tumor Detection  
**Date:** December 8, 2025  
**Goal:** Create a clean, modular, maintainable Streamlit frontend structure

---

## 📊 Overview

**Current State:**
- Single monolithic file: `app_v2.py` (1,187 lines)
- CSS embedded in Python strings
- All logic mixed together
- Hard to maintain and test

**Target State:**
- Modular structure with separated concerns
- CSS in external files
- Reusable components
- Easy to test and maintain
- Team collaboration friendly

---

## 📋 Task List

### **Phase 1: Foundation (Tasks 1-4)** 
✅ **Infrastructure & Configuration**

- [x] **Task 1:** Create directory structure: `styles/`, `components/`, `utils/`, `config/` 
- [x] **Task 2:** Create `config/settings.py` - Extract constants (API_URL, colors, thresholds) 
- [x] **Task 3:** Create `styles/main.css` - Extract CSS and **fix warning-box bug** 
- [x] **Task 4:** Create `styles/theme.css` - CSS variables for theming 

**Phase 1 Summary:**
- 4 directories created
- 215 lines in settings.py
- 238 lines in main.css (warning-box bug FIXED!)
- 226 lines in theme.css
- **Total: 679 lines of configuration & styling**

---

### **Phase 2: Utilities (Tasks 5-7)** 
✅ **Reusable Helper Functions**

- [x] **Task 5:** Create `utils/api_client.py` - API functions (health, model info, classify, segment)
- [x] **Task 6:** Create `utils/image_utils.py` - Image utilities (base64 conversion, etc.)
- [x] **Task 7:** Create `utils/validators.py` - File validation (size, format, dimensions)

**Phase 2 Summary:**
- 455 lines in api_client.py (9 API functions)
- 390 lines in image_utils.py (13 image functions)
- 451 lines in validators.py (10 validation functions)
- **Total: 1,296 lines of utility code**

---

### **Phase 3: UI Components (Tasks 8-14)** 
✅ **Modular UI Components**

- [x] **Task 8:** Create `components/header.py` - Header with branding & disclaimer
- [x] **Task 9:** Create `components/sidebar.py` - System status with multi-task model
- [x] **Task 10:** Create `components/multitask_tab.py` - Multi-task prediction tab
- [x] **Task 11:** Create `components/classification_tab.py` - Classification tab
- [x] **Task 12:** Create `components/segmentation_tab.py` - Segmentation tab
- [x] **Task 13:** Create `components/batch_tab.py` - Batch processing tab
- [x] **Task 14:** Create `components/patient_tab.py` - Patient analysis tab
- [ ] **Task 15:** Create `components/__init__.py` - Export all components

**Phase 3 Summary:**
- 95 lines in header.py (2 functions)
- 213 lines in sidebar.py (2 functions)
- 332 lines in multitask_tab.py (8 helper functions)
- 238 lines in classification_tab.py (5 helper functions)
- 276 lines in segmentation_tab.py (4 helper functions)
- 273 lines in batch_tab.py (7 helper functions)
- 332 lines in patient_tab.py (9 helper functions)
- **Total: 1,759 lines of component code with 37 helper functions**

---

### **Phase 4: Integration (Tasks 15-18)** 
✅ **Bring It All Together**

- [ ] **Task 15:** Create `components/__init__.py` - Export all components
- [ ] **Task 16:** Refactor `app_v2.py` - Use modular components (clean main file)
- [ ] **Task 17:** Test the application - Verify all functionality works
- [ ] **Task 18:** Create `README.md` - Document new structure

---

## 🎯 Final Structure

```
app/frontend/
├── __init__.py
├── app_v2.py                    # ← Refactored (minimal, clean)
├── app.py                       # ← Legacy (keep for reference)
├── config/
│   ├── __init__.py              
│   └── settings.py              # Constants & configuration
├── styles/
│   ├── main.css                 # Main CSS (warning-box fixed!)
│   └── theme.css                # CSS variables & theming
├── utils/
│   ├── __init__.py             
│   ├── api_client.py            # API communication
│   ├── image_utils.py           # Image processing
│   └── validators.py            # Input validation
├── components/
│   ├── __init__.py             
│   ├── header.py                # App header
│   ├── sidebar.py               # System status sidebar
│   ├── multitask_tab.py         # Multi-task tab
│   ├── classification_tab.py    # Classification tab
│   ├── segmentation_tab.py      # Segmentation tab
│   ├── batch_tab.py             # Batch processing tab
│   └── patient_tab.py           # Patient analysis tab
├── README.md                    # Documentation
└── REFACTORING_PLAN.md          # This file
```

---

## 📈 Progress Tracking

**Total Tasks:** 18  
**Completed:** 14 
**In Progress:** 1 
**Remaining:** 3  

**Progress:** 78% (14/18 tasks)

**Estimated Time:** 45-60 minutes  
**Time Elapsed:** ~40 minutes  
**Time Remaining:** ~5-20 minutes

**Files Created:** 14 / 16  
**Files Modified:** 0 / 1  

**Lines of Code Written:** 3,734 lines

---

## ✅ Benefits

- **Maintainability:** Each component in its own file
- **Testability:** Can unit test components separately
- **Collaboration:** Multiple developers can work simultaneously
- **Reusability:** Components can be reused across different apps
- **Best Practices:** Follows Streamlit and Python conventions
- **Scalability:** Easy to add new features and components

---

## 🔧 Key Improvements

1. **CSS Separation:** External CSS files instead of embedded strings
2. **Fixed Bugs:** Warning-box background color fixed (#000000 → #fff3cd)
3. **API Client:** Centralized API communication with error handling
4. **Validation:** Input validation for file uploads
5. **Type Hints:** Proper type annotations throughout
6. **Documentation:** Comprehensive docstrings and README

---

## 📝 Notes

- Keep `app.py` as legacy reference
- Maintain backward compatibility
- All existing functionality preserved
- No breaking changes to API
- Streamlit-native approach (no React/Next.js)

---

## 🚀 Next Steps

1. Review and approve this plan
2. Execute tasks sequentially (1-18)
3. Test after each phase
4. Deploy refactored version
5. Archive old version

---

## 🎉 Completed Milestones

### **Phase 1: Foundation** (Dec 8, 2025 - 15:12)

**Achievements:**
- Created modular directory structure
- Extracted all constants to `config/settings.py`
- Separated CSS to `styles/main.css` with bug fixes
- Created comprehensive theme system in `styles/theme.css`
- Added responsive design & accessibility features
- Implemented dark mode support
- Total: 679 lines of clean, organized code

**Critical Fixes:**
- Warning-box background: #000000 → #fff3cd (now readable!)
- CSS variables for consistent theming
- Responsive breakpoints for mobile
- Accessibility features (keyboard nav, high contrast, reduced motion)

---

### **Phase 2: Utilities** (Dec 8, 2025 - 15:20)

**Achievements:**
- Created `utils/api_client.py` with 9 API functions (455 lines)
  - Health checks, model info
  - Classification (single, batch)
  - Segmentation (basic, uncertainty, batch)
  - Multi-task prediction
  - Patient-level analysis
- Created `utils/image_utils.py` with 13 image functions (390 lines)
  - Base64 conversion (4 functions)
  - Image processing (4 functions)
  - Validation (2 functions)
  - Display helpers (3 functions)
- Created `utils/validators.py` with 10 validation functions (451 lines)
  - File upload validation (3 functions)
  - Input parameter validation (3 functions)
  - Data integrity validation (2 functions)
  - Streamlit-specific validators (2 functions)
- Total: 1,296 lines of utility code

**Key Features:**
- Comprehensive error handling with tuple returns (result, error)
- Type hints throughout all functions
- Detailed docstrings with examples
- Streamlit integration for auto-display of validation messages
- Uses centralized settings from config/settings.py

---

### **Phase 3: UI Components** (Dec 8, 2025 - 15:38)

**Achievements:**
- Created `components/header.py` (95 lines, 2 functions)
  - Main header with title, subtitle, version badge
  - Medical disclaimer
  - Simplified header variant
- Created `components/sidebar.py` (213 lines, 2 functions)
  - API health status
  - Multi-task model status display
  - Model information expandables
  - About section
- Created `components/multitask_tab.py` (332 lines, 8 helper functions)
  - Unified classification + segmentation
  - Performance metrics
  - Grad-CAM visualization
  - Clinical interpretation
- Created `components/classification_tab.py` (238 lines, 5 helper functions)
  - Tumor detection
  - Raw and calibrated probabilities
  - Grad-CAM visualization
- Created `components/segmentation_tab.py` (276 lines, 4 helper functions)
  - Tumor localization
  - Basic and uncertainty-based segmentation
  - 4-column visualizations
- Created `components/batch_tab.py` (273 lines, 7 helper functions)
  - Batch classification and segmentation
  - Image preview
  - Summary statistics
  - CSV download
- Created `components/patient_tab.py` (332 lines, 9 helper functions)
  - Patient-level analysis
  - Volume estimation
  - Slice-by-slice analysis
  - Distribution charts
  - CSV and JSON download
- Total: 1,759 lines of component code with 37 helper functions

**Key Features:**
- All components use session state for results
- Consistent use of ModelConfig and UIConfig
- Clean API client integration
- Comprehensive error handling
- Modular helper functions for better organization

---

**Status:** Phase 1-3 Complete (78%) | Phase 4 In Progress (Task 15: components/__init__.py)
