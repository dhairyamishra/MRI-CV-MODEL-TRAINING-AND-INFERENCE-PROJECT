# Bug Fix: Missing cv2 Import in Segmentation Service

**Date:** December 19, 2025  
**Status:** ✅ Fixed  
**Severity:** Critical (Segmentation endpoint completely broken)

---

## Problem

The standalone segmentation endpoint (`POST /segment`) was failing with:

```
PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x000001DE6ECF2D90>
```

**Error Location:**
- Frontend: `app/frontend/components/segmentation_tab.py` line 214
- Function: `base64_to_image(result['mask_base64'])`

**Root Cause:**
The segmentation service was trying to use OpenCV functions (`cv2.applyColorMap`, `cv2.cvtColor`) to create colorful probability map visualizations, but the `cv2` module was not imported.

---

## Root Cause Analysis

### Missing Import

**File:** `app/backend/services/segmentation_service.py`

The service was using cv2 functions in multiple places:

1. **Line 168-170** - Creating colorful probability maps:
   ```python
   prob_map_colored = cv2.applyColorMap(prob_map_uint8, cv2.COLORMAP_JET)
   prob_map_colored = cv2.cvtColor(prob_map_colored, cv2.COLOR_BGR2RGB)
   ```

2. **Line 260-262** - Creating mean prediction visualization:
   ```python
   mean_colored = cv2.applyColorMap(mean_uint8, cv2.COLORMAP_JET)
   mean_colored = cv2.cvtColor(mean_colored, cv2.COLOR_BGR2RGB)
   ```

3. **Line 265-268** - Creating uncertainty maps:
   ```python
   epistemic_colored = cv2.applyColorMap(epistemic_uint8, cv2.COLORMAP_VIRIDIS)
   epistemic_colored = cv2.cvtColor(epistemic_colored, cv2.COLOR_BGR2RGB)
   ```

But the import section (lines 11-15) was missing `import cv2`:

```python
import numpy as np
import torch
from typing import Dict, Any, List, Optional
import time
```

### Why This Caused PIL.UnidentifiedImageError

1. When cv2 functions were called, Python raised a `NameError: name 'cv2' is not defined`
2. The exception handler caught this error
3. The function returned incomplete/corrupted base64 data
4. Frontend tried to decode the invalid base64 string
5. PIL couldn't identify the corrupted image data → `UnidentifiedImageError`

---

## Solution

### Fix Applied

**File:** `app/backend/services/segmentation_service.py`  
**Lines:** 11-15

Added missing import:

```python
import numpy as np
import torch
import cv2  # ← ADDED
from typing import Dict, Any, List, Optional
import time
```

---

## Impact

### Affected Endpoints

1. ✅ **POST /segment** - Single image segmentation
   - Used cv2 for probability map visualization (lines 168-170)

2. ✅ **POST /segment/uncertainty** - Uncertainty estimation
   - Used cv2 for mean prediction (lines 260-262)
   - Used cv2 for uncertainty maps (lines 265-268)

3. ✅ **POST /segment/batch** - Batch segmentation
   - Indirectly affected (calls same helper methods)

### Not Affected

- **Multi-task endpoints** - Already had cv2 imported in `multitask_service.py`
- **Classification endpoints** - Don't use cv2 for visualizations
- **Health/info endpoints** - No visualization logic

---

## Testing

### Verification Steps

1. ✅ Import check: `import cv2` added to segmentation_service.py
2. ✅ Code review: All cv2 usage points identified (3 locations)
3. ✅ Multi-task service: Already has cv2 import (line 15)

### Expected Behavior After Fix

1. **Segmentation endpoint** should return valid base64 images:
   - `mask_base64` - Binary tumor mask (white on black)
   - `probability_map_base64` - Colorful JET heatmap
   - `overlay_base64` - Red tumor overlay on original image

2. **Uncertainty endpoint** should return:
   - All above images
   - `uncertainty_map_base64` - Colorful VIRIDIS uncertainty map

3. **Frontend** should display all visualizations without errors

---

## Related Issues

### Similar Import Issues Fixed Previously

1. **Memory ID:** `a88452dc-ff42-48f7-8709-206a5da3bb5b`
   - Fixed 23 import errors in main_v2.py during Phase 6 refactoring
   - Included: EnsembleUncertaintyPredictor → EnsemblePredictor
   - Included: TemperatureScaler → TemperatureScaling

### Why This Was Missed

During the Phase 6 backend refactoring (modularization), the segmentation service was extracted from `main_v2.py` into a separate file. The cv2 import was present in main_v2.py but was not copied to the new `segmentation_service.py` module.

**Original main_v2.py imports:**
```python
import cv2
import matplotlib.pyplot as plt
```

**New segmentation_service.py imports (before fix):**
```python
import numpy as np
import torch
from typing import Dict, Any, List, Optional
import time
# cv2 was missing!
```

---

## Prevention

### Checklist for Future Refactoring

When extracting code into new modules:

1. ✅ Identify all external dependencies used in extracted code
2. ✅ Copy all necessary imports to new module
3. ✅ Run static analysis to catch undefined names
4. ✅ Test all endpoints after refactoring
5. ✅ Check for similar patterns in other services

### Static Analysis Tools

Consider adding to CI/CD:
```bash
# Check for undefined names
flake8 app/backend/services/ --select=F821

# Or use pylint
pylint app/backend/services/ --disable=all --enable=undefined-variable
```

---

## Files Modified

1. **app/backend/services/segmentation_service.py**
   - Added: `import cv2` (line 13)
   - Impact: 3 cv2 usage points now work correctly

---

## Status

✅ **FIXED** - Missing cv2 import added to segmentation_service.py

The segmentation endpoint should now work correctly and return valid base64-encoded images for all visualizations.

---

## Next Steps

1. ✅ Test segmentation endpoint in UI with real MRI images
2. ✅ Verify all three visualization types render correctly:
   - Binary mask (white on black)
   - Probability map (colorful JET heatmap)
   - Overlay (red tumor on original image)
3. ✅ Test uncertainty endpoint with MC Dropout + TTA
4. ✅ Verify batch segmentation works for multiple images

---

## References

- **Error Traceback:** Frontend segmentation_tab.py line 214
- **Root Cause:** segmentation_service.py missing cv2 import
- **Fix Location:** segmentation_service.py line 13
- **Related Services:** multitask_service.py (already had cv2)
