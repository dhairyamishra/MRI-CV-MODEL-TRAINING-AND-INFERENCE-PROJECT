"""
PHASE 3: Frontend & User Experience Testing Package

This package contains comprehensive frontend and user experience tests for SliceWise:

test_streamlit_ui_components.py (6 tests)
- Core UI components (header, sidebar, multi-task tab, classification tab, segmentation tab)
- Interactive elements (file upload, progress indicators, result visualization, export functions, settings controls, error display)
- Visualization components (Grad-CAM overlays, segmentation masks, uncertainty maps, ROC curves, confusion matrices, volume rendering)

test_frontend_backend_integration.py (5 tests)
- API client request formation and response parsing
- State management and session persistence
- Real-time updates and streaming responses
- Error handling and retry logic
- End-to-end frontend-backend integration scenarios

test_user_workflow_validation.py (6 tests)
- Classification workflow (upload → predict → view Grad-CAM → export)
- Segmentation workflow (upload → segment → view uncertainty → download)
- Batch processing (upload multiple → monitor progress → review all results)
- Patient analysis (upload stack → analyze volume → explore 3D view)
- Multi-task analysis (upload → get both results → compare outputs)
- Edge case handling and error recovery

Total: 17 comprehensive frontend and user experience tests covering the complete user journey.
"""

# Frontend & User Experience Testing Package for SliceWise Phase 3
