"""
Multi-task prediction endpoint for SliceWise Backend.

This module provides the unified multi-task prediction endpoint that
performs both classification and conditional segmentation.

Extracted from main_v2.py (lines 874-971).
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends
import numpy as np
from PIL import Image
import io

from app.backend.services.model_loader import ModelManager, get_model_manager
from app.backend.services.multitask_service import MultiTaskService
from app.backend.models.responses import MultiTaskResponse


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/predict_multitask", tags=["multitask"])


# ============================================================================
# Dependency Injection
# ============================================================================

def get_multitask_service(
    model_manager: ModelManager = Depends(get_model_manager)
) -> MultiTaskService:
    """Get MultiTaskService instance."""
    return MultiTaskService(model_manager)


# ============================================================================
# Endpoints
# ============================================================================

@router.post("", response_model=MultiTaskResponse)
async def predict_multitask(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(True, description="Include Grad-CAM visualization"),
    service: MultiTaskService = Depends(get_multitask_service)
):
    """
    Unified multi-task prediction: classification + conditional segmentation.
    
    Copied from main_v2.py predict_multitask() (lines 874-971).
    
    This endpoint uses the multi-task model to perform both classification
    and segmentation in a single forward pass. Segmentation is only computed
    if the tumor probability is above the threshold (default: 0.3).
    
    Benefits:
    - ~40% faster than separate models
    - 9.4% fewer parameters
    - Excellent performance: 91.3% accuracy, 97.1% sensitivity
    
    Args:
        file: Uploaded image file
        include_gradcam: Whether to include Grad-CAM visualization
        service: Injected MultiTaskService
    
    Returns:
        MultiTaskResponse with classification and optional segmentation
        
    Raises:
        HTTPException: If prediction fails or model not loaded
    """
    if not service.is_available():
        raise HTTPException(status_code=503, detail="Multi-task model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Run conditional prediction (recommended)
        result = await service.predict_conditional(
            image_array,
            include_gradcam=include_gradcam
        )
        
        return result
    
    except Exception as e:
        import traceback
        error_detail = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ Error in /predict_multitask:")
        print(error_detail)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Multi-task router endpoints:")
    print("  POST /predict_multitask")
