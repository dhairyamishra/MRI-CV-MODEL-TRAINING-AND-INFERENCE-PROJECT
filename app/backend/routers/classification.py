"""
Classification endpoints for SliceWise Backend.

This module provides classification-related API endpoints including
single image classification, batch processing, and Grad-CAM visualization.

Extracted from main_v2.py (lines 457-568).
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends
from typing import List
import numpy as np
from PIL import Image
import io

from app.backend.services.model_loader import ModelManager, get_model_manager
from app.backend.services.classification_service import ClassificationService
from app.backend.models.responses import ClassificationResponse, BatchResponse
from app.backend.utils.validators import validate_batch_size


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/classify", tags=["classification"])


# ============================================================================
# Dependency Injection
# ============================================================================

def get_classification_service(
    model_manager: ModelManager = Depends(get_model_manager)
) -> ClassificationService:
    """Get ClassificationService instance."""
    return ClassificationService(model_manager)


# ============================================================================
# Endpoints
# ============================================================================

@router.post("", response_model=ClassificationResponse)
async def classify_slice(
    file: UploadFile = File(...),
    return_gradcam: bool = Query(False, description="Include Grad-CAM visualization"),
    service: ClassificationService = Depends(get_classification_service)
):
    """
    Classify a single MRI slice with optional calibration and Grad-CAM.
    
    Copied from main_v2.py classify_slice() (lines 457-511).
    
    Args:
        file: Uploaded image file
        return_gradcam: Whether to include Grad-CAM visualization
        service: Injected ClassificationService
    
    Returns:
        ClassificationResponse with prediction results
        
    Raises:
        HTTPException: If classification fails or model not loaded
    """
    if not service.is_available():
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Run classification
        result = await service.classify_single(
            image_array,
            return_gradcam=return_gradcam,
            apply_calibration=True
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@router.post("/gradcam", response_model=ClassificationResponse)
async def classify_with_gradcam(
    file: UploadFile = File(...),
    service: ClassificationService = Depends(get_classification_service)
):
    """
    Classify with Grad-CAM visualization (convenience endpoint).
    
    Copied from main_v2.py (lines 514-517).
    
    Args:
        file: Uploaded image file
        service: Injected ClassificationService
    
    Returns:
        ClassificationResponse with Grad-CAM overlay
    """
    return await classify_slice(file, return_gradcam=True, service=service)


@router.post("/batch", response_model=BatchResponse)
async def classify_batch(
    files: List[UploadFile] = File(...),
    return_gradcam: bool = Query(False, description="Include Grad-CAM (not recommended for large batches)"),
    service: ClassificationService = Depends(get_classification_service)
):
    """
    Classify multiple MRI slices in a batch.
    
    Copied from main_v2.py classify_batch() (lines 520-568).
    
    Args:
        files: List of uploaded image files
        return_gradcam: Whether to include Grad-CAM
        service: Injected ClassificationService
    
    Returns:
        BatchResponse with batch results and summary
        
    Raises:
        HTTPException: If batch processing fails or exceeds limits
    """
    if not service.is_available():
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    # Validate batch size
    validate_batch_size(len(files), max_batch_size=100, operation="classification")
    
    try:
        # Read all images
        images = []
        filenames = []
        
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image)
            images.append(image_array)
            filenames.append(file.filename)
        
        # Run batch classification
        result = await service.classify_batch(
            images,
            filenames=filenames,
            return_gradcam=return_gradcam
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch classification failed: {str(e)}"
        )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Classification router endpoints:")
    print("  POST /classify")
    print("  POST /classify/gradcam")
    print("  POST /classify/batch")
