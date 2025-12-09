"""
Segmentation endpoints for SliceWise Backend.

This module provides segmentation-related API endpoints including
single image segmentation, uncertainty estimation, and batch processing.

Extracted from main_v2.py (lines 575-790).
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends
from typing import List
import numpy as np
from PIL import Image
import io

from app.backend.services.model_loader import ModelManager, get_model_manager
from app.backend.services.segmentation_service import SegmentationService
from app.backend.models.responses import SegmentationResponse, BatchResponse
from app.backend.utils.validators import validate_batch_size


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/segment", tags=["segmentation"])


# ============================================================================
# Dependency Injection
# ============================================================================

def get_segmentation_service(
    model_manager: ModelManager = Depends(get_model_manager)
) -> SegmentationService:
    """Get SegmentationService instance."""
    return SegmentationService(model_manager)


# ============================================================================
# Endpoints
# ============================================================================

@router.post("", response_model=SegmentationResponse)
async def segment_slice(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Probability threshold"),
    min_object_size: int = Query(50, ge=0, description="Minimum tumor area in pixels"),
    apply_postprocessing: bool = Query(True, description="Apply morphological post-processing"),
    return_overlay: bool = Query(True, description="Return overlay visualization"),
    service: SegmentationService = Depends(get_segmentation_service)
):
    """
    Segment a single MRI slice with optional post-processing.
    
    Copied from main_v2.py segment_slice() (lines 575-657).
    
    Args:
        file: Uploaded image file
        threshold: Probability threshold for segmentation
        min_object_size: Minimum tumor area in pixels
        apply_postprocessing: Whether to apply morphological operations
        return_overlay: Whether to return overlay visualization
        service: Injected SegmentationService
    
    Returns:
        SegmentationResponse with segmentation results
        
    Raises:
        HTTPException: If segmentation fails or model not loaded
    """
    if not service.is_available():
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Run segmentation
        result = await service.segment_single(
            image_array,
            threshold=threshold,
            apply_postprocessing=apply_postprocessing,
            min_object_size=min_object_size,
            return_overlay=return_overlay
        )
        
        return result
    
    except Exception as e:
        import traceback
        print(f"Error in /segment endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )


@router.post("/uncertainty", response_model=SegmentationResponse)
async def segment_with_uncertainty(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    min_object_size: int = Query(50, ge=0),
    mc_iterations: int = Query(10, ge=1, le=50, description="MC Dropout iterations"),
    use_tta: bool = Query(True, description="Use Test-Time Augmentation"),
    service: SegmentationService = Depends(get_segmentation_service)
):
    """
    Segment with uncertainty estimation using MC Dropout and/or TTA.
    
    Copied from main_v2.py segment_with_uncertainty() (lines 660-726).
    
    Args:
        file: Uploaded image file
        threshold: Probability threshold
        min_object_size: Minimum tumor area in pixels
        mc_iterations: Number of MC Dropout iterations
        use_tta: Whether to use Test-Time Augmentation
        service: Injected SegmentationService
    
    Returns:
        SegmentationResponse with uncertainty metrics
        
    Raises:
        HTTPException: If uncertainty estimation fails
    """
    if not service.is_available():
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Run uncertainty estimation
        result = await service.segment_with_uncertainty(
            image_array,
            threshold=threshold,
            min_object_size=min_object_size,
            mc_iterations=mc_iterations,
            use_tta=use_tta
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Uncertainty estimation failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchResponse)
async def segment_batch(
    files: List[UploadFile] = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    min_object_size: int = Query(50, ge=0),
    service: SegmentationService = Depends(get_segmentation_service)
):
    """
    Segment multiple MRI slices in a batch.
    
    Copied from main_v2.py segment_batch() (lines 729-790).
    
    Args:
        files: List of uploaded image files
        threshold: Probability threshold
        min_object_size: Minimum tumor area in pixels
        service: Injected SegmentationService
    
    Returns:
        BatchResponse with batch results and summary
        
    Raises:
        HTTPException: If batch segmentation fails or exceeds limits
    """
    if not service.is_available():
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    
    # Validate batch size
    validate_batch_size(len(files), max_batch_size=100, operation="segmentation")
    
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
        
        # Run batch segmentation
        result = await service.segment_batch(
            images,
            filenames=filenames,
            threshold=threshold,
            min_object_size=min_object_size
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch segmentation failed: {str(e)}"
        )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Segmentation router endpoints:")
    print("  POST /segment")
    print("  POST /segment/uncertainty")
    print("  POST /segment/batch")
