"""
Patient-level analysis endpoint for SliceWise Backend.

This module provides the patient-level analysis endpoint for analyzing
stacks of MRI slices with volume estimation.

Extracted from main_v2.py (lines 797-867).
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from typing import List
import numpy as np
from PIL import Image
import io

from app.backend.services.model_loader import ModelManager, get_model_manager
from app.backend.services.segmentation_service import SegmentationService
from app.backend.services.patient_service import PatientService
from app.backend.models.responses import PatientAnalysisResponse


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/patient", tags=["patient"])


# ============================================================================
# Dependency Injection
# ============================================================================

def get_patient_service(
    model_manager: ModelManager = Depends(get_model_manager)
) -> PatientService:
    """Get PatientService instance."""
    segmentation_service = SegmentationService(model_manager)
    return PatientService(model_manager, segmentation_service)


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/analyze_stack", response_model=PatientAnalysisResponse)
async def analyze_patient_stack(
    files: List[UploadFile] = File(...),
    patient_id: str = Form(...),
    threshold: float = Form(0.5),
    min_object_size: int = Form(50),
    slice_thickness_mm: float = Form(1.0, description="Slice thickness for volume calculation"),
    pixel_spacing_mm: float = Form(1.0, description="Pixel spacing in mm"),
    service: PatientService = Depends(get_patient_service)
):
    """
    Analyze a stack of MRI slices for patient-level tumor detection and volume estimation.
    
    Copied from main_v2.py analyze_patient_stack() (lines 797-867).
    
    Args:
        files: List of uploaded MRI slice images
        patient_id: Patient identifier
        threshold: Probability threshold for segmentation
        min_object_size: Minimum tumor area in pixels
        slice_thickness_mm: Slice thickness for volume calculation
        pixel_spacing_mm: Pixel spacing in mm (assumes square pixels)
        service: Injected PatientService
    
    Returns:
        PatientAnalysisResponse with patient-level results
        
    Raises:
        HTTPException: If analysis fails or model not loaded
    """
    if not service.is_available():
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    
    # Validate number of slices
    if len(files) < 1:
        raise HTTPException(
            status_code=400,
            detail="At least 1 slice required for patient analysis"
        )
    
    if len(files) > 500:
        raise HTTPException(
            status_code=400,
            detail="Maximum 500 slices per patient analysis"
        )
    
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
        
        # Run patient-level analysis
        result = await service.analyze_stack(
            images=images,
            patient_id=patient_id,
            filenames=filenames,
            threshold=threshold,
            min_object_size=min_object_size,
            slice_thickness_mm=slice_thickness_mm,
            pixel_spacing_mm=pixel_spacing_mm
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Patient analysis failed: {str(e)}"
        )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Patient router endpoints:")
    print("  POST /patient/analyze_stack")
