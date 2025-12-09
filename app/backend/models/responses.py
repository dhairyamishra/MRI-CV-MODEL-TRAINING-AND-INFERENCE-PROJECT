"""
Pydantic response models for SliceWise Backend API.

This module contains all Pydantic models used for API response validation.
Copied from main_v2.py (lines 80-150) and enhanced with additional documentation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ============================================================================
# Health & Info Responses
# ============================================================================

class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Provides information about the API status and loaded models.
    """
    status: str = Field(..., description="API status: 'healthy' or 'no_models_loaded'")
    classifier_loaded: bool = Field(..., description="Whether classifier model is loaded")
    segmentation_loaded: bool = Field(..., description="Whether segmentation model is loaded")
    calibration_loaded: bool = Field(..., description="Whether calibration is loaded")
    multitask_loaded: bool = Field(..., description="Whether multi-task model is loaded")
    device: str = Field(..., description="Device being used: 'cuda' or 'cpu'")
    timestamp: str = Field(..., description="ISO timestamp of health check")


class ModelInfoResponse(BaseModel):
    """
    Response model for model information endpoint.
    
    Provides detailed information about all loaded models.
    """
    classifier: Dict[str, Any] = Field(..., description="Classifier model information")
    segmentation: Dict[str, Any] = Field(..., description="Segmentation model information")
    multitask: Dict[str, Any] = Field(..., description="Multi-task model information")
    features: List[str] = Field(..., description="List of available features")


# ============================================================================
# Classification Responses
# ============================================================================

class ClassificationResponse(BaseModel):
    """
    Response model for classification predictions.
    
    Contains prediction results with optional calibration and Grad-CAM visualization.
    """
    predicted_class: int = Field(..., description="Predicted class index (0=No Tumor, 1=Tumor)")
    predicted_label: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    calibrated_probabilities: Optional[Dict[str, float]] = Field(
        None, 
        description="Temperature-scaled calibrated probabilities"
    )
    gradcam_overlay: Optional[str] = Field(
        None, 
        description="Base64-encoded Grad-CAM overlay image"
    )


# ============================================================================
# Segmentation Responses
# ============================================================================

class SegmentationResponse(BaseModel):
    """
    Response model for segmentation predictions.
    
    Contains segmentation mask, probability map, and optional uncertainty estimation.
    """
    has_tumor: bool = Field(..., description="Whether tumor was detected")
    tumor_probability: float = Field(..., ge=0.0, le=1.0, description="Maximum tumor probability")
    tumor_area_pixels: int = Field(..., ge=0, description="Tumor area in pixels")
    tumor_area_mm2: Optional[float] = Field(
        None, 
        ge=0.0, 
        description="Tumor area in mm² (requires pixel spacing metadata)"
    )
    num_components: int = Field(..., ge=0, description="Number of connected tumor components")
    mask_base64: str = Field(..., description="Base64-encoded binary segmentation mask")
    probability_map_base64: Optional[str] = Field(
        None, 
        description="Base64-encoded probability map"
    )
    uncertainty_map_base64: Optional[str] = Field(
        None, 
        description="Base64-encoded uncertainty map (if uncertainty estimation used)"
    )
    overlay_base64: Optional[str] = Field(
        None, 
        description="Base64-encoded overlay of mask on original image"
    )
    metrics: Optional[Dict[str, float]] = Field(
        None, 
        description="Additional metrics (e.g., uncertainty statistics)"
    )


# ============================================================================
# Patient Analysis Responses
# ============================================================================

class PatientAnalysisResponse(BaseModel):
    """
    Response model for patient-level analysis.
    
    Contains aggregated results from analyzing a stack of MRI slices.
    """
    patient_id: str = Field(..., description="Patient identifier")
    num_slices: int = Field(..., ge=1, description="Number of slices analyzed")
    has_tumor: bool = Field(..., description="Whether tumor was detected in patient")
    tumor_volume_mm3: Optional[float] = Field(
        None, 
        ge=0.0, 
        description="Estimated tumor volume in mm³"
    )
    affected_slices: int = Field(..., ge=0, description="Number of slices with tumor")
    slice_predictions: List[Dict[str, Any]] = Field(
        ..., 
        description="Per-slice prediction results"
    )
    patient_level_metrics: Dict[str, float] = Field(
        ..., 
        description="Patient-level aggregated metrics"
    )


# ============================================================================
# Batch Processing Responses
# ============================================================================

class BatchResponse(BaseModel):
    """
    Response model for batch processing.
    
    Contains results from processing multiple images in a batch.
    """
    num_images: int = Field(..., ge=1, description="Number of images processed")
    processing_time_seconds: float = Field(..., ge=0.0, description="Total processing time")
    results: List[Dict[str, Any]] = Field(..., description="Per-image results")
    summary: Dict[str, Any] = Field(..., description="Batch summary statistics")


# ============================================================================
# Multi-Task Responses
# ============================================================================

class MultiTaskResponse(BaseModel):
    """
    Response model for multi-task predictions.
    
    Contains both classification and optional segmentation results from the
    unified multi-task model.
    """
    classification: Dict[str, Any] = Field(..., description="Classification results")
    segmentation: Optional[Dict[str, Any]] = Field(
        None, 
        description="Segmentation results (if computed)"
    )
    segmentation_computed: bool = Field(
        ..., 
        description="Whether segmentation was computed"
    )
    recommendation: str = Field(..., description="Clinical recommendation based on results")
    gradcam_overlay: Optional[str] = Field(
        None, 
        description="Base64-encoded Grad-CAM overlay"
    )
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Create a sample health response
    health = HealthResponse(
        status="healthy",
        classifier_loaded=True,
        segmentation_loaded=True,
        calibration_loaded=True,
        multitask_loaded=True,
        device="cuda",
        timestamp="2025-12-08T21:00:00"
    )
    print("Health Response:")
    print(health.model_dump_json(indent=2))
    
    # Example: Create a sample classification response
    classification = ClassificationResponse(
        predicted_class=1,
        predicted_label="Tumor",
        confidence=0.95,
        probabilities={"No Tumor": 0.05, "Tumor": 0.95}
    )
    print("\nClassification Response:")
    print(classification.model_dump_json(indent=2))
