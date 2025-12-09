"""
Health and information endpoints for SliceWise Backend.

This module provides health check and model information endpoints.

Extracted from main_v2.py (lines 337-450).
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any

from app.backend.services.model_loader import ModelManager, get_model_manager
from app.backend.models.responses import HealthResponse, ModelInfoResponse


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(tags=["health"])


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/", response_model=dict)
async def root():
    """
    Root endpoint with API information.
    
    Copied from main_v2.py (lines 337-365).
    
    Returns:
        Dictionary with API metadata and endpoint information
    """
    return {
        "name": "SliceWise API",
        "version": "2.0.0",
        "description": "Comprehensive Brain Tumor Detection API",
        "endpoints": {
            "health": "/healthz",
            "model_info": "/model/info",
            "multitask": {
                "predict": "/predict_multitask"
            },
            "classification": {
                "classify": "/classify",
                "classify_with_gradcam": "/classify/gradcam",
                "classify_batch": "/classify/batch"
            },
            "segmentation": {
                "segment": "/segment",
                "segment_with_uncertainty": "/segment/uncertainty",
                "segment_batch": "/segment/batch"
            },
            "patient_analysis": {
                "analyze_stack": "/patient/analyze_stack"
            },
            "docs": "/docs"
        }
    }


@router.get("/healthz", response_model=HealthResponse)
async def health_check(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Health check endpoint.
    
    Copied from main_v2.py (lines 368-389).
    
    Args:
        model_manager: Injected ModelManager instance
    
    Returns:
        HealthResponse with system status
    """
    health_info = model_manager.get_health_info()
    
    return HealthResponse(**health_info)


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Get comprehensive model information.
    
    Copied from main_v2.py (lines 392-450).
    
    Args:
        model_manager: Injected ModelManager instance
    
    Returns:
        ModelInfoResponse with detailed model information
    """
    info = model_manager.get_model_info()
    
    return ModelInfoResponse(**info)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Health router endpoints:")
    print("  GET  /")
    print("  GET  /healthz")
    print("  GET  /model/info")
