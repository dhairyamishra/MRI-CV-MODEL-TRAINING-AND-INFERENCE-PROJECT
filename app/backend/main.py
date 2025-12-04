"""
FastAPI backend for SliceWise MRI Brain Tumor Detection.

This API provides endpoints for:
- Health checks
- Single slice classification
- Batch classification
- Model information
"""

import sys
from pathlib import Path
from typing import Optional, List
import numpy as np
from PIL import Image
import io
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predict import ClassifierPredictor
from src.eval.grad_cam import GradCAM, visualize_single_image
import torch

# Initialize FastAPI app
app = FastAPI(
    title="SliceWise API",
    description="Brain Tumor Detection API for MRI Slices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance (loaded on startup)
predictor: Optional[ClassifierPredictor] = None
MODEL_LOADED = False


# Pydantic models for request/response
class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: dict


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str
    num_classes: int
    class_names: List[str]
    input_size: tuple


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor, MODEL_LOADED
    
    try:
        # Path to best model checkpoint
        checkpoint_path = project_root / "checkpoints" / "cls" / "best_model.pth"
        
        if checkpoint_path.exists():
            predictor = ClassifierPredictor(
                checkpoint_path=str(checkpoint_path),
                model_name="efficientnet"
            )
            MODEL_LOADED = True
            print(f"✓ Model loaded successfully from {checkpoint_path}")
            print(f"✓ Using device: {predictor.device}")
        else:
            print(f"⚠ Model checkpoint not found at {checkpoint_path}")
            print("⚠ API will run but predictions will fail")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        MODEL_LOADED = False


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SliceWise API",
        "version": "1.0.0",
        "description": "Brain Tumor Detection API for MRI Slices",
        "endpoints": {
            "health": "/healthz",
            "model_info": "/model/info",
            "classify": "/classify_slice",
            "classify_batch": "/classify_batch",
            "docs": "/docs"
        }
    }


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "model_not_loaded",
        model_loaded=MODEL_LOADED,
        device=str(predictor.device) if predictor else "unknown"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name="EfficientNet-B0",
        num_classes=2,
        class_names=predictor.class_names,
        input_size=(256, 256)
    )


@app.post("/classify_slice", response_model=PredictionResponse)
async def classify_slice(
    file: UploadFile = File(...),
    return_probabilities: bool = Query(True, description="Return class probabilities")
):
    """
    Classify a single MRI slice.
    
    Args:
        file: Uploaded image file (JPG, PNG, etc.)
        return_probabilities: Whether to return class probabilities
    
    Returns:
        Prediction results including class, label, confidence, and probabilities
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Run prediction
        result = predictor.predict(image_array, return_probabilities)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/classify_batch")
async def classify_batch(
    files: List[UploadFile] = File(...),
    return_probabilities: bool = Query(True, description="Return class probabilities")
):
    """
    Classify multiple MRI slices in a batch.
    
    Args:
        files: List of uploaded image files
        return_probabilities: Whether to return class probabilities
    
    Returns:
        List of prediction results
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    try:
        # Read all images
        images = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            images.append(np.array(image))
        
        # Run batch prediction
        results = predictor.predict_batch(images, return_probabilities)
        
        return {
            "num_images": len(images),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/classify_with_gradcam")
async def classify_with_gradcam(
    file: UploadFile = File(...)
):
    """
    Classify a slice and return Grad-CAM visualization.
    
    Args:
        file: Uploaded image file
    
    Returns:
        Prediction results with Grad-CAM heatmap (base64 encoded)
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Run prediction
        result = predictor.predict(image_array, return_probabilities=True)
        
        # Generate Grad-CAM
        tensor = predictor.preprocess_image(image_array)
        cam, overlay = visualize_single_image(
            predictor.model,
            tensor,
            predictor.device
        )
        
        # Convert overlay to base64
        overlay_pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format='PNG')
        overlay_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            **result,
            "gradcam_overlay": overlay_base64
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
