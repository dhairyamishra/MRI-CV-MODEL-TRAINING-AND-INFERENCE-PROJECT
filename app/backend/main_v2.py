"""
FastAPI backend for SliceWise MRI Brain Tumor Detection - Phase 6 Complete Demo.

This enhanced API provides comprehensive endpoints for:
- Classification with calibration and Grad-CAM
- Segmentation with uncertainty estimation
- Patient-level analysis and volume estimation
- Batch processing and stack analysis
- Model information and health checks

Integrates all components from Phases 0-5.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image
import io
import base64
import tempfile
import zipfile
import json
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predict import ClassifierPredictor
from src.inference.infer_seg2d import SegmentationPredictor
from src.inference.uncertainty import MCDropoutPredictor, TTAPredictor, EnsemblePredictor
from src.inference.postprocess import postprocess_mask
from src.eval.grad_cam import GradCAM, visualize_single_image
from src.eval.calibration import TemperatureScaling
from src.eval.patient_level_eval import PatientLevelEvaluator

# Initialize FastAPI app
app = FastAPI(
    title="SliceWise API v2",
    description="Comprehensive Brain Tumor Detection API with Classification, Segmentation, and Uncertainty Estimation",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (loaded on startup)
classifier_predictor: Optional[ClassifierPredictor] = None
segmentation_predictor: Optional[SegmentationPredictor] = None
uncertainty_predictor: Optional[EnsemblePredictor] = None
temperature_scaler: Optional[TemperatureScaling] = None

CLASSIFIER_LOADED = False
SEGMENTATION_LOADED = False
CALIBRATION_LOADED = False


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    classifier_loaded: bool
    segmentation_loaded: bool
    calibration_loaded: bool
    device: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    classifier: Dict[str, Any]
    segmentation: Dict[str, Any]
    features: List[str]


class ClassificationResponse(BaseModel):
    """Response model for classification predictions."""
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    calibrated_probabilities: Optional[Dict[str, float]] = None
    gradcam_overlay: Optional[str] = None


class SegmentationResponse(BaseModel):
    """Response model for segmentation predictions."""
    has_tumor: bool
    tumor_probability: float
    tumor_area_pixels: int
    tumor_area_mm2: Optional[float] = None
    num_components: int
    mask_base64: str
    probability_map_base64: Optional[str] = None
    uncertainty_map_base64: Optional[str] = None
    overlay_base64: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


class PatientAnalysisResponse(BaseModel):
    """Response model for patient-level analysis."""
    patient_id: str
    num_slices: int
    has_tumor: bool
    tumor_volume_mm3: Optional[float] = None
    affected_slices: int
    slice_predictions: List[Dict[str, Any]]
    patient_level_metrics: Dict[str, float]


class BatchResponse(BaseModel):
    """Response model for batch processing."""
    num_images: int
    processing_time_seconds: float
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global classifier_predictor, segmentation_predictor, uncertainty_predictor
    global temperature_scaler, CLASSIFIER_LOADED, SEGMENTATION_LOADED, CALIBRATION_LOADED
    
    print("=" * 80)
    print("SliceWise API v2 - Starting up...")
    print("=" * 80)
    
    # Load classifier
    try:
        classifier_path = project_root / "checkpoints" / "cls" / "best_model.pth"
        if classifier_path.exists():
            classifier_predictor = ClassifierPredictor(
                checkpoint_path=str(classifier_path),
                model_name="efficientnet"
            )
            CLASSIFIER_LOADED = True
            print(f"✓ Classifier loaded from {classifier_path}")
        else:
            print(f"⚠ Classifier not found at {classifier_path}")
    except Exception as e:
        print(f"✗ Error loading classifier: {e}")
    
    # Load calibration
    try:
        calibration_path = project_root / "checkpoints" / "cls" / "temperature_scaler.pth"
        if calibration_path.exists() and CLASSIFIER_LOADED:
            temperature_scaler = TemperatureScaling()
            checkpoint = torch.load(calibration_path, map_location=classifier_predictor.device, weights_only=False)
            temperature_scaler.temperature.data = checkpoint['temperature']
            CALIBRATION_LOADED = True
            print(f"✓ Calibration loaded (T={temperature_scaler.temperature.item():.4f})")
        else:
            print(f"⚠ Calibration not found at {calibration_path}")
    except Exception as e:
        print(f"⚠ Error loading calibration: {e}")
    
    # Load segmentation
    try:
        seg_path = project_root / "checkpoints" / "seg" / "best_model.pth"
        if seg_path.exists():
            segmentation_predictor = SegmentationPredictor(
                checkpoint_path=str(seg_path)
            )
            SEGMENTATION_LOADED = True
            print(f"✓ Segmentation loaded from {seg_path}")
            
            # Initialize uncertainty predictor
            uncertainty_predictor = EnsemblePredictor(
                model=segmentation_predictor.model,
                n_mc_samples=10,
                use_tta=True,
                device=segmentation_predictor.device
            )
            print(f"✓ Uncertainty estimation initialized")
        else:
            print(f"⚠ Segmentation not found at {seg_path}")
    except Exception as e:
        print(f"✗ Error loading segmentation: {e}")
    
    print("=" * 80)
    device = classifier_predictor.device if classifier_predictor else "unknown"
    print(f"Device: {device}")
    print(f"Classifier: {'✓' if CLASSIFIER_LOADED else '✗'}")
    print(f"Calibration: {'✓' if CALIBRATION_LOADED else '✗'}")
    print(f"Segmentation: {'✓' if SEGMENTATION_LOADED else '✗'}")
    print("=" * 80)


# ============================================================================
# Utility Functions
# ============================================================================

def numpy_to_base64_png(array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded PNG."""
    # Normalize to 0-255 if needed
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = (array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    if len(array.shape) == 2:
        image = Image.fromarray(array, mode='L')
    else:
        image = Image.fromarray(array)
    
    # Encode to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create overlay of mask on image."""
    # Ensure image is RGB
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()
    
    # Normalize image to 0-255
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    
    # Create colored mask (red for tumor)
    colored_mask = np.zeros_like(image_rgb)
    colored_mask[mask > 0] = [255, 0, 0]  # Red
    
    # Blend
    overlay = ((1 - alpha) * image_rgb + alpha * colored_mask).astype(np.uint8)
    return overlay


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SliceWise API v2",
        "version": "2.0.0",
        "description": "Comprehensive Brain Tumor Detection API",
        "endpoints": {
            "health": "/healthz",
            "model_info": "/model/info",
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


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    device = "unknown"
    if classifier_predictor:
        device = str(classifier_predictor.device)
    elif segmentation_predictor:
        device = str(segmentation_predictor.device)
    
    status = "healthy" if (CLASSIFIER_LOADED or SEGMENTATION_LOADED) else "no_models_loaded"
    
    return HealthResponse(
        status=status,
        classifier_loaded=CLASSIFIER_LOADED,
        segmentation_loaded=SEGMENTATION_LOADED,
        calibration_loaded=CALIBRATION_LOADED,
        device=device,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get comprehensive model information."""
    classifier_info = {}
    if CLASSIFIER_LOADED:
        classifier_info = {
            "architecture": "EfficientNet-B0",
            "num_classes": 2,
            "class_names": classifier_predictor.class_names,
            "input_size": (256, 256),
            "calibrated": CALIBRATION_LOADED
        }
    
    segmentation_info = {}
    if SEGMENTATION_LOADED:
        segmentation_info = {
            "architecture": "U-Net 2D",
            "parameters": "31.4M",
            "input_size": (256, 256),
            "output_channels": 1,
            "uncertainty_estimation": True
        }
    
    features = []
    if CLASSIFIER_LOADED:
        features.extend(["classification", "grad_cam"])
        if CALIBRATION_LOADED:
            features.append("calibration")
    if SEGMENTATION_LOADED:
        features.extend(["segmentation", "uncertainty_estimation", "patient_level_analysis"])
    
    return ModelInfoResponse(
        classifier=classifier_info,
        segmentation=segmentation_info,
        features=features
    )


# ============================================================================
# Classification Endpoints
# ============================================================================

@app.post("/classify", response_model=ClassificationResponse)
async def classify_slice(
    file: UploadFile = File(...),
    return_gradcam: bool = Query(False, description="Include Grad-CAM visualization")
):
    """
    Classify a single MRI slice with optional calibration and Grad-CAM.
    """
    if not CLASSIFIER_LOADED:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Run prediction
        result = classifier_predictor.predict(image_array, return_probabilities=True)
        
        # Apply calibration if available
        calibrated_probs = None
        if CALIBRATION_LOADED:
            tensor = classifier_predictor.preprocess_image(image_array)
            with torch.no_grad():
                logits = classifier_predictor.model(tensor)
                calibrated_logits = temperature_scaler(logits)
                calibrated_probs_tensor = torch.softmax(calibrated_logits, dim=1)[0]
                calibrated_probs = {
                    name: float(calibrated_probs_tensor[i])
                    for i, name in enumerate(classifier_predictor.class_names)
                }
        
        # Generate Grad-CAM if requested
        gradcam_base64 = None
        if return_gradcam:
            tensor = classifier_predictor.preprocess_image(image_array)
            cam, overlay = visualize_single_image(
                classifier_predictor.model,
                tensor,
                classifier_predictor.device
            )
            gradcam_base64 = numpy_to_base64_png(overlay)
        
        return ClassificationResponse(
            predicted_class=result['predicted_class'],
            predicted_label=result['predicted_label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            calibrated_probabilities=calibrated_probs,
            gradcam_overlay=gradcam_base64
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/classify/gradcam", response_model=ClassificationResponse)
async def classify_with_gradcam(file: UploadFile = File(...)):
    """Classify with Grad-CAM visualization (convenience endpoint)."""
    return await classify_slice(file, return_gradcam=True)


@app.post("/classify/batch", response_model=BatchResponse)
async def classify_batch(
    files: List[UploadFile] = File(...),
    return_gradcam: bool = Query(False)
):
    """Classify multiple MRI slices in a batch."""
    if not CLASSIFIER_LOADED:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per batch")
    
    try:
        import time
        start_time = time.time()
        
        results = []
        tumor_count = 0
        
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image)
            
            result = classifier_predictor.predict(image_array, return_probabilities=True)
            
            if result['predicted_label'] == 'Tumor':
                tumor_count += 1
            
            results.append({
                "filename": file.filename,
                **result
            })
        
        processing_time = time.time() - start_time
        
        return BatchResponse(
            num_images=len(files),
            processing_time_seconds=processing_time,
            results=results,
            summary={
                "tumor_detected": tumor_count,
                "no_tumor": len(files) - tumor_count,
                "avg_time_per_image": processing_time / len(files)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")


# ============================================================================
# Segmentation Endpoints
# ============================================================================

@app.post("/segment", response_model=SegmentationResponse)
async def segment_slice(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Probability threshold"),
    min_object_size: int = Query(50, ge=0, description="Minimum tumor area in pixels"),
    apply_postprocessing: bool = Query(True, description="Apply morphological post-processing"),
    return_overlay: bool = Query(True, description="Return overlay visualization")
):
    """
    Segment a single MRI slice with optional post-processing.
    """
    if not SEGMENTATION_LOADED:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Ensure grayscale
        if len(image_array.shape) == 3:
            image_array = image_array[:, :, 0]
        
        # Normalize to [0, 1]
        if image_array.max() > 1.0:
            image_array = image_array.astype(np.float32) / 255.0
        
        # Run segmentation
        result = segmentation_predictor.predict_slice(image_array)
        prob_map = result.get('prob', result['mask'])
        binary_mask = result['mask']
        
        # Apply post-processing if requested
        if apply_postprocessing:
            binary_mask, stats = postprocess_mask(
                prob_map,
                threshold=threshold,
                min_object_size=min_object_size,
                fill_holes_size=None,
                morphology_op='close',
                morphology_kernel=3
            )
        else:
            binary_mask = (prob_map > threshold).astype(np.uint8)
            stats = {
                'num_components': np.unique(binary_mask).sum() - 1,
                'total_area': binary_mask.sum()
            }
        
        # Calculate metrics
        has_tumor = stats.get('final_pixels', stats.get('total_area', 0)) > 0
        tumor_probability = float(prob_map.max()) if has_tumor else 0.0
        
        # Create visualizations
        mask_base64 = numpy_to_base64_png(binary_mask * 255)
        prob_map_base64 = numpy_to_base64_png(prob_map)
        
        overlay_base64 = None
        if return_overlay:
            overlay = create_overlay(image_array, binary_mask)
            overlay_base64 = numpy_to_base64_png(overlay)
        
        return SegmentationResponse(
            has_tumor=has_tumor,
            tumor_probability=tumor_probability,
            tumor_area_pixels=int(stats.get('final_pixels', stats.get('total_area', 0))),
            tumor_area_mm2=None,  # Would need pixel spacing metadata
            num_components=int(stats.get('num_components', 0)),
            mask_base64=mask_base64,
            probability_map_base64=prob_map_base64,
            overlay_base64=overlay_base64
        )
    
    except Exception as e:
        import traceback
        print(f"Error in /segment endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/segment/uncertainty", response_model=SegmentationResponse)
async def segment_with_uncertainty(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    min_object_size: int = Query(50, ge=0),
    mc_iterations: int = Query(10, ge=1, le=50, description="MC Dropout iterations"),
    use_tta: bool = Query(True, description="Use Test-Time Augmentation")
):
    """
    Segment with uncertainty estimation using MC Dropout and/or TTA.
    """
    if not SEGMENTATION_LOADED:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        if len(image_array.shape) == 3:
            image_array = image_array[:, :, 0]
        if image_array.max() > 1.0:
            image_array = image_array.astype(np.float32) / 255.0
        
        # Run uncertainty estimation
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).float()
        result = uncertainty_predictor.predict_with_uncertainty(image_tensor)
        
        # Apply post-processing
        binary_mask, stats = postprocess_mask(
            result['mean'],
            threshold=threshold,
            min_object_size=min_object_size
        )
        
        # Create visualizations
        mask_base64 = numpy_to_base64_png(binary_mask * 255)
        prob_map_base64 = numpy_to_base64_png((result['mean'] * 255).astype(np.uint8))
        uncertainty_base64 = numpy_to_base64_png((result['epistemic'] * 255).astype(np.uint8))
        overlay_base64 = numpy_to_base64_png(create_overlay(image_array, binary_mask))
        
        return SegmentationResponse(
            has_tumor=stats['total_area'] > 0,
            tumor_probability=float(result['mean'].max()),
            tumor_area_pixels=int(stats['total_area']),
            num_components=int(stats['num_components']),
            mask_base64=mask_base64,
            prob_map_base64=prob_map_base64,
            overlay_base64=overlay_base64,
            uncertainty_map_base64=uncertainty_base64,
            metrics={
                'epistemic_uncertainty': float(result['epistemic'].mean()),
                'aleatoric_uncertainty': float(result['aleatoric'].mean()),
                'total_uncertainty': float(result['std'].mean())
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Uncertainty estimation failed: {str(e)}")


@app.post("/segment/batch", response_model=BatchResponse)
async def segment_batch(
    files: List[UploadFile] = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    min_object_size: int = Query(50, ge=0)
):
    """Segment multiple MRI slices in a batch."""
    if not SEGMENTATION_LOADED:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per batch")
    
    try:
        import time
        start_time = time.time()
        
        results = []
        tumor_count = 0
        total_tumor_area = 0
        
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image)
            
            if len(image_array.shape) == 3:
                image_array = image_array[:, :, 0]
            if image_array.max() > 1.0:
                image_array = image_array.astype(np.float32) / 255.0
            
            result = segmentation_predictor.predict_slice(image_array)
            prob_map = result.get('prob', result['mask'])
            binary_mask = result['mask']
            binary_mask, stats = postprocess_mask(prob_map, threshold=threshold, min_object_size=min_object_size)
            
            has_tumor = stats['total_area'] > 0
            if has_tumor:
                tumor_count += 1
                total_tumor_area += stats['total_area']
            
            results.append({
                "filename": file.filename,
                "has_tumor": has_tumor,
                "tumor_area_pixels": int(stats['total_area']),
                "tumor_probability": float(prob_map.max())
            })
        
        processing_time = time.time() - start_time
        
        return BatchResponse(
            num_images=len(files),
            processing_time_seconds=processing_time,
            results=results,
            summary={
                "slices_with_tumor": tumor_count,
                "slices_without_tumor": len(files) - tumor_count,
                "total_tumor_area_pixels": total_tumor_area,
                "avg_time_per_image": processing_time / len(files)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch segmentation failed: {str(e)}")


# ============================================================================
# Patient-Level Analysis Endpoints
# ============================================================================

@app.post("/patient/analyze_stack", response_model=PatientAnalysisResponse)
async def analyze_patient_stack(
    files: List[UploadFile] = File(...),
    patient_id: str = Form(...),
    threshold: float = Form(0.5),
    min_object_size: int = Form(50),
    slice_thickness_mm: float = Form(1.0, description="Slice thickness for volume calculation")
):
    """
    Analyze a stack of MRI slices for patient-level tumor detection and volume estimation.
    """
    if not SEGMENTATION_LOADED:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    
    try:
        slice_predictions = []
        affected_slices = 0
        total_tumor_volume_pixels = 0
        
        for idx, file in enumerate(files):
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image)
            
            if len(image_array.shape) == 3:
                image_array = image_array[:, :, 0]
            if image_array.max() > 1.0:
                image_array = image_array.astype(np.float32) / 255.0
            
            result = segmentation_predictor.predict_slice(image_array)
            prob_map = result.get('prob', result['mask'])
            binary_mask = result['mask']
            binary_mask, stats = postprocess_mask(prob_map, threshold=threshold, min_object_size=min_object_size)
            
            has_tumor = stats['total_area'] > 0
            if has_tumor:
                affected_slices += 1
                total_tumor_volume_pixels += stats['total_area']
            
            slice_predictions.append({
                "slice_index": idx,
                "filename": file.filename,
                "has_tumor": has_tumor,
                "tumor_area_pixels": int(stats['total_area']),
                "max_probability": float(prob_map.max())
            })
        
        # Patient-level decision: tumor present if any slice has tumor
        patient_has_tumor = affected_slices > 0
        
        # Estimate volume (simplified: area × thickness)
        # Assumes square pixels with 1mm spacing (would need DICOM metadata for accuracy)
        tumor_volume_mm3 = None
        if patient_has_tumor:
            pixel_area_mm2 = 1.0  # Assume 1mm × 1mm pixels
            tumor_volume_mm3 = total_tumor_volume_pixels * pixel_area_mm2 * slice_thickness_mm
        
        return PatientAnalysisResponse(
            patient_id=patient_id,
            num_slices=len(files),
            has_tumor=patient_has_tumor,
            tumor_volume_mm3=tumor_volume_mm3,
            affected_slices=affected_slices,
            slice_predictions=slice_predictions,
            patient_level_metrics={
                "affected_slice_ratio": affected_slices / len(files),
                "avg_tumor_area_per_slice": total_tumor_volume_pixels / len(files),
                "max_tumor_area": max([p['tumor_area_pixels'] for p in slice_predictions])
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patient analysis failed: {str(e)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
