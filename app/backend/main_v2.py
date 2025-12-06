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
import matplotlib.pyplot as plt

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
from src.inference.multi_task_predictor import MultiTaskPredictor
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
multitask_predictor: Optional[MultiTaskPredictor] = None
temperature_scaler: Optional[TemperatureScaling] = None

CLASSIFIER_LOADED = False
SEGMENTATION_LOADED = False
CALIBRATION_LOADED = False
MULTITASK_LOADED = False


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    classifier_loaded: bool
    segmentation_loaded: bool
    calibration_loaded: bool
    multitask_loaded: bool
    device: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    classifier: Dict[str, Any]
    segmentation: Dict[str, Any]
    multitask: Dict[str, Any]
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


class MultiTaskResponse(BaseModel):
    """Response model for multi-task predictions."""
    classification: Dict[str, Any]
    segmentation: Optional[Dict[str, Any]] = None
    segmentation_computed: bool
    recommendation: str
    gradcam_overlay: Optional[str] = None
    processing_time_ms: float


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global classifier_predictor, segmentation_predictor, uncertainty_predictor
    global multitask_predictor, temperature_scaler
    global CLASSIFIER_LOADED, SEGMENTATION_LOADED, CALIBRATION_LOADED, MULTITASK_LOADED
    
    print("=" * 80)
    print("SliceWise API v2 - Starting up...")
    print("=" * 80)
    
    # Load multi-task model (PRIORITY)
    try:
        multitask_path = project_root / "checkpoints" / "multitask_joint" / "best_model.pth"
        if multitask_path.exists():
            multitask_predictor = MultiTaskPredictor(
                checkpoint_path=str(multitask_path),
                base_filters=32,
                depth=3,
                classification_threshold=0.3,
                segmentation_threshold=0.5
            )
            MULTITASK_LOADED = True
            print(f"[OK] Multi-task model loaded from {multitask_path}")
        else:
            print(f"[WARN] Multi-task model not found at {multitask_path}")
    except Exception as e:
        import traceback
        print(f"[ERROR] Error loading multi-task model: {e}")
        print(f"[ERROR] Traceback:")
        traceback.print_exc()
    
    # Load classifier
    try:
        classifier_path = project_root / "checkpoints" / "cls" / "best_model.pth"
        if classifier_path.exists():
            classifier_predictor = ClassifierPredictor(
                checkpoint_path=str(classifier_path),
                model_name="efficientnet"
            )
            CLASSIFIER_LOADED = True
            print(f"[OK] Classifier loaded from {classifier_path}")
        else:
            print(f"[WARN] Classifier not found at {classifier_path}")
    except Exception as e:
        print(f"[ERROR] Error loading classifier: {e}")
    
    # Load calibration
    try:
        calibration_path = project_root / "checkpoints" / "cls" / "temperature_scaler.pth"
        if calibration_path.exists() and CLASSIFIER_LOADED:
            temperature_scaler = TemperatureScaling()
            checkpoint = torch.load(calibration_path, map_location=classifier_predictor.device, weights_only=False)
            temperature_scaler.temperature.data = checkpoint['temperature']
            CALIBRATION_LOADED = True
            print(f"[OK] Calibration loaded (T={temperature_scaler.temperature.item():.4f})")
        else:
            print(f"[WARN] Calibration not found at {calibration_path}")
    except Exception as e:
        print(f"[WARN] Error loading calibration: {e}")
    
    # Load segmentation
    try:
        seg_path = project_root / "checkpoints" / "seg" / "best_model.pth"
        if seg_path.exists():
            segmentation_predictor = SegmentationPredictor(
                checkpoint_path=str(seg_path)
            )
            SEGMENTATION_LOADED = True
            print(f"[OK] Segmentation loaded from {seg_path}")
            
            # Initialize uncertainty predictor
            uncertainty_predictor = EnsemblePredictor(
                model=segmentation_predictor.model,
                n_mc_samples=10,
                use_tta=True,
                device=segmentation_predictor.device
            )
            print(f"[OK] Uncertainty estimation initialized")
        else:
            print(f"[WARN] Segmentation not found at {seg_path}")
    except Exception as e:
        print(f"[ERROR] Error loading segmentation: {e}")
    
    print("=" * 80)
    device = "unknown"
    if multitask_predictor:
        device = str(multitask_predictor.device)
    elif classifier_predictor:
        device = str(classifier_predictor.device)
    elif segmentation_predictor:
        device = str(segmentation_predictor.device)
    
    print(f"Device: {device}")
    print(f"Multi-task: {'[OK]' if MULTITASK_LOADED else '[NO]'}")
    print(f"Classifier: {'[OK]' if CLASSIFIER_LOADED else '[NO]'}")
    print(f"Calibration: {'[OK]' if CALIBRATION_LOADED else '[NO]'}")
    print(f"Segmentation: {'[OK]' if SEGMENTATION_LOADED else '[NO]'}")
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


def preprocess_image_for_segmentation(image_array: np.ndarray) -> np.ndarray:
    """
    Preprocess image for segmentation model.
    
    CRITICAL: Must match training preprocessing (z-score normalization)!
    
    Args:
        image_array: Input image array
    
    Returns:
        Preprocessed image array
    """
    # Ensure grayscale
    if len(image_array.shape) == 3:
        image_array = image_array[:, :, 0]
    
    # Convert to float32
    image_array = image_array.astype(np.float32)
    
    # Normalize to [0, 1] first if needed
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    
    # CRITICAL: Apply z-score normalization (same as training!)
    # The model was trained on z-score normalized images
    mean = np.mean(image_array)
    std = np.std(image_array)
    if std > 0:
        image_array = (image_array - mean) / std
    else:
        image_array = image_array - mean
    
    return image_array


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


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    device = "unknown"
    if multitask_predictor:
        device = str(multitask_predictor.device)
    elif classifier_predictor:
        device = str(classifier_predictor.device)
    elif segmentation_predictor:
        device = str(segmentation_predictor.device)
    
    status = "healthy" if (MULTITASK_LOADED or CLASSIFIER_LOADED or SEGMENTATION_LOADED) else "no_models_loaded"
    
    return HealthResponse(
        status=status,
        classifier_loaded=CLASSIFIER_LOADED,
        segmentation_loaded=SEGMENTATION_LOADED,
        calibration_loaded=CALIBRATION_LOADED,
        multitask_loaded=MULTITASK_LOADED,
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
    
    multitask_info = {}
    if MULTITASK_LOADED:
        model_params = multitask_predictor.get_model_info()
        multitask_info = {
            "architecture": "Multi-Task U-Net",
            "parameters": model_params['parameters'],
            "input_size": model_params['input_size'],
            "tasks": model_params['tasks'],
            "classification_threshold": model_params['classification_threshold'],
            "segmentation_threshold": model_params['segmentation_threshold'],
            "performance": {
                "classification_accuracy": 0.9130,
                "classification_sensitivity": 0.9714,
                "classification_roc_auc": 0.9184,
                "segmentation_dice": 0.7650,
                "segmentation_iou": 0.6401,
                "combined_metric": 0.8390
            }
        }
    
    features = []
    if MULTITASK_LOADED:
        features.extend(["multi_task", "conditional_segmentation", "unified_inference"])
    if CLASSIFIER_LOADED:
        features.extend(["classification", "grad_cam"])
        if CALIBRATION_LOADED:
            features.append("calibration")
    if SEGMENTATION_LOADED:
        features.extend(["segmentation", "uncertainty_estimation", "patient_level_analysis"])
    
    return ModelInfoResponse(
        classifier=classifier_info,
        segmentation=segmentation_info,
        multitask=multitask_info,
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
        
        # Save original image for visualization (before z-score normalization)
        if len(image_array.shape) == 3:
            image_original = image_array[:, :, 0].astype(np.float32)
        else:
            image_original = image_array.astype(np.float32)
        
        if image_original.max() > 1.0:
            image_original = image_original / 255.0
        
        # Preprocess image for segmentation (applies z-score normalization)
        image_array = preprocess_image_for_segmentation(image_array)
        
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
        
        # Create visualizations using ORIGINAL image (not z-score normalized)
        mask_base64 = numpy_to_base64_png(binary_mask * 255)
        prob_map_base64 = numpy_to_base64_png(prob_map)
        
        overlay_base64 = None
        if return_overlay:
            overlay = create_overlay(image_original, binary_mask)
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
        
        # Save original image for visualization (before z-score normalization)
        if len(image_array.shape) == 3:
            image_original = image_array[:, :, 0].astype(np.float32)
        else:
            image_original = image_array.astype(np.float32)
        
        if image_original.max() > 1.0:
            image_original = image_original / 255.0
        
        # Preprocess image for segmentation (applies z-score normalization)
        image_array = preprocess_image_for_segmentation(image_array)
        
        # Run uncertainty estimation
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).float()
        result = uncertainty_predictor.predict_with_uncertainty(image_tensor)
        
        # Apply post-processing
        binary_mask, stats = postprocess_mask(
            result['mean'],
            threshold=threshold,
            min_object_size=min_object_size
        )
        
        # Create visualizations using ORIGINAL image (not z-score normalized)
        mask_base64 = numpy_to_base64_png(binary_mask * 255)
        prob_map_base64 = numpy_to_base64_png((result['mean'] * 255).astype(np.uint8))
        uncertainty_base64 = numpy_to_base64_png((result['epistemic'] * 255).astype(np.uint8))
        overlay_base64 = numpy_to_base64_png(create_overlay(image_original, binary_mask))
        
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
            
            # Preprocess image for segmentation
            image_array = preprocess_image_for_segmentation(image_array)
            
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
            
            # Preprocess image for segmentation
            image_array = preprocess_image_for_segmentation(image_array)
            
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
# Multi-Task Endpoint
# ============================================================================

@app.post("/predict_multitask", response_model=MultiTaskResponse)
async def predict_multitask(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(True, description="Include Grad-CAM visualization")
):
    """
    Unified multi-task prediction: classification + conditional segmentation.
    
    This endpoint uses the multi-task model to perform both classification
    and segmentation in a single forward pass. Segmentation is only computed
    if the tumor probability is above the threshold (default: 0.3).
    
    Benefits:
    - ~40% faster than separate models
    - 9.4% fewer parameters
    - Excellent performance: 91.3% accuracy, 97.1% sensitivity
    """
    if not MULTITASK_LOADED:
        raise HTTPException(status_code=503, detail="Multi-task model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Run conditional prediction (recommended)
        if include_gradcam:
            result = multitask_predictor.predict_full(image_array, include_gradcam=True)
        else:
            result = multitask_predictor.predict_conditional(image_array)
        
        # Build response
        response = {
            "classification": result['classification'],
            "segmentation_computed": result['segmentation_computed'],
            "recommendation": result['recommendation']
        }
        
        # Add segmentation if computed
        if result['segmentation_computed']:
            mask = result['segmentation']['mask']
            prob_map = result['segmentation']['prob_map']
            
            # Create overlay
            image_original = result['image_original']
            overlay = create_overlay(image_original, mask, alpha=0.4)
            
            response["segmentation"] = {
                "mask_available": True,
                "tumor_area_pixels": result['segmentation']['tumor_area_pixels'],
                "tumor_percentage": result['segmentation']['tumor_percentage'],
                "mask_base64": numpy_to_base64_png(mask),
                "prob_map_base64": numpy_to_base64_png(prob_map),
                "overlay_base64": numpy_to_base64_png(overlay)
            }
        else:
            response["segmentation"] = {
                "mask_available": False,
                "message": "Segmentation not computed (tumor probability below threshold)"
            }
        
        # Add Grad-CAM if requested
        if include_gradcam and 'gradcam' in result:
            heatmap = result['gradcam']['heatmap']
            image_original = result['image_original']
            
            # Resize heatmap to match image size
            import cv2
            heatmap_resized = cv2.resize(heatmap, (image_original.shape[1], image_original.shape[0]))
            
            # Create Grad-CAM overlay
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]  # RGB
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            
            # Blend with original
            image_rgb = np.stack([image_original] * 3, axis=-1)
            if image_rgb.max() <= 1.0:
                image_rgb = (image_rgb * 255).astype(np.uint8)
            
            gradcam_overlay = ((1 - 0.4) * image_rgb + 0.4 * heatmap_colored).astype(np.uint8)
            response["gradcam_overlay"] = numpy_to_base64_png(gradcam_overlay)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        response["processing_time_ms"] = round(processing_time, 2)
        
        return MultiTaskResponse(**response)
    
    except Exception as e:
        import traceback
        error_detail = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ Error in /predict_multitask:")
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


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
