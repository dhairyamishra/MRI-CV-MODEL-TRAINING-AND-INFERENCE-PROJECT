"""
API Client for SliceWise Frontend.

This module handles all communication with the FastAPI backend,
including health checks, model info, and prediction requests.
"""

from typing import Dict, List, Optional, Any, Tuple
import requests
from pathlib import Path

# Import settings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    API_URL,
    API_TIMEOUT_SECONDS,
    API_HEALTH_CHECK_TIMEOUT,
    API_BATCH_TIMEOUT,
    API_PATIENT_ANALYSIS_TIMEOUT
)


# ============================================================================
# Health & Info Functions
# ============================================================================

def check_api_health() -> Optional[Dict[str, Any]]:
    """
    Check if API is running and healthy.
    
    Returns:
        Dict with health status or None if API is unavailable.
        
    Example:
        >>> health = check_api_health()
        >>> if health and health['status'] == 'healthy':
        >>>     print("API is ready!")
    """
    try:
        response = requests.get(
            f"{API_URL}/healthz",
            timeout=API_HEALTH_CHECK_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None


def get_model_info() -> Optional[Dict[str, Any]]:
    """
    Get comprehensive model information from API.
    
    Returns:
        Dict with model details or None if request fails.
        
    Example:
        >>> info = get_model_info()
        >>> if info and info.get('classifier'):
        >>>     print(f"Classifier: {info['classifier']['architecture']}")
    """
    try:
        response = requests.get(
            f"{API_URL}/model/info",
            timeout=API_HEALTH_CHECK_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None


# ============================================================================
# Classification Functions
# ============================================================================

def classify_image(
    image_bytes: bytes,
    return_gradcam: bool = False
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Classify a single MRI slice.
    
    Args:
        image_bytes: Image file as bytes
        return_gradcam: Whether to include Grad-CAM visualization
        
    Returns:
        Tuple of (result_dict, error_message)
        
    Example:
        >>> with open('mri.png', 'rb') as f:
        >>>     result, error = classify_image(f.read(), return_gradcam=True)
        >>> if result:
        >>>     print(f"Prediction: {result['predicted_label']}")
    """
    try:
        files = {"file": ("image.png", image_bytes, "image/png")}
        params = {"return_gradcam": return_gradcam}
        
        response = requests.post(
            f"{API_URL}/classify",
            files=files,
            params=params,
            timeout=API_TIMEOUT_SECONDS
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def classify_batch(
    files_list: List[Tuple[str, bytes]],
    return_gradcam: bool = False
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Classify multiple MRI slices in a batch.
    
    Args:
        files_list: List of (filename, image_bytes) tuples
        return_gradcam: Whether to include Grad-CAM visualizations
        
    Returns:
        Tuple of (result_dict, error_message)
        
    Example:
        >>> files = [("img1.png", bytes1), ("img2.png", bytes2)]
        >>> result, error = classify_batch(files)
        >>> if result:
        >>>     print(f"Processed {result['num_images']} images")
    """
    try:
        files = [("files", (name, data, "image/png")) for name, data in files_list]
        params = {"return_gradcam": return_gradcam}
        
        response = requests.post(
            f"{API_URL}/classify/batch",
            files=files,
            params=params,
            timeout=API_BATCH_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return None, "Batch request timed out. Try with fewer images."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


# ============================================================================
# Segmentation Functions
# ============================================================================

def segment_image(
    image_bytes: bytes,
    threshold: float = 0.5,
    min_area: int = 50,
    apply_postprocessing: bool = True,
    return_overlay: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Segment tumor regions in an MRI slice.
    
    Args:
        image_bytes: Image file as bytes
        threshold: Probability threshold (0.0-1.0)
        min_area: Minimum tumor area in pixels
        apply_postprocessing: Apply morphological post-processing
        return_overlay: Return overlay visualization
        
    Returns:
        Tuple of (result_dict, error_message)
        
    Example:
        >>> with open('mri.png', 'rb') as f:
        >>>     result, error = segment_image(f.read(), threshold=0.5)
        >>> if result:
        >>>     print(f"Tumor detected: {result['has_tumor']}")
    """
    try:
        files = {"file": ("image.png", image_bytes, "image/png")}
        params = {
            "threshold": threshold,
            "min_object_size": min_area,
            "apply_postprocessing": apply_postprocessing,
            "return_overlay": return_overlay
        }
        
        response = requests.post(
            f"{API_URL}/segment",
            files=files,
            params=params,
            timeout=API_TIMEOUT_SECONDS
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def segment_with_uncertainty(
    image_bytes: bytes,
    threshold: float = 0.5,
    min_area: int = 50,
    mc_iterations: int = 10,
    use_tta: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Segment with uncertainty estimation using MC Dropout and TTA.
    
    Args:
        image_bytes: Image file as bytes
        threshold: Probability threshold (0.0-1.0)
        min_area: Minimum tumor area in pixels
        mc_iterations: Number of MC Dropout iterations
        use_tta: Use Test-Time Augmentation
        
    Returns:
        Tuple of (result_dict, error_message)
        
    Example:
        >>> with open('mri.png', 'rb') as f:
        >>>     result, error = segment_with_uncertainty(f.read(), mc_iterations=10)
        >>> if result and result.get('metrics'):
        >>>     print(f"Uncertainty: {result['metrics']['epistemic_uncertainty']}")
    """
    try:
        files = {"file": ("image.png", image_bytes, "image/png")}
        params = {
            "threshold": threshold,
            "min_object_size": min_area,
            "mc_iterations": mc_iterations,
            "use_tta": use_tta
        }
        
        response = requests.post(
            f"{API_URL}/segment/uncertainty",
            files=files,
            params=params,
            timeout=API_TIMEOUT_SECONDS * 2  # Uncertainty takes longer
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return None, "Uncertainty estimation timed out. Try reducing iterations."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def segment_batch(
    files_list: List[Tuple[str, bytes]],
    threshold: float = 0.5,
    min_area: int = 50
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Segment multiple MRI slices in a batch.
    
    Args:
        files_list: List of (filename, image_bytes) tuples
        threshold: Probability threshold (0.0-1.0)
        min_area: Minimum tumor area in pixels
        
    Returns:
        Tuple of (result_dict, error_message)
    """
    try:
        files = [("files", (name, data, "image/png")) for name, data in files_list]
        params = {
            "threshold": threshold,
            "min_object_size": min_area
        }
        
        response = requests.post(
            f"{API_URL}/segment/batch",
            files=files,
            params=params,
            timeout=API_BATCH_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return None, "Batch segmentation timed out. Try with fewer images."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


# ============================================================================
# Multi-Task Functions
# ============================================================================

def predict_multitask(
    image_bytes: bytes,
    include_gradcam: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Run multi-task prediction (classification + conditional segmentation).
    
    Args:
        image_bytes: Image file as bytes
        include_gradcam: Include Grad-CAM visualization
        
    Returns:
        Tuple of (result_dict, error_message)
        
    Example:
        >>> with open('mri.png', 'rb') as f:
        >>>     result, error = predict_multitask(f.read())
        >>> if result:
        >>>     print(f"Classification: {result['classification']['predicted_label']}")
        >>>     if result['segmentation_computed']:
        >>>         print(f"Segmentation: {result['segmentation']['tumor_area_pixels']} px")
    """
    try:
        files = {"file": ("image.png", image_bytes, "image/png")}
        params = {"include_gradcam": include_gradcam}
        
        response = requests.post(
            f"{API_URL}/predict_multitask",
            files=files,
            params=params,
            timeout=API_TIMEOUT_SECONDS
        )
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 503:
            return None, "Multi-task model not loaded on the server"
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


# ============================================================================
# Patient-Level Analysis Functions
# ============================================================================

def analyze_patient_stack(
    files_list: List[Tuple[str, bytes]],
    patient_id: str,
    threshold: float = 0.5,
    min_area: int = 50,
    slice_thickness_mm: float = 1.0
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Analyze a stack of MRI slices for patient-level tumor detection.
    
    Args:
        files_list: List of (filename, image_bytes) tuples
        patient_id: Patient identifier
        threshold: Probability threshold (0.0-1.0)
        min_area: Minimum tumor area in pixels
        slice_thickness_mm: Slice thickness for volume calculation
        
    Returns:
        Tuple of (result_dict, error_message)
        
    Example:
        >>> files = [(f"slice_{i}.png", data) for i, data in enumerate(slice_data)]
        >>> result, error = analyze_patient_stack(files, "PATIENT_001")
        >>> if result:
        >>>     print(f"Tumor volume: {result['tumor_volume_mm3']} mm³")
    """
    try:
        files = [("files", (name, data, "image/png")) for name, data in files_list]
        data = {
            "patient_id": patient_id,
            "threshold": threshold,
            "min_area": min_area,
            "slice_thickness_mm": slice_thickness_mm
        }
        
        response = requests.post(
            f"{API_URL}/patient/analyze_stack",
            files=files,
            data=data,
            timeout=API_PATIENT_ANALYSIS_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return None, "Patient analysis timed out. Try with fewer slices."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


# ============================================================================
# Export all functions
# ============================================================================

__all__ = [
    'check_api_health',
    'get_model_info',
    'classify_image',
    'classify_batch',
    'segment_image',
    'segment_with_uncertainty',
    'segment_batch',
    'predict_multitask',
    'analyze_patient_stack',
]
