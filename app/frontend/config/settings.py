"""
Configuration settings for SliceWise Frontend.

This module contains all constants, configuration values, and settings
used throughout the Streamlit application.
"""

from typing import Dict, List

# ============================================================================
# API Configuration
# ============================================================================

API_URL = "http://localhost:8000"
API_TIMEOUT_SECONDS = 30
API_HEALTH_CHECK_TIMEOUT = 2
API_BATCH_TIMEOUT = 60
API_PATIENT_ANALYSIS_TIMEOUT = 120

# ============================================================================
# Color Palette
# ============================================================================

class Colors:
    """Application color scheme."""
    
    # Primary colors
    PRIMARY_BLUE = "#1f77b4"
    SUCCESS_GREEN = "#28a745"
    DANGER_RED = "#dc3545"
    WARNING_YELLOW = "#ffc107"
    INFO_BLUE = "#17a2b8"
    
    # Text colors
    TEXT_DARK = "#333333"
    TEXT_MEDIUM = "#666666"
    TEXT_LIGHT = "#999999"
    
    # Background colors
    BG_LIGHT = "#f0f2f6"
    BG_SUCCESS = "#d4edda"
    BG_WARNING = "#fff3cd"  # Fixed from #000000!
    BG_INFO = "#d1ecf1"
    BG_DANGER = "#f8d7da"
    
    # Chart colors
    CHART_TUMOR = "#dc3545"
    CHART_NO_TUMOR = "#28a745"
    CHART_NEUTRAL = "#1f77b4"


# ============================================================================
# Model Configuration
# ============================================================================

class ModelConfig:
    """Model-related configuration."""
    
    # Classification thresholds
    CLASSIFICATION_THRESHOLD_LOW = 0.3
    CLASSIFICATION_THRESHOLD_MEDIUM = 0.5
    CLASSIFICATION_THRESHOLD_HIGH = 0.7
    
    # Segmentation thresholds
    SEGMENTATION_THRESHOLD_DEFAULT = 0.5
    SEGMENTATION_THRESHOLD_MIN = 0.0
    SEGMENTATION_THRESHOLD_MAX = 1.0
    SEGMENTATION_THRESHOLD_STEP = 0.05
    
    # Multi-task thresholds
    MULTITASK_CLASSIFICATION_THRESHOLD = 0.3
    MULTITASK_SEGMENTATION_THRESHOLD = 0.5
    
    # Post-processing
    MIN_TUMOR_AREA_PIXELS = 50
    MIN_TUMOR_AREA_MIN = 0
    MIN_TUMOR_AREA_MAX = 500
    
    # Uncertainty estimation
    MC_DROPOUT_ITERATIONS_DEFAULT = 10
    MC_DROPOUT_ITERATIONS_MIN = 1
    MC_DROPOUT_ITERATIONS_MAX = 50
    USE_TTA_DEFAULT = True


# ============================================================================
# UI Configuration
# ============================================================================

class UIConfig:
    """UI-related configuration."""
    
    # Image display widths (pixels)
    IMAGE_WIDTH_SMALL = 200
    IMAGE_WIDTH_MEDIUM = 250
    IMAGE_WIDTH_LARGE = 300
    IMAGE_WIDTH_XLARGE = 400
    
    # Chart sizes
    CHART_WIDTH_SMALL = 6
    CHART_HEIGHT_SMALL = 2
    CHART_WIDTH_MEDIUM = 8
    CHART_HEIGHT_MEDIUM = 3
    CHART_WIDTH_LARGE = 12
    CHART_HEIGHT_LARGE = 4
    
    # File upload
    MAX_BATCH_SIZE = 100
    ALLOWED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp"]
    MAX_FILE_SIZE_MB = 10
    
    # Patient analysis
    SLICE_THICKNESS_DEFAULT = 1.0
    SLICE_THICKNESS_MIN = 0.1
    SLICE_THICKNESS_MAX = 10.0
    SLICE_THICKNESS_STEP = 0.1


# ============================================================================
# Application Metadata
# ============================================================================

class AppMetadata:
    """Application metadata and information."""
    
    APP_TITLE = "SliceWise v2 - Brain Tumor Detection"
    APP_ICON = "🧠"
    APP_VERSION = "2.0.0"
    APP_PHASE = "Phase 6 Complete"
    
    # About information
    TOTAL_LINES_OF_CODE = "~12,000+"
    
    # Features
    FEATURES = [
        "🔍 Classification with Grad-CAM",
        "🎯 Probability calibration",
        "🎨 Tumor segmentation",
        "📊 Uncertainty estimation",
        "👤 Patient-level analysis",
        "📦 Batch processing"
    ]
    
    # Model class names
    CLASS_NAMES = ["No Tumor", "Tumor"]


# ============================================================================
# Medical Disclaimer
# ============================================================================

MEDICAL_DISCLAIMER = """
⚠️ **Medical Disclaimer:** This tool is for research and educational purposes only.
It is NOT a medical device and should NOT be used for clinical diagnosis.
Always consult qualified healthcare professionals for medical advice.
"""


# ============================================================================
# Clinical Interpretation Guidelines
# ============================================================================

class ClinicalGuidelines:
    """Clinical interpretation guidelines."""
    
    HIGH_CONFIDENCE_TUMOR = """
    **High Confidence Tumor Detection**
    - Strong evidence of tumor presence
    - Recommend immediate radiologist review
    - Consider additional imaging (contrast-enhanced MRI)
    """
    
    MODERATE_CONFIDENCE_TUMOR = """
    **Moderate Confidence Detection**
    - Uncertain tumor presence
    - Recommend radiologist review and follow-up imaging
    - Consider patient history and symptoms
    """
    
    LOW_PROBABILITY_TUMOR = """
    **Low Probability of Tumor**
    - Low probability of tumor
    - Routine follow-up recommended
    - Monitor for changes in symptoms
    """
    
    NO_TUMOR_HIGH_CONFIDENCE = """
    The model found **no clear signs of tumor** in this MRI slice.
    However, this does not rule out the presence of abnormalities.
    """
    
    NO_TUMOR_MODERATE_CONFIDENCE = """
    The model suggests **no tumor**, but with moderate confidence.
    Further examination may be warranted.
    """


# ============================================================================
# Export all settings
# ============================================================================

__all__ = [
    'API_URL',
    'API_TIMEOUT_SECONDS',
    'API_HEALTH_CHECK_TIMEOUT',
    'API_BATCH_TIMEOUT',
    'API_PATIENT_ANALYSIS_TIMEOUT',
    'Colors',
    'ModelConfig',
    'UIConfig',
    'AppMetadata',
    'MEDICAL_DISCLAIMER',
    'ClinicalGuidelines',
]
