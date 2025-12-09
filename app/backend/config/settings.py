"""
Centralized configuration for SliceWise Backend API.

This module provides all configuration settings for the backend API,
including paths, thresholds, preprocessing parameters, and API settings.
"""

from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel, Field


# ============================================================================
# Path Configuration
# ============================================================================

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class ModelPaths(BaseModel):
    """Model checkpoint paths."""
    
    # Multi-task model (priority)
    multitask_checkpoint: Path = PROJECT_ROOT / "checkpoints" / "multitask_joint" / "best_model.pth"
    
    # Standalone models
    classifier_checkpoint: Path = PROJECT_ROOT / "checkpoints" / "cls" / "best_model.pth"
    segmentation_checkpoint: Path = PROJECT_ROOT / "checkpoints" / "seg" / "best_model.pth"
    
    # Calibration
    calibration_checkpoint: Path = PROJECT_ROOT / "checkpoints" / "cls" / "temperature_scaler.pth"
    
    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# Model Configuration
# ============================================================================

class ModelThresholds(BaseModel):
    """Model prediction thresholds."""
    
    # Classification thresholds
    classification_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Segmentation thresholds
    segmentation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Multi-task thresholds
    multitask_classification_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    multitask_segmentation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    
    # Input size
    input_size: Tuple[int, int] = (256, 256)
    
    # Classification
    num_classes: int = 2
    class_names: List[str] = ["No Tumor", "Tumor"]
    
    # Model names
    classifier_architecture: str = "efficientnet"
    segmentation_architecture: str = "unet2d"
    multitask_architecture: str = "multitask_unet"


# ============================================================================
# Preprocessing Configuration
# ============================================================================

class PreprocessingConfig(BaseModel):
    """Image preprocessing parameters."""
    
    # Normalization methods
    classification_normalization: str = "minmax"  # minmax, zscore
    segmentation_normalization: str = "zscore"    # CRITICAL: must match training!
    
    # Z-score parameters (for segmentation)
    apply_zscore: bool = True
    
    # Min-max parameters (for classification)
    min_val: float = 0.0
    max_val: float = 1.0
    
    # Image size
    target_size: Tuple[int, int] = (256, 256)


# ============================================================================
# Postprocessing Configuration
# ============================================================================

class PostprocessingConfig(BaseModel):
    """Segmentation postprocessing parameters."""
    
    # Default parameters
    default_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    default_min_object_size: int = Field(default=50, ge=0)
    
    # Morphological operations
    morphology_op: str = "close"  # none, open, close, dilate, erode
    morphology_kernel: int = Field(default=3, ge=1)
    
    # Hole filling
    fill_holes_size: int = Field(default=0, ge=0)  # 0 = disabled
    
    # Connected components
    connectivity: int = Field(default=2, ge=1, le=2)  # 1=4-connected, 2=8-connected


# ============================================================================
# Uncertainty Configuration
# ============================================================================

class UncertaintyConfig(BaseModel):
    """Uncertainty estimation parameters."""
    
    # MC Dropout
    mc_dropout_samples: int = Field(default=10, ge=1, le=50)
    
    # Test-Time Augmentation
    use_tta: bool = True
    tta_augmentations: int = 6
    
    # Ensemble
    use_ensemble: bool = True


# ============================================================================
# API Configuration
# ============================================================================

class APIConfig(BaseModel):
    """FastAPI server configuration."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"
    
    # API metadata
    title: str = "SliceWise API"
    description: str = "Comprehensive Brain Tumor Detection API with Classification, Segmentation, and Uncertainty Estimation"
    version: str = "2.0.0"
    
    # CORS settings
    cors_origins: List[str] = ["*"]  # In production, specify allowed origins
    cors_credentials: bool = True
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # Timeouts (seconds)
    request_timeout: int = 300
    startup_timeout: int = 60


# ============================================================================
# Batch Processing Configuration
# ============================================================================

class BatchLimits(BaseModel):
    """Batch processing limits."""
    
    # Maximum batch sizes
    max_classification_batch: int = 100
    max_segmentation_batch: int = 100
    max_patient_stack_size: int = 500
    
    # Processing timeouts (seconds)
    batch_timeout: int = 600
    patient_analysis_timeout: int = 1200


# ============================================================================
# Visualization Configuration
# ============================================================================

class VisualizationConfig(BaseModel):
    """Visualization parameters."""
    
    # Overlay settings
    overlay_alpha: float = Field(default=0.4, ge=0.0, le=1.0)
    tumor_color: Tuple[int, int, int] = (255, 0, 0)  # Red
    
    # Grad-CAM settings
    gradcam_alpha: float = Field(default=0.4, ge=0.0, le=1.0)
    gradcam_colormap: str = "jet"
    
    # Uncertainty map settings
    uncertainty_colormap: str = "viridis"
    
    # Image format
    output_format: str = "PNG"
    output_quality: int = 95


# ============================================================================
# Patient Analysis Configuration
# ============================================================================

class PatientAnalysisConfig(BaseModel):
    """Patient-level analysis parameters."""
    
    # Volume estimation
    default_slice_thickness_mm: float = 1.0
    default_pixel_spacing_mm: float = 1.0
    
    # Aggregation
    min_slices_for_analysis: int = 1
    tumor_detection_threshold: float = 0.5  # Fraction of slices with tumor


# ============================================================================
# Logging Configuration
# ============================================================================

class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    # Log levels
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Log files
    enable_file_logging: bool = False
    log_file: Path = PROJECT_ROOT / "logs" / "backend.log"
    
    # Structured logging
    enable_json_logging: bool = False


# ============================================================================
# Main Settings Class
# ============================================================================

class Settings(BaseModel):
    """
    Main settings class combining all configuration.
    
    This class provides a single point of access to all backend configuration.
    """
    
    # Sub-configurations
    paths: ModelPaths = ModelPaths()
    thresholds: ModelThresholds = ModelThresholds()
    model: ModelConfig = ModelConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    postprocessing: PostprocessingConfig = PostprocessingConfig()
    uncertainty: UncertaintyConfig = UncertaintyConfig()
    api: APIConfig = APIConfig()
    batch: BatchLimits = BatchLimits()
    visualization: VisualizationConfig = VisualizationConfig()
    patient: PatientAnalysisConfig = PatientAnalysisConfig()
    logging: LoggingConfig = LoggingConfig()
    
    class Config:
        arbitrary_types_allowed = True
    
    def __repr__(self) -> str:
        return f"Settings(api_version={self.api.version}, project_root={PROJECT_ROOT})"


# ============================================================================
# Global Settings Instance
# ============================================================================

# Create global settings instance
settings = Settings()


# ============================================================================
# Helper Functions
# ============================================================================

def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    This function is used for dependency injection in FastAPI endpoints.
    
    Returns:
        Settings: The global settings instance.
    """
    return settings


def print_settings() -> None:
    """Print current settings (for debugging)."""
    print("=" * 80)
    print("SliceWise Backend Configuration")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"API Version: {settings.api.version}")
    print(f"API Host: {settings.api.host}:{settings.api.port}")
    print(f"\nModel Paths:")
    print(f"  Multi-task: {settings.paths.multitask_checkpoint}")
    print(f"  Classifier: {settings.paths.classifier_checkpoint}")
    print(f"  Segmentation: {settings.paths.segmentation_checkpoint}")
    print(f"\nThresholds:")
    print(f"  Classification: {settings.thresholds.classification_threshold}")
    print(f"  Segmentation: {settings.thresholds.segmentation_threshold}")
    print(f"\nBatch Limits:")
    print(f"  Classification: {settings.batch.max_classification_batch}")
    print(f"  Segmentation: {settings.batch.max_segmentation_batch}")
    print("=" * 80)


if __name__ == "__main__":
    # Test configuration loading
    print_settings()
