"""
Model loading and management for SliceWise Backend.

This module provides the ModelManager singleton class that handles loading
and managing all model instances (classifier, segmentation, multi-task, etc.).

Extracted from main_v2.py startup_event() (lines 156-252).
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predict import ClassifierPredictor
from src.inference.infer_seg2d import SegmentationPredictor
from src.inference.uncertainty import EnsemblePredictor
from src.inference.multi_task_predictor import MultiTaskPredictor
from src.eval.calibration import TemperatureScaling

from app.backend.config.settings import settings


# ============================================================================
# Model Manager Singleton
# ============================================================================

class ModelManager:
    """
    Singleton class for managing all model instances.
    
    This class handles loading and providing access to all models:
    - Multi-task model (priority)
    - Standalone classifier
    - Standalone segmentation
    - Calibration (temperature scaling)
    - Uncertainty estimation (MC Dropout + TTA)
    
    Extracted and adapted from main_v2.py startup_event() (lines 156-252).
    """
    
    _instance: Optional['ModelManager'] = None
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize model manager (only once due to singleton)."""
        if self._initialized:
            return
        
        # Model instances
        self.classifier: Optional[ClassifierPredictor] = None
        self.segmentation: Optional[SegmentationPredictor] = None
        self.multitask: Optional[MultiTaskPredictor] = None
        self.temperature_scaler: Optional[TemperatureScaling] = None
        self.uncertainty: Optional[EnsemblePredictor] = None
        
        # Status flags
        self.classifier_loaded: bool = False
        self.segmentation_loaded: bool = False
        self.multitask_loaded: bool = False
        self.calibration_loaded: bool = False
        
        self._initialized = True
    
    # ========================================================================
    # Model Loading Methods
    # ========================================================================
    
    def load_multitask(self) -> bool:
        """
        Load multi-task model (PRIORITY).
        
        Extracted from main_v2.py (lines 168-184).
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            multitask_path = settings.paths.multitask_checkpoint
            
            if not multitask_path.exists():
                print(f"[WARN] Multi-task model not found at {multitask_path}")
                return False
            
            self.multitask = MultiTaskPredictor(
                checkpoint_path=str(multitask_path),
                classification_threshold=settings.thresholds.multitask_classification_threshold,
                segmentation_threshold=settings.thresholds.multitask_segmentation_threshold
            )
            self.multitask_loaded = True
            print(f"[OK] Multi-task model loaded from {multitask_path}")
            return True
            
        except Exception as e:
            import traceback
            print(f"[ERROR] Error loading multi-task model: {e}")
            print(f"[ERROR] Traceback:")
            traceback.print_exc()
            return False
    
    def load_classifier(self) -> bool:
        """
        Load standalone classifier model.
        
        Extracted from main_v2.py (lines 186-199).
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            classifier_path = settings.paths.classifier_checkpoint
            
            if not classifier_path.exists():
                print(f"[WARN] Classifier not found at {classifier_path}")
                return False
            
            self.classifier = ClassifierPredictor(
                checkpoint_path=str(classifier_path),
                model_name=settings.model.classifier_architecture
            )
            self.classifier_loaded = True
            print(f"[OK] Classifier loaded from {classifier_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading classifier: {e}")
            return False
    
    def load_calibration(self) -> bool:
        """
        Load temperature scaling calibration.
        
        Requires classifier to be loaded first.
        Extracted from main_v2.py (lines 201-213).
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.classifier_loaded:
            print(f"[WARN] Cannot load calibration without classifier")
            return False
        
        try:
            calibration_path = settings.paths.calibration_checkpoint
            
            if not calibration_path.exists():
                print(f"[WARN] Calibration not found at {calibration_path}")
                return False
            
            self.temperature_scaler = TemperatureScaling()
            checkpoint = torch.load(
                calibration_path,
                map_location=self.classifier.device,
                weights_only=False
            )
            self.temperature_scaler.temperature.data = checkpoint['temperature']
            self.calibration_loaded = True
            
            temp_value = self.temperature_scaler.temperature.item()
            print(f"[OK] Calibration loaded (T={temp_value:.4f})")
            return True
            
        except Exception as e:
            print(f"[WARN] Error loading calibration: {e}")
            return False
    
    def load_segmentation(self) -> bool:
        """
        Load standalone segmentation model.
        
        Also initializes uncertainty estimation.
        Extracted from main_v2.py (lines 215-236).
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            seg_path = settings.paths.segmentation_checkpoint
            
            if not seg_path.exists():
                print(f"[WARN] Segmentation not found at {seg_path}")
                return False
            
            self.segmentation = SegmentationPredictor(
                checkpoint_path=str(seg_path)
            )
            self.segmentation_loaded = True
            print(f"[OK] Segmentation loaded from {seg_path}")
            
            # Initialize uncertainty predictor
            self.uncertainty = EnsemblePredictor(
                model=self.segmentation.model,
                n_mc_samples=settings.uncertainty.mc_dropout_samples,
                use_tta=settings.uncertainty.use_tta,
                device=self.segmentation.device
            )
            print(f"[OK] Uncertainty estimation initialized")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading segmentation: {e}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all available models.
        
        Returns:
            Dictionary with loading status for each model
        """
        print("=" * 80)
        print("SliceWise API - Loading Models...")
        print("=" * 80)
        
        results = {
            'multitask': self.load_multitask(),
            'classifier': self.load_classifier(),
            'calibration': self.load_calibration() if self.classifier_loaded else False,
            'segmentation': self.load_segmentation(),
        }
        
        print("=" * 80)
        print(f"Device: {self.get_device()}")
        print(f"Multi-task: {'[OK]' if self.multitask_loaded else '[NO]'}")
        print(f"Classifier: {'[OK]' if self.classifier_loaded else '[NO]'}")
        print(f"Calibration: {'[OK]' if self.calibration_loaded else '[NO]'}")
        print(f"Segmentation: {'[OK]' if self.segmentation_loaded else '[NO]'}")
        print("=" * 80)
        
        return results
    
    # ========================================================================
    # Status & Info Methods
    # ========================================================================
    
    def get_device(self) -> str:
        """
        Get current device being used.
        
        Extracted from main_v2.py (lines 239-245).
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if self.multitask:
            return str(self.multitask.device)
        elif self.classifier:
            return str(self.classifier.device)
        elif self.segmentation:
            return str(self.segmentation.device)
        return "unknown"
    
    def is_ready(self) -> bool:
        """
        Check if at least one model is loaded.
        
        Extracted from main_v2.py (line 379).
        
        Returns:
            True if any model is loaded
        """
        return (self.multitask_loaded or 
                self.classifier_loaded or 
                self.segmentation_loaded)
    
    def get_status(self) -> str:
        """
        Get overall status.
        
        Returns:
            'healthy' if models loaded, 'no_models_loaded' otherwise
        """
        return "healthy" if self.is_ready() else "no_models_loaded"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Extracted and adapted from main_v2.py (lines 392-450).
        
        Returns:
            Dictionary with model information
        """
        info = {
            'classifier': {},
            'segmentation': {},
            'multitask': {},
            'features': []
        }
        
        # Classifier info
        if self.classifier_loaded:
            info['classifier'] = {
                'architecture': settings.model.classifier_architecture,
                'num_classes': settings.model.num_classes,
                'class_names': settings.model.class_names,
                'input_size': settings.model.input_size,
                'calibrated': self.calibration_loaded
            }
        
        # Segmentation info
        if self.segmentation_loaded:
            info['segmentation'] = {
                'architecture': settings.model.segmentation_architecture,
                'parameters': "31.4M",
                'input_size': settings.model.input_size,
                'output_channels': 1,
                'uncertainty_estimation': True
            }
        
        # Multi-task info
        if self.multitask_loaded:
            model_params = self.multitask.get_model_info()
            info['multitask'] = {
                'architecture': settings.model.multitask_architecture,
                'parameters': model_params['parameters'],
                'input_size': model_params['input_size'],
                'tasks': model_params['tasks'],
                'classification_threshold': model_params['classification_threshold'],
                'segmentation_threshold': model_params['segmentation_threshold'],
                'performance': {
                    'classification_accuracy': 0.9130,
                    'classification_sensitivity': 0.9714,
                    'classification_roc_auc': 0.9184,
                    'segmentation_dice': 0.7650,
                    'segmentation_iou': 0.6401,
                    'combined_metric': 0.8390
                }
            }
        
        # Features
        if self.multitask_loaded:
            info['features'].extend(['multi_task', 'conditional_segmentation', 'unified_inference'])
        if self.classifier_loaded:
            info['features'].extend(['classification', 'grad_cam'])
            if self.calibration_loaded:
                info['features'].append('calibration')
        if self.segmentation_loaded:
            info['features'].extend(['segmentation', 'uncertainty_estimation', 'patient_level_analysis'])
        
        return info
    
    def get_health_info(self) -> Dict[str, Any]:
        """
        Get health check information.
        
        Returns:
            Dictionary with health status
        """
        return {
            'status': self.get_status(),
            'classifier_loaded': self.classifier_loaded,
            'segmentation_loaded': self.segmentation_loaded,
            'calibration_loaded': self.calibration_loaded,
            'multitask_loaded': self.multitask_loaded,
            'device': self.get_device(),
            'timestamp': datetime.now().isoformat()
        }
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ModelManager("
                f"multitask={self.multitask_loaded}, "
                f"classifier={self.classifier_loaded}, "
                f"segmentation={self.segmentation_loaded}, "
                f"device={self.get_device()})")


# ============================================================================
# Global Instance
# ============================================================================

# Create global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Get the global ModelManager instance.
    
    This function is used for dependency injection in FastAPI endpoints.
    
    Returns:
        ModelManager: The global model manager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# ============================================================================
# Testing & Validation
# ============================================================================

if __name__ == "__main__":
    # Test model manager
    print("Testing ModelManager...")
    
    manager = get_model_manager()
    print(f"\n1. Created ModelManager: {manager}")
    
    # Test loading models
    print("\n2. Loading all models:")
    results = manager.load_all_models()
    
    print("\n3. Model loading results:")
    for model_name, loaded in results.items():
        status = "✓" if loaded else "✗"
        print(f"   {status} {model_name}: {loaded}")
    
    print("\n4. Manager status:")
    print(f"   Is ready: {manager.is_ready()}")
    print(f"   Status: {manager.get_status()}")
    print(f"   Device: {manager.get_device()}")
    
    print("\n5. Health info:")
    health = manager.get_health_info()
    for key, value in health.items():
        print(f"   {key}: {value}")
    
    print("\n✅ ModelManager test complete!")
