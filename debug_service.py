"""Quick debug script to check service availability"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.backend.services.model_loader import get_model_manager
from app.backend.services.classification_service import ClassificationService
from app.backend.services.segmentation_service import SegmentationService

print("Loading model manager...")
manager = get_model_manager()
manager.load_all_models()

print("\n" + "="*80)
print("Model Manager Status:")
print("="*80)
print(f"Classifier loaded: {manager.classifier_loaded}")
print(f"Segmentation loaded: {manager.segmentation_loaded}")
print(f"Multi-task loaded: {manager.multitask_loaded}")
print(f"Calibration loaded: {manager.calibration_loaded}")

print("\n" + "="*80)
print("Classification Service:")
print("="*80)
cls_service = ClassificationService(manager)
print(f"Is available: {cls_service.is_available()}")
print(f"  - Classifier loaded: {manager.classifier_loaded}")
print(f"  - Multi-task loaded: {manager.multitask_loaded}")
print(f"  - Should be available: {manager.classifier_loaded or manager.multitask_loaded}")

print("\n" + "="*80)
print("Segmentation Service:")
print("="*80)
seg_service = SegmentationService(manager)
print(f"Is available: {seg_service.is_available()}")
print(f"  - Segmentation loaded: {manager.segmentation_loaded}")
print(f"  - Multi-task loaded: {manager.multitask_loaded}")
print(f"  - Should be available: {manager.segmentation_loaded or manager.multitask_loaded}")

print("\n" + "="*80)
