"""
Diagnostic script to test multi-task model loading for backend.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("MULTI-TASK MODEL LOADING DIAGNOSTIC")
print("=" * 80)

# Check if checkpoint exists
checkpoint_path = project_root / "checkpoints" / "multitask_joint" / "best_model.pth"
print(f"\n1. Checking checkpoint path...")
print(f"   Path: {checkpoint_path}")
print(f"   Exists: {checkpoint_path.exists()}")

if not checkpoint_path.exists():
    print(f"\n❌ ERROR: Checkpoint file not found!")
    sys.exit(1)

# Check if config exists
config_path = project_root / "checkpoints" / "multitask_joint" / "model_config.json"
print(f"\n2. Checking model config...")
print(f"   Path: {config_path}")
print(f"   Exists: {config_path.exists()}")

# Try to import MultiTaskPredictor
print(f"\n3. Importing MultiTaskPredictor...")
try:
    from src.inference.multi_task_predictor import MultiTaskPredictor
    print(f"   ✓ Import successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try to load the model
print(f"\n4. Loading multi-task model...")
try:
    predictor = MultiTaskPredictor(
        checkpoint_path=str(checkpoint_path),
        classification_threshold=0.3,
        segmentation_threshold=0.5
    )
    print(f"   ✓ Model loaded successfully!")
    print(f"   Device: {predictor.device}")
    print(f"   Parameters: {predictor.model.get_num_params()}")
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try a dummy prediction
print(f"\n5. Testing dummy prediction...")
try:
    import torch
    import numpy as np
    
    # Create dummy image (256x256)
    dummy_image = np.random.rand(256, 256).astype(np.float32)
    
    result = predictor.predict_single(dummy_image)
    print(f"   ✓ Prediction successful!")
    print(f"   Classification: {result['classification']['predicted_label']}")
    print(f"   Confidence: {result['classification']['confidence']:.4f}")
    print(f"   Tumor probability: {result['classification']['tumor_probability']:.4f}")
    print(f"   Has segmentation: {result['segmentation'] is not None}")
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED - Model is ready for backend!")
print("=" * 80)
