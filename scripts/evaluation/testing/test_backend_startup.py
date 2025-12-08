"""
Quick diagnostic script to test backend startup and identify errors.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("Testing backend imports and model loading...")
print("=" * 80)

try:
    print("\n1. Testing MultiTaskPredictor import...")
    from src.inference.multi_task_predictor import MultiTaskPredictor
    print("   [OK] Import successful")
    
    print("\n2. Testing MultiTaskPredictor creation...")
    checkpoint_path = project_root / "checkpoints" / "multitask_joint" / "best_model.pth"
    
    if not checkpoint_path.exists():
        print(f"   ✗ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    predictor = MultiTaskPredictor(
        checkpoint_path=str(checkpoint_path),
        classification_threshold=0.3,
        segmentation_threshold=0.5
    )
    print("   [OK] Predictor created successfully")
    
    print("\n3. Testing backend imports...")
    from app.backend.main_v2 import app
    print("   [OK] Backend imports successful")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! Backend should start successfully.")
    print("\nYou can now run:")
    print("  python scripts/run_multitask_demo.py")
    
except Exception as e:
    print(f"\n✗ Error encountered:")
    print(f"  {type(e).__name__}: {e}")
    print("\n" + "=" * 80)
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
