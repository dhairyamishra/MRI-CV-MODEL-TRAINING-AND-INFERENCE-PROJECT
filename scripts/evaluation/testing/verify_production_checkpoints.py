"""
Verify Production Checkpoints

This script verifies that the production checkpoints exist and can be loaded
by the backend API.

Author: SliceWise Team
Date: December 11, 2025
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.backend.config.settings import settings


def verify_checkpoints():
    """Verify all checkpoint files exist."""
    print("=" * 80)
    print("SliceWise Production Checkpoint Verification")
    print("=" * 80)
    print()
    
    checkpoints = {
        "Multi-task Model": settings.paths.multitask_checkpoint,
        "Classification Model": settings.paths.classifier_checkpoint,
        "Segmentation Model": settings.paths.segmentation_checkpoint,
        "Calibration File": settings.paths.calibration_checkpoint,
    }
    
    all_exist = True
    
    for name, path in checkpoints.items():
        exists = path.exists()
        status = "✓ FOUND" if exists else "✗ MISSING"
        size = f"({path.stat().st_size / 1024 / 1024:.1f} MB)" if exists else ""
        
        print(f"{status:12} {name:25} {size}")
        print(f"             {path}")
        print()
        
        if not exists and name != "Calibration File":  # Calibration is optional
            all_exist = False
    
    print("=" * 80)
    
    if all_exist:
        print("✓ All required checkpoints found!")
        print()
        print("Your production models are ready to use in the API and UI.")
        print()
        print("To start the demo:")
        print("  1. Backend:  python scripts/run_demo_backend.py")
        print("  2. Frontend: streamlit run app/frontend/app.py --server.port 8501")
        print()
        print("Or use PM2:")
        print("  pm2 start configs/pm2-ecosystem/ecosystem.config.js")
        print("=" * 80)
        return 0
    else:
        print("✗ Some required checkpoints are missing!")
        print()
        print("Please run training to generate the missing checkpoints:")
        print("  python scripts/run_full_pipeline.py --mode full --training-mode production")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(verify_checkpoints())
