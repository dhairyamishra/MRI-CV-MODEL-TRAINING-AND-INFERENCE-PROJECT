#!/usr/bin/env python3
"""
Run demo with production models from seg_production and cls_production.

This script:
1. Copies production models to the expected demo locations
2. Starts the backend and frontend servers
3. Opens browser to the UI

Usage:
    python scripts/run_demo_with_production_models.py
"""

import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_models():
    """Copy production models to demo locations."""
    print("=" * 70)
    print("Setting up models for demo...")
    print("=" * 70)
    
    checkpoints_dir = project_root / "checkpoints"
    
    # Setup segmentation model
    seg_production = checkpoints_dir / "seg_production" / "best_model.pth"
    seg_demo = checkpoints_dir / "seg" / "best_model.pth"
    
    if seg_production.exists():
        seg_demo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seg_production, seg_demo)
        print(f"[OK] Copied segmentation model: {seg_production.name}")
        print(f"  → {seg_demo}")
    else:
        print(f"⚠️  Segmentation model not found: {seg_production}")
    
    # Setup classification model (if exists)
    cls_production = checkpoints_dir / "cls_production" / "best_model.pth"
    cls_demo = checkpoints_dir / "cls" / "best_model.pth"
    
    if cls_production.exists():
        cls_demo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cls_production, cls_demo)
        print(f"[OK] Copied classification model: {cls_production.name}")
        print(f"  → {cls_demo}")
    else:
        print(f"ℹ️  Classification model not found (optional): {cls_production}")
    
    print()


def main():
    """Run the demo."""
    print("\n" + "=" * 70)
    print("SliceWise Demo - Production Models")
    print("=" * 70)
    print()
    
    # Setup models
    setup_models()
    
    # Run the main demo script
    print("Starting demo servers...")
    print("=" * 70)
    print()
    
    demo_script = project_root / "scripts" / "run_demo.py"
    
    try:
        subprocess.run([sys.executable, str(demo_script)], check=True)
    except KeyboardInterrupt:
        print("\n\nShutting down demo...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
