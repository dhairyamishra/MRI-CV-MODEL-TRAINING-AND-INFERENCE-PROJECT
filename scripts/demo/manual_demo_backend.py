"""
Helper script to run the SliceWise Phase 6 demo backend.

This script starts the FastAPI backend server with the comprehensive
Phase 6 API that includes classification, segmentation, uncertainty estimation,
and patient-level analysis.

Usage:
    python scripts/run_demo_backend.py [--port PORT] [--host HOST] [--reload]
"""

import sys
import argparse
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_models_exist():
    """Check if required model checkpoints exist."""
    checkpoints_dir = project_root / "checkpoints"
    
    classifier_path = checkpoints_dir / "cls" / "best_model.pth"
    segmentation_path = checkpoints_dir / "seg" / "best_model.pth"
    calibration_path = checkpoints_dir / "cls" / "temperature_scaler.pth"
    
    models_status = {
        "classifier": classifier_path.exists(),
        "segmentation": segmentation_path.exists(),
        "calibration": calibration_path.exists()
    }
    
    print("=" * 80)
    print("Model Checkpoint Status:")
    print("=" * 80)
    print(f"Classifier:    {'[OK] Found' if models_status['classifier'] else '✗ Not found'} ({classifier_path})")
    print(f"Segmentation:  {'[OK] Found' if models_status['segmentation'] else '✗ Not found'} ({segmentation_path})")
    print(f"Calibration:   {'[OK] Found' if models_status['calibration'] else '✗ Not found'} ({calibration_path})")
    print("=" * 80)
    
    if not any(models_status.values()):
        print("\n⚠️  WARNING: No model checkpoints found!")
        print("The API will start but predictions will fail.")
        print("\nTo train models, run:")
        print("  - Classification: python scripts/train_classifier.py")
        print("  - Segmentation:   python scripts/train_segmentation.py")
        print("  - Calibration:    python scripts/calibrate_classifier.py")
        print()
    
    return models_status


def main():
    """Main function to run the backend."""
    parser = argparse.ArgumentParser(description="Run SliceWise Phase 6 Demo Backend")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--no-check", action="store_true", help="Skip model checkpoint check")
    
    args = parser.parse_args()
    
    # Check if models exist
    if not args.no_check:
        check_models_exist()
        print()
    
    # Path to backend (using new modular main.py)
    backend_path = project_root / "app" / "backend" / "main.py"
    
    if not backend_path.exists():
        print(f"✗ Backend file not found: {backend_path}")
        print("Please ensure main.py exists in app/backend/")
        sys.exit(1)
    
    print("=" * 80)
    print("Starting SliceWise Phase 6 Demo Backend")
    print("=" * 80)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"API Docs: http://localhost:{args.port}/docs")
    print(f"Health Check: http://localhost:{args.port}/healthz")
    print("=" * 80)
    print("\nPress Ctrl+C to stop the server\n")
    
    # Build uvicorn command
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.backend.main:app",  # Updated from main_v2 to main
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", "info"
    ]
    
    if args.reload:
        cmd.append("--reload")
    
    try:
        # Run uvicorn
        subprocess.run(cmd, cwd=str(project_root), check=True)
    except KeyboardInterrupt:
        print("\n\n[OK] Server stopped")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
