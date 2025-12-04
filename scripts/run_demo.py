"""
Unified script to run both SliceWise Phase 6 backend and frontend together.

This script starts both the FastAPI backend and Streamlit frontend
in separate processes for easy demo deployment.

Usage:
    python scripts/run_demo.py [--backend-port PORT] [--frontend-port PORT]
"""

import sys
import argparse
from pathlib import Path
import subprocess
import time
import signal
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_models_exist():
    """Check if required model checkpoints exist."""
    checkpoints_dir = project_root / "checkpoints"
    
    classifier_path = checkpoints_dir / "cls" / "best_model.pth"
    segmentation_path = checkpoints_dir / "seg" / "best_model.pth"
    
    models_status = {
        "classifier": classifier_path.exists(),
        "segmentation": segmentation_path.exists()
    }
    
    if not any(models_status.values()):
        print("\n⚠️  WARNING: No model checkpoints found!")
        print("The demo will start but predictions will fail.")
        print("\nTo train models, run:")
        print("  - Classification: python scripts/train_classifier.py")
        print("  - Segmentation:   python scripts/train_segmentation.py")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    
    return models_status


def main():
    """Main function to run both backend and frontend."""
    parser = argparse.ArgumentParser(description="Run SliceWise Phase 6 Complete Demo")
    parser.add_argument("--backend-port", type=int, default=8000, help="Backend port (default: 8000)")
    parser.add_argument("--frontend-port", type=int, default=8501, help="Frontend port (default: 8501)")
    parser.add_argument("--no-check", action="store_true", help="Skip model checkpoint check")
    
    args = parser.parse_args()
    
    # Check if models exist
    if not args.no_check:
        check_models_exist()
    
    print("=" * 80)
    print("SliceWise Phase 6 - Complete Demo Application")
    print("=" * 80)
    print("\nStarting both backend and frontend servers...")
    print(f"\n📡 Backend API:  http://localhost:{args.backend_port}")
    print(f"   API Docs:     http://localhost:{args.backend_port}/docs")
    print(f"\n🖥️  Frontend UI:  http://localhost:{args.frontend_port}")
    print("\n" + "=" * 80)
    print("Press Ctrl+C to stop both servers")
    print("=" * 80 + "\n")
    
    # Start backend
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.backend.main_v2:app",
        "--host", "0.0.0.0",
        "--port", str(args.backend_port),
        "--log-level", "info"
    ]
    
    print("Starting backend...")
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=str(project_root)
    )
    
    # Wait for backend to start
    print("Waiting for backend to initialize...")
    time.sleep(3)
    
    # Start frontend
    frontend_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(project_root / "app" / "frontend" / "app_v2.py"),
        "--server.port", str(args.frontend_port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    print("Starting frontend...")
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=str(project_root)
    )
    
    print("\n✓ Both servers started successfully!")
    print(f"\n🌐 Open your browser to: http://localhost:{args.frontend_port}")
    print("\n" + "=" * 80)
    print("Server logs will appear below. Press Ctrl+C to stop.")
    print("=" * 80 + "\n")
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nShutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        
        # Wait for processes to terminate
        try:
            backend_process.wait(timeout=5)
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
            frontend_process.kill()
        
        print("✓ Servers stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Keep the script running and show logs
        while True:
            # Check if processes are still running
            backend_status = backend_process.poll()
            frontend_status = frontend_process.poll()
            
            if backend_status is not None:
                print(f"\n✗ Backend process exited with code {backend_status}")
                frontend_process.terminate()
                break
            
            if frontend_status is not None:
                print(f"\n✗ Frontend process exited with code {frontend_status}")
                backend_process.terminate()
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
