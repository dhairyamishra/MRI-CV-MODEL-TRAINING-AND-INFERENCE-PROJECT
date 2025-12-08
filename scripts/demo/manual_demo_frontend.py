"""
Helper script to run the SliceWise Phase 6 demo frontend.

This script starts the Streamlit frontend with the comprehensive
Phase 6 UI that includes classification, segmentation, batch processing,
and patient-level analysis.

Usage:
    python scripts/run_demo_frontend.py [--port PORT]
"""

import sys
import argparse
from pathlib import Path
import subprocess
import requests
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_backend_running(api_url="http://localhost:8000", timeout=2):
    """Check if the backend API is running."""
    try:
        response = requests.get(f"{api_url}/healthz", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def main():
    """Main function to run the frontend."""
    parser = argparse.ArgumentParser(description="Run SliceWise Phase 6 Demo Frontend")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on (default: 8501)")
    parser.add_argument("--backend-url", type=str, default="http://localhost:8000", help="Backend API URL")
    parser.add_argument("--no-check", action="store_true", help="Skip backend health check")
    
    args = parser.parse_args()
    
    # Path to frontend
    frontend_path = project_root / "app" / "frontend" / "app_v2.py"
    
    if not frontend_path.exists():
        print(f"✗ Frontend file not found: {frontend_path}")
        print("Please ensure app_v2.py exists in app/frontend/")
        sys.exit(1)
    
    # Check if backend is running
    if not args.no_check:
        print("Checking backend API status...")
        backend_running = check_backend_running(args.backend_url)
        
        if backend_running:
            print(f"[OK] Backend API is running at {args.backend_url}")
        else:
            print(f"⚠️  WARNING: Backend API is not responding at {args.backend_url}")
            print("\nThe frontend will start, but you need to start the backend first:")
            print("  python scripts/run_demo_backend.py")
            print("\nContinuing in 3 seconds...")
            time.sleep(3)
    
    print("\n" + "=" * 80)
    print("Starting SliceWise Phase 6 Demo Frontend")
    print("=" * 80)
    print(f"Frontend URL: http://localhost:{args.port}")
    print(f"Backend API: {args.backend_url}")
    print("=" * 80)
    print("\nPress Ctrl+C to stop the server\n")
    
    # Build streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(frontend_path),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        # Run streamlit
        subprocess.run(cmd, cwd=str(project_root), check=True)
    except KeyboardInterrupt:
        print("\n\n[OK] Frontend stopped")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running frontend: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
