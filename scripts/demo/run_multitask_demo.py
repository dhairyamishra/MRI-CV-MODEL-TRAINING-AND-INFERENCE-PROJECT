"""
Multi-Task Demo Launcher

This script launches both the FastAPI backend and Streamlit frontend
for the multi-task brain tumor detection demo.

Features:
- Checks if multi-task model checkpoint exists
- Starts backend server with health check
- Starts frontend server
- Opens browser automatically
- Handles graceful shutdown

Usage:
    python scripts/run_multitask_demo.py
"""

import sys
import subprocess
import time
import requests
from pathlib import Path
import webbrowser
import signal
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration
BACKEND_HOST = "localhost"
BACKEND_PORT = 8000
FRONTEND_PORT = 8501
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

# Paths
BACKEND_SCRIPT = project_root / "app" / "backend" / "main_v2.py"
FRONTEND_SCRIPT = project_root / "app" / "frontend" / "app_v2.py"
MULTITASK_CHECKPOINT = project_root / "checkpoints" / "multitask_joint" / "best_model.pth"

# Process handles
backend_process = None
frontend_process = None


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("🎯 SliceWise Multi-Task Demo Launcher")
    print("=" * 80)
    print()
    print("This demo showcases the unified multi-task model that performs")
    print("both classification and segmentation in a single forward pass.")
    print()
    print("Benefits:")
    print("  🚀 ~40% faster inference")
    print("  💾 9.4% fewer parameters")
    print("  🎯 Conditional segmentation (smart resource usage)")
    print("  📊 91.3% accuracy, 97.1% sensitivity")
    print()
    print("=" * 80)
    print()


def check_checkpoint():
    """Check if multi-task model checkpoint exists."""
    print("📁 Checking for multi-task model checkpoint...")
    
    if MULTITASK_CHECKPOINT.exists():
        size_mb = MULTITASK_CHECKPOINT.stat().st_size / (1024 * 1024)
        print(f"✓ Checkpoint found: {MULTITASK_CHECKPOINT}")
        print(f"  Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ Checkpoint not found: {MULTITASK_CHECKPOINT}")
        print()
        print("Please ensure you have trained the multi-task model:")
        print("  python scripts/train_multitask_joint.py")
        print()
        return False


def check_port_available(port):
    """Check if a port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False


def wait_for_backend(timeout=30):
    """Wait for backend to be ready."""
    print(f"⏳ Waiting for backend to start (timeout: {timeout}s)...")
    
    start_time = time.time()
    dots_printed = 0
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BACKEND_URL}/healthz", timeout=2)
            if response.status_code == 200:
                data = response.json()
                print(f"\n✓ Backend is ready!")
                print(f"  Status: {data['status']}")
                print(f"  Multi-task loaded: {data['multitask_loaded']}")
                print(f"  Device: {data['device']}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
        dots_printed += 1
        
        # Print elapsed time every 10 seconds
        if dots_printed % 10 == 0:
            elapsed = int(time.time() - start_time)
            print(f" ({elapsed}s)", end="", flush=True)
    
    print()
    print("✗ Backend failed to start within timeout")
    print("\n💡 Tip: The backend may still be loading. Check the backend process output for errors.")
    return False


def start_backend():
    """Start FastAPI backend server."""
    global backend_process
    
    print()
    print("🚀 Starting backend server...")
    print(f"  URL: {BACKEND_URL}")
    print(f"  Script: {BACKEND_SCRIPT}")
    
    # Check if port is available
    if not check_port_available(BACKEND_PORT):
        print(f"✗ Port {BACKEND_PORT} is already in use")
        print(f"  Please stop the existing process or change the port")
        return False
    
    # Start backend
    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "app.backend.main_v2:app",
        "--host", BACKEND_HOST,
        "--port", str(BACKEND_PORT),
        "--reload"
    ]
    
    try:
        # Create log file for backend output
        log_file = project_root / "backend_startup.log"
        log_handle = open(log_file, 'w')
        
        backend_process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        print(f"  Backend process started (PID: {backend_process.pid})")
        print(f"  Logs: {log_file}")
        
        # Wait for backend to be ready
        if wait_for_backend():
            print()
            log_handle.close()
            return True
        else:
            # Backend failed to start, show the logs
            log_handle.close()
            print("\n❌ Backend startup failed. Last 20 lines of log:")
            print("=" * 80)
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-20:]:
                        print(line.rstrip())
            except Exception as e:
                print(f"Could not read log file: {e}")
            print("=" * 80)
            return False
    
    except Exception as e:
        print(f"✗ Failed to start backend: {e}")
        return False


def start_frontend():
    """Start Streamlit frontend server."""
    global frontend_process
    
    print()
    print("🎨 Starting frontend server...")
    print(f"  URL: {FRONTEND_URL}")
    print(f"  Script: {FRONTEND_SCRIPT}")
    
    # Check if port is available
    if not check_port_available(FRONTEND_PORT):
        print(f"✗ Port {FRONTEND_PORT} is already in use")
        print(f"  Please stop the existing process or change the port")
        return False
    
    # Start frontend
    cmd = [
        sys.executable,
        "-m", "streamlit",
        "run",
        str(FRONTEND_SCRIPT),
        "--server.port", str(FRONTEND_PORT),
        "--server.headless", "true"
    ]
    
    try:
        frontend_process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Wait a bit for frontend to start
        print("⏳ Waiting for frontend to start...")
        time.sleep(3)
        
        print("✓ Frontend started!")
        print()
        return True
    
    except Exception as e:
        print(f"✗ Failed to start frontend: {e}")
        return False


def open_browser():
    """Open browser to frontend URL."""
    print("🌐 Opening browser...")
    try:
        webbrowser.open(FRONTEND_URL)
        print(f"✓ Browser opened to {FRONTEND_URL}")
    except Exception as e:
        print(f"⚠ Could not open browser automatically: {e}")
        print(f"  Please open {FRONTEND_URL} manually")


def cleanup():
    """Cleanup and shutdown servers."""
    global backend_process, frontend_process
    
    print()
    print("=" * 80)
    print("🛑 Shutting down servers...")
    print("=" * 80)
    
    if frontend_process:
        print("Stopping frontend...")
        frontend_process.terminate()
        try:
            frontend_process.wait(timeout=5)
            print("✓ Frontend stopped")
        except subprocess.TimeoutExpired:
            frontend_process.kill()
            print("✓ Frontend killed")
    
    if backend_process:
        print("Stopping backend...")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
            print("✓ Backend stopped")
        except subprocess.TimeoutExpired:
            backend_process.kill()
            print("✓ Backend killed")
    
    print()
    print("👋 Demo stopped. Thank you for using SliceWise!")
    print("=" * 80)


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print()
    print("Received interrupt signal...")
    cleanup()
    sys.exit(0)


def main():
    """Main function."""
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Print banner
    print_banner()
    
    # Check checkpoint
    if not check_checkpoint():
        sys.exit(1)
    
    # Start backend
    if not start_backend():
        print()
        print("✗ Failed to start backend. Exiting.")
        cleanup()
        sys.exit(1)
    
    # Start frontend
    if not start_frontend():
        print()
        print("✗ Failed to start frontend. Exiting.")
        cleanup()
        sys.exit(1)
    
    # Open browser
    open_browser()
    
    # Print instructions
    print()
    print("=" * 80)
    print("✅ Demo is running!")
    print("=" * 80)
    print()
    print("📍 URLs:")
    print(f"  Frontend: {FRONTEND_URL}")
    print(f"  Backend:  {BACKEND_URL}")
    print(f"  API Docs: {BACKEND_URL}/docs")
    print()
    print("🎯 Features:")
    print("  • Multi-Task tab: Unified classification + segmentation")
    print("  • Conditional segmentation (only if tumor prob >= 30%)")
    print("  • Grad-CAM visualization for interpretability")
    print("  • Performance metrics display")
    print("  • Clinical recommendations")
    print()
    print("⌨️  Press Ctrl+C to stop the demo")
    print("=" * 80)
    print()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process and backend_process.poll() is not None:
                print("⚠ Backend process stopped unexpectedly")
                break
            
            if frontend_process and frontend_process.poll() is not None:
                print("⚠ Frontend process stopped unexpectedly")
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        cleanup()


if __name__ == "__main__":
    main()
