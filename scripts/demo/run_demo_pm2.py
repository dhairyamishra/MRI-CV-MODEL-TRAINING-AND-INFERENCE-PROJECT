"""
PM2-Based Multi-Task Demo Launcher

This script uses PM2 to manage both the FastAPI backend and Streamlit frontend
for the multi-task brain tumor detection demo. PM2 provides better process
management, automatic restarts, and proper cleanup on Windows.

Features:
- Uses PM2 for robust process management
- Checks if multi-task model checkpoint exists
- Health checks for both backend and frontend
- Opens browser automatically
- Graceful shutdown with proper cleanup
- Logs stored in logs/ directory

Prerequisites:
    npm install -g pm2

Usage:
    python scripts/demo/run_demo_pm2.py
    
    # To stop the demo:
    pm2 stop configs/pm2-ecosystem/ecosystem.config.js
    
    # To view logs:
    pm2 logs
    
    # To monitor processes:
    pm2 monit
"""

import sys
import subprocess
import time
import requests
from pathlib import Path
import webbrowser
import signal
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration
BACKEND_HOST = "localhost"
BACKEND_PORT = 8000
FRONTEND_PORT = 8501
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

# Paths
ECOSYSTEM_CONFIG = project_root / "configs" / "pm2-ecosystem" / "ecosystem.config.js"
MULTITASK_CHECKPOINT = project_root / "checkpoints" / "multitask_joint" / "best_model.pth"
LOGS_DIR = project_root / "logs"

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_banner():
    """Print welcome banner."""
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'🎯 SliceWise Multi-Task Demo Launcher (PM2)'.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print()
    print("This demo showcases the unified multi-task model that performs")
    print("both classification and segmentation in a single forward pass.")
    print()
    print(f"{Colors.OKGREEN}Benefits:{Colors.ENDC}")
    print("  🚀 ~40% faster inference")
    print("  💾 9.4% fewer parameters")
    print("  🎯 Conditional segmentation (smart resource usage)")
    print("  📊 91.3% accuracy, 97.1% sensitivity")
    print()
    print(f"{Colors.OKCYAN}PM2 Advantages:{Colors.ENDC}")
    print("  [OK] Robust process management")
    print("  [OK] Automatic restart on failure")
    print("  [OK] Centralized logging")
    print("  [OK] Easy monitoring with 'pm2 monit'")
    print()
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print()


def check_pm2_installed():
    """Check if PM2 is installed."""
    print(f"{Colors.OKBLUE}📦 Checking PM2 installation...{Colors.ENDC}")
    
    try:
        result = subprocess.run(
            ["pm2", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip()
        print(f"{Colors.OKGREEN}[OK] PM2 is installed (version {version}){Colors.ENDC}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Colors.FAIL}✗ PM2 is not installed{Colors.ENDC}")
        print()
        print(f"{Colors.WARNING}Please install PM2 using:{Colors.ENDC}")
        print("  npm install -g pm2")
        print()
        print("Or using yarn:")
        print("  yarn global add pm2")
        print()
        return False


def check_checkpoint():
    """Check if multi-task model checkpoint exists."""
    print(f"{Colors.OKBLUE}📁 Checking for multi-task model checkpoint...{Colors.ENDC}")
    
    if MULTITASK_CHECKPOINT.exists():
        size_mb = MULTITASK_CHECKPOINT.stat().st_size / (1024 * 1024)
        print(f"{Colors.OKGREEN}[OK] Checkpoint found: {MULTITASK_CHECKPOINT}{Colors.ENDC}")
        print(f"  Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"{Colors.FAIL}✗ Checkpoint not found: {MULTITASK_CHECKPOINT}{Colors.ENDC}")
        print()
        print("Please ensure you have trained the multi-task model:")
        print("  python scripts/run_full_pipeline.py --mode train-eval --training-mode quick")
        print()
        return False


def check_ecosystem_config():
    """Check if ecosystem.config.js exists."""
    print(f"{Colors.OKBLUE}📄 Checking ecosystem configuration...{Colors.ENDC}")
    
    if ECOSYSTEM_CONFIG.exists():
        print(f"{Colors.OKGREEN}[OK] Ecosystem config found: {ECOSYSTEM_CONFIG}{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.FAIL}✗ Ecosystem config not found: {ECOSYSTEM_CONFIG}{Colors.ENDC}")
        print()
        print("The ecosystem.config.js file should be in configs/pm2-ecosystem/.")
        return False


def create_logs_directory():
    """Create logs directory if it doesn't exist."""
    LOGS_DIR.mkdir(exist_ok=True)
    print(f"{Colors.OKGREEN}[OK] Logs directory ready: {LOGS_DIR}{Colors.ENDC}")


def stop_existing_pm2_processes():
    """Stop any existing PM2 processes for this project."""
    print(f"{Colors.OKBLUE}🛑 Stopping any existing PM2 processes...{Colors.ENDC}")
    
    try:
        # Stop all processes in the ecosystem
        subprocess.run(
            ["pm2", "delete", str(ECOSYSTEM_CONFIG)],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        print(f"{Colors.OKGREEN}[OK] Existing processes stopped{Colors.ENDC}")
    except subprocess.CalledProcessError:
        # No processes were running, which is fine
        pass


def start_pm2_processes():
    """Start PM2 processes using ecosystem config."""
    print()
    print(f"{Colors.OKBLUE}🚀 Starting PM2 processes...{Colors.ENDC}")
    
    try:
        result = subprocess.run(
            ["pm2", "start", str(ECOSYSTEM_CONFIG)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"{Colors.OKGREEN}[OK] PM2 processes started successfully{Colors.ENDC}")
        print()
        print(result.stdout)
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"{Colors.FAIL}✗ Failed to start PM2 processes{Colors.ENDC}")
        print(f"{Colors.FAIL}{e.stderr}{Colors.ENDC}")
        return False


def wait_for_backend(timeout=60):
    """Wait for backend to be ready."""
    print(f"{Colors.OKBLUE}⏳ Waiting for backend to start (timeout: {timeout}s)...{Colors.ENDC}")
    
    start_time = time.time()
    dots_printed = 0
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BACKEND_URL}/healthz", timeout=2)
            if response.status_code == 200:
                data = response.json()
                print(f"\n{Colors.OKGREEN}[OK] Backend is ready!{Colors.ENDC}")
                print(f"  Status: {data['status']}")
                print(f"  Multi-task loaded: {data.get('multitask_loaded', 'N/A')}")
                print(f"  Device: {data.get('device', 'N/A')}")
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
    print(f"{Colors.FAIL}✗ Backend failed to start within timeout{Colors.ENDC}")
    print()
    print(f"{Colors.WARNING}💡 Tip: Check backend logs with:{Colors.ENDC}")
    print(f"  pm2 logs slicewise-backend")
    return False


def wait_for_frontend(timeout=30):
    """Wait for frontend to be ready."""
    print()
    print(f"{Colors.OKBLUE}⏳ Waiting for frontend to start (timeout: {timeout}s)...{Colors.ENDC}")
    
    start_time = time.time()
    dots_printed = 0
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(FRONTEND_URL, timeout=2)
            if response.status_code == 200:
                print(f"\n{Colors.OKGREEN}[OK] Frontend is ready!{Colors.ENDC}")
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
    print(f"{Colors.FAIL}✗ Frontend failed to start within timeout{Colors.ENDC}")
    print()
    print(f"{Colors.WARNING}💡 Tip: Check frontend logs with:{Colors.ENDC}")
    print(f"  pm2 logs slicewise-frontend")
    return False


def open_browser():
    """Open browser to frontend URL."""
    print()
    print(f"{Colors.OKBLUE}🌐 Opening browser...{Colors.ENDC}")
    try:
        webbrowser.open(FRONTEND_URL)
        print(f"{Colors.OKGREEN}[OK] Browser opened to {FRONTEND_URL}{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.WARNING}⚠ Could not open browser automatically: {e}{Colors.ENDC}")
        print(f"  Please open {FRONTEND_URL} manually")


def print_instructions():
    """Print usage instructions."""
    print()
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'✅ Demo is running!'.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print()
    print(f"{Colors.OKGREEN}📍 URLs:{Colors.ENDC}")
    print(f"  Frontend: {FRONTEND_URL}")
    print(f"  Backend:  {BACKEND_URL}")
    print(f"  API Docs: {BACKEND_URL}/docs")
    print()
    print(f"{Colors.OKGREEN}🎯 Features:{Colors.ENDC}")
    print("  • Multi-Task tab: Unified classification + segmentation")
    print("  • Conditional segmentation (only if tumor prob >= 30%)")
    print("  • Grad-CAM visualization for interpretability")
    print("  • Performance metrics display")
    print("  • Clinical recommendations")
    print()
    print(f"{Colors.OKCYAN}📊 PM2 Commands:{Colors.ENDC}")
    print("  pm2 status              - View process status")
    print("  pm2 logs                - View all logs (live)")
    print("  pm2 logs slicewise-backend   - View backend logs")
    print("  pm2 logs slicewise-frontend  - View frontend logs")
    print("  pm2 monit               - Monitor processes (interactive)")
    print("  pm2 restart all         - Restart all processes")
    print("  pm2 stop all            - Stop all processes")
    print("  pm2 delete all          - Stop and remove all processes")
    print()
    print(f"{Colors.WARNING}⌨️  To stop the demo:{Colors.ENDC}")
    print("  pm2 stop configs/pm2-ecosystem/ecosystem.config.js")
    print("  # or")
    print("  pm2 delete configs/pm2-ecosystem/ecosystem.config.js")
    print()
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print()


def main():
    """Main function."""
    # Print banner
    print_banner()
    
    # Check PM2 installation
    if not check_pm2_installed():
        sys.exit(1)
    
    # Check checkpoint
    if not check_checkpoint():
        sys.exit(1)
    
    # Check ecosystem config
    if not check_ecosystem_config():
        sys.exit(1)
    
    # Create logs directory
    create_logs_directory()
    
    # Stop any existing processes
    stop_existing_pm2_processes()
    
    # Start PM2 processes
    if not start_pm2_processes():
        print()
        print(f"{Colors.FAIL}✗ Failed to start PM2 processes. Exiting.{Colors.ENDC}")
        sys.exit(1)
    
    # Wait for backend
    if not wait_for_backend():
        print()
        print(f"{Colors.FAIL}✗ Backend failed to start. Stopping PM2 processes.{Colors.ENDC}")
        subprocess.run(["pm2", "delete", str(ECOSYSTEM_CONFIG)], cwd=str(project_root))
        sys.exit(1)
    
    # Wait for frontend
    if not wait_for_frontend():
        print()
        print(f"{Colors.FAIL}✗ Frontend failed to start. Stopping PM2 processes.{Colors.ENDC}")
        subprocess.run(["pm2", "delete", str(ECOSYSTEM_CONFIG)], cwd=str(project_root))
        sys.exit(1)
    
    # Open browser
    open_browser()
    
    # Print instructions
    print_instructions()
    
    print(f"{Colors.OKGREEN}[OK] Demo launched successfully!{Colors.ENDC}")
    print(f"{Colors.OKBLUE}ℹ  Processes are managed by PM2 and will continue running in the background.{Colors.ENDC}")
    print()


if __name__ == "__main__":
    main()
