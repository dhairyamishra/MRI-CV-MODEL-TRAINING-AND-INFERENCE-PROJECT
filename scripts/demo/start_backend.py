"""
Backend starter script for PM2.

This wrapper script starts the FastAPI backend using uvicorn.
It's designed to be run by PM2 without spawning new terminal windows.

Updated to use the new modular backend (main.py instead of main_v2.py).
"""
import sys
import os
import io

# Fix encoding for pythonw.exe on Windows (no console)
if sys.stdout is None:
    sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')
if sys.stderr is None:
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')

# Set UTF-8 encoding for print statements
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure we're in the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
sys.path.insert(0, project_root)

# Start uvicorn
if __name__ == "__main__":
    import uvicorn
    
    # Use the new modular backend (main.py)
    # Disable reload on Windows when running under PM2 to prevent handle issues
    uvicorn.run(
        "app.backend.main:app",  # Updated from main_v2 to main
        host="localhost",
        port=8000,
        reload=False,  # Disabled for PM2 compatibility on Windows
        log_level="info",
        access_log=True
    )
