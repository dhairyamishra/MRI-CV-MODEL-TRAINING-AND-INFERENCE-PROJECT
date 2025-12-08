"""
Frontend starter script for PM2.

This wrapper script starts the Streamlit frontend.
It's designed to be run by PM2 without spawning new terminal windows.
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

# Start streamlit
if __name__ == "__main__":
    from streamlit.web import cli as stcli
    
    sys.argv = [
        "streamlit",
        "run",
        "app/frontend/app.py",
        "--server.port=8501",
        "--server.headless=true"
    ]
    
    sys.exit(stcli.main())
