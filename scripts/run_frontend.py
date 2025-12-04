"""
Script to run the Streamlit frontend application.
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("="*60)
    print("SliceWise - Starting Frontend Application")
    print("="*60)
    print("\nMake sure the backend API is running on http://localhost:8000")
    print("Frontend will be available at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application\n")
    
    # Run streamlit
    frontend_path = project_root / "app" / "frontend" / "app.py"
    subprocess.run([
        "streamlit", "run",
        str(frontend_path),
        "--server.port", "8501",
        "--server.address", "localhost"
    ])
