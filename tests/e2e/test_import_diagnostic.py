"""
Diagnostic script to identify why frontend imports are failing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")
print("\n" + "="*80)
print("TESTING FRONTEND IMPORTS")
print("="*80 + "\n")

# Test 1: Import api_client
print("1. Testing: from app.frontend.utils.api_client import APIClient")
try:
    from app.frontend.utils.api_client import APIClient
    print("   ✅ SUCCESS: APIClient imported")
except ImportError as e:
    print(f"   ❌ FAILED: {e}")
except Exception as e:
    print(f"   ❌ ERROR: {type(e).__name__}: {e}")

# Test 2: Import image_utils
print("\n2. Testing: from app.frontend.utils.image_utils import process_uploaded_image, validate_image")
try:
    from app.frontend.utils.image_utils import process_uploaded_image, validate_image
    print("   ✅ SUCCESS: image_utils imported")
except ImportError as e:
    print(f"   ❌ FAILED: {e}")
except Exception as e:
    print(f"   ❌ ERROR: {type(e).__name__}: {e}")

# Test 3: Import validators
print("\n3. Testing: from app.frontend.utils.validators import validate_api_response")
try:
    from app.frontend.utils.validators import validate_api_response
    print("   ✅ SUCCESS: validators imported")
except ImportError as e:
    print(f"   ❌ FAILED: {e}")
except Exception as e:
    print(f"   ❌ ERROR: {type(e).__name__}: {e}")

# Test 4: Import components
print("\n4. Testing: from app.frontend.components.header import render_header")
try:
    from app.frontend.components.header import render_header
    print("   ✅ SUCCESS: header component imported")
except ImportError as e:
    print(f"   ❌ FAILED: {e}")
except Exception as e:
    print(f"   ❌ ERROR: {type(e).__name__}: {e}")

# Test 5: Import main app
print("\n5. Testing: from app.frontend.app import main")
try:
    from app.frontend.app import main as streamlit_app
    print("   ✅ SUCCESS: main app imported")
except ImportError as e:
    print(f"   ❌ FAILED: {e}")
except Exception as e:
    print(f"   ❌ ERROR: {type(e).__name__}: {e}")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
