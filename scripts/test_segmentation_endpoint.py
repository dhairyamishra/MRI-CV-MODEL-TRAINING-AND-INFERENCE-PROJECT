"""
Test script for segmentation endpoint.

This script tests the /segment endpoint directly to debug base64 encoding issues.
"""

import requests
from pathlib import Path
import json

# Configuration
API_URL = "http://localhost:8000"
TEST_IMAGE = Path("test_output/original.png")  # Use existing test image

def test_segmentation_endpoint():
    """Test the segmentation endpoint."""
    print("=" * 80)
    print("Testing Segmentation Endpoint")
    print("=" * 80)
    
    # Check if test image exists
    if not TEST_IMAGE.exists():
        print(f"❌ Test image not found: {TEST_IMAGE}")
        print("Please run the demo first to generate test images.")
        return
    
    print(f"\n1. Loading test image: {TEST_IMAGE}")
    
    # Read image
    with open(TEST_IMAGE, 'rb') as f:
        image_data = f.read()
    
    print(f"   Image size: {len(image_data)} bytes")
    
    # Test endpoint
    print(f"\n2. Sending request to {API_URL}/segment")
    
    try:
        response = requests.post(
            f"{API_URL}/segment",
            files={"file": ("test.png", image_data, "image/png")},
            params={
                "threshold": 0.5,
                "min_object_size": 50,
                "apply_postprocessing": True,
                "return_overlay": True
            },
            timeout=30
        )
        
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n3. Response received:")
            print(f"   has_tumor: {result.get('has_tumor')}")
            print(f"   tumor_probability: {result.get('tumor_probability')}")
            print(f"   tumor_area_pixels: {result.get('tumor_area_pixels')}")
            print(f"   num_components: {result.get('num_components')}")
            
            # Check base64 fields
            print(f"\n4. Base64 fields:")
            mask_b64 = result.get('mask_base64', '')
            prob_b64 = result.get('probability_map_base64', '')
            overlay_b64 = result.get('overlay_base64', '')
            
            print(f"   mask_base64 length: {len(mask_b64)}")
            print(f"   probability_map_base64 length: {len(prob_b64) if prob_b64 else 0}")
            print(f"   overlay_base64 length: {len(overlay_b64) if overlay_b64 else 0}")
            
            # Try to decode mask
            if mask_b64:
                print(f"\n5. Testing base64 decode:")
                try:
                    import base64
                    from PIL import Image
                    import io
                    
                    img_bytes = base64.b64decode(mask_b64)
                    img = Image.open(io.BytesIO(img_bytes))
                    print(f"   ✅ Successfully decoded mask image: {img.size}, mode={img.mode}")
                except Exception as e:
                    print(f"   ❌ Failed to decode mask: {str(e)}")
                    print(f"   First 100 chars of mask_base64: {mask_b64[:100]}")
            else:
                print(f"\n5. ❌ mask_base64 is empty!")
            
            # Save full response
            output_file = Path("test_output/segmentation_response.json")
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                # Don't save base64 strings (too large)
                result_copy = result.copy()
                if 'mask_base64' in result_copy:
                    result_copy['mask_base64'] = f"<{len(result_copy['mask_base64'])} chars>"
                if 'probability_map_base64' in result_copy:
                    result_copy['probability_map_base64'] = f"<{len(result_copy.get('probability_map_base64', ''))} chars>"
                if 'overlay_base64' in result_copy:
                    result_copy['overlay_base64'] = f"<{len(result_copy.get('overlay_base64', ''))} chars>"
                json.dump(result_copy, f, indent=2)
            print(f"\n6. Response saved to: {output_file}")
            
        else:
            print(f"\n❌ Error response:")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Could not connect to API at {API_URL}")
        print("   Make sure the backend is running:")
        print("   python scripts/demo/run_demo_backend.py")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_segmentation_endpoint()
