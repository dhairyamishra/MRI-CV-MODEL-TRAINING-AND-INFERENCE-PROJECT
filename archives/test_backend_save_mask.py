#!/usr/bin/env python3
"""
Test backend API and save the mask images to verify what's being sent.
"""

import requests
import base64
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io

# Test image path
project_root = Path(__file__).parent.parent
test_image_path = project_root / "data" / "dataset_examples" / "kaggle" / "yes_tumor" / "sample_000" / "image.png"

# Load image
image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)

print(f"\n{'='*80}")
print(f"Testing Backend API - Saving Mask Images")
print(f"{'='*80}\n")

# Make request to backend
url = "http://localhost:8000/predict_multitask"
_, buffer = cv2.imencode('.png', image)
files = {'file': ('test.png', buffer.tobytes(), 'image/png')}
params = {'include_gradcam': True}

print(f"Sending request to: {url}")

try:
    response = requests.post(url, files=files, params=params, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\nResponse received successfully!")
        print(f"Tumor percentage: {result['segmentation']['tumor_percentage']:.2f}%")
        print(f"Tumor pixels: {result['segmentation']['tumor_area_pixels']}")
        
        # Decode and save mask
        if result['segmentation']['mask_available']:
            # Decode binary mask
            mask_base64 = result['segmentation']['mask_base64']
            mask_bytes = base64.b64decode(mask_base64)
            mask_image = Image.open(io.BytesIO(mask_bytes))
            mask_array = np.array(mask_image)
            
            # Decode probability map
            prob_base64 = result['segmentation']['prob_map_base64']
            prob_bytes = base64.b64decode(prob_base64)
            prob_image = Image.open(io.BytesIO(prob_bytes))
            prob_array = np.array(prob_image)
            
            # Decode overlay
            overlay_base64 = result['segmentation']['overlay_base64']
            overlay_bytes = base64.b64decode(overlay_base64)
            overlay_image = Image.open(io.BytesIO(overlay_bytes))
            overlay_array = np.array(overlay_image)
            
            # Save images
            output_dir = project_root / "test_output"
            output_dir.mkdir(exist_ok=True)
            
            cv2.imwrite(str(output_dir / "original.png"), image)
            cv2.imwrite(str(output_dir / "mask_from_api.png"), mask_array)
            cv2.imwrite(str(output_dir / "prob_map_from_api.png"), prob_array)
            cv2.imwrite(str(output_dir / "overlay_from_api.png"), overlay_array)
            
            print(f"\n✅ Images saved to: {output_dir}")
            print(f"\nMask statistics:")
            print(f"  - Shape: {mask_array.shape}")
            print(f"  - Dtype: {mask_array.dtype}")
            print(f"  - Unique values: {np.unique(mask_array)}")
            print(f"  - Min: {mask_array.min()}, Max: {mask_array.max()}")
            print(f"  - Tumor pixels (255): {(mask_array == 255).sum()}")
            print(f"  - Background pixels (0): {(mask_array == 0).sum()}")
            print(f"  - Percentage tumor: {(mask_array == 255).sum() / mask_array.size * 100:.2f}%")
            
            # Check if mask looks inverted
            tumor_pixels_from_stats = result['segmentation']['tumor_area_pixels']
            tumor_pixels_from_mask = (mask_array == 255).sum()
            background_pixels_from_mask = (mask_array == 0).sum()
            
            print(f"\n🔍 Verification:")
            print(f"  - API reports: {tumor_pixels_from_stats} tumor pixels")
            print(f"  - Mask has: {tumor_pixels_from_mask} white pixels (255)")
            print(f"  - Mask has: {background_pixels_from_mask} black pixels (0)")
            
            if tumor_pixels_from_mask == tumor_pixels_from_stats:
                print(f"  ✅ MATCH: White pixels (255) = tumor")
            elif background_pixels_from_mask == tumor_pixels_from_stats:
                print(f"  ⚠️  INVERTED: Black pixels (0) = tumor (display issue!)")
            else:
                print(f"  ❌ MISMATCH: Neither matches!")
        
        print(f"\n{'='*80}")
        print(f"Test Complete")
        print(f"{'='*80}\n")
        
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ Exception: {e}")
    import traceback
    traceback.print_exc()
