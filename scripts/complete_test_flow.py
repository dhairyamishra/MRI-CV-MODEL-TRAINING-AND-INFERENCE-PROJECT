#!/usr/bin/env python3
"""
Complete test flow for skull boundary detection fix.
Tests the same image through command-line and API to verify consistency.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import requests
import base64
from PIL import Image
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.multi_task_predictor import MultiTaskPredictor

print(f"\n{'='*80}")
print(f"COMPLETE TEST FLOW - Skull Boundary Detection")
print(f"{'='*80}\n")

# Test image path
test_image_path = project_root / "data" / "dataset_examples" / "kaggle" / "yes_tumor" / "sample_000" / "image.png"
print(f"Test Image: {test_image_path}")
print(f"Image exists: {test_image_path.exists()}")

if not test_image_path.exists():
    print(f"\n❌ Test image not found!")
    sys.exit(1)

# Load image
image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
print(f"Image shape: {image.shape}")
print(f"Image range: [{image.min()}, {image.max()}]")

# ============================================================================
# TEST 1: Command-Line Predictor
# ============================================================================
print(f"\n{'='*80}")
print(f"TEST 1: Command-Line MultiTaskPredictor")
print(f"{'='*80}\n")

checkpoint_path = project_root / "checkpoints" / "multitask_joint" / "best_model.pth"
print(f"Loading model from: {checkpoint_path}")

predictor = MultiTaskPredictor(str(checkpoint_path))
result_cli = predictor.predict_full(image, include_gradcam=False)

print(f"\n✅ Command-Line Results:")
print(f"  Classification: {result_cli['classification']['predicted_label']}")
print(f"  Confidence: {result_cli['classification']['confidence']*100:.1f}%")

if result_cli['segmentation_computed']:
    seg = result_cli['segmentation']
    print(f"  Segmentation:")
    print(f"    - Tumor pixels: {seg['tumor_area_pixels']}")
    print(f"    - Tumor %: {seg['tumor_percentage']:.2f}%")
    print(f"    - Mask shape: {seg['mask'].shape}")
    print(f"    - Mask unique: {np.unique(seg['mask'])}")

# ============================================================================
# TEST 2: Backend API
# ============================================================================
print(f"\n{'='*80}")
print(f"TEST 2: Backend API (/predict_multitask)")
print(f"{'='*80}\n")

url = "http://localhost:8000/predict_multitask"
_, buffer = cv2.imencode('.png', image)
files = {'file': ('test.png', buffer.tobytes(), 'image/png')}
params = {'include_gradcam': False}

print(f"Sending request to: {url}")

try:
    response = requests.post(url, files=files, params=params, timeout=30)
    
    if response.status_code == 200:
        result_api = response.json()
        
        print(f"\n✅ API Results:")
        print(f"  Classification: {result_api['classification']['predicted_label']}")
        print(f"  Confidence: {result_api['classification']['confidence']*100:.1f}%")
        
        if result_api['segmentation_computed']:
            seg = result_api['segmentation']
            print(f"  Segmentation:")
            print(f"    - Tumor pixels: {seg['tumor_area_pixels']}")
            print(f"    - Tumor %: {seg['tumor_percentage']:.2f}%")
            
            # Decode mask
            mask_base64 = seg['mask_base64']
            mask_bytes = base64.b64decode(mask_base64)
            mask_image = Image.open(io.BytesIO(mask_bytes))
            mask_array = np.array(mask_image)
            
            print(f"    - Mask shape: {mask_array.shape}")
            print(f"    - Mask unique: {np.unique(mask_array)}")
            print(f"    - White pixels (255): {(mask_array == 255).sum()}")
    else:
        print(f"\n❌ API Error: {response.status_code}")
        print(f"Response: {response.text}")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ API Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# COMPARISON
# ============================================================================
print(f"\n{'='*80}")
print(f"COMPARISON")
print(f"{'='*80}\n")

if result_cli['segmentation_computed'] and result_api['segmentation_computed']:
    cli_pixels = result_cli['segmentation']['tumor_area_pixels']
    api_pixels = result_api['segmentation']['tumor_area_pixels']
    cli_pct = result_cli['segmentation']['tumor_percentage']
    api_pct = result_api['segmentation']['tumor_percentage']
    
    print(f"Tumor Pixels:")
    print(f"  Command-Line: {cli_pixels}")
    print(f"  API:          {api_pixels}")
    print(f"  Match: {'✅ YES' if cli_pixels == api_pixels else '❌ NO'}")
    
    print(f"\nTumor Percentage:")
    print(f"  Command-Line: {cli_pct:.2f}%")
    print(f"  API:          {api_pct:.2f}%")
    print(f"  Match: {'✅ YES' if abs(cli_pct - api_pct) < 0.01 else '❌ NO'}")
    
    if cli_pixels == api_pixels:
        print(f"\n✅ SUCCESS: Command-line and API results match!")
        print(f"\n📋 NEXT STEP:")
        print(f"   Upload this image in the UI:")
        print(f"   {test_image_path}")
        print(f"\n   Expected results in UI:")
        print(f"   - Tumor Area: {cli_pixels} px")
        print(f"   - Tumor %: {cli_pct:.2f}%")
        print(f"   - Binary Mask: White regions = tumor, Black = background")
    else:
        print(f"\n⚠️  WARNING: Results don't match!")
        print(f"   This suggests inconsistency in the model or preprocessing.")
else:
    print(f"⚠️  Segmentation not computed in one or both tests")

print(f"\n{'='*80}")
print(f"Test Complete")
print(f"{'='*80}\n")
