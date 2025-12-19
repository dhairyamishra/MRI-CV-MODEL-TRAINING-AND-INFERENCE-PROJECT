#!/usr/bin/env python3
"""
Test backend API directly to see skull boundary detection in action.
"""

import requests
import base64
from pathlib import Path
import cv2
import numpy as np

# Test image path
project_root = Path(__file__).parent.parent
test_image_path = project_root / "data" / "dataset_examples" / "kaggle" / "yes_tumor" / "sample_000" / "image.png"

# Load and encode image
image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
_, buffer = cv2.imencode('.png', image)
image_base64 = base64.b64encode(buffer).decode('utf-8')

print(f"\n{'='*80}")
print(f"Testing Backend API - Multi-Task Prediction")
print(f"{'='*80}\n")

# Make request to backend using multipart/form-data (file upload)
url = "http://localhost:8000/predict_multitask"

print(f"Sending request to: {url}")
print(f"Image shape: {image.shape}")
print(f"Image range: [{image.min()}, {image.max()}]")

try:
    # Encode image as PNG bytes for file upload
    _, buffer = cv2.imencode('.png', image)
    files = {'file': ('test.png', buffer.tobytes(), 'image/png')}
    params = {'include_gradcam': True}
    
    response = requests.post(url, files=files, params=params, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n{'='*80}")
        print(f"Response Received")
        print(f"{'='*80}\n")
        
        # Classification
        cls = result['classification']
        print(f"Classification:")
        print(f"  Predicted: {cls['predicted_label']}")
        print(f"  Confidence: {cls['confidence']*100:.1f}%")
        print(f"  Tumor Probability: {cls['tumor_probability']*100:.1f}%")
        
        # Segmentation
        if result['segmentation_computed']:
            seg = result['segmentation']
            print(f"\nSegmentation:")
            print(f"  Tumor Area: {seg['tumor_area_pixels']} pixels")
            print(f"  Tumor %: {seg['tumor_percentage']:.2f}%")
            print(f"  Mask Available: {seg['mask_available']}")
        else:
            print(f"\nSegmentation: Not computed (below threshold)")
        
        # Processing time
        print(f"\nProcessing Time: {result['processing_time_ms']:.2f}ms")
        
        # Recommendation
        print(f"\nRecommendation: {result['recommendation']}")
        
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
