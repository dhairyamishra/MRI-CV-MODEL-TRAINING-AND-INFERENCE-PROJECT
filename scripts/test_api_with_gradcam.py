#!/usr/bin/env python3
"""
Test API with include_gradcam=True (as the UI does).
"""

from pathlib import Path
import cv2
import requests

project_root = Path(__file__).parent.parent
test_image_path = project_root / "data" / "dataset_examples" / "kaggle" / "yes_tumor" / "sample_000" / "image.png"

image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)

print(f"\n{'='*80}")
print(f"Testing API with include_gradcam=True (UI behavior)")
print(f"{'='*80}\n")

url = "http://localhost:8000/predict_multitask"
_, buffer = cv2.imencode('.png', image)
files = {'file': ('test.png', buffer.tobytes(), 'image/png')}

# Test with include_gradcam=True (default in UI)
params = {'include_gradcam': True}

print(f"Request: {url}?include_gradcam=True")

response = requests.post(url, files=files, params=params, timeout=30)

if response.status_code == 200:
    result = response.json()
    
    print(f"\n✅ Results:")
    print(f"  Classification: {result['classification']['predicted_label']}")
    print(f"  Confidence: {result['classification']['confidence']*100:.1f}%")
    
    if result['segmentation_computed']:
        seg = result['segmentation']
        print(f"  Segmentation:")
        print(f"    - Tumor pixels: {seg['tumor_area_pixels']}")
        print(f"    - Tumor %: {seg['tumor_percentage']:.2f}%")
        
        print(f"\n📋 Expected (from command-line test):")
        print(f"    - Tumor pixels: 16635")
        print(f"    - Tumor %: 25.38%")
        print(f"    - Confidence: 62.7%")
        
        if seg['tumor_area_pixels'] == 16635:
            print(f"\n✅ MATCH: API with Grad-CAM matches command-line!")
        else:
            print(f"\n❌ MISMATCH: API gives different results")
            print(f"   Difference: {abs(seg['tumor_area_pixels'] - 16635)} pixels")
else:
    print(f"\n❌ Error: {response.status_code}")
    print(f"Response: {response.text}")

print(f"\n{'='*80}\n")
