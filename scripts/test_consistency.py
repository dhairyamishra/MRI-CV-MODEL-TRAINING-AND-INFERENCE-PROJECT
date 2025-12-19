#!/usr/bin/env python3
"""
Test if the model gives consistent results across multiple runs.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.multi_task_predictor import MultiTaskPredictor

test_image_path = project_root / "data" / "dataset_examples" / "kaggle" / "yes_tumor" / "sample_000" / "image.png"
checkpoint_path = project_root / "checkpoints" / "multitask_joint" / "best_model.pth"

print(f"\n{'='*80}")
print(f"Testing Model Consistency (5 runs)")
print(f"{'='*80}\n")

# Load model once
predictor = MultiTaskPredictor(str(checkpoint_path))
image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)

results = []
for i in range(5):
    print(f"Run {i+1}...", end=" ")
    result = predictor.predict_full(image, include_gradcam=False)
    
    if result['segmentation_computed']:
        tumor_pixels = result['segmentation']['tumor_area_pixels']
        tumor_pct = result['segmentation']['tumor_percentage']
        confidence = result['classification']['confidence']
        results.append((tumor_pixels, tumor_pct, confidence))
        print(f"Tumor: {tumor_pixels} px ({tumor_pct:.2f}%), Confidence: {confidence*100:.1f}%")
    else:
        print("No segmentation")

print(f"\n{'='*80}")
print(f"Analysis")
print(f"{'='*80}")

if len(results) > 0:
    tumor_pixels_list = [r[0] for r in results]
    tumor_pct_list = [r[1] for r in results]
    confidence_list = [r[2] for r in results]
    
    print(f"\nTumor Pixels:")
    print(f"  Min: {min(tumor_pixels_list)}")
    print(f"  Max: {max(tumor_pixels_list)}")
    print(f"  Mean: {np.mean(tumor_pixels_list):.0f}")
    print(f"  Std: {np.std(tumor_pixels_list):.2f}")
    print(f"  All identical: {'✅ YES' if len(set(tumor_pixels_list)) == 1 else '❌ NO'}")
    
    print(f"\nConfidence:")
    print(f"  Min: {min(confidence_list)*100:.1f}%")
    print(f"  Max: {max(confidence_list)*100:.1f}%")
    print(f"  Mean: {np.mean(confidence_list)*100:.1f}%")
    print(f"  Std: {np.std(confidence_list)*100:.3f}%")
    print(f"  All identical: {'✅ YES' if len(set(confidence_list)) == 1 else '❌ NO'}")
    
    if len(set(tumor_pixels_list)) == 1 and len(set(confidence_list)) == 1:
        print(f"\n✅ Model is CONSISTENT across runs")
    else:
        print(f"\n⚠️  Model shows VARIABILITY across runs")
        print(f"   This suggests dropout or other stochastic behavior")

print(f"\n{'='*80}\n")
