#!/usr/bin/env python3
"""
Quick test to see debug output from multi-task predictor.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.multi_task_predictor import MultiTaskPredictor

# Load model - Use the model from run_full_pipeline.py
# checkpoint_path = project_root / "checkpoints" / "1000_epoch_multitask_joint" / "best_model.pth"  # Old 1000-epoch model
checkpoint_path = project_root / "checkpoints" / "multitask_joint" / "best_model.pth"  # Pipeline model
predictor = MultiTaskPredictor(str(checkpoint_path))

# Load a Kaggle test image
test_image_path = project_root / "data" / "dataset_examples" / "kaggle" / "yes_tumor" / "sample_000" / "image.png"
image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)

print(f"\n{'='*80}")
print(f"Testing Multi-Task Predictor with Kaggle Image")
print(f"{'='*80}\n")

# Run prediction
result = predictor.predict_full(image, include_gradcam=False)

print(f"\n{'='*80}")
print(f"Prediction Complete")
print(f"{'='*80}\n")

# Check results
if result['segmentation_computed']:
    seg = result['segmentation']
    print(f"Segmentation Results:")
    print(f"  - Tumor pixels: {seg['tumor_area_pixels']}")
    print(f"  - Tumor percentage: {seg['tumor_percentage']:.2f}%")
    print(f"  - Mask unique values: {np.unique(seg['mask'])}")
    print(f"  - Mask shape: {seg['mask'].shape}")
else:
    print(f"Segmentation not computed: {result['recommendation']}")
