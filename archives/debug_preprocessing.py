#!/usr/bin/env python3
"""
Debug preprocessing differences between command-line and API.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import io

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

test_image_path = project_root / "data" / "dataset_examples" / "kaggle" / "yes_tumor" / "sample_000" / "image.png"

print(f"\n{'='*80}")
print(f"Debugging Preprocessing Differences")
print(f"{'='*80}\n")

# Method 1: cv2.imread (command-line style)
print("Method 1: cv2.imread (command-line)")
image_cv2 = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
print(f"  Shape: {image_cv2.shape}")
print(f"  Dtype: {image_cv2.dtype}")
print(f"  Range: [{image_cv2.min()}, {image_cv2.max()}]")
print(f"  Mean: {image_cv2.mean():.2f}")

# Method 2: PIL -> numpy (API style)
print("\nMethod 2: PIL Image.open -> np.array (API)")
with open(test_image_path, 'rb') as f:
    image_bytes = f.read()
image_pil = Image.open(io.BytesIO(image_bytes))
print(f"  PIL mode: {image_pil.mode}")
print(f"  PIL size: {image_pil.size}")

image_np = np.array(image_pil)
print(f"  Shape: {image_np.shape}")
print(f"  Dtype: {image_np.dtype}")
print(f"  Range: [{image_np.min()}, {image_np.max()}]")
print(f"  Mean: {image_np.mean():.2f}")

# Check if they're identical
print(f"\n{'='*80}")
print(f"Comparison")
print(f"{'='*80}")
print(f"Arrays identical: {np.array_equal(image_cv2, image_np)}")
print(f"Max difference: {np.abs(image_cv2.astype(float) - image_np.astype(float)).max()}")

if not np.array_equal(image_cv2, image_np):
    print(f"\n⚠️  WARNING: Preprocessing methods produce different arrays!")
    print(f"This could explain the inconsistent results.")
    
    # Check if it's just a shape issue
    if len(image_np.shape) == 3:
        print(f"\n  PIL loaded as RGB/RGBA (shape: {image_np.shape})")
        print(f"  Converting to grayscale...")
        image_pil_gray = image_pil.convert('L')
        image_np_gray = np.array(image_pil_gray)
        print(f"  After conversion: {image_np_gray.shape}")
        print(f"  Arrays identical now: {np.array_equal(image_cv2, image_np_gray)}")
else:
    print(f"\n✅ Preprocessing methods are identical")

print(f"\n{'='*80}\n")
