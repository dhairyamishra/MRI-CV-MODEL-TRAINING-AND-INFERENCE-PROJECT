#!/usr/bin/env python3
"""Test if mask encoding is inverting the image."""

import numpy as np
from PIL import Image
import io
import base64

# Create a simple test mask
mask = np.zeros((256, 256), dtype=np.float32)
mask[50:200, 50:200] = 1.0  # White square in center

print(f"Original mask:")
print(f"  - Unique values: {np.unique(mask)}")
print(f"  - Center pixel (should be 1): {mask[100, 100]}")
print(f"  - Corner pixel (should be 0): {mask[0, 0]}")

# Scale to 0-255
mask_255 = (mask * 255).astype(np.uint8)
print(f"\nScaled mask (0-255):")
print(f"  - Unique values: {np.unique(mask_255)}")
print(f"  - Center pixel (should be 255): {mask_255[100, 100]}")
print(f"  - Corner pixel (should be 0): {mask_255[0, 0]}")

# Convert to PIL Image
pil_image = Image.fromarray(mask_255, mode='L')
print(f"\nPIL Image:")
print(f"  - Mode: {pil_image.mode}")
print(f"  - Size: {pil_image.size}")

# Save and reload to check
pil_image.save('test_mask.png')
reloaded = Image.open('test_mask.png')
reloaded_array = np.array(reloaded)

print(f"\nReloaded from PNG:")
print(f"  - Unique values: {np.unique(reloaded_array)}")
print(f"  - Center pixel (should be 255): {reloaded_array[100, 100]}")
print(f"  - Corner pixel (should be 0): {reloaded_array[0, 0]}")

if reloaded_array[100, 100] == 255 and reloaded_array[0, 0] == 0:
    print(f"\n✅ Encoding is CORRECT - no inversion")
else:
    print(f"\n❌ Encoding is INVERTED!")

# Clean up
import os
os.remove('test_mask.png')
