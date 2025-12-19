#!/usr/bin/env python3
"""
Visualize multi-task segmentation results with skull boundary detection.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.multi_task_predictor import MultiTaskPredictor

# Load model
checkpoint_path = project_root / "checkpoints" / "1000_epoch_multitask_joint" / "best_model.pth"
predictor = MultiTaskPredictor(str(checkpoint_path))

# Load a Kaggle test image
test_image_path = project_root / "data" / "dataset_examples" / "kaggle" / "yes_tumor" / "sample_000" / "image.png"
image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)

print(f"\n{'='*80}")
print(f"Testing Multi-Task Predictor with Visualization")
print(f"{'='*80}\n")

# Run prediction
result = predictor.predict_full(image, include_gradcam=False)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Original, Probability Map, Binary Mask
axes[0, 0].imshow(result['image_original'], cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

if result['segmentation_computed']:
    seg = result['segmentation']
    
    # Probability map
    axes[0, 1].imshow(seg['prob_map'], cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title(f"Probability Map\n(max: {seg['prob_map'].max():.3f})")
    axes[0, 1].axis('off')
    
    # Binary mask
    axes[0, 2].imshow(seg['mask'], cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f"Binary Mask\n({seg['tumor_percentage']:.1f}% tumor)")
    axes[0, 2].axis('off')
    
    # Row 2: Overlay, Skull boundary, Statistics
    # Create overlay
    overlay = result['image_original'].copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        overlay = overlay.astype(np.float32) / 255.0
    
    # Add red overlay where tumor is detected
    overlay[:, :, 0] = np.where(seg['mask'] > 0.5, 
                                 np.minimum(overlay[:, :, 0] + 0.5, 1.0), 
                                 overlay[:, :, 0])
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Tumor Overlay (Red)')
    axes[1, 0].axis('off')
    
    # Detect and show skull boundary
    skull_mask = predictor._detect_skull_boundary(result['image_original'])
    if skull_mask is not None:
        axes[1, 1].imshow(skull_mask, cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title(f"Skull Boundary Mask\n({skull_mask.sum()/skull_mask.size*100:.1f}% coverage)")
        axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'Skull detection failed', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
    
    # Statistics
    stats_text = f"""
Classification:
  Predicted: {result['classification']['predicted_label']}
  Confidence: {result['classification']['confidence']*100:.1f}%
  Tumor Prob: {result['classification']['tumor_probability']*100:.1f}%

Segmentation:
  Tumor Pixels: {seg['tumor_area_pixels']:,}
  Total Pixels: {seg['mask'].size:,}
  Tumor %: {seg['tumor_percentage']:.2f}%
  
Probability Map:
  Min: {seg['prob_map'].min():.3f}
  Max: {seg['prob_map'].max():.3f}
  Mean: {seg['prob_map'].mean():.3f}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, 
                   fontsize=10, family='monospace',
                   verticalalignment='center',
                   transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
else:
    for ax in axes.flat[1:]:
        ax.text(0.5, 0.5, 'Segmentation not computed', 
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

plt.tight_layout()

# Save figure
output_path = project_root / "multitask_segmentation_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {output_path}")

# Show plot
plt.show()

print(f"\n{'='*80}")
print(f"Visualization Complete")
print(f"{'='*80}\n")
