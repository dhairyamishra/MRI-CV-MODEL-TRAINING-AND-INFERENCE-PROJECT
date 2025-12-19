#!/usr/bin/env python3
"""
Visualize robust brain masking results.

Shows:
1. Original image
2. Brain mask
3. Masked image (z-scored)
4. Quality metrics

Compares passed vs failed quality checks.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def visualize_robust_masking(
    processed_dir: str = "data/processed/kaggle_robust_test",
    num_passed: int = 6,
    num_failed: int = 6,
    save_path: str = None,
):
    """
    Visualize robust masking results.
    
    Args:
        processed_dir: Directory with processed .npz files
        num_passed: Number of passed samples to show
        num_failed: Number of failed samples to show
        save_path: Path to save figure (optional)
    """
    print("\n" + "="*70)
    print("Robust Brain Masking Visualization")
    print("="*70)
    
    processed_path = Path(processed_dir)
    
    if not processed_path.exists():
        print(f"Error: Processed directory not found: {processed_path}")
        return
    
    # Load all .npz files
    npz_files = sorted(processed_path.glob("*.npz"))
    
    if not npz_files:
        print(f"Error: No .npz files found in {processed_path}")
        return
    
    print(f"\nFound {len(npz_files)} processed files")
    print(f"Loading samples...\n")
    
    # Separate passed and failed samples
    passed_samples = []
    failed_samples = []
    
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        # Extract quality info
        metadata = data['metadata'].item()
        quality = metadata.get('mask_quality', {})
        
        sample_info = {
            'file': npz_file,
            'image': data['image'],
            'mask': data['mask'] if 'mask' in data else None,
            'label': int(data['label']),
            'quality': quality,
        }
        
        if quality.get('passed', False):
            passed_samples.append(sample_info)
        else:
            failed_samples.append(sample_info)
        
        # Stop early if we have enough
        if len(passed_samples) >= num_passed * 2 and len(failed_samples) >= num_failed * 2:
            break
    
    print(f"Passed samples: {len(passed_samples)}")
    print(f"Failed samples: {len(failed_samples)}")
    
    # Sample randomly
    import random
    random.seed(42)
    
    if len(passed_samples) > num_passed:
        passed_samples = random.sample(passed_samples, num_passed)
    if len(failed_samples) > num_failed:
        failed_samples = random.sample(failed_samples, num_failed)
    
    # Create figure
    total_samples = len(passed_samples) + len(failed_samples)
    fig, axes = plt.subplots(total_samples, 4, figsize=(16, 4*total_samples))
    
    if total_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Plot passed samples
    for idx, sample in enumerate(passed_samples):
        plot_sample(axes[idx], sample, f"PASSED #{idx+1}")
    
    # Plot failed samples
    for idx, sample in enumerate(failed_samples):
        plot_sample(axes[len(passed_samples) + idx], sample, f"FAILED #{idx+1}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved to: {save_path}")
    else:
        save_path = 'robust_masking_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved to: {save_path}")
    
    plt.show()


def plot_sample(axes_row, sample, title_prefix):
    """Plot a single sample across 4 columns."""
    
    image = sample['image'].squeeze()  # (H, W)
    mask = sample['mask'] if sample['mask'] is not None else np.ones_like(image)
    label = sample['label']
    quality = sample['quality']
    
    # Get quality metrics
    area_frac = quality.get('area_fraction', 0) * 100
    border_frac = quality.get('border_fraction', 0) * 100
    reason = quality.get('reason', 'unknown')
    
    label_text = "Tumor" if label == 1 else "No Tumor"
    
    # Column 1: Original image (denormalize for visualization)
    # The image is z-scored, so we need to show it in a reasonable range
    img_display = image.copy()
    # Clip to reasonable range for visualization
    img_display = np.clip(img_display, -3, 3)
    img_display = (img_display + 3) / 6  # Map [-3, 3] to [0, 1]
    
    axes_row[0].imshow(img_display, cmap='gray', vmin=0, vmax=1)
    axes_row[0].set_title(f'{title_prefix}\n{label_text}', fontsize=10, fontweight='bold')
    axes_row[0].axis('off')
    
    # Column 2: Brain mask
    axes_row[1].imshow(mask, cmap='gray', vmin=0, vmax=255)
    axes_row[1].set_title(f'Brain Mask\nArea: {area_frac:.1f}%', fontsize=10)
    axes_row[1].axis('off')
    
    # Column 3: Masked image (what model sees)
    # Show the actual z-scored image
    foreground = mask > 0
    img_viz = img_display.copy()
    img_viz[~foreground] = 0  # Black background
    
    axes_row[2].imshow(img_viz, cmap='gray', vmin=0, vmax=1)
    axes_row[2].set_title('Masked Image\n(Model Input)', fontsize=10)
    axes_row[2].axis('off')
    
    # Column 4: Quality info
    axes_row[3].axis('off')
    
    # Create quality info text
    quality_text = f"Quality Metrics:\n\n"
    quality_text += f"Area: {area_frac:.1f}%\n"
    quality_text += f"Border: {border_frac:.1f}%\n"
    quality_text += f"Status: {reason}\n\n"
    
    # Add color-coded box
    if quality.get('passed', False):
        box_color = 'lightgreen'
        status_text = "✓ PASSED"
    else:
        box_color = 'lightcoral'
        status_text = "✗ FAILED"
    
    axes_row[3].text(0.1, 0.5, quality_text, fontsize=9, 
                     verticalalignment='center', family='monospace')
    axes_row[3].add_patch(Rectangle((0, 0), 1, 1, 
                                    facecolor=box_color, alpha=0.3, 
                                    transform=axes_row[3].transAxes))
    axes_row[3].text(0.5, 0.9, status_text, fontsize=11, fontweight='bold',
                     horizontalalignment='center', verticalalignment='top',
                     transform=axes_row[3].transAxes)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize robust brain masking results"
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed/kaggle_robust_test',
        help='Directory with processed .npz files'
    )
    parser.add_argument(
        '--num-passed',
        type=int,
        default=6,
        help='Number of passed samples to show'
    )
    parser.add_argument(
        '--num-failed',
        type=int,
        default=6,
        help='Number of failed samples to show'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='robust_masking_visualization.png',
        help='Path to save visualization'
    )
    
    args = parser.parse_args()
    
    visualize_robust_masking(
        processed_dir=args.processed_dir,
        num_passed=args.num_passed,
        num_failed=args.num_failed,
        save_path=args.save_path,
    )


if __name__ == '__main__':
    main()
