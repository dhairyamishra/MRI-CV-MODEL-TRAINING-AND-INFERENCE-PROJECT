#!/usr/bin/env python3
"""
Visualize data augmentation pipeline for both Kaggle and BraTS datasets.

This script shows:
1. Original images
2. Augmented versions (rotation, flip, brightness, etc.)
3. Skull boundary masks
4. Segmentation masks (for BraTS)

Usage:
    python scripts/visualize_augmentations.py --dataset kaggle --num-samples 5
    python scripts/visualize_augmentations.py --dataset brats --num-samples 5
    python scripts/visualize_augmentations.py --dataset both --num-samples 3
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.kaggle_mri_dataset import KaggleBrainMRIDataset
from src.data.brats2d_dataset import BraTS2DSliceDataset
from src.data.transforms import (
    SkullBoundaryMask,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation90,
    RandomIntensityShift,
    RandomIntensityScale,
    RandomGaussianNoise,
)


def visualize_kaggle_augmentations(num_samples: int = 5, save_path: str = None):
    """
    Visualize Kaggle dataset with augmentations.
    
    Args:
        num_samples: Number of samples to visualize
        save_path: Path to save the figure (optional)
    """
    print("\n" + "="*70)
    print("Kaggle Dataset Augmentation Visualization")
    print("="*70)
    
    # Load datasets
    train_dir = project_root / "data" / "processed" / "kaggle" / "train"
    
    # Dataset without augmentation
    dataset = KaggleBrainMRIDataset(
        data_dir=str(train_dir),
        transform=None  # No augmentation
    )
    
    # Create augmentation transforms
    skull_mask = SkullBoundaryMask(threshold_percentile=1.0, kernel_size=5)
    h_flip = RandomHorizontalFlip(p=1.0)  # Always apply for demo
    rotation = RandomRotation90(p=1.0)
    intensity_shift = RandomIntensityShift(shift_range=0.1, p=1.0)
    
    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Loading {num_samples} random samples...\n")
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Sample random indices
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    for idx, sample_idx in enumerate(indices):
        # Get original image
        img_orig, label = dataset[sample_idx]
        
        # Convert to numpy for visualization
        img_orig_np = img_orig.squeeze().numpy()
        
        # Apply augmentations manually
        img_aug = h_flip(rotation(intensity_shift(img_orig)))
        img_aug_np = img_aug.squeeze().numpy()
        
        # Apply skull boundary mask
        img_masked = skull_mask(img_orig)
        img_masked_np = img_masked.squeeze().numpy()
        
        # Get the binary mask used
        gray = img_orig_np
        threshold = np.percentile(gray, 1.0)
        binary_mask = (gray > threshold).astype(np.uint8)
        
        # Plot
        label_text = "Tumor" if label == 1 else "No Tumor"
        
        # Original
        axes[idx, 0].imshow(img_orig_np, cmap='gray')
        axes[idx, 0].set_title(f'Original\n{label_text}', fontsize=10)
        axes[idx, 0].axis('off')
        
        # Augmented
        axes[idx, 1].imshow(img_aug_np, cmap='gray')
        axes[idx, 1].set_title('Augmented\n(rotation, flip, brightness)', fontsize=10)
        axes[idx, 1].axis('off')
        
        # Binary mask
        axes[idx, 2].imshow(binary_mask, cmap='gray')
        axes[idx, 2].set_title('Skull Boundary Mask', fontsize=10)
        axes[idx, 2].axis('off')
        
        # Masked image
        axes[idx, 3].imshow(img_masked_np, cmap='gray')
        axes[idx, 3].set_title('Masked Image\n(background removed)', fontsize=10)
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    else:
        plt.savefig('kaggle_augmentations.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: kaggle_augmentations.png")
    
    plt.show()


def visualize_brats_augmentations(num_samples: int = 5, save_path: str = None):
    """
    Visualize BraTS dataset with augmentations and segmentation masks.
    
    Args:
        num_samples: Number of samples to visualize
        save_path: Path to save the figure (optional)
    """
    print("\n" + "="*70)
    print("BraTS Dataset Augmentation Visualization")
    print("="*70)
    
    # Load dataset
    train_dir = project_root / "data" / "processed" / "brats2d" / "train"
    
    # Dataset without augmentation
    dataset = BraTS2DSliceDataset(
        data_dir=str(train_dir),
        transform=None  # No augmentation
    )
    
    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Loading {num_samples} random samples with tumors...\n")
    
    # Find samples with tumors (non-empty masks)
    tumor_indices = []
    for i in range(min(len(dataset), 1000)):  # Check first 1000
        _, mask = dataset[i]
        if mask.sum() > 0:  # Has tumor
            tumor_indices.append(i)
        if len(tumor_indices) >= num_samples * 3:  # Get more than needed
            break
    
    if len(tumor_indices) < num_samples:
        print(f"Warning: Only found {len(tumor_indices)} samples with tumors")
        num_samples = len(tumor_indices)
    
    # Sample random tumor indices
    indices = np.random.choice(tumor_indices, size=num_samples, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(indices):
        # Get image and mask
        img, mask = dataset[sample_idx]
        
        # Convert to numpy for visualization
        img_np = img.squeeze().numpy()
        mask_np = mask.squeeze().numpy()
        
        # Create overlay (image + mask)
        overlay = img_np.copy()
        overlay = np.stack([overlay, overlay, overlay], axis=-1)
        overlay[mask_np > 0.5] = [1.0, 0.0, 0.0]  # Red for tumor
        
        tumor_pixels = int(mask_np.sum())
        
        # Plot
        # Original image
        axes[idx, 0].imshow(img_np, cmap='gray')
        axes[idx, 0].set_title(f'MRI Image\nTumor pixels: {tumor_pixels}', fontsize=10)
        axes[idx, 0].axis('off')
        
        # Segmentation mask
        axes[idx, 1].imshow(mask_np, cmap='hot')
        axes[idx, 1].set_title('Segmentation Mask', fontsize=10)
        axes[idx, 1].axis('off')
        
        # Overlay
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title('Overlay (Tumor in Red)', fontsize=10)
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    else:
        plt.savefig('brats_augmentations.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: brats_augmentations.png")
    
    plt.show()


def visualize_augmentation_types(dataset_type: str = 'kaggle'):
    """
    Show different types of augmentations applied to a single image.
    
    Args:
        dataset_type: 'kaggle' or 'brats'
    """
    print("\n" + "="*70)
    print(f"Augmentation Types Visualization ({dataset_type.upper()})")
    print("="*70)
    
    # Load dataset
    if dataset_type == 'kaggle':
        train_dir = project_root / "data" / "processed" / "kaggle" / "train"
        dataset = KaggleBrainMRIDataset(
            data_dir=str(train_dir),
            transform=None
        )
    else:  # brats
        train_dir = project_root / "data" / "processed" / "brats2d" / "train"
        dataset = BraTS2DSliceDataset(
            data_dir=str(train_dir),
            transform=None
        )
    
    # Get one sample
    if dataset_type == 'brats':
        # Find a sample with tumor
        for i in range(min(len(dataset), 1000)):
            if dataset_type == 'brats':
                img, mask = dataset[i]
                if mask.sum() > 0:
                    sample_idx = i
                    break
            else:
                sample_idx = i
                break
    else:
        sample_idx = np.random.randint(0, len(dataset))
    
    # Get original
    if dataset_type == 'kaggle':
        img_orig, label = dataset[sample_idx]
    else:
        img_orig, mask = dataset[sample_idx]
    
    img_orig_np = img_orig.squeeze().numpy()
    
    # Create augmentation transforms
    h_flip = RandomHorizontalFlip(p=1.0)
    v_flip = RandomVerticalFlip(p=1.0)
    rotation = RandomRotation90(p=1.0)
    intensity_shift = RandomIntensityShift(shift_range=0.1, p=1.0)
    intensity_scale = RandomIntensityScale(scale_range=(0.9, 1.1), p=1.0)
    noise = RandomGaussianNoise(std=0.01, p=1.0)
    
    # Generate multiple augmented versions
    num_augmentations = 8
    augmented_images = []
    
    print(f"\nGenerating {num_augmentations} augmented versions of sample {sample_idx}...\n")
    
    # Apply different combinations
    augmented_images.append(h_flip(img_orig).squeeze().numpy())
    augmented_images.append(v_flip(img_orig).squeeze().numpy())
    augmented_images.append(rotation(img_orig).squeeze().numpy())
    augmented_images.append(intensity_shift(img_orig).squeeze().numpy())
    augmented_images.append(intensity_scale(img_orig).squeeze().numpy())
    augmented_images.append(noise(img_orig).squeeze().numpy())
    augmented_images.append(h_flip(rotation(img_orig)).squeeze().numpy())
    augmented_images.append(intensity_shift(rotation(img_orig)).squeeze().numpy())
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    # Plot original
    axes[0].imshow(img_orig_np, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot augmented versions
    for i, aug_img in enumerate(augmented_images):
        axes[i+1].imshow(aug_img, cmap='gray')
        axes[i+1].set_title(f'Augmentation {i+1}', fontsize=10)
        axes[i+1].axis('off')
    
    plt.suptitle(f'{dataset_type.upper()} Dataset: Original vs Augmented Versions', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_name = f'{dataset_type}_augmentation_types.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {save_name}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize data augmentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize Kaggle dataset augmentations
  python scripts/visualize_augmentations.py --dataset kaggle --num-samples 5
  
  # Visualize BraTS dataset with segmentation masks
  python scripts/visualize_augmentations.py --dataset brats --num-samples 5
  
  # Visualize both datasets
  python scripts/visualize_augmentations.py --dataset both --num-samples 3
  
  # Show different augmentation types
  python scripts/visualize_augmentations.py --dataset kaggle --show-types
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['kaggle', 'brats', 'both'],
        default='both',
        help='Which dataset to visualize (default: both)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to visualize (default: 5)'
    )
    
    parser.add_argument(
        '--show-types',
        action='store_true',
        help='Show different augmentation types for a single image'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='.',
        help='Directory to save visualizations (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Create save directory if needed
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Data Augmentation Visualization Script")
    print("="*70)
    print(f"\nDataset: {args.dataset}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Save directory: {save_dir}")
    
    # Show augmentation types if requested
    if args.show_types:
        if args.dataset == 'both':
            visualize_augmentation_types('kaggle')
            visualize_augmentation_types('brats')
        else:
            visualize_augmentation_types(args.dataset)
        return
    
    # Visualize datasets
    if args.dataset in ['kaggle', 'both']:
        save_path = save_dir / 'kaggle_augmentations.png'
        visualize_kaggle_augmentations(args.num_samples, str(save_path))
    
    if args.dataset in ['brats', 'both']:
        save_path = save_dir / 'brats_augmentations.png'
        visualize_brats_augmentations(args.num_samples, str(save_path))
    
    print("\n" + "="*70)
    print("✓ Visualization Complete!")
    print("="*70)
    print("\nGenerated files:")
    if args.dataset in ['kaggle', 'both']:
        print(f"  - {save_dir / 'kaggle_augmentations.png'}")
    if args.dataset in ['brats', 'both']:
        print(f"  - {save_dir / 'brats_augmentations.png'}")
    print()


if __name__ == '__main__':
    main()
