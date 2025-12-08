#!/usr/bin/env python3
"""
Export and compare examples from Kaggle and BraTS datasets.

This script:
1. Loads samples from both datasets
2. Saves them as PNG images for easy viewing
3. Generates detailed metadata JSON files
4. Creates comparison visualizations
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def load_kaggle_sample(npz_path: Path) -> Dict:
    """
    Load a Kaggle dataset sample.
    
    Args:
        npz_path: Path to .npz file
        
    Returns:
        Dictionary with image, label, and metadata
    """
    data = np.load(npz_path, allow_pickle=True)
    
    return {
        'image': data['image'],  # (1, H, W)
        'label': int(data['label']),
        'metadata': data['metadata'].item() if 'metadata' in data else {},
        'source': 'kaggle',
        'filename': npz_path.name,
    }


def load_brats_sample(npz_path: Path) -> Dict:
    """
    Load a BraTS dataset sample.
    
    Args:
        npz_path: Path to .npz file
        
    Returns:
        Dictionary with image, mask, and metadata
    """
    data = np.load(npz_path, allow_pickle=True)
    
    return {
        'image': data['image'],  # (1, H, W)
        'mask': data['mask'] if 'mask' in data else None,  # (1, H, W)
        'metadata': data['metadata'].item() if 'metadata' in data else {},
        'source': 'brats',
        'filename': npz_path.name,
    }


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 255] uint8 for display.
    
    Args:
        image: Input image array
        
    Returns:
        Normalized uint8 image
    """
    # Remove channel dimension if present
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    
    # Normalize to [0, 1]
    img_min = image.min()
    img_max = image.max()
    
    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
    else:
        normalized = np.zeros_like(image)
    
    # Convert to uint8
    return (normalized * 255).astype(np.uint8)


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create an overlay of mask on image.
    
    Args:
        image: Grayscale image (H, W)
        mask: Binary mask (H, W)
        alpha: Transparency of overlay
        
    Returns:
        RGB overlay image
    """
    # Normalize image to uint8
    img_display = normalize_for_display(image)
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
    
    # Remove channel dimension from mask if present
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    
    # Create red overlay for tumor regions
    overlay = img_rgb.copy()
    overlay[mask > 0] = [255, 0, 0]  # Red for tumor
    
    # Blend
    blended = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)
    
    return blended


def save_kaggle_example(sample: Dict, output_dir: Path, index: int):
    """
    Save a Kaggle dataset example.
    
    Args:
        sample: Sample dictionary
        output_dir: Output directory
        index: Sample index
    """
    image = sample['image']
    label = sample['label']
    metadata = sample['metadata']
    
    # Create subdirectory
    sample_dir = output_dir / f"kaggle_{index:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    img_display = normalize_for_display(image)
    cv2.imwrite(str(sample_dir / "image.png"), img_display)
    
    # Save metadata
    metadata_full = {
        'index': index,
        'source': 'kaggle',
        'filename': sample['filename'],
        'label': label,
        'label_name': 'tumor' if label == 1 else 'no_tumor',
        'shape': image.shape,
        'dtype': str(image.dtype),
        'value_range': [float(image.min()), float(image.max())],
        'original_metadata': metadata,
    }
    
    with open(sample_dir / "metadata.json", 'w') as f:
        json.dump(metadata_full, f, indent=2)
    
    return sample_dir


def save_brats_example(sample: Dict, output_dir: Path, index: int):
    """
    Save a BraTS dataset example.
    
    Args:
        sample: Sample dictionary
        output_dir: Output directory
        index: Sample index
    """
    image = sample['image']
    mask = sample['mask']
    metadata = sample['metadata']
    
    # Create subdirectory
    sample_dir = output_dir / f"brats_{index:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    img_display = normalize_for_display(image)
    cv2.imwrite(str(sample_dir / "image.png"), img_display)
    
    # Save mask
    if mask is not None:
        mask_display = normalize_for_display(mask)
        cv2.imwrite(str(sample_dir / "mask.png"), mask_display)
        
        # Save overlay
        overlay = create_overlay(image, mask)
        cv2.imwrite(str(sample_dir / "overlay.png"), overlay)
    
    # Save metadata
    metadata_full = {
        'index': index,
        'source': 'brats',
        'filename': sample['filename'],
        'has_tumor': bool(metadata.get('has_tumor', False)),
        'tumor_pixels': int(metadata.get('tumor_pixels', 0)),
        'patient_id': metadata.get('patient_id', 'unknown'),
        'slice_idx': metadata.get('slice_idx', -1),
        'modality': metadata.get('modality', 'unknown'),
        'shape': image.shape,
        'dtype': str(image.dtype),
        'value_range': [float(image.min()), float(image.max())],
        'normalization': metadata.get('normalize_method', 'unknown'),
        'original_metadata': metadata,
    }
    
    with open(sample_dir / "metadata.json", 'w') as f:
        json.dump(metadata_full, f, indent=2)
    
    return sample_dir


def create_comparison_grid(
    kaggle_samples: List[Dict],
    brats_samples: List[Dict],
    output_path: Path,
    samples_per_dataset: int = 5
):
    """
    Create a comparison grid visualization.
    
    Args:
        kaggle_samples: List of Kaggle samples
        brats_samples: List of BraTS samples
        output_path: Output path for visualization
        samples_per_dataset: Number of samples to show per dataset
    """
    n_kaggle = min(len(kaggle_samples), samples_per_dataset)
    n_brats = min(len(brats_samples), samples_per_dataset)
    
    fig, axes = plt.subplots(2, max(n_kaggle, n_brats), figsize=(4 * max(n_kaggle, n_brats), 8))
    
    if max(n_kaggle, n_brats) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Dataset Comparison: Kaggle vs BraTS', fontsize=16, fontweight='bold')
    
    # Plot Kaggle samples
    for i in range(max(n_kaggle, n_brats)):
        if i < n_kaggle:
            sample = kaggle_samples[i]
            img = normalize_for_display(sample['image'])
            label = sample['label']
            
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f"Kaggle #{i+1}\n{'Tumor' if label == 1 else 'No Tumor'}", 
                                fontsize=10, fontweight='bold')
            axes[0, i].axis('off')
        else:
            axes[0, i].axis('off')
    
    # Plot BraTS samples
    for i in range(max(n_kaggle, n_brats)):
        if i < n_brats:
            sample = brats_samples[i]
            image = sample['image']
            mask = sample['mask']
            metadata = sample['metadata']
            
            # Create overlay if mask exists
            if mask is not None and np.sum(mask) > 0:
                display_img = create_overlay(image, mask)
                has_tumor = True
            else:
                display_img = normalize_for_display(image)
                has_tumor = False
            
            axes[1, i].imshow(display_img, cmap='gray' if not has_tumor else None)
            axes[1, i].set_title(
                f"BraTS #{i+1}\n{metadata.get('patient_id', 'Unknown')}\n"
                f"Slice {metadata.get('slice_idx', -1)} | "
                f"{'Tumor' if has_tumor else 'No Tumor'}",
                fontsize=9
            )
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved comparison grid: {output_path}")


def export_dataset_examples(
    kaggle_dir: str = "data/processed/kaggle",
    brats_dir: str = "data/processed/brats2d",
    output_dir: str = "data/dataset_examples",
    num_samples: int = 20,
    kaggle_with_tumor: int = 20,
    kaggle_without_tumor: int = 20,
    brats_with_tumor: int = 20,
    brats_without_tumor: int = 20,
):
    """
    Export examples from both datasets.
    
    Args:
        kaggle_dir: Kaggle processed data directory
        brats_dir: BraTS processed data directory
        output_dir: Output directory for examples
        num_samples: Total number of samples to export per dataset
        kaggle_with_tumor: Number of Kaggle samples with tumor
        kaggle_without_tumor: Number of Kaggle samples without tumor
        brats_with_tumor: Number of BraTS samples with tumor
        brats_without_tumor: Number of BraTS samples without tumor
    """
    print("=" * 70)
    print("Dataset Examples Export")
    print("=" * 70)
    
    kaggle_path = Path(kaggle_dir)
    brats_path = Path(brats_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    kaggle_output = output_path / "kaggle"
    brats_output = output_path / "brats"
    kaggle_output.mkdir(exist_ok=True)
    brats_output.mkdir(exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Kaggle dir: {kaggle_path}")
    print(f"  BraTS dir:  {brats_path}")
    print(f"  Output dir: {output_path}")
    print(f"  Samples per dataset: {num_samples}")
    
    # ========== Process Kaggle Dataset ==========
    print("\n[Processing Kaggle Dataset]")
    
    kaggle_samples = []
    
    if kaggle_path.exists():
        # Find files
        yes_files = sorted(kaggle_path.glob("yes_*.npz"))
        no_files = sorted(kaggle_path.glob("no_*.npz"))
        
        print(f"  Found: {len(yes_files)} tumor, {len(no_files)} no-tumor samples")
        
        # Sample with tumor
        for i, npz_file in enumerate(yes_files[:kaggle_with_tumor]):
            sample = load_kaggle_sample(npz_file)
            kaggle_samples.append(sample)
            save_kaggle_example(sample, kaggle_output, i)
        
        # Sample without tumor
        for i, npz_file in enumerate(no_files[:kaggle_without_tumor]):
            sample = load_kaggle_sample(npz_file)
            kaggle_samples.append(sample)
            save_kaggle_example(sample, kaggle_output, kaggle_with_tumor + i)
        
        print(f"  [OK] Exported {len(kaggle_samples)} Kaggle samples")
    else:
        print(f"  ⚠️  Kaggle directory not found: {kaggle_path}")
    
    # ========== Process BraTS Dataset ==========
    print("\n[Processing BraTS Dataset]")
    
    brats_samples = []
    
    if brats_path.exists():
        # Find all files
        all_files = sorted(brats_path.glob("*.npz"))
        
        print(f"  Found: {len(all_files)} total slices")
        
        # Separate by tumor presence
        tumor_files = []
        no_tumor_files = []
        
        for npz_file in all_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                metadata = data['metadata'].item() if 'metadata' in data else {}
                
                if metadata.get('has_tumor', False):
                    tumor_files.append(npz_file)
                else:
                    no_tumor_files.append(npz_file)
            except Exception:
                continue
        
        print(f"  Categorized: {len(tumor_files)} tumor, {len(no_tumor_files)} no-tumor")
        
        # Sample with tumor
        for i, npz_file in enumerate(tumor_files[:brats_with_tumor]):
            sample = load_brats_sample(npz_file)
            brats_samples.append(sample)
            save_brats_example(sample, brats_output, i)
        
        # Sample without tumor
        for i, npz_file in enumerate(no_tumor_files[:brats_without_tumor]):
            sample = load_brats_sample(npz_file)
            brats_samples.append(sample)
            save_brats_example(sample, brats_output, brats_with_tumor + i)
        
        print(f"  [OK] Exported {len(brats_samples)} BraTS samples")
    else:
        print(f"  ⚠️  BraTS directory not found: {brats_path}")
    
    # ========== Create Comparison Visualization ==========
    if kaggle_samples and brats_samples:
        print("\n[Creating Comparison Visualization]")
        comparison_path = output_path / "dataset_comparison.png"
        create_comparison_grid(kaggle_samples, brats_samples, comparison_path)
    
    # ========== Create Summary ==========
    print("\n" + "=" * 70)
    print("[OK] Export Complete!")
    print("=" * 70)
    
    summary = {
        'kaggle': {
            'total_exported': len(kaggle_samples),
            'with_tumor': sum(1 for s in kaggle_samples if s['label'] == 1),
            'without_tumor': sum(1 for s in kaggle_samples if s['label'] == 0),
            'output_dir': str(kaggle_output.absolute()),
        },
        'brats': {
            'total_exported': len(brats_samples),
            'with_tumor': sum(1 for s in brats_samples if s['metadata'].get('has_tumor', False)),
            'without_tumor': sum(1 for s in brats_samples if not s['metadata'].get('has_tumor', False)),
            'output_dir': str(brats_output.absolute()),
        },
        'comparison_visualization': str((output_path / "dataset_comparison.png").absolute()) if kaggle_samples and brats_samples else None,
    }
    
    # Save summary
    with open(output_path / "export_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary:")
    print(f"  Kaggle: {summary['kaggle']['total_exported']} samples "
          f"({summary['kaggle']['with_tumor']} tumor, {summary['kaggle']['without_tumor']} no-tumor)")
    print(f"  BraTS:  {summary['brats']['total_exported']} samples "
          f"({summary['brats']['with_tumor']} tumor, {summary['brats']['without_tumor']} no-tumor)")
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"  - kaggle/: Individual Kaggle samples with metadata")
    print(f"  - brats/: Individual BraTS samples with metadata and overlays")
    print(f"  - dataset_comparison.png: Side-by-side comparison")
    print(f"  - export_summary.json: Export statistics")
    
    print("\n📁 Explore the examples:")
    print(f"   {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Export and compare examples from Kaggle and BraTS datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export 10 samples from each dataset
  python scripts/export_dataset_examples.py
  
  # Export more samples
  python scripts/export_dataset_examples.py --num-samples 20
  
  # Custom distribution
  python scripts/export_dataset_examples.py \\
      --kaggle-with-tumor 8 \\
      --kaggle-without-tumor 2 \\
      --brats-with-tumor 8 \\
      --brats-without-tumor 2
        """
    )
    
    parser.add_argument(
        "--kaggle-dir",
        type=str,
        default="data/processed/kaggle",
        help="Kaggle processed data directory"
    )
    
    parser.add_argument(
        "--brats-dir",
        type=str,
        default="data/processed/brats2d",
        help="BraTS processed data directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dataset_examples",
        help="Output directory for examples"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Total samples per dataset (default: 20)"
    )
    
    parser.add_argument(
        "--kaggle-with-tumor",
        type=int,
        default=20,
        help="Kaggle samples with tumor (default: 20)"
    )
    
    parser.add_argument(
        "--kaggle-without-tumor",
        type=int,
        default=20,
        help="Kaggle samples without tumor (default: 20)"
    )
    
    parser.add_argument(
        "--brats-with-tumor",
        type=int,
        default=20,
        help="BraTS samples with tumor (default: 20)"
    )
    
    parser.add_argument(
        "--brats-without-tumor",
        type=int,
        default=20,
        help="BraTS samples without tumor (default: 20)"
    )
    
    args = parser.parse_args()
    
    export_dataset_examples(
        kaggle_dir=args.kaggle_dir,
        brats_dir=args.brats_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        kaggle_with_tumor=args.kaggle_with_tumor,
        kaggle_without_tumor=args.kaggle_without_tumor,
        brats_with_tumor=args.brats_with_tumor,
        brats_without_tumor=args.brats_without_tumor,
    )


if __name__ == "__main__":
    main()
