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


def save_kaggle_example(sample: Dict, output_dir: Path, index: int, has_tumor: bool):
    """
    Save a Kaggle dataset example.
    
    Args:
        sample: Sample dictionary
        output_dir: Output directory
        index: Sample index
        has_tumor: Whether sample has tumor
    """
    image = sample['image']
    label = sample['label']
    metadata = sample['metadata']
    
    # Create subdirectory based on tumor presence
    tumor_dir = output_dir / ("yes_tumor" if has_tumor else "no_tumor")
    tumor_dir.mkdir(parents=True, exist_ok=True)
    
    sample_dir = tumor_dir / f"sample_{index:03d}"
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


def save_brats_example(sample: Dict, output_dir: Path, index: int, has_tumor: bool):
    """
    Save a BraTS dataset example.
    
    Args:
        sample: Sample dictionary
        output_dir: Output directory
        index: Sample index
        has_tumor: Whether sample has tumor
    """
    image = sample['image']
    mask = sample['mask']
    metadata = sample['metadata']
    
    # Create subdirectory based on tumor presence
    tumor_dir = output_dir / ("yes_tumor" if has_tumor else "no_tumor")
    tumor_dir.mkdir(parents=True, exist_ok=True)
    
    sample_dir = tumor_dir / f"sample_{index:03d}"
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
    kaggle_dir: str = "data/raw/kaggle_brain_mri",
    brats_dir: str = "data/raw/brats2020",
    output_dir: str = "data/dataset_examples",
    num_samples: int = 20,
    kaggle_with_tumor: int = 10,
    kaggle_without_tumor: int = 10,
    brats_with_tumor: int = 10,
    brats_without_tumor: int = 10,
):
    """
    Export examples from RAW datasets (before preprocessing).
    
    Args:
        kaggle_dir: Kaggle RAW data directory (JPG files)
        brats_dir: BraTS RAW data directory (NIfTI files)
        output_dir: Output directory for examples
        num_samples: Total number of samples to export per dataset
        kaggle_with_tumor: Number of Kaggle samples with tumor
        kaggle_without_tumor: Number of Kaggle samples without tumor
        brats_with_tumor: Number of BraTS samples with tumor
        brats_without_tumor: Number of BraTS samples without tumor
    """
    print("=" * 70)
    print("Dataset Examples Export (from RAW data)")
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
    print(f"  Kaggle RAW dir: {kaggle_path}")
    print(f"  BraTS RAW dir:  {brats_path}")
    print(f"  Output dir:     {output_path}")
    
    # ========== Process Kaggle Dataset ==========
    print("\n[Processing Kaggle Dataset from RAW JPG files]")
    
    kaggle_samples = []
    
    if kaggle_path.exists():
        # Find RAW JPG files in yes/ and no/ subdirectories
        yes_dir = kaggle_path / "yes"
        no_dir = kaggle_path / "no"
        
        yes_files = sorted(yes_dir.glob("*.jpg")) if yes_dir.exists() else []
        no_files = sorted(no_dir.glob("*.jpg")) if no_dir.exists() else []
        
        print(f"  Found: {len(yes_files)} tumor, {len(no_files)} no-tumor RAW JPG files")
        
        # Sample with tumor - convert JPG to npz format for compatibility
        print(f"  Exporting {kaggle_with_tumor} tumor samples...")
        for i, jpg_file in enumerate(yes_files[:kaggle_with_tumor]):
            # Load JPG and convert to npz-like format
            img = cv2.imread(str(jpg_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize to 256x256
                img = cv2.resize(img, (256, 256))
                # Normalize to [0, 1] and add channel dimension
                img_normalized = img.astype(np.float32) / 255.0
                img_normalized = np.expand_dims(img_normalized, axis=0)  # (1, H, W)
                
                sample = {
                    'image': img_normalized,
                    'label': 1,
                    'metadata': {'source': 'raw_jpg', 'original_file': jpg_file.name},
                    'source': 'kaggle',
                    'filename': jpg_file.name,
                }
                kaggle_samples.append(sample)
                save_kaggle_example(sample, kaggle_output, i, has_tumor=True)
        
        # Sample without tumor
        print(f"  Exporting {kaggle_without_tumor} no-tumor samples...")
        for i, jpg_file in enumerate(no_files[:kaggle_without_tumor]):
            img = cv2.imread(str(jpg_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                img_normalized = img.astype(np.float32) / 255.0
                img_normalized = np.expand_dims(img_normalized, axis=0)
                
                sample = {
                    'image': img_normalized,
                    'label': 0,
                    'metadata': {'source': 'raw_jpg', 'original_file': jpg_file.name},
                    'source': 'kaggle',
                    'filename': jpg_file.name,
                }
                kaggle_samples.append(sample)
                save_kaggle_example(sample, kaggle_output, i, has_tumor=False)
        
        print(f"  [OK] Exported {len(kaggle_samples)} Kaggle samples")
    else:
        print(f"  ⚠️  Kaggle directory not found: {kaggle_path}")
    
    # ========== Process BraTS Dataset ==========
    print("\n[Processing BraTS Dataset from RAW NIfTI files]")
    
    brats_samples = []
    
    if brats_path.exists():
        # Find patient directories
        patient_dirs = sorted([d for d in brats_path.iterdir() if d.is_dir() and d.name.startswith('BraTS')])
        
        print(f"  Found: {len(patient_dirs)} patient directories")
        print(f"  Extracting slices with/without tumors...")
        
        # We'll extract middle slices from volumes
        tumor_samples_collected = 0
        no_tumor_samples_collected = 0
        
        for patient_dir in tqdm(patient_dirs, desc="  Scanning patients"):
            if tumor_samples_collected >= brats_with_tumor and no_tumor_samples_collected >= brats_without_tumor:
                break
            
            try:
                # Find FLAIR and segmentation files
                flair_file = None
                seg_file = None
                
                for ext in ['.nii.gz', '.nii']:
                    if flair_file is None:
                        matches = list(patient_dir.glob(f"*_flair{ext}"))
                        if matches:
                            flair_file = matches[0]
                    if seg_file is None:
                        matches = list(patient_dir.glob(f"*_seg{ext}"))
                        if matches:
                            seg_file = matches[0]
                
                if flair_file is None or seg_file is None:
                    continue
                
                # Load volumes (only import nibabel when needed)
                import nibabel as nib
                flair_vol = nib.load(str(flair_file)).get_fdata()
                seg_vol = nib.load(str(seg_file)).get_fdata()
                
                # Get middle slice
                mid_slice = flair_vol.shape[2] // 2
                
                # Try a few slices around the middle
                for slice_offset in [0, -5, 5, -10, 10]:
                    slice_idx = mid_slice + slice_offset
                    if slice_idx < 0 or slice_idx >= flair_vol.shape[2]:
                        continue
                    
                    img_slice = flair_vol[:, :, slice_idx]
                    seg_slice = seg_vol[:, :, slice_idx]
                    
                    has_tumor_flag = np.sum(seg_slice > 0) >= 100
                    
                    # Collect tumor sample
                    if has_tumor_flag and tumor_samples_collected < brats_with_tumor:
                        # Normalize and resize
                        img_min, img_max = img_slice.min(), img_slice.max()
                        if img_max > img_min:
                            img_norm = ((img_slice - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            img_norm = np.zeros_like(img_slice, dtype=np.uint8)
                        
                        img_norm = cv2.resize(img_norm, (256, 256))
                        seg_norm = cv2.resize((seg_slice > 0).astype(np.uint8) * 255, (256, 256), interpolation=cv2.INTER_NEAREST)
                        
                        # Convert to npz-like format
                        img_final = (img_norm.astype(np.float32) / 255.0)
                        img_final = np.expand_dims(img_final, axis=0)
                        seg_final = np.expand_dims(seg_norm, axis=0)
                        
                        sample = {
                            'image': img_final,
                            'mask': seg_final,
                            'metadata': {
                                'source': 'raw_nifti',
                                'patient_id': patient_dir.name,
                                'slice_idx': slice_idx,
                                'modality': 'flair',
                                'has_tumor': True,
                                'tumor_pixels': int(np.sum(seg_norm > 0))
                            },
                            'source': 'brats',
                            'filename': f"{patient_dir.name}_slice{slice_idx:03d}.npz",
                        }
                        brats_samples.append(sample)
                        save_brats_example(sample, brats_output, tumor_samples_collected, has_tumor=True)
                        tumor_samples_collected += 1
                        break
                    
                    # Collect no-tumor sample
                    elif not has_tumor_flag and no_tumor_samples_collected < brats_without_tumor:
                        img_min, img_max = img_slice.min(), img_slice.max()
                        if img_max > img_min:
                            img_norm = ((img_slice - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            img_norm = np.zeros_like(img_slice, dtype=np.uint8)
                        
                        img_norm = cv2.resize(img_norm, (256, 256))
                        
                        img_final = (img_norm.astype(np.float32) / 255.0)
                        img_final = np.expand_dims(img_final, axis=0)
                        
                        sample = {
                            'image': img_final,
                            'mask': None,
                            'metadata': {
                                'source': 'raw_nifti',
                                'patient_id': patient_dir.name,
                                'slice_idx': slice_idx,
                                'modality': 'flair',
                                'has_tumor': False,
                                'tumor_pixels': 0
                            },
                            'source': 'brats',
                            'filename': f"{patient_dir.name}_slice{slice_idx:03d}.npz",
                        }
                        brats_samples.append(sample)
                        save_brats_example(sample, brats_output, no_tumor_samples_collected, has_tumor=False)
                        no_tumor_samples_collected += 1
                        break
                        
            except Exception as e:
                print(f"    ⚠️  Failed to process {patient_dir.name}: {e}")
                continue
        
        print(f"  [OK] Exported {len(brats_samples)} BraTS samples ({tumor_samples_collected} tumor, {no_tumor_samples_collected} no-tumor)")
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
        default="data/raw/kaggle_brain_mri",
        help="Kaggle RAW data directory (JPG files)"
    )
    
    parser.add_argument(
        "--brats-dir",
        type=str,
        default="data/raw/brats2020",
        help="BraTS RAW data directory (NIfTI files)"
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
        default=10,
        help="Kaggle samples with tumor (default: 10)"
    )
    
    parser.add_argument(
        "--kaggle-without-tumor",
        type=int,
        default=10,
        help="Kaggle samples without tumor (default: 10)"
    )
    
    parser.add_argument(
        "--brats-with-tumor",
        type=int,
        default=10,
        help="BraTS samples with tumor (default: 10)"
    )
    
    parser.add_argument(
        "--brats-without-tumor",
        type=int,
        default=10,
        help="BraTS samples without tumor (default: 10)"
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
