"""
Preprocess BraTS 3D volumes to 2D slices.

Converts 3D NIfTI volumes (.nii or .nii.gz) to 2D slices saved as .npz files.
Supports multiple modalities (FLAIR, T1, T1ce, T2) and segmentation masks.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm


def normalize_intensity(
    image: np.ndarray,
    method: str = "zscore",
    clip_percentile: Tuple[float, float] = (1, 99)
) -> np.ndarray:
    """
    Normalize image intensities.
    
    Args:
        image: Input image array
        method: Normalization method ('zscore', 'minmax', or 'percentile')
        clip_percentile: Percentile values for clipping
    
    Returns:
        Normalized image
    """
    if method == "zscore":
        # Z-score normalization (mean=0, std=1)
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            normalized = (image - mean) / std
        else:
            normalized = image - mean
    
    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image)
    
    elif method == "percentile":
        # Percentile-based clipping and normalization
        p_low, p_high = np.percentile(image, clip_percentile)
        image_clipped = np.clip(image, p_low, p_high)
        normalized = (image_clipped - p_low) / (p_high - p_low)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float32)


def load_nifti_volume(file_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Load NIfTI volume and extract metadata.
    
    Args:
        file_path: Path to .nii or .nii.gz file
    
    Returns:
        Tuple of (volume array, metadata dict)
    """
    nii = nib.load(str(file_path))
    volume = nii.get_fdata()
    
    # Extract metadata
    metadata = {
        'shape': volume.shape,
        'affine': nii.affine.tolist(),
        'pixdim': nii.header['pixdim'].tolist(),
    }
    
    return volume, metadata


def has_tumor(mask_slice: np.ndarray, min_pixels: int = 1) -> bool:
    """
    Check if a mask slice contains tumor pixels.
    
    Args:
        mask_slice: 2D mask array
        min_pixels: Minimum number of tumor pixels
    
    Returns:
        True if slice has at least min_pixels tumor pixels
    """
    return np.sum(mask_slice > 0) >= min_pixels


def process_patient(
    patient_dir: Path,
    output_dir: Path,
    modality: str = "flair",
    target_size: Optional[Tuple[int, int]] = None,
    normalize_method: str = "zscore",
    min_tumor_pixels: int = 100,
    save_all_slices: bool = False,
) -> int:
    """
    Process a single patient's data.
    
    Args:
        patient_dir: Directory containing patient's NIfTI files
        output_dir: Output directory for processed slices
        modality: Which modality to use ('flair', 't1', 't1ce', 't2')
        target_size: Resize slices to (H, W), None to keep original
        normalize_method: Intensity normalization method
        min_tumor_pixels: Minimum tumor pixels to keep a slice
        save_all_slices: If True, save all slices; if False, only with tumor
    
    Returns:
        Number of slices saved
    """
    patient_id = patient_dir.name
    
    # Find modality file
    modality_file = None
    for ext in ['.nii.gz', '.nii']:
        pattern = f"*_{modality}{ext}"
        matches = list(patient_dir.glob(pattern))
        if matches:
            modality_file = matches[0]
            break
    
    if modality_file is None:
        return 0
    
    # Find segmentation mask
    seg_file = None
    for ext in ['.nii.gz', '.nii']:
        pattern = f"*_seg{ext}"
        matches = list(patient_dir.glob(pattern))
        if matches:
            seg_file = matches[0]
            break
    
    if seg_file is None:
        return 0
    
    # Load volumes
    try:
        image_volume, image_metadata = load_nifti_volume(modality_file)
        mask_volume, mask_metadata = load_nifti_volume(seg_file)
    except Exception as e:
        print(f"\n✗ Error loading {patient_id}: {e}")
        return 0
    
    # Verify shapes match (alignment check)
    if image_volume.shape != mask_volume.shape:
        print(f"\n⚠️  Shape mismatch {patient_id}: "
              f"image {image_volume.shape} vs mask {mask_volume.shape}")
        return 0
    
    # Normalize image volume
    image_volume = normalize_intensity(image_volume, method=normalize_method)
    
    # Convert mask to binary (any tumor label > 0)
    mask_volume = (mask_volume > 0).astype(np.uint8)
    
    # Process each slice along depth axis (axis 2)
    depth = image_volume.shape[2]
    slices_saved = 0
    
    for slice_idx in range(depth):
        # Extract slice
        image_slice = image_volume[:, :, slice_idx]  # (H, W)
        mask_slice = mask_volume[:, :, slice_idx]    # (H, W)
        
        # Filter out empty slices if requested
        if not save_all_slices and not has_tumor(mask_slice, min_tumor_pixels):
            continue
        
        # Resize if needed
        if target_size is not None:
            try:
                from skimage.transform import resize
                image_slice = resize(
                    image_slice,
                    target_size,
                    order=1,  # Bilinear for images
                    preserve_range=True,
                    anti_aliasing=True
                )
                mask_slice = resize(
                    mask_slice,
                    target_size,
                    order=0,  # Nearest neighbor for masks
                    preserve_range=True,
                    anti_aliasing=False
                )
                mask_slice = (mask_slice > 0.5).astype(np.uint8)
            except ImportError:
                print("\n⚠️  scikit-image not installed, skipping resize")
                pass
        
        # Add channel dimension: (C, H, W)
        image_slice = image_slice[np.newaxis, :, :]  # (1, H, W)
        mask_slice = mask_slice[np.newaxis, :, :]    # (1, H, W)
        
        # Create metadata
        slice_metadata = {
            'patient_id': patient_id,
            'slice_idx': slice_idx,
            'modality': modality,
            'original_shape': image_metadata['shape'],
            'has_tumor': bool(np.sum(mask_slice) > 0),
            'tumor_pixels': int(np.sum(mask_slice)),
            'normalize_method': normalize_method,
            'pixdim': image_metadata['pixdim'],
        }
        
        # Save as .npz
        output_file = output_dir / f"{patient_id}_slice_{slice_idx:03d}.npz"
        np.savez_compressed(
            output_file,
            image=image_slice.astype(np.float32),
            mask=mask_slice.astype(np.uint8),
            metadata=slice_metadata
        )
        
        slices_saved += 1
    
    return slices_saved


def preprocess_brats_dataset(
    input_dir: str = "data/raw/brats2020",
    output_dir: str = "data/processed/brats2d",
    modality: str = "flair",
    target_size: Optional[Tuple[int, int]] = (256, 256),
    normalize_method: str = "zscore",
    min_tumor_pixels: int = 100,
    save_all_slices: bool = False,
    max_patients: Optional[int] = None,
):
    """
    Preprocess entire BraTS dataset.
    
    Args:
        input_dir: Directory containing patient folders
        output_dir: Output directory for processed slices
        modality: Which modality to use
        target_size: Resize slices to (H, W)
        normalize_method: Intensity normalization method
        min_tumor_pixels: Minimum tumor pixels to keep a slice
        save_all_slices: If True, save all slices
        max_patients: Maximum patients to process (for testing)
    """
    print("=" * 70)
    print("BraTS 3D → 2D Slice Preprocessing")
    print("=" * 70)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"✗ Error: Input directory not found: {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Input:  {input_path.absolute()}")
    print(f"  Output: {output_path.absolute()}")
    print(f"  Modality: {modality.upper()}")
    print(f"  Target size: {target_size}")
    print(f"  Normalization: {normalize_method}")
    print(f"  Min tumor pixels: {min_tumor_pixels}")
    print(f"  Save all slices: {save_all_slices}")
    
    # Find all patient directories
    patient_dirs = sorted([
        d for d in input_path.iterdir() 
        if d.is_dir() and d.name.startswith("BraTS")
    ])
    
    # Filter out parent directories (keep only those with .nii files)
    patient_dirs = [d for d in patient_dirs if len(list(d.glob("*.nii*"))) > 0]
    
    if max_patients:
        patient_dirs = patient_dirs[:max_patients]
        print(f"  Processing: First {max_patients} patients (testing mode)")
    
    print(f"\nFound {len(patient_dirs)} patient folders")
    
    if len(patient_dirs) == 0:
        print("✗ No valid patient folders found!")
        sys.exit(1)
    
    # Process each patient
    total_slices = 0
    successful_patients = 0
    
    print("\nProcessing patients...")
    for patient_dir in tqdm(patient_dirs, desc="Patients"):
        try:
            slices_saved = process_patient(
                patient_dir=patient_dir,
                output_dir=output_path,
                modality=modality,
                target_size=target_size,
                normalize_method=normalize_method,
                min_tumor_pixels=min_tumor_pixels,
                save_all_slices=save_all_slices,
            )
            
            if slices_saved > 0:
                total_slices += slices_saved
                successful_patients += 1
        
        except Exception as e:
            print(f"\n✗ Error processing {patient_dir.name}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ Preprocessing Complete!")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  Patients processed: {successful_patients}/{len(patient_dirs)}")
    print(f"  Total slices saved: {total_slices:,}")
    if successful_patients > 0:
        print(f"  Avg slices/patient: {total_slices/successful_patients:.1f}")
    print(f"  Output directory: {output_path.absolute()}")
    
    # Storage info
    output_files = list(output_path.glob("*.npz"))
    if output_files:
        total_size = sum(f.stat().st_size for f in output_files)
        print(f"  Total size: {total_size / (1024**3):.2f} GB")
    
    print("\nNext steps:")
    print("1. Create train/val/test splits:")
    print(f"   python src/data/split_brats.py --input {output_dir}")
    print("2. Visualize samples:")
    print(f"   python scripts/visualize_brats_data.py")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS 3D volumes to 2D slices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all patients with FLAIR modality
  python src/data/preprocess_brats_2d.py \\
      --input data/raw/brats2020 \\
      --output data/processed/brats2d \\
      --modality flair

  # Test with first 10 patients
  python src/data/preprocess_brats_2d.py \\
      --max-patients 10

  # Save all slices (not just tumor slices)
  python src/data/preprocess_brats_2d.py \\
      --save-all-slices

  # Use different normalization
  python src/data/preprocess_brats_2d.py \\
      --normalize minmax
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/brats2020",
        help="Input directory with BraTS patient folders",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/brats2d",
        help="Output directory for processed slices",
    )
    
    parser.add_argument(
        "--modality",
        type=str,
        default="flair",
        choices=["flair", "t1", "t1ce", "t2"],
        help="MRI modality to use (default: flair)",
    )
    
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help="Target size for slices (default: 256 256)",
    )
    
    parser.add_argument(
        "--normalize",
        type=str,
        default="zscore",
        choices=["zscore", "minmax", "percentile"],
        help="Intensity normalization method (default: zscore)",
    )
    
    parser.add_argument(
        "--min-tumor-pixels",
        type=int,
        default=100,
        help="Minimum tumor pixels to keep a slice (default: 100)",
    )
    
    parser.add_argument(
        "--save-all-slices",
        action="store_true",
        help="Save all slices, not just those with tumors",
    )
    
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Maximum number of patients to process (for testing)",
    )
    
    args = parser.parse_args()
    
    preprocess_brats_dataset(
        input_dir=args.input,
        output_dir=args.output,
        modality=args.modality,
        target_size=tuple(args.target_size) if args.target_size else None,
        normalize_method=args.normalize,
        min_tumor_pixels=args.min_tumor_pixels,
        save_all_slices=args.save_all_slices,
        max_patients=args.max_patients,
    )


if __name__ == "__main__":
    main()
