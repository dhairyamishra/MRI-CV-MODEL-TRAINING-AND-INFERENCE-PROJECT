#!/usr/bin/env python3
"""
Download BraTS (Brain Tumor Segmentation) dataset from Kaggle.

Supports BraTS 2020 and 2021 datasets with automatic extraction and verification.
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("Error: kagglehub not installed. Install with: pip install kagglehub")
    sys.exit(1)


# Dataset configurations
BRATS_DATASETS = {
    "2020": {
        "kaggle_id": "awsaf49/brats20-dataset-training-validation",
        "expected_folders": ["BraTS2020_TrainingData", "BraTS2020_ValidationData"],
        "num_training": 369,
        "num_validation": 125,
    },
    "2021": {
        "kaggle_id": "dschettler8845/brats-2021-task1",
        "expected_folders": ["BraTS2021_Training_Data"],
        "num_training": 1251,
        "num_validation": 219,
    },
}


def download_brats_dataset(
    version: str = "2020",
    output_dir: str = "data/raw/brats2020",
    force: bool = False,
):
    """
    Download BraTS dataset from Kaggle.
    
    Args:
        version: BraTS version ('2020' or '2021')
        output_dir: Target directory for dataset
        force: Force re-download even if data exists
    """
    print("=" * 70)
    print(f"Downloading BraTS {version} Dataset from Kaggle")
    print("=" * 70)
    
    # Validate version
    if version not in BRATS_DATASETS:
        print(f"Error: Unsupported version '{version}'")
        print(f"Supported versions: {', '.join(BRATS_DATASETS.keys())}")
        sys.exit(1)
    
    dataset_config = BRATS_DATASETS[version]
    kaggle_id = dataset_config["kaggle_id"]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDataset: BraTS {version}")
    print(f"Kaggle ID: {kaggle_id}")
    print(f"Output directory: {output_path.absolute()}")
    
    # Check if data already exists
    if not force and output_path.exists():
        existing_folders = list(output_path.glob("BraTS*"))
        if len(existing_folders) > 0:
            print(f"\n⚠️  Found {len(existing_folders)} existing BraTS folders")
            print("Use --force to re-download")
            
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Download cancelled.")
                return str(output_path.absolute())
    
    try:
        # Download dataset using kagglehub
        print("\n[1/3] Downloading dataset from Kaggle...")
        print("(This may take 10-30 minutes depending on your connection)")
        print("Dataset size: ~20 GB for BraTS 2020, ~70 GB for BraTS 2021")
        
        download_path = kagglehub.dataset_download(kaggle_id)
        print(f"[OK] Dataset downloaded to: {download_path}")
        
        # Copy to target directory
        print(f"\n[2/3] Organizing dataset to {output_dir}...")
        
        download_path = Path(download_path)
        
        # Find and copy BraTS folders
        copied_count = 0
        for item in download_path.rglob("BraTS*"):
            if item.is_dir():
                dest = output_path / item.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
                copied_count += 1
                if copied_count % 10 == 0:
                    print(f"  Copied {copied_count} patient folders...")
        
        print(f"[OK] Copied {copied_count} patient folders")
        
        # Verify download
        print("\n[3/3] Verifying dataset...")
        
        patient_folders = sorted(output_path.glob("BraTS*"))
        
        if len(patient_folders) == 0:
            print("✗ Error: No BraTS patient folders found!")
            print(f"  Expected folders like: {dataset_config['expected_folders']}")
            sys.exit(1)
        
        print(f"[OK] Found {len(patient_folders)} patient folders")
        
        # Verify a sample patient folder
        sample_folder = patient_folders[0]
        modalities = list(sample_folder.glob("*.nii.gz"))
        
        print(f"\nSample patient: {sample_folder.name}")
        print(f"  Files found: {len(modalities)}")
        
        expected_files = ["flair", "t1", "t1ce", "t2", "seg"]
        found_files = []
        for modality in expected_files:
            matching = [f for f in modalities if modality in f.name.lower()]
            if matching:
                found_files.append(modality)
                print(f"  [OK] {modality.upper()}: {matching[0].name}")
            else:
                print(f"  ✗ {modality.upper()}: NOT FOUND")
        
        if len(found_files) < 4:  # At least 4 modalities (seg is optional for validation)
            print("\n⚠️  Warning: Some modalities are missing")
        
        # Summary
        print("\n" + "=" * 70)
        print("[OK] Download completed successfully!")
        print("=" * 70)
        print(f"\nDataset location: {output_path.absolute()}")
        print(f"Total patients: {len(patient_folders)}")
        print(f"Expected training patients: {dataset_config['num_training']}")
        
        if len(patient_folders) < dataset_config['num_training']:
            print(f"\n⚠️  Warning: Found fewer patients than expected")
            print(f"   Expected: {dataset_config['num_training']}")
            print(f"   Found: {len(patient_folders)}")
        
        print("\nNext steps:")
        print("1. Verify the data structure")
        print("2. Run preprocessing to extract 2D slices:")
        print(f"   python src/data/preprocess_brats_2d.py \\")
        print(f"       --input {output_dir} \\")
        print(f"       --output data/processed/brats2d \\")
        print(f"       --modality flair")
        print("3. Create train/val/test splits")
        print("4. Visualize samples with: python scripts/visualize_brats_data.py")
        
        return str(output_path.absolute())
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have Kaggle API credentials configured")
        print("   - Go to https://www.kaggle.com/account")
        print("   - Click 'Create New API Token'")
        print("   - Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)")
        print("   - Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("2. Check your internet connection")
        print("3. Verify you have enough disk space (~25 GB for BraTS 2020)")
        print("4. Try downloading manually from:")
        print(f"   https://www.kaggle.com/datasets/{kaggle_id}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download BraTS dataset from Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download BraTS 2020 (default)
  python scripts/download_brats_data.py
  
  # Download BraTS 2021
  python scripts/download_brats_data.py --version 2021 --output data/raw/brats2021
  
  # Force re-download
  python scripts/download_brats_data.py --force

Supported versions:
  - 2020: BraTS 2020 (369 training, 125 validation) ~20 GB
  - 2021: BraTS 2021 (1,251 training, 219 validation) ~70 GB

Note: Requires Kaggle API credentials configured.
        """
    )
    
    parser.add_argument(
        "--version",
        type=str,
        default="2020",
        choices=["2020", "2021"],
        help="BraTS version to download (default: 2020)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/brats2020",
        help="Output directory for dataset (default: data/raw/brats2020)",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists",
    )
    
    args = parser.parse_args()
    
    # Adjust default output directory based on version
    if args.output == "data/raw/brats2020" and args.version != "2020":
        args.output = f"data/raw/brats{args.version}"
    
    download_brats_dataset(
        version=args.version,
        output_dir=args.output,
        force=args.force,
    )


if __name__ == "__main__":
    main()
