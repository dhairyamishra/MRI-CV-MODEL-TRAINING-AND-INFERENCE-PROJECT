#!/usr/bin/env python3
"""
Download Kaggle Brain MRI dataset for brain tumor detection.
Dataset: navoneel/brain-mri-images-for-brain-tumor-detection
"""

import os
import shutil
import sys
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("Error: kagglehub not installed. Install with: pip install kaggle")
    sys.exit(1)


def download_kaggle_brain_mri(target_dir: str = "data/raw/kaggle_brain_mri"):
    """
    Download Kaggle Brain MRI dataset.
    
    Args:
        target_dir: Target directory to store the dataset
    """
    print("=" * 60)
    print("Downloading Kaggle Brain MRI Dataset")
    print("=" * 60)
    
    # Dataset handle
    dataset_handle = "navoneel/brain-mri-images-for-brain-tumor-detection"
    
    print(f"\nDataset: {dataset_handle}")
    print(f"Target directory: {target_dir}")
    
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download dataset
        print("\n[1/3] Downloading dataset from Kaggle...")
        print("(This may take a few minutes depending on your connection)")
        
        download_path = kagglehub.dataset_download(dataset_handle)
        print(f"[OK] Dataset downloaded to: {download_path}")
        
        # Copy to target directory
        print(f"\n[2/3] Copying dataset to {target_dir}...")
        
        download_path = Path(download_path)
        
        # Copy all files from download location to target
        if download_path.exists():
            for item in download_path.iterdir():
                dest = target_path / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                    print(f"  Copied: {item.name}")
                elif item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                    print(f"  Copied directory: {item.name}")
        
        # Verify download
        print("\n[3/3] Verifying download...")
        
        yes_dir = target_path / "yes"
        no_dir = target_path / "no"
        
        if yes_dir.exists() and no_dir.exists():
            yes_count = len(list(yes_dir.glob("*.jpg")))
            no_count = len(list(no_dir.glob("*.jpg")))
            
            print(f"[OK] Found {yes_count} images with tumors (yes/)")
            print(f"[OK] Found {no_count} images without tumors (no/)")
            print(f"[OK] Total: {yes_count + no_count} images")
        else:
            print("⚠ Warning: Expected 'yes/' and 'no/' directories not found")
            print(f"  Contents of {target_dir}:")
            for item in target_path.iterdir():
                print(f"    - {item.name}")
        
        print("\n" + "=" * 60)
        print("[OK] Download completed successfully!")
        print("=" * 60)
        print(f"\nDataset location: {target_path.absolute()}")
        print("\nNext steps:")
        print("1. Verify the data structure")
        print("2. Run preprocessing to convert to .npz format")
        print("3. Create train/val/test splits")
        
        return str(target_path.absolute())
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have Kaggle API credentials configured")
        print("   - Go to https://www.kaggle.com/account")
        print("   - Click 'Create New API Token'")
        print("   - Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)")
        print("   - Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("2. Check your internet connection")
        print("3. Verify the dataset is still available on Kaggle")
        sys.exit(1)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Kaggle Brain MRI dataset"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/raw/kaggle_brain_mri",
        help="Target directory for dataset (default: data/raw/kaggle_brain_mri)",
    )
    
    args = parser.parse_args()
    
    download_kaggle_brain_mri(args.target_dir)


if __name__ == "__main__":
    main()
