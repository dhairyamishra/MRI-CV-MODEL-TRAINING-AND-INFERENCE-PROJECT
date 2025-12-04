#!/usr/bin/env python3
"""
Verify and fix BraTS dataset structure.

Checks if patient folders are properly organized and moves them if needed.
"""

import argparse
import shutil
from pathlib import Path


def verify_brats_structure(data_dir: str = "data/raw/brats2020", fix: bool = False):
    """
    Verify BraTS dataset structure and optionally fix it.
    
    Args:
        data_dir: Directory containing BraTS data
        fix: If True, reorganize files to correct structure
    """
    print("=" * 70)
    print("Verifying BraTS Dataset Structure")
    print("=" * 70)
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"✗ Error: Directory not found: {data_path}")
        return
    
    print(f"\nDataset directory: {data_path.absolute()}")
    
    # Find all potential patient folders
    all_folders = list(data_path.rglob("BraTS*"))
    print(f"\nFound {len(all_folders)} BraTS folders")
    
    # Separate patient folders from parent folders
    patient_folders = []
    parent_folders = []
    
    for folder in all_folders:
        if folder.is_dir():
            # Check if it contains NIfTI files (.nii or .nii.gz)
            nifti_files = list(folder.glob("*.nii.gz")) + list(folder.glob("*.nii"))
            if len(nifti_files) > 0:
                patient_folders.append(folder)
            else:
                # Check if it contains subfolders with NIfTI files
                has_patient_subfolders = any(
                    len(list(subfolder.glob("*.nii.gz"))) + len(list(subfolder.glob("*.nii"))) > 0
                    for subfolder in folder.iterdir()
                    if subfolder.is_dir()
                )
                if has_patient_subfolders:
                    parent_folders.append(folder)
    
    print(f"\nStructure analysis:")
    print(f"  Patient folders (with .nii/.nii.gz files): {len(patient_folders)}")
    print(f"  Parent folders (containing patients): {len(parent_folders)}")
    
    # Check a sample patient folder
    if patient_folders:
        sample = patient_folders[0]
        nifti_files = list(sample.glob("*.nii.gz")) + list(sample.glob("*.nii"))
        
        print(f"\nSample patient folder: {sample.name}")
        print(f"  Location: {sample.parent}")
        print(f"  Files: {len(nifti_files)}")
        
        modalities = ['flair', 't1', 't1ce', 't2', 'seg']
        for mod in modalities:
            matching = [f for f in nifti_files if mod in f.name.lower()]
            if matching:
                print(f"  ✓ {mod.upper():6s}: {matching[0].name}")
            else:
                print(f"  ✗ {mod.upper():6s}: NOT FOUND")
    
    # Check if reorganization is needed
    needs_reorganization = False
    if parent_folders:
        print(f"\n⚠️  Found {len(parent_folders)} parent folder(s) containing patient data")
        print("   Patient folders need to be moved to the root level")
        needs_reorganization = True
        
        for parent in parent_folders[:3]:  # Show first 3
            print(f"   - {parent.name}")
            subfolders = [f for f in parent.iterdir() if f.is_dir()]
            print(f"     Contains {len(subfolders)} patient folders")
    
    # Fix structure if requested
    if needs_reorganization and fix:
        print("\n" + "=" * 70)
        print("Reorganizing Dataset Structure")
        print("=" * 70)
        
        moved_count = 0
        for parent in parent_folders:
            print(f"\nProcessing: {parent.name}")
            
            # Find all patient folders inside
            patient_subfolders = [
                f for f in parent.iterdir()
                if f.is_dir() and (len(list(f.glob("*.nii.gz"))) + len(list(f.glob("*.nii")))) > 0
            ]
            
            print(f"  Found {len(patient_subfolders)} patient folders")
            
            # Move each patient folder to root
            for patient_folder in patient_subfolders:
                dest = data_path / patient_folder.name
                
                if dest.exists():
                    print(f"  ⚠️  Skipping {patient_folder.name} (already exists)")
                    continue
                
                try:
                    shutil.move(str(patient_folder), str(dest))
                    moved_count += 1
                    if moved_count % 10 == 0:
                        print(f"  Moved {moved_count} folders...")
                except Exception as e:
                    print(f"  ✗ Error moving {patient_folder.name}: {e}")
            
            # Remove empty parent folder
            try:
                if not any(parent.iterdir()):
                    parent.rmdir()
                    print(f"  Removed empty parent folder: {parent.name}")
            except:
                pass
        
        print(f"\n✓ Moved {moved_count} patient folders to root level")
        
        # Re-verify
        print("\nRe-verifying structure...")
        verify_brats_structure(data_dir, fix=False)
        return
    
    # Summary
    print("\n" + "=" * 70)
    if patient_folders and not needs_reorganization:
        print("✓ Dataset structure is correct!")
        print(f"✓ Found {len(patient_folders)} valid patient folders")
    elif needs_reorganization and not fix:
        print("⚠️  Dataset needs reorganization")
        print(f"   Run with --fix to reorganize automatically")
    else:
        print("✗ No valid patient folders found")
    print("=" * 70)
    
    if patient_folders and not needs_reorganization:
        print("\nNext steps:")
        print("1. Run preprocessing:")
        print(f"   python src/data/preprocess_brats_2d.py \\")
        print(f"       --input {data_dir} \\")
        print(f"       --output data/processed/brats2d \\")
        print(f"       --modality flair")


def main():
    parser = argparse.ArgumentParser(
        description="Verify and fix BraTS dataset structure"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/brats2020",
        help="Input directory with BraTS data",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix structure by moving patient folders to root level",
    )
    
    args = parser.parse_args()
    
    verify_brats_structure(args.input, args.fix)


if __name__ == "__main__":
    main()
