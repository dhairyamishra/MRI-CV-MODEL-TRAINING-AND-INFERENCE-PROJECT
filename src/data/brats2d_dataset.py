"""
PyTorch Dataset for BraTS 2D slices.

Loads preprocessed .npz files containing image slices and segmentation masks.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTS2DSliceDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2D segmentation slices.
    
    Loads preprocessed .npz files containing:
    - image: (1, H, W) normalized MRI slice
    - mask: (1, H, W) binary segmentation mask
    - metadata: dict with patient_id, slice_idx, etc.
    
    Args:
        data_dir: Directory containing .npz files
        transform: Optional transform to apply to images and masks
        return_metadata: If True, return (image, mask, metadata)
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_metadata = return_metadata
        
        # Get all .npz files
        self.file_paths = sorted(self.data_dir.glob("*.npz"))
        
        if not self.file_paths:
            raise ValueError(f"No .npz files found in {self.data_dir}")
        
        # Load metadata for quick access
        self.metadata_list = []
        for file_path in self.file_paths:
            data = np.load(file_path, allow_pickle=True)
            self.metadata_list.append(data['metadata'].item())
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, ...]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            If return_metadata=False: (image, mask)
            If return_metadata=True: (image, mask, metadata)
            
            - image: torch.Tensor of shape (1, H, W)
            - mask: torch.Tensor of shape (1, H, W)
            - metadata: dict with sample information
        """
        # Load data
        file_path = self.file_paths[idx]
        data = np.load(file_path, allow_pickle=True)
        
        # Extract image and mask
        image = data['image']  # (1, H, W)
        mask = data['mask']    # (1, H, W)
        metadata = data['metadata'].item()
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        # Apply transforms if provided
        if self.transform is not None:
            # For segmentation, we need to transform both image and mask together
            # The transform should handle this appropriately
            image, mask = self.transform(image, mask)
        
        if self.return_metadata:
            return image, mask, metadata
        else:
            return image, mask
    
    def get_sample_metadata(self, idx: int) -> dict:
        """
        Get metadata for a specific sample.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Metadata dictionary
        """
        return self.metadata_list[idx]
    
    def get_statistics(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        total_samples = len(self)
        
        # Count slices with tumor
        tumor_count = sum(1 for meta in self.metadata_list if meta['has_tumor'])
        no_tumor_count = total_samples - tumor_count
        
        # Get unique patients
        unique_patients = set(meta['patient_id'] for meta in self.metadata_list)
        
        # Calculate average tumor pixels
        avg_tumor_pixels = np.mean([
            meta['tumor_pixels'] for meta in self.metadata_list if meta['has_tumor']
        ]) if tumor_count > 0 else 0
        
        return {
            'total_slices': total_samples,
            'tumor_slices': tumor_count,
            'no_tumor_slices': no_tumor_count,
            'tumor_percentage': (tumor_count / total_samples * 100) if total_samples > 0 else 0,
            'unique_patients': len(unique_patients),
            'avg_tumor_pixels': float(avg_tumor_pixels),
        }


def create_dataloaders(
    train_dir: str = "data/processed/brats2d/train",
    val_dir: str = "data/processed/brats2d/val",
    test_dir: str = "data/processed/brats2d/test",
    batch_size: int = 16,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_dir: Directory with training .npz files
        val_dir: Directory with validation .npz files
        test_dir: Directory with test .npz files
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = BraTS2DSliceDataset(train_dir, transform=train_transform)
    val_dataset = BraTS2DSliceDataset(val_dir, transform=val_transform)
    test_dataset = BraTS2DSliceDataset(test_dir, transform=val_transform)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing BraTS2DSliceDataset...")
    
    # Create dataset
    dataset = BraTS2DSliceDataset("data/processed/brats2d")
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Total slices: {stats['total_slices']}")
    print(f"  Tumor slices: {stats['tumor_slices']} ({stats['tumor_percentage']:.1f}%)")
    print(f"  No tumor slices: {stats['no_tumor_slices']}")
    print(f"  Unique patients: {stats['unique_patients']}")
    print(f"  Avg tumor pixels: {stats['avg_tumor_pixels']:.1f}")
    
    # Get a sample
    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask dtype: {mask.dtype}")
        print(f"  Mask unique values: {torch.unique(mask).tolist()}")
        print(f"  Tumor pixels: {mask.sum().item()}")
        
        # Get metadata
        metadata = dataset.get_sample_metadata(0)
        print(f"\nMetadata:")
        print(f"  Patient ID: {metadata['patient_id']}")
        print(f"  Slice index: {metadata['slice_idx']}")
        print(f"  Modality: {metadata['modality']}")
        print(f"  Has tumor: {metadata['has_tumor']}")
    
    print("\n✓ All tests passed!")
