"""
PyTorch Dataset for BraTS Classification.

Derives binary classification labels from BraTS segmentation masks.
Compatible with the existing classification pipeline (Grad-CAM, evaluation, etc.)
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSClassificationDataset(Dataset):
    """
    PyTorch Dataset for BraTS binary classification.
    
    Derives labels from segmentation masks:
    - Label 0: No tumor (mask is all zeros)
    - Label 1: Has tumor (mask has non-zero pixels)
    
    This allows using BraTS data for classification tasks like Grad-CAM.
    
    Args:
        data_dir: Directory containing .npz files
        transform: Optional transform to apply to images
        min_tumor_pixels: Minimum pixels to consider as "has tumor"
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        min_tumor_pixels: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.min_tumor_pixels = min_tumor_pixels
        
        # Get all .npz files
        self.file_paths = sorted(self.data_dir.glob("*.npz"))
        
        if not self.file_paths:
            raise ValueError(f"No .npz files found in {self.data_dir}")
        
        # Precompute labels and metadata
        self.labels = []
        self.metadata_list = []
        
        for file_path in self.file_paths:
            data = np.load(file_path, allow_pickle=True)
            mask = data['mask']
            metadata = data['metadata'].item()
            
            # Derive binary label from mask
            has_tumor = np.sum(mask > 0) >= self.min_tumor_pixels
            label = 1 if has_tumor else 0
            
            self.labels.append(label)
            self.metadata_list.append(metadata)
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (image, label)
            - image: torch.Tensor of shape (1, H, W)
            - label: int (0 or 1)
        """
        # Load data
        file_path = self.file_paths[idx]
        data = np.load(file_path, allow_pickle=True)
        
        # Extract image
        image = data['image']  # (1, H, W)
        
        # Get precomputed label
        label = self.labels[idx]
        
        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        
        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def get_sample_metadata(self, idx: int) -> dict:
        """
        Get metadata for a specific sample.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary with metadata
        """
        return self.metadata_list[idx]
    
    def get_class_distribution(self) -> dict:
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary with class counts
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        return {
            'no_tumor': class_counts.get(0, 0),
            'has_tumor': class_counts.get(1, 0),
            'total': len(self.labels),
            'tumor_ratio': class_counts.get(1, 0) / len(self.labels) if len(self.labels) > 0 else 0.0,
        }
    
    def get_statistics(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        class_dist = self.get_class_distribution()
        
        # Get unique patients
        patient_ids = [meta['patient_id'] for meta in self.metadata_list]
        unique_patients = len(set(patient_ids))
        
        return {
            'total_slices': len(self),
            'unique_patients': unique_patients,
            'class_distribution': class_dist,
            'avg_slices_per_patient': len(self) / unique_patients if unique_patients > 0 else 0,
        }


def create_brats_classification_dataloaders(
    train_dir: str = "data/processed/brats2d/train",
    val_dir: str = "data/processed/brats2d/val",
    test_dir: str = "data/processed/brats2d/test",
    batch_size: int = 32,
    num_workers: int = 0,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    min_tumor_pixels: int = 10,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, validation, and test dataloaders for BraTS classification.
    
    Args:
        train_dir: Directory with training .npz files
        val_dir: Directory with validation .npz files
        test_dir: Directory with test .npz files
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        min_tumor_pixels: Minimum pixels to consider as "has tumor"
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = BraTSClassificationDataset(
        train_dir, 
        transform=train_transform,
        min_tumor_pixels=min_tumor_pixels
    )
    val_dataset = BraTSClassificationDataset(
        val_dir, 
        transform=val_transform,
        min_tumor_pixels=min_tumor_pixels
    )
    test_dataset = BraTSClassificationDataset(
        test_dir, 
        transform=val_transform,
        min_tumor_pixels=min_tumor_pixels
    )
    
    # Print statistics
    print("\nBraTS Classification Dataset Statistics:")
    print(f"\nTrain set:")
    train_stats = train_dataset.get_statistics()
    print(f"  Total slices: {train_stats['total_slices']}")
    print(f"  Patients: {train_stats['unique_patients']}")
    print(f"  No tumor: {train_stats['class_distribution']['no_tumor']}")
    print(f"  Has tumor: {train_stats['class_distribution']['has_tumor']}")
    print(f"  Tumor ratio: {train_stats['class_distribution']['tumor_ratio']:.2%}")
    
    print(f"\nVal set:")
    val_stats = val_dataset.get_statistics()
    print(f"  Total slices: {val_stats['total_slices']}")
    print(f"  No tumor: {val_stats['class_distribution']['no_tumor']}")
    print(f"  Has tumor: {val_stats['class_distribution']['has_tumor']}")
    
    print(f"\nTest set:")
    test_stats = test_dataset.get_statistics()
    print(f"  Total slices: {test_stats['total_slices']}")
    print(f"  No tumor: {test_stats['class_distribution']['no_tumor']}")
    print(f"  Has tumor: {test_stats['class_distribution']['has_tumor']}")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test the dataset."""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print("Testing BraTS Classification Dataset...")
    
    # Test with train data
    dataset = BraTSClassificationDataset("data/processed/brats2d/train")
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test a sample
    image, label = dataset[0]
    metadata = dataset.get_sample_metadata(0)
    
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Label: {label} ({'Tumor' if label == 1 else 'No Tumor'})")
    print(f"  Patient ID: {metadata['patient_id']}")
    print(f"  Slice index: {metadata['slice_idx']}")
    
    print("\n✓ All tests passed!")
