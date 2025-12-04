"""
PyTorch Dataset for Kaggle Brain MRI dataset.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class KaggleBrainMRIDataset(Dataset):
    """
    PyTorch Dataset for Kaggle Brain MRI classification.
    
    Loads preprocessed .npz files containing:
    - image: (1, H, W) normalized MRI slice
    - label: 0 (no tumor) or 1 (tumor present)
    - metadata: dict with image_id, class, etc.
    
    Args:
        data_dir: Directory containing .npz files (e.g., 'data/processed/kaggle/train')
        transform: Optional transform to apply to images
        return_metadata: If True, return (image, label, metadata) instead of (image, label)
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
        
        # Load labels for quick access
        self.labels = []
        for file_path in self.file_paths:
            data = np.load(file_path, allow_pickle=True)
            self.labels.append(int(data['label']))
        
        self.labels = np.array(self.labels)
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            If return_metadata=False: (image, label)
            If return_metadata=True: (image, label, metadata)
            
            - image: torch.Tensor of shape (1, H, W)
            - label: int (0 or 1)
            - metadata: dict with image information
        """
        # Load data
        file_path = self.file_paths[idx]
        data = np.load(file_path, allow_pickle=True)
        
        # Extract image and label
        image = data['image']  # (1, H, W)
        label = int(data['label'])
        metadata = data['metadata'].item() if 'metadata' in data else {}
        
        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        
        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        if self.return_metadata:
            return image, label, metadata
        else:
            return image, label
    
    def get_class_distribution(self) -> dict:
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary with class counts and percentages
        """
        n_tumor = np.sum(self.labels == 1)
        n_no_tumor = np.sum(self.labels == 0)
        total = len(self.labels)
        
        return {
            "tumor": {
                "count": int(n_tumor),
                "percentage": float(n_tumor / total * 100),
            },
            "no_tumor": {
                "count": int(n_no_tumor),
                "percentage": float(n_no_tumor / total * 100),
            },
            "total": total,
        }
    
    def get_sample_metadata(self, idx: int) -> dict:
        """
        Get metadata for a specific sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Metadata dictionary
        """
        file_path = self.file_paths[idx]
        data = np.load(file_path, allow_pickle=True)
        return data['metadata'].item() if 'metadata' in data else {}


def create_dataloaders(
    train_dir: str = "data/processed/kaggle/train",
    val_dir: str = "data/processed/kaggle/val",
    test_dir: str = "data/processed/kaggle/test",
    batch_size: int = 32,
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
        num_workers: Number of worker processes for data loading
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = KaggleBrainMRIDataset(train_dir, transform=train_transform)
    val_dataset = KaggleBrainMRIDataset(val_dir, transform=val_transform)
    test_dataset = KaggleBrainMRIDataset(test_dir, transform=val_transform)
    
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
    # Example usage
    print("Testing KaggleBrainMRIDataset...")
    
    # Create dataset
    dataset = KaggleBrainMRIDataset("data/processed/kaggle/train")
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Get a sample
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {image.shape}")
        print(f"  Label: {label}")
        print(f"  Metadata: {dataset.get_sample_metadata(0)}")
