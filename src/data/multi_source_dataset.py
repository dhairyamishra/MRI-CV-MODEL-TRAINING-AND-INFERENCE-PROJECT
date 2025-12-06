"""
Multi-Source Dataset for Multi-Task Learning.

Combines BraTS and Kaggle datasets into a unified format:
- BraTS: Has segmentation masks + derived classification labels
- Kaggle: Has classification labels only (no masks)

Each sample returns a dictionary:
{
    "image": tensor(C, H, W),
    "mask": tensor(H, W) or None,
    "cls": int(0 or 1),
    "source": "brats" or "kaggle"
}

This enables training a multi-task model on both datasets simultaneously.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiSourceDataset(Dataset):
    """
    Unified dataset combining BraTS and Kaggle data.
    
    Args:
        brats_dir: Directory with BraTS .npz files (or None to skip)
        kaggle_dir: Directory with Kaggle .npz files (or None to skip)
        transform: Optional transform to apply to images
        min_tumor_pixels: Minimum pixels to consider as "has tumor" for BraTS
    """
    
    def __init__(
        self,
        brats_dir: Optional[str] = None,
        kaggle_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        min_tumor_pixels: int = 10,
    ):
        if brats_dir is None and kaggle_dir is None:
            raise ValueError("At least one of brats_dir or kaggle_dir must be provided")
        
        self.transform = transform
        self.min_tumor_pixels = min_tumor_pixels
        
        # Collect all samples
        self.samples = []
        
        # Load BraTS samples
        if brats_dir is not None:
            brats_path = Path(brats_dir)
            if brats_path.exists():
                brats_files = sorted(brats_path.glob("*.npz"))
                for file_path in brats_files:
                    self.samples.append({
                        'path': file_path,
                        'source': 'brats',
                    })
                print(f"Loaded {len(brats_files)} BraTS samples from {brats_dir}")
        
        # Load Kaggle samples
        if kaggle_dir is not None:
            kaggle_path = Path(kaggle_dir)
            if kaggle_path.exists():
                kaggle_files = sorted(kaggle_path.glob("*.npz"))
                for file_path in kaggle_files:
                    self.samples.append({
                        'path': file_path,
                        'source': 'kaggle',
                    })
                print(f"Loaded {len(kaggle_files)} Kaggle samples from {kaggle_dir}")
        
        if len(self.samples) == 0:
            raise ValueError("No samples found in provided directories")
        
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary with keys:
            - image: torch.Tensor of shape (C, H, W)
            - mask: torch.Tensor of shape (H, W) or None
            - cls: int (0 or 1)
            - source: str ("brats" or "kaggle")
            - has_mask: bool (True if mask is available)
        """
        sample_info = self.samples[idx]
        file_path = sample_info['path']
        source = sample_info['source']
        
        # Load data
        data = np.load(file_path, allow_pickle=True)
        
        # Extract image
        image = data['image']  # (C, H, W)
        image = torch.from_numpy(image).float()
        
        # Initialize output dictionary
        output = {
            'source': source,
        }
        
        if source == 'brats':
            # BraTS: Has both mask and classification label
            mask = data['mask']  # (1, H, W) or (H, W)
            
            # Ensure mask is (H, W)
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            
            mask = torch.from_numpy(mask).float()
            
            # Derive classification label from mask
            has_tumor = torch.sum(mask > 0) >= self.min_tumor_pixels
            cls_label = 1 if has_tumor else 0
            
            output['mask'] = mask
            output['cls'] = cls_label
            output['has_mask'] = True
            
        elif source == 'kaggle':
            # Kaggle: Has classification label only
            cls_label = int(data['label'])
            
            # Ensure label is 0 or 1 (remap if necessary)
            if cls_label not in [0, 1]:
                # If labels are 1 and 2, remap to 0 and 1
                cls_label = cls_label - 1 if cls_label > 0 else 0
            
            # Validate label is in valid range
            assert cls_label in [0, 1], f"Invalid label {cls_label} in {file_path}"
            
            output['mask'] = None
            output['cls'] = cls_label
            output['has_mask'] = False
        
        # Apply transforms if provided
        if self.transform is not None:
            if output['has_mask']:
                # Transform both image and mask
                image, mask = self.transform(image, output['mask'])
                output['mask'] = mask
            else:
                # Transform image only
                image = self.transform(image)
        
        output['image'] = image
        
        return output
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        n_brats = sum(1 for s in self.samples if s['source'] == 'brats')
        n_kaggle = sum(1 for s in self.samples if s['source'] == 'kaggle')
        
        # Count classification labels
        n_tumor = 0
        n_no_tumor = 0
        
        for sample in self.samples:
            data = np.load(sample['path'], allow_pickle=True)
            
            if sample['source'] == 'brats':
                mask = data['mask']
                has_tumor = np.sum(mask > 0) >= self.min_tumor_pixels
                if has_tumor:
                    n_tumor += 1
                else:
                    n_no_tumor += 1
            else:  # kaggle
                label = int(data['label'])
                if label == 1:
                    n_tumor += 1
                else:
                    n_no_tumor += 1
        
        return {
            'total_samples': len(self.samples),
            'brats_samples': n_brats,
            'kaggle_samples': n_kaggle,
            'tumor_samples': n_tumor,
            'no_tumor_samples': n_no_tumor,
            'tumor_ratio': n_tumor / len(self.samples) if len(self.samples) > 0 else 0,
            'brats_ratio': n_brats / len(self.samples) if len(self.samples) > 0 else 0,
            'kaggle_ratio': n_kaggle / len(self.samples) if len(self.samples) > 0 else 0,
        }


class BraTSOnlyDataset(Dataset):
    """
    BraTS-only dataset for segmentation warm-up training.
    
    Returns both mask and classification label.
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
        
        self.file_paths = sorted(self.data_dir.glob("*.npz"))
        
        if not self.file_paths:
            raise ValueError(f"No .npz files found in {self.data_dir}")
        
        print(f"Loaded {len(self.file_paths)} BraTS samples")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with image, mask, cls, source, has_mask
        """
        file_path = self.file_paths[idx]
        data = np.load(file_path, allow_pickle=True)
        
        # Extract image and mask
        image = data['image']  # (1, H, W)
        mask = data['mask']    # (1, H, W) or (H, W)
        
        # Ensure mask is (H, W)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        # Derive classification label
        has_tumor = torch.sum(mask > 0) >= self.min_tumor_pixels
        cls_label = 1 if has_tumor else 0
        
        # Apply transforms
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        return {
            'image': image,
            'mask': mask,
            'cls': cls_label,
            'source': 'brats',
            'has_mask': True,
        }


class KaggleOnlyDataset(Dataset):
    """
    Kaggle-only dataset for classification training.
    
    Returns classification label only (no mask).
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        self.file_paths = sorted(self.data_dir.glob("*.npz"))
        
        if not self.file_paths:
            raise ValueError(f"No .npz files found in {self.data_dir}")
        
        print(f"Loaded {len(self.file_paths)} Kaggle samples")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with image, mask=None, cls, source, has_mask=False
        """
        file_path = self.file_paths[idx]
        data = np.load(file_path, allow_pickle=True)
        
        # Extract image and label
        image = data['image']  # (1, H, W)
        cls_label = int(data['label'])
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return {
            'image': image,
            'mask': None,
            'cls': cls_label,
            'source': 'kaggle',
            'has_mask': False,
        }


if __name__ == "__main__":
    """Test the multi-source dataset."""
    print("Testing MultiSourceDataset...")
    
    # Test with both datasets
    try:
        dataset = MultiSourceDataset(
            brats_dir="data/processed/brats2d/train",
            kaggle_dir="data/processed/kaggle_unified/train",
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Get statistics
        stats = dataset.get_statistics()
        print(f"\nDataset statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}" if 'ratio' in key else f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Test a few samples
        print(f"\nTesting samples:")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Source: {sample['source']}")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Has mask: {sample['has_mask']}")
            if sample['has_mask']:
                print(f"  Mask shape: {sample['mask'].shape}")
                print(f"  Mask sum: {sample['mask'].sum().item()}")
            print(f"  Classification label: {sample['cls']}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have preprocessed data in:")
        print("  - data/processed/brats2d/train")
        print("  - data/processed/kaggle_unified/train")
