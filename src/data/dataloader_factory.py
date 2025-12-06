"""
Dataloader Factory for Multi-Task Learning.

Creates dataloaders for different training strategies:
- Option A: Alternating batches (separate BraTS and Kaggle loaders)
- Option B: Mixed batches (single unified loader)
- BraTS-only: For segmentation warm-up
- Kaggle-only: For classification head training

Supports both single-source and multi-source training scenarios.
"""

from typing import Callable, Optional, Tuple, Dict
import torch
from torch.utils.data import DataLoader

from src.data.multi_source_dataset import (
    MultiSourceDataset,
    BraTSOnlyDataset,
    KaggleOnlyDataset,
)


def create_multitask_dataloaders(
    brats_train_dir: Optional[str] = None,
    brats_val_dir: Optional[str] = None,
    brats_test_dir: Optional[str] = None,
    kaggle_train_dir: Optional[str] = None,
    kaggle_val_dir: Optional[str] = None,
    kaggle_test_dir: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    mode: str = "mixed",
    min_tumor_pixels: int = 10,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for multi-task learning.
    
    Args:
        brats_train_dir: BraTS training directory
        brats_val_dir: BraTS validation directory
        brats_test_dir: BraTS test directory
        kaggle_train_dir: Kaggle training directory
        kaggle_val_dir: Kaggle validation directory
        kaggle_test_dir: Kaggle test directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        mode: "mixed" (single loader) or "alternating" (separate loaders)
        min_tumor_pixels: Minimum pixels to consider as "has tumor"
    
    Returns:
        Dictionary of dataloaders based on mode:
        - "mixed": {"train": loader, "val": loader, "test": loader}
        - "alternating": {
            "train_brats": loader, "train_kaggle": loader,
            "val": loader, "test": loader
          }
    """
    print("=" * 70)
    print(f"Creating Multi-Task Dataloaders (mode: {mode})")
    print("=" * 70)
    
    if mode == "mixed":
        # Option B: Single mixed dataloader
        print("\nMode: Mixed batches (single dataloader)")
        
        # Training loader
        train_dataset = MultiSourceDataset(
            brats_dir=brats_train_dir,
            kaggle_dir=kaggle_train_dir,
            transform=train_transform,
            min_tumor_pixels=min_tumor_pixels,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=multitask_collate_fn,
        )
        
        # Validation loader
        val_dataset = MultiSourceDataset(
            brats_dir=brats_val_dir,
            kaggle_dir=kaggle_val_dir,
            transform=val_transform,
            min_tumor_pixels=min_tumor_pixels,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=multitask_collate_fn,
        )
        
        # Test loader
        test_dataset = MultiSourceDataset(
            brats_dir=brats_test_dir,
            kaggle_dir=kaggle_test_dir,
            transform=val_transform,
            min_tumor_pixels=min_tumor_pixels,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=multitask_collate_fn,
        )
        
        print(f"\nDataloader sizes:")
        print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
        print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
        print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
        
        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
    
    elif mode == "alternating":
        # Option A: Separate dataloaders for alternating batches
        print("\nMode: Alternating batches (separate dataloaders)")
        
        # BraTS training loader
        brats_train_dataset = BraTSOnlyDataset(
            data_dir=brats_train_dir,
            transform=train_transform,
            min_tumor_pixels=min_tumor_pixels,
        )
        
        brats_train_loader = DataLoader(
            brats_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=multitask_collate_fn,
        )
        
        # Kaggle training loader
        kaggle_train_dataset = KaggleOnlyDataset(
            data_dir=kaggle_train_dir,
            transform=train_transform,
        )
        
        kaggle_train_loader = DataLoader(
            kaggle_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=multitask_collate_fn,
        )
        
        # Validation loader (mixed)
        val_dataset = MultiSourceDataset(
            brats_dir=brats_val_dir,
            kaggle_dir=kaggle_val_dir,
            transform=val_transform,
            min_tumor_pixels=min_tumor_pixels,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=multitask_collate_fn,
        )
        
        # Test loader (mixed)
        test_dataset = MultiSourceDataset(
            brats_dir=brats_test_dir,
            kaggle_dir=kaggle_test_dir,
            transform=val_transform,
            min_tumor_pixels=min_tumor_pixels,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=multitask_collate_fn,
        )
        
        print(f"\nDataloader sizes:")
        print(f"  Train BraTS:  {len(brats_train_loader)} batches ({len(brats_train_dataset)} samples)")
        print(f"  Train Kaggle: {len(kaggle_train_loader)} batches ({len(kaggle_train_dataset)} samples)")
        print(f"  Val:          {len(val_loader)} batches ({len(val_dataset)} samples)")
        print(f"  Test:         {len(test_loader)} batches ({len(test_dataset)} samples)")
        
        return {
            "train_brats": brats_train_loader,
            "train_kaggle": kaggle_train_loader,
            "val": val_loader,
            "test": test_loader,
        }
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'mixed' or 'alternating'")


def create_brats_only_dataloaders(
    train_dir: str = "data/processed/brats2d/train",
    val_dir: str = "data/processed/brats2d/val",
    test_dir: str = "data/processed/brats2d/test",
    batch_size: int = 16,
    num_workers: int = 0,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    min_tumor_pixels: int = 10,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create BraTS-only dataloaders for segmentation warm-up.
    
    Args:
        train_dir: Training directory
        val_dir: Validation directory
        test_dir: Test directory
        batch_size: Batch size
        num_workers: Number of workers
        train_transform: Training transform
        val_transform: Validation transform
        min_tumor_pixels: Minimum tumor pixels
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("=" * 70)
    print("Creating BraTS-Only Dataloaders (Segmentation Warm-up)")
    print("=" * 70)
    
    # Create datasets
    train_dataset = BraTSOnlyDataset(train_dir, train_transform, min_tumor_pixels)
    val_dataset = BraTSOnlyDataset(val_dir, val_transform, min_tumor_pixels)
    test_dataset = BraTSOnlyDataset(test_dir, val_transform, min_tumor_pixels)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=multitask_collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=multitask_collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=multitask_collate_fn,
    )
    
    print(f"\nDataloader sizes:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


def multitask_collate_fn(batch):
    """
    Custom collate function for multi-task batches.
    
    Handles batches where some samples have masks and others don't.
    
    Args:
        batch: List of dictionaries from dataset
    
    Returns:
        Dictionary with batched tensors:
        - images: (B, C, H, W)
        - masks: (B, H, W) or None for samples without masks
        - cls_labels: (B,)
        - sources: List of strings
        - has_masks: (B,) boolean tensor
    """
    # Stack images
    images = torch.stack([item['image'] for item in batch])
    
    # Stack classification labels
    cls_labels = torch.tensor([item['cls'] for item in batch], dtype=torch.long)
    
    # Handle masks (some may be None)
    has_masks = torch.tensor([item['has_mask'] for item in batch], dtype=torch.bool)
    
    if has_masks.any():
        # Create mask tensor, filling None with zeros
        mask_list = []
        for item in batch:
            if item['mask'] is not None:
                mask_list.append(item['mask'])
            else:
                # Create dummy mask with same shape
                dummy_mask = torch.zeros_like(batch[0]['mask'] if batch[0]['mask'] is not None else images[0, 0])
                mask_list.append(dummy_mask)
        masks = torch.stack(mask_list)
    else:
        # No masks in this batch
        masks = None
    
    # Collect sources
    sources = [item['source'] for item in batch]
    
    return {
        'images': images,
        'masks': masks,
        'cls_labels': cls_labels,
        'sources': sources,
        'has_masks': has_masks,
    }


if __name__ == "__main__":
    """Test the dataloader factory."""
    print("Testing Dataloader Factory...")
    
    # Test mixed mode
    print("\n" + "=" * 70)
    print("Test 1: Mixed mode")
    print("=" * 70)
    
    try:
        loaders = create_multitask_dataloaders(
            brats_train_dir="data/processed/brats2d/train",
            kaggle_train_dir="data/processed/kaggle_unified/train",
            brats_val_dir="data/processed/brats2d/val",
            kaggle_val_dir="data/processed/kaggle_unified/val",
            batch_size=8,
            mode="mixed",
        )
        
        # Test a batch
        batch = next(iter(loaders['train']))
        print(f"\nSample batch:")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Masks shape: {batch['masks'].shape if batch['masks'] is not None else None}")
        print(f"  Cls labels shape: {batch['cls_labels'].shape}")
        print(f"  Has masks: {batch['has_masks']}")
        print(f"  Sources: {batch['sources']}")
        
        print("\n✓ Mixed mode test passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    # Test alternating mode
    print("\n" + "=" * 70)
    print("Test 2: Alternating mode")
    print("=" * 70)
    
    try:
        loaders = create_multitask_dataloaders(
            brats_train_dir="data/processed/brats2d/train",
            kaggle_train_dir="data/processed/kaggle_unified/train",
            brats_val_dir="data/processed/brats2d/val",
            kaggle_val_dir="data/processed/kaggle_unified/val",
            batch_size=8,
            mode="alternating",
        )
        
        # Test BraTS batch
        brats_batch = next(iter(loaders['train_brats']))
        print(f"\nBraTS batch:")
        print(f"  Images shape: {brats_batch['images'].shape}")
        print(f"  Masks shape: {brats_batch['masks'].shape}")
        print(f"  All have masks: {brats_batch['has_masks'].all()}")
        
        # Test Kaggle batch
        kaggle_batch = next(iter(loaders['train_kaggle']))
        print(f"\nKaggle batch:")
        print(f"  Images shape: {kaggle_batch['images'].shape}")
        print(f"  Masks: {kaggle_batch['masks']}")
        print(f"  None have masks: {(~kaggle_batch['has_masks']).all()}")
        
        print("\n✓ Alternating mode test passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    # Test BraTS-only mode
    print("\n" + "=" * 70)
    print("Test 3: BraTS-only mode")
    print("=" * 70)
    
    try:
        train_loader, val_loader, test_loader = create_brats_only_dataloaders(
            batch_size=8,
        )
        
        batch = next(iter(train_loader))
        print(f"\nBraTS-only batch:")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Masks shape: {batch['masks'].shape}")
        print(f"  Cls labels shape: {batch['cls_labels'].shape}")
        
        print("\n✓ BraTS-only mode test passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
