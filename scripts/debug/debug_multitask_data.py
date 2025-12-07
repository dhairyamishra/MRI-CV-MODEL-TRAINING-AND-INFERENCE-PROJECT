"""
Debug script to check multi-task dataset and identify issues.
"""

import sys
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.multi_source_dataset import MultiSourceDataset


def custom_collate_fn(batch):
    """Custom collate function to handle None values in masks."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['cls'] for item in batch], dtype=torch.long)
    sources = [item['source'] for item in batch]
    has_masks = torch.tensor([item['has_mask'] for item in batch], dtype=torch.bool)
    
    # For masks, collect non-None values
    masks_list = []
    for item in batch:
        if item['mask'] is not None:
            masks_list.append(item['mask'])
    
    # Stack masks if any exist
    if masks_list:
        masks = torch.stack(masks_list)
    else:
        masks = None
    
    return {
        'image': images,
        'mask': masks,
        'cls': labels,
        'source': sources,
        'has_mask': has_masks,
    }


def main():
    print("=" * 80)
    print("Multi-Task Dataset Debugger")
    print("=" * 80)
    
    # Load config
    config_path = "configs/multitask_joint_quick_test.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nLoading datasets...")
    
    # Load train dataset
    train_dataset = MultiSourceDataset(
        brats_dir=config['data']['brats_train_dir'],
        kaggle_dir=config['data']['kaggle_train_dir'],
        transform=None,
    )
    
    print(f"\n{'='*80}")
    print("Checking individual samples...")
    print(f"{'='*80}")
    
    # Check first few samples
    for i in range(min(10, len(train_dataset))):
        sample = train_dataset[i]
        print(f"\nSample {i}:")
        print(f"  Source: {sample['source']}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Image range: [{sample['image'].min():.4f}, {sample['image'].max():.4f}]")
        print(f"  Has mask: {sample['has_mask']}")
        
        if sample['has_mask']:
            print(f"  Mask shape: {sample['mask'].shape}")
            print(f"  Mask range: [{sample['mask'].min():.4f}, {sample['mask'].max():.4f}]")
            print(f"  Mask unique values: {sample['mask'].unique().tolist()}")
        
        print(f"  Classification label: {sample['cls']}")
        print(f"  Label type: {type(sample['cls'])}")
    
    print(f"\n{'='*80}")
    print("Checking dataloader batches...")
    print(f"{'='*80}")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )
    
    # Check first batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch info:")
    print(f"  Images shape: {batch['image'].shape}")
    print(f"  Images range: [{batch['image'].min():.4f}, {batch['image'].max():.4f}]")
    print(f"  Labels: {batch['cls'].tolist()}")
    print(f"  Labels shape: {batch['cls'].shape}")
    print(f"  Labels dtype: {batch['cls'].dtype}")
    print(f"  Labels min/max: [{batch['cls'].min()}, {batch['cls'].max()}]")
    print(f"  Has masks: {batch['has_mask'].tolist()}")
    print(f"  Sources: {batch['source']}")
    
    if batch['mask'] is not None:
        print(f"  Masks shape: {batch['mask'].shape}")
        print(f"  Masks range: [{batch['mask'].min():.4f}, {batch['mask'].max():.4f}]")
        print(f"  Masks unique values: {batch['mask'].unique().tolist()}")
    else:
        print(f"  Masks: None (no samples with masks in this batch)")
    
    print(f"\n{'='*80}")
    print("Label Statistics")
    print(f"{'='*80}")
    
    # Count labels
    all_labels = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        all_labels.append(sample['cls'])
    
    unique_labels = set(all_labels)
    print(f"\nUnique labels found: {sorted(unique_labels)}")
    print(f"Label counts:")
    for label in sorted(unique_labels):
        count = all_labels.count(label)
        print(f"  Label {label}: {count} samples ({count/len(all_labels)*100:.1f}%)")
    
    # Check if any labels are out of range
    invalid_labels = [l for l in all_labels if l not in [0, 1]]
    if invalid_labels:
        print(f"\n⚠️  WARNING: Found {len(invalid_labels)} invalid labels!")
        print(f"  Invalid labels: {set(invalid_labels)}")
    else:
        print(f"\n✓ All labels are valid (0 or 1)")
    
    print(f"\n{'='*80}")
    print("✓ Debug complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
