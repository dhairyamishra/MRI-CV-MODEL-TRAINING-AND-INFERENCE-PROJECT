#!/usr/bin/env python3
"""
Check what's inside a checkpoint file.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

checkpoint_path = project_root / "checkpoints" / "1000_epoch_multitask_joint" / "best_model.pth"

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"\n{'='*80}")
print(f"Checkpoint Keys:")
print(f"{'='*80}")
for key in checkpoint.keys():
    if key != 'model_state_dict' and key != 'state_dict':
        print(f"  - {key}: {type(checkpoint[key])}")

if 'config' in checkpoint:
    print(f"\n{'='*80}")
    print(f"Config:")
    print(f"{'='*80}")
    import json
    print(json.dumps(checkpoint['config'], indent=2, default=str))

if 'epoch' in checkpoint:
    print(f"\nEpoch: {checkpoint['epoch']}")

if 'best_val_metric' in checkpoint:
    print(f"Best Val Metric: {checkpoint['best_val_metric']}")

if 'val_dice' in checkpoint:
    print(f"Val Dice: {checkpoint['val_dice']}")

if 'val_acc' in checkpoint:
    print(f"Val Accuracy: {checkpoint['val_acc']}")

print(f"\n{'='*80}")
print(f"Model State Dict Keys (first 10):")
print(f"{'='*80}")
state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
state_dict = checkpoint[state_dict_key]
for i, key in enumerate(list(state_dict.keys())[:10]):
    print(f"  {i+1}. {key}: {state_dict[key].shape}")

print(f"\nTotal parameters in state dict: {len(state_dict)}")
