"""
Fix model_config.json to match the actual trained model architecture.
"""

import json
from pathlib import Path

# Project root
project_root = Path(__file__).parent.parent

# Correct model config (from multitask_joint_production.yaml)
correct_config = {
    "base_filters": 64,
    "depth": 4,
    "in_channels": 1,
    "seg_out_channels": 1,
    "cls_num_classes": 2
}

# Path to model config
config_path = project_root / "checkpoints" / "multitask_joint" / "model_config.json"

print("=" * 80)
print("FIXING MODEL CONFIG")
print("=" * 80)

# Backup old config
if config_path.exists():
    backup_path = config_path.with_suffix('.json.backup')
    with open(config_path, 'r') as f:
        old_config = json.load(f)
    
    with open(backup_path, 'w') as f:
        json.dump(old_config, f, indent=2)
    
    print(f"\n✓ Backed up old config to: {backup_path}")
    print(f"  Old config: base_filters={old_config.get('base_filters')}, depth={old_config.get('depth')}")

# Write correct config
with open(config_path, 'w') as f:
    json.dump(correct_config, f, indent=2)

print(f"\n✓ Wrote correct config to: {config_path}")
print(f"  New config: base_filters={correct_config['base_filters']}, depth={correct_config['depth']}")

print("\n" + "=" * 80)
print("✓ Model config fixed!")
print("=" * 80)
print("\nNow restart the backend:")
print("  python app/backend/main_v2.py")
