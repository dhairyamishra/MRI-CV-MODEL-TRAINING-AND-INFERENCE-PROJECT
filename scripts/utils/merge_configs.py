"""
Config merger utility - Combines hierarchical configs into final configs.

This tool merges base configs, stage configs, and mode configs into final
training configurations. It eliminates duplication and ensures consistency.

Usage:
    # Generate a single config
    python scripts/utils/merge_configs.py --stage 1 --mode quick
    python scripts/utils/merge_configs.py --stage 2 --mode production
    
    # Generate all 9 configs at once
    python scripts/utils/merge_configs.py --all
    
    # Custom output directory
    python scripts/utils/merge_configs.py --all --output-dir configs/generated

Merge Order (later overrides earlier):
    1. base/common.yaml
    2. base/training_defaults.yaml
    3. stages/stageN_*.yaml
    4. modes/MODE.yaml

References are resolved:
    - model.architecture: "multitask_medium" → expanded from base/model_architectures.yaml
    - augmentation.preset: "moderate" → expanded from base/augmentation_presets.yaml
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary with override values
        
    Returns:
        Merged dictionary (base is not modified)
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> Dict:
    """
    Load YAML file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Parsed YAML as dictionary
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def resolve_references(config: Dict, base_configs: Dict) -> Dict:
    """
    Resolve references like 'architecture: multitask_medium' and 'preset: moderate'.
    
    Args:
        config: Config dictionary with potential references
        base_configs: Dictionary containing architectures and augmentation_presets
        
    Returns:
        Config with references expanded
    """
    # Resolve model architecture reference
    if 'model' in config and 'architecture' in config['model']:
        arch_name = config['model']['architecture']
        if arch_name in base_configs['architectures']:
            # Save any existing model overrides
            existing_model_params = {k: v for k, v in config['model'].items() if k != 'architecture'}
            # Start with architecture parameters as base
            arch_params = base_configs['architectures'][arch_name].copy()
            # Overlay existing parameters (preserves overrides from mode configs)
            arch_params.update(existing_model_params)
            # Replace model config
            config['model'] = arch_params
    
    # Resolve augmentation preset reference
    if 'augmentation' in config and 'preset' in config['augmentation']:
        preset_name = config['augmentation']['preset']
        if preset_name in base_configs['augmentation_presets']:
            # Replace preset reference with actual augmentation config
            config['augmentation'] = base_configs['augmentation_presets'][preset_name].copy()
    
    return config


def merge_config(stage: int, mode: str) -> Dict:
    """
    Merge configs for a specific stage and mode.
    
    Args:
        stage: Stage number (1, 2, or 3)
        mode: Mode name ('quick', 'baseline', or 'production')
        
    Returns:
        Merged configuration dictionary
    """
    base_dir = Path("configs/base")
    stages_dir = Path("configs/stages")
    modes_dir = Path("configs/modes")
    
    # Load base configs
    common = load_yaml(base_dir / "common.yaml")
    architectures = load_yaml(base_dir / "model_architectures.yaml")
    training_defaults = load_yaml(base_dir / "training_defaults.yaml")
    augmentation_presets = load_yaml(base_dir / "augmentation_presets.yaml")
    
    # Load stage config
    stage_files = {
        1: "stage1_seg_warmup.yaml",
        2: "stage2_cls_head.yaml",
        3: "stage3_joint.yaml"
    }
    stage_config = load_yaml(stages_dir / stage_files[stage])
    
    # Load mode config
    mode_files = {
        "quick": "quick_test.yaml",
        "baseline": "baseline.yaml",
        "production": "production.yaml"
    }
    mode_config = load_yaml(modes_dir / mode_files[mode])
    
    # Merge in order: common -> training_defaults -> stage -> mode
    # Later configs override earlier ones
    merged = deep_merge(common, training_defaults)
    merged = deep_merge(merged, stage_config)
    merged = deep_merge(merged, mode_config)
    
    # Resolve references (architecture names, augmentation presets)
    base_configs = {
        'architectures': architectures['architectures'],
        'augmentation_presets': augmentation_presets['augmentation_presets']
    }
    merged = resolve_references(merged, base_configs)
    
    # Add metadata for tracking
    merged['_metadata'] = {
        'stage': stage,
        'mode': mode,
        'generated_at': datetime.now().isoformat(),
        'generated_from': [
            'base/common.yaml',
            'base/training_defaults.yaml',
            f'stages/{stage_files[stage]}',
            f'modes/{mode_files[mode]}'
        ]
    }
    
    return merged


def save_yaml(config: Dict, path: Path) -> None:
    """
    Save config to YAML file.
    
    Args:
        config: Configuration dictionary
        path: Output file path
    """
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description='Merge hierarchical configs into final training configs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single config
  python scripts/utils/merge_configs.py --stage 1 --mode quick
  
  # Generate all 9 configs
  python scripts/utils/merge_configs.py --all
  
  # Custom output directory
  python scripts/utils/merge_configs.py --all --output-dir configs/generated
        """
    )
    parser.add_argument('--stage', type=int, choices=[1, 2, 3],
                        help='Stage number (1=seg_warmup, 2=cls_head, 3=joint)')
    parser.add_argument('--mode', choices=['quick', 'baseline', 'production'],
                        help='Training mode')
    parser.add_argument('--all', action='store_true',
                        help='Generate all 9 configs (3 stages × 3 modes)')
    parser.add_argument('--output-dir', default='configs/final',
                        help='Output directory for generated configs (default: configs/final)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        # Generate all 9 configs
        print("Generating all 9 configuration files...")
        print("=" * 60)
        
        for stage in [1, 2, 3]:
            for mode in ['quick', 'baseline', 'production']:
                config = merge_config(stage, mode)
                output_file = output_dir / f"stage{stage}_{mode}.yaml"
                save_yaml(config, output_file)
                print(f"✓ Generated: {output_file}")
        
        print("=" * 60)
        print(f"Successfully generated 9 configs in {output_dir}/")
        print("\nGenerated files:")
        for stage in [1, 2, 3]:
            for mode in ['quick', 'baseline', 'production']:
                print(f"  - stage{stage}_{mode}.yaml")
    
    else:
        # Generate single config
        if not args.stage or not args.mode:
            parser.error("--stage and --mode are required (or use --all)")
        
        config = merge_config(args.stage, args.mode)
        output_file = output_dir / f"stage{args.stage}_{args.mode}.yaml"
        save_yaml(config, output_file)
        print(f"✓ Generated: {output_file}")


if __name__ == "__main__":
    main()
