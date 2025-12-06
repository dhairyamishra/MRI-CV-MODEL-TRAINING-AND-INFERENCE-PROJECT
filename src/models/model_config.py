"""
Model Configuration Management

This module provides utilities to save and load model architecture configurations
alongside checkpoints to prevent architecture mismatches.

Usage:
    # During training - save config with checkpoint
    config = ModelConfig(base_filters=64, depth=4, in_channels=1)
    config.save(checkpoint_dir / "model_config.json")
    
    # During inference - load config from checkpoint
    config = ModelConfig.load(checkpoint_dir / "model_config.json")
    model = create_multi_task_model(**config.to_dict())
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Core architecture
    base_filters: int = 64
    depth: int = 4
    in_channels: int = 1
    seg_out_channels: int = 1
    cls_num_classes: int = 2
    
    # Optional metadata
    model_type: str = "multitask"
    version: str = "1.0"
    description: Optional[str] = None
    
    # Training metadata (optional)
    trained_on: Optional[str] = None
    training_config: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"✓ Model config saved to: {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'ModelConfig':
        """Load configuration from JSON file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model config not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Model config loaded from: {path}")
        return cls(**data)
    
    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: Path) -> 'ModelConfig':
        """
        Load configuration from checkpoint directory.
        Looks for model_config.json in the directory.
        """
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "model_config.json"
        
        if not config_path.exists():
            # Fallback: try to infer from checkpoint filename or use defaults
            print(f"⚠ Model config not found at {config_path}")
            print("  Using default configuration (base_filters=64, depth=4)")
            return cls()
        
        return cls.load(config_path)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get parameters needed for model creation."""
        return {
            'base_filters': self.base_filters,
            'depth': self.depth,
            'in_channels': self.in_channels,
            'seg_out_channels': self.seg_out_channels,
            'cls_num_classes': self.cls_num_classes,
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"ModelConfig(\n"
            f"  base_filters={self.base_filters},\n"
            f"  depth={self.depth},\n"
            f"  in_channels={self.in_channels},\n"
            f"  seg_out_channels={self.seg_out_channels},\n"
            f"  cls_num_classes={self.cls_num_classes}\n"
            f")"
        )


def save_model_with_config(
    model,
    checkpoint_path: Path,
    config: ModelConfig,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Save model checkpoint along with its configuration.
    
    Args:
        model: PyTorch model to save
        checkpoint_path: Path to save checkpoint
        config: ModelConfig instance
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        epoch: Current epoch number
        metrics: Optional training metrics
    """
    import torch
    
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Model checkpoint saved to: {checkpoint_path}")
    
    # Save model config in same directory
    config_path = checkpoint_path.parent / "model_config.json"
    config.save(config_path)


def load_model_with_config(
    checkpoint_path: Path,
    model_factory_fn,
    device: str = 'cuda'
):
    """
    Load model checkpoint along with its configuration.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_factory_fn: Function to create model (e.g., create_multi_task_model)
        device: Device to load model on
    
    Returns:
        tuple: (model, config, checkpoint_dict)
    """
    import torch
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load config from checkpoint directory
    config = ModelConfig.from_checkpoint_dir(checkpoint_path.parent)
    
    # Create model with config
    model = model_factory_fn(**config.get_model_params())
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  Architecture: {config}")
    
    return model, config, checkpoint


# Example usage
if __name__ == "__main__":
    # Example: Create and save config
    config = ModelConfig(
        base_filters=64,
        depth=4,
        in_channels=1,
        seg_out_channels=1,
        cls_num_classes=2,
        description="Multi-task model for brain tumor detection",
        trained_on="BraTS 2020 + Kaggle Brain MRI"
    )
    
    print("Example ModelConfig:")
    print(config)
    print("\nModel parameters:")
    print(config.get_model_params())
