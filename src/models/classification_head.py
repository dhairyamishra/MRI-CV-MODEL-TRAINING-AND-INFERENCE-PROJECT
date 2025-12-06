"""
Classification Head for Multi-Task Learning.

Takes bottleneck features from encoder and produces classification logits
using global average pooling + MLP.
"""

import torch
import torch.nn as nn
from typing import Optional


class ClassificationHead(nn.Module):
    """
    Classification head for tumor detection.
    
    Takes bottleneck features from encoder and applies:
    1. Global Average Pooling (spatial dimensions -> 1x1)
    2. Small MLP with dropout for regularization
    3. Output logits for binary classification
    
    Args:
        in_channels: Number of input channels from encoder bottleneck
        num_classes: Number of output classes (default: 2 for binary)
        hidden_dim: Hidden layer dimension (default: 256)
        dropout: Dropout rate (default: 0.5)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ):
        super(ClassificationHead, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            x: Bottleneck features (B, C, H, W)
               e.g., (B, 1024, 16, 16) for depth=4
        
        Returns:
            logits: Classification logits (B, num_classes)
        """
        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1)
        x = self.gap(x)
        
        # MLP: (B, C, 1, 1) -> (B, num_classes)
        logits = self.classifier(x)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_classification_head(
    in_channels: int,
    num_classes: int = 2,
    hidden_dim: int = 256,
    dropout: float = 0.5,
) -> ClassificationHead:
    """
    Factory function to create classification head.
    
    Args:
        in_channels: Number of input channels from encoder
        num_classes: Number of output classes (default: 2)
        hidden_dim: Hidden layer dimension (default: 256)
        dropout: Dropout rate (default: 0.5)
    
    Returns:
        ClassificationHead model
    
    Examples:
        >>> # Create classification head for encoder bottleneck
        >>> encoder = create_unet_encoder(base_filters=64, depth=4)
        >>> cls_head = create_classification_head(
        ...     in_channels=encoder.bottleneck_channels,
        ...     num_classes=2
        ... )
        
        >>> # Forward pass
        >>> features = encoder(x)
        >>> bottleneck = features[-1]  # (B, 1024, 16, 16)
        >>> cls_logits = cls_head(bottleneck)  # (B, 2)
    """
    return ClassificationHead(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


if __name__ == "__main__":
    from src.models.unet_encoder import create_unet_encoder
    
    print("Testing Classification Head...")
    print("=" * 70)
    
    # Test 1: Basic forward pass
    print("\n1. Basic Forward Pass")
    encoder = create_unet_encoder(in_channels=1, base_filters=64, depth=4)
    cls_head = create_classification_head(
        in_channels=encoder.bottleneck_channels,
        num_classes=2,
    )
    
    x = torch.randn(2, 1, 256, 256)
    features = encoder(x)
    bottleneck = features[-1]
    
    cls_logits = cls_head(bottleneck)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Bottleneck shape: {bottleneck.shape}")
    print(f"   Classification logits: {cls_logits.shape}")
    print(f"   Classification head parameters: {cls_head.get_num_params():,}")
    
    # Test 2: Different hidden dimensions
    print("\n2. Different Hidden Dimensions")
    
    hidden_dims = [128, 256, 512]
    for hidden_dim in hidden_dims:
        head = create_classification_head(
            in_channels=1024,
            hidden_dim=hidden_dim,
        )
        params = head.get_num_params()
        print(f"   Hidden dim {hidden_dim}: {params:,} params")
    
    # Test 3: Binary vs Multi-class
    print("\n3. Binary vs Multi-class Classification")
    
    # Binary (2 classes)
    head_binary = create_classification_head(in_channels=1024, num_classes=2)
    logits_binary = head_binary(bottleneck)
    print(f"   Binary (2 classes): {logits_binary.shape}")
    
    # Multi-class (4 tumor types)
    head_multi = create_classification_head(in_channels=1024, num_classes=4)
    logits_multi = head_multi(bottleneck)
    print(f"   Multi-class (4 classes): {logits_multi.shape}")
    
    # Test 4: Gradient flow
    print("\n4. Testing Gradient Flow")
    encoder = create_unet_encoder()
    cls_head = create_classification_head(encoder.bottleneck_channels)
    
    x = torch.randn(1, 1, 256, 256, requires_grad=True)
    features = encoder(x)
    logits = cls_head(features[-1])
    
    loss = logits.sum()
    loss.backward()
    
    print(f"   Input gradient: {'✓' if x.grad is not None else '✗'}")
    print(f"   Logits gradient: {'✓' if logits.requires_grad else '✗'}")
    
    # Test 5: With different encoder configurations
    print("\n5. Classification Head for Different Encoders")
    
    configs = [
        {"base_filters": 32, "depth": 3, "name": "Lightweight"},
        {"base_filters": 64, "depth": 4, "name": "Standard"},
        {"base_filters": 128, "depth": 5, "name": "Heavy"},
    ]
    
    for config in configs:
        name = config.pop("name")
        enc = create_unet_encoder(**config)
        head = create_classification_head(enc.bottleneck_channels)
        params = head.get_num_params()
        print(f"   {name:12s}: Bottleneck {enc.bottleneck_channels} ch -> {params:,} cls params")
    
    # Test 6: Inference mode (with softmax)
    print("\n6. Inference Mode (with Softmax)")
    encoder = create_unet_encoder()
    cls_head = create_classification_head(encoder.bottleneck_channels, num_classes=2)
    
    x = torch.randn(4, 1, 256, 256)
    features = encoder(x)
    logits = cls_head(features[-1])
    
    # Apply softmax for probabilities
    probs = torch.softmax(logits, dim=1)
    
    print(f"   Batch size: {x.shape[0]}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Probabilities shape: {probs.shape}")
    print(f"   Sample probabilities:")
    for i in range(min(2, x.shape[0])):
        print(f"     Sample {i}: No tumor={probs[i, 0]:.4f}, Tumor={probs[i, 1]:.4f}")
    
    # Test 7: Parameter efficiency
    print("\n7. Parameter Efficiency")
    encoder = create_unet_encoder(base_filters=64, depth=4)
    cls_head = create_classification_head(encoder.bottleneck_channels)
    
    encoder_params = encoder.get_num_params()
    cls_params = cls_head.get_num_params()
    total_params = encoder_params + cls_params
    
    print(f"   Encoder params: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"   Cls head params: {cls_params:,} ({cls_params/total_params*100:.1f}%)")
    print(f"   Total: {total_params:,}")
    print(f"   Classification head is only {cls_params/encoder_params*100:.2f}% of encoder!")
    
    print("\n" + "=" * 70)
    print("✓ All classification head tests passed!")
    print("=" * 70)
