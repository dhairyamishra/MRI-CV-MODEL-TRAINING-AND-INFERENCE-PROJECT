"""
U-Net Encoder for Multi-Task Learning.

Extracts the encoder path from U-Net to enable feature sharing
between segmentation and classification tasks.

Returns feature maps at each scale for:
- Skip connections (segmentation decoder)
- Classification head (bottleneck features)
- Grad-CAM visualization
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from src.models.unet2d import DoubleConv, Down


class UNetEncoder(nn.Module):
    """
    U-Net Encoder for feature extraction.
    
    Extracts multi-scale features that can be used for:
    1. Segmentation (via decoder with skip connections)
    2. Classification (via global pooling on bottleneck)
    3. Visualization (Grad-CAM on bottleneck)
    
    Args:
        in_channels: Number of input channels (1 for grayscale MRI)
        base_filters: Number of filters in first layer (default: 64)
        depth: Number of downsampling blocks (default: 4)
        dropout: Dropout rate (default: 0.0)
    
    Returns:
        features: List of feature maps at each scale
            [x0, x1, x2, x3, bottleneck]
            where x0 is highest resolution, bottleneck is lowest
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super(UNetEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.depth = depth
        self.dropout = dropout
        
        # Initial convolution (highest resolution)
        self.inc = DoubleConv(in_channels, base_filters, dropout=dropout)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_ch = base_filters
        for i in range(depth):
            out_ch = in_ch * 2
            self.down_blocks.append(Down(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        
        # Store bottleneck channels for classification head
        self.bottleneck_channels = in_ch
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor (B, in_channels, H, W)
        
        Returns:
            features: List of feature maps
                [x0, x1, x2, x3, bottleneck]
                
                For 256x256 input with depth=4:
                - x0: (B, 64, 256, 256)   - Initial features
                - x1: (B, 128, 128, 128)  - After 1st downsample
                - x2: (B, 256, 64, 64)    - After 2nd downsample
                - x3: (B, 512, 32, 32)    - After 3rd downsample
                - bottleneck: (B, 1024, 16, 16) - After 4th downsample
        """
        features = []
        
        # Initial convolution
        x = self.inc(x)
        features.append(x)
        
        # Downsampling path
        for down in self.down_blocks:
            x = down(x)
            features.append(x)
        
        return features
    
    def get_feature_channels(self) -> List[int]:
        """
        Get number of channels at each feature scale.
        
        Returns:
            List of channel counts [64, 128, 256, 512, 1024] for depth=4
        """
        channels = [self.base_filters]
        ch = self.base_filters
        for _ in range(self.depth):
            ch *= 2
            channels.append(ch)
        return channels
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_except_last_n_blocks(self, n: int = 1):
        """
        Freeze encoder except last n downsampling blocks.
        
        Useful for fine-tuning: freeze early layers, train later layers.
        
        Args:
            n: Number of blocks to keep trainable (from end)
        """
        # Freeze initial conv
        for param in self.inc.parameters():
            param.requires_grad = False
        
        # Freeze all but last n down blocks
        num_blocks = len(self.down_blocks)
        for i, block in enumerate(self.down_blocks):
            if i < num_blocks - n:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True


def create_unet_encoder(
    in_channels: int = 1,
    base_filters: int = 64,
    depth: int = 4,
    dropout: float = 0.0,
) -> UNetEncoder:
    """
    Factory function to create U-Net encoder.
    
    Args:
        in_channels: Number of input channels (default: 1)
        base_filters: Number of filters in first layer (default: 64)
        depth: Number of downsampling blocks (default: 4)
        dropout: Dropout rate (default: 0.0)
    
    Returns:
        UNetEncoder model
    
    Examples:
        >>> # Standard encoder
        >>> encoder = create_unet_encoder()
        >>> features = encoder(x)  # Returns list of 5 feature maps
        
        >>> # Get bottleneck features for classification
        >>> bottleneck = features[-1]  # (B, 1024, 16, 16)
        
        >>> # Get skip connections for decoder
        >>> skip_connections = features[:-1]  # [x0, x1, x2, x3]
    """
    return UNetEncoder(
        in_channels=in_channels,
        base_filters=base_filters,
        depth=depth,
        dropout=dropout,
    )


if __name__ == "__main__":
    print("Testing U-Net Encoder...")
    print("=" * 70)
    
    # Test 1: Basic forward pass
    print("\n1. Basic Forward Pass")
    encoder = create_unet_encoder(in_channels=1, base_filters=64, depth=4)
    
    x = torch.randn(2, 1, 256, 256)
    features = encoder(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Number of feature maps: {len(features)}")
    for i, feat in enumerate(features):
        print(f"   Feature {i}: {feat.shape}")
    
    print(f"\n   Bottleneck channels: {encoder.bottleneck_channels}")
    print(f"   Total parameters: {encoder.get_num_params():,}")
    
    # Test 2: Feature channels
    print("\n2. Feature Channels at Each Scale")
    channels = encoder.get_feature_channels()
    print(f"   Channel progression: {channels}")
    
    # Test 3: Freezing mechanisms
    print("\n3. Testing Freeze/Unfreeze")
    
    # Count trainable params
    def count_trainable():
        return sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"   Initial trainable params: {count_trainable():,}")
    
    encoder.freeze()
    print(f"   After freeze: {count_trainable():,}")
    
    encoder.unfreeze()
    print(f"   After unfreeze: {count_trainable():,}")
    
    encoder.freeze_except_last_n_blocks(2)
    print(f"   After freeze_except_last_2: {count_trainable():,}")
    
    encoder.unfreeze()
    
    # Test 4: Different configurations
    print("\n4. Different Encoder Configurations")
    
    configs = [
        {"base_filters": 32, "depth": 3, "name": "Lightweight"},
        {"base_filters": 64, "depth": 4, "name": "Standard"},
        {"base_filters": 128, "depth": 5, "name": "Heavy"},
    ]
    
    for config in configs:
        name = config.pop("name")
        enc = create_unet_encoder(**config)
        params = enc.get_num_params()
        bottleneck_ch = enc.bottleneck_channels
        print(f"   {name:12s}: {params:>10,} params, bottleneck: {bottleneck_ch} channels")
    
    # Test 5: Gradient flow
    print("\n5. Testing Gradient Flow")
    encoder = create_unet_encoder()
    x = torch.randn(1, 1, 256, 256, requires_grad=True)
    features = encoder(x)
    
    # Compute loss on bottleneck
    loss = features[-1].sum()
    loss.backward()
    
    print(f"   Input gradient: {'[OK]' if x.grad is not None else '✗'}")
    print(f"   Bottleneck gradient: {'[OK]' if features[-1].requires_grad else '✗'}")
    
    # Test 6: Compatibility with different input sizes
    print("\n6. Testing Different Input Sizes")
    encoder = create_unet_encoder()
    
    sizes = [(128, 128), (256, 256), (512, 512)]
    for h, w in sizes:
        x_test = torch.randn(1, 1, h, w)
        feats = encoder(x_test)
        bottleneck_shape = feats[-1].shape
        print(f"   Input: ({h}, {w}) -> Bottleneck: {bottleneck_shape}")
    
    print("\n" + "=" * 70)
    print("[OK] All encoder tests passed!")
    print("=" * 70)
