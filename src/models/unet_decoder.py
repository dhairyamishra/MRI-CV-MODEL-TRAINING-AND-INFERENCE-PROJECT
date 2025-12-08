"""
U-Net Decoder for Segmentation.

Takes feature maps from encoder and produces segmentation mask
using skip connections and upsampling.
"""

import torch
import torch.nn as nn
from typing import List

from src.models.unet2d import Up, OutConv


class UNetDecoder(nn.Module):
    """
    U-Net Decoder for segmentation.
    
    Takes multi-scale features from encoder and upsamples
    with skip connections to produce segmentation mask.
    
    Args:
        encoder_channels: List of channel counts from encoder
            e.g., [64, 128, 256, 512, 1024] for depth=4
        out_channels: Number of output classes (1 for binary)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        encoder_channels: List[int],
        out_channels: int = 1,
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super(UNetDecoder, self).__init__()
        
        self.encoder_channels = encoder_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.dropout = dropout
        
        # Build upsampling blocks
        # encoder_channels = [64, 128, 256, 512, 1024]
        # We upsample from 1024 -> 512 -> 256 -> 128 -> 64
        
        self.up_blocks = nn.ModuleList()
        
        # Start from bottleneck (last encoder channel)
        in_ch = encoder_channels[-1]
        
        # Create upsampling blocks (reverse order)
        for i in range(len(encoder_channels) - 1, 0, -1):
            out_ch = encoder_channels[i - 1]
            self.up_blocks.append(Up(in_ch, out_ch, bilinear, dropout))
            in_ch = out_ch
        
        # Output convolution
        self.outc = OutConv(encoder_channels[0], out_channels)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            features: List of feature maps from encoder
                [x0, x1, x2, x3, bottleneck]
                where x0 is highest resolution
        
        Returns:
            logits: Segmentation logits (B, out_channels, H, W)
        """
        # Start from bottleneck
        x = features[-1]
        
        # Upsample with skip connections (reverse order)
        skip_connections = features[:-1]  # [x0, x1, x2, x3]
        
        for up, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = up(x, skip)
        
        # Output convolution
        logits = self.outc(x)
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unet_decoder(
    encoder_channels: List[int],
    out_channels: int = 1,
    bilinear: bool = True,
    dropout: float = 0.0,
) -> UNetDecoder:
    """
    Factory function to create U-Net decoder.
    
    Args:
        encoder_channels: Channel counts from encoder
        out_channels: Number of output classes (default: 1)
        bilinear: Use bilinear upsampling (default: True)
        dropout: Dropout rate (default: 0.0)
    
    Returns:
        UNetDecoder model
    
    Examples:
        >>> # Create decoder matching encoder
        >>> encoder_channels = [64, 128, 256, 512, 1024]
        >>> decoder = create_unet_decoder(encoder_channels, out_channels=1)
        
        >>> # Forward pass
        >>> features = encoder(x)  # From UNetEncoder
        >>> seg_logits = decoder(features)
    """
    return UNetDecoder(
        encoder_channels=encoder_channels,
        out_channels=out_channels,
        bilinear=bilinear,
        dropout=dropout,
    )


if __name__ == "__main__":
    from src.models.unet_encoder import create_unet_encoder
    
    print("Testing U-Net Decoder...")
    print("=" * 70)
    
    # Test 1: Basic forward pass
    print("\n1. Basic Forward Pass")
    encoder = create_unet_encoder(in_channels=1, base_filters=64, depth=4)
    encoder_channels = encoder.get_feature_channels()
    
    decoder = create_unet_decoder(encoder_channels, out_channels=1)
    
    x = torch.randn(2, 1, 256, 256)
    features = encoder(x)
    output = decoder(features)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Encoder channels: {encoder_channels}")
    print(f"   Output shape: {output.shape}")
    print(f"   Decoder parameters: {decoder.get_num_params():,}")
    
    # Test 2: Multi-class segmentation
    print("\n2. Multi-class Segmentation")
    decoder_multi = create_unet_decoder(encoder_channels, out_channels=4)
    output_multi = decoder_multi(features)
    
    print(f"   Output shape (4 classes): {output_multi.shape}")
    print(f"   Parameters: {decoder_multi.get_num_params():,}")
    
    # Test 3: Encoder + Decoder = Original U-Net
    print("\n3. Verifying Encoder + Decoder = U-Net")
    from src.models.unet2d import create_unet
    
    # Original U-Net
    unet_original = create_unet(in_channels=1, out_channels=1, base_filters=64, depth=4)
    output_original = unet_original(x)
    
    # Modular version
    encoder = create_unet_encoder(in_channels=1, base_filters=64, depth=4)
    decoder = create_unet_decoder(encoder.get_feature_channels(), out_channels=1)
    
    features = encoder(x)
    output_modular = decoder(features)
    
    print(f"   Original U-Net output: {output_original.shape}")
    print(f"   Modular version output: {output_modular.shape}")
    print(f"   Original params: {unet_original.get_num_params():,}")
    print(f"   Modular params: {encoder.get_num_params() + decoder.get_num_params():,}")
    
    # Test 4: Gradient flow
    print("\n4. Testing Gradient Flow")
    encoder = create_unet_encoder()
    decoder = create_unet_decoder(encoder.get_feature_channels())
    
    x = torch.randn(1, 1, 256, 256, requires_grad=True)
    features = encoder(x)
    output = decoder(features)
    
    loss = output.sum()
    loss.backward()
    
    print(f"   Input gradient: {'[OK]' if x.grad is not None else '✗'}")
    print(f"   Output gradient: {'[OK]' if output.requires_grad else '✗'}")
    
    # Test 5: Different configurations
    print("\n5. Different Decoder Configurations")
    
    configs = [
        {"base_filters": 32, "depth": 3, "name": "Lightweight"},
        {"base_filters": 64, "depth": 4, "name": "Standard"},
        {"base_filters": 128, "depth": 5, "name": "Heavy"},
    ]
    
    for config in configs:
        name = config.pop("name")
        enc = create_unet_encoder(**config)
        dec = create_unet_decoder(enc.get_feature_channels())
        params = dec.get_num_params()
        print(f"   {name:12s}: {params:>10,} decoder params")
    
    print("\n" + "=" * 70)
    print("[OK] All decoder tests passed!")
    print("=" * 70)
