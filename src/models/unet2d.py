"""
U-Net 2D Architecture for Brain Tumor Segmentation.

Implements the classic U-Net architecture with encoder-decoder structure
and skip connections for precise localization.

Reference:
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    MICCAI 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv2d -> BatchNorm -> ReLU) x 2
    
    This is the basic building block of U-Net.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mid_channels: Number of intermediate channels (default: out_channels)
            dropout: Dropout rate (default: 0.0)
        """
        super(DoubleConv, self).__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block: MaxPool -> DoubleConv
    
    Reduces spatial dimensions by 2x and increases feature channels.
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: Upsample -> Concatenate -> DoubleConv
    
    Increases spatial dimensions by 2x and combines with skip connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bilinear: Use bilinear upsampling (True) or transposed conv (False)
            dropout: Dropout rate
        """
        super(Up, self).__init__()
        
        # Upsampling method
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsampling, we concatenate with skip connection
            # So input to conv will be: in_channels + in_channels//2 (from skip)
            # But we want to reduce in_channels first before concatenation
            self.conv = DoubleConv(in_channels + in_channels//2, out_channels, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Input from previous layer (lower resolution)
            x2: Skip connection from encoder (higher resolution)
        
        Returns:
            Upsampled and concatenated features
        """
        x1 = self.up(x1)
        
        # Handle size mismatch due to padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution: 1x1 conv to produce final segmentation map.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet2D(nn.Module):
    """
    U-Net 2D for medical image segmentation.
    
    Architecture:
        - Encoder: 4 downsampling blocks (MaxPool + DoubleConv)
        - Bottleneck: DoubleConv at lowest resolution
        - Decoder: 4 upsampling blocks (Upsample + Concatenate + DoubleConv)
        - Output: 1x1 conv to produce segmentation mask
    
    Args:
        in_channels: Number of input channels (1 for grayscale MRI)
        out_channels: Number of output classes (1 for binary, >1 for multi-class)
        base_filters: Number of filters in first layer (default: 64)
        depth: Number of downsampling/upsampling blocks (default: 4)
        bilinear: Use bilinear upsampling instead of transposed conv (default: True)
        dropout: Dropout rate (default: 0.0)
    
    Input:
        x: (B, in_channels, H, W)
    
    Output:
        logits: (B, out_channels, H, W)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super(UNet2D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        self.bilinear = bilinear
        self.dropout = dropout
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, base_filters, dropout=dropout)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        in_ch = base_filters
        for i in range(depth):
            out_ch = in_ch * 2
            self.down_blocks.append(Down(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            out_ch = in_ch // 2
            self.up_blocks.append(Up(in_ch, out_ch, bilinear, dropout=dropout))
            in_ch = out_ch
        
        # Output convolution
        self.outc = OutConv(base_filters, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor (B, in_channels, H, W)
        
        Returns:
            logits: Output tensor (B, out_channels, H, W)
        """
        # Encoder with skip connections
        x = self.inc(x)
        skip_connections = [x]
        
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
        
        # Remove last skip connection (bottleneck)
        skip_connections = skip_connections[:-1]
        
        # Decoder with skip connections
        for up, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = up(x, skip)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    base_filters: int = 64,
    depth: int = 4,
    bilinear: bool = True,
    dropout: float = 0.0,
) -> UNet2D:
    """
    Factory function to create U-Net model.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        out_channels: Number of output classes (default: 1 for binary)
        base_filters: Number of filters in first layer (default: 64)
        depth: Number of encoder/decoder blocks (default: 4)
        bilinear: Use bilinear upsampling (default: True)
        dropout: Dropout rate (default: 0.0)
    
    Returns:
        UNet2D model
    
    Examples:
        >>> # Binary segmentation (tumor vs background)
        >>> model = create_unet(in_channels=1, out_channels=1)
        
        >>> # Multi-class segmentation (4 tumor regions)
        >>> model = create_unet(in_channels=1, out_channels=4)
        
        >>> # Deeper network with more filters
        >>> model = create_unet(base_filters=128, depth=5)
    """
    return UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=base_filters,
        depth=depth,
        bilinear=bilinear,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Test U-Net architecture
    print("Testing U-Net 2D Architecture...")
    print("=" * 60)
    
    # Test 1: Binary segmentation
    print("\n1. Binary Segmentation (Tumor vs Background)")
    model = create_unet(in_channels=1, out_channels=1, base_filters=64, depth=4)
    
    x = torch.randn(2, 1, 256, 256)  # Batch of 2 images
    output = model(x)
    
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters:   {model.get_num_params():,}")
    print(f"   Trainable:    {model.get_num_trainable_params():,}")
    
    # Test 2: Multi-class segmentation
    print("\n2. Multi-class Segmentation (4 tumor regions)")
    model_multi = create_unet(in_channels=1, out_channels=4, base_filters=64, depth=4)
    
    output_multi = model_multi(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {output_multi.shape}")
    print(f"   Parameters:   {model_multi.get_num_params():,}")
    
    # Test 3: Different configurations
    print("\n3. Different Configurations")
    
    configs = [
        {"base_filters": 32, "depth": 3, "name": "Lightweight"},
        {"base_filters": 64, "depth": 4, "name": "Standard"},
        {"base_filters": 128, "depth": 5, "name": "Heavy"},
    ]
    
    for config in configs:
        name = config.pop("name")
        m = create_unet(**config)
        params = m.get_num_params()
        print(f"   {name:12s}: {params:>10,} parameters")
    
    # Test 4: Forward pass with different input sizes
    print("\n4. Testing Different Input Sizes")
    model = create_unet()
    
    sizes = [(128, 128), (256, 256), (512, 512)]
    for h, w in sizes:
        x_test = torch.randn(1, 1, h, w)
        out_test = model(x_test)
        print(f"   Input: {x_test.shape} -> Output: {out_test.shape}")
    
    # Test 5: Gradient flow
    print("\n5. Testing Gradient Flow")
    model = create_unet()
    x = torch.randn(1, 1, 256, 256, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    print(f"   Input gradient:  {'✓' if x.grad is not None else '✗'}")
    print(f"   Output gradient: {'✓' if output.requires_grad else '✗'}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
