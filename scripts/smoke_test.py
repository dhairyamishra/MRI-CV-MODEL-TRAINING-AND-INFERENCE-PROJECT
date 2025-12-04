#!/usr/bin/env python3
"""
Smoke test for SliceWise project.
Tests basic functionality: load fake slice, pass through tiny U-Net, save output.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class TinyUNet(nn.Module):
    """Minimal U-Net for smoke testing."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())

        # Decoder
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
        )

        # Output
        self.out = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)

        # Bottleneck
        x = self.bottleneck(x2)

        # Decoder (simplified - no skip connections for smoke test)
        x = self.dec2(x)
        x = self.dec1(x)

        # Output
        x = self.out(x)
        return x


def create_fake_mri_slice(size=(256, 256)):
    """Create a fake MRI slice with a circular 'tumor'."""
    image = np.zeros(size, dtype=np.float32)

    # Add some background noise
    image += np.random.randn(*size) * 0.1

    # Add a circular "tumor" in the center
    center = (size[0] // 2, size[1] // 2)
    radius = 30
    y, x = np.ogrid[: size[0], : size[1]]
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius**2
    image[mask] = 1.0

    # Normalize
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    return image


def main():
    print("=" * 60)
    print("SliceWise Smoke Test")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create output directory
    output_dir = Path("assets/smoke_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Created output directory: {output_dir}")

    # Step 1: Create fake MRI slice
    print("\n[1/4] Creating fake MRI slice...")
    fake_slice = create_fake_mri_slice(size=(256, 256))
    print(f"  - Shape: {fake_slice.shape}")
    print(f"  - Value range: [{fake_slice.min():.3f}, {fake_slice.max():.3f}]")

    # Save input image
    input_img = Image.fromarray((fake_slice * 255).astype(np.uint8))
    input_path = output_dir / "input_slice.png"
    input_img.save(input_path)
    print(f"  - Saved to: {input_path}")

    # Step 2: Create tiny U-Net
    print("\n[2/4] Creating tiny U-Net model...")
    model = TinyUNet(in_channels=1, out_channels=1)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  - Number of parameters: {num_params:,}")

    # Step 3: Run inference
    print("\n[3/4] Running inference...")
    with torch.no_grad():
        # Convert to tensor and add batch/channel dimensions
        input_tensor = torch.from_numpy(fake_slice).unsqueeze(0).unsqueeze(0)
        print(f"  - Input tensor shape: {input_tensor.shape}")

        # Forward pass
        output_tensor = model(input_tensor)
        print(f"  - Output tensor shape: {output_tensor.shape}")

        # Apply sigmoid to get probabilities
        output_probs = torch.sigmoid(output_tensor)

        # Convert to numpy
        output_mask = output_probs.squeeze().numpy()
        print(f"  - Output value range: [{output_mask.min():.3f}, {output_mask.max():.3f}]")

    # Step 4: Save output
    print("\n[4/4] Saving output mask...")
    output_img = Image.fromarray((output_mask * 255).astype(np.uint8))
    output_path = output_dir / "output_mask.png"
    output_img.save(output_path)
    print(f"  - Saved to: {output_path}")

    # Create overlay
    print("\n[5/5] Creating overlay visualization...")
    overlay = np.stack([fake_slice, fake_slice, fake_slice], axis=-1)
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], output_mask)  # Red channel for mask
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    overlay_path = output_dir / "overlay.png"
    overlay_img.save(overlay_path)
    print(f"  - Saved to: {overlay_path}")

    print("\n" + "=" * 60)
    print("✓ Smoke test completed successfully!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - {input_path}")
    print(f"  - {output_path}")
    print(f"  - {overlay_path}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ Smoke test failed with error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
