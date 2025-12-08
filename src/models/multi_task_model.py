"""
Multi-Task Model for Joint Segmentation and Classification.

Combines UNetEncoder, UNetDecoder, and ClassificationHead
to perform both tumor segmentation and classification from
a shared encoder.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

from src.models.unet_encoder import UNetEncoder, create_unet_encoder
from src.models.unet_decoder import UNetDecoder, create_unet_decoder
from src.models.classification_head import ClassificationHead, create_classification_head


class MultiTaskModel(nn.Module):
    """
    Multi-task model for segmentation and classification.
    
    Architecture:
        Input (B, 1, 256, 256)
            ↓
        Shared Encoder → Features [x0, x1, x2, x3, bottleneck]
            ↓                           ↓
        Decoder (skip connections)   Classification Head
            ↓                           ↓
        Segmentation Mask (B, 1, 256, 256)   Class Logits (B, 2)
    
    Args:
        in_channels: Number of input channels (default: 1)
        seg_out_channels: Number of segmentation output channels (default: 1)
        cls_num_classes: Number of classification classes (default: 2)
        base_filters: Base number of filters (default: 64)
        depth: Encoder/decoder depth (default: 4)
        bilinear: Use bilinear upsampling (default: True)
        dropout: Dropout rate (default: 0.0)
        cls_hidden_dim: Classification head hidden dimension (default: 256)
        cls_dropout: Classification head dropout (default: 0.5)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        seg_out_channels: int = 1,
        cls_num_classes: int = 2,
        base_filters: int = 64,
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.0,
        cls_hidden_dim: int = 256,
        cls_dropout: float = 0.5,
    ):
        super(MultiTaskModel, self).__init__()
        
        self.in_channels = in_channels
        self.seg_out_channels = seg_out_channels
        self.cls_num_classes = cls_num_classes
        self.base_filters = base_filters
        self.depth = depth
        
        # Shared encoder
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_filters=base_filters,
            depth=depth,
            dropout=dropout,
        )
        
        # Segmentation decoder
        encoder_channels = self.encoder.get_feature_channels()
        self.seg_decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            out_channels=seg_out_channels,
            bilinear=bilinear,
            dropout=dropout,
        )
        
        # Classification head
        self.cls_head = ClassificationHead(
            in_channels=self.encoder.bottleneck_channels,
            num_classes=cls_num_classes,
            hidden_dim=cls_hidden_dim,
            dropout=cls_dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        do_seg: bool = True,
        do_cls: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task model.
        
        Args:
            x: Input tensor (B, in_channels, H, W)
            do_seg: Whether to compute segmentation (default: True)
            do_cls: Whether to compute classification (default: True)
        
        Returns:
            Dictionary with keys:
            - 'seg': Segmentation logits (B, seg_out_channels, H, W) if do_seg
            - 'cls': Classification logits (B, cls_num_classes) if do_cls
            - 'features': List of encoder features (for Grad-CAM, etc.)
        """
        # Shared encoder
        features = self.encoder(x)
        
        output = {'features': features}
        
        # Segmentation branch
        if do_seg:
            seg_logits = self.seg_decoder(features)
            output['seg'] = seg_logits
        
        # Classification branch
        if do_cls:
            bottleneck = features[-1]
            cls_logits = self.cls_head(bottleneck)
            output['cls'] = cls_logits
        
        return output
    
    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        return {
            'encoder': self.encoder.get_num_params(),
            'seg_decoder': self.seg_decoder.get_num_params(),
            'cls_head': self.cls_head.get_num_params(),
            'total': sum(p.numel() for p in self.parameters()),
        }
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        self.encoder.freeze()
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        self.encoder.unfreeze()
    
    def freeze_encoder_except_last_n_blocks(self, n: int = 1):
        """Freeze encoder except last n blocks."""
        self.encoder.freeze_except_last_n_blocks(n)
    
    def freeze_seg_decoder(self):
        """Freeze segmentation decoder."""
        for param in self.seg_decoder.parameters():
            param.requires_grad = False
    
    def unfreeze_seg_decoder(self):
        """Unfreeze segmentation decoder."""
        for param in self.seg_decoder.parameters():
            param.requires_grad = True
    
    def freeze_cls_head(self):
        """Freeze classification head."""
        for param in self.cls_head.parameters():
            param.requires_grad = False
    
    def unfreeze_cls_head(self):
        """Unfreeze classification head."""
        for param in self.cls_head.parameters():
            param.requires_grad = True
    
    def load_encoder_from_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """
        Load encoder weights from a checkpoint.
        
        Useful for initializing from a pretrained segmentation model.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce key matching
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract encoder state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter encoder keys
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.') or key.startswith('inc.') or key.startswith('down_blocks.'):
                # Remove 'encoder.' prefix if present
                new_key = key.replace('encoder.', '')
                encoder_state_dict[new_key] = value
        
        # Load into encoder
        self.encoder.load_state_dict(encoder_state_dict, strict=strict)
        print(f"Loaded encoder weights from {checkpoint_path}")


def create_multi_task_model(
    in_channels: int = 1,
    seg_out_channels: int = 1,
    cls_num_classes: int = 2,
    base_filters: int = 64,
    depth: int = 4,
    bilinear: bool = True,
    dropout: float = 0.0,
    cls_hidden_dim: int = 256,
    cls_dropout: float = 0.5,
) -> MultiTaskModel:
    """
    Factory function to create multi-task model.
    
    Args:
        in_channels: Number of input channels (default: 1)
        seg_out_channels: Segmentation output channels (default: 1)
        cls_num_classes: Classification classes (default: 2)
        base_filters: Base filters (default: 64)
        depth: Encoder depth (default: 4)
        bilinear: Use bilinear upsampling (default: True)
        dropout: Encoder/decoder dropout (default: 0.0)
        cls_hidden_dim: Classification hidden dim (default: 256)
        cls_dropout: Classification dropout (default: 0.5)
    
    Returns:
        MultiTaskModel
    
    Examples:
        >>> # Create standard multi-task model
        >>> model = create_multi_task_model()
        
        >>> # Forward pass (both tasks)
        >>> output = model(x)
        >>> seg_logits = output['seg']  # (B, 1, 256, 256)
        >>> cls_logits = output['cls']  # (B, 2)
        
        >>> # Forward pass (segmentation only)
        >>> output = model(x, do_seg=True, do_cls=False)
        
        >>> # Forward pass (classification only)
        >>> output = model(x, do_seg=False, do_cls=True)
    """
    return MultiTaskModel(
        in_channels=in_channels,
        seg_out_channels=seg_out_channels,
        cls_num_classes=cls_num_classes,
        base_filters=base_filters,
        depth=depth,
        bilinear=bilinear,
        dropout=dropout,
        cls_hidden_dim=cls_hidden_dim,
        cls_dropout=cls_dropout,
    )


if __name__ == "__main__":
    print("Testing Multi-Task Model...")
    print("=" * 70)
    
    # Test 1: Basic forward pass (both tasks)
    print("\n1. Basic Forward Pass (Both Tasks)")
    model = create_multi_task_model()
    
    x = torch.randn(2, 1, 256, 256)
    output = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Segmentation output: {output['seg'].shape}")
    print(f"   Classification output: {output['cls'].shape}")
    print(f"   Number of feature maps: {len(output['features'])}")
    
    params = model.get_num_params()
    print(f"\n   Parameter breakdown:")
    for key, value in params.items():
        print(f"     {key}: {value:,}")
    
    # Test 2: Selective forward pass
    print("\n2. Selective Forward Pass")
    
    # Segmentation only
    output_seg = model(x, do_seg=True, do_cls=False)
    print(f"   Seg only: {list(output_seg.keys())}")
    
    # Classification only
    output_cls = model(x, do_seg=False, do_cls=True)
    print(f"   Cls only: {list(output_cls.keys())}")
    
    # Test 3: Freezing mechanisms
    print("\n3. Testing Freeze/Unfreeze")
    
    def count_trainable():
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Initial trainable: {count_trainable():,}")
    
    model.freeze_encoder()
    print(f"   After freeze encoder: {count_trainable():,}")
    
    model.unfreeze_encoder()
    model.freeze_seg_decoder()
    print(f"   After freeze seg decoder: {count_trainable():,}")
    
    model.unfreeze_seg_decoder()
    model.freeze_cls_head()
    print(f"   After freeze cls head: {count_trainable():,}")
    
    model.unfreeze_cls_head()
    model.freeze_encoder_except_last_n_blocks(2)
    print(f"   After freeze encoder except last 2: {count_trainable():,}")
    
    model.unfreeze_encoder()
    
    # Test 4: Gradient flow
    print("\n4. Testing Gradient Flow")
    model = create_multi_task_model()
    
    x = torch.randn(1, 1, 256, 256, requires_grad=True)
    output = model(x)
    
    # Compute combined loss
    seg_loss = output['seg'].sum()
    cls_loss = output['cls'].sum()
    total_loss = seg_loss + cls_loss
    total_loss.backward()
    
    print(f"   Input gradient: {'[OK]' if x.grad is not None else '✗'}")
    print(f"   Seg output gradient: {'[OK]' if output['seg'].requires_grad else '✗'}")
    print(f"   Cls output gradient: {'[OK]' if output['cls'].requires_grad else '✗'}")
    
    # Test 5: Different configurations
    print("\n5. Different Model Configurations")
    
    configs = [
        {"base_filters": 32, "depth": 3, "name": "Lightweight"},
        {"base_filters": 64, "depth": 4, "name": "Standard"},
        {"base_filters": 128, "depth": 5, "name": "Heavy"},
    ]
    
    for config in configs:
        name = config.pop("name")
        m = create_multi_task_model(**config)
        params = m.get_num_params()
        print(f"   {name:12s}: {params['total']:>10,} total params")
        print(f"                 Encoder: {params['encoder']:>10,} ({params['encoder']/params['total']*100:.1f}%)")
        print(f"                 Seg Dec: {params['seg_decoder']:>10,} ({params['seg_decoder']/params['total']*100:.1f}%)")
        print(f"                 Cls Head: {params['cls_head']:>10,} ({params['cls_head']/params['total']*100:.1f}%)")
    
    # Test 6: Batch inference
    print("\n6. Batch Inference")
    model = create_multi_task_model()
    model.eval()
    
    with torch.no_grad():
        x_batch = torch.randn(8, 1, 256, 256)
        output = model(x_batch)
        
        # Apply activations
        seg_probs = torch.sigmoid(output['seg'])
        cls_probs = torch.softmax(output['cls'], dim=1)
        
        print(f"   Batch size: {x_batch.shape[0]}")
        print(f"   Seg probs range: [{seg_probs.min():.4f}, {seg_probs.max():.4f}]")
        print(f"   Cls probs shape: {cls_probs.shape}")
        print(f"   Sample cls probs: No tumor={cls_probs[0, 0]:.4f}, Tumor={cls_probs[0, 1]:.4f}")
    
    # Test 7: Comparison with separate models
    print("\n7. Comparison with Separate Models")
    from src.models.unet2d import create_unet
    from src.models.classifier import create_efficientnet_classifier
    
    # Multi-task model
    multi_task = create_multi_task_model(base_filters=64, depth=4)
    multi_task_params = multi_task.get_num_params()['total']
    
    # Separate models
    unet = create_unet(in_channels=1, out_channels=1, base_filters=64, depth=4)
    unet_params = unet.get_num_params()
    
    # Note: EfficientNet has ~4M params
    efficientnet_params = 4_000_000  # Approximate
    
    separate_total = unet_params + efficientnet_params
    
    print(f"   Multi-task model: {multi_task_params:,} params")
    print(f"   Separate models: {separate_total:,} params")
    print(f"   Savings: {separate_total - multi_task_params:,} params ({(1 - multi_task_params/separate_total)*100:.1f}%)")
    print(f"   Multi-task is {multi_task_params/separate_total*100:.1f}% the size!")
    
    print("\n" + "=" * 70)
    print("[OK] All multi-task model tests passed!")
    print("=" * 70)
