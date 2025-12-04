"""
Brain Tumor Classifier using EfficientNet-B0 backbone.

This module implements a binary classifier for detecting the presence of brain tumors
in MRI slices. It adapts EfficientNet-B0 for single-channel (grayscale) input.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class BrainTumorClassifier(nn.Module):
    """
    Binary classifier for brain tumor detection in MRI slices.
    
    Uses EfficientNet-B0 as the backbone, adapted for single-channel input.
    
    Args:
        pretrained (bool): Whether to use ImageNet pretrained weights. Default: True
        num_classes (int): Number of output classes. Default: 2 (tumor/no tumor)
        dropout (float): Dropout rate before final classifier. Default: 0.3
        freeze_backbone (bool): Whether to freeze backbone weights. Default: False
    
    Input:
        x: Tensor of shape (B, 1, H, W) - single-channel MRI slices
    
    Output:
        logits: Tensor of shape (B, num_classes) - class logits
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super(BrainTumorClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Load EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Adapt first conv layer for single-channel input
        # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        original_conv = self.backbone.features[0][0]
        
        # Create new conv layer with 1 input channel
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # If pretrained, average the RGB weights to initialize single-channel weights
        if pretrained:
            with torch.no_grad():
                # Average across the 3 input channels
                self.backbone.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
        # Store feature extractor for Grad-CAM
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        self.classifier = self.backbone.classifier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
        
        Returns:
            logits: Output tensor of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone (useful for Grad-CAM).
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
        
        Returns:
            features: Feature maps from the last conv layer
        """
        x = self.features(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for extract_features() - used by CAM regularization loss.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
        
        Returns:
            features: Feature maps from the last conv layer
        """
        return self.extract_features(x)
    
    def get_cam_target_layer(self):
        """
        Get the target layer for Grad-CAM visualization.
        
        Returns:
            target_layer: The last convolutional layer
        """
        # For EfficientNet-B0, the last conv layer is features[-1]
        return self.features[-1]
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = True
    
    def get_num_trainable_params(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class ConvNeXtClassifier(nn.Module):
    """
    Alternative classifier using ConvNeXt-Tiny backbone.
    
    ConvNeXt is a modern CNN architecture that achieves competitive performance
    with vision transformers while being more efficient.
    
    Args:
        pretrained (bool): Whether to use ImageNet pretrained weights. Default: True
        num_classes (int): Number of output classes. Default: 2
        dropout (float): Dropout rate before final classifier. Default: 0.3
        freeze_backbone (bool): Whether to freeze backbone weights. Default: False
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super(ConvNeXtClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Load ConvNeXt-Tiny
        if pretrained:
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            self.backbone = models.convnext_tiny(weights=weights)
        else:
            self.backbone = models.convnext_tiny(weights=None)
        
        # Adapt first conv layer for single-channel input
        original_conv = self.backbone.features[0][0]
        
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize with averaged RGB weights if pretrained
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
                if original_conv.bias is not None:
                    self.backbone.features[0][0].bias = original_conv.bias
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Get the number of features
        num_features = self.backbone.classifier[2].in_features
        
        # Replace classifier head
        self.backbone.classifier[2] = nn.Linear(num_features, num_classes)
        
        # Add dropout before classifier
        self.backbone.classifier[0] = nn.Flatten(1)
        self.backbone.classifier[1] = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier."""
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone (useful for Grad-CAM).
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
        
        Returns:
            features: Feature maps from the last conv layer
        """
        x = self.backbone.features(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for extract_features() - used by CAM regularization loss.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
        
        Returns:
            features: Feature maps from the last conv layer
        """
        return self.extract_features(x)
    
    def get_cam_target_layer(self):
        """Get the target layer for Grad-CAM."""
        return self.backbone.features[-1]


def create_classifier(
    model_name: str = "efficientnet",
    pretrained: bool = True,
    num_classes: int = 2,
    dropout: float = 0.3,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Factory function to create a classifier model.
    
    Args:
        model_name: Name of the model architecture ('efficientnet' or 'convnext')
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes
        dropout: Dropout rate
        freeze_backbone: Whether to freeze backbone weights
    
    Returns:
        model: Initialized classifier model
    
    Example:
        >>> model = create_classifier('efficientnet', pretrained=True)
        >>> print(f"Trainable params: {model.get_num_trainable_params():,}")
    """
    model_name = model_name.lower()
    
    if model_name == "efficientnet":
        return BrainTumorClassifier(
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
    elif model_name == "convnext":
        return ConvNeXtClassifier(
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'efficientnet' or 'convnext'.")


if __name__ == "__main__":
    # Test the classifier
    print("Testing BrainTumorClassifier...")
    
    model = BrainTumorClassifier(pretrained=False)
    print(f"Total parameters: {model.get_num_total_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    # Test forward pass
    x = torch.randn(4, 1, 256, 256)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test feature extraction
    features = model.extract_features(x)
    print(f"Feature map shape: {features.shape}")
    
    print("\nAll tests passed! ")
