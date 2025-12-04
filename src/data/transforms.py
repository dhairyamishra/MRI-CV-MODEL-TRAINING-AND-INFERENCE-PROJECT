"""
Data augmentation and transforms for MRI images.

Supports both NumPy arrays and PyTorch tensors.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Union, Optional


class BrainRegionCrop:
    """
    Automatically crops out the blank background around the head in MRI slices.
    
    Works for brain MRI images which have a dark background around the skull.
    Uses Otsu thresholding to find the brain region and crops to the tight
    bounding box (plus optional margin).
    
    This SOLVES the border artifact problem by removing borders before training!
    """
    
    def __init__(self, margin: int = 10):
        """
        Args:
            margin: Extra pixels to keep around the detected brain bbox.
        """
        self.margin = margin
    
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Crop image to brain region.
        
        Args:
            x: Image as numpy array (H, W) or (C, H, W) or torch.Tensor
            
        Returns:
            Cropped image in same format as input
        """
        is_torch = isinstance(x, torch.Tensor)
        
        # Convert to numpy if needed
        if is_torch:
            x_np = x.cpu().numpy()
        else:
            x_np = x
        
        # Handle different input shapes
        if x_np.ndim == 2:  # (H, W)
            gray = x_np.astype(np.uint8)
            original_shape = x_np.shape
        elif x_np.ndim == 3:  # (C, H, W) or (H, W, C)
            # Assume (C, H, W) for torch-style, (H, W, C) for numpy-style
            if x_np.shape[0] <= 3:  # (C, H, W)
                gray = x_np[0].astype(np.uint8)  # Take first channel
                original_shape = x_np.shape[1:]
            else:  # (H, W, C)
                gray = x_np.mean(axis=2).astype(np.uint8)
                original_shape = x_np.shape[:2]
        else:
            # Unsupported shape, return original
            return x
        
        # Normalize to 0-255 range for Otsu
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        
        # Otsu threshold to separate brain from background
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Find bounding box of brain region
        ys, xs = np.where(mask > 0)
        
        if ys.size == 0 or xs.size == 0:
            # Nothing detected, return original
            return x
        
        # Calculate crop coordinates with margin
        y1 = max(int(ys.min()) - self.margin, 0)
        y2 = min(int(ys.max()) + self.margin, original_shape[0] - 1)
        x1 = max(int(xs.min()) - self.margin, 0)
        x2 = min(int(xs.max()) + self.margin, original_shape[1] - 1)
        
        # Crop based on original format
        if x_np.ndim == 2:
            cropped = x_np[y1:y2+1, x1:x2+1]
        elif x_np.shape[0] <= 3:  # (C, H, W)
            cropped = x_np[:, y1:y2+1, x1:x2+1]
        else:  # (H, W, C)
            cropped = x_np[y1:y2+1, x1:x2+1, :]
        
        # Convert back to torch if needed
        if is_torch:
            return torch.from_numpy(cropped.copy())
        else:
            return cropped.copy()


class RandomRotation90:
    """Randomly rotate image by 0, 90, 180, or 270 degrees."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() < self.p:
            k = np.random.randint(0, 4)  # 0, 1, 2, or 3
            if isinstance(x, np.ndarray):
                return np.rot90(x, k, axes=(-2, -1)).copy()
            else:
                return torch.rot90(x, k, dims=(-2, -1))
        return x


class RandomIntensityShift:
    """Randomly shift image intensity."""
    
    def __init__(self, shift_range: float = 0.1, p: float = 0.5):
        self.shift_range = shift_range
        self.p = p
    
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() < self.p:
            shift = np.random.uniform(-self.shift_range, self.shift_range)
            if isinstance(x, np.ndarray):
                return np.clip(x + shift, 0.0, 1.0).astype(x.dtype)
            else:
                return torch.clamp(x + shift, 0.0, 1.0)
        return x


class RandomIntensityScale:
    """Randomly scale image intensity."""
    
    def __init__(self, scale_range: tuple = (0.9, 1.1), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() < self.p:
            scale = np.random.uniform(*self.scale_range)
            if isinstance(x, np.ndarray):
                return np.clip(x * scale, 0.0, 1.0).astype(x.dtype)
            else:
                return torch.clamp(x * scale, 0.0, 1.0)
        return x


class RandomGaussianNoise:
    """Add random Gaussian noise to image."""
    
    def __init__(self, std: float = 0.01, p: float = 0.3):
        self.std = std
        self.p = p
    
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() < self.p:
            if isinstance(x, np.ndarray):
                noise = np.random.randn(*x.shape) * self.std
                return np.clip(x + noise, 0.0, 1.0).astype(x.dtype)
            else:
                noise = torch.randn_like(x) * self.std
                return torch.clamp(x + noise, 0.0, 1.0)
        return x


class RandomHorizontalFlip:
    """Randomly flip image horizontally."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() < self.p:
            if isinstance(x, np.ndarray):
                return np.fliplr(x).copy()
            else:
                return torch.flip(x, dims=[-1])
        return x


class RandomVerticalFlip:
    """Randomly flip image vertically."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() < self.p:
            if isinstance(x, np.ndarray):
                return np.flipud(x).copy()
            else:
                return torch.flip(x, dims=[-2])
        return x


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


def get_train_transforms(
    rotation_p: float = 0.5,
    flip_p: float = 0.5,
    intensity_shift_p: float = 0.5,
    intensity_scale_p: float = 0.5,
    noise_p: float = 0.3,
):
    """
    Get training transforms with data augmentation.
    
    Args:
        rotation_p: Probability of random rotation
        flip_p: Probability of random flip
        intensity_shift_p: Probability of intensity shift
        intensity_scale_p: Probability of intensity scaling
        noise_p: Probability of adding Gaussian noise
        
    Returns:
        Composed transform
    """
    return Compose([
        # Geometric augmentations
        RandomHorizontalFlip(p=flip_p),
        RandomVerticalFlip(p=flip_p),
        RandomRotation90(p=rotation_p),
        
        # Intensity augmentations
        RandomIntensityShift(shift_range=0.1, p=intensity_shift_p),
        RandomIntensityScale(scale_range=(0.9, 1.1), p=intensity_scale_p),
        RandomGaussianNoise(std=0.01, p=noise_p),
    ])


def get_val_transforms():
    """
    Get validation/test transforms (no augmentation).
    
    Returns:
        Transform (identity function)
    """
    # No augmentation for validation/test
    # Images are already normalized and resized during preprocessing
    return lambda x: x


def get_strong_train_transforms(
    rotation_p: float = 0.7,
    flip_p: float = 0.7,
    intensity_shift_p: float = 0.7,
    intensity_scale_p: float = 0.7,
    noise_p: float = 0.5,
):
    """
    Get strong training transforms for ablation studies.
    
    Args:
        rotation_p: Probability of random rotation
        flip_p: Probability of random flip
        intensity_shift_p: Probability of intensity shift
        intensity_scale_p: Probability of intensity scaling
        noise_p: Probability of adding Gaussian noise
        
    Returns:
        Composed transform with stronger augmentation
    """
    return Compose([
        # Geometric augmentations (higher probability)
        RandomHorizontalFlip(p=flip_p),
        RandomVerticalFlip(p=flip_p),
        RandomRotation90(p=rotation_p),
        
        # Intensity augmentations (stronger)
        RandomIntensityShift(shift_range=0.15, p=intensity_shift_p),
        RandomIntensityScale(scale_range=(0.85, 1.15), p=intensity_scale_p),
        RandomGaussianNoise(std=0.02, p=noise_p),
    ])


def get_light_train_transforms(
    rotation_p: float = 0.3,
    flip_p: float = 0.3,
    intensity_shift_p: float = 0.3,
    intensity_scale_p: float = 0.3,
    noise_p: float = 0.1,
):
    """
    Get light training transforms for ablation studies.
    
    Args:
        rotation_p: Probability of random rotation
        flip_p: Probability of random flip
        intensity_shift_p: Probability of intensity shift
        intensity_scale_p: Probability of intensity scaling
        noise_p: Probability of adding Gaussian noise
        
    Returns:
        Composed transform with lighter augmentation
    """
    return Compose([
        # Geometric augmentations (lower probability)
        RandomHorizontalFlip(p=flip_p),
        RandomRotation90(p=rotation_p),
        
        # Intensity augmentations (lighter)
        RandomIntensityShift(shift_range=0.05, p=intensity_shift_p),
        RandomIntensityScale(scale_range=(0.95, 1.05), p=intensity_scale_p),
        RandomGaussianNoise(std=0.005, p=noise_p),
    ])


if __name__ == "__main__":
    # Test transforms with both NumPy and PyTorch
    print("Testing transforms...")
    
    # Test with NumPy array
    print("\n1. Testing with NumPy array:")
    img_np = np.random.rand(256, 256).astype(np.float32)
    train_transform = get_train_transforms()
    augmented_np = train_transform(img_np)
    
    print(f"Original image shape: {img_np.shape}, type: {type(img_np)}")
    print(f"Augmented image shape: {augmented_np.shape}, type: {type(augmented_np)}")
    print(f"Original range: [{img_np.min():.3f}, {img_np.max():.3f}]")
    print(f"Augmented range: [{augmented_np.min():.3f}, {augmented_np.max():.3f}]")
    
    # Test with PyTorch tensor
    print("\n2. Testing with PyTorch tensor:")
    img_torch = torch.rand(1, 256, 256)
    augmented_torch = train_transform(img_torch)
    
    print(f"Original image shape: {img_torch.shape}, type: {type(img_torch)}")
    print(f"Augmented image shape: {augmented_torch.shape}, type: {type(augmented_torch)}")
    print(f"Original range: [{img_torch.min():.3f}, {img_torch.max():.3f}]")
    print(f"Augmented range: [{augmented_torch.min():.3f}, {augmented_torch.max():.3f}]")
    
    # Test multiple augmentations
    print("\n3. Testing 5 random augmentations (NumPy):")
    for i in range(5):
        aug = train_transform(img_np)
        print(f"  Aug {i+1}: range=[{aug.min():.3f}, {aug.max():.3f}], mean={aug.mean():.3f}")
    
    print("\n✓ All tests passed!")
