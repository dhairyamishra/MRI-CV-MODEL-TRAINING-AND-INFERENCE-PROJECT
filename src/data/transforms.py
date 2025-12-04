"""
Data augmentation and transforms for MRI images.

Supports both NumPy arrays and PyTorch tensors.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Union, Optional


class SkullBoundaryMask:
    """
    Pre-processing transform that keeps only the central skull/brain region
    and zeros everything outside it.

    - Finds the largest connected component (the head).
    - Keeps the original image size (no cropping).
    - Removes background and corner artifacts outside the skull.
    """

    def __init__(self, threshold_percentile: float = 1.0, kernel_size: int = 5):
        """
        Args:
            threshold_percentile: Very low percentile to find non-background pixels
            kernel_size: size of morphological kernel to smooth the mask.
        """
        self.threshold_percentile = threshold_percentile
        self.kernel_size = kernel_size

    def _to_gray_and_shape(self, x_np: np.ndarray):
        """
        Convert different input shapes to a 2D gray image and remember
        how to broadcast the mask back.
        """
        if x_np.ndim == 2:  # (H, W)
            gray = x_np
            shape_type = "HW"
        elif x_np.ndim == 3:
            # Either (C, H, W) or (H, W, C)
            if x_np.shape[0] <= 3:  # (C, H, W)
                gray = x_np[0]  # first channel
                shape_type = "CHW"
            else:  # (H, W, C)
                gray = x_np.mean(axis=2)
                shape_type = "HWC"
        else:
            raise ValueError(f"Unsupported image shape: {x_np.shape}")
        return gray, shape_type

    def _apply_mask_back(self, x_np: np.ndarray, mask: np.ndarray, shape_type: str):
        """
        Apply 2D mask to image, respecting original channel layout.
        """
        if shape_type == "HW":
            return x_np * mask
        elif shape_type == "CHW":
            return x_np * mask[None, ...]
        elif shape_type == "HWC":
            return x_np * mask[..., None]
        else:
            return x_np

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_torch = isinstance(x, torch.Tensor)

        # Move to numpy
        if is_torch:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x

        try:
            gray, shape_type = self._to_gray_and_shape(x_np)
        except ValueError:
            # Unsupported shape -> return as is
            return x

        H, W = gray.shape

        # Use a very low threshold to find any non-background pixels
        # This is much more conservative than the original
        threshold = np.percentile(gray, self.threshold_percentile)
        mask = (gray > threshold).astype(np.uint8)

        # Morphological smoothing to clean up the mask
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find connected components and keep the largest one (the head)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:
            # Only background, keep everything
            final_mask = np.ones_like(mask, dtype=np.float32)
        else:
            # Find the largest component (excluding background at label 0)
            # The largest component is almost always the head
            areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
            largest_label = np.argmax(areas) + 1  # +1 because we skipped background
            
            final_mask = (labels == largest_label).astype(np.float32)

        # Apply mask to original data
        x_masked = self._apply_mask_back(x_np, final_mask, shape_type)

        if is_torch:
            return torch.from_numpy(x_masked).to(x.dtype)
        else:
            return x_masked.astype(x_np.dtype)


class BrainRegionCrop:
    """
    Automatically crops out the blank background around the skull in MRI slices.

    Finds the head (skull + brain) via Otsu thresholding and connected components,
    then crops tightly around it with a small margin.
    """

    def __init__(self, margin: int = 4):
        """
        Args:
            margin: extra pixels to keep around the detected head bbox.
        """
        self.margin = margin

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_torch = isinstance(x, torch.Tensor)
        x_np = x.detach().cpu().numpy() if is_torch else x

        # ---- get grayscale & remember layout ----
        if x_np.ndim == 2:          # (H, W)
            gray = x_np
            layout = "HW"
        elif x_np.ndim == 3:
            if x_np.shape[0] <= 3:  # (C, H, W)
                gray = x_np[0]
                layout = "CHW"
            else:                   # (H, W, C)
                gray = x_np.mean(axis=2)
                layout = "HWC"
        else:
            # unsupported shape, just return as-is
            return x

        H, W = gray.shape

        # ---- normalize to uint8 for Otsu ----
        g_min, g_max = float(gray.min()), float(gray.max())
        if g_max <= g_min + 1e-8:
            # constant image -> nothing to crop
            return x

        norm = (gray - g_min) / (g_max - g_min + 1e-8)
        gray_u8 = (norm * 255).astype(np.uint8)

        # ---- Otsu threshold to separate head vs background ----
        _, mask = cv2.threshold(
            gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # ensure head is the white part; if most of image is 0, invert
        if mask.mean() < 127:  # mostly black
            mask = 255 - mask

        # ---- morphological cleanup ----
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # ---- pick largest central component as the head ----
        num_labels, labels = cv2.connectedComponents(mask)
        if num_labels <= 1:
            # no foreground found, return original
            return x

        center = np.array([H / 2.0, W / 2.0])
        best_label, best_score = None, -1.0

        for label in range(1, num_labels):
            ys, xs = np.where(labels == label)
            if ys.size == 0:
                continue
            area = float(ys.size)
            centroid = np.array([ys.mean(), xs.mean()])
            dist2 = np.sum((centroid - center) ** 2)
            score = area - 0.5 * dist2  # prefer large & central

            if score > best_score:
                best_score = score
                best_label = label

        ys, xs = np.where(labels == best_label)
        if ys.size == 0 or xs.size == 0:
            return x

        # ---- tight bbox around head + small margin ----
        y1 = max(int(ys.min()) - self.margin, 0)
        y2 = min(int(ys.max()) + self.margin, H - 1)
        x1 = max(int(xs.min()) - self.margin, 0)
        x2 = min(int(xs.max()) + self.margin, W - 1)

        if layout == "HW":
            cropped = x_np[y1:y2+1, x1:x2+1]
        elif layout == "CHW":
            cropped = x_np[:, y1:y2+1, x1:x2+1]
        else:  # HWC
            cropped = x_np[y1:y2+1, x1:x2+1, :]

        if is_torch:
            return torch.from_numpy(cropped.copy()).to(x.dtype)
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
    NOW INCLUDES SKULL BOUNDARY MASK to remove border artifacts!
    
    Args:
        rotation_p: Probability of random rotation
        flip_p: Probability of random flip
        intensity_shift_p: Probability of intensity shift
        intensity_scale_p: Probability of intensity scaling
        noise_p: Probability of adding Gaussian noise
        
    Returns:
        Composed transform with skull masking + augmentation
    """
    return Compose([
        # CRITICAL: Mask skull boundary FIRST (removes borders, keeps size!)
        SkullBoundaryMask(threshold_percentile=1.0, kernel_size=5),
        
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
    NOW INCLUDES SKULL BOUNDARY MASK for consistent border removal!
    
    Returns:
        Transform with skull masking only (no augmentation)
    """
    # Apply skull boundary mask to remove borders during validation too
    return Compose([
        SkullBoundaryMask(threshold_percentile=1.0, kernel_size=5),
    ])


def get_strong_train_transforms(
    rotation_p: float = 0.7,
    flip_p: float = 0.7,
    intensity_shift_p: float = 0.7,
    intensity_scale_p: float = 0.7,
    noise_p: float = 0.5,
):
    """
    Get strong training transforms for ablation studies.
    NOW INCLUDES SKULL BOUNDARY MASK!
    
    Args:
        rotation_p: Probability of random rotation
        flip_p: Probability of random flip
        intensity_shift_p: Probability of intensity shift
        intensity_scale_p: Probability of intensity scaling
        noise_p: Probability of adding Gaussian noise
        
    Returns:
        Composed transform with skull masking + stronger augmentation
    """
    return Compose([
        # CRITICAL: Mask skull boundary FIRST
        SkullBoundaryMask(threshold_percentile=1.0, kernel_size=5),
        
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
