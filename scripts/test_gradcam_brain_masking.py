"""
Test script to verify Grad-CAM brain masking eliminates background artifacts.

This script:
1. Loads a BraTS MRI image with visible skull boundary
2. Runs Grad-CAM with and without brain masking
3. Generates side-by-side comparison visualizations
4. Saves results for inspection

Usage:
    python scripts/test_gradcam_brain_masking.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.multi_task_predictor import create_multi_task_predictor


def load_test_images() -> dict:
    """Load test MRI images from dataset examples (raw PNG exports)."""
    images = {}
    
    # Load from dataset_examples directory (raw PNG exports)
    examples_dir = project_root / "data" / "dataset_examples"
    
    # Try to load Kaggle image
    kaggle_examples = examples_dir / "kaggle" / "yes_tumor"
    if kaggle_examples.exists():
        sample_dirs = sorted(list(kaggle_examples.glob("sample_*")))
        for sample_dir in sample_dirs[:1]:  # Load first sample
            image_path = sample_dir / "image.png"
            if image_path.exists():
                try:
                    img = Image.open(image_path).convert('L')
                    img_array = np.array(img).astype(np.float32)
                    images['kaggle'] = {
                        'image': img_array,
                        'path': image_path,
                        'dataset': 'Kaggle (Raw Export)'
                    }
                    print(f"[INFO] Loaded Kaggle image from: {image_path}")
                    break
                except Exception as e:
                    print(f"[WARNING] Failed to load {image_path}: {e}")
    
    # Try to load BraTS image
    brats_examples = examples_dir / "brats" / "yes_tumor"
    if brats_examples.exists():
        sample_dirs = sorted(list(brats_examples.glob("sample_*")))
        for sample_dir in sample_dirs[:1]:  # Load first sample
            image_path = sample_dir / "image.png"
            if image_path.exists():
                try:
                    img = Image.open(image_path).convert('L')
                    img_array = np.array(img).astype(np.float32)
                    images['brats'] = {
                        'image': img_array,
                        'path': image_path,
                        'dataset': 'BraTS (Raw Export)'
                    }
                    print(f"[INFO] Loaded BraTS image from: {image_path}")
                    break
                except Exception as e:
                    print(f"[WARNING] Failed to load {image_path}: {e}")
    
    if not images:
        print("[WARNING] No dataset examples found, creating synthetic images")
        # Create synthetic Kaggle-like image
        kaggle_synth = np.zeros((256, 256), dtype=np.float32)
        y, x = np.ogrid[:256, :256]
        mask = (x - 128)**2 + (y - 128)**2 <= 100**2
        kaggle_synth[mask] = np.random.randn(mask.sum()) * 50 + 128
        images['kaggle'] = {'image': kaggle_synth, 'path': 'synthetic', 'dataset': 'Kaggle (synthetic)'}
        
        # Create synthetic BraTS-like image
        brats_synth = kaggle_synth.copy()
        brats_synth[100:120, 150:170] = 200
        images['brats'] = {'image': brats_synth, 'path': 'synthetic', 'dataset': 'BraTS (synthetic)'}
    
    return images


def create_gradcam_overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create Grad-CAM overlay visualization."""
    # Normalize image to [0, 255]
    img_normalized = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255)
    img_uint8 = np.uint8(img_normalized)
    
    # Convert grayscale to RGB
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    
    # Resize CAM to match image size
    h, w = image.shape
    cam_resized = cv2.resize(cam, (w, h))
    
    # Convert CAM to heatmap
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend images
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def process_single_image(predictor, image_data: dict, output_dir: Path):
    """Process a single image with and without brain masking."""
    image = image_data['image']
    dataset = image_data['dataset']
    
    print(f"\n{'='*80}")
    print(f"Processing {dataset} Image")
    print(f"{'='*80}")
    print(f"      Path: {image_data['path']}")
    print(f"      Shape: {image.shape}")
    print(f"      Range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Run Grad-CAM WITHOUT brain masking
    print(f"\n  [1/2] Running Grad-CAM WITHOUT brain masking...")
    result_no_mask = predictor.predict_with_gradcam(image, use_brain_mask=False)
    cam_no_mask = result_no_mask['gradcam']['heatmap']
    pred_no_mask = result_no_mask['classification']['predicted_label']
    conf_no_mask = result_no_mask['classification']['confidence']
    
    print(f"        Prediction: {pred_no_mask} ({conf_no_mask*100:.1f}%)")
    print(f"        CAM shape: {cam_no_mask.shape}")
    
    # Run Grad-CAM WITH brain masking
    print(f"\n  [2/2] Running Grad-CAM WITH brain masking...")
    result_with_mask = predictor.predict_with_gradcam(image, use_brain_mask=True)
    cam_with_mask = result_with_mask['gradcam']['heatmap']
    pred_with_mask = result_with_mask['classification']['predicted_label']
    conf_with_mask = result_with_mask['classification']['confidence']
    brain_masked = result_with_mask['gradcam']['brain_masked']
    
    print(f"        Prediction: {pred_with_mask} ({conf_with_mask*100:.1f}%)")
    print(f"        Brain mask applied: {brain_masked}")
    
    # Normalize image for display
    image_display = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Create overlays
    overlay_no_mask = create_gradcam_overlay(image_display, cam_no_mask)
    overlay_with_mask = create_gradcam_overlay(image_display, cam_with_mask)
    
    return {
        'image': image_display,
        'cam_no_mask': cam_no_mask,
        'cam_with_mask': cam_with_mask,
        'overlay_no_mask': overlay_no_mask,
        'overlay_with_mask': overlay_with_mask,
        'pred_no_mask': pred_no_mask,
        'conf_no_mask': conf_no_mask,
        'pred_with_mask': pred_with_mask,
        'conf_with_mask': conf_with_mask,
        'brain_masked': brain_masked,
        'dataset': dataset
    }


def main():
    """Main test function."""
    print("=" * 80)
    print("Testing Grad-CAM Brain Masking on Kaggle and BraTS Images")
    print("=" * 80)
    
    # Configuration
    checkpoint_path = project_root / "checkpoints" / "multitask_joint" / "best_model.pth"
    output_dir = project_root / "results" / "gradcam_masking_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("[INFO] Please train the multi-task model first or specify correct checkpoint path")
        return
    
    # Load test images
    print("\n[1/4] Loading test images from both datasets...")
    test_images = load_test_images()
    print(f"      Loaded {len(test_images)} images: {list(test_images.keys())}")
    
    # Create predictor
    print("\n[2/4] Loading multi-task model...")
    predictor = create_multi_task_predictor(
        checkpoint_path=str(checkpoint_path),
        device='cuda'
    )
    
    # Process each image
    print("\n[3/4] Processing images...")
    results = {}
    for key, image_data in test_images.items():
        results[key] = process_single_image(predictor, image_data, output_dir)
    
    # Create comparison visualizations
    print(f"\n[4/4] Creating comparison visualizations...")
    
    # Create separate figures for each dataset
    for key, result in results.items():
        dataset = result['dataset']
        
        # Create comparison figure (2 rows x 3 cols)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Without brain masking
        axes[0, 0].imshow(result['image'], cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(result['cam_no_mask'], cmap='jet')
        axes[0, 1].set_title('Grad-CAM Heatmap\n(WITHOUT Brain Masking)', fontsize=12, fontweight='bold', color='red')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(result['overlay_no_mask'])
        axes[0, 2].set_title(f'Overlay\nPred: {result["pred_no_mask"]} ({result["conf_no_mask"]*100:.1f}%)', 
                             fontsize=12, fontweight='bold', color='red')
        axes[0, 2].axis('off')
        
        # Row 2: With brain masking
        axes[1, 0].imshow(result['image'], cmap='gray')
        axes[1, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(result['cam_with_mask'], cmap='jet')
        axes[1, 1].set_title('Grad-CAM Heatmap\n(WITH Brain Masking)', fontsize=12, fontweight='bold', color='green')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(result['overlay_with_mask'])
        axes[1, 2].set_title(f'Overlay\nPred: {result["pred_with_mask"]} ({result["conf_with_mask"]*100:.1f}%)', 
                             fontsize=12, fontweight='bold', color='green')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Grad-CAM Brain Masking Comparison - {dataset}\n(Top: Without Masking, Bottom: With Masking)', 
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"gradcam_masking_comparison_{key}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"      Saved {dataset} comparison to: {output_path}")
        
        # Save individual images
        cv2.imwrite(str(output_dir / f"{key}_cam_no_mask.png"), np.uint8(result['cam_no_mask'] * 255))
        cv2.imwrite(str(output_dir / f"{key}_cam_with_mask.png"), np.uint8(result['cam_with_mask'] * 255))
        cv2.imwrite(str(output_dir / f"{key}_overlay_no_mask.png"), cv2.cvtColor(result['overlay_no_mask'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / f"{key}_overlay_with_mask.png"), cv2.cvtColor(result['overlay_with_mask'], cv2.COLOR_RGB2BGR))
        
        plt.close(fig)
    
    print("\n" + "=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nProcessed {len(results)} images:")
    for key, result in results.items():
        print(f"  - {result['dataset']}: Brain mask {'✅ applied' if result['brain_masked'] else '❌ not applied'}")
    print("\nKey Observations:")
    print("  - WITHOUT masking: Background may show red/hot activations (artifacts)")
    print("  - WITH masking: Background should be black (clean, brain-focused)")
    print("\nOpen the comparison images to verify the fix!")
    
    # Show last plot
    plt.show()


if __name__ == "__main__":
    main()
