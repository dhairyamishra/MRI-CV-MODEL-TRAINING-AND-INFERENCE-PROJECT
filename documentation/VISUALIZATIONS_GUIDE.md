# SliceWise Visualizations - Comprehensive Guide to Generated Outputs

**Version:** 2.0.0 (Multi-Task + Explainability)  
**Date:** December 8, 2025  
**Status:** ✅ Production Ready  

---

## 🎯 Executive Summary

The `visualizations/` directory contains a comprehensive suite of visual outputs generated during SliceWise training, evaluation, and inference processes. These visualizations are critical for:

- **Model Interpretability**: Understanding AI decision-making processes
- **Performance Analysis**: Visual assessment of model accuracy and errors
- **Clinical Validation**: Medical-grade visualization for tumor detection
- **Debugging & Development**: Identifying model strengths and weaknesses
- **Research Documentation**: Visual evidence for publications and reports

**Key Achievements:**
- ✅ **Multi-modal visualizations**: Classification, segmentation, and overlay displays
- ✅ **Explainability tools**: Grad-CAM heatmaps showing model attention
- ✅ **Clinical-grade outputs**: Medical imaging standards and annotations
- ✅ **Automated generation**: Integrated into training and evaluation pipelines
- ✅ **Performance metrics**: Visual distribution analysis and error quantification

---

## 🏗️ Visualization Architecture Overview

### Directory Structure

```
visualizations/
├── multitask_gradcam/          # 🔍 Multi-task Grad-CAM visualizations
├── gradcam/                    # 🎯 Legacy Grad-CAM outputs
├── gradcam_with_cam_reg/       # 📊 Grad-CAM with regularization
├── brain_crop_test/            # 🧠 Brain extraction validation
├── classification_production/  # 📈 Production classification results
│   ├── confusion_matrix.png    # Classification error patterns
│   ├── roc_curve.png          # Discrimination ability visualization
│   ├── pr_curve.png           # Precision-recall trade-offs
│   ├── calibration_plot.png   # Confidence calibration quality
│   └── metrics_summary.json   # Performance metrics data
└── [additional directories created during evaluation]
```

### Generation Pipeline

Visualizations are created at multiple stages:

```
1. 📊 Training Phase    → Loss curves, metric tracking (Weights & Biases)
2. 🔍 Evaluation Phase  → Performance analysis, error visualization
3. 🎯 Inference Phase   → Grad-CAM, uncertainty maps, overlays
4. 📋 Testing Phase     → Validation samples, preprocessing checks
```

### Visualization Types by Function

| Function | Visualization Type | Purpose | Location |
|----------|-------------------|---------|----------|
| **Explainability** | Grad-CAM Heatmaps | Model attention regions | `multitask_gradcam/` |
| **Performance** | Error Overlays | Prediction vs ground truth | `outputs/seg/evaluation/` |
| **Validation** | Brain Crops | Preprocessing verification | `brain_crop_test/` |
| **Analysis** | Metric Distributions | Statistical performance | `outputs/*/evaluation/` |
| **Debugging** | Dataset Examples | Data quality assessment | `assets/grad_cam_examples/` |

---

## 🔍 Grad-CAM Visualizations (`multitask_gradcam/`)

### Purpose: Model Explainability & Interpretability

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations show **which regions of the MRI scan the AI model focuses on** when making predictions. This is critical for:

- **Clinical Trust**: Understanding AI decision-making
- **Error Analysis**: Why models make correct/incorrect predictions
- **Research Validation**: Evidence of learned medical features
- **Regulatory Compliance**: Explainable AI requirements

### Generation Process

**Technical Implementation:**
```python
# Grad-CAM computation in src/eval/grad_cam.py
def generate_gradcam(self, input_tensor, target_class):
    # Forward pass to get activations
    output = self.model(input_tensor)
    
    # Backward pass for gradients
    self.model.zero_grad()
    output[:, target_class].backward()
    
    # Compute Grad-CAM
    weights = self.gradients.mean(dim=[2, 3], keepdim=True)
    cam = (weights * self.activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    
    # Normalize and resize
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear')
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    return cam.squeeze().cpu().numpy()
```

### Output Format

Each Grad-CAM visualization is a **composite image** showing:

```
┌─────────────────────────────────────┐
│                                     │
│        Original MRI Slice           │
│        (Grayscale, 256×256)         │
│                                     │
├─────────────────────────────────────┤
│                                     │
│      Grad-CAM Heatmap Overlay       │
│   (Red=High Attention, Blue=Low)    │
│                                     │
├─────────────────────────────────────┤
│                                     │
│        Prediction Confidence        │
│        Correct: ✓ 95.2%             │
│        Incorrect: ✗ 23.1%           │
└─────────────────────────────────────┘
```

### File Naming Convention

```
gradcam_[correct|incorrect]_[NNN]_sample_[IDX].png

Where:
- correct/incorrect: Model prediction accuracy
- NNN: Sequential sample number (000-999)
- IDX: Original dataset sample index
```

**Examples:**
- `gradcam_correct_000_sample_0.png` - Correct prediction, sample 0
- `gradcam_incorrect_001_sample_3.png` - Incorrect prediction, sample 3

### Clinical Interpretation

#### Correct Predictions
**Expected Patterns:**
- **Tumor regions**: Bright red heatmaps over suspicious areas
- **Anatomical focus**: Attention on brain tissue, avoiding background
- **Confidence correlation**: High confidence → intense, focused heatmaps

#### Incorrect Predictions
**Common Issues:**
- **False negatives**: Model ignores visible tumors (low attention)
- **False positives**: High attention on normal tissue
- **Edge cases**: Subtle tumors or imaging artifacts

### Generation Commands

```bash
# Generate Grad-CAM for multi-task model
python scripts/evaluation/multitask/generate_multitask_gradcam.py \
    --checkpoint checkpoints/multitask_joint/best_model.pth \
    --num-samples 50 \
    --output-dir visualizations/multitask_gradcam

# Generate Grad-CAM for classification model
python scripts/evaluation/multitask/generate_multitask_gradcam.py \
    --checkpoint checkpoints/multitask_cls_head/best_model.pth \
    --task classification \
    --num-samples 25
```

### Quality Assessment

**Evaluation Criteria:**
- **Localization accuracy**: Heatmap overlap with ground truth tumors
- **Sparsity**: Focused attention vs diffuse activation
- **Clinical relevance**: Attention on medically significant regions
- **Consistency**: Similar patterns for similar cases

---

## 🎨 Segmentation Visualizations (`outputs/seg/evaluation/`)

### Purpose: Quantitative & Qualitative Performance Analysis

Segmentation visualizations provide **detailed assessment** of tumor boundary detection, showing:
- **Prediction accuracy**: How well AI detects tumor boundaries
- **Error patterns**: Types of segmentation mistakes
- **Clinical utility**: Visual validation for medical use
- **Debugging insights**: Understanding model limitations

### 4-Panel Visualization Format

Each segmentation sample creates a **4-panel comparison**:

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Input Image   │  Ground Truth   │  Prediction     │   Error Overlay  │
│   (Grayscale)   │   (Red overlay) │ (Green overlay) │ (TP/FP/FN)       │
│                 │                 │                 │                  │
│   Original      │   Expert        │   AI Model      │   Error Analysis │
│   MRI slice     │   annotation    │   prediction    │   (Color coded)  │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### Color Coding Scheme

#### Ground Truth Panel (Panel 2)
- **Background**: Original MRI slice (grayscale)
- **Tumor regions**: Red overlay (semi-transparent)

#### Prediction Panel (Panel 3)
- **Background**: Original MRI slice (grayscale)
- **Predicted tumor**: Green overlay (semi-transparent)
- **Metrics display**: Dice score shown in title

#### Error Overlay Panel (Panel 4)
- **True Positive (TP)**: Green - Correctly identified tumor
- **False Positive (FP)**: Red - AI detected tumor where none exists
- **False Negative (FN)**: Blue - AI missed actual tumor
- **True Negative (TN)**: Background - Correctly identified normal tissue

### Generation Process

**Created during segmentation evaluation:**
```python
# From src/eval/eval_seg2d.py
def create_overlay(image, pred_mask, gt_mask):
    """Create TP/FP/FN overlay visualization."""
    # Convert masks to binary
    pred = (pred_mask > threshold).astype(np.uint8)
    gt = gt_mask.astype(np.uint8)
    
    # Calculate error regions
    tp = pred & gt          # True positives
    fp = pred & ~gt         # False positives  
    fn = ~pred & gt         # False negatives
    
    # Create RGB overlay
    overlay = np.zeros((*image.shape, 3), dtype=np.uint8)
    overlay[tp == 1] = [0, 255, 0]    # Green for TP
    overlay[fp == 1] = [255, 0, 0]    # Red for FP
    overlay[fn == 1] = [0, 0, 255]    # Blue for FN
    
    return overlay
```

### Performance Metrics Visualization

#### Metrics Distribution Plots
**File**: `metrics_distribution.png`

Shows statistical distribution of segmentation performance:
- **Dice coefficient**: Overlap accuracy (0-1 scale)
- **IoU (Jaccard)**: Intersection over union
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/true positive rate
- **F1 Score**: Harmonic mean of precision/recall
- **Specificity**: True negative rate

**Interpretation:**
- **Normal distribution**: Consistent performance
- **Skewed distributions**: Systematic biases
- **Outliers**: Problematic cases needing investigation

### File Organization

```
outputs/seg/evaluation/
├── evaluation_results.json     # 📊 Detailed metrics
├── metrics_distribution.png    # 📈 Performance histograms
├── visualizations/             # 🖼️ Individual sample overlays
│   ├── sample_0000.png        # First visualization
│   ├── sample_0001.png        # Second visualization
│   └── ...
└── [additional evaluation outputs]
```

### Clinical Assessment Criteria

#### Excellent Performance (Dice > 0.8)
- **Predictions**: Green overlays closely match red ground truth
- **Errors**: Minimal red/blue regions
- **Clinical utility**: Reliable for treatment planning

#### Good Performance (Dice 0.7-0.8)
- **Predictions**: Reasonable boundary approximation
- **Errors**: Some over/under-segmentation
- **Clinical utility**: Useful with expert review

#### Needs Improvement (Dice < 0.7)
- **Predictions**: Significant boundary errors
- **Errors**: Large red/blue regions
- **Clinical utility**: Requires substantial expert correction

---

## 🧠 Brain Crop Visualizations (`brain_crop_test/`)

### Purpose: Preprocessing Validation

Brain crop visualizations validate the **brain extraction algorithm** that removes skull and background from raw MRI scans.

**Critical for:**
- **Preprocessing verification**: Ensuring clean brain regions
- **Quality control**: Detecting extraction failures
- **Debugging**: Identifying preprocessing edge cases
- **Clinical validity**: Confirming anatomically correct extractions

### Visualization Format

Shows before/after comparison of brain extraction:

```
┌─────────────────┬─────────────────┐
│   Raw MRI Scan  │  Extracted Brain │
│   (With skull)  │    (Clean)       │
│                 │                  │
│   Original      │   Preprocessed   │
│   FLAIR/T1/T2   │   Brain tissue   │
└─────────────────┴─────────────────┴
```

### Generation Context

Created during **BraTS data preprocessing** (`src/data/preprocess_brats_2d.py`):

```python
def extract_brain_region(image, threshold=0.1):
    """Extract brain tissue from skull/background."""
    # Intensity-based brain extraction
    brain_mask = image > threshold
    
    # Morphological operations to clean mask
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=1000)
    brain_mask = morphology.binary_closing(brain_mask, footprint=np.ones((5,5)))
    
    # Apply mask to original image
    extracted = image * brain_mask
    
    return extracted, brain_mask
```

---

## 📊 Performance Analysis Visualizations

### Training Metrics (Weights & Biases)

**Purpose**: Real-time monitoring of training progress

**Visualizations Include:**
- **Loss curves**: Training/validation loss over time
- **Accuracy metrics**: Classification accuracy progression
- **Segmentation metrics**: Dice/IoU improvement
- **Learning rate**: Scheduler adjustments
- **Gradient norms**: Training stability indicators

### Evaluation Summary Plots

**Purpose**: Comprehensive performance assessment

**Generated Files:**
- `confusion_matrix.png` - Classification error patterns
- `roc_curve.png` - Classification discrimination ability
- `pr_curve.png` - Precision-recall trade-offs
- `calibration_plot.png` - Confidence calibration quality

### Uncertainty Visualizations

**Purpose**: Model confidence and reliability assessment

**Types:**
- **Uncertainty heatmaps**: Regions of prediction uncertainty
- **Confidence distributions**: Probability distribution analysis
- **Monte Carlo dropout**: Multiple prediction variability

---

## 📋 Dataset Examples (`assets/grad_cam_examples/`)

### Purpose: Training Data Quality Assessment

Contains **example visualizations** from the training dataset to:
- **Verify data quality**: Ensure proper preprocessing
- **Document dataset**: Show representative samples
- **Debug issues**: Compare training data with evaluation failures
- **Research reference**: Consistent examples for publications

### Current Examples

**File**: `gradcam_correct_000_sample_0.png` through `gradcam_incorrect_002_sample_5.png`

Shows **16 representative samples**:
- **13 correct predictions**: Demonstrating successful learning
- **3 incorrect predictions**: Highlighting current limitations

**Clinical Relevance:**
- **Correct cases**: Expected attention patterns for various tumor types
- **Incorrect cases**: Common failure modes and edge cases
- **Variability**: Different tumor sizes, locations, and imaging qualities

---

## 🚀 Generation Workflow Integration

### Automated Pipeline Integration

Visualizations are **automatically generated** during standard workflows:

```bash
# Full pipeline (includes all visualizations)
python scripts/run_full_pipeline.py --mode full --training-mode baseline

# Evaluation only (segmentation + Grad-CAM)
python scripts/evaluation/multitask/evaluate_multitask.py
python scripts/evaluation/multitask/generate_multitask_gradcam.py --num-samples 50

# Segmentation evaluation (overlays + metrics)
python src/eval/eval_seg2d.py --checkpoint checkpoints/best_model.pth --max-visualizations 20
```

### Manual Generation Options

```bash
# Custom Grad-CAM generation
python scripts/evaluation/multitask/generate_multitask_gradcam.py \
    --checkpoint custom_model.pth \
    --num-samples 100 \
    --output-dir custom_visualizations

# Segmentation evaluation with custom settings
python src/eval/eval_seg2d.py \
    --checkpoint model.pth \
    --threshold 0.3 \
    --max-visualizations 50 \
    --output-dir custom_eval
```

### Batch Processing

**For large-scale evaluation:**
- **Parallel generation**: Multiple samples processed simultaneously
- **Memory management**: Efficient GPU utilization
- **Progress tracking**: Real-time completion status
- **Error handling**: Robust failure recovery

---

## 🔬 Technical Specifications

### Image Formats & Resolutions

| Visualization Type | Resolution | Format | Color Depth |
|-------------------|------------|--------|-------------|
| **Grad-CAM** | 256×256 | PNG | 24-bit RGB |
| **Segmentation** | 256×256 | PNG | 24-bit RGB |
| **Overlays** | 256×256 | PNG | 24-bit RGB |
| **Metrics plots** | Variable | PNG | 24-bit RGB |
| **Brain crops** | Original | PNG | 8-bit Grayscale |

### File Size Estimates

| Visualization Type | Avg File Size | Samples | Total Size |
|-------------------|----------------|---------|------------|
| **Grad-CAM** | 50-200 KB | 50 | 2.5-10 MB |
| **Segmentation** | 100-300 KB | 20 | 2-6 MB |
| **Metrics plots** | 200-500 KB | 5 | 1-2.5 MB |
| **Examples** | 50-150 KB | 16 | ~1.5 MB |

### Generation Performance

| Operation | Time per Sample | GPU Memory | Notes |
|-----------|-----------------|------------|-------|
| **Grad-CAM** | 0.5-1.0 sec | 1-2 GB | Batch processing recommended |
| **Segmentation eval** | 0.2-0.5 sec | 2-4 GB | Includes overlay generation |
| **Metrics plots** | 5-10 sec | Minimal | Matplotlib rendering |
| **Brain crops** | 0.1-0.3 sec | Minimal | CPU preprocessing |

---

## 📊 Clinical Interpretation Guidelines

### For Radiologists & Clinicians

#### Grad-CAM Assessment
1. **Attention patterns**: Should focus on suspicious anatomical regions
2. **Confidence correlation**: High confidence should show focused, intense heatmaps
3. **Clinical relevance**: Heatmaps should align with radiological expertise
4. **Error identification**: Low attention on visible tumors indicates model limitations

#### Segmentation Evaluation
1. **Boundary accuracy**: Green overlays should closely follow red ground truth
2. **Error patterns**: 
   - Red (FP): Over-segmentation of normal tissue
   - Blue (FN): Under-segmentation missing tumor regions
3. **Clinical impact**: Assess whether errors affect treatment decisions
4. **Consistency**: Performance should be reliable across similar cases

### Quality Control Checklist

- [ ] **Grad-CAM localization**: Heatmaps overlap with tumor regions
- [ ] **Segmentation boundaries**: Accurate tumor delineation
- [ ] **Error quantification**: Acceptable FP/FN rates for clinical use
- [ ] **Consistency**: Similar cases produce similar results
- [ ] **Robustness**: Performance maintained across imaging conditions

---

## 🔧 Customization & Extension

### Adding New Visualization Types

**Template for new visualizations:**
```python
def create_custom_visualization(image, prediction, ground_truth):
    """Create custom visualization for specific analysis."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input')
    
    # Custom analysis
    custom_analysis = perform_custom_analysis(prediction, ground_truth)
    axes[1].imshow(custom_analysis)
    axes[1].set_title('Custom Analysis')
    
    # Results overlay
    overlay = create_overlay_visualization(image, custom_analysis)
    axes[2].imshow(overlay)
    axes[2].set_title('Results')
    
    # Save
    plt.savefig('custom_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
```

### Integration Points

**Where to add custom visualizations:**
- **Training**: `src/training/train_*.py` (progress monitoring)
- **Evaluation**: `src/eval/eval_*.py` (performance analysis)
- **Inference**: `src/inference/*.py` (prediction explanations)
- **Scripts**: `scripts/evaluation/*/generate_*.py` (batch processing)

---

## 📚 Related Documentation

- **[SCRIPTS_ARCHITECTURE_AND_USAGE.md](documentation/SCRIPTS_ARCHITECTURE_AND_USAGE.md)** - How visualizations are generated
- **[APP_ARCHITECTURE_AND_FUNCTIONALITY.md](documentation/APP_ARCHITECTURE_AND_FUNCTIONALITY.md)** - Web interface visualization
- **[SRC_ARCHITECTURE_AND_IMPLEMENTATION.md](documentation/SRC_ARCHITECTURE_AND_IMPLEMENTATION.md)** - Core visualization code

---

*Built with ❤️ for transparent, interpretable medical AI.*
