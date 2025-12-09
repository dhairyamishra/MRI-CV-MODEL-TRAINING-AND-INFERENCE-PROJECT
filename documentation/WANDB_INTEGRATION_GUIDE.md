# Weights & Biases (W&B) Integration - Experiment Tracking & Visualization

**Version:** 2.0.0 (Multi-Task + Production Ready)  
**Date:** December 8, 2025  
**Status:** ✅ Production Ready  

---

## 🎯 Executive Summary

Weights & Biases (W&B) is the experiment tracking and visualization platform integrated into SliceWise for comprehensive machine learning experiment management. It provides real-time monitoring, hyperparameter tracking, model performance visualization, and collaboration tools for the brain tumor detection pipeline.

**Key Achievements:**
- ✅ **Real-time experiment tracking** during training and evaluation
- ✅ **Comprehensive metric logging** (loss curves, accuracy, AUC, Dice scores)
- ✅ **Model artifact management** with automatic versioning
- ✅ **Interactive dashboards** for experiment comparison and analysis
- ✅ **Team collaboration** with shared projects and reports
- ✅ **Integration with CI/CD** for automated model validation

---

## 🏗️ What is Weights & Biases?

### Overview

**Weights & Biases (W&B)** is a machine learning experiment tracking platform that provides:

- **Experiment Logging**: Track hyperparameters, metrics, and artifacts
- **Real-time Monitoring**: Live dashboards during training
- **Model Comparison**: Side-by-side analysis of different runs
- **Collaboration**: Team sharing and commenting on experiments
- **Automation**: Integration with CI/CD pipelines

### Why W&B for Medical AI?

**Medical AI requires special attention to:**
- **Reproducibility**: All experiments must be fully documented
- **Regulatory Compliance**: Audit trails for model development
- **Performance Validation**: Rigorous evaluation of clinical metrics
- **Explainability**: Understanding model behavior and limitations
- **Collaboration**: Multi-disciplinary teams (clinicians, engineers, researchers)

---

## 📁 W&B Directory Structure

The `wandb/` directory contains experiment runs and cached data:

```
wandb/
├── run-20251204_003148-4fymgc3m/    # Individual experiment run
├── run-20251204_003736-9onpjqq4/    # Format: date_time-run_id
├── run-20251204_030352-5ua01jxu/
├── run-20251204_030528-ywsi0563/
├── run-20251204_031021-u68dlptc/
├── run-20251204_031114-u60230hk/
├── run-20251204_031248-785xhpkb/
├── run-20251204_031444-82eclzsg/
├── run-20251204_031558-u8sfgf0v/    # Latest production run
└── [additional run directories...]
```

**Run Directory Contents:**
- `config.yaml` - Hyperparameters and configuration
- `wandb-metadata.json` - Run metadata and system info
- `output.log` - Training logs and console output
- `files/` - Model checkpoints and artifacts
- `media/` - Charts, plots, and visualizations

---

## 🚀 How W&B is Used in SliceWise

### Integration Points

W&B is integrated into **multiple components** of the SliceWise pipeline:

#### 1. Training Scripts (`src/training/`)
- **Real-time metric logging** during training epochs
- **Hyperparameter tracking** from configuration files
- **Model checkpoint uploading** with automatic versioning
- **System resource monitoring** (GPU usage, memory)

#### 2. Evaluation Scripts (`src/eval/`)
- **Performance metric visualization** (ROC curves, confusion matrices)
- **Grad-CAM explainability** uploads for model interpretation
- **Statistical analysis** of model predictions

#### 3. Configuration System (`configs/`)
- **Hyperparameter management** through YAML configurations
- **Experiment metadata** tracking and organization
- **Run comparison** across different training modes

### Training Integration

**From `src/training/train_cls.py`:**

```python
# W&B initialization
def _init_wandb(self):
    """Initialize Weights & Biases logging."""
    wandb.init(
        project=self.config['logging']['wandb_project'],
        entity=self.config['logging'].get('wandb_entity'),
        config=self.config,
        name=f"cls_{self.config['model']['name']}_{self.config['seed']}"
    )

# Real-time logging during training
if self.use_wandb:
    wandb.log({
        'epoch': epoch,
        'train/loss': train_loss,
        'train/accuracy': train_acc,
        'val/loss': val_loss,
        **{f'val/{k}': v for k, v in val_metrics.items()}
    })

# Model watching for gradient/flow tracking
if self.use_wandb:
    wandb.watch(self.model, log='all', log_freq=100)
```

### Configuration Setup

**W&B is configured through the hierarchical config system:**

```yaml
# In configs/modes/baseline.yaml
wandb:
  enabled: true
  project: "slicewise-multitask-baseline"

# In configs/modes/quick_test.yaml  
wandb:
  enabled: false  # Disabled for quick testing

# In configs/modes/production.yaml
wandb:
  enabled: true
  project: "slicewise-production"
```

**Generated final config:**
```yaml
wandb:
  enabled: true
  project: slicewise-multitask-baseline
```

---

## 🎯 W&B Dashboard Features

### Real-Time Training Monitoring

#### 1. **Metrics Panel**
- **Training Loss**: Real-time loss curve with smoothing
- **Validation Metrics**: Accuracy, AUC, F1-score, precision, recall
- **Learning Rate**: Scheduler adjustments over time
- **Gradient Norms**: Training stability monitoring

#### 2. **System Metrics**
- **GPU Utilization**: Memory usage and compute utilization
- **Training Speed**: Samples/second, epoch time
- **Memory Usage**: RAM and GPU memory consumption

#### 3. **Hyperparameter Tracking**
- **Model Architecture**: Network depth, filters, learning rate
- **Training Settings**: Batch size, epochs, optimizer parameters
- **Data Configuration**: Augmentation settings, dataset splits

### Experiment Comparison

#### Run Comparison Table
- **Side-by-side metrics** for different experiments
- **Performance ranking** by validation accuracy or Dice score
- **Hyperparameter correlation** analysis
- **Statistical significance** testing

#### Parallel Coordinates Plot
- **Multi-dimensional analysis** of hyperparameter effects
- **Performance optimization** guidance
- **Sensitivity analysis** for different parameters

### Model Artifacts

#### Checkpoint Management
- **Automatic versioning** of model checkpoints
- **Download links** for best-performing models
- **Metadata tracking** (training time, dataset used)
- **Integration with deployment** pipelines

#### Visualization Storage
- **Grad-CAM heatmaps** for model explainability
- **ROC curves** and confusion matrices
- **Training curves** and performance plots
- **Custom charts** from evaluation scripts

---

## 🛠️ Setting Up W&B

### Prerequisites

1. **Install W&B**:
   ```bash
   pip install wandb
   ```

2. **Create W&B Account**:
   - Sign up at [wandb.ai](https://wandb.ai)
   - Create a team (optional for collaboration)

3. **Get API Key**:
   ```bash
   wandb login
   # Or set environment variable
   export WANDB_API_KEY=your_api_key_here
   ```

### Configuration

#### Basic Setup
```bash
# Login (one-time setup)
wandb login

# Check login status
wandb status
```

#### Team/Organization Setup
```yaml
# In config file
wandb:
  enabled: true
  project: "slicewise-medical-ai"
  entity: "your-team-name"  # For team projects
```

### Environment Variables

```bash
# Set API key
export WANDB_API_KEY=your_api_key

# Set default entity (team)
export WANDB_ENTITY=your-team-name

# Set base URL (for on-premises)
export WANDB_BASE_URL=https://your-wandb-server.com
```

---

## 📊 Using W&B During Training

### Starting Training with W&B

```bash
# Training automatically initializes W&B
python scripts/training/multitask/train_multitask_joint.py \
    --config configs/final/stage3_baseline.yaml

# Or use the full pipeline
python scripts/run_full_pipeline.py --mode full --training-mode baseline
```

### Real-Time Monitoring

#### During Training
1. **Open the URL** printed in the console (e.g., `https://wandb.ai/...`)
2. **Watch live metrics** update in real-time
3. **Monitor system resources** (GPU memory, training speed)
4. **View logged images** (Grad-CAM visualizations)

#### Dashboard Overview
- **Overview Tab**: Summary statistics and key metrics
- **Charts Tab**: Detailed plots and comparisons
- **Logs Tab**: Training console output
- **Artifacts Tab**: Model checkpoints and files
- **Files Tab**: Configuration files and metadata

### Experiment Organization

#### Project Structure
```
your-team/
├── slicewise-quick-test/     # Quick validation runs
├── slicewise-baseline/       # Standard experiments  
├── slicewise-production/     # Production training runs
└── slicewise-research/       # Research experiments
```

#### Run Naming Convention
- **Automatic**: `cls_efficientnet_42` (model_seed)
- **Custom**: Set `name` in config for meaningful identifiers
- **Tags**: Add tags for experiment categorization

---

## 📈 Analyzing Results

### Performance Analysis

#### Key Metrics to Monitor

**Classification Metrics:**
- **Accuracy**: Overall correctness (target: >90%)
- **ROC-AUC**: Discrimination ability (target: >0.90)
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Balanced metric for imbalanced data

**Segmentation Metrics:**
- **Dice Coefficient**: Overlap accuracy (target: >0.75)
- **IoU (Jaccard)**: Intersection over union
- **Specificity**: True negative rate (important for medical imaging)

#### Training Stability

**Monitor for:**
- **Loss convergence**: Smooth decrease without oscillations
- **Validation improvement**: Consistent gains with training
- **Overfitting detection**: Training/validation divergence
- **Learning rate effectiveness**: Appropriate decay schedule

### Experiment Comparison

#### Best Practices

1. **Control Variables**: Change one parameter at a time
2. **Reproducible Seeds**: Use fixed seeds for fair comparison
3. **Consistent Evaluation**: Same test set and metrics
4. **Document Changes**: Log what was modified and why

#### Common Comparisons

```python
# Compare different architectures
experiments = [
    "efficientnet_b0_baseline",
    "convnext_tiny_baseline", 
    "resnet50_baseline"
]

# Compare training strategies
experiments = [
    "dice_bce_loss",
    "focal_loss",
    "combined_loss"
]

# Compare augmentation levels
experiments = [
    "minimal_aug",
    "moderate_aug", 
    "aggressive_aug"
]
```

### Model Selection

#### Criteria for Best Model

1. **Validation Performance**: Highest validation metrics
2. **Training Stability**: Consistent convergence without overfitting
3. **Inference Speed**: Fast enough for clinical deployment
4. **Explainability**: Clear Grad-CAM visualizations
5. **Robustness**: Good performance on edge cases

---

## 🤝 Team Collaboration

### Sharing Experiments

#### Public Projects
- **Shareable links** for external collaborators
- **Public reports** for publication and documentation
- **Embeddable charts** for presentations and papers

#### Team Features
- **Role-based access**: Admin, member, viewer permissions
- **Private teams**: Secure collaboration for sensitive medical data
- **Audit logs**: Track who accessed what and when

### Reporting and Documentation

#### Creating Reports
1. **Select key runs** to include in report
2. **Add markdown sections** explaining methodology
3. **Embed interactive charts** for dynamic exploration
4. **Share with stakeholders** for review and feedback

#### Publication Support
- **Export high-quality plots** for papers
- **Generate summary statistics** for methods sections
- **Document hyperparameters** for reproducibility
- **Create comparison tables** for ablation studies

---

## 🚀 Advanced W&B Features

### Sweeps (Hyperparameter Optimization)

```yaml
# sweep.yaml
program: scripts/training/multitask/train_multitask_joint.py
method: bayes
metric:
  name: val_dice
  goal: maximize
parameters:
  learning_rate:
    min: 1e-5
    max: 1e-3
  batch_size:
    values: [8, 16, 32]
  model_depth:
    values: [3, 4, 5]
```

```bash
# Run hyperparameter sweep
wandb sweep sweep.yaml
wandb agent your-sweep-id
```

### Custom Metrics and Visualizations

#### Logging Custom Metrics
```python
# In training script
wandb.log({
    'custom/medical_accuracy': medical_accuracy_score,
    'custom/clinical_f1': clinical_f1_score,
    'custom/tumor_detection_rate': detection_rate
})
```

#### Logging Images and Charts
```python
# Log Grad-CAM visualization
wandb.log({"gradcam": wandb.Image(gradcam_overlay)})

# Log confusion matrix
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    y_true=labels,
    preds=predictions,
    class_names=['No Tumor', 'Tumor']
)})
```

### Integration with CI/CD

#### Automated Testing
```yaml
# .github/workflows/train.yml
- name: Run Training with W&B
  run: |
    python scripts/run_full_pipeline.py --mode full --training-mode quick
  env:
    WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
    WANDB_PROJECT: slicewise-ci
```

#### Model Validation
- **Automated performance checks** against baseline metrics
- **Regression detection** for model performance drops
- **Artifact promotion** to production when tests pass

---

## 🔒 Medical AI Compliance

### Regulatory Considerations

#### FDA Requirements
- **Audit Trail**: Complete record of model development
- **Reproducibility**: Exact hyperparameters and random seeds
- **Performance Validation**: Rigorous testing on held-out data
- **Change Tracking**: Documentation of any model modifications

#### HIPAA Compliance
- **Data Privacy**: No patient data stored in W&B
- **Access Controls**: Team-based permissions for sensitive projects
- **Audit Logging**: Track who accessed model artifacts

### Best Practices

#### Data Handling
- **Anonymized data only**: No patient identifiers in logs
- **Aggregated metrics**: Individual predictions not stored
- **Secure storage**: Use private projects for sensitive work

#### Model Governance
- **Version control**: Tag important model versions
- **Change documentation**: Explain why models were modified
- **Performance monitoring**: Track model drift in production

---

## 🐛 Troubleshooting

### Common Issues

#### "wandb: Network error"
```bash
# Check internet connection
curl -s https://api.wandb.ai

# Check API key
wandb login --relogin

# Use offline mode if needed
export WANDB_MODE=offline
```

#### "CUDA out of memory during logging"
```python
# Reduce logging frequency
wandb.watch(model, log='all', log_freq=500)  # Less frequent

# Log only gradients, not parameters
wandb.watch(model, log='gradients', log_freq=100)
```

#### "Run not appearing in dashboard"
```bash
# Check project name
wandb status

# Force sync
wandb sync

# Check entity/team settings
export WANDB_ENTITY=your-team-name
```

#### "Permission denied"
```bash
# Check API key permissions
wandb login --relogin

# Verify team membership
wandb status

# Contact team admin for access
```

---

## 📊 Performance Impact

### Resource Usage

| Feature | CPU Impact | GPU Impact | Network Impact |
|---------|------------|------------|----------------|
| **Basic logging** | Minimal | Minimal | Low |
| **Model watching** | Low | Low | Medium |
| **Image logging** | Low | Minimal | High |
| **Large artifacts** | Minimal | Minimal | Very High |

### Training Overhead

- **Basic metrics**: <1% training time overhead
- **Model watching**: 2-5% training time overhead
- **Image logging**: 5-10% training time overhead (first epoch only)

### Storage Requirements

| Data Type | Size per Epoch | 50 Epoch Run |
|-----------|----------------|--------------|
| **Metrics** | ~1KB | ~50KB |
| **Checkpoints** | 100-500MB | 5-25GB |
| **Images** | 50-200KB | 2.5-10MB |
| **Logs** | ~10KB | ~500KB |

---

## 📚 Related Documentation

- **[README.md](../README.md)** - Project overview and quick start
- **[CONFIG_SYSTEM_ARCHITECTURE.md](CONFIG_SYSTEM_ARCHITECTURE.md)** - Configuration management
- **[SCRIPTS_ARCHITECTURE_AND_USAGE.md](SCRIPTS_ARCHITECTURE_AND_USAGE.md)** - Training scripts
- **[VISUALIZATIONS_GUIDE.md](VISUALIZATIONS_GUIDE.md)** - Experiment visualizations

---

*Built with ❤️ for transparent, reproducible medical AI development.*
