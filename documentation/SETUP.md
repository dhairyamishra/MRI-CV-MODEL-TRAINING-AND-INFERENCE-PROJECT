# SliceWise Setup Guide

This guide will help you set up the SliceWise project on your local machine or HPC cluster.

## Prerequisites

- Python 3.10 or 3.11
- Git
- CUDA-capable GPU (optional, but recommended for training)
- 50+ GB free disk space for datasets

## Quick Start (Local)

```bash
# Clone the repository
git clone <repository-url>
cd slicewise

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run smoke test
python scripts/smoke_test.py
```

## Detailed Installation

### 1. Create Virtual Environment

**Option A: venv (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Option B: conda**
```bash
conda create -n slicewise python=3.10
conda activate slicewise
```

### 2. Install PyTorch

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install SliceWise

**Development mode (recommended for contributors):**
```bash
pip install -e ".[dev]"
```

**Standard installation:**
```bash
pip install -e .
```

**From requirements.txt:**
```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check MONAI installation
python -c "import monai; print(f'MONAI: {monai.__version__}')"

# Run smoke test
python scripts/smoke_test.py
```

## HPC Setup (NYU HPC or similar)

### 1. Request GPU Access

Submit a request for GPU allocation (A100, T4, or L4).

### 2. Load Modules

```bash
# Load required modules
module load python/3.10
module load cuda/11.8
module load cudnn/8.6.0
```

### 3. Create Environment in Scratch Space

```bash
# Navigate to scratch space
cd /scratch/$USER

# Clone repository
git clone <repository-url> slicewise
cd slicewise

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

### 4. Configure Paths

Edit `configs/hpc.yaml` to set correct paths:
```yaml
paths:
  data_root: "/scratch/$USER/slicewise"
  datasets:
    brats2020: "/scratch/$USER/datasets/brats2020"
    # ... etc
```

### 5. Submit Test Job

```bash
# Create a test job script
cat > test_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=slicewise_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

module load python/3.10
module load cuda/11.8
source venv/bin/activate

python scripts/smoke_test.py
EOF

# Submit job
sbatch test_job.sh
```

## Data Setup

### 1. Configure Kaggle API

```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Copy your kaggle.json (download from kaggle.com/account)
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Download Kaggle Brain MRI Dataset

```bash
# Using kagglehub (recommended)
python -c "import kagglehub; path = kagglehub.dataset_download('navoneel/brain-mri-images-for-brain-tumor-detection'); print(f'Downloaded to: {path}')"

# Or using Kaggle CLI
kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
unzip brain-mri-images-for-brain-tumor-detection.zip -d data/raw/kaggle_brain_mri/
```

### 3. Download BraTS Dataset

See `documentation/DATA_README.md` for detailed instructions on accessing BraTS 2020/2021.

## Development Setup

### 1. Install Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run code formatting and linting before each commit.

### 2. Run Pre-commit Manually

```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_datasets.py
```

### 4. Code Formatting

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
ruff check src/ tests/
```

## Experiment Tracking Setup

### Weights & Biases (W&B)

```bash
# Login to W&B
wandb login

# Or set API key
export WANDB_API_KEY=your_api_key_here
```

Edit `configs/hpc.yaml` or `configs/local.yaml`:
```yaml
logging:
  use_wandb: true
  wandb_project: "slicewise"
  wandb_entity: "your_username"
```

### MLflow (Alternative)

```bash
# Start MLflow server
mlflow ui --port 5000

# In another terminal, run training
python -m src.training.train_seg2d --config configs/seg2d_baseline.yaml
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:
```yaml
training:
  batch_size:
    train: 8  # Reduce from 32
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Slow Data Loading

Increase number of workers in config:
```yaml
compute:
  num_workers: 8  # Adjust based on CPU cores
```

### Pre-commit Hook Failures

```bash
# Update hooks
pre-commit autoupdate

# Clear cache
pre-commit clean
```

## Next Steps

1. **Run smoke test**: `python scripts/smoke_test.py`
2. **Download data**: See `documentation/DATA_README.md`
3. **Explore notebooks**: `jupyter notebook jupyter_notebooks/`
4. **Start training**: See `README.md` for training instructions

## Getting Help

- Check `documentation/PROJECT_STRUCTURE.md` for codebase organization
- Check `documentation/DATA_README.md` for dataset information
- Check `README.md` for usage examples
- Open an issue on GitHub for bugs or questions
