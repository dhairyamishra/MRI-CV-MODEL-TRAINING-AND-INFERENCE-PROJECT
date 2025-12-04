# SliceWise Quick Start Guide

Welcome to SliceWise! This guide will get you up and running quickly.

## 🚀 Installation (5 minutes)

```bash
# 1. Navigate to project directory
cd slicewise

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Verify installation
python scripts/smoke_test.py
```

**Expected output:** Three images saved to `assets/smoke_test/`

## ✅ Verify Setup

```bash
# Run unit tests
pytest tests/ -v

# Check code formatting
black --check src/ tests/
isort --check-only src/ tests/
ruff check src/ tests/
```

## 📊 Download Data

### Kaggle Brain MRI Dataset (Quick Start)

```python
# In Python or Jupyter notebook
import kagglehub

# Download dataset (requires Kaggle API credentials)
path = kagglehub.dataset_download('navoneel/brain-mri-images-for-brain-tumor-detection')
print(f'Dataset downloaded to: {path}')
```

**Setup Kaggle API:**
1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### BraTS Dataset (Advanced)

See `documentation/DATA_README.md` for detailed instructions on accessing BraTS 2020/2021.

## 🎯 Next Steps

### Option 1: Explore the Codebase
```bash
# Read documentation
cat documentation/PROJECT_STRUCTURE.md
cat documentation/DATA_README.md
cat documentation/SETUP.md

# Explore source code structure
ls -R src/
```

### Option 2: Start Development (Phase 1)

**Implement data preprocessing:**
1. Create `src/data/preprocess_brats_2d.py`
2. Create `src/data/brats2d_dataset.py`
3. Create `src/data/transforms.py`

**Test with notebook:**
```bash
jupyter notebook jupyter_notebooks/
# Create: 01_visualize_brats_slices.ipynb
```

### Option 3: Run Existing Notebook

If you have the original notebook:
```bash
jupyter notebook jupyter_notebooks/MRI-Brain-Tumor-Detecor.ipynb
```

## 🛠️ Development Workflow

### Before Committing Code

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check linting
ruff check src/ tests/

# Run tests
pytest tests/

# Or use pre-commit (recommended)
pre-commit install
pre-commit run --all-files
```

### Creating New Features

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement feature**
   - Add code to appropriate `src/` module
   - Add tests to `tests/`
   - Update documentation if needed

3. **Test and format**
   ```bash
   pytest tests/
   black src/ tests/
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```

## 📁 Project Structure Overview

```
slicewise/
├── src/              # Source code (your main work area)
│   ├── data/        # Data loading and preprocessing
│   ├── models/      # Neural network architectures
│   ├── training/    # Training scripts and losses
│   ├── eval/        # Evaluation and metrics
│   └── inference/   # Inference and post-processing
├── configs/         # YAML configuration files
├── scripts/         # Utility scripts
├── tests/           # Unit tests
├── jupyter_notebooks/       # Jupyter notebooks
└── app/             # FastAPI backend + Streamlit frontend
```

## 🔧 Configuration

### Local Development
Use `configs/local.yaml`:
- Small batch sizes
- CPU or single GPU
- Relative paths
- W&B disabled

### HPC Training
Use `configs/hpc.yaml`:
- Large batch sizes
- Multi-GPU support
- Scratch space paths
- W&B enabled

## 📚 Key Documentation

- **documentation/SETUP.md** - Detailed installation guide
- **documentation/DATA_README.md** - Dataset documentation
- **documentation/PROJECT_STRUCTURE.md** - Codebase organization
- **documentation/PHASE0_COMPLETE.md** - Phase 0 completion summary
- **documentation/FULL-PLAN.md** - Complete project roadmap

## 🐛 Troubleshooting

### Import Errors
```bash
# Reinstall in editable mode
pip install -e .
```

### CUDA Not Available
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with CUDA support
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Pre-commit Hook Failures
```bash
# Update hooks
pre-commit autoupdate

# Run manually to see errors
pre-commit run --all-files
```

## 💡 Tips

1. **Use configs for everything** - Don't hardcode hyperparameters
2. **Write tests** - Add tests for new functionality
3. **Document your code** - Add docstrings to functions/classes
4. **Follow code style** - Use black, isort, ruff
5. **Commit often** - Small, focused commits are better

## 🎓 Learning Resources

- **PyTorch:** https://pytorch.org/tutorials/
- **MONAI:** https://docs.monai.io/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Streamlit:** https://docs.streamlit.io/

## 🤝 Getting Help

1. Check documentation files in `documentation/` folder
2. Review `documentation/PROJECT_STRUCTURE.md` for codebase layout
3. Look at existing code examples
4. Open an issue on GitHub

---

**Phase 0 Complete!** ✅  
Ready to start Phase 1: Data Acquisition & 2D Preprocessing

Happy coding! 🚀
