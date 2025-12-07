#!/usr/bin/env python3
"""
Helper script to train U-Net segmentation model.

Usage:
    python scripts/train_segmentation.py
    python scripts/train_segmentation.py --config configs/seg2d_baseline.yaml
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.train_seg2d import main

if __name__ == "__main__":
    main()
