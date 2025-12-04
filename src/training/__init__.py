"""
Training module for SliceWise MRI Brain Tumor Detection.

This module contains training loops, loss functions, and utilities for model training.
"""

from .train_cls import train_classifier

__all__ = ["train_classifier"]
