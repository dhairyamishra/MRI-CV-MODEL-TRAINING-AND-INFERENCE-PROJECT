"""
Evaluation module for SliceWise MRI Brain Tumor Detection.

This module contains evaluation scripts, metrics, and visualization tools.
"""

from .eval_cls import evaluate_classifier
from .grad_cam import GradCAM, generate_gradcam_visualizations

__all__ = ["evaluate_classifier", "GradCAM", "generate_gradcam_visualizations"]
