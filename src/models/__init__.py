"""
Models module for SliceWise MRI Brain Tumor Detection.

This module contains neural network architectures for classification and segmentation.
"""

from .classifier import BrainTumorClassifier

__all__ = ["BrainTumorClassifier"]
