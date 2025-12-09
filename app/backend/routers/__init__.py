"""
API routers for endpoint organization.

This module contains all FastAPI routers that define the API endpoints.
"""

from . import health
from . import classification
from . import segmentation
from . import multitask
from . import patient

__all__ = [
    "health",
    "classification",
    "segmentation",
    "multitask",
    "patient",
]
