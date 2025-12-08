"""
SliceWise Frontend Components Package.

This package contains all UI components for the SliceWise application.
Import components directly from this package for cleaner code.

Example:
    >>> from components import render_header, render_sidebar
    >>> from components import render_multitask_tab, render_classification_tab
    
    >>> # In your main app
    >>> render_header()
    >>> api_available = render_sidebar()
    >>> if api_available:
    >>>     render_multitask_tab()
"""

# Import all component functions
from .header import render_header, render_simple_header
from .sidebar import render_sidebar, render_simple_sidebar
from .multitask_tab import render_multitask_tab
from .classification_tab import render_classification_tab
from .segmentation_tab import render_segmentation_tab
from .batch_tab import render_batch_tab
from .patient_tab import render_patient_tab

# Define what gets exported when using "from components import *"
__all__ = [
    # Header components
    'render_header',
    'render_simple_header',
    
    # Sidebar components
    'render_sidebar',
    'render_simple_sidebar',
    
    # Tab components
    'render_multitask_tab',
    'render_classification_tab',
    'render_segmentation_tab',
    'render_batch_tab',
    'render_patient_tab',
]

# Package metadata
__version__ = '2.0.0'
__author__ = 'SliceWise Team'
__description__ = 'Modular UI components for SliceWise MRI Brain Tumor Detection'
