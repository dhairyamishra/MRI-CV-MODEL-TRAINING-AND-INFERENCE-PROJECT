"""
SliceWise - AI-Powered Brain Tumor Detection System

A comprehensive Streamlit application for MRI brain tumor analysis featuring:
- Multi-task prediction (classification + segmentation)
- Standalone classification with Grad-CAM
- Tumor segmentation with uncertainty estimation
- Batch processing for multiple images
- Patient-level analysis with volume estimation

This is the refactored modular version using separated components.
All UI components are imported from the components package.

Author: SliceWise Team
Version: 2.0.0 (Modular)
Date: December 8, 2025
"""

import streamlit as st
from pathlib import Path

# Import modular components
from components import (
    render_header,
    render_sidebar,
    render_multitask_tab,
    render_classification_tab,
    render_segmentation_tab,
    render_batch_tab,
    render_patient_tab
)

# Import settings for CSS loading
from config.settings import AppMetadata


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title=f"{AppMetadata.APP_TITLE} - Brain Tumor Detection",
    page_icon=AppMetadata.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/slicewise',
        'Report a bug': 'https://github.com/yourusername/slicewise/issues',
        'About': f'{AppMetadata.APP_TITLE} - AI-powered MRI brain tumor detection'
    }
)


# ============================================================================
# Load Custom CSS
# ============================================================================

def load_css():
    """Load custom CSS from external files."""
    css_dir = Path(__file__).parent / "styles"
    
    # Load theme CSS
    theme_css_path = css_dir / "theme.css"
    if theme_css_path.exists():
        with open(theme_css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Load main CSS
    main_css_path = css_dir / "main.css"
    if main_css_path.exists():
        with open(main_css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """
    Main application entry point.
    
    Orchestrates the entire UI by:
    1. Loading custom CSS
    2. Rendering header
    3. Rendering sidebar (with API health check)
    4. Creating tabs for different functionalities
    5. Rendering appropriate tab content
    """
    # Load custom styling
    load_css()
    
    # Render header with branding and disclaimer
    render_header()
    
    # Render sidebar and check API availability
    api_available = render_sidebar()
    
    # If API is not available, show error and stop
    if not api_available:
        st.error("⚠️ Backend API is not available. Please start the server first.")
        st.code("python app/backend/main_v2.py", language="bash")
        st.info("""
        **Quick Start:**
        1. Open a terminal
        2. Run: `python app/backend/main_v2.py`
        3. Wait for "Application startup complete"
        4. Refresh this page
        """)
        return
    
    # Create main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Multi-Task",
        "🔍 Classification",
        "🎨 Segmentation",
        "📦 Batch Processing",
        "👤 Patient Analysis"
    ])
    
    # Render tab content
    with tab1:
        render_multitask_tab()
    
    with tab2:
        render_classification_tab()
    
    with tab3:
        render_segmentation_tab()
    
    with tab4:
        render_batch_tab()
    
    with tab5:
        render_patient_tab()
    
    # Footer
    st.divider()
    st.caption(
        f"{AppMetadata.APP_TITLE} v{AppMetadata.APP_VERSION} - {AppMetadata.APP_PHASE} | "
        f"Built with ❤️ using PyTorch, FastAPI, and Streamlit"
    )


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
