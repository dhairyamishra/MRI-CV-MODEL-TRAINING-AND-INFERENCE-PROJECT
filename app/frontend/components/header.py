"""
Header Component for SliceWise Frontend.

This module renders the application header including title,
subtitle, version badge, and medical disclaimer.
"""

import streamlit as st
from pathlib import Path
import sys

# Import settings
from app.frontend.config.settings import AppMetadata


def render_header():
    """
    Render application header with title, subtitle, and medical disclaimer.
    
    Displays:
    - App title with icon
    - Subtitle describing the application
    - Version badge
    - Medical disclaimer warning box
    
    Example:
        >>> from components.header import render_header
        >>> render_header()
    """
    # Create centered layout for header
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Main title
        st.markdown(
            f'<div class="main-header">{AppMetadata.APP_ICON} SliceWise v2</div>',
            unsafe_allow_html=True
        )
        
        # Subtitle
        st.markdown(
            '<div class="sub-header">Comprehensive AI-Powered Brain Tumor Detection</div>',
            unsafe_allow_html=True
        )
        
        # Version badge
        st.markdown(
            f'<div style="text-align: center;"><span class="version-badge">{AppMetadata.APP_PHASE}</span></div>',
            unsafe_allow_html=True
        )
    
    # Medical disclaimer warning box
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
        It is NOT a medical device and should NOT be used for clinical diagnosis.
        Always consult qualified healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)


def render_simple_header(show_disclaimer: bool = True):
    """
    Render a simplified header without columns.
    
    Args:
        show_disclaimer: Whether to show the medical disclaimer
        
    Example:
        >>> from components.header import render_simple_header
        >>> render_simple_header(show_disclaimer=False)
    """
    # Simple centered title
    st.markdown(
        f'<div class="main-header" style="text-align: center;">{AppMetadata.APP_ICON} SliceWise v2</div>',
        unsafe_allow_html=True
    )
    
    # Subtitle
    st.markdown(
        '<div class="sub-header" style="text-align: center;">AI-Powered Brain Tumor Detection</div>',
        unsafe_allow_html=True
    )
    
    # Medical disclaimer (optional)
    if show_disclaimer:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
            It is NOT a medical device and should NOT be used for clinical diagnosis.
            Always consult qualified healthcare professionals for medical advice.
        </div>
        """, unsafe_allow_html=True)


# Export functions
__all__ = ['render_header', 'render_simple_header']
