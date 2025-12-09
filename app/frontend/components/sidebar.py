"""
Sidebar Component for SliceWise Frontend.

This module renders the sidebar with API health status,
model information, and application details.
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
import sys

# Import settings and utilities
from app.frontend.config.settings import AppMetadata, Colors
from app.frontend.utils.api_client import check_api_health, get_model_info


def render_sidebar() -> bool:
    """
    Render sidebar with system status, model info, and about section.
    
    Displays:
    - API connection status
    - Model loading status (Multi-Task, Classifier, Segmentation)
    - Calibration status
    - Device information
    - Model details (expandable)
    - About section with features
    
    Returns:
        bool: True if API is available, False otherwise
        
    Example:
        >>> from components.sidebar import render_sidebar
        >>> api_available = render_sidebar()
        >>> if not api_available:
        >>>     st.stop()
    """
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Check API health
        health = check_api_health()
        
        if health and health['status'] in ['healthy', 'no_models_loaded']:
            st.success("[OK] API Connected")
            
            # Show model status in 3 columns (Multi-Task, Classifier, Segmentation)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Multi-Task**")
                if health.get('multitask_loaded', False):
                    st.markdown("[OK]")
                    st.markdown(
                        f'<span style="color: {Colors.SUCCESS_GREEN}; font-size: 0.8rem;">Loaded</span>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("✗")
                    st.markdown(
                        f'<span style="color: {Colors.DANGER_RED}; font-size: 0.8rem;">Not loaded</span>',
                        unsafe_allow_html=True
                    )
            
            with col2:
                st.markdown("**Classifier**")
                if health['classifier_loaded']:
                    st.markdown("[OK]")
                    st.markdown(
                        f'<span style="color: {Colors.SUCCESS_GREEN}; font-size: 0.8rem;">Loaded</span>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("✗")
                    st.markdown(
                        f'<span style="color: {Colors.DANGER_RED}; font-size: 0.8rem;">Not loaded</span>',
                        unsafe_allow_html=True
                    )
            
            with col3:
                st.markdown("**Segmentation**")
                if health['segmentation_loaded']:
                    st.markdown("[OK]")
                    st.markdown(
                        f'<span style="color: {Colors.SUCCESS_GREEN}; font-size: 0.8rem;">Loaded</span>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("✗")
                    st.markdown(
                        f'<span style="color: {Colors.DANGER_RED}; font-size: 0.8rem;">Not loaded</span>',
                        unsafe_allow_html=True
                    )
            
            # Additional features
            st.markdown("")  # Spacing
            if health.get('calibration_loaded', False):
                st.info("🎯 Calibration: Enabled")
            
            st.caption(f"Device: {health['device']}")
            st.caption(f"Updated: {datetime.fromisoformat(health['timestamp']).strftime('%H:%M:%S')}")
        else:
            st.error("✗ API Not Available")
            st.warning("Please start the backend server:\n```bash\npython app/backend/main_v2.py\n```")
            return False
        
        st.divider()
        
        # Model information
        if health and (health.get('multitask_loaded') or health['classifier_loaded'] or health['segmentation_loaded']):
            st.subheader("📊 Model Info")
            model_info = get_model_info()
            if model_info:
                # Multi-Task Model Details
                if model_info.get('multitask'):
                    with st.expander("Multi-Task Model Details", expanded=False):
                        mt_info = model_info['multitask']
                        st.write(f"**Architecture:** {mt_info.get('architecture', 'N/A')}")
                        st.write(f"**Parameters:** {mt_info.get('parameters', 'N/A')}")
                        st.write(f"**Tasks:** {', '.join(mt_info.get('tasks', []))}")
                        st.write(f"**Classification Threshold:** {mt_info.get('classification_threshold', 'N/A')}")
                        st.write(f"**Segmentation Threshold:** {mt_info.get('segmentation_threshold', 'N/A')}")
                        
                        # Performance metrics
                        if mt_info.get('performance'):
                            st.write("**Performance:**")
                            perf = mt_info['performance']
                            st.write(f"- Accuracy: {perf.get('classification_accuracy', 0)*100:.1f}%")
                            st.write(f"- Sensitivity: {perf.get('classification_sensitivity', 0)*100:.1f}%")
                            st.write(f"- Dice Score: {perf.get('segmentation_dice', 0)*100:.1f}%")
                
                # Classifier Details
                if model_info.get('classifier'):
                    with st.expander("Classifier Details"):
                        cls_info = model_info['classifier']
                        st.write(f"**Architecture:** {cls_info.get('architecture', 'N/A')}")
                        st.write(f"**Classes:** {', '.join(cls_info.get('class_names', []))}")
                        st.write(f"**Calibrated:** {'Yes' if cls_info.get('calibrated') else 'No'}")
                
                # Segmentation Details
                if model_info.get('segmentation'):
                    with st.expander("Segmentation Details"):
                        seg_info = model_info['segmentation']
                        st.write(f"**Architecture:** {seg_info.get('architecture', 'N/A')}")
                        st.write(f"**Parameters:** {seg_info.get('parameters', 'N/A')}")
                        st.write(f"**Uncertainty:** {'Yes' if seg_info.get('uncertainty_estimation') else 'No'}")
                
                # Available Features
                features = model_info.get('features', [])
                if features:
                    st.write("**Available Features:**")
                    for feature in features:
                        st.write(f"- {feature.replace('_', ' ').title()}")
        
        st.divider()
        
        # About section
        _render_about_section()
        
        return True


def _render_about_section():
    """
    Render the about section in the sidebar.
    
    Internal helper function to display application information.
    """
    st.subheader("ℹ️ About")
    
    # Build features list from AppMetadata
    features_text = "\n".join([f"- {feature}" for feature in AppMetadata.FEATURES])
    
    st.markdown(f"""
    **SliceWise v2** integrates all Phase 0-5 components:
    
    {features_text}
    
    **Total:** {AppMetadata.TOTAL_LINES_OF_CODE} lines of code
    """)


def render_simple_sidebar() -> bool:
    """
    Render a simplified sidebar with only API status.
    
    Returns:
        bool: True if API is available, False otherwise
        
    Example:
        >>> from components.sidebar import render_simple_sidebar
        >>> api_available = render_simple_sidebar()
    """
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Check API health
        health = check_api_health()
        
        if health and health['status'] in ['healthy', 'no_models_loaded']:
            st.success("[OK] API Connected")
            st.caption(f"Device: {health['device']}")
            return True
        else:
            st.error("✗ API Not Available")
            return False


# Export functions
__all__ = ['render_sidebar', 'render_simple_sidebar']
