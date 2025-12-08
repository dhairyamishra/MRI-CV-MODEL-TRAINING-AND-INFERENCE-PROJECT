"""
Batch Processing Tab Component for SliceWise Frontend.

This module renders the batch processing tab for multiple images
with classification or segmentation modes.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
from PIL import Image

# Import settings and utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import UIConfig, ModelConfig
from utils.api_client import classify_batch, segment_batch
from utils.validators import validate_batch_and_display


def render_batch_tab():
    """
    Render batch processing tab for multiple images.
    
    Displays:
    - Processing mode selection (Classification/Segmentation)
    - Multiple file upload
    - Image preview
    - Batch processing options
    - Results summary and detailed table
    - CSV download
    
    Example:
        >>> from components.batch_tab import render_batch_tab
        >>> render_batch_tab()
    """
    st.header("📦 Batch Processing")
    
    st.markdown("""
    Upload multiple MRI slices for batch classification or segmentation.
    Results can be downloaded as CSV for further analysis.
    """)
    
    # Processing mode selection
    mode = st.radio("Processing Mode", ["Classification", "Segmentation"], horizontal=True)
    
    # File upload
    uploaded_files = st.file_uploader(
        f"Upload multiple MRI slices (max {UIConfig.MAX_BATCH_SIZE})",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files:
        # Validate batch
        is_valid, valid_files = validate_batch_and_display(uploaded_files, UIConfig.MAX_BATCH_SIZE)
        
        if is_valid:
            # Show preview
            _render_preview(valid_files)
            
            # Options and process button
            if mode == "Segmentation":
                threshold, min_area = _render_segmentation_options()
                if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                    _process_batch_segmentation(valid_files, threshold, min_area)
            else:
                if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                    _process_batch_classification(valid_files)
    
    # Display results
    if 'batch_result' in st.session_state:
        _render_results(st.session_state['batch_result'], st.session_state['batch_mode'])


def _render_preview(files: list):
    """
    Render preview of uploaded images.
    
    Args:
        files: List of uploaded files
    """
    with st.expander("Preview Images"):
        cols = st.columns(min(5, len(files)))
        for idx, (col, file) in enumerate(zip(cols, files[:5])):
            with col:
                image = Image.open(file)
                st.image(image, caption=file.name, width=UIConfig.IMAGE_WIDTH_LARGE)
        if len(files) > 5:
            st.caption(f"... and {len(files) - 5} more")


def _render_segmentation_options():
    """
    Render segmentation-specific options.
    
    Returns:
        Tuple of (threshold, min_area)
    """
    col1, col2 = st.columns(2)
    
    with col1:
        threshold = st.slider(
            "Threshold",
            ModelConfig.SEGMENTATION_THRESHOLD_MIN,
            ModelConfig.SEGMENTATION_THRESHOLD_MAX,
            ModelConfig.SEGMENTATION_THRESHOLD_DEFAULT,
            ModelConfig.SEGMENTATION_THRESHOLD_STEP,
            key="batch_thresh"
        )
    
    with col2:
        min_area = st.number_input(
            "Min Area",
            ModelConfig.MIN_TUMOR_AREA_MIN,
            ModelConfig.MIN_TUMOR_AREA_MAX,
            ModelConfig.MIN_TUMOR_AREA_PIXELS,
            key="batch_min_area"
        )
    
    return threshold, min_area


def _process_batch_classification(files: list):
    """
    Process batch classification.
    
    Args:
        files: List of uploaded files
    """
    with st.spinner(f"Processing {len(files)} images..."):
        # Prepare files
        files_list = []
        for file in files:
            file.seek(0)  # Reset file pointer
            files_list.append((file.name, file.read()))
        
        # Call API
        result, error = classify_batch(files_list, return_gradcam=False)
        
        if error:
            st.error(f"❌ {error}")
            return
        
        # Store in session state and rerun
        st.session_state['batch_result'] = result
        st.session_state['batch_mode'] = "Classification"
        st.rerun()


def _process_batch_segmentation(files: list, threshold: float, min_area: int):
    """
    Process batch segmentation.
    
    Args:
        files: List of uploaded files
        threshold: Probability threshold
        min_area: Minimum tumor area
    """
    with st.spinner(f"Processing {len(files)} images..."):
        # Prepare files
        files_list = []
        for file in files:
            file.seek(0)  # Reset file pointer
            files_list.append((file.name, file.read()))
        
        # Call API
        result, error = segment_batch(files_list, threshold, min_area)
        
        if error:
            st.error(f"❌ {error}")
            return
        
        # Store in session state and rerun
        st.session_state['batch_result'] = result
        st.session_state['batch_mode'] = "Segmentation"
        st.rerun()


def _render_results(result: dict, mode: str):
    """
    Render batch processing results.
    
    Args:
        result: Batch processing result dictionary
        mode: Processing mode (Classification or Segmentation)
    """
    st.divider()
    st.header("📊 Batch Results")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Images", result['num_images'])
    
    with col2:
        st.metric("Processing Time", f"{result['processing_time_seconds']:.2f}s")
    
    with col3:
        avg_time = result['processing_time_seconds'] / result['num_images']
        st.metric("Avg Time/Image", f"{avg_time:.3f}s")
    
    # Summary statistics
    st.subheader("📈 Summary")
    summary = result['summary']
    
    if mode == "Classification":
        _render_classification_summary(summary)
    else:
        _render_segmentation_summary(summary)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    results_df = pd.DataFrame(result['results'])
    st.dataframe(results_df, use_container_width=True)
    
    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"batch_{mode.lower()}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Clear button
    if st.button("🔄 Process Another Batch", key="batch_clear"):
        del st.session_state['batch_result']
        del st.session_state['batch_mode']
        st.rerun()


def _render_classification_summary(summary: dict):
    """
    Render classification summary statistics.
    
    Args:
        summary: Summary statistics dictionary
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🔴 Tumor Detected", summary['tumor_detected'])
    
    with col2:
        st.metric("🟢 No Tumor", summary['no_tumor'])


def _render_segmentation_summary(summary: dict):
    """
    Render segmentation summary statistics.
    
    Args:
        summary: Summary statistics dictionary
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Slices with Tumor", summary['slices_with_tumor'])
    
    with col2:
        st.metric("Slices without Tumor", summary['slices_without_tumor'])
    
    with col3:
        st.metric("Total Tumor Area", f"{summary['total_tumor_area_pixels']} px")


# Export function
__all__ = ['render_batch_tab']
