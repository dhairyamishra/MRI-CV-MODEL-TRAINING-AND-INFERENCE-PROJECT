"""
Patient Analysis Tab Component for SliceWise Frontend.

This module renders the patient-level analysis tab for comprehensive
tumor detection and volume estimation across multiple slices.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path
import sys
from PIL import Image

# Import settings and utilities
from app.frontend.config.settings import UIConfig, ModelConfig, Colors
from app.frontend.utils.api_client import analyze_patient_stack
from app.frontend.utils.validators import validate_patient_id


def render_patient_tab():
    """
    Render patient-level analysis tab.
    
    Displays:
    - Patient ID input
    - Multiple slice upload
    - Slice preview
    - Analysis options
    - Patient-level results
    - Slice-by-slice analysis
    - Tumor distribution charts
    - CSV and JSON download
    
    Example:
        >>> from components.patient_tab import render_patient_tab
        >>> render_patient_tab()
    """
    st.header("👤 Patient-Level Analysis")
    
    st.markdown("""
    Upload a stack of MRI slices from a single patient for comprehensive analysis.
    The system will provide patient-level tumor detection and volume estimation.
    """)
    
    # Patient ID input
    patient_id = st.text_input("Patient ID", value="PATIENT_001", key="patient_id")
    
    # Validate patient ID
    is_valid, error = validate_patient_id(patient_id)
    if not is_valid:
        st.warning(f"⚠️ {error}")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload patient MRI stack (multiple slices)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="patient_upload"
    )
    
    if uploaded_files and is_valid:
        st.info(f"📁 {len(uploaded_files)} slices uploaded for patient: {patient_id}")
        
        # Show preview
        _render_preview(uploaded_files)
        
        # Options
        threshold, min_area, slice_thickness = _render_options()
        
        # Analyze button
        if st.button("🔬 Analyze Patient", type="primary", use_container_width=True):
            _analyze_patient(uploaded_files, patient_id, threshold, min_area, slice_thickness)
    
    # Display results
    if 'patient_result' in st.session_state:
        _render_results(st.session_state['patient_result'])


def _render_preview(files: list):
    """
    Render preview of uploaded slices.
    
    Args:
        files: List of uploaded files
    """
    with st.expander("Preview Slices"):
        cols = st.columns(min(5, len(files)))
        for idx, (col, file) in enumerate(zip(cols, files[:5])):
            with col:
                image = Image.open(file)
                st.image(image, caption=f"Slice {idx}", width=UIConfig.IMAGE_WIDTH_LARGE)
        if len(files) > 5:
            st.caption(f"... and {len(files) - 5} more slices")


def _render_options():
    """
    Render analysis options.
    
    Returns:
        Tuple of (threshold, min_area, slice_thickness)
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        threshold = st.slider(
            "Threshold",
            ModelConfig.SEGMENTATION_THRESHOLD_MIN,
            ModelConfig.SEGMENTATION_THRESHOLD_MAX,
            ModelConfig.SEGMENTATION_THRESHOLD_DEFAULT,
            ModelConfig.SEGMENTATION_THRESHOLD_STEP,
            key="patient_thresh"
        )
    
    with col2:
        min_area = st.number_input(
            "Min Area",
            ModelConfig.MIN_TUMOR_AREA_MIN,
            ModelConfig.MIN_TUMOR_AREA_MAX,
            ModelConfig.MIN_TUMOR_AREA_PIXELS,
            key="patient_min_area"
        )
    
    with col3:
        slice_thickness = st.number_input(
            "Slice Thickness (mm)",
            0.1, 10.0, 1.0, 0.1,
            key="patient_thickness"
        )
    
    return threshold, min_area, slice_thickness


def _analyze_patient(files: list, patient_id: str, threshold: float, min_area: int, slice_thickness: float):
    """
    Analyze patient stack.
    
    Args:
        files: List of uploaded files
        patient_id: Patient identifier
        threshold: Probability threshold
        min_area: Minimum tumor area
        slice_thickness: Slice thickness in mm
    """
    with st.spinner(f"Analyzing {len(files)} slices..."):
        # Prepare files
        files_list = []
        for file in files:
            file.seek(0)
            files_list.append((file.name, file.read()))
        
        # Call API
        result, error = analyze_patient_stack(
            files_list,
            patient_id,
            threshold,
            min_area,
            slice_thickness
        )
        
        if error:
            st.error(f"❌ {error}")
            return
        
        # Store in session state and rerun
        st.session_state['patient_result'] = result
        st.rerun()


def _render_results(result: dict):
    """
    Render patient analysis results.
    
    Args:
        result: Patient analysis result dictionary
    """
    st.divider()
    st.header("📊 Patient Analysis Results")
    
    # Patient summary
    _render_patient_summary(result)
    
    # Patient-level metrics
    _render_patient_metrics(result)
    
    # Slice-by-slice analysis
    _render_slice_analysis(result)
    
    # Tumor distribution charts
    _render_tumor_distribution(result)
    
    # Download buttons
    _render_download_buttons(result)
    
    # Clear button
    if st.button("🔄 Analyze Another Patient", key="patient_clear"):
        del st.session_state['patient_result']
        st.rerun()


def _render_patient_summary(result: dict):
    """Render patient summary metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Patient ID", result['patient_id'])
    
    with col2:
        st.metric("Total Slices", result['num_slices'])
    
    with col3:
        status = "🔴 Detected" if result['has_tumor'] else "🟢 Not Detected"
        st.metric("Tumor Status", status)
    
    with col4:
        if result['tumor_volume_mm3']:
            st.metric("Tumor Volume", f"{result['tumor_volume_mm3']:.1f} mm³")
        else:
            st.metric("Tumor Volume", "N/A")


def _render_patient_metrics(result: dict):
    """Render patient-level metrics."""
    st.subheader("📈 Patient-Level Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Affected Slices", result['affected_slices'])
    
    with col2:
        ratio = result['patient_level_metrics']['affected_slice_ratio']
        st.metric("Affected Ratio", f"{ratio:.1%}")
    
    with col3:
        max_area = result['patient_level_metrics']['max_tumor_area']
        st.metric("Max Tumor Area", f"{max_area} px")


def _render_slice_analysis(result: dict):
    """Render slice-by-slice analysis table."""
    st.subheader("🔬 Slice-by-Slice Analysis")
    
    slice_df = pd.DataFrame(result['slice_predictions'])
    
    # Add color coding for tumor-positive slices
    def highlight_tumor(row):
        if row['has_tumor']:
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)
    
    styled_df = slice_df.style.apply(highlight_tumor, axis=1)
    st.dataframe(styled_df, use_container_width=True)


def _render_tumor_distribution(result: dict):
    """Render tumor distribution charts."""
    st.subheader("📊 Tumor Distribution")
    
    slice_df = pd.DataFrame(result['slice_predictions'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(UIConfig.CHART_WIDTH_LARGE, UIConfig.CHART_HEIGHT_MEDIUM))
    
    # Tumor area per slice
    ax1.bar(
        slice_df['slice_index'],
        slice_df['tumor_area_pixels'],
        color=Colors.CHART_TUMOR,
        alpha=0.7
    )
    ax1.set_xlabel('Slice Index')
    ax1.set_ylabel('Tumor Area (pixels)')
    ax1.set_title('Tumor Area per Slice')
    ax1.grid(axis='y', alpha=0.3)
    
    # Probability per slice
    ax2.plot(
        slice_df['slice_index'],
        slice_df['max_probability'],
        marker='o',
        color=Colors.CHART_NEUTRAL
    )
    ax2.axhline(
        y=0.5,
        color=Colors.CHART_TUMOR,
        linestyle='--',
        alpha=0.5,
        label='Threshold'
    )
    ax2.set_xlabel('Slice Index')
    ax2.set_ylabel('Max Probability')
    ax2.set_title('Tumor Probability per Slice')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _render_download_buttons(result: dict):
    """Render download buttons for results."""
    col1, col2 = st.columns(2)
    
    slice_df = pd.DataFrame(result['slice_predictions'])
    
    with col1:
        csv = slice_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Slice Results (CSV)",
            data=csv,
            file_name=f"patient_{result['patient_id']}_slices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_str = json.dumps(result, indent=2)
        st.download_button(
            label="📥 Download Full Report (JSON)",
            data=json_str,
            file_name=f"patient_{result['patient_id']}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


# Export function
__all__ = ['render_patient_tab']
