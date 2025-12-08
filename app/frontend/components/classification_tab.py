"""
Classification Tab Component for SliceWise Frontend.

This module renders the classification tab for tumor detection
with Grad-CAM visualization and calibrated probabilities.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from PIL import Image

# Import settings and utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import Colors, UIConfig
from utils.api_client import classify_image
from utils.image_utils import base64_to_image, image_to_bytes


def render_classification_tab():
    """
    Render classification tab for tumor detection.
    
    Displays:
    - File upload
    - Options (Grad-CAM, calibration)
    - Classification results
    - Probability charts (raw and calibrated)
    - Grad-CAM visualization
    
    Example:
        >>> from components.classification_tab import render_classification_tab
        >>> render_classification_tab()
    """
    st.header("🔍 Classification: Tumor Detection")
    
    st.markdown("""
    Upload an MRI slice to classify whether it contains a tumor. 
    The model provides predictions with confidence scores and optional Grad-CAM visualization.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    # Left column: Upload
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an MRI slice...",
            type=["jpg", "jpeg", "png", "bmp"],
            key="cls_upload"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=UIConfig.IMAGE_WIDTH_LARGE)
            st.caption(f"Size: {image.size[0]}×{image.size[1]} | Mode: {image.mode}")
    
    # Right column: Options
    with col2:
        st.subheader("⚙️ Options")
        show_gradcam = st.checkbox("Generate Grad-CAM", value=True, key="cls_gradcam")
        show_calibration = st.checkbox("Show Calibrated Probabilities", value=True, key="cls_calib")
        
        if uploaded_file:
            if st.button("🔍 Classify", type="primary", use_container_width=True):
                _run_classification(image, show_gradcam)
    
    # Display results
    if 'cls_result' in st.session_state:
        _render_results(st.session_state['cls_result'], image, show_gradcam, show_calibration)


def _run_classification(image: Image.Image, show_gradcam: bool):
    """
    Run classification and store results in session state.
    
    Args:
        image: PIL Image object
        show_gradcam: Whether to generate Grad-CAM
    """
    with st.spinner("Analyzing..."):
        # Prepare image
        img_bytes = image_to_bytes(image)
        
        # Call API
        result, error = classify_image(img_bytes, return_gradcam=show_gradcam)
        
        if error:
            st.error(f"❌ {error}")
            return
        
        # Store in session state and rerun
        st.session_state['cls_result'] = result
        st.rerun()


def _render_results(result: dict, image: Image.Image, show_gradcam: bool, show_calibration: bool):
    """
    Render classification results.
    
    Args:
        result: Classification result dictionary
        image: Original PIL Image
        show_gradcam: Whether Grad-CAM was requested
        show_calibration: Whether to show calibrated probabilities
    """
    st.divider()
    st.header("📊 Results")
    
    # Prediction summary
    predicted_label = result['predicted_label']
    confidence = result['confidence']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color = "🔴" if predicted_label == "Tumor" else "🟢"
        st.metric("Prediction", f"{color} {predicted_label}")
    
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        certainty = _get_certainty_level(confidence)
        st.metric("Certainty", certainty)
    
    # Probability charts
    col1, col2 = st.columns(2)
    
    with col1:
        _render_probability_chart(
            result['probabilities'],
            "📈 Raw Probabilities"
        )
    
    with col2:
        if show_calibration and result.get('calibrated_probabilities'):
            _render_probability_chart(
                result['calibrated_probabilities'],
                "🎯 Calibrated Probabilities"
            )
    
    # Grad-CAM visualization
    if show_gradcam and result.get('gradcam_overlay'):
        _render_gradcam(image, result)
    
    # Clear button
    if st.button("🔄 Analyze Another Image", key="cls_clear"):
        del st.session_state['cls_result']
        st.rerun()


def _get_certainty_level(confidence: float) -> str:
    """
    Get certainty level description based on confidence.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Certainty level string
    """
    if confidence > 0.9:
        return "Very High"
    elif confidence > 0.75:
        return "High"
    elif confidence > 0.6:
        return "Moderate"
    else:
        return "Low"


def _render_probability_chart(probabilities: dict, title: str):
    """
    Render probability bar chart.
    
    Args:
        probabilities: Dictionary of class probabilities
        title: Chart title
    """
    st.subheader(title)
    
    prob_df = pd.DataFrame({
        'Class': list(probabilities.keys()),
        'Probability': list(probabilities.values())
    })
    
    fig, ax = plt.subplots(figsize=(UIConfig.CHART_WIDTH_SMALL, UIConfig.CHART_HEIGHT_MEDIUM))
    bars = ax.barh(prob_df['Class'], prob_df['Probability'])
    
    # Color bars
    colors = [
        Colors.CHART_NO_TUMOR if label == 'No Tumor' else Colors.CHART_TUMOR
        for label in prob_df['Class']
    ]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Probability')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (cls, prob) in enumerate(zip(prob_df['Class'], prob_df['Probability'])):
        ax.text(prob + 0.02, i, f'{prob:.1%}', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _render_gradcam(original_image: Image.Image, result: dict):
    """
    Render Grad-CAM visualization.
    
    Args:
        original_image: Original PIL Image
        result: Classification result with Grad-CAM
    """
    st.subheader("🔥 Grad-CAM Visualization")
    st.markdown("""
    The heatmap shows which regions the model focused on when making its prediction.
    **Red/Yellow** = high importance, **Blue** = low importance.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original", width=UIConfig.IMAGE_WIDTH_XLARGE)
    with col2:
        gradcam_img = base64_to_image(result['gradcam_overlay'])
        st.image(gradcam_img, caption="Grad-CAM Overlay", width=UIConfig.IMAGE_WIDTH_XLARGE)


# Export function
__all__ = ['render_classification_tab']
