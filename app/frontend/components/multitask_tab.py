"""
Multi-Task Tab Component for SliceWise Frontend.

This module renders the multi-task prediction tab with unified
classification and conditional segmentation.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import settings and utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import Colors, UIConfig, ClinicalGuidelines
from utils.api_client import predict_multitask
from utils.image_utils import base64_to_image, image_to_bytes
from utils.validators import validate_and_display_file
from PIL import Image


def render_multitask_tab():
    """
    Render multi-task prediction tab with conditional display.
    
    Displays:
    - Model description and benefits
    - Performance metrics (expandable)
    - File upload
    - Prediction results (classification + segmentation)
    - Grad-CAM visualization
    - Clinical interpretation
    
    Example:
        >>> from components.multitask_tab import render_multitask_tab
        >>> render_multitask_tab()
    """
    st.markdown("### 🎯 Multi-Task Prediction")
    st.markdown("""
    This unified model performs **both classification and segmentation** in a single forward pass.
    Segmentation is only computed if the tumor probability is above the threshold (30%).
    
    **Benefits:**
    - 🚀 ~40% faster inference (single forward pass)
    - 💾 9.4% fewer parameters (2.0M vs 2.2M)
    - 🎯 Smart resource usage (conditional segmentation)
    - 📊 Excellent performance: 91.3% accuracy, 97.1% sensitivity
    """)
    
    # Performance metrics in expandable section
    _render_performance_metrics()
    
    st.divider()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload MRI Slice (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        key="multitask_upload",
        help="Upload a single MRI slice for multi-task analysis"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        include_gradcam = st.checkbox("Include Grad-CAM visualization", value=True, key="multitask_gradcam")
    
    if uploaded_file is not None:
        # Validate and display image
        is_valid, image = validate_and_display_file(uploaded_file, "MRI slice")
        
        if is_valid:
            # Display original image (centered)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("**📸 Original MRI Slice**")
                st.image(image, width=UIConfig.IMAGE_WIDTH_XLARGE, caption="Input Image")
            
            # Predict button (centered)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                predict_button = st.button(
                    "🎯 Run Multi-Task Prediction",
                    type="primary",
                    key="multitask_predict",
                    use_container_width=True
                )
            
            if predict_button:
                _run_prediction(image, include_gradcam)


def _render_performance_metrics():
    """Render performance metrics in expandable section."""
    with st.expander("📊 Model Performance Metrics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Classification Performance**")
            st.metric("Accuracy", "91.30%", help="Overall classification accuracy")
            st.metric("Sensitivity (Recall)", "97.14%", help="Only 4 missed tumors out of 140!")
            st.metric("ROC-AUC", "0.9184", help="Area under ROC curve")
            st.metric("F1 Score", "95.10%", help="Harmonic mean of precision and recall")
        
        with col2:
            st.markdown("**Segmentation Performance**")
            st.metric("Dice Score", "76.50% ± 13.97%", help="Overlap between prediction and ground truth")
            st.metric("IoU Score", "64.01% ± 18.37%", help="Intersection over Union")
            st.markdown("**Combined Metric**")
            st.metric("Overall Score", "83.90%", help="Weighted combination of both tasks")


def _run_prediction(image: Image.Image, include_gradcam: bool):
    """
    Run multi-task prediction and display results.
    
    Args:
        image: PIL Image object
        include_gradcam: Whether to include Grad-CAM visualization
    """
    with st.spinner("Running unified inference..."):
        # Prepare image
        img_bytes = image_to_bytes(image)
        
        # Call API
        result, error = predict_multitask(img_bytes, include_gradcam)
        
        if error:
            st.error(f"❌ {error}")
            return
        
        # Display results
        st.success("✅ Prediction completed successfully!")
        
        # Processing time
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.metric("⚡ Processing Time", f"{result['processing_time_ms']:.2f} ms")
        
        # Classification results
        _render_classification_results(result)
        
        # Segmentation results
        _render_segmentation_results(result, image)
        
        # Grad-CAM visualization
        if include_gradcam and result.get('gradcam_overlay'):
            _render_gradcam(result)
        
        # Comprehensive comparison
        if result['segmentation_computed'] and result['segmentation'].get('mask_available'):
            _render_comprehensive_view(image, result, include_gradcam)
        
        # Clinical interpretation
        _render_clinical_interpretation(result['classification']['tumor_probability'])
        
        # Medical disclaimer
        _render_medical_disclaimer()


def _render_classification_results(result: dict):
    """Render classification results section."""
    st.markdown("---")
    st.markdown("### 📊 Classification Results")
    
    cls_result = result['classification']
    tumor_prob = cls_result['tumor_probability']
    confidence = cls_result['confidence']
    predicted_label = cls_result['predicted_label']
    
    # Display prediction with color coding
    if predicted_label == "Tumor":
        if tumor_prob >= 0.7:
            st.error(f"⚠️ **{predicted_label} Detected** (High Confidence: {tumor_prob*100:.1f}%)")
        else:
            st.warning(f"⚠️ **{predicted_label} Detected** (Moderate Confidence: {tumor_prob*100:.1f}%)")
    else:
        st.success(f"✅ **{predicted_label}** (Confidence: {confidence*100:.1f}%)")
    
    # Probability metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Tumor Probability", f"{tumor_prob*100:.2f}%")
    with col2:
        st.metric("📈 Confidence", f"{confidence*100:.2f}%")
    with col3:
        st.metric("🏷️ Predicted Class", predicted_label)
    
    # Probability distribution chart
    probs = cls_result['probabilities']
    prob_df = pd.DataFrame({
        'Class': list(probs.keys()),
        'Probability': [v * 100 for v in probs.values()]
    })
    
    fig, ax = plt.subplots(figsize=(UIConfig.CHART_WIDTH_SMALL, UIConfig.CHART_HEIGHT_SMALL))
    ax.barh(
        prob_df['Class'],
        prob_df['Probability'],
        color=[Colors.CHART_NO_TUMOR, Colors.CHART_TUMOR]
    )
    ax.set_xlabel('Probability (%)', fontsize=10)
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Recommendation
    st.info(f"💡 **Recommendation:** {result['recommendation']}")


def _render_segmentation_results(result: dict, original_image: Image.Image):
    """Render segmentation results section."""
    st.markdown("---")
    st.markdown("### 🎨 Segmentation Results")
    
    tumor_prob = result['classification']['tumor_probability']
    
    if result['segmentation_computed']:
        seg_result = result['segmentation']
        
        if seg_result['mask_available']:
            st.success("✅ Segmentation mask generated (tumor probability above threshold)")
            
            # Tumor statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📏 Tumor Area", f"{seg_result['tumor_area_pixels']} px")
            with col2:
                st.metric("📊 Tumor %", f"{seg_result['tumor_percentage']:.2f}%")
            with col3:
                total_pixels = 256 * 256
                st.metric("🖼️ Total Pixels", f"{total_pixels}")
            
            # Visualizations
            st.markdown("**Visualizations**")
            
            # Decode images
            mask_img = base64_to_image(seg_result['mask_base64'])
            prob_map_img = base64_to_image(seg_result['prob_map_base64'])
            overlay_img = base64_to_image(seg_result['overlay_base64'])
            
            # Display in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Binary Mask**")
                st.image(mask_img, width=UIConfig.IMAGE_WIDTH_MEDIUM)
            with col2:
                st.markdown("**Probability Map**")
                st.image(prob_map_img, width=UIConfig.IMAGE_WIDTH_MEDIUM)
            with col3:
                st.markdown("**Overlay**")
                st.image(overlay_img, width=UIConfig.IMAGE_WIDTH_MEDIUM)
        else:
            st.info("ℹ️ Segmentation mask not available")
    else:
        st.info(f"ℹ️ Segmentation not computed (tumor probability {tumor_prob*100:.1f}% is below 30% threshold)")
        st.markdown("""
        **Why?** The model uses conditional segmentation to save computational resources.
        Segmentation is only performed when there's a significant probability of tumor presence.
        """)


def _render_gradcam(result: dict):
    """Render Grad-CAM visualization section."""
    st.markdown("---")
    st.markdown("### 🔍 Grad-CAM Visualization")
    st.markdown("Grad-CAM highlights the regions that the model focuses on for classification.")
    
    gradcam_img = base64_to_image(result['gradcam_overlay'])
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(gradcam_img, width=UIConfig.IMAGE_WIDTH_XLARGE, caption="Grad-CAM Attention Map")


def _render_comprehensive_view(original_image: Image.Image, result: dict, include_gradcam: bool):
    """Render comprehensive comparison view."""
    st.markdown("---")
    st.markdown("### 📸 Comprehensive View")
    
    seg_result = result['segmentation']
    overlay_img = base64_to_image(seg_result['overlay_base64'])
    prob_map_img = base64_to_image(seg_result['prob_map_base64'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Original**")
        st.image(original_image, width=UIConfig.IMAGE_WIDTH_SMALL)
    with col2:
        st.markdown("**Grad-CAM**")
        if include_gradcam and result.get('gradcam_overlay'):
            gradcam_img = base64_to_image(result['gradcam_overlay'])
            st.image(gradcam_img, width=UIConfig.IMAGE_WIDTH_SMALL)
        else:
            st.info("Enable Grad-CAM")
    with col3:
        st.markdown("**Segmentation**")
        st.image(overlay_img, width=UIConfig.IMAGE_WIDTH_SMALL)
    with col4:
        st.markdown("**Probability Map**")
        st.image(prob_map_img, width=UIConfig.IMAGE_WIDTH_SMALL)


def _render_clinical_interpretation(tumor_prob: float):
    """Render clinical interpretation section."""
    st.markdown("---")
    with st.expander("🏥 Clinical Interpretation", expanded=False):
        if tumor_prob >= 0.7:
            st.error(ClinicalGuidelines.HIGH_CONFIDENCE_TUMOR)
        elif tumor_prob >= 0.3:
            st.warning(ClinicalGuidelines.MODERATE_CONFIDENCE_TUMOR)
        else:
            st.success(ClinicalGuidelines.LOW_PROBABILITY_TUMOR)


def _render_medical_disclaimer():
    """Render medical disclaimer section."""
    with st.expander("⚠️ Medical Disclaimer", expanded=False):
        st.warning("""
        - This is a research tool and NOT approved for clinical diagnosis
        - All predictions should be verified by qualified medical professionals
        - Model trained on limited dataset (BraTS 2020 + Kaggle)
        - Performance may vary on different MRI scanners and protocols
        """)


# Export function
__all__ = ['render_multitask_tab']
