"""
Segmentation Tab Component for SliceWise Frontend.

This module renders the segmentation tab for tumor localization
with optional uncertainty estimation.
"""

import streamlit as st
from pathlib import Path
import sys
from PIL import Image

# Import settings and utilities
from app.frontend.config.settings import UIConfig, ModelConfig
from app.frontend.utils.api_client import segment_image, segment_with_uncertainty
from app.frontend.utils.image_utils import base64_to_image, image_to_bytes


def render_segmentation_tab():
    """
    Render segmentation tab for tumor localization.
    
    Displays:
    - File upload
    - Options (threshold, min area, uncertainty)
    - Segmentation results
    - Visualizations (mask, probability map, overlay)
    - Uncertainty analysis (optional)
    
    Example:
        >>> from components.segmentation_tab import render_segmentation_tab
        >>> render_segmentation_tab()
    """
    st.header("🎨 Segmentation: Tumor Localization")
    
    st.markdown("""
    Upload an MRI slice to segment tumor regions. 
    The model provides pixel-level predictions with optional uncertainty estimation.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    # Left column: Upload
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an MRI slice...",
            type=["jpg", "jpeg", "png", "bmp"],
            key="seg_upload"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=UIConfig.IMAGE_WIDTH_LARGE)
            st.caption(f"Size: {image.size[0]}×{image.size[1]}")
    
    # Right column: Options
    with col2:
        st.subheader("⚙️ Options")
        threshold = st.slider(
            "Probability Threshold",
            ModelConfig.SEGMENTATION_THRESHOLD_MIN,
            ModelConfig.SEGMENTATION_THRESHOLD_MAX,
            ModelConfig.SEGMENTATION_THRESHOLD_DEFAULT,
            ModelConfig.SEGMENTATION_THRESHOLD_STEP,
            key="seg_thresh"
        )
        min_area = st.number_input(
            "Min Tumor Area (pixels)",
            ModelConfig.MIN_TUMOR_AREA_MIN,
            ModelConfig.MIN_TUMOR_AREA_MAX,
            ModelConfig.MIN_TUMOR_AREA_PIXELS,
            key="seg_min_area"
        )
        use_uncertainty = st.checkbox("Estimate Uncertainty", value=False, key="seg_uncertainty")
        
        if use_uncertainty:
            mc_iterations = st.slider(
                "MC Dropout Iterations",
                ModelConfig.MC_DROPOUT_ITERATIONS_MIN,
                ModelConfig.MC_DROPOUT_ITERATIONS_MAX,
                ModelConfig.MC_DROPOUT_ITERATIONS_DEFAULT,
                key="seg_mc"
            )
            use_tta = st.checkbox(
                "Use Test-Time Augmentation",
                value=ModelConfig.USE_TTA_DEFAULT,
                key="seg_tta"
            )
        
        if uploaded_file:
            if st.button("🎨 Segment", type="primary", use_container_width=True):
                if use_uncertainty:
                    _run_segmentation_with_uncertainty(
                        image, threshold, min_area, mc_iterations, use_tta
                    )
                else:
                    _run_segmentation(image, threshold, min_area)
    
    # Display results
    if 'seg_result' in st.session_state:
        _render_results(st.session_state['seg_result'], st.session_state['seg_image'])


def _run_segmentation(image: Image.Image, threshold: float, min_area: int):
    """
    Run basic segmentation.
    
    Args:
        image: PIL Image object
        threshold: Probability threshold
        min_area: Minimum tumor area in pixels
    """
    with st.spinner("Segmenting..."):
        # Prepare image
        img_bytes = image_to_bytes(image)
        
        # Call API
        result, error = segment_image(
            img_bytes,
            threshold=threshold,
            min_area=min_area,
            apply_postprocessing=True,
            return_overlay=True
        )
        
        if error:
            st.error(f"❌ {error}")
            return
        
        # Store in session state and rerun
        st.session_state['seg_result'] = result
        st.session_state['seg_image'] = image
        st.rerun()


def _run_segmentation_with_uncertainty(
    image: Image.Image,
    threshold: float,
    min_area: int,
    mc_iterations: int,
    use_tta: bool
):
    """
    Run segmentation with uncertainty estimation.
    
    Args:
        image: PIL Image object
        threshold: Probability threshold
        min_area: Minimum tumor area in pixels
        mc_iterations: Number of MC Dropout iterations
        use_tta: Whether to use test-time augmentation
    """
    with st.spinner("Segmenting with uncertainty estimation..."):
        # Prepare image
        img_bytes = image_to_bytes(image)
        
        # Call API
        result, error = segment_with_uncertainty(
            img_bytes,
            threshold=threshold,
            min_area=min_area,
            mc_iterations=mc_iterations,
            use_tta=use_tta
        )
        
        if error:
            st.error(f"❌ {error}")
            return
        
        # Store in session state and rerun
        st.session_state['seg_result'] = result
        st.session_state['seg_image'] = image
        st.rerun()


def _render_results(result: dict, original_image: Image.Image):
    """
    Render segmentation results.
    
    Args:
        result: Segmentation result dictionary
        original_image: Original PIL Image
    """
    st.divider()
    st.header("📊 Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "🔴 Detected" if result['has_tumor'] else "🟢 Not Detected"
        st.metric("Tumor", status)
    
    with col2:
        st.metric("Max Probability", f"{result['tumor_probability']:.1%}")
    
    with col3:
        st.metric("Tumor Area", f"{result['tumor_area_pixels']} px")
    
    with col4:
        st.metric("Components", result['num_components'])
    
    # Visualizations
    st.subheader("🖼️ Visualizations")
    
    cols = st.columns(4)
    
    with cols[0]:
        st.image(original_image, caption="Original", width=UIConfig.IMAGE_WIDTH_LARGE)
    
    with cols[1]:
        mask_img = base64_to_image(result['mask_base64'])
        st.image(mask_img, caption="Binary Mask", width=UIConfig.IMAGE_WIDTH_LARGE)
    
    with cols[2]:
        if result.get('probability_map_base64'):
            prob_img = base64_to_image(result['probability_map_base64'])
            st.image(prob_img, caption="Probability Map", width=UIConfig.IMAGE_WIDTH_LARGE)
    
    with cols[3]:
        if result.get('overlay_base64'):
            overlay_img = base64_to_image(result['overlay_base64'])
            st.image(overlay_img, caption="Overlay", width=UIConfig.IMAGE_WIDTH_LARGE)
    
    # Uncertainty metrics
    if result.get('uncertainty_map_base64'):
        _render_uncertainty_analysis(result)
    
    # Clear button
    if st.button("🔄 Segment Another Image", key="seg_clear"):
        del st.session_state['seg_result']
        del st.session_state['seg_image']
        st.rerun()


def _render_uncertainty_analysis(result: dict):
    """
    Render uncertainty analysis section.
    
    Args:
        result: Segmentation result with uncertainty metrics
    """
    st.subheader("📊 Uncertainty Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uncertainty_img = base64_to_image(result['uncertainty_map_base64'])
        st.image(
            uncertainty_img,
            caption="Epistemic Uncertainty",
            width=UIConfig.IMAGE_WIDTH_LARGE
        )
    
    with col2:
        if result.get('metrics'):
            metrics = result['metrics']
            st.metric(
                "Mean Epistemic Uncertainty",
                f"{metrics.get('mean_epistemic_uncertainty', 0):.4f}"
            )
            st.metric(
                "Max Epistemic Uncertainty",
                f"{metrics.get('max_epistemic_uncertainty', 0):.4f}"
            )
            st.metric(
                "Mean Aleatoric Uncertainty",
                f"{metrics.get('mean_aleatoric_uncertainty', 0):.4f}"
            )


# Export function
__all__ = ['render_segmentation_tab']
