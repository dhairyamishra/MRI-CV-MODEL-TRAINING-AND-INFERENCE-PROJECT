"""
Streamlit frontend for SliceWise MRI Brain Tumor Detection - Phase 6 Complete Demo.

This comprehensive UI provides:
- Classification with Grad-CAM and calibration
- Segmentation with uncertainty estimation
- Batch processing for multiple slices
- Patient-level analysis and volume estimation
- Interactive visualizations and downloadable results

Integrates all components from Phases 0-5.
"""

import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import zipfile
import tempfile

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="SliceWise v2 - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = "http://localhost:8000"

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .version-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #000000;
        border-left: 4px solid #000000;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Utility Functions
# ============================================================================

def check_api_health():
    """Check if API is running and healthy."""
    try:
        response = requests.get(f"{API_URL}/healthz", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data
        return None
    except:
        return None


def get_model_info():
    """Get model information from API."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def base64_to_image(base64_str):
    """Convert base64 string to PIL Image."""
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes))


def image_to_bytes(image):
    """Convert PIL Image to bytes."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


# ============================================================================
# Header and Sidebar
# ============================================================================

def render_header():
    """Render application header."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">🧠 SliceWise v2</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Comprehensive AI-Powered Brain Tumor Detection</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div style="text-align: center;"><span class="version-badge">Phase 6 Complete</span></div>',
            unsafe_allow_html=True
        )
    
    # Medical disclaimer
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
        It is NOT a medical device and should NOT be used for clinical diagnosis.
        Always consult qualified healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with API status and settings."""
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Check API health
        health = check_api_health()
        
        if health and health['status'] in ['healthy', 'no_models_loaded']:
            st.success("✓ API Connected")
            
            # Show model status
            col1, col2 = st.columns(2)
            with col1:
                if health['classifier_loaded']:
                    st.metric("Classifier", "✓", delta="Ready")
                else:
                    st.metric("Classifier", "✗", delta="Not loaded")
            with col2:
                if health['segmentation_loaded']:
                    st.metric("Segmentation", "✓", delta="Ready")
                else:
                    st.metric("Segmentation", "✗", delta="Not loaded")
            
            # Additional features
            if health['calibration_loaded']:
                st.info("🎯 Calibration: Enabled")
            
            st.caption(f"Device: {health['device']}")
            st.caption(f"Updated: {datetime.fromisoformat(health['timestamp']).strftime('%H:%M:%S')}")
        else:
            st.error("✗ API Not Available")
            st.warning("Please start the backend server:\n```bash\npython app/backend/main_v2.py\n```")
            return False
        
        st.divider()
        
        # Model information
        if health and (health['classifier_loaded'] or health['segmentation_loaded']):
            st.subheader("📊 Model Info")
            model_info = get_model_info()
            if model_info:
                if model_info.get('classifier'):
                    with st.expander("Classifier Details"):
                        cls_info = model_info['classifier']
                        st.write(f"**Architecture:** {cls_info.get('architecture', 'N/A')}")
                        st.write(f"**Classes:** {', '.join(cls_info.get('class_names', []))}")
                        st.write(f"**Calibrated:** {'Yes' if cls_info.get('calibrated') else 'No'}")
                
                if model_info.get('segmentation'):
                    with st.expander("Segmentation Details"):
                        seg_info = model_info['segmentation']
                        st.write(f"**Architecture:** {seg_info.get('architecture', 'N/A')}")
                        st.write(f"**Parameters:** {seg_info.get('parameters', 'N/A')}")
                        st.write(f"**Uncertainty:** {'Yes' if seg_info.get('uncertainty_estimation') else 'No'}")
                
                # Features
                features = model_info.get('features', [])
                if features:
                    st.write("**Available Features:**")
                    for feature in features:
                        st.write(f"- {feature.replace('_', ' ').title()}")
        
        st.divider()
        
        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        **SliceWise v2** integrates all Phase 0-5 components:
        
        - 🔍 Classification with Grad-CAM
        - 🎯 Probability calibration
        - 🎨 Tumor segmentation
        - 📊 Uncertainty estimation
        - 👤 Patient-level analysis
        - 📦 Batch processing
        
        **Total:** ~12,000+ lines of code
        """)
        
        return True


# ============================================================================
# Tab 1: Classification
# ============================================================================

def render_classification_tab():
    """Render classification tab."""
    st.header("🔍 Classification: Tumor Detection")
    
    st.markdown("""
    Upload an MRI slice to classify whether it contains a tumor. 
    The model provides predictions with confidence scores and optional Grad-CAM visualization.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an MRI slice...",
            type=["jpg", "jpeg", "png", "bmp"],
            key="cls_upload"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            st.caption(f"Size: {image.size[0]}×{image.size[1]} | Mode: {image.mode}")
    
    with col2:
        st.subheader("⚙️ Options")
        show_gradcam = st.checkbox("Generate Grad-CAM", value=True, key="cls_gradcam")
        show_calibration = st.checkbox("Show Calibrated Probabilities", value=True, key="cls_calib")
        
        if uploaded_file:
            if st.button("🔍 Classify", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        img_bytes = image_to_bytes(image)
                        files = {"file": ("image.png", img_bytes, "image/png")}
                        params = {"return_gradcam": show_gradcam}
                        
                        response = requests.post(
                            f"{API_URL}/classify",
                            files=files,
                            params=params,
                            timeout=15
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['cls_result'] = result
                            st.rerun()
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Request failed: {str(e)}")
    
    # Display results
    if 'cls_result' in st.session_state:
        st.divider()
        result = st.session_state['cls_result']
        
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
            certainty = "Very High" if confidence > 0.9 else "High" if confidence > 0.75 else "Moderate" if confidence > 0.6 else "Low"
            st.metric("Certainty", certainty)
        
        # Probabilities
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Raw Probabilities")
            probs = result['probabilities']
            prob_df = pd.DataFrame({
                'Class': list(probs.keys()),
                'Probability': list(probs.values())
            })
            
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.barh(prob_df['Class'], prob_df['Probability'])
            colors = ['#28a745' if label == 'No Tumor' else '#dc3545' for label in prob_df['Class']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            ax.set_xlabel('Probability')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)
            for i, (cls, prob) in enumerate(zip(prob_df['Class'], prob_df['Probability'])):
                ax.text(prob + 0.02, i, f'{prob:.1%}', va='center')
            st.pyplot(fig)
        
        with col2:
            if show_calibration and result.get('calibrated_probabilities'):
                st.subheader("🎯 Calibrated Probabilities")
                cal_probs = result['calibrated_probabilities']
                cal_df = pd.DataFrame({
                    'Class': list(cal_probs.keys()),
                    'Probability': list(cal_probs.values())
                })
                
                fig, ax = plt.subplots(figsize=(6, 3))
                bars = ax.barh(cal_df['Class'], cal_df['Probability'])
                colors = ['#28a745' if label == 'No Tumor' else '#dc3545' for label in cal_df['Class']]
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                ax.set_xlabel('Probability')
                ax.set_xlim(0, 1)
                ax.grid(axis='x', alpha=0.3)
                for i, (cls, prob) in enumerate(zip(cal_df['Class'], cal_df['Probability'])):
                    ax.text(prob + 0.02, i, f'{prob:.1%}', va='center')
                st.pyplot(fig)
        
        # Grad-CAM
        if show_gradcam and result.get('gradcam_overlay'):
            st.subheader("🔥 Grad-CAM Visualization")
            st.markdown("""
            The heatmap shows which regions the model focused on when making its prediction.
            **Red/Yellow** = high importance, **Blue** = low importance.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", width=400)
            with col2:
                gradcam_img = base64_to_image(result['gradcam_overlay'])
                st.image(gradcam_img, caption="Grad-CAM Overlay", width=400)
        
        # Clear button
        if st.button("🔄 Analyze Another Image", key="cls_clear"):
            del st.session_state['cls_result']
            st.rerun()


# ============================================================================
# Tab 2: Segmentation
# ============================================================================

def render_segmentation_tab():
    """Render segmentation tab."""
    st.header("🎨 Segmentation: Tumor Localization")
    
    st.markdown("""
    Upload an MRI slice to segment tumor regions. 
    The model provides pixel-level predictions with optional uncertainty estimation.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an MRI slice...",
            type=["jpg", "jpeg", "png", "bmp"],
            key="seg_upload"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            st.caption(f"Size: {image.size[0]}×{image.size[1]}")
    
    with col2:
        st.subheader("⚙️ Options")
        threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.5, 0.05, key="seg_thresh")
        min_area = st.number_input("Min Tumor Area (pixels)", 0, 500, 50, key="seg_min_area")
        use_uncertainty = st.checkbox("Estimate Uncertainty", value=False, key="seg_uncertainty")
        
        if use_uncertainty:
            mc_iterations = st.slider("MC Dropout Iterations", 1, 50, 10, key="seg_mc")
            use_tta = st.checkbox("Use Test-Time Augmentation", value=True, key="seg_tta")
        
        if uploaded_file:
            if st.button("🎨 Segment", type="primary", use_container_width=True):
                with st.spinner("Segmenting..."):
                    try:
                        img_bytes = image_to_bytes(image)
                        files = {"file": ("image.png", img_bytes, "image/png")}
                        
                        if use_uncertainty:
                            params = {
                                "threshold": threshold,
                                "min_area": min_area,
                                "mc_iterations": mc_iterations,
                                "use_tta": use_tta
                            }
                            response = requests.post(
                                f"{API_URL}/segment/uncertainty",
                                files=files,
                                params=params,
                                timeout=30
                            )
                        else:
                            params = {
                                "threshold": threshold,
                                "min_area": min_area,
                                "apply_postprocessing": True,
                                "return_overlay": True
                            }
                            response = requests.post(
                                f"{API_URL}/segment",
                                files=files,
                                params=params,
                                timeout=15
                            )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['seg_result'] = result
                            st.session_state['seg_image'] = image
                            st.rerun()
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Request failed: {str(e)}")
    
    # Display results
    if 'seg_result' in st.session_state:
        st.divider()
        result = st.session_state['seg_result']
        original_image = st.session_state['seg_image']
        
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
            st.image(original_image, caption="Original", width=300)
        
        with cols[1]:
            mask_img = base64_to_image(result['mask_base64'])
            st.image(mask_img, caption="Binary Mask", width=300)
        
        with cols[2]:
            if result.get('probability_map_base64'):
                prob_img = base64_to_image(result['probability_map_base64'])
                st.image(prob_img, caption="Probability Map", width=300)
        
        with cols[3]:
            if result.get('overlay_base64'):
                overlay_img = base64_to_image(result['overlay_base64'])
                st.image(overlay_img, caption="Overlay", width=300)
        
        # Uncertainty metrics
        if result.get('uncertainty_map_base64'):
            st.subheader("📊 Uncertainty Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                uncertainty_img = base64_to_image(result['uncertainty_map_base64'])
                st.image(uncertainty_img, caption="Epistemic Uncertainty", width=300)
            
            with col2:
                if result.get('metrics'):
                    metrics = result['metrics']
                    st.metric("Mean Epistemic Uncertainty", f"{metrics.get('mean_epistemic_uncertainty', 0):.4f}")
                    st.metric("Max Epistemic Uncertainty", f"{metrics.get('max_epistemic_uncertainty', 0):.4f}")
                    st.metric("Mean Aleatoric Uncertainty", f"{metrics.get('mean_aleatoric_uncertainty', 0):.4f}")
        
        # Clear button
        if st.button("🔄 Segment Another Image", key="seg_clear"):
            del st.session_state['seg_result']
            del st.session_state['seg_image']
            st.rerun()


# ============================================================================
# Tab 3: Batch Processing
# ============================================================================

def render_batch_tab():
    """Render batch processing tab."""
    st.header("📦 Batch Processing")
    
    st.markdown("""
    Upload multiple MRI slices for batch classification or segmentation.
    Results can be downloaded as CSV for further analysis.
    """)
    
    mode = st.radio("Processing Mode", ["Classification", "Segmentation"], horizontal=True)
    
    uploaded_files = st.file_uploader(
        "Upload multiple MRI slices (max 100)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files uploaded")
        
        # Show preview
        with st.expander("Preview Images"):
            cols = st.columns(min(5, len(uploaded_files)))
            for idx, (col, file) in enumerate(zip(cols, uploaded_files[:5])):
                with col:
                    image = Image.open(file)
                    st.image(image, caption=file.name, width=300)
            if len(uploaded_files) > 5:
                st.caption(f"... and {len(uploaded_files) - 5} more")
        
        # Options
        col1, col2 = st.columns(2)
        
        with col1:
            if mode == "Segmentation":
                threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05, key="batch_thresh")
                min_area = st.number_input("Min Area", 0, 500, 50, key="batch_min_area")
        
        with col2:
            if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(uploaded_files)} images..."):
                    try:
                        files = []
                        for file in uploaded_files:
                            file.seek(0)  # Reset file pointer
                            files.append(("files", (file.name, file.read(), "image/png")))
                        
                        if mode == "Classification":
                            response = requests.post(
                                f"{API_URL}/classify/batch",
                                files=files,
                                timeout=60
                            )
                        else:
                            params = {"threshold": threshold, "min_area": min_area}
                            response = requests.post(
                                f"{API_URL}/segment/batch",
                                files=files,
                                params=params,
                                timeout=60
                            )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['batch_result'] = result
                            st.session_state['batch_mode'] = mode
                            st.rerun()
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Request failed: {str(e)}")
    
    # Display results
    if 'batch_result' in st.session_state:
        st.divider()
        result = st.session_state['batch_result']
        mode = st.session_state['batch_mode']
        
        st.header("📊 Batch Results")
        
        # Summary
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
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔴 Tumor Detected", summary['tumor_detected'])
            with col2:
                st.metric("🟢 No Tumor", summary['no_tumor'])
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Slices with Tumor", summary['slices_with_tumor'])
            with col2:
                st.metric("Slices without Tumor", summary['slices_without_tumor'])
            with col3:
                st.metric("Total Tumor Area", f"{summary['total_tumor_area_pixels']} px")
        
        # Detailed results
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


# ============================================================================
# Tab 4: Patient Analysis
# ============================================================================

def render_patient_tab():
    """Render patient-level analysis tab."""
    st.header("👤 Patient-Level Analysis")
    
    st.markdown("""
    Upload a stack of MRI slices from a single patient for comprehensive analysis.
    The system will provide patient-level tumor detection and volume estimation.
    """)
    
    patient_id = st.text_input("Patient ID", value="PATIENT_001", key="patient_id")
    
    uploaded_files = st.file_uploader(
        "Upload patient MRI stack (multiple slices)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="patient_upload"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} slices uploaded for patient: {patient_id}")
        
        # Show preview
        with st.expander("Preview Slices"):
            cols = st.columns(min(5, len(uploaded_files)))
            for idx, (col, file) in enumerate(zip(cols, uploaded_files[:5])):
                with col:
                    image = Image.open(file)
                    st.image(image, caption=f"Slice {idx}", width=300)
            if len(uploaded_files) > 5:
                st.caption(f"... and {len(uploaded_files) - 5} more slices")
        
        # Options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05, key="patient_thresh")
        
        with col2:
            min_area = st.number_input("Min Area", 0, 500, 50, key="patient_min_area")
        
        with col3:
            slice_thickness = st.number_input("Slice Thickness (mm)", 0.1, 10.0, 1.0, 0.1, key="patient_thickness")
        
        if st.button("🔬 Analyze Patient", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {len(uploaded_files)} slices..."):
                try:
                    files = []
                    for file in uploaded_files:
                        file.seek(0)
                        files.append(("files", (file.name, file.read(), "image/png")))
                    
                    data = {
                        "patient_id": patient_id,
                        "threshold": threshold,
                        "min_area": min_area,
                        "slice_thickness_mm": slice_thickness
                    }
                    
                    response = requests.post(
                        f"{API_URL}/patient/analyze_stack",
                        files=files,
                        data=data,
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['patient_result'] = result
                        st.rerun()
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
    
    # Display results
    if 'patient_result' in st.session_state:
        st.divider()
        result = st.session_state['patient_result']
        
        st.header("📊 Patient Analysis Results")
        
        # Patient summary
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
        
        # Patient-level metrics
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
        
        # Slice-by-slice results
        st.subheader("🔬 Slice-by-Slice Analysis")
        
        slice_df = pd.DataFrame(result['slice_predictions'])
        
        # Add color coding
        def highlight_tumor(row):
            if row['has_tumor']:
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
        
        styled_df = slice_df.style.apply(highlight_tumor, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Visualization
        st.subheader("📊 Tumor Distribution")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Tumor area per slice
        ax1.bar(slice_df['slice_index'], slice_df['tumor_area_pixels'], color='#dc3545', alpha=0.7)
        ax1.set_xlabel('Slice Index')
        ax1.set_ylabel('Tumor Area (pixels)')
        ax1.set_title('Tumor Area per Slice')
        ax1.grid(axis='y', alpha=0.3)
        
        # Probability per slice
        ax2.plot(slice_df['slice_index'], slice_df['max_probability'], marker='o', color='#1f77b4')
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax2.set_xlabel('Slice Index')
        ax2.set_ylabel('Max Probability')
        ax2.set_title('Tumor Probability per Slice')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download results
        col1, col2 = st.columns(2)
        
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
        
        # Clear button
        if st.button("🔄 Analyze Another Patient", key="patient_clear"):
            del st.session_state['patient_result']
            st.rerun()


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application."""
    render_header()
    
    # Render sidebar and check API status
    api_available = render_sidebar()
    
    if not api_available:
        st.error("⚠️ Backend API is not available. Please start the server first.")
        st.code("python app/backend/main_v2.py", language="bash")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Classification",
        "🎨 Segmentation",
        "📦 Batch Processing",
        "👤 Patient Analysis"
    ])
    
    with tab1:
        render_classification_tab()
    
    with tab2:
        render_segmentation_tab()
    
    with tab3:
        render_batch_tab()
    
    with tab4:
        render_patient_tab()
    
    # Footer
    st.divider()
    st.caption("SliceWise v2 - Phase 6 Complete | Built with ❤️ using PyTorch, FastAPI, and Streamlit")


if __name__ == "__main__":
    main()
