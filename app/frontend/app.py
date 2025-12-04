"""
Streamlit frontend for SliceWise MRI Brain Tumor Detection.

This app provides a user-friendly interface for:
- Uploading MRI slices
- Running classification
- Viewing Grad-CAM visualizations
- Exploring predictions
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

# Page configuration
st.set_page_config(
    page_title="SliceWise - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
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
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running and healthy."""
    try:
        response = requests.get(f"{API_URL}/healthz", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data['model_loaded'], data['device']
        return False, "unknown"
    except:
        return False, "unknown"


def get_model_info():
    """Get model information from API."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def classify_image(image_bytes):
    """Send image to API for classification."""
    try:
        files = {"file": ("image.png", image_bytes, "image/png")}
        response = requests.post(
            f"{API_URL}/classify_slice",
            files=files,
            params={"return_probabilities": True},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, f"Request failed: {str(e)}"


def classify_with_gradcam(image_bytes):
    """Send image to API for classification with Grad-CAM."""
    try:
        files = {"file": ("image.png", image_bytes, "image/png")}
        response = requests.post(
            f"{API_URL}/classify_with_gradcam",
            files=files,
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, f"Request failed: {str(e)}"


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">🧠 SliceWise</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Brain Tumor Detection from MRI Slices</div>',
        unsafe_allow_html=True
    )
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
        It is NOT a medical device and should NOT be used for clinical diagnosis.
        Always consult qualified healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check API health
        model_loaded, device = check_api_health()
        
        if model_loaded:
            st.success("✓ API Connected")
            st.info(f"Device: {device}")
        else:
            st.error("✗ API Not Available")
            st.warning("Please start the backend server:\n```bash\npython app/backend/main.py\n```")
        
        st.divider()
        
        # Model info
        if model_loaded:
            st.subheader("📊 Model Info")
            model_info = get_model_info()
            if model_info:
                st.write(f"**Architecture:** {model_info['model_name']}")
                st.write(f"**Classes:** {', '.join(model_info['class_names'])}")
                st.write(f"**Input Size:** {model_info['input_size']}")
        
        st.divider()
        
        # Options
        st.subheader("🎛️ Options")
        show_gradcam = st.checkbox("Show Grad-CAM", value=True, help="Display explainability heatmap")
        show_probabilities = st.checkbox("Show Probabilities", value=True)
        
        st.divider()
        
        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        **SliceWise** uses deep learning to detect brain tumors in MRI slices.
        
        **Features:**
        - Binary classification (Tumor/No Tumor)
        - Grad-CAM visualization
        - Confidence scores
        
        **Model:** EfficientNet-B0
        """)
    
    # Main content
    if not model_loaded:
        st.error("⚠️ Backend API is not available. Please start the server first.")
        st.code("python app/backend/main.py", language="bash")
        return
    
    # File uploader
    st.header("📤 Upload MRI Slice")
    uploaded_file = st.file_uploader(
        "Choose an MRI image...",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a brain MRI slice image"
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Image info
            st.caption(f"Size: {image.size[0]}×{image.size[1]} | Mode: {image.mode}")
        
        # Classify button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Classify with or without Grad-CAM
                if show_gradcam:
                    result, error = classify_with_gradcam(img_byte_arr)
                else:
                    result, error = classify_image(img_byte_arr)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    # Store result in session state
                    st.session_state['result'] = result
                    st.session_state['show_gradcam'] = show_gradcam
                    st.rerun()
    
    # Display results if available
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        st.divider()
        st.header("📊 Results")
        
        # Prediction summary
        predicted_label = result['predicted_label']
        confidence = result['confidence']
        
        # Color based on prediction
        if predicted_label == "Tumor":
            color = "🔴"
            status = "danger"
        else:
            color = "🟢"
            status = "success"
        
        # Display prediction
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.metric(
                label="Prediction",
                value=f"{color} {predicted_label}"
            )
        
        with col2:
            st.metric(
                label="Confidence",
                value=f"{confidence:.1%}"
            )
        
        with col3:
            if confidence > 0.9:
                certainty = "Very High"
            elif confidence > 0.75:
                certainty = "High"
            elif confidence > 0.6:
                certainty = "Moderate"
            else:
                certainty = "Low"
            st.metric(
                label="Certainty",
                value=certainty
            )
        
        # Probabilities
        if show_probabilities and 'probabilities' in result:
            st.subheader("📈 Class Probabilities")
            probs = result['probabilities']
            
            # Create bar chart
            prob_df = pd.DataFrame({
                'Class': list(probs.keys()),
                'Probability': list(probs.values())
            })
            
            fig, ax = plt.subplots(figsize=(8, 3))
            bars = ax.barh(prob_df['Class'], prob_df['Probability'])
            
            # Color bars
            colors = ['#28a745' if label == 'No Tumor' else '#dc3545' 
                     for label in prob_df['Class']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xlabel('Probability')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)
            
            # Add percentage labels
            for i, (cls, prob) in enumerate(zip(prob_df['Class'], prob_df['Probability'])):
                ax.text(prob + 0.02, i, f'{prob:.1%}', va='center')
            
            st.pyplot(fig)
        
        # Grad-CAM visualization
        if st.session_state.get('show_gradcam') and 'gradcam_overlay' in result:
            st.subheader("🔥 Grad-CAM Visualization")
            st.markdown("""
            The heatmap shows which regions of the image the model focused on when making its prediction.
            **Red/Yellow** areas indicate high importance, **Blue** areas indicate low importance.
            """)
            
            # Decode base64 image
            gradcam_bytes = base64.b64decode(result['gradcam_overlay'])
            gradcam_image = Image.open(io.BytesIO(gradcam_bytes))
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                st.image(gradcam_image, caption="Grad-CAM Overlay", use_container_width=True)
        
        # Interpretation
        st.divider()
        st.subheader("💡 Interpretation")
        
        if predicted_label == "Tumor":
            if confidence > 0.8:
                st.warning("""
                The model detected a **high probability of tumor presence** in this MRI slice.
                The Grad-CAM visualization shows the regions that contributed most to this prediction.
                
                ⚠️ **Important:** This is an AI prediction and should NOT be used for medical diagnosis.
                Please consult a qualified radiologist or healthcare professional.
                """)
            else:
                st.info("""
                The model detected a **possible tumor presence**, but with moderate confidence.
                Additional analysis and expert review are recommended.
                """)
        else:
            if confidence > 0.8:
                st.success("""
                The model found **no clear signs of tumor** in this MRI slice.
                However, this does not rule out the presence of abnormalities.
                """)
            else:
                st.info("""
                The model suggests **no tumor**, but with moderate confidence.
                Further examination may be warranted.
                """)
        
        # Download results
        st.divider()
        if st.button("🔄 Analyze Another Image"):
            del st.session_state['result']
            st.rerun()


if __name__ == "__main__":
    main()
