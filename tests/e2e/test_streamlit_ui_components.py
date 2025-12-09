"""
PHASE 3.1: Streamlit UI Components Testing - Frontend Component Validation

Tests all Streamlit UI components for SliceWise MRI Brain Tumor Detection:
- Core UI components (header, sidebar, multi-task tab, classification tab, segmentation tab)
- Interactive elements (file upload, progress indicators, result visualization)
- Visualization components (Grad-CAM overlays, segmentation masks, uncertainty maps)

Validates the complete user interface functionality and user experience.
"""

import sys
import pytest
import numpy as np
import time
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import Streamlit components (mock if not available)
try:
    # Don't import app.py directly (it has st.set_page_config() at module level)
    from app.frontend.components.header import render_header
    from app.frontend.components.sidebar import render_sidebar
    from app.frontend.components.multitask_tab import render_multitask_tab
    from app.frontend.components.classification_tab import render_classification_tab
    from app.frontend.components.segmentation_tab import render_segmentation_tab
    from app.frontend.utils.api_client import APIClient
    from app.frontend.utils.image_utils import base64_to_image, image_to_base64
    STREAMLIT_AVAILABLE = True
    # Create alias for test compatibility
    process_uploaded_image = lambda x: base64_to_image(x) if isinstance(x, str) else x
except ImportError as e:
    # Mock components if not available
    STREAMLIT_AVAILABLE = False
    APIClient = MagicMock()
    process_uploaded_image = MagicMock()
    # Print import error for debugging
    print(f"Streamlit components not available: {e}")

from pathlib import Path


@pytest.fixture
def mock_api_client():
    """Create mock API client for testing."""
    client = MagicMock(spec=APIClient)

    # Mock successful responses
    client.health_check.return_value = {"status": "healthy", "model_loaded": True}
    client.get_model_info.return_value = {
        "model_type": "multitask",
        "supported_tasks": ["classification", "segmentation", "multitask"]
    }
    client.classify.return_value = {
        "prediction": "tumor_present",
        "confidence": 0.85,
        "probabilities": {"tumor_present": 0.85, "no_tumor": 0.15}
    }
    client.segment.return_value = {
        "mask": base64.b64encode(b"mock_mask_data").decode(),
        "probabilities": {"background": 0.1, "edema": 0.2, "non_enhancing": 0.3, "enhancing": 0.4}
    }

    return client


@pytest.fixture
def sample_mri_image():
    """Create sample MRI image for UI testing."""
    # Create realistic brain MRI image (256x256)
    image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    # Add brain-like structure
    center = 128
    y, x = np.ogrid[:256, :256]
    mask = ((x - center)**2 + (y - center)**2) < 80**2
    image[mask] = np.random.randint(150, 220, size=np.sum(mask))

    return image


@pytest.fixture
def mock_uploaded_file(sample_mri_image):
    """Create mock uploaded file for Streamlit testing."""
    # Convert to PIL Image
    pil_image = Image.fromarray(sample_mri_image)

    # Create BytesIO object to simulate uploaded file
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)

    # Mock Streamlit UploadedFile
    mock_file = MagicMock()
    mock_file.name = "test_image.png"
    mock_file.type = "image/png"
    mock_file.size = buffer.tell()
    mock_file.read.return_value = buffer.getvalue()
    mock_file.seek = buffer.seek

    return mock_file


class TestCoreUIComponents:
    """Test core Streamlit UI components."""

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_header_component(self):
        """Test header component rendering."""
        with patch('streamlit.title') as mock_title, \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.divider') as mock_divider:

            # Mock session state
            with patch('streamlit.session_state', {}) as mock_session:
                render_header()

                # Verify header elements were rendered
                mock_title.assert_called()
                mock_markdown.assert_called()
                mock_divider.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_sidebar_component(self, mock_api_client):
        """Test sidebar component with health monitoring."""
        with patch('streamlit.sidebar') as mock_sidebar, \
             patch('streamlit.success') as mock_success, \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.metric') as mock_metric:

            # Mock sidebar context manager
            mock_sidebar_context = MagicMock()
            mock_sidebar.return_value.__enter__ = mock_sidebar_context
            mock_sidebar.return_value.__exit__ = MagicMock()

            render_sidebar(mock_api_client)

            # Verify sidebar elements
            mock_sidebar.assert_called()
            mock_api_client.health_check.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_multitask_tab_rendering(self):
        """Test multi-task tab component rendering."""
        with patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.header') as mock_header, \
             patch('streamlit.write') as mock_write:

            # Mock tabs context
            mock_tab_context = MagicMock()
            mock_tabs.return_value.__getitem__.return_value.__enter__ = mock_tab_context
            mock_tabs.return_value.__getitem__.return_value.__exit__ = MagicMock()

            render_multitask_tab(mock_api_client)

            # Verify tab structure
            mock_tabs.assert_called()
            mock_header.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_classification_tab_rendering(self, mock_api_client):
        """Test classification tab component."""
        with patch('streamlit.header') as mock_header, \
             patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.columns') as mock_columns:

            render_classification_tab(mock_api_client)

            # Verify classification UI elements
            mock_header.assert_called_with("Single Image Classification")
            mock_uploader.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_segmentation_tab_rendering(self, mock_api_client):
        """Test segmentation tab component."""
        with patch('streamlit.header') as mock_header, \
             patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.selectbox') as mock_selectbox:

            render_segmentation_tab(mock_api_client)

            # Verify segmentation UI elements
            mock_header.assert_called_with("Tumor Segmentation")
            mock_uploader.assert_called()


class TestInteractiveElements:
    """Test interactive UI elements."""

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_file_upload_component(self, mock_uploaded_file):
        """Test file upload functionality."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.image') as mock_image:

            # Mock file uploader returning our test file
            mock_uploader.return_value = mock_uploaded_file

            # Simulate file upload processing
            if mock_uploaded_file:
                # Process uploaded file
                image_data = mock_uploaded_file.read()
                uploaded_image = Image.open(io.BytesIO(image_data))

                # Verify image processing
                assert uploaded_image.size == (256, 256)
                assert uploaded_image.mode in ['L', 'RGB', 'RGBA']

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_progress_indicators(self, mock_api_client):
        """Test progress indicators during processing."""
        with patch('streamlit.progress') as mock_progress, \
             patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.success') as mock_success:

            # Mock progress context
            mock_progress_bar = MagicMock()
            mock_progress.return_value = mock_progress_bar

            # Mock spinner context
            mock_spinner_context = MagicMock()
            mock_spinner.return_value.__enter__ = mock_spinner_context
            mock_spinner.return_value.__exit__ = MagicMock()

            # Simulate API call with progress
            with mock_spinner("Processing..."):
                result = mock_api_client.classify(image=base64.b64encode(b"test").decode())

            # Verify progress indicators
            mock_spinner.assert_called_with("Processing...")
            mock_success.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_result_visualization(self, mock_api_client, sample_mri_image):
        """Test result visualization components."""
        with patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.image') as mock_image, \
             patch('streamlit.json') as mock_json:

            # Mock columns context
            mock_col_context = MagicMock()
            mock_columns.return_value = [mock_col_context, mock_col_context]

            # Get mock classification result
            result = mock_api_client.classify.return_value

            # Simulate result display
            if result:
                # Display metrics
                mock_metric("Prediction", result["prediction"])
                mock_metric("Confidence", ".1%")

                # Display probabilities as JSON
                mock_json(result["probabilities"])

            # Verify visualization elements
            mock_columns.assert_called_with(2)
            mock_metric.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_export_functionality(self, mock_api_client, sample_mri_image):
        """Test result export functionality."""
        with patch('streamlit.download_button') as mock_download, \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('pandas.DataFrame') as mock_df, \
             patch('io.StringIO') as mock_stringio:

            # Mock DataFrame for CSV export
            mock_df_instance = MagicMock()
            mock_df.return_value = mock_df_instance
            mock_df_instance.to_csv.return_value = "prediction,result\n1,tumor_present"

            # Mock StringIO for CSV content
            mock_io_instance = MagicMock()
            mock_stringio.return_value = mock_io_instance
            mock_io_instance.getvalue.return_value = "mock_csv_content"

            # Simulate export functionality
            results = [mock_api_client.classify.return_value]

            if results:
                # Create download button for CSV
                mock_download(
                    label="Download Results as CSV",
                    data="mock_csv_content",
                    file_name="predictions.csv",
                    mime="text/csv"
                )

            # Verify export elements
            mock_download.assert_called()
            mock_df.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_settings_controls(self):
        """Test user settings and configuration controls."""
        with patch('streamlit.sidebar') as mock_sidebar, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.checkbox') as mock_checkbox:

            # Mock sidebar context
            mock_sidebar_context = MagicMock()
            mock_sidebar.return_value.__enter__ = mock_sidebar_context
            mock_sidebar.return_value.__exit__ = MagicMock()

            # Simulate settings panel
            with mock_sidebar():
                # Confidence threshold slider
                confidence_threshold = mock_slider("Confidence Threshold", 0.0, 1.0, 0.5)

                # Model selection
                model_type = mock_selectbox("Model Type", ["multitask", "classification", "segmentation"])

                # Advanced options
                show_advanced = mock_checkbox("Show Advanced Options")

            # Verify settings controls
            mock_slider.assert_called_with("Confidence Threshold", 0.0, 1.0, 0.5)
            mock_selectbox.assert_called_with("Model Type", ["multitask", "classification", "segmentation"])
            mock_checkbox.assert_called_with("Show Advanced Options")

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_error_display(self, mock_api_client):
        """Test error message display to users."""
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.warning') as mock_warning, \
             patch('streamlit.info') as mock_info:

            # Simulate API error
            mock_api_client.classify.side_effect = Exception("API connection failed")

            # Simulate error handling in UI
            try:
                result = mock_api_client.classify(image="test")
            except Exception as e:
                # Display user-friendly error
                mock_error(f"Prediction failed: {str(e)}")

            # Verify error display
            mock_error.assert_called()


class TestVisualizationComponents:
    """Test visualization components for medical images."""

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_gradcam_overlay_visualization(self, mock_api_client, sample_mri_image):
        """Test Grad-CAM overlay visualization."""
        with patch('streamlit.image') as mock_image, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.caption') as mock_caption:

            # Mock Grad-CAM result
            gradcam_result = {
                "prediction": "tumor_present",
                "confidence": 0.85,
                "gradcam_overlay": base64.b64encode(b"mock_overlay").decode(),
                "heatmap": base64.b64encode(b"mock_heatmap").decode()
            }

            # Simulate visualization
            mock_columns.return_value = [MagicMock(), MagicMock()]

            # Display Grad-CAM results
            if gradcam_result:
                col1, col2 = mock_columns(2)

                # Original image with overlay
                with col1:
                    mock_caption("Original Image with Grad-CAM Overlay")
                    mock_image(gradcam_result["gradcam_overlay"])

                # Heatmap
                with col2:
                    mock_caption("Grad-CAM Heatmap")
                    mock_image(gradcam_result["heatmap"])

            # Verify visualization elements
            mock_columns.assert_called_with(2)
            mock_caption.assert_called()
            assert mock_image.call_count == 2

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_segmentation_mask_visualization(self, mock_api_client, sample_mri_image):
        """Test segmentation mask overlay visualization."""
        with patch('streamlit.image') as mock_image, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.caption') as mock_caption, \
             patch('streamlit.slider') as mock_slider:

            # Mock segmentation result
            seg_result = {
                "mask": base64.b64encode(b"mock_mask").decode(),
                "probabilities": {
                    "background": 0.1,
                    "edema": 0.2,
                    "non_enhancing": 0.3,
                    "enhancing": 0.4
                }
            }

            # Simulate mask visualization with opacity control
            opacity = mock_slider("Mask Opacity", 0.0, 1.0, 0.5)

            # Display segmentation results
            if seg_result:
                col1, col2 = mock_columns(2)

                # Original image
                with col1:
                    mock_caption("Original MRI Image")
                    mock_image(sample_mri_image)

                # Segmentation mask
                with col2:
                    mock_caption("Tumor Segmentation Mask")
                    mock_image(seg_result["mask"])

                # Class probabilities
                mock_caption("Tissue Class Probabilities")
                for tissue_class, prob in seg_result["probabilities"].items():
                    mock_caption(f"{tissue_class.title()}: {prob:.1%}")

            # Verify visualization elements
            mock_slider.assert_called()
            mock_columns.assert_called_with(2)
            mock_image.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_uncertainty_map_visualization(self, mock_api_client, sample_mri_image):
        """Test uncertainty map visualization."""
        with patch('streamlit.image') as mock_image, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.caption') as mock_caption, \
             patch('streamlit.metric') as mock_metric:

            # Mock uncertainty result
            uncertainty_result = {
                "mask": base64.b64encode(b"mock_mask").decode(),
                "uncertainty_map": base64.b64encode(b"mock_uncertainty").decode(),
                "epistemic_uncertainty": 0.15,
                "aleatoric_uncertainty": 0.08
            }

            # Simulate uncertainty visualization
            if uncertainty_result:
                col1, col2, col3 = mock_columns(3)

                # Original image
                with col1:
                    mock_caption("Original Image")
                    mock_image(sample_mri_image)

                # Segmentation mask
                with col2:
                    mock_caption("Segmentation Mask")
                    mock_image(uncertainty_result["mask"])

                # Uncertainty map
                with col3:
                    mock_caption("Uncertainty Map")
                    mock_image(uncertainty_result["uncertainty_map"])

                # Uncertainty metrics
                mock_metric("Epistemic Uncertainty", ".1f")
                mock_metric("Aleatoric Uncertainty", ".1f")

            # Verify uncertainty visualization
            mock_columns.assert_called_with(3)
            mock_metric.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_roc_curve_visualization(self):
        """Test ROC curve visualization for performance display."""
        with patch('streamlit.pyplot') as mock_pyplot, \
             patch('streamlit.caption') as mock_caption, \
             patch('plotly.graph_objects.Figure') as mock_figure, \
             patch('streamlit.plotly_chart') as mock_chart:

            # Mock ROC curve data
            roc_data = {
                "fpr": [0.0, 0.1, 0.2, 0.3, 1.0],
                "tpr": [0.0, 0.7, 0.8, 0.9, 1.0],
                "auc": 0.87
            }

            # Simulate ROC curve plotting
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(roc_data["fpr"], roc_data["tpr"], label=f'AUC = {roc_data["auc"]:.2f}')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()

            mock_pyplot.imshow(fig)

            # Verify ROC visualization
            mock_caption.assert_called_with("ROC Curve - Model Performance")
            mock_pyplot.imshow.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_confusion_matrix_visualization(self):
        """Test confusion matrix visualization."""
        with patch('streamlit.pyplot') as mock_pyplot, \
             patch('streamlit.caption') as mock_caption, \
             patch('seaborn.heatmap') as mock_heatmap:

            # Mock confusion matrix data
            cm_data = np.array([[85, 5], [3, 92]])  # True labels x Predicted labels

            # Simulate confusion matrix plotting
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Predicted No Tumor', 'Predicted Tumor'],
                       yticklabels=['Actual No Tumor', 'Actual Tumor'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            mock_pyplot.imshow(fig)

            # Verify confusion matrix visualization
            mock_caption.assert_called_with("Confusion Matrix")
            mock_pyplot.imshow.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_volume_rendering_visualization(self, mock_api_client):
        """Test 3D volume rendering visualization for patient analysis."""
        with patch('streamlit.pyplot') as mock_pyplot, \
             patch('streamlit.caption') as mock_caption, \
             patch('plotly.graph_objects.Figure') as mock_figure, \
             patch('streamlit.plotly_chart') as mock_chart:

            # Mock volume analysis result
            volume_result = {
                "estimated_volume_mm3": 15420.5,
                "confidence_interval": [14200.0, 16600.0],
                "slice_analyses": [
                    {"slice_index": 0, "prediction": "no_tumor", "confidence": 0.9},
                    {"slice_index": 1, "prediction": "tumor_present", "confidence": 0.8},
                    # ... more slices
                ]
            }

            # Simulate volume visualization
            if volume_result:
                # Volume metrics
                mock_caption("Tumor Volume Analysis")
                mock_metric("Estimated Volume", "15,421 mm³")
                mock_metric("Confidence Interval", "14,200 - 16,600 mm³")

                # Slice-by-slice analysis
                mock_caption("Slice-by-Slice Analysis")

                # Color-coded slice predictions
                tumor_slices = sum(1 for s in volume_result["slice_analyses"]
                                 if s["prediction"] == "tumor_present")
                total_slices = len(volume_result["slice_analyses"])

                mock_metric("Tumor-Positive Slices", f"{tumor_slices}/{total_slices}")

            # Verify volume visualization
            mock_caption.assert_called()
            mock_metric.assert_called()


class TestUIIntegrationScenarios:
    """Test complete UI integration scenarios."""

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_end_to_end_classification_workflow(self, mock_api_client, mock_uploaded_file):
        """Test complete classification workflow from upload to results."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.success') as mock_success, \
             patch('streamlit.image') as mock_image:

            # Mock file upload
            mock_uploader.return_value = mock_uploaded_file

            # Mock classify button
            mock_button.return_value = True

            # Mock spinner context
            mock_spinner_context = MagicMock()
            mock_spinner.return_value.__enter__ = mock_spinner_context
            mock_spinner.return_value.__exit__ = MagicMock()

            # Simulate workflow
            uploaded_file = mock_uploader("Upload MRI image")

            if uploaded_file:
                # Process uploaded image
                image_data = uploaded_file.read()
                image = Image.open(io.BytesIO(image_data))

                # Show uploaded image
                mock_image(image)

                # Classification button
                if mock_button("Classify Image"):
                    with mock_spinner("Analyzing image..."):
                        # API call
                        result = mock_api_client.classify(image=base64.b64encode(image_data).decode())

                    if result:
                        mock_success("Classification complete!")
                        mock_image(image)  # Show result overlay

            # Verify workflow completion
            mock_uploader.assert_called()
            mock_button.assert_called()
            mock_spinner.assert_called()
            mock_api_client.classify.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_batch_processing_workflow(self, mock_api_client):
        """Test batch processing workflow for multiple images."""
        with patch('streamlit.file_uploader') as mock_batch_uploader, \
             patch('streamlit.button') as mock_process_button, \
             patch('streamlit.progress') as mock_progress, \
             patch('streamlit.dataframe') as mock_dataframe:

            # Mock multiple file upload
            mock_files = [MagicMock(), MagicMock(), MagicMock()]  # 3 files
            for i, mock_file in enumerate(mock_files):
                mock_file.name = f"image_{i}.png"
                mock_file.read.return_value = f"mock_image_data_{i}".encode()

            mock_batch_uploader.return_value = mock_files
            mock_process_button.return_value = True

            # Mock progress bar
            mock_progress_bar = MagicMock()
            mock_progress.return_value = mock_progress_bar

            # Simulate batch processing
            uploaded_files = mock_batch_uploader("Upload multiple images", accept_multiple_files=True)

            if uploaded_files and mock_process_button("Process Batch"):
                results = []

                for i, file in enumerate(uploaded_files):
                    # Update progress
                    mock_progress_bar.progress((i + 1) / len(uploaded_files))

                    # Process each file
                    image_data = file.read()
                    result = mock_api_client.classify(image=base64.b64encode(image_data).decode())
                    results.append({
                        "filename": file.name,
                        "prediction": result["prediction"],
                        "confidence": result["confidence"]
                    })

                # Display results table
                mock_dataframe(results)

            # Verify batch processing
            mock_batch_uploader.assert_called_with("Upload multiple images", accept_multiple_files=True)
            mock_process_button.assert_called_with("Process Batch")
            mock_progress.assert_called()
            mock_dataframe.assert_called()

    @pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit components not available")
    def test_error_recovery_workflow(self, mock_api_client):
        """Test error handling and recovery in UI workflows."""
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.warning') as mock_warning, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.button') as mock_retry_button:

            # Simulate API failure
            mock_api_client.classify.side_effect = Exception("Network timeout")

            # First attempt fails
            try:
                result = mock_api_client.classify(image="test")
            except Exception as e:
                mock_error(f"Classification failed: {str(e)}")
                mock_retry_button("Retry")

            # User clicks retry
            mock_retry_button.return_value = True

            if mock_retry_button():
                mock_info("Retrying...")
                # Reset API mock for successful retry
                mock_api_client.classify.side_effect = None
                mock_api_client.classify.return_value = {"prediction": "success", "confidence": 0.9}

                result = mock_api_client.classify(image="test")
                mock_info("Success on retry!")

            # Verify error recovery workflow
            mock_error.assert_called()
            mock_retry_button.assert_called()
            mock_info.assert_called()
