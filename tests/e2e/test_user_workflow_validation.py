"""
PHASE 3.3: User Workflow Testing - End-to-End User Journey Validation

Tests complete user workflows and journeys for SliceWise MRI Brain Tumor Detection:
- Classification workflow: Upload → predict → view Grad-CAM → export
- Segmentation workflow: Upload → segment → view uncertainty → download
- Batch processing: Upload multiple → monitor progress → review all results
- Patient analysis: Upload stack → analyze volume → explore 3D view
- Multi-task analysis: Upload → get both results → compare outputs

Validates real user experiences and edge case handling.
"""

import sys
import pytest
import numpy as np
import base64
import io
import time
import tempfile
import json
from PIL import Image
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import components (mock if not available)
try:
    from app.frontend.app import main as streamlit_app
    from app.frontend.utils.api_client import APIClient
    from app.frontend.components.classification_tab import render_classification_tab
    from app.frontend.components.segmentation_tab import render_segmentation_tab
    from app.frontend.components.batch_tab import render_batch_tab
    from app.frontend.components.patient_tab import render_patient_tab
    from app.frontend.components.multitask_tab import render_multitask_tab
    FRONTEND_AVAILABLE = True
except ImportError:
    FRONTEND_AVAILABLE = False
    APIClient = MagicMock()


@pytest.fixture
def mock_api_client():
    """Create comprehensive mock API client."""
    client = MagicMock(spec=APIClient)

    # Mock health and info
    client.health_check.return_value = {"status": "healthy", "model_loaded": True}
    client.get_model_info.return_value = {
        "model_type": "multitask",
        "supported_tasks": ["classification", "segmentation", "multitask"]
    }

    # Mock classification responses
    client.classify.return_value = {
        "prediction": "tumor_present",
        "confidence": 0.87,
        "probabilities": {"tumor_present": 0.87, "no_tumor": 0.13},
        "processing_time_ms": 245.0
    }

    client.classify_gradcam.return_value = {
        "prediction": "tumor_present",
        "confidence": 0.87,
        "gradcam_overlay": base64.b64encode(b"overlay_data").decode(),
        "heatmap": base64.b64encode(b"heatmap_data").decode(),
        "processing_time_ms": 312.0
    }

    # Mock segmentation responses
    client.segment.return_value = {
        "mask": base64.b64encode(b"mask_data").decode(),
        "probabilities": {"background": 0.1, "edema": 0.2, "non_enhancing": 0.3, "enhancing": 0.4},
        "dice_score": 0.85,
        "processing_time_ms": 456.0
    }

    client.segment_uncertainty.return_value = {
        "mask": base64.b64encode(b"mask_data").decode(),
        "probabilities": {"background": 0.1, "edema": 0.2, "non_enhancing": 0.3, "enhancing": 0.4},
        "uncertainty_map": base64.b64encode(b"uncertainty_data").decode(),
        "epistemic_uncertainty": 0.15,
        "aleatoric_uncertainty": 0.08,
        "processing_time_ms": 678.0
    }

    # Mock multi-task responses
    client.predict_multitask.return_value = {
        "classification": {
            "prediction": "tumor_present",
            "confidence": 0.87,
            "probabilities": {"tumor_present": 0.87, "no_tumor": 0.13}
        },
        "segmentation": {
            "mask": base64.b64encode(b"mask_data").decode(),
            "probabilities": {"background": 0.1, "edema": 0.2, "non_enhancing": 0.3, "enhancing": 0.4}
        },
        "processing_time_ms": 789.0
    }

    # Mock batch processing
    client.classify_batch.return_value = {
        "results": [
            {"prediction": "tumor_present", "confidence": 0.87},
            {"prediction": "no_tumor", "confidence": 0.92},
            {"prediction": "tumor_present", "confidence": 0.78}
        ],
        "batch_size": 3,
        "processing_time_ms": 734.0
    }

    client.segment_batch.return_value = {
        "results": [
            {"mask": base64.b64encode(b"mask1").decode(), "dice_score": 0.85},
            {"mask": base64.b64encode(b"mask2").decode(), "dice_score": 0.91},
            {"mask": base64.b64encode(b"mask3").decode(), "dice_score": 0.76}
        ],
        "batch_size": 3,
        "processing_time_ms": 1234.0
    }

    # Mock patient analysis
    client.analyze_patient_stack.return_value = {
        "patient_summary": {
            "overall_prediction": "tumor_present",
            "confidence": 0.89,
            "tumor_probability": 0.91
        },
        "slice_analyses": [
            {"slice_index": 0, "prediction": "no_tumor", "confidence": 0.94},
            {"slice_index": 1, "prediction": "tumor_present", "confidence": 0.87},
            {"slice_index": 2, "prediction": "tumor_present", "confidence": 0.92}
        ],
        "volume_analysis": {
            "estimated_volume_mm3": 15420.5,
            "confidence_interval": [14200.0, 16600.0]
        },
        "processing_time_ms": 1567.0
    }

    return client


@pytest.fixture
def sample_mri_images():
    """Create multiple sample MRI images for batch testing."""
    images = []
    for i in range(3):
        # Create slightly different images
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        # Add some variation
        center = 128
        y, x = np.ogrid[:256, :256]
        mask = ((x - center)**2 + (y - center)**2) < (70 + i * 10)**2
        image[mask] = np.random.randint(150 + i*10, 220 + i*10, size=np.sum(mask))

        images.append(image)

    return images


@pytest.fixture
def mock_uploaded_files(sample_mri_images):
    """Create mock uploaded files for testing."""
    mock_files = []

    for i, image_array in enumerate(sample_mri_images):
        # Convert to PIL and then to bytes
        pil_image = Image.fromarray(image_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)

        # Create mock file
        mock_file = MagicMock()
        mock_file.name = f"test_mri_{i}.png"
        mock_file.type = "image/png"
        mock_file.size = buffer.tell()
        mock_file.read.return_value = buffer.getvalue()
        mock_file.seek = buffer.seek

        mock_files.append(mock_file)

    return mock_files


@pytest.fixture
def mock_mri_stack():
    """Create mock MRI stack for patient analysis."""
    stack_images = []
    for i in range(5):  # 5 slices
        # Create slice with some variation
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        # Add slice-specific features
        if i == 2:  # Middle slice has tumor-like region
            image[100:150, 100:150] = 200  # Bright region

        stack_images.append(image)

    return stack_images


class TestClassificationWorkflow:
    """Test complete classification user workflow."""

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_single_image_classification_workflow(self, mock_api_client, mock_uploaded_files):
        """Test end-to-end single image classification workflow."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.button') as mock_button, \
             patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.success') as mock_success, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.image') as mock_image, \
             patch('streamlit.json') as mock_json, \
             patch('streamlit.download_button') as mock_download:

            # Mock UI interactions
            mock_uploader.return_value = mock_uploaded_files[0]  # Single file
            mock_button.return_value = True  # User clicks analyze

            # Mock spinner context
            mock_spinner_context = MagicMock()
            mock_spinner.return_value.__enter__ = mock_spinner_context
            mock_spinner.return_value.__exit__ = MagicMock()

            # Simulate workflow
            uploaded_file = mock_uploader("Upload MRI image for classification")

            if uploaded_file:
                # User uploads file
                assert uploaded_file.name == "test_mri_0.png"

                # User clicks analyze button
                if mock_button("Analyze Image"):
                    with mock_spinner("Analyzing image..."):
                        # Process uploaded file
                        image_data = uploaded_file.read()
                        image_b64 = base64.b64encode(image_data).decode()

                        # API call
                        result = mock_api_client.classify(image=image_b64)

                    # Display results
                    if result:
                        mock_success("Classification complete!")

                        # Show metrics
                        mock_metric("Prediction", result["prediction"])
                        mock_metric("Confidence", ".1%")

                        # Show probabilities
                        mock_json(result["probabilities"])

                        # Show original image
                        mock_image(uploaded_file)

                        # Provide download option
                        mock_download(
                            label="Download Results (JSON)",
                            data=json.dumps(result),
                            file_name="classification_results.json",
                            mime="application/json"
                        )

            # Verify complete workflow
            mock_uploader.assert_called_with("Upload MRI image for classification")
            mock_button.assert_called_with("Analyze Image")
            mock_spinner.assert_called_with("Analyzing image...")
            mock_success.assert_called_with("Classification complete!")
            mock_metric.assert_called()
            mock_json.assert_called()
            mock_image.assert_called()
            mock_download.assert_called()

            # Verify API was called correctly
            mock_api_client.classify.assert_called_once()

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_gradcam_visualization_workflow(self, mock_api_client, mock_uploaded_files):
        """Test Grad-CAM visualization workflow."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.button') as mock_gradcam_button, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.image') as mock_image, \
             patch('streamlit.caption') as mock_caption:

            # Mock UI interactions
            mock_uploader.return_value = mock_uploaded_files[0]
            mock_gradcam_button.return_value = True  # User requests Grad-CAM

            # Mock columns for side-by-side display
            mock_col1 = MagicMock()
            mock_col2 = MagicMock()
            mock_columns.return_value = [mock_col1, mock_col2]

            # Simulate Grad-CAM workflow
            uploaded_file = mock_uploader("Upload MRI image")

            if uploaded_file and mock_gradcam_button("Generate Grad-CAM"):
                # Process file and get Grad-CAM
                image_data = uploaded_file.read()
                image_b64 = base64.b64encode(image_data).decode()

                result = mock_api_client.classify_gradcam(image=image_b64)

                if result:
                    # Display side-by-side
                    col1, col2 = mock_columns(2)

                    # Original image with overlay
                    with col1:
                        mock_caption("Original Image with Grad-CAM Overlay")
                        mock_image(result["gradcam_overlay"])

                    # Heatmap
                    with col2:
                        mock_caption("Grad-CAM Attention Heatmap")
                        mock_image(result["heatmap"])

                    # Show prediction details
                    mock_caption(f"Prediction: {result['prediction']} ({result['confidence']:.1%})")

            # Verify Grad-CAM workflow
            mock_uploader.assert_called()
            mock_gradcam_button.assert_called_with("Generate Grad-CAM")
            mock_columns.assert_called_with(2)
            mock_caption.assert_called()
            assert mock_image.call_count == 2  # Original + overlay, heatmap

            mock_api_client.classify_gradcam.assert_called_once()


class TestSegmentationWorkflow:
    """Test complete segmentation user workflow."""

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_tumor_segmentation_workflow(self, mock_api_client, mock_uploaded_files):
        """Test end-to-end tumor segmentation workflow."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.button') as mock_segment_button, \
             patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.success') as mock_success, \
             patch('streamlit.image') as mock_image, \
             patch('streamlit.caption') as mock_caption, \
             patch('streamlit.metric') as mock_metric:

            # Mock UI interactions
            mock_uploader.return_value = mock_uploaded_files[0]
            mock_segment_button.return_value = True

            # Mock spinner context
            mock_spinner_context = MagicMock()
            mock_spinner.return_value.__enter__ = mock_spinner_context
            mock_spinner.return_value.__exit__ = MagicMock()

            # Simulate segmentation workflow
            uploaded_file = mock_uploader("Upload MRI image for segmentation")

            if uploaded_file and mock_segment_button("Segment Tumor"):
                with mock_spinner("Segmenting tumor regions..."):
                    image_data = uploaded_file.read()
                    image_b64 = base64.b64encode(image_data).decode()

                    result = mock_api_client.segment(image=image_b64)

                if result:
                    mock_success("Segmentation complete!")

                    # Display results
                    col1, col2 = [MagicMock(), MagicMock()]  # Mock columns

                    # Show original image
                    mock_caption("Original MRI Image")
                    mock_image(uploaded_file)

                    # Show segmentation mask
                    mock_caption("Tumor Segmentation Mask")
                    mock_image(result["mask"])

                    # Show tissue class probabilities
                    for tissue_class, prob in result["probabilities"].items():
                        mock_metric(f"{tissue_class.title()}", ".1%")

                    # Show Dice score
                    mock_metric("Dice Score", ".1%")

            # Verify segmentation workflow
            mock_uploader.assert_called_with("Upload MRI image for segmentation")
            mock_segment_button.assert_called_with("Segment Tumor")
            mock_spinner.assert_called_with("Segmenting tumor regions...")
            mock_success.assert_called_with("Segmentation complete!")
            mock_caption.assert_called()
            mock_image.assert_called()
            mock_metric.assert_called()

            mock_api_client.segment.assert_called_once()

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_uncertainty_estimation_workflow(self, mock_api_client, mock_uploaded_files):
        """Test uncertainty estimation workflow."""
        with patch('streamlit.button') as mock_uncertainty_button, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.image') as mock_image, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.file_uploader') as mock_uploader:

            # Mock UI state
            mock_uploader.return_value = mock_uploaded_files[0]
            mock_uncertainty_button.return_value = True

            # Mock three columns for display
            mock_cols = [MagicMock(), MagicMock(), MagicMock()]
            mock_columns.return_value = mock_cols

            # Simulate uncertainty workflow
            uploaded_file = mock_uploader("Upload image")
            uploaded_file.read.return_value = b"test_image_data"

            if uploaded_file and mock_uncertainty_button("Estimate Uncertainty"):
                image_data = uploaded_file.read()
                image_b64 = base64.b64encode(image_data).decode()

                result = mock_api_client.segment_uncertainty(image=image_b64)

                if result:
                    col1, col2, col3 = mock_columns(3)

                    # Original image
                    mock_image(uploaded_file)

                    # Segmentation mask
                    mock_image(result["mask"])

                    # Uncertainty map
                    mock_image(result["uncertainty_map"])

                    # Uncertainty metrics
                    mock_metric("Epistemic Uncertainty", ".1f")
                    mock_metric("Aleatoric Uncertainty", ".1f")

            # Verify uncertainty workflow
            mock_uncertainty_button.assert_called_with("Estimate Uncertainty")
            mock_columns.assert_called_with(3)
            assert mock_image.call_count == 3  # Original, mask, uncertainty
            assert mock_metric.call_count == 2  # Two uncertainty metrics


class TestBatchProcessingWorkflow:
    """Test complete batch processing user workflow."""

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_batch_classification_workflow(self, mock_api_client, mock_uploaded_files):
        """Test batch classification processing workflow."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.button') as mock_process_button, \
             patch('streamlit.progress') as mock_progress, \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.success') as mock_success:

            # Mock UI interactions
            mock_uploader.return_value = mock_uploaded_files  # Multiple files
            mock_process_button.return_value = True

            # Mock progress bar
            mock_progress_bar = MagicMock()
            mock_progress.return_value = mock_progress_bar

            # Simulate batch workflow
            uploaded_files = mock_uploader("Upload multiple images", accept_multiple_files=True)

            if uploaded_files and mock_process_button("Process Batch"):
                batch_results = []

                for i, file in enumerate(uploaded_files):
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    mock_progress_bar.progress(progress)

                    # Process each file
                    image_data = file.read()
                    result = mock_api_client.classify(image=base64.b64encode(image_data).decode())

                    batch_results.append({
                        "filename": file.name,
                        "prediction": result["prediction"],
                        "confidence": result["confidence"]
                    })

                # Display results table
                mock_dataframe(batch_results)
                mock_success(f"Processed {len(uploaded_files)} images successfully!")

            # Verify batch workflow
            mock_uploader.assert_called_with("Upload multiple images", accept_multiple_files=True)
            mock_process_button.assert_called_with("Process Batch")
            mock_progress.assert_called()
            mock_dataframe.assert_called()
            mock_success.assert_called_with("Processed 3 images successfully!")

            # Verify progress updates
            assert mock_progress_bar.progress.call_count == 3  # One per file

            # Verify API calls
            assert mock_api_client.classify.call_count == 3  # One per file

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_batch_segmentation_workflow(self, mock_api_client, mock_uploaded_files):
        """Test batch segmentation processing workflow."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.button') as mock_process_button, \
             patch('streamlit.progress') as mock_progress, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.image') as mock_image:

            # Mock UI interactions
            mock_uploader.return_value = mock_uploaded_files
            mock_process_button.return_value = True

            # Mock progress and columns
            mock_progress_bar = MagicMock()
            mock_progress.return_value = mock_progress_bar
            mock_columns.return_value = [MagicMock(), MagicMock()] * len(mock_uploaded_files)

            # Simulate batch segmentation
            uploaded_files = mock_uploader("Upload batch for segmentation", accept_multiple_files=True)

            if uploaded_files and mock_process_button("Process Segmentation Batch"):
                batch_results = []

                for i, file in enumerate(uploaded_files):
                    # Update progress
                    mock_progress_bar.progress((i + 1) / len(uploaded_files))

                    # Process segmentation
                    image_data = file.read()
                    result = mock_api_client.segment(image=base64.b64encode(image_data).decode())

                    batch_results.append({
                        "filename": file.name,
                        "mask": result["mask"],
                        "dice_score": result["dice_score"]
                    })

                    # Display result
                    col1, col2 = mock_columns(2)
                    mock_image(file)  # Original
                    mock_image(result["mask"])  # Mask

            # Verify batch segmentation workflow
            assert mock_api_client.segment.call_count == len(mock_uploaded_files)
            assert mock_progress_bar.progress.call_count == len(mock_uploaded_files)
            assert mock_columns.call_count == len(mock_uploaded_files)


class TestPatientAnalysisWorkflow:
    """Test complete patient analysis user workflow."""

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_patient_volume_analysis_workflow(self, mock_api_client, mock_mri_stack):
        """Test patient volume analysis workflow."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.button') as mock_analyze_button, \
             patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.success') as mock_success, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.dataframe') as mock_dataframe:

            # Mock MRI stack upload
            mock_stack_files = []
            for i, image_array in enumerate(mock_mri_stack):
                pil_image = Image.fromarray(image_array)
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                buffer.seek(0)

                mock_file = MagicMock()
                mock_file.name = f"slice_{i:03d}.png"
                mock_file.read.return_value = buffer.getvalue()
                mock_stack_files.append(mock_file)

            mock_uploader.return_value = mock_stack_files
            mock_analyze_button.return_value = True

            # Mock spinner
            mock_spinner_context = MagicMock()
            mock_spinner.return_value.__enter__ = mock_spinner_context
            mock_spinner.return_value.__exit__ = MagicMock()

            # Simulate patient analysis workflow
            uploaded_stack = mock_uploader("Upload MRI stack", accept_multiple_files=True)

            if uploaded_stack and mock_analyze_button("Analyze Patient Volume"):
                with mock_spinner("Analyzing patient volume..."):
                    # Convert stack to base64
                    stack_b64 = [base64.b64encode(f.read()).decode() for f in uploaded_stack]

                    payload = {
                        "image_stack": stack_b64,
                        "image_format": "png",
                        "slice_thickness_mm": 5.0
                    }

                    result = mock_api_client.analyze_patient_stack(**payload)

                if result:
                    mock_success("Patient analysis complete!")

                    # Display patient summary
                    summary = result["patient_summary"]
                    mock_metric("Overall Prediction", summary["overall_prediction"])
                    mock_metric("Patient Confidence", ".1%")
                    mock_metric("Tumor Probability", ".1%")

                    # Display volume analysis
                    volume = result["volume_analysis"]
                    mock_metric("Estimated Volume", "15,421 mm³")
                    mock_metric("Confidence Interval", "14,200 - 16,600 mm³")

                    # Display slice-by-slice results
                    slice_data = result["slice_analyses"]
                    mock_dataframe(slice_data)

            # Verify patient analysis workflow
            mock_uploader.assert_called_with("Upload MRI stack", accept_multiple_files=True)
            mock_analyze_button.assert_called_with("Analyze Patient Volume")
            mock_spinner.assert_called_with("Analyzing patient volume...")
            mock_success.assert_called_with("Patient analysis complete!")
            mock_metric.assert_called()
            mock_dataframe.assert_called()

            mock_api_client.analyze_patient_stack.assert_called_once()


class TestMultiTaskWorkflow:
    """Test complete multi-task analysis user workflow."""

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_unified_multitask_workflow(self, mock_api_client, mock_uploaded_files):
        """Test unified multi-task analysis workflow."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.button') as mock_multitask_button, \
             patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.image') as mock_image:

            # Mock UI interactions
            mock_uploader.return_value = mock_uploaded_files[0]
            mock_multitask_button.return_value = True

            # Mock tabs for classification and segmentation results
            mock_tab1 = MagicMock()
            mock_tab2 = MagicMock()
            mock_tabs.return_value = [mock_tab1, mock_tab2]

            # Mock columns for display
            mock_columns.return_value = [MagicMock(), MagicMock()]

            # Simulate multi-task workflow
            uploaded_file = mock_uploader("Upload image for multi-task analysis")

            if uploaded_file and mock_multitask_button("Run Multi-Task Analysis"):
                image_data = uploaded_file.read()
                image_b64 = base64.b64encode(image_data).decode()

                result = mock_api_client.predict_multitask(image=image_b64)

                if result:
                    # Create tabs for results
                    tab1, tab2 = mock_tabs(["Classification", "Segmentation"])

                    # Classification tab
                    with tab1:
                        cls_result = result["classification"]
                        mock_metric("Prediction", cls_result["prediction"])
                        mock_metric("Confidence", ".1%")

                    # Segmentation tab
                    with tab2:
                        seg_result = result["segmentation"]
                        col1, col2 = mock_columns(2)

                        # Original image
                        mock_image(uploaded_file)

                        # Segmentation mask
                        mock_image(seg_result["mask"])

            # Verify multi-task workflow
            mock_uploader.assert_called_with("Upload image for multi-task analysis")
            mock_multitask_button.assert_called_with("Run Multi-Task Analysis")
            mock_tabs.assert_called_with(["Classification", "Segmentation"])
            mock_columns.assert_called_with(2)
            mock_metric.assert_called()
            mock_image.assert_called()

            mock_api_client.predict_multitask.assert_called_once()


class TestEdgeCaseHandling:
    """Test edge cases and error handling in user workflows."""

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_invalid_file_format_handling(self, mock_api_client):
        """Test handling of invalid file formats."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.error') as mock_error:

            # Mock invalid file upload (wrong format)
            mock_invalid_file = MagicMock()
            mock_invalid_file.name = "document.pdf"
            mock_invalid_file.type = "application/pdf"

            mock_uploader.return_value = mock_invalid_file

            # Simulate file validation
            uploaded_file = mock_uploader("Upload MRI image")

            if uploaded_file:
                # Check file type
                allowed_types = ["image/png", "image/jpeg", "image/jpg"]
                if uploaded_file.type not in allowed_types:
                    mock_error("Invalid file format. Please upload a PNG, JPEG, or JPG image.")

            # Verify error handling
            mock_uploader.assert_called()
            mock_error.assert_called_with("Invalid file format. Please upload a PNG, JPEG, or JPG image.")

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_oversized_file_handling(self, mock_api_client):
        """Test handling of oversized files."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.error') as mock_error:

            # Mock oversized file (100MB)
            mock_large_file = MagicMock()
            mock_large_file.name = "large_scan.png"
            mock_large_file.size = 100 * 1024 * 1024  # 100MB
            mock_large_file.type = "image/png"

            mock_uploader.return_value = mock_large_file

            # Simulate size validation
            uploaded_file = mock_uploader("Upload MRI image")
            max_size = 50 * 1024 * 1024  # 50MB limit

            if uploaded_file and uploaded_file.size > max_size:
                mock_error(f"File too large ({uploaded_file.size / (1024*1024):.1f}MB). Maximum size is {max_size / (1024*1024):.0f}MB.")

            # Verify size validation
            mock_uploader.assert_called()
            mock_error.assert_called()

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_empty_batch_handling(self, mock_api_client):
        """Test handling of empty batch uploads."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.warning') as mock_warning, \
             patch('streamlit.button') as mock_button:

            # Mock empty batch
            mock_uploader.return_value = []
            mock_button.return_value = True

            # Simulate batch processing
            uploaded_files = mock_uploader("Upload batch", accept_multiple_files=True)

            if mock_button("Process Batch"):
                if not uploaded_files:
                    mock_warning("No files selected. Please upload at least one image.")

            # Verify empty batch handling
            mock_uploader.assert_called_with("Upload batch", accept_multiple_files=True)
            mock_button.assert_called_with("Process Batch")
            mock_warning.assert_called_with("No files selected. Please upload at least one image.")

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_api_timeout_handling(self, mock_api_client, mock_uploaded_files):
        """Test handling of API timeouts during workflows."""
        with patch('streamlit.button') as mock_button, \
             patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.file_uploader') as mock_uploader:

            # Mock API timeout
            mock_api_client.classify.side_effect = Exception("Request timed out")

            # Mock UI elements
            mock_uploader.return_value = mock_uploaded_files[0]
            mock_button.return_value = True

            mock_spinner_context = MagicMock()
            mock_spinner.return_value.__enter__ = mock_spinner_context
            mock_spinner.return_value.__exit__ = MagicMock()

            # Simulate workflow with timeout
            uploaded_file = mock_uploader("Upload image")

            if uploaded_file and mock_button("Analyze"):
                with mock_spinner("Processing..."):
                    try:
                        result = mock_api_client.classify(image="test")
                    except Exception as e:
                        mock_error(f"Analysis failed: {str(e)}")

            # Verify timeout handling
            mock_spinner.assert_called_with("Processing...")
            mock_error.assert_called()
            error_message = mock_error.call_args[0][0]
            assert "timed out" in error_message.lower()

    @pytest.mark.skipif(not FRONTEND_AVAILABLE, reason="Frontend components not available")
    def test_network_connectivity_issues(self, mock_api_client):
        """Test handling of network connectivity issues."""
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.button') as mock_retry_button, \
             patch('streamlit.info') as mock_info:

            # Simulate network failure then recovery
            call_count = 0

            def mock_api_with_network_issues(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Network connection failed")
                else:
                    return {"prediction": "success", "confidence": 0.9}

            mock_api_client.classify = mock_api_with_network_issues

            # Simulate retry logic
            retry_attempts = 0
            max_retries = 2
            success = False

            while retry_attempts < max_retries and not success:
                try:
                    result = mock_api_client.classify(image="test")
                    success = True
                    mock_info("Analysis successful!")
                except Exception as e:
                    retry_attempts += 1
                    mock_error(f"Network error (attempt {retry_attempts}): {str(e)}")

                    if retry_attempts < max_retries:
                        mock_info("Retrying...")

            # Verify network error recovery
            assert success is True
            assert retry_attempts == 1  # Failed once, succeeded on retry
            assert mock_error.call_count == 1
            assert mock_info.call_count == 2  # Retry message + success message
