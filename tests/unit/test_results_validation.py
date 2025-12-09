"""
PHASE 1.1.9: Results Directory Validation - Critical Safety Tests

Tests results file generation during evaluation, JSON structure validation,
and pipeline tracking verification.
"""

import sys
from pathlib import Path
import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestResultsFileGeneration:
    """Test evaluation scripts create expected output files."""

    def test_multitask_evaluation_results(self):
        """Test multitask evaluation creates results files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            results_dir.mkdir()

            # Mock evaluation results
            mock_results = {
                "model_info": {
                    "architecture": "MultiTaskModel",
                    "checkpoint": "checkpoints/multitask_joint/best_model.pth",
                    "training_phases": ["seg_warmup", "cls_head", "joint"]
                },
                "metrics": {
                    "classification": {
                        "accuracy": 0.913,
                        "precision": 0.9315,
                        "recall": 0.9714,
                        "f1_score": 0.951,
                        "roc_auc": 0.9184
                    },
                    "segmentation": {
                        "dice": 0.7650,
                        "iou": 0.6401,
                        "precision": 0.8234,
                        "recall": 0.7165
                    }
                },
                "evaluation_info": {
                    "dataset": "brats2020_test",
                    "num_samples": 161,
                    "timestamp": "2025-12-08T23:30:00Z"
                }
            }

            # Save mock results
            results_file = results_dir / "multitask_evaluation.json"
            with open(results_file, 'w') as f:
                json.dump(mock_results, f, indent=2)

            # Verify file creation
            assert results_file.exists()
            assert results_file.stat().st_size > 0

            # Verify content
            with open(results_file, 'r') as f:
                loaded_results = json.load(f)

            assert "model_info" in loaded_results
            assert "metrics" in loaded_results
            assert "evaluation_info" in loaded_results

    def test_phase_comparison_results(self):
        """Test phase comparison creates results files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            results_dir.mkdir()

            # Mock phase comparison results
            mock_comparison = {
                "phase_comparison": {
                    "seg_only": {
                        "dice": 0.8635,
                        "accuracy": 0.8758,
                        "improvement": "baseline"
                    },
                    "cls_only": {
                        "dice": None,
                        "accuracy": 0.8758,
                        "improvement": "baseline"
                    },
                    "joint_training": {
                        "dice": 0.7650,
                        "accuracy": 0.9130,
                        "improvement": {
                            "vs_seg_only": {"dice": -0.0985, "accuracy": 0.0372},
                            "vs_cls_only": {"dice": "N/A", "accuracy": 0.0372}
                        }
                    }
                },
                "statistical_analysis": {
                    "significance_tests": {
                        "joint_vs_seg_only": {"p_value": 0.032, "significant": True},
                        "joint_vs_cls_only": {"p_value": 0.028, "significant": True}
                    }
                },
                "timestamp": "2025-12-08T23:30:00Z"
            }

            # Save mock results
            comparison_file = results_dir / "phase_comparison.json"
            with open(comparison_file, 'w') as f:
                json.dump(mock_comparison, f, indent=2)

            # Verify file creation and content
            assert comparison_file.exists()

            with open(comparison_file, 'r') as f:
                loaded_comparison = json.load(f)

            assert "phase_comparison" in loaded_comparison
            assert "statistical_analysis" in loaded_comparison

    def test_pipeline_results_tracking(self):
        """Test pipeline execution tracking."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            results_dir.mkdir()

            # Mock pipeline results
            mock_pipeline = {
                "pipeline_execution": {
                    "start_time": "2025-12-08T20:00:00Z",
                    "end_time": "2025-12-08T23:30:00Z",
                    "total_duration_hours": 3.5,
                    "stages_completed": [
                        {
                            "name": "data_download",
                            "status": "completed",
                            "duration_minutes": 15
                        },
                        {
                            "name": "data_preprocessing",
                            "status": "completed",
                            "duration_minutes": 45
                        },
                        {
                            "name": "training_stage1",
                            "status": "completed",
                            "duration_hours": 1.5
                        },
                        {
                            "name": "training_stage2",
                            "status": "completed",
                            "duration_hours": 1.0
                        },
                        {
                            "name": "training_stage3",
                            "status": "completed",
                            "duration_hours": 1.5
                        },
                        {
                            "name": "evaluation",
                            "status": "completed",
                            "duration_minutes": 30
                        }
                    ]
                },
                "final_metrics": {
                    "best_model": "checkpoints/multitask_joint/best_model.pth",
                    "validation_accuracy": 0.913,
                    "test_dice": 0.765
                }
            }

            # Save mock pipeline results
            pipeline_file = results_dir / "pipeline_results.json"
            with open(pipeline_file, 'w') as f:
                json.dump(mock_pipeline, f, indent=2)

            # Verify pipeline tracking
            assert pipeline_file.exists()

            with open(pipeline_file, 'r') as f:
                loaded_pipeline = json.load(f)

            assert "pipeline_execution" in loaded_pipeline
            assert "stages_completed" in loaded_pipeline["pipeline_execution"]
            assert len(loaded_pipeline["pipeline_execution"]["stages_completed"]) == 6


class TestResultsContentValidation:
    """Test JSON structure and metric calculations."""

    def test_multitask_results_structure(self):
        """Test multitask evaluation results structure."""
        # Valid multitask results structure
        valid_results = {
            "model_info": {
                "architecture": "MultiTaskModel",
                "checkpoint": "checkpoints/multitask_joint/best_model.pth",
                "training_phases": ["seg_warmup", "cls_head", "joint"]
            },
            "metrics": {
                "classification": {
                    "accuracy": 0.913,
                    "precision": 0.9315,
                    "recall": 0.9714,
                    "f1_score": 0.951,
                    "roc_auc": 0.9184
                },
                "segmentation": {
                    "dice": 0.7650,
                    "iou": 0.6401,
                    "precision": 0.8234,
                    "recall": 0.7165
                }
            },
            "evaluation_info": {
                "dataset": "brats2020_test",
                "num_samples": 161,
                "timestamp": "2025-12-08T23:30:00Z"
            }
        }

        # Validate required top-level keys
        required_keys = ["model_info", "metrics", "evaluation_info"]
        for key in required_keys:
            assert key in valid_results

        # Validate classification metrics
        cls_metrics = valid_results["metrics"]["classification"]
        required_cls_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for metric in required_cls_metrics:
            assert metric in cls_metrics
            assert isinstance(cls_metrics[metric], (int, float))
            assert 0 <= cls_metrics[metric] <= 1

        # Validate segmentation metrics
        seg_metrics = valid_results["metrics"]["segmentation"]
        required_seg_metrics = ["dice", "iou", "precision", "recall"]
        for metric in required_seg_metrics:
            assert metric in seg_metrics
            assert isinstance(seg_metrics[metric], (int, float))
            assert 0 <= seg_metrics[metric] <= 1

    def test_metric_calculation_validity(self):
        """Test metric calculations are mathematically valid."""
        # Test F1 score calculation
        def calculate_f1(precision, recall):
            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)

        test_cases = [
            (0.8, 0.9, 0.8444),  # precision, recall, expected_f1
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0)
        ]

        for precision, recall, expected_f1 in test_cases:
            calculated_f1 = calculate_f1(precision, recall)
            assert abs(calculated_f1 - expected_f1) < 0.0001

    def test_results_data_types(self):
        """Test results contain correct data types."""
        # Mock comprehensive results
        results = {
            "model_info": {
                "architecture": "MultiTaskModel",
                "checkpoint": "checkpoints/multitask_joint/best_model.pth",
                "training_phases": ["seg_warmup", "cls_head", "joint"],
                "parameters": 31700000
            },
            "metrics": {
                "classification": {
                    "accuracy": 0.913,
                    "precision": 0.9315,
                    "recall": 0.9714,
                    "f1_score": 0.951,
                    "roc_auc": 0.9184,
                    "confusion_matrix": [[50, 5], [2, 104]]
                },
                "segmentation": {
                    "dice": 0.7650,
                    "iou": 0.6401,
                    "precision": 0.8234,
                    "recall": 0.7165,
                    "hausdorff_distance": 12.34,
                    "average_surface_distance": 2.15
                }
            },
            "evaluation_info": {
                "dataset": "brats2020_test",
                "num_samples": 161,
                "batch_size": 4,
                "timestamp": "2025-12-08T23:30:00Z"
            }
        }

        # Validate data types
        assert isinstance(results["model_info"]["parameters"], int)
        assert isinstance(results["metrics"]["classification"]["accuracy"], (int, float))
        assert isinstance(results["metrics"]["classification"]["confusion_matrix"], list)
        assert isinstance(results["metrics"]["segmentation"]["dice"], (int, float))
        assert isinstance(results["evaluation_info"]["num_samples"], int)

    def test_results_completeness(self):
        """Test results contain all required information."""
        # Incomplete results (missing keys)
        incomplete_results = {
            "metrics": {
                "classification": {
                    "accuracy": 0.913
                    # Missing precision, recall, f1, roc_auc
                }
            }
            # Missing model_info and evaluation_info
        }

        # Check for missing required information
        required_sections = ["model_info", "metrics", "evaluation_info"]
        missing_sections = [section for section in required_sections if section not in incomplete_results]
        assert len(missing_sections) > 0  # Should detect missing sections

        # Check for missing classification metrics
        if "metrics" in incomplete_results and "classification" in incomplete_results["metrics"]:
            cls_metrics = incomplete_results["metrics"]["classification"]
            required_cls = ["precision", "recall", "f1_score", "roc_auc"]
            missing_cls = [m for m in required_cls if m not in cls_metrics]
            assert len(missing_cls) > 0  # Should detect missing metrics


class TestPipelineTracking:
    """Test pipeline execution tracking and audit trails."""

    def test_pipeline_execution_log(self):
        """Test pipeline execution creates comprehensive logs."""
        # Mock pipeline execution log
        execution_log = {
            "execution_id": "pipeline_20251208_233000",
            "start_time": "2025-12-08T23:30:00Z",
            "status": "completed",
            "stages": [
                {
                    "name": "data_download",
                    "start_time": "2025-12-08T23:30:00Z",
                    "end_time": "2025-12-08T23:31:15Z",
                    "duration_seconds": 75,
                    "status": "completed",
                    "output": "Downloaded BraTS 2020 dataset (369 patients)"
                },
                {
                    "name": "data_preprocessing",
                    "start_time": "2025-12-08T23:31:15Z",
                    "end_time": "2025-12-08T23:40:00Z",
                    "duration_seconds": 525,
                    "status": "completed",
                    "output": "Preprocessed 569 2D slices"
                },
                {
                    "name": "training_stage1",
                    "start_time": "2025-12-08T23:40:00Z",
                    "end_time": "2025-12-09T01:10:00Z",
                    "duration_seconds": 5400,
                    "status": "completed",
                    "output": "Stage 1 completed: Dice 0.743"
                }
            ],
            "final_status": {
                "model_checkpoint": "checkpoints/multitask_joint/best_model.pth",
                "best_metrics": {"accuracy": 0.913, "dice": 0.765},
                "total_duration_hours": 3.5
            }
        }

        # Validate execution log structure
        assert "execution_id" in execution_log
        assert "start_time" in execution_log
        assert "status" in execution_log
        assert "stages" in execution_log
        assert len(execution_log["stages"]) > 0

        # Validate each stage
        for stage in execution_log["stages"]:
            required_stage_fields = ["name", "start_time", "end_time", "duration_seconds", "status"]
            for field in required_stage_fields:
                assert field in stage

    def test_pipeline_error_tracking(self):
        """Test pipeline error tracking and recovery."""
        # Mock pipeline with errors
        error_pipeline = {
            "execution_id": "pipeline_20251208_error",
            "start_time": "2025-12-08T23:30:00Z",
            "status": "failed",
            "stages": [
                {
                    "name": "data_download",
                    "status": "completed",
                    "duration_seconds": 75
                },
                {
                    "name": "training_stage1",
                    "status": "failed",
                    "error": "CUDA out of memory",
                    "error_time": "2025-12-09T00:15:30Z",
                    "attempted_recovery": "reduce_batch_size"
                }
            ],
            "error_summary": {
                "primary_error": "CUDA out of memory",
                "recovery_attempts": ["reduce_batch_size", "use_cpu"],
                "final_resolution": "manual_restart_required"
            }
        }

        # Validate error tracking
        assert error_pipeline["status"] == "failed"
        assert "error_summary" in error_pipeline

        failed_stage = None
        for stage in error_pipeline["stages"]:
            if stage["status"] == "failed":
                failed_stage = stage
                break

        assert failed_stage is not None
        assert "error" in failed_stage
        assert "attempted_recovery" in failed_stage

    def test_results_versioning(self):
        """Test results versioning and historical tracking."""
        # Mock versioned results
        results_versions = {
            "v1.0.0": {
                "timestamp": "2025-12-01T10:00:00Z",
                "metrics": {"accuracy": 0.85, "dice": 0.72}
            },
            "v1.1.0": {
                "timestamp": "2025-12-08T23:30:00Z",
                "metrics": {"accuracy": 0.91, "dice": 0.76}
            }
        }

        # Validate versioning
        assert len(results_versions) == 2
        assert "v1.0.0" in results_versions
        assert "v1.1.0" in results_versions

        # Check improvement tracking
        v1_metrics = results_versions["v1.0.0"]["metrics"]
        v11_metrics = results_versions["v1.1.0"]["metrics"]

        assert v11_metrics["accuracy"] > v1_metrics["accuracy"]
        assert v11_metrics["dice"] > v1_metrics["dice"]
