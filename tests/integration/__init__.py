"""
PHASE 2: API & Integration Testing Package

This package contains comprehensive integration tests for SliceWise MRI Brain Tumor Detection:

test_fastapi_endpoints.py (12 tests)
- Health & Info endpoints (GET /healthz, /model/info)
- Classification endpoints (POST /classify, /classify/gradcam, /classify/batch)
- Segmentation endpoints (POST /segment, /segment/uncertainty, /segment/batch)
- Multi-task endpoints (POST /predict_multitask with conditional execution)
- Patient analysis endpoints (POST /patient/analyze_stack)

test_api_integration_performance.py (10 tests)
- Pydantic schema validation for all request/response models
- Base64 image encoding/decoding functionality
- JSON response format consistency
- CORS configuration validation
- Concurrent load testing under multiple users
- Request queuing and rate limiting
- Memory usage monitoring under load
- Error handling and recovery testing
- Timeout management for long-running requests
- Connection pooling and reuse efficiency

test_backend_integration.py (8 tests)
- ModelManager singleton pattern and integration
- Service layer instantiation and dependency injection
- Business logic processing pipelines
- Error propagation across service layers
- Comprehensive logging integration
- Performance monitoring and metrics collection
- Configuration integration and validation
- End-to-end backend integration testing

test_external_integration.py (5 tests)
- W&B experiment logging and artifact management
- Cloud storage integration for model artifacts
- External monitoring service integration
- Notification system integration
- Database integration preparation (future)

Total: 35 comprehensive integration tests covering the complete API and external service ecosystem.
"""

# Integration tests package for SliceWise Phase 2
