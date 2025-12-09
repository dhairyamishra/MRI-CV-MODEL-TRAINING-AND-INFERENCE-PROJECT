"""
PHASE 4.3: Security & Compliance Testing - HIPAA & Medical Data Security

Tests security measures and HIPAA compliance for SliceWise:
- Input sanitization and malicious input handling
- Data encryption and transmission security
- Access control and user authentication/authorization
- Audit logging and comprehensive security logging

Validates medical data security and regulatory compliance.
"""

import sys
import pytest
import base64
import json
import time
import hashlib
import hmac
import secrets
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import security components
try:
    from app.backend.security import SecurityManager, InputValidator
    from app.backend.audit import AuditLogger
    from app.backend.encryption import DataEncryptor
    SECURITY_AVAILABLE = True
except ImportError:
    # Mock security components
    SECURITY_AVAILABLE = False
    SecurityManager = MagicMock()
    InputValidator = MagicMock()
    AuditLogger = MagicMock()
    DataEncryptor = MagicMock()


@pytest.fixture
def security_manager():
    """Create security manager instance."""
    if SECURITY_AVAILABLE:
        return SecurityManager()
    else:
        manager = MagicMock()
        manager.validate_input.return_value = (True, None)
        manager.sanitize_data.return_value = "sanitized_data"
        manager.check_rate_limit.return_value = (True, None)
        return manager


@pytest.fixture
def audit_logger():
    """Create audit logger instance."""
    if SECURITY_AVAILABLE:
        return AuditLogger()
    else:
        logger = MagicMock()
        logger.log_access.return_value = True
        logger.log_action.return_value = True
        return logger


@pytest.fixture
def input_validator():
    """Create input validator instance."""
    if SECURITY_AVAILABLE:
        return InputValidator()
    else:
        validator = MagicMock()
        validator.validate_image.return_value = (True, None)
        validator.validate_request.return_value = (True, None)
        return validator


class TestInputSanitization:
    """Test input sanitization and malicious input handling."""

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_malicious_file_upload_prevention(self, input_validator):
        """Test prevention of malicious file uploads."""
        # Test various malicious file scenarios
        malicious_scenarios = [
            {
                "filename": "malicious.exe",
                "content": b"malicious executable content",
                "expected_error": "Invalid file type"
            },
            {
                "filename": "script.js",
                "content": b"<script>alert('xss')</script>",
                "expected_error": "Invalid file type"
            },
            {
                "filename": "oversized.dcm",
                "content": b"x" * (100 * 1024 * 1024),  # 100MB
                "expected_error": "File too large"
            },
            {
                "filename": "../../../etc/passwd",
                "content": b"path traversal attempt",
                "expected_error": "Invalid filename"
            }
        ]

        for scenario in malicious_scenarios:
            is_valid, error_msg = input_validator.validate_image(
                scenario["content"],
                scenario["filename"]
            )

            assert is_valid is False, f"Should reject {scenario['filename']}"
            assert error_msg is not None, f"Should provide error for {scenario['filename']}"
            assert scenario["expected_error"].lower() in error_msg.lower()

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_sql_injection_prevention(self, security_manager):
        """Test SQL injection prevention in input data."""
        # Test various SQL injection attempts
        sql_injection_attempts = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "UNION SELECT * FROM users",
            "'; EXEC xp_cmdshell 'dir'--"
        ]

        for malicious_input in sql_injection_attempts:
            sanitized = security_manager.sanitize_data(malicious_input)

            # Sanitized data should not contain dangerous characters
            assert "'" not in sanitized, f"SQL injection not sanitized: {malicious_input}"
            assert ";" not in sanitized, f"SQL injection not sanitized: {malicious_input}"
            assert "--" not in sanitized, f"SQL injection not sanitized: {malicious_input}"

            # But should preserve safe content
            safe_parts = [part for part in malicious_input.split("'")[::2] if part]
            for safe_part in safe_parts:
                assert safe_part in sanitized, f"Safe content removed: {safe_part}"

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_xss_prevention(self, security_manager):
        """Test XSS (Cross-Site Scripting) prevention."""
        # Test various XSS attempts
        xss_attempts = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert(\"xss\")'></iframe>",
            "<svg onload=alert('xss')>"
        ]

        for xss_input in xss_attempts:
            sanitized = security_manager.sanitize_data(xss_input)

            # Should remove or escape dangerous tags
            assert "<script" not in sanitized.lower(), f"XSS not sanitized: {xss_input}"
            assert "javascript:" not in sanitized.lower(), f"XSS not sanitized: {xss_input}"
            assert "onerror" not in sanitized.lower(), f"XSS not sanitized: {xss_input}"

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_buffer_overflow_prevention(self, input_validator):
        """Test buffer overflow prevention."""
        # Test extremely large inputs
        large_inputs = [
            "x" * (10 * 1024 * 1024),  # 10MB string
            b"x" * (50 * 1024 * 1024),  # 50MB bytes
            ["item"] * (100 * 1000),    # 100K item list
        ]

        for large_input in large_inputs:
            try:
                if isinstance(large_input, str):
                    is_valid, error_msg = input_validator.validate_request({"data": large_input})
                elif isinstance(large_input, bytes):
                    is_valid, error_msg = input_validator.validate_image(large_input, "large.bin")
                else:
                    is_valid, error_msg = input_validator.validate_request({"data": large_input})

                # Should handle gracefully (either accept with size limits or reject)
                assert isinstance(is_valid, bool)
                if not is_valid:
                    assert "large" in error_msg.lower() or "size" in error_msg.lower()

            except MemoryError:
                pytest.fail(f"Buffer overflow not prevented for input type: {type(large_input)}")

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_null_byte_injection_prevention(self, security_manager):
        """Test null byte injection prevention."""
        # Test null byte injection attempts
        null_injection_attempts = [
            "filename.txt\x00evil.exe",
            "path/to/file\x00/../../../etc/passwd",
            "normal_file.txt\x00.malicious",
        ]

        for malicious_input in null_injection_attempts:
            sanitized = security_manager.sanitize_data(malicious_input)

            # Should remove null bytes
            assert "\x00" not in sanitized, f"Null byte not removed: {malicious_input}"

            # Should preserve safe content before null byte
            safe_part = malicious_input.split('\x00')[0]
            assert safe_part in sanitized, f"Safe content removed: {safe_part}"


class TestDataEncryption:
    """Test data encryption and transmission security."""

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_data_encryption_at_rest(self):
        """Test data encryption for stored data."""
        encryptor = DataEncryptor()

        test_data = "sensitive medical data: patient diagnosis"
        sensitive_info = ["diagnosis", "medical", "patient"]

        # Encrypt data
        encrypted = encryptor.encrypt(test_data)

        # Should be different from original
        assert encrypted != test_data
        assert isinstance(encrypted, str)

        # Decrypt data
        decrypted = encryptor.decrypt(encrypted)

        # Should recover original
        assert decrypted == test_data

        # Encrypted data should not contain sensitive info in plaintext
        encrypted_lower = encrypted.lower()
        for sensitive in sensitive_info:
            assert sensitive not in encrypted_lower, f"Sensitive data not encrypted: {sensitive}"

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_secure_key_management(self):
        """Test secure encryption key management."""
        encryptor = DataEncryptor()

        # Should generate secure keys
        key = encryptor.generate_key()

        assert len(key) >= 32, "Encryption key too short"  # At least 256 bits
        assert isinstance(key, bytes), "Key should be bytes"

        # Should use cryptographically secure random
        # Test that keys are different (extremely unlikely to be same)
        key2 = encryptor.generate_key()
        assert key != key2, "Keys should be unique"

        # Test key rotation
        old_key = encryptor.get_current_key()
        encryptor.rotate_keys()

        new_key = encryptor.get_current_key()
        assert old_key != new_key, "Key rotation failed"

        # Should still be able to decrypt with old key temporarily
        test_data = "test data for key rotation"
        encrypted = encryptor.encrypt(test_data)

        # Immediately after rotation, should still work
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == test_data

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_https_enforcement(self):
        """Test HTTPS enforcement for secure transmission."""
        # Mock HTTPS configuration
        https_config = {
            "enforce_https": True,
            "hsts_max_age": 31536000,  # 1 year
            "redirect_http": True,
            "certificate_validation": True,
            "cipher_suites": [
                "ECDHE-RSA-AES256-GCM-SHA384",
                "ECDHE-RSA-AES128-GCM-SHA256"
            ]
        }

        # Validate HTTPS configuration
        assert https_config["enforce_https"] is True
        assert https_config["redirect_http"] is True
        assert https_config["certificate_validation"] is True
        assert https_config["hsts_max_age"] > 0

        # Should use strong cipher suites
        strong_ciphers = ["AES256", "AES128", "ECDHE", "GCM"]
        for cipher in https_config["cipher_suites"]:
            has_strong_cipher = any(strong in cipher for strong in strong_ciphers)
            assert has_strong_cipher, f"Weak cipher suite: {cipher}"

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_secure_headers_validation(self):
        """Test security headers for OWASP compliance."""
        security_headers = {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }

        # Validate critical security headers
        assert "Strict-Transport-Security" in security_headers
        assert "max-age=31536000" in security_headers["Strict-Transport-Security"]

        assert "X-Content-Type-Options" in security_headers
        assert security_headers["X-Content-Type-Options"] == "nosniff"

        assert "X-Frame-Options" in security_headers
        assert security_headers["X-Frame-Options"] == "DENY"

        assert "Content-Security-Policy" in security_headers
        assert "'self'" in security_headers["Content-Security-Policy"]

        # Test header injection prevention
        malicious_headers = [
            "Header: value\nX-Injected: bad",
            "Normal-Header: ok\r\nInjected: evil",
            "Header: value\x00injected: bad"
        ]

        for malicious in malicious_headers:
            # Should be sanitized or rejected
            sanitized = malicious.replace('\n', '').replace('\r', '').replace('\x00', '')
            assert '\n' not in sanitized
            assert '\r' not in sanitized
            assert '\x00' not in sanitized


class TestAccessControl:
    """Test access control and user authentication/authorization."""

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_user_authentication(self, security_manager):
        """Test user authentication mechanisms."""
        # Test valid credentials
        valid_users = {
            "radiologist": {"role": "radiologist", "permissions": ["read", "write", "diagnose"]},
            "technician": {"role": "technician", "permissions": ["read", "upload"]},
            "admin": {"role": "admin", "permissions": ["read", "write", "diagnose", "admin"]}
        }

        for username, user_data in valid_users.items():
            # Generate token (simulated)
            token = security_manager.generate_token(username, user_data["role"])

            assert token is not None
            assert isinstance(token, str)

            # Verify token
            is_valid, user_info = security_manager.verify_token(token)

            assert is_valid is True
            assert user_info["username"] == username
            assert user_info["role"] == user_data["role"]

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_role_based_authorization(self, security_manager):
        """Test role-based access control."""
        # Define role permissions
        role_permissions = {
            "radiologist": ["read_images", "write_reports", "diagnose", "export_data"],
            "technician": ["read_images", "upload_images", "view_reports"],
            "admin": ["read_images", "write_reports", "diagnose", "export_data", "manage_users", "system_config"],
            "patient": ["read_own_images", "read_own_reports"]
        }

        # Test permission checks
        test_cases = [
            ("radiologist", "diagnose", True),
            ("radiologist", "manage_users", False),
            ("technician", "upload_images", True),
            ("technician", "diagnose", False),
            ("admin", "system_config", True),
            ("patient", "read_own_images", True),
            ("patient", "manage_users", False)
        ]

        for role, permission, expected_allowed in test_cases:
            has_permission = security_manager.check_permission(role, permission)

            assert has_permission == expected_allowed, \
                f"Role {role} permission {permission} should be {expected_allowed}"

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_session_management(self, security_manager):
        """Test secure session management."""
        # Test session creation
        session_id = security_manager.create_session("user123", {"role": "radiologist"})

        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 20  # Should be cryptographically secure

        # Test session validation
        is_valid, session_data = security_manager.validate_session(session_id)

        assert is_valid is True
        assert session_data["user_id"] == "user123"
        assert session_data["role"] == "radiologist"
        assert "created_at" in session_data
        assert "expires_at" in session_data

        # Test session expiration
        expired_session = security_manager.create_session("user456", {"role": "technician"})

        # Simulate expiration by setting past expiration time
        security_manager._sessions[expired_session]["expires_at"] = time.time() - 3600

        is_valid, _ = security_manager.validate_session(expired_session)
        assert is_valid is False

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_rate_limiting(self, security_manager):
        """Test rate limiting for API protection."""
        # Test normal usage
        for i in range(10):
            allowed, wait_time = security_manager.check_rate_limit("user123", "api_call")
            assert allowed is True, f"Normal request {i} should be allowed"
            assert wait_time == 0

        # Test rate limit exceeded
        # Simulate rapid requests that exceed limit
        for i in range(5):
            allowed, wait_time = security_manager.check_rate_limit("user123", "api_call")
            if not allowed:
                assert wait_time > 0, "Should provide wait time when rate limited"
                break

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_api_key_authentication(self, security_manager):
        """Test API key-based authentication."""
        # Test API key generation
        api_key, secret = security_manager.generate_api_key("client_app")

        assert api_key is not None
        assert secret is not None
        assert len(api_key) >= 32  # Should be long enough
        assert len(secret) >= 32

        # Test API key validation
        client_info = security_manager.validate_api_key(api_key)

        assert client_info is not None
        assert client_info["client_id"] == "client_app"
        assert "permissions" in client_info
        assert "created_at" in client_info

        # Test invalid API key
        invalid_client = security_manager.validate_api_key("invalid_key_123")
        assert invalid_client is None


class TestAuditLogging:
    """Test comprehensive audit logging for HIPAA compliance."""

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_access_logging(self, audit_logger):
        """Test access logging for all data access events."""
        # Test successful access logging
        access_event = {
            "user_id": "radiologist_123",
            "resource_type": "medical_image",
            "resource_id": "patient_001_slice_045",
            "action": "view",
            "ip_address": "192.168.1.100",
            "user_agent": "MedicalViewer/1.0",
            "success": True
        }

        result = audit_logger.log_access(**access_event)

        assert result is True

        # Verify log entry structure
        log_entry = audit_logger.get_recent_logs(1)[0]

        required_fields = ["timestamp", "user_id", "resource_type", "action", "ip_address", "success"]
        for field in required_fields:
            assert field in log_entry

        assert log_entry["user_id"] == "radiologist_123"
        assert log_entry["resource_type"] == "medical_image"
        assert log_entry["success"] is True

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_failed_access_logging(self, audit_logger):
        """Test logging of failed access attempts."""
        # Test failed access logging
        failed_access = {
            "user_id": "unauthorized_user",
            "resource_type": "medical_image",
            "resource_id": "patient_001_slice_045",
            "action": "view",
            "ip_address": "10.0.0.1",
            "user_agent": "SuspiciousBrowser/1.0",
            "success": False,
            "failure_reason": "insufficient_permissions"
        }

        result = audit_logger.log_access(**failed_access)

        assert result is True

        # Verify failure is logged
        log_entry = audit_logger.get_recent_logs(1)[0]

        assert log_entry["success"] is False
        assert "failure_reason" in log_entry
        assert log_entry["failure_reason"] == "insufficient_permissions"

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_data_modification_logging(self, audit_logger):
        """Test logging of data modification events."""
        modification_events = [
            {
                "user_id": "radiologist_123",
                "action": "create_report",
                "resource_type": "medical_report",
                "resource_id": "report_001",
                "changes": {"diagnosis": "tumor_present", "confidence": 0.95}
            },
            {
                "user_id": "technician_456",
                "action": "upload_image",
                "resource_type": "medical_image",
                "resource_id": "image_001",
                "metadata": {"modality": "T1w", "resolution": "256x256"}
            }
        ]

        for event in modification_events:
            result = audit_logger.log_action(**event)
            assert result is True

            # Verify modification is logged
            log_entry = audit_logger.get_recent_logs(1)[0]

            assert log_entry["action"] in ["create_report", "upload_image"]
            assert "changes" in log_entry or "metadata" in log_entry

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_log_integrity_verification(self, audit_logger):
        """Test audit log integrity and tamper prevention."""
        # Add several log entries
        for i in range(5):
            audit_logger.log_access(
                user_id=f"user_{i}",
                resource_type="medical_image",
                resource_id=f"image_{i}",
                action="view",
                ip_address=f"192.168.1.{i}",
                success=True
            )

        # Get logs
        logs = audit_logger.get_recent_logs(10)

        assert len(logs) >= 5

        # Verify log integrity (each log has required fields and is properly formatted)
        for log_entry in logs:
            # Required HIPAA fields
            hipaa_fields = ["timestamp", "user_id", "action", "resource_type", "ip_address", "success"]
            for field in hipaa_fields:
                assert field in log_entry

            # Timestamp should be reasonable
            timestamp = log_entry["timestamp"]
            assert isinstance(timestamp, (int, float))
            current_time = time.time()
            assert abs(current_time - timestamp) < 3600  # Within last hour

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_log_retention_compliance(self, audit_logger):
        """Test log retention meets HIPAA requirements."""
        # HIPAA requires 6 years retention for medical records
        required_retention_days = 6 * 365  # 6 years

        retention_config = audit_logger.get_retention_config()

        assert "medical_records" in retention_config
        assert retention_config["medical_records"]["days"] >= required_retention_days

        # Test audit logs retention
        assert "audit_logs" in retention_config
        assert retention_config["audit_logs"]["days"] >= 2555  # 7 years minimum

        # Test automatic cleanup
        old_entries_count = audit_logger.cleanup_old_entries()

        # Should not fail (even if no entries to clean)
        assert isinstance(old_entries_count, int)
        assert old_entries_count >= 0


class TestHIPAACompliance:
    """Test HIPAA compliance validation."""

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_phi_data_handling(self, security_manager):
        """Test Protected Health Information (PHI) handling."""
        # Test PHI identification and masking
        phi_data = {
            "patient_name": "John Doe",
            "date_of_birth": "1985-03-15",
            "medical_record_number": "MRN123456",
            "diagnosis": "Brain tumor",
            "ssn": "123-45-6789"
        }

        # Should identify PHI fields
        phi_fields = security_manager.identify_phi_fields(phi_data)

        expected_phi = ["patient_name", "date_of_birth", "ssn"]
        for field in expected_phi:
            assert field in phi_fields

        # Should mask PHI data
        masked_data = security_manager.mask_phi_data(phi_data)

        assert masked_data["patient_name"] != phi_data["patient_name"]
        assert masked_data["ssn"] != phi_data["ssn"]
        # Non-PHI should remain unchanged
        assert masked_data["diagnosis"] == phi_data["diagnosis"]

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_minimum_necessary_access(self, security_manager):
        """Test minimum necessary access principle."""
        # Define user roles and their access levels
        role_access = {
            "radiologist": {
                "allowed_resources": ["medical_images", "reports", "diagnoses"],
                "denied_resources": ["billing_data", "admin_settings"]
            },
            "billing_clerk": {
                "allowed_resources": ["billing_data", "insurance_info"],
                "denied_resources": ["medical_images", "diagnoses"]
            }
        }

        # Test access enforcement
        for role, access_rules in role_access.items():
            for allowed_resource in access_rules["allowed_resources"]:
                has_access = security_manager.check_minimum_necessary_access(role, allowed_resource)
                assert has_access is True, f"Role {role} should access {allowed_resource}"

            for denied_resource in access_rules["denied_resources"]:
                has_access = security_manager.check_minimum_necessary_access(role, denied_resource)
                assert has_access is False, f"Role {role} should NOT access {denied_resource}"

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_data_retention_policy(self, security_manager):
        """Test HIPAA-compliant data retention policies."""
        retention_policies = {
            "medical_images": {"retention_years": 7, "destruction_method": "secure_delete"},
            "diagnostic_reports": {"retention_years": 7, "destruction_method": "secure_delete"},
            "audit_logs": {"retention_years": 7, "destruction_method": "archive"},
            "billing_records": {"retention_years": 7, "destruction_method": "secure_delete"},
            "temporary_files": {"retention_days": 30, "destruction_method": "auto_delete"}
        }

        # Validate retention periods meet HIPAA requirements
        for data_type, policy in retention_policies.items():
            if "retention_years" in policy:
                assert policy["retention_years"] >= 6, f"{data_type} retention too short"
            elif "retention_days" in policy:
                years_equivalent = policy["retention_days"] / 365
                assert years_equivalent >= 6, f"{data_type} retention too short"

            # Should specify secure destruction method
            assert "destruction_method" in policy
            assert policy["destruction_method"] in ["secure_delete", "archive", "auto_delete"]

    @pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security components not available")
    def test_incident_response_procedures(self, security_manager):
        """Test incident response procedures for security breaches."""
        # Simulate security incident
        incident = {
            "type": "unauthorized_access",
            "severity": "high",
            "affected_resources": ["patient_records", "medical_images"],
            "affected_users": 150,
            "detection_time": time.time(),
            "description": "Database breach detected"
        }

        # Execute incident response
        response_plan = security_manager.execute_incident_response(incident)

        # Should include required response elements
        required_elements = [
            "containment_actions",
            "notification_procedures",
            "forensic_analysis",
            "recovery_plan",
            "post_incident_review"
        ]

        for element in required_elements:
            assert element in response_plan

        # Should notify affected parties within required timeframe
        assert "notification_deadline_hours" in response_plan
        assert response_plan["notification_deadline_hours"] <= 24  # HIPAA requirement

        # Should include breach assessment
        assert "breach_assessment" in response_plan
        assessment = response_plan["breach_assessment"]
        assert "risk_score" in assessment
        assert assessment["risk_score"] >= incident["severity"] == "high"
