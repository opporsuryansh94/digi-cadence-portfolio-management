"""
Comprehensive Security Tests for Digi-Cadence Portfolio Management Platform
Tests authentication, authorization, encryption, input validation, and audit logging
"""

import pytest
import jwt
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import bcrypt
import secrets

# Import security modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from security.security_manager import (
    SecurityManager, AuditLogger, UserRole, Permission, SecurityContext,
    require_authentication, require_role, create_security_manager
)

class TestSecurityManager:
    """Test security manager functionality"""
    
    @pytest.fixture
    def security_manager(self):
        """Create security manager for testing"""
        config = {
            'jwt_secret_key': 'test_secret_key_123',
            'jwt_algorithm': 'HS256',
            'jwt_expiration_hours': 1,
            'password_min_length': 8,
            'password_complexity_required': True,
            'max_login_attempts': 3,
            'lockout_duration_minutes': 5
        }
        return SecurityManager(config)
    
    def test_password_hashing(self, security_manager):
        """Test password hashing and verification"""
        
        # Test valid password
        password = "TestPassword123!"
        hashed = security_manager.hash_password(password)
        
        assert hashed != password
        assert security_manager.verify_password(password, hashed)
        assert not security_manager.verify_password("wrong_password", hashed)
    
    def test_password_strength_validation(self, security_manager):
        """Test password strength validation"""
        
        # Test weak passwords
        weak_passwords = [
            "123",  # Too short
            "password",  # No uppercase, digits, special chars
            "PASSWORD",  # No lowercase, digits, special chars
            "Password123",  # No special chars
            "Password!"  # No digits
        ]
        
        for weak_password in weak_passwords:
            with pytest.raises(ValueError):
                security_manager.hash_password(weak_password)
        
        # Test strong password
        strong_password = "StrongPassword123!"
        hashed = security_manager.hash_password(strong_password)
        assert hashed is not None
    
    def test_jwt_token_generation_and_verification(self, security_manager):
        """Test JWT token generation and verification"""
        
        user_id = "test_user_123"
        organization_id = "test_org_456"
        roles = [UserRole.BRAND_MANAGER, UserRole.ANALYST]
        
        # Generate token
        token = security_manager.generate_jwt_token(user_id, organization_id, roles)
        assert token is not None
        
        # Verify token
        payload = security_manager.verify_jwt_token(token)
        assert payload['user_id'] == user_id
        assert payload['organization_id'] == organization_id
        assert UserRole.BRAND_MANAGER.value in payload['roles']
        assert UserRole.ANALYST.value in payload['roles']
        assert Permission.BRAND_READ.value in payload['permissions']
    
    def test_jwt_token_expiration(self, security_manager):
        """Test JWT token expiration"""
        
        # Create token with short expiration
        security_manager.config['jwt_expiration_hours'] = 0.001  # Very short expiration
        
        user_id = "test_user_123"
        organization_id = "test_org_456"
        roles = [UserRole.VIEWER]
        
        token = security_manager.generate_jwt_token(user_id, organization_id, roles)
        
        # Wait for token to expire
        time.sleep(0.1)
        
        # Verify token is expired
        with pytest.raises(jwt.ExpiredSignatureError):
            security_manager.verify_jwt_token(token)
    
    def test_refresh_token_generation(self, security_manager):
        """Test refresh token generation"""
        
        user_id = "test_user_123"
        refresh_token = security_manager.generate_refresh_token(user_id)
        
        assert refresh_token is not None
        
        # Verify refresh token
        payload = security_manager.verify_jwt_token(refresh_token)
        assert payload['user_id'] == user_id
        assert payload['token_type'] == 'refresh'
    
    def test_role_based_permissions(self, security_manager):
        """Test role-based permission system"""
        
        # Test admin permissions
        admin_permissions = security_manager._get_permissions_for_roles([UserRole.ADMIN])
        assert Permission.SYSTEM_ADMIN in admin_permissions
        assert Permission.USER_ADMIN in admin_permissions
        
        # Test brand manager permissions
        bm_permissions = security_manager._get_permissions_for_roles([UserRole.BRAND_MANAGER])
        assert Permission.BRAND_READ in bm_permissions
        assert Permission.BRAND_WRITE in bm_permissions
        assert Permission.SYSTEM_ADMIN not in bm_permissions
        
        # Test viewer permissions
        viewer_permissions = security_manager._get_permissions_for_roles([UserRole.VIEWER])
        assert Permission.PORTFOLIO_READ in viewer_permissions
        assert Permission.BRAND_WRITE not in viewer_permissions
    
    def test_permission_checking(self, security_manager):
        """Test permission checking"""
        
        # Create security context
        security_context = SecurityContext(
            user_id="test_user",
            organization_id="test_org",
            roles=[UserRole.BRAND_MANAGER],
            permissions=security_manager._get_permissions_for_roles([UserRole.BRAND_MANAGER]),
            session_id="test_session",
            ip_address="127.0.0.1",
            user_agent="test_agent",
            timestamp=datetime.utcnow()
        )
        
        # Test allowed permission
        assert security_manager.check_permission(security_context, Permission.BRAND_READ)
        
        # Test denied permission
        assert not security_manager.check_permission(security_context, Permission.SYSTEM_ADMIN)
    
    def test_data_encryption_decryption(self, security_manager):
        """Test data encryption and decryption"""
        
        sensitive_data = "This is sensitive information"
        
        # Encrypt data
        encrypted = security_manager.encrypt_data(sensitive_data)
        assert encrypted != sensitive_data
        
        # Decrypt data
        decrypted = security_manager.decrypt_data(encrypted)
        assert decrypted == sensitive_data
    
    def test_pii_encryption(self, security_manager):
        """Test PII encryption"""
        
        pii_data = {
            'name': 'John Doe',
            'email': 'john.doe@example.com',
            'phone': '+1-555-123-4567',
            'address': '123 Main St, City, State',
            'department': 'Marketing'  # Not PII
        }
        
        encrypted_pii = security_manager.encrypt_pii(pii_data)
        
        # PII fields should be encrypted
        assert encrypted_pii['email'] != pii_data['email']
        assert encrypted_pii['phone'] != pii_data['phone']
        assert encrypted_pii['address'] != pii_data['address']
        
        # Non-PII fields should remain unchanged
        assert encrypted_pii['name'] == pii_data['name']
        assert encrypted_pii['department'] == pii_data['department']
    
    def test_failed_login_tracking(self, security_manager):
        """Test failed login attempt tracking"""
        
        user_id = "test_user"
        ip_address = "192.168.1.100"
        
        # Test normal failed attempts
        assert not security_manager.track_failed_login(user_id, ip_address)
        assert not security_manager.track_failed_login(user_id, ip_address)
        
        # Test account lockout after max attempts
        assert security_manager.track_failed_login(user_id, ip_address)  # Should trigger lockout
        assert security_manager.is_account_locked(user_id, ip_address)
        
        # Test clearing failed attempts
        security_manager.clear_failed_attempts(user_id, ip_address)
        assert not security_manager.is_account_locked(user_id, ip_address)
    
    def test_input_validation(self, security_manager):
        """Test input validation for security"""
        
        # Test safe inputs
        safe_inputs = [
            "normal text input",
            "user@example.com",
            "Product Name 123"
        ]
        
        for safe_input in safe_inputs:
            assert security_manager.validate_input(safe_input)
        
        # Test dangerous inputs
        dangerous_inputs = [
            "'; DROP TABLE users; --",  # SQL injection
            "<script>alert('xss')</script>",  # XSS
            "javascript:alert('xss')",  # XSS
            "SELECT * FROM users WHERE id = 1",  # SQL
            "cat /etc/passwd"  # Command injection
        ]
        
        for dangerous_input in dangerous_inputs:
            assert not security_manager.validate_input(dangerous_input)
    
    def test_input_sanitization(self, security_manager):
        """Test input sanitization"""
        
        dangerous_input = "<script>alert('test')</script>"
        sanitized = security_manager.sanitize_input(dangerous_input)
        
        assert "<script>" not in sanitized
        assert "alert" in sanitized  # Content should remain but tags removed
    
    def test_csrf_token_generation_verification(self, security_manager):
        """Test CSRF token generation and verification"""
        
        session_id = "test_session_123"
        
        # Generate CSRF token
        csrf_token = security_manager.generate_csrf_token(session_id)
        assert csrf_token is not None
        
        # Verify valid token
        assert security_manager.verify_csrf_token(csrf_token, session_id)
        
        # Verify invalid token
        assert not security_manager.verify_csrf_token("invalid_token", session_id)
        
        # Verify token with wrong session
        assert not security_manager.verify_csrf_token(csrf_token, "wrong_session")
    
    def test_security_headers(self, security_manager):
        """Test security headers generation"""
        
        headers = security_manager.get_security_headers()
        
        required_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Content-Security-Policy'
        ]
        
        for header in required_headers:
            assert header in headers
            assert headers[header] is not None
    
    def test_security_context_creation(self, security_manager):
        """Test security context creation"""
        
        user_id = "test_user"
        organization_id = "test_org"
        roles = [UserRole.CMO]
        session_id = "test_session"
        ip_address = "127.0.0.1"
        user_agent = "test_agent"
        
        context = security_manager.create_security_context(
            user_id, organization_id, roles, session_id, ip_address, user_agent
        )
        
        assert context.user_id == user_id
        assert context.organization_id == organization_id
        assert UserRole.CMO in context.roles
        assert Permission.PORTFOLIO_READ in context.permissions
        assert context.session_id == session_id


class TestAuditLogger:
    """Test audit logging functionality"""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger for testing"""
        return AuditLogger()
    
    @patch('logging.FileHandler')
    def test_audit_logging_setup(self, mock_file_handler, audit_logger):
        """Test audit logger setup"""
        
        # Verify logger is configured
        assert audit_logger.logger is not None
        assert audit_logger.logger.name == 'digi_cadence_audit'
    
    @patch.object(AuditLogger, 'logger')
    def test_user_action_logging(self, mock_logger, audit_logger):
        """Test user action logging"""
        
        user_id = "test_user"
        action = "portfolio_access"
        resource = "portfolio_overview"
        details = {"brand_count": 5}
        ip_address = "192.168.1.100"
        
        audit_logger.log_user_action(user_id, action, resource, details, ip_address)
        
        # Verify logging was called
        mock_logger.info.assert_called_once()
        
        # Verify log content
        call_args = mock_logger.info.call_args[0][0]
        log_data = json.loads(call_args)
        
        assert log_data['user_id'] == user_id
        assert log_data['action'] == action
        assert log_data['resource'] == resource
        assert log_data['details'] == details
        assert log_data['ip_address'] == ip_address
    
    @patch.object(AuditLogger, 'logger')
    def test_data_access_logging(self, mock_logger, audit_logger):
        """Test data access logging"""
        
        user_id = "test_user"
        data_type = "brand"
        data_id = "brand_123"
        access_type = "read"
        ip_address = "192.168.1.100"
        
        audit_logger.log_data_access(user_id, data_type, data_id, access_type, ip_address)
        
        # Verify logging was called
        mock_logger.info.assert_called_once()
        
        # Verify log content
        call_args = mock_logger.info.call_args[0][0]
        log_data = json.loads(call_args)
        
        assert log_data['user_id'] == user_id
        assert log_data['action'] == f"data_{access_type}"
        assert log_data['resource'] == f"{data_type}:{data_id}"
    
    @patch.object(AuditLogger, 'logger')
    def test_security_event_logging(self, mock_logger, audit_logger):
        """Test security event logging"""
        
        event_type = "failed_login"
        severity = "WARNING"
        details = {"ip_address": "192.168.1.100", "attempts": 3}
        user_id = "test_user"
        
        audit_logger.log_security_event(event_type, severity, details, user_id)
        
        # Verify logging was called
        mock_logger.info.assert_called_once()
        
        # Verify log content
        call_args = mock_logger.info.call_args[0][0]
        log_data = json.loads(call_args)
        
        assert log_data['event_type'] == event_type
        assert log_data['severity'] == severity
        assert log_data['details'] == details
        assert log_data['user_id'] == user_id


class TestSecurityDecorators:
    """Test security decorators"""
    
    def test_require_authentication_decorator(self):
        """Test authentication requirement decorator"""
        
        @require_authentication
        def protected_function(security_context=None):
            return "success"
        
        # Test with valid security context
        security_context = Mock()
        result = protected_function(security_context=security_context)
        assert result == "success"
    
    def test_require_role_decorator(self):
        """Test role requirement decorator"""
        
        @require_role(UserRole.ADMIN)
        def admin_function(security_context=None):
            return "admin_success"
        
        # Test with admin role
        admin_context = Mock()
        admin_context.roles = [UserRole.ADMIN]
        result = admin_function(security_context=admin_context)
        assert result == "admin_success"
        
        # Test without admin role
        user_context = Mock()
        user_context.roles = [UserRole.VIEWER]
        
        with pytest.raises(PermissionError):
            admin_function(security_context=user_context)


class TestSecurityIntegration:
    """Integration tests for security features"""
    
    @pytest.fixture
    def security_manager(self):
        """Create security manager for integration testing"""
        return create_security_manager()
    
    def test_complete_authentication_flow(self, security_manager):
        """Test complete authentication flow"""
        
        # Step 1: User registration (password hashing)
        user_id = "integration_test_user"
        password = "SecurePassword123!"
        hashed_password = security_manager.hash_password(password)
        
        # Step 2: User login (password verification)
        assert security_manager.verify_password(password, hashed_password)
        
        # Step 3: Token generation
        organization_id = "test_org"
        roles = [UserRole.BRAND_MANAGER]
        token = security_manager.generate_jwt_token(user_id, organization_id, roles)
        
        # Step 4: Token verification
        payload = security_manager.verify_jwt_token(token)
        assert payload['user_id'] == user_id
        
        # Step 5: Security context creation
        context = security_manager.create_security_context(
            user_id, organization_id, roles, "session_123", "127.0.0.1", "test_agent"
        )
        
        # Step 6: Permission checking
        assert security_manager.check_permission(context, Permission.BRAND_READ)
    
    def test_security_violation_scenarios(self, security_manager):
        """Test various security violation scenarios"""
        
        # Test SQL injection attempt
        malicious_input = "'; DROP TABLE users; --"
        assert not security_manager.validate_input(malicious_input)
        
        # Test XSS attempt
        xss_input = "<script>alert('xss')</script>"
        assert not security_manager.validate_input(xss_input)
        
        # Test invalid token
        with pytest.raises(jwt.InvalidTokenError):
            security_manager.verify_jwt_token("invalid.token.here")
        
        # Test expired token
        expired_payload = {
            'user_id': 'test',
            'exp': datetime.utcnow() - timedelta(hours=1)  # Expired
        }
        expired_token = jwt.encode(expired_payload, security_manager.config['jwt_secret_key'])
        
        with pytest.raises(jwt.ExpiredSignatureError):
            security_manager.verify_jwt_token(expired_token)
    
    def test_concurrent_security_operations(self, security_manager):
        """Test security operations under concurrent access"""
        
        import threading
        import time
        
        results = []
        
        def generate_tokens():
            for i in range(10):
                token = security_manager.generate_jwt_token(
                    f"user_{i}", "org_1", [UserRole.VIEWER]
                )
                results.append(token)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=generate_tokens)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all tokens are valid and unique
        assert len(results) == 50
        assert len(set(results)) == 50  # All tokens should be unique
        
        # Verify all tokens are valid
        for token in results:
            payload = security_manager.verify_jwt_token(token)
            assert 'user_id' in payload
    
    @patch('logging.FileHandler')
    def test_audit_trail_integration(self, mock_file_handler, security_manager):
        """Test complete audit trail integration"""
        
        audit_logger = AuditLogger()
        
        # Simulate user actions
        user_id = "audit_test_user"
        
        # Login event
        audit_logger.log_security_event(
            "user_login", "INFO", 
            {"ip_address": "127.0.0.1", "success": True}, 
            user_id
        )
        
        # Data access event
        audit_logger.log_data_access(
            user_id, "portfolio", "portfolio_123", "read", "127.0.0.1"
        )
        
        # User action event
        audit_logger.log_user_action(
            user_id, "brand_update", "brand_456", 
            {"field": "name", "old_value": "OldName", "new_value": "NewName"},
            "127.0.0.1"
        )
        
        # Logout event
        audit_logger.log_security_event(
            "user_logout", "INFO",
            {"session_duration": 3600},
            user_id
        )
        
        # Verify audit logging was called multiple times
        assert audit_logger.logger.info.call_count == 4


class TestSecurityPerformance:
    """Test security performance under load"""
    
    @pytest.fixture
    def security_manager(self):
        """Create security manager for performance testing"""
        return create_security_manager()
    
    def test_password_hashing_performance(self, security_manager):
        """Test password hashing performance"""
        
        import time
        
        password = "TestPassword123!"
        iterations = 10
        
        start_time = time.time()
        for _ in range(iterations):
            security_manager.hash_password(password)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        
        # Password hashing should complete within reasonable time
        assert avg_time < 1.0  # Less than 1 second per hash
    
    def test_token_generation_performance(self, security_manager):
        """Test JWT token generation performance"""
        
        import time
        
        user_id = "perf_test_user"
        organization_id = "perf_test_org"
        roles = [UserRole.BRAND_MANAGER]
        iterations = 100
        
        start_time = time.time()
        for i in range(iterations):
            security_manager.generate_jwt_token(f"{user_id}_{i}", organization_id, roles)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        
        # Token generation should be fast
        assert avg_time < 0.01  # Less than 10ms per token
    
    def test_encryption_performance(self, security_manager):
        """Test data encryption performance"""
        
        import time
        
        data = "This is test data for encryption performance testing" * 10
        iterations = 100
        
        start_time = time.time()
        for _ in range(iterations):
            encrypted = security_manager.encrypt_data(data)
            security_manager.decrypt_data(encrypted)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        
        # Encryption/decryption should be fast
        assert avg_time < 0.01  # Less than 10ms per operation


if __name__ == "__main__":
    # Run security tests
    pytest.main([__file__, "-v", "--tb=short"])

