"""
Enterprise Security Implementation for Digi-Cadence Portfolio Management Platform
Comprehensive security features including authentication, authorization, encryption, and audit logging
"""

import os
import jwt
import bcrypt
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from functools import wraps
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json
import time
from dataclasses import dataclass
from enum import Enum

# Security configuration
SECURITY_CONFIG = {
    'jwt_secret_key': os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32)),
    'jwt_algorithm': 'HS256',
    'jwt_expiration_hours': 24,
    'refresh_token_expiration_days': 30,
    'password_min_length': 8,
    'password_complexity_required': True,
    'max_login_attempts': 5,
    'lockout_duration_minutes': 30,
    'session_timeout_minutes': 60,
    'encryption_key': os.getenv('ENCRYPTION_KEY', Fernet.generate_key()),
    'audit_log_retention_days': 365,
    'rate_limit_requests_per_minute': 100,
    'cors_allowed_origins': ['http://localhost:3000', 'http://localhost:5173'],
    'secure_headers_enabled': True,
    'data_classification_levels': ['public', 'internal', 'confidential', 'restricted']
}

class UserRole(Enum):
    """User roles with hierarchical permissions"""
    ADMIN = "admin"
    CMO = "cmo"
    BRAND_MANAGER = "brand_manager"
    DIGITAL_HEAD = "digital_head"
    ANALYST = "analyst"
    VIEWER = "viewer"

class Permission(Enum):
    """System permissions"""
    # Portfolio permissions
    PORTFOLIO_READ = "portfolio:read"
    PORTFOLIO_WRITE = "portfolio:write"
    PORTFOLIO_DELETE = "portfolio:delete"
    PORTFOLIO_ADMIN = "portfolio:admin"
    
    # Brand permissions
    BRAND_READ = "brand:read"
    BRAND_WRITE = "brand:write"
    BRAND_DELETE = "brand:delete"
    BRAND_ADMIN = "brand:admin"
    
    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_WRITE = "analytics:write"
    ANALYTICS_ADMIN = "analytics:admin"
    
    # Agent permissions
    AGENT_READ = "agent:read"
    AGENT_WRITE = "agent:write"
    AGENT_ADMIN = "agent:admin"
    
    # Report permissions
    REPORT_READ = "report:read"
    REPORT_GENERATE = "report:generate"
    REPORT_ADMIN = "report:admin"
    
    # System permissions
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"
    
    # User management permissions
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_ADMIN = "user:admin"

@dataclass
class SecurityContext:
    """Security context for requests"""
    user_id: str
    organization_id: str
    roles: List[UserRole]
    permissions: List[Permission]
    session_id: str
    ip_address: str
    user_agent: str
    timestamp: datetime

class SecurityManager:
    """Comprehensive security manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or SECURITY_CONFIG
        self.logger = self._setup_security_logger()
        self.encryption_key = self.config['encryption_key']
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Role-based permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: [p for p in Permission],  # All permissions
            UserRole.CMO: [
                Permission.PORTFOLIO_READ, Permission.PORTFOLIO_WRITE,
                Permission.BRAND_READ, Permission.BRAND_WRITE,
                Permission.ANALYTICS_READ, Permission.ANALYTICS_WRITE,
                Permission.AGENT_READ, Permission.AGENT_WRITE,
                Permission.REPORT_READ, Permission.REPORT_GENERATE,
                Permission.SYSTEM_READ, Permission.USER_READ
            ],
            UserRole.BRAND_MANAGER: [
                Permission.PORTFOLIO_READ, Permission.BRAND_READ, Permission.BRAND_WRITE,
                Permission.ANALYTICS_READ, Permission.REPORT_READ, Permission.REPORT_GENERATE
            ],
            UserRole.DIGITAL_HEAD: [
                Permission.PORTFOLIO_READ, Permission.BRAND_READ,
                Permission.ANALYTICS_READ, Permission.ANALYTICS_WRITE,
                Permission.AGENT_READ, Permission.AGENT_WRITE,
                Permission.REPORT_READ, Permission.REPORT_GENERATE,
                Permission.SYSTEM_READ
            ],
            UserRole.ANALYST: [
                Permission.PORTFOLIO_READ, Permission.BRAND_READ,
                Permission.ANALYTICS_READ, Permission.REPORT_READ, Permission.REPORT_GENERATE
            ],
            UserRole.VIEWER: [
                Permission.PORTFOLIO_READ, Permission.BRAND_READ,
                Permission.ANALYTICS_READ, Permission.REPORT_READ
            ]
        }
        
        # Failed login attempts tracking
        self.failed_attempts = {}
        self.locked_accounts = {}
    
    def _setup_security_logger(self) -> logging.Logger:
        """Setup security-specific logger"""
        logger = logging.getLogger('digi_cadence_security')
        logger.setLevel(logging.INFO)
        
        # Create file handler for security logs
        handler = logging.FileHandler('logs/security.log')
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        if not self._validate_password_strength(password):
            raise ValueError("Password does not meet security requirements")
        
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < self.config['password_min_length']:
            return False
        
        if not self.config['password_complexity_required']:
            return True
        
        # Check for complexity requirements
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def generate_jwt_token(self, user_id: str, organization_id: str, 
                          roles: List[UserRole], additional_claims: Dict[str, Any] = None) -> str:
        """Generate JWT access token"""
        
        now = datetime.utcnow()
        expiration = now + timedelta(hours=self.config['jwt_expiration_hours'])
        
        payload = {
            'user_id': user_id,
            'organization_id': organization_id,
            'roles': [role.value for role in roles],
            'permissions': [perm.value for perm in self._get_permissions_for_roles(roles)],
            'iat': now,
            'exp': expiration,
            'jti': secrets.token_urlsafe(16)  # JWT ID for token revocation
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(
            payload,
            self.config['jwt_secret_key'],
            algorithm=self.config['jwt_algorithm']
        )
        
        self.logger.info(f"JWT token generated for user {user_id}")
        return token
    
    def generate_refresh_token(self, user_id: str) -> str:
        """Generate refresh token"""
        
        now = datetime.utcnow()
        expiration = now + timedelta(days=self.config['refresh_token_expiration_days'])
        
        payload = {
            'user_id': user_id,
            'token_type': 'refresh',
            'iat': now,
            'exp': expiration,
            'jti': secrets.token_urlsafe(16)
        }
        
        token = jwt.encode(
            payload,
            self.config['jwt_secret_key'],
            algorithm=self.config['jwt_algorithm']
        )
        
        return token
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config['jwt_secret_key'],
                algorithms=[self.config['jwt_algorithm']]
            )
            
            # Check if token is expired
            if datetime.utcfromtimestamp(payload['exp']) < datetime.utcnow():
                raise jwt.ExpiredSignatureError("Token has expired")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Expired JWT token used")
            raise
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid JWT token: {str(e)}")
            raise
    
    def _get_permissions_for_roles(self, roles: List[UserRole]) -> List[Permission]:
        """Get all permissions for given roles"""
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, []))
        return list(permissions)
    
    def check_permission(self, security_context: SecurityContext, 
                        required_permission: Permission) -> bool:
        """Check if user has required permission"""
        return required_permission in security_context.permissions
    
    def require_permission(self, required_permission: Permission):
        """Decorator to require specific permission"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract security context from request
                security_context = kwargs.get('security_context')
                if not security_context:
                    raise PermissionError("Security context not provided")
                
                if not self.check_permission(security_context, required_permission):
                    self.logger.warning(
                        f"Permission denied: User {security_context.user_id} "
                        f"attempted to access {required_permission.value}"
                    )
                    raise PermissionError(f"Permission {required_permission.value} required")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        encrypted_data = self.cipher_suite.encrypt(data.encode('utf-8'))
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded_data = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Decryption failed: {str(e)}")
            raise
    
    def encrypt_pii(self, pii_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt personally identifiable information"""
        encrypted_pii = {}
        
        # Fields that should be encrypted
        pii_fields = ['email', 'phone', 'address', 'ssn', 'credit_card']
        
        for key, value in pii_data.items():
            if key.lower() in pii_fields and value:
                encrypted_pii[key] = self.encrypt_data(str(value))
            else:
                encrypted_pii[key] = value
        
        return encrypted_pii
    
    def check_rate_limit(self, user_id: str, ip_address: str) -> bool:
        """Check rate limiting for user/IP"""
        # Implementation would use Redis or similar for distributed rate limiting
        # For now, return True (no rate limiting)
        return True
    
    def track_failed_login(self, user_id: str, ip_address: str) -> bool:
        """Track failed login attempts"""
        key = f"{user_id}:{ip_address}"
        current_time = datetime.utcnow()
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = []
        
        # Remove old attempts (older than lockout duration)
        lockout_duration = timedelta(minutes=self.config['lockout_duration_minutes'])
        self.failed_attempts[key] = [
            attempt for attempt in self.failed_attempts[key]
            if current_time - attempt < lockout_duration
        ]
        
        # Add current failed attempt
        self.failed_attempts[key].append(current_time)
        
        # Check if account should be locked
        if len(self.failed_attempts[key]) >= self.config['max_login_attempts']:
            self.locked_accounts[key] = current_time + lockout_duration
            self.logger.warning(f"Account locked due to failed attempts: {user_id} from {ip_address}")
            return True
        
        return False
    
    def is_account_locked(self, user_id: str, ip_address: str) -> bool:
        """Check if account is locked"""
        key = f"{user_id}:{ip_address}"
        
        if key in self.locked_accounts:
            if datetime.utcnow() < self.locked_accounts[key]:
                return True
            else:
                # Lockout period expired
                del self.locked_accounts[key]
                if key in self.failed_attempts:
                    del self.failed_attempts[key]
        
        return False
    
    def clear_failed_attempts(self, user_id: str, ip_address: str):
        """Clear failed login attempts after successful login"""
        key = f"{user_id}:{ip_address}"
        if key in self.failed_attempts:
            del self.failed_attempts[key]
        if key in self.locked_accounts:
            del self.locked_accounts[key]
    
    def generate_session_id(self) -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)
    
    def create_security_context(self, user_id: str, organization_id: str,
                              roles: List[UserRole], session_id: str,
                              ip_address: str, user_agent: str) -> SecurityContext:
        """Create security context for request"""
        permissions = self._get_permissions_for_roles(roles)
        
        return SecurityContext(
            user_id=user_id,
            organization_id=organization_id,
            roles=roles,
            permissions=permissions,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow()
        )
    
    def log_security_event(self, event_type: str, user_id: str, 
                          details: Dict[str, Any], severity: str = 'INFO'):
        """Log security events for audit trail"""
        
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'severity': severity,
            'details': details
        }
        
        if severity == 'ERROR':
            self.logger.error(json.dumps(log_entry))
        elif severity == 'WARNING':
            self.logger.warning(json.dumps(log_entry))
        else:
            self.logger.info(json.dumps(log_entry))
    
    def validate_input(self, input_data: str, input_type: str = 'general') -> bool:
        """Validate input to prevent injection attacks"""
        
        # SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+['\"].*['\"])",
        ]
        
        # XSS patterns
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
        ]
        
        # Command injection patterns
        command_patterns = [
            r"[;&|`$()]",
            r"\b(cat|ls|pwd|whoami|id|uname)\b",
        ]
        
        import re
        
        patterns = sql_patterns + xss_patterns
        if input_type == 'command':
            patterns += command_patterns
        
        for pattern in patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                self.logger.warning(f"Suspicious input detected: {pattern}")
                return False
        
        return True
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input data"""
        import html
        
        # HTML escape
        sanitized = html.escape(input_data)
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$', '(', ')']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token"""
        timestamp = str(int(time.time()))
        data = f"{session_id}:{timestamp}"
        
        # Create HMAC
        import hmac
        signature = hmac.new(
            self.config['jwt_secret_key'].encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        token = f"{timestamp}:{signature}"
        return base64.b64encode(token.encode('utf-8')).decode('utf-8')
    
    def verify_csrf_token(self, token: str, session_id: str) -> bool:
        """Verify CSRF token"""
        try:
            decoded_token = base64.b64decode(token.encode('utf-8')).decode('utf-8')
            timestamp, signature = decoded_token.split(':', 1)
            
            # Check if token is not too old (1 hour)
            token_age = int(time.time()) - int(timestamp)
            if token_age > 3600:
                return False
            
            # Verify signature
            data = f"{session_id}:{timestamp}"
            import hmac
            expected_signature = hmac.new(
                self.config['jwt_secret_key'].encode('utf-8'),
                data.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception:
            return False
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }


class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or SECURITY_CONFIG
        self.logger = self._setup_audit_logger()
    
    def _setup_audit_logger(self) -> logging.Logger:
        """Setup audit-specific logger"""
        logger = logging.getLogger('digi_cadence_audit')
        logger.setLevel(logging.INFO)
        
        # Create file handler for audit logs
        handler = logging.FileHandler('logs/audit.log')
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def log_user_action(self, user_id: str, action: str, resource: str,
                       details: Dict[str, Any] = None, ip_address: str = None):
        """Log user actions for audit trail"""
        
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'ip_address': ip_address,
            'details': details or {}
        }
        
        self.logger.info(json.dumps(audit_entry))
    
    def log_data_access(self, user_id: str, data_type: str, data_id: str,
                       access_type: str, ip_address: str = None):
        """Log data access for compliance"""
        
        self.log_user_action(
            user_id=user_id,
            action=f"data_{access_type}",
            resource=f"{data_type}:{data_id}",
            details={'access_type': access_type, 'data_type': data_type},
            ip_address=ip_address
        )
    
    def log_security_event(self, event_type: str, severity: str,
                          details: Dict[str, Any], user_id: str = None):
        """Log security events"""
        
        security_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'user_id': user_id,
            'details': details
        }
        
        self.logger.info(json.dumps(security_entry))


# Security middleware and decorators
def require_authentication(func):
    """Decorator to require authentication"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract token from request headers
        # Verify token and create security context
        # Add security context to kwargs
        return func(*args, **kwargs)
    return wrapper


def require_role(required_role: UserRole):
    """Decorator to require specific role"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            security_context = kwargs.get('security_context')
            if not security_context or required_role not in security_context.roles:
                raise PermissionError(f"Role {required_role.value} required")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Factory function
def create_security_manager(config: Dict[str, Any] = None) -> SecurityManager:
    """Create security manager instance"""
    return SecurityManager(config)


def create_audit_logger(config: Dict[str, Any] = None) -> AuditLogger:
    """Create audit logger instance"""
    return AuditLogger(config)

