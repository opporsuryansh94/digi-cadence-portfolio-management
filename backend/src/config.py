"""
Configuration management for Digi-Cadence Portfolio Management Platform
Supports multiple environments and dynamic configuration
"""

import os
from datetime import timedelta
from typing import Dict, Any

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'digi-cadence-portfolio-secret-key-2024'
    
    # Database Configuration
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_RECORD_QUERIES = True
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 30
    }
    
    # PostgreSQL Configuration
    DB_HOST = os.environ.get('DB_HOST') or 'localhost'
    DB_PORT = os.environ.get('DB_PORT') or '5432'
    DB_NAME = os.environ.get('DB_NAME') or 'digi_cadence_portfolio'
    DB_USER = os.environ.get('DB_USER') or 'postgres'
    DB_PASSWORD = os.environ.get('DB_PASSWORD') or 'password'
    
    SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Redis Configuration
    REDIS_HOST = os.environ.get('REDIS_HOST') or 'localhost'
    REDIS_PORT = os.environ.get('REDIS_PORT') or '6379'
    REDIS_DB = os.environ.get('REDIS_DB') or '0'
    REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD') or None
    
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}" if REDIS_PASSWORD else f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    
    # Celery Configuration
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    CELERY_TASK_SERIALIZER = 'json'
    CELERY_RESULT_SERIALIZER = 'json'
    CELERY_ACCEPT_CONTENT = ['json']
    CELERY_TIMEZONE = 'UTC'
    CELERY_ENABLE_UTC = True
    
    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # MCP Server Configuration
    MCP_SERVERS = {
        'analysis': {
            'host': os.environ.get('MCP_ANALYSIS_HOST') or 'localhost',
            'port': int(os.environ.get('MCP_ANALYSIS_PORT') or '8001'),
            'timeout': 300,
            'max_workers': 4
        },
        'reporting': {
            'host': os.environ.get('MCP_REPORTING_HOST') or 'localhost',
            'port': int(os.environ.get('MCP_REPORTING_PORT') or '8002'),
            'timeout': 180,
            'max_workers': 2
        },
        'integration': {
            'host': os.environ.get('MCP_INTEGRATION_HOST') or 'localhost',
            'port': int(os.environ.get('MCP_INTEGRATION_PORT') or '8003'),
            'timeout': 120,
            'max_workers': 3
        },
        'orchestration': {
            'host': os.environ.get('MCP_ORCHESTRATION_HOST') or 'localhost',
            'port': int(os.environ.get('MCP_ORCHESTRATION_PORT') or '8004'),
            'timeout': 60,
            'max_workers': 2
        }
    }
    
    # Analytics Configuration
    GENETIC_ALGORITHM_CONFIG = {
        'population_size': 50,
        'num_generations': 100,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'elite_size': 10,
        'max_recommendations': 10
    }
    
    SHAP_CONFIG = {
        'background_dataset_size': 100,
        'max_iterations': 1000,
        'explainer_type': 'auto',  # auto, tree, linear, kernel
        'feature_perturbation': 'interventional'
    }
    
    # File Storage Configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    REPORTS_FOLDER = os.environ.get('REPORTS_FOLDER') or 'reports'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Security Configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY') or 'digi-cadence-encryption-key-2024'
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Rate Limiting Configuration
    RATELIMIT_STORAGE_URL = REDIS_URL
    RATELIMIT_DEFAULT = "1000 per hour"
    
    # Portfolio Management Configuration
    PORTFOLIO_CONFIG = {
        'max_brands_per_project': 50,
        'max_projects_per_organization': 100,
        'max_metrics_per_project': 500,
        'default_analysis_timeout': 300,
        'cache_ttl': 3600,  # 1 hour
        'batch_size': 1000
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Override for local development
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        f"postgresql://postgres:password@localhost:5432/digi_cadence_dev"
    
    # Relaxed security for development
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=8)
    
    # Enhanced logging for development
    LOG_LEVEL = 'DEBUG'
    SQLALCHEMY_ECHO = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    
    # Use in-memory database for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False
    
    # Fast JWT expiration for testing
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)
    
    # Reduced timeouts for testing
    MCP_SERVERS = {
        'analysis': {'host': 'localhost', 'port': 8001, 'timeout': 30, 'max_workers': 1},
        'reporting': {'host': 'localhost', 'port': 8002, 'timeout': 30, 'max_workers': 1},
        'integration': {'host': 'localhost', 'port': 8003, 'timeout': 30, 'max_workers': 1},
        'orchestration': {'host': 'localhost', 'port': 8004, 'timeout': 30, 'max_workers': 1}
    }

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Strict CORS for production
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'https://yourdomain.com').split(',')
    
    # Production database with connection pooling
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 50,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 100,
        'pool_timeout': 30
    }
    
    # Enhanced rate limiting for production
    RATELIMIT_DEFAULT = "500 per hour"
    
    # Production logging
    LOG_LEVEL = 'WARNING'

class StagingConfig(Config):
    """Staging configuration"""
    DEBUG = False
    TESTING = False
    
    # Staging database
    SQLALCHEMY_DATABASE_URI = os.environ.get('STAGING_DATABASE_URL') or \
        f"postgresql://postgres:password@localhost:5432/digi_cadence_staging"
    
    # Moderate security for staging
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=12)
    
    # Staging logging
    LOG_LEVEL = 'INFO'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'staging': StagingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: str = None) -> Config:
    """Get configuration based on environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, DevelopmentConfig)

# Dynamic configuration loader
class ConfigManager:
    """Dynamic configuration manager for runtime updates"""
    
    def __init__(self):
        self._config_cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    def get_portfolio_config(self, organization_id: str) -> Dict[str, Any]:
        """Get organization-specific portfolio configuration"""
        # This would typically load from database
        # For now, return default configuration
        return Config.PORTFOLIO_CONFIG
    
    def get_mcp_config(self, server_type: str) -> Dict[str, Any]:
        """Get MCP server configuration"""
        return Config.MCP_SERVERS.get(server_type, {})
    
    def get_analytics_config(self, analysis_type: str) -> Dict[str, Any]:
        """Get analytics configuration for specific analysis type"""
        if analysis_type == 'genetic_algorithm':
            return Config.GENETIC_ALGORITHM_CONFIG
        elif analysis_type == 'shap':
            return Config.SHAP_CONFIG
        else:
            return {}
    
    def update_config(self, config_type: str, config_data: Dict[str, Any]) -> bool:
        """Update configuration dynamically"""
        try:
            # This would typically update database and cache
            self._config_cache[config_type] = config_data
            return True
        except Exception as e:
            print(f"Error updating configuration: {e}")
            return False

# Global configuration manager instance
config_manager = ConfigManager()

