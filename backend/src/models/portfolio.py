"""
Core database models for Digi-Cadence Portfolio Management Platform
Multi-tenant architecture with support for multiple brands and projects
"""

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import UUIDType, EncryptedType
from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine
from datetime import datetime
import uuid

db = SQLAlchemy()

# Base model with common fields
class BaseModel(db.Model):
    __abstract__ = True
    
    id = Column(UUIDType(binary=False), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

# Organization model for multi-tenancy
class Organization(BaseModel):
    __tablename__ = 'organizations'
    
    name = Column(String(255), nullable=False)
    description = Column(Text)
    industry = Column(String(100))
    country = Column(String(100))
    timezone = Column(String(50), default='UTC')
    settings = Column(JSON, default={})
    subscription_tier = Column(String(50), default='basic')
    
    # Relationships
    users = relationship('User', back_populates='organization', cascade='all, delete-orphan')
    projects = relationship('Project', back_populates='organization', cascade='all, delete-orphan')
    brands = relationship('Brand', back_populates='organization', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Organization {self.name}>'

# User model with role-based access control
class User(BaseModel):
    __tablename__ = 'users'
    
    organization_id = Column(UUIDType(binary=False), ForeignKey('organizations.id'), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(EncryptedType(String, 'secret-key', AesEngine, 'pkcs5'))
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    role = Column(String(50), nullable=False)  # admin, brand_manager, cmo, digital_head, analyst
    permissions = Column(JSON, default={})
    last_login = Column(DateTime)
    is_verified = Column(Boolean, default=False)
    
    # Relationships
    organization = relationship('Organization', back_populates='users')
    user_projects = relationship('UserProject', back_populates='user', cascade='all, delete-orphan')
    user_brands = relationship('UserBrand', back_populates='user', cascade='all, delete-orphan')
    
    # Indexes
    __table_args__ = (
        Index('idx_user_org_email', 'organization_id', 'email'),
        Index('idx_user_role', 'role'),
    )
    
    def __repr__(self):
        return f'<User {self.email}>'

# Project model for multi-project support
class Project(BaseModel):
    __tablename__ = 'projects'
    
    organization_id = Column(UUIDType(binary=False), ForeignKey('organizations.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    project_type = Column(String(100))  # brand_measurement, competitive_analysis, etc.
    status = Column(String(50), default='active')  # active, archived, completed
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    settings = Column(JSON, default={})
    
    # Relationships
    organization = relationship('Organization', back_populates='projects')
    project_brands = relationship('ProjectBrand', back_populates='project', cascade='all, delete-orphan')
    user_projects = relationship('UserProject', back_populates='project', cascade='all, delete-orphan')
    metrics = relationship('Metric', back_populates='project', cascade='all, delete-orphan')
    
    # Indexes
    __table_args__ = (
        Index('idx_project_org_status', 'organization_id', 'status'),
        Index('idx_project_type', 'project_type'),
    )
    
    def __repr__(self):
        return f'<Project {self.name}>'

# Brand model for multi-brand support
class Brand(BaseModel):
    __tablename__ = 'brands'
    
    organization_id = Column(UUIDType(binary=False), ForeignKey('organizations.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    industry = Column(String(100))
    category = Column(String(100))
    brand_type = Column(String(50))  # primary, competitor, benchmark
    logo_url = Column(String(500))
    website_url = Column(String(500))
    settings = Column(JSON, default={})
    
    # Relationships
    organization = relationship('Organization', back_populates='brands')
    project_brands = relationship('ProjectBrand', back_populates='brand', cascade='all, delete-orphan')
    user_brands = relationship('UserBrand', back_populates='brand', cascade='all, delete-orphan')
    brand_metrics = relationship('BrandMetric', back_populates='brand', cascade='all, delete-orphan')
    
    # Indexes
    __table_args__ = (
        Index('idx_brand_org_type', 'organization_id', 'brand_type'),
        Index('idx_brand_industry', 'industry'),
    )
    
    def __repr__(self):
        return f'<Brand {self.name}>'

# Junction table for Project-Brand many-to-many relationship
class ProjectBrand(BaseModel):
    __tablename__ = 'project_brands'
    
    project_id = Column(UUIDType(binary=False), ForeignKey('projects.id'), nullable=False)
    brand_id = Column(UUIDType(binary=False), ForeignKey('brands.id'), nullable=False)
    role = Column(String(50), nullable=False)  # primary, competitor, benchmark
    weight = Column(Float, default=1.0)
    settings = Column(JSON, default={})
    
    # Relationships
    project = relationship('Project', back_populates='project_brands')
    brand = relationship('Brand', back_populates='project_brands')
    
    # Indexes
    __table_args__ = (
        Index('idx_project_brand_unique', 'project_id', 'brand_id', unique=True),
        Index('idx_project_brand_role', 'project_id', 'role'),
    )

# Junction table for User-Project access control
class UserProject(BaseModel):
    __tablename__ = 'user_projects'
    
    user_id = Column(UUIDType(binary=False), ForeignKey('users.id'), nullable=False)
    project_id = Column(UUIDType(binary=False), ForeignKey('projects.id'), nullable=False)
    access_level = Column(String(50), nullable=False)  # read, write, admin
    permissions = Column(JSON, default={})
    
    # Relationships
    user = relationship('User', back_populates='user_projects')
    project = relationship('Project', back_populates='user_projects')
    
    # Indexes
    __table_args__ = (
        Index('idx_user_project_unique', 'user_id', 'project_id', unique=True),
        Index('idx_user_project_access', 'user_id', 'access_level'),
    )

# Junction table for User-Brand access control
class UserBrand(BaseModel):
    __tablename__ = 'user_brands'
    
    user_id = Column(UUIDType(binary=False), ForeignKey('users.id'), nullable=False)
    brand_id = Column(UUIDType(binary=False), ForeignKey('brands.id'), nullable=False)
    access_level = Column(String(50), nullable=False)  # read, write, admin
    permissions = Column(JSON, default={})
    
    # Relationships
    user = relationship('User', back_populates='user_brands')
    brand = relationship('Brand', back_populates='user_brands')
    
    # Indexes
    __table_args__ = (
        Index('idx_user_brand_unique', 'user_id', 'brand_id', unique=True),
        Index('idx_user_brand_access', 'user_id', 'access_level'),
    )

# Metric definition model
class Metric(BaseModel):
    __tablename__ = 'metrics'
    
    project_id = Column(UUIDType(binary=False), ForeignKey('projects.id'), nullable=False)
    name = Column(String(255), nullable=False)
    display_name = Column(String(255))
    description = Column(Text)
    section_name = Column(String(100), nullable=False)
    platform_name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # maximize, minimize
    unit = Column(String(50))
    weight = Column(Float, default=1.0)
    is_normalized = Column(Boolean, default=False)
    normalization_method = Column(String(50))
    settings = Column(JSON, default={})
    
    # Relationships
    project = relationship('Project', back_populates='metrics')
    brand_metrics = relationship('BrandMetric', back_populates='metric', cascade='all, delete-orphan')
    
    # Indexes
    __table_args__ = (
        Index('idx_metric_project_section', 'project_id', 'section_name'),
        Index('idx_metric_platform', 'platform_name'),
        Index('idx_metric_type', 'metric_type'),
    )
    
    def __repr__(self):
        return f'<Metric {self.name}>'

# Brand metric values model
class BrandMetric(BaseModel):
    __tablename__ = 'brand_metrics'
    
    brand_id = Column(UUIDType(binary=False), ForeignKey('brands.id'), nullable=False)
    metric_id = Column(UUIDType(binary=False), ForeignKey('metrics.id'), nullable=False)
    raw_value = Column(Float)
    normalized_value = Column(Float)
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    data_source = Column(String(255))
    confidence_score = Column(Float)
    metadata = Column(JSON, default={})
    
    # Relationships
    brand = relationship('Brand', back_populates='brand_metrics')
    metric = relationship('Metric', back_populates='brand_metrics')
    
    # Indexes
    __table_args__ = (
        Index('idx_brand_metric_unique', 'brand_id', 'metric_id', 'period_start', unique=True),
        Index('idx_brand_metric_period', 'brand_id', 'period_start', 'period_end'),
        Index('idx_metric_period', 'metric_id', 'period_start'),
    )

# Analysis results model for storing optimization results
class AnalysisResult(BaseModel):
    __tablename__ = 'analysis_results'
    
    project_id = Column(UUIDType(binary=False), ForeignKey('projects.id'), nullable=False)
    analysis_type = Column(String(100), nullable=False)  # genetic_optimization, gap_analysis, etc.
    brand_ids = Column(JSON, nullable=False)  # List of brand IDs analyzed
    input_parameters = Column(JSON, default={})
    results = Column(JSON, nullable=False)
    execution_time = Column(Float)
    status = Column(String(50), default='completed')  # running, completed, failed
    error_message = Column(Text)
    
    # Indexes
    __table_args__ = (
        Index('idx_analysis_project_type', 'project_id', 'analysis_type'),
        Index('idx_analysis_status', 'status'),
        Index('idx_analysis_created', 'created_at'),
    )

# Report model for storing generated reports
class Report(BaseModel):
    __tablename__ = 'reports'
    
    project_id = Column(UUIDType(binary=False), ForeignKey('projects.id'), nullable=False)
    report_type = Column(String(100), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    brand_ids = Column(JSON, nullable=False)  # List of brand IDs in report
    parameters = Column(JSON, default={})
    content = Column(JSON, nullable=False)  # Report content and data
    file_path = Column(String(500))  # Path to generated file
    format = Column(String(50), default='json')  # json, pdf, excel
    generated_by = Column(UUIDType(binary=False), ForeignKey('users.id'))
    is_scheduled = Column(Boolean, default=False)
    schedule_config = Column(JSON, default={})
    
    # Indexes
    __table_args__ = (
        Index('idx_report_project_type', 'project_id', 'report_type'),
        Index('idx_report_generated_by', 'generated_by'),
        Index('idx_report_scheduled', 'is_scheduled'),
    )

# MCP Server status and configuration
class MCPServer(BaseModel):
    __tablename__ = 'mcp_servers'
    
    name = Column(String(100), nullable=False, unique=True)
    server_type = Column(String(50), nullable=False)  # analysis, reporting, integration, orchestration
    host = Column(String(255), nullable=False)
    port = Column(Integer, nullable=False)
    status = Column(String(50), default='inactive')  # active, inactive, error
    version = Column(String(50))
    configuration = Column(JSON, default={})
    health_check_url = Column(String(500))
    last_health_check = Column(DateTime)
    
    # Indexes
    __table_args__ = (
        Index('idx_mcp_server_type', 'server_type'),
        Index('idx_mcp_server_status', 'status'),
    )

# Agent status and configuration
class Agent(BaseModel):
    __tablename__ = 'agents'
    
    name = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False)  # data_processing, analysis, reporting, orchestration
    agent_class = Column(String(100), nullable=False)  # Specific agent class name
    mcp_server_id = Column(UUIDType(binary=False), ForeignKey('mcp_servers.id'))
    status = Column(String(50), default='inactive')  # active, inactive, error, busy
    configuration = Column(JSON, default={})
    last_activity = Column(DateTime)
    performance_metrics = Column(JSON, default={})
    
    # Indexes
    __table_args__ = (
        Index('idx_agent_type_status', 'agent_type', 'status'),
        Index('idx_agent_server', 'mcp_server_id'),
    )

