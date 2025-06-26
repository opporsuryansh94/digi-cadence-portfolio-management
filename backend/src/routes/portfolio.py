"""
Portfolio management routes for Digi-Cadence Platform
Handles organizations, projects, brands, and multi-dimensional relationships
"""

from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import and_, or_, func
from sqlalchemy.orm import joinedload
from src.models.portfolio import (
    db, Organization, User, Project, Brand, ProjectBrand, 
    UserProject, UserBrand, Metric, BrandMetric
)
from datetime import datetime
import uuid

portfolio_bp = Blueprint('portfolio', __name__)

# Organization Management
@portfolio_bp.route('/organizations', methods=['GET'])
def get_organizations():
    """Get all organizations with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        search = request.args.get('search', '')
        
        query = Organization.query.filter(Organization.is_active == True)
        
        if search:
            query = query.filter(Organization.name.ilike(f'%{search}%'))
        
        organizations = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'organizations': [{
                'id': str(org.id),
                'name': org.name,
                'description': org.description,
                'industry': org.industry,
                'country': org.country,
                'subscription_tier': org.subscription_tier,
                'created_at': org.created_at.isoformat(),
                'project_count': len(org.projects),
                'brand_count': len(org.brands),
                'user_count': len(org.users)
            } for org in organizations.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': organizations.total,
                'pages': organizations.pages,
                'has_next': organizations.has_next,
                'has_prev': organizations.has_prev
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/organizations', methods=['POST'])
def create_organization():
    """Create a new organization"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'industry']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Check if organization name already exists
        existing_org = Organization.query.filter_by(name=data['name']).first()
        if existing_org:
            return jsonify({'error': 'Organization name already exists'}), 400
        
        organization = Organization(
            name=data['name'],
            description=data.get('description'),
            industry=data['industry'],
            country=data.get('country'),
            timezone=data.get('timezone', 'UTC'),
            settings=data.get('settings', {}),
            subscription_tier=data.get('subscription_tier', 'basic')
        )
        
        db.session.add(organization)
        db.session.commit()
        
        return jsonify({
            'id': str(organization.id),
            'name': organization.name,
            'message': 'Organization created successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Project Management
@portfolio_bp.route('/organizations/<org_id>/projects', methods=['GET'])
def get_organization_projects(org_id):
    """Get all projects for an organization"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        status = request.args.get('status', '')
        project_type = request.args.get('type', '')
        
        query = Project.query.filter(
            Project.organization_id == org_id,
            Project.is_active == True
        )
        
        if status:
            query = query.filter(Project.status == status)
        if project_type:
            query = query.filter(Project.project_type == project_type)
        
        projects = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'projects': [{
                'id': str(project.id),
                'name': project.name,
                'description': project.description,
                'project_type': project.project_type,
                'status': project.status,
                'start_date': project.start_date.isoformat() if project.start_date else None,
                'end_date': project.end_date.isoformat() if project.end_date else None,
                'created_at': project.created_at.isoformat(),
                'brand_count': len(project.project_brands),
                'metric_count': len(project.metrics)
            } for project in projects.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': projects.total,
                'pages': projects.pages
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/organizations/<org_id>/projects', methods=['POST'])
def create_project(org_id):
    """Create a new project"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'project_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Verify organization exists
        organization = Organization.query.get(org_id)
        if not organization:
            return jsonify({'error': 'Organization not found'}), 404
        
        project = Project(
            organization_id=org_id,
            name=data['name'],
            description=data.get('description'),
            project_type=data['project_type'],
            status=data.get('status', 'active'),
            start_date=datetime.fromisoformat(data['start_date']) if data.get('start_date') else None,
            end_date=datetime.fromisoformat(data['end_date']) if data.get('end_date') else None,
            settings=data.get('settings', {})
        )
        
        db.session.add(project)
        db.session.commit()
        
        return jsonify({
            'id': str(project.id),
            'name': project.name,
            'message': 'Project created successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Brand Management
@portfolio_bp.route('/organizations/<org_id>/brands', methods=['GET'])
def get_organization_brands(org_id):
    """Get all brands for an organization"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        brand_type = request.args.get('type', '')
        industry = request.args.get('industry', '')
        
        query = Brand.query.filter(
            Brand.organization_id == org_id,
            Brand.is_active == True
        )
        
        if brand_type:
            query = query.filter(Brand.brand_type == brand_type)
        if industry:
            query = query.filter(Brand.industry == industry)
        
        brands = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'brands': [{
                'id': str(brand.id),
                'name': brand.name,
                'description': brand.description,
                'industry': brand.industry,
                'category': brand.category,
                'brand_type': brand.brand_type,
                'logo_url': brand.logo_url,
                'website_url': brand.website_url,
                'created_at': brand.created_at.isoformat(),
                'project_count': len(brand.project_brands)
            } for brand in brands.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': brands.total,
                'pages': brands.pages
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/organizations/<org_id>/brands', methods=['POST'])
def create_brand(org_id):
    """Create a new brand"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'brand_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Verify organization exists
        organization = Organization.query.get(org_id)
        if not organization:
            return jsonify({'error': 'Organization not found'}), 404
        
        brand = Brand(
            organization_id=org_id,
            name=data['name'],
            description=data.get('description'),
            industry=data.get('industry'),
            category=data.get('category'),
            brand_type=data['brand_type'],
            logo_url=data.get('logo_url'),
            website_url=data.get('website_url'),
            settings=data.get('settings', {})
        )
        
        db.session.add(brand)
        db.session.commit()
        
        return jsonify({
            'id': str(brand.id),
            'name': brand.name,
            'message': 'Brand created successfully'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Multi-Brand, Multi-Project Relationships
@portfolio_bp.route('/projects/<project_id>/brands', methods=['GET'])
def get_project_brands(project_id):
    """Get all brands associated with a project"""
    try:
        project_brands = db.session.query(ProjectBrand, Brand).join(
            Brand, ProjectBrand.brand_id == Brand.id
        ).filter(
            ProjectBrand.project_id == project_id,
            ProjectBrand.is_active == True,
            Brand.is_active == True
        ).all()
        
        return jsonify({
            'brands': [{
                'id': str(brand.id),
                'name': brand.name,
                'brand_type': brand.brand_type,
                'role': project_brand.role,
                'weight': project_brand.weight,
                'settings': project_brand.settings
            } for project_brand, brand in project_brands]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/projects/<project_id>/brands', methods=['POST'])
def add_brand_to_project(project_id):
    """Add a brand to a project"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['brand_id', 'role']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Check if relationship already exists
        existing = ProjectBrand.query.filter_by(
            project_id=project_id,
            brand_id=data['brand_id']
        ).first()
        
        if existing:
            return jsonify({'error': 'Brand already associated with project'}), 400
        
        project_brand = ProjectBrand(
            project_id=project_id,
            brand_id=data['brand_id'],
            role=data['role'],
            weight=data.get('weight', 1.0),
            settings=data.get('settings', {})
        )
        
        db.session.add(project_brand)
        db.session.commit()
        
        return jsonify({'message': 'Brand added to project successfully'}), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Portfolio Analytics Endpoints
@portfolio_bp.route('/organizations/<org_id>/portfolio/summary', methods=['GET'])
def get_portfolio_summary(org_id):
    """Get portfolio summary for an organization"""
    try:
        # Get organization with related data
        organization = Organization.query.options(
            joinedload(Organization.projects),
            joinedload(Organization.brands)
        ).get(org_id)
        
        if not organization:
            return jsonify({'error': 'Organization not found'}), 404
        
        # Calculate portfolio metrics
        total_projects = len([p for p in organization.projects if p.is_active])
        total_brands = len([b for b in organization.brands if b.is_active])
        active_projects = len([p for p in organization.projects if p.status == 'active'])
        
        # Get project types distribution
        project_types = {}
        for project in organization.projects:
            if project.is_active:
                project_types[project.project_type] = project_types.get(project.project_type, 0) + 1
        
        # Get brand types distribution
        brand_types = {}
        for brand in organization.brands:
            if brand.is_active:
                brand_types[brand.brand_type] = brand_types.get(brand.brand_type, 0) + 1
        
        return jsonify({
            'organization': {
                'id': str(organization.id),
                'name': organization.name,
                'industry': organization.industry
            },
            'portfolio_metrics': {
                'total_projects': total_projects,
                'active_projects': active_projects,
                'total_brands': total_brands,
                'project_types': project_types,
                'brand_types': brand_types
            },
            'recent_activity': {
                'recent_projects': [{
                    'id': str(p.id),
                    'name': p.name,
                    'created_at': p.created_at.isoformat()
                } for p in sorted(organization.projects, key=lambda x: x.created_at, reverse=True)[:5]],
                'recent_brands': [{
                    'id': str(b.id),
                    'name': b.name,
                    'created_at': b.created_at.isoformat()
                } for b in sorted(organization.brands, key=lambda x: x.created_at, reverse=True)[:5]]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/portfolio/cross-project-analysis', methods=['POST'])
def cross_project_analysis():
    """Perform cross-project analysis for multiple projects and brands"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['project_ids', 'brand_ids']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        project_ids = data['project_ids']
        brand_ids = data['brand_ids']
        analysis_type = data.get('analysis_type', 'comparison')
        
        # Get projects and brands
        projects = Project.query.filter(Project.id.in_(project_ids)).all()
        brands = Brand.query.filter(Brand.id.in_(brand_ids)).all()
        
        if not projects:
            return jsonify({'error': 'No valid projects found'}), 404
        if not brands:
            return jsonify({'error': 'No valid brands found'}), 404
        
        # Perform cross-project analysis
        analysis_results = {
            'projects': [{
                'id': str(p.id),
                'name': p.name,
                'project_type': p.project_type,
                'brand_count': len(p.project_brands)
            } for p in projects],
            'brands': [{
                'id': str(b.id),
                'name': b.name,
                'brand_type': b.brand_type,
                'project_count': len(b.project_brands)
            } for b in brands],
            'analysis_type': analysis_type,
            'cross_project_metrics': {
                'total_combinations': len(project_ids) * len(brand_ids),
                'unique_projects': len(project_ids),
                'unique_brands': len(brand_ids)
            }
        }
        
        return jsonify(analysis_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Bulk Operations
@portfolio_bp.route('/portfolio/bulk/brands', methods=['POST'])
def bulk_create_brands():
    """Bulk create brands for an organization"""
    try:
        data = request.get_json()
        
        if 'brands' not in data or 'organization_id' not in data:
            return jsonify({'error': 'Missing brands data or organization_id'}), 400
        
        organization_id = data['organization_id']
        brands_data = data['brands']
        
        # Verify organization exists
        organization = Organization.query.get(organization_id)
        if not organization:
            return jsonify({'error': 'Organization not found'}), 404
        
        created_brands = []
        errors = []
        
        for brand_data in brands_data:
            try:
                brand = Brand(
                    organization_id=organization_id,
                    name=brand_data['name'],
                    description=brand_data.get('description'),
                    industry=brand_data.get('industry'),
                    category=brand_data.get('category'),
                    brand_type=brand_data.get('brand_type', 'primary'),
                    logo_url=brand_data.get('logo_url'),
                    website_url=brand_data.get('website_url'),
                    settings=brand_data.get('settings', {})
                )
                db.session.add(brand)
                created_brands.append(brand)
            except Exception as e:
                errors.append(f"Error creating brand {brand_data.get('name', 'unknown')}: {str(e)}")
        
        if created_brands:
            db.session.commit()
        
        return jsonify({
            'created_count': len(created_brands),
            'error_count': len(errors),
            'created_brands': [{
                'id': str(brand.id),
                'name': brand.name
            } for brand in created_brands],
            'errors': errors
        }), 201 if created_brands else 400
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

