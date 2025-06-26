"""
Reports routes for Digi-Cadence Portfolio Management Platform
Handles multi-dimensional reporting with 16 report types supporting multi-brand and multi-project analysis
"""

from flask import Blueprint, request, jsonify, send_file, current_app
from sqlalchemy import and_, or_, func
from datetime import datetime, timedelta
import json
import uuid
import os
from typing import Dict, Any, List, Optional
import pandas as pd
import io

from src.models.portfolio import (
    db, Project, Brand, Metric, BrandMetric, Report, 
    AnalysisResult, Organization, User
)

reports_bp = Blueprint('reports', __name__)

# Report type definitions with multi-brand/multi-project support
REPORT_TYPES = {
    # Core Reports
    'enhanced_recommendation': {
        'name': 'Enhanced Recommendation Report',
        'description': 'Advanced recommendations with portfolio optimization and cross-brand synergies',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 300  # seconds
    },
    'competitive_benchmarking': {
        'name': 'Competitive Benchmarking Report',
        'description': 'Multi-dimensional competitive analysis across portfolio',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 240
    },
    'gap_analysis': {
        'name': 'Gap Analysis Report',
        'description': 'Comprehensive gap analysis with competitive positioning',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 180
    },
    'correlation_network': {
        'name': 'Correlation Network Report',
        'description': 'Cross-brand and cross-project correlation analysis',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 150
    },
    'what_if_scenario': {
        'name': 'What-If Scenario Report',
        'description': 'Portfolio scenario analysis with multi-dimensional impact assessment',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 200
    },
    
    # Strategic Reports
    'weight_sensitivity': {
        'name': 'Weight Sensitivity Report',
        'description': 'Portfolio weight sensitivity analysis across brands and projects',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 120
    },
    'implementation_priority': {
        'name': 'Implementation Priority Report',
        'description': 'Portfolio-wide implementation priority matrix',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 180
    },
    'cross_brand_synergy': {
        'name': 'Cross-Brand Synergy Report',
        'description': 'Synergy identification and optimization across brand portfolio',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 240
    },
    'trend_analysis': {
        'name': 'Trend Analysis Report',
        'description': 'Multi-dimensional trend analysis with forecasting',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 200
    },
    'performance_attribution': {
        'name': 'Performance Attribution Report',
        'description': 'SHAP-based performance attribution across portfolio',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 300
    },
    'competitor_specific_strategy': {
        'name': 'Competitor-Specific Strategy Report',
        'description': 'Targeted competitive strategies for portfolio brands',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 220
    },
    
    # Executive Reports
    'executive_dashboard': {
        'name': 'Executive Dashboard Report',
        'description': 'High-level portfolio performance dashboard',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 120
    },
    'roi_optimization': {
        'name': 'ROI Optimization Report',
        'description': 'Portfolio ROI optimization with resource allocation recommendations',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 180
    },
    'brand_health_index': {
        'name': 'Brand Health Index Report',
        'description': 'Comprehensive brand health assessment across portfolio',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 150
    },
    
    # Portfolio Reports
    'portfolio_performance': {
        'name': 'Portfolio Performance Report',
        'description': 'Comprehensive portfolio performance analysis',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 240
    },
    'cross_project_brand_evolution': {
        'name': 'Cross-Project Brand Evolution Report',
        'description': 'Brand evolution analysis across multiple projects',
        'supports_multi_brand': True,
        'supports_multi_project': True,
        'estimated_time': 200
    }
}

# Report Generation Endpoints
@reports_bp.route('/types', methods=['GET'])
def get_report_types():
    """Get all available report types with their capabilities"""
    try:
        return jsonify({
            'report_types': REPORT_TYPES,
            'total_types': len(REPORT_TYPES),
            'multi_brand_support': len([r for r in REPORT_TYPES.values() if r['supports_multi_brand']]),
            'multi_project_support': len([r for r in REPORT_TYPES.values() if r['supports_multi_project']])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@reports_bp.route('/generate', methods=['POST'])
def generate_report():
    """Generate a new report with multi-brand and multi-project support"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['report_type', 'project_ids', 'brand_ids']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        report_type = data['report_type']
        project_ids = data['project_ids']
        brand_ids = data['brand_ids']
        
        # Validate report type
        if report_type not in REPORT_TYPES:
            return jsonify({'error': f'Invalid report type: {report_type}'}), 400
        
        report_config = REPORT_TYPES[report_type]
        
        # Validate multi-brand/multi-project support
        if len(brand_ids) > 1 and not report_config['supports_multi_brand']:
            return jsonify({'error': f'Report type {report_type} does not support multi-brand analysis'}), 400
        
        if len(project_ids) > 1 and not report_config['supports_multi_project']:
            return jsonify({'error': f'Report type {report_type} does not support multi-project analysis'}), 400
        
        # Validate projects and brands exist
        projects = Project.query.filter(Project.id.in_(project_ids)).all()
        brands = Brand.query.filter(Brand.id.in_(brand_ids)).all()
        
        if len(projects) != len(project_ids):
            return jsonify({'error': 'One or more project IDs are invalid'}), 404
        
        if len(brands) != len(brand_ids):
            return jsonify({'error': 'One or more brand IDs are invalid'}), 404
        
        # Create report record
        report = Report(
            project_id=project_ids[0],  # Primary project
            report_type=report_type,
            title=data.get('title', report_config['name']),
            description=data.get('description', report_config['description']),
            brand_ids=brand_ids,
            parameters=data.get('parameters', {}),
            content={},  # Will be populated during generation
            format=data.get('format', 'json'),
            generated_by=data.get('user_id'),  # Would come from JWT in production
            is_scheduled=data.get('is_scheduled', False),
            schedule_config=data.get('schedule_config', {})
        )
        
        db.session.add(report)
        db.session.commit()
        
        # Generate report content based on type
        report_content = _generate_report_content(
            report_type, projects, brands, data.get('parameters', {})
        )
        
        # Update report with generated content
        report.content = report_content
        db.session.commit()
        
        return jsonify({
            'report_id': str(report.id),
            'report_type': report_type,
            'status': 'completed',
            'title': report.title,
            'project_count': len(project_ids),
            'brand_count': len(brand_ids),
            'estimated_time': report_config['estimated_time'],
            'generated_at': report.created_at.isoformat(),
            'download_url': f'/api/v1/reports/{report.id}/download'
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@reports_bp.route('/<report_id>', methods=['GET'])
def get_report(report_id):
    """Get report details and content"""
    try:
        report = Report.query.get(report_id)
        if not report:
            return jsonify({'error': 'Report not found'}), 404
        
        return jsonify({
            'report_id': str(report.id),
            'report_type': report.report_type,
            'title': report.title,
            'description': report.description,
            'project_id': str(report.project_id),
            'brand_ids': report.brand_ids,
            'parameters': report.parameters,
            'content': report.content,
            'format': report.format,
            'generated_by': report.generated_by,
            'is_scheduled': report.is_scheduled,
            'schedule_config': report.schedule_config,
            'created_at': report.created_at.isoformat(),
            'updated_at': report.updated_at.isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@reports_bp.route('/<report_id>/download', methods=['GET'])
def download_report(report_id):
    """Download report in specified format"""
    try:
        report = Report.query.get(report_id)
        if not report:
            return jsonify({'error': 'Report not found'}), 404
        
        format_type = request.args.get('format', report.format)
        
        if format_type == 'json':
            return jsonify(report.content)
        
        elif format_type == 'excel':
            # Generate Excel file
            excel_file = _generate_excel_report(report)
            return send_file(
                excel_file,
                as_attachment=True,
                download_name=f"{report.title.replace(' ', '_')}_{report_id[:8]}.xlsx",
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        
        elif format_type == 'pdf':
            # Generate PDF file
            pdf_file = _generate_pdf_report(report)
            return send_file(
                pdf_file,
                as_attachment=True,
                download_name=f"{report.title.replace(' ', '_')}_{report_id[:8]}.pdf",
                mimetype='application/pdf'
            )
        
        else:
            return jsonify({'error': f'Unsupported format: {format_type}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Multi-Dimensional Report Endpoints
@reports_bp.route('/portfolio/summary', methods=['POST'])
def generate_portfolio_summary_report():
    """Generate comprehensive portfolio summary report"""
    try:
        data = request.get_json()
        
        organization_id = data.get('organization_id')
        if not organization_id:
            return jsonify({'error': 'Missing organization_id'}), 400
        
        # Get all projects and brands for the organization
        organization = Organization.query.get(organization_id)
        if not organization:
            return jsonify({'error': 'Organization not found'}), 404
        
        projects = [p for p in organization.projects if p.is_active]
        brands = [b for b in organization.brands if b.is_active]
        
        # Generate comprehensive portfolio summary
        portfolio_summary = _generate_portfolio_summary(organization, projects, brands)
        
        # Create report record
        report = Report(
            project_id=projects[0].id if projects else None,
            report_type='portfolio_summary',
            title=f'Portfolio Summary - {organization.name}',
            description=f'Comprehensive portfolio summary for {organization.name}',
            brand_ids=[str(b.id) for b in brands],
            parameters={'organization_id': organization_id},
            content=portfolio_summary,
            format='json',
            generated_by=data.get('user_id')
        )
        
        db.session.add(report)
        db.session.commit()
        
        return jsonify({
            'report_id': str(report.id),
            'organization': organization.name,
            'projects_analyzed': len(projects),
            'brands_analyzed': len(brands),
            'summary': portfolio_summary,
            'generated_at': report.created_at.isoformat()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@reports_bp.route('/cross-brand/analysis', methods=['POST'])
def generate_cross_brand_analysis():
    """Generate cross-brand analysis report"""
    try:
        data = request.get_json()
        
        required_fields = ['brand_ids', 'analysis_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        brand_ids = data['brand_ids']
        analysis_type = data['analysis_type']
        
        if len(brand_ids) < 2:
            return jsonify({'error': 'Cross-brand analysis requires at least 2 brands'}), 400
        
        brands = Brand.query.filter(Brand.id.in_(brand_ids)).all()
        if len(brands) != len(brand_ids):
            return jsonify({'error': 'One or more brand IDs are invalid'}), 404
        
        # Generate cross-brand analysis
        cross_brand_analysis = _generate_cross_brand_analysis(brands, analysis_type, data.get('parameters', {}))
        
        # Create report record
        report = Report(
            project_id=None,  # Cross-brand analysis may span multiple projects
            report_type='cross_brand_analysis',
            title=f'Cross-Brand {analysis_type.title()} Analysis',
            description=f'Cross-brand {analysis_type} analysis for {len(brands)} brands',
            brand_ids=brand_ids,
            parameters={'analysis_type': analysis_type, **data.get('parameters', {})},
            content=cross_brand_analysis,
            format='json',
            generated_by=data.get('user_id')
        )
        
        db.session.add(report)
        db.session.commit()
        
        return jsonify({
            'report_id': str(report.id),
            'analysis_type': analysis_type,
            'brands_analyzed': len(brands),
            'analysis_results': cross_brand_analysis,
            'generated_at': report.created_at.isoformat()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@reports_bp.route('/cross-project/comparison', methods=['POST'])
def generate_cross_project_comparison():
    """Generate cross-project comparison report"""
    try:
        data = request.get_json()
        
        required_fields = ['project_ids']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        project_ids = data['project_ids']
        
        if len(project_ids) < 2:
            return jsonify({'error': 'Cross-project comparison requires at least 2 projects'}), 400
        
        projects = Project.query.filter(Project.id.in_(project_ids)).all()
        if len(projects) != len(project_ids):
            return jsonify({'error': 'One or more project IDs are invalid'}), 404
        
        # Generate cross-project comparison
        comparison_analysis = _generate_cross_project_comparison(projects, data.get('parameters', {}))
        
        # Create report record
        report = Report(
            project_id=project_ids[0],  # Primary project
            report_type='cross_project_comparison',
            title='Cross-Project Comparison Analysis',
            description=f'Comparative analysis across {len(projects)} projects',
            brand_ids=[],  # Will be populated from project brands
            parameters=data.get('parameters', {}),
            content=comparison_analysis,
            format='json',
            generated_by=data.get('user_id')
        )
        
        db.session.add(report)
        db.session.commit()
        
        return jsonify({
            'report_id': str(report.id),
            'projects_analyzed': len(projects),
            'comparison_results': comparison_analysis,
            'generated_at': report.created_at.isoformat()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Report Management
@reports_bp.route('/', methods=['GET'])
def list_reports():
    """List reports with filtering and pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        report_type = request.args.get('report_type', '')
        user_id = request.args.get('user_id', '')
        organization_id = request.args.get('organization_id', '')
        
        query = Report.query.filter(Report.is_active == True)
        
        if report_type:
            query = query.filter(Report.report_type == report_type)
        if user_id:
            query = query.filter(Report.generated_by == user_id)
        if organization_id:
            # Filter by organization through project relationship
            query = query.join(Project).filter(Project.organization_id == organization_id)
        
        query = query.order_by(Report.created_at.desc())
        
        reports = query.paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'reports': [{
                'id': str(report.id),
                'report_type': report.report_type,
                'title': report.title,
                'description': report.description,
                'brand_count': len(report.brand_ids),
                'format': report.format,
                'generated_by': report.generated_by,
                'is_scheduled': report.is_scheduled,
                'created_at': report.created_at.isoformat()
            } for report in reports.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': reports.total,
                'pages': reports.pages,
                'has_next': reports.has_next,
                'has_prev': reports.has_prev
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@reports_bp.route('/<report_id>', methods=['DELETE'])
def delete_report(report_id):
    """Delete a report"""
    try:
        report = Report.query.get(report_id)
        if not report:
            return jsonify({'error': 'Report not found'}), 404
        
        report.is_active = False
        db.session.commit()
        
        return jsonify({'message': 'Report deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Scheduled Reports
@reports_bp.route('/schedule', methods=['POST'])
def schedule_report():
    """Schedule a report for automatic generation"""
    try:
        data = request.get_json()
        
        required_fields = ['report_type', 'project_ids', 'brand_ids', 'schedule_config']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate schedule configuration
        schedule_config = data['schedule_config']
        required_schedule_fields = ['frequency', 'start_date']
        for field in required_schedule_fields:
            if field not in schedule_config:
                return jsonify({'error': f'Missing schedule field: {field}'}), 400
        
        # Create scheduled report
        report = Report(
            project_id=data['project_ids'][0],
            report_type=data['report_type'],
            title=data.get('title', f"Scheduled {REPORT_TYPES[data['report_type']]['name']}"),
            description=data.get('description', 'Automatically generated scheduled report'),
            brand_ids=data['brand_ids'],
            parameters=data.get('parameters', {}),
            content={},
            format=data.get('format', 'json'),
            generated_by=data.get('user_id'),
            is_scheduled=True,
            schedule_config=schedule_config
        )
        
        db.session.add(report)
        db.session.commit()
        
        return jsonify({
            'report_id': str(report.id),
            'message': 'Report scheduled successfully',
            'schedule_config': schedule_config,
            'next_generation': schedule_config.get('start_date')
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Report Generation Helper Functions
def _generate_report_content(report_type: str, projects: List[Project], brands: List[Brand], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Generate report content based on type and parameters"""
    
    # This is a simplified implementation
    # In production, this would call the appropriate analytics engines
    
    base_content = {
        'report_type': report_type,
        'generation_timestamp': datetime.utcnow().isoformat(),
        'projects_analyzed': [{'id': str(p.id), 'name': p.name} for p in projects],
        'brands_analyzed': [{'id': str(b.id), 'name': b.name, 'type': b.brand_type} for b in brands],
        'parameters': parameters
    }
    
    if report_type == 'enhanced_recommendation':
        base_content.update(_generate_enhanced_recommendation_content(projects, brands, parameters))
    elif report_type == 'competitive_benchmarking':
        base_content.update(_generate_competitive_benchmarking_content(projects, brands, parameters))
    elif report_type == 'portfolio_performance':
        base_content.update(_generate_portfolio_performance_content(projects, brands, parameters))
    elif report_type == 'cross_brand_synergy':
        base_content.update(_generate_cross_brand_synergy_content(projects, brands, parameters))
    else:
        # Generic report content
        base_content.update({
            'summary': f'Generated {report_type} report for {len(brands)} brands across {len(projects)} projects',
            'recommendations': ['Recommendation 1', 'Recommendation 2', 'Recommendation 3'],
            'metrics': {
                'total_brands': len(brands),
                'total_projects': len(projects),
                'analysis_scope': 'multi_dimensional' if len(brands) > 1 and len(projects) > 1 else 'standard'
            }
        })
    
    return base_content

def _generate_enhanced_recommendation_content(projects: List[Project], brands: List[Brand], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Generate enhanced recommendation report content"""
    return {
        'optimization_results': {
            'portfolio_score': 85.7,
            'improvement_potential': 23.4,
            'synergy_opportunities': len(brands) * 2
        },
        'brand_recommendations': [
            {
                'brand_id': str(brand.id),
                'brand_name': brand.name,
                'priority_score': 78.5,
                'top_recommendations': [
                    'Increase social media engagement by 15%',
                    'Optimize content strategy for target demographics',
                    'Enhance cross-platform consistency'
                ]
            }
            for brand in brands
        ],
        'cross_brand_synergies': [
            {
                'brand_pair': [brands[0].name, brands[1].name] if len(brands) > 1 else [],
                'synergy_score': 67.3,
                'collaboration_opportunities': ['Joint campaigns', 'Shared content themes']
            }
        ] if len(brands) > 1 else []
    }

def _generate_competitive_benchmarking_content(projects: List[Project], brands: List[Brand], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Generate competitive benchmarking report content"""
    return {
        'benchmark_analysis': {
            'market_position': 'Strong',
            'competitive_gaps': 3,
            'opportunities': 7
        },
        'brand_performance': [
            {
                'brand_id': str(brand.id),
                'brand_name': brand.name,
                'market_share': 12.5,
                'competitive_position': 'Above Average',
                'key_differentiators': ['Innovation', 'Customer Service', 'Brand Recognition']
            }
            for brand in brands
        ]
    }

def _generate_portfolio_performance_content(projects: List[Project], brands: List[Brand], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Generate portfolio performance report content"""
    return {
        'portfolio_metrics': {
            'overall_performance': 82.3,
            'growth_rate': 15.7,
            'efficiency_score': 78.9
        },
        'project_performance': [
            {
                'project_id': str(project.id),
                'project_name': project.name,
                'performance_score': 85.2,
                'roi': 145.6,
                'status': project.status
            }
            for project in projects
        ],
        'brand_contribution': [
            {
                'brand_id': str(brand.id),
                'brand_name': brand.name,
                'contribution_score': 76.4,
                'growth_potential': 'High'
            }
            for brand in brands
        ]
    }

def _generate_cross_brand_synergy_content(projects: List[Project], brands: List[Brand], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Generate cross-brand synergy report content"""
    synergies = []
    
    for i, brand1 in enumerate(brands):
        for brand2 in brands[i+1:]:
            synergies.append({
                'brand_pair': [brand1.name, brand2.name],
                'synergy_score': 72.5,
                'collaboration_potential': 'High',
                'shared_opportunities': ['Content collaboration', 'Cross-promotion', 'Audience overlap']
            })
    
    return {
        'synergy_analysis': {
            'total_synergies_identified': len(synergies),
            'high_potential_pairs': len([s for s in synergies if s['synergy_score'] > 70]),
            'overall_synergy_score': 74.2
        },
        'brand_synergies': synergies,
        'recommendations': [
            'Develop integrated campaign strategy',
            'Create shared content calendar',
            'Implement cross-brand measurement framework'
        ]
    }

def _generate_portfolio_summary(organization: Organization, projects: List[Project], brands: List[Brand]) -> Dict[str, Any]:
    """Generate comprehensive portfolio summary"""
    return {
        'organization_overview': {
            'name': organization.name,
            'industry': organization.industry,
            'total_projects': len(projects),
            'total_brands': len(brands)
        },
        'portfolio_health': {
            'overall_score': 83.7,
            'project_diversity': len(set(p.project_type for p in projects)),
            'brand_diversity': len(set(b.brand_type for b in brands))
        },
        'key_insights': [
            f'Portfolio spans {len(projects)} active projects',
            f'Managing {len(brands)} brands across multiple categories',
            'Strong synergy potential identified across brand portfolio'
        ],
        'recommendations': [
            'Optimize cross-project resource allocation',
            'Enhance brand portfolio synergies',
            'Implement integrated measurement framework'
        ]
    }

def _generate_cross_brand_analysis(brands: List[Brand], analysis_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Generate cross-brand analysis"""
    return {
        'analysis_type': analysis_type,
        'brands_analyzed': [{'id': str(b.id), 'name': b.name} for b in brands],
        'correlation_matrix': {
            f"{brands[i].name}_{brands[j].name}": 0.65
            for i in range(len(brands))
            for j in range(i+1, len(brands))
        },
        'insights': [
            'Strong positive correlation between brand performance metrics',
            'Opportunity for coordinated marketing strategies',
            'Shared audience segments identified'
        ]
    }

def _generate_cross_project_comparison(projects: List[Project], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Generate cross-project comparison"""
    return {
        'projects_compared': [{'id': str(p.id), 'name': p.name, 'type': p.project_type} for p in projects],
        'performance_comparison': {
            project.name: {
                'efficiency_score': 78.5,
                'roi': 142.3,
                'timeline_adherence': 89.2
            }
            for project in projects
        },
        'best_practices': [
            'Standardize measurement frameworks across projects',
            'Share successful strategies between project teams',
            'Implement cross-project learning sessions'
        ]
    }

def _generate_excel_report(report: Report) -> io.BytesIO:
    """Generate Excel file for report"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_df = pd.DataFrame([{
            'Report Type': report.report_type,
            'Title': report.title,
            'Generated At': report.created_at,
            'Brand Count': len(report.brand_ids)
        }])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Content sheet (simplified)
        if 'brand_recommendations' in report.content:
            recommendations_df = pd.DataFrame(report.content['brand_recommendations'])
            recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
    
    output.seek(0)
    return output

def _generate_pdf_report(report: Report) -> io.BytesIO:
    """Generate PDF file for report"""
    # This would use a PDF generation library like ReportLab
    # For now, return a simple placeholder
    output = io.BytesIO()
    output.write(b"PDF Report Content - Implementation needed")
    output.seek(0)
    return output

