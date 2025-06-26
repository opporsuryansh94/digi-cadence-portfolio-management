"""
Analytics routes for Digi-Cadence Portfolio Management Platform
Handles genetic algorithms, SHAP analysis, and portfolio optimization
"""

from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import and_, or_, func
from src.models.portfolio import (
    db, Project, Brand, Metric, BrandMetric, AnalysisResult
)
from src.analytics.genetic_optimizer import GeneticPortfolioOptimizer
from src.analytics.shap_analyzer import SHAPPortfolioAnalyzer
from src.analytics.correlation_analyzer import CorrelationAnalyzer
from src.analytics.gap_analyzer import CompetitiveGapAnalyzer
from datetime import datetime
import json
import uuid

analytics_bp = Blueprint('analytics', __name__)

# Portfolio Optimization Endpoints
@analytics_bp.route('/portfolio/optimize', methods=['POST'])
def optimize_portfolio():
    """Run genetic algorithm optimization for portfolio"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['project_ids', 'brand_ids']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        project_ids = data['project_ids']
        brand_ids = data['brand_ids']
        optimization_params = data.get('parameters', {})
        
        # Get projects and brands
        projects = Project.query.filter(Project.id.in_(project_ids)).all()
        brands = Brand.query.filter(Brand.id.in_(brand_ids)).all()
        
        if not projects or not brands:
            return jsonify({'error': 'Invalid project or brand IDs'}), 404
        
        # Initialize genetic optimizer
        optimizer = GeneticPortfolioOptimizer(
            projects=projects,
            brands=brands,
            config=current_app.config.get('GENETIC_ALGORITHM_CONFIG', {})
        )
        
        # Update configuration with user parameters
        if optimization_params:
            optimizer.update_config(optimization_params)
        
        # Run optimization
        start_time = datetime.utcnow()
        optimization_result = optimizer.optimize_portfolio(
            num_generations=optimization_params.get('num_generations', 100),
            population_size=optimization_params.get('population_size', 50),
            mutation_rate=optimization_params.get('mutation_rate', 0.2)
        )
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Store analysis result
        analysis_result = AnalysisResult(
            project_id=project_ids[0],  # Primary project
            analysis_type='genetic_optimization',
            brand_ids=brand_ids,
            input_parameters=optimization_params,
            results=optimization_result,
            execution_time=execution_time,
            status='completed'
        )
        
        db.session.add(analysis_result)
        db.session.commit()
        
        return jsonify({
            'analysis_id': str(analysis_result.id),
            'optimization_result': optimization_result,
            'execution_time': execution_time,
            'projects_analyzed': len(project_ids),
            'brands_analyzed': len(brand_ids)
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@analytics_bp.route('/portfolio/shap-analysis', methods=['POST'])
def shap_analysis():
    """Perform SHAP analysis for portfolio attribution"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['project_ids', 'brand_ids', 'target_metric']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        project_ids = data['project_ids']
        brand_ids = data['brand_ids']
        target_metric = data['target_metric']
        shap_params = data.get('parameters', {})
        
        # Get projects and brands
        projects = Project.query.filter(Project.id.in_(project_ids)).all()
        brands = Brand.query.filter(Brand.id.in_(brand_ids)).all()
        
        if not projects or not brands:
            return jsonify({'error': 'Invalid project or brand IDs'}), 404
        
        # Initialize SHAP analyzer
        shap_analyzer = SHAPPortfolioAnalyzer(
            projects=projects,
            brands=brands,
            config=current_app.config.get('SHAP_CONFIG', {})
        )
        
        # Run SHAP analysis
        start_time = datetime.utcnow()
        shap_result = shap_analyzer.analyze_portfolio_attribution(
            target_metric=target_metric,
            background_size=shap_params.get('background_size', 100),
            explainer_type=shap_params.get('explainer_type', 'auto')
        )
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Store analysis result
        analysis_result = AnalysisResult(
            project_id=project_ids[0],
            analysis_type='shap_analysis',
            brand_ids=brand_ids,
            input_parameters={
                'target_metric': target_metric,
                **shap_params
            },
            results=shap_result,
            execution_time=execution_time,
            status='completed'
        )
        
        db.session.add(analysis_result)
        db.session.commit()
        
        return jsonify({
            'analysis_id': str(analysis_result.id),
            'shap_result': shap_result,
            'execution_time': execution_time,
            'target_metric': target_metric
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@analytics_bp.route('/portfolio/gap-analysis', methods=['POST'])
def gap_analysis():
    """Perform competitive gap analysis across portfolio"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['project_ids', 'primary_brand_ids', 'competitor_brand_ids']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        project_ids = data['project_ids']
        primary_brand_ids = data['primary_brand_ids']
        competitor_brand_ids = data['competitor_brand_ids']
        gap_params = data.get('parameters', {})
        
        # Get projects and brands
        projects = Project.query.filter(Project.id.in_(project_ids)).all()
        primary_brands = Brand.query.filter(Brand.id.in_(primary_brand_ids)).all()
        competitor_brands = Brand.query.filter(Brand.id.in_(competitor_brand_ids)).all()
        
        if not projects or not primary_brands or not competitor_brands:
            return jsonify({'error': 'Invalid project or brand IDs'}), 404
        
        # Initialize gap analyzer
        gap_analyzer = CompetitiveGapAnalyzer(
            projects=projects,
            primary_brands=primary_brands,
            competitor_brands=competitor_brands
        )
        
        # Run gap analysis
        start_time = datetime.utcnow()
        gap_result = gap_analyzer.analyze_competitive_gaps(
            gap_threshold=gap_params.get('gap_threshold', 5.0),
            analysis_scope=gap_params.get('scope', 'all_metrics')
        )
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Store analysis result
        analysis_result = AnalysisResult(
            project_id=project_ids[0],
            analysis_type='gap_analysis',
            brand_ids=primary_brand_ids + competitor_brand_ids,
            input_parameters=gap_params,
            results=gap_result,
            execution_time=execution_time,
            status='completed'
        )
        
        db.session.add(analysis_result)
        db.session.commit()
        
        return jsonify({
            'analysis_id': str(analysis_result.id),
            'gap_analysis': gap_result,
            'execution_time': execution_time,
            'primary_brands_count': len(primary_brand_ids),
            'competitor_brands_count': len(competitor_brand_ids)
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@analytics_bp.route('/portfolio/correlation-analysis', methods=['POST'])
def correlation_analysis():
    """Perform correlation analysis across portfolio metrics"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['project_ids', 'brand_ids']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        project_ids = data['project_ids']
        brand_ids = data['brand_ids']
        correlation_params = data.get('parameters', {})
        
        # Get projects and brands
        projects = Project.query.filter(Project.id.in_(project_ids)).all()
        brands = Brand.query.filter(Brand.id.in_(brand_ids)).all()
        
        if not projects or not brands:
            return jsonify({'error': 'Invalid project or brand IDs'}), 404
        
        # Initialize correlation analyzer
        correlation_analyzer = CorrelationAnalyzer(
            projects=projects,
            brands=brands
        )
        
        # Run correlation analysis
        start_time = datetime.utcnow()
        correlation_result = correlation_analyzer.analyze_portfolio_correlations(
            correlation_threshold=correlation_params.get('threshold', 0.7),
            method=correlation_params.get('method', 'pearson')
        )
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Store analysis result
        analysis_result = AnalysisResult(
            project_id=project_ids[0],
            analysis_type='correlation_analysis',
            brand_ids=brand_ids,
            input_parameters=correlation_params,
            results=correlation_result,
            execution_time=execution_time,
            status='completed'
        )
        
        db.session.add(analysis_result)
        db.session.commit()
        
        return jsonify({
            'analysis_id': str(analysis_result.id),
            'correlation_analysis': correlation_result,
            'execution_time': execution_time
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Scenario Analysis
@analytics_bp.route('/portfolio/scenario-analysis', methods=['POST'])
def scenario_analysis():
    """Perform what-if scenario analysis for portfolio"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['project_ids', 'brand_ids', 'scenarios']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        project_ids = data['project_ids']
        brand_ids = data['brand_ids']
        scenarios = data['scenarios']
        
        # Get projects and brands
        projects = Project.query.filter(Project.id.in_(project_ids)).all()
        brands = Brand.query.filter(Brand.id.in_(brand_ids)).all()
        
        if not projects or not brands:
            return jsonify({'error': 'Invalid project or brand IDs'}), 404
        
        scenario_results = []
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', f'Scenario_{len(scenario_results) + 1}')
            metric_changes = scenario.get('metric_changes', {})
            
            # Initialize optimizer for scenario
            optimizer = GeneticPortfolioOptimizer(
                projects=projects,
                brands=brands,
                config=current_app.config.get('GENETIC_ALGORITHM_CONFIG', {})
            )
            
            # Apply scenario changes
            scenario_result = optimizer.run_scenario_analysis(
                scenario_name=scenario_name,
                metric_changes=metric_changes
            )
            
            scenario_results.append({
                'scenario_name': scenario_name,
                'metric_changes': metric_changes,
                'results': scenario_result
            })
        
        # Store analysis result
        analysis_result = AnalysisResult(
            project_id=project_ids[0],
            analysis_type='scenario_analysis',
            brand_ids=brand_ids,
            input_parameters={'scenarios': scenarios},
            results={'scenario_results': scenario_results},
            execution_time=0,  # Will be calculated by individual scenarios
            status='completed'
        )
        
        db.session.add(analysis_result)
        db.session.commit()
        
        return jsonify({
            'analysis_id': str(analysis_result.id),
            'scenario_results': scenario_results,
            'scenarios_analyzed': len(scenarios)
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Analysis History and Results
@analytics_bp.route('/analysis/history', methods=['GET'])
def get_analysis_history():
    """Get analysis history with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        analysis_type = request.args.get('type', '')
        project_id = request.args.get('project_id', '')
        
        query = AnalysisResult.query.filter(AnalysisResult.is_active == True)
        
        if analysis_type:
            query = query.filter(AnalysisResult.analysis_type == analysis_type)
        if project_id:
            query = query.filter(AnalysisResult.project_id == project_id)
        
        query = query.order_by(AnalysisResult.created_at.desc())
        
        analyses = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'analyses': [{
                'id': str(analysis.id),
                'analysis_type': analysis.analysis_type,
                'project_id': str(analysis.project_id),
                'brand_ids': analysis.brand_ids,
                'status': analysis.status,
                'execution_time': analysis.execution_time,
                'created_at': analysis.created_at.isoformat(),
                'has_results': bool(analysis.results)
            } for analysis in analyses.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': analyses.total,
                'pages': analyses.pages
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analytics_bp.route('/analysis/<analysis_id>/results', methods=['GET'])
def get_analysis_results(analysis_id):
    """Get detailed results for a specific analysis"""
    try:
        analysis = AnalysisResult.query.get(analysis_id)
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        return jsonify({
            'analysis_id': str(analysis.id),
            'analysis_type': analysis.analysis_type,
            'project_id': str(analysis.project_id),
            'brand_ids': analysis.brand_ids,
            'input_parameters': analysis.input_parameters,
            'results': analysis.results,
            'execution_time': analysis.execution_time,
            'status': analysis.status,
            'created_at': analysis.created_at.isoformat(),
            'updated_at': analysis.updated_at.isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Real-time Analytics
@analytics_bp.route('/portfolio/real-time-metrics', methods=['POST'])
def get_real_time_metrics():
    """Get real-time portfolio metrics"""
    try:
        data = request.get_json()
        
        project_ids = data.get('project_ids', [])
        brand_ids = data.get('brand_ids', [])
        metric_names = data.get('metrics', [])
        
        # Build query for real-time metrics
        query = db.session.query(BrandMetric, Metric, Brand).join(
            Metric, BrandMetric.metric_id == Metric.id
        ).join(
            Brand, BrandMetric.brand_id == Brand.id
        )
        
        if project_ids:
            query = query.filter(Metric.project_id.in_(project_ids))
        if brand_ids:
            query = query.filter(Brand.id.in_(brand_ids))
        if metric_names:
            query = query.filter(Metric.name.in_(metric_names))
        
        # Get latest metrics
        query = query.order_by(BrandMetric.period_start.desc())
        
        metrics_data = query.limit(1000).all()
        
        # Format response
        real_time_data = {}
        for brand_metric, metric, brand in metrics_data:
            brand_key = str(brand.id)
            if brand_key not in real_time_data:
                real_time_data[brand_key] = {
                    'brand_name': brand.name,
                    'brand_type': brand.brand_type,
                    'metrics': {}
                }
            
            real_time_data[brand_key]['metrics'][metric.name] = {
                'raw_value': brand_metric.raw_value,
                'normalized_value': brand_metric.normalized_value,
                'period_start': brand_metric.period_start.isoformat() if brand_metric.period_start else None,
                'confidence_score': brand_metric.confidence_score,
                'metric_type': metric.metric_type,
                'section': metric.section_name,
                'platform': metric.platform_name
            }
        
        return jsonify({
            'real_time_metrics': real_time_data,
            'timestamp': datetime.utcnow().isoformat(),
            'brands_count': len(real_time_data),
            'metrics_count': sum(len(brand_data['metrics']) for brand_data in real_time_data.values())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Batch Analytics
@analytics_bp.route('/portfolio/batch-analysis', methods=['POST'])
def batch_analysis():
    """Run multiple analytics in batch"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['project_ids', 'brand_ids', 'analysis_types']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        project_ids = data['project_ids']
        brand_ids = data['brand_ids']
        analysis_types = data['analysis_types']
        batch_params = data.get('parameters', {})
        
        batch_results = []
        total_execution_time = 0
        
        for analysis_type in analysis_types:
            try:
                start_time = datetime.utcnow()
                
                if analysis_type == 'optimization':
                    # Run genetic optimization
                    projects = Project.query.filter(Project.id.in_(project_ids)).all()
                    brands = Brand.query.filter(Brand.id.in_(brand_ids)).all()
                    
                    optimizer = GeneticPortfolioOptimizer(projects=projects, brands=brands)
                    result = optimizer.optimize_portfolio()
                    
                elif analysis_type == 'gap_analysis':
                    # Run gap analysis
                    projects = Project.query.filter(Project.id.in_(project_ids)).all()
                    primary_brands = Brand.query.filter(Brand.id.in_(brand_ids[:len(brand_ids)//2])).all()
                    competitor_brands = Brand.query.filter(Brand.id.in_(brand_ids[len(brand_ids)//2:])).all()
                    
                    gap_analyzer = CompetitiveGapAnalyzer(projects, primary_brands, competitor_brands)
                    result = gap_analyzer.analyze_competitive_gaps()
                    
                elif analysis_type == 'correlation':
                    # Run correlation analysis
                    projects = Project.query.filter(Project.id.in_(project_ids)).all()
                    brands = Brand.query.filter(Brand.id.in_(brand_ids)).all()
                    
                    correlation_analyzer = CorrelationAnalyzer(projects, brands)
                    result = correlation_analyzer.analyze_portfolio_correlations()
                    
                else:
                    result = {'error': f'Unknown analysis type: {analysis_type}'}
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                total_execution_time += execution_time
                
                batch_results.append({
                    'analysis_type': analysis_type,
                    'status': 'completed',
                    'execution_time': execution_time,
                    'results': result
                })
                
            except Exception as e:
                batch_results.append({
                    'analysis_type': analysis_type,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Store batch analysis result
        analysis_result = AnalysisResult(
            project_id=project_ids[0],
            analysis_type='batch_analysis',
            brand_ids=brand_ids,
            input_parameters={
                'analysis_types': analysis_types,
                'batch_params': batch_params
            },
            results={'batch_results': batch_results},
            execution_time=total_execution_time,
            status='completed'
        )
        
        db.session.add(analysis_result)
        db.session.commit()
        
        return jsonify({
            'analysis_id': str(analysis_result.id),
            'batch_results': batch_results,
            'total_execution_time': total_execution_time,
            'analyses_completed': len([r for r in batch_results if r['status'] == 'completed']),
            'analyses_failed': len([r for r in batch_results if r['status'] == 'failed'])
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

