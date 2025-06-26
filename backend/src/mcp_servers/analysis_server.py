"""
Analysis MCP Server for Digi-Cadence Portfolio Management Platform
Handles portfolio optimization, analytics processing, and multi-dimensional analysis
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

from src.mcp_servers.base_server import (
    BaseMCPServer, MCPServerConfig, MCPServerType, MCPRequest, 
    ValidationError, AuthorizationError, MCPServerError
)
from src.analytics.genetic_optimizer import GeneticPortfolioOptimizer
from src.models.portfolio import Project, Brand, Metric, BrandMetric, AnalysisResult, db

class AnalysisMCPServer(BaseMCPServer):
    """
    Analysis MCP Server for portfolio optimization and analytics
    Supports multi-brand, multi-project analysis with advanced algorithms
    """
    
    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        
        # Thread pool for CPU-intensive analytics
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Analysis cache for expensive computations
        self.analysis_cache = {}
        
        # Active analysis jobs tracking
        self.active_jobs = {}
        
        # Analytics configuration
        self.analytics_config = {
            'genetic_algorithm': {
                'population_size': 50,
                'num_generations': 100,
                'mutation_rate': 0.2,
                'crossover_rate': 0.8,
                'elite_size': 10,
                'convergence_threshold': 0.001
            },
            'shap_analysis': {
                'background_dataset_size': 100,
                'max_iterations': 1000,
                'explainer_type': 'auto'
            },
            'correlation_analysis': {
                'min_correlation_threshold': 0.3,
                'method': 'pearson'
            }
        }
        
        self.logger.info("Analysis MCP Server initialized")
    
    async def initialize_server_specific_handlers(self):
        """Initialize analysis-specific request handlers"""
        
        # Portfolio optimization handlers
        self.register_handler("optimize_portfolio", self._handle_portfolio_optimization)
        self.register_handler("optimize_multi_brand", self._handle_multi_brand_optimization)
        self.register_handler("optimize_cross_project", self._handle_cross_project_optimization)
        
        # SHAP analysis handlers
        self.register_handler("shap_analysis", self._handle_shap_analysis)
        self.register_handler("shap_multi_brand", self._handle_multi_brand_shap)
        self.register_handler("shap_attribution", self._handle_attribution_analysis)
        
        # Gap analysis handlers
        self.register_handler("gap_analysis", self._handle_gap_analysis)
        self.register_handler("competitive_gap", self._handle_competitive_gap_analysis)
        self.register_handler("portfolio_gap", self._handle_portfolio_gap_analysis)
        
        # Correlation analysis handlers
        self.register_handler("correlation_analysis", self._handle_correlation_analysis)
        self.register_handler("cross_brand_correlation", self._handle_cross_brand_correlation)
        self.register_handler("cross_project_correlation", self._handle_cross_project_correlation)
        
        # Scenario analysis handlers
        self.register_handler("scenario_analysis", self._handle_scenario_analysis)
        self.register_handler("what_if_analysis", self._handle_what_if_analysis)
        self.register_handler("sensitivity_analysis", self._handle_sensitivity_analysis)
        
        # Batch analysis handlers
        self.register_handler("batch_analysis", self._handle_batch_analysis)
        self.register_handler("scheduled_analysis", self._handle_scheduled_analysis)
        
        # Job management handlers
        self.register_handler("get_job_status", self._handle_get_job_status)
        self.register_handler("cancel_job", self._handle_cancel_job)
        self.register_handler("list_jobs", self._handle_list_jobs)
        
        # Configuration handlers
        self.register_handler("update_config", self._handle_update_config)
        self.register_handler("get_config", self._handle_get_config)
        
        self.logger.info("Analysis MCP Server handlers registered")
    
    async def _check_method_authorization(self, request: MCPRequest):
        """Check method-specific authorization for analysis operations"""
        
        # Methods requiring write access
        write_methods = {
            'optimize_portfolio', 'optimize_multi_brand', 'optimize_cross_project',
            'shap_analysis', 'shap_multi_brand', 'gap_analysis', 'scenario_analysis',
            'batch_analysis', 'scheduled_analysis', 'update_config'
        }
        
        # Methods requiring admin access
        admin_methods = {
            'cancel_job', 'update_config'
        }
        
        if request.method in admin_methods:
            if request.authorization_level not in ['admin', 'system']:
                raise AuthorizationError(f"Admin access required for {request.method}")
        
        elif request.method in write_methods:
            if request.authorization_level not in ['write', 'admin', 'system']:
                raise AuthorizationError(f"Write access required for {request.method}")
    
    # Portfolio Optimization Handlers
    async def _handle_portfolio_optimization(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle portfolio optimization requests"""
        params = request.params
        
        # Validate required parameters
        required_params = ['project_ids', 'brand_ids']
        for param in required_params:
            if param not in params:
                raise ValidationError(f"Missing required parameter: {param}")
        
        project_ids = params['project_ids']
        brand_ids = params['brand_ids']
        optimization_params = params.get('optimization_params', {})
        
        # Create job ID for tracking
        job_id = f"portfolio_opt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request.request_id[:8]}"
        
        # Start optimization in background
        task = asyncio.create_task(
            self._run_portfolio_optimization(job_id, project_ids, brand_ids, optimization_params)
        )
        
        self.active_jobs[job_id] = {
            'task': task,
            'type': 'portfolio_optimization',
            'status': 'running',
            'start_time': datetime.utcnow(),
            'project_ids': project_ids,
            'brand_ids': brand_ids,
            'user_id': request.user_id
        }
        
        return {
            'job_id': job_id,
            'status': 'started',
            'message': 'Portfolio optimization started',
            'estimated_completion': (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        }
    
    async def _run_portfolio_optimization(self, job_id: str, project_ids: List[str], 
                                        brand_ids: List[str], optimization_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run portfolio optimization in background"""
        try:
            # Update job status
            self.active_jobs[job_id]['status'] = 'processing'
            
            # Load projects and brands from database
            # In a real implementation, this would query the database
            # For now, we'll simulate the data loading
            projects = []  # Would load from database
            brands = []    # Would load from database
            
            # Create optimizer
            optimizer = GeneticPortfolioOptimizer(
                projects=projects,
                brands=brands,
                config=self.analytics_config['genetic_algorithm']
            )
            
            # Update configuration with user parameters
            if optimization_params:
                optimizer.update_config(optimization_params)
            
            # Run optimization in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                optimizer.optimize_portfolio
            )
            
            # Store results
            analysis_result = {
                'job_id': job_id,
                'type': 'portfolio_optimization',
                'project_ids': project_ids,
                'brand_ids': brand_ids,
                'optimization_result': {
                    'best_fitness': result.best_chromosome.fitness_score,
                    'portfolio_impact': result.best_chromosome.portfolio_impact,
                    'synergy_score': result.best_chromosome.synergy_score,
                    'feasibility_score': result.best_chromosome.feasibility_score,
                    'brand_recommendations': result.brand_recommendations,
                    'convergence_metrics': result.convergence_metrics,
                    'execution_summary': result.execution_summary
                },
                'completion_time': datetime.utcnow().isoformat()
            }
            
            # Cache results
            await self.cache_set(f"analysis_result:{job_id}", analysis_result, ttl=86400)  # 24 hours
            
            # Update job status
            self.active_jobs[job_id]['status'] = 'completed'
            self.active_jobs[job_id]['result'] = analysis_result
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization error for job {job_id}: {e}")
            
            # Update job status
            self.active_jobs[job_id]['status'] = 'failed'
            self.active_jobs[job_id]['error'] = str(e)
            
            raise MCPServerError(f"Portfolio optimization failed: {e}")
    
    async def _handle_multi_brand_optimization(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle multi-brand optimization with cross-brand synergies"""
        params = request.params
        
        # Validate parameters
        if 'brand_groups' not in params:
            raise ValidationError("Missing brand_groups parameter")
        
        brand_groups = params['brand_groups']
        project_id = params.get('project_id')
        optimization_params = params.get('optimization_params', {})
        
        # Create job ID
        job_id = f"multi_brand_opt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request.request_id[:8]}"
        
        # Start optimization
        task = asyncio.create_task(
            self._run_multi_brand_optimization(job_id, brand_groups, project_id, optimization_params)
        )
        
        self.active_jobs[job_id] = {
            'task': task,
            'type': 'multi_brand_optimization',
            'status': 'running',
            'start_time': datetime.utcnow(),
            'brand_groups': brand_groups,
            'project_id': project_id,
            'user_id': request.user_id
        }
        
        return {
            'job_id': job_id,
            'status': 'started',
            'message': 'Multi-brand optimization started',
            'brand_groups_count': len(brand_groups)
        }
    
    async def _run_multi_brand_optimization(self, job_id: str, brand_groups: List[List[str]], 
                                          project_id: Optional[str], optimization_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-brand optimization with synergy analysis"""
        try:
            self.active_jobs[job_id]['status'] = 'processing'
            
            results = {}
            
            # Optimize each brand group
            for i, brand_group in enumerate(brand_groups):
                group_id = f"group_{i+1}"
                
                # Run optimization for this brand group
                # This would use the genetic optimizer with cross-brand synergy calculations
                group_result = {
                    'brand_ids': brand_group,
                    'optimization_score': np.random.uniform(0.7, 0.95),  # Simulated
                    'synergy_score': np.random.uniform(0.6, 0.9),        # Simulated
                    'recommendations': [
                        {
                            'brand_id': brand_id,
                            'improvement_potential': np.random.uniform(0.1, 0.3),
                            'priority_metrics': ['engagement', 'reach', 'conversion']
                        }
                        for brand_id in brand_group
                    ]
                }
                
                results[group_id] = group_result
            
            # Calculate cross-group synergies
            cross_group_synergies = self._calculate_cross_group_synergies(brand_groups)
            
            analysis_result = {
                'job_id': job_id,
                'type': 'multi_brand_optimization',
                'brand_groups': brand_groups,
                'project_id': project_id,
                'group_results': results,
                'cross_group_synergies': cross_group_synergies,
                'overall_portfolio_score': np.mean([r['optimization_score'] for r in results.values()]),
                'completion_time': datetime.utcnow().isoformat()
            }
            
            # Cache and update job
            await self.cache_set(f"analysis_result:{job_id}", analysis_result, ttl=86400)
            self.active_jobs[job_id]['status'] = 'completed'
            self.active_jobs[job_id]['result'] = analysis_result
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Multi-brand optimization error for job {job_id}: {e}")
            self.active_jobs[job_id]['status'] = 'failed'
            self.active_jobs[job_id]['error'] = str(e)
            raise MCPServerError(f"Multi-brand optimization failed: {e}")
    
    def _calculate_cross_group_synergies(self, brand_groups: List[List[str]]) -> Dict[str, Any]:
        """Calculate synergies between brand groups"""
        synergies = {}
        
        for i, group1 in enumerate(brand_groups):
            for j, group2 in enumerate(brand_groups[i+1:], i+1):
                synergy_key = f"group_{i+1}_group_{j+1}"
                
                # Calculate synergy score (simplified)
                # In practice, this would analyze shared metrics, market overlap, etc.
                synergy_score = np.random.uniform(0.2, 0.8)
                
                synergies[synergy_key] = {
                    'synergy_score': synergy_score,
                    'shared_metrics': np.random.randint(3, 8),
                    'market_overlap': np.random.uniform(0.1, 0.6),
                    'collaboration_potential': 'high' if synergy_score > 0.6 else 'medium' if synergy_score > 0.4 else 'low'
                }
        
        return synergies
    
    # SHAP Analysis Handlers
    async def _handle_shap_analysis(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle SHAP analysis requests"""
        params = request.params
        
        required_params = ['project_ids', 'brand_ids', 'target_metric']
        for param in required_params:
            if param not in params:
                raise ValidationError(f"Missing required parameter: {param}")
        
        project_ids = params['project_ids']
        brand_ids = params['brand_ids']
        target_metric = params['target_metric']
        shap_params = params.get('shap_params', {})
        
        job_id = f"shap_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request.request_id[:8]}"
        
        task = asyncio.create_task(
            self._run_shap_analysis(job_id, project_ids, brand_ids, target_metric, shap_params)
        )
        
        self.active_jobs[job_id] = {
            'task': task,
            'type': 'shap_analysis',
            'status': 'running',
            'start_time': datetime.utcnow(),
            'project_ids': project_ids,
            'brand_ids': brand_ids,
            'target_metric': target_metric,
            'user_id': request.user_id
        }
        
        return {
            'job_id': job_id,
            'status': 'started',
            'message': 'SHAP analysis started',
            'target_metric': target_metric
        }
    
    async def _run_shap_analysis(self, job_id: str, project_ids: List[str], brand_ids: List[str], 
                               target_metric: str, shap_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run SHAP analysis for feature attribution"""
        try:
            self.active_jobs[job_id]['status'] = 'processing'
            
            # Simulate SHAP analysis results
            # In practice, this would use the SHAPPortfolioAnalyzer
            shap_values = {}
            feature_importance = {}
            
            for brand_id in brand_ids:
                # Simulate SHAP values for each brand
                features = ['engagement', 'reach', 'conversion', 'sentiment', 'share_of_voice']
                shap_values[brand_id] = {
                    feature: np.random.uniform(-0.5, 0.5) for feature in features
                }
                
                # Calculate feature importance
                feature_importance[brand_id] = {
                    feature: abs(value) for feature, value in shap_values[brand_id].items()
                }
            
            # Global feature importance
            global_importance = {}
            for feature in features:
                global_importance[feature] = np.mean([
                    feature_importance[brand_id][feature] for brand_id in brand_ids
                ])
            
            analysis_result = {
                'job_id': job_id,
                'type': 'shap_analysis',
                'project_ids': project_ids,
                'brand_ids': brand_ids,
                'target_metric': target_metric,
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'global_importance': global_importance,
                'top_features': sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:5],
                'completion_time': datetime.utcnow().isoformat()
            }
            
            await self.cache_set(f"analysis_result:{job_id}", analysis_result, ttl=86400)
            self.active_jobs[job_id]['status'] = 'completed'
            self.active_jobs[job_id]['result'] = analysis_result
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"SHAP analysis error for job {job_id}: {e}")
            self.active_jobs[job_id]['status'] = 'failed'
            self.active_jobs[job_id]['error'] = str(e)
            raise MCPServerError(f"SHAP analysis failed: {e}")
    
    # Job Management Handlers
    async def _handle_get_job_status(self, request: MCPRequest) -> Dict[str, Any]:
        """Get status of a specific job"""
        params = request.params
        
        if 'job_id' not in params:
            raise ValidationError("Missing job_id parameter")
        
        job_id = params['job_id']
        
        if job_id not in self.active_jobs:
            # Check cache for completed jobs
            cached_result = await self.cache_get(f"analysis_result:{job_id}")
            if cached_result:
                return {
                    'job_id': job_id,
                    'status': 'completed',
                    'result': cached_result
                }
            else:
                raise ValidationError(f"Job not found: {job_id}")
        
        job_info = self.active_jobs[job_id]
        
        response = {
            'job_id': job_id,
            'type': job_info['type'],
            'status': job_info['status'],
            'start_time': job_info['start_time'].isoformat(),
            'user_id': job_info['user_id']
        }
        
        if job_info['status'] == 'completed' and 'result' in job_info:
            response['result'] = job_info['result']
        elif job_info['status'] == 'failed' and 'error' in job_info:
            response['error'] = job_info['error']
        
        return response
    
    async def _handle_list_jobs(self, request: MCPRequest) -> Dict[str, Any]:
        """List all jobs for the user"""
        user_id = request.user_id
        
        user_jobs = []
        for job_id, job_info in self.active_jobs.items():
            if job_info['user_id'] == user_id:
                user_jobs.append({
                    'job_id': job_id,
                    'type': job_info['type'],
                    'status': job_info['status'],
                    'start_time': job_info['start_time'].isoformat()
                })
        
        return {
            'jobs': user_jobs,
            'total_jobs': len(user_jobs)
        }
    
    async def _handle_cancel_job(self, request: MCPRequest) -> Dict[str, Any]:
        """Cancel a running job"""
        params = request.params
        
        if 'job_id' not in params:
            raise ValidationError("Missing job_id parameter")
        
        job_id = params['job_id']
        
        if job_id not in self.active_jobs:
            raise ValidationError(f"Job not found: {job_id}")
        
        job_info = self.active_jobs[job_id]
        
        # Check if user owns the job or has admin access
        if job_info['user_id'] != request.user_id and request.authorization_level not in ['admin', 'system']:
            raise AuthorizationError("Cannot cancel job owned by another user")
        
        if job_info['status'] in ['completed', 'failed', 'cancelled']:
            return {
                'job_id': job_id,
                'status': job_info['status'],
                'message': f"Job already {job_info['status']}"
            }
        
        # Cancel the task
        if 'task' in job_info:
            job_info['task'].cancel()
        
        job_info['status'] = 'cancelled'
        
        return {
            'job_id': job_id,
            'status': 'cancelled',
            'message': 'Job cancelled successfully'
        }
    
    # Configuration Handlers
    async def _handle_get_config(self, request: MCPRequest) -> Dict[str, Any]:
        """Get current analytics configuration"""
        return {
            'analytics_config': self.analytics_config,
            'server_config': {
                'max_connections': self.config.max_connections,
                'timeout': self.config.timeout,
                'thread_pool_workers': self.thread_pool._max_workers
            }
        }
    
    async def _handle_update_config(self, request: MCPRequest) -> Dict[str, Any]:
        """Update analytics configuration"""
        params = request.params
        
        if 'config_updates' not in params:
            raise ValidationError("Missing config_updates parameter")
        
        config_updates = params['config_updates']
        
        # Update configuration
        for section, updates in config_updates.items():
            if section in self.analytics_config:
                self.analytics_config[section].update(updates)
        
        return {
            'message': 'Configuration updated successfully',
            'updated_config': self.analytics_config
        }
    
    # Placeholder handlers for other analysis types
    async def _handle_gap_analysis(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle gap analysis requests"""
        # Implementation would go here
        return {'message': 'Gap analysis not yet implemented'}
    
    async def _handle_correlation_analysis(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle correlation analysis requests"""
        # Implementation would go here
        return {'message': 'Correlation analysis not yet implemented'}
    
    async def _handle_scenario_analysis(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle scenario analysis requests"""
        # Implementation would go here
        return {'message': 'Scenario analysis not yet implemented'}
    
    async def _handle_batch_analysis(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle batch analysis requests"""
        # Implementation would go here
        return {'message': 'Batch analysis not yet implemented'}
    
    # Additional placeholder handlers
    async def _handle_cross_project_optimization(self, request: MCPRequest) -> Dict[str, Any]:
        return {'message': 'Cross-project optimization not yet implemented'}
    
    async def _handle_multi_brand_shap(self, request: MCPRequest) -> Dict[str, Any]:
        return {'message': 'Multi-brand SHAP not yet implemented'}
    
    async def _handle_attribution_analysis(self, request: MCPRequest) -> Dict[str, Any]:
        return {'message': 'Attribution analysis not yet implemented'}
    
    async def _handle_competitive_gap_analysis(self, request: MCPRequest) -> Dict[str, Any]:
        return {'message': 'Competitive gap analysis not yet implemented'}
    
    async def _handle_portfolio_gap_analysis(self, request: MCPRequest) -> Dict[str, Any]:
        return {'message': 'Portfolio gap analysis not yet implemented'}
    
    async def _handle_cross_brand_correlation(self, request: MCPRequest) -> Dict[str, Any]:
        return {'message': 'Cross-brand correlation not yet implemented'}
    
    async def _handle_cross_project_correlation(self, request: MCPRequest) -> Dict[str, Any]:
        return {'message': 'Cross-project correlation not yet implemented'}
    
    async def _handle_what_if_analysis(self, request: MCPRequest) -> Dict[str, Any]:
        return {'message': 'What-if analysis not yet implemented'}
    
    async def _handle_sensitivity_analysis(self, request: MCPRequest) -> Dict[str, Any]:
        return {'message': 'Sensitivity analysis not yet implemented'}
    
    async def _handle_scheduled_analysis(self, request: MCPRequest) -> Dict[str, Any]:
        return {'message': 'Scheduled analysis not yet implemented'}
    
    async def cleanup_resources(self):
        """Cleanup server-specific resources"""
        # Cancel all active jobs
        for job_id, job_info in self.active_jobs.items():
            if 'task' in job_info and not job_info['task'].done():
                job_info['task'].cancel()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        self.logger.info("Analysis MCP Server resources cleaned up")

# Server factory function
def create_analysis_server(host: str = "0.0.0.0", port: int = 8001, **kwargs) -> AnalysisMCPServer:
    """Create and configure Analysis MCP Server"""
    config = MCPServerConfig(
        server_type=MCPServerType.ANALYSIS,
        host=host,
        port=port,
        **kwargs
    )
    
    server = AnalysisMCPServer(config)
    return server

# Main entry point for running the server
async def main():
    """Main entry point for Analysis MCP Server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analysis MCP Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8001, help='Port to bind to')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start server
    server = create_analysis_server(host=args.host, port=args.port, log_level=args.log_level)
    await server.initialize_server_specific_handlers()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logging.info("Shutting down Analysis MCP Server...")
        await server.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

