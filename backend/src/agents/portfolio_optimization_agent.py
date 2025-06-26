"""
Portfolio Optimization Agent for Digi-Cadence Portfolio Management Platform
Intelligent agent responsible for optimizing portfolio performance across multiple brands and projects
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import json

from src.agents.base_agent import BaseAgent, AgentCapability, AgentTask, TaskPriority
from src.models.portfolio import Project, Brand, Organization

class PortfolioOptimizationAgent(BaseAgent):
    """
    Intelligent agent for portfolio optimization
    Uses genetic algorithms and advanced analytics to optimize portfolio performance
    """
    
    def __init__(self, agent_id: str = "portfolio_optimizer", config: Dict[str, Any] = None):
        # Default configuration
        default_config = {
            'optimization_frequency': 'daily',  # daily, weekly, monthly
            'optimization_objectives': ['roi', 'reach', 'engagement', 'cost_efficiency'],
            'constraint_types': ['budget', 'resource', 'brand_guidelines'],
            'genetic_algorithm_config': {
                'population_size': 100,
                'generations': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'elite_size': 10
            },
            'optimization_scope': 'portfolio',  # portfolio, brand, project
            'risk_tolerance': 'medium',  # low, medium, high
            'rebalancing_threshold': 0.15,  # 15% deviation triggers rebalancing
            'min_improvement_threshold': 0.05,  # 5% minimum improvement to implement changes
            'max_workers': 4,
            'optimization_timeout': 3600,  # 1 hour timeout
            'enable_real_time_optimization': True,
            'enable_predictive_optimization': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            agent_id=agent_id,
            name="Portfolio Optimization Agent",
            capabilities=[
                AgentCapability.OPTIMIZATION,
                AgentCapability.ANALYSIS,
                AgentCapability.FORECASTING,
                AgentCapability.STRATEGY
            ],
            config=default_config
        )
        
        # Optimization state
        self.current_portfolio_state = {}
        self.optimization_history = []
        self.active_optimizations = {}
        self.optimization_results = {}
        
        # Performance tracking
        self.baseline_metrics = {}
        self.optimization_impact = {}
        
        # Constraints and objectives
        self.constraints = {}
        self.objectives = {}
        
        # Register message handlers
        self._register_message_handlers()
        
        self.logger.info("Portfolio Optimization Agent initialized")
    
    def get_required_config_keys(self) -> List[str]:
        """Return required configuration keys"""
        return ['optimization_frequency', 'optimization_objectives']
    
    def _register_message_handlers(self):
        """Register message handlers for inter-agent communication"""
        self.register_message_handler('optimization_request', self._handle_optimization_request)
        self.register_message_handler('constraint_update', self._handle_constraint_update)
        self.register_message_handler('objective_update', self._handle_objective_update)
        self.register_message_handler('performance_alert', self._handle_performance_alert)
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process optimization tasks"""
        task_type = task.task_type
        parameters = task.parameters
        
        self.logger.info(f"Processing task: {task_type}")
        
        try:
            if task_type == 'portfolio_optimization':
                return await self._optimize_portfolio(parameters)
            elif task_type == 'brand_optimization':
                return await self._optimize_brand(parameters)
            elif task_type == 'project_optimization':
                return await self._optimize_project(parameters)
            elif task_type == 'constraint_optimization':
                return await self._optimize_with_constraints(parameters)
            elif task_type == 'multi_objective_optimization':
                return await self._multi_objective_optimization(parameters)
            elif task_type == 'rebalancing_analysis':
                return await self._analyze_rebalancing_needs(parameters)
            elif task_type == 'performance_optimization':
                return await self._optimize_performance_metrics(parameters)
            elif task_type == 'predictive_optimization':
                return await self._predictive_optimization(parameters)
            elif task_type == 'risk_optimization':
                return await self._optimize_risk_profile(parameters)
            elif task_type == 'resource_allocation':
                return await self._optimize_resource_allocation(parameters)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Task processing failed: {e}")
            raise
    
    async def _optimize_portfolio(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize entire portfolio performance"""
        
        optimization_id = parameters.get('optimization_id', f"opt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        scope = parameters.get('scope', 'all_brands')
        objectives = parameters.get('objectives', self.config['optimization_objectives'])
        constraints = parameters.get('constraints', {})
        
        self.logger.info(f"Starting portfolio optimization {optimization_id}")
        
        # Get current portfolio state
        portfolio_state = await self._get_portfolio_state(parameters)
        
        # Set up optimization problem
        optimization_config = self.config['genetic_algorithm_config'].copy()
        optimization_config.update(parameters.get('genetic_config', {}))
        
        # Run genetic algorithm optimization
        if self.genetic_optimizer:
            optimization_result = self.genetic_optimizer.optimize(
                optimization_type='portfolio_performance',
                objectives=objectives,
                constraints=constraints,
                **optimization_config
            )
        else:
            # Fallback optimization using analytical methods
            optimization_result = await self._analytical_optimization(portfolio_state, objectives, constraints)
        
        # Analyze optimization results
        analysis_result = await self._analyze_optimization_results(optimization_result, portfolio_state)
        
        # Generate recommendations
        recommendations = await self._generate_optimization_recommendations(analysis_result)
        
        # Calculate expected impact
        expected_impact = await self._calculate_expected_impact(optimization_result, portfolio_state)
        
        # Store optimization results
        self.optimization_results[optimization_id] = {
            'optimization_result': optimization_result,
            'analysis_result': analysis_result,
            'recommendations': recommendations,
            'expected_impact': expected_impact,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'completed'
        }
        
        # Update optimization history
        self.optimization_history.append({
            'optimization_id': optimization_id,
            'type': 'portfolio_optimization',
            'objectives': objectives,
            'timestamp': datetime.utcnow().isoformat(),
            'success': True,
            'improvement_achieved': expected_impact.get('total_improvement', 0)
        })
        
        result = {
            'optimization_id': optimization_id,
            'status': 'completed',
            'optimization_result': optimization_result,
            'analysis': analysis_result,
            'recommendations': recommendations,
            'expected_impact': expected_impact,
            'implementation_priority': self._calculate_implementation_priority(expected_impact),
            'next_optimization_date': self._calculate_next_optimization_date()
        }
        
        # Emit optimization completed event
        await self.emit_event('optimization_completed', {
            'optimization_id': optimization_id,
            'type': 'portfolio_optimization',
            'improvement': expected_impact.get('total_improvement', 0)
        })
        
        return result
    
    async def _optimize_brand(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize specific brand performance"""
        
        brand_id = parameters.get('brand_id')
        brand_name = parameters.get('brand_name')
        objectives = parameters.get('objectives', ['engagement', 'reach', 'conversion'])
        
        if not brand_id and not brand_name:
            raise ValueError("Either brand_id or brand_name must be provided")
        
        self.logger.info(f"Optimizing brand: {brand_name or brand_id}")
        
        # Get brand-specific data
        brand_data = await self._get_brand_data(brand_id, brand_name)
        
        # Run brand-specific optimization
        if self.genetic_optimizer:
            optimization_result = self.genetic_optimizer.optimize(
                optimization_type='brand_performance',
                brand_filter={'brand_id': brand_id, 'brand_name': brand_name},
                objectives=objectives,
                **self.config['genetic_algorithm_config']
            )
        else:
            optimization_result = await self._analytical_brand_optimization(brand_data, objectives)
        
        # Analyze brand optimization results
        analysis_result = await self._analyze_brand_optimization(optimization_result, brand_data)
        
        # Generate brand-specific recommendations
        recommendations = await self._generate_brand_recommendations(analysis_result, brand_data)
        
        return {
            'brand_id': brand_id,
            'brand_name': brand_name,
            'optimization_result': optimization_result,
            'analysis': analysis_result,
            'recommendations': recommendations,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _optimize_project(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize specific project performance"""
        
        project_id = parameters.get('project_id')
        project_name = parameters.get('project_name')
        objectives = parameters.get('objectives', ['roi', 'efficiency', 'reach'])
        
        if not project_id and not project_name:
            raise ValueError("Either project_id or project_name must be provided")
        
        self.logger.info(f"Optimizing project: {project_name or project_id}")
        
        # Get project-specific data
        project_data = await self._get_project_data(project_id, project_name)
        
        # Run project-specific optimization
        if self.genetic_optimizer:
            optimization_result = self.genetic_optimizer.optimize(
                optimization_type='project_performance',
                project_filter={'project_id': project_id, 'project_name': project_name},
                objectives=objectives,
                **self.config['genetic_algorithm_config']
            )
        else:
            optimization_result = await self._analytical_project_optimization(project_data, objectives)
        
        # Analyze project optimization results
        analysis_result = await self._analyze_project_optimization(optimization_result, project_data)
        
        # Generate project-specific recommendations
        recommendations = await self._generate_project_recommendations(analysis_result, project_data)
        
        return {
            'project_id': project_id,
            'project_name': project_name,
            'optimization_result': optimization_result,
            'analysis': analysis_result,
            'recommendations': recommendations,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _optimize_with_constraints(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio with specific constraints"""
        
        constraints = parameters.get('constraints', {})
        objectives = parameters.get('objectives', self.config['optimization_objectives'])
        
        self.logger.info(f"Optimizing with constraints: {list(constraints.keys())}")
        
        # Validate constraints
        validated_constraints = await self._validate_constraints(constraints)
        
        # Get current portfolio state
        portfolio_state = await self._get_portfolio_state(parameters)
        
        # Run constrained optimization
        if self.genetic_optimizer:
            optimization_result = self.genetic_optimizer.optimize(
                optimization_type='constrained_optimization',
                objectives=objectives,
                constraints=validated_constraints,
                **self.config['genetic_algorithm_config']
            )
        else:
            optimization_result = await self._analytical_constrained_optimization(
                portfolio_state, objectives, validated_constraints
            )
        
        # Analyze constraint satisfaction
        constraint_analysis = await self._analyze_constraint_satisfaction(optimization_result, validated_constraints)
        
        return {
            'optimization_result': optimization_result,
            'constraint_analysis': constraint_analysis,
            'constraints_satisfied': constraint_analysis.get('all_satisfied', False),
            'constraint_violations': constraint_analysis.get('violations', []),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _multi_objective_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multi-objective optimization with Pareto frontier analysis"""
        
        objectives = parameters.get('objectives', self.config['optimization_objectives'])
        weights = parameters.get('weights', {})
        
        self.logger.info(f"Multi-objective optimization for: {objectives}")
        
        # Get portfolio state
        portfolio_state = await self._get_portfolio_state(parameters)
        
        # Run multi-objective optimization
        if self.genetic_optimizer:
            optimization_result = self.genetic_optimizer.optimize(
                optimization_type='multi_objective',
                objectives=objectives,
                objective_weights=weights,
                **self.config['genetic_algorithm_config']
            )
        else:
            optimization_result = await self._analytical_multi_objective_optimization(
                portfolio_state, objectives, weights
            )
        
        # Analyze Pareto frontier
        pareto_analysis = await self._analyze_pareto_frontier(optimization_result, objectives)
        
        # Generate trade-off analysis
        tradeoff_analysis = await self._analyze_objective_tradeoffs(optimization_result, objectives)
        
        return {
            'optimization_result': optimization_result,
            'pareto_analysis': pareto_analysis,
            'tradeoff_analysis': tradeoff_analysis,
            'recommended_solution': pareto_analysis.get('recommended_solution'),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _analyze_rebalancing_needs(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio rebalancing needs"""
        
        threshold = parameters.get('threshold', self.config['rebalancing_threshold'])
        
        self.logger.info("Analyzing portfolio rebalancing needs")
        
        # Get current and target allocations
        current_allocation = await self._get_current_allocation()
        target_allocation = await self._get_target_allocation()
        
        # Calculate deviations
        deviations = {}
        rebalancing_needed = False
        
        for asset, current_weight in current_allocation.items():
            target_weight = target_allocation.get(asset, 0)
            deviation = abs(current_weight - target_weight)
            deviations[asset] = {
                'current_weight': current_weight,
                'target_weight': target_weight,
                'deviation': deviation,
                'deviation_percentage': deviation / target_weight if target_weight > 0 else 0,
                'needs_rebalancing': deviation > threshold
            }
            
            if deviation > threshold:
                rebalancing_needed = True
        
        # Generate rebalancing recommendations
        rebalancing_recommendations = []
        if rebalancing_needed:
            rebalancing_recommendations = await self._generate_rebalancing_recommendations(deviations)
        
        return {
            'rebalancing_needed': rebalancing_needed,
            'deviations': deviations,
            'threshold_used': threshold,
            'rebalancing_recommendations': rebalancing_recommendations,
            'estimated_impact': await self._estimate_rebalancing_impact(deviations),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _optimize_performance_metrics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize specific performance metrics"""
        
        target_metrics = parameters.get('target_metrics', ['roi', 'engagement_rate', 'conversion_rate'])
        improvement_targets = parameters.get('improvement_targets', {})
        
        self.logger.info(f"Optimizing performance metrics: {target_metrics}")
        
        # Get current metric performance
        current_performance = await self._get_current_metric_performance(target_metrics)
        
        # Run metric-specific optimization
        optimization_results = {}
        
        for metric in target_metrics:
            target_improvement = improvement_targets.get(metric, 0.1)  # 10% default improvement
            
            if self.genetic_optimizer:
                metric_optimization = self.genetic_optimizer.optimize(
                    optimization_type='metric_optimization',
                    target_metric=metric,
                    improvement_target=target_improvement,
                    **self.config['genetic_algorithm_config']
                )
            else:
                metric_optimization = await self._analytical_metric_optimization(
                    metric, current_performance[metric], target_improvement
                )
            
            optimization_results[metric] = metric_optimization
        
        # Analyze metric optimization results
        analysis_result = await self._analyze_metric_optimization_results(optimization_results, current_performance)
        
        return {
            'target_metrics': target_metrics,
            'current_performance': current_performance,
            'optimization_results': optimization_results,
            'analysis': analysis_result,
            'achievable_improvements': analysis_result.get('achievable_improvements', {}),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _predictive_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform predictive optimization based on forecasted trends"""
        
        forecast_horizon = parameters.get('forecast_horizon', 30)  # 30 days
        confidence_threshold = parameters.get('confidence_threshold', 0.8)
        
        self.logger.info(f"Predictive optimization for {forecast_horizon} days ahead")
        
        # Get trend forecasts
        if self.trend_analyzer:
            trend_forecasts = self.trend_analyzer.analyze(
                analysis_type='forecasting',
                forecast_horizon=forecast_horizon
            )
        else:
            trend_forecasts = await self._generate_simple_forecasts(forecast_horizon)
        
        # Filter high-confidence forecasts
        reliable_forecasts = await self._filter_reliable_forecasts(trend_forecasts, confidence_threshold)
        
        # Optimize based on predicted trends
        if self.genetic_optimizer:
            predictive_optimization = self.genetic_optimizer.optimize(
                optimization_type='predictive_optimization',
                trend_forecasts=reliable_forecasts,
                forecast_horizon=forecast_horizon,
                **self.config['genetic_algorithm_config']
            )
        else:
            predictive_optimization = await self._analytical_predictive_optimization(reliable_forecasts)
        
        # Analyze predictive optimization results
        analysis_result = await self._analyze_predictive_optimization(predictive_optimization, reliable_forecasts)
        
        return {
            'forecast_horizon': forecast_horizon,
            'trend_forecasts': trend_forecasts,
            'reliable_forecasts': reliable_forecasts,
            'predictive_optimization': predictive_optimization,
            'analysis': analysis_result,
            'confidence_score': analysis_result.get('confidence_score', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _optimize_risk_profile(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio risk profile"""
        
        target_risk_level = parameters.get('target_risk_level', self.config['risk_tolerance'])
        risk_metrics = parameters.get('risk_metrics', ['volatility', 'var', 'max_drawdown'])
        
        self.logger.info(f"Optimizing risk profile to {target_risk_level} risk level")
        
        # Get current risk profile
        current_risk_profile = await self._get_current_risk_profile(risk_metrics)
        
        # Run risk optimization
        if self.genetic_optimizer:
            risk_optimization = self.genetic_optimizer.optimize(
                optimization_type='risk_optimization',
                target_risk_level=target_risk_level,
                risk_metrics=risk_metrics,
                **self.config['genetic_algorithm_config']
            )
        else:
            risk_optimization = await self._analytical_risk_optimization(
                current_risk_profile, target_risk_level, risk_metrics
            )
        
        # Analyze risk optimization results
        analysis_result = await self._analyze_risk_optimization(risk_optimization, current_risk_profile)
        
        return {
            'target_risk_level': target_risk_level,
            'current_risk_profile': current_risk_profile,
            'risk_optimization': risk_optimization,
            'analysis': analysis_result,
            'risk_adjusted_recommendations': analysis_result.get('recommendations', []),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _optimize_resource_allocation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation across portfolio"""
        
        total_budget = parameters.get('total_budget')
        resource_constraints = parameters.get('resource_constraints', {})
        allocation_objectives = parameters.get('objectives', ['roi_maximization', 'risk_minimization'])
        
        self.logger.info(f"Optimizing resource allocation with budget: {total_budget}")
        
        # Get current resource allocation
        current_allocation = await self._get_current_resource_allocation()
        
        # Run resource allocation optimization
        if self.genetic_optimizer:
            allocation_optimization = self.genetic_optimizer.optimize(
                optimization_type='resource_allocation',
                total_budget=total_budget,
                resource_constraints=resource_constraints,
                objectives=allocation_objectives,
                **self.config['genetic_algorithm_config']
            )
        else:
            allocation_optimization = await self._analytical_resource_allocation(
                total_budget, resource_constraints, allocation_objectives
            )
        
        # Analyze allocation optimization results
        analysis_result = await self._analyze_allocation_optimization(allocation_optimization, current_allocation)
        
        return {
            'total_budget': total_budget,
            'current_allocation': current_allocation,
            'optimized_allocation': allocation_optimization,
            'analysis': analysis_result,
            'allocation_changes': analysis_result.get('allocation_changes', {}),
            'expected_roi_improvement': analysis_result.get('roi_improvement', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # Helper methods for data retrieval and analysis
    
    async def _get_portfolio_state(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get current portfolio state"""
        # This would typically query the database
        # For now, return simulated data
        return {
            'brands': ['Brand_A', 'Brand_B', 'Brand_C'],
            'projects': ['Project_1', 'Project_2', 'Project_3'],
            'current_metrics': {
                'total_roi': 3.2,
                'total_reach': 1500000,
                'average_engagement': 6.5,
                'total_budget': 500000,
                'cost_efficiency': 0.85
            },
            'allocations': {
                'Brand_A': 0.4,
                'Brand_B': 0.35,
                'Brand_C': 0.25
            }
        }
    
    async def _get_brand_data(self, brand_id: str, brand_name: str) -> Dict[str, Any]:
        """Get brand-specific data"""
        return {
            'brand_id': brand_id,
            'brand_name': brand_name,
            'current_metrics': {
                'roi': 3.5,
                'reach': 500000,
                'engagement_rate': 7.2,
                'conversion_rate': 4.1,
                'cost_per_acquisition': 25.0
            },
            'historical_performance': [],
            'budget_allocation': 150000
        }
    
    async def _get_project_data(self, project_id: str, project_name: str) -> Dict[str, Any]:
        """Get project-specific data"""
        return {
            'project_id': project_id,
            'project_name': project_name,
            'current_metrics': {
                'roi': 2.8,
                'efficiency': 0.82,
                'reach': 800000,
                'completion_rate': 0.75
            },
            'resource_allocation': {},
            'timeline': {}
        }
    
    async def _analytical_optimization(self, portfolio_state: Dict[str, Any], 
                                     objectives: List[str], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analytical optimization when genetic optimizer is not available"""
        
        # Simple analytical optimization
        current_metrics = portfolio_state.get('current_metrics', {})
        allocations = portfolio_state.get('allocations', {})
        
        # Calculate improvement suggestions based on objectives
        improvements = {}
        for objective in objectives:
            if objective == 'roi':
                improvements[objective] = {
                    'current_value': current_metrics.get('total_roi', 0),
                    'target_value': current_metrics.get('total_roi', 0) * 1.15,
                    'improvement_percentage': 15.0,
                    'recommended_actions': ['Increase high-ROI brand allocation', 'Optimize cost structure']
                }
            elif objective == 'reach':
                improvements[objective] = {
                    'current_value': current_metrics.get('total_reach', 0),
                    'target_value': current_metrics.get('total_reach', 0) * 1.20,
                    'improvement_percentage': 20.0,
                    'recommended_actions': ['Expand to new channels', 'Increase reach-focused campaigns']
                }
            elif objective == 'engagement':
                improvements[objective] = {
                    'current_value': current_metrics.get('average_engagement', 0),
                    'target_value': current_metrics.get('average_engagement', 0) * 1.10,
                    'improvement_percentage': 10.0,
                    'recommended_actions': ['Improve content quality', 'Optimize posting times']
                }
        
        return {
            'optimization_type': 'analytical',
            'objectives': objectives,
            'improvements': improvements,
            'recommended_allocations': allocations,  # Keep current for now
            'confidence_score': 0.7
        }
    
    async def _analyze_optimization_results(self, optimization_result: Dict[str, Any], 
                                          portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization results"""
        
        improvements = optimization_result.get('improvements', {})
        
        analysis = {
            'total_improvement_potential': 0,
            'objective_analysis': {},
            'feasibility_assessment': 'high',
            'implementation_complexity': 'medium',
            'expected_timeline': '2-4 weeks'
        }
        
        for objective, improvement_data in improvements.items():
            improvement_pct = improvement_data.get('improvement_percentage', 0)
            analysis['total_improvement_potential'] += improvement_pct
            
            analysis['objective_analysis'][objective] = {
                'improvement_percentage': improvement_pct,
                'feasibility': 'high' if improvement_pct < 20 else 'medium',
                'priority': 'high' if improvement_pct > 15 else 'medium'
            }
        
        analysis['total_improvement_potential'] /= len(improvements) if improvements else 1
        
        return analysis
    
    async def _generate_optimization_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable optimization recommendations"""
        
        recommendations = []
        objective_analysis = analysis_result.get('objective_analysis', {})
        
        for objective, analysis in objective_analysis.items():
            if analysis.get('priority') == 'high':
                recommendations.append({
                    'type': 'optimization',
                    'objective': objective,
                    'priority': 'high',
                    'action': f"Implement {objective} optimization strategies",
                    'expected_improvement': analysis.get('improvement_percentage', 0),
                    'timeline': '2-3 weeks',
                    'resources_required': 'medium'
                })
        
        # Add general recommendations
        recommendations.append({
            'type': 'monitoring',
            'priority': 'medium',
            'action': 'Implement continuous performance monitoring',
            'timeline': '1 week',
            'resources_required': 'low'
        })
        
        return recommendations
    
    async def _calculate_expected_impact(self, optimization_result: Dict[str, Any], 
                                       portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate expected impact of optimization"""
        
        improvements = optimization_result.get('improvements', {})
        current_metrics = portfolio_state.get('current_metrics', {})
        
        impact = {
            'total_improvement': 0,
            'metric_impacts': {},
            'financial_impact': 0,
            'risk_impact': 'neutral'
        }
        
        for objective, improvement_data in improvements.items():
            improvement_pct = improvement_data.get('improvement_percentage', 0)
            current_value = improvement_data.get('current_value', 0)
            
            impact['metric_impacts'][objective] = {
                'current_value': current_value,
                'improved_value': improvement_data.get('target_value', current_value),
                'absolute_improvement': improvement_data.get('target_value', current_value) - current_value,
                'percentage_improvement': improvement_pct
            }
            
            impact['total_improvement'] += improvement_pct
        
        impact['total_improvement'] /= len(improvements) if improvements else 1
        
        # Estimate financial impact
        if 'roi' in improvements:
            roi_improvement = improvements['roi'].get('improvement_percentage', 0)
            current_budget = current_metrics.get('total_budget', 0)
            impact['financial_impact'] = current_budget * (roi_improvement / 100)
        
        return impact
    
    def _calculate_implementation_priority(self, expected_impact: Dict[str, Any]) -> str:
        """Calculate implementation priority based on expected impact"""
        
        total_improvement = expected_impact.get('total_improvement', 0)
        financial_impact = expected_impact.get('financial_impact', 0)
        
        if total_improvement > 20 or financial_impact > 50000:
            return 'high'
        elif total_improvement > 10 or financial_impact > 20000:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_next_optimization_date(self) -> str:
        """Calculate when next optimization should be performed"""
        
        frequency = self.config['optimization_frequency']
        
        if frequency == 'daily':
            next_date = datetime.utcnow() + timedelta(days=1)
        elif frequency == 'weekly':
            next_date = datetime.utcnow() + timedelta(weeks=1)
        elif frequency == 'monthly':
            next_date = datetime.utcnow() + timedelta(days=30)
        else:
            next_date = datetime.utcnow() + timedelta(weeks=1)  # Default to weekly
        
        return next_date.isoformat()
    
    # Message handlers
    
    async def _handle_optimization_request(self, sender_id: str, data: Dict[str, Any]):
        """Handle optimization request from another agent"""
        
        request_type = data.get('request_type', 'portfolio_optimization')
        parameters = data.get('parameters', {})
        priority = TaskPriority(data.get('priority', TaskPriority.MEDIUM.value))
        
        # Add optimization task to queue
        task_id = await self.add_task(request_type, parameters, priority)
        
        # Send response back to requesting agent
        await self.send_message(sender_id, 'optimization_request_acknowledged', {
            'task_id': task_id,
            'estimated_completion': '30 minutes'
        })
    
    async def _handle_constraint_update(self, sender_id: str, data: Dict[str, Any]):
        """Handle constraint updates"""
        
        constraint_type = data.get('constraint_type')
        constraint_value = data.get('constraint_value')
        
        if constraint_type:
            self.constraints[constraint_type] = constraint_value
            self.logger.info(f"Updated constraint {constraint_type}: {constraint_value}")
    
    async def _handle_objective_update(self, sender_id: str, data: Dict[str, Any]):
        """Handle objective updates"""
        
        objective_type = data.get('objective_type')
        objective_config = data.get('objective_config')
        
        if objective_type:
            self.objectives[objective_type] = objective_config
            self.logger.info(f"Updated objective {objective_type}: {objective_config}")
    
    async def _handle_performance_alert(self, sender_id: str, data: Dict[str, Any]):
        """Handle performance alerts from monitoring agents"""
        
        alert_type = data.get('alert_type')
        severity = data.get('severity', 'medium')
        
        if severity == 'high':
            # Trigger immediate optimization
            await self.add_task('performance_optimization', data, TaskPriority.HIGH)
            self.logger.warning(f"High severity performance alert: {alert_type}")
        else:
            self.logger.info(f"Performance alert received: {alert_type}")
    
    # Additional helper methods (simplified implementations)
    
    async def _validate_constraints(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Validate optimization constraints"""
        # Simple validation - in practice would be more comprehensive
        return constraints
    
    async def _get_current_allocation(self) -> Dict[str, float]:
        """Get current portfolio allocation"""
        return {'Brand_A': 0.4, 'Brand_B': 0.35, 'Brand_C': 0.25}
    
    async def _get_target_allocation(self) -> Dict[str, float]:
        """Get target portfolio allocation"""
        return {'Brand_A': 0.45, 'Brand_B': 0.30, 'Brand_C': 0.25}
    
    async def _generate_rebalancing_recommendations(self, deviations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate rebalancing recommendations"""
        recommendations = []
        
        for asset, deviation_info in deviations.items():
            if deviation_info['needs_rebalancing']:
                action = 'increase' if deviation_info['current_weight'] < deviation_info['target_weight'] else 'decrease'
                recommendations.append({
                    'asset': asset,
                    'action': action,
                    'current_weight': deviation_info['current_weight'],
                    'target_weight': deviation_info['target_weight'],
                    'adjustment_needed': abs(deviation_info['current_weight'] - deviation_info['target_weight'])
                })
        
        return recommendations
    
    async def _estimate_rebalancing_impact(self, deviations: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate impact of rebalancing"""
        total_deviation = sum(abs(d['deviation']) for d in deviations.values())
        
        return {
            'total_deviation': total_deviation,
            'estimated_improvement': min(total_deviation * 0.5, 0.15),  # Cap at 15%
            'implementation_cost': total_deviation * 1000,  # Simplified cost estimate
            'expected_timeline': '1-2 weeks'
        }

# Factory function for creating portfolio optimization agent
def create_portfolio_optimization_agent(config: Dict[str, Any] = None) -> PortfolioOptimizationAgent:
    """Create Portfolio Optimization Agent with specified configuration"""
    return PortfolioOptimizationAgent(config=config)

