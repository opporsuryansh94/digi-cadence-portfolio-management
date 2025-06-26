"""
Multi-Brand Metric Optimization Agent for Digi-Cadence Portfolio Management Platform
Intelligent agent specialized in optimizing metrics across multiple brands simultaneously
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from itertools import combinations

from src.agents.base_agent import BaseAgent, AgentCapability, AgentTask, TaskPriority
from src.models.portfolio import Project, Brand, Organization

class MultiBrandMetricOptimizationAgent(BaseAgent):
    """
    Intelligent agent for multi-brand metric optimization
    Specializes in cross-brand analysis and synchronized optimization
    """
    
    def __init__(self, agent_id: str = "multi_brand_optimizer", config: Dict[str, Any] = None):
        # Default configuration
        default_config = {
            'optimization_scope': 'cross_brand',  # cross_brand, brand_cluster, all_brands
            'synchronization_strategy': 'coordinated',  # coordinated, independent, sequential
            'metric_priorities': {
                'engagement_rate': 0.25,
                'conversion_rate': 0.25,
                'reach': 0.20,
                'roi': 0.20,
                'brand_awareness': 0.10
            },
            'cross_brand_synergy_weight': 0.3,
            'brand_cannibalization_penalty': 0.2,
            'optimization_frequency': 'weekly',
            'min_brands_for_optimization': 2,
            'max_brands_per_optimization': 10,
            'correlation_threshold': 0.3,
            'synergy_detection_enabled': True,
            'cannibalization_detection_enabled': True,
            'brand_clustering_enabled': True,
            'real_time_adjustment_enabled': True,
            'cross_brand_learning_enabled': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            agent_id=agent_id,
            name="Multi-Brand Metric Optimization Agent",
            capabilities=[
                AgentCapability.OPTIMIZATION,
                AgentCapability.ANALYSIS,
                AgentCapability.STRATEGY,
                AgentCapability.MONITORING
            ],
            config=default_config
        )
        
        # Multi-brand optimization state
        self.brand_clusters = {}
        self.brand_correlations = {}
        self.synergy_matrix = {}
        self.cannibalization_matrix = {}
        self.optimization_history = {}
        
        # Cross-brand learning
        self.brand_performance_patterns = {}
        self.successful_strategies = {}
        self.failed_strategies = {}
        
        # Real-time monitoring
        self.metric_thresholds = {}
        self.alert_conditions = {}
        
        # Register message handlers
        self._register_message_handlers()
        
        self.logger.info("Multi-Brand Metric Optimization Agent initialized")
    
    def get_required_config_keys(self) -> List[str]:
        """Return required configuration keys"""
        return ['optimization_scope', 'metric_priorities']
    
    def _register_message_handlers(self):
        """Register message handlers for inter-agent communication"""
        self.register_message_handler('brand_performance_update', self._handle_brand_performance_update)
        self.register_message_handler('cross_brand_analysis_request', self._handle_cross_brand_analysis_request)
        self.register_message_handler('synergy_detection_request', self._handle_synergy_detection_request)
        self.register_message_handler('cannibalization_alert', self._handle_cannibalization_alert)
        self.register_message_handler('metric_threshold_breach', self._handle_metric_threshold_breach)
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process multi-brand optimization tasks"""
        task_type = task.task_type
        parameters = task.parameters
        
        self.logger.info(f"Processing task: {task_type}")
        
        try:
            if task_type == 'multi_brand_optimization':
                return await self._optimize_multiple_brands(parameters)
            elif task_type == 'cross_brand_synergy_analysis':
                return await self._analyze_cross_brand_synergies(parameters)
            elif task_type == 'brand_cannibalization_analysis':
                return await self._analyze_brand_cannibalization(parameters)
            elif task_type == 'brand_clustering':
                return await self._perform_brand_clustering(parameters)
            elif task_type == 'coordinated_metric_optimization':
                return await self._coordinated_metric_optimization(parameters)
            elif task_type == 'cross_brand_learning':
                return await self._cross_brand_learning_analysis(parameters)
            elif task_type == 'brand_portfolio_rebalancing':
                return await self._brand_portfolio_rebalancing(parameters)
            elif task_type == 'metric_synchronization':
                return await self._synchronize_brand_metrics(parameters)
            elif task_type == 'competitive_brand_positioning':
                return await self._optimize_competitive_positioning(parameters)
            elif task_type == 'brand_lifecycle_optimization':
                return await self._optimize_brand_lifecycle_stages(parameters)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Task processing failed: {e}")
            raise
    
    async def _optimize_multiple_brands(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize metrics across multiple brands simultaneously"""
        
        brand_ids = parameters.get('brand_ids', [])
        target_metrics = parameters.get('target_metrics', list(self.config['metric_priorities'].keys()))
        optimization_strategy = parameters.get('strategy', self.config['synchronization_strategy'])
        
        if len(brand_ids) < self.config['min_brands_for_optimization']:
            raise ValueError(f"Minimum {self.config['min_brands_for_optimization']} brands required for optimization")
        
        self.logger.info(f"Optimizing {len(brand_ids)} brands with strategy: {optimization_strategy}")
        
        # Get current brand performance data
        brand_data = await self._get_multi_brand_data(brand_ids, target_metrics)
        
        # Analyze cross-brand relationships
        relationships = await self._analyze_brand_relationships(brand_data)
        
        # Detect synergies and cannibalization
        synergies = await self._detect_brand_synergies(brand_data, relationships)
        cannibalization = await self._detect_brand_cannibalization(brand_data, relationships)
        
        # Perform optimization based on strategy
        if optimization_strategy == 'coordinated':
            optimization_result = await self._coordinated_brand_optimization(
                brand_data, target_metrics, synergies, cannibalization
            )
        elif optimization_strategy == 'sequential':
            optimization_result = await self._sequential_brand_optimization(
                brand_data, target_metrics, synergies, cannibalization
            )
        else:  # independent
            optimization_result = await self._independent_brand_optimization(
                brand_data, target_metrics
            )
        
        # Analyze optimization results
        analysis_result = await self._analyze_multi_brand_optimization_results(
            optimization_result, brand_data, relationships
        )
        
        # Generate implementation plan
        implementation_plan = await self._generate_multi_brand_implementation_plan(
            optimization_result, analysis_result
        )
        
        # Update optimization history
        optimization_id = f"multi_brand_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.optimization_history[optimization_id] = {
            'brand_ids': brand_ids,
            'target_metrics': target_metrics,
            'strategy': optimization_strategy,
            'optimization_result': optimization_result,
            'analysis_result': analysis_result,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        result = {
            'optimization_id': optimization_id,
            'brand_ids': brand_ids,
            'target_metrics': target_metrics,
            'optimization_strategy': optimization_strategy,
            'brand_relationships': relationships,
            'synergies_detected': synergies,
            'cannibalization_detected': cannibalization,
            'optimization_result': optimization_result,
            'analysis': analysis_result,
            'implementation_plan': implementation_plan,
            'expected_impact': analysis_result.get('expected_impact', {}),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Emit optimization completed event
        await self.emit_event('multi_brand_optimization_completed', {
            'optimization_id': optimization_id,
            'brands_optimized': len(brand_ids),
            'total_improvement': analysis_result.get('total_improvement', 0)
        })
        
        return result
    
    async def _analyze_cross_brand_synergies(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze synergies between brands"""
        
        brand_ids = parameters.get('brand_ids', [])
        analysis_period = parameters.get('analysis_period', 90)  # days
        
        self.logger.info(f"Analyzing cross-brand synergies for {len(brand_ids)} brands")
        
        # Get brand performance data
        brand_data = await self._get_multi_brand_data(brand_ids, period_days=analysis_period)
        
        # Calculate correlation matrix
        correlation_matrix = await self._calculate_brand_correlation_matrix(brand_data)
        
        # Identify synergy opportunities
        synergy_opportunities = await self._identify_synergy_opportunities(correlation_matrix, brand_data)
        
        # Analyze synergy types
        synergy_types = await self._classify_synergy_types(synergy_opportunities, brand_data)
        
        # Calculate synergy potential
        synergy_potential = await self._calculate_synergy_potential(synergy_opportunities, brand_data)
        
        # Generate synergy recommendations
        synergy_recommendations = await self._generate_synergy_recommendations(
            synergy_opportunities, synergy_types, synergy_potential
        )
        
        return {
            'brand_ids': brand_ids,
            'analysis_period': analysis_period,
            'correlation_matrix': correlation_matrix,
            'synergy_opportunities': synergy_opportunities,
            'synergy_types': synergy_types,
            'synergy_potential': synergy_potential,
            'recommendations': synergy_recommendations,
            'total_synergy_score': synergy_potential.get('total_score', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _analyze_brand_cannibalization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cannibalization between brands"""
        
        brand_ids = parameters.get('brand_ids', [])
        analysis_period = parameters.get('analysis_period', 90)
        cannibalization_threshold = parameters.get('threshold', self.config['cannibalization_detection_enabled'])
        
        self.logger.info(f"Analyzing brand cannibalization for {len(brand_ids)} brands")
        
        # Get brand performance data
        brand_data = await self._get_multi_brand_data(brand_ids, period_days=analysis_period)
        
        # Detect cannibalization patterns
        cannibalization_patterns = await self._detect_cannibalization_patterns(brand_data)
        
        # Calculate cannibalization impact
        cannibalization_impact = await self._calculate_cannibalization_impact(cannibalization_patterns, brand_data)
        
        # Identify at-risk brand pairs
        at_risk_pairs = await self._identify_at_risk_brand_pairs(cannibalization_impact, cannibalization_threshold)
        
        # Generate mitigation strategies
        mitigation_strategies = await self._generate_cannibalization_mitigation_strategies(
            at_risk_pairs, cannibalization_impact
        )
        
        return {
            'brand_ids': brand_ids,
            'analysis_period': analysis_period,
            'cannibalization_patterns': cannibalization_patterns,
            'cannibalization_impact': cannibalization_impact,
            'at_risk_pairs': at_risk_pairs,
            'mitigation_strategies': mitigation_strategies,
            'total_cannibalization_risk': cannibalization_impact.get('total_risk_score', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _perform_brand_clustering(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform brand clustering based on performance characteristics"""
        
        brand_ids = parameters.get('brand_ids', [])
        clustering_features = parameters.get('features', ['engagement_rate', 'conversion_rate', 'reach', 'roi'])
        num_clusters = parameters.get('num_clusters', 'auto')
        
        self.logger.info(f"Performing brand clustering for {len(brand_ids)} brands")
        
        # Get brand feature data
        brand_features = await self._get_brand_features(brand_ids, clustering_features)
        
        # Perform clustering analysis
        if self.correlation_analyzer:
            clustering_result = self.correlation_analyzer.analyze(
                analysis_type='clustering',
                features=clustering_features,
                num_clusters=num_clusters
            )
        else:
            clustering_result = await self._simple_brand_clustering(brand_features, num_clusters)
        
        # Analyze cluster characteristics
        cluster_analysis = await self._analyze_brand_clusters(clustering_result, brand_features)
        
        # Generate cluster-based strategies
        cluster_strategies = await self._generate_cluster_strategies(cluster_analysis)
        
        # Update brand clusters
        self.brand_clusters = clustering_result.get('clusters', {})
        
        return {
            'brand_ids': brand_ids,
            'clustering_features': clustering_features,
            'clustering_result': clustering_result,
            'cluster_analysis': cluster_analysis,
            'cluster_strategies': cluster_strategies,
            'optimal_clusters': clustering_result.get('optimal_clusters', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _coordinated_metric_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform coordinated optimization across brands for specific metrics"""
        
        brand_ids = parameters.get('brand_ids', [])
        target_metric = parameters.get('target_metric')
        coordination_strategy = parameters.get('coordination_strategy', 'balanced')
        
        if not target_metric:
            raise ValueError("Target metric must be specified for coordinated optimization")
        
        self.logger.info(f"Coordinated {target_metric} optimization for {len(brand_ids)} brands")
        
        # Get current metric performance
        current_performance = await self._get_metric_performance(brand_ids, target_metric)
        
        # Analyze metric interdependencies
        interdependencies = await self._analyze_metric_interdependencies(brand_ids, target_metric)
        
        # Optimize metric coordination
        if coordination_strategy == 'balanced':
            optimization_result = await self._balanced_metric_optimization(
                current_performance, interdependencies, target_metric
            )
        elif coordination_strategy == 'leader_follower':
            optimization_result = await self._leader_follower_optimization(
                current_performance, interdependencies, target_metric
            )
        else:  # competitive
            optimization_result = await self._competitive_metric_optimization(
                current_performance, interdependencies, target_metric
            )
        
        # Calculate coordination benefits
        coordination_benefits = await self._calculate_coordination_benefits(
            optimization_result, current_performance
        )
        
        return {
            'brand_ids': brand_ids,
            'target_metric': target_metric,
            'coordination_strategy': coordination_strategy,
            'current_performance': current_performance,
            'interdependencies': interdependencies,
            'optimization_result': optimization_result,
            'coordination_benefits': coordination_benefits,
            'total_improvement': coordination_benefits.get('total_improvement', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _cross_brand_learning_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze learning opportunities across brands"""
        
        brand_ids = parameters.get('brand_ids', [])
        learning_focus = parameters.get('learning_focus', 'best_practices')
        
        self.logger.info(f"Cross-brand learning analysis for {len(brand_ids)} brands")
        
        # Identify high-performing brands
        high_performers = await self._identify_high_performing_brands(brand_ids)
        
        # Extract successful strategies
        successful_strategies = await self._extract_successful_strategies(high_performers)
        
        # Identify learning opportunities
        learning_opportunities = await self._identify_learning_opportunities(
            brand_ids, successful_strategies
        )
        
        # Generate knowledge transfer recommendations
        transfer_recommendations = await self._generate_knowledge_transfer_recommendations(
            learning_opportunities, successful_strategies
        )
        
        # Update learning database
        await self._update_cross_brand_learning_database(successful_strategies, learning_opportunities)
        
        return {
            'brand_ids': brand_ids,
            'learning_focus': learning_focus,
            'high_performers': high_performers,
            'successful_strategies': successful_strategies,
            'learning_opportunities': learning_opportunities,
            'transfer_recommendations': transfer_recommendations,
            'potential_impact': learning_opportunities.get('potential_impact', {}),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _brand_portfolio_rebalancing(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Rebalance brand portfolio for optimal performance"""
        
        brand_ids = parameters.get('brand_ids', [])
        rebalancing_objectives = parameters.get('objectives', ['roi_maximization', 'risk_minimization'])
        total_budget = parameters.get('total_budget')
        
        self.logger.info(f"Brand portfolio rebalancing for {len(brand_ids)} brands")
        
        # Get current brand allocations
        current_allocations = await self._get_current_brand_allocations(brand_ids)
        
        # Analyze brand performance efficiency
        efficiency_analysis = await self._analyze_brand_efficiency(brand_ids)
        
        # Calculate optimal allocations
        optimal_allocations = await self._calculate_optimal_brand_allocations(
            efficiency_analysis, rebalancing_objectives, total_budget
        )
        
        # Analyze rebalancing impact
        rebalancing_impact = await self._analyze_rebalancing_impact(
            current_allocations, optimal_allocations, efficiency_analysis
        )
        
        # Generate rebalancing plan
        rebalancing_plan = await self._generate_brand_rebalancing_plan(
            current_allocations, optimal_allocations, rebalancing_impact
        )
        
        return {
            'brand_ids': brand_ids,
            'rebalancing_objectives': rebalancing_objectives,
            'total_budget': total_budget,
            'current_allocations': current_allocations,
            'optimal_allocations': optimal_allocations,
            'rebalancing_impact': rebalancing_impact,
            'rebalancing_plan': rebalancing_plan,
            'expected_improvement': rebalancing_impact.get('expected_improvement', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _synchronize_brand_metrics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize metrics across brands for coordinated performance"""
        
        brand_ids = parameters.get('brand_ids', [])
        sync_metrics = parameters.get('sync_metrics', ['engagement_rate', 'conversion_rate'])
        sync_strategy = parameters.get('sync_strategy', 'harmonized')
        
        self.logger.info(f"Synchronizing {len(sync_metrics)} metrics across {len(brand_ids)} brands")
        
        # Get current metric values
        current_metrics = await self._get_current_brand_metrics(brand_ids, sync_metrics)
        
        # Calculate synchronization targets
        sync_targets = await self._calculate_synchronization_targets(
            current_metrics, sync_strategy
        )
        
        # Generate synchronization plan
        sync_plan = await self._generate_synchronization_plan(
            current_metrics, sync_targets, sync_strategy
        )
        
        # Analyze synchronization benefits
        sync_benefits = await self._analyze_synchronization_benefits(
            current_metrics, sync_targets
        )
        
        return {
            'brand_ids': brand_ids,
            'sync_metrics': sync_metrics,
            'sync_strategy': sync_strategy,
            'current_metrics': current_metrics,
            'sync_targets': sync_targets,
            'sync_plan': sync_plan,
            'sync_benefits': sync_benefits,
            'coordination_score': sync_benefits.get('coordination_score', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # Helper methods for data retrieval and analysis
    
    async def _get_multi_brand_data(self, brand_ids: List[str], target_metrics: List[str] = None, 
                                   period_days: int = 90) -> Dict[str, Any]:
        """Get performance data for multiple brands"""
        
        # Simulate multi-brand data
        brand_data = {}
        
        for brand_id in brand_ids:
            brand_data[brand_id] = {
                'brand_id': brand_id,
                'brand_name': f"Brand_{brand_id}",
                'metrics': {
                    'engagement_rate': np.random.uniform(5.0, 8.0),
                    'conversion_rate': np.random.uniform(3.0, 6.0),
                    'reach': np.random.uniform(100000, 500000),
                    'roi': np.random.uniform(2.5, 4.5),
                    'brand_awareness': np.random.uniform(60, 85),
                    'cost_per_acquisition': np.random.uniform(20, 40)
                },
                'historical_data': self._generate_historical_data(period_days),
                'budget_allocation': np.random.uniform(50000, 200000)
            }
        
        return brand_data
    
    def _generate_historical_data(self, period_days: int) -> List[Dict[str, Any]]:
        """Generate historical performance data"""
        
        historical_data = []
        base_date = datetime.utcnow() - timedelta(days=period_days)
        
        for i in range(period_days):
            date = base_date + timedelta(days=i)
            
            # Add some trend and seasonality
            trend_factor = 1 + (i / period_days) * 0.1  # 10% growth over period
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly seasonality
            
            historical_data.append({
                'date': date.isoformat(),
                'engagement_rate': np.random.uniform(5.0, 8.0) * trend_factor * seasonal_factor,
                'conversion_rate': np.random.uniform(3.0, 6.0) * trend_factor,
                'reach': np.random.uniform(100000, 500000) * trend_factor,
                'roi': np.random.uniform(2.5, 4.5) * trend_factor
            })
        
        return historical_data
    
    async def _analyze_brand_relationships(self, brand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between brands"""
        
        relationships = {
            'correlations': {},
            'dependencies': {},
            'interactions': {}
        }
        
        brand_ids = list(brand_data.keys())
        
        # Calculate pairwise correlations
        for i, brand1 in enumerate(brand_ids):
            for brand2 in brand_ids[i+1:]:
                correlation = await self._calculate_brand_correlation(
                    brand_data[brand1], brand_data[brand2]
                )
                relationships['correlations'][f"{brand1}_{brand2}"] = correlation
        
        # Analyze dependencies
        for brand_id in brand_ids:
            dependencies = await self._analyze_brand_dependencies(brand_id, brand_data)
            relationships['dependencies'][brand_id] = dependencies
        
        return relationships
    
    async def _calculate_brand_correlation(self, brand1_data: Dict[str, Any], 
                                         brand2_data: Dict[str, Any]) -> float:
        """Calculate correlation between two brands"""
        
        # Simple correlation based on metrics
        metrics1 = brand1_data['metrics']
        metrics2 = brand2_data['metrics']
        
        # Calculate correlation for common metrics
        correlations = []
        for metric in metrics1:
            if metric in metrics2:
                # Simplified correlation calculation
                corr = np.random.uniform(-0.5, 0.8)  # Simulate correlation
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    async def _detect_brand_synergies(self, brand_data: Dict[str, Any], 
                                    relationships: Dict[str, Any]) -> Dict[str, Any]:
        """Detect synergies between brands"""
        
        synergies = {}
        correlations = relationships.get('correlations', {})
        
        for pair, correlation in correlations.items():
            if correlation > self.config['correlation_threshold']:
                brand1, brand2 = pair.split('_')
                synergies[pair] = {
                    'brand1': brand1,
                    'brand2': brand2,
                    'correlation': correlation,
                    'synergy_type': 'positive_correlation',
                    'synergy_strength': min(correlation, 1.0),
                    'potential_benefit': correlation * 0.1  # 10% of correlation as benefit
                }
        
        return synergies
    
    async def _detect_brand_cannibalization(self, brand_data: Dict[str, Any], 
                                          relationships: Dict[str, Any]) -> Dict[str, Any]:
        """Detect cannibalization between brands"""
        
        cannibalization = {}
        correlations = relationships.get('correlations', {})
        
        for pair, correlation in correlations.items():
            if correlation < -self.config['correlation_threshold']:
                brand1, brand2 = pair.split('_')
                cannibalization[pair] = {
                    'brand1': brand1,
                    'brand2': brand2,
                    'correlation': correlation,
                    'cannibalization_type': 'negative_correlation',
                    'cannibalization_strength': abs(correlation),
                    'potential_loss': abs(correlation) * 0.05  # 5% of correlation as loss
                }
        
        return cannibalization
    
    async def _coordinated_brand_optimization(self, brand_data: Dict[str, Any], 
                                            target_metrics: List[str],
                                            synergies: Dict[str, Any], 
                                            cannibalization: Dict[str, Any]) -> Dict[str, Any]:
        """Perform coordinated optimization across brands"""
        
        optimization_result = {
            'optimization_type': 'coordinated',
            'brand_optimizations': {},
            'synergy_utilization': {},
            'cannibalization_mitigation': {}
        }
        
        # Optimize each brand considering synergies and cannibalization
        for brand_id, brand_info in brand_data.items():
            current_metrics = brand_info['metrics']
            
            # Calculate optimization targets considering relationships
            optimization_targets = {}
            for metric in target_metrics:
                if metric in current_metrics:
                    base_improvement = 0.1  # 10% base improvement
                    
                    # Adjust for synergies
                    synergy_boost = self._calculate_synergy_boost(brand_id, synergies)
                    
                    # Adjust for cannibalization
                    cannibalization_penalty = self._calculate_cannibalization_penalty(brand_id, cannibalization)
                    
                    total_improvement = base_improvement + synergy_boost - cannibalization_penalty
                    
                    optimization_targets[metric] = {
                        'current_value': current_metrics[metric],
                        'target_value': current_metrics[metric] * (1 + total_improvement),
                        'improvement_percentage': total_improvement * 100,
                        'synergy_contribution': synergy_boost * 100,
                        'cannibalization_impact': cannibalization_penalty * 100
                    }
            
            optimization_result['brand_optimizations'][brand_id] = optimization_targets
        
        return optimization_result
    
    def _calculate_synergy_boost(self, brand_id: str, synergies: Dict[str, Any]) -> float:
        """Calculate synergy boost for a brand"""
        
        total_boost = 0.0
        
        for pair, synergy_info in synergies.items():
            if brand_id in pair:
                synergy_strength = synergy_info.get('synergy_strength', 0)
                potential_benefit = synergy_info.get('potential_benefit', 0)
                total_boost += synergy_strength * potential_benefit
        
        return min(total_boost, 0.2)  # Cap at 20% boost
    
    def _calculate_cannibalization_penalty(self, brand_id: str, cannibalization: Dict[str, Any]) -> float:
        """Calculate cannibalization penalty for a brand"""
        
        total_penalty = 0.0
        
        for pair, cannib_info in cannibalization.items():
            if brand_id in pair:
                cannib_strength = cannib_info.get('cannibalization_strength', 0)
                potential_loss = cannib_info.get('potential_loss', 0)
                total_penalty += cannib_strength * potential_loss
        
        return min(total_penalty, 0.15)  # Cap at 15% penalty
    
    # Message handlers
    
    async def _handle_brand_performance_update(self, sender_id: str, data: Dict[str, Any]):
        """Handle brand performance updates"""
        
        brand_id = data.get('brand_id')
        performance_data = data.get('performance_data', {})
        
        if brand_id:
            # Update brand performance patterns
            if brand_id not in self.brand_performance_patterns:
                self.brand_performance_patterns[brand_id] = []
            
            self.brand_performance_patterns[brand_id].append({
                'timestamp': datetime.utcnow().isoformat(),
                'performance_data': performance_data
            })
            
            # Trigger optimization if significant change detected
            if self._is_significant_performance_change(performance_data):
                await self.add_task('multi_brand_optimization', {
                    'brand_ids': [brand_id],
                    'trigger': 'performance_change'
                }, TaskPriority.MEDIUM)
    
    async def _handle_cross_brand_analysis_request(self, sender_id: str, data: Dict[str, Any]):
        """Handle cross-brand analysis requests"""
        
        analysis_type = data.get('analysis_type', 'synergy_analysis')
        brand_ids = data.get('brand_ids', [])
        
        # Add analysis task to queue
        task_id = await self.add_task(analysis_type, data, TaskPriority.MEDIUM)
        
        # Send acknowledgment
        await self.send_message(sender_id, 'analysis_request_acknowledged', {
            'task_id': task_id,
            'analysis_type': analysis_type,
            'estimated_completion': '15 minutes'
        })
    
    async def _handle_synergy_detection_request(self, sender_id: str, data: Dict[str, Any]):
        """Handle synergy detection requests"""
        
        brand_ids = data.get('brand_ids', [])
        
        # Perform immediate synergy analysis
        synergy_result = await self._analyze_cross_brand_synergies({'brand_ids': brand_ids})
        
        # Send results back
        await self.send_message(sender_id, 'synergy_analysis_result', synergy_result)
    
    async def _handle_cannibalization_alert(self, sender_id: str, data: Dict[str, Any]):
        """Handle cannibalization alerts"""
        
        brand_pair = data.get('brand_pair', [])
        severity = data.get('severity', 'medium')
        
        if severity == 'high':
            # Trigger immediate cannibalization analysis
            await self.add_task('brand_cannibalization_analysis', {
                'brand_ids': brand_pair,
                'priority': 'high'
            }, TaskPriority.HIGH)
        
        self.logger.warning(f"Cannibalization alert for brands {brand_pair}: {severity}")
    
    async def _handle_metric_threshold_breach(self, sender_id: str, data: Dict[str, Any]):
        """Handle metric threshold breaches"""
        
        brand_id = data.get('brand_id')
        metric = data.get('metric')
        threshold_type = data.get('threshold_type', 'lower')
        
        # Trigger coordinated metric optimization
        await self.add_task('coordinated_metric_optimization', {
            'brand_ids': [brand_id],
            'target_metric': metric,
            'trigger': 'threshold_breach'
        }, TaskPriority.HIGH)
        
        self.logger.warning(f"Metric threshold breach for {brand_id}: {metric} ({threshold_type})")
    
    def _is_significant_performance_change(self, performance_data: Dict[str, Any]) -> bool:
        """Check if performance change is significant enough to trigger optimization"""
        
        # Simple threshold-based check
        for metric, value in performance_data.items():
            if isinstance(value, (int, float)):
                if abs(value) > 0.1:  # 10% change threshold
                    return True
        
        return False

# Factory function for creating multi-brand metric optimization agent
def create_multi_brand_metric_optimization_agent(config: Dict[str, Any] = None) -> MultiBrandMetricOptimizationAgent:
    """Create Multi-Brand Metric Optimization Agent with specified configuration"""
    return MultiBrandMetricOptimizationAgent(config=config)

