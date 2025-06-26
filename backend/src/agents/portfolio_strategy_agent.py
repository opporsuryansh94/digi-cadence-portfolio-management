"""
Portfolio Strategy Agent for Digi-Cadence Portfolio Management Platform
Intelligent agent specialized in strategic planning and decision-making for portfolio management
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from enum import Enum

from src.agents.base_agent import BaseAgent, AgentCapability, AgentTask, TaskPriority
from src.models.portfolio import Project, Brand, Organization

class StrategyType(Enum):
    """Strategy type enumeration"""
    GROWTH = "growth"
    OPTIMIZATION = "optimization"
    DIVERSIFICATION = "diversification"
    CONSOLIDATION = "consolidation"
    INNOVATION = "innovation"
    DEFENSIVE = "defensive"

class StrategyPriority(Enum):
    """Strategy priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class PortfolioStrategyAgent(BaseAgent):
    """
    Intelligent agent for portfolio strategy development and execution
    Specializes in strategic planning, decision-making, and strategic optimization
    """
    
    def __init__(self, agent_id: str = "portfolio_strategist", config: Dict[str, Any] = None):
        # Default configuration
        default_config = {
            'strategy_horizons': [30, 90, 180, 365],  # days
            'strategy_types': ['growth', 'optimization', 'diversification', 'consolidation'],
            'decision_frameworks': ['swot', 'porter_five_forces', 'bcg_matrix', 'ansoff_matrix'],
            'strategic_objectives': ['market_share_growth', 'profitability', 'brand_equity', 'innovation'],
            'risk_appetite': 'medium',  # low, medium, high
            'strategic_review_frequency': 'monthly',
            'competitive_analysis_enabled': True,
            'market_opportunity_analysis_enabled': True,
            'resource_optimization_enabled': True,
            'strategic_alignment_threshold': 0.8,
            'implementation_tracking_enabled': True,
            'adaptive_strategy_enabled': True,
            'stakeholder_analysis_enabled': True,
            'scenario_planning_enabled': True,
            'strategic_kpi_monitoring': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            agent_id=agent_id,
            name="Portfolio Strategy Agent",
            capabilities=[
                AgentCapability.STRATEGY,
                AgentCapability.ANALYSIS,
                AgentCapability.OPTIMIZATION,
                AgentCapability.MONITORING
            ],
            config=default_config
        )
        
        # Strategy state
        self.active_strategies = {}
        self.strategy_history = {}
        self.strategic_objectives = {}
        self.implementation_plans = {}
        
        # Decision-making frameworks
        self.decision_frameworks = {}
        self.strategic_models = {}
        
        # Competitive intelligence
        self.competitive_landscape = {}
        self.market_opportunities = {}
        self.threat_assessments = {}
        
        # Strategic KPIs
        self.strategic_kpis = {}
        self.kpi_targets = {}
        self.performance_tracking = {}
        
        # Stakeholder analysis
        self.stakeholder_map = {}
        self.stakeholder_priorities = {}
        
        # Register message handlers
        self._register_message_handlers()
        
        self.logger.info("Portfolio Strategy Agent initialized")
    
    def get_required_config_keys(self) -> List[str]:
        """Return required configuration keys"""
        return ['strategy_horizons', 'strategic_objectives']
    
    def _register_message_handlers(self):
        """Register message handlers for inter-agent communication"""
        self.register_message_handler('strategy_request', self._handle_strategy_request)
        self.register_message_handler('competitive_intelligence', self._handle_competitive_intelligence)
        self.register_message_handler('market_opportunity', self._handle_market_opportunity)
        self.register_message_handler('strategic_kpi_update', self._handle_strategic_kpi_update)
        self.register_message_handler('implementation_update', self._handle_implementation_update)
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process strategy tasks"""
        task_type = task.task_type
        parameters = task.parameters
        
        self.logger.info(f"Processing task: {task_type}")
        
        try:
            if task_type == 'portfolio_strategy_development':
                return await self._develop_portfolio_strategy(parameters)
            elif task_type == 'competitive_analysis':
                return await self._conduct_competitive_analysis(parameters)
            elif task_type == 'market_opportunity_analysis':
                return await self._analyze_market_opportunities(parameters)
            elif task_type == 'strategic_planning':
                return await self._create_strategic_plan(parameters)
            elif task_type == 'strategy_optimization':
                return await self._optimize_strategy(parameters)
            elif task_type == 'implementation_planning':
                return await self._create_implementation_plan(parameters)
            elif task_type == 'strategic_review':
                return await self._conduct_strategic_review(parameters)
            elif task_type == 'scenario_planning':
                return await self._conduct_scenario_planning(parameters)
            elif task_type == 'stakeholder_analysis':
                return await self._conduct_stakeholder_analysis(parameters)
            elif task_type == 'strategic_alignment_assessment':
                return await self._assess_strategic_alignment(parameters)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Task processing failed: {e}")
            raise
    
    async def _develop_portfolio_strategy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Develop comprehensive portfolio strategy"""
        
        strategy_horizon = parameters.get('strategy_horizon', 365)
        strategic_objectives = parameters.get('strategic_objectives', self.config['strategic_objectives'])
        strategy_type = parameters.get('strategy_type', StrategyType.OPTIMIZATION.value)
        
        self.logger.info(f"Developing {strategy_type} strategy for {strategy_horizon} days")
        
        # Conduct situational analysis
        situational_analysis = await self._conduct_situational_analysis(parameters)
        
        # Analyze competitive landscape
        competitive_analysis = await self._conduct_competitive_analysis(parameters)
        
        # Identify market opportunities
        market_opportunities = await self._analyze_market_opportunities(parameters)
        
        # Assess internal capabilities
        capability_assessment = await self._assess_internal_capabilities(parameters)
        
        # Develop strategic options
        strategic_options = await self._develop_strategic_options(
            situational_analysis, competitive_analysis, market_opportunities, capability_assessment
        )
        
        # Evaluate and select strategy
        strategy_evaluation = await self._evaluate_strategic_options(strategic_options, strategic_objectives)
        selected_strategy = await self._select_optimal_strategy(strategy_evaluation)
        
        # Create implementation roadmap
        implementation_roadmap = await self._create_implementation_roadmap(selected_strategy, strategy_horizon)
        
        # Define success metrics
        success_metrics = await self._define_success_metrics(selected_strategy, strategic_objectives)
        
        # Risk assessment
        risk_assessment = await self._assess_strategic_risks(selected_strategy, implementation_roadmap)
        
        # Store strategy
        strategy_id = f"strategy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.active_strategies[strategy_id] = {
            'strategy_type': strategy_type,
            'strategy_horizon': strategy_horizon,
            'strategic_objectives': strategic_objectives,
            'selected_strategy': selected_strategy,
            'implementation_roadmap': implementation_roadmap,
            'success_metrics': success_metrics,
            'risk_assessment': risk_assessment,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'active'
        }
        
        result = {
            'strategy_id': strategy_id,
            'strategy_type': strategy_type,
            'strategy_horizon': strategy_horizon,
            'strategic_objectives': strategic_objectives,
            'situational_analysis': situational_analysis,
            'competitive_analysis': competitive_analysis,
            'market_opportunities': market_opportunities,
            'capability_assessment': capability_assessment,
            'strategic_options': strategic_options,
            'strategy_evaluation': strategy_evaluation,
            'selected_strategy': selected_strategy,
            'implementation_roadmap': implementation_roadmap,
            'success_metrics': success_metrics,
            'risk_assessment': risk_assessment,
            'strategic_priority': self._calculate_strategic_priority(selected_strategy),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Emit strategy development completed event
        await self.emit_event('strategy_development_completed', {
            'strategy_id': strategy_id,
            'strategy_type': strategy_type,
            'strategic_priority': result['strategic_priority']
        })
        
        return result
    
    async def _conduct_competitive_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive competitive analysis"""
        
        analysis_scope = parameters.get('analysis_scope', 'market_leaders')
        competitive_factors = parameters.get('competitive_factors', [
            'market_share', 'brand_strength', 'innovation', 'pricing', 'distribution'
        ])
        
        self.logger.info(f"Conducting competitive analysis: {analysis_scope}")
        
        # Identify key competitors
        key_competitors = await self._identify_key_competitors(analysis_scope)
        
        # Analyze competitive positioning
        competitive_positioning = await self._analyze_competitive_positioning(key_competitors, competitive_factors)
        
        # Assess competitive strengths and weaknesses
        competitive_swot = await self._conduct_competitive_swot(key_competitors)
        
        # Analyze competitive strategies
        competitive_strategies = await self._analyze_competitive_strategies(key_competitors)
        
        # Identify competitive gaps and opportunities
        competitive_gaps = await self._identify_competitive_gaps(competitive_positioning, competitive_swot)
        
        # Develop competitive response strategies
        response_strategies = await self._develop_competitive_response_strategies(
            competitive_gaps, competitive_strategies
        )
        
        return {
            'analysis_scope': analysis_scope,
            'competitive_factors': competitive_factors,
            'key_competitors': key_competitors,
            'competitive_positioning': competitive_positioning,
            'competitive_swot': competitive_swot,
            'competitive_strategies': competitive_strategies,
            'competitive_gaps': competitive_gaps,
            'response_strategies': response_strategies,
            'competitive_threat_level': self._assess_competitive_threat_level(competitive_positioning),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _analyze_market_opportunities(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market opportunities for portfolio growth"""
        
        market_segments = parameters.get('market_segments', ['existing', 'adjacent', 'new'])
        opportunity_types = parameters.get('opportunity_types', [
            'market_expansion', 'product_innovation', 'channel_development', 'partnership'
        ])
        
        self.logger.info(f"Analyzing market opportunities across {len(market_segments)} segments")
        
        # Market size and growth analysis
        market_analysis = await self._conduct_market_analysis(market_segments)
        
        # Identify opportunity areas
        opportunity_areas = await self._identify_opportunity_areas(market_analysis, opportunity_types)
        
        # Assess opportunity attractiveness
        opportunity_assessment = await self._assess_opportunity_attractiveness(opportunity_areas)
        
        # Evaluate market entry barriers
        entry_barriers = await self._evaluate_market_entry_barriers(opportunity_areas)
        
        # Prioritize opportunities
        opportunity_prioritization = await self._prioritize_opportunities(
            opportunity_assessment, entry_barriers
        )
        
        # Develop market entry strategies
        market_entry_strategies = await self._develop_market_entry_strategies(opportunity_prioritization)
        
        return {
            'market_segments': market_segments,
            'opportunity_types': opportunity_types,
            'market_analysis': market_analysis,
            'opportunity_areas': opportunity_areas,
            'opportunity_assessment': opportunity_assessment,
            'entry_barriers': entry_barriers,
            'opportunity_prioritization': opportunity_prioritization,
            'market_entry_strategies': market_entry_strategies,
            'total_opportunity_value': opportunity_assessment.get('total_value', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _create_strategic_plan(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive strategic plan"""
        
        planning_horizon = parameters.get('planning_horizon', 365)
        strategic_themes = parameters.get('strategic_themes', [
            'growth', 'efficiency', 'innovation', 'customer_experience'
        ])
        
        self.logger.info(f"Creating strategic plan for {planning_horizon} days")
        
        # Define strategic vision and mission
        strategic_vision = await self._define_strategic_vision(parameters)
        
        # Set strategic goals and objectives
        strategic_goals = await self._set_strategic_goals(strategic_themes, planning_horizon)
        
        # Develop strategic initiatives
        strategic_initiatives = await self._develop_strategic_initiatives(strategic_goals)
        
        # Create resource allocation plan
        resource_allocation = await self._create_resource_allocation_plan(strategic_initiatives)
        
        # Develop timeline and milestones
        timeline_milestones = await self._develop_timeline_milestones(strategic_initiatives, planning_horizon)
        
        # Define governance structure
        governance_structure = await self._define_governance_structure(strategic_initiatives)
        
        # Create monitoring and control framework
        monitoring_framework = await self._create_monitoring_framework(strategic_goals, strategic_initiatives)
        
        return {
            'planning_horizon': planning_horizon,
            'strategic_themes': strategic_themes,
            'strategic_vision': strategic_vision,
            'strategic_goals': strategic_goals,
            'strategic_initiatives': strategic_initiatives,
            'resource_allocation': resource_allocation,
            'timeline_milestones': timeline_milestones,
            'governance_structure': governance_structure,
            'monitoring_framework': monitoring_framework,
            'plan_complexity_score': self._calculate_plan_complexity(strategic_initiatives),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _optimize_strategy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize existing strategy based on performance and market changes"""
        
        strategy_id = parameters.get('strategy_id')
        optimization_focus = parameters.get('optimization_focus', ['performance', 'efficiency', 'risk'])
        
        if not strategy_id or strategy_id not in self.active_strategies:
            raise ValueError("Valid strategy_id required for optimization")
        
        current_strategy = self.active_strategies[strategy_id]
        
        self.logger.info(f"Optimizing strategy {strategy_id}")
        
        # Analyze current strategy performance
        performance_analysis = await self._analyze_strategy_performance(current_strategy)
        
        # Identify optimization opportunities
        optimization_opportunities = await self._identify_optimization_opportunities(
            current_strategy, performance_analysis, optimization_focus
        )
        
        # Develop optimization recommendations
        optimization_recommendations = await self._develop_optimization_recommendations(
            optimization_opportunities
        )
        
        # Assess optimization impact
        optimization_impact = await self._assess_optimization_impact(
            current_strategy, optimization_recommendations
        )
        
        # Create optimized strategy
        optimized_strategy = await self._create_optimized_strategy(
            current_strategy, optimization_recommendations
        )
        
        # Update implementation roadmap
        updated_roadmap = await self._update_implementation_roadmap(
            current_strategy['implementation_roadmap'], optimization_recommendations
        )
        
        # Update strategy
        self.active_strategies[strategy_id].update({
            'selected_strategy': optimized_strategy,
            'implementation_roadmap': updated_roadmap,
            'optimization_history': self.active_strategies[strategy_id].get('optimization_history', []) + [{
                'timestamp': datetime.utcnow().isoformat(),
                'optimization_focus': optimization_focus,
                'optimization_impact': optimization_impact
            }],
            'last_optimized': datetime.utcnow().isoformat()
        })
        
        return {
            'strategy_id': strategy_id,
            'optimization_focus': optimization_focus,
            'performance_analysis': performance_analysis,
            'optimization_opportunities': optimization_opportunities,
            'optimization_recommendations': optimization_recommendations,
            'optimization_impact': optimization_impact,
            'optimized_strategy': optimized_strategy,
            'updated_roadmap': updated_roadmap,
            'improvement_score': optimization_impact.get('improvement_score', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _conduct_scenario_planning(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct scenario planning for strategic decision-making"""
        
        scenario_types = parameters.get('scenario_types', ['best_case', 'worst_case', 'most_likely'])
        planning_horizon = parameters.get('planning_horizon', 365)
        key_variables = parameters.get('key_variables', [
            'market_growth', 'competitive_intensity', 'economic_conditions', 'technology_disruption'
        ])
        
        self.logger.info(f"Conducting scenario planning for {len(scenario_types)} scenarios")
        
        # Define scenario parameters
        scenario_parameters = await self._define_scenario_parameters(scenario_types, key_variables)
        
        # Develop scenarios
        scenarios = {}
        for scenario_type in scenario_types:
            scenario = await self._develop_scenario(
                scenario_type, scenario_parameters[scenario_type], planning_horizon
            )
            scenarios[scenario_type] = scenario
        
        # Analyze scenario implications
        scenario_implications = await self._analyze_scenario_implications(scenarios)
        
        # Develop contingency plans
        contingency_plans = await self._develop_contingency_plans(scenarios, scenario_implications)
        
        # Assess scenario probabilities
        scenario_probabilities = await self._assess_scenario_probabilities(scenarios)
        
        # Create strategic options for each scenario
        strategic_options = await self._create_scenario_strategic_options(scenarios, contingency_plans)
        
        return {
            'scenario_types': scenario_types,
            'planning_horizon': planning_horizon,
            'key_variables': key_variables,
            'scenario_parameters': scenario_parameters,
            'scenarios': scenarios,
            'scenario_implications': scenario_implications,
            'contingency_plans': contingency_plans,
            'scenario_probabilities': scenario_probabilities,
            'strategic_options': strategic_options,
            'recommended_strategy': self._recommend_robust_strategy(strategic_options, scenario_probabilities),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _conduct_stakeholder_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive stakeholder analysis"""
        
        stakeholder_categories = parameters.get('stakeholder_categories', [
            'customers', 'employees', 'shareholders', 'partners', 'regulators', 'communities'
        ])
        
        self.logger.info(f"Conducting stakeholder analysis for {len(stakeholder_categories)} categories")
        
        # Identify key stakeholders
        key_stakeholders = await self._identify_key_stakeholders(stakeholder_categories)
        
        # Analyze stakeholder interests and influence
        stakeholder_mapping = await self._map_stakeholder_interests_influence(key_stakeholders)
        
        # Assess stakeholder relationships
        relationship_assessment = await self._assess_stakeholder_relationships(key_stakeholders)
        
        # Identify stakeholder expectations
        stakeholder_expectations = await self._identify_stakeholder_expectations(key_stakeholders)
        
        # Develop stakeholder engagement strategies
        engagement_strategies = await self._develop_stakeholder_engagement_strategies(
            stakeholder_mapping, relationship_assessment, stakeholder_expectations
        )
        
        # Create stakeholder communication plan
        communication_plan = await self._create_stakeholder_communication_plan(engagement_strategies)
        
        return {
            'stakeholder_categories': stakeholder_categories,
            'key_stakeholders': key_stakeholders,
            'stakeholder_mapping': stakeholder_mapping,
            'relationship_assessment': relationship_assessment,
            'stakeholder_expectations': stakeholder_expectations,
            'engagement_strategies': engagement_strategies,
            'communication_plan': communication_plan,
            'stakeholder_risk_score': self._calculate_stakeholder_risk_score(relationship_assessment),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # Helper methods for strategic analysis
    
    async def _conduct_situational_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive situational analysis"""
        
        # SWOT Analysis
        swot_analysis = await self._conduct_swot_analysis()
        
        # Porter's Five Forces
        five_forces = await self._conduct_five_forces_analysis()
        
        # Market analysis
        market_analysis = await self._conduct_market_analysis(['current_market'])
        
        # Internal capability analysis
        capability_analysis = await self._assess_internal_capabilities(parameters)
        
        return {
            'swot_analysis': swot_analysis,
            'five_forces_analysis': five_forces,
            'market_analysis': market_analysis,
            'capability_analysis': capability_analysis,
            'overall_situation_score': self._calculate_situation_score(swot_analysis, five_forces)
        }
    
    async def _conduct_swot_analysis(self) -> Dict[str, Any]:
        """Conduct SWOT analysis"""
        
        # Simulated SWOT analysis
        return {
            'strengths': [
                'Strong brand portfolio',
                'Advanced analytics capabilities',
                'Experienced team',
                'Technology infrastructure'
            ],
            'weaknesses': [
                'Limited market presence in new segments',
                'Resource constraints',
                'Process inefficiencies'
            ],
            'opportunities': [
                'Digital transformation trends',
                'Emerging markets',
                'New technology adoption',
                'Strategic partnerships'
            ],
            'threats': [
                'Increased competition',
                'Economic uncertainty',
                'Regulatory changes',
                'Technology disruption'
            ]
        }
    
    async def _conduct_five_forces_analysis(self) -> Dict[str, Any]:
        """Conduct Porter's Five Forces analysis"""
        
        return {
            'competitive_rivalry': {
                'intensity': 'high',
                'score': 4.2,
                'factors': ['Many competitors', 'Price competition', 'Innovation race']
            },
            'supplier_power': {
                'intensity': 'medium',
                'score': 3.1,
                'factors': ['Multiple suppliers available', 'Switching costs moderate']
            },
            'buyer_power': {
                'intensity': 'medium',
                'score': 3.5,
                'factors': ['Price sensitivity', 'Multiple options available']
            },
            'threat_of_substitutes': {
                'intensity': 'medium',
                'score': 3.0,
                'factors': ['Alternative solutions exist', 'Technology evolution']
            },
            'barriers_to_entry': {
                'intensity': 'medium',
                'score': 3.3,
                'factors': ['Capital requirements', 'Brand recognition needed']
            },
            'overall_attractiveness': 3.2
        }
    
    async def _develop_strategic_options(self, situational_analysis: Dict[str, Any],
                                       competitive_analysis: Dict[str, Any],
                                       market_opportunities: Dict[str, Any],
                                       capability_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Develop strategic options based on analysis"""
        
        strategic_options = []
        
        # Growth strategy option
        strategic_options.append({
            'strategy_type': StrategyType.GROWTH.value,
            'description': 'Aggressive market expansion and brand portfolio growth',
            'key_initiatives': [
                'New market entry',
                'Brand acquisition',
                'Product line extension',
                'Digital channel expansion'
            ],
            'resource_requirements': 'high',
            'risk_level': 'high',
            'expected_roi': 4.5,
            'implementation_complexity': 'high'
        })
        
        # Optimization strategy option
        strategic_options.append({
            'strategy_type': StrategyType.OPTIMIZATION.value,
            'description': 'Focus on operational efficiency and performance optimization',
            'key_initiatives': [
                'Process automation',
                'Cost optimization',
                'Performance improvement',
                'Resource reallocation'
            ],
            'resource_requirements': 'medium',
            'risk_level': 'low',
            'expected_roi': 3.2,
            'implementation_complexity': 'medium'
        })
        
        # Diversification strategy option
        strategic_options.append({
            'strategy_type': StrategyType.DIVERSIFICATION.value,
            'description': 'Diversify portfolio to reduce risk and capture new opportunities',
            'key_initiatives': [
                'New industry entry',
                'Technology diversification',
                'Geographic expansion',
                'Partnership development'
            ],
            'resource_requirements': 'high',
            'risk_level': 'medium',
            'expected_roi': 3.8,
            'implementation_complexity': 'high'
        })
        
        return strategic_options
    
    async def _evaluate_strategic_options(self, strategic_options: List[Dict[str, Any]],
                                        strategic_objectives: List[str]) -> Dict[str, Any]:
        """Evaluate strategic options against objectives"""
        
        evaluation_criteria = {
            'market_share_growth': 0.25,
            'profitability': 0.30,
            'brand_equity': 0.20,
            'innovation': 0.15,
            'risk_management': 0.10
        }
        
        option_scores = {}
        
        for option in strategic_options:
            strategy_type = option['strategy_type']
            
            # Calculate weighted score
            scores = {}
            for criterion, weight in evaluation_criteria.items():
                # Simulate scoring based on strategy type
                if strategy_type == StrategyType.GROWTH.value:
                    score = np.random.uniform(7, 9) if criterion in ['market_share_growth', 'innovation'] else np.random.uniform(5, 7)
                elif strategy_type == StrategyType.OPTIMIZATION.value:
                    score = np.random.uniform(8, 9) if criterion in ['profitability', 'risk_management'] else np.random.uniform(6, 8)
                else:  # DIVERSIFICATION
                    score = np.random.uniform(6, 8)
                
                scores[criterion] = score
            
            weighted_score = sum(scores[criterion] * evaluation_criteria[criterion] for criterion in scores)
            
            option_scores[strategy_type] = {
                'individual_scores': scores,
                'weighted_score': weighted_score,
                'ranking': 0  # Will be set after all scores calculated
            }
        
        # Rank options
        sorted_options = sorted(option_scores.items(), key=lambda x: x[1]['weighted_score'], reverse=True)
        for i, (strategy_type, score_info) in enumerate(sorted_options):
            option_scores[strategy_type]['ranking'] = i + 1
        
        return {
            'evaluation_criteria': evaluation_criteria,
            'option_scores': option_scores,
            'recommended_option': sorted_options[0][0],
            'score_summary': {
                'highest_score': sorted_options[0][1]['weighted_score'],
                'lowest_score': sorted_options[-1][1]['weighted_score'],
                'score_range': sorted_options[0][1]['weighted_score'] - sorted_options[-1][1]['weighted_score']
            }
        }
    
    def _calculate_strategic_priority(self, selected_strategy: Dict[str, Any]) -> str:
        """Calculate strategic priority based on strategy characteristics"""
        
        expected_roi = selected_strategy.get('expected_roi', 0)
        risk_level = selected_strategy.get('risk_level', 'medium')
        resource_requirements = selected_strategy.get('resource_requirements', 'medium')
        
        if expected_roi > 4.0 and risk_level == 'low':
            return StrategyPriority.CRITICAL.value
        elif expected_roi > 3.5 or risk_level == 'high':
            return StrategyPriority.HIGH.value
        elif expected_roi > 3.0:
            return StrategyPriority.MEDIUM.value
        else:
            return StrategyPriority.LOW.value
    
    # Message handlers
    
    async def _handle_strategy_request(self, sender_id: str, data: Dict[str, Any]):
        """Handle strategy development requests"""
        
        strategy_type = data.get('strategy_type', 'portfolio_strategy_development')
        parameters = data.get('parameters', {})
        priority = TaskPriority(data.get('priority', TaskPriority.MEDIUM.value))
        
        # Add strategy task to queue
        task_id = await self.add_task(strategy_type, parameters, priority)
        
        # Send acknowledgment
        await self.send_message(sender_id, 'strategy_request_acknowledged', {
            'task_id': task_id,
            'strategy_type': strategy_type,
            'estimated_completion': '45 minutes'
        })
    
    async def _handle_competitive_intelligence(self, sender_id: str, data: Dict[str, Any]):
        """Handle competitive intelligence updates"""
        
        competitor_id = data.get('competitor_id')
        intelligence_data = data.get('intelligence_data', {})
        
        if competitor_id:
            if competitor_id not in self.competitive_landscape:
                self.competitive_landscape[competitor_id] = {}
            
            self.competitive_landscape[competitor_id].update({
                'last_update': datetime.utcnow().isoformat(),
                'intelligence_data': intelligence_data,
                'source': sender_id
            })
            
            self.logger.info(f"Updated competitive intelligence for {competitor_id}")
    
    async def _handle_market_opportunity(self, sender_id: str, data: Dict[str, Any]):
        """Handle market opportunity notifications"""
        
        opportunity_id = data.get('opportunity_id')
        opportunity_data = data.get('opportunity_data', {})
        
        if opportunity_id:
            self.market_opportunities[opportunity_id] = {
                'opportunity_data': opportunity_data,
                'identified_by': sender_id,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'identified'
            }
            
            # Trigger opportunity analysis if high value
            opportunity_value = opportunity_data.get('estimated_value', 0)
            if opportunity_value > 100000:  # High-value opportunity
                await self.add_task('market_opportunity_analysis', {
                    'opportunity_id': opportunity_id,
                    'opportunity_data': opportunity_data
                }, TaskPriority.HIGH)
    
    async def _handle_strategic_kpi_update(self, sender_id: str, data: Dict[str, Any]):
        """Handle strategic KPI updates"""
        
        kpi_name = data.get('kpi_name')
        kpi_value = data.get('kpi_value')
        
        if kpi_name:
            self.strategic_kpis[kpi_name] = {
                'value': kpi_value,
                'timestamp': datetime.utcnow().isoformat(),
                'source': sender_id
            }
            
            # Check if KPI threshold breached
            target_value = self.kpi_targets.get(kpi_name, {}).get('target', 0)
            if target_value and abs(kpi_value - target_value) / target_value > 0.1:  # 10% deviation
                self.logger.warning(f"Strategic KPI {kpi_name} deviation detected: {kpi_value} vs target {target_value}")
    
    async def _handle_implementation_update(self, sender_id: str, data: Dict[str, Any]):
        """Handle strategy implementation updates"""
        
        strategy_id = data.get('strategy_id')
        implementation_status = data.get('implementation_status', {})
        
        if strategy_id and strategy_id in self.active_strategies:
            if 'implementation_tracking' not in self.active_strategies[strategy_id]:
                self.active_strategies[strategy_id]['implementation_tracking'] = []
            
            self.active_strategies[strategy_id]['implementation_tracking'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'status_update': implementation_status,
                'reported_by': sender_id
            })
            
            self.logger.info(f"Implementation update received for strategy {strategy_id}")

# Factory function for creating portfolio strategy agent
def create_portfolio_strategy_agent(config: Dict[str, Any] = None) -> PortfolioStrategyAgent:
    """Create Portfolio Strategy Agent with specified configuration"""
    return PortfolioStrategyAgent(config=config)

