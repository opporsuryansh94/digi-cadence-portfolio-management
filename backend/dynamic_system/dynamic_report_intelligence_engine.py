"""
Dynamic Report Intelligence Engine
Core engine for generating adaptive strategic reports based on actual Digi-Cadence data patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import warnings
from datetime import datetime, timedelta
import logging
import json
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

warnings.filterwarnings('ignore')

class DynamicReportIntelligenceEngine:
    """
    Core engine for generating dynamic strategic reports based on actual data patterns
    """
    
    def __init__(self, data_manager, score_analyzer, multi_selection_manager, hyperparameter_optimizer):
        """
        Initialize the report intelligence engine
        
        Args:
            data_manager: DynamicDataManager instance
            score_analyzer: DynamicScoreAnalyzer instance
            multi_selection_manager: DynamicMultiSelectionManager instance
            hyperparameter_optimizer: AdaptiveHyperparameterOptimizer instance
        """
        self.data_manager = data_manager
        self.score_analyzer = score_analyzer
        self.multi_selection_manager = multi_selection_manager
        self.hyperparameter_optimizer = hyperparameter_optimizer
        
        # Report registry
        self.available_reports = {}
        self.report_dependencies = {}
        self.report_cache = {}
        
        # Analysis context
        self.current_context = {}
        self.analysis_results = {}
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize report catalog
        self._initialize_report_catalog()
        
        self.logger.info("Dynamic Report Intelligence Engine initialized")
    
    def _initialize_report_catalog(self):
        """Initialize the catalog of available reports"""
        self.available_reports = {
            # Category A: DC Score Intelligence Reports (8 Reports)
            'dc_score_performance_analysis': {
                'title': 'Dynamic DC Score Performance Analysis',
                'category': 'dc_score_intelligence',
                'description': 'Contextual analysis of actual DC scores with performance insights',
                'data_requirements': ['dc_scores', 'sectional_scores'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'high'
            },
            'sectional_score_deep_dive': {
                'title': 'Sectional Score Deep Dive Analysis',
                'category': 'dc_score_intelligence',
                'description': 'Dynamic analysis of Marketplace, Digital Spends, Organic Performance, Socialwatch',
                'data_requirements': ['sectional_scores', 'platform_scores'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'high'
            },
            'score_revenue_correlation': {
                'title': 'Score-to-Revenue Correlation Analysis',
                'category': 'dc_score_intelligence',
                'description': 'Establishes actual correlation between scores and revenue',
                'data_requirements': ['dc_scores', 'business_outcomes'],
                'min_brands': 2,
                'min_projects': 1,
                'complexity': 'high',
                'business_impact': 'very_high'
            },
            'market_share_impact': {
                'title': 'Market Share Impact Analysis',
                'category': 'dc_score_intelligence',
                'description': 'How DC scores impact actual market share',
                'data_requirements': ['dc_scores', 'market_data'],
                'min_brands': 2,
                'min_projects': 1,
                'complexity': 'high',
                'business_impact': 'very_high'
            },
            'customer_acquisition_efficiency': {
                'title': 'Customer Acquisition Efficiency Analysis',
                'category': 'dc_score_intelligence',
                'description': 'Correlates scores with CAC and acquisition metrics',
                'data_requirements': ['dc_scores', 'customer_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'high'
            },
            'brand_equity_correlation': {
                'title': 'Brand Equity Correlation Analysis',
                'category': 'dc_score_intelligence',
                'description': 'Relationship between scores and brand equity',
                'data_requirements': ['dc_scores', 'brand_equity_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'high'
            },
            'bestseller_rank_optimization': {
                'title': 'Bestseller Rank Optimization Analysis',
                'category': 'dc_score_intelligence',
                'description': 'Scores correlation with bestseller rankings',
                'data_requirements': ['dc_scores', 'ranking_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'very_high'
            },
            'sales_performance_correlation': {
                'title': 'Sales Performance Correlation Analysis',
                'category': 'dc_score_intelligence',
                'description': 'DC scores impact on actual sales performance',
                'data_requirements': ['dc_scores', 'sales_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'very_high'
            },
            
            # Category B: Dynamic Business Outcome Reports (8 Reports)
            'revenue_impact_optimization': {
                'title': 'Revenue Impact Optimization',
                'category': 'business_outcome',
                'description': 'Dynamic revenue optimization based on score patterns',
                'data_requirements': ['dc_scores', 'revenue_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'high',
                'business_impact': 'very_high'
            },
            'market_position_enhancement': {
                'title': 'Market Position Enhancement Strategy',
                'category': 'business_outcome',
                'description': 'Positioning strategy based on competitive scores',
                'data_requirements': ['dc_scores', 'competitive_data'],
                'min_brands': 2,
                'min_projects': 1,
                'complexity': 'high',
                'business_impact': 'very_high'
            },
            'customer_lifetime_value_enhancement': {
                'title': 'Customer Lifetime Value Enhancement',
                'category': 'business_outcome',
                'description': 'Correlates scores with CLV improvements',
                'data_requirements': ['dc_scores', 'customer_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'high',
                'business_impact': 'very_high'
            },
            'conversion_rate_optimization': {
                'title': 'Conversion Rate Optimization Analysis',
                'category': 'business_outcome',
                'description': 'Score-driven conversion optimization strategies',
                'data_requirements': ['dc_scores', 'conversion_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'high'
            },
            'roi_maximization_strategy': {
                'title': 'ROI Maximization Strategy',
                'category': 'business_outcome',
                'description': 'Optimize ROI based on score performance patterns',
                'data_requirements': ['dc_scores', 'investment_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'high',
                'business_impact': 'very_high'
            },
            'competitive_advantage_analysis': {
                'title': 'Competitive Advantage Analysis',
                'category': 'business_outcome',
                'description': 'Identify advantages based on score differentials',
                'data_requirements': ['dc_scores', 'competitive_data'],
                'min_brands': 2,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'high'
            },
            'market_penetration_strategy': {
                'title': 'Market Penetration Strategy',
                'category': 'business_outcome',
                'description': 'Score-based market penetration opportunities',
                'data_requirements': ['dc_scores', 'market_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'high'
            },
            'brand_portfolio_optimization': {
                'title': 'Brand Portfolio Optimization',
                'category': 'business_outcome',
                'description': 'Optimize brand portfolio based on score synergies',
                'data_requirements': ['dc_scores', 'portfolio_data'],
                'min_brands': 3,
                'min_projects': 1,
                'complexity': 'high',
                'business_impact': 'very_high'
            },
            
            # Category C: Predictive Intelligence Reports (6 Reports)
            'performance_forecasting': {
                'title': 'Performance Forecasting Analysis',
                'category': 'predictive_intelligence',
                'description': 'Predict future performance based on score trends',
                'data_requirements': ['dc_scores', 'historical_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'high',
                'business_impact': 'very_high'
            },
            'trend_prediction_analysis': {
                'title': 'Trend Prediction Analysis',
                'category': 'predictive_intelligence',
                'description': 'Predict market and performance trends',
                'data_requirements': ['dc_scores', 'trend_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'high',
                'business_impact': 'high'
            },
            'risk_assessment_analysis': {
                'title': 'Risk Assessment Analysis',
                'category': 'predictive_intelligence',
                'description': 'Assess risks based on score volatility patterns',
                'data_requirements': ['dc_scores', 'risk_indicators'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'high'
            },
            'opportunity_identification': {
                'title': 'Opportunity Identification Analysis',
                'category': 'predictive_intelligence',
                'description': 'Identify future opportunities from score patterns',
                'data_requirements': ['dc_scores', 'market_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'high'
            },
            'scenario_planning_analysis': {
                'title': 'Scenario Planning Analysis',
                'category': 'predictive_intelligence',
                'description': 'Model different scenarios based on score changes',
                'data_requirements': ['dc_scores', 'scenario_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'high',
                'business_impact': 'very_high'
            },
            'growth_trajectory_modeling': {
                'title': 'Growth Trajectory Modeling',
                'category': 'predictive_intelligence',
                'description': 'Model growth trajectories based on score improvements',
                'data_requirements': ['dc_scores', 'growth_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'high',
                'business_impact': 'very_high'
            },
            
            # Category D: Executive Intelligence Reports (3 Reports)
            'executive_performance_dashboard': {
                'title': 'Executive Performance Dashboard',
                'category': 'executive_intelligence',
                'description': 'High-level performance overview for executives',
                'data_requirements': ['dc_scores', 'kpi_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'low',
                'business_impact': 'very_high'
            },
            'strategic_planning_insights': {
                'title': 'Strategic Planning Insights',
                'category': 'executive_intelligence',
                'description': 'Strategic insights for long-term planning',
                'data_requirements': ['dc_scores', 'strategic_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'very_high'
            },
            'investment_priority_analysis': {
                'title': 'Investment Priority Analysis',
                'category': 'executive_intelligence',
                'description': 'Prioritize investments based on score impact potential',
                'data_requirements': ['dc_scores', 'investment_data'],
                'min_brands': 1,
                'min_projects': 1,
                'complexity': 'medium',
                'business_impact': 'very_high'
            }
        }
    
    def analyze_data_for_report_selection(self, selected_projects: List[int], 
                                        selected_brands: List[str]) -> Dict[str, Any]:
        """
        Analyze available data to determine which reports can be generated
        
        Args:
            selected_projects: List of selected project IDs
            selected_brands: List of selected brand IDs
            
        Returns:
            Dict with analysis results and available reports
        """
        try:
            self.logger.info(f"Analyzing data for report selection: {len(selected_projects)} projects, {len(selected_brands)} brands")
            
            # Get data characteristics
            data_analysis = {
                'data_availability': self._assess_data_availability(selected_projects, selected_brands),
                'data_quality': self._assess_data_quality(selected_projects, selected_brands),
                'correlation_potential': self._assess_correlation_potential(selected_projects, selected_brands),
                'business_outcome_availability': self._assess_business_outcome_availability(selected_projects, selected_brands),
                'temporal_coverage': self._assess_temporal_coverage(selected_projects, selected_brands)
            }
            
            # Determine available reports
            available_reports = self._determine_available_reports(data_analysis, selected_projects, selected_brands)
            
            # Prioritize reports based on data quality and business impact
            prioritized_reports = self._prioritize_reports(available_reports, data_analysis)
            
            # Generate recommendations
            recommendations = self._generate_report_recommendations(prioritized_reports, data_analysis)
            
            result = {
                'data_analysis': data_analysis,
                'available_reports': available_reports,
                'prioritized_reports': prioritized_reports,
                'recommendations': recommendations,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Store context for report generation
            self.current_context = {
                'selected_projects': selected_projects,
                'selected_brands': selected_brands,
                'data_analysis': data_analysis,
                'available_reports': available_reports
            }
            
            self.logger.info(f"Data analysis completed: {len(available_reports)} reports available")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in data analysis for report selection: {str(e)}")
            raise
    
    def generate_dynamic_report(self, report_id: str, customization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a specific dynamic report based on actual data
        
        Args:
            report_id: ID of the report to generate
            customization_params: Optional customization parameters
            
        Returns:
            Dict with generated report content
        """
        try:
            self.logger.info(f"Generating dynamic report: {report_id}")
            
            # Check if report is available
            if report_id not in self.current_context.get('available_reports', {}):
                raise ValueError(f"Report {report_id} is not available for current data selection")
            
            # Get report configuration
            report_config = self.available_reports.get(report_id, {})
            
            # Apply customization
            if customization_params:
                report_config = self._apply_customization(report_config, customization_params)
            
            # Generate report based on category
            category = report_config.get('category', '')
            
            if category == 'dc_score_intelligence':
                report_content = self._generate_dc_score_intelligence_report(report_id, report_config)
            elif category == 'business_outcome':
                report_content = self._generate_business_outcome_report(report_id, report_config)
            elif category == 'predictive_intelligence':
                report_content = self._generate_predictive_intelligence_report(report_id, report_config)
            elif category == 'executive_intelligence':
                report_content = self._generate_executive_intelligence_report(report_id, report_config)
            else:
                raise ValueError(f"Unknown report category: {category}")
            
            # Add metadata
            report_content['metadata'] = {
                'report_id': report_id,
                'report_title': report_config.get('title', ''),
                'category': category,
                'generation_timestamp': datetime.now().isoformat(),
                'data_context': {
                    'projects': self.current_context.get('selected_projects', []),
                    'brands': self.current_context.get('selected_brands', [])
                },
                'customization_applied': customization_params is not None
            }
            
            # Cache report
            self.report_cache[report_id] = report_content
            
            self.logger.info(f"Dynamic report generated successfully: {report_id}")
            return report_content
            
        except Exception as e:
            self.logger.error(f"Error generating dynamic report {report_id}: {str(e)}")
            raise
    
    def _assess_data_availability(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """Assess data availability for selected projects and brands"""
        availability = {
            'dc_scores_available': False,
            'sectional_scores_available': False,
            'platform_scores_available': False,
            'business_outcomes_available': False,
            'historical_data_available': False,
            'competitive_data_available': False,
            'overall_completeness': 0.0,
            'missing_data_areas': []
        }
        
        try:
            # Check DC scores availability
            dc_scores_count = 0
            sectional_scores_count = 0
            total_combinations = len(selected_projects) * len(selected_brands)
            
            for project_id in selected_projects:
                project_data = self.data_manager.project_data.get(project_id, {})
                
                for brand in selected_brands:
                    if brand in project_data.get('brands_available', []):
                        dc_scores_count += 1
                        
                        # Check for sectional data
                        if 'sectional_scores' in project_data:
                            sectional_scores_count += 1
            
            # Calculate availability percentages
            if total_combinations > 0:
                dc_availability = dc_scores_count / total_combinations
                sectional_availability = sectional_scores_count / total_combinations
                
                availability['dc_scores_available'] = dc_availability > 0.5
                availability['sectional_scores_available'] = sectional_availability > 0.5
                availability['overall_completeness'] = (dc_availability + sectional_availability) / 2
            
            # Check for business outcome data
            availability['business_outcomes_available'] = self._check_business_outcome_data(selected_projects, selected_brands)
            
            # Check for historical data
            availability['historical_data_available'] = self._check_historical_data(selected_projects, selected_brands)
            
            # Check for competitive data
            availability['competitive_data_available'] = len(selected_brands) > 1
            
            # Identify missing data areas
            if not availability['dc_scores_available']:
                availability['missing_data_areas'].append('DC Scores')
            if not availability['sectional_scores_available']:
                availability['missing_data_areas'].append('Sectional Scores')
            if not availability['business_outcomes_available']:
                availability['missing_data_areas'].append('Business Outcomes')
            
        except Exception as e:
            self.logger.error(f"Error assessing data availability: {str(e)}")
        
        return availability
    
    def _assess_data_quality(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """Assess data quality for selected projects and brands"""
        quality = {
            'overall_quality_score': 0.0,
            'completeness_score': 0.0,
            'consistency_score': 0.0,
            'accuracy_score': 0.0,
            'timeliness_score': 0.0,
            'quality_issues': []
        }
        
        try:
            quality_scores = []
            
            for project_id in selected_projects:
                project_data = self.data_manager.project_data.get(project_id, {})
                project_quality = project_data.get('data_quality', {})
                
                if project_quality:
                    completeness = project_quality.get('overall_completeness', 0) / 100.0
                    consistency = project_quality.get('consistency_score', 80) / 100.0  # Default assumption
                    accuracy = project_quality.get('accuracy_score', 85) / 100.0      # Default assumption
                    timeliness = project_quality.get('timeliness_score', 90) / 100.0   # Default assumption
                    
                    project_score = (completeness + consistency + accuracy + timeliness) / 4
                    quality_scores.append(project_score)
            
            if quality_scores:
                quality['overall_quality_score'] = np.mean(quality_scores)
                quality['completeness_score'] = np.mean([s * 4 for s in quality_scores])  # Approximate
                quality['consistency_score'] = 0.8  # Default
                quality['accuracy_score'] = 0.85   # Default
                quality['timeliness_score'] = 0.9   # Default
            
            # Identify quality issues
            if quality['overall_quality_score'] < 0.7:
                quality['quality_issues'].append('Overall data quality below recommended threshold')
            if quality['completeness_score'] < 0.6:
                quality['quality_issues'].append('Data completeness issues detected')
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {str(e)}")
        
        return quality
    
    def _assess_correlation_potential(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, float]:
        """Assess potential for meaningful correlations"""
        potential = {
            'score_business_correlation': 0.0,
            'cross_brand_correlation': 0.0,
            'temporal_correlation': 0.0,
            'sectional_correlation': 0.0
        }
        
        try:
            # Base potential on data characteristics
            brand_count = len(selected_brands)
            project_count = len(selected_projects)
            
            # More brands and projects = higher correlation potential
            potential['score_business_correlation'] = min(1.0, (brand_count * project_count) / 10.0)
            potential['cross_brand_correlation'] = min(1.0, brand_count / 5.0)
            potential['temporal_correlation'] = min(1.0, project_count / 3.0)
            potential['sectional_correlation'] = 0.8  # Generally good potential
            
        except Exception as e:
            self.logger.error(f"Error assessing correlation potential: {str(e)}")
        
        return potential
    
    def _assess_business_outcome_availability(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, bool]:
        """Assess availability of business outcome data"""
        outcomes = {
            'revenue_data': False,
            'market_share_data': False,
            'customer_data': False,
            'sales_data': False,
            'roi_data': False,
            'ranking_data': False
        }
        
        try:
            # Check for business outcome indicators in project data
            for project_id in selected_projects:
                project_data = self.data_manager.project_data.get(project_id, {})
                
                # Look for business outcome indicators
                if 'business_outcomes' in project_data:
                    outcomes['revenue_data'] = True
                    outcomes['sales_data'] = True
                
                # Check for market-related metrics
                metrics = project_data.get('metrics_available', [])
                for metric in metrics:
                    metric_lower = metric.lower()
                    if 'revenue' in metric_lower or 'sales' in metric_lower:
                        outcomes['revenue_data'] = True
                        outcomes['sales_data'] = True
                    elif 'market' in metric_lower or 'share' in metric_lower:
                        outcomes['market_share_data'] = True
                    elif 'customer' in metric_lower or 'acquisition' in metric_lower:
                        outcomes['customer_data'] = True
                    elif 'roi' in metric_lower or 'return' in metric_lower:
                        outcomes['roi_data'] = True
                    elif 'rank' in metric_lower or 'position' in metric_lower:
                        outcomes['ranking_data'] = True
        
        except Exception as e:
            self.logger.error(f"Error assessing business outcome availability: {str(e)}")
        
        return outcomes
    
    def _assess_temporal_coverage(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """Assess temporal coverage of the data"""
        coverage = {
            'time_span_months': 0,
            'data_frequency': 'unknown',
            'historical_depth': 'shallow',
            'trend_analysis_feasible': False,
            'forecasting_feasible': False
        }
        
        try:
            # Estimate temporal coverage based on project count and data patterns
            project_count = len(selected_projects)
            
            if project_count >= 12:
                coverage['time_span_months'] = 12
                coverage['historical_depth'] = 'deep'
                coverage['trend_analysis_feasible'] = True
                coverage['forecasting_feasible'] = True
            elif project_count >= 6:
                coverage['time_span_months'] = 6
                coverage['historical_depth'] = 'medium'
                coverage['trend_analysis_feasible'] = True
                coverage['forecasting_feasible'] = False
            else:
                coverage['time_span_months'] = project_count
                coverage['historical_depth'] = 'shallow'
                coverage['trend_analysis_feasible'] = False
                coverage['forecasting_feasible'] = False
            
            coverage['data_frequency'] = 'monthly'  # Assumption based on typical business reporting
            
        except Exception as e:
            self.logger.error(f"Error assessing temporal coverage: {str(e)}")
        
        return coverage
    
    def _determine_available_reports(self, data_analysis: Dict[str, Any], 
                                   selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Dict[str, Any]]:
        """Determine which reports can be generated based on data analysis"""
        available_reports = {}
        
        try:
            data_availability = data_analysis.get('data_availability', {})
            data_quality = data_analysis.get('data_quality', {})
            business_outcomes = data_analysis.get('business_outcome_availability', {})
            temporal_coverage = data_analysis.get('temporal_coverage', {})
            
            for report_id, report_config in self.available_reports.items():
                # Check minimum requirements
                min_brands = report_config.get('min_brands', 1)
                min_projects = report_config.get('min_projects', 1)
                
                if len(selected_brands) < min_brands or len(selected_projects) < min_projects:
                    continue
                
                # Check data requirements
                data_requirements = report_config.get('data_requirements', [])
                requirements_met = True
                missing_requirements = []
                
                for requirement in data_requirements:
                    if requirement == 'dc_scores' and not data_availability.get('dc_scores_available', False):
                        requirements_met = False
                        missing_requirements.append('DC Scores')
                    elif requirement == 'sectional_scores' and not data_availability.get('sectional_scores_available', False):
                        requirements_met = False
                        missing_requirements.append('Sectional Scores')
                    elif requirement == 'business_outcomes' and not any(business_outcomes.values()):
                        requirements_met = False
                        missing_requirements.append('Business Outcomes')
                    elif requirement == 'historical_data' and not temporal_coverage.get('trend_analysis_feasible', False):
                        requirements_met = False
                        missing_requirements.append('Historical Data')
                
                # Assess report feasibility
                feasibility_score = self._calculate_report_feasibility(report_config, data_analysis)
                
                if requirements_met and feasibility_score > 0.5:
                    available_reports[report_id] = {
                        'config': report_config,
                        'feasibility_score': feasibility_score,
                        'data_quality_score': data_quality.get('overall_quality_score', 0.0),
                        'missing_requirements': missing_requirements,
                        'estimated_insights_quality': self._estimate_insights_quality(report_config, data_analysis)
                    }
        
        except Exception as e:
            self.logger.error(f"Error determining available reports: {str(e)}")
        
        return available_reports
    
    def _calculate_report_feasibility(self, report_config: Dict[str, Any], data_analysis: Dict[str, Any]) -> float:
        """Calculate feasibility score for a report"""
        try:
            feasibility_factors = []
            
            # Data availability factor
            data_availability = data_analysis.get('data_availability', {})
            availability_score = data_availability.get('overall_completeness', 0.0)
            feasibility_factors.append(availability_score * 0.4)
            
            # Data quality factor
            data_quality = data_analysis.get('data_quality', {})
            quality_score = data_quality.get('overall_quality_score', 0.0)
            feasibility_factors.append(quality_score * 0.3)
            
            # Correlation potential factor
            correlation_potential = data_analysis.get('correlation_potential', {})
            avg_correlation = np.mean(list(correlation_potential.values()))
            feasibility_factors.append(avg_correlation * 0.2)
            
            # Complexity factor (inverse - simpler reports are more feasible)
            complexity = report_config.get('complexity', 'medium')
            complexity_scores = {'low': 1.0, 'medium': 0.8, 'high': 0.6}
            feasibility_factors.append(complexity_scores.get(complexity, 0.8) * 0.1)
            
            return sum(feasibility_factors)
            
        except Exception as e:
            self.logger.error(f"Error calculating report feasibility: {str(e)}")
            return 0.0
    
    def _estimate_insights_quality(self, report_config: Dict[str, Any], data_analysis: Dict[str, Any]) -> str:
        """Estimate the quality of insights that can be generated"""
        try:
            data_quality = data_analysis.get('data_quality', {}).get('overall_quality_score', 0.0)
            correlation_potential = np.mean(list(data_analysis.get('correlation_potential', {}).values()))
            
            combined_score = (data_quality + correlation_potential) / 2
            
            if combined_score > 0.8:
                return 'high'
            elif combined_score > 0.6:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            self.logger.error(f"Error estimating insights quality: {str(e)}")
            return 'medium'
    
    def _prioritize_reports(self, available_reports: Dict[str, Dict[str, Any]], 
                          data_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize reports based on feasibility and business impact"""
        prioritized = []
        
        try:
            for report_id, report_info in available_reports.items():
                config = report_info['config']
                
                # Calculate priority score
                feasibility = report_info['feasibility_score']
                business_impact_scores = {'low': 0.3, 'medium': 0.6, 'high': 0.8, 'very_high': 1.0}
                business_impact = business_impact_scores.get(config.get('business_impact', 'medium'), 0.6)
                data_quality = report_info['data_quality_score']
                
                priority_score = (feasibility * 0.4 + business_impact * 0.4 + data_quality * 0.2)
                
                prioritized.append({
                    'report_id': report_id,
                    'title': config.get('title', ''),
                    'category': config.get('category', ''),
                    'priority_score': priority_score,
                    'feasibility_score': feasibility,
                    'business_impact': config.get('business_impact', 'medium'),
                    'estimated_insights_quality': report_info['estimated_insights_quality'],
                    'complexity': config.get('complexity', 'medium')
                })
            
            # Sort by priority score
            prioritized.sort(key=lambda x: x['priority_score'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error prioritizing reports: {str(e)}")
        
        return prioritized
    
    def _generate_report_recommendations(self, prioritized_reports: List[Dict[str, Any]], 
                                       data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for report selection"""
        recommendations = {
            'recommended_reports': [],
            'quick_wins': [],
            'high_impact_reports': [],
            'data_improvement_suggestions': [],
            'analysis_strategy': ''
        }
        
        try:
            # Top 5 recommended reports
            recommendations['recommended_reports'] = prioritized_reports[:5]
            
            # Quick wins (high feasibility, medium+ impact)
            for report in prioritized_reports:
                if (report['feasibility_score'] > 0.8 and 
                    report['business_impact'] in ['medium', 'high', 'very_high'] and
                    report['complexity'] in ['low', 'medium']):
                    recommendations['quick_wins'].append(report)
            
            # High impact reports
            for report in prioritized_reports:
                if report['business_impact'] in ['high', 'very_high']:
                    recommendations['high_impact_reports'].append(report)
            
            # Data improvement suggestions
            data_availability = data_analysis.get('data_availability', {})
            missing_areas = data_availability.get('missing_data_areas', [])
            
            for area in missing_areas:
                recommendations['data_improvement_suggestions'].append(
                    f"Improve {area} data collection to enable additional reports"
                )
            
            # Analysis strategy
            data_quality = data_analysis.get('data_quality', {}).get('overall_quality_score', 0.0)
            
            if data_quality > 0.8:
                recommendations['analysis_strategy'] = 'comprehensive_analysis'
            elif data_quality > 0.6:
                recommendations['analysis_strategy'] = 'selective_analysis'
            else:
                recommendations['analysis_strategy'] = 'data_improvement_first'
        
        except Exception as e:
            self.logger.error(f"Error generating report recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_dc_score_intelligence_report(self, report_id: str, report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate DC Score Intelligence category reports"""
        # This will be implemented in the next phase with specific report generators
        return {
            'report_type': 'dc_score_intelligence',
            'content': f"Dynamic {report_config.get('title', '')} report content will be generated here",
            'insights': [],
            'recommendations': [],
            'visualizations': [],
            'data_sources': self.current_context.get('selected_projects', []),
            'brands_analyzed': self.current_context.get('selected_brands', [])
        }
    
    def _generate_business_outcome_report(self, report_id: str, report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Business Outcome category reports"""
        # This will be implemented in the next phase with specific report generators
        return {
            'report_type': 'business_outcome',
            'content': f"Dynamic {report_config.get('title', '')} report content will be generated here",
            'insights': [],
            'recommendations': [],
            'visualizations': [],
            'data_sources': self.current_context.get('selected_projects', []),
            'brands_analyzed': self.current_context.get('selected_brands', [])
        }
    
    def _generate_predictive_intelligence_report(self, report_id: str, report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Predictive Intelligence category reports"""
        # This will be implemented in the next phase with specific report generators
        return {
            'report_type': 'predictive_intelligence',
            'content': f"Dynamic {report_config.get('title', '')} report content will be generated here",
            'insights': [],
            'recommendations': [],
            'visualizations': [],
            'data_sources': self.current_context.get('selected_projects', []),
            'brands_analyzed': self.current_context.get('selected_brands', [])
        }
    
    def _generate_executive_intelligence_report(self, report_id: str, report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Executive Intelligence category reports"""
        # This will be implemented in the next phase with specific report generators
        return {
            'report_type': 'executive_intelligence',
            'content': f"Dynamic {report_config.get('title', '')} report content will be generated here",
            'insights': [],
            'recommendations': [],
            'visualizations': [],
            'data_sources': self.current_context.get('selected_projects', []),
            'brands_analyzed': self.current_context.get('selected_brands', [])
        }
    
    def get_report_catalog(self) -> Dict[str, Any]:
        """Get the complete catalog of available reports"""
        catalog = {
            'total_reports': len(self.available_reports),
            'categories': {},
            'reports_by_category': {},
            'complexity_distribution': {},
            'business_impact_distribution': {}
        }
        
        try:
            # Group by category
            for report_id, config in self.available_reports.items():
                category = config.get('category', 'unknown')
                
                if category not in catalog['categories']:
                    catalog['categories'][category] = 0
                    catalog['reports_by_category'][category] = []
                
                catalog['categories'][category] += 1
                catalog['reports_by_category'][category].append({
                    'id': report_id,
                    'title': config.get('title', ''),
                    'description': config.get('description', ''),
                    'complexity': config.get('complexity', 'medium'),
                    'business_impact': config.get('business_impact', 'medium')
                })
            
            # Complexity distribution
            complexity_counts = {}
            impact_counts = {}
            
            for config in self.available_reports.values():
                complexity = config.get('complexity', 'medium')
                impact = config.get('business_impact', 'medium')
                
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
                impact_counts[impact] = impact_counts.get(impact, 0) + 1
            
            catalog['complexity_distribution'] = complexity_counts
            catalog['business_impact_distribution'] = impact_counts
        
        except Exception as e:
            self.logger.error(f"Error generating report catalog: {str(e)}")
        
        return catalog
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current status of the report intelligence engine"""
        status = {
            'engine_initialized': True,
            'reports_available': len(self.available_reports),
            'current_context_set': bool(self.current_context),
            'cached_reports': len(self.report_cache),
            'last_analysis_timestamp': self.current_context.get('analysis_timestamp'),
            'selected_projects': self.current_context.get('selected_projects', []),
            'selected_brands': self.current_context.get('selected_brands', [])
        }
        
        return status
    
    # Helper methods
    def _check_business_outcome_data(self, selected_projects: List[int], selected_brands: List[str]) -> bool:
        """Check if business outcome data is available"""
        try:
            for project_id in selected_projects:
                project_data = self.data_manager.project_data.get(project_id, {})
                if 'business_outcomes' in project_data:
                    return True
                
                # Check metrics for business-related indicators
                metrics = project_data.get('metrics_available', [])
                business_indicators = ['revenue', 'sales', 'roi', 'conversion', 'market', 'customer']
                
                for metric in metrics:
                    if any(indicator in metric.lower() for indicator in business_indicators):
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking business outcome data: {str(e)}")
            return False
    
    def _check_historical_data(self, selected_projects: List[int], selected_brands: List[str]) -> bool:
        """Check if sufficient historical data is available"""
        try:
            return len(selected_projects) >= 3  # Minimum for trend analysis
        except Exception as e:
            self.logger.error(f"Error checking historical data: {str(e)}")
            return False
    
    def _apply_customization(self, report_config: Dict[str, Any], customization_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply customization parameters to report configuration"""
        customized_config = report_config.copy()
        
        try:
            # Apply customizations
            if 'focus_areas' in customization_params:
                customized_config['focus_areas'] = customization_params['focus_areas']
            
            if 'analysis_depth' in customization_params:
                customized_config['analysis_depth'] = customization_params['analysis_depth']
            
            if 'visualization_preferences' in customization_params:
                customized_config['visualization_preferences'] = customization_params['visualization_preferences']
            
            if 'business_context' in customization_params:
                customized_config['business_context'] = customization_params['business_context']
        
        except Exception as e:
            self.logger.error(f"Error applying customization: {str(e)}")
        
        return customized_config

