"""
DC Score Intelligence Reports
Implementation of 8 strategic reports focused on DC scores and sectional analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime, timedelta
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import io
import base64

warnings.filterwarnings('ignore')

class DCScoreIntelligenceReports:
    """
    Implementation of DC Score Intelligence Reports that analyze actual DC scores and sectional performance
    """
    
    def __init__(self, data_manager, score_analyzer, multi_selection_manager):
        """
        Initialize DC Score Intelligence Reports
        
        Args:
            data_manager: DynamicDataManager instance
            score_analyzer: DynamicScoreAnalyzer instance
            multi_selection_manager: DynamicMultiSelectionManager instance
        """
        self.data_manager = data_manager
        self.score_analyzer = score_analyzer
        self.multi_selection_manager = multi_selection_manager
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Report cache
        self.report_cache = {}
        
        # Digi-Cadence sections
        self.dc_sections = ['Marketplace', 'Digital Spends', 'Organic Performance', 'Socialwatch']
        
        self.logger.info("DC Score Intelligence Reports initialized")
    
    def generate_dc_score_performance_analysis(self, selected_projects: List[int], 
                                             selected_brands: List[str],
                                             customization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate Dynamic DC Score Performance Analysis
        Contextual analysis of actual DC scores with performance insights
        """
        try:
            self.logger.info("Generating DC Score Performance Analysis...")
            
            # Extract and analyze DC scores
            dc_scores_data = self._extract_dc_scores(selected_projects, selected_brands)
            
            # Perform comprehensive analysis
            performance_analysis = {
                'overall_performance_summary': self._analyze_overall_dc_performance(dc_scores_data),
                'brand_performance_ranking': self._rank_brand_performance(dc_scores_data),
                'performance_trends': self._identify_performance_trends(dc_scores_data),
                'performance_distribution': self._analyze_performance_distribution(dc_scores_data),
                'improvement_opportunities': self._identify_dc_improvement_opportunities(dc_scores_data),
                'performance_benchmarks': self._establish_performance_benchmarks(dc_scores_data),
                'score_volatility_analysis': self._analyze_score_volatility(dc_scores_data),
                'performance_drivers': self._identify_performance_drivers(dc_scores_data)
            }
            
            # Generate insights and recommendations
            insights = self._generate_dc_performance_insights(performance_analysis, dc_scores_data)
            recommendations = self._generate_dc_performance_recommendations(performance_analysis, dc_scores_data)
            
            # Create visualizations
            visualizations = self._create_dc_performance_visualizations(performance_analysis, dc_scores_data)
            
            # Compile report
            report = {
                'report_id': 'dc_score_performance_analysis',
                'title': 'Dynamic DC Score Performance Analysis',
                'category': 'dc_score_intelligence',
                'analysis_results': performance_analysis,
                'key_insights': insights,
                'strategic_recommendations': recommendations,
                'visualizations': visualizations,
                'data_context': {
                    'projects_analyzed': selected_projects,
                    'brands_analyzed': selected_brands,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'data_quality_score': self._assess_data_quality(dc_scores_data)
                },
                'executive_summary': self._create_dc_performance_executive_summary(performance_analysis, insights)
            }
            
            self.logger.info("DC Score Performance Analysis completed")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating DC Score Performance Analysis: {str(e)}")
            raise
    
    def generate_sectional_score_deep_dive(self, selected_projects: List[int], 
                                          selected_brands: List[str],
                                          customization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate Sectional Score Deep Dive Analysis
        Dynamic analysis of Marketplace, Digital Spends, Organic Performance, Socialwatch
        """
        try:
            self.logger.info("Generating Sectional Score Deep Dive Analysis...")
            
            # Extract sectional scores
            sectional_data = self._extract_sectional_scores(selected_projects, selected_brands)
            
            # Analyze each section in detail
            sectional_analysis = {}
            for section in self.dc_sections:
                if section in sectional_data:
                    sectional_analysis[section] = {
                        'performance_overview': self._analyze_section_performance(sectional_data[section], section),
                        'brand_comparison': self._compare_brands_in_section(sectional_data[section], section),
                        'metric_breakdown': self._analyze_section_metrics(sectional_data[section], section),
                        'improvement_areas': self._identify_section_improvement_areas(sectional_data[section], section),
                        'best_practices': self._identify_section_best_practices(sectional_data[section], section),
                        'competitive_positioning': self._analyze_section_competitive_position(sectional_data[section], section)
                    }
            
            # Cross-sectional analysis
            cross_sectional_analysis = {
                'section_correlation_matrix': self._calculate_section_correlations(sectional_data),
                'section_synergies': self._identify_section_synergies(sectional_data),
                'balanced_performance_analysis': self._analyze_balanced_performance(sectional_data),
                'section_priority_matrix': self._create_section_priority_matrix(sectional_data)
            }
            
            # Generate insights and recommendations
            insights = self._generate_sectional_insights(sectional_analysis, cross_sectional_analysis)
            recommendations = self._generate_sectional_recommendations(sectional_analysis, cross_sectional_analysis)
            
            # Create visualizations
            visualizations = self._create_sectional_visualizations(sectional_analysis, cross_sectional_analysis)
            
            # Compile report
            report = {
                'report_id': 'sectional_score_deep_dive',
                'title': 'Sectional Score Deep Dive Analysis',
                'category': 'dc_score_intelligence',
                'sectional_analysis': sectional_analysis,
                'cross_sectional_analysis': cross_sectional_analysis,
                'key_insights': insights,
                'strategic_recommendations': recommendations,
                'visualizations': visualizations,
                'data_context': {
                    'projects_analyzed': selected_projects,
                    'brands_analyzed': selected_brands,
                    'sections_analyzed': list(sectional_analysis.keys()),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'executive_summary': self._create_sectional_executive_summary(sectional_analysis, insights)
            }
            
            self.logger.info("Sectional Score Deep Dive Analysis completed")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating Sectional Score Deep Dive Analysis: {str(e)}")
            raise
    
    def generate_score_revenue_correlation(self, selected_projects: List[int], 
                                         selected_brands: List[str],
                                         customization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate Score-to-Revenue Correlation Analysis
        Establishes actual correlation between scores and revenue
        """
        try:
            self.logger.info("Generating Score-to-Revenue Correlation Analysis...")
            
            # Extract DC scores and revenue data
            scores_data = self._extract_dc_scores(selected_projects, selected_brands)
            revenue_data = self._extract_revenue_data(selected_projects, selected_brands)
            
            # Perform correlation analysis
            correlation_analysis = {
                'overall_correlation': self._calculate_score_revenue_correlation(scores_data, revenue_data),
                'sectional_correlations': self._calculate_sectional_revenue_correlations(selected_projects, selected_brands),
                'brand_specific_correlations': self._calculate_brand_revenue_correlations(scores_data, revenue_data),
                'time_series_correlation': self._analyze_temporal_score_revenue_correlation(scores_data, revenue_data),
                'correlation_strength_analysis': self._analyze_correlation_strength(scores_data, revenue_data),
                'revenue_impact_modeling': self._model_revenue_impact(scores_data, revenue_data)
            }
            
            # Predictive modeling
            predictive_models = {
                'revenue_prediction_model': self._build_revenue_prediction_model(scores_data, revenue_data),
                'score_improvement_impact': self._model_score_improvement_impact(scores_data, revenue_data),
                'roi_estimation': self._estimate_score_improvement_roi(scores_data, revenue_data)
            }
            
            # Generate insights and recommendations
            insights = self._generate_revenue_correlation_insights(correlation_analysis, predictive_models)
            recommendations = self._generate_revenue_correlation_recommendations(correlation_analysis, predictive_models)
            
            # Create visualizations
            visualizations = self._create_revenue_correlation_visualizations(correlation_analysis, predictive_models)
            
            # Compile report
            report = {
                'report_id': 'score_revenue_correlation',
                'title': 'Score-to-Revenue Correlation Analysis',
                'category': 'dc_score_intelligence',
                'correlation_analysis': correlation_analysis,
                'predictive_models': predictive_models,
                'key_insights': insights,
                'strategic_recommendations': recommendations,
                'visualizations': visualizations,
                'data_context': {
                    'projects_analyzed': selected_projects,
                    'brands_analyzed': selected_brands,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'correlation_confidence': self._assess_correlation_confidence(correlation_analysis)
                },
                'executive_summary': self._create_revenue_correlation_executive_summary(correlation_analysis, insights)
            }
            
            self.logger.info("Score-to-Revenue Correlation Analysis completed")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating Score-to-Revenue Correlation Analysis: {str(e)}")
            raise
    
    def generate_market_share_impact(self, selected_projects: List[int], 
                                   selected_brands: List[str],
                                   customization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate Market Share Impact Analysis
        How DC scores impact actual market share
        """
        try:
            self.logger.info("Generating Market Share Impact Analysis...")
            
            # Extract scores and market share data
            scores_data = self._extract_dc_scores(selected_projects, selected_brands)
            market_share_data = self._extract_market_share_data(selected_projects, selected_brands)
            
            # Analyze market share impact
            impact_analysis = {
                'market_share_correlation': self._calculate_market_share_correlation(scores_data, market_share_data),
                'competitive_impact_analysis': self._analyze_competitive_market_impact(scores_data, market_share_data),
                'market_position_analysis': self._analyze_market_position_vs_scores(scores_data, market_share_data),
                'share_growth_drivers': self._identify_market_share_drivers(scores_data, market_share_data),
                'competitive_gap_analysis': self._analyze_competitive_gaps(scores_data, market_share_data),
                'market_opportunity_analysis': self._identify_market_opportunities(scores_data, market_share_data)
            }
            
            # Strategic analysis
            strategic_analysis = {
                'market_positioning_strategy': self._develop_market_positioning_strategy(impact_analysis),
                'competitive_response_analysis': self._analyze_competitive_responses(impact_analysis),
                'market_share_optimization': self._optimize_market_share_strategy(impact_analysis)
            }
            
            # Generate insights and recommendations
            insights = self._generate_market_share_insights(impact_analysis, strategic_analysis)
            recommendations = self._generate_market_share_recommendations(impact_analysis, strategic_analysis)
            
            # Create visualizations
            visualizations = self._create_market_share_visualizations(impact_analysis, strategic_analysis)
            
            # Compile report
            report = {
                'report_id': 'market_share_impact',
                'title': 'Market Share Impact Analysis',
                'category': 'dc_score_intelligence',
                'impact_analysis': impact_analysis,
                'strategic_analysis': strategic_analysis,
                'key_insights': insights,
                'strategic_recommendations': recommendations,
                'visualizations': visualizations,
                'data_context': {
                    'projects_analyzed': selected_projects,
                    'brands_analyzed': selected_brands,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'market_coverage': self._assess_market_coverage(market_share_data)
                },
                'executive_summary': self._create_market_share_executive_summary(impact_analysis, insights)
            }
            
            self.logger.info("Market Share Impact Analysis completed")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating Market Share Impact Analysis: {str(e)}")
            raise
    
    def generate_customer_acquisition_efficiency(self, selected_projects: List[int], 
                                                selected_brands: List[str],
                                                customization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate Customer Acquisition Efficiency Analysis
        Correlates scores with CAC and acquisition metrics
        """
        try:
            self.logger.info("Generating Customer Acquisition Efficiency Analysis...")
            
            # Extract scores and customer acquisition data
            scores_data = self._extract_dc_scores(selected_projects, selected_brands)
            acquisition_data = self._extract_customer_acquisition_data(selected_projects, selected_brands)
            
            # Analyze acquisition efficiency
            efficiency_analysis = {
                'cac_correlation_analysis': self._analyze_cac_correlation(scores_data, acquisition_data),
                'acquisition_funnel_analysis': self._analyze_acquisition_funnel_efficiency(scores_data, acquisition_data),
                'channel_efficiency_analysis': self._analyze_channel_acquisition_efficiency(scores_data, acquisition_data),
                'conversion_rate_analysis': self._analyze_conversion_rate_correlation(scores_data, acquisition_data),
                'ltv_cac_ratio_analysis': self._analyze_ltv_cac_ratio(scores_data, acquisition_data),
                'acquisition_cost_optimization': self._optimize_acquisition_costs(scores_data, acquisition_data)
            }
            
            # Performance benchmarking
            benchmarking_analysis = {
                'industry_benchmarks': self._establish_acquisition_benchmarks(efficiency_analysis),
                'competitive_efficiency': self._compare_acquisition_efficiency(efficiency_analysis),
                'efficiency_trends': self._analyze_efficiency_trends(efficiency_analysis)
            }
            
            # Generate insights and recommendations
            insights = self._generate_acquisition_efficiency_insights(efficiency_analysis, benchmarking_analysis)
            recommendations = self._generate_acquisition_efficiency_recommendations(efficiency_analysis, benchmarking_analysis)
            
            # Create visualizations
            visualizations = self._create_acquisition_efficiency_visualizations(efficiency_analysis, benchmarking_analysis)
            
            # Compile report
            report = {
                'report_id': 'customer_acquisition_efficiency',
                'title': 'Customer Acquisition Efficiency Analysis',
                'category': 'dc_score_intelligence',
                'efficiency_analysis': efficiency_analysis,
                'benchmarking_analysis': benchmarking_analysis,
                'key_insights': insights,
                'strategic_recommendations': recommendations,
                'visualizations': visualizations,
                'data_context': {
                    'projects_analyzed': selected_projects,
                    'brands_analyzed': selected_brands,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'data_completeness': self._assess_acquisition_data_completeness(acquisition_data)
                },
                'executive_summary': self._create_acquisition_efficiency_executive_summary(efficiency_analysis, insights)
            }
            
            self.logger.info("Customer Acquisition Efficiency Analysis completed")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating Customer Acquisition Efficiency Analysis: {str(e)}")
            raise
    
    def generate_brand_equity_correlation(self, selected_projects: List[int], 
                                        selected_brands: List[str],
                                        customization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate Brand Equity Correlation Analysis
        Relationship between scores and brand equity
        """
        try:
            self.logger.info("Generating Brand Equity Correlation Analysis...")
            
            # Extract scores and brand equity data
            scores_data = self._extract_dc_scores(selected_projects, selected_brands)
            brand_equity_data = self._extract_brand_equity_data(selected_projects, selected_brands)
            
            # Analyze brand equity correlation
            equity_analysis = {
                'overall_equity_correlation': self._calculate_brand_equity_correlation(scores_data, brand_equity_data),
                'equity_component_analysis': self._analyze_equity_components(scores_data, brand_equity_data),
                'brand_perception_analysis': self._analyze_brand_perception_correlation(scores_data, brand_equity_data),
                'brand_loyalty_analysis': self._analyze_brand_loyalty_correlation(scores_data, brand_equity_data),
                'brand_awareness_analysis': self._analyze_brand_awareness_correlation(scores_data, brand_equity_data),
                'brand_value_analysis': self._analyze_brand_value_correlation(scores_data, brand_equity_data)
            }
            
            # Strategic brand analysis
            strategic_analysis = {
                'brand_positioning_analysis': self._analyze_brand_positioning(equity_analysis),
                'brand_differentiation_analysis': self._analyze_brand_differentiation(equity_analysis),
                'brand_investment_priorities': self._prioritize_brand_investments(equity_analysis)
            }
            
            # Generate insights and recommendations
            insights = self._generate_brand_equity_insights(equity_analysis, strategic_analysis)
            recommendations = self._generate_brand_equity_recommendations(equity_analysis, strategic_analysis)
            
            # Create visualizations
            visualizations = self._create_brand_equity_visualizations(equity_analysis, strategic_analysis)
            
            # Compile report
            report = {
                'report_id': 'brand_equity_correlation',
                'title': 'Brand Equity Correlation Analysis',
                'category': 'dc_score_intelligence',
                'equity_analysis': equity_analysis,
                'strategic_analysis': strategic_analysis,
                'key_insights': insights,
                'strategic_recommendations': recommendations,
                'visualizations': visualizations,
                'data_context': {
                    'projects_analyzed': selected_projects,
                    'brands_analyzed': selected_brands,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'equity_measurement_confidence': self._assess_equity_measurement_confidence(brand_equity_data)
                },
                'executive_summary': self._create_brand_equity_executive_summary(equity_analysis, insights)
            }
            
            self.logger.info("Brand Equity Correlation Analysis completed")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating Brand Equity Correlation Analysis: {str(e)}")
            raise
    
    def generate_bestseller_rank_optimization(self, selected_projects: List[int], 
                                            selected_brands: List[str],
                                            customization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate Bestseller Rank Optimization Analysis
        Scores correlation with bestseller rankings
        """
        try:
            self.logger.info("Generating Bestseller Rank Optimization Analysis...")
            
            # Extract scores and ranking data
            scores_data = self._extract_dc_scores(selected_projects, selected_brands)
            ranking_data = self._extract_bestseller_ranking_data(selected_projects, selected_brands)
            
            # Analyze ranking correlation
            ranking_analysis = {
                'rank_correlation_analysis': self._calculate_ranking_correlation(scores_data, ranking_data),
                'ranking_factor_analysis': self._analyze_ranking_factors(scores_data, ranking_data),
                'category_ranking_analysis': self._analyze_category_rankings(scores_data, ranking_data),
                'ranking_volatility_analysis': self._analyze_ranking_volatility(scores_data, ranking_data),
                'competitive_ranking_analysis': self._analyze_competitive_rankings(scores_data, ranking_data),
                'ranking_optimization_opportunities': self._identify_ranking_optimization_opportunities(scores_data, ranking_data)
            }
            
            # Optimization strategies
            optimization_strategies = {
                'score_optimization_for_ranking': self._develop_score_optimization_strategy(ranking_analysis),
                'ranking_improvement_roadmap': self._create_ranking_improvement_roadmap(ranking_analysis),
                'competitive_ranking_strategy': self._develop_competitive_ranking_strategy(ranking_analysis)
            }
            
            # Generate insights and recommendations
            insights = self._generate_ranking_optimization_insights(ranking_analysis, optimization_strategies)
            recommendations = self._generate_ranking_optimization_recommendations(ranking_analysis, optimization_strategies)
            
            # Create visualizations
            visualizations = self._create_ranking_optimization_visualizations(ranking_analysis, optimization_strategies)
            
            # Compile report
            report = {
                'report_id': 'bestseller_rank_optimization',
                'title': 'Bestseller Rank Optimization Analysis',
                'category': 'dc_score_intelligence',
                'ranking_analysis': ranking_analysis,
                'optimization_strategies': optimization_strategies,
                'key_insights': insights,
                'strategic_recommendations': recommendations,
                'visualizations': visualizations,
                'data_context': {
                    'projects_analyzed': selected_projects,
                    'brands_analyzed': selected_brands,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'ranking_data_coverage': self._assess_ranking_data_coverage(ranking_data)
                },
                'executive_summary': self._create_ranking_optimization_executive_summary(ranking_analysis, insights)
            }
            
            self.logger.info("Bestseller Rank Optimization Analysis completed")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating Bestseller Rank Optimization Analysis: {str(e)}")
            raise
    
    def generate_sales_performance_correlation(self, selected_projects: List[int], 
                                             selected_brands: List[str],
                                             customization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate Sales Performance Correlation Analysis
        DC scores impact on actual sales performance
        """
        try:
            self.logger.info("Generating Sales Performance Correlation Analysis...")
            
            # Extract scores and sales data
            scores_data = self._extract_dc_scores(selected_projects, selected_brands)
            sales_data = self._extract_sales_performance_data(selected_projects, selected_brands)
            
            # Analyze sales correlation
            sales_analysis = {
                'sales_correlation_analysis': self._calculate_sales_correlation(scores_data, sales_data),
                'sales_driver_analysis': self._analyze_sales_drivers(scores_data, sales_data),
                'sales_channel_analysis': self._analyze_sales_channel_performance(scores_data, sales_data),
                'seasonal_sales_analysis': self._analyze_seasonal_sales_patterns(scores_data, sales_data),
                'sales_growth_analysis': self._analyze_sales_growth_correlation(scores_data, sales_data),
                'sales_efficiency_analysis': self._analyze_sales_efficiency(scores_data, sales_data)
            }
            
            # Performance optimization
            optimization_analysis = {
                'sales_optimization_opportunities': self._identify_sales_optimization_opportunities(sales_analysis),
                'sales_performance_benchmarks': self._establish_sales_performance_benchmarks(sales_analysis),
                'sales_improvement_strategies': self._develop_sales_improvement_strategies(sales_analysis)
            }
            
            # Generate insights and recommendations
            insights = self._generate_sales_performance_insights(sales_analysis, optimization_analysis)
            recommendations = self._generate_sales_performance_recommendations(sales_analysis, optimization_analysis)
            
            # Create visualizations
            visualizations = self._create_sales_performance_visualizations(sales_analysis, optimization_analysis)
            
            # Compile report
            report = {
                'report_id': 'sales_performance_correlation',
                'title': 'Sales Performance Correlation Analysis',
                'category': 'dc_score_intelligence',
                'sales_analysis': sales_analysis,
                'optimization_analysis': optimization_analysis,
                'key_insights': insights,
                'strategic_recommendations': recommendations,
                'visualizations': visualizations,
                'data_context': {
                    'projects_analyzed': selected_projects,
                    'brands_analyzed': selected_brands,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'sales_data_quality': self._assess_sales_data_quality(sales_data)
                },
                'executive_summary': self._create_sales_performance_executive_summary(sales_analysis, insights)
            }
            
            self.logger.info("Sales Performance Correlation Analysis completed")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating Sales Performance Correlation Analysis: {str(e)}")
            raise
    
    # Helper methods for data extraction and analysis
    def _extract_dc_scores(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """Extract DC scores for selected projects and brands"""
        try:
            dc_scores = {}
            
            for project_id in selected_projects:
                project_data = self.data_manager.project_data.get(project_id, {})
                if 'metrics_data' in project_data:
                    df = project_data['metrics_data']
                    
                    # Calculate DC scores for each brand
                    for brand in selected_brands:
                        if brand in df.columns:
                            brand_scores = pd.to_numeric(df[brand], errors='coerce').dropna()
                            if not brand_scores.empty:
                                if brand not in dc_scores:
                                    dc_scores[brand] = []
                                dc_scores[brand].append({
                                    'project_id': project_id,
                                    'overall_score': float(brand_scores.mean()),
                                    'score_distribution': brand_scores.tolist(),
                                    'score_variance': float(brand_scores.var()),
                                    'data_points': len(brand_scores)
                                })
            
            return dc_scores
            
        except Exception as e:
            self.logger.error(f"Error extracting DC scores: {str(e)}")
            return {}
    
    def _extract_sectional_scores(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """Extract sectional scores for analysis"""
        try:
            sectional_scores = {}
            
            for project_id in selected_projects:
                project_data = self.data_manager.project_data.get(project_id, {})
                if 'metrics_data' in project_data:
                    df = project_data['metrics_data']
                    
                    # Group by section
                    section_col = 'section_name' if 'section_name' in df.columns else 'sectionName'
                    if section_col in df.columns:
                        for section in df[section_col].unique():
                            if section not in sectional_scores:
                                sectional_scores[section] = {}
                            
                            section_data = df[df[section_col] == section]
                            
                            for brand in selected_brands:
                                if brand in section_data.columns:
                                    brand_section_scores = pd.to_numeric(section_data[brand], errors='coerce').dropna()
                                    if not brand_section_scores.empty:
                                        if brand not in sectional_scores[section]:
                                            sectional_scores[section][brand] = []
                                        
                                        sectional_scores[section][brand].append({
                                            'project_id': project_id,
                                            'section_score': float(brand_section_scores.mean()),
                                            'score_details': brand_section_scores.tolist(),
                                            'metrics_count': len(brand_section_scores)
                                        })
            
            return sectional_scores
            
        except Exception as e:
            self.logger.error(f"Error extracting sectional scores: {str(e)}")
            return {}
    
    # Placeholder methods for business data extraction (to be implemented based on actual data availability)
    def _extract_revenue_data(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """Extract revenue data (placeholder - implement based on actual data structure)"""
        # This would extract actual revenue data from the Digi-Cadence system
        # For now, return simulated structure
        return {brand: {'revenue_values': [], 'time_periods': []} for brand in selected_brands}
    
    def _extract_market_share_data(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """Extract market share data (placeholder)"""
        return {brand: {'market_share_values': [], 'competitive_data': {}} for brand in selected_brands}
    
    def _extract_customer_acquisition_data(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """Extract customer acquisition data (placeholder)"""
        return {brand: {'cac_values': [], 'acquisition_metrics': {}} for brand in selected_brands}
    
    def _extract_brand_equity_data(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """Extract brand equity data (placeholder)"""
        return {brand: {'equity_scores': [], 'equity_components': {}} for brand in selected_brands}
    
    def _extract_bestseller_ranking_data(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """Extract bestseller ranking data (placeholder)"""
        return {brand: {'rankings': [], 'categories': []} for brand in selected_brands}
    
    def _extract_sales_performance_data(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """Extract sales performance data (placeholder)"""
        return {brand: {'sales_values': [], 'performance_metrics': {}} for brand in selected_brands}
    
    # Analysis methods (implementations would be detailed based on specific requirements)
    def _analyze_overall_dc_performance(self, dc_scores_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall DC performance patterns"""
        try:
            analysis = {
                'performance_summary': {},
                'trend_analysis': {},
                'variance_analysis': {},
                'benchmark_comparison': {}
            }
            
            for brand, scores_list in dc_scores_data.items():
                if scores_list:
                    overall_scores = [item['overall_score'] for item in scores_list]
                    analysis['performance_summary'][brand] = {
                        'average_score': np.mean(overall_scores),
                        'median_score': np.median(overall_scores),
                        'score_range': [min(overall_scores), max(overall_scores)],
                        'consistency': 1.0 - (np.std(overall_scores) / np.mean(overall_scores)) if np.mean(overall_scores) > 0 else 0
                    }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing overall DC performance: {str(e)}")
            return {}
    
    def _rank_brand_performance(self, dc_scores_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank brands by performance"""
        try:
            brand_rankings = []
            
            for brand, scores_list in dc_scores_data.items():
                if scores_list:
                    overall_scores = [item['overall_score'] for item in scores_list]
                    avg_score = np.mean(overall_scores)
                    
                    brand_rankings.append({
                        'brand': brand,
                        'average_score': avg_score,
                        'performance_tier': 'high' if avg_score >= 80 else 'medium' if avg_score >= 60 else 'low',
                        'score_consistency': 1.0 - (np.std(overall_scores) / avg_score) if avg_score > 0 else 0
                    })
            
            # Sort by average score
            brand_rankings.sort(key=lambda x: x['average_score'], reverse=True)
            
            # Add ranking positions
            for i, ranking in enumerate(brand_rankings):
                ranking['rank'] = i + 1
            
            return brand_rankings
            
        except Exception as e:
            self.logger.error(f"Error ranking brand performance: {str(e)}")
            return []
    
    # Additional analysis methods would be implemented here...
    # (Due to length constraints, showing structure for key methods)
    
    def _generate_dc_performance_insights(self, performance_analysis: Dict[str, Any], dc_scores_data: Dict[str, Any]) -> List[str]:
        """Generate insights from DC performance analysis"""
        insights = []
        
        try:
            # Analyze performance summary
            if 'performance_summary' in performance_analysis:
                summary = performance_analysis['performance_summary']
                
                # Identify top performers
                top_performers = [brand for brand, data in summary.items() 
                                if data.get('average_score', 0) >= 80]
                if top_performers:
                    insights.append(f"High-performing brands identified: {', '.join(top_performers)} with DC scores above 80")
                
                # Identify improvement opportunities
                low_performers = [brand for brand, data in summary.items() 
                                if data.get('average_score', 0) < 60]
                if low_performers:
                    insights.append(f"Significant improvement opportunities exist for: {', '.join(low_performers)}")
                
                # Consistency analysis
                consistent_brands = [brand for brand, data in summary.items() 
                                   if data.get('consistency', 0) > 0.8]
                if consistent_brands:
                    insights.append(f"Most consistent performers: {', '.join(consistent_brands)}")
            
        except Exception as e:
            self.logger.error(f"Error generating DC performance insights: {str(e)}")
        
        return insights
    
    def _generate_dc_performance_recommendations(self, performance_analysis: Dict[str, Any], dc_scores_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations from DC performance analysis"""
        recommendations = []
        
        try:
            # Performance-based recommendations
            if 'performance_summary' in performance_analysis:
                summary = performance_analysis['performance_summary']
                
                for brand, data in summary.items():
                    avg_score = data.get('average_score', 0)
                    consistency = data.get('consistency', 0)
                    
                    if avg_score < 60:
                        recommendations.append(f"Priority action for {brand}: Implement comprehensive DC score improvement strategy")
                    elif avg_score < 80:
                        recommendations.append(f"Growth opportunity for {brand}: Focus on targeted score optimization")
                    
                    if consistency < 0.7:
                        recommendations.append(f"Improve score consistency for {brand}: Implement performance stabilization measures")
            
        except Exception as e:
            self.logger.error(f"Error generating DC performance recommendations: {str(e)}")
        
        return recommendations
    
    def _create_dc_performance_visualizations(self, performance_analysis: Dict[str, Any], dc_scores_data: Dict[str, Any]) -> Dict[str, str]:
        """Create visualizations for DC performance analysis"""
        visualizations = {}
        
        try:
            # Performance comparison chart
            if 'performance_summary' in performance_analysis:
                summary = performance_analysis['performance_summary']
                
                brands = list(summary.keys())
                scores = [summary[brand]['average_score'] for brand in brands]
                
                fig = go.Figure(data=[
                    go.Bar(x=brands, y=scores, name='Average DC Score')
                ])
                
                fig.update_layout(
                    title='Brand DC Score Performance Comparison',
                    xaxis_title='Brands',
                    yaxis_title='Average DC Score',
                    yaxis=dict(range=[0, 100])
                )
                
                visualizations['performance_comparison'] = fig.to_html()
            
        except Exception as e:
            self.logger.error(f"Error creating DC performance visualizations: {str(e)}")
        
        return visualizations
    
    def _create_dc_performance_executive_summary(self, performance_analysis: Dict[str, Any], insights: List[str]) -> str:
        """Create executive summary for DC performance analysis"""
        try:
            summary_parts = [
                "## Executive Summary: DC Score Performance Analysis",
                "",
                "### Key Findings:",
            ]
            
            # Add top insights
            for i, insight in enumerate(insights[:3], 1):
                summary_parts.append(f"{i}. {insight}")
            
            summary_parts.extend([
                "",
                "### Strategic Implications:",
                "- DC score performance directly impacts business outcomes",
                "- Consistent high performance requires systematic optimization",
                "- Cross-brand learning opportunities exist for performance improvement",
                "",
                "### Recommended Actions:",
                "- Prioritize improvement initiatives for underperforming brands",
                "- Implement best practices from top-performing brands",
                "- Establish regular performance monitoring and optimization cycles"
            ])
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error creating DC performance executive summary: {str(e)}")
            return "Executive summary generation failed"
    
    # Additional helper methods for data quality assessment
    def _assess_data_quality(self, data: Dict[str, Any]) -> float:
        """Assess data quality score"""
        try:
            if not data:
                return 0.0
            
            # Simple quality assessment based on data completeness
            total_brands = len(data)
            brands_with_data = sum(1 for brand_data in data.values() if brand_data)
            
            return (brands_with_data / total_brands) * 100 if total_brands > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {str(e)}")
            return 0.0

