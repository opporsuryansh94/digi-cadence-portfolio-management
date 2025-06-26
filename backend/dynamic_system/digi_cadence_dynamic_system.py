"""
Digi-Cadence Dynamic Enhancement System
Main integration system that brings together all dynamic components
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import pandas as pd
import numpy as np

# Import all dynamic components
try:
    from dynamic_data_manager import DynamicDataManager
    from adaptive_api_client import AdaptiveAPIClient
    from dynamic_score_analyzer import DynamicScoreAnalyzer
    from adaptive_hyperparameter_optimizer import AdaptiveHyperparameterOptimizer
    from dynamic_multi_selection_manager import DynamicMultiSelectionManager
    from dynamic_report_intelligence_engine import DynamicReportIntelligenceEngine
    from dc_score_intelligence_reports import DCScoreIntelligenceReports
    from business_outcome_reports import BusinessOutcomeReports
    from predictive_intelligence_reports import PredictiveIntelligenceReports
    from executive_intelligence_reports import ExecutiveIntelligenceReports
except ImportError as e:
    print(f"Warning: Some components not available: {e}")

class DigiCadenceDynamicSystem:
    """
    Main integration system for Digi-Cadence Dynamic Enhancement
    Coordinates all dynamic components and provides unified interface
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Digi-Cadence Dynamic System
        
        Args:
            config: Configuration dictionary for system initialization
        """
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # System configuration
        self.config = config or self._load_default_config()
        
        # Initialize core components
        self.data_manager = None
        self.api_client = None
        self.score_analyzer = None
        self.hyperparameter_optimizer = None
        self.multi_selection_manager = None
        self.report_engine = None
        
        # Initialize report generators
        self.dc_score_reports = None
        self.business_outcome_reports = None
        self.predictive_reports = None
        self.executive_reports = None
        
        # System state
        self.is_initialized = False
        self.available_reports = []
        self.system_metrics = {}
        
        self.logger.info("Digi-Cadence Dynamic System initialized")
    
    def initialize_system(self) -> bool:
        """
        Initialize all system components
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Digi-Cadence Dynamic System components...")
            
            # Initialize core components
            self._initialize_core_components()
            
            # Initialize report generators
            self._initialize_report_generators()
            
            # Validate system integration
            self._validate_system_integration()
            
            # Load available reports
            self._load_available_reports()
            
            self.is_initialized = True
            self.logger.info("Digi-Cadence Dynamic System initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}")
            return False
    
    def _initialize_core_components(self):
        """Initialize core system components"""
        try:
            # Initialize Data Manager
            self.data_manager = DynamicDataManager(
                api_config=self.config.get('api_config', {}),
                database_config=self.config.get('database_config', {})
            )
            
            # Initialize API Client
            self.api_client = AdaptiveAPIClient(
                base_urls=self.config.get('api_base_urls', []),
                authentication_config=self.config.get('auth_config', {})
            )
            
            # Initialize Score Analyzer
            self.score_analyzer = DynamicScoreAnalyzer(
                data_manager=self.data_manager,
                analysis_config=self.config.get('analysis_config', {})
            )
            
            # Initialize Hyperparameter Optimizer
            self.hyperparameter_optimizer = AdaptiveHyperparameterOptimizer(
                optimization_config=self.config.get('optimization_config', {})
            )
            
            # Initialize Multi-Selection Manager
            self.multi_selection_manager = DynamicMultiSelectionManager(
                data_manager=self.data_manager,
                score_analyzer=self.score_analyzer
            )
            
            # Initialize Report Engine
            self.report_engine = DynamicReportIntelligenceEngine(
                data_manager=self.data_manager,
                score_analyzer=self.score_analyzer,
                multi_selection_manager=self.multi_selection_manager
            )
            
            self.logger.info("Core components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Core component initialization failed: {str(e)}")
            raise
    
    def _initialize_report_generators(self):
        """Initialize report generator components"""
        try:
            # Initialize DC Score Intelligence Reports
            self.dc_score_reports = DCScoreIntelligenceReports(
                data_manager=self.data_manager,
                score_analyzer=self.score_analyzer,
                multi_selection_manager=self.multi_selection_manager
            )
            
            # Initialize Business Outcome Reports
            self.business_outcome_reports = BusinessOutcomeReports(
                data_manager=self.data_manager,
                score_analyzer=self.score_analyzer,
                multi_selection_manager=self.multi_selection_manager
            )
            
            # Initialize Predictive Intelligence Reports
            self.predictive_reports = PredictiveIntelligenceReports(
                data_manager=self.data_manager,
                score_analyzer=self.score_analyzer,
                multi_selection_manager=self.multi_selection_manager
            )
            
            # Initialize Executive Intelligence Reports
            self.executive_reports = ExecutiveIntelligenceReports(
                data_manager=self.data_manager,
                score_analyzer=self.score_analyzer,
                multi_selection_manager=self.multi_selection_manager
            )
            
            self.logger.info("Report generators initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Report generator initialization failed: {str(e)}")
            raise
    
    def _validate_system_integration(self):
        """Validate system component integration"""
        try:
            # Test data manager connectivity
            if not self.data_manager.test_connection():
                raise Exception("Data manager connection test failed")
            
            # Test API client connectivity
            if not self.api_client.test_connectivity():
                self.logger.warning("API client connectivity test failed - some features may be limited")
            
            # Test score analyzer functionality
            if not self.score_analyzer.validate_analyzer():
                raise Exception("Score analyzer validation failed")
            
            self.logger.info("System integration validation completed")
            
        except Exception as e:
            self.logger.error(f"System integration validation failed: {str(e)}")
            raise
    
    def _load_available_reports(self):
        """Load list of available reports"""
        try:
            self.available_reports = [
                # DC Score Intelligence Reports (8 reports)
                {
                    'id': 'dc_score_performance_analysis',
                    'name': 'Dynamic DC Score Performance Analysis',
                    'category': 'dc_score_intelligence',
                    'generator': self.dc_score_reports.generate_dc_score_performance_analysis
                },
                {
                    'id': 'sectional_score_deep_dive',
                    'name': 'Sectional Score Deep Dive Analysis',
                    'category': 'dc_score_intelligence',
                    'generator': self.dc_score_reports.generate_sectional_score_deep_dive
                },
                {
                    'id': 'score_revenue_correlation',
                    'name': 'Score-to-Revenue Correlation Analysis',
                    'category': 'dc_score_intelligence',
                    'generator': self.dc_score_reports.generate_score_revenue_correlation_analysis
                },
                {
                    'id': 'market_share_impact',
                    'name': 'Market Share Impact Analysis',
                    'category': 'dc_score_intelligence',
                    'generator': self.dc_score_reports.generate_market_share_impact_analysis
                },
                {
                    'id': 'customer_acquisition_efficiency',
                    'name': 'Customer Acquisition Efficiency Analysis',
                    'category': 'dc_score_intelligence',
                    'generator': self.dc_score_reports.generate_customer_acquisition_efficiency_analysis
                },
                {
                    'id': 'brand_equity_correlation',
                    'name': 'Brand Equity Correlation Analysis',
                    'category': 'dc_score_intelligence',
                    'generator': self.dc_score_reports.generate_brand_equity_correlation_analysis
                },
                {
                    'id': 'bestseller_rank_optimization',
                    'name': 'Bestseller Rank Optimization Analysis',
                    'category': 'dc_score_intelligence',
                    'generator': self.dc_score_reports.generate_bestseller_rank_optimization_analysis
                },
                {
                    'id': 'sales_performance_correlation',
                    'name': 'Sales Performance Correlation Analysis',
                    'category': 'dc_score_intelligence',
                    'generator': self.dc_score_reports.generate_sales_performance_correlation_analysis
                },
                
                # Business Outcome Reports (8 reports)
                {
                    'id': 'revenue_impact_optimization',
                    'name': 'Revenue Impact Optimization',
                    'category': 'business_outcome',
                    'generator': self.business_outcome_reports.generate_revenue_impact_optimization
                },
                {
                    'id': 'market_position_enhancement',
                    'name': 'Market Position Enhancement Strategy',
                    'category': 'business_outcome',
                    'generator': self.business_outcome_reports.generate_market_position_enhancement_strategy
                },
                {
                    'id': 'customer_lifetime_value_enhancement',
                    'name': 'Customer Lifetime Value Enhancement',
                    'category': 'business_outcome',
                    'generator': self.business_outcome_reports.generate_customer_lifetime_value_enhancement
                },
                {
                    'id': 'conversion_rate_optimization',
                    'name': 'Conversion Rate Optimization Analysis',
                    'category': 'business_outcome',
                    'generator': self.business_outcome_reports.generate_conversion_rate_optimization_analysis
                },
                {
                    'id': 'roi_maximization_strategy',
                    'name': 'ROI Maximization Strategy',
                    'category': 'business_outcome',
                    'generator': self.business_outcome_reports.generate_roi_maximization_strategy
                },
                {
                    'id': 'competitive_advantage_analysis',
                    'name': 'Competitive Advantage Analysis',
                    'category': 'business_outcome',
                    'generator': self.business_outcome_reports.generate_competitive_advantage_analysis
                },
                {
                    'id': 'market_penetration_strategy',
                    'name': 'Market Penetration Strategy',
                    'category': 'business_outcome',
                    'generator': self.business_outcome_reports.generate_market_penetration_strategy
                },
                {
                    'id': 'brand_portfolio_optimization',
                    'name': 'Brand Portfolio Optimization',
                    'category': 'business_outcome',
                    'generator': self.business_outcome_reports.generate_brand_portfolio_optimization
                },
                
                # Predictive Intelligence Reports (6 reports)
                {
                    'id': 'performance_forecasting_analysis',
                    'name': 'Performance Forecasting Analysis',
                    'category': 'predictive_intelligence',
                    'generator': self.predictive_reports.generate_performance_forecasting_analysis
                },
                {
                    'id': 'trend_prediction_analysis',
                    'name': 'Trend Prediction Analysis',
                    'category': 'predictive_intelligence',
                    'generator': self.predictive_reports.generate_trend_prediction_analysis
                },
                {
                    'id': 'risk_assessment_analysis',
                    'name': 'Risk Assessment Analysis',
                    'category': 'predictive_intelligence',
                    'generator': self.predictive_reports.generate_risk_assessment_analysis
                },
                {
                    'id': 'opportunity_identification_analysis',
                    'name': 'Opportunity Identification Analysis',
                    'category': 'predictive_intelligence',
                    'generator': self.predictive_reports.generate_opportunity_identification_analysis
                },
                {
                    'id': 'scenario_planning_analysis',
                    'name': 'Scenario Planning Analysis',
                    'category': 'predictive_intelligence',
                    'generator': self.predictive_reports.generate_scenario_planning_analysis
                },
                {
                    'id': 'growth_trajectory_modeling',
                    'name': 'Growth Trajectory Modeling',
                    'category': 'predictive_intelligence',
                    'generator': self.predictive_reports.generate_growth_trajectory_modeling
                },
                
                # Executive Intelligence Reports (3 reports)
                {
                    'id': 'executive_performance_dashboard',
                    'name': 'Executive Performance Dashboard',
                    'category': 'executive_intelligence',
                    'generator': self.executive_reports.generate_executive_performance_dashboard
                },
                {
                    'id': 'strategic_planning_insights',
                    'name': 'Strategic Planning Insights',
                    'category': 'executive_intelligence',
                    'generator': self.executive_reports.generate_strategic_planning_insights
                },
                {
                    'id': 'investment_priority_analysis',
                    'name': 'Investment Priority Analysis',
                    'category': 'executive_intelligence',
                    'generator': self.executive_reports.generate_investment_priority_analysis
                }
            ]
            
            self.logger.info(f"Loaded {len(self.available_reports)} available reports")
            
        except Exception as e:
            self.logger.error(f"Failed to load available reports: {str(e)}")
            self.available_reports = []
    
    def generate_report(self, report_id: str, selected_projects: List[int], 
                       selected_brands: List[str], 
                       customization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a specific report
        
        Args:
            report_id: ID of the report to generate
            selected_projects: List of selected project IDs
            selected_brands: List of selected brand names
            customization_params: Optional customization parameters
            
        Returns:
            Dict containing the generated report
        """
        try:
            if not self.is_initialized:
                raise Exception("System not initialized. Call initialize_system() first.")
            
            # Find the report
            report_config = None
            for report in self.available_reports:
                if report['id'] == report_id:
                    report_config = report
                    break
            
            if not report_config:
                raise Exception(f"Report '{report_id}' not found")
            
            self.logger.info(f"Generating report: {report_config['name']}")
            
            # Optimize hyperparameters for this specific analysis
            optimized_params = self.hyperparameter_optimizer.optimize_for_analysis(
                selected_projects, selected_brands, report_config['category']
            )
            
            # Merge with customization parameters
            final_params = {**(customization_params or {}), **optimized_params}
            
            # Generate the report
            report_data = report_config['generator'](
                selected_projects, selected_brands, final_params
            )
            
            # Add system metadata
            report_data['system_metadata'] = {
                'generation_timestamp': datetime.now().isoformat(),
                'system_version': '1.0.0',
                'optimized_parameters': optimized_params,
                'data_quality_score': self._assess_data_quality(selected_projects, selected_brands)
            }
            
            self.logger.info(f"Report '{report_id}' generated successfully")
            return report_data
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise
    
    def generate_multiple_reports(self, report_ids: List[str], selected_projects: List[int], 
                                 selected_brands: List[str],
                                 customization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate multiple reports
        
        Args:
            report_ids: List of report IDs to generate
            selected_projects: List of selected project IDs
            selected_brands: List of selected brand names
            customization_params: Optional customization parameters
            
        Returns:
            Dict containing all generated reports
        """
        try:
            self.logger.info(f"Generating {len(report_ids)} reports...")
            
            reports = {}
            failed_reports = []
            
            for report_id in report_ids:
                try:
                    reports[report_id] = self.generate_report(
                        report_id, selected_projects, selected_brands, customization_params
                    )
                except Exception as e:
                    self.logger.error(f"Failed to generate report '{report_id}': {str(e)}")
                    failed_reports.append(report_id)
            
            result = {
                'generated_reports': reports,
                'failed_reports': failed_reports,
                'generation_summary': {
                    'total_requested': len(report_ids),
                    'successfully_generated': len(reports),
                    'failed': len(failed_reports),
                    'generation_timestamp': datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"Multiple report generation completed: {len(reports)} successful, {len(failed_reports)} failed")
            return result
            
        except Exception as e:
            self.logger.error(f"Multiple report generation failed: {str(e)}")
            raise
    
    def get_available_reports(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available reports
        
        Args:
            category: Optional category filter
            
        Returns:
            List of available reports
        """
        if category:
            return [report for report in self.available_reports if report['category'] == category]
        return self.available_reports
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status information
        
        Returns:
            Dict containing system status
        """
        try:
            status = {
                'is_initialized': self.is_initialized,
                'available_reports_count': len(self.available_reports),
                'core_components_status': {
                    'data_manager': self.data_manager is not None,
                    'api_client': self.api_client is not None,
                    'score_analyzer': self.score_analyzer is not None,
                    'hyperparameter_optimizer': self.hyperparameter_optimizer is not None,
                    'multi_selection_manager': self.multi_selection_manager is not None,
                    'report_engine': self.report_engine is not None
                },
                'report_generators_status': {
                    'dc_score_reports': self.dc_score_reports is not None,
                    'business_outcome_reports': self.business_outcome_reports is not None,
                    'predictive_reports': self.predictive_reports is not None,
                    'executive_reports': self.executive_reports is not None
                },
                'system_metrics': self.system_metrics,
                'last_updated': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {str(e)}")
            return {'error': str(e)}
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default system configuration"""
        return {
            'api_config': {
                'timeout': 30,
                'retry_attempts': 3,
                'retry_delay': 1
            },
            'database_config': {
                'connection_timeout': 30,
                'query_timeout': 60
            },
            'api_base_urls': [
                'http://localhost:7000',
                'http://localhost:8001',
                'http://localhost:8002'
            ],
            'auth_config': {
                'auth_type': 'bearer_token',
                'token_refresh_threshold': 300
            },
            'analysis_config': {
                'confidence_threshold': 0.8,
                'min_data_points': 10
            },
            'optimization_config': {
                'n_trials': 100,
                'timeout': 300,
                'n_jobs': -1
            }
        }
    
    def _assess_data_quality(self, selected_projects: List[int], selected_brands: List[str]) -> float:
        """Assess data quality for selected projects and brands"""
        try:
            if self.data_manager:
                return self.data_manager.assess_data_quality(selected_projects, selected_brands)
            return 0.0
        except Exception as e:
            self.logger.error(f"Data quality assessment failed: {str(e)}")
            return 0.0

# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    system = DigiCadenceDynamicSystem()
    
    # Initialize all components
    if system.initialize_system():
        print("✅ Digi-Cadence Dynamic System initialized successfully!")
        
        # Get system status
        status = system.get_system_status()
        print(f"📊 System Status: {status['is_initialized']}")
        print(f"📈 Available Reports: {status['available_reports_count']}")
        
        # Get available reports by category
        dc_reports = system.get_available_reports('dc_score_intelligence')
        print(f"🎯 DC Score Intelligence Reports: {len(dc_reports)}")
        
        business_reports = system.get_available_reports('business_outcome')
        print(f"💼 Business Outcome Reports: {len(business_reports)}")
        
        predictive_reports = system.get_available_reports('predictive_intelligence')
        print(f"🔮 Predictive Intelligence Reports: {len(predictive_reports)}")
        
        executive_reports = system.get_available_reports('executive_intelligence')
        print(f"👔 Executive Intelligence Reports: {len(executive_reports)}")
        
        print("\n🚀 System ready for report generation!")
        
    else:
        print("❌ System initialization failed!")

