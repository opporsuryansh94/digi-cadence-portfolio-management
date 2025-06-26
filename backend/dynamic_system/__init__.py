"""
Digi-Cadence Dynamic Enhancement System

A comprehensive, adaptive analytics platform that provides intelligent insights 
and strategic recommendations based on DC scores and business outcomes.

Features:
- 25 dynamic reports across 4 categories
- Automatic hyperparameter tuning
- Multi-brand/multi-project analysis capabilities
- Real-time API integration
- Contextual insights generation
"""

__version__ = "1.0.0"
__author__ = "Digi-Cadence Development Team"

# Import main system class
from .digi_cadence_dynamic_system import DigiCadenceDynamicSystem

# Import enhanced recommendation system
from .enhanced_digi_cadence_recommendation_system import EnhancedDigiCadenceRecommendationSystem

# Import core components
from .dynamic_data_manager import DynamicDataManager
from .adaptive_api_client import AdaptiveAPIClient
from .dynamic_score_analyzer import DynamicScoreAnalyzer
from .adaptive_hyperparameter_optimizer import AdaptiveHyperparameterOptimizer
from .dynamic_multi_selection_manager import DynamicMultiSelectionManager
from .dynamic_report_intelligence_engine import DynamicReportIntelligenceEngine

# Import report generators
from .dc_score_intelligence_reports import DCScoreIntelligenceReports
from .business_outcome_reports import BusinessOutcomeReports
from .predictive_intelligence_reports import PredictiveIntelligenceReports
from .executive_intelligence_reports import ExecutiveIntelligenceReports

__all__ = [
    'DigiCadenceDynamicSystem',
    'EnhancedDigiCadenceRecommendationSystem',
    'DynamicDataManager',
    'AdaptiveAPIClient',
    'DynamicScoreAnalyzer',
    'AdaptiveHyperparameterOptimizer',
    'DynamicMultiSelectionManager',
    'DynamicReportIntelligenceEngine',
    'DCScoreIntelligenceReports',
    'BusinessOutcomeReports',
    'PredictiveIntelligenceReports',
    'ExecutiveIntelligenceReports'
]

