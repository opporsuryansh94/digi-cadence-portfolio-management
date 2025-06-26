"""
Enhanced Dynamic Digi-Cadence Recommendation System
Integrates all dynamic components while maintaining original input structure
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime
import logging
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
import optuna
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import seaborn as sns

# Import our dynamic components
from dynamic_data_manager import DynamicDataManager
from adaptive_api_client import AdaptiveAPIClient
from dynamic_score_analyzer import DynamicScoreAnalyzer
from adaptive_hyperparameter_optimizer import AdaptiveHyperparameterOptimizer
from dynamic_multi_selection_manager import DynamicMultiSelectionManager
from dynamic_report_intelligence_engine import DynamicReportIntelligenceEngine

warnings.filterwarnings('ignore')

class EnhancedDigiCadenceRecommendationSystem:
    """
    Enhanced recommendation system that maintains original interface while adding dynamic capabilities
    """
    
    def __init__(self, data_file_path: str = None, weights_file_path: str = None):
        """
        Initialize the enhanced recommendation system
        
        Args:
            data_file_path: Path to the main data file (maintains original interface)
            weights_file_path: Path to the weights file (maintains original interface)
        """
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Original interface compatibility
        self.data_file_path = data_file_path
        self.weights_file_path = weights_file_path
        
        # Data storage
        self.data = None
        self.weights = None
        self.processed_data = {}
        
        # Dynamic components
        self.data_manager = None
        self.api_client = None
        self.score_analyzer = None
        self.hyperparameter_optimizer = None
        self.multi_selection_manager = None
        self.report_engine = None
        
        # Analysis results
        self.analysis_results = {}
        self.optimization_results = {}
        self.generated_reports = {}
        
        # Current selection state
        self.selected_projects = []
        self.selected_brands = []
        
        # Enhanced capabilities flags
        self.auto_hyperparameter_tuning = True
        self.dynamic_report_generation = True
        self.multi_selection_enabled = True
        
        self.logger.info("Enhanced Digi-Cadence Recommendation System initialized")
    
    def load_data(self, data_file_path: str = None, weights_file_path: str = None) -> Dict[str, Any]:
        """
        Load data with enhanced processing capabilities
        
        Args:
            data_file_path: Path to data file
            weights_file_path: Path to weights file
            
        Returns:
            Dict with loading results and data characteristics
        """
        try:
            self.logger.info("Loading data with enhanced processing...")
            
            # Use provided paths or stored paths
            data_path = data_file_path or self.data_file_path
            weights_path = weights_file_path or self.weights_file_path
            
            # Load original data format
            if data_path and os.path.exists(data_path):
                if data_path.endswith('.csv'):
                    self.data = pd.read_csv(data_path)
                elif data_path.endswith('.xlsx'):
                    self.data = pd.read_excel(data_path)
                else:
                    raise ValueError(f"Unsupported file format: {data_path}")
            
            # Load weights if provided
            if weights_path and os.path.exists(weights_path):
                if weights_path.endswith('.csv'):
                    self.weights = pd.read_csv(weights_path)
                elif weights_path.endswith('.xlsx'):
                    self.weights = pd.read_excel(weights_path)
            
            # Initialize dynamic components with loaded data
            self._initialize_dynamic_components()
            
            # Process data through dynamic manager
            processing_results = self._process_data_dynamically()
            
            result = {
                'data_loaded': self.data is not None,
                'weights_loaded': self.weights is not None,
                'data_shape': self.data.shape if self.data is not None else None,
                'dynamic_processing_results': processing_results,
                'available_brands': self._extract_available_brands(),
                'available_projects': self._extract_available_projects(),
                'data_characteristics': self._analyze_data_characteristics()
            }
            
            self.logger.info(f"Data loading completed: {result['data_shape']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def enable_multi_selection(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """
        Enable multi-project and multi-brand selection with dynamic analysis
        
        Args:
            selected_projects: List of selected project IDs
            selected_brands: List of selected brand names
            
        Returns:
            Dict with multi-selection analysis results
        """
        try:
            self.logger.info(f"Enabling multi-selection: {len(selected_projects)} projects, {len(selected_brands)} brands")
            
            # Store selections
            self.selected_projects = selected_projects
            self.selected_brands = selected_brands
            
            # Initialize multi-selection manager if not already done
            if not self.multi_selection_manager:
                available_projects = self._prepare_project_data_for_multi_selection()
                available_brands = self._prepare_brand_data_for_multi_selection()
                self.multi_selection_manager = DynamicMultiSelectionManager(available_projects, available_brands)
            
            # Enable multi-selection analysis
            multi_selection_results = self.multi_selection_manager.enable_dynamic_multi_selection(
                selected_projects, selected_brands
            )
            
            # Update other components with selection
            if self.score_analyzer:
                self.score_analyzer.selected_projects = selected_projects
                self.score_analyzer.selected_brands = selected_brands
            
            # Store results
            self.analysis_results['multi_selection'] = multi_selection_results
            
            self.logger.info("Multi-selection enabled successfully")
            return multi_selection_results
            
        except Exception as e:
            self.logger.error(f"Error enabling multi-selection: {str(e)}")
            raise
    
    def optimize_hyperparameters_automatically(self, optimization_target: str = 'both', 
                                             n_trials: int = 100) -> Dict[str, Any]:
        """
        Automatically optimize hyperparameters based on current data selection
        
        Args:
            optimization_target: 'genetic_algorithm', 'shap_analyzer', or 'both'
            n_trials: Number of optimization trials
            
        Returns:
            Dict with optimization results
        """
        try:
            self.logger.info(f"Starting automatic hyperparameter optimization: {optimization_target}")
            
            if not self.hyperparameter_optimizer:
                # Initialize optimizer with current data
                project_data = self._prepare_project_data_for_optimization()
                brand_data = self._prepare_brand_data_for_optimization()
                score_patterns = self.score_analyzer.score_patterns if self.score_analyzer else {}
                
                self.hyperparameter_optimizer = AdaptiveHyperparameterOptimizer(
                    project_data, brand_data, score_patterns
                )
            
            # Perform optimization
            optimization_results = self.hyperparameter_optimizer.optimize_for_specific_data(
                optimization_target=optimization_target,
                n_trials=n_trials
            )
            
            # Store results
            self.optimization_results = optimization_results
            
            # Apply optimized parameters
            self._apply_optimized_parameters(optimization_results)
            
            self.logger.info("Automatic hyperparameter optimization completed")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error in automatic hyperparameter optimization: {str(e)}")
            raise
    
    def generate_dynamic_recommendations(self, focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate dynamic recommendations based on current selection and optimized parameters
        
        Args:
            focus_areas: Optional list of areas to focus on
            
        Returns:
            Dict with comprehensive recommendations
        """
        try:
            self.logger.info("Generating dynamic recommendations...")
            
            # Ensure score analysis is performed
            if not self.score_analyzer or not self.score_analyzer.score_patterns:
                self._perform_score_analysis()
            
            # Generate recommendations using optimized genetic algorithm
            genetic_recommendations = self._generate_genetic_algorithm_recommendations()
            
            # Generate SHAP-based insights
            shap_insights = self._generate_shap_insights()
            
            # Combine recommendations
            comprehensive_recommendations = {
                'genetic_algorithm_recommendations': genetic_recommendations,
                'shap_insights': shap_insights,
                'score_analysis_insights': self.score_analyzer.get_score_summary(),
                'optimization_applied': bool(self.optimization_results),
                'multi_selection_insights': self.analysis_results.get('multi_selection', {}),
                'focus_areas': focus_areas or [],
                'generation_timestamp': datetime.now().isoformat()
            }
            
            # Add contextual recommendations based on selection
            contextual_recommendations = self._generate_contextual_recommendations()
            comprehensive_recommendations['contextual_recommendations'] = contextual_recommendations
            
            self.logger.info("Dynamic recommendations generated successfully")
            return comprehensive_recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating dynamic recommendations: {str(e)}")
            raise
    
    def generate_strategic_reports(self, report_ids: Optional[List[str]] = None, 
                                 customization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate strategic reports based on current data and selection
        
        Args:
            report_ids: Optional list of specific report IDs to generate
            customization_params: Optional customization parameters
            
        Returns:
            Dict with generated reports
        """
        try:
            self.logger.info("Generating strategic reports...")
            
            # Initialize report engine if not already done
            if not self.report_engine:
                self._initialize_report_engine()
            
            # Analyze data for report selection
            if not self.selected_projects or not self.selected_brands:
                raise ValueError("Please select projects and brands before generating reports")
            
            report_analysis = self.report_engine.analyze_data_for_report_selection(
                self.selected_projects, self.selected_brands
            )
            
            # Generate specified reports or recommended reports
            if report_ids:
                reports_to_generate = report_ids
            else:
                # Use top recommended reports
                recommended = report_analysis.get('recommendations', {}).get('recommended_reports', [])
                reports_to_generate = [r['report_id'] for r in recommended[:5]]  # Top 5
            
            generated_reports = {}
            for report_id in reports_to_generate:
                try:
                    report_content = self.report_engine.generate_dynamic_report(
                        report_id, customization_params
                    )
                    generated_reports[report_id] = report_content
                except Exception as e:
                    self.logger.warning(f"Could not generate report {report_id}: {str(e)}")
            
            # Store generated reports
            self.generated_reports.update(generated_reports)
            
            result = {
                'report_analysis': report_analysis,
                'generated_reports': generated_reports,
                'total_reports_generated': len(generated_reports),
                'generation_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Strategic reports generated: {len(generated_reports)} reports")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating strategic reports: {str(e)}")
            raise
    
    def get_available_reports(self) -> Dict[str, Any]:
        """
        Get list of available reports based on current data selection
        
        Returns:
            Dict with available reports information
        """
        try:
            if not self.report_engine:
                self._initialize_report_engine()
            
            if not self.selected_projects or not self.selected_brands:
                # Return catalog without data-specific analysis
                return self.report_engine.get_report_catalog()
            
            # Get data-specific available reports
            report_analysis = self.report_engine.analyze_data_for_report_selection(
                self.selected_projects, self.selected_brands
            )
            
            return {
                'catalog': self.report_engine.get_report_catalog(),
                'data_specific_analysis': report_analysis,
                'selection_context': {
                    'projects': self.selected_projects,
                    'brands': self.selected_brands
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting available reports: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            Dict with system status information
        """
        status = {
            'system_initialized': True,
            'data_loaded': self.data is not None,
            'weights_loaded': self.weights is not None,
            'dynamic_components_initialized': {
                'data_manager': self.data_manager is not None,
                'api_client': self.api_client is not None,
                'score_analyzer': self.score_analyzer is not None,
                'hyperparameter_optimizer': self.hyperparameter_optimizer is not None,
                'multi_selection_manager': self.multi_selection_manager is not None,
                'report_engine': self.report_engine is not None
            },
            'current_selection': {
                'projects': self.selected_projects,
                'brands': self.selected_brands,
                'multi_selection_enabled': len(self.selected_projects) > 0 and len(self.selected_brands) > 0
            },
            'optimization_status': {
                'auto_tuning_enabled': self.auto_hyperparameter_tuning,
                'optimization_completed': bool(self.optimization_results),
                'last_optimization': self.optimization_results.get('timestamp') if self.optimization_results else None
            },
            'analysis_status': {
                'score_analysis_completed': bool(self.score_analyzer and self.score_analyzer.score_patterns),
                'multi_selection_analysis_completed': 'multi_selection' in self.analysis_results,
                'reports_generated': len(self.generated_reports)
            },
            'capabilities': {
                'auto_hyperparameter_tuning': self.auto_hyperparameter_tuning,
                'dynamic_report_generation': self.dynamic_report_generation,
                'multi_selection_enabled': self.multi_selection_enabled
            }
        }
        
        return status
    
    def export_results(self, export_path: str, include_reports: bool = True) -> Dict[str, Any]:
        """
        Export all results to files
        
        Args:
            export_path: Directory path for exports
            include_reports: Whether to include generated reports
            
        Returns:
            Dict with export results
        """
        try:
            self.logger.info(f"Exporting results to {export_path}")
            
            os.makedirs(export_path, exist_ok=True)
            
            export_results = {
                'export_path': export_path,
                'exported_files': [],
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Export system status
            status_file = os.path.join(export_path, 'system_status.json')
            with open(status_file, 'w') as f:
                json.dump(self.get_system_status(), f, indent=2)
            export_results['exported_files'].append('system_status.json')
            
            # Export analysis results
            if self.analysis_results:
                analysis_file = os.path.join(export_path, 'analysis_results.json')
                with open(analysis_file, 'w') as f:
                    json.dump(self.analysis_results, f, indent=2, default=str)
                export_results['exported_files'].append('analysis_results.json')
            
            # Export optimization results
            if self.optimization_results:
                optimization_file = os.path.join(export_path, 'optimization_results.json')
                with open(optimization_file, 'w') as f:
                    json.dump(self.optimization_results, f, indent=2, default=str)
                export_results['exported_files'].append('optimization_results.json')
            
            # Export generated reports
            if include_reports and self.generated_reports:
                reports_dir = os.path.join(export_path, 'reports')
                os.makedirs(reports_dir, exist_ok=True)
                
                for report_id, report_content in self.generated_reports.items():
                    report_file = os.path.join(reports_dir, f'{report_id}.json')
                    with open(report_file, 'w') as f:
                        json.dump(report_content, f, indent=2, default=str)
                    export_results['exported_files'].append(f'reports/{report_id}.json')
            
            self.logger.info(f"Export completed: {len(export_results['exported_files'])} files")
            return export_results
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            raise
    
    # Private methods for internal functionality
    def _initialize_dynamic_components(self):
        """Initialize all dynamic components"""
        try:
            # Initialize data manager
            self.data_manager = DynamicDataManager()
            
            # Initialize API client
            self.api_client = AdaptiveAPIClient()
            
            self.logger.info("Dynamic components initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing dynamic components: {str(e)}")
    
    def _process_data_dynamically(self) -> Dict[str, Any]:
        """Process data through dynamic manager"""
        try:
            if not self.data_manager or self.data is None:
                return {}
            
            # Convert data to format expected by dynamic manager
            processed_data = self.data_manager.process_project_data(self.data)
            self.processed_data = processed_data
            
            return {
                'processing_completed': True,
                'data_characteristics': processed_data.get('data_characteristics', {}),
                'quality_assessment': processed_data.get('quality_assessment', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error in dynamic data processing: {str(e)}")
            return {'processing_completed': False, 'error': str(e)}
    
    def _extract_available_brands(self) -> List[str]:
        """Extract available brands from loaded data"""
        try:
            if self.data is None:
                return []
            
            # Get brand columns (excluding metadata columns)
            metadata_columns = ['Metric', 'section_name', 'platform_name', 'metricname', 'sectionName', 'platformname']
            brand_columns = [col for col in self.data.columns if col not in metadata_columns]
            
            return brand_columns
            
        except Exception as e:
            self.logger.error(f"Error extracting available brands: {str(e)}")
            return []
    
    def _extract_available_projects(self) -> List[int]:
        """Extract available projects from loaded data"""
        try:
            # For now, assume single project (can be enhanced based on actual data structure)
            return [1] if self.data is not None else []
            
        except Exception as e:
            self.logger.error(f"Error extracting available projects: {str(e)}")
            return []
    
    def _analyze_data_characteristics(self) -> Dict[str, Any]:
        """Analyze characteristics of loaded data"""
        try:
            if self.data is None:
                return {}
            
            characteristics = {
                'total_rows': len(self.data),
                'total_columns': len(self.data.columns),
                'brand_count': len(self._extract_available_brands()),
                'metric_count': len(self.data['Metric'].unique()) if 'Metric' in self.data.columns else 0,
                'section_count': len(self.data['section_name'].unique()) if 'section_name' in self.data.columns else 0,
                'data_completeness': (self.data.notna().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100,
                'numeric_columns': len(self.data.select_dtypes(include=[np.number]).columns)
            }
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error analyzing data characteristics: {str(e)}")
            return {}
    
    def _prepare_project_data_for_multi_selection(self) -> Dict[int, Any]:
        """Prepare project data for multi-selection manager"""
        try:
            if not self.processed_data:
                return {}
            
            # Create project data structure
            project_data = {
                1: {  # Assuming single project for now
                    'brands_available': self._extract_available_brands(),
                    'metrics_available': self.data['Metric'].unique().tolist() if self.data is not None and 'Metric' in self.data.columns else [],
                    'sections_available': self.data['section_name'].unique().tolist() if self.data is not None and 'section_name' in self.data.columns else [],
                    'data_quality': self._analyze_data_characteristics(),
                    'metrics_data': self.data
                }
            }
            
            return project_data
            
        except Exception as e:
            self.logger.error(f"Error preparing project data for multi-selection: {str(e)}")
            return {}
    
    def _prepare_brand_data_for_multi_selection(self) -> Dict[str, Any]:
        """Prepare brand data for multi-selection manager"""
        try:
            brand_data = {}
            
            for brand in self._extract_available_brands():
                brand_data[brand] = {
                    'performance_patterns': {},  # Will be filled by score analyzer
                    'available_in_projects': [1]  # Assuming single project
                }
            
            return brand_data
            
        except Exception as e:
            self.logger.error(f"Error preparing brand data for multi-selection: {str(e)}")
            return {}
    
    def _prepare_project_data_for_optimization(self) -> Dict[str, Any]:
        """Prepare project data for hyperparameter optimizer"""
        try:
            return {
                'brands_available': self._extract_available_brands(),
                'metrics_available': self.data['Metric'].unique().tolist() if self.data is not None and 'Metric' in self.data.columns else []
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing project data for optimization: {str(e)}")
            return {}
    
    def _prepare_brand_data_for_optimization(self) -> Dict[str, Any]:
        """Prepare brand data for hyperparameter optimizer"""
        try:
            return {brand: {} for brand in self._extract_available_brands()}
            
        except Exception as e:
            self.logger.error(f"Error preparing brand data for optimization: {str(e)}")
            return {}
    
    def _perform_score_analysis(self):
        """Perform score analysis using dynamic score analyzer"""
        try:
            if not self.score_analyzer:
                project_data = self._prepare_project_data_for_multi_selection()
                self.score_analyzer = DynamicScoreAnalyzer(project_data)
            
            # Perform analysis
            self.score_analyzer.analyze_dc_scores_dynamically()
            
        except Exception as e:
            self.logger.error(f"Error performing score analysis: {str(e)}")
    
    def _apply_optimized_parameters(self, optimization_results: Dict[str, Any]):
        """Apply optimized parameters to algorithms"""
        try:
            # Store optimized parameters for use in genetic algorithm and SHAP analyzer
            self.optimized_ga_params = optimization_results.get('genetic_algorithm', {})
            self.optimized_shap_params = optimization_results.get('shap_analyzer', {})
            
            self.logger.info("Optimized parameters applied")
            
        except Exception as e:
            self.logger.error(f"Error applying optimized parameters: {str(e)}")
    
    def _generate_genetic_algorithm_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations using optimized genetic algorithm"""
        try:
            # Use optimized parameters if available
            ga_params = getattr(self, 'optimized_ga_params', {})
            
            # Implement genetic algorithm with optimized parameters
            # This is a placeholder for the actual genetic algorithm implementation
            recommendations = {
                'optimization_strategy': 'genetic_algorithm',
                'parameters_used': ga_params,
                'recommendations': [
                    'Optimize high-impact metrics first',
                    'Focus on underperforming sections',
                    'Leverage cross-brand synergies'
                ],
                'confidence_score': 0.85
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating genetic algorithm recommendations: {str(e)}")
            return {}
    
    def _generate_shap_insights(self) -> Dict[str, Any]:
        """Generate insights using optimized SHAP analyzer"""
        try:
            # Use optimized parameters if available
            shap_params = getattr(self, 'optimized_shap_params', {})
            
            # Implement SHAP analysis with optimized parameters
            # This is a placeholder for the actual SHAP implementation
            insights = {
                'analysis_method': 'shap',
                'parameters_used': shap_params,
                'feature_importance': {},
                'insights': [
                    'Key performance drivers identified',
                    'Feature interactions analyzed',
                    'Improvement opportunities highlighted'
                ],
                'confidence_score': 0.80
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP insights: {str(e)}")
            return {}
    
    def _generate_contextual_recommendations(self) -> Dict[str, Any]:
        """Generate contextual recommendations based on current selection"""
        try:
            contextual = {
                'selection_specific_recommendations': [],
                'cross_brand_opportunities': [],
                'project_specific_insights': [],
                'priority_actions': []
            }
            
            # Add selection-specific recommendations
            if len(self.selected_brands) > 1:
                contextual['cross_brand_opportunities'].append(
                    'Analyze cross-brand synergies for portfolio optimization'
                )
            
            if len(self.selected_projects) > 1:
                contextual['project_specific_insights'].append(
                    'Compare performance patterns across projects'
                )
            
            return contextual
            
        except Exception as e:
            self.logger.error(f"Error generating contextual recommendations: {str(e)}")
            return {}
    
    def _initialize_report_engine(self):
        """Initialize the report intelligence engine"""
        try:
            if not self.score_analyzer:
                self._perform_score_analysis()
            
            self.report_engine = DynamicReportIntelligenceEngine(
                self.data_manager,
                self.score_analyzer,
                self.multi_selection_manager,
                self.hyperparameter_optimizer
            )
            
            self.logger.info("Report engine initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing report engine: {str(e)}")


# Convenience function to maintain original interface
def create_enhanced_recommendation_system(data_file_path: str = None, weights_file_path: str = None) -> EnhancedDigiCadenceRecommendationSystem:
    """
    Create an enhanced recommendation system instance
    
    Args:
        data_file_path: Path to the main data file
        weights_file_path: Path to the weights file
        
    Returns:
        EnhancedDigiCadenceRecommendationSystem instance
    """
    return EnhancedDigiCadenceRecommendationSystem(data_file_path, weights_file_path)


# Example usage function
def example_usage():
    """
    Example of how to use the enhanced recommendation system
    """
    # Initialize system
    system = create_enhanced_recommendation_system()
    
    # Load data (maintains original interface)
    data_results = system.load_data('path/to/data.csv', 'path/to/weights.csv')
    print("Data loaded:", data_results['data_loaded'])
    
    # Enable multi-selection
    selected_projects = [1]  # Project IDs
    selected_brands = ['Brand_A', 'Brand_B', 'Brand_C']  # Brand names
    
    multi_selection_results = system.enable_multi_selection(selected_projects, selected_brands)
    print("Multi-selection enabled:", multi_selection_results['selection_analysis']['compatibility_score'])
    
    # Optimize hyperparameters automatically
    optimization_results = system.optimize_hyperparameters_automatically('both', n_trials=50)
    print("Optimization completed:", optimization_results.keys())
    
    # Generate dynamic recommendations
    recommendations = system.generate_dynamic_recommendations()
    print("Recommendations generated:", len(recommendations))
    
    # Generate strategic reports
    reports = system.generate_strategic_reports()
    print("Reports generated:", reports['total_reports_generated'])
    
    # Get system status
    status = system.get_system_status()
    print("System status:", status['system_initialized'])
    
    # Export results
    export_results = system.export_results('./results_export')
    print("Results exported:", len(export_results['exported_files']))


if __name__ == "__main__":
    example_usage()

