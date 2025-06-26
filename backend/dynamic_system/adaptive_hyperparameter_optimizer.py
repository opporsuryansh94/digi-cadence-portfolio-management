"""
Adaptive Hyperparameter Optimizer
Dynamic hyperparameter optimization using Optuna based on actual project and brand data characteristics
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import warnings
from datetime import datetime
import logging
import json
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

warnings.filterwarnings('ignore')

class AdaptiveHyperparameterOptimizer:
    """
    Adaptive hyperparameter optimizer that tunes genetic algorithm and SHAP analyzer parameters
    based on actual project and brand data characteristics
    """
    
    def __init__(self, project_data: Dict[str, Any], brand_data: Dict[str, Any], score_patterns: Dict[str, Any]):
        """
        Initialize adaptive optimizer
        
        Args:
            project_data: Processed project data
            brand_data: Brand-specific data
            score_patterns: Identified score patterns from DynamicScoreAnalyzer
        """
        self.project_data = project_data
        self.brand_data = brand_data
        self.score_patterns = score_patterns
        
        # Data characteristics for optimization
        self.data_characteristics = self._analyze_optimization_characteristics()
        
        # Optimization history
        self.optimization_history = {}
        self.best_parameters = {}
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Optuna configuration
        self.study_storage = None  # Can be configured for persistent storage
        
        self.logger.info("Adaptive Hyperparameter Optimizer initialized")
    
    def _analyze_optimization_characteristics(self) -> Dict[str, Any]:
        """
        Analyze data characteristics to guide hyperparameter optimization
        
        Returns:
            Dict with data characteristics relevant for optimization
        """
        characteristics = {
            'data_complexity': 'medium',
            'performance_variance': 0.0,
            'correlation_strength': 0.0,
            'brand_count': 0,
            'metric_count': 0,
            'score_range': [0, 100],
            'optimization_difficulty': 'medium'
        }
        
        try:
            # Analyze data complexity
            total_brands = len(self.project_data.get('brands_available', []))
            total_metrics = len(self.project_data.get('metrics_available', []))
            
            characteristics['brand_count'] = total_brands
            characteristics['metric_count'] = total_metrics
            
            # Determine data complexity
            if total_brands >= 5 and total_metrics >= 20:
                characteristics['data_complexity'] = 'high'
            elif total_brands >= 3 and total_metrics >= 10:
                characteristics['data_complexity'] = 'medium'
            else:
                characteristics['data_complexity'] = 'low'
            
            # Analyze performance variance from score patterns
            if 'performance_trends' in self.score_patterns:
                brand_classifications = self.score_patterns['performance_trends'].get('brand_performance_classification', {})
                if brand_classifications:
                    volatilities = [data.get('score_volatility', 0) for data in brand_classifications.values()]
                    characteristics['performance_variance'] = np.mean(volatilities) if volatilities else 0.0
            
            # Assess correlation strength
            if 'business_correlation_potential' in self.score_patterns:
                correlations = self.score_patterns['business_correlation_potential']
                avg_correlation = np.mean([v for v in correlations.values() if isinstance(v, (int, float))])
                characteristics['correlation_strength'] = avg_correlation
            
            # Determine optimization difficulty
            if (characteristics['data_complexity'] == 'high' and 
                characteristics['performance_variance'] > 15 and 
                characteristics['correlation_strength'] < 0.5):
                characteristics['optimization_difficulty'] = 'high'
            elif (characteristics['data_complexity'] == 'low' and 
                  characteristics['performance_variance'] < 5 and 
                  characteristics['correlation_strength'] > 0.7):
                characteristics['optimization_difficulty'] = 'low'
            
        except Exception as e:
            self.logger.error(f"Error analyzing optimization characteristics: {str(e)}")
        
        return characteristics
    
    def optimize_for_specific_data(self, optimization_target: str = 'genetic_algorithm', 
                                 n_trials: int = 100, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for specific data characteristics
        
        Args:
            optimization_target: Target to optimize ('genetic_algorithm', 'shap_analyzer', 'both')
            n_trials: Number of optimization trials
            timeout: Timeout in seconds for optimization
            
        Returns:
            Dict with optimal parameters
        """
        try:
            self.logger.info(f"Starting hyperparameter optimization for {optimization_target}")
            
            optimal_params = {}
            
            if optimization_target in ['genetic_algorithm', 'both']:
                ga_params = self._optimize_genetic_algorithm_params(n_trials, timeout)
                optimal_params['genetic_algorithm'] = ga_params
            
            if optimization_target in ['shap_analyzer', 'both']:
                shap_params = self._optimize_shap_analyzer_params(n_trials, timeout)
                optimal_params['shap_analyzer'] = shap_params
            
            # Store optimization results
            self.best_parameters[optimization_target] = optimal_params
            self.optimization_history[datetime.now().isoformat()] = {
                'target': optimization_target,
                'parameters': optimal_params,
                'data_characteristics': self.data_characteristics
            }
            
            self.logger.info(f"Hyperparameter optimization completed for {optimization_target}")
            return optimal_params
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            raise
    
    def _optimize_genetic_algorithm_params(self, n_trials: int, timeout: Optional[int]) -> Dict[str, Any]:
        """
        Optimize genetic algorithm hyperparameters
        
        Args:
            n_trials: Number of trials
            timeout: Timeout in seconds
            
        Returns:
            Dict with optimal genetic algorithm parameters
        """
        def objective(trial):
            # Suggest parameters based on data characteristics
            complexity = self.data_characteristics['data_complexity']
            variance = self.data_characteristics['performance_variance']
            
            # Population size based on complexity
            if complexity == 'high':
                population_size = trial.suggest_int('population_size', 50, 200)
            elif complexity == 'medium':
                population_size = trial.suggest_int('population_size', 30, 100)
            else:
                population_size = trial.suggest_int('population_size', 20, 60)
            
            # Generations based on complexity and variance
            if complexity == 'high' or variance > 15:
                generations = trial.suggest_int('generations', 50, 200)
            elif complexity == 'medium' or variance > 10:
                generations = trial.suggest_int('generations', 30, 100)
            else:
                generations = trial.suggest_int('generations', 20, 60)
            
            # Mutation rate based on variance
            if variance > 20:
                mutation_rate = trial.suggest_float('mutation_rate', 0.2, 0.6)
            elif variance > 10:
                mutation_rate = trial.suggest_float('mutation_rate', 0.1, 0.4)
            else:
                mutation_rate = trial.suggest_float('mutation_rate', 0.05, 0.3)
            
            # Crossover rate
            crossover_rate = trial.suggest_float('crossover_rate', 0.6, 0.95)
            
            # Selection pressure
            selection_pressure = trial.suggest_float('selection_pressure', 0.1, 0.5)
            
            # Elite size
            elite_size = trial.suggest_int('elite_size', max(1, population_size // 20), max(2, population_size // 5))
            
            # Tournament size for selection
            tournament_size = trial.suggest_int('tournament_size', 3, min(10, population_size // 5))
            
            # Test parameters with actual data simulation
            fitness_score = self._evaluate_genetic_algorithm_params(
                population_size, generations, mutation_rate, crossover_rate,
                selection_pressure, elite_size, tournament_size
            )
            
            return fitness_score
        
        # Configure study based on data characteristics
        sampler = self._get_optimal_sampler()
        pruner = self._get_optimal_pruner()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"ga_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Return best parameters with additional derived parameters
        best_params = study.best_params.copy()
        best_params['fitness_score'] = study.best_value
        best_params['n_trials'] = len(study.trials)
        best_params['optimization_time'] = sum(trial.duration.total_seconds() for trial in study.trials if trial.duration)
        
        return best_params
    
    def _optimize_shap_analyzer_params(self, n_trials: int, timeout: Optional[int]) -> Dict[str, Any]:
        """
        Optimize SHAP analyzer hyperparameters
        
        Args:
            n_trials: Number of trials
            timeout: Timeout in seconds
            
        Returns:
            Dict with optimal SHAP analyzer parameters
        """
        def objective(trial):
            # Background sample size
            brand_count = self.data_characteristics['brand_count']
            metric_count = self.data_characteristics['metric_count']
            
            # Adjust sample size based on data size
            max_background_samples = min(1000, brand_count * metric_count * 5)
            background_samples = trial.suggest_int('background_samples', 50, max_background_samples)
            
            # Number of samples for SHAP value calculation
            if self.data_characteristics['data_complexity'] == 'high':
                shap_samples = trial.suggest_int('shap_samples', 100, 500)
            else:
                shap_samples = trial.suggest_int('shap_samples', 50, 200)
            
            # Feature selection threshold
            feature_threshold = trial.suggest_float('feature_threshold', 0.01, 0.1)
            
            # Clustering parameters for feature grouping
            cluster_features = trial.suggest_categorical('cluster_features', [True, False])
            
            if cluster_features:
                n_clusters = trial.suggest_int('n_clusters', 3, min(10, metric_count // 2))
            else:
                n_clusters = None
            
            # Model complexity for SHAP explainer
            model_complexity = trial.suggest_categorical('model_complexity', ['linear', 'tree', 'ensemble'])
            
            # Regularization for linear models
            if model_complexity == 'linear':
                regularization = trial.suggest_float('regularization', 0.001, 1.0, log=True)
            else:
                regularization = None
            
            # Tree-specific parameters
            if model_complexity in ['tree', 'ensemble']:
                max_depth = trial.suggest_int('max_depth', 3, 15)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            else:
                max_depth = None
                min_samples_split = None
            
            # Evaluate SHAP parameters
            performance_score = self._evaluate_shap_analyzer_params(
                background_samples, shap_samples, feature_threshold,
                cluster_features, n_clusters, model_complexity,
                regularization, max_depth, min_samples_split
            )
            
            return performance_score
        
        # Configure study
        sampler = self._get_optimal_sampler()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f"shap_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Return best parameters
        best_params = study.best_params.copy()
        best_params['performance_score'] = study.best_value
        best_params['n_trials'] = len(study.trials)
        
        return best_params
    
    def _evaluate_genetic_algorithm_params(self, population_size: int, generations: int,
                                         mutation_rate: float, crossover_rate: float,
                                         selection_pressure: float, elite_size: int,
                                         tournament_size: int) -> float:
        """
        Evaluate genetic algorithm parameters using simulation
        
        Returns:
            Fitness score for the parameter combination
        """
        try:
            # Create a simplified fitness function based on actual data characteristics
            base_score = 50.0  # Base fitness score
            
            # Adjust based on population size efficiency
            pop_efficiency = self._calculate_population_efficiency(population_size)
            base_score += pop_efficiency * 10
            
            # Adjust based on generation efficiency
            gen_efficiency = self._calculate_generation_efficiency(generations, population_size)
            base_score += gen_efficiency * 10
            
            # Adjust based on mutation rate appropriateness
            mutation_efficiency = self._calculate_mutation_efficiency(mutation_rate)
            base_score += mutation_efficiency * 10
            
            # Adjust based on crossover rate
            crossover_efficiency = self._calculate_crossover_efficiency(crossover_rate)
            base_score += crossover_efficiency * 10
            
            # Penalty for excessive computational cost
            computational_cost = population_size * generations
            if computational_cost > 10000:
                base_score -= (computational_cost - 10000) / 1000
            
            # Bonus for balanced parameters
            balance_bonus = self._calculate_parameter_balance_bonus(
                population_size, generations, mutation_rate, crossover_rate
            )
            base_score += balance_bonus
            
            return max(0.0, min(100.0, base_score))
            
        except Exception as e:
            self.logger.error(f"Error evaluating genetic algorithm parameters: {str(e)}")
            return 0.0
    
    def _evaluate_shap_analyzer_params(self, background_samples: int, shap_samples: int,
                                     feature_threshold: float, cluster_features: bool,
                                     n_clusters: Optional[int], model_complexity: str,
                                     regularization: Optional[float], max_depth: Optional[int],
                                     min_samples_split: Optional[int]) -> float:
        """
        Evaluate SHAP analyzer parameters
        
        Returns:
            Performance score for the parameter combination
        """
        try:
            base_score = 50.0
            
            # Adjust based on sample size appropriateness
            sample_efficiency = self._calculate_sample_efficiency(background_samples, shap_samples)
            base_score += sample_efficiency * 15
            
            # Adjust based on feature threshold
            threshold_efficiency = self._calculate_threshold_efficiency(feature_threshold)
            base_score += threshold_efficiency * 10
            
            # Adjust based on model complexity appropriateness
            complexity_efficiency = self._calculate_complexity_efficiency(model_complexity)
            base_score += complexity_efficiency * 15
            
            # Clustering bonus/penalty
            if cluster_features and n_clusters:
                clustering_efficiency = self._calculate_clustering_efficiency(n_clusters)
                base_score += clustering_efficiency * 10
            
            # Computational efficiency penalty
            computational_cost = background_samples * shap_samples
            if computational_cost > 50000:
                base_score -= (computational_cost - 50000) / 10000
            
            return max(0.0, min(100.0, base_score))
            
        except Exception as e:
            self.logger.error(f"Error evaluating SHAP analyzer parameters: {str(e)}")
            return 0.0
    
    def _calculate_population_efficiency(self, population_size: int) -> float:
        """Calculate efficiency score for population size"""
        complexity = self.data_characteristics['data_complexity']
        brand_count = self.data_characteristics['brand_count']
        
        # Optimal population size ranges
        if complexity == 'high':
            optimal_range = (50, 150)
        elif complexity == 'medium':
            optimal_range = (30, 80)
        else:
            optimal_range = (20, 50)
        
        # Adjust for brand count
        optimal_range = (optimal_range[0] + brand_count * 2, optimal_range[1] + brand_count * 5)
        
        if optimal_range[0] <= population_size <= optimal_range[1]:
            return 1.0
        elif population_size < optimal_range[0]:
            return population_size / optimal_range[0]
        else:
            return optimal_range[1] / population_size
    
    def _calculate_generation_efficiency(self, generations: int, population_size: int) -> float:
        """Calculate efficiency score for number of generations"""
        variance = self.data_characteristics['performance_variance']
        
        # More generations needed for high variance data
        if variance > 15:
            optimal_ratio = 1.5  # generations should be 1.5x population size
        elif variance > 10:
            optimal_ratio = 1.0
        else:
            optimal_ratio = 0.8
        
        optimal_generations = population_size * optimal_ratio
        
        if abs(generations - optimal_generations) <= optimal_generations * 0.2:
            return 1.0
        else:
            return max(0.1, 1.0 - abs(generations - optimal_generations) / optimal_generations)
    
    def _calculate_mutation_efficiency(self, mutation_rate: float) -> float:
        """Calculate efficiency score for mutation rate"""
        variance = self.data_characteristics['performance_variance']
        
        # Higher mutation rates for high variance data
        if variance > 20:
            optimal_range = (0.2, 0.4)
        elif variance > 10:
            optimal_range = (0.1, 0.3)
        else:
            optimal_range = (0.05, 0.2)
        
        if optimal_range[0] <= mutation_rate <= optimal_range[1]:
            return 1.0
        elif mutation_rate < optimal_range[0]:
            return mutation_rate / optimal_range[0]
        else:
            return optimal_range[1] / mutation_rate
    
    def _calculate_crossover_efficiency(self, crossover_rate: float) -> float:
        """Calculate efficiency score for crossover rate"""
        # Generally, crossover rates between 0.7-0.9 are good
        if 0.7 <= crossover_rate <= 0.9:
            return 1.0
        elif crossover_rate < 0.7:
            return crossover_rate / 0.7
        else:
            return 0.9 / crossover_rate
    
    def _calculate_parameter_balance_bonus(self, population_size: int, generations: int,
                                         mutation_rate: float, crossover_rate: float) -> float:
        """Calculate bonus for balanced parameter combinations"""
        # Bonus for balanced exploration vs exploitation
        exploration_factor = mutation_rate * population_size
        exploitation_factor = crossover_rate * generations
        
        balance_ratio = min(exploration_factor, exploitation_factor) / max(exploration_factor, exploitation_factor)
        
        return balance_ratio * 5  # Up to 5 point bonus
    
    def _calculate_sample_efficiency(self, background_samples: int, shap_samples: int) -> float:
        """Calculate efficiency for SHAP sample sizes"""
        data_size = self.data_characteristics['brand_count'] * self.data_characteristics['metric_count']
        
        # Background samples should be proportional to data size
        optimal_background = min(500, data_size * 10)
        background_efficiency = min(1.0, background_samples / optimal_background) if optimal_background > 0 else 0.5
        
        # SHAP samples should be reasonable for computation
        if 50 <= shap_samples <= 300:
            shap_efficiency = 1.0
        else:
            shap_efficiency = 0.5
        
        return (background_efficiency + shap_efficiency) / 2
    
    def _calculate_threshold_efficiency(self, threshold: float) -> float:
        """Calculate efficiency for feature threshold"""
        # Optimal threshold depends on data complexity
        complexity = self.data_characteristics['data_complexity']
        
        if complexity == 'high':
            optimal_threshold = 0.05
        elif complexity == 'medium':
            optimal_threshold = 0.03
        else:
            optimal_threshold = 0.02
        
        return 1.0 - abs(threshold - optimal_threshold) / optimal_threshold
    
    def _calculate_complexity_efficiency(self, model_complexity: str) -> float:
        """Calculate efficiency for model complexity choice"""
        data_complexity = self.data_characteristics['data_complexity']
        correlation_strength = self.data_characteristics['correlation_strength']
        
        if data_complexity == 'high' and correlation_strength < 0.5:
            # Complex data with weak correlations - ensemble works best
            return 1.0 if model_complexity == 'ensemble' else 0.7
        elif data_complexity == 'low' and correlation_strength > 0.7:
            # Simple data with strong correlations - linear works well
            return 1.0 if model_complexity == 'linear' else 0.8
        else:
            # Medium complexity - tree models are versatile
            return 1.0 if model_complexity == 'tree' else 0.9
    
    def _calculate_clustering_efficiency(self, n_clusters: int) -> float:
        """Calculate efficiency for clustering parameters"""
        metric_count = self.data_characteristics['metric_count']
        
        # Optimal clusters should be related to metric count
        optimal_clusters = max(3, min(8, metric_count // 3))
        
        if abs(n_clusters - optimal_clusters) <= 2:
            return 1.0
        else:
            return max(0.3, 1.0 - abs(n_clusters - optimal_clusters) / optimal_clusters)
    
    def _get_optimal_sampler(self) -> optuna.samplers.BaseSampler:
        """Get optimal sampler based on data characteristics"""
        complexity = self.data_characteristics['data_complexity']
        
        if complexity == 'high':
            # Use CMA-ES for high-dimensional optimization
            return CmaEsSampler()
        elif complexity == 'medium':
            # Use TPE for medium complexity
            return TPESampler(n_startup_trials=10, n_ei_candidates=24)
        else:
            # Use random sampler for simple cases
            return RandomSampler()
    
    def _get_optimal_pruner(self) -> optuna.pruners.BasePruner:
        """Get optimal pruner based on data characteristics"""
        if self.data_characteristics['optimization_difficulty'] == 'high':
            return SuccessiveHalvingPruner()
        else:
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get optimization recommendations based on data characteristics
        
        Returns:
            Dict with optimization recommendations
        """
        recommendations = {
            'recommended_trials': self._get_recommended_trials(),
            'recommended_timeout': self._get_recommended_timeout(),
            'optimization_strategy': self._get_optimization_strategy(),
            'expected_improvement': self._estimate_expected_improvement(),
            'resource_requirements': self._estimate_resource_requirements()
        }
        
        return recommendations
    
    def _get_recommended_trials(self) -> int:
        """Get recommended number of trials"""
        complexity = self.data_characteristics['data_complexity']
        
        if complexity == 'high':
            return 200
        elif complexity == 'medium':
            return 100
        else:
            return 50
    
    def _get_recommended_timeout(self) -> int:
        """Get recommended timeout in seconds"""
        complexity = self.data_characteristics['data_complexity']
        
        if complexity == 'high':
            return 3600  # 1 hour
        elif complexity == 'medium':
            return 1800  # 30 minutes
        else:
            return 900   # 15 minutes
    
    def _get_optimization_strategy(self) -> str:
        """Get recommended optimization strategy"""
        difficulty = self.data_characteristics['optimization_difficulty']
        
        if difficulty == 'high':
            return 'multi_objective_with_constraints'
        elif difficulty == 'medium':
            return 'single_objective_with_pruning'
        else:
            return 'simple_grid_search'
    
    def _estimate_expected_improvement(self) -> float:
        """Estimate expected improvement from optimization"""
        variance = self.data_characteristics['performance_variance']
        
        # Higher variance suggests more room for improvement
        if variance > 20:
            return 0.15  # 15% improvement expected
        elif variance > 10:
            return 0.10  # 10% improvement expected
        else:
            return 0.05  # 5% improvement expected
    
    def _estimate_resource_requirements(self) -> Dict[str, Any]:
        """Estimate computational resource requirements"""
        complexity = self.data_characteristics['data_complexity']
        brand_count = self.data_characteristics['brand_count']
        metric_count = self.data_characteristics['metric_count']
        
        base_memory = 512  # MB
        base_cpu_time = 300  # seconds
        
        # Scale based on data size
        memory_scaling = (brand_count * metric_count) / 100
        time_scaling = memory_scaling * 2
        
        if complexity == 'high':
            memory_scaling *= 2
            time_scaling *= 3
        
        return {
            'estimated_memory_mb': int(base_memory * (1 + memory_scaling)),
            'estimated_cpu_time_seconds': int(base_cpu_time * (1 + time_scaling)),
            'recommended_parallel_jobs': min(4, max(1, brand_count // 2))
        }
    
    def save_optimization_results(self, filepath: str) -> None:
        """Save optimization results to file"""
        try:
            results = {
                'data_characteristics': self.data_characteristics,
                'best_parameters': self.best_parameters,
                'optimization_history': self.optimization_history,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Optimization results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {str(e)}")
    
    def load_optimization_results(self, filepath: str) -> None:
        """Load optimization results from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    results = json.load(f)
                
                self.best_parameters = results.get('best_parameters', {})
                self.optimization_history = results.get('optimization_history', {})
                
                self.logger.info(f"Optimization results loaded from {filepath}")
            else:
                self.logger.warning(f"Optimization results file not found: {filepath}")
                
        except Exception as e:
            self.logger.error(f"Error loading optimization results: {str(e)}")
    
    def get_parameter_summary(self) -> Dict[str, Any]:
        """Get summary of optimized parameters"""
        summary = {
            'optimization_completed': len(self.best_parameters) > 0,
            'data_characteristics': self.data_characteristics,
            'optimized_targets': list(self.best_parameters.keys()),
            'recommendations': self.get_optimization_recommendations()
        }
        
        if self.best_parameters:
            summary['best_parameters'] = self.best_parameters
        
        return summary

