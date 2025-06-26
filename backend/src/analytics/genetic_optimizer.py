"""
Genetic Portfolio Optimizer for Digi-Cadence Platform
Advanced genetic algorithm implementation for multi-brand, multi-project optimization
"""

import numpy as np
import pandas as pd
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from src.models.portfolio import Project, Brand, Metric, BrandMetric
from src.analytics.base_analyzer import BaseAnalyzer

@dataclass
class OptimizationChromosome:
    """Represents a solution chromosome in the genetic algorithm"""
    brand_id: str
    metric_improvements: Dict[str, float]  # metric_id -> target_value
    fitness_score: float = 0.0
    portfolio_impact: float = 0.0
    synergy_score: float = 0.0
    feasibility_score: float = 1.0

@dataclass
class PortfolioOptimizationResult:
    """Results from portfolio optimization"""
    best_chromosome: OptimizationChromosome
    population_evolution: List[Dict[str, Any]]
    convergence_metrics: Dict[str, Any]
    brand_recommendations: Dict[str, List[Dict[str, Any]]]
    portfolio_metrics: Dict[str, Any]
    execution_summary: Dict[str, Any]

class GeneticPortfolioOptimizer(BaseAnalyzer):
    """
    Advanced genetic algorithm optimizer for portfolio management
    Supports multi-brand, multi-project optimization with synergy identification
    """
    
    def __init__(self, projects: List[Project], brands: List[Brand], config: Dict[str, Any] = None):
        super().__init__(projects, brands)
        
        # Default configuration
        self.config = {
            'population_size': 50,
            'num_generations': 100,
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'elite_size': 10,
            'max_recommendations': 10,
            'convergence_threshold': 0.001,
            'max_stagnant_generations': 20,
            'tournament_size': 3,
            'diversity_weight': 0.1,
            'synergy_weight': 0.2,
            'feasibility_weight': 0.3
        }
        
        if config:
            self.config.update(config)
        
        self.logger = logging.getLogger(__name__)
        self.metrics_data = self._load_metrics_data()
        self.brand_baselines = self._calculate_brand_baselines()
        self.competitive_benchmarks = self._calculate_competitive_benchmarks()
        
    def _load_metrics_data(self) -> pd.DataFrame:
        """Load and prepare metrics data for optimization"""
        try:
            # Get all metrics for the projects
            all_metrics = []
            for project in self.projects:
                for metric in project.metrics:
                    for brand_metric in metric.brand_metrics:
                        if brand_metric.brand_id in [b.id for b in self.brands]:
                            all_metrics.append({
                                'project_id': str(project.id),
                                'project_name': project.name,
                                'brand_id': str(brand_metric.brand_id),
                                'brand_name': next(b.name for b in self.brands if b.id == brand_metric.brand_id),
                                'metric_id': str(metric.id),
                                'metric_name': metric.name,
                                'section_name': metric.section_name,
                                'platform_name': metric.platform_name,
                                'metric_type': metric.metric_type,
                                'weight': metric.weight,
                                'raw_value': brand_metric.raw_value,
                                'normalized_value': brand_metric.normalized_value,
                                'period_start': brand_metric.period_start,
                                'confidence_score': brand_metric.confidence_score or 1.0
                            })
            
            return pd.DataFrame(all_metrics)
        except Exception as e:
            self.logger.error(f"Error loading metrics data: {e}")
            return pd.DataFrame()
    
    def _calculate_brand_baselines(self) -> Dict[str, Dict[str, float]]:
        """Calculate baseline performance for each brand"""
        baselines = {}
        
        for brand in self.brands:
            brand_id = str(brand.id)
            brand_metrics = self.metrics_data[self.metrics_data['brand_id'] == brand_id]
            
            baselines[brand_id] = {}
            for _, metric_row in brand_metrics.iterrows():
                metric_id = metric_row['metric_id']
                baselines[brand_id][metric_id] = {
                    'current_value': metric_row['normalized_value'] or metric_row['raw_value'] or 0,
                    'metric_type': metric_row['metric_type'],
                    'weight': metric_row['weight'],
                    'confidence': metric_row['confidence_score']
                }
        
        return baselines
    
    def _calculate_competitive_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Calculate competitive benchmarks for each metric"""
        benchmarks = {}
        
        # Group by metric and calculate competitive statistics
        for metric_id in self.metrics_data['metric_id'].unique():
            metric_data = self.metrics_data[self.metrics_data['metric_id'] == metric_id]
            
            values = metric_data['normalized_value'].fillna(metric_data['raw_value']).dropna()
            if len(values) > 0:
                benchmarks[metric_id] = {
                    'best_value': values.max(),
                    'worst_value': values.min(),
                    'average_value': values.mean(),
                    'median_value': values.median(),
                    'std_value': values.std(),
                    'percentile_75': values.quantile(0.75),
                    'percentile_90': values.quantile(0.90)
                }
        
        return benchmarks
    
    def optimize_portfolio(self, 
                          num_generations: int = None,
                          population_size: int = None,
                          mutation_rate: float = None) -> PortfolioOptimizationResult:
        """
        Run genetic algorithm optimization for the entire portfolio
        """
        # Update configuration if parameters provided
        if num_generations:
            self.config['num_generations'] = num_generations
        if population_size:
            self.config['population_size'] = population_size
        if mutation_rate:
            self.config['mutation_rate'] = mutation_rate
        
        self.logger.info(f"Starting portfolio optimization with {len(self.brands)} brands across {len(self.projects)} projects")
        
        # Initialize population
        population = self._initialize_population()
        
        # Track evolution
        evolution_history = []
        best_fitness_history = []
        stagnant_generations = 0
        
        for generation in range(self.config['num_generations']):
            # Evaluate fitness for all chromosomes
            population = self._evaluate_population_fitness(population)
            
            # Track best fitness
            best_fitness = max(chromosome.fitness_score for chromosome in population)
            best_fitness_history.append(best_fitness)
            
            # Check for convergence
            if len(best_fitness_history) > 1:
                improvement = best_fitness - best_fitness_history[-2]
                if improvement < self.config['convergence_threshold']:
                    stagnant_generations += 1
                else:
                    stagnant_generations = 0
                
                if stagnant_generations >= self.config['max_stagnant_generations']:
                    self.logger.info(f"Converged at generation {generation}")
                    break
            
            # Record generation statistics
            generation_stats = {
                'generation': generation,
                'best_fitness': best_fitness,
                'average_fitness': np.mean([c.fitness_score for c in population]),
                'diversity_score': self._calculate_population_diversity(population),
                'population_size': len(population)
            }
            evolution_history.append(generation_stats)
            
            # Selection
            selected_population = self._selection(population)
            
            # Crossover
            offspring = self._crossover(selected_population)
            
            # Mutation
            mutated_population = self._mutation(offspring)
            
            # Combine elite with new population
            elite = sorted(population, key=lambda x: x.fitness_score, reverse=True)[:self.config['elite_size']]
            population = elite + mutated_population[:self.config['population_size'] - self.config['elite_size']]
        
        # Get best solution
        best_chromosome = max(population, key=lambda x: x.fitness_score)
        
        # Generate comprehensive results
        brand_recommendations = self._generate_brand_recommendations(population)
        portfolio_metrics = self._calculate_portfolio_metrics(best_chromosome, population)
        
        convergence_metrics = {
            'generations_run': len(evolution_history),
            'final_fitness': best_fitness_history[-1] if best_fitness_history else 0,
            'fitness_improvement': best_fitness_history[-1] - best_fitness_history[0] if len(best_fitness_history) > 1 else 0,
            'convergence_rate': self._calculate_convergence_rate(best_fitness_history),
            'stagnant_generations': stagnant_generations
        }
        
        execution_summary = {
            'total_brands_optimized': len(self.brands),
            'total_projects_analyzed': len(self.projects),
            'total_metrics_considered': len(self.metrics_data['metric_id'].unique()),
            'optimization_config': self.config,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return PortfolioOptimizationResult(
            best_chromosome=best_chromosome,
            population_evolution=evolution_history,
            convergence_metrics=convergence_metrics,
            brand_recommendations=brand_recommendations,
            portfolio_metrics=portfolio_metrics,
            execution_summary=execution_summary
        )
    
    def _initialize_population(self) -> List[OptimizationChromosome]:
        """Initialize the genetic algorithm population"""
        population = []
        
        for _ in range(self.config['population_size']):
            # Randomly select a brand for this chromosome
            brand = random.choice(self.brands)
            brand_id = str(brand.id)
            
            # Get available metrics for this brand
            brand_metrics = self.brand_baselines.get(brand_id, {})
            
            # Randomly select metrics to optimize
            num_metrics = min(
                random.randint(1, self.config['max_recommendations']),
                len(brand_metrics)
            )
            
            selected_metrics = random.sample(list(brand_metrics.keys()), num_metrics)
            
            # Generate improvement targets for selected metrics
            metric_improvements = {}
            for metric_id in selected_metrics:
                baseline = brand_metrics[metric_id]
                benchmark = self.competitive_benchmarks.get(metric_id, {})
                
                if baseline['metric_type'] == 'maximize':
                    # For maximize metrics, target between current and best benchmark
                    current_val = baseline['current_value']
                    best_val = benchmark.get('best_value', current_val * 1.2)
                    target = random.uniform(current_val, min(best_val, current_val * 1.5))
                else:
                    # For minimize metrics, target between best benchmark and current
                    current_val = baseline['current_value']
                    best_val = benchmark.get('best_value', current_val * 0.8)
                    target = random.uniform(max(best_val, current_val * 0.5), current_val)
                
                metric_improvements[metric_id] = target
            
            chromosome = OptimizationChromosome(
                brand_id=brand_id,
                metric_improvements=metric_improvements
            )
            population.append(chromosome)
        
        return population
    
    def _evaluate_population_fitness(self, population: List[OptimizationChromosome]) -> List[OptimizationChromosome]:
        """Evaluate fitness for all chromosomes in the population"""
        
        # Use parallel processing for fitness evaluation
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_chromosome = {
                executor.submit(self._calculate_chromosome_fitness, chromosome): chromosome
                for chromosome in population
            }
            
            for future in as_completed(future_to_chromosome):
                chromosome = future_to_chromosome[future]
                try:
                    fitness_components = future.result()
                    chromosome.fitness_score = fitness_components['total_fitness']
                    chromosome.portfolio_impact = fitness_components['portfolio_impact']
                    chromosome.synergy_score = fitness_components['synergy_score']
                    chromosome.feasibility_score = fitness_components['feasibility_score']
                except Exception as e:
                    self.logger.error(f"Error calculating fitness for chromosome: {e}")
                    chromosome.fitness_score = 0.0
        
        return population
    
    def _calculate_chromosome_fitness(self, chromosome: OptimizationChromosome) -> Dict[str, float]:
        """Calculate comprehensive fitness score for a chromosome"""
        
        # 1. Portfolio Impact Score
        portfolio_impact = self._calculate_portfolio_impact(chromosome)
        
        # 2. Synergy Score (cross-brand and cross-project synergies)
        synergy_score = self._calculate_synergy_score(chromosome)
        
        # 3. Feasibility Score (how realistic the improvements are)
        feasibility_score = self._calculate_feasibility_score(chromosome)
        
        # 4. Diversity Score (to maintain population diversity)
        diversity_score = self._calculate_diversity_score(chromosome)
        
        # Weighted combination
        total_fitness = (
            portfolio_impact * (1 - self.config['synergy_weight'] - self.config['feasibility_weight'] - self.config['diversity_weight']) +
            synergy_score * self.config['synergy_weight'] +
            feasibility_score * self.config['feasibility_weight'] +
            diversity_score * self.config['diversity_weight']
        )
        
        return {
            'total_fitness': total_fitness,
            'portfolio_impact': portfolio_impact,
            'synergy_score': synergy_score,
            'feasibility_score': feasibility_score,
            'diversity_score': diversity_score
        }
    
    def _calculate_portfolio_impact(self, chromosome: OptimizationChromosome) -> float:
        """Calculate the overall portfolio impact of the chromosome"""
        total_impact = 0.0
        total_weight = 0.0
        
        brand_baselines = self.brand_baselines.get(chromosome.brand_id, {})
        
        for metric_id, target_value in chromosome.metric_improvements.items():
            baseline = brand_baselines.get(metric_id, {})
            if not baseline:
                continue
            
            current_value = baseline['current_value']
            weight = baseline['weight']
            metric_type = baseline['metric_type']
            
            # Calculate improvement
            if metric_type == 'maximize':
                improvement = (target_value - current_value) / max(current_value, 1e-6)
            else:
                improvement = (current_value - target_value) / max(current_value, 1e-6)
            
            # Weight by metric importance
            weighted_improvement = improvement * weight
            total_impact += weighted_improvement
            total_weight += weight
        
        return total_impact / max(total_weight, 1e-6)
    
    def _calculate_synergy_score(self, chromosome: OptimizationChromosome) -> float:
        """Calculate synergy score based on cross-brand and cross-project effects"""
        synergy_score = 0.0
        
        # Cross-brand synergies
        for other_brand in self.brands:
            if str(other_brand.id) != chromosome.brand_id:
                synergy_score += self._calculate_cross_brand_synergy(chromosome, str(other_brand.id))
        
        # Cross-project synergies
        for project in self.projects:
            synergy_score += self._calculate_cross_project_synergy(chromosome, str(project.id))
        
        return synergy_score / max(len(self.brands) + len(self.projects), 1)
    
    def _calculate_cross_brand_synergy(self, chromosome: OptimizationChromosome, other_brand_id: str) -> float:
        """Calculate synergy between brands"""
        # This is a simplified synergy calculation
        # In practice, this would involve complex correlation analysis
        
        synergy = 0.0
        brand_metrics = set(chromosome.metric_improvements.keys())
        other_brand_metrics = set(self.brand_baselines.get(other_brand_id, {}).keys())
        
        # Synergy based on shared metrics
        shared_metrics = brand_metrics.intersection(other_brand_metrics)
        synergy += len(shared_metrics) / max(len(brand_metrics), 1) * 0.5
        
        return synergy
    
    def _calculate_cross_project_synergy(self, chromosome: OptimizationChromosome, project_id: str) -> float:
        """Calculate synergy across projects"""
        # Simplified cross-project synergy calculation
        synergy = 0.0
        
        # Check if brand is involved in the project
        project_metrics = self.metrics_data[
            (self.metrics_data['project_id'] == project_id) &
            (self.metrics_data['brand_id'] == chromosome.brand_id)
        ]
        
        if not project_metrics.empty:
            # Synergy based on project involvement
            synergy += 0.3
            
            # Additional synergy based on metric coverage
            project_metric_ids = set(project_metrics['metric_id'].unique())
            chromosome_metric_ids = set(chromosome.metric_improvements.keys())
            
            coverage = len(chromosome_metric_ids.intersection(project_metric_ids)) / max(len(project_metric_ids), 1)
            synergy += coverage * 0.2
        
        return synergy
    
    def _calculate_feasibility_score(self, chromosome: OptimizationChromosome) -> float:
        """Calculate how feasible the proposed improvements are"""
        feasibility_scores = []
        
        brand_baselines = self.brand_baselines.get(chromosome.brand_id, {})
        
        for metric_id, target_value in chromosome.metric_improvements.items():
            baseline = brand_baselines.get(metric_id, {})
            benchmark = self.competitive_benchmarks.get(metric_id, {})
            
            if not baseline or not benchmark:
                feasibility_scores.append(0.5)  # Neutral score for missing data
                continue
            
            current_value = baseline['current_value']
            confidence = baseline['confidence']
            
            # Check if target is within reasonable bounds
            if baseline['metric_type'] == 'maximize':
                max_feasible = benchmark.get('percentile_90', current_value * 1.5)
                feasibility = 1.0 - max(0, (target_value - max_feasible) / max_feasible)
            else:
                min_feasible = benchmark.get('percentile_90', current_value * 0.5)
                feasibility = 1.0 - max(0, (min_feasible - target_value) / max(min_feasible, 1e-6))
            
            # Adjust by confidence score
            feasibility *= confidence
            feasibility_scores.append(max(0, min(1, feasibility)))
        
        return np.mean(feasibility_scores) if feasibility_scores else 0.5
    
    def _calculate_diversity_score(self, chromosome: OptimizationChromosome) -> float:
        """Calculate diversity score to maintain population diversity"""
        # This is a placeholder for diversity calculation
        # In practice, this would compare against other chromosomes in the population
        return random.uniform(0.3, 0.7)
    
    def _calculate_population_diversity(self, population: List[OptimizationChromosome]) -> float:
        """Calculate overall population diversity"""
        if len(population) < 2:
            return 1.0
        
        # Calculate diversity based on different metrics being optimized
        all_metrics = set()
        for chromosome in population:
            all_metrics.update(chromosome.metric_improvements.keys())
        
        diversity_scores = []
        for chromosome in population:
            chromosome_metrics = set(chromosome.metric_improvements.keys())
            diversity = len(chromosome_metrics) / max(len(all_metrics), 1)
            diversity_scores.append(diversity)
        
        return np.std(diversity_scores)
    
    def _selection(self, population: List[OptimizationChromosome]) -> List[OptimizationChromosome]:
        """Tournament selection for genetic algorithm"""
        selected = []
        
        for _ in range(len(population)):
            # Tournament selection
            tournament = random.sample(population, min(self.config['tournament_size'], len(population)))
            winner = max(tournament, key=lambda x: x.fitness_score)
            selected.append(winner)
        
        return selected
    
    def _crossover(self, population: List[OptimizationChromosome]) -> List[OptimizationChromosome]:
        """Crossover operation for genetic algorithm"""
        offspring = []
        
        for i in range(0, len(population) - 1, 2):
            parent1 = population[i]
            parent2 = population[i + 1]
            
            if random.random() < self.config['crossover_rate']:
                child1, child2 = self._single_point_crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
        
        return offspring
    
    def _single_point_crossover(self, parent1: OptimizationChromosome, parent2: OptimizationChromosome) -> Tuple[OptimizationChromosome, OptimizationChromosome]:
        """Single point crossover between two chromosomes"""
        
        # Combine metrics from both parents
        all_metrics = list(set(parent1.metric_improvements.keys()) | set(parent2.metric_improvements.keys()))
        
        if len(all_metrics) <= 1:
            return parent1, parent2
        
        crossover_point = random.randint(1, len(all_metrics) - 1)
        
        # Create children
        child1_metrics = {}
        child2_metrics = {}
        
        for i, metric_id in enumerate(all_metrics):
            if i < crossover_point:
                if metric_id in parent1.metric_improvements:
                    child1_metrics[metric_id] = parent1.metric_improvements[metric_id]
                if metric_id in parent2.metric_improvements:
                    child2_metrics[metric_id] = parent2.metric_improvements[metric_id]
            else:
                if metric_id in parent2.metric_improvements:
                    child1_metrics[metric_id] = parent2.metric_improvements[metric_id]
                if metric_id in parent1.metric_improvements:
                    child2_metrics[metric_id] = parent1.metric_improvements[metric_id]
        
        child1 = OptimizationChromosome(
            brand_id=parent1.brand_id,
            metric_improvements=child1_metrics
        )
        
        child2 = OptimizationChromosome(
            brand_id=parent2.brand_id,
            metric_improvements=child2_metrics
        )
        
        return child1, child2
    
    def _mutation(self, population: List[OptimizationChromosome]) -> List[OptimizationChromosome]:
        """Mutation operation for genetic algorithm"""
        mutated_population = []
        
        for chromosome in population:
            if random.random() < self.config['mutation_rate']:
                mutated_chromosome = self._mutate_chromosome(chromosome)
                mutated_population.append(mutated_chromosome)
            else:
                mutated_population.append(chromosome)
        
        return mutated_population
    
    def _mutate_chromosome(self, chromosome: OptimizationChromosome) -> OptimizationChromosome:
        """Mutate a single chromosome"""
        mutated_improvements = chromosome.metric_improvements.copy()
        
        # Randomly select metrics to mutate
        metrics_to_mutate = random.sample(
            list(mutated_improvements.keys()),
            max(1, int(len(mutated_improvements) * 0.3))
        )
        
        brand_baselines = self.brand_baselines.get(chromosome.brand_id, {})
        
        for metric_id in metrics_to_mutate:
            baseline = brand_baselines.get(metric_id, {})
            benchmark = self.competitive_benchmarks.get(metric_id, {})
            
            if baseline and benchmark:
                current_value = baseline['current_value']
                
                # Add random variation
                if baseline['metric_type'] == 'maximize':
                    max_val = benchmark.get('best_value', current_value * 1.2)
                    variation = random.uniform(-0.1, 0.2) * (max_val - current_value)
                else:
                    min_val = benchmark.get('best_value', current_value * 0.8)
                    variation = random.uniform(-0.2, 0.1) * (current_value - min_val)
                
                mutated_improvements[metric_id] = max(0, mutated_improvements[metric_id] + variation)
        
        return OptimizationChromosome(
            brand_id=chromosome.brand_id,
            metric_improvements=mutated_improvements
        )
    
    def _generate_brand_recommendations(self, population: List[OptimizationChromosome]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate brand-specific recommendations from the population"""
        brand_recommendations = {}
        
        # Group chromosomes by brand
        brand_chromosomes = {}
        for chromosome in population:
            if chromosome.brand_id not in brand_chromosomes:
                brand_chromosomes[chromosome.brand_id] = []
            brand_chromosomes[chromosome.brand_id].append(chromosome)
        
        # Generate recommendations for each brand
        for brand_id, chromosomes in brand_chromosomes.items():
            # Sort by fitness score
            chromosomes.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # Get top recommendations
            top_chromosomes = chromosomes[:min(5, len(chromosomes))]
            
            recommendations = []
            for i, chromosome in enumerate(top_chromosomes):
                brand_baselines = self.brand_baselines.get(brand_id, {})
                
                metric_recommendations = []
                for metric_id, target_value in chromosome.metric_improvements.items():
                    baseline = brand_baselines.get(metric_id, {})
                    
                    if baseline:
                        current_value = baseline['current_value']
                        improvement = target_value - current_value
                        improvement_pct = (improvement / max(current_value, 1e-6)) * 100
                        
                        metric_recommendations.append({
                            'metric_id': metric_id,
                            'current_value': current_value,
                            'target_value': target_value,
                            'improvement': improvement,
                            'improvement_percentage': improvement_pct,
                            'metric_type': baseline['metric_type'],
                            'weight': baseline['weight']
                        })
                
                recommendations.append({
                    'rank': i + 1,
                    'fitness_score': chromosome.fitness_score,
                    'portfolio_impact': chromosome.portfolio_impact,
                    'synergy_score': chromosome.synergy_score,
                    'feasibility_score': chromosome.feasibility_score,
                    'metric_recommendations': metric_recommendations
                })
            
            brand_recommendations[brand_id] = recommendations
        
        return brand_recommendations
    
    def _calculate_portfolio_metrics(self, best_chromosome: OptimizationChromosome, population: List[OptimizationChromosome]) -> Dict[str, Any]:
        """Calculate overall portfolio metrics"""
        
        # Population statistics
        fitness_scores = [c.fitness_score for c in population]
        
        portfolio_metrics = {
            'best_fitness': best_chromosome.fitness_score,
            'average_fitness': np.mean(fitness_scores),
            'fitness_std': np.std(fitness_scores),
            'population_diversity': self._calculate_population_diversity(population),
            'total_brands_optimized': len(set(c.brand_id for c in population)),
            'total_metrics_optimized': len(set().union(*[c.metric_improvements.keys() for c in population])),
            'average_improvements_per_brand': np.mean([len(c.metric_improvements) for c in population]),
            'portfolio_synergy_score': np.mean([c.synergy_score for c in population]),
            'average_feasibility': np.mean([c.feasibility_score for c in population])
        }
        
        return portfolio_metrics
    
    def _calculate_convergence_rate(self, fitness_history: List[float]) -> float:
        """Calculate the convergence rate of the algorithm"""
        if len(fitness_history) < 2:
            return 0.0
        
        # Calculate average improvement per generation
        improvements = [fitness_history[i] - fitness_history[i-1] for i in range(1, len(fitness_history))]
        return np.mean(improvements)
    
    def run_scenario_analysis(self, scenario_name: str, metric_changes: Dict[str, float]) -> Dict[str, Any]:
        """Run scenario analysis with specific metric changes"""
        
        # Create a modified version of the baseline data
        modified_baselines = self.brand_baselines.copy()
        
        # Apply scenario changes
        for brand_id in modified_baselines:
            for metric_id, change_factor in metric_changes.items():
                if metric_id in modified_baselines[brand_id]:
                    current_value = modified_baselines[brand_id][metric_id]['current_value']
                    modified_baselines[brand_id][metric_id]['current_value'] = current_value * change_factor
        
        # Temporarily replace baselines
        original_baselines = self.brand_baselines
        self.brand_baselines = modified_baselines
        
        try:
            # Run optimization with modified baselines
            result = self.optimize_portfolio(num_generations=50)  # Reduced generations for scenario
            
            scenario_result = {
                'scenario_name': scenario_name,
                'metric_changes': metric_changes,
                'optimization_result': {
                    'best_fitness': result.best_chromosome.fitness_score,
                    'portfolio_impact': result.best_chromosome.portfolio_impact,
                    'synergy_score': result.best_chromosome.synergy_score,
                    'feasibility_score': result.best_chromosome.feasibility_score
                },
                'convergence_metrics': result.convergence_metrics,
                'execution_summary': result.execution_summary
            }
            
            return scenario_result
            
        finally:
            # Restore original baselines
            self.brand_baselines = original_baselines
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update optimizer configuration"""
        self.config.update(new_config)
        self.logger.info(f"Updated optimizer configuration: {new_config}")

