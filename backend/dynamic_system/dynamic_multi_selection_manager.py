"""
Dynamic Multi-Selection Manager
Enables dynamic multi-brand and multi-project selection with adaptive analysis capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import warnings
from datetime import datetime
import logging
from itertools import combinations
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json

warnings.filterwarnings('ignore')

class DynamicMultiSelectionManager:
    """
    Dynamic manager for handling multiple project and brand selections with adaptive analysis
    """
    
    def __init__(self, available_projects: Dict[str, Any], available_brands: Dict[str, Any]):
        """
        Initialize multi-selection manager
        
        Args:
            available_projects: Dict of available projects with their metadata
            available_brands: Dict of available brands with their data
        """
        self.available_projects = available_projects
        self.available_brands = available_brands
        
        # Selection state
        self.selected_projects = []
        self.selected_brands = []
        
        # Analysis results
        self.selection_analysis = {}
        self.compatibility_matrix = {}
        self.synergy_analysis = {}
        self.cross_analysis_opportunities = {}
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Dynamic Multi-Selection Manager initialized")
    
    def enable_dynamic_multi_selection(self, selected_projects: List[int], 
                                     selected_brands: List[str]) -> Dict[str, Any]:
        """
        Enable multi-selection with dynamic analysis
        
        Args:
            selected_projects: List of selected project IDs
            selected_brands: List of selected brand IDs
            
        Returns:
            Dict with selection analysis and recommendations
        """
        try:
            self.logger.info(f"Enabling multi-selection: {len(selected_projects)} projects, {len(selected_brands)} brands")
            
            # Store selections
            self.selected_projects = selected_projects
            self.selected_brands = selected_brands
            
            # Analyze selection characteristics
            selection_analysis = {
                'project_compatibility': self._analyze_project_compatibility(selected_projects),
                'brand_synergies': self._analyze_brand_synergies(selected_brands),
                'data_availability': self._check_cross_selection_data_availability(selected_projects, selected_brands),
                'analysis_opportunities': self._identify_cross_analysis_opportunities(selected_projects, selected_brands),
                'complexity_assessment': self._assess_selection_complexity(selected_projects, selected_brands),
                'resource_requirements': self._estimate_resource_requirements(selected_projects, selected_brands)
            }
            
            # Adapt analysis approach based on selection
            analysis_approach = self._adapt_analysis_approach(selection_analysis)
            
            # Store results
            self.selection_analysis = selection_analysis
            
            result = {
                'selection_analysis': selection_analysis,
                'recommended_approach': analysis_approach,
                'available_reports': self._identify_available_reports(selection_analysis),
                'optimization_strategy': self._recommend_optimization_strategy(selection_analysis),
                'execution_plan': self._create_execution_plan(selection_analysis, analysis_approach)
            }
            
            self.logger.info("Multi-selection analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in multi-selection enablement: {str(e)}")
            raise
    
    def _analyze_project_compatibility(self, selected_projects: List[int]) -> Dict[str, Any]:
        """
        Analyze compatibility between selected projects
        
        Args:
            selected_projects: List of selected project IDs
            
        Returns:
            Dict with project compatibility analysis
        """
        compatibility = {
            'compatibility_score': 0.0,
            'compatible_pairs': [],
            'incompatible_pairs': [],
            'common_metrics': [],
            'common_brands': [],
            'temporal_alignment': {},
            'data_structure_similarity': 0.0
        }
        
        try:
            if len(selected_projects) < 2:
                compatibility['compatibility_score'] = 1.0
                return compatibility
            
            # Analyze pairwise compatibility
            compatible_pairs = []
            incompatible_pairs = []
            
            for i, proj1 in enumerate(selected_projects):
                for j, proj2 in enumerate(selected_projects[i+1:], i+1):
                    pair_compatibility = self._assess_project_pair_compatibility(proj1, proj2)
                    
                    if pair_compatibility['score'] >= 0.7:
                        compatible_pairs.append({
                            'projects': (proj1, proj2),
                            'score': pair_compatibility['score'],
                            'reasons': pair_compatibility['reasons']
                        })
                    else:
                        incompatible_pairs.append({
                            'projects': (proj1, proj2),
                            'score': pair_compatibility['score'],
                            'issues': pair_compatibility['issues']
                        })
            
            # Calculate overall compatibility score
            if len(selected_projects) > 1:
                total_pairs = len(selected_projects) * (len(selected_projects) - 1) // 2
                compatible_count = len(compatible_pairs)
                compatibility['compatibility_score'] = compatible_count / total_pairs if total_pairs > 0 else 0.0
            
            compatibility['compatible_pairs'] = compatible_pairs
            compatibility['incompatible_pairs'] = incompatible_pairs
            
            # Find common elements across projects
            compatibility['common_metrics'] = self._find_common_metrics(selected_projects)
            compatibility['common_brands'] = self._find_common_brands(selected_projects)
            
            # Assess data structure similarity
            compatibility['data_structure_similarity'] = self._assess_data_structure_similarity(selected_projects)
            
        except Exception as e:
            self.logger.error(f"Error analyzing project compatibility: {str(e)}")
        
        return compatibility
    
    def _assess_project_pair_compatibility(self, proj1: int, proj2: int) -> Dict[str, Any]:
        """Assess compatibility between two projects"""
        compatibility = {
            'score': 0.0,
            'reasons': [],
            'issues': []
        }
        
        try:
            # Get project data
            proj1_data = self.available_projects.get(proj1, {})
            proj2_data = self.available_projects.get(proj2, {})
            
            if not proj1_data or not proj2_data:
                compatibility['score'] = 0.0
                compatibility['issues'].append("Missing project data")
                return compatibility
            
            score_components = []
            
            # Check metric overlap
            proj1_metrics = set(proj1_data.get('metrics_available', []))
            proj2_metrics = set(proj2_data.get('metrics_available', []))
            
            if proj1_metrics and proj2_metrics:
                metric_overlap = len(proj1_metrics.intersection(proj2_metrics)) / len(proj1_metrics.union(proj2_metrics))
                score_components.append(metric_overlap * 0.3)
                
                if metric_overlap > 0.5:
                    compatibility['reasons'].append(f"Good metric overlap ({metric_overlap:.2f})")
                else:
                    compatibility['issues'].append(f"Low metric overlap ({metric_overlap:.2f})")
            
            # Check brand overlap
            proj1_brands = set(proj1_data.get('brands_available', []))
            proj2_brands = set(proj2_data.get('brands_available', []))
            
            if proj1_brands and proj2_brands:
                brand_overlap = len(proj1_brands.intersection(proj2_brands)) / len(proj1_brands.union(proj2_brands))
                score_components.append(brand_overlap * 0.3)
                
                if brand_overlap > 0.3:
                    compatibility['reasons'].append(f"Good brand overlap ({brand_overlap:.2f})")
                else:
                    compatibility['issues'].append(f"Low brand overlap ({brand_overlap:.2f})")
            
            # Check data quality compatibility
            proj1_quality = proj1_data.get('data_quality', {}).get('overall_completeness', 0)
            proj2_quality = proj2_data.get('data_quality', {}).get('overall_completeness', 0)
            
            quality_similarity = 1.0 - abs(proj1_quality - proj2_quality) / 100.0
            score_components.append(quality_similarity * 0.2)
            
            if quality_similarity > 0.8:
                compatibility['reasons'].append("Similar data quality levels")
            else:
                compatibility['issues'].append("Different data quality levels")
            
            # Check section compatibility
            proj1_sections = set(proj1_data.get('sections_available', []))
            proj2_sections = set(proj2_data.get('sections_available', []))
            
            if proj1_sections and proj2_sections:
                section_overlap = len(proj1_sections.intersection(proj2_sections)) / len(proj1_sections.union(proj2_sections))
                score_components.append(section_overlap * 0.2)
                
                if section_overlap > 0.7:
                    compatibility['reasons'].append("Compatible section structure")
                else:
                    compatibility['issues'].append("Different section structures")
            
            # Calculate final score
            compatibility['score'] = sum(score_components) if score_components else 0.0
            
        except Exception as e:
            self.logger.error(f"Error assessing project pair compatibility: {str(e)}")
        
        return compatibility
    
    def _analyze_brand_synergies(self, selected_brands: List[str]) -> Dict[str, Any]:
        """
        Analyze synergies between selected brands
        
        Args:
            selected_brands: List of selected brand IDs
            
        Returns:
            Dict with brand synergy analysis
        """
        synergies = {
            'synergy_score': 0.0,
            'synergy_pairs': [],
            'competitive_pairs': [],
            'collaboration_opportunities': [],
            'market_coverage': {},
            'performance_complementarity': {}
        }
        
        try:
            if len(selected_brands) < 2:
                synergies['synergy_score'] = 1.0
                return synergies
            
            # Analyze pairwise synergies
            synergy_pairs = []
            competitive_pairs = []
            
            for i, brand1 in enumerate(selected_brands):
                for j, brand2 in enumerate(selected_brands[i+1:], i+1):
                    pair_analysis = self._assess_brand_pair_synergy(brand1, brand2)
                    
                    if pair_analysis['synergy_type'] == 'synergistic':
                        synergy_pairs.append({
                            'brands': (brand1, brand2),
                            'synergy_score': pair_analysis['score'],
                            'synergy_areas': pair_analysis['synergy_areas']
                        })
                    elif pair_analysis['synergy_type'] == 'competitive':
                        competitive_pairs.append({
                            'brands': (brand1, brand2),
                            'competition_intensity': pair_analysis['score'],
                            'competition_areas': pair_analysis['competition_areas']
                        })
            
            synergies['synergy_pairs'] = synergy_pairs
            synergies['competitive_pairs'] = competitive_pairs
            
            # Calculate overall synergy score
            if len(selected_brands) > 1:
                total_pairs = len(selected_brands) * (len(selected_brands) - 1) // 2
                synergistic_count = len(synergy_pairs)
                synergies['synergy_score'] = synergistic_count / total_pairs if total_pairs > 0 else 0.0
            
            # Identify collaboration opportunities
            synergies['collaboration_opportunities'] = self._identify_collaboration_opportunities(selected_brands)
            
            # Analyze market coverage
            synergies['market_coverage'] = self._analyze_market_coverage(selected_brands)
            
            # Assess performance complementarity
            synergies['performance_complementarity'] = self._assess_performance_complementarity(selected_brands)
            
        except Exception as e:
            self.logger.error(f"Error analyzing brand synergies: {str(e)}")
        
        return synergies
    
    def _assess_brand_pair_synergy(self, brand1: str, brand2: str) -> Dict[str, Any]:
        """Assess synergy between two brands"""
        analysis = {
            'score': 0.0,
            'synergy_type': 'neutral',
            'synergy_areas': [],
            'competition_areas': []
        }
        
        try:
            # Get brand data
            brand1_data = self.available_brands.get(brand1, {})
            brand2_data = self.available_brands.get(brand2, {})
            
            if not brand1_data or not brand2_data:
                return analysis
            
            # Analyze performance patterns
            brand1_performance = brand1_data.get('performance_patterns', {})
            brand2_performance = brand2_data.get('performance_patterns', {})
            
            # Check for complementary strengths/weaknesses
            complementarity_score = self._calculate_complementarity_score(brand1_performance, brand2_performance)
            
            # Check for competitive overlap
            competition_score = self._calculate_competition_score(brand1_data, brand2_data)
            
            if complementarity_score > 0.6:
                analysis['synergy_type'] = 'synergistic'
                analysis['score'] = complementarity_score
                analysis['synergy_areas'] = self._identify_synergy_areas(brand1_performance, brand2_performance)
            elif competition_score > 0.7:
                analysis['synergy_type'] = 'competitive'
                analysis['score'] = competition_score
                analysis['competition_areas'] = self._identify_competition_areas(brand1_data, brand2_data)
            else:
                analysis['synergy_type'] = 'neutral'
                analysis['score'] = 0.5
            
        except Exception as e:
            self.logger.error(f"Error assessing brand pair synergy: {str(e)}")
        
        return analysis
    
    def _calculate_complementarity_score(self, perf1: Dict[str, Any], perf2: Dict[str, Any]) -> float:
        """Calculate complementarity score between two performance patterns"""
        try:
            if not perf1 or not perf2:
                return 0.0
            
            # Get sectional strengths
            strengths1 = set(perf1.get('sectional_strengths', {}).keys())
            strengths2 = set(perf2.get('sectional_strengths', {}).keys())
            
            # Get sectional weaknesses
            weaknesses1 = set(perf1.get('improvement_opportunities', []))
            weaknesses2 = set(perf2.get('improvement_opportunities', []))
            
            # Calculate complementarity
            complementary_areas = 0
            total_areas = len(strengths1.union(strengths2).union(weaknesses1).union(weaknesses2))
            
            if total_areas > 0:
                # Brand 1 strong where Brand 2 weak
                complementary_areas += len(strengths1.intersection(weaknesses2))
                # Brand 2 strong where Brand 1 weak
                complementary_areas += len(strengths2.intersection(weaknesses1))
                
                return complementary_areas / total_areas
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating complementarity score: {str(e)}")
            return 0.0
    
    def _calculate_competition_score(self, brand1_data: Dict[str, Any], brand2_data: Dict[str, Any]) -> float:
        """Calculate competition intensity between two brands"""
        try:
            # Check for similar performance patterns
            perf1 = brand1_data.get('performance_patterns', {})
            perf2 = brand2_data.get('performance_patterns', {})
            
            if not perf1 or not perf2:
                return 0.0
            
            # Similar strengths indicate competition
            strengths1 = set(perf1.get('sectional_strengths', {}).keys())
            strengths2 = set(perf2.get('sectional_strengths', {}).keys())
            
            if strengths1 and strengths2:
                overlap = len(strengths1.intersection(strengths2))
                total = len(strengths1.union(strengths2))
                return overlap / total if total > 0 else 0.0
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating competition score: {str(e)}")
            return 0.0
    
    def _check_cross_selection_data_availability(self, selected_projects: List[int], 
                                               selected_brands: List[str]) -> Dict[str, Any]:
        """
        Check data availability for cross-selection analysis
        
        Args:
            selected_projects: Selected project IDs
            selected_brands: Selected brand IDs
            
        Returns:
            Dict with data availability assessment
        """
        availability = {
            'overall_availability': 0.0,
            'project_brand_matrix': {},
            'missing_combinations': [],
            'data_quality_matrix': {},
            'recommended_combinations': [],
            'analysis_feasibility': {}
        }
        
        try:
            # Create project-brand availability matrix
            matrix = {}
            total_combinations = 0
            available_combinations = 0
            
            for project_id in selected_projects:
                matrix[project_id] = {}
                project_data = self.available_projects.get(project_id, {})
                
                for brand in selected_brands:
                    total_combinations += 1
                    
                    # Check if brand data exists for this project
                    brand_available = brand in project_data.get('brands_available', [])
                    data_quality = self._assess_combination_data_quality(project_id, brand)
                    
                    matrix[project_id][brand] = {
                        'available': brand_available,
                        'data_quality': data_quality,
                        'completeness': data_quality.get('completeness', 0.0)
                    }
                    
                    if brand_available and data_quality.get('completeness', 0) > 0.5:
                        available_combinations += 1
                    else:
                        availability['missing_combinations'].append((project_id, brand))
            
            availability['project_brand_matrix'] = matrix
            availability['overall_availability'] = available_combinations / total_combinations if total_combinations > 0 else 0.0
            
            # Recommend best combinations
            availability['recommended_combinations'] = self._recommend_best_combinations(matrix)
            
            # Assess analysis feasibility
            availability['analysis_feasibility'] = self._assess_analysis_feasibility(matrix, selected_projects, selected_brands)
            
        except Exception as e:
            self.logger.error(f"Error checking cross-selection data availability: {str(e)}")
        
        return availability
    
    def _assess_combination_data_quality(self, project_id: int, brand: str) -> Dict[str, Any]:
        """Assess data quality for a specific project-brand combination"""
        quality = {
            'completeness': 0.0,
            'consistency': 0.0,
            'timeliness': 0.0,
            'overall_score': 0.0
        }
        
        try:
            project_data = self.available_projects.get(project_id, {})
            
            if brand in project_data.get('brands_available', []):
                # Get data quality metrics
                data_quality = project_data.get('data_quality', {})
                brand_completeness = data_quality.get('column_completeness', {}).get(brand, 0.0)
                
                quality['completeness'] = brand_completeness / 100.0 if brand_completeness > 0 else 0.0
                quality['consistency'] = 0.8  # Assume good consistency for available data
                quality['timeliness'] = 0.9    # Assume recent data
                
                # Calculate overall score
                quality['overall_score'] = (quality['completeness'] * 0.5 + 
                                          quality['consistency'] * 0.3 + 
                                          quality['timeliness'] * 0.2)
        
        except Exception as e:
            self.logger.error(f"Error assessing combination data quality: {str(e)}")
        
        return quality
    
    def _identify_cross_analysis_opportunities(self, selected_projects: List[int], 
                                            selected_brands: List[str]) -> Dict[str, Any]:
        """
        Identify opportunities for cross-analysis
        
        Args:
            selected_projects: Selected project IDs
            selected_brands: Selected brand IDs
            
        Returns:
            Dict with cross-analysis opportunities
        """
        opportunities = {
            'cross_project_analysis': [],
            'cross_brand_analysis': [],
            'portfolio_analysis': [],
            'competitive_analysis': [],
            'trend_analysis': [],
            'synergy_analysis': []
        }
        
        try:
            # Cross-project opportunities
            if len(selected_projects) > 1:
                opportunities['cross_project_analysis'] = [
                    'project_performance_comparison',
                    'cross_project_trend_analysis',
                    'project_portfolio_optimization',
                    'resource_allocation_optimization'
                ]
            
            # Cross-brand opportunities
            if len(selected_brands) > 1:
                opportunities['cross_brand_analysis'] = [
                    'brand_performance_benchmarking',
                    'competitive_positioning_analysis',
                    'brand_synergy_identification',
                    'market_share_analysis'
                ]
            
            # Portfolio analysis (multiple projects and brands)
            if len(selected_projects) > 1 and len(selected_brands) > 1:
                opportunities['portfolio_analysis'] = [
                    'portfolio_performance_optimization',
                    'cross_portfolio_synergies',
                    'resource_allocation_across_portfolio',
                    'risk_diversification_analysis'
                ]
            
            # Competitive analysis opportunities
            if len(selected_brands) >= 2:
                opportunities['competitive_analysis'] = [
                    'competitive_gap_analysis',
                    'market_positioning_analysis',
                    'competitive_advantage_identification',
                    'market_share_dynamics'
                ]
            
            # Trend analysis opportunities
            opportunities['trend_analysis'] = [
                'performance_trend_identification',
                'seasonal_pattern_analysis',
                'growth_trajectory_analysis',
                'market_trend_correlation'
            ]
            
            # Synergy analysis opportunities
            if len(selected_brands) >= 2 or len(selected_projects) >= 2:
                opportunities['synergy_analysis'] = [
                    'cross_brand_synergy_identification',
                    'project_synergy_analysis',
                    'resource_sharing_opportunities',
                    'collaborative_strategy_development'
                ]
        
        except Exception as e:
            self.logger.error(f"Error identifying cross-analysis opportunities: {str(e)}")
        
        return opportunities
    
    def _assess_selection_complexity(self, selected_projects: List[int], selected_brands: List[str]) -> Dict[str, Any]:
        """Assess complexity of the current selection"""
        complexity = {
            'overall_complexity': 'medium',
            'computational_complexity': 0.0,
            'analytical_complexity': 0.0,
            'data_complexity': 0.0,
            'complexity_factors': [],
            'simplification_recommendations': []
        }
        
        try:
            # Calculate computational complexity
            total_combinations = len(selected_projects) * len(selected_brands)
            computational_complexity = min(1.0, total_combinations / 50.0)  # Normalize to 0-1
            
            # Calculate analytical complexity
            analysis_types = len(self._identify_cross_analysis_opportunities(selected_projects, selected_brands))
            analytical_complexity = min(1.0, analysis_types / 20.0)  # Normalize to 0-1
            
            # Calculate data complexity
            total_data_points = 0
            for project_id in selected_projects:
                project_data = self.available_projects.get(project_id, {})
                metrics_count = len(project_data.get('metrics_available', []))
                total_data_points += metrics_count * len(selected_brands)
            
            data_complexity = min(1.0, total_data_points / 1000.0)  # Normalize to 0-1
            
            # Overall complexity
            overall_score = (computational_complexity + analytical_complexity + data_complexity) / 3
            
            if overall_score > 0.7:
                complexity['overall_complexity'] = 'high'
            elif overall_score > 0.4:
                complexity['overall_complexity'] = 'medium'
            else:
                complexity['overall_complexity'] = 'low'
            
            complexity['computational_complexity'] = computational_complexity
            complexity['analytical_complexity'] = analytical_complexity
            complexity['data_complexity'] = data_complexity
            
            # Identify complexity factors
            if len(selected_projects) > 5:
                complexity['complexity_factors'].append('High number of projects')
            if len(selected_brands) > 10:
                complexity['complexity_factors'].append('High number of brands')
            if total_combinations > 100:
                complexity['complexity_factors'].append('High number of combinations')
            
            # Provide simplification recommendations
            if complexity['overall_complexity'] == 'high':
                complexity['simplification_recommendations'] = [
                    'Consider reducing number of projects or brands',
                    'Focus on most important combinations first',
                    'Use phased analysis approach',
                    'Implement parallel processing'
                ]
        
        except Exception as e:
            self.logger.error(f"Error assessing selection complexity: {str(e)}")
        
        return complexity
    
    def _adapt_analysis_approach(self, selection_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt analysis approach based on selection characteristics
        
        Args:
            selection_analysis: Results from selection analysis
            
        Returns:
            Dict with adapted analysis approach
        """
        approach = {
            'analysis_strategy': 'comprehensive',
            'processing_order': [],
            'parallelization_strategy': {},
            'resource_allocation': {},
            'optimization_priorities': [],
            'reporting_strategy': {}
        }
        
        try:
            complexity = selection_analysis.get('complexity_assessment', {})
            compatibility = selection_analysis.get('project_compatibility', {})
            synergies = selection_analysis.get('brand_synergies', {})
            
            # Determine analysis strategy
            if complexity.get('overall_complexity') == 'high':
                approach['analysis_strategy'] = 'phased_incremental'
                approach['optimization_priorities'] = [
                    'data_preprocessing',
                    'parallel_processing',
                    'memory_optimization',
                    'result_caching'
                ]
            elif complexity.get('overall_complexity') == 'low':
                approach['analysis_strategy'] = 'comprehensive_batch'
                approach['optimization_priorities'] = [
                    'comprehensive_analysis',
                    'detailed_reporting'
                ]
            else:
                approach['analysis_strategy'] = 'adaptive_hybrid'
                approach['optimization_priorities'] = [
                    'balanced_processing',
                    'selective_analysis'
                ]
            
            # Determine processing order
            if compatibility.get('compatibility_score', 0) > 0.7:
                approach['processing_order'] = [
                    'compatible_project_groups',
                    'individual_projects',
                    'cross_project_analysis'
                ]
            else:
                approach['processing_order'] = [
                    'individual_projects',
                    'pairwise_comparisons',
                    'selective_cross_analysis'
                ]
            
            # Parallelization strategy
            if len(self.selected_projects) > 2 or len(self.selected_brands) > 5:
                approach['parallelization_strategy'] = {
                    'parallel_projects': True,
                    'parallel_brands': True,
                    'parallel_reports': True,
                    'max_workers': min(8, len(self.selected_projects) + len(self.selected_brands))
                }
            else:
                approach['parallelization_strategy'] = {
                    'parallel_projects': False,
                    'parallel_brands': False,
                    'parallel_reports': True,
                    'max_workers': 2
                }
            
            # Resource allocation
            total_resources = 100  # Percentage
            if synergies.get('synergy_score', 0) > 0.6:
                approach['resource_allocation'] = {
                    'synergy_analysis': 30,
                    'individual_analysis': 40,
                    'cross_analysis': 20,
                    'reporting': 10
                }
            else:
                approach['resource_allocation'] = {
                    'individual_analysis': 50,
                    'comparative_analysis': 30,
                    'cross_analysis': 10,
                    'reporting': 10
                }
            
            # Reporting strategy
            if len(self.selected_projects) > 3 or len(self.selected_brands) > 5:
                approach['reporting_strategy'] = {
                    'summary_reports': True,
                    'detailed_reports': False,
                    'interactive_dashboards': True,
                    'executive_summaries': True
                }
            else:
                approach['reporting_strategy'] = {
                    'summary_reports': True,
                    'detailed_reports': True,
                    'interactive_dashboards': False,
                    'executive_summaries': True
                }
        
        except Exception as e:
            self.logger.error(f"Error adapting analysis approach: {str(e)}")
        
        return approach
    
    def _identify_available_reports(self, selection_analysis: Dict[str, Any]) -> List[str]:
        """Identify which reports are available based on selection analysis"""
        available_reports = []
        
        try:
            opportunities = selection_analysis.get('analysis_opportunities', {})
            data_availability = selection_analysis.get('data_availability', {})
            
            # Base reports always available
            available_reports.extend([
                'individual_brand_performance',
                'individual_project_analysis'
            ])
            
            # Cross-project reports
            if len(self.selected_projects) > 1 and data_availability.get('overall_availability', 0) > 0.5:
                available_reports.extend([
                    'cross_project_comparison',
                    'project_portfolio_analysis',
                    'resource_allocation_optimization'
                ])
            
            # Cross-brand reports
            if len(self.selected_brands) > 1:
                available_reports.extend([
                    'brand_competitive_analysis',
                    'brand_synergy_analysis',
                    'market_positioning_analysis'
                ])
            
            # Advanced reports for complex selections
            if len(self.selected_projects) > 1 and len(self.selected_brands) > 1:
                available_reports.extend([
                    'portfolio_optimization_analysis',
                    'cross_portfolio_synergies',
                    'strategic_planning_analysis'
                ])
        
        except Exception as e:
            self.logger.error(f"Error identifying available reports: {str(e)}")
        
        return available_reports
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of current selection and analysis"""
        summary = {
            'selection_timestamp': datetime.now().isoformat(),
            'selected_projects': self.selected_projects,
            'selected_brands': self.selected_brands,
            'selection_size': {
                'projects': len(self.selected_projects),
                'brands': len(self.selected_brands),
                'total_combinations': len(self.selected_projects) * len(self.selected_brands)
            },
            'analysis_status': 'completed' if self.selection_analysis else 'pending',
            'key_insights': self._generate_selection_insights(),
            'recommendations': self._generate_selection_recommendations()
        }
        
        if self.selection_analysis:
            summary['analysis_results'] = {
                'compatibility_score': self.selection_analysis.get('project_compatibility', {}).get('compatibility_score', 0),
                'synergy_score': self.selection_analysis.get('brand_synergies', {}).get('synergy_score', 0),
                'data_availability': self.selection_analysis.get('data_availability', {}).get('overall_availability', 0),
                'complexity_level': self.selection_analysis.get('complexity_assessment', {}).get('overall_complexity', 'unknown')
            }
        
        return summary
    
    def _generate_selection_insights(self) -> List[str]:
        """Generate key insights from selection analysis"""
        insights = []
        
        try:
            if self.selection_analysis:
                # Project compatibility insights
                compatibility = self.selection_analysis.get('project_compatibility', {})
                if compatibility.get('compatibility_score', 0) > 0.8:
                    insights.append("Selected projects are highly compatible for cross-analysis")
                elif compatibility.get('compatibility_score', 0) < 0.5:
                    insights.append("Selected projects have limited compatibility - consider individual analysis")
                
                # Brand synergy insights
                synergies = self.selection_analysis.get('brand_synergies', {})
                if synergies.get('synergy_score', 0) > 0.6:
                    insights.append("Strong synergies identified between selected brands")
                elif len(synergies.get('competitive_pairs', [])) > 0:
                    insights.append("Competitive relationships detected between some brands")
                
                # Data availability insights
                data_availability = self.selection_analysis.get('data_availability', {})
                if data_availability.get('overall_availability', 0) > 0.8:
                    insights.append("Excellent data availability for comprehensive analysis")
                elif data_availability.get('overall_availability', 0) < 0.5:
                    insights.append("Limited data availability may restrict analysis scope")
        
        except Exception as e:
            self.logger.error(f"Error generating selection insights: {str(e)}")
        
        return insights
    
    def _generate_selection_recommendations(self) -> List[str]:
        """Generate recommendations based on selection analysis"""
        recommendations = []
        
        try:
            if self.selection_analysis:
                complexity = self.selection_analysis.get('complexity_assessment', {})
                
                if complexity.get('overall_complexity') == 'high':
                    recommendations.extend([
                        "Consider phased analysis approach due to high complexity",
                        "Implement parallel processing for better performance",
                        "Focus on most important brand-project combinations first"
                    ])
                
                # Data availability recommendations
                data_availability = self.selection_analysis.get('data_availability', {})
                missing_combinations = data_availability.get('missing_combinations', [])
                
                if missing_combinations:
                    recommendations.append(f"Consider excluding {len(missing_combinations)} combinations with insufficient data")
                
                # Synergy recommendations
                synergies = self.selection_analysis.get('brand_synergies', {})
                if synergies.get('synergy_score', 0) > 0.6:
                    recommendations.append("Prioritize synergy analysis reports for maximum insights")
        
        except Exception as e:
            self.logger.error(f"Error generating selection recommendations: {str(e)}")
        
        return recommendations
    
    # Helper methods for internal calculations
    def _find_common_metrics(self, projects: List[int]) -> List[str]:
        """Find metrics common across all selected projects"""
        if not projects:
            return []
        
        common_metrics = None
        for project_id in projects:
            project_data = self.available_projects.get(project_id, {})
            project_metrics = set(project_data.get('metrics_available', []))
            
            if common_metrics is None:
                common_metrics = project_metrics
            else:
                common_metrics = common_metrics.intersection(project_metrics)
        
        return list(common_metrics) if common_metrics else []
    
    def _find_common_brands(self, projects: List[int]) -> List[str]:
        """Find brands common across all selected projects"""
        if not projects:
            return []
        
        common_brands = None
        for project_id in projects:
            project_data = self.available_projects.get(project_id, {})
            project_brands = set(project_data.get('brands_available', []))
            
            if common_brands is None:
                common_brands = project_brands
            else:
                common_brands = common_brands.intersection(project_brands)
        
        return list(common_brands) if common_brands else []
    
    def _assess_data_structure_similarity(self, projects: List[int]) -> float:
        """Assess similarity of data structures across projects"""
        if len(projects) < 2:
            return 1.0
        
        try:
            # Compare section and platform structures
            structures = []
            for project_id in projects:
                project_data = self.available_projects.get(project_id, {})
                structure = {
                    'sections': set(project_data.get('sections_available', [])),
                    'platforms': set(project_data.get('platforms_available', []))
                }
                structures.append(structure)
            
            # Calculate average pairwise similarity
            similarities = []
            for i in range(len(structures)):
                for j in range(i+1, len(structures)):
                    struct1, struct2 = structures[i], structures[j]
                    
                    # Section similarity
                    section_sim = len(struct1['sections'].intersection(struct2['sections'])) / \
                                len(struct1['sections'].union(struct2['sections'])) if \
                                len(struct1['sections'].union(struct2['sections'])) > 0 else 0
                    
                    # Platform similarity
                    platform_sim = len(struct1['platforms'].intersection(struct2['platforms'])) / \
                                 len(struct1['platforms'].union(struct2['platforms'])) if \
                                 len(struct1['platforms'].union(struct2['platforms'])) > 0 else 0
                    
                    # Average similarity
                    similarities.append((section_sim + platform_sim) / 2)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"Error assessing data structure similarity: {str(e)}")
            return 0.0

