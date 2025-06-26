"""
Dynamic Score Analyzer
Adaptive score analysis based on actual data patterns and business context
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class DynamicScoreAnalyzer:
    """
    Dynamic analyzer that adapts to actual DC score patterns and provides contextual insights
    """
    
    def __init__(self, project_data: Dict[str, Any], brand_data: Optional[Dict[str, Any]] = None):
        """
        Initialize with project and brand data
        
        Args:
            project_data: Processed project data from DynamicDataManager
            brand_data: Optional brand-specific data
        """
        self.project_data = project_data
        self.brand_data = brand_data or {}
        self.score_patterns = {}
        self.sectional_analysis = {}
        self.competitive_analysis = {}
        self.business_correlations = {}
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Digi-Cadence specific sections
        self.dc_sections = ['Marketplace', 'Digital Spends', 'Organic Performance', 'Socialwatch']
        
        # Initialize analysis
        self._initialize_analysis()
    
    def _initialize_analysis(self):
        """Initialize the analysis framework"""
        try:
            self.logger.info("Initializing dynamic score analysis...")
            
            # Extract and validate score data
            self.score_data = self._extract_score_data()
            
            # Identify available brands and metrics
            self.available_brands = self._identify_available_brands()
            self.available_metrics = self._identify_available_metrics()
            self.available_sections = self._identify_available_sections()
            
            self.logger.info(f"Analysis initialized: {len(self.available_brands)} brands, {len(self.available_metrics)} metrics, {len(self.available_sections)} sections")
            
        except Exception as e:
            self.logger.error(f"Error initializing analysis: {str(e)}")
            raise
    
    def analyze_dc_scores_dynamically(self) -> Dict[str, Any]:
        """
        Analyze DC scores and sectional scores dynamically based on actual data patterns
        
        Returns:
            Dict containing comprehensive score analysis
        """
        try:
            self.logger.info("Starting dynamic DC score analysis...")
            
            # Extract current scores
            current_scores = self._extract_current_scores()
            
            # Identify patterns specific to this data
            patterns = {
                'performance_trends': self._identify_performance_trends(current_scores),
                'sectional_strengths': self._identify_sectional_strengths(current_scores),
                'improvement_opportunities': self._identify_improvement_opportunities(current_scores),
                'competitive_position': self._assess_competitive_position(current_scores),
                'business_correlation_potential': self._assess_business_correlation_potential(current_scores),
                'score_distribution_analysis': self._analyze_score_distributions(current_scores),
                'outlier_detection': self._detect_score_outliers(current_scores),
                'clustering_analysis': self._perform_clustering_analysis(current_scores)
            }
            
            # Store patterns for future use
            self.score_patterns = patterns
            
            self.logger.info("Dynamic DC score analysis completed")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in dynamic score analysis: {str(e)}")
            raise
    
    def _extract_score_data(self) -> Dict[str, pd.DataFrame]:
        """Extract score data from project data"""
        score_data = {}
        
        try:
            for project_id, project_info in self.project_data.items():
                if 'metrics_data' in project_info and project_info['metrics_data'] is not None:
                    score_data[project_id] = project_info['metrics_data']
                elif 'normalized_scores' in project_info and project_info['normalized_scores'] is not None:
                    score_data[project_id] = project_info['normalized_scores']
        
        except Exception as e:
            self.logger.error(f"Error extracting score data: {str(e)}")
        
        return score_data
    
    def _identify_available_brands(self) -> List[str]:
        """Identify available brands from the data"""
        brands = set()
        
        try:
            for project_id, df in self.score_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Get brand columns (excluding metadata columns)
                    brand_columns = [col for col in df.columns 
                                   if col not in ['Metric', 'section_name', 'platform_name', 'metricname', 'sectionName', 'platformname']]
                    brands.update(brand_columns)
        
        except Exception as e:
            self.logger.error(f"Error identifying brands: {str(e)}")
        
        return list(brands)
    
    def _identify_available_metrics(self) -> List[str]:
        """Identify available metrics from the data"""
        metrics = set()
        
        try:
            for project_id, df in self.score_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if 'Metric' in df.columns:
                        metrics.update(df['Metric'].unique())
                    elif 'metricname' in df.columns:
                        metrics.update(df['metricname'].unique())
        
        except Exception as e:
            self.logger.error(f"Error identifying metrics: {str(e)}")
        
        return list(metrics)
    
    def _identify_available_sections(self) -> List[str]:
        """Identify available sections from the data"""
        sections = set()
        
        try:
            for project_id, df in self.score_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if 'section_name' in df.columns:
                        sections.update(df['section_name'].unique())
                    elif 'sectionName' in df.columns:
                        sections.update(df['sectionName'].unique())
        
        except Exception as e:
            self.logger.error(f"Error identifying sections: {str(e)}")
        
        return list(sections)
    
    def _extract_current_scores(self) -> Dict[str, Dict[str, Any]]:
        """Extract current DC scores and sectional scores"""
        current_scores = {}
        
        try:
            for project_id, df in self.score_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    project_scores = {}
                    
                    # Calculate DC scores (overall weighted average)
                    dc_scores = self._calculate_dc_scores(df)
                    project_scores['dc_scores'] = dc_scores
                    
                    # Calculate sectional scores
                    sectional_scores = self._calculate_sectional_scores(df)
                    project_scores['sectional_scores'] = sectional_scores
                    
                    # Calculate platform-wise scores
                    platform_scores = self._calculate_platform_scores(df)
                    project_scores['platform_scores'] = platform_scores
                    
                    # Calculate metric-wise scores
                    metric_scores = self._calculate_metric_scores(df)
                    project_scores['metric_scores'] = metric_scores
                    
                    current_scores[project_id] = project_scores
        
        except Exception as e:
            self.logger.error(f"Error extracting current scores: {str(e)}")
        
        return current_scores
    
    def _calculate_dc_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall DC scores for each brand"""
        dc_scores = {}
        
        try:
            brand_columns = [col for col in df.columns 
                           if col not in ['Metric', 'section_name', 'platform_name', 'metricname', 'sectionName', 'platformname']]
            
            for brand in brand_columns:
                brand_scores = pd.to_numeric(df[brand], errors='coerce').dropna()
                if not brand_scores.empty:
                    # Simple average for now - can be enhanced with weights
                    dc_scores[brand] = float(brand_scores.mean())
        
        except Exception as e:
            self.logger.error(f"Error calculating DC scores: {str(e)}")
        
        return dc_scores
    
    def _calculate_sectional_scores(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate sectional scores for each brand"""
        sectional_scores = {}
        
        try:
            section_col = 'section_name' if 'section_name' in df.columns else 'sectionName'
            
            if section_col in df.columns:
                brand_columns = [col for col in df.columns 
                               if col not in ['Metric', 'section_name', 'platform_name', 'metricname', 'sectionName', 'platformname']]
                
                for brand in brand_columns:
                    brand_sectional = {}
                    
                    for section in df[section_col].unique():
                        section_data = df[df[section_col] == section]
                        section_scores = pd.to_numeric(section_data[brand], errors='coerce').dropna()
                        
                        if not section_scores.empty:
                            brand_sectional[section] = float(section_scores.mean())
                    
                    if brand_sectional:
                        sectional_scores[brand] = brand_sectional
        
        except Exception as e:
            self.logger.error(f"Error calculating sectional scores: {str(e)}")
        
        return sectional_scores
    
    def _calculate_platform_scores(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate platform-wise scores for each brand"""
        platform_scores = {}
        
        try:
            platform_col = 'platform_name' if 'platform_name' in df.columns else 'platformname'
            
            if platform_col in df.columns:
                brand_columns = [col for col in df.columns 
                               if col not in ['Metric', 'section_name', 'platform_name', 'metricname', 'sectionName', 'platformname']]
                
                for brand in brand_columns:
                    brand_platform = {}
                    
                    for platform in df[platform_col].unique():
                        platform_data = df[df[platform_col] == platform]
                        platform_brand_scores = pd.to_numeric(platform_data[brand], errors='coerce').dropna()
                        
                        if not platform_brand_scores.empty:
                            brand_platform[platform] = float(platform_brand_scores.mean())
                    
                    if brand_platform:
                        platform_scores[brand] = brand_platform
        
        except Exception as e:
            self.logger.error(f"Error calculating platform scores: {str(e)}")
        
        return platform_scores
    
    def _calculate_metric_scores(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate individual metric scores for each brand"""
        metric_scores = {}
        
        try:
            metric_col = 'Metric' if 'Metric' in df.columns else 'metricname'
            
            if metric_col in df.columns:
                brand_columns = [col for col in df.columns 
                               if col not in ['Metric', 'section_name', 'platform_name', 'metricname', 'sectionName', 'platformname']]
                
                for brand in brand_columns:
                    brand_metrics = {}
                    
                    for metric in df[metric_col].unique():
                        metric_data = df[df[metric_col] == metric]
                        metric_brand_scores = pd.to_numeric(metric_data[brand], errors='coerce').dropna()
                        
                        if not metric_brand_scores.empty:
                            brand_metrics[metric] = float(metric_brand_scores.mean())
                    
                    if brand_metrics:
                        metric_scores[brand] = brand_metrics
        
        except Exception as e:
            self.logger.error(f"Error calculating metric scores: {str(e)}")
        
        return metric_scores
    
    def _identify_performance_trends(self, current_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Identify performance trends across brands and projects"""
        trends = {
            'overall_trends': {},
            'sectional_trends': {},
            'brand_performance_classification': {},
            'improvement_trajectories': {}
        }
        
        try:
            # Analyze overall performance trends
            all_dc_scores = {}
            for project_id, project_scores in current_scores.items():
                if 'dc_scores' in project_scores:
                    for brand, score in project_scores['dc_scores'].items():
                        if brand not in all_dc_scores:
                            all_dc_scores[brand] = []
                        all_dc_scores[brand].append(score)
            
            # Classify brand performance
            for brand, scores in all_dc_scores.items():
                avg_score = np.mean(scores)
                std_score = np.std(scores) if len(scores) > 1 else 0
                
                if avg_score >= 80:
                    performance_tier = 'high_performer'
                elif avg_score >= 60:
                    performance_tier = 'medium_performer'
                else:
                    performance_tier = 'low_performer'
                
                consistency = 'consistent' if std_score <= 10 else 'volatile'
                
                trends['brand_performance_classification'][brand] = {
                    'performance_tier': performance_tier,
                    'consistency': consistency,
                    'average_score': avg_score,
                    'score_volatility': std_score
                }
            
            # Analyze sectional trends
            sectional_performance = {}
            for project_id, project_scores in current_scores.items():
                if 'sectional_scores' in project_scores:
                    for brand, sections in project_scores['sectional_scores'].items():
                        if brand not in sectional_performance:
                            sectional_performance[brand] = {}
                        
                        for section, score in sections.items():
                            if section not in sectional_performance[brand]:
                                sectional_performance[brand][section] = []
                            sectional_performance[brand][section].append(score)
            
            # Identify sectional strengths and weaknesses
            for brand, sections in sectional_performance.items():
                brand_sectional_trends = {}
                for section, scores in sections.items():
                    avg_score = np.mean(scores)
                    if avg_score >= 75:
                        trend = 'strength'
                    elif avg_score <= 50:
                        trend = 'weakness'
                    else:
                        trend = 'neutral'
                    
                    brand_sectional_trends[section] = {
                        'trend': trend,
                        'average_score': avg_score,
                        'score_range': [min(scores), max(scores)] if len(scores) > 1 else [avg_score, avg_score]
                    }
                
                trends['sectional_trends'][brand] = brand_sectional_trends
        
        except Exception as e:
            self.logger.error(f"Error identifying performance trends: {str(e)}")
        
        return trends
    
    def _identify_sectional_strengths(self, current_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Identify sectional strengths and weaknesses for each brand"""
        sectional_analysis = {
            'brand_sectional_profiles': {},
            'section_rankings': {},
            'improvement_priorities': {},
            'cross_section_correlations': {}
        }
        
        try:
            # Analyze sectional performance for each brand
            for project_id, project_scores in current_scores.items():
                if 'sectional_scores' in project_scores:
                    for brand, sections in project_scores['sectional_scores'].items():
                        if brand not in sectional_analysis['brand_sectional_profiles']:
                            sectional_analysis['brand_sectional_profiles'][brand] = {}
                        
                        # Rank sections by performance
                        sorted_sections = sorted(sections.items(), key=lambda x: x[1], reverse=True)
                        
                        sectional_analysis['brand_sectional_profiles'][brand] = {
                            'strongest_section': sorted_sections[0] if sorted_sections else None,
                            'weakest_section': sorted_sections[-1] if sorted_sections else None,
                            'section_scores': sections,
                            'section_ranking': [section for section, score in sorted_sections]
                        }
                        
                        # Identify improvement priorities
                        improvement_priorities = []
                        for section, score in sections.items():
                            if score < 60:  # Below average performance
                                impact_potential = self._assess_section_impact_potential(section)
                                improvement_priorities.append({
                                    'section': section,
                                    'current_score': score,
                                    'improvement_potential': 100 - score,
                                    'impact_potential': impact_potential,
                                    'priority_score': (100 - score) * impact_potential
                                })
                        
                        # Sort by priority score
                        improvement_priorities.sort(key=lambda x: x['priority_score'], reverse=True)
                        sectional_analysis['improvement_priorities'][brand] = improvement_priorities
        
        except Exception as e:
            self.logger.error(f"Error identifying sectional strengths: {str(e)}")
        
        return sectional_analysis
    
    def _assess_section_impact_potential(self, section: str) -> float:
        """Assess the business impact potential of improving a specific section"""
        # Impact weights based on business importance (can be made dynamic)
        impact_weights = {
            'Marketplace': 0.9,  # High impact on sales and visibility
            'Digital Spends': 0.8,  # High impact on ROI and efficiency
            'Organic Performance': 0.7,  # Medium-high impact on long-term growth
            'Socialwatch': 0.6  # Medium impact on brand awareness
        }
        
        return impact_weights.get(section, 0.5)  # Default medium impact
    
    def _identify_improvement_opportunities(self, current_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Identify specific improvement opportunities based on score analysis"""
        opportunities = {
            'quick_wins': {},
            'strategic_investments': {},
            'competitive_gaps': {},
            'synergy_opportunities': {}
        }
        
        try:
            for project_id, project_scores in current_scores.items():
                if 'sectional_scores' in project_scores and 'metric_scores' in project_scores:
                    
                    for brand in project_scores['sectional_scores'].keys():
                        brand_opportunities = {
                            'quick_wins': [],
                            'strategic_investments': [],
                            'competitive_gaps': []
                        }
                        
                        # Identify quick wins (high impact, low effort)
                        sectional_scores = project_scores['sectional_scores'][brand]
                        for section, score in sectional_scores.items():
                            if 50 <= score <= 70:  # Medium performance with improvement potential
                                impact_potential = self._assess_section_impact_potential(section)
                                if impact_potential >= 0.7:
                                    brand_opportunities['quick_wins'].append({
                                        'section': section,
                                        'current_score': score,
                                        'target_score': min(85, score + 15),
                                        'impact_potential': impact_potential,
                                        'effort_estimate': 'medium'
                                    })
                        
                        # Identify strategic investments (high impact, high effort)
                        for section, score in sectional_scores.items():
                            if score < 50:  # Low performance requiring significant investment
                                impact_potential = self._assess_section_impact_potential(section)
                                if impact_potential >= 0.8:
                                    brand_opportunities['strategic_investments'].append({
                                        'section': section,
                                        'current_score': score,
                                        'target_score': 70,
                                        'impact_potential': impact_potential,
                                        'effort_estimate': 'high',
                                        'investment_required': 'significant'
                                    })
                        
                        opportunities['quick_wins'][brand] = brand_opportunities['quick_wins']
                        opportunities['strategic_investments'][brand] = brand_opportunities['strategic_investments']
        
        except Exception as e:
            self.logger.error(f"Error identifying improvement opportunities: {str(e)}")
        
        return opportunities
    
    def _assess_competitive_position(self, current_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess competitive position based on cross-brand analysis"""
        competitive_analysis = {
            'brand_rankings': {},
            'competitive_gaps': {},
            'market_position': {},
            'differentiation_opportunities': {}
        }
        
        try:
            if len(self.available_brands) > 1:
                # Aggregate scores across all projects for competitive analysis
                brand_aggregate_scores = {}
                
                for project_id, project_scores in current_scores.items():
                    if 'dc_scores' in project_scores:
                        for brand, score in project_scores['dc_scores'].items():
                            if brand not in brand_aggregate_scores:
                                brand_aggregate_scores[brand] = []
                            brand_aggregate_scores[brand].append(score)
                
                # Calculate average scores and rank brands
                brand_averages = {brand: np.mean(scores) for brand, scores in brand_aggregate_scores.items()}
                brand_ranking = sorted(brand_averages.items(), key=lambda x: x[1], reverse=True)
                
                competitive_analysis['brand_rankings'] = {
                    'overall_ranking': brand_ranking,
                    'market_leader': brand_ranking[0] if brand_ranking else None,
                    'market_laggard': brand_ranking[-1] if brand_ranking else None
                }
                
                # Identify competitive gaps
                if len(brand_ranking) >= 2:
                    leader_score = brand_ranking[0][1]
                    
                    for brand, score in brand_ranking[1:]:
                        gap = leader_score - score
                        competitive_analysis['competitive_gaps'][brand] = {
                            'gap_to_leader': gap,
                            'gap_percentage': (gap / leader_score) * 100 if leader_score > 0 else 0,
                            'position': brand_ranking.index((brand, score)) + 1,
                            'total_brands': len(brand_ranking)
                        }
        
        except Exception as e:
            self.logger.error(f"Error assessing competitive position: {str(e)}")
        
        return competitive_analysis
    
    def _assess_business_correlation_potential(self, current_scores: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Assess potential for business outcome correlations"""
        correlation_potential = {
            'revenue_correlation_potential': 0.0,
            'market_share_correlation_potential': 0.0,
            'customer_acquisition_correlation_potential': 0.0,
            'brand_equity_correlation_potential': 0.0
        }
        
        try:
            # Assess based on data availability and score patterns
            total_brands = len(self.available_brands)
            total_projects = len(current_scores)
            
            # Higher potential with more brands and projects
            base_potential = min(1.0, (total_brands * total_projects) / 10.0)
            
            # Assess score variance (higher variance = better correlation potential)
            all_scores = []
            for project_scores in current_scores.values():
                if 'dc_scores' in project_scores:
                    all_scores.extend(project_scores['dc_scores'].values())
            
            if all_scores:
                score_variance = np.var(all_scores)
                variance_factor = min(1.0, score_variance / 400.0)  # Normalize to 0-1
                
                # Adjust potential based on variance
                adjusted_potential = base_potential * (0.5 + 0.5 * variance_factor)
                
                correlation_potential = {
                    'revenue_correlation_potential': adjusted_potential * 0.9,
                    'market_share_correlation_potential': adjusted_potential * 0.8,
                    'customer_acquisition_correlation_potential': adjusted_potential * 0.7,
                    'brand_equity_correlation_potential': adjusted_potential * 0.6
                }
        
        except Exception as e:
            self.logger.error(f"Error assessing business correlation potential: {str(e)}")
        
        return correlation_potential
    
    def _analyze_score_distributions(self, current_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze score distributions for statistical insights"""
        distribution_analysis = {
            'brand_distributions': {},
            'sectional_distributions': {},
            'overall_statistics': {}
        }
        
        try:
            # Analyze brand score distributions
            for project_id, project_scores in current_scores.items():
                if 'dc_scores' in project_scores:
                    for brand, score in project_scores['dc_scores'].items():
                        if brand not in distribution_analysis['brand_distributions']:
                            distribution_analysis['brand_distributions'][brand] = []
                        distribution_analysis['brand_distributions'][brand].append(score)
            
            # Calculate distribution statistics
            for brand, scores in distribution_analysis['brand_distributions'].items():
                if scores:
                    distribution_analysis['brand_distributions'][brand] = {
                        'scores': scores,
                        'mean': np.mean(scores),
                        'median': np.median(scores),
                        'std': np.std(scores),
                        'min': min(scores),
                        'max': max(scores),
                        'skewness': stats.skew(scores) if len(scores) > 2 else 0,
                        'kurtosis': stats.kurtosis(scores) if len(scores) > 2 else 0
                    }
        
        except Exception as e:
            self.logger.error(f"Error analyzing score distributions: {str(e)}")
        
        return distribution_analysis
    
    def _detect_score_outliers(self, current_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Detect outliers in score data"""
        outlier_analysis = {
            'brand_outliers': {},
            'sectional_outliers': {},
            'metric_outliers': {}
        }
        
        try:
            # Detect brand-level outliers
            for project_id, project_scores in current_scores.items():
                if 'dc_scores' in project_scores:
                    scores = list(project_scores['dc_scores'].values())
                    if len(scores) > 2:
                        Q1 = np.percentile(scores, 25)
                        Q3 = np.percentile(scores, 75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = []
                        for brand, score in project_scores['dc_scores'].items():
                            if score < lower_bound or score > upper_bound:
                                outliers.append({
                                    'brand': brand,
                                    'score': score,
                                    'type': 'high' if score > upper_bound else 'low'
                                })
                        
                        if outliers:
                            outlier_analysis['brand_outliers'][project_id] = outliers
        
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {str(e)}")
        
        return outlier_analysis
    
    def _perform_clustering_analysis(self, current_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform clustering analysis to identify brand groups"""
        clustering_analysis = {
            'brand_clusters': {},
            'cluster_characteristics': {},
            'cluster_recommendations': {}
        }
        
        try:
            if len(self.available_brands) >= 3:
                # Prepare data for clustering
                brand_features = []
                brand_names = []
                
                for project_id, project_scores in current_scores.items():
                    if 'sectional_scores' in project_scores:
                        for brand, sections in project_scores['sectional_scores'].items():
                            if len(sections) >= 3:  # Need sufficient features
                                features = list(sections.values())
                                brand_features.append(features)
                                brand_names.append(brand)
                
                if len(brand_features) >= 3:
                    # Perform K-means clustering
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(brand_features)
                    
                    # Determine optimal number of clusters
                    max_clusters = min(4, len(brand_features) - 1)
                    
                    if max_clusters >= 2:
                        kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(scaled_features)
                        
                        # Organize results
                        for i, brand in enumerate(brand_names):
                            cluster_id = cluster_labels[i]
                            if cluster_id not in clustering_analysis['brand_clusters']:
                                clustering_analysis['brand_clusters'][cluster_id] = []
                            clustering_analysis['brand_clusters'][cluster_id].append(brand)
                        
                        # Analyze cluster characteristics
                        for cluster_id, brands in clustering_analysis['brand_clusters'].items():
                            cluster_features = [brand_features[i] for i, brand in enumerate(brand_names) if brand in brands]
                            if cluster_features:
                                avg_features = np.mean(cluster_features, axis=0)
                                clustering_analysis['cluster_characteristics'][cluster_id] = {
                                    'average_scores': avg_features.tolist(),
                                    'brand_count': len(brands),
                                    'brands': brands
                                }
        
        except Exception as e:
            self.logger.error(f"Error in clustering analysis: {str(e)}")
        
        return clustering_analysis
    
    def get_score_summary(self) -> Dict[str, Any]:
        """Get comprehensive score summary"""
        if not self.score_patterns:
            self.analyze_dc_scores_dynamically()
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'brands_analyzed': len(self.available_brands),
            'sections_analyzed': len(self.available_sections),
            'metrics_analyzed': len(self.available_metrics),
            'key_insights': self._generate_key_insights(),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _generate_key_insights(self) -> List[str]:
        """Generate key insights from the analysis"""
        insights = []
        
        try:
            if 'performance_trends' in self.score_patterns:
                trends = self.score_patterns['performance_trends']
                
                # High performers
                high_performers = [brand for brand, data in trends.get('brand_performance_classification', {}).items() 
                                 if data.get('performance_tier') == 'high_performer']
                if high_performers:
                    insights.append(f"High performing brands identified: {', '.join(high_performers)}")
                
                # Improvement opportunities
                low_performers = [brand for brand, data in trends.get('brand_performance_classification', {}).items() 
                                if data.get('performance_tier') == 'low_performer']
                if low_performers:
                    insights.append(f"Brands with significant improvement potential: {', '.join(low_performers)}")
        
        except Exception as e:
            self.logger.error(f"Error generating key insights: {str(e)}")
        
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            if 'improvement_opportunities' in self.score_patterns:
                opportunities = self.score_patterns['improvement_opportunities']
                
                # Quick wins
                for brand, quick_wins in opportunities.get('quick_wins', {}).items():
                    if quick_wins:
                        recommendations.append(f"Focus on quick wins for {brand}: {', '.join([qw['section'] for qw in quick_wins[:2]])}")
                
                # Strategic investments
                for brand, investments in opportunities.get('strategic_investments', {}).items():
                    if investments:
                        recommendations.append(f"Consider strategic investment in {brand}: {investments[0]['section']}")
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations

