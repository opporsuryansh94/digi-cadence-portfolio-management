"""
Correlation Analyzer for Digi-Cadence Portfolio Management Platform
Provides comprehensive correlation analysis across brands, projects, and metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from src.analytics.base_analyzer import BaseAnalyzer
from src.models.portfolio import Project, Brand, Metric, BrandMetric

class CorrelationAnalyzer(BaseAnalyzer):
    """
    Advanced correlation analyzer for portfolio management
    Supports multi-dimensional correlation analysis across brands, projects, and time periods
    """
    
    def __init__(self, projects: List[Project], brands: List[Brand], config: Dict[str, Any] = None):
        super().__init__(projects, brands)
        
        # Correlation analysis configuration
        self.config = config or {
            'correlation_methods': ['pearson', 'spearman', 'kendall'],
            'significance_threshold': 0.05,
            'min_correlation_threshold': 0.3,
            'clustering_threshold': 0.7,
            'network_threshold': 0.5,
            'time_window_days': 30,
            'min_data_points': 10,
            'outlier_detection': True,
            'seasonal_adjustment': True
        }
        
        # Analysis results storage
        self.correlation_matrices = {}
        self.correlation_networks = {}
        self.cluster_results = {}
        self.time_series_correlations = {}
        
        self.logger.info(f"Correlation Analyzer initialized with {len(projects)} projects and {len(brands)} brands")
    
    def analyze(self, analysis_type: str = 'comprehensive', time_period: str = 'last_6_months', 
                metrics: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive correlation analysis
        
        Args:
            analysis_type: Type of analysis ('comprehensive', 'cross_brand', 'cross_project', 'temporal')
            time_period: Time period for analysis
            metrics: Specific metrics to analyze
            **kwargs: Additional analysis parameters
        
        Returns:
            Dictionary containing correlation analysis results
        """
        try:
            self.logger.info(f"Starting correlation analysis: {analysis_type}")
            
            # Prepare data for analysis
            analysis_data = self._prepare_correlation_data(time_period, metrics)
            
            if analysis_data.empty:
                raise ValueError("No data available for correlation analysis")
            
            results = {
                'analysis_type': analysis_type,
                'time_period': time_period,
                'metrics_analyzed': metrics or 'all_available',
                'data_points': len(analysis_data),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            if analysis_type == 'comprehensive':
                results.update(self._perform_comprehensive_analysis(analysis_data, **kwargs))
            elif analysis_type == 'cross_brand':
                results.update(self._perform_cross_brand_analysis(analysis_data, **kwargs))
            elif analysis_type == 'cross_project':
                results.update(self._perform_cross_project_analysis(analysis_data, **kwargs))
            elif analysis_type == 'temporal':
                results.update(self._perform_temporal_analysis(analysis_data, **kwargs))
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Generate insights and recommendations
            results['insights'] = self._generate_correlation_insights(results)
            results['recommendations'] = self._generate_correlation_recommendations(results)
            
            self.logger.info("Correlation analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            raise
    
    def _prepare_correlation_data(self, time_period: str, metrics: List[str] = None) -> pd.DataFrame:
        """Prepare data for correlation analysis"""
        
        # Calculate time range
        end_date = datetime.utcnow()
        if time_period == 'last_6_months':
            start_date = end_date - timedelta(days=180)
        elif time_period == 'last_year':
            start_date = end_date - timedelta(days=365)
        elif time_period == 'last_3_months':
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=180)
        
        # Simulate data loading (in production, this would query BrandMetric table)
        data_points = []
        
        # Default metrics if none specified
        if not metrics:
            metrics = [
                'engagement_rate', 'reach', 'impressions', 'click_through_rate',
                'conversion_rate', 'cost_per_acquisition', 'return_on_ad_spend',
                'brand_awareness', 'sentiment_score', 'share_of_voice',
                'website_traffic', 'social_mentions', 'video_views'
            ]
        
        for project in self.projects:
            for brand in self.brands:
                # Generate time series data
                current_date = start_date
                while current_date <= end_date:
                    data_point = {
                        'project_id': str(project.id),
                        'project_name': project.name,
                        'project_type': project.project_type or 'unknown',
                        'brand_id': str(brand.id),
                        'brand_name': brand.name,
                        'brand_type': brand.brand_type or 'unknown',
                        'date': current_date,
                        'week': current_date.isocalendar()[1],
                        'month': current_date.month,
                        'quarter': (current_date.month - 1) // 3 + 1,
                        'day_of_week': current_date.weekday(),
                        'is_weekend': 1 if current_date.weekday() >= 5 else 0
                    }
                    
                    # Add metric values with realistic correlations
                    base_performance = np.random.uniform(0.5, 1.0)  # Base performance factor
                    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * current_date.timetuple().tm_yday / 365)
                    
                    for metric in metrics:
                        # Create realistic correlations between metrics
                        if metric == 'engagement_rate':
                            value = base_performance * 100 * seasonal_factor + np.random.normal(0, 10)
                        elif metric == 'reach':
                            # Correlated with engagement
                            engagement = data_point.get('engagement_rate', 50)
                            value = engagement * 1000 + np.random.normal(0, 5000)
                        elif metric == 'conversion_rate':
                            # Inversely correlated with reach (quality vs quantity)
                            reach = data_point.get('reach', 50000)
                            value = max(0.1, 10 - (reach / 10000) + np.random.normal(0, 2))
                        elif metric == 'cost_per_acquisition':
                            # Inversely correlated with conversion rate
                            conversion = data_point.get('conversion_rate', 5)
                            value = max(1, 50 / conversion + np.random.normal(0, 10))
                        else:
                            # Other metrics with some correlation to base performance
                            value = base_performance * np.random.uniform(20, 100) + np.random.normal(0, 5)
                        
                        data_point[metric] = max(0, value)  # Ensure non-negative values
                    
                    data_points.append(data_point)
                    current_date += timedelta(days=7)  # Weekly data points
        
        df = pd.DataFrame(data_points)
        
        # Apply outlier detection if enabled
        if self.config['outlier_detection']:
            df = self._remove_outliers(df, metrics)
        
        # Apply seasonal adjustment if enabled
        if self.config['seasonal_adjustment']:
            df = self._apply_seasonal_adjustment(df, metrics)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        
        df_clean = df.copy()
        
        for metric in metrics:
            if metric in df_clean.columns:
                Q1 = df_clean[metric].quantile(0.25)
                Q3 = df_clean[metric].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Replace outliers with median
                median_value = df_clean[metric].median()
                df_clean.loc[(df_clean[metric] < lower_bound) | (df_clean[metric] > upper_bound), metric] = median_value
        
        return df_clean
    
    def _apply_seasonal_adjustment(self, df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """Apply seasonal adjustment to metrics"""
        
        df_adjusted = df.copy()
        
        for metric in metrics:
            if metric in df_adjusted.columns:
                # Simple seasonal adjustment using moving average
                df_adjusted[f'{metric}_seasonal_adj'] = df_adjusted.groupby(['brand_id'])[metric].transform(
                    lambda x: x - x.rolling(window=12, min_periods=1).mean() + x.mean()
                )
        
        return df_adjusted
    
    def _perform_comprehensive_analysis(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis"""
        
        results = {}
        
        # 1. Overall correlation matrix
        results['overall_correlations'] = self._calculate_correlation_matrix(data)
        
        # 2. Cross-brand correlations
        results['cross_brand_correlations'] = self._analyze_cross_brand_correlations(data)
        
        # 3. Cross-project correlations
        results['cross_project_correlations'] = self._analyze_cross_project_correlations(data)
        
        # 4. Temporal correlations
        results['temporal_correlations'] = self._analyze_temporal_correlations(data)
        
        # 5. Network analysis
        results['correlation_networks'] = self._build_correlation_networks(data)
        
        # 6. Clustering analysis
        results['clustering_analysis'] = self._perform_clustering_analysis(data)
        
        # 7. Principal component analysis
        results['pca_analysis'] = self._perform_pca_analysis(data)
        
        return results
    
    def _calculate_correlation_matrix(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation matrices using different methods"""
        
        # Get numeric columns (metrics)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove non-metric columns
        exclude_cols = ['project_id', 'brand_id', 'week', 'month', 'quarter', 'day_of_week', 'is_weekend']
        metric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(metric_cols) < 2:
            return {'error': 'Insufficient metrics for correlation analysis'}
        
        correlation_results = {}
        
        for method in self.config['correlation_methods']:
            try:
                if method == 'pearson':
                    corr_matrix = data[metric_cols].corr(method='pearson')
                elif method == 'spearman':
                    corr_matrix = data[metric_cols].corr(method='spearman')
                elif method == 'kendall':
                    corr_matrix = data[metric_cols].corr(method='kendall')
                else:
                    continue
                
                # Calculate p-values for significance testing
                p_values = self._calculate_correlation_pvalues(data[metric_cols], method)
                
                # Filter significant correlations
                significant_correlations = self._filter_significant_correlations(
                    corr_matrix, p_values, self.config['significance_threshold']
                )
                
                correlation_results[method] = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'p_values': p_values,
                    'significant_correlations': significant_correlations,
                    'strong_correlations': self._identify_strong_correlations(corr_matrix),
                    'summary_stats': {
                        'mean_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
                        'max_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()),
                        'min_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min())
                    }
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate {method} correlation: {e}")
                correlation_results[method] = {'error': str(e)}
        
        return correlation_results
    
    def _calculate_correlation_pvalues(self, data: pd.DataFrame, method: str) -> Dict[str, Dict[str, float]]:
        """Calculate p-values for correlation coefficients"""
        
        p_values = {}
        columns = data.columns.tolist()
        
        for i, col1 in enumerate(columns):
            p_values[col1] = {}
            for j, col2 in enumerate(columns):
                if i == j:
                    p_values[col1][col2] = 0.0
                else:
                    try:
                        if method == 'pearson':
                            _, p_val = stats.pearsonr(data[col1].dropna(), data[col2].dropna())
                        elif method == 'spearman':
                            _, p_val = stats.spearmanr(data[col1].dropna(), data[col2].dropna())
                        elif method == 'kendall':
                            _, p_val = stats.kendalltau(data[col1].dropna(), data[col2].dropna())
                        else:
                            p_val = 1.0
                        
                        p_values[col1][col2] = float(p_val)
                    except:
                        p_values[col1][col2] = 1.0
        
        return p_values
    
    def _filter_significant_correlations(self, corr_matrix: pd.DataFrame, p_values: Dict[str, Dict[str, float]], 
                                       threshold: float) -> List[Dict[str, Any]]:
        """Filter correlations by statistical significance"""
        
        significant_correlations = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    correlation = corr_matrix.loc[col1, col2]
                    p_value = p_values.get(col1, {}).get(col2, 1.0)
                    
                    if p_value < threshold and abs(correlation) >= self.config['min_correlation_threshold']:
                        significant_correlations.append({
                            'metric1': col1,
                            'metric2': col2,
                            'correlation': float(correlation),
                            'p_value': float(p_value),
                            'significance': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.5 else 'weak',
                            'direction': 'positive' if correlation > 0 else 'negative'
                        })
        
        # Sort by absolute correlation strength
        significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return significant_correlations
    
    def _identify_strong_correlations(self, corr_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify strong correlations (above threshold)"""
        
        strong_correlations = []
        threshold = self.config['min_correlation_threshold']
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    correlation = corr_matrix.loc[col1, col2]
                    
                    if abs(correlation) >= threshold:
                        strong_correlations.append({
                            'metric1': col1,
                            'metric2': col2,
                            'correlation': float(correlation),
                            'strength': 'very_strong' if abs(correlation) > 0.8 else 'strong' if abs(correlation) > 0.6 else 'moderate'
                        })
        
        return strong_correlations
    
    def _analyze_cross_brand_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations across different brands"""
        
        cross_brand_results = {}
        
        # Get metric columns
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'project_type', 'brand_id', 'brand_name', 'brand_type',
            'date', 'week', 'month', 'quarter', 'day_of_week', 'is_weekend'
        ]]
        
        brand_correlations = {}
        
        # Calculate correlations between same metrics across different brands
        for metric in metric_cols:
            brand_metric_data = data.pivot_table(
                index='date', 
                columns='brand_name', 
                values=metric, 
                aggfunc='mean'
            )
            
            if brand_metric_data.shape[1] > 1:  # Need at least 2 brands
                brand_corr_matrix = brand_metric_data.corr()
                brand_correlations[metric] = {
                    'correlation_matrix': brand_corr_matrix.to_dict(),
                    'average_correlation': float(brand_corr_matrix.values[np.triu_indices_from(brand_corr_matrix.values, k=1)].mean()),
                    'max_correlation': float(brand_corr_matrix.values[np.triu_indices_from(brand_corr_matrix.values, k=1)].max()),
                    'min_correlation': float(brand_corr_matrix.values[np.triu_indices_from(brand_corr_matrix.values, k=1)].min())
                }
        
        cross_brand_results['metric_correlations_across_brands'] = brand_correlations
        
        # Analyze brand similarity based on metric patterns
        brand_similarity = self._calculate_brand_similarity(data, metric_cols)
        cross_brand_results['brand_similarity'] = brand_similarity
        
        # Identify brand clusters
        brand_clusters = self._cluster_brands_by_metrics(data, metric_cols)
        cross_brand_results['brand_clusters'] = brand_clusters
        
        return cross_brand_results
    
    def _calculate_brand_similarity(self, data: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
        """Calculate similarity between brands based on metric patterns"""
        
        # Aggregate metrics by brand
        brand_metrics = data.groupby('brand_name')[metric_cols].mean()
        
        if len(brand_metrics) < 2:
            return {'error': 'Insufficient brands for similarity analysis'}
        
        # Calculate correlation matrix between brands
        brand_similarity_matrix = brand_metrics.T.corr()
        
        # Find most similar brand pairs
        similar_pairs = []
        for i, brand1 in enumerate(brand_similarity_matrix.columns):
            for j, brand2 in enumerate(brand_similarity_matrix.columns):
                if i < j:
                    similarity = brand_similarity_matrix.loc[brand1, brand2]
                    similar_pairs.append({
                        'brand1': brand1,
                        'brand2': brand2,
                        'similarity': float(similarity),
                        'similarity_level': 'high' if similarity > 0.8 else 'medium' if similarity > 0.6 else 'low'
                    })
        
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'similarity_matrix': brand_similarity_matrix.to_dict(),
            'most_similar_pairs': similar_pairs[:10],  # Top 10 most similar pairs
            'average_similarity': float(brand_similarity_matrix.values[np.triu_indices_from(brand_similarity_matrix.values, k=1)].mean())
        }
    
    def _cluster_brands_by_metrics(self, data: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
        """Cluster brands based on metric patterns"""
        
        # Aggregate metrics by brand
        brand_metrics = data.groupby('brand_name')[metric_cols].mean()
        
        if len(brand_metrics) < 3:
            return {'error': 'Insufficient brands for clustering'}
        
        # Standardize metrics
        scaler = StandardScaler()
        brand_metrics_scaled = scaler.fit_transform(brand_metrics)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(brand_metrics_scaled, method='ward')
        
        # Determine optimal number of clusters (simple heuristic)
        n_clusters = min(max(2, len(brand_metrics) // 3), 5)
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Organize results
        clusters = {}
        for i, brand in enumerate(brand_metrics.index):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(brand)
        
        # Calculate cluster characteristics
        cluster_characteristics = {}
        for cluster_id, brands in clusters.items():
            cluster_data = brand_metrics.loc[brands]
            cluster_characteristics[cluster_id] = {
                'brands': brands,
                'size': len(brands),
                'centroid': cluster_data.mean().to_dict(),
                'variance': cluster_data.var().to_dict()
            }
        
        return {
            'clusters': clusters,
            'cluster_characteristics': cluster_characteristics,
            'n_clusters': n_clusters,
            'silhouette_score': self._calculate_silhouette_score(brand_metrics_scaled, cluster_labels)
        }
    
    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality"""
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(data, labels))
        except:
            return 0.0
    
    def _analyze_cross_project_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations across different projects"""
        
        cross_project_results = {}
        
        # Get metric columns
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'project_type', 'brand_id', 'brand_name', 'brand_type',
            'date', 'week', 'month', 'quarter', 'day_of_week', 'is_weekend'
        ]]
        
        project_correlations = {}
        
        # Calculate correlations between same metrics across different projects
        for metric in metric_cols:
            project_metric_data = data.pivot_table(
                index='date', 
                columns='project_name', 
                values=metric, 
                aggfunc='mean'
            )
            
            if project_metric_data.shape[1] > 1:  # Need at least 2 projects
                project_corr_matrix = project_metric_data.corr()
                project_correlations[metric] = {
                    'correlation_matrix': project_corr_matrix.to_dict(),
                    'average_correlation': float(project_corr_matrix.values[np.triu_indices_from(project_corr_matrix.values, k=1)].mean()),
                    'max_correlation': float(project_corr_matrix.values[np.triu_indices_from(project_corr_matrix.values, k=1)].max()),
                    'min_correlation': float(project_corr_matrix.values[np.triu_indices_from(project_corr_matrix.values, k=1)].min())
                }
        
        cross_project_results['metric_correlations_across_projects'] = project_correlations
        
        # Analyze project similarity
        project_similarity = self._calculate_project_similarity(data, metric_cols)
        cross_project_results['project_similarity'] = project_similarity
        
        return cross_project_results
    
    def _calculate_project_similarity(self, data: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
        """Calculate similarity between projects based on metric patterns"""
        
        # Aggregate metrics by project
        project_metrics = data.groupby('project_name')[metric_cols].mean()
        
        if len(project_metrics) < 2:
            return {'error': 'Insufficient projects for similarity analysis'}
        
        # Calculate correlation matrix between projects
        project_similarity_matrix = project_metrics.T.corr()
        
        # Find most similar project pairs
        similar_pairs = []
        for i, project1 in enumerate(project_similarity_matrix.columns):
            for j, project2 in enumerate(project_similarity_matrix.columns):
                if i < j:
                    similarity = project_similarity_matrix.loc[project1, project2]
                    similar_pairs.append({
                        'project1': project1,
                        'project2': project2,
                        'similarity': float(similarity),
                        'similarity_level': 'high' if similarity > 0.8 else 'medium' if similarity > 0.6 else 'low'
                    })
        
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'similarity_matrix': project_similarity_matrix.to_dict(),
            'most_similar_pairs': similar_pairs[:10],
            'average_similarity': float(project_similarity_matrix.values[np.triu_indices_from(project_similarity_matrix.values, k=1)].mean())
        }
    
    def _analyze_temporal_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how correlations change over time"""
        
        temporal_results = {}
        
        # Get metric columns
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'project_type', 'brand_id', 'brand_name', 'brand_type',
            'date', 'week', 'month', 'quarter', 'day_of_week', 'is_weekend'
        ]]
        
        # Rolling correlation analysis
        rolling_correlations = self._calculate_rolling_correlations(data, metric_cols)
        temporal_results['rolling_correlations'] = rolling_correlations
        
        # Seasonal correlation patterns
        seasonal_correlations = self._analyze_seasonal_correlations(data, metric_cols)
        temporal_results['seasonal_correlations'] = seasonal_correlations
        
        # Correlation stability analysis
        stability_analysis = self._analyze_correlation_stability(data, metric_cols)
        temporal_results['stability_analysis'] = stability_analysis
        
        return temporal_results
    
    def _calculate_rolling_correlations(self, data: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
        """Calculate rolling correlations over time"""
        
        # Sort by date
        data_sorted = data.sort_values('date')
        
        # Calculate rolling correlations for key metric pairs
        window_size = self.config['time_window_days'] // 7  # Convert to weeks
        rolling_results = {}
        
        # Select top metric pairs for rolling analysis
        if len(metric_cols) >= 2:
            # For demonstration, analyze first few metric pairs
            metric_pairs = [(metric_cols[i], metric_cols[j]) for i in range(min(3, len(metric_cols))) 
                           for j in range(i+1, min(i+4, len(metric_cols)))]
            
            for metric1, metric2 in metric_pairs:
                pair_key = f"{metric1}_{metric2}"
                
                # Calculate rolling correlation
                rolling_corr = data_sorted.set_index('date')[[metric1, metric2]].rolling(
                    window=f'{window_size}W'
                ).corr().unstack()[metric2][metric1]
                
                rolling_results[pair_key] = {
                    'metric1': metric1,
                    'metric2': metric2,
                    'rolling_correlation': rolling_corr.dropna().to_dict(),
                    'mean_correlation': float(rolling_corr.mean()),
                    'correlation_volatility': float(rolling_corr.std()),
                    'trend': 'increasing' if rolling_corr.iloc[-1] > rolling_corr.iloc[0] else 'decreasing'
                }
        
        return rolling_results
    
    def _analyze_seasonal_correlations(self, data: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
        """Analyze seasonal patterns in correlations"""
        
        seasonal_results = {}
        
        # Group by quarters
        quarterly_correlations = {}
        for quarter in [1, 2, 3, 4]:
            quarter_data = data[data['quarter'] == quarter]
            if len(quarter_data) > self.config['min_data_points']:
                quarter_corr = quarter_data[metric_cols].corr()
                quarterly_correlations[f'Q{quarter}'] = {
                    'correlation_matrix': quarter_corr.to_dict(),
                    'average_correlation': float(quarter_corr.values[np.triu_indices_from(quarter_corr.values, k=1)].mean())
                }
        
        seasonal_results['quarterly_correlations'] = quarterly_correlations
        
        # Weekend vs weekday correlations
        weekday_data = data[data['is_weekend'] == 0]
        weekend_data = data[data['is_weekend'] == 1]
        
        if len(weekday_data) > self.config['min_data_points'] and len(weekend_data) > self.config['min_data_points']:
            weekday_corr = weekday_data[metric_cols].corr()
            weekend_corr = weekend_data[metric_cols].corr()
            
            seasonal_results['weekday_vs_weekend'] = {
                'weekday_correlations': {
                    'correlation_matrix': weekday_corr.to_dict(),
                    'average_correlation': float(weekday_corr.values[np.triu_indices_from(weekday_corr.values, k=1)].mean())
                },
                'weekend_correlations': {
                    'correlation_matrix': weekend_corr.to_dict(),
                    'average_correlation': float(weekend_corr.values[np.triu_indices_from(weekend_corr.values, k=1)].mean())
                }
            }
        
        return seasonal_results
    
    def _analyze_correlation_stability(self, data: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
        """Analyze stability of correlations over time"""
        
        # Split data into time periods
        data_sorted = data.sort_values('date')
        n_periods = 4  # Split into 4 time periods
        period_size = len(data_sorted) // n_periods
        
        period_correlations = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(data_sorted)
            period_data = data_sorted.iloc[start_idx:end_idx]
            
            if len(period_data) > self.config['min_data_points']:
                period_corr = period_data[metric_cols].corr()
                period_correlations.append(period_corr.values[np.triu_indices_from(period_corr.values, k=1)])
        
        if len(period_correlations) >= 2:
            # Calculate stability metrics
            correlation_array = np.array(period_correlations)
            stability_metrics = {
                'correlation_variance': float(np.var(correlation_array, axis=0).mean()),
                'correlation_range': float(np.ptp(correlation_array, axis=0).mean()),
                'stability_score': float(1 - np.std(correlation_array, axis=0).mean()),  # Higher = more stable
                'periods_analyzed': len(period_correlations)
            }
        else:
            stability_metrics = {'error': 'Insufficient data for stability analysis'}
        
        return stability_metrics
    
    def _build_correlation_networks(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Build network representations of correlations"""
        
        # Get metric columns
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'project_type', 'brand_id', 'brand_name', 'brand_type',
            'date', 'week', 'month', 'quarter', 'day_of_week', 'is_weekend'
        ]]
        
        # Calculate correlation matrix
        corr_matrix = data[metric_cols].corr()
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (metrics)
        for metric in metric_cols:
            G.add_node(metric)
        
        # Add edges (correlations above threshold)
        threshold = self.config['network_threshold']
        for i, metric1 in enumerate(metric_cols):
            for j, metric2 in enumerate(metric_cols):
                if i < j:
                    correlation = corr_matrix.loc[metric1, metric2]
                    if abs(correlation) >= threshold:
                        G.add_edge(metric1, metric2, weight=abs(correlation), correlation=correlation)
        
        # Calculate network metrics
        network_metrics = {
            'nodes': len(G.nodes()),
            'edges': len(G.edges()),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'connected_components': nx.number_connected_components(G)
        }
        
        # Identify central metrics
        centrality_metrics = {
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G),
            'closeness_centrality': nx.closeness_centrality(G),
            'eigenvector_centrality': nx.eigenvector_centrality(G) if len(G.edges()) > 0 else {}
        }
        
        # Find communities/clusters in the network
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            community_info = {
                'num_communities': len(communities),
                'communities': [list(community) for community in communities],
                'modularity': nx.community.modularity(G, communities)
            }
        except:
            community_info = {'error': 'Community detection failed'}
        
        return {
            'network_metrics': network_metrics,
            'centrality_metrics': centrality_metrics,
            'community_structure': community_info,
            'edge_list': [(u, v, d['correlation']) for u, v, d in G.edges(data=True)]
        }
    
    def _perform_clustering_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis on metrics"""
        
        # Get metric columns
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'project_type', 'brand_id', 'brand_name', 'brand_type',
            'date', 'week', 'month', 'quarter', 'day_of_week', 'is_weekend'
        ]]
        
        if len(metric_cols) < 3:
            return {'error': 'Insufficient metrics for clustering analysis'}
        
        # Calculate correlation matrix
        corr_matrix = data[metric_cols].corr()
        
        # Convert correlation to distance matrix
        distance_matrix = 1 - abs(corr_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        # Determine optimal number of clusters
        n_clusters = min(max(2, len(metric_cols) // 2), 6)
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Organize clusters
        clusters = {}
        for i, metric in enumerate(metric_cols):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(metric)
        
        # Calculate cluster characteristics
        cluster_characteristics = {}
        for cluster_id, metrics in clusters.items():
            if len(metrics) > 1:
                cluster_corr_matrix = corr_matrix.loc[metrics, metrics]
                cluster_characteristics[cluster_id] = {
                    'metrics': metrics,
                    'size': len(metrics),
                    'internal_correlation': float(cluster_corr_matrix.values[np.triu_indices_from(cluster_corr_matrix.values, k=1)].mean()),
                    'cohesion': float(cluster_corr_matrix.values[np.triu_indices_from(cluster_corr_matrix.values, k=1)].std())
                }
        
        return {
            'clusters': clusters,
            'cluster_characteristics': cluster_characteristics,
            'n_clusters': n_clusters,
            'silhouette_score': self._calculate_silhouette_score(distance_matrix.values, cluster_labels)
        }
    
    def _perform_pca_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Principal Component Analysis"""
        
        # Get metric columns
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'project_type', 'brand_id', 'brand_name', 'brand_type',
            'date', 'week', 'month', 'quarter', 'day_of_week', 'is_weekend'
        ]]
        
        if len(metric_cols) < 3:
            return {'error': 'Insufficient metrics for PCA analysis'}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[metric_cols])
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Get component loadings
        components = pca.components_
        
        # Identify important features for each component
        component_features = {}
        for i, component in enumerate(components[:5]):  # Top 5 components
            feature_importance = abs(component)
            top_features = np.argsort(feature_importance)[-5:][::-1]  # Top 5 features
            
            component_features[f'PC{i+1}'] = {
                'explained_variance': float(explained_variance_ratio[i]),
                'top_features': [
                    {
                        'feature': metric_cols[idx],
                        'loading': float(component[idx]),
                        'importance': float(feature_importance[idx])
                    }
                    for idx in top_features
                ]
            }
        
        return {
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'component_features': component_features,
            'n_components_95_variance': int(np.argmax(cumulative_variance >= 0.95) + 1),
            'total_variance_explained': float(cumulative_variance[-1])
        }
    
    def _generate_correlation_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights from correlation analysis"""
        
        insights = []
        
        # Overall correlation insights
        if 'overall_correlations' in results:
            pearson_results = results['overall_correlations'].get('pearson', {})
            if 'summary_stats' in pearson_results:
                mean_corr = pearson_results['summary_stats']['mean_correlation']
                max_corr = pearson_results['summary_stats']['max_correlation']
                
                insights.append(f"Average correlation strength across metrics: {mean_corr:.3f}")
                insights.append(f"Strongest correlation observed: {max_corr:.3f}")
                
                if mean_corr > 0.5:
                    insights.append("High overall correlation suggests strong interdependencies between metrics")
                elif mean_corr < 0.2:
                    insights.append("Low overall correlation suggests metrics are largely independent")
        
        # Cross-brand insights
        if 'cross_brand_correlations' in results:
            brand_similarity = results['cross_brand_correlations'].get('brand_similarity', {})
            if 'average_similarity' in brand_similarity:
                avg_similarity = brand_similarity['average_similarity']
                insights.append(f"Average brand similarity: {avg_similarity:.3f}")
                
                if avg_similarity > 0.7:
                    insights.append("High brand similarity suggests unified optimization opportunities")
                elif avg_similarity < 0.3:
                    insights.append("Low brand similarity suggests need for brand-specific strategies")
        
        # Network insights
        if 'correlation_networks' in results:
            network_metrics = results['correlation_networks'].get('network_metrics', {})
            if 'density' in network_metrics:
                density = network_metrics['density']
                insights.append(f"Correlation network density: {density:.3f}")
                
                if density > 0.5:
                    insights.append("Dense correlation network indicates highly interconnected metrics")
                elif density < 0.2:
                    insights.append("Sparse correlation network suggests independent metric groups")
        
        # PCA insights
        if 'pca_analysis' in results:
            pca_results = results['pca_analysis']
            if 'n_components_95_variance' in pca_results:
                n_components = pca_results['n_components_95_variance']
                total_metrics = len(pca_results.get('explained_variance_ratio', []))
                
                insights.append(f"{n_components} components explain 95% of variance in {total_metrics} metrics")
                
                if n_components < total_metrics * 0.5:
                    insights.append("High dimensionality reduction potential - metrics show strong patterns")
        
        return insights
    
    def _generate_correlation_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from correlation analysis"""
        
        recommendations = []
        
        # Strong correlation recommendations
        if 'overall_correlations' in results:
            pearson_results = results['overall_correlations'].get('pearson', {})
            strong_correlations = pearson_results.get('strong_correlations', [])
            
            for corr in strong_correlations[:3]:  # Top 3 strong correlations
                if corr['correlation'] > 0.7:
                    recommendations.append({
                        'type': 'optimization_opportunity',
                        'priority': 'High',
                        'metrics': [corr['metric1'], corr['metric2']],
                        'correlation': corr['correlation'],
                        'action': f"Leverage strong positive correlation between {corr['metric1']} and {corr['metric2']}",
                        'implementation': 'Optimize both metrics simultaneously for maximum impact'
                    })
                elif corr['correlation'] < -0.7:
                    recommendations.append({
                        'type': 'trade_off_analysis',
                        'priority': 'Medium',
                        'metrics': [corr['metric1'], corr['metric2']],
                        'correlation': corr['correlation'],
                        'action': f"Analyze trade-off between {corr['metric1']} and {corr['metric2']}",
                        'implementation': 'Balance optimization to avoid negative impact on correlated metric'
                    })
        
        # Brand clustering recommendations
        if 'cross_brand_correlations' in results:
            brand_clusters = results['cross_brand_correlations'].get('brand_clusters', {})
            if 'clusters' in brand_clusters:
                for cluster_id, cluster_info in brand_clusters['cluster_characteristics'].items():
                    if cluster_info['size'] > 1:
                        recommendations.append({
                            'type': 'brand_grouping',
                            'priority': 'Medium',
                            'brands': cluster_info['brands'],
                            'action': f"Develop unified strategy for brand cluster {cluster_id}",
                            'implementation': 'Apply similar optimization approaches to brands in the same cluster'
                        })
        
        # Network centrality recommendations
        if 'correlation_networks' in results:
            centrality = results['correlation_networks'].get('centrality_metrics', {})
            degree_centrality = centrality.get('degree_centrality', {})
            
            if degree_centrality:
                # Find most central metric
                central_metric = max(degree_centrality.items(), key=lambda x: x[1])
                recommendations.append({
                    'type': 'focus_metric',
                    'priority': 'High',
                    'metric': central_metric[0],
                    'centrality_score': central_metric[1],
                    'action': f"Focus optimization efforts on {central_metric[0]} (highest network centrality)",
                    'implementation': 'Improvements to this metric will have cascading effects on connected metrics'
                })
        
        return recommendations
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update analyzer configuration"""
        self.config.update(new_config)
        self.logger.info(f"Correlation analyzer configuration updated: {new_config}")

# Factory function for creating correlation analyzer
def create_correlation_analyzer(projects: List[Project], brands: List[Brand], config: Dict[str, Any] = None) -> CorrelationAnalyzer:
    """Create Correlation Analyzer with specified configuration"""
    return CorrelationAnalyzer(projects, brands, config)

