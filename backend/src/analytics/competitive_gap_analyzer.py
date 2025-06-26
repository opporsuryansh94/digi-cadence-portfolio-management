"""
Competitive Gap Analyzer for Digi-Cadence Portfolio Management Platform
Provides comprehensive competitive gap analysis across brands, projects, and market segments
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from src.analytics.base_analyzer import BaseAnalyzer
from src.models.portfolio import Project, Brand, Metric, BrandMetric

class CompetitiveGapAnalyzer(BaseAnalyzer):
    """
    Advanced competitive gap analyzer for portfolio management
    Supports multi-dimensional competitive analysis with gap identification and strategic recommendations
    """
    
    def __init__(self, projects: List[Project], brands: List[Brand], config: Dict[str, Any] = None):
        super().__init__(projects, brands)
        
        # Competitive analysis configuration
        self.config = config or {
            'gap_threshold': 0.2,  # Minimum gap to be considered significant
            'benchmark_method': 'market_leader',  # market_leader, industry_average, top_quartile
            'competitive_dimensions': [
                'performance', 'efficiency', 'reach', 'engagement', 'conversion', 'cost_effectiveness'
            ],
            'gap_severity_levels': {
                'critical': 0.5,
                'major': 0.3,
                'moderate': 0.15,
                'minor': 0.05
            },
            'opportunity_scoring_weights': {
                'gap_size': 0.4,
                'market_potential': 0.3,
                'implementation_feasibility': 0.3
            },
            'competitor_similarity_threshold': 0.7,
            'time_decay_factor': 0.9,  # For weighting recent data more heavily
            'min_data_points': 15
        }
        
        # Analysis results storage
        self.gap_analysis_results = {}
        self.competitive_positioning = {}
        self.opportunity_matrix = {}
        self.benchmark_data = {}
        
        self.logger.info(f"Competitive Gap Analyzer initialized with {len(projects)} projects and {len(brands)} brands")
    
    def analyze(self, analysis_type: str = 'comprehensive', competitors: List[str] = None,
                time_period: str = 'last_6_months', metrics: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive competitive gap analysis
        
        Args:
            analysis_type: Type of analysis ('comprehensive', 'brand_specific', 'metric_specific', 'opportunity_mapping')
            competitors: List of competitor names/IDs for comparison
            time_period: Time period for analysis
            metrics: Specific metrics to analyze
            **kwargs: Additional analysis parameters
        
        Returns:
            Dictionary containing competitive gap analysis results
        """
        try:
            self.logger.info(f"Starting competitive gap analysis: {analysis_type}")
            
            # Prepare data for analysis
            analysis_data = self._prepare_competitive_data(time_period, metrics, competitors)
            
            if analysis_data.empty:
                raise ValueError("No data available for competitive gap analysis")
            
            results = {
                'analysis_type': analysis_type,
                'time_period': time_period,
                'competitors_analyzed': competitors or 'simulated_market',
                'metrics_analyzed': metrics or 'all_available',
                'data_points': len(analysis_data),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            if analysis_type == 'comprehensive':
                results.update(self._perform_comprehensive_gap_analysis(analysis_data, **kwargs))
            elif analysis_type == 'brand_specific':
                results.update(self._perform_brand_specific_analysis(analysis_data, **kwargs))
            elif analysis_type == 'metric_specific':
                results.update(self._perform_metric_specific_analysis(analysis_data, **kwargs))
            elif analysis_type == 'opportunity_mapping':
                results.update(self._perform_opportunity_mapping(analysis_data, **kwargs))
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Generate strategic insights and recommendations
            results['strategic_insights'] = self._generate_strategic_insights(results)
            results['gap_recommendations'] = self._generate_gap_recommendations(results)
            results['priority_matrix'] = self._create_priority_matrix(results)
            
            self.logger.info("Competitive gap analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Competitive gap analysis failed: {e}")
            raise
    
    def _prepare_competitive_data(self, time_period: str, metrics: List[str] = None, 
                                competitors: List[str] = None) -> pd.DataFrame:
        """Prepare competitive data for analysis"""
        
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
        
        # Default metrics if none specified
        if not metrics:
            metrics = [
                'engagement_rate', 'reach', 'impressions', 'click_through_rate',
                'conversion_rate', 'cost_per_acquisition', 'return_on_ad_spend',
                'brand_awareness', 'sentiment_score', 'share_of_voice',
                'website_traffic', 'social_mentions', 'video_views',
                'customer_acquisition_cost', 'lifetime_value', 'retention_rate'
            ]
        
        # Prepare our brand data
        our_data = []
        for project in self.projects:
            for brand in self.brands:
                current_date = start_date
                while current_date <= end_date:
                    # Simulate our brand performance data
                    base_performance = np.random.uniform(0.6, 0.9)  # Our brands perform reasonably well
                    seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * current_date.timetuple().tm_yday / 365)
                    
                    data_point = {
                        'entity_type': 'our_brand',
                        'entity_id': str(brand.id),
                        'entity_name': brand.name,
                        'project_id': str(project.id),
                        'project_name': project.name,
                        'brand_type': brand.brand_type or 'unknown',
                        'date': current_date,
                        'week': current_date.isocalendar()[1],
                        'month': current_date.month,
                        'quarter': (current_date.month - 1) // 3 + 1
                    }
                    
                    # Add metric values
                    for metric in metrics:
                        if metric == 'engagement_rate':
                            value = base_performance * 8 * seasonal_factor + np.random.normal(0, 1)
                        elif metric == 'reach':
                            value = base_performance * 50000 * seasonal_factor + np.random.normal(0, 5000)
                        elif metric == 'conversion_rate':
                            value = base_performance * 5 * seasonal_factor + np.random.normal(0, 0.5)
                        elif metric == 'cost_per_acquisition':
                            value = max(1, 30 / base_performance + np.random.normal(0, 5))
                        elif metric == 'return_on_ad_spend':
                            value = base_performance * 4 * seasonal_factor + np.random.normal(0, 0.5)
                        elif metric == 'brand_awareness':
                            value = base_performance * 70 * seasonal_factor + np.random.normal(0, 5)
                        elif metric == 'share_of_voice':
                            value = base_performance * 25 * seasonal_factor + np.random.normal(0, 3)
                        else:
                            value = base_performance * np.random.uniform(20, 100) + np.random.normal(0, 5)
                        
                        data_point[metric] = max(0, value)
                    
                    our_data.append(data_point)
                    current_date += timedelta(days=7)  # Weekly data
        
        # Generate competitive data (simulated market competitors)
        competitive_data = []
        
        # Define competitor profiles
        if not competitors:
            competitors = ['Market_Leader', 'Strong_Competitor_A', 'Strong_Competitor_B', 'Niche_Player', 'Emerging_Brand']
        
        competitor_profiles = {
            'Market_Leader': {'performance_factor': 1.2, 'consistency': 0.9},
            'Strong_Competitor_A': {'performance_factor': 1.1, 'consistency': 0.8},
            'Strong_Competitor_B': {'performance_factor': 1.05, 'consistency': 0.85},
            'Niche_Player': {'performance_factor': 0.9, 'consistency': 0.7},
            'Emerging_Brand': {'performance_factor': 0.8, 'consistency': 0.6}
        }
        
        for competitor in competitors:
            profile = competitor_profiles.get(competitor, {'performance_factor': 1.0, 'consistency': 0.75})
            
            current_date = start_date
            while current_date <= end_date:
                base_performance = np.random.uniform(0.7, 1.0) * profile['performance_factor']
                seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * current_date.timetuple().tm_yday / 365)
                noise_factor = profile['consistency']
                
                data_point = {
                    'entity_type': 'competitor',
                    'entity_id': competitor,
                    'entity_name': competitor,
                    'project_id': 'market',
                    'project_name': 'Market',
                    'brand_type': 'competitor',
                    'date': current_date,
                    'week': current_date.isocalendar()[1],
                    'month': current_date.month,
                    'quarter': (current_date.month - 1) // 3 + 1
                }
                
                # Add metric values for competitors
                for metric in metrics:
                    if metric == 'engagement_rate':
                        value = base_performance * 8 * seasonal_factor + np.random.normal(0, 1 * noise_factor)
                    elif metric == 'reach':
                        value = base_performance * 60000 * seasonal_factor + np.random.normal(0, 8000 * noise_factor)
                    elif metric == 'conversion_rate':
                        value = base_performance * 6 * seasonal_factor + np.random.normal(0, 0.7 * noise_factor)
                    elif metric == 'cost_per_acquisition':
                        value = max(1, 25 / base_performance + np.random.normal(0, 7 * noise_factor))
                    elif metric == 'return_on_ad_spend':
                        value = base_performance * 5 * seasonal_factor + np.random.normal(0, 0.7 * noise_factor)
                    elif metric == 'brand_awareness':
                        value = base_performance * 80 * seasonal_factor + np.random.normal(0, 7 * noise_factor)
                    elif metric == 'share_of_voice':
                        value = base_performance * 30 * seasonal_factor + np.random.normal(0, 5 * noise_factor)
                    else:
                        value = base_performance * np.random.uniform(25, 120) + np.random.normal(0, 8 * noise_factor)
                    
                    data_point[metric] = max(0, value)
                
                competitive_data.append(data_point)
                current_date += timedelta(days=7)
        
        # Combine our data and competitive data
        all_data = our_data + competitive_data
        df = pd.DataFrame(all_data)
        
        # Add derived metrics
        df = self._add_competitive_derived_metrics(df, metrics)
        
        return df
    
    def _add_competitive_derived_metrics(self, df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """Add derived metrics for competitive analysis"""
        
        # Calculate efficiency ratios
        if 'reach' in metrics and 'cost_per_acquisition' in metrics:
            df['reach_efficiency'] = df['reach'] / (df['cost_per_acquisition'] + 1)
        
        if 'engagement_rate' in metrics and 'reach' in metrics:
            df['engagement_reach_ratio'] = df['engagement_rate'] / (df['reach'] / 1000 + 1)
        
        if 'conversion_rate' in metrics and 'click_through_rate' in metrics:
            df['conversion_efficiency'] = df['conversion_rate'] / (df.get('click_through_rate', 1) + 0.1)
        
        # Calculate composite scores
        performance_metrics = ['engagement_rate', 'conversion_rate', 'return_on_ad_spend']
        available_performance_metrics = [m for m in performance_metrics if m in df.columns]
        
        if available_performance_metrics:
            # Normalize metrics to 0-1 scale for composite score
            scaler = MinMaxScaler()
            normalized_performance = scaler.fit_transform(df[available_performance_metrics])
            df['performance_composite_score'] = normalized_performance.mean(axis=1) * 100
        
        # Calculate market position indicators
        df['time_weight'] = df['date'].apply(
            lambda x: self.config['time_decay_factor'] ** ((datetime.utcnow() - x).days / 30)
        )
        
        return df
    
    def _perform_comprehensive_gap_analysis(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive competitive gap analysis"""
        
        results = {}
        
        # 1. Overall competitive positioning
        results['competitive_positioning'] = self._analyze_competitive_positioning(data)
        
        # 2. Metric-level gap analysis
        results['metric_gaps'] = self._analyze_metric_gaps(data)
        
        # 3. Brand-level competitive analysis
        results['brand_competitive_analysis'] = self._analyze_brand_competitiveness(data)
        
        # 4. Benchmark analysis
        results['benchmark_analysis'] = self._perform_benchmark_analysis(data)
        
        # 5. Competitive clustering
        results['competitive_clustering'] = self._perform_competitive_clustering(data)
        
        # 6. Gap trend analysis
        results['gap_trends'] = self._analyze_gap_trends(data)
        
        # 7. Opportunity identification
        results['opportunities'] = self._identify_competitive_opportunities(data)
        
        return results
    
    def _analyze_competitive_positioning(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall competitive positioning"""
        
        # Get metric columns
        metric_cols = [col for col in data.columns if col not in [
            'entity_type', 'entity_id', 'entity_name', 'project_id', 'project_name',
            'brand_type', 'date', 'week', 'month', 'quarter', 'time_weight'
        ]]
        
        # Calculate weighted averages for recent performance
        our_brands_data = data[data['entity_type'] == 'our_brand']
        competitors_data = data[data['entity_type'] == 'competitor']
        
        positioning_results = {}
        
        # Overall market position
        our_performance = {}
        market_performance = {}
        
        for metric in metric_cols:
            if metric in data.columns:
                # Weighted average for our brands
                our_weighted_avg = np.average(
                    our_brands_data[metric], 
                    weights=our_brands_data['time_weight']
                )
                
                # Weighted average for market/competitors
                market_weighted_avg = np.average(
                    competitors_data[metric], 
                    weights=competitors_data['time_weight']
                )
                
                our_performance[metric] = float(our_weighted_avg)
                market_performance[metric] = float(market_weighted_avg)
        
        # Calculate relative positioning
        relative_positioning = {}
        for metric in metric_cols:
            if metric in our_performance and metric in market_performance:
                market_value = market_performance[metric]
                our_value = our_performance[metric]
                
                if market_value > 0:
                    # For cost metrics (lower is better), invert the ratio
                    if 'cost' in metric.lower() or 'acquisition' in metric.lower():
                        relative_position = market_value / our_value if our_value > 0 else 0
                    else:
                        relative_position = our_value / market_value
                    
                    relative_positioning[metric] = {
                        'our_value': our_value,
                        'market_value': market_value,
                        'relative_position': float(relative_position),
                        'position_category': self._categorize_position(relative_position),
                        'gap': float(abs(our_value - market_value)),
                        'gap_percentage': float(abs(our_value - market_value) / market_value * 100) if market_value > 0 else 0
                    }
        
        # Calculate overall competitive score
        position_scores = [pos['relative_position'] for pos in relative_positioning.values()]
        overall_score = np.mean(position_scores) if position_scores else 0
        
        positioning_results = {
            'our_performance': our_performance,
            'market_performance': market_performance,
            'relative_positioning': relative_positioning,
            'overall_competitive_score': float(overall_score),
            'competitive_strength': self._categorize_competitive_strength(overall_score),
            'metrics_analyzed': len(relative_positioning)
        }
        
        return positioning_results
    
    def _categorize_position(self, relative_position: float) -> str:
        """Categorize competitive position"""
        if relative_position >= 1.2:
            return 'market_leader'
        elif relative_position >= 1.05:
            return 'above_market'
        elif relative_position >= 0.95:
            return 'at_market'
        elif relative_position >= 0.8:
            return 'below_market'
        else:
            return 'significantly_behind'
    
    def _categorize_competitive_strength(self, overall_score: float) -> str:
        """Categorize overall competitive strength"""
        if overall_score >= 1.15:
            return 'dominant'
        elif overall_score >= 1.05:
            return 'strong'
        elif overall_score >= 0.95:
            return 'competitive'
        elif overall_score >= 0.85:
            return 'challenged'
        else:
            return 'weak'
    
    def _analyze_metric_gaps(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze gaps at the metric level"""
        
        metric_cols = [col for col in data.columns if col not in [
            'entity_type', 'entity_id', 'entity_name', 'project_id', 'project_name',
            'brand_type', 'date', 'week', 'month', 'quarter', 'time_weight'
        ]]
        
        our_brands_data = data[data['entity_type'] == 'our_brand']
        competitors_data = data[data['entity_type'] == 'competitor']
        
        metric_gaps = {}
        
        for metric in metric_cols:
            if metric in data.columns:
                # Calculate benchmarks
                benchmarks = self._calculate_benchmarks(competitors_data, metric)
                
                # Our performance
                our_values = our_brands_data[metric]
                our_avg = np.average(our_values, weights=our_brands_data['time_weight'])
                
                # Calculate gaps against different benchmarks
                gaps = {}
                for benchmark_type, benchmark_value in benchmarks.items():
                    if benchmark_value > 0:
                        # For cost metrics, gap calculation is inverted
                        if 'cost' in metric.lower() or 'acquisition' in metric.lower():
                            gap = (our_avg - benchmark_value) / benchmark_value
                            gap_direction = 'higher_cost' if gap > 0 else 'lower_cost'
                        else:
                            gap = (benchmark_value - our_avg) / benchmark_value
                            gap_direction = 'behind' if gap > 0 else 'ahead'
                        
                        gaps[benchmark_type] = {
                            'gap_value': float(gap),
                            'gap_percentage': float(gap * 100),
                            'gap_direction': gap_direction,
                            'gap_severity': self._categorize_gap_severity(abs(gap)),
                            'benchmark_value': float(benchmark_value),
                            'our_value': float(our_avg)
                        }
                
                # Calculate improvement potential
                improvement_potential = self._calculate_improvement_potential(
                    our_avg, benchmarks, metric
                )
                
                metric_gaps[metric] = {
                    'gaps': gaps,
                    'improvement_potential': improvement_potential,
                    'priority_score': self._calculate_metric_priority_score(gaps, improvement_potential),
                    'volatility': float(our_values.std()),
                    'trend': self._calculate_metric_trend(our_brands_data, metric)
                }
        
        return metric_gaps
    
    def _calculate_benchmarks(self, competitors_data: pd.DataFrame, metric: str) -> Dict[str, float]:
        """Calculate different benchmark values for a metric"""
        
        metric_values = competitors_data[metric]
        weighted_values = np.average(metric_values, weights=competitors_data['time_weight'])
        
        benchmarks = {
            'market_average': float(weighted_values),
            'market_leader': float(metric_values.max()),
            'top_quartile': float(metric_values.quantile(0.75)),
            'median': float(metric_values.median())
        }
        
        return benchmarks
    
    def _categorize_gap_severity(self, gap_magnitude: float) -> str:
        """Categorize gap severity based on magnitude"""
        
        severity_levels = self.config['gap_severity_levels']
        
        if gap_magnitude >= severity_levels['critical']:
            return 'critical'
        elif gap_magnitude >= severity_levels['major']:
            return 'major'
        elif gap_magnitude >= severity_levels['moderate']:
            return 'moderate'
        elif gap_magnitude >= severity_levels['minor']:
            return 'minor'
        else:
            return 'negligible'
    
    def _calculate_improvement_potential(self, our_value: float, benchmarks: Dict[str, float], 
                                       metric: str) -> Dict[str, Any]:
        """Calculate improvement potential for a metric"""
        
        # Use market leader as the target for improvement potential
        target_value = benchmarks.get('market_leader', our_value)
        
        if target_value > our_value:
            if 'cost' in metric.lower() or 'acquisition' in metric.lower():
                # For cost metrics, improvement means reduction
                potential_improvement = (our_value - target_value) / our_value
                improvement_direction = 'reduce'
            else:
                # For performance metrics, improvement means increase
                potential_improvement = (target_value - our_value) / our_value
                improvement_direction = 'increase'
            
            return {
                'potential_improvement_percentage': float(potential_improvement * 100),
                'target_value': float(target_value),
                'current_value': float(our_value),
                'improvement_direction': improvement_direction,
                'feasibility': self._assess_improvement_feasibility(potential_improvement)
            }
        else:
            return {
                'potential_improvement_percentage': 0.0,
                'target_value': float(our_value),
                'current_value': float(our_value),
                'improvement_direction': 'maintain',
                'feasibility': 'already_leading'
            }
    
    def _assess_improvement_feasibility(self, improvement_percentage: float) -> str:
        """Assess feasibility of improvement based on percentage"""
        
        if improvement_percentage <= 0.1:  # 10%
            return 'high'
        elif improvement_percentage <= 0.25:  # 25%
            return 'medium'
        elif improvement_percentage <= 0.5:  # 50%
            return 'low'
        else:
            return 'very_low'
    
    def _calculate_metric_priority_score(self, gaps: Dict[str, Any], 
                                       improvement_potential: Dict[str, Any]) -> float:
        """Calculate priority score for metric improvement"""
        
        # Get gap against market leader
        market_leader_gap = gaps.get('market_leader', {})
        gap_magnitude = abs(market_leader_gap.get('gap_value', 0))
        
        # Get improvement potential
        improvement_pct = improvement_potential.get('potential_improvement_percentage', 0) / 100
        
        # Calculate weighted priority score
        weights = self.config['opportunity_scoring_weights']
        
        gap_score = min(gap_magnitude, 1.0)  # Cap at 1.0
        potential_score = min(improvement_pct, 1.0)  # Cap at 1.0
        feasibility_score = self._get_feasibility_score(improvement_potential.get('feasibility', 'low'))
        
        priority_score = (
            gap_score * weights['gap_size'] +
            potential_score * weights['market_potential'] +
            feasibility_score * weights['implementation_feasibility']
        )
        
        return float(priority_score)
    
    def _get_feasibility_score(self, feasibility: str) -> float:
        """Convert feasibility assessment to numeric score"""
        
        feasibility_scores = {
            'high': 1.0,
            'medium': 0.7,
            'low': 0.4,
            'very_low': 0.2,
            'already_leading': 0.1
        }
        
        return feasibility_scores.get(feasibility, 0.5)
    
    def _calculate_metric_trend(self, data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Calculate trend for a metric over time"""
        
        # Sort by date and calculate trend
        data_sorted = data.sort_values('date')
        
        if len(data_sorted) < 3:
            return {'trend': 'insufficient_data'}
        
        # Simple linear trend calculation
        x = np.arange(len(data_sorted))
        y = data_sorted[metric].values
        
        # Calculate correlation coefficient as trend indicator
        correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
        
        # Calculate percentage change from first to last
        first_value = y[0]
        last_value = y[-1]
        percentage_change = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0
        
        return {
            'trend_direction': 'improving' if correlation > 0.1 else 'declining' if correlation < -0.1 else 'stable',
            'trend_strength': abs(correlation),
            'percentage_change': float(percentage_change),
            'correlation_coefficient': float(correlation)
        }
    
    def _analyze_brand_competitiveness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze competitiveness at the brand level"""
        
        our_brands_data = data[data['entity_type'] == 'our_brand']
        competitors_data = data[data['entity_type'] == 'competitor']
        
        brand_analysis = {}
        
        # Analyze each of our brands
        for brand_name in our_brands_data['entity_name'].unique():
            brand_data = our_brands_data[our_brands_data['entity_name'] == brand_name]
            
            # Get metric columns
            metric_cols = [col for col in brand_data.columns if col not in [
                'entity_type', 'entity_id', 'entity_name', 'project_id', 'project_name',
                'brand_type', 'date', 'week', 'month', 'quarter', 'time_weight'
            ]]
            
            # Calculate brand performance vs market
            brand_performance = {}
            competitive_gaps = {}
            
            for metric in metric_cols:
                if metric in brand_data.columns:
                    # Brand performance
                    brand_avg = np.average(brand_data[metric], weights=brand_data['time_weight'])
                    
                    # Market benchmarks
                    benchmarks = self._calculate_benchmarks(competitors_data, metric)
                    
                    # Calculate gaps
                    market_leader_gap = (benchmarks['market_leader'] - brand_avg) / benchmarks['market_leader']
                    if 'cost' in metric.lower():
                        market_leader_gap = -market_leader_gap  # Invert for cost metrics
                    
                    brand_performance[metric] = float(brand_avg)
                    competitive_gaps[metric] = {
                        'gap_vs_leader': float(market_leader_gap),
                        'gap_vs_average': float((benchmarks['market_average'] - brand_avg) / benchmarks['market_average']),
                        'position_percentile': float(stats.percentileofscore(competitors_data[metric], brand_avg))
                    }
            
            # Calculate overall brand competitive score
            gap_scores = [abs(gap['gap_vs_leader']) for gap in competitive_gaps.values()]
            overall_gap = np.mean(gap_scores) if gap_scores else 0
            competitive_score = max(0, 1 - overall_gap)
            
            # Identify brand strengths and weaknesses
            strengths = []
            weaknesses = []
            
            for metric, gap_info in competitive_gaps.items():
                if gap_info['position_percentile'] >= 75:
                    strengths.append(metric)
                elif gap_info['position_percentile'] <= 25:
                    weaknesses.append(metric)
            
            brand_analysis[brand_name] = {
                'performance_metrics': brand_performance,
                'competitive_gaps': competitive_gaps,
                'overall_competitive_score': float(competitive_score),
                'competitive_position': self._categorize_competitive_strength(competitive_score),
                'strengths': strengths,
                'weaknesses': weaknesses,
                'improvement_priority': float(1 - competitive_score)  # Higher score = higher priority
            }
        
        return brand_analysis
    
    def _perform_benchmark_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform detailed benchmark analysis"""
        
        competitors_data = data[data['entity_type'] == 'competitor']
        our_brands_data = data[data['entity_type'] == 'our_brand']
        
        # Get metric columns
        metric_cols = [col for col in data.columns if col not in [
            'entity_type', 'entity_id', 'entity_name', 'project_id', 'project_name',
            'brand_type', 'date', 'week', 'month', 'quarter', 'time_weight'
        ]]
        
        benchmark_results = {}
        
        # Calculate comprehensive benchmarks
        for metric in metric_cols:
            if metric in data.columns:
                competitor_values = competitors_data[metric]
                our_values = our_brands_data[metric]
                
                # Statistical benchmarks
                benchmarks = {
                    'market_leader': float(competitor_values.max()),
                    'top_quartile': float(competitor_values.quantile(0.75)),
                    'median': float(competitor_values.median()),
                    'bottom_quartile': float(competitor_values.quantile(0.25)),
                    'market_average': float(competitor_values.mean()),
                    'market_std': float(competitor_values.std())
                }
                
                # Our performance
                our_performance = {
                    'current_value': float(our_values.mean()),
                    'best_performance': float(our_values.max()),
                    'worst_performance': float(our_values.min()),
                    'std_deviation': float(our_values.std())
                }
                
                # Gap analysis against each benchmark
                gap_analysis = {}
                for benchmark_name, benchmark_value in benchmarks.items():
                    if benchmark_name != 'market_std':
                        gap = (benchmark_value - our_performance['current_value']) / benchmark_value
                        if 'cost' in metric.lower():
                            gap = -gap  # Invert for cost metrics
                        
                        gap_analysis[benchmark_name] = {
                            'gap_percentage': float(gap * 100),
                            'gap_severity': self._categorize_gap_severity(abs(gap)),
                            'target_value': float(benchmark_value),
                            'improvement_needed': float(benchmark_value - our_performance['current_value'])
                        }
                
                benchmark_results[metric] = {
                    'benchmarks': benchmarks,
                    'our_performance': our_performance,
                    'gap_analysis': gap_analysis,
                    'competitive_position': self._calculate_competitive_position(
                        our_performance['current_value'], competitor_values
                    )
                }
        
        return benchmark_results
    
    def _calculate_competitive_position(self, our_value: float, competitor_values: pd.Series) -> Dict[str, Any]:
        """Calculate detailed competitive position"""
        
        percentile = stats.percentileofscore(competitor_values, our_value)
        rank = len(competitor_values[competitor_values > our_value]) + 1
        total_competitors = len(competitor_values)
        
        return {
            'percentile': float(percentile),
            'rank': int(rank),
            'total_competitors': int(total_competitors),
            'position_category': self._categorize_percentile_position(percentile)
        }
    
    def _categorize_percentile_position(self, percentile: float) -> str:
        """Categorize position based on percentile"""
        
        if percentile >= 90:
            return 'top_tier'
        elif percentile >= 75:
            return 'strong'
        elif percentile >= 50:
            return 'average'
        elif percentile >= 25:
            return 'below_average'
        else:
            return 'weak'
    
    def _perform_competitive_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis to identify competitive groups"""
        
        # Get metric columns
        metric_cols = [col for col in data.columns if col not in [
            'entity_type', 'entity_id', 'entity_name', 'project_id', 'project_name',
            'brand_type', 'date', 'week', 'month', 'quarter', 'time_weight'
        ]]
        
        # Aggregate data by entity
        entity_performance = data.groupby(['entity_name', 'entity_type'])[metric_cols].mean().reset_index()
        
        if len(entity_performance) < 3:
            return {'error': 'Insufficient entities for clustering analysis'}
        
        # Standardize metrics
        scaler = StandardScaler()
        scaled_metrics = scaler.fit_transform(entity_performance[metric_cols])
        
        # Determine optimal number of clusters
        max_clusters = min(5, len(entity_performance) - 1)
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_metrics)
            silhouette_avg = silhouette_score(scaled_metrics, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Use the number of clusters with highest silhouette score
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_metrics)
        
        # Organize results
        entity_performance['cluster'] = cluster_labels
        
        clusters = {}
        for cluster_id in range(optimal_clusters):
            cluster_entities = entity_performance[entity_performance['cluster'] == cluster_id]
            
            # Separate our brands from competitors
            our_brands = cluster_entities[cluster_entities['entity_type'] == 'our_brand']['entity_name'].tolist()
            competitors = cluster_entities[cluster_entities['entity_type'] == 'competitor']['entity_name'].tolist()
            
            # Calculate cluster characteristics
            cluster_metrics = cluster_entities[metric_cols].mean()
            
            clusters[cluster_id] = {
                'our_brands': our_brands,
                'competitors': competitors,
                'total_entities': len(cluster_entities),
                'cluster_characteristics': cluster_metrics.to_dict(),
                'cluster_performance_level': self._categorize_cluster_performance(cluster_metrics, entity_performance[metric_cols])
            }
        
        return {
            'clusters': clusters,
            'optimal_clusters': optimal_clusters,
            'silhouette_score': float(max(silhouette_scores)),
            'clustering_insights': self._generate_clustering_insights(clusters)
        }
    
    def _categorize_cluster_performance(self, cluster_metrics: pd.Series, all_metrics: pd.DataFrame) -> str:
        """Categorize cluster performance level"""
        
        # Calculate percentile of cluster average vs all entities
        cluster_avg = cluster_metrics.mean()
        all_avg = all_metrics.mean(axis=1)
        percentile = stats.percentileofscore(all_avg, cluster_avg)
        
        return self._categorize_percentile_position(percentile)
    
    def _generate_clustering_insights(self, clusters: Dict[str, Any]) -> List[str]:
        """Generate insights from clustering analysis"""
        
        insights = []
        
        # Find which cluster our brands are in
        our_brand_clusters = []
        for cluster_id, cluster_info in clusters.items():
            if cluster_info['our_brands']:
                our_brand_clusters.append((cluster_id, cluster_info))
        
        if our_brand_clusters:
            for cluster_id, cluster_info in our_brand_clusters:
                performance_level = cluster_info['cluster_performance_level']
                competitors_in_cluster = cluster_info['competitors']
                
                insights.append(f"Our brands in cluster {cluster_id} are positioned at {performance_level} level")
                
                if competitors_in_cluster:
                    insights.append(f"Direct competitors in same cluster: {', '.join(competitors_in_cluster[:3])}")
                
                if performance_level in ['weak', 'below_average']:
                    insights.append(f"Cluster {cluster_id} represents improvement opportunity for our brands")
                elif performance_level in ['top_tier', 'strong']:
                    insights.append(f"Cluster {cluster_id} represents our competitive strength area")
        
        return insights
    
    def _analyze_gap_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how competitive gaps are trending over time"""
        
        # Get metric columns
        metric_cols = [col for col in data.columns if col not in [
            'entity_type', 'entity_id', 'entity_name', 'project_id', 'project_name',
            'brand_type', 'date', 'week', 'month', 'quarter', 'time_weight'
        ]]
        
        # Calculate monthly gaps
        data['year_month'] = data['date'].dt.to_period('M')
        monthly_trends = {}
        
        for metric in metric_cols:
            if metric in data.columns:
                monthly_gaps = []
                
                for period in data['year_month'].unique():
                    period_data = data[data['year_month'] == period]
                    our_data = period_data[period_data['entity_type'] == 'our_brand']
                    competitor_data = period_data[period_data['entity_type'] == 'competitor']
                    
                    if len(our_data) > 0 and len(competitor_data) > 0:
                        our_avg = our_data[metric].mean()
                        market_leader = competitor_data[metric].max()
                        
                        gap = (market_leader - our_avg) / market_leader if market_leader > 0 else 0
                        if 'cost' in metric.lower():
                            gap = -gap
                        
                        monthly_gaps.append({
                            'period': str(period),
                            'gap': float(gap),
                            'our_value': float(our_avg),
                            'market_leader_value': float(market_leader)
                        })
                
                if len(monthly_gaps) >= 3:
                    # Calculate trend
                    gaps = [g['gap'] for g in monthly_gaps]
                    x = np.arange(len(gaps))
                    correlation = np.corrcoef(x, gaps)[0, 1] if len(gaps) > 1 else 0
                    
                    monthly_trends[metric] = {
                        'monthly_gaps': monthly_gaps,
                        'trend_direction': 'improving' if correlation < -0.1 else 'worsening' if correlation > 0.1 else 'stable',
                        'trend_strength': float(abs(correlation)),
                        'latest_gap': float(gaps[-1]),
                        'gap_change': float(gaps[-1] - gaps[0]) if len(gaps) > 1 else 0
                    }
        
        return monthly_trends
    
    def _identify_competitive_opportunities(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify competitive opportunities based on gap analysis"""
        
        # Get metric columns
        metric_cols = [col for col in data.columns if col not in [
            'entity_type', 'entity_id', 'entity_name', 'project_id', 'project_name',
            'brand_type', 'date', 'week', 'month', 'quarter', 'time_weight'
        ]]
        
        our_brands_data = data[data['entity_type'] == 'our_brand']
        competitors_data = data[data['entity_type'] == 'competitor']
        
        opportunities = []
        
        for metric in metric_cols:
            if metric in data.columns:
                # Calculate current gap
                our_avg = np.average(our_brands_data[metric], weights=our_brands_data['time_weight'])
                market_leader = competitors_data[metric].max()
                
                gap = (market_leader - our_avg) / market_leader if market_leader > 0 else 0
                if 'cost' in metric.lower():
                    gap = -gap
                
                # Only consider significant gaps as opportunities
                if gap > self.config['gap_threshold']:
                    # Calculate opportunity score
                    improvement_potential = gap
                    market_potential = self._estimate_market_potential(metric, gap)
                    implementation_feasibility = self._estimate_implementation_feasibility(metric, gap)
                    
                    weights = self.config['opportunity_scoring_weights']
                    opportunity_score = (
                        improvement_potential * weights['gap_size'] +
                        market_potential * weights['market_potential'] +
                        implementation_feasibility * weights['implementation_feasibility']
                    )
                    
                    opportunities.append({
                        'metric': metric,
                        'gap_percentage': float(gap * 100),
                        'current_value': float(our_avg),
                        'target_value': float(market_leader),
                        'improvement_potential': float(improvement_potential),
                        'market_potential': float(market_potential),
                        'implementation_feasibility': float(implementation_feasibility),
                        'opportunity_score': float(opportunity_score),
                        'priority': 'high' if opportunity_score > 0.7 else 'medium' if opportunity_score > 0.4 else 'low'
                    })
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        return {
            'opportunities': opportunities,
            'high_priority_count': len([o for o in opportunities if o['priority'] == 'high']),
            'total_opportunities': len(opportunities)
        }
    
    def _estimate_market_potential(self, metric: str, gap: float) -> float:
        """Estimate market potential for improving a metric"""
        
        # Simple heuristic based on metric type and gap size
        high_impact_metrics = ['conversion_rate', 'return_on_ad_spend', 'engagement_rate', 'brand_awareness']
        medium_impact_metrics = ['reach', 'click_through_rate', 'share_of_voice']
        
        base_potential = 0.5  # Default
        
        if metric in high_impact_metrics:
            base_potential = 0.8
        elif metric in medium_impact_metrics:
            base_potential = 0.6
        
        # Adjust based on gap size
        gap_multiplier = min(1.0, gap * 2)  # Larger gaps have more potential
        
        return base_potential * gap_multiplier
    
    def _estimate_implementation_feasibility(self, metric: str, gap: float) -> float:
        """Estimate implementation feasibility for improving a metric"""
        
        # Metrics that are typically easier to improve
        easy_metrics = ['reach', 'impressions', 'website_traffic']
        medium_metrics = ['engagement_rate', 'click_through_rate', 'social_mentions']
        hard_metrics = ['conversion_rate', 'brand_awareness', 'return_on_ad_spend']
        
        base_feasibility = 0.5  # Default
        
        if metric in easy_metrics:
            base_feasibility = 0.8
        elif metric in medium_metrics:
            base_feasibility = 0.6
        elif metric in hard_metrics:
            base_feasibility = 0.4
        
        # Larger gaps are typically harder to close
        gap_penalty = min(0.3, gap * 0.5)
        
        return max(0.1, base_feasibility - gap_penalty)
    
    def _generate_strategic_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate strategic insights from competitive gap analysis"""
        
        insights = []
        
        # Overall competitive position insights
        positioning = results.get('competitive_positioning', {})
        if 'overall_competitive_score' in positioning:
            score = positioning['overall_competitive_score']
            strength = positioning['competitive_strength']
            
            insights.append(f"Overall competitive position: {strength} (score: {score:.2f})")
            
            if strength in ['weak', 'challenged']:
                insights.append("Significant competitive gaps identified - strategic intervention required")
            elif strength in ['dominant', 'strong']:
                insights.append("Strong competitive position - focus on maintaining leadership")
        
        # Metric-specific insights
        metric_gaps = results.get('metric_gaps', {})
        critical_gaps = []
        for metric, gap_info in metric_gaps.items():
            priority_score = gap_info.get('priority_score', 0)
            if priority_score > 0.7:
                critical_gaps.append(metric)
        
        if critical_gaps:
            insights.append(f"Critical improvement areas: {', '.join(critical_gaps[:3])}")
        
        # Opportunity insights
        opportunities = results.get('opportunities', {})
        high_priority_count = opportunities.get('high_priority_count', 0)
        if high_priority_count > 0:
            insights.append(f"{high_priority_count} high-priority competitive opportunities identified")
        
        # Clustering insights
        clustering = results.get('competitive_clustering', {})
        if 'clustering_insights' in clustering:
            insights.extend(clustering['clustering_insights'][:2])  # Add top 2 clustering insights
        
        return insights
    
    def _generate_gap_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from gap analysis"""
        
        recommendations = []
        
        # High-priority metric improvements
        opportunities = results.get('opportunities', {}).get('opportunities', [])
        for opportunity in opportunities[:5]:  # Top 5 opportunities
            if opportunity['priority'] == 'high':
                recommendations.append({
                    'type': 'metric_improvement',
                    'priority': 'High',
                    'metric': opportunity['metric'],
                    'gap_percentage': opportunity['gap_percentage'],
                    'action': f"Close {opportunity['gap_percentage']:.1f}% gap in {opportunity['metric']}",
                    'target_value': opportunity['target_value'],
                    'expected_impact': 'Significant competitive advantage',
                    'implementation_feasibility': opportunity['implementation_feasibility']
                })
        
        # Brand-specific recommendations
        brand_analysis = results.get('brand_competitive_analysis', {})
        for brand_name, brand_info in brand_analysis.items():
            if brand_info.get('improvement_priority', 0) > 0.7:
                weaknesses = brand_info.get('weaknesses', [])
                if weaknesses:
                    recommendations.append({
                        'type': 'brand_improvement',
                        'priority': 'Medium',
                        'brand': brand_name,
                        'action': f"Address weaknesses in {', '.join(weaknesses[:2])} for {brand_name}",
                        'focus_areas': weaknesses,
                        'expected_impact': 'Improved brand competitiveness'
                    })
        
        # Strategic positioning recommendations
        positioning = results.get('competitive_positioning', {})
        competitive_strength = positioning.get('competitive_strength', '')
        
        if competitive_strength in ['weak', 'challenged']:
            recommendations.append({
                'type': 'strategic_repositioning',
                'priority': 'High',
                'action': 'Develop comprehensive competitive improvement strategy',
                'focus': 'Address fundamental competitive disadvantages',
                'expected_impact': 'Transform competitive position'
            })
        elif competitive_strength == 'competitive':
            recommendations.append({
                'type': 'competitive_enhancement',
                'priority': 'Medium',
                'action': 'Identify and exploit competitive advantages',
                'focus': 'Strengthen position in key metrics',
                'expected_impact': 'Achieve competitive leadership'
            })
        
        return recommendations
    
    def _create_priority_matrix(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create priority matrix for competitive improvements"""
        
        opportunities = results.get('opportunities', {}).get('opportunities', [])
        
        # Categorize opportunities by impact and feasibility
        matrix = {
            'high_impact_high_feasibility': [],
            'high_impact_low_feasibility': [],
            'low_impact_high_feasibility': [],
            'low_impact_low_feasibility': []
        }
        
        for opportunity in opportunities:
            impact = opportunity.get('market_potential', 0)
            feasibility = opportunity.get('implementation_feasibility', 0)
            
            if impact > 0.6 and feasibility > 0.6:
                matrix['high_impact_high_feasibility'].append(opportunity)
            elif impact > 0.6 and feasibility <= 0.6:
                matrix['high_impact_low_feasibility'].append(opportunity)
            elif impact <= 0.6 and feasibility > 0.6:
                matrix['low_impact_high_feasibility'].append(opportunity)
            else:
                matrix['low_impact_low_feasibility'].append(opportunity)
        
        # Add recommendations for each quadrant
        matrix_recommendations = {
            'high_impact_high_feasibility': 'Immediate action - quick wins with high impact',
            'high_impact_low_feasibility': 'Strategic investment - high impact but requires significant resources',
            'low_impact_high_feasibility': 'Fill-in projects - easy to implement but limited impact',
            'low_impact_low_feasibility': 'Avoid - low impact and difficult to implement'
        }
        
        return {
            'priority_matrix': matrix,
            'recommendations': matrix_recommendations,
            'immediate_action_count': len(matrix['high_impact_high_feasibility']),
            'strategic_investment_count': len(matrix['high_impact_low_feasibility'])
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update analyzer configuration"""
        self.config.update(new_config)
        self.logger.info(f"Competitive Gap analyzer configuration updated: {new_config}")

# Factory function for creating competitive gap analyzer
def create_competitive_gap_analyzer(projects: List[Project], brands: List[Brand], config: Dict[str, Any] = None) -> CompetitiveGapAnalyzer:
    """Create Competitive Gap Analyzer with specified configuration"""
    return CompetitiveGapAnalyzer(projects, brands, config)

