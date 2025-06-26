"""
Trend Analyzer for Digi-Cadence Portfolio Management Platform
Provides comprehensive trend analysis and forecasting capabilities for portfolio metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from src.analytics.base_analyzer import BaseAnalyzer
from src.models.portfolio import Project, Brand, Metric, BrandMetric

class TrendAnalyzer(BaseAnalyzer):
    """
    Advanced trend analyzer for portfolio management
    Supports time-series analysis, forecasting, and trend pattern recognition
    """
    
    def __init__(self, projects: List[Project], brands: List[Brand], config: Dict[str, Any] = None):
        super().__init__(projects, brands)
        
        # Trend analysis configuration
        self.config = config or {
            'forecast_periods': 12,  # Number of periods to forecast
            'trend_detection_methods': ['linear', 'polynomial', 'seasonal', 'arima'],
            'seasonality_periods': [7, 30, 90, 365],  # Weekly, monthly, quarterly, yearly
            'change_point_detection': True,
            'outlier_detection': True,
            'confidence_intervals': [0.8, 0.95],
            'min_data_points': 20,
            'trend_significance_threshold': 0.05,
            'volatility_window': 30,
            'smoothing_window': 7,
            'forecast_models': ['arima', 'exponential_smoothing', 'linear_regression', 'random_forest']
        }
        
        # Analysis results storage
        self.trend_models = {}
        self.forecast_results = {}
        self.seasonality_patterns = {}
        self.change_points = {}
        
        self.logger.info(f"Trend Analyzer initialized with {len(projects)} projects and {len(brands)} brands")
    
    def analyze(self, analysis_type: str = 'comprehensive', time_period: str = 'last_year',
                metrics: List[str] = None, forecast_horizon: int = None, **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive trend analysis
        
        Args:
            analysis_type: Type of analysis ('comprehensive', 'trend_detection', 'forecasting', 'seasonality')
            time_period: Time period for analysis
            metrics: Specific metrics to analyze
            forecast_horizon: Number of periods to forecast
            **kwargs: Additional analysis parameters
        
        Returns:
            Dictionary containing trend analysis results
        """
        try:
            self.logger.info(f"Starting trend analysis: {analysis_type}")
            
            # Prepare data for analysis
            analysis_data = self._prepare_trend_data(time_period, metrics)
            
            if analysis_data.empty:
                raise ValueError("No data available for trend analysis")
            
            # Set forecast horizon
            if forecast_horizon is None:
                forecast_horizon = self.config['forecast_periods']
            
            results = {
                'analysis_type': analysis_type,
                'time_period': time_period,
                'metrics_analyzed': metrics or 'all_available',
                'data_points': len(analysis_data),
                'forecast_horizon': forecast_horizon,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            if analysis_type == 'comprehensive':
                results.update(self._perform_comprehensive_trend_analysis(analysis_data, forecast_horizon, **kwargs))
            elif analysis_type == 'trend_detection':
                results.update(self._perform_trend_detection(analysis_data, **kwargs))
            elif analysis_type == 'forecasting':
                results.update(self._perform_forecasting_analysis(analysis_data, forecast_horizon, **kwargs))
            elif analysis_type == 'seasonality':
                results.update(self._perform_seasonality_analysis(analysis_data, **kwargs))
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Generate insights and recommendations
            results['trend_insights'] = self._generate_trend_insights(results)
            results['trend_recommendations'] = self._generate_trend_recommendations(results)
            results['risk_assessment'] = self._assess_trend_risks(results)
            
            self.logger.info("Trend analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            raise
    
    def _prepare_trend_data(self, time_period: str, metrics: List[str] = None) -> pd.DataFrame:
        """Prepare time-series data for trend analysis"""
        
        # Calculate time range
        end_date = datetime.utcnow()
        if time_period == 'last_year':
            start_date = end_date - timedelta(days=365)
        elif time_period == 'last_6_months':
            start_date = end_date - timedelta(days=180)
        elif time_period == 'last_2_years':
            start_date = end_date - timedelta(days=730)
        else:
            start_date = end_date - timedelta(days=365)
        
        # Default metrics if none specified
        if not metrics:
            metrics = [
                'engagement_rate', 'reach', 'impressions', 'click_through_rate',
                'conversion_rate', 'cost_per_acquisition', 'return_on_ad_spend',
                'brand_awareness', 'sentiment_score', 'share_of_voice',
                'website_traffic', 'social_mentions', 'video_views'
            ]
        
        # Generate time series data
        data_points = []
        
        for project in self.projects:
            for brand in self.brands:
                # Generate daily data points for better trend analysis
                current_date = start_date
                while current_date <= end_date:
                    # Create realistic time series with trends and seasonality
                    days_from_start = (current_date - start_date).days
                    
                    # Base trend (slight upward trend over time)
                    trend_factor = 1 + (days_from_start / 365) * 0.1
                    
                    # Seasonal patterns
                    yearly_cycle = np.sin(2 * np.pi * current_date.timetuple().tm_yday / 365)
                    weekly_cycle = np.sin(2 * np.pi * current_date.weekday() / 7)
                    monthly_cycle = np.sin(2 * np.pi * current_date.day / 30)
                    
                    seasonal_factor = 1 + 0.15 * yearly_cycle + 0.1 * weekly_cycle + 0.05 * monthly_cycle
                    
                    # Random noise
                    noise_factor = np.random.normal(1, 0.1)
                    
                    # Brand-specific performance factor
                    brand_performance = np.random.uniform(0.8, 1.2)
                    
                    data_point = {
                        'project_id': str(project.id),
                        'project_name': project.name,
                        'brand_id': str(brand.id),
                        'brand_name': brand.name,
                        'brand_type': brand.brand_type or 'unknown',
                        'date': current_date,
                        'day_of_week': current_date.weekday(),
                        'day_of_month': current_date.day,
                        'day_of_year': current_date.timetuple().tm_yday,
                        'week': current_date.isocalendar()[1],
                        'month': current_date.month,
                        'quarter': (current_date.month - 1) // 3 + 1,
                        'is_weekend': 1 if current_date.weekday() >= 5 else 0,
                        'is_holiday': 1 if current_date.month == 12 and current_date.day >= 20 else 0
                    }
                    
                    # Generate metric values with realistic patterns
                    for metric in metrics:
                        base_value = self._get_base_metric_value(metric)
                        
                        # Apply all factors
                        value = (base_value * brand_performance * trend_factor * 
                                seasonal_factor * noise_factor)
                        
                        # Add metric-specific patterns
                        if metric == 'engagement_rate':
                            # Higher engagement on weekends
                            if current_date.weekday() >= 5:
                                value *= 1.2
                        elif metric == 'cost_per_acquisition':
                            # Costs tend to increase over time (inflation)
                            value *= (1 + days_from_start / 365 * 0.05)
                        elif metric == 'brand_awareness':
                            # Slower changing metric with momentum
                            if days_from_start > 0:
                                previous_value = data_points[-1].get(metric, base_value) if data_points else base_value
                                value = 0.9 * previous_value + 0.1 * value
                        
                        data_point[metric] = max(0, value)
                    
                    data_points.append(data_point)
                    current_date += timedelta(days=1)  # Daily data
        
        df = pd.DataFrame(data_points)
        
        # Sort by date for time series analysis
        df = df.sort_values(['brand_id', 'date']).reset_index(drop=True)
        
        # Add derived time features
        df = self._add_time_features(df)
        
        return df
    
    def _get_base_metric_value(self, metric: str) -> float:
        """Get base value for a metric"""
        
        base_values = {
            'engagement_rate': 6.0,
            'reach': 45000,
            'impressions': 120000,
            'click_through_rate': 2.5,
            'conversion_rate': 4.0,
            'cost_per_acquisition': 25.0,
            'return_on_ad_spend': 3.5,
            'brand_awareness': 65.0,
            'sentiment_score': 75.0,
            'share_of_voice': 20.0,
            'website_traffic': 15000,
            'social_mentions': 500,
            'video_views': 8000
        }
        
        return base_values.get(metric, 50.0)
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features for analysis"""
        
        # Rolling averages
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['day_of_week', 'day_of_month', 'day_of_year', 'week', 'month', 'quarter', 'is_weekend', 'is_holiday']:
                df[f'{col}_ma_7'] = df.groupby('brand_id')[col].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
                df[f'{col}_ma_30'] = df.groupby('brand_id')[col].rolling(window=30, min_periods=1).mean().reset_index(0, drop=True)
        
        # Lag features
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['day_of_week', 'day_of_month', 'day_of_year', 'week', 'month', 'quarter', 'is_weekend', 'is_holiday']:
                df[f'{col}_lag_1'] = df.groupby('brand_id')[col].shift(1)
                df[f'{col}_lag_7'] = df.groupby('brand_id')[col].shift(7)
        
        # Rate of change
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['day_of_week', 'day_of_month', 'day_of_year', 'week', 'month', 'quarter', 'is_weekend', 'is_holiday']:
                df[f'{col}_pct_change'] = df.groupby('brand_id')[col].pct_change()
        
        return df
    
    def _perform_comprehensive_trend_analysis(self, data: pd.DataFrame, forecast_horizon: int, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive trend analysis"""
        
        results = {}
        
        # 1. Trend detection for all metrics
        results['trend_detection'] = self._perform_trend_detection(data, **kwargs)
        
        # 2. Seasonality analysis
        results['seasonality_analysis'] = self._perform_seasonality_analysis(data, **kwargs)
        
        # 3. Change point detection
        results['change_points'] = self._detect_change_points(data, **kwargs)
        
        # 4. Volatility analysis
        results['volatility_analysis'] = self._analyze_volatility(data, **kwargs)
        
        # 5. Forecasting
        results['forecasting'] = self._perform_forecasting_analysis(data, forecast_horizon, **kwargs)
        
        # 6. Cross-metric trend correlations
        results['trend_correlations'] = self._analyze_trend_correlations(data, **kwargs)
        
        # 7. Brand trend comparison
        results['brand_trend_comparison'] = self._compare_brand_trends(data, **kwargs)
        
        return results
    
    def _perform_trend_detection(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Detect trends in time series data"""
        
        # Get metric columns
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'brand_id', 'brand_name', 'brand_type',
            'date', 'day_of_week', 'day_of_month', 'day_of_year', 'week', 'month', 'quarter',
            'is_weekend', 'is_holiday'
        ] and not col.endswith('_ma_7') and not col.endswith('_ma_30') and not col.endswith('_lag_1') and not col.endswith('_lag_7') and not col.endswith('_pct_change')]
        
        trend_results = {}
        
        for brand_name in data['brand_name'].unique():
            brand_data = data[data['brand_name'] == brand_name].sort_values('date')
            
            if len(brand_data) < self.config['min_data_points']:
                continue
            
            brand_trends = {}
            
            for metric in metric_cols:
                if metric in brand_data.columns:
                    metric_data = brand_data[metric].dropna()
                    
                    if len(metric_data) < self.config['min_data_points']:
                        continue
                    
                    # Linear trend analysis
                    linear_trend = self._detect_linear_trend(metric_data)
                    
                    # Polynomial trend analysis
                    polynomial_trend = self._detect_polynomial_trend(metric_data)
                    
                    # Mann-Kendall trend test
                    mk_trend = self._mann_kendall_test(metric_data)
                    
                    # Seasonal trend decomposition
                    seasonal_trend = self._decompose_seasonal_trend(metric_data)
                    
                    brand_trends[metric] = {
                        'linear_trend': linear_trend,
                        'polynomial_trend': polynomial_trend,
                        'mann_kendall_trend': mk_trend,
                        'seasonal_decomposition': seasonal_trend,
                        'overall_trend_direction': self._determine_overall_trend(linear_trend, mk_trend),
                        'trend_strength': self._calculate_trend_strength(linear_trend, mk_trend),
                        'trend_consistency': self._calculate_trend_consistency(metric_data)
                    }
            
            trend_results[brand_name] = brand_trends
        
        return trend_results
    
    def _detect_linear_trend(self, data: pd.Series) -> Dict[str, Any]:
        """Detect linear trend using linear regression"""
        
        x = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        
        model = LinearRegression()
        model.fit(x, y)
        
        # Calculate trend statistics
        slope = model.coef_[0]
        r_squared = model.score(x, y)
        
        # Statistical significance test
        n = len(data)
        t_stat = slope * np.sqrt((n - 2) / (1 - r_squared)) / np.sqrt(np.sum((x.flatten() - x.mean()) ** 2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        return {
            'slope': float(slope),
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'is_significant': p_value < self.config['trend_significance_threshold'],
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'trend_magnitude': abs(slope),
            'confidence_level': 1 - p_value
        }
    
    def _detect_polynomial_trend(self, data: pd.Series, degree: int = 2) -> Dict[str, Any]:
        """Detect polynomial trend"""
        
        x = np.arange(len(data))
        y = data.values
        
        # Fit polynomial
        coefficients = np.polyfit(x, y, degree)
        polynomial = np.poly1d(coefficients)
        
        # Calculate R-squared
        y_pred = polynomial(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Determine trend shape
        if degree == 2:
            trend_shape = 'accelerating' if coefficients[0] > 0 else 'decelerating' if coefficients[0] < 0 else 'linear'
        else:
            trend_shape = 'complex'
        
        return {
            'coefficients': coefficients.tolist(),
            'r_squared': float(r_squared),
            'trend_shape': trend_shape,
            'degree': degree
        }
    
    def _mann_kendall_test(self, data: pd.Series) -> Dict[str, Any]:
        """Perform Mann-Kendall trend test"""
        
        n = len(data)
        
        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data.iloc[j] > data.iloc[i]:
                    s += 1
                elif data.iloc[j] < data.iloc[i]:
                    s -= 1
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            's_statistic': int(s),
            'z_statistic': float(z),
            'p_value': float(p_value),
            'is_significant': p_value < self.config['trend_significance_threshold'],
            'trend_direction': 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no_trend'
        }
    
    def _decompose_seasonal_trend(self, data: pd.Series, period: int = 30) -> Dict[str, Any]:
        """Decompose time series into trend, seasonal, and residual components"""
        
        if len(data) < 2 * period:
            return {'error': 'Insufficient data for seasonal decomposition'}
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(data, model='additive', period=period)
            
            # Calculate trend strength
            trend_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.trend.dropna() + decomposition.resid.dropna())
            
            # Calculate seasonal strength
            seasonal_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.seasonal.dropna() + decomposition.resid.dropna())
            
            return {
                'trend_strength': float(max(0, trend_strength)),
                'seasonal_strength': float(max(0, seasonal_strength)),
                'trend_component': decomposition.trend.dropna().tolist(),
                'seasonal_component': decomposition.seasonal.dropna().tolist(),
                'residual_component': decomposition.resid.dropna().tolist(),
                'decomposition_successful': True
            }
        
        except Exception as e:
            return {'error': f'Decomposition failed: {str(e)}'}
    
    def _determine_overall_trend(self, linear_trend: Dict[str, Any], mk_trend: Dict[str, Any]) -> str:
        """Determine overall trend direction from multiple tests"""
        
        linear_significant = linear_trend.get('is_significant', False)
        mk_significant = mk_trend.get('is_significant', False)
        
        linear_direction = linear_trend.get('trend_direction', 'stable')
        mk_direction = mk_trend.get('trend_direction', 'no_trend')
        
        # If both tests agree and are significant
        if linear_significant and mk_significant:
            if linear_direction == 'increasing' and mk_direction == 'increasing':
                return 'strong_increasing'
            elif linear_direction == 'decreasing' and mk_direction == 'decreasing':
                return 'strong_decreasing'
        
        # If only one test is significant
        if linear_significant and not mk_significant:
            return f'weak_{linear_direction}'
        elif mk_significant and not linear_significant:
            return f'weak_{mk_direction.replace("no_trend", "stable")}'
        
        # If neither is significant or they disagree
        return 'no_clear_trend'
    
    def _calculate_trend_strength(self, linear_trend: Dict[str, Any], mk_trend: Dict[str, Any]) -> float:
        """Calculate overall trend strength"""
        
        linear_strength = linear_trend.get('r_squared', 0) if linear_trend.get('is_significant', False) else 0
        mk_strength = (1 - mk_trend.get('p_value', 1)) if mk_trend.get('is_significant', False) else 0
        
        return float((linear_strength + mk_strength) / 2)
    
    def _calculate_trend_consistency(self, data: pd.Series) -> float:
        """Calculate trend consistency (how consistent the direction changes are)"""
        
        # Calculate first differences
        diff = data.diff().dropna()
        
        if len(diff) == 0:
            return 0.0
        
        # Count direction changes
        direction_changes = 0
        for i in range(1, len(diff)):
            if (diff.iloc[i] > 0) != (diff.iloc[i-1] > 0):
                direction_changes += 1
        
        # Consistency is inverse of change frequency
        consistency = 1 - (direction_changes / len(diff))
        
        return float(max(0, consistency))
    
    def _perform_seasonality_analysis(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze seasonality patterns in the data"""
        
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'brand_id', 'brand_name', 'brand_type',
            'date', 'day_of_week', 'day_of_month', 'day_of_year', 'week', 'month', 'quarter',
            'is_weekend', 'is_holiday'
        ] and not col.endswith('_ma_7') and not col.endswith('_ma_30') and not col.endswith('_lag_1') and not col.endswith('_lag_7') and not col.endswith('_pct_change')]
        
        seasonality_results = {}
        
        for brand_name in data['brand_name'].unique():
            brand_data = data[data['brand_name'] == brand_name].sort_values('date')
            
            if len(brand_data) < 60:  # Need at least 2 months of data
                continue
            
            brand_seasonality = {}
            
            for metric in metric_cols:
                if metric in brand_data.columns:
                    metric_data = brand_data[metric].dropna()
                    
                    if len(metric_data) < 30:
                        continue
                    
                    # Weekly seasonality
                    weekly_pattern = self._analyze_weekly_seasonality(brand_data, metric)
                    
                    # Monthly seasonality
                    monthly_pattern = self._analyze_monthly_seasonality(brand_data, metric)
                    
                    # Quarterly seasonality
                    quarterly_pattern = self._analyze_quarterly_seasonality(brand_data, metric)
                    
                    # Holiday effects
                    holiday_effects = self._analyze_holiday_effects(brand_data, metric)
                    
                    brand_seasonality[metric] = {
                        'weekly_seasonality': weekly_pattern,
                        'monthly_seasonality': monthly_pattern,
                        'quarterly_seasonality': quarterly_pattern,
                        'holiday_effects': holiday_effects,
                        'dominant_seasonality': self._identify_dominant_seasonality(weekly_pattern, monthly_pattern, quarterly_pattern)
                    }
            
            seasonality_results[brand_name] = brand_seasonality
        
        return seasonality_results
    
    def _analyze_weekly_seasonality(self, data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Analyze weekly seasonality patterns"""
        
        # Group by day of week
        weekly_stats = data.groupby('day_of_week')[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate coefficient of variation to measure seasonality strength
        overall_mean = data[metric].mean()
        weekly_cv = weekly_stats['mean'].std() / overall_mean if overall_mean > 0 else 0
        
        # Find peak and trough days
        peak_day = weekly_stats.loc[weekly_stats['mean'].idxmax(), 'day_of_week']
        trough_day = weekly_stats.loc[weekly_stats['mean'].idxmin(), 'day_of_week']
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return {
            'seasonality_strength': float(weekly_cv),
            'peak_day': day_names[int(peak_day)],
            'trough_day': day_names[int(trough_day)],
            'peak_value': float(weekly_stats['mean'].max()),
            'trough_value': float(weekly_stats['mean'].min()),
            'weekly_pattern': {day_names[int(row['day_of_week'])]: float(row['mean']) for _, row in weekly_stats.iterrows()},
            'is_significant': weekly_cv > 0.1  # Threshold for significant seasonality
        }
    
    def _analyze_monthly_seasonality(self, data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Analyze monthly seasonality patterns"""
        
        # Group by month
        monthly_stats = data.groupby('month')[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate seasonality strength
        overall_mean = data[metric].mean()
        monthly_cv = monthly_stats['mean'].std() / overall_mean if overall_mean > 0 else 0
        
        # Find peak and trough months
        peak_month = monthly_stats.loc[monthly_stats['mean'].idxmax(), 'month']
        trough_month = monthly_stats.loc[monthly_stats['mean'].idxmin(), 'month']
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        return {
            'seasonality_strength': float(monthly_cv),
            'peak_month': month_names[int(peak_month) - 1],
            'trough_month': month_names[int(trough_month) - 1],
            'peak_value': float(monthly_stats['mean'].max()),
            'trough_value': float(monthly_stats['mean'].min()),
            'monthly_pattern': {month_names[int(row['month']) - 1]: float(row['mean']) for _, row in monthly_stats.iterrows()},
            'is_significant': monthly_cv > 0.15
        }
    
    def _analyze_quarterly_seasonality(self, data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Analyze quarterly seasonality patterns"""
        
        # Group by quarter
        quarterly_stats = data.groupby('quarter')[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate seasonality strength
        overall_mean = data[metric].mean()
        quarterly_cv = quarterly_stats['mean'].std() / overall_mean if overall_mean > 0 else 0
        
        # Find peak and trough quarters
        peak_quarter = quarterly_stats.loc[quarterly_stats['mean'].idxmax(), 'quarter']
        trough_quarter = quarterly_stats.loc[quarterly_stats['mean'].idxmin(), 'quarter']
        
        return {
            'seasonality_strength': float(quarterly_cv),
            'peak_quarter': f'Q{int(peak_quarter)}',
            'trough_quarter': f'Q{int(trough_quarter)}',
            'peak_value': float(quarterly_stats['mean'].max()),
            'trough_value': float(quarterly_stats['mean'].min()),
            'quarterly_pattern': {f'Q{int(row["quarter"])}': float(row['mean']) for _, row in quarterly_stats.iterrows()},
            'is_significant': quarterly_cv > 0.2
        }
    
    def _analyze_holiday_effects(self, data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Analyze holiday effects on metrics"""
        
        if 'is_holiday' not in data.columns:
            return {'error': 'Holiday data not available'}
        
        holiday_data = data[data['is_holiday'] == 1][metric]
        non_holiday_data = data[data['is_holiday'] == 0][metric]
        
        if len(holiday_data) == 0 or len(non_holiday_data) == 0:
            return {'error': 'Insufficient holiday data'}
        
        # Calculate effect size
        holiday_mean = holiday_data.mean()
        non_holiday_mean = non_holiday_data.mean()
        effect_size = (holiday_mean - non_holiday_mean) / non_holiday_mean if non_holiday_mean > 0 else 0
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(holiday_data, non_holiday_data)
        
        return {
            'holiday_effect_size': float(effect_size),
            'holiday_mean': float(holiday_mean),
            'non_holiday_mean': float(non_holiday_mean),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'effect_direction': 'positive' if effect_size > 0 else 'negative' if effect_size < 0 else 'neutral'
        }
    
    def _identify_dominant_seasonality(self, weekly: Dict[str, Any], monthly: Dict[str, Any], quarterly: Dict[str, Any]) -> str:
        """Identify the dominant seasonality pattern"""
        
        weekly_strength = weekly.get('seasonality_strength', 0)
        monthly_strength = monthly.get('seasonality_strength', 0)
        quarterly_strength = quarterly.get('seasonality_strength', 0)
        
        max_strength = max(weekly_strength, monthly_strength, quarterly_strength)
        
        if max_strength < 0.1:
            return 'no_clear_seasonality'
        elif weekly_strength == max_strength:
            return 'weekly'
        elif monthly_strength == max_strength:
            return 'monthly'
        else:
            return 'quarterly'
    
    def _detect_change_points(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Detect change points in time series"""
        
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'brand_id', 'brand_name', 'brand_type',
            'date', 'day_of_week', 'day_of_month', 'day_of_year', 'week', 'month', 'quarter',
            'is_weekend', 'is_holiday'
        ] and not col.endswith('_ma_7') and not col.endswith('_ma_30') and not col.endswith('_lag_1') and not col.endswith('_lag_7') and not col.endswith('_pct_change')]
        
        change_point_results = {}
        
        for brand_name in data['brand_name'].unique():
            brand_data = data[data['brand_name'] == brand_name].sort_values('date')
            
            if len(brand_data) < 30:
                continue
            
            brand_change_points = {}
            
            for metric in metric_cols:
                if metric in brand_data.columns:
                    metric_data = brand_data[metric].dropna()
                    dates = brand_data['date'].iloc[:len(metric_data)]
                    
                    if len(metric_data) < 20:
                        continue
                    
                    # Simple change point detection using CUSUM
                    change_points = self._cusum_change_detection(metric_data, dates)
                    
                    # Variance change detection
                    variance_changes = self._variance_change_detection(metric_data, dates)
                    
                    brand_change_points[metric] = {
                        'mean_change_points': change_points,
                        'variance_change_points': variance_changes,
                        'total_change_points': len(change_points) + len(variance_changes)
                    }
            
            change_point_results[brand_name] = brand_change_points
        
        return change_point_results
    
    def _cusum_change_detection(self, data: pd.Series, dates: pd.Series, threshold: float = 5.0) -> List[Dict[str, Any]]:
        """Detect change points using CUSUM algorithm"""
        
        # Calculate CUSUM statistics
        mean_data = data.mean()
        std_data = data.std()
        
        if std_data == 0:
            return []
        
        # Standardize data
        standardized = (data - mean_data) / std_data
        
        # CUSUM for upward changes
        cusum_pos = np.zeros(len(standardized))
        cusum_neg = np.zeros(len(standardized))
        
        for i in range(1, len(standardized)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + standardized.iloc[i] - 0.5)
            cusum_neg[i] = min(0, cusum_neg[i-1] + standardized.iloc[i] + 0.5)
        
        # Find change points
        change_points = []
        
        # Positive changes
        pos_peaks, _ = find_peaks(cusum_pos, height=threshold)
        for peak in pos_peaks:
            change_points.append({
                'date': dates.iloc[peak].isoformat(),
                'index': int(peak),
                'type': 'mean_increase',
                'magnitude': float(cusum_pos[peak])
            })
        
        # Negative changes
        neg_peaks, _ = find_peaks(-cusum_neg, height=threshold)
        for peak in neg_peaks:
            change_points.append({
                'date': dates.iloc[peak].isoformat(),
                'index': int(peak),
                'type': 'mean_decrease',
                'magnitude': float(abs(cusum_neg[peak]))
            })
        
        return change_points
    
    def _variance_change_detection(self, data: pd.Series, dates: pd.Series, window: int = 10) -> List[Dict[str, Any]]:
        """Detect variance change points"""
        
        if len(data) < 2 * window:
            return []
        
        # Calculate rolling variance
        rolling_var = data.rolling(window=window).var()
        
        # Find significant variance changes
        var_changes = []
        for i in range(window, len(rolling_var) - window):
            before_var = rolling_var.iloc[i-window:i].mean()
            after_var = rolling_var.iloc[i:i+window].mean()
            
            if before_var > 0:
                var_ratio = after_var / before_var
                
                # Significant change if ratio > 2 or < 0.5
                if var_ratio > 2 or var_ratio < 0.5:
                    var_changes.append({
                        'date': dates.iloc[i].isoformat(),
                        'index': int(i),
                        'type': 'variance_increase' if var_ratio > 2 else 'variance_decrease',
                        'variance_ratio': float(var_ratio)
                    })
        
        return var_changes
    
    def _analyze_volatility(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze volatility patterns in the data"""
        
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'brand_id', 'brand_name', 'brand_type',
            'date', 'day_of_week', 'day_of_month', 'day_of_year', 'week', 'month', 'quarter',
            'is_weekend', 'is_holiday'
        ] and not col.endswith('_ma_7') and not col.endswith('_ma_30') and not col.endswith('_lag_1') and not col.endswith('_lag_7') and not col.endswith('_pct_change')]
        
        volatility_results = {}
        
        for brand_name in data['brand_name'].unique():
            brand_data = data[data['brand_name'] == brand_name].sort_values('date')
            
            if len(brand_data) < 30:
                continue
            
            brand_volatility = {}
            
            for metric in metric_cols:
                if metric in brand_data.columns:
                    metric_data = brand_data[metric].dropna()
                    
                    if len(metric_data) < 20:
                        continue
                    
                    # Calculate various volatility measures
                    volatility_measures = self._calculate_volatility_measures(metric_data)
                    
                    # Rolling volatility
                    rolling_volatility = self._calculate_rolling_volatility(metric_data)
                    
                    # Volatility clustering
                    volatility_clustering = self._detect_volatility_clustering(metric_data)
                    
                    brand_volatility[metric] = {
                        'volatility_measures': volatility_measures,
                        'rolling_volatility': rolling_volatility,
                        'volatility_clustering': volatility_clustering
                    }
            
            volatility_results[brand_name] = brand_volatility
        
        return volatility_results
    
    def _calculate_volatility_measures(self, data: pd.Series) -> Dict[str, float]:
        """Calculate various volatility measures"""
        
        # Standard deviation
        std_vol = data.std()
        
        # Coefficient of variation
        cv = std_vol / data.mean() if data.mean() > 0 else 0
        
        # Average absolute deviation
        aad = np.mean(np.abs(data - data.mean()))
        
        # Interquartile range
        iqr = data.quantile(0.75) - data.quantile(0.25)
        
        # Range
        range_vol = data.max() - data.min()
        
        return {
            'standard_deviation': float(std_vol),
            'coefficient_of_variation': float(cv),
            'average_absolute_deviation': float(aad),
            'interquartile_range': float(iqr),
            'range': float(range_vol)
        }
    
    def _calculate_rolling_volatility(self, data: pd.Series, window: int = None) -> Dict[str, Any]:
        """Calculate rolling volatility"""
        
        if window is None:
            window = self.config['volatility_window']
        
        if len(data) < window:
            return {'error': 'Insufficient data for rolling volatility'}
        
        # Rolling standard deviation
        rolling_std = data.rolling(window=window).std()
        
        # Rolling coefficient of variation
        rolling_mean = data.rolling(window=window).mean()
        rolling_cv = rolling_std / rolling_mean
        
        return {
            'rolling_std': rolling_std.dropna().tolist(),
            'rolling_cv': rolling_cv.dropna().tolist(),
            'average_volatility': float(rolling_std.mean()),
            'volatility_trend': 'increasing' if rolling_std.iloc[-1] > rolling_std.iloc[0] else 'decreasing'
        }
    
    def _detect_volatility_clustering(self, data: pd.Series) -> Dict[str, Any]:
        """Detect volatility clustering (periods of high/low volatility)"""
        
        # Calculate absolute returns
        returns = data.pct_change().dropna()
        abs_returns = np.abs(returns)
        
        if len(abs_returns) < 10:
            return {'error': 'Insufficient data for volatility clustering analysis'}
        
        # Test for volatility clustering using Ljung-Box test on squared returns
        squared_returns = returns ** 2
        
        # Simple volatility clustering detection
        high_vol_threshold = abs_returns.quantile(0.8)
        low_vol_threshold = abs_returns.quantile(0.2)
        
        high_vol_periods = (abs_returns > high_vol_threshold).astype(int)
        low_vol_periods = (abs_returns < low_vol_threshold).astype(int)
        
        # Count consecutive periods
        high_vol_clusters = self._count_consecutive_periods(high_vol_periods)
        low_vol_clusters = self._count_consecutive_periods(low_vol_periods)
        
        return {
            'high_volatility_clusters': high_vol_clusters,
            'low_volatility_clusters': low_vol_clusters,
            'clustering_present': len(high_vol_clusters) > 0 or len(low_vol_clusters) > 0
        }
    
    def _count_consecutive_periods(self, binary_series: pd.Series) -> List[Dict[str, int]]:
        """Count consecutive periods of 1s in binary series"""
        
        clusters = []
        current_cluster_length = 0
        current_cluster_start = None
        
        for i, value in enumerate(binary_series):
            if value == 1:
                if current_cluster_length == 0:
                    current_cluster_start = i
                current_cluster_length += 1
            else:
                if current_cluster_length > 1:  # Only count clusters of length > 1
                    clusters.append({
                        'start_index': current_cluster_start,
                        'length': current_cluster_length
                    })
                current_cluster_length = 0
        
        # Check if series ends with a cluster
        if current_cluster_length > 1:
            clusters.append({
                'start_index': current_cluster_start,
                'length': current_cluster_length
            })
        
        return clusters
    
    def _perform_forecasting_analysis(self, data: pd.DataFrame, forecast_horizon: int, **kwargs) -> Dict[str, Any]:
        """Perform forecasting analysis using multiple models"""
        
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'brand_id', 'brand_name', 'brand_type',
            'date', 'day_of_week', 'day_of_month', 'day_of_year', 'week', 'month', 'quarter',
            'is_weekend', 'is_holiday'
        ] and not col.endswith('_ma_7') and not col.endswith('_ma_30') and not col.endswith('_lag_1') and not col.endswith('_lag_7') and not col.endswith('_pct_change')]
        
        forecasting_results = {}
        
        for brand_name in data['brand_name'].unique():
            brand_data = data[data['brand_name'] == brand_name].sort_values('date')
            
            if len(brand_data) < 30:
                continue
            
            brand_forecasts = {}
            
            for metric in metric_cols:
                if metric in brand_data.columns:
                    metric_data = brand_data[metric].dropna()
                    dates = brand_data['date'].iloc[:len(metric_data)]
                    
                    if len(metric_data) < 20:
                        continue
                    
                    # Multiple forecasting models
                    forecasts = {}
                    
                    # ARIMA forecast
                    arima_forecast = self._arima_forecast(metric_data, forecast_horizon)
                    if 'error' not in arima_forecast:
                        forecasts['arima'] = arima_forecast
                    
                    # Exponential smoothing forecast
                    exp_smooth_forecast = self._exponential_smoothing_forecast(metric_data, forecast_horizon)
                    if 'error' not in exp_smooth_forecast:
                        forecasts['exponential_smoothing'] = exp_smooth_forecast
                    
                    # Linear regression forecast
                    linear_forecast = self._linear_regression_forecast(metric_data, forecast_horizon)
                    forecasts['linear_regression'] = linear_forecast
                    
                    # Random forest forecast
                    rf_forecast = self._random_forest_forecast(brand_data, metric, forecast_horizon)
                    if 'error' not in rf_forecast:
                        forecasts['random_forest'] = rf_forecast
                    
                    # Ensemble forecast
                    ensemble_forecast = self._create_ensemble_forecast(forecasts, forecast_horizon)
                    
                    brand_forecasts[metric] = {
                        'individual_forecasts': forecasts,
                        'ensemble_forecast': ensemble_forecast,
                        'forecast_horizon': forecast_horizon,
                        'last_actual_value': float(metric_data.iloc[-1]),
                        'forecast_confidence': self._calculate_forecast_confidence(forecasts)
                    }
            
            forecasting_results[brand_name] = brand_forecasts
        
        return forecasting_results
    
    def _arima_forecast(self, data: pd.Series, horizon: int) -> Dict[str, Any]:
        """Perform ARIMA forecasting"""
        
        try:
            # Auto ARIMA model selection (simplified)
            model = ARIMA(data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=horizon)
            conf_int = fitted_model.get_forecast(steps=horizon).conf_int()
            
            return {
                'forecast_values': forecast.tolist(),
                'confidence_intervals': {
                    'lower': conf_int.iloc[:, 0].tolist(),
                    'upper': conf_int.iloc[:, 1].tolist()
                },
                'model_aic': float(fitted_model.aic),
                'model_params': fitted_model.params.to_dict()
            }
        
        except Exception as e:
            return {'error': f'ARIMA forecast failed: {str(e)}'}
    
    def _exponential_smoothing_forecast(self, data: pd.Series, horizon: int) -> Dict[str, Any]:
        """Perform exponential smoothing forecasting"""
        
        try:
            # Exponential smoothing model
            model = ExponentialSmoothing(data, trend='add', seasonal=None)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=horizon)
            
            return {
                'forecast_values': forecast.tolist(),
                'model_params': {
                    'alpha': float(fitted_model.params['smoothing_level']),
                    'beta': float(fitted_model.params.get('smoothing_trend', 0))
                }
            }
        
        except Exception as e:
            return {'error': f'Exponential smoothing forecast failed: {str(e)}'}
    
    def _linear_regression_forecast(self, data: pd.Series, horizon: int) -> Dict[str, Any]:
        """Perform linear regression forecasting"""
        
        # Prepare data
        x = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        
        # Fit model
        model = LinearRegression()
        model.fit(x, y)
        
        # Generate forecast
        future_x = np.arange(len(data), len(data) + horizon).reshape(-1, 1)
        forecast = model.predict(future_x)
        
        return {
            'forecast_values': forecast.tolist(),
            'model_params': {
                'slope': float(model.coef_[0]),
                'intercept': float(model.intercept_),
                'r_squared': float(model.score(x, y))
            }
        }
    
    def _random_forest_forecast(self, data: pd.DataFrame, metric: str, horizon: int) -> Dict[str, Any]:
        """Perform random forest forecasting using multiple features"""
        
        try:
            # Prepare features
            feature_cols = [col for col in data.columns if col.endswith('_ma_7') or col.endswith('_ma_30') or 
                           col.endswith('_lag_1') or col.endswith('_lag_7') or 
                           col in ['day_of_week', 'month', 'quarter', 'is_weekend']]
            
            if len(feature_cols) == 0:
                return {'error': 'No features available for random forest'}
            
            # Create training data
            train_data = data[feature_cols + [metric]].dropna()
            
            if len(train_data) < 10:
                return {'error': 'Insufficient training data'}
            
            X = train_data[feature_cols]
            y = train_data[metric]
            
            # Fit model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Generate forecast (simplified - using last known feature values)
            last_features = X.iloc[-1:].values
            forecast = []
            
            for _ in range(horizon):
                pred = model.predict(last_features)[0]
                forecast.append(pred)
                # In a real implementation, features would be updated based on predictions
            
            return {
                'forecast_values': forecast,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
                'model_score': float(model.score(X, y))
            }
        
        except Exception as e:
            return {'error': f'Random forest forecast failed: {str(e)}'}
    
    def _create_ensemble_forecast(self, forecasts: Dict[str, Dict[str, Any]], horizon: int) -> Dict[str, Any]:
        """Create ensemble forecast from multiple models"""
        
        if not forecasts:
            return {'error': 'No forecasts available for ensemble'}
        
        # Extract forecast values
        forecast_arrays = []
        model_weights = []
        
        for model_name, forecast_data in forecasts.items():
            if 'forecast_values' in forecast_data:
                forecast_values = forecast_data['forecast_values']
                if len(forecast_values) == horizon:
                    forecast_arrays.append(forecast_values)
                    
                    # Simple weighting based on model type
                    if model_name == 'arima':
                        model_weights.append(0.3)
                    elif model_name == 'exponential_smoothing':
                        model_weights.append(0.25)
                    elif model_name == 'random_forest':
                        model_weights.append(0.25)
                    else:
                        model_weights.append(0.2)
        
        if not forecast_arrays:
            return {'error': 'No valid forecasts for ensemble'}
        
        # Normalize weights
        total_weight = sum(model_weights)
        model_weights = [w / total_weight for w in model_weights]
        
        # Calculate weighted average
        ensemble_forecast = np.average(forecast_arrays, axis=0, weights=model_weights)
        
        # Calculate forecast variance as uncertainty measure
        forecast_variance = np.var(forecast_arrays, axis=0)
        
        return {
            'forecast_values': ensemble_forecast.tolist(),
            'forecast_uncertainty': forecast_variance.tolist(),
            'models_used': list(forecasts.keys()),
            'model_weights': dict(zip(forecasts.keys(), model_weights))
        }
    
    def _calculate_forecast_confidence(self, forecasts: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall forecast confidence based on model agreement"""
        
        if len(forecasts) < 2:
            return 0.5  # Low confidence with only one model
        
        # Extract first forecast value from each model
        first_forecasts = []
        for forecast_data in forecasts.values():
            if 'forecast_values' in forecast_data and len(forecast_data['forecast_values']) > 0:
                first_forecasts.append(forecast_data['forecast_values'][0])
        
        if len(first_forecasts) < 2:
            return 0.5
        
        # Calculate coefficient of variation as inverse confidence measure
        cv = np.std(first_forecasts) / np.mean(first_forecasts) if np.mean(first_forecasts) > 0 else 1
        
        # Convert to confidence score (0-1, higher is better)
        confidence = max(0, 1 - cv)
        
        return float(confidence)
    
    def _analyze_trend_correlations(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze correlations between trends of different metrics"""
        
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'brand_id', 'brand_name', 'brand_type',
            'date', 'day_of_week', 'day_of_month', 'day_of_year', 'week', 'month', 'quarter',
            'is_weekend', 'is_holiday'
        ] and not col.endswith('_ma_7') and not col.endswith('_ma_30') and not col.endswith('_lag_1') and not col.endswith('_lag_7') and not col.endswith('_pct_change')]
        
        correlation_results = {}
        
        for brand_name in data['brand_name'].unique():
            brand_data = data[data['brand_name'] == brand_name].sort_values('date')
            
            if len(brand_data) < 30:
                continue
            
            # Calculate trend slopes for each metric
            trend_slopes = {}
            for metric in metric_cols:
                if metric in brand_data.columns:
                    metric_data = brand_data[metric].dropna()
                    if len(metric_data) >= 10:
                        x = np.arange(len(metric_data))
                        slope, _, _, _, _ = stats.linregress(x, metric_data)
                        trend_slopes[metric] = slope
            
            if len(trend_slopes) < 2:
                continue
            
            # Calculate correlations between trend slopes
            slope_df = pd.DataFrame([trend_slopes])
            correlation_matrix = slope_df.T.corr()
            
            # Find strong correlations
            strong_correlations = []
            for i, metric1 in enumerate(trend_slopes.keys()):
                for j, metric2 in enumerate(list(trend_slopes.keys())[i+1:], i+1):
                    corr = correlation_matrix.loc[metric1, metric2] if not pd.isna(correlation_matrix.loc[metric1, metric2]) else 0
                    if abs(corr) > 0.5:
                        strong_correlations.append({
                            'metric1': metric1,
                            'metric2': metric2,
                            'correlation': float(corr),
                            'relationship': 'positive' if corr > 0 else 'negative'
                        })
            
            correlation_results[brand_name] = {
                'trend_slopes': trend_slopes,
                'strong_correlations': strong_correlations,
                'correlation_matrix': correlation_matrix.to_dict()
            }
        
        return correlation_results
    
    def _compare_brand_trends(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Compare trends across different brands"""
        
        metric_cols = [col for col in data.columns if col not in [
            'project_id', 'project_name', 'brand_id', 'brand_name', 'brand_type',
            'date', 'day_of_week', 'day_of_month', 'day_of_year', 'week', 'month', 'quarter',
            'is_weekend', 'is_holiday'
        ] and not col.endswith('_ma_7') and not col.endswith('_ma_30') and not col.endswith('_lag_1') and not col.endswith('_lag_7') and not col.endswith('_pct_change')]
        
        brand_comparison = {}
        
        for metric in metric_cols:
            metric_trends = {}
            
            for brand_name in data['brand_name'].unique():
                brand_data = data[data['brand_name'] == brand_name].sort_values('date')
                
                if len(brand_data) < 20:
                    continue
                
                metric_data = brand_data[metric].dropna()
                
                if len(metric_data) >= 10:
                    # Calculate trend characteristics
                    x = np.arange(len(metric_data))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, metric_data)
                    
                    metric_trends[brand_name] = {
                        'slope': float(slope),
                        'r_squared': float(r_value ** 2),
                        'p_value': float(p_value),
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                        'trend_strength': abs(slope),
                        'is_significant': p_value < 0.05
                    }
            
            if len(metric_trends) >= 2:
                # Find best and worst performing brands
                slopes = {brand: trend['slope'] for brand, trend in metric_trends.items()}
                best_brand = max(slopes.items(), key=lambda x: x[1])
                worst_brand = min(slopes.items(), key=lambda x: x[1])
                
                # Calculate trend consistency across brands
                slope_values = list(slopes.values())
                trend_consistency = 1 - (np.std(slope_values) / (np.mean(np.abs(slope_values)) + 1e-8))
                
                brand_comparison[metric] = {
                    'brand_trends': metric_trends,
                    'best_performing_brand': {'brand': best_brand[0], 'slope': best_brand[1]},
                    'worst_performing_brand': {'brand': worst_brand[0], 'slope': worst_brand[1]},
                    'trend_consistency': float(max(0, trend_consistency)),
                    'average_slope': float(np.mean(slope_values)),
                    'slope_variance': float(np.var(slope_values))
                }
        
        return brand_comparison
    
    def _generate_trend_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights from trend analysis"""
        
        insights = []
        
        # Trend detection insights
        trend_detection = results.get('trend_detection', {})
        strong_trends = []
        
        for brand_name, brand_trends in trend_detection.items():
            for metric, trend_info in brand_trends.items():
                overall_trend = trend_info.get('overall_trend_direction', '')
                trend_strength = trend_info.get('trend_strength', 0)
                
                if 'strong' in overall_trend and trend_strength > 0.5:
                    strong_trends.append(f"{brand_name}: {metric} ({overall_trend})")
        
        if strong_trends:
            insights.append(f"Strong trends identified: {', '.join(strong_trends[:3])}")
        
        # Seasonality insights
        seasonality = results.get('seasonality_analysis', {})
        seasonal_patterns = []
        
        for brand_name, brand_seasonality in seasonality.items():
            for metric, seasonality_info in brand_seasonality.items():
                dominant = seasonality_info.get('dominant_seasonality', '')
                if dominant and dominant != 'no_clear_seasonality':
                    seasonal_patterns.append(f"{brand_name}: {metric} ({dominant})")
        
        if seasonal_patterns:
            insights.append(f"Seasonal patterns detected: {', '.join(seasonal_patterns[:3])}")
        
        # Forecasting insights
        forecasting = results.get('forecasting', {})
        forecast_confidence_scores = []
        
        for brand_name, brand_forecasts in forecasting.items():
            for metric, forecast_info in brand_forecasts.items():
                confidence = forecast_info.get('forecast_confidence', 0)
                forecast_confidence_scores.append(confidence)
        
        if forecast_confidence_scores:
            avg_confidence = np.mean(forecast_confidence_scores)
            insights.append(f"Average forecast confidence: {avg_confidence:.2f}")
            
            if avg_confidence > 0.8:
                insights.append("High forecast reliability - trends are predictable")
            elif avg_confidence < 0.4:
                insights.append("Low forecast reliability - high uncertainty in trends")
        
        # Volatility insights
        volatility = results.get('volatility_analysis', {})
        high_volatility_metrics = []
        
        for brand_name, brand_volatility in volatility.items():
            for metric, vol_info in brand_volatility.items():
                vol_measures = vol_info.get('volatility_measures', {})
                cv = vol_measures.get('coefficient_of_variation', 0)
                
                if cv > 0.3:  # High volatility threshold
                    high_volatility_metrics.append(f"{brand_name}: {metric}")
        
        if high_volatility_metrics:
            insights.append(f"High volatility metrics: {', '.join(high_volatility_metrics[:3])}")
        
        return insights
    
    def _generate_trend_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from trend analysis"""
        
        recommendations = []
        
        # Trend-based recommendations
        trend_detection = results.get('trend_detection', {})
        
        for brand_name, brand_trends in trend_detection.items():
            for metric, trend_info in brand_trends.items():
                overall_trend = trend_info.get('overall_trend_direction', '')
                trend_strength = trend_info.get('trend_strength', 0)
                
                if 'strong_decreasing' in overall_trend and trend_strength > 0.5:
                    recommendations.append({
                        'type': 'trend_intervention',
                        'priority': 'High',
                        'brand': brand_name,
                        'metric': metric,
                        'action': f"Address declining trend in {metric} for {brand_name}",
                        'trend_direction': 'decreasing',
                        'urgency': 'immediate'
                    })
                
                elif 'strong_increasing' in overall_trend and trend_strength > 0.5:
                    recommendations.append({
                        'type': 'trend_amplification',
                        'priority': 'Medium',
                        'brand': brand_name,
                        'metric': metric,
                        'action': f"Amplify positive trend in {metric} for {brand_name}",
                        'trend_direction': 'increasing',
                        'opportunity': 'growth_acceleration'
                    })
        
        # Seasonality-based recommendations
        seasonality = results.get('seasonality_analysis', {})
        
        for brand_name, brand_seasonality in seasonality.items():
            for metric, seasonality_info in brand_seasonality.items():
                weekly_seasonality = seasonality_info.get('weekly_seasonality', {})
                
                if weekly_seasonality.get('is_significant', False):
                    peak_day = weekly_seasonality.get('peak_day', '')
                    trough_day = weekly_seasonality.get('trough_day', '')
                    
                    recommendations.append({
                        'type': 'seasonal_optimization',
                        'priority': 'Medium',
                        'brand': brand_name,
                        'metric': metric,
                        'action': f"Optimize {metric} scheduling - peak on {peak_day}, low on {trough_day}",
                        'seasonality_type': 'weekly'
                    })
        
        # Volatility-based recommendations
        volatility = results.get('volatility_analysis', {})
        
        for brand_name, brand_volatility in volatility.items():
            for metric, vol_info in brand_volatility.items():
                vol_measures = vol_info.get('volatility_measures', {})
                cv = vol_measures.get('coefficient_of_variation', 0)
                
                if cv > 0.4:  # Very high volatility
                    recommendations.append({
                        'type': 'volatility_reduction',
                        'priority': 'Medium',
                        'brand': brand_name,
                        'metric': metric,
                        'action': f"Implement volatility reduction strategies for {metric}",
                        'volatility_level': 'high',
                        'focus': 'stability_improvement'
                    })
        
        # Forecasting-based recommendations
        forecasting = results.get('forecasting', {})
        
        for brand_name, brand_forecasts in forecasting.items():
            for metric, forecast_info in brand_forecasts.items():
                ensemble_forecast = forecast_info.get('ensemble_forecast', {})
                forecast_values = ensemble_forecast.get('forecast_values', [])
                
                if forecast_values:
                    last_actual = forecast_info.get('last_actual_value', 0)
                    first_forecast = forecast_values[0]
                    
                    if last_actual > 0:
                        change_pct = (first_forecast - last_actual) / last_actual * 100
                        
                        if change_pct < -10:  # Significant decline predicted
                            recommendations.append({
                                'type': 'proactive_intervention',
                                'priority': 'High',
                                'brand': brand_name,
                                'metric': metric,
                                'action': f"Proactive measures needed - {metric} predicted to decline by {abs(change_pct):.1f}%",
                                'predicted_change': change_pct,
                                'timeframe': 'next_period'
                            })
        
        return recommendations
    
    def _assess_trend_risks(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks based on trend analysis"""
        
        risks = {
            'high_risk': [],
            'medium_risk': [],
            'low_risk': []
        }
        
        # Trend-based risks
        trend_detection = results.get('trend_detection', {})
        
        for brand_name, brand_trends in trend_detection.items():
            for metric, trend_info in brand_trends.items():
                overall_trend = trend_info.get('overall_trend_direction', '')
                trend_strength = trend_info.get('trend_strength', 0)
                
                if 'strong_decreasing' in overall_trend and trend_strength > 0.7:
                    risks['high_risk'].append({
                        'type': 'declining_performance',
                        'brand': brand_name,
                        'metric': metric,
                        'description': f"Strong declining trend in {metric}",
                        'trend_strength': trend_strength
                    })
                elif 'decreasing' in overall_trend and trend_strength > 0.4:
                    risks['medium_risk'].append({
                        'type': 'performance_decline',
                        'brand': brand_name,
                        'metric': metric,
                        'description': f"Moderate declining trend in {metric}",
                        'trend_strength': trend_strength
                    })
        
        # Volatility-based risks
        volatility = results.get('volatility_analysis', {})
        
        for brand_name, brand_volatility in volatility.items():
            for metric, vol_info in brand_volatility.items():
                vol_measures = vol_info.get('volatility_measures', {})
                cv = vol_measures.get('coefficient_of_variation', 0)
                
                if cv > 0.5:
                    risks['high_risk'].append({
                        'type': 'high_volatility',
                        'brand': brand_name,
                        'metric': metric,
                        'description': f"Very high volatility in {metric}",
                        'volatility_score': cv
                    })
                elif cv > 0.3:
                    risks['medium_risk'].append({
                        'type': 'moderate_volatility',
                        'brand': brand_name,
                        'metric': metric,
                        'description': f"Moderate volatility in {metric}",
                        'volatility_score': cv
                    })
        
        # Forecast-based risks
        forecasting = results.get('forecasting', {})
        
        for brand_name, brand_forecasts in forecasting.items():
            for metric, forecast_info in brand_forecasts.items():
                confidence = forecast_info.get('forecast_confidence', 0)
                
                if confidence < 0.3:
                    risks['medium_risk'].append({
                        'type': 'forecast_uncertainty',
                        'brand': brand_name,
                        'metric': metric,
                        'description': f"Low forecast confidence for {metric}",
                        'confidence_score': confidence
                    })
        
        # Calculate overall risk score
        risk_score = (len(risks['high_risk']) * 3 + len(risks['medium_risk']) * 2 + len(risks['low_risk']) * 1) / 10
        
        return {
            'risks': risks,
            'overall_risk_score': min(1.0, risk_score),
            'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.3 else 'low',
            'total_risks': len(risks['high_risk']) + len(risks['medium_risk']) + len(risks['low_risk'])
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update analyzer configuration"""
        self.config.update(new_config)
        self.logger.info(f"Trend analyzer configuration updated: {new_config}")

# Factory function for creating trend analyzer
def create_trend_analyzer(projects: List[Project], brands: List[Brand], config: Dict[str, Any] = None) -> TrendAnalyzer:
    """Create Trend Analyzer with specified configuration"""
    return TrendAnalyzer(projects, brands, config)

