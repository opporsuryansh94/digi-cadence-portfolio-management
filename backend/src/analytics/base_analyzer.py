"""
Base analyzer class for Digi-Cadence Portfolio Management Platform
Provides common functionality for all analytics components
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from src.models.portfolio import Project, Brand, Metric, BrandMetric

class BaseAnalyzer(ABC):
    """
    Abstract base class for all analytics components
    Provides common functionality and interface
    """
    
    def __init__(self, projects: List[Project], brands: List[Brand]):
        self.projects = projects
        self.brands = brands
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate inputs
        if not projects:
            raise ValueError("At least one project must be provided")
        if not brands:
            raise ValueError("At least one brand must be provided")
        
        self.logger.info(f"Initialized {self.__class__.__name__} with {len(projects)} projects and {len(brands)} brands")
    
    def get_project_ids(self) -> List[str]:
        """Get list of project IDs"""
        return [str(project.id) for project in self.projects]
    
    def get_brand_ids(self) -> List[str]:
        """Get list of brand IDs"""
        return [str(brand.id) for brand in self.brands]
    
    def get_project_by_id(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        for project in self.projects:
            if str(project.id) == project_id:
                return project
        return None
    
    def get_brand_by_id(self, brand_id: str) -> Optional[Brand]:
        """Get brand by ID"""
        for brand in self.brands:
            if str(brand.id) == brand_id:
                return brand
        return None
    
    def validate_metric_data(self, metrics_data: pd.DataFrame) -> bool:
        """Validate metrics data format and completeness"""
        required_columns = [
            'project_id', 'brand_id', 'metric_id', 'metric_name',
            'raw_value', 'normalized_value', 'metric_type'
        ]
        
        for col in required_columns:
            if col not in metrics_data.columns:
                self.logger.error(f"Missing required column: {col}")
                return False
        
        if metrics_data.empty:
            self.logger.warning("Metrics data is empty")
            return False
        
        return True
    
    def normalize_metric_values(self, values: pd.Series, metric_type: str) -> pd.Series:
        """Normalize metric values to 0-100 scale"""
        if values.empty:
            return values
        
        min_val = values.min()
        max_val = values.max()
        
        if min_val == max_val:
            return pd.Series([50.0] * len(values), index=values.index)
        
        if metric_type == 'maximize':
            # Higher values are better
            normalized = ((values - min_val) / (max_val - min_val)) * 100
        else:
            # Lower values are better (minimize)
            normalized = ((max_val - values) / (max_val - min_val)) * 100
        
        return normalized
    
    def calculate_weighted_score(self, values: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted score from values and weights"""
        if not values or not weights:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for key, value in values.items():
            weight = weights.get(key, 1.0)
            total_weighted_score += value * weight
            total_weight += weight
        
        return total_weighted_score / max(total_weight, 1e-6)
    
    def filter_data_by_confidence(self, data: pd.DataFrame, min_confidence: float = 0.5) -> pd.DataFrame:
        """Filter data by confidence score"""
        if 'confidence_score' in data.columns:
            return data[data['confidence_score'] >= min_confidence]
        return data
    
    def get_time_series_data(self, brand_id: str, metric_id: str, 
                           start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Get time series data for a specific brand and metric"""
        # This would typically query the database
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def calculate_correlation_matrix(self, data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix for metrics"""
        if data.empty:
            return pd.DataFrame()
        
        # Pivot data to have metrics as columns
        pivot_data = data.pivot_table(
            index=['brand_id', 'period_start'],
            columns='metric_name',
            values='normalized_value',
            aggfunc='mean'
        )
        
        return pivot_data.corr(method=method)
    
    def identify_outliers(self, values: pd.Series, method: str = 'iqr') -> pd.Series:
        """Identify outliers in data"""
        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (values < lower_bound) | (values > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((values - values.mean()) / values.std())
            return z_scores > 3
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def calculate_trend(self, time_series: pd.Series) -> Dict[str, float]:
        """Calculate trend metrics for time series data"""
        if len(time_series) < 2:
            return {'trend': 0.0, 'slope': 0.0, 'r_squared': 0.0}
        
        # Simple linear regression
        x = np.arange(len(time_series))
        y = time_series.values
        
        # Calculate slope and intercept
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend direction
        if slope > 0.1:
            trend = 1.0  # Increasing
        elif slope < -0.1:
            trend = -1.0  # Decreasing
        else:
            trend = 0.0  # Stable
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_squared,
            'intercept': intercept
        }
    
    def aggregate_by_time_period(self, data: pd.DataFrame, period: str = 'month') -> pd.DataFrame:
        """Aggregate data by time period"""
        if 'period_start' not in data.columns:
            return data
        
        data = data.copy()
        data['period_start'] = pd.to_datetime(data['period_start'])
        
        if period == 'month':
            data['time_period'] = data['period_start'].dt.to_period('M')
        elif period == 'quarter':
            data['time_period'] = data['period_start'].dt.to_period('Q')
        elif period == 'year':
            data['time_period'] = data['period_start'].dt.to_period('Y')
        else:
            raise ValueError(f"Unknown time period: {period}")
        
        # Aggregate by time period
        aggregated = data.groupby(['brand_id', 'metric_id', 'time_period']).agg({
            'raw_value': 'mean',
            'normalized_value': 'mean',
            'confidence_score': 'mean'
        }).reset_index()
        
        return aggregated
    
    def calculate_portfolio_diversity(self, brand_metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate portfolio diversity score"""
        if not brand_metrics:
            return 0.0
        
        # Calculate diversity based on metric value distributions
        all_values = []
        for brand_data in brand_metrics.values():
            all_values.extend(brand_data.values())
        
        if not all_values:
            return 0.0
        
        # Use coefficient of variation as diversity measure
        mean_val = np.mean(all_values)
        std_val = np.std(all_values)
        
        if mean_val == 0:
            return 0.0
        
        diversity = std_val / mean_val
        return min(1.0, diversity)  # Cap at 1.0
    
    def generate_summary_statistics(self, data: pd.DataFrame, group_by: List[str] = None) -> Dict[str, Any]:
        """Generate summary statistics for the data"""
        if data.empty:
            return {}
        
        summary = {
            'total_records': len(data),
            'unique_brands': data['brand_id'].nunique() if 'brand_id' in data.columns else 0,
            'unique_projects': data['project_id'].nunique() if 'project_id' in data.columns else 0,
            'unique_metrics': data['metric_id'].nunique() if 'metric_id' in data.columns else 0,
            'date_range': {
                'start': data['period_start'].min().isoformat() if 'period_start' in data.columns and not data['period_start'].isna().all() else None,
                'end': data['period_start'].max().isoformat() if 'period_start' in data.columns and not data['period_start'].isna().all() else None
            }
        }
        
        # Add value statistics if available
        if 'normalized_value' in data.columns:
            values = data['normalized_value'].dropna()
            if not values.empty:
                summary['value_statistics'] = {
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75))
                }
        
        return summary
    
    @abstractmethod
    def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Abstract method that must be implemented by all analyzer subclasses
        """
        pass

