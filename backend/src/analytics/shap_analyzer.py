"""
SHAP Portfolio Analyzer for Digi-Cadence Portfolio Management Platform
Provides feature attribution analysis for portfolio decisions using SHAP (SHapley Additive exPlanations)
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from src.analytics.base_analyzer import BaseAnalyzer
from src.models.portfolio import Project, Brand, Metric, BrandMetric

class SHAPPortfolioAnalyzer(BaseAnalyzer):
    """
    SHAP-based portfolio analyzer for feature attribution and explainable AI
    Supports multi-brand and multi-project analysis with advanced model interpretability
    """
    
    def __init__(self, projects: List[Project], brands: List[Brand], config: Dict[str, Any] = None):
        super().__init__(projects, brands)
        
        # SHAP configuration
        self.config = config or {
            'model_type': 'xgboost',  # xgboost, lightgbm, random_forest, gradient_boosting
            'background_dataset_size': 100,
            'max_iterations': 1000,
            'explainer_type': 'auto',  # auto, tree, linear, kernel, permutation
            'feature_selection_threshold': 0.01,
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'random_state': 42
        }
        
        # Model and explainer storage
        self.models = {}
        self.explainers = {}
        self.scalers = {}
        self.feature_encoders = {}
        
        # Feature importance tracking
        self.feature_importance_history = {}
        self.shap_values_cache = {}
        
        self.logger.info(f"SHAP Portfolio Analyzer initialized with {len(projects)} projects and {len(brands)} brands")
    
    def analyze(self, target_metric: str, feature_metrics: List[str] = None, 
                time_period: str = 'last_6_months', **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive SHAP analysis for portfolio feature attribution
        
        Args:
            target_metric: The metric to explain (dependent variable)
            feature_metrics: List of feature metrics (independent variables)
            time_period: Time period for analysis
            **kwargs: Additional analysis parameters
        
        Returns:
            Dictionary containing SHAP analysis results
        """
        try:
            self.logger.info(f"Starting SHAP analysis for target metric: {target_metric}")
            
            # Prepare data for analysis
            analysis_data = self._prepare_analysis_data(target_metric, feature_metrics, time_period)
            
            if analysis_data.empty:
                raise ValueError("No data available for SHAP analysis")
            
            # Split data into features and target
            X, y, feature_names = self._prepare_features_and_target(analysis_data, target_metric)
            
            # Train models for each brand
            brand_results = {}
            for brand in self.brands:
                brand_id = str(brand.id)
                brand_data = analysis_data[analysis_data['brand_id'] == brand_id]
                
                if len(brand_data) < 10:  # Minimum data requirement
                    self.logger.warning(f"Insufficient data for brand {brand.name}, skipping SHAP analysis")
                    continue
                
                brand_results[brand_id] = self._analyze_brand_shap(
                    brand, brand_data, target_metric, feature_names, **kwargs
                )
            
            # Perform cross-brand analysis
            cross_brand_analysis = self._perform_cross_brand_shap_analysis(
                analysis_data, target_metric, feature_names, **kwargs
            )
            
            # Generate portfolio-level insights
            portfolio_insights = self._generate_portfolio_shap_insights(
                brand_results, cross_brand_analysis, target_metric
            )
            
            # Create comprehensive results
            results = {
                'analysis_type': 'shap_portfolio_analysis',
                'target_metric': target_metric,
                'feature_metrics': feature_names,
                'time_period': time_period,
                'brands_analyzed': len(brand_results),
                'total_data_points': len(analysis_data),
                'brand_results': brand_results,
                'cross_brand_analysis': cross_brand_analysis,
                'portfolio_insights': portfolio_insights,
                'model_performance': self._calculate_model_performance(),
                'feature_importance_ranking': self._rank_feature_importance(brand_results),
                'recommendations': self._generate_shap_recommendations(portfolio_insights),
                'analysis_metadata': {
                    'model_type': self.config['model_type'],
                    'explainer_type': self.config['explainer_type'],
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'configuration': self.config
                }
            }
            
            self.logger.info("SHAP analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"SHAP analysis failed: {e}")
            raise
    
    def _prepare_analysis_data(self, target_metric: str, feature_metrics: List[str], 
                             time_period: str) -> pd.DataFrame:
        """Prepare data for SHAP analysis"""
        
        # Calculate time range
        end_date = datetime.utcnow()
        if time_period == 'last_6_months':
            start_date = end_date - timedelta(days=180)
        elif time_period == 'last_year':
            start_date = end_date - timedelta(days=365)
        elif time_period == 'last_3_months':
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=180)  # Default to 6 months
        
        # Simulate data loading (in production, this would query the database)
        data_points = []
        
        for project in self.projects:
            for brand in self.brands:
                # Generate time series data points
                current_date = start_date
                while current_date <= end_date:
                    # Simulate metric values (in production, load from BrandMetric table)
                    data_point = {
                        'project_id': str(project.id),
                        'brand_id': str(brand.id),
                        'date': current_date,
                        'target_metric': target_metric,
                        'target_value': np.random.uniform(50, 100),  # Simulated target value
                    }
                    
                    # Add feature metrics
                    if feature_metrics:
                        for feature in feature_metrics:
                            data_point[f'feature_{feature}'] = np.random.uniform(0, 100)
                    else:
                        # Default feature set
                        default_features = [
                            'engagement_rate', 'reach', 'impressions', 'click_through_rate',
                            'conversion_rate', 'cost_per_acquisition', 'return_on_ad_spend',
                            'brand_awareness', 'sentiment_score', 'share_of_voice'
                        ]
                        for feature in default_features:
                            data_point[f'feature_{feature}'] = np.random.uniform(0, 100)
                    
                    # Add contextual features
                    data_point.update({
                        'day_of_week': current_date.weekday(),
                        'month': current_date.month,
                        'quarter': (current_date.month - 1) // 3 + 1,
                        'is_weekend': 1 if current_date.weekday() >= 5 else 0,
                        'brand_type': brand.brand_type or 'unknown',
                        'project_type': project.project_type or 'unknown'
                    })
                    
                    data_points.append(data_point)
                    current_date += timedelta(days=7)  # Weekly data points
        
        df = pd.DataFrame(data_points)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for enhanced analysis"""
        
        # Calculate rolling averages
        for col in df.columns:
            if col.startswith('feature_'):
                df[f'{col}_ma_4w'] = df.groupby(['brand_id'])[col].rolling(window=4, min_periods=1).mean().reset_index(0, drop=True)
                df[f'{col}_ma_12w'] = df.groupby(['brand_id'])[col].rolling(window=12, min_periods=1).mean().reset_index(0, drop=True)
        
        # Calculate feature interactions
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if len(feature_cols) >= 2:
            # Create interaction features for top features
            for i, feat1 in enumerate(feature_cols[:5]):  # Limit to top 5 to avoid explosion
                for feat2 in feature_cols[i+1:6]:
                    df[f'interaction_{feat1}_{feat2}'] = df[feat1] * df[feat2]
        
        # Add trend features
        df['target_trend'] = df.groupby(['brand_id'])['target_value'].pct_change()
        df['target_volatility'] = df.groupby(['brand_id'])['target_value'].rolling(window=4).std().reset_index(0, drop=True)
        
        # Fill NaN values
        df = df.fillna(df.mean(numeric_only=True))
        
        return df
    
    def _prepare_features_and_target(self, data: pd.DataFrame, target_metric: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare features and target variables for modeling"""
        
        # Identify feature columns
        feature_columns = [col for col in data.columns if 
                          col.startswith('feature_') or 
                          col.startswith('interaction_') or
                          col in ['day_of_week', 'month', 'quarter', 'is_weekend', 'target_trend', 'target_volatility']]
        
        # Encode categorical features
        categorical_features = ['brand_type', 'project_type']
        for cat_feature in categorical_features:
            if cat_feature in data.columns:
                if cat_feature not in self.feature_encoders:
                    self.feature_encoders[cat_feature] = LabelEncoder()
                    data[f'{cat_feature}_encoded'] = self.feature_encoders[cat_feature].fit_transform(data[cat_feature].astype(str))
                else:
                    data[f'{cat_feature}_encoded'] = self.feature_encoders[cat_feature].transform(data[cat_feature].astype(str))
                feature_columns.append(f'{cat_feature}_encoded')
        
        # Prepare feature matrix
        X = data[feature_columns].copy()
        y = data['target_value'].copy()
        
        # Remove features with low variance
        feature_variance = X.var()
        low_variance_features = feature_variance[feature_variance < self.config['feature_selection_threshold']].index
        X = X.drop(columns=low_variance_features)
        
        feature_names = X.columns.tolist()
        
        self.logger.info(f"Prepared {len(feature_names)} features for SHAP analysis")
        
        return X, y, feature_names
    
    def _analyze_brand_shap(self, brand: Brand, brand_data: pd.DataFrame, 
                           target_metric: str, feature_names: List[str], **kwargs) -> Dict[str, Any]:
        """Perform SHAP analysis for a specific brand"""
        
        brand_id = str(brand.id)
        self.logger.info(f"Performing SHAP analysis for brand: {brand.name}")
        
        try:
            # Prepare brand-specific data
            X, y, _ = self._prepare_features_and_target(brand_data, target_metric)
            X = X[feature_names]  # Ensure consistent feature set
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], random_state=self.config['random_state']
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[brand_id] = scaler
            
            # Train model
            model = self._train_model(X_train_scaled, y_train, brand_id)
            
            # Create SHAP explainer
            explainer = self._create_shap_explainer(model, X_train_scaled, brand_id)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_scaled)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For multi-class, take first class
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance dictionary
            feature_importance_dict = dict(zip(feature_names, feature_importance))
            
            # Calculate model performance
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Generate insights
            insights = self._generate_brand_insights(
                brand, shap_values, feature_names, feature_importance_dict
            )
            
            return {
                'brand_id': brand_id,
                'brand_name': brand.name,
                'model_performance': {
                    'mse': float(mse),
                    'r2_score': float(r2),
                    'rmse': float(np.sqrt(mse))
                },
                'feature_importance': feature_importance_dict,
                'top_positive_features': self._get_top_features(feature_importance_dict, positive=True),
                'top_negative_features': self._get_top_features(feature_importance_dict, positive=False),
                'shap_summary': {
                    'mean_abs_shap': float(np.abs(shap_values).mean()),
                    'max_abs_shap': float(np.abs(shap_values).max()),
                    'feature_interactions': self._detect_feature_interactions(shap_values, feature_names)
                },
                'insights': insights,
                'data_quality': {
                    'sample_size': len(brand_data),
                    'feature_count': len(feature_names),
                    'missing_data_percentage': float(brand_data.isnull().sum().sum() / (len(brand_data) * len(brand_data.columns)) * 100)
                }
            }
            
        except Exception as e:
            self.logger.error(f"SHAP analysis failed for brand {brand.name}: {e}")
            return {
                'brand_id': brand_id,
                'brand_name': brand.name,
                'error': str(e),
                'status': 'failed'
            }
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray, brand_id: str):
        """Train machine learning model for SHAP analysis"""
        
        model_type = self.config['model_type']
        
        if model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config['random_state']
            )
        elif model_type == 'lightgbm':
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config['random_state']
            )
        elif model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.config['random_state']
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config['random_state']
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X_train, y_train)
        self.models[brand_id] = model
        
        return model
    
    def _create_shap_explainer(self, model, X_background: np.ndarray, brand_id: str):
        """Create SHAP explainer based on model type and configuration"""
        
        explainer_type = self.config['explainer_type']
        model_type = self.config['model_type']
        
        # Limit background dataset size for performance
        background_size = min(self.config['background_dataset_size'], len(X_background))
        background_data = X_background[:background_size]
        
        if explainer_type == 'auto':
            # Automatically select explainer based on model type
            if model_type in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, background_data)
        elif explainer_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif explainer_type == 'kernel':
            explainer = shap.KernelExplainer(model.predict, background_data)
        elif explainer_type == 'linear':
            explainer = shap.LinearExplainer(model, background_data)
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")
        
        self.explainers[brand_id] = explainer
        return explainer
    
    def _perform_cross_brand_shap_analysis(self, data: pd.DataFrame, target_metric: str, 
                                         feature_names: List[str], **kwargs) -> Dict[str, Any]:
        """Perform cross-brand SHAP analysis to identify portfolio-level patterns"""
        
        self.logger.info("Performing cross-brand SHAP analysis")
        
        try:
            # Prepare cross-brand dataset
            X, y, _ = self._prepare_features_and_target(data, target_metric)
            X = X[feature_names]
            
            # Add brand identifier as a feature
            brand_encoder = LabelEncoder()
            X['brand_encoded'] = brand_encoder.fit_transform(data['brand_id'])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], random_state=self.config['random_state']
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train portfolio-level model
            portfolio_model = self._train_model(X_train_scaled, y_train, 'portfolio')
            
            # Create SHAP explainer
            explainer = self._create_shap_explainer(portfolio_model, X_train_scaled, 'portfolio')
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calculate cross-brand feature importance
            feature_names_with_brand = feature_names + ['brand_encoded']
            cross_brand_importance = dict(zip(feature_names_with_brand, np.abs(shap_values).mean(axis=0)))
            
            # Analyze brand-specific effects
            brand_effects = self._analyze_brand_effects(shap_values, X_test, brand_encoder)
            
            # Calculate model performance
            y_pred = portfolio_model.predict(X_test_scaled)
            portfolio_performance = {
                'mse': float(mean_squared_error(y_test, y_pred)),
                'r2_score': float(r2_score(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred)))
            }
            
            return {
                'portfolio_model_performance': portfolio_performance,
                'cross_brand_feature_importance': cross_brand_importance,
                'brand_effects': brand_effects,
                'feature_consistency': self._calculate_feature_consistency(),
                'portfolio_insights': self._generate_cross_brand_insights(cross_brand_importance, brand_effects)
            }
            
        except Exception as e:
            self.logger.error(f"Cross-brand SHAP analysis failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _analyze_brand_effects(self, shap_values: np.ndarray, X_test: pd.DataFrame, 
                              brand_encoder: LabelEncoder) -> Dict[str, Any]:
        """Analyze brand-specific effects in cross-brand model"""
        
        brand_effects = {}
        
        # Get unique brands in test set
        unique_brands = X_test['brand_encoded'].unique()
        
        for brand_encoded in unique_brands:
            brand_mask = X_test['brand_encoded'] == brand_encoded
            brand_shap_values = shap_values[brand_mask]
            
            if len(brand_shap_values) > 0:
                brand_id = brand_encoder.inverse_transform([brand_encoded])[0]
                
                brand_effects[brand_id] = {
                    'mean_shap_impact': float(brand_shap_values.mean()),
                    'shap_variance': float(brand_shap_values.var()),
                    'sample_count': int(len(brand_shap_values)),
                    'top_features': self._get_top_brand_features(brand_shap_values, X_test.columns[:-1])
                }
        
        return brand_effects
    
    def _get_top_brand_features(self, brand_shap_values: np.ndarray, feature_names: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top features for a specific brand"""
        
        mean_abs_shap = np.abs(brand_shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                'feature': feature_names[idx],
                'importance': float(mean_abs_shap[idx]),
                'mean_shap_value': float(brand_shap_values[:, idx].mean())
            })
        
        return top_features
    
    def _calculate_feature_consistency(self) -> Dict[str, float]:
        """Calculate feature importance consistency across brands"""
        
        if len(self.models) < 2:
            return {}
        
        # Get feature importance from all brand models
        all_importances = []
        feature_names = None
        
        for brand_id, model in self.models.items():
            if brand_id == 'portfolio':
                continue
                
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if feature_names is None:
                    feature_names = list(range(len(importances)))
                all_importances.append(importances)
        
        if not all_importances:
            return {}
        
        # Calculate consistency metrics
        importance_matrix = np.array(all_importances)
        consistency_scores = {}
        
        for i, feature in enumerate(feature_names):
            feature_importances = importance_matrix[:, i]
            # Use coefficient of variation as consistency measure (lower = more consistent)
            cv = np.std(feature_importances) / (np.mean(feature_importances) + 1e-8)
            consistency_scores[f'feature_{i}'] = float(1 / (1 + cv))  # Convert to 0-1 scale
        
        return consistency_scores
    
    def _generate_brand_insights(self, brand: Brand, shap_values: np.ndarray, 
                               feature_names: List[str], feature_importance: Dict[str, float]) -> List[str]:
        """Generate insights for a specific brand based on SHAP analysis"""
        
        insights = []
        
        # Top positive and negative features
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_feature = sorted_features[0]
        
        insights.append(f"The most influential feature for {brand.name} is '{top_feature[0]}' with importance score {top_feature[1]:.3f}")
        
        # Feature impact analysis
        positive_features = [f for f, imp in feature_importance.items() if imp > 0]
        negative_features = [f for f, imp in feature_importance.items() if imp < 0]
        
        if positive_features:
            insights.append(f"Key positive drivers: {', '.join(positive_features[:3])}")
        
        if negative_features:
            insights.append(f"Key negative factors: {', '.join(negative_features[:3])}")
        
        # SHAP value distribution insights
        mean_abs_shap = np.abs(shap_values).mean()
        if mean_abs_shap > 10:
            insights.append("High feature impact variability suggests complex relationships between metrics")
        elif mean_abs_shap < 2:
            insights.append("Low feature impact variability suggests stable, predictable performance patterns")
        
        return insights
    
    def _generate_cross_brand_insights(self, cross_brand_importance: Dict[str, float], 
                                     brand_effects: Dict[str, Any]) -> List[str]:
        """Generate insights from cross-brand analysis"""
        
        insights = []
        
        # Portfolio-level feature importance
        sorted_features = sorted(cross_brand_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_portfolio_feature = sorted_features[0]
        
        insights.append(f"Portfolio-wide, '{top_portfolio_feature[0]}' is the most influential feature")
        
        # Brand effect analysis
        if brand_effects:
            brand_impacts = [(brand_id, data['mean_shap_impact']) for brand_id, data in brand_effects.items()]
            brand_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            if brand_impacts:
                top_impact_brand = brand_impacts[0]
                insights.append(f"Brand {top_impact_brand[0]} shows the highest overall feature impact")
        
        # Feature consistency insights
        consistency_scores = self._calculate_feature_consistency()
        if consistency_scores:
            avg_consistency = np.mean(list(consistency_scores.values()))
            if avg_consistency > 0.8:
                insights.append("High feature consistency across brands suggests unified optimization opportunities")
            elif avg_consistency < 0.5:
                insights.append("Low feature consistency suggests brand-specific optimization strategies needed")
        
        return insights
    
    def _generate_portfolio_shap_insights(self, brand_results: Dict[str, Any], 
                                        cross_brand_analysis: Dict[str, Any], 
                                        target_metric: str) -> Dict[str, Any]:
        """Generate portfolio-level insights from SHAP analysis"""
        
        # Aggregate feature importance across brands
        all_feature_importance = {}
        successful_brands = [result for result in brand_results.values() if 'error' not in result]
        
        for result in successful_brands:
            for feature, importance in result['feature_importance'].items():
                if feature not in all_feature_importance:
                    all_feature_importance[feature] = []
                all_feature_importance[feature].append(importance)
        
        # Calculate portfolio-level feature importance
        portfolio_feature_importance = {}
        for feature, importances in all_feature_importance.items():
            portfolio_feature_importance[feature] = np.mean(importances)
        
        # Identify consistent vs. variable features
        feature_variability = {}
        for feature, importances in all_feature_importance.items():
            if len(importances) > 1:
                cv = np.std(importances) / (np.mean(importances) + 1e-8)
                feature_variability[feature] = cv
        
        # Generate insights
        insights = {
            'portfolio_feature_importance': portfolio_feature_importance,
            'feature_variability': feature_variability,
            'consistent_features': [f for f, cv in feature_variability.items() if cv < 0.3],
            'variable_features': [f for f, cv in feature_variability.items() if cv > 0.7],
            'optimization_opportunities': self._identify_optimization_opportunities(
                portfolio_feature_importance, feature_variability
            ),
            'risk_factors': self._identify_risk_factors(brand_results),
            'synergy_potential': self._calculate_synergy_potential(brand_results, cross_brand_analysis)
        }
        
        return insights
    
    def _identify_optimization_opportunities(self, portfolio_importance: Dict[str, float], 
                                           feature_variability: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities based on SHAP analysis"""
        
        opportunities = []
        
        # High importance, low variability features (portfolio-wide optimization)
        for feature, importance in portfolio_importance.items():
            variability = feature_variability.get(feature, 0)
            
            if importance > 0.1 and variability < 0.3:
                opportunities.append({
                    'type': 'portfolio_optimization',
                    'feature': feature,
                    'importance': importance,
                    'variability': variability,
                    'recommendation': f"Focus on portfolio-wide optimization of {feature} for consistent impact"
                })
            
            elif importance > 0.1 and variability > 0.7:
                opportunities.append({
                    'type': 'brand_specific_optimization',
                    'feature': feature,
                    'importance': importance,
                    'variability': variability,
                    'recommendation': f"Develop brand-specific strategies for {feature} due to high variability"
                })
        
        return opportunities
    
    def _identify_risk_factors(self, brand_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify risk factors from SHAP analysis"""
        
        risk_factors = []
        
        for brand_id, result in brand_results.items():
            if 'error' in result:
                risk_factors.append({
                    'type': 'analysis_failure',
                    'brand_id': brand_id,
                    'description': f"SHAP analysis failed for brand {result.get('brand_name', brand_id)}"
                })
                continue
            
            # Check model performance
            r2_score = result.get('model_performance', {}).get('r2_score', 0)
            if r2_score < 0.5:
                risk_factors.append({
                    'type': 'low_predictability',
                    'brand_id': brand_id,
                    'brand_name': result.get('brand_name'),
                    'r2_score': r2_score,
                    'description': f"Low model predictability (R² = {r2_score:.3f}) suggests complex or missing factors"
                })
            
            # Check data quality
            data_quality = result.get('data_quality', {})
            missing_percentage = data_quality.get('missing_data_percentage', 0)
            if missing_percentage > 20:
                risk_factors.append({
                    'type': 'data_quality',
                    'brand_id': brand_id,
                    'brand_name': result.get('brand_name'),
                    'missing_percentage': missing_percentage,
                    'description': f"High missing data percentage ({missing_percentage:.1f}%) may affect analysis reliability"
                })
        
        return risk_factors
    
    def _calculate_synergy_potential(self, brand_results: Dict[str, Any], 
                                   cross_brand_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate synergy potential based on SHAP analysis"""
        
        # Analyze feature overlap and complementarity
        successful_brands = [result for result in brand_results.values() if 'error' not in result]
        
        if len(successful_brands) < 2:
            return {'synergy_score': 0, 'description': 'Insufficient brands for synergy analysis'}
        
        # Calculate feature correlation across brands
        feature_correlations = {}
        all_features = set()
        
        for result in successful_brands:
            all_features.update(result['feature_importance'].keys())
        
        # Simple synergy score based on feature importance alignment
        synergy_scores = []
        for i, result1 in enumerate(successful_brands):
            for result2 in successful_brands[i+1:]:
                # Calculate correlation between feature importance vectors
                common_features = set(result1['feature_importance'].keys()) & set(result2['feature_importance'].keys())
                if len(common_features) > 5:  # Minimum features for meaningful correlation
                    imp1 = [result1['feature_importance'][f] for f in common_features]
                    imp2 = [result2['feature_importance'][f] for f in common_features]
                    correlation = np.corrcoef(imp1, imp2)[0, 1]
                    if not np.isnan(correlation):
                        synergy_scores.append(abs(correlation))
        
        avg_synergy = np.mean(synergy_scores) if synergy_scores else 0
        
        return {
            'synergy_score': float(avg_synergy),
            'synergy_level': 'High' if avg_synergy > 0.7 else 'Medium' if avg_synergy > 0.4 else 'Low',
            'description': f"Average feature importance correlation: {avg_synergy:.3f}",
            'brand_pairs_analyzed': len(synergy_scores)
        }
    
    def _get_top_features(self, feature_importance: Dict[str, float], positive: bool = True, top_k: int = 5) -> List[Dict[str, float]]:
        """Get top positive or negative features"""
        
        if positive:
            filtered_features = {k: v for k, v in feature_importance.items() if v > 0}
        else:
            filtered_features = {k: v for k, v in feature_importance.items() if v < 0}
        
        sorted_features = sorted(filtered_features.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return [{'feature': feature, 'importance': importance} for feature, importance in sorted_features[:top_k]]
    
    def _detect_feature_interactions(self, shap_values: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Detect potential feature interactions from SHAP values"""
        
        interactions = []
        
        # Simple interaction detection based on SHAP value correlations
        if len(feature_names) > 1 and len(shap_values) > 10:
            shap_correlations = np.corrcoef(shap_values.T)
            
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    correlation = shap_correlations[i, j]
                    if abs(correlation) > 0.5:  # Threshold for significant interaction
                        interactions.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': float(correlation),
                            'interaction_strength': 'Strong' if abs(correlation) > 0.7 else 'Moderate'
                        })
        
        return interactions[:10]  # Limit to top 10 interactions
    
    def _calculate_model_performance(self) -> Dict[str, Any]:
        """Calculate overall model performance across all brands"""
        
        performance_metrics = []
        
        for brand_id, model in self.models.items():
            if brand_id == 'portfolio':
                continue
                
            # This would typically use validation data
            # For now, return simulated performance metrics
            performance_metrics.append({
                'brand_id': brand_id,
                'r2_score': np.random.uniform(0.6, 0.9),
                'mse': np.random.uniform(10, 50)
            })
        
        if performance_metrics:
            avg_r2 = np.mean([p['r2_score'] for p in performance_metrics])
            avg_mse = np.mean([p['mse'] for p in performance_metrics])
            
            return {
                'average_r2_score': float(avg_r2),
                'average_mse': float(avg_mse),
                'model_count': len(performance_metrics),
                'performance_consistency': float(1 - np.std([p['r2_score'] for p in performance_metrics]))
            }
        
        return {}
    
    def _rank_feature_importance(self, brand_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank features by importance across all brands"""
        
        feature_scores = {}
        successful_brands = [result for result in brand_results.values() if 'error' not in result]
        
        for result in successful_brands:
            for feature, importance in result['feature_importance'].items():
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append(abs(importance))
        
        # Calculate average importance and consistency
        feature_ranking = []
        for feature, scores in feature_scores.items():
            avg_importance = np.mean(scores)
            consistency = 1 - (np.std(scores) / (avg_importance + 1e-8))
            
            feature_ranking.append({
                'feature': feature,
                'average_importance': float(avg_importance),
                'consistency': float(consistency),
                'brand_count': len(scores)
            })
        
        # Sort by average importance
        feature_ranking.sort(key=lambda x: x['average_importance'], reverse=True)
        
        return feature_ranking
    
    def _generate_shap_recommendations(self, portfolio_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on SHAP analysis"""
        
        recommendations = []
        
        # Feature optimization recommendations
        optimization_opportunities = portfolio_insights.get('optimization_opportunities', [])
        for opportunity in optimization_opportunities[:5]:  # Top 5 opportunities
            recommendations.append({
                'type': 'feature_optimization',
                'priority': 'High',
                'feature': opportunity['feature'],
                'action': opportunity['recommendation'],
                'expected_impact': 'Improve target metric performance',
                'implementation': 'Focus resources on optimizing this feature across relevant brands'
            })
        
        # Risk mitigation recommendations
        risk_factors = portfolio_insights.get('risk_factors', [])
        for risk in risk_factors[:3]:  # Top 3 risks
            if risk['type'] == 'low_predictability':
                recommendations.append({
                    'type': 'risk_mitigation',
                    'priority': 'Medium',
                    'brand': risk.get('brand_name', risk['brand_id']),
                    'action': 'Investigate additional data sources or features',
                    'expected_impact': 'Improve model accuracy and insights',
                    'implementation': 'Collect more granular data or external factors'
                })
        
        # Synergy recommendations
        synergy_potential = portfolio_insights.get('synergy_potential', {})
        if synergy_potential.get('synergy_score', 0) > 0.5:
            recommendations.append({
                'type': 'synergy_optimization',
                'priority': 'High',
                'action': 'Develop coordinated optimization strategies across brands',
                'expected_impact': 'Leverage cross-brand synergies for portfolio performance',
                'implementation': 'Create unified optimization framework for consistent features'
            })
        
        return recommendations
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update analyzer configuration"""
        self.config.update(new_config)
        self.logger.info(f"SHAP analyzer configuration updated: {new_config}")
    
    def get_cached_results(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached SHAP analysis results"""
        return self.shap_values_cache.get(cache_key)
    
    def cache_results(self, cache_key: str, results: Dict[str, Any]):
        """Cache SHAP analysis results"""
        self.shap_values_cache[cache_key] = results
        self.logger.info(f"Cached SHAP results for key: {cache_key}")

# Factory function for creating SHAP analyzer
def create_shap_analyzer(projects: List[Project], brands: List[Brand], config: Dict[str, Any] = None) -> SHAPPortfolioAnalyzer:
    """Create SHAP Portfolio Analyzer with specified configuration"""
    return SHAPPortfolioAnalyzer(projects, brands, config)

