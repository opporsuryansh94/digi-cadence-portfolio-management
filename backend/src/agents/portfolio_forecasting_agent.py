"""
Portfolio Forecasting Agent for Digi-Cadence Portfolio Management Platform
Intelligent agent specialized in predictive analytics and forecasting for portfolio management
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

from src.agents.base_agent import BaseAgent, AgentCapability, AgentTask, TaskPriority
from src.models.portfolio import Project, Brand, Organization

class PortfolioForecastingAgent(BaseAgent):
    """
    Intelligent agent for portfolio forecasting and predictive analytics
    Specializes in trend prediction, scenario modeling, and risk forecasting
    """
    
    def __init__(self, agent_id: str = "portfolio_forecaster", config: Dict[str, Any] = None):
        # Default configuration
        default_config = {
            'forecast_horizons': [7, 30, 90, 180, 365],  # days
            'forecast_metrics': ['roi', 'engagement_rate', 'conversion_rate', 'reach', 'cost_efficiency'],
            'forecasting_models': ['arima', 'random_forest', 'linear_regression', 'ensemble'],
            'confidence_levels': [0.80, 0.90, 0.95],
            'scenario_types': ['optimistic', 'realistic', 'pessimistic'],
            'trend_detection_sensitivity': 0.05,
            'seasonality_detection_enabled': True,
            'anomaly_detection_enabled': True,
            'ensemble_weights': {
                'arima': 0.3,
                'random_forest': 0.4,
                'linear_regression': 0.2,
                'trend_analyzer': 0.1
            },
            'forecast_accuracy_threshold': 0.85,
            'real_time_forecasting_enabled': True,
            'adaptive_model_selection': True,
            'forecast_update_frequency': 'daily',
            'risk_forecasting_enabled': True,
            'market_factor_integration': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            agent_id=agent_id,
            name="Portfolio Forecasting Agent",
            capabilities=[
                AgentCapability.FORECASTING,
                AgentCapability.ANALYSIS,
                AgentCapability.MONITORING,
                AgentCapability.STRATEGY
            ],
            config=default_config
        )
        
        # Forecasting state
        self.forecast_models = {}
        self.forecast_history = {}
        self.model_performance = {}
        self.active_forecasts = {}
        
        # Trend and seasonality detection
        self.trend_patterns = {}
        self.seasonality_patterns = {}
        self.anomaly_patterns = {}
        
        # Risk forecasting
        self.risk_models = {}
        self.risk_forecasts = {}
        
        # Market factors
        self.market_factors = {}
        self.external_indicators = {}
        
        # Register message handlers
        self._register_message_handlers()
        
        self.logger.info("Portfolio Forecasting Agent initialized")
    
    def get_required_config_keys(self) -> List[str]:
        """Return required configuration keys"""
        return ['forecast_horizons', 'forecast_metrics']
    
    def _register_message_handlers(self):
        """Register message handlers for inter-agent communication"""
        self.register_message_handler('forecast_request', self._handle_forecast_request)
        self.register_message_handler('trend_analysis_request', self._handle_trend_analysis_request)
        self.register_message_handler('scenario_modeling_request', self._handle_scenario_modeling_request)
        self.register_message_handler('risk_forecast_request', self._handle_risk_forecast_request)
        self.register_message_handler('market_update', self._handle_market_update)
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process forecasting tasks"""
        task_type = task.task_type
        parameters = task.parameters
        
        self.logger.info(f"Processing task: {task_type}")
        
        try:
            if task_type == 'portfolio_forecasting':
                return await self._forecast_portfolio_performance(parameters)
            elif task_type == 'brand_forecasting':
                return await self._forecast_brand_performance(parameters)
            elif task_type == 'project_forecasting':
                return await self._forecast_project_performance(parameters)
            elif task_type == 'trend_analysis':
                return await self._analyze_trends(parameters)
            elif task_type == 'scenario_modeling':
                return await self._model_scenarios(parameters)
            elif task_type == 'risk_forecasting':
                return await self._forecast_risks(parameters)
            elif task_type == 'seasonality_analysis':
                return await self._analyze_seasonality(parameters)
            elif task_type == 'anomaly_detection':
                return await self._detect_anomalies(parameters)
            elif task_type == 'forecast_accuracy_analysis':
                return await self._analyze_forecast_accuracy(parameters)
            elif task_type == 'predictive_optimization':
                return await self._predictive_optimization_analysis(parameters)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Task processing failed: {e}")
            raise
    
    async def _forecast_portfolio_performance(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast overall portfolio performance"""
        
        forecast_horizon = parameters.get('forecast_horizon', 30)
        target_metrics = parameters.get('target_metrics', self.config['forecast_metrics'])
        confidence_level = parameters.get('confidence_level', 0.90)
        include_scenarios = parameters.get('include_scenarios', True)
        
        self.logger.info(f"Forecasting portfolio performance for {forecast_horizon} days")
        
        # Get historical portfolio data
        historical_data = await self._get_portfolio_historical_data(target_metrics)
        
        # Generate forecasts for each metric
        forecasts = {}
        for metric in target_metrics:
            metric_forecast = await self._generate_metric_forecast(
                historical_data[metric], forecast_horizon, confidence_level
            )
            forecasts[metric] = metric_forecast
        
        # Generate ensemble forecast
        ensemble_forecast = await self._generate_ensemble_forecast(forecasts, historical_data)
        
        # Scenario modeling
        scenarios = {}
        if include_scenarios:
            scenarios = await self._generate_portfolio_scenarios(
                forecasts, forecast_horizon, target_metrics
            )
        
        # Risk assessment
        risk_assessment = await self._assess_forecast_risks(forecasts, scenarios)
        
        # Generate insights and recommendations
        insights = await self._generate_forecast_insights(forecasts, scenarios, risk_assessment)
        
        # Store forecast
        forecast_id = f"portfolio_forecast_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.active_forecasts[forecast_id] = {
            'type': 'portfolio',
            'horizon': forecast_horizon,
            'metrics': target_metrics,
            'forecasts': forecasts,
            'ensemble_forecast': ensemble_forecast,
            'scenarios': scenarios,
            'risk_assessment': risk_assessment,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        result = {
            'forecast_id': forecast_id,
            'forecast_horizon': forecast_horizon,
            'target_metrics': target_metrics,
            'confidence_level': confidence_level,
            'forecasts': forecasts,
            'ensemble_forecast': ensemble_forecast,
            'scenarios': scenarios,
            'risk_assessment': risk_assessment,
            'insights': insights,
            'forecast_accuracy_score': ensemble_forecast.get('accuracy_score', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Emit forecast completed event
        await self.emit_event('portfolio_forecast_completed', {
            'forecast_id': forecast_id,
            'horizon': forecast_horizon,
            'metrics_forecasted': len(target_metrics)
        })
        
        return result
    
    async def _forecast_brand_performance(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast specific brand performance"""
        
        brand_id = parameters.get('brand_id')
        brand_name = parameters.get('brand_name')
        forecast_horizon = parameters.get('forecast_horizon', 30)
        target_metrics = parameters.get('target_metrics', ['engagement_rate', 'conversion_rate', 'reach'])
        
        if not brand_id and not brand_name:
            raise ValueError("Either brand_id or brand_name must be provided")
        
        self.logger.info(f"Forecasting brand performance: {brand_name or brand_id}")
        
        # Get brand historical data
        brand_historical_data = await self._get_brand_historical_data(brand_id, brand_name, target_metrics)
        
        # Generate brand-specific forecasts
        brand_forecasts = {}
        for metric in target_metrics:
            if metric in brand_historical_data:
                forecast = await self._generate_metric_forecast(
                    brand_historical_data[metric], forecast_horizon, 0.90
                )
                brand_forecasts[metric] = forecast
        
        # Analyze brand-specific trends
        brand_trends = await self._analyze_brand_trends(brand_historical_data)
        
        # Generate brand scenarios
        brand_scenarios = await self._generate_brand_scenarios(brand_forecasts, brand_trends)
        
        return {
            'brand_id': brand_id,
            'brand_name': brand_name,
            'forecast_horizon': forecast_horizon,
            'target_metrics': target_metrics,
            'forecasts': brand_forecasts,
            'trends': brand_trends,
            'scenarios': brand_scenarios,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _forecast_project_performance(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast specific project performance"""
        
        project_id = parameters.get('project_id')
        project_name = parameters.get('project_name')
        forecast_horizon = parameters.get('forecast_horizon', 30)
        target_metrics = parameters.get('target_metrics', ['roi', 'efficiency', 'completion_rate'])
        
        if not project_id and not project_name:
            raise ValueError("Either project_id or project_name must be provided")
        
        self.logger.info(f"Forecasting project performance: {project_name or project_id}")
        
        # Get project historical data
        project_historical_data = await self._get_project_historical_data(project_id, project_name, target_metrics)
        
        # Generate project-specific forecasts
        project_forecasts = {}
        for metric in target_metrics:
            if metric in project_historical_data:
                forecast = await self._generate_metric_forecast(
                    project_historical_data[metric], forecast_horizon, 0.90
                )
                project_forecasts[metric] = forecast
        
        # Analyze project lifecycle stage
        lifecycle_analysis = await self._analyze_project_lifecycle(project_historical_data)
        
        # Generate project scenarios
        project_scenarios = await self._generate_project_scenarios(project_forecasts, lifecycle_analysis)
        
        return {
            'project_id': project_id,
            'project_name': project_name,
            'forecast_horizon': forecast_horizon,
            'target_metrics': target_metrics,
            'forecasts': project_forecasts,
            'lifecycle_analysis': lifecycle_analysis,
            'scenarios': project_scenarios,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _analyze_trends(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in portfolio data"""
        
        analysis_scope = parameters.get('scope', 'portfolio')  # portfolio, brand, project
        analysis_period = parameters.get('analysis_period', 90)
        target_metrics = parameters.get('target_metrics', self.config['forecast_metrics'])
        
        self.logger.info(f"Analyzing trends for {analysis_scope} over {analysis_period} days")
        
        # Get historical data based on scope
        if analysis_scope == 'portfolio':
            historical_data = await self._get_portfolio_historical_data(target_metrics, analysis_period)
        elif analysis_scope == 'brand':
            brand_id = parameters.get('brand_id')
            historical_data = await self._get_brand_historical_data(brand_id, None, target_metrics, analysis_period)
        else:  # project
            project_id = parameters.get('project_id')
            historical_data = await self._get_project_historical_data(project_id, None, target_metrics, analysis_period)
        
        # Perform trend analysis using trend analyzer
        if self.trend_analyzer:
            trend_analysis = self.trend_analyzer.analyze(
                analysis_type='trend_detection',
                data=historical_data,
                sensitivity=self.config['trend_detection_sensitivity']
            )
        else:
            trend_analysis = await self._simple_trend_analysis(historical_data)
        
        # Detect trend changes
        trend_changes = await self._detect_trend_changes(historical_data, trend_analysis)
        
        # Analyze trend strength and direction
        trend_strength = await self._analyze_trend_strength(trend_analysis)
        
        # Generate trend predictions
        trend_predictions = await self._generate_trend_predictions(trend_analysis, 30)  # 30-day prediction
        
        return {
            'analysis_scope': analysis_scope,
            'analysis_period': analysis_period,
            'target_metrics': target_metrics,
            'trend_analysis': trend_analysis,
            'trend_changes': trend_changes,
            'trend_strength': trend_strength,
            'trend_predictions': trend_predictions,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _model_scenarios(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Model different scenarios for portfolio performance"""
        
        scenario_types = parameters.get('scenario_types', self.config['scenario_types'])
        forecast_horizon = parameters.get('forecast_horizon', 90)
        target_metrics = parameters.get('target_metrics', self.config['forecast_metrics'])
        scenario_factors = parameters.get('scenario_factors', {})
        
        self.logger.info(f"Modeling {len(scenario_types)} scenarios for {forecast_horizon} days")
        
        # Get baseline forecast
        baseline_forecast = await self._get_baseline_forecast(target_metrics, forecast_horizon)
        
        # Generate scenarios
        scenarios = {}
        for scenario_type in scenario_types:
            scenario = await self._generate_scenario(
                scenario_type, baseline_forecast, scenario_factors, forecast_horizon
            )
            scenarios[scenario_type] = scenario
        
        # Compare scenarios
        scenario_comparison = await self._compare_scenarios(scenarios, baseline_forecast)
        
        # Analyze scenario probabilities
        scenario_probabilities = await self._analyze_scenario_probabilities(scenarios)
        
        # Generate scenario recommendations
        scenario_recommendations = await self._generate_scenario_recommendations(
            scenarios, scenario_comparison, scenario_probabilities
        )
        
        return {
            'scenario_types': scenario_types,
            'forecast_horizon': forecast_horizon,
            'target_metrics': target_metrics,
            'baseline_forecast': baseline_forecast,
            'scenarios': scenarios,
            'scenario_comparison': scenario_comparison,
            'scenario_probabilities': scenario_probabilities,
            'recommendations': scenario_recommendations,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _forecast_risks(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast portfolio risks"""
        
        risk_types = parameters.get('risk_types', ['volatility', 'downside', 'correlation', 'liquidity'])
        forecast_horizon = parameters.get('forecast_horizon', 30)
        confidence_level = parameters.get('confidence_level', 0.95)
        
        self.logger.info(f"Forecasting {len(risk_types)} risk types for {forecast_horizon} days")
        
        # Get historical risk data
        historical_risk_data = await self._get_historical_risk_data(risk_types)
        
        # Forecast each risk type
        risk_forecasts = {}
        for risk_type in risk_types:
            risk_forecast = await self._forecast_risk_type(
                risk_type, historical_risk_data[risk_type], forecast_horizon, confidence_level
            )
            risk_forecasts[risk_type] = risk_forecast
        
        # Calculate portfolio risk metrics
        portfolio_risk_metrics = await self._calculate_portfolio_risk_metrics(risk_forecasts)
        
        # Generate risk scenarios
        risk_scenarios = await self._generate_risk_scenarios(risk_forecasts, forecast_horizon)
        
        # Risk mitigation recommendations
        risk_mitigation = await self._generate_risk_mitigation_recommendations(
            risk_forecasts, portfolio_risk_metrics
        )
        
        return {
            'risk_types': risk_types,
            'forecast_horizon': forecast_horizon,
            'confidence_level': confidence_level,
            'risk_forecasts': risk_forecasts,
            'portfolio_risk_metrics': portfolio_risk_metrics,
            'risk_scenarios': risk_scenarios,
            'risk_mitigation': risk_mitigation,
            'overall_risk_score': portfolio_risk_metrics.get('overall_risk_score', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _analyze_seasonality(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seasonality patterns in portfolio data"""
        
        analysis_period = parameters.get('analysis_period', 365)  # 1 year
        target_metrics = parameters.get('target_metrics', self.config['forecast_metrics'])
        seasonality_types = parameters.get('seasonality_types', ['weekly', 'monthly', 'quarterly'])
        
        self.logger.info(f"Analyzing seasonality patterns over {analysis_period} days")
        
        # Get historical data
        historical_data = await self._get_portfolio_historical_data(target_metrics, analysis_period)
        
        # Analyze seasonality patterns
        seasonality_analysis = {}
        for metric in target_metrics:
            if metric in historical_data:
                metric_seasonality = await self._analyze_metric_seasonality(
                    historical_data[metric], seasonality_types
                )
                seasonality_analysis[metric] = metric_seasonality
        
        # Detect seasonal trends
        seasonal_trends = await self._detect_seasonal_trends(seasonality_analysis)
        
        # Generate seasonal forecasts
        seasonal_forecasts = await self._generate_seasonal_forecasts(seasonality_analysis, 90)  # 90-day forecast
        
        # Seasonal optimization recommendations
        seasonal_recommendations = await self._generate_seasonal_recommendations(
            seasonality_analysis, seasonal_trends
        )
        
        return {
            'analysis_period': analysis_period,
            'target_metrics': target_metrics,
            'seasonality_types': seasonality_types,
            'seasonality_analysis': seasonality_analysis,
            'seasonal_trends': seasonal_trends,
            'seasonal_forecasts': seasonal_forecasts,
            'recommendations': seasonal_recommendations,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _detect_anomalies(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in portfolio data"""
        
        detection_period = parameters.get('detection_period', 30)
        target_metrics = parameters.get('target_metrics', self.config['forecast_metrics'])
        sensitivity = parameters.get('sensitivity', 'medium')
        
        self.logger.info(f"Detecting anomalies over {detection_period} days")
        
        # Get recent data
        recent_data = await self._get_portfolio_historical_data(target_metrics, detection_period)
        
        # Detect anomalies for each metric
        anomalies = {}
        for metric in target_metrics:
            if metric in recent_data:
                metric_anomalies = await self._detect_metric_anomalies(
                    recent_data[metric], sensitivity
                )
                anomalies[metric] = metric_anomalies
        
        # Analyze anomaly patterns
        anomaly_patterns = await self._analyze_anomaly_patterns(anomalies)
        
        # Assess anomaly impact
        anomaly_impact = await self._assess_anomaly_impact(anomalies, recent_data)
        
        # Generate anomaly alerts
        anomaly_alerts = await self._generate_anomaly_alerts(anomalies, anomaly_impact)
        
        return {
            'detection_period': detection_period,
            'target_metrics': target_metrics,
            'sensitivity': sensitivity,
            'anomalies': anomalies,
            'anomaly_patterns': anomaly_patterns,
            'anomaly_impact': anomaly_impact,
            'alerts': anomaly_alerts,
            'total_anomalies_detected': sum(len(a.get('anomaly_points', [])) for a in anomalies.values()),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # Helper methods for data generation and analysis
    
    async def _get_portfolio_historical_data(self, target_metrics: List[str], 
                                           period_days: int = 90) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical portfolio data"""
        
        historical_data = {}
        base_date = datetime.utcnow() - timedelta(days=period_days)
        
        for metric in target_metrics:
            metric_data = []
            base_value = self._get_base_metric_value(metric)
            
            for i in range(period_days):
                date = base_date + timedelta(days=i)
                
                # Add trend, seasonality, and noise
                trend_factor = 1 + (i / period_days) * 0.1  # 10% growth over period
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly seasonality
                noise_factor = 1 + np.random.normal(0, 0.05)  # 5% noise
                
                value = base_value * trend_factor * seasonal_factor * noise_factor
                
                metric_data.append({
                    'date': date.isoformat(),
                    'value': max(value, 0),  # Ensure non-negative values
                    'trend_component': trend_factor,
                    'seasonal_component': seasonal_factor,
                    'noise_component': noise_factor
                })
            
            historical_data[metric] = metric_data
        
        return historical_data
    
    def _get_base_metric_value(self, metric: str) -> float:
        """Get base value for a metric"""
        
        base_values = {
            'roi': 3.0,
            'engagement_rate': 6.5,
            'conversion_rate': 4.2,
            'reach': 250000,
            'cost_efficiency': 0.75,
            'brand_awareness': 70.0
        }
        
        return base_values.get(metric, 1.0)
    
    async def _generate_metric_forecast(self, historical_data: List[Dict[str, Any]], 
                                      forecast_horizon: int, confidence_level: float) -> Dict[str, Any]:
        """Generate forecast for a specific metric"""
        
        # Extract values and dates
        values = [point['value'] for point in historical_data]
        dates = [datetime.fromisoformat(point['date']) for point in historical_data]
        
        # Generate forecasts using different models
        forecasts = {}
        
        # Linear regression forecast
        lr_forecast = await self._linear_regression_forecast(values, forecast_horizon)
        forecasts['linear_regression'] = lr_forecast
        
        # Random forest forecast
        rf_forecast = await self._random_forest_forecast(values, forecast_horizon)
        forecasts['random_forest'] = rf_forecast
        
        # Simple trend forecast
        trend_forecast = await self._trend_forecast(values, forecast_horizon)
        forecasts['trend'] = trend_forecast
        
        # Ensemble forecast
        ensemble_forecast = await self._create_ensemble_forecast(forecasts)
        
        # Calculate confidence intervals
        confidence_intervals = await self._calculate_confidence_intervals(
            ensemble_forecast, confidence_level
        )
        
        # Generate forecast dates
        last_date = dates[-1]
        forecast_dates = [
            (last_date + timedelta(days=i+1)).isoformat() 
            for i in range(forecast_horizon)
        ]
        
        return {
            'forecast_horizon': forecast_horizon,
            'confidence_level': confidence_level,
            'individual_forecasts': forecasts,
            'ensemble_forecast': ensemble_forecast,
            'confidence_intervals': confidence_intervals,
            'forecast_dates': forecast_dates,
            'forecast_accuracy_score': ensemble_forecast.get('accuracy_score', 0.8),
            'model_weights': self.config['ensemble_weights']
        }
    
    async def _linear_regression_forecast(self, values: List[float], horizon: int) -> Dict[str, Any]:
        """Generate linear regression forecast"""
        
        X = np.array(range(len(values))).reshape(-1, 1)
        y = np.array(values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast
        future_X = np.array(range(len(values), len(values) + horizon)).reshape(-1, 1)
        forecast_values = model.predict(future_X)
        
        return {
            'model_type': 'linear_regression',
            'forecast_values': forecast_values.tolist(),
            'model_score': model.score(X, y),
            'trend_slope': model.coef_[0],
            'intercept': model.intercept_
        }
    
    async def _random_forest_forecast(self, values: List[float], horizon: int) -> Dict[str, Any]:
        """Generate random forest forecast"""
        
        # Create features (lagged values)
        window_size = min(5, len(values) - 1)
        X, y = [], []
        
        for i in range(window_size, len(values)):
            X.append(values[i-window_size:i])
            y.append(values[i])
        
        if len(X) == 0:
            # Fallback to simple average
            avg_value = np.mean(values)
            return {
                'model_type': 'random_forest',
                'forecast_values': [avg_value] * horizon,
                'model_score': 0.5,
                'feature_importance': []
            }
        
        X, y = np.array(X), np.array(y)
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Forecast
        forecast_values = []
        current_window = values[-window_size:]
        
        for _ in range(horizon):
            next_value = model.predict([current_window])[0]
            forecast_values.append(next_value)
            current_window = current_window[1:] + [next_value]
        
        return {
            'model_type': 'random_forest',
            'forecast_values': forecast_values,
            'model_score': model.score(X, y),
            'feature_importance': model.feature_importances_.tolist()
        }
    
    async def _trend_forecast(self, values: List[float], horizon: int) -> Dict[str, Any]:
        """Generate simple trend-based forecast"""
        
        if len(values) < 2:
            avg_value = np.mean(values)
            return {
                'model_type': 'trend',
                'forecast_values': [avg_value] * horizon,
                'trend_direction': 'stable',
                'trend_strength': 0
            }
        
        # Calculate trend
        x = np.arange(len(values))
        trend_slope = np.polyfit(x, values, 1)[0]
        
        # Generate forecast
        last_value = values[-1]
        forecast_values = [last_value + trend_slope * (i + 1) for i in range(horizon)]
        
        return {
            'model_type': 'trend',
            'forecast_values': forecast_values,
            'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable',
            'trend_strength': abs(trend_slope),
            'trend_slope': trend_slope
        }
    
    async def _create_ensemble_forecast(self, forecasts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create ensemble forecast from individual forecasts"""
        
        weights = self.config['ensemble_weights']
        ensemble_values = []
        
        # Get forecast length
        forecast_length = len(list(forecasts.values())[0]['forecast_values'])
        
        for i in range(forecast_length):
            weighted_sum = 0
            total_weight = 0
            
            for model_name, forecast in forecasts.items():
                weight = weights.get(model_name, 0.25)  # Default weight
                if i < len(forecast['forecast_values']):
                    weighted_sum += weight * forecast['forecast_values'][i]
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_values.append(weighted_sum / total_weight)
            else:
                ensemble_values.append(0)
        
        # Calculate ensemble accuracy
        accuracy_scores = [f.get('model_score', 0.5) for f in forecasts.values()]
        ensemble_accuracy = np.mean(accuracy_scores)
        
        return {
            'model_type': 'ensemble',
            'forecast_values': ensemble_values,
            'accuracy_score': ensemble_accuracy,
            'component_models': list(forecasts.keys()),
            'model_weights': weights
        }
    
    async def _calculate_confidence_intervals(self, forecast: Dict[str, Any], 
                                            confidence_level: float) -> Dict[str, Any]:
        """Calculate confidence intervals for forecast"""
        
        forecast_values = forecast['forecast_values']
        accuracy_score = forecast.get('accuracy_score', 0.8)
        
        # Estimate prediction error based on accuracy
        prediction_error = (1 - accuracy_score) * 0.2  # 20% max error
        
        # Calculate confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 1.645 if confidence_level == 0.90 else 1.28
        
        lower_bounds = [value * (1 - z_score * prediction_error) for value in forecast_values]
        upper_bounds = [value * (1 + z_score * prediction_error) for value in forecast_values]
        
        return {
            'confidence_level': confidence_level,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'prediction_error': prediction_error,
            'z_score': z_score
        }
    
    # Message handlers
    
    async def _handle_forecast_request(self, sender_id: str, data: Dict[str, Any]):
        """Handle forecast requests from other agents"""
        
        forecast_type = data.get('forecast_type', 'portfolio_forecasting')
        parameters = data.get('parameters', {})
        priority = TaskPriority(data.get('priority', TaskPriority.MEDIUM.value))
        
        # Add forecast task to queue
        task_id = await self.add_task(forecast_type, parameters, priority)
        
        # Send acknowledgment
        await self.send_message(sender_id, 'forecast_request_acknowledged', {
            'task_id': task_id,
            'forecast_type': forecast_type,
            'estimated_completion': '20 minutes'
        })
    
    async def _handle_trend_analysis_request(self, sender_id: str, data: Dict[str, Any]):
        """Handle trend analysis requests"""
        
        # Perform immediate trend analysis
        trend_result = await self._analyze_trends(data)
        
        # Send results back
        await self.send_message(sender_id, 'trend_analysis_result', trend_result)
    
    async def _handle_scenario_modeling_request(self, sender_id: str, data: Dict[str, Any]):
        """Handle scenario modeling requests"""
        
        # Add scenario modeling task
        task_id = await self.add_task('scenario_modeling', data, TaskPriority.MEDIUM)
        
        await self.send_message(sender_id, 'scenario_modeling_acknowledged', {
            'task_id': task_id,
            'estimated_completion': '25 minutes'
        })
    
    async def _handle_risk_forecast_request(self, sender_id: str, data: Dict[str, Any]):
        """Handle risk forecasting requests"""
        
        # Add risk forecasting task
        task_id = await self.add_task('risk_forecasting', data, TaskPriority.HIGH)
        
        await self.send_message(sender_id, 'risk_forecast_acknowledged', {
            'task_id': task_id,
            'estimated_completion': '30 minutes'
        })
    
    async def _handle_market_update(self, sender_id: str, data: Dict[str, Any]):
        """Handle market factor updates"""
        
        market_factor = data.get('market_factor')
        factor_value = data.get('factor_value')
        
        if market_factor:
            self.market_factors[market_factor] = {
                'value': factor_value,
                'timestamp': datetime.utcnow().isoformat(),
                'source': sender_id
            }
            
            self.logger.info(f"Updated market factor {market_factor}: {factor_value}")

# Factory function for creating portfolio forecasting agent
def create_portfolio_forecasting_agent(config: Dict[str, Any] = None) -> PortfolioForecastingAgent:
    """Create Portfolio Forecasting Agent with specified configuration"""
    return PortfolioForecastingAgent(config=config)

