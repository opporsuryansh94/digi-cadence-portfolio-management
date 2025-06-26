"""
Portfolio Performance Report Generator for Digi-Cadence Portfolio Management Platform
Comprehensive portfolio performance analysis with multi-brand and multi-project support
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.reports.base_generator import BaseReportGenerator, ReportType

class PortfolioPerformanceGenerator(BaseReportGenerator):
    """
    Portfolio Performance Report Generator
    Provides comprehensive analysis of portfolio performance across brands and projects
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(ReportType.PORTFOLIO_PERFORMANCE, config)
        
        # Portfolio-specific configuration
        self.portfolio_config = {
            'performance_metrics': [
                'brand_awareness', 'brand_consideration', 'brand_preference',
                'purchase_intent', 'customer_satisfaction', 'net_promoter_score',
                'market_share', 'revenue', 'roi', 'engagement_rate'
            ],
            'benchmark_percentiles': [25, 50, 75, 90],
            'performance_thresholds': {
                'excellent': 0.9,
                'good': 0.7,
                'average': 0.5,
                'poor': 0.3
            },
            'trend_analysis_periods': [7, 30, 90],
            'correlation_threshold': 0.7,
            'volatility_threshold': 0.3,
            'growth_rate_threshold': 0.05
        }
        
        self.config.update(self.portfolio_config)
    
    async def generate_report(self, 
                            organization_ids: List[str],
                            project_ids: List[str] = None,
                            brand_ids: List[str] = None,
                            date_range: Tuple[datetime, datetime] = None,
                            parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive portfolio performance report"""
        
        start_time = datetime.utcnow()
        self.report_metadata['created_at'] = start_time.isoformat()
        
        # Prepare data
        data = await self.prepare_data(organization_ids, project_ids, brand_ids, date_range)
        
        # Portfolio performance analysis
        portfolio_analysis = await self._analyze_portfolio_performance(data)
        
        # Brand performance analysis
        brand_analysis = await self._analyze_brand_performance(data)
        
        # Project performance analysis
        project_analysis = await self._analyze_project_performance(data)
        
        # Cross-dimensional analysis
        cross_analysis = await self._analyze_cross_dimensional_performance(data)
        
        # Performance benchmarking
        benchmarking = await self._perform_benchmarking_analysis(data)
        
        # Trend analysis
        trend_analysis = await self._analyze_performance_trends(data)
        
        # Risk analysis
        risk_analysis = await self._analyze_performance_risks(data)
        
        # Generate visualizations
        visualizations = await self.generate_visualizations(data, [
            'portfolio_overview', 'brand_performance_matrix', 'project_comparison',
            'performance_trends', 'correlation_analysis', 'risk_assessment'
        ])
        
        # Generate insights
        insights = await self.generate_insights(data)
        
        # Add portfolio-specific insights
        portfolio_insights = await self._generate_portfolio_insights(
            portfolio_analysis, brand_analysis, project_analysis, cross_analysis
        )
        insights.extend(portfolio_insights)
        
        # Generate recommendations
        recommendations = await self.generate_recommendations(data, insights)
        
        # Add portfolio-specific recommendations
        portfolio_recommendations = await self._generate_portfolio_recommendations(
            portfolio_analysis, brand_analysis, project_analysis, risk_analysis
        )
        recommendations.extend(portfolio_recommendations)
        
        # Executive summary
        executive_summary = await self._generate_executive_summary(
            portfolio_analysis, brand_analysis, project_analysis, insights, recommendations
        )
        
        # Performance scoring
        performance_scores = await self._calculate_performance_scores(
            portfolio_analysis, brand_analysis, project_analysis
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        self.report_metadata['processing_time'] = processing_time
        
        return {
            'report_type': self.report_type.value,
            'metadata': self.report_metadata,
            'executive_summary': executive_summary,
            'data_summary': data['data_summary'],
            'portfolio_analysis': portfolio_analysis,
            'brand_analysis': brand_analysis,
            'project_analysis': project_analysis,
            'cross_analysis': cross_analysis,
            'benchmarking': benchmarking,
            'trend_analysis': trend_analysis,
            'risk_analysis': risk_analysis,
            'performance_scores': performance_scores,
            'insights': insights,
            'recommendations': recommendations,
            'visualizations': visualizations,
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def _analyze_portfolio_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall portfolio performance"""
        
        metrics_data = data['metrics_data']
        
        # Calculate portfolio-level aggregations
        portfolio_metrics = {}
        
        for metric in self.config['performance_metrics']:
            metric_values = []
            for brand_id, brand_metrics in metrics_data.items():
                if metric in brand_metrics:
                    metric_values.extend(brand_metrics[metric])
            
            if metric_values:
                portfolio_metrics[metric] = {
                    'mean': np.mean(metric_values),
                    'median': np.median(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values),
                    'percentiles': {
                        str(p): np.percentile(metric_values, p) 
                        for p in self.config['benchmark_percentiles']
                    }
                }
        
        # Calculate portfolio health score
        health_score = await self._calculate_portfolio_health_score(portfolio_metrics)
        
        # Portfolio diversity analysis
        diversity_analysis = await self._analyze_portfolio_diversity(data)
        
        # Portfolio balance analysis
        balance_analysis = await self._analyze_portfolio_balance(data)
        
        # Portfolio efficiency analysis
        efficiency_analysis = await self._analyze_portfolio_efficiency(data)
        
        return {
            'portfolio_metrics': portfolio_metrics,
            'health_score': health_score,
            'diversity_analysis': diversity_analysis,
            'balance_analysis': balance_analysis,
            'efficiency_analysis': efficiency_analysis,
            'total_brands': len(data['brands']),
            'total_projects': len(data['projects']),
            'total_organizations': len(data['organizations'])
        }
    
    async def _analyze_brand_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual brand performance"""
        
        brand_performance = {}
        metrics_data = data['metrics_data']
        
        for brand_id, brand_metrics in metrics_data.items():
            brand_info = data['brands'][brand_id]
            
            # Calculate brand-level metrics
            brand_stats = {}
            for metric in self.config['performance_metrics']:
                if metric in brand_metrics:
                    values = brand_metrics[metric]
                    brand_stats[metric] = {
                        'current_value': values[-1] if values else 0,
                        'average_value': np.mean(values),
                        'trend': self._calculate_trend(values),
                        'volatility': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,
                        'growth_rate': self._calculate_growth_rate(values)
                    }
            
            # Brand performance score
            performance_score = await self._calculate_brand_performance_score(brand_stats)
            
            # Brand ranking
            brand_ranking = await self._calculate_brand_ranking(brand_id, metrics_data)
            
            # Brand efficiency metrics
            efficiency_metrics = await self._calculate_brand_efficiency(brand_metrics)
            
            brand_performance[brand_id] = {
                'brand_info': brand_info,
                'performance_metrics': brand_stats,
                'performance_score': performance_score,
                'ranking': brand_ranking,
                'efficiency_metrics': efficiency_metrics,
                'performance_category': self._categorize_performance(performance_score)
            }
        
        return brand_performance
    
    async def _analyze_project_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project-level performance"""
        
        project_performance = {}
        
        # Group brands by project
        project_brands = {}
        for brand_id, brand_info in data['brands'].items():
            project_id = brand_info['project_id']
            if project_id not in project_brands:
                project_brands[project_id] = []
            project_brands[project_id].append(brand_id)
        
        # Analyze each project
        for project_id, brand_ids in project_brands.items():
            project_info = data['projects'][project_id]
            
            # Aggregate metrics across project brands
            project_metrics = {}
            for metric in self.config['performance_metrics']:
                metric_values = []
                for brand_id in brand_ids:
                    if brand_id in data['metrics_data'] and metric in data['metrics_data'][brand_id]:
                        metric_values.extend(data['metrics_data'][brand_id][metric])
                
                if metric_values:
                    project_metrics[metric] = {
                        'average': np.mean(metric_values),
                        'total': np.sum(metric_values) if metric in ['revenue'] else np.mean(metric_values),
                        'best_brand': max(brand_ids, key=lambda b: np.mean(data['metrics_data'][b][metric]) if b in data['metrics_data'] and metric in data['metrics_data'][b] else 0),
                        'worst_brand': min(brand_ids, key=lambda b: np.mean(data['metrics_data'][b][metric]) if b in data['metrics_data'] and metric in data['metrics_data'][b] else 0)
                    }
            
            # Project performance score
            performance_score = await self._calculate_project_performance_score(project_metrics)
            
            # Project brand synergy analysis
            synergy_analysis = await self._analyze_project_brand_synergy(brand_ids, data['metrics_data'])
            
            project_performance[project_id] = {
                'project_info': project_info,
                'brand_count': len(brand_ids),
                'brand_ids': brand_ids,
                'performance_metrics': project_metrics,
                'performance_score': performance_score,
                'synergy_analysis': synergy_analysis,
                'performance_category': self._categorize_performance(performance_score)
            }
        
        return project_performance
    
    async def _analyze_cross_dimensional_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance across multiple dimensions"""
        
        # Cross-brand correlation analysis
        brand_correlations = await self._analyze_brand_correlations(data['metrics_data'])
        
        # Cross-project performance comparison
        project_comparison = await self._compare_project_performance(data)
        
        # Organization-level analysis
        organization_analysis = await self._analyze_organization_performance(data)
        
        # Performance distribution analysis
        distribution_analysis = await self._analyze_performance_distribution(data)
        
        # Synergy opportunity analysis
        synergy_opportunities = await self._identify_synergy_opportunities(data)
        
        return {
            'brand_correlations': brand_correlations,
            'project_comparison': project_comparison,
            'organization_analysis': organization_analysis,
            'distribution_analysis': distribution_analysis,
            'synergy_opportunities': synergy_opportunities
        }
    
    async def _perform_benchmarking_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform benchmarking analysis"""
        
        benchmarks = {}
        metrics_data = data['metrics_data']
        
        # Calculate industry benchmarks (simulated)
        industry_benchmarks = {
            'brand_awareness': {'excellent': 0.8, 'good': 0.6, 'average': 0.4, 'poor': 0.2},
            'brand_consideration': {'excellent': 0.6, 'good': 0.4, 'average': 0.25, 'poor': 0.1},
            'revenue': {'excellent': 800000, 'good': 500000, 'average': 300000, 'poor': 100000},
            'roi': {'excellent': 4.0, 'good': 3.0, 'average': 2.0, 'poor': 1.0}
        }
        
        # Compare each brand against benchmarks
        brand_benchmarks = {}
        for brand_id, brand_metrics in metrics_data.items():
            brand_benchmarks[brand_id] = {}
            
            for metric, values in brand_metrics.items():
                if metric in industry_benchmarks and values:
                    current_value = np.mean(values)
                    benchmark_levels = industry_benchmarks[metric]
                    
                    # Determine performance level
                    if current_value >= benchmark_levels['excellent']:
                        level = 'excellent'
                    elif current_value >= benchmark_levels['good']:
                        level = 'good'
                    elif current_value >= benchmark_levels['average']:
                        level = 'average'
                    else:
                        level = 'poor'
                    
                    brand_benchmarks[brand_id][metric] = {
                        'current_value': current_value,
                        'benchmark_level': level,
                        'gap_to_excellent': benchmark_levels['excellent'] - current_value,
                        'percentile_rank': self._calculate_percentile_rank(current_value, values)
                    }
        
        # Portfolio-level benchmarking
        portfolio_benchmarks = await self._calculate_portfolio_benchmarks(brand_benchmarks)
        
        return {
            'industry_benchmarks': industry_benchmarks,
            'brand_benchmarks': brand_benchmarks,
            'portfolio_benchmarks': portfolio_benchmarks
        }
    
    async def _analyze_performance_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends"""
        
        trend_analysis = {}
        metrics_data = data['metrics_data']
        
        for period in self.config['trend_analysis_periods']:
            period_analysis = {}
            
            for brand_id, brand_metrics in metrics_data.items():
                brand_trends = {}
                
                for metric, values in brand_metrics.items():
                    if len(values) >= period:
                        recent_values = values[-period:]
                        
                        # Calculate trend metrics
                        trend_slope, trend_r2 = self._calculate_trend_metrics(recent_values)
                        
                        brand_trends[metric] = {
                            'trend_direction': 'up' if trend_slope > 0 else 'down' if trend_slope < 0 else 'stable',
                            'trend_strength': abs(trend_slope),
                            'trend_r_squared': trend_r2,
                            'period_change': (recent_values[-1] - recent_values[0]) / recent_values[0] if recent_values[0] != 0 else 0,
                            'volatility': np.std(recent_values)
                        }
                
                period_analysis[brand_id] = brand_trends
            
            trend_analysis[f'{period}_day_trends'] = period_analysis
        
        # Overall trend summary
        trend_summary = await self._summarize_trends(trend_analysis)
        
        return {
            'trend_analysis': trend_analysis,
            'trend_summary': trend_summary
        }
    
    async def _analyze_performance_risks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance risks"""
        
        risk_analysis = {}
        metrics_data = data['metrics_data']
        
        for brand_id, brand_metrics in metrics_data.items():
            brand_risks = {}
            
            for metric, values in brand_metrics.items():
                if values:
                    # Volatility risk
                    volatility = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                    volatility_risk = 'high' if volatility > self.config['volatility_threshold'] else 'medium' if volatility > self.config['volatility_threshold'] / 2 else 'low'
                    
                    # Trend risk
                    trend_slope, _ = self._calculate_trend_metrics(values)
                    trend_risk = 'high' if trend_slope < -0.01 else 'medium' if trend_slope < 0 else 'low'
                    
                    # Performance risk
                    current_value = values[-1]
                    average_value = np.mean(values)
                    performance_risk = 'high' if current_value < average_value * 0.8 else 'medium' if current_value < average_value * 0.9 else 'low'
                    
                    brand_risks[metric] = {
                        'volatility_risk': volatility_risk,
                        'trend_risk': trend_risk,
                        'performance_risk': performance_risk,
                        'overall_risk': self._calculate_overall_risk([volatility_risk, trend_risk, performance_risk])
                    }
            
            risk_analysis[brand_id] = brand_risks
        
        # Portfolio risk summary
        portfolio_risk = await self._calculate_portfolio_risk(risk_analysis)
        
        return {
            'brand_risks': risk_analysis,
            'portfolio_risk': portfolio_risk
        }
    
    # Helper methods
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        slope, _ = self._calculate_trend_metrics(values)
        
        if slope > 0.01:
            return 'up'
        elif slope < -0.01:
            return 'down'
        else:
            return 'stable'
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate growth rate"""
        if len(values) < 2:
            return 0.0
        
        return (values[-1] - values[0]) / values[0] if values[0] != 0 else 0.0
    
    def _calculate_trend_metrics(self, values: List[float]) -> Tuple[float, float]:
        """Calculate trend slope and R-squared"""
        if len(values) < 2:
            return 0.0, 0.0
        
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        return slope, r_value ** 2
    
    def _categorize_performance(self, score: float) -> str:
        """Categorize performance based on score"""
        thresholds = self.config['performance_thresholds']
        
        if score >= thresholds['excellent']:
            return 'excellent'
        elif score >= thresholds['good']:
            return 'good'
        elif score >= thresholds['average']:
            return 'average'
        else:
            return 'poor'
    
    def _calculate_overall_risk(self, risk_levels: List[str]) -> str:
        """Calculate overall risk level"""
        risk_scores = {'low': 1, 'medium': 2, 'high': 3}
        avg_score = np.mean([risk_scores[level] for level in risk_levels])
        
        if avg_score >= 2.5:
            return 'high'
        elif avg_score >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_percentile_rank(self, value: float, values: List[float]) -> float:
        """Calculate percentile rank of value in list"""
        return stats.percentileofscore(values, value)
    
    # Placeholder methods for complex calculations
    
    async def _calculate_portfolio_health_score(self, portfolio_metrics: Dict[str, Any]) -> float:
        """Calculate overall portfolio health score"""
        # Simplified calculation - weight different metrics
        weights = {
            'brand_awareness': 0.15,
            'revenue': 0.25,
            'roi': 0.20,
            'customer_satisfaction': 0.15,
            'market_share': 0.15,
            'engagement_rate': 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in portfolio_metrics:
                # Normalize score to 0-1 range
                normalized_score = min(portfolio_metrics[metric]['mean'] / 1.0, 1.0)
                weighted_score += normalized_score * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    async def _calculate_brand_performance_score(self, brand_stats: Dict[str, Any]) -> float:
        """Calculate brand performance score"""
        # Simplified scoring based on multiple metrics
        scores = []
        
        for metric, stats in brand_stats.items():
            if 'average_value' in stats:
                # Normalize based on metric type
                if metric in ['brand_awareness', 'brand_consideration', 'brand_preference']:
                    normalized = min(stats['average_value'], 1.0)
                elif metric == 'roi':
                    normalized = min(stats['average_value'] / 5.0, 1.0)  # Assume max ROI of 5
                elif metric == 'revenue':
                    normalized = min(stats['average_value'] / 1000000, 1.0)  # Assume max revenue of 1M
                else:
                    normalized = min(stats['average_value'], 1.0)
                
                scores.append(normalized)
        
        return np.mean(scores) if scores else 0.0
    
    async def _generate_portfolio_insights(self, portfolio_analysis: Dict[str, Any],
                                         brand_analysis: Dict[str, Any],
                                         project_analysis: Dict[str, Any],
                                         cross_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate portfolio-specific insights"""
        
        insights = []
        
        # Portfolio health insight
        health_score = portfolio_analysis['health_score']
        insights.append({
            'type': 'portfolio_health',
            'title': f'Portfolio Health Score: {health_score:.2f}',
            'description': f'Overall portfolio health is {"excellent" if health_score > 0.8 else "good" if health_score > 0.6 else "needs improvement"}',
            'importance_score': 0.9,
            'confidence': 0.95,
            'impact': 'high',
            'category': 'portfolio'
        })
        
        # Top performing brand insight
        best_brand = max(brand_analysis.items(), key=lambda x: x[1]['performance_score'])
        insights.append({
            'type': 'top_performer',
            'title': f'Top Performing Brand: {best_brand[1]["brand_info"]["name"]}',
            'description': f'Achieves {best_brand[1]["performance_score"]:.2f} performance score',
            'importance_score': 0.8,
            'confidence': 0.9,
            'impact': 'medium',
            'category': 'brand'
        })
        
        return insights
    
    async def _generate_portfolio_recommendations(self, portfolio_analysis: Dict[str, Any],
                                                brand_analysis: Dict[str, Any],
                                                project_analysis: Dict[str, Any],
                                                risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate portfolio-specific recommendations"""
        
        recommendations = []
        
        # Portfolio optimization recommendation
        recommendations.append({
            'type': 'portfolio_optimization',
            'title': 'Optimize Portfolio Balance',
            'description': 'Reallocate resources from underperforming brands to high-potential opportunities',
            'priority_score': 0.85,
            'impact': 'high',
            'effort': 'medium',
            'timeline': '3-6 months',
            'category': 'optimization'
        })
        
        # Risk mitigation recommendation
        high_risk_brands = [brand_id for brand_id, risks in risk_analysis['brand_risks'].items() 
                           if any(risk.get('overall_risk') == 'high' for risk in risks.values())]
        
        if high_risk_brands:
            recommendations.append({
                'type': 'risk_mitigation',
                'title': 'Address High-Risk Brands',
                'description': f'Implement risk mitigation strategies for {len(high_risk_brands)} high-risk brands',
                'priority_score': 0.9,
                'impact': 'high',
                'effort': 'high',
                'timeline': '1-3 months',
                'category': 'risk'
            })
        
        return recommendations
    
    async def _generate_executive_summary(self, portfolio_analysis: Dict[str, Any],
                                        brand_analysis: Dict[str, Any],
                                        project_analysis: Dict[str, Any],
                                        insights: List[Dict[str, Any]],
                                        recommendations: List[Dict[str, Any]]) -> str:
        """Generate executive summary"""
        
        health_score = portfolio_analysis['health_score']
        total_brands = portfolio_analysis['total_brands']
        total_projects = portfolio_analysis['total_projects']
        
        top_insights = [insight['title'] for insight in insights[:3]]
        top_recommendations = [rec['title'] for rec in recommendations[:3]]
        
        summary = f"""
Portfolio Performance Executive Summary

Portfolio Health Score: {health_score:.2f}/1.00 ({self._categorize_performance(health_score).title()})

Portfolio Overview:
- {total_brands} brands across {total_projects} projects analyzed
- Portfolio demonstrates {"strong" if health_score > 0.7 else "moderate" if health_score > 0.5 else "weak"} overall performance

Key Insights:
{chr(10).join(f"• {insight}" for insight in top_insights)}

Priority Recommendations:
{chr(10).join(f"• {rec}" for rec in top_recommendations)}

The portfolio shows {"promising" if health_score > 0.6 else "concerning"} performance trends with opportunities for optimization and growth.
        """.strip()
        
        return summary

# Factory function
def create_portfolio_performance_generator(config: Dict[str, Any] = None) -> PortfolioPerformanceGenerator:
    """Create Portfolio Performance Report Generator"""
    return PortfolioPerformanceGenerator(config)

