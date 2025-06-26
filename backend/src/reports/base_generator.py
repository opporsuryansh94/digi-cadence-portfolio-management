"""
Base Report Generator for Digi-Cadence Portfolio Management Platform
Provides foundation for all 16 report types with multi-brand and multi-project support
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import json
from enum import Enum
from abc import ABC, abstractmethod
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

from src.models.portfolio import Organization, Project, Brand, BrandMetric, AnalyticsResult

class ReportType(Enum):
    """Report type enumeration"""
    PORTFOLIO_PERFORMANCE = "portfolio_performance"
    BRAND_EQUITY_ANALYSIS = "brand_equity_analysis"
    COMPETITIVE_INTELLIGENCE = "competitive_intelligence"
    DIGITAL_MARKETING_EFFECTIVENESS = "digital_marketing_effectiveness"
    CROSS_BRAND_SYNERGY = "cross_brand_synergy"
    MARKET_OPPORTUNITY_ANALYSIS = "market_opportunity_analysis"
    CUSTOMER_JOURNEY_ANALYSIS = "customer_journey_analysis"
    CAMPAIGN_PERFORMANCE = "campaign_performance"
    ATTRIBUTION_ANALYSIS = "attribution_analysis"
    PREDICTIVE_INSIGHTS = "predictive_insights"
    RISK_ASSESSMENT = "risk_assessment"
    ROI_OPTIMIZATION = "roi_optimization"
    TREND_ANALYSIS = "trend_analysis"
    CROSS_PROJECT_BRAND_EVOLUTION = "cross_project_brand_evolution"
    STRATEGIC_RECOMMENDATIONS = "strategic_recommendations"
    EXECUTIVE_SUMMARY = "executive_summary"

class ReportFormat(Enum):
    """Report format enumeration"""
    JSON = "json"
    PDF = "pdf"
    EXCEL = "excel"
    HTML = "html"
    CSV = "csv"

class ReportFrequency(Enum):
    """Report frequency enumeration"""
    REAL_TIME = "real_time"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    ON_DEMAND = "on_demand"

class BaseReportGenerator(ABC):
    """
    Base class for all report generators
    Provides common functionality for multi-brand and multi-project reporting
    """
    
    def __init__(self, report_type: ReportType, config: Dict[str, Any] = None):
        self.report_type = report_type
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Default configuration
        self.default_config = {
            'include_visualizations': True,
            'include_insights': True,
            'include_recommendations': True,
            'confidence_threshold': 0.8,
            'significance_level': 0.05,
            'max_brands_per_chart': 10,
            'max_projects_per_analysis': 20,
            'date_range_days': 90,
            'comparison_periods': 3,
            'export_formats': ['json', 'pdf'],
            'chart_theme': 'plotly_white',
            'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'font_family': 'Arial, sans-serif',
            'chart_height': 400,
            'chart_width': 800
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Report metadata
        self.report_metadata = {
            'report_type': report_type.value,
            'generator_class': self.__class__.__name__,
            'version': '1.0.0',
            'created_at': None,
            'generated_by': None,
            'data_sources': [],
            'processing_time': None
        }
        
        # Data cache
        self.data_cache = {}
        self.visualization_cache = {}
        
        self.logger.info(f"Initialized {report_type.value} report generator")
    
    @abstractmethod
    async def generate_report(self, 
                            organization_ids: List[str],
                            project_ids: List[str] = None,
                            brand_ids: List[str] = None,
                            date_range: Tuple[datetime, datetime] = None,
                            parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate report for specified organizations, projects, and brands
        
        Args:
            organization_ids: List of organization IDs
            project_ids: List of project IDs (optional, all if None)
            brand_ids: List of brand IDs (optional, all if None)
            date_range: Date range tuple (start, end)
            parameters: Additional parameters for report generation
            
        Returns:
            Complete report data structure
        """
        pass
    
    async def prepare_data(self,
                          organization_ids: List[str],
                          project_ids: List[str] = None,
                          brand_ids: List[str] = None,
                          date_range: Tuple[datetime, datetime] = None) -> Dict[str, Any]:
        """Prepare data for report generation"""
        
        start_time = datetime.utcnow()
        
        # Set default date range if not provided
        if not date_range:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=self.config['date_range_days'])
            date_range = (start_date, end_date)
        
        # Load organizations
        organizations = await self._load_organizations(organization_ids)
        
        # Load projects
        if project_ids:
            projects = await self._load_projects(project_ids, organization_ids)
        else:
            projects = await self._load_all_projects(organization_ids)
        
        # Load brands
        if brand_ids:
            brands = await self._load_brands(brand_ids, [p.id for p in projects])
        else:
            brands = await self._load_all_brands([p.id for p in projects])
        
        # Load metrics data
        metrics_data = await self._load_metrics_data(
            [b.id for b in brands], date_range
        )
        
        # Load analytics results
        analytics_data = await self._load_analytics_data(
            organization_ids, [p.id for p in projects], [b.id for b in brands], date_range
        )
        
        # Prepare data structure
        prepared_data = {
            'organizations': {org.id: self._serialize_organization(org) for org in organizations},
            'projects': {proj.id: self._serialize_project(proj) for proj in projects},
            'brands': {brand.id: self._serialize_brand(brand) for brand in brands},
            'metrics_data': metrics_data,
            'analytics_data': analytics_data,
            'date_range': {
                'start': date_range[0].isoformat(),
                'end': date_range[1].isoformat()
            },
            'data_summary': {
                'organizations_count': len(organizations),
                'projects_count': len(projects),
                'brands_count': len(brands),
                'metrics_count': len(metrics_data),
                'analytics_results_count': len(analytics_data)
            }
        }
        
        # Cache data
        cache_key = f"{'-'.join(organization_ids)}_{'-'.join([p.id for p in projects])}_{'-'.join([b.id for b in brands])}"
        self.data_cache[cache_key] = prepared_data
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        self.logger.info(f"Data preparation completed in {processing_time:.2f} seconds")
        
        return prepared_data
    
    async def generate_visualizations(self, data: Dict[str, Any], 
                                    visualization_types: List[str] = None) -> Dict[str, Any]:
        """Generate visualizations for the report"""
        
        if not self.config['include_visualizations']:
            return {}
        
        visualizations = {}
        
        # Default visualization types based on report type
        if not visualization_types:
            visualization_types = self._get_default_visualizations()
        
        for viz_type in visualization_types:
            try:
                if viz_type == 'performance_trends':
                    visualizations[viz_type] = await self._create_performance_trends_chart(data)
                elif viz_type == 'brand_comparison':
                    visualizations[viz_type] = await self._create_brand_comparison_chart(data)
                elif viz_type == 'project_overview':
                    visualizations[viz_type] = await self._create_project_overview_chart(data)
                elif viz_type == 'correlation_matrix':
                    visualizations[viz_type] = await self._create_correlation_matrix(data)
                elif viz_type == 'distribution_analysis':
                    visualizations[viz_type] = await self._create_distribution_chart(data)
                elif viz_type == 'time_series':
                    visualizations[viz_type] = await self._create_time_series_chart(data)
                elif viz_type == 'heatmap':
                    visualizations[viz_type] = await self._create_heatmap(data)
                elif viz_type == 'scatter_plot':
                    visualizations[viz_type] = await self._create_scatter_plot(data)
                else:
                    self.logger.warning(f"Unknown visualization type: {viz_type}")
                    
            except Exception as e:
                self.logger.error(f"Error creating {viz_type} visualization: {e}")
                continue
        
        return visualizations
    
    async def generate_insights(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from the data"""
        
        if not self.config['include_insights']:
            return []
        
        insights = []
        
        # Performance insights
        performance_insights = await self._analyze_performance_patterns(data)
        insights.extend(performance_insights)
        
        # Trend insights
        trend_insights = await self._analyze_trends(data)
        insights.extend(trend_insights)
        
        # Correlation insights
        correlation_insights = await self._analyze_correlations(data)
        insights.extend(correlation_insights)
        
        # Anomaly insights
        anomaly_insights = await self._detect_anomalies(data)
        insights.extend(anomaly_insights)
        
        # Cross-brand insights
        cross_brand_insights = await self._analyze_cross_brand_patterns(data)
        insights.extend(cross_brand_insights)
        
        # Cross-project insights
        cross_project_insights = await self._analyze_cross_project_patterns(data)
        insights.extend(cross_project_insights)
        
        # Sort insights by importance
        insights.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        
        return insights
    
    async def generate_recommendations(self, data: Dict[str, Any], 
                                     insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on data and insights"""
        
        if not self.config['include_recommendations']:
            return []
        
        recommendations = []
        
        # Performance optimization recommendations
        perf_recommendations = await self._generate_performance_recommendations(data, insights)
        recommendations.extend(perf_recommendations)
        
        # Budget allocation recommendations
        budget_recommendations = await self._generate_budget_recommendations(data, insights)
        recommendations.extend(budget_recommendations)
        
        # Strategic recommendations
        strategic_recommendations = await self._generate_strategic_recommendations(data, insights)
        recommendations.extend(strategic_recommendations)
        
        # Risk mitigation recommendations
        risk_recommendations = await self._generate_risk_recommendations(data, insights)
        recommendations.extend(risk_recommendations)
        
        # Cross-brand synergy recommendations
        synergy_recommendations = await self._generate_synergy_recommendations(data, insights)
        recommendations.extend(synergy_recommendations)
        
        # Prioritize recommendations
        recommendations.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        return recommendations
    
    async def export_report(self, report_data: Dict[str, Any], 
                          format: ReportFormat, 
                          output_path: str = None) -> str:
        """Export report in specified format"""
        
        if format == ReportFormat.JSON:
            return await self._export_json(report_data, output_path)
        elif format == ReportFormat.PDF:
            return await self._export_pdf(report_data, output_path)
        elif format == ReportFormat.EXCEL:
            return await self._export_excel(report_data, output_path)
        elif format == ReportFormat.HTML:
            return await self._export_html(report_data, output_path)
        elif format == ReportFormat.CSV:
            return await self._export_csv(report_data, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    # Helper methods for data loading
    
    async def _load_organizations(self, organization_ids: List[str]) -> List[Organization]:
        """Load organizations from database"""
        # Simulated data loading - replace with actual database queries
        organizations = []
        for org_id in organization_ids:
            org = Organization(
                id=org_id,
                name=f"Organization {org_id}",
                industry="Technology",
                created_at=datetime.utcnow()
            )
            organizations.append(org)
        return organizations
    
    async def _load_projects(self, project_ids: List[str], 
                           organization_ids: List[str]) -> List[Project]:
        """Load specific projects from database"""
        # Simulated data loading
        projects = []
        for proj_id in project_ids:
            project = Project(
                id=proj_id,
                name=f"Project {proj_id}",
                organization_id=organization_ids[0],
                status="active",
                created_at=datetime.utcnow()
            )
            projects.append(project)
        return projects
    
    async def _load_all_projects(self, organization_ids: List[str]) -> List[Project]:
        """Load all projects for organizations"""
        # Simulated data loading
        projects = []
        for i, org_id in enumerate(organization_ids):
            for j in range(3):  # 3 projects per organization
                project = Project(
                    id=f"proj_{org_id}_{j}",
                    name=f"Project {j+1} - Org {org_id}",
                    organization_id=org_id,
                    status="active",
                    created_at=datetime.utcnow()
                )
                projects.append(project)
        return projects
    
    async def _load_brands(self, brand_ids: List[str], 
                         project_ids: List[str]) -> List[Brand]:
        """Load specific brands from database"""
        # Simulated data loading
        brands = []
        for brand_id in brand_ids:
            brand = Brand(
                id=brand_id,
                name=f"Brand {brand_id}",
                project_id=project_ids[0] if project_ids else "default_project",
                category="Consumer",
                created_at=datetime.utcnow()
            )
            brands.append(brand)
        return brands
    
    async def _load_all_brands(self, project_ids: List[str]) -> List[Brand]:
        """Load all brands for projects"""
        # Simulated data loading
        brands = []
        for i, proj_id in enumerate(project_ids):
            for j in range(2):  # 2 brands per project
                brand = Brand(
                    id=f"brand_{proj_id}_{j}",
                    name=f"Brand {j+1} - Project {proj_id}",
                    project_id=proj_id,
                    category="Consumer",
                    created_at=datetime.utcnow()
                )
                brands.append(brand)
        return brands
    
    async def _load_metrics_data(self, brand_ids: List[str], 
                               date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Load metrics data for brands within date range"""
        # Simulated metrics data
        metrics_data = {}
        
        for brand_id in brand_ids:
            # Generate sample time series data
            dates = pd.date_range(date_range[0], date_range[1], freq='D')
            
            metrics_data[brand_id] = {
                'brand_awareness': np.random.uniform(0.3, 0.9, len(dates)).tolist(),
                'brand_consideration': np.random.uniform(0.2, 0.7, len(dates)).tolist(),
                'brand_preference': np.random.uniform(0.1, 0.6, len(dates)).tolist(),
                'purchase_intent': np.random.uniform(0.05, 0.4, len(dates)).tolist(),
                'customer_satisfaction': np.random.uniform(0.6, 0.95, len(dates)).tolist(),
                'net_promoter_score': np.random.uniform(-20, 80, len(dates)).tolist(),
                'market_share': np.random.uniform(0.05, 0.25, len(dates)).tolist(),
                'revenue': np.random.uniform(100000, 1000000, len(dates)).tolist(),
                'roi': np.random.uniform(1.2, 4.5, len(dates)).tolist(),
                'engagement_rate': np.random.uniform(0.02, 0.15, len(dates)).tolist(),
                'dates': [d.isoformat() for d in dates]
            }
        
        return metrics_data
    
    async def _load_analytics_data(self, organization_ids: List[str],
                                 project_ids: List[str],
                                 brand_ids: List[str],
                                 date_range: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
        """Load analytics results from database"""
        # Simulated analytics data
        analytics_data = []
        
        for brand_id in brand_ids:
            analytics_data.append({
                'id': f"analytics_{brand_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                'brand_id': brand_id,
                'analysis_type': 'performance_analysis',
                'results': {
                    'performance_score': np.random.uniform(0.6, 0.95),
                    'trend_direction': np.random.choice(['up', 'down', 'stable']),
                    'volatility': np.random.uniform(0.1, 0.4),
                    'correlation_strength': np.random.uniform(0.3, 0.9)
                },
                'confidence_score': np.random.uniform(0.7, 0.95),
                'created_at': datetime.utcnow().isoformat()
            })
        
        return analytics_data
    
    # Serialization methods
    
    def _serialize_organization(self, org: Organization) -> Dict[str, Any]:
        """Serialize organization object"""
        return {
            'id': org.id,
            'name': org.name,
            'industry': getattr(org, 'industry', 'Unknown'),
            'created_at': org.created_at.isoformat() if org.created_at else None
        }
    
    def _serialize_project(self, project: Project) -> Dict[str, Any]:
        """Serialize project object"""
        return {
            'id': project.id,
            'name': project.name,
            'organization_id': project.organization_id,
            'status': getattr(project, 'status', 'active'),
            'created_at': project.created_at.isoformat() if project.created_at else None
        }
    
    def _serialize_brand(self, brand: Brand) -> Dict[str, Any]:
        """Serialize brand object"""
        return {
            'id': brand.id,
            'name': brand.name,
            'project_id': brand.project_id,
            'category': getattr(brand, 'category', 'Unknown'),
            'created_at': brand.created_at.isoformat() if brand.created_at else None
        }
    
    # Visualization methods
    
    def _get_default_visualizations(self) -> List[str]:
        """Get default visualizations for this report type"""
        return ['performance_trends', 'brand_comparison', 'project_overview']
    
    async def _create_performance_trends_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance trends chart"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Brand Awareness', 'Revenue', 'ROI', 'Market Share'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = self.config['color_palette']
        
        for i, (brand_id, metrics) in enumerate(data['metrics_data'].items()):
            brand_name = data['brands'][brand_id]['name']
            color = colors[i % len(colors)]
            
            # Brand Awareness
            fig.add_trace(
                go.Scatter(
                    x=metrics['dates'],
                    y=metrics['brand_awareness'],
                    name=f"{brand_name} - Awareness",
                    line=dict(color=color),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Revenue
            fig.add_trace(
                go.Scatter(
                    x=metrics['dates'],
                    y=metrics['revenue'],
                    name=f"{brand_name} - Revenue",
                    line=dict(color=color),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # ROI
            fig.add_trace(
                go.Scatter(
                    x=metrics['dates'],
                    y=metrics['roi'],
                    name=f"{brand_name} - ROI",
                    line=dict(color=color),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Market Share
            fig.add_trace(
                go.Scatter(
                    x=metrics['dates'],
                    y=metrics['market_share'],
                    name=f"{brand_name} - Market Share",
                    line=dict(color=color),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Performance Trends Across Brands",
            height=self.config['chart_height'] * 2,
            font=dict(family=self.config['font_family']),
            template=self.config['chart_theme']
        )
        
        return {
            'chart_data': fig.to_dict(),
            'chart_html': fig.to_html(),
            'chart_image': self._fig_to_base64(fig)
        }
    
    async def _create_brand_comparison_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create brand comparison chart"""
        
        brands = []
        awareness = []
        revenue = []
        roi = []
        
        for brand_id, metrics in data['metrics_data'].items():
            brands.append(data['brands'][brand_id]['name'])
            awareness.append(np.mean(metrics['brand_awareness']))
            revenue.append(np.mean(metrics['revenue']))
            roi.append(np.mean(metrics['roi']))
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Brand Awareness', 'Average Revenue', 'Average ROI'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = self.config['color_palette']
        
        # Brand Awareness
        fig.add_trace(
            go.Bar(
                x=brands,
                y=awareness,
                name="Brand Awareness",
                marker_color=colors[0]
            ),
            row=1, col=1
        )
        
        # Revenue
        fig.add_trace(
            go.Bar(
                x=brands,
                y=revenue,
                name="Revenue",
                marker_color=colors[1]
            ),
            row=1, col=2
        )
        
        # ROI
        fig.add_trace(
            go.Bar(
                x=brands,
                y=roi,
                name="ROI",
                marker_color=colors[2]
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            title="Brand Performance Comparison",
            height=self.config['chart_height'],
            font=dict(family=self.config['font_family']),
            template=self.config['chart_theme'],
            showlegend=False
        )
        
        return {
            'chart_data': fig.to_dict(),
            'chart_html': fig.to_html(),
            'chart_image': self._fig_to_base64(fig)
        }
    
    def _fig_to_base64(self, fig) -> str:
        """Convert plotly figure to base64 image"""
        img_bytes = fig.to_image(format="png")
        img_base64 = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{img_base64}"
    
    # Analysis methods (placeholder implementations)
    
    async def _analyze_performance_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze performance patterns in the data"""
        insights = []
        
        # Example insight
        insights.append({
            'type': 'performance_pattern',
            'title': 'Strong Brand Performance Correlation',
            'description': 'Brand awareness and revenue show strong positive correlation across all brands',
            'importance_score': 0.85,
            'confidence': 0.92,
            'impact': 'high',
            'category': 'performance'
        })
        
        return insights
    
    async def _analyze_trends(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze trends in the data"""
        insights = []
        
        # Example trend insight
        insights.append({
            'type': 'trend_analysis',
            'title': 'Upward Revenue Trend',
            'description': 'Revenue shows consistent upward trend across 80% of brands',
            'importance_score': 0.78,
            'confidence': 0.88,
            'impact': 'medium',
            'category': 'trend'
        })
        
        return insights
    
    # Export methods (placeholder implementations)
    
    async def _export_json(self, report_data: Dict[str, Any], output_path: str = None) -> str:
        """Export report as JSON"""
        if not output_path:
            output_path = f"/tmp/report_{self.report_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return output_path
    
    async def _export_pdf(self, report_data: Dict[str, Any], output_path: str = None) -> str:
        """Export report as PDF"""
        # Placeholder - implement PDF generation
        if not output_path:
            output_path = f"/tmp/report_{self.report_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create markdown content and convert to PDF
        markdown_content = self._generate_markdown_report(report_data)
        markdown_path = output_path.replace('.pdf', '.md')
        
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)
        
        # Convert to PDF using manus utility
        import subprocess
        subprocess.run(['manus-md-to-pdf', markdown_path, output_path])
        
        return output_path
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate markdown content for the report"""
        content = f"""# {self.report_type.value.replace('_', ' ').title()} Report

## Executive Summary
{report_data.get('executive_summary', 'Report generated successfully')}

## Data Overview
- Organizations: {report_data.get('data_summary', {}).get('organizations_count', 0)}
- Projects: {report_data.get('data_summary', {}).get('projects_count', 0)}
- Brands: {report_data.get('data_summary', {}).get('brands_count', 0)}

## Key Insights
"""
        
        for insight in report_data.get('insights', [])[:5]:
            content += f"### {insight.get('title', 'Insight')}\n"
            content += f"{insight.get('description', 'No description available')}\n\n"
        
        content += "## Recommendations\n"
        for rec in report_data.get('recommendations', [])[:5]:
            content += f"- {rec.get('title', 'Recommendation')}: {rec.get('description', 'No description')}\n"
        
        return content

# Factory function for creating report generators
def create_report_generator(report_type: ReportType, config: Dict[str, Any] = None) -> BaseReportGenerator:
    """Create appropriate report generator based on type"""
    
    # Import specific generators
    from src.reports.generators.portfolio_performance_generator import PortfolioPerformanceGenerator
    from src.reports.generators.brand_equity_generator import BrandEquityGenerator
    # ... other generators
    
    if report_type == ReportType.PORTFOLIO_PERFORMANCE:
        return PortfolioPerformanceGenerator(config)
    elif report_type == ReportType.BRAND_EQUITY_ANALYSIS:
        return BrandEquityGenerator(config)
    # ... other report types
    else:
        # Return base generator for unsupported types
        class GenericReportGenerator(BaseReportGenerator):
            async def generate_report(self, organization_ids, project_ids=None, brand_ids=None, date_range=None, parameters=None):
                data = await self.prepare_data(organization_ids, project_ids, brand_ids, date_range)
                visualizations = await self.generate_visualizations(data)
                insights = await self.generate_insights(data)
                recommendations = await self.generate_recommendations(data, insights)
                
                return {
                    'report_type': self.report_type.value,
                    'metadata': self.report_metadata,
                    'data_summary': data['data_summary'],
                    'insights': insights,
                    'recommendations': recommendations,
                    'visualizations': visualizations,
                    'generated_at': datetime.utcnow().isoformat()
                }
        
        return GenericReportGenerator(report_type, config)

