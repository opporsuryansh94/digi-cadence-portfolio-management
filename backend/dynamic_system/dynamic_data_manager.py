"""
Dynamic Digi-Cadence Data Manager
Adaptive data processing based on actual project characteristics and patterns
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime, timedelta
import json
import logging

warnings.filterwarnings('ignore')

class DynamicDigiCadenceDataManager:
    """
    Dynamic data manager that adapts to actual Digi-Cadence project and brand characteristics
    """
    
    def __init__(self, project_ids: List[int], brand_ids: Optional[List[str]] = None, token: str = None):
        """
        Initialize with dynamic project and brand selection
        
        Args:
            project_ids: List of project IDs to analyze
            brand_ids: Optional list of specific brand IDs to focus on
            token: Authentication token for Digi-Cadence APIs
        """
        self.project_ids = project_ids if isinstance(project_ids, list) else [project_ids]
        self.brand_ids = brand_ids or []
        self.token = token
        self.data_characteristics = None
        self.raw_data = {}
        self.processed_data = {}
        
        # API endpoints (actual Digi-Cadence structure)
        self.api_endpoints = {
            'frontend': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:7000",
            'normalized': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8001",
            'process_metric': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8027",
            'analytics_metric': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8025",
            'weight_sum': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8002",
            'insights': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8004",
            'brand_data': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8018"
        }
        
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def analyze_and_adapt_to_data(self) -> Dict[str, Any]:
        """
        Fetch actual data from Digi-Cadence APIs and analyze characteristics dynamically
        
        Returns:
            Dict containing processed data adapted to actual characteristics
        """
        try:
            # Fetch actual data from multiple sources
            self.logger.info(f"Fetching data for projects: {self.project_ids}")
            self.raw_data = self._fetch_actual_data()
            
            # Analyze data characteristics dynamically
            self.data_characteristics = self._analyze_data_characteristics(self.raw_data)
            
            # Adapt processing approach based on characteristics
            self.processed_data = self._adapt_processing_approach(self.raw_data, self.data_characteristics)
            
            self.logger.info("Data analysis and adaptation completed successfully")
            return self.processed_data
            
        except Exception as e:
            self.logger.error(f"Error in data analysis and adaptation: {str(e)}")
            raise
    
    def _fetch_actual_data(self) -> Dict[str, Any]:
        """
        Fetch actual data from Digi-Cadence APIs dynamically
        
        Returns:
            Dict containing raw data from various sources
        """
        raw_data = {}
        
        for project_id in self.project_ids:
            project_data = {}
            
            try:
                # Fetch normalized data (core DC scores and sectional scores)
                normalized_data = self._fetch_normalized_data(project_id)
                if normalized_data is not None:
                    project_data['normalized'] = normalized_data
                
                # Fetch project metadata
                project_metadata = self._fetch_project_metadata(project_id)
                if project_metadata is not None:
                    project_data['metadata'] = project_metadata
                
                # Fetch brand-specific data if brand_ids specified
                if self.brand_ids:
                    brand_data = self._fetch_brand_data(project_id, self.brand_ids)
                    if brand_data is not None:
                        project_data['brands'] = brand_data
                
                # Fetch additional analytics data
                analytics_data = self._fetch_analytics_data(project_id)
                if analytics_data is not None:
                    project_data['analytics'] = analytics_data
                
                raw_data[project_id] = project_data
                
            except Exception as e:
                self.logger.warning(f"Error fetching data for project {project_id}: {str(e)}")
                continue
        
        return raw_data
    
    def _fetch_normalized_data(self, project_id: int) -> Optional[pd.DataFrame]:
        """
        Fetch normalized data (DC scores and sectional scores) for a project
        
        Args:
            project_id: Project ID to fetch data for
            
        Returns:
            DataFrame with normalized scores or None if failed
        """
        try:
            url = f"{self.api_endpoints['normalized']}/normalized/normalized_value"
            params = {'project_ids': project_id}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    
                    # Create pivot table for easier analysis (Section/Platform/Metric structure)
                    if not df.empty and all(col in df.columns for col in ['metricname', 'sectionName', 'platformname', 'brandName', 'normalized']):
                        pivot_df = df.pivot_table(
                            index=['metricname', 'sectionName', 'platformname'],
                            columns='brandName',
                            values='normalized',
                            aggfunc='first'
                        ).reset_index()
                        
                        pivot_df.columns.name = None
                        pivot_df = pivot_df.rename(columns={
                            'metricname': 'Metric',
                            'sectionName': 'section_name',
                            'platformname': 'platform_name'
                        })
                        
                        return pivot_df
                    else:
                        return df
                        
            else:
                self.logger.warning(f"Failed to fetch normalized data for project {project_id}: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching normalized data for project {project_id}: {str(e)}")
            return None
    
    def _fetch_project_metadata(self, project_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch project metadata including metrics structure and weights
        
        Args:
            project_id: Project ID to fetch metadata for
            
        Returns:
            Dict with project metadata or None if failed
        """
        try:
            url = f"{self.api_endpoints['frontend']}/api/v1/project/get-project/"
            params = {'project_id': project_id}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'project' in data:
                    return data['project']
            else:
                self.logger.warning(f"Failed to fetch project metadata for project {project_id}: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error fetching project metadata for project {project_id}: {str(e)}")
            
        return None
    
    def _fetch_brand_data(self, project_id: int, brand_ids: List[str]) -> Optional[Dict[str, Any]]:
        """
        Fetch brand-specific data for selected brands
        
        Args:
            project_id: Project ID
            brand_ids: List of brand IDs to fetch data for
            
        Returns:
            Dict with brand data or None if failed
        """
        brand_data = {}
        
        for brand_id in brand_ids:
            try:
                url = f"{self.api_endpoints['brand_data']}/brands/{brand_id}/project_id/{project_id}"
                
                response = requests.get(url, headers=self.headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    brand_data[brand_id] = data
                else:
                    self.logger.warning(f"Failed to fetch data for brand {brand_id} in project {project_id}: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching data for brand {brand_id} in project {project_id}: {str(e)}")
                continue
        
        return brand_data if brand_data else None
    
    def _fetch_analytics_data(self, project_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch additional analytics data for comprehensive analysis
        
        Args:
            project_id: Project ID to fetch analytics for
            
        Returns:
            Dict with analytics data or None if failed
        """
        analytics_data = {}
        
        try:
            # Try to fetch insights data
            insights_url = f"{self.api_endpoints['insights']}/get_multi_data"
            insights_response = requests.get(insights_url, headers=self.headers, timeout=30)
            
            if insights_response.status_code == 200:
                analytics_data['insights'] = insights_response.json()
            
            # Try to fetch weight sum data
            weight_url = f"{self.api_endpoints['weight_sum']}/weight_sum"
            weight_response = requests.get(weight_url, headers=self.headers, timeout=30)
            
            if weight_response.status_code == 200:
                analytics_data['weights'] = weight_response.json()
                
        except Exception as e:
            self.logger.error(f"Error fetching analytics data for project {project_id}: {str(e)}")
        
        return analytics_data if analytics_data else None
    
    def _analyze_data_characteristics(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze actual data characteristics to determine processing approach
        
        Args:
            raw_data: Raw data fetched from APIs
            
        Returns:
            Dict containing data characteristics analysis
        """
        characteristics = {
            'projects_count': len(raw_data),
            'brands_available': set(),
            'sections_available': set(),
            'platforms_available': set(),
            'metrics_available': set(),
            'score_ranges': {},
            'performance_patterns': {},
            'correlation_strength': {},
            'business_outcome_availability': {},
            'competitive_data_presence': False,
            'data_completeness': {},
            'temporal_data_available': False
        }
        
        for project_id, project_data in raw_data.items():
            if 'normalized' in project_data and project_data['normalized'] is not None:
                df = project_data['normalized']
                
                # Extract available brands, sections, platforms, metrics
                if not df.empty:
                    if 'section_name' in df.columns:
                        characteristics['sections_available'].update(df['section_name'].unique())
                    if 'platform_name' in df.columns:
                        characteristics['platforms_available'].update(df['platform_name'].unique())
                    if 'Metric' in df.columns:
                        characteristics['metrics_available'].update(df['Metric'].unique())
                    
                    # Get brand columns (excluding metadata columns)
                    brand_columns = [col for col in df.columns if col not in ['Metric', 'section_name', 'platform_name']]
                    characteristics['brands_available'].update(brand_columns)
                    
                    # Analyze score ranges for each brand
                    for brand in brand_columns:
                        if brand not in characteristics['score_ranges']:
                            characteristics['score_ranges'][brand] = {}
                        
                        brand_scores = df[brand].dropna()
                        if not brand_scores.empty:
                            # Convert to numeric, handling any string values
                            numeric_scores = pd.to_numeric(brand_scores, errors='coerce').dropna()
                            if not numeric_scores.empty:
                                characteristics['score_ranges'][brand] = {
                                    'min': float(numeric_scores.min()),
                                    'max': float(numeric_scores.max()),
                                    'mean': float(numeric_scores.mean()),
                                    'std': float(numeric_scores.std()),
                                    'count': len(numeric_scores)
                                }
                
                # Analyze performance patterns
                characteristics['performance_patterns'][project_id] = self._identify_performance_patterns(df)
                
                # Check data completeness
                characteristics['data_completeness'][project_id] = self._assess_data_completeness(df)
        
        # Convert sets to lists for JSON serialization
        characteristics['brands_available'] = list(characteristics['brands_available'])
        characteristics['sections_available'] = list(characteristics['sections_available'])
        characteristics['platforms_available'] = list(characteristics['platforms_available'])
        characteristics['metrics_available'] = list(characteristics['metrics_available'])
        
        # Assess overall correlation potential
        characteristics['correlation_strength'] = self._assess_correlation_potential(raw_data)
        
        # Check for competitive data
        characteristics['competitive_data_presence'] = len(characteristics['brands_available']) > 1
        
        return characteristics
    
    def _identify_performance_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify performance patterns in the data
        
        Args:
            df: DataFrame with normalized scores
            
        Returns:
            Dict with identified patterns
        """
        patterns = {
            'high_performers': [],
            'low_performers': [],
            'consistent_performers': [],
            'volatile_performers': [],
            'section_strengths': {},
            'improvement_opportunities': []
        }
        
        if df.empty:
            return patterns
        
        try:
            # Get brand columns
            brand_columns = [col for col in df.columns if col not in ['Metric', 'section_name', 'platform_name']]
            
            for brand in brand_columns:
                brand_scores = pd.to_numeric(df[brand], errors='coerce').dropna()
                
                if not brand_scores.empty:
                    mean_score = brand_scores.mean()
                    std_score = brand_scores.std()
                    
                    # Classify performance level
                    if mean_score >= 80:
                        patterns['high_performers'].append(brand)
                    elif mean_score <= 60:
                        patterns['low_performers'].append(brand)
                    
                    # Classify consistency
                    if std_score <= 10:
                        patterns['consistent_performers'].append(brand)
                    elif std_score >= 20:
                        patterns['volatile_performers'].append(brand)
            
            # Analyze section-wise performance
            if 'section_name' in df.columns:
                for section in df['section_name'].unique():
                    section_data = df[df['section_name'] == section]
                    section_scores = {}
                    
                    for brand in brand_columns:
                        brand_section_scores = pd.to_numeric(section_data[brand], errors='coerce').dropna()
                        if not brand_section_scores.empty:
                            section_scores[brand] = brand_section_scores.mean()
                    
                    if section_scores:
                        patterns['section_strengths'][section] = section_scores
        
        except Exception as e:
            self.logger.error(f"Error identifying performance patterns: {str(e)}")
        
        return patterns
    
    def _assess_data_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Assess data completeness for quality evaluation
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Dict with completeness metrics
        """
        if df.empty:
            return {'overall_completeness': 0.0}
        
        try:
            total_cells = df.size
            non_null_cells = df.count().sum()
            
            completeness = {
                'overall_completeness': (non_null_cells / total_cells) * 100 if total_cells > 0 else 0.0,
                'column_completeness': {}
            }
            
            for col in df.columns:
                col_completeness = (df[col].count() / len(df)) * 100
                completeness['column_completeness'][col] = col_completeness
            
            return completeness
            
        except Exception as e:
            self.logger.error(f"Error assessing data completeness: {str(e)}")
            return {'overall_completeness': 0.0}
    
    def _assess_correlation_potential(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess potential for correlation analysis
        
        Args:
            raw_data: Raw data from APIs
            
        Returns:
            Dict with correlation potential metrics
        """
        correlation_potential = {
            'cross_brand_correlation': 0.0,
            'cross_project_correlation': 0.0,
            'business_correlation_potential': 0.0
        }
        
        try:
            # Assess cross-brand correlation potential
            total_brands = len(self.data_characteristics.get('brands_available', []))
            if total_brands > 1:
                correlation_potential['cross_brand_correlation'] = min(1.0, total_brands / 5.0)
            
            # Assess cross-project correlation potential
            total_projects = len(raw_data)
            if total_projects > 1:
                correlation_potential['cross_project_correlation'] = min(1.0, total_projects / 3.0)
            
            # Assess business correlation potential (based on data availability)
            business_data_indicators = 0
            for project_data in raw_data.values():
                if 'analytics' in project_data:
                    business_data_indicators += 1
                if 'brands' in project_data:
                    business_data_indicators += 1
            
            if business_data_indicators > 0:
                correlation_potential['business_correlation_potential'] = min(1.0, business_data_indicators / len(raw_data))
        
        except Exception as e:
            self.logger.error(f"Error assessing correlation potential: {str(e)}")
        
        return correlation_potential
    
    def _adapt_processing_approach(self, raw_data: Dict[str, Any], characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt processing approach based on data characteristics
        
        Args:
            raw_data: Raw data from APIs
            characteristics: Analyzed data characteristics
            
        Returns:
            Dict with processed data adapted to characteristics
        """
        processed_data = {
            'metrics_data': {},
            'weights_data': {},
            'structure_data': {},
            'business_data': {},
            'competitive_data': {},
            'processing_metadata': {
                'approach': 'adaptive',
                'characteristics': characteristics,
                'processing_timestamp': datetime.now().isoformat()
            }
        }
        
        try:
            for project_id, project_data in raw_data.items():
                project_processed = {}
                
                # Process normalized data based on characteristics
                if 'normalized' in project_data and project_data['normalized'] is not None:
                    df = project_data['normalized']
                    
                    # Adapt based on data completeness
                    completeness = characteristics['data_completeness'].get(project_id, {})
                    if completeness.get('overall_completeness', 0) > 70:
                        # High completeness - use full processing
                        project_processed['metrics_data'] = self._process_high_quality_data(df)
                    else:
                        # Lower completeness - use robust processing
                        project_processed['metrics_data'] = self._process_robust_data(df)
                
                # Process metadata for weights and structure
                if 'metadata' in project_data and project_data['metadata'] is not None:
                    metadata = project_data['metadata']
                    project_processed['weights_data'] = self._extract_weights_data(metadata)
                    project_processed['structure_data'] = self._extract_structure_data(metadata)
                
                # Process business data if available
                if 'analytics' in project_data and project_data['analytics'] is not None:
                    project_processed['business_data'] = self._process_business_data(project_data['analytics'])
                
                # Process brand data for competitive analysis
                if 'brands' in project_data and project_data['brands'] is not None:
                    project_processed['competitive_data'] = self._process_competitive_data(project_data['brands'])
                
                processed_data['metrics_data'][project_id] = project_processed.get('metrics_data')
                processed_data['weights_data'][project_id] = project_processed.get('weights_data')
                processed_data['structure_data'][project_id] = project_processed.get('structure_data')
                processed_data['business_data'][project_id] = project_processed.get('business_data')
                processed_data['competitive_data'][project_id] = project_processed.get('competitive_data')
        
        except Exception as e:
            self.logger.error(f"Error in adaptive processing: {str(e)}")
        
        return processed_data
    
    def _process_high_quality_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process high-quality data with full feature set"""
        try:
            # Clean and validate data
            processed_df = df.copy()
            
            # Handle any remaining data quality issues
            brand_columns = [col for col in processed_df.columns if col not in ['Metric', 'section_name', 'platform_name']]
            
            for brand in brand_columns:
                # Convert to numeric and handle non-numeric values
                processed_df[brand] = pd.to_numeric(processed_df[brand], errors='coerce')
                
                # Fill missing values with section/platform median if available
                if processed_df[brand].isna().any():
                    processed_df[brand] = processed_df[brand].fillna(processed_df[brand].median())
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing high quality data: {str(e)}")
            return df
    
    def _process_robust_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process lower quality data with robust methods"""
        try:
            # More conservative processing for incomplete data
            processed_df = df.copy()
            
            brand_columns = [col for col in processed_df.columns if col not in ['Metric', 'section_name', 'platform_name']]
            
            for brand in brand_columns:
                # Convert to numeric
                processed_df[brand] = pd.to_numeric(processed_df[brand], errors='coerce')
                
                # Only fill missing values if we have enough data points
                if processed_df[brand].count() > len(processed_df) * 0.5:
                    processed_df[brand] = processed_df[brand].fillna(processed_df[brand].median())
                else:
                    # Mark columns with too much missing data
                    processed_df[f'{brand}_quality'] = processed_df[brand].notna()
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing robust data: {str(e)}")
            return df
    
    def _extract_weights_data(self, metadata: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Extract weights data from project metadata"""
        try:
            if 'metrics' in metadata:
                weights_data = []
                project_name = metadata.get('project_name', 'Unknown')
                
                for metric in metadata['metrics']:
                    weight_row = {
                        'section_name': metric.get('section', {}).get('name', ''),
                        'platform_name': metric.get('platform', {}).get('name', ''),
                        'metrics_new_name': metric.get('metric_new_name', ''),
                        project_name: metric.get('weights', 0)
                    }
                    weights_data.append(weight_row)
                
                return pd.DataFrame(weights_data)
        
        except Exception as e:
            self.logger.error(f"Error extracting weights data: {str(e)}")
        
        return None
    
    def _extract_structure_data(self, metadata: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Extract structure data from project metadata"""
        try:
            if 'metrics' in metadata:
                structure_data = []
                
                for metric in metadata['metrics']:
                    structure_row = {
                        'section_name': metric.get('section', {}).get('name', ''),
                        'platform_name': metric.get('platform', {}).get('name', ''),
                        'metrics_new_name': metric.get('metric_new_name', '')
                    }
                    structure_data.append(structure_row)
                
                return pd.DataFrame(structure_data)
        
        except Exception as e:
            self.logger.error(f"Error extracting structure data: {str(e)}")
        
        return None
    
    def _process_business_data(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process business/analytics data for correlation analysis"""
        try:
            business_data = {}
            
            if 'insights' in analytics_data:
                business_data['insights'] = analytics_data['insights']
            
            if 'weights' in analytics_data:
                business_data['weights'] = analytics_data['weights']
            
            return business_data
            
        except Exception as e:
            self.logger.error(f"Error processing business data: {str(e)}")
            return {}
    
    def _process_competitive_data(self, brands_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process brand data for competitive analysis"""
        try:
            competitive_data = {}
            
            for brand_id, brand_info in brands_data.items():
                competitive_data[brand_id] = brand_info
            
            return competitive_data
            
        except Exception as e:
            self.logger.error(f"Error processing competitive data: {str(e)}")
            return {}
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of processed data for reporting
        
        Returns:
            Dict with data summary
        """
        if not self.data_characteristics:
            return {"error": "No data processed yet. Call analyze_and_adapt_to_data() first."}
        
        summary = {
            'projects_analyzed': self.data_characteristics['projects_count'],
            'brands_available': len(self.data_characteristics['brands_available']),
            'sections_available': len(self.data_characteristics['sections_available']),
            'platforms_available': len(self.data_characteristics['platforms_available']),
            'metrics_available': len(self.data_characteristics['metrics_available']),
            'competitive_analysis_possible': self.data_characteristics['competitive_data_presence'],
            'correlation_potential': self.data_characteristics['correlation_strength'],
            'data_quality': 'high' if any(
                comp.get('overall_completeness', 0) > 80 
                for comp in self.data_characteristics.get('data_completeness', {}).values()
            ) else 'medium'
        }
        
        return summary
    
    def get_available_brands(self) -> List[str]:
        """Get list of available brands for selection"""
        if self.data_characteristics:
            return self.data_characteristics.get('brands_available', [])
        return []
    
    def get_available_sections(self) -> List[str]:
        """Get list of available sections"""
        if self.data_characteristics:
            return self.data_characteristics.get('sections_available', [])
        return []
    
    def get_recommended_analysis_approach(self) -> Dict[str, Any]:
        """
        Get recommended analysis approach based on data characteristics
        
        Returns:
            Dict with recommended approach and reasoning
        """
        if not self.data_characteristics:
            return {"error": "No data analyzed yet"}
        
        recommendations = {
            'recommended_reports': [],
            'analysis_approach': 'standard',
            'reasoning': [],
            'confidence_level': 'medium'
        }
        
        # Determine recommended reports based on data availability
        if self.data_characteristics['competitive_data_presence']:
            recommendations['recommended_reports'].extend([
                'competitive_positioning_analysis',
                'competitive_advantage_analysis'
            ])
            recommendations['reasoning'].append("Multiple brands available for competitive analysis")
        
        if self.data_characteristics['correlation_strength'].get('business_correlation_potential', 0) > 0.5:
            recommendations['recommended_reports'].extend([
                'dc_score_performance_analysis',
                'business_outcome_correlation'
            ])
            recommendations['reasoning'].append("Strong business correlation potential detected")
        
        if len(self.data_characteristics['brands_available']) >= 3:
            recommendations['analysis_approach'] = 'comprehensive'
            recommendations['confidence_level'] = 'high'
            recommendations['reasoning'].append("Sufficient brands for comprehensive analysis")
        
        return recommendations

