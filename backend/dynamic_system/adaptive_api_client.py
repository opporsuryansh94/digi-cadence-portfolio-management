"""
Adaptive Digi-Cadence API Client
Dynamic API integration based on available endpoints and data requirements
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import logging
import time
from datetime import datetime
import concurrent.futures
from urllib.parse import urljoin
import warnings

warnings.filterwarnings('ignore')

class AdaptiveDigiCadenceAPIClient:
    """
    Adaptive API client that discovers available endpoints and fetches data dynamically
    based on project and brand requirements
    """
    
    def __init__(self, project_ids: Union[int, List[int]], brand_ids: Optional[List[str]] = None, token: str = None):
        """
        Initialize adaptive API client
        
        Args:
            project_ids: Project ID(s) to work with
            brand_ids: Optional list of brand IDs to focus on
            token: Authentication token
        """
        self.project_ids = project_ids if isinstance(project_ids, list) else [project_ids]
        self.brand_ids = brand_ids or []
        self.token = token
        
        # Base API endpoints (actual Digi-Cadence infrastructure)
        self.base_endpoints = {
            'frontend': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:7000",
            'normalized': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8001",
            'process_metric': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8027",
            'analytics_metric': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8025",
            'weight_sum': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8002",
            'insights': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8004",
            'brand_data': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8018",
            'competitive': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8019",
            'forecasting': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8020",
            'optimization': "http://ec2-13-127-182-134.ap-south-1.compute.amazonaws.com:8021"
        }
        
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.available_endpoints = {}
        self.endpoint_capabilities = {}
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Discover available endpoints
        self._discover_available_endpoints()
    
    def _discover_available_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover which endpoints are available and their capabilities
        
        Returns:
            Dict mapping endpoint names to their capabilities
        """
        self.logger.info("Discovering available API endpoints...")
        
        for endpoint_name, base_url in self.base_endpoints.items():
            try:
                # Test endpoint availability with a simple health check
                health_response = self._test_endpoint_health(base_url)
                
                if health_response['available']:
                    self.available_endpoints[endpoint_name] = {
                        'base_url': base_url,
                        'status': 'available',
                        'response_time': health_response['response_time'],
                        'capabilities': self._discover_endpoint_capabilities(endpoint_name, base_url)
                    }
                    self.logger.info(f"✓ {endpoint_name} endpoint available at {base_url}")
                else:
                    self.logger.warning(f"✗ {endpoint_name} endpoint unavailable at {base_url}")
                    
            except Exception as e:
                self.logger.warning(f"Error testing {endpoint_name} endpoint: {str(e)}")
        
        self.logger.info(f"Discovered {len(self.available_endpoints)} available endpoints")
        return self.available_endpoints
    
    def _test_endpoint_health(self, base_url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Test if an endpoint is available and responsive
        
        Args:
            base_url: Base URL to test
            timeout: Request timeout in seconds
            
        Returns:
            Dict with availability status and response time
        """
        try:
            start_time = time.time()
            
            # Try common health check endpoints
            health_endpoints = ['/', '/health', '/status', '/api/health']
            
            for health_path in health_endpoints:
                try:
                    url = urljoin(base_url, health_path)
                    response = requests.get(url, headers=self.headers, timeout=timeout)
                    
                    if response.status_code in [200, 404]:  # 404 is OK, means server is responding
                        response_time = time.time() - start_time
                        return {
                            'available': True,
                            'response_time': response_time,
                            'status_code': response.status_code
                        }
                        
                except requests.exceptions.RequestException:
                    continue
            
            return {'available': False, 'response_time': None}
            
        except Exception as e:
            self.logger.debug(f"Health check failed for {base_url}: {str(e)}")
            return {'available': False, 'response_time': None}
    
    def _discover_endpoint_capabilities(self, endpoint_name: str, base_url: str) -> Dict[str, Any]:
        """
        Discover what capabilities each endpoint provides
        
        Args:
            endpoint_name: Name of the endpoint
            base_url: Base URL of the endpoint
            
        Returns:
            Dict describing endpoint capabilities
        """
        capabilities = {
            'data_types': [],
            'supported_operations': [],
            'requires_project_id': False,
            'supports_brand_filtering': False,
            'supports_batch_requests': False
        }
        
        # Define known capabilities based on endpoint patterns
        endpoint_capabilities_map = {
            'frontend': {
                'data_types': ['project_metadata', 'user_data', 'configuration'],
                'supported_operations': ['get_project', 'list_projects', 'get_user_info'],
                'requires_project_id': True,
                'supports_brand_filtering': False,
                'supports_batch_requests': False
            },
            'normalized': {
                'data_types': ['normalized_scores', 'dc_scores', 'sectional_scores'],
                'supported_operations': ['get_normalized_data', 'get_scores_by_project'],
                'requires_project_id': True,
                'supports_brand_filtering': True,
                'supports_batch_requests': True
            },
            'analytics_metric': {
                'data_types': ['analytics_data', 'performance_metrics', 'insights'],
                'supported_operations': ['get_analytics', 'calculate_metrics'],
                'requires_project_id': True,
                'supports_brand_filtering': True,
                'supports_batch_requests': False
            },
            'weight_sum': {
                'data_types': ['weights_data', 'aggregated_scores'],
                'supported_operations': ['get_weights', 'calculate_weighted_scores'],
                'requires_project_id': True,
                'supports_brand_filtering': False,
                'supports_batch_requests': False
            },
            'insights': {
                'data_types': ['business_insights', 'recommendations', 'trends'],
                'supported_operations': ['get_insights', 'generate_recommendations'],
                'requires_project_id': True,
                'supports_brand_filtering': True,
                'supports_batch_requests': False
            },
            'brand_data': {
                'data_types': ['brand_performance', 'brand_metrics', 'competitive_data'],
                'supported_operations': ['get_brand_data', 'compare_brands'],
                'requires_project_id': True,
                'supports_brand_filtering': True,
                'supports_batch_requests': True
            }
        }
        
        if endpoint_name in endpoint_capabilities_map:
            capabilities.update(endpoint_capabilities_map[endpoint_name])
        
        return capabilities
    
    def fetch_adaptive_data(self) -> Dict[str, Any]:
        """
        Fetch data adaptively based on available endpoints and requirements
        
        Returns:
            Dict containing all successfully fetched data
        """
        self.logger.info("Starting adaptive data fetch...")
        
        # Analyze data requirements based on project and brand selection
        data_requirements = self._analyze_data_requirements()
        
        # Fetch data from available endpoints
        fetched_data = {}
        
        for requirement in data_requirements:
            if self._endpoint_available(requirement['endpoint']):
                try:
                    data = self._fetch_from_endpoint(
                        requirement['endpoint'],
                        requirement['params'],
                        requirement['data_type']
                    )
                    
                    if data is not None:
                        fetched_data[requirement['data_type']] = data
                        self.logger.info(f"✓ Successfully fetched {requirement['data_type']}")
                    else:
                        self.logger.warning(f"✗ Failed to fetch {requirement['data_type']}")
                        
                except Exception as e:
                    self.logger.error(f"Error fetching {requirement['data_type']}: {str(e)}")
        
        # Consolidate and validate fetched data
        consolidated_data = self._consolidate_fetched_data(fetched_data)
        
        self.logger.info(f"Adaptive data fetch completed. Retrieved {len(consolidated_data)} data types.")
        return consolidated_data
    
    def _analyze_data_requirements(self) -> List[Dict[str, Any]]:
        """
        Analyze what data is needed based on project and brand selection
        
        Returns:
            List of data requirements with endpoint and parameter information
        """
        requirements = []
        
        # Core requirements for all projects
        for project_id in self.project_ids:
            # Normalized scores (essential for DC score analysis)
            requirements.append({
                'data_type': 'normalized_scores',
                'endpoint': 'normalized',
                'params': {'project_ids': project_id},
                'priority': 'high',
                'required': True
            })
            
            # Project metadata
            requirements.append({
                'data_type': 'project_metadata',
                'endpoint': 'frontend',
                'params': {'project_id': project_id},
                'priority': 'high',
                'required': True
            })
            
            # Analytics data
            requirements.append({
                'data_type': 'analytics_data',
                'endpoint': 'analytics_metric',
                'params': {'project_id': project_id},
                'priority': 'medium',
                'required': False
            })
            
            # Weights data
            requirements.append({
                'data_type': 'weights_data',
                'endpoint': 'weight_sum',
                'params': {'project_id': project_id},
                'priority': 'medium',
                'required': False
            })
            
            # Insights data
            requirements.append({
                'data_type': 'insights_data',
                'endpoint': 'insights',
                'params': {'project_id': project_id},
                'priority': 'low',
                'required': False
            })
        
        # Brand-specific requirements
        if self.brand_ids:
            for project_id in self.project_ids:
                for brand_id in self.brand_ids:
                    requirements.append({
                        'data_type': f'brand_data_{brand_id}',
                        'endpoint': 'brand_data',
                        'params': {'brand_id': brand_id, 'project_id': project_id},
                        'priority': 'medium',
                        'required': False
                    })
        
        # Multi-project requirements
        if len(self.project_ids) > 1:
            requirements.append({
                'data_type': 'cross_project_analytics',
                'endpoint': 'analytics_metric',
                'params': {'project_ids': self.project_ids},
                'priority': 'low',
                'required': False
            })
        
        return requirements
    
    def _endpoint_available(self, endpoint_name: str) -> bool:
        """
        Check if an endpoint is available
        
        Args:
            endpoint_name: Name of the endpoint to check
            
        Returns:
            True if endpoint is available, False otherwise
        """
        return endpoint_name in self.available_endpoints and \
               self.available_endpoints[endpoint_name]['status'] == 'available'
    
    def _fetch_from_endpoint(self, endpoint_name: str, params: Dict[str, Any], data_type: str) -> Optional[Any]:
        """
        Fetch data from a specific endpoint
        
        Args:
            endpoint_name: Name of the endpoint
            params: Parameters for the request
            data_type: Type of data being fetched
            
        Returns:
            Fetched data or None if failed
        """
        if not self._endpoint_available(endpoint_name):
            self.logger.warning(f"Endpoint {endpoint_name} not available")
            return None
        
        base_url = self.available_endpoints[endpoint_name]['base_url']
        
        try:
            # Route to appropriate fetch method based on endpoint
            if endpoint_name == 'normalized':
                return self._fetch_normalized_data(base_url, params)
            elif endpoint_name == 'frontend':
                return self._fetch_project_metadata(base_url, params)
            elif endpoint_name == 'analytics_metric':
                return self._fetch_analytics_data(base_url, params)
            elif endpoint_name == 'weight_sum':
                return self._fetch_weights_data(base_url, params)
            elif endpoint_name == 'insights':
                return self._fetch_insights_data(base_url, params)
            elif endpoint_name == 'brand_data':
                return self._fetch_brand_data(base_url, params)
            else:
                return self._fetch_generic_data(base_url, params, data_type)
                
        except Exception as e:
            self.logger.error(f"Error fetching from {endpoint_name}: {str(e)}")
            return None
    
    def _fetch_normalized_data(self, base_url: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Fetch normalized scores data"""
        try:
            url = f"{base_url}/normalized/normalized_value"
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    
                    # Create structured format for easier analysis
                    if not df.empty and all(col in df.columns for col in ['metricname', 'sectionName', 'platformname', 'brandName', 'normalized']):
                        # Pivot to get brands as columns
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
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching normalized data: {str(e)}")
            return None
    
    def _fetch_project_metadata(self, base_url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch project metadata"""
        try:
            url = f"{base_url}/api/v1/project/get-project/"
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('project') if 'project' in data else data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching project metadata: {str(e)}")
            return None
    
    def _fetch_analytics_data(self, base_url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch analytics data"""
        try:
            # Try multiple possible endpoints
            endpoints_to_try = [
                "/analytics/metrics",
                "/get_analytics",
                "/analytics",
                "/"
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    url = f"{base_url}{endpoint}"
                    response = requests.get(url, headers=self.headers, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        return response.json()
                        
                except requests.exceptions.RequestException:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching analytics data: {str(e)}")
            return None
    
    def _fetch_weights_data(self, base_url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch weights data"""
        try:
            url = f"{base_url}/weight_sum"
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching weights data: {str(e)}")
            return None
    
    def _fetch_insights_data(self, base_url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch insights data"""
        try:
            url = f"{base_url}/get_multi_data"
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching insights data: {str(e)}")
            return None
    
    def _fetch_brand_data(self, base_url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch brand-specific data"""
        try:
            brand_id = params.get('brand_id')
            project_id = params.get('project_id')
            
            if brand_id and project_id:
                url = f"{base_url}/brands/{brand_id}/project_id/{project_id}"
                response = requests.get(url, headers=self.headers, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching brand data: {str(e)}")
            return None
    
    def _fetch_generic_data(self, base_url: str, params: Dict[str, Any], data_type: str) -> Optional[Dict[str, Any]]:
        """Fetch data from generic endpoint"""
        try:
            response = requests.get(base_url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching generic data ({data_type}): {str(e)}")
            return None
    
    def _consolidate_fetched_data(self, fetched_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate and validate fetched data
        
        Args:
            fetched_data: Raw fetched data from various endpoints
            
        Returns:
            Consolidated and validated data structure
        """
        consolidated = {
            'projects': {},
            'brands': {},
            'metadata': {},
            'analytics': {},
            'data_quality': {},
            'fetch_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Organize data by project
            for project_id in self.project_ids:
                project_data = {}
                
                # Core normalized scores
                if 'normalized_scores' in fetched_data:
                    project_data['normalized_scores'] = fetched_data['normalized_scores']
                
                # Project metadata
                if 'project_metadata' in fetched_data:
                    project_data['metadata'] = fetched_data['project_metadata']
                
                # Analytics data
                if 'analytics_data' in fetched_data:
                    project_data['analytics'] = fetched_data['analytics_data']
                
                # Weights data
                if 'weights_data' in fetched_data:
                    project_data['weights'] = fetched_data['weights_data']
                
                consolidated['projects'][project_id] = project_data
            
            # Organize brand-specific data
            for key, value in fetched_data.items():
                if key.startswith('brand_data_'):
                    brand_id = key.replace('brand_data_', '')
                    consolidated['brands'][brand_id] = value
            
            # Add data quality assessment
            consolidated['data_quality'] = self._assess_data_quality(fetched_data)
            
        except Exception as e:
            self.logger.error(f"Error consolidating data: {str(e)}")
        
        return consolidated
    
    def _assess_data_quality(self, fetched_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality and completeness of fetched data
        
        Args:
            fetched_data: Fetched data to assess
            
        Returns:
            Dict with data quality metrics
        """
        quality_assessment = {
            'completeness_score': 0.0,
            'data_types_available': len(fetched_data),
            'missing_critical_data': [],
            'data_freshness': 'unknown',
            'reliability_score': 0.0
        }
        
        try:
            # Critical data types
            critical_data_types = ['normalized_scores', 'project_metadata']
            available_critical = sum(1 for dt in critical_data_types if dt in fetched_data)
            
            quality_assessment['completeness_score'] = (available_critical / len(critical_data_types)) * 100
            
            # Identify missing critical data
            quality_assessment['missing_critical_data'] = [
                dt for dt in critical_data_types if dt not in fetched_data
            ]
            
            # Assess reliability based on successful fetches
            total_attempted = len(self._analyze_data_requirements())
            successful_fetches = len(fetched_data)
            quality_assessment['reliability_score'] = (successful_fetches / total_attempted) * 100 if total_attempted > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {str(e)}")
        
        return quality_assessment
    
    def get_endpoint_status(self) -> Dict[str, Any]:
        """
        Get status of all endpoints
        
        Returns:
            Dict with endpoint status information
        """
        return {
            'available_endpoints': list(self.available_endpoints.keys()),
            'total_endpoints': len(self.base_endpoints),
            'availability_rate': len(self.available_endpoints) / len(self.base_endpoints) * 100,
            'endpoint_details': self.available_endpoints
        }
    
    def test_connectivity(self) -> Dict[str, Any]:
        """
        Test connectivity to all endpoints
        
        Returns:
            Dict with connectivity test results
        """
        connectivity_results = {}
        
        for endpoint_name, base_url in self.base_endpoints.items():
            health_check = self._test_endpoint_health(base_url)
            connectivity_results[endpoint_name] = {
                'url': base_url,
                'available': health_check['available'],
                'response_time': health_check.get('response_time'),
                'status': 'healthy' if health_check['available'] else 'unreachable'
            }
        
        return {
            'test_timestamp': datetime.now().isoformat(),
            'results': connectivity_results,
            'summary': {
                'total_tested': len(self.base_endpoints),
                'available': sum(1 for r in connectivity_results.values() if r['available']),
                'unavailable': sum(1 for r in connectivity_results.values() if not r['available'])
            }
        }
    
    def fetch_batch_data(self, batch_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fetch data in batch for improved performance
        
        Args:
            batch_requests: List of request specifications
            
        Returns:
            Dict with batch fetch results
        """
        batch_results = {}
        
        # Use concurrent requests for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_request = {}
            
            for i, request in enumerate(batch_requests):
                if self._endpoint_available(request.get('endpoint', '')):
                    future = executor.submit(
                        self._fetch_from_endpoint,
                        request['endpoint'],
                        request.get('params', {}),
                        request.get('data_type', f'batch_request_{i}')
                    )
                    future_to_request[future] = request
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    result = future.result()
                    if result is not None:
                        batch_results[request.get('data_type', 'unknown')] = result
                except Exception as e:
                    self.logger.error(f"Batch request failed: {str(e)}")
        
        return batch_results
    
    def get_data_schema(self, data_type: str) -> Optional[Dict[str, Any]]:
        """
        Get schema information for a specific data type
        
        Args:
            data_type: Type of data to get schema for
            
        Returns:
            Schema information or None if not available
        """
        schemas = {
            'normalized_scores': {
                'required_columns': ['Metric', 'section_name', 'platform_name'],
                'brand_columns': 'dynamic',
                'data_types': {
                    'Metric': 'string',
                    'section_name': 'string',
                    'platform_name': 'string',
                    'brand_scores': 'numeric'
                },
                'expected_sections': ['Marketplace', 'Digital Spends', 'Organic Performance', 'Socialwatch']
            },
            'project_metadata': {
                'required_fields': ['project_name', 'project_id', 'metrics'],
                'optional_fields': ['description', 'created_at', 'updated_at'],
                'nested_structures': ['metrics', 'categories', 'platforms']
            }
        }
        
        return schemas.get(data_type)

