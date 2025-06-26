"""
MCP Server Management Routes for Digi-Cadence Platform
Handles communication with distributed MCP servers and server management
"""

from flask import Blueprint, request, jsonify, current_app
import aiohttp
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
import logging

from src.models.portfolio import db, MCPServer, Agent
from src.mcp_servers.base_server import MCPRequest, MCPResponse

mcp_bp = Blueprint('mcp', __name__)
logger = logging.getLogger(__name__)

class MCPServerManager:
    """Manager for communicating with MCP servers"""
    
    def __init__(self):
        self.server_connections = {}
        self.server_health_status = {}
    
    async def send_request(self, server_type: str, method: str, params: Dict[str, Any], 
                          user_id: str = None, organization_id: str = None, 
                          authorization_level: str = 'read') -> Dict[str, Any]:
        """Send request to specific MCP server"""
        
        # Get server configuration
        server_config = current_app.config.get('MCP_SERVERS', {}).get(server_type)
        if not server_config:
            raise ValueError(f"Unknown server type: {server_type}")
        
        # Create MCP request
        mcp_request = MCPRequest(
            request_id=str(uuid.uuid4()),
            method=method,
            params=params,
            user_id=user_id,
            organization_id=organization_id,
            authorization_level=authorization_level
        )
        
        # Send HTTP request to MCP server
        url = f"http://{server_config['host']}:{server_config['port']}/mcp"
        timeout = aiohttp.ClientTimeout(total=server_config.get('timeout', 300))
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=mcp_request.__dict__) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"MCP server error: {response.status} - {error_text}")
        
        except Exception as e:
            logger.error(f"Error communicating with {server_type} server: {e}")
            raise
    
    async def check_server_health(self, server_type: str) -> Dict[str, Any]:
        """Check health of specific MCP server"""
        try:
            result = await self.send_request(server_type, 'health', {})
            self.server_health_status[server_type] = {
                'status': 'healthy',
                'last_check': datetime.utcnow().isoformat(),
                'details': result
            }
            return result
        except Exception as e:
            self.server_health_status[server_type] = {
                'status': 'unhealthy',
                'last_check': datetime.utcnow().isoformat(),
                'error': str(e)
            }
            raise
    
    async def check_all_servers_health(self) -> Dict[str, Any]:
        """Check health of all MCP servers"""
        server_types = ['analysis', 'reporting', 'integration', 'orchestration']
        health_results = {}
        
        for server_type in server_types:
            try:
                health_results[server_type] = await self.check_server_health(server_type)
            except Exception as e:
                health_results[server_type] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        return health_results

# Global MCP server manager
mcp_manager = MCPServerManager()

# Server Management Endpoints
@mcp_bp.route('/servers', methods=['GET'])
def get_mcp_servers():
    """Get list of all MCP servers and their status"""
    try:
        servers = MCPServer.query.filter(MCPServer.is_active == True).all()
        
        server_list = []
        for server in servers:
            server_info = {
                'id': str(server.id),
                'name': server.name,
                'server_type': server.server_type,
                'host': server.host,
                'port': server.port,
                'status': server.status,
                'version': server.version,
                'last_health_check': server.last_health_check.isoformat() if server.last_health_check else None,
                'created_at': server.created_at.isoformat()
            }
            server_list.append(server_info)
        
        return jsonify({
            'servers': server_list,
            'total_servers': len(server_list),
            'active_servers': len([s for s in server_list if s['status'] == 'active'])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mcp_bp.route('/servers/health', methods=['GET'])
def check_servers_health():
    """Check health of all MCP servers"""
    try:
        # Run async health check
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            health_results = loop.run_until_complete(mcp_manager.check_all_servers_health())
        finally:
            loop.close()
        
        # Update database with health status
        for server_type, health_data in health_results.items():
            server = MCPServer.query.filter_by(server_type=server_type).first()
            if server:
                server.status = 'active' if health_data.get('status') == 'healthy' else 'error'
                server.last_health_check = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'health_check_results': health_results,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mcp_bp.route('/servers/<server_type>/health', methods=['GET'])
def check_server_health(server_type):
    """Check health of specific MCP server"""
    try:
        # Run async health check
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            health_result = loop.run_until_complete(mcp_manager.check_server_health(server_type))
        finally:
            loop.close()
        
        return jsonify({
            'server_type': server_type,
            'health_result': health_result,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Analysis Server Endpoints
@mcp_bp.route('/analysis/optimize', methods=['POST'])
def request_portfolio_optimization():
    """Request portfolio optimization from Analysis MCP Server"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['project_ids', 'brand_ids']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract user information (would come from JWT token in production)
        user_id = data.get('user_id', 'anonymous')
        organization_id = data.get('organization_id')
        authorization_level = data.get('authorization_level', 'write')
        
        # Prepare parameters
        params = {
            'project_ids': data['project_ids'],
            'brand_ids': data['brand_ids'],
            'optimization_params': data.get('optimization_params', {})
        }
        
        # Run async request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                mcp_manager.send_request(
                    'analysis', 
                    'optimize_portfolio', 
                    params,
                    user_id=user_id,
                    organization_id=organization_id,
                    authorization_level=authorization_level
                )
            )
        finally:
            loop.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mcp_bp.route('/analysis/shap', methods=['POST'])
def request_shap_analysis():
    """Request SHAP analysis from Analysis MCP Server"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['project_ids', 'brand_ids', 'target_metric']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        user_id = data.get('user_id', 'anonymous')
        organization_id = data.get('organization_id')
        authorization_level = data.get('authorization_level', 'write')
        
        params = {
            'project_ids': data['project_ids'],
            'brand_ids': data['brand_ids'],
            'target_metric': data['target_metric'],
            'shap_params': data.get('shap_params', {})
        }
        
        # Run async request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                mcp_manager.send_request(
                    'analysis', 
                    'shap_analysis', 
                    params,
                    user_id=user_id,
                    organization_id=organization_id,
                    authorization_level=authorization_level
                )
            )
        finally:
            loop.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mcp_bp.route('/analysis/multi-brand', methods=['POST'])
def request_multi_brand_optimization():
    """Request multi-brand optimization from Analysis MCP Server"""
    try:
        data = request.get_json()
        
        if 'brand_groups' not in data:
            return jsonify({'error': 'Missing brand_groups field'}), 400
        
        user_id = data.get('user_id', 'anonymous')
        organization_id = data.get('organization_id')
        authorization_level = data.get('authorization_level', 'write')
        
        params = {
            'brand_groups': data['brand_groups'],
            'project_id': data.get('project_id'),
            'optimization_params': data.get('optimization_params', {})
        }
        
        # Run async request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                mcp_manager.send_request(
                    'analysis', 
                    'optimize_multi_brand', 
                    params,
                    user_id=user_id,
                    organization_id=organization_id,
                    authorization_level=authorization_level
                )
            )
        finally:
            loop.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Job Management Endpoints
@mcp_bp.route('/jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    """Get status of a specific analysis job"""
    try:
        user_id = request.args.get('user_id', 'anonymous')
        
        params = {'job_id': job_id}
        
        # Run async request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                mcp_manager.send_request(
                    'analysis', 
                    'get_job_status', 
                    params,
                    user_id=user_id
                )
            )
        finally:
            loop.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mcp_bp.route('/jobs', methods=['GET'])
def list_user_jobs():
    """List all jobs for the current user"""
    try:
        user_id = request.args.get('user_id', 'anonymous')
        
        params = {}
        
        # Run async request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                mcp_manager.send_request(
                    'analysis', 
                    'list_jobs', 
                    params,
                    user_id=user_id
                )
            )
        finally:
            loop.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mcp_bp.route('/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Cancel a specific analysis job"""
    try:
        user_id = request.args.get('user_id', 'anonymous')
        authorization_level = request.args.get('authorization_level', 'write')
        
        params = {'job_id': job_id}
        
        # Run async request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                mcp_manager.send_request(
                    'analysis', 
                    'cancel_job', 
                    params,
                    user_id=user_id,
                    authorization_level=authorization_level
                )
            )
        finally:
            loop.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Configuration Management
@mcp_bp.route('/servers/<server_type>/config', methods=['GET'])
def get_server_config(server_type):
    """Get configuration for specific MCP server"""
    try:
        params = {}
        
        # Run async request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                mcp_manager.send_request(server_type, 'get_config', params)
            )
        finally:
            loop.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mcp_bp.route('/servers/<server_type>/config', methods=['PUT'])
def update_server_config(server_type):
    """Update configuration for specific MCP server"""
    try:
        data = request.get_json()
        
        if 'config_updates' not in data:
            return jsonify({'error': 'Missing config_updates field'}), 400
        
        user_id = data.get('user_id', 'anonymous')
        authorization_level = data.get('authorization_level', 'admin')
        
        params = {'config_updates': data['config_updates']}
        
        # Run async request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                mcp_manager.send_request(
                    server_type, 
                    'update_config', 
                    params,
                    user_id=user_id,
                    authorization_level=authorization_level
                )
            )
        finally:
            loop.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Server Metrics
@mcp_bp.route('/servers/<server_type>/metrics', methods=['GET'])
def get_server_metrics(server_type):
    """Get metrics for specific MCP server"""
    try:
        params = {}
        
        # Run async request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                mcp_manager.send_request(server_type, 'metrics', params)
            )
        finally:
            loop.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Batch Operations
@mcp_bp.route('/analysis/batch', methods=['POST'])
def request_batch_analysis():
    """Request batch analysis from Analysis MCP Server"""
    try:
        data = request.get_json()
        
        required_fields = ['project_ids', 'brand_ids', 'analysis_types']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        user_id = data.get('user_id', 'anonymous')
        organization_id = data.get('organization_id')
        authorization_level = data.get('authorization_level', 'write')
        
        params = {
            'project_ids': data['project_ids'],
            'brand_ids': data['brand_ids'],
            'analysis_types': data['analysis_types'],
            'parameters': data.get('parameters', {})
        }
        
        # Run async request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                mcp_manager.send_request(
                    'analysis', 
                    'batch_analysis', 
                    params,
                    user_id=user_id,
                    organization_id=organization_id,
                    authorization_level=authorization_level
                )
            )
        finally:
            loop.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Server Registration and Management
@mcp_bp.route('/servers/register', methods=['POST'])
def register_mcp_server():
    """Register a new MCP server"""
    try:
        data = request.get_json()
        
        required_fields = ['name', 'server_type', 'host', 'port']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Check if server already exists
        existing_server = MCPServer.query.filter_by(
            name=data['name'],
            server_type=data['server_type']
        ).first()
        
        if existing_server:
            return jsonify({'error': 'Server already registered'}), 400
        
        # Create new server record
        server = MCPServer(
            name=data['name'],
            server_type=data['server_type'],
            host=data['host'],
            port=data['port'],
            version=data.get('version', '1.0.0'),
            configuration=data.get('configuration', {}),
            health_check_url=data.get('health_check_url'),
            status='inactive'
        )
        
        db.session.add(server)
        db.session.commit()
        
        return jsonify({
            'id': str(server.id),
            'message': 'MCP server registered successfully'
        }), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@mcp_bp.route('/servers/<server_id>', methods=['DELETE'])
def unregister_mcp_server(server_id):
    """Unregister an MCP server"""
    try:
        server = MCPServer.query.get(server_id)
        if not server:
            return jsonify({'error': 'Server not found'}), 404
        
        server.is_active = False
        db.session.commit()
        
        return jsonify({'message': 'MCP server unregistered successfully'})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Agent Management
@mcp_bp.route('/agents', methods=['GET'])
def get_agents():
    """Get list of all agents"""
    try:
        agents = Agent.query.filter(Agent.is_active == True).all()
        
        agent_list = []
        for agent in agents:
            agent_info = {
                'id': str(agent.id),
                'name': agent.name,
                'agent_type': agent.agent_type,
                'agent_class': agent.agent_class,
                'mcp_server_id': str(agent.mcp_server_id) if agent.mcp_server_id else None,
                'status': agent.status,
                'last_activity': agent.last_activity.isoformat() if agent.last_activity else None,
                'performance_metrics': agent.performance_metrics
            }
            agent_list.append(agent_info)
        
        return jsonify({
            'agents': agent_list,
            'total_agents': len(agent_list),
            'active_agents': len([a for a in agent_list if a['status'] == 'active'])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# System Status
@mcp_bp.route('/status', methods=['GET'])
def get_system_status():
    """Get overall MCP system status"""
    try:
        # Get server counts
        total_servers = MCPServer.query.filter(MCPServer.is_active == True).count()
        active_servers = MCPServer.query.filter(
            MCPServer.is_active == True,
            MCPServer.status == 'active'
        ).count()
        
        # Get agent counts
        total_agents = Agent.query.filter(Agent.is_active == True).count()
        active_agents = Agent.query.filter(
            Agent.is_active == True,
            Agent.status == 'active'
        ).count()
        
        # Get recent health check status
        recent_health = mcp_manager.server_health_status
        
        return jsonify({
            'system_status': {
                'servers': {
                    'total': total_servers,
                    'active': active_servers,
                    'health_status': recent_health
                },
                'agents': {
                    'total': total_agents,
                    'active': active_agents
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

