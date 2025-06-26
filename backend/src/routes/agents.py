"""
Agent management routes for Digi-Cadence Portfolio Management Platform
Handles agent lifecycle, communication, and coordination
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import asyncio
import logging

from src.agents.base_agent import agent_manager, AgentCapability, TaskPriority
from src.agents.portfolio_optimization_agent import create_portfolio_optimization_agent
from src.agents.multi_brand_metric_optimization_agent import create_multi_brand_metric_optimization_agent
from src.agents.portfolio_forecasting_agent import create_portfolio_forecasting_agent
from src.agents.portfolio_strategy_agent import create_portfolio_strategy_agent

agents_bp = Blueprint('agents', __name__)
logger = logging.getLogger(__name__)

@agents_bp.route('/agents', methods=['GET'])
def get_all_agents():
    """Get status of all agents"""
    try:
        agents_status = agent_manager.get_all_agents_status()
        
        return jsonify({
            'success': True,
            'agents': agents_status,
            'total_agents': len(agents_status),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting agents status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/<agent_id>', methods=['GET'])
def get_agent_status(agent_id):
    """Get status of specific agent"""
    try:
        agent = agent_manager.get_agent(agent_id)
        
        if not agent:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_id} not found'
            }), 404
        
        agent_status = agent.get_status()
        agent_metrics = agent.get_metrics()
        
        return jsonify({
            'success': True,
            'agent_status': agent_status,
            'agent_metrics': agent_metrics,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting agent {agent_id} status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/create', methods=['POST'])
def create_agent():
    """Create a new agent"""
    try:
        data = request.get_json()
        agent_type = data.get('agent_type')
        agent_config = data.get('config', {})
        
        if not agent_type:
            return jsonify({
                'success': False,
                'error': 'agent_type is required'
            }), 400
        
        # Create agent based on type
        if agent_type == 'portfolio_optimization':
            agent = create_portfolio_optimization_agent(agent_config)
        elif agent_type == 'multi_brand_metric_optimization':
            agent = create_multi_brand_metric_optimization_agent(agent_config)
        elif agent_type == 'portfolio_forecasting':
            agent = create_portfolio_forecasting_agent(agent_config)
        elif agent_type == 'portfolio_strategy':
            agent = create_portfolio_strategy_agent(agent_config)
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown agent type: {agent_type}'
            }), 400
        
        # Register agent
        agent_manager.register_agent(agent)
        
        return jsonify({
            'success': True,
            'agent_id': agent.agent_id,
            'agent_name': agent.name,
            'agent_type': agent_type,
            'capabilities': [c.value for c in agent.capabilities],
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/<agent_id>/start', methods=['POST'])
def start_agent(agent_id):
    """Start an agent"""
    try:
        agent = agent_manager.get_agent(agent_id)
        
        if not agent:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_id} not found'
            }), 404
        
        # Start agent asynchronously
        asyncio.create_task(agent.start())
        
        return jsonify({
            'success': True,
            'message': f'Agent {agent_id} started',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting agent {agent_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/<agent_id>/stop', methods=['POST'])
def stop_agent(agent_id):
    """Stop an agent"""
    try:
        agent = agent_manager.get_agent(agent_id)
        
        if not agent:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_id} not found'
            }), 404
        
        # Stop agent asynchronously
        asyncio.create_task(agent.stop())
        
        return jsonify({
            'success': True,
            'message': f'Agent {agent_id} stopped',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error stopping agent {agent_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/<agent_id>/pause', methods=['POST'])
def pause_agent(agent_id):
    """Pause an agent"""
    try:
        agent = agent_manager.get_agent(agent_id)
        
        if not agent:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_id} not found'
            }), 404
        
        # Pause agent asynchronously
        asyncio.create_task(agent.pause())
        
        return jsonify({
            'success': True,
            'message': f'Agent {agent_id} paused',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error pausing agent {agent_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/<agent_id>/resume', methods=['POST'])
def resume_agent(agent_id):
    """Resume an agent"""
    try:
        agent = agent_manager.get_agent(agent_id)
        
        if not agent:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_id} not found'
            }), 404
        
        # Resume agent asynchronously
        asyncio.create_task(agent.resume())
        
        return jsonify({
            'success': True,
            'message': f'Agent {agent_id} resumed',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error resuming agent {agent_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/<agent_id>/tasks', methods=['POST'])
def add_agent_task(agent_id):
    """Add a task to an agent"""
    try:
        agent = agent_manager.get_agent(agent_id)
        
        if not agent:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_id} not found'
            }), 404
        
        data = request.get_json()
        task_type = data.get('task_type')
        parameters = data.get('parameters', {})
        priority = data.get('priority', 'medium')
        scheduled_at = data.get('scheduled_at')
        
        if not task_type:
            return jsonify({
                'success': False,
                'error': 'task_type is required'
            }), 400
        
        # Convert priority string to enum
        priority_enum = TaskPriority.MEDIUM
        if priority.lower() == 'low':
            priority_enum = TaskPriority.LOW
        elif priority.lower() == 'high':
            priority_enum = TaskPriority.HIGH
        elif priority.lower() == 'critical':
            priority_enum = TaskPriority.CRITICAL
        
        # Parse scheduled_at if provided
        scheduled_datetime = None
        if scheduled_at:
            try:
                scheduled_datetime = datetime.fromisoformat(scheduled_at)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid scheduled_at format. Use ISO format.'
                }), 400
        
        # Add task asynchronously
        async def add_task():
            return await agent.add_task(task_type, parameters, priority_enum, scheduled_datetime)
        
        task_id = asyncio.run(add_task())
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'agent_id': agent_id,
            'task_type': task_type,
            'priority': priority,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error adding task to agent {agent_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/<agent_id>/tasks/<task_id>', methods=['GET'])
def get_task_status(agent_id, task_id):
    """Get status of a specific task"""
    try:
        agent = agent_manager.get_agent(agent_id)
        
        if not agent:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_id} not found'
            }), 404
        
        # Get task status asynchronously
        async def get_status():
            return await agent.get_task_status(task_id)
        
        task_status = asyncio.run(get_status())
        
        if not task_status:
            return jsonify({
                'success': False,
                'error': f'Task {task_id} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'task_status': task_status,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting task {task_id} status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/<agent_id>/tasks/<task_id>/cancel', methods=['POST'])
def cancel_task(agent_id, task_id):
    """Cancel a task"""
    try:
        agent = agent_manager.get_agent(agent_id)
        
        if not agent:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_id} not found'
            }), 404
        
        # Cancel task asynchronously
        async def cancel():
            return await agent.cancel_task(task_id)
        
        cancelled = asyncio.run(cancel())
        
        if not cancelled:
            return jsonify({
                'success': False,
                'error': f'Task {task_id} could not be cancelled'
            }), 400
        
        return jsonify({
            'success': True,
            'message': f'Task {task_id} cancelled',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/broadcast', methods=['POST'])
def broadcast_message():
    """Broadcast message to all agents"""
    try:
        data = request.get_json()
        message_type = data.get('message_type')
        message_data = data.get('data', {})
        sender_id = data.get('sender_id', 'system')
        
        if not message_type:
            return jsonify({
                'success': False,
                'error': 'message_type is required'
            }), 400
        
        # Broadcast message asynchronously
        async def broadcast():
            await agent_manager.broadcast_message(message_type, message_data, sender_id)
        
        asyncio.run(broadcast())
        
        return jsonify({
            'success': True,
            'message': f'Message {message_type} broadcasted to all agents',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/<sender_id>/message/<recipient_id>', methods=['POST'])
def send_message(sender_id, recipient_id):
    """Send message between agents"""
    try:
        data = request.get_json()
        message_type = data.get('message_type')
        message_data = data.get('data', {})
        
        if not message_type:
            return jsonify({
                'success': False,
                'error': 'message_type is required'
            }), 400
        
        # Route message asynchronously
        async def route():
            return await agent_manager.route_message(sender_id, recipient_id, message_type, message_data)
        
        routed = asyncio.run(route())
        
        if not routed:
            return jsonify({
                'success': False,
                'error': f'Could not route message to {recipient_id}'
            }), 400
        
        return jsonify({
            'success': True,
            'message': f'Message sent from {sender_id} to {recipient_id}',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/start-all', methods=['POST'])
def start_all_agents():
    """Start all registered agents"""
    try:
        # Start all agents asynchronously
        async def start_all():
            await agent_manager.start_all_agents()
        
        asyncio.run(start_all())
        
        agents_count = len(agent_manager.agents)
        
        return jsonify({
            'success': True,
            'message': f'Started {agents_count} agents',
            'agents_started': agents_count,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting all agents: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/stop-all', methods=['POST'])
def stop_all_agents():
    """Stop all registered agents"""
    try:
        # Stop all agents asynchronously
        async def stop_all():
            await agent_manager.stop_all_agents()
        
        asyncio.run(stop_all())
        
        agents_count = len(agent_manager.agents)
        
        return jsonify({
            'success': True,
            'message': f'Stopped {agents_count} agents',
            'agents_stopped': agents_count,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error stopping all agents: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/initialize-default', methods=['POST'])
def initialize_default_agents():
    """Initialize default set of agents for the platform"""
    try:
        # Create default agents
        agents_created = []
        
        # Portfolio Optimization Agent
        portfolio_optimizer = create_portfolio_optimization_agent()
        agent_manager.register_agent(portfolio_optimizer)
        agents_created.append({
            'agent_id': portfolio_optimizer.agent_id,
            'name': portfolio_optimizer.name,
            'type': 'portfolio_optimization'
        })
        
        # Multi-Brand Metric Optimization Agent
        multi_brand_optimizer = create_multi_brand_metric_optimization_agent()
        agent_manager.register_agent(multi_brand_optimizer)
        agents_created.append({
            'agent_id': multi_brand_optimizer.agent_id,
            'name': multi_brand_optimizer.name,
            'type': 'multi_brand_metric_optimization'
        })
        
        # Portfolio Forecasting Agent
        portfolio_forecaster = create_portfolio_forecasting_agent()
        agent_manager.register_agent(portfolio_forecaster)
        agents_created.append({
            'agent_id': portfolio_forecaster.agent_id,
            'name': portfolio_forecaster.name,
            'type': 'portfolio_forecasting'
        })
        
        # Portfolio Strategy Agent
        portfolio_strategist = create_portfolio_strategy_agent()
        agent_manager.register_agent(portfolio_strategist)
        agents_created.append({
            'agent_id': portfolio_strategist.agent_id,
            'name': portfolio_strategist.name,
            'type': 'portfolio_strategy'
        })
        
        # Start all agents
        async def start_all():
            await agent_manager.start_all_agents()
        
        asyncio.run(start_all())
        
        return jsonify({
            'success': True,
            'message': 'Default agents initialized and started',
            'agents_created': agents_created,
            'total_agents': len(agents_created),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error initializing default agents: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/agents/health', methods=['GET'])
def agents_health_check():
    """Health check for all agents"""
    try:
        agents_status = agent_manager.get_all_agents_status()
        
        health_summary = {
            'total_agents': len(agents_status),
            'running_agents': 0,
            'idle_agents': 0,
            'error_agents': 0,
            'paused_agents': 0,
            'overall_health': 'healthy'
        }
        
        for agent_id, status in agents_status.items():
            agent_status = status.get('status', 'unknown')
            if agent_status == 'running':
                health_summary['running_agents'] += 1
            elif agent_status == 'idle':
                health_summary['idle_agents'] += 1
            elif agent_status == 'error':
                health_summary['error_agents'] += 1
            elif agent_status == 'paused':
                health_summary['paused_agents'] += 1
        
        # Determine overall health
        if health_summary['error_agents'] > 0:
            health_summary['overall_health'] = 'unhealthy'
        elif health_summary['running_agents'] == 0 and health_summary['idle_agents'] == 0:
            health_summary['overall_health'] = 'down'
        elif health_summary['paused_agents'] > health_summary['total_agents'] / 2:
            health_summary['overall_health'] = 'degraded'
        
        return jsonify({
            'success': True,
            'health_summary': health_summary,
            'agents_status': agents_status,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error checking agents health: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

