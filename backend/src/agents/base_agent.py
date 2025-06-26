"""
Base Agent Framework for Digi-Cadence Portfolio Management Platform
Provides foundation for intelligent agents that orchestrate analytics and optimization
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
import threading
import time
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.analytics.genetic_optimizer import GeneticPortfolioOptimizer
from src.analytics.shap_analyzer import SHAPPortfolioAnalyzer
from src.analytics.correlation_analyzer import CorrelationAnalyzer
from src.analytics.competitive_gap_analyzer import CompetitiveGapAnalyzer
from src.analytics.trend_analyzer import TrendAnalyzer
from src.models.portfolio import Project, Brand, Organization

class AgentStatus(Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AgentCapability(Enum):
    """Agent capability types"""
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    FORECASTING = "forecasting"
    MONITORING = "monitoring"
    REPORTING = "reporting"
    STRATEGY = "strategy"

@dataclass
class AgentTask:
    """Represents a task for an agent"""
    task_id: str
    task_type: str
    priority: TaskPriority
    parameters: Dict[str, Any]
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
            elif isinstance(value, TaskPriority):
                data[key] = value.value
        return data

@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time: float = 0.0
    last_activity: Optional[datetime] = None
    uptime_seconds: float = 0.0
    success_rate: float = 0.0
    
    def update_success_rate(self):
        """Update success rate based on completed and failed tasks"""
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks > 0:
            self.success_rate = self.tasks_completed / total_tasks
        else:
            self.success_rate = 0.0

class BaseAgent(ABC):
    """
    Base class for all intelligent agents in the Digi-Cadence platform
    Provides common functionality for task management, communication, and analytics integration
    """
    
    def __init__(self, agent_id: str, name: str, capabilities: List[AgentCapability], 
                 config: Dict[str, Any] = None):
        """
        Initialize base agent
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            capabilities: List of agent capabilities
            config: Agent configuration parameters
        """
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.config = config or {}
        
        # Agent state
        self.status = AgentStatus.IDLE
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.last_heartbeat = datetime.utcnow()
        
        # Task management
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.completed_tasks = []
        self.task_history = []
        
        # Performance metrics
        self.metrics = AgentMetrics()
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        self.running = False
        self.agent_loop_task = None
        
        # Analytics engines (will be injected)
        self.genetic_optimizer = None
        self.shap_analyzer = None
        self.correlation_analyzer = None
        self.competitive_gap_analyzer = None
        self.trend_analyzer = None
        
        # Communication
        self.message_handlers = {}
        self.event_listeners = {}
        
        # Logging
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        self.logger.setLevel(logging.INFO)
        
        # Configuration validation
        self._validate_config()
        
        self.logger.info(f"Agent {self.name} ({self.agent_id}) initialized with capabilities: {[c.value for c in capabilities]}")
    
    def _validate_config(self):
        """Validate agent configuration"""
        required_configs = self.get_required_config_keys()
        for key in required_configs:
            if key not in self.config:
                raise ValueError(f"Required configuration key '{key}' missing for agent {self.name}")
    
    @abstractmethod
    def get_required_config_keys(self) -> List[str]:
        """Return list of required configuration keys"""
        pass
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """
        Process a specific task - must be implemented by subclasses
        
        Args:
            task: Task to process
            
        Returns:
            Task result dictionary
        """
        pass
    
    def inject_analytics_engines(self, projects: List[Project], brands: List[Brand]):
        """Inject analytics engines for the agent to use"""
        try:
            self.genetic_optimizer = GeneticPortfolioOptimizer(projects, brands)
            self.shap_analyzer = SHAPPortfolioAnalyzer(projects, brands)
            self.correlation_analyzer = CorrelationAnalyzer(projects, brands)
            self.competitive_gap_analyzer = CompetitiveGapAnalyzer(projects, brands)
            self.trend_analyzer = TrendAnalyzer(projects, brands)
            
            self.logger.info("Analytics engines injected successfully")
        except Exception as e:
            self.logger.error(f"Failed to inject analytics engines: {e}")
            raise
    
    async def start(self):
        """Start the agent"""
        if self.running:
            self.logger.warning("Agent is already running")
            return
        
        self.running = True
        self.started_at = datetime.utcnow()
        self.status = AgentStatus.RUNNING
        
        # Start the main agent loop
        self.agent_loop_task = asyncio.create_task(self._agent_loop())
        
        self.logger.info(f"Agent {self.name} started")
    
    async def stop(self):
        """Stop the agent"""
        if not self.running:
            self.logger.warning("Agent is not running")
            return
        
        self.running = False
        self.status = AgentStatus.IDLE
        
        # Cancel the agent loop
        if self.agent_loop_task:
            self.agent_loop_task.cancel()
            try:
                await self.agent_loop_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info(f"Agent {self.name} stopped")
    
    async def pause(self):
        """Pause the agent"""
        if self.status == AgentStatus.RUNNING:
            self.status = AgentStatus.PAUSED
            self.logger.info(f"Agent {self.name} paused")
    
    async def resume(self):
        """Resume the agent"""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.RUNNING
            self.logger.info(f"Agent {self.name} resumed")
    
    async def add_task(self, task_type: str, parameters: Dict[str, Any], 
                      priority: TaskPriority = TaskPriority.MEDIUM,
                      scheduled_at: Optional[datetime] = None) -> str:
        """
        Add a task to the agent's queue
        
        Args:
            task_type: Type of task to execute
            parameters: Task parameters
            priority: Task priority
            scheduled_at: When to execute the task (None for immediate)
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = AgentTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            parameters=parameters,
            created_at=datetime.utcnow(),
            scheduled_at=scheduled_at
        )
        
        await self.task_queue.put(task)
        self.logger.info(f"Task {task_id} ({task_type}) added to queue with priority {priority.name}")
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return task.to_dict()
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or active task"""
        # Remove from queue (this is simplified - in practice would need queue manipulation)
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = "cancelled"
            task.completed_at = datetime.utcnow()
            self.completed_tasks.append(task)
            del self.active_tasks[task_id]
            self.logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    async def _agent_loop(self):
        """Main agent processing loop"""
        self.logger.info("Agent loop started")
        
        while self.running:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.utcnow()
                
                # Update uptime
                if self.started_at:
                    self.metrics.uptime_seconds = (datetime.utcnow() - self.started_at).total_seconds()
                
                # Skip processing if paused
                if self.status == AgentStatus.PAUSED:
                    await asyncio.sleep(1)
                    continue
                
                # Process scheduled tasks
                await self._process_scheduled_tasks()
                
                # Get next task from queue
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                    await self._execute_task(task)
                except asyncio.TimeoutError:
                    # No task available, continue loop
                    pass
                
                # Cleanup completed tasks
                await self._cleanup_old_tasks()
                
                # Update metrics
                self.metrics.last_activity = datetime.utcnow()
                self.metrics.update_success_rate()
                
            except Exception as e:
                self.logger.error(f"Error in agent loop: {e}")
                self.status = AgentStatus.ERROR
                await asyncio.sleep(5)  # Wait before retrying
                self.status = AgentStatus.RUNNING
    
    async def _process_scheduled_tasks(self):
        """Process tasks that are scheduled for execution"""
        current_time = datetime.utcnow()
        
        # This is a simplified implementation
        # In practice, would need a proper scheduler
        for task in list(self.active_tasks.values()):
            if (task.scheduled_at and task.scheduled_at <= current_time and 
                task.status == "scheduled"):
                await self._execute_task(task)
    
    async def _execute_task(self, task: AgentTask):
        """Execute a single task"""
        task.started_at = datetime.utcnow()
        task.status = "running"
        self.active_tasks[task.task_id] = task
        
        self.logger.info(f"Executing task {task.task_id} ({task.task_type})")
        
        try:
            # Execute the task
            start_time = time.time()
            result = await self.process_task(task)
            execution_time = time.time() - start_time
            
            # Update task
            task.completed_at = datetime.utcnow()
            task.status = "completed"
            task.result = result
            
            # Update metrics
            self.metrics.tasks_completed += 1
            if self.metrics.tasks_completed == 1:
                self.metrics.average_execution_time = execution_time
            else:
                self.metrics.average_execution_time = (
                    (self.metrics.average_execution_time * (self.metrics.tasks_completed - 1) + execution_time) /
                    self.metrics.tasks_completed
                )
            
            self.logger.info(f"Task {task.task_id} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
            task.error = str(e)
            task.retry_count += 1
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.status = "pending"
                task.started_at = None
                await self.task_queue.put(task)  # Re-queue for retry
                self.logger.info(f"Task {task.task_id} queued for retry ({task.retry_count}/{task.max_retries})")
            else:
                task.status = "failed"
                task.completed_at = datetime.utcnow()
                self.metrics.tasks_failed += 1
                self.logger.error(f"Task {task.task_id} failed permanently after {task.max_retries} retries")
        
        finally:
            # Move to completed tasks
            if task.status in ["completed", "failed"]:
                self.completed_tasks.append(task)
                self.task_history.append(task.to_dict())
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
    
    async def _cleanup_old_tasks(self):
        """Clean up old completed tasks to prevent memory leaks"""
        max_completed_tasks = self.config.get('max_completed_tasks', 1000)
        max_task_history = self.config.get('max_task_history', 5000)
        
        # Keep only recent completed tasks
        if len(self.completed_tasks) > max_completed_tasks:
            self.completed_tasks = self.completed_tasks[-max_completed_tasks:]
        
        # Keep only recent task history
        if len(self.task_history) > max_task_history:
            self.task_history = self.task_history[-max_task_history:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'status': self.status.value,
            'capabilities': [c.value for c in self.capabilities],
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'queue_size': self.task_queue.qsize(),
            'metrics': {
                'tasks_completed': self.metrics.tasks_completed,
                'tasks_failed': self.metrics.tasks_failed,
                'success_rate': self.metrics.success_rate,
                'average_execution_time': self.metrics.average_execution_time,
                'uptime_seconds': self.metrics.uptime_seconds
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed agent metrics"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'metrics': asdict(self.metrics),
            'task_distribution': self._get_task_distribution(),
            'performance_trends': self._get_performance_trends()
        }
    
    def _get_task_distribution(self) -> Dict[str, int]:
        """Get distribution of task types"""
        distribution = {}
        for task in self.task_history:
            task_type = task.get('task_type', 'unknown')
            distribution[task_type] = distribution.get(task_type, 0) + 1
        return distribution
    
    def _get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends over time"""
        # Simplified implementation
        recent_tasks = [t for t in self.task_history if 
                       datetime.fromisoformat(t['created_at']) > datetime.utcnow() - timedelta(hours=24)]
        
        return {
            'tasks_last_24h': len(recent_tasks),
            'success_rate_last_24h': len([t for t in recent_tasks if t['status'] == 'completed']) / len(recent_tasks) if recent_tasks else 0,
            'average_execution_time_last_24h': sum([t.get('execution_time', 0) for t in recent_tasks]) / len(recent_tasks) if recent_tasks else 0
        }
    
    async def send_message(self, recipient_agent_id: str, message_type: str, data: Dict[str, Any]):
        """Send message to another agent"""
        message = {
            'sender_id': self.agent_id,
            'recipient_id': recipient_agent_id,
            'message_type': message_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # In a real implementation, this would use a message broker
        self.logger.info(f"Sending message to {recipient_agent_id}: {message_type}")
        
        # For now, just log the message
        return message
    
    async def handle_message(self, message: Dict[str, Any]):
        """Handle incoming message from another agent"""
        message_type = message.get('message_type')
        sender_id = message.get('sender_id')
        data = message.get('data', {})
        
        self.logger.info(f"Received message from {sender_id}: {message_type}")
        
        # Call registered message handler
        if message_type in self.message_handlers:
            try:
                await self.message_handlers[message_type](sender_id, data)
            except Exception as e:
                self.logger.error(f"Error handling message {message_type}: {e}")
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a message handler for a specific message type"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered handler for message type: {message_type}")
    
    def register_event_listener(self, event_type: str, listener: Callable):
        """Register an event listener"""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(listener)
        self.logger.info(f"Registered listener for event type: {event_type}")
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to all registered listeners"""
        if event_type in self.event_listeners:
            for listener in self.event_listeners[event_type]:
                try:
                    await listener(self.agent_id, event_type, data)
                except Exception as e:
                    self.logger.error(f"Error in event listener for {event_type}: {e}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update agent configuration"""
        self.config.update(new_config)
        self.logger.info(f"Agent configuration updated: {new_config}")
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.agent_id}, name={self.name}, status={self.status.value})>"

class AgentManager:
    """
    Manages multiple agents and coordinates their activities
    """
    
    def __init__(self):
        self.agents = {}
        self.agent_registry = {}
        self.message_broker = {}
        self.logger = logging.getLogger("agent_manager")
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the manager"""
        self.agents[agent.agent_id] = agent
        self.agent_registry[agent.name] = agent.agent_id
        self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            del self.agents[agent_id]
            # Remove from registry
            for name, aid in list(self.agent_registry.items()):
                if aid == agent_id:
                    del self.agent_registry[name]
                    break
            self.logger.info(f"Unregistered agent: {agent.name} ({agent_id})")
    
    async def start_all_agents(self):
        """Start all registered agents"""
        for agent in self.agents.values():
            await agent.start()
        self.logger.info(f"Started {len(self.agents)} agents")
    
    async def stop_all_agents(self):
        """Stop all registered agents"""
        for agent in self.agents.values():
            await agent.stop()
        self.logger.info(f"Stopped {len(self.agents)} agents")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agent_by_name(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        agent_id = self.agent_registry.get(name)
        return self.agents.get(agent_id) if agent_id else None
    
    def get_all_agents_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        return {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}
    
    async def broadcast_message(self, message_type: str, data: Dict[str, Any], 
                               sender_id: str = "system"):
        """Broadcast message to all agents"""
        message = {
            'sender_id': sender_id,
            'message_type': message_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for agent in self.agents.values():
            if agent.agent_id != sender_id:  # Don't send to sender
                await agent.handle_message(message)
    
    async def route_message(self, sender_id: str, recipient_id: str, 
                           message_type: str, data: Dict[str, Any]):
        """Route message between agents"""
        if recipient_id in self.agents:
            message = {
                'sender_id': sender_id,
                'recipient_id': recipient_id,
                'message_type': message_type,
                'data': data,
                'timestamp': datetime.utcnow().isoformat()
            }
            await self.agents[recipient_id].handle_message(message)
            return True
        return False

# Global agent manager instance
agent_manager = AgentManager()

