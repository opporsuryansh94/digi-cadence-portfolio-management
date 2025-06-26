"""
Base MCP Server Framework for Digi-Cadence Portfolio Management Platform
Implements robust authorization, error handling, and common functionality
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import jwt
from cryptography.fernet import Fernet
import redis
import aiohttp
from aiohttp import web, WSMsgType
import ssl

class MCPServerType(Enum):
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    INTEGRATION = "integration"
    ORCHESTRATION = "orchestration"

class AuthorizationLevel(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SYSTEM = "system"

@dataclass
class MCPRequest:
    """Standard MCP request structure"""
    request_id: str
    method: str
    params: Dict[str, Any]
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    authorization_level: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

@dataclass
class MCPResponse:
    """Standard MCP response structure"""
    request_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

@dataclass
class MCPServerConfig:
    """MCP Server configuration"""
    server_type: MCPServerType
    host: str = "0.0.0.0"
    port: int = 8000
    max_connections: int = 100
    timeout: int = 300
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str = "mcp-server-secret"
    encryption_key: Optional[str] = None
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_health_check: bool = True

class MCPServerError(Exception):
    """Base exception for MCP server errors"""
    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR", details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class AuthorizationError(MCPServerError):
    """Authorization related errors"""
    def __init__(self, message: str = "Unauthorized access", details: Dict[str, Any] = None):
        super().__init__(message, "AUTHORIZATION_ERROR", details)

class ValidationError(MCPServerError):
    """Request validation errors"""
    def __init__(self, message: str = "Invalid request", details: Dict[str, Any] = None):
        super().__init__(message, "VALIDATION_ERROR", details)

class BaseMCPServer(ABC):
    """
    Base MCP Server with robust authorization and error handling
    All MCP servers inherit from this base class
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.server_id = str(uuid.uuid4())
        self.start_time = datetime.utcnow()
        self.active_connections = set()
        self.request_handlers = {}
        self.middleware_stack = []
        
        # Setup logging
        self.logger = logging.getLogger(f"MCP-{config.server_type.value}")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Setup Redis for caching and session management
        self.redis_client = redis.from_url(config.redis_url)
        
        # Setup encryption
        if config.encryption_key:
            self.cipher = Fernet(config.encryption_key.encode())
        else:
            self.cipher = None
        
        # Metrics tracking
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_error': 0,
            'active_sessions': 0,
            'average_response_time': 0.0
        }
        
        # Register default handlers
        self._register_default_handlers()
        
        self.logger.info(f"Initialized {config.server_type.value} MCP Server with ID: {self.server_id}")
    
    def _register_default_handlers(self):
        """Register default request handlers"""
        self.register_handler("health", self._handle_health_check)
        self.register_handler("info", self._handle_server_info)
        self.register_handler("metrics", self._handle_metrics)
        self.register_handler("ping", self._handle_ping)
    
    def register_handler(self, method: str, handler: Callable):
        """Register a request handler for a specific method"""
        self.request_handlers[method] = handler
        self.logger.debug(f"Registered handler for method: {method}")
    
    def add_middleware(self, middleware: Callable):
        """Add middleware to the processing stack"""
        self.middleware_stack.append(middleware)
        self.logger.debug(f"Added middleware: {middleware.__name__}")
    
    async def start_server(self):
        """Start the MCP server"""
        try:
            app = web.Application()
            
            # Setup routes
            app.router.add_post('/mcp', self._handle_http_request)
            app.router.add_get('/ws', self._handle_websocket)
            app.router.add_get('/health', self._handle_http_health)
            app.router.add_get('/metrics', self._handle_http_metrics)
            
            # Setup SSL context if enabled
            ssl_context = None
            if self.config.ssl_enabled and self.config.ssl_cert_path and self.config.ssl_key_path:
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(self.config.ssl_cert_path, self.config.ssl_key_path)
            
            # Start server
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(
                runner, 
                self.config.host, 
                self.config.port,
                ssl_context=ssl_context
            )
            
            await site.start()
            
            self.logger.info(f"MCP Server started on {self.config.host}:{self.config.port}")
            self.logger.info(f"Server type: {self.config.server_type.value}")
            self.logger.info(f"SSL enabled: {self.config.ssl_enabled}")
            
            # Keep server running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise
    
    async def _handle_http_request(self, request: web.Request) -> web.Response:
        """Handle HTTP MCP requests"""
        try:
            # Parse request
            body = await request.json()
            mcp_request = MCPRequest(**body)
            
            # Process request
            response = await self._process_request(mcp_request)
            
            return web.json_response(asdict(response))
            
        except Exception as e:
            self.logger.error(f"HTTP request error: {e}")
            error_response = MCPResponse(
                request_id=body.get('request_id', 'unknown') if 'body' in locals() else 'unknown',
                success=False,
                error={
                    'code': 'HTTP_REQUEST_ERROR',
                    'message': str(e)
                }
            )
            return web.json_response(asdict(error_response), status=500)
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket MCP connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        connection_id = str(uuid.uuid4())
        self.active_connections.add(connection_id)
        self.metrics['active_sessions'] += 1
        
        self.logger.info(f"WebSocket connection established: {connection_id}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        # Parse message
                        data = json.loads(msg.data)
                        mcp_request = MCPRequest(**data)
                        
                        # Process request
                        response = await self._process_request(mcp_request)
                        
                        # Send response
                        await ws.send_text(json.dumps(asdict(response)))
                        
                    except Exception as e:
                        self.logger.error(f"WebSocket message error: {e}")
                        error_response = MCPResponse(
                            request_id=data.get('request_id', 'unknown') if 'data' in locals() else 'unknown',
                            success=False,
                            error={
                                'code': 'WEBSOCKET_MESSAGE_ERROR',
                                'message': str(e)
                            }
                        )
                        await ws.send_text(json.dumps(asdict(error_response)))
                        
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {ws.exception()}")
                    break
                    
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
        finally:
            self.active_connections.discard(connection_id)
            self.metrics['active_sessions'] -= 1
            self.logger.info(f"WebSocket connection closed: {connection_id}")
        
        return ws
    
    async def _process_request(self, request: MCPRequest) -> MCPResponse:
        """Process MCP request with middleware and authorization"""
        start_time = datetime.utcnow()
        
        try:
            # Update metrics
            self.metrics['requests_total'] += 1
            
            # Apply middleware
            for middleware in self.middleware_stack:
                request = await middleware(request)
            
            # Validate request
            await self._validate_request(request)
            
            # Check authorization
            await self._check_authorization(request)
            
            # Get handler
            handler = self.request_handlers.get(request.method)
            if not handler:
                raise ValidationError(f"Unknown method: {request.method}")
            
            # Execute handler
            result = await handler(request)
            
            # Create success response
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics['requests_success'] += 1
            self._update_average_response_time(execution_time)
            
            return MCPResponse(
                request_id=request.request_id,
                success=True,
                data=result,
                execution_time=execution_time
            )
            
        except MCPServerError as e:
            # Handle known errors
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics['requests_error'] += 1
            
            self.logger.warning(f"MCP Server Error: {e.message}")
            
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error={
                    'code': e.error_code,
                    'message': e.message,
                    'details': e.details
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            # Handle unexpected errors
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics['requests_error'] += 1
            
            self.logger.error(f"Unexpected error: {e}")
            
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error={
                    'code': 'INTERNAL_ERROR',
                    'message': 'An unexpected error occurred',
                    'details': {'error': str(e)}
                },
                execution_time=execution_time
            )
    
    async def _validate_request(self, request: MCPRequest):
        """Validate MCP request structure and content"""
        if not request.request_id:
            raise ValidationError("Missing request_id")
        
        if not request.method:
            raise ValidationError("Missing method")
        
        if not isinstance(request.params, dict):
            raise ValidationError("Params must be a dictionary")
        
        # Additional validation can be added here
    
    async def _check_authorization(self, request: MCPRequest):
        """Check request authorization"""
        # Skip authorization for public methods
        public_methods = {'health', 'info', 'ping'}
        if request.method in public_methods:
            return
        
        # Check if user_id is provided
        if not request.user_id:
            raise AuthorizationError("Missing user_id")
        
        # Validate JWT token if provided
        auth_header = request.params.get('authorization')
        if auth_header:
            try:
                token = auth_header.replace('Bearer ', '')
                payload = jwt.decode(token, self.config.jwt_secret, algorithms=['HS256'])
                
                # Verify user_id matches token
                if payload.get('user_id') != request.user_id:
                    raise AuthorizationError("Invalid token for user")
                
                # Check token expiration
                if payload.get('exp', 0) < datetime.utcnow().timestamp():
                    raise AuthorizationError("Token expired")
                
                # Set authorization level from token
                request.authorization_level = payload.get('authorization_level', 'read')
                
            except jwt.InvalidTokenError:
                raise AuthorizationError("Invalid JWT token")
        
        # Check organization access
        if request.organization_id:
            # This would typically check database for user-organization relationship
            # For now, we'll assume access is granted
            pass
        
        # Method-specific authorization checks
        await self._check_method_authorization(request)
    
    async def _check_method_authorization(self, request: MCPRequest):
        """Check method-specific authorization requirements"""
        # This can be overridden by subclasses for specific authorization logic
        pass
    
    def _update_average_response_time(self, execution_time: float):
        """Update average response time metric"""
        current_avg = self.metrics['average_response_time']
        total_requests = self.metrics['requests_success']
        
        if total_requests == 1:
            self.metrics['average_response_time'] = execution_time
        else:
            # Calculate running average
            self.metrics['average_response_time'] = (
                (current_avg * (total_requests - 1) + execution_time) / total_requests
            )
    
    # Default handlers
    async def _handle_health_check(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle health check requests"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'status': 'healthy',
            'server_id': self.server_id,
            'server_type': self.config.server_type.value,
            'uptime_seconds': uptime,
            'active_connections': len(self.active_connections),
            'version': '1.0.0'
        }
    
    async def _handle_server_info(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle server info requests"""
        return {
            'server_id': self.server_id,
            'server_type': self.config.server_type.value,
            'host': self.config.host,
            'port': self.config.port,
            'ssl_enabled': self.config.ssl_enabled,
            'max_connections': self.config.max_connections,
            'timeout': self.config.timeout,
            'start_time': self.start_time.isoformat(),
            'available_methods': list(self.request_handlers.keys())
        }
    
    async def _handle_metrics(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle metrics requests"""
        if not self.config.enable_metrics:
            raise ValidationError("Metrics are disabled")
        
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'server_metrics': self.metrics,
            'uptime_seconds': uptime,
            'active_connections': len(self.active_connections),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _handle_ping(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle ping requests"""
        return {
            'pong': True,
            'timestamp': datetime.utcnow().isoformat(),
            'server_id': self.server_id
        }
    
    async def _handle_http_health(self, request: web.Request) -> web.Response:
        """Handle HTTP health check"""
        health_data = await self._handle_health_check(MCPRequest(
            request_id=str(uuid.uuid4()),
            method='health',
            params={}
        ))
        return web.json_response(health_data)
    
    async def _handle_http_metrics(self, request: web.Request) -> web.Response:
        """Handle HTTP metrics request"""
        try:
            metrics_data = await self._handle_metrics(MCPRequest(
                request_id=str(uuid.uuid4()),
                method='metrics',
                params={}
            ))
            return web.json_response(metrics_data)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    async def initialize_server_specific_handlers(self):
        """Initialize server-specific request handlers"""
        pass
    
    @abstractmethod
    async def cleanup_resources(self):
        """Cleanup server-specific resources"""
        pass
    
    # Utility methods
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not self.cipher:
            return data
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self.cipher:
            return encrypted_data
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    async def cache_set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache"""
        try:
            serialized_value = json.dumps(value)
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.setex, key, ttl, serialized_value
            )
        except Exception as e:
            self.logger.warning(f"Cache set error: {e}")
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, key
            )
            if value:
                return json.loads(value.decode())
            return None
        except Exception as e:
            self.logger.warning(f"Cache get error: {e}")
            return None
    
    async def cache_delete(self, key: str):
        """Delete value from cache"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.delete, key
            )
        except Exception as e:
            self.logger.warning(f"Cache delete error: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the server"""
        self.logger.info("Shutting down MCP server...")
        
        # Close active connections
        for connection_id in list(self.active_connections):
            self.logger.info(f"Closing connection: {connection_id}")
        
        # Cleanup resources
        await self.cleanup_resources()
        
        # Close Redis connection
        self.redis_client.close()
        
        self.logger.info("MCP server shutdown complete")

