"""
Service Mesh for Microservices Communication.
Provides secure, observable, and reliable service-to-service communication.
"""

import asyncio
import json
import time
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
import warnings
warnings.filterwarnings('ignore')

try:
    import aiohttp
    import yarl
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class ServiceMeshProtocol(Enum):
    """Service mesh protocols."""
    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"
    TCP = "tcp"
    WEBSOCKET = "websocket"

class TrafficPolicy(Enum):
    """Traffic management policies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONN = "least_conn"
    RANDOM = "random"
    WEIGHTED = "weighted"
    CIRCUIT_BREAKER = "circuit_breaker"

class SecurityPolicy(Enum):
    """Security policies."""
    MTLS = "mtls"  # Mutual TLS
    JWT = "jwt"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    NONE = "none"

class ServiceStatus(Enum):
    """Service status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"

@dataclass
class ServiceEndpoint:
    """Service endpoint definition."""
    service_name: str
    host: str
    port: int
    protocol: ServiceMeshProtocol = ServiceMeshProtocol.HTTP
    version: str = "v1"
    namespace: str = "default"
    health_check_path: str = "/health"
    status: ServiceStatus = ServiceStatus.UNKNOWN
    metadata: Dict[str, str] = field(default_factory=dict)
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0

@dataclass
class ServiceRoute:
    """Service routing configuration."""
    name: str
    source_service: str
    destination_service: str
    path_pattern: str = "/*"
    method: str = "*"
    weight: int = 100
    timeout: int = 30
    retries: int = 3
    circuit_breaker_enabled: bool = True
    rate_limit: Optional[int] = None  # requests per second
    headers: Dict[str, str] = field(default_factory=dict)

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: int = 60  # seconds
    monitor_window: int = 300  # seconds
    enabled: bool = True

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_second: int = 100
    burst_size: int = 200
    window_size: int = 60  # seconds
    enabled: bool = True

@dataclass
class SecurityConfig:
    """Security configuration."""
    policy: SecurityPolicy = SecurityPolicy.NONE
    mtls_cert_path: Optional[str] = None
    mtls_key_path: Optional[str] = None
    jwt_secret: Optional[str] = None
    api_keys: Set[str] = field(default_factory=set)
    allowed_origins: Set[str] = field(default_factory=set)
    encryption_enabled: bool = False

@dataclass
class ServiceMetrics:
    """Service communication metrics."""
    service_name: str
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    active_connections: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ServiceCall:
    """Service call record for tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    source_service: str
    destination_service: str
    method: str
    path: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

class ServiceMesh:
    """Service mesh for microservices communication."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Service mesh configuration
        self.mesh_config = {
            'enabled': True,
            'proxy_port': 9090,
            'admin_port': 9091,
            'discovery_interval': 30,
            'health_check_interval': 15,
            'metrics_collection_interval': 60,
            'trace_sampling_rate': 0.1,  # 10% sampling
            'circuit_breaker_enabled': True,
            'rate_limiting_enabled': True,
            'mtls_enabled': False,
            'observability_enabled': True
        }
        
        # Service registry
        self.services: Dict[str, ServiceEndpoint] = {}
        self.service_routes: Dict[str, List[ServiceRoute]] = {}
        
        # Traffic management
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        
        # Security
        self.security_configs: Dict[str, SecurityConfig] = {}
        
        # Metrics and observability
        self.service_metrics: Dict[str, ServiceMetrics] = {}
        self.service_calls: List[ServiceCall] = []
        self.active_traces: Dict[str, List[ServiceCall]] = {}
        
        # HTTP session for service calls
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Service mesh state
        self.mesh_active = False
        self.discovery_task = None
        self.metrics_task = None
        
        # Setup default services and routes
        self._setup_default_services()
        
        self.logger.info("ServiceMesh initialized")
    
    def _setup_default_services(self):
        """Setup default services for trading system."""
        try:
            # Trading Engine Service
            self.services["trading-engine"] = ServiceEndpoint(
                service_name="trading-engine",
                host="trading-engine",
                port=8000,
                protocol=ServiceMeshProtocol.HTTP,
                version="v1",
                namespace="trading-system",
                health_check_path="/health",
                metadata={
                    "tier": "application",
                    "component": "core",
                    "critical": "true"
                }
            )
            
            # Market Data Service
            self.services["market-data"] = ServiceEndpoint(
                service_name="market-data",
                host="market-data",
                port=8002,
                protocol=ServiceMeshProtocol.HTTP,
                version="v1",
                namespace="trading-system",
                health_check_path="/health",
                metadata={
                    "tier": "data",
                    "component": "ingestion",
                    "critical": "true"
                }
            )
            
            # Risk Manager Service
            self.services["risk-manager"] = ServiceEndpoint(
                service_name="risk-manager",
                host="risk-manager",
                port=8003,
                protocol=ServiceMeshProtocol.HTTP,
                version="v1",
                namespace="trading-system",
                health_check_path="/health",
                metadata={
                    "tier": "application",
                    "component": "risk",
                    "critical": "true"
                }
            )
            
            # HFT Module Service
            self.services["hft-module"] = ServiceEndpoint(
                service_name="hft-module",
                host="hft-module",
                port=8006,
                protocol=ServiceMeshProtocol.HTTP,
                version="v1",
                namespace="trading-system",
                health_check_path="/health",
                metadata={
                    "tier": "application",
                    "component": "hft",
                    "critical": "true",
                    "latency": "ultra-low"
                }
            )
            
            # Analytics Service
            self.services["analytics"] = ServiceEndpoint(
                service_name="analytics",
                host="analytics",
                port=8004,
                protocol=ServiceMeshProtocol.HTTP,
                version="v1",
                namespace="trading-system",
                health_check_path="/health",
                metadata={
                    "tier": "analytics",
                    "component": "compute",
                    "critical": "false"
                }
            )
            
            # ML Engine Service
            self.services["ml-engine"] = ServiceEndpoint(
                service_name="ml-engine",
                host="ml-engine",
                port=8005,
                protocol=ServiceMeshProtocol.HTTP,
                version="v1",
                namespace="trading-system",
                health_check_path="/health",
                metadata={
                    "tier": "ml",
                    "component": "inference",
                    "critical": "false",
                    "gpu": "required"
                }
            )
            
            # Setup default routing rules
            self._setup_default_routes()
            
            # Setup default security configs
            self._setup_default_security()
            
        except Exception as e:
            self.logger.error(f"Failed to setup default services: {e}")
    
    def _setup_default_routes(self):
        """Setup default service routes."""
        try:
            # Trading Engine routes
            self.service_routes["trading-engine"] = [
                ServiceRoute(
                    name="market-data-route",
                    source_service="trading-engine",
                    destination_service="market-data",
                    path_pattern="/api/market/*",
                    timeout=5,  # Fast timeout for market data
                    retries=2,
                    circuit_breaker_enabled=True,
                    rate_limit=1000  # High rate limit
                ),
                ServiceRoute(
                    name="risk-check-route",
                    source_service="trading-engine",
                    destination_service="risk-manager",
                    path_pattern="/api/risk/*",
                    timeout=10,
                    retries=3,
                    circuit_breaker_enabled=True,
                    rate_limit=500
                ),
                ServiceRoute(
                    name="analytics-route",
                    source_service="trading-engine",
                    destination_service="analytics",
                    path_pattern="/api/analytics/*",
                    timeout=30,
                    retries=1,
                    circuit_breaker_enabled=False,  # Analytics not critical
                    rate_limit=100
                )
            ]
            
            # HFT Module routes (ultra-low latency)
            self.service_routes["hft-module"] = [
                ServiceRoute(
                    name="hft-market-data",
                    source_service="hft-module",
                    destination_service="market-data",
                    path_pattern="/api/market/realtime/*",
                    timeout=1,  # Ultra-low timeout
                    retries=0,  # No retries for HFT
                    circuit_breaker_enabled=False,
                    rate_limit=10000  # Very high rate limit
                ),
                ServiceRoute(
                    name="hft-trading-route",
                    source_service="hft-module",
                    destination_service="trading-engine",
                    path_pattern="/api/orders/*",
                    timeout=2,
                    retries=1,
                    circuit_breaker_enabled=True,
                    rate_limit=5000
                )
            ]
            
            # Market Data routes
            self.service_routes["market-data"] = [
                ServiceRoute(
                    name="analytics-feed",
                    source_service="market-data",
                    destination_service="analytics",
                    path_pattern="/api/data/stream/*",
                    timeout=15,
                    retries=2,
                    circuit_breaker_enabled=False,
                    rate_limit=200
                ),
                ServiceRoute(
                    name="ml-feed",
                    source_service="market-data",
                    destination_service="ml-engine",
                    path_pattern="/api/data/features/*",
                    timeout=20,
                    retries=1,
                    circuit_breaker_enabled=False,
                    rate_limit=50
                )
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to setup default routes: {e}")
    
    def _setup_default_security(self):
        """Setup default security configurations."""
        try:
            # High-security services
            high_security_services = ["trading-engine", "risk-manager", "hft-module"]
            for service in high_security_services:
                self.security_configs[service] = SecurityConfig(
                    policy=SecurityPolicy.JWT,
                    encryption_enabled=True,
                    allowed_origins={"trading-system", "admin-console"}
                )
            
            # Medium-security services
            medium_security_services = ["market-data"]
            for service in medium_security_services:
                self.security_configs[service] = SecurityConfig(
                    policy=SecurityPolicy.API_KEY,
                    api_keys={"market-api-key-1", "market-api-key-2"},
                    encryption_enabled=False
                )
            
            # Low-security services
            low_security_services = ["analytics", "ml-engine"]
            for service in low_security_services:
                self.security_configs[service] = SecurityConfig(
                    policy=SecurityPolicy.NONE,
                    encryption_enabled=False
                )
            
        except Exception as e:
            self.logger.error(f"Failed to setup default security: {e}")
    
    async def start_service_mesh(self):
        """Start service mesh operations."""
        try:
            if self.mesh_active:
                return
            
            self.mesh_active = True
            
            # Initialize HTTP session
            if HAS_AIOHTTP:
                timeout = aiohttp.ClientTimeout(total=30)
                self.http_session = aiohttp.ClientSession(timeout=timeout)
            
            # Start service discovery
            self.discovery_task = asyncio.create_task(self._service_discovery_loop())
            
            # Start metrics collection
            self.metrics_task = asyncio.create_task(self._metrics_collection_loop())
            
            # Initialize circuit breakers
            await self._initialize_circuit_breakers()
            
            # Initialize rate limiters
            await self._initialize_rate_limiters()
            
            self.logger.info("Service mesh started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start service mesh: {e}")
    
    async def stop_service_mesh(self):
        """Stop service mesh operations."""
        try:
            self.mesh_active = False
            
            # Cancel tasks
            if self.discovery_task:
                self.discovery_task.cancel()
                try:
                    await self.discovery_task
                except asyncio.CancelledError:
                    pass
            
            if self.metrics_task:
                self.metrics_task.cancel()
                try:
                    await self.metrics_task
                except asyncio.CancelledError:
                    pass
            
            # Close HTTP session
            if self.http_session:
                await self.http_session.close()
                self.http_session = None
            
            self.logger.info("Service mesh stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop service mesh: {e}")
    
    async def _service_discovery_loop(self):
        """Service discovery and health checking loop."""
        try:
            while self.mesh_active:
                # Perform health checks on all services
                await self._perform_health_checks()
                
                # Update service metrics
                await self._update_service_metrics()
                
                await asyncio.sleep(self.mesh_config['discovery_interval'])
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Service discovery loop error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all registered services."""
        try:
            tasks = []
            for service_name, endpoint in self.services.items():
                task = asyncio.create_task(self._check_service_health(endpoint))
                tasks.append((service_name, task))
            
            # Wait for all health checks
            for service_name, task in tasks:
                try:
                    is_healthy = await task
                    await self._update_service_status(service_name, is_healthy)
                except Exception as e:
                    self.logger.error(f"Health check failed for {service_name}: {e}")
                    await self._update_service_status(service_name, False)
                    
        except Exception as e:
            self.logger.error(f"Failed to perform health checks: {e}")
    
    async def _check_service_health(self, endpoint: ServiceEndpoint) -> bool:
        """Check health of a specific service."""
        try:
            if not self.http_session:
                return True  # Assume healthy if no HTTP session
            
            url = f"http://{endpoint.host}:{endpoint.port}{endpoint.health_check_path}"
            
            async with self.http_session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.debug(f"Health check failed for {endpoint.service_name}: {e}")
            return False
    
    async def _update_service_status(self, service_name: str, is_healthy: bool):
        """Update service status based on health check."""
        try:
            if service_name not in self.services:
                return
            
            endpoint = self.services[service_name]
            endpoint.last_health_check = datetime.now()
            
            if is_healthy:
                endpoint.consecutive_failures = 0
                if endpoint.status != ServiceStatus.HEALTHY:
                    endpoint.status = ServiceStatus.HEALTHY
                    self.logger.info(f"Service {service_name} is now healthy")
            else:
                endpoint.consecutive_failures += 1
                if endpoint.consecutive_failures >= 3:
                    if endpoint.status != ServiceStatus.UNHEALTHY:
                        endpoint.status = ServiceStatus.UNHEALTHY
                        self.logger.warning(f"Service {service_name} is now unhealthy")
                elif endpoint.consecutive_failures >= 1:
                    endpoint.status = ServiceStatus.DEGRADED
                    
        except Exception as e:
            self.logger.error(f"Failed to update service status for {service_name}: {e}")
    
    async def make_service_call(self, source_service: str, destination_service: str,
                              method: str, path: str, headers: Optional[Dict[str, str]] = None,
                              data: Optional[Any] = None, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """Make a service-to-service call through the mesh."""
        try:
            # Generate trace ID if not provided
            if not trace_id:
                trace_id = str(uuid.uuid4())
            
            span_id = str(uuid.uuid4())
            
            # Find route
            route = self._find_route(source_service, destination_service, path)
            if not route:
                return {'error': 'No route found', 'status_code': 404}
            
            # Check circuit breaker
            if route.circuit_breaker_enabled:
                if self._is_circuit_breaker_open(source_service, destination_service):
                    return {'error': 'Circuit breaker open', 'status_code': 503}
            
            # Check rate limit
            if route.rate_limit:
                if not self._check_rate_limit(source_service, destination_service, route.rate_limit):
                    return {'error': 'Rate limit exceeded', 'status_code': 429}
            
            # Get destination endpoint
            destination_endpoint = self.services.get(destination_service)
            if not destination_endpoint or destination_endpoint.status == ServiceStatus.UNHEALTHY:
                return {'error': 'Service unavailable', 'status_code': 503}
            
            # Create service call record
            service_call = ServiceCall(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=None,
                source_service=source_service,
                destination_service=destination_service,
                method=method,
                path=path,
                start_time=datetime.now(),
                headers=headers or {}
            )
            
            # Add tracing headers
            call_headers = (headers or {}).copy()
            call_headers.update({
                'X-Trace-Id': trace_id,
                'X-Span-Id': span_id,
                'X-Source-Service': source_service
            })
            
            # Apply security policy
            security_config = self.security_configs.get(destination_service)
            if security_config:
                call_headers.update(self._apply_security_headers(security_config))
            
            # Make the actual call
            response = await self._execute_service_call(
                destination_endpoint, method, path, call_headers, data, route.timeout
            )
            
            # Update service call record
            service_call.end_time = datetime.now()
            service_call.status_code = response.get('status_code', 0)
            service_call.response_time = (service_call.end_time - service_call.start_time).total_seconds()
            
            if response.get('status_code', 0) >= 400:
                service_call.error_message = response.get('error', 'Unknown error')
                self._record_circuit_breaker_failure(source_service, destination_service)
            else:
                self._record_circuit_breaker_success(source_service, destination_service)
            
            # Store service call for tracing
            self.service_calls.append(service_call)
            if len(self.service_calls) > 10000:
                self.service_calls = self.service_calls[-5000:]
            
            # Update metrics
            await self._update_call_metrics(service_call)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Service call failed: {source_service} -> {destination_service}: {e}")
            return {'error': str(e), 'status_code': 500}
    
    async def _execute_service_call(self, endpoint: ServiceEndpoint, method: str, path: str,
                                  headers: Dict[str, str], data: Any, timeout: int) -> Dict[str, Any]:
        """Execute the actual HTTP service call."""
        try:
            if not self.http_session:
                return {'error': 'HTTP session not available', 'status_code': 500}
            
            url = f"http://{endpoint.host}:{endpoint.port}{path}"
            
            # Determine HTTP method and make call
            call_timeout = aiohttp.ClientTimeout(total=timeout)
            
            if method.upper() == 'GET':
                async with self.http_session.get(url, headers=headers, timeout=call_timeout) as response:
                    response_data = await response.text()
                    return {
                        'status_code': response.status,
                        'data': response_data,
                        'headers': dict(response.headers)
                    }
            elif method.upper() == 'POST':
                async with self.http_session.post(url, headers=headers, json=data, timeout=call_timeout) as response:
                    response_data = await response.text()
                    return {
                        'status_code': response.status,
                        'data': response_data,
                        'headers': dict(response.headers)
                    }
            elif method.upper() == 'PUT':
                async with self.http_session.put(url, headers=headers, json=data, timeout=call_timeout) as response:
                    response_data = await response.text()
                    return {
                        'status_code': response.status,
                        'data': response_data,
                        'headers': dict(response.headers)
                    }
            elif method.upper() == 'DELETE':
                async with self.http_session.delete(url, headers=headers, timeout=call_timeout) as response:
                    response_data = await response.text()
                    return {
                        'status_code': response.status,
                        'data': response_data,
                        'headers': dict(response.headers)
                    }
            else:
                return {'error': f'Unsupported method: {method}', 'status_code': 405}
                
        except asyncio.TimeoutError:
            return {'error': 'Request timeout', 'status_code': 408}
        except Exception as e:
            return {'error': str(e), 'status_code': 500}
    
    def _find_route(self, source_service: str, destination_service: str, path: str) -> Optional[ServiceRoute]:
        """Find matching route for service call."""
        try:
            routes = self.service_routes.get(source_service, [])
            
            for route in routes:
                if route.destination_service == destination_service:
                    # Simple pattern matching (would use regex in production)
                    if route.path_pattern == "/*" or path.startswith(route.path_pattern.replace("*", "")):
                        return route
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find route: {e}")
            return None
    
    async def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for all service routes."""
        try:
            for source_service, routes in self.service_routes.items():
                for route in routes:
                    if route.circuit_breaker_enabled:
                        cb_key = f"{source_service}:{route.destination_service}"
                        self.circuit_breakers[cb_key] = {
                            'state': 'closed',  # closed, open, half-open
                            'failure_count': 0,
                            'success_count': 0,
                            'last_failure_time': None,
                            'config': CircuitBreakerConfig()
                        }
                        
        except Exception as e:
            self.logger.error(f"Failed to initialize circuit breakers: {e}")
    
    async def _initialize_rate_limiters(self):
        """Initialize rate limiters for all service routes."""
        try:
            for source_service, routes in self.service_routes.items():
                for route in routes:
                    if route.rate_limit:
                        rl_key = f"{source_service}:{route.destination_service}"
                        self.rate_limiters[rl_key] = {
                            'requests': [],
                            'config': RateLimitConfig(requests_per_second=route.rate_limit)
                        }
                        
        except Exception as e:
            self.logger.error(f"Failed to initialize rate limiters: {e}")
    
    def _is_circuit_breaker_open(self, source_service: str, destination_service: str) -> bool:
        """Check if circuit breaker is open."""
        try:
            cb_key = f"{source_service}:{destination_service}"
            if cb_key not in self.circuit_breakers:
                return False
            
            cb = self.circuit_breakers[cb_key]
            
            if cb['state'] == 'open':
                # Check if timeout has passed
                if cb['last_failure_time']:
                    time_since_failure = (datetime.now() - cb['last_failure_time']).total_seconds()
                    if time_since_failure > cb['config'].timeout:
                        cb['state'] = 'half-open'
                        cb['success_count'] = 0
                        return False
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check circuit breaker: {e}")
            return False
    
    def _record_circuit_breaker_failure(self, source_service: str, destination_service: str):
        """Record circuit breaker failure."""
        try:
            cb_key = f"{source_service}:{destination_service}"
            if cb_key not in self.circuit_breakers:
                return
            
            cb = self.circuit_breakers[cb_key]
            cb['failure_count'] += 1
            cb['last_failure_time'] = datetime.now()
            
            if cb['failure_count'] >= cb['config'].failure_threshold:
                cb['state'] = 'open'
                self.logger.warning(f"Circuit breaker opened: {cb_key}")
                
        except Exception as e:
            self.logger.error(f"Failed to record circuit breaker failure: {e}")
    
    def _record_circuit_breaker_success(self, source_service: str, destination_service: str):
        """Record circuit breaker success."""
        try:
            cb_key = f"{source_service}:{destination_service}"
            if cb_key not in self.circuit_breakers:
                return
            
            cb = self.circuit_breakers[cb_key]
            
            if cb['state'] == 'half-open':
                cb['success_count'] += 1
                if cb['success_count'] >= cb['config'].success_threshold:
                    cb['state'] = 'closed'
                    cb['failure_count'] = 0
                    self.logger.info(f"Circuit breaker closed: {cb_key}")
            elif cb['state'] == 'closed':
                # Reset failure count on success
                cb['failure_count'] = max(0, cb['failure_count'] - 1)
                
        except Exception as e:
            self.logger.error(f"Failed to record circuit breaker success: {e}")
    
    def _check_rate_limit(self, source_service: str, destination_service: str, rate_limit: int) -> bool:
        """Check rate limit for service call."""
        try:
            rl_key = f"{source_service}:{destination_service}"
            if rl_key not in self.rate_limiters:
                return True
            
            rl = self.rate_limiters[rl_key]
            current_time = datetime.now()
            
            # Remove old requests outside the window
            window_start = current_time - timedelta(seconds=rl['config'].window_size)
            rl['requests'] = [req_time for req_time in rl['requests'] if req_time >= window_start]
            
            # Check if under rate limit
            if len(rl['requests']) < rate_limit:
                rl['requests'].append(current_time)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check rate limit: {e}")
            return True  # Allow on error
    
    def _apply_security_headers(self, security_config: SecurityConfig) -> Dict[str, str]:
        """Apply security headers based on configuration."""
        try:
            headers = {}
            
            if security_config.policy == SecurityPolicy.JWT:
                # Would add JWT token
                headers['Authorization'] = 'Bearer jwt-token-here'
            elif security_config.policy == SecurityPolicy.API_KEY:
                # Would add API key
                headers['X-API-Key'] = 'api-key-here'
            
            if security_config.encryption_enabled:
                headers['X-Encryption-Required'] = 'true'
            
            return headers
            
        except Exception as e:
            self.logger.error(f"Failed to apply security headers: {e}")
            return {}
    
    async def _metrics_collection_loop(self):
        """Collect service mesh metrics."""
        try:
            while self.mesh_active:
                await self._collect_mesh_metrics()
                await asyncio.sleep(self.mesh_config['metrics_collection_interval'])
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Metrics collection loop error: {e}")
    
    async def _collect_mesh_metrics(self):
        """Collect and update service mesh metrics."""
        try:
            # Update service metrics based on recent calls
            current_time = datetime.now()
            one_minute_ago = current_time - timedelta(minutes=1)
            
            for service_name in self.services:
                if service_name not in self.service_metrics:
                    self.service_metrics[service_name] = ServiceMetrics(service_name=service_name)
                
                metrics = self.service_metrics[service_name]
                
                # Get recent calls for this service
                recent_calls = [
                    call for call in self.service_calls
                    if call.destination_service == service_name and call.start_time >= one_minute_ago
                ]
                
                if recent_calls:
                    metrics.request_count = len(recent_calls)
                    metrics.success_count = len([c for c in recent_calls if c.status_code and c.status_code < 400])
                    metrics.error_count = metrics.request_count - metrics.success_count
                    
                    response_times = [c.response_time for c in recent_calls if c.response_time]
                    if response_times:
                        metrics.avg_response_time = statistics.mean(response_times)
                        metrics.p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
                        metrics.p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
                
                metrics.last_updated = current_time
                
        except Exception as e:
            self.logger.error(f"Failed to collect mesh metrics: {e}")
    
    async def _update_service_metrics(self):
        """Update service-level metrics."""
        try:
            for service_name, endpoint in self.services.items():
                if service_name not in self.service_metrics:
                    self.service_metrics[service_name] = ServiceMetrics(service_name=service_name)
                
                # Would integrate with actual service metrics here
                # For now, just update the timestamp
                self.service_metrics[service_name].last_updated = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Failed to update service metrics: {e}")
    
    async def _update_call_metrics(self, service_call: ServiceCall):
        """Update metrics for a specific service call."""
        try:
            service_name = service_call.destination_service
            
            if service_name not in self.service_metrics:
                self.service_metrics[service_name] = ServiceMetrics(service_name=service_name)
            
            metrics = self.service_metrics[service_name]
            
            # Update counters (would be more sophisticated in production)
            if service_call.status_code and service_call.status_code < 400:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
            
            metrics.request_count += 1
            
        except Exception as e:
            self.logger.error(f"Failed to update call metrics: {e}")
    
    def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific service."""
        try:
            if service_name not in self.services:
                return None
            
            endpoint = self.services[service_name]
            metrics = self.service_metrics.get(service_name, ServiceMetrics(service_name=service_name))
            
            # Get circuit breaker status
            cb_status = {}
            for cb_key, cb_data in self.circuit_breakers.items():
                if cb_key.endswith(f":{service_name}"):
                    source = cb_key.split(':')[0]
                    cb_status[source] = cb_data['state']
            
            return {
                'service_name': service_name,
                'status': endpoint.status.value,
                'host': endpoint.host,
                'port': endpoint.port,
                'protocol': endpoint.protocol.value,
                'version': endpoint.version,
                'namespace': endpoint.namespace,
                'last_health_check': endpoint.last_health_check.isoformat() if endpoint.last_health_check else None,
                'consecutive_failures': endpoint.consecutive_failures,
                'metadata': endpoint.metadata,
                'metrics': {
                    'request_count': metrics.request_count,
                    'success_count': metrics.success_count,
                    'error_count': metrics.error_count,
                    'avg_response_time': round(metrics.avg_response_time, 3),
                    'p95_response_time': round(metrics.p95_response_time, 3),
                    'p99_response_time': round(metrics.p99_response_time, 3)
                },
                'circuit_breakers': cb_status
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get service status for {service_name}: {e}")
            return None
    
    def get_service_mesh_summary(self) -> Dict[str, Any]:
        """Get service mesh summary."""
        try:
            healthy_services = sum(
                1 for endpoint in self.services.values()
                if endpoint.status == ServiceStatus.HEALTHY
            )
            
            total_routes = sum(len(routes) for routes in self.service_routes.values())
            
            active_circuit_breakers = len([
                cb for cb in self.circuit_breakers.values()
                if cb['state'] != 'closed'
            ])
            
            recent_calls = len([
                call for call in self.service_calls
                if call.start_time > datetime.now() - timedelta(hours=1)
            ])
            
            avg_response_times = [
                metrics.avg_response_time for metrics in self.service_metrics.values()
                if metrics.avg_response_time > 0
            ]
            overall_avg_rt = statistics.mean(avg_response_times) if avg_response_times else 0
            
            return {
                'enabled': self.mesh_config['enabled'],
                'active': self.mesh_active,
                'total_services': len(self.services),
                'healthy_services': healthy_services,
                'service_health_percentage': f"{(healthy_services / len(self.services) * 100):.1f}%" if self.services else "0%",
                'total_routes': total_routes,
                'circuit_breakers_active': active_circuit_breakers,
                'recent_service_calls': recent_calls,
                'average_response_time': f"{overall_avg_rt:.3f}s",
                'security_policies_enabled': len(self.security_configs),
                'observability_enabled': self.mesh_config['observability_enabled'],
                'trace_sampling_rate': f"{self.mesh_config['trace_sampling_rate'] * 100}%"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate service mesh summary: {e}")
            return {'error': 'Unable to generate summary'}