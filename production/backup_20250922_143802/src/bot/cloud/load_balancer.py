"""
Load Balancer for Traffic Distribution.
Distributes incoming traffic across multiple instances with health checking and failover.
"""

import asyncio
import json
import time
import hashlib
import random
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
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

class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"

class HealthCheckType(Enum):
    """Health check types."""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    CUSTOM = "custom"

class BackendStatus(Enum):
    """Backend server status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"

@dataclass
class Backend:
    """Backend server configuration."""
    id: str
    host: str
    port: int
    weight: int = 100
    max_connections: int = 1000
    current_connections: int = 0
    status: BackendStatus = BackendStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    response_times: List[float] = field(default_factory=list)
    total_requests: int = 0
    failed_requests: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheck:
    """Health check configuration."""
    type: HealthCheckType
    interval: int = 30  # seconds
    timeout: int = 5
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    path: str = "/health"
    expected_status: int = 200
    expected_body: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    custom_checker: Optional[Callable] = None

@dataclass
class LoadBalancerPool:
    """Load balancer pool configuration."""
    name: str
    algorithm: LoadBalancingAlgorithm
    backends: List[Backend] = field(default_factory=list)
    health_check: Optional[HealthCheck] = None
    sticky_sessions: bool = False
    session_affinity_key: str = "client_ip"
    max_retries: int = 3
    retry_timeout: int = 5
    failover_enabled: bool = True
    drain_timeout: int = 300  # seconds
    
@dataclass
class RequestMetrics:
    """Request metrics."""
    pool_name: str
    backend_id: str
    timestamp: datetime
    response_time: float
    status_code: int
    success: bool
    client_ip: str
    user_agent: str = ""
    request_size: int = 0
    response_size: int = 0

@dataclass
class LoadBalancerStats:
    """Load balancer statistics."""
    pool_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    requests_per_second: float = 0.0
    active_connections: int = 0
    healthy_backends: int = 0
    total_backends: int = 0
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    last_updated: datetime = field(default_factory=datetime.now)

class LoadBalancer:
    """High-performance load balancer for trading system."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Load balancer configuration
        self.lb_config = {
            'enabled': True,
            'listen_port': 8080,
            'admin_port': 8081,
            'health_check_workers': 5,
            'max_concurrent_requests': 10000,
            'request_timeout': 30,
            'keep_alive_timeout': 75,
            'client_timeout': 60,
            'upstream_timeout': 30,
            'buffer_size': 64 * 1024,  # 64KB
            'enable_compression': True,
            'enable_ssl': False,
            'ssl_cert_path': None,
            'ssl_key_path': None
        }
        
        # Load balancer pools
        self.pools: Dict[str, LoadBalancerPool] = {}
        self.session_store: Dict[str, str] = {}  # session_id -> backend_id
        
        # Metrics and monitoring
        self.request_metrics: List[RequestMetrics] = []
        self.pool_stats: Dict[str, LoadBalancerStats] = {}
        
        # Health checking
        self.health_check_active = False
        self.health_check_tasks: List[asyncio.Task] = []
        
        # Round-robin counters
        self.rr_counters: Dict[str, int] = {}
        
        # HTTP session for health checks
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Setup default pools
        self._setup_default_pools()
        
        self.logger.info("LoadBalancer initialized")
    
    def _setup_default_pools(self):
        """Setup default load balancer pools for trading services."""
        try:
            # Trading Engine Pool
            trading_backends = [
                Backend(id="trading-1", host="127.0.0.1", port=8000, weight=100),
                Backend(id="trading-2", host="127.0.0.1", port=8010, weight=100),
                Backend(id="trading-3", host="127.0.0.1", port=8020, weight=100)
            ]
            
            trading_health_check = HealthCheck(
                type=HealthCheckType.HTTP,
                interval=15,
                timeout=5,
                healthy_threshold=2,
                unhealthy_threshold=3,
                path="/health",
                expected_status=200
            )
            
            self.pools["trading-engine"] = LoadBalancerPool(
                name="trading-engine",
                algorithm=LoadBalancingAlgorithm.LEAST_RESPONSE_TIME,
                backends=trading_backends,
                health_check=trading_health_check,
                sticky_sessions=False,
                max_retries=2,
                failover_enabled=True
            )
            
            # Market Data Pool
            market_data_backends = [
                Backend(id="market-1", host="127.0.0.1", port=8002, weight=150),
                Backend(id="market-2", host="127.0.0.1", port=8012, weight=100)
            ]
            
            market_data_health_check = HealthCheck(
                type=HealthCheckType.HTTP,
                interval=10,
                timeout=3,
                path="/health",
                expected_status=200
            )
            
            self.pools["market-data"] = LoadBalancerPool(
                name="market-data",
                algorithm=LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN,
                backends=market_data_backends,
                health_check=market_data_health_check,
                max_retries=3,
                failover_enabled=True
            )
            
            # HFT Module Pool (ultra-low latency)
            hft_backends = [
                Backend(id="hft-1", host="127.0.0.1", port=8006, weight=100),
                Backend(id="hft-2", host="127.0.0.1", port=8016, weight=100),
                Backend(id="hft-3", host="127.0.0.1", port=8026, weight=100),
                Backend(id="hft-4", host="127.0.0.1", port=8036, weight=100),
                Backend(id="hft-5", host="127.0.0.1", port=8046, weight=100)
            ]
            
            hft_health_check = HealthCheck(
                type=HealthCheckType.HTTP,
                interval=5,  # Very frequent for HFT
                timeout=1,   # Ultra-low timeout
                path="/health",
                expected_status=200
            )
            
            self.pools["hft-module"] = LoadBalancerPool(
                name="hft-module",
                algorithm=LoadBalancingAlgorithm.LEAST_RESPONSE_TIME,
                backends=hft_backends,
                health_check=hft_health_check,
                sticky_sessions=True,  # Session affinity for HFT
                session_affinity_key="client_ip",
                max_retries=1,  # Minimal retries for latency
                retry_timeout=1
            )
            
            # Analytics Pool
            analytics_backends = [
                Backend(id="analytics-1", host="127.0.0.1", port=8004, weight=100),
                Backend(id="analytics-2", host="127.0.0.1", port=8014, weight=100)
            ]
            
            analytics_health_check = HealthCheck(
                type=HealthCheckType.HTTP,
                interval=30,
                timeout=10,
                path="/health",
                expected_status=200
            )
            
            self.pools["analytics"] = LoadBalancerPool(
                name="analytics",
                algorithm=LoadBalancingAlgorithm.LEAST_CONNECTIONS,
                backends=analytics_backends,
                health_check=analytics_health_check,
                max_retries=2,
                failover_enabled=True
            )
            
            # ML Engine Pool
            ml_backends = [
                Backend(id="ml-1", host="127.0.0.1", port=8005, weight=100)
            ]
            
            ml_health_check = HealthCheck(
                type=HealthCheckType.HTTP,
                interval=60,  # Less frequent for ML
                timeout=15,   # Longer timeout
                path="/health",
                expected_status=200
            )
            
            self.pools["ml-engine"] = LoadBalancerPool(
                name="ml-engine",
                algorithm=LoadBalancingAlgorithm.ROUND_ROBIN,
                backends=ml_backends,
                health_check=ml_health_check,
                max_retries=1,
                failover_enabled=False  # ML has long startup time
            )
            
            # Initialize stats for all pools
            for pool_name in self.pools:
                self.pool_stats[pool_name] = LoadBalancerStats(pool_name=pool_name)
                self.rr_counters[pool_name] = 0
            
        except Exception as e:
            self.logger.error(f"Failed to setup default pools: {e}")
    
    async def start_load_balancer(self):
        """Start load balancer services."""
        try:
            # Initialize HTTP session for health checks
            if HAS_AIOHTTP:
                timeout = aiohttp.ClientTimeout(total=30)
                self.http_session = aiohttp.ClientSession(timeout=timeout)
            
            # Start health checking
            await self.start_health_checks()
            
            # Start metrics collection
            asyncio.create_task(self._metrics_collection_loop())
            
            self.logger.info("Load balancer started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start load balancer: {e}")
    
    async def stop_load_balancer(self):
        """Stop load balancer services."""
        try:
            # Stop health checks
            await self.stop_health_checks()
            
            # Close HTTP session
            if self.http_session:
                await self.http_session.close()
                self.http_session = None
            
            self.logger.info("Load balancer stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop load balancer: {e}")
    
    async def start_health_checks(self):
        """Start health check tasks for all pools."""
        try:
            if self.health_check_active:
                return
            
            self.health_check_active = True
            
            # Start health check tasks for each pool
            for pool_name, pool in self.pools.items():
                if pool.health_check:
                    task = asyncio.create_task(
                        self._health_check_loop(pool_name, pool)
                    )
                    self.health_check_tasks.append(task)
            
            self.logger.info(f"Started health checks for {len(self.health_check_tasks)} pools")
            
        except Exception as e:
            self.logger.error(f"Failed to start health checks: {e}")
    
    async def stop_health_checks(self):
        """Stop all health check tasks."""
        try:
            self.health_check_active = False
            
            # Cancel all health check tasks
            for task in self.health_check_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self.health_check_tasks.clear()
            self.logger.info("Stopped all health checks")
            
        except Exception as e:
            self.logger.error(f"Failed to stop health checks: {e}")
    
    async def _health_check_loop(self, pool_name: str, pool: LoadBalancerPool):
        """Health check loop for a specific pool."""
        try:
            while self.health_check_active:
                await self._perform_health_checks(pool)
                await asyncio.sleep(pool.health_check.interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Health check loop error for {pool_name}: {e}")
    
    async def _perform_health_checks(self, pool: LoadBalancerPool):
        """Perform health checks for all backends in a pool."""
        try:
            if not pool.health_check:
                return
            
            # Create health check tasks for all backends
            tasks = []
            for backend in pool.backends:
                task = asyncio.create_task(
                    self._check_backend_health(backend, pool.health_check)
                )
                tasks.append((backend, task))
            
            # Wait for all health checks to complete
            for backend, task in tasks:
                try:
                    is_healthy = await task
                    await self._update_backend_status(backend, is_healthy, pool.health_check)
                except Exception as e:
                    self.logger.error(f"Health check failed for {backend.id}: {e}")
                    await self._update_backend_status(backend, False, pool.health_check)
                    
        except Exception as e:
            self.logger.error(f"Failed to perform health checks for {pool.name}: {e}")
    
    async def _check_backend_health(self, backend: Backend, health_check: HealthCheck) -> bool:
        """Check health of a specific backend."""
        try:
            start_time = time.time()
            
            if health_check.type == HealthCheckType.HTTP:
                url = f"http://{backend.host}:{backend.port}{health_check.path}"
                
                if self.http_session:
                    async with self.http_session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=health_check.timeout),
                        headers=health_check.custom_headers
                    ) as response:
                        response_time = time.time() - start_time
                        
                        # Update response time history
                        backend.response_times.append(response_time)
                        if len(backend.response_times) > 100:
                            backend.response_times = backend.response_times[-50:]
                        
                        # Check status code
                        if response.status != health_check.expected_status:
                            return False
                        
                        # Check expected body if specified
                        if health_check.expected_body:
                            body = await response.text()
                            if health_check.expected_body not in body:
                                return False
                        
                        return True
                else:
                    # Fallback without aiohttp
                    return True
                    
            elif health_check.type == HealthCheckType.TCP:
                # TCP connection test
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(backend.host, backend.port),
                        timeout=health_check.timeout
                    )
                    writer.close()
                    await writer.wait_closed()
                    return True
                except:
                    return False
                    
            elif health_check.type == HealthCheckType.CUSTOM:
                # Custom health check
                if health_check.custom_checker:
                    return await health_check.custom_checker(backend)
                return True
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Health check failed for {backend.id}: {e}")
            return False
    
    async def _update_backend_status(self, backend: Backend, is_healthy: bool, health_check: HealthCheck):
        """Update backend status based on health check result."""
        try:
            backend.last_health_check = datetime.now()
            
            if is_healthy:
                backend.consecutive_successes += 1
                backend.consecutive_failures = 0
                
                # Mark as healthy if enough consecutive successes
                if (backend.status != BackendStatus.HEALTHY and 
                    backend.consecutive_successes >= health_check.healthy_threshold):
                    backend.status = BackendStatus.HEALTHY
                    self.logger.info(f"Backend {backend.id} is now healthy")
                    
            else:
                backend.consecutive_failures += 1
                backend.consecutive_successes = 0
                
                # Mark as unhealthy if enough consecutive failures
                if (backend.status == BackendStatus.HEALTHY and 
                    backend.consecutive_failures >= health_check.unhealthy_threshold):
                    backend.status = BackendStatus.UNHEALTHY
                    self.logger.warning(f"Backend {backend.id} is now unhealthy")
                    
        except Exception as e:
            self.logger.error(f"Failed to update backend status for {backend.id}: {e}")
    
    async def select_backend(self, pool_name: str, client_ip: str = "127.0.0.1", 
                           session_id: Optional[str] = None) -> Optional[Backend]:
        """Select best backend for request based on load balancing algorithm."""
        try:
            if pool_name not in self.pools:
                return None
            
            pool = self.pools[pool_name]
            healthy_backends = [b for b in pool.backends if b.status == BackendStatus.HEALTHY]
            
            if not healthy_backends:
                # No healthy backends, try unhealthy ones as last resort
                self.logger.warning(f"No healthy backends in pool {pool_name}")
                healthy_backends = [b for b in pool.backends if b.status == BackendStatus.UNHEALTHY]
                if not healthy_backends:
                    return None
            
            # Handle sticky sessions
            if pool.sticky_sessions and session_id:
                if session_id in self.session_store:
                    backend_id = self.session_store[session_id]
                    backend = next((b for b in healthy_backends if b.id == backend_id), None)
                    if backend:
                        return backend
            
            # Select backend based on algorithm
            selected_backend = None
            
            if pool.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
                selected_backend = self._select_round_robin(pool_name, healthy_backends)
            
            elif pool.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
                selected_backend = self._select_weighted_round_robin(pool_name, healthy_backends)
            
            elif pool.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
                selected_backend = self._select_least_connections(healthy_backends)
            
            elif pool.algorithm == LoadBalancingAlgorithm.WEIGHTED_LEAST_CONNECTIONS:
                selected_backend = self._select_weighted_least_connections(healthy_backends)
            
            elif pool.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
                selected_backend = self._select_least_response_time(healthy_backends)
            
            elif pool.algorithm == LoadBalancingAlgorithm.IP_HASH:
                selected_backend = self._select_ip_hash(client_ip, healthy_backends)
            
            elif pool.algorithm == LoadBalancingAlgorithm.RESOURCE_BASED:
                selected_backend = self._select_resource_based(healthy_backends)
            
            elif pool.algorithm == LoadBalancingAlgorithm.RANDOM:
                selected_backend = random.choice(healthy_backends)
            
            elif pool.algorithm == LoadBalancingAlgorithm.WEIGHTED_RANDOM:
                selected_backend = self._select_weighted_random(healthy_backends)
            
            # Store session affinity
            if pool.sticky_sessions and session_id and selected_backend:
                self.session_store[session_id] = selected_backend.id
            
            return selected_backend
            
        except Exception as e:
            self.logger.error(f"Failed to select backend from pool {pool_name}: {e}")
            return None
    
    def _select_round_robin(self, pool_name: str, backends: List[Backend]) -> Backend:
        """Round-robin selection."""
        if pool_name not in self.rr_counters:
            self.rr_counters[pool_name] = 0
        
        backend = backends[self.rr_counters[pool_name] % len(backends)]
        self.rr_counters[pool_name] += 1
        return backend
    
    def _select_weighted_round_robin(self, pool_name: str, backends: List[Backend]) -> Backend:
        """Weighted round-robin selection."""
        # Create weighted list
        weighted_backends = []
        for backend in backends:
            weighted_backends.extend([backend] * (backend.weight // 10))
        
        if not weighted_backends:
            return backends[0]
        
        if pool_name not in self.rr_counters:
            self.rr_counters[pool_name] = 0
        
        backend = weighted_backends[self.rr_counters[pool_name] % len(weighted_backends)]
        self.rr_counters[pool_name] += 1
        return backend
    
    def _select_least_connections(self, backends: List[Backend]) -> Backend:
        """Least connections selection."""
        return min(backends, key=lambda b: b.current_connections)
    
    def _select_weighted_least_connections(self, backends: List[Backend]) -> Backend:
        """Weighted least connections selection."""
        def score(backend):
            if backend.weight == 0:
                return float('inf')
            return backend.current_connections / backend.weight
        
        return min(backends, key=score)
    
    def _select_least_response_time(self, backends: List[Backend]) -> Backend:
        """Least response time selection."""
        def avg_response_time(backend):
            if not backend.response_times:
                return 0.0
            return statistics.mean(backend.response_times[-10:])  # Last 10 requests
        
        return min(backends, key=avg_response_time)
    
    def _select_ip_hash(self, client_ip: str, backends: List[Backend]) -> Backend:
        """IP hash selection for consistent routing."""
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return backends[hash_value % len(backends)]
    
    def _select_resource_based(self, backends: List[Backend]) -> Backend:
        """Resource-based selection (combines connections and response time)."""
        def resource_score(backend):
            conn_score = backend.current_connections / backend.max_connections
            rt_score = statistics.mean(backend.response_times[-5:]) / 1000 if backend.response_times else 0
            return conn_score + rt_score
        
        return min(backends, key=resource_score)
    
    def _select_weighted_random(self, backends: List[Backend]) -> Backend:
        """Weighted random selection."""
        total_weight = sum(b.weight for b in backends)
        if total_weight == 0:
            return random.choice(backends)
        
        r = random.randint(1, total_weight)
        for backend in backends:
            r -= backend.weight
            if r <= 0:
                return backend
        
        return backends[-1]
    
    async def record_request(self, pool_name: str, backend_id: str, response_time: float,
                           status_code: int, client_ip: str, success: bool = True):
        """Record request metrics."""
        try:
            # Update backend stats
            pool = self.pools.get(pool_name)
            if pool:
                backend = next((b for b in pool.backends if b.id == backend_id), None)
                if backend:
                    backend.total_requests += 1
                    if not success:
                        backend.failed_requests += 1
                    
                    # Update response time
                    backend.response_times.append(response_time)
                    if len(backend.response_times) > 100:
                        backend.response_times = backend.response_times[-50:]
            
            # Store request metrics
            request_metric = RequestMetrics(
                pool_name=pool_name,
                backend_id=backend_id,
                timestamp=datetime.now(),
                response_time=response_time,
                status_code=status_code,
                success=success,
                client_ip=client_ip
            )
            
            self.request_metrics.append(request_metric)
            
            # Keep only recent metrics
            if len(self.request_metrics) > 10000:
                self.request_metrics = self.request_metrics[-5000:]
                
        except Exception as e:
            self.logger.error(f"Failed to record request metrics: {e}")
    
    async def _metrics_collection_loop(self):
        """Collect and update load balancer metrics."""
        try:
            while True:
                await self._update_pool_stats()
                await asyncio.sleep(60)  # Update every minute
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Metrics collection loop error: {e}")
    
    async def _update_pool_stats(self):
        """Update statistics for all pools."""
        try:
            current_time = datetime.now()
            one_minute_ago = current_time - timedelta(minutes=1)
            
            for pool_name, pool in self.pools.items():
                stats = self.pool_stats[pool_name]
                
                # Get recent metrics
                recent_metrics = [
                    m for m in self.request_metrics
                    if m.pool_name == pool_name and m.timestamp >= one_minute_ago
                ]
                
                # Update basic stats
                stats.total_requests = sum(b.total_requests for b in pool.backends)
                stats.failed_requests = sum(b.failed_requests for b in pool.backends)
                stats.successful_requests = stats.total_requests - stats.failed_requests
                
                # Calculate response time stats
                if recent_metrics:
                    response_times = [m.response_time for m in recent_metrics]
                    stats.avg_response_time = statistics.mean(response_times)
                    stats.min_response_time = min(response_times)
                    stats.max_response_time = max(response_times)
                    stats.requests_per_second = len(recent_metrics) / 60.0
                
                # Count healthy backends
                stats.healthy_backends = sum(
                    1 for b in pool.backends if b.status == BackendStatus.HEALTHY
                )
                stats.total_backends = len(pool.backends)
                
                # Active connections
                stats.active_connections = sum(b.current_connections for b in pool.backends)
                
                stats.last_updated = current_time
                
        except Exception as e:
            self.logger.error(f"Failed to update pool stats: {e}")
    
    def add_backend(self, pool_name: str, backend: Backend):
        """Add backend to a pool."""
        try:
            if pool_name in self.pools:
                self.pools[pool_name].backends.append(backend)
                self.logger.info(f"Added backend {backend.id} to pool {pool_name}")
            else:
                self.logger.error(f"Pool {pool_name} does not exist")
                
        except Exception as e:
            self.logger.error(f"Failed to add backend to pool {pool_name}: {e}")
    
    def remove_backend(self, pool_name: str, backend_id: str):
        """Remove backend from a pool."""
        try:
            if pool_name not in self.pools:
                self.logger.error(f"Pool {pool_name} does not exist")
                return
            
            pool = self.pools[pool_name]
            pool.backends = [b for b in pool.backends if b.id != backend_id]
            
            self.logger.info(f"Removed backend {backend_id} from pool {pool_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to remove backend from pool {pool_name}: {e}")
    
    def drain_backend(self, pool_name: str, backend_id: str):
        """Drain backend (mark for removal but allow existing connections)."""
        try:
            if pool_name not in self.pools:
                return
            
            pool = self.pools[pool_name]
            backend = next((b for b in pool.backends if b.id == backend_id), None)
            
            if backend:
                backend.status = BackendStatus.DRAINING
                self.logger.info(f"Draining backend {backend_id} in pool {pool_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to drain backend {backend_id}: {e}")
    
    def get_pool_status(self, pool_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific pool."""
        try:
            if pool_name not in self.pools:
                return None
            
            pool = self.pools[pool_name]
            stats = self.pool_stats.get(pool_name, LoadBalancerStats(pool_name=pool_name))
            
            backend_status = []
            for backend in pool.backends:
                backend_info = {
                    'id': backend.id,
                    'host': backend.host,
                    'port': backend.port,
                    'status': backend.status.value,
                    'weight': backend.weight,
                    'current_connections': backend.current_connections,
                    'max_connections': backend.max_connections,
                    'total_requests': backend.total_requests,
                    'failed_requests': backend.failed_requests,
                    'consecutive_failures': backend.consecutive_failures,
                    'last_health_check': backend.last_health_check.isoformat() if backend.last_health_check else None,
                    'avg_response_time': statistics.mean(backend.response_times[-10:]) if backend.response_times else 0
                }
                backend_status.append(backend_info)
            
            return {
                'name': pool.name,
                'algorithm': pool.algorithm.value,
                'total_backends': len(pool.backends),
                'healthy_backends': sum(1 for b in pool.backends if b.status == BackendStatus.HEALTHY),
                'sticky_sessions': pool.sticky_sessions,
                'failover_enabled': pool.failover_enabled,
                'stats': {
                    'total_requests': stats.total_requests,
                    'successful_requests': stats.successful_requests,
                    'failed_requests': stats.failed_requests,
                    'avg_response_time': round(stats.avg_response_time, 3),
                    'requests_per_second': round(stats.requests_per_second, 2),
                    'active_connections': stats.active_connections
                },
                'backends': backend_status
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get pool status for {pool_name}: {e}")
            return None
    
    def get_load_balancer_summary(self) -> Dict[str, Any]:
        """Get load balancer summary."""
        try:
            total_backends = sum(len(pool.backends) for pool in self.pools.values())
            healthy_backends = sum(
                sum(1 for b in pool.backends if b.status == BackendStatus.HEALTHY)
                for pool in self.pools.values()
            )
            
            total_requests = sum(stats.total_requests for stats in self.pool_stats.values())
            total_rps = sum(stats.requests_per_second for stats in self.pool_stats.values())
            
            avg_response_times = [
                stats.avg_response_time for stats in self.pool_stats.values()
                if stats.avg_response_time > 0
            ]
            overall_avg_rt = statistics.mean(avg_response_times) if avg_response_times else 0
            
            return {
                'enabled': self.lb_config['enabled'],
                'health_checks_active': self.health_check_active,
                'total_pools': len(self.pools),
                'total_backends': total_backends,
                'healthy_backends': healthy_backends,
                'health_percentage': f"{(healthy_backends / total_backends * 100):.1f}%" if total_backends > 0 else "0%",
                'total_requests_handled': total_requests,
                'current_requests_per_second': round(total_rps, 2),
                'average_response_time': f"{overall_avg_rt:.3f}s",
                'active_sessions': len(self.session_store),
                'algorithms_in_use': list(set(pool.algorithm.value for pool in self.pools.values())),
                'sticky_sessions_enabled': sum(1 for pool in self.pools.values() if pool.sticky_sessions)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate load balancer summary: {e}")
            return {'error': 'Unable to generate summary'}