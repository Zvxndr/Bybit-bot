"""
Container Orchestrator for Managing Container Lifecycle.
Handles container creation, deployment, scaling, and lifecycle management.
"""

import asyncio
import docker
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class ContainerState(Enum):
    """Container states."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"

class RestartPolicy(Enum):
    """Container restart policies."""
    NO = "no"
    ON_FAILURE = "on-failure"
    ALWAYS = "always"
    UNLESS_STOPPED = "unless-stopped"

class NetworkMode(Enum):
    """Container network modes."""
    BRIDGE = "bridge"
    HOST = "host"
    NONE = "none"
    CONTAINER = "container"
    CUSTOM = "custom"

@dataclass
class ContainerConfig:
    """Container configuration."""
    name: str
    image: str
    command: Optional[List[str]] = None
    environment: Dict[str, str] = field(default_factory=dict)
    ports: Dict[str, int] = field(default_factory=dict)  # container_port: host_port
    volumes: Dict[str, str] = field(default_factory=dict)  # host_path: container_path
    working_dir: Optional[str] = None
    user: Optional[str] = None
    restart_policy: RestartPolicy = RestartPolicy.UNLESS_STOPPED
    network_mode: NetworkMode = NetworkMode.BRIDGE
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    health_check: Optional[Dict[str, Any]] = None
    privileged: bool = False
    auto_remove: bool = False

@dataclass
class ContainerInfo:
    """Container information and status."""
    container_id: str
    name: str
    image: str
    state: ContainerState
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    ports: Dict[str, int] = field(default_factory=dict)
    networks: List[str] = field(default_factory=list)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_rx: float = 0.0
    network_tx: float = 0.0
    restart_count: int = 0
    health_status: str = "unknown"

@dataclass
class ContainerMetrics:
    """Container performance metrics."""
    container_id: str
    name: str
    cpu_percent: float
    memory_usage: int
    memory_limit: int
    memory_percent: float
    network_rx_bytes: int
    network_tx_bytes: int
    block_read: int
    block_write: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ContainerEvent:
    """Container event for monitoring."""
    container_id: str
    container_name: str
    event_type: str  # start, stop, die, health_status, etc.
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)

class ContainerOrchestrator:
    """Container lifecycle management and orchestration."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            self.docker_available = True
            self.logger.info("Docker client connected successfully")
        except Exception as e:
            self.logger.error(f"Failed to connect to Docker: {e}")
            self.docker_client = None
            self.docker_available = False
        
        # Container tracking
        self.containers: Dict[str, ContainerInfo] = {}
        self.container_configs: Dict[str, ContainerConfig] = {}
        self.container_metrics: Dict[str, ContainerMetrics] = {}
        
        # Event handling
        self.event_queue = queue.Queue()
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Default container configurations
        self.default_configs = {
            'trading_engine': ContainerConfig(
                name='trading-engine',
                image='trading-system/trading-engine:latest',
                ports={'8000': 8000, '8001': 8001},
                environment={
                    'SERVICE_NAME': 'trading-engine',
                    'LOG_LEVEL': 'INFO',
                    'PYTHONUNBUFFERED': '1'
                },
                cpu_limit='2.0',
                memory_limit='4g',
                restart_policy=RestartPolicy.UNLESS_STOPPED,
                health_check={
                    'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                    'interval': 30,
                    'timeout': 10,
                    'retries': 3,
                    'start_period': 60
                },
                labels={
                    'service': 'trading-engine',
                    'tier': 'application',
                    'environment': 'production'
                }
            ),
            'market_data': ContainerConfig(
                name='market-data',
                image='trading-system/market-data:latest',
                ports={'8002': 8002},
                environment={
                    'SERVICE_NAME': 'market-data',
                    'REDIS_URL': 'redis://redis:6379',
                    'KAFKA_BROKERS': 'kafka:9092'
                },
                cpu_limit='1.0',
                memory_limit='2g',
                restart_policy=RestartPolicy.UNLESS_STOPPED
            ),
            'risk_manager': ContainerConfig(
                name='risk-manager',
                image='trading-system/risk-manager:latest',
                ports={'8003': 8003},
                environment={
                    'SERVICE_NAME': 'risk-manager',
                    'DATABASE_URL': 'postgresql://user:pass@postgres:5432/trading'
                },
                cpu_limit='1.0',
                memory_limit='2g',
                restart_policy=RestartPolicy.UNLESS_STOPPED
            ),
            'analytics': ContainerConfig(
                name='analytics',
                image='trading-system/analytics:latest',
                ports={'8004': 8004},
                environment={
                    'SERVICE_NAME': 'analytics',
                    'DASK_SCHEDULER': 'dask-scheduler:8786'
                },
                cpu_limit='2.0',
                memory_limit='8g',
                restart_policy=RestartPolicy.UNLESS_STOPPED
            ),
            'ml_engine': ContainerConfig(
                name='ml-engine',
                image='trading-system/ml-engine:latest',
                ports={'8005': 8005, '8501': 8501},
                environment={
                    'SERVICE_NAME': 'ml-engine',
                    'TENSORFLOW_SERVING_MODEL_SERVER_WORKERS': '4',
                    'CUDA_VISIBLE_DEVICES': '0'
                },
                cpu_limit='4.0',
                memory_limit='16g',
                restart_policy=RestartPolicy.UNLESS_STOPPED
            ),
            'hft_module': ContainerConfig(
                name='hft-module',
                image='trading-system/hft-module:latest',
                ports={'8006': 8006},
                environment={
                    'SERVICE_NAME': 'hft-module',
                    'LATENCY_MODE': 'ultra_low',
                    'CPU_AFFINITY': '0,1',
                    'RT_PRIORITY': '99'
                },
                cpu_limit='1.0',
                memory_limit='1g',
                restart_policy=RestartPolicy.UNLESS_STOPPED,
                privileged=True  # For RT scheduling
            ),
            'redis': ContainerConfig(
                name='redis',
                image='redis:7-alpine',
                ports={'6379': 6379},
                command=['redis-server', '--appendonly', 'yes'],
                volumes={'/data/redis': '/data'},
                cpu_limit='0.5',
                memory_limit='1g',
                restart_policy=RestartPolicy.UNLESS_STOPPED
            ),
            'postgres': ContainerConfig(
                name='postgres',
                image='postgres:15-alpine',
                ports={'5432': 5432},
                environment={
                    'POSTGRES_DB': 'trading',
                    'POSTGRES_USER': 'trading_user',
                    'POSTGRES_PASSWORD': 'secure_password'
                },
                volumes={'/data/postgres': '/var/lib/postgresql/data'},
                cpu_limit='1.0',
                memory_limit='2g',
                restart_policy=RestartPolicy.UNLESS_STOPPED
            ),
            'nginx': ContainerConfig(
                name='nginx-lb',
                image='nginx:alpine',
                ports={'80': 80, '443': 443},
                volumes={
                    '/config/nginx': '/etc/nginx/conf.d',
                    '/certs': '/etc/ssl/certs'
                },
                cpu_limit='0.5',
                memory_limit='512m',
                restart_policy=RestartPolicy.UNLESS_STOPPED
            )
        }
        
        self.logger.info("ContainerOrchestrator initialized")
    
    async def create_container(self, config: ContainerConfig) -> Optional[str]:
        """Create a new container."""
        try:
            if not self.docker_available:
                self.logger.error("Docker is not available")
                return None
            
            # Build container creation parameters
            container_params = {
                'name': config.name,
                'image': config.image,
                'detach': True,
                'environment': config.environment,
                'labels': config.labels,
                'restart_policy': {'Name': config.restart_policy.value},
                'auto_remove': config.auto_remove,
                'privileged': config.privileged
            }
            
            # Add command if specified
            if config.command:
                container_params['command'] = config.command
            
            # Add working directory
            if config.working_dir:
                container_params['working_dir'] = config.working_dir
            
            # Add user
            if config.user:
                container_params['user'] = config.user
            
            # Add port mappings
            if config.ports:
                container_params['ports'] = config.ports
            
            # Add volume mappings
            if config.volumes:
                container_params['volumes'] = config.volumes
            
            # Add resource limits
            host_config = {}
            if config.cpu_limit:
                host_config['cpu_quota'] = int(float(config.cpu_limit) * 100000)
                host_config['cpu_period'] = 100000
            
            if config.memory_limit:
                # Convert memory limit to bytes
                memory_bytes = self._parse_memory_limit(config.memory_limit)
                host_config['mem_limit'] = memory_bytes
            
            if host_config:
                container_params['host_config'] = self.docker_client.api.create_host_config(**host_config)
            
            # Add health check
            if config.health_check:
                container_params['healthcheck'] = config.health_check
            
            # Add network mode
            if config.network_mode != NetworkMode.BRIDGE:
                container_params['network_mode'] = config.network_mode.value
            
            # Create container
            container = self.docker_client.containers.create(**container_params)
            container_id = container.id
            
            # Store container info
            self.container_configs[container_id] = config
            
            container_info = ContainerInfo(
                container_id=container_id,
                name=config.name,
                image=config.image,
                state=ContainerState.CREATED,
                status="Created",
                created_at=datetime.now()
            )
            
            self.containers[container_id] = container_info
            
            self.logger.info(f"Created container: {config.name} ({container_id[:12]})")
            return container_id
            
        except Exception as e:
            self.logger.error(f"Failed to create container {config.name}: {e}")
            return None
    
    async def start_container(self, container_id: str) -> bool:
        """Start a container."""
        try:
            if not self.docker_available:
                return False
            
            container = self.docker_client.containers.get(container_id)
            container.start()
            
            # Update container info
            if container_id in self.containers:
                self.containers[container_id].state = ContainerState.RUNNING
                self.containers[container_id].status = "Running"
                self.containers[container_id].started_at = datetime.now()
            
            self.logger.info(f"Started container: {container_id[:12]}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start container {container_id}: {e}")
            return False
    
    async def stop_container(self, container_id: str, timeout: int = 10) -> bool:
        """Stop a container."""
        try:
            if not self.docker_available:
                return False
            
            container = self.docker_client.containers.get(container_id)
            container.stop(timeout=timeout)
            
            # Update container info
            if container_id in self.containers:
                self.containers[container_id].state = ContainerState.EXITED
                self.containers[container_id].status = "Exited"
                self.containers[container_id].finished_at = datetime.now()
            
            self.logger.info(f"Stopped container: {container_id[:12]}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop container {container_id}: {e}")
            return False
    
    async def restart_container(self, container_id: str, timeout: int = 10) -> bool:
        """Restart a container."""
        try:
            if not self.docker_available:
                return False
            
            container = self.docker_client.containers.get(container_id)
            container.restart(timeout=timeout)
            
            # Update container info
            if container_id in self.containers:
                self.containers[container_id].state = ContainerState.RUNNING
                self.containers[container_id].status = "Running"
                self.containers[container_id].started_at = datetime.now()
                self.containers[container_id].restart_count += 1
            
            self.logger.info(f"Restarted container: {container_id[:12]}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restart container {container_id}: {e}")
            return False
    
    async def remove_container(self, container_id: str, force: bool = False) -> bool:
        """Remove a container."""
        try:
            if not self.docker_available:
                return False
            
            container = self.docker_client.containers.get(container_id)
            container.remove(force=force)
            
            # Remove from tracking
            if container_id in self.containers:
                del self.containers[container_id]
            if container_id in self.container_configs:
                del self.container_configs[container_id]
            if container_id in self.container_metrics:
                del self.container_metrics[container_id]
            
            self.logger.info(f"Removed container: {container_id[:12]}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove container {container_id}: {e}")
            return False
    
    async def deploy_service(self, service_name: str, custom_config: Optional[ContainerConfig] = None) -> Optional[str]:
        """Deploy a service with predefined or custom configuration."""
        try:
            # Use custom config or default
            config = custom_config or self.default_configs.get(service_name)
            if not config:
                self.logger.error(f"No configuration found for service: {service_name}")
                return None
            
            # Check if container already exists
            existing_container = self._find_container_by_name(config.name)
            if existing_container:
                self.logger.info(f"Container {config.name} already exists, restarting...")
                await self.restart_container(existing_container.container_id)
                return existing_container.container_id
            
            # Create and start container
            container_id = await self.create_container(config)
            if container_id:
                success = await self.start_container(container_id)
                if success:
                    self.logger.info(f"Successfully deployed service: {service_name}")
                    return container_id
                else:
                    # Clean up failed container
                    await self.remove_container(container_id, force=True)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to deploy service {service_name}: {e}")
            return None
    
    async def deploy_trading_stack(self) -> Dict[str, Optional[str]]:
        """Deploy complete trading system stack."""
        try:
            deployment_results = {}
            
            # Deploy infrastructure services first
            infrastructure_services = ['redis', 'postgres']
            for service in infrastructure_services:
                self.logger.info(f"Deploying infrastructure service: {service}")
                container_id = await self.deploy_service(service)
                deployment_results[service] = container_id
                
                if container_id:
                    # Wait for service to be ready
                    await asyncio.sleep(5)
            
            # Deploy application services
            app_services = ['market_data', 'risk_manager', 'analytics', 'ml_engine', 'hft_module', 'trading_engine']
            for service in app_services:
                self.logger.info(f"Deploying application service: {service}")
                container_id = await self.deploy_service(service)
                deployment_results[service] = container_id
                
                if container_id:
                    await asyncio.sleep(2)
            
            # Deploy load balancer
            self.logger.info("Deploying load balancer")
            container_id = await self.deploy_service('nginx')
            deployment_results['nginx'] = container_id
            
            successful_deployments = sum(1 for cid in deployment_results.values() if cid is not None)
            total_services = len(deployment_results)
            
            self.logger.info(f"Trading stack deployment completed: {successful_deployments}/{total_services} services deployed")
            return deployment_results
            
        except Exception as e:
            self.logger.error(f"Failed to deploy trading stack: {e}")
            return {}
    
    async def scale_service(self, service_name: str, replicas: int) -> List[str]:
        """Scale a service to specified number of replicas."""
        try:
            if replicas < 1:
                self.logger.error("Replicas must be at least 1")
                return []
            
            # Get base configuration
            base_config = self.default_configs.get(service_name)
            if not base_config:
                self.logger.error(f"No configuration found for service: {service_name}")
                return []
            
            # Find existing containers for this service
            existing_containers = [
                info for info in self.containers.values()
                if info.name.startswith(service_name)
            ]
            
            current_replicas = len(existing_containers)
            container_ids = []
            
            if replicas > current_replicas:
                # Scale up - create new containers
                for i in range(current_replicas, replicas):
                    replica_config = ContainerConfig(
                        name=f"{service_name}-{i+1}",
                        image=base_config.image,
                        command=base_config.command,
                        environment=base_config.environment.copy(),
                        ports={str(int(port) + i): int(host_port) + i 
                              for port, host_port in base_config.ports.items()},
                        volumes=base_config.volumes,
                        working_dir=base_config.working_dir,
                        user=base_config.user,
                        restart_policy=base_config.restart_policy,
                        network_mode=base_config.network_mode,
                        cpu_limit=base_config.cpu_limit,
                        memory_limit=base_config.memory_limit,
                        labels=base_config.labels.copy(),
                        health_check=base_config.health_check,
                        privileged=base_config.privileged,
                        auto_remove=base_config.auto_remove
                    )
                    
                    # Update environment with replica info
                    replica_config.environment['REPLICA_ID'] = str(i + 1)
                    replica_config.labels['replica'] = str(i + 1)
                    
                    container_id = await self.deploy_service(f"{service_name}-{i+1}", replica_config)
                    if container_id:
                        container_ids.append(container_id)
            
            elif replicas < current_replicas:
                # Scale down - remove excess containers
                containers_to_remove = existing_containers[replicas:]
                for container_info in containers_to_remove:
                    await self.stop_container(container_info.container_id)
                    await self.remove_container(container_info.container_id)
                
                # Keep remaining containers
                container_ids = [info.container_id for info in existing_containers[:replicas]]
            
            else:
                # No scaling needed
                container_ids = [info.container_id for info in existing_containers]
            
            self.logger.info(f"Scaled {service_name} to {replicas} replicas")
            return container_ids
            
        except Exception as e:
            self.logger.error(f"Failed to scale service {service_name}: {e}")
            return []
    
    async def get_container_stats(self, container_id: str) -> Optional[ContainerMetrics]:
        """Get container performance statistics."""
        try:
            if not self.docker_available:
                return None
            
            container = self.docker_client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            # Parse CPU usage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            cpu_percent = 0.0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
            
            # Parse memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
            
            # Parse network I/O
            network_rx = 0
            network_tx = 0
            if 'networks' in stats:
                for interface_stats in stats['networks'].values():
                    network_rx += interface_stats['rx_bytes']
                    network_tx += interface_stats['tx_bytes']
            
            # Parse block I/O
            block_read = 0
            block_write = 0
            if 'blkio_stats' in stats and 'io_service_bytes_recursive' in stats['blkio_stats']:
                for entry in stats['blkio_stats']['io_service_bytes_recursive']:
                    if entry['op'] == 'Read':
                        block_read += entry['value']
                    elif entry['op'] == 'Write':
                        block_write += entry['value']
            
            container_info = self.containers.get(container_id)
            container_name = container_info.name if container_info else "unknown"
            
            metrics = ContainerMetrics(
                container_id=container_id,
                name=container_name,
                cpu_percent=cpu_percent,
                memory_usage=memory_usage,
                memory_limit=memory_limit,
                memory_percent=memory_percent,
                network_rx_bytes=network_rx,
                network_tx_bytes=network_tx,
                block_read=block_read,
                block_write=block_write
            )
            
            self.container_metrics[container_id] = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get container stats for {container_id}: {e}")
            return None
    
    async def get_container_logs(self, container_id: str, tail: int = 100) -> str:
        """Get container logs."""
        try:
            if not self.docker_available:
                return ""
            
            container = self.docker_client.containers.get(container_id)
            logs = container.logs(tail=tail, timestamps=True).decode('utf-8')
            
            return logs
            
        except Exception as e:
            self.logger.error(f"Failed to get logs for container {container_id}: {e}")
            return ""
    
    async def execute_command(self, container_id: str, command: Union[str, List[str]]) -> Dict[str, Any]:
        """Execute command in container."""
        try:
            if not self.docker_available:
                return {'success': False, 'error': 'Docker not available'}
            
            container = self.docker_client.containers.get(container_id)
            
            # Execute command
            result = container.exec_run(command, detach=False)
            
            return {
                'success': True,
                'exit_code': result.exit_code,
                'output': result.output.decode('utf-8') if result.output else '',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute command in container {container_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def start_monitoring(self):
        """Start container event monitoring."""
        try:
            if not self.docker_available or self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_events, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("Container monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop container event monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Container monitoring stopped")
    
    def add_event_handler(self, event_type: str, handler: Callable[[ContainerEvent], None]):
        """Add event handler for specific event types."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def _monitor_events(self):
        """Monitor Docker events."""
        try:
            if not self.docker_client:
                return
            
            for event in self.docker_client.events(decode=True):
                if not self.monitoring_active:
                    break
                
                if event.get('Type') == 'container':
                    container_event = ContainerEvent(
                        container_id=event.get('id', ''),
                        container_name=event.get('Actor', {}).get('Attributes', {}).get('name', ''),
                        event_type=event.get('Action', ''),
                        timestamp=datetime.fromtimestamp(event.get('time', 0)),
                        attributes=event.get('Actor', {}).get('Attributes', {})
                    )
                    
                    # Add to queue
                    self.event_queue.put(container_event)
                    
                    # Update container info
                    self._update_container_info(container_event)
                    
                    # Call event handlers
                    handlers = self.event_handlers.get(container_event.event_type, [])
                    for handler in handlers:
                        try:
                            handler(container_event)
                        except Exception as e:
                            self.logger.error(f"Event handler error: {e}")
                            
        except Exception as e:
            self.logger.error(f"Error monitoring events: {e}")
    
    def _update_container_info(self, event: ContainerEvent):
        """Update container information based on event."""
        try:
            container_id = event.container_id
            
            if container_id in self.containers:
                container_info = self.containers[container_id]
                
                if event.event_type == 'start':
                    container_info.state = ContainerState.RUNNING
                    container_info.status = "Running"
                    container_info.started_at = event.timestamp
                elif event.event_type == 'stop':
                    container_info.state = ContainerState.EXITED
                    container_info.status = "Exited"
                    container_info.finished_at = event.timestamp
                elif event.event_type == 'die':
                    container_info.state = ContainerState.EXITED
                    container_info.status = "Died"
                    container_info.finished_at = event.timestamp
                    if 'exitCode' in event.attributes:
                        container_info.exit_code = int(event.attributes['exitCode'])
                elif event.event_type == 'health_status':
                    container_info.health_status = event.attributes.get('health_status', 'unknown')
                
        except Exception as e:
            self.logger.error(f"Failed to update container info: {e}")
    
    def _find_container_by_name(self, name: str) -> Optional[ContainerInfo]:
        """Find container by name."""
        for container_info in self.containers.values():
            if container_info.name == name:
                return container_info
        return None
    
    def _parse_memory_limit(self, memory_limit: str) -> int:
        """Parse memory limit string to bytes."""
        try:
            memory_limit = memory_limit.lower()
            
            if memory_limit.endswith('k'):
                return int(memory_limit[:-1]) * 1024
            elif memory_limit.endswith('m'):
                return int(memory_limit[:-1]) * 1024 * 1024
            elif memory_limit.endswith('g'):
                return int(memory_limit[:-1]) * 1024 * 1024 * 1024
            else:
                return int(memory_limit)
                
        except ValueError:
            return 1024 * 1024 * 1024  # Default 1GB
    
    async def health_check_all(self) -> Dict[str, str]:
        """Perform health check on all containers."""
        try:
            health_status = {}
            
            for container_id, container_info in self.containers.items():
                try:
                    if not self.docker_available:
                        health_status[container_info.name] = "unknown"
                        continue
                    
                    container = self.docker_client.containers.get(container_id)
                    
                    # Check if container is running
                    if container.status != 'running':
                        health_status[container_info.name] = "stopped"
                        continue
                    
                    # Check health if health check is configured
                    health = container.attrs.get('State', {}).get('Health', {})
                    if health:
                        health_status[container_info.name] = health.get('Status', 'unknown')
                    else:
                        health_status[container_info.name] = "running"
                        
                except Exception as e:
                    health_status[container_info.name] = "error"
                    self.logger.error(f"Health check failed for {container_info.name}: {e}")
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Failed to perform health check: {e}")
            return {}
    
    def get_orchestration_summary(self) -> Dict[str, Any]:
        """Get container orchestration summary."""
        try:
            running_containers = sum(
                1 for info in self.containers.values()
                if info.state == ContainerState.RUNNING
            )
            
            total_cpu_limit = 0
            total_memory_limit = 0
            
            for container_id in self.containers:
                if container_id in self.container_configs:
                    config = self.container_configs[container_id]
                    if config.cpu_limit:
                        total_cpu_limit += float(config.cpu_limit)
                    if config.memory_limit:
                        # Convert to GB for summary
                        memory_bytes = self._parse_memory_limit(config.memory_limit)
                        total_memory_limit += memory_bytes / (1024 ** 3)
            
            return {
                'docker_available': self.docker_available,
                'total_containers': len(self.containers),
                'running_containers': running_containers,
                'monitoring_active': self.monitoring_active,
                'resource_allocation': {
                    'total_cpu_limit': f"{total_cpu_limit} cores",
                    'total_memory_limit': f"{total_memory_limit:.2f} GB"
                },
                'event_queue_size': self.event_queue.qsize(),
                'available_services': list(self.default_configs.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate orchestration summary: {e}")
            return {'error': 'Unable to generate summary'}