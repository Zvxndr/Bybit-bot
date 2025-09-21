""""""

Unified API Initialization - Phase 3 API ConsolidationAPI Module Initialization



This module provides the main entry point for the unified API system.This module provides the production-grade API layer for the cryptocurrency

It orchestrates the initialization of all API components and providestrading bot, including real-time prediction serving, monitoring endpoints,

a single interface for the trading bot to interact with Bybit APIs.and WebSocket streaming capabilities.

"""

Key Features:

- Single initialization point for all API componentsfrom .prediction_service import create_app, run_server, PredictionAPI

- Automatic component orchestration

- Health monitoring and diagnostics__all__ = ['create_app', 'run_server', 'PredictionAPI']
- Error recovery and failover
- Configuration validation
- Connection management
- Performance monitoring
- Australian compliance integration
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path

# Import unified API components
from .unified_bybit_client import UnifiedBybitClient, BybitCredentials
from .websocket_manager import UnifiedWebSocketManager, StreamType
from .market_data_pipeline import UnifiedMarketDataPipeline, PipelineConfig
from .config import (
    UnifiedAPIConfig, ConfigurationManager, Environment,
    create_default_config, create_production_config, create_testnet_config
)

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# SYSTEM STATUS AND METRICS
# ============================================================================

@dataclass
class APISystemStatus:
    """Overall API system status"""
    is_initialized: bool = False
    is_healthy: bool = False
    initialization_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
    # Component status
    rest_client_status: str = "not_initialized"
    websocket_manager_status: str = "not_initialized"
    market_data_pipeline_status: str = "not_initialized"
    
    # Performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    
    # Error tracking
    last_error: Optional[str] = None
    error_count: int = 0
    
    def get_success_rate(self) -> float:
        """Get request success rate percentage"""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0

# ============================================================================
# UNIFIED API SYSTEM
# ============================================================================

class UnifiedAPISystem:
    """
    Unified API System
    
    Main orchestrator for all Bybit API operations
    """
    
    def __init__(
        self,
        config: Optional[UnifiedAPIConfig] = None,
        config_file: Optional[str] = None
    ):
        # Configuration
        if config_file:
            self.config_manager = ConfigurationManager()
            self.config_manager.load_from_file(config_file)
            self.config = self.config_manager.get_config()
        elif config:
            self.config = config
            self.config_manager = ConfigurationManager(config)
        else:
            self.config = create_default_config()
            self.config_manager = ConfigurationManager(self.config)
        
        # Core components
        self.rest_client: Optional[UnifiedBybitClient] = None
        self.websocket_manager: Optional[UnifiedWebSocketManager] = None
        self.market_data_pipeline: Optional[UnifiedMarketDataPipeline] = None
        
        # System state
        self.status = APISystemStatus()
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Monitoring
        self.health_check_interval = 60  # seconds
        self.performance_metrics: Dict[str, List[float]] = {
            'latency': [],
            'success_rate': [],
            'error_rate': []
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'system_started': [],
            'system_stopped': [],
            'health_check': [],
            'error_occurred': [],
            'component_reconnected': []
        }
        
        logger.info(f"Unified API System initialized for {self.config.environment.value}")
    
    async def initialize(self, symbols: Optional[List[str]] = None) -> bool:
        """Initialize all API components"""
        if self.status.is_initialized:
            logger.warning("API system already initialized")
            return True
        
        logger.info("Initializing Unified API System...")
        
        try:
            start_time = time.time()
            
            # Validate configuration
            if not self._validate_configuration():
                raise RuntimeError("Configuration validation failed")
            
            # Initialize REST client
            await self._initialize_rest_client()
            
            # Initialize WebSocket manager if enabled
            if self.config.enable_websockets:
                await self._initialize_websocket_manager()
            
            # Initialize market data pipeline if enabled
            if self.config.enable_market_data and symbols:
                await self._initialize_market_data_pipeline(symbols)
            
            # Start background monitoring
            await self._start_background_tasks()
            
            # Update status
            self.status.is_initialized = True
            self.status.initialization_time = datetime.now()
            self.status.is_healthy = True
            self.is_running = True
            
            initialization_duration = time.time() - start_time
            logger.info(f"API System initialized successfully in {initialization_duration:.2f}s")
            
            # Emit system started event
            await self._emit_event('system_started', {
                'initialization_time': self.status.initialization_time.isoformat(),
                'duration_seconds': initialization_duration,
                'components_initialized': self._get_initialized_components()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize API system: {e}")
            self.status.last_error = str(e)
            self.status.error_count += 1
            self.status.is_healthy = False
            
            # Cleanup on failure
            await self.shutdown()
            
            raise
    
    async def shutdown(self):
        """Shutdown all API components"""
        if not self.is_running:
            logger.warning("API system not running")
            return
        
        logger.info("Shutting down Unified API System...")
        
        try:
            # Signal shutdown
            self.is_running = False
            self.shutdown_event.set()
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.background_tasks.clear()
            
            # Shutdown components
            if self.market_data_pipeline:
                await self.market_data_pipeline.stop()
                self.market_data_pipeline = None
                self.status.market_data_pipeline_status = "stopped"
            
            if self.websocket_manager:
                await self.websocket_manager.stop()
                self.websocket_manager = None
                self.status.websocket_manager_status = "stopped"
            
            if self.rest_client:
                await self.rest_client.disconnect()
                self.rest_client = None
                self.status.rest_client_status = "stopped"
            
            # Update status
            self.status.is_initialized = False
            self.status.is_healthy = False
            
            logger.info("API System shutdown complete")
            
            # Emit system stopped event
            await self._emit_event('system_stopped', {
                'shutdown_time': datetime.now().isoformat(),
                'uptime_seconds': self.get_uptime()
            })
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.status.last_error = str(e)
            self.status.error_count += 1
    
    async def _initialize_rest_client(self):
        """Initialize REST API client"""
        logger.info("Initializing REST client...")
        
        if not self.config.credentials:
            raise RuntimeError("Credentials required for REST client")
        
        self.rest_client = UnifiedBybitClient(
            credentials=self.config.credentials,
            enable_rate_limiting=self.config.rate_limits.enable_rate_limiting,
            connection_pool_size=self.config.connection.pool_size,
            request_timeout=self.config.connection.timeout,
            max_retries=self.config.connection.max_retries,
            enable_websockets=False  # WebSocket handled separately
        )
        
        await self.rest_client.connect()
        self.status.rest_client_status = "connected"
        
        logger.info("REST client initialized successfully")
    
    async def _initialize_websocket_manager(self):
        """Initialize WebSocket manager"""
        logger.info("Initializing WebSocket manager...")
        
        if not self.config.credentials:
            raise RuntimeError("Credentials required for WebSocket manager")
        
        self.websocket_manager = UnifiedWebSocketManager(self.config.credentials)
        await self.websocket_manager.start()
        self.status.websocket_manager_status = "connected"
        
        # Add event callbacks
        self.websocket_manager.add_event_callback(
            'connection_established',
            self._handle_websocket_connected
        )
        self.websocket_manager.add_event_callback(
            'connection_lost',
            self._handle_websocket_disconnected
        )
        
        logger.info("WebSocket manager initialized successfully")
    
    async def _initialize_market_data_pipeline(self, symbols: List[str]):
        """Initialize market data pipeline"""
        logger.info(f"Initializing market data pipeline for {len(symbols)} symbols...")
        
        if not self.config.credentials:
            raise RuntimeError("Credentials required for market data pipeline")
        
        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            cache_size=self.config.cache.cache_size,
            cache_ttl_seconds=self.config.cache.cache_ttl_seconds,
            timezone=self.config.australian_compliance.timezone,
            enable_compression=self.config.performance.enable_batching,
            batch_size=self.config.performance.batch_size
        )
        
        self.market_data_pipeline = UnifiedMarketDataPipeline(
            credentials=self.config.credentials,
            config=pipeline_config
        )
        
        await self.market_data_pipeline.start(symbols)
        self.status.market_data_pipeline_status = "running"
        
        logger.info("Market data pipeline initialized successfully")
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        self.background_tasks = [
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._configuration_monitor())
        ]
        
        logger.info("Background monitoring tasks started")
    
    async def _health_monitor(self):
        """Monitor system health"""
        while self.is_running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _performance_monitor(self):
        """Monitor system performance"""
        while self.is_running:
            try:
                # Update uptime
                if self.status.initialization_time:
                    self.status.uptime_seconds = (
                        datetime.now() - self.status.initialization_time
                    ).total_seconds()
                
                # Collect performance metrics
                await self._collect_performance_metrics()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(30)
    
    async def _configuration_monitor(self):
        """Monitor configuration changes"""
        while self.is_running:
            try:
                # Check for configuration changes
                if self.config_manager.reload_if_changed():
                    logger.info("Configuration reloaded, updating components...")
                    # Note: In a full implementation, you'd handle config changes here
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in configuration monitor: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        overall_healthy = True
        
        # Check REST client
        if self.rest_client:
            try:
                client_health = await self.rest_client.health_check()
                health_data['components']['rest_client'] = client_health
                
                if client_health.get('overall_status') != 'healthy':
                    overall_healthy = False
                    
                # Update metrics
                self.status.total_requests = client_health.get('request_count', 0)
                self.status.failed_requests = client_health.get('error_count', 0)
                self.status.successful_requests = self.status.total_requests - self.status.failed_requests
                
            except Exception as e:
                logger.error(f"REST client health check failed: {e}")
                health_data['components']['rest_client'] = {'status': 'error', 'error': str(e)}
                overall_healthy = False
        
        # Check WebSocket manager
        if self.websocket_manager:
            try:
                ws_status = self.websocket_manager.get_status()
                health_data['components']['websocket_manager'] = ws_status
                
                if not ws_status.get('is_running'):
                    overall_healthy = False
            except Exception as e:
                logger.error(f"WebSocket manager health check failed: {e}")
                health_data['components']['websocket_manager'] = {'status': 'error', 'error': str(e)}
                overall_healthy = False
        
        # Check market data pipeline
        if self.market_data_pipeline:
            try:
                pipeline_status = self.market_data_pipeline.get_status()
                health_data['components']['market_data_pipeline'] = pipeline_status
                
                if not pipeline_status.get('is_running'):
                    overall_healthy = False
            except Exception as e:
                logger.error(f"Market data pipeline health check failed: {e}")
                health_data['components']['market_data_pipeline'] = {'status': 'error', 'error': str(e)}
                overall_healthy = False
        
        # Update system status
        self.status.is_healthy = overall_healthy
        
        # Emit health check event
        await self._emit_event('health_check', health_data)
        
        if not overall_healthy:
            logger.warning("System health check failed")
    
    async def _collect_performance_metrics(self):
        """Collect performance metrics"""
        try:
            # Calculate success rate
            success_rate = self.status.get_success_rate()
            self.performance_metrics['success_rate'].append(success_rate)
            
            # Calculate error rate
            error_rate = 100.0 - success_rate
            self.performance_metrics['error_rate'].append(error_rate)
            
            # Collect latency from REST client
            if self.rest_client:
                client_status = self.rest_client.get_connection_status()
                # Note: In a real implementation, you'd extract latency metrics
                
            # Trim metrics history (keep last 1000 samples)
            for metric_list in self.performance_metrics.values():
                if len(metric_list) > 1000:
                    metric_list.pop(0)
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    async def _handle_websocket_connected(self, data: Dict[str, Any]):
        """Handle WebSocket connection established"""
        logger.info(f"WebSocket connected: {data}")
        await self._emit_event('component_reconnected', {
            'component': 'websocket',
            'data': data
        })
    
    async def _handle_websocket_disconnected(self, data: Dict[str, Any]):
        """Handle WebSocket connection lost"""
        logger.warning(f"WebSocket disconnected: {data}")
        await self._emit_event('error_occurred', {
            'component': 'websocket',
            'error': 'connection_lost',
            'data': data
        })
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered callbacks"""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in {event_type} event callback: {e}")
    
    def _validate_configuration(self) -> bool:
        """Validate system configuration"""
        try:
            # Validate credentials if required
            if (self.config.enable_trading or 
                self.config.enable_websockets or 
                self.config.enable_market_data):
                
                if not self.config.credentials:
                    logger.error("Credentials required for enabled features")
                    return False
                
                if not self.config.validate_credentials():
                    logger.error("Invalid credentials format")
                    return False
            
            # Validate Australian compliance settings
            if self.config.australian_compliance.enable_tax_reporting:
                if not self.config.australian_compliance.timezone.startswith('Australia/'):
                    logger.warning("Non-Australian timezone for tax reporting")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def _get_initialized_components(self) -> List[str]:
        """Get list of initialized components"""
        components = []
        
        if self.rest_client:
            components.append('rest_client')
        if self.websocket_manager:
            components.append('websocket_manager')
        if self.market_data_pipeline:
            components.append('market_data_pipeline')
        
        return components
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    def get_rest_client(self) -> Optional[UnifiedBybitClient]:
        """Get REST API client"""
        return self.rest_client
    
    def get_websocket_manager(self) -> Optional[UnifiedWebSocketManager]:
        """Get WebSocket manager"""
        return self.websocket_manager
    
    def get_market_data_pipeline(self) -> Optional[UnifiedMarketDataPipeline]:
        """Get market data pipeline"""
        return self.market_data_pipeline
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system': {
                'is_initialized': self.status.is_initialized,
                'is_healthy': self.status.is_healthy,
                'is_running': self.is_running,
                'initialization_time': self.status.initialization_time.isoformat() if self.status.initialization_time else None,
                'uptime_seconds': self.status.uptime_seconds,
                'uptime_readable': str(timedelta(seconds=int(self.status.uptime_seconds)))
            },
            'components': {
                'rest_client': self.status.rest_client_status,
                'websocket_manager': self.status.websocket_manager_status,
                'market_data_pipeline': self.status.market_data_pipeline_status
            },
            'performance': {
                'total_requests': self.status.total_requests,
                'successful_requests': self.status.successful_requests,
                'failed_requests': self.status.failed_requests,
                'success_rate_pct': self.status.get_success_rate(),
                'average_latency_ms': self.status.average_latency_ms
            },
            'errors': {
                'error_count': self.status.error_count,
                'last_error': self.status.last_error
            },
            'configuration': self.config.get_summary()
        }
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        if not self.status.initialization_time:
            return 0.0
        return (datetime.now() - self.status.initialization_time).total_seconds()
    
    def add_event_callback(self, event_type: str, callback: Callable):
        """Add event callback"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
            logger.info(f"Added callback for {event_type} event")
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    async def reload_configuration(self):
        """Reload configuration and update components"""
        logger.info("Reloading system configuration...")
        
        old_config = self.config
        self.config = self.config_manager.get_config()
        
        # Note: In a full implementation, you'd update components based on config changes
        logger.info("Configuration reloaded successfully")

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

async def create_api_system(
    api_key: str,
    api_secret: str,
    environment: Environment = Environment.TESTNET,
    symbols: Optional[List[str]] = None,
    **config_overrides
) -> UnifiedAPISystem:
    """Create and initialize API system"""
    
    # Create configuration
    if environment == Environment.MAINNET:
        config = create_production_config(api_key, api_secret)
    else:
        config = create_testnet_config(api_key, api_secret)
    
    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create and initialize system
    system = UnifiedAPISystem(config)
    await system.initialize(symbols)
    
    return system

async def create_api_system_from_config(
    config: UnifiedAPIConfig,
    symbols: Optional[List[str]] = None
) -> UnifiedAPISystem:
    """Create API system from configuration"""
    
    system = UnifiedAPISystem(config)
    await system.initialize(symbols)
    
    return system

async def create_api_system_from_file(
    config_file: str,
    symbols: Optional[List[str]] = None
) -> UnifiedAPISystem:
    """Create API system from configuration file"""
    
    system = UnifiedAPISystem(config_file=config_file)
    await system.initialize(symbols)
    
    return system

# ============================================================================
# CONTEXT MANAGER
# ============================================================================

class APISystemContext:
    """Context manager for API system"""
    
    def __init__(self, system: UnifiedAPISystem):
        self.system = system
    
    async def __aenter__(self):
        return self.system
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.system.shutdown()

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'UnifiedAPISystem',
    'APISystemStatus',
    'APISystemContext',
    'create_api_system',
    'create_api_system_from_config',
    'create_api_system_from_file'
]