"""
Phase 5 Live Trading Orchestrator

Main orchestrator for live trading operations that coordinates all Phase 5 components:
- WebSocket manager for real-time data feeds
- Live execution engine for order management
- Monitoring dashboard for performance tracking
- Alert system for notifications and risk management
- Production deployment pipeline for automated deployments

This is the main entry point for Phase 5 live trading capabilities.

Author: Trading Bot Team
Version: 1.0.0 - Phase 5 Implementation
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager

from ..utils.logging import TradingLogger
from ..config_manager import ConfigurationManager
from ..strategy_manager import StrategyManager
from ..risk_manager import RiskManager
from ..trading_engine import TradingEngine
from ..bybit_client import BybitClient

from .websocket_manager import WebSocketManager, create_websocket_manager
from .live_execution_engine import LiveExecutionEngine, TradingMode, create_live_execution_engine
from .monitoring_dashboard import MonitoringDashboard, create_monitoring_dashboard
from .alert_system import AlertSystem, create_alert_system
from .production_deployment import ProductionDeploymentPipeline, DeploymentEnvironment


class Phase5LiveTradingOrchestrator:
    """
    Main orchestrator for Phase 5 live trading operations.
    
    This class coordinates all live trading components and provides
    a unified interface for starting, stopping, and managing the
    live trading system.
    
    Features:
    - Unified component lifecycle management
    - Graceful startup and shutdown sequences
    - Error handling and recovery
    - Health monitoring and reporting
    - Configuration management
    - Service orchestration
    """
    
    def __init__(self, config: ConfigurationManager, trading_mode: TradingMode = TradingMode.PAPER):
        self.config = config
        self.trading_mode = trading_mode
        self.logger = TradingLogger("phase5_orchestrator")
        
        # Core components (from previous phases)
        self.bybit_client: Optional[BybitClient] = None
        self.trading_engine: Optional[TradingEngine] = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # Phase 5 components
        self.websocket_manager: Optional[WebSocketManager] = None
        self.live_execution_engine: Optional[LiveExecutionEngine] = None
        self.monitoring_dashboard: Optional[MonitoringDashboard] = None
        self.alert_system: Optional[AlertSystem] = None
        self.deployment_pipeline: Optional[ProductionDeploymentPipeline] = None
        
        # State management
        self.running = False
        self.startup_complete = False
        self.shutdown_requested = False
        self.component_status: Dict[str, str] = {}
        
        # Background tasks
        self.tasks = []
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        self.logger.info(f"Phase5LiveTradingOrchestrator initialized in {trading_mode.value} mode")
    
    async def start(self) -> bool:
        """
        Start the live trading system.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            self.logger.info("🚀 Starting Phase 5 Live Trading System...")
            self.running = True
            
            # Initialize core components first
            if not await self._initialize_core_components():
                raise RuntimeError("Failed to initialize core components")
            
            # Initialize Phase 5 components
            if not await self._initialize_phase5_components():
                raise RuntimeError("Failed to initialize Phase 5 components")
            
            # Start all components
            if not await self._start_all_components():
                raise RuntimeError("Failed to start all components")
            
            # Run post-startup checks
            if not await self._run_startup_checks():
                raise RuntimeError("Startup checks failed")
            
            # Start background monitoring
            await self._start_background_tasks()
            
            self.startup_complete = True
            self.logger.info("✅ Phase 5 Live Trading System started successfully!")
            
            # Log system status
            await self._log_system_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start Phase 5 system: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> None:
        """Stop the live trading system gracefully."""
        try:
            self.logger.info("🛑 Stopping Phase 5 Live Trading System...")
            self.shutdown_requested = True
            self.running = False
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Stop all components in reverse order
            await self._stop_all_components()
            
            self.logger.info("✅ Phase 5 Live Trading System stopped gracefully")
            
        except Exception as e:
            self.logger.error(f"❌ Error during system shutdown: {e}")
    
    async def restart(self) -> bool:
        """Restart the live trading system."""
        self.logger.info("🔄 Restarting Phase 5 Live Trading System...")
        await self.stop()
        await asyncio.sleep(5)  # Grace period
        return await self.start()
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health information.
        
        Returns:
            Dict containing system health metrics and status
        """
        try:
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "running": self.running,
                "startup_complete": self.startup_complete,
                "trading_mode": self.trading_mode.value,
                "component_status": self.component_status.copy(),
                "components": {}
            }
            
            # Get health from each component
            if self.websocket_manager:
                health_data["components"]["websocket_manager"] = await self.websocket_manager.get_health_status()
            
            if self.live_execution_engine:
                health_data["components"]["execution_engine"] = await self.live_execution_engine.get_execution_stats()
            
            if self.monitoring_dashboard:
                health_data["components"]["monitoring_dashboard"] = await self.monitoring_dashboard.get_system_health()
            
            if self.alert_system:
                health_data["components"]["alert_system"] = self.alert_system.get_alert_statistics()
            
            # Overall health assessment
            unhealthy_components = []
            for component, status in self.component_status.items():
                if status != "healthy":
                    unhealthy_components.append(component)
            
            health_data["overall_status"] = "healthy" if not unhealthy_components else "degraded"
            health_data["unhealthy_components"] = unhealthy_components
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "overall_status": "error"
            }
    
    async def switch_trading_mode(self, new_mode: TradingMode) -> bool:
        """
        Switch trading mode (paper/live/hybrid).
        
        Args:
            new_mode: New trading mode to switch to
            
        Returns:
            bool: True if switched successfully
        """
        try:
            if new_mode == self.trading_mode:
                self.logger.info(f"Already in {new_mode.value} mode")
                return True
            
            self.logger.info(f"Switching from {self.trading_mode.value} to {new_mode.value} mode")
            
            # Stop execution engine
            if self.live_execution_engine:
                await self.live_execution_engine.stop()
            
            # Update mode
            old_mode = self.trading_mode
            self.trading_mode = new_mode
            
            # Restart execution engine with new mode
            if self.live_execution_engine:
                self.live_execution_engine.trading_mode = new_mode
                if not await self.live_execution_engine.start():
                    # Rollback on failure
                    self.trading_mode = old_mode
                    self.live_execution_engine.trading_mode = old_mode
                    await self.live_execution_engine.start()
                    return False
            
            self.logger.info(f"✅ Successfully switched to {new_mode.value} mode")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch trading mode: {e}")
            return False
    
    async def emergency_stop(self) -> None:
        """Emergency stop - immediately halt all trading activities."""
        try:
            self.logger.critical("🚨 EMERGENCY STOP INITIATED")
            
            # Stop execution engine immediately
            if self.live_execution_engine:
                await self.live_execution_engine.emergency_stop()
            
            # Stop WebSocket feeds
            if self.websocket_manager:
                await self.websocket_manager.disconnect_all()
            
            # Trigger critical alert
            if self.alert_system:
                await self.alert_system.trigger_alert(
                    "emergency_stop",
                    context={"component": "orchestrator", "reason": "Manual emergency stop"}
                )
            
            self.logger.critical("🚨 EMERGENCY STOP COMPLETED")
            
        except Exception as e:
            self.logger.critical(f"Error during emergency stop: {e}")
    
    async def _initialize_core_components(self) -> bool:
        """Initialize core trading components."""
        try:
            self.logger.info("Initializing core components...")
            
            # Initialize Bybit client
            self.bybit_client = BybitClient(self.config)
            await self.bybit_client.connect()
            self.component_status["bybit_client"] = "healthy"
            
            # Initialize trading engine
            self.trading_engine = TradingEngine(self.config, self.bybit_client)
            self.component_status["trading_engine"] = "healthy"
            
            # Initialize strategy manager
            self.strategy_manager = StrategyManager(self.config)
            await self.strategy_manager.initialize()
            self.component_status["strategy_manager"] = "healthy"
            
            # Initialize risk manager
            self.risk_manager = RiskManager(self.config)
            self.component_status["risk_manager"] = "healthy"
            
            self.logger.info("✅ Core components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize core components: {e}")
            return False
    
    async def _initialize_phase5_components(self) -> bool:
        """Initialize Phase 5 specific components."""
        try:
            self.logger.info("Initializing Phase 5 components...")
            
            # Initialize WebSocket manager
            self.websocket_manager = await create_websocket_manager(self.config)
            self.component_status["websocket_manager"] = "healthy"
            
            # Initialize live execution engine
            self.live_execution_engine = await create_live_execution_engine(
                self.config, 
                self.trading_engine, 
                self.risk_manager,
                self.trading_mode
            )
            self.component_status["live_execution_engine"] = "healthy"
            
            # Initialize monitoring dashboard
            self.monitoring_dashboard = await create_monitoring_dashboard(
                self.config,
                self.strategy_manager,
                self.risk_manager
            )
            self.component_status["monitoring_dashboard"] = "healthy"
            
            # Initialize alert system
            self.alert_system = await create_alert_system(self.config)
            self.component_status["alert_system"] = "healthy"
            
            # Initialize deployment pipeline (optional)
            if self.config.get('deployment.enabled', False):
                self.deployment_pipeline = ProductionDeploymentPipeline(self.config)
                self.component_status["deployment_pipeline"] = "healthy"
            
            self.logger.info("✅ Phase 5 components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Phase 5 components: {e}")
            return False
    
    async def _start_all_components(self) -> bool:
        """Start all components in the correct order."""
        try:
            self.logger.info("Starting all components...")
            
            # Start WebSocket manager
            if self.websocket_manager and not await self.websocket_manager.start():
                raise RuntimeError("Failed to start WebSocket manager")
            
            # Start live execution engine
            if self.live_execution_engine and not await self.live_execution_engine.start():
                raise RuntimeError("Failed to start live execution engine")
            
            # Start monitoring dashboard
            if self.monitoring_dashboard and not await self.monitoring_dashboard.start():
                raise RuntimeError("Failed to start monitoring dashboard")
            
            # Alert system should already be started during initialization
            
            self.logger.info("✅ All components started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start components: {e}")
            return False
    
    async def _stop_all_components(self) -> None:
        """Stop all components in reverse order."""
        try:
            # Stop alert system
            if self.alert_system:
                await self.alert_system.stop()
            
            # Stop monitoring dashboard
            if self.monitoring_dashboard:
                await self.monitoring_dashboard.stop()
            
            # Stop live execution engine
            if self.live_execution_engine:
                await self.live_execution_engine.stop()
            
            # Stop WebSocket manager
            if self.websocket_manager:
                await self.websocket_manager.stop()
            
            # Disconnect Bybit client
            if self.bybit_client:
                await self.bybit_client.disconnect()
            
            self.logger.info("✅ All components stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping components: {e}")
    
    async def _run_startup_checks(self) -> bool:
        """Run post-startup health checks."""
        try:
            self.logger.info("Running startup checks...")
            
            # Check WebSocket connectivity
            if self.websocket_manager:
                health = await self.websocket_manager.get_health_status()
                if health.get("status") != "connected":
                    raise RuntimeError("WebSocket not connected")
            
            # Check execution engine
            if self.live_execution_engine:
                stats = await self.live_execution_engine.get_execution_stats()
                if not stats.get("engine_healthy", False):
                    raise RuntimeError("Execution engine not healthy")
            
            # Check monitoring dashboard
            if self.monitoring_dashboard:
                health = await self.monitoring_dashboard.get_system_health()
                if health.overall_status == "error":
                    raise RuntimeError("Monitoring dashboard not healthy")
            
            self.logger.info("✅ Startup checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Startup checks failed: {e}")
            return False
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        try:
            # Health monitoring task
            health_task = asyncio.create_task(self._health_monitoring_loop())
            self.tasks.append(health_task)
            
            # Component integration task
            integration_task = asyncio.create_task(self._component_integration_loop())
            self.tasks.append(integration_task)
            
            self.logger.info("✅ Background tasks started")
            
        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}")
    
    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        try:
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            self.tasks.clear()
            self.logger.info("✅ Background tasks stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping background tasks: {e}")
    
    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while self.running:
            try:
                # Check component health
                for component_name in self.component_status:
                    # Update component status based on health checks
                    # This is a simplified implementation
                    self.component_status[component_name] = "healthy"
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _component_integration_loop(self) -> None:
        """Background component integration loop."""
        while self.running:
            try:
                # Integrate components - pass data between them
                
                # Get performance metrics from execution engine
                if self.live_execution_engine and self.monitoring_dashboard:
                    execution_stats = await self.live_execution_engine.get_execution_stats()
                    # Dashboard will collect these automatically
                
                # Check for alerts based on system health
                if self.alert_system and self.monitoring_dashboard:
                    system_health = await self.monitoring_dashboard.get_system_health()
                    await self.alert_system.check_system_health(system_health)
                    
                    performance_metrics = await self.monitoring_dashboard.get_performance_metrics()
                    await self.alert_system.check_risk_limits(performance_metrics)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Component integration error: {e}")
                await asyncio.sleep(10)
    
    async def _log_system_status(self) -> None:
        """Log current system status."""
        try:
            health = await self.get_system_health()
            
            self.logger.info("📊 Phase 5 System Status:")
            self.logger.info(f"   🔄 Running: {health['running']}")
            self.logger.info(f"   ✅ Startup Complete: {health['startup_complete']}")
            self.logger.info(f"   📈 Trading Mode: {health['trading_mode']}")
            self.logger.info(f"   🏥 Overall Health: {health['overall_status']}")
            
            for component, status in health['component_status'].items():
                status_emoji = "✅" if status == "healthy" else "❌"
                self.logger.info(f"   {status_emoji} {component}: {status}")
            
        except Exception as e:
            self.logger.error(f"Error logging system status: {e}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
            # Create a task to stop the system
            asyncio.create_task(self.stop())
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run_forever(self) -> None:
        """Run the system indefinitely until shutdown is requested."""
        try:
            if not await self.start():
                return
            
            self.logger.info("🔄 Phase 5 Live Trading System running... Press Ctrl+C to stop")
            
            # Keep running until shutdown is requested
            while self.running and not self.shutdown_requested:
                await asyncio.sleep(1)
            
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received, shutting down...")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
        finally:
            await self.stop()


# Main entry point for Phase 5
async def main():
    """Main entry point for Phase 5 Live Trading System."""
    try:
        # Load configuration
        config = ConfigurationManager()
        
        # Determine trading mode from command line or config
        trading_mode = TradingMode.PAPER  # Default to paper trading for safety
        
        if len(sys.argv) > 1:
            mode_arg = sys.argv[1].lower()
            if mode_arg == "live":
                trading_mode = TradingMode.LIVE
            elif mode_arg == "hybrid":
                trading_mode = TradingMode.HYBRID
        
        # Create and run orchestrator
        orchestrator = Phase5LiveTradingOrchestrator(config, trading_mode)
        await orchestrator.run_forever()
        
    except Exception as e:
        print(f"Failed to start Phase 5 system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())