"""
ML Risk Management Integration Example

This example demonstrates how to integrate the ML-enhanced risk management system
into the existing Bybit trading bot. It shows:

1. How to set up the ML risk management system
2. How to integrate it with existing ML controllers
3. How to configure different risk levels for different environments
4. How to handle emergency stops and circuit breakers
5. How to monitor and log risk events

This integration ensures that all ML-generated trades pass through comprehensive
risk validation before execution, providing multiple layers of safety.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd

# Import existing bot components
from ..bot.core import TradingBot
from ..bot.integration.ml_integration_controller import MLIntegrationController
from ..bot.exchanges.bybit_client import BybitClient

# Import ML risk management components
from ..bot.risk import (
    UnifiedRiskManager, RiskParameters,
    MLRiskManager, MLTradeExecutionPipeline, MLRiskConfigManager,
    MLTradeRequest, ExecutionPriority, TradeValidationResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLEnhancedTradingBot:
    """
    Enhanced Trading Bot with ML Risk Management Integration
    
    This class demonstrates how to integrate the ML risk management system
    into the existing trading bot architecture.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", 
                 environment: str = "development"):
        """Initialize the ML-enhanced trading bot"""
        
        self.environment = environment
        self.config_path = config_path
        
        # Initialize core components
        self.trading_bot: Optional[TradingBot] = None
        self.ml_controller: Optional[MLIntegrationController] = None
        self.bybit_client: Optional[BybitClient] = None
        
        # Initialize risk management components
        self.unified_risk_manager: Optional[UnifiedRiskManager] = None
        self.ml_risk_manager: Optional[MLRiskManager] = None
        self.ml_execution_pipeline: Optional[MLTradeExecutionPipeline] = None
        self.ml_risk_config: Optional[MLRiskConfigManager] = None
        
        # State tracking
        self.is_running = False
        self.trade_requests_processed = 0
        self.trades_blocked = 0
        self.emergency_stops_triggered = 0
        
        logger.info(f"ML Enhanced Trading Bot initialized for {environment}")
    
    async def initialize(self):
        """Initialize all components"""
        
        logger.info("Initializing ML Enhanced Trading Bot...")
        
        try:
            # Step 1: Initialize ML risk configuration
            await self._initialize_risk_configuration()
            
            # Step 2: Initialize core trading components
            await self._initialize_core_components()
            
            # Step 3: Initialize risk management components
            await self._initialize_risk_management()
            
            # Step 4: Initialize ML execution pipeline
            await self._initialize_execution_pipeline()
            
            # Step 5: Set up integrations
            await self._setup_integrations()
            
            logger.info("ML Enhanced Trading Bot initialization complete")
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise
    
    async def _initialize_risk_configuration(self):
        """Initialize ML risk configuration"""
        
        # Create ML risk configuration manager
        self.ml_risk_config = MLRiskConfigManager(
            config_path=self.config_path,
            environment=self.environment,
            trading_mode="paper_trading"  # Start with paper trading
        )
        
        # Log configuration summary
        config = self.ml_risk_config.get_config()
        logger.info(f"ML Risk Configuration loaded:")
        logger.info(f"  - Min confidence threshold: {config.ml_risk_thresholds.min_confidence_threshold}")
        logger.info(f"  - Daily loss limit: {config.circuit_breakers.daily_loss_limit}")
        logger.info(f"  - Emergency stop at {config.emergency_stops.max_portfolio_drawdown} drawdown")
    
    async def _initialize_core_components(self):
        """Initialize core trading bot components"""
        
        # Initialize the main trading bot
        # In a real implementation, you would load the actual configuration
        self.trading_bot = TradingBot(
            paper_trade=True,  # Start with paper trading for safety
            dashboard_only=False,
            backtest_only=False
        )
        
        # Initialize ML integration controller
        self.ml_controller = MLIntegrationController(
            config_path=self.config_path
        )
        
        # Initialize Bybit client (mock for this example)
        # self.bybit_client = BybitClient(config=...)
        
        logger.info("Core components initialized")
    
    async def _initialize_risk_management(self):
        """Initialize risk management components"""
        
        # Create unified risk manager with default parameters
        risk_params = RiskParameters(
            max_portfolio_risk=Decimal('0.02'),
            max_position_size=Decimal('0.10'),
            enable_tax_optimization=True
        )
        
        self.unified_risk_manager = UnifiedRiskManager(risk_params)
        
        # Create ML risk manager with configuration
        ml_risk_params = {
            'min_confidence_threshold': self.ml_risk_config.config.ml_risk_thresholds.min_confidence_threshold,
            'max_uncertainty_threshold': self.ml_risk_config.config.ml_risk_thresholds.max_uncertainty_threshold,
            'daily_loss_limit': self.ml_risk_config.config.circuit_breakers.daily_loss_limit,
            'circuit_breaker_thresholds': {
                'daily_loss_limit': self.ml_risk_config.config.circuit_breakers.daily_loss_limit,
                'volatility_spike': self.ml_risk_config.config.circuit_breakers.volatility_spike_multiplier,
                'model_performance': self.ml_risk_config.config.circuit_breakers.model_performance_threshold
            }
        }
        
        self.ml_risk_manager = MLRiskManager(
            unified_risk_manager=self.unified_risk_manager,
            ml_risk_params=ml_risk_params
        )
        
        logger.info("Risk management components initialized")
    
    async def _initialize_execution_pipeline(self):
        """Initialize ML trade execution pipeline"""
        
        # Create execution pipeline configuration
        execution_config = {
            'execution': {
                'default_timeout_seconds': self.ml_risk_config.config.execution.default_timeout_seconds,
                'retry_attempts': self.ml_risk_config.config.execution.retry_attempts,
                'max_slippage_tolerance': self.ml_risk_config.config.execution.default_slippage_tolerance
            },
            'risk_monitoring': {
                'position_check_interval': self.ml_risk_config.config.position_monitoring.check_interval_seconds,
                'max_concurrent_positions': self.ml_risk_config.config.position_monitoring.max_concurrent_positions
            }
        }
        
        self.ml_execution_pipeline = MLTradeExecutionPipeline(
            ml_risk_manager=self.ml_risk_manager,
            unified_risk_manager=self.unified_risk_manager,
            exchange_client=self.bybit_client,
            config=execution_config
        )
        
        logger.info("ML execution pipeline initialized")
    
    async def _setup_integrations(self):
        """Set up integrations between components"""
        
        # Override the ML controller's execute_decision method to use our pipeline
        if self.ml_controller:
            self.ml_controller._execute_ml_decision = self._execute_ml_decision_with_risk_management
        
        # Set up circuit breaker callbacks
        await self._setup_circuit_breaker_callbacks()
        
        # Set up monitoring callbacks
        await self._setup_monitoring_callbacks()
        
        logger.info("Component integrations set up")
    
    async def _setup_circuit_breaker_callbacks(self):
        """Set up circuit breaker callbacks"""
        
        # Register callbacks for circuit breaker events
        # In a real implementation, these would be proper callback registrations
        
        async def on_daily_loss_limit_triggered():
            logger.critical("DAILY LOSS LIMIT CIRCUIT BREAKER TRIGGERED")
            await self._handle_daily_loss_limit_breaker()
        
        async def on_model_performance_degraded():
            logger.warning("MODEL PERFORMANCE CIRCUIT BREAKER TRIGGERED")
            await self._handle_model_performance_breaker()
        
        async def on_emergency_stop_activated():
            logger.critical("EMERGENCY STOP ACTIVATED")
            await self._handle_emergency_stop()
        
        # Store callbacks for later use
        self.circuit_breaker_callbacks = {
            'daily_loss_limit': on_daily_loss_limit_triggered,
            'model_performance': on_model_performance_degraded,
            'emergency_stop': on_emergency_stop_activated
        }
    
    async def _setup_monitoring_callbacks(self):
        """Set up monitoring and alerting callbacks"""
        
        async def on_trade_blocked(symbol: str, reasons: list):
            self.trades_blocked += 1
            logger.warning(f"Trade blocked for {symbol}: {reasons}")
            
            # Could send alerts, update dashboard, etc.
            await self._send_trade_blocked_alert(symbol, reasons)
        
        async def on_high_risk_trade(symbol: str, risk_level: str, confidence: float):
            logger.warning(f"High risk trade detected for {symbol}: "
                          f"risk={risk_level}, confidence={confidence:.1%}")
            
            # Could require manual approval, send alerts, etc.
            await self._handle_high_risk_trade(symbol, risk_level, confidence)
        
        # Store monitoring callbacks
        self.monitoring_callbacks = {
            'trade_blocked': on_trade_blocked,
            'high_risk_trade': on_high_risk_trade
        }
    
    async def _execute_ml_decision_with_risk_management(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute ML decision through the risk management pipeline
        
        This replaces the original execute_decision method in the ML controller
        """
        
        try:
            # Create ML trade request
            trade_request = MLTradeRequest(
                request_id=f"ml_trade_{datetime.now().isoformat()}_{self.trade_requests_processed}",
                symbol=decision_data['symbol'],
                side=decision_data['side'],
                signal_data=decision_data['signal_data'],
                ml_predictions=decision_data['ml_predictions'],
                market_data=decision_data['market_data'],
                priority=ExecutionPriority.NORMAL,
                expires_at=datetime.now() + timedelta(minutes=30)
            )
            
            self.trade_requests_processed += 1
            
            # Submit to execution pipeline
            request_id = await self.ml_execution_pipeline.submit_trade_request(trade_request)
            
            if not request_id:
                return {
                    'status': 'rejected',
                    'reason': 'System not operational or invalid request',
                    'request_id': None
                }
            
            # Monitor execution (in background)
            asyncio.create_task(self._monitor_trade_execution(request_id))
            
            return {
                'status': 'submitted',
                'request_id': request_id,
                'symbol': trade_request.symbol,
                'side': trade_request.side
            }
            
        except Exception as e:
            logger.error(f"Error executing ML decision: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'request_id': None
            }
    
    async def _monitor_trade_execution(self, request_id: str):
        """Monitor trade execution progress"""
        
        max_wait_time = 300  # 5 minutes
        check_interval = 10  # 10 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            # Check execution status
            status = self.ml_execution_pipeline.get_execution_status(request_id)
            
            if status:
                if status['status'] in ['executed_successfully', 'execution_failed', 'validation_failed']:
                    # Execution completed
                    await self._handle_execution_completed(request_id, status)
                    break
            
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
        
        if elapsed_time >= max_wait_time:
            logger.warning(f"Trade execution monitoring timeout for {request_id}")
    
    async def _handle_execution_completed(self, request_id: str, status: Dict[str, Any]):
        """Handle completed trade execution"""
        
        if status['status'] == 'executed_successfully':
            logger.info(f"Trade executed successfully: {request_id}")
            
            if 'result' in status:
                result = status['result']
                logger.info(f"  - Size: {result.get('executed_size', 'N/A')}")
                logger.info(f"  - Price: {result.get('executed_price', 'N/A')}")
                logger.info(f"  - Fees: {result.get('total_fees', 'N/A')}")
        
        elif status['status'] == 'validation_failed':
            logger.warning(f"Trade validation failed: {request_id}")
            
            if 'result' in status:
                result = status['result']
                blocked_reasons = result.get('blocked_reasons', [])
                logger.warning(f"  - Blocked reasons: {blocked_reasons}")
                
                # Call monitoring callback
                if 'trade_blocked' in self.monitoring_callbacks:
                    await self.monitoring_callbacks['trade_blocked'](
                        result.get('symbol', 'Unknown'), blocked_reasons
                    )
        
        elif status['status'] == 'execution_failed':
            logger.error(f"Trade execution failed: {request_id}")
            
            if 'result' in status:
                result = status['result']
                error_message = result.get('error_message', 'Unknown error')
                logger.error(f"  - Error: {error_message}")
    
    async def start_trading(self):
        """Start the trading system"""
        
        if not self.trading_bot:
            raise RuntimeError("Bot not initialized. Call initialize() first.")
        
        logger.info("Starting ML Enhanced Trading System...")
        
        self.is_running = True
        
        try:
            # Start the main trading loop
            await self._main_trading_loop()
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            await self.stop_trading()
    
    async def _main_trading_loop(self):
        """Main trading loop with ML risk management"""
        
        logger.info("Starting main trading loop...")
        
        while self.is_running:
            try:
                # Check system health
                await self._check_system_health()
                
                # Check and handle circuit breakers
                await self._check_circuit_breakers()
                
                # Process ML signals through risk management
                if self.ml_controller:
                    await self._process_ml_signals()
                
                # Update risk metrics
                await self._update_risk_metrics()
                
                # Sleep between iterations
                await asyncio.sleep(10)  # 10 second loop interval
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(30)  # Wait longer after errors
    
    async def _check_system_health(self):
        """Check overall system health"""
        
        # Check emergency stop status
        if self.ml_risk_manager.emergency_stop.is_active:
            logger.warning("Emergency stop is active - trading halted")
            return
        
        # Check critical circuit breakers
        active_breakers = [
            breaker_type for breaker_type, breaker in self.ml_risk_manager.circuit_breakers.items()
            if breaker.is_active
        ]
        
        if active_breakers:
            logger.warning(f"Active circuit breakers: {[b.value for b in active_breakers]}")
    
    async def _check_circuit_breakers(self):
        """Check and handle circuit breakers"""
        
        # Create mock market data for circuit breaker checks
        market_data = {
            'volatility': 0.025,  # Current market volatility
            'volume': 1000000,    # Current volume
            'price': 50000        # Current price
        }
        
        # Check for circuit breaker triggers
        triggered_breakers = await self.ml_risk_manager.check_and_trigger_circuit_breakers(market_data)
        
        # Handle triggered breakers
        for breaker_type in triggered_breakers:
            callback_key = breaker_type.value
            if callback_key in self.circuit_breaker_callbacks:
                await self.circuit_breaker_callbacks[callback_key]()
    
    async def _process_ml_signals(self):
        """Process ML signals through the risk management system"""
        
        # In a real implementation, this would get actual ML signals
        # For demonstration, we'll create a mock signal
        
        mock_signal = {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'signal_data': {
                'position_size': '1000',
                'signal_strength': 0.75
            },
            'ml_predictions': {
                'confidence': 0.72,
                'uncertainty': 0.25,
                'ensemble_agreement': 0.8,
                'stability': 0.7
            },
            'market_data': {
                'volatility': 0.025,
                'volume': 1000000,
                'portfolio_value': 100000,
                'returns': pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])  # Mock returns
            }
        }
        
        # Process through risk management
        result = await self._execute_ml_decision_with_risk_management(mock_signal)
        
        if result['status'] == 'rejected':
            logger.info(f"ML signal rejected: {result['reason']}")
        elif result['status'] == 'submitted':
            logger.info(f"ML signal submitted for execution: {result['request_id']}")
    
    async def _update_risk_metrics(self):
        """Update and log risk metrics"""
        
        # Get system metrics
        system_metrics = self.ml_execution_pipeline.get_system_metrics()
        risk_status = self.ml_risk_manager.get_system_status()
        
        # Log key metrics periodically
        if self.trade_requests_processed % 10 == 0:  # Every 10 trades
            logger.info("=== RISK METRICS UPDATE ===")
            logger.info(f"Trades processed: {system_metrics['execution_metrics']['total_requests']}")
            logger.info(f"Successful executions: {system_metrics['execution_metrics']['successful_executions']}")
            logger.info(f"Blocked trades: {system_metrics['execution_metrics']['blocked_trades']}")
            logger.info(f"Failed executions: {system_metrics['execution_metrics']['failed_executions']}")
            logger.info(f"Active executions: {system_metrics['active_executions']}")
            logger.info(f"Emergency stop active: {risk_status['emergency_stop']['active']}")
            logger.info("========================")
    
    async def stop_trading(self):
        """Stop the trading system"""
        
        logger.info("Stopping ML Enhanced Trading System...")
        
        self.is_running = False
        
        # Emergency halt all trading
        if self.ml_execution_pipeline:
            await self.ml_execution_pipeline.emergency_halt_all_trading(
                "System shutdown requested"
            )
        
        # Stop monitoring tasks (in a real implementation)
        # await self._stop_monitoring_tasks()
        
        logger.info("Trading system stopped")
    
    # Circuit breaker handlers
    async def _handle_daily_loss_limit_breaker(self):
        """Handle daily loss limit circuit breaker"""
        logger.critical("Daily loss limit reached - halting all trading")
        
        # Could send email alerts, update dashboard, etc.
        await self._send_emergency_alert("Daily loss limit circuit breaker triggered")
    
    async def _handle_model_performance_breaker(self):
        """Handle model performance circuit breaker"""
        logger.warning("Model performance degraded - reducing trading activity")
        
        # Could reduce position sizes, require higher confidence, etc.
        # Update configuration to be more conservative
        if self.ml_risk_config:
            current_config = self.ml_risk_config.get_config()
            current_config.ml_risk_thresholds.min_confidence_threshold = 0.8
            logger.info("Increased minimum confidence threshold to 80%")
    
    async def _handle_emergency_stop(self):
        """Handle emergency stop activation"""
        self.emergency_stops_triggered += 1
        logger.critical("EMERGENCY STOP ACTIVATED - ALL TRADING HALTED")
        
        # Send immediate alerts
        await self._send_emergency_alert("Emergency stop activated")
        
        # Stop trading loop
        self.is_running = False
    
    # Alert handlers
    async def _send_trade_blocked_alert(self, symbol: str, reasons: list):
        """Send alert for blocked trade"""
        logger.info(f"Trade blocked alert: {symbol} - {reasons}")
        # In real implementation: send email, SMS, push notification, etc.
    
    async def _handle_high_risk_trade(self, symbol: str, risk_level: str, confidence: float):
        """Handle high risk trade detection"""
        logger.warning(f"High risk trade detected: {symbol} (risk: {risk_level}, confidence: {confidence:.1%})")
        # In real implementation: require manual approval, send alerts, etc.
    
    async def _send_emergency_alert(self, message: str):
        """Send emergency alert"""
        logger.critical(f"EMERGENCY ALERT: {message}")
        # In real implementation: send immediate notifications via multiple channels
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            'system': {
                'running': self.is_running,
                'environment': self.environment,
                'trades_processed': self.trade_requests_processed,
                'trades_blocked': self.trades_blocked,
                'emergency_stops': self.emergency_stops_triggered
            }
        }
        
        if self.ml_execution_pipeline:
            status['execution'] = self.ml_execution_pipeline.get_system_metrics()
        
        if self.ml_risk_manager:
            status['risk'] = self.ml_risk_manager.get_system_status()
        
        return status

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def main():
    """Example usage of the ML Enhanced Trading Bot"""
    
    # Create and initialize the bot
    bot = MLEnhancedTradingBot(
        config_path="config/config.yaml",
        environment="development"  # Start with development environment
    )
    
    try:
        # Initialize all components
        await bot.initialize()
        
        # Start trading (would run indefinitely in real usage)
        # For this example, we'll run for a short time
        logger.info("Starting trading for demonstration...")
        
        # In a real implementation, this would be:
        # await bot.start_trading()
        
        # For demo, we'll just show the status
        status = bot.get_status()
        logger.info(f"Bot status: {status}")
        
        # Demonstrate emergency stop
        logger.info("Demonstrating emergency stop...")
        await bot.ml_risk_manager.activate_emergency_stop(
            reason="Demonstration of emergency stop functionality",
            manual_override=True,
            override_code="DEMO1234"
        )
        
        # Show status after emergency stop
        status = bot.get_status()
        logger.info(f"Status after emergency stop: {status['risk']['emergency_stop']}")
        
        # Deactivate emergency stop
        await bot.ml_risk_manager.deactivate_emergency_stop(override_code="DEMO1234")
        logger.info("Emergency stop deactivated")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    
    finally:
        # Clean shutdown
        await bot.stop_trading()

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())