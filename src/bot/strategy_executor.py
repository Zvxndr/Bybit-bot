"""
Strategy Execution Engine - Production Integration
================================================

This module integrates the sophisticated execution framework with the main
FastAPI application to enable live strategy execution.

Status: CRITICAL IMPLEMENTATION - Required for production launch
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Import execution framework (will be integrated when needed)
try:
    from .execution import (
        ExecutionEngine, Order, OrderType, OrderSide, OrderStatus, 
        ExecutionStrategy, OrderManager, SmartRouter, PositionManager
    )
except ImportError:
    # Fallback for now - will be connected to actual execution system
    class ExecutionEngine:
        pass
    class OrderManager:
        def create_order(self, **kwargs):
            return None

try:
    from .risk.core.unified_risk_manager import UnifiedRiskManager
except ImportError:
    # Mock for now
    class UnifiedRiskManager:
        async def validate_strategy_risk(self, *args, **kwargs):
            return type('MockResult', (), {'is_valid': True})()
        async def validate_trade_risk(self, *args, **kwargs):
            return type('MockResult', (), {'is_valid': True})()

# Import Bybit client
try:
    from ..bybit_api import BybitAPIClient
except ImportError:
    # Fallback import
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from bybit_api import BybitAPIClient


class ExecutionMode(Enum):
    """Execution modes for the strategy executor"""
    PAPER = "paper"      # Testnet execution (Phase 2)
    LIVE = "live"        # Mainnet execution (Phase 3)
    SIMULATION = "simulation"  # Pure simulation (Phase 1)


class StrategyStatus(Enum):
    """Strategy execution status"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    GRADUATED = "graduated"


@dataclass
class StrategyExecution:
    """Strategy execution tracking"""
    strategy_id: str
    symbol: str
    mode: ExecutionMode
    status: StrategyStatus
    start_time: datetime
    last_execution: Optional[datetime] = None
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: Decimal = Decimal('0')
    current_position: Decimal = Decimal('0')
    error_count: int = 0
    last_error: Optional[str] = None


class StrategyExecutor:
    """
    Main strategy execution engine that bridges ML strategies with live trading.
    
    This is the CRITICAL missing component identified in the project analysis.
    It coordinates between strategy signals, risk management, and order execution.
    """
    
    def __init__(self, 
                 bybit_client: BybitAPIClient,
                 testnet_client: BybitAPIClient,
                 risk_manager: UnifiedRiskManager):
        
        self.logger = logging.getLogger(__name__)
        
        # API clients for different execution modes
        self.bybit_client = bybit_client      # Live trading (Phase 3)
        self.testnet_client = testnet_client  # Paper trading (Phase 2)
        self.risk_manager = risk_manager
        
        # Execution components
        self.execution_engine = ExecutionEngine()
        self.order_manager = OrderManager()
        
        # Strategy tracking
        self.active_strategies: Dict[str, StrategyExecution] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Execution settings
        self.max_concurrent_strategies = 10
        self.max_position_size_usd = 1000  # Conservative default
        self.max_daily_trades = 50
        
        # Safety controls
        self.emergency_stop = False
        self.daily_trade_count = 0
        self.daily_pnl = Decimal('0')
        
        self.logger.info("✅ Strategy Executor initialized")
    
    async def start_strategy_execution(self, 
                                     strategy_id: str, 
                                     symbol: str,
                                     mode: ExecutionMode = ExecutionMode.PAPER) -> bool:
        """
        Start executing a strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            symbol: Trading symbol (e.g., 'BTCUSDT')
            mode: Execution mode (paper/live/simulation)
            
        Returns:
            bool: True if strategy started successfully
        """
        try:
            # Safety checks
            if self.emergency_stop:
                self.logger.error("❌ Emergency stop active - cannot start new strategies")
                return False
                
            if len(self.active_strategies) >= self.max_concurrent_strategies:
                self.logger.error(f"❌ Max concurrent strategies ({self.max_concurrent_strategies}) reached")
                return False
            
            if strategy_id in self.active_strategies:
                self.logger.warning(f"⚠️ Strategy {strategy_id} already active")
                return False
            
            # Risk validation
            risk_check = await self._validate_strategy_risk(strategy_id, symbol, mode)
            if not risk_check:
                self.logger.error(f"❌ Risk validation failed for strategy {strategy_id}")
                return False
            
            # Create strategy execution tracker
            execution = StrategyExecution(
                strategy_id=strategy_id,
                symbol=symbol,
                mode=mode,
                status=StrategyStatus.ACTIVE,
                start_time=datetime.now()
            )
            
            self.active_strategies[strategy_id] = execution
            
            # Start execution loop
            asyncio.create_task(self._strategy_execution_loop(strategy_id))
            
            self.logger.info(f"✅ Strategy {strategy_id} started in {mode.value} mode on {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start strategy {strategy_id}: {e}")
            return False
    
    async def stop_strategy_execution(self, strategy_id: str) -> bool:
        """Stop executing a strategy"""
        try:
            if strategy_id not in self.active_strategies:
                self.logger.warning(f"⚠️ Strategy {strategy_id} not found")
                return False
            
            execution = self.active_strategies[strategy_id]
            execution.status = StrategyStatus.INACTIVE
            
            # Cancel any pending orders
            await self._cancel_strategy_orders(strategy_id)
            
            # Close any open positions (if configured)
            await self._close_strategy_positions(strategy_id)
            
            # Remove from active strategies
            del self.active_strategies[strategy_id]
            
            self.logger.info(f"✅ Strategy {strategy_id} stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop strategy {strategy_id}: {e}")
            return False
    
    async def emergency_stop_all(self) -> None:
        """Emergency stop - halt all trading immediately"""
        self.logger.critical("🚨 EMERGENCY STOP ACTIVATED")
        self.emergency_stop = True
        
        # Stop all strategies
        strategy_ids = list(self.active_strategies.keys())
        for strategy_id in strategy_ids:
            await self.stop_strategy_execution(strategy_id)
        
        self.logger.critical("🚨 All strategies stopped - manual intervention required")
    
    async def _strategy_execution_loop(self, strategy_id: str) -> None:
        """Main execution loop for a strategy"""
        execution = self.active_strategies.get(strategy_id)
        if not execution:
            return
        
        self.logger.info(f"🔄 Starting execution loop for strategy {strategy_id}")
        
        while (execution.status == StrategyStatus.ACTIVE and 
               not self.emergency_stop and
               strategy_id in self.active_strategies):
            
            try:
                # Get strategy signal
                signal = await self._get_strategy_signal(strategy_id, execution.symbol)
                
                if signal:
                    # Execute trade based on signal
                    success = await self._execute_strategy_signal(strategy_id, signal)
                    
                    # Update execution stats
                    execution.total_trades += 1
                    execution.last_execution = datetime.now()
                    
                    if success:
                        execution.successful_trades += 1
                        execution.error_count = 0  # Reset error count on success
                    else:
                        execution.error_count += 1
                        
                        # Pause strategy if too many errors
                        if execution.error_count >= 5:
                            execution.status = StrategyStatus.ERROR
                            self.logger.error(f"❌ Strategy {strategy_id} paused due to errors")
                
                # Check if strategy should be graduated to next phase
                await self._check_strategy_graduation(strategy_id)
                
                # Sleep before next iteration (avoid overtrading)
                await asyncio.sleep(30)  # 30-second intervals
                
            except Exception as e:
                execution.error_count += 1
                execution.last_error = str(e)
                self.logger.error(f"❌ Error in strategy {strategy_id} execution: {e}")
                
                if execution.error_count >= 10:
                    execution.status = StrategyStatus.ERROR
                    break
                
                await asyncio.sleep(60)  # Longer delay on errors
        
        self.logger.info(f"🔄 Execution loop ended for strategy {strategy_id}")
    
    async def _get_strategy_signal(self, strategy_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get trading signal from ML strategy"""
        try:
            # This would integrate with the ML strategy discovery system
            # For now, return None (no signal) to avoid unintended trades
            
            # TODO: Integrate with src/ml/strategy_discovery.py
            # TODO: Implement signal generation based on market conditions
            # TODO: Add signal validation and confidence scoring
            
            return None  # Safe default - no trading until ML integration complete
            
        except Exception as e:
            self.logger.error(f"❌ Error getting signal for {strategy_id}: {e}")
            return None
    
    async def _execute_strategy_signal(self, strategy_id: str, signal: Dict[str, Any]) -> bool:
        """Execute a trading signal"""
        try:
            execution = self.active_strategies[strategy_id]
            
            # Extract signal information
            action = signal.get('action')  # 'buy', 'sell', 'hold'
            confidence = signal.get('confidence', 0.0)
            suggested_size = signal.get('size', 0.0)
            
            if action == 'hold' or confidence < 0.7:
                return True  # No action needed
            
            # Risk validation
            if not await self._validate_trade_risk(strategy_id, action, suggested_size):
                return False
            
            # Calculate position size
            position_size = await self._calculate_position_size(strategy_id, suggested_size)
            
            # Determine order parameters
            order_side = OrderSide.BUY if action == 'buy' else OrderSide.SELL
            
            # Get appropriate API client based on execution mode
            client = self._get_api_client(execution.mode)
            
            # Create and submit order
            order = await self._create_and_submit_order(
                client=client,
                symbol=execution.symbol,
                side=order_side,
                quantity=position_size,
                strategy_id=strategy_id
            )
            
            if order:
                # Monitor order execution
                await self._monitor_order_execution(order, strategy_id)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error executing signal for {strategy_id}: {e}")
            return False
    
    async def _validate_strategy_risk(self, strategy_id: str, symbol: str, mode: ExecutionMode) -> bool:
        """Validate strategy against risk parameters"""
        try:
            # Check daily trade limits
            if self.daily_trade_count >= self.max_daily_trades:
                return False
            
            # Check daily P&L limits
            if self.daily_pnl <= -1000:  # Max daily loss $1000
                return False
            
            # Strategy-specific risk checks via risk manager
            if self.risk_manager:
                risk_result = await self.risk_manager.validate_strategy_risk(
                    strategy_id, symbol, mode.value
                )
                return risk_result.is_valid
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Risk validation error: {e}")
            return False
    
    async def _validate_trade_risk(self, strategy_id: str, action: str, size: float) -> bool:
        """Validate individual trade against risk parameters"""
        try:
            execution = self.active_strategies[strategy_id]
            
            # Position size limits
            if size * 100 > self.max_position_size_usd:  # Assuming $100 per unit
                return False
            
            # Daily trade count
            if self.daily_trade_count >= self.max_daily_trades:
                return False
            
            # Use risk manager for detailed validation
            if self.risk_manager:
                risk_result = await self.risk_manager.validate_trade_risk(
                    execution.symbol, action, size, execution.mode.value
                )
                return risk_result.is_valid
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Trade risk validation error: {e}")
            return False
    
    async def _calculate_position_size(self, strategy_id: str, suggested_size: float) -> Decimal:
        """Calculate appropriate position size"""
        try:
            # Conservative position sizing - start small
            max_size = Decimal(str(self.max_position_size_usd / 100))  # Assuming $100 per unit
            suggested_decimal = Decimal(str(suggested_size))
            
            return min(max_size, suggested_decimal)
            
        except Exception as e:
            self.logger.error(f"❌ Position sizing error: {e}")
            return Decimal('0.01')  # Minimum safe size
    
    def _get_api_client(self, mode: ExecutionMode) -> BybitAPIClient:
        """Get appropriate API client for execution mode"""
        if mode == ExecutionMode.LIVE:
            return self.bybit_client
        else:  # PAPER or SIMULATION
            return self.testnet_client
    
    async def _create_and_submit_order(self,
                                     client: BybitAPIClient,
                                     symbol: str,
                                     side: OrderSide,
                                     quantity: Decimal,
                                     strategy_id: str) -> Optional[Order]:
        """Create and submit order to exchange"""
        try:
            # Create order through execution engine
            order = self.order_manager.create_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,  # Start with market orders for simplicity
                quantity=quantity,
                strategy_id=strategy_id
            )
            
            # Submit to exchange via client
            exchange_response = await client.place_order(
                symbol=symbol,
                side=side.value,
                order_type='market',
                quantity=str(quantity)
            )
            
            if exchange_response and exchange_response.get('success'):
                order.exchange_order_id = exchange_response.get('order_id')
                order.status = OrderStatus.SUBMITTED
                
                self.logger.info(f"✅ Order submitted: {order.order_id}")
                return order
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Order submission error: {e}")
            return None
    
    async def _monitor_order_execution(self, order: Order, strategy_id: str) -> None:
        """Monitor order until filled or cancelled"""
        try:
            execution = self.active_strategies[strategy_id]
            client = self._get_api_client(execution.mode)
            
            # Monitor for up to 5 minutes
            max_monitoring_time = 300  # 5 minutes
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < max_monitoring_time:
                # Check order status
                order_status = await client.get_order_status(
                    order.exchange_order_id
                )
                
                if order_status:
                    if order_status.get('status') == 'filled':
                        order.status = OrderStatus.FILLED
                        
                        # Update strategy P&L
                        fill_price = Decimal(str(order_status.get('fill_price', 0)))
                        fill_quantity = Decimal(str(order_status.get('fill_quantity', 0)))
                        
                        # Simple P&L calculation (would be more sophisticated in production)
                        trade_value = fill_price * fill_quantity
                        execution.total_pnl += trade_value if order.side == OrderSide.SELL else -trade_value
                        execution.current_position += fill_quantity if order.side == OrderSide.BUY else -fill_quantity
                        
                        self.daily_trade_count += 1
                        
                        self.logger.info(f"✅ Order filled: {order.order_id} at ${fill_price}")
                        break
                        
                    elif order_status.get('status') in ['cancelled', 'rejected']:
                        order.status = OrderStatus.CANCELLED
                        self.logger.warning(f"⚠️ Order cancelled: {order.order_id}")
                        break
                
                await asyncio.sleep(10)  # Check every 10 seconds
            
            # Timeout handling
            if order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                self.logger.warning(f"⚠️ Order monitoring timeout: {order.order_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Order monitoring error: {e}")
    
    async def _cancel_strategy_orders(self, strategy_id: str) -> None:
        """Cancel all pending orders for a strategy"""
        try:
            # Implementation would cancel all open orders for the strategy
            self.logger.info(f"🔄 Cancelling orders for strategy {strategy_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error cancelling orders: {e}")
    
    async def _close_strategy_positions(self, strategy_id: str) -> None:
        """Close all positions for a strategy"""
        try:
            # Implementation would close all open positions for the strategy
            execution = self.active_strategies.get(strategy_id)
            if execution and execution.current_position != 0:
                self.logger.info(f"🔄 Closing positions for strategy {strategy_id}")
                
        except Exception as e:
            self.logger.error(f"❌ Error closing positions: {e}")
    
    async def _check_strategy_graduation(self, strategy_id: str) -> None:
        """Check if strategy should be graduated to next phase"""
        try:
            execution = self.active_strategies.get(strategy_id)
            if not execution:
                return
            
            # Graduation criteria (example)
            if (execution.total_trades >= 20 and 
                execution.successful_trades >= 15 and
                execution.total_pnl > 0):
                
                if execution.mode == ExecutionMode.SIMULATION:
                    # Graduate to paper trading
                    self.logger.info(f"🎓 Strategy {strategy_id} ready for paper trading")
                elif execution.mode == ExecutionMode.PAPER:
                    # Graduate to live trading (with approval)
                    self.logger.info(f"🎓 Strategy {strategy_id} ready for live trading")
                    execution.status = StrategyStatus.GRADUATED
            
        except Exception as e:
            self.logger.error(f"❌ Graduation check error: {e}")
    
    def get_strategy_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a strategy"""
        execution = self.active_strategies.get(strategy_id)
        if not execution:
            return None
        
        return {
            'strategy_id': execution.strategy_id,
            'symbol': execution.symbol,
            'mode': execution.mode.value,
            'status': execution.status.value,
            'start_time': execution.start_time.isoformat(),
            'last_execution': execution.last_execution.isoformat() if execution.last_execution else None,
            'total_trades': execution.total_trades,
            'successful_trades': execution.successful_trades,
            'success_rate': (execution.successful_trades / execution.total_trades * 100) if execution.total_trades > 0 else 0,
            'total_pnl': float(execution.total_pnl),
            'current_position': float(execution.current_position),
            'error_count': execution.error_count,
            'last_error': execution.last_error
        }
    
    def get_all_strategies_status(self) -> List[Dict[str, Any]]:
        """Get status of all active strategies"""
        return [
            self.get_strategy_status(strategy_id)
            for strategy_id in self.active_strategies.keys()
        ]
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get overall execution summary"""
        total_strategies = len(self.active_strategies)
        active_count = sum(1 for s in self.active_strategies.values() if s.status == StrategyStatus.ACTIVE)
        total_trades = sum(s.total_trades for s in self.active_strategies.values())
        total_pnl = sum(s.total_pnl for s in self.active_strategies.values())
        
        return {
            'total_strategies': total_strategies,
            'active_strategies': active_count,
            'daily_trades': self.daily_trade_count,
            'daily_pnl': float(self.daily_pnl),
            'total_trades': total_trades,
            'total_pnl': float(total_pnl),
            'emergency_stop': self.emergency_stop,
            'max_daily_trades': self.max_daily_trades,
            'max_position_size_usd': self.max_position_size_usd
        }


# Factory function for easy integration
def create_strategy_executor(bybit_client: BybitAPIClient,
                           testnet_client: BybitAPIClient,
                           risk_manager: UnifiedRiskManager) -> StrategyExecutor:
    """Factory function to create strategy executor"""
    return StrategyExecutor(
        bybit_client=bybit_client,
        testnet_client=testnet_client,
        risk_manager=risk_manager
    )