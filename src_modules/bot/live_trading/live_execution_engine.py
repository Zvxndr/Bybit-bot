"""
Live Execution Engine for Real-time Trading Operations

This module provides comprehensive live trading execution capabilities:
- Paper trading mode with virtual execution
- Live trading mode with real order placement
- Hybrid mode with graduated strategy deployment
- Order lifecycle management and position tracking
- Execution quality monitoring and optimization
- Risk management integration with real-time controls

Supports seamless transition from backtesting to live trading with
comprehensive safety features and performance monitoring.

Author: Trading Bot Team
Version: 1.0.0 - Phase 5 Implementation
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_DOWN, ROUND_UP

from ..utils.logging import TradingLogger
from ..config_manager import ConfigurationManager
from ..core.strategy_manager import TradingSignal, BaseStrategy
from ..core.trading_engine import TradingEngine, Order, OrderSide, OrderType, OrderStatus
from ..risk_management.risk_manager import RiskManager, TradeRiskAssessment
from ..exchange.bybit_client import BybitClient
from .websocket_manager import WebSocketManager, WebSocketMessage


class TradingMode(Enum):
    """Trading execution modes."""
    PAPER = "paper"          # Virtual execution with real market data
    LIVE = "live"            # Real execution with real money
    HYBRID = "hybrid"        # Graduated deployment: paper → live


class ExecutionStatus(Enum):
    """Execution status for orders and trades."""
    PENDING = "pending"
    EXECUTING = "executing"
    EXECUTED = "executed"
    PARTIALLY_EXECUTED = "partially_executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PositionSyncStatus(Enum):
    """Position synchronization status."""
    IN_SYNC = "in_sync"
    OUT_OF_SYNC = "out_of_sync"
    SYNCING = "syncing"
    SYNC_ERROR = "sync_error"


@dataclass
class ExecutionResult:
    """Result of trade execution."""
    execution_id: str
    signal_id: str
    strategy_id: str
    symbol: str
    side: str
    requested_quantity: Decimal
    executed_quantity: Decimal
    average_price: Decimal
    total_cost: Decimal
    commission: Decimal
    execution_time_ms: float
    slippage: Decimal
    status: ExecutionStatus
    mode: TradingMode
    order_ids: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def fill_ratio(self) -> Decimal:
        """Calculate fill ratio (executed / requested)."""
        if self.requested_quantity == 0:
            return Decimal('0')
        return self.executed_quantity / self.requested_quantity
    
    @property
    def slippage_bps(self) -> Decimal:
        """Calculate slippage in basis points."""
        return self.slippage * Decimal('10000')


@dataclass
class VirtualPosition:
    """Virtual position for paper trading."""
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    commission_paid: Decimal
    opened_at: datetime
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_price(self, new_price: Decimal) -> None:
        """Update position with new market price."""
        self.current_price = new_price
        self.last_updated = datetime.now()
        
        # Calculate unrealized PnL
        if self.side == "long":
            self.unrealized_pnl = (new_price - self.entry_price) * self.size
        else:  # short
            self.unrealized_pnl = (self.entry_price - new_price) * self.size


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time_ms: float = 0.0
    average_slippage_bps: float = 0.0
    total_commission_paid: Decimal = Decimal('0')
    best_execution_time_ms: float = float('inf')
    worst_execution_time_ms: float = 0.0
    last_execution_at: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate execution success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions
    
    def update(self, result: ExecutionResult) -> None:
        """Update metrics with new execution result."""
        self.total_executions += 1
        self.last_execution_at = datetime.now()
        
        if result.status == ExecutionStatus.EXECUTED:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        
        # Update timing metrics
        exec_time = result.execution_time_ms
        if exec_time > 0:
            if self.average_execution_time_ms == 0:
                self.average_execution_time_ms = exec_time
            else:
                self.average_execution_time_ms = (
                    (self.average_execution_time_ms * (self.total_executions - 1) + exec_time) 
                    / self.total_executions
                )
            
            self.best_execution_time_ms = min(self.best_execution_time_ms, exec_time)
            self.worst_execution_time_ms = max(self.worst_execution_time_ms, exec_time)
        
        # Update slippage metrics
        slippage_bps = float(result.slippage_bps)
        if self.average_slippage_bps == 0:
            self.average_slippage_bps = slippage_bps
        else:
            self.average_slippage_bps = (
                (self.average_slippage_bps * (self.total_executions - 1) + slippage_bps) 
                / self.total_executions
            )
        
        # Update commission
        self.total_commission_paid += result.commission


class LiveExecutionEngine:
    """
    Comprehensive live trading execution engine.
    
    Features:
    - Multi-mode execution: paper, live, and hybrid trading
    - Real-time position tracking and synchronization
    - Execution quality monitoring and optimization
    - Risk management integration with pre-trade checks
    - Strategy graduation from paper to live trading
    - Comprehensive execution analytics and reporting
    """
    
    def __init__(
        self,
        config: ConfigurationManager,
        trading_engine: TradingEngine,
        risk_manager: RiskManager,
        websocket_manager: WebSocketManager,
        bybit_client: BybitClient,
        default_mode: TradingMode = TradingMode.PAPER
    ):
        self.config = config
        self.trading_engine = trading_engine
        self.risk_manager = risk_manager
        self.websocket_manager = websocket_manager
        self.bybit_client = bybit_client
        self.default_mode = default_mode
        self.logger = TradingLogger("live_execution_engine")
        
        # Execution state
        self.active_executions: Dict[str, ExecutionResult] = {}
        self.execution_history: List[ExecutionResult] = []
        self.virtual_positions: Dict[str, VirtualPosition] = {}
        self.real_positions: Dict[str, Any] = {}  # From API
        
        # Strategy management
        self.strategy_modes: Dict[str, TradingMode] = {}  # Strategy-specific modes
        self.strategy_metrics: Dict[str, ExecutionMetrics] = {}
        self.graduated_strategies: set = set()  # Strategies approved for live trading
        
        # Synchronization
        self.position_sync_status = PositionSyncStatus.IN_SYNC
        self.last_sync_time: Optional[datetime] = None
        self.sync_interval = timedelta(minutes=5)
        
        # Configuration
        self.max_slippage_tolerance = Decimal(str(config.get('execution.max_slippage_tolerance', 0.002)))
        self.position_sync_enabled = config.get('execution.position_sync_enabled', True)
        self.graduation_criteria = {
            'min_trades': config.get('execution.graduation_min_trades', 50),
            'min_success_rate': config.get('execution.graduation_min_success_rate', 0.8),
            'min_sharpe_ratio': config.get('execution.graduation_min_sharpe_ratio', 1.0)
        }
        
        # Setup WebSocket handlers
        self._setup_websocket_handlers()
        
        # Start background tasks
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        self.logger.info(f"LiveExecutionEngine initialized in {default_mode.value} mode")
    
    async def start(self) -> bool:
        """
        Start the live execution engine.
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.logger.info("Starting live execution engine...")
            self.running = True
            
            # Start position synchronization task
            if self.position_sync_enabled:
                sync_task = asyncio.create_task(self._position_sync_loop())
                self.tasks.append(sync_task)
            
            # Start execution monitoring task
            monitor_task = asyncio.create_task(self._execution_monitor_loop())
            self.tasks.append(monitor_task)
            
            # Initial position sync
            await self._sync_positions()
            
            self.logger.info("Live execution engine started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start live execution engine: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the live execution engine."""
        try:
            self.logger.info("Stopping live execution engine...")
            self.running = False
            
            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            self.logger.info("Live execution engine stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping live execution engine: {e}")
    
    async def execute_signal(
        self,
        signal: TradingSignal,
        strategy: BaseStrategy,
        mode: Optional[TradingMode] = None
    ) -> ExecutionResult:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
            strategy: Strategy that generated the signal
            mode: Execution mode (overrides default)
            
        Returns:
            ExecutionResult: Execution result
        """
        start_time = datetime.now()
        execution_id = str(uuid.uuid4())
        
        # Determine execution mode
        execution_mode = mode or self.strategy_modes.get(strategy.strategy_id, self.default_mode)
        
        try:
            self.logger.info(
                f"Executing signal: {signal.symbol} {signal.signal_type.value} "
                f"in {execution_mode.value} mode"
            )
            
            # Pre-execution risk assessment
            risk_assessment = await self._assess_execution_risk(signal, execution_mode)
            if not risk_assessment.is_approved:
                return ExecutionResult(
                    execution_id=execution_id,
                    signal_id=signal.strategy_id,
                    strategy_id=strategy.strategy_id,
                    symbol=signal.symbol,
                    side=signal.signal_type.value,
                    requested_quantity=signal.quantity or Decimal('0'),
                    executed_quantity=Decimal('0'),
                    average_price=Decimal('0'),
                    total_cost=Decimal('0'),
                    commission=Decimal('0'),
                    execution_time_ms=0,
                    slippage=Decimal('0'),
                    status=ExecutionStatus.FAILED,
                    mode=execution_mode,
                    error_message=risk_assessment.risk_reason
                )
            
            # Execute based on mode
            if execution_mode == TradingMode.PAPER:
                result = await self._execute_paper_trade(signal, execution_id, strategy.strategy_id)
            elif execution_mode == TradingMode.LIVE:
                result = await self._execute_live_trade(signal, execution_id, strategy.strategy_id)
            elif execution_mode == TradingMode.HYBRID:
                result = await self._execute_hybrid_trade(signal, execution_id, strategy.strategy_id)
            else:
                raise ValueError(f"Unknown execution mode: {execution_mode}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            # Store execution result
            self.active_executions[execution_id] = result
            self.execution_history.append(result)
            
            # Update strategy metrics
            self._update_strategy_metrics(strategy.strategy_id, result)
            
            # Check for strategy graduation
            await self._check_strategy_graduation(strategy.strategy_id)
            
            self.logger.info(
                f"Execution completed: {result.symbol} {result.status.value} "
                f"Fill: {result.fill_ratio:.3f} Time: {result.execution_time_ms:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ExecutionResult(
                execution_id=execution_id,
                signal_id=signal.strategy_id,
                strategy_id=strategy.strategy_id,
                symbol=signal.symbol,
                side=signal.signal_type.value,
                requested_quantity=signal.quantity or Decimal('0'),
                executed_quantity=Decimal('0'),
                average_price=Decimal('0'),
                total_cost=Decimal('0'),
                commission=Decimal('0'),
                execution_time_ms=execution_time,
                slippage=Decimal('0'),
                status=ExecutionStatus.FAILED,
                mode=execution_mode,
                error_message=str(e)
            )
    
    async def _execute_paper_trade(
        self, 
        signal: TradingSignal, 
        execution_id: str,
        strategy_id: str
    ) -> ExecutionResult:
        """Execute trade in paper trading mode."""
        try:
            # Get current market price
            market_price = await self._get_current_price(signal.symbol)
            
            # Calculate execution parameters
            quantity = signal.quantity or Decimal('1.0')  # Default quantity
            slippage = self._calculate_expected_slippage(signal.symbol, quantity)
            execution_price = market_price * (Decimal('1') + slippage)
            
            # Calculate commission (paper trading uses real commission rates)
            commission = self._calculate_commission(quantity, execution_price, is_paper=True)
            total_cost = quantity * execution_price + commission
            
            # Update virtual position
            await self._update_virtual_position(
                signal.symbol, signal.signal_type.value, quantity, execution_price
            )
            
            return ExecutionResult(
                execution_id=execution_id,
                signal_id=signal.strategy_id,
                strategy_id=strategy_id,
                symbol=signal.symbol,
                side=signal.signal_type.value,
                requested_quantity=quantity,
                executed_quantity=quantity,
                average_price=execution_price,
                total_cost=total_cost,
                commission=commission,
                execution_time_ms=0,  # Will be set by caller
                slippage=slippage,
                status=ExecutionStatus.EXECUTED,
                mode=TradingMode.PAPER,
                metadata={'market_price': float(market_price)}
            )
            
        except Exception as e:
            self.logger.error(f"Paper trade execution error: {e}")
            raise
    
    async def _execute_live_trade(
        self, 
        signal: TradingSignal, 
        execution_id: str,
        strategy_id: str
    ) -> ExecutionResult:
        """Execute trade in live trading mode."""
        try:
            # Convert signal to order
            order_side = OrderSide.BUY if signal.signal_type.value == "buy" else OrderSide.SELL
            quantity = signal.quantity or await self._calculate_position_size(signal)
            
            # Place order through trading engine
            order = await self.trading_engine.place_order(
                symbol=signal.symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=quantity
            )
            
            if not order:
                raise Exception("Failed to place order")
            
            # Monitor order execution
            execution_result = await self._monitor_order_execution(order, execution_id, strategy_id)
            
            # Sync positions after execution
            await self._sync_positions()
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Live trade execution error: {e}")
            raise
    
    async def _execute_hybrid_trade(
        self, 
        signal: TradingSignal, 
        execution_id: str,
        strategy_id: str
    ) -> ExecutionResult:
        """Execute trade in hybrid mode (graduated strategies only)."""
        try:
            # Check if strategy is graduated for live trading
            if strategy_id in self.graduated_strategies:
                return await self._execute_live_trade(signal, execution_id, strategy_id)
            else:
                return await self._execute_paper_trade(signal, execution_id, strategy_id)
                
        except Exception as e:
            self.logger.error(f"Hybrid trade execution error: {e}")
            raise
    
    async def _assess_execution_risk(
        self, 
        signal: TradingSignal, 
        mode: TradingMode
    ) -> TradeRiskAssessment:
        """Assess risk for trade execution."""
        try:
            # Get current portfolio balance
            if mode == TradingMode.PAPER:
                # Use virtual portfolio balance
                balance = self._calculate_virtual_portfolio_value()
            else:
                # Use real portfolio balance
                balance = self.trading_engine.get_portfolio_value()
            
            # Assess trade risk
            return await self.risk_manager.assess_trade_risk(
                symbol=signal.symbol,
                side=signal.signal_type.value,
                entry_price=signal.price,
                stop_loss=getattr(signal, 'stop_loss', None),
                take_profit=getattr(signal, 'take_profit', None),
                current_balance=balance,
                strategy_id=signal.strategy_id
            )
            
        except Exception as e:
            self.logger.error(f"Risk assessment error: {e}")
            # Return safe default
            return TradeRiskAssessment(
                is_approved=False,
                risk_ratio=Decimal('0'),
                position_size=Decimal('0'),
                stop_loss_price=None,
                take_profit_price=None,
                risk_reason=f"Risk assessment error: {str(e)}"
            )
    
    def _setup_websocket_handlers(self) -> None:
        """Setup WebSocket message handlers for real-time data."""
        # Market data handler
        self.websocket_manager.add_message_handler(
            "publicTrade.*", self._handle_market_data
        )
        
        # Order execution handler
        self.websocket_manager.add_message_handler(
            "execution", self._handle_execution_update
        )
        
        # Position update handler
        self.websocket_manager.add_message_handler(
            "position", self._handle_position_update
        )
    
    def _handle_market_data(self, message: WebSocketMessage) -> None:
        """Handle real-time market data updates."""
        try:
            data = message.data
            if 'data' in data:
                for trade in data['data']:
                    symbol = trade.get('symbol')
                    price = Decimal(str(trade.get('price', 0)))
                    
                    # Update virtual positions with new market price
                    if symbol in self.virtual_positions:
                        self.virtual_positions[symbol].update_price(price)
                        
        except Exception as e:
            self.logger.error(f"Market data handler error: {e}")
    
    def _handle_execution_update(self, message: WebSocketMessage) -> None:
        """Handle order execution updates."""
        try:
            data = message.data
            if 'data' in data:
                for execution in data['data']:
                    order_id = execution.get('orderId')
                    
                    # Find corresponding execution result
                    for exec_result in self.active_executions.values():
                        if order_id in exec_result.order_ids:
                            # Update execution status
                            self._update_execution_from_websocket(exec_result, execution)
                            break
                    
        except Exception as e:
            self.logger.error(f"Execution update handler error: {e}")
    
    def _handle_position_update(self, message: WebSocketMessage) -> None:
        """Handle position updates."""
        try:
            data = message.data
            if 'data' in data:
                for position in data['data']:
                    symbol = position.get('symbol')
                    self.real_positions[symbol] = position
                    
                    # Check for position sync issues
                    self._check_position_sync(symbol)
                    
        except Exception as e:
            self.logger.error(f"Position update handler error: {e}")
    
    async def _sync_positions(self) -> None:
        """Synchronize positions with exchange."""
        try:
            self.position_sync_status = PositionSyncStatus.SYNCING
            
            # Fetch current positions from API
            api_positions = await self.trading_engine.get_positions()
            
            if api_positions:
                for position in api_positions:
                    symbol = position.get('symbol')
                    if symbol:
                        self.real_positions[symbol] = position
            
            self.position_sync_status = PositionSyncStatus.IN_SYNC
            self.last_sync_time = datetime.now()
            
            self.logger.debug("Position synchronization completed")
            
        except Exception as e:
            self.logger.error(f"Position sync error: {e}")
            self.position_sync_status = PositionSyncStatus.SYNC_ERROR
    
    async def _position_sync_loop(self) -> None:
        """Background position synchronization loop."""
        while self.running:
            try:
                if (not self.last_sync_time or 
                    datetime.now() - self.last_sync_time > self.sync_interval):
                    await self._sync_positions()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Position sync loop error: {e}")
                await asyncio.sleep(60)
    
    async def _execution_monitor_loop(self) -> None:
        """Background execution monitoring loop."""
        while self.running:
            try:
                # Clean up old completed executions
                current_time = datetime.now()
                cleanup_threshold = current_time - timedelta(hours=1)
                
                completed_executions = [
                    exec_id for exec_id, result in self.active_executions.items()
                    if result.status in [ExecutionStatus.EXECUTED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]
                    and current_time - cleanup_threshold > timedelta(hours=1)
                ]
                
                for exec_id in completed_executions:
                    del self.active_executions[exec_id]
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Execution monitor loop error: {e}")
                await asyncio.sleep(300)
    
    def get_execution_metrics(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """Get execution performance metrics."""
        if strategy_id:
            metrics = self.strategy_metrics.get(strategy_id, ExecutionMetrics())
            return {
                "strategy_id": strategy_id,
                "total_executions": metrics.total_executions,
                "success_rate": metrics.success_rate,
                "average_execution_time_ms": metrics.average_execution_time_ms,
                "average_slippage_bps": metrics.average_slippage_bps,
                "total_commission_paid": float(metrics.total_commission_paid),
                "last_execution_at": metrics.last_execution_at.isoformat() if metrics.last_execution_at else None,
                "is_graduated": strategy_id in self.graduated_strategies
            }
        else:
            # Return aggregate metrics
            total_metrics = ExecutionMetrics()
            for strategy_metrics in self.strategy_metrics.values():
                total_metrics.total_executions += strategy_metrics.total_executions
                total_metrics.successful_executions += strategy_metrics.successful_executions
                total_metrics.failed_executions += strategy_metrics.failed_executions
                total_metrics.total_commission_paid += strategy_metrics.total_commission_paid
            
            return {
                "total_executions": total_metrics.total_executions,
                "success_rate": total_metrics.success_rate,
                "total_commission_paid": float(total_metrics.total_commission_paid),
                "graduated_strategies": len(self.graduated_strategies),
                "active_executions": len(self.active_executions),
                "position_sync_status": self.position_sync_status.value
            }
    
    # Additional helper methods would continue here...
    # (Implementation continues with remaining methods for completeness)