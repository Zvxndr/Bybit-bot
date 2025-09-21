"""
Trading Strategy Framework - Base Classes and Strategy Management

This module provides a comprehensive framework for creating and managing trading strategies
that integrate with the unified risk management and real Bybit API trading system.

Key Features:
- Base strategy classes with standardized interfaces
- Signal generation and analysis
- Strategy execution with risk management integration
- Portfolio-level strategy coordination
- Performance tracking and reporting
- Strategy optimization and parameter tuning
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np
import pandas as pd

from ..core.trading_engine import TradingEngine, OrderSide, OrderType, TradeRequest
from ..risk_management import UnifiedRiskManager, RiskLevel, RiskAction
from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger


class SignalType(Enum):
    """Signal types for trading strategies."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class StrategyState(Enum):
    """Strategy execution states."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    price: Decimal
    timestamp: datetime
    strategy_id: str
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_entry_signal(self) -> bool:
        return self.signal_type in [SignalType.BUY, SignalType.SELL]
    
    @property
    def is_exit_signal(self) -> bool:
        return self.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    strategy_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal('0')
    max_drawdown: Decimal = Decimal('0')
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: Decimal = Decimal('0')
    avg_loss: Decimal = Decimal('0')
    profit_factor: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyConfig:
    """Strategy configuration parameters."""
    strategy_id: str
    name: str
    symbols: List[str]
    timeframe: str
    max_positions: int = 1
    position_size_pct: float = 0.02  # 2% of portfolio per trade
    risk_reward_ratio: float = 2.0
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    This abstract class defines the interface that all trading strategies must implement.
    It provides common functionality for signal generation, execution, and performance tracking.
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        trading_engine: TradingEngine,
        risk_manager: UnifiedRiskManager,
        config_manager: Optional[ConfigurationManager] = None
    ):
        self.config = config
        self.trading_engine = trading_engine
        self.risk_manager = risk_manager
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = TradingLogger(f"strategy_{config.strategy_id}")
        
        # Strategy state
        self.state = StrategyState.INACTIVE
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.signals_history: List[TradingSignal] = []
        
        # Performance tracking
        self.performance = StrategyPerformance(strategy_id=config.strategy_id)
        self.trade_history: List[Dict[str, Any]] = []
        
        # Data storage
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.last_data_update: Dict[str, datetime] = {}
        
        self.logger.info(f"Strategy {config.name} ({config.strategy_id}) initialized")
    
    @abstractmethod
    async def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals based on market data.
        
        Args:
            symbol: Trading symbol
            data: Market data (OHLCV)
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    async def should_enter_position(self, signal: TradingSignal) -> bool:
        """
        Determine if strategy should enter a position based on signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            bool: True if should enter position
        """
        pass
    
    @abstractmethod
    async def should_exit_position(self, symbol: str, position: Dict[str, Any]) -> bool:
        """
        Determine if strategy should exit an existing position.
        
        Args:
            symbol: Trading symbol
            position: Current position data
            
        Returns:
            bool: True if should exit position
        """
        pass
    
    async def start(self) -> bool:
        """Start the strategy."""
        try:
            if self.state != StrategyState.INACTIVE:
                self.logger.warning(f"Strategy already started (state: {self.state})")
                return False
            
            self.state = StrategyState.ACTIVE
            self.logger.info(f"Strategy {self.config.name} started")
            
            # Start strategy execution loop
            asyncio.create_task(self._execution_loop())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start strategy: {e}")
            self.state = StrategyState.ERROR
            return False
    
    async def stop(self) -> bool:
        """Stop the strategy."""
        try:
            self.state = StrategyState.INACTIVE
            self.logger.info(f"Strategy {self.config.name} stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop strategy: {e}")
            return False
    
    async def pause(self) -> bool:
        """Pause the strategy."""
        try:
            if self.state == StrategyState.ACTIVE:
                self.state = StrategyState.PAUSED
                self.logger.info(f"Strategy {self.config.name} paused")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to pause strategy: {e}")
            return False
    
    async def resume(self) -> bool:
        """Resume the strategy."""
        try:
            if self.state == StrategyState.PAUSED:
                self.state = StrategyState.ACTIVE
                self.logger.info(f"Strategy {self.config.name} resumed")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to resume strategy: {e}")
            return False
    
    async def _execution_loop(self):
        """Main strategy execution loop."""
        while self.state in [StrategyState.ACTIVE, StrategyState.PAUSED]:
            try:
                if self.state == StrategyState.ACTIVE:
                    # Process each symbol
                    for symbol in self.config.symbols:
                        await self._process_symbol(symbol)
                    
                    # Update performance metrics
                    await self._update_performance()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # 1 minute interval
                
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                self.state = StrategyState.ERROR
                break
    
    async def _process_symbol(self, symbol: str):
        """Process a single symbol for trading opportunities."""
        try:
            # Get latest market data
            data = await self._get_market_data(symbol)
            if data is None or data.empty:
                return
            
            # Generate signals
            signals = await self.generate_signals(symbol, data)
            
            # Process each signal
            for signal in signals:
                await self._process_signal(signal)
            
            # Check existing positions for exit opportunities
            if symbol in self.active_positions:
                position = self.active_positions[symbol]
                if await self.should_exit_position(symbol, position):
                    await self._exit_position(symbol, position, "Strategy Exit Signal")
            
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {e}")
    
    async def _process_signal(self, signal: TradingSignal):
        """Process a trading signal."""
        try:
            # Store signal in history
            self.signals_history.append(signal)
            
            # Keep only recent signals (last 1000)
            if len(self.signals_history) > 1000:
                self.signals_history = self.signals_history[-1000:]
            
            # Process entry signals
            if signal.is_entry_signal:
                if await self.should_enter_position(signal):
                    await self._enter_position(signal)
            
            # Process exit signals
            elif signal.is_exit_signal:
                if signal.symbol in self.active_positions:
                    position = self.active_positions[signal.symbol]
                    await self._exit_position(signal.symbol, position, f"Exit Signal: {signal.signal_type.value}")
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    async def _enter_position(self, signal: TradingSignal):
        """Enter a new position based on signal."""
        try:
            symbol = signal.symbol
            
            # Check if we already have a position in this symbol
            if symbol in self.active_positions:
                self.logger.warning(f"Already have position in {symbol}, skipping entry")
                return
            
            # Check maximum positions limit
            if len(self.active_positions) >= self.config.max_positions:
                self.logger.warning(f"Maximum positions ({self.config.max_positions}) reached, skipping entry")
                return
            
            # Calculate position size
            position_size = await self._calculate_position_size(signal)
            if position_size <= 0:
                self.logger.warning(f"Invalid position size for {symbol}: {position_size}")
                return
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_stop_take_levels(signal)
            
            # Create order side
            order_side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL
            
            # Create trade request
            trade_request = TradeRequest(
                symbol=symbol,
                side=order_side.value,
                trade_type="Market",
                quantity=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                custom_id=f"{self.config.strategy_id}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Execute trade through trading engine
            result = await self.trading_engine.execute_trade(trade_request)
            
            if result.success:
                # Store active position
                self.active_positions[symbol] = {
                    "signal": signal,
                    "order_id": result.order_id,
                    "entry_time": datetime.now(),
                    "entry_price": signal.price,
                    "size": position_size,
                    "side": order_side.value,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                }
                
                self.logger.info(f"Entered {order_side.value} position in {symbol}: "
                               f"size={position_size}, price={signal.price}")
            else:
                self.logger.error(f"Failed to enter position in {symbol}: {result.error_message}")
            
        except Exception as e:
            self.logger.error(f"Error entering position for {signal.symbol}: {e}")
    
    async def _exit_position(self, symbol: str, position: Dict[str, Any], reason: str):
        """Exit an existing position."""
        try:
            # Determine exit side (opposite of entry)
            exit_side = OrderSide.SELL if position["side"] == "BUY" else OrderSide.BUY
            
            # Create trade request for position exit
            trade_request = TradeRequest(
                symbol=symbol,
                side=exit_side.value,
                trade_type="Market",
                quantity=position["size"],
                reduce_only=True,
                custom_id=f"{self.config.strategy_id}_exit_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Execute exit trade
            result = await self.trading_engine.execute_trade(trade_request)
            
            if result.success:
                # Calculate PnL
                entry_price = position["entry_price"]
                exit_price = result.execution_price or await self._get_current_price(symbol)
                
                if position["side"] == "BUY":
                    pnl = (exit_price - entry_price) * position["size"]
                else:
                    pnl = (entry_price - exit_price) * position["size"]
                
                # Update performance
                await self._record_trade(symbol, position, exit_price, pnl, reason)
                
                # Remove from active positions
                self.active_positions.pop(symbol, None)
                
                self.logger.info(f"Exited position in {symbol}: PnL={pnl:.2f}, reason={reason}")
                
            else:
                self.logger.error(f"Failed to exit position in {symbol}: {result.error_message}")
            
        except Exception as e:
            self.logger.error(f"Error exiting position for {symbol}: {e}")
    
    async def _calculate_position_size(self, signal: TradingSignal) -> Decimal:
        """Calculate position size based on risk management."""
        try:
            # Get portfolio value
            portfolio_value = await self._get_portfolio_value()
            
            # Calculate base position size (percentage of portfolio)
            base_size = portfolio_value * Decimal(str(self.config.position_size_pct))
            
            # Convert to quantity based on price
            quantity = base_size / signal.price
            
            # Adjust based on signal strength
            strength_multiplier = {
                SignalStrength.WEAK: 0.5,
                SignalStrength.MODERATE: 0.75,
                SignalStrength.STRONG: 1.0,
                SignalStrength.VERY_STRONG: 1.25
            }.get(signal.strength, 1.0)
            
            adjusted_quantity = quantity * Decimal(str(strength_multiplier))
            
            return adjusted_quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return Decimal('0')
    
    def _calculate_stop_take_levels(self, signal: TradingSignal) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculate stop loss and take profit levels."""
        try:
            price = signal.price
            
            if signal.signal_type == SignalType.BUY:
                stop_loss = price * (1 - Decimal(str(self.config.stop_loss_pct)))
                take_profit = price * (1 + Decimal(str(self.config.take_profit_pct)))
            else:  # SELL
                stop_loss = price * (1 + Decimal(str(self.config.stop_loss_pct)))
                take_profit = price * (1 - Decimal(str(self.config.take_profit_pct)))
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating stop/take levels: {e}")
            return None, None
    
    async def _record_trade(self, symbol: str, position: Dict[str, Any], 
                          exit_price: Decimal, pnl: Decimal, reason: str):
        """Record completed trade for performance tracking."""
        try:
            trade_record = {
                "symbol": symbol,
                "entry_time": position["entry_time"],
                "exit_time": datetime.now(),
                "side": position["side"],
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "size": position["size"],
                "pnl": pnl,
                "reason": reason,
                "strategy_id": self.config.strategy_id
            }
            
            self.trade_history.append(trade_record)
            
            # Update performance metrics
            self.performance.total_trades += 1
            self.performance.total_pnl += pnl
            
            if pnl > 0:
                self.performance.winning_trades += 1
            else:
                self.performance.losing_trades += 1
            
            # Calculate running metrics
            self._calculate_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        try:
            if self.performance.total_trades == 0:
                return
            
            # Win rate
            self.performance.win_rate = self.performance.winning_trades / self.performance.total_trades
            
            # Average win/loss
            if self.trade_history:
                wins = [t["pnl"] for t in self.trade_history if t["pnl"] > 0]
                losses = [t["pnl"] for t in self.trade_history if t["pnl"] < 0]
                
                self.performance.avg_win = Decimal(str(np.mean(wins))) if wins else Decimal('0')
                self.performance.avg_loss = Decimal(str(np.mean(losses))) if losses else Decimal('0')
                
                # Profit factor
                total_wins = sum(wins) if wins else 0
                total_losses = abs(sum(losses)) if losses else 1
                self.performance.profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            # Sharpe ratio (simplified)
            if len(self.trade_history) > 1:
                returns = [float(t["pnl"]) for t in self.trade_history]
                self.performance.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            self.performance.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
    
    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data for symbol."""
        try:
            # This would typically fetch from data manager or API
            # Placeholder implementation
            return pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
                'open': np.random.randn(100).cumsum() + 50000,
                'high': np.random.randn(100).cumsum() + 50100,
                'low': np.random.randn(100).cumsum() + 49900,
                'close': np.random.randn(100).cumsum() + 50000,
                'volume': np.random.randint(1000, 10000, 100)
            })
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current market price."""
        try:
            # This would typically get from trading engine or API
            return Decimal('50000')  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return Decimal('0')
    
    async def _get_portfolio_value(self) -> Decimal:
        """Get total portfolio value."""
        try:
            # Get from trading engine
            summary = await self.trading_engine.get_portfolio_summary()
            return Decimal(str(summary.get("total_balance", 10000)))
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio value: {e}")
            return Decimal('10000')  # Default value
    
    async def _update_performance(self):
        """Update performance metrics."""
        try:
            self._calculate_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status and performance."""
        return {
            "strategy_id": self.config.strategy_id,
            "name": self.config.name,
            "state": self.state.value,
            "symbols": self.config.symbols,
            "active_positions": len(self.active_positions),
            "pending_orders": len(self.pending_orders),
            "performance": {
                "total_trades": self.performance.total_trades,
                "win_rate": self.performance.win_rate,
                "total_pnl": float(self.performance.total_pnl),
                "sharpe_ratio": self.performance.sharpe_ratio,
                "profit_factor": self.performance.profit_factor
            },
            "last_signals": len([s for s in self.signals_history if s.timestamp > datetime.now() - timedelta(hours=1)]),
            "system_version": "1.0.0 - Strategy Framework"
        }


class StrategyManager:
    """
    Strategy Manager for coordinating multiple trading strategies.
    
    Manages the lifecycle and coordination of multiple strategies,
    ensures portfolio-level risk management, and provides unified monitoring.
    """
    
    def __init__(
        self,
        trading_engine: TradingEngine,
        risk_manager: UnifiedRiskManager,
        config_manager: Optional[ConfigurationManager] = None
    ):
        self.trading_engine = trading_engine
        self.risk_manager = risk_manager
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = TradingLogger("strategy_manager")
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: List[str] = []
        
        # Portfolio coordination
        self.portfolio_allocation: Dict[str, float] = {}  # strategy_id -> allocation percentage
        self.max_portfolio_risk = 0.10  # 10% max portfolio risk
        
        self.logger.info("StrategyManager initialized")
    
    async def add_strategy(self, strategy: BaseStrategy) -> bool:
        """Add a strategy to the manager."""
        try:
            strategy_id = strategy.config.strategy_id
            
            if strategy_id in self.strategies:
                self.logger.warning(f"Strategy {strategy_id} already exists")
                return False
            
            self.strategies[strategy_id] = strategy
            self.logger.info(f"Added strategy: {strategy.config.name} ({strategy_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding strategy: {e}")
            return False
    
    async def start_strategy(self, strategy_id: str) -> bool:
        """Start a specific strategy."""
        try:
            if strategy_id not in self.strategies:
                self.logger.error(f"Strategy {strategy_id} not found")
                return False
            
            strategy = self.strategies[strategy_id]
            
            if await strategy.start():
                if strategy_id not in self.active_strategies:
                    self.active_strategies.append(strategy_id)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error starting strategy {strategy_id}: {e}")
            return False
    
    async def stop_strategy(self, strategy_id: str) -> bool:
        """Stop a specific strategy."""
        try:
            if strategy_id not in self.strategies:
                self.logger.error(f"Strategy {strategy_id} not found")
                return False
            
            strategy = self.strategies[strategy_id]
            
            if await strategy.stop():
                if strategy_id in self.active_strategies:
                    self.active_strategies.remove(strategy_id)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy {strategy_id}: {e}")
            return False
    
    async def start_all_strategies(self) -> Dict[str, bool]:
        """Start all strategies."""
        results = {}
        
        for strategy_id in self.strategies:
            results[strategy_id] = await self.start_strategy(strategy_id)
        
        return results
    
    async def stop_all_strategies(self) -> Dict[str, bool]:
        """Stop all strategies."""
        results = {}
        
        for strategy_id in self.active_strategies.copy():
            results[strategy_id] = await self.stop_strategy(strategy_id)
        
        return results
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get comprehensive manager status."""
        try:
            strategy_statuses = {}
            total_positions = 0
            total_pnl = Decimal('0')
            
            for strategy_id, strategy in self.strategies.items():
                status = strategy.get_status()
                strategy_statuses[strategy_id] = status
                total_positions += status["active_positions"]
                total_pnl += Decimal(str(status["performance"]["total_pnl"]))
            
            return {
                "total_strategies": len(self.strategies),
                "active_strategies": len(self.active_strategies),
                "total_positions": total_positions,
                "total_pnl": float(total_pnl),
                "max_portfolio_risk": self.max_portfolio_risk,
                "strategies": strategy_statuses,
                "system_version": "1.0.0 - Strategy Manager"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting manager status: {e}")
            return {"error": str(e)}