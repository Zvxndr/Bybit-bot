"""
Strategy Management System for Bybit Trading Bot

This module implements comprehensive strategy management including:
- Strategy lifecycle management (load, start, stop, monitor)
- Signal processing and trade execution coordination
- Multi-strategy support with individual performance tracking
- Strategy parameter optimization and adaptation
- Real-time strategy performance monitoring

Coordinates between ML models, risk management, and trading engine
to provide seamless strategy execution and management.

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal
from abc import ABC, abstractmethod

from ..config_manager import ConfigurationManager
from ..utils.logging import TradingLogger
from .trading_engine import TradingEngine, Order, OrderSide, OrderType
from ..risk_management.portfolio_manager import PortfolioManager
from ..risk_management.risk_manager import RiskManager, TradeRiskAssessment
from ..ml.model_manager import ModelManager


class StrategyStatus(Enum):
    """Strategy status enumeration."""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    symbol: str
    signal_type: SignalType
    confidence: Decimal
    entry_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'confidence': float(self.confidence),
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': float(self.take_profit) if self.take_profit else None,
            'quantity': float(self.quantity) if self.quantity else None,
            'timestamp': self.timestamp.isoformat(),
            'strategy_id': self.strategy_id,
            'metadata': self.metadata
        }


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    strategy_id: str
    total_trades: int
    winning_trades: int
    total_pnl: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    average_trade_duration: timedelta
    last_trade_time: Optional[datetime] = None
    start_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_id': self.strategy_id,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_pnl': float(self.total_pnl),
            'win_rate': float(self.win_rate),
            'profit_factor': float(self.profit_factor),
            'sharpe_ratio': float(self.sharpe_ratio),
            'max_drawdown': float(self.max_drawdown),
            'current_drawdown': float(self.current_drawdown),
            'average_trade_duration_hours': self.average_trade_duration.total_seconds() / 3600,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'start_time': self.start_time.isoformat(),
            'runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All trading strategies must inherit from this class and implement
    the required methods for signal generation and lifecycle management.
    """
    
    def __init__(self, strategy_id: str, config: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.config = config
        self.logger = TradingLogger(f"strategy_{strategy_id}")
        self.status = StrategyStatus.INACTIVE
        self.last_signal_time: Optional[datetime] = None
        
    @abstractmethod
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate trading signal based on market data.
        
        Args:
            symbol: Trading symbol
            data: Historical market data
            
        Returns:
            TradingSignal or None
        """
        pass
    
    @abstractmethod
    async def on_trade_executed(self, signal: TradingSignal, order: Order) -> None:
        """
        Called when a trade based on this strategy's signal is executed.
        
        Args:
            signal: The original trading signal
            order: The executed order
        """
        pass
    
    async def on_start(self) -> bool:
        """Called when strategy is starting. Override for custom initialization."""
        self.status = StrategyStatus.ACTIVE
        self.logger.info(f"Strategy {self.strategy_id} started")
        return True
    
    async def on_stop(self) -> None:
        """Called when strategy is stopping. Override for custom cleanup."""
        self.status = StrategyStatus.INACTIVE
        self.logger.info(f"Strategy {self.strategy_id} stopped")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.config.copy()
    
    def update_parameters(self, new_params: Dict[str, Any]) -> None:
        """Update strategy parameters."""
        self.config.update(new_params)
        self.logger.info(f"Strategy {self.strategy_id} parameters updated")


class MLStrategy(BaseStrategy):
    """
    ML-based trading strategy using the model manager.
    
    This strategy uses machine learning models to generate trading signals
    based on technical indicators, market data, and other features.
    """
    
    def __init__(self, strategy_id: str, config: Dict[str, Any], model_manager: ModelManager):
        super().__init__(strategy_id, config)
        self.model_manager = model_manager
        self.confidence_threshold = Decimal(str(config.get('confidence_threshold', 0.6)))
        self.lookback_periods = config.get('lookback_periods', 100)
        
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate ML-based trading signal."""
        try:
            if len(data) < self.lookback_periods:
                return None
            
            # Get model prediction
            features = await self._prepare_features(data)
            prediction = await self.model_manager.predict(features, symbol)
            
            if not prediction or 'signal' not in prediction:
                return None
            
            signal_value = prediction['signal']
            confidence = Decimal(str(prediction.get('confidence', 0.5)))
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                return None
            
            # Determine signal type
            if signal_value > 0.6:
                signal_type = SignalType.BUY
            elif signal_value < 0.4:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            if signal_type == SignalType.HOLD:
                return None
            
            # Calculate entry price and risk levels
            current_price = Decimal(str(data.iloc[-1]['close']))
            volatility = self._calculate_volatility(data)
            
            # Set stop loss and take profit based on volatility
            stop_loss_pct = Decimal('0.02') + volatility * Decimal('0.5')  # 2% + volatility adjustment
            take_profit_pct = stop_loss_pct * Decimal('2')  # 2:1 reward-to-risk ratio
            
            if signal_type == SignalType.BUY:
                stop_loss = current_price * (Decimal('1') - stop_loss_pct)
                take_profit = current_price * (Decimal('1') + take_profit_pct)
            else:  # SELL
                stop_loss = current_price * (Decimal('1') + stop_loss_pct)
                take_profit = current_price * (Decimal('1') - take_profit_pct)
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_id=self.strategy_id,
                metadata={
                    'model_prediction': signal_value,
                    'volatility': float(volatility),
                    'features_used': list(features.keys()) if isinstance(features, dict) else []
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating ML signal for {symbol}: {e}")
            return None
    
    async def _prepare_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Prepare features for ML model."""
        try:
            # Use the latest data point for features
            latest = data.iloc[-1]
            features = {
                'close': float(latest['close']),
                'volume': float(latest['volume']),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'open': float(latest['open'])
            }
            
            # Add technical indicators if available
            if 'sma_20' in data.columns:
                features['sma_20'] = float(latest['sma_20'])
            if 'rsi' in data.columns:
                features['rsi'] = float(latest['rsi'])
            if 'macd' in data.columns:
                features['macd'] = float(latest['macd'])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return {}
    
    def _calculate_volatility(self, data: pd.DataFrame, periods: int = 20) -> Decimal:
        """Calculate price volatility."""
        try:
            if len(data) < periods:
                return Decimal('0.02')  # Default 2% volatility
            
            returns = data['close'].pct_change().dropna()
            volatility = returns.tail(periods).std()
            return Decimal(str(max(0.005, min(0.1, volatility))))  # Clamp between 0.5% and 10%
            
        except Exception:
            return Decimal('0.02')
    
    async def on_trade_executed(self, signal: TradingSignal, order: Order) -> None:
        """Handle trade execution feedback for ML model."""
        try:
            # Record trade for model learning
            trade_data = {
                'signal': signal.to_dict(),
                'order': {
                    'order_id': order.order_id,
                    'filled_quantity': float(order.filled_quantity),
                    'average_price': float(order.average_price) if order.average_price else None,
                    'timestamp': order.timestamp.isoformat() if order.timestamp else None
                }
            }
            
            # Update model with trade outcome (if supported)
            await self.model_manager.update_with_trade_outcome(trade_data)
            
        except Exception as e:
            self.logger.error(f"Error handling trade execution feedback: {e}")


class StrategyManager:
    """
    Comprehensive strategy management system.
    
    Features:
    - Multi-strategy support with independent execution
    - Real-time signal processing and execution
    - Individual strategy performance tracking
    - Risk-adjusted position sizing per strategy
    - Strategy lifecycle management (start/stop/pause)
    - Automated strategy parameter optimization
    """
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        trading_engine: TradingEngine,
        portfolio_manager: PortfolioManager,
        risk_manager: RiskManager,
        model_manager: ModelManager
    ):
        self.config = config_manager
        self.trading_engine = trading_engine
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.model_manager = model_manager
        self.logger = TradingLogger("strategy_manager")
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.active_signals: Dict[str, TradingSignal] = {}  # symbol -> signal
        
        # Configuration
        self.max_concurrent_trades = config_manager.get('strategy.max_concurrent_trades', 5)
        self.signal_cooldown = timedelta(seconds=config_manager.get('strategy.signal_cooldown_seconds', 300))
        self.min_signal_confidence = Decimal(str(config_manager.get('strategy.min_signal_confidence', 0.6)))
        
        # Execution tracking
        self.pending_executions: Dict[str, TradingSignal] = {}
        self.execution_tasks: List[asyncio.Task] = []
        
        self.logger.info("StrategyManager initialized")
    
    async def add_strategy(self, strategy: BaseStrategy) -> bool:
        """
        Add a new strategy to the manager.
        
        Args:
            strategy: Strategy instance to add
            
        Returns:
            bool: True if added successfully
        """
        try:
            if strategy.strategy_id in self.strategies:
                self.logger.warning(f"Strategy {strategy.strategy_id} already exists")
                return False
            
            self.strategies[strategy.strategy_id] = strategy
            self.strategy_performance[strategy.strategy_id] = StrategyPerformance(
                strategy_id=strategy.strategy_id,
                total_trades=0,
                winning_trades=0,
                total_pnl=Decimal('0'),
                win_rate=Decimal('0'),
                profit_factor=Decimal('0'),
                sharpe_ratio=Decimal('0'),
                max_drawdown=Decimal('0'),
                current_drawdown=Decimal('0'),
                average_trade_duration=timedelta(hours=1)
            )
            
            self.logger.info(f"Added strategy: {strategy.strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding strategy {strategy.strategy_id}: {e}")
            return False
    
    async def start_strategy(self, strategy_id: str) -> bool:
        """
        Start a strategy.
        
        Args:
            strategy_id: ID of strategy to start
            
        Returns:
            bool: True if started successfully
        """
        try:
            if strategy_id not in self.strategies:
                self.logger.error(f"Strategy {strategy_id} not found")
                return False
            
            strategy = self.strategies[strategy_id]
            
            if strategy.status == StrategyStatus.ACTIVE:
                self.logger.warning(f"Strategy {strategy_id} is already active")
                return True
            
            strategy.status = StrategyStatus.STARTING
            success = await strategy.on_start()
            
            if success:
                strategy.status = StrategyStatus.ACTIVE
                self.logger.info(f"Started strategy: {strategy_id}")
                return True
            else:
                strategy.status = StrategyStatus.ERROR
                self.logger.error(f"Failed to start strategy: {strategy_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting strategy {strategy_id}: {e}")
            return False
    
    async def stop_strategy(self, strategy_id: str) -> bool:
        """
        Stop a strategy.
        
        Args:
            strategy_id: ID of strategy to stop
            
        Returns:
            bool: True if stopped successfully
        """
        try:
            if strategy_id not in self.strategies:
                self.logger.error(f"Strategy {strategy_id} not found")
                return False
            
            strategy = self.strategies[strategy_id]
            strategy.status = StrategyStatus.STOPPING
            
            await strategy.on_stop()
            strategy.status = StrategyStatus.INACTIVE
            
            # Cancel any pending executions for this strategy
            pending_to_remove = [
                symbol for symbol, signal in self.pending_executions.items()
                if signal.strategy_id == strategy_id
            ]
            for symbol in pending_to_remove:
                del self.pending_executions[symbol]
            
            self.logger.info(f"Stopped strategy: {strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy {strategy_id}: {e}")
            return False
    
    async def process_market_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Process market data through all active strategies.
        
        Args:
            symbol: Trading symbol
            data: Market data
        """
        try:
            # Process data through each active strategy
            for strategy_id, strategy in self.strategies.items():
                if strategy.status != StrategyStatus.ACTIVE:
                    continue
                
                # Check signal cooldown
                if (strategy.last_signal_time and 
                    datetime.now() - strategy.last_signal_time < self.signal_cooldown):
                    continue
                
                # Generate signal
                signal = await strategy.generate_signal(symbol, data)
                
                if signal and signal.confidence >= self.min_signal_confidence:
                    strategy.last_signal_time = datetime.now()
                    await self._process_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error processing market data for {symbol}: {e}")
    
    async def _process_signal(self, signal: TradingSignal) -> None:
        """
        Process a trading signal.
        
        Args:
            signal: Trading signal to process
        """
        try:
            # Check if we have too many concurrent trades
            active_trades = len(self.pending_executions)
            if active_trades >= self.max_concurrent_trades:
                self.logger.warning(f"Max concurrent trades reached: {active_trades}")
                return
            
            # Check for conflicting signals
            if signal.symbol in self.active_signals:
                existing_signal = self.active_signals[signal.symbol]
                if existing_signal.signal_type != signal.signal_type:
                    self.logger.info(f"Conflicting signals for {signal.symbol}, using higher confidence")
                    if signal.confidence <= existing_signal.confidence:
                        return
            
            # Store signal
            self.active_signals[signal.symbol] = signal
            
            # Execute signal asynchronously
            task = asyncio.create_task(self._execute_signal(signal))
            self.execution_tasks.append(task)
            
            # Clean up completed tasks
            self.execution_tasks = [t for t in self.execution_tasks if not t.done()]
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    async def _execute_signal(self, signal: TradingSignal) -> None:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
        """
        try:
            self.pending_executions[signal.symbol] = signal
            
            # Get current portfolio balance
            portfolio_value = self.portfolio_manager.get_total_value()
            
            # Assess trade risk
            side = "buy" if signal.signal_type == SignalType.BUY else "sell"
            risk_assessment = await self.risk_manager.assess_trade_risk(
                symbol=signal.symbol,
                side=side,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                current_balance=portfolio_value,
                strategy_id=signal.strategy_id
            )
            
            if not risk_assessment.is_approved:
                self.logger.warning(
                    f"Trade rejected by risk manager: {signal.symbol} - {risk_assessment.risk_reason}"
                )
                return
            
            # Use risk-adjusted position size
            quantity = risk_assessment.position_size
            
            # Place order
            order_side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL
            order = await self.trading_engine.place_order(
                symbol=signal.symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=quantity
            )
            
            if order:
                self.logger.info(
                    f"Order placed for signal: {signal.symbol} {signal.signal_type.value} "
                    f"Qty: {quantity} Confidence: {signal.confidence}"
                )
                
                # Notify strategy of execution
                strategy = self.strategies.get(signal.strategy_id)
                if strategy:
                    await strategy.on_trade_executed(signal, order)
                
                # Update performance tracking
                await self._update_strategy_performance(signal.strategy_id, order)
            else:
                self.logger.error(f"Failed to place order for signal: {signal.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error executing signal {signal.symbol}: {e}")
        finally:
            # Clean up
            if signal.symbol in self.pending_executions:
                del self.pending_executions[signal.symbol]
            if signal.symbol in self.active_signals:
                del self.active_signals[signal.symbol]
    
    async def _update_strategy_performance(self, strategy_id: str, order: Order) -> None:
        """Update strategy performance metrics."""
        try:
            if strategy_id not in self.strategy_performance:
                return
            
            perf = self.strategy_performance[strategy_id]
            perf.total_trades += 1
            perf.last_trade_time = datetime.now()
            
            # Note: PnL will be updated when position is closed
            # This is just tracking trade execution
            
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {e}")
    
    def get_strategy_status(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get strategy status and performance.
        
        Args:
            strategy_id: Specific strategy ID, or None for all strategies
            
        Returns:
            Dictionary of strategy information
        """
        try:
            if strategy_id:
                if strategy_id not in self.strategies:
                    return {}
                
                strategy = self.strategies[strategy_id]
                performance = self.strategy_performance.get(strategy_id)
                
                return {
                    'strategy_id': strategy_id,
                    'status': strategy.status.value,
                    'parameters': strategy.get_parameters(),
                    'performance': performance.to_dict() if performance else {},
                    'last_signal_time': strategy.last_signal_time.isoformat() if strategy.last_signal_time else None
                }
            else:
                # Return all strategies
                result = {}
                for sid, strategy in self.strategies.items():
                    result[sid] = self.get_strategy_status(sid)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error getting strategy status: {e}")
            return {}
    
    def get_active_signals(self) -> Dict[str, Dict[str, Any]]:
        """Get all active trading signals."""
        return {
            symbol: signal.to_dict() 
            for symbol, signal in self.active_signals.items()
        }
    
    async def create_ml_strategy(self, strategy_id: str, config: Dict[str, Any]) -> bool:
        """
        Create and add an ML-based strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            config: Strategy configuration
            
        Returns:
            bool: True if created successfully
        """
        try:
            ml_strategy = MLStrategy(strategy_id, config, self.model_manager)
            return await self.add_strategy(ml_strategy)
            
        except Exception as e:
            self.logger.error(f"Error creating ML strategy {strategy_id}: {e}")
            return False