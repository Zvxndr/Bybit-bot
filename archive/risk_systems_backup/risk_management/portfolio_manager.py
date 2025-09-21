"""
Portfolio Management System for Bybit Trading Bot

This module implements comprehensive portfolio management including:
- Multi-asset portfolio tracking and allocation
- Real-time performance calculation and analytics
- Risk-adjusted position management
- Portfolio rebalancing and optimization
- Historical performance tracking and reporting

Integrates with TradingEngine and RiskManager for complete
portfolio oversight and management.

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_DOWN, ROUND_UP

from ..config_manager import ConfigurationManager
from ..utils.logging import setup_logger
from .risk_manager import RiskManager, RiskMetrics


@dataclass
class AssetAllocation:
    """Asset allocation data structure."""
    symbol: str
    target_weight: Decimal
    current_weight: Decimal
    current_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_return: Decimal
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def weight_deviation(self) -> Decimal:
        """Calculate deviation from target weight."""
        return abs(self.current_weight - self.target_weight)
    
    def needs_rebalancing(self, threshold: Decimal = Decimal('0.05')) -> bool:
        """Check if asset needs rebalancing based on weight deviation."""
        return self.weight_deviation > threshold


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    total_value: Decimal
    total_pnl: Decimal
    total_pnl_percentage: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    daily_return: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    total_trades: int
    winning_trades: int
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/logging."""
        return {
            'total_value': float(self.total_value),
            'total_pnl': float(self.total_pnl),
            'total_pnl_percentage': float(self.total_pnl_percentage),
            'unrealized_pnl': float(self.unrealized_pnl),
            'realized_pnl': float(self.realized_pnl),
            'daily_return': float(self.daily_return),
            'sharpe_ratio': float(self.sharpe_ratio),
            'max_drawdown': float(self.max_drawdown),
            'current_drawdown': float(self.current_drawdown),
            'win_rate': float(self.win_rate),
            'profit_factor': float(self.profit_factor),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'average_win': float(self.average_win),
            'average_loss': float(self.average_loss),
            'largest_win': float(self.largest_win),
            'largest_loss': float(self.largest_loss),
            'timestamp': self.timestamp.isoformat()
        }


class PortfolioManager:
    """
    Comprehensive portfolio management system.
    
    Features:
    - Multi-asset portfolio tracking with real-time updates
    - Performance analytics and risk metrics
    - Portfolio rebalancing and allocation management
    - Historical performance tracking and reporting
    - Integration with risk management system
    - Automated portfolio optimization
    """
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        risk_manager: RiskManager,
        initial_balance: Decimal = Decimal('10000')
    ):
        self.config = config_manager
        self.risk_manager = risk_manager
        self.logger = setup_logger("portfolio_manager")
        
        # Portfolio state
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions: Dict[str, AssetAllocation] = {}
        self.target_allocations: Dict[str, Decimal] = {}
        
        # Performance tracking
        self.peak_value = initial_balance
        self.daily_values: List[Tuple[datetime, Decimal]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.return_series: pd.Series = pd.Series(dtype=float)
        
        # Configuration
        self.rebalance_threshold = Decimal(str(config_manager.get('portfolio.rebalance_threshold', 0.05)))
        self.max_positions = config_manager.get('portfolio.max_positions', 10)
        self.min_position_size = Decimal(str(config_manager.get('portfolio.min_position_size', 100)))
        
        # Performance calculation intervals
        self.last_daily_update = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        self.logger.info(f"PortfolioManager initialized with balance: {initial_balance}")
    
    async def add_position(
        self,
        symbol: str,
        quantity: Decimal,
        entry_price: Decimal,
        side: str = "long"
    ) -> bool:
        """
        Add a new position to the portfolio.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            entry_price: Entry price
            side: Position side (long/short)
            
        Returns:
            bool: True if position added successfully
        """
        try:
            position_value = quantity * entry_price
            
            # Check if we have enough cash
            if position_value > self.cash_balance:
                self.logger.error(f"Insufficient cash for position: {position_value} > {self.cash_balance}")
                return False
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                self.logger.error(f"Maximum positions reached: {self.max_positions}")
                return False
            
            # Check minimum position size
            if position_value < self.min_position_size:
                self.logger.error(f"Position size below minimum: {position_value} < {self.min_position_size}")
                return False
            
            # Update or create position
            if symbol in self.positions:
                # Update existing position (average in)
                current_pos = self.positions[symbol]
                total_quantity = current_pos.current_value / entry_price + quantity  # Approximate
                weighted_price = (current_pos.current_value + position_value) / total_quantity
                
                self.positions[symbol].current_value += position_value
            else:
                # Create new position
                portfolio_value = self.get_total_value()
                current_weight = position_value / portfolio_value if portfolio_value > 0 else Decimal('0')
                
                self.positions[symbol] = AssetAllocation(
                    symbol=symbol,
                    target_weight=current_weight,
                    current_weight=current_weight,
                    current_value=position_value,
                    unrealized_pnl=Decimal('0'),
                    realized_pnl=Decimal('0'),
                    total_return=Decimal('0')
                )
            
            # Update cash balance
            self.cash_balance -= position_value
            
            self.logger.info(f"Added position: {symbol} {quantity} @ {entry_price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding position {symbol}: {e}")
            return False
    
    async def update_position(
        self,
        symbol: str,
        current_price: Decimal,
        unrealized_pnl: Decimal
    ) -> None:
        """
        Update an existing position with current market data.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            unrealized_pnl: Current unrealized PnL
        """
        try:
            if symbol not in self.positions:
                self.logger.warning(f"Position {symbol} not found for update")
                return
            
            position = self.positions[symbol]
            
            # Update position values
            # Approximate current value based on price change
            price_change = current_price / (position.current_value / (position.current_value / current_price + unrealized_pnl))
            position.current_value = position.current_value * price_change
            position.unrealized_pnl = unrealized_pnl
            position.total_return = position.realized_pnl + position.unrealized_pnl
            position.last_updated = datetime.now()
            
            # Recalculate portfolio weights
            await self._recalculate_weights()
            
        except Exception as e:
            self.logger.error(f"Error updating position {symbol}: {e}")
    
    async def close_position(
        self,
        symbol: str,
        exit_price: Decimal,
        realized_pnl: Decimal
    ) -> bool:
        """
        Close a position and update portfolio.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            realized_pnl: Realized PnL from the trade
            
        Returns:
            bool: True if position closed successfully
        """
        try:
            if symbol not in self.positions:
                self.logger.warning(f"Position {symbol} not found for closing")
                return False
            
            position = self.positions[symbol]
            
            # Update cash balance with realized value
            self.cash_balance += position.current_value + realized_pnl
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'pnl': float(realized_pnl),
                'exit_price': float(exit_price),
                'duration': (datetime.now() - position.last_updated).total_seconds() / 3600,  # hours
                'return_pct': float(realized_pnl / position.current_value * 100) if position.current_value > 0 else 0
            }
            self.trade_history.append(trade_record)
            
            # Remove position
            del self.positions[symbol]
            
            # Recalculate weights
            await self._recalculate_weights()
            
            self.logger.info(f"Closed position: {symbol} PnL: {realized_pnl}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    def get_total_value(self) -> Decimal:
        """
        Calculate total portfolio value.
        
        Returns:
            Total portfolio value including cash and positions
        """
        total_positions_value = sum(pos.current_value for pos in self.positions.values())
        return self.cash_balance + total_positions_value
    
    def get_unrealized_pnl(self) -> Decimal:
        """Get total unrealized PnL across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_realized_pnl(self) -> Decimal:
        """Get total realized PnL across all closed positions."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    async def calculate_performance_metrics(self) -> PerformanceMetrics:
        """
        Calculate comprehensive portfolio performance metrics.
        
        Returns:
            PerformanceMetrics object
        """
        try:
            # Current portfolio values
            total_value = self.get_total_value()
            unrealized_pnl = self.get_unrealized_pnl()
            realized_pnl = self.get_realized_pnl()
            total_pnl = unrealized_pnl + realized_pnl
            total_pnl_percentage = (total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else Decimal('0')
            
            # Update peak value and calculate drawdown
            if total_value > self.peak_value:
                self.peak_value = total_value
            
            current_drawdown = Decimal('0')
            max_drawdown = Decimal('0')
            if self.peak_value > 0:
                current_drawdown = (self.peak_value - total_value) / self.peak_value * 100
                
                # Calculate max drawdown from historical data
                if len(self.daily_values) > 1:
                    values = [float(v[1]) for v in self.daily_values]
                    peak = max(values)
                    trough = min(values)
                    max_drawdown = (peak - trough) / peak * 100 if peak > 0 else 0
            
            # Daily return calculation
            daily_return = Decimal('0')
            if len(self.daily_values) >= 2:
                yesterday_value = self.daily_values[-2][1]
                daily_return = (total_value - yesterday_value) / yesterday_value * 100 if yesterday_value > 0 else Decimal('0')
            
            # Sharpe ratio from return series
            sharpe_ratio = Decimal('0')
            if len(self.return_series) > 30:
                excess_returns = self.return_series - 0.02/252  # Assuming 2% risk-free rate
                if excess_returns.std() > 0:
                    sharpe_ratio = Decimal(str(excess_returns.mean() / excess_returns.std() * np.sqrt(252)))
            
            # Trade statistics
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
            win_rate = Decimal(str(winning_trades / total_trades * 100)) if total_trades > 0 else Decimal('0')
            
            # Profit factor
            total_wins = sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0)
            total_losses = abs(sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0))
            profit_factor = Decimal(str(total_wins / total_losses)) if total_losses > 0 else Decimal('0')
            
            # Average win/loss
            wins = [trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0]
            losses = [abs(trade['pnl']) for trade in self.trade_history if trade['pnl'] < 0]
            
            average_win = Decimal(str(sum(wins) / len(wins))) if wins else Decimal('0')
            average_loss = Decimal(str(sum(losses) / len(losses))) if losses else Decimal('0')
            largest_win = Decimal(str(max(wins))) if wins else Decimal('0')
            largest_loss = Decimal(str(max(losses))) if losses else Decimal('0')
            
            return PerformanceMetrics(
                total_value=total_value,
                total_pnl=total_pnl,
                total_pnl_percentage=total_pnl_percentage,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                daily_return=daily_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=Decimal(str(max_drawdown)),
                current_drawdown=current_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=winning_trades,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics(
                total_value=self.get_total_value(),
                total_pnl=Decimal('0'),
                total_pnl_percentage=Decimal('0'),
                unrealized_pnl=Decimal('0'),
                realized_pnl=Decimal('0'),
                daily_return=Decimal('0'),
                sharpe_ratio=Decimal('0'),
                max_drawdown=Decimal('0'),
                current_drawdown=Decimal('0'),
                win_rate=Decimal('0'),
                profit_factor=Decimal('0'),
                total_trades=0,
                winning_trades=0,
                average_win=Decimal('0'),
                average_loss=Decimal('0'),
                largest_win=Decimal('0'),
                largest_loss=Decimal('0')
            )
    
    async def check_rebalancing_needed(self) -> List[str]:
        """
        Check which positions need rebalancing.
        
        Returns:
            List of symbols that need rebalancing
        """
        rebalance_needed = []
        
        try:
            for symbol, position in self.positions.items():
                if position.needs_rebalancing(self.rebalance_threshold):
                    rebalance_needed.append(symbol)
                    self.logger.info(
                        f"Rebalancing needed for {symbol}: "
                        f"Current: {position.current_weight:.3f}, "
                        f"Target: {position.target_weight:.3f}"
                    )
            
            return rebalance_needed
            
        except Exception as e:
            self.logger.error(f"Error checking rebalancing: {e}")
            return []
    
    async def update_daily_performance(self) -> None:
        """Update daily performance tracking."""
        try:
            now = datetime.now()
            current_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Check if we need to record daily value
            if current_date > self.last_daily_update:
                total_value = self.get_total_value()
                self.daily_values.append((current_date, total_value))
                
                # Calculate daily return for return series
                if len(self.daily_values) >= 2:
                    yesterday_value = self.daily_values[-2][1]
                    daily_return = float((total_value - yesterday_value) / yesterday_value) if yesterday_value > 0 else 0.0
                    
                    self.return_series = pd.concat([
                        self.return_series,
                        pd.Series([daily_return], index=[current_date])
                    ])
                    
                    # Keep only last 252 trading days (1 year)
                    if len(self.return_series) > 252:
                        self.return_series = self.return_series.tail(252)
                
                # Keep only last 365 daily values
                if len(self.daily_values) > 365:
                    self.daily_values = self.daily_values[-365:]
                
                self.last_daily_update = current_date
                
                # Update risk manager with performance data
                latest_pnl = None
                if len(self.trade_history) > 0:
                    latest_trade = max(self.trade_history, key=lambda x: x['timestamp'])
                    if latest_trade['timestamp'].date() == now.date():
                        latest_pnl = Decimal(str(latest_trade['pnl']))
                
                self.risk_manager.update_performance(total_value, latest_pnl)
                
        except Exception as e:
            self.logger.error(f"Error updating daily performance: {e}")
    
    async def _recalculate_weights(self) -> None:
        """Recalculate portfolio weights after position changes."""
        try:
            total_value = self.get_total_value()
            
            if total_value > 0:
                for position in self.positions.values():
                    position.current_weight = position.current_value / total_value
                    
        except Exception as e:
            self.logger.error(f"Error recalculating weights: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        try:
            total_value = self.get_total_value()
            
            summary = {
                'total_value': float(total_value),
                'cash_balance': float(self.cash_balance),
                'invested_value': float(total_value - self.cash_balance),
                'total_pnl': float(self.get_unrealized_pnl() + self.get_realized_pnl()),
                'unrealized_pnl': float(self.get_unrealized_pnl()),
                'realized_pnl': float(self.get_realized_pnl()),
                'cash_percentage': float(self.cash_balance / total_value * 100) if total_value > 0 else 0,
                'number_of_positions': len(self.positions),
                'positions': {}
            }
            
            # Add position details
            for symbol, position in self.positions.items():
                summary['positions'][symbol] = {
                    'current_value': float(position.current_value),
                    'current_weight': float(position.current_weight),
                    'target_weight': float(position.target_weight),
                    'unrealized_pnl': float(position.unrealized_pnl),
                    'total_return': float(position.total_return),
                    'needs_rebalancing': position.needs_rebalancing(self.rebalance_threshold)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {
                'total_value': 0,
                'cash_balance': 0,
                'invested_value': 0,
                'total_pnl': 0,
                'unrealized_pnl': 0,
                'realized_pnl': 0,
                'cash_percentage': 0,
                'number_of_positions': 0,
                'positions': {}
            }