"""
Real-time Position Management System.

This module provides comprehensive position tracking and management including:

- Real-time position tracking across multiple symbols
- Position size monitoring and alerts
- PnL calculation and tracking
- Risk metrics calculation (VAR, drawdown, etc.)
- Position rebalancing and hedging
- Exposure analysis and reporting
- Portfolio-level position management
- Position-based risk controls and limits
- Automated position adjustments
- Cross-exchange position aggregation

The system maintains real-time position state and provides sophisticated
analytics for risk management and portfolio optimization.
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import statistics
import sqlite3
import json

from .order_management import Order, OrderSide, OrderStatus, OrderFill
from ..utils.logging import TradingLogger


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    HEDGED = "hedged"


@dataclass
class PositionSnapshot:
    """Snapshot of position state at a specific time."""
    
    timestamp: datetime
    symbol: str
    size: Decimal
    average_price: Decimal
    market_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_pnl: Decimal
    margin_used: Decimal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'size': str(self.size),
            'average_price': str(self.average_price),
            'market_price': str(self.market_price),
            'unrealized_pnl': str(self.unrealized_pnl),
            'realized_pnl': str(self.realized_pnl),
            'total_pnl': str(self.total_pnl),
            'margin_used': str(self.margin_used)
        }


@dataclass
class Position:
    """
    Comprehensive position tracking with real-time updates.
    
    This class maintains detailed position state including size,
    average price, PnL, and risk metrics.
    """
    
    symbol: str
    exchange: str
    
    # Position size and direction
    size: Decimal = Decimal('0')          # Positive = long, negative = short
    average_price: Decimal = Decimal('0') # Volume-weighted average price
    
    # Market data
    market_price: Decimal = Decimal('0')  # Current market price
    mark_price: Decimal = Decimal('0')    # Mark price for margin calculation
    
    # PnL tracking
    realized_pnl: Decimal = Decimal('0')   # Realized profit/loss
    unrealized_pnl: Decimal = Decimal('0') # Unrealized profit/loss
    
    # Margin and leverage
    margin_used: Decimal = Decimal('0')    # Margin currently used
    leverage: Decimal = Decimal('1')       # Effective leverage
    
    # Risk metrics
    max_size: Decimal = Decimal('0')       # Maximum position size reached
    drawdown: Decimal = Decimal('0')       # Current drawdown from peak
    max_drawdown: Decimal = Decimal('0')   # Maximum drawdown
    
    # Timestamps
    opened_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Trade tracking
    trades: List[Dict[str, Any]] = field(default_factory=list)
    total_volume: Decimal = Decimal('0')   # Total traded volume
    total_fees: Decimal = Decimal('0')     # Total fees paid
    
    # Position metadata
    strategy_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    notes: Optional[str] = None
    
    # History
    snapshots: List[PositionSnapshot] = field(default_factory=list)
    max_snapshots: int = 1000
    
    @property
    def side(self) -> PositionSide:
        """Get position side."""
        if self.size > 0:
            return PositionSide.LONG
        elif self.size < 0:
            return PositionSide.SHORT
        else:
            return PositionSide.FLAT
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.size > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.size == 0
    
    @property
    def abs_size(self) -> Decimal:
        """Get absolute position size."""
        return abs(self.size)
    
    @property
    def notional_value(self) -> Decimal:
        """Get notional value of position."""
        return self.abs_size * self.market_price
    
    @property
    def total_pnl(self) -> Decimal:
        """Get total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def pnl_percentage(self) -> float:
        """Get PnL as percentage of initial investment."""
        if self.average_price <= 0 or self.abs_size <= 0:
            return 0.0
        
        initial_value = self.abs_size * self.average_price
        return float(self.total_pnl / initial_value * 100)
    
    @property
    def return_on_margin(self) -> float:
        """Get return on margin used."""
        if self.margin_used <= 0:
            return 0.0
        
        return float(self.total_pnl / self.margin_used * 100)
    
    def update_market_price(self, price: Decimal, mark_price: Optional[Decimal] = None) -> None:
        """Update market price and recalculate unrealized PnL."""
        self.market_price = price
        if mark_price:
            self.mark_price = mark_price
        else:
            self.mark_price = price
        
        # Calculate unrealized PnL
        if self.size != 0 and self.average_price > 0:
            if self.is_long:
                self.unrealized_pnl = self.size * (self.market_price - self.average_price)
            else:
                self.unrealized_pnl = self.size * (self.average_price - self.market_price)
        else:
            self.unrealized_pnl = Decimal('0')
        
        # Update drawdown
        peak_value = max([s.total_pnl for s in self.snapshots[-50:]] + [self.total_pnl])
        current_drawdown = peak_value - self.total_pnl
        self.drawdown = max(Decimal('0'), current_drawdown)
        self.max_drawdown = max(self.max_drawdown, self.drawdown)
        
        self.last_updated = datetime.now()
    
    def add_trade(
        self,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        fee: Decimal = Decimal('0'),
        trade_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add a trade to the position."""
        timestamp = timestamp or datetime.now()
        
        # Record trade
        trade = {
            'trade_id': trade_id or str(uuid.uuid4()),
            'timestamp': timestamp.isoformat(),
            'side': side.value,
            'quantity': str(quantity),
            'price': str(price),
            'fee': str(fee),
            'notional': str(quantity * price)
        }
        self.trades.append(trade)
        
        # Update position size and average price
        old_size = self.size
        
        if side == OrderSide.BUY:
            new_size = self.size + quantity
        else:
            new_size = self.size - quantity
        
        # Calculate new average price
        if new_size == 0:
            # Position closed
            self.realized_pnl += self._calculate_realized_pnl(
                old_size, quantity, price, side
            )
            self.average_price = Decimal('0')
        elif (old_size >= 0 and new_size >= 0) or (old_size <= 0 and new_size <= 0):
            # Same direction trade - update average price
            if new_size != 0:
                total_cost = (old_size * self.average_price) + (
                    quantity * price if side == OrderSide.BUY else -quantity * price
                )
                self.average_price = abs(total_cost / new_size)
        else:
            # Position direction change
            if old_size != 0:
                # Calculate realized PnL for closed portion
                self.realized_pnl += self._calculate_realized_pnl(
                    old_size, min(abs(old_size), quantity), price, side
                )
            
            # Set new average price for remaining position
            if new_size != 0:
                self.average_price = price
        
        # Update position size
        self.size = new_size
        
        # Update tracking metrics
        self.total_volume += quantity
        self.total_fees += fee
        
        # Track maximum position size
        if abs(self.size) > abs(self.max_size):
            self.max_size = self.size
        
        # Set opened timestamp for first trade
        if self.opened_at is None and self.size != 0:
            self.opened_at = timestamp
        
        self.last_updated = timestamp
    
    def _calculate_realized_pnl(
        self,
        old_size: Decimal,
        trade_quantity: Decimal,
        trade_price: Decimal,
        trade_side: OrderSide
    ) -> Decimal:
        """Calculate realized PnL for a trade."""
        if old_size == 0:
            return Decimal('0')
        
        # Determine if this is a closing trade
        if (old_size > 0 and trade_side == OrderSide.SELL) or \
           (old_size < 0 and trade_side == OrderSide.BUY):
            # Closing trade
            closing_quantity = min(abs(old_size), trade_quantity)
            
            if old_size > 0:
                # Closing long position
                pnl = closing_quantity * (trade_price - self.average_price)
            else:
                # Closing short position
                pnl = closing_quantity * (self.average_price - trade_price)
            
            return pnl
        
        return Decimal('0')
    
    def take_snapshot(self) -> PositionSnapshot:
        """Take a snapshot of current position state."""
        snapshot = PositionSnapshot(
            timestamp=datetime.now(),
            symbol=self.symbol,
            size=self.size,
            average_price=self.average_price,
            market_price=self.market_price,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            total_pnl=self.total_pnl,
            margin_used=self.margin_used
        )
        
        # Add to history
        self.snapshots.append(snapshot)
        
        # Trim history if needed
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
        
        return snapshot
    
    def calculate_var(self, confidence: float = 0.95, periods: int = 50) -> Decimal:
        """Calculate Value at Risk based on recent PnL history."""
        if len(self.snapshots) < periods:
            return Decimal('0')
        
        # Get recent PnL changes
        recent_snapshots = self.snapshots[-periods:]
        pnl_changes = []
        
        for i in range(1, len(recent_snapshots)):
            pnl_change = recent_snapshots[i].total_pnl - recent_snapshots[i-1].total_pnl
            pnl_changes.append(float(pnl_change))
        
        if not pnl_changes:
            return Decimal('0')
        
        # Calculate VaR
        var_percentile = (1 - confidence) * 100
        var_value = np.percentile(pnl_changes, var_percentile)
        
        return Decimal(str(abs(var_value)))
    
    def get_performance_metrics(self, periods: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if periods:
            snapshots = self.snapshots[-periods:] if len(self.snapshots) >= periods else self.snapshots
        else:
            snapshots = self.snapshots
        
        if not snapshots:
            return {}
        
        # Calculate metrics
        pnl_values = [float(s.total_pnl) for s in snapshots]
        
        metrics = {
            'total_pnl': str(self.total_pnl),
            'realized_pnl': str(self.realized_pnl),
            'unrealized_pnl': str(self.unrealized_pnl),
            'pnl_percentage': self.pnl_percentage,
            'return_on_margin': self.return_on_margin,
            'max_drawdown': str(self.max_drawdown),
            'current_drawdown': str(self.drawdown),
            'total_volume': str(self.total_volume),
            'total_fees': str(self.total_fees),
            'trade_count': len(self.trades),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor(),
            'sharpe_ratio': self._calculate_sharpe_ratio(pnl_values),
            'var_95': str(self.calculate_var(0.95)),
            'var_99': str(self.calculate_var(0.99)),
            'holding_period': str(datetime.now() - self.opened_at) if self.opened_at else None
        }
        
        return metrics
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate based on individual trades."""
        if not self.trades:
            return 0.0
        
        # This is simplified - in practice, you'd need to track individual trade PnL
        # For now, we'll use a basic approximation
        winning_trades = sum(1 for trade in self.trades if float(trade.get('pnl', 0)) > 0)
        return winning_trades / len(self.trades) if self.trades else 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = Decimal('0')
        gross_loss = Decimal('0')
        
        # Calculate from PnL changes
        for i in range(1, len(self.snapshots)):
            pnl_change = self.snapshots[i].total_pnl - self.snapshots[i-1].total_pnl
            if pnl_change > 0:
                gross_profit += pnl_change
            else:
                gross_loss += abs(pnl_change)
        
        return float(gross_profit / gross_loss) if gross_loss > 0 else 0.0
    
    def _calculate_sharpe_ratio(self, pnl_values: List[float]) -> float:
        """Calculate Sharpe ratio from PnL series."""
        if len(pnl_values) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(pnl_values)):
            if pnl_values[i-1] != 0:
                ret = (pnl_values[i] - pnl_values[i-1]) / abs(pnl_values[i-1])
                returns.append(ret)
        
        if not returns:
            return 0.0
        
        mean_return = statistics.mean(returns)
        return_std = statistics.stdev(returns) if len(returns) > 1 else 0.0
        
        return mean_return / return_std if return_std > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary representation."""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'size': str(self.size),
            'side': self.side.value,
            'average_price': str(self.average_price),
            'market_price': str(self.market_price),
            'mark_price': str(self.mark_price),
            'realized_pnl': str(self.realized_pnl),
            'unrealized_pnl': str(self.unrealized_pnl),
            'total_pnl': str(self.total_pnl),
            'margin_used': str(self.margin_used),
            'leverage': str(self.leverage),
            'notional_value': str(self.notional_value),
            'pnl_percentage': self.pnl_percentage,
            'return_on_margin': self.return_on_margin,
            'max_drawdown': str(self.max_drawdown),
            'current_drawdown': str(self.drawdown),
            'total_volume': str(self.total_volume),
            'total_fees': str(self.total_fees),
            'trade_count': len(self.trades),
            'opened_at': self.opened_at.isoformat() if self.opened_at else None,
            'last_updated': self.last_updated.isoformat(),
            'strategy_id': self.strategy_id,
            'tags': self.tags,
            'notes': self.notes
        }


class PositionRiskMonitor:
    """
    Position-based risk monitoring and alerting.
    
    This class monitors positions for risk violations and
    triggers alerts or automatic actions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("PositionRiskMonitor")
        
        # Risk thresholds
        self.risk_limits = self.config['risk_limits']
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, Dict], None]] = []
        
        # Violation history
        self.violations: List[Dict[str, Any]] = []
        
    def _default_config(self) -> Dict:
        """Default risk monitoring configuration."""
        return {
            'risk_limits': {
                'max_position_size_usd': 100000,      # $100k max position
                'max_drawdown_pct': 10,                # 10% max drawdown
                'max_leverage': 5,                     # 5x max leverage
                'var_limit_pct': 5,                    # 5% portfolio VaR limit
                'concentration_limit_pct': 20,         # 20% max single position
                'correlation_limit': 0.8,              # Max correlation between positions
            },
            'alert_thresholds': {
                'warning_drawdown_pct': 5,             # 5% warning threshold
                'warning_var_pct': 3,                  # 3% VaR warning
                'warning_concentration_pct': 15,       # 15% concentration warning
            },
            'check_interval': 30,                      # Seconds between checks
            'enable_auto_actions': False,              # Auto-reduce positions
        }
    
    def check_position_risk(self, position: Position, portfolio_value: Decimal) -> List[Dict[str, Any]]:
        """Check position for risk violations."""
        violations = []
        
        # Position size check
        notional_value = position.notional_value
        max_position_usd = Decimal(str(self.risk_limits['max_position_size_usd']))
        
        if notional_value > max_position_usd:
            violations.append({
                'type': 'position_size',
                'severity': 'error',
                'message': f'Position size ${notional_value} exceeds limit ${max_position_usd}',
                'current_value': float(notional_value),
                'limit_value': float(max_position_usd),
                'symbol': position.symbol
            })
        
        # Drawdown check
        if position.max_drawdown > 0:
            drawdown_pct = float(position.drawdown / position.notional_value * 100)
            max_drawdown_pct = self.risk_limits['max_drawdown_pct']
            
            if drawdown_pct > max_drawdown_pct:
                violations.append({
                    'type': 'drawdown',
                    'severity': 'error',
                    'message': f'Drawdown {drawdown_pct:.2f}% exceeds limit {max_drawdown_pct}%',
                    'current_value': drawdown_pct,
                    'limit_value': max_drawdown_pct,
                    'symbol': position.symbol
                })
            elif drawdown_pct > self.config['alert_thresholds']['warning_drawdown_pct']:
                violations.append({
                    'type': 'drawdown',
                    'severity': 'warning',
                    'message': f'Drawdown {drawdown_pct:.2f}% approaching limit',
                    'current_value': drawdown_pct,
                    'limit_value': max_drawdown_pct,
                    'symbol': position.symbol
                })
        
        # Leverage check
        if position.leverage > Decimal(str(self.risk_limits['max_leverage'])):
            violations.append({
                'type': 'leverage',
                'severity': 'error',
                'message': f'Leverage {position.leverage}x exceeds limit {self.risk_limits["max_leverage"]}x',
                'current_value': float(position.leverage),
                'limit_value': self.risk_limits['max_leverage'],
                'symbol': position.symbol
            })
        
        # Concentration check
        if portfolio_value > 0:
            concentration_pct = float(notional_value / portfolio_value * 100)
            max_concentration = self.risk_limits['concentration_limit_pct']
            
            if concentration_pct > max_concentration:
                violations.append({
                    'type': 'concentration',
                    'severity': 'error',
                    'message': f'Position concentration {concentration_pct:.2f}% exceeds limit {max_concentration}%',
                    'current_value': concentration_pct,
                    'limit_value': max_concentration,
                    'symbol': position.symbol
                })
        
        # VaR check
        var_95 = position.calculate_var(0.95)
        if var_95 > 0 and notional_value > 0:
            var_pct = float(var_95 / notional_value * 100)
            var_limit = self.risk_limits['var_limit_pct']
            
            if var_pct > var_limit:
                violations.append({
                    'type': 'var',
                    'severity': 'error',
                    'message': f'VaR {var_pct:.2f}% exceeds limit {var_limit}%',
                    'current_value': var_pct,
                    'limit_value': var_limit,
                    'symbol': position.symbol
                })
        
        # Record violations
        for violation in violations:
            violation['timestamp'] = datetime.now().isoformat()
            violation['position_id'] = f"{position.exchange}:{position.symbol}"
            self.violations.append(violation)
        
        # Trigger alerts
        if violations:
            self._trigger_alerts(position, violations)
        
        return violations
    
    def _trigger_alerts(self, position: Position, violations: List[Dict[str, Any]]) -> None:
        """Trigger risk alerts."""
        for callback in self.alert_callbacks:
            try:
                callback(f"{position.exchange}:{position.symbol}", violations)
            except Exception as e:
                self.logger.error(f"Error in risk alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[str, List[Dict]], None]) -> None:
        """Add risk alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_violation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent violations."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_violations = [
            v for v in self.violations
            if datetime.fromisoformat(v['timestamp']) > cutoff_time
        ]
        
        # Group by type and severity
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_symbol = defaultdict(int)
        
        for violation in recent_violations:
            by_type[violation['type']] += 1
            by_severity[violation['severity']] += 1
            by_symbol[violation['symbol']] += 1
        
        return {
            'total_violations': len(recent_violations),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'by_symbol': dict(by_symbol),
            'time_period_hours': hours
        }


class PositionManager:
    """
    Comprehensive position management system.
    
    This class provides centralized position tracking, monitoring,
    and management across multiple symbols and exchanges.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("PositionManager")
        
        # Position storage
        self.positions: Dict[str, Position] = {}  # key: "exchange:symbol"
        
        # Risk monitor
        self.risk_monitor = PositionRiskMonitor(self.config.get('risk_monitor', {}))
        
        # Market data cache
        self.market_prices: Dict[str, Decimal] = {}
        self.mark_prices: Dict[str, Decimal] = {}
        
        # Position callbacks
        self.position_callbacks: List[Callable[[Position, str], None]] = []
        
        # Background monitoring
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Database for persistence
        self.db_path = self.config.get('database_path', 'positions.db')
        self._init_database()
        
    def _default_config(self) -> Dict:
        """Default configuration for position manager."""
        return {
            'auto_update_prices': True,
            'snapshot_interval': 300,        # 5 minutes
            'monitoring_interval': 30,       # 30 seconds
            'database_path': 'positions.db',
            'max_position_history': 10000,
            'enable_persistence': True,
            'risk_monitor': {}
        }
    
    def _init_database(self) -> None:
        """Initialize SQLite database for position persistence."""
        if not self.config['enable_persistence']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    data TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    data TEXT NOT NULL,
                    FOREIGN KEY (position_id) REFERENCES positions (id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions (symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_exchange ON positions (exchange)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_position_id ON position_snapshots (position_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON position_snapshots (timestamp)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize position database: {e}")
    
    def get_position(self, symbol: str, exchange: str = "default") -> Position:
        """Get or create position for symbol."""
        position_key = f"{exchange}:{symbol}"
        
        if position_key not in self.positions:
            position = Position(
                symbol=symbol,
                exchange=exchange
            )
            self.positions[position_key] = position
            
            # Load from database if available
            self._load_position_from_db(position_key)
            
            self.logger.info(f"Created new position: {position_key}")
        
        return self.positions[position_key]
    
    def update_position_from_fill(self, fill: OrderFill, order: Order) -> None:
        """Update position from order fill."""
        position = self.get_position(order.symbol, order.exchange or "default")
        
        # Convert fill to trade
        side = order.side
        quantity = fill.quantity
        price = fill.price
        fee = fill.fee
        
        # Add trade to position
        position.add_trade(
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            trade_id=fill.trade_id,
            timestamp=fill.timestamp
        )
        
        # Update market price if available
        market_price = self.market_prices.get(order.symbol)
        if market_price:
            position.update_market_price(market_price)
        
        # Notify callbacks
        self._notify_callbacks(position, 'trade_added')
        
        # Save to database
        self._save_position_to_db(position)
        
        self.logger.info(
            f"Updated position {position.symbol} from fill: "
            f"{side.value} {quantity} @ {price}"
        )
    
    def update_market_price(self, symbol: str, price: Decimal, mark_price: Optional[Decimal] = None) -> None:
        """Update market price for a symbol."""
        self.market_prices[symbol] = price
        if mark_price:
            self.mark_prices[symbol] = mark_price
        
        # Update all positions with this symbol
        for position_key, position in self.positions.items():
            if position.symbol == symbol:
                position.update_market_price(price, mark_price)
                self._notify_callbacks(position, 'price_updated')
    
    def update_all_market_prices(self, prices: Dict[str, Decimal]) -> None:
        """Update market prices for multiple symbols."""
        for symbol, price in prices.items():
            self.update_market_price(symbol, price)
    
    def get_positions(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        side: Optional[PositionSide] = None,
        min_size: Optional[Decimal] = None
    ) -> List[Position]:
        """Get positions with optional filters."""
        positions = []
        
        for position in self.positions.values():
            # Apply filters
            if symbol and position.symbol != symbol:
                continue
            if exchange and position.exchange != exchange:
                continue
            if side and position.side != side:
                continue
            if min_size and position.abs_size < min_size:
                continue
            
            positions.append(position)
        
        return positions
    
    def get_open_positions(self, exchange: Optional[str] = None) -> List[Position]:
        """Get all open (non-flat) positions."""
        return self.get_positions(
            exchange=exchange,
            min_size=Decimal('0.000001')  # Exclude flat positions
        )
    
    def get_portfolio_value(self, exchange: Optional[str] = None) -> Decimal:
        """Get total portfolio value (sum of position notional values)."""
        total_value = Decimal('0')
        
        for position in self.get_positions(exchange=exchange):
            total_value += position.notional_value
        
        return total_value
    
    def get_portfolio_pnl(self, exchange: Optional[str] = None) -> Dict[str, Decimal]:
        """Get portfolio-level PnL metrics."""
        total_realized = Decimal('0')
        total_unrealized = Decimal('0')
        
        for position in self.get_positions(exchange=exchange):
            total_realized += position.realized_pnl
            total_unrealized += position.unrealized_pnl
        
        return {
            'realized_pnl': total_realized,
            'unrealized_pnl': total_unrealized,
            'total_pnl': total_realized + total_unrealized
        }
    
    def get_exposure_analysis(self, exchange: Optional[str] = None) -> Dict[str, Any]:
        """Get portfolio exposure analysis."""
        positions = self.get_open_positions(exchange)
        
        if not positions:
            return {
                'total_long_exposure': Decimal('0'),
                'total_short_exposure': Decimal('0'),
                'net_exposure': Decimal('0'),
                'gross_exposure': Decimal('0'),
                'num_long_positions': 0,
                'num_short_positions': 0,
                'largest_position': None,
                'concentration_ratio': 0.0
            }
        
        # Calculate exposures
        long_exposure = sum(pos.notional_value for pos in positions if pos.is_long)
        short_exposure = sum(pos.notional_value for pos in positions if pos.is_short)
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure
        
        # Position counts
        num_long = sum(1 for pos in positions if pos.is_long)
        num_short = sum(1 for pos in positions if pos.is_short)
        
        # Largest position
        largest_position = max(positions, key=lambda p: p.notional_value)
        
        # Concentration ratio (largest position / total exposure)
        concentration_ratio = (
            float(largest_position.notional_value / gross_exposure)
            if gross_exposure > 0 else 0.0
        )
        
        return {
            'total_long_exposure': long_exposure,
            'total_short_exposure': short_exposure,
            'net_exposure': net_exposure,
            'gross_exposure': gross_exposure,
            'num_long_positions': num_long,
            'num_short_positions': num_short,
            'largest_position': {
                'symbol': largest_position.symbol,
                'notional_value': largest_position.notional_value,
                'side': largest_position.side.value
            },
            'concentration_ratio': concentration_ratio
        }
    
    def calculate_portfolio_var(
        self,
        confidence: float = 0.95,
        exchange: Optional[str] = None
    ) -> Decimal:
        """Calculate portfolio-level Value at Risk."""
        positions = self.get_open_positions(exchange)
        
        if not positions:
            return Decimal('0')
        
        # Simple approach: sum individual VaRs (ignores correlation)
        # In practice, you'd want to use a correlation matrix
        total_var = Decimal('0')
        
        for position in positions:
            position_var = position.calculate_var(confidence)
            total_var += position_var
        
        return total_var
    
    def check_risk_limits(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """Check all positions against risk limits."""
        all_violations = []
        portfolio_value = self.get_portfolio_value(exchange)
        
        for position in self.get_open_positions(exchange):
            violations = self.risk_monitor.check_position_risk(position, portfolio_value)
            all_violations.extend(violations)
        
        return all_violations
    
    def close_position(
        self,
        symbol: str,
        exchange: str = "default",
        reason: Optional[str] = None
    ) -> bool:
        """Mark position as closed (for manual closing)."""
        position_key = f"{exchange}:{symbol}"
        position = self.positions.get(position_key)
        
        if not position or position.is_flat:
            return False
        
        # For manual closing, we just zero out the position
        # In practice, this would trigger closing orders
        position.size = Decimal('0')
        position.unrealized_pnl = Decimal('0')
        position.last_updated = datetime.now()
        
        if reason:
            position.notes = f"{position.notes or ''} Closed: {reason}".strip()
        
        # Notify callbacks
        self._notify_callbacks(position, 'position_closed')
        
        # Save to database
        self._save_position_to_db(position)
        
        self.logger.info(f"Closed position {position_key}" + (f": {reason}" if reason else ""))
        
        return True
    
    def start_monitoring(self) -> None:
        """Start background position monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started position monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background position monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped position monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        last_snapshot_time = datetime.now()
        
        while self.is_monitoring:
            try:
                now = datetime.now()
                
                # Take snapshots periodically
                if (now - last_snapshot_time).total_seconds() >= self.config['snapshot_interval']:
                    for position in self.get_open_positions():
                        position.take_snapshot()
                        self._save_snapshot_to_db(position, position.snapshots[-1])
                    
                    last_snapshot_time = now
                
                # Check risk limits
                violations = self.check_risk_limits()
                if violations:
                    self.logger.warning(f"Risk violations detected: {len(violations)} violations")
                
                # Sleep until next check
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in position monitoring loop: {e}")
                time.sleep(10)  # Error backoff
    
    def _save_position_to_db(self, position: Position) -> None:
        """Save position to database."""
        if not self.config['enable_persistence']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            position_id = f"{position.exchange}:{position.symbol}"
            position_data = json.dumps(position.to_dict())
            
            cursor.execute("""
                INSERT OR REPLACE INTO positions (id, symbol, exchange, data, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (position_id, position.symbol, position.exchange, position_data, datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save position to database: {e}")
    
    def _save_snapshot_to_db(self, position: Position, snapshot: PositionSnapshot) -> None:
        """Save position snapshot to database."""
        if not self.config['enable_persistence']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            position_id = f"{position.exchange}:{position.symbol}"
            snapshot_data = json.dumps(snapshot.to_dict())
            
            cursor.execute("""
                INSERT INTO position_snapshots (position_id, timestamp, data)
                VALUES (?, ?, ?)
            """, (position_id, snapshot.timestamp, snapshot_data))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save position snapshot to database: {e}")
    
    def _load_position_from_db(self, position_key: str) -> None:
        """Load position from database."""
        if not self.config['enable_persistence']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT data FROM positions WHERE id = ?", (position_key,))
            result = cursor.fetchone()
            
            if result:
                position_data = json.loads(result[0])
                # Here you would reconstruct the position from saved data
                # This is simplified - in practice you'd need more sophisticated loading
                self.logger.info(f"Loaded position {position_key} from database")
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to load position from database: {e}")
    
    def add_position_callback(self, callback: Callable[[Position, str], None]) -> None:
        """Add position event callback."""
        self.position_callbacks.append(callback)
    
    def remove_position_callback(self, callback: Callable[[Position, str], None]) -> None:
        """Remove position event callback."""
        if callback in self.position_callbacks:
            self.position_callbacks.remove(callback)
    
    def _notify_callbacks(self, position: Position, event_type: str) -> None:
        """Notify position event callbacks."""
        for callback in self.position_callbacks:
            try:
                callback(position, event_type)
            except Exception as e:
                self.logger.error(f"Error in position callback: {e}")
    
    def get_position_summary(self, exchange: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive position summary."""
        positions = self.get_open_positions(exchange)
        portfolio_pnl = self.get_portfolio_pnl(exchange)
        exposure_analysis = self.get_exposure_analysis(exchange)
        portfolio_var = self.calculate_portfolio_var(exchange=exchange)
        
        return {
            'total_positions': len(positions),
            'portfolio_value': str(self.get_portfolio_value(exchange)),
            'portfolio_pnl': {k: str(v) for k, v in portfolio_pnl.items()},
            'exposure_analysis': {
                **exposure_analysis,
                'total_long_exposure': str(exposure_analysis['total_long_exposure']),
                'total_short_exposure': str(exposure_analysis['total_short_exposure']),
                'net_exposure': str(exposure_analysis['net_exposure']),
                'gross_exposure': str(exposure_analysis['gross_exposure'])
            },
            'portfolio_var_95': str(portfolio_var),
            'positions': [pos.to_dict() for pos in positions]
        }
    
    def export_positions(self, file_path: str, format: str = 'json') -> bool:
        """Export positions to file."""
        try:
            summary = self.get_position_summary()
            
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
            elif format.lower() == 'csv':
                # Convert to DataFrame and export
                df = pd.DataFrame([pos.to_dict() for pos in self.get_open_positions()])
                df.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported positions to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export positions: {e}")
            return False