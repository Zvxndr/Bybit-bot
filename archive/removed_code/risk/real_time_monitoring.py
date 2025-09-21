"""
Real-Time Risk Monitoring System for Cryptocurrency Trading.

This module provides real-time risk monitoring capabilities including:

- Live position and P&L tracking
- Dynamic risk limit monitoring
- Real-time correlation and volatility tracking
- Automated risk alerts and notifications
- Circuit breakers and emergency stops
- Performance attribution monitoring
- Risk budget tracking and allocation
- Real-time stress testing

The system is designed to operate continuously and provide
immediate alerts when risk thresholds are breached.
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import json
import sqlite3
from pathlib import Path

from ..utils.logging import TradingLogger
from .portfolio_analysis import RiskAlert, RiskLevel, PortfolioRiskMonitor


class MonitoringStatus(Enum):
    """Risk monitoring status."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class CircuitBreakerStatus(Enum):
    """Circuit breaker status."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker triggered
    HALF_OPEN = "half_open"  # Testing if system recovered


@dataclass
class PositionSnapshot:
    """Snapshot of position at a point in time."""
    
    timestamp: datetime
    asset: str
    quantity: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    entry_price: float
    current_price: float
    position_age: timedelta
    risk_contribution: float
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RiskSnapshot:
    """Snapshot of portfolio risk metrics at a point in time."""
    
    timestamp: datetime
    total_portfolio_value: float
    daily_pnl: float
    portfolio_var: float
    portfolio_volatility: float
    max_drawdown_current: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    tail_risk: float
    positions: Dict[str, PositionSnapshot] = field(default_factory=dict)
    active_alerts: List[RiskAlert] = field(default_factory=list)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class CircuitBreaker:
    """Circuit breaker configuration and state."""
    
    name: str
    metric_name: str
    threshold_value: float
    comparison: str  # 'greater', 'less', 'equal'
    action: str  # 'pause_trading', 'close_positions', 'alert_only'
    status: CircuitBreakerStatus = CircuitBreakerStatus.CLOSED
    trigger_count: int = 0
    last_triggered: Optional[datetime] = None
    cooldown_period: int = 300  # 5 minutes default
    
    def is_triggered(self, current_value: float) -> bool:
        """Check if circuit breaker should be triggered."""
        if self.status == CircuitBreakerStatus.OPEN:
            # Check if cooldown period has passed
            if (self.last_triggered and 
                (datetime.now() - self.last_triggered).total_seconds() > self.cooldown_period):
                self.status = CircuitBreakerStatus.HALF_OPEN
                return False
            return True
        
        # Check threshold
        if self.comparison == 'greater':
            return current_value > self.threshold_value
        elif self.comparison == 'less':
            return current_value < self.threshold_value
        else:  # equal
            return abs(current_value - self.threshold_value) < 1e-6
    
    def trigger(self) -> None:
        """Trigger the circuit breaker."""
        self.status = CircuitBreakerStatus.OPEN
        self.trigger_count += 1
        self.last_triggered = datetime.now()
    
    def reset(self) -> None:
        """Reset the circuit breaker."""
        self.status = CircuitBreakerStatus.CLOSED
        self.last_triggered = None


class RealTimeRiskMonitor:
    """
    Real-time risk monitoring system with continuous tracking and alerting.
    
    This class provides comprehensive real-time risk monitoring including:
    - Continuous position and P&L tracking
    - Real-time risk metric calculation
    - Automated alerting and notifications
    - Circuit breaker functionality
    - Historical risk data storage
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("RealTimeRiskMonitor")
        
        # Monitoring state
        self.status = MonitoringStatus.STOPPED
        self.is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Risk monitor components
        self.portfolio_monitor = PortfolioRiskMonitor(self.config.get('portfolio_monitor', {}))
        
        # Data storage
        self.risk_history: deque = deque(maxlen=self.config['max_history_size'])
        self.position_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config['max_position_history'])
        )
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._initialize_circuit_breakers()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[RiskAlert], None]] = []
        
        # Performance tracking
        self.performance_start_time = datetime.now()
        self.monitoring_stats = {
            'total_snapshots': 0,
            'alerts_generated': 0,
            'circuit_breaker_triggers': 0,
            'last_update': None
        }
        
        # Database for persistence
        self.db_path = Path(self.config['database_path'])
        self._initialize_database()
        
    def _default_config(self) -> Dict:
        """Default configuration for real-time risk monitor."""
        return {
            'monitoring_interval': 30,       # Seconds between risk checks
            'max_history_size': 10000,       # Maximum risk snapshots to keep
            'max_position_history': 1000,    # Maximum position snapshots per asset
            'database_path': 'data/risk_monitoring.db',
            'enable_circuit_breakers': True,
            'enable_persistence': True,
            'alert_cooldown': 60,            # Seconds between similar alerts
            'risk_limits': {
                'max_daily_loss': 0.05,      # 5% max daily loss
                'max_portfolio_var': 0.03,   # 3% daily VaR limit
                'max_position_size': 0.2,    # 20% max position size
                'max_correlation': 0.8,      # Maximum average correlation
                'min_liquidity_ratio': 0.1,  # Minimum liquid assets
                'max_drawdown': 0.15,        # 15% maximum drawdown
            },
            'circuit_breakers': {
                'daily_loss_limit': {
                    'threshold': 0.05,
                    'action': 'pause_trading'
                },
                'var_limit': {
                    'threshold': 0.03,
                    'action': 'alert_only'
                },
                'drawdown_limit': {
                    'threshold': 0.15,
                    'action': 'close_positions'
                }
            },
            'portfolio_monitor': {}
        }
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers from configuration."""
        if not self.config['enable_circuit_breakers']:
            return
        
        for name, config in self.config['circuit_breakers'].items():
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                metric_name=name.replace('_limit', ''),
                threshold_value=config['threshold'],
                comparison='greater',
                action=config['action']
            )
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for risk data persistence."""
        if not self.config['enable_persistence']:
            return
        
        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create tables
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS risk_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    portfolio_value REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    portfolio_var REAL NOT NULL,
                    portfolio_volatility REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    correlation_risk REAL NOT NULL,
                    concentration_risk REAL NOT NULL,
                    liquidity_risk REAL NOT NULL,
                    tail_risk REAL NOT NULL,
                    alert_count INTEGER NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    market_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    risk_contribution REAL NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    asset TEXT,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    message TEXT NOT NULL,
                    action_required BOOLEAN NOT NULL
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_risk_timestamp ON risk_snapshots(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_position_timestamp ON position_snapshots(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON risk_alerts(timestamp)')
    
    def start_monitoring(self) -> None:
        """Start real-time risk monitoring."""
        if self.is_monitoring:
            self.logger.warning("Risk monitoring is already running")
            return
        
        self.logger.info("Starting real-time risk monitoring")
        self.status = MonitoringStatus.ACTIVE
        self.is_monitoring = True
        self._stop_event.clear()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        self.performance_start_time = datetime.now()
    
    def stop_monitoring(self) -> None:
        """Stop real-time risk monitoring."""
        if not self.is_monitoring:
            self.logger.warning("Risk monitoring is not running")
            return
        
        self.logger.info("Stopping real-time risk monitoring")
        self.status = MonitoringStatus.STOPPED
        self.is_monitoring = False
        self._stop_event.set()
        
        # Wait for monitoring thread to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
    
    def pause_monitoring(self) -> None:
        """Pause risk monitoring temporarily."""
        if self.status == MonitoringStatus.ACTIVE:
            self.status = MonitoringStatus.PAUSED
            self.logger.info("Risk monitoring paused")
    
    def resume_monitoring(self) -> None:
        """Resume risk monitoring."""
        if self.status == MonitoringStatus.PAUSED:
            self.status = MonitoringStatus.ACTIVE
            self.logger.info("Risk monitoring resumed")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        self.logger.info("Risk monitoring loop started")
        
        while self.is_monitoring and not self._stop_event.is_set():
            try:
                if self.status == MonitoringStatus.ACTIVE:
                    # Perform risk monitoring cycle
                    self._perform_monitoring_cycle()
                
                # Wait for next cycle
                self._stop_event.wait(self.config['monitoring_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.status = MonitoringStatus.ERROR
                time.sleep(5)  # Brief pause before retrying
        
        self.logger.info("Risk monitoring loop stopped")
    
    def update_positions(
        self,
        positions: Dict[str, Dict[str, Any]],
        market_prices: Dict[str, float]
    ) -> RiskSnapshot:
        """
        Update positions and calculate current risk metrics.
        
        Args:
            positions: Dictionary of current positions
            market_prices: Current market prices
            
        Returns:
            RiskSnapshot with current risk state
        """
        current_time = datetime.now()
        
        # Create position snapshots
        position_snapshots = {}
        total_portfolio_value = 0.0
        
        for asset, position_data in positions.items():
            current_price = market_prices.get(asset, position_data.get('entry_price', 0))
            quantity = position_data.get('quantity', 0)
            entry_price = position_data.get('entry_price', current_price)
            
            market_value = quantity * current_price
            unrealized_pnl = (current_price - entry_price) * quantity
            realized_pnl = position_data.get('realized_pnl', 0)
            
            position_snapshot = PositionSnapshot(
                timestamp=current_time,
                asset=asset,
                quantity=quantity,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                entry_price=entry_price,
                current_price=current_price,
                position_age=current_time - position_data.get('entry_time', current_time),
                risk_contribution=0.0  # Will be calculated below
            )
            
            position_snapshots[asset] = position_snapshot
            total_portfolio_value += market_value
        
        # Calculate risk contributions
        for asset, snapshot in position_snapshots.items():
            if total_portfolio_value > 0:
                snapshot.risk_contribution = abs(snapshot.market_value) / total_portfolio_value
        
        # Calculate portfolio-level risk metrics
        portfolio_weights = {
            asset: snapshot.market_value / total_portfolio_value if total_portfolio_value > 0 else 0
            for asset, snapshot in position_snapshots.items()
        }
        
        # Calculate daily P&L
        daily_pnl = sum(snapshot.unrealized_pnl + snapshot.realized_pnl 
                       for snapshot in position_snapshots.values())
        daily_pnl_pct = daily_pnl / total_portfolio_value if total_portfolio_value > 0 else 0
        
        # Calculate additional risk metrics (simplified for real-time)
        portfolio_var = self._estimate_real_time_var(position_snapshots, market_prices)
        portfolio_volatility = self._estimate_real_time_volatility(position_snapshots)
        max_drawdown = self._calculate_current_drawdown()
        correlation_risk = self._estimate_correlation_risk(portfolio_weights)
        concentration_risk = self._calculate_concentration_risk(portfolio_weights)
        liquidity_risk = self._estimate_liquidity_risk(position_snapshots)
        tail_risk = self._estimate_tail_risk(position_snapshots)
        
        # Generate alerts
        active_alerts = self._generate_real_time_alerts(
            total_portfolio_value, daily_pnl_pct, portfolio_var, 
            concentration_risk, correlation_risk, position_snapshots
        )
        
        # Create risk snapshot
        risk_snapshot = RiskSnapshot(
            timestamp=current_time,
            total_portfolio_value=total_portfolio_value,
            daily_pnl=daily_pnl_pct,
            portfolio_var=portfolio_var,
            portfolio_volatility=portfolio_volatility,
            max_drawdown_current=max_drawdown,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk,
            tail_risk=tail_risk,
            positions=position_snapshots,
            active_alerts=active_alerts
        )
        
        # Store snapshot
        self._store_risk_snapshot(risk_snapshot)
        
        # Check circuit breakers
        self._check_circuit_breakers(risk_snapshot)
        
        # Update monitoring stats
        self.monitoring_stats['total_snapshots'] += 1
        self.monitoring_stats['alerts_generated'] += len(active_alerts)
        self.monitoring_stats['last_update'] = current_time
        
        return risk_snapshot
    
    def _perform_monitoring_cycle(self) -> None:
        """Perform a single monitoring cycle."""
        # This would be called by external system with current positions and prices
        # For now, we'll just update the last monitoring time
        self.monitoring_stats['last_update'] = datetime.now()
    
    def _estimate_real_time_var(
        self,
        positions: Dict[str, PositionSnapshot],
        market_prices: Dict[str, float]
    ) -> float:
        """Estimate VaR in real-time using position data."""
        # Simplified VaR estimation for real-time use
        # In practice, would use more sophisticated models
        
        total_value = sum(pos.market_value for pos in positions.values())
        if total_value == 0:
            return 0.0
        
        # Assume average crypto volatility of 60% annually
        daily_vol = 0.6 / np.sqrt(252)
        
        # Approximate VaR using normal distribution (95% confidence)
        var_estimate = total_value * daily_vol * 1.645  # 95% VaR
        
        return var_estimate / total_value if total_value > 0 else 0.0
    
    def _estimate_real_time_volatility(self, positions: Dict[str, PositionSnapshot]) -> float:
        """Estimate portfolio volatility in real-time."""
        # Simplified volatility estimation
        # In practice, would use historical data and correlation matrix
        
        if not positions:
            return 0.0
        
        # Weighted average of assumed asset volatilities
        asset_volatilities = {
            'BTC': 0.6, 'ETH': 0.7, 'ADA': 0.8, 'SOL': 0.9,
            'DOGE': 1.2, 'MATIC': 0.85, 'LINK': 0.75
        }
        
        total_value = sum(pos.market_value for pos in positions.values())
        if total_value == 0:
            return 0.0
        
        weighted_vol = 0.0
        for asset, position in positions.items():
            weight = position.market_value / total_value
            asset_vol = asset_volatilities.get(asset.upper(), 0.8)  # Default 80%
            weighted_vol += weight * asset_vol
        
        return weighted_vol
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from historical data."""
        if len(self.risk_history) < 2:
            return 0.0
        
        # Get historical portfolio values
        values = [snapshot.total_portfolio_value for snapshot in self.risk_history]
        
        if not values:
            return 0.0
        
        # Calculate running maximum and current drawdown
        running_max = max(values)
        current_value = values[-1]
        
        if running_max == 0:
            return 0.0
        
        drawdown = (running_max - current_value) / running_max
        return max(0, drawdown)
    
    def _estimate_correlation_risk(self, portfolio_weights: Dict[str, float]) -> float:
        """Estimate correlation risk for real-time monitoring."""
        # Simplified correlation risk estimation
        # Assume higher correlation during market stress
        
        if len(portfolio_weights) <= 1:
            return 0.0
        
        # Estimate based on asset types
        crypto_majors = {'BTC', 'ETH'}
        alts = set(portfolio_weights.keys()) - crypto_majors
        
        major_weight = sum(w for asset, w in portfolio_weights.items() 
                          if asset.upper() in crypto_majors)
        alt_weight = sum(w for asset, w in portfolio_weights.items() 
                        if asset.upper() in alts)
        
        # Higher correlation risk when dominated by alts
        correlation_risk = min(0.9, alt_weight * 0.8 + major_weight * 0.4)
        
        return correlation_risk
    
    def _calculate_concentration_risk(self, portfolio_weights: Dict[str, float]) -> float:
        """Calculate concentration risk using Herfindahl-Hirschman Index."""
        if not portfolio_weights:
            return 0.0
        
        # HHI calculation
        hhi = sum(weight ** 2 for weight in portfolio_weights.values())
        
        return hhi
    
    def _estimate_liquidity_risk(self, positions: Dict[str, PositionSnapshot]) -> float:
        """Estimate liquidity risk based on position sizes and assets."""
        if not positions:
            return 0.0
        
        # Simplified liquidity risk based on asset types and sizes
        liquidity_scores = {
            'BTC': 0.1, 'ETH': 0.15, 'ADA': 0.3, 'SOL': 0.25,
            'DOGE': 0.4, 'MATIC': 0.35, 'LINK': 0.3
        }
        
        total_value = sum(pos.market_value for pos in positions.values())
        if total_value == 0:
            return 0.0
        
        weighted_liquidity_risk = 0.0
        for asset, position in positions.items():
            weight = position.market_value / total_value
            asset_liquidity_risk = liquidity_scores.get(asset.upper(), 0.5)
            weighted_liquidity_risk += weight * asset_liquidity_risk
        
        return weighted_liquidity_risk
    
    def _estimate_tail_risk(self, positions: Dict[str, PositionSnapshot]) -> float:
        """Estimate tail risk for portfolio."""
        # Simplified tail risk estimation
        # Higher tail risk for more volatile/exotic assets
        
        if not positions:
            return 0.0
        
        tail_risk_scores = {
            'BTC': 0.2, 'ETH': 0.25, 'ADA': 0.3, 'SOL': 0.35,
            'DOGE': 0.5, 'MATIC': 0.4, 'LINK': 0.3
        }
        
        total_value = sum(pos.market_value for pos in positions.values())
        if total_value == 0:
            return 0.0
        
        weighted_tail_risk = 0.0
        for asset, position in positions.items():
            weight = position.market_value / total_value
            asset_tail_risk = tail_risk_scores.get(asset.upper(), 0.4)
            weighted_tail_risk += weight * asset_tail_risk
        
        return weighted_tail_risk
    
    def _generate_real_time_alerts(
        self,
        portfolio_value: float,
        daily_pnl_pct: float,
        portfolio_var: float,
        concentration_risk: float,
        correlation_risk: float,
        positions: Dict[str, PositionSnapshot]
    ) -> List[RiskAlert]:
        """Generate real-time risk alerts."""
        alerts = []
        current_time = datetime.now()
        
        # Daily loss limit
        if daily_pnl_pct < -self.config['risk_limits']['max_daily_loss']:
            alerts.append(RiskAlert(
                alert_type='daily_loss_limit',
                risk_level=RiskLevel.HIGH,
                asset=None,
                metric_name='daily_pnl',
                current_value=daily_pnl_pct,
                threshold_value=-self.config['risk_limits']['max_daily_loss'],
                message=f"Daily loss ({daily_pnl_pct:.2%}) exceeds limit ({self.config['risk_limits']['max_daily_loss']:.2%})",
                action_required=True,
                timestamp=current_time
            ))
        
        # VaR limit
        if portfolio_var > self.config['risk_limits']['max_portfolio_var']:
            alerts.append(RiskAlert(
                alert_type='var_limit',
                risk_level=RiskLevel.MEDIUM,
                asset=None,
                metric_name='portfolio_var',
                current_value=portfolio_var,
                threshold_value=self.config['risk_limits']['max_portfolio_var'],
                message=f"Portfolio VaR ({portfolio_var:.2%}) exceeds limit ({self.config['risk_limits']['max_portfolio_var']:.2%})",
                action_required=True,
                timestamp=current_time
            ))
        
        # Position size limits
        for asset, position in positions.items():
            position_weight = position.market_value / portfolio_value if portfolio_value > 0 else 0
            if position_weight > self.config['risk_limits']['max_position_size']:
                alerts.append(RiskAlert(
                    alert_type='position_size_limit',
                    risk_level=RiskLevel.MEDIUM,
                    asset=asset,
                    metric_name='position_weight',
                    current_value=position_weight,
                    threshold_value=self.config['risk_limits']['max_position_size'],
                    message=f"Position {asset} ({position_weight:.2%}) exceeds size limit ({self.config['risk_limits']['max_position_size']:.2%})",
                    action_required=True,
                    timestamp=current_time
                ))
        
        # Concentration risk
        if concentration_risk > 0.5:  # High concentration threshold
            alerts.append(RiskAlert(
                alert_type='concentration_risk',
                risk_level=RiskLevel.MEDIUM,
                asset=None,
                metric_name='concentration_risk',
                current_value=concentration_risk,
                threshold_value=0.5,
                message=f"Portfolio concentration risk is high ({concentration_risk:.2f})",
                action_required=False,
                timestamp=current_time
            ))
        
        # Correlation risk
        if correlation_risk > self.config['risk_limits']['max_correlation']:
            alerts.append(RiskAlert(
                alert_type='correlation_risk',
                risk_level=RiskLevel.MEDIUM,
                asset=None,
                metric_name='correlation_risk',
                current_value=correlation_risk,
                threshold_value=self.config['risk_limits']['max_correlation'],
                message=f"Portfolio correlation risk is high ({correlation_risk:.2f})",
                action_required=False,
                timestamp=current_time
            ))
        
        # Filter alerts based on cooldown
        filtered_alerts = self._filter_alerts_by_cooldown(alerts)
        
        # Notify alert callbacks
        for alert in filtered_alerts:
            self._notify_alert_callbacks(alert)
        
        return filtered_alerts
    
    def _filter_alerts_by_cooldown(self, alerts: List[RiskAlert]) -> List[RiskAlert]:
        """Filter alerts based on cooldown period."""
        if len(self.risk_history) == 0:
            return alerts
        
        filtered_alerts = []
        current_time = datetime.now()
        
        for alert in alerts:
            # Check if similar alert was recently generated
            is_duplicate = False
            
            for recent_snapshot in reversed(list(self.risk_history)[-10:]):  # Check last 10 snapshots
                for recent_alert in recent_snapshot.active_alerts:
                    time_diff = (current_time - recent_alert.timestamp).total_seconds()
                    
                    if (time_diff < self.config['alert_cooldown'] and
                        alert.alert_type == recent_alert.alert_type and
                        alert.asset == recent_alert.asset):
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    break
            
            if not is_duplicate:
                filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def _notify_alert_callbacks(self, alert: RiskAlert) -> None:
        """Notify registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def _check_circuit_breakers(self, risk_snapshot: RiskSnapshot) -> None:
        """Check and trigger circuit breakers if necessary."""
        if not self.config['enable_circuit_breakers']:
            return
        
        current_time = datetime.now()
        
        for name, breaker in self.circuit_breakers.items():
            # Get current value for the metric
            current_value = self._get_metric_value(risk_snapshot, breaker.metric_name)
            
            if breaker.is_triggered(current_value):
                if breaker.status == CircuitBreakerStatus.CLOSED:
                    # Trigger circuit breaker
                    breaker.trigger()
                    self.monitoring_stats['circuit_breaker_triggers'] += 1
                    
                    self.logger.critical(f"Circuit breaker '{name}' triggered! "
                                       f"Current value: {current_value}, "
                                       f"Threshold: {breaker.threshold_value}")
                    
                    # Execute circuit breaker action
                    self._execute_circuit_breaker_action(breaker, risk_snapshot)
                    
                    # Generate critical alert
                    alert = RiskAlert(
                        alert_type='circuit_breaker',
                        risk_level=RiskLevel.EXTREME,
                        asset=None,
                        metric_name=breaker.metric_name,
                        current_value=current_value,
                        threshold_value=breaker.threshold_value,
                        message=f"Circuit breaker '{name}' triggered! Action: {breaker.action}",
                        action_required=True,
                        timestamp=current_time
                    )
                    
                    self._notify_alert_callbacks(alert)
    
    def _get_metric_value(self, risk_snapshot: RiskSnapshot, metric_name: str) -> float:
        """Get metric value from risk snapshot."""
        metric_mapping = {
            'daily_loss': -risk_snapshot.daily_pnl,  # Convert to positive for loss
            'var': risk_snapshot.portfolio_var,
            'drawdown': risk_snapshot.max_drawdown_current,
            'concentration': risk_snapshot.concentration_risk,
            'correlation': risk_snapshot.correlation_risk
        }
        
        return metric_mapping.get(metric_name, 0.0)
    
    def _execute_circuit_breaker_action(
        self,
        breaker: CircuitBreaker,
        risk_snapshot: RiskSnapshot
    ) -> None:
        """Execute circuit breaker action."""
        if breaker.action == 'pause_trading':
            self.pause_monitoring()
            self.logger.critical(f"Trading paused due to circuit breaker: {breaker.name}")
        
        elif breaker.action == 'close_positions':
            self.logger.critical(f"Close positions signal due to circuit breaker: {breaker.name}")
            # In practice, this would trigger position closing logic
        
        elif breaker.action == 'alert_only':
            self.logger.warning(f"Circuit breaker alert only: {breaker.name}")
        
        # Additional actions could be implemented here
    
    def _store_risk_snapshot(self, risk_snapshot: RiskSnapshot) -> None:
        """Store risk snapshot in memory and database."""
        # Store in memory
        self.risk_history.append(risk_snapshot)
        
        # Store position snapshots by asset
        for asset, position in risk_snapshot.positions.items():
            self.position_history[asset].append(position)
        
        # Store in database if enabled
        if self.config['enable_persistence']:
            self._persist_risk_snapshot(risk_snapshot)
    
    def _persist_risk_snapshot(self, risk_snapshot: RiskSnapshot) -> None:
        """Persist risk snapshot to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store risk snapshot
                conn.execute('''
                    INSERT INTO risk_snapshots (
                        timestamp, portfolio_value, daily_pnl, portfolio_var,
                        portfolio_volatility, max_drawdown, correlation_risk,
                        concentration_risk, liquidity_risk, tail_risk, alert_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    risk_snapshot.timestamp.isoformat(),
                    risk_snapshot.total_portfolio_value,
                    risk_snapshot.daily_pnl,
                    risk_snapshot.portfolio_var,
                    risk_snapshot.portfolio_volatility,
                    risk_snapshot.max_drawdown_current,
                    risk_snapshot.correlation_risk,
                    risk_snapshot.concentration_risk,
                    risk_snapshot.liquidity_risk,
                    risk_snapshot.tail_risk,
                    len(risk_snapshot.active_alerts)
                ))
                
                # Store position snapshots
                for position in risk_snapshot.positions.values():
                    conn.execute('''
                        INSERT INTO position_snapshots (
                            timestamp, asset, quantity, market_value, unrealized_pnl,
                            realized_pnl, entry_price, current_price, risk_contribution
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        position.timestamp.isoformat(),
                        position.asset,
                        position.quantity,
                        position.market_value,
                        position.unrealized_pnl,
                        position.realized_pnl,
                        position.entry_price,
                        position.current_price,
                        position.risk_contribution
                    ))
                
                # Store alerts
                for alert in risk_snapshot.active_alerts:
                    conn.execute('''
                        INSERT INTO risk_alerts (
                            timestamp, alert_type, risk_level, asset, metric_name,
                            current_value, threshold_value, message, action_required
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        alert.timestamp.isoformat(),
                        alert.alert_type,
                        alert.risk_level.value,
                        alert.asset,
                        alert.metric_name,
                        alert.current_value,
                        alert.threshold_value,
                        alert.message,
                        alert.action_required
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error persisting risk snapshot: {e}")
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]) -> None:
        """Add callback function for risk alerts."""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[RiskAlert], None]) -> None:
        """Remove alert callback function."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def get_current_risk_status(self) -> Dict[str, Any]:
        """Get current risk monitoring status and metrics."""
        if not self.risk_history:
            return {'status': 'no_data'}
        
        latest_snapshot = self.risk_history[-1]
        
        return {
            'monitoring_status': self.status.value,
            'last_update': latest_snapshot.timestamp.isoformat(),
            'portfolio_value': latest_snapshot.total_portfolio_value,
            'daily_pnl': latest_snapshot.daily_pnl,
            'portfolio_var': latest_snapshot.portfolio_var,
            'max_drawdown': latest_snapshot.max_drawdown_current,
            'active_alerts': len(latest_snapshot.active_alerts),
            'circuit_breakers': {
                name: {
                    'status': breaker.status.value,
                    'trigger_count': breaker.trigger_count,
                    'last_triggered': breaker.last_triggered.isoformat() if breaker.last_triggered else None
                }
                for name, breaker in self.circuit_breakers.items()
            },
            'monitoring_stats': self.monitoring_stats
        }
    
    def get_risk_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[RiskSnapshot]:
        """Get historical risk snapshots."""
        snapshots = list(self.risk_history)
        
        # Filter by time range
        if start_time:
            snapshots = [s for s in snapshots if s.timestamp >= start_time]
        if end_time:
            snapshots = [s for s in snapshots if s.timestamp <= end_time]
        
        # Apply limit
        if limit:
            snapshots = snapshots[-limit:]
        
        return snapshots
    
    def reset_circuit_breaker(self, breaker_name: str) -> bool:
        """Reset a specific circuit breaker."""
        if breaker_name in self.circuit_breakers:
            self.circuit_breakers[breaker_name].reset()
            self.logger.info(f"Circuit breaker '{breaker_name}' reset")
            return True
        return False
    
    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self.circuit_breakers.values():
            breaker.reset()
        self.logger.info("All circuit breakers reset")
    
    def export_risk_data(
        self,
        filepath: str,
        format: str = 'csv',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """Export risk monitoring data to file."""
        try:
            snapshots = self.get_risk_history(start_time, end_time)
            
            if not snapshots:
                self.logger.warning("No risk data to export")
                return False
            
            # Convert to DataFrame
            data = []
            for snapshot in snapshots:
                data.append({
                    'timestamp': snapshot.timestamp,
                    'portfolio_value': snapshot.total_portfolio_value,
                    'daily_pnl': snapshot.daily_pnl,
                    'portfolio_var': snapshot.portfolio_var,
                    'portfolio_volatility': snapshot.portfolio_volatility,
                    'max_drawdown': snapshot.max_drawdown_current,
                    'correlation_risk': snapshot.correlation_risk,
                    'concentration_risk': snapshot.concentration_risk,
                    'liquidity_risk': snapshot.liquidity_risk,
                    'tail_risk': snapshot.tail_risk,
                    'alert_count': len(snapshot.active_alerts)
                })
            
            df = pd.DataFrame(data)
            
            # Export based on format
            if format.lower() == 'csv':
                df.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                df.to_json(filepath, orient='records', date_format='iso')
            elif format.lower() == 'parquet':
                df.to_parquet(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Risk data exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting risk data: {e}")
            return False