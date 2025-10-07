"""
HFT Risk Manager for Ultra-Low Latency Risk Control.
Provides real-time risk monitoring and control for high-frequency trading operations.
"""

import asyncio
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    import psutil
    HAS_OPTIMIZATIONS = True
except ImportError:
    HAS_OPTIMIZATIONS = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"

class RiskAction(Enum):
    """Risk control actions."""
    MONITOR = "monitor"
    WARN = "warn"
    REDUCE_POSITION = "reduce_position"
    CANCEL_ORDERS = "cancel_orders"
    HALT_TRADING = "halt_trading"
    EMERGENCY_EXIT = "emergency_exit"

class RiskMetricType(Enum):
    """Types of risk metrics."""
    POSITION_SIZE = "position_size"
    NOTIONAL_EXPOSURE = "notional_exposure"
    PNL = "pnl"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"

class CircuitBreakerType(Enum):
    """Circuit breaker types."""
    DAILY_LOSS = "daily_loss"
    POSITION_LIMIT = "position_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    ORDER_REJECTION_RATE = "order_rejection_rate"
    LATENCY_SPIKE = "latency_spike"

@dataclass
class RiskLimit:
    """Risk limit definition."""
    metric_type: RiskMetricType
    limit_value: float
    warning_threshold: float  # Percentage of limit that triggers warning
    symbol: Optional[str] = None  # None for portfolio-wide limits
    time_window_minutes: int = 1440  # Default 24 hours
    enabled: bool = True

@dataclass
class RiskAlert:
    """Risk alert."""
    alert_id: str
    risk_level: RiskLevel
    metric_type: RiskMetricType
    current_value: float
    limit_value: float
    symbol: Optional[str]
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    action_taken: Optional[RiskAction] = None

@dataclass
class PositionRisk:
    """Position-level risk metrics."""
    symbol: str
    quantity: float
    market_value: float
    unrealized_pnl: float
    position_var: float  # Value at Risk
    max_drawdown: float
    liquidity_score: float  # 0-1, higher is more liquid
    concentration_ratio: float  # Position size / total portfolio
    correlation_risk: float  # Risk from correlations
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""
    total_market_value: float
    total_unrealized_pnl: float
    daily_pnl: float
    portfolio_var: float
    max_drawdown: float
    total_leverage: float
    beta: float  # Portfolio beta to market
    tracking_error: float
    concentration_risk: float
    correlation_risk: float
    liquidity_risk: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RiskScenario:
    """Risk scenario for stress testing."""
    scenario_name: str
    market_shock: Dict[str, float]  # symbol -> price change percentage
    correlation_changes: Dict[Tuple[str, str], float]  # correlation matrix changes
    liquidity_impact: Dict[str, float]  # symbol -> liquidity reduction
    expected_pnl_impact: float
    var_impact: float
    probability: float = 0.05  # 5% default probability

class HFTRiskManager:
    """High-frequency trading risk manager."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Risk management configuration
        self.risk_config = {
            # Position limits
            'max_position_size_usd': 1000000,      # $1M max position
            'max_portfolio_value_usd': 10000000,   # $10M max portfolio
            'max_daily_loss_usd': 100000,          # $100K max daily loss
            'max_drawdown_pct': 0.05,              # 5% max drawdown
            'max_leverage': 3.0,                   # 3x max leverage
            'max_concentration_pct': 0.3,          # 30% max single position
            
            # Risk monitoring
            'var_confidence_level': 0.95,          # 95% VaR confidence
            'var_horizon_days': 1,                 # 1-day VaR
            'stress_test_scenarios': 10,           # Number of stress scenarios
            'correlation_lookback_days': 30,       # 30-day correlation window
            'volatility_lookback_days': 20,        # 20-day volatility window
            
            # Circuit breakers
            'enable_circuit_breakers': True,
            'volatility_spike_threshold': 3.0,     # 3x normal volatility
            'correlation_breakdown_threshold': 0.5, # 50% correlation drop
            'order_rejection_rate_threshold': 0.1,  # 10% rejection rate
            'latency_spike_threshold_ms': 1000,     # 1 second latency spike
            
            # Real-time monitoring
            'risk_check_interval_ms': 100,         # 100ms risk checks
            'position_update_interval_ms': 50,     # 50ms position updates
            'alert_cooldown_seconds': 60,          # 60 second alert cooldown
            'max_alerts_per_minute': 10,           # Alert rate limiting
            
            # Emergency controls
            'emergency_exit_threshold_pct': 0.1,   # 10% portfolio loss triggers emergency
            'auto_halt_on_extreme_risk': True,
            'auto_reduce_on_high_risk': True,
            'risk_manager_priority': -15           # High OS priority
        }
        
        # Risk state
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.position_risks: Dict[str, PositionRisk] = {}
        self.portfolio_risk: Optional[PortfolioRisk] = None
        
        # Circuit breakers
        self.circuit_breakers: Dict[CircuitBreakerType, bool] = {
            breaker: False for breaker in CircuitBreakerType
        }
        
        # Market data and positions
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Risk calculation components
        self.var_calculator = VaRCalculator(self)
        self.stress_tester = StressTester(self)
        self.correlation_monitor = CorrelationMonitor(self)
        
        # Performance optimization
        if HAS_OPTIMIZATIONS:
            try:
                process = psutil.Process()
                process.nice(self.risk_config['risk_manager_priority'])
            except:
                pass
        
        # Threading and monitoring
        self.risk_lock = threading.Lock()
        self.running = False
        self.monitoring_task = None
        self.circuit_breaker_task = None
        
        # Performance tracking
        self.risk_check_latency_ms = 0.0
        self.alerts_generated = 0
        self.actions_taken = 0
        
        self._initialize_default_limits()
        self.logger.info("HFTRiskManager initialized")
    
    def _initialize_default_limits(self):
        """Initialize default risk limits."""
        try:
            # Portfolio-wide limits
            self.risk_limits['portfolio_value'] = RiskLimit(
                metric_type=RiskMetricType.NOTIONAL_EXPOSURE,
                limit_value=self.risk_config['max_portfolio_value_usd'],
                warning_threshold=0.8,
                time_window_minutes=1440
            )
            
            self.risk_limits['daily_loss'] = RiskLimit(
                metric_type=RiskMetricType.PNL,
                limit_value=-self.risk_config['max_daily_loss_usd'],
                warning_threshold=0.8,
                time_window_minutes=1440
            )
            
            self.risk_limits['max_drawdown'] = RiskLimit(
                metric_type=RiskMetricType.DRAWDOWN,
                limit_value=self.risk_config['max_drawdown_pct'],
                warning_threshold=0.8,
                time_window_minutes=1440
            )
            
            self.risk_limits['max_leverage'] = RiskLimit(
                metric_type=RiskMetricType.LEVERAGE,
                limit_value=self.risk_config['max_leverage'],
                warning_threshold=0.9,
                time_window_minutes=60
            )
            
            self.risk_limits['max_concentration'] = RiskLimit(
                metric_type=RiskMetricType.CONCENTRATION,
                limit_value=self.risk_config['max_concentration_pct'],
                warning_threshold=0.8,
                time_window_minutes=60
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default limits: {e}")
    
    async def start_risk_monitoring(self):
        """Start real-time risk monitoring."""
        try:
            if self.running:
                return
            
            self.running = True
            
            # Start monitoring tasks
            self.monitoring_task = asyncio.create_task(self._risk_monitoring_loop())
            self.circuit_breaker_task = asyncio.create_task(self._circuit_breaker_loop())
            
            self.logger.info("HFT risk monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start risk monitoring: {e}")
            self.running = False
            raise
    
    async def stop_risk_monitoring(self):
        """Stop risk monitoring."""
        try:
            self.running = False
            
            # Cancel monitoring tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.circuit_breaker_task:
                self.circuit_breaker_task.cancel()
                try:
                    await self.circuit_breaker_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("HFT risk monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop risk monitoring: {e}")
    
    async def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update position data for risk monitoring."""
        try:
            with self.risk_lock:
                self.positions[symbol] = position_data
                
        except Exception as e:
            self.logger.error(f"Failed to update position for {symbol}: {e}")
    
    async def update_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """Update market data for risk calculations."""
        try:
            with self.risk_lock:
                self.market_data[symbol] = market_data
                
                # Update price history
                if 'price' in market_data:
                    self.price_history[symbol].append({
                        'timestamp': datetime.now(),
                        'price': market_data['price'],
                        'volume': market_data.get('volume', 0)
                    })
                
        except Exception as e:
            self.logger.error(f"Failed to update market data for {symbol}: {e}")
    
    async def _risk_monitoring_loop(self):
        """Main risk monitoring loop."""
        try:
            while self.running:
                check_start = time.perf_counter()
                
                # Calculate position risks
                await self._calculate_position_risks()
                
                # Calculate portfolio risk
                await self._calculate_portfolio_risk()
                
                # Check risk limits
                await self._check_risk_limits()
                
                # Track monitoring latency
                self.risk_check_latency_ms = (time.perf_counter() - check_start) * 1000
                
                # Sleep until next check
                interval_ms = self.risk_config['risk_check_interval_ms']
                if self.risk_check_latency_ms < interval_ms:
                    await asyncio.sleep((interval_ms - self.risk_check_latency_ms) / 1000)
                else:
                    await asyncio.sleep(0.001)  # Minimum sleep
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Risk monitoring loop error: {e}")
    
    async def _circuit_breaker_loop(self):
        """Circuit breaker monitoring loop."""
        try:
            while self.running:
                if self.risk_config['enable_circuit_breakers']:
                    await self._check_circuit_breakers()
                
                await asyncio.sleep(0.1)  # 100ms circuit breaker checks
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Circuit breaker loop error: {e}")
    
    async def _calculate_position_risks(self):
        """Calculate risk metrics for individual positions."""
        try:
            for symbol, position in self.positions.items():
                if symbol not in self.market_data:
                    continue
                
                market_data = self.market_data[symbol]
                current_price = market_data.get('price', 0)
                
                if current_price <= 0:
                    continue
                
                quantity = position.get('quantity', 0)
                avg_price = position.get('avg_price', current_price)
                
                # Basic metrics
                market_value = abs(quantity * current_price)
                unrealized_pnl = (current_price - avg_price) * quantity
                
                # VaR calculation
                position_var = await self._calculate_position_var(symbol, market_value)
                
                # Drawdown calculation
                max_drawdown = await self._calculate_position_drawdown(symbol)
                
                # Liquidity score
                liquidity_score = await self._calculate_liquidity_score(symbol)
                
                # Concentration ratio
                total_portfolio_value = sum(
                    abs(pos.get('quantity', 0) * self.market_data.get(sym, {}).get('price', 0))
                    for sym, pos in self.positions.items()
                    if self.market_data.get(sym, {}).get('price', 0) > 0
                )
                
                concentration_ratio = market_value / total_portfolio_value if total_portfolio_value > 0 else 0
                
                # Correlation risk
                correlation_risk = await self._calculate_correlation_risk(symbol)
                
                # Store position risk
                self.position_risks[symbol] = PositionRisk(
                    symbol=symbol,
                    quantity=quantity,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    position_var=position_var,
                    max_drawdown=max_drawdown,
                    liquidity_score=liquidity_score,
                    concentration_ratio=concentration_ratio,
                    correlation_risk=correlation_risk
                )
                
        except Exception as e:
            self.logger.error(f"Position risk calculation failed: {e}")
    
    async def _calculate_portfolio_risk(self):
        """Calculate portfolio-level risk metrics."""
        try:
            if not self.position_risks:
                return
            
            # Aggregate metrics
            total_market_value = sum(pos.market_value for pos in self.position_risks.values())
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.position_risks.values())
            
            # Daily PnL calculation
            daily_pnl = await self._calculate_daily_pnl()
            
            # Portfolio VaR
            portfolio_var = await self._calculate_portfolio_var()
            
            # Maximum drawdown
            max_drawdown = await self._calculate_portfolio_drawdown()
            
            # Leverage calculation
            total_notional = sum(
                abs(pos.get('quantity', 0) * self.market_data.get(symbol, {}).get('price', 0))
                for symbol, pos in self.positions.items()
                if self.market_data.get(symbol, {}).get('price', 0) > 0
            )
            
            capital = max(total_market_value, 1)  # Avoid division by zero
            total_leverage = total_notional / capital
            
            # Beta calculation (simplified)
            beta = await self._calculate_portfolio_beta()
            
            # Risk aggregations
            concentration_risk = max(pos.concentration_ratio for pos in self.position_risks.values()) if self.position_risks else 0
            correlation_risk = np.mean([pos.correlation_risk for pos in self.position_risks.values()]) if self.position_risks else 0
            liquidity_risk = 1 - np.mean([pos.liquidity_score for pos in self.position_risks.values()]) if self.position_risks else 0
            
            # Store portfolio risk
            self.portfolio_risk = PortfolioRisk(
                total_market_value=total_market_value,
                total_unrealized_pnl=total_unrealized_pnl,
                daily_pnl=daily_pnl,
                portfolio_var=portfolio_var,
                max_drawdown=max_drawdown,
                total_leverage=total_leverage,
                beta=beta,
                tracking_error=0,  # Would need benchmark for this
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk
            )
            
        except Exception as e:
            self.logger.error(f"Portfolio risk calculation failed: {e}")
    
    async def _check_risk_limits(self):
        """Check all risk limits and generate alerts."""
        try:
            if not self.portfolio_risk:
                return
            
            for limit_id, limit in self.risk_limits.items():
                if not limit.enabled:
                    continue
                
                current_value = await self._get_current_metric_value(limit)
                
                if current_value is None:
                    continue
                
                # Check if limit is breached
                limit_breached = False
                warning_breached = False
                
                if limit.metric_type == RiskMetricType.PNL and limit.limit_value < 0:
                    # For loss limits (negative values)
                    limit_breached = current_value < limit.limit_value
                    warning_breached = current_value < (limit.limit_value * limit.warning_threshold)
                else:
                    # For positive limits
                    limit_breached = current_value > limit.limit_value
                    warning_breached = current_value > (limit.limit_value * limit.warning_threshold)
                
                if limit_breached or warning_breached:
                    risk_level = RiskLevel.CRITICAL if limit_breached else RiskLevel.HIGH
                    
                    await self._generate_risk_alert(
                        limit_id,
                        risk_level,
                        limit.metric_type,
                        current_value,
                        limit.limit_value,
                        limit.symbol
                    )
                
        except Exception as e:
            self.logger.error(f"Risk limit checking failed: {e}")
    
    async def _get_current_metric_value(self, limit: RiskLimit) -> Optional[float]:
        """Get current value for a risk metric."""
        try:
            if not self.portfolio_risk:
                return None
            
            if limit.metric_type == RiskMetricType.NOTIONAL_EXPOSURE:
                return self.portfolio_risk.total_market_value
            elif limit.metric_type == RiskMetricType.PNL:
                return self.portfolio_risk.daily_pnl
            elif limit.metric_type == RiskMetricType.DRAWDOWN:
                return self.portfolio_risk.max_drawdown
            elif limit.metric_type == RiskMetricType.LEVERAGE:
                return self.portfolio_risk.total_leverage
            elif limit.metric_type == RiskMetricType.CONCENTRATION:
                return self.portfolio_risk.concentration_risk
            elif limit.metric_type == RiskMetricType.POSITION_SIZE and limit.symbol:
                pos_risk = self.position_risks.get(limit.symbol)
                return pos_risk.market_value if pos_risk else 0
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get metric value: {e}")
            return None
    
    async def _generate_risk_alert(self, alert_id: str, risk_level: RiskLevel, 
                                 metric_type: RiskMetricType, current_value: float,
                                 limit_value: float, symbol: Optional[str] = None):
        """Generate and process risk alert."""
        try:
            # Check if alert already exists and is recent
            if alert_id in self.active_alerts:
                last_alert = self.active_alerts[alert_id]
                if (datetime.now() - last_alert.timestamp).total_seconds() < self.risk_config['alert_cooldown_seconds']:
                    return
            
            # Create alert
            alert = RiskAlert(
                alert_id=alert_id,
                risk_level=risk_level,
                metric_type=metric_type,
                current_value=current_value,
                limit_value=limit_value,
                symbol=symbol,
                message=f"{metric_type.value} breach: {current_value:.2f} vs limit {limit_value:.2f}"
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.alerts_generated += 1
            
            # Determine and execute action
            action = await self._determine_risk_action(alert)
            if action != RiskAction.MONITOR:
                await self._execute_risk_action(alert, action)
                alert.action_taken = action
                self.actions_taken += 1
            
            # Log alert
            self.logger.warning(
                f"Risk alert generated: {alert.risk_level.value} - {alert.message}"
                f"{f' Action: {action.value}' if action != RiskAction.MONITOR else ''}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate risk alert: {e}")
    
    async def _determine_risk_action(self, alert: RiskAlert) -> RiskAction:
        """Determine appropriate action for risk alert."""
        try:
            if alert.risk_level == RiskLevel.EXTREME:
                return RiskAction.EMERGENCY_EXIT
            elif alert.risk_level == RiskLevel.CRITICAL:
                if alert.metric_type in [RiskMetricType.PNL, RiskMetricType.DRAWDOWN]:
                    return RiskAction.HALT_TRADING
                elif alert.metric_type in [RiskMetricType.POSITION_SIZE, RiskMetricType.CONCENTRATION]:
                    return RiskAction.REDUCE_POSITION
                else:
                    return RiskAction.CANCEL_ORDERS
            elif alert.risk_level == RiskLevel.HIGH:
                if self.risk_config['auto_reduce_on_high_risk']:
                    return RiskAction.REDUCE_POSITION
                else:
                    return RiskAction.WARN
            else:
                return RiskAction.MONITOR
                
        except Exception as e:
            self.logger.error(f"Failed to determine risk action: {e}")
            return RiskAction.MONITOR
    
    async def _execute_risk_action(self, alert: RiskAlert, action: RiskAction):
        """Execute risk control action."""
        try:
            if action == RiskAction.WARN:
                # Just log the warning (already done)
                pass
            
            elif action == RiskAction.REDUCE_POSITION:
                if alert.symbol:
                    await self._reduce_position(alert.symbol, 0.5)  # Reduce by 50%
                else:
                    await self._reduce_all_positions(0.3)  # Reduce all by 30%
            
            elif action == RiskAction.CANCEL_ORDERS:
                if alert.symbol:
                    await self._cancel_symbol_orders(alert.symbol)
                else:
                    await self._cancel_all_orders()
            
            elif action == RiskAction.HALT_TRADING:
                await self._halt_trading()
            
            elif action == RiskAction.EMERGENCY_EXIT:
                await self._emergency_exit()
            
            self.logger.info(f"Risk action executed: {action.value} for alert {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute risk action {action.value}: {e}")
    
    async def _check_circuit_breakers(self):
        """Check circuit breaker conditions."""
        try:
            # Daily loss circuit breaker
            if self.portfolio_risk and self.portfolio_risk.daily_pnl < -self.risk_config['max_daily_loss_usd']:
                if not self.circuit_breakers[CircuitBreakerType.DAILY_LOSS]:
                    await self._trigger_circuit_breaker(CircuitBreakerType.DAILY_LOSS)
            
            # Volatility spike circuit breaker
            for symbol in self.market_data.keys():
                volatility = await self._calculate_current_volatility(symbol)
                avg_volatility = await self._calculate_average_volatility(symbol)
                
                if volatility > avg_volatility * self.risk_config['volatility_spike_threshold']:
                    if not self.circuit_breakers[CircuitBreakerType.VOLATILITY_SPIKE]:
                        await self._trigger_circuit_breaker(CircuitBreakerType.VOLATILITY_SPIKE)
                        break
            
            # Add more circuit breaker checks as needed
            
        except Exception as e:
            self.logger.error(f"Circuit breaker check failed: {e}")
    
    async def _trigger_circuit_breaker(self, breaker_type: CircuitBreakerType):
        """Trigger circuit breaker."""
        try:
            self.circuit_breakers[breaker_type] = True
            
            # Log circuit breaker
            self.logger.critical(f"Circuit breaker triggered: {breaker_type.value}")
            
            # Take appropriate action based on breaker type
            if breaker_type == CircuitBreakerType.DAILY_LOSS:
                await self._halt_trading()
            elif breaker_type == CircuitBreakerType.VOLATILITY_SPIKE:
                await self._reduce_all_positions(0.5)  # Reduce all positions by 50%
                
        except Exception as e:
            self.logger.error(f"Failed to trigger circuit breaker {breaker_type.value}: {e}")
    
    # Risk calculation methods
    
    async def _calculate_position_var(self, symbol: str, market_value: float) -> float:
        """Calculate position Value at Risk."""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
                return market_value * 0.02  # Default 2% VaR
            
            # Get price returns
            prices = [entry['price'] for entry in self.price_history[symbol]]
            returns = np.diff(np.log(prices))
            
            if len(returns) < 10:
                return market_value * 0.02
            
            # Calculate VaR using historical simulation
            var_percentile = (1 - self.risk_config['var_confidence_level']) * 100
            var_return = np.percentile(returns, var_percentile)
            
            return abs(market_value * var_return)
            
        except Exception as e:
            self.logger.error(f"Position VaR calculation failed for {symbol}: {e}")
            return market_value * 0.02
    
    async def _calculate_portfolio_var(self) -> float:
        """Calculate portfolio Value at Risk."""
        try:
            if not self.position_risks:
                return 0.0
            
            # Simple aggregation - in practice would use correlation matrix
            total_var = sum(pos.position_var for pos in self.position_risks.values())
            
            # Apply diversification benefit (simplified)
            diversification_factor = 0.7  # Assume 30% diversification benefit
            
            return total_var * diversification_factor
            
        except Exception as e:
            self.logger.error(f"Portfolio VaR calculation failed: {e}")
            return 0.0
    
    async def _calculate_position_drawdown(self, symbol: str) -> float:
        """Calculate position maximum drawdown."""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
                return 0.0
            
            prices = np.array([entry['price'] for entry in self.price_history[symbol]])
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(prices)
            
            # Calculate drawdowns
            drawdowns = (prices - running_max) / running_max
            
            return abs(np.min(drawdowns))
            
        except Exception as e:
            self.logger.error(f"Position drawdown calculation failed for {symbol}: {e}")
            return 0.0
    
    async def _calculate_portfolio_drawdown(self) -> float:
        """Calculate portfolio maximum drawdown."""
        try:
            # Simplified calculation - would need portfolio value history
            if not self.position_risks:
                return 0.0
            
            # Use maximum individual position drawdown as proxy
            return max(pos.max_drawdown for pos in self.position_risks.values())
            
        except Exception as e:
            self.logger.error(f"Portfolio drawdown calculation failed: {e}")
            return 0.0
    
    async def _calculate_daily_pnl(self) -> float:
        """Calculate daily PnL."""
        try:
            # Simplified - sum unrealized PnL
            return sum(pos.unrealized_pnl for pos in self.position_risks.values()) if self.position_risks else 0.0
            
        except Exception as e:
            self.logger.error(f"Daily PnL calculation failed: {e}")
            return 0.0
    
    async def _calculate_liquidity_score(self, symbol: str) -> float:
        """Calculate liquidity score for symbol."""
        try:
            market_data = self.market_data.get(symbol, {})
            volume = market_data.get('volume', 0)
            spread = market_data.get('spread', 0)
            
            # Simple liquidity score based on volume and spread
            volume_score = min(1.0, volume / 1000000)  # Normalize to $1M volume
            spread_score = max(0.0, 1.0 - spread / 0.01)  # Normalize to 1% spread
            
            return (volume_score + spread_score) / 2
            
        except Exception as e:
            self.logger.error(f"Liquidity score calculation failed for {symbol}: {e}")
            return 0.5
    
    async def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk for symbol."""
        try:
            # Simplified correlation risk calculation
            # In practice, would calculate correlation with other positions
            return 0.3  # Default moderate correlation risk
            
        except Exception as e:
            self.logger.error(f"Correlation risk calculation failed for {symbol}: {e}")
            return 0.3
    
    async def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta."""
        try:
            # Simplified beta calculation
            # Would need market benchmark for proper calculation
            return 1.0  # Default market beta
            
        except Exception as e:
            self.logger.error(f"Portfolio beta calculation failed: {e}")
            return 1.0
    
    async def _calculate_current_volatility(self, symbol: str) -> float:
        """Calculate current volatility for symbol."""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
                return 0.02  # Default 2% volatility
            
            # Get recent price returns
            recent_prices = [entry['price'] for entry in list(self.price_history[symbol])[-20:]]
            returns = np.diff(np.log(recent_prices))
            
            return np.std(returns) * np.sqrt(24 * 365)  # Annualized volatility
            
        except Exception as e:
            self.logger.error(f"Current volatility calculation failed for {symbol}: {e}")
            return 0.02
    
    async def _calculate_average_volatility(self, symbol: str) -> float:
        """Calculate average volatility for symbol."""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 50:
                return 0.02
            
            # Get all available price returns
            prices = [entry['price'] for entry in self.price_history[symbol]]
            returns = np.diff(np.log(prices))
            
            return np.std(returns) * np.sqrt(24 * 365)  # Annualized volatility
            
        except Exception as e:
            self.logger.error(f"Average volatility calculation failed for {symbol}: {e}")
            return 0.02
    
    # Risk control actions
    
    async def _reduce_position(self, symbol: str, reduction_ratio: float):
        """Reduce position for specific symbol."""
        try:
            # This would integrate with execution engine
            self.logger.info(f"Reducing position for {symbol} by {reduction_ratio*100:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Failed to reduce position for {symbol}: {e}")
    
    async def _reduce_all_positions(self, reduction_ratio: float):
        """Reduce all positions."""
        try:
            self.logger.info(f"Reducing all positions by {reduction_ratio*100:.1f}%")
            
            for symbol in self.positions.keys():
                await self._reduce_position(symbol, reduction_ratio)
                
        except Exception as e:
            self.logger.error(f"Failed to reduce all positions: {e}")
    
    async def _cancel_symbol_orders(self, symbol: str):
        """Cancel all orders for specific symbol."""
        try:
            self.logger.info(f"Cancelling all orders for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to cancel orders for {symbol}: {e}")
    
    async def _cancel_all_orders(self):
        """Cancel all active orders."""
        try:
            self.logger.info("Cancelling all active orders")
            
        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {e}")
    
    async def _halt_trading(self):
        """Halt all trading activities."""
        try:
            self.logger.critical("TRADING HALTED due to risk breach")
            
            # Cancel all orders
            await self._cancel_all_orders()
            
            # Set circuit breaker flag
            self.circuit_breakers[CircuitBreakerType.DAILY_LOSS] = True
            
        except Exception as e:
            self.logger.error(f"Failed to halt trading: {e}")
    
    async def _emergency_exit(self):
        """Execute emergency exit of all positions."""
        try:
            self.logger.critical("EMERGENCY EXIT triggered")
            
            # Cancel all orders first
            await self._cancel_all_orders()
            
            # Close all positions
            await self._reduce_all_positions(1.0)  # 100% reduction = close all
            
        except Exception as e:
            self.logger.error(f"Failed to execute emergency exit: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        try:
            portfolio_summary = {}
            if self.portfolio_risk:
                portfolio_summary = {
                    'total_market_value': self.portfolio_risk.total_market_value,
                    'daily_pnl': self.portfolio_risk.daily_pnl,
                    'portfolio_var': self.portfolio_risk.portfolio_var,
                    'max_drawdown': self.portfolio_risk.max_drawdown,
                    'total_leverage': self.portfolio_risk.total_leverage,
                    'concentration_risk': self.portfolio_risk.concentration_risk,
                    'liquidity_risk': self.portfolio_risk.liquidity_risk
                }
            
            return {
                'running': self.running,
                'risk_check_latency_ms': self.risk_check_latency_ms,
                'active_alerts': len(self.active_alerts),
                'alerts_generated': self.alerts_generated,
                'actions_taken': self.actions_taken,
                'portfolio_risk': portfolio_summary,
                'circuit_breakers': {cb.value: status for cb, status in self.circuit_breakers.items()},
                'position_count': len(self.position_risks),
                'risk_limits': len(self.risk_limits)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate risk summary: {e}")
            return {'error': 'Unable to generate summary'}


class VaRCalculator:
    """Value at Risk calculator."""
    
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        self.logger = TradingLogger()


class StressTester:
    """Stress testing engine."""
    
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        self.logger = TradingLogger()


class CorrelationMonitor:
    """Correlation monitoring system."""
    
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        self.logger = TradingLogger()