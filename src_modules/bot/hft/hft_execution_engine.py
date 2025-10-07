"""
HFT Execution Engine for Ultra-Low Latency Trading.
Provides microsecond-level order execution with advanced routing and smart order management.
"""

import asyncio
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import psutil
    import uvloop  # Ultra-fast event loop
    HAS_OPTIMIZATIONS = True
except ImportError:
    HAS_OPTIMIZATIONS = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class ExecutionStrategy(Enum):
    """Execution strategies."""
    IMMEDIATE = "immediate"           # Execute immediately at market
    TWAP = "twap"                    # Time Weighted Average Price
    VWAP = "vwap"                    # Volume Weighted Average Price
    ICEBERG = "iceberg"              # Hidden iceberg orders
    SNIPER = "sniper"                # Wait for optimal fill price
    AGGRESSIVE = "aggressive"        # Market impact optimized
    STEALTH = "stealth"              # Minimize market footprint
    LIQUIDITY_SEEKING = "liquidity_seeking"  # Seek hidden liquidity
    MOMENTUM = "momentum"            # Follow momentum signals
    MEAN_REVERSION = "mean_reversion"  # Counter-trend execution

class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    ROUTING = "routing"
    SENT = "sent"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"

class ExecutionVenue(Enum):
    """Execution venues."""
    PRIMARY_EXCHANGE = "primary"
    DARK_POOL = "dark_pool"
    CROSS_NETWORK = "cross_network"
    SMART_ROUTER = "smart_router"

class UrgencyLevel(Enum):
    """Execution urgency levels."""
    LOW = "low"           # Price priority
    MEDIUM = "medium"     # Balanced
    HIGH = "high"         # Time priority
    CRITICAL = "critical" # Immediate execution

@dataclass
class ExecutionOrder:
    """HFT execution order."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market', 'limit', 'stop'
    strategy: ExecutionStrategy
    urgency: UrgencyLevel
    
    # Execution parameters
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    max_slippage_bps: float = 10.0
    max_participation_rate: float = 0.2  # 20% of volume
    
    # Timing constraints
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_window_ms: int = 5000  # 5 second default window
    
    # Advanced parameters
    iceberg_size: Optional[float] = None
    minimum_fill_size: Optional[float] = None
    allow_partial_fills: bool = True
    preferred_venues: List[ExecutionVenue] = field(default_factory=list)
    
    # State tracking
    status: OrderStatus = OrderStatus.PENDING
    created_timestamp: datetime = field(default_factory=datetime.now)
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    total_fees: float = 0.0
    execution_latency_ms: float = 0.0
    
    # Performance tracking
    benchmark_price: Optional[float] = None
    implementation_shortfall: float = 0.0
    market_impact_bps: float = 0.0

@dataclass
class ExecutionFill:
    """Order fill information."""
    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    venue: str
    fees: float = 0.0
    liquidity_flag: str = "unknown"  # 'maker', 'taker', 'hidden'

@dataclass
class MarketData:
    """Real-time market data for execution."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    mid: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    vwap: float = 0.0
    
    # Level 2 data
    bid_levels: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    ask_levels: List[Tuple[float, float]] = field(default_factory=list)
    
    # Derived metrics
    spread_bps: float = field(init=False)
    liquidity_score: float = field(init=False)
    
    def __post_init__(self):
        if self.mid > 0:
            self.spread_bps = ((self.ask - self.bid) / self.mid) * 10000
        else:
            self.spread_bps = 0.0
        
        # Simple liquidity score based on top-of-book size
        self.liquidity_score = min(self.bid_size, self.ask_size) / max(self.bid_size, self.ask_size)

@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    
    # Latency metrics
    avg_execution_latency_ms: float = 0.0
    p95_execution_latency_ms: float = 0.0
    p99_execution_latency_ms: float = 0.0
    
    # Performance metrics
    avg_implementation_shortfall_bps: float = 0.0
    avg_market_impact_bps: float = 0.0
    avg_slippage_bps: float = 0.0
    
    # Fill rates
    fill_rate: float = 0.0
    average_fill_time_ms: float = 0.0
    
    # Cost analysis
    total_fees_paid: float = 0.0
    avg_fee_rate_bps: float = 0.0

class HFTExecutionEngine:
    """Ultra-low latency execution engine."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Execution configuration
        self.exec_config = {
            'max_concurrent_orders': 1000,
            'execution_timeout_ms': 30000,
            'latency_target_microseconds': 100,
            'slippage_tolerance_bps': 5.0,
            'max_order_rate_per_second': 100,
            'smart_routing_enabled': True,
            'dark_pool_preference': 0.3,  # 30% preference for dark pools
            'minimum_size_for_iceberg': 10.0,
            'twap_slice_duration_ms': 1000,
            'vwap_lookback_minutes': 15,
            'aggressive_timeout_ms': 500,
            'stealth_randomization_pct': 0.1,  # 10% randomization for stealth
            'momentum_threshold_bps': 20,
            'mean_reversion_zscore': 2.0
        }
        
        # Execution state
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.order_history: deque = deque(maxlen=100000)
        self.fill_history: deque = deque(maxlen=100000)
        self.market_data: Dict[str, MarketData] = {}
        
        # Execution strategies
        self.strategy_handlers = {
            ExecutionStrategy.IMMEDIATE: self._execute_immediate,
            ExecutionStrategy.TWAP: self._execute_twap,
            ExecutionStrategy.VWAP: self._execute_vwap,
            ExecutionStrategy.ICEBERG: self._execute_iceberg,
            ExecutionStrategy.SNIPER: self._execute_sniper,
            ExecutionStrategy.AGGRESSIVE: self._execute_aggressive,
            ExecutionStrategy.STEALTH: self._execute_stealth,
            ExecutionStrategy.LIQUIDITY_SEEKING: self._execute_liquidity_seeking,
            ExecutionStrategy.MOMENTUM: self._execute_momentum,
            ExecutionStrategy.MEAN_REVERSION: self._execute_mean_reversion
        }
        
        # Performance optimization
        if HAS_OPTIMIZATIONS:
            # Set high priority for execution thread
            try:
                process = psutil.Process()
                process.nice(-10)  # High priority
            except:
                pass
        
        # Threading and async management
        self.exec_lock = threading.Lock()
        self.running = False
        self.execution_tasks: Dict[str, asyncio.Task] = {}
        self.main_execution_task = None
        
        # Performance tracking
        self.metrics = ExecutionMetrics()
        self.latency_measurements: deque = deque(maxlen=10000)
        
        # Order rate limiting
        self.order_rate_limiter = []  # Sliding window for rate limiting
        
        self.logger.info("HFTExecutionEngine initialized")
    
    async def start_execution_engine(self):
        """Start the execution engine."""
        try:
            if self.running:
                return
            
            self.running = True
            
            # Use optimized event loop if available
            if HAS_OPTIMIZATIONS:
                try:
                    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                except:
                    pass
            
            # Start main execution loop
            self.main_execution_task = asyncio.create_task(self._execution_loop())
            
            self.logger.info("HFT execution engine started")
            
        except Exception as e:
            self.logger.error(f"Failed to start execution engine: {e}")
            self.running = False
            raise
    
    async def stop_execution_engine(self):
        """Stop the execution engine."""
        try:
            self.running = False
            
            # Cancel all active execution tasks
            for task in self.execution_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            if self.execution_tasks:
                await asyncio.gather(*self.execution_tasks.values(), return_exceptions=True)
            
            # Cancel main execution task
            if self.main_execution_task:
                self.main_execution_task.cancel()
                try:
                    await self.main_execution_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel any remaining active orders
            await self._cancel_all_orders()
            
            self.logger.info("HFT execution engine stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop execution engine: {e}")
    
    async def submit_order(self, order: ExecutionOrder) -> str:
        """Submit order for execution."""
        try:
            execution_start = time.perf_counter()
            
            # Validate order
            if not await self._validate_order(order):
                order.status = OrderStatus.REJECTED
                return order.order_id
            
            # Check rate limits
            if not await self._check_rate_limits():
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"Order rejected due to rate limits: {order.order_id}")
                return order.order_id
            
            # Add to active orders
            with self.exec_lock:
                self.active_orders[order.order_id] = order
                order.status = OrderStatus.ROUTING
            
            # Create execution task
            strategy_handler = self.strategy_handlers.get(order.strategy, self._execute_immediate)
            execution_task = asyncio.create_task(strategy_handler(order))
            self.execution_tasks[order.order_id] = execution_task
            
            # Track submission latency
            submission_latency_ms = (time.perf_counter() - execution_start) * 1000
            order.execution_latency_ms = submission_latency_ms
            
            self.logger.debug(f"Order submitted: {order.order_id} {order.strategy.value} {order.symbol} {order.side} {order.quantity}")
            
            return order.order_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
            return order.order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel active order."""
        try:
            with self.exec_lock:
                if order_id not in self.active_orders:
                    return False
                
                order = self.active_orders[order_id]
                order.status = OrderStatus.CANCELLED
            
            # Cancel execution task
            if order_id in self.execution_tasks:
                self.execution_tasks[order_id].cancel()
                del self.execution_tasks[order_id]
            
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def update_market_data(self, symbol: str, market_data: MarketData):
        """Update market data for execution decisions."""
        try:
            self.market_data[symbol] = market_data
            
        except Exception as e:
            self.logger.error(f"Failed to update market data for {symbol}: {e}")
    
    async def _execution_loop(self):
        """Main execution monitoring loop."""
        try:
            while self.running:
                # Monitor order execution
                await self._monitor_executions()
                
                # Clean completed tasks
                await self._cleanup_completed_tasks()
                
                # Update metrics
                await self._update_metrics()
                
                # Ultra-short sleep for high-frequency monitoring
                await asyncio.sleep(0.001)  # 1ms monitoring interval
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Execution loop error: {e}")
    
    async def _monitor_executions(self):
        """Monitor active order executions."""
        try:
            current_time = datetime.now()
            
            for order_id, order in list(self.active_orders.items()):
                # Check for timeouts
                if (current_time - order.created_timestamp).total_seconds() * 1000 > order.execution_window_ms:
                    if order.status in [OrderStatus.PENDING, OrderStatus.ROUTING, OrderStatus.SENT]:
                        await self.cancel_order(order_id)
                        order.status = OrderStatus.EXPIRED
                        self.logger.warning(f"Order expired: {order_id}")
                
                # Move completed orders to history
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.FAILED]:
                    with self.exec_lock:
                        if order_id in self.active_orders:
                            del self.active_orders[order_id]
                            self.order_history.append(order)
            
        except Exception as e:
            self.logger.error(f"Execution monitoring failed: {e}")
    
    async def _cleanup_completed_tasks(self):
        """Clean up completed execution tasks."""
        try:
            completed_tasks = []
            
            for order_id, task in self.execution_tasks.items():
                if task.done():
                    completed_tasks.append(order_id)
            
            for order_id in completed_tasks:
                del self.execution_tasks[order_id]
                
        except Exception as e:
            self.logger.error(f"Task cleanup failed: {e}")
    
    async def _validate_order(self, order: ExecutionOrder) -> bool:
        """Validate order parameters."""
        try:
            # Basic validation
            if order.quantity <= 0:
                self.logger.error(f"Invalid quantity: {order.quantity}")
                return False
            
            if order.symbol not in self.market_data:
                self.logger.error(f"No market data for symbol: {order.symbol}")
                return False
            
            # Strategy-specific validation
            if order.strategy == ExecutionStrategy.ICEBERG and not order.iceberg_size:
                order.iceberg_size = order.quantity / 10  # Default to 10 slices
            
            if order.strategy in [ExecutionStrategy.TWAP, ExecutionStrategy.VWAP]:
                if not order.end_time:
                    order.end_time = datetime.now() + timedelta(minutes=5)  # Default 5 minutes
            
            return True
            
        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            return False
    
    async def _check_rate_limits(self) -> bool:
        """Check if order submission is within rate limits."""
        try:
            current_time = time.time()
            
            # Clean old entries (beyond 1 second)
            self.order_rate_limiter = [
                timestamp for timestamp in self.order_rate_limiter
                if current_time - timestamp < 1.0
            ]
            
            # Check if we're within limits
            if len(self.order_rate_limiter) >= self.exec_config['max_order_rate_per_second']:
                return False
            
            # Add current timestamp
            self.order_rate_limiter.append(current_time)
            return True
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return False
    
    # Execution Strategy Implementations
    
    async def _execute_immediate(self, order: ExecutionOrder):
        """Execute order immediately at market price."""
        try:
            market_data = self.market_data.get(order.symbol)
            if not market_data:
                order.status = OrderStatus.FAILED
                return
            
            # Determine execution price
            if order.side == 'buy':
                execution_price = market_data.ask
                available_size = market_data.ask_size
            else:
                execution_price = market_data.bid
                available_size = market_data.bid_size
            
            # Check if we can fill the full order
            fill_quantity = min(order.quantity, available_size)
            
            if fill_quantity < order.minimum_fill_size if order.minimum_fill_size else 0:
                order.status = OrderStatus.REJECTED
                return
            
            # Simulate execution
            await self._simulate_fill(order, fill_quantity, execution_price)
            
        except Exception as e:
            self.logger.error(f"Immediate execution failed for order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
    
    async def _execute_twap(self, order: ExecutionOrder):
        """Execute order using Time Weighted Average Price strategy."""
        try:
            if not order.end_time:
                order.end_time = datetime.now() + timedelta(minutes=5)
            
            total_duration = (order.end_time - datetime.now()).total_seconds()
            slice_duration = self.exec_config['twap_slice_duration_ms'] / 1000
            
            if total_duration <= 0:
                await self._execute_immediate(order)
                return
            
            num_slices = max(1, int(total_duration / slice_duration))
            slice_quantity = order.quantity / num_slices
            
            order.status = OrderStatus.SENT
            
            for i in range(num_slices):
                if order.status != OrderStatus.SENT:
                    break
                
                # Create child order for this slice
                child_order = ExecutionOrder(
                    order_id=f"{order.order_id}_slice_{i}",
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_quantity,
                    order_type=order.order_type,
                    strategy=ExecutionStrategy.IMMEDIATE,
                    urgency=order.urgency,
                    limit_price=order.limit_price
                )
                
                await self._execute_immediate(child_order)
                
                # Update parent order
                if child_order.status == OrderStatus.FILLED:
                    order.filled_quantity += child_order.filled_quantity
                    
                    # Update average fill price
                    if order.filled_quantity > 0:
                        total_value = (order.avg_fill_price * (order.filled_quantity - child_order.filled_quantity) +
                                     child_order.avg_fill_price * child_order.filled_quantity)
                        order.avg_fill_price = total_value / order.filled_quantity
                
                # Wait for next slice
                if i < num_slices - 1:
                    await asyncio.sleep(slice_duration)
            
            # Check if order is complete
            if order.filled_quantity >= order.quantity * 0.95:  # 95% fill threshold
                order.status = OrderStatus.FILLED
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
            else:
                order.status = OrderStatus.FAILED
                
        except Exception as e:
            self.logger.error(f"TWAP execution failed for order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
    
    async def _execute_vwap(self, order: ExecutionOrder):
        """Execute order using Volume Weighted Average Price strategy."""
        try:
            market_data = self.market_data.get(order.symbol)
            if not market_data:
                order.status = OrderStatus.FAILED
                return
            
            # Use participation rate to determine slice size
            current_volume_rate = market_data.volume / (self.exec_config['vwap_lookback_minutes'] * 60)
            target_participation = order.max_participation_rate
            slice_size = min(order.quantity * 0.1, current_volume_rate * target_participation)
            
            order.status = OrderStatus.SENT
            remaining_quantity = order.quantity
            
            while remaining_quantity > 0 and order.status == OrderStatus.SENT:
                current_slice_size = min(remaining_quantity, slice_size)
                
                # Create child order
                child_order = ExecutionOrder(
                    order_id=f"{order.order_id}_vwap_{int(time.time() * 1000)}",
                    symbol=order.symbol,
                    side=order.side,
                    quantity=current_slice_size,
                    order_type=order.order_type,
                    strategy=ExecutionStrategy.IMMEDIATE,
                    urgency=order.urgency
                )
                
                await self._execute_immediate(child_order)
                
                if child_order.status == OrderStatus.FILLED:
                    order.filled_quantity += child_order.filled_quantity
                    remaining_quantity -= child_order.filled_quantity
                    
                    # Update average fill price
                    if order.filled_quantity > 0:
                        total_value = (order.avg_fill_price * (order.filled_quantity - child_order.filled_quantity) +
                                     child_order.avg_fill_price * child_order.filled_quantity)
                        order.avg_fill_price = total_value / order.filled_quantity
                
                # Adaptive delay based on market conditions
                delay_ms = max(100, 1000 / current_volume_rate)  # Minimum 100ms delay
                await asyncio.sleep(delay_ms / 1000)
            
            # Finalize order status
            if remaining_quantity <= 0:
                order.status = OrderStatus.FILLED
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
            else:
                order.status = OrderStatus.FAILED
                
        except Exception as e:
            self.logger.error(f"VWAP execution failed for order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
    
    async def _execute_iceberg(self, order: ExecutionOrder):
        """Execute iceberg order with hidden quantity."""
        try:
            iceberg_size = order.iceberg_size or (order.quantity / 10)
            order.status = OrderStatus.SENT
            
            remaining_quantity = order.quantity
            
            while remaining_quantity > 0 and order.status == OrderStatus.SENT:
                visible_size = min(remaining_quantity, iceberg_size)
                
                # Add randomization for stealth
                randomization = 1 + (np.random.random() - 0.5) * 0.1  # ±5% randomization
                visible_size *= randomization
                visible_size = min(visible_size, remaining_quantity)
                
                # Create visible child order
                child_order = ExecutionOrder(
                    order_id=f"{order.order_id}_iceberg_{int(time.time() * 1000)}",
                    symbol=order.symbol,
                    side=order.side,
                    quantity=visible_size,
                    order_type=order.order_type,
                    strategy=ExecutionStrategy.IMMEDIATE,
                    urgency=order.urgency,
                    limit_price=order.limit_price
                )
                
                await self._execute_immediate(child_order)
                
                if child_order.status == OrderStatus.FILLED:
                    order.filled_quantity += child_order.filled_quantity
                    remaining_quantity -= child_order.filled_quantity
                    
                    # Update average fill price
                    if order.filled_quantity > 0:
                        total_value = (order.avg_fill_price * (order.filled_quantity - child_order.filled_quantity) +
                                     child_order.avg_fill_price * child_order.filled_quantity)
                        order.avg_fill_price = total_value / order.filled_quantity
                
                # Random delay between icebergs
                delay_ms = np.random.randint(500, 2000)  # 0.5-2 second delay
                await asyncio.sleep(delay_ms / 1000)
            
            # Finalize status
            if remaining_quantity <= 0:
                order.status = OrderStatus.FILLED
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
            else:
                order.status = OrderStatus.FAILED
                
        except Exception as e:
            self.logger.error(f"Iceberg execution failed for order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
    
    async def _execute_sniper(self, order: ExecutionOrder):
        """Execute sniper strategy - wait for optimal price."""
        try:
            market_data = self.market_data.get(order.symbol)
            if not market_data:
                order.status = OrderStatus.FAILED
                return
            
            # Set target price based on current mid price and desired improvement
            improvement_bps = 5  # 5 bps price improvement target
            improvement_amount = market_data.mid * (improvement_bps / 10000)
            
            if order.side == 'buy':
                target_price = market_data.mid - improvement_amount
            else:
                target_price = market_data.mid + improvement_amount
            
            order.status = OrderStatus.SENT
            start_time = time.time()
            timeout_seconds = order.execution_window_ms / 1000
            
            # Wait for favorable price
            while order.status == OrderStatus.SENT:
                current_market = self.market_data.get(order.symbol)
                if not current_market:
                    break
                
                # Check if target price is achievable
                if order.side == 'buy' and current_market.ask <= target_price:
                    await self._execute_immediate(order)
                    break
                elif order.side == 'sell' and current_market.bid >= target_price:
                    await self._execute_immediate(order)
                    break
                
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    # Execute at market if timeout reached
                    await self._execute_immediate(order)
                    break
                
                await asyncio.sleep(0.01)  # 10ms polling interval
                
        except Exception as e:
            self.logger.error(f"Sniper execution failed for order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
    
    async def _execute_aggressive(self, order: ExecutionOrder):
        """Execute aggressive strategy with market impact optimization."""
        try:
            market_data = self.market_data.get(order.symbol)
            if not market_data:
                order.status = OrderStatus.FAILED
                return
            
            # Assess market impact and liquidity
            total_liquidity = sum(size for _, size in market_data.bid_levels + market_data.ask_levels)
            impact_ratio = order.quantity / total_liquidity if total_liquidity > 0 else 1.0
            
            if impact_ratio > 0.1:  # High impact order - slice it
                # Divide into aggressive slices
                num_slices = min(10, max(2, int(impact_ratio * 20)))
                slice_size = order.quantity / num_slices
                
                order.status = OrderStatus.SENT
                
                for i in range(num_slices):
                    if order.status != OrderStatus.SENT:
                        break
                    
                    child_order = ExecutionOrder(
                        order_id=f"{order.order_id}_aggressive_{i}",
                        symbol=order.symbol,
                        side=order.side,
                        quantity=slice_size,
                        order_type=order.order_type,
                        strategy=ExecutionStrategy.IMMEDIATE,
                        urgency=UrgencyLevel.CRITICAL
                    )
                    
                    await self._execute_immediate(child_order)
                    
                    if child_order.status == OrderStatus.FILLED:
                        order.filled_quantity += child_order.filled_quantity
                        
                        # Update average fill price
                        if order.filled_quantity > 0:
                            total_value = (order.avg_fill_price * (order.filled_quantity - child_order.filled_quantity) +
                                         child_order.avg_fill_price * child_order.filled_quantity)
                            order.avg_fill_price = total_value / order.filled_quantity
                    
                    # Very short delay for aggressive execution
                    await asyncio.sleep(0.05)  # 50ms between slices
                
                # Finalize status
                if order.filled_quantity >= order.quantity * 0.95:
                    order.status = OrderStatus.FILLED
                elif order.filled_quantity > 0:
                    order.status = OrderStatus.PARTIALLY_FILLED
                else:
                    order.status = OrderStatus.FAILED
            else:
                # Low impact - execute immediately
                await self._execute_immediate(order)
                
        except Exception as e:
            self.logger.error(f"Aggressive execution failed for order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
    
    async def _execute_stealth(self, order: ExecutionOrder):
        """Execute stealth strategy to minimize market footprint."""
        try:
            # Highly randomized execution to avoid detection
            randomization_pct = self.exec_config['stealth_randomization_pct']
            
            # Random slice sizes
            num_slices = np.random.randint(5, 15)
            slice_sizes = np.random.dirichlet(np.ones(num_slices)) * order.quantity
            
            # Random delays
            base_delay_ms = np.random.randint(1000, 5000)  # 1-5 second base delay
            
            order.status = OrderStatus.SENT
            
            for i, slice_size in enumerate(slice_sizes):
                if order.status != OrderStatus.SENT:
                    break
                
                # Add randomization to slice size
                randomized_size = slice_size * (1 + (np.random.random() - 0.5) * randomization_pct)
                randomized_size = min(randomized_size, order.quantity - order.filled_quantity)
                
                if randomized_size <= 0:
                    continue
                
                child_order = ExecutionOrder(
                    order_id=f"{order.order_id}_stealth_{i}",
                    symbol=order.symbol,
                    side=order.side,
                    quantity=randomized_size,
                    order_type=order.order_type,
                    strategy=ExecutionStrategy.IMMEDIATE,
                    urgency=UrgencyLevel.LOW
                )
                
                await self._execute_immediate(child_order)
                
                if child_order.status == OrderStatus.FILLED:
                    order.filled_quantity += child_order.filled_quantity
                    
                    # Update average fill price
                    if order.filled_quantity > 0:
                        total_value = (order.avg_fill_price * (order.filled_quantity - child_order.filled_quantity) +
                                     child_order.avg_fill_price * child_order.filled_quantity)
                        order.avg_fill_price = total_value / order.filled_quantity
                
                # Random delay
                if i < len(slice_sizes) - 1:
                    delay_variation = np.random.randint(-500, 1000)  # ±0.5 to +1 second variation
                    delay_ms = max(100, base_delay_ms + delay_variation)
                    await asyncio.sleep(delay_ms / 1000)
            
            # Finalize status
            if order.filled_quantity >= order.quantity * 0.95:
                order.status = OrderStatus.FILLED
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
            else:
                order.status = OrderStatus.FAILED
                
        except Exception as e:
            self.logger.error(f"Stealth execution failed for order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
    
    async def _execute_liquidity_seeking(self, order: ExecutionOrder):
        """Execute liquidity seeking strategy."""
        try:
            # This would integrate with dark pools and hidden liquidity
            # For now, simulate by using limit orders slightly inside the spread
            
            market_data = self.market_data.get(order.symbol)
            if not market_data:
                order.status = OrderStatus.FAILED
                return
            
            # Place passive orders to capture hidden liquidity
            improvement_bps = 1  # 1 bps inside spread
            improvement = market_data.mid * (improvement_bps / 10000)
            
            if order.side == 'buy':
                limit_price = market_data.bid + improvement
            else:
                limit_price = market_data.ask - improvement
            
            # Simulate limit order execution with time delay
            order.status = OrderStatus.SENT
            order.limit_price = limit_price
            
            # Wait for potential fill
            await asyncio.sleep(2.0)  # 2 second wait for liquidity
            
            # Simulate partial fill based on liquidity score
            fill_probability = market_data.liquidity_score * 0.7  # Up to 70% fill chance
            
            if np.random.random() < fill_probability:
                # Partial fill
                fill_ratio = np.random.uniform(0.3, 0.8)  # 30-80% fill
                fill_quantity = order.quantity * fill_ratio
                
                await self._simulate_fill(order, fill_quantity, limit_price)
                
                if fill_quantity >= order.quantity * 0.95:
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED
            else:
                # No fill - execute remaining at market
                await self._execute_immediate(order)
                
        except Exception as e:
            self.logger.error(f"Liquidity seeking execution failed for order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
    
    async def _execute_momentum(self, order: ExecutionOrder):
        """Execute momentum-based strategy."""
        try:
            market_data = self.market_data.get(order.symbol)
            if not market_data:
                order.status = OrderStatus.FAILED
                return
            
            # Simple momentum detection (would use more sophisticated signals in practice)
            price_change_bps = abs(market_data.last_price - market_data.vwap) / market_data.vwap * 10000
            
            if price_change_bps > self.exec_config['momentum_threshold_bps']:
                # Strong momentum - execute aggressively
                await self._execute_aggressive(order)
            else:
                # Weak momentum - use TWAP
                await self._execute_twap(order)
                
        except Exception as e:
            self.logger.error(f"Momentum execution failed for order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
    
    async def _execute_mean_reversion(self, order: ExecutionOrder):
        """Execute mean reversion strategy."""
        try:
            market_data = self.market_data.get(order.symbol)
            if not market_data:
                order.status = OrderStatus.FAILED
                return
            
            # Calculate deviation from VWAP
            if market_data.vwap > 0:
                deviation_bps = (market_data.mid - market_data.vwap) / market_data.vwap * 10000
                z_score = abs(deviation_bps) / 20  # Assume 20 bps standard deviation
                
                if z_score > self.exec_config['mean_reversion_zscore']:
                    # High deviation - expect reversion, use passive orders
                    await self._execute_liquidity_seeking(order)
                else:
                    # Normal range - use standard execution
                    await self._execute_immediate(order)
            else:
                await self._execute_immediate(order)
                
        except Exception as e:
            self.logger.error(f"Mean reversion execution failed for order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
    
    async def _simulate_fill(self, order: ExecutionOrder, quantity: float, price: float, venue: str = "primary"):
        """Simulate order fill."""
        try:
            # Create fill record
            fill = ExecutionFill(
                fill_id=f"fill_{order.order_id}_{int(time.time() * 1000000)}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=quantity,
                price=price,
                timestamp=datetime.now(),
                venue=venue,
                fees=quantity * price * 0.0001,  # 0.01% fee simulation
                liquidity_flag="taker"  # Assume taker for simplicity
            )
            
            # Update order
            order.filled_quantity += quantity
            order.total_fees += fill.fees
            
            if order.filled_quantity == 0:
                order.avg_fill_price = price
            else:
                # Update weighted average
                total_value = order.avg_fill_price * (order.filled_quantity - quantity) + price * quantity
                order.avg_fill_price = total_value / order.filled_quantity
            
            # Update order status
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            
            # Store fill
            self.fill_history.append(fill)
            
            # Calculate performance metrics
            if order.benchmark_price:
                order.implementation_shortfall = (order.avg_fill_price - order.benchmark_price) / order.benchmark_price * 10000
                if order.side == 'sell':
                    order.implementation_shortfall *= -1  # Reverse for sell orders
            
            self.logger.debug(f"Order fill simulated: {fill.fill_id} {quantity}@{price}")
            
        except Exception as e:
            self.logger.error(f"Fill simulation failed: {e}")
    
    async def _cancel_all_orders(self):
        """Cancel all active orders."""
        try:
            order_ids = list(self.active_orders.keys())
            for order_id in order_ids:
                await self.cancel_order(order_id)
            
            self.logger.info(f"Cancelled {len(order_ids)} active orders")
            
        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {e}")
    
    async def _update_metrics(self):
        """Update execution metrics."""
        try:
            if not self.order_history:
                return
            
            # Calculate metrics from recent orders
            recent_orders = list(self.order_history)[-1000:]  # Last 1000 orders
            
            total_orders = len(recent_orders)
            filled_orders = sum(1 for order in recent_orders if order.status == OrderStatus.FILLED)
            cancelled_orders = sum(1 for order in recent_orders if order.status == OrderStatus.CANCELLED)
            rejected_orders = sum(1 for order in recent_orders if order.status == OrderStatus.REJECTED)
            
            self.metrics.total_orders = total_orders
            self.metrics.filled_orders = filled_orders
            self.metrics.cancelled_orders = cancelled_orders
            self.metrics.rejected_orders = rejected_orders
            
            if total_orders > 0:
                self.metrics.fill_rate = filled_orders / total_orders
            
            # Latency metrics
            latencies = [order.execution_latency_ms for order in recent_orders if order.execution_latency_ms > 0]
            if latencies:
                self.metrics.avg_execution_latency_ms = np.mean(latencies)
                self.metrics.p95_execution_latency_ms = np.percentile(latencies, 95)
                self.metrics.p99_execution_latency_ms = np.percentile(latencies, 99)
            
            # Performance metrics
            filled_orders_list = [order for order in recent_orders if order.status == OrderStatus.FILLED]
            if filled_orders_list:
                shortfalls = [order.implementation_shortfall for order in filled_orders_list if order.implementation_shortfall != 0]
                if shortfalls:
                    self.metrics.avg_implementation_shortfall_bps = np.mean(shortfalls)
                
                fees = [order.total_fees for order in filled_orders_list]
                total_value = sum(order.avg_fill_price * order.filled_quantity for order in filled_orders_list)
                if total_value > 0:
                    self.metrics.total_fees_paid = sum(fees)
                    self.metrics.avg_fee_rate_bps = (sum(fees) / total_value) * 10000
            
        except Exception as e:
            self.logger.error(f"Metrics update failed: {e}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution engine summary."""
        try:
            return {
                'running': self.running,
                'active_orders': len(self.active_orders),
                'execution_tasks': len(self.execution_tasks),
                'total_orders_processed': self.metrics.total_orders,
                'fill_rate': self.metrics.fill_rate,
                'avg_execution_latency_ms': self.metrics.avg_execution_latency_ms,
                'p95_latency_ms': self.metrics.p95_execution_latency_ms,
                'p99_latency_ms': self.metrics.p99_execution_latency_ms,
                'avg_implementation_shortfall_bps': self.metrics.avg_implementation_shortfall_bps,
                'total_fees_paid': self.metrics.total_fees_paid,
                'avg_fee_rate_bps': self.metrics.avg_fee_rate_bps
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate execution summary: {e}")
            return {'error': 'Unable to generate summary'}
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific order."""
        try:
            # Check active orders first
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
            else:
                # Check order history
                for order in self.order_history:
                    if order.order_id == order_id:
                        break
                else:
                    return None
            
            return {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'filled_quantity': order.filled_quantity,
                'avg_fill_price': order.avg_fill_price,
                'status': order.status.value,
                'strategy': order.strategy.value,
                'execution_latency_ms': order.execution_latency_ms,
                'implementation_shortfall_bps': order.implementation_shortfall,
                'total_fees': order.total_fees,
                'created_timestamp': order.created_timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {e}")
            return None