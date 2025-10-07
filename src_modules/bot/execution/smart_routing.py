"""
Smart Order Routing and Execution Optimization.

This module provides sophisticated order routing and execution strategies
to minimize slippage, reduce market impact, and optimize fill rates:

- TWAP (Time-Weighted Average Price) execution
- VWAP (Volume-Weighted Average Price) execution
- Iceberg order management
- Smart order splitting and timing
- Market impact minimization
- Liquidity detection and utilization
- Execution cost analysis
- Dynamic routing based on market conditions

The system analyzes market microstructure, order book depth, and historical
patterns to determine optimal execution strategies for different order sizes
and market conditions.
"""

import asyncio
import threading
import time
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import statistics

from .order_management import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce, OrderPriority,
    OrderManager, OrderBook
)
from ..utils.logging import TradingLogger


class ExecutionStrategy(Enum):
    """Execution strategy enumeration."""
    IMMEDIATE = "immediate"           # Market order, immediate execution
    PASSIVE = "passive"              # Post-only limit orders
    AGGRESSIVE = "aggressive"        # Take liquidity aggressively
    TWAP = "time_weighted_average"   # Time-weighted average price
    VWAP = "volume_weighted_average" # Volume-weighted average price
    ICEBERG = "iceberg"             # Large order hidden execution
    ADAPTIVE = "adaptive"            # Adaptive based on conditions
    POV = "participation_of_volume"  # Percentage of volume strategy
    ARRIVAL_PRICE = "arrival_price"  # Minimize slippage from arrival price


class MarketCondition(Enum):
    """Market condition enumeration."""
    QUIET = "quiet"           # Low volatility, good liquidity
    ACTIVE = "active"         # Normal market conditions
    VOLATILE = "volatile"     # High volatility
    STRESSED = "stressed"     # Poor liquidity, wide spreads
    TRENDING = "trending"     # Strong directional movement
    RANGING = "ranging"       # Sideways price action


@dataclass
class ExecutionPlan:
    """
    Execution plan for complex orders.
    
    This class defines how a large order should be broken down
    and executed over time to minimize market impact.
    """
    
    plan_id: str
    parent_order_id: str
    strategy: ExecutionStrategy
    total_quantity: Decimal
    symbol: str
    side: OrderSide
    
    # Execution parameters
    max_participation_rate: float = 0.1    # Max % of volume
    max_slice_size: Optional[Decimal] = None
    min_slice_size: Optional[Decimal] = None
    time_horizon: Optional[timedelta] = None
    
    # Timing parameters
    slice_interval: timedelta = timedelta(seconds=30)
    randomize_timing: bool = True
    timing_variance: float = 0.2
    
    # Price parameters
    price_limit: Optional[Decimal] = None
    max_slippage_bps: Optional[float] = None
    aggression_factor: float = 0.5
    
    # Adaptive parameters
    market_condition: Optional[MarketCondition] = None
    urgency: float = 0.5  # 0=patient, 1=urgent
    
    # State tracking
    executed_quantity: Decimal = Decimal('0')
    remaining_quantity: Optional[Decimal] = None
    child_orders: List[str] = field(default_factory=list)
    is_active: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.total_quantity
        if self.plan_id is None:
            self.plan_id = str(uuid.uuid4())
    
    @property
    def completion_rate(self) -> float:
        """Get execution completion rate (0-1)."""
        if self.total_quantity <= 0:
            return 0.0
        return float(self.executed_quantity / self.total_quantity)
    
    @property
    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.remaining_quantity <= 0
    
    def update_execution(self, filled_quantity: Decimal) -> None:
        """Update execution progress."""
        self.executed_quantity += filled_quantity
        self.remaining_quantity = self.total_quantity - self.executed_quantity
        
        if self.is_complete and not self.completed_at:
            self.completed_at = datetime.now()


@dataclass
class MarketMetrics:
    """Market microstructure metrics for execution decisions."""
    
    symbol: str
    timestamp: datetime
    
    # Price metrics
    mid_price: Decimal
    bid_price: Decimal
    ask_price: Decimal
    spread_bps: float
    
    # Liquidity metrics
    bid_size: Decimal
    ask_size: Decimal
    market_depth: Dict[str, Decimal]  # {'1%': depth, '2%': depth, etc.}
    
    # Volume metrics
    volume_1m: Decimal
    volume_5m: Decimal
    volume_15m: Decimal
    avg_trade_size: Decimal
    
    # Volatility metrics
    volatility_1m: float
    volatility_5m: float
    price_impact_1bps: float  # Price impact per 1bps of volume
    
    # Order flow metrics
    buy_pressure: float       # Buy volume / Total volume
    sell_pressure: float      # Sell volume / Total volume
    order_imbalance: float    # (Bid size - Ask size) / (Bid size + Ask size)
    
    # Market condition
    condition: MarketCondition
    liquidity_score: float    # 0-1, higher = better liquidity
    execution_difficulty: float  # 0-1, higher = more difficult


class MarketAnalyzer:
    """
    Market microstructure analyzer for execution optimization.
    
    This class analyzes real-time market data to assess execution
    conditions and optimize routing strategies.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("MarketAnalyzer")
        
        # Historical data storage
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Market metrics cache
        self.metrics_cache: Dict[str, MarketMetrics] = {}
        self.cache_ttl = timedelta(seconds=30)
        
    def _default_config(self) -> Dict:
        """Default configuration for market analyzer."""
        return {
            'volatility_window': 60,          # Seconds for volatility calculation
            'volume_window': 300,             # Seconds for volume metrics
            'depth_levels': [0.01, 0.02, 0.05, 0.1],  # Depth analysis levels
            'min_liquidity_score': 0.3,      # Minimum acceptable liquidity
            'high_volatility_threshold': 0.02, # 2% per hour
            'wide_spread_threshold': 50,      # 50 bps
            'update_interval': 5,             # Seconds between updates
        }
    
    def analyze_market_conditions(
        self,
        symbol: str,
        order_book: OrderBook,
        recent_trades: Optional[List[Dict]] = None,
        force_refresh: bool = False
    ) -> MarketMetrics:
        """
        Analyze current market conditions for a symbol.
        
        Args:
            symbol: Trading symbol
            order_book: Current order book
            recent_trades: Recent trade data
            force_refresh: Force cache refresh
            
        Returns:
            Market metrics analysis
        """
        # Check cache
        if not force_refresh and symbol in self.metrics_cache:
            cached_metrics = self.metrics_cache[symbol]
            if datetime.now() - cached_metrics.timestamp < self.cache_ttl:
                return cached_metrics
        
        # Calculate basic price metrics
        mid_price = order_book.mid_price or Decimal('0')
        bid_price = order_book.best_bid or Decimal('0')
        ask_price = order_book.best_ask or Decimal('0')
        spread_bps = order_book.spread_bps or 0.0
        
        # Calculate liquidity metrics
        bid_size = order_book.bids[0][1] if order_book.bids else Decimal('0')
        ask_size = order_book.asks[0][1] if order_book.asks else Decimal('0')
        market_depth = self._calculate_market_depth(order_book)
        
        # Calculate volume metrics
        volume_metrics = self._calculate_volume_metrics(symbol, recent_trades)
        
        # Calculate volatility metrics
        volatility_metrics = self._calculate_volatility_metrics(symbol)
        
        # Calculate order flow metrics
        order_flow = self._calculate_order_flow_metrics(order_book, recent_trades)
        
        # Determine market condition
        market_condition = self._classify_market_condition(
            spread_bps, volatility_metrics, volume_metrics, market_depth
        )
        
        # Calculate scores
        liquidity_score = self._calculate_liquidity_score(market_depth, spread_bps, volume_metrics)
        execution_difficulty = self._calculate_execution_difficulty(
            market_condition, liquidity_score, volatility_metrics
        )
        
        # Create metrics object
        metrics = MarketMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            mid_price=mid_price,
            bid_price=bid_price,
            ask_price=ask_price,
            spread_bps=spread_bps,
            bid_size=bid_size,
            ask_size=ask_size,
            market_depth=market_depth,
            volume_1m=volume_metrics.get('volume_1m', Decimal('0')),
            volume_5m=volume_metrics.get('volume_5m', Decimal('0')),
            volume_15m=volume_metrics.get('volume_15m', Decimal('0')),
            avg_trade_size=volume_metrics.get('avg_trade_size', Decimal('0')),
            volatility_1m=volatility_metrics.get('volatility_1m', 0.0),
            volatility_5m=volatility_metrics.get('volatility_5m', 0.0),
            price_impact_1bps=volatility_metrics.get('price_impact', 0.0),
            buy_pressure=order_flow.get('buy_pressure', 0.5),
            sell_pressure=order_flow.get('sell_pressure', 0.5),
            order_imbalance=order_flow.get('order_imbalance', 0.0),
            condition=market_condition,
            liquidity_score=liquidity_score,
            execution_difficulty=execution_difficulty
        )
        
        # Cache metrics
        self.metrics_cache[symbol] = metrics
        
        return metrics
    
    def _calculate_market_depth(self, order_book: OrderBook) -> Dict[str, Decimal]:
        """Calculate market depth at various price levels."""
        depth = {}
        
        if not order_book.mid_price:
            return depth
        
        for level in self.config['depth_levels']:
            bid_depth = Decimal('0')
            ask_depth = Decimal('0')
            
            # Calculate bid side depth
            min_bid_price = order_book.mid_price * (1 - Decimal(str(level)))
            for price, size in order_book.bids:
                if price >= min_bid_price:
                    bid_depth += size
                else:
                    break
            
            # Calculate ask side depth
            max_ask_price = order_book.mid_price * (1 + Decimal(str(level)))
            for price, size in order_book.asks:
                if price <= max_ask_price:
                    ask_depth += size
                else:
                    break
            
            depth[f'{level*100:.1f}%'] = min(bid_depth, ask_depth)
        
        return depth
    
    def _calculate_volume_metrics(self, symbol: str, recent_trades: Optional[List[Dict]]) -> Dict[str, Any]:
        """Calculate volume-based metrics."""
        metrics = {
            'volume_1m': Decimal('0'),
            'volume_5m': Decimal('0'),
            'volume_15m': Decimal('0'),
            'avg_trade_size': Decimal('0'),
            'trade_count': 0
        }
        
        if not recent_trades:
            return metrics
        
        now = datetime.now()
        volumes_1m = []
        volumes_5m = []
        volumes_15m = []
        trade_sizes = []
        
        for trade in recent_trades:
            trade_time = trade.get('timestamp')
            if not trade_time:
                continue
            
            if isinstance(trade_time, str):
                trade_time = datetime.fromisoformat(trade_time)
            
            time_diff = (now - trade_time).total_seconds()
            trade_size = Decimal(str(trade.get('amount', 0)))
            
            trade_sizes.append(trade_size)
            
            if time_diff <= 60:
                volumes_1m.append(trade_size)
            if time_diff <= 300:
                volumes_5m.append(trade_size)
            if time_diff <= 900:
                volumes_15m.append(trade_size)
        
        metrics['volume_1m'] = sum(volumes_1m)
        metrics['volume_5m'] = sum(volumes_5m)
        metrics['volume_15m'] = sum(volumes_15m)
        metrics['avg_trade_size'] = sum(trade_sizes) / len(trade_sizes) if trade_sizes else Decimal('0')
        metrics['trade_count'] = len(trade_sizes)
        
        return metrics
    
    def _calculate_volatility_metrics(self, symbol: str) -> Dict[str, float]:
        """Calculate volatility-based metrics."""
        metrics = {
            'volatility_1m': 0.0,
            'volatility_5m': 0.0,
            'price_impact': 0.0
        }
        
        price_history = list(self.price_history[symbol])
        if len(price_history) < 10:
            return metrics
        
        # Calculate returns
        returns = []
        for i in range(1, len(price_history)):
            if price_history[i-1] > 0:
                ret = float(price_history[i] / price_history[i-1] - 1)
                returns.append(ret)
        
        if len(returns) < 5:
            return metrics
        
        # 1-minute volatility (last 60 points, assuming 1-second intervals)
        recent_returns = returns[-60:] if len(returns) >= 60 else returns
        if recent_returns:
            metrics['volatility_1m'] = statistics.stdev(recent_returns) * math.sqrt(60)
        
        # 5-minute volatility
        recent_returns = returns[-300:] if len(returns) >= 300 else returns
        if recent_returns:
            metrics['volatility_5m'] = statistics.stdev(recent_returns) * math.sqrt(300)
        
        # Simple price impact estimate (very basic)
        if len(returns) >= 10:
            abs_returns = [abs(r) for r in returns[-10:]]
            metrics['price_impact'] = statistics.mean(abs_returns) * 10000  # in bps
        
        return metrics
    
    def _calculate_order_flow_metrics(
        self,
        order_book: OrderBook,
        recent_trades: Optional[List[Dict]]
    ) -> Dict[str, float]:
        """Calculate order flow metrics."""
        metrics = {
            'buy_pressure': 0.5,
            'sell_pressure': 0.5,
            'order_imbalance': 0.0
        }
        
        # Order book imbalance
        if order_book.bids and order_book.asks:
            bid_size = order_book.bids[0][1]
            ask_size = order_book.asks[0][1]
            total_size = bid_size + ask_size
            
            if total_size > 0:
                metrics['order_imbalance'] = float((bid_size - ask_size) / total_size)
        
        # Trade flow analysis
        if recent_trades:
            buy_volume = Decimal('0')
            sell_volume = Decimal('0')
            
            for trade in recent_trades[-50:]:  # Last 50 trades
                volume = Decimal(str(trade.get('amount', 0)))
                side = trade.get('side', 'unknown')
                
                if side == 'buy':
                    buy_volume += volume
                elif side == 'sell':
                    sell_volume += volume
            
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                metrics['buy_pressure'] = float(buy_volume / total_volume)
                metrics['sell_pressure'] = float(sell_volume / total_volume)
        
        return metrics
    
    def _classify_market_condition(
        self,
        spread_bps: float,
        volatility_metrics: Dict[str, float],
        volume_metrics: Dict[str, Any],
        market_depth: Dict[str, Decimal]
    ) -> MarketCondition:
        """Classify current market condition."""
        volatility_1m = volatility_metrics.get('volatility_1m', 0.0)
        volume_1m = volume_metrics.get('volume_1m', Decimal('0'))
        
        # Check for stressed conditions
        if (spread_bps > self.config['wide_spread_threshold'] or
            sum([float(d) for d in market_depth.values()]) < 1000):  # Low depth
            return MarketCondition.STRESSED
        
        # Check for volatile conditions
        if volatility_1m > self.config['high_volatility_threshold']:
            return MarketCondition.VOLATILE
        
        # Check for quiet conditions
        if volatility_1m < 0.005 and volume_1m < Decimal('10000'):  # Low vol, low volume
            return MarketCondition.QUIET
        
        # Determine trending vs ranging (simplified)
        price_impact = volatility_metrics.get('price_impact', 0.0)
        if price_impact > 10:  # High price impact suggests trending
            return MarketCondition.TRENDING
        elif price_impact < 5:
            return MarketCondition.RANGING
        
        return MarketCondition.ACTIVE
    
    def _calculate_liquidity_score(
        self,
        market_depth: Dict[str, Decimal],
        spread_bps: float,
        volume_metrics: Dict[str, Any]
    ) -> float:
        """Calculate liquidity score (0-1, higher = better)."""
        # Depth component (40% weight)
        depth_scores = []
        for level, depth in market_depth.items():
            # Normalize depth (very approximate)
            normalized_depth = min(float(depth) / 10000, 1.0)
            depth_scores.append(normalized_depth)
        
        avg_depth_score = statistics.mean(depth_scores) if depth_scores else 0.0
        
        # Spread component (30% weight)
        spread_score = max(0, 1 - spread_bps / 100)  # 100bps = 0 score
        
        # Volume component (30% weight)
        volume_1m = float(volume_metrics.get('volume_1m', 0))
        volume_score = min(volume_1m / 50000, 1.0)  # 50k volume = 1.0 score
        
        # Weighted average
        liquidity_score = (
            0.4 * avg_depth_score +
            0.3 * spread_score +
            0.3 * volume_score
        )
        
        return max(0.0, min(1.0, liquidity_score))
    
    def _calculate_execution_difficulty(
        self,
        market_condition: MarketCondition,
        liquidity_score: float,
        volatility_metrics: Dict[str, float]
    ) -> float:
        """Calculate execution difficulty (0-1, higher = more difficult)."""
        # Base difficulty by market condition
        condition_difficulty = {
            MarketCondition.QUIET: 0.2,
            MarketCondition.ACTIVE: 0.4,
            MarketCondition.VOLATILE: 0.7,
            MarketCondition.STRESSED: 0.9,
            MarketCondition.TRENDING: 0.6,
            MarketCondition.RANGING: 0.3
        }
        
        base_difficulty = condition_difficulty.get(market_condition, 0.5)
        
        # Adjust for liquidity
        liquidity_adjustment = (1 - liquidity_score) * 0.3
        
        # Adjust for volatility
        volatility_1m = volatility_metrics.get('volatility_1m', 0.0)
        volatility_adjustment = min(volatility_1m * 10, 0.3)  # Cap at 0.3
        
        total_difficulty = base_difficulty + liquidity_adjustment + volatility_adjustment
        
        return max(0.0, min(1.0, total_difficulty))
    
    def update_price_history(self, symbol: str, price: Decimal) -> None:
        """Update price history for a symbol."""
        self.price_history[symbol].append(price)
    
    def update_volume_history(self, symbol: str, volume: Decimal) -> None:
        """Update volume history for a symbol."""
        self.volume_history[symbol].append(volume)
    
    def update_trade_history(self, symbol: str, trade: Dict) -> None:
        """Update trade history for a symbol."""
        self.trade_history[symbol].append(trade)


class SmartRouter:
    """
    Smart order routing system for optimal execution.
    
    This class determines the best execution strategy based on
    order characteristics and current market conditions.
    """
    
    def __init__(self, order_manager: OrderManager, config: Optional[Dict] = None):
        self.order_manager = order_manager
        self.config = config or self._default_config()
        self.logger = TradingLogger("SmartRouter")
        
        # Market analyzer
        self.market_analyzer = MarketAnalyzer(self.config.get('market_analyzer', {}))
        
        # Execution plans
        self.execution_plans: Dict[str, ExecutionPlan] = {}
        
        # Execution engines
        self.execution_engines = {
            ExecutionStrategy.TWAP: self._execute_twap,
            ExecutionStrategy.VWAP: self._execute_vwap,
            ExecutionStrategy.ICEBERG: self._execute_iceberg,
            ExecutionStrategy.ADAPTIVE: self._execute_adaptive,
            ExecutionStrategy.POV: self._execute_pov
        }
        
        # Background execution thread
        self.execution_thread = None
        self.is_running = False
        
    def _default_config(self) -> Dict:
        """Default configuration for smart router."""
        return {
            'large_order_threshold': Decimal('50000'),    # USD value
            'max_market_impact_bps': 20,                  # 20 bps max impact
            'default_participation_rate': 0.1,            # 10% of volume
            'min_execution_time': timedelta(minutes=5),   # Minimum execution time
            'max_execution_time': timedelta(hours=4),     # Maximum execution time
            'slice_randomization': 0.2,                   # 20% timing variance
            'aggressive_threshold': 0.8,                  # Urgency threshold
            'passive_threshold': 0.3,                     # Patience threshold
            'market_analyzer': {},
            'execution_interval': 10,                     # Seconds between checks
        }
    
    def route_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.LIMIT,
        strategy: Optional[ExecutionStrategy] = None,
        urgency: float = 0.5,
        max_slippage_bps: Optional[float] = None,
        time_horizon: Optional[timedelta] = None,
        **kwargs
    ) -> Union[Order, ExecutionPlan]:
        """
        Route an order using optimal execution strategy.
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            order_type: Base order type
            strategy: Execution strategy (auto-selected if None)
            urgency: Urgency level (0=patient, 1=urgent)
            max_slippage_bps: Maximum acceptable slippage
            time_horizon: Target execution time
            **kwargs: Additional parameters
            
        Returns:
            Either a simple Order or complex ExecutionPlan
        """
        # Get market analysis
        order_book = self.order_manager.get_order_book(symbol)
        if not order_book:
            raise ValueError(f"No order book available for {symbol}")
        
        market_metrics = self.market_analyzer.analyze_market_conditions(symbol, order_book)
        
        # Determine if this needs complex execution
        estimated_value = quantity * (market_metrics.mid_price or Decimal('1'))
        needs_complex_execution = (
            estimated_value > self.config['large_order_threshold'] or
            strategy in [ExecutionStrategy.TWAP, ExecutionStrategy.VWAP, ExecutionStrategy.ICEBERG]
        )
        
        if not needs_complex_execution:
            # Simple execution
            return self._create_simple_order(
                symbol, side, quantity, order_type, market_metrics, urgency, **kwargs
            )
        
        # Complex execution - create execution plan
        if strategy is None:
            strategy = self._select_execution_strategy(
                quantity, estimated_value, market_metrics, urgency
            )
        
        execution_plan = self._create_execution_plan(
            symbol, side, quantity, strategy, market_metrics,
            urgency, max_slippage_bps, time_horizon, **kwargs
        )
        
        # Start execution
        self._start_execution_plan(execution_plan)
        
        return execution_plan
    
    def _create_simple_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType,
        market_metrics: MarketMetrics,
        urgency: float,
        **kwargs
    ) -> Order:
        """Create a simple order for immediate execution."""
        
        # Determine price based on urgency and market conditions
        if order_type == OrderType.MARKET:
            price = None
        elif order_type == OrderType.LIMIT:
            if urgency > self.config['aggressive_threshold']:
                # Aggressive pricing
                aggression = 0.8
            elif urgency < self.config['passive_threshold']:
                # Passive pricing
                aggression = 0.2
            else:
                # Balanced pricing
                aggression = urgency
            
            price = self.order_manager.get_optimal_price(symbol, side, aggression)
            if not price:
                price = market_metrics.ask_price if side == OrderSide.BUY else market_metrics.bid_price
        else:
            price = kwargs.get('price')
        
        # Create order
        order = self.order_manager.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            **kwargs
        )
        
        self.logger.info(f"Created simple order {order.order_id}: {side.value} {quantity} {symbol}")
        
        return order
    
    def _select_execution_strategy(
        self,
        quantity: Decimal,
        estimated_value: Decimal,
        market_metrics: MarketMetrics,
        urgency: float
    ) -> ExecutionStrategy:
        """Automatically select optimal execution strategy."""
        
        # High urgency -> aggressive strategies
        if urgency > 0.8:
            if market_metrics.liquidity_score > 0.7:
                return ExecutionStrategy.AGGRESSIVE
            else:
                return ExecutionStrategy.ADAPTIVE
        
        # Low urgency -> passive strategies
        if urgency < 0.3:
            if market_metrics.condition == MarketCondition.QUIET:
                return ExecutionStrategy.TWAP
            else:
                return ExecutionStrategy.ADAPTIVE
        
        # Large orders
        if estimated_value > self.config['large_order_threshold'] * 5:
            if market_metrics.volume_5m > quantity * 10:  # Good volume
                return ExecutionStrategy.VWAP
            else:
                return ExecutionStrategy.ICEBERG
        
        # Market conditions
        if market_metrics.condition == MarketCondition.VOLATILE:
            return ExecutionStrategy.ADAPTIVE
        elif market_metrics.condition == MarketCondition.STRESSED:
            return ExecutionStrategy.ICEBERG
        elif market_metrics.condition == MarketCondition.TRENDING:
            return ExecutionStrategy.POV
        
        # Default to adaptive
        return ExecutionStrategy.ADAPTIVE
    
    def _create_execution_plan(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        strategy: ExecutionStrategy,
        market_metrics: MarketMetrics,
        urgency: float,
        max_slippage_bps: Optional[float],
        time_horizon: Optional[timedelta],
        **kwargs
    ) -> ExecutionPlan:
        """Create detailed execution plan."""
        
        # Calculate time horizon if not provided
        if time_horizon is None:
            if urgency > 0.8:
                time_horizon = self.config['min_execution_time']
            elif urgency < 0.2:
                time_horizon = self.config['max_execution_time']
            else:
                # Scale between min and max based on urgency
                min_time = self.config['min_execution_time'].total_seconds()
                max_time = self.config['max_execution_time'].total_seconds()
                target_time = max_time - (urgency * (max_time - min_time))
                time_horizon = timedelta(seconds=target_time)
        
        # Calculate slice parameters
        if strategy == ExecutionStrategy.TWAP:
            slice_interval = timedelta(seconds=max(30, time_horizon.total_seconds() / 20))
            max_slice_size = quantity / 10
        elif strategy == ExecutionStrategy.VWAP:
            slice_interval = timedelta(seconds=60)
            # Size based on expected volume
            expected_volume = market_metrics.volume_5m * (time_horizon.total_seconds() / 300)
            max_slice_size = expected_volume * Decimal(str(self.config['default_participation_rate']))
        elif strategy == ExecutionStrategy.ICEBERG:
            slice_interval = timedelta(seconds=45)
            max_slice_size = min(quantity / 20, market_metrics.avg_trade_size * 5)
        else:
            slice_interval = timedelta(seconds=30)
            max_slice_size = quantity / 15
        
        # Create execution plan
        plan = ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            parent_order_id=str(uuid.uuid4()),
            strategy=strategy,
            total_quantity=quantity,
            symbol=symbol,
            side=side,
            max_participation_rate=self.config['default_participation_rate'],
            max_slice_size=max_slice_size,
            min_slice_size=max_slice_size / 5,
            time_horizon=time_horizon,
            slice_interval=slice_interval,
            randomize_timing=True,
            timing_variance=self.config['slice_randomization'],
            price_limit=kwargs.get('price_limit'),
            max_slippage_bps=max_slippage_bps,
            aggression_factor=urgency,
            market_condition=market_metrics.condition,
            urgency=urgency
        )
        
        # Store plan
        self.execution_plans[plan.plan_id] = plan
        
        self.logger.info(
            f"Created execution plan {plan.plan_id}: "
            f"{strategy.value} {quantity} {symbol} over {time_horizon}"
        )
        
        return plan
    
    def _start_execution_plan(self, plan: ExecutionPlan) -> None:
        """Start executing an execution plan."""
        plan.is_active = True
        plan.started_at = datetime.now()
        
        # Start background execution if not running
        if not self.is_running:
            self.start_execution_engine()
        
        self.logger.info(f"Started execution plan {plan.plan_id}")
    
    def start_execution_engine(self) -> None:
        """Start background execution engine."""
        if self.is_running:
            return
        
        self.is_running = True
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        
        self.logger.info("Started execution engine")
    
    def stop_execution_engine(self) -> None:
        """Stop background execution engine."""
        self.is_running = False
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5)
        
        self.logger.info("Stopped execution engine")
    
    def _execution_loop(self) -> None:
        """Main execution loop."""
        while self.is_running:
            try:
                # Process active execution plans
                for plan in list(self.execution_plans.values()):
                    if plan.is_active and not plan.is_complete:
                        self._process_execution_plan(plan)
                
                # Sleep until next check
                time.sleep(self.config['execution_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                time.sleep(5)  # Error backoff
    
    def _process_execution_plan(self, plan: ExecutionPlan) -> None:
        """Process a single execution plan."""
        try:
            # Get execution engine for strategy
            execution_engine = self.execution_engines.get(plan.strategy)
            if not execution_engine:
                self.logger.error(f"No execution engine for strategy {plan.strategy.value}")
                return
            
            # Execute next slice
            execution_engine(plan)
            
        except Exception as e:
            self.logger.error(f"Error processing execution plan {plan.plan_id}: {e}")
    
    def _execute_twap(self, plan: ExecutionPlan) -> None:
        """Execute TWAP (Time-Weighted Average Price) strategy."""
        now = datetime.now()
        
        # Check if it's time for next slice
        if plan.child_orders:
            last_order_time = max(
                self.order_manager.get_order(order_id).created_at
                for order_id in plan.child_orders
                if self.order_manager.get_order(order_id)
            )
            
            if now - last_order_time < plan.slice_interval:
                return
        
        # Calculate slice size
        remaining_time = plan.time_horizon - (now - plan.started_at)
        if remaining_time.total_seconds() <= 0:
            # Time expired, execute remaining quantity aggressively
            slice_size = plan.remaining_quantity
        else:
            slices_remaining = max(1, remaining_time.total_seconds() / plan.slice_interval.total_seconds())
            slice_size = plan.remaining_quantity / Decimal(str(slices_remaining))
        
        # Apply size limits
        slice_size = min(slice_size, plan.max_slice_size or slice_size)
        slice_size = max(slice_size, plan.min_slice_size or slice_size / 10)
        
        # Don't create tiny orders
        if slice_size < plan.total_quantity * Decimal('0.01'):
            return
        
        # Create child order
        self._create_child_order(plan, slice_size, aggression=0.4)
    
    def _execute_vwap(self, plan: ExecutionPlan) -> None:
        """Execute VWAP (Volume-Weighted Average Price) strategy."""
        # Get recent volume data
        order_book = self.order_manager.get_order_book(plan.symbol)
        if not order_book:
            return
        
        market_metrics = self.market_analyzer.analyze_market_conditions(plan.symbol, order_book)
        
        # Calculate target participation rate
        recent_volume = market_metrics.volume_1m  # 1-minute volume
        if recent_volume <= 0:
            return
        
        # Target slice size based on volume participation
        target_rate = min(plan.max_participation_rate, 0.2)  # Cap at 20%
        slice_size = recent_volume * Decimal(str(target_rate))
        
        # Apply limits
        slice_size = min(slice_size, plan.remaining_quantity)
        slice_size = min(slice_size, plan.max_slice_size or slice_size)
        slice_size = max(slice_size, plan.min_slice_size or slice_size / 5)
        
        # Check timing
        now = datetime.now()
        if plan.child_orders:
            last_order = max(
                self.order_manager.get_order(order_id).created_at
                for order_id in plan.child_orders
                if self.order_manager.get_order(order_id)
            )
            
            if now - last_order < timedelta(seconds=30):
                return
        
        # Create child order with moderate aggression
        self._create_child_order(plan, slice_size, aggression=0.5)
    
    def _execute_iceberg(self, plan: ExecutionPlan) -> None:
        """Execute Iceberg strategy (large order hidden execution)."""
        # Check existing child orders
        active_child_orders = [
            self.order_manager.get_order(order_id)
            for order_id in plan.child_orders
            if self.order_manager.get_order(order_id) and self.order_manager.get_order(order_id).is_open
        ]
        
        # Only maintain 1-2 active child orders at a time
        if len(active_child_orders) >= 2:
            return
        
        # Calculate slice size (small to hide intent)
        slice_size = min(
            plan.max_slice_size or plan.remaining_quantity / 20,
            plan.remaining_quantity
        )
        
        # Make it even smaller for iceberg
        slice_size = slice_size / 2
        
        # Ensure minimum size
        if slice_size < plan.min_slice_size or plan.min_slice_size or slice_size:
            slice_size = plan.min_slice_size or slice_size
        
        # Create passive child order
        self._create_child_order(plan, slice_size, aggression=0.2)
    
    def _execute_adaptive(self, plan: ExecutionPlan) -> None:
        """Execute Adaptive strategy (adjusts based on market conditions)."""
        # Analyze current market conditions
        order_book = self.order_manager.get_order_book(plan.symbol)
        if not order_book:
            return
        
        market_metrics = self.market_analyzer.analyze_market_conditions(plan.symbol, order_book)
        
        # Adjust strategy based on conditions
        if market_metrics.condition == MarketCondition.QUIET:
            # Use TWAP-like approach
            self._execute_twap(plan)
        elif market_metrics.condition == MarketCondition.VOLATILE:
            # Use smaller, more frequent orders
            plan.max_slice_size = min(plan.max_slice_size or plan.total_quantity, plan.total_quantity / 30)
            self._execute_iceberg(plan)
        elif market_metrics.condition == MarketCondition.STRESSED:
            # Very conservative approach
            plan.max_slice_size = min(plan.max_slice_size or plan.total_quantity, plan.total_quantity / 50)
            self._execute_iceberg(plan)
        else:
            # Normal conditions - use VWAP approach
            self._execute_vwap(plan)
    
    def _execute_pov(self, plan: ExecutionPlan) -> None:
        """Execute Participation of Volume (POV) strategy."""
        # Similar to VWAP but with fixed participation rate
        order_book = self.order_manager.get_order_book(plan.symbol)
        if not order_book:
            return
        
        market_metrics = self.market_analyzer.analyze_market_conditions(plan.symbol, order_book)
        
        # Calculate slice based on participation rate
        recent_volume = market_metrics.volume_1m
        if recent_volume <= 0:
            return
        
        slice_size = recent_volume * Decimal(str(plan.max_participation_rate))
        slice_size = min(slice_size, plan.remaining_quantity)
        
        # Check timing
        now = datetime.now()
        if plan.child_orders:
            last_order_time = max(
                self.order_manager.get_order(order_id).created_at
                for order_id in plan.child_orders
                if self.order_manager.get_order(order_id)
            )
            
            if now - last_order_time < timedelta(seconds=45):
                return
        
        # Create child order
        self._create_child_order(plan, slice_size, aggression=0.6)
    
    def _create_child_order(
        self,
        plan: ExecutionPlan,
        slice_size: Decimal,
        aggression: float = 0.5
    ) -> Optional[Order]:
        """Create a child order for an execution plan."""
        if slice_size <= 0 or plan.remaining_quantity <= 0:
            return None
        
        # Ensure we don't exceed remaining quantity
        slice_size = min(slice_size, plan.remaining_quantity)
        
        # Get optimal price
        optimal_price = self.order_manager.get_optimal_price(
            plan.symbol, plan.side, aggression
        )
        
        try:
            # Create child order
            child_order = self.order_manager.create_order(
                symbol=plan.symbol,
                side=plan.side,
                order_type=OrderType.LIMIT,
                quantity=slice_size,
                price=optimal_price,
                strategy_id=f"execution_plan_{plan.plan_id}",
                tags={
                    'execution_plan': plan.plan_id,
                    'execution_strategy': plan.strategy.value,
                    'slice_number': str(len(plan.child_orders) + 1)
                }
            )
            
            # Add to plan
            plan.child_orders.append(child_order.order_id)
            
            # Update plan progress (optimistic)
            plan.update_execution(slice_size)
            
            self.logger.info(
                f"Created child order {child_order.order_id} for plan {plan.plan_id}: "
                f"{slice_size} @ {optimal_price}"
            )
            
            return child_order
            
        except Exception as e:
            self.logger.error(f"Failed to create child order for plan {plan.plan_id}: {e}")
            return None
    
    def get_execution_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get execution plan status."""
        plan = self.execution_plans.get(plan_id)
        if not plan:
            return None
        
        # Get child order statuses
        child_order_statuses = []
        total_filled = Decimal('0')
        
        for order_id in plan.child_orders:
            order = self.order_manager.get_order(order_id)
            if order:
                child_order_statuses.append({
                    'order_id': order.order_id,
                    'status': order.status.value,
                    'quantity': str(order.quantity),
                    'filled_quantity': str(order.filled_quantity),
                    'price': str(order.price) if order.price else None,
                    'created_at': order.created_at.isoformat()
                })
                total_filled += order.filled_quantity
        
        # Calculate execution statistics
        if plan.started_at:
            elapsed_time = datetime.now() - plan.started_at
            completion_rate = float(total_filled / plan.total_quantity) if plan.total_quantity > 0 else 0
            
            if plan.time_horizon and plan.time_horizon.total_seconds() > 0:
                time_progress = elapsed_time.total_seconds() / plan.time_horizon.total_seconds()
            else:
                time_progress = 0
        else:
            elapsed_time = timedelta(0)
            completion_rate = 0
            time_progress = 0
        
        return {
            'plan_id': plan.plan_id,
            'strategy': plan.strategy.value,
            'symbol': plan.symbol,
            'side': plan.side.value,
            'total_quantity': str(plan.total_quantity),
            'executed_quantity': str(total_filled),
            'remaining_quantity': str(plan.total_quantity - total_filled),
            'completion_rate': completion_rate,
            'time_progress': time_progress,
            'elapsed_time': str(elapsed_time),
            'is_active': plan.is_active,
            'is_complete': plan.is_complete,
            'child_orders': child_order_statuses,
            'started_at': plan.started_at.isoformat() if plan.started_at else None,
            'completed_at': plan.completed_at.isoformat() if plan.completed_at else None
        }
    
    def cancel_execution_plan(self, plan_id: str, reason: Optional[str] = None) -> bool:
        """Cancel an execution plan."""
        plan = self.execution_plans.get(plan_id)
        if not plan:
            return False
        
        # Cancel all child orders
        cancelled_count = 0
        for order_id in plan.child_orders:
            if self.order_manager.cancel_order(order_id, f"Plan cancelled: {reason or 'User request'}"):
                cancelled_count += 1
        
        # Deactivate plan
        plan.is_active = False
        plan.completed_at = datetime.now()
        
        self.logger.info(
            f"Cancelled execution plan {plan_id}: "
            f"cancelled {cancelled_count} child orders"
        )
        
        return True