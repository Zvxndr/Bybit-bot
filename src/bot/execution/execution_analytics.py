"""
Execution Analytics and Performance Tracking.

This module provides comprehensive analytics for trade execution including:

- Order execution performance metrics
- Slippage analysis and tracking
- Fill rate and timing analysis
- Market impact measurement
- Execution cost analysis
- Strategy performance evaluation
- Real-time execution monitoring
- Historical performance reporting
- Execution optimization insights
- Benchmark comparisons (TWAP, VWAP, etc.)

The system tracks all execution metrics to help optimize trading strategies
and identify areas for improvement in order execution.
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
import sqlite3
import json

from .order_management import Order, OrderStatus, OrderFill, OrderType, OrderSide
from .smart_routing import ExecutionPlan, ExecutionStrategy
from ..utils.logging import TradingLogger


class BenchmarkType(Enum):
    """Benchmark type enumeration."""
    ARRIVAL_PRICE = "arrival_price"     # Price when order was created
    TWAP = "time_weighted_average"      # Time-weighted average price
    VWAP = "volume_weighted_average"    # Volume-weighted average price
    OPEN_PRICE = "open_price"          # Session open price
    CLOSE_PRICE = "close_price"        # Session close price
    MID_PRICE = "mid_price"            # Mid price during execution


@dataclass
class ExecutionMetrics:
    """Container for execution performance metrics."""
    
    order_id: str
    symbol: str
    side: OrderSide
    strategy: Optional[str]
    
    # Order details
    total_quantity: Decimal
    filled_quantity: Decimal
    average_fill_price: Decimal
    
    # Timing metrics
    order_created_at: datetime
    first_fill_at: Optional[datetime]
    last_fill_at: Optional[datetime]
    total_execution_time: Optional[timedelta]
    
    # Price metrics
    arrival_price: Decimal              # Price when order created
    benchmark_prices: Dict[str, Decimal] = field(default_factory=dict)
    
    # Performance metrics
    slippage_bps: float = 0.0          # Slippage in basis points
    market_impact_bps: float = 0.0     # Market impact in basis points
    timing_risk_bps: float = 0.0       # Timing risk vs benchmark
    
    # Cost metrics
    total_fees: Decimal = Decimal('0')
    total_cost: Decimal = Decimal('0') # Total execution cost
    cost_per_share: Decimal = Decimal('0')
    
    # Fill metrics
    fill_rate: float = 0.0             # Percentage filled
    num_fills: int = 0
    avg_fill_size: Decimal = Decimal('0')
    
    # Quality metrics
    participation_rate: float = 0.0     # % of market volume
    aggressiveness_score: float = 0.0   # 0=passive, 1=aggressive
    execution_quality: float = 0.0      # Overall quality score (0-1)
    
    def __post_init__(self):
        # Calculate derived metrics
        if self.total_quantity > 0:
            self.fill_rate = float(self.filled_quantity / self.total_quantity)
        
        if self.num_fills > 0 and self.filled_quantity > 0:
            self.avg_fill_size = self.filled_quantity / self.num_fills
        
        if self.first_fill_at and self.last_fill_at:
            self.total_execution_time = self.last_fill_at - self.first_fill_at
        elif self.first_fill_at:
            self.total_execution_time = self.first_fill_at - self.order_created_at
    
    def calculate_slippage(self, benchmark_type: BenchmarkType = BenchmarkType.ARRIVAL_PRICE) -> float:
        """Calculate slippage against benchmark."""
        benchmark_price = self.benchmark_prices.get(benchmark_type.value, self.arrival_price)
        
        if benchmark_price <= 0 or self.average_fill_price <= 0:
            return 0.0
        
        if self.side == OrderSide.BUY:
            # For buy orders, slippage is positive if we paid more than benchmark
            slippage = (self.average_fill_price - benchmark_price) / benchmark_price
        else:
            # For sell orders, slippage is positive if we received less than benchmark
            slippage = (benchmark_price - self.average_fill_price) / benchmark_price
        
        return float(slippage * 10000)  # Convert to basis points
    
    def calculate_implementation_shortfall(self) -> Dict[str, float]:
        """Calculate implementation shortfall components."""
        if self.arrival_price <= 0 or self.total_quantity <= 0:
            return {}
        
        # Market impact (difference between arrival price and average fill price)
        if self.side == OrderSide.BUY:
            market_impact = (self.average_fill_price - self.arrival_price) / self.arrival_price
        else:
            market_impact = (self.arrival_price - self.average_fill_price) / self.arrival_price
        
        # Timing risk (unfilled portion at current market price vs arrival price)
        unfilled_quantity = self.total_quantity - self.filled_quantity
        timing_risk = 0.0
        
        if unfilled_quantity > 0:
            # This would require current market price - simplified for now
            current_price = self.arrival_price  # Placeholder
            
            if self.side == OrderSide.BUY:
                timing_risk = (current_price - self.arrival_price) / self.arrival_price
            else:
                timing_risk = (self.arrival_price - current_price) / self.arrival_price
            
            timing_risk *= float(unfilled_quantity / self.total_quantity)
        
        # Total implementation shortfall
        implementation_shortfall = market_impact + timing_risk
        
        return {
            'market_impact_bps': market_impact * 10000,
            'timing_risk_bps': timing_risk * 10000,
            'implementation_shortfall_bps': implementation_shortfall * 10000,
            'fill_rate': self.fill_rate
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'strategy': self.strategy,
            'total_quantity': str(self.total_quantity),
            'filled_quantity': str(self.filled_quantity),
            'average_fill_price': str(self.average_fill_price),
            'order_created_at': self.order_created_at.isoformat(),
            'first_fill_at': self.first_fill_at.isoformat() if self.first_fill_at else None,
            'last_fill_at': self.last_fill_at.isoformat() if self.last_fill_at else None,
            'total_execution_time': str(self.total_execution_time) if self.total_execution_time else None,
            'arrival_price': str(self.arrival_price),
            'benchmark_prices': {k: str(v) for k, v in self.benchmark_prices.items()},
            'slippage_bps': self.slippage_bps,
            'market_impact_bps': self.market_impact_bps,
            'timing_risk_bps': self.timing_risk_bps,
            'total_fees': str(self.total_fees),
            'total_cost': str(self.total_cost),
            'cost_per_share': str(self.cost_per_share),
            'fill_rate': self.fill_rate,
            'num_fills': self.num_fills,
            'avg_fill_size': str(self.avg_fill_size),
            'participation_rate': self.participation_rate,
            'aggressiveness_score': self.aggressiveness_score,
            'execution_quality': self.execution_quality
        }


@dataclass
class StrategyMetrics:
    """Aggregated metrics for an execution strategy."""
    
    strategy_name: str
    total_orders: int = 0
    total_volume: Decimal = Decimal('0')
    
    # Performance metrics
    avg_slippage_bps: float = 0.0
    avg_market_impact_bps: float = 0.0
    avg_fill_rate: float = 0.0
    avg_execution_time: Optional[timedelta] = None
    
    # Cost metrics
    total_fees: Decimal = Decimal('0')
    avg_cost_per_share: Decimal = Decimal('0')
    
    # Quality metrics
    fill_rate_90th_percentile: float = 0.0
    slippage_75th_percentile: float = 0.0
    execution_quality_score: float = 0.0
    
    # Risk metrics
    slippage_volatility: float = 0.0
    max_slippage_bps: float = 0.0
    tail_slippage_95th_percentile: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'strategy_name': self.strategy_name,
            'total_orders': self.total_orders,
            'total_volume': str(self.total_volume),
            'avg_slippage_bps': self.avg_slippage_bps,
            'avg_market_impact_bps': self.avg_market_impact_bps,
            'avg_fill_rate': self.avg_fill_rate,
            'avg_execution_time': str(self.avg_execution_time) if self.avg_execution_time else None,
            'total_fees': str(self.total_fees),
            'avg_cost_per_share': str(self.avg_cost_per_share),
            'fill_rate_90th_percentile': self.fill_rate_90th_percentile,
            'slippage_75th_percentile': self.slippage_75th_percentile,
            'execution_quality_score': self.execution_quality_score,
            'slippage_volatility': self.slippage_volatility,
            'max_slippage_bps': self.max_slippage_bps,
            'tail_slippage_95th_percentile': self.tail_slippage_95th_percentile
        }


@dataclass
class MarketImpactModel:
    """Market impact model parameters."""
    
    symbol: str
    
    # Linear impact model: impact = α + β * size + γ * volatility
    alpha: float = 0.0      # Fixed impact component
    beta: float = 0.0       # Size impact coefficient
    gamma: float = 0.0      # Volatility impact coefficient
    
    # Non-linear components
    sqrt_impact: float = 0.0     # Square root impact coefficient
    log_impact: float = 0.0      # Logarithmic impact coefficient
    
    # Model statistics
    r_squared: float = 0.0       # Model fit quality
    sample_size: int = 0         # Number of observations
    last_updated: datetime = field(default_factory=datetime.now)
    
    def predict_impact(
        self,
        trade_size: Decimal,
        volatility: float,
        participation_rate: float
    ) -> float:
        """Predict market impact for a trade."""
        size_factor = float(trade_size)
        
        # Linear component
        impact = self.alpha + self.beta * size_factor + self.gamma * volatility
        
        # Non-linear components
        if self.sqrt_impact != 0:
            impact += self.sqrt_impact * math.sqrt(size_factor)
        
        if self.log_impact != 0 and size_factor > 0:
            impact += self.log_impact * math.log(size_factor)
        
        # Participation rate adjustment
        if participation_rate > 0:
            impact *= (1 + participation_rate)
        
        return max(0, impact)  # Impact cannot be negative


class ExecutionAnalyzer:
    """
    Comprehensive execution analysis and performance tracking.
    
    This class analyzes order execution performance, calculates
    various metrics, and provides insights for optimization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("ExecutionAnalyzer")
        
        # Metrics storage
        self.execution_metrics: Dict[str, ExecutionMetrics] = {}
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        
        # Market data for benchmarks
        self.market_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.volume_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Market impact models
        self.impact_models: Dict[str, MarketImpactModel] = {}
        
        # Database for persistence
        self.db_path = self.config.get('database_path', 'execution_analytics.db')
        self._init_database()
        
    def _default_config(self) -> Dict:
        """Default configuration for execution analyzer."""
        return {
            'database_path': 'execution_analytics.db',
            'enable_persistence': True,
            'benchmark_window_minutes': 60,    # Window for benchmark calculations
            'impact_model_update_interval': 3600,  # Update models every hour
            'max_metrics_history': 50000,      # Maximum metrics to keep in memory
            'outlier_threshold': 3.0,          # Z-score for outlier detection
            'min_sample_size': 50,             # Minimum samples for statistical analysis
        }
    
    def _init_database(self) -> None:
        """Initialize SQLite database for analytics persistence."""
        if not self.config['enable_persistence']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create execution metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_metrics (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy TEXT,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create strategy metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_metrics (
                    strategy_name TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create market impact models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_impact_models (
                    symbol TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_symbol ON execution_metrics (symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_strategy ON execution_metrics (strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_created_at ON execution_metrics (created_at)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics database: {e}")
    
    def analyze_order_execution(
        self,
        order: Order,
        market_data: Optional[Dict[str, Any]] = None
    ) -> ExecutionMetrics:
        """Analyze execution performance for a completed order."""
        
        # Calculate basic metrics
        total_quantity = order.quantity
        filled_quantity = order.filled_quantity
        average_fill_price = order.average_fill_price or Decimal('0')
        
        # Get timing information
        first_fill_time = None
        last_fill_time = None
        total_fees = Decimal('0')
        
        if order.fills:
            fill_times = [fill.timestamp for fill in order.fills]
            first_fill_time = min(fill_times)
            last_fill_time = max(fill_times)
            total_fees = sum(fill.fee for fill in order.fills)
        
        # Get arrival price (price when order was created)
        arrival_price = self._get_arrival_price(order.symbol, order.created_at)
        
        # Calculate benchmark prices
        benchmark_prices = self._calculate_benchmark_prices(
            order.symbol,
            order.created_at,
            first_fill_time,
            last_fill_time
        )
        
        # Create execution metrics
        metrics = ExecutionMetrics(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            strategy=order.strategy_id,
            total_quantity=total_quantity,
            filled_quantity=filled_quantity,
            average_fill_price=average_fill_price,
            order_created_at=order.created_at,
            first_fill_at=first_fill_time,
            last_fill_at=last_fill_time,
            arrival_price=arrival_price,
            benchmark_prices=benchmark_prices,
            total_fees=total_fees,
            num_fills=len(order.fills)
        )
        
        # Calculate performance metrics
        metrics.slippage_bps = metrics.calculate_slippage(BenchmarkType.ARRIVAL_PRICE)
        
        # Calculate market impact
        if market_data:
            metrics.market_impact_bps = self._calculate_market_impact(order, market_data)
        
        # Calculate execution costs
        metrics.total_cost = total_fees
        if filled_quantity > 0:
            metrics.cost_per_share = metrics.total_cost / filled_quantity
        
        # Calculate participation rate
        if market_data and 'volume' in market_data:
            execution_volume = market_data['volume']
            if execution_volume > 0:
                metrics.participation_rate = float(filled_quantity / execution_volume)
        
        # Calculate aggressiveness score
        metrics.aggressiveness_score = self._calculate_aggressiveness_score(order)
        
        # Calculate execution quality score
        metrics.execution_quality = self._calculate_execution_quality_score(metrics)
        
        # Store metrics
        self.execution_metrics[order.order_id] = metrics
        
        # Update strategy metrics
        if order.strategy_id:
            self._update_strategy_metrics(order.strategy_id, metrics)
        
        # Save to database
        self._save_execution_metrics(metrics)
        
        self.logger.info(
            f"Analyzed execution for order {order.order_id}: "
            f"slippage={metrics.slippage_bps:.2f}bps, "
            f"fill_rate={metrics.fill_rate:.2%}"
        )
        
        return metrics
    
    def analyze_execution_plan(self, plan: ExecutionPlan, child_orders: List[Order]) -> Dict[str, Any]:
        """Analyze execution performance for a complex execution plan."""
        if not child_orders:
            return {}
        
        # Aggregate metrics from child orders
        total_quantity = plan.total_quantity
        total_filled = sum(order.filled_quantity for order in child_orders)
        total_fees = sum(
            sum(fill.fee for fill in order.fills)
            for order in child_orders
        )
        
        # Calculate volume-weighted average fill price
        total_value = Decimal('0')
        for order in child_orders:
            if order.fills:
                for fill in order.fills:
                    total_value += fill.price * fill.quantity
        
        avg_fill_price = total_value / total_filled if total_filled > 0 else Decimal('0')
        
        # Get timing information
        all_fills = []
        for order in child_orders:
            all_fills.extend(order.fills)
        
        if all_fills:
            fill_times = [fill.timestamp for fill in all_fills]
            first_fill_time = min(fill_times)
            last_fill_time = max(fill_times)
            execution_time = last_fill_time - first_fill_time
        else:
            first_fill_time = None
            last_fill_time = None
            execution_time = None
        
        # Calculate arrival price
        arrival_price = self._get_arrival_price(plan.symbol, plan.started_at)
        
        # Calculate slippage
        slippage_bps = 0.0
        if arrival_price > 0 and avg_fill_price > 0:
            if plan.side == OrderSide.BUY:
                slippage = (avg_fill_price - arrival_price) / arrival_price
            else:
                slippage = (arrival_price - avg_fill_price) / arrival_price
            slippage_bps = float(slippage * 10000)
        
        # Analyze execution quality
        planned_time = plan.time_horizon
        actual_time = execution_time
        
        time_efficiency = 1.0
        if planned_time and actual_time:
            time_efficiency = min(1.0, planned_time.total_seconds() / actual_time.total_seconds())
        
        # Calculate participation rate analysis
        participation_analysis = self._analyze_participation_rate(plan, child_orders)
        
        return {
            'plan_id': plan.plan_id,
            'strategy': plan.strategy.value,
            'symbol': plan.symbol,
            'side': plan.side.value,
            'total_quantity': str(total_quantity),
            'filled_quantity': str(total_filled),
            'fill_rate': float(total_filled / total_quantity) if total_quantity > 0 else 0.0,
            'average_fill_price': str(avg_fill_price),
            'arrival_price': str(arrival_price),
            'slippage_bps': slippage_bps,
            'total_fees': str(total_fees),
            'num_child_orders': len(child_orders),
            'num_fills': len(all_fills),
            'execution_time': str(execution_time) if execution_time else None,
            'planned_time': str(planned_time) if planned_time else None,
            'time_efficiency': time_efficiency,
            'first_fill_at': first_fill_time.isoformat() if first_fill_time else None,
            'last_fill_at': last_fill_time.isoformat() if last_fill_time else None,
            'participation_analysis': participation_analysis
        }
    
    def _get_arrival_price(self, symbol: str, timestamp: datetime) -> Decimal:
        """Get arrival price (market price when order was created)."""
        # This is a simplified implementation
        # In practice, you'd look up the exact market price at the timestamp
        
        market_history = self.market_data.get(symbol, deque())
        if not market_history:
            return Decimal('0')
        
        # Find closest price to timestamp
        closest_price = Decimal('0')
        min_time_diff = float('inf')
        
        for data_point in market_history:
            if 'timestamp' in data_point and 'price' in data_point:
                point_time = data_point['timestamp']
                if isinstance(point_time, str):
                    point_time = datetime.fromisoformat(point_time)
                
                time_diff = abs((timestamp - point_time).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_price = Decimal(str(data_point['price']))
        
        return closest_price
    
    def _calculate_benchmark_prices(
        self,
        symbol: str,
        order_time: datetime,
        first_fill: Optional[datetime],
        last_fill: Optional[datetime]
    ) -> Dict[str, Decimal]:
        """Calculate benchmark prices (TWAP, VWAP, etc.)."""
        benchmarks = {}
        
        if not first_fill or not last_fill:
            return benchmarks
        
        # Get market data for the execution period
        execution_start = first_fill
        execution_end = last_fill
        
        market_history = self.market_data.get(symbol, deque())
        volume_history = self.volume_data.get(symbol, deque())
        
        execution_prices = []
        execution_volumes = []
        
        for data_point in market_history:
            if 'timestamp' in data_point and 'price' in data_point:
                point_time = data_point['timestamp']
                if isinstance(point_time, str):
                    point_time = datetime.fromisoformat(point_time)
                
                if execution_start <= point_time <= execution_end:
                    execution_prices.append(Decimal(str(data_point['price'])))
        
        for data_point in volume_history:
            if 'timestamp' in data_point and 'volume' in data_point:
                point_time = data_point['timestamp']
                if isinstance(point_time, str):
                    point_time = datetime.fromisoformat(point_time)
                
                if execution_start <= point_time <= execution_end:
                    execution_volumes.append(Decimal(str(data_point['volume'])))
        
        # Calculate TWAP
        if execution_prices:
            benchmarks['twap'] = sum(execution_prices) / len(execution_prices)
        
        # Calculate VWAP
        if execution_prices and execution_volumes and len(execution_prices) == len(execution_volumes):
            total_value = sum(p * v for p, v in zip(execution_prices, execution_volumes))
            total_volume = sum(execution_volumes)
            if total_volume > 0:
                benchmarks['vwap'] = total_value / total_volume
        
        return benchmarks
    
    def _calculate_market_impact(self, order: Order, market_data: Dict[str, Any]) -> float:
        """Calculate market impact of an order."""
        # Simplified market impact calculation
        # In practice, this would be more sophisticated
        
        if not order.fills:
            return 0.0
        
        # Get pre and post execution prices
        pre_price = market_data.get('pre_execution_price', 0)
        post_price = market_data.get('post_execution_price', 0)
        
        if pre_price <= 0 or post_price <= 0:
            return 0.0
        
        # Calculate impact
        if order.side == OrderSide.BUY:
            impact = (post_price - pre_price) / pre_price
        else:
            impact = (pre_price - post_price) / pre_price
        
        return float(impact * 10000)  # Convert to basis points
    
    def _calculate_aggressiveness_score(self, order: Order) -> float:
        """Calculate aggressiveness score based on order characteristics."""
        score = 0.5  # Default neutral score
        
        # Market orders are more aggressive
        if order.order_type == OrderType.MARKET:
            score += 0.3
        
        # IOC orders are more aggressive
        if hasattr(order, 'time_in_force') and order.time_in_force.value == 'immediate_or_cancel':
            score += 0.2
        
        # Large orders relative to average daily volume are more aggressive
        # This would require additional market data
        
        return min(1.0, max(0.0, score))
    
    def _calculate_execution_quality_score(self, metrics: ExecutionMetrics) -> float:
        """Calculate overall execution quality score."""
        score = 0.0
        components = 0
        
        # Fill rate component (30% weight)
        if metrics.fill_rate > 0:
            score += metrics.fill_rate * 0.3
            components += 0.3
        
        # Slippage component (40% weight) - lower is better
        if metrics.slippage_bps != 0:
            # Normalize slippage to 0-1 scale (assuming 50bps is poor)
            slippage_score = max(0, 1 - abs(metrics.slippage_bps) / 50)
            score += slippage_score * 0.4
            components += 0.4
        
        # Speed component (20% weight)
        if metrics.total_execution_time:
            # Faster execution is better (up to a point)
            execution_seconds = metrics.total_execution_time.total_seconds()
            # Assume 5 minutes is ideal, penalty for too fast or too slow
            ideal_time = 300  # 5 minutes
            time_score = max(0, 1 - abs(execution_seconds - ideal_time) / ideal_time)
            score += time_score * 0.2
            components += 0.2
        
        # Cost component (10% weight) - lower fees are better
        if metrics.total_cost > 0 and metrics.filled_quantity > 0:
            # This is very simplified - would need benchmark fee rates
            cost_per_unit = float(metrics.total_cost / metrics.filled_quantity)
            # Assume 0.1% is reasonable fee rate
            reasonable_fee = float(metrics.average_fill_price) * 0.001
            cost_score = max(0, 1 - cost_per_unit / reasonable_fee) if reasonable_fee > 0 else 0
            score += cost_score * 0.1
            components += 0.1
        
        return score / components if components > 0 else 0.0
    
    def _analyze_participation_rate(self, plan: ExecutionPlan, child_orders: List[Order]) -> Dict[str, Any]:
        """Analyze participation rate for execution plan."""
        # This would require market volume data during execution
        # Simplified implementation
        
        total_filled = sum(order.filled_quantity for order in child_orders)
        target_participation = plan.max_participation_rate
        
        return {
            'target_participation_rate': target_participation,
            'estimated_actual_rate': 0.0,  # Would calculate from market data
            'total_filled': str(total_filled),
            'analysis': 'Requires market volume data for accurate analysis'
        }
    
    def _update_strategy_metrics(self, strategy_id: str, metrics: ExecutionMetrics) -> None:
        """Update aggregated strategy metrics."""
        if strategy_id not in self.strategy_metrics:
            self.strategy_metrics[strategy_id] = StrategyMetrics(strategy_name=strategy_id)
        
        strategy_metrics = self.strategy_metrics[strategy_id]
        
        # Update counts
        strategy_metrics.total_orders += 1
        strategy_metrics.total_volume += metrics.filled_quantity
        strategy_metrics.total_fees += metrics.total_fees
        
        # Update averages (simple incremental average)
        n = strategy_metrics.total_orders
        
        strategy_metrics.avg_slippage_bps = (
            (strategy_metrics.avg_slippage_bps * (n - 1) + metrics.slippage_bps) / n
        )
        
        strategy_metrics.avg_market_impact_bps = (
            (strategy_metrics.avg_market_impact_bps * (n - 1) + metrics.market_impact_bps) / n
        )
        
        strategy_metrics.avg_fill_rate = (
            (strategy_metrics.avg_fill_rate * (n - 1) + metrics.fill_rate) / n
        )
        
        if strategy_metrics.total_volume > 0:
            strategy_metrics.avg_cost_per_share = strategy_metrics.total_fees / strategy_metrics.total_volume
        
        # Update percentiles and other statistics periodically
        if n % 10 == 0:  # Every 10 orders
            self._recalculate_strategy_statistics(strategy_id)
    
    def _recalculate_strategy_statistics(self, strategy_id: str) -> None:
        """Recalculate strategy statistics from all historical data."""
        # Get all metrics for this strategy
        strategy_order_metrics = [
            metrics for metrics in self.execution_metrics.values()
            if metrics.strategy == strategy_id
        ]
        
        if not strategy_order_metrics:
            return
        
        strategy_metrics = self.strategy_metrics[strategy_id]
        
        # Calculate percentiles
        slippage_values = [m.slippage_bps for m in strategy_order_metrics]
        fill_rates = [m.fill_rate for m in strategy_order_metrics]
        
        if slippage_values:
            strategy_metrics.slippage_75th_percentile = np.percentile(slippage_values, 75)
            strategy_metrics.max_slippage_bps = max(slippage_values)
            strategy_metrics.tail_slippage_95th_percentile = np.percentile(slippage_values, 95)
            strategy_metrics.slippage_volatility = float(np.std(slippage_values))
        
        if fill_rates:
            strategy_metrics.fill_rate_90th_percentile = np.percentile(fill_rates, 90)
        
        # Calculate execution quality score
        quality_scores = [m.execution_quality for m in strategy_order_metrics if m.execution_quality > 0]
        if quality_scores:
            strategy_metrics.execution_quality_score = statistics.mean(quality_scores)
    
    def update_market_data(self, symbol: str, price: Decimal, volume: Decimal, timestamp: Optional[datetime] = None) -> None:
        """Update market data for benchmark calculations."""
        timestamp = timestamp or datetime.now()
        
        price_data = {
            'timestamp': timestamp,
            'price': price
        }
        
        volume_data = {
            'timestamp': timestamp,
            'volume': volume
        }
        
        self.market_data[symbol].append(price_data)
        self.volume_data[symbol].append(volume_data)
    
    def get_execution_metrics(self, order_id: str) -> Optional[ExecutionMetrics]:
        """Get execution metrics for a specific order."""
        return self.execution_metrics.get(order_id)
    
    def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Get aggregated metrics for a strategy."""
        return self.strategy_metrics.get(strategy_id)
    
    def get_symbol_performance(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get execution performance for a specific symbol."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        symbol_metrics = [
            metrics for metrics in self.execution_metrics.values()
            if metrics.symbol == symbol and metrics.order_created_at > cutoff_date
        ]
        
        if not symbol_metrics:
            return {}
        
        # Calculate aggregated statistics
        slippage_values = [m.slippage_bps for m in symbol_metrics]
        fill_rates = [m.fill_rate for m in symbol_metrics]
        execution_times = [
            m.total_execution_time.total_seconds()
            for m in symbol_metrics
            if m.total_execution_time
        ]
        
        total_volume = sum(m.filled_quantity for m in symbol_metrics)
        total_fees = sum(m.total_fees for m in symbol_metrics)
        
        return {
            'symbol': symbol,
            'period_days': days,
            'total_orders': len(symbol_metrics),
            'total_volume': str(total_volume),
            'total_fees': str(total_fees),
            'avg_slippage_bps': statistics.mean(slippage_values) if slippage_values else 0,
            'median_slippage_bps': statistics.median(slippage_values) if slippage_values else 0,
            'slippage_std_bps': statistics.stdev(slippage_values) if len(slippage_values) > 1 else 0,
            'avg_fill_rate': statistics.mean(fill_rates) if fill_rates else 0,
            'median_fill_rate': statistics.median(fill_rates) if fill_rates else 0,
            'avg_execution_time_seconds': statistics.mean(execution_times) if execution_times else 0,
            'median_execution_time_seconds': statistics.median(execution_times) if execution_times else 0,
            'slippage_percentiles': {
                '25th': np.percentile(slippage_values, 25) if slippage_values else 0,
                '75th': np.percentile(slippage_values, 75) if slippage_values else 0,
                '95th': np.percentile(slippage_values, 95) if slippage_values else 0
            }
        }
    
    def generate_execution_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive execution report for a time period."""
        # Filter metrics by date range
        period_metrics = [
            metrics for metrics in self.execution_metrics.values()
            if start_date <= metrics.order_created_at <= end_date
        ]
        
        if not period_metrics:
            return {'error': 'No execution data found for the specified period'}
        
        # Overall statistics
        total_orders = len(period_metrics)
        total_volume = sum(m.filled_quantity for m in period_metrics)
        total_fees = sum(m.total_fees for m in period_metrics)
        
        # Slippage statistics
        slippage_values = [m.slippage_bps for m in period_metrics]
        fill_rates = [m.fill_rate for m in period_metrics]
        
        # Performance by strategy
        strategy_performance = {}
        for strategy_id in set(m.strategy for m in period_metrics if m.strategy):
            strategy_metrics = [m for m in period_metrics if m.strategy == strategy_id]
            strategy_slippage = [m.slippage_bps for m in strategy_metrics]
            strategy_fill_rates = [m.fill_rate for m in strategy_metrics]
            
            strategy_performance[strategy_id] = {
                'orders': len(strategy_metrics),
                'volume': str(sum(m.filled_quantity for m in strategy_metrics)),
                'avg_slippage_bps': statistics.mean(strategy_slippage) if strategy_slippage else 0,
                'avg_fill_rate': statistics.mean(strategy_fill_rates) if strategy_fill_rates else 0
            }
        
        # Performance by symbol
        symbol_performance = {}
        for symbol in set(m.symbol for m in period_metrics):
            symbol_metrics = [m for m in period_metrics if m.symbol == symbol]
            symbol_slippage = [m.slippage_bps for m in symbol_metrics]
            symbol_fill_rates = [m.fill_rate for m in symbol_metrics]
            
            symbol_performance[symbol] = {
                'orders': len(symbol_metrics),
                'volume': str(sum(m.filled_quantity for m in symbol_metrics)),
                'avg_slippage_bps': statistics.mean(symbol_slippage) if symbol_slippage else 0,
                'avg_fill_rate': statistics.mean(symbol_fill_rates) if symbol_fill_rates else 0
            }
        
        return {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'overall_statistics': {
                'total_orders': total_orders,
                'total_volume': str(total_volume),
                'total_fees': str(total_fees),
                'avg_slippage_bps': statistics.mean(slippage_values) if slippage_values else 0,
                'median_slippage_bps': statistics.median(slippage_values) if slippage_values else 0,
                'slippage_std_bps': statistics.stdev(slippage_values) if len(slippage_values) > 1 else 0,
                'avg_fill_rate': statistics.mean(fill_rates) if fill_rates else 0,
                'median_fill_rate': statistics.median(fill_rates) if fill_rates else 0
            },
            'slippage_distribution': {
                'percentiles': {
                    '10th': np.percentile(slippage_values, 10) if slippage_values else 0,
                    '25th': np.percentile(slippage_values, 25) if slippage_values else 0,
                    '50th': np.percentile(slippage_values, 50) if slippage_values else 0,
                    '75th': np.percentile(slippage_values, 75) if slippage_values else 0,
                    '90th': np.percentile(slippage_values, 90) if slippage_values else 0,
                    '95th': np.percentile(slippage_values, 95) if slippage_values else 0
                }
            },
            'performance_by_strategy': strategy_performance,
            'performance_by_symbol': symbol_performance
        }
    
    def _save_execution_metrics(self, metrics: ExecutionMetrics) -> None:
        """Save execution metrics to database."""
        if not self.config['enable_persistence']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            data = json.dumps(metrics.to_dict())
            
            cursor.execute("""
                INSERT OR REPLACE INTO execution_metrics (order_id, symbol, strategy, data)
                VALUES (?, ?, ?, ?)
            """, (metrics.order_id, metrics.symbol, metrics.strategy, data))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save execution metrics: {e}")
    
    def export_metrics(self, file_path: str, format: str = 'csv', start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> bool:
        """Export execution metrics to file."""
        try:
            # Filter metrics by date if specified
            metrics_to_export = list(self.execution_metrics.values())
            
            if start_date:
                metrics_to_export = [m for m in metrics_to_export if m.order_created_at >= start_date]
            
            if end_date:
                metrics_to_export = [m for m in metrics_to_export if m.order_created_at <= end_date]
            
            if format.lower() == 'csv':
                # Convert to DataFrame
                data = [metrics.to_dict() for metrics in metrics_to_export]
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
            elif format.lower() == 'json':
                data = [metrics.to_dict() for metrics in metrics_to_export]
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported {len(metrics_to_export)} execution metrics to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export execution metrics: {e}")
            return False