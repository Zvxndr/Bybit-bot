"""
Advanced Order Flow Analysis Engine for High-Frequency Trading.
Analyzes order book dynamics, trade flow patterns, and market microstructure signals.
"""

import asyncio
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Any, Union
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
    from scipy.signal import savgol_filter
    import talib
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class FlowDirection(Enum):
    """Order flow direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    AGGRESSIVE_BUY = "aggressive_buy"
    AGGRESSIVE_SELL = "aggressive_sell"

class OrderType(Enum):
    """Order types for flow analysis."""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    ICEBERG = "iceberg"
    HIDDEN = "hidden"

class VolumeProfile(Enum):
    """Volume profile types."""
    POINT_OF_CONTROL = "poc"          # Highest volume price level
    VALUE_AREA_HIGH = "vah"           # Top of value area (70% volume)
    VALUE_AREA_LOW = "val"            # Bottom of value area (70% volume)
    VOLUME_WEIGHTED_AVERAGE = "vwap"  # Volume weighted average price

class MarketRegime(Enum):
    """Market regime identification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

@dataclass
class Trade:
    """Individual trade data."""
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    trade_id: str
    exchange: str = ""
    is_aggressive: bool = False
    value: float = field(init=False)
    
    def __post_init__(self):
        self.value = self.price * self.quantity

@dataclass
class OrderBookLevel:
    """Order book level data."""
    price: float
    quantity: float
    orders: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    mid_price: float = field(init=False)
    spread: float = field(init=False)
    spread_bps: float = field(init=False)
    
    def __post_init__(self):
        if self.bids and self.asks:
            best_bid = self.bids[0].price
            best_ask = self.asks[0].price
            self.mid_price = (best_bid + best_ask) / 2
            self.spread = best_ask - best_bid
            self.spread_bps = (self.spread / self.mid_price) * 10000 if self.mid_price > 0 else 0

@dataclass
class FlowMetrics:
    """Order flow metrics."""
    symbol: str
    timestamp: datetime
    
    # Volume metrics
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    net_volume: float = 0.0
    total_volume: float = 0.0
    volume_imbalance: float = 0.0  # (buy - sell) / total
    
    # Trade metrics
    trade_count: int = 0
    avg_trade_size: float = 0.0
    large_trade_ratio: float = 0.0
    
    # Aggressive flow
    aggressive_buy_volume: float = 0.0
    aggressive_sell_volume: float = 0.0
    aggressive_ratio: float = 0.0
    
    # Price impact
    price_impact_bps: float = 0.0
    realized_spread_bps: float = 0.0
    
    # Flow direction and strength
    flow_direction: FlowDirection = FlowDirection.NEUTRAL
    flow_strength: float = 0.0  # 0-1 scale
    
    # Advanced metrics
    order_arrival_rate: float = 0.0
    cancellation_rate: float = 0.0
    hidden_liquidity_ratio: float = 0.0

@dataclass
class VolumeProfileData:
    """Volume profile analysis data."""
    symbol: str
    timestamp: datetime
    price_levels: List[float]
    volume_at_price: List[float]
    point_of_control: float
    value_area_high: float
    value_area_low: float
    vwap: float
    total_volume: float

@dataclass
class MarketMicrostructure:
    """Market microstructure analysis."""
    symbol: str
    timestamp: datetime
    
    # Liquidity metrics
    bid_liquidity: float = 0.0
    ask_liquidity: float = 0.0
    liquidity_imbalance: float = 0.0
    effective_spread_bps: float = 0.0
    
    # Order book metrics
    order_book_pressure: float = 0.0  # Net pressure from order book
    depth_imbalance: float = 0.0      # Weighted depth imbalance
    price_improvement: float = 0.0     # Price improvement opportunities
    
    # Market regime
    regime: MarketRegime = MarketRegime.RANGE_BOUND
    volatility_regime: str = "normal"
    trend_strength: float = 0.0
    
    # Timing metrics
    time_between_trades_ms: float = 0.0
    quote_update_frequency: float = 0.0
    market_impact_decay_ms: float = 0.0

class OrderFlowAnalyzer:
    """Advanced order flow analysis engine."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Configuration
        self.flow_config = {
            'analysis_window_seconds': 60,        # 1 minute analysis window
            'tick_aggregation_ms': 100,           # 100ms tick aggregation
            'large_trade_threshold_percentile': 90, # 90th percentile for large trades
            'aggressive_trade_threshold_bps': 2,  # 2 bps from mid for aggressive
            'min_volume_for_analysis': 1000,      # Minimum volume for reliable analysis
            'order_book_depth_levels': 20,        # Number of order book levels to analyze
            'volume_profile_bins': 50,            # Number of price bins for volume profile
            'flow_strength_decay': 0.95,          # Flow strength exponential decay
            'liquidity_threshold_usd': 10000,     # $10k liquidity threshold
            'regime_detection_lookback': 300,     # 5 minute lookback for regime detection
            'microstructure_update_ms': 50        # 50ms microstructure updates
        }
        
        # Data storage
        self.trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.order_books: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.flow_metrics: Dict[str, FlowMetrics] = {}
        self.volume_profiles: Dict[str, VolumeProfileData] = {}
        self.microstructure: Dict[str, MarketMicrostructure] = {}
        
        # Analysis components
        self.tick_aggregator = TickAggregator(self)
        self.flow_detector = FlowDetector(self)
        self.regime_analyzer = RegimeAnalyzer(self)
        self.liquidity_analyzer = LiquidityAnalyzer(self)
        
        # Real-time analysis state
        self.analyzing_symbols: Set[str] = set()
        self.flow_lock = threading.Lock()
        self.running = False
        self.analysis_task = None
        
        # Performance tracking
        self.analysis_latency_ms = 0.0
        self.processed_trades = 0
        self.processed_quotes = 0
        
        self.logger.info("OrderFlowAnalyzer initialized")
    
    async def start_analysis(self, symbols: List[str]):
        """Start order flow analysis for symbols."""
        try:
            if self.running:
                return
            
            self.running = True
            self.analyzing_symbols = set(symbols)
            
            # Initialize data structures
            for symbol in symbols:
                self.flow_metrics[symbol] = FlowMetrics(
                    symbol=symbol,
                    timestamp=datetime.now()
                )
                self.microstructure[symbol] = MarketMicrostructure(
                    symbol=symbol,
                    timestamp=datetime.now()
                )
            
            # Start analysis task
            self.analysis_task = asyncio.create_task(
                self._analysis_loop()
            )
            
            self.logger.info(f"Order flow analysis started for {len(symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Failed to start order flow analysis: {e}")
            self.running = False
            raise
    
    async def stop_analysis(self):
        """Stop order flow analysis."""
        try:
            self.running = False
            
            if self.analysis_task:
                self.analysis_task.cancel()
                try:
                    await self.analysis_task
                except asyncio.CancelledError:
                    pass
            
            self.analyzing_symbols.clear()
            self.logger.info("Order flow analysis stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop order flow analysis: {e}")
    
    async def add_trade(self, trade: Trade):
        """Add trade data for analysis."""
        try:
            if trade.symbol not in self.analyzing_symbols:
                return
            
            with self.flow_lock:
                self.trades[trade.symbol].append(trade)
                self.processed_trades += 1
                
        except Exception as e:
            self.logger.error(f"Failed to add trade: {e}")
    
    async def add_order_book(self, order_book: OrderBookSnapshot):
        """Add order book snapshot for analysis."""
        try:
            if order_book.symbol not in self.analyzing_symbols:
                return
            
            with self.flow_lock:
                self.order_books[order_book.symbol].append(order_book)
                self.processed_quotes += 1
                
        except Exception as e:
            self.logger.error(f"Failed to add order book: {e}")
    
    async def _analysis_loop(self):
        """Main analysis loop."""
        try:
            while self.running:
                analysis_start = time.perf_counter()
                
                # Analyze each symbol
                for symbol in self.analyzing_symbols:
                    await self._analyze_symbol(symbol)
                
                # Calculate analysis latency
                self.analysis_latency_ms = (time.perf_counter() - analysis_start) * 1000
                
                # Sleep until next analysis cycle
                await asyncio.sleep(self.flow_config['microstructure_update_ms'] / 1000)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Order flow analysis loop error: {e}")
    
    async def _analyze_symbol(self, symbol: str):
        """Analyze order flow for a specific symbol."""
        try:
            if symbol not in self.trades or symbol not in self.order_books:
                return
            
            # Get recent data
            now = datetime.now()
            analysis_window = timedelta(seconds=self.flow_config['analysis_window_seconds'])
            cutoff_time = now - analysis_window
            
            # Filter recent trades and order books
            recent_trades = [
                trade for trade in self.trades[symbol]
                if trade.timestamp >= cutoff_time
            ]
            
            recent_order_books = [
                ob for ob in self.order_books[symbol]
                if ob.timestamp >= cutoff_time
            ]
            
            if len(recent_trades) < 10 or len(recent_order_books) < 10:
                return
            
            # Update flow metrics
            await self._update_flow_metrics(symbol, recent_trades, recent_order_books)
            
            # Update volume profile
            await self._update_volume_profile(symbol, recent_trades)
            
            # Update microstructure analysis
            await self._update_microstructure(symbol, recent_trades, recent_order_books)
            
        except Exception as e:
            self.logger.error(f"Symbol analysis failed for {symbol}: {e}")
    
    async def _update_flow_metrics(self, symbol: str, trades: List[Trade], order_books: List[OrderBookSnapshot]):
        """Update flow metrics for symbol."""
        try:
            if not trades:
                return
            
            metrics = self.flow_metrics[symbol]
            
            # Reset metrics
            metrics.timestamp = datetime.now()
            metrics.buy_volume = 0.0
            metrics.sell_volume = 0.0
            metrics.aggressive_buy_volume = 0.0
            metrics.aggressive_sell_volume = 0.0
            metrics.trade_count = len(trades)
            
            # Calculate trade sizes for percentile analysis
            trade_sizes = [trade.quantity for trade in trades]
            large_trade_threshold = np.percentile(trade_sizes, self.flow_config['large_trade_threshold_percentile'])
            large_trades = 0
            
            # Process each trade
            for trade in trades:
                if trade.side == 'buy':
                    metrics.buy_volume += trade.quantity
                    if trade.is_aggressive:
                        metrics.aggressive_buy_volume += trade.quantity
                else:
                    metrics.sell_volume += trade.quantity
                    if trade.is_aggressive:
                        metrics.aggressive_sell_volume += trade.quantity
                
                if trade.quantity >= large_trade_threshold:
                    large_trades += 1
            
            # Calculate derived metrics
            metrics.total_volume = metrics.buy_volume + metrics.sell_volume
            metrics.net_volume = metrics.buy_volume - metrics.sell_volume
            
            if metrics.total_volume > 0:
                metrics.volume_imbalance = metrics.net_volume / metrics.total_volume
                metrics.avg_trade_size = metrics.total_volume / metrics.trade_count
                metrics.large_trade_ratio = large_trades / metrics.trade_count
            
            # Aggressive flow metrics
            total_aggressive = metrics.aggressive_buy_volume + metrics.aggressive_sell_volume
            if metrics.total_volume > 0:
                metrics.aggressive_ratio = total_aggressive / metrics.total_volume
            
            # Determine flow direction and strength
            metrics.flow_direction = self._determine_flow_direction(metrics)
            metrics.flow_strength = self._calculate_flow_strength(metrics)
            
            # Calculate price impact
            if len(order_books) >= 2:
                metrics.price_impact_bps = await self._calculate_price_impact(trades, order_books)
            
        except Exception as e:
            self.logger.error(f"Flow metrics update failed for {symbol}: {e}")
    
    def _determine_flow_direction(self, metrics: FlowMetrics) -> FlowDirection:
        """Determine overall flow direction."""
        try:
            # Strong directional flow thresholds
            strong_threshold = 0.3
            aggressive_threshold = 0.6
            
            if abs(metrics.volume_imbalance) < 0.1 and metrics.aggressive_ratio < 0.3:
                return FlowDirection.NEUTRAL
            
            if metrics.volume_imbalance > strong_threshold:
                if metrics.aggressive_ratio > aggressive_threshold:
                    return FlowDirection.AGGRESSIVE_BUY
                else:
                    return FlowDirection.BULLISH
            elif metrics.volume_imbalance < -strong_threshold:
                if metrics.aggressive_ratio > aggressive_threshold:
                    return FlowDirection.AGGRESSIVE_SELL
                else:
                    return FlowDirection.BEARISH
            else:
                return FlowDirection.NEUTRAL
                
        except Exception as e:
            self.logger.error(f"Flow direction determination failed: {e}")
            return FlowDirection.NEUTRAL
    
    def _calculate_flow_strength(self, metrics: FlowMetrics) -> float:
        """Calculate flow strength (0-1 scale)."""
        try:
            # Combine multiple factors for strength calculation
            imbalance_strength = abs(metrics.volume_imbalance)
            aggressive_strength = metrics.aggressive_ratio
            volume_strength = min(1.0, metrics.total_volume / self.flow_config['min_volume_for_analysis'])
            large_trade_strength = metrics.large_trade_ratio
            
            # Weighted combination
            strength = (
                imbalance_strength * 0.4 +
                aggressive_strength * 0.3 +
                volume_strength * 0.2 +
                large_trade_strength * 0.1
            )
            
            return min(1.0, strength)
            
        except Exception as e:
            self.logger.error(f"Flow strength calculation failed: {e}")
            return 0.0
    
    async def _calculate_price_impact(self, trades: List[Trade], order_books: List[OrderBookSnapshot]) -> float:
        """Calculate average price impact in basis points."""
        try:
            if not trades or not order_books:
                return 0.0
            
            price_impacts = []
            
            for trade in trades:
                # Find closest order book before trade
                closest_ob = None
                min_time_diff = float('inf')
                
                for ob in order_books:
                    time_diff = abs((trade.timestamp - ob.timestamp).total_seconds())
                    if time_diff < min_time_diff and ob.timestamp <= trade.timestamp:
                        min_time_diff = time_diff
                        closest_ob = ob
                
                if closest_ob and min_time_diff < 1.0:  # Within 1 second
                    # Calculate impact as deviation from mid price
                    impact_bps = abs(trade.price - closest_ob.mid_price) / closest_ob.mid_price * 10000
                    price_impacts.append(impact_bps)
            
            return np.mean(price_impacts) if price_impacts else 0.0
            
        except Exception as e:
            self.logger.error(f"Price impact calculation failed: {e}")
            return 0.0
    
    async def _update_volume_profile(self, symbol: str, trades: List[Trade]):
        """Update volume profile for symbol."""
        try:
            if not trades:
                return
            
            # Get price range
            prices = [trade.price for trade in trades]
            volumes = [trade.quantity for trade in trades]
            
            min_price = min(prices)
            max_price = max(prices)
            
            if max_price <= min_price:
                return
            
            # Create price bins
            num_bins = self.flow_config['volume_profile_bins']
            price_bins = np.linspace(min_price, max_price, num_bins)
            volume_at_price = np.zeros(num_bins)
            
            # Aggregate volume by price
            for trade in trades:
                bin_index = min(num_bins - 1, int((trade.price - min_price) / (max_price - min_price) * (num_bins - 1)))
                volume_at_price[bin_index] += trade.quantity
            
            # Calculate key levels
            total_volume = sum(volumes)
            poc_index = np.argmax(volume_at_price)
            point_of_control = price_bins[poc_index]
            
            # Calculate VWAP
            total_value = sum(trade.price * trade.quantity for trade in trades)
            vwap = total_value / total_volume if total_volume > 0 else 0
            
            # Calculate value area (70% of volume)
            cumulative_volume = np.cumsum(volume_at_price)
            value_area_volume = total_volume * 0.7
            
            # Find value area boundaries
            val_index = np.argmax(cumulative_volume >= total_volume * 0.15)  # 15th percentile
            vah_index = np.argmax(cumulative_volume >= total_volume * 0.85)  # 85th percentile
            
            value_area_low = price_bins[val_index]
            value_area_high = price_bins[vah_index]
            
            # Store volume profile
            self.volume_profiles[symbol] = VolumeProfileData(
                symbol=symbol,
                timestamp=datetime.now(),
                price_levels=price_bins.tolist(),
                volume_at_price=volume_at_price.tolist(),
                point_of_control=point_of_control,
                value_area_high=value_area_high,
                value_area_low=value_area_low,
                vwap=vwap,
                total_volume=total_volume
            )
            
        except Exception as e:
            self.logger.error(f"Volume profile update failed for {symbol}: {e}")
    
    async def _update_microstructure(self, symbol: str, trades: List[Trade], order_books: List[OrderBookSnapshot]):
        """Update market microstructure analysis."""
        try:
            if not order_books:
                return
            
            microstructure = self.microstructure[symbol]
            microstructure.timestamp = datetime.now()
            
            # Get latest order book
            latest_ob = order_books[-1]
            
            # Calculate liquidity metrics
            bid_liquidity = sum(level.quantity * level.price for level in latest_ob.bids[:5])
            ask_liquidity = sum(level.quantity * level.price for level in latest_ob.asks[:5])
            
            microstructure.bid_liquidity = bid_liquidity
            microstructure.ask_liquidity = ask_liquidity
            
            total_liquidity = bid_liquidity + ask_liquidity
            if total_liquidity > 0:
                microstructure.liquidity_imbalance = (bid_liquidity - ask_liquidity) / total_liquidity
            
            # Calculate effective spread
            microstructure.effective_spread_bps = latest_ob.spread_bps
            
            # Calculate order book pressure
            microstructure.order_book_pressure = await self._calculate_order_book_pressure(order_books)
            
            # Calculate depth imbalance
            microstructure.depth_imbalance = await self._calculate_depth_imbalance(latest_ob)
            
            # Detect market regime
            microstructure.regime = await self._detect_market_regime(symbol, trades, order_books)
            
            # Calculate timing metrics
            if len(trades) >= 2:
                trade_intervals = []
                for i in range(1, len(trades)):
                    interval_ms = (trades[i].timestamp - trades[i-1].timestamp).total_seconds() * 1000
                    trade_intervals.append(interval_ms)
                
                microstructure.time_between_trades_ms = np.mean(trade_intervals)
            
            # Quote update frequency
            if len(order_books) >= 2:
                quote_intervals = []
                for i in range(1, len(order_books)):
                    interval_ms = (order_books[i].timestamp - order_books[i-1].timestamp).total_seconds() * 1000
                    quote_intervals.append(interval_ms)
                
                if quote_intervals:
                    microstructure.quote_update_frequency = 1000 / np.mean(quote_intervals)  # Updates per second
            
        except Exception as e:
            self.logger.error(f"Microstructure update failed for {symbol}: {e}")
    
    async def _calculate_order_book_pressure(self, order_books: List[OrderBookSnapshot]) -> float:
        """Calculate order book pressure."""
        try:
            if len(order_books) < 2:
                return 0.0
            
            pressures = []
            
            for i in range(1, len(order_books)):
                current_ob = order_books[i]
                previous_ob = order_books[i-1]
                
                # Calculate bid/ask quantity changes
                current_bid_qty = sum(level.quantity for level in current_ob.bids[:5])
                current_ask_qty = sum(level.quantity for level in current_ob.asks[:5])
                
                previous_bid_qty = sum(level.quantity for level in previous_ob.bids[:5])
                previous_ask_qty = sum(level.quantity for level in previous_ob.asks[:5])
                
                bid_change = current_bid_qty - previous_bid_qty
                ask_change = current_ask_qty - previous_ask_qty
                
                # Net pressure (positive = buying pressure)
                net_pressure = bid_change - ask_change
                pressures.append(net_pressure)
            
            return np.mean(pressures) if pressures else 0.0
            
        except Exception as e:
            self.logger.error(f"Order book pressure calculation failed: {e}")
            return 0.0
    
    async def _calculate_depth_imbalance(self, order_book: OrderBookSnapshot) -> float:
        """Calculate weighted depth imbalance."""
        try:
            # Weight levels by proximity to mid price
            mid_price = order_book.mid_price
            
            weighted_bid_depth = 0.0
            weighted_ask_depth = 0.0
            
            for i, bid_level in enumerate(order_book.bids[:10]):
                weight = 1.0 / (i + 1)  # Closer levels get higher weight
                weighted_bid_depth += bid_level.quantity * weight
            
            for i, ask_level in enumerate(order_book.asks[:10]):
                weight = 1.0 / (i + 1)  # Closer levels get higher weight
                weighted_ask_depth += ask_level.quantity * weight
            
            total_depth = weighted_bid_depth + weighted_ask_depth
            
            if total_depth > 0:
                return (weighted_bid_depth - weighted_ask_depth) / total_depth
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Depth imbalance calculation failed: {e}")
            return 0.0
    
    async def _detect_market_regime(self, symbol: str, trades: List[Trade], order_books: List[OrderBookSnapshot]) -> MarketRegime:
        """Detect current market regime."""
        try:
            if len(trades) < 50 or len(order_books) < 20:
                return MarketRegime.RANGE_BOUND
            
            # Calculate price movement and volatility
            prices = [trade.price for trade in trades]
            price_changes = np.diff(prices)
            
            # Trend detection
            if len(prices) >= 20:
                recent_prices = prices[-20:]
                trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                trend_strength = abs(trend_slope) / np.mean(recent_prices)
                
                if trend_strength > 0.001:  # 0.1% threshold
                    if trend_slope > 0:
                        return MarketRegime.TRENDING_UP
                    else:
                        return MarketRegime.TRENDING_DOWN
            
            # Volatility regime
            volatility = np.std(price_changes) / np.mean(prices) if prices else 0
            
            if volatility > 0.002:  # 0.2% threshold
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.0005:  # 0.05% threshold
                return MarketRegime.LOW_VOLATILITY
            
            # Breakout detection (sudden volume and price movement)
            recent_volumes = [trade.quantity for trade in trades[-10:]]
            avg_recent_volume = np.mean(recent_volumes)
            historical_avg_volume = np.mean([trade.quantity for trade in trades[:-10]])
            
            if avg_recent_volume > historical_avg_volume * 2 and volatility > 0.001:
                return MarketRegime.BREAKOUT
            
            return MarketRegime.RANGE_BOUND
            
        except Exception as e:
            self.logger.error(f"Market regime detection failed for {symbol}: {e}")
            return MarketRegime.RANGE_BOUND
    
    def get_flow_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive flow analysis for symbol."""
        try:
            if symbol not in self.flow_metrics:
                return None
            
            metrics = self.flow_metrics[symbol]
            volume_profile = self.volume_profiles.get(symbol)
            microstructure = self.microstructure.get(symbol)
            
            analysis = {
                'symbol': symbol,
                'timestamp': metrics.timestamp.isoformat(),
                
                # Flow metrics
                'flow_direction': metrics.flow_direction.value,
                'flow_strength': metrics.flow_strength,
                'volume_imbalance': metrics.volume_imbalance,
                'aggressive_ratio': metrics.aggressive_ratio,
                'total_volume': metrics.total_volume,
                'trade_count': metrics.trade_count,
                'avg_trade_size': metrics.avg_trade_size,
                'large_trade_ratio': metrics.large_trade_ratio,
                'price_impact_bps': metrics.price_impact_bps,
                
                # Volume profile
                'volume_profile': {
                    'point_of_control': volume_profile.point_of_control if volume_profile else None,
                    'value_area_high': volume_profile.value_area_high if volume_profile else None,
                    'value_area_low': volume_profile.value_area_low if volume_profile else None,
                    'vwap': volume_profile.vwap if volume_profile else None
                } if volume_profile else None,
                
                # Microstructure
                'microstructure': {
                    'liquidity_imbalance': microstructure.liquidity_imbalance if microstructure else None,
                    'order_book_pressure': microstructure.order_book_pressure if microstructure else None,
                    'depth_imbalance': microstructure.depth_imbalance if microstructure else None,
                    'market_regime': microstructure.regime.value if microstructure else None,
                    'effective_spread_bps': microstructure.effective_spread_bps if microstructure else None,
                    'time_between_trades_ms': microstructure.time_between_trades_ms if microstructure else None
                } if microstructure else None
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to get flow analysis for {symbol}: {e}")
            return None
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get analysis performance summary."""
        try:
            return {
                'running': self.running,
                'analyzing_symbols': len(self.analyzing_symbols),
                'processed_trades': self.processed_trades,
                'processed_quotes': self.processed_quotes,
                'analysis_latency_ms': self.analysis_latency_ms,
                'symbols_with_flow_data': len(self.flow_metrics),
                'symbols_with_volume_profile': len(self.volume_profiles),
                'symbols_with_microstructure': len(self.microstructure)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate analysis summary: {e}")
            return {'error': 'Unable to generate summary'}


class TickAggregator:
    """Aggregates tick data for analysis."""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.logger = TradingLogger()


class FlowDetector:
    """Detects order flow patterns."""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.logger = TradingLogger()


class RegimeAnalyzer:
    """Analyzes market regimes."""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.logger = TradingLogger()


class LiquidityAnalyzer:
    """Analyzes liquidity conditions."""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.logger = TradingLogger()