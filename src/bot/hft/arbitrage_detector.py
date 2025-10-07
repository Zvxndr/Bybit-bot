"""
Advanced Arbitrage Detection Engine for High-Frequency Trading.
Detects and analyzes various arbitrage opportunities across markets and instruments.
"""

import asyncio
import numpy as np
import pandas as pd
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
    from scipy.optimize import minimize
    import networkx as nx
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class ArbitrageType(Enum):
    """Types of arbitrage opportunities."""
    SPATIAL = "spatial"              # Cross-exchange price differences
    TEMPORAL = "temporal"            # Time-based price discrepancies
    TRIANGULAR = "triangular"        # Three-currency arbitrage
    STATISTICAL = "statistical"     # Mean reversion opportunities
    MERGER = "merger"                # Merger arbitrage
    CALENDAR = "calendar"            # Calendar spread arbitrage
    VOLATILITY = "volatility"        # Volatility arbitrage
    FUNDING_RATE = "funding_rate"    # Funding rate arbitrage
    BASIS = "basis"                  # Futures-spot basis
    DIVIDEND = "dividend"            # Dividend arbitrage

class ArbitrageStatus(Enum):
    """Arbitrage opportunity status."""
    DETECTED = "detected"
    VALIDATED = "validated"
    EXECUTING = "executing"
    EXECUTED = "executed"
    EXPIRED = "expired"
    FAILED = "failed"

class RiskLevel(Enum):
    """Risk levels for arbitrage opportunities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity data structure."""
    opportunity_id: str
    arbitrage_type: ArbitrageType
    symbols: List[str]
    exchanges: List[str]
    expected_profit: float
    expected_profit_bps: float
    confidence_score: float
    risk_level: RiskLevel
    execution_time_window_ms: int
    required_capital: float
    detection_timestamp: datetime = field(default_factory=datetime.now)
    expiration_timestamp: Optional[datetime] = None
    status: ArbitrageStatus = ArbitrageStatus.DETECTED
    
    # Price and quantity data
    buy_price: Optional[float] = None
    sell_price: Optional[float] = None
    buy_exchange: Optional[str] = None
    sell_exchange: Optional[str] = None
    max_quantity: Optional[float] = None
    
    # Statistical data
    price_spread: Optional[float] = None
    spread_zscore: Optional[float] = None
    volume_weighted_spread: Optional[float] = None
    execution_costs: Optional[float] = None
    slippage_estimate: Optional[float] = None
    
    # Triangular arbitrage specific
    currency_path: Optional[List[str]] = None
    implied_rates: Optional[List[float]] = None
    
    # Additional metadata
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    execution_strategy: Optional[str] = None

@dataclass
class MarketPrice:
    """Market price data structure."""
    symbol: str
    exchange: str
    bid: float
    ask: float
    mid: float
    volume: float
    timestamp: datetime = field(default_factory=datetime.now)
    spread_bps: float = 0.0

@dataclass
class ArbitrageMetrics:
    """Arbitrage detection metrics."""
    total_opportunities_detected: int = 0
    opportunities_by_type: Dict[ArbitrageType, int] = field(default_factory=dict)
    avg_profit_bps: float = 0.0
    avg_confidence_score: float = 0.0
    successful_executions: int = 0
    failed_executions: int = 0
    total_profit_realized: float = 0.0
    detection_latency_ms: float = 0.0
    false_positive_rate: float = 0.0

class ArbitrageDetector:
    """Advanced arbitrage detection engine."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Arbitrage detection configuration
        self.arb_config = {
            'min_profit_bps': 5,           # Minimum 5 bps profit
            'min_confidence': 0.7,         # Minimum 70% confidence
            'max_execution_time_ms': 500,  # Max 500ms execution window
            'min_volume_usd': 1000,        # Minimum $1k volume
            'max_slippage_bps': 3,         # Maximum 3 bps slippage
            'enable_spatial': True,        # Enable cross-exchange arbitrage
            'enable_triangular': True,     # Enable triangular arbitrage
            'enable_statistical': True,    # Enable statistical arbitrage
            'enable_basis': True,          # Enable futures-spot arbitrage
            'spatial_min_spread_bps': 8,   # Minimum spatial spread
            'triangular_min_profit_bps': 3, # Minimum triangular profit
            'statistical_lookback_periods': 100,
            'statistical_zscore_threshold': 2.0,
            'max_opportunities_per_symbol': 5,
            'opportunity_ttl_seconds': 30
        }
        
        # Market data storage
        self.market_prices: Dict[str, Dict[str, MarketPrice]] = defaultdict(dict)  # symbol -> exchange -> price
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        
        # Arbitrage opportunities
        self.active_opportunities: Dict[str, ArbitrageOpportunity] = {}
        self.opportunity_history: deque = deque(maxlen=10000)
        self.metrics = ArbitrageMetrics()
        
        # Detection engines
        self.spatial_detector = SpatialArbitrageDetector(self)
        self.triangular_detector = TriangularArbitrageDetector(self)
        self.statistical_detector = StatisticalArbitrageDetector(self)
        
        # Threading and monitoring
        self.arb_lock = threading.Lock()
        self.running = False
        self.detection_task = None
        
        self.logger.info("ArbitrageDetector initialized")
    
    async def start_detection(self, symbols: List[str], exchanges: List[str]):
        """Start arbitrage detection."""
        try:
            if self.running:
                return
            
            self.running = True
            
            # Initialize detection for symbol-exchange pairs
            for symbol in symbols:
                for exchange in exchanges:
                    if exchange not in self.market_prices[symbol]:
                        self.market_prices[symbol][exchange] = MarketPrice(
                            symbol=symbol,
                            exchange=exchange,
                            bid=0.0,
                            ask=0.0,
                            mid=0.0,
                            volume=0.0
                        )
            
            # Start detection loop
            self.detection_task = asyncio.create_task(
                self._detection_loop(symbols, exchanges)
            )
            
            self.logger.info(f"Arbitrage detection started for {len(symbols)} symbols across {len(exchanges)} exchanges")
            
        except Exception as e:
            self.logger.error(f"Failed to start arbitrage detection: {e}")
            self.running = False
            raise
    
    async def stop_detection(self):
        """Stop arbitrage detection."""
        try:
            self.running = False
            
            # Stop detection task
            if self.detection_task:
                self.detection_task.cancel()
                try:
                    await self.detection_task
                except asyncio.CancelledError:
                    pass
            
            # Clear active opportunities
            self.active_opportunities.clear()
            
            self.logger.info("Arbitrage detection stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop arbitrage detection: {e}")
    
    async def update_market_price(self, symbol: str, exchange: str, bid: float, ask: float, volume: float):
        """Update market price data."""
        try:
            with self.arb_lock:
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                spread_bps = ((ask - bid) / mid * 10000) if mid > 0 else 0
                
                market_price = MarketPrice(
                    symbol=symbol,
                    exchange=exchange,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    volume=volume,
                    timestamp=datetime.now(),
                    spread_bps=spread_bps
                )
                
                self.market_prices[symbol][exchange] = market_price
                self.price_history[f"{symbol}_{exchange}"].append({
                    'timestamp': market_price.timestamp,
                    'mid': mid,
                    'bid': bid,
                    'ask': ask,
                    'volume': volume
                })
                
        except Exception as e:
            self.logger.error(f"Failed to update market price for {symbol}@{exchange}: {e}")
    
    async def _detection_loop(self, symbols: List[str], exchanges: List[str]):
        """Main arbitrage detection loop."""
        try:
            while self.running:
                detection_start = time.perf_counter()
                
                # Clean expired opportunities
                await self._clean_expired_opportunities()
                
                # Detect spatial arbitrage
                if self.arb_config['enable_spatial']:
                    await self._detect_spatial_arbitrage(symbols, exchanges)
                
                # Detect triangular arbitrage
                if self.arb_config['enable_triangular']:
                    await self._detect_triangular_arbitrage(symbols, exchanges)
                
                # Detect statistical arbitrage
                if self.arb_config['enable_statistical']:
                    await self._detect_statistical_arbitrage(symbols, exchanges)
                
                # Update detection metrics
                detection_time_ms = (time.perf_counter() - detection_start) * 1000
                self.metrics.detection_latency_ms = detection_time_ms
                
                # Sleep until next detection cycle
                await asyncio.sleep(0.001)  # 1ms detection cycle
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Arbitrage detection loop error: {e}")
    
    async def _detect_spatial_arbitrage(self, symbols: List[str], exchanges: List[str]):
        """Detect spatial (cross-exchange) arbitrage opportunities."""
        try:
            for symbol in symbols:
                if len(exchanges) < 2:
                    continue
                
                prices = self.market_prices[symbol]
                valid_prices = {ex: price for ex, price in prices.items() 
                              if price.bid > 0 and price.ask > 0 and ex in exchanges}
                
                if len(valid_prices) < 2:
                    continue
                
                # Find best bid and ask across exchanges
                best_bid_exchange = max(valid_prices.keys(), key=lambda x: valid_prices[x].bid)
                best_ask_exchange = min(valid_prices.keys(), key=lambda x: valid_prices[x].ask)
                
                if best_bid_exchange == best_ask_exchange:
                    continue
                
                best_bid = valid_prices[best_bid_exchange].bid
                best_ask = valid_prices[best_ask_exchange].ask
                
                if best_bid <= best_ask:
                    continue
                
                # Calculate profit metrics
                profit_per_unit = best_bid - best_ask
                mid_price = (best_bid + best_ask) / 2
                profit_bps = (profit_per_unit / mid_price) * 10000
                
                if profit_bps < self.arb_config['spatial_min_spread_bps']:
                    continue
                
                # Calculate maximum tradeable quantity
                bid_volume = valid_prices[best_bid_exchange].volume
                ask_volume = valid_prices[best_ask_exchange].volume
                max_quantity = min(bid_volume, ask_volume)
                
                if max_quantity * mid_price < self.arb_config['min_volume_usd']:
                    continue
                
                # Estimate execution costs and slippage
                execution_costs = await self._estimate_execution_costs(symbol, [best_bid_exchange, best_ask_exchange])
                slippage_estimate = await self._estimate_slippage(symbol, max_quantity, [best_bid_exchange, best_ask_exchange])
                
                net_profit_bps = profit_bps - execution_costs - slippage_estimate
                
                if net_profit_bps < self.arb_config['min_profit_bps']:
                    continue
                
                # Calculate confidence score
                confidence = await self._calculate_spatial_confidence(symbol, best_bid_exchange, best_ask_exchange)
                
                if confidence < self.arb_config['min_confidence']:
                    continue
                
                # Check if similar opportunity already exists
                opportunity_key = f"spatial_{symbol}_{best_bid_exchange}_{best_ask_exchange}"
                if opportunity_key in self.active_opportunities:
                    continue
                
                # Create arbitrage opportunity
                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"spatial_{symbol}_{int(time.time() * 1000)}",
                    arbitrage_type=ArbitrageType.SPATIAL,
                    symbols=[symbol],
                    exchanges=[best_bid_exchange, best_ask_exchange],
                    expected_profit=profit_per_unit * max_quantity,
                    expected_profit_bps=net_profit_bps,
                    confidence_score=confidence,
                    risk_level=self._assess_risk_level(net_profit_bps, confidence),
                    execution_time_window_ms=self.arb_config['max_execution_time_ms'],
                    required_capital=best_ask * max_quantity,
                    expiration_timestamp=datetime.now() + timedelta(seconds=self.arb_config['opportunity_ttl_seconds']),
                    buy_price=best_ask,
                    sell_price=best_bid,
                    buy_exchange=best_ask_exchange,
                    sell_exchange=best_bid_exchange,
                    max_quantity=max_quantity,
                    price_spread=profit_per_unit,
                    execution_costs=execution_costs,
                    slippage_estimate=slippage_estimate
                )
                
                await self._register_opportunity(opportunity)
                
        except Exception as e:
            self.logger.error(f"Spatial arbitrage detection failed: {e}")
    
    async def _detect_triangular_arbitrage(self, symbols: List[str], exchanges: List[str]):
        """Detect triangular arbitrage opportunities."""
        try:
            if not HAS_SCIPY:
                return
            
            # Group symbols by base/quote currencies
            currency_pairs = {}
            for symbol in symbols:
                if '/' in symbol:
                    base, quote = symbol.split('/')
                    currency_pairs[symbol] = (base, quote)
            
            currencies = set()
            for base, quote in currency_pairs.values():
                currencies.add(base)
                currencies.add(quote)
            
            if len(currencies) < 3:
                return
            
            # Find triangular opportunities for each exchange
            for exchange in exchanges:
                await self._find_triangular_opportunities_for_exchange(
                    exchange, currency_pairs, currencies
                )
                
        except Exception as e:
            self.logger.error(f"Triangular arbitrage detection failed: {e}")
    
    async def _find_triangular_opportunities_for_exchange(self, exchange: str, currency_pairs: Dict[str, Tuple[str, str]], currencies: Set[str]):
        """Find triangular arbitrage opportunities for a specific exchange."""
        try:
            # Build currency graph
            graph = nx.DiGraph()
            
            for symbol, (base, quote) in currency_pairs.items():
                if exchange in self.market_prices[symbol]:
                    price_data = self.market_prices[symbol][exchange]
                    
                    if price_data.bid > 0 and price_data.ask > 0:
                        # Add edges for both directions
                        # Base -> Quote (sell base for quote)
                        graph.add_edge(base, quote, 
                                     rate=price_data.bid, 
                                     symbol=symbol, 
                                     side='sell')
                        # Quote -> Base (buy base with quote)
                        graph.add_edge(quote, base, 
                                     rate=1/price_data.ask, 
                                     symbol=symbol, 
                                     side='buy')
            
            # Find triangular paths
            for start_currency in currencies:
                try:
                    # Find all simple cycles starting from this currency
                    cycles = list(nx.simple_cycles(graph))
                    
                    for cycle in cycles:
                        if len(cycle) == 3 and start_currency in cycle:
                            await self._evaluate_triangular_cycle(exchange, cycle, graph)
                            
                except nx.NetworkXNoCycle:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Triangular opportunity search failed for {exchange}: {e}")
    
    async def _evaluate_triangular_cycle(self, exchange: str, cycle: List[str], graph: nx.DiGraph):
        """Evaluate a triangular arbitrage cycle."""
        try:
            if len(cycle) != 3:
                return
            
            # Calculate the path rates
            path_rate = 1.0
            path_symbols = []
            path_sides = []
            min_volume = float('inf')
            
            for i in range(len(cycle)):
                from_currency = cycle[i]
                to_currency = cycle[(i + 1) % len(cycle)]
                
                if not graph.has_edge(from_currency, to_currency):
                    return
                
                edge_data = graph[from_currency][to_currency]
                rate = edge_data['rate']
                symbol = edge_data['symbol']
                side = edge_data['side']
                
                path_rate *= rate
                path_symbols.append(symbol)
                path_sides.append(side)
                
                # Get volume constraint
                if symbol in self.market_prices and exchange in self.market_prices[symbol]:
                    volume = self.market_prices[symbol][exchange].volume
                    min_volume = min(min_volume, volume)
            
            # Check if profitable
            profit_ratio = path_rate - 1.0
            profit_bps = profit_ratio * 10000
            
            if profit_bps < self.arb_config['triangular_min_profit_bps']:
                return
            
            # Estimate execution costs
            execution_costs = len(path_symbols) * 2  # Assume 2 bps per trade
            net_profit_bps = profit_bps - execution_costs
            
            if net_profit_bps < self.arb_config['min_profit_bps']:
                return
            
            # Calculate required capital (starting with first currency)
            start_currency = cycle[0]
            required_capital = 1000  # $1000 base amount for calculation
            
            # Create opportunity
            opportunity_id = f"triangular_{exchange}_{'_'.join(cycle)}_{int(time.time() * 1000)}"
            
            opportunity = ArbitrageOpportunity(
                opportunity_id=opportunity_id,
                arbitrage_type=ArbitrageType.TRIANGULAR,
                symbols=path_symbols,
                exchanges=[exchange],
                expected_profit=required_capital * profit_ratio,
                expected_profit_bps=net_profit_bps,
                confidence_score=0.8,  # Default confidence for triangular
                risk_level=self._assess_risk_level(net_profit_bps, 0.8),
                execution_time_window_ms=self.arb_config['max_execution_time_ms'],
                required_capital=required_capital,
                expiration_timestamp=datetime.now() + timedelta(seconds=self.arb_config['opportunity_ttl_seconds']),
                max_quantity=min_volume,
                currency_path=cycle,
                implied_rates=[1/rate for rate in [path_rate]],
                execution_strategy='triangular_simultaneous'
            )
            
            await self._register_opportunity(opportunity)
            
        except Exception as e:
            self.logger.error(f"Triangular cycle evaluation failed: {e}")
    
    async def _detect_statistical_arbitrage(self, symbols: List[str], exchanges: List[str]):
        """Detect statistical arbitrage opportunities."""
        try:
            if len(symbols) < 2:
                return
            
            # Update correlation matrix
            await self._update_correlation_matrix(symbols, exchanges)
            
            # Find cointegrated pairs
            cointegrated_pairs = await self._find_cointegrated_pairs(symbols, exchanges)
            
            for pair in cointegrated_pairs:
                await self._evaluate_statistical_opportunity(pair, exchanges)
                
        except Exception as e:
            self.logger.error(f"Statistical arbitrage detection failed: {e}")
    
    async def _update_correlation_matrix(self, symbols: List[str], exchanges: List[str]):
        """Update correlation matrix between symbol pairs."""
        try:
            if not HAS_SCIPY:
                return
            
            # Get price series for correlation calculation
            price_series = {}
            
            for symbol in symbols:
                for exchange in exchanges:
                    key = f"{symbol}_{exchange}"
                    history = self.price_history[key]
                    
                    if len(history) >= 50:  # Minimum data points
                        prices = [entry['mid'] for entry in history if entry['mid'] > 0]
                        if len(prices) >= 50:
                            price_series[key] = np.array(prices[-100:])  # Last 100 points
            
            # Calculate correlations
            keys = list(price_series.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    key1, key2 = keys[i], keys[j]
                    
                    if len(price_series[key1]) > 0 and len(price_series[key2]) > 0:
                        min_len = min(len(price_series[key1]), len(price_series[key2]))
                        corr, p_value = stats.pearsonr(
                            price_series[key1][-min_len:],
                            price_series[key2][-min_len:]
                        )
                        
                        if not np.isnan(corr):
                            self.correlation_matrix[(key1, key2)] = corr
                            self.correlation_matrix[(key2, key1)] = corr
            
        except Exception as e:
            self.logger.error(f"Correlation matrix update failed: {e}")
    
    async def _find_cointegrated_pairs(self, symbols: List[str], exchanges: List[str]) -> List[Tuple[str, str]]:
        """Find cointegrated symbol pairs."""
        try:
            cointegrated_pairs = []
            
            # Look for highly correlated pairs
            for (key1, key2), correlation in self.correlation_matrix.items():
                if abs(correlation) > 0.8:  # High correlation threshold
                    # Extract symbol and exchange
                    symbol1, exchange1 = key1.rsplit('_', 1)
                    symbol2, exchange2 = key2.rsplit('_', 1)
                    
                    # Only consider same exchange for statistical arbitrage
                    if exchange1 == exchange2 and symbol1 != symbol2:
                        cointegrated_pairs.append((key1, key2))
            
            return cointegrated_pairs[:10]  # Limit to top 10 pairs
            
        except Exception as e:
            self.logger.error(f"Cointegration search failed: {e}")
            return []
    
    async def _evaluate_statistical_opportunity(self, pair: Tuple[str, str], exchanges: List[str]):
        """Evaluate statistical arbitrage opportunity for a pair."""
        try:
            key1, key2 = pair
            
            # Get price histories
            history1 = list(self.price_history[key1])
            history2 = list(self.price_history[key2])
            
            if len(history1) < 50 or len(history2) < 50:
                return
            
            # Calculate price ratio
            prices1 = np.array([entry['mid'] for entry in history1[-100:]])
            prices2 = np.array([entry['mid'] for entry in history2[-100:]])
            
            min_len = min(len(prices1), len(prices2))
            if min_len < 50:
                return
            
            prices1 = prices1[-min_len:]
            prices2 = prices2[-min_len:]
            
            # Calculate spread
            spread = prices1 / prices2
            spread_mean = np.mean(spread)
            spread_std = np.std(spread)
            
            if spread_std == 0:
                return
            
            # Current spread and z-score
            current_spread = spread[-1]
            z_score = (current_spread - spread_mean) / spread_std
            
            if abs(z_score) < self.arb_config['statistical_zscore_threshold']:
                return
            
            # Extract symbols and exchanges
            symbol1, exchange1 = key1.rsplit('_', 1)
            symbol2, exchange2 = key2.rsplit('_', 1)
            
            if exchange1 != exchange2:
                return
            
            # Determine trade direction
            if z_score > 0:  # Spread too high - sell symbol1, buy symbol2
                long_symbol = symbol2
                short_symbol = symbol1
                expected_direction = -1
            else:  # Spread too low - buy symbol1, sell symbol2
                long_symbol = symbol1
                short_symbol = symbol2
                expected_direction = 1
            
            # Calculate expected profit
            expected_reversion = abs(z_score) * spread_std
            profit_bps = (expected_reversion / current_spread) * 10000
            
            if profit_bps < self.arb_config['min_profit_bps']:
                return
            
            # Get current prices
            if (symbol1 not in self.market_prices or exchange1 not in self.market_prices[symbol1] or
                symbol2 not in self.market_prices or exchange2 not in self.market_prices[symbol2]):
                return
            
            price1_data = self.market_prices[symbol1][exchange1]
            price2_data = self.market_prices[symbol2][exchange2]
            
            max_quantity = min(price1_data.volume, price2_data.volume)
            required_capital = max(price1_data.mid * max_quantity, price2_data.mid * max_quantity)
            
            # Create opportunity
            opportunity = ArbitrageOpportunity(
                opportunity_id=f"statistical_{symbol1}_{symbol2}_{exchange1}_{int(time.time() * 1000)}",
                arbitrage_type=ArbitrageType.STATISTICAL,
                symbols=[symbol1, symbol2],
                exchanges=[exchange1],
                expected_profit=required_capital * (profit_bps / 10000),
                expected_profit_bps=profit_bps,
                confidence_score=min(0.9, abs(z_score) / 3.0),  # Higher z-score = higher confidence
                risk_level=self._assess_risk_level(profit_bps, abs(z_score) / 3.0),
                execution_time_window_ms=5000,  # Longer window for statistical
                required_capital=required_capital,
                expiration_timestamp=datetime.now() + timedelta(minutes=5),  # Longer TTL
                max_quantity=max_quantity,
                spread_zscore=z_score,
                execution_strategy='statistical_mean_reversion'
            )
            
            await self._register_opportunity(opportunity)
            
        except Exception as e:
            self.logger.error(f"Statistical opportunity evaluation failed: {e}")
    
    async def _estimate_execution_costs(self, symbol: str, exchanges: List[str]) -> float:
        """Estimate execution costs in basis points."""
        try:
            # Simplified execution cost model
            base_cost_bps = 1.0  # Base 1 bps per trade
            
            # Multiple exchange penalty
            exchange_penalty = (len(exchanges) - 1) * 0.5
            
            # Market volatility adjustment
            total_spread = 0
            valid_exchanges = 0
            
            for exchange in exchanges:
                if symbol in self.market_prices and exchange in self.market_prices[symbol]:
                    price_data = self.market_prices[symbol][exchange]
                    total_spread += price_data.spread_bps
                    valid_exchanges += 1
            
            if valid_exchanges > 0:
                avg_spread_bps = total_spread / valid_exchanges
                volatility_adjustment = max(0, min(2.0, avg_spread_bps / 10))  # Max 2 bps adjustment
            else:
                volatility_adjustment = 1.0
            
            total_cost_bps = base_cost_bps + exchange_penalty + volatility_adjustment
            
            return total_cost_bps
            
        except Exception as e:
            self.logger.error(f"Execution cost estimation failed: {e}")
            return 3.0  # Default 3 bps
    
    async def _estimate_slippage(self, symbol: str, quantity: float, exchanges: List[str]) -> float:
        """Estimate slippage in basis points."""
        try:
            # Simplified slippage model
            base_slippage_bps = 0.5
            
            # Volume impact
            total_volume = 0
            valid_exchanges = 0
            
            for exchange in exchanges:
                if symbol in self.market_prices and exchange in self.market_prices[symbol]:
                    price_data = self.market_prices[symbol][exchange]
                    total_volume += price_data.volume
                    valid_exchanges += 1
            
            if valid_exchanges > 0:
                avg_volume = total_volume / valid_exchanges
                if avg_volume > 0:
                    volume_impact = min(2.0, quantity / avg_volume)  # Max 2 bps impact
                else:
                    volume_impact = 2.0
            else:
                volume_impact = 2.0
            
            total_slippage_bps = base_slippage_bps + volume_impact
            
            return total_slippage_bps
            
        except Exception as e:
            self.logger.error(f"Slippage estimation failed: {e}")
            return 2.0  # Default 2 bps
    
    async def _calculate_spatial_confidence(self, symbol: str, exchange1: str, exchange2: str) -> float:
        """Calculate confidence score for spatial arbitrage."""
        try:
            confidence = 0.5  # Base confidence
            
            # Price data quality
            price1 = self.market_prices[symbol][exchange1]
            price2 = self.market_prices[symbol][exchange2]
            
            # Recent data bonus
            now = datetime.now()
            age1 = (now - price1.timestamp).total_seconds()
            age2 = (now - price2.timestamp).total_seconds()
            
            if age1 < 1 and age2 < 1:  # Fresh data
                confidence += 0.2
            elif age1 < 5 and age2 < 5:  # Recent data
                confidence += 0.1
            
            # Volume confidence
            if price1.volume > 1000 and price2.volume > 1000:
                confidence += 0.1
            
            # Spread tightness
            if price1.spread_bps < 10 and price2.spread_bps < 10:
                confidence += 0.1
            
            # Historical consistency (if we have history)
            key1 = f"{symbol}_{exchange1}"
            key2 = f"{symbol}_{exchange2}"
            
            if (key1, key2) in self.correlation_matrix:
                correlation = abs(self.correlation_matrix[(key1, key2)])
                confidence += correlation * 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Spatial confidence calculation failed: {e}")
            return 0.5
    
    def _assess_risk_level(self, profit_bps: float, confidence: float) -> RiskLevel:
        """Assess risk level of arbitrage opportunity."""
        try:
            risk_score = 0
            
            # Profit margin factor
            if profit_bps < 10:
                risk_score += 2
            elif profit_bps < 20:
                risk_score += 1
            
            # Confidence factor
            if confidence < 0.6:
                risk_score += 3
            elif confidence < 0.8:
                risk_score += 1
            
            # Determine risk level
            if risk_score <= 1:
                return RiskLevel.LOW
            elif risk_score <= 3:
                return RiskLevel.MEDIUM
            elif risk_score <= 5:
                return RiskLevel.HIGH
            else:
                return RiskLevel.EXTREME
                
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return RiskLevel.HIGH
    
    async def _register_opportunity(self, opportunity: ArbitrageOpportunity):
        """Register a new arbitrage opportunity."""
        try:
            # Check if we have too many opportunities for this symbol combination
            symbol_key = '_'.join(sorted(opportunity.symbols))
            existing_count = sum(1 for opp in self.active_opportunities.values()
                               if '_'.join(sorted(opp.symbols)) == symbol_key)
            
            if existing_count >= self.arb_config['max_opportunities_per_symbol']:
                return
            
            # Add to active opportunities
            self.active_opportunities[opportunity.opportunity_id] = opportunity
            self.opportunity_history.append(opportunity)
            
            # Update metrics
            self.metrics.total_opportunities_detected += 1
            if opportunity.arbitrage_type not in self.metrics.opportunities_by_type:
                self.metrics.opportunities_by_type[opportunity.arbitrage_type] = 0
            self.metrics.opportunities_by_type[opportunity.arbitrage_type] += 1
            
            # Log opportunity
            self.logger.info(
                f"Arbitrage opportunity detected: {opportunity.arbitrage_type.value} "
                f"{opportunity.symbols} {opportunity.expected_profit_bps:.2f}bps "
                f"confidence={opportunity.confidence_score:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to register opportunity: {e}")
    
    async def _clean_expired_opportunities(self):
        """Clean expired arbitrage opportunities."""
        try:
            now = datetime.now()
            expired_ids = []
            
            for opp_id, opportunity in self.active_opportunities.items():
                if (opportunity.expiration_timestamp and 
                    now > opportunity.expiration_timestamp and
                    opportunity.status in [ArbitrageStatus.DETECTED, ArbitrageStatus.VALIDATED]):
                    
                    opportunity.status = ArbitrageStatus.EXPIRED
                    expired_ids.append(opp_id)
            
            # Remove expired opportunities
            for opp_id in expired_ids:
                del self.active_opportunities[opp_id]
            
            if expired_ids:
                self.logger.debug(f"Cleaned {len(expired_ids)} expired opportunities")
                
        except Exception as e:
            self.logger.error(f"Failed to clean expired opportunities: {e}")
    
    def get_active_opportunities(self, arbitrage_type: Optional[ArbitrageType] = None, 
                                min_profit_bps: Optional[float] = None) -> List[ArbitrageOpportunity]:
        """Get active arbitrage opportunities with optional filtering."""
        try:
            opportunities = list(self.active_opportunities.values())
            
            if arbitrage_type:
                opportunities = [opp for opp in opportunities if opp.arbitrage_type == arbitrage_type]
            
            if min_profit_bps:
                opportunities = [opp for opp in opportunities if opp.expected_profit_bps >= min_profit_bps]
            
            # Sort by expected profit descending
            opportunities.sort(key=lambda x: x.expected_profit_bps, reverse=True)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Failed to get active opportunities: {e}")
            return []
    
    def get_detection_metrics(self) -> Dict[str, Any]:
        """Get arbitrage detection metrics."""
        try:
            if self.metrics.total_opportunities_detected > 0:
                self.metrics.avg_profit_bps = np.mean([
                    opp.expected_profit_bps for opp in self.opportunity_history
                    if opp.expected_profit_bps > 0
                ])
                
                self.metrics.avg_confidence_score = np.mean([
                    opp.confidence_score for opp in self.opportunity_history
                ])
            
            return {
                'running': self.running,
                'total_opportunities_detected': self.metrics.total_opportunities_detected,
                'active_opportunities': len(self.active_opportunities),
                'opportunities_by_type': dict(self.metrics.opportunities_by_type),
                'avg_profit_bps': self.metrics.avg_profit_bps,
                'avg_confidence_score': self.metrics.avg_confidence_score,
                'successful_executions': self.metrics.successful_executions,
                'failed_executions': self.metrics.failed_executions,
                'total_profit_realized': self.metrics.total_profit_realized,
                'detection_latency_ms': self.metrics.detection_latency_ms,
                'false_positive_rate': self.metrics.false_positive_rate
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate detection metrics: {e}")
            return {'error': 'Unable to generate metrics'}


class SpatialArbitrageDetector:
    """Specialized spatial arbitrage detector."""
    
    def __init__(self, main_detector):
        self.main_detector = main_detector
        self.logger = TradingLogger()


class TriangularArbitrageDetector:
    """Specialized triangular arbitrage detector."""
    
    def __init__(self, main_detector):
        self.main_detector = main_detector
        self.logger = TradingLogger()


class StatisticalArbitrageDetector:
    """Specialized statistical arbitrage detector."""
    
    def __init__(self, main_detector):
        self.main_detector = main_detector
        self.logger = TradingLogger()