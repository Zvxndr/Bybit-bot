"""
Cross-Exchange Feature Calculator

Advanced feature engineering system that calculates sophisticated features across multiple
cryptocurrency exchanges to identify arbitrage opportunities, market inefficiencies, and
enhanced trading signals.

Key Features:
- Cross-exchange spread analysis and arbitrage detection
- Volume-weighted price comparisons and flow analysis
- Market depth and liquidity analysis across exchanges
- Correlation analysis between exchange prices and volumes
- Order book imbalance detection and flow prediction
- Exchange-specific premium/discount calculations
- Market efficiency scoring and timing signals

Supported Exchanges:
- Bybit (primary trading venue)
- Binance (reference and arbitbage)
- OKX (additional liquidity source)

Generated Features:
- Price spreads and percentage deviations
- Volume-weighted average prices (VWAP) comparisons
- Arbitrage opportunity scores and profitability estimates
- Market depth ratios and liquidity scores
- Price leadership indicators and lag analysis
- Volatility spreads and correlation coefficients
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
from collections import defaultdict, deque
import warnings
import statistics

from ..utils.logging import TradingLogger
from ..config.manager import ConfigurationManager


@dataclass
class CrossExchangeFeatures:
    """Container for cross-exchange features."""
    timestamp: datetime
    symbol: str
    
    # Price-based features
    bybit_price: float
    binance_price: float
    okx_price: float
    avg_price: float
    price_std: float
    
    # Spread features
    bybit_binance_spread: float
    bybit_okx_spread: float
    binance_okx_spread: float
    max_spread: float
    spread_volatility: float
    
    # Volume features
    bybit_volume: float
    binance_volume: float
    okx_volume: float
    total_volume: float
    volume_weighted_price: float
    volume_concentration: float
    
    # Arbitrage features
    best_arbitrage_spread: float
    arbitrage_opportunity: bool
    estimated_profit_bps: float
    arbitrage_direction: str  # "bybit_to_binance", "binance_to_okx", etc.
    
    # Market depth features
    depth_ratio_bybit: float
    depth_ratio_binance: float
    depth_ratio_okx: float
    liquidity_score: float
    
    # Correlation features
    price_correlation_1h: float
    volume_correlation_1h: float
    
    # Efficiency features
    price_efficiency_score: float
    market_leadership_score: Dict[str, float]


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity."""
    timestamp: datetime
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_bps: float
    estimated_profit_bps: float
    min_volume: float
    confidence_score: float
    duration_seconds: float


class CrossExchangeFeatureCalculator:
    """
    Advanced cross-exchange feature calculator for cryptocurrency trading.
    
    Analyzes market data across multiple exchanges to generate sophisticated
    features for trading algorithms, including arbitrage detection, market
    efficiency analysis, and liquidity assessment.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = TradingLogger("CrossExchangeFeatureCalculator")
        
        # Supported exchanges
        self.exchanges = ['bybit', 'binance', 'okx']
        self.primary_exchange = 'bybit'
        
        # Configuration
        self.config = {
            'min_arbitrage_bps': config_manager.get('features.cross_exchange.min_arbitrage_bps', 10),
            'max_spread_threshold': config_manager.get('features.cross_exchange.max_spread_threshold', 500),  # 5%
            'volume_weight_threshold': config_manager.get('features.cross_exchange.volume_weight_threshold', 0.1),
            'correlation_window': config_manager.get('features.cross_exchange.correlation_window', 60),  # minutes
            'efficiency_window': config_manager.get('features.cross_exchange.efficiency_window', 120),  # minutes
            'arbitrage_confidence_threshold': config_manager.get('features.cross_exchange.arbitrage_confidence', 0.7),
            'trading_fees': {
                'bybit': config_manager.get('fees.bybit.taker', 0.001),  # 0.1%
                'binance': config_manager.get('fees.binance.taker', 0.001),  # 0.1%
                'okx': config_manager.get('fees.okx.taker', 0.001)  # 0.1%
            }
        }
        
        # Data storage for historical analysis
        self.price_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        self.volume_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        self.depth_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
        
        # Feature cache
        self.feature_cache: Dict[str, CrossExchangeFeatures] = {}
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        self.max_opportunities = 1000
        
        # Statistical tracking
        self.correlation_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.efficiency_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Market leadership tracking
        self.price_changes: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
        self.leadership_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def calculate_features(self, market_data: Dict[str, Dict[str, Any]], symbol: str) -> Optional[CrossExchangeFeatures]:
        """
        Calculate comprehensive cross-exchange features.
        
        Args:
            market_data: Dict with exchange data {exchange: {price, volume, bid, ask, timestamp}}
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            CrossExchangeFeatures object or None if insufficient data
        """
        try:
            # Validate input data
            if not self._validate_market_data(market_data, symbol):
                return None
            
            timestamp = datetime.now()
            
            # Extract price and volume data
            prices = self._extract_prices(market_data)
            volumes = self._extract_volumes(market_data)
            depths = self._extract_depths(market_data)
            
            # Update historical data
            self._update_history(symbol, prices, volumes, depths, timestamp)
            
            # Calculate price-based features
            price_features = self._calculate_price_features(prices)
            
            # Calculate spread features
            spread_features = self._calculate_spread_features(prices)
            
            # Calculate volume features
            volume_features = self._calculate_volume_features(prices, volumes)
            
            # Calculate arbitrage features
            arbitrage_features = self._calculate_arbitrage_features(prices, volumes, symbol)
            
            # Calculate market depth features
            depth_features = self._calculate_depth_features(depths)
            
            # Calculate correlation features
            correlation_features = self._calculate_correlation_features(symbol)
            
            # Calculate efficiency features
            efficiency_features = self._calculate_efficiency_features(symbol, prices)
            
            # Combine all features
            features = CrossExchangeFeatures(
                timestamp=timestamp,
                symbol=symbol,
                # Price features
                bybit_price=prices.get('bybit', 0.0),
                binance_price=prices.get('binance', 0.0),
                okx_price=prices.get('okx', 0.0),
                avg_price=price_features['avg_price'],
                price_std=price_features['price_std'],
                # Spread features
                bybit_binance_spread=spread_features['bybit_binance_spread'],
                bybit_okx_spread=spread_features['bybit_okx_spread'],
                binance_okx_spread=spread_features['binance_okx_spread'],
                max_spread=spread_features['max_spread'],
                spread_volatility=spread_features['spread_volatility'],
                # Volume features
                bybit_volume=volumes.get('bybit', 0.0),
                binance_volume=volumes.get('binance', 0.0),
                okx_volume=volumes.get('okx', 0.0),
                total_volume=volume_features['total_volume'],
                volume_weighted_price=volume_features['vwap'],
                volume_concentration=volume_features['volume_concentration'],
                # Arbitrage features
                best_arbitrage_spread=arbitrage_features['best_spread'],
                arbitrage_opportunity=arbitrage_features['has_opportunity'],
                estimated_profit_bps=arbitrage_features['estimated_profit_bps'],
                arbitrage_direction=arbitrage_features['direction'],
                # Depth features
                depth_ratio_bybit=depth_features.get('bybit', 1.0),
                depth_ratio_binance=depth_features.get('binance', 1.0),
                depth_ratio_okx=depth_features.get('okx', 1.0),
                liquidity_score=depth_features.get('liquidity_score', 0.5),
                # Correlation features
                price_correlation_1h=correlation_features['price_correlation'],
                volume_correlation_1h=correlation_features['volume_correlation'],
                # Efficiency features
                price_efficiency_score=efficiency_features['efficiency_score'],
                market_leadership_score=efficiency_features['leadership_scores']
            )
            
            # Cache features
            self.feature_cache[f"{symbol}_{timestamp.isoformat()}"] = features
            
            # Clean old cache entries
            self._cleanup_cache()
            
            self.logger.debug(f"Calculated cross-exchange features for {symbol}: "
                            f"spread={spread_features['max_spread']:.2f}bps, "
                            f"arbitrage={arbitrage_features['has_opportunity']}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating cross-exchange features for {symbol}: {e}")
            return None
    
    def _validate_market_data(self, market_data: Dict[str, Dict[str, Any]], symbol: str) -> bool:
        """Validate that market data is sufficient for feature calculation."""
        if not market_data:
            return False
        
        # Check that we have data from at least 2 exchanges
        valid_exchanges = 0
        for exchange in self.exchanges:
            if exchange in market_data:
                exchange_data = market_data[exchange]
                if (isinstance(exchange_data, dict) and 
                    'price' in exchange_data and 
                    exchange_data['price'] > 0):
                    valid_exchanges += 1
        
        return valid_exchanges >= 2
    
    def _extract_prices(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Extract prices from market data."""
        prices = {}
        for exchange in self.exchanges:
            if exchange in market_data and 'price' in market_data[exchange]:
                price = market_data[exchange]['price']
                if isinstance(price, (int, float)) and price > 0:
                    prices[exchange] = float(price)
        return prices
    
    def _extract_volumes(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Extract volumes from market data."""
        volumes = {}
        for exchange in self.exchanges:
            if exchange in market_data and 'volume' in market_data[exchange]:
                volume = market_data[exchange]['volume']
                if isinstance(volume, (int, float)) and volume >= 0:
                    volumes[exchange] = float(volume)
        return volumes
    
    def _extract_depths(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Extract market depth information."""
        depths = {}
        for exchange in self.exchanges:
            if exchange in market_data:
                exchange_data = market_data[exchange]
                depth = {}
                
                if 'bid' in exchange_data and 'ask' in exchange_data:
                    try:
                        bid = float(exchange_data['bid'])
                        ask = float(exchange_data['ask'])
                        if bid > 0 and ask > bid:
                            depth['bid'] = bid
                            depth['ask'] = ask
                            depth['spread'] = ask - bid
                            depth['mid_price'] = (bid + ask) / 2
                    except (ValueError, TypeError):
                        pass
                
                if 'bid_size' in exchange_data and 'ask_size' in exchange_data:
                    try:
                        depth['bid_size'] = float(exchange_data['bid_size'])
                        depth['ask_size'] = float(exchange_data['ask_size'])
                    except (ValueError, TypeError):
                        pass
                
                if depth:
                    depths[exchange] = depth
        
        return depths
    
    def _update_history(self, symbol: str, prices: Dict[str, float], volumes: Dict[str, float], 
                       depths: Dict[str, Dict[str, float]], timestamp: datetime):
        """Update historical data for correlation and trend analysis."""
        for exchange in self.exchanges:
            if exchange in prices:
                self.price_history[symbol][exchange].append((timestamp, prices[exchange]))
                
                # Track price changes for leadership analysis
                if len(self.price_history[symbol][exchange]) > 1:
                    prev_price = self.price_history[symbol][exchange][-2][1]
                    price_change = (prices[exchange] - prev_price) / prev_price
                    self.price_changes[symbol][exchange].append((timestamp, price_change))
            
            if exchange in volumes:
                self.volume_history[symbol][exchange].append((timestamp, volumes[exchange]))
            
            if exchange in depths:
                self.depth_history[symbol][exchange].append((timestamp, depths[exchange]))
    
    def _calculate_price_features(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate price-based statistical features."""
        if not prices:
            return {'avg_price': 0.0, 'price_std': 0.0}
        
        price_values = list(prices.values())
        avg_price = statistics.mean(price_values)
        price_std = statistics.stdev(price_values) if len(price_values) > 1 else 0.0
        
        return {
            'avg_price': avg_price,
            'price_std': price_std
        }
    
    def _calculate_spread_features(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate spread-based features between exchanges."""
        spreads = {}
        
        # Calculate pairwise spreads in basis points
        if 'bybit' in prices and 'binance' in prices:
            spread_bps = abs(prices['bybit'] - prices['binance']) / prices['binance'] * 10000
            spreads['bybit_binance_spread'] = spread_bps
        else:
            spreads['bybit_binance_spread'] = 0.0
        
        if 'bybit' in prices and 'okx' in prices:
            spread_bps = abs(prices['bybit'] - prices['okx']) / prices['okx'] * 10000
            spreads['bybit_okx_spread'] = spread_bps
        else:
            spreads['bybit_okx_spread'] = 0.0
        
        if 'binance' in prices and 'okx' in prices:
            spread_bps = abs(prices['binance'] - prices['okx']) / prices['okx'] * 10000
            spreads['binance_okx_spread'] = spread_bps
        else:
            spreads['binance_okx_spread'] = 0.0
        
        # Calculate aggregate spread features
        spread_values = [v for v in spreads.values() if v > 0]
        spreads['max_spread'] = max(spread_values) if spread_values else 0.0
        spreads['spread_volatility'] = statistics.stdev(spread_values) if len(spread_values) > 1 else 0.0
        
        return spreads
    
    def _calculate_volume_features(self, prices: Dict[str, float], volumes: Dict[str, float]) -> Dict[str, float]:
        """Calculate volume-weighted features."""
        total_volume = sum(volumes.values())
        
        if total_volume == 0:
            return {
                'total_volume': 0.0,
                'vwap': 0.0,
                'volume_concentration': 0.0
            }
        
        # Volume-weighted average price
        vwap = 0.0
        total_value = 0.0
        for exchange in self.exchanges:
            if exchange in prices and exchange in volumes:
                total_value += prices[exchange] * volumes[exchange]
        
        vwap = total_value / total_volume if total_volume > 0 else 0.0
        
        # Volume concentration (Herfindahl-Hirschman Index)
        volume_shares = [vol / total_volume for vol in volumes.values()]
        volume_concentration = sum(share ** 2 for share in volume_shares)
        
        return {
            'total_volume': total_volume,
            'vwap': vwap,
            'volume_concentration': volume_concentration
        }
    
    def _calculate_arbitrage_features(self, prices: Dict[str, float], volumes: Dict[str, float], 
                                    symbol: str) -> Dict[str, Any]:
        """Calculate arbitrage opportunity features."""
        if len(prices) < 2:
            return {
                'best_spread': 0.0,
                'has_opportunity': False,
                'estimated_profit_bps': 0.0,
                'direction': 'none'
            }
        
        best_spread = 0.0
        best_opportunity = None
        
        # Check all exchange pairs for arbitrage opportunities
        exchanges = list(prices.keys())
        for i, buy_exchange in enumerate(exchanges):
            for j, sell_exchange in enumerate(exchanges):
                if i >= j:
                    continue
                
                buy_price = prices[buy_exchange]
                sell_price = prices[sell_exchange]
                
                # Calculate spread (sell high, buy low)
                if sell_price > buy_price:
                    spread_bps = (sell_price - buy_price) / buy_price * 10000
                    
                    # Account for trading fees
                    buy_fee = self.config['trading_fees'].get(buy_exchange, 0.001)
                    sell_fee = self.config['trading_fees'].get(sell_exchange, 0.001)
                    
                    # Estimated profit after fees
                    gross_profit_bps = spread_bps
                    fee_cost_bps = (buy_fee + sell_fee) * 10000
                    net_profit_bps = gross_profit_bps - fee_cost_bps
                    
                    if net_profit_bps > best_spread:
                        best_spread = net_profit_bps
                        
                        # Estimate minimum volume for arbitrage
                        min_volume = min(
                            volumes.get(buy_exchange, 0),
                            volumes.get(sell_exchange, 0)
                        )
                        
                        # Calculate confidence score
                        confidence = self._calculate_arbitrage_confidence(
                            buy_exchange, sell_exchange, spread_bps, min_volume, symbol
                        )
                        
                        best_opportunity = {
                            'buy_exchange': buy_exchange,
                            'sell_exchange': sell_exchange,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'spread_bps': spread_bps,
                            'net_profit_bps': net_profit_bps,
                            'min_volume': min_volume,
                            'confidence': confidence
                        }
        
        # Determine if there's a viable arbitrage opportunity
        has_opportunity = (best_spread >= self.config['min_arbitrage_bps'] and 
                          best_opportunity and 
                          best_opportunity['confidence'] >= self.config['arbitrage_confidence_threshold'])
        
        # Create arbitrage opportunity record
        if has_opportunity and best_opportunity:
            opportunity = ArbitrageOpportunity(
                timestamp=datetime.now(),
                symbol=symbol,
                buy_exchange=best_opportunity['buy_exchange'],
                sell_exchange=best_opportunity['sell_exchange'],
                buy_price=best_opportunity['buy_price'],
                sell_price=best_opportunity['sell_price'],
                spread_bps=best_opportunity['spread_bps'],
                estimated_profit_bps=best_opportunity['net_profit_bps'],
                min_volume=best_opportunity['min_volume'],
                confidence_score=best_opportunity['confidence'],
                duration_seconds=0.0  # Will be updated when opportunity closes
            )
            
            self.arbitrage_opportunities.append(opportunity)
            
            # Keep opportunities list manageable
            if len(self.arbitrage_opportunities) > self.max_opportunities:
                self.arbitrage_opportunities = self.arbitrage_opportunities[-self.max_opportunities:]
            
            direction = f"{best_opportunity['buy_exchange']}_to_{best_opportunity['sell_exchange']}"
        else:
            direction = 'none'
        
        return {
            'best_spread': best_spread,
            'has_opportunity': has_opportunity,
            'estimated_profit_bps': best_opportunity['net_profit_bps'] if best_opportunity else 0.0,
            'direction': direction
        }
    
    def _calculate_arbitrage_confidence(self, buy_exchange: str, sell_exchange: str, 
                                      spread_bps: float, min_volume: float, symbol: str) -> float:
        """Calculate confidence score for arbitrage opportunity."""
        confidence_factors = []
        
        # Spread size factor (larger spreads are more reliable)
        spread_factor = min(1.0, spread_bps / 100.0)  # Normalize to 100 bps
        confidence_factors.append(spread_factor)
        
        # Volume factor (higher volume = more confidence)
        volume_factor = min(1.0, min_volume / 1000.0)  # Normalize to 1000 units
        confidence_factors.append(volume_factor)
        
        # Historical spread stability
        historical_spreads = self._get_historical_spreads(buy_exchange, sell_exchange, symbol)
        if historical_spreads:
            spread_stability = 1.0 - (statistics.stdev(historical_spreads) / statistics.mean(historical_spreads))
            spread_stability = max(0.0, min(1.0, spread_stability))
            confidence_factors.append(spread_stability)
        
        # Exchange reliability factor
        exchange_reliability = {
            'bybit': 0.95,
            'binance': 0.98,
            'okx': 0.90
        }
        
        avg_reliability = (exchange_reliability.get(buy_exchange, 0.8) + 
                          exchange_reliability.get(sell_exchange, 0.8)) / 2
        confidence_factors.append(avg_reliability)
        
        # Calculate weighted average confidence
        weights = [0.3, 0.2, 0.3, 0.2]  # Spread, volume, stability, reliability
        confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
        
        return min(1.0, max(0.0, confidence))
    
    def _get_historical_spreads(self, exchange1: str, exchange2: str, symbol: str) -> List[float]:
        """Get historical spreads between two exchanges."""
        spreads = []
        
        history1 = self.price_history[symbol][exchange1]
        history2 = self.price_history[symbol][exchange2]
        
        if len(history1) < 10 or len(history2) < 10:
            return spreads
        
        # Get last 50 data points for both exchanges
        recent1 = list(history1)[-50:]
        recent2 = list(history2)[-50:]
        
        # Calculate spreads for overlapping timestamps
        for ts1, price1 in recent1:
            for ts2, price2 in recent2:
                if abs((ts1 - ts2).total_seconds()) < 60:  # Within 1 minute
                    spread_bps = abs(price1 - price2) / price2 * 10000
                    spreads.append(spread_bps)
                    break
        
        return spreads
    
    def _calculate_depth_features(self, depths: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate market depth and liquidity features."""
        features = {}
        
        total_bid_size = 0.0
        total_ask_size = 0.0
        
        for exchange in self.exchanges:
            if exchange in depths:
                depth = depths[exchange]
                
                # Depth ratio (bid/ask size ratio)
                if 'bid_size' in depth and 'ask_size' in depth:
                    bid_size = depth['bid_size']
                    ask_size = depth['ask_size']
                    
                    total_bid_size += bid_size
                    total_ask_size += ask_size
                    
                    if ask_size > 0:
                        depth_ratio = bid_size / ask_size
                        features[exchange] = depth_ratio
                    else:
                        features[exchange] = 1.0
                else:
                    features[exchange] = 1.0
        
        # Overall liquidity score
        total_liquidity = total_bid_size + total_ask_size
        if total_liquidity > 0:
            # Normalize liquidity score (higher is better)
            features['liquidity_score'] = min(1.0, total_liquidity / 10000.0)
        else:
            features['liquidity_score'] = 0.0
        
        return features
    
    def _calculate_correlation_features(self, symbol: str) -> Dict[str, float]:
        """Calculate price and volume correlations between exchanges."""
        price_corr = self._calculate_price_correlation(symbol)
        volume_corr = self._calculate_volume_correlation(symbol)
        
        return {
            'price_correlation': price_corr,
            'volume_correlation': volume_corr
        }
    
    def _calculate_price_correlation(self, symbol: str) -> float:
        """Calculate average price correlation between exchanges."""
        correlations = []
        
        exchanges = list(self.price_history[symbol].keys())
        if len(exchanges) < 2:
            return 0.0
        
        # Get price series for correlation calculation
        price_series = {}
        min_length = float('inf')
        
        for exchange in exchanges:
            history = list(self.price_history[symbol][exchange])
            if len(history) >= 20:  # Minimum data points for correlation
                prices = [price for _, price in history[-100:]]  # Last 100 points
                price_series[exchange] = prices
                min_length = min(min_length, len(prices))
        
        if len(price_series) < 2 or min_length < 10:
            return 0.0
        
        # Truncate all series to minimum length
        for exchange in price_series:
            price_series[exchange] = price_series[exchange][-min_length:]
        
        # Calculate pairwise correlations
        exchanges = list(price_series.keys())
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                try:
                    corr = np.corrcoef(price_series[exchanges[i]], price_series[exchanges[j]])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                except Exception:
                    continue
        
        return statistics.mean(correlations) if correlations else 0.0
    
    def _calculate_volume_correlation(self, symbol: str) -> float:
        """Calculate average volume correlation between exchanges."""
        correlations = []
        
        exchanges = list(self.volume_history[symbol].keys())
        if len(exchanges) < 2:
            return 0.0
        
        # Get volume series for correlation calculation
        volume_series = {}
        min_length = float('inf')
        
        for exchange in exchanges:
            history = list(self.volume_history[symbol][exchange])
            if len(history) >= 20:
                volumes = [volume for _, volume in history[-100:]]
                volume_series[exchange] = volumes
                min_length = min(min_length, len(volumes))
        
        if len(volume_series) < 2 or min_length < 10:
            return 0.0
        
        # Truncate all series to minimum length
        for exchange in volume_series:
            volume_series[exchange] = volume_series[exchange][-min_length:]
        
        # Calculate pairwise correlations
        exchanges = list(volume_series.keys())
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                try:
                    corr = np.corrcoef(volume_series[exchanges[i]], volume_series[exchanges[j]])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                except Exception:
                    continue
        
        return statistics.mean(correlations) if correlations else 0.0
    
    def _calculate_efficiency_features(self, symbol: str, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Calculate market efficiency and price leadership features."""
        # Price efficiency score (how quickly prices converge)
        efficiency_score = self._calculate_price_efficiency_score(symbol, current_prices)
        
        # Market leadership scores (which exchange leads price movements)
        leadership_scores = self._calculate_leadership_scores(symbol)
        
        return {
            'efficiency_score': efficiency_score,
            'leadership_scores': leadership_scores
        }
    
    def _calculate_price_efficiency_score(self, symbol: str, current_prices: Dict[str, float]) -> float:
        """Calculate how efficiently prices are aligned across exchanges."""
        if len(current_prices) < 2:
            return 0.5
        
        # Calculate price dispersion
        prices = list(current_prices.values())
        mean_price = statistics.mean(prices)
        price_std = statistics.stdev(prices) if len(prices) > 1 else 0.0
        
        # Efficiency score inversely related to price dispersion
        if mean_price > 0:
            cv = price_std / mean_price  # Coefficient of variation
            efficiency_score = 1.0 / (1.0 + cv * 100)  # Scale CV by 100
        else:
            efficiency_score = 0.5
        
        return min(1.0, max(0.0, efficiency_score))
    
    def _calculate_leadership_scores(self, symbol: str) -> Dict[str, float]:
        """Calculate which exchanges tend to lead price movements."""
        leadership_scores = {}
        
        if symbol not in self.price_changes:
            return {exchange: 0.33 for exchange in self.exchanges}  # Default equal weights
        
        # Analyze price change timing and magnitude
        for exchange in self.exchanges:
            if exchange not in self.price_changes[symbol]:
                leadership_scores[exchange] = 0.0
                continue
            
            changes = list(self.price_changes[symbol][exchange])
            if len(changes) < 10:
                leadership_scores[exchange] = 0.33
                continue
            
            # Calculate average magnitude of price changes
            recent_changes = [abs(change) for _, change in changes[-50:]]
            avg_change_magnitude = statistics.mean(recent_changes)
            
            # Calculate consistency of direction
            directional_changes = [change for _, change in changes[-50:]]
            positive_changes = sum(1 for change in directional_changes if change > 0)
            consistency = abs(positive_changes / len(directional_changes) - 0.5) * 2
            
            # Leadership score based on magnitude and consistency
            leadership_score = (avg_change_magnitude * 1000 + consistency) / 2
            leadership_scores[exchange] = min(1.0, leadership_score)
        
        # Normalize leadership scores to sum to 1
        total_score = sum(leadership_scores.values())
        if total_score > 0:
            leadership_scores = {exchange: score / total_score 
                               for exchange, score in leadership_scores.items()}
        else:
            leadership_scores = {exchange: 1.0 / len(self.exchanges) 
                               for exchange in self.exchanges}
        
        return leadership_scores
    
    def _cleanup_cache(self):
        """Clean up old cached features."""
        if len(self.feature_cache) > 10000:
            # Keep only the most recent 5000 features
            sorted_keys = sorted(self.feature_cache.keys())
            keys_to_remove = sorted_keys[:-5000]
            for key in keys_to_remove:
                del self.feature_cache[key]
    
    # Public interface methods
    def get_recent_features(self, symbol: str, hours: int = 1) -> List[CrossExchangeFeatures]:
        """Get recent cross-exchange features for a symbol."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_features = []
        for key, features in self.feature_cache.items():
            if (features.symbol == symbol and 
                features.timestamp > cutoff_time):
                recent_features.append(features)
        
        return sorted(recent_features, key=lambda x: x.timestamp)
    
    def get_arbitrage_opportunities(self, symbol: Optional[str] = None, 
                                  hours: int = 24) -> List[ArbitrageOpportunity]:
        """Get recent arbitrage opportunities."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        opportunities = [
            opp for opp in self.arbitrage_opportunities
            if opp.timestamp > cutoff_time and (symbol is None or opp.symbol == symbol)
        ]
        
        return sorted(opportunities, key=lambda x: x.timestamp, reverse=True)
    
    def get_market_efficiency_report(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market efficiency report."""
        recent_features = self.get_recent_features(symbol, hours=24)
        
        if not recent_features:
            return {'status': 'no_data', 'symbol': symbol}
        
        # Calculate efficiency statistics
        efficiency_scores = [f.price_efficiency_score for f in recent_features]
        avg_efficiency = statistics.mean(efficiency_scores)
        
        # Calculate spread statistics
        max_spreads = [f.max_spread for f in recent_features]
        avg_spread = statistics.mean(max_spreads)
        spread_volatility = statistics.stdev(max_spreads) if len(max_spreads) > 1 else 0.0
        
        # Count arbitrage opportunities
        arbitrage_count = sum(1 for f in recent_features if f.arbitrage_opportunity)
        arbitrage_rate = arbitrage_count / len(recent_features)
        
        # Get latest leadership scores
        latest_leadership = recent_features[-1].market_leadership_score if recent_features else {}
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'average_efficiency_score': avg_efficiency,
            'average_spread_bps': avg_spread,
            'spread_volatility': spread_volatility,
            'arbitrage_opportunity_rate': arbitrage_rate,
            'total_arbitrage_opportunities': arbitrage_count,
            'market_leadership': latest_leadership,
            'data_points': len(recent_features)
        }
    
    def get_exchange_comparison(self, symbol: str) -> Dict[str, Any]:
        """Get detailed comparison between exchanges."""
        recent_features = self.get_recent_features(symbol, hours=6)
        
        if not recent_features:
            return {'status': 'no_data', 'symbol': symbol}
        
        # Calculate average prices and volumes by exchange
        exchange_stats = {}
        for exchange in self.exchanges:
            prices = []
            volumes = []
            
            for features in recent_features:
                if exchange == 'bybit':
                    prices.append(features.bybit_price)
                    volumes.append(features.bybit_volume)
                elif exchange == 'binance':
                    prices.append(features.binance_price)
                    volumes.append(features.binance_volume)
                elif exchange == 'okx':
                    prices.append(features.okx_price)
                    volumes.append(features.okx_volume)
            
            # Filter out zero values
            prices = [p for p in prices if p > 0]
            volumes = [v for v in volumes if v > 0]
            
            if prices:
                exchange_stats[exchange] = {
                    'avg_price': statistics.mean(prices),
                    'price_std': statistics.stdev(prices) if len(prices) > 1 else 0.0,
                    'avg_volume': statistics.mean(volumes) if volumes else 0.0,
                    'data_points': len(prices)
                }
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'exchange_statistics': exchange_stats,
            'analysis_period_hours': 6
        }


# Example usage and testing
if __name__ == "__main__":
    import json
    from ..config.manager import ConfigurationManager
    
    def main():
        # Configure logging
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize calculator
        config_manager = ConfigurationManager()
        calculator = CrossExchangeFeatureCalculator(config_manager)
        
        # Create sample market data
        sample_data = {
            'bybit': {
                'price': 45000.0,
                'volume': 1000.0,
                'bid': 44950.0,
                'ask': 45050.0,
                'bid_size': 500.0,
                'ask_size': 600.0,
                'timestamp': datetime.now()
            },
            'binance': {
                'price': 45025.0,
                'volume': 1200.0,
                'bid': 44975.0,
                'ask': 45075.0,
                'bid_size': 400.0,
                'ask_size': 550.0,
                'timestamp': datetime.now()
            },
            'okx': {
                'price': 44980.0,
                'volume': 800.0,
                'bid': 44930.0,
                'ask': 45030.0,
                'bid_size': 350.0,
                'ask_size': 450.0,
                'timestamp': datetime.now()
            }
        }
        
        # Calculate features
        features = calculator.calculate_features(sample_data, 'BTCUSDT')
        
        if features:
            print(f"Cross-exchange features calculated:")
            print(f"  Average price: ${features.avg_price:.2f}")
            print(f"  Max spread: {features.max_spread:.2f} bps")
            print(f"  Arbitrage opportunity: {features.arbitrage_opportunity}")
            print(f"  Estimated profit: {features.estimated_profit_bps:.2f} bps")
            print(f"  Volume-weighted price: ${features.volume_weighted_price:.2f}")
            print(f"  Price efficiency score: {features.price_efficiency_score:.3f}")
            
            # Get market analysis
            efficiency_report = calculator.get_market_efficiency_report('BTCUSDT')
            print(f"\nMarket efficiency report: {json.dumps(efficiency_report, indent=2, default=str)}")
    
    # Run the example
    main()