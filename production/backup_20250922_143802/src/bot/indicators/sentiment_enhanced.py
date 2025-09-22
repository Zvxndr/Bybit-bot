"""
Sentiment-Enhanced Technical Indicators

Advanced technical analysis system that combines traditional technical indicators
with sentiment data to generate more accurate and timely trading signals.

Key Features:
- Traditional indicators enhanced with sentiment weighting
- News sentiment momentum indicators
- Fear & Greed enhanced trend signals
- Social sentiment oscillators
- Multi-timeframe sentiment-technical convergence
- Adaptive indicator parameters based on market sentiment

Enhanced Indicators:
- Sentiment-Weighted Moving Averages (SWMA)
- Sentiment-Enhanced RSI (SRSI)
- News-Momentum MACD (NMACD)
- Fear-Greed Bollinger Bands (FGBB)
- Sentiment Trend Strength (STS)
- Social Volume Price Trend (SVPT)
- Sentiment Divergence Indicator (SDI)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import asyncio
from collections import defaultdict, deque
import warnings
import statistics
from enum import Enum

from ..utils.logging import TradingLogger
from ..config.manager import ConfigurationManager


class SentimentStrength(Enum):
    """Sentiment strength levels."""
    EXTREMELY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    EXTREMELY_BULLISH = 2


class SignalStrength(Enum):
    """Signal strength levels."""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


@dataclass
class SentimentIndicatorResult:
    """Result from a sentiment-enhanced indicator."""
    timestamp: datetime
    symbol: str
    indicator_name: str
    value: float
    signal_strength: SignalStrength
    sentiment_influence: float  # -1 to 1
    traditional_value: float    # Traditional indicator value
    sentiment_adjustment: float # Adjustment made by sentiment
    confidence_score: float     # 0 to 1
    metadata: Dict[str, Any]


@dataclass
class TechnicalSignal:
    """Technical analysis signal with sentiment enhancement."""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: SignalStrength
    confidence: float
    price_level: float
    sentiment_score: float
    contributing_indicators: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    expected_duration: str  # 'short', 'medium', 'long'
    metadata: Dict[str, Any]


class SentimentEnhancedIndicators:
    """
    Comprehensive sentiment-enhanced technical indicators system.
    
    Combines traditional technical analysis with sentiment data to provide
    more accurate and timely trading signals that account for market psychology
    and news-driven price movements.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = TradingLogger("SentimentEnhancedIndicators")
        
        # Configuration
        self.config = {
            'sentiment_weight': config_manager.get('indicators.sentiment_weight', 0.3),
            'min_confidence_threshold': config_manager.get('indicators.min_confidence', 0.6),
            'lookback_periods': {
                'short': config_manager.get('indicators.periods.short', 12),
                'medium': config_manager.get('indicators.periods.medium', 26),
                'long': config_manager.get('indicators.periods.long', 50)
            },
            'sentiment_smoothing': config_manager.get('indicators.sentiment_smoothing', 5),
            'volatility_adjustment': config_manager.get('indicators.volatility_adjustment', True),
            'multi_timeframe': config_manager.get('indicators.multi_timeframe', True)
        }
        
        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.sentiment_data: Dict[str, pd.DataFrame] = {}
        self.indicator_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        
        # Signal tracking
        self.active_signals: List[TechnicalSignal] = []
        self.signal_history: deque = deque(maxlen=10000)
        
        # Indicator cache
        self.indicator_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Sentiment strength thresholds
        self.sentiment_thresholds = {
            'extremely_bearish': -0.8,
            'bearish': -0.3,
            'neutral_low': -0.1,
            'neutral_high': 0.1,
            'bullish': 0.3,
            'extremely_bullish': 0.8
        }
        
        # Initialize indicators
        self.indicators = {
            'swma': self._calculate_sentiment_weighted_ma,
            'srsi': self._calculate_sentiment_enhanced_rsi,
            'nmacd': self._calculate_news_momentum_macd,
            'fgbb': self._calculate_fear_greed_bollinger,
            'sts': self._calculate_sentiment_trend_strength,
            'svpt': self._calculate_social_volume_price_trend,
            'sdi': self._calculate_sentiment_divergence
        }
    
    def update_data(self, symbol: str, price_data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None):
        """Update price and sentiment data for analysis."""
        try:
            # Store price data
            if not price_data.empty:
                # Ensure required columns exist
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if all(col in price_data.columns for col in required_columns):
                    price_data = price_data.sort_values('timestamp')
                    self.price_data[symbol] = price_data.tail(1000)  # Keep last 1000 candles
                    self.logger.debug(f"Updated price data for {symbol}: {len(price_data)} candles")
                else:
                    self.logger.warning(f"Price data missing required columns for {symbol}")
                    return
            
            # Store sentiment data
            if sentiment_data is not None and not sentiment_data.empty:
                sentiment_data = sentiment_data.sort_values('timestamp')
                self.sentiment_data[symbol] = sentiment_data.tail(500)  # Keep last 500 sentiment points
                self.logger.debug(f"Updated sentiment data for {symbol}: {len(sentiment_data)} points")
            
        except Exception as e:
            self.logger.error(f"Error updating data for {symbol}: {e}")
    
    def calculate_all_indicators(self, symbol: str) -> Dict[str, SentimentIndicatorResult]:
        """Calculate all sentiment-enhanced indicators for a symbol."""
        if symbol not in self.price_data:
            self.logger.warning(f"No price data available for {symbol}")
            return {}
        
        results = {}
        
        for indicator_name, calculator_func in self.indicators.items():
            try:
                result = calculator_func(symbol)
                if result:
                    results[indicator_name] = result
                    
                    # Cache the result
                    self.indicator_cache[symbol][indicator_name] = result
                    
                    # Add to history
                    self.indicator_history[symbol][indicator_name].append(result)
                    
            except Exception as e:
                self.logger.error(f"Error calculating {indicator_name} for {symbol}: {e}")
        
        self.logger.debug(f"Calculated {len(results)} indicators for {symbol}")
        return results
    
    def _calculate_sentiment_weighted_ma(self, symbol: str) -> Optional[SentimentIndicatorResult]:
        """Calculate Sentiment-Weighted Moving Average."""
        price_data = self.price_data[symbol]
        if len(price_data) < self.config['lookback_periods']['medium']:
            return None
        
        # Traditional moving average
        period = self.config['lookback_periods']['medium']
        traditional_ma = price_data['close'].rolling(window=period).mean().iloc[-1]
        
        # Get sentiment influence
        sentiment_influence = self._get_current_sentiment(symbol)
        
        # Calculate sentiment weights
        sentiment_weights = self._calculate_sentiment_weights(symbol, period)
        
        if sentiment_weights is None:
            # Fallback to traditional MA
            sentiment_weighted_ma = traditional_ma
            sentiment_adjustment = 0.0
        else:
            # Apply sentiment weighting
            recent_prices = price_data['close'].tail(period)
            sentiment_weighted_ma = np.average(recent_prices, weights=sentiment_weights)
            sentiment_adjustment = sentiment_weighted_ma - traditional_ma
        
        # Determine signal strength
        current_price = price_data['close'].iloc[-1]
        price_distance = abs(current_price - sentiment_weighted_ma) / current_price
        
        if price_distance < 0.005:  # Within 0.5%
            signal_strength = SignalStrength.VERY_STRONG
        elif price_distance < 0.01:  # Within 1%
            signal_strength = SignalStrength.STRONG
        elif price_distance < 0.02:  # Within 2%
            signal_strength = SignalStrength.MODERATE
        elif price_distance < 0.03:  # Within 3%
            signal_strength = SignalStrength.WEAK
        else:
            signal_strength = SignalStrength.VERY_WEAK
        
        # Calculate confidence based on sentiment data availability and consistency
        confidence = self._calculate_indicator_confidence(symbol, sentiment_influence)
        
        return SentimentIndicatorResult(
            timestamp=datetime.now(),
            symbol=symbol,
            indicator_name='SWMA',
            value=sentiment_weighted_ma,
            signal_strength=signal_strength,
            sentiment_influence=sentiment_influence,
            traditional_value=traditional_ma,
            sentiment_adjustment=sentiment_adjustment,
            confidence_score=confidence,
            metadata={
                'period': period,
                'price_distance_pct': price_distance * 100,
                'current_price': current_price
            }
        )
    
    def _calculate_sentiment_enhanced_rsi(self, symbol: str) -> Optional[SentimentIndicatorResult]:
        """Calculate Sentiment-Enhanced RSI."""
        price_data = self.price_data[symbol]
        period = 14
        
        if len(price_data) < period + 1:
            return None
        
        # Traditional RSI calculation
        delta = price_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        traditional_rsi = 100 - (100 / (1 + rs))
        traditional_rsi_value = traditional_rsi.iloc[-1]
        
        # Get sentiment influence
        sentiment_influence = self._get_current_sentiment(symbol)
        
        # Adjust RSI based on sentiment
        sentiment_adjustment = sentiment_influence * 20  # Max ±20 points adjustment
        sentiment_enhanced_rsi = np.clip(traditional_rsi_value + sentiment_adjustment, 0, 100)
        
        # Determine signal strength based on RSI levels
        if sentiment_enhanced_rsi <= 20 or sentiment_enhanced_rsi >= 80:
            signal_strength = SignalStrength.VERY_STRONG
        elif sentiment_enhanced_rsi <= 30 or sentiment_enhanced_rsi >= 70:
            signal_strength = SignalStrength.STRONG
        elif sentiment_enhanced_rsi <= 40 or sentiment_enhanced_rsi >= 60:
            signal_strength = SignalStrength.MODERATE
        elif sentiment_enhanced_rsi <= 45 or sentiment_enhanced_rsi >= 55:
            signal_strength = SignalStrength.WEAK
        else:
            signal_strength = SignalStrength.VERY_WEAK
        
        confidence = self._calculate_indicator_confidence(symbol, sentiment_influence)
        
        return SentimentIndicatorResult(
            timestamp=datetime.now(),
            symbol=symbol,
            indicator_name='SRSI',
            value=sentiment_enhanced_rsi,
            signal_strength=signal_strength,
            sentiment_influence=sentiment_influence,
            traditional_value=traditional_rsi_value,
            sentiment_adjustment=sentiment_adjustment,
            confidence_score=confidence,
            metadata={
                'overbought_level': 70,
                'oversold_level': 30,
                'period': period
            }
        )
    
    def _calculate_news_momentum_macd(self, symbol: str) -> Optional[SentimentIndicatorResult]:
        """Calculate News-Momentum MACD."""
        price_data = self.price_data[symbol]
        short_period = self.config['lookback_periods']['short']
        long_period = self.config['lookback_periods']['medium']
        signal_period = 9
        
        if len(price_data) < long_period + signal_period:
            return None
        
        # Traditional MACD calculation
        ema_short = price_data['close'].ewm(span=short_period).mean()
        ema_long = price_data['close'].ewm(span=long_period).mean()
        traditional_macd = ema_short - ema_long
        macd_signal = traditional_macd.ewm(span=signal_period).mean()
        traditional_histogram = traditional_macd - macd_signal
        traditional_value = traditional_histogram.iloc[-1]
        
        # Get news momentum factor
        news_momentum = self._calculate_news_momentum(symbol)
        sentiment_influence = news_momentum * 0.5  # Scale to [-0.5, 0.5]
        
        # Adjust MACD based on news momentum
        news_enhanced_macd = traditional_macd * (1 + sentiment_influence)
        enhanced_signal = news_enhanced_macd.ewm(span=signal_period).mean()
        enhanced_histogram = news_enhanced_macd - enhanced_signal
        enhanced_value = enhanced_histogram.iloc[-1]
        
        sentiment_adjustment = enhanced_value - traditional_value
        
        # Determine signal strength
        histogram_magnitude = abs(enhanced_value)
        if histogram_magnitude > 100:
            signal_strength = SignalStrength.VERY_STRONG
        elif histogram_magnitude > 50:
            signal_strength = SignalStrength.STRONG
        elif histogram_magnitude > 20:
            signal_strength = SignalStrength.MODERATE
        elif histogram_magnitude > 10:
            signal_strength = SignalStrength.WEAK
        else:
            signal_strength = SignalStrength.VERY_WEAK
        
        confidence = self._calculate_indicator_confidence(symbol, sentiment_influence)
        
        return SentimentIndicatorResult(
            timestamp=datetime.now(),
            symbol=symbol,
            indicator_name='NMACD',
            value=enhanced_value,
            signal_strength=signal_strength,
            sentiment_influence=sentiment_influence,
            traditional_value=traditional_value,
            sentiment_adjustment=sentiment_adjustment,
            confidence_score=confidence,
            metadata={
                'short_period': short_period,
                'long_period': long_period,
                'signal_period': signal_period,
                'news_momentum': news_momentum
            }
        )
    
    def _calculate_fear_greed_bollinger(self, symbol: str) -> Optional[SentimentIndicatorResult]:
        """Calculate Fear & Greed enhanced Bollinger Bands."""
        price_data = self.price_data[symbol]
        period = 20
        std_multiplier = 2.0
        
        if len(price_data) < period:
            return None
        
        # Traditional Bollinger Bands
        sma = price_data['close'].rolling(window=period).mean()
        std = price_data['close'].rolling(window=period).std()
        
        traditional_upper = sma + (std * std_multiplier)
        traditional_lower = sma - (std * std_multiplier)
        traditional_width = traditional_upper - traditional_lower
        traditional_width_value = traditional_width.iloc[-1]
        
        # Get Fear & Greed influence
        fear_greed_score = self._get_fear_greed_adjustment(symbol)
        sentiment_influence = fear_greed_score
        
        # Adjust band width based on Fear & Greed
        # High fear (negative) = wider bands, High greed (positive) = narrower bands
        width_adjustment = 1 + (sentiment_influence * -0.5)  # Invert sentiment for width
        adjusted_std_multiplier = std_multiplier * width_adjustment
        
        enhanced_upper = sma + (std * adjusted_std_multiplier)
        enhanced_lower = sma - (std * adjusted_std_multiplier)
        enhanced_width = enhanced_upper - enhanced_lower
        enhanced_width_value = enhanced_width.iloc[-1]
        
        sentiment_adjustment = enhanced_width_value - traditional_width_value
        
        # Calculate position within bands
        current_price = price_data['close'].iloc[-1]
        current_sma = sma.iloc[-1]
        band_position = (current_price - current_sma) / (enhanced_width_value / 2)
        
        # Determine signal strength based on band position
        abs_position = abs(band_position)
        if abs_position > 0.8:
            signal_strength = SignalStrength.VERY_STRONG
        elif abs_position > 0.6:
            signal_strength = SignalStrength.STRONG
        elif abs_position > 0.4:
            signal_strength = SignalStrength.MODERATE
        elif abs_position > 0.2:
            signal_strength = SignalStrength.WEAK
        else:
            signal_strength = SignalStrength.VERY_WEAK
        
        confidence = self._calculate_indicator_confidence(symbol, sentiment_influence)
        
        return SentimentIndicatorResult(
            timestamp=datetime.now(),
            symbol=symbol,
            indicator_name='FGBB',
            value=enhanced_width_value,
            signal_strength=signal_strength,
            sentiment_influence=sentiment_influence,
            traditional_value=traditional_width_value,
            sentiment_adjustment=sentiment_adjustment,
            confidence_score=confidence,
            metadata={
                'period': period,
                'std_multiplier': adjusted_std_multiplier,
                'band_position': band_position,
                'fear_greed_score': fear_greed_score,
                'current_price': current_price,
                'upper_band': enhanced_upper.iloc[-1],
                'lower_band': enhanced_lower.iloc[-1]
            }
        )
    
    def _calculate_sentiment_trend_strength(self, symbol: str) -> Optional[SentimentIndicatorResult]:
        """Calculate Sentiment Trend Strength indicator."""
        price_data = self.price_data[symbol]
        period = self.config['lookback_periods']['medium']
        
        if len(price_data) < period:
            return None
        
        # Calculate price trend strength
        recent_prices = price_data['close'].tail(period)
        price_changes = recent_prices.pct_change().dropna()
        
        # Traditional trend strength (percentage of positive moves)
        positive_moves = (price_changes > 0).sum()
        traditional_trend_strength = positive_moves / len(price_changes)
        
        # Get sentiment trend
        sentiment_trend = self._calculate_sentiment_trend(symbol, period)
        sentiment_influence = sentiment_trend
        
        # Combine price trend with sentiment trend
        combined_weight = 0.7  # 70% price, 30% sentiment
        sentiment_enhanced_strength = (
            combined_weight * traditional_trend_strength + 
            (1 - combined_weight) * (sentiment_trend + 1) / 2  # Normalize sentiment to 0-1
        )
        
        sentiment_adjustment = sentiment_enhanced_strength - traditional_trend_strength
        
        # Determine signal strength
        if sentiment_enhanced_strength > 0.8:
            signal_strength = SignalStrength.VERY_STRONG
        elif sentiment_enhanced_strength > 0.65:
            signal_strength = SignalStrength.STRONG
        elif sentiment_enhanced_strength > 0.5:
            signal_strength = SignalStrength.MODERATE
        elif sentiment_enhanced_strength > 0.35:
            signal_strength = SignalStrength.WEAK
        else:
            signal_strength = SignalStrength.VERY_WEAK
        
        confidence = self._calculate_indicator_confidence(symbol, sentiment_influence)
        
        return SentimentIndicatorResult(
            timestamp=datetime.now(),
            symbol=symbol,
            indicator_name='STS',
            value=sentiment_enhanced_strength,
            signal_strength=signal_strength,
            sentiment_influence=sentiment_influence,
            traditional_value=traditional_trend_strength,
            sentiment_adjustment=sentiment_adjustment,
            confidence_score=confidence,
            metadata={
                'period': period,
                'positive_moves': positive_moves,
                'total_moves': len(price_changes),
                'sentiment_trend': sentiment_trend
            }
        )
    
    def _calculate_social_volume_price_trend(self, symbol: str) -> Optional[SentimentIndicatorResult]:
        """Calculate Social Volume Price Trend indicator."""
        price_data = self.price_data[symbol]
        period = 14
        
        if len(price_data) < period:
            return None
        
        # Traditional Volume Price Trend
        typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
        price_change_pct = typical_price.pct_change()
        volume_price_trend = (price_change_pct * price_data['volume']).rolling(window=period).sum()
        traditional_vpt = volume_price_trend.iloc[-1]
        
        # Get social volume factor
        social_volume = self._calculate_social_volume_factor(symbol)
        sentiment_influence = social_volume
        
        # Adjust VPT with social volume
        social_volume_multiplier = 1 + sentiment_influence  # Range [0, 2]
        enhanced_vpt = traditional_vpt * social_volume_multiplier
        
        sentiment_adjustment = enhanced_vpt - traditional_vpt
        
        # Normalize for signal strength calculation
        vpt_magnitude = abs(enhanced_vpt)
        avg_volume = price_data['volume'].tail(period).mean()
        normalized_vpt = vpt_magnitude / avg_volume if avg_volume > 0 else 0
        
        # Determine signal strength
        if normalized_vpt > 2.0:
            signal_strength = SignalStrength.VERY_STRONG
        elif normalized_vpt > 1.0:
            signal_strength = SignalStrength.STRONG
        elif normalized_vpt > 0.5:
            signal_strength = SignalStrength.MODERATE
        elif normalized_vpt > 0.2:
            signal_strength = SignalStrength.WEAK
        else:
            signal_strength = SignalStrength.VERY_WEAK
        
        confidence = self._calculate_indicator_confidence(symbol, sentiment_influence)
        
        return SentimentIndicatorResult(
            timestamp=datetime.now(),
            symbol=symbol,
            indicator_name='SVPT',
            value=enhanced_vpt,
            signal_strength=signal_strength,
            sentiment_influence=sentiment_influence,
            traditional_value=traditional_vpt,
            sentiment_adjustment=sentiment_adjustment,
            confidence_score=confidence,
            metadata={
                'period': period,
                'social_volume_factor': social_volume,
                'normalized_vpt': normalized_vpt,
                'avg_volume': avg_volume
            }
        )
    
    def _calculate_sentiment_divergence(self, symbol: str) -> Optional[SentimentIndicatorResult]:
        """Calculate Sentiment Divergence Indicator."""
        price_data = self.price_data[symbol]
        period = 20
        
        if len(price_data) < period:
            return None
        
        # Calculate price momentum
        price_momentum = price_data['close'].pct_change(period).iloc[-1]
        
        # Calculate sentiment momentum
        sentiment_momentum = self._calculate_sentiment_momentum(symbol, period)
        
        # Calculate divergence (when price and sentiment move in opposite directions)
        divergence = price_momentum - sentiment_momentum
        sentiment_influence = sentiment_momentum
        
        # Traditional value would be just price momentum
        traditional_value = price_momentum
        sentiment_adjustment = divergence
        
        # Signal strength based on divergence magnitude
        abs_divergence = abs(divergence)
        if abs_divergence > 0.15:  # 15% divergence
            signal_strength = SignalStrength.VERY_STRONG
        elif abs_divergence > 0.10:  # 10% divergence
            signal_strength = SignalStrength.STRONG
        elif abs_divergence > 0.05:  # 5% divergence
            signal_strength = SignalStrength.MODERATE
        elif abs_divergence > 0.02:  # 2% divergence
            signal_strength = SignalStrength.WEAK
        else:
            signal_strength = SignalStrength.VERY_WEAK
        
        confidence = self._calculate_indicator_confidence(symbol, sentiment_influence)
        
        return SentimentIndicatorResult(
            timestamp=datetime.now(),
            symbol=symbol,
            indicator_name='SDI',
            value=divergence,
            signal_strength=signal_strength,
            sentiment_influence=sentiment_influence,
            traditional_value=traditional_value,
            sentiment_adjustment=sentiment_adjustment,
            confidence_score=confidence,
            metadata={
                'period': period,
                'price_momentum': price_momentum,
                'sentiment_momentum': sentiment_momentum,
                'divergence_type': 'bearish' if divergence < 0 else 'bullish'
            }
        )
    
    # Helper methods for sentiment calculations
    def _get_current_sentiment(self, symbol: str) -> float:
        """Get current sentiment score for symbol."""
        if symbol not in self.sentiment_data or self.sentiment_data[symbol].empty:
            return 0.0
        
        sentiment_df = self.sentiment_data[symbol]
        
        # Get most recent sentiment data
        recent_sentiment = sentiment_df.tail(self.config['sentiment_smoothing'])
        
        # Calculate weighted average (more recent = higher weight)
        weights = np.arange(1, len(recent_sentiment) + 1)
        weighted_sentiment = np.average(recent_sentiment['sentiment_score'], weights=weights)
        
        # Normalize to [-1, 1] range
        return np.clip(weighted_sentiment, -1.0, 1.0)
    
    def _calculate_sentiment_weights(self, symbol: str, period: int) -> Optional[np.ndarray]:
        """Calculate sentiment-based weights for moving average."""
        if symbol not in self.sentiment_data or self.sentiment_data[symbol].empty:
            return None
        
        # Get recent sentiment scores
        sentiment_df = self.sentiment_data[symbol]
        if len(sentiment_df) < period:
            return None
        
        recent_sentiment = sentiment_df.tail(period)['sentiment_score'].values
        
        # Convert sentiment to weights (positive sentiment = higher weight)
        # Scale sentiment [-1,1] to weights [0.5, 1.5]
        weights = 1.0 + (recent_sentiment * 0.5)
        
        # Normalize weights
        weights = weights / np.sum(weights) * period
        
        return weights
    
    def _calculate_news_momentum(self, symbol: str) -> float:
        """Calculate momentum from news sentiment."""
        if symbol not in self.sentiment_data or self.sentiment_data[symbol].empty:
            return 0.0
        
        sentiment_df = self.sentiment_data[symbol]
        
        # Look at sentiment changes over recent periods
        if len(sentiment_df) < 5:
            return 0.0
        
        recent_sentiment = sentiment_df.tail(10)['sentiment_score']
        
        # Calculate momentum as rate of change in sentiment
        sentiment_momentum = recent_sentiment.diff().tail(5).mean()
        
        return np.clip(sentiment_momentum, -1.0, 1.0)
    
    def _get_fear_greed_adjustment(self, symbol: str) -> float:
        """Get Fear & Greed index adjustment factor."""
        if symbol not in self.sentiment_data or self.sentiment_data[symbol].empty:
            return 0.0
        
        sentiment_df = self.sentiment_data[symbol]
        
        # Look for fear_greed_index column
        if 'fear_greed_index' in sentiment_df.columns:
            recent_fg = sentiment_df['fear_greed_index'].tail(1).iloc[0]
            # Convert from 0-100 scale to -1 to 1 scale
            # 0-25: Extreme Fear (-1 to -0.5)
            # 25-45: Fear (-0.5 to -0.1)
            # 45-55: Neutral (-0.1 to 0.1)
            # 55-75: Greed (0.1 to 0.5)
            # 75-100: Extreme Greed (0.5 to 1)
            
            if recent_fg <= 25:
                return -1.0 + (recent_fg / 25) * 0.5
            elif recent_fg <= 45:
                return -0.5 + ((recent_fg - 25) / 20) * 0.4
            elif recent_fg <= 55:
                return -0.1 + ((recent_fg - 45) / 10) * 0.2
            elif recent_fg <= 75:
                return 0.1 + ((recent_fg - 55) / 20) * 0.4
            else:
                return 0.5 + ((recent_fg - 75) / 25) * 0.5
        
        return 0.0
    
    def _calculate_sentiment_trend(self, symbol: str, period: int) -> float:
        """Calculate sentiment trend over specified period."""
        if symbol not in self.sentiment_data or self.sentiment_data[symbol].empty:
            return 0.0
        
        sentiment_df = self.sentiment_data[symbol]
        
        if len(sentiment_df) < period:
            return 0.0
        
        recent_sentiment = sentiment_df.tail(period)['sentiment_score']
        
        # Calculate linear regression slope as trend
        x = np.arange(len(recent_sentiment))
        if len(x) > 1 and recent_sentiment.std() > 0:
            trend_slope = np.polyfit(x, recent_sentiment, 1)[0]
            return np.clip(trend_slope * period, -1.0, 1.0)  # Scale by period
        
        return 0.0
    
    def _calculate_social_volume_factor(self, symbol: str) -> float:
        """Calculate social volume factor from sentiment data."""
        if symbol not in self.sentiment_data or self.sentiment_data[symbol].empty:
            return 0.0
        
        sentiment_df = self.sentiment_data[symbol]
        
        # Look for volume-related sentiment metrics
        volume_columns = ['news_volume', 'social_volume', 'mention_count']
        
        for col in volume_columns:
            if col in sentiment_df.columns:
                recent_volume = sentiment_df[col].tail(5).mean()
                # Normalize and scale
                if recent_volume > 0:
                    return min(1.0, recent_volume / 100.0)  # Assume 100 is high volume
        
        # Fallback: use sentiment magnitude as proxy for social volume
        recent_sentiment = sentiment_df['sentiment_score'].tail(5)
        social_activity = recent_sentiment.abs().mean()
        
        return social_activity
    
    def _calculate_sentiment_momentum(self, symbol: str, period: int) -> float:
        """Calculate sentiment momentum over specified period."""
        if symbol not in self.sentiment_data or self.sentiment_data[symbol].empty:
            return 0.0
        
        sentiment_df = self.sentiment_data[symbol]
        
        if len(sentiment_df) < period:
            return 0.0
        
        # Calculate momentum as percentage change in sentiment
        current_sentiment = sentiment_df['sentiment_score'].tail(1).iloc[0]
        past_sentiment = sentiment_df['sentiment_score'].iloc[-period]
        
        if abs(past_sentiment) > 0.01:  # Avoid division by very small numbers
            momentum = (current_sentiment - past_sentiment) / abs(past_sentiment)
        else:
            momentum = current_sentiment - past_sentiment
        
        return np.clip(momentum, -1.0, 1.0)
    
    def _calculate_indicator_confidence(self, symbol: str, sentiment_influence: float) -> float:
        """Calculate confidence score for an indicator."""
        confidence_factors = []
        
        # Data availability factor
        price_data_quality = 1.0 if symbol in self.price_data else 0.0
        sentiment_data_quality = 0.8 if symbol in self.sentiment_data and not self.sentiment_data[symbol].empty else 0.2
        confidence_factors.append((price_data_quality + sentiment_data_quality) / 2)
        
        # Sentiment influence factor (moderate influence is more confident)
        sentiment_confidence = 1.0 - abs(sentiment_influence)  # Less extreme = more confident
        confidence_factors.append(sentiment_confidence)
        
        # Data recency factor
        if symbol in self.sentiment_data and not self.sentiment_data[symbol].empty:
            latest_sentiment_time = self.sentiment_data[symbol]['timestamp'].max()
            time_diff = (datetime.now() - latest_sentiment_time).total_seconds()
            recency_factor = max(0.0, 1.0 - time_diff / 3600)  # Decay over 1 hour
            confidence_factors.append(recency_factor)
        else:
            confidence_factors.append(0.5)
        
        # Calculate weighted average confidence
        return statistics.mean(confidence_factors)
    
    def generate_trading_signals(self, symbol: str) -> List[TechnicalSignal]:
        """Generate comprehensive trading signals from all indicators."""
        # Calculate all indicators
        indicators = self.calculate_all_indicators(symbol)
        
        if not indicators:
            return []
        
        signals = []
        
        # Analyze indicator consensus
        buy_signals = []
        sell_signals = []
        hold_signals = []
        
        for name, indicator in indicators.items():
            signal_type, strength = self._interpret_indicator_signal(name, indicator)
            
            if signal_type == 'buy':
                buy_signals.append((name, strength, indicator.confidence_score))
            elif signal_type == 'sell':
                sell_signals.append((name, strength, indicator.confidence_score))
            else:
                hold_signals.append((name, strength, indicator.confidence_score))
        
        # Generate consensus signals
        if buy_signals:
            signal = self._create_consensus_signal(symbol, 'buy', buy_signals, indicators)
            if signal:
                signals.append(signal)
        
        if sell_signals:
            signal = self._create_consensus_signal(symbol, 'sell', sell_signals, indicators)
            if signal:
                signals.append(signal)
        
        # Store signals
        for signal in signals:
            self.active_signals.append(signal)
            self.signal_history.append(signal)
        
        # Clean up old signals
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_signals = [s for s in self.active_signals if s.timestamp > cutoff_time]
        
        return signals
    
    def _interpret_indicator_signal(self, indicator_name: str, indicator: SentimentIndicatorResult) -> Tuple[str, SignalStrength]:
        """Interpret an indicator result into a trading signal."""
        current_price = self.price_data[indicator.symbol]['close'].iloc[-1]
        
        if indicator_name == 'SWMA':
            if current_price > indicator.value * 1.01:  # 1% above MA
                return 'buy', indicator.signal_strength
            elif current_price < indicator.value * 0.99:  # 1% below MA
                return 'sell', indicator.signal_strength
            else:
                return 'hold', indicator.signal_strength
        
        elif indicator_name == 'SRSI':
            if indicator.value <= 30:  # Oversold
                return 'buy', indicator.signal_strength
            elif indicator.value >= 70:  # Overbought
                return 'sell', indicator.signal_strength
            else:
                return 'hold', indicator.signal_strength
        
        elif indicator_name == 'NMACD':
            if indicator.value > 0:  # Bullish momentum
                return 'buy', indicator.signal_strength
            elif indicator.value < 0:  # Bearish momentum
                return 'sell', indicator.signal_strength
            else:
                return 'hold', indicator.signal_strength
        
        elif indicator_name == 'FGBB':
            # Check band position from metadata
            band_position = indicator.metadata.get('band_position', 0)
            if band_position < -0.5:  # Near lower band
                return 'buy', indicator.signal_strength
            elif band_position > 0.5:  # Near upper band
                return 'sell', indicator.signal_strength
            else:
                return 'hold', indicator.signal_strength
        
        elif indicator_name == 'STS':
            if indicator.value > 0.6:  # Strong uptrend
                return 'buy', indicator.signal_strength
            elif indicator.value < 0.4:  # Strong downtrend
                return 'sell', indicator.signal_strength
            else:
                return 'hold', indicator.signal_strength
        
        elif indicator_name == 'SVPT':
            if indicator.value > 0:  # Positive volume-price trend
                return 'buy', indicator.signal_strength
            elif indicator.value < 0:  # Negative volume-price trend
                return 'sell', indicator.signal_strength
            else:
                return 'hold', indicator.signal_strength
        
        elif indicator_name == 'SDI':
            # Divergence signals potential reversal
            if indicator.value > 0.05:  # Positive divergence
                return 'buy', indicator.signal_strength
            elif indicator.value < -0.05:  # Negative divergence
                return 'sell', indicator.signal_strength
            else:
                return 'hold', indicator.signal_strength
        
        return 'hold', SignalStrength.WEAK
    
    def _create_consensus_signal(self, symbol: str, signal_type: str, 
                               signal_list: List[Tuple[str, SignalStrength, float]], 
                               indicators: Dict[str, SentimentIndicatorResult]) -> Optional[TechnicalSignal]:
        """Create a consensus signal from multiple indicators."""
        if not signal_list:
            return None
        
        # Calculate weighted consensus
        total_weight = 0
        weighted_strength = 0
        total_confidence = 0
        contributing_indicators = []
        
        for indicator_name, strength, confidence in signal_list:
            weight = strength.value * confidence
            total_weight += weight
            weighted_strength += strength.value * weight
            total_confidence += confidence
            contributing_indicators.append(indicator_name)
        
        if total_weight == 0:
            return None
        
        # Calculate consensus metrics
        avg_strength_value = weighted_strength / total_weight
        avg_confidence = total_confidence / len(signal_list)
        
        # Convert to SignalStrength enum
        consensus_strength = SignalStrength(max(1, min(5, round(avg_strength_value))))
        
        # Only generate signal if confidence is above threshold
        if avg_confidence < self.config['min_confidence_threshold']:
            return None
        
        # Calculate overall sentiment score
        sentiment_scores = [ind.sentiment_influence for ind in indicators.values()]
        avg_sentiment = statistics.mean(sentiment_scores) if sentiment_scores else 0.0
        
        # Determine risk level
        if avg_confidence > 0.8 and consensus_strength.value >= 4:
            risk_level = 'low'
        elif avg_confidence > 0.6 and consensus_strength.value >= 3:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        # Determine expected duration based on indicator types
        if 'NMACD' in contributing_indicators or 'STS' in contributing_indicators:
            duration = 'medium'
        elif 'SRSI' in contributing_indicators:
            duration = 'short'
        else:
            duration = 'long'
        
        current_price = self.price_data[symbol]['close'].iloc[-1]
        
        return TechnicalSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=signal_type,
            strength=consensus_strength,
            confidence=avg_confidence,
            price_level=current_price,
            sentiment_score=avg_sentiment,
            contributing_indicators=contributing_indicators,
            risk_level=risk_level,
            expected_duration=duration,
            metadata={
                'num_indicators': len(signal_list),
                'consensus_weight': total_weight,
                'individual_signals': signal_list
            }
        )
    
    # Public interface methods
    def get_indicator_value(self, symbol: str, indicator_name: str) -> Optional[SentimentIndicatorResult]:
        """Get the latest value for a specific indicator."""
        if symbol in self.indicator_cache and indicator_name in self.indicator_cache[symbol]:
            return self.indicator_cache[symbol][indicator_name]
        return None
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[TechnicalSignal]:
        """Get currently active trading signals."""
        if symbol:
            return [s for s in self.active_signals if s.symbol == symbol]
        return self.active_signals.copy()
    
    def get_signal_history(self, symbol: str, hours: int = 24) -> List[TechnicalSignal]:
        """Get signal history for a symbol."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            s for s in self.signal_history 
            if s.symbol == symbol and s.timestamp > cutoff_time
        ]
    
    def get_indicator_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive indicator summary."""
        indicators = self.calculate_all_indicators(symbol)
        
        if not indicators:
            return {'status': 'no_data', 'symbol': symbol}
        
        # Calculate overall sentiment influence
        sentiment_influences = [ind.sentiment_influence for ind in indicators.values()]
        avg_sentiment_influence = statistics.mean(sentiment_influences)
        
        # Count signal types
        signal_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        for name, indicator in indicators.items():
            signal_type, _ = self._interpret_indicator_signal(name, indicator)
            signal_counts[signal_type] += 1
        
        # Calculate confidence distribution
        confidences = [ind.confidence_score for ind in indicators.values()]
        avg_confidence = statistics.mean(confidences)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'total_indicators': len(indicators),
            'average_sentiment_influence': avg_sentiment_influence,
            'average_confidence': avg_confidence,
            'signal_distribution': signal_counts,
            'indicators': {name: {
                'value': ind.value,
                'signal_strength': ind.signal_strength.name,
                'confidence': ind.confidence_score,
                'sentiment_influence': ind.sentiment_influence
            } for name, ind in indicators.items()}
        }


# Example usage and testing
if __name__ == "__main__":
    import json
    from datetime import timedelta
    from ..config.manager import ConfigurationManager
    
    def main():
        # Configure logging
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize indicators system
        config_manager = ConfigurationManager()
        indicators = SentimentEnhancedIndicators(config_manager)
        
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        price_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(45000, 1000, 100),
            'high': np.random.normal(45200, 1000, 100),
            'low': np.random.normal(44800, 1000, 100),
            'close': np.random.normal(45000, 1000, 100),
            'volume': np.random.exponential(1000, 100)
        })
        
        # Create sample sentiment data
        sentiment_data = pd.DataFrame({
            'timestamp': dates[::2],  # Every 2 hours
            'sentiment_score': np.random.normal(0, 0.3, 50),
            'fear_greed_index': np.random.uniform(20, 80, 50),
            'news_volume': np.random.exponential(10, 50)
        })
        
        # Update data
        indicators.update_data('BTCUSDT', price_data, sentiment_data)
        
        # Calculate all indicators
        results = indicators.calculate_all_indicators('BTCUSDT')
        
        print("Sentiment-Enhanced Indicators Results:")
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Value: {result.value:.4f}")
            print(f"  Signal Strength: {result.signal_strength.name}")
            print(f"  Sentiment Influence: {result.sentiment_influence:.3f}")
            print(f"  Confidence: {result.confidence_score:.3f}")
        
        # Generate trading signals
        signals = indicators.generate_trading_signals('BTCUSDT')
        
        print(f"\nGenerated {len(signals)} trading signals:")
        for signal in signals:
            print(f"  {signal.signal_type.upper()}: {signal.strength.name} "
                  f"(Confidence: {signal.confidence:.3f}, Risk: {signal.risk_level})")
        
        # Get summary
        summary = indicators.get_indicator_summary('BTCUSDT')
        print(f"\nIndicator Summary: {json.dumps(summary, indent=2, default=str)}")
    
    # Run the example
    main()