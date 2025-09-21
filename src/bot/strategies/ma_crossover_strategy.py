"""
Moving Average Crossover Strategy - Example Implementation

This module provides a concrete implementation of the strategy framework
using a classic moving average crossover approach with risk management integration.

Strategy Logic:
- Buy signal when fast MA crosses above slow MA
- Sell signal when fast MA crosses below slow MA
- Risk management through position sizing and stop losses
- Integration with unified risk management system
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from decimal import Decimal

from .strategy_framework import (
    BaseStrategy, 
    TradingSignal, 
    SignalType, 
    SignalStrength,
    StrategyConfig
)
from ..core.trading_engine import TradingEngine
from ..risk_management import UnifiedRiskManager
from ..core.configuration_manager import ConfigurationManager


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy Implementation.
    
    This strategy uses two moving averages to generate trading signals:
    - Fast MA (shorter period) and Slow MA (longer period)
    - Buy when fast MA crosses above slow MA
    - Sell when fast MA crosses below slow MA
    - Includes momentum confirmation and volatility filtering
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        trading_engine: TradingEngine,
        risk_manager: UnifiedRiskManager,
        config_manager: ConfigurationManager = None,
        fast_period: int = 20,
        slow_period: int = 50,
        volume_threshold: float = 1.2,
        momentum_period: int = 14
    ):
        super().__init__(config, trading_engine, risk_manager, config_manager)
        
        # Strategy parameters
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_threshold = volume_threshold  # Volume multiplier for confirmation
        self.momentum_period = momentum_period
        
        # Internal state
        self.last_signals: Dict[str, TradingSignal] = {}
        self.ma_history: Dict[str, pd.DataFrame] = {}
        
        self.logger.info(f"MA Crossover Strategy initialized: fast={fast_period}, slow={slow_period}")
    
    async def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals based on moving average crossover.
        
        Args:
            symbol: Trading symbol
            data: OHLCV market data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        try:
            if len(data) < self.slow_period + 10:  # Need enough data
                return signals
            
            # Calculate moving averages
            data['fast_ma'] = data['close'].rolling(window=self.fast_period).mean()
            data['slow_ma'] = data['close'].rolling(window=self.slow_period).mean()
            
            # Calculate additional indicators
            data['volume_ma'] = data['volume'].rolling(window=20).mean()
            data['rsi'] = self._calculate_rsi(data['close'], self.momentum_period)
            data['atr'] = self._calculate_atr(data, 14)
            
            # Get latest values
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            
            current_price = Decimal(str(latest['close']))
            fast_ma_current = latest['fast_ma']
            slow_ma_current = latest['slow_ma']
            fast_ma_previous = previous['fast_ma']
            slow_ma_previous = previous['slow_ma']
            
            # Check for crossover
            bullish_crossover = (fast_ma_previous <= slow_ma_previous and 
                               fast_ma_current > slow_ma_current)
            
            bearish_crossover = (fast_ma_previous >= slow_ma_previous and 
                               fast_ma_current < slow_ma_current)
            
            # Generate buy signal
            if bullish_crossover:
                signal_strength = self._determine_signal_strength(data, latest, "bullish")
                confidence = self._calculate_confidence(data, latest, "bullish")
                
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=signal_strength,
                    price=current_price,
                    timestamp=pd.Timestamp.now(),
                    strategy_id=self.config.strategy_id,
                    confidence=confidence,
                    metadata={
                        "fast_ma": fast_ma_current,
                        "slow_ma": slow_ma_current,
                        "rsi": latest['rsi'],
                        "volume_ratio": latest['volume'] / latest['volume_ma'],
                        "atr": latest['atr']
                    }
                )
                
                signals.append(signal)
                self.last_signals[symbol] = signal
                self.logger.info(f"Generated BUY signal for {symbol}: strength={signal_strength.value}, confidence={confidence:.2f}")
            
            # Generate sell signal
            elif bearish_crossover:
                signal_strength = self._determine_signal_strength(data, latest, "bearish")
                confidence = self._calculate_confidence(data, latest, "bearish")
                
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=signal_strength,
                    price=current_price,
                    timestamp=pd.Timestamp.now(),
                    strategy_id=self.config.strategy_id,
                    confidence=confidence,
                    metadata={
                        "fast_ma": fast_ma_current,
                        "slow_ma": slow_ma_current,
                        "rsi": latest['rsi'],
                        "volume_ratio": latest['volume'] / latest['volume_ma'],
                        "atr": latest['atr']
                    }
                )
                
                signals.append(signal)
                self.last_signals[symbol] = signal
                self.logger.info(f"Generated SELL signal for {symbol}: strength={signal_strength.value}, confidence={confidence:.2f}")
            
            # Store MA history for analysis
            self.ma_history[symbol] = data[['close', 'fast_ma', 'slow_ma', 'rsi', 'atr']].tail(100)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return signals
    
    async def should_enter_position(self, signal: TradingSignal) -> bool:
        """
        Determine if strategy should enter position based on additional filters.
        
        Args:
            signal: Trading signal
            
        Returns:
            bool: True if should enter position
        """
        try:
            # Check signal confidence threshold
            if signal.confidence < 0.6:  # 60% minimum confidence
                self.logger.info(f"Signal confidence too low: {signal.confidence:.2f}")
                return False
            
            # Check RSI levels to avoid overbought/oversold extremes
            rsi = signal.metadata.get("rsi", 50)
            
            if signal.signal_type == SignalType.BUY:
                if rsi > 70:  # Overbought
                    self.logger.info(f"RSI too high for buy signal: {rsi:.2f}")
                    return False
            
            elif signal.signal_type == SignalType.SELL:
                if rsi < 30:  # Oversold
                    self.logger.info(f"RSI too low for sell signal: {rsi:.2f}")
                    return False
            
            # Check volume confirmation
            volume_ratio = signal.metadata.get("volume_ratio", 1.0)
            if volume_ratio < self.volume_threshold:
                self.logger.info(f"Insufficient volume confirmation: {volume_ratio:.2f}")
                return False
            
            # Check if we're not switching sides too quickly (whipsaw protection)
            last_signal = self.last_signals.get(signal.symbol)
            if last_signal and last_signal.timestamp > pd.Timestamp.now() - pd.Timedelta(hours=4):
                if ((signal.signal_type == SignalType.BUY and last_signal.signal_type == SignalType.SELL) or
                    (signal.signal_type == SignalType.SELL and last_signal.signal_type == SignalType.BUY)):
                    self.logger.info("Whipsaw protection: too recent opposite signal")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in should_enter_position: {e}")
            return False
    
    async def should_exit_position(self, symbol: str, position: Dict[str, Any]) -> bool:
        """
        Determine if strategy should exit existing position.
        
        Args:
            symbol: Trading symbol
            position: Current position data
            
        Returns:
            bool: True if should exit position
        """
        try:
            # Get current market data
            data = await self._get_market_data(symbol)
            if data is None or len(data) < 10:
                return False
            
            # Calculate current indicators
            data['fast_ma'] = data['close'].rolling(window=self.fast_period).mean()
            data['slow_ma'] = data['close'].rolling(window=self.slow_period).mean()
            data['rsi'] = self._calculate_rsi(data['close'], self.momentum_period)
            
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            
            fast_ma_current = latest['fast_ma']
            slow_ma_current = latest['slow_ma']
            fast_ma_previous = previous['fast_ma']
            slow_ma_previous = previous['slow_ma']
            current_rsi = latest['rsi']
            
            # Check for opposite crossover
            if position["side"] == "BUY":
                # Exit long position on bearish crossover
                bearish_crossover = (fast_ma_previous >= slow_ma_previous and 
                                   fast_ma_current < slow_ma_current)
                
                # Also exit if RSI becomes extremely overbought
                rsi_exit = current_rsi > 80
                
                if bearish_crossover or rsi_exit:
                    reason = "bearish crossover" if bearish_crossover else "RSI overbought"
                    self.logger.info(f"Exit signal for long position in {symbol}: {reason}")
                    return True
            
            elif position["side"] == "SELL":
                # Exit short position on bullish crossover
                bullish_crossover = (fast_ma_previous <= slow_ma_previous and 
                                   fast_ma_current > slow_ma_current)
                
                # Also exit if RSI becomes extremely oversold
                rsi_exit = current_rsi < 20
                
                if bullish_crossover or rsi_exit:
                    reason = "bullish crossover" if bullish_crossover else "RSI oversold"
                    self.logger.info(f"Exit signal for short position in {symbol}: {reason}")
                    return True
            
            # Time-based exit (optional - hold for maximum time)
            entry_time = position.get("entry_time")
            if entry_time and pd.Timestamp.now() - entry_time > pd.Timedelta(days=7):
                self.logger.info(f"Time-based exit for {symbol}: held for over 7 days")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in should_exit_position for {symbol}: {e}")
            return False
    
    def _determine_signal_strength(self, data: pd.DataFrame, latest: pd.Series, direction: str) -> SignalStrength:
        """Determine signal strength based on multiple factors."""
        try:
            score = 0
            
            # MA separation (stronger signal if MAs are further apart)
            ma_separation = abs(latest['fast_ma'] - latest['slow_ma']) / latest['close']
            if ma_separation > 0.02:  # 2%
                score += 2
            elif ma_separation > 0.01:  # 1%
                score += 1
            
            # Volume confirmation
            volume_ratio = latest['volume'] / latest['volume_ma']
            if volume_ratio > 2.0:
                score += 2
            elif volume_ratio > self.volume_threshold:
                score += 1
            
            # RSI momentum alignment
            rsi = latest['rsi']
            if direction == "bullish":
                if 40 < rsi < 70:  # Good momentum, not overbought
                    score += 1
                elif rsi > 50:  # Above midline
                    score += 1
            else:  # bearish
                if 30 < rsi < 60:  # Good momentum, not oversold
                    score += 1
                elif rsi < 50:  # Below midline
                    score += 1
            
            # Trend consistency (check if price is trending in signal direction)
            price_trend = data['close'].iloc[-5:].pct_change().mean()
            if direction == "bullish" and price_trend > 0:
                score += 1
            elif direction == "bearish" and price_trend < 0:
                score += 1
            
            # Map score to strength
            if score >= 5:
                return SignalStrength.VERY_STRONG
            elif score >= 3:
                return SignalStrength.STRONG
            elif score >= 2:
                return SignalStrength.MODERATE
            else:
                return SignalStrength.WEAK
                
        except Exception as e:
            self.logger.error(f"Error determining signal strength: {e}")
            return SignalStrength.WEAK
    
    def _calculate_confidence(self, data: pd.DataFrame, latest: pd.Series, direction: str) -> float:
        """Calculate signal confidence based on technical indicators."""
        try:
            confidence_factors = []
            
            # MA separation confidence
            ma_separation = abs(latest['fast_ma'] - latest['slow_ma']) / latest['close']
            separation_confidence = min(ma_separation / 0.03, 1.0)  # Max at 3% separation
            confidence_factors.append(separation_confidence)
            
            # Volume confidence
            volume_ratio = latest['volume'] / latest['volume_ma']
            volume_confidence = min((volume_ratio - 1.0) / 2.0, 1.0)  # Max at 3x average volume
            confidence_factors.append(max(volume_confidence, 0))
            
            # Trend consistency confidence
            recent_returns = data['close'].pct_change().iloc[-10:]
            trend_consistency = len([r for r in recent_returns if 
                                   (r > 0 and direction == "bullish") or 
                                   (r < 0 and direction == "bearish")]) / len(recent_returns)
            confidence_factors.append(trend_consistency)
            
            # RSI alignment confidence
            rsi = latest['rsi']
            if direction == "bullish":
                rsi_confidence = (rsi - 30) / 40 if rsi > 30 else 0  # Better if RSI > 30 but not too high
                rsi_confidence = min(rsi_confidence, 1.0)
            else:
                rsi_confidence = (70 - rsi) / 40 if rsi < 70 else 0  # Better if RSI < 70 but not too low
                rsi_confidence = min(rsi_confidence, 1.0)
            
            confidence_factors.append(max(rsi_confidence, 0))
            
            # Average all confidence factors
            overall_confidence = sum(confidence_factors) / len(confidence_factors)
            
            return min(max(overall_confidence, 0.1), 0.95)  # Clamp between 10% and 95%
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return pd.Series([0.01] * len(data), index=data.index)
    
    def get_strategy_details(self) -> Dict[str, Any]:
        """Get detailed strategy information."""
        base_status = self.get_status()
        
        # Add MA-specific details
        ma_details = {
            "strategy_type": "Moving Average Crossover",
            "parameters": {
                "fast_period": self.fast_period,
                "slow_period": self.slow_period,
                "volume_threshold": self.volume_threshold,
                "momentum_period": self.momentum_period
            },
            "recent_signals": {
                symbol: {
                    "type": signal.signal_type.value,
                    "strength": signal.strength.value,
                    "confidence": signal.confidence,
                    "timestamp": signal.timestamp.isoformat()
                }
                for symbol, signal in self.last_signals.items()
            },
            "ma_analysis": {
                symbol: {
                    "current_fast_ma": float(df['fast_ma'].iloc[-1]) if not df.empty else None,
                    "current_slow_ma": float(df['slow_ma'].iloc[-1]) if not df.empty else None,
                    "current_rsi": float(df['rsi'].iloc[-1]) if not df.empty else None,
                    "ma_separation_pct": float(abs(df['fast_ma'].iloc[-1] - df['slow_ma'].iloc[-1]) / df['close'].iloc[-1] * 100) if not df.empty else None
                }
                for symbol, df in self.ma_history.items()
                if not df.empty
            }
        }
        
        # Merge with base status
        base_status.update(ma_details)
        return base_status


# Example usage and configuration
def create_btc_ma_strategy(trading_engine: TradingEngine, risk_manager: UnifiedRiskManager) -> MovingAverageCrossoverStrategy:
    """Create a BTC-focused MA crossover strategy."""
    
    config = StrategyConfig(
        strategy_id="btc_ma_crossover_001",
        name="BTC Moving Average Crossover",
        symbols=["BTCUSDT"],
        timeframe="1h",
        max_positions=1,
        position_size_pct=0.02,  # 2% of portfolio per trade
        risk_reward_ratio=2.0,
        stop_loss_pct=0.03,      # 3% stop loss
        take_profit_pct=0.06,    # 6% take profit
        parameters={
            "fast_period": 20,
            "slow_period": 50,
            "volume_threshold": 1.5,
            "momentum_period": 14
        }
    )
    
    return MovingAverageCrossoverStrategy(
        config=config,
        trading_engine=trading_engine,
        risk_manager=risk_manager,
        fast_period=20,
        slow_period=50,
        volume_threshold=1.5,
        momentum_period=14
    )


def create_eth_ma_strategy(trading_engine: TradingEngine, risk_manager: UnifiedRiskManager) -> MovingAverageCrossoverStrategy:
    """Create an ETH-focused MA crossover strategy."""
    
    config = StrategyConfig(
        strategy_id="eth_ma_crossover_001",
        name="ETH Moving Average Crossover",
        symbols=["ETHUSDT"],
        timeframe="4h",
        max_positions=1,
        position_size_pct=0.015,  # 1.5% of portfolio per trade
        risk_reward_ratio=2.5,
        stop_loss_pct=0.025,      # 2.5% stop loss
        take_profit_pct=0.0625,   # 6.25% take profit
        parameters={
            "fast_period": 12,
            "slow_period": 26,
            "volume_threshold": 1.3,
            "momentum_period": 14
        }
    )
    
    return MovingAverageCrossoverStrategy(
        config=config,
        trading_engine=trading_engine,
        risk_manager=risk_manager,
        fast_period=12,
        slow_period=26,
        volume_threshold=1.3,
        momentum_period=14
    )