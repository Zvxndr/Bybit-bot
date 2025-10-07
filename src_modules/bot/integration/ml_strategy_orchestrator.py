"""
ML Strategy Orchestrator - Unified Trading Strategy Coordination

This module orchestrates the combination of ML predictions with traditional 
trading strategies, providing a unified decision-making system that balances
ML insights with proven algorithmic approaches.

Key Features:
- Combines ML predictions with traditional strategies (RSI, MACD, etc.)
- Dynamic strategy weighting based on market conditions
- Regime-aware strategy selection and parameter adjustment
- Risk-adjusted position sizing using both ML and traditional signals
- Strategy performance monitoring and adaptive rebalancing
- Fallback mechanisms when ML predictions are unreliable
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass
import json

# Import ML components
from .ml_feature_pipeline import MLFeatures, MLPrediction, MLSignalType, MLConfidenceLevel
from .ml_model_manager import MLModelManager, EnsemblePrediction

# Import traditional strategy components (would be actual imports in practice)
try:
    from ..strategies import (
        RSIStrategy, MACDStrategy, BollingerBandsStrategy, 
        MomentumStrategy, MeanReversionStrategy, BreakoutStrategy
    )
    TRADITIONAL_STRATEGIES_AVAILABLE = True
except ImportError:
    logger.warning("Traditional strategy modules not available")
    TRADITIONAL_STRATEGIES_AVAILABLE = False

# Import risk management
try:
    from ..risk.core.unified_risk_manager import UnifiedRiskManager
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("Unified risk manager not available")
    RISK_MANAGER_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class StrategyType(Enum):
    """Types of trading strategies"""
    ML_ONLY = "ml_only"
    TRADITIONAL_ONLY = "traditional_only"
    ML_TRADITIONAL_COMBINED = "ml_traditional_combined"
    ML_ENHANCED_TRADITIONAL = "ml_enhanced_traditional"
    REGIME_ADAPTIVE = "regime_adaptive"

class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNCERTAIN = "uncertain"

class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class TraditionalSignal:
    """Traditional strategy signal"""
    strategy_name: str
    signal_type: MLSignalType  # Reuse ML signal types for consistency
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    supporting_indicators: Dict[str, float]
    timestamp: datetime

@dataclass
class CombinedSignal:
    """Combined ML and traditional signal"""
    ml_signal: MLPrediction
    traditional_signals: List[TraditionalSignal]
    final_signal: MLSignalType
    final_confidence: float
    strategy_weights: Dict[str, float]
    position_size: float  # Recommended position size (0.0 to 1.0)
    risk_metrics: Dict[str, float]
    market_regime: MarketRegime
    timestamp: datetime

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_signals: int
    correct_signals: int
    accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    recent_performance: List[float]  # Last N outcomes
    last_updated: datetime

# ============================================================================
# ML STRATEGY ORCHESTRATOR
# ============================================================================

class MLStrategyOrchestrator:
    """
    Unified ML Strategy Orchestrator
    
    Combines ML predictions with traditional strategies for optimal trading decisions
    """
    
    def __init__(self, ml_model_manager: MLModelManager, config: Dict[str, Any] = None):
        self.ml_model_manager = ml_model_manager
        self.config = config or self._get_default_config()
        
        # Traditional strategies
        self.traditional_strategies: Dict[str, Any] = {}
        
        # Risk manager
        self.risk_manager = None
        
        # Strategy performance tracking
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.signal_history: List[CombinedSignal] = []
        
        # Market regime detection
        self.current_regime = MarketRegime.UNCERTAIN
        self.regime_confidence = 0.0
        
        # Strategy weights (dynamic)
        self.strategy_weights = self.config['strategy_weights'].copy()
        
        # Initialize components
        asyncio.create_task(self._initialize_components())
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ML Strategy Orchestrator"""
        return {
            'strategy_weights': {
                'ml_ensemble': 0.4,
                'rsi': 0.15,
                'macd': 0.15,
                'bollinger': 0.1,
                'momentum': 0.1,
                'mean_reversion': 0.1
            },
            'regime_adaptive_weights': {
                MarketRegime.TRENDING_UP: {
                    'ml_ensemble': 0.35,
                    'momentum': 0.25,
                    'breakout': 0.2,
                    'rsi': 0.1,
                    'macd': 0.1
                },
                MarketRegime.TRENDING_DOWN: {
                    'ml_ensemble': 0.35,
                    'mean_reversion': 0.25,
                    'rsi': 0.2,
                    'bollinger': 0.15,
                    'momentum': 0.05
                },
                MarketRegime.SIDEWAYS: {
                    'ml_ensemble': 0.3,
                    'mean_reversion': 0.3,
                    'bollinger': 0.2,
                    'rsi': 0.2
                },
                MarketRegime.HIGH_VOLATILITY: {
                    'ml_ensemble': 0.5,  # ML better in volatile conditions
                    'breakout': 0.2,
                    'momentum': 0.15,
                    'bollinger': 0.15
                }
            },
            'signal_thresholds': {
                'min_ml_confidence': 0.6,
                'min_traditional_confidence': 0.5,
                'consensus_threshold': 0.7,  # Required agreement for strong signals
                'position_size_multiplier': 1.0
            },
            'risk_management': {
                'max_position_size': 0.1,  # 10% max position
                'stop_loss_multiplier': 2.0,
                'take_profit_multiplier': 3.0,
                'correlation_adjustment': True
            },
            'performance_tracking': {
                'lookback_window': 100,
                'rebalance_frequency': 20,
                'performance_decay': 0.95  # Exponential decay for historical performance
            }
        }
    
    async def _initialize_components(self):
        """Initialize all components"""
        logger.info("Initializing ML Strategy Orchestrator...")
        
        try:
            # Initialize traditional strategies
            await self._initialize_traditional_strategies()
            
            # Initialize risk manager
            await self._initialize_risk_manager()
            
            # Initialize performance tracking
            self._initialize_performance_tracking()
            
            logger.info("ML Strategy Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML Strategy Orchestrator: {e}")
    
    async def _initialize_traditional_strategies(self):
        """Initialize traditional trading strategies"""
        if not TRADITIONAL_STRATEGIES_AVAILABLE:
            logger.warning("Traditional strategies not available, using mock implementations")
            self._initialize_mock_strategies()
            return
        
        try:
            # RSI Strategy
            self.traditional_strategies['rsi'] = RSIStrategy(
                period=14,
                oversold_threshold=30,
                overbought_threshold=70
            )
            
            # MACD Strategy  
            self.traditional_strategies['macd'] = MACDStrategy(
                fast_period=12,
                slow_period=26,
                signal_period=9
            )
            
            # Bollinger Bands Strategy
            self.traditional_strategies['bollinger'] = BollingerBandsStrategy(
                period=20,
                std_dev=2.0
            )
            
            # Momentum Strategy
            self.traditional_strategies['momentum'] = MomentumStrategy(
                lookback_period=14
            )
            
            # Mean Reversion Strategy
            self.traditional_strategies['mean_reversion'] = MeanReversionStrategy(
                lookback_period=20,
                threshold=2.0
            )
            
            # Breakout Strategy
            self.traditional_strategies['breakout'] = BreakoutStrategy(
                lookback_period=20,
                breakout_threshold=1.5
            )
            
            logger.info(f"Initialized {len(self.traditional_strategies)} traditional strategies")
            
        except Exception as e:
            logger.error(f"Error initializing traditional strategies: {e}")
            self._initialize_mock_strategies()
    
    def _initialize_mock_strategies(self):
        """Initialize mock traditional strategies for testing"""
        # Mock implementations that return random signals for testing
        class MockStrategy:
            def __init__(self, strategy_name: str):
                self.strategy_name = strategy_name
                self.np_random = np.random.RandomState(42)  # Deterministic for testing
            
            def generate_signal(self, data: Dict[str, Any]) -> TraditionalSignal:
                # Generate deterministic mock signals
                signal_value = self.np_random.uniform(-1, 1)
                
                if signal_value > 0.3:
                    signal_type = MLSignalType.BUY
                elif signal_value < -0.3:
                    signal_type = MLSignalType.SELL
                else:
                    signal_type = MLSignalType.HOLD
                
                return TraditionalSignal(
                    strategy_name=self.strategy_name,
                    signal_type=signal_type,
                    strength=abs(signal_value),
                    confidence=min(0.9, abs(signal_value) + 0.3),
                    supporting_indicators={f"{self.strategy_name}_value": signal_value},
                    timestamp=datetime.now()
                )
        
        strategy_names = ['rsi', 'macd', 'bollinger', 'momentum', 'mean_reversion', 'breakout']
        for name in strategy_names:
            self.traditional_strategies[name] = MockStrategy(name)
    
    async def _initialize_risk_manager(self):
        """Initialize risk manager"""
        if RISK_MANAGER_AVAILABLE:
            self.risk_manager = UnifiedRiskManager()
        else:
            logger.warning("Risk manager not available, using basic risk controls")
            self.risk_manager = None
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking for all strategies"""
        all_strategies = ['ml_ensemble'] + list(self.traditional_strategies.keys())
        
        for strategy_name in all_strategies:
            self.strategy_performance[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name,
                total_signals=0,
                correct_signals=0,
                accuracy=0.5,  # Start with neutral accuracy
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                recent_performance=[],
                last_updated=datetime.now()
            )
    
    async def generate_combined_signal(self, features: MLFeatures, 
                                     symbol: str, current_price: float) -> CombinedSignal:
        """
        Generate combined signal from ML predictions and traditional strategies
        
        Args:
            features: ML features from feature pipeline
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            CombinedSignal with unified recommendation
        """
        
        # Get ML prediction
        ml_ensemble = await self.ml_model_manager.predict(features, symbol)
        ml_signal = ml_ensemble.final_prediction
        
        # Get traditional strategy signals
        traditional_signals = await self._get_traditional_signals(features, symbol, current_price)
        
        # Detect market regime
        market_regime = await self._detect_market_regime(features)
        
        # Get regime-specific strategy weights
        current_weights = self._get_regime_weights(market_regime) 
        
        # Combine signals with weighted approach
        final_signal, final_confidence = self._combine_signals(
            ml_signal, traditional_signals, current_weights
        )
        
        # Calculate position size
        position_size = await self._calculate_position_size(
            ml_signal, traditional_signals, final_signal, final_confidence, symbol
        )
        
        # Calculate risk metrics
        risk_metrics = await self._calculate_risk_metrics(
            final_signal, position_size, symbol, current_price
        )
        
        # Create combined signal
        combined_signal = CombinedSignal(
            ml_signal=ml_signal,
            traditional_signals=traditional_signals,
            final_signal=final_signal,
            final_confidence=final_confidence,
            strategy_weights=current_weights,
            position_size=position_size,
            risk_metrics=risk_metrics,
            market_regime=market_regime,
            timestamp=datetime.now()
        )
        
        # Update signal history
        await self._update_signal_history(combined_signal)
        
        return combined_signal
    
    async def _get_traditional_signals(self, features: MLFeatures, 
                                     symbol: str, current_price: float) -> List[TraditionalSignal]:
        """Get signals from all traditional strategies"""
        traditional_signals = []
        
        # Prepare data for traditional strategies
        strategy_data = self._prepare_traditional_strategy_data(features, current_price)
        
        for strategy_name, strategy in self.traditional_strategies.items():
            try:
                signal = await self._get_strategy_signal(strategy, strategy_data, strategy_name)
                if signal:
                    traditional_signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error getting signal from {strategy_name}: {e}")
        
        return traditional_signals
    
    def _prepare_traditional_strategy_data(self, features: MLFeatures, 
                                         current_price: float) -> Dict[str, Any]:
        """Prepare data format for traditional strategies"""
        return {
            'price': current_price,
            'technical_indicators': features.technical_indicators,
            'volume': features.microstructure_features.get('volume', 0),
            'volatility': features.technical_indicators.get('volatility', 0),
            'timestamp': features.timestamp
        }
    
    async def _get_strategy_signal(self, strategy: Any, data: Dict[str, Any], 
                                 strategy_name: str) -> Optional[TraditionalSignal]:
        """Get signal from a specific traditional strategy"""
        try:
            if hasattr(strategy, 'generate_signal'):
                return strategy.generate_signal(data)
            else:
                # Fallback for mock strategies or different interfaces
                return strategy.generate_signal(data)
                
        except Exception as e:
            logger.error(f"Error generating signal from {strategy_name}: {e}")
            return None
    
    async def _detect_market_regime(self, features: MLFeatures) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Use various indicators to detect regime
            volatility = features.technical_indicators.get('volatility', 0)
            trend_strength = features.technical_indicators.get('adx', 0)
            rsi = features.technical_indicators.get('rsi', 50)
            
            # High volatility regime
            if volatility > 0.03:  # 3% daily volatility
                return MarketRegime.HIGH_VOLATILITY
            
            # Low volatility regime
            if volatility < 0.01:  # 1% daily volatility
                return MarketRegime.LOW_VOLATILITY
            
            # Trending regimes
            if trend_strength > 25:  # Strong trend
                if rsi > 55:
                    return MarketRegime.TRENDING_UP
                elif rsi < 45:
                    return MarketRegime.TRENDING_DOWN
            
            # Sideways regime (weak trend)
            if trend_strength < 20:
                return MarketRegime.SIDEWAYS
            
            return MarketRegime.UNCERTAIN
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.UNCERTAIN
    
    def _get_regime_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get strategy weights based on market regime"""
        if regime in self.config['regime_adaptive_weights']:
            regime_weights = self.config['regime_adaptive_weights'][regime].copy()
            
            # Ensure all strategies have weights
            for strategy in self.strategy_weights:
                if strategy not in regime_weights:
                    regime_weights[strategy] = self.strategy_weights[strategy]
            
            return regime_weights
        else:
            return self.strategy_weights.copy()
    
    def _combine_signals(self, ml_signal: MLPrediction, traditional_signals: List[TraditionalSignal],
                        weights: Dict[str, float]) -> Tuple[MLSignalType, float]:
        """Combine ML and traditional signals using weighted approach"""
        
        # Signal strength mapping
        signal_values = {
            MLSignalType.STRONG_SELL: -1.0,
            MLSignalType.SELL: -0.5,
            MLSignalType.HOLD: 0.0,
            MLSignalType.BUY: 0.5,
            MLSignalType.STRONG_BUY: 1.0
        }
        
        # Calculate weighted signal
        total_weighted_signal = 0.0
        total_weight = 0.0
        
        # Add ML signal
        ml_weight = weights.get('ml_ensemble', 0.4)
        ml_signal_value = signal_values.get(ml_signal.signal_type, 0.0)
        ml_weighted_contribution = ml_signal_value * ml_signal.confidence * ml_weight
        
        total_weighted_signal += ml_weighted_contribution
        total_weight += ml_weight * ml_signal.confidence
        
        # Add traditional signals
        for trad_signal in traditional_signals:
            trad_weight = weights.get(trad_signal.strategy_name, 0.1)
            trad_signal_value = signal_values.get(trad_signal.signal_type, 0.0)
            trad_weighted_contribution = trad_signal_value * trad_signal.confidence * trad_weight
            
            total_weighted_signal += trad_weighted_contribution
            total_weight += trad_weight * trad_signal.confidence
        
        # Calculate final signal
        if total_weight == 0:
            return MLSignalType.HOLD, 0.0
        
        final_signal_value = total_weighted_signal / total_weight
        final_confidence = min(1.0, total_weight)  # Confidence based on total weight
        
        # Convert back to signal type
        if final_signal_value > 0.7:
            final_signal_type = MLSignalType.STRONG_BUY
        elif final_signal_value > 0.2:
            final_signal_type = MLSignalType.BUY
        elif final_signal_value < -0.7:
            final_signal_type = MLSignalType.STRONG_SELL
        elif final_signal_value < -0.2:
            final_signal_type = MLSignalType.SELL
        else:
            final_signal_type = MLSignalType.HOLD
        
        return final_signal_type, final_confidence
    
    async def _calculate_position_size(self, ml_signal: MLPrediction, 
                                     traditional_signals: List[TraditionalSignal],
                                     final_signal: MLSignalType, final_confidence: float,
                                     symbol: str) -> float:
        """Calculate recommended position size"""
        
        # Base position size from confidence
        base_size = final_confidence * self.config['signal_thresholds']['position_size_multiplier']
        
        # Apply risk management constraints
        max_size = self.config['risk_management']['max_position_size']
        position_size = min(base_size, max_size)
        
        # Adjust for signal strength
        if final_signal in [MLSignalType.STRONG_BUY, MLSignalType.STRONG_SELL]:
            position_size *= 1.2  # Increase for strong signals
        elif final_signal == MLSignalType.HOLD:
            position_size = 0.0
        
        # Apply risk manager constraints if available
        if self.risk_manager:
            try:
                risk_adjusted_size = await self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    signal_strength=final_confidence,
                    market_conditions={'volatility': 0.02}  # Would get from features
                )
                position_size = min(position_size, risk_adjusted_size)
            except Exception as e:
                logger.error(f"Error applying risk management: {e}")
        
        return max(0.0, position_size)
    
    async def _calculate_risk_metrics(self, final_signal: MLSignalType, position_size: float,
                                    symbol: str, current_price: float) -> Dict[str, float]:
        """Calculate risk metrics for the signal"""
        
        risk_metrics = {
            'position_size': position_size,
            'max_loss': position_size * current_price * 0.02,  # 2% stop loss
            'expected_profit': position_size * current_price * 0.03,  # 3% take profit
            'risk_reward_ratio': 1.5,
            'var_95': position_size * current_price * 0.05,  # 5% VaR
        }
        
        # Adjust for signal direction
        if final_signal in [MLSignalType.SELL, MLSignalType.STRONG_SELL]:
            risk_metrics['max_loss'] *= -1  # Short position
            risk_metrics['expected_profit'] *= -1
        
        return risk_metrics
    
    async def _update_signal_history(self, combined_signal: CombinedSignal):
        """Update signal history and performance tracking"""
        
        # Add to signal history
        self.signal_history.append(combined_signal)
        
        # Maintain rolling window
        max_history = self.config['performance_tracking']['lookback_window']
        if len(self.signal_history) > max_history:
            self.signal_history = self.signal_history[-max_history:]
        
        # Update strategy weights if needed
        if len(self.signal_history) % self.config['performance_tracking']['rebalance_frequency'] == 0:
            await self._rebalance_strategy_weights()
    
    async def _rebalance_strategy_weights(self):
        """Rebalance strategy weights based on recent performance"""
        try:
            logger.info("Rebalancing strategy weights based on performance...")
            
            # Calculate performance scores for each strategy
            performance_scores = {}
            
            for strategy_name, performance in self.strategy_performance.items():
                if performance.total_signals > 10:  # Need minimum signals
                    # Combine accuracy and recent performance
                    recent_avg = np.mean(performance.recent_performance[-10:]) if performance.recent_performance else 0.5
                    performance_score = (performance.accuracy * 0.7) + (recent_avg * 0.3)
                    performance_scores[strategy_name] = performance_score
                else:
                    performance_scores[strategy_name] = 0.5  # Neutral score
            
            # Update weights based on performance
            total_performance = sum(performance_scores.values())
            if total_performance > 0:
                for strategy_name in self.strategy_weights:
                    if strategy_name in performance_scores:
                        new_weight = performance_scores[strategy_name] / total_performance
                        # Smooth the weight change
                        self.strategy_weights[strategy_name] = (
                            self.strategy_weights[strategy_name] * 0.8 + new_weight * 0.2
                        )
            
            logger.info(f"Updated strategy weights: {self.strategy_weights}")
            
        except Exception as e:
            logger.error(f"Error rebalancing strategy weights: {e}")
    
    def update_strategy_performance(self, strategy_name: str, outcome: float):
        """Update performance metrics for a strategy"""
        if strategy_name not in self.strategy_performance:
            return
        
        performance = self.strategy_performance[strategy_name]
        
        # Update totals
        performance.total_signals += 1
        if outcome > 0:  # Successful signal
            performance.correct_signals += 1
        
        # Update accuracy
        performance.accuracy = performance.correct_signals / performance.total_signals
        
        # Update recent performance
        performance.recent_performance.append(outcome)
        max_recent = 50  # Keep last 50 outcomes
        if len(performance.recent_performance) > max_recent:
            performance.recent_performance = performance.recent_performance[-max_recent:]
        
        # Update timestamp
        performance.last_updated = datetime.now()
    
    def get_strategy_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for all strategies"""
        summary = {}
        
        for strategy_name, performance in self.strategy_performance.items():
            summary[strategy_name] = {
                'total_signals': performance.total_signals,
                'accuracy': performance.accuracy,
                'recent_avg_performance': np.mean(performance.recent_performance[-10:]) if performance.recent_performance else 0.0,
                'current_weight': self.strategy_weights.get(strategy_name, 0.0),
                'last_updated': performance.last_updated.isoformat()
            }
        
        return summary
    
    def get_current_market_regime(self) -> Dict[str, Any]:
        """Get current market regime information"""
        return {
            'regime': self.current_regime.value,
            'confidence': self.regime_confidence,
            'regime_weights': self._get_regime_weights(self.current_regime)
        }

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'MLStrategyOrchestrator',
    'StrategyType',
    'MarketRegime', 
    'SignalStrength',
    'TraditionalSignal',
    'CombinedSignal',
    'StrategyPerformance'
]