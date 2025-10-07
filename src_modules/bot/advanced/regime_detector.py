"""
Market Regime Detection System

This module provides sophisticated market regime detection and classification
for the Bybit trading bot. It includes multiple regime identification algorithms
and filtering capabilities for strategy activation/deactivation.

Key Features:
- Multiple regime detection algorithms (volatility, trend, volatility clustering)
- Real-time regime classification
- Regime transition detection
- Strategy filtering based on market conditions
- Historical regime analysis
- Regime persistence tracking

Author: Trading Bot Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

# Configure logging
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    BULL_MARKET = "bull_market"           # Strong upward trend, low volatility
    BEAR_MARKET = "bear_market"           # Strong downward trend, low volatility
    SIDEWAYS = "sideways"                 # No clear trend, low volatility
    HIGH_VOLATILITY = "high_volatility"   # High volatility, unclear trend
    CRASH = "crash"                       # Extreme downward movement, very high volatility
    BUBBLE = "bubble"                     # Extreme upward movement, high volatility
    RECOVERY = "recovery"                 # Transition from bear to bull
    DISTRIBUTION = "distribution"         # Transition from bull to bear
    

@dataclass
class RegimeClassification:
    """Regime classification result"""
    regime: MarketRegime
    confidence: float
    start_time: datetime
    end_time: Optional[datetime] = None
    features: Optional[Dict[str, float]] = None
    persistence_score: float = 0.0


@dataclass
class RegimeFeatures:
    """Market regime features"""
    volatility: float
    trend_strength: float
    trend_direction: float
    momentum: float
    volume_trend: float
    drawdown: float
    return_skewness: float
    return_kurtosis: float
    correlation_breakdown: float
    regime_persistence: float


class RegimeDetector:
    """
    Advanced Market Regime Detection System
    
    This class provides comprehensive market regime detection using multiple
    algorithms and sophisticated statistical methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the regime detector
        
        Args:
            config: Configuration dictionary with regime detection parameters
        """
        self.config = config or self._get_default_config()
        self.current_regime = None
        self.regime_history = []
        self.features_scaler = StandardScaler()
        self.gmm_model = None
        self.regime_thresholds = self._initialize_thresholds()
        self.regime_transitions = []
        
        logger.info("RegimeDetector initialized with configuration")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for regime detection"""
        return {
            'volatility_window': 20,
            'trend_window': 50,
            'momentum_window': 14,
            'regime_persistence_threshold': 0.7,
            'volatility_threshold_low': 0.15,
            'volatility_threshold_high': 0.40,
            'trend_threshold': 0.02,
            'crash_threshold': -0.20,
            'bubble_threshold': 0.30,
            'min_regime_duration': 5,  # days
            'confidence_threshold': 0.6,
            'use_volume_confirmation': True,
            'regime_clustering_components': 6,
            'feature_lookback_days': 252  # 1 year
        }
    
    def _initialize_thresholds(self) -> Dict[MarketRegime, Dict[str, float]]:
        """Initialize regime classification thresholds"""
        return {
            MarketRegime.BULL_MARKET: {
                'volatility_max': self.config['volatility_threshold_low'],
                'trend_direction_min': self.config['trend_threshold'],
                'momentum_min': 0.0,
                'drawdown_max': -0.10
            },
            MarketRegime.BEAR_MARKET: {
                'volatility_max': self.config['volatility_threshold_low'],
                'trend_direction_max': -self.config['trend_threshold'],
                'momentum_max': 0.0,
                'drawdown_min': -0.15
            },
            MarketRegime.SIDEWAYS: {
                'volatility_max': self.config['volatility_threshold_low'],
                'trend_strength_max': 0.5,
                'drawdown_max': -0.08,
                'drawdown_min': -0.08
            },
            MarketRegime.HIGH_VOLATILITY: {
                'volatility_min': self.config['volatility_threshold_high'],
                'trend_strength_max': 0.3
            },
            MarketRegime.CRASH: {
                'volatility_min': self.config['volatility_threshold_high'],
                'trend_direction_max': self.config['crash_threshold'],
                'momentum_max': -0.5
            },
            MarketRegime.BUBBLE: {
                'volatility_min': self.config['volatility_threshold_low'],
                'trend_direction_min': self.config['bubble_threshold'],
                'momentum_min': 0.3
            },
            MarketRegime.RECOVERY: {
                'volatility_max': self.config['volatility_threshold_high'],
                'trend_direction_min': 0.0,
                'momentum_min': 0.1,
                'previous_regime': MarketRegime.BEAR_MARKET
            },
            MarketRegime.DISTRIBUTION: {
                'volatility_min': self.config['volatility_threshold_low'],
                'trend_direction_max': 0.0,
                'momentum_max': -0.1,
                'previous_regime': MarketRegime.BULL_MARKET
            }
        }
    
    def calculate_regime_features(self, data: pd.DataFrame) -> RegimeFeatures:
        """
        Calculate comprehensive market regime features
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            RegimeFeatures object with calculated features
        """
        try:
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            # Volatility (realized volatility)
            volatility = returns.rolling(
                window=self.config['volatility_window']
            ).std().iloc[-1] * np.sqrt(252)
            
            # Trend strength and direction
            prices = data['close'].values
            trend_window = min(len(prices), self.config['trend_window'])
            
            if trend_window > 1:
                x = np.arange(trend_window)
                y = prices[-trend_window:]
                slope, _, r_value, _, _ = stats.linregress(x, y)
                
                trend_strength = abs(r_value)
                trend_direction = slope / prices[-1]  # Normalized slope
            else:
                trend_strength = 0.0
                trend_direction = 0.0
            
            # Momentum (RSI-like calculation)
            momentum_window = min(len(returns), self.config['momentum_window'])
            if momentum_window > 1:
                recent_returns = returns.iloc[-momentum_window:]
                gains = recent_returns[recent_returns > 0].sum()
                losses = abs(recent_returns[recent_returns < 0].sum())
                
                if losses != 0:
                    rs = gains / losses
                    momentum = (rs - 1) / (rs + 1)  # Normalized RSI
                else:
                    momentum = 1.0 if gains > 0 else 0.0
            else:
                momentum = 0.0
            
            # Volume trend (if available)
            if 'volume' in data.columns:
                volume_ma_short = data['volume'].rolling(10).mean().iloc[-1]
                volume_ma_long = data['volume'].rolling(30).mean().iloc[-1]
                volume_trend = (volume_ma_short / volume_ma_long - 1) if volume_ma_long > 0 else 0.0
            else:
                volume_trend = 0.0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = ((cumulative_returns - rolling_max) / rolling_max).min()
            
            # Return distribution characteristics
            return_skewness = returns.skew() if len(returns) > 3 else 0.0
            return_kurtosis = returns.kurtosis() if len(returns) > 4 else 0.0
            
            # Correlation breakdown (measure of market stress)
            if len(returns) > 30:
                recent_corr = returns.rolling(10).corr(returns.shift(1)).mean()
                long_term_corr = returns.rolling(60).corr(returns.shift(1)).mean()
                correlation_breakdown = abs(recent_corr - long_term_corr)
            else:
                correlation_breakdown = 0.0
            
            # Regime persistence (how stable current conditions are)
            if len(self.regime_history) > 0:
                recent_regimes = [r.regime for r in self.regime_history[-10:]]
                if recent_regimes:
                    most_common_regime = max(set(recent_regimes), key=recent_regimes.count)
                    regime_persistence = recent_regimes.count(most_common_regime) / len(recent_regimes)
                else:
                    regime_persistence = 0.0
            else:
                regime_persistence = 0.0
            
            return RegimeFeatures(
                volatility=volatility,
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                momentum=momentum,
                volume_trend=volume_trend,
                drawdown=drawdown,
                return_skewness=return_skewness,
                return_kurtosis=return_kurtosis,
                correlation_breakdown=correlation_breakdown,
                regime_persistence=regime_persistence
            )
            
        except Exception as e:
            logger.error(f"Error calculating regime features: {e}")
            # Return default features
            return RegimeFeatures(
                volatility=0.0, trend_strength=0.0, trend_direction=0.0,
                momentum=0.0, volume_trend=0.0, drawdown=0.0,
                return_skewness=0.0, return_kurtosis=0.0,
                correlation_breakdown=0.0, regime_persistence=0.0
            )
    
    def classify_regime_rule_based(self, features: RegimeFeatures) -> Tuple[MarketRegime, float]:
        """
        Classify market regime using rule-based approach
        
        Args:
            features: Calculated regime features
            
        Returns:
            Tuple of (regime, confidence)
        """
        try:
            regime_scores = {}
            
            # Check each regime against its thresholds
            for regime, thresholds in self.regime_thresholds.items():
                score = 1.0
                criteria_met = 0
                total_criteria = 0
                
                # Volatility checks
                if 'volatility_min' in thresholds:
                    total_criteria += 1
                    if features.volatility >= thresholds['volatility_min']:
                        criteria_met += 1
                    else:
                        score *= 0.5
                
                if 'volatility_max' in thresholds:
                    total_criteria += 1
                    if features.volatility <= thresholds['volatility_max']:
                        criteria_met += 1
                    else:
                        score *= 0.5
                
                # Trend checks
                if 'trend_direction_min' in thresholds:
                    total_criteria += 1
                    if features.trend_direction >= thresholds['trend_direction_min']:
                        criteria_met += 1
                    else:
                        score *= 0.7
                
                if 'trend_direction_max' in thresholds:
                    total_criteria += 1
                    if features.trend_direction <= thresholds['trend_direction_max']:
                        criteria_met += 1
                    else:
                        score *= 0.7
                
                if 'trend_strength_max' in thresholds:
                    total_criteria += 1
                    if features.trend_strength <= thresholds['trend_strength_max']:
                        criteria_met += 1
                    else:
                        score *= 0.8
                
                # Momentum checks
                if 'momentum_min' in thresholds:
                    total_criteria += 1
                    if features.momentum >= thresholds['momentum_min']:
                        criteria_met += 1
                    else:
                        score *= 0.6
                
                if 'momentum_max' in thresholds:
                    total_criteria += 1
                    if features.momentum <= thresholds['momentum_max']:
                        criteria_met += 1
                    else:
                        score *= 0.6
                
                # Drawdown checks
                if 'drawdown_max' in thresholds:
                    total_criteria += 1
                    if features.drawdown >= thresholds['drawdown_max']:
                        criteria_met += 1
                    else:
                        score *= 0.8
                
                if 'drawdown_min' in thresholds:
                    total_criteria += 1
                    if features.drawdown <= thresholds['drawdown_min']:
                        criteria_met += 1
                    else:
                        score *= 0.8
                
                # Previous regime dependency
                if 'previous_regime' in thresholds and len(self.regime_history) > 0:
                    if self.regime_history[-1].regime == thresholds['previous_regime']:
                        score *= 1.3  # Boost for valid transitions
                    else:
                        score *= 0.3  # Penalty for invalid transitions
                
                # Calculate final score
                if total_criteria > 0:
                    criteria_ratio = criteria_met / total_criteria
                    regime_scores[regime] = score * criteria_ratio
                else:
                    regime_scores[regime] = score * 0.5
            
            # Find best regime
            if regime_scores:
                best_regime = max(regime_scores, key=regime_scores.get)
                confidence = regime_scores[best_regime]
                
                # Apply confidence threshold
                if confidence < self.config['confidence_threshold']:
                    # Default to sideways if no clear regime
                    best_regime = MarketRegime.SIDEWAYS
                    confidence = 0.5
                
                return best_regime, min(confidence, 1.0)
            else:
                return MarketRegime.SIDEWAYS, 0.5
                
        except Exception as e:
            logger.error(f"Error in rule-based regime classification: {e}")
            return MarketRegime.SIDEWAYS, 0.5
    
    def classify_regime_ml(self, data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Classify market regime using machine learning approach
        
        Args:
            data: Historical OHLCV data
            
        Returns:
            Tuple of (regime, confidence)
        """
        try:
            if len(data) < self.config['feature_lookback_days']:
                logger.warning("Insufficient data for ML regime classification")
                return MarketRegime.SIDEWAYS, 0.5
            
            # Prepare features for ML model
            features_list = []
            
            # Calculate rolling features
            returns = data['close'].pct_change().dropna()
            
            for i in range(self.config['volatility_window'], len(data)):
                window_data = data.iloc[i-self.config['volatility_window']:i+1]
                features = self.calculate_regime_features(window_data)
                
                feature_vector = [
                    features.volatility,
                    features.trend_strength,
                    features.trend_direction,
                    features.momentum,
                    features.volume_trend,
                    features.drawdown,
                    features.return_skewness,
                    features.return_kurtosis,
                    features.correlation_breakdown
                ]
                
                features_list.append(feature_vector)
            
            if len(features_list) < 50:  # Minimum samples for clustering
                logger.warning("Insufficient samples for ML regime classification")
                return MarketRegime.SIDEWAYS, 0.5
            
            features_array = np.array(features_list)
            
            # Train Gaussian Mixture Model if not already trained
            if self.gmm_model is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Standardize features
                    features_scaled = self.features_scaler.fit_transform(features_array)
                    
                    # Fit GMM
                    self.gmm_model = GaussianMixture(
                        n_components=self.config['regime_clustering_components'],
                        covariance_type='full',
                        random_state=42
                    )
                    self.gmm_model.fit(features_scaled)
            
            # Classify current regime
            current_features = features_list[-1]
            current_features_scaled = self.features_scaler.transform([current_features])
            
            # Get cluster probabilities
            cluster_probs = self.gmm_model.predict_proba(current_features_scaled)[0]
            dominant_cluster = np.argmax(cluster_probs)
            confidence = cluster_probs[dominant_cluster]
            
            # Map cluster to regime (simplified mapping)
            cluster_to_regime = {
                0: MarketRegime.BULL_MARKET,
                1: MarketRegime.BEAR_MARKET,
                2: MarketRegime.SIDEWAYS,
                3: MarketRegime.HIGH_VOLATILITY,
                4: MarketRegime.CRASH,
                5: MarketRegime.BUBBLE
            }
            
            regime = cluster_to_regime.get(dominant_cluster, MarketRegime.SIDEWAYS)
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error in ML regime classification: {e}")
            return MarketRegime.SIDEWAYS, 0.5
    
    def detect_regime(self, data: pd.DataFrame, method: str = 'combined') -> RegimeClassification:
        """
        Detect current market regime
        
        Args:
            data: OHLCV data
            method: Detection method ('rule_based', 'ml', 'combined')
            
        Returns:
            RegimeClassification object
        """
        try:
            features = self.calculate_regime_features(data)
            
            if method == 'rule_based':
                regime, confidence = self.classify_regime_rule_based(features)
            elif method == 'ml':
                regime, confidence = self.classify_regime_ml(data)
            elif method == 'combined':
                # Combine both methods
                regime_rb, conf_rb = self.classify_regime_rule_based(features)
                regime_ml, conf_ml = self.classify_regime_ml(data)
                
                # Weighted combination
                if conf_rb > conf_ml:
                    regime = regime_rb
                    confidence = (conf_rb * 0.7 + conf_ml * 0.3)
                else:
                    regime = regime_ml
                    confidence = (conf_ml * 0.7 + conf_rb * 0.3)
            else:
                raise ValueError(f"Unknown detection method: {method}")
            
            # Create classification result
            classification = RegimeClassification(
                regime=regime,
                confidence=confidence,
                start_time=datetime.now(),
                features=features.__dict__,
                persistence_score=features.regime_persistence
            )
            
            # Check for regime transition
            if self.current_regime != regime:
                if self.current_regime is not None:
                    # Record transition
                    transition = {
                        'from_regime': self.current_regime,
                        'to_regime': regime,
                        'timestamp': datetime.now(),
                        'confidence': confidence
                    }
                    self.regime_transitions.append(transition)
                    logger.info(f"Regime transition detected: {self.current_regime} -> {regime}")
                
                self.current_regime = regime
            
            # Update regime history
            self.regime_history.append(classification)
            
            # Keep history manageable
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-500:]
            
            return classification
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            # Return default classification
            return RegimeClassification(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.5,
                start_time=datetime.now()
            )
    
    def should_trade_in_regime(self, regime: MarketRegime, strategy_type: str) -> bool:
        """
        Determine if a strategy should trade in the given regime
        
        Args:
            regime: Current market regime
            strategy_type: Type of trading strategy
            
        Returns:
            Boolean indicating if trading should occur
        """
        # Define regime-strategy compatibility matrix
        compatibility_matrix = {
            'trend_following': {
                MarketRegime.BULL_MARKET: True,
                MarketRegime.BEAR_MARKET: True,
                MarketRegime.SIDEWAYS: False,
                MarketRegime.HIGH_VOLATILITY: False,
                MarketRegime.CRASH: False,
                MarketRegime.BUBBLE: True,
                MarketRegime.RECOVERY: True,
                MarketRegime.DISTRIBUTION: False
            },
            'mean_reversion': {
                MarketRegime.BULL_MARKET: False,
                MarketRegime.BEAR_MARKET: False,
                MarketRegime.SIDEWAYS: True,
                MarketRegime.HIGH_VOLATILITY: True,
                MarketRegime.CRASH: False,
                MarketRegime.BUBBLE: False,
                MarketRegime.RECOVERY: True,
                MarketRegime.DISTRIBUTION: True
            },
            'momentum': {
                MarketRegime.BULL_MARKET: True,
                MarketRegime.BEAR_MARKET: True,
                MarketRegime.SIDEWAYS: False,
                MarketRegime.HIGH_VOLATILITY: False,
                MarketRegime.CRASH: False,
                MarketRegime.BUBBLE: True,
                MarketRegime.RECOVERY: True,
                MarketRegime.DISTRIBUTION: False
            },
            'volatility': {
                MarketRegime.BULL_MARKET: False,
                MarketRegime.BEAR_MARKET: False,
                MarketRegime.SIDEWAYS: False,
                MarketRegime.HIGH_VOLATILITY: True,
                MarketRegime.CRASH: True,
                MarketRegime.BUBBLE: True,
                MarketRegime.RECOVERY: False,
                MarketRegime.DISTRIBUTION: False
            }
        }
        
        return compatibility_matrix.get(strategy_type, {}).get(regime, False)
    
    def get_regime_statistics(self) -> Dict:
        """
        Get comprehensive regime statistics
        
        Returns:
            Dictionary with regime statistics
        """
        try:
            if not self.regime_history:
                return {}
            
            # Regime distribution
            regime_counts = {}
            for classification in self.regime_history:
                regime = classification.regime
                regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
            
            # Regime durations
            regime_durations = {}
            current_regime = None
            current_start = None
            
            for classification in self.regime_history:
                if classification.regime != current_regime:
                    if current_regime is not None and current_start is not None:
                        duration = (classification.start_time - current_start).days
                        if current_regime.value not in regime_durations:
                            regime_durations[current_regime.value] = []
                        regime_durations[current_regime.value].append(duration)
                    
                    current_regime = classification.regime
                    current_start = classification.start_time
            
            # Average durations
            avg_durations = {}
            for regime, durations in regime_durations.items():
                avg_durations[regime] = np.mean(durations) if durations else 0
            
            # Recent transitions
            recent_transitions = self.regime_transitions[-10:] if self.regime_transitions else []
            
            # Current regime persistence
            current_persistence = 0.0
            if len(self.regime_history) > 0:
                current_regime = self.regime_history[-1].regime
                recent_regimes = [r.regime for r in self.regime_history[-20:]]
                current_persistence = recent_regimes.count(current_regime) / len(recent_regimes)
            
            return {
                'regime_distribution': regime_counts,
                'average_regime_durations': avg_durations,
                'recent_transitions': [
                    {
                        'from': t['from_regime'].value,
                        'to': t['to_regime'].value,
                        'timestamp': t['timestamp'].isoformat(),
                        'confidence': t['confidence']
                    } for t in recent_transitions
                ],
                'current_regime': self.current_regime.value if self.current_regime else None,
                'current_persistence': current_persistence,
                'total_classifications': len(self.regime_history),
                'total_transitions': len(self.regime_transitions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating regime statistics: {e}")
            return {}
    
    def reset_history(self):
        """Reset regime history and transitions"""
        self.regime_history = []
        self.regime_transitions = []
        self.current_regime = None
        self.gmm_model = None
        logger.info("Regime detector history reset")


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate different market regimes
    prices = []
    regime_periods = [
        (0, 50, 'bull'),      # Bull market
        (50, 100, 'sideways'), # Sideways
        (100, 120, 'crash'),   # Crash
        (120, 180, 'recovery'), # Recovery
        (180, 250, 'bull'),    # Bull market
        (250, 300, 'high_vol'), # High volatility
        (300, 365, 'bear')     # Bear market
    ]
    
    price = 100
    for start, end, regime_type in regime_periods:
        period_length = end - start
        
        if regime_type == 'bull':
            trend = np.cumsum(np.random.normal(0.002, 0.015, period_length))
        elif regime_type == 'bear':
            trend = np.cumsum(np.random.normal(-0.002, 0.015, period_length))
        elif regime_type == 'sideways':
            trend = np.cumsum(np.random.normal(0, 0.008, period_length))
        elif regime_type == 'crash':
            trend = np.cumsum(np.random.normal(-0.005, 0.03, period_length))
        elif regime_type == 'recovery':
            trend = np.cumsum(np.random.normal(0.003, 0.02, period_length))
        elif regime_type == 'high_vol':
            trend = np.cumsum(np.random.normal(0, 0.04, period_length))
        
        period_prices = price * (1 + trend)
        prices.extend(period_prices)
        price = period_prices[-1]
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': np.array(prices) * (1 + np.random.uniform(0, 0.02, len(prices))),
        'low': np.array(prices) * (1 - np.random.uniform(0, 0.02, len(prices))),
        'volume': np.random.uniform(1000000, 5000000, len(prices))
    })
    test_data['open'] = test_data['close'].shift(1)
    test_data = test_data.dropna()
    
    # Test regime detector
    detector = RegimeDetector()
    
    print("Testing Market Regime Detection System")
    print("=" * 50)
    
    # Test regime detection on different periods
    test_periods = [
        (50, "Bull Market Period"),
        (80, "Sideways Period"),
        (110, "Crash Period"),
        (150, "Recovery Period"),
        (200, "Bull Market Period 2"),
        (280, "High Volatility Period"),
        (330, "Bear Market Period")
    ]
    
    for period_end, description in test_periods:
        period_data = test_data.iloc[:period_end]
        classification = detector.detect_regime(period_data)
        
        print(f"\n{description}:")
        print(f"  Detected Regime: {classification.regime.value}")
        print(f"  Confidence: {classification.confidence:.3f}")
        print(f"  Persistence Score: {classification.persistence_score:.3f}")
        
        # Test strategy filtering
        for strategy_type in ['trend_following', 'mean_reversion', 'momentum', 'volatility']:
            should_trade = detector.should_trade_in_regime(classification.regime, strategy_type)
            print(f"  {strategy_type}: {'✓' if should_trade else '✗'}")
    
    # Print final statistics
    print(f"\nFinal Regime Statistics:")
    print("=" * 30)
    stats = detector.get_regime_statistics()
    
    print(f"Total Classifications: {stats.get('total_classifications', 0)}")
    print(f"Total Transitions: {stats.get('total_transitions', 0)}")
    print(f"Current Regime: {stats.get('current_regime', 'None')}")
    print(f"Current Persistence: {stats.get('current_persistence', 0):.3f}")
    
    print(f"\nRegime Distribution:")
    for regime, count in stats.get('regime_distribution', {}).items():
        print(f"  {regime}: {count}")
    
    print(f"\nAverage Regime Durations:")
    for regime, duration in stats.get('average_regime_durations', {}).items():
        print(f"  {regime}: {duration:.1f} days")