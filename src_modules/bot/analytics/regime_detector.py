"""
Advanced Regime Detection Engine.
Identifies market regimes and regime transitions using multiple methodologies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class MarketRegime(Enum):
    """Market regime types."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"

@dataclass
class RegimeTransition:
    """Market regime transition information."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_date: datetime
    transition_probability: float
    confidence_score: float
    duration_from: int  # days in previous regime
    expected_duration_to: int  # expected days in new regime
    regime_features: Dict[str, float]
    economic_indicators: Dict[str, float]

@dataclass
class RegimeAnalysis:
    """Complete regime analysis result."""
    current_regime: MarketRegime
    regime_probability: float
    regime_confidence: float
    alternative_regimes: Dict[MarketRegime, float]
    regime_duration: int
    expected_regime_change: Optional[datetime]
    regime_characteristics: Dict[str, float]
    historical_transitions: List[RegimeTransition]
    regime_forecast: Dict[MarketRegime, float]
    analysis_timestamp: datetime

class RegimeDetector:
    """Advanced market regime detection and analysis."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Regime detection configuration
        self.regime_config = {
            'lookback_period': 252,  # 1 year
            'min_regime_duration': 20,  # minimum 20 days
            'transition_threshold': 0.7,
            'volatility_threshold': {
                'low': 0.15,  # 15% annualized
                'high': 0.30  # 30% annualized
            },
            'trend_threshold': {
                'bull': 0.10,  # 10% cumulative return threshold
                'bear': -0.10
            },
            'features': [
                'returns', 'volatility', 'volume', 'momentum',
                'rsi', 'bollinger_position', 'drawdown'
            ]
        }
        
        # Regime models
        self.regime_models = {}
        self.regime_history = []
        self.current_regime = None
        
        # Feature scalers
        self.feature_scaler = StandardScaler()
        
        self.logger.info("RegimeDetector initialized")
    
    async def detect_current_regime(self, 
                                  market_data: pd.DataFrame,
                                  method: str = 'gaussian_mixture') -> RegimeAnalysis:
        """Detect current market regime using specified method."""
        try:
            # Extract features
            features = await self._extract_regime_features(market_data)
            
            # Detect regime
            if method == 'gaussian_mixture':
                regime_result = await self._gaussian_mixture_detection(features)
            elif method == 'markov_switching':
                regime_result = await self._markov_switching_detection(features)
            elif method == 'threshold_based':
                regime_result = await self._threshold_based_detection(features)
            elif method == 'clustering':
                regime_result = await self._clustering_detection(features)
            else:
                raise ValueError(f"Unsupported detection method: {method}")
            
            # Analyze regime transitions
            transitions = await self._analyze_regime_transitions(features)
            
            # Forecast regime changes
            regime_forecast = await self._forecast_regime_changes(features, regime_result)
            
            # Create analysis result
            analysis = RegimeAnalysis(
                current_regime=regime_result['regime'],
                regime_probability=regime_result['probability'],
                regime_confidence=regime_result['confidence'],
                alternative_regimes=regime_result['alternatives'],
                regime_duration=regime_result['duration'],
                expected_regime_change=regime_result['expected_change'],
                regime_characteristics=regime_result['characteristics'],
                historical_transitions=transitions,
                regime_forecast=regime_forecast,
                analysis_timestamp=datetime.now()
            )
            
            # Update regime history
            self._update_regime_history(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            raise
    
    async def _extract_regime_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection."""
        try:
            features = pd.DataFrame(index=market_data.index)
            
            # Price-based features
            if 'close' in market_data.columns:
                prices = market_data['close']
                
                # Returns
                features['returns'] = prices.pct_change()
                
                # Volatility (rolling 20-day)
                features['volatility'] = features['returns'].rolling(20).std() * np.sqrt(252)
                
                # Momentum (20-day return)
                features['momentum'] = prices.pct_change(20)
                
                # RSI
                delta = features['returns']
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features['rsi'] = 100 - (100 / (1 + rs))
                
                # Bollinger Band position
                sma_20 = prices.rolling(20).mean()
                std_20 = prices.rolling(20).std()
                features['bollinger_position'] = (prices - sma_20) / (2 * std_20)
                
                # Drawdown
                cumulative = (1 + features['returns']).cumprod()
                running_max = cumulative.expanding().max()
                features['drawdown'] = (cumulative - running_max) / running_max
            
            # Volume-based features
            if 'volume' in market_data.columns:
                volume = market_data['volume']
                features['volume_sma_ratio'] = volume / volume.rolling(20).mean()
                features['volume_volatility'] = volume.rolling(20).std() / volume.rolling(20).mean()
            
            # Additional technical indicators
            if 'high' in market_data.columns and 'low' in market_data.columns:
                # True Range and ATR
                high_low = market_data['high'] - market_data['low']
                high_close = np.abs(market_data['high'] - market_data['close'].shift())
                low_close = np.abs(market_data['low'] - market_data['close'].shift())
                
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                features['atr'] = true_range.rolling(14).mean()
                features['atr_ratio'] = features['atr'] / market_data['close']
            
            # Market structure features
            if len(features) > 50:
                # Trend strength
                features['trend_strength'] = features['returns'].rolling(50).apply(
                    lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0
                )
                
                # Return autocorrelation
                features['return_autocorr'] = features['returns'].rolling(50).apply(
                    lambda x: x.autocorr(1) if len(x) > 10 else 0
                )
            
            # Remove NaN values and infinite values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='forward').fillna(method='backward')
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise
    
    async def _gaussian_mixture_detection(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes using Gaussian Mixture Models."""
        try:
            # Select key features
            key_features = ['returns', 'volatility', 'momentum', 'rsi']
            available_features = [f for f in key_features if f in features.columns]
            
            if len(available_features) < 2:
                raise ValueError("Insufficient features for regime detection")
            
            # Prepare data
            feature_data = features[available_features].dropna()
            
            if len(feature_data) < 100:
                raise ValueError("Insufficient data for reliable regime detection")
            
            # Scale features
            scaled_features = self.feature_scaler.fit_transform(feature_data)
            
            # Fit Gaussian Mixture Model
            n_regimes = 3  # Bull, Bear, Sideways
            gmm = GaussianMixture(n_components=n_regimes, random_state=42)
            regime_labels = gmm.fit_predict(scaled_features)
            
            # Get probabilities
            regime_probs = gmm.predict_proba(scaled_features)
            
            # Identify current regime
            current_prob = regime_probs[-1]
            current_regime_idx = np.argmax(current_prob)
            current_regime_prob = current_prob[current_regime_idx]
            
            # Map regime indices to regime types
            regime_mapping = await self._map_regimes_to_types(
                regime_labels, feature_data, gmm.means_
            )
            
            current_regime = regime_mapping[current_regime_idx]
            
            # Alternative regimes
            alternative_regimes = {}
            for i, prob in enumerate(current_prob):
                if i != current_regime_idx:
                    alternative_regimes[regime_mapping[i]] = prob
            
            # Calculate regime duration
            regime_duration = self._calculate_regime_duration(regime_labels)
            
            # Regime characteristics
            characteristics = await self._calculate_regime_characteristics(
                feature_data.tail(20), current_regime
            )
            
            return {
                'regime': current_regime,
                'probability': current_regime_prob,
                'confidence': current_regime_prob,
                'alternatives': alternative_regimes,
                'duration': regime_duration,
                'expected_change': self._estimate_regime_change_date(regime_duration),
                'characteristics': characteristics
            }
            
        except Exception as e:
            self.logger.error(f"Gaussian mixture detection failed: {e}")
            raise
    
    async def _markov_switching_detection(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes using Markov Switching models (simplified implementation)."""
        try:
            # Simplified Markov switching using volatility regimes
            returns = features['returns'].dropna()
            
            # Calculate rolling volatility
            vol_window = 20
            volatility = returns.rolling(vol_window).std()
            
            # Define volatility thresholds
            vol_low = volatility.quantile(0.33)
            vol_high = volatility.quantile(0.67)
            
            # Classify regimes based on volatility
            current_vol = volatility.iloc[-1]
            
            if current_vol < vol_low:
                current_regime = MarketRegime.LOW_VOLATILITY
                regime_prob = 0.8
            elif current_vol > vol_high:
                current_regime = MarketRegime.HIGH_VOLATILITY
                regime_prob = 0.8
            else:
                # Check trend for sideways vs trending
                recent_returns = returns.tail(20).mean()
                if recent_returns > 0.001:  # Daily return > 0.1%
                    current_regime = MarketRegime.BULL_MARKET
                elif recent_returns < -0.001:
                    current_regime = MarketRegime.BEAR_MARKET
                else:
                    current_regime = MarketRegime.SIDEWAYS
                regime_prob = 0.6
            
            # Simple alternatives
            all_regimes = [MarketRegime.BULL_MARKET, MarketRegime.BEAR_MARKET, 
                          MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOLATILITY, 
                          MarketRegime.LOW_VOLATILITY]
            
            alternative_regimes = {}
            remaining_prob = 1 - regime_prob
            for regime in all_regimes:
                if regime != current_regime:
                    alternative_regimes[regime] = remaining_prob / (len(all_regimes) - 1)
            
            return {
                'regime': current_regime,
                'probability': regime_prob,
                'confidence': regime_prob,
                'alternatives': alternative_regimes,
                'duration': 30,  # Placeholder
                'expected_change': None,
                'characteristics': {'volatility': current_vol}
            }
            
        except Exception as e:
            self.logger.error(f"Markov switching detection failed: {e}")
            raise
    
    async def _threshold_based_detection(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes using threshold-based rules."""
        try:
            # Get recent data
            recent_features = features.tail(20)
            
            # Calculate key metrics
            avg_return = recent_features['returns'].mean()
            avg_volatility = recent_features['volatility'].mean() if 'volatility' in recent_features else 0
            current_drawdown = recent_features['drawdown'].iloc[-1] if 'drawdown' in recent_features else 0
            
            # Apply thresholds
            if avg_volatility > self.regime_config['volatility_threshold']['high']:
                if current_drawdown < -0.20:  # 20% drawdown
                    current_regime = MarketRegime.CRISIS
                    confidence = 0.85
                else:
                    current_regime = MarketRegime.HIGH_VOLATILITY
                    confidence = 0.75
            elif avg_volatility < self.regime_config['volatility_threshold']['low']:
                current_regime = MarketRegime.LOW_VOLATILITY
                confidence = 0.75
            elif avg_return > 0.002:  # Daily return > 0.2%
                current_regime = MarketRegime.BULL_MARKET
                confidence = 0.70
            elif avg_return < -0.002:
                current_regime = MarketRegime.BEAR_MARKET
                confidence = 0.70
            else:
                current_regime = MarketRegime.SIDEWAYS
                confidence = 0.60
            
            # Simple alternatives
            alternative_regimes = {
                MarketRegime.SIDEWAYS: 0.2,
                MarketRegime.BULL_MARKET: 0.15,
                MarketRegime.BEAR_MARKET: 0.15
            }
            
            # Remove current regime from alternatives
            if current_regime in alternative_regimes:
                del alternative_regimes[current_regime]
            
            return {
                'regime': current_regime,
                'probability': confidence,
                'confidence': confidence,
                'alternatives': alternative_regimes,
                'duration': 25,  # Placeholder
                'expected_change': None,
                'characteristics': {
                    'avg_return': avg_return,
                    'avg_volatility': avg_volatility,
                    'drawdown': current_drawdown
                }
            }
            
        except Exception as e:
            self.logger.error(f"Threshold-based detection failed: {e}")
            raise
    
    async def _clustering_detection(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes using clustering algorithms."""
        try:
            # Select features for clustering
            cluster_features = ['returns', 'volatility', 'momentum']
            available_features = [f for f in cluster_features if f in features.columns]
            
            if len(available_features) < 2:
                raise ValueError("Insufficient features for clustering")
            
            # Prepare data
            feature_data = features[available_features].dropna()
            scaled_features = self.feature_scaler.fit_transform(feature_data)
            
            # Apply K-means clustering
            n_clusters = 4
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Get current cluster
            current_cluster = cluster_labels[-1]
            
            # Map clusters to regime types
            cluster_centers = kmeans.cluster_centers_
            regime_mapping = {}
            
            for i, center in enumerate(cluster_centers):
                # Analyze cluster characteristics
                returns_idx = available_features.index('returns') if 'returns' in available_features else 0
                vol_idx = available_features.index('volatility') if 'volatility' in available_features else 1
                
                avg_return = center[returns_idx]
                avg_vol = center[vol_idx] if vol_idx < len(center) else 0
                
                if avg_vol > 1.0:  # High volatility cluster
                    regime_mapping[i] = MarketRegime.HIGH_VOLATILITY
                elif avg_return > 0.5:
                    regime_mapping[i] = MarketRegime.BULL_MARKET
                elif avg_return < -0.5:
                    regime_mapping[i] = MarketRegime.BEAR_MARKET
                else:
                    regime_mapping[i] = MarketRegime.SIDEWAYS
            
            current_regime = regime_mapping.get(current_cluster, MarketRegime.SIDEWAYS)
            
            # Calculate confidence based on distance to cluster center
            current_point = scaled_features[-1]
            current_center = cluster_centers[current_cluster]
            distance = np.linalg.norm(current_point - current_center)
            confidence = max(0.5, 1.0 - distance / 2.0)  # Simple distance-based confidence
            
            return {
                'regime': current_regime,
                'probability': confidence,
                'confidence': confidence,
                'alternatives': {r: 0.1 for r in MarketRegime if r != current_regime},
                'duration': 20,  # Placeholder
                'expected_change': None,
                'characteristics': {
                    'cluster_id': int(current_cluster),
                    'distance_to_center': distance
                }
            }
            
        except Exception as e:
            self.logger.error(f"Clustering detection failed: {e}")
            raise
    
    async def _map_regimes_to_types(self, regime_labels: np.ndarray, 
                                  features: pd.DataFrame,
                                  regime_means: np.ndarray) -> Dict[int, MarketRegime]:
        """Map regime indices to market regime types."""
        mapping = {}
        
        for i, mean in enumerate(regime_means):
            # Analyze regime characteristics
            regime_mask = regime_labels == i
            regime_data = features[regime_mask]
            
            if len(regime_data) > 0:
                avg_return = regime_data['returns'].mean() if 'returns' in regime_data else 0
                avg_vol = regime_data['volatility'].mean() if 'volatility' in regime_data else 0
                
                # Classification logic
                if avg_vol > 0.25:  # High volatility
                    if avg_return < -0.01:
                        mapping[i] = MarketRegime.CRISIS
                    else:
                        mapping[i] = MarketRegime.HIGH_VOLATILITY
                elif avg_return > 0.005:  # Positive returns
                    mapping[i] = MarketRegime.BULL_MARKET
                elif avg_return < -0.005:  # Negative returns
                    mapping[i] = MarketRegime.BEAR_MARKET
                else:
                    mapping[i] = MarketRegime.SIDEWAYS
            else:
                mapping[i] = MarketRegime.SIDEWAYS
        
        return mapping
    
    def _calculate_regime_duration(self, regime_labels: np.ndarray) -> int:
        """Calculate how long current regime has lasted."""
        if len(regime_labels) == 0:
            return 0
        
        current_regime = regime_labels[-1]
        duration = 1
        
        # Count consecutive days in current regime
        for i in range(len(regime_labels) - 2, -1, -1):
            if regime_labels[i] == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    async def _calculate_regime_characteristics(self, recent_data: pd.DataFrame, 
                                             regime: MarketRegime) -> Dict[str, float]:
        """Calculate characteristics of current regime."""
        characteristics = {}
        
        if 'returns' in recent_data.columns:
            characteristics['avg_return'] = recent_data['returns'].mean()
            characteristics['return_volatility'] = recent_data['returns'].std()
        
        if 'volatility' in recent_data.columns:
            characteristics['avg_volatility'] = recent_data['volatility'].mean()
        
        if 'momentum' in recent_data.columns:
            characteristics['avg_momentum'] = recent_data['momentum'].mean()
        
        if 'rsi' in recent_data.columns:
            characteristics['avg_rsi'] = recent_data['rsi'].mean()
        
        return characteristics
    
    def _estimate_regime_change_date(self, current_duration: int) -> Optional[datetime]:
        """Estimate when regime might change based on historical patterns."""
        # Historical average regime durations (simplified)
        avg_durations = {
            MarketRegime.BULL_MARKET: 180,
            MarketRegime.BEAR_MARKET: 90,
            MarketRegime.SIDEWAYS: 60,
            MarketRegime.HIGH_VOLATILITY: 30,
            MarketRegime.LOW_VOLATILITY: 120
        }
        
        # Use default if regime not in mapping
        avg_duration = 90
        
        # Estimate change probability increases after half the average duration
        if current_duration > avg_duration / 2:
            days_to_change = max(7, avg_duration - current_duration)
            return datetime.now() + timedelta(days=days_to_change)
        
        return None
    
    async def _analyze_regime_transitions(self, features: pd.DataFrame) -> List[RegimeTransition]:
        """Analyze historical regime transitions."""
        # Simplified - in practice would use stored regime history
        transitions = []
        
        # Mock transition for demonstration
        if len(features) > 50:
            mock_transition = RegimeTransition(
                from_regime=MarketRegime.SIDEWAYS,
                to_regime=MarketRegime.BULL_MARKET,
                transition_date=datetime.now() - timedelta(days=30),
                transition_probability=0.75,
                confidence_score=0.80,
                duration_from=45,
                expected_duration_to=120,
                regime_features={'volatility': 0.20, 'momentum': 0.05},
                economic_indicators={'sentiment': 0.6}
            )
            transitions.append(mock_transition)
        
        return transitions
    
    async def _forecast_regime_changes(self, features: pd.DataFrame, 
                                     current_result: Dict[str, Any]) -> Dict[MarketRegime, float]:
        """Forecast probability of regime changes."""
        # Simplified forecasting model
        current_regime = current_result['regime']
        
        # Base transition probabilities
        transition_matrix = {
            MarketRegime.BULL_MARKET: {
                MarketRegime.SIDEWAYS: 0.3,
                MarketRegime.BEAR_MARKET: 0.1,
                MarketRegime.HIGH_VOLATILITY: 0.2
            },
            MarketRegime.BEAR_MARKET: {
                MarketRegime.RECOVERY: 0.4,
                MarketRegime.SIDEWAYS: 0.3,
                MarketRegime.CRISIS: 0.1
            },
            MarketRegime.SIDEWAYS: {
                MarketRegime.BULL_MARKET: 0.35,
                MarketRegime.BEAR_MARKET: 0.25,
                MarketRegime.HIGH_VOLATILITY: 0.15
            }
        }
        
        # Get base probabilities
        base_probs = transition_matrix.get(current_regime, {})
        
        # Adjust based on current conditions
        adjusted_probs = {}
        for regime, prob in base_probs.items():
            # Simple adjustment based on current volatility
            if 'volatility' in features.columns:
                current_vol = features['volatility'].iloc[-1]
                if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS] and current_vol > 0.3:
                    prob *= 1.5  # Increase probability of high vol regimes
                elif regime in [MarketRegime.LOW_VOLATILITY] and current_vol < 0.15:
                    prob *= 1.3
            
            adjusted_probs[regime] = min(prob, 0.8)  # Cap at 80%
        
        return adjusted_probs
    
    def _update_regime_history(self, analysis: RegimeAnalysis):
        """Update regime history with new analysis."""
        self.regime_history.append({
            'timestamp': analysis.analysis_timestamp,
            'regime': analysis.current_regime,
            'probability': analysis.regime_probability,
            'confidence': analysis.regime_confidence
        })
        
        # Keep only recent history
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        self.current_regime = analysis.current_regime
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of regime detection results."""
        if not self.regime_history:
            return {'message': 'No regime history available'}
        
        recent_regimes = [entry['regime'] for entry in self.regime_history[-30:]]  # Last 30 analyses
        regime_counts = {}
        
        for regime in recent_regimes:
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
        
        return {
            'current_regime': self.current_regime.value if self.current_regime else None,
            'regime_history_length': len(self.regime_history),
            'recent_regime_distribution': regime_counts,
            'most_common_recent_regime': max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else None
        }