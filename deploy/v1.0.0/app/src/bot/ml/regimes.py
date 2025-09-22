"""
Market Regime Detection for Adaptive Trading Strategies.

This module provides sophisticated market regime detection capabilities
using multiple methodologies:

- Hidden Markov Models (HMM) for regime identification
- Volatility-based regime detection
- Trend-based regime classification
- Multi-factor regime models
- Real-time regime switching detection
- Regime persistence analysis

The regime detection system enables adaptive strategies that can
adjust parameters based on market conditions, improving robustness
and performance across different market environments.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats, cluster
from sklearn.cluster import KMeans, GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logging import TradingLogger


@dataclass
class RegimeInfo:
    """Information about a market regime."""
    
    regime_id: int
    name: str
    description: str
    characteristics: Dict[str, float]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration: Optional[timedelta] = None
    probability: float = 0.0


class HMMRegimeDetector:
    """
    Hidden Markov Model based regime detection.
    
    Uses HMM to identify latent market states based on observable
    market features like returns, volatility, and volume.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("HMMRegimeDetector")
        
        self.model = None
        self.scaler = StandardScaler()
        self.regime_info = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for HMM regime detection."""
        return {
            'n_regimes': 3,
            'covariance_type': 'full',  # full, diag, spherical, tied
            'n_iter': 1000,
            'tol': 1e-6,
            'random_state': 42,
            'features': ['returns', 'volatility', 'volume_ratio'],
            'lookback_window': 252,  # Days for regime probability calculation
            'min_regime_duration': 5,  # Minimum days for regime persistence
        }
    
    def fit(self, data: pd.DataFrame) -> 'HMMRegimeDetector':
        """
        Fit HMM regime detection model.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting HMM with {self.config['n_regimes']} regimes")
        
        # Prepare features
        features = self._prepare_features(data)
        
        if features.empty:
            raise ValueError("No valid features for regime detection")
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.config['n_regimes'],
            covariance_type=self.config['covariance_type'],
            n_iter=self.config['n_iter'],
            tol=self.config['tol'],
            random_state=self.config['random_state']
        )
        
        self.model.fit(scaled_features)
        
        # Predict regimes for training data
        regime_sequence = self.model.predict(scaled_features)
        regime_probs = self.model.predict_proba(scaled_features)
        
        # Analyze regime characteristics
        self._analyze_regime_characteristics(data, features, regime_sequence)
        
        self.logger.info("HMM regime detection model fitted successfully")
        
        return self
    
    def predict_regimes(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Predict market regimes for new data.
        
        Args:
            data: Market data
            
        Returns:
            Tuple of (regime_series, regime_probabilities_dataframe)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        features = self._prepare_features(data)
        
        if features.empty:
            return pd.Series(index=data.index), pd.DataFrame(index=data.index)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Predict regimes
        regime_sequence = self.model.predict(scaled_features)
        regime_probs = self.model.predict_proba(scaled_features)
        
        # Create result series and dataframes
        regime_series = pd.Series(regime_sequence, index=features.index)
        
        prob_columns = [f'regime_{i}_prob' for i in range(self.config['n_regimes'])]
        regime_probs_df = pd.DataFrame(
            regime_probs, 
            index=features.index, 
            columns=prob_columns
        )
        
        return regime_series, regime_probs_df
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime detection."""
        features = pd.DataFrame(index=data.index)
        
        # Returns
        if 'returns' in self.config['features']:
            features['returns'] = data['close'].pct_change()
        
        # Volatility
        if 'volatility' in self.config['features']:
            returns = data['close'].pct_change()
            features['volatility'] = returns.rolling(window=20).std()
        
        # Volume ratio
        if 'volume_ratio' in self.config['features'] and 'volume' in data.columns:
            vol_ma = data['volume'].rolling(window=20).mean()
            features['volume_ratio'] = data['volume'] / vol_ma
        
        # Additional features
        if 'rsi' in self.config['features']:
            features['rsi'] = self._calculate_rsi(data['close'])
        
        if 'macd' in self.config['features']:
            features['macd'] = self._calculate_macd(data['close'])
        
        # Clean features
        features = features.dropna()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _analyze_regime_characteristics(
        self, 
        data: pd.DataFrame, 
        features: pd.DataFrame, 
        regime_sequence: np.ndarray
    ):
        """Analyze characteristics of each regime."""
        self.regime_info = {}
        
        returns = data['close'].pct_change().reindex(features.index)
        
        for regime_id in range(self.config['n_regimes']):
            regime_mask = regime_sequence == regime_id
            regime_returns = returns[regime_mask]
            regime_features = features[regime_mask]
            
            # Calculate regime characteristics
            characteristics = {
                'mean_return': regime_returns.mean() * 252,  # Annualized
                'volatility': regime_returns.std() * np.sqrt(252),  # Annualized
                'sharpe_ratio': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                'skewness': regime_returns.skew(),
                'kurtosis': regime_returns.kurtosis(),
                'max_drawdown': self._calculate_max_drawdown(regime_returns),
                'frequency': regime_mask.sum() / len(regime_mask),
                'avg_duration': self._calculate_avg_duration(regime_mask),
            }
            
            # Add feature characteristics
            for feature in regime_features.columns:
                characteristics[f'avg_{feature}'] = regime_features[feature].mean()
                characteristics[f'std_{feature}'] = regime_features[feature].std()
            
            # Determine regime name based on characteristics
            regime_name = self._classify_regime(characteristics)
            
            self.regime_info[regime_id] = RegimeInfo(
                regime_id=regime_id,
                name=regime_name,
                description=self._describe_regime(characteristics),
                characteristics=characteristics
            )
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_avg_duration(self, regime_mask: np.ndarray) -> float:
        """Calculate average duration of regime periods."""
        durations = []
        current_duration = 0
        
        for i, is_regime in enumerate(regime_mask):
            if is_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Add final duration if regime continues to end
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def _classify_regime(self, characteristics: Dict[str, float]) -> str:
        """Classify regime based on characteristics."""
        mean_return = characteristics.get('mean_return', 0)
        volatility = characteristics.get('volatility', 0)
        
        if mean_return > 0.05 and volatility < 0.15:
            return "Bull Market"
        elif mean_return < -0.05 and volatility > 0.25:
            return "Bear Market" 
        elif volatility > 0.3:
            return "High Volatility"
        elif volatility < 0.1:
            return "Low Volatility"
        else:
            return "Neutral Market"
    
    def _describe_regime(self, characteristics: Dict[str, float]) -> str:
        """Generate description of regime characteristics."""
        mean_return = characteristics.get('mean_return', 0)
        volatility = characteristics.get('volatility', 0)
        sharpe_ratio = characteristics.get('sharpe_ratio', 0)
        
        return (
            f"Mean return: {mean_return:.2%}, "
            f"Volatility: {volatility:.2%}, "
            f"Sharpe ratio: {sharpe_ratio:.2f}"
        )


class VolatilityRegimeDetector:
    """
    Volatility-based regime detection using threshold or clustering methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("VolatilityRegimeDetector")
        
        self.thresholds = None
        self.regime_info = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for volatility regime detection."""
        return {
            'method': 'quantile',  # quantile, kmeans, gaussian_mixture
            'n_regimes': 3,
            'volatility_window': 20,
            'quantiles': [0.33, 0.67],  # For quantile method
            'min_regime_duration': 3,
        }
    
    def fit(self, data: pd.DataFrame) -> 'VolatilityRegimeDetector':
        """Fit volatility regime detection model."""
        self.logger.info(f"Fitting volatility regime detector with {self.config['method']} method")
        
        # Calculate volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.config['volatility_window']).std() * np.sqrt(252)
        volatility = volatility.dropna()
        
        if self.config['method'] == 'quantile':
            self.thresholds = volatility.quantile(self.config['quantiles']).values
            
        elif self.config['method'] == 'kmeans':
            kmeans = KMeans(n_clusters=self.config['n_regimes'], random_state=42)
            kmeans.fit(volatility.values.reshape(-1, 1))
            self.thresholds = np.sort(kmeans.cluster_centers_.flatten())
            
        elif self.config['method'] == 'gaussian_mixture':
            gmm = GaussianMixture(n_components=self.config['n_regimes'], random_state=42)
            gmm.fit(volatility.values.reshape(-1, 1))
            self.thresholds = np.sort(gmm.means_.flatten())
        
        # Analyze regime characteristics
        regimes = self._classify_volatility(volatility)
        self._analyze_volatility_regimes(data, volatility, regimes)
        
        return self
    
    def predict_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Predict volatility regimes."""
        if self.thresholds is None:
            raise ValueError("Model must be fitted before prediction")
        
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.config['volatility_window']).std() * np.sqrt(252)
        
        regimes = self._classify_volatility(volatility)
        
        return regimes
    
    def _classify_volatility(self, volatility: pd.Series) -> pd.Series:
        """Classify volatility into regimes."""
        regimes = pd.Series(0, index=volatility.index)
        
        for i, threshold in enumerate(self.thresholds):
            regimes[volatility > threshold] = i + 1
        
        return regimes
    
    def _analyze_volatility_regimes(
        self, 
        data: pd.DataFrame, 
        volatility: pd.Series, 
        regimes: pd.Series
    ):
        """Analyze characteristics of volatility regimes."""
        returns = data['close'].pct_change().reindex(volatility.index)
        
        for regime_id in regimes.unique():
            if pd.isna(regime_id):
                continue
                
            regime_mask = regimes == regime_id
            regime_returns = returns[regime_mask]
            regime_vol = volatility[regime_mask]
            
            characteristics = {
                'mean_return': regime_returns.mean() * 252,
                'volatility': regime_vol.mean(),
                'frequency': regime_mask.sum() / len(regime_mask),
            }
            
            if regime_id == 0:
                regime_name = "Low Volatility"
            elif regime_id == len(self.thresholds):
                regime_name = "High Volatility"
            else:
                regime_name = f"Medium Volatility {int(regime_id)}"
            
            self.regime_info[regime_id] = RegimeInfo(
                regime_id=int(regime_id),
                name=regime_name,
                description=f"Volatility: {characteristics['volatility']:.2%}",
                characteristics=characteristics
            )


class TrendRegimeDetector:
    """
    Trend-based regime detection using moving averages and trend indicators.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("TrendRegimeDetector")
        
    def _default_config(self) -> Dict:
        """Default configuration for trend regime detection."""
        return {
            'short_window': 20,
            'long_window': 60,
            'trend_threshold': 0.02,  # 2% threshold for trend strength
            'min_regime_duration': 5,
        }
    
    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect trend regimes."""
        prices = data['close']
        
        # Calculate moving averages
        ma_short = prices.rolling(window=self.config['short_window']).mean()
        ma_long = prices.rolling(window=self.config['long_window']).mean()
        
        # Calculate trend strength
        trend_strength = (ma_short - ma_long) / ma_long
        
        # Classify regimes
        regimes = pd.Series(1, index=data.index)  # Neutral = 1
        regimes[trend_strength > self.config['trend_threshold']] = 2  # Uptrend
        regimes[trend_strength < -self.config['trend_threshold']] = 0  # Downtrend
        
        # Apply minimum duration filter
        regimes = self._apply_min_duration_filter(regimes)
        
        return regimes
    
    def _apply_min_duration_filter(self, regimes: pd.Series) -> pd.Series:
        """Apply minimum duration filter to reduce noise."""
        filtered_regimes = regimes.copy()
        min_duration = self.config['min_regime_duration']
        
        current_regime = regimes.iloc[0]
        regime_start = 0
        
        for i in range(1, len(regimes)):
            if regimes.iloc[i] != current_regime:
                # Check if previous regime was too short
                if i - regime_start < min_duration:
                    # Extend previous regime
                    filtered_regimes.iloc[regime_start:i] = filtered_regimes.iloc[regime_start-1] if regime_start > 0 else current_regime
                
                current_regime = regimes.iloc[i]
                regime_start = i
        
        return filtered_regimes


class MultiFactorRegimeDetector:
    """
    Multi-factor regime detection combining multiple indicators.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("MultiFactorRegimeDetector")
        
        # Initialize component detectors
        self.hmm_detector = HMMRegimeDetector(self.config.get('hmm', {}))
        self.vol_detector = VolatilityRegimeDetector(self.config.get('volatility', {}))
        self.trend_detector = TrendRegimeDetector(self.config.get('trend', {}))
        
        self.regime_weights = None
        
    def _default_config(self) -> Dict:
        """Default configuration for multi-factor regime detection."""
        return {
            'combination_method': 'weighted_average',  # weighted_average, majority_vote, ensemble
            'weights': {
                'hmm': 0.5,
                'volatility': 0.3,
                'trend': 0.2
            },
            'hmm': {'n_regimes': 3},
            'volatility': {'n_regimes': 3},
            'trend': {}
        }
    
    def fit(self, data: pd.DataFrame) -> 'MultiFactorRegimeDetector':
        """Fit multi-factor regime detection model."""
        self.logger.info("Fitting multi-factor regime detection model")
        
        # Fit individual detectors
        self.hmm_detector.fit(data)
        self.vol_detector.fit(data)
        
        # No fitting needed for trend detector
        
        self.logger.info("Multi-factor regime detection model fitted")
        
        return self
    
    def predict_regimes(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Predict regimes using multi-factor approach.
        
        Returns:
            Tuple of (combined_regimes, individual_predictions)
        """
        # Get predictions from individual detectors
        hmm_regimes, hmm_probs = self.hmm_detector.predict_regimes(data)
        vol_regimes = self.vol_detector.predict_regimes(data)
        trend_regimes = self.trend_detector.detect_regimes(data)
        
        # Align all predictions
        common_index = hmm_regimes.index.intersection(vol_regimes.index).intersection(trend_regimes.index)
        
        hmm_regimes = hmm_regimes.reindex(common_index)
        vol_regimes = vol_regimes.reindex(common_index)
        trend_regimes = trend_regimes.reindex(common_index)
        
        # Combine predictions
        if self.config['combination_method'] == 'weighted_average':
            combined_regimes = self._weighted_average_combination(
                hmm_regimes, vol_regimes, trend_regimes
            )
        elif self.config['combination_method'] == 'majority_vote':
            combined_regimes = self._majority_vote_combination(
                hmm_regimes, vol_regimes, trend_regimes
            )
        else:
            combined_regimes = hmm_regimes  # Default to HMM
        
        # Create individual predictions dataframe
        individual_predictions = pd.DataFrame({
            'hmm_regime': hmm_regimes,
            'volatility_regime': vol_regimes,
            'trend_regime': trend_regimes,
            'combined_regime': combined_regimes
        }, index=common_index)
        
        return combined_regimes, individual_predictions
    
    def _weighted_average_combination(
        self, 
        hmm_regimes: pd.Series, 
        vol_regimes: pd.Series, 
        trend_regimes: pd.Series
    ) -> pd.Series:
        """Combine regimes using weighted average."""
        weights = self.config['weights']
        
        # Normalize regimes to 0-1 scale
        hmm_norm = hmm_regimes / hmm_regimes.max() if hmm_regimes.max() > 0 else hmm_regimes
        vol_norm = vol_regimes / vol_regimes.max() if vol_regimes.max() > 0 else vol_regimes
        trend_norm = trend_regimes / trend_regimes.max() if trend_regimes.max() > 0 else trend_regimes
        
        # Calculate weighted combination
        combined = (
            weights['hmm'] * hmm_norm +
            weights['volatility'] * vol_norm +
            weights['trend'] * trend_norm
        )
        
        # Convert back to discrete regimes (0, 1, 2)
        combined_regimes = pd.cut(combined, bins=3, labels=[0, 1, 2]).astype(int)
        
        return combined_regimes
    
    def _majority_vote_combination(
        self, 
        hmm_regimes: pd.Series, 
        vol_regimes: pd.Series, 
        trend_regimes: pd.Series
    ) -> pd.Series:
        """Combine regimes using majority voting."""
        # Stack all regime predictions
        regime_df = pd.DataFrame({
            'hmm': hmm_regimes,
            'vol': vol_regimes,
            'trend': trend_regimes
        })
        
        # Find mode (most common) regime for each timestamp
        combined_regimes = regime_df.mode(axis=1)[0]
        
        return combined_regimes


class RegimeAnalyzer:
    """
    Comprehensive regime analysis and reporting.
    """
    
    def __init__(self):
        self.logger = TradingLogger("RegimeAnalyzer")
    
    def analyze_regime_performance(
        self, 
        data: pd.DataFrame, 
        regimes: pd.Series, 
        strategy_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Analyze performance across different regimes."""
        analysis = {}
        
        returns = data['close'].pct_change()
        
        for regime_id in regimes.unique():
            if pd.isna(regime_id):
                continue
            
            regime_mask = regimes == regime_id
            regime_returns = returns[regime_mask]
            
            regime_analysis = {
                'frequency': regime_mask.sum() / len(regimes),
                'mean_return': regime_returns.mean() * 252,
                'volatility': regime_returns.std() * np.sqrt(252),
                'sharpe_ratio': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(regime_returns),
                'positive_days': (regime_returns > 0).sum() / len(regime_returns),
            }
            
            # Strategy performance in regime (if provided)
            if strategy_returns is not None:
                strategy_regime_returns = strategy_returns[regime_mask]
                regime_analysis.update({
                    'strategy_return': strategy_regime_returns.mean() * 252,
                    'strategy_volatility': strategy_regime_returns.std() * np.sqrt(252),
                    'strategy_sharpe': strategy_regime_returns.mean() / strategy_regime_returns.std() * np.sqrt(252) if strategy_regime_returns.std() > 0 else 0,
                })
            
            analysis[f'regime_{regime_id}'] = regime_analysis
        
        return analysis
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def generate_regime_report(
        self, 
        regime_analysis: Dict[str, Any], 
        detector_type: str = "Multi-Factor"
    ) -> str:
        """Generate comprehensive regime analysis report."""
        report = f"""
{detector_type} Regime Analysis Report
{'='*60}

"""
        
        for regime_name, analysis in regime_analysis.items():
            report += f"""
{regime_name.replace('_', ' ').title()}:
- Frequency: {analysis['frequency']:.1%}
- Mean Return: {analysis['mean_return']:.1%}
- Volatility: {analysis['volatility']:.1%}  
- Sharpe Ratio: {analysis['sharpe_ratio']:.2f}
- Max Drawdown: {analysis['max_drawdown']:.1%}
- Positive Days: {analysis['positive_days']:.1%}
"""
            
            if 'strategy_return' in analysis:
                report += f"""- Strategy Return: {analysis['strategy_return']:.1%}
- Strategy Sharpe: {analysis['strategy_sharpe']:.2f}
"""
        
        return report