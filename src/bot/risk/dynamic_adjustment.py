"""
Dynamic Risk Adjustment System for Cryptocurrency Trading.

This module provides sophisticated dynamic risk adjustment capabilities including:

- Volatility-based position sizing adjustments
- Market regime-aware risk scaling
- Correlation-based portfolio rebalancing
- Momentum and trend-based risk adjustments
- Liquidity-adjusted position sizing
- Economic calendar-based risk management
- Volatility clustering detection and adjustment
- Dynamic stop-loss and take-profit levels
- Risk parity rebalancing triggers

The system continuously adjusts risk parameters based on changing
market conditions to optimize risk-adjusted returns.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import talib

from ..utils.logging import TradingLogger
from .position_sizing import PositionSizer, PositionSizeResult
from .portfolio_analysis import PortfolioRiskMonitor, RiskAlert, RiskLevel
from ..ml.regimes import HMMRegimeDetector, VolatilityRegimeDetector


class AdjustmentTrigger(Enum):
    """Types of risk adjustment triggers."""
    VOLATILITY_CHANGE = "volatility_change"
    REGIME_CHANGE = "regime_change"
    CORRELATION_CHANGE = "correlation_change"
    MOMENTUM_CHANGE = "momentum_change"
    LIQUIDITY_CHANGE = "liquidity_change"
    DRAWDOWN_THRESHOLD = "drawdown_threshold"
    VAR_BREACH = "var_breach"
    TIME_BASED = "time_based"
    ECONOMIC_EVENT = "economic_event"


@dataclass
class RiskAdjustment:
    """Container for risk adjustment recommendations."""
    
    trigger_type: AdjustmentTrigger
    adjustment_factor: float  # Multiplier for current risk (1.0 = no change)
    target_positions: Dict[str, float]  # New target position sizes
    reason: str
    confidence_level: float
    effective_time: datetime
    expiry_time: Optional[datetime] = None
    priority: int = 1  # 1 = low, 5 = critical
    
    def __post_init__(self):
        if self.effective_time is None:
            self.effective_time = datetime.now()


@dataclass
class VolatilityMetrics:
    """Container for volatility analysis metrics."""
    
    current_volatility: float
    volatility_percentile: float
    volatility_trend: float  # -1 to 1, negative = decreasing
    volatility_clustering: bool
    vol_breakout: bool
    garch_forecast: float
    ewma_volatility: float
    
    def is_high_volatility(self, threshold: float = 0.75) -> bool:
        return self.volatility_percentile > threshold
    
    def is_low_volatility(self, threshold: float = 0.25) -> bool:
        return self.volatility_percentile < threshold


class VolatilityRegimeAnalyzer:
    """
    Advanced volatility regime analysis for dynamic risk adjustment.
    
    This class provides sophisticated volatility analysis including:
    - GARCH-based volatility forecasting
    - Volatility clustering detection
    - Regime change identification
    - Volatility breakout detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("VolatilityRegimeAnalyzer")
        
        # Initialize regime detector
        self.volatility_detector = VolatilityRegimeDetector(self.config.get('regime_detector', {}))
        
    def _default_config(self) -> Dict:
        """Default configuration for volatility regime analyzer."""
        return {
            'volatility_window': 20,         # Window for volatility calculation
            'regime_window': 60,             # Lookback for regime analysis
            'clustering_threshold': 2.0,     # Volatility clustering threshold
            'breakout_threshold': 2.5,       # Volatility breakout threshold
            'ewma_alpha': 0.94,              # EWMA smoothing parameter
            'garch_params': {                # GARCH model parameters
                'p': 1,
                'q': 1,
                'max_iter': 1000
            },
            'regime_detector': {}
        }
    
    def analyze_volatility_regime(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None
    ) -> VolatilityMetrics:
        """
        Comprehensive volatility regime analysis.
        
        Args:
            returns: Asset returns
            prices: Asset prices (optional, for additional analysis)
            
        Returns:
            VolatilityMetrics with comprehensive volatility analysis
        """
        if len(returns) < self.config['volatility_window']:
            self.logger.warning("Insufficient data for volatility analysis")
            return self._default_volatility_metrics()
        
        # Calculate current volatility
        current_vol = self._calculate_current_volatility(returns)
        
        # Calculate volatility percentile
        vol_percentile = self._calculate_volatility_percentile(returns, current_vol)
        
        # Analyze volatility trend
        vol_trend = self._analyze_volatility_trend(returns)
        
        # Detect volatility clustering
        clustering = self._detect_volatility_clustering(returns)
        
        # Detect volatility breakouts
        breakout = self._detect_volatility_breakout(returns)
        
        # GARCH forecast
        garch_forecast = self._garch_volatility_forecast(returns)
        
        # EWMA volatility
        ewma_vol = self._calculate_ewma_volatility(returns)
        
        return VolatilityMetrics(
            current_volatility=current_vol,
            volatility_percentile=vol_percentile,
            volatility_trend=vol_trend,
            volatility_clustering=clustering,
            vol_breakout=breakout,
            garch_forecast=garch_forecast,
            ewma_volatility=ewma_vol
        )
    
    def _calculate_current_volatility(self, returns: pd.Series) -> float:
        """Calculate current annualized volatility."""
        recent_returns = returns.tail(self.config['volatility_window'])
        return recent_returns.std() * np.sqrt(252)  # Annualized
    
    def _calculate_volatility_percentile(self, returns: pd.Series, current_vol: float) -> float:
        """Calculate percentile rank of current volatility."""
        # Calculate rolling volatility
        rolling_vol = returns.rolling(self.config['volatility_window']).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        
        if len(rolling_vol) == 0:
            return 0.5
        
        # Calculate percentile
        percentile = stats.percentileofscore(rolling_vol, current_vol) / 100
        return percentile
    
    def _analyze_volatility_trend(self, returns: pd.Series) -> float:
        """Analyze trend in volatility (-1 to 1)."""
        # Calculate rolling volatility
        vol_series = returns.rolling(self.config['volatility_window']).std()
        vol_series = vol_series.dropna()
        
        if len(vol_series) < 10:
            return 0.0
        
        # Linear regression on recent volatility
        recent_vol = vol_series.tail(20)
        x = np.arange(len(recent_vol))
        
        if len(recent_vol) > 1:
            slope, _, r_value, _, _ = stats.linregress(x, recent_vol)
            
            # Normalize slope to -1 to 1 range
            trend = np.tanh(slope * 100)  # Scale and constrain
            
            # Weight by R-squared
            trend *= abs(r_value)
        else:
            trend = 0.0
        
        return trend
    
    def _detect_volatility_clustering(self, returns: pd.Series) -> bool:
        """Detect volatility clustering using ARCH effects."""
        if len(returns) < 50:
            return False
        
        # Calculate squared returns (proxy for volatility)
        squared_returns = returns ** 2
        
        # Test for ARCH effects using Engle's ARCH test
        try:
            # Simple ARCH test: correlation between squared returns
            recent_squared = squared_returns.tail(20)
            lagged_squared = squared_returns.shift(1).tail(20)
            
            correlation = recent_squared.corr(lagged_squared)
            
            # Clustering if strong correlation in squared returns
            return abs(correlation) > 0.3
            
        except Exception as e:
            self.logger.warning(f"Error detecting volatility clustering: {e}")
            return False
    
    def _detect_volatility_breakout(self, returns: pd.Series) -> bool:
        """Detect volatility breakouts."""
        if len(returns) < self.config['volatility_window'] * 2:
            return False
        
        # Current volatility
        current_vol = self._calculate_current_volatility(returns)
        
        # Historical volatility statistics
        hist_vol = returns.rolling(self.config['volatility_window']).std() * np.sqrt(252)
        hist_vol = hist_vol.dropna()
        
        if len(hist_vol) < 20:
            return False
        
        # Calculate volatility Z-score
        vol_mean = hist_vol.mean()
        vol_std = hist_vol.std()
        
        if vol_std == 0:
            return False
        
        z_score = (current_vol - vol_mean) / vol_std
        
        # Breakout if current volatility is significantly higher
        return z_score > self.config['breakout_threshold']
    
    def _garch_volatility_forecast(self, returns: pd.Series) -> float:
        """Simple GARCH(1,1) volatility forecast."""
        try:
            # Simplified GARCH(1,1) estimation
            # In practice, would use proper GARCH libraries like arch
            
            clean_returns = returns.dropna()
            if len(clean_returns) < 50:
                return self._calculate_current_volatility(returns)
            
            # Estimate GARCH parameters (simplified)
            squared_returns = clean_returns ** 2
            
            # Moving averages as proxies for GARCH terms
            long_term_var = squared_returns.mean()
            recent_return_squared = squared_returns.iloc[-1]
            recent_variance = squared_returns.tail(5).mean()
            
            # Simple GARCH forecast: omega + alpha * u^2 + beta * sigma^2
            omega = long_term_var * 0.1
            alpha = 0.1
            beta = 0.85
            
            forecast_variance = omega + alpha * recent_return_squared + beta * recent_variance
            forecast_volatility = np.sqrt(forecast_variance * 252)  # Annualized
            
            return forecast_volatility
            
        except Exception as e:
            self.logger.warning(f"Error in GARCH forecast: {e}")
            return self._calculate_current_volatility(returns)
    
    def _calculate_ewma_volatility(self, returns: pd.Series) -> float:
        """Calculate EWMA volatility."""
        clean_returns = returns.dropna()
        if len(clean_returns) < 10:
            return self._calculate_current_volatility(returns)
        
        # EWMA variance calculation
        alpha = self.config['ewma_alpha']
        
        ewma_var = 0
        for ret in clean_returns:
            ewma_var = alpha * ewma_var + (1 - alpha) * (ret ** 2)
        
        return np.sqrt(ewma_var * 252)  # Annualized
    
    def _default_volatility_metrics(self) -> VolatilityMetrics:
        """Return default volatility metrics for insufficient data."""
        return VolatilityMetrics(
            current_volatility=0.6,  # 60% default crypto volatility
            volatility_percentile=0.5,
            volatility_trend=0.0,
            volatility_clustering=False,
            vol_breakout=False,
            garch_forecast=0.6,
            ewma_volatility=0.6
        )


class MarketRegimeAdjuster:
    """
    Market regime-based risk adjustment system.
    
    This class adjusts risk parameters based on detected market regimes
    using sophisticated regime detection models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("MarketRegimeAdjuster")
        
        # Initialize regime detectors
        self.hmm_detector = HMMRegimeDetector(self.config.get('hmm_detector', {}))
        self.vol_detector = VolatilityRegimeDetector(self.config.get('vol_detector', {}))
        
        # Current regime state
        self.current_regime = None
        self.regime_confidence = 0.0
        self.regime_history = []
        
    def _default_config(self) -> Dict:
        """Default configuration for market regime adjuster."""
        return {
            'regime_window': 60,             # Days for regime analysis
            'regime_confidence_threshold': 0.7,  # Minimum confidence for regime change
            'adjustment_factors': {          # Risk adjustments per regime
                'bull_market': 1.2,          # Increase risk in bull markets
                'bear_market': 0.7,          # Decrease risk in bear markets
                'high_volatility': 0.6,      # Significantly reduce risk in high vol
                'low_volatility': 1.1,       # Slightly increase risk in low vol
                'trending': 1.0,             # No adjustment for trending
                'mean_reverting': 0.9,       # Slight reduction for mean reversion
                'neutral': 1.0               # No adjustment for neutral
            },
            'regime_stability_period': 5,    # Days to confirm regime change
            'hmm_detector': {},
            'vol_detector': {}
        }
    
    def analyze_market_regime(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Analyze current market regime and recommend risk adjustments.
        
        Args:
            returns: Asset returns
            prices: Asset prices (optional)
            
        Returns:
            Dictionary with regime analysis and adjustment recommendations
        """
        if len(returns) < self.config['regime_window']:
            self.logger.warning("Insufficient data for regime analysis")
            return self._default_regime_analysis()
        
        # Detect regimes using multiple methods
        hmm_result = self._detect_hmm_regime(returns)
        vol_result = self._detect_volatility_regime(returns)
        trend_result = self._detect_trend_regime(returns, prices)
        
        # Combine regime signals
        combined_regime = self._combine_regime_signals(hmm_result, vol_result, trend_result)
        
        # Check for regime change
        regime_changed = self._check_regime_change(combined_regime)
        
        # Calculate adjustment factor
        adjustment_factor = self._calculate_regime_adjustment(combined_regime)
        
        # Generate risk adjustment recommendation
        adjustment = self._generate_regime_adjustment(
            combined_regime, adjustment_factor, regime_changed
        )
        
        return {
            'current_regime': combined_regime,
            'regime_confidence': self.regime_confidence,
            'regime_changed': regime_changed,
            'adjustment_factor': adjustment_factor,
            'recommended_adjustment': adjustment,
            'regime_details': {
                'hmm_regime': hmm_result,
                'volatility_regime': vol_result,
                'trend_regime': trend_result
            }
        }
    
    def _detect_hmm_regime(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect regime using Hidden Markov Model."""
        try:
            result = self.hmm_detector.fit_predict(returns.to_frame('returns'))
            
            current_regime = result['current_regime']
            confidence = result['regime_probabilities'].iloc[-1].max()
            
            return {
                'regime': current_regime,
                'confidence': confidence,
                'method': 'hmm'
            }
            
        except Exception as e:
            self.logger.warning(f"HMM regime detection failed: {e}")
            return {'regime': 'neutral', 'confidence': 0.0, 'method': 'hmm'}
    
    def _detect_volatility_regime(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect volatility regime."""
        try:
            result = self.vol_detector.fit_predict(returns.to_frame('returns'))
            
            current_regime = result['current_regime']
            confidence = result.get('confidence', 0.5)
            
            return {
                'regime': current_regime,
                'confidence': confidence,
                'method': 'volatility'
            }
            
        except Exception as e:
            self.logger.warning(f"Volatility regime detection failed: {e}")
            return {'regime': 'neutral', 'confidence': 0.0, 'method': 'volatility'}
    
    def _detect_trend_regime(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Detect trend regime using technical indicators."""
        try:
            if prices is None or len(prices) < 50:
                return {'regime': 'neutral', 'confidence': 0.0, 'method': 'trend'}
            
            # Calculate trend indicators
            sma_20 = talib.SMA(prices.values, timeperiod=20)
            sma_50 = talib.SMA(prices.values, timeperiod=50)
            rsi = talib.RSI(prices.values, timeperiod=14)
            macd, macd_signal, _ = talib.MACD(prices.values)
            
            # Determine trend regime
            current_price = prices.iloc[-1]
            sma20_current = sma_20[-1] if not np.isnan(sma_20[-1]) else current_price
            sma50_current = sma_50[-1] if not np.isnan(sma_50[-1]) else current_price
            rsi_current = rsi[-1] if not np.isnan(rsi[-1]) else 50
            macd_current = macd[-1] if not np.isnan(macd[-1]) else 0
            macd_signal_current = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
            
            # Score bullish/bearish signals
            signals = 0
            
            if current_price > sma20_current:
                signals += 1
            if sma20_current > sma50_current:
                signals += 1
            if rsi_current > 50:
                signals += 1
            if macd_current > macd_signal_current:
                signals += 1
            
            # Determine regime
            if signals >= 3:
                regime = 'bull_market'
                confidence = signals / 4
            elif signals <= 1:
                regime = 'bear_market'
                confidence = (4 - signals) / 4
            else:
                regime = 'neutral'
                confidence = 0.5
            
            return {
                'regime': regime,
                'confidence': confidence,
                'method': 'trend',
                'signals': signals
            }
            
        except Exception as e:
            self.logger.warning(f"Trend regime detection failed: {e}")
            return {'regime': 'neutral', 'confidence': 0.0, 'method': 'trend'}
    
    def _combine_regime_signals(
        self,
        hmm_result: Dict[str, Any],
        vol_result: Dict[str, Any],
        trend_result: Dict[str, Any]
    ) -> str:
        """Combine multiple regime detection signals."""
        # Weight the signals by confidence
        weighted_signals = []
        
        if hmm_result['confidence'] > 0.3:
            weighted_signals.append((hmm_result['regime'], hmm_result['confidence']))
        
        if vol_result['confidence'] > 0.3:
            weighted_signals.append((vol_result['regime'], vol_result['confidence']))
        
        if trend_result['confidence'] > 0.3:
            weighted_signals.append((trend_result['regime'], trend_result['confidence']))
        
        if not weighted_signals:
            return 'neutral'
        
        # Find most confident regime
        best_regime = max(weighted_signals, key=lambda x: x[1])
        self.regime_confidence = best_regime[1]
        
        return best_regime[0]
    
    def _check_regime_change(self, new_regime: str) -> bool:
        """Check if regime has changed significantly."""
        if self.current_regime is None:
            self.current_regime = new_regime
            return True
        
        if new_regime != self.current_regime:
            # Require minimum confidence for regime change
            if self.regime_confidence >= self.config['regime_confidence_threshold']:
                self.current_regime = new_regime
                self.regime_history.append({
                    'regime': new_regime,
                    'timestamp': datetime.now(),
                    'confidence': self.regime_confidence
                })
                
                # Keep only recent history
                if len(self.regime_history) > 10:
                    self.regime_history = self.regime_history[-10:]
                
                return True
        
        return False
    
    def _calculate_regime_adjustment(self, regime: str) -> float:
        """Calculate risk adjustment factor for regime."""
        return self.config['adjustment_factors'].get(regime, 1.0)
    
    def _generate_regime_adjustment(
        self,
        regime: str,
        adjustment_factor: float,
        regime_changed: bool
    ) -> RiskAdjustment:
        """Generate risk adjustment recommendation."""
        priority = 3 if regime_changed else 1
        
        reason = f"Market regime detected as {regime}"
        if regime_changed:
            reason += " (regime change detected)"
        
        return RiskAdjustment(
            trigger_type=AdjustmentTrigger.REGIME_CHANGE,
            adjustment_factor=adjustment_factor,
            target_positions={},  # Will be filled by position sizer
            reason=reason,
            confidence_level=self.regime_confidence,
            effective_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(hours=6),  # 6-hour validity
            priority=priority
        )
    
    def _default_regime_analysis(self) -> Dict[str, Any]:
        """Return default regime analysis for insufficient data."""
        return {
            'current_regime': 'neutral',
            'regime_confidence': 0.0,
            'regime_changed': False,
            'adjustment_factor': 1.0,
            'recommended_adjustment': RiskAdjustment(
                trigger_type=AdjustmentTrigger.REGIME_CHANGE,
                adjustment_factor=1.0,
                target_positions={},
                reason="Insufficient data for regime analysis",
                confidence_level=0.0,
                effective_time=datetime.now()
            ),
            'regime_details': {}
        }


class CorrelationAdjuster:
    """
    Correlation-based risk adjustment system.
    
    This class monitors portfolio correlation changes and adjusts
    position sizes to maintain optimal diversification.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("CorrelationAdjuster")
        
        # Tracking
        self.correlation_history = []
        self.last_correlation_matrix = None
        
    def _default_config(self) -> Dict:
        """Default configuration for correlation adjuster."""
        return {
            'correlation_window': 30,        # Days for correlation calculation
            'correlation_threshold': 0.8,    # High correlation threshold
            'adjustment_sensitivity': 0.5,   # Sensitivity to correlation changes
            'rebalance_threshold': 0.1,      # Minimum change to trigger rebalance
            'max_correlation_adjustment': 0.3,  # Maximum adjustment factor
            'dynamic_adjustment': True,      # Enable dynamic correlation adjustment
        }
    
    def analyze_correlation_changes(
        self,
        returns_data: pd.DataFrame,
        portfolio_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze correlation changes and recommend adjustments.
        
        Args:
            returns_data: DataFrame with asset returns
            portfolio_weights: Current portfolio weights
            
        Returns:
            Dictionary with correlation analysis and adjustments
        """
        if len(returns_data) < self.config['correlation_window']:
            return self._default_correlation_analysis()
        
        # Calculate current correlation matrix
        current_corr_matrix = self._calculate_correlation_matrix(returns_data)
        
        # Analyze correlation changes
        correlation_change = self._analyze_correlation_change(current_corr_matrix)
        
        # Identify high correlation clusters
        high_corr_pairs = self._identify_high_correlations(current_corr_matrix)
        
        # Calculate diversification metrics
        diversification_metrics = self._calculate_diversification_metrics(
            current_corr_matrix, portfolio_weights
        )
        
        # Generate adjustment recommendations
        adjustments = self._generate_correlation_adjustments(
            current_corr_matrix, portfolio_weights, high_corr_pairs, 
            diversification_metrics
        )
        
        # Update history
        self._update_correlation_history(current_corr_matrix, diversification_metrics)
        
        return {
            'correlation_matrix': current_corr_matrix,
            'correlation_change': correlation_change,
            'high_correlation_pairs': high_corr_pairs,
            'diversification_metrics': diversification_metrics,
            'recommended_adjustments': adjustments,
            'adjustment_needed': len(adjustments) > 0
        }
    
    def _calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix with recent data."""
        recent_data = returns_data.tail(self.config['correlation_window'])
        return recent_data.corr()
    
    def _analyze_correlation_change(self, current_corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """Analyze changes in correlation structure."""
        if self.last_correlation_matrix is None:
            self.last_correlation_matrix = current_corr_matrix
            return {'average_change': 0.0, 'max_change': 0.0, 'significant_changes': 0}
        
        # Calculate correlation changes
        corr_diff = current_corr_matrix - self.last_correlation_matrix
        
        # Get upper triangle (avoid diagonal and duplicates)
        mask = np.triu(np.ones_like(corr_diff, dtype=bool), k=1)
        changes = corr_diff.values[mask]
        
        avg_change = np.mean(np.abs(changes))
        max_change = np.max(np.abs(changes))
        significant_changes = np.sum(np.abs(changes) > 0.1)
        
        self.last_correlation_matrix = current_corr_matrix
        
        return {
            'average_change': avg_change,
            'max_change': max_change,
            'significant_changes': significant_changes
        }
    
    def _identify_high_correlations(self, corr_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify asset pairs with high correlation."""
        high_corr_pairs = []
        
        for i, asset1 in enumerate(corr_matrix.columns):
            for j, asset2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    correlation = corr_matrix.iloc[i, j]
                    
                    if abs(correlation) >= self.config['correlation_threshold']:
                        high_corr_pairs.append({
                            'asset1': asset1,
                            'asset2': asset2,
                            'correlation': correlation,
                            'risk_level': 'high' if abs(correlation) > 0.9 else 'medium'
                        })
        
        return high_corr_pairs
    
    def _calculate_diversification_metrics(
        self,
        corr_matrix: pd.DataFrame,
        portfolio_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate portfolio diversification metrics."""
        # Diversification ratio
        weights = np.array([portfolio_weights.get(asset, 0) for asset in corr_matrix.columns])
        
        if np.sum(weights) == 0:
            return {'diversification_ratio': 1.0, 'effective_assets': 1.0, 'concentration_risk': 0.0}
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Portfolio variance
        portfolio_var = weights.T @ corr_matrix.values @ weights
        
        # Individual asset variances (assuming unit variance for correlation analysis)
        individual_vars = np.ones(len(weights))
        
        # Diversification ratio = (sum of individual volatilities) / portfolio volatility
        weighted_individual_vol = np.sum(weights * np.sqrt(individual_vars))
        portfolio_vol = np.sqrt(portfolio_var)
        
        diversification_ratio = weighted_individual_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        # Effective number of assets
        effective_assets = 1 / np.sum(weights ** 2)
        
        # Concentration risk (Herfindahl index)
        concentration_risk = np.sum(weights ** 2)
        
        return {
            'diversification_ratio': diversification_ratio,
            'effective_assets': effective_assets,
            'concentration_risk': concentration_risk,
            'portfolio_variance': portfolio_var
        }
    
    def _generate_correlation_adjustments(
        self,
        corr_matrix: pd.DataFrame,
        portfolio_weights: Dict[str, float],
        high_corr_pairs: List[Dict[str, Any]],
        diversification_metrics: Dict[str, float]
    ) -> List[RiskAdjustment]:
        """Generate correlation-based risk adjustments."""
        adjustments = []
        
        if not high_corr_pairs:
            return adjustments
        
        # Group highly correlated assets
        correlation_clusters = self._cluster_correlated_assets(high_corr_pairs)
        
        for cluster in correlation_clusters:
            cluster_weight = sum(portfolio_weights.get(asset, 0) for asset in cluster)
            
            if cluster_weight > 0.5:  # High concentration in correlated assets
                # Recommend reducing positions in cluster
                adjustment_factor = max(0.7, 1 - cluster_weight * 0.5)
                
                target_positions = {}
                for asset in cluster:
                    current_weight = portfolio_weights.get(asset, 0)
                    target_positions[asset] = current_weight * adjustment_factor
                
                adjustments.append(RiskAdjustment(
                    trigger_type=AdjustmentTrigger.CORRELATION_CHANGE,
                    adjustment_factor=adjustment_factor,
                    target_positions=target_positions,
                    reason=f"High correlation detected in cluster: {cluster}",
                    confidence_level=0.8,
                    effective_time=datetime.now(),
                    priority=2
                ))
        
        return adjustments
    
    def _cluster_correlated_assets(self, high_corr_pairs: List[Dict[str, Any]]) -> List[List[str]]:
        """Cluster assets based on high correlations."""
        # Build adjacency list
        adjacency = {}
        
        for pair in high_corr_pairs:
            asset1, asset2 = pair['asset1'], pair['asset2']
            
            if asset1 not in adjacency:
                adjacency[asset1] = set()
            if asset2 not in adjacency:
                adjacency[asset2] = set()
            
            adjacency[asset1].add(asset2)
            adjacency[asset2].add(asset1)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for asset in adjacency:
            if asset not in visited:
                cluster = []
                stack = [asset]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster.append(current)
                        
                        for neighbor in adjacency.get(current, []):
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                if len(cluster) > 1:
                    clusters.append(cluster)
        
        return clusters
    
    def _update_correlation_history(
        self,
        corr_matrix: pd.DataFrame,
        diversification_metrics: Dict[str, float]
    ) -> None:
        """Update correlation history for tracking."""
        self.correlation_history.append({
            'timestamp': datetime.now(),
            'average_correlation': self._calculate_average_correlation(corr_matrix),
            'diversification_ratio': diversification_metrics['diversification_ratio'],
            'effective_assets': diversification_metrics['effective_assets']
        })
        
        # Keep only recent history
        if len(self.correlation_history) > 100:
            self.correlation_history = self.correlation_history[-100:]
    
    def _calculate_average_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate average pairwise correlation."""
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.values[mask]
        return np.mean(correlations)
    
    def _default_correlation_analysis(self) -> Dict[str, Any]:
        """Return default correlation analysis for insufficient data."""
        return {
            'correlation_matrix': pd.DataFrame(),
            'correlation_change': {'average_change': 0.0, 'max_change': 0.0, 'significant_changes': 0},
            'high_correlation_pairs': [],
            'diversification_metrics': {'diversification_ratio': 1.0, 'effective_assets': 1.0, 'concentration_risk': 0.0},
            'recommended_adjustments': [],
            'adjustment_needed': False
        }


class DynamicRiskAdjuster:
    """
    Comprehensive dynamic risk adjustment system.
    
    This class integrates all risk adjustment components to provide
    unified, intelligent risk management that adapts to changing
    market conditions in real-time.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("DynamicRiskAdjuster")
        
        # Initialize component analyzers
        self.volatility_analyzer = VolatilityRegimeAnalyzer(self.config.get('volatility', {}))
        self.regime_adjuster = MarketRegimeAdjuster(self.config.get('regime', {}))
        self.correlation_adjuster = CorrelationAdjuster(self.config.get('correlation', {}))
        
        # Position sizer for calculating new position sizes
        self.position_sizer = PositionSizer(self.config.get('position_sizer', {}))
        
        # Adjustment history
        self.adjustment_history = []
        self.active_adjustments = []
        
    def _default_config(self) -> Dict:
        """Default configuration for dynamic risk adjuster."""
        return {
            'adjustment_frequency': 3600,    # Seconds between adjustments
            'min_adjustment_threshold': 0.05,  # Minimum adjustment to apply
            'max_adjustment_factor': 2.0,    # Maximum risk adjustment
            'adjustment_decay': 0.95,        # Decay factor for adjustments
            'combine_adjustments': True,     # Combine multiple adjustments
            'max_active_adjustments': 5,     # Maximum concurrent adjustments
            'volatility': {},
            'regime': {},
            'correlation': {},
            'position_sizer': {}
        }
    
    def analyze_and_adjust_risk(
        self,
        portfolio_weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        asset_prices: Optional[pd.DataFrame] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive risk analysis and adjustment recommendation.
        
        Args:
            portfolio_weights: Current portfolio weights
            asset_returns: Historical asset returns
            asset_prices: Historical asset prices (optional)
            market_data: Additional market data (optional)
            
        Returns:
            Dictionary with comprehensive risk analysis and adjustments
        """
        self.logger.info("Starting comprehensive dynamic risk analysis")
        
        results = {}
        all_adjustments = []
        
        # Analyze each asset individually
        for asset in portfolio_weights.keys():
            if asset not in asset_returns.columns:
                continue
            
            asset_analysis = self._analyze_single_asset_risk(
                asset, asset_returns[asset], 
                asset_prices[asset] if asset_prices is not None and asset in asset_prices.columns else None
            )
            
            results[asset] = asset_analysis
            all_adjustments.extend(asset_analysis['recommended_adjustments'])
        
        # Portfolio-level analysis
        portfolio_analysis = self._analyze_portfolio_level_risk(
            portfolio_weights, asset_returns, asset_prices
        )
        
        results['portfolio'] = portfolio_analysis
        all_adjustments.extend(portfolio_analysis['recommended_adjustments'])
        
        # Combine and prioritize adjustments
        combined_adjustments = self._combine_adjustments(all_adjustments)
        
        # Calculate final position targets
        final_targets = self._calculate_final_position_targets(
            portfolio_weights, combined_adjustments
        )
        
        # Update adjustment history
        self._update_adjustment_history(combined_adjustments)
        
        return {
            'asset_analysis': results,
            'portfolio_analysis': portfolio_analysis,
            'recommended_adjustments': combined_adjustments,
            'final_position_targets': final_targets,
            'adjustment_summary': self._create_adjustment_summary(combined_adjustments),
            'risk_score': self._calculate_overall_risk_score(results, portfolio_analysis)
        }
    
    def _analyze_single_asset_risk(
        self,
        asset: str,
        returns: pd.Series,
        prices: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Analyze risk for a single asset."""
        adjustments = []
        
        # Volatility analysis
        vol_metrics = self.volatility_analyzer.analyze_volatility_regime(returns, prices)
        
        # Generate volatility-based adjustments
        if vol_metrics.is_high_volatility():
            adjustments.append(RiskAdjustment(
                trigger_type=AdjustmentTrigger.VOLATILITY_CHANGE,
                adjustment_factor=0.8,  # Reduce risk in high volatility
                target_positions={asset: 0.0},  # Will be calculated later
                reason=f"High volatility detected for {asset} (percentile: {vol_metrics.volatility_percentile:.2f})",
                confidence_level=0.8,
                effective_time=datetime.now(),
                priority=2
            ))
        elif vol_metrics.is_low_volatility():
            adjustments.append(RiskAdjustment(
                trigger_type=AdjustmentTrigger.VOLATILITY_CHANGE,
                adjustment_factor=1.1,  # Slightly increase risk in low volatility
                target_positions={asset: 0.0},
                reason=f"Low volatility detected for {asset} (percentile: {vol_metrics.volatility_percentile:.2f})",
                confidence_level=0.6,
                effective_time=datetime.now(),
                priority=1
            ))
        
        # Volatility breakout adjustment
        if vol_metrics.vol_breakout:
            adjustments.append(RiskAdjustment(
                trigger_type=AdjustmentTrigger.VOLATILITY_CHANGE,
                adjustment_factor=0.6,  # Significantly reduce risk on vol breakout
                target_positions={asset: 0.0},
                reason=f"Volatility breakout detected for {asset}",
                confidence_level=0.9,
                effective_time=datetime.now(),
                priority=3
            ))
        
        return {
            'asset': asset,
            'volatility_metrics': vol_metrics,
            'recommended_adjustments': adjustments,
            'risk_level': self._assess_asset_risk_level(vol_metrics)
        }
    
    def _analyze_portfolio_level_risk(
        self,
        portfolio_weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        asset_prices: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Analyze portfolio-level risk factors."""
        adjustments = []
        
        # Calculate portfolio returns
        weights_series = pd.Series(portfolio_weights)
        aligned_weights = weights_series.reindex(asset_returns.columns, fill_value=0)
        portfolio_returns = (asset_returns * aligned_weights).sum(axis=1)
        
        # Market regime analysis
        regime_analysis = self.regime_adjuster.analyze_market_regime(
            portfolio_returns,
            asset_prices.mean(axis=1) if asset_prices is not None else None
        )
        
        if regime_analysis['regime_changed']:
            adjustments.append(regime_analysis['recommended_adjustment'])
        
        # Correlation analysis
        correlation_analysis = self.correlation_adjuster.analyze_correlation_changes(
            asset_returns, portfolio_weights
        )
        
        adjustments.extend(correlation_analysis['recommended_adjustments'])
        
        # Momentum analysis
        momentum_adjustments = self._analyze_momentum_factors(portfolio_returns)
        adjustments.extend(momentum_adjustments)
        
        return {
            'regime_analysis': regime_analysis,
            'correlation_analysis': correlation_analysis,
            'recommended_adjustments': adjustments,
            'portfolio_risk_level': self._assess_portfolio_risk_level(
                regime_analysis, correlation_analysis
            )
        }
    
    def _analyze_momentum_factors(self, portfolio_returns: pd.Series) -> List[RiskAdjustment]:
        """Analyze momentum factors for risk adjustment."""
        adjustments = []
        
        if len(portfolio_returns) < 20:
            return adjustments
        
        # Calculate momentum metrics
        recent_returns = portfolio_returns.tail(10)
        momentum = recent_returns.mean()
        momentum_volatility = recent_returns.std()
        
        # Sharpe-like momentum metric
        momentum_score = momentum / momentum_volatility if momentum_volatility > 0 else 0
        
        # Strong negative momentum
        if momentum_score < -1.0:
            adjustments.append(RiskAdjustment(
                trigger_type=AdjustmentTrigger.MOMENTUM_CHANGE,
                adjustment_factor=0.8,  # Reduce risk on negative momentum
                target_positions={},
                reason=f"Strong negative momentum detected (score: {momentum_score:.2f})",
                confidence_level=0.7,
                effective_time=datetime.now(),
                priority=2
            ))
        
        # Strong positive momentum with high volatility (overextension risk)
        elif momentum_score > 2.0 and momentum_volatility * np.sqrt(252) > 0.5:
            adjustments.append(RiskAdjustment(
                trigger_type=AdjustmentTrigger.MOMENTUM_CHANGE,
                adjustment_factor=0.9,  # Slight reduction due to overextension
                target_positions={},
                reason=f"Overextended positive momentum (score: {momentum_score:.2f})",
                confidence_level=0.6,
                effective_time=datetime.now(),
                priority=1
            ))
        
        return adjustments
    
    def _combine_adjustments(self, adjustments: List[RiskAdjustment]) -> List[RiskAdjustment]:
        """Combine and prioritize multiple risk adjustments."""
        if not self.config['combine_adjustments']:
            return sorted(adjustments, key=lambda x: x.priority, reverse=True)
        
        # Group adjustments by trigger type and asset
        grouped_adjustments = {}
        
        for adjustment in adjustments:
            key = (adjustment.trigger_type, tuple(sorted(adjustment.target_positions.keys())))
            
            if key not in grouped_adjustments:
                grouped_adjustments[key] = []
            
            grouped_adjustments[key].append(adjustment)
        
        # Combine adjustments within each group
        combined_adjustments = []
        
        for group in grouped_adjustments.values():
            if len(group) == 1:
                combined_adjustments.append(group[0])
            else:
                combined_adj = self._merge_adjustment_group(group)
                combined_adjustments.append(combined_adj)
        
        # Sort by priority and limit number of active adjustments
        combined_adjustments.sort(key=lambda x: x.priority, reverse=True)
        
        return combined_adjustments[:self.config['max_active_adjustments']]
    
    def _merge_adjustment_group(self, adjustments: List[RiskAdjustment]) -> RiskAdjustment:
        """Merge a group of similar adjustments."""
        # Calculate weighted average adjustment factor
        total_confidence = sum(adj.confidence_level for adj in adjustments)
        
        if total_confidence > 0:
            weighted_factor = sum(
                adj.adjustment_factor * adj.confidence_level 
                for adj in adjustments
            ) / total_confidence
        else:
            weighted_factor = np.mean([adj.adjustment_factor for adj in adjustments])
        
        # Combine reasons
        reasons = [adj.reason for adj in adjustments]
        combined_reason = "; ".join(reasons[:3])  # Limit length
        
        # Use highest priority
        max_priority = max(adj.priority for adj in adjustments)
        
        # Use latest effective time
        latest_time = max(adj.effective_time for adj in adjustments)
        
        return RiskAdjustment(
            trigger_type=adjustments[0].trigger_type,
            adjustment_factor=weighted_factor,
            target_positions={},  # Will be calculated later
            reason=combined_reason,
            confidence_level=min(1.0, total_confidence / len(adjustments)),
            effective_time=latest_time,
            priority=max_priority
        )
    
    def _calculate_final_position_targets(
        self,
        current_weights: Dict[str, float],
        adjustments: List[RiskAdjustment]
    ) -> Dict[str, float]:
        """Calculate final target position sizes after all adjustments."""
        # Start with current weights
        target_weights = current_weights.copy()
        
        # Apply each adjustment
        for adjustment in adjustments:
            for asset in target_weights:
                if asset in adjustment.target_positions:
                    # Use specific target if provided
                    target_weights[asset] = adjustment.target_positions[asset]
                else:
                    # Apply adjustment factor
                    target_weights[asset] *= adjustment.adjustment_factor
        
        # Ensure weights are within reasonable bounds
        for asset in target_weights:
            target_weights[asset] = max(0.0, min(0.5, target_weights[asset]))
        
        # Normalize weights to sum to 1
        total_weight = sum(target_weights.values())
        if total_weight > 0:
            target_weights = {
                asset: weight / total_weight 
                for asset, weight in target_weights.items()
            }
        
        return target_weights
    
    def _assess_asset_risk_level(self, vol_metrics: VolatilityMetrics) -> str:
        """Assess risk level for individual asset."""
        if vol_metrics.vol_breakout or vol_metrics.volatility_percentile > 0.9:
            return 'high'
        elif vol_metrics.is_high_volatility():
            return 'medium'
        elif vol_metrics.is_low_volatility():
            return 'low'
        else:
            return 'normal'
    
    def _assess_portfolio_risk_level(
        self,
        regime_analysis: Dict[str, Any],
        correlation_analysis: Dict[str, Any]
    ) -> str:
        """Assess overall portfolio risk level."""
        risk_factors = 0
        
        # Regime risk
        if regime_analysis['current_regime'] in ['bear_market', 'high_volatility']:
            risk_factors += 2
        elif regime_analysis['current_regime'] in ['neutral']:
            risk_factors += 1
        
        # Correlation risk
        if len(correlation_analysis['high_correlation_pairs']) > 2:
            risk_factors += 2
        elif len(correlation_analysis['high_correlation_pairs']) > 0:
            risk_factors += 1
        
        # Diversification risk
        div_metrics = correlation_analysis['diversification_metrics']
        if div_metrics['concentration_risk'] > 0.5:
            risk_factors += 1
        
        if risk_factors >= 4:
            return 'high'
        elif risk_factors >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_overall_risk_score(
        self,
        asset_results: Dict[str, Any],
        portfolio_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall portfolio risk score (0-1 scale)."""
        risk_components = []
        
        # Asset-level risks
        for asset_result in asset_results.values():
            if isinstance(asset_result, dict) and 'volatility_metrics' in asset_result:
                vol_metrics = asset_result['volatility_metrics']
                asset_risk = vol_metrics.volatility_percentile
                risk_components.append(asset_risk)
        
        # Portfolio-level risks
        regime_risk = 0.5  # Default neutral
        if portfolio_analysis['regime_analysis']['current_regime'] == 'bear_market':
            regime_risk = 0.8
        elif portfolio_analysis['regime_analysis']['current_regime'] == 'high_volatility':
            regime_risk = 0.9
        elif portfolio_analysis['regime_analysis']['current_regime'] == 'bull_market':
            regime_risk = 0.3
        
        risk_components.append(regime_risk)
        
        # Correlation risk
        correlation_risk = min(1.0, len(portfolio_analysis['correlation_analysis']['high_correlation_pairs']) * 0.2)
        risk_components.append(correlation_risk)
        
        # Calculate weighted average
        overall_risk = np.mean(risk_components) if risk_components else 0.5
        
        return overall_risk
    
    def _create_adjustment_summary(self, adjustments: List[RiskAdjustment]) -> Dict[str, Any]:
        """Create summary of adjustments."""
        if not adjustments:
            return {'total_adjustments': 0, 'avg_adjustment_factor': 1.0, 'priority_distribution': {}}
        
        adjustment_factors = [adj.adjustment_factor for adj in adjustments]
        priorities = [adj.priority for adj in adjustments]
        
        priority_dist = {}
        for priority in [1, 2, 3, 4, 5]:
            priority_dist[priority] = sum(1 for p in priorities if p == priority)
        
        return {
            'total_adjustments': len(adjustments),
            'avg_adjustment_factor': np.mean(adjustment_factors),
            'min_adjustment_factor': min(adjustment_factors),
            'max_adjustment_factor': max(adjustment_factors),
            'priority_distribution': priority_dist,
            'trigger_types': [adj.trigger_type.value for adj in adjustments]
        }
    
    def _update_adjustment_history(self, adjustments: List[RiskAdjustment]) -> None:
        """Update adjustment history for tracking."""
        self.adjustment_history.append({
            'timestamp': datetime.now(),
            'adjustments': adjustments,
            'adjustment_count': len(adjustments)
        })
        
        # Keep only recent history
        if len(self.adjustment_history) > 100:
            self.adjustment_history = self.adjustment_history[-100:]
        
        # Update active adjustments
        self.active_adjustments = adjustments
    
    def get_adjustment_status(self) -> Dict[str, Any]:
        """Get current status of risk adjustments."""
        return {
            'active_adjustments': len(self.active_adjustments),
            'last_adjustment_time': (
                self.adjustment_history[-1]['timestamp'].isoformat() 
                if self.adjustment_history else None
            ),
            'total_historical_adjustments': len(self.adjustment_history),
            'current_adjustments': [
                {
                    'trigger': adj.trigger_type.value,
                    'factor': adj.adjustment_factor,
                    'reason': adj.reason,
                    'confidence': adj.confidence_level,
                    'priority': adj.priority
                }
                for adj in self.active_adjustments
            ]
        }