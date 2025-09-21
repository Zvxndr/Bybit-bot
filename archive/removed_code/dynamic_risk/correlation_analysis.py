"""
Dynamic Correlation Analysis System.

This module provides comprehensive correlation analysis for dynamic risk management
including:

- Real-time correlation matrix calculation and monitoring
- Cross-asset correlation analysis and regime detection
- Rolling correlation estimation with multiple time windows
- Correlation breakdowns and structural breaks detection
- Portfolio correlation risk assessment
- Dynamic hedging ratios based on correlation changes
- Correlation clustering for risk grouping
- Tail correlation analysis during stress periods
- Cross-market correlation monitoring (crypto, traditional assets)
- Correlation-based position sizing adjustments

The system continuously monitors correlations across all portfolio assets to
detect regime changes and adjust risk management parameters accordingly.
"""

import asyncio
import threading
import time
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import statistics
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sqlite3
import json

from ..utils.logging import TradingLogger


class CorrelationRegime(Enum):
    """Correlation regime classification."""
    LOW_CORRELATION = "low_correlation"        # Correlations below 25th percentile
    NORMAL_CORRELATION = "normal_correlation"  # Correlations between 25th-75th percentile
    HIGH_CORRELATION = "high_correlation"      # Correlations above 75th percentile
    CRISIS_CORRELATION = "crisis_correlation"  # Extreme correlations during stress


class CorrelationTrend(Enum):
    """Correlation trend direction."""
    INCREASING = "increasing"    # Correlations increasing
    DECREASING = "decreasing"    # Correlations decreasing
    STABLE = "stable"           # Correlations stable
    VOLATILE = "volatile"       # Correlations highly variable


@dataclass
class CorrelationMetrics:
    """Container for comprehensive correlation metrics."""
    
    symbol_pair: Tuple[str, str]
    timestamp: datetime
    
    # Basic correlation measures
    correlation_1h: float       # 1-hour rolling correlation
    correlation_4h: float       # 4-hour rolling correlation
    correlation_24h: float      # 24-hour rolling correlation
    correlation_7d: float       # 7-day rolling correlation
    correlation_30d: float      # 30-day rolling correlation
    
    # Advanced correlation estimates
    spearman_correlation: float # Spearman rank correlation
    kendall_correlation: float  # Kendall tau correlation
    tail_correlation: float     # Correlation during extreme moves
    
    # Regime and trend analysis
    correlation_regime: CorrelationRegime
    regime_confidence: float
    correlation_trend: CorrelationTrend
    trend_strength: float
    
    # Statistical measures
    correlation_percentile: float    # Current correlation vs historical
    correlation_zscore: float       # Z-score vs historical mean
    correlation_stability: float    # Stability of correlation over time
    
    # Risk metrics
    correlation_volatility: float   # Volatility of correlation
    beta_coefficient: float         # Beta of asset 2 vs asset 1
    tracking_error: float          # Tracking error between assets
    
    # Time-varying measures
    dynamic_correlation: float      # DCC-GARCH style dynamic correlation
    correlation_half_life: float    # Half-life of correlation changes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol_pair': list(self.symbol_pair),
            'timestamp': self.timestamp.isoformat(),
            'correlation_1h': self.correlation_1h,
            'correlation_4h': self.correlation_4h,
            'correlation_24h': self.correlation_24h,
            'correlation_7d': self.correlation_7d,
            'correlation_30d': self.correlation_30d,
            'spearman_correlation': self.spearman_correlation,
            'kendall_correlation': self.kendall_correlation,
            'tail_correlation': self.tail_correlation,
            'correlation_regime': self.correlation_regime.value,
            'regime_confidence': self.regime_confidence,
            'correlation_trend': self.correlation_trend.value,
            'trend_strength': self.trend_strength,
            'correlation_percentile': self.correlation_percentile,
            'correlation_zscore': self.correlation_zscore,
            'correlation_stability': self.correlation_stability,
            'correlation_volatility': self.correlation_volatility,
            'beta_coefficient': self.beta_coefficient,
            'tracking_error': self.tracking_error,
            'dynamic_correlation': self.dynamic_correlation,
            'correlation_half_life': self.correlation_half_life
        }


@dataclass
class PortfolioCorrelationMetrics:
    """Container for portfolio-wide correlation metrics."""
    
    timestamp: datetime
    symbols: List[str]
    
    # Matrix-level metrics
    correlation_matrix: np.ndarray
    average_correlation: float
    max_correlation: float
    min_correlation: float
    correlation_eigenvalues: np.ndarray
    
    # Diversification metrics
    diversification_ratio: float    # Portfolio vol / weighted average vol
    effective_assets: float         # Effective number of independent assets
    concentration_risk: float       # Risk concentration measure
    
    # Risk clustering
    correlation_clusters: Dict[int, List[str]]  # Asset clusters by correlation
    cluster_correlations: Dict[int, float]      # Average correlation within clusters
    
    # Principal component analysis
    explained_variance: np.ndarray   # Variance explained by each PC
    factor_loadings: np.ndarray     # Loading of each asset on each PC
    
    # Regime metrics
    portfolio_regime: CorrelationRegime
    regime_transition_probability: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbols': self.symbols,
            'correlation_matrix': self.correlation_matrix.tolist(),
            'average_correlation': self.average_correlation,
            'max_correlation': self.max_correlation,
            'min_correlation': self.min_correlation,
            'correlation_eigenvalues': self.correlation_eigenvalues.tolist(),
            'diversification_ratio': self.diversification_ratio,
            'effective_assets': self.effective_assets,
            'concentration_risk': self.concentration_risk,
            'correlation_clusters': {str(k): v for k, v in self.correlation_clusters.items()},
            'cluster_correlations': {str(k): v for k, v in self.cluster_correlations.items()},
            'explained_variance': self.explained_variance.tolist(),
            'factor_loadings': self.factor_loadings.tolist(),
            'portfolio_regime': self.portfolio_regime.value,
            'regime_transition_probability': self.regime_transition_probability
        }


class DynamicCorrelationCalculator:
    """
    Dynamic correlation calculator with multiple estimation methods.
    
    This class provides various correlation estimation techniques for
    robust correlation measurement in different market conditions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("DynamicCorrelationCalculator")
        
        # Return data storage
        self.return_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # DCC-GARCH parameters (simplified)
        self.dcc_params: Dict[Tuple[str, str], Dict] = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for correlation calculator."""
        return {
            'min_observations': 30,      # Minimum observations for correlation
            'rolling_windows': [24, 168, 720, 2160],  # 1h, 1w, 1m, 3m
            'tail_threshold': 0.05,      # Threshold for tail correlation (5%)
            'dcc_alpha': 0.01,          # DCC-GARCH alpha parameter
            'dcc_beta': 0.95,           # DCC-GARCH beta parameter
            'ewma_lambda': 0.94,        # EWMA decay factor
            'stability_window': 168,     # Window for stability calculation
        }
    
    def add_return_data(self, symbol: str, return_value: float, timestamp: Optional[datetime] = None) -> None:
        """Add return data for correlation calculation."""
        timestamp = timestamp or datetime.now()
        
        data_point = {
            'return': return_value,
            'timestamp': timestamp
        }
        
        self.return_data[symbol].append(data_point)
    
    def calculate_pearson_correlation(
        self,
        symbol1: str,
        symbol2: str,
        window_hours: int = 24
    ) -> float:
        """Calculate Pearson correlation coefficient."""
        returns1 = self._get_aligned_returns(symbol1, symbol2, window_hours)
        returns2 = self._get_aligned_returns(symbol2, symbol1, window_hours)
        
        if len(returns1) < self.config['min_observations'] or len(returns2) < self.config['min_observations']:
            return 0.0
        
        if len(returns1) != len(returns2):
            return 0.0
        
        try:
            correlation, _ = stats.pearsonr(returns1, returns2)
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0
    
    def calculate_spearman_correlation(
        self,
        symbol1: str,
        symbol2: str,
        window_hours: int = 24
    ) -> float:
        """Calculate Spearman rank correlation."""
        returns1 = self._get_aligned_returns(symbol1, symbol2, window_hours)
        returns2 = self._get_aligned_returns(symbol2, symbol1, window_hours)
        
        if len(returns1) < self.config['min_observations'] or len(returns2) < self.config['min_observations']:
            return 0.0
        
        try:
            correlation, _ = stats.spearmanr(returns1, returns2)
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0
    
    def calculate_kendall_correlation(
        self,
        symbol1: str,
        symbol2: str,
        window_hours: int = 24
    ) -> float:
        """Calculate Kendall tau correlation."""
        returns1 = self._get_aligned_returns(symbol1, symbol2, window_hours)
        returns2 = self._get_aligned_returns(symbol2, symbol1, window_hours)
        
        if len(returns1) < self.config['min_observations'] or len(returns2) < self.config['min_observations']:
            return 0.0
        
        try:
            correlation, _ = stats.kendalltau(returns1, returns2)
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0
    
    def calculate_tail_correlation(
        self,
        symbol1: str,
        symbol2: str,
        window_hours: int = 168,
        tail_threshold: Optional[float] = None
    ) -> float:
        """Calculate correlation during extreme movements (tail correlation)."""
        tail_threshold = tail_threshold or self.config['tail_threshold']
        
        returns1 = self._get_aligned_returns(symbol1, symbol2, window_hours)
        returns2 = self._get_aligned_returns(symbol2, symbol1, window_hours)
        
        if len(returns1) < 100 or len(returns2) < 100:  # Need more data for tail analysis
            return 0.0
        
        # Calculate thresholds for extreme moves
        threshold1_low = np.percentile(returns1, tail_threshold * 100)
        threshold1_high = np.percentile(returns1, (1 - tail_threshold) * 100)
        threshold2_low = np.percentile(returns2, tail_threshold * 100)
        threshold2_high = np.percentile(returns2, (1 - tail_threshold) * 100)
        
        # Find extreme observations
        extreme_mask = (
            (returns1 <= threshold1_low) | (returns1 >= threshold1_high) |
            (returns2 <= threshold2_low) | (returns2 >= threshold2_high)
        )
        
        extreme_returns1 = np.array(returns1)[extreme_mask]
        extreme_returns2 = np.array(returns2)[extreme_mask]
        
        if len(extreme_returns1) < 10:  # Need sufficient extreme observations
            return 0.0
        
        try:
            correlation, _ = stats.pearsonr(extreme_returns1, extreme_returns2)
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0
    
    def calculate_beta_coefficient(
        self,
        dependent_symbol: str,
        independent_symbol: str,
        window_hours: int = 168
    ) -> float:
        """Calculate beta coefficient (dependent vs independent)."""
        returns_dep = self._get_aligned_returns(dependent_symbol, independent_symbol, window_hours)
        returns_indep = self._get_aligned_returns(independent_symbol, dependent_symbol, window_hours)
        
        if len(returns_dep) < self.config['min_observations'] or len(returns_indep) < self.config['min_observations']:
            return 1.0
        
        try:
            # Calculate beta using linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(returns_indep, returns_dep)
            return slope
        except Exception:
            return 1.0
    
    def calculate_tracking_error(
        self,
        symbol1: str,
        symbol2: str,
        window_hours: int = 168
    ) -> float:
        """Calculate tracking error between two assets."""
        returns1 = self._get_aligned_returns(symbol1, symbol2, window_hours)
        returns2 = self._get_aligned_returns(symbol2, symbol1, window_hours)
        
        if len(returns1) < self.config['min_observations'] or len(returns2) < self.config['min_observations']:
            return 0.0
        
        # Calculate return differences
        return_diff = np.array(returns1) - np.array(returns2)
        
        # Tracking error is standard deviation of return differences
        tracking_error = np.std(return_diff, ddof=1)
        
        # Annualize tracking error
        periods_per_year = 365.25 * 24 / 1  # Assuming hourly data
        annual_tracking_error = tracking_error * math.sqrt(periods_per_year)
        
        return annual_tracking_error
    
    def calculate_dynamic_correlation(
        self,
        symbol1: str,
        symbol2: str
    ) -> float:
        """Calculate dynamic correlation using DCC-GARCH approach (simplified)."""
        symbol_pair = tuple(sorted([symbol1, symbol2]))
        
        returns1 = self._get_aligned_returns(symbol1, symbol2, 720)  # 30 days
        returns2 = self._get_aligned_returns(symbol2, symbol1, 720)
        
        if len(returns1) < 100 or len(returns2) < 100:
            return self.calculate_pearson_correlation(symbol1, symbol2, 168)
        
        # Initialize DCC parameters if not exists
        if symbol_pair not in self.dcc_params:
            self.dcc_params[symbol_pair] = {
                'Q': np.cov([returns1[:50], returns2[:50]]),  # Unconditional covariance
                'alpha': self.config['dcc_alpha'],
                'beta': self.config['dcc_beta']
            }
        
        params = self.dcc_params[symbol_pair]
        
        # Simplified DCC update (using recent data)
        recent_returns1 = returns1[-20:]  # Last 20 observations
        recent_returns2 = returns2[-20:]
        
        # Calculate standardized residuals (simplified)
        std1 = np.std(recent_returns1, ddof=1)
        std2 = np.std(recent_returns2, ddof=1)
        
        if std1 == 0 or std2 == 0:
            return 0.0
        
        eps1 = np.array(recent_returns1) / std1
        eps2 = np.array(recent_returns2) / std2
        
        # Update DCC parameters (simplified)
        outer_product = np.outer(eps1[-1:], eps2[-1:])
        Q_t = (1 - params['alpha'] - params['beta']) * params['Q'] + \
              params['alpha'] * outer_product + \
              params['beta'] * params['Q']
        
        # Extract correlation
        if Q_t.shape == (1, 1):
            correlation = 0.0
        else:
            correlation = Q_t[0, 1] / math.sqrt(Q_t[0, 0] * Q_t[1, 1])
        
        # Update stored parameters
        params['Q'] = Q_t
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_correlation_stability(
        self,
        symbol1: str,
        symbol2: str,
        window_hours: Optional[int] = None
    ) -> float:
        """Calculate stability of correlation over time."""
        window_hours = window_hours or self.config['stability_window']
        
        # Calculate rolling correlations
        correlations = []
        step_size = max(1, window_hours // 10)  # 10 correlation measurements
        
        for i in range(0, window_hours, step_size):
            start_hour = i
            end_hour = i + (window_hours // 5)  # Each correlation uses 1/5 of window
            
            returns1 = self._get_aligned_returns(symbol1, symbol2, end_hour, start_hour)
            returns2 = self._get_aligned_returns(symbol2, symbol1, end_hour, start_hour)
            
            if len(returns1) >= 20 and len(returns2) >= 20:
                try:
                    corr, _ = stats.pearsonr(returns1, returns2)
                    if not np.isnan(corr):
                        correlations.append(corr)
                except Exception:
                    continue
        
        if len(correlations) < 3:
            return 0.0
        
        # Stability is inverse of correlation volatility
        correlation_vol = np.std(correlations, ddof=1)
        stability = 1.0 / (1.0 + correlation_vol)  # Scale 0-1
        
        return stability
    
    def calculate_correlation_half_life(
        self,
        symbol1: str,
        symbol2: str
    ) -> float:
        """Calculate half-life of correlation changes."""
        # Get correlation time series
        correlations = []
        timestamps = []
        
        # Calculate daily correlations over past month
        for days_back in range(30, 0, -1):
            start_time = datetime.now() - timedelta(days=days_back + 1)
            end_time = datetime.now() - timedelta(days=days_back)
            
            returns1 = self._get_returns_in_period(symbol1, start_time, end_time)
            returns2 = self._get_returns_in_period(symbol2, start_time, end_time)
            
            if len(returns1) >= 10 and len(returns2) >= 10:
                try:
                    corr, _ = stats.pearsonr(returns1, returns2)
                    if not np.isnan(corr):
                        correlations.append(corr)
                        timestamps.append(end_time)
                except Exception:
                    continue
        
        if len(correlations) < 10:
            return 30.0  # Default 30-day half-life
        
        # Calculate correlation changes
        correlation_changes = np.diff(correlations)
        
        # Fit AR(1) model to estimate persistence
        # corr_t = alpha * corr_{t-1} + epsilon
        if len(correlation_changes) < 5:
            return 30.0
        
        try:
            # Simple AR(1) coefficient estimation
            corr_lagged = correlations[:-1]
            corr_current = correlations[1:]
            
            slope, _, _, _, _ = stats.linregress(corr_lagged, corr_current)
            
            # Half-life calculation: ln(0.5) / ln(persistence)
            if slope > 0 and slope < 1:
                half_life = -math.log(0.5) / math.log(slope)
                return min(max(half_life, 1.0), 365.0)  # Bound between 1 day and 1 year
            
        except Exception:
            pass
        
        return 30.0  # Default half-life
    
    def _get_aligned_returns(
        self,
        symbol1: str,
        symbol2: str,
        window_hours: int,
        offset_hours: int = 0
    ) -> List[float]:
        """Get aligned returns for two symbols."""
        cutoff_time = datetime.now() - timedelta(hours=window_hours + offset_hours)
        start_time = datetime.now() - timedelta(hours=offset_hours) if offset_hours > 0 else datetime.now()
        
        # Get data for symbol1
        data1 = [
            d for d in self.return_data[symbol1]
            if cutoff_time <= d['timestamp'] <= start_time
        ]
        
        # Get data for symbol2
        data2 = [
            d for d in self.return_data[symbol2]
            if cutoff_time <= d['timestamp'] <= start_time
        ]
        
        if not data1 or not data2:
            return []
        
        # Create timestamp alignment
        timestamps1 = {d['timestamp']: d['return'] for d in data1}
        timestamps2 = {d['timestamp']: d['return'] for d in data2}
        
        # Find common timestamps
        common_timestamps = sorted(set(timestamps1.keys()) & set(timestamps2.keys()))
        
        # Return aligned returns for symbol1
        return [timestamps1[ts] for ts in common_timestamps]
    
    def _get_returns_in_period(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[float]:
        """Get returns for a symbol in specific time period."""
        data = [
            d for d in self.return_data[symbol]
            if start_time <= d['timestamp'] <= end_time
        ]
        
        return [d['return'] for d in data]


class CorrelationRegimeDetector:
    """
    Correlation regime detection and classification system.
    
    This class identifies different correlation regimes and tracks
    transitions for dynamic risk management.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("CorrelationRegimeDetector")
        
        # Historical correlation data
        self.correlation_history: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=2160))
        
        # Current regimes
        self.current_regimes: Dict[Tuple[str, str], CorrelationRegime] = {}
        self.regime_start_times: Dict[Tuple[str, str], datetime] = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for regime detector."""
        return {
            'regime_thresholds': {
                'low_correlation': 0.25,      # 25th percentile
                'high_correlation': 0.75,     # 75th percentile
                'crisis_correlation': 0.95    # 95th percentile
            },
            'min_regime_duration': 7200,      # Minimum 2 hours in regime
            'lookback_days': 90,              # Days of history for percentiles
            'crisis_threshold': 0.8,          # Absolute correlation for crisis
        }
    
    def add_correlation_observation(
        self,
        symbol1: str,
        symbol2: str,
        correlation: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add correlation observation for regime detection."""
        timestamp = timestamp or datetime.now()
        symbol_pair = tuple(sorted([symbol1, symbol2]))
        
        corr_point = {
            'correlation': correlation,
            'timestamp': timestamp
        }
        
        self.correlation_history[symbol_pair].append(corr_point)
    
    def detect_regime(
        self,
        symbol1: str,
        symbol2: str,
        current_correlation: float
    ) -> Tuple[CorrelationRegime, float]:
        """Detect current correlation regime and confidence."""
        symbol_pair = tuple(sorted([symbol1, symbol2]))
        corr_history = list(self.correlation_history[symbol_pair])
        
        if len(corr_history) < 50:  # Need sufficient history
            return CorrelationRegime.NORMAL_CORRELATION, 0.5
        
        # Extract correlation values
        historical_corrs = [abs(c['correlation']) for c in corr_history if not np.isnan(c['correlation'])]
        
        if len(historical_corrs) < 50:
            return CorrelationRegime.NORMAL_CORRELATION, 0.5
        
        current_abs_corr = abs(current_correlation)
        
        # Check for crisis correlation first
        if current_abs_corr > self.config['crisis_threshold']:
            return CorrelationRegime.CRISIS_CORRELATION, 0.9
        
        # Calculate percentiles
        percentiles = np.percentile(historical_corrs, [25, 75, 95])
        p25, p75, p95 = percentiles
        
        # Classify regime
        if current_abs_corr <= p25:
            regime = CorrelationRegime.LOW_CORRELATION
            confidence = (p25 - current_abs_corr) / p25 if p25 > 0 else 0.5
        elif current_abs_corr <= p75:
            regime = CorrelationRegime.NORMAL_CORRELATION
            confidence = 0.8  # High confidence in normal regime
        elif current_abs_corr <= p95:
            regime = CorrelationRegime.HIGH_CORRELATION
            confidence = (current_abs_corr - p75) / (p95 - p75) if p95 > p75 else 0.5
        else:
            regime = CorrelationRegime.CRISIS_CORRELATION
            confidence = min(1.0, (current_abs_corr - p95) / (1.0 - p95)) if p95 < 1.0 else 0.9
        
        # Smooth regime transitions
        if symbol_pair in self.current_regimes:
            old_regime = self.current_regimes[symbol_pair]
            if old_regime != regime:
                # Check minimum duration
                if symbol_pair in self.regime_start_times:
                    duration = datetime.now() - self.regime_start_times[symbol_pair]
                    if duration.total_seconds() < self.config['min_regime_duration']:
                        # Stay in old regime
                        return old_regime, confidence * 0.8
        
        # Update regime tracking
        if symbol_pair not in self.current_regimes or self.current_regimes[symbol_pair] != regime:
            self.current_regimes[symbol_pair] = regime
            self.regime_start_times[symbol_pair] = datetime.now()
        
        return regime, min(1.0, max(0.0, confidence))
    
    def detect_trend(
        self,
        symbol1: str,
        symbol2: str
    ) -> Tuple[CorrelationTrend, float]:
        """Detect correlation trend and strength."""
        symbol_pair = tuple(sorted([symbol1, symbol2]))
        corr_history = list(self.correlation_history[symbol_pair])
        
        if len(corr_history) < 20:
            return CorrelationTrend.STABLE, 0.0
        
        # Get recent correlation values
        recent_corrs = [c['correlation'] for c in corr_history[-20:]]
        
        if len(recent_corrs) < 10:
            return CorrelationTrend.STABLE, 0.0
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_corrs))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_corrs)
        
        # Determine trend direction and strength
        trend_strength = abs(r_value)  # Use correlation coefficient as strength
        
        if p_value > 0.05:  # Not statistically significant
            return CorrelationTrend.STABLE, 0.0
        
        # Check for high volatility of correlations
        corr_changes = np.diff(recent_corrs)
        if np.std(corr_changes) > 0.2:  # High correlation volatility
            return CorrelationTrend.VOLATILE, np.std(corr_changes)
        
        if abs(slope) < 0.01:  # Small slope
            return CorrelationTrend.STABLE, trend_strength
        
        if slope > 0:
            trend = CorrelationTrend.INCREASING
        else:
            trend = CorrelationTrend.DECREASING
        
        return trend, min(1.0, trend_strength)


class PortfolioCorrelationAnalyzer:
    """
    Portfolio-wide correlation analysis system.
    
    This class analyzes correlations across all portfolio assets and
    provides portfolio-level correlation metrics and risk measures.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("PortfolioCorrelationAnalyzer")
        
        # Components
        self.correlation_calculator = DynamicCorrelationCalculator(self.config.get('calculator', {}))
        
    def _default_config(self) -> Dict:
        """Default configuration for portfolio analyzer."""
        return {
            'min_assets': 3,              # Minimum assets for portfolio analysis
            'cluster_method': 'ward',     # Clustering method
            'n_clusters': 3,             # Number of correlation clusters
            'pca_variance_threshold': 0.95,  # PCA variance to retain
            'calculator': {}
        }
    
    def calculate_correlation_matrix(
        self,
        symbols: List[str],
        window_hours: int = 168
    ) -> np.ndarray:
        """Calculate correlation matrix for given symbols."""
        n_assets = len(symbols)
        correlation_matrix = np.eye(n_assets)
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                correlation = self.correlation_calculator.calculate_pearson_correlation(
                    symbols[i], symbols[j], window_hours
                )
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def calculate_portfolio_metrics(
        self,
        symbols: List[str],
        weights: Optional[np.ndarray] = None,
        window_hours: int = 168
    ) -> Optional[PortfolioCorrelationMetrics]:
        """Calculate comprehensive portfolio correlation metrics."""
        if len(symbols) < self.config['min_assets']:
            return None
        
        weights = weights or np.ones(len(symbols)) / len(symbols)
        
        try:
            # Calculate correlation matrix
            correlation_matrix = self.calculate_correlation_matrix(symbols, window_hours)
            
            # Basic matrix statistics
            upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            average_correlation = np.mean(upper_triangle)
            max_correlation = np.max(upper_triangle)
            min_correlation = np.min(upper_triangle)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            # Diversification metrics
            diversification_ratio = self._calculate_diversification_ratio(
                correlation_matrix, weights, symbols, window_hours
            )
            effective_assets = self._calculate_effective_assets(correlation_matrix, weights)
            concentration_risk = self._calculate_concentration_risk(correlation_matrix, weights)
            
            # Correlation clustering
            correlation_clusters, cluster_correlations = self._perform_correlation_clustering(
                correlation_matrix, symbols
            )
            
            # Principal component analysis
            explained_variance, factor_loadings = self._perform_pca(correlation_matrix)
            
            # Portfolio regime detection
            portfolio_regime = self._detect_portfolio_regime(average_correlation, eigenvalues)
            regime_transition_prob = self._calculate_regime_transition_probability(
                symbols, correlation_matrix
            )
            
            return PortfolioCorrelationMetrics(
                timestamp=datetime.now(),
                symbols=symbols,
                correlation_matrix=correlation_matrix,
                average_correlation=average_correlation,
                max_correlation=max_correlation,
                min_correlation=min_correlation,
                correlation_eigenvalues=eigenvalues,
                diversification_ratio=diversification_ratio,
                effective_assets=effective_assets,
                concentration_risk=concentration_risk,
                correlation_clusters=correlation_clusters,
                cluster_correlations=cluster_correlations,
                explained_variance=explained_variance,
                factor_loadings=factor_loadings,
                portfolio_regime=portfolio_regime,
                regime_transition_probability=regime_transition_prob
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio correlation metrics: {e}")
            return None
    
    def _calculate_diversification_ratio(
        self,
        correlation_matrix: np.ndarray,
        weights: np.ndarray,
        symbols: List[str],
        window_hours: int
    ) -> float:
        """Calculate diversification ratio."""
        try:
            # Get individual asset volatilities
            volatilities = []
            for symbol in symbols:
                returns = self._get_symbol_returns(symbol, window_hours)
                if len(returns) > 10:
                    vol = np.std(returns, ddof=1) * math.sqrt(365.25 * 24)  # Annualized
                    volatilities.append(vol)
                else:
                    volatilities.append(0.1)  # Default volatility
            
            volatilities = np.array(volatilities)
            
            # Weighted average of individual volatilities
            weighted_avg_vol = np.sum(weights * volatilities)
            
            # Portfolio volatility
            portfolio_variance = np.dot(weights, np.dot(np.diag(volatilities**2) * correlation_matrix, weights))
            portfolio_vol = math.sqrt(portfolio_variance)
            
            # Diversification ratio
            if portfolio_vol > 0:
                return weighted_avg_vol / portfolio_vol
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _calculate_effective_assets(
        self,
        correlation_matrix: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Calculate effective number of assets (inverse of concentration)."""
        try:
            # Calculate portfolio concentration using correlation matrix
            concentration = np.sum(np.outer(weights, weights) * correlation_matrix)
            effective_assets = 1.0 / concentration if concentration > 0 else len(weights)
            return effective_assets
        except Exception:
            return len(weights)
    
    def _calculate_concentration_risk(
        self,
        correlation_matrix: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Calculate concentration risk measure."""
        try:
            # Herfindahl-Hirschman Index adapted for correlations
            n_assets = len(weights)
            
            # Weight concentration
            weight_concentration = np.sum(weights**2)
            
            # Correlation concentration (how correlated the portfolio is)
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            correlation_concentration = (avg_correlation + 1) / 2  # Scale to [0,1]
            
            # Combined concentration risk
            concentration_risk = (weight_concentration + correlation_concentration) / 2
            
            return concentration_risk
            
        except Exception:
            return 0.5
    
    def _perform_correlation_clustering(
        self,
        correlation_matrix: np.ndarray,
        symbols: List[str]
    ) -> Tuple[Dict[int, List[str]], Dict[int, float]]:
        """Perform hierarchical clustering based on correlations."""
        try:
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(correlation_matrix)
            
            # Perform hierarchical clustering
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method=self.config['cluster_method'])
            
            # Get cluster assignments
            cluster_assignments = fcluster(linkage_matrix, self.config['n_clusters'], criterion='maxclust')
            
            # Group symbols by cluster
            clusters = defaultdict(list)
            for i, cluster_id in enumerate(cluster_assignments):
                clusters[cluster_id].append(symbols[i])
            
            # Calculate average correlation within each cluster
            cluster_correlations = {}
            for cluster_id, cluster_symbols in clusters.items():
                if len(cluster_symbols) > 1:
                    cluster_indices = [symbols.index(s) for s in cluster_symbols]
                    cluster_corr_matrix = correlation_matrix[np.ix_(cluster_indices, cluster_indices)]
                    upper_triangle = cluster_corr_matrix[np.triu_indices_from(cluster_corr_matrix, k=1)]
                    cluster_correlations[cluster_id] = np.mean(upper_triangle)
                else:
                    cluster_correlations[cluster_id] = 0.0
            
            return dict(clusters), cluster_correlations
            
        except Exception as e:
            self.logger.error(f"Error in correlation clustering: {e}")
            # Return single cluster with all assets
            return {1: symbols}, {1: 0.0}
    
    def _perform_pca(
        self,
        correlation_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform principal component analysis on correlation matrix."""
        try:
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
            
            # Sort by eigenvalues (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Calculate explained variance
            total_variance = np.sum(eigenvalues)
            explained_variance = eigenvalues / total_variance if total_variance > 0 else eigenvalues
            
            # Factor loadings (eigenvectors scaled by sqrt of eigenvalues)
            factor_loadings = eigenvectors * np.sqrt(np.abs(eigenvalues))
            
            return explained_variance, factor_loadings
            
        except Exception as e:
            self.logger.error(f"Error in PCA analysis: {e}")
            n_assets = correlation_matrix.shape[0]
            return np.ones(n_assets) / n_assets, np.eye(n_assets)
    
    def _detect_portfolio_regime(
        self,
        average_correlation: float,
        eigenvalues: np.ndarray
    ) -> CorrelationRegime:
        """Detect portfolio-wide correlation regime."""
        # Use first eigenvalue as measure of systematic risk
        first_eigenvalue_ratio = eigenvalues[0] / len(eigenvalues) if len(eigenvalues) > 0 else 0.5
        
        # High average correlation or high first eigenvalue indicates crisis
        if average_correlation > 0.8 or first_eigenvalue_ratio > 0.7:
            return CorrelationRegime.CRISIS_CORRELATION
        elif average_correlation > 0.5 or first_eigenvalue_ratio > 0.5:
            return CorrelationRegime.HIGH_CORRELATION
        elif average_correlation < 0.2 and first_eigenvalue_ratio < 0.3:
            return CorrelationRegime.LOW_CORRELATION
        else:
            return CorrelationRegime.NORMAL_CORRELATION
    
    def _calculate_regime_transition_probability(
        self,
        symbols: List[str],
        correlation_matrix: np.ndarray
    ) -> float:
        """Calculate probability of regime transition."""
        try:
            # Simplified approach: use correlation volatility as proxy
            correlations = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            
            # Calculate recent correlation changes for each pair
            correlation_volatilities = []
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    stability = self.correlation_calculator.calculate_correlation_stability(
                        symbols[i], symbols[j]
                    )
                    correlation_volatilities.append(1.0 - stability)  # Instability
            
            if correlation_volatilities:
                avg_instability = np.mean(correlation_volatilities)
                # Transform to probability (higher instability = higher transition probability)
                transition_prob = min(1.0, avg_instability * 2)  # Scale appropriately
                return transition_prob
            
            return 0.1  # Default low transition probability
            
        except Exception:
            return 0.1
    
    def _get_symbol_returns(self, symbol: str, window_hours: int) -> List[float]:
        """Get returns for a symbol (placeholder - would integrate with data source)."""
        # This would integrate with the correlation calculator's return data
        return self.correlation_calculator._get_aligned_returns(symbol, symbol, window_hours)


class DynamicCorrelationAnalyzer:
    """
    Main dynamic correlation analysis system.
    
    This class combines all correlation analysis components to provide
    comprehensive correlation monitoring and regime detection for risk management.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("DynamicCorrelationAnalyzer")
        
        # Core components
        self.correlation_calculator = DynamicCorrelationCalculator(self.config.get('calculator', {}))
        self.regime_detector = CorrelationRegimeDetector(self.config.get('regime_detector', {}))
        self.portfolio_analyzer = PortfolioCorrelationAnalyzer(self.config.get('portfolio_analyzer', {}))
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Callbacks for correlation changes
        self.correlation_callbacks: List[Callable[[str, str, CorrelationMetrics], None]] = []
        self.portfolio_callbacks: List[Callable[[PortfolioCorrelationMetrics], None]] = []
        
        # Database for persistence
        self.db_path = self.config.get('database_path', 'correlation_analysis.db')
        self._init_database()
        
    def _default_config(self) -> Dict:
        """Default configuration for correlation analyzer."""
        return {
            'update_interval': 300,       # 5 minutes
            'database_path': 'correlation_analysis.db',
            'enable_persistence': True,
            'portfolio_symbols': [],      # Symbols to include in portfolio analysis
            'alert_thresholds': {
                'regime_change': True,
                'high_correlation': 0.8,
                'correlation_spike': 0.3   # Change in correlation
            },
            'calculator': {},
            'regime_detector': {},
            'portfolio_analyzer': {}
        }
    
    def _init_database(self) -> None:
        """Initialize SQLite database for correlation data."""
        if not self.config['enable_persistence']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS correlation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol1 TEXT NOT NULL,
                    symbol2 TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    symbols TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_corr_symbols_time ON correlation_metrics (symbol1, symbol2, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_time ON portfolio_metrics (timestamp)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize correlation database: {e}")
    
    def add_return_data(self, symbol: str, return_value: float, timestamp: Optional[datetime] = None) -> None:
        """Add return data for correlation analysis."""
        self.correlation_calculator.add_return_data(symbol, return_value, timestamp)
    
    def calculate_pairwise_metrics(
        self,
        symbol1: str,
        symbol2: str
    ) -> Optional[CorrelationMetrics]:
        """Calculate comprehensive correlation metrics for a symbol pair."""
        try:
            # Calculate various correlation measures
            correlation_1h = self.correlation_calculator.calculate_pearson_correlation(symbol1, symbol2, 1)
            correlation_4h = self.correlation_calculator.calculate_pearson_correlation(symbol1, symbol2, 4)
            correlation_24h = self.correlation_calculator.calculate_pearson_correlation(symbol1, symbol2, 24)
            correlation_7d = self.correlation_calculator.calculate_pearson_correlation(symbol1, symbol2, 168)
            correlation_30d = self.correlation_calculator.calculate_pearson_correlation(symbol1, symbol2, 720)
            
            spearman_correlation = self.correlation_calculator.calculate_spearman_correlation(symbol1, symbol2, 168)
            kendall_correlation = self.correlation_calculator.calculate_kendall_correlation(symbol1, symbol2, 168)
            tail_correlation = self.correlation_calculator.calculate_tail_correlation(symbol1, symbol2, 168)
            
            # Use 24h correlation as primary measure
            primary_correlation = correlation_24h
            
            # Add correlation observation for regime detection
            self.regime_detector.add_correlation_observation(symbol1, symbol2, primary_correlation)
            
            # Detect regime and trend
            correlation_regime, regime_confidence = self.regime_detector.detect_regime(
                symbol1, symbol2, primary_correlation
            )
            correlation_trend, trend_strength = self.regime_detector.detect_trend(symbol1, symbol2)
            
            # Calculate statistical measures
            symbol_pair = tuple(sorted([symbol1, symbol2]))
            corr_history = [c['correlation'] for c in self.regime_detector.correlation_history[symbol_pair]]
            
            if len(corr_history) >= 30:
                correlation_percentile = stats.percentileofscore(corr_history, primary_correlation) / 100
                corr_mean = np.mean(corr_history)
                corr_std = np.std(corr_history)
                correlation_zscore = (primary_correlation - corr_mean) / corr_std if corr_std > 0 else 0
            else:
                correlation_percentile = 0.5
                correlation_zscore = 0.0
            
            # Advanced metrics
            correlation_stability = self.correlation_calculator.calculate_correlation_stability(symbol1, symbol2)
            beta_coefficient = self.correlation_calculator.calculate_beta_coefficient(symbol1, symbol2)
            tracking_error = self.correlation_calculator.calculate_tracking_error(symbol1, symbol2)
            dynamic_correlation = self.correlation_calculator.calculate_dynamic_correlation(symbol1, symbol2)
            correlation_half_life = self.correlation_calculator.calculate_correlation_half_life(symbol1, symbol2)
            
            # Correlation volatility
            correlation_volatility = 1.0 - correlation_stability
            
            # Create metrics object
            metrics = CorrelationMetrics(
                symbol_pair=(symbol1, symbol2),
                timestamp=datetime.now(),
                correlation_1h=correlation_1h,
                correlation_4h=correlation_4h,
                correlation_24h=correlation_24h,
                correlation_7d=correlation_7d,
                correlation_30d=correlation_30d,
                spearman_correlation=spearman_correlation,
                kendall_correlation=kendall_correlation,
                tail_correlation=tail_correlation,
                correlation_regime=correlation_regime,
                regime_confidence=regime_confidence,
                correlation_trend=correlation_trend,
                trend_strength=trend_strength,
                correlation_percentile=correlation_percentile,
                correlation_zscore=correlation_zscore,
                correlation_stability=correlation_stability,
                correlation_volatility=correlation_volatility,
                beta_coefficient=beta_coefficient,
                tracking_error=tracking_error,
                dynamic_correlation=dynamic_correlation,
                correlation_half_life=correlation_half_life
            )
            
            # Save to database
            if self.config['enable_persistence']:
                self._save_correlation_metrics(metrics)
            
            # Check for alerts
            self._check_correlation_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation metrics for {symbol1}-{symbol2}: {e}")
            return None
    
    def calculate_portfolio_metrics(
        self,
        symbols: Optional[List[str]] = None,
        weights: Optional[np.ndarray] = None
    ) -> Optional[PortfolioCorrelationMetrics]:
        """Calculate portfolio-wide correlation metrics."""
        symbols = symbols or self.config['portfolio_symbols']
        
        if not symbols or len(symbols) < 2:
            return None
        
        try:
            # Ensure we have correlation calculator data for portfolio analyzer
            self.portfolio_analyzer.correlation_calculator = self.correlation_calculator
            
            metrics = self.portfolio_analyzer.calculate_portfolio_metrics(symbols, weights)
            
            if metrics and self.config['enable_persistence']:
                self._save_portfolio_metrics(metrics)
            
            # Check for portfolio alerts
            if metrics:
                self._check_portfolio_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio correlation metrics: {e}")
            return None
    
    def _save_correlation_metrics(self, metrics: CorrelationMetrics) -> None:
        """Save correlation metrics to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            data = json.dumps(metrics.to_dict())
            
            cursor.execute("""
                INSERT INTO correlation_metrics (symbol1, symbol2, timestamp, data)
                VALUES (?, ?, ?, ?)
            """, (metrics.symbol_pair[0], metrics.symbol_pair[1], metrics.timestamp, data))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save correlation metrics: {e}")
    
    def _save_portfolio_metrics(self, metrics: PortfolioCorrelationMetrics) -> None:
        """Save portfolio metrics to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            data = json.dumps(metrics.to_dict())
            symbols_str = ','.join(metrics.symbols)
            
            cursor.execute("""
                INSERT INTO portfolio_metrics (timestamp, symbols, data)
                VALUES (?, ?, ?)
            """, (metrics.timestamp, symbols_str, data))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save portfolio metrics: {e}")
    
    def _check_correlation_alerts(self, metrics: CorrelationMetrics) -> None:
        """Check correlation metrics for alert conditions."""
        alerts = []
        
        # Regime change alert
        if self.config['alert_thresholds']['regime_change']:
            if metrics.correlation_regime == CorrelationRegime.CRISIS_CORRELATION:
                alerts.append(f"Crisis correlation detected for {metrics.symbol_pair}: {metrics.correlation_24h:.3f}")
        
        # High correlation alert
        high_threshold = self.config['alert_thresholds']['high_correlation']
        if abs(metrics.correlation_24h) > high_threshold:
            alerts.append(f"High correlation for {metrics.symbol_pair}: {metrics.correlation_24h:.3f}")
        
        # Correlation spike alert
        spike_threshold = self.config['alert_thresholds']['correlation_spike']
        if abs(metrics.correlation_zscore) > spike_threshold:
            alerts.append(f"Correlation spike for {metrics.symbol_pair}: {metrics.correlation_zscore:.2f} standard deviations")
        
        # Notify callbacks
        if alerts:
            for callback in self.correlation_callbacks:
                try:
                    callback(metrics.symbol_pair[0], metrics.symbol_pair[1], metrics)
                except Exception as e:
                    self.logger.error(f"Error in correlation callback: {e}")
            
            self.logger.warning(f"Correlation alerts: {alerts}")
    
    def _check_portfolio_alerts(self, metrics: PortfolioCorrelationMetrics) -> None:
        """Check portfolio metrics for alert conditions."""
        alerts = []
        
        # High average correlation
        if metrics.average_correlation > 0.7:
            alerts.append(f"High portfolio correlation: {metrics.average_correlation:.3f}")
        
        # Low diversification
        if metrics.diversification_ratio < 1.2:
            alerts.append(f"Low diversification ratio: {metrics.diversification_ratio:.2f}")
        
        # High concentration risk
        if metrics.concentration_risk > 0.8:
            alerts.append(f"High concentration risk: {metrics.concentration_risk:.3f}")
        
        # Crisis regime
        if metrics.portfolio_regime == CorrelationRegime.CRISIS_CORRELATION:
            alerts.append("Portfolio in crisis correlation regime")
        
        # Notify callbacks
        if alerts:
            for callback in self.portfolio_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    self.logger.error(f"Error in portfolio callback: {e}")
            
            self.logger.warning(f"Portfolio alerts: {alerts}")
    
    def add_correlation_callback(
        self,
        callback: Callable[[str, str, CorrelationMetrics], None]
    ) -> None:
        """Add callback for correlation changes."""
        self.correlation_callbacks.append(callback)
    
    def add_portfolio_callback(
        self,
        callback: Callable[[PortfolioCorrelationMetrics], None]
    ) -> None:
        """Add callback for portfolio correlation changes."""
        self.portfolio_callbacks.append(callback)
    
    def start_monitoring(self, symbols: Optional[List[str]] = None) -> None:
        """Start background correlation monitoring."""
        if self.is_monitoring:
            return
        
        if symbols:
            self.config['portfolio_symbols'] = symbols
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started correlation monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background correlation monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped correlation monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Get all symbols with return data
                symbols = list(self.correlation_calculator.return_data.keys())
                
                if len(symbols) >= 2:
                    # Calculate pairwise correlations
                    for i in range(len(symbols)):
                        for j in range(i + 1, len(symbols)):
                            metrics = self.calculate_pairwise_metrics(symbols[i], symbols[j])
                            if metrics:
                                self.logger.debug(
                                    f"Updated correlation for {symbols[i]}-{symbols[j]}: "
                                    f"corr={metrics.correlation_24h:.3f}, "
                                    f"regime={metrics.correlation_regime.value}"
                                )
                    
                    # Calculate portfolio metrics
                    portfolio_metrics = self.calculate_portfolio_metrics(symbols)
                    if portfolio_metrics:
                        self.logger.debug(
                            f"Updated portfolio metrics: "
                            f"avg_corr={portfolio_metrics.average_correlation:.3f}, "
                            f"div_ratio={portfolio_metrics.diversification_ratio:.2f}"
                        )
                
                # Sleep until next update
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in correlation monitoring loop: {e}")
                time.sleep(60)  # Error backoff
    
    def get_correlation_matrix(
        self,
        symbols: List[str],
        window_hours: int = 168
    ) -> np.ndarray:
        """Get current correlation matrix for given symbols."""
        return self.portfolio_analyzer.calculate_correlation_matrix(symbols, window_hours)
    
    def get_pairwise_correlation(
        self,
        symbol1: str,
        symbol2: str,
        window_hours: int = 24
    ) -> float:
        """Get current correlation between two symbols."""
        return self.correlation_calculator.calculate_pearson_correlation(symbol1, symbol2, window_hours)