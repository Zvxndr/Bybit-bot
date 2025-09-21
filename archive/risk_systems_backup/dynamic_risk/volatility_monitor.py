"""
Adaptive Volatility Monitoring System.

This module provides sophisticated volatility analysis and regime detection for
dynamic risk management including:

- Multi-timeframe volatility estimation (GARCH, EWMA, realized volatility)
- Volatility regime detection and classification
- Dynamic volatility forecasting with confidence intervals
- Volatility surface modeling for options-like instruments
- Cross-asset volatility correlation analysis
- Volatility breakout and mean reversion detection
- Real-time volatility alerts and monitoring
- Historical volatility percentile analysis
- Volatility clustering detection
- Risk-adjusted position sizing based on volatility regimes

The system continuously monitors market volatility across multiple assets and
timeframes to provide real-time risk adjustment signals for portfolio management.
"""

import asyncio
import threading
import time
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
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
from scipy.optimize import minimize
import sqlite3
import json

from ..utils.logging import TradingLogger


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    VERY_LOW = "very_low"       # <10th percentile
    LOW = "low"                 # 10th-25th percentile
    NORMAL = "normal"           # 25th-75th percentile
    HIGH = "high"               # 75th-90th percentile
    VERY_HIGH = "very_high"     # >90th percentile
    EXTREME = "extreme"         # >99th percentile


class TrendState(Enum):
    """Volatility trend state."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class VolatilityMetrics:
    """Container for comprehensive volatility metrics."""
    
    symbol: str
    timestamp: datetime
    
    # Basic volatility measures
    realized_vol_1h: float      # 1-hour realized volatility (annualized)
    realized_vol_4h: float      # 4-hour realized volatility
    realized_vol_24h: float     # 24-hour realized volatility
    realized_vol_7d: float      # 7-day realized volatility
    
    # Advanced volatility estimates
    garch_vol: float            # GARCH(1,1) volatility forecast
    ewma_vol: float             # EWMA volatility estimate
    parkinson_vol: float        # Parkinson high-low volatility
    garman_klass_vol: float     # Garman-Klass OHLC volatility
    
    # Regime classification
    vol_regime: VolatilityRegime
    regime_confidence: float    # Confidence in regime classification
    regime_duration: timedelta  # Time in current regime
    
    # Trend analysis
    vol_trend: TrendState
    trend_strength: float       # Strength of volatility trend (0-1)
    trend_duration: timedelta   # Duration of current trend
    
    # Statistical measures
    vol_percentile: float       # Current vol vs historical distribution
    vol_zscore: float          # Z-score vs historical mean
    vol_skewness: float        # Skewness of recent volatility
    vol_kurtosis: float        # Kurtosis of recent volatility
    
    # Forecasting
    vol_forecast_1h: float      # 1-hour ahead forecast
    vol_forecast_4h: float      # 4-hour ahead forecast
    vol_forecast_24h: float     # 24-hour ahead forecast
    forecast_confidence: float  # Confidence in forecast
    
    # Risk metrics
    downside_vol: float         # Downside volatility
    upside_vol: float          # Upside volatility
    vol_of_vol: float          # Volatility of volatility
    max_vol_1d: float          # Maximum vol in last 24h
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'realized_vol_1h': self.realized_vol_1h,
            'realized_vol_4h': self.realized_vol_4h,
            'realized_vol_24h': self.realized_vol_24h,
            'realized_vol_7d': self.realized_vol_7d,
            'garch_vol': self.garch_vol,
            'ewma_vol': self.ewma_vol,
            'parkinson_vol': self.parkinson_vol,
            'garman_klass_vol': self.garman_klass_vol,
            'vol_regime': self.vol_regime.value,
            'regime_confidence': self.regime_confidence,
            'regime_duration': str(self.regime_duration),
            'vol_trend': self.vol_trend.value,
            'trend_strength': self.trend_strength,
            'trend_duration': str(self.trend_duration),
            'vol_percentile': self.vol_percentile,
            'vol_zscore': self.vol_zscore,
            'vol_skewness': self.vol_skewness,
            'vol_kurtosis': self.vol_kurtosis,
            'vol_forecast_1h': self.vol_forecast_1h,
            'vol_forecast_4h': self.vol_forecast_4h,
            'vol_forecast_24h': self.vol_forecast_24h,
            'forecast_confidence': self.forecast_confidence,
            'downside_vol': self.downside_vol,
            'upside_vol': self.upside_vol,
            'vol_of_vol': self.vol_of_vol,
            'max_vol_1d': self.max_vol_1d
        }


class GARCHModel:
    """
    GARCH(1,1) volatility model implementation.
    
    This class implements a simplified GARCH(1,1) model for volatility forecasting
    using maximum likelihood estimation.
    """
    
    def __init__(self):
        self.omega = 0.0001  # Long-term variance
        self.alpha = 0.1     # ARCH coefficient
        self.beta = 0.85     # GARCH coefficient
        self.is_fitted = False
        
    def fit(self, returns: np.ndarray, max_iter: int = 100) -> bool:
        """Fit GARCH model to return series."""
        try:
            if len(returns) < 50:  # Need sufficient data
                return False
            
            # Initial parameter guesses
            initial_params = [0.0001, 0.1, 0.85]  # omega, alpha, beta
            
            # Parameter bounds
            bounds = [(1e-6, 1), (0.01, 0.99), (0.01, 0.99)]
            
            # Constraint: alpha + beta < 1 for stationarity
            constraints = {'type': 'ineq', 'fun': lambda x: 0.99 - x[1] - x[2]}
            
            # Optimize parameters
            result = minimize(
                self._log_likelihood,
                initial_params,
                args=(returns,),
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': max_iter}
            )
            
            if result.success:
                self.omega, self.alpha, self.beta = result.x
                self.is_fitted = True
                return True
            
            return False
            
        except Exception:
            return False
    
    def _log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Calculate negative log-likelihood for GARCH model."""
        omega, alpha, beta = params
        
        # Initialize variance
        variance = np.var(returns)
        log_likelihood = 0
        
        for ret in returns:
            # GARCH(1,1) variance equation
            variance = omega + alpha * ret**2 + beta * variance
            
            # Ensure positive variance
            variance = max(variance, 1e-8)
            
            # Log-likelihood contribution
            log_likelihood += -0.5 * (np.log(2 * np.pi * variance) + ret**2 / variance)
        
        return -log_likelihood  # Return negative for minimization
    
    def forecast(self, returns: np.ndarray, horizon: int = 1) -> float:
        """Forecast volatility for given horizon."""
        if not self.is_fitted or len(returns) == 0:
            return 0.0
        
        # Get latest return and variance
        latest_return = returns[-1]
        
        # Calculate current conditional variance
        variance = self.omega
        for ret in returns[-min(100, len(returns)):]:  # Use recent history
            variance = self.omega + self.alpha * ret**2 + self.beta * variance
        
        # Multi-step forecast
        forecast_variance = variance
        for _ in range(horizon):
            forecast_variance = self.omega + (self.alpha + self.beta) * forecast_variance
        
        return math.sqrt(forecast_variance)


class EWMAModel:
    """
    Exponentially Weighted Moving Average volatility model.
    
    This class implements EWMA volatility estimation with configurable
    decay factor for different time horizons.
    """
    
    def __init__(self, lambda_param: float = 0.94):
        self.lambda_param = lambda_param
        self.variance = None
        
    def update(self, returns: np.ndarray) -> float:
        """Update EWMA volatility estimate."""
        if len(returns) == 0:
            return 0.0
        
        if self.variance is None:
            # Initialize with sample variance
            self.variance = np.var(returns) if len(returns) > 1 else returns[0]**2
        
        # Update EWMA variance
        for ret in returns:
            self.variance = self.lambda_param * self.variance + (1 - self.lambda_param) * ret**2
        
        return math.sqrt(self.variance)
    
    def forecast(self, horizon: int = 1) -> float:
        """Forecast volatility for given horizon."""
        if self.variance is None:
            return 0.0
        
        # EWMA forecast (constant volatility assumption)
        return math.sqrt(self.variance)


class VolatilityEstimator:
    """
    Multi-method volatility estimator combining various approaches.
    
    This class provides comprehensive volatility estimation using
    multiple methodologies for robust volatility measurement.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("VolatilityEstimator")
        
        # Volatility models
        self.garch_models: Dict[str, GARCHModel] = {}
        self.ewma_models: Dict[str, EWMAModel] = {}
        
        # Data storage
        self.price_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.ohlc_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
    def _default_config(self) -> Dict:
        """Default configuration for volatility estimator."""
        return {
            'min_observations': 100,     # Minimum data points for estimation
            'garch_update_freq': 3600,   # Update GARCH model every hour
            'ewma_lambda': 0.94,         # EWMA decay factor
            'annualization_factor': 365.25 * 24,  # For hourly data to annual
            'outlier_threshold': 5.0,    # Z-score threshold for outliers
            'regime_lookback': 2160,     # 90 days of hourly data
        }
    
    def add_price_data(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """Add price data point for volatility calculation."""
        timestamp = timestamp or datetime.now()
        
        data_point = {
            'price': price,
            'timestamp': timestamp
        }
        
        self.price_data[symbol].append(data_point)
    
    def add_ohlc_data(
        self,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add OHLC data for advanced volatility estimates."""
        timestamp = timestamp or datetime.now()
        
        ohlc_point = {
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'timestamp': timestamp
        }
        
        self.ohlc_data[symbol].append(ohlc_point)
    
    def calculate_realized_volatility(
        self,
        symbol: str,
        window_hours: int = 24
    ) -> float:
        """Calculate realized volatility from price data."""
        price_data = list(self.price_data[symbol])
        
        if len(price_data) < 2:
            return 0.0
        
        # Filter data for time window
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_data = [
            d for d in price_data
            if d['timestamp'] > cutoff_time
        ]
        
        if len(recent_data) < 2:
            return 0.0
        
        # Calculate returns
        prices = [d['price'] for d in recent_data]
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = math.log(prices[i] / prices[i-1])
                returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        # Calculate realized volatility
        variance = np.var(returns, ddof=1)
        
        # Annualize volatility
        periods_per_year = self.config['annualization_factor'] / window_hours
        annual_volatility = math.sqrt(variance * periods_per_year)
        
        return annual_volatility
    
    def calculate_parkinson_volatility(self, symbol: str, window_hours: int = 24) -> float:
        """Calculate Parkinson high-low volatility estimator."""
        ohlc_data = list(self.ohlc_data[symbol])
        
        if len(ohlc_data) < 2:
            return 0.0
        
        # Filter data for time window
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_data = [
            d for d in ohlc_data
            if d['timestamp'] > cutoff_time
        ]
        
        if len(recent_data) < 2:
            return 0.0
        
        # Calculate Parkinson estimator
        hl_ratios = []
        for d in recent_data:
            if d['low'] > 0 and d['high'] > d['low']:
                hl_ratio = math.log(d['high'] / d['low'])
                hl_ratios.append(hl_ratio**2)
        
        if not hl_ratios:
            return 0.0
        
        # Parkinson volatility
        parkinson_var = np.mean(hl_ratios) / (4 * math.log(2))
        
        # Annualize
        periods_per_year = self.config['annualization_factor'] / window_hours
        annual_volatility = math.sqrt(parkinson_var * periods_per_year)
        
        return annual_volatility
    
    def calculate_garman_klass_volatility(self, symbol: str, window_hours: int = 24) -> float:
        """Calculate Garman-Klass OHLC volatility estimator."""
        ohlc_data = list(self.ohlc_data[symbol])
        
        if len(ohlc_data) < 2:
            return 0.0
        
        # Filter data for time window
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_data = [
            d for d in ohlc_data
            if d['timestamp'] > cutoff_time
        ]
        
        if len(recent_data) < 2:
            return 0.0
        
        # Calculate Garman-Klass estimator
        gk_terms = []
        for d in recent_data:
            if d['low'] > 0 and d['high'] > d['low'] and d['close'] > 0 and d['open'] > 0:
                # Garman-Klass formula
                term1 = 0.5 * (math.log(d['high'] / d['low']))**2
                term2 = (2 * math.log(2) - 1) * (math.log(d['close'] / d['open']))**2
                gk_terms.append(term1 - term2)
        
        if not gk_terms:
            return 0.0
        
        # Garman-Klass volatility
        gk_var = np.mean(gk_terms)
        
        # Annualize
        periods_per_year = self.config['annualization_factor'] / window_hours
        annual_volatility = math.sqrt(gk_var * periods_per_year)
        
        return annual_volatility
    
    def calculate_garch_volatility(self, symbol: str) -> float:
        """Calculate GARCH volatility forecast."""
        if symbol not in self.garch_models:
            self.garch_models[symbol] = GARCHModel()
        
        garch_model = self.garch_models[symbol]
        
        # Get return data
        returns = self._get_returns(symbol, lookback_hours=720)  # 30 days
        
        if len(returns) < self.config['min_observations']:
            return 0.0
        
        # Fit model if not fitted or periodically refit
        if not garch_model.is_fitted:
            if not garch_model.fit(returns):
                return 0.0
        
        # Get forecast
        forecast_vol = garch_model.forecast(returns, horizon=1)
        
        # Annualize
        annual_vol = forecast_vol * math.sqrt(self.config['annualization_factor'])
        
        return annual_vol
    
    def calculate_ewma_volatility(self, symbol: str) -> float:
        """Calculate EWMA volatility estimate."""
        if symbol not in self.ewma_models:
            self.ewma_models[symbol] = EWMAModel(self.config['ewma_lambda'])
        
        ewma_model = self.ewma_models[symbol]
        
        # Get recent returns
        returns = self._get_returns(symbol, lookback_hours=168)  # 1 week
        
        if len(returns) < 10:
            return 0.0
        
        # Update EWMA
        vol = ewma_model.update(returns)
        
        # Annualize
        annual_vol = vol * math.sqrt(self.config['annualization_factor'])
        
        return annual_vol
    
    def _get_returns(self, symbol: str, lookback_hours: int = 24) -> np.ndarray:
        """Get return series for volatility calculation."""
        price_data = list(self.price_data[symbol])
        
        if len(price_data) < 2:
            return np.array([])
        
        # Filter for lookback period
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_data = [
            d for d in price_data
            if d['timestamp'] > cutoff_time
        ]
        
        if len(recent_data) < 2:
            return np.array([])
        
        # Calculate returns
        prices = [d['price'] for d in recent_data]
        returns = []
        
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = math.log(prices[i] / prices[i-1])
                returns.append(ret)
        
        return np.array(returns)
    
    def calculate_downside_volatility(self, symbol: str, window_hours: int = 24) -> float:
        """Calculate downside volatility (volatility of negative returns)."""
        returns = self._get_returns(symbol, window_hours)
        
        if len(returns) < 10:
            return 0.0
        
        # Filter negative returns
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) < 5:
            return 0.0
        
        # Calculate downside volatility
        downside_var = np.var(negative_returns, ddof=1)
        periods_per_year = self.config['annualization_factor'] / window_hours
        annual_vol = math.sqrt(downside_var * periods_per_year)
        
        return annual_vol
    
    def calculate_upside_volatility(self, symbol: str, window_hours: int = 24) -> float:
        """Calculate upside volatility (volatility of positive returns)."""
        returns = self._get_returns(symbol, window_hours)
        
        if len(returns) < 10:
            return 0.0
        
        # Filter positive returns
        positive_returns = returns[returns > 0]
        
        if len(positive_returns) < 5:
            return 0.0
        
        # Calculate upside volatility
        upside_var = np.var(positive_returns, ddof=1)
        periods_per_year = self.config['annualization_factor'] / window_hours
        annual_vol = math.sqrt(upside_var * periods_per_year)
        
        return annual_vol
    
    def calculate_volatility_of_volatility(self, symbol: str) -> float:
        """Calculate volatility of volatility metric."""
        # Get historical volatility estimates
        vol_estimates = []
        
        # Calculate rolling volatility over past periods
        for i in range(7, 0, -1):  # Last 7 days
            start_hours = i * 24
            end_hours = (i - 1) * 24
            
            vol = self.calculate_realized_volatility(symbol, window_hours=24)
            if vol > 0:
                vol_estimates.append(vol)
        
        if len(vol_estimates) < 3:
            return 0.0
        
        # Calculate volatility of the volatility estimates
        vol_of_vol = np.std(vol_estimates, ddof=1) if len(vol_estimates) > 1 else 0.0
        
        return vol_of_vol


class VolatilityRegimeDetector:
    """
    Volatility regime detection and classification system.
    
    This class identifies different volatility regimes and tracks
    regime transitions for dynamic risk management.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("VolatilityRegimeDetector")
        
        # Historical volatility for regime detection
        self.volatility_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2160))  # 90 days hourly
        
        # Current regimes
        self.current_regimes: Dict[str, VolatilityRegime] = {}
        self.regime_start_times: Dict[str, datetime] = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for regime detector."""
        return {
            'regime_thresholds': {
                'very_low': 0.10,     # 10th percentile
                'low': 0.25,          # 25th percentile
                'high': 0.75,         # 75th percentile
                'very_high': 0.90,    # 90th percentile
                'extreme': 0.99       # 99th percentile
            },
            'min_regime_duration': 3600,  # Minimum 1 hour in regime
            'regime_smoothing': 0.8,      # Smoothing factor for regime transitions
            'lookback_days': 90,          # Days of history for percentiles
        }
    
    def add_volatility_observation(self, symbol: str, volatility: float, timestamp: Optional[datetime] = None) -> None:
        """Add volatility observation for regime detection."""
        timestamp = timestamp or datetime.now()
        
        vol_point = {
            'volatility': volatility,
            'timestamp': timestamp
        }
        
        self.volatility_history[symbol].append(vol_point)
    
    def detect_regime(self, symbol: str, current_volatility: float) -> Tuple[VolatilityRegime, float]:
        """Detect current volatility regime and confidence."""
        vol_history = list(self.volatility_history[symbol])
        
        if len(vol_history) < 50:  # Need sufficient history
            return VolatilityRegime.NORMAL, 0.5
        
        # Extract volatility values
        historical_vols = [v['volatility'] for v in vol_history if v['volatility'] > 0]
        
        if len(historical_vols) < 50:
            return VolatilityRegime.NORMAL, 0.5
        
        # Calculate percentiles
        percentiles = np.percentile(historical_vols, [10, 25, 75, 90, 99])
        p10, p25, p75, p90, p99 = percentiles
        
        # Classify regime
        if current_volatility <= p10:
            regime = VolatilityRegime.VERY_LOW
            confidence = (p10 - current_volatility) / p10 if p10 > 0 else 0.5
        elif current_volatility <= p25:
            regime = VolatilityRegime.LOW
            confidence = (p25 - current_volatility) / (p25 - p10) if p25 > p10 else 0.5
        elif current_volatility <= p75:
            regime = VolatilityRegime.NORMAL
            confidence = 0.8  # High confidence in normal regime
        elif current_volatility <= p90:
            regime = VolatilityRegime.HIGH
            confidence = (current_volatility - p75) / (p90 - p75) if p90 > p75 else 0.5
        elif current_volatility <= p99:
            regime = VolatilityRegime.VERY_HIGH
            confidence = (current_volatility - p90) / (p99 - p90) if p99 > p90 else 0.5
        else:
            regime = VolatilityRegime.EXTREME
            confidence = min(1.0, (current_volatility - p99) / p99) if p99 > 0 else 0.5
        
        # Smooth regime transitions
        if symbol in self.current_regimes:
            old_regime = self.current_regimes[symbol]
            if old_regime != regime:
                # Check minimum duration
                if symbol in self.regime_start_times:
                    duration = datetime.now() - self.regime_start_times[symbol]
                    if duration.total_seconds() < self.config['min_regime_duration']:
                        # Stay in old regime
                        return old_regime, confidence * 0.8
        
        # Update regime tracking
        if symbol not in self.current_regimes or self.current_regimes[symbol] != regime:
            self.current_regimes[symbol] = regime
            self.regime_start_times[symbol] = datetime.now()
        
        return regime, min(1.0, max(0.0, confidence))
    
    def get_regime_duration(self, symbol: str) -> timedelta:
        """Get duration of current volatility regime."""
        if symbol not in self.regime_start_times:
            return timedelta(0)
        
        return datetime.now() - self.regime_start_times[symbol]
    
    def detect_trend(self, symbol: str) -> Tuple[TrendState, float]:
        """Detect volatility trend and strength."""
        vol_history = list(self.volatility_history[symbol])
        
        if len(vol_history) < 20:
            return TrendState.STABLE, 0.0
        
        # Get recent volatility values
        recent_vols = [v['volatility'] for v in vol_history[-20:]]
        
        if len(recent_vols) < 10:
            return TrendState.STABLE, 0.0
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_vols))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_vols)
        
        # Determine trend direction and strength
        trend_strength = abs(r_value)  # Use correlation coefficient as strength
        
        if p_value > 0.05:  # Not statistically significant
            return TrendState.STABLE, 0.0
        
        if abs(slope) < np.std(recent_vols) * 0.1:  # Small slope relative to volatility
            return TrendState.STABLE, trend_strength
        
        if slope > 0:
            trend = TrendState.INCREASING
        else:
            trend = TrendState.DECREASING
        
        # Check for high volatility of volatility (volatile regime)
        vol_changes = np.diff(recent_vols)
        if np.std(vol_changes) > np.mean(recent_vols) * 0.2:
            trend = TrendState.VOLATILE
            trend_strength = np.std(vol_changes) / np.mean(recent_vols)
        
        return trend, min(1.0, trend_strength)


class AdaptiveVolatilityMonitor:
    """
    Main adaptive volatility monitoring system.
    
    This class combines all volatility analysis components to provide
    comprehensive volatility monitoring and regime detection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("AdaptiveVolatilityMonitor")
        
        # Core components
        self.volatility_estimator = VolatilityEstimator(self.config.get('estimator', {}))
        self.regime_detector = VolatilityRegimeDetector(self.config.get('regime_detector', {}))
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Callbacks for regime changes
        self.regime_callbacks: List[Callable[[str, VolatilityMetrics], None]] = []
        
        # Database for persistence
        self.db_path = self.config.get('database_path', 'volatility_monitor.db')
        self._init_database()
        
    def _default_config(self) -> Dict:
        """Default configuration for volatility monitor."""
        return {
            'update_interval': 300,       # 5 minutes
            'database_path': 'volatility_monitor.db',
            'enable_persistence': True,
            'alert_thresholds': {
                'regime_change': True,
                'extreme_volatility': 0.99,
                'volatility_spike': 2.0    # 2x normal volatility
            },
            'estimator': {},
            'regime_detector': {}
        }
    
    def _init_database(self) -> None:
        """Initialize SQLite database for volatility data."""
        if not self.config['enable_persistence']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS volatility_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vol_symbol_time ON volatility_metrics (symbol, timestamp)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize volatility database: {e}")
    
    def add_market_data(
        self,
        symbol: str,
        price: float,
        open_price: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
        volume: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add market data for volatility analysis."""
        # Add price data
        self.volatility_estimator.add_price_data(symbol, price, timestamp)
        
        # Add OHLC data if available
        if all(x is not None for x in [open_price, high, low]):
            self.volatility_estimator.add_ohlc_data(
                symbol, open_price, high, low, price, volume, timestamp
            )
    
    def calculate_comprehensive_metrics(self, symbol: str) -> Optional[VolatilityMetrics]:
        """Calculate comprehensive volatility metrics for a symbol."""
        try:
            # Calculate various volatility measures
            realized_vol_1h = self.volatility_estimator.calculate_realized_volatility(symbol, 1)
            realized_vol_4h = self.volatility_estimator.calculate_realized_volatility(symbol, 4)
            realized_vol_24h = self.volatility_estimator.calculate_realized_volatility(symbol, 24)
            realized_vol_7d = self.volatility_estimator.calculate_realized_volatility(symbol, 168)
            
            garch_vol = self.volatility_estimator.calculate_garch_volatility(symbol)
            ewma_vol = self.volatility_estimator.calculate_ewma_volatility(symbol)
            parkinson_vol = self.volatility_estimator.calculate_parkinson_volatility(symbol)
            garman_klass_vol = self.volatility_estimator.calculate_garman_klass_volatility(symbol)
            
            # Use 24h realized vol as primary measure
            primary_vol = realized_vol_24h
            if primary_vol <= 0:
                primary_vol = ewma_vol
            if primary_vol <= 0:
                return None
            
            # Add volatility observation for regime detection
            self.regime_detector.add_volatility_observation(symbol, primary_vol)
            
            # Detect regime and trend
            vol_regime, regime_confidence = self.regime_detector.detect_regime(symbol, primary_vol)
            regime_duration = self.regime_detector.get_regime_duration(symbol)
            vol_trend, trend_strength = self.regime_detector.detect_trend(symbol)
            
            # Calculate statistical measures
            vol_history = [v['volatility'] for v in self.regime_detector.volatility_history[symbol]]
            if len(vol_history) >= 30:
                vol_percentile = stats.percentileofscore(vol_history, primary_vol) / 100
                vol_mean = np.mean(vol_history)
                vol_std = np.std(vol_history)
                vol_zscore = (primary_vol - vol_mean) / vol_std if vol_std > 0 else 0
                vol_skewness = stats.skew(vol_history)
                vol_kurtosis = stats.kurtosis(vol_history)
            else:
                vol_percentile = 0.5
                vol_zscore = 0.0
                vol_skewness = 0.0
                vol_kurtosis = 0.0
            
            # Simple forecasting (can be enhanced with more sophisticated models)
            vol_forecast_1h = garch_vol if garch_vol > 0 else primary_vol
            vol_forecast_4h = vol_forecast_1h * 0.98  # Slight mean reversion
            vol_forecast_24h = vol_forecast_1h * 0.95
            forecast_confidence = 0.7 if garch_vol > 0 else 0.5
            
            # Risk metrics
            downside_vol = self.volatility_estimator.calculate_downside_volatility(symbol)
            upside_vol = self.volatility_estimator.calculate_upside_volatility(symbol)
            vol_of_vol = self.volatility_estimator.calculate_volatility_of_volatility(symbol)
            
            # Maximum volatility in last 24h
            recent_vols = [v['volatility'] for v in self.regime_detector.volatility_history[symbol][-24:]]
            max_vol_1d = max(recent_vols) if recent_vols else primary_vol
            
            # Create metrics object
            metrics = VolatilityMetrics(
                symbol=symbol,
                timestamp=datetime.now(),
                realized_vol_1h=realized_vol_1h,
                realized_vol_4h=realized_vol_4h,
                realized_vol_24h=realized_vol_24h,
                realized_vol_7d=realized_vol_7d,
                garch_vol=garch_vol,
                ewma_vol=ewma_vol,
                parkinson_vol=parkinson_vol,
                garman_klass_vol=garman_klass_vol,
                vol_regime=vol_regime,
                regime_confidence=regime_confidence,
                regime_duration=regime_duration,
                vol_trend=vol_trend,
                trend_strength=trend_strength,
                trend_duration=timedelta(0),  # Would need to track trend changes
                vol_percentile=vol_percentile,
                vol_zscore=vol_zscore,
                vol_skewness=vol_skewness,
                vol_kurtosis=vol_kurtosis,
                vol_forecast_1h=vol_forecast_1h,
                vol_forecast_4h=vol_forecast_4h,
                vol_forecast_24h=vol_forecast_24h,
                forecast_confidence=forecast_confidence,
                downside_vol=downside_vol,
                upside_vol=upside_vol,
                vol_of_vol=vol_of_vol,
                max_vol_1d=max_vol_1d
            )
            
            # Save to database
            if self.config['enable_persistence']:
                self._save_metrics(metrics)
            
            # Check for alerts
            self._check_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics for {symbol}: {e}")
            return None
    
    def _save_metrics(self, metrics: VolatilityMetrics) -> None:
        """Save volatility metrics to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            data = json.dumps(metrics.to_dict())
            
            cursor.execute("""
                INSERT INTO volatility_metrics (symbol, timestamp, data)
                VALUES (?, ?, ?)
            """, (metrics.symbol, metrics.timestamp, data))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save volatility metrics: {e}")
    
    def _check_alerts(self, metrics: VolatilityMetrics) -> None:
        """Check volatility metrics for alert conditions."""
        alerts = []
        
        # Regime change alert
        if self.config['alert_thresholds']['regime_change']:
            if metrics.vol_regime in [VolatilityRegime.VERY_HIGH, VolatilityRegime.EXTREME]:
                alerts.append(f"High volatility regime detected for {metrics.symbol}: {metrics.vol_regime.value}")
        
        # Extreme volatility alert
        extreme_threshold = self.config['alert_thresholds']['extreme_volatility']
        if metrics.vol_percentile > extreme_threshold:
            alerts.append(f"Extreme volatility for {metrics.symbol}: {metrics.vol_percentile:.1%} percentile")
        
        # Volatility spike alert
        spike_threshold = self.config['alert_thresholds']['volatility_spike']
        if metrics.vol_zscore > spike_threshold:
            alerts.append(f"Volatility spike for {metrics.symbol}: {metrics.vol_zscore:.1f} standard deviations")
        
        # Notify callbacks
        if alerts:
            for callback in self.regime_callbacks:
                try:
                    callback(metrics.symbol, metrics)
                except Exception as e:
                    self.logger.error(f"Error in volatility callback: {e}")
            
            self.logger.warning(f"Volatility alerts: {alerts}")
    
    def add_regime_callback(self, callback: Callable[[str, VolatilityMetrics], None]) -> None:
        """Add callback for volatility regime changes."""
        self.regime_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start background volatility monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started volatility monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background volatility monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped volatility monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Get all symbols being monitored
                symbols = set(self.volatility_estimator.price_data.keys())
                
                # Calculate metrics for each symbol
                for symbol in symbols:
                    metrics = self.calculate_comprehensive_metrics(symbol)
                    if metrics:
                        self.logger.debug(
                            f"Updated volatility metrics for {symbol}: "
                            f"vol={metrics.realized_vol_24h:.2%}, "
                            f"regime={metrics.vol_regime.value}"
                        )
                
                # Sleep until next update
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in volatility monitoring loop: {e}")
                time.sleep(60)  # Error backoff
    
    def get_current_metrics(self, symbol: str) -> Optional[VolatilityMetrics]:
        """Get current volatility metrics for a symbol."""
        return self.calculate_comprehensive_metrics(symbol)
    
    def get_historical_metrics(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get historical volatility metrics from database."""
        if not self.config['enable_persistence']:
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT data FROM volatility_metrics
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, (symbol, start_date, end_date))
            
            results = []
            for row in cursor.fetchall():
                data = json.loads(row[0])
                results.append(data)
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve historical volatility metrics: {e}")
            return []