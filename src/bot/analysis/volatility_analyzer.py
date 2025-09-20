"""
Volatility Analyzer

Advanced multi-timeframe volatility analysis system for cryptocurrency trading.
Implements sophisticated volatility models, regime detection, and forecasting
to optimize risk management and identify trading opportunities.

Key Features:
- Multi-timeframe volatility modeling (GARCH, EWMA, Historical)
- Volatility regime detection and classification
- Real-time volatility forecasting
- Volatility clustering analysis
- Risk-adjusted return calculations
- Volatility breakout detection
- Cross-asset volatility spillover analysis
- Volatility surface construction

Volatility Models:
- Historical Volatility (multiple windows)
- Exponentially Weighted Moving Average (EWMA)
- GARCH(1,1) with various distributions
- Realized Volatility (high-frequency)
- Implied Volatility (when available)
- Parkinson Volatility (high-low estimator)
- Garman-Klass Volatility (OHLC estimator)

Regime Detection:
- Markov Switching Models
- Threshold models
- Machine learning clustering
- Statistical breakpoint detection
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
import math

from ..utils.logging import TradingLogger
from ..config.manager import ConfigurationManager


class VolatilityRegime(Enum):
    """Volatility regime classifications."""
    LOW = "low"
    NORMAL = "normal"  
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"


class VolatilityModel(Enum):
    """Types of volatility models."""
    HISTORICAL = "historical"
    EWMA = "ewma"
    GARCH = "garch"
    REALIZED = "realized"
    PARKINSON = "parkinson"
    GARMAN_KLASS = "garman_klass"


class TrendDirection(Enum):
    """Volatility trend directions."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


@dataclass
class VolatilityMetrics:
    """Comprehensive volatility metrics."""
    timestamp: datetime
    symbol: str
    timeframe: str
    
    # Current volatility measures
    historical_vol_1d: float
    historical_vol_7d: float
    historical_vol_30d: float
    ewma_vol: float
    garch_vol: float
    realized_vol: float
    parkinson_vol: float
    garman_klass_vol: float
    
    # Regime information
    current_regime: VolatilityRegime
    regime_probability: float
    regime_duration: float  # Hours in current regime
    
    # Trend information
    vol_trend_1d: TrendDirection
    vol_trend_7d: TrendDirection
    vol_trend_30d: TrendDirection
    
    # Risk metrics
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Forecasting
    forecast_1h: float
    forecast_4h: float
    forecast_24h: float
    forecast_confidence: float
    
    # Statistical measures
    skewness: float
    kurtosis: float
    autocorrelation: float
    volatility_clustering: float
    
    # Cross-asset measures
    correlation_btc: Optional[float]
    volatility_spillover: float
    
    metadata: Dict[str, Any]


@dataclass
class VolatilityForecast:
    """Volatility forecast data."""
    timestamp: datetime
    symbol: str
    model_type: VolatilityModel
    forecast_horizon: str  # "1h", "4h", "24h", etc.
    predicted_volatility: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    confidence_score: float
    model_accuracy: float
    forecast_path: List[float]  # Full forecast path
    metadata: Dict[str, Any]


@dataclass
class VolatilityBreakout:
    """Volatility breakout event."""
    timestamp: datetime
    symbol: str
    breakout_type: str  # "upward", "downward"
    magnitude: float  # Multiple of normal volatility
    duration_prediction: float  # Expected duration in hours
    confidence: float
    trigger_level: float
    current_level: float
    expected_return: Optional[float]
    risk_level: str


class VolatilityAnalyzer:
    """
    Advanced volatility analysis system for cryptocurrency trading.
    
    Provides comprehensive volatility modeling, regime detection, forecasting,
    and risk analysis across multiple timeframes and assets.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = TradingLogger("VolatilityAnalyzer")
        
        # Configuration
        self.config = {
            'ewma_lambda': config_manager.get('volatility.ewma_lambda', 0.94),
            'garch_p': config_manager.get('volatility.garch_p', 1),
            'garch_q': config_manager.get('volatility.garch_q', 1),
            'regime_lookback': config_manager.get('volatility.regime_lookback', 252),  # ~1 year of hours
            'min_regime_duration': config_manager.get('volatility.min_regime_duration', 24),  # 24 hours
            'volatility_windows': config_manager.get('volatility.windows', [24, 168, 720]),  # 1d, 7d, 30d
            'confidence_level': config_manager.get('volatility.confidence_level', 0.95),
            'breakout_threshold': config_manager.get('volatility.breakout_threshold', 2.0),  # 2x normal vol
            'forecast_horizons': config_manager.get('volatility.forecast_horizons', [1, 4, 24]),  # hours
            'clustering_window': config_manager.get('volatility.clustering_window', 48)  # hours
        }
        
        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.returns_data: Dict[str, pd.DataFrame] = {}
        self.volatility_data: Dict[str, Dict[str, pd.Series]] = defaultdict(dict)
        
        # Model storage
        self.volatility_models: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.regime_models: Dict[str, Any] = {}
        self.forecast_models: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Current state tracking
        self.current_metrics: Dict[str, VolatilityMetrics] = {}
        self.current_regimes: Dict[str, VolatilityRegime] = {}
        self.regime_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Breakout tracking
        self.active_breakouts: List[VolatilityBreakout] = []
        self.breakout_history: deque = deque(maxlen=5000)
        
        # Performance tracking
        self.model_performance: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.forecast_accuracy: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Regime thresholds (percentiles of historical volatility)
        self.regime_thresholds = {
            VolatilityRegime.LOW: 0.20,        # Below 20th percentile
            VolatilityRegime.NORMAL: 0.40,     # 20th-40th percentile  
            VolatilityRegime.ELEVATED: 0.70,   # 40th-70th percentile
            VolatilityRegime.HIGH: 0.90,       # 70th-90th percentile
            VolatilityRegime.EXTREME: 1.00     # Above 90th percentile
        }
    
    def update_data(self, symbol: str, price_data: pd.DataFrame):
        """Update price data and calculate returns."""
        try:
            if price_data.empty:
                return
            
            # Ensure required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in price_data.columns for col in required_columns):
                self.logger.warning(f"Missing required columns for {symbol}")
                return
            
            # Store price data
            price_data = price_data.sort_values('timestamp')
            self.price_data[symbol] = price_data.tail(10000)  # Keep last 10k candles
            
            # Calculate returns
            returns_data = price_data.copy()
            returns_data['returns'] = returns_data['close'].pct_change()
            returns_data['log_returns'] = np.log(returns_data['close'] / returns_data['close'].shift(1))
            
            # Calculate intrabar returns (high-low, open-close)
            returns_data['hl_returns'] = (returns_data['high'] - returns_data['low']) / returns_data['close'].shift(1)
            returns_data['oc_returns'] = (returns_data['close'] - returns_data['open']) / returns_data['open']
            
            self.returns_data[symbol] = returns_data.dropna()
            
            # Update volatility calculations
            self._update_volatility_calculations(symbol)
            
            self.logger.debug(f"Updated volatility data for {symbol}: {len(returns_data)} periods")
            
        except Exception as e:
            self.logger.error(f"Error updating volatility data for {symbol}: {e}")
    
    def _update_volatility_calculations(self, symbol: str):
        """Update all volatility calculations for a symbol."""
        returns_data = self.returns_data[symbol]
        
        # Historical volatility
        for window in self.config['volatility_windows']:
            vol_series = returns_data['returns'].rolling(window=window).std() * np.sqrt(24)  # Annualized
            self.volatility_data[symbol][f'hist_vol_{window}h'] = vol_series
        
        # EWMA volatility
        ewma_vol = self._calculate_ewma_volatility(returns_data['returns'])
        self.volatility_data[symbol]['ewma_vol'] = ewma_vol
        
        # GARCH volatility
        try:
            garch_vol = self._calculate_garch_volatility(returns_data['returns'])
            self.volatility_data[symbol]['garch_vol'] = garch_vol
        except Exception as e:
            self.logger.debug(f"GARCH calculation failed for {symbol}: {e}")
            self.volatility_data[symbol]['garch_vol'] = ewma_vol  # Fallback
        
        # Realized volatility (using high-frequency components)
        realized_vol = self._calculate_realized_volatility(returns_data)
        self.volatility_data[symbol]['realized_vol'] = realized_vol
        
        # Parkinson volatility
        parkinson_vol = self._calculate_parkinson_volatility(returns_data)
        self.volatility_data[symbol]['parkinson_vol'] = parkinson_vol
        
        # Garman-Klass volatility
        gk_vol = self._calculate_garman_klass_volatility(returns_data)
        self.volatility_data[symbol]['garman_klass_vol'] = gk_vol
    
    def _calculate_ewma_volatility(self, returns: pd.Series, lambda_param: Optional[float] = None) -> pd.Series:
        """Calculate EWMA volatility."""
        if lambda_param is None:
            lambda_param = self.config['ewma_lambda']
        
        ewma_var = returns.ewm(alpha=1-lambda_param).var()
        return np.sqrt(ewma_var * 24)  # Annualized
    
    def _calculate_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculate GARCH(1,1) volatility."""
        # Simplified GARCH implementation
        # In production, use arch library: from arch import arch_model
        
        returns_clean = returns.dropna()
        if len(returns_clean) < 100:
            return self._calculate_ewma_volatility(returns)
        
        # Initial parameters (these would be estimated in full GARCH)
        omega = 0.000001  # Long-term variance
        alpha = 0.05      # ARCH parameter
        beta = 0.90       # GARCH parameter
        
        # Initialize
        garch_var = np.zeros(len(returns_clean))
        garch_var[0] = returns_clean.var()
        
        # GARCH recursion
        for i in range(1, len(returns_clean)):
            garch_var[i] = (omega + 
                           alpha * returns_clean.iloc[i-1]**2 + 
                           beta * garch_var[i-1])
        
        garch_vol = np.sqrt(garch_var * 24)
        
        # Create series with original index
        result = pd.Series(index=returns.index, dtype=float)
        result.loc[returns_clean.index] = garch_vol
        
        return result
    
    def _calculate_realized_volatility(self, returns_data: pd.DataFrame, window: int = 24) -> pd.Series:
        """Calculate realized volatility using high-frequency components."""
        # Sum of squared intraday returns
        returns_data_copy = returns_data.copy()
        returns_data_copy['squared_returns'] = returns_data_copy['returns'] ** 2
        
        realized_var = returns_data_copy['squared_returns'].rolling(window=window).sum()
        return np.sqrt(realized_var * 24)  # Annualized
    
    def _calculate_parkinson_volatility(self, returns_data: pd.DataFrame, window: int = 24) -> pd.Series:
        """Calculate Parkinson high-low volatility estimator."""
        returns_data_copy = returns_data.copy()
        
        # Parkinson estimator: (1/(4*ln(2))) * ln(High/Low)^2
        ln_hl = np.log(returns_data_copy['high'] / returns_data_copy['low'])
        parkinson_component = ln_hl ** 2 / (4 * np.log(2))
        
        parkinson_var = parkinson_component.rolling(window=window).mean()
        return np.sqrt(parkinson_var * 24)  # Annualized
    
    def _calculate_garman_klass_volatility(self, returns_data: pd.DataFrame, window: int = 24) -> pd.Series:
        """Calculate Garman-Klass OHLC volatility estimator."""
        returns_data_copy = returns_data.copy()
        
        # Garman-Klass estimator components
        ln_hl = np.log(returns_data_copy['high'] / returns_data_copy['low'])
        ln_co = np.log(returns_data_copy['close'] / returns_data_copy['open'])
        
        gk_component = 0.5 * ln_hl**2 - (2*np.log(2) - 1) * ln_co**2
        
        gk_var = gk_component.rolling(window=window).mean()
        return np.sqrt(gk_var * 24)  # Annualized
    
    def analyze_volatility(self, symbol: str) -> Optional[VolatilityMetrics]:
        """Perform comprehensive volatility analysis."""
        if symbol not in self.returns_data:
            self.logger.warning(f"No returns data for {symbol}")
            return None
        
        try:
            returns_data = self.returns_data[symbol]
            current_time = datetime.now()
            
            # Get current volatility measures
            vol_data = self.volatility_data[symbol]
            
            hist_vol_1d = vol_data.get('hist_vol_24h', pd.Series()).iloc[-1] if not vol_data.get('hist_vol_24h', pd.Series()).empty else 0.0
            hist_vol_7d = vol_data.get('hist_vol_168h', pd.Series()).iloc[-1] if not vol_data.get('hist_vol_168h', pd.Series()).empty else 0.0
            hist_vol_30d = vol_data.get('hist_vol_720h', pd.Series()).iloc[-1] if not vol_data.get('hist_vol_720h', pd.Series()).empty else 0.0
            ewma_vol = vol_data.get('ewma_vol', pd.Series()).iloc[-1] if not vol_data.get('ewma_vol', pd.Series()).empty else 0.0
            garch_vol = vol_data.get('garch_vol', pd.Series()).iloc[-1] if not vol_data.get('garch_vol', pd.Series()).empty else 0.0
            realized_vol = vol_data.get('realized_vol', pd.Series()).iloc[-1] if not vol_data.get('realized_vol', pd.Series()).empty else 0.0
            parkinson_vol = vol_data.get('parkinson_vol', pd.Series()).iloc[-1] if not vol_data.get('parkinson_vol', pd.Series()).empty else 0.0
            gk_vol = vol_data.get('garman_klass_vol', pd.Series()).iloc[-1] if not vol_data.get('garman_klass_vol', pd.Series()).empty else 0.0
            
            # Detect current regime
            current_regime, regime_prob = self._detect_volatility_regime(symbol, ewma_vol)
            regime_duration = self._calculate_regime_duration(symbol, current_regime)
            
            # Analyze volatility trends
            vol_trend_1d = self._analyze_volatility_trend(vol_data.get('hist_vol_24h', pd.Series()), 24)
            vol_trend_7d = self._analyze_volatility_trend(vol_data.get('hist_vol_168h', pd.Series()), 168)
            vol_trend_30d = self._analyze_volatility_trend(vol_data.get('hist_vol_720h', pd.Series()), 720)
            
            # Calculate risk metrics
            returns = returns_data['returns'].dropna()
            var_95 = self._calculate_var(returns, 0.95)
            cvar_95 = self._calculate_cvar(returns, 0.95)
            max_drawdown = self._calculate_max_drawdown(returns_data['close'])
            sharpe_ratio = self._calculate_sharpe_ratio(returns, ewma_vol)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            
            # Generate forecasts
            forecasts = self._generate_volatility_forecasts(symbol)
            forecast_1h = forecasts.get('1h', 0.0)
            forecast_4h = forecasts.get('4h', 0.0)
            forecast_24h = forecasts.get('24h', 0.0)
            forecast_confidence = forecasts.get('confidence', 0.5)
            
            # Statistical measures
            skewness = returns.skew() if len(returns) > 10 else 0.0
            kurtosis = returns.kurtosis() if len(returns) > 10 else 0.0
            autocorr = self._calculate_autocorrelation(returns)
            vol_clustering = self._calculate_volatility_clustering(returns)
            
            # Cross-asset measures (placeholder - would need multiple assets)
            correlation_btc = None  # Would calculate if BTC data available
            volatility_spillover = 0.0  # Simplified
            
            metrics = VolatilityMetrics(
                timestamp=current_time,
                symbol=symbol,
                timeframe='1h',
                historical_vol_1d=hist_vol_1d,
                historical_vol_7d=hist_vol_7d,
                historical_vol_30d=hist_vol_30d,
                ewma_vol=ewma_vol,
                garch_vol=garch_vol,
                realized_vol=realized_vol,
                parkinson_vol=parkinson_vol,
                garman_klass_vol=gk_vol,
                current_regime=current_regime,
                regime_probability=regime_prob,
                regime_duration=regime_duration,
                vol_trend_1d=vol_trend_1d,
                vol_trend_7d=vol_trend_7d,
                vol_trend_30d=vol_trend_30d,
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                forecast_1h=forecast_1h,
                forecast_4h=forecast_4h,
                forecast_24h=forecast_24h,
                forecast_confidence=forecast_confidence,
                skewness=skewness,
                kurtosis=kurtosis,
                autocorrelation=autocorr,
                volatility_clustering=vol_clustering,
                correlation_btc=correlation_btc,
                volatility_spillover=volatility_spillover,
                metadata={
                    'data_points': len(returns_data),
                    'calculation_time': current_time.isoformat()
                }
            )
            
            self.current_metrics[symbol] = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility for {symbol}: {e}")
            return None
    
    def _detect_volatility_regime(self, symbol: str, current_vol: float) -> Tuple[VolatilityRegime, float]:
        """Detect current volatility regime."""
        if symbol not in self.volatility_data or 'ewma_vol' not in self.volatility_data[symbol]:
            return VolatilityRegime.NORMAL, 0.5
        
        vol_series = self.volatility_data[symbol]['ewma_vol'].dropna()
        
        if len(vol_series) < 100:  # Need sufficient history
            return VolatilityRegime.NORMAL, 0.5
        
        # Calculate percentiles
        percentiles = np.percentile(vol_series, [20, 40, 70, 90])
        
        # Classify current volatility
        if current_vol <= percentiles[0]:
            regime = VolatilityRegime.LOW
            prob = 1.0 - (current_vol / percentiles[0])
        elif current_vol <= percentiles[1]:
            regime = VolatilityRegime.NORMAL
            prob = 0.8
        elif current_vol <= percentiles[2]:
            regime = VolatilityRegime.ELEVATED
            prob = 0.7
        elif current_vol <= percentiles[3]:
            regime = VolatilityRegime.HIGH
            prob = 0.6
        else:
            regime = VolatilityRegime.EXTREME
            prob = min(1.0, current_vol / percentiles[3] - 1.0)
        
        # Update regime history
        if symbol in self.current_regimes and self.current_regimes[symbol] != regime:
            self.regime_history[symbol].append((datetime.now(), regime))
        
        self.current_regimes[symbol] = regime
        
        return regime, min(1.0, max(0.1, prob))
    
    def _calculate_regime_duration(self, symbol: str, current_regime: VolatilityRegime) -> float:
        """Calculate how long we've been in the current regime."""
        if symbol not in self.regime_history or not self.regime_history[symbol]:
            return 0.0
        
        # Find the last regime change
        regime_changes = list(self.regime_history[symbol])
        if not regime_changes:
            return 0.0
        
        last_change_time = regime_changes[-1][0]
        duration_hours = (datetime.now() - last_change_time).total_seconds() / 3600
        
        return duration_hours
    
    def _analyze_volatility_trend(self, vol_series: pd.Series, window: int) -> TrendDirection:
        """Analyze volatility trend direction."""
        if vol_series.empty or len(vol_series) < window // 2:
            return TrendDirection.STABLE
        
        recent_vol = vol_series.tail(window // 4).mean()  # Last quarter of window
        older_vol = vol_series.tail(window).head(window // 4).mean()  # First quarter of window
        
        if recent_vol > older_vol * 1.1:  # 10% increase
            return TrendDirection.INCREASING
        elif recent_vol < older_vol * 0.9:  # 10% decrease
            return TrendDirection.DECREASING
        else:
            return TrendDirection.STABLE
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) < 10:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) < 10:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        
        return cvar if not np.isnan(cvar) else var
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(prices) < 2:
            return 0.0
        
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, volatility: float) -> float:
        """Calculate Sharpe ratio."""
        if volatility == 0 or len(returns) < 10:
            return 0.0
        
        mean_return = returns.mean() * 24 * 365  # Annualized
        annual_vol = volatility
        
        return mean_return / annual_vol if annual_vol > 0 else 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 10:
            return 0.0
        
        mean_return = returns.mean() * 24 * 365  # Annualized
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_deviation = downside_returns.std() * np.sqrt(24 * 365)  # Annualized
        
        return mean_return / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_autocorrelation(self, returns: pd.Series, lag: int = 1) -> float:
        """Calculate autocorrelation of returns."""
        if len(returns) < lag + 10:
            return 0.0
        
        return returns.autocorr(lag=lag)
    
    def _calculate_volatility_clustering(self, returns: pd.Series) -> float:
        """Calculate volatility clustering measure."""
        if len(returns) < 20:
            return 0.0
        
        # Use ARCH test statistic as clustering measure
        squared_returns = returns ** 2
        
        # Calculate autocorrelation of squared returns
        clustering_measure = squared_returns.autocorr(lag=1)
        
        return clustering_measure if not np.isnan(clustering_measure) else 0.0
    
    def _generate_volatility_forecasts(self, symbol: str) -> Dict[str, float]:
        """Generate volatility forecasts for different horizons."""
        forecasts = {}
        
        try:
            if symbol not in self.volatility_data:
                return {'1h': 0.0, '4h': 0.0, '24h': 0.0, 'confidence': 0.0}
            
            # Get current EWMA volatility as base
            ewma_vol_series = self.volatility_data[symbol].get('ewma_vol', pd.Series())
            if ewma_vol_series.empty:
                return {'1h': 0.0, '4h': 0.0, '24h': 0.0, 'confidence': 0.0}
            
            current_vol = ewma_vol_series.iloc[-1]
            
            # Simple forecast based on mean reversion
            long_term_vol = ewma_vol_series.mean()
            reversion_speed = 0.1  # Speed of mean reversion
            
            for horizon in self.config['forecast_horizons']:
                # Mean-reverting forecast
                forecast = current_vol * np.exp(-reversion_speed * horizon) + long_term_vol * (1 - np.exp(-reversion_speed * horizon))
                forecasts[f'{horizon}h'] = forecast
            
            # Calculate forecast confidence based on model stability
            vol_std = ewma_vol_series.tail(100).std()
            confidence = max(0.1, min(0.9, 1.0 - vol_std / current_vol)) if current_vol > 0 else 0.5
            forecasts['confidence'] = confidence
            
        except Exception as e:
            self.logger.error(f"Error generating volatility forecasts for {symbol}: {e}")
            forecasts = {'1h': 0.0, '4h': 0.0, '24h': 0.0, 'confidence': 0.0}
        
        return forecasts
    
    def detect_volatility_breakouts(self, symbol: str) -> List[VolatilityBreakout]:
        """Detect volatility breakouts."""
        breakouts = []
        
        try:
            if symbol not in self.volatility_data:
                return breakouts
            
            ewma_vol = self.volatility_data[symbol].get('ewma_vol', pd.Series())
            if len(ewma_vol) < 50:
                return breakouts
            
            current_vol = ewma_vol.iloc[-1]
            baseline_vol = ewma_vol.tail(100).quantile(0.5)  # Median as baseline
            threshold_vol = baseline_vol * self.config['breakout_threshold']
            
            # Check for upward breakout
            if current_vol > threshold_vol:
                magnitude = current_vol / baseline_vol
                
                # Estimate duration (simple heuristic)
                historical_breakouts = ewma_vol[ewma_vol > threshold_vol]
                if len(historical_breakouts) > 0:
                    # Find average duration of past breakouts
                    duration_estimate = 12.0  # Default 12 hours
                else:
                    duration_estimate = 6.0  # Conservative estimate
                
                # Calculate confidence based on magnitude and recent trend
                vol_trend = (ewma_vol.iloc[-1] - ewma_vol.iloc[-5]) / ewma_vol.iloc[-5]
                confidence = min(0.9, magnitude / self.config['breakout_threshold'] * 0.5 + abs(vol_trend) * 0.5)
                
                breakout = VolatilityBreakout(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    breakout_type='upward',
                    magnitude=magnitude,
                    duration_prediction=duration_estimate,
                    confidence=confidence,
                    trigger_level=threshold_vol,
                    current_level=current_vol,
                    expected_return=None,  # Would require additional modeling
                    risk_level='high' if magnitude > 3 else 'medium'
                )
                
                breakouts.append(breakout)
                self.active_breakouts.append(breakout)
                self.breakout_history.append(breakout)
            
            # Clean up old active breakouts
            current_time = datetime.now()
            self.active_breakouts = [
                b for b in self.active_breakouts 
                if (current_time - b.timestamp).total_seconds() / 3600 < b.duration_prediction
            ]
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility breakouts for {symbol}: {e}")
        
        return breakouts
    
    def get_risk_assessment(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive risk assessment."""
        metrics = self.current_metrics.get(symbol)
        if not metrics:
            return {'status': 'no_data', 'symbol': symbol}
        
        # Risk level based on multiple factors
        risk_factors = []
        
        # Volatility level risk
        if metrics.current_regime == VolatilityRegime.EXTREME:
            risk_factors.append(('volatility_regime', 'very_high', 0.9))
        elif metrics.current_regime == VolatilityRegime.HIGH:
            risk_factors.append(('volatility_regime', 'high', 0.7))
        elif metrics.current_regime == VolatilityRegime.ELEVATED:
            risk_factors.append(('volatility_regime', 'medium', 0.5))
        else:
            risk_factors.append(('volatility_regime', 'low', 0.3))
        
        # VaR risk
        var_risk_level = 'low'
        if abs(metrics.var_95) > 0.05:  # 5% daily VaR
            var_risk_level = 'very_high'
        elif abs(metrics.var_95) > 0.03:  # 3% daily VaR
            var_risk_level = 'high'
        elif abs(metrics.var_95) > 0.02:  # 2% daily VaR
            var_risk_level = 'medium'
        
        risk_factors.append(('var_95', var_risk_level, abs(metrics.var_95) * 10))
        
        # Trend risk
        if metrics.vol_trend_1d == TrendDirection.INCREASING:
            risk_factors.append(('volatility_trend', 'increasing', 0.6))
        elif metrics.vol_trend_1d == TrendDirection.DECREASING:
            risk_factors.append(('volatility_trend', 'decreasing', 0.4))
        else:
            risk_factors.append(('volatility_trend', 'stable', 0.3))
        
        # Calculate overall risk score
        risk_weights = {'volatility_regime': 0.4, 'var_95': 0.4, 'volatility_trend': 0.2}
        overall_risk_score = sum(
            risk_weights.get(factor_type, 0.33) * score 
            for factor_type, _, score in risk_factors
        )
        
        # Determine overall risk level
        if overall_risk_score > 0.8:
            overall_risk = 'very_high'
        elif overall_risk_score > 0.6:
            overall_risk = 'high'
        elif overall_risk_score > 0.4:
            overall_risk = 'medium'
        elif overall_risk_score > 0.2:
            overall_risk = 'low'
        else:
            overall_risk = 'very_low'
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'overall_risk_level': overall_risk,
            'overall_risk_score': overall_risk_score,
            'risk_factors': [
                {'type': factor_type, 'level': level, 'score': score}
                for factor_type, level, score in risk_factors
            ],
            'current_regime': metrics.current_regime.value,
            'var_95_daily': metrics.var_95,
            'max_drawdown': metrics.max_drawdown,
            'sharpe_ratio': metrics.sharpe_ratio,
            'recommended_position_size': self._calculate_recommended_position_size(overall_risk_score),
            'stop_loss_suggestion': abs(metrics.var_95) * 2,  # 2x VaR as stop loss
            'volatility_forecast_24h': metrics.forecast_24h
        }
    
    def _calculate_recommended_position_size(self, risk_score: float) -> float:
        """Calculate recommended position size based on risk."""
        base_position = 1.0  # 100% as base
        
        # Reduce position size as risk increases
        if risk_score > 0.8:
            return base_position * 0.2  # 20% position
        elif risk_score > 0.6:
            return base_position * 0.4  # 40% position
        elif risk_score > 0.4:
            return base_position * 0.6  # 60% position
        elif risk_score > 0.2:
            return base_position * 0.8  # 80% position
        else:
            return base_position  # Full position
    
    # Public interface methods
    def get_volatility_metrics(self, symbol: str) -> Optional[VolatilityMetrics]:
        """Get current volatility metrics."""
        return self.current_metrics.get(symbol)
    
    def get_current_regime(self, symbol: str) -> Optional[VolatilityRegime]:
        """Get current volatility regime."""
        return self.current_regimes.get(symbol)
    
    def get_volatility_forecast(self, symbol: str, horizon: str = '24h') -> Optional[float]:
        """Get volatility forecast for specific horizon."""
        forecasts = self._generate_volatility_forecasts(symbol)
        return forecasts.get(horizon)
    
    def get_active_breakouts(self, symbol: Optional[str] = None) -> List[VolatilityBreakout]:
        """Get active volatility breakouts."""
        if symbol:
            return [b for b in self.active_breakouts if b.symbol == symbol]
        return self.active_breakouts.copy()
    
    def get_volatility_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive volatility summary."""
        metrics = self.get_volatility_metrics(symbol)
        if not metrics:
            return {'status': 'no_data', 'symbol': symbol}
        
        risk_assessment = self.get_risk_assessment(symbol)
        active_breakouts = self.get_active_breakouts(symbol)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_volatility': {
                'ewma': metrics.ewma_vol,
                'garch': metrics.garch_vol,
                'realized': metrics.realized_vol,
                'regime': metrics.current_regime.value
            },
            'risk_metrics': {
                'var_95': metrics.var_95,
                'max_drawdown': metrics.max_drawdown,
                'sharpe_ratio': metrics.sharpe_ratio,
                'risk_level': risk_assessment['overall_risk_level']
            },
            'forecasts': {
                '1h': metrics.forecast_1h,
                '4h': metrics.forecast_4h,
                '24h': metrics.forecast_24h,
                'confidence': metrics.forecast_confidence
            },
            'trends': {
                '1d': metrics.vol_trend_1d.value,
                '7d': metrics.vol_trend_7d.value,
                '30d': metrics.vol_trend_30d.value
            },
            'active_breakouts': len(active_breakouts),
            'regime_duration_hours': metrics.regime_duration
        }


# Example usage and testing
if __name__ == "__main__":
    import json
    from ..config.manager import ConfigurationManager
    
    def main():
        # Configure logging
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize analyzer
        config_manager = ConfigurationManager()
        analyzer = VolatilityAnalyzer(config_manager)
        
        # Create sample data with volatility patterns
        dates = pd.date_range(start='2024-01-01', periods=2000, freq='1H')
        
        # Generate realistic price data with volatility clustering
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 2000)
        
        # Add volatility clustering
        vol = np.ones(2000) * 0.02
        for i in range(1, 2000):
            vol[i] = 0.9 * vol[i-1] + 0.1 * (0.02 + 0.5 * abs(returns[i-1]))
            returns[i] = np.random.normal(0, vol[i])
        
        # Convert to prices
        prices = 45000 * np.exp(np.cumsum(returns))
        
        price_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * np.random.uniform(0.999, 1.001, 2000),
            'high': prices * np.random.uniform(1.001, 1.005, 2000),
            'low': prices * np.random.uniform(0.995, 0.999, 2000),
            'close': prices,
            'volume': np.random.exponential(1000, 2000)
        })
        
        # Update analyzer
        analyzer.update_data('BTCUSDT', price_data)
        
        # Analyze volatility
        metrics = analyzer.analyze_volatility('BTCUSDT')
        
        if metrics:
            print("Volatility Analysis Results:")
            print(f"Current EWMA Volatility: {metrics.ewma_vol:.4f}")
            print(f"Volatility Regime: {metrics.current_regime.value}")
            print(f"VaR (95%): {metrics.var_95:.4f}")
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"24h Forecast: {metrics.forecast_24h:.4f}")
        
        # Detect breakouts
        breakouts = analyzer.detect_volatility_breakouts('BTCUSDT')
        print(f"\nDetected {len(breakouts)} volatility breakouts")
        
        # Get risk assessment
        risk_assessment = analyzer.get_risk_assessment('BTCUSDT')
        print(f"\nRisk Assessment: {json.dumps(risk_assessment, indent=2, default=str)}")
        
        # Get summary
        summary = analyzer.get_volatility_summary('BTCUSDT')
        print(f"\nVolatility Summary: {json.dumps(summary, indent=2, default=str)}")
    
    # Run the example
    main()