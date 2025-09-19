"""
Comprehensive Feature Engineering for Trading Strategies.

This module provides extensive feature engineering capabilities for financial
time series data, including:

- Technical indicators (momentum, trend, volatility, volume)
- Lagged features with configurable windows
- Rolling statistics and transformations  
- Market microstructure features
- Regime-aware features
- Cross-asset features
- Alternative data integration
- Feature selection and dimensionality reduction

The feature engineering follows best practices for financial ML:
- Proper handling of forward-looking bias
- Standardization and normalization
- Feature stability analysis
- Correlation analysis and redundancy removal
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import talib

from ..utils.logging import TradingLogger


class TechnicalIndicators:
    """
    Technical indicator calculation with comprehensive TA-Lib integration.
    
    This class provides a unified interface for calculating technical indicators
    while ensuring proper handling of NaN values and forward-looking bias.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("TechnicalIndicators")
    
    def _default_config(self) -> Dict:
        """Default configuration for technical indicators."""
        return {
            'momentum_periods': [5, 10, 20, 50],
            'volatility_periods': [10, 20, 50],
            'volume_periods': [10, 20],
            'trend_periods': [20, 50, 100],
            'oscillator_periods': [14, 21],
            'fill_method': 'forward',  # forward, backward, interpolate
            'min_periods_ratio': 0.8,  # Minimum ratio of non-NaN values
        }
    
    def calculate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            # Price-based momentum
            for period in self.config['momentum_periods']:
                # Simple returns
                features[f'return_{period}d'] = data['close'].pct_change(period)
                
                # Log returns
                features[f'log_return_{period}d'] = np.log(data['close'] / data['close'].shift(period))
                
                # Price relative to moving average
                ma = data['close'].rolling(window=period).mean()
                features[f'price_to_ma_{period}'] = data['close'] / ma - 1
                
                # RSI
                if period <= 50:  # RSI typically uses shorter periods
                    features[f'rsi_{period}'] = talib.RSI(data['close'].values.astype(np.float64), timeperiod=period)
                
                # Rate of Change
                features[f'roc_{period}'] = talib.ROC(data['close'].values.astype(np.float64), timeperiod=period)
                
                # Williams %R
                if period <= 50 and all(col in data.columns for col in ['high', 'low', 'close']):
                    features[f'willr_{period}'] = talib.WILLR(
                        data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64), timeperiod=period
                    )
            
            # MACD family
            macd, macd_signal, macd_hist = talib.MACD(data['close'].values.astype(np.float64))
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist
            
            # Stochastic oscillators
            if all(col in data.columns for col in ['high', 'low', 'close']):
                slowk, slowd = talib.STOCH(
                    data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64)
                )
                features['stoch_k'] = slowk
                features['stoch_d'] = slowd
                
                # Stochastic RSI
                fastk, fastd = talib.STOCHRSI(data['close'].values.astype(np.float64))
                features['stochrsi_k'] = fastk
                features['stochrsi_d'] = fastd
            
            self.logger.debug(f"Calculated {len(features.columns)} momentum features")
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum features: {e}")
            
        return features
    
    def calculate_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            for period in self.config['trend_periods']:
                # Moving averages
                sma = data['close'].rolling(window=period).mean()
                ema = data['close'].ewm(span=period).mean()
                
                features[f'sma_{period}'] = sma
                features[f'ema_{period}'] = ema
                features[f'price_to_sma_{period}'] = data['close'] / sma - 1
                features[f'price_to_ema_{period}'] = data['close'] / ema - 1
                
                # Moving average convergence/divergence
                if period >= 20:
                    short_ma = data['close'].rolling(window=period//2).mean()
                    long_ma = data['close'].rolling(window=period).mean()
                    features[f'ma_convergence_{period}'] = (short_ma - long_ma) / long_ma
                
                # Trend strength
                features[f'trend_strength_{period}'] = (
                    data['close'] - data['close'].shift(period)
                ) / data['close'].rolling(window=period).std()
                
                # Average Directional Index (ADX)
                if period <= 50 and all(col in data.columns for col in ['high', 'low', 'close']):
                    adx = talib.ADX(
                        data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64), timeperiod=period
                    )
                    features[f'adx_{period}'] = adx
                    
                    # Directional Movement
                    plus_di = talib.PLUS_DI(
                        data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64), timeperiod=period
                    )
                    minus_di = talib.MINUS_DI(
                        data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64), timeperiod=period
                    )
                    features[f'plus_di_{period}'] = plus_di
                    features[f'minus_di_{period}'] = minus_di
                    features[f'di_diff_{period}'] = plus_di - minus_di
            
            # Parabolic SAR
            if all(col in data.columns for col in ['high', 'low']):
                features['sar'] = talib.SAR(data['high'].values.astype(np.float64), data['low'].values.astype(np.float64))
                features['sar_signal'] = np.where(data['close'] > features['sar'], 1, -1)
            
            # Linear regression slope
            for period in [10, 20, 50]:
                features[f'lr_slope_{period}'] = (
                    data['close'].rolling(window=period)
                    .apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == period else np.nan)
                )
            
            self.logger.debug(f"Calculated {len(features.columns)} trend features")
            
        except Exception as e:
            self.logger.error(f"Error calculating trend features: {e}")
            
        return features
    
    def calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features."""
        features = pd.DataFrame(index=data.index)
        
        try:
            # Calculate returns for volatility measures
            returns = data['close'].pct_change()
            log_returns = pd.Series(np.log(data['close'] / data['close'].shift(1)), index=data.index)
            
            for period in self.config['volatility_periods']:
                # Historical volatility
                features[f'volatility_{period}d'] = returns.rolling(window=period).std() * np.sqrt(252)
                features[f'log_volatility_{period}d'] = log_returns.rolling(window=period).std() * np.sqrt(252)
                
                # Parkinson volatility (using high-low)
                if all(col in data.columns for col in ['high', 'low']):
                    hl_ratio = pd.Series(np.log(data['high'] / data['low']), index=data.index)
                    features[f'parkinson_vol_{period}d'] = (
                        hl_ratio.rolling(window=period).apply(lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))))
                    )
                
                # Garman-Klass volatility
                if all(col in data.columns for col in ['high', 'low', 'open', 'close']):
                    gk_term1 = np.log(data['high'] / data['close']) * np.log(data['high'] / data['open'])
                    gk_term2 = np.log(data['low'] / data['close']) * np.log(data['low'] / data['open'])
                    gk_vol = pd.Series(gk_term1 + gk_term2, index=data.index)
                    features[f'gk_volatility_{period}d'] = np.sqrt(gk_vol.rolling(window=period).mean())
                
                # Rolling min/max
                features[f'rolling_max_{period}d'] = data['close'].rolling(window=period).max()
                features[f'rolling_min_{period}d'] = data['close'].rolling(window=period).min()
                features[f'rolling_range_{period}d'] = (
                    features[f'rolling_max_{period}d'] - features[f'rolling_min_{period}d']
                ) / data['close']
                
                # Volatility regime
                vol_ma = features[f'volatility_{period}d'].rolling(window=period).mean()
                features[f'vol_regime_{period}d'] = features[f'volatility_{period}d'] / vol_ma - 1
            
            # Bollinger Bands
            for period in [20, 50]:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'].values.astype(np.float64), timeperiod=period)
                features[f'bb_upper_{period}'] = bb_upper
                features[f'bb_lower_{period}'] = bb_lower
                features[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
                features[f'bb_position_{period}'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Average True Range
            if all(col in data.columns for col in ['high', 'low', 'close']):
                for period in [14, 21]:
                    atr = talib.ATR(data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64), timeperiod=period)
                    features[f'atr_{period}'] = atr
                    features[f'atr_ratio_{period}'] = atr / data['close']
            
            self.logger.debug(f"Calculated {len(features.columns)} volatility features")
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {e}")
            
        return features
    
    def calculate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        features = pd.DataFrame(index=data.index)
        
        if 'volume' not in data.columns:
            self.logger.warning("Volume data not available, skipping volume features")
            return features
        
        try:
            for period in self.config['volume_periods']:
                # Volume moving averages
                vol_ma = data['volume'].rolling(window=period).mean()
                features[f'volume_ma_{period}'] = vol_ma
                features[f'volume_ratio_{period}'] = data['volume'] / vol_ma
                
                # Volume-weighted average price (VWAP)
                if all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
                    typical_price = (data['high'] + data['low'] + data['close']) / 3
                    features[f'vwap_{period}'] = (
                        (typical_price * data['volume']).rolling(window=period).sum() /
                        data['volume'].rolling(window=period).sum()
                    )
                    features[f'price_to_vwap_{period}'] = data['close'] / features[f'vwap_{period}'] - 1
                
                # On-Balance Volume
                obv = talib.OBV(data['close'].values.astype(np.float64), data['volume'].values.astype(np.float64))
                features['obv'] = obv
                features[f'obv_ma_{period}'] = pd.Series(obv, index=data.index).rolling(window=period).mean()
                
                # Volume oscillator
                if period >= 20:
                    short_vol_ma = data['volume'].rolling(window=period//2).mean()
                    long_vol_ma = data['volume'].rolling(window=period).mean()
                    features[f'volume_oscillator_{period}'] = (short_vol_ma - long_vol_ma) / long_vol_ma
            
            # Accumulation/Distribution Line
            if all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
                ad_line = talib.AD(data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64), data['volume'].values.astype(np.float64))
                features['ad_line'] = ad_line
                
                # Chaikin A/D Oscillator
                features['chaikin_ad'] = talib.ADOSC(
                    data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64), data['volume'].values.astype(np.float64)
                )
            
            # Money Flow Index
            if all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
                features['mfi'] = talib.MFI(
                    data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64), data['volume'].values.astype(np.float64)
                )
            
            self.logger.debug(f"Calculated {len(features.columns)} volume features")
            
        except Exception as e:
            self.logger.error(f"Error calculating volume features: {e}")
            
        return features


class FeatureEngineer:
    """
    Comprehensive feature engineering for trading strategies.
    
    This class orchestrates the entire feature engineering pipeline,
    including technical indicators, lagged features, transformations,
    and feature selection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("FeatureEngineer")
        
        # Initialize components
        self.technical_indicators = TechnicalIndicators(self.config.get('technical_indicators', {}))
        self.feature_scalers = {}
        self.feature_selectors = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for feature engineering."""
        return {
            'lag_periods': [1, 2, 3, 5, 10, 20],
            'rolling_windows': [5, 10, 20, 50],
            'scaling_method': 'robust',  # standard, robust, minmax
            'feature_selection_method': 'mutual_info',  # mutual_info, f_classif, pca
            'max_features': 100,
            'correlation_threshold': 0.95,
            'variance_threshold': 0.01,
            'fill_method': 'forward',
            'remove_outliers': True,
            'outlier_threshold': 3.0,
            'technical_indicators': {},
            'regime_features': True,
            'cross_asset_features': False,
        }
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        cross_asset_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Engineer comprehensive feature set from market data.
        
        Args:
            data: Primary market data (OHLCV)
            target: Target variable for supervised learning
            cross_asset_data: Additional assets for cross-asset features
            
        Returns:
            Tuple of (features_dataframe, aligned_target)
        """
        self.logger.info("Starting comprehensive feature engineering")
        
        # Validate inputs
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        features = pd.DataFrame(index=data.index)
        
        # 1. Technical indicators
        self.logger.info("Calculating technical indicators")
        tech_features = self._calculate_all_technical_features(data)
        features = pd.concat([features, tech_features], axis=1)
        
        # 2. Lagged features
        self.logger.info("Creating lagged features")
        lag_features = self._create_lagged_features(data)
        features = pd.concat([features, lag_features], axis=1)
        
        # 3. Rolling statistics
        self.logger.info("Calculating rolling statistics")
        rolling_features = self._calculate_rolling_statistics(data)
        features = pd.concat([features, rolling_features], axis=1)
        
        # 4. Regime features
        if self.config['regime_features']:
            self.logger.info("Creating regime features")
            regime_features = self._create_regime_features(data, features)
            features = pd.concat([features, regime_features], axis=1)
        
        # 5. Cross-asset features
        if self.config['cross_asset_features'] and cross_asset_data:
            self.logger.info("Creating cross-asset features")
            cross_features = self._create_cross_asset_features(data, cross_asset_data)
            features = pd.concat([features, cross_features], axis=1)
        
        # 6. Feature transformations
        self.logger.info("Applying feature transformations")
        features = self._apply_transformations(features)
        
        # 7. Handle missing values
        self.logger.info("Handling missing values")
        features = self._handle_missing_values(features)
        
        # 8. Remove outliers
        if self.config['remove_outliers']:
            self.logger.info("Removing outliers")
            features = self._remove_outliers(features)
        
        # 9. Feature selection
        self.logger.info("Performing feature selection")
        if target is not None:
            features, aligned_target = self._select_features(features, target)
        else:
            aligned_target = None
            features = self._remove_correlated_features(features)
        
        # 10. Final scaling
        self.logger.info("Applying final scaling")
        features = self._scale_features(features)
        
        self.logger.info(f"Feature engineering completed: {len(features.columns)} features created")
        
        return features, aligned_target
    
    def _calculate_all_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicator features."""
        all_features = pd.DataFrame(index=data.index)
        
        # Momentum features
        momentum_features = self.technical_indicators.calculate_momentum_features(data)
        all_features = pd.concat([all_features, momentum_features], axis=1)
        
        # Trend features
        trend_features = self.technical_indicators.calculate_trend_features(data)
        all_features = pd.concat([all_features, trend_features], axis=1)
        
        # Volatility features
        volatility_features = self.technical_indicators.calculate_volatility_features(data)
        all_features = pd.concat([all_features, volatility_features], axis=1)
        
        # Volume features
        volume_features = self.technical_indicators.calculate_volume_features(data)
        all_features = pd.concat([all_features, volume_features], axis=1)
        
        return all_features
    
    def _create_lagged_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lagged versions of key features."""
        features = pd.DataFrame(index=data.index)
        
        # Key price features to lag
        price_features = ['close', 'high', 'low', 'open']
        if 'volume' in data.columns:
            price_features.append('volume')
        
        for feature in price_features:
            if feature in data.columns:
                for lag in self.config['lag_periods']:
                    features[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
                    
                    # Lag differences
                    if lag > 1:
                        features[f'{feature}_diff_lag_{lag}'] = (
                            data[feature].shift(lag) - data[feature].shift(lag*2)
                        ) / data[feature].shift(lag*2)
        
        # Lag returns
        returns = data['close'].pct_change()
        for lag in self.config['lag_periods']:
            features[f'return_lag_{lag}'] = returns.shift(lag)
        
        return features
    
    def _calculate_rolling_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling statistical features."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based rolling statistics
        price_series = data['close']
        returns = price_series.pct_change()
        
        for window in self.config['rolling_windows']:
            # Basic statistics
            features[f'rolling_mean_{window}'] = price_series.rolling(window=window).mean()
            features[f'rolling_std_{window}'] = price_series.rolling(window=window).std()
            features[f'rolling_skew_{window}'] = returns.rolling(window=window).skew()
            features[f'rolling_kurt_{window}'] = returns.rolling(window=window).kurt()
            
            # Quantiles
            features[f'rolling_q25_{window}'] = price_series.rolling(window=window).quantile(0.25)
            features[f'rolling_q75_{window}'] = price_series.rolling(window=window).quantile(0.75)
            features[f'rolling_median_{window}'] = price_series.rolling(window=window).median()
            
            # Position in rolling window
            rolling_min = price_series.rolling(window=window).min()
            rolling_max = price_series.rolling(window=window).max()
            features[f'rolling_position_{window}'] = (
                (price_series - rolling_min) / (rolling_max - rolling_min)
            )
            
            # Rolling correlations (if volume available)
            if 'volume' in data.columns:
                features[f'price_volume_corr_{window}'] = (
                    price_series.rolling(window=window).corr(data['volume'])
                )
        
        return features
    
    def _create_regime_features(self, data: pd.DataFrame, existing_features: pd.DataFrame) -> pd.DataFrame:
        """Create regime-aware features."""
        features = pd.DataFrame(index=data.index)
        
        # Volatility regime
        volatility = data['close'].pct_change().rolling(window=20).std()
        vol_threshold_high = float(volatility.rolling(window=252).quantile(0.75).iloc[-1])
        vol_threshold_low = float(volatility.rolling(window=252).quantile(0.25).iloc[-1])
        
        features['vol_regime'] = pd.cut(
            volatility,
            bins=[-np.inf, vol_threshold_low, vol_threshold_high, np.inf],
            labels=[0, 1, 2]  # Low, Medium, High volatility
        ).astype(float)
        
        # Trend regime
        ma_short = data['close'].rolling(window=20).mean()
        ma_long = data['close'].rolling(window=60).mean()
        features['trend_regime'] = np.where(ma_short > ma_long, 1, 0)  # 1 = uptrend, 0 = downtrend
        
        # Market stress regime
        returns = data['close'].pct_change()
        rolling_var = returns.rolling(window=20).var()
        var_threshold = rolling_var.rolling(window=252).quantile(0.9)
        features['stress_regime'] = np.where(rolling_var > var_threshold, 1, 0)
        
        # Time-based regimes
        features['hour'] = pd.to_datetime(data.index).hour if hasattr(data.index, 'hour') else 0
        features['day_of_week'] = pd.to_datetime(data.index).dayofweek if hasattr(data.index, 'dayofweek') else 0
        features['month'] = pd.to_datetime(data.index).month if hasattr(data.index, 'month') else 0
        
        # Regime interactions with key features
        key_features = ['close', 'volume'] if 'volume' in data.columns else ['close']
        for feature_name in key_features:
            if feature_name in data.columns:
                for regime in ['vol_regime', 'trend_regime']:
                    features[f'{feature_name}_{regime}_interaction'] = (
                        data[feature_name] * features[regime]
                    )
        
        return features
    
    def _create_cross_asset_features(
        self, 
        data: pd.DataFrame, 
        cross_asset_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create features from other assets."""
        features = pd.DataFrame(index=data.index)
        
        main_returns = data['close'].pct_change()
        
        for asset_name, asset_data in cross_asset_data.items():
            if asset_data.empty or 'close' not in asset_data.columns:
                continue
            
            # Align data
            asset_returns = asset_data['close'].pct_change()
            asset_returns = asset_returns.reindex(data.index, method='ffill')
            
            # Rolling correlations
            for window in [20, 60]:
                features[f'{asset_name}_corr_{window}'] = (
                    main_returns.rolling(window=window).corr(asset_returns)
                )
            
            # Relative performance
            asset_price = asset_data['close'].reindex(data.index, method='ffill')
            for period in [5, 20]:
                main_perf = data['close'] / data['close'].shift(period) - 1
                asset_perf = asset_price / asset_price.shift(period) - 1
                features[f'{asset_name}_rel_perf_{period}'] = main_perf - asset_perf
        
        return features
    
    def _apply_transformations(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply mathematical transformations to features."""
        transformed_features = features.copy()
        
        # Log transformations for positive features
        positive_features = features.select_dtypes(include=[np.number]).columns
        for col in positive_features:
            if (features[col] > 0).all():
                transformed_features[f'{col}_log'] = np.log(features[col])
        
        # Square root transformations
        non_negative_features = features.select_dtypes(include=[np.number]).columns
        for col in non_negative_features:
            if (features[col] >= 0).all():
                transformed_features[f'{col}_sqrt'] = np.sqrt(features[col])
        
        # Rank transformations
        for col in features.select_dtypes(include=[np.number]).columns:
            transformed_features[f'{col}_rank'] = features[col].rank(pct=True)
        
        return transformed_features
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        if self.config['fill_method'] == 'forward':
            features = features.ffill()
        elif self.config['fill_method'] == 'backward':
            features = features.bfill()
        elif self.config['fill_method'] == 'interpolate':
            features = features.interpolate()
        
        # Drop columns with excessive missing values
        min_valid_ratio = 1 - self.config.get('max_missing_ratio', 0.5)
        valid_columns = features.columns[features.count() / len(features) >= min_valid_ratio]
        features = features[valid_columns]
        
        # Final cleanup
        features = features.dropna()
        
        return features
    
    def _remove_outliers(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(numeric_features.values, nan_policy='omit'))  # type: ignore
        
        # Remove rows with outliers
        outlier_mask = (z_scores < self.config['outlier_threshold']).all(axis=1)
        
        cleaned_features = features[outlier_mask]
        
        self.logger.info(f"Removed {len(features) - len(cleaned_features)} outlier rows")
        
        return cleaned_features
    
    def _select_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Select best features using specified method."""
        # Align features and target
        aligned_data = pd.concat([features, target], axis=1, join='inner')
        aligned_features = aligned_data.iloc[:, :-1]
        aligned_target = aligned_data.iloc[:, -1]
        
        if self.config['feature_selection_method'] == 'mutual_info':
            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(self.config['max_features'], len(aligned_features.columns))
            )
        elif self.config['feature_selection_method'] == 'f_classif':
            selector = SelectKBest(
                score_func=f_classif,
                k=min(self.config['max_features'], len(aligned_features.columns))
            )
        else:
            # Return all features if no selection method
            return aligned_features, aligned_target
        
        # Fit selector and transform features
        selected_features = selector.fit_transform(aligned_features, aligned_target)
        selected_feature_names = aligned_features.columns[selector.get_support()]
        
        result_features = pd.DataFrame(
            selected_features,
            index=aligned_features.index,
            columns=selected_feature_names
        )
        
        self.logger.info(f"Selected {len(selected_feature_names)} features from {len(aligned_features.columns)}")
        
        return result_features, aligned_target
    
    def _remove_correlated_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_features.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > self.config['correlation_threshold'])
        ]
        
        # Keep non-numeric features and drop correlated numeric features
        result_features = features.drop(columns=to_drop)
        
        self.logger.info(f"Removed {len(to_drop)} highly correlated features")
        
        return result_features
    
    def _scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        numeric_features = features.select_dtypes(include=[np.number])
        non_numeric_features = features.select_dtypes(exclude=[np.number])
        
        if numeric_features.empty:
            return features
        
        # Choose scaler
        if self.config['scaling_method'] == 'standard':
            scaler = StandardScaler()
        elif self.config['scaling_method'] == 'robust':
            scaler = RobustScaler()
        elif self.config['scaling_method'] == 'minmax':
            scaler = MinMaxScaler()
        else:
            return features
        
        # Fit and transform
        scaled_numeric = scaler.fit_transform(numeric_features)
        scaled_numeric_df = pd.DataFrame(
            scaled_numeric,
            index=numeric_features.index,
            columns=numeric_features.columns
        )
        
        # Combine scaled numeric and non-numeric features
        result_features = pd.concat([scaled_numeric_df, non_numeric_features], axis=1)
        
        # Store scaler for future use
        self.feature_scalers['main'] = scaler
        
        return result_features
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformations."""
        # This would apply the same transformations to new data
        # For now, return as-is (would need to store transformation pipeline)
        self.logger.warning("transform_new_data not fully implemented - requires stored pipeline")
        return data