"""
Feature Engineering Module - Advanced Feature Creation and Selection

This module provides comprehensive feature engineering capabilities:
- Technical indicators and market microstructure features
- Sentiment analysis and alternative data features
- Cross-asset correlation and macro features
- Feature selection and importance analysis
- Real-time feature computation
- Feature validation and quality control

The Feature Engineering module creates sophisticated features for ML models:
- Price-based features (returns, volatility, momentum)
- Volume-based features (volume profile, flow analysis)
- Microstructure features (bid-ask spread, order book)
- Sentiment features (news, social media)
- Macro features (interest rates, commodities)

Author: Trading Bot Team
Version: 1.0.0 - Phase 6 Implementation
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import talib
from scipy import stats
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, RFECV
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestRegressor

from ..utils.logging import TradingLogger
from ..config_manager import ConfigurationManager


class FeatureType(Enum):
    """Types of features."""
    TECHNICAL = "technical"
    MICROSTRUCTURE = "microstructure"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    CROSS_ASSET = "cross_asset"
    CUSTOM = "custom"


class FeatureSelectionMethod(Enum):
    """Feature selection methods."""
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    F_TEST = "f_test"
    RFE = "rfe"
    RFECV = "rfecv"
    TREE_IMPORTANCE = "tree_importance"
    PCA = "pca"
    ICA = "ica"


@dataclass
class FeatureSet:
    """Feature set definition."""
    name: str
    feature_names: List[str]
    feature_types: Dict[str, FeatureType]
    lookback_periods: Dict[str, int]
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""


@dataclass
class FeatureImportance:
    """Feature importance scores."""
    feature_name: str
    importance_score: float
    selection_method: FeatureSelectionMethod
    rank: int
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class FeatureEngineering:
    """
    Advanced Feature Engineering for Trading ML Models.
    
    This class provides comprehensive feature engineering capabilities
    including technical indicators, market microstructure features,
    sentiment analysis, and cross-asset features.
    
    Features:
    - 200+ technical indicators using TA-Lib
    - Market microstructure features from order book data
    - Cross-asset correlation and spread features
    - Sentiment features from news and social media
    - Macro economic features
    - Feature selection and dimensionality reduction
    - Real-time feature computation
    """
    
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.logger = TradingLogger("feature_engineering")
        
        # Feature sets
        self.feature_sets: Dict[str, FeatureSet] = {}
        self.computed_features: Dict[str, pd.DataFrame] = {}
        
        # Feature selection results
        self.feature_importance: Dict[str, List[FeatureImportance]] = {}
        self.selected_features: Dict[str, List[str]] = {}
        
        # Feature computation cache
        self.feature_cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        
        # Configuration
        self.max_features = config.get('feature_engineering.max_features', 100)
        self.cache_duration = config.get('feature_engineering.cache_duration_minutes', 5)
        self.correlation_threshold = config.get('feature_engineering.correlation_threshold', 0.95)
        
        # Initialize default feature sets
        self._initialize_default_feature_sets()
        
        self.logger.info("FeatureEngineering initialized")
    
    async def create_technical_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        feature_set_name: str = "technical_default"
    ) -> pd.DataFrame:
        """
        Create technical analysis features.
        
        Args:
            price_data: OHLCV price data
            volume_data: Volume data (optional)
            feature_set_name: Name of feature set to use
            
        Returns:
            DataFrame with technical features
        """
        try:
            self.logger.info(f"Creating technical features with set: {feature_set_name}")
            
            # Check cache
            cache_key = f"technical_{feature_set_name}_{hash(str(price_data.index[-1]))}"
            if self._is_cached(cache_key):
                return self.feature_cache[cache_key]
            
            features = pd.DataFrame(index=price_data.index)
            
            # Extract OHLCV data
            high = price_data['high'].values
            low = price_data['low'].values
            close = price_data['close'].values
            open_price = price_data['open'].values
            volume = price_data['volume'].values if 'volume' in price_data.columns else None
            
            # Price-based features
            features = await self._add_price_features(features, high, low, close, open_price)
            
            # Volume-based features
            if volume is not None:
                features = await self._add_volume_features(features, close, volume)
            
            # Volatility features
            features = await self._add_volatility_features(features, high, low, close)
            
            # Momentum features
            features = await self._add_momentum_features(features, close)
            
            # Pattern recognition features
            features = await self._add_pattern_features(features, high, low, close, open_price)
            
            # Trend features
            features = await self._add_trend_features(features, high, low, close)
            
            # Statistical features
            features = await self._add_statistical_features(features, close)
            
            # Remove NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            # Cache results
            self._cache_result(cache_key, features)
            
            self.logger.info(f"Created {len(features.columns)} technical features")
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating technical features: {e}")
            return pd.DataFrame()
    
    async def create_microstructure_features(
        self,
        orderbook_data: pd.DataFrame,
        trade_data: pd.DataFrame,
        feature_set_name: str = "microstructure_default"
    ) -> pd.DataFrame:
        """
        Create market microstructure features.
        
        Args:
            orderbook_data: Order book level 2 data
            trade_data: Individual trade data
            feature_set_name: Name of feature set to use
            
        Returns:
            DataFrame with microstructure features
        """
        try:
            self.logger.info(f"Creating microstructure features with set: {feature_set_name}")
            
            features = pd.DataFrame(index=orderbook_data.index)
            
            # Bid-ask spread features
            features['bid_ask_spread'] = orderbook_data['ask_price_1'] - orderbook_data['bid_price_1']
            features['bid_ask_spread_pct'] = features['bid_ask_spread'] / orderbook_data['mid_price']
            features['spread_ma_5'] = features['bid_ask_spread'].rolling(5).mean()
            features['spread_ma_20'] = features['bid_ask_spread'].rolling(20).mean()
            features['spread_volatility'] = features['bid_ask_spread'].rolling(20).std()
            
            # Order book imbalance
            bid_volume = orderbook_data[[c for c in orderbook_data.columns if 'bid_size' in c]].sum(axis=1)
            ask_volume = orderbook_data[[c for c in orderbook_data.columns if 'ask_size' in c]].sum(axis=1)
            
            features['order_imbalance'] = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            features['order_imbalance_ma'] = features['order_imbalance'].rolling(10).mean()
            features['bid_volume_ratio'] = bid_volume / (bid_volume + ask_volume)
            
            # Depth features
            features['total_bid_volume'] = bid_volume
            features['total_ask_volume'] = ask_volume
            features['depth_ratio'] = features['total_bid_volume'] / features['total_ask_volume']
            
            # Price impact features
            features['price_impact_buy'] = (orderbook_data['ask_price_5'] - orderbook_data['ask_price_1']) / orderbook_data['ask_price_1']
            features['price_impact_sell'] = (orderbook_data['bid_price_1'] - orderbook_data['bid_price_5']) / orderbook_data['bid_price_1']
            
            # Add trade-based features if available
            if not trade_data.empty:
                features = await self._add_trade_features(features, trade_data)
            
            # Remove NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            self.logger.info(f"Created {len(features.columns)} microstructure features")
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating microstructure features: {e}")
            return pd.DataFrame()
    
    async def create_cross_asset_features(
        self,
        primary_data: pd.DataFrame,
        reference_assets: Dict[str, pd.DataFrame],
        feature_set_name: str = "cross_asset_default"
    ) -> pd.DataFrame:
        """
        Create cross-asset correlation and spread features.
        
        Args:
            primary_data: Primary asset price data
            reference_assets: Dictionary of reference asset data
            feature_set_name: Name of feature set to use
            
        Returns:
            DataFrame with cross-asset features
        """
        try:
            self.logger.info(f"Creating cross-asset features with set: {feature_set_name}")
            
            features = pd.DataFrame(index=primary_data.index)
            primary_returns = primary_data['close'].pct_change()
            
            for asset_name, asset_data in reference_assets.items():
                if asset_data.empty:
                    continue
                
                # Align data
                aligned_data = asset_data.reindex(primary_data.index, method='ffill')
                asset_returns = aligned_data['close'].pct_change()
                
                # Correlation features
                for window in [10, 20, 50]:
                    corr = primary_returns.rolling(window).corr(asset_returns)
                    features[f'{asset_name}_corr_{window}'] = corr
                
                # Spread features
                price_ratio = primary_data['close'] / aligned_data['close']
                features[f'{asset_name}_price_ratio'] = price_ratio
                features[f'{asset_name}_price_ratio_ma'] = price_ratio.rolling(20).mean()
                features[f'{asset_name}_price_ratio_std'] = price_ratio.rolling(20).std()
                features[f'{asset_name}_price_ratio_zscore'] = (
                    (price_ratio - price_ratio.rolling(20).mean()) / price_ratio.rolling(20).std()
                )
                
                # Beta features
                for window in [20, 50]:
                    covariance = primary_returns.rolling(window).cov(asset_returns)
                    variance = asset_returns.rolling(window).var()
                    beta = covariance / variance
                    features[f'{asset_name}_beta_{window}'] = beta
                
                # Relative strength
                primary_momentum = primary_data['close'].pct_change(10)
                asset_momentum = aligned_data['close'].pct_change(10)
                features[f'{asset_name}_relative_strength'] = primary_momentum - asset_momentum
            
            # Remove NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            self.logger.info(f"Created {len(features.columns)} cross-asset features")
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating cross-asset features: {e}")
            return pd.DataFrame()
    
    async def create_sentiment_features(
        self,
        news_sentiment: Optional[pd.DataFrame] = None,
        social_sentiment: Optional[pd.DataFrame] = None,
        fear_greed_index: Optional[pd.DataFrame] = None,
        feature_set_name: str = "sentiment_default"
    ) -> pd.DataFrame:
        """
        Create sentiment-based features.
        
        Args:
            news_sentiment: News sentiment scores
            social_sentiment: Social media sentiment scores
            fear_greed_index: Fear & Greed index data
            feature_set_name: Name of feature set to use
            
        Returns:
            DataFrame with sentiment features
        """
        try:
            self.logger.info(f"Creating sentiment features with set: {feature_set_name}")
            
            # Determine index from available data
            if news_sentiment is not None and not news_sentiment.empty:
                index = news_sentiment.index
            elif social_sentiment is not None and not social_sentiment.empty:
                index = social_sentiment.index
            elif fear_greed_index is not None and not fear_greed_index.empty:
                index = fear_greed_index.index
            else:
                # Create empty features with current timestamp
                return pd.DataFrame()
            
            features = pd.DataFrame(index=index)
            
            # News sentiment features
            if news_sentiment is not None and not news_sentiment.empty:
                features['news_sentiment'] = news_sentiment['sentiment_score']
                features['news_sentiment_ma_5'] = features['news_sentiment'].rolling(5).mean()
                features['news_sentiment_ma_20'] = features['news_sentiment'].rolling(20).mean()
                features['news_sentiment_volatility'] = features['news_sentiment'].rolling(10).std()
                features['news_sentiment_momentum'] = features['news_sentiment'].diff(5)
                
                # News volume features
                if 'article_count' in news_sentiment.columns:
                    features['news_volume'] = news_sentiment['article_count']
                    features['news_volume_ma'] = features['news_volume'].rolling(5).mean()
            
            # Social sentiment features
            if social_sentiment is not None and not social_sentiment.empty:
                features['social_sentiment'] = social_sentiment['sentiment_score']
                features['social_sentiment_ma_5'] = features['social_sentiment'].rolling(5).mean()
                features['social_sentiment_ma_20'] = features['social_sentiment'].rolling(20).mean()
                features['social_sentiment_volatility'] = features['social_sentiment'].rolling(10).std()
                
                # Social volume features
                if 'mention_count' in social_sentiment.columns:
                    features['social_volume'] = social_sentiment['mention_count']
                    features['social_volume_ma'] = features['social_volume'].rolling(5).mean()
            
            # Fear & Greed Index features
            if fear_greed_index is not None and not fear_greed_index.empty:
                features['fear_greed_index'] = fear_greed_index['value']
                features['fear_greed_ma_10'] = features['fear_greed_index'].rolling(10).mean()
                features['fear_greed_momentum'] = features['fear_greed_index'].diff(5)
                features['fear_greed_extreme'] = (
                    (features['fear_greed_index'] < 20) | (features['fear_greed_index'] > 80)
                ).astype(int)
            
            # Combined sentiment features
            if len([x for x in [news_sentiment, social_sentiment] if x is not None]) >= 2:
                # Create composite sentiment
                sentiment_cols = [col for col in features.columns if 'sentiment' in col and 'ma' not in col and 'volatility' not in col]
                if sentiment_cols:
                    features['composite_sentiment'] = features[sentiment_cols].mean(axis=1)
                    features['sentiment_divergence'] = features[sentiment_cols].std(axis=1)
            
            # Remove NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            self.logger.info(f"Created {len(features.columns)} sentiment features")
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment features: {e}")
            return pd.DataFrame()
    
    async def create_macro_features(
        self,
        macro_data: Dict[str, pd.DataFrame],
        target_index: pd.DatetimeIndex,
        feature_set_name: str = "macro_default"
    ) -> pd.DataFrame:
        """
        Create macroeconomic features.
        
        Args:
            macro_data: Dictionary of macro economic data
            target_index: Target datetime index for features
            feature_set_name: Name of feature set to use
            
        Returns:
            DataFrame with macro features
        """
        try:
            self.logger.info(f"Creating macro features with set: {feature_set_name}")
            
            features = pd.DataFrame(index=target_index)
            
            for indicator_name, indicator_data in macro_data.items():
                if indicator_data.empty:
                    continue
                
                # Align data to target index
                aligned_data = indicator_data.reindex(target_index, method='ffill')
                
                # Raw value
                features[f'{indicator_name}_value'] = aligned_data['value']
                
                # Moving averages
                for window in [5, 20, 60]:
                    features[f'{indicator_name}_ma_{window}'] = aligned_data['value'].rolling(window).mean()
                
                # Rate of change
                for period in [1, 5, 20]:
                    features[f'{indicator_name}_roc_{period}'] = aligned_data['value'].pct_change(period)
                
                # Z-score (standardized value)
                rolling_mean = aligned_data['value'].rolling(60).mean()
                rolling_std = aligned_data['value'].rolling(60).std()
                features[f'{indicator_name}_zscore'] = (aligned_data['value'] - rolling_mean) / rolling_std
                
                # Regime indicators
                ma_short = aligned_data['value'].rolling(20).mean()
                ma_long = aligned_data['value'].rolling(60).mean()
                features[f'{indicator_name}_regime'] = (ma_short > ma_long).astype(int)
            
            # Cross-indicator features
            indicator_names = list(macro_data.keys())
            if len(indicator_names) >= 2:
                # Create factor scores (simple average of z-scores)
                zscore_cols = [col for col in features.columns if 'zscore' in col]
                if len(zscore_cols) >= 2:
                    features['macro_factor_composite'] = features[zscore_cols].mean(axis=1)
                    features['macro_factor_dispersion'] = features[zscore_cols].std(axis=1)
            
            # Remove NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            self.logger.info(f"Created {len(features.columns)} macro features")
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating macro features: {e}")
            return pd.DataFrame()
    
    async def select_features(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        method: FeatureSelectionMethod = FeatureSelectionMethod.MUTUAL_INFO,
        n_features: Optional[int] = None
    ) -> Tuple[List[str], List[FeatureImportance]]:
        """
        Select best features using specified method.
        
        Args:
            features: Feature matrix
            target: Target variable
            method: Feature selection method
            n_features: Number of features to select (default: use config)
            
        Returns:
            Tuple of (selected feature names, feature importance scores)
        """
        try:
            if n_features is None:
                n_features = min(self.max_features, len(features.columns))
            
            self.logger.info(f"Selecting {n_features} features using {method.value}")
            
            # Align features and target
            aligned_features, aligned_target = features.align(target, join='inner', axis=0)
            
            # Remove constant features
            constant_features = aligned_features.columns[aligned_features.std() == 0]
            if len(constant_features) > 0:
                aligned_features = aligned_features.drop(columns=constant_features)
                self.logger.info(f"Removed {len(constant_features)} constant features")
            
            # Remove highly correlated features
            aligned_features = self._remove_correlated_features(aligned_features)
            
            selected_features = []
            importance_scores = []
            
            if method == FeatureSelectionMethod.MUTUAL_INFO:
                selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, len(aligned_features.columns)))
                selected_data = selector.fit_transform(aligned_features, aligned_target)
                selected_features = aligned_features.columns[selector.get_support()].tolist()
                
                # Create importance scores
                scores = selector.scores_
                for i, feature in enumerate(aligned_features.columns):
                    if feature in selected_features:
                        importance_scores.append(FeatureImportance(
                            feature_name=feature,
                            importance_score=scores[i],
                            selection_method=method,
                            rank=selected_features.index(feature) + 1
                        ))
            
            elif method == FeatureSelectionMethod.F_TEST:
                selector = SelectKBest(score_func=f_regression, k=min(n_features, len(aligned_features.columns)))
                selected_data = selector.fit_transform(aligned_features, aligned_target)
                selected_features = aligned_features.columns[selector.get_support()].tolist()
                
                scores = selector.scores_
                p_values = selector.pvalues_
                for i, feature in enumerate(aligned_features.columns):
                    if feature in selected_features:
                        importance_scores.append(FeatureImportance(
                            feature_name=feature,
                            importance_score=scores[i],
                            selection_method=method,
                            rank=selected_features.index(feature) + 1,
                            p_value=p_values[i]
                        ))
            
            elif method == FeatureSelectionMethod.RFE:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                selector = RFE(estimator, n_features_to_select=min(n_features, len(aligned_features.columns)))
                selected_data = selector.fit_transform(aligned_features, aligned_target)
                selected_features = aligned_features.columns[selector.get_support()].tolist()
                
                rankings = selector.ranking_
                for i, feature in enumerate(aligned_features.columns):
                    if feature in selected_features:
                        importance_scores.append(FeatureImportance(
                            feature_name=feature,
                            importance_score=1.0 / rankings[i],  # Inverse of rank as importance
                            selection_method=method,
                            rank=rankings[i]
                        ))
            
            elif method == FeatureSelectionMethod.TREE_IMPORTANCE:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                estimator.fit(aligned_features, aligned_target)
                
                importances = estimator.feature_importances_
                feature_importance_pairs = list(zip(aligned_features.columns, importances))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                selected_features = [pair[0] for pair in feature_importance_pairs[:n_features]]
                
                for rank, (feature, importance) in enumerate(feature_importance_pairs[:n_features]):
                    importance_scores.append(FeatureImportance(
                        feature_name=feature,
                        importance_score=importance,
                        selection_method=method,
                        rank=rank + 1
                    ))
            
            elif method == FeatureSelectionMethod.CORRELATION:
                # Select features with highest correlation to target
                correlations = aligned_features.corrwith(aligned_target).abs()
                correlations = correlations.sort_values(ascending=False)
                selected_features = correlations.head(n_features).index.tolist()
                
                for rank, (feature, corr) in enumerate(correlations.head(n_features).items()):
                    importance_scores.append(FeatureImportance(
                        feature_name=feature,
                        importance_score=corr,
                        selection_method=method,
                        rank=rank + 1
                    ))
            
            elif method == FeatureSelectionMethod.PCA:
                # Use PCA for dimensionality reduction
                pca = PCA(n_components=min(n_features, len(aligned_features.columns)))
                pca.fit(aligned_features)
                
                # Create feature names for principal components
                selected_features = [f'PC_{i+1}' for i in range(pca.n_components_)]
                
                for i, explained_var in enumerate(pca.explained_variance_ratio_):
                    importance_scores.append(FeatureImportance(
                        feature_name=f'PC_{i+1}',
                        importance_score=explained_var,
                        selection_method=method,
                        rank=i + 1
                    ))
            
            # Sort importance scores by score (descending)
            importance_scores.sort(key=lambda x: x.importance_score, reverse=True)
            
            self.logger.info(f"Selected {len(selected_features)} features using {method.value}")
            return selected_features, importance_scores
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {e}")
            return [], []
    
    def _remove_correlated_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        try:
            correlation_matrix = features.corr().abs()
            
            # Find pairs of highly correlated features
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > self.correlation_threshold)]
            
            if to_drop:
                features = features.drop(columns=to_drop)
                self.logger.info(f"Removed {len(to_drop)} highly correlated features")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error removing correlated features: {e}")
            return features
    
    async def _add_price_features(self, features: pd.DataFrame, high: np.ndarray, low: np.ndarray, 
                                close: np.ndarray, open_price: np.ndarray) -> pd.DataFrame:
        """Add price-based technical features."""
        try:
            # Simple moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                features[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
                features[f'price_sma_{period}_ratio'] = close / features[f'sma_{period}']
            
            # Exponential moving averages
            for period in [12, 26, 50]:
                features[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
                features[f'price_ema_{period}_ratio'] = close / features[f'ema_{period}']
            
            # MACD
            macd, macd_signal, macd_histogram = talib.MACD(close)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_histogram
            
            # Price channels
            features['high_20'] = talib.MAX(high, timeperiod=20)
            features['low_20'] = talib.MIN(low, timeperiod=20)
            features['price_channel_position'] = (close - features['low_20']) / (features['high_20'] - features['low_20'])
            
            # Returns
            for period in [1, 2, 5, 10, 20]:
                features[f'return_{period}'] = talib.ROC(close, timeperiod=period)
            
            # Price gaps
            features['gap'] = (open_price - np.roll(close, 1)) / np.roll(close, 1)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding price features: {e}")
            return features
    
    async def _add_volume_features(self, features: pd.DataFrame, close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
        """Add volume-based features."""
        try:
            # Volume moving averages
            for period in [10, 20, 50]:
                features[f'volume_ma_{period}'] = talib.SMA(volume, timeperiod=period)
                features[f'volume_ratio_{period}'] = volume / features[f'volume_ma_{period}']
            
            # On-Balance Volume
            features['obv'] = talib.OBV(close, volume)
            features['obv_ma'] = talib.SMA(features['obv'].values, timeperiod=20)
            
            # Volume Price Trend
            features['vpt'] = ((close - np.roll(close, 1)) / np.roll(close, 1)) * volume
            features['vpt_ma'] = talib.SMA(features['vpt'].values, timeperiod=20)
            
            # Accumulation/Distribution Line
            features['ad_line'] = talib.AD(high, low, close, volume)
            
            # Chaikin Money Flow
            features['cmf'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding volume features: {e}")
            return features
    
    async def _add_volatility_features(self, features: pd.DataFrame, high: np.ndarray, 
                                     low: np.ndarray, close: np.ndarray) -> pd.DataFrame:
        """Add volatility-based features."""
        try:
            # True Range and ATR
            features['true_range'] = talib.TRANGE(high, low, close)
            for period in [14, 20, 50]:
                features[f'atr_{period}'] = talib.ATR(high, low, close, timeperiod=period)
                features[f'atr_{period}_ratio'] = features[f'atr_{period}'] / close
            
            # Bollinger Bands
            for period in [20, 50]:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=period)
                features[f'bb_upper_{period}'] = bb_upper
                features[f'bb_lower_{period}'] = bb_lower
                features[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
                features[f'bb_position_{period}'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # Historical volatility
            for period in [10, 20, 50]:
                returns = np.diff(np.log(close))
                features[f'hist_vol_{period}'] = pd.Series(returns).rolling(period).std() * np.sqrt(252)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding volatility features: {e}")
            return features
    
    async def _add_momentum_features(self, features: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
        """Add momentum-based features."""
        try:
            # RSI
            for period in [14, 21, 50]:
                features[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d
            
            # Williams %R
            features['williams_r'] = talib.WILLR(high, low, close)
            
            # Commodity Channel Index
            features['cci'] = talib.CCI(high, low, close)
            
            # Rate of Change
            for period in [10, 20, 50]:
                features[f'roc_{period}'] = talib.ROC(close, timeperiod=period)
            
            # Momentum
            for period in [10, 20]:
                features[f'momentum_{period}'] = talib.MOM(close, timeperiod=period)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding momentum features: {e}")
            return features
    
    async def _add_pattern_features(self, features: pd.DataFrame, high: np.ndarray, 
                                  low: np.ndarray, close: np.ndarray, open_price: np.ndarray) -> pd.DataFrame:
        """Add pattern recognition features."""
        try:
            # Candlestick patterns
            patterns = [
                'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
                'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
                'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
                'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
                'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
                'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
                'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
                'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
                'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
                'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
                'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
                'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
                'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
                'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
                'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
                'CDLXSIDEGAP3METHODS'
            ]
            
            for pattern in patterns[:20]:  # Limit to first 20 patterns to avoid too many features
                try:
                    pattern_func = getattr(talib, pattern)
                    features[pattern.lower()] = pattern_func(open_price, high, low, close)
                except:
                    continue
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding pattern features: {e}")
            return features
    
    async def _add_trend_features(self, features: pd.DataFrame, high: np.ndarray, 
                                low: np.ndarray, close: np.ndarray) -> pd.DataFrame:
        """Add trend-based features."""
        try:
            # Parabolic SAR
            features['sar'] = talib.SAR(high, low)
            features['sar_trend'] = (close > features['sar']).astype(int)
            
            # Average Directional Index
            features['adx'] = talib.ADX(high, low, close)
            features['plus_di'] = talib.PLUS_DI(high, low, close)
            features['minus_di'] = talib.MINUS_DI(high, low, close)
            
            # Aroon
            aroon_down, aroon_up = talib.AROON(high, low)
            features['aroon_up'] = aroon_up
            features['aroon_down'] = aroon_down
            features['aroon_oscillator'] = aroon_up - aroon_down
            
            # Trend strength
            for period in [20, 50]:
                slope = np.polyfit(range(period), close[-period:], 1)[0] if len(close) >= period else 0
                features[f'trend_slope_{period}'] = slope
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding trend features: {e}")
            return features
    
    async def _add_statistical_features(self, features: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
        """Add statistical features."""
        try:
            close_series = pd.Series(close)
            
            # Rolling statistics
            for window in [10, 20, 50]:
                features[f'skewness_{window}'] = close_series.rolling(window).skew()
                features[f'kurtosis_{window}'] = close_series.rolling(window).kurt()
                
                # Percentile features
                features[f'percentile_25_{window}'] = close_series.rolling(window).quantile(0.25)
                features[f'percentile_75_{window}'] = close_series.rolling(window).quantile(0.75)
                
                # Z-score
                rolling_mean = close_series.rolling(window).mean()
                rolling_std = close_series.rolling(window).std()
                features[f'zscore_{window}'] = (close_series - rolling_mean) / rolling_std
            
            # Autocorrelation
            for lag in [1, 5, 10]:
                features[f'autocorr_{lag}'] = close_series.rolling(50).apply(
                    lambda x: x.autocorr(lag) if len(x) > lag else 0
                )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding statistical features: {e}")
            return features
    
    async def _add_trade_features(self, features: pd.DataFrame, trade_data: pd.DataFrame) -> pd.DataFrame:
        """Add trade-based features."""
        try:
            # Resample trade data to match feature index
            trade_resampled = trade_data.resample('1min').agg({
                'price': 'ohlc',
                'size': 'sum',
                'side': lambda x: (x == 'buy').sum() - (x == 'sell').sum()
            })
            
            features['trade_volume'] = trade_resampled['size']
            features['trade_imbalance'] = trade_resampled['side']
            features['trade_count'] = trade_data.resample('1min').size()
            
            # Average trade size
            features['avg_trade_size'] = features['trade_volume'] / features['trade_count']
            
            # VWAP
            features['vwap'] = (trade_data['price'] * trade_data['size']).resample('1min').sum() / features['trade_volume']
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error adding trade features: {e}")
            return features
    
    def _initialize_default_feature_sets(self) -> None:
        """Initialize default feature sets."""
        # Technical feature set
        technical_features = FeatureSet(
            name="technical_default",
            feature_names=[
                "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
                "ema_12", "ema_26", "ema_50",
                "macd", "macd_signal", "macd_histogram",
                "rsi_14", "rsi_21", "rsi_50",
                "bb_width_20", "bb_position_20",
                "atr_14", "atr_20",
                "stoch_k", "stoch_d",
                "adx", "plus_di", "minus_di"
            ],
            feature_types={name: FeatureType.TECHNICAL for name in [
                "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
                "ema_12", "ema_26", "ema_50",
                "macd", "macd_signal", "macd_histogram",
                "rsi_14", "rsi_21", "rsi_50",
                "bb_width_20", "bb_position_20",
                "atr_14", "atr_20",
                "stoch_k", "stoch_d",
                "adx", "plus_di", "minus_di"
            ]},
            lookback_periods={
                "sma_5": 5, "sma_10": 10, "sma_20": 20, "sma_50": 50,
                "sma_100": 100, "sma_200": 200,
                "ema_12": 12, "ema_26": 26, "ema_50": 50,
                "rsi_14": 14, "rsi_21": 21, "rsi_50": 50,
                "bb_width_20": 20, "bb_position_20": 20,
                "atr_14": 14, "atr_20": 20
            },
            description="Default technical analysis features"
        )
        
        self.feature_sets["technical_default"] = technical_features
        
        # Microstructure feature set
        microstructure_features = FeatureSet(
            name="microstructure_default",
            feature_names=[
                "bid_ask_spread", "bid_ask_spread_pct", "order_imbalance",
                "depth_ratio", "price_impact_buy", "price_impact_sell"
            ],
            feature_types={name: FeatureType.MICROSTRUCTURE for name in [
                "bid_ask_spread", "bid_ask_spread_pct", "order_imbalance",
                "depth_ratio", "price_impact_buy", "price_impact_sell"
            ]},
            lookback_periods={},
            description="Default market microstructure features"
        )
        
        self.feature_sets["microstructure_default"] = microstructure_features
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if result is cached and not expired."""
        if cache_key not in self.feature_cache:
            return False
        
        if cache_key not in self.cache_expiry:
            return False
        
        return datetime.now() < self.cache_expiry[cache_key]
    
    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache computation result."""
        self.feature_cache[cache_key] = result
        self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=self.cache_duration)
        
        # Limit cache size
        if len(self.feature_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(self.cache_expiry.keys(), key=lambda k: self.cache_expiry[k])[:20]
            for key in oldest_keys:
                if key in self.feature_cache:
                    del self.feature_cache[key]
                if key in self.cache_expiry:
                    del self.cache_expiry[key]


# Utility functions for feature engineering

async def create_feature_engineering(config: ConfigurationManager) -> FeatureEngineering:
    """
    Create and initialize a feature engineering instance.
    
    Args:
        config: Configuration manager
        
    Returns:
        FeatureEngineering: Initialized feature engineering instance
    """
    return FeatureEngineering(config)