"""
ML Integration Layer - Phase 2.5

This module creates a unified integration layer that connects the comprehensive
ML capabilities (from both ml/ and machine_learning/ packages) directly into
the main trading loop.

Key Components:
- MLFeaturePipeline: Unified feature engineering from both systems
- MLModelManager: Manages predictions from multiple ML models  
- MLStrategyOrchestrator: Combines ML predictions with traditional strategies
- MLExecutionOptimizer: ML-enhanced order execution
- MLPerformanceMonitor: Tracks ML strategy performance

Integration Architecture:
    Market Data → Feature Pipeline → Model Manager → Strategy Orchestrator
                                                              ↓
    Risk Manager ← Execution Engine ← Order Optimizer ← Trading Decision
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass

# Import our existing ML components
try:
    from ..ml import (
        FeatureEngineer, TechnicalIndicators, RegimeAnalyzer,
        LightGBMTrader, XGBoostTrader, EnsembleTrader, DynamicEnsemble
    )
except ImportError:
    print("Warning: ml package not available")
    FeatureEngineer = TechnicalIndicators = RegimeAnalyzer = None
    LightGBMTrader = XGBoostTrader = EnsembleTrader = DynamicEnsemble = None

try:
    from ..machine_learning import (
        MLEngine, FeatureEngineering, PredictionEngine, ModelManager
    )
except ImportError:
    print("Warning: machine_learning package not available")
    MLEngine = FeatureEngineering = PredictionEngine = ModelManager = None

# Import our unified risk management system
from ..risk import UnifiedRiskManager, RiskParameters, PositionSizeMethod

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class MLSignalType(Enum):
    """Types of ML trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class MLStrategyType(Enum):
    """Types of ML-enhanced strategies"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    REGIME_SWITCHING = "regime_switching"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"

class MLConfidenceLevel(Enum):
    """ML prediction confidence levels"""
    VERY_LOW = "very_low"      # 0-20%
    LOW = "low"                # 20-40%
    MODERATE = "moderate"      # 40-60%
    HIGH = "high"              # 60-80%
    VERY_HIGH = "very_high"    # 80-100%

@dataclass
class MLFeatures:
    """Unified ML features from all systems"""
    technical_indicators: Dict[str, float]
    microstructure_features: Dict[str, float]
    regime_features: Dict[str, float]
    cross_asset_features: Dict[str, float]
    alternative_data: Dict[str, float]
    timestamp: datetime
    symbol: str

@dataclass
class MLPrediction:
    """ML model prediction with metadata"""
    signal_type: MLSignalType
    confidence: float
    probability_distribution: Dict[str, float]
    feature_importance: Dict[str, float]
    model_name: str
    timestamp: datetime
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    holding_period: Optional[int] = None

@dataclass
class MLTradingDecision:
    """ML-enhanced trading decision"""
    symbol: str
    action: MLSignalType
    size: Decimal
    confidence: float
    ml_predictions: List[MLPrediction]
    traditional_signals: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    execution_params: Dict[str, Any]
    timestamp: datetime

# ============================================================================
# ML FEATURE PIPELINE
# ============================================================================

class MLFeaturePipeline:
    """
    Unified feature engineering pipeline combining capabilities from
    both ml/ and machine_learning/ packages
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Initialize feature engineering components
        self.feature_engineer = None
        self.technical_indicators = None
        self.regime_analyzer = None
        self.advanced_features = None
        
        self._initialize_components()
        
        # Feature cache for performance
        self.feature_cache = {}
        self.cache_ttl = timedelta(minutes=1)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for feature pipeline"""
        return {
            'technical_indicators': {
                'sma_periods': [5, 10, 20, 50, 200],
                'ema_periods': [12, 26, 50],
                'rsi_period': 14,
                'bollinger_period': 20,
                'macd_params': (12, 26, 9),
                'stochastic_params': (14, 3, 3)
            },
            'microstructure': {
                'orderbook_levels': 10,
                'trade_flow_window': 60,
                'volume_profile_bins': 20
            },
            'regime_detection': {
                'volatility_window': 30,
                'trend_window': 50,
                'regime_types': ['volatility', 'trend', 'correlation']
            },
            'alternative_data': {
                'sentiment_sources': ['news', 'social'],
                'macro_indicators': ['vix', 'dxy', 'yields']
            }
        }
    
    def _initialize_components(self):
        """Initialize feature engineering components"""
        try:
            if FeatureEngineer:
                self.feature_engineer = FeatureEngineer()
                logger.info("Initialized FeatureEngineer from ml package")
                
            if TechnicalIndicators:
                self.technical_indicators = TechnicalIndicators()
                logger.info("Initialized TechnicalIndicators")
                
            if RegimeAnalyzer:
                self.regime_analyzer = RegimeAnalyzer()
                logger.info("Initialized RegimeAnalyzer")
                
            if FeatureEngineering:
                self.advanced_features = FeatureEngineering()
                logger.info("Initialized FeatureEngineering from machine_learning package")
                
        except Exception as e:
            logger.error(f"Error initializing feature components: {e}")
    
    async def process(self, market_data: Dict[str, Any]) -> MLFeatures:
        """
        Process market data into comprehensive ML features
        
        Args:
            market_data: Dictionary containing OHLCV data, orderbook, trades, etc.
            
        Returns:
            MLFeatures object with all engineered features
        """
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        # Check cache first
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        if cache_key in self.feature_cache:
            cached_features, cache_time = self.feature_cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                return cached_features
        
        features = MLFeatures(
            technical_indicators={},
            microstructure_features={},
            regime_features={},
            cross_asset_features={},
            alternative_data={},
            timestamp=datetime.now(),
            symbol=symbol
        )
        
        try:
            # 1. Technical Indicators (from both systems)
            features.technical_indicators = await self._extract_technical_features(market_data)
            
            # 2. Market Microstructure Features
            features.microstructure_features = await self._extract_microstructure_features(market_data)
            
            # 3. Regime Features
            features.regime_features = await self._extract_regime_features(market_data)
            
            # 4. Cross-Asset Features
            features.cross_asset_features = await self._extract_cross_asset_features(market_data)
            
            # 5. Alternative Data Features
            features.alternative_data = await self._extract_alternative_data_features(market_data)
            
            # Cache the results
            self.feature_cache[cache_key] = (features, datetime.now())
            
            logger.debug(f"Extracted {self._count_features(features)} features for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing features for {symbol}: {e}")
            
        return features
    
    async def _extract_technical_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract technical indicator features"""
        features = {}
        
        try:
            # Get OHLCV data
            df = market_data.get('ohlcv_df')
            if df is None or df.empty:
                logger.warning("No OHLCV data available for technical features")
                return features
            
            # Use FeatureEngineer if available
            if self.feature_engineer:
                tech_features = self.feature_engineer.calculate_technical_indicators(df)
                features.update(tech_features)
            
            # Use TechnicalIndicators if available
            if self.technical_indicators:
                additional_tech = self.technical_indicators.calculate_all_indicators(df)
                features.update(additional_tech)
            
            # Use advanced feature engineering if available
            if self.advanced_features:
                advanced_tech = await self.advanced_features.extract_technical_features(df)
                features.update(advanced_tech)
            
            # Manual calculation of key indicators if components not available
            if not features:
                features = await self._calculate_basic_technical_indicators(df)
                
        except Exception as e:
            logger.error(f"Error extracting technical features: {e}")
            
        return features
    
    async def _extract_microstructure_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract market microstructure features"""
        features = {}
        
        try:
            # Order book features
            orderbook = market_data.get('orderbook')
            if orderbook:
                features.update(self._calculate_orderbook_features(orderbook))
            
            # Trade flow features
            trades = market_data.get('trades')
            if trades:
                features.update(self._calculate_trade_flow_features(trades))
            
            # Volume profile features
            volume_data = market_data.get('volume_profile')
            if volume_data:
                features.update(self._calculate_volume_profile_features(volume_data))
                
        except Exception as e:
            logger.error(f"Error extracting microstructure features: {e}")
            
        return features
    
    async def _extract_regime_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract market regime features"""
        features = {}
        
        try:
            if self.regime_analyzer:
                df = market_data.get('ohlcv_df')
                if df is not None and not df.empty:
                    regime_info = self.regime_analyzer.detect_current_regime(df)
                    features['volatility_regime'] = regime_info.volatility_regime
                    features['trend_regime'] = regime_info.trend_regime
                    features['regime_probability'] = regime_info.confidence
                    
        except Exception as e:
            logger.error(f"Error extracting regime features: {e}")
            
        return features
    
    async def _extract_cross_asset_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract cross-asset correlation and relationship features"""
        features = {}
        
        try:
            cross_asset_data = market_data.get('cross_asset_data', {})
            
            # Bitcoin correlation (for altcoins)
            btc_data = cross_asset_data.get('BTC')
            if btc_data:
                features['btc_correlation_1h'] = self._calculate_correlation(
                    market_data.get('returns_1h', []), 
                    btc_data.get('returns_1h', [])
                )
                features['btc_correlation_4h'] = self._calculate_correlation(
                    market_data.get('returns_4h', []), 
                    btc_data.get('returns_4h', [])
                )
            
            # Market index correlations
            for asset in ['ETH', 'BNB', 'SOL']:
                asset_data = cross_asset_data.get(asset)
                if asset_data:
                    corr = self._calculate_correlation(
                        market_data.get('returns_1h', []),
                        asset_data.get('returns_1h', [])
                    )
                    features[f'{asset.lower()}_correlation'] = corr
                    
        except Exception as e:
            logger.error(f"Error extracting cross-asset features: {e}")
            
        return features
    
    async def _extract_alternative_data_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract alternative data features (sentiment, macro, etc.)"""
        features = {}
        
        try:
            alt_data = market_data.get('alternative_data', {})
            
            # Sentiment features
            sentiment_data = alt_data.get('sentiment')
            if sentiment_data:
                features['news_sentiment'] = sentiment_data.get('news_score', 0)
                features['social_sentiment'] = sentiment_data.get('social_score', 0)
                features['sentiment_momentum'] = sentiment_data.get('momentum', 0)
            
            # Macro features
            macro_data = alt_data.get('macro')
            if macro_data:
                features['vix_level'] = macro_data.get('vix', 0)
                features['dxy_change'] = macro_data.get('dxy_change', 0)
                features['yields_10y'] = macro_data.get('yields_10y', 0)
                
        except Exception as e:
            logger.error(f"Error extracting alternative data features: {e}")
            
        return features
    
    def _calculate_correlation(self, returns1: List[float], returns2: List[float]) -> float:
        """Calculate correlation between two return series"""
        try:
            if len(returns1) < 10 or len(returns2) < 10:
                return 0.0
                
            corr = np.corrcoef(returns1, returns2)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _count_features(self, features: MLFeatures) -> int:
        """Count total number of features"""
        return (len(features.technical_indicators) + 
                len(features.microstructure_features) +
                len(features.regime_features) +
                len(features.cross_asset_features) +
                len(features.alternative_data))
    
    async def _calculate_basic_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic technical indicators manually if components not available"""
        features = {}
        
        try:
            # Simple Moving Averages
            for period in [5, 10, 20, 50]:
                if len(df) >= period:
                    sma = df['close'].rolling(window=period).mean().iloc[-1]
                    features[f'sma_{period}'] = float(sma)
                    
                    # Price relative to SMA
                    features[f'price_to_sma_{period}'] = float(df['close'].iloc[-1] / sma - 1)
            
            # RSI
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features['rsi'] = float(rsi.iloc[-1])
            
            # Volatility
            if len(df) >= 20:
                volatility = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
                features['volatility_20d'] = float(volatility.iloc[-1])
                
        except Exception as e:
            logger.error(f"Error calculating basic technical indicators: {e}")
            
        return features
    
    def _calculate_orderbook_features(self, orderbook: Dict[str, Any]) -> Dict[str, float]:
        """Calculate features from order book data"""
        features = {}
        
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if bids and asks:
                # Spread
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                features['bid_ask_spread'] = (best_ask - best_bid) / best_ask
                
                # Order book imbalance
                bid_volume = sum(float(bid[1]) for bid in bids[:5])
                ask_volume = sum(float(ask[1]) for ask in asks[:5])
                total_volume = bid_volume + ask_volume
                if total_volume > 0:
                    features['orderbook_imbalance'] = (bid_volume - ask_volume) / total_volume
                    
        except Exception as e:
            logger.error(f"Error calculating orderbook features: {e}")
            
        return features
    
    def _calculate_trade_flow_features(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate features from trade flow data"""
        features = {}
        
        try:
            if trades:
                # Buy/sell ratio
                buy_volume = sum(float(trade['quantity']) for trade in trades if trade.get('side') == 'buy')
                sell_volume = sum(float(trade['quantity']) for trade in trades if trade.get('side') == 'sell')
                total_volume = buy_volume + sell_volume
                
                if total_volume > 0:
                    features['buy_sell_ratio'] = buy_volume / total_volume
                    
                # Trade size distribution
                trade_sizes = [float(trade['quantity']) for trade in trades]
                if trade_sizes:
                    features['avg_trade_size'] = np.mean(trade_sizes)
                    features['trade_size_std'] = np.std(trade_sizes)
                    
        except Exception as e:
            logger.error(f"Error calculating trade flow features: {e}")
            
        return features
    
    def _calculate_volume_profile_features(self, volume_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate features from volume profile data"""
        features = {}
        
        try:
            # Point of Control (POC)
            features['poc_distance'] = volume_data.get('poc_distance', 0)
            
            # Volume distribution metrics
            features['volume_at_ask'] = volume_data.get('volume_at_ask', 0)
            features['volume_at_bid'] = volume_data.get('volume_at_bid', 0)
            
        except Exception as e:
            logger.error(f"Error calculating volume profile features: {e}")
            
        return features

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'MLFeaturePipeline',
    'MLFeatures',
    'MLPrediction', 
    'MLTradingDecision',
    'MLSignalType',
    'MLStrategyType',
    'MLConfidenceLevel'
]