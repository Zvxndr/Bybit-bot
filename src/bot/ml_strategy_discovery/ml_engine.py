"""
ML Strategy Discovery Engine
Implements machine learning-first approach to trading strategy development
Focused on Australian market conditions and risk parameters
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """ML Strategy types"""
    TREND_FOLLOWING = "ml_trend_following"
    MEAN_REVERSION = "ml_mean_reversion"
    MOMENTUM = "ml_momentum"
    VOLATILITY = "ml_volatility"
    MULTI_FACTOR = "ml_multi_factor"

class ModelType(Enum):
    """Machine learning model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    RIDGE_REGRESSION = "ridge"
    ELASTIC_NET = "elastic_net"
    ENSEMBLE = "ensemble"

@dataclass
class FeatureSet:
    """Feature engineering configuration"""
    technical_indicators: Any = field(default_factory=lambda: [
        'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'ema_5', 'ema_10', 'ema_20',
        'rsi_14', 'rsi_21',
        'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'atr_14', 'atr_21',
        'volume_sma_10', 'volume_ratio',
        'price_change_1', 'price_change_5', 'price_change_10'
    ])
    
    market_features: Any = field(default_factory=lambda: [
        'btc_dominance', 'market_cap_total',
        'fear_greed_index', 'funding_rates',
        'open_interest', 'futures_basis'
    ])
    
    macro_features: Any = field(default_factory=lambda: [
        'aud_usd_rate', 'aud_usd_volatility',
        'rba_cash_rate', 'asx_200_return',
        'us_dxy', 'gold_price_aud',
        'aussie_market_hours'
    ])
    
    time_features: Any = field(default_factory=lambda: [
        'hour_of_day', 'day_of_week', 'day_of_month',
        'is_australian_business_hours', 'is_weekend',
        'sydney_session', 'london_session', 'ny_session'
    ])

@dataclass
class ModelConfiguration:
    """ML model configuration"""
    model_type: ModelType
    target_horizon: int  # Prediction horizon in periods
    lookback_window: int  # Historical data window
    retraining_frequency: int  # Retrain every N periods
    feature_selection_threshold: float = 0.05
    cross_validation_folds: int = 5
    australian_bias: float = 0.3  # Bias toward Australian-friendly features

@dataclass
class StrategySignal:
    """ML-generated trading signal"""
    timestamp: Any
    symbol: Any
    strategy_type: Any
    signal_strength: Any  # -1 to 1
    confidence: Any  # 0 to 1
    predicted_return: Any
    prediction_horizon: Any
    features_used: Any
    model_version: Any
    australian_risk_adjusted: Any = True

class FeatureEngineer:
    """
    Advanced feature engineering for Australian crypto trading
    """
    
    def __init__(self, feature_set: FeatureSet):
        self.feature_set = feature_set
        self.scalers = {}
        
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        
        # Price-based features
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        
        # Volume features
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        
        # Price changes
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        
        return df
    
    def create_australian_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Australian-specific features"""
        
        # Australian market hours (AEST/AEDT)
        df['hour_aest'] = pd.to_datetime(df.index).tz_convert('Australia/Sydney').hour
        df['is_australian_business_hours'] = (
            (df['hour_aest'] >= 9) & (df['hour_aest'] <= 17)
        ).astype(int)
        
        df['sydney_session'] = (
            (df['hour_aest'] >= 8) & (df['hour_aest'] <= 17)
        ).astype(int)
        
        # Weekend indicator (affects liquidity)
        df['is_weekend'] = pd.to_datetime(df.index).dayofweek.isin([5, 6]).astype(int)
        
        # Australian public holidays (simplified)
        # In practice, would use holidays library
        df['is_aus_holiday'] = 0  # Placeholder
        
        # Time-based features
        df['hour_of_day'] = pd.to_datetime(df.index).hour
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['day_of_month'] = pd.to_datetime(df.index).day
        
        return df
    
    def create_macro_features(self, df: pd.DataFrame, macro_data: Dict) -> pd.DataFrame:
        """Create macroeconomic features relevant to Australian traders"""
        
        # AUD/USD exchange rate impact
        if 'aud_usd_rate' in macro_data:
            df['aud_usd_rate'] = macro_data['aud_usd_rate']
            df['aud_usd_change'] = df['aud_usd_rate'].pct_change()
            df['aud_usd_volatility'] = df['aud_usd_change'].rolling(20).std()
        
        # RBA cash rate (Australian monetary policy)
        if 'rba_cash_rate' in macro_data:
            df['rba_cash_rate'] = macro_data['rba_cash_rate']
        
        # ASX 200 correlation
        if 'asx_200' in macro_data:
            df['asx_200_return'] = macro_data['asx_200'].pct_change()
            
        # Gold price in AUD (traditional Australian safe haven)
        if 'gold_price_aud' in macro_data:
            df['gold_price_aud'] = macro_data['gold_price_aud']
            df['gold_return_aud'] = df['gold_price_aud'].pct_change()
        
        return df
    
    def create_target_variables(
        self,
        df: pd.DataFrame,
        horizons = None
    ) -> pd.DataFrame:
        """Create target variables for different prediction horizons"""
        
        if horizons is None:
            horizons = [1, 5, 10, 20]
        
        for horizon in horizons:
            # Forward returns
            df[f'return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon)
            
            # Binary direction
            df[f'direction_{horizon}'] = (df[f'return_{horizon}'] > 0).astype(int)
            
            # Volatility-adjusted returns
            volatility = df['close'].pct_change().rolling(20).std()
            df[f'risk_adj_return_{horizon}'] = df[f'return_{horizon}'] / volatility
        
        return df
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        macro_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        
        logger.info(f"Engineering features for {len(df)} data points")
        
        # Technical features
        df = self.create_technical_features(df)
        
        # Australian-specific features
        df = self.create_australian_features(df)
        
        # Macro features
        if macro_data:
            df = self.create_macro_features(df, macro_data)
        
        # Target variables
        df = self.create_target_variables(df)
        
        # Remove rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        logger.info(f"Feature engineering complete. Rows: {initial_rows} -> {final_rows}")
        
        return df

class MLStrategyModel:
    """
    Individual ML model for strategy generation
    """
    
    def __init__(self, config: ModelConfiguration):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_importance = {}
        self.performance_metrics = {}
        self.last_training_date = None
        self.version = "1.0.0"
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on configuration"""
        
        if self.config.model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.config.model_type == ModelType.GRADIENT_BOOSTING:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        
        elif self.config.model_type == ModelType.RIDGE_REGRESSION:
            self.model = Ridge(alpha=1.0)
        
        elif self.config.model_type == ModelType.ELASTIC_NET:
            self.model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        
        # Initialize scaler
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        
        target_col = f'return_{self.config.target_horizon}'
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) == 0:
            raise ValueError("No features available for training")
        
        # Prepare feature matrix
        X = df[available_features].values
        y = df[target_col].values
        
        # Remove rows with NaN in target
        valid_idx = ~pd.isna(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        return X, y
    
    def train(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train the ML model"""
        
        logger.info(f"Training {self.config.model_type.value} model")
        
        X, y = self.prepare_features(df, feature_columns)
        
        if len(X) < 100:
            raise ValueError("Insufficient data for training (minimum 100 samples required)")
        
        # Split data chronologically (important for time series)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Validation predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_val = self.model.predict(X_val_scaled)
        
        # Calculate performance metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        val_mse = mean_squared_error(y_val, y_pred_val)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            available_features = [col for col in feature_columns if col in df.columns]
            self.feature_importance = dict(zip(
                available_features,
                self.model.feature_importances_
            ))
        
        self.performance_metrics = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'overfitting_ratio': val_mse / train_mse if train_mse > 0 else float('inf'),
            'samples_trained': len(X_train),
            'samples_validated': len(X_val)
        }
        
        self.last_training_date = datetime.now()
        
        logger.info(f"Training complete. Val MSE: {val_mse:.6f}, Val MAE: {val_mae:.6f}")
        
        return self.performance_metrics
    
    def predict(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> np.ndarray:
        """Generate predictions"""
        
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained before prediction")
        
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].values
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        symbol: str,
        strategy_type: StrategyType
    ) -> Optional[StrategySignal]:
        """Generate trading signal from model prediction"""
        
        try:
            predictions = self.predict(df.tail(1), feature_columns)
            
            if len(predictions) == 0:
                return None
            
            prediction = predictions[0]
            
            # Calculate signal strength (-1 to 1)
            # Normalize prediction based on historical volatility
            recent_returns = df['close'].pct_change().tail(20)
            volatility = recent_returns.std()
            
            signal_strength = np.clip(
                prediction / (2 * volatility) if volatility > 0 else 0,
                -1, 1
            )
            
            # Calculate confidence based on model performance
            val_mse = self.performance_metrics.get('val_mse', float('inf'))
            confidence = max(0, min(1, 1 - val_mse * 100))  # Adjust scaling as needed
            
            # Australian risk adjustment
            if self.config.australian_bias > 0:
                # Reduce signal strength for higher Australian risk aversion
                signal_strength *= (1 - self.config.australian_bias * 0.3)
                confidence *= (1 - self.config.australian_bias * 0.1)
            
            return StrategySignal(
                timestamp=datetime.now(),
                symbol=symbol,
                strategy_type=strategy_type,
                signal_strength=signal_strength,
                confidence=confidence,
                predicted_return=prediction,
                prediction_horizon=self.config.target_horizon,
                features_used=[col for col in feature_columns if col in df.columns],
                model_version=self.version,
                australian_risk_adjusted=True
            )
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_importance': self.feature_importance,
            'performance_metrics': self.performance_metrics,
            'version': self.version,
            'last_training_date': self.last_training_date
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_importance = data['feature_importance']
        self.performance_metrics = data['performance_metrics']
        self.version = data['version']
        self.last_training_date = data['last_training_date']

class MLStrategyDiscoveryEngine:
    """
    Main ML Strategy Discovery Engine
    Coordinates multiple ML models for comprehensive strategy generation
    """
    
    def __init__(self, australian_bias: float = 0.3):
        self.australian_bias = australian_bias
        self.feature_engineer = FeatureEngineer(FeatureSet())
        self.models = {}
        self.strategy_allocations = {
            StrategyType.TREND_FOLLOWING: 0.45,
            StrategyType.MEAN_REVERSION: 0.25,
            StrategyType.MOMENTUM: 0.15,
            StrategyType.VOLATILITY: 0.10,
            StrategyType.MULTI_FACTOR: 0.05
        }
        
        self._initialize_models()
        
        logger.info(f"Initialized ML Strategy Discovery Engine with {len(self.models)} models")
    
    def _initialize_models(self):
        """Initialize ML models for different strategies"""
        
        model_configs = [
            # Trend following models
            ModelConfiguration(
                model_type=ModelType.GRADIENT_BOOSTING,
                target_horizon=10,
                lookback_window=50,
                retraining_frequency=100,
                australian_bias=self.australian_bias
            ),
            
            # Mean reversion models
            ModelConfiguration(
                model_type=ModelType.RANDOM_FOREST,
                target_horizon=5,
                lookback_window=30,
                retraining_frequency=50,
                australian_bias=self.australian_bias
            ),
            
            # Momentum models
            ModelConfiguration(
                model_type=ModelType.ELASTIC_NET,
                target_horizon=3,
                lookback_window=20,
                retraining_frequency=25,
                australian_bias=self.australian_bias
            ),
            
            # Volatility models
            ModelConfiguration(
                model_type=ModelType.RIDGE_REGRESSION,
                target_horizon=1,
                lookback_window=15,
                retraining_frequency=20,
                australian_bias=self.australian_bias
            )
        ]
        
        for i, config in enumerate(model_configs):
            model_id = f"model_{i}_{config.model_type.value}_{config.target_horizon}"
            self.models[model_id] = MLStrategyModel(config)
    
    def train_models(
        self,
        data: Dict[str, pd.DataFrame],
        macro_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Train all ML models"""
        
        training_results = {}
        
        for symbol, df in data.items():
            logger.info(f"Training models for {symbol}")
            
            # Engineer features
            df_features = self.feature_engineer.engineer_features(df, macro_data)
            
            # Get feature columns
            feature_columns = [col for col in df_features.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume'] 
                             and not col.startswith('return_')
                             and not col.startswith('direction_')]
            
            symbol_results = {}
            
            for model_id, model in self.models.items():
                try:
                    metrics = model.train(df_features, feature_columns)
                    symbol_results[model_id] = {
                        'success': True,
                        'metrics': metrics,
                        'feature_count': len(feature_columns)
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_id} for {symbol}: {e}")
                    symbol_results[model_id] = {
                        'success': False,
                        'error': str(e)
                    }
            
            training_results[symbol] = symbol_results
        
        return training_results
    
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        macro_data: Optional[Dict] = None
    ) -> List[StrategySignal]:
        """Generate trading signals from all models"""
        
        signals = []
        
        for symbol, df in data.items():
            # Engineer features
            df_features = self.feature_engineer.engineer_features(df, macro_data)
            
            # Get feature columns
            feature_columns = [col for col in df_features.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume'] 
                             and not col.startswith('return_')
                             and not col.startswith('direction_')]
            
            for model_id, model in self.models.items():
                try:
                    # Determine strategy type based on model characteristics
                    if model.config.target_horizon >= 10:
                        strategy_type = StrategyType.TREND_FOLLOWING
                    elif model.config.target_horizon <= 3:
                        strategy_type = StrategyType.MOMENTUM
                    else:
                        strategy_type = StrategyType.MEAN_REVERSION
                    
                    signal = model.generate_signal(
                        df_features,
                        feature_columns,
                        symbol,
                        strategy_type
                    )
                    
                    if signal and abs(signal.signal_strength) > 0.1:  # Minimum threshold
                        signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error generating signal from {model_id} for {symbol}: {e}")
        
        # Sort signals by confidence and strength
        signals.sort(key=lambda s: s.confidence * abs(s.signal_strength), reverse=True)
        
        return signals
    
    def evaluate_model_performance(self) -> Dict[str, Any]:
        """Evaluate performance of all models"""
        
        performance_summary = {}
        
        for model_id, model in self.models.items():
            if model.performance_metrics:
                performance_summary[model_id] = {
                    'validation_mse': model.performance_metrics.get('val_mse'),
                    'validation_mae': model.performance_metrics.get('val_mae'),
                    'overfitting_ratio': model.performance_metrics.get('overfitting_ratio'),
                    'last_training': model.last_training_date,
                    'top_features': dict(sorted(
                        model.feature_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]) if model.feature_importance else {}
                }
        
        return performance_summary
    
    def get_strategy_recommendations(
        self,
        signals: List[StrategySignal],
        portfolio_value: Decimal
    ) -> Dict[str, Any]:
        """Get strategy recommendations based on ML signals and Australian risk parameters"""
        
        # Group signals by strategy type
        strategy_signals = {}
        for signal in signals:
            if signal.strategy_type not in strategy_signals:
                strategy_signals[signal.strategy_type] = []
            strategy_signals[signal.strategy_type].append(signal)
        
        recommendations = {}
        
        for strategy_type, allocation in self.strategy_allocations.items():
            if strategy_type in strategy_signals:
                signals_for_strategy = strategy_signals[strategy_type]
                
                # Calculate average signal strength and confidence
                avg_strength = np.mean([s.signal_strength for s in signals_for_strategy])
                avg_confidence = np.mean([s.confidence for s in signals_for_strategy])
                
                # Australian risk adjustment
                australian_adjusted_strength = avg_strength * (1 - self.australian_bias * 0.2)
                australian_adjusted_confidence = avg_confidence * (1 - self.australian_bias * 0.1)
                
                # Calculate recommended allocation
                base_allocation = portfolio_value * Decimal(str(allocation))
                confidence_adjusted_allocation = base_allocation * Decimal(str(australian_adjusted_confidence))
                
                recommendations[strategy_type.value] = {
                    'signal_count': len(signals_for_strategy),
                    'average_strength': australian_adjusted_strength,
                    'average_confidence': australian_adjusted_confidence,
                    'base_allocation_aud': base_allocation,
                    'recommended_allocation_aud': confidence_adjusted_allocation,
                    'top_signals': signals_for_strategy[:3],  # Top 3 signals
                    'australian_risk_adjusted': True
                }
        
        return recommendations

# Usage example
if __name__ == "__main__":
    # Initialize ML Strategy Discovery Engine with Australian bias
    engine = MLStrategyDiscoveryEngine(australian_bias=0.3)
    
    # Example data preparation
    dates = pd.date_range('2024-01-01', '2024-09-01', freq='1H')
    sample_data = {
        'BTCAUD': pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100000,
            'high': np.random.randn(len(dates)).cumsum() + 101000,
            'low': np.random.randn(len(dates)).cumsum() + 99000,
            'close': np.random.randn(len(dates)).cumsum() + 100000,
            'volume': np.random.rand(len(dates)) * 1000
        }, index=dates)
    }
    
    # Train models
    print("Training ML models...")
    training_results = engine.train_models(sample_data)
    
    # Generate signals
    print("Generating trading signals...")
    signals = engine.generate_signals(sample_data)
    
    print(f"Generated {len(signals)} signals")
    for signal in signals[:5]:  # Show top 5 signals
        print(f"Signal: {signal.symbol} {signal.strategy_type.value} "
              f"Strength: {signal.signal_strength:.3f} "
              f"Confidence: {signal.confidence:.3f}")
    
    # Get recommendations
    recommendations = engine.get_strategy_recommendations(signals, Decimal('50000'))
    print(f"\nStrategy recommendations for $50,000 AUD portfolio:")
    for strategy, rec in recommendations.items():
        print(f"{strategy}: ${rec['recommended_allocation_aud']:,.2f} AUD "
              f"(Confidence: {rec['average_confidence']:.3f})")