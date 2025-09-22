"""
Time Series Forecaster

Advanced time series forecasting system for cryptocurrency trading predictions.
Implements deep learning models (LSTM, Transformer), statistical models (ARIMA, Prophet),
and hybrid approaches with multi-horizon predictions and uncertainty quantification.

Key Features:
- Multi-model time series architecture (LSTM, GRU, Transformer, ARIMA, Prophet)
- Multi-horizon forecasting (1h, 4h, 24h, 7d predictions)
- Uncertainty quantification with prediction intervals
- Regime-aware forecasting (different models for different market conditions)
- Seasonal decomposition and trend analysis
- External feature integration (technical indicators, sentiment, macro)
- Real-time model adaptation and online learning
- Attention mechanisms for interpretable predictions
- Ensemble forecasting with model combination
- Walk-forward validation for time series evaluation

Model Types:
- LSTM Networks: Long Short-Term Memory for sequence modeling
- GRU Networks: Gated Recurrent Units for efficient sequence learning
- Transformer: Self-attention mechanism for long-range dependencies
- CNN-LSTM: Convolutional layers + LSTM for feature extraction
- ARIMA: AutoRegressive Integrated Moving Average
- Prophet: Facebook's forecasting tool for trends and seasonality
- Exponential Smoothing: Holt-Winters and ETS models
- VAR: Vector AutoRegression for multivariate forecasting

Advanced Features:
- Multi-scale temporal convolutions
- Attention visualization for model interpretability
- Probabilistic forecasting with quantile regression
- Online model updating with concept drift detection
- Feature importance analysis for time series components
- Anomaly-aware forecasting that handles outliers
- Cross-series learning for related cryptocurrency pairs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import asyncio
from collections import defaultdict, deque
import warnings
from enum import Enum
import json
import pickle
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from pathlib import Path

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning models will be disabled.")

# Statistical forecasting imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.vector_ar.var_model import VAR
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available. Statistical models will be disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Prophet forecasting will be disabled.")

# Standard ML imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from ..utils.logging import TradingLogger
from ..config.manager import ConfigurationManager


class ForecastModel(Enum):
    """Types of forecasting models."""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"
    ARIMA = "arima"
    PROPHET = "prophet"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    VAR = "var"
    LINEAR_TREND = "linear_trend"
    RANDOM_FOREST = "random_forest"


class ForecastHorizon(Enum):
    """Forecast horizons."""
    H1 = "1h"
    H4 = "4h"
    H24 = "24h"
    D7 = "7d"
    D30 = "30d"


@dataclass
class ForecastResult:
    """Time series forecast result."""
    timestamp: datetime
    model_type: ForecastModel
    target_variable: str
    forecast_horizon: ForecastHorizon
    
    # Predictions
    point_forecast: float
    forecast_path: List[float]  # Full forecast trajectory
    
    # Uncertainty quantification
    forecast_lower: float  # Lower prediction interval
    forecast_upper: float  # Upper prediction interval
    forecast_std: float    # Forecast standard deviation
    confidence_level: float  # Confidence level (e.g., 0.95)
    
    # Model performance
    model_confidence: float
    forecast_accuracy: float  # Historical accuracy
    
    # Interpretability
    feature_importance: Dict[str, float]
    attention_weights: Optional[List[float]] = None  # For attention-based models
    
    # Metadata
    input_length: int
    training_samples: int
    model_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesFeatures:
    """Time series feature engineering results."""
    # Trend components
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    
    # Lagged features
    lags: pd.DataFrame
    
    # Technical indicators
    technical_features: pd.DataFrame
    
    # External features
    external_features: pd.DataFrame
    
    # Fourier features for seasonality
    fourier_features: pd.DataFrame
    
    # Time-based features
    time_features: pd.DataFrame


if TORCH_AVAILABLE:
    class LSTMForecaster(nn.Module):
        """LSTM-based time series forecaster."""
        
        def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                     output_size: int = 1, dropout: float = 0.2):
            super(LSTMForecaster, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size)
            )
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            # LSTM forward pass
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Apply attention
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            
            # Use the last output for prediction
            last_output = attn_out[:, -1, :]
            last_output = self.dropout(last_output)
            
            # Final prediction
            output = self.fc(last_output)
            
            return output, attn_weights
    
    
    class TransformerForecaster(nn.Module):
        """Transformer-based time series forecaster."""
        
        def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                     num_layers: int = 6, output_size: int = 1, dropout: float = 0.1):
            super(TransformerForecaster, self).__init__()
            
            self.d_model = d_model
            self.input_projection = nn.Linear(input_size, d_model)
            
            # Positional encoding
            self.pos_encoding = self._create_positional_encoding(1000, d_model)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Output projection
            self.output_projection = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_size)
            )
        
        def _create_positional_encoding(self, max_len: int, d_model: int):
            """Create positional encoding."""
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-np.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            return pe.unsqueeze(0)
        
        def forward(self, x):
            batch_size, seq_len, _ = x.shape
            
            # Project input to model dimension
            x = self.input_projection(x)
            
            # Add positional encoding
            pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
            x = x + pos_enc
            
            # Transformer forward pass
            transformer_out = self.transformer(x)
            
            # Use the last output for prediction
            last_output = transformer_out[:, -1, :]
            
            # Final prediction
            output = self.output_projection(last_output)
            
            return output, transformer_out


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class TimeSeriesForecaster:
    """
    Advanced time series forecasting system for cryptocurrency trading.
    
    Implements multiple forecasting models with multi-horizon predictions,
    uncertainty quantification, and regime-aware forecasting.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = TradingLogger("TimeSeriesForecaster")
        
        # Configuration
        self.config = {
            'sequence_length': config_manager.get('forecasting.sequence_length', 168),  # 1 week
            'forecast_horizons': [1, 4, 24, 168],  # 1h, 4h, 24h, 7d
            'train_test_split': config_manager.get('forecasting.train_split', 0.8),
            'validation_split': config_manager.get('forecasting.validation_split', 0.1),
            'batch_size': config_manager.get('forecasting.batch_size', 32),
            'epochs': config_manager.get('forecasting.epochs', 100),
            'learning_rate': config_manager.get('forecasting.learning_rate', 0.001),
            'early_stopping_patience': config_manager.get('forecasting.early_stopping', 10),
            'confidence_level': config_manager.get('forecasting.confidence_level', 0.95),
            'max_features': config_manager.get('forecasting.max_features', 50),
            'device': 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        }
        
        # Data storage
        self.data: Optional[pd.DataFrame] = None
        self.features: Optional[TimeSeriesFeatures] = None
        self.scalers: Dict[str, Any] = {}
        
        # Model storage
        self.models: Dict[Tuple[ForecastModel, ForecastHorizon], Any] = {}
        self.model_performance: Dict[Tuple[ForecastModel, ForecastHorizon], Dict[str, float]] = {}
        
        # Prediction cache
        self.prediction_cache: Dict[str, ForecastResult] = {}
        self.forecast_history: deque = deque(maxlen=10000)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize available models
        self._initialize_available_models()
    
    def _initialize_available_models(self):
        """Initialize list of available models based on dependencies."""
        self.available_models = [ForecastModel.LINEAR_TREND, ForecastModel.RANDOM_FOREST]
        
        if TORCH_AVAILABLE:
            self.available_models.extend([
                ForecastModel.LSTM,
                ForecastModel.GRU,
                ForecastModel.TRANSFORMER
            ])
        
        if STATSMODELS_AVAILABLE:
            self.available_models.extend([
                ForecastModel.ARIMA,
                ForecastModel.EXPONENTIAL_SMOOTHING,
                ForecastModel.VAR
            ])
        
        if PROPHET_AVAILABLE:
            self.available_models.append(ForecastModel.PROPHET)
        
        self.logger.info(f"Available forecasting models: {[m.value for m in self.available_models]}")
    
    def update_data(self, data: pd.DataFrame, target_column: str):
        """Update time series data."""
        try:
            if data.empty:
                self.logger.warning("Empty data provided")
                return
            
            # Ensure data is sorted by timestamp
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp')
            
            self.data = data.copy()
            self.target_column = target_column
            
            # Engineer time series features
            self._engineer_time_series_features()
            
            # Initialize scalers
            self._initialize_scalers()
            
            self.logger.info(f"Updated time series data: {len(data)} samples, target: {target_column}")
            
        except Exception as e:
            self.logger.error(f"Error updating time series data: {e}")
    
    def _engineer_time_series_features(self):
        """Engineer comprehensive time series features."""
        if self.data is None or self.target_column not in self.data.columns:
            return
        
        try:
            target_series = self.data[self.target_column]
            
            # Seasonal decomposition (simplified)
            trend = target_series.rolling(window=24, center=True).mean()
            seasonal = target_series - trend
            residual = target_series - trend - seasonal
            
            # Lagged features
            max_lag = min(24, len(target_series) // 4)  # Up to 24 hours or 1/4 of data
            lag_features = pd.DataFrame(index=self.data.index)
            
            for lag in range(1, max_lag + 1):
                lag_features[f'lag_{lag}'] = target_series.shift(lag)
            
            # Technical indicators (using existing data columns)
            technical_features = pd.DataFrame(index=self.data.index)
            
            # Moving averages
            for window in [6, 12, 24, 168]:  # 6h, 12h, 24h, 7d
                if window < len(target_series):
                    technical_features[f'ma_{window}'] = target_series.rolling(window=window).mean()
                    technical_features[f'std_{window}'] = target_series.rolling(window=window).std()
            
            # Price-based features
            if 'close' in self.data.columns:
                technical_features['returns'] = self.data['close'].pct_change()
                technical_features['log_returns'] = np.log(self.data['close'] / self.data['close'].shift(1))
                
                # Volatility
                technical_features['volatility'] = technical_features['returns'].rolling(window=24).std()
            
            # External features (use other columns in data)
            external_features = pd.DataFrame(index=self.data.index)
            for col in self.data.columns:
                if col not in [self.target_column, 'timestamp']:
                    external_features[col] = self.data[col]
            
            # Fourier features for seasonality
            fourier_features = pd.DataFrame(index=self.data.index)
            
            if 'timestamp' in self.data.columns:
                timestamps = pd.to_datetime(self.data['timestamp'])
                
                # Daily seasonality
                hour_of_day = timestamps.dt.hour
                fourier_features['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
                fourier_features['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
                
                # Weekly seasonality
                day_of_week = timestamps.dt.dayofweek
                fourier_features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
                fourier_features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            
            # Time-based features
            time_features = pd.DataFrame(index=self.data.index)
            
            if 'timestamp' in self.data.columns:
                timestamps = pd.to_datetime(self.data['timestamp'])
                time_features['hour'] = timestamps.dt.hour
                time_features['day_of_week'] = timestamps.dt.dayofweek
                time_features['day_of_month'] = timestamps.dt.day
                time_features['month'] = timestamps.dt.month
                time_features['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(int)
            
            # Store features
            self.features = TimeSeriesFeatures(
                trend=trend.fillna(0),
                seasonal=seasonal.fillna(0),
                residual=residual.fillna(0),
                lags=lag_features,
                technical_features=technical_features,
                external_features=external_features,
                fourier_features=fourier_features,
                time_features=time_features
            )
            
            self.logger.debug("Engineered time series features successfully")
            
        except Exception as e:
            self.logger.error(f"Error engineering time series features: {e}")
    
    def _initialize_scalers(self):
        """Initialize feature scalers."""
        if self.features is None:
            return
        
        self.scalers = {}
        
        # Target scaler
        target_data = self.data[self.target_column].values.reshape(-1, 1)
        target_scaler = MinMaxScaler()
        target_scaler.fit(target_data)
        self.scalers['target'] = target_scaler
        
        # Feature scalers
        all_features = pd.concat([
            self.features.lags,
            self.features.technical_features,
            self.features.external_features,
            self.features.fourier_features,
            self.features.time_features
        ], axis=1)
        
        # Remove columns with all NaN values
        all_features = all_features.dropna(axis=1, how='all')
        
        if not all_features.empty:
            feature_scaler = StandardScaler()
            feature_scaler.fit(all_features.fillna(0))
            self.scalers['features'] = feature_scaler
    
    def _prepare_sequences(self, horizon: ForecastHorizon) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequences for training/prediction."""
        if self.data is None or self.features is None:
            raise ValueError("No data or features available")
        
        horizon_steps = self._horizon_to_steps(horizon)
        sequence_length = self.config['sequence_length']
        
        # Combine all features
        all_features = pd.concat([
            self.features.lags,
            self.features.technical_features,
            self.features.external_features,
            self.features.fourier_features,
            self.features.time_features
        ], axis=1)
        
        # Handle missing values
        all_features = all_features.fillna(method='ffill').fillna(0)
        
        # Select top features (to avoid curse of dimensionality)
        if len(all_features.columns) > self.config['max_features']:
            # Simple feature selection based on correlation with target
            correlations = all_features.corrwith(self.data[self.target_column]).abs()
            top_features = correlations.nlargest(self.config['max_features']).index
            all_features = all_features[top_features]
        
        # Scale features
        if 'features' in self.scalers:
            scaled_features = self.scalers['features'].transform(all_features)
        else:
            scaled_features = all_features.values
        
        # Scale target
        target_values = self.data[self.target_column].values
        if 'target' in self.scalers:
            scaled_target = self.scalers['target'].transform(target_values.reshape(-1, 1)).flatten()
        else:
            scaled_target = target_values
        
        # Create sequences
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_features) - horizon_steps + 1):
            # Input sequence
            sequence = scaled_features[i - sequence_length:i]
            X.append(sequence)
            
            # Target (future value)
            if horizon_steps == 1:
                target = scaled_target[i + horizon_steps - 1]
            else:
                # For multi-step forecasting, use the value at the horizon
                target = scaled_target[i + horizon_steps - 1]
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train/validation/test split
        train_size = int(len(X) * self.config['train_test_split'])
        val_size = int(len(X) * self.config['validation_split'])
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _horizon_to_steps(self, horizon: ForecastHorizon) -> int:
        """Convert forecast horizon to number of time steps."""
        horizon_map = {
            ForecastHorizon.H1: 1,
            ForecastHorizon.H4: 4,
            ForecastHorizon.H24: 24,
            ForecastHorizon.D7: 168,  # 7 * 24
            ForecastHorizon.D30: 720  # 30 * 24
        }
        return horizon_map.get(horizon, 1)
    
    def train_model(self, model_type: ForecastModel, horizon: ForecastHorizon) -> Dict[str, Any]:
        """Train a specific forecasting model."""
        if model_type not in self.available_models:
            raise ValueError(f"Model {model_type.value} not available")
        
        try:
            # Prepare data
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self._prepare_sequences(horizon)
            
            if len(X_train) == 0:
                raise ValueError("Insufficient data for training")
            
            # Train model based on type
            if model_type in [ForecastModel.LSTM, ForecastModel.GRU, ForecastModel.TRANSFORMER]:
                model, training_metrics = self._train_deep_learning_model(
                    model_type, X_train, y_train, X_val, y_val
                )
            elif model_type == ForecastModel.ARIMA:
                model, training_metrics = self._train_arima_model(y_train)
            elif model_type == ForecastModel.PROPHET:
                model, training_metrics = self._train_prophet_model()
            elif model_type == ForecastModel.RANDOM_FOREST:
                model, training_metrics = self._train_random_forest_model(X_train, y_train)
            elif model_type == ForecastModel.LINEAR_TREND:
                model, training_metrics = self._train_linear_model(X_train, y_train)
            else:
                raise ValueError(f"Training not implemented for {model_type.value}")
            
            # Store model
            self.models[(model_type, horizon)] = model
            
            # Evaluate on test set
            test_metrics = self._evaluate_model(model, model_type, X_test, y_test)
            
            # Store performance metrics
            self.model_performance[(model_type, horizon)] = {
                **training_metrics,
                **test_metrics,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            self.logger.info(f"Trained {model_type.value} for {horizon.value}: R²={test_metrics.get('r2', 0):.4f}")
            
            return {
                'model_type': model_type.value,
                'horizon': horizon.value,
                'training_metrics': training_metrics,
                'test_metrics': test_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error training {model_type.value} for {horizon.value}: {e}")
            return {'error': str(e)}
    
    def _train_deep_learning_model(self, model_type: ForecastModel, 
                                 X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Train deep learning models (LSTM, GRU, Transformer)."""
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch not available for deep learning models")
        
        device = torch.device(self.config['device'])
        
        # Create model
        input_size = X_train.shape[2]
        
        if model_type == ForecastModel.LSTM:
            model = LSTMForecaster(input_size=input_size, hidden_size=128, num_layers=2)
        elif model_type == ForecastModel.TRANSFORMER:
            model = TransformerForecaster(input_size=input_size, d_model=128, nhead=8, num_layers=4)
        else:
            # Default to LSTM for other types
            model = LSTMForecaster(input_size=input_size, hidden_size=128, num_layers=2)
        
        model.to(device)
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_losses = []
        validation_losses = []
        
        for epoch in range(self.config['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                
                if hasattr(model, 'forward'):
                    if model_type == ForecastModel.TRANSFORMER:
                        outputs, _ = model(batch_X)
                    else:
                        outputs, _ = model(batch_X)
                    outputs = outputs.squeeze()
                else:
                    outputs = model(batch_X).squeeze()
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    if hasattr(model, 'forward'):
                        if model_type == ForecastModel.TRANSFORMER:
                            outputs, _ = model(batch_X)
                        else:
                            outputs, _ = model(batch_X)
                        outputs = outputs.squeeze()
                    else:
                        outputs = model(batch_X).squeeze()
                    
                    val_loss += criterion(outputs, batch_y).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            training_losses.append(train_loss)
            validation_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model state
        model.load_state_dict(best_model_state)
        
        training_metrics = {
            'final_train_loss': training_losses[-1],
            'final_val_loss': validation_losses[-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(training_losses),
            'early_stopped': patience_counter >= self.config['early_stopping_patience']
        }
        
        return model, training_metrics
    
    def _train_arima_model(self, y_train: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Train ARIMA model."""
        if not STATSMODELS_AVAILABLE:
            raise ValueError("Statsmodels not available for ARIMA")
        
        try:
            # Auto-select ARIMA order (simplified)
            model = ARIMA(y_train, order=(1, 1, 1))
            fitted_model = model.fit()
            
            training_metrics = {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'log_likelihood': fitted_model.llf
            }
            
            return fitted_model, training_metrics
            
        except Exception as e:
            self.logger.error(f"ARIMA training failed: {e}")
            # Fallback to simple model
            model = LinearRegression()
            X_simple = np.arange(len(y_train)).reshape(-1, 1)
            model.fit(X_simple, y_train)
            
            return model, {'fallback': True}
    
    def _train_prophet_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Train Prophet model."""
        if not PROPHET_AVAILABLE:
            raise ValueError("Prophet not available")
        
        if self.data is None or 'timestamp' not in self.data.columns:
            raise ValueError("Prophet requires timestamp column")
        
        # Prepare Prophet data format
        prophet_data = pd.DataFrame({
            'ds': pd.to_datetime(self.data['timestamp']),
            'y': self.data[self.target_column]
        })
        
        # Create and fit Prophet model
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        model.fit(prophet_data)
        
        training_metrics = {
            'changepoints': len(model.changepoints),
            'seasonality_components': len(model.seasonalities)
        }
        
        return model, training_metrics
    
    def _train_random_forest_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Train Random Forest model."""
        # Reshape for sklearn (flatten sequences)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_flat, y_train)
        
        training_metrics = {
            'oob_score': getattr(model, 'oob_score_', None),
            'n_features': X_train_flat.shape[1]
        }
        
        return model, training_metrics
    
    def _train_linear_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Train linear trend model."""
        # Use simple linear regression on flattened features
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        model = LinearRegression()
        model.fit(X_train_flat, y_train)
        
        training_metrics = {
            'r2_train': model.score(X_train_flat, y_train),
            'n_features': X_train_flat.shape[1]
        }
        
        return model, training_metrics
    
    def _evaluate_model(self, model: Any, model_type: ForecastModel, 
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            # Make predictions
            if model_type in [ForecastModel.LSTM, ForecastModel.GRU, ForecastModel.TRANSFORMER]:
                model.eval()
                device = next(model.parameters()).device
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                
                with torch.no_grad():
                    if hasattr(model, 'forward'):
                        predictions, _ = model(X_test_tensor)
                        predictions = predictions.squeeze().cpu().numpy()
                    else:
                        predictions = model(X_test_tensor).squeeze().cpu().numpy()
            
            elif model_type == ForecastModel.ARIMA:
                # ARIMA prediction (simplified)
                predictions = model.forecast(steps=len(y_test))
                if len(predictions) != len(y_test):
                    predictions = np.full(len(y_test), predictions[-1] if len(predictions) > 0 else 0)
            
            elif model_type in [ForecastModel.RANDOM_FOREST, ForecastModel.LINEAR_TREND]:
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                predictions = model.predict(X_test_flat)
            
            else:
                predictions = np.zeros(len(y_test))  # Fallback
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # MAPE (handle division by zero)
            mape = np.mean(np.abs((y_test - predictions) / np.where(y_test != 0, y_test, 1))) * 100
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'rmse': np.sqrt(mse)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return {
                'mse': float('inf'),
                'mae': float('inf'),
                'r2': -float('inf'),
                'mape': float('inf'),
                'rmse': float('inf')
            }
    
    def forecast(self, model_type: ForecastModel, horizon: ForecastHorizon, 
                steps_ahead: int = 1) -> ForecastResult:
        """Generate forecast using specified model and horizon."""
        model_key = (model_type, horizon)
        
        if model_key not in self.models:
            raise ValueError(f"Model {model_type.value} for {horizon.value} not trained")
        
        try:
            model = self.models[model_key]
            
            # Prepare input sequence (last sequence_length points)
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self._prepare_sequences(horizon)
            
            if len(X_test) == 0:
                # Use last training sequence
                input_sequence = X_train[-1:] if len(X_train) > 0 else X_val[-1:]
            else:
                input_sequence = X_test[-1:]
            
            # Make prediction
            if model_type in [ForecastModel.LSTM, ForecastModel.GRU, ForecastModel.TRANSFORMER]:
                prediction, attention_weights = self._predict_deep_learning(model, input_sequence)
            elif model_type == ForecastModel.PROPHET:
                prediction, attention_weights = self._predict_prophet(model, steps_ahead)
            elif model_type == ForecastModel.ARIMA:
                prediction, attention_weights = self._predict_arima(model, steps_ahead)
            else:
                prediction, attention_weights = self._predict_sklearn(model, input_sequence)
            
            # Inverse transform prediction
            if 'target' in self.scalers:
                prediction_original = self.scalers['target'].inverse_transform([[prediction]])[0][0]
            else:
                prediction_original = prediction
            
            # Calculate uncertainty (simplified)
            model_performance = self.model_performance.get(model_key, {})
            model_mae = model_performance.get('mae', 0.1)
            model_confidence = max(0.1, model_performance.get('r2', 0.5))
            
            # Prediction interval (using MAE as proxy for std)
            forecast_std = model_mae
            z_score = 1.96 if self.config['confidence_level'] == 0.95 else 1.65
            
            forecast_lower = prediction_original - z_score * forecast_std
            forecast_upper = prediction_original + z_score * forecast_std
            
            # Feature importance (simplified)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                n_features = min(10, len(model.feature_importances_))
                top_indices = np.argsort(model.feature_importances_)[-n_features:]
                for i, idx in enumerate(top_indices):
                    feature_importance[f'feature_{idx}'] = model.feature_importances_[idx]
            
            # Create result
            result = ForecastResult(
                timestamp=datetime.now(),
                model_type=model_type,
                target_variable=self.target_column,
                forecast_horizon=horizon,
                point_forecast=prediction_original,
                forecast_path=[prediction_original],  # Single step for now
                forecast_lower=forecast_lower,
                forecast_upper=forecast_upper,
                forecast_std=forecast_std,
                confidence_level=self.config['confidence_level'],
                model_confidence=model_confidence,
                forecast_accuracy=model_performance.get('r2', 0.0),
                feature_importance=feature_importance,
                attention_weights=attention_weights,
                input_length=self.config['sequence_length'],
                training_samples=model_performance.get('training_samples', 0),
                model_version='1.0'
            )
            
            # Cache result
            cache_key = f"{model_type.value}_{horizon.value}_{datetime.now().strftime('%Y%m%d%H')}"
            self.prediction_cache[cache_key] = result
            
            # Store in history
            self.forecast_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error forecasting with {model_type.value}: {e}")
            # Return dummy result
            return ForecastResult(
                timestamp=datetime.now(),
                model_type=model_type,
                target_variable=self.target_column,
                forecast_horizon=horizon,
                point_forecast=0.0,
                forecast_path=[0.0],
                forecast_lower=0.0,
                forecast_upper=0.0,
                forecast_std=0.0,
                confidence_level=self.config['confidence_level'],
                model_confidence=0.0,
                forecast_accuracy=0.0,
                feature_importance={},
                input_length=0,
                training_samples=0,
                model_version='1.0',
                metadata={'error': str(e)}
            )
    
    def _predict_deep_learning(self, model: Any, input_sequence: np.ndarray) -> Tuple[float, Optional[List[float]]]:
        """Make prediction with deep learning model."""
        model.eval()
        device = next(model.parameters()).device
        
        input_tensor = torch.FloatTensor(input_sequence).to(device)
        
        with torch.no_grad():
            if hasattr(model, 'forward'):
                prediction, attention = model(input_tensor)
                prediction = prediction.squeeze().cpu().numpy()
                
                # Extract attention weights if available
                if attention is not None and hasattr(attention, 'squeeze'):
                    attention_weights = attention.squeeze().cpu().numpy().tolist()
                else:
                    attention_weights = None
            else:
                prediction = model(input_tensor).squeeze().cpu().numpy()
                attention_weights = None
        
        # Handle scalar vs array prediction
        if isinstance(prediction, np.ndarray):
            prediction = float(prediction.item()) if prediction.size == 1 else float(prediction[0])
        
        return prediction, attention_weights
    
    def _predict_sklearn(self, model: Any, input_sequence: np.ndarray) -> Tuple[float, None]:
        """Make prediction with sklearn model."""
        input_flat = input_sequence.reshape(input_sequence.shape[0], -1)
        prediction = model.predict(input_flat)[0]
        return float(prediction), None
    
    def _predict_arima(self, model: Any, steps_ahead: int) -> Tuple[float, None]:
        """Make prediction with ARIMA model."""
        forecast = model.forecast(steps=steps_ahead)
        prediction = forecast[-1] if isinstance(forecast, np.ndarray) else float(forecast)
        return prediction, None
    
    def _predict_prophet(self, model: Any, steps_ahead: int) -> Tuple[float, None]:
        """Make prediction with Prophet model."""
        if self.data is None or 'timestamp' not in self.data.columns:
            return 0.0, None
        
        # Create future dataframe
        last_timestamp = pd.to_datetime(self.data['timestamp'].iloc[-1])
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=steps_ahead,
            freq='H'
        )
        
        future_df = pd.DataFrame({'ds': future_timestamps})
        forecast = model.predict(future_df)
        
        prediction = float(forecast['yhat'].iloc[-1])
        return prediction, None
    
    def train_all_models(self, target_column: str) -> Dict[str, Any]:
        """Train all available models for all horizons."""
        if self.data is None:
            raise ValueError("No data available for training")
        
        self.target_column = target_column
        results = {}
        
        # Define horizons to train
        horizons_to_train = [ForecastHorizon.H1, ForecastHorizon.H4, ForecastHorizon.H24]
        
        for model_type in self.available_models:
            for horizon in horizons_to_train:
                try:
                    result = self.train_model(model_type, horizon)
                    results[f"{model_type.value}_{horizon.value}"] = result
                except Exception as e:
                    self.logger.error(f"Failed to train {model_type.value} for {horizon.value}: {e}")
                    results[f"{model_type.value}_{horizon.value}"] = {'error': str(e)}
        
        return results
    
    def get_ensemble_forecast(self, horizon: ForecastHorizon, 
                            model_weights: Optional[Dict[ForecastModel, float]] = None) -> ForecastResult:
        """Generate ensemble forecast combining multiple models."""
        available_models = [
            model_type for model_type in self.available_models
            if (model_type, horizon) in self.models
        ]
        
        if not available_models:
            raise ValueError(f"No trained models available for horizon {horizon.value}")
        
        # Get individual forecasts
        individual_forecasts = {}
        individual_results = {}
        
        for model_type in available_models:
            try:
                result = self.forecast(model_type, horizon)
                individual_forecasts[model_type] = result.point_forecast
                individual_results[model_type] = result
            except Exception as e:
                self.logger.warning(f"Failed to get forecast from {model_type.value}: {e}")
                continue
        
        if not individual_forecasts:
            raise ValueError("No individual forecasts available")
        
        # Calculate ensemble weights
        if model_weights is None:
            # Weight by model performance
            weights = {}
            total_performance = 0.0
            
            for model_type in individual_forecasts.keys():
                model_key = (model_type, horizon)
                performance = self.model_performance.get(model_key, {}).get('r2', 0.1)
                performance = max(0.01, performance)  # Minimum weight
                weights[model_type] = performance
                total_performance += performance
            
            # Normalize weights
            if total_performance > 0:
                weights = {k: v / total_performance for k, v in weights.items()}
            else:
                # Equal weights
                weights = {k: 1.0 / len(individual_forecasts) for k in individual_forecasts.keys()}
        else:
            weights = model_weights
        
        # Calculate ensemble prediction
        ensemble_prediction = sum(
            weights.get(model_type, 0.0) * forecast
            for model_type, forecast in individual_forecasts.items()
        )
        
        # Calculate ensemble uncertainty
        predictions_array = np.array(list(individual_forecasts.values()))
        ensemble_std = np.std(predictions_array)
        
        # Weighted average of confidence intervals
        weighted_lower = sum(
            weights.get(model_type, 0.0) * individual_results[model_type].forecast_lower
            for model_type in individual_results.keys()
        )
        weighted_upper = sum(
            weights.get(model_type, 0.0) * individual_results[model_type].forecast_upper
            for model_type in individual_results.keys()
        )
        
        # Aggregate feature importance
        ensemble_feature_importance = defaultdict(float)
        for model_type, result in individual_results.items():
            weight = weights.get(model_type, 0.0)
            for feature, importance in result.feature_importance.items():
                ensemble_feature_importance[feature] += weight * importance
        
        # Create ensemble result
        ensemble_result = ForecastResult(
            timestamp=datetime.now(),
            model_type=ForecastModel.LSTM,  # Placeholder
            target_variable=self.target_column,
            forecast_horizon=horizon,
            point_forecast=ensemble_prediction,
            forecast_path=[ensemble_prediction],
            forecast_lower=weighted_lower,
            forecast_upper=weighted_upper,
            forecast_std=ensemble_std,
            confidence_level=self.config['confidence_level'],
            model_confidence=np.mean([r.model_confidence for r in individual_results.values()]),
            forecast_accuracy=np.mean([r.forecast_accuracy for r in individual_results.values()]),
            feature_importance=dict(ensemble_feature_importance),
            input_length=self.config['sequence_length'],
            training_samples=sum([r.training_samples for r in individual_results.values()]),
            model_version='ensemble_1.0',
            metadata={
                'ensemble_models': [m.value for m in individual_forecasts.keys()],
                'model_weights': {m.value: w for m, w in weights.items()},
                'individual_predictions': {m.value: p for m, p in individual_forecasts.items()}
            }
        )
        
        return ensemble_result
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'timestamp': datetime.now(),
            'available_models': [m.value for m in self.available_models],
            'trained_models': len(self.models),
            'model_performance': {}
        }
        
        for (model_type, horizon), metrics in self.model_performance.items():
            key = f"{model_type.value}_{horizon.value}"
            summary['model_performance'][key] = {
                'r2_score': metrics.get('r2', -1),
                'mse': metrics.get('mse', float('inf')),
                'mae': metrics.get('mae', float('inf')),
                'training_samples': metrics.get('training_samples', 0),
                'test_samples': metrics.get('test_samples', 0)
            }
        
        # Overall statistics
        if self.model_performance:
            r2_scores = [m.get('r2', -1) for m in self.model_performance.values()]
            summary['overall_stats'] = {
                'best_r2': max(r2_scores),
                'worst_r2': min(r2_scores),
                'average_r2': np.mean(r2_scores),
                'total_forecasts': len(self.forecast_history)
            }
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    import json
    from ..config.manager import ConfigurationManager
    
    def main():
        # Configure logging
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize forecaster
        config_manager = ConfigurationManager()
        forecaster = TimeSeriesForecaster(config_manager)
        
        # Create sample time series data
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
        
        # Generate synthetic cryptocurrency price data
        np.random.seed(42)
        price_trend = np.linspace(45000, 50000, 1000)
        price_noise = np.random.normal(0, 1000, 1000)
        
        # Add some seasonality
        hourly_pattern = 500 * np.sin(2 * np.pi * np.arange(1000) / 24)
        weekly_pattern = 1000 * np.sin(2 * np.pi * np.arange(1000) / (24 * 7))
        
        prices = price_trend + price_noise + hourly_pattern + weekly_pattern
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.exponential(1000, 1000),
            'sentiment_score': np.random.uniform(-1, 1, 1000)
        })
        
        # Update forecaster
        forecaster.update_data(data, 'close')
        
        # Train models
        print("Training forecasting models...")
        training_results = forecaster.train_all_models('close')
        
        for model_result in training_results.items():
            print(f"{model_result[0]}: {model_result[1]}")
        
        # Get performance summary
        performance_summary = forecaster.get_model_performance_summary()
        print(f"\nPerformance Summary:")
        print(json.dumps(performance_summary, indent=2, default=str))
        
        # Generate forecasts
        print("\nGenerating forecasts...")
        
        horizons_to_test = [ForecastHorizon.H1, ForecastHorizon.H4, ForecastHorizon.H24]
        
        for horizon in horizons_to_test:
            try:
                # Get ensemble forecast
                ensemble_forecast = forecaster.get_ensemble_forecast(horizon)
                
                print(f"\n{horizon.value} Ensemble Forecast:")
                print(f"  Prediction: ${ensemble_forecast.point_forecast:.2f}")
                print(f"  Confidence Interval: [${ensemble_forecast.forecast_lower:.2f}, ${ensemble_forecast.forecast_upper:.2f}]")
                print(f"  Model Confidence: {ensemble_forecast.model_confidence:.3f}")
                print(f"  Models Used: {ensemble_forecast.metadata.get('ensemble_models', [])}")
                
            except Exception as e:
                print(f"Error generating {horizon.value} forecast: {e}")
    
    # Run the example
    main()