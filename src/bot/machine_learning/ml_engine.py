"""
Machine Learning Engine - Core ML Framework

This module provides the core machine learning framework for intelligent trading:
- Multi-model support (TensorFlow, PyTorch, scikit-learn)
- Real-time prediction capabilities
- Online learning and model adaptation
- Performance monitoring and evaluation
- Model versioning and deployment
- Feature importance analysis

The ML Engine integrates with market data to provide:
- Price prediction models
- Volatility forecasting
- Market regime detection
- Risk prediction
- Strategy optimization

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
import pickle
import joblib
from pathlib import Path

# ML Framework imports
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Traditional ML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from ..utils.logging import TradingLogger
from ..config_manager import ConfigurationManager


class ModelType(Enum):
    """Types of ML models supported."""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"


class PredictionType(Enum):
    """Types of predictions."""
    PRICE = "price"
    DIRECTION = "direction"
    VOLATILITY = "volatility"
    RETURN = "return"
    REGIME = "regime"
    RISK = "risk"


class ModelStatus(Enum):
    """Model status states."""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ERROR = "error"


@dataclass
class PredictionResult:
    """ML model prediction result."""
    prediction_type: PredictionType
    value: Union[float, int, str]
    confidence: float
    timestamp: datetime
    model_name: str
    model_version: str
    features_used: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    model_version: str
    prediction_type: PredictionType
    mse: float
    mae: float
    r2_score: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    validation_samples: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLEngine:
    """
    Core Machine Learning Engine for intelligent trading.
    
    This class provides a unified interface for training, evaluating,
    and deploying machine learning models for trading applications.
    
    Features:
    - Multi-framework support (TensorFlow, PyTorch, scikit-learn)
    - Real-time prediction capabilities
    - Online learning and model adaptation
    - Performance monitoring and evaluation
    - Model versioning and persistence
    - Feature importance analysis
    """
    
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.logger = TradingLogger("ml_engine")
        
        # Model registry
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.model_status: Dict[str, ModelStatus] = {}
        
        # Feature preprocessing
        self.scalers: Dict[str, Any] = {}
        self.feature_columns: Dict[str, List[str]] = {}
        
        # Configuration
        self.model_save_path = Path(config.get('ml.model_save_path', 'models'))
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.prediction_history: List[PredictionResult] = []
        self.training_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Online learning parameters
        self.online_learning_enabled = config.get('ml.online_learning.enabled', True)
        self.adaptation_window = config.get('ml.online_learning.adaptation_window', 1000)
        self.retrain_threshold = config.get('ml.online_learning.retrain_threshold', 0.1)
        
        self.logger.info("MLEngine initialized")
    
    async def train_model(
        self,
        model_name: str,
        model_type: ModelType,
        prediction_type: PredictionType,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Train a machine learning model.
        
        Args:
            model_name: Unique name for the model
            model_type: Type of model to train
            prediction_type: Type of prediction the model makes
            X: Feature matrix
            y: Target vector
            validation_split: Fraction of data for validation
            hyperparameters: Model-specific hyperparameters
            
        Returns:
            bool: True if training successful
        """
        try:
            self.logger.info(f"Training model {model_name} ({model_type.value}) for {prediction_type.value}")
            self.model_status[model_name] = ModelStatus.TRAINING
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
            
            # Feature preprocessing
            scaler = self._get_scaler(model_type)
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers[model_name] = scaler
            
            # Train model based on type
            model = None
            training_history = []
            
            if model_type == ModelType.LSTM and TF_AVAILABLE:
                model, history = await self._train_lstm_model(
                    X_train_scaled, y_train, X_val_scaled, y_val, hyperparameters
                )
                training_history = history.history if history else []
                
            elif model_type == ModelType.TRANSFORMER and TF_AVAILABLE:
                model, history = await self._train_transformer_model(
                    X_train_scaled, y_train, X_val_scaled, y_val, hyperparameters
                )
                training_history = history.history if history else []
                
            elif model_type == ModelType.RANDOM_FOREST:
                model = await self._train_random_forest_model(
                    X_train_scaled, y_train, hyperparameters
                )
                
            elif model_type == ModelType.XGBOOST and XGB_AVAILABLE:
                model = await self._train_xgboost_model(
                    X_train_scaled, y_train, X_val_scaled, y_val, hyperparameters
                )
                
            elif model_type == ModelType.LINEAR_REGRESSION:
                model = await self._train_linear_model(
                    X_train_scaled, y_train, hyperparameters
                )
                
            elif model_type == ModelType.ENSEMBLE:
                model = await self._train_ensemble_model(
                    X_train_scaled, y_train, X_val_scaled, y_val, hyperparameters
                )
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            if model is None:
                raise RuntimeError("Model training failed")
            
            # Store model
            self.models[model_name] = model
            self.model_metadata[model_name] = {
                'model_type': model_type,
                'prediction_type': prediction_type,
                'feature_count': X.shape[1],
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'created_at': datetime.now(),
                'hyperparameters': hyperparameters or {},
                'version': '1.0.0'
            }
            self.training_history[model_name] = training_history
            
            # Evaluate model
            performance = await self._evaluate_model(
                model_name, model, X_val_scaled, y_val, prediction_type
            )
            self.model_performance[model_name] = performance
            
            # Save model
            await self._save_model(model_name)
            
            self.model_status[model_name] = ModelStatus.TRAINED
            self.logger.info(f"Model {model_name} trained successfully. R² Score: {performance.r2_score:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model {model_name}: {e}")
            self.model_status[model_name] = ModelStatus.ERROR
            return False
    
    async def predict(
        self,
        model_name: str,
        features: np.ndarray,
        return_confidence: bool = True
    ) -> Optional[PredictionResult]:
        """
        Make a prediction using a trained model.
        
        Args:
            model_name: Name of the model to use
            features: Feature vector for prediction
            return_confidence: Whether to return confidence score
            
        Returns:
            PredictionResult: Prediction result with confidence
        """
        try:
            if model_name not in self.models:
                self.logger.error(f"Model {model_name} not found")
                return None
            
            model = self.models[model_name]
            metadata = self.model_metadata[model_name]
            
            # Preprocess features
            if model_name in self.scalers:
                features_scaled = self.scalers[model_name].transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            
            # Make prediction
            prediction = None
            confidence = 0.0
            
            if metadata['model_type'] in [ModelType.LSTM, ModelType.TRANSFORMER]:
                # TensorFlow models
                pred = model.predict(features_scaled, verbose=0)
                prediction = float(pred[0][0])
                
                if return_confidence:
                    # Estimate confidence based on model uncertainty
                    # For neural networks, use prediction variance or ensemble
                    confidence = min(1.0, 1.0 / (1.0 + abs(prediction) * 0.1))
                    
            elif metadata['model_type'] in [
                ModelType.RANDOM_FOREST, 
                ModelType.XGBOOST,
                ModelType.LINEAR_REGRESSION,
                ModelType.RIDGE_REGRESSION,
                ModelType.GRADIENT_BOOSTING
            ]:
                # Scikit-learn models
                prediction = float(model.predict(features_scaled)[0])
                
                if return_confidence and hasattr(model, 'predict_proba'):
                    # For classification models
                    probas = model.predict_proba(features_scaled)
                    confidence = float(np.max(probas))
                elif return_confidence and metadata['model_type'] == ModelType.RANDOM_FOREST:
                    # For regression Random Forest, use prediction variance
                    tree_predictions = [tree.predict(features_scaled)[0] for tree in model.estimators_]
                    prediction_std = np.std(tree_predictions)
                    confidence = max(0.1, 1.0 / (1.0 + prediction_std))
                else:
                    confidence = 0.8  # Default confidence
            
            # Create prediction result
            result = PredictionResult(
                prediction_type=metadata['prediction_type'],
                value=prediction,
                confidence=confidence,
                timestamp=datetime.now(),
                model_name=model_name,
                model_version=metadata['version'],
                features_used=self.feature_columns.get(model_name, []),
                metadata={
                    'model_type': metadata['model_type'].value,
                    'feature_count': len(features)
                }
            )
            
            # Store prediction for performance tracking
            self.prediction_history.append(result)
            
            # Limit history size
            if len(self.prediction_history) > 10000:
                self.prediction_history = self.prediction_history[-5000:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction with model {model_name}: {e}")
            return None
    
    async def update_model_online(
        self,
        model_name: str,
        new_features: np.ndarray,
        new_targets: np.ndarray
    ) -> bool:
        """
        Update model with new data using online learning.
        
        Args:
            model_name: Name of the model to update
            new_features: New feature data
            new_targets: New target data
            
        Returns:
            bool: True if update successful
        """
        try:
            if not self.online_learning_enabled:
                return False
            
            if model_name not in self.models:
                self.logger.error(f"Model {model_name} not found")
                return False
            
            model = self.models[model_name]
            metadata = self.model_metadata[model_name]
            model_type = metadata['model_type']
            
            # Preprocess new data
            if model_name in self.scalers:
                new_features_scaled = self.scalers[model_name].transform(new_features)
            else:
                new_features_scaled = new_features
            
            # Online learning based on model type
            if model_type in [ModelType.LSTM, ModelType.TRANSFORMER] and TF_AVAILABLE:
                # For neural networks, use partial fit or retrain on recent data
                await self._update_neural_network_online(
                    model_name, model, new_features_scaled, new_targets
                )
                
            elif model_type == ModelType.RANDOM_FOREST:
                # Random Forest doesn't support online learning, trigger retrain if needed
                if len(new_features) >= self.retrain_threshold * metadata['training_samples']:
                    await self._trigger_model_retrain(model_name, new_features_scaled, new_targets)
                    
            elif hasattr(model, 'partial_fit'):
                # For models that support partial fit
                model.partial_fit(new_features_scaled, new_targets)
                
            else:
                # For other models, accumulate data and retrain periodically
                await self._accumulate_and_retrain(model_name, new_features_scaled, new_targets)
            
            self.logger.info(f"Model {model_name} updated with {len(new_features)} new samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating model {model_name} online: {e}")
            return False
    
    async def get_model_performance(self, model_name: str) -> Optional[ModelPerformance]:
        """Get performance metrics for a model."""
        return self.model_performance.get(model_name)
    
    async def get_feature_importance(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get feature importance for a model."""
        try:
            if model_name not in self.models:
                return None
            
            model = self.models[model_name]
            metadata = self.model_metadata[model_name]
            model_type = metadata['model_type']
            
            importance = {}
            feature_names = self.feature_columns.get(model_name, [f"feature_{i}" for i in range(metadata['feature_count'])])
            
            if model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
                importance_scores = model.feature_importances_
                importance = dict(zip(feature_names, importance_scores))
                
            elif model_type == ModelType.XGBOOST and XGB_AVAILABLE:
                importance_scores = model.feature_importances_
                importance = dict(zip(feature_names, importance_scores))
                
            elif model_type in [ModelType.LINEAR_REGRESSION, ModelType.RIDGE_REGRESSION]:
                # Use coefficient magnitudes as importance
                importance_scores = np.abs(model.coef_)
                importance = dict(zip(feature_names, importance_scores))
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance for {model_name}: {e}")
            return None
    
    def _get_scaler(self, model_type: ModelType):
        """Get appropriate scaler for model type."""
        if model_type in [ModelType.LSTM, ModelType.TRANSFORMER]:
            return MinMaxScaler(feature_range=(0, 1))
        else:
            return StandardScaler()
    
    async def _train_lstm_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]]
    ) -> Tuple[Any, Any]:
        """Train LSTM model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTM training")
        
        # Default hyperparameters
        params = {
            'lstm_units': 50,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'patience': 10
        }
        if hyperparameters:
            params.update(hyperparameters)
        
        # Reshape for LSTM (samples, timesteps, features)
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        # Build LSTM model
        model = models.Sequential([
            layers.LSTM(params['lstm_units'], return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
            layers.Dropout(params['dropout']),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        # Training callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params['patience'],
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        return model, history
    
    async def _train_transformer_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray, 
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]]
    ) -> Tuple[Any, Any]:
        """Train Transformer model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available for Transformer training")
        
        # Default hyperparameters
        params = {
            'num_heads': 8,
            'ff_dim': 32,
            'num_transformer_blocks': 2,
            'mlp_units': [128],
            'dropout': 0.1,
            'mlp_dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'patience': 10
        }
        if hyperparameters:
            params.update(hyperparameters)
        
        # Reshape for Transformer
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        # Build Transformer model
        inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
        x = inputs
        
        for _ in range(params['num_transformer_blocks']):
            # Multi-head self-attention
            attention_output = layers.MultiHeadAttention(
                num_heads=params['num_heads'],
                key_dim=X_train.shape[2],
                dropout=params['dropout']
            )(x, x)
            
            # Skip connection and layer norm
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Feed forward network
            ffn_output = layers.Dense(params['ff_dim'], activation='relu')(x)
            ffn_output = layers.Dropout(params['dropout'])(ffn_output)
            ffn_output = layers.Dense(X_train.shape[2])(ffn_output)
            
            # Skip connection and layer norm
            x = layers.Add()([x, ffn_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global average pooling and MLP head
        x = layers.GlobalAveragePooling1D()(x)
        
        for dim in params['mlp_units']:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(params['mlp_dropout'])(x)
        
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        # Training callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params['patience'],
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        return model, history
    
    async def _train_random_forest_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]]
    ) -> Any:
        """Train Random Forest model."""
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        if hyperparameters:
            params.update(hyperparameters)
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        return model
    
    async def _train_xgboost_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]]
    ) -> Any:
        """Train XGBoost model."""
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        if hyperparameters:
            params.update(hyperparameters)
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        return model
    
    async def _train_linear_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]]
    ) -> Any:
        """Train linear regression model."""
        params = hyperparameters or {}
        
        if 'alpha' in params:
            # Ridge regression
            model = Ridge(alpha=params['alpha'])
        else:
            # Linear regression
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        return model
    
    async def _train_ensemble_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]]
    ) -> Any:
        """Train ensemble model."""
        from sklearn.ensemble import VotingRegressor
        
        # Create base models
        models = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('lr', Ridge(alpha=1.0))
        ]
        
        if XGB_AVAILABLE:
            models.append(('xgb', xgb.XGBRegressor(n_estimators=50, random_state=42)))
        
        ensemble = VotingRegressor(models)
        ensemble.fit(X_train, y_train)
        
        return ensemble
    
    async def _evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        prediction_type: PredictionType
    ) -> ModelPerformance:
        """Evaluate model performance."""
        metadata = self.model_metadata[model_name]
        
        # Make predictions
        if metadata['model_type'] in [ModelType.LSTM, ModelType.TRANSFORMER]:
            y_pred = model.predict(X_val, verbose=0).flatten()
        else:
            y_pred = model.predict(X_val)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # For classification metrics (if applicable)
        accuracy = None
        precision = None
        recall = None
        f1_score = None
        
        if prediction_type in [PredictionType.DIRECTION, PredictionType.REGIME]:
            # Convert to classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as f1
            
            # Convert predictions to classes
            y_val_class = (y_val > 0).astype(int)
            y_pred_class = (y_pred > 0).astype(int)
            
            accuracy = accuracy_score(y_val_class, y_pred_class)
            precision = precision_score(y_val_class, y_pred_class, average='weighted', zero_division=0)
            recall = recall_score(y_val_class, y_pred_class, average='weighted', zero_division=0)
            f1_score = f1(y_val_class, y_pred_class, average='weighted', zero_division=0)
        
        return ModelPerformance(
            model_name=model_name,
            model_version=metadata['version'],
            prediction_type=prediction_type,
            mse=mse,
            mae=mae,
            r2_score=r2,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            validation_samples=len(X_val)
        )
    
    async def _save_model(self, model_name: str) -> None:
        """Save model to disk."""
        try:
            model_path = self.model_save_path / f"{model_name}.pkl"
            metadata_path = self.model_save_path / f"{model_name}_metadata.pkl"
            scaler_path = self.model_save_path / f"{model_name}_scaler.pkl"
            
            # Save model
            model = self.models[model_name]
            metadata = self.model_metadata[model_name]
            
            if metadata['model_type'] in [ModelType.LSTM, ModelType.TRANSFORMER]:
                # Save TensorFlow model
                tf_model_path = self.model_save_path / f"{model_name}_tf"
                model.save(tf_model_path)
            else:
                # Save scikit-learn model
                joblib.dump(model, model_path)
            
            # Save metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Save scaler
            if model_name in self.scalers:
                joblib.dump(self.scalers[model_name], scaler_path)
            
            self.logger.info(f"Model {model_name} saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {e}")
    
    async def _load_model(self, model_name: str) -> bool:
        """Load model from disk."""
        try:
            model_path = self.model_save_path / f"{model_name}.pkl"
            metadata_path = self.model_save_path / f"{model_name}_metadata.pkl"
            scaler_path = self.model_save_path / f"{model_name}_scaler.pkl"
            tf_model_path = self.model_save_path / f"{model_name}_tf"
            
            # Load metadata
            if not metadata_path.exists():
                return False
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Load model
            if metadata['model_type'] in [ModelType.LSTM, ModelType.TRANSFORMER] and tf_model_path.exists():
                model = tf.keras.models.load_model(tf_model_path)
            elif model_path.exists():
                model = joblib.load(model_path)
            else:
                return False
            
            # Load scaler
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                self.scalers[model_name] = scaler
            
            # Store loaded components
            self.models[model_name] = model
            self.model_metadata[model_name] = metadata
            self.model_status[model_name] = ModelStatus.TRAINED
            
            self.logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    async def _update_neural_network_online(
        self,
        model_name: str,
        model: Any,
        new_features: np.ndarray,
        new_targets: np.ndarray
    ) -> None:
        """Update neural network with new data."""
        # For neural networks, we can do a few epochs of training on new data
        if len(new_features.shape) == 2:
            new_features = new_features.reshape(new_features.shape[0], 1, new_features.shape[1])
        
        model.fit(
            new_features, new_targets,
            epochs=5,
            batch_size=min(32, len(new_features)),
            verbose=0
        )
    
    async def _trigger_model_retrain(
        self,
        model_name: str,
        new_features: np.ndarray,
        new_targets: np.ndarray
    ) -> None:
        """Trigger model retraining with accumulated data."""
        # This would involve collecting historical data and retraining
        # For now, just log the trigger
        self.logger.info(f"Model {model_name} retrain triggered with {len(new_features)} new samples")
    
    async def _accumulate_and_retrain(
        self,
        model_name: str,
        new_features: np.ndarray,
        new_targets: np.ndarray
    ) -> None:
        """Accumulate data and retrain periodically."""
        # Store new data for periodic retraining
        if not hasattr(self, '_accumulated_data'):
            self._accumulated_data = {}
        
        if model_name not in self._accumulated_data:
            self._accumulated_data[model_name] = {'features': [], 'targets': []}
        
        self._accumulated_data[model_name]['features'].append(new_features)
        self._accumulated_data[model_name]['targets'].append(new_targets)
        
        # Check if we should retrain
        total_samples = sum(len(f) for f in self._accumulated_data[model_name]['features'])
        if total_samples >= self.adaptation_window:
            # Trigger retraining
            await self._trigger_model_retrain(model_name, new_features, new_targets)
            # Clear accumulated data
            self._accumulated_data[model_name] = {'features': [], 'targets': []}


# Utility functions for ML engine integration

async def create_ml_engine(config: ConfigurationManager) -> MLEngine:
    """
    Create and initialize an ML engine.
    
    Args:
        config: Configuration manager
        
    Returns:
        MLEngine: Initialized ML engine instance
    """
    return MLEngine(config)