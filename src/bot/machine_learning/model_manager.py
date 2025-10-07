"""
Advanced Model Manager for ML model lifecycle management.
Handles model training, versioning, deployment, and performance monitoring.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import json
import os
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import logging
from pathlib import Path
import hashlib
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ARCHIVED = "archived"

class ModelType(Enum):
    """Types of models supported."""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LINEAR_REGRESSION = "linear_regression"
    SVM = "svm"
    ENSEMBLE = "ensemble"

@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    training_data_hash: str
    feature_columns: List[str]
    target_column: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    deployment_config: Dict[str, Any]
    model_path: str
    scaler_path: Optional[str]
    description: str
    tags: List[str]

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    validation_split: float = 0.2
    test_split: float = 0.2
    cross_validation_folds: int = 5
    early_stopping: bool = True
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    regularization: Dict[str, float] = None
    feature_selection: bool = True
    scaling_method: str = 'standard'  # standard, minmax, robust

@dataclass
class ModelPerformance:
    """Model performance tracking."""
    model_id: str
    timestamp: datetime
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    mse: Optional[float]
    mae: Optional[float]
    r2_score: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    win_rate: Optional[float]
    profit_factor: Optional[float]
    custom_metrics: Dict[str, float]

class ModelManager:
    """Advanced model lifecycle management system."""
    
    def __init__(self, models_dir: str = "models"):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Setup directories
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.models_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.scalers_dir = self.models_dir / "scalers"
        self.scalers_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.model_registry = {}
        self.deployed_models = {}
        
        # Performance tracking
        self.performance_history = {}
        
        # Training queue
        self.training_queue = []
        self.training_in_progress = False
        
        # Load existing models
        self._load_model_registry()
        
        self.logger.info(f"ModelManager initialized with models directory: {models_dir}")
    
    async def train_model(self, 
                         model_name: str,
                         model_type: ModelType,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         feature_columns: List[str],
                         target_column: str,
                         config: TrainingConfig,
                         description: str = "",
                         tags: List[str] = None) -> str:
        """Train a new model with the given configuration."""
        try:
            self.logger.info(f"Starting training for model: {model_name}")
            
            # Generate model ID
            data_hash = self._calculate_data_hash(X_train, y_train)
            model_id = self._generate_model_id(model_name, model_type, data_hash)
            
            # Prepare data
            X_processed, y_processed, scaler = await self._prepare_training_data(
                X_train, y_train, config
            )
            
            # Split data
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_processed, y_processed, 
                test_size=config.validation_split,
                random_state=42
            )
            
            # Train model based on type
            model, training_history = await self._train_model_by_type(
                model_type, X_train_split, y_train_split, X_val, y_val, config
            )
            
            if model is None:
                raise ValueError(f"Failed to train {model_type.value} model")
            
            # Evaluate model
            performance_metrics = await self._evaluate_model(
                model, X_val, y_val, model_type
            )
            
            # Save model and metadata
            model_path = await self._save_model(model, model_id, model_type)
            scaler_path = await self._save_scaler(scaler, model_id) if scaler else None
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                version="1.0.0",
                status=ModelStatus.TRAINED,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                training_data_hash=data_hash,
                feature_columns=feature_columns,
                target_column=target_column,
                hyperparameters=config.hyperparameters,
                performance_metrics=performance_metrics,
                deployment_config={},
                model_path=str(model_path),
                scaler_path=str(scaler_path) if scaler_path else None,
                description=description,
                tags=tags or []
            )
            
            # Register model
            self.model_registry[model_id] = metadata
            await self._save_metadata(metadata)
            
            # Track performance
            perf_record = ModelPerformance(
                model_id=model_id,
                timestamp=datetime.now(),
                **performance_metrics,
                custom_metrics={}
            )
            
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            self.performance_history[model_id].append(perf_record)
            
            self.logger.info(f"Model {model_name} trained successfully with ID: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Model training failed for {model_name}: {e}")
            raise
    
    async def deploy_model(self, model_id: str, deployment_config: Dict[str, Any] = None) -> bool:
        """Deploy a trained model for predictions."""
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.model_registry[model_id]
            
            if metadata.status != ModelStatus.TRAINED:
                raise ValueError(f"Model {model_id} is not in trained status")
            
            # Load model
            model = await self._load_model(metadata.model_path, metadata.model_type)
            if model is None:
                raise ValueError(f"Failed to load model from {metadata.model_path}")
            
            # Load scaler if exists
            scaler = None
            if metadata.scaler_path:
                scaler = await self._load_scaler(metadata.scaler_path)
            
            # Update deployment config
            if deployment_config:
                metadata.deployment_config.update(deployment_config)
            
            # Register as deployed
            self.deployed_models[model_id] = {
                'model': model,
                'scaler': scaler,
                'metadata': metadata,
                'deployed_at': datetime.now(),
                'prediction_count': 0,
                'last_prediction': None
            }
            
            # Update status
            metadata.status = ModelStatus.DEPLOYED
            metadata.updated_at = datetime.now()
            await self._save_metadata(metadata)
            
            self.logger.info(f"Model {model_id} deployed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model deployment failed for {model_id}: {e}")
            return False
    
    async def predict(self, model_id: str, features: np.ndarray) -> Optional[Union[float, np.ndarray]]:
        """Make prediction using deployed model."""
        try:
            if model_id not in self.deployed_models:
                raise ValueError(f"Model {model_id} is not deployed")
            
            deployment_info = self.deployed_models[model_id]
            model = deployment_info['model']
            scaler = deployment_info['scaler']
            metadata = deployment_info['metadata']
            
            # Preprocess features
            if scaler:
                features_scaled = scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            
            # Make prediction based on model type
            if metadata.model_type == ModelType.LSTM and HAS_TF:
                prediction = model.predict(features_scaled)[0][0]
            elif metadata.model_type == ModelType.TRANSFORMER and HAS_TF:
                prediction = model.predict(features_scaled)[0][0]
            elif metadata.model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, 
                                       ModelType.LINEAR_REGRESSION, ModelType.SVM]:
                prediction = model.predict(features_scaled)[0]
            else:
                raise ValueError(f"Unsupported model type: {metadata.model_type}")
            
            # Update prediction count
            deployment_info['prediction_count'] += 1
            deployment_info['last_prediction'] = datetime.now()
            
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"Prediction failed for model {model_id}: {e}")
            return None
    
    async def retrain_model(self, model_id: str, 
                          X_new: np.ndarray, y_new: np.ndarray,
                          config: TrainingConfig = None) -> str:
        """Retrain an existing model with new data."""
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found")
            
            metadata = self.model_registry[model_id]
            
            # Create new version
            new_version = self._increment_version(metadata.version)
            new_model_id = f"{model_id}_v{new_version}"
            
            # Use existing config if not provided
            if config is None:
                config = TrainingConfig(
                    model_type=metadata.model_type,
                    hyperparameters=metadata.hyperparameters
                )
            
            # Train new version
            new_model_id = await self.train_model(
                model_name=metadata.model_name,
                model_type=metadata.model_type,
                X_train=X_new,
                y_train=y_new,
                feature_columns=metadata.feature_columns,
                target_column=metadata.target_column,
                config=config,
                description=f"Retrained version of {metadata.model_name}",
                tags=metadata.tags + ["retrained"]
            )
            
            # Update version in metadata
            self.model_registry[new_model_id].version = new_version
            
            self.logger.info(f"Model retrained successfully: {new_model_id}")
            return new_model_id
            
        except Exception as e:
            self.logger.error(f"Model retraining failed for {model_id}: {e}")
            raise
    
    async def archive_model(self, model_id: str):
        """Archive an old model."""
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found")
            
            metadata = self.model_registry[model_id]
            metadata.status = ModelStatus.ARCHIVED
            metadata.updated_at = datetime.now()
            
            # Remove from deployed models
            if model_id in self.deployed_models:
                del self.deployed_models[model_id]
            
            await self._save_metadata(metadata)
            
            self.logger.info(f"Model {model_id} archived")
            
        except Exception as e:
            self.logger.error(f"Failed to archive model {model_id}: {e}")
    
    async def get_model_performance(self, model_id: str) -> List[ModelPerformance]:
        """Get performance history for a model."""
        return self.performance_history.get(model_id, [])
    
    def list_models(self, status: ModelStatus = None, 
                   model_type: ModelType = None) -> List[ModelMetadata]:
        """List models with optional filtering."""
        models = list(self.model_registry.values())
        
        if status:
            models = [m for m in models if m.status == status]
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        return models
    
    def get_deployed_models(self) -> Dict[str, Dict]:
        """Get information about deployed models."""
        return {
            model_id: {
                'metadata': info['metadata'],
                'deployed_at': info['deployed_at'],
                'prediction_count': info['prediction_count'],
                'last_prediction': info['last_prediction']
            }
            for model_id, info in self.deployed_models.items()
        }
    
    async def _prepare_training_data(self, X: np.ndarray, y: np.ndarray, 
                                   config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Prepare data for training with scaling and preprocessing."""
        scaler = None
        
        # Apply scaling
        if config.scaling_method == 'standard':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        elif config.scaling_method == 'minmax':
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        return X_scaled, y, scaler
    
    async def _train_model_by_type(self, model_type: ModelType,
                                 X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 config: TrainingConfig) -> Tuple[Any, Dict]:
        """Train model based on its type."""
        try:
            if model_type == ModelType.RANDOM_FOREST:
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                
                # Determine if classification or regression
                is_classification = len(np.unique(y_train)) < 10 and np.issubdtype(y_train.dtype, np.integer)
                
                if is_classification:
                    model = RandomForestClassifier(**config.hyperparameters)
                else:
                    model = RandomForestRegressor(**config.hyperparameters)
                
                model.fit(X_train, y_train)
                return model, {}
            
            elif model_type == ModelType.XGBOOST:
                try:
                    import xgboost as xgb
                    model = xgb.XGBRegressor(**config.hyperparameters)
                    model.fit(X_train, y_train)
                    return model, {}
                except ImportError:
                    self.logger.error("XGBoost not installed")
                    return None, {}
            
            elif model_type == ModelType.LINEAR_REGRESSION:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression(**config.hyperparameters)
                model.fit(X_train, y_train)
                return model, {}
            
            elif model_type == ModelType.SVM:
                from sklearn.svm import SVR
                model = SVR(**config.hyperparameters)
                model.fit(X_train, y_train)
                return model, {}
            
            elif model_type == ModelType.LSTM and HAS_TF:
                return await self._train_lstm_model(X_train, y_train, X_val, y_val, config)
            
            elif model_type == ModelType.TRANSFORMER and HAS_TF:
                return await self._train_transformer_model(X_train, y_train, X_val, y_val, config)
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Training failed for {model_type}: {e}")
            return None, {}
    
    async def _train_lstm_model(self, X_train, y_train, X_val, y_val, config):
        """Train LSTM model using TensorFlow."""
        if not HAS_TF:
            raise ImportError("TensorFlow not available")
        
        # Reshape for LSTM (samples, timesteps, features)
        X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_reshaped = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        
        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(config.hyperparameters.get('units', 50),
                               return_sequences=config.hyperparameters.get('return_sequences', False)),
            tf.keras.layers.Dropout(config.hyperparameters.get('dropout', 0.2)),
            tf.keras.layers.Dense(config.hyperparameters.get('dense_units', 25)),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Training with early stopping
        callbacks = []
        if config.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            )
        
        history = model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=config.max_epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return model, history.history
    
    async def _train_transformer_model(self, X_train, y_train, X_val, y_val, config):
        """Train Transformer model using TensorFlow."""
        if not HAS_TF:
            raise ImportError("TensorFlow not available")
        
        # Simplified transformer for time series
        seq_len = 1
        d_model = config.hyperparameters.get('d_model', 64)
        num_heads = config.hyperparameters.get('num_heads', 4)
        
        inputs = tf.keras.Input(shape=(seq_len, X_train.shape[1]))
        
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )(inputs, inputs)
        
        # Add & Norm
        x = tf.keras.layers.Add()([inputs, attention])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Feed forward
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 2, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        ffn_output = ffn(x)
        x = tf.keras.layers.Add()([x, ffn_output])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Global average pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Reshape data
        X_train_reshaped = X_train.reshape((X_train.shape[0], seq_len, X_train.shape[1]))
        X_val_reshaped = X_val.reshape((X_val.shape[0], seq_len, X_val.shape[1]))
        
        history = model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=config.max_epochs,
            batch_size=config.batch_size,
            verbose=0
        )
        
        return model, history.history
    
    async def _evaluate_model(self, model, X_test, y_test, model_type):
        """Evaluate trained model performance."""
        try:
            # Make predictions
            if model_type in [ModelType.LSTM, ModelType.TRANSFORMER] and HAS_TF:
                if len(X_test.shape) == 2:
                    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                y_pred = model.predict(X_test).flatten()
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # R² score
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'r2_score': float(r2),
                'rmse': float(np.sqrt(mse))
            }
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return {}
    
    async def _save_model(self, model, model_id: str, model_type: ModelType) -> Path:
        """Save trained model to disk."""
        model_path = self.models_dir / f"{model_id}.pkl"
        
        try:
            if model_type in [ModelType.LSTM, ModelType.TRANSFORMER] and HAS_TF:
                # Save TensorFlow model
                tf_path = self.models_dir / f"{model_id}_tf"
                model.save(tf_path)
                return tf_path
            else:
                # Save sklearn model
                joblib.dump(model, model_path)
                return model_path
                
        except Exception as e:
            self.logger.error(f"Failed to save model {model_id}: {e}")
            raise
    
    async def _load_model(self, model_path: str, model_type: ModelType):
        """Load model from disk."""
        try:
            path = Path(model_path)
            
            if model_type in [ModelType.LSTM, ModelType.TRANSFORMER] and HAS_TF:
                return tf.keras.models.load_model(path)
            else:
                return joblib.load(path)
                
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            return None
    
    async def _save_scaler(self, scaler, model_id: str) -> Path:
        """Save scaler to disk."""
        scaler_path = self.scalers_dir / f"{model_id}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        return scaler_path
    
    async def _load_scaler(self, scaler_path: str):
        """Load scaler from disk."""
        try:
            return joblib.load(scaler_path)
        except Exception as e:
            self.logger.error(f"Failed to load scaler from {scaler_path}: {e}")
            return None
    
    async def _save_metadata(self, metadata: ModelMetadata):
        """Save model metadata to disk."""
        metadata_path = self.metadata_dir / f"{metadata.model_id}.json"
        
        # Convert to dict and handle datetime serialization
        metadata_dict = asdict(metadata)
        metadata_dict['created_at'] = metadata.created_at.isoformat()
        metadata_dict['updated_at'] = metadata.updated_at.isoformat()
        metadata_dict['model_type'] = metadata.model_type.value
        metadata_dict['status'] = metadata.status.value
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def _load_model_registry(self):
        """Load existing model registry from disk."""
        try:
            for metadata_file in self.metadata_dir.glob("*.json"):
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                # Convert back to objects
                metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
                metadata_dict['model_type'] = ModelType(metadata_dict['model_type'])
                metadata_dict['status'] = ModelStatus(metadata_dict['status'])
                
                metadata = ModelMetadata(**metadata_dict)
                self.model_registry[metadata.model_id] = metadata
            
            self.logger.info(f"Loaded {len(self.model_registry)} models from registry")
            
        except Exception as e:
            self.logger.error(f"Failed to load model registry: {e}")
    
    def _generate_model_id(self, model_name: str, model_type: ModelType, data_hash: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{model_type.value}_{timestamp}_{data_hash[:8]}"
    
    def _calculate_data_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        """Calculate hash of training data."""
        data_str = f"{X.tobytes()}{y.tobytes()}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _increment_version(self, current_version: str) -> str:
        """Increment model version."""
        try:
            major, minor, patch = map(int, current_version.split('.'))
            return f"{major}.{minor}.{patch + 1}"
        except:
            return "1.0.1"
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of model registry."""
        status_counts = {}
        type_counts = {}
        
        for metadata in self.model_registry.values():
            status_counts[metadata.status.value] = status_counts.get(metadata.status.value, 0) + 1
            type_counts[metadata.model_type.value] = type_counts.get(metadata.model_type.value, 0) + 1
        
        return {
            'total_models': len(self.model_registry),
            'deployed_models': len(self.deployed_models),
            'status_distribution': status_counts,
            'type_distribution': type_counts,
            'models_directory': str(self.models_dir),
            'last_updated': max([m.updated_at for m in self.model_registry.values()]) if self.model_registry else None
        }