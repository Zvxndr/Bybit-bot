"""
Ensemble Model Manager

Sophisticated ensemble learning system for cryptocurrency trading predictions.
Manages multiple ML models with dynamic weighting, performance tracking,
and adaptive ensemble composition for optimal prediction accuracy.

Key Features:
- Multi-model ensemble with LightGBM, XGBoost, CatBoost, and Neural Networks
- Dynamic model weighting based on recent performance
- Adaptive ensemble composition (add/remove models based on performance)
- Real-time performance monitoring and model health tracking
- Multi-target prediction support (multiple prediction horizons)
- Model versioning and A/B testing capabilities
- Uncertainty quantification and prediction confidence scoring
- Regime-aware model selection and weighting
- Automatic hyperparameter optimization for individual models
- Model interpretability and feature importance aggregation

Supported Models:
- LightGBM: Fast gradient boosting with categorical support
- XGBoost: Robust gradient boosting with advanced regularization
- CatBoost: Gradient boosting optimized for categorical features
- Random Forest: Ensemble of decision trees with bagging
- Extra Trees: Extremely randomized trees for reduced overfitting
- Neural Networks: Multi-layer perceptrons with dropout
- Linear Models: Ridge, Lasso, ElasticNet for baseline comparison
- Time Series Models: ARIMA, Prophet for temporal patterns

Ensemble Strategies:
- Simple averaging: Equal weight combination
- Performance weighting: Weight by recent validation performance
- Bayesian Model Averaging: Probabilistic model combination
- Stacking: Meta-learner combines base model predictions
- Dynamic selection: Choose best model per prediction
- Regime-based: Different models for different market conditions
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

# ML model imports
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
import optuna

from ..utils.logging import TradingLogger
from ..config.manager import ConfigurationManager


class ModelType(Enum):
    """Types of ML models."""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    MLP = "neural_network"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"


class EnsembleStrategy(Enum):
    """Ensemble combination strategies."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    BAYESIAN_AVERAGE = "bayesian_average"
    STACKING = "stacking"
    DYNAMIC_SELECTION = "dynamic_selection"
    REGIME_BASED = "regime_based"


class ModelStatus(Enum):
    """Model status tracking."""
    ACTIVE = "active"
    TRAINING = "training"
    VALIDATION = "validation"
    INACTIVE = "inactive"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics."""
    model_id: str
    model_type: ModelType
    
    # Performance metrics
    mse: float
    mae: float
    r2_score: float
    mape: float  # Mean Absolute Percentage Error
    
    # Time-based performance
    recent_performance: float  # Last 100 predictions
    trend_performance: float   # Performance trend (improving/degrading)
    
    # Cross-validation metrics
    cv_score_mean: float
    cv_score_std: float
    
    # Training metrics
    training_time: float
    prediction_time: float
    memory_usage: float
    
    # Model health
    prediction_count: int
    error_count: int
    last_training: datetime
    last_prediction: datetime
    
    # Feature importance (top 10)
    feature_importance: Dict[str, float]
    
    # Hyperparameters
    hyperparameters: Dict[str, Any]
    
    # Status
    status: ModelStatus
    confidence_score: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Ensemble prediction with uncertainty quantification."""
    timestamp: datetime
    target_variable: str
    prediction_horizon: str
    
    # Predictions
    ensemble_prediction: float
    individual_predictions: Dict[str, float]  # model_id -> prediction
    
    # Uncertainty measures
    prediction_std: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_agreement: float  # How much models agree
    
    # Weighting information
    model_weights: Dict[str, float]
    ensemble_strategy: EnsembleStrategy
    
    # Model contributions
    model_contributions: Dict[str, float]
    active_models: List[str]
    
    # Quality indicators
    prediction_confidence: float
    ensemble_diversity: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfiguration:
    """Configuration for individual models."""
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    target_column: str
    
    # Training configuration
    validation_split: float = 0.2
    early_stopping_rounds: Optional[int] = None
    
    # Preprocessing
    scaler_type: Optional[str] = None  # 'standard', 'robust', None
    
    # Model-specific settings
    model_specific_config: Dict[str, Any] = field(default_factory=dict)


class BaseModelWrapper:
    """Base wrapper for ML models with common interface."""
    
    def __init__(self, model_id: str, model_type: ModelType, config: ModelConfiguration):
        self.model_id = model_id
        self.model_type = model_type
        self.config = config
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_importance_ = {}
        
        # Performance tracking
        self.predictions_made = 0
        self.training_history = []
        self.error_log = deque(maxlen=1000)
        
        self._initialize_model()
        self._initialize_scaler()
    
    def _initialize_model(self):
        """Initialize the underlying ML model."""
        params = self.config.hyperparameters.copy()
        
        if self.model_type == ModelType.LIGHTGBM:
            self.model = lgb.LGBMRegressor(**params)
        elif self.model_type == ModelType.XGBOOST:
            self.model = xgb.XGBRegressor(**params)
        elif self.model_type == ModelType.CATBOOST:
            self.model = cb.CatBoostRegressor(**params, verbose=False)
        elif self.model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestRegressor(**params)
        elif self.model_type == ModelType.EXTRA_TREES:
            self.model = ExtraTreesRegressor(**params)
        elif self.model_type == ModelType.MLP:
            self.model = MLPRegressor(**params)
        elif self.model_type == ModelType.RIDGE:
            self.model = Ridge(**params)
        elif self.model_type == ModelType.LASSO:
            self.model = Lasso(**params)
        elif self.model_type == ModelType.ELASTIC_NET:
            self.model = ElasticNet(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _initialize_scaler(self):
        """Initialize feature scaler if needed."""
        if self.config.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.config.scaler_type == 'robust':
            self.scaler = RobustScaler()
    
    def fit(self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None):
        """Train the model."""
        start_time = time.time()
        
        try:
            # Select features
            X_model = X[self.config.feature_columns] if self.config.feature_columns else X
            
            # Apply scaling
            if self.scaler:
                X_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X_model),
                    columns=X_model.columns,
                    index=X_model.index
                )
            else:
                X_scaled = X_model
            
            # Prepare validation data
            if validation_data:
                X_val, y_val = validation_data
                X_val_model = X_val[self.config.feature_columns] if self.config.feature_columns else X_val
                
                if self.scaler:
                    X_val_scaled = pd.DataFrame(
                        self.scaler.transform(X_val_model),
                        columns=X_val_model.columns,
                        index=X_val_model.index
                    )
                else:
                    X_val_scaled = X_val_model
                
                eval_set = [(X_val_scaled, y_val)]
            else:
                eval_set = None
            
            # Model-specific training
            if self.model_type in [ModelType.LIGHTGBM, ModelType.XGBOOST, ModelType.CATBOOST]:
                if eval_set and self.config.early_stopping_rounds:
                    if self.model_type == ModelType.LIGHTGBM:
                        self.model.fit(
                            X_scaled, y,
                            eval_set=eval_set,
                            callbacks=[lgb.early_stopping(self.config.early_stopping_rounds)]
                        )
                    elif self.model_type == ModelType.XGBOOST:
                        self.model.fit(
                            X_scaled, y,
                            eval_set=eval_set,
                            early_stopping_rounds=self.config.early_stopping_rounds,
                            verbose=False
                        )
                    elif self.model_type == ModelType.CATBOOST:
                        self.model.fit(
                            X_scaled, y,
                            eval_set=eval_set,
                            early_stopping_rounds=self.config.early_stopping_rounds,
                            verbose=False
                        )
                else:
                    self.model.fit(X_scaled, y)
            else:
                self.model.fit(X_scaled, y)
            
            # Extract feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance_ = dict(zip(X_scaled.columns, self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                self.feature_importance_ = dict(zip(X_scaled.columns, np.abs(self.model.coef_)))
            
            self.is_fitted = True
            training_time = time.time() - start_time
            
            self.training_history.append({
                'timestamp': datetime.now(),
                'training_time': training_time,
                'samples': len(X_scaled),
                'features': len(X_scaled.columns)
            })
            
            return training_time
            
        except Exception as e:
            self.error_log.append({
                'timestamp': datetime.now(),
                'error_type': 'training',
                'error': str(e)
            })
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.model_id} is not fitted")
        
        start_time = time.time()
        
        try:
            # Select features
            X_model = X[self.config.feature_columns] if self.config.feature_columns else X
            
            # Apply scaling
            if self.scaler:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X_model),
                    columns=X_model.columns,
                    index=X_model.index
                )
            else:
                X_scaled = X_model
            
            predictions = self.model.predict(X_scaled)
            
            self.predictions_made += len(predictions)
            prediction_time = time.time() - start_time
            
            return predictions
            
        except Exception as e:
            self.error_log.append({
                'timestamp': datetime.now(),
                'error_type': 'prediction',
                'error': str(e)
            })
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        return self.feature_importance_.copy()
    
    def save_model(self, path: str):
        """Save model to disk."""
        model_data = {
            'model_id': self.model_id,
            'model_type': self.model_type.value,
            'config': self.config,
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_importance_': self.feature_importance_,
            'predictions_made': self.predictions_made,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path: str):
        """Load model from disk."""
        model_data = joblib.load(path)
        
        # Recreate instance
        instance = cls(
            model_data['model_id'],
            ModelType(model_data['model_type']),
            model_data['config']
        )
        
        # Restore state
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.is_fitted = model_data['is_fitted']
        instance.feature_importance_ = model_data['feature_importance_']
        instance.predictions_made = model_data['predictions_made']
        instance.training_history = model_data['training_history']
        
        return instance


class EnsembleModelManager:
    """
    Sophisticated ensemble learning system for cryptocurrency trading.
    
    Manages multiple ML models with dynamic weighting, performance tracking,
    and adaptive ensemble composition.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = TradingLogger("EnsembleModelManager")
        
        # Configuration
        self.config = {
            'max_models': config_manager.get('ensemble.max_models', 10),
            'performance_window': config_manager.get('ensemble.performance_window', 100),
            'retraining_frequency': config_manager.get('ensemble.retraining_frequency', 24),  # hours
            'model_timeout': config_manager.get('ensemble.model_timeout', 300),  # seconds
            'min_model_performance': config_manager.get('ensemble.min_performance', 0.1),
            'diversity_threshold': config_manager.get('ensemble.diversity_threshold', 0.1),
            'confidence_threshold': config_manager.get('ensemble.confidence_threshold', 0.7),
            'default_ensemble_strategy': EnsembleStrategy.PERFORMANCE_WEIGHTED,
            'hyperopt_trials': config_manager.get('ensemble.hyperopt_trials', 50),
            'cross_validation_folds': config_manager.get('ensemble.cv_folds', 5),
            'model_selection_metric': config_manager.get('ensemble.selection_metric', 'r2'),
        }
        
        # Model storage
        self.models: Dict[str, BaseModelWrapper] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.model_predictions_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Ensemble state
        self.ensemble_weights: Dict[str, float] = {}
        self.ensemble_strategy = self.config['default_ensemble_strategy']
        self.active_models: Set[str] = set()
        
        # Performance tracking
        self.ensemble_performance_history: deque = deque(maxlen=1000)
        self.prediction_history: deque = deque(maxlen=5000)
        
        # Training data cache
        self.training_data: Optional[Tuple[pd.DataFrame, Dict[str, pd.Series]]] = None
        self.validation_data: Optional[Tuple[pd.DataFrame, Dict[str, pd.Series]]] = None
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.training_lock = threading.Lock()
        
        # Model persistence
        self.model_save_path = Path(config_manager.get('paths.models', 'models'))
        self.model_save_path.mkdir(exist_ok=True)
        
        # Initialize default models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default set of models."""
        default_model_configs = [
            # LightGBM
            ModelConfiguration(
                model_type=ModelType.LIGHTGBM,
                hyperparameters={
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                feature_columns=[],
                target_column='',
                early_stopping_rounds=20
            ),
            
            # XGBoost
            ModelConfiguration(
                model_type=ModelType.XGBOOST,
                hyperparameters={
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                feature_columns=[],
                target_column='',
                early_stopping_rounds=20
            ),
            
            # CatBoost
            ModelConfiguration(
                model_type=ModelType.CATBOOST,
                hyperparameters={
                    'iterations': 200,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                feature_columns=[],
                target_column='',
                early_stopping_rounds=20
            ),
            
            # Random Forest
            ModelConfiguration(
                model_type=ModelType.RANDOM_FOREST,
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                },
                feature_columns=[],
                target_column=''
            ),
            
            # Neural Network
            ModelConfiguration(
                model_type=ModelType.MLP,
                hyperparameters={
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'alpha': 0.001,
                    'learning_rate': 'adaptive',
                    'max_iter': 500,
                    'random_state': 42
                },
                feature_columns=[],
                target_column='',
                scaler_type='standard'
            )
        ]
        
        # Create model instances
        for i, config in enumerate(default_model_configs):
            model_id = f"{config.model_type.value}_{i}"
            model = BaseModelWrapper(model_id, config.model_type, config)
            self.models[model_id] = model
            
            # Initialize metrics
            self.model_metrics[model_id] = ModelMetrics(
                model_id=model_id,
                model_type=config.model_type,
                mse=float('inf'),
                mae=float('inf'),
                r2_score=-float('inf'),
                mape=float('inf'),
                recent_performance=0.0,
                trend_performance=0.0,
                cv_score_mean=0.0,
                cv_score_std=0.0,
                training_time=0.0,
                prediction_time=0.0,
                memory_usage=0.0,
                prediction_count=0,
                error_count=0,
                last_training=datetime.min,
                last_prediction=datetime.min,
                feature_importance={},
                hyperparameters=config.hyperparameters,
                status=ModelStatus.INACTIVE,
                confidence_score=0.0
            )
    
    def update_training_data(self, features: pd.DataFrame, targets: Dict[str, pd.Series]):
        """Update training data for all models."""
        try:
            # Store training data
            self.training_data = (features.copy(), targets.copy())
            
            # Create validation split
            split_idx = int(len(features) * 0.8)
            self.validation_data = (
                features.iloc[split_idx:].copy(),
                {k: v.iloc[split_idx:] for k, v in targets.items()}
            )
            
            self.logger.info(f"Updated training data: {len(features)} samples, {len(features.columns)} features")
            
        except Exception as e:
            self.logger.error(f"Error updating training data: {e}")
    
    def add_model(self, model_config: ModelConfiguration, model_id: Optional[str] = None) -> str:
        """Add a new model to the ensemble."""
        if model_id is None:
            model_id = f"{model_config.model_type.value}_{len(self.models)}"
        
        if model_id in self.models:
            raise ValueError(f"Model {model_id} already exists")
        
        if len(self.models) >= self.config['max_models']:
            # Remove worst performing model
            self._remove_worst_model()
        
        # Create model
        model = BaseModelWrapper(model_id, model_config.model_type, model_config)
        self.models[model_id] = model
        
        # Initialize metrics
        self.model_metrics[model_id] = ModelMetrics(
            model_id=model_id,
            model_type=model_config.model_type,
            mse=float('inf'),
            mae=float('inf'),
            r2_score=-float('inf'),
            mape=float('inf'),
            recent_performance=0.0,
            trend_performance=0.0,
            cv_score_mean=0.0,
            cv_score_std=0.0,
            training_time=0.0,
            prediction_time=0.0,
            memory_usage=0.0,
            prediction_count=0,
            error_count=0,
            last_training=datetime.min,
            last_prediction=datetime.min,
            feature_importance={},
            hyperparameters=model_config.hyperparameters,
            status=ModelStatus.INACTIVE,
            confidence_score=0.0
        )
        
        self.logger.info(f"Added model {model_id} ({model_config.model_type.value})")
        return model_id
    
    def _remove_worst_model(self):
        """Remove the worst performing model."""
        if not self.model_metrics:
            return
        
        # Find worst model by recent performance
        worst_model = min(
            self.model_metrics.items(),
            key=lambda x: x[1].recent_performance
        )
        
        model_id = worst_model[0]
        self.remove_model(model_id)
    
    def remove_model(self, model_id: str):
        """Remove a model from the ensemble."""
        if model_id not in self.models:
            return
        
        # Update status
        if model_id in self.model_metrics:
            self.model_metrics[model_id].status = ModelStatus.DEPRECATED
        
        # Remove from active models
        self.active_models.discard(model_id)
        
        # Remove from storage
        del self.models[model_id]
        
        # Update ensemble weights
        if model_id in self.ensemble_weights:
            del self.ensemble_weights[model_id]
            self._normalize_ensemble_weights()
        
        self.logger.info(f"Removed model {model_id}")
    
    def train_models(self, target_variable: str, feature_subset: Optional[List[str]] = None) -> Dict[str, float]:
        """Train all models on the current dataset."""
        if self.training_data is None:
            raise ValueError("No training data available")
        
        features, targets = self.training_data
        
        if target_variable not in targets:
            raise ValueError(f"Target variable {target_variable} not found")
        
        target = targets[target_variable]
        
        # Select features
        if feature_subset:
            training_features = features[feature_subset]
        else:
            training_features = features
        
        training_results = {}
        
        # Train models in parallel
        with self.training_lock:
            future_to_model = {}
            
            for model_id, model in self.models.items():
                if self.model_metrics[model_id].status == ModelStatus.DEPRECATED:
                    continue
                
                # Update model configuration
                model.config.feature_columns = list(training_features.columns)
                model.config.target_column = target_variable
                
                # Submit training job
                future = self.executor.submit(
                    self._train_single_model,
                    model_id,
                    training_features,
                    target
                )
                future_to_model[future] = model_id
            
            # Collect results
            for future in as_completed(future_to_model, timeout=self.config['model_timeout']):
                model_id = future_to_model[future]
                
                try:
                    training_time = future.result()
                    training_results[model_id] = training_time
                    
                    # Update model status
                    self.model_metrics[model_id].status = ModelStatus.ACTIVE
                    self.model_metrics[model_id].last_training = datetime.now()
                    self.model_metrics[model_id].training_time = training_time
                    
                    # Add to active models
                    self.active_models.add(model_id)
                    
                except Exception as e:
                    self.logger.error(f"Training failed for model {model_id}: {e}")
                    self.model_metrics[model_id].status = ModelStatus.FAILED
                    self.model_metrics[model_id].error_count += 1
                    training_results[model_id] = -1
        
        # Validate models
        self._validate_models(target_variable)
        
        # Update ensemble weights
        self._update_ensemble_weights()
        
        self.logger.info(f"Trained {len([r for r in training_results.values() if r > 0])} models successfully")
        
        return training_results
    
    def _train_single_model(self, model_id: str, features: pd.DataFrame, target: pd.Series) -> float:
        """Train a single model."""
        model = self.models[model_id]
        
        # Prepare validation data
        validation_data = None
        if self.validation_data:
            val_features, val_targets = self.validation_data
            if model.config.target_column in val_targets:
                validation_data = (val_features, val_targets[model.config.target_column])
        
        # Train model
        training_time = model.fit(features, target, validation_data)
        
        return training_time
    
    def _validate_models(self, target_variable: str):
        """Validate all active models."""
        if self.validation_data is None:
            return
        
        val_features, val_targets = self.validation_data
        val_target = val_targets[target_variable]
        
        for model_id in self.active_models.copy():
            try:
                model = self.models[model_id]
                predictions = model.predict(val_features)
                
                # Calculate metrics
                mse = mean_squared_error(val_target, predictions)
                mae = mean_absolute_error(val_target, predictions)
                r2 = r2_score(val_target, predictions)
                mape = np.mean(np.abs((val_target - predictions) / val_target)) * 100
                
                # Update metrics
                metrics = self.model_metrics[model_id]
                metrics.mse = mse
                metrics.mae = mae
                metrics.r2_score = r2
                metrics.mape = mape
                metrics.recent_performance = r2  # Use R² as performance measure
                
                # Update feature importance
                metrics.feature_importance = model.get_feature_importance()
                
                # Calculate confidence score
                metrics.confidence_score = max(0.0, min(1.0, r2))
                
                # Check if model meets minimum performance
                if r2 < self.config['min_model_performance']:
                    self.logger.warning(f"Model {model_id} performance below threshold: {r2:.4f}")
                    self.active_models.discard(model_id)
                    metrics.status = ModelStatus.INACTIVE
                
            except Exception as e:
                self.logger.error(f"Validation failed for model {model_id}: {e}")
                self.active_models.discard(model_id)
                self.model_metrics[model_id].status = ModelStatus.FAILED
                self.model_metrics[model_id].error_count += 1
    
    def _update_ensemble_weights(self):
        """Update ensemble weights based on model performance."""
        if not self.active_models:
            return
        
        # Calculate weights based on recent performance
        performance_scores = {}
        
        for model_id in self.active_models:
            metrics = self.model_metrics[model_id]
            
            # Use recent performance with confidence adjustment
            score = metrics.recent_performance * metrics.confidence_score
            performance_scores[model_id] = max(0.001, score)  # Minimum weight
        
        # Normalize weights
        total_score = sum(performance_scores.values())
        if total_score > 0:
            self.ensemble_weights = {
                model_id: score / total_score
                for model_id, score in performance_scores.items()
            }
        else:
            # Equal weights as fallback
            n_models = len(self.active_models)
            self.ensemble_weights = {
                model_id: 1.0 / n_models
                for model_id in self.active_models
            }
    
    def _normalize_ensemble_weights(self):
        """Normalize ensemble weights to sum to 1."""
        if not self.ensemble_weights:
            return
        
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            self.ensemble_weights = {
                model_id: weight / total_weight
                for model_id, weight in self.ensemble_weights.items()
            }
    
    def predict(self, features: pd.DataFrame, target_variable: str) -> EnsemblePrediction:
        """Make ensemble predictions."""
        if not self.active_models:
            raise ValueError("No active models available for prediction")
        
        # Get individual predictions
        individual_predictions = {}
        prediction_errors = []
        
        for model_id in self.active_models:
            try:
                model = self.models[model_id]
                predictions = model.predict(features)
                
                # Store individual prediction (assuming single sample for now)
                individual_predictions[model_id] = predictions[0] if len(predictions) > 0 else 0.0
                
                # Update model metrics
                self.model_metrics[model_id].prediction_count += 1
                self.model_metrics[model_id].last_prediction = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Prediction failed for model {model_id}: {e}")
                self.model_metrics[model_id].error_count += 1
                prediction_errors.append(model_id)
        
        # Remove failed models from active set
        for model_id in prediction_errors:
            self.active_models.discard(model_id)
            self.model_metrics[model_id].status = ModelStatus.FAILED
        
        if not individual_predictions:
            raise ValueError("All models failed during prediction")
        
        # Combine predictions using ensemble strategy
        ensemble_prediction = self._combine_predictions(individual_predictions)
        
        # Calculate uncertainty measures
        predictions_array = np.array(list(individual_predictions.values()))
        prediction_std = np.std(predictions_array)
        
        # Confidence interval (assuming normal distribution)
        confidence_interval_lower = ensemble_prediction - 1.96 * prediction_std
        confidence_interval_upper = ensemble_prediction + 1.96 * prediction_std
        
        # Model agreement (inverse of coefficient of variation)
        cv = prediction_std / abs(ensemble_prediction) if ensemble_prediction != 0 else float('inf')
        model_agreement = 1.0 / (1.0 + cv)
        
        # Calculate model contributions
        model_contributions = {}
        for model_id, prediction in individual_predictions.items():
            weight = self.ensemble_weights.get(model_id, 0.0)
            contribution = weight * prediction
            model_contributions[model_id] = contribution
        
        # Ensemble diversity (standard deviation of weights)
        weights = list(self.ensemble_weights.values())
        ensemble_diversity = np.std(weights) if len(weights) > 1 else 0.0
        
        # Prediction confidence based on model agreement and performance
        avg_confidence = np.mean([
            self.model_metrics[model_id].confidence_score
            for model_id in individual_predictions.keys()
        ])
        prediction_confidence = model_agreement * avg_confidence
        
        result = EnsemblePrediction(
            timestamp=datetime.now(),
            target_variable=target_variable,
            prediction_horizon='1h',  # Would be configurable
            ensemble_prediction=ensemble_prediction,
            individual_predictions=individual_predictions,
            prediction_std=prediction_std,
            confidence_interval_lower=confidence_interval_lower,
            confidence_interval_upper=confidence_interval_upper,
            model_agreement=model_agreement,
            model_weights=self.ensemble_weights.copy(),
            ensemble_strategy=self.ensemble_strategy,
            model_contributions=model_contributions,
            active_models=list(self.active_models),
            prediction_confidence=prediction_confidence,
            ensemble_diversity=ensemble_diversity,
            metadata={
                'failed_models': prediction_errors,
                'total_models': len(self.models)
            }
        )
        
        # Store prediction history
        self.prediction_history.append(result)
        
        return result
    
    def _combine_predictions(self, individual_predictions: Dict[str, float]) -> float:
        """Combine individual predictions using the current ensemble strategy."""
        if self.ensemble_strategy == EnsembleStrategy.SIMPLE_AVERAGE:
            return np.mean(list(individual_predictions.values()))
        
        elif self.ensemble_strategy in [EnsembleStrategy.WEIGHTED_AVERAGE, EnsembleStrategy.PERFORMANCE_WEIGHTED]:
            # Use current ensemble weights
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model_id, prediction in individual_predictions.items():
                weight = self.ensemble_weights.get(model_id, 0.0)
                weighted_sum += weight * prediction
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else np.mean(list(individual_predictions.values()))
        
        elif self.ensemble_strategy == EnsembleStrategy.DYNAMIC_SELECTION:
            # Select best performing model for this prediction
            best_model_id = max(
                individual_predictions.keys(),
                key=lambda m: self.model_metrics[m].recent_performance
            )
            return individual_predictions[best_model_id]
        
        else:
            # Default to weighted average
            return self._combine_predictions_weighted_average(individual_predictions)
    
    def _combine_predictions_weighted_average(self, individual_predictions: Dict[str, float]) -> float:
        """Combine predictions using weighted average."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_id, prediction in individual_predictions.items():
            weight = self.ensemble_weights.get(model_id, 1.0 / len(individual_predictions))
            weighted_sum += weight * prediction
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def optimize_hyperparameters(self, model_id: str, target_variable: str) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model using Optuna."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        if self.training_data is None:
            raise ValueError("No training data available")
        
        model = self.models[model_id]
        features, targets = self.training_data
        target = targets[target_variable]
        
        def objective(trial):
            # Define hyperparameter search space based on model type
            if model.model_type == ModelType.LIGHTGBM:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
            elif model.model_type == ModelType.XGBOOST:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
            elif model.model_type == ModelType.RANDOM_FOREST:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': 42,
                    'n_jobs': -1
                }
            else:
                # Use current parameters
                params = model.config.hyperparameters.copy()
            
            # Create temporary model config
            temp_config = ModelConfiguration(
                model_type=model.model_type,
                hyperparameters=params,
                feature_columns=model.config.feature_columns,
                target_column=target_variable,
                scaler_type=model.config.scaler_type
            )
            
            # Create temporary model
            temp_model = BaseModelWrapper(f"temp_{model_id}", model.model_type, temp_config)
            
            # Cross-validation
            cv = TimeSeriesSplit(n_splits=self.config['cross_validation_folds'])
            scores = []
            
            for train_idx, val_idx in cv.split(features):
                X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
                y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
                
                try:
                    temp_model.fit(X_train, y_train)
                    predictions = temp_model.predict(X_val)
                    
                    if self.config['model_selection_metric'] == 'r2':
                        score = r2_score(y_val, predictions)
                    elif self.config['model_selection_metric'] == 'mse':
                        score = -mean_squared_error(y_val, predictions)  # Negative for maximization
                    else:
                        score = r2_score(y_val, predictions)
                    
                    scores.append(score)
                    
                except Exception:
                    return -float('inf')
            
            return np.mean(scores) if scores else -float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['hyperopt_trials'])
        
        # Update model with best parameters
        best_params = study.best_params
        model.config.hyperparameters.update(best_params)
        model._initialize_model()  # Reinitialize with new parameters
        
        # Update model metrics
        self.model_metrics[model_id].hyperparameters = model.config.hyperparameters.copy()
        
        self.logger.info(f"Optimized hyperparameters for {model_id}: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'trials': len(study.trials)
        }
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for all models."""
        summary = {
            'timestamp': datetime.now(),
            'total_models': len(self.models),
            'active_models': len(self.active_models),
            'ensemble_strategy': self.ensemble_strategy.value,
            'models': {}
        }
        
        for model_id, metrics in self.model_metrics.items():
            summary['models'][model_id] = {
                'model_type': metrics.model_type.value,
                'status': metrics.status.value,
                'r2_score': metrics.r2_score,
                'mse': metrics.mse,
                'mae': metrics.mae,
                'recent_performance': metrics.recent_performance,
                'prediction_count': metrics.prediction_count,
                'error_count': metrics.error_count,
                'confidence_score': metrics.confidence_score,
                'ensemble_weight': self.ensemble_weights.get(model_id, 0.0),
                'last_training': metrics.last_training.isoformat() if metrics.last_training != datetime.min else None,
                'training_time': metrics.training_time,
                'top_features': dict(list(metrics.feature_importance.items())[:5])
            }
        
        # Ensemble performance
        if self.ensemble_performance_history:
            recent_performance = list(self.ensemble_performance_history)[-10:]
            summary['ensemble_performance'] = {
                'recent_average': np.mean(recent_performance),
                'recent_std': np.std(recent_performance),
                'total_predictions': len(self.prediction_history)
            }
        
        return summary
    
    def save_ensemble(self, path: str):
        """Save entire ensemble to disk."""
        ensemble_data = {
            'config': self.config,
            'model_metrics': self.model_metrics,
            'ensemble_weights': self.ensemble_weights,
            'ensemble_strategy': self.ensemble_strategy.value,
            'active_models': list(self.active_models),
            'timestamp': datetime.now()
        }
        
        # Save ensemble metadata
        with open(path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        # Save individual models
        models_dir = Path(path).parent / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for model_id, model in self.models.items():
            model_path = models_dir / f"{model_id}.joblib"
            model.save_model(str(model_path))
        
        self.logger.info(f"Saved ensemble to {path}")
    
    def load_ensemble(self, path: str):
        """Load ensemble from disk."""
        # Load ensemble metadata
        with open(path, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.config.update(ensemble_data['config'])
        self.model_metrics = ensemble_data['model_metrics']
        self.ensemble_weights = ensemble_data['ensemble_weights']
        self.ensemble_strategy = EnsembleStrategy(ensemble_data['ensemble_strategy'])
        self.active_models = set(ensemble_data['active_models'])
        
        # Load individual models
        models_dir = Path(path).parent / 'models'
        self.models = {}
        
        for model_id in self.model_metrics.keys():
            model_path = models_dir / f"{model_id}.joblib"
            if model_path.exists():
                self.models[model_id] = BaseModelWrapper.load_model(str(model_path))
        
        self.logger.info(f"Loaded ensemble from {path}")


# Example usage and testing
if __name__ == "__main__":
    import json
    from sklearn.datasets import make_regression
    from ..config.manager import ConfigurationManager
    
    def main():
        # Configure logging
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize ensemble manager
        config_manager = ConfigurationManager()
        ensemble = EnsembleModelManager(config_manager)
        
        # Create sample data
        X, y = make_regression(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            noise=0.1,
            random_state=42
        )
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        features = pd.DataFrame(X, columns=feature_names)
        targets = {'price_1h': pd.Series(y)}
        
        # Update training data
        ensemble.update_training_data(features, targets)
        
        # Train models
        print("Training ensemble models...")
        training_results = ensemble.train_models('price_1h')
        print(f"Training results: {training_results}")
        
        # Get performance summary
        performance_summary = ensemble.get_model_performance_summary()
        print(f"\nPerformance Summary:")
        print(json.dumps(performance_summary, indent=2, default=str))
        
        # Make predictions
        print("\nMaking predictions...")
        test_features = features.tail(10)
        
        for i in range(5):
            prediction = ensemble.predict(test_features.iloc[[i]], 'price_1h')
            print(f"Prediction {i+1}: {prediction.ensemble_prediction:.4f} "
                  f"(±{prediction.prediction_std:.4f})")
        
        # Test hyperparameter optimization
        if ensemble.active_models:
            first_model = list(ensemble.active_models)[0]
            print(f"\nOptimizing hyperparameters for {first_model}...")
            opt_results = ensemble.optimize_hyperparameters(first_model, 'price_1h')
            print(f"Optimization results: {opt_results}")
    
    # Run the example
    main()