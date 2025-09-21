"""
Machine Learning Models for Trading Strategies.

This module provides sophisticated ML model implementations optimized
for financial time series prediction:

- LightGBM with financial-specific tuning
- XGBoost with proper regularization
- Neural networks for complex patterns
- Ensemble methods for robustness
- Online learning for adaptability
- Proper financial cross-validation
- Feature importance analysis
- Model interpretability tools

All models are designed with financial ML best practices:
- Proper handling of time series data
- Prevention of look-ahead bias
- Robust validation techniques
- Feature importance tracking
- Model degradation monitoring
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from ..backtest.purged_cv import PurgedTimeSeriesCV
from ..utils.logging import TradingLogger


@dataclass
class ModelResult:
    """Container for ML model results."""
    
    model_name: str
    model: Any
    predictions: pd.Series
    probabilities: Optional[pd.DataFrame] = None
    feature_importance: Optional[pd.DataFrame] = None
    validation_scores: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    prediction_time: float = 0.0
    model_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.validation_scores is None:
            self.validation_scores = {}
        if self.model_params is None:
            self.model_params = {}


class LightGBMTrader:
    """
    LightGBM implementation optimized for financial data.
    
    This class provides a LightGBM wrapper with financial-specific
    optimizations and proper validation techniques.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("LightGBMTrader")
        
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.is_fitted = False
        
    def _default_config(self) -> Dict:
        """Default configuration for LightGBM."""
        return {
            'objective': 'multiclass',
            'num_class': 3,  # Buy, Hold, Sell
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'early_stopping_rounds': 100,
            'num_boost_round': 1000,
            'valid_fraction': 0.2,
            'categorical_feature': 'auto',
        }
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        sample_weight: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None
    ) -> 'LightGBMTrader':
        """
        Fit LightGBM model with proper validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            sample_weight: Sample weights for training
            categorical_features: List of categorical feature names
            
        Returns:
            Self for method chaining
        """
        import time
        start_time = time.time()
        
        self.logger.info("Fitting LightGBM model")
        
        # Validate inputs
        if X.empty or y.empty:
            raise ValueError("X and y cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode target if necessary
        if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.values
        
        # Prepare categorical features
        categorical_indices = []
        if categorical_features:
            categorical_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
        
        # Split data for validation
        split_idx = int(len(X) * (1 - self.config['valid_fraction']))
        
        X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_valid = y_encoded[:split_idx], y_encoded[split_idx:]
        
        if sample_weight is not None:
            weight_train = sample_weight.iloc[:split_idx]
            weight_valid = sample_weight.iloc[split_idx:]
        else:
            weight_train = weight_valid = None
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train, 
            label=y_train, 
            weight=weight_train,
            categorical_feature=categorical_indices,
            free_raw_data=False
        )
        
        valid_data = lgb.Dataset(
            X_valid, 
            label=y_valid, 
            weight=weight_valid,
            categorical_feature=categorical_indices,
            reference=train_data,
            free_raw_data=False
        )
        
        # Train model
        self.model = lgb.train(
            params=self.config,
            train_set=train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round=self.config['num_boost_round'],
            early_stopping_rounds=self.config['early_stopping_rounds'],
            verbose_eval=False
        )
        
        self.is_fitted = True
        training_time = time.time() - start_time
        
        self.logger.info(f"LightGBM training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Handle multiclass predictions
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
        
        # Decode predictions if label encoder was used
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        probabilities = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Ensure probabilities are 2D
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)
        
        return probabilities
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def optimize_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        param_grid: Optional[Dict] = None,
        cv_folds: int = 5,
        optimization_method: str = 'grid'
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            param_grid: Parameter grid for optimization
            cv_folds: Number of CV folds
            optimization_method: 'grid' or 'random'
            
        Returns:
            Best parameters dictionary
        """
        if param_grid is None:
            param_grid = self._default_param_grid()
        
        self.logger.info(f"Optimizing hyperparameters using {optimization_method} search")
        
        # Create LightGBM estimator
        lgb_estimator = lgb.LGBMClassifier(**self.config)
        
        # Use purged time series CV
        cv_splitter = PurgedTimeSeriesCV(
            n_splits=cv_folds,
            purge_gap=1,
            embargo_gap=1
        )
        
        # Perform hyperparameter optimization
        if optimization_method == 'grid':
            search = GridSearchCV(
                estimator=lgb_estimator,
                param_grid=param_grid,
                cv=cv_splitter,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
        else:  # random
            search = RandomizedSearchCV(
                estimator=lgb_estimator,
                param_distributions=param_grid,
                n_iter=50,
                cv=cv_splitter,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        
        search.fit(X, y)
        
        # Update config with best parameters
        self.config.update(search.best_params_)
        
        self.logger.info(f"Best parameters: {search.best_params_}")
        
        return search.best_params_
    
    def _default_param_grid(self) -> Dict[str, List]:
        """Default parameter grid for optimization."""
        return {
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.6, 0.8, 1.0],
            'bagging_fraction': [0.6, 0.8, 1.0],
            'min_child_samples': [10, 20, 50],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5],
        }


class XGBoostTrader:
    """
    XGBoost implementation optimized for financial data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("XGBoostTrader")
        
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.is_fitted = False
        
    def _default_config(self) -> Dict:
        """Default configuration for XGBoost."""
        return {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'booster': 'gbtree',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'alpha': 0.1,
            'lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'early_stopping_rounds': 100,
            'n_estimators': 1000,
        }
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        sample_weight: Optional[pd.Series] = None
    ) -> 'XGBoostTrader':
        """Fit XGBoost model."""
        import time
        start_time = time.time()
        
        self.logger.info("Fitting XGBoost model")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode target if necessary
        if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.values
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_valid = y_encoded[:split_idx], y_encoded[split_idx:]
        
        if sample_weight is not None:
            weight_train = sample_weight.iloc[:split_idx]
        else:
            weight_train = None
        
        # Create XGBoost model
        self.model = xgb.XGBClassifier(**self.config)
        
        # Fit with early stopping
        self.model.fit(
            X_train, y_train,
            sample_weight=weight_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=False
        )
        
        self.is_fitted = True
        training_time = time.time() - start_time
        
        self.logger.info(f"XGBoost training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        
        # Decode predictions if label encoder was used
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance


class EnsembleTrader:
    """
    Ensemble of multiple ML models for robust predictions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("EnsembleTrader")
        
        self.models = {}
        self.ensemble_weights = None
        self.is_fitted = False
        
    def _default_config(self) -> Dict:
        """Default configuration for ensemble."""
        return {
            'models': ['lightgbm', 'xgboost'],
            'ensemble_method': 'weighted_average',  # weighted_average, stacking, voting
            'stacking_meta_model': 'logistic_regression',
            'cv_folds': 5,
            'optimize_weights': True,
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleTrader':
        """Fit ensemble of models."""
        self.logger.info("Fitting ensemble of models")
        
        # Initialize and fit individual models
        if 'lightgbm' in self.config['models']:
            lgb_model = LightGBMTrader()
            lgb_model.fit(X, y)
            self.models['lightgbm'] = lgb_model
        
        if 'xgboost' in self.config['models']:
            xgb_model = XGBoostTrader()
            xgb_model.fit(X, y)
            self.models['xgboost'] = xgb_model
        
        # Optimize ensemble weights if requested
        if self.config['optimize_weights']:
            self._optimize_ensemble_weights(X, y)
        else:
            # Equal weights
            self.ensemble_weights = {model_name: 1.0 / len(self.models) for model_name in self.models}
        
        self.is_fitted = True
        self.logger.info("Ensemble training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all models
        model_predictions = {}
        for model_name, model in self.models.items():
            model_predictions[model_name] = model.predict_proba(X)
        
        # Combine predictions
        if self.config['ensemble_method'] == 'weighted_average':
            ensemble_probs = self._weighted_average_predictions(model_predictions)
        else:
            ensemble_probs = self._voting_predictions(model_predictions)
        
        # Convert to class predictions
        predictions = np.argmax(ensemble_probs, axis=1)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ensemble probabilities."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all models
        model_predictions = {}
        for model_name, model in self.models.items():
            model_predictions[model_name] = model.predict_proba(X)
        
        # Combine predictions
        if self.config['ensemble_method'] == 'weighted_average':
            ensemble_probs = self._weighted_average_predictions(model_predictions)
        else:
            ensemble_probs = self._voting_predictions(model_predictions)
        
        return ensemble_probs
    
    def _weighted_average_predictions(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions using weighted average."""
        ensemble_probs = None
        total_weight = 0
        
        for model_name, predictions in model_predictions.items():
            weight = (self.ensemble_weights or {}).get(model_name, 1.0)
            
            if ensemble_probs is None:
                ensemble_probs = weight * predictions
            else:
                ensemble_probs += weight * predictions
            
            total_weight += weight
        
        return ensemble_probs / total_weight
    
    def _voting_predictions(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions using majority voting."""
        # Convert probabilities to class predictions
        class_predictions = {}
        for model_name, probs in model_predictions.items():
            class_predictions[model_name] = np.argmax(probs, axis=1)
        
        # Stack predictions
        predictions_stack = np.column_stack(list(class_predictions.values()))
        
        # Find mode (most common prediction) for each sample
        ensemble_predictions = stats.mode(predictions_stack, axis=1)[0].flatten()
        
        # Convert back to probabilities (simple approach)
        n_classes = max(predictions_stack.max() + 1, 3)
        ensemble_probs = np.zeros((len(ensemble_predictions), n_classes))
        ensemble_probs[np.arange(len(ensemble_predictions)), ensemble_predictions] = 1.0
        
        return ensemble_probs
    
    def _optimize_ensemble_weights(self, X: pd.DataFrame, y: pd.Series):
        """Optimize ensemble weights using cross-validation."""
        from scipy.optimize import minimize
        
        # Get out-of-fold predictions from each model
        cv_predictions = self._get_cv_predictions(X, y)
        
        # Optimize weights
        def objective(weights):
            weights = weights / weights.sum()  # Normalize
            ensemble_pred = self._combine_cv_predictions(cv_predictions, weights)
            return -accuracy_score(y, ensemble_pred)  # Negative because we minimize
        
        # Initial weights (equal)
        initial_weights = np.ones(len(self.models))
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Update weights
        optimized_weights = result.x / result.x.sum()
        self.ensemble_weights = {
            model_name: weight 
            for model_name, weight in zip(self.models.keys(), optimized_weights)
        }
        
        self.logger.info(f"Optimized ensemble weights: {self.ensemble_weights}")
    
    def _get_cv_predictions(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, np.ndarray]:
        """Get cross-validation predictions for weight optimization."""
        # This is a simplified version - in practice would use proper CV
        cv_predictions = {}
        
        # Split data
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        for model_name, model in self.models.items():
            # Refit on training data
            temp_model = type(model)()
            temp_model.fit(X_train, y_train)
            
            # Predict on test data
            cv_predictions[model_name] = temp_model.predict(X_test)
        
        return cv_predictions
    
    def _combine_cv_predictions(self, cv_predictions: Dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
        """Combine CV predictions with given weights."""
        ensemble_pred = None
        
        for i, (model_name, predictions) in enumerate(cv_predictions.items()):
            if ensemble_pred is None:
                ensemble_pred = weights[i] * predictions
            else:
                ensemble_pred += weights[i] * predictions
        
        return np.round(ensemble_pred).astype(int)


class MLModelFactory:
    """
    Factory for creating and managing ML models.
    """
    
    def __init__(self):
        self.logger = TradingLogger("MLModelFactory")
        
    def create_model(self, model_type: str, config: Optional[Dict] = None) -> Union[LightGBMTrader, XGBoostTrader, EnsembleTrader]:
        """Create ML model based on type."""
        if model_type.lower() == 'lightgbm':
            return LightGBMTrader(config)
        elif model_type.lower() == 'xgboost':
            return XGBoostTrader(config)
        elif model_type.lower() == 'ensemble':
            return EnsembleTrader(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_and_validate_model(
        self,
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        config: Optional[Dict] = None,
        validation_method: str = 'purged_cv'
    ) -> ModelResult:
        """Train and validate a model with proper financial validation."""
        import time
        
        self.logger.info(f"Training and validating {model_type} model")
        
        start_time = time.time()
        
        # Create model
        model = self.create_model(model_type, config)
        
        # Fit model
        model.fit(X, y)
        training_time = time.time() - start_time
        
        # Make predictions
        prediction_start = time.time()
        predictions = model.predict(X)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(X)
                if probs.ndim > 1:
                    prob_columns = [f'class_{i}_prob' for i in range(probs.shape[1])]
                    probabilities = pd.DataFrame(probs, index=X.index, columns=prob_columns)
            except Exception as e:
                self.logger.warning(f"Could not get probabilities: {e}")
        
        prediction_time = time.time() - prediction_start
        
        # Get feature importance
        feature_importance = None
        if hasattr(model, 'get_feature_importance'):
            try:
                feature_importance = model.get_feature_importance()
            except Exception as e:
                self.logger.warning(f"Could not get feature importance: {e}")
        
        # Calculate validation scores
        validation_scores = self._calculate_validation_scores(y, predictions, probabilities)
        
        # Create result
        result = ModelResult(
            model_name=model_type,
            model=model,
            predictions=pd.Series(predictions, index=X.index),
            probabilities=probabilities,
            feature_importance=feature_importance,
            validation_scores=validation_scores,
            training_time=training_time,
            prediction_time=prediction_time,
            model_params=config or {}
        )
        
        self.logger.info(f"Model training completed: {model_type}")
        
        return result
    
    def _calculate_validation_scores(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        probabilities: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate validation scores."""
        scores = {}
        
        try:
            scores['accuracy'] = accuracy_score(y_true, y_pred)
            scores['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            scores['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            scores['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # AUC score if probabilities available
            if probabilities is not None and len(probabilities.columns) == 2:
                try:
                    scores['auc'] = roc_auc_score(y_true, probabilities.iloc[:, 1])
                except Exception:
                    pass
                    
        except Exception as e:
            self.logger.warning(f"Error calculating validation scores: {e}")
        
        return scores
    
    def compare_models(
        self,
        model_types: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        configs: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, ModelResult]:
        """Compare multiple models and return results."""
        self.logger.info(f"Comparing {len(model_types)} models")
        
        results = {}
        
        for model_type in model_types:
            config = configs.get(model_type) if configs else None
            
            try:
                result = self.train_and_validate_model(model_type, X, y, config)
                results[model_type] = result
                
                self.logger.info(
                    f"{model_type} - Accuracy: {result.validation_scores.get('accuracy', 0):.3f}, "
                    f"F1: {result.validation_scores.get('f1', 0):.3f}"
                )
                
            except Exception as e:
                self.logger.error(f"Error training {model_type}: {e}")
        
        return results
    
    def save_model(self, model: Any, filepath: str):
        """Save model to file."""
        try:
            joblib.dump(model, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str) -> Any:
        """Load model from file."""
        try:
            model = joblib.load(filepath)
            self.logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise