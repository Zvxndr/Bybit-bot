"""
Advanced Ensemble Methods for Trading Strategies.

This module provides sophisticated ensemble techniques that combine
multiple models and strategies for improved robustness and performance:

- Meta-learning ensemble approaches
- Dynamic ensemble weighting based on market conditions
- Multi-strategy ensemble combining different approaches
- Online ensemble learning with adaptive weights
- Regime-aware ensemble selection
- Ensemble diversity optimization
- Stacking and blending techniques
- Ensemble uncertainty quantification

All ensemble methods are designed for financial applications with
proper handling of non-stationarity and regime changes.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from .models import ModelResult, LightGBMTrader, XGBoostTrader
from .regimes import MultiFactorRegimeDetector
from ..utils.logging import TradingLogger


@dataclass
class EnsembleWeight:
    """Container for ensemble weights with metadata."""
    
    model_name: str
    weight: float
    confidence: float
    last_updated: datetime
    performance_history: List[float]
    regime_weights: Optional[Dict[int, float]] = None


@dataclass
class EnsembleResult:
    """Container for ensemble prediction results."""
    
    predictions: pd.Series
    probabilities: pd.DataFrame
    individual_predictions: Dict[str, pd.Series]
    individual_probabilities: Dict[str, pd.DataFrame]
    weights: Dict[str, float]
    confidence_scores: pd.Series
    ensemble_uncertainty: pd.Series
    performance_metrics: Dict[str, float]


class DynamicEnsemble:
    """
    Dynamic ensemble with adaptive weighting based on performance and market conditions.
    
    This ensemble automatically adjusts model weights based on recent performance,
    market regimes, and prediction confidence.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("DynamicEnsemble")
        
        self.models = {}
        self.weights = {}
        self.performance_history = {}
        self.regime_detector = None
        self.is_fitted = False
        
    def _default_config(self) -> Dict:
        """Default configuration for dynamic ensemble."""
        return {
            'models': ['lightgbm', 'xgboost'],
            'weight_update_method': 'exponential_decay',  # exponential_decay, performance_based, regime_aware
            'performance_window': 50,  # Lookback window for performance calculation
            'min_weight': 0.05,  # Minimum weight for any model
            'max_weight': 0.7,   # Maximum weight for any model
            'weight_decay': 0.95,  # Decay factor for exponential weighting
            'confidence_threshold': 0.6,  # Threshold for prediction confidence
            'regime_adaptation': True,  # Use regime-aware weighting
            'online_learning': True,  # Update weights online
            'diversity_bonus': 0.1,  # Bonus for diverse predictions
            'uncertainty_quantification': True,
        }
    
    def add_model(self, model_name: str, model: Any, initial_weight: float = None):
        """Add a model to the ensemble."""
        if initial_weight is None:
            initial_weight = 1.0 / (len(self.models) + 1)
            
        self.models[model_name] = model
        self.weights[model_name] = EnsembleWeight(
            model_name=model_name,
            weight=initial_weight,
            confidence=0.5,
            last_updated=datetime.now(),
            performance_history=[]
        )
        self.performance_history[model_name] = []
        
        self.logger.info(f"Added model {model_name} with initial weight {initial_weight:.3f}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DynamicEnsemble':
        """Fit all models in the ensemble."""
        self.logger.info("Fitting dynamic ensemble")
        
        # Fit individual models
        for model_name, model in self.models.items():
            self.logger.info(f"Fitting {model_name}")
            
            if hasattr(model, 'fit'):
                model.fit(X, y)
            else:
                self.logger.warning(f"Model {model_name} does not have fit method")
        
        # Initialize regime detector if enabled
        if self.config['regime_adaptation']:
            self.regime_detector = MultiFactorRegimeDetector()
            self.regime_detector.fit(X)
        
        # Calculate initial weights based on cross-validation performance
        self._calculate_initial_weights(X, y)
        
        self.is_fitted = True
        self.logger.info("Dynamic ensemble fitting completed")
        
        return self
    
    def predict(self, X: pd.DataFrame, update_weights: bool = True) -> EnsembleResult:
        """Make ensemble predictions with dynamic weighting."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get individual model predictions
        individual_predictions = {}
        individual_probabilities = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict'):
                individual_predictions[model_name] = pd.Series(
                    model.predict(X), 
                    index=X.index
                )
            
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                individual_probabilities[model_name] = pd.DataFrame(
                    probs, 
                    index=X.index,
                    columns=[f'class_{i}' for i in range(probs.shape[1])]
                )
        
        # Get current regime if regime adaptation is enabled
        current_regime = None
        if self.regime_detector is not None:
            current_regime, _ = self.regime_detector.predict_regimes(X)
        
        # Calculate dynamic weights
        current_weights = self._calculate_dynamic_weights(
            individual_predictions, 
            individual_probabilities, 
            current_regime
        )
        
        # Combine predictions
        ensemble_predictions, ensemble_probabilities = self._combine_predictions(
            individual_predictions, 
            individual_probabilities, 
            current_weights
        )
        
        # Calculate confidence and uncertainty
        confidence_scores = self._calculate_confidence(ensemble_probabilities)
        ensemble_uncertainty = self._calculate_uncertainty(individual_probabilities, current_weights)
        
        # Update weights if online learning is enabled
        if update_weights and self.config['online_learning']:
            self._update_weights_online(individual_predictions, current_weights)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_ensemble_metrics(
            individual_predictions, current_weights
        )
        
        return EnsembleResult(
            predictions=ensemble_predictions,
            probabilities=ensemble_probabilities,
            individual_predictions=individual_predictions,
            individual_probabilities=individual_probabilities,
            weights=current_weights,
            confidence_scores=confidence_scores,
            ensemble_uncertainty=ensemble_uncertainty,
            performance_metrics=performance_metrics
        )
    
    def _calculate_initial_weights(self, X: pd.DataFrame, y: pd.Series):
        """Calculate initial weights based on cross-validation performance."""
        from sklearn.model_selection import cross_val_score
        
        cv_scores = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    # Use a simple scoring method
                    scores = []
                    
                    # Simple train-test split for scoring
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                    
                    # Fit on training data
                    temp_model = type(model)()
                    if hasattr(temp_model, 'fit'):
                        temp_model.fit(X_train, y_train)
                        predictions = temp_model.predict(X_test)
                        score = accuracy_score(y_test, predictions)
                        cv_scores[model_name] = score
                    else:
                        cv_scores[model_name] = 0.5  # Default score
                else:
                    cv_scores[model_name] = 0.5  # Default score
                    
            except Exception as e:
                self.logger.warning(f"Error calculating CV score for {model_name}: {e}")
                cv_scores[model_name] = 0.5
        
        # Convert scores to weights
        total_score = sum(cv_scores.values())
        if total_score > 0:
            for model_name in self.weights:
                self.weights[model_name].weight = cv_scores[model_name] / total_score
        
        self.logger.info(f"Initial weights: {{k: v.weight for k, v in self.weights.items()}}")
    
    def _calculate_dynamic_weights(
        self, 
        predictions: Dict[str, pd.Series], 
        probabilities: Dict[str, pd.DataFrame],
        current_regime: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Calculate dynamic weights based on current conditions."""
        if self.config['weight_update_method'] == 'performance_based':
            return self._performance_based_weights()
        elif self.config['weight_update_method'] == 'regime_aware':
            return self._regime_aware_weights(current_regime)
        else:
            return self._exponential_decay_weights()
    
    def _performance_based_weights(self) -> Dict[str, float]:
        """Calculate weights based on recent performance."""
        weights = {}
        performance_scores = {}
        
        for model_name, weight_info in self.weights.items():
            if len(weight_info.performance_history) > 0:
                # Use recent performance (last N predictions)
                recent_performance = weight_info.performance_history[-self.config['performance_window']:]
                performance_scores[model_name] = np.mean(recent_performance)
            else:
                performance_scores[model_name] = 0.5  # Default
        
        # Convert to weights
        total_score = sum(performance_scores.values())
        if total_score > 0:
            for model_name in self.weights:
                weights[model_name] = performance_scores[model_name] / total_score
        else:
            # Equal weights if no performance data
            weights = {model_name: 1.0 / len(self.models) for model_name in self.models}
        
        # Apply min/max constraints
        weights = self._apply_weight_constraints(weights)
        
        return weights
    
    def _regime_aware_weights(self, current_regime: Optional[pd.Series]) -> Dict[str, float]:
        """Calculate regime-aware weights."""
        if current_regime is None:
            return self._exponential_decay_weights()
        
        weights = {}
        
        # Get the most common regime in recent data
        recent_regime = current_regime.iloc[-min(20, len(current_regime)):].mode()
        if len(recent_regime) > 0:
            dominant_regime = recent_regime.iloc[0]
        else:
            dominant_regime = 0
        
        # Use regime-specific weights if available
        for model_name, weight_info in self.weights.items():
            if weight_info.regime_weights and dominant_regime in weight_info.regime_weights:
                weights[model_name] = weight_info.regime_weights[dominant_regime]
            else:
                weights[model_name] = weight_info.weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return self._apply_weight_constraints(weights)
    
    def _exponential_decay_weights(self) -> Dict[str, float]:
        """Calculate weights using exponential decay of past performance."""
        weights = {}
        
        for model_name, weight_info in self.weights.items():
            # Start with current weight and apply decay if performance is poor
            current_weight = weight_info.weight
            
            if len(weight_info.performance_history) > 0:
                recent_performance = np.mean(weight_info.performance_history[-5:])
                if recent_performance < 0.5:  # Below average performance
                    current_weight *= self.config['weight_decay']
                elif recent_performance > 0.6:  # Above average performance
                    current_weight *= (1 + (1 - self.config['weight_decay']))
            
            weights[model_name] = current_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return self._apply_weight_constraints(weights)
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max weight constraints."""
        # Apply minimum weight
        for model_name in weights:
            weights[model_name] = max(weights[model_name], self.config['min_weight'])
        
        # Apply maximum weight
        for model_name in weights:
            weights[model_name] = min(weights[model_name], self.config['max_weight'])
        
        # Renormalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _combine_predictions(
        self, 
        predictions: Dict[str, pd.Series], 
        probabilities: Dict[str, pd.DataFrame],
        weights: Dict[str, float]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Combine individual predictions using weights."""
        if not predictions:
            raise ValueError("No predictions to combine")
        
        # Get common index
        common_index = None
        for pred in predictions.values():
            if common_index is None:
                common_index = pred.index
            else:
                common_index = common_index.intersection(pred.index)
        
        # Combine probabilities if available
        if probabilities:
            ensemble_probs = None
            total_weight = 0
            
            for model_name, probs in probabilities.items():
                weight = weights.get(model_name, 0)
                if weight > 0:
                    model_probs = probs.reindex(common_index).fillna(0)
                    
                    if ensemble_probs is None:
                        ensemble_probs = weight * model_probs
                    else:
                        ensemble_probs += weight * model_probs
                    
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_probs = ensemble_probs / total_weight
            
            # Convert to predictions
            ensemble_predictions = pd.Series(
                np.argmax(ensemble_probs.values, axis=1),
                index=ensemble_probs.index
            )
            
            return ensemble_predictions, ensemble_probs
        
        else:
            # Combine predictions directly (voting)
            prediction_matrix = pd.DataFrame({
                model_name: pred.reindex(common_index) 
                for model_name, pred in predictions.items()
            })
            
            # Weighted voting
            weighted_predictions = pd.Series(0, index=common_index)
            
            for model_name, pred in predictions.items():
                weight = weights.get(model_name, 0)
                weighted_predictions += weight * pred.reindex(common_index).fillna(0)
            
            ensemble_predictions = np.round(weighted_predictions).astype(int)
            
            # Create dummy probabilities
            n_classes = max(prediction_matrix.max().max() + 1, 3)
            ensemble_probs = pd.DataFrame(
                np.eye(n_classes)[ensemble_predictions],
                index=common_index,
                columns=[f'class_{i}' for i in range(n_classes)]
            )
            
            return ensemble_predictions, ensemble_probs
    
    def _calculate_confidence(self, probabilities: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for ensemble predictions."""
        # Confidence based on maximum probability
        max_probs = probabilities.max(axis=1)
        
        # Normalize to 0-1 scale
        confidence = (max_probs - 0.5) * 2  # Convert from [0.5, 1] to [0, 1]
        confidence = confidence.clip(0, 1)
        
        return confidence
    
    def _calculate_uncertainty(
        self, 
        individual_probabilities: Dict[str, pd.DataFrame],
        weights: Dict[str, float]
    ) -> pd.Series:
        """Calculate ensemble uncertainty using prediction diversity."""
        if not individual_probabilities:
            return pd.Series(0.5, index=individual_probabilities[list(individual_probabilities.keys())[0]].index)
        
        # Calculate prediction entropy
        uncertainty_scores = []
        
        common_index = None
        for probs in individual_probabilities.values():
            if common_index is None:
                common_index = probs.index
            else:
                common_index = common_index.intersection(probs.index)
        
        for idx in common_index:
            model_predictions = []
            
            for model_name, probs in individual_probabilities.items():
                weight = weights.get(model_name, 0)
                if weight > 0:
                    model_pred = np.argmax(probs.loc[idx].values)
                    model_predictions.append(model_pred)
            
            if model_predictions:
                # Calculate diversity (standard deviation of predictions)
                uncertainty = np.std(model_predictions) / max(model_predictions) if max(model_predictions) > 0 else 0
            else:
                uncertainty = 0.5
            
            uncertainty_scores.append(uncertainty)
        
        return pd.Series(uncertainty_scores, index=common_index)
    
    def _update_weights_online(self, predictions: Dict[str, pd.Series], current_weights: Dict[str, float]):
        """Update weights based on online learning."""
        # This is a simplified online update - in practice would use actual performance feedback
        for model_name, weight_info in self.weights.items():
            # Update weight based on recent prediction confidence
            if model_name in current_weights:
                new_weight = current_weights[model_name]
                
                # Exponential moving average
                alpha = 0.1  # Learning rate
                weight_info.weight = (1 - alpha) * weight_info.weight + alpha * new_weight
                weight_info.last_updated = datetime.now()
    
    def _calculate_ensemble_metrics(
        self, 
        predictions: Dict[str, pd.Series], 
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate ensemble performance metrics."""
        metrics = {}
        
        # Weight distribution metrics
        weight_values = list(weights.values())
        metrics['weight_entropy'] = stats.entropy(weight_values) if len(weight_values) > 1 else 0
        metrics['weight_max'] = max(weight_values) if weight_values else 0
        metrics['weight_min'] = min(weight_values) if weight_values else 0
        metrics['weight_std'] = np.std(weight_values) if len(weight_values) > 1 else 0
        
        # Prediction diversity
        if len(predictions) > 1:
            pred_matrix = pd.DataFrame(predictions)
            pred_corr = pred_matrix.corr()
            
            # Average correlation between models
            upper_triangle = pred_corr.where(np.triu(np.ones(pred_corr.shape), k=1).astype(bool))
            metrics['prediction_correlation'] = upper_triangle.stack().mean()
            
            # Prediction disagreement rate
            disagreement_count = 0
            total_comparisons = 0
            
            for i, pred1 in enumerate(predictions.values()):
                for j, pred2 in enumerate(list(predictions.values())[i+1:], i+1):
                    disagreement_count += (pred1 != pred2).sum()
                    total_comparisons += len(pred1)
            
            metrics['disagreement_rate'] = disagreement_count / total_comparisons if total_comparisons > 0 else 0
        
        return metrics
    
    def get_model_performance_summary(self) -> pd.DataFrame:
        """Get performance summary for all models."""
        summary_data = []
        
        for model_name, weight_info in self.weights.items():
            summary_data.append({
                'model': model_name,
                'current_weight': weight_info.weight,
                'confidence': weight_info.confidence,
                'last_updated': weight_info.last_updated,
                'performance_history_length': len(weight_info.performance_history),
                'avg_recent_performance': np.mean(weight_info.performance_history[-10:]) if weight_info.performance_history else 0
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_weight_evolution(self, save_path: Optional[str] = None):
        """Plot evolution of model weights over time."""
        # This would require storing weight history - simplified version
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot current weights as bar chart
        model_names = list(self.weights.keys())
        current_weights = [self.weights[name].weight for name in model_names]
        
        ax.bar(model_names, current_weights)
        ax.set_ylabel('Weight')
        ax.set_title('Current Ensemble Model Weights')
        ax.set_ylim(0, 1)
        
        # Add weight values on bars
        for i, weight in enumerate(current_weights):
            ax.text(i, weight + 0.01, f'{weight:.3f}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Weight plot saved to {save_path}")
        else:
            plt.show()


class StackingEnsemble:
    """
    Stacking ensemble that uses a meta-model to combine base model predictions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("StackingEnsemble")
        
        self.base_models = {}
        self.meta_model = None
        self.is_fitted = False
        
    def _default_config(self) -> Dict:
        """Default configuration for stacking ensemble."""
        return {
            'meta_model_type': 'logistic_regression',  # logistic_regression, random_forest
            'cv_folds': 5,
            'use_probabilities': True,  # Use probabilities as meta-features
            'include_original_features': False,  # Include original features in meta-model
        }
    
    def add_base_model(self, model_name: str, model: Any):
        """Add a base model to the stacking ensemble."""
        self.base_models[model_name] = model
        self.logger.info(f"Added base model: {model_name}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'StackingEnsemble':
        """Fit stacking ensemble."""
        self.logger.info("Fitting stacking ensemble")
        
        # Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X, y)
        
        # Create meta-model
        if self.config['meta_model_type'] == 'logistic_regression':
            self.meta_model = LogisticRegression(random_state=42)
        elif self.config['meta_model_type'] == 'random_forest':
            self.meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Fit meta-model
        self.meta_model.fit(meta_features, y)
        
        # Fit base models on full dataset
        for model_name, model in self.base_models.items():
            self.logger.info(f"Fitting base model: {model_name}")
            model.fit(X, y)
        
        self.is_fitted = True
        self.logger.info("Stacking ensemble fitting completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make stacking ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get base model predictions
        meta_features = self._get_meta_features_predict(X)
        
        # Use meta-model to make final prediction
        predictions = self.meta_model.predict(meta_features)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get base model predictions
        meta_features = self._get_meta_features_predict(X)
        
        # Use meta-model to predict probabilities
        probabilities = self.meta_model.predict_proba(meta_features)
        
        return probabilities
    
    def _generate_meta_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Generate meta-features using cross-validation."""
        from sklearn.model_selection import StratifiedKFold
        
        meta_features = pd.DataFrame(index=X.index)
        cv = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=42)
        
        for model_name, model in self.base_models.items():
            model_predictions = np.zeros(len(X))
            model_probabilities = None
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Fit model on training fold
                fold_model = type(model)()  # Create new instance
                if hasattr(fold_model, 'fit'):
                    fold_model.fit(X_train, y_train)
                    
                    # Predict on validation fold
                    val_predictions = fold_model.predict(X_val)
                    model_predictions[val_idx] = val_predictions
                    
                    # Get probabilities if available and requested
                    if self.config['use_probabilities'] and hasattr(fold_model, 'predict_proba'):
                        val_probs = fold_model.predict_proba(X_val)
                        
                        if model_probabilities is None:
                            model_probabilities = np.zeros((len(X), val_probs.shape[1]))
                        
                        model_probabilities[val_idx] = val_probs
            
            # Add predictions as meta-feature
            meta_features[f'{model_name}_pred'] = model_predictions
            
            # Add probabilities as meta-features
            if model_probabilities is not None:
                for i in range(model_probabilities.shape[1]):
                    meta_features[f'{model_name}_prob_{i}'] = model_probabilities[:, i]
        
        # Include original features if requested
        if self.config['include_original_features']:
            meta_features = pd.concat([meta_features, X], axis=1)
        
        return meta_features
    
    def _get_meta_features_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get meta-features for prediction."""
        meta_features = pd.DataFrame(index=X.index)
        
        for model_name, model in self.base_models.items():
            # Get predictions
            predictions = model.predict(X)
            meta_features[f'{model_name}_pred'] = predictions
            
            # Get probabilities if available and requested
            if self.config['use_probabilities'] and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                
                for i in range(probabilities.shape[1]):
                    meta_features[f'{model_name}_prob_{i}'] = probabilities[:, i]
        
        # Include original features if requested
        if self.config['include_original_features']:
            meta_features = pd.concat([meta_features, X], axis=1)
        
        return meta_features