"""
Advanced Prediction Engine for real-time market predictions and strategy signals.
Supports ensemble predictions, confidence scoring, and multi-timeframe analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
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

from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from scipy import stats

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class PredictionType(Enum):
    """Types of predictions supported by the engine."""
    PRICE = "price"
    DIRECTION = "direction"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    TREND_STRENGTH = "trend_strength"
    MARKET_REGIME = "market_regime"

@dataclass
class PredictionResult:
    """Result of a prediction operation."""
    prediction_type: PredictionType
    symbol: str
    timeframe: str
    prediction: Union[float, int, str]
    confidence: float
    probability_distribution: Optional[Dict[str, float]]
    feature_importance: Optional[Dict[str, float]]
    model_consensus: Dict[str, Any]
    timestamp: datetime
    horizon_minutes: int
    metadata: Dict[str, Any]

@dataclass
class EnsemblePrediction:
    """Result of ensemble prediction combining multiple models."""
    primary_prediction: PredictionResult
    model_predictions: List[PredictionResult]
    consensus_score: float
    disagreement_score: float
    weight_distribution: Dict[str, float]
    risk_adjusted_prediction: Union[float, int, str]

class PredictionEngine:
    """Advanced prediction engine with ensemble methods and confidence scoring."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Prediction configuration
        self.prediction_config = {
            'ensemble_method': 'weighted_voting',  # simple_voting, weighted_voting, stacked
            'confidence_threshold': 0.6,
            'consensus_threshold': 0.7,
            'max_prediction_horizon': 1440,  # 24 hours in minutes
            'model_weights': {
                'lstm': 0.35,
                'transformer': 0.25,
                'random_forest': 0.2,
                'xgboost': 0.2
            },
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'prediction_types': [
                PredictionType.PRICE,
                PredictionType.DIRECTION,
                PredictionType.VOLATILITY
            ]
        }
        
        # Active models for predictions
        self.active_models = {}
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance tracking
        self.prediction_performance = {
            'accuracy_scores': {},
            'mse_scores': {},
            'mae_scores': {},
            'confidence_calibration': {},
            'model_weights_history': []
        }
        
        # Real-time prediction tracking
        self.active_predictions = {}
        
        self.logger.info("PredictionEngine initialized with ensemble capabilities")
    
    async def register_model(self, model_name: str, model: Any, model_type: str = 'sklearn'):
        """Register a trained model for predictions."""
        try:
            self.active_models[model_name] = {
                'model': model,
                'type': model_type,
                'registered_at': datetime.now(),
                'prediction_count': 0,
                'accuracy_score': 0.0,
                'weight': self.prediction_config['model_weights'].get(model_name, 0.1)
            }
            
            self.logger.info(f"Model {model_name} registered for predictions")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register model {model_name}: {e}")
            return False
    
    async def predict_price(self, symbol: str, features: np.ndarray, 
                          timeframe: str = '1m', horizon_minutes: int = 5) -> PredictionResult:
        """Predict future price for a symbol."""
        try:
            cache_key = f"price_{symbol}_{timeframe}_{horizon_minutes}"
            
            # Check cache
            if self._is_cached_valid(cache_key):
                return self.prediction_cache[cache_key]
            
            # Get predictions from all models
            model_predictions = []
            for model_name, model_info in self.active_models.items():
                try:
                    pred = await self._predict_with_model(
                        model_info, features, PredictionType.PRICE
                    )
                    if pred is not None:
                        model_predictions.append({
                            'model': model_name,
                            'prediction': pred,
                            'weight': model_info['weight']
                        })
                except Exception as e:
                    self.logger.warning(f"Model {model_name} prediction failed: {e}")
            
            if not model_predictions:
                raise ValueError("No models available for prediction")
            
            # Calculate ensemble prediction
            ensemble_pred = self._calculate_ensemble_prediction(model_predictions)
            
            # Calculate confidence
            confidence = self._calculate_confidence(model_predictions, ensemble_pred)
            
            # Feature importance (if available)
            feature_importance = await self._calculate_feature_importance(
                model_predictions, features
            )
            
            result = PredictionResult(
                prediction_type=PredictionType.PRICE,
                symbol=symbol,
                timeframe=timeframe,
                prediction=ensemble_pred,
                confidence=confidence,
                probability_distribution=None,
                feature_importance=feature_importance,
                model_consensus=self._calculate_model_consensus(model_predictions),
                timestamp=datetime.now(),
                horizon_minutes=horizon_minutes,
                metadata={
                    'num_models': len(model_predictions),
                    'ensemble_method': self.prediction_config['ensemble_method']
                }
            )
            
            # Cache result
            self.prediction_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Price prediction failed for {symbol}: {e}")
            raise
    
    async def predict_direction(self, symbol: str, features: np.ndarray, 
                              timeframe: str = '1m', horizon_minutes: int = 5) -> PredictionResult:
        """Predict price direction (up/down/sideways)."""
        try:
            cache_key = f"direction_{symbol}_{timeframe}_{horizon_minutes}"
            
            if self._is_cached_valid(cache_key):
                return self.prediction_cache[cache_key]
            
            # Get direction predictions from all models
            model_predictions = []
            for model_name, model_info in self.active_models.items():
                try:
                    pred = await self._predict_direction_with_model(
                        model_info, features
                    )
                    if pred is not None:
                        model_predictions.append({
                            'model': model_name,
                            'prediction': pred,
                            'weight': model_info['weight']
                        })
                except Exception as e:
                    self.logger.warning(f"Direction prediction failed for {model_name}: {e}")
            
            if not model_predictions:
                raise ValueError("No models available for direction prediction")
            
            # Calculate ensemble direction
            direction_votes = {}
            total_weight = 0
            
            for pred in model_predictions:
                direction = pred['prediction']
                weight = pred['weight']
                
                if direction not in direction_votes:
                    direction_votes[direction] = 0
                direction_votes[direction] += weight
                total_weight += weight
            
            # Normalize votes
            for direction in direction_votes:
                direction_votes[direction] /= total_weight
            
            # Get winning direction
            ensemble_direction = max(direction_votes.items(), key=lambda x: x[1])[0]
            confidence = direction_votes[ensemble_direction]
            
            result = PredictionResult(
                prediction_type=PredictionType.DIRECTION,
                symbol=symbol,
                timeframe=timeframe,
                prediction=ensemble_direction,
                confidence=confidence,
                probability_distribution=direction_votes,
                feature_importance=await self._calculate_feature_importance(
                    model_predictions, features
                ),
                model_consensus=self._calculate_model_consensus(model_predictions),
                timestamp=datetime.now(),
                horizon_minutes=horizon_minutes,
                metadata={
                    'direction_votes': direction_votes,
                    'total_models': len(model_predictions)
                }
            )
            
            self.prediction_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Direction prediction failed for {symbol}: {e}")
            raise
    
    async def predict_volatility(self, symbol: str, features: np.ndarray, 
                               timeframe: str = '1m', horizon_minutes: int = 5) -> PredictionResult:
        """Predict future volatility."""
        try:
            cache_key = f"volatility_{symbol}_{timeframe}_{horizon_minutes}"
            
            if self._is_cached_valid(cache_key):
                return self.prediction_cache[cache_key]
            
            # Get volatility predictions
            model_predictions = []
            for model_name, model_info in self.active_models.items():
                try:
                    pred = await self._predict_volatility_with_model(
                        model_info, features
                    )
                    if pred is not None:
                        model_predictions.append({
                            'model': model_name,
                            'prediction': pred,
                            'weight': model_info['weight']
                        })
                except Exception as e:
                    self.logger.warning(f"Volatility prediction failed for {model_name}: {e}")
            
            if not model_predictions:
                raise ValueError("No models available for volatility prediction")
            
            # Calculate weighted average volatility
            ensemble_vol = sum(
                pred['prediction'] * pred['weight'] 
                for pred in model_predictions
            ) / sum(pred['weight'] for pred in model_predictions)
            
            # Calculate confidence based on agreement
            vol_values = [pred['prediction'] for pred in model_predictions]
            confidence = 1.0 - (np.std(vol_values) / np.mean(vol_values)) if np.mean(vol_values) > 0 else 0.0
            confidence = max(0.0, min(1.0, confidence))
            
            result = PredictionResult(
                prediction_type=PredictionType.VOLATILITY,
                symbol=symbol,
                timeframe=timeframe,
                prediction=ensemble_vol,
                confidence=confidence,
                probability_distribution=None,
                feature_importance=await self._calculate_feature_importance(
                    model_predictions, features
                ),
                model_consensus=self._calculate_model_consensus(model_predictions),
                timestamp=datetime.now(),
                horizon_minutes=horizon_minutes,
                metadata={
                    'volatility_range': [min(vol_values), max(vol_values)],
                    'model_std': np.std(vol_values)
                }
            )
            
            self.prediction_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Volatility prediction failed for {symbol}: {e}")
            raise
    
    async def ensemble_predict(self, symbol: str, features: np.ndarray, 
                             prediction_types: List[PredictionType],
                             timeframe: str = '1m', 
                             horizon_minutes: int = 5) -> EnsemblePrediction:
        """Generate ensemble predictions across multiple prediction types."""
        try:
            predictions = []
            
            # Get predictions for each type
            for pred_type in prediction_types:
                if pred_type == PredictionType.PRICE:
                    pred = await self.predict_price(symbol, features, timeframe, horizon_minutes)
                elif pred_type == PredictionType.DIRECTION:
                    pred = await self.predict_direction(symbol, features, timeframe, horizon_minutes)
                elif pred_type == PredictionType.VOLATILITY:
                    pred = await self.predict_volatility(symbol, features, timeframe, horizon_minutes)
                else:
                    continue
                
                predictions.append(pred)
            
            if not predictions:
                raise ValueError("No valid predictions generated")
            
            # Select primary prediction (highest confidence)
            primary_pred = max(predictions, key=lambda x: x.confidence)
            
            # Calculate consensus metrics
            avg_confidence = np.mean([p.confidence for p in predictions])
            confidence_std = np.std([p.confidence for p in predictions])
            
            consensus_score = avg_confidence * (1.0 - confidence_std)
            disagreement_score = confidence_std
            
            # Weight distribution
            total_confidence = sum(p.confidence for p in predictions)
            weight_dist = {
                p.prediction_type.value: p.confidence / total_confidence 
                for p in predictions
            }
            
            # Risk-adjusted prediction (conservative approach)
            risk_adjustment = 1.0 - disagreement_score
            risk_adjusted_pred = primary_pred.prediction
            
            if primary_pred.prediction_type == PredictionType.PRICE:
                risk_adjusted_pred *= risk_adjustment
            
            ensemble = EnsemblePrediction(
                primary_prediction=primary_pred,
                model_predictions=predictions,
                consensus_score=consensus_score,
                disagreement_score=disagreement_score,
                weight_distribution=weight_dist,
                risk_adjusted_prediction=risk_adjusted_pred
            )
            
            return ensemble
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed for {symbol}: {e}")
            raise
    
    async def update_model_performance(self, model_name: str, actual_values: List[float], 
                                     predicted_values: List[float], 
                                     prediction_type: PredictionType):
        """Update model performance metrics."""
        try:
            if model_name not in self.active_models:
                return
            
            # Calculate metrics based on prediction type
            if prediction_type == PredictionType.DIRECTION:
                # Classification accuracy
                accuracy = accuracy_score(actual_values, predicted_values)
                self.prediction_performance['accuracy_scores'][model_name] = accuracy
                
                # Update model weight based on performance
                self.active_models[model_name]['accuracy_score'] = accuracy
                
            else:
                # Regression metrics
                mse = mean_squared_error(actual_values, predicted_values)
                mae = mean_absolute_error(actual_values, predicted_values)
                
                self.prediction_performance['mse_scores'][model_name] = mse
                self.prediction_performance['mae_scores'][model_name] = mae
                
                # Update model weight (inverse of error)
                error_score = 1.0 / (1.0 + mse)
                self.active_models[model_name]['accuracy_score'] = error_score
            
            # Adaptive weight adjustment
            await self._update_model_weights()
            
            self.logger.info(f"Updated performance for model {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to update model performance: {e}")
    
    async def _predict_with_model(self, model_info: Dict, features: np.ndarray, 
                                pred_type: PredictionType) -> Optional[float]:
        """Make prediction with a specific model."""
        try:
            model = model_info['model']
            model_type = model_info['type']
            
            if model_type == 'tensorflow' and HAS_TF:
                prediction = model.predict(features.reshape(1, -1))[0][0]
            elif model_type == 'pytorch' and HAS_TORCH:
                with torch.no_grad():
                    tensor_input = torch.FloatTensor(features).unsqueeze(0)
                    prediction = model(tensor_input).item()
            elif model_type == 'sklearn':
                prediction = model.predict(features.reshape(1, -1))[0]
            else:
                return None
            
            return float(prediction)
            
        except Exception as e:
            self.logger.warning(f"Model prediction failed: {e}")
            return None
    
    async def _predict_direction_with_model(self, model_info: Dict, 
                                          features: np.ndarray) -> Optional[str]:
        """Predict direction with a specific model."""
        try:
            model = model_info['model']
            model_type = model_info['type']
            
            if hasattr(model, 'predict_proba'):
                # Classification model with probabilities
                proba = model.predict_proba(features.reshape(1, -1))[0]
                classes = model.classes_
                
                if len(classes) == 3:  # up, down, sideways
                    max_idx = np.argmax(proba)
                    return classes[max_idx]
                elif len(classes) == 2:  # up, down
                    return 'up' if proba[1] > 0.5 else 'down'
            else:
                # Regression model - convert to direction
                pred = await self._predict_with_model(model_info, features, PredictionType.DIRECTION)
                if pred is None:
                    return None
                
                if pred > 0.01:
                    return 'up'
                elif pred < -0.01:
                    return 'down'
                else:
                    return 'sideways'
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Direction prediction failed: {e}")
            return None
    
    async def _predict_volatility_with_model(self, model_info: Dict, 
                                           features: np.ndarray) -> Optional[float]:
        """Predict volatility with a specific model."""
        try:
            # Use standard prediction method
            vol_pred = await self._predict_with_model(
                model_info, features, PredictionType.VOLATILITY
            )
            
            # Ensure non-negative volatility
            if vol_pred is not None:
                vol_pred = max(0.0, vol_pred)
            
            return vol_pred
            
        except Exception as e:
            self.logger.warning(f"Volatility prediction failed: {e}")
            return None
    
    def _calculate_ensemble_prediction(self, model_predictions: List[Dict]) -> float:
        """Calculate ensemble prediction from model outputs."""
        if self.prediction_config['ensemble_method'] == 'simple_voting':
            return np.mean([pred['prediction'] for pred in model_predictions])
        
        elif self.prediction_config['ensemble_method'] == 'weighted_voting':
            total_weight = sum(pred['weight'] for pred in model_predictions)
            weighted_sum = sum(
                pred['prediction'] * pred['weight'] 
                for pred in model_predictions
            )
            return weighted_sum / total_weight
        
        else:  # Default to weighted voting
            return self._calculate_ensemble_prediction(model_predictions)
    
    def _calculate_confidence(self, model_predictions: List[Dict], 
                            ensemble_pred: float) -> float:
        """Calculate confidence score for ensemble prediction."""
        if not model_predictions:
            return 0.0
        
        # Calculate agreement among models
        predictions = [pred['prediction'] for pred in model_predictions]
        weights = [pred['weight'] for pred in model_predictions]
        
        # Weighted standard deviation
        mean_pred = np.average(predictions, weights=weights)
        variance = np.average((predictions - mean_pred) ** 2, weights=weights)
        std_dev = np.sqrt(variance)
        
        # Confidence inversely related to disagreement
        if mean_pred != 0:
            coefficient_of_variation = std_dev / abs(mean_pred)
            confidence = 1.0 / (1.0 + coefficient_of_variation)
        else:
            confidence = 1.0 - std_dev
        
        return max(0.0, min(1.0, confidence))
    
    async def _calculate_feature_importance(self, model_predictions: List[Dict], 
                                          features: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance from available models."""
        try:
            importance_scores = {}
            feature_names = [f"feature_{i}" for i in range(len(features))]
            
            for pred in model_predictions:
                model_name = pred['model']
                model_info = self.active_models[model_name]
                model = model_info['model']
                
                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, importance in enumerate(importances):
                        if i < len(feature_names):
                            feat_name = feature_names[i]
                            if feat_name not in importance_scores:
                                importance_scores[feat_name] = []
                            importance_scores[feat_name].append(importance)
            
            # Average importance scores
            avg_importance = {}
            for feat_name, scores in importance_scores.items():
                avg_importance[feat_name] = np.mean(scores)
            
            return avg_importance
            
        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
            return {}
    
    def _calculate_model_consensus(self, model_predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate consensus metrics among models."""
        predictions = [pred['prediction'] for pred in model_predictions]
        
        return {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'range': np.max(predictions) - np.min(predictions),
            'agreement_score': 1.0 - (np.std(predictions) / (np.mean(predictions) + 1e-8))
        }
    
    def _is_cached_valid(self, cache_key: str) -> bool:
        """Check if cached prediction is still valid."""
        if cache_key not in self.prediction_cache:
            return False
        
        cached_result = self.prediction_cache[cache_key]
        age = (datetime.now() - cached_result.timestamp).total_seconds()
        
        return age < self.cache_ttl
    
    async def _update_model_weights(self):
        """Update model weights based on performance."""
        try:
            total_performance = 0
            model_scores = {}
            
            # Calculate performance scores
            for model_name, model_info in self.active_models.items():
                score = model_info.get('accuracy_score', 0.0)
                model_scores[model_name] = score
                total_performance += score
            
            # Update weights proportionally
            if total_performance > 0:
                for model_name in self.active_models:
                    score = model_scores[model_name]
                    new_weight = score / total_performance
                    self.active_models[model_name]['weight'] = new_weight
            
            # Save weight history
            self.prediction_performance['model_weights_history'].append({
                'timestamp': datetime.now(),
                'weights': {name: info['weight'] for name, info in self.active_models.items()}
            })
            
        except Exception as e:
            self.logger.error(f"Failed to update model weights: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models."""
        return {
            'active_models': len(self.active_models),
            'prediction_cache_size': len(self.prediction_cache),
            'accuracy_scores': self.prediction_performance['accuracy_scores'],
            'mse_scores': self.prediction_performance['mse_scores'],
            'mae_scores': self.prediction_performance['mae_scores'],
            'model_weights': {
                name: info['weight'] for name, info in self.active_models.items()
            },
            'cache_hit_rate': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        }