"""
ML Model Manager - Unified ML Prediction System

This module manages predictions from multiple ML models and systems, combining
capabilities from both ml/ and machine_learning/ packages into a unified
prediction engine for the trading system.

Key Features:
- Manages multiple ML models (LightGBM, XGBoost, Neural Networks)
- Ensemble predictions with dynamic weighting
- Model performance monitoring and adaptive selection
- Confidence-based prediction filtering
- Real-time model inference with caching
- Model health monitoring and automatic fallbacks
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass
import json
from pathlib import Path

# Import ML components
try:
    from ..ml import (
        LightGBMTrader, XGBoostTrader, EnsembleTrader, DynamicEnsemble,
        RegimeAnalyzer, MLModelFactory
    )
    ML_PACKAGE_AVAILABLE = True
except ImportError:
    logger.warning("ml package not available")
    ML_PACKAGE_AVAILABLE = False

try:
    from ..machine_learning import (
        MLEngine, PredictionEngine, ModelManager as MLModelManagerBase
    )
    MACHINE_LEARNING_PACKAGE_AVAILABLE = True
except ImportError:
    logger.warning("machine_learning package not available")
    MACHINE_LEARNING_PACKAGE_AVAILABLE = False

from .ml_feature_pipeline import MLFeatures, MLPrediction, MLSignalType, MLConfidenceLevel

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class ModelType(Enum):
    """Types of ML models supported"""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    REGIME_AWARE = "regime_aware"

class ModelStatus(Enum):
    """Status of ML models"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class ModelMetadata:
    """Metadata for ML models"""
    model_id: str
    model_type: ModelType
    version: str
    created_at: datetime
    last_updated: datetime
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    training_data_period: Tuple[datetime, datetime]
    status: ModelStatus

@dataclass
class EnsemblePrediction:
    """Ensemble prediction combining multiple models"""
    final_prediction: MLPrediction
    individual_predictions: List[MLPrediction]
    model_weights: Dict[str, float]
    consensus_strength: float
    uncertainty: float

# ============================================================================
# ML MODEL MANAGER
# ============================================================================

class MLModelManager:
    """
    Unified ML Model Manager for trading predictions
    
    Coordinates multiple ML models and provides unified prediction interface
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Model storage
        self.active_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self.model_performance: Dict[str, List[float]] = {}
        
        # Ensemble components
        self.ensemble_engine = None
        self.regime_analyzer = None
        
        # Prediction cache
        self.prediction_cache: Dict[str, Tuple[EnsemblePrediction, datetime]] = {}
        self.cache_ttl = timedelta(seconds=30)
        
        # Performance tracking
        self.performance_history = []
        self.prediction_history = []
        
        # Initialize models
        asyncio.create_task(self._initialize_models())
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ML Model Manager"""
        return {
            'models': {
                'lightgbm_trend': {
                    'enabled': True,
                    'weight': 0.3,
                    'confidence_threshold': 0.6
                },
                'lightgbm_mean_reversion': {
                    'enabled': True,
                    'weight': 0.25,
                    'confidence_threshold': 0.6
                },
                'xgboost_momentum': {
                    'enabled': True,
                    'weight': 0.25,
                    'confidence_threshold': 0.65
                },
                'neural_network': {
                    'enabled': False,  # Disabled by default
                    'weight': 0.2,
                    'confidence_threshold': 0.7
                }
            },
            'ensemble': {
                'method': 'dynamic_weighted',  # 'simple_average', 'weighted_average', 'dynamic_weighted'
                'confidence_weighting': True,
                'performance_weighting': True,
                'regime_aware': True
            },
            'performance': {
                'tracking_window': 100,  # Number of predictions to track
                'rebalance_frequency': 20,  # Rebalance weights every N predictions
                'min_confidence': 0.5,  # Minimum confidence for predictions
                'max_uncertainty': 0.3  # Maximum uncertainty for ensemble
            },
            'fallback': {
                'enable_fallback': True,
                'fallback_to_traditional': True,
                'error_threshold': 0.1  # 10% error rate before fallback
            }
        }
    
    async def _initialize_models(self):
        """Initialize all ML models"""
        logger.info("Initializing ML models...")
        
        try:
            if ML_PACKAGE_AVAILABLE:
                await self._initialize_ml_package_models()
                
            if MACHINE_LEARNING_PACKAGE_AVAILABLE:
                await self._initialize_machine_learning_models()
                
            # Initialize ensemble engine
            await self._initialize_ensemble_engine()
            
            logger.info(f"Initialized {len(self.active_models)} ML models")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    async def _initialize_ml_package_models(self):
        """Initialize models from the ml/ package"""
        try:
            # LightGBM models
            if self.config['models']['lightgbm_trend']['enabled']:
                lightgbm_trend = LightGBMTrader(strategy_type='trend_following')
                self.active_models['lightgbm_trend'] = lightgbm_trend
                self.model_metadata['lightgbm_trend'] = ModelMetadata(
                    model_id='lightgbm_trend',
                    model_type=ModelType.LIGHTGBM,
                    version='1.0',
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    performance_metrics={},
                    feature_importance={},
                    training_data_period=(datetime.now() - timedelta(days=90), datetime.now()),
                    status=ModelStatus.ACTIVE
                )
                
            if self.config['models']['lightgbm_mean_reversion']['enabled']:
                lightgbm_mr = LightGBMTrader(strategy_type='mean_reversion')
                self.active_models['lightgbm_mean_reversion'] = lightgbm_mr
                self.model_metadata['lightgbm_mean_reversion'] = ModelMetadata(
                    model_id='lightgbm_mean_reversion',
                    model_type=ModelType.LIGHTGBM,
                    version='1.0',
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    performance_metrics={},
                    feature_importance={},
                    training_data_period=(datetime.now() - timedelta(days=90), datetime.now()),
                    status=ModelStatus.ACTIVE
                )
            
            # XGBoost models
            if self.config['models']['xgboost_momentum']['enabled']:
                xgboost_model = XGBoostTrader(strategy_type='momentum')
                self.active_models['xgboost_momentum'] = xgboost_model
                self.model_metadata['xgboost_momentum'] = ModelMetadata(
                    model_id='xgboost_momentum',
                    model_type=ModelType.XGBOOST,
                    version='1.0',
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    performance_metrics={},
                    feature_importance={},
                    training_data_period=(datetime.now() - timedelta(days=90), datetime.now()),
                    status=ModelStatus.ACTIVE
                )
                
            logger.info("Initialized models from ml/ package")
            
        except Exception as e:
            logger.error(f"Error initializing ml/ package models: {e}")
    
    async def _initialize_machine_learning_models(self):
        """Initialize models from the machine_learning/ package"""
        try:
            # ML Engine
            ml_engine = MLEngine()
            self.active_models['ml_engine'] = ml_engine
            
            # Prediction Engine
            prediction_engine = PredictionEngine()
            self.active_models['prediction_engine'] = prediction_engine
            
            logger.info("Initialized models from machine_learning/ package")
            
        except Exception as e:
            logger.error(f"Error initializing machine_learning/ package models: {e}")
    
    async def _initialize_ensemble_engine(self):
        """Initialize ensemble prediction engine"""
        try:
            if DynamicEnsemble and len(self.active_models) > 1:
                self.ensemble_engine = DynamicEnsemble(
                    models=list(self.active_models.values()),
                    weighting_method=self.config['ensemble']['method']
                )
                logger.info("Initialized ensemble engine")
                
        except Exception as e:
            logger.error(f"Error initializing ensemble engine: {e}")
    
    async def predict(self, features: MLFeatures, symbol: str) -> EnsemblePrediction:
        """
        Get ensemble prediction from all active models
        
        Args:
            features: ML features from feature pipeline
            symbol: Trading symbol
            
        Returns:
            EnsemblePrediction with combined model outputs
        """
        # Check cache first
        cache_key = f"{symbol}_{features.timestamp.strftime('%Y%m%d_%H%M%S')}"
        if cache_key in self.prediction_cache:
            cached_prediction, cache_time = self.prediction_cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                return cached_prediction
        
        individual_predictions = []
        
        # Get predictions from all active models
        for model_name, model in self.active_models.items():
            try:
                if self.model_metadata[model_name].status != ModelStatus.ACTIVE:
                    continue
                    
                prediction = await self._get_model_prediction(model_name, model, features, symbol)
                if prediction:
                    individual_predictions.append(prediction)
                    
            except Exception as e:
                logger.error(f"Error getting prediction from {model_name}: {e}")
                await self._handle_model_error(model_name, e)
        
        if not individual_predictions:
            logger.warning("No valid predictions from any model")
            return self._create_empty_ensemble_prediction(symbol)
        
        # Create ensemble prediction
        ensemble_prediction = await self._create_ensemble_prediction(
            individual_predictions, symbol
        )
        
        # Cache the result
        self.prediction_cache[cache_key] = (ensemble_prediction, datetime.now())
        
        # Update performance tracking
        await self._update_performance_tracking(ensemble_prediction)
        
        return ensemble_prediction
    
    async def _get_model_prediction(self, model_name: str, model: Any, 
                                  features: MLFeatures, symbol: str) -> Optional[MLPrediction]:
        """Get prediction from a specific model"""
        try:
            # Convert features to format expected by model
            feature_dict = self._convert_features_for_model(features, model_name)
            
            # Get prediction based on model type
            if hasattr(model, 'predict'):
                raw_prediction = await self._safe_model_predict(model, feature_dict)
            else:
                logger.warning(f"Model {model_name} does not have predict method")
                return None
            
            if raw_prediction is None:
                return None
            
            # Convert raw prediction to MLPrediction
            ml_prediction = self._convert_raw_prediction(raw_prediction, model_name, symbol)
            
            # Apply confidence threshold
            min_confidence = self.config['models'].get(model_name, {}).get('confidence_threshold', 0.5)
            if ml_prediction.confidence < min_confidence:
                logger.debug(f"Prediction from {model_name} below confidence threshold: {ml_prediction.confidence:.3f}")
                return None
            
            return ml_prediction
            
        except Exception as e:
            logger.error(f"Error getting prediction from {model_name}: {e}")
            return None
    
    async def _safe_model_predict(self, model: Any, features: Dict[str, Any]) -> Any:
        """Safely call model prediction with timeout"""
        try:
            # Implement timeout for model prediction
            prediction = await asyncio.wait_for(
                self._call_model_predict(model, features),
                timeout=5.0  # 5 second timeout
            )
            return prediction
        except asyncio.TimeoutError:
            logger.error("Model prediction timed out")
            return None
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return None
    
    async def _call_model_predict(self, model: Any, features: Dict[str, Any]) -> Any:
        """Call model predict method (async wrapper)"""
        # Handle both sync and async predict methods
        if asyncio.iscoroutinefunction(model.predict):
            return await model.predict(features)
        else:
            # Run sync prediction in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, model.predict, features)
    
    def _convert_features_for_model(self, features: MLFeatures, model_name: str) -> Dict[str, Any]:
        """Convert MLFeatures to format expected by specific model"""
        # Combine all feature dictionaries
        all_features = {}
        all_features.update(features.technical_indicators)
        all_features.update(features.microstructure_features)
        all_features.update(features.regime_features)
        all_features.update(features.cross_asset_features)
        all_features.update(features.alternative_data)
        
        # Model-specific feature selection/transformation could go here
        # For now, return all features
        return all_features
    
    def _convert_raw_prediction(self, raw_prediction: Any, model_name: str, symbol: str) -> MLPrediction:
        """Convert raw model prediction to MLPrediction format"""
        
        # Handle different prediction formats
        if isinstance(raw_prediction, dict):
            signal_type = self._extract_signal_type(raw_prediction)
            confidence = raw_prediction.get('confidence', 0.5)
            probability_dist = raw_prediction.get('probability_distribution', {})
            feature_importance = raw_prediction.get('feature_importance', {})
            expected_return = raw_prediction.get('expected_return')
            expected_volatility = raw_prediction.get('expected_volatility')
            
        elif hasattr(raw_prediction, 'prediction'):
            # Handle structured prediction objects
            signal_type = self._extract_signal_type({'prediction': raw_prediction.prediction})
            confidence = getattr(raw_prediction, 'confidence', 0.5)
            probability_dist = getattr(raw_prediction, 'probability_distribution', {})
            feature_importance = getattr(raw_prediction, 'feature_importance', {})
            expected_return = getattr(raw_prediction, 'expected_return', None)
            expected_volatility = getattr(raw_prediction, 'expected_volatility', None)
            
        else:
            # Handle simple numeric predictions
            signal_type = self._numeric_to_signal_type(raw_prediction)
            confidence = 0.5  # Default confidence
            probability_dist = {}
            feature_importance = {}
            expected_return = None
            expected_volatility = None
        
        return MLPrediction(
            signal_type=signal_type,
            confidence=confidence,
            probability_distribution=probability_dist,
            feature_importance=feature_importance,
            model_name=model_name,
            timestamp=datetime.now(),
            expected_return=expected_return,
            expected_volatility=expected_volatility
        )
    
    def _extract_signal_type(self, prediction_dict: Dict[str, Any]) -> MLSignalType:
        """Extract signal type from prediction dictionary"""
        prediction = prediction_dict.get('prediction', 0)
        
        if isinstance(prediction, str):
            return MLSignalType(prediction.lower())
        elif isinstance(prediction, (int, float)):
            return self._numeric_to_signal_type(prediction)
        else:
            return MLSignalType.HOLD
    
    def _numeric_to_signal_type(self, value: Union[int, float]) -> MLSignalType:
        """Convert numeric prediction to signal type"""
        if value > 0.7:
            return MLSignalType.STRONG_BUY
        elif value > 0.3:
            return MLSignalType.BUY
        elif value < -0.7:
            return MLSignalType.STRONG_SELL
        elif value < -0.3:
            return MLSignalType.SELL
        else:
            return MLSignalType.HOLD
    
    async def _create_ensemble_prediction(self, predictions: List[MLPrediction], 
                                        symbol: str) -> EnsemblePrediction:
        """Create ensemble prediction from individual model predictions"""
        
        if not predictions:
            return self._create_empty_ensemble_prediction(symbol)
        
        # Calculate model weights
        model_weights = await self._calculate_model_weights(predictions)
        
        # Calculate weighted signal
        weighted_signals = {}
        total_weight = 0
        
        for prediction in predictions:
            weight = model_weights.get(prediction.model_name, 0)
            signal_value = self._signal_to_numeric(prediction.signal_type)
            confidence_weight = weight * prediction.confidence
            
            if prediction.signal_type.value not in weighted_signals:
                weighted_signals[prediction.signal_type.value] = 0
            
            weighted_signals[prediction.signal_type.value] += confidence_weight
            total_weight += confidence_weight
        
        # Determine final signal
        if total_weight == 0:
            final_signal_type = MLSignalType.HOLD
            final_confidence = 0.0
        else:
            # Normalize weights
            for signal in weighted_signals:
                weighted_signals[signal] /= total_weight
            
            # Get strongest signal
            final_signal = max(weighted_signals.items(), key=lambda x: x[1])
            final_signal_type = MLSignalType(final_signal[0])
            final_confidence = final_signal[1]
        
        # Calculate consensus strength
        consensus_strength = self._calculate_consensus_strength(predictions)
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(predictions, consensus_strength)
        
        # Create combined feature importance
        combined_feature_importance = self._combine_feature_importance(predictions, model_weights)
        
        # Create final prediction
        final_prediction = MLPrediction(
            signal_type=final_signal_type,
            confidence=final_confidence,
            probability_distribution=weighted_signals,
            feature_importance=combined_feature_importance,
            model_name='ensemble',
            timestamp=datetime.now(),
            expected_return=np.mean([p.expected_return for p in predictions if p.expected_return]),
            expected_volatility=np.mean([p.expected_volatility for p in predictions if p.expected_volatility])
        )
        
        return EnsemblePrediction(
            final_prediction=final_prediction,
            individual_predictions=predictions,
            model_weights=model_weights,
            consensus_strength=consensus_strength,
            uncertainty=uncertainty
        )
    
    async def _calculate_model_weights(self, predictions: List[MLPrediction]) -> Dict[str, float]:
        """Calculate dynamic weights for each model based on performance"""
        weights = {}
        
        # Base weights from configuration
        for prediction in predictions:
            model_name = prediction.model_name
            base_weight = self.config['models'].get(model_name, {}).get('weight', 1.0)
            weights[model_name] = base_weight
        
        # Adjust weights based on recent performance
        if self.config['ensemble']['performance_weighting']:
            for prediction in predictions:
                model_name = prediction.model_name
                if model_name in self.model_performance:
                    recent_performance = np.mean(self.model_performance[model_name][-10:])  # Last 10 predictions
                    performance_multiplier = max(0.1, min(2.0, recent_performance))  # Clamp between 0.1 and 2.0
                    weights[model_name] *= performance_multiplier
        
        # Adjust weights based on confidence
        if self.config['ensemble']['confidence_weighting']:
            for prediction in predictions:
                model_name = prediction.model_name
                confidence_multiplier = prediction.confidence
                weights[model_name] *= confidence_multiplier
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] /= total_weight
        
        return weights
    
    def _signal_to_numeric(self, signal_type: MLSignalType) -> float:
        """Convert signal type to numeric value"""
        signal_map = {
            MLSignalType.STRONG_SELL: -1.0,
            MLSignalType.SELL: -0.5,
            MLSignalType.HOLD: 0.0,
            MLSignalType.BUY: 0.5,
            MLSignalType.STRONG_BUY: 1.0
        }
        return signal_map.get(signal_type, 0.0)
    
    def _calculate_consensus_strength(self, predictions: List[MLPrediction]) -> float:
        """Calculate consensus strength among predictions"""
        if len(predictions) <= 1:
            return 1.0
        
        # Count predictions by signal type
        signal_counts = {}
        for prediction in predictions:
            signal = prediction.signal_type
            if signal not in signal_counts:
                signal_counts[signal] = 0
            signal_counts[signal] += prediction.confidence
        
        # Calculate consensus as ratio of dominant signal
        total_confidence = sum(signal_counts.values())
        max_signal_confidence = max(signal_counts.values())
        
        return max_signal_confidence / total_confidence if total_confidence > 0 else 0.0
    
    def _calculate_uncertainty(self, predictions: List[MLPrediction], consensus_strength: float) -> float:
        """Calculate prediction uncertainty"""
        # Base uncertainty from consensus
        consensus_uncertainty = 1.0 - consensus_strength
        
        # Add uncertainty from individual model confidence
        avg_confidence = np.mean([p.confidence for p in predictions])
        confidence_uncertainty = 1.0 - avg_confidence
        
        # Combine uncertainties
        combined_uncertainty = (consensus_uncertainty + confidence_uncertainty) / 2
        
        return min(1.0, combined_uncertainty)
    
    def _combine_feature_importance(self, predictions: List[MLPrediction], 
                                  model_weights: Dict[str, float]) -> Dict[str, float]:
        """Combine feature importance from all models"""
        combined_importance = {}
        
        for prediction in predictions:
            model_weight = model_weights.get(prediction.model_name, 0)
            
            for feature, importance in prediction.feature_importance.items():
                if feature not in combined_importance:
                    combined_importance[feature] = 0
                combined_importance[feature] += importance * model_weight
        
        return combined_importance
    
    def _create_empty_ensemble_prediction(self, symbol: str) -> EnsemblePrediction:
        """Create empty ensemble prediction when no valid predictions available"""
        empty_prediction = MLPrediction(
            signal_type=MLSignalType.HOLD,
            confidence=0.0,
            probability_distribution={},
            feature_importance={},
            model_name='empty',
            timestamp=datetime.now()
        )
        
        return EnsemblePrediction(
            final_prediction=empty_prediction,
            individual_predictions=[],
            model_weights={},
            consensus_strength=0.0,
            uncertainty=1.0
        )
    
    async def _handle_model_error(self, model_name: str, error: Exception):
        """Handle model errors and update status"""
        logger.error(f"Model {model_name} error: {error}")
        
        if model_name in self.model_metadata:
            self.model_metadata[model_name].status = ModelStatus.ERROR
        
        # Implement error recovery logic here
        # Could include model retraining, fallback mechanisms, etc.
    
    async def _update_performance_tracking(self, ensemble_prediction: EnsemblePrediction):
        """Update performance tracking for models"""
        # Add to prediction history
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'prediction': ensemble_prediction,
            'symbol': 'tracked_symbol'  # Would be passed from caller
        })
        
        # Maintain rolling window
        max_history = self.config['performance']['tracking_window']
        if len(self.prediction_history) > max_history:
            self.prediction_history = self.prediction_history[-max_history:]
    
    async def update_model_performance(self, model_name: str, actual_outcome: float, 
                                     predicted_outcome: float):
        """Update model performance metrics with actual outcomes"""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = []
        
        # Calculate accuracy/error metric
        error = abs(actual_outcome - predicted_outcome)
        accuracy = max(0, 1 - error)  # Simple accuracy metric
        
        self.model_performance[model_name].append(accuracy)
        
        # Maintain rolling window
        max_history = self.config['performance']['tracking_window']
        if len(self.model_performance[model_name]) > max_history:
            self.model_performance[model_name] = self.model_performance[model_name][-max_history:]
        
        # Update model metadata
        if model_name in self.model_metadata:
            self.model_metadata[model_name].performance_metrics['recent_accuracy'] = np.mean(
                self.model_performance[model_name][-10:]  # Last 10 predictions
            )
            self.model_metadata[model_name].last_updated = datetime.now()
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models"""
        status = {}
        
        for model_name, metadata in self.model_metadata.items():
            status[model_name] = {
                'status': metadata.status.value,
                'performance': metadata.performance_metrics,
                'last_updated': metadata.last_updated.isoformat(),
                'active': model_name in self.active_models
            }
        
        return status
    
    async def retrain_model(self, model_name: str, training_data: pd.DataFrame) -> bool:
        """Retrain a specific model with new data"""
        try:
            if model_name not in self.active_models:
                logger.error(f"Model {model_name} not found")
                return False
            
            model = self.active_models[model_name]
            
            # Set model to training status
            if model_name in self.model_metadata:
                self.model_metadata[model_name].status = ModelStatus.TRAINING
            
            # Retrain model (implementation depends on model type)
            if hasattr(model, 'retrain'):
                await model.retrain(training_data)
            elif hasattr(model, 'fit'):
                await model.fit(training_data)
            else:
                logger.error(f"Model {model_name} does not support retraining")
                return False
            
            # Update metadata
            if model_name in self.model_metadata:
                self.model_metadata[model_name].status = ModelStatus.ACTIVE
                self.model_metadata[model_name].last_updated = datetime.now()
            
            logger.info(f"Successfully retrained model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error retraining model {model_name}: {e}")
            if model_name in self.model_metadata:
                self.model_metadata[model_name].status = ModelStatus.ERROR
            return False

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'MLModelManager',
    'ModelType',
    'ModelStatus', 
    'ModelMetadata',
    'EnsemblePrediction'
]