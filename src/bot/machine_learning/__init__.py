"""
Machine Learning Package - Phase 6 Implementation

This package contains advanced machine learning components for intelligent trading:
- ML Engine: Core machine learning framework with TensorFlow/PyTorch integration
- Feature Engineering: Advanced feature extraction and selection
- Model Management: Model training, validation, and deployment
- Prediction Engine: Real-time predictions and model inference
- Adaptive Learning: Online learning and model adaptation
- Model Monitoring: Performance tracking and drift detection

Author: Trading Bot Team
Version: 1.0.0 - Phase 6 Implementation
"""

from .ml_engine import (
    MLEngine,
    ModelType,
    PredictionResult,
    ModelPerformance
)
from .feature_engineering import (
    FeatureEngineering,
    FeatureSet,
    FeatureImportance
)
from .prediction_engine import (
    PredictionEngine,
    PredictionType,
    PredictionResult,
    EnsemblePrediction
)
from .model_manager import (
    ModelManager,
    ModelType as ModelManagerType,
    ModelStatus,
    TrainingConfig,
    ModelMetadata
)

__all__ = [
    'MLEngine',
    'ModelType',
    'PredictionResult',
    'ModelPerformance',
    'FeatureEngineering',
    'FeatureSet',
    'FeatureImportance',
    'PredictionEngine',
    'PredictionType',
    'EnsemblePrediction',
    'ModelManager',
    'ModelManagerType',
    'ModelStatus',
    'TrainingConfig',
    'ModelMetadata',
]