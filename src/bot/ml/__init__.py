"""
Machine Learning package for trading strategies.

This package provides comprehensive ML capabilities including:
- Feature engineering with technical indicators and transformations
- Market regime detection using multiple methodologies
- Advanced ML models (LightGBM, XGBoost) optimized for finance
- Sophisticated ensemble methods with dynamic weighting
- Proper financial ML validation techniques
- Model interpretability and performance monitoring

All components are designed specifically for financial applications
with proper handling of time series data and prevention of look-ahead bias.
"""

from .features import FeatureEngineer, TechnicalIndicators
from .regimes import (
    HMMRegimeDetector,
    VolatilityRegimeDetector, 
    TrendRegimeDetector,
    MultiFactorRegimeDetector,
    RegimeAnalyzer,
    RegimeInfo
)
from .models import (
    LightGBMTrader,
    XGBoostTrader,
    EnsembleTrader,
    MLModelFactory,
    ModelResult
)
from .ensemble import (
    DynamicEnsemble,
    StackingEnsemble,
    EnsembleResult,
    EnsembleWeight
)

__all__ = [
    # Feature Engineering
    'FeatureEngineer',
    'TechnicalIndicators',
    
    # Regime Detection
    'HMMRegimeDetector',
    'VolatilityRegimeDetector',
    'TrendRegimeDetector', 
    'MultiFactorRegimeDetector',
    'RegimeAnalyzer',
    'RegimeInfo',
    
    # ML Models
    'LightGBMTrader',
    'XGBoostTrader',
    'EnsembleTrader',
    'MLModelFactory',
    'ModelResult',
    
    # Advanced Ensembles
    'DynamicEnsemble',
    'StackingEnsemble',
    'EnsembleResult',
    'EnsembleWeight',
]