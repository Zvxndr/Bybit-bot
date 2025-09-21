"""
Backtesting and validation package for the trading bot.

This package provides comprehensive strategy validation including:
- Walk-Forward Analysis (WFO)
- Combinatorial Symmetric Cross-Validation (CSCV)
- Permutation testing for statistical significance
- Purged TimeSeries Cross-Validation for ML models
- Mode-specific validation thresholds
"""

from .walkforward import WalkForwardAnalyzer, WalkForwardResult
from .permutation import PermutationTester, PermutationResult
from .cscv import CSCVValidator, CSCVResult
from .purged_cv import PurgedTimeSeriesCV, PurgedTimeSeriesSplit, PurgedCVResult
from .validator import StrategyValidator, ValidationResult, ValidationThresholds

__all__ = [
    "WalkForwardAnalyzer",
    "WalkForwardResult",
    "PermutationTester", 
    "PermutationResult",
    "CSCVValidator",
    "CSCVResult",
    "PurgedTimeSeriesCV",
    "PurgedTimeSeriesSplit",
    "PurgedCVResult",
    "StrategyValidator",
    "ValidationResult",
    "ValidationThresholds",
]