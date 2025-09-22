"""
Advanced Analytics Package - Phase 6 Implementation

This package provides sophisticated analytics capabilities for institutional-grade trading:
- Performance Attribution: Factor-based performance analysis
- Risk Analytics: VaR, stress testing, scenario analysis
- Regime Detection: Market regime identification and adaptation
- Factor Analysis: Multi-factor model analysis and decomposition
- Predictive Analytics: Forward-looking performance and risk metrics
- Portfolio Analytics: Multi-asset portfolio analysis and optimization
- Quantitative Research: Statistical analysis and research tools

Author: Trading Bot Team
Version: 1.0.0 - Phase 6 Implementation
"""

from .performance_attribution import (
    PerformanceAttributor,
    AttributionResult,
    FactorContribution
)
from .risk_analytics import (
    RiskAnalyzer,
    VaRCalculator,
    StressTester,
    RiskMetrics
)
from .regime_detector import (
    RegimeDetector,
    MarketRegime,
    RegimeTransition
)
from .factor_analyzer import (
    FactorAnalyzer,
    FactorModel,
    FactorLoadings
)

__all__ = [
    'PerformanceAttributor',
    'AttributionResult',
    'FactorContribution',
    'RiskAnalyzer',
    'VaRCalculator',
    'StressTester',
    'RiskMetrics',
    'RegimeDetector',
    'MarketRegime',
    'RegimeTransition',
    'FactorAnalyzer',
    'FactorModel',
    'FactorLoadings',
]