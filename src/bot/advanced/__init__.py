"""
Advanced Features Module for the Bybit Trading Bot

This module provides advanced trading features including:
- Market regime detection and filtering
- Portfolio optimization systems
- Automated reporting with performance analysis
- News sentiment analysis and blackout rules
- Dynamic parameter optimization

Author: Trading Bot Team
Version: 1.0.0
"""

from .regime_detector import RegimeDetector, MarketRegime
from .portfolio_optimizer import PortfolioOptimizer, OptimizationMethod
from .automated_reporter import AutomatedReporter, ReportType
from .news_analyzer import NewsAnalyzer, SentimentLevel
from .parameter_optimizer import ParameterOptimizer, OptimizationResult

__all__ = [
    'RegimeDetector',
    'MarketRegime',
    'PortfolioOptimizer',
    'OptimizationMethod',
    'AutomatedReporter',
    'ReportType',
    'NewsAnalyzer',
    'SentimentLevel',
    'ParameterOptimizer',
    'OptimizationResult'
]

__version__ = '1.0.0'