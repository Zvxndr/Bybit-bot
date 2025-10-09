"""
Core Trading Bot Module
Provides base classes and utilities for the trading system
"""

from .strategy_manager import BaseStrategy, TradingSignal, SignalType

__all__ = ['BaseStrategy', 'TradingSignal', 'SignalType']