"""
Trading Strategies Package

This package contains the trading strategy framework and concrete strategy implementations
that integrate with the unified risk management and real Bybit API trading system.

Components:
- strategy_framework: Base classes and strategy management
- ma_crossover_strategy: Moving average crossover implementation
- Additional strategies can be added following the same pattern

Key Features:
- Standardized strategy interface
- Integrated risk management
- Performance tracking and reporting
- Strategy coordination and portfolio management
"""

from .strategy_framework import (
    BaseStrategy,
    StrategyManager,
    TradingSignal,
    SignalType,
    SignalStrength,
    StrategyState,
    StrategyConfig,
    StrategyPerformance
)

from .ma_crossover_strategy import (
    MovingAverageCrossoverStrategy,
    create_btc_ma_strategy,
    create_eth_ma_strategy
)

__all__ = [
    # Framework classes
    'BaseStrategy',
    'StrategyManager', 
    'TradingSignal',
    'SignalType',
    'SignalStrength',
    'StrategyState',
    'StrategyConfig',
    'StrategyPerformance',
    
    # Concrete strategies
    'MovingAverageCrossoverStrategy',
    'create_btc_ma_strategy',
    'create_eth_ma_strategy'
]

# Default strategy factory functions
def create_default_strategy_manager(trading_engine, risk_manager, config_manager=None):
    """Create a strategy manager with default configuration."""
    return StrategyManager(trading_engine, risk_manager, config_manager)

def get_available_strategies():
    """Get list of available strategy implementations."""
    return {
        'ma_crossover': {
            'class': MovingAverageCrossoverStrategy,
            'name': 'Moving Average Crossover',
            'description': 'Classic MA crossover strategy with risk management',
            'factory_functions': {
                'btc': create_btc_ma_strategy,
                'eth': create_eth_ma_strategy
            }
        }
        # Additional strategies can be added here
    }

# These will be implemented in Phase 3
# from .base import TradingStrategy
# from .technical import TechnicalStrategy
# from .ml_strategy import MLStrategy

__all__ = [
    # "TradingStrategy",
    # "TechnicalStrategy", 
    # "MLStrategy",
]