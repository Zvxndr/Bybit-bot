"""
ML Strategy Discovery Module
Machine learning-first approach to trading strategy development
"""

from .ml_engine import (
    MLStrategyDiscoveryEngine,
    StrategyType,
    ModelType,
    FeatureSet,
    ModelConfiguration,
    StrategySignal,
    FeatureEngineer,
    MLStrategyModel
)

from .data_infrastructure import (
    MultiExchangeDataCollector,
    TransferCostDatabase,
    AustralianDataProvider,
    ExchangeName,
    DataType,
    TransferCost,
    ExchangeInfo,
    MarketData
)

__all__ = [
    # Main engine
    'MLStrategyDiscoveryEngine',
    
    # ML components
    'StrategyType',
    'ModelType', 
    'FeatureSet',
    'ModelConfiguration',
    'StrategySignal',
    'FeatureEngineer',
    'MLStrategyModel',
    
    # Data infrastructure
    'MultiExchangeDataCollector',
    'TransferCostDatabase',
    'AustralianDataProvider',
    'ExchangeName',
    'DataType',
    'TransferCost',
    'ExchangeInfo',
    'MarketData'
]