"""
Trading Engine Integration Package
Australian-compliant trading engine integration with ML strategies and arbitrage
"""

from .australian_trading_engine import (
    AustralianTradingEngineIntegration,
    AustralianComplianceExecutor,
    AustralianOrderRouter,
    MLStrategyExecutor,
    ArbitrageExecutor,
    AustralianTradeRequest,
    AustralianExecutionResult,
    ExecutionPriority,
    TradeSource
)

from .signal_processing_manager import (
    SignalProcessingManager,
    SignalConflictResolver,
    ProcessingSignal,
    SignalConflictType,
    SignalStatus
)

from .integration_coordinator import (
    AustralianTradingSystemCoordinator,
    SystemConfiguration,
    SystemStatus
)

__all__ = [
    # Main integration components
    'AustralianTradingEngineIntegration',
    'SignalProcessingManager', 
    'AustralianTradingSystemCoordinator',
    
    # Execution components
    'AustralianComplianceExecutor',
    'AustralianOrderRouter',
    'MLStrategyExecutor',
    'ArbitrageExecutor',
    
    # Signal processing components
    'SignalConflictResolver',
    'ProcessingSignal',
    
    # Data structures
    'AustralianTradeRequest',
    'AustralianExecutionResult',
    'SystemConfiguration',
    'SystemStatus',
    
    # Enums
    'ExecutionPriority',
    'TradeSource',
    'SignalConflictType',
    'SignalStatus'
]

# Package version
__version__ = "1.0.0"

# Package metadata
__author__ = "Australian Trading System"
__description__ = "Comprehensive Australian-compliant cryptocurrency trading engine integration"
__license__ = "Private"