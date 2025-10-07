"""
ML Integration Package - Unified ML Trading System

This package provides a complete ML-enhanced trading system that integrates
machine learning capabilities into the main trading loop. It combines both
ML packages (ml/ and machine_learning/) into a unified system with:

- Feature Pipeline: Comprehensive feature engineering
- Model Manager: ML model coordination and ensemble predictions  
- Strategy Orchestrator: ML + traditional strategy combination
- Execution Optimizer: ML-enhanced order execution
- Performance Monitor: Real-time performance tracking
- Integration Controller: Main system orchestrator

The system is designed to be production-ready with proper error handling,
performance monitoring, risk management integration, and adaptive learning.
"""

# Core ML Integration Components
from .ml_feature_pipeline import (
    MLFeaturePipeline,
    MLFeatures,
    MLPrediction,
    MLSignalType,
    MLConfidenceLevel
)

from .ml_model_manager import (
    MLModelManager,
    ModelType,
    ModelStatus,
    ModelMetadata,
    EnsemblePrediction
)

from .ml_strategy_orchestrator import (
    MLStrategyOrchestrator,
    StrategyType,
    MarketRegime,
    SignalStrength,
    TraditionalSignal,
    CombinedSignal,
    StrategyPerformance
)

from .ml_execution_optimizer import (
    MLExecutionOptimizer,
    ExecutionStrategy,
    LiquidityCondition,
    MarketImpactLevel,
    ExecutionPrediction,
    OrderSlice,
    ExecutionPlan,
    ExecutionPerformance
)

from .ml_performance_monitor import (
    MLPerformanceMonitor,
    PerformanceMetric,
    AlertLevel,
    PredictionOutcome,
    TradePerformance,
    ModelPerformanceMetrics,
    PerformanceAlert,
    PerformanceReport
)

from .ml_integration_controller import (
    MLIntegrationController,
    SystemStatus,
    TradingMode,
    TradingDecision,
    SystemHealthMetrics
)

# Version info
__version__ = "1.0.0"
__author__ = "Bybit Trading Bot ML Team"

# Package metadata
__all__ = [
    # Feature Pipeline
    'MLFeaturePipeline',
    'MLFeatures', 
    'MLPrediction',
    'MLSignalType',
    'MLConfidenceLevel',
    
    # Model Manager
    'MLModelManager',
    'ModelType',
    'ModelStatus',
    'ModelMetadata',
    'EnsemblePrediction',
    
    # Strategy Orchestrator
    'MLStrategyOrchestrator',
    'StrategyType',
    'MarketRegime',
    'SignalStrength',
    'TraditionalSignal',
    'CombinedSignal',
    'StrategyPerformance',
    
    # Execution Optimizer
    'MLExecutionOptimizer',
    'ExecutionStrategy',
    'LiquidityCondition',
    'MarketImpactLevel',
    'ExecutionPrediction',
    'OrderSlice',
    'ExecutionPlan',
    'ExecutionPerformance',
    
    # Performance Monitor
    'MLPerformanceMonitor',
    'PerformanceMetric',
    'AlertLevel',
    'PredictionOutcome',
    'TradePerformance',
    'ModelPerformanceMetrics',
    'PerformanceAlert',
    'PerformanceReport',
    
    # Integration Controller
    'MLIntegrationController',
    'SystemStatus',
    'TradingMode',
    'TradingDecision',
    'SystemHealthMetrics'
]

# Convenience imports for common use cases
def create_ml_trading_system(config_path: str = None) -> MLIntegrationController:
    """
    Create a complete ML trading system with all components
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        MLIntegrationController instance with all components initialized
    """
    return MLIntegrationController(config_path)

def create_feature_pipeline(config: dict = None) -> MLFeaturePipeline:
    """Create ML feature pipeline"""
    return MLFeaturePipeline(config)

def create_model_manager(config: dict = None) -> MLModelManager:
    """Create ML model manager"""
    return MLModelManager(config)

def create_strategy_orchestrator(model_manager: MLModelManager, 
                               config: dict = None) -> MLStrategyOrchestrator:
    """Create ML strategy orchestrator"""
    return MLStrategyOrchestrator(model_manager, config)

def create_execution_optimizer(config: dict = None) -> MLExecutionOptimizer:
    """Create ML execution optimizer"""  
    return MLExecutionOptimizer(config)

def create_performance_monitor(config: dict = None) -> MLPerformanceMonitor:
    """Create ML performance monitor"""
    return MLPerformanceMonitor(config)

# Package documentation
DOCUMENTATION = {
    'overview': '''
    ML Integration Package provides a complete machine learning enhanced trading system
    that seamlessly integrates ML capabilities into the main trading loop.
    ''',
    
    'architecture': '''
    The package follows a modular architecture:
    
    1. MLFeaturePipeline: Generates comprehensive ML features from market data
    2. MLModelManager: Manages multiple ML models and creates ensemble predictions
    3. MLStrategyOrchestrator: Combines ML predictions with traditional strategies
    4. MLExecutionOptimizer: Optimizes trade execution using ML
    5. MLPerformanceMonitor: Tracks and analyzes ML system performance
    6. MLIntegrationController: Main orchestrator that coordinates all components
    ''',
    
    'usage_example': '''
    # Create complete ML trading system
    ml_system = create_ml_trading_system("config.json")
    
    # Generate trading decision
    market_data = {"price": 50000, "volume": 1000000}
    decision = await ml_system.generate_trading_decision("BTCUSD", market_data)
    
    # Execute decision if appropriate
    if decision.recommended_action.startswith("EXECUTE"):
        result = await ml_system.execute_decision(decision)
    ''',
    
    'key_features': [
        'Unified ML and traditional strategy integration',
        'Real-time performance monitoring and alerts',
        'Advanced execution optimization',
        'Comprehensive risk management integration',
        'Adaptive learning and model rebalancing',
        'Production-ready error handling and recovery',
        'Extensive logging and debugging support'
    ]
}

def get_package_info() -> dict:
    """Get comprehensive package information"""
    return {
        'version': __version__,
        'author': __author__,
        'components': len(__all__),
        'documentation': DOCUMENTATION
    }

# Initialize logging for the package
import logging
logger = logging.getLogger(__name__)
logger.info(f"ML Integration Package v{__version__} loaded successfully")