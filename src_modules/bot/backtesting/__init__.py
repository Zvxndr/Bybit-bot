"""
Backtesting package for the Bybit trading bot.

This package provides comprehensive backtesting capabilities including:
- Basic backtesting engine (BacktestEngine)
- Enhanced Bybit-specific backtesting (BybitEnhancedBacktestEngine)
- Comprehensive fee simulation with VIP tier support
- Advanced liquidation risk management
- Realistic trade execution simulation

Phase 3 Components (Enhanced Backtesting):
- BybitEnhancedBacktestEngine: Core enhanced backtesting with Bybit features
- BybitFeeSimulator: Comprehensive fee calculation and optimization
- BybitLiquidationRiskManager: Advanced risk assessment and management
- BybitExecutionSimulator: Realistic execution modeling

Author: Trading Bot Team
Version: 1.0.0
"""

# Basic backtesting components
from .backtest_engine import BacktestEngine, BacktestTrade, BacktestResults

# Phase 3 Enhanced components
from .bybit_enhanced_backtest_engine import (
    BybitEnhancedBacktestEngine,
    BybitVIPTier,
    BybitContractType,
    BybitTrade,
    BybitBacktestResults
)

from .bybit_fee_simulator import (
    BybitFeeCalculator,
    BybitVIPTierRequirements,
    BybitFeeOptimizer,
    FeeAnalysisResult
)

from .bybit_liquidation_risk_manager import (
    BybitLiquidationRiskManager,
    LiquidationRiskAssessment,
    MarginTier,
    RiskLevel
)

from .bybit_execution_simulator import (
    BybitExecutionSimulator,
    OrderType,
    ExecutionStrategy,
    ExecutionResult,
    OrderBookSnapshot
)

__all__ = [
    # Basic components
    "BacktestEngine",
    "BacktestTrade", 
    "BacktestResults",
    
    # Enhanced backtesting
    "BybitEnhancedBacktestEngine",
    "BybitVIPTier",
    "BybitContractType", 
    "BybitTrade",
    "BybitBacktestResults",
    
    # Fee simulation
    "BybitFeeCalculator",
    "BybitVIPTierRequirements",
    "BybitFeeOptimizer",
    "FeeAnalysisResult",
    
    # Risk management
    "BybitLiquidationRiskManager",
    "LiquidationRiskAssessment",
    "MarginTier",
    "RiskLevel",
    
    # Execution simulation
    "BybitExecutionSimulator",
    "OrderType",
    "ExecutionStrategy",
    "ExecutionResult",
    "OrderBookSnapshot"
]

# Version information
__version__ = "1.0.0"
__phase__ = "3"

# Integration utilities
def create_enhanced_backtest_engine(
    config_manager,
    initial_balance=10000,
    vip_tier=BybitVIPTier.NO_VIP,
    contract_type=BybitContractType.LINEAR_PERPETUAL
) -> BybitEnhancedBacktestEngine:
    """
    Convenience function to create enhanced backtest engine.
    
    Args:
        config_manager: Configuration manager instance
        initial_balance: Initial trading balance
        vip_tier: Bybit VIP tier for fee calculations
        contract_type: Contract type for trading
        
    Returns:
        Configured BybitEnhancedBacktestEngine instance
    """
    from decimal import Decimal
    
    return BybitEnhancedBacktestEngine(
        config_manager=config_manager,
        initial_balance=Decimal(str(initial_balance)),
        vip_tier=vip_tier,
        contract_type=contract_type
    )

def get_optimal_vip_tier(monthly_volume: float, monthly_trades: int) -> BybitVIPTier:
    """Get optimal VIP tier based on trading volume and frequency."""
    calculator = BybitFeeCalculator()
    optimizer = BybitFeeOptimizer(calculator)
    
    result = optimizer.find_optimal_vip_tier(
        monthly_volume=monthly_volume,
        monthly_trades=monthly_trades
    )
    
    return result.recommended_tier
