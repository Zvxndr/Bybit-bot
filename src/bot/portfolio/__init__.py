"""
Multi-Asset Portfolio Management Package - Phase 6 Implementation

This package provides advanced portfolio management capabilities for multi-asset trading:
- Portfolio Manager: Core portfolio optimization and management
- Asset Allocator: Dynamic asset allocation algorithms
- Correlation Analyzer: Cross-asset correlation analysis and monitoring
- Rebalancer: Automated portfolio rebalancing with multiple strategies
- Risk Budgeter: Advanced risk budgeting and constraint management
- Performance Tracker: Portfolio performance attribution and analysis

Author: Trading Bot Team
Version: 1.0.0 - Phase 6 Implementation
"""

from .portfolio_manager import (
    PortfolioManager,
    Portfolio,
    Position,
    PortfolioConstraints
)
from .asset_allocator import (
    AssetAllocator,
    AllocationStrategy,
    AllocationResult,
    OptimizationMethod
)
from .correlation_analyzer import (
    CorrelationAnalyzer,
    CorrelationRegime,
    CorrelationMeasure,
    CorrelationAnalysis
)
from .rebalancer import (
    PortfolioRebalancer,
    RebalanceStrategy,
    RebalanceTrigger,
    RebalanceSignal,
    RebalanceTransaction,
    RebalanceResult
)
from .risk_budgeter import (
    RiskBudgeter,
    RiskBudgetType,
    RiskConstraintType,
    RiskBudget,
    RiskConstraint,
    RiskAttribution,
    RiskReport
)

__all__ = [
    # Core portfolio management
    'PortfolioManager',
    'Portfolio',
    'Position',
    'PortfolioConstraints',
    
    # Asset allocation
    'AssetAllocator',
    'AllocationStrategy',
    'AllocationResult',
    'OptimizationMethod',
    
    # Correlation analysis
    'CorrelationAnalyzer',
    'CorrelationRegime',
    'CorrelationMeasure',
    'CorrelationAnalysis',
    
    # Rebalancing
    'PortfolioRebalancer',
    'RebalanceStrategy',
    'RebalanceTrigger',
    'RebalanceSignal',
    'RebalanceTransaction',
    'RebalanceResult',
    
    # Risk budgeting
    'RiskBudgeter',
    'RiskBudgetType',
    'RiskConstraintType',
    'RiskBudget',
    'RiskConstraint',
    'RiskAttribution',
    'RiskReport',
]