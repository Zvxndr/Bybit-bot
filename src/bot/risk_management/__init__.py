"""
Australian Risk Management System
Tax-aware position sizing with regulatory compliance monitoring
Integrates with ML strategies and arbitrage systems for Australian traders
"""

from .australian_risk_manager import (
    AustralianRiskCalculator,
    TaxAwarePositionManager,
    ComplianceMonitor,
    RiskParameters,
    PositionRisk,
    PositionType,
    RiskLevel
)

from .portfolio_risk_controller import (
    PortfolioRiskController,
    PortfolioState,
    RiskAlert,
    AlertLevel,
    ActionType
)

# Backward compatibility - existing system components
try:
    from .unified_risk_manager import (
        UnifiedRiskManager, 
        PositionSizeMethod,
        RiskMetrics, 
        TradeRiskAssessment, 
        PortfolioRiskProfile
    )
except ImportError:
    UnifiedRiskManager = None
    PositionSizeMethod = None
    RiskMetrics = None
    TradeRiskAssessment = None
    PortfolioRiskProfile = None

try:
    from .risk_manager import RiskManager
except ImportError:
    RiskManager = None

# Primary exports - Australian-focused risk management
__all__ = [
    # Australian risk management (primary)
    'AustralianRiskCalculator',
    'TaxAwarePositionManager',
    'ComplianceMonitor',
    'PortfolioRiskController',
    
    # Supporting classes
    'RiskParameters',
    'PositionRisk',
    'PositionType',
    'PortfolioState',
    'RiskAlert',
    'AlertLevel',
    'ActionType',
    
    # Backward compatibility
    'UnifiedRiskManager',
    'RiskManager',
    'PositionSizeMethod',
    'RiskMetrics',
    'TradeRiskAssessment',
    'PortfolioRiskProfile'
]

# Default Australian risk manager - recommended for new Australian-focused code
def get_australian_risk_manager():
    """Get Australian risk management system instance."""
    from .australian_risk_manager import AustralianRiskCalculator, RiskParameters
    from .portfolio_risk_controller import PortfolioRiskController
    
    risk_params = RiskParameters()
    risk_calculator = AustralianRiskCalculator(risk_params)
    
    return risk_calculator