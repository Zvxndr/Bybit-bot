"""
Risk Management system for the Bybit Trading Bot.

UNIFIED SYSTEM: All risk management has been consolidated into UnifiedRiskManager.
This provides comprehensive risk management functionality including:
- Advanced position sizing (Kelly, Risk Parity, Volatility Targeting)
- Real-time portfolio risk monitoring with circuit breakers
- Dynamic risk adjustment based on market conditions
- Comprehensive risk metrics and alerting

The individual components are still available for backward compatibility but are deprecated.
"""

from .unified_risk_manager import (
    UnifiedRiskManager, 
    RiskLevel, 
    RiskAction, 
    PositionSizeMethod,
    RiskMetrics, 
    TradeRiskAssessment, 
    PortfolioRiskProfile
)

# Backward compatibility - DEPRECATED
# These imports are maintained for existing code but should migrate to UnifiedRiskManager
try:
    from .risk_manager import RiskManager
except ImportError:
    RiskManager = None
    
try:
    from .position_sizer import PositionSizer
except ImportError:
    PositionSizer = None
    
try:
    from .portfolio_monitor import PortfolioMonitor
except ImportError:
    PortfolioMonitor = None
    
try:
    from .risk_calculator import RiskCalculator
except ImportError:
    RiskCalculator = None

# Primary exports - Use these for new development
__all__ = [
    # Primary unified system
    'UnifiedRiskManager',
    'RiskLevel',
    'RiskAction', 
    'PositionSizeMethod',
    'RiskMetrics',
    'TradeRiskAssessment',
    'PortfolioRiskProfile',
    
    # Deprecated - for backward compatibility only
    'RiskManager',
    'PositionSizer', 
    'PortfolioMonitor',
    'RiskCalculator'
]

# Default risk manager instance - recommended for new code
def get_risk_manager():
    """Get the unified risk manager instance."""
    return UnifiedRiskManager()