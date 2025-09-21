"""
Risk Management Engine for Cryptocurrency Trading Bot.

This package provides comprehensive risk management capabilities including:

- Advanced position sizing methods (Kelly, Risk Parity, Volatility Targeting)
- Portfolio risk analysis (correlation, sector exposure, tail risk)
- Real-time risk monitoring with circuit breakers
- Dynamic risk adjustment based on market conditions
- Comprehensive risk metrics and alerting

The risk management system is designed to work with cryptocurrency
markets and their unique characteristics including high volatility,
correlation dynamics, and regime changes.

Key Components:
    - PositionSizer: Advanced position sizing using multiple methods
    - PortfolioRiskMonitor: Comprehensive portfolio risk analysis
    - RealTimeRiskMonitor: Real-time monitoring with alerts and circuit breakers
    - DynamicRiskAdjuster: Adaptive risk management based on market conditions

Example Usage:
    ```python
    from bot.risk import PositionSizer, RealTimeRiskMonitor
    
    # Initialize position sizer
    position_sizer = PositionSizer()
    
    # Calculate position size
    result = position_sizer.calculate_position_size(
        asset='BTC',
        returns=btc_returns,
        strategy_returns=strategy_returns
    )
    
    # Start real-time monitoring
    risk_monitor = RealTimeRiskMonitor()
    risk_monitor.start_monitoring()
    ```
"""

from .position_sizing import (
    PositionSizer,
    KellyCriterion,
    VolatilityTargeting,
    RiskParity,
    MaxDrawdownSizing,
    PositionSizeResult
)

from .portfolio_analysis import (
    PortfolioRiskMonitor,
    CorrelationAnalyzer,
    SectorExposureAnalyzer,
    TailRiskAnalyzer,
    RiskAlert,
    RiskLevel,
    PortfolioRiskMetrics
)

from .real_time_monitoring import (
    RealTimeRiskMonitor,
    PositionSnapshot,
    RiskSnapshot,
    CircuitBreaker,
    CircuitBreakerStatus,
    MonitoringStatus
)

from .dynamic_adjustment import (
    DynamicRiskAdjuster,
    VolatilityRegimeAnalyzer,
    MarketRegimeAdjuster,
    CorrelationAdjuster,
    RiskAdjustment,
    AdjustmentTrigger,
    VolatilityMetrics
)

# Version information
__version__ = "1.0.0"
__author__ = "Crypto Trading Bot Team"

# Export all classes and functions
__all__ = [
    # Position Sizing
    "PositionSizer",
    "KellyCriterion", 
    "VolatilityTargeting",
    "RiskParity",
    "MaxDrawdownSizing",
    "PositionSizeResult",
    
    # Portfolio Analysis
    "PortfolioRiskMonitor",
    "CorrelationAnalyzer",
    "SectorExposureAnalyzer", 
    "TailRiskAnalyzer",
    "RiskAlert",
    "RiskLevel",
    "PortfolioRiskMetrics",
    
    # Real-time Monitoring
    "RealTimeRiskMonitor",
    "PositionSnapshot",
    "RiskSnapshot", 
    "CircuitBreaker",
    "CircuitBreakerStatus",
    "MonitoringStatus",
    
    # Dynamic Adjustment
    "DynamicRiskAdjuster",
    "VolatilityRegimeAnalyzer",
    "MarketRegimeAdjuster",
    "CorrelationAdjuster",
    "RiskAdjustment",
    "AdjustmentTrigger",
    "VolatilityMetrics",
]

# Module-level configuration defaults
DEFAULT_RISK_CONFIG = {
    'position_sizing': {
        'primary_method': 'volatility_targeting',
        'max_position_override': 0.2,
        'min_position_override': 0.01,
        'use_ensemble': True
    },
    'portfolio_analysis': {
        'correlation_threshold': 0.7,
        'max_sector_exposure': 0.4,
        'var_confidence_level': 0.95
    },
    'real_time_monitoring': {
        'monitoring_interval': 30,
        'enable_circuit_breakers': True,
        'max_daily_loss': 0.05
    },
    'dynamic_adjustment': {
        'adjustment_frequency': 3600,
        'volatility_adjustment': True,
        'regime_adjustment': True
    }
}

def create_risk_management_system(config=None):
    """
    Factory function to create a complete risk management system.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with all risk management components
    """
    if config is None:
        config = DEFAULT_RISK_CONFIG
    
    return {
        'position_sizer': PositionSizer(config.get('position_sizing', {})),
        'portfolio_monitor': PortfolioRiskMonitor(config.get('portfolio_analysis', {})),
        'real_time_monitor': RealTimeRiskMonitor(config.get('real_time_monitoring', {})),
        'dynamic_adjuster': DynamicRiskAdjuster(config.get('dynamic_adjustment', {}))
    }

def get_risk_management_info():
    """Get information about the risk management system."""
    return {
        'version': __version__,
        'components': len(__all__),
        'capabilities': [
            'Advanced Position Sizing',
            'Portfolio Risk Analysis', 
            'Real-time Risk Monitoring',
            'Dynamic Risk Adjustment',
            'Circuit Breakers',
            'Correlation Analysis',
            'Regime Detection',
            'Tail Risk Analysis',
            'Volatility Targeting',
            'Risk Parity'
        ],
        'supported_methods': {
            'position_sizing': ['Kelly Criterion', 'Volatility Targeting', 'Risk Parity', 'Max Drawdown'],
            'risk_analysis': ['VaR/CVaR', 'Correlation Analysis', 'Tail Risk', 'Stress Testing'],
            'monitoring': ['Real-time Alerts', 'Circuit Breakers', 'Performance Attribution'],
            'adjustment': ['Volatility Regime', 'Market Regime', 'Correlation Changes']
        }
    }
# These will be implemented in Phase 5
# from .base import RiskManager
# from .dynamic_risk import DynamicRiskManager
# from .portfolio_risk import PortfolioRiskManager

__all__ = [
    # "RiskManager",
    # "DynamicRiskManager",
    # "PortfolioRiskManager",
]