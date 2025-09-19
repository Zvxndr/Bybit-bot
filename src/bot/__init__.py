"""
Bybit Trading Bot

A sophisticated algorithmic trading system for cryptocurrency perpetual swaps
with dynamic risk management and statistical validation.

Core Features:
- Dual-mode operation (conservative/aggressive)
- Dynamic risk scaling based on account balance
- Statistical strategy validation (WFO, CSCV, permutation testing)
- Machine learning integration with purged cross-validation
- Australian tax compliance and CGT calculation
- Real-time dashboard and monitoring

Architecture:
- Event-driven core with state management
- Modular design with replaceable components
- Database-centric approach for audit trails
- Configurable behavior without code changes
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Developer"
__email__ = "developer@example.com"

# Import main components for easy access
from .config import Config
from .database import DatabaseManager
from .core import TradingBot

# Note: These will be implemented in later phases
# from .risk import RiskManager, DynamicRiskManager  
# from .strategies import TradingStrategy
# from .execution import ExecutionEngine

__all__ = [
    "Config",
    "DatabaseManager",
    "TradingBot",
    # "RiskManager",
    # "DynamicRiskManager", 
    # "TradingStrategy",
    # "ExecutionEngine",
]