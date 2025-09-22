"""
Database package for the trading bot.

This package provides:
- SQLAlchemy models for all trading data
- Database connection management
- Migration support with Alembic
- Australian tax tracking integration
"""

from .manager import DatabaseManager
from .models import (
    Base,
    Trade,
    StrategyPerformance,
    Portfolio,
    RiskEvent,
    TaxEvent,
    MarketData,
    StrategyMetadata
)

__all__ = [
    "DatabaseManager",
    "Base", 
    "Trade",
    "StrategyPerformance",
    "Portfolio",
    "RiskEvent", 
    "TaxEvent",
    "MarketData",
    "StrategyMetadata",
]