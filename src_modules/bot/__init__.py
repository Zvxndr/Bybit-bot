"""
Bybit Trading Bot - Main Package

This package provides a comprehensive trading bot for Bybit exchange with:
- Advanced backtesting capabilities (Phase 3)
- Historical data pipeline (Phase 2) 
- Strategy development framework
- Risk management systems

Author: Trading Bot Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__phase__ = "3"

# Only import what's needed for backtesting integration
# Avoid importing TradingBot to prevent circular dependencies during testing
