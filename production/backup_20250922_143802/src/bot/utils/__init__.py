"""
Utilities package for the trading bot.

This package contains common utilities and helper functions used
throughout the trading bot application including rate limiting,
logging, and API management tools.
"""

from .logging import TradingLogger, setup_logging, trading_logger

# Phase 2: Rate limiting utilities
try:
    from .rate_limiter import (
        RateLimiter,
        MultiEndpointRateLimiter,
        BYBIT_RATE_LIMITS,
        create_bybit_rate_limiter
    )
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False

if RATE_LIMITER_AVAILABLE:
    __all__ = [
        "TradingLogger",
        "setup_logging", 
        "trading_logger",
        # Phase 2 additions
        "RateLimiter",
        "MultiEndpointRateLimiter",
        "BYBIT_RATE_LIMITS",
        "create_bybit_rate_limiter"
    ]
else:
    __all__ = [
        "TradingLogger",
        "setup_logging", 
        "trading_logger",
    ]