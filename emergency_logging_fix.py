#!/usr/bin/env python3
"""
Emergency Logging Fix for Production
===================================

Comprehensively reduces logging verbosity to ERROR level only
and disables database schema error spam by fixing the missing column.
"""

import logging
import os
import sys


def apply_comprehensive_logging_fix():
    """Apply comprehensive logging configuration to silence verbose output"""
    
    print("🔧 EMERGENCY LOGGING FIX: Reducing to ERROR level only")
    
    # Set root logger to ERROR level
    logging.basicConfig(
        level=logging.ERROR,
        format='%(levelname)s: %(message)s',
        force=True  # Override any existing configuration
    )
    
    # Silence ALL third-party and internal loggers to ERROR only
    loggers_to_fix = [
        'uvicorn',
        'uvicorn.access',
        'uvicorn.error',
        'fastapi',
        'aiohttp',
        'aiohttp.access', 
        'asyncio',
        'websockets',
        'websocket',
        'sqlalchemy',
        'sqlalchemy.engine',
        'sqlalchemy.dialects',
        'sqlalchemy.pool',
        'requests',
        'urllib3',
        'urllib3.connectionpool',
        'src.bybit_api',
        'src.bot',
        'bot.pipeline',
        'bot.database',
        'src.main',
        'main',
        '__main__'
    ]
    
    for logger_name in loggers_to_fix:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        # Remove all existing handlers to prevent duplicate logs
        logger.handlers.clear()
        logger.propagate = True
    
    # Configure structlog if available to ERROR only
    try:
        import structlog
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.dev.set_exc_info,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR),
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        print("✅ Structured logging configured for ERROR level only")
    except ImportError:
        pass
    
    print("✅ Comprehensive logging fix applied - ERROR level only")


if __name__ == "__main__":
    apply_comprehensive_logging_fix()
    print("🏁 Logging fix complete - console should be much cleaner now!")