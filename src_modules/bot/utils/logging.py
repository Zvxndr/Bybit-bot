"""
Logging utilities for the trading bot.

This module provides structured logging with JSON formatting, 
file rotation, and integration with the configuration system.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from ..config import LoggingConfig


class JSONFormatter:
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, include_fields: Optional[list] = None):
        self.include_fields = include_fields or [
            "timestamp", "level", "message", "strategy_id", "symbol", "mode", "balance"
        ]
    
    def format(self, record: Dict[str, Any]) -> str:
        """Format log record as JSON."""
        # Extract basic fields
        log_data = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "module": record["name"],
            "function": record["function"],
            "line": record["line"]
        }
        
        # Add extra fields if present
        extra = record.get("extra", {})
        for field in self.include_fields:
            if field in extra:
                log_data[field] = extra[field]
        
        # Add exception info if present
        if record.get("exception"):
            log_data["exception"] = {
                "type": record["exception"]["type"].__name__,
                "value": str(record["exception"]["value"]),
                "traceback": record["exception"]["traceback"]
            }
        
        return json.dumps(log_data, default=str)


def setup_logging(config: LoggingConfig) -> None:
    """
    Setup logging configuration based on config settings.
    
    Args:
        config: Logging configuration object
    """
    # Remove default logger
    logger.remove()
    
    # Setup console logging
    if config.console.get("enabled", True):
        console_level = config.console.get("level", config.level)
        
        if config.format == "json":
            logger.add(
                sys.stderr,
                level=console_level,
                format=JSONFormatter(config.fields).format,
                colorize=False
            )
        else:
            # Text format with colors
            logger.add(
                sys.stderr,
                level=console_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                colorize=True
            )
    
    # Setup file logging
    if config.file.get("enabled", True):
        log_path = Path(config.file.get("path", "./logs/trading_bot.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        rotation = config.file.get("rotation", "daily")
        max_size = config.file.get("max_size", "100MB")
        backup_count = config.file.get("backup_count", 10)
        
        if config.format == "json":
            logger.add(
                log_path,
                level=config.level,
                format=JSONFormatter(config.fields).format,
                rotation=rotation,
                retention=backup_count,
                compression="gz"
            )
        else:
            logger.add(
                log_path,
                level=config.level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation=rotation,  
                retention=backup_count,
                compression="gz"
            )
    
    logger.info("Logging configured successfully")


class TradingLogger:
    """
    Enhanced logger with trading-specific context.
    
    This class provides convenient methods for logging trading-related
    events with automatic context injection.
    """
    
    def __init__(self, strategy_id: Optional[str] = None):
        self.strategy_id = strategy_id
        self.context = {}
    
    def bind(self, **kwargs) -> "TradingLogger":
        """Bind context variables to logger."""
        new_logger = TradingLogger(self.strategy_id)
        new_logger.context = {**self.context, **kwargs}
        return new_logger
    
    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method with context injection."""
        context = {**self.context, **kwargs}
        if self.strategy_id:
            context["strategy_id"] = self.strategy_id
            
        getattr(logger.bind(**context), level)(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log("critical", message, **kwargs)
    
    # Trading-specific logging methods
    
    def trade_executed(self, symbol: str, side: str, amount: float, price: float, **kwargs):
        """Log trade execution."""
        self._log(
            "info",
            f"Trade executed: {side.upper()} {amount} {symbol} @ {price}",
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            **kwargs
        )
    
    def signal_generated(self, symbol: str, signal: str, confidence: float, **kwargs):
        """Log signal generation."""
        self._log(
            "info",
            f"Signal generated: {signal} for {symbol} (confidence: {confidence:.2%})",
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            **kwargs
        )
    
    def risk_limit_hit(self, limit_type: str, current_value: float, limit_value: float, **kwargs):
        """Log risk limit violations."""
        self._log(
            "warning",
            f"Risk limit hit: {limit_type} - Current: {current_value}, Limit: {limit_value}",
            limit_type=limit_type,
            current_value=current_value,
            limit_value=limit_value,
            **kwargs
        )
    
    def strategy_performance(self, equity: float, drawdown: float, sharpe: float, **kwargs):
        """Log strategy performance metrics."""
        self._log(
            "info",
            f"Strategy performance - Equity: {equity:.2f}, DD: {drawdown:.2%}, Sharpe: {sharpe:.2f}",
            equity=equity,
            drawdown=drawdown,
            sharpe=sharpe,
            **kwargs
        )
    
    def mode_switch(self, old_mode: str, new_mode: str, reason: str, **kwargs):
        """Log trading mode switches."""
        self._log(
            "info",
            f"Trading mode switched: {old_mode} -> {new_mode} (Reason: {reason})",
            old_mode=old_mode,
            new_mode=new_mode,
            reason=reason,
            **kwargs
        )


# Global trading logger instance
trading_logger = TradingLogger()