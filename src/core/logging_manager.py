"""
Logging Manager - Windows-Safe Logging
=====================================

Provides Windows-compatible logging that handles Unicode properly
and prevents console encoding crashes.
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class WindowsSafeFormatter(logging.Formatter):
    """Formatter that handles Windows console encoding issues"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if we're on Windows and in console mode
        self.is_windows_console = (
            sys.platform == "win32" and 
            hasattr(sys.stdout, 'encoding') and 
            sys.stdout.encoding.lower() in ['cp1252', 'ascii']
        )
    
    def format(self, record):
        """Format log message with Windows-safe encoding"""
        msg = super().format(record)
        
        if self.is_windows_console:
            # Replace Unicode emojis with ASCII equivalents for console
            emoji_replacements = {
                '✅': '[OK]',
                '❌': '[ERROR]',
                '⚠️': '[WARN]',
                '🔧': '[DEBUG]',
                '🚀': '[START]',
                '💥': '[CRASH]',
                '🔍': '[SEARCH]',
                '🎯': '[TARGET]',
                '📊': '[DATA]',
                '💰': '[MONEY]',
                '🏁': '[END]',
                '🔥': '[FIRE]',
                '⏰': '[TIME]',
                '📅': '[DATE]',
                '🤖': '[BOT]',
                '📡': '[API]',
                '🛡️': '[SHIELD]',
                '🎉': '[SUCCESS]'
            }
            
            for emoji, replacement in emoji_replacements.items():
                msg = msg.replace(emoji, replacement)
            
            # Ensure ASCII-safe encoding
            msg = msg.encode('ascii', 'replace').decode('ascii')
        
        return msg


class LoggingManager:
    """Manages application logging with Windows compatibility"""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.loggers = {}
        
    def setup_comprehensive_logging(self):
        """Setup comprehensive logging system"""
        # Create formatters
        console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        file_format = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        
        # Console formatter (Windows-safe)
        console_formatter = WindowsSafeFormatter(console_format)
        
        # File formatter (can handle Unicode)
        file_formatter = logging.Formatter(file_format)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Create main logger
        logger = logging.getLogger("main")
        logger.info("Application logging system initialized")
        
        return logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a named logger"""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]


# Global logging manager
logging_manager = LoggingManager()