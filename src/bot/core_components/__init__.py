"""
Bot Core Module

This module contains the core components for the Bybit Trading Bot,
including configuration management, trading logic, and system orchestration.
"""

# Import core components for external access
from .config.manager import UnifiedConfigurationManager
from .config.schema import UnifiedConfigurationSchema

# Import TradingBot from the parent core.py file
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    # Import from the core.py file in the parent directory  
    from core import TradingBot
except ImportError:
    # Fallback import strategy
    TradingBot = None

__all__ = [
    'UnifiedConfigurationManager',
    'UnifiedConfigurationSchema',
    'TradingBot'
]