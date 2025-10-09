"""
Core Strategy Management Module
Provides base classes for trading strategies
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    signal_type: SignalType
    symbol: str
    price: float
    quantity: float
    timestamp: datetime
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_active = False
        
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate trading signal based on market data"""
        return None
        
    def start(self):
        """Start the strategy"""
        self.is_active = True
        
    def stop(self):
        """Stop the strategy"""
        self.is_active = False