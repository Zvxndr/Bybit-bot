"""
Shared State Manager
====================

Manages shared data between the main trading application and the frontend server.
Provides real-time access to trading metrics, system status, and performance data.
"""

import threading
from datetime import datetime
from typing import Dict, Any, Optional
import time

class SharedState:
    """Thread-safe shared state for trading bot data"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._start_time = time.time()
        
        # Initialize default state
        self._state = {
            "system": {
                "status": "initializing",
                "start_time": self._start_time,
                "version": "2.0.0",
                "mode": "testnet",
                "uptime": 0
            },
            "trading": {
                "strategies_active": 0,
                "positions_count": 0,
                "balance": "Initializing...",
                "daily_pnl": "Calculating...",
                "margin_used": "0.00",
                "margin_available": "0.00"
            },
            "positions": [],
            "performance": {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "max_drawdown": 0.0
            },
            "logs": [],
            "last_update": datetime.now().isoformat()
        }
    
    def update_system_status(self, status: str):
        """Update system status"""
        with self._lock:
            self._state["system"]["status"] = status
            self._state["system"]["uptime"] = time.time() - self._start_time
            self._state["last_update"] = datetime.now().isoformat()
    
    def update_trading_data(self, **kwargs):
        """Update trading-related data"""
        with self._lock:
            for key, value in kwargs.items():
                if key in self._state["trading"]:
                    self._state["trading"][key] = value
            self._state["last_update"] = datetime.now().isoformat()
    
    def update_positions(self, positions: list):
        """Update positions data"""
        with self._lock:
            self._state["positions"] = positions
            self._state["trading"]["positions_count"] = len(positions)
            self._state["last_update"] = datetime.now().isoformat()
    
    def add_log_entry(self, level: str, message: str):
        """Add a log entry"""
        with self._lock:
            log_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "level": level,
                "message": message
            }
            self._state["logs"].insert(0, log_entry)  # Add to front
            
            # Keep only last 50 log entries
            if len(self._state["logs"]) > 50:
                self._state["logs"] = self._state["logs"][:50]
            
            self._state["last_update"] = datetime.now().isoformat()
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get complete state data"""
        with self._lock:
            # Calculate uptime
            current_uptime = time.time() - self._start_time
            self._state["system"]["uptime"] = current_uptime
            
            # Format uptime as readable string
            hours = int(current_uptime // 3600)
            minutes = int((current_uptime % 3600) // 60)
            seconds = int(current_uptime % 60)
            self._state["system"]["uptime_str"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            return self._state.copy()
    
    def get_trading_data(self) -> Dict[str, Any]:
        """Get trading-specific data"""
        with self._lock:
            return {
                "trading_bot": self._state["trading"].copy(),
                "system": self._state["system"].copy(),
                "last_update": self._state["last_update"]
            }
    
    def get_positions_data(self) -> Dict[str, Any]:
        """Get positions data"""
        with self._lock:
            total_pnl = sum(float(pos.get("pnl", "0").replace("+", "")) for pos in self._state["positions"])
            
            return {
                "positions": self._state["positions"].copy(),
                "total_pnl": f"{total_pnl:+.2f} USDT",
                "margin_used": self._state["trading"]["margin_used"],
                "margin_available": self._state["trading"]["margin_available"],
                "last_update": self._state["last_update"]
            }

# Global shared state instance
shared_state = SharedState()