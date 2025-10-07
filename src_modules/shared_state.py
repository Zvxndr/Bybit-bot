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
import logging

# Setup logging for shared state
logger = logging.getLogger(__name__)

class SharedState:
    """Thread-safe shared state for trading bot data"""
    
    def __init__(self):
        logger.debug("🔧 Initializing SharedState...")
        self._lock = threading.RLock()
        self._start_time = time.time()
        
        # Speed Demon deployment status
        self.speed_demon_status = None
        logger.debug("🔧 Speed Demon status initialized to None")
        
        # Initialize default state
        logger.debug("🔧 Setting up default state...")
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
        
        # Initialize bot control flags to safe defaults
        self.emergency_stop = False  # NOT in emergency stop by default
        self.paused = False          # NOT paused by default  
        self.close_all_positions = False
        self.cancel_all_orders = False
        self.last_action = "initialize"
        logger.debug("🔧 Bot control flags initialized to safe defaults")
    
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
    
    def clear_all_data(self):
        """Clear all trading data and reset to defaults"""
        import os
        import shutil
        from pathlib import Path
        
        logger.warning("🔥 Starting comprehensive data wipe...")
        
        with self._lock:
            # Reset to initial state
            self._start_time = time.time()
            
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
                    "balance": "0.00 USDT",
                    "daily_pnl": "0.00 USDT",
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
            
            # Clear historical data files and directories
            try:
                data_paths = [
                    # Cache directories  
                    "src/data/speed_demon_cache",
                    "src/data/historical_cache",
                    "data/historical",
                    "data/cache",
                    "data/models",
                    "data/strategies",
                    # General data and logs
                    "data",
                    "logs",
                    "cache",
                    # Any .db files
                    "*.db"
                ]
                
                files_cleared = 0
                dirs_cleared = 0
                
                for pattern in data_paths:
                    if "*" in pattern:
                        # Handle glob patterns
                        for path in Path(".").glob(pattern):
                            try:
                                if path.is_file():
                                    path.unlink()
                                    files_cleared += 1
                                    logger.info(f"🗑️ Cleared file: {path}")
                                elif path.is_dir():
                                    shutil.rmtree(path)
                                    dirs_cleared += 1
                                    logger.info(f"🗑️ Cleared directory: {path}")
                            except Exception as e:
                                logger.warning(f"⚠️ Could not clear {path}: {e}")
                    else:
                        # Handle direct paths
                        if os.path.exists(pattern):
                            try:
                                if os.path.isdir(pattern):
                                    shutil.rmtree(pattern)
                                    dirs_cleared += 1
                                    logger.info(f"🗑️ Cleared directory: {pattern}")
                                else:
                                    os.remove(pattern)
                                    files_cleared += 1
                                    logger.info(f"🗑️ Cleared file: {pattern}")
                            except Exception as e:
                                logger.warning(f"⚠️ Could not clear {pattern}: {e}")
                
                # Also clear any remaining database files in subdirectories
                for db_file in Path(".").rglob("*.db"):
                    try:
                        db_file.unlink()
                        files_cleared += 1
                        logger.info(f"🗑️ Cleared database: {db_file}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not clear database {db_file}: {e}")
                
                logger.warning(f"🔥 Data wipe completed: {files_cleared} files, {dirs_cleared} directories cleared")
                        
            except Exception as e:
                logger.error(f"❌ Error during data wipe: {e}")
        
        # Log the reset
        self.add_log_entry("WARNING", f"🔥 SYSTEM RESET - All trading data and historical files cleared - {files_cleared} files, {dirs_cleared} directories removed")
        logger.warning("🔥 Data wipe operation completed successfully")
    
    def get_multi_environment_balance(self) -> Dict[str, Any]:
        """Get balance data for all environments (testnet, mainnet, paper)"""
        with self._lock:
            balance_data = {
                "testnet": {"total": 0.0, "available": 0.0, "used": 0.0, "unrealized": 0.0},
                "mainnet": {"total": 0.0, "available": 0.0, "used": 0.0, "unrealized": 0.0},
                "paper": {"total": 100000.0, "available": 100000.0, "used": 0.0, "unrealized": 0.0}
            }
            
            # Attempt to get real balance data from API
            try:
                import os
                api_key = os.getenv('BYBIT_API_KEY') or os.getenv('BYBIT_TESTNET_API_KEY')
                api_secret = os.getenv('BYBIT_API_SECRET') or os.getenv('BYBIT_TESTNET_API_SECRET')
                
                if api_key and api_secret:
                    from pybit.unified_trading import HTTP
                    
                    # Get testnet balance
                    try:
                        testnet_client = HTTP(api_key=api_key, api_secret=api_secret, testnet=True)
                        response = testnet_client.get_wallet_balance(accountType="UNIFIED")
                        
                        if response.get('retCode') == 0:
                            wallet_list = response.get('result', {}).get('list', [])
                            if wallet_list:
                                wallet = wallet_list[0]  # First wallet
                                coins = wallet.get('coin', [])
                                for coin in coins:
                                    if coin.get('coin') == 'USDT':
                                        balance_data["testnet"] = {
                                            "total": float(coin.get('walletBalance', '0')),
                                            "available": float(coin.get('availableToWithdraw', '0')),
                                            "used": float(coin.get('locked', '0')),
                                            "unrealized": float(coin.get('unrealisedPnl', '0'))
                                        }
                                        break
                    except Exception as e:
                        logger.debug(f"Could not fetch testnet balance: {e}")
                    
                    # Get mainnet balance if not in testnet mode
                    if os.getenv('BYBIT_TESTNET', 'true').lower() != 'true':
                        try:
                            mainnet_client = HTTP(api_key=api_key, api_secret=api_secret, testnet=False)
                            response = mainnet_client.get_wallet_balance(accountType="UNIFIED")
                            
                            if response.get('retCode') == 0:
                                wallet_list = response.get('result', {}).get('list', [])
                                if wallet_list:
                                    wallet = wallet_list[0]
                                    coins = wallet.get('coin', [])
                                    for coin in coins:
                                        if coin.get('coin') == 'USDT':
                                            balance_data["mainnet"] = {
                                                "total": float(coin.get('walletBalance', '0')),
                                                "available": float(coin.get('availableToWithdraw', '0')),
                                                "used": float(coin.get('locked', '0')),
                                                "unrealized": float(coin.get('unrealisedPnl', '0'))
                                            }
                                            break
                        except Exception as e:
                            logger.debug(f"Could not fetch mainnet balance: {e}")
                            
            except Exception as e:
                logger.debug(f"Balance API integration not available: {e}")
            
            return balance_data
    
    def reset_bot_control_flags(self):
        """Reset all bot control flags to safe defaults"""
        logger.debug("🔧 Resetting bot control flags to safe defaults")
        with self._lock:
            self.emergency_stop = False
            self.paused = False
            self.close_all_positions = False 
            self.cancel_all_orders = False
            self.last_action = "reset"
            
            # Also clear from state dict
            if "bot_control" in self._state:
                self._state["bot_control"] = {
                    "emergency_stop": False,
                    "paused": False,
                    "close_all_positions": False,
                    "cancel_all_orders": False,
                    "last_action": "reset"
                }
            self._state["last_update"] = datetime.now().isoformat()
        logger.info("✅ Bot control flags reset to safe defaults")

    def set_bot_control(self, control_key: str, value: Any):
        """Set bot control flags for UI communication"""
        logger.debug(f"🔧 Setting bot control: {control_key} = {value}")
        with self._lock:
            # Update the attribute directly for backward compatibility
            setattr(self, control_key, value)
            
            # Also store in state for consistency
            if "bot_control" not in self._state:
                self._state["bot_control"] = {}
            self._state["bot_control"][control_key] = value
            self._state["last_update"] = datetime.now().isoformat()
    
    def get_bot_control(self, control_key: str, default=None):
        """Get bot control flag value"""
        return getattr(self, control_key, default)
    
    def is_emergency_stopped(self) -> bool:
        """Check if emergency stop is active"""
        return getattr(self, 'emergency_stop', False)
    
    def is_paused(self) -> bool:
        """Check if bot is paused"""
        return getattr(self, 'paused', False)  # Changed from 'is_paused' to 'paused'
    
    def should_close_all_positions(self) -> bool:
        """Check and reset close all positions flag"""
        with self._lock:
            if getattr(self, 'close_all_positions', False):
                self.close_all_positions = False  # Reset flag after check
                return True
            return False
    
    def should_cancel_all_orders(self) -> bool:
        """Check and reset cancel all orders flag"""
        with self._lock:
            if getattr(self, 'cancel_all_orders', False):
                self.cancel_all_orders = False  # Reset flag after check
                return True
            return False

# Global shared state instance
shared_state = SharedState()