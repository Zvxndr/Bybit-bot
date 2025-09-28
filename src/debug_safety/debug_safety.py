"""
Debug Safety Manager
===================

Comprehensive debugging system to prevent trading execution during development.
This ensures no real money is at risk during the debugging phase.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

try:
    import yaml
except ImportError:
    print("Warning: PyYAML not installed. Using basic configuration.")
    # Create a minimal yaml substitute
    class SimpleYAML:
        @staticmethod
        def safe_load(content):
            # Basic YAML parsing for debug config
            if isinstance(content, str):
                lines = content.split('\n')
            else:
                lines = content.read().split('\n')
            
            config = {}
            for line in lines:
                if ':' in line and not line.strip().startswith('#'):
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert basic types
                    if value.lower() == 'true':
                        config[key] = True
                    elif value.lower() == 'false':
                        config[key] = False
                    elif value.isdigit():
                        config[key] = int(value)
                    else:
                        config[key] = value.strip('"\'')
            
            # Provide safe defaults if parsing fails
            if not config:
                return {
                    'debug_mode': True,
                    'debug_settings': {'disable_real_trading': True},
                    'phase': {'trading_allowed': False}
                }
            return config
        
        @staticmethod
        def dump(data, f, **kwargs):
            # Simple YAML output
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
    
    yaml = SimpleYAML()

logger = logging.getLogger(__name__)

class DebugSafetyManager:
    """Manages debug mode and prevents real trading during development"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/debug.yaml"
        self.config = self._load_debug_config()
        self.debug_mode = self.config.get('debug_mode', True)
        self.start_time = datetime.now()
        
        # Initialize debug session
        self._initialize_debug_session()
        
    def _load_debug_config(self) -> Dict[str, Any]:
        """Load debug configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info("🔧 Debug configuration loaded successfully")
                    return config
            else:
                logger.warning(f"⚠️ Debug config file not found: {self.config_path}")
                return self._get_default_debug_config()
        except Exception as e:
            logger.error(f"❌ Error loading debug config: {e}")
            return self._get_default_debug_config()
    
    def _get_default_debug_config(self) -> Dict[str, Any]:
        """Return safe default debug configuration"""
        return {
            'debug_mode': True,
            'debug_settings': {
                'disable_real_trading': True,
                'disable_api_orders': True,
                'force_testnet': True,
                'mock_api_responses': True,
                'show_debug_warnings': True
            },
            'phase': {
                'current': 'PRIVATE_USE_DEBUGGING',
                'trading_allowed': False
            }
        }
    
    def _initialize_debug_session(self):
        """Initialize debug session with safety checks"""
        if self.debug_mode:
            logger.warning("🚨 DEBUG MODE ACTIVE - Trading execution disabled")
            logger.info("🔧 Debug Safety Manager initialized")
            logger.info(f"📅 Debug session started at: {self.start_time}")
            
            # Check max runtime
            max_runtime = self.config.get('debug_settings', {}).get('max_debug_runtime', 3600)
            end_time = self.start_time + timedelta(seconds=max_runtime)
            logger.info(f"⏰ Debug session will auto-shutdown at: {end_time}")
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is active"""
        return self.debug_mode
    
    def is_trading_allowed(self) -> bool:
        """Check if trading operations are allowed"""
        if self.debug_mode:
            return False
        return self.config.get('phase', {}).get('trading_allowed', False)
    
    def should_use_testnet(self) -> bool:
        """Force testnet usage in debug mode"""
        if self.debug_mode:
            return True
        return self.config.get('debug_settings', {}).get('force_testnet', False)
    
    def should_mock_api_calls(self) -> bool:
        """Check if API calls should be mocked"""
        return self.config.get('debug_settings', {}).get('mock_api_responses', False)
    
    def block_trading_operation(self, operation: str) -> bool:
        """Block trading operations in debug mode"""
        if self.debug_mode:
            settings = self.config.get('debug_settings', {})
            
            if operation == 'place_order' and settings.get('disable_api_orders', True):
                logger.warning(f"🚫 BLOCKED: {operation} - Debug mode prevents real trading")
                return True
                
            if operation == 'modify_position' and settings.get('disable_position_changes', True):
                logger.warning(f"🚫 BLOCKED: {operation} - Debug mode prevents position changes")
                return True
                
            if operation == 'real_trading' and settings.get('disable_real_trading', True):
                logger.warning(f"🚫 BLOCKED: {operation} - Debug mode active")
                return True
        
        return False
    
    def get_mock_data(self, data_type: str) -> Any:
        """Get mock data for testing"""
        mock_data = self.config.get('mock_data', {})
        
        if data_type == 'balances':
            return {
                'testnet': mock_data.get('testnet_balance', 10000.00),
                'mainnet': mock_data.get('mainnet_balance', 0.00),
                'paper': mock_data.get('paper_balance', 100000.00)
            }
        
        elif data_type == 'positions':
            return mock_data.get('mock_positions', [])
        
        elif data_type == 'trades':
            return mock_data.get('mock_trades', [])
        
        return None
    
    def log_debug_action(self, action: str, details: str = ""):
        """Log debug actions for tracking"""
        logger.debug(f"🔧 DEBUG ACTION: {action} - {details}")
    
    def check_runtime_limit(self) -> bool:
        """Check if debug session has exceeded runtime limit"""
        if not self.debug_mode:
            return False
            
        max_runtime = self.config.get('debug_settings', {}).get('max_debug_runtime', 3600)
        current_runtime = (datetime.now() - self.start_time).total_seconds()
        
        if current_runtime > max_runtime:
            logger.warning(f"⏰ Debug session exceeded max runtime ({max_runtime}s)")
            return True
        
        return False
    
    def get_debug_status(self) -> Dict[str, Any]:
        """Get current debug status for UI display"""
        runtime = (datetime.now() - self.start_time).total_seconds()
        max_runtime = self.config.get('debug_settings', {}).get('max_debug_runtime', 3600)
        
        return {
            'debug_mode': self.debug_mode,
            'phase': self.config.get('phase', {}).get('current', 'UNKNOWN'),
            'trading_allowed': self.is_trading_allowed(),
            'runtime_seconds': int(runtime),
            'max_runtime_seconds': max_runtime,
            'time_remaining': max_runtime - runtime,
            'testnet_forced': self.should_use_testnet(),
            'api_mocked': self.should_mock_api_calls()
        }
    
    def shutdown_debug_session(self):
        """Safely shutdown debug session"""
        runtime = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"🏁 Debug session ended after {runtime:.1f} seconds")
        logger.info("✅ Debug Safety Manager shutdown complete")

# Global debug manager instance
_debug_manager = None

def get_debug_manager() -> DebugSafetyManager:
    """Get or create global debug manager instance"""
    global _debug_manager
    if _debug_manager is None:
        _debug_manager = DebugSafetyManager()
    return _debug_manager

def is_debug_mode() -> bool:
    """Quick check if debug mode is active"""
    return get_debug_manager().is_debug_mode()

def block_trading_if_debug(operation: str) -> bool:
    """Quick check to block trading operations"""
    return get_debug_manager().block_trading_operation(operation)