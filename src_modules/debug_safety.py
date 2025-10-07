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

# Import historical data provider for realistic debugging data
try:
    from .historical_data_provider import get_historical_data_provider
    historical_data_available = True
except ImportError:
    historical_data_available = False
    logging.warning("Historical data provider not available, falling back to simple mock data")

# Handle yaml import with fallback
try:
    import yaml
except ImportError:
    # Create a minimal yaml substitute for deployment
    class SimpleYAML:
        @staticmethod
        def safe_load(content):
            """Basic configuration parser for debug config"""
            if hasattr(content, 'read'):
                content = content.read()
            
            # Return safe debug defaults if parsing fails
            return {
                'debug_mode': True,
                'debug_settings': {
                    'disable_real_trading': True,
                    'disable_api_orders': True,
                    'force_testnet': True,
                    'mock_api_responses': True,
                    'max_debug_runtime': 3600
                },
                'phase': {
                    'current': 'DEPLOYMENT_DEBUG',
                    'trading_allowed': False
                },
                'mock_data': {
                    'testnet_balance': 10000.00,
                    'mainnet_balance': 0.00,
                    'paper_balance': 100000.00
                }
            }
        
        @staticmethod
        def dump(data, f, **kwargs):
            # Simple config output
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
    
    yaml = SimpleYAML()

logger = logging.getLogger(__name__)

class DebugSafetyManager:
    """Manages debug mode and prevents real trading during development"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Check for production mode via environment variables
        self.is_production_env = os.getenv('PRODUCTION_MODE', '').lower() == 'true'
        self.deployment_env = os.getenv('DEPLOYMENT_ENV', 'development')
        
        # Determine config file based on environment
        if self.is_production_env:
            self.config_path = config_path or "config/production.yaml"
        else:
            self.config_path = config_path or "config/debug.yaml"
            
        self.config = self._load_debug_config()
        self.debug_mode = self.config.get('debug_mode', True)
        self.start_time = datetime.now()
        
        # Log the mode we're operating in
        mode = "PRODUCTION" if not self.debug_mode else "DEBUG"
        logger.info(f"🔧 Safety Manager initialized in {mode} mode")
        logger.info(f"🔧 Environment: {self.deployment_env}")
        logger.info(f"🔧 Config file: {self.config_path}")
        
        # Initialize session
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
        """Initialize session with appropriate safety checks"""
        if self.debug_mode:
            logger.warning("🚨 DEBUG MODE ACTIVE - Trading execution disabled")
            logger.info("🔧 Debug Safety Manager initialized")
            logger.info(f"📅 Debug session started at: {self.start_time}")
            
            # Check max runtime for debug mode
            max_runtime = self.config.get('debug_settings', {}).get('max_debug_runtime', 3600)
            end_time = self.start_time + timedelta(seconds=max_runtime)
            logger.info(f"⏰ Debug session will auto-shutdown at: {end_time}")
        else:
            logger.info("🚀 PRODUCTION MODE ACTIVE - Live trading enabled")
            logger.info("🔧 Production Safety Manager initialized") 
            logger.info(f"📅 Production session started at: {self.start_time}")
            
            # Log production safety settings
            safety_settings = self.config.get('safety_systems', {})
            logger.info(f"🛡️ Emergency stop: {safety_settings.get('emergency_stop_enabled', True)}")
            logger.info(f"🛡️ Position monitoring: {safety_settings.get('position_monitoring', True)}")
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is active"""
        return self.debug_mode
    
    def is_trading_allowed(self) -> bool:
        """Check if trading operations are allowed"""
        if self.debug_mode:
            return False
        # In production mode, check if live trading is enabled
        return self.config.get('production_settings', {}).get('enable_live_trading', False)
    
    def should_use_testnet(self) -> bool:
        """Determine if testnet should be used"""
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
        """Get realistic data for testing using historical data when available"""
        
        # Try to use historical data first
        if historical_data_available:
            try:
                provider = get_historical_data_provider()
                
                if data_type == 'balances':
                    return provider.get_realistic_balances()
                
                elif data_type == 'positions':
                    return provider.get_historical_positions()
                
                elif data_type == 'trades':
                    return provider.get_historical_trades()
                
                elif data_type == 'market_data':
                    return provider.get_market_data_sample()
                    
            except Exception as e:
                logger.warning(f"🔄 Historical data unavailable ({e}), using fallback mock data")
        
        # Fallback to configuration-based mock data
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