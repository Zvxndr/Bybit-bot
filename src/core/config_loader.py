"""
Configuration Loader - Centralized Config Management
==================================================

Handles loading and validation of all configuration files
with proper error handling and fallback defaults.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigLoader:
    """Centralized configuration loading and management"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger("config_loader")
        self._loaded_configs = {}
        
    def load_config(self, config_name: str, required: bool = False) -> Dict[str, Any]:
        """
        Load configuration file with proper error handling
        
        Args:
            config_name: Name of config file (without .yaml extension)
            required: Whether to raise error if config not found
            
        Returns:
            Configuration dictionary with safe defaults
        """
        if config_name in self._loaded_configs:
            return self._loaded_configs[config_name]
        
        config_file = self.config_dir / f"{config_name}.yaml"
        
        try:
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                    self.logger.info(f"[OK] Loaded config: {config_name}")
                    self._loaded_configs[config_name] = config
                    return config
            else:
                if required:
                    raise FileNotFoundError(f"Required config file not found: {config_file}")
                
                self.logger.warning(f"[WARN] Config file not found: {config_name}, using defaults")
                default_config = self._get_default_config(config_name)
                self._loaded_configs[config_name] = default_config
                return default_config
                
        except yaml.YAMLError as e:
            self.logger.error(f"[ERROR] YAML parsing error in {config_name}: {e}")
            if required:
                raise
            return self._get_default_config(config_name)
        
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load config {config_name}: {e}")
            if required:
                raise
            return self._get_default_config(config_name)
    
    def _get_default_config(self, config_name: str) -> Dict[str, Any]:
        """Get safe default configuration for known config types"""
        
        defaults = {
            'debug': {
                'debug_mode': True,
                'debug_settings': {
                    'disable_real_trading': True,
                    'disable_api_orders': True,
                    'force_testnet': True,
                    'mock_api_responses': True,
                    'show_debug_warnings': True
                },
                'phase': {
                    'current': 'DEVELOPMENT',
                    'trading_allowed': False
                }
            },
            'config': {
                'environment': 'development',
                'trading': {
                    'enabled': False,
                    'risk_management': {
                        'max_risk_per_trade': 0.01,
                        'max_portfolio_risk': 0.05
                    }
                },
                'speed_demon': {
                    'enabled': True,
                    'dynamic_risk_scaling': {
                        'enabled': True,
                        'small_account_risk': 0.02,
                        'large_account_risk': 0.005,
                        'transition_start': 10000,
                        'transition_end': 100000
                    }
                }
            },
            'secrets': {
                'bybit_api_key': '',
                'bybit_api_secret': '',
                'environment': 'testnet'
            }
        }
        
        return defaults.get(config_name, {})
    
    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug configuration with safe defaults"""
        return self.load_config('debug')
    
    def get_main_config(self) -> Dict[str, Any]:
        """Get main application configuration"""
        return self.load_config('config')
    
    def get_secrets_config(self) -> Dict[str, Any]:
        """Get secrets configuration (optional)"""
        return self.load_config('secrets')
    
    def is_debug_mode(self) -> bool:
        """Check if application is in debug mode"""
        debug_config = self.get_debug_config()
        return debug_config.get('debug_mode', True)
    
    def get_speed_demon_config(self) -> Dict[str, Any]:
        """Get Speed Demon configuration"""
        main_config = self.get_main_config()
        return main_config.get('speed_demon', {
            'enabled': True,
            'dynamic_risk_scaling': {
                'enabled': True,
                'small_account_risk': 0.02,
                'large_account_risk': 0.005,
                'transition_start': 10000,
                'transition_end': 100000
            }
        })


# Global configuration loader
config_loader = ConfigLoader()