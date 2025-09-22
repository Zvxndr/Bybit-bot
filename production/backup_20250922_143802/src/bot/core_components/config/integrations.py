"""
Configuration Integration Layer - Phase 4 Configuration Consolidation

This module provides integration points between the unified configuration system
and existing Phase 1-3 systems, ensuring seamless compatibility and migration.

Integration Points:
- Phase 1: Unified Risk Management System
- Phase 2.5: ML Integration Layer
- Phase 3: Unified API System
- Legacy configuration systems

Features:
- Backward compatibility with existing configuration interfaces
- Automatic configuration adaptation for Phase 1-3 systems
- Configuration bridge patterns
- Migration utilities
- Integration testing utilities
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

try:
    from .schema import UnifiedConfigurationSchema, Environment, TradingMode
    from .manager import UnifiedConfigurationManager
except ImportError:
    from schema import UnifiedConfigurationSchema, Environment, TradingMode
    from manager import UnifiedConfigurationManager

logger = logging.getLogger(__name__)

# ============================================================================
# PHASE 1 INTEGRATION - UNIFIED RISK MANAGEMENT
# ============================================================================

class RiskManagementConfigAdapter:
    """Adapter for Phase 1 Unified Risk Management System"""
    
    def __init__(self, unified_config: UnifiedConfigurationSchema):
        self.config = unified_config
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration in Phase 1 format"""
        trading_mode_config = self.config.get_trading_mode_config()
        
        risk_config = {
            'trading_mode': self.config.trading.mode.value,
            'base_balance': float(self.config.trading.base_balance),
            'enable_risk_management': self.config.enable_risk_management,
            
            # Risk limits
            'portfolio_drawdown_limit': trading_mode_config.portfolio_drawdown_limit,
            'strategy_drawdown_limit': trading_mode_config.strategy_drawdown_limit,
            'sharpe_ratio_min': trading_mode_config.sharpe_ratio_min,
            'var_daily_limit': trading_mode_config.var_daily_limit,
            'consistency_min': trading_mode_config.consistency_min,
        }
        
        # Mode-specific configuration
        if self.config.trading.mode == TradingMode.AGGRESSIVE:
            aggressive_config = self.config.trading.aggressive_mode
            if aggressive_config:
                risk_config.update({
                    'max_risk_ratio': aggressive_config.max_risk_ratio,
                    'min_risk_ratio': aggressive_config.min_risk_ratio,
                    'balance_thresholds': aggressive_config.balance_thresholds,
                    'risk_decay': aggressive_config.risk_decay
                })
        elif self.config.trading.mode == TradingMode.CONSERVATIVE:
            conservative_config = self.config.trading.conservative_mode
            if conservative_config:
                risk_config.update({
                    'risk_ratio': conservative_config.risk_ratio
                })
        
        return risk_config
    
    def get_australian_compliance_config(self) -> Dict[str, Any]:
        """Get Australian compliance configuration"""
        compliance = self.config.australian_compliance
        
        return {
            'timezone': compliance.timezone,
            'tax_year': compliance.tax_year,
            'enable_tax_reporting': compliance.enable_tax_reporting,
            'cgt_threshold': float(compliance.cgt_threshold),
            'record_all_trades': compliance.record_all_trades,
            'export_format': compliance.export_format,
            'reporting_currency': compliance.reporting_currency
        }

# ============================================================================
# PHASE 2.5 INTEGRATION - ML INTEGRATION LAYER
# ============================================================================

class MLIntegrationConfigAdapter:
    """Adapter for Phase 2.5 ML Integration Layer"""
    
    def __init__(self, unified_config: UnifiedConfigurationSchema):
        self.config = unified_config
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get ML configuration in Phase 2.5 format"""
        ml_config = self.config.ml
        
        return {
            'enable_ml_integration': self.config.enable_ml_integration,
            
            # Feature engineering
            'lookback_periods': ml_config.lookback_periods,
            'technical_indicators': ml_config.technical_indicators,
            'statistical_features': ml_config.statistical_features,
            
            # Model configurations
            'models': ml_config.models,
            
            # Cross-validation
            'cross_validation': ml_config.cross_validation,
            
            # Trading pairs and timeframes
            'symbols': self.config.trading.symbols,
            'timeframes': self.config.trading.timeframes
        }
    
    def get_backtesting_config(self) -> Dict[str, Any]:
        """Get backtesting configuration"""
        backtesting = self.config.backtesting
        
        return {
            'enable_backtesting': self.config.enable_backtesting,
            'walk_forward': backtesting.walk_forward,
            'validation': backtesting.validation
        }

# ============================================================================
# PHASE 3 INTEGRATION - UNIFIED API SYSTEM
# ============================================================================

class APISystemConfigAdapter:
    """Adapter for Phase 3 Unified API System"""
    
    def __init__(self, unified_config: UnifiedConfigurationSchema):
        self.config = unified_config
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration in Phase 3 format"""
        credentials = self.config.get_current_credentials()
        
        api_config = {
            'enable_trading': self.config.enable_trading,
            'enable_websockets': self.config.enable_websockets,
            'enable_market_data': self.config.enable_market_data,
            
            # Credentials
            'credentials': {
                'api_key': credentials.api_key if credentials else '',
                'api_secret': credentials.api_secret if credentials else '',
                'is_testnet': credentials.is_testnet if credentials else True,
                'base_url': credentials.base_url if credentials else '',
                'recv_window': credentials.recv_window if credentials else 5000
            } if credentials else {},
            
            # Rate limiting
            'rate_limits': {
                'market_data': self.config.rate_limits.market_data,
                'trading': self.config.rate_limits.trading,
                'account': self.config.rate_limits.account,
                'websocket': self.config.rate_limits.websocket,
                'enable_rate_limiting': self.config.rate_limits.enable_rate_limiting
            },
            
            # Connection settings
            'connection': {
                'pool_size': self.config.connection.pool_size,
                'timeout': self.config.connection.timeout,
                'max_retries': self.config.connection.max_retries,
                'keepalive_timeout': self.config.connection.keepalive_timeout,
                'enable_compression': self.config.connection.enable_compression,
                'ssl_verify': self.config.connection.ssl_verify
            },
            
            # WebSocket settings
            'websocket': {
                'ping_interval': self.config.websocket.ping_interval,
                'ping_timeout': self.config.websocket.ping_timeout,
                'close_timeout': self.config.websocket.close_timeout,
                'max_reconnect_attempts': self.config.websocket.max_reconnect_attempts,
                'base_reconnect_delay': self.config.websocket.base_reconnect_delay,
                'max_reconnect_delay': self.config.websocket.max_reconnect_delay,
                'enable_compression': self.config.websocket.enable_compression,
                'message_queue_size': self.config.websocket.message_queue_size
            },
            
            # Cache settings
            'cache': {
                'backend': self.config.cache.backend.value,
                'enable_caching': self.config.cache.enable_caching,
                'cache_size': self.config.cache.cache_size,
                'cache_ttl_seconds': self.config.cache.cache_ttl_seconds,
                'cleanup_interval': self.config.cache.cleanup_interval,
                'redis_host': self.config.cache.redis_host,
                'redis_port': self.config.cache.redis_port,
                'redis_db': self.config.cache.redis_db,
                'redis_password': self.config.cache.redis_password
            }
        }
        
        return api_config

# ============================================================================
# LEGACY CONFIGURATION BRIDGE
# ============================================================================

class LegacyConfigurationBridge:
    """Bridge for legacy configuration systems"""
    
    def __init__(self, unified_config: UnifiedConfigurationSchema):
        self.config = unified_config
        self.risk_adapter = RiskManagementConfigAdapter(unified_config)
        self.ml_adapter = MLIntegrationConfigAdapter(unified_config)
        self.api_adapter = APISystemConfigAdapter(unified_config)
    
    def get_legacy_config_dict(self) -> Dict[str, Any]:
        """Get configuration in legacy dictionary format"""
        return {
            # Environment and metadata
            'environment': self.config.environment.value,
            'version': self.config.version,
            
            # Trading configuration
            'trading': {
                'mode': self.config.trading.mode.value,
                'base_balance': float(self.config.trading.base_balance),
                'symbols': self.config.trading.symbols,
                'timeframes': self.config.trading.timeframes,
                'order_type': self.config.trading.order_type,
                'max_slippage': self.config.trading.max_slippage,
                'order_timeout': self.config.trading.order_timeout
            },
            
            # Exchange configuration
            'bybit': self._get_legacy_exchange_config(),
            
            # Database configuration  
            'database': {
                'dialect': self.config.database.dialect.value,
                'host': self.config.database.host,
                'port': self.config.database.port,
                'database': self.config.database.database,
                'username': self.config.database.username,
                'password': self.config.database.password,
                'ssl_mode': self.config.database.ssl_mode,
                'pool_size': self.config.database.pool_size,
                'max_overflow': self.config.database.max_overflow,
                'echo': self.config.database.echo
            },
            
            # Logging configuration
            'logging': {
                'level': self.config.logging.level.value,
                'enable_file_logging': self.config.logging.enable_file_logging,
                'log_file_path': self.config.logging.log_file_path,
                'max_file_size_mb': self.config.logging.max_file_size_mb,
                'backup_count': self.config.logging.backup_count,
                'enable_console_logging': self.config.logging.enable_console_logging,
                'log_format': self.config.logging.log_format
            },
            
            # Feature flags
            'features': {
                'enable_trading': self.config.enable_trading,
                'enable_websockets': self.config.enable_websockets,
                'enable_market_data': self.config.enable_market_data,
                'enable_risk_management': self.config.enable_risk_management,
                'enable_ml_integration': self.config.enable_ml_integration,
                'enable_backtesting': self.config.enable_backtesting
            },
            
            # Component-specific configurations
            'risk_management': self.risk_adapter.get_risk_config(),
            'ml': self.ml_adapter.get_ml_config(),
            'api': self.api_adapter.get_api_config(),
            'australian_compliance': self.risk_adapter.get_australian_compliance_config()
        }
    
    def _get_legacy_exchange_config(self) -> Dict[str, Any]:
        """Get exchange configuration in legacy format"""
        legacy_config = {}
        
        for env, credentials in self.config.exchange.items():
            env_key = 'testnet' if credentials.is_testnet else 'mainnet'
            if env_key not in legacy_config:
                legacy_config[env_key] = {}
            
            legacy_config[env_key] = {
                'api_key': credentials.api_key,
                'api_secret': credentials.api_secret,
                'base_url': credentials.base_url,
                'recv_window': credentials.recv_window
            }
        
        return legacy_config

# ============================================================================
# INTEGRATION UTILITIES
# ============================================================================

class ConfigurationIntegrationManager:
    """Manager for configuration integration operations"""
    
    def __init__(self, config_manager: UnifiedConfigurationManager):
        self.config_manager = config_manager
        self.config = config_manager.get_configuration()
        
        if self.config:
            self.legacy_bridge = LegacyConfigurationBridge(self.config)
            self.risk_adapter = RiskManagementConfigAdapter(self.config)
            self.ml_adapter = MLIntegrationConfigAdapter(self.config)
            self.api_adapter = APISystemConfigAdapter(self.config)
    
    def update_phase1_risk_system(self, risk_manager_path: str):
        """Update Phase 1 risk management system with new configuration"""
        if not self.config:
            raise ValueError("No configuration loaded")
        
        risk_config = self.risk_adapter.get_risk_config()
        
        # This would integrate with the actual Phase 1 system
        logger.info(f"Would update Phase 1 risk system at {risk_manager_path}")
        logger.info(f"Risk configuration: {risk_config}")
    
    def update_phase25_ml_system(self, ml_integration_path: str):
        """Update Phase 2.5 ML integration system with new configuration"""
        if not self.config:
            raise ValueError("No configuration loaded")
        
        ml_config = self.ml_adapter.get_ml_config()
        
        # This would integrate with the actual Phase 2.5 system
        logger.info(f"Would update Phase 2.5 ML system at {ml_integration_path}")
        logger.info(f"ML configuration: {ml_config}")
    
    def update_phase3_api_system(self, api_system_path: str):
        """Update Phase 3 API system with new configuration"""
        if not self.config:
            raise ValueError("No configuration loaded")
        
        api_config = self.api_adapter.get_api_config()
        
        # This would integrate with the actual Phase 3 system
        logger.info(f"Would update Phase 3 API system at {api_system_path}")
        logger.info(f"API configuration: {api_config}")
    
    def validate_integration_compatibility(self) -> Dict[str, bool]:
        """Validate compatibility with existing Phase 1-3 systems"""
        if not self.config:
            return {'error': 'No configuration loaded'}
        
        compatibility = {
            'phase1_risk_management': True,
            'phase25_ml_integration': True, 
            'phase3_api_system': True,
            'legacy_systems': True
        }
        
        # Check Phase 1 compatibility
        try:
            risk_config = self.risk_adapter.get_risk_config()
            required_risk_fields = [
                'trading_mode', 'base_balance', 'portfolio_drawdown_limit'
            ]
            for field in required_risk_fields:
                if field not in risk_config:
                    compatibility['phase1_risk_management'] = False
                    break
        except Exception:
            compatibility['phase1_risk_management'] = False
        
        # Check Phase 2.5 compatibility
        try:
            ml_config = self.ml_adapter.get_ml_config()
            required_ml_fields = [
                'enable_ml_integration', 'models', 'technical_indicators'
            ]
            for field in required_ml_fields:
                if field not in ml_config:
                    compatibility['phase25_ml_integration'] = False
                    break
        except Exception:
            compatibility['phase25_ml_integration'] = False
        
        # Check Phase 3 compatibility
        try:
            api_config = self.api_adapter.get_api_config()
            required_api_fields = [
                'credentials', 'rate_limits', 'connection', 'websocket'
            ]
            for field in required_api_fields:
                if field not in api_config:
                    compatibility['phase3_api_system'] = False
                    break
        except Exception:
            compatibility['phase3_api_system'] = False
        
        # Check legacy compatibility
        try:
            legacy_config = self.legacy_bridge.get_legacy_config_dict()
            required_legacy_fields = [
                'environment', 'trading', 'database', 'logging'
            ]
            for field in required_legacy_fields:
                if field not in legacy_config:
                    compatibility['legacy_systems'] = False
                    break
        except Exception:
            compatibility['legacy_systems'] = False
        
        return compatibility
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration report"""
        if not self.config:
            return {'error': 'No configuration loaded'}
        
        compatibility = self.validate_integration_compatibility()
        
        report = {
            'timestamp': str(logger.handlers[0].formatter.formatTime() if logger.handlers else 'unknown'),
            'configuration_summary': self.config.get_summary(),
            'compatibility': compatibility,
            'integration_status': {
                'phase1_ready': compatibility.get('phase1_risk_management', False),
                'phase25_ready': compatibility.get('phase25_ml_integration', False),
                'phase3_ready': compatibility.get('phase3_api_system', False),
                'legacy_ready': compatibility.get('legacy_systems', False)
            },
            'recommendations': []
        }
        
        # Add recommendations based on compatibility
        if not compatibility.get('phase1_risk_management'):
            report['recommendations'].append(
                "Review Phase 1 risk management configuration compatibility"
            )
        
        if not compatibility.get('phase25_ml_integration'):
            report['recommendations'].append(
                "Review Phase 2.5 ML integration configuration compatibility"
            )
        
        if not compatibility.get('phase3_api_system'):
            report['recommendations'].append(
                "Review Phase 3 API system configuration compatibility"
            )
        
        if not compatibility.get('legacy_systems'):
            report['recommendations'].append(
                "Review legacy system configuration compatibility"
            )
        
        return report

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_integration_manager(
    config_manager: UnifiedConfigurationManager
) -> ConfigurationIntegrationManager:
    """Create configuration integration manager"""
    return ConfigurationIntegrationManager(config_manager)

def get_legacy_config_dict(config: UnifiedConfigurationSchema) -> Dict[str, Any]:
    """Get configuration in legacy dictionary format"""
    bridge = LegacyConfigurationBridge(config)
    return bridge.get_legacy_config_dict()

def validate_phase_compatibility(config: UnifiedConfigurationSchema) -> Dict[str, bool]:
    """Validate compatibility with all phases"""
    # Create a temporary manager for validation
    from .manager import UnifiedConfigurationManager
    temp_manager = UnifiedConfigurationManager()
    temp_manager._config = config
    
    integration_manager = ConfigurationIntegrationManager(temp_manager)
    return integration_manager.validate_integration_compatibility()

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'RiskManagementConfigAdapter',
    'MLIntegrationConfigAdapter', 
    'APISystemConfigAdapter',
    'LegacyConfigurationBridge',
    'ConfigurationIntegrationManager',
    'create_integration_manager',
    'get_legacy_config_dict',
    'validate_phase_compatibility'
]