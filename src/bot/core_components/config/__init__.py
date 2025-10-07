"""
Unified Configuration Package - Phase 4 Configuration Consolidation

This package provides a complete, unified configuration management system that
consolidates all scattered configuration implementations across the trading bot.

Features:
- Single source of truth for all configuration
- Environment-specific configuration management
- Secure secrets management with encryption
- Configuration validation and type safety
- Hot reloading and dynamic updates
- Integration with all existing systems (Phase 1-3)
- Australian compliance configuration
- Comprehensive CLI tools
- Configuration migration utilities
- Performance monitoring and health checks

Components:
- schema.py: Configuration schemas and models
- manager.py: Main configuration management system
- cli.py: Command-line interface
- integrations.py: Integration with existing systems

Usage:
```python
from src.bot.core.config import (
    UnifiedConfigurationManager, 
    UnifiedConfigurationSchema,
    Environment
)

# Create configuration manager
manager = UnifiedConfigurationManager()

# Load configuration
config = manager.load_configuration(environment=Environment.DEVELOPMENT)

# Use configuration
if config.enable_trading:
    credentials = config.get_current_credentials()
    # ... trading logic
```

CLI Usage:
```bash
# Initialize new configuration
python -m src.bot.core.config.cli init --environment development

# Validate configuration
python -m src.bot.core.config.cli validate

# Switch environments
python -m src.bot.core.config.cli use production

# Export configuration
python -m src.bot.core.config.cli export config_backup.yaml
```
"""

from .schema import (
    # Main configuration schema
    UnifiedConfigurationSchema,
    
    # Enums
    Environment,
    TradingMode, 
    LogLevel,
    DatabaseDialect,
    CacheBackend,
    
    # Component configurations
    ExchangeCredentials,
    DatabaseConfig,
    TradingConfig,
    MLConfig,
    BacktestingConfig,
    AustralianComplianceConfig,
    LoggingConfig,
    SecurityConfig,
    MonitoringConfig,
    APIConfig,
    RateLimitConfig,
    ConnectionConfig,
    WebSocketConfig,
    CacheConfig,
    
    # Utility classes
    SecretsManager,
    EnvironmentManager,
    ConfigurationWatcher
)

from .manager import (
    # Main manager
    UnifiedConfigurationManager,
    
    # Validation and migration
    ConfigurationValidator,
    ConfigurationMigrator,
    ValidationResult,
    
    # Convenience functions
    create_configuration_manager,
    load_unified_configuration
)

# Version information
__version__ = "1.0.0"
__author__ = "Trading Bot Team"
__description__ = "Unified Configuration Management System"

# Default configuration paths
DEFAULT_CONFIG_FILE = "src/bot/core/config/config.yaml"
DEFAULT_SECRETS_KEY = ".secrets_key"

# Convenience functions for common operations
def get_default_manager(workspace_root: str = None) -> UnifiedConfigurationManager:
    """Get default configuration manager instance"""
    return create_configuration_manager(workspace_root=workspace_root)

def quick_load(environment_name: str = None) -> UnifiedConfigurationSchema:
    """Quick load configuration for environment"""
    env = Environment(environment_name) if environment_name else None
    return load_unified_configuration(environment=env)

# Export all public components
__all__ = [
    # Main classes
    'UnifiedConfigurationManager',
    'UnifiedConfigurationSchema',
    'ConfigurationValidator',
    'ConfigurationMigrator',
    'ValidationResult',
    
    # Enums
    'Environment',
    'TradingMode',
    'LogLevel', 
    'DatabaseDialect',
    'CacheBackend',
    
    # Configuration models
    'ExchangeCredentials',
    'DatabaseConfig',
    'TradingConfig',
    'MLConfig',
    'BacktestingConfig',
    'AustralianComplianceConfig',
    'LoggingConfig',
    'SecurityConfig',
    'MonitoringConfig',
    'APIConfig',
    'RateLimitConfig',
    'ConnectionConfig',
    'WebSocketConfig',
    'CacheConfig',
    
    # Utility classes
    'SecretsManager',
    'EnvironmentManager',
    'ConfigurationWatcher',
    
    # Convenience functions
    'create_configuration_manager',
    'load_unified_configuration',
    'get_default_manager',
    'quick_load',
    
    # Constants
    'DEFAULT_CONFIG_FILE',
    'DEFAULT_SECRETS_KEY'
]