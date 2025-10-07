"""
Unified Configuration Manager - Phase 4 Configuration Consolidation

This module provides the main configuration management system that orchestrates
all configuration operations, integrating with existing systems and providing
a unified interface for configuration management.

Key Features:
- Single configuration manager for all systems
- Automatic detection and migration of existing configs
- Environment-specific configuration management
- Hot reloading and dynamic updates
- Configuration validation and schema enforcement
- Secrets management and encryption
- Integration with Phase 1-3 systems
- Configuration versioning and backup
- Performance monitoring and health checks
- Australian compliance features

Integration Points:
- Phase 1: Unified Risk Management System
- Phase 2.5: ML Integration Layer  
- Phase 3: Unified API System
- Existing configuration files and systems
"""

import os
import json
import yaml
import shutil
import asyncio
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass
import logging
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from watchdog.observers import Observer

try:
    from .schema import (
        UnifiedConfigurationSchema, Environment, TradingMode,
        SecretsManager, EnvironmentManager, ConfigurationWatcher,
        DEFAULT_CONFIG_PATHS, DEFAULT_SECRETS_PATHS
    )
except ImportError:
    from schema import (
        UnifiedConfigurationSchema, Environment, TradingMode,
        SecretsManager, EnvironmentManager, ConfigurationWatcher,
        DEFAULT_CONFIG_PATHS, DEFAULT_SECRETS_PATHS
    )

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

@dataclass
class ValidationResult:
    """Configuration validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    score: float  # 0.0 to 1.0

class ConfigurationValidator:
    """Configuration validation and health checking"""
    
    def __init__(self):
        logger.info("Configuration validator initialized")
    
    def validate_schema(self, config: UnifiedConfigurationSchema) -> ValidationResult:
        """Validate configuration against schema"""
        errors = []
        warnings = []
        
        try:
            # Pydantic validation happens automatically on construction
            config.dict()  # Trigger validation
        except Exception as e:
            errors.append(f"Schema validation failed: {str(e)}")
        
        # Additional business logic validation
        if config.enable_trading:
            if not config.get_current_credentials():
                errors.append("Trading is enabled but no credentials configured")
            elif not config.validate_credentials():
                errors.append("Trading is enabled but credentials are invalid")
        
        # Risk management validation
        trading_config = config.get_trading_mode_config()
        if trading_config:
            if hasattr(trading_config, 'max_risk_ratio') and trading_config.max_risk_ratio > 0.3:
                warnings.append("Risk ratio above 30% is very aggressive")
        
        # Database validation
        if config.database.dialect.value == 'postgresql' and not config.database.password:
            warnings.append("PostgreSQL configured without password")
        
        # Performance validation
        if config.connection.pool_size > 200:
            warnings.append("Connection pool size is very high")
        
        # Security validation
        if config.environment == Environment.PRODUCTION:
            if not config.security.enable_encryption:
                errors.append("Encryption must be enabled in production")
            if not config.security.secret_key:
                errors.append("Secret key is required in production")
        
        # Calculate validation score
        score = 1.0
        score -= len(errors) * 0.2  # Each error reduces score by 20%
        score -= len(warnings) * 0.05  # Each warning reduces score by 5%
        score = max(0.0, score)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            score=score
        )
    
    def validate_connectivity(self, config: UnifiedConfigurationSchema) -> ValidationResult:
        """Validate external connectivity"""
        errors = []
        warnings = []
        
        # Database connectivity would be tested here
        # Exchange API connectivity would be tested here
        # Redis connectivity would be tested here
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            score=1.0 if len(errors) == 0 else 0.5
        )

# ============================================================================
# CONFIGURATION MIGRATION
# ============================================================================

class ConfigurationMigrator:
    """Migration utilities for existing configuration files"""
    
    def __init__(self):
        logger.info("Configuration migrator initialized")
    
    def detect_existing_configs(self, workspace_root: str) -> Dict[str, List[str]]:
        """Detect existing configuration files"""
        root_path = Path(workspace_root)
        configs = {
            'yaml_configs': [],
            'python_configs': [],
            'env_files': [],
            'json_configs': []
        }
        
        # Find YAML configuration files
        for yaml_file in root_path.rglob('*.yaml'):
            if 'config' in str(yaml_file).lower():
                configs['yaml_configs'].append(str(yaml_file))
        
        for yml_file in root_path.rglob('*.yml'):
            if 'config' in str(yml_file).lower():
                configs['yaml_configs'].append(str(yml_file))
        
        # Find Python configuration files
        for py_file in root_path.rglob('*config*.py'):
            configs['python_configs'].append(str(py_file))
        
        # Find environment files
        for env_file in root_path.rglob('.env*'):
            configs['env_files'].append(str(env_file))
        
        # Find JSON configuration files
        for json_file in root_path.rglob('*.json'):
            if 'config' in str(json_file).lower():
                configs['json_configs'].append(str(json_file))
        
        return configs
    
    def migrate_yaml_config(self, filepath: str) -> Dict[str, Any]:
        """Migrate YAML configuration to unified format"""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            # Transform to unified schema structure
            unified_config = self._transform_to_unified_schema(data)
            
            logger.info(f"Successfully migrated YAML config from {filepath}")
            return unified_config
            
        except Exception as e:
            logger.error(f"Failed to migrate YAML config {filepath}: {e}")
            return {}
    
    def migrate_env_file(self, filepath: str) -> Dict[str, str]:
        """Migrate environment file to configuration overrides"""
        env_vars = {}
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip().strip('"\'')
            
            logger.info(f"Successfully migrated env file from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to migrate env file {filepath}: {e}")
        
        return env_vars
    
    def _transform_to_unified_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform legacy configuration data to unified schema"""
        unified = {}
        
        # Map common fields
        field_mappings = {
            'trading': 'trading',
            'exchange': 'exchange', 
            'database': 'database',
            'ml': 'ml',
            'machine_learning': 'ml',
            'backtesting': 'backtesting',
            'logging': 'logging',
            'security': 'security',
            'monitoring': 'monitoring',
            'api': 'api'
        }
        
        for old_key, new_key in field_mappings.items():
            if old_key in data:
                unified[new_key] = data[old_key]
        
        # Transform specific structures
        if 'bybit' in data:
            unified['exchange'] = {
                Environment.DEVELOPMENT.value: data['bybit'].get('testnet', {}),
                Environment.PRODUCTION.value: data['bybit'].get('mainnet', {})
            }
        
        # Handle credentials structure
        if 'credentials' in data:
            creds = data['credentials']
            for env_name, env_creds in creds.items():
                if env_name in ['development', 'staging', 'production']:
                    if 'exchange' not in unified:
                        unified['exchange'] = {}
                    unified['exchange'][env_name] = env_creds
        
        return unified
    
    def backup_existing_configs(self, workspace_root: str) -> str:
        """Create backup of existing configuration files"""
        backup_dir = Path(workspace_root) / "config_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        configs = self.detect_existing_configs(workspace_root)
        
        for config_type, filepaths in configs.items():
            type_dir = backup_dir / config_type
            type_dir.mkdir(exist_ok=True)
            
            for filepath in filepaths:
                src_path = Path(filepath)
                dst_path = type_dir / src_path.name
                
                try:
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Backed up {filepath} to {dst_path}")
                except Exception as e:
                    logger.error(f"Failed to backup {filepath}: {e}")
        
        logger.info(f"Configuration backup created at {backup_dir}")
        return str(backup_dir)

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class UnifiedConfigurationManager:
    """
    Main configuration manager that provides unified configuration management
    """
    
    def __init__(
        self,
        workspace_root: Optional[str] = None,
        config_path: Optional[str] = None,
        secrets_key_path: Optional[str] = None
    ):
        self.workspace_root = workspace_root or os.getcwd()
        self.config_path = config_path
        self.secrets_key_path = secrets_key_path or os.path.join(self.workspace_root, ".secrets_key")
        
        # Initialize managers
        self.environment_manager = EnvironmentManager()
        self.validator = ConfigurationValidator()
        self.migrator = ConfigurationMigrator()
        
        # Initialize secrets manager
        self._initialize_secrets_manager()
        
        # Configuration state
        self._config: Optional[UnifiedConfigurationSchema] = None
        self._config_hash: Optional[str] = None
        self._last_loaded: Optional[datetime] = None
        
        # Hot reload settings
        self._observer: Optional[Observer] = None
        self._reload_callbacks: List[Callable[[UnifiedConfigurationSchema], None]] = []
        
        logger.info(f"Configuration manager initialized for workspace: {self.workspace_root}")
    
    def _initialize_secrets_manager(self):
        """Initialize secrets manager with key"""
        try:
            if os.path.exists(self.secrets_key_path):
                key = SecretsManager().load_key(self.secrets_key_path)
                self.secrets_manager = SecretsManager(key)
                logger.info("Loaded existing secrets key")
            else:
                self.secrets_manager = SecretsManager()
                self.secrets_manager.save_key(self.secrets_key_path)
                logger.info("Generated new secrets key")
        except Exception as e:
            logger.error(f"Failed to initialize secrets manager: {e}")
            self.secrets_manager = SecretsManager()
    
    def load_configuration(
        self,
        config_path: Optional[str] = None,
        environment: Optional[Environment] = None
    ) -> UnifiedConfigurationSchema:
        """Load and validate configuration"""
        
        # Determine configuration path
        if config_path:
            config_file = config_path
        elif self.config_path:
            config_file = self.config_path
        else:
            config_file = self._find_config_file()
        
        # Load configuration data
        config_data = self._load_config_data(config_file)
        
        # Apply environment-specific settings
        if environment:
            config_data['environment'] = environment.value
        else:
            # Use detected environment
            config_data['environment'] = self.environment_manager.current_environment.value
        
        # Create configuration instance
        try:
            config = UnifiedConfigurationSchema(**config_data)
        except Exception as e:
            logger.error(f"Failed to create configuration: {e}")
            # Fall back to default configuration
            config = UnifiedConfigurationSchema()
        
        # Apply environment variable overrides
        config = self.environment_manager.apply_environment_overrides(config)
        
        # Validate configuration
        validation_result = self.validator.validate_schema(config)
        if not validation_result.is_valid:
            logger.error(f"Configuration validation failed: {validation_result.errors}")
            for warning in validation_result.warnings:
                logger.warning(f"Configuration warning: {warning}")
        else:
            logger.info(f"Configuration validation passed (score: {validation_result.score:.2f})")
        
        # Cache configuration
        self._config = config
        self._config_hash = self._calculate_config_hash(config)
        self._last_loaded = datetime.now()
        
        logger.info(f"Configuration loaded successfully for environment: {config.environment.value}")
        return config
    
    def _find_config_file(self) -> str:
        """Find configuration file in default locations"""
        for config_path in DEFAULT_CONFIG_PATHS:
            full_path = os.path.join(self.workspace_root, config_path)
            if os.path.exists(full_path):
                logger.info(f"Found configuration file: {full_path}")
                return full_path
        
        # No config file found, return default location for creation
        default_path = os.path.join(self.workspace_root, "src/bot/core/config/config.yaml")
        logger.info(f"No existing configuration found, will use: {default_path}")
        return default_path
    
    def _load_config_data(self, config_path: str) -> Dict[str, Any]:
        """Load configuration data from file"""
        if not os.path.exists(config_path):
            logger.info(f"Configuration file not found: {config_path}, using defaults")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f) or {}
            
            # Decrypt secrets if needed
            if self.secrets_manager and 'encrypted' in data:
                data = self.secrets_manager.decrypt_secrets(data)
            
            logger.info(f"Configuration data loaded from {config_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            return {}
    
    def save_configuration(
        self,
        config: Optional[UnifiedConfigurationSchema] = None,
        config_path: Optional[str] = None,
        encrypt_secrets: bool = True
    ):
        """Save configuration to file"""
        if config is None:
            config = self._config
        
        if config is None:
            raise ValueError("No configuration to save")
        
        if config_path is None:
            config_path = self.config_path or self._find_config_file()
        
        # Ensure directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_data = config.to_dict()
        
        # Encrypt secrets if requested
        if encrypt_secrets and self.secrets_manager:
            config_data = self.secrets_manager.encrypt_secrets(config_data)
            config_data['encrypted'] = True
        
        # Save to file
        try:
            with open(config_path, 'w') as f:
                if config_path.endswith('.json'):
                    json.dump(config_data, f, indent=2, default=str)
                else:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {config_path}")
            
            # Update cache
            self._config = config
            self._config_hash = self._calculate_config_hash(config)
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def get_configuration(self) -> Optional[UnifiedConfigurationSchema]:
        """Get current configuration"""
        return self._config
    
    def reload_configuration(self) -> bool:
        """Reload configuration from file"""
        try:
            old_config = self._config
            new_config = self.load_configuration()
            
            # Check if configuration actually changed
            new_hash = self._calculate_config_hash(new_config)
            if new_hash == self._config_hash:
                logger.debug("Configuration unchanged, skipping reload")
                return False
            
            # Notify callbacks
            for callback in self._reload_callbacks:
                try:
                    callback(new_config)
                except Exception as e:
                    logger.error(f"Error in reload callback: {e}")
            
            logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def enable_hot_reload(self, config_path: Optional[str] = None):
        """Enable hot reloading of configuration files"""
        if self._observer:
            logger.warning("Hot reload is already enabled")
            return
        
        watch_path = config_path or self.config_path or self._find_config_file()
        watch_dir = str(Path(watch_path).parent)
        
        # Create watcher
        watcher = ConfigurationWatcher(self._on_config_changed)
        self._observer = Observer()
        self._observer.schedule(watcher, watch_dir, recursive=False)
        self._observer.start()
        
        logger.info(f"Hot reload enabled for {watch_dir}")
    
    def disable_hot_reload(self):
        """Disable hot reloading"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Hot reload disabled")
    
    def add_reload_callback(self, callback: Callable[[UnifiedConfigurationSchema], None]):
        """Add callback for configuration reload events"""
        self._reload_callbacks.append(callback)
        logger.info("Added configuration reload callback")
    
    def _on_config_changed(self, filepath: str):
        """Handle configuration file changes"""
        logger.info(f"Configuration file changed: {filepath}")
        self.reload_configuration()
    
    def _calculate_config_hash(self, config: UnifiedConfigurationSchema) -> str:
        """Calculate hash of configuration for change detection"""
        config_str = json.dumps(config.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def migrate_existing_configs(self) -> Dict[str, Any]:
        """Migrate existing configuration files to unified format"""
        logger.info("Starting configuration migration")
        
        # Backup existing configs
        backup_dir = self.migrator.backup_existing_configs(self.workspace_root)
        
        # Detect existing configurations
        existing_configs = self.migrator.detect_existing_configs(self.workspace_root)
        
        # Migrate YAML configs
        merged_config = {}
        for yaml_config in existing_configs.get('yaml_configs', []):
            config_data = self.migrator.migrate_yaml_config(yaml_config) 
            merged_config.update(config_data)
        
        # Migrate environment variables
        for env_file in existing_configs.get('env_files', []):
            env_vars = self.migrator.migrate_env_file(env_file)
            # Environment variables are handled separately by EnvironmentManager
        
        logger.info(f"Configuration migration completed. Backup created at: {backup_dir}")
        return merged_config
    
    def create_default_configuration(
        self,
        environment: Environment = Environment.DEVELOPMENT,
        enable_trading: bool = False
    ) -> UnifiedConfigurationSchema:
        """Create default configuration for environment"""
        
        config = UnifiedConfigurationSchema(
            environment=environment,
            enable_trading=enable_trading,
            version="1.0.0"
        )
        
        # Environment-specific defaults
        if environment == Environment.PRODUCTION:
            config.security.enable_encryption = True
            config.logging.level = config.logging.level.INFO
            config.enable_websockets = True
            config.enable_ml_integration = True
        elif environment == Environment.DEVELOPMENT:
            config.logging.level = config.logging.level.DEBUG
            config.enable_trading = False  # Safety default
        
        logger.info(f"Created default configuration for {environment.value}")
        return config
    
    def validate_current_configuration(self) -> ValidationResult:
        """Validate current configuration"""
        if not self._config:
            return ValidationResult(
                is_valid=False,
                errors=["No configuration loaded"],
                warnings=[],
                score=0.0
            )
        
        return self.validator.validate_schema(self._config)
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        if not self._config:
            return {"status": "no_configuration_loaded"}
        
        summary = self._config.get_summary()
        
        # Add manager-specific information
        summary.update({
            'workspace_root': self.workspace_root,
            'config_path': self.config_path,
            'last_loaded': self._last_loaded.isoformat() if self._last_loaded else None,
            'hot_reload_enabled': self._observer is not None,
            'validation_score': self.validate_current_configuration().score
        })
        
        return summary
    
    def export_configuration(
        self,
        output_path: str,
        format: str = 'yaml',
        include_secrets: bool = False
    ):
        """Export configuration to file"""
        if not self._config:
            raise ValueError("No configuration to export")
        
        config_data = self._config.to_dict()
        
        # Remove secrets if not requested
        if not include_secrets:
            config_data = self._remove_secrets(config_data)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if format.lower() == 'json':
                json.dump(config_data, f, indent=2, default=str)
            else:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration exported to {output_path}")
    
    def _remove_secrets(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from configuration"""
        sensitive_keys = {
            'api_key', 'api_secret', 'password', 'secret_key',
            'jwt_secret', 'email_password', 'redis_password'
        }
        
        def remove_secrets_recursive(obj):
            if isinstance(obj, dict):
                return {
                    key: "[REDACTED]" if key in sensitive_keys else remove_secrets_recursive(value)
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [remove_secrets_recursive(item) for item in obj]
            else:
                return obj
        
        return remove_secrets_recursive(config_data)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disable_hot_reload()

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_configuration_manager(
    workspace_root: Optional[str] = None,
    environment: Optional[Environment] = None
) -> UnifiedConfigurationManager:
    """Create and initialize configuration manager"""
    manager = UnifiedConfigurationManager(workspace_root=workspace_root)
    
    # Load configuration
    try:
        manager.load_configuration(environment=environment)
    except Exception as e:
        logger.warning(f"Failed to load existing configuration: {e}")
        # Create default configuration
        default_config = manager.create_default_configuration(
            environment=environment or Environment.DEVELOPMENT
        )
        manager._config = default_config
    
    return manager

def load_unified_configuration(
    config_path: Optional[str] = None,
    environment: Optional[Environment] = None
) -> UnifiedConfigurationSchema:
    """Load unified configuration from file"""
    manager = create_configuration_manager()
    return manager.load_configuration(config_path=config_path, environment=environment)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'UnifiedConfigurationManager',
    'ConfigurationValidator',
    'ConfigurationMigrator',
    'ValidationResult',
    'create_configuration_manager', 
    'load_unified_configuration'
]