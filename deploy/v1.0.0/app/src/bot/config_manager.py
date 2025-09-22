"""
Comprehensive Configuration Management System - Phase 10

This module provides a robust configuration management system that handles:
- All component settings across phases
- Environment variables and secrets management
- API keys and credentials
- Deployment configurations  
- Configuration validation and defaults
- Dynamic configuration updates
- Configuration profiles (dev, staging, production)

Author: Trading Bot Team
Version: 1.0.0
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import logging
from datetime import datetime
import hashlib


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigurationError(Exception):
    """Configuration-related errors"""
    pass


@dataclass
class EnvironmentCredentials:
    """Environment-specific exchange credentials"""
    api_key: str = ""
    api_secret: str = ""
    is_testnet: bool = True
    base_url: str = ""
    
    def validate(self):
        if not self.api_key or not self.api_secret:
            raise ConfigurationError("API key and secret are required")
        if not self.base_url:
            raise ConfigurationError("Base URL is required")


@dataclass
class ExchangeConfig:
    """Exchange connection configuration with environment-specific credentials"""
    name: str = "bybit"
    environments: Dict[str, EnvironmentCredentials] = field(default_factory=dict)
    rate_limit: int = 10
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        # Initialize default environments if not provided
        if not self.environments:
            self.environments = {
                "development": EnvironmentCredentials(
                    is_testnet=True,
                    base_url="https://api-testnet.bybit.com"
                ),
                "staging": EnvironmentCredentials(
                    is_testnet=True,
                    base_url="https://api-testnet.bybit.com"
                ),
                "production": EnvironmentCredentials(
                    is_testnet=False,
                    base_url="https://api.bybit.com"
                )
            }
    
    def get_credentials(self, environment: str) -> EnvironmentCredentials:
        """Get credentials for specific environment"""
        env_name = environment.lower()
        if env_name not in self.environments:
            raise ConfigurationError(f"Environment '{environment}' not configured")
        return self.environments[env_name]
    
    def validate(self):
        if self.rate_limit <= 0:
            raise ConfigurationError("Rate limit must be positive")
        
        # Validate all environment credentials
        for env_name, credentials in self.environments.items():
            try:
                credentials.validate()
            except ConfigurationError as e:
                raise ConfigurationError(f"Environment '{env_name}': {e}")


@dataclass 
class TradingConfig:
    """Trading strategy configuration"""
    trading_pairs: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    base_currency: str = "USDT"
    initial_capital: float = 10000.0
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_trades: int = 50
    min_trade_size: float = 10.0
    max_spread_tolerance: float = 0.002
    update_frequency_seconds: int = 5
    strategy_type: str = "momentum"
    
    def validate(self):
        if not self.trading_pairs:
            raise ConfigurationError("At least one trading pair is required")
        if self.initial_capital <= 0:
            raise ConfigurationError("Initial capital must be positive")
        if not (0 < self.max_position_size <= 1):
            raise ConfigurationError("Max position size must be between 0 and 1")


@dataclass
class RiskManagementConfig:
    """Risk management configuration"""
    max_portfolio_risk: float = 0.02  # 2% per trade
    max_daily_loss: float = 0.05  # 5% daily loss limit
    max_drawdown: float = 0.15  # 15% maximum drawdown
    volatility_lookback: int = 20
    stop_loss_percent: float = 0.02  # 2% stop loss
    take_profit_percent: float = 0.04  # 4% take profit
    correlation_threshold: float = 0.7
    var_confidence_level: float = 0.95
    max_leverage: float = 1.0
    
    def validate(self):
        if not (0 < self.max_portfolio_risk <= 1):
            raise ConfigurationError("Max portfolio risk must be between 0 and 1")
        if not (0 < self.max_daily_loss <= 1):
            raise ConfigurationError("Max daily loss must be between 0 and 1")
        if not (0 < self.max_drawdown <= 1):
            raise ConfigurationError("Max drawdown must be between 0 and 1")


@dataclass
class BacktestingConfig:
    """Backtesting configuration"""
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    benchmark: str = "BTCUSDT"
    rebalance_frequency: str = "daily"
    optimization_metric: str = "sharpe_ratio"
    
    def validate(self):
        if self.initial_capital <= 0:
            raise ConfigurationError("Initial capital must be positive")
        if self.commission < 0:
            raise ConfigurationError("Commission cannot be negative")


@dataclass
class MonitoringConfig:
    """System monitoring configuration"""
    max_memory_usage_mb: int = 2048
    max_cpu_usage_percent: float = 80.0
    disk_usage_threshold: float = 0.85
    check_interval_seconds: int = 30
    health_check_timeout: int = 10
    log_level: str = "INFO"
    log_file_path: str = "logs/trading_bot.log"
    log_max_size_mb: int = 100
    log_backup_count: int = 5
    
    def validate(self):
        if self.max_memory_usage_mb <= 0:
            raise ConfigurationError("Max memory usage must be positive")
        if not (0 < self.max_cpu_usage_percent <= 100):
            raise ConfigurationError("Max CPU usage must be between 0 and 100")


@dataclass
class AlertingConfig:
    """Alerting and notification configuration"""
    enable_email_alerts: bool = False
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = field(default_factory=list)
    enable_discord_alerts: bool = False
    discord_webhook_url: str = ""
    enable_telegram_alerts: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    alert_cooldown_minutes: int = 15
    
    def validate(self):
        if self.enable_email_alerts:
            if not self.email_smtp_host or not self.email_username:
                raise ConfigurationError("Email configuration incomplete")
        if self.enable_discord_alerts and not self.discord_webhook_url:
            raise ConfigurationError("Discord webhook URL required")
        if self.enable_telegram_alerts and not self.telegram_bot_token:
            raise ConfigurationError("Telegram bot token required")


@dataclass
class DatabaseConfig:
    """Database configuration"""
    database_type: str = "sqlite"
    database_url: str = "sqlite:///trading_bot.db"
    host: str = "localhost"
    port: int = 5432
    username: str = ""
    password: str = ""
    database_name: str = "trading_bot"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    def validate(self):
        if self.database_type not in ["sqlite", "postgresql", "mysql"]:
            raise ConfigurationError("Unsupported database type")
        if self.database_type != "sqlite" and not self.database_url:
            raise ConfigurationError("Database URL required for non-SQLite databases")


@dataclass
class TaxReportingConfig:
    """Tax and reporting configuration"""
    tax_jurisdiction: str = "US"
    accounting_method: str = "FIFO"  # FIFO, LIFO, HIFO
    base_currency: str = "USD"
    enable_automated_reports: bool = True
    report_frequency_hours: int = 24
    report_output_directory: str = "reports"
    include_unrealized_gains: bool = False
    export_formats: List[str] = field(default_factory=lambda: ["PDF", "CSV", "JSON"])
    
    def validate(self):
        if self.accounting_method not in ["FIFO", "LIFO", "HIFO"]:
            raise ConfigurationError("Invalid accounting method")
        if self.report_frequency_hours <= 0:
            raise ConfigurationError("Report frequency must be positive")


@dataclass
class AdvancedFeaturesConfig:
    """Advanced features configuration"""
    enable_regime_detection: bool = True
    enable_portfolio_optimization: bool = True
    enable_news_analysis: bool = True
    enable_parameter_optimization: bool = True
    regime_update_frequency_minutes: int = 60
    news_update_frequency_minutes: int = 30
    optimization_frequency_days: int = 7
    news_sources: List[str] = field(default_factory=lambda: ["coindesk", "cointelegraph"])
    sentiment_model: str = "transformer"
    optimization_method: str = "bayesian"
    
    def validate(self):
        if self.regime_update_frequency_minutes <= 0:
            raise ConfigurationError("Regime update frequency must be positive")
        if self.optimization_method not in ["bayesian", "genetic", "grid_search"]:
            raise ConfigurationError("Invalid optimization method")


@dataclass
class APIConfig:
    """API interface configuration"""
    enable_rest_api: bool = True
    enable_websocket: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    websocket_port: int = 8081
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 60
    authentication_required: bool = True
    jwt_secret: str = ""
    jwt_expiration_hours: int = 24
    
    def validate(self):
        if self.authentication_required and not self.jwt_secret:
            raise ConfigurationError("JWT secret required when authentication is enabled")
        if self.port <= 0 or self.port > 65535:
            raise ConfigurationError("Port must be between 1 and 65535")


@dataclass
class ComprehensiveConfig:
    """Main configuration class containing all subsystems"""
    environment: Environment = Environment.DEVELOPMENT
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    tax_reporting: TaxReportingConfig = field(default_factory=TaxReportingConfig)
    advanced_features: AdvancedFeaturesConfig = field(default_factory=AdvancedFeaturesConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # System metadata
    config_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def validate(self):
        """Validate all configuration sections"""
        self.exchange.validate()
        self.trading.validate()
        self.risk_management.validate()
        self.backtesting.validate()
        self.monitoring.validate()
        self.alerting.validate()
        self.database.validate()
        self.tax_reporting.validate()
        self.advanced_features.validate()
        self.api.validate()


class ConfigurationManager:
    """
    Comprehensive Configuration Management System
    
    Handles loading, validation, and management of all bot configurations
    with support for multiple environments and dynamic updates.
    """
    
    def __init__(self, config_directory: str = "config"):
        self.config_directory = Path(config_directory)
        self.config_directory.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.config: Optional[ComprehensiveConfig] = None
        self.config_hash = ""
        
        # Environment variable mappings for environment-specific credentials
        self.env_mappings = {
            "INITIAL_CAPITAL": "trading.initial_capital",
            "MAX_DAILY_LOSS": "risk_management.max_daily_loss",
            "DATABASE_URL": "database.database_url",
            "EMAIL_PASSWORD": "alerting.email_password",
            "JWT_SECRET": "api.jwt_secret",
            "ENVIRONMENT": "environment"
        }
    
    def load_config(self, 
                   config_file: Optional[str] = None,
                   environment: Optional[Environment] = None) -> ComprehensiveConfig:
        """
        Load configuration from file with environment overrides
        
        Args:
            config_file: Path to configuration file
            environment: Target environment
            
        Returns:
            Loaded and validated configuration
        """
        try:
            # Determine environment
            env = environment or Environment(os.getenv("ENVIRONMENT", "development"))
            
            # Load base configuration
            if config_file:
                config_path = Path(config_file)
            else:
                config_path = self.config_directory / f"config_{env.value}.yaml"
                if not config_path.exists():
                    config_path = self.config_directory / "config.yaml"
            
            # Load configuration from file
            if config_path.exists():
                self.config = self._load_from_file(config_path)
            else:
                self.logger.warning(f"Configuration file not found: {config_path}")
                self.config = ComprehensiveConfig()
            
            # Set environment
            self.config.environment = env
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            # Load environment-specific API credentials
            self._load_environment_credentials()
            
            # Apply environment-specific settings
            self._apply_environment_settings()
            
            # Validate configuration
            self.config.validate()
            
            # Update timestamp
            self.config.updated_at = datetime.now().isoformat()
            
            # Calculate configuration hash
            self.config_hash = self._calculate_config_hash()
            
            self.logger.info(f"Configuration loaded successfully for environment: {env.value}")
            return self.config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _load_from_file(self, config_path: Path) -> ComprehensiveConfig:
        """Load configuration from YAML or JSON file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported config file format: {config_path.suffix}")
            
            # Convert flat dict to nested structure if needed
            config_dict = self._unflatten_dict(data) if isinstance(data, dict) else data
            
            # Create configuration object
            return self._dict_to_config(config_dict)
            
        except Exception as e:
            raise ConfigurationError(f"Error loading config file {config_path}: {e}")
    
    def _dict_to_config(self, data: Dict[str, Any]) -> ComprehensiveConfig:
        """Convert dictionary to ComprehensiveConfig object"""
        try:
            # Handle environment field specially
            if 'environment' in data and isinstance(data['environment'], str):
                data['environment'] = Environment(data['environment'])
            
            # Create nested configuration objects
            config_dict = {}
            
            for field_name, field_type in ComprehensiveConfig.__annotations__.items():
                if field_name in data:
                    if hasattr(field_type, '__origin__'):  # Skip generic types
                        config_dict[field_name] = data[field_name]
                    elif field_name == 'environment':
                        config_dict[field_name] = data[field_name]
                    else:
                        # Create nested config object
                        if isinstance(data[field_name], dict):
                            config_dict[field_name] = field_type(**data[field_name])
                        else:
                            config_dict[field_name] = data[field_name]
            
            return ComprehensiveConfig(**config_dict)
            
        except Exception as e:
            self.logger.error(f"Error creating config object: {e}")
            return ComprehensiveConfig()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        for env_var, config_path in self.env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert value to appropriate type
                    converted_value = self._convert_env_value(env_value, config_path)
                    
                    # Set the value in config
                    self._set_nested_value(self.config, config_path, converted_value)
                    
                    self.logger.info(f"Applied environment override: {env_var} -> {config_path}")
                    
                except Exception as e:
                    self.logger.error(f"Error applying environment override {env_var}: {e}")
    
    def _convert_env_value(self, value: str, config_path: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Numeric conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Special handling for environment enum
        if config_path == "environment":
            return Environment(value)
        
        # Default to string
        return value
    
    def _set_nested_value(self, obj: Any, path: str, value: Any):
        """Set a nested value in the configuration object"""
        parts = path.split('.')
        current = obj
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        setattr(current, parts[-1], value)
    
    def _load_environment_credentials(self):
        """Load environment-specific API credentials from environment variables"""
        if not self.config:
            return
        
        env_name = self.config.environment.value
        
        # Load testnet credentials (for development and staging)
        testnet_key = os.getenv("BYBIT_TESTNET_API_KEY")
        testnet_secret = os.getenv("BYBIT_TESTNET_API_SECRET")
        
        # Load live credentials (for production)
        live_key = os.getenv("BYBIT_LIVE_API_KEY")
        live_secret = os.getenv("BYBIT_LIVE_API_SECRET")
        
        # Apply credentials based on environment
        if env_name in ["development", "staging", "testing"]:
            if testnet_key and testnet_secret:
                if env_name not in self.config.exchange.environments:
                    self.config.exchange.environments[env_name] = EnvironmentCredentials()
                self.config.exchange.environments[env_name].api_key = testnet_key
                self.config.exchange.environments[env_name].api_secret = testnet_secret
        
        elif env_name == "production":
            if live_key and live_secret:
                if "production" not in self.config.exchange.environments:
                    self.config.exchange.environments["production"] = EnvironmentCredentials()
                self.config.exchange.environments["production"].api_key = live_key
                self.config.exchange.environments["production"].api_secret = live_secret
    
    def _apply_environment_settings(self):
        """Apply environment-specific configuration settings"""
        if not self.config:
            return
            
        env = self.config.environment
        
        if env == Environment.DEVELOPMENT:
            # Development settings - uses testnet credentials
            if hasattr(self.config, 'monitoring') and self.config.monitoring:
                self.config.monitoring.log_level = "DEBUG"
            if hasattr(self.config, 'api') and self.config.api:
                self.config.api.cors_origins = ["*"]
            if hasattr(self.config, 'database') and self.config.database:
                self.config.database.echo = True
            
        elif env == Environment.STAGING:
            # Staging settings - uses testnet credentials
            if hasattr(self.config, 'monitoring') and self.config.monitoring:
                self.config.monitoring.log_level = "INFO"
            if hasattr(self.config, 'alerting') and self.config.alerting:
                self.config.alerting.enable_email_alerts = True
            
        elif env == Environment.PRODUCTION:
            # Production settings - uses live credentials
            if hasattr(self.config, 'monitoring') and self.config.monitoring:
                self.config.monitoring.log_level = "INFO"
            if hasattr(self.config, 'api') and self.config.api:
                self.config.api.authentication_required = True
            if hasattr(self.config, 'alerting') and self.config.alerting:
                self.config.alerting.enable_email_alerts = True
            
        elif env == Environment.TESTING:
            # Testing settings - uses testnet credentials
            if hasattr(self.config, 'monitoring') and self.config.monitoring:
                self.config.monitoring.log_level = "DEBUG"
            if hasattr(self.config, 'database') and self.config.database:
                self.config.database.database_url = "sqlite:///:memory:"
    
    def save_config(self, config_path: Optional[str] = None):
        """Save current configuration to file"""
        if not self.config:
            raise ConfigurationError("No configuration to save")
        
        try:
            if config_path:
                output_path = Path(config_path)
            else:
                output_path = self.config_directory / f"config_{self.config.environment.value}.yaml"
            
            # Convert config to dictionary
            config_dict = asdict(self.config)
            
            # Handle enum serialization
            config_dict['environment'] = self.config.environment.value
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if configuration was updated
        """
        if not self.config:
            raise ConfigurationError("No configuration loaded")
        
        try:
            # Apply updates
            for path, value in updates.items():
                self._set_nested_value(self.config, path, value)
            
            # Validate updated configuration
            self.config.validate()
            
            # Update timestamp
            self.config.updated_at = datetime.now().isoformat()
            
            # Recalculate hash
            old_hash = self.config_hash
            self.config_hash = self._calculate_config_hash()
            
            self.logger.info("Configuration updated successfully")
            return old_hash != self.config_hash
            
        except Exception as e:
            raise ConfigurationError(f"Failed to update configuration: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        if not self.config:
            return {}
        
        return {
            'environment': self.config.environment.value,
            'config_version': self.config.config_version,
            'created_at': self.config.created_at,
            'updated_at': self.config.updated_at,
            'config_hash': self.config_hash,
            'trading_pairs': self.config.trading.trading_pairs,
            'initial_capital': self.config.trading.initial_capital,
            'advanced_features_enabled': {
                'regime_detection': self.config.advanced_features.enable_regime_detection,
                'portfolio_optimization': self.config.advanced_features.enable_portfolio_optimization,
                'news_analysis': self.config.advanced_features.enable_news_analysis,
                'parameter_optimization': self.config.advanced_features.enable_parameter_optimization
            },
            'risk_settings': {
                'max_portfolio_risk': self.config.risk_management.max_portfolio_risk,
                'max_daily_loss': self.config.risk_management.max_daily_loss,
                'max_drawdown': self.config.risk_management.max_drawdown
            }
        }
    
    def create_default_config_files(self):
        """Create default configuration files for all environments"""
        environments = [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION, Environment.TESTING]
        
        for env in environments:
            config = ComprehensiveConfig(environment=env)
            
            # Apply environment-specific defaults
            if env == Environment.DEVELOPMENT:
                config.trading.initial_capital = 1000.0
                config.monitoring.log_level = "DEBUG"
                
            elif env == Environment.STAGING:
                config.trading.initial_capital = 5000.0
                config.alerting.enable_email_alerts = True
                
            elif env == Environment.PRODUCTION:
                config.trading.initial_capital = 10000.0
                config.api.authentication_required = True
                config.alerting.enable_email_alerts = True
            
            # Save configuration
            config_path = self.config_directory / f"config_{env.value}.yaml"
            config_dict = asdict(config)
            config_dict['environment'] = env.value
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"Created default config file: {config_path}")
    
    def _calculate_config_hash(self) -> str:
        """Calculate hash of current configuration"""
        if not self.config:
            return ""
        
        config_str = json.dumps(asdict(self.config), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _unflatten_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat dictionary with dot notation to nested structure"""
        result = {}
        
        for key, value in data.items():
            if '.' in key:
                parts = key.split('.')
                current = result
                
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                current[parts[-1]] = value
            else:
                result[key] = value
        
        return result
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return bool(self.config and self.config.environment == Environment.PRODUCTION)
    
    @property
    def is_testnet(self) -> bool:
        """Check if using testnet based on current environment"""
        if not self.config:
            return True  # Default to testnet for safety
        
        env_name = self.config.environment.value
        try:
            credentials = self.config.exchange.get_credentials(env_name)
            return credentials.is_testnet
        except ConfigurationError:
            return True  # Default to testnet for safety
    
    def get_current_credentials(self) -> EnvironmentCredentials:
        """Get credentials for the current environment"""
        if not self.config:
            raise ConfigurationError("No configuration loaded")
        
        env_name = self.config.environment.value
        return self.config.exchange.get_credentials(env_name)


# Example usage
def main():
    """Example usage of the configuration management system"""
    print("Phase 10: Comprehensive Configuration Management System")
    print("=" * 60)
    
    # Initialize configuration manager
    config_manager = ConfigurationManager()
    
    # Create default configuration files
    print("Creating default configuration files...")
    config_manager.create_default_config_files()
    
    # Load configuration for development
    print("\nLoading development configuration...")
    config = config_manager.load_config(environment=Environment.DEVELOPMENT)
    
    # Display configuration summary
    summary = config_manager.get_config_summary()
    print(f"\nConfiguration Summary:")
    print(f"Environment: {summary['environment']}")
    print(f"Trading Pairs: {summary['trading_pairs']}")
    print(f"Initial Capital: ${summary['initial_capital']:,.2f}")
    print(f"Config Hash: {summary['config_hash']}")
    
    print(f"\nAdvanced Features:")
    for feature, enabled in summary['advanced_features_enabled'].items():
        status = "✅ Enabled" if enabled else "❌ Disabled"
        print(f"  {feature.replace('_', ' ').title()}: {status}")
    
    print(f"\nRisk Settings:")
    for setting, value in summary['risk_settings'].items():
        print(f"  {setting.replace('_', ' ').title()}: {value:.1%}")
    
    # Test configuration updates
    print(f"\nTesting configuration updates...")
    updates = {
        'trading.initial_capital': 15000.0,
        'risk_management.max_daily_loss': 0.03,
        'advanced_features.enable_news_analysis': False
    }
    
    config_changed = config_manager.update_config(updates)
    print(f"Configuration changed: {config_changed}")
    
    # Save updated configuration
    print(f"\nSaving updated configuration...")
    config_manager.save_config()
    
    print(f"\n🎉 Configuration Management System Complete!")
    print(f"✅ Multi-environment support")
    print(f"✅ Environment variable overrides")
    print(f"✅ Configuration validation")
    print(f"✅ Dynamic updates")
    print(f"✅ Default configuration generation")


if __name__ == "__main__":
    main()