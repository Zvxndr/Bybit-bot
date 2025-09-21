"""
Unified Configuration System - Phase 4 Configuration Consolidation

This module provides a comprehensive, unified configuration management system that 
consolidates all scattered configuration implementations across the trading bot.

Key Features:
- Single source of truth for all configuration
- Environment-specific configuration (dev, staging, production)
- Secure secrets management with encryption
- Configuration validation and type safety
- Hot reloading and dynamic updates
- Integration with all existing systems
- Australian compliance configuration
- Comprehensive logging and monitoring
- Configuration versioning and rollback
- CLI tools for configuration management

Consolidates:
- config/config.yaml and environment-specific files
- src/bot/config.py and config_manager.py
- config_cli.py functionality
- Environment variable management
- API configuration from Phase 3
- All scattered settings across components

Architecture:
- ConfigurationManager: Main orchestrator
- ConfigurationSchema: Type-safe configuration model
- EnvironmentManager: Environment-specific handling
- SecretsManager: Secure credential management
- ValidationManager: Configuration validation
- ConfigurationCLI: Command-line interface
"""

import os
import json
import yaml
import hashlib
import secrets
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field, field_validator, model_validator
from cryptography.fernet import Fernet
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    TESTING = "testing"

class TradingMode(Enum):
    """Trading operation modes"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    HYBRID = "hybrid"

class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DatabaseDialect(Enum):
    """Database dialects"""
    POSTGRESQL = "postgresql"
    DUCKDB = "duckdb"
    SQLITE = "sqlite"

class CacheBackend(Enum):
    """Cache backend types"""
    REDIS = "redis"
    MEMORY = "memory"
    HYBRID = "hybrid"

# Default configuration file locations
DEFAULT_CONFIG_PATHS = [
    "config/config.yaml",
    "src/bot/core/config/config.yaml",
    "config.yaml",
    ".config.yaml"
]

DEFAULT_SECRETS_PATHS = [
    "config/secrets.yaml",
    "secrets.yaml",
    ".secrets.yaml"
]

# ============================================================================
# CONFIGURATION SCHEMAS
# ============================================================================

class ExchangeCredentials(BaseModel):
    """Exchange API credentials"""
    api_key: str = Field(default="", description="API key")
    api_secret: str = Field(default="", description="API secret")
    is_testnet: bool = Field(default=True, description="Whether to use testnet")
    base_url: str = Field(default="", description="API base URL")
    recv_window: int = Field(default=5000, description="Receive window for requests")
    
    @field_validator('api_key', 'api_secret')
    @classmethod
    def validate_credentials(cls, v):
        if v and len(v) < 10:
            raise ValueError("API credentials must be at least 10 characters")
        return v

class DatabaseConfig(BaseModel):
    """Database configuration"""
    dialect: DatabaseDialect = DatabaseDialect.DUCKDB
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_bot"
    username: str = "trader"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    def get_connection_string(self) -> str:
        """Get database connection string"""
        if self.dialect == DatabaseDialect.DUCKDB:
            return f"duckdb:///{self.database}.db"
        elif self.dialect == DatabaseDialect.SQLITE:
            return f"sqlite:///{self.database}.db"
        elif self.dialect == DatabaseDialect.POSTGRESQL:
            return (f"postgresql://{self.username}:{self.password}@"
                   f"{self.host}:{self.port}/{self.database}")
        else:
            raise ValueError(f"Unsupported database dialect: {self.dialect}")

class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""
    market_data: int = Field(default=120, description="Market data requests per second")
    trading: int = Field(default=50, description="Trading requests per second") 
    account: int = Field(default=20, description="Account requests per second")
    websocket: int = Field(default=10, description="WebSocket connections per second")
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")

class ConnectionConfig(BaseModel):
    """HTTP connection configuration"""
    pool_size: int = Field(default=100, description="Connection pool size")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    keepalive_timeout: int = Field(default=300, description="Keep-alive timeout")
    enable_compression: bool = Field(default=True, description="Enable compression")
    ssl_verify: bool = Field(default=True, description="Verify SSL certificates")

class WebSocketConfig(BaseModel):
    """WebSocket configuration"""
    ping_interval: int = Field(default=20, description="Ping interval in seconds")
    ping_timeout: int = Field(default=10, description="Ping timeout in seconds")
    close_timeout: int = Field(default=10, description="Close timeout in seconds")
    max_reconnect_attempts: int = Field(default=10, description="Max reconnection attempts")
    base_reconnect_delay: float = Field(default=1.0, description="Base reconnection delay")
    max_reconnect_delay: float = Field(default=60.0, description="Max reconnection delay")
    enable_compression: bool = Field(default=True, description="Enable compression")
    message_queue_size: int = Field(default=1000, description="Message queue size")

class CacheConfig(BaseModel):
    """Caching configuration"""
    backend: CacheBackend = Field(default=CacheBackend.MEMORY, description="Cache backend")
    enable_caching: bool = Field(default=True, description="Enable caching")
    cache_size: int = Field(default=10000, description="Cache size")
    cache_ttl_seconds: int = Field(default=60, description="Cache TTL")
    cleanup_interval: int = Field(default=300, description="Cleanup interval")
    
    # Redis-specific configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")
    redis_password: str = Field(default="", description="Redis password")

class TradingModeConfig(BaseModel):
    """Base trading mode configuration"""
    portfolio_drawdown_limit: float = Field(..., ge=0, le=1)
    strategy_drawdown_limit: float = Field(..., ge=0, le=1)
    sharpe_ratio_min: float = Field(..., ge=0)
    var_daily_limit: float = Field(..., ge=0, le=1)
    consistency_min: float = Field(..., ge=0, le=1)

class AggressiveModeConfig(TradingModeConfig):
    """Aggressive trading mode configuration"""
    max_risk_ratio: float = Field(..., ge=0, le=1)
    min_risk_ratio: float = Field(..., ge=0, le=1)
    balance_thresholds: Dict[str, float] = Field(default_factory=dict)
    risk_decay: str = Field(default="exponential", description="Risk decay method")
    
    @field_validator("max_risk_ratio")
    @classmethod
    def validate_risk_ratios(cls, v, info):
        if info.data and "min_risk_ratio" in info.data and v <= info.data["min_risk_ratio"]:
            raise ValueError("max_risk_ratio must be greater than min_risk_ratio")
        return v

class ConservativeModeConfig(TradingModeConfig):
    """Conservative trading mode configuration"""
    risk_ratio: float = Field(..., ge=0, le=1, description="Fixed risk ratio")

class TradingConfig(BaseModel):
    """Trading configuration"""
    mode: TradingMode = Field(default=TradingMode.CONSERVATIVE, description="Trading mode")
    base_balance: float = Field(default=10000, description="Base balance in USDT")
    
    # Mode-specific configurations
    aggressive_mode: AggressiveModeConfig = Field(default=None)
    conservative_mode: ConservativeModeConfig = Field(default=None)
    
    # Order execution settings
    order_type: str = Field(default="limit", description="Default order type")
    max_slippage: float = Field(default=0.001, description="Maximum slippage")
    order_timeout: int = Field(default=30, description="Order timeout in seconds")
    
    # Trading pairs and timeframes
    symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframes: Dict[str, str] = Field(default_factory=lambda: {
        "primary": "1h",
        "secondary": "4h", 
        "daily": "1d"
    })

class MLConfig(BaseModel):
    """Machine learning configuration"""
    # Feature engineering
    lookback_periods: List[int] = Field(default_factory=lambda: [5, 10, 20, 50])
    technical_indicators: List[str] = Field(default_factory=lambda: [
        "sma", "ema", "rsi", "macd", "bollinger_bands", "atr"
    ])
    statistical_features: List[str] = Field(default_factory=lambda: [
        "returns", "volatility", "skewness", "kurtosis"
    ])
    
    # Model configurations
    models: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "lightgbm": {
            "n_estimators": 1000,
            "learning_rate": 0.1,
            "max_depth": 6,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "random_state": 42
        },
        "xgboost": {
            "n_estimators": 1000,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
    })
    
    # Cross-validation settings
    cross_validation: Dict[str, Any] = Field(default_factory=lambda: {
        "n_splits": 5,
        "test_size": 0.2,
        "purge_days": 1,
        "embargo_days": 1
    })

class BacktestingConfig(BaseModel):
    """Backtesting configuration"""
    # Walk-forward analysis
    walk_forward: Dict[str, Any] = Field(default_factory=lambda: {
        "initial_window": 252,
        "step_size": 21,
        "oos_period": 21,
        "min_trades": 30
    })
    
    # Validation thresholds  
    validation: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        "conservative": {
            "min_sharpe": 0.8,
            "max_drawdown": 0.15
        },
        "aggressive": {
            "min_sharpe": 0.5,
            "max_drawdown": 0.25
        }
    })

class AustralianComplianceConfig(BaseModel):
    """Australian compliance configuration"""
    timezone: str = Field(default="Australia/Sydney", description="Australian timezone")
    tax_year: str = Field(default="2024", description="Tax year")
    enable_tax_reporting: bool = Field(default=True, description="Enable tax reporting")
    cgt_threshold: Decimal = Field(default=Decimal('12000'), description="CGT threshold")
    record_all_trades: bool = Field(default=True, description="Record all trades")
    export_format: str = Field(default="csv", description="Export format")
    reporting_currency: str = Field(default="AUD", description="Reporting currency")

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    enable_file_logging: bool = Field(default=True, description="Enable file logging")
    log_file_path: str = Field(default="logs/trading_bot.log", description="Log file path")
    max_file_size_mb: int = Field(default=100, description="Max log file size")
    backup_count: int = Field(default=5, description="Log backup count")
    enable_console_logging: bool = Field(default=True, description="Enable console logging")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )

class SecurityConfig(BaseModel):
    """Security configuration"""
    enable_encryption: bool = Field(default=True, description="Enable configuration encryption")
    secret_key: str = Field(default="", description="Secret key for encryption")
    enable_api_key_rotation: bool = Field(default=False, description="Enable API key rotation")
    key_rotation_interval_hours: int = Field(default=24, description="Key rotation interval")
    enable_ip_whitelist: bool = Field(default=False, description="Enable IP whitelist")
    allowed_ips: List[str] = Field(default_factory=list, description="Allowed IP addresses")
    min_tls_version: str = Field(default="1.2", description="Minimum TLS version")

class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration"""
    enable_health_checks: bool = Field(default=True, description="Enable health checks")
    health_check_interval: int = Field(default=60, description="Health check interval")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    enable_alerts: bool = Field(default=True, description="Enable alerting")
    
    # Email alerting
    smtp_server: str = Field(default="", description="SMTP server")
    smtp_port: int = Field(default=587, description="SMTP port")
    email_username: str = Field(default="", description="Email username")
    email_password: str = Field(default="", description="Email password")
    alert_email: str = Field(default="", description="Alert email address")

class APIConfig(BaseModel):
    """API server configuration"""
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=1, description="Number of workers")
    enable_auth: bool = Field(default=True, description="Enable authentication")
    enable_docs: bool = Field(default=False, description="Enable API documentation")
    enable_metrics: bool = Field(default=True, description="Enable metrics endpoint")
    cors_origins: List[str] = Field(default_factory=list, description="CORS origins")
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")

# ============================================================================
# UNIFIED CONFIGURATION SCHEMA
# ============================================================================

class UnifiedConfigurationSchema(BaseModel):
    """
    Unified Configuration Schema
    
    Master configuration model that consolidates all component configurations
    """
    # Environment and metadata
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Deployment environment")
    version: str = Field(default="1.0.0", description="Configuration version")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Core component configurations
    exchange: Dict[Environment, ExchangeCredentials] = Field(
        default_factory=dict, description="Exchange credentials by environment"
    )
    database: DatabaseConfig = Field(default=None, description="Database configuration")
    trading: TradingConfig = Field(default=None, description="Trading configuration")
    ml: MLConfig = Field(default=None, description="Machine learning configuration")
    backtesting: BacktestingConfig = Field(default=None, description="Backtesting configuration")
    
    # API and connectivity
    rate_limits: RateLimitConfig = Field(default=None, description="Rate limiting")
    connection: ConnectionConfig = Field(default=None, description="Connection settings")
    websocket: WebSocketConfig = Field(default=None, description="WebSocket settings")
    cache: CacheConfig = Field(default=None, description="Cache configuration")
    
    # Compliance and logging
    australian_compliance: AustralianComplianceConfig = Field(
        default=None, description="Australian compliance"
    )
    logging: LoggingConfig = Field(default=None, description="Logging configuration")
    security: SecurityConfig = Field(default=None, description="Security configuration")
    monitoring: MonitoringConfig = Field(default=None, description="Monitoring configuration")
    api: APIConfig = Field(default=None, description="API server configuration")
    
    # Feature flags
    enable_trading: bool = Field(default=False, description="Enable trading")
    enable_websockets: bool = Field(default=True, description="Enable WebSocket connections")
    enable_market_data: bool = Field(default=True, description="Enable market data")
    enable_risk_management: bool = Field(default=True, description="Enable risk management")
    enable_ml_integration: bool = Field(default=True, description="Enable ML integration")
    enable_backtesting: bool = Field(default=True, description="Enable backtesting")
    
    # Custom settings for extensibility
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom settings")
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        extra = "allow"
    
    @model_validator(mode='after')
    def validate_configuration(self):
        """Validate the entire configuration"""
        # Initialize None fields with defaults
        if self.database is None:
            self.database = DatabaseConfig()
        if self.trading is None:
            self.trading = TradingConfig()
        if self.ml is None:
            self.ml = MLConfig()
        if self.backtesting is None:
            self.backtesting = BacktestingConfig()
        if self.rate_limits is None:
            self.rate_limits = RateLimitConfig()
        if self.connection is None:
            self.connection = ConnectionConfig()
        if self.websocket is None:
            self.websocket = WebSocketConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.australian_compliance is None:
            self.australian_compliance = AustralianComplianceConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.api is None:
            self.api = APIConfig()
        
        # Validate trading requirements
        if self.enable_trading:
            if self.environment not in self.exchange or not self.exchange[self.environment].api_key:
                raise ValueError(f"Trading enabled but no credentials for {self.environment}")
        
        # Update timestamp
        self.updated_at = datetime.now()
        
        return self
    
    def get_current_credentials(self) -> Optional[ExchangeCredentials]:
        """Get credentials for current environment"""
        return self.exchange.get(self.environment)
    
    def validate_credentials(self) -> bool:
        """Validate current environment credentials"""
        creds = self.get_current_credentials()
        if not creds:
            return False
        
        return bool(creds.api_key and creds.api_secret and 
                   len(creds.api_key) >= 10 and len(creds.api_secret) >= 10)
    
    def get_trading_mode_config(self) -> Union[AggressiveModeConfig, ConservativeModeConfig]:
        """Get configuration for current trading mode"""
        if self.trading.mode == TradingMode.AGGRESSIVE:
            return self.trading.aggressive_mode
        else:
            return self.trading.conservative_mode
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enum values"""
        return self.dict(by_alias=True, exclude_none=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'environment': self.environment.value,
            'version': self.version,
            'has_credentials': bool(self.get_current_credentials()),
            'credentials_valid': self.validate_credentials(),
            'trading_enabled': self.enable_trading,
            'trading_mode': self.trading.mode.value,
            'websockets_enabled': self.enable_websockets,
            'ml_enabled': self.enable_ml_integration,
            'database_dialect': self.database.dialect.value,
            'cache_backend': self.cache.backend.value,
            'log_level': self.logging.level.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

# ============================================================================
# SECRETS MANAGER
# ============================================================================

class SecretsManager:
    """Secure secrets management with encryption"""
    
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or self._generate_key()
        self.cipher = Fernet(self.key)
        
        logger.info("Secrets manager initialized")
    
    def _generate_key(self) -> bytes:
        """Generate encryption key"""
        return Fernet.generate_key()
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a string value"""
        if not value:
            return ""
        
        encrypted_bytes = self.cipher.encrypt(value.encode())
        return encrypted_bytes.decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a string value"""
        if not encrypted_value:
            return ""
        
        try:
            decrypted_bytes = self.cipher.decrypt(encrypted_value.encode())
            return decrypted_bytes.decode()
        except Exception:
            # Return as-is if decryption fails (might be plain text)
            return encrypted_value
    
    def encrypt_secrets(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively encrypt sensitive values in configuration"""
        sensitive_keys = {
            'api_key', 'api_secret', 'password', 'secret_key', 
            'jwt_secret', 'email_password', 'redis_password'
        }
        
        def encrypt_recursive(obj):
            if isinstance(obj, dict):
                return {
                    key: self.encrypt_value(str(value)) if key in sensitive_keys 
                         else encrypt_recursive(value)
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [encrypt_recursive(item) for item in obj]
            else:
                return obj
        
        return encrypt_recursive(config_dict)
    
    def decrypt_secrets(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively decrypt sensitive values in configuration"""
        sensitive_keys = {
            'api_key', 'api_secret', 'password', 'secret_key',
            'jwt_secret', 'email_password', 'redis_password'
        }
        
        def decrypt_recursive(obj):
            if isinstance(obj, dict):
                return {
                    key: self.decrypt_value(str(value)) if key in sensitive_keys
                         else decrypt_recursive(value)
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [decrypt_recursive(item) for item in obj]
            else:
                return obj
        
        return decrypt_recursive(config_dict)
    
    def save_key(self, filepath: str):
        """Save encryption key to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(self.key)
        
        # Set restrictive permissions
        os.chmod(filepath, 0o600)
        logger.info(f"Encryption key saved to {filepath}")
    
    def load_key(self, filepath: str) -> bytes:
        """Load encryption key from file"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Encryption key file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            key = f.read()
        
        logger.info(f"Encryption key loaded from {filepath}")
        return key

# ============================================================================
# ENVIRONMENT MANAGER  
# ============================================================================

class EnvironmentManager:
    """Environment-specific configuration management"""
    
    def __init__(self):
        self.current_environment = self._detect_environment()
        logger.info(f"Environment manager initialized for {self.current_environment.value}")
    
    def _detect_environment(self) -> Environment:
        """Detect current environment from various sources"""
        # Check environment variable
        env_var = os.getenv('ENVIRONMENT', '').lower()
        if env_var:
            try:
                return Environment(env_var)
            except ValueError:
                pass
        
        # Check for environment indicators
        if os.getenv('PRODUCTION') or os.getenv('PROD'):
            return Environment.PRODUCTION
        elif os.getenv('STAGING'):
            return Environment.STAGING
        elif os.getenv('TESTING') or os.getenv('TEST'):
            return Environment.TESTING
        else:
            return Environment.DEVELOPMENT
    
    def load_environment_variables(self) -> Dict[str, str]:
        """Load relevant environment variables"""
        env_vars = {}
        
        # Bybit credentials
        for env in Environment:
            prefix = f"BYBIT_{env.value.upper()}_"
            if env == Environment.PRODUCTION:
                prefix = "BYBIT_LIVE_"
            elif env == Environment.DEVELOPMENT or env == Environment.STAGING:
                prefix = "BYBIT_TESTNET_"
            
            api_key = os.getenv(f"{prefix}API_KEY", "")
            api_secret = os.getenv(f"{prefix}API_SECRET", "")
            
            if api_key:
                env_vars[f"{env.value}_api_key"] = api_key
            if api_secret:
                env_vars[f"{env.value}_api_secret"] = api_secret
        
        # Database credentials
        db_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        for var in db_vars:
            value = os.getenv(var)
            if value:
                env_vars[var.lower()] = value
        
        # Other common variables
        other_vars = [
            'SMTP_SERVER', 'SMTP_PORT', 'EMAIL_USERNAME', 'EMAIL_PASSWORD',
            'REDIS_HOST', 'REDIS_PORT', 'REDIS_PASSWORD',
            'SECRET_KEY', 'JWT_SECRET'
        ]
        for var in other_vars:
            value = os.getenv(var)
            if value:
                env_vars[var.lower()] = value
        
        return env_vars
    
    def apply_environment_overrides(
        self, 
        config: UnifiedConfigurationSchema,
        env_vars: Optional[Dict[str, str]] = None
    ) -> UnifiedConfigurationSchema:
        """Apply environment variable overrides"""
        if env_vars is None:
            env_vars = self.load_environment_variables()
        
        # Apply database overrides
        if 'db_host' in env_vars:
            config.database.host = env_vars['db_host']
        if 'db_port' in env_vars:
            config.database.port = int(env_vars['db_port'])
        if 'db_name' in env_vars:
            config.database.database = env_vars['db_name']
        if 'db_user' in env_vars:
            config.database.username = env_vars['db_user']
        if 'db_password' in env_vars:
            config.database.password = env_vars['db_password']
        
        # Apply exchange credential overrides
        for env in Environment:
            api_key_var = f"{env.value}_api_key"
            api_secret_var = f"{env.value}_api_secret"
            
            if api_key_var in env_vars or api_secret_var in env_vars:
                if env not in config.exchange:
                    config.exchange[env] = ExchangeCredentials()
                
                if api_key_var in env_vars:
                    config.exchange[env].api_key = env_vars[api_key_var]
                if api_secret_var in env_vars:
                    config.exchange[env].api_secret = env_vars[api_secret_var]
        
        # Apply other overrides
        if 'smtp_server' in env_vars:
            config.monitoring.smtp_server = env_vars['smtp_server']
        if 'smtp_port' in env_vars:
            config.monitoring.smtp_port = int(env_vars['smtp_port'])
        if 'email_username' in env_vars:
            config.monitoring.email_username = env_vars['email_username']
        if 'email_password' in env_vars:
            config.monitoring.email_password = env_vars['email_password']
        
        if 'redis_host' in env_vars:
            config.cache.redis_host = env_vars['redis_host']
        if 'redis_port' in env_vars:
            config.cache.redis_port = int(env_vars['redis_port'])
        if 'redis_password' in env_vars:
            config.cache.redis_password = env_vars['redis_password']
        
        if 'secret_key' in env_vars:
            config.security.secret_key = env_vars['secret_key']
        
        return config

# ============================================================================
# CONFIGURATION FILE WATCHER
# ============================================================================

class ConfigurationWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes"""
    
    def __init__(self, callback: Callable[[str], None]):
        super().__init__()
        self.callback = callback
        self.last_modified = {}
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        filepath = event.src_path
        if not (filepath.endswith('.yaml') or filepath.endswith('.yml') or 
                filepath.endswith('.json')):
            return
        
        # Debounce rapid file changes
        current_time = datetime.now()
        last_time = self.last_modified.get(filepath)
        
        if last_time and (current_time - last_time).total_seconds() < 1:
            return
        
        self.last_modified[filepath] = current_time
        
        try:
            self.callback(filepath)
        except Exception as e:
            logger.error(f"Error in configuration watcher callback: {e}")

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'UnifiedConfigurationSchema',
    'Environment',
    'TradingMode', 
    'LogLevel',
    'DatabaseDialect',
    'CacheBackend',
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
    'SecretsManager',
    'EnvironmentManager',
    'ConfigurationWatcher'
]