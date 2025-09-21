"""
Unified API Configuration - Phase 3 API Consolidation

This module provides centralized configuration management for all API operations.
It handles credentials, environment settings, rate limits, connection parameters,
and Australian compliance requirements in a unified system.

Key Features:
- Environment-specific configurations (testnet/mainnet)
- Secure credential management
- Rate limiting configuration
- Connection pooling settings
- WebSocket configuration
- Australian compliance settings
- Configuration validation
- Environment variable support
- Configuration hot-reloading
"""

import os
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path

from .unified_bybit_client import Environment, BybitCredentials

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AustralianTaxYear(Enum):
    """Australian tax year periods"""
    FY2023 = "2023"
    FY2024 = "2024"
    FY2025 = "2025"

# Default rate limits (requests per second)
DEFAULT_RATE_LIMITS = {
    'market_data': 120,
    'trading': 50,
    'account': 20,
    'websocket': 10
}

# Default connection settings
DEFAULT_CONNECTION_SETTINGS = {
    'pool_size': 100,
    'timeout': 30,
    'max_retries': 3,
    'keepalive_timeout': 300
}

# Default WebSocket settings
DEFAULT_WEBSOCKET_SETTINGS = {
    'ping_interval': 20,
    'ping_timeout': 10,
    'close_timeout': 10,
    'max_reconnect_attempts': 10,
    'base_reconnect_delay': 1.0,
    'max_reconnect_delay': 60.0
}

# ============================================================================
# CONFIGURATION DATA CLASSES
# ============================================================================

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    market_data: int = 120
    trading: int = 50
    account: int = 20
    websocket: int = 10
    enable_rate_limiting: bool = True
    
    def __post_init__(self):
        if self.market_data <= 0 or self.trading <= 0 or self.account <= 0:
            raise ValueError("Rate limits must be positive integers")

@dataclass
class ConnectionConfig:
    """HTTP connection configuration"""
    pool_size: int = 100
    timeout: int = 30
    max_retries: int = 3
    keepalive_timeout: int = 300
    enable_compression: bool = True
    ssl_verify: bool = True
    
    def __post_init__(self):
        if self.pool_size <= 0:
            raise ValueError("Pool size must be positive")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")

@dataclass
class WebSocketConfig:
    """WebSocket configuration"""
    ping_interval: int = 20
    ping_timeout: int = 10
    close_timeout: int = 10
    max_reconnect_attempts: int = 10
    base_reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    enable_compression: bool = True
    message_queue_size: int = 1000
    
    def __post_init__(self):
        if self.ping_interval <= 0 or self.ping_timeout <= 0:
            raise ValueError("Ping intervals must be positive")
        if self.max_reconnect_attempts < 0:
            raise ValueError("Max reconnect attempts cannot be negative")
        if self.base_reconnect_delay <= 0:
            raise ValueError("Base reconnect delay must be positive")

@dataclass
class CacheConfig:
    """Data caching configuration"""
    enable_caching: bool = True
    cache_size: int = 10000
    cache_ttl_seconds: int = 60
    ticker_ttl: int = 10
    orderbook_ttl: int = 5
    trade_ttl: int = 60
    cleanup_interval: int = 300
    
    def __post_init__(self):
        if self.cache_size <= 0:
            raise ValueError("Cache size must be positive")
        if self.cache_ttl_seconds <= 0:
            raise ValueError("Cache TTL must be positive")

@dataclass
class AustralianComplianceConfig:
    """Australian compliance and tax configuration"""
    timezone: str = "Australia/Sydney"
    tax_year: AustralianTaxYear = AustralianTaxYear.FY2024
    enable_tax_reporting: bool = True
    cgt_threshold: Decimal = Decimal('12000')  # CGT discount threshold
    record_all_trades: bool = True
    export_format: str = "csv"
    reporting_currency: str = "AUD"
    
    def __post_init__(self):
        if self.cgt_threshold < 0:
            raise ValueError("CGT threshold cannot be negative")

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    enable_file_logging: bool = True
    log_file_path: str = "logs/bybit_bot.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    enable_console_logging: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        if self.max_file_size_mb <= 0:
            raise ValueError("Max file size must be positive")
        if self.backup_count < 0:
            raise ValueError("Backup count cannot be negative")

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_api_key_rotation: bool = False
    key_rotation_interval_hours: int = 24
    enable_ip_whitelist: bool = False
    allowed_ips: List[str] = field(default_factory=list)
    enable_request_signing: bool = True
    min_tls_version: str = "1.2"
    
    def __post_init__(self):
        if self.key_rotation_interval_hours <= 0:
            raise ValueError("Key rotation interval must be positive")

@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    enable_batching: bool = True
    batch_size: int = 100
    batch_timeout_ms: int = 1000
    enable_parallel_requests: bool = True
    max_parallel_requests: int = 10
    enable_request_prioritization: bool = True
    
    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_parallel_requests <= 0:
            raise ValueError("Max parallel requests must be positive")

# ============================================================================
# UNIFIED API CONFIGURATION
# ============================================================================

@dataclass
class UnifiedAPIConfig:
    """
    Unified API Configuration
    
    Centralized configuration for all API operations
    """
    # Environment and credentials
    environment: Environment = Environment.TESTNET
    credentials: Optional[BybitCredentials] = None
    
    # Component configurations
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    australian_compliance: AustralianComplianceConfig = field(default_factory=AustralianComplianceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Feature flags
    enable_websockets: bool = True
    enable_market_data: bool = True
    enable_trading: bool = True
    enable_account_monitoring: bool = True
    enable_risk_management: bool = True
    enable_ml_integration: bool = True
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate the entire configuration"""
        # Validate credentials if trading is enabled
        if self.enable_trading and not self.credentials:
            raise ValueError("Credentials required when trading is enabled")
        
        # Validate Australian compliance settings
        if self.australian_compliance.enable_tax_reporting:
            if not self.australian_compliance.timezone.startswith('Australia/'):
                logger.warning("Non-Australian timezone specified for tax reporting")
        
        # Validate performance settings
        if (self.performance.enable_parallel_requests and 
            self.performance.max_parallel_requests > self.connection.pool_size):
            logger.warning("Max parallel requests exceeds connection pool size")
        
        # Log configuration summary
        logger.info(f"Configuration validated for {self.environment.value} environment")
    
    @classmethod
    def from_environment(cls) -> 'UnifiedAPIConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Environment
        env_name = os.getenv('BYBIT_ENVIRONMENT', 'testnet').lower()
        if env_name in ['mainnet', 'main', 'prod', 'production']:
            config.environment = Environment.MAINNET
        else:
            config.environment = Environment.TESTNET
        
        # Credentials
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')
        
        if api_key and api_secret:
            config.credentials = BybitCredentials(
                api_key=api_key,
                api_secret=api_secret,
                environment=config.environment,
                recv_window=int(os.getenv('BYBIT_RECV_WINDOW', '5000'))
            )
        
        # Rate limits
        config.rate_limits.market_data = int(os.getenv('BYBIT_RATE_LIMIT_MARKET_DATA', '120'))
        config.rate_limits.trading = int(os.getenv('BYBIT_RATE_LIMIT_TRADING', '50'))
        config.rate_limits.account = int(os.getenv('BYBIT_RATE_LIMIT_ACCOUNT', '20'))
        config.rate_limits.enable_rate_limiting = os.getenv('BYBIT_ENABLE_RATE_LIMITING', 'true').lower() == 'true'
        
        # Connection settings
        config.connection.pool_size = int(os.getenv('BYBIT_CONNECTION_POOL_SIZE', '100'))
        config.connection.timeout = int(os.getenv('BYBIT_CONNECTION_TIMEOUT', '30'))
        config.connection.max_retries = int(os.getenv('BYBIT_MAX_RETRIES', '3'))
        
        # WebSocket settings
        config.websocket.ping_interval = int(os.getenv('BYBIT_WS_PING_INTERVAL', '20'))
        config.websocket.max_reconnect_attempts = int(os.getenv('BYBIT_WS_MAX_RECONNECT', '10'))
        
        # Cache settings
        config.cache.enable_caching = os.getenv('BYBIT_ENABLE_CACHING', 'true').lower() == 'true'
        config.cache.cache_size = int(os.getenv('BYBIT_CACHE_SIZE', '10000'))
        config.cache.cache_ttl_seconds = int(os.getenv('BYBIT_CACHE_TTL', '60'))
        
        # Australian compliance
        config.australian_compliance.timezone = os.getenv('BYBIT_TIMEZONE', 'Australia/Sydney')
        config.australian_compliance.enable_tax_reporting = os.getenv('BYBIT_ENABLE_TAX_REPORTING', 'true').lower() == 'true'
        config.australian_compliance.reporting_currency = os.getenv('BYBIT_REPORTING_CURRENCY', 'AUD')
        
        # Logging
        log_level = os.getenv('BYBIT_LOG_LEVEL', 'INFO').upper()
        if hasattr(LogLevel, log_level):
            config.logging.level = LogLevel[log_level]
        
        config.logging.enable_file_logging = os.getenv('BYBIT_ENABLE_FILE_LOGGING', 'true').lower() == 'true'
        config.logging.log_file_path = os.getenv('BYBIT_LOG_FILE_PATH', 'logs/bybit_bot.log')
        
        # Feature flags
        config.enable_websockets = os.getenv('BYBIT_ENABLE_WEBSOCKETS', 'true').lower() == 'true'
        config.enable_trading = os.getenv('BYBIT_ENABLE_TRADING', 'false').lower() == 'true'
        config.enable_risk_management = os.getenv('BYBIT_ENABLE_RISK_MANAGEMENT', 'true').lower() == 'true'
        config.enable_ml_integration = os.getenv('BYBIT_ENABLE_ML_INTEGRATION', 'true').lower() == 'true'
        
        logger.info("Configuration loaded from environment variables")
        return config
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'UnifiedAPIConfig':
        """Load configuration from JSON file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedAPIConfig':
        """Create configuration from dictionary"""
        # Handle environment
        env_str = data.get('environment', 'testnet')
        if isinstance(env_str, str):
            environment = Environment.MAINNET if env_str.lower() in ['mainnet', 'main'] else Environment.TESTNET
        else:
            environment = env_str
        
        # Handle credentials
        credentials = None
        if 'credentials' in data:
            cred_data = data['credentials']
            credentials = BybitCredentials(
                api_key=cred_data['api_key'],
                api_secret=cred_data['api_secret'],
                environment=environment,
                recv_window=cred_data.get('recv_window', 5000)
            )
        
        # Create component configs
        components = {}
        
        if 'rate_limits' in data:
            components['rate_limits'] = RateLimitConfig(**data['rate_limits'])
        
        if 'connection' in data:
            components['connection'] = ConnectionConfig(**data['connection'])
        
        if 'websocket' in data:
            components['websocket'] = WebSocketConfig(**data['websocket'])
        
        if 'cache' in data:
            components['cache'] = CacheConfig(**data['cache'])
        
        if 'australian_compliance' in data:
            ac_data = data['australian_compliance'].copy()
            if 'tax_year' in ac_data and isinstance(ac_data['tax_year'], str):
                ac_data['tax_year'] = AustralianTaxYear(ac_data['tax_year'])
            if 'cgt_threshold' in ac_data:
                ac_data['cgt_threshold'] = Decimal(str(ac_data['cgt_threshold']))
            components['australian_compliance'] = AustralianComplianceConfig(**ac_data)
        
        if 'logging' in data:
            log_data = data['logging'].copy()
            if 'level' in log_data and isinstance(log_data['level'], str):
                log_data['level'] = LogLevel(log_data['level'].upper())
            components['logging'] = LoggingConfig(**log_data)
        
        if 'security' in data:
            components['security'] = SecurityConfig(**data['security'])
        
        if 'performance' in data:
            components['performance'] = PerformanceConfig(**data['performance'])
        
        # Create main config
        config = cls(
            environment=environment,
            credentials=credentials,
            **components
        )
        
        # Set feature flags
        for flag in ['enable_websockets', 'enable_market_data', 'enable_trading', 
                    'enable_account_monitoring', 'enable_risk_management', 'enable_ml_integration']:
            if flag in data:
                setattr(config, flag, data[flag])
        
        # Set custom settings
        if 'custom_settings' in data:
            config.custom_settings = data['custom_settings']
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = asdict(self)
        
        # Convert enums to strings
        result['environment'] = self.environment.value
        result['australian_compliance']['tax_year'] = self.australian_compliance.tax_year.value
        result['logging']['level'] = self.logging.level.value
        
        # Convert Decimal to string
        result['australian_compliance']['cgt_threshold'] = str(self.australian_compliance.cgt_threshold)
        
        # Handle credentials separately for security
        if self.credentials:
            result['credentials'] = {
                'api_key': '***REDACTED***',
                'api_secret': '***REDACTED***',
                'environment': self.credentials.environment.value,
                'recv_window': self.credentials.recv_window
            }
        
        return result
    
    def save_to_file(self, file_path: Union[str, Path], include_credentials: bool = False):
        """Save configuration to JSON file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        # Include actual credentials if requested (be careful!)
        if include_credentials and self.credentials:
            data['credentials'] = {
                'api_key': self.credentials.api_key,
                'api_secret': self.credentials.api_secret,
                'environment': self.credentials.environment.value,
                'recv_window': self.credentials.recv_window
            }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
    
    def update_credentials(self, api_key: str, api_secret: str):
        """Update API credentials"""
        self.credentials = BybitCredentials(
            api_key=api_key,
            api_secret=api_secret,
            environment=self.environment
        )
        logger.info("API credentials updated")
    
    def validate_credentials(self) -> bool:
        """Validate API credentials format"""
        if not self.credentials:
            return False
        
        try:
            # Basic format validation
            if (len(self.credentials.api_key) < 10 or 
                len(self.credentials.api_secret) < 10):
                return False
            
            if not all(c.isalnum() or c in '.-_' for c in self.credentials.api_key):
                return False
            
            return True
        except Exception:
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'environment': self.environment.value,
            'has_credentials': self.credentials is not None,
            'credentials_valid': self.validate_credentials(),
            'rate_limiting_enabled': self.rate_limits.enable_rate_limiting,
            'websockets_enabled': self.enable_websockets,
            'trading_enabled': self.enable_trading,
            'risk_management_enabled': self.enable_risk_management,
            'ml_integration_enabled': self.enable_ml_integration,
            'caching_enabled': self.cache.enable_caching,
            'tax_reporting_enabled': self.australian_compliance.enable_tax_reporting,
            'timezone': self.australian_compliance.timezone,
            'log_level': self.logging.level.value,
            'connection_pool_size': self.connection.pool_size,
            'cache_size': self.cache.cache_size
        }

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigurationManager:
    """Configuration manager with hot-reloading support"""
    
    def __init__(self, config: Optional[UnifiedAPIConfig] = None):
        self.config = config or UnifiedAPIConfig()
        self.config_file: Optional[Path] = None
        self.last_modified: Optional[datetime] = None
        self.reload_callbacks: List[Callable] = []
        
        logger.info("Configuration manager initialized")
    
    def load_from_file(self, file_path: Union[str, Path], watch_for_changes: bool = True):
        """Load configuration from file with optional watching"""
        file_path = Path(file_path)
        
        self.config = UnifiedAPIConfig.from_file(file_path)
        self.config_file = file_path
        self.last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        if watch_for_changes:
            # Start background task to watch for changes
            # Note: In a real implementation, you might use watchdog library
            pass
        
        logger.info(f"Configuration loaded from {file_path}")
    
    def reload_if_changed(self) -> bool:
        """Reload configuration if file has changed"""
        if not self.config_file or not self.last_modified:
            return False
        
        try:
            current_modified = datetime.fromtimestamp(self.config_file.stat().st_mtime)
            
            if current_modified > self.last_modified:
                logger.info("Configuration file changed, reloading...")
                
                old_config = self.config
                self.config = UnifiedAPIConfig.from_file(self.config_file)
                self.last_modified = current_modified
                
                # Call reload callbacks
                for callback in self.reload_callbacks:
                    try:
                        callback(old_config, self.config)
                    except Exception as e:
                        logger.error(f"Error in reload callback: {e}")
                
                logger.info("Configuration reloaded successfully")
                return True
        
        except Exception as e:
            logger.error(f"Error checking configuration file: {e}")
        
        return False
    
    def add_reload_callback(self, callback: Callable):
        """Add callback for configuration reloads"""
        self.reload_callbacks.append(callback)
    
    def get_config(self) -> UnifiedAPIConfig:
        """Get current configuration (with auto-reload check)"""
        self.reload_if_changed()
        return self.config

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_default_config(environment: Environment = Environment.TESTNET) -> UnifiedAPIConfig:
    """Create default configuration"""
    return UnifiedAPIConfig(environment=environment)

def create_production_config(api_key: str, api_secret: str) -> UnifiedAPIConfig:
    """Create production configuration"""
    credentials = BybitCredentials(
        api_key=api_key,
        api_secret=api_secret,
        environment=Environment.MAINNET
    )
    
    config = UnifiedAPIConfig(
        environment=Environment.MAINNET,
        credentials=credentials,
        enable_trading=True
    )
    
    # Production-specific settings
    config.logging.level = LogLevel.WARNING
    config.cache.cache_size = 50000
    config.performance.enable_batching = True
    config.security.enable_request_signing = True
    
    return config

def create_testnet_config(api_key: str = "", api_secret: str = "") -> UnifiedAPIConfig:
    """Create testnet configuration"""
    credentials = None
    if api_key and api_secret:
        credentials = BybitCredentials(
            api_key=api_key,
            api_secret=api_secret,
            environment=Environment.TESTNET
        )
    
    config = UnifiedAPIConfig(
        environment=Environment.TESTNET,
        credentials=credentials,
        enable_trading=bool(credentials)
    )
    
    # Testnet-specific settings
    config.logging.level = LogLevel.DEBUG
    config.rate_limits.market_data = 60  # Lower rate limits for testing
    config.cache.cache_ttl_seconds = 30  # Shorter cache TTL
    
    return config

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'UnifiedAPIConfig',
    'ConfigurationManager',
    'RateLimitConfig',
    'ConnectionConfig',
    'WebSocketConfig',
    'CacheConfig',
    'AustralianComplianceConfig',
    'LoggingConfig',
    'SecurityConfig',
    'PerformanceConfig',
    'LogLevel',
    'AustralianTaxYear',
    'create_default_config',
    'create_production_config',
    'create_testnet_config'
]