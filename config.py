"""
Bybit Trading Bot Configuration
Centralized configuration management with environment variable support
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class APIConfig:
    """Bybit API configuration"""
    api_key: str = os.getenv('BYBIT_API_KEY', '')
    secret_key: str = os.getenv('BYBIT_SECRET_KEY', '')
    testnet: bool = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
    base_url: str = 'https://api-testnet.bybit.com' if testnet else 'https://api.bybit.com'

@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    enabled: bool = os.getenv('TRADING_ENABLED', 'false').lower() == 'true'
    default_symbol: str = os.getenv('DEFAULT_SYMBOL', 'BTCUSDT')
    position_size: float = float(os.getenv('POSITION_SIZE', '0.001'))
    max_position_size: float = float(os.getenv('MAX_POSITION_SIZE', '0.01'))
    stop_loss_percentage: float = float(os.getenv('STOP_LOSS_PERCENTAGE', '2.0'))
    take_profit_percentage: float = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '3.0'))

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_daily_loss: float = float(os.getenv('MAX_DAILY_LOSS', '100.0'))
    max_drawdown_percentage: float = float(os.getenv('MAX_DRAWDOWN_PERCENTAGE', '10.0'))
    risk_per_trade: float = float(os.getenv('RISK_PER_TRADE', '1.0'))
    max_concurrent_positions: int = int(os.getenv('MAX_CONCURRENT_POSITIONS', '3'))

@dataclass
class MLConfig:
    """Machine Learning configuration"""
    model_update_interval: int = int(os.getenv('ML_MODEL_UPDATE_INTERVAL', '3600'))
    prediction_confidence_threshold: float = float(os.getenv('PREDICTION_CONFIDENCE_THRESHOLD', '0.7'))
    feature_lookback_period: int = int(os.getenv('FEATURE_LOOKBACK_PERIOD', '100'))
    model_training_frequency: str = os.getenv('MODEL_TRAINING_FREQUENCY', 'daily')

@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = os.getenv('DATABASE_TYPE', 'sqlite')
    url: str = os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')
    timescale_host: str = os.getenv('TIMESCALE_HOST', 'localhost')
    timescale_port: int = int(os.getenv('TIMESCALE_PORT', '5432'))
    timescale_db: str = os.getenv('TIMESCALE_DB', 'trading_data')
    timescale_user: str = os.getenv('TIMESCALE_USER', 'postgres')
    timescale_password: str = os.getenv('TIMESCALE_PASSWORD', '')

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    backend_port: int = int(os.getenv('DASHBOARD_BACKEND_PORT', '8001'))
    frontend_port: int = int(os.getenv('DASHBOARD_FRONTEND_PORT', '3000'))
    websocket_url: str = os.getenv('WEBSOCKET_URL', 'ws://localhost:8001/ws')
    api_base_url: str = os.getenv('API_BASE_URL', 'http://localhost:8001')

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_file: str = os.getenv('LOG_FILE', 'logs/trading_bot.log')
    enable_performance_monitoring: bool = os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true'
    alert_email: str = os.getenv('ALERT_EMAIL', '')
    slack_webhook_url: str = os.getenv('SLACK_WEBHOOK_URL', '')

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = os.getenv('SECRET_KEY', 'default-secret-key-change-in-production')
    jwt_secret: str = os.getenv('JWT_SECRET', 'default-jwt-secret-change-in-production')
    encryption_key: str = os.getenv('ENCRYPTION_KEY', 'default-encryption-key-change-in-production')

@dataclass
class PerformanceConfig:
    """Performance settings"""
    max_workers: int = int(os.getenv('MAX_WORKERS', '4'))
    async_executor_threads: int = int(os.getenv('ASYNC_EXECUTOR_THREADS', '8'))
    cache_ttl: int = int(os.getenv('CACHE_TTL', '300'))
    request_timeout: int = int(os.getenv('REQUEST_TIMEOUT', '30'))

@dataclass
class DevelopmentConfig:
    """Development settings"""
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    testing: bool = os.getenv('TESTING', 'false').lower() == 'true'
    dev_mode: bool = os.getenv('DEV_MODE', 'false').lower() == 'true'
    mock_trading: bool = os.getenv('MOCK_TRADING', 'true').lower() == 'true'

class Config:
    """Main configuration class that aggregates all config sections"""
    
    def __init__(self):
        self.api = APIConfig()
        self.trading = TradingConfig()
        self.risk = RiskConfig()
        self.ml = MLConfig()
        self.database = DatabaseConfig()
        self.dashboard = DashboardConfig()
        self.monitoring = MonitoringConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        self.development = DevelopmentConfig()
        
        # Ensure log directory exists
        Path(self.monitoring.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = {}
        
        # Validate API configuration
        if not self.api.api_key:
            issues['api_key'] = 'Bybit API key is required'
        if not self.api.secret_key:
            issues['secret_key'] = 'Bybit secret key is required'
            
        # Validate trading configuration
        if self.trading.position_size <= 0:
            issues['position_size'] = 'Position size must be positive'
        if self.trading.max_position_size < self.trading.position_size:
            issues['max_position_size'] = 'Max position size must be >= position size'
            
        # Validate risk configuration
        if self.risk.max_daily_loss <= 0:
            issues['max_daily_loss'] = 'Max daily loss must be positive'
        if self.risk.risk_per_trade <= 0 or self.risk.risk_per_trade > 100:
            issues['risk_per_trade'] = 'Risk per trade must be between 0 and 100'
            
        # Validate ML configuration
        if self.ml.prediction_confidence_threshold <= 0 or self.ml.prediction_confidence_threshold > 1:
            issues['confidence_threshold'] = 'Confidence threshold must be between 0 and 1'
            
        return issues
    
    def is_production_ready(self) -> bool:
        """Check if configuration is ready for production"""
        issues = self.validate()
        
        production_checks = [
            self.api.api_key != '',
            self.api.secret_key != '',
            self.security.secret_key != 'default-secret-key-change-in-production',
            self.security.jwt_secret != 'default-jwt-secret-change-in-production',
            not self.development.debug,
            not self.development.testing,
            not self.development.dev_mode
        ]
        
        return len(issues) == 0 and all(production_checks)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging/debugging"""
        return {
            'api_configured': bool(self.api.api_key and self.api.secret_key),
            'testnet_mode': self.api.testnet,
            'trading_enabled': self.trading.enabled,
            'mock_trading': self.development.mock_trading,
            'debug_mode': self.development.debug,
            'production_ready': self.is_production_ready()
        }

# Global configuration instance
config = Config()

# Configuration validation on import
if __name__ == "__main__":
    print("🔧 Bybit Trading Bot Configuration")
    print("=" * 40)
    
    summary = config.get_summary()
    for key, value in summary.items():
        status = "✅" if value else "❌"
        print(f"{status} {key.replace('_', ' ').title()}: {value}")
    
    print("\n🔍 Configuration Validation:")
    issues = config.validate()
    if issues:
        print("❌ Configuration Issues Found:")
        for key, issue in issues.items():
            print(f"  - {key}: {issue}")
    else:
        print("✅ Configuration Valid")
    
    print(f"\n🚀 Production Ready: {config.is_production_ready()}")