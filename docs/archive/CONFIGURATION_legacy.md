# Configuration Templates and Examples

## Table of Contents
1. [Configuration File Structure](#configuration-file-structure)
2. [Environment-Specific Configurations](#environment-specific-configurations)
3. [Trading Configuration](#trading-configuration)
4. [Risk Management Configuration](#risk-management-configuration)
5. [Exchange Configuration](#exchange-configuration)
6. [Strategy Configuration](#strategy-configuration)
7. [Monitoring Configuration](#monitoring-configuration)
8. [Environment Variables](#environment-variables)

## Configuration File Structure

The bot uses YAML configuration files with a hierarchical structure. All configuration files should be placed in the `config/` directory.

### Default Configuration Template

```yaml
# config/default.yaml
# Base configuration template - DO NOT modify directly
# Copy to environment-specific files and customize

# Application metadata
app:
  name: "Bybit Trading Bot"
  version: "1.0.0"
  environment: "development"
  log_level: "INFO"

# Trading configuration
trading:
  # Core trading settings
  enabled: true
  max_concurrent_trades: 5
  max_position_size: 0.1  # Maximum position size as percentage of portfolio
  min_trade_value: 10.0   # Minimum trade value in USDT
  
  # Order execution
  default_order_type: "LIMIT"
  slippage_tolerance: 0.001  # 0.1%
  order_timeout: 30  # seconds
  
  # Position management
  enable_stop_loss: true
  enable_take_profit: true
  default_stop_loss_pct: 0.02  # 2%
  default_take_profit_pct: 0.04  # 4%
  
  # Trading hours (UTC)
  trading_hours:
    enabled: false
    start_time: "00:00"
    end_time: "23:59"
    timezone: "UTC"

# Risk management configuration
risk:
  # Portfolio risk limits
  max_portfolio_risk: 0.02  # 2% of portfolio per trade
  max_daily_loss: 0.05      # 5% maximum daily loss
  max_drawdown: 0.15        # 15% maximum drawdown
  
  # Position limits
  max_leverage: 1.0
  max_position_correlation: 0.7
  max_sector_exposure: 0.3
  
  # Circuit breakers
  circuit_breakers:
    enabled: true
    daily_loss_threshold: 0.03  # 3%
    consecutive_losses: 5
    rapid_loss_threshold: 0.02  # 2% loss in 1 hour
  
  # Risk assessment
  confidence_threshold: 0.6
  min_risk_reward_ratio: 1.5
  
  # VaR configuration
  var_confidence: 0.95
  var_lookback_days: 30

# Exchange configuration
exchange:
  name: "bybit"
  
  # API settings
  api:
    testnet: true  # Set to false for production
    base_url: "https://api-testnet.bybit.com"
    timeout: 30
    max_retries: 3
    retry_delay: 1
  
  # Rate limiting
  rate_limits:
    orders_per_second: 10
    requests_per_minute: 120
    
  # Supported symbols
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
    - "ADAUSDT"
    - "SOLUSDT"

# Strategy configuration
strategies:
  # Moving Average Crossover Strategy
  ma_crossover:
    enabled: true
    class: "MovingAverageCrossoverStrategy"
    parameters:
      fast_period: 10
      slow_period: 50
      min_confidence: 0.7
    symbols:
      - "BTCUSDT"
      - "ETHUSDT"
    
  # RSI Strategy
  rsi_strategy:
    enabled: false
    class: "RSIStrategy"
    parameters:
      period: 14
      overbought: 70
      oversold: 30
      min_confidence: 0.6
    symbols:
      - "BTCUSDT"

# Data management
data:
  # Data sources
  sources:
    primary: "bybit"
    backup: null
  
  # Data storage
  storage:
    type: "file"  # Options: file, database, cloud
    path: "data/"
    retention_days: 90
  
  # Market data
  market_data:
    intervals: ["1m", "5m", "15m", "1h", "4h", "1d"]
    max_candles: 1000
    update_frequency: 60  # seconds

# Monitoring and alerting
monitoring:
  # Metrics collection
  metrics:
    enabled: true
    port: 8080
    endpoint: "/metrics"
    
  # Alerting
  alerts:
    enabled: true
    channels:
      email:
        enabled: false
        smtp_server: "smtp.gmail.com"
        smtp_port: 587
        from_email: ""
        to_emails: []
      
      slack:
        enabled: false
        webhook_url: ""
        channel: "#trading-alerts"
      
      telegram:
        enabled: false
        bot_token: ""
        chat_id: ""
  
  # Health checks
  health:
    enabled: true
    check_interval: 60  # seconds
    endpoint: "/health"

# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # File logging
  file:
    enabled: true
    path: "logs/"
    filename: "trading_bot.log"
    max_size: "100MB"
    backup_count: 5
    rotation: "daily"
  
  # Console logging
  console:
    enabled: true
    level: "INFO"
  
  # Component logging levels
  components:
    trading_engine: "INFO"
    risk_manager: "INFO"
    exchange_client: "DEBUG"
    strategy_manager: "INFO"
    data_manager: "INFO"

# Performance settings
performance:
  # Threading
  max_workers: 4
  
  # Caching
  cache:
    enabled: true
    ttl: 300  # seconds
    max_size: 1000
  
  # Database connection pool
  database:
    pool_size: 10
    max_overflow: 20
    pool_timeout: 30

# Security settings
security:
  # API key encryption
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
  
  # Rate limiting
  internal_rate_limits:
    enabled: true
    requests_per_second: 100
  
  # IP whitelisting
  ip_whitelist:
    enabled: false
    allowed_ips: []
```

## Environment-Specific Configurations

### Development Configuration

```yaml
# config/development.yaml
# Development environment configuration

# Inherit from default configuration
extends: "default.yaml"

# Override development-specific settings
app:
  environment: "development"
  log_level: "DEBUG"

# Use testnet for development
exchange:
  api:
    testnet: true
    base_url: "https://api-testnet.bybit.com"

# Relaxed risk limits for testing
risk:
  max_portfolio_risk: 0.05  # 5% for testing
  max_daily_loss: 0.10      # 10% for testing

# Enable all logging
logging:
  level: "DEBUG"
  components:
    trading_engine: "DEBUG"
    risk_manager: "DEBUG"
    exchange_client: "DEBUG"
    strategy_manager: "DEBUG"
    data_manager: "DEBUG"

# Shorter intervals for testing
data:
  market_data:
    update_frequency: 30  # 30 seconds

# Enable metrics for monitoring
monitoring:
  metrics:
    enabled: true
  alerts:
    enabled: false  # Disable alerts in development
```

### Testing Configuration

```yaml
# config/testing.yaml
# Testing environment configuration

extends: "default.yaml"

app:
  environment: "testing"
  log_level: "WARNING"

# Mock configuration for testing
exchange:
  name: "mock"
  api:
    testnet: true
    mock_mode: true

# Minimal risk limits for unit tests
risk:
  max_portfolio_risk: 1.0
  max_daily_loss: 1.0
  circuit_breakers:
    enabled: false

# Disable external services
monitoring:
  metrics:
    enabled: false
  alerts:
    enabled: false

# Memory storage for tests
data:
  storage:
    type: "memory"
    retention_days: 1

# Minimal logging
logging:
  level: "ERROR"
  file:
    enabled: false
  console:
    enabled: false
```

### Staging Configuration

```yaml
# config/staging.yaml
# Staging environment configuration

extends: "default.yaml"

app:
  environment: "staging"
  log_level: "INFO"

# Use testnet but with production-like settings
exchange:
  api:
    testnet: true
    base_url: "https://api-testnet.bybit.com"

# Production-like risk settings
risk:
  max_portfolio_risk: 0.01  # 1%
  max_daily_loss: 0.03      # 3%
  circuit_breakers:
    enabled: true

# Enable monitoring
monitoring:
  metrics:
    enabled: true
  alerts:
    enabled: true
    channels:
      email:
        enabled: true
        from_email: "staging-bot@example.com"
        to_emails: ["team@example.com"]

# Production-like data settings
data:
  storage:
    retention_days: 30
  market_data:
    update_frequency: 60
```

### Production Configuration

```yaml
# config/production.yaml
# Production environment configuration

extends: "default.yaml"

app:
  environment: "production"
  log_level: "INFO"

# Production exchange settings
exchange:
  api:
    testnet: false
    base_url: "https://api.bybit.com"
    timeout: 30
    max_retries: 3

# Conservative risk settings
risk:
  max_portfolio_risk: 0.005  # 0.5%
  max_daily_loss: 0.02       # 2%
  max_drawdown: 0.10         # 10%
  circuit_breakers:
    enabled: true
    daily_loss_threshold: 0.015  # 1.5%

# Production monitoring
monitoring:
  metrics:
    enabled: true
  alerts:
    enabled: true
    channels:
      email:
        enabled: true
      slack:
        enabled: true
      telegram:
        enabled: true

# Persistent data storage
data:
  storage:
    type: "database"
    retention_days: 365

# Optimized performance
performance:
  max_workers: 8
  cache:
    enabled: true
    ttl: 300
    max_size: 5000

# Enhanced security
security:
  encryption:
    enabled: true
  internal_rate_limits:
    enabled: true
  ip_whitelist:
    enabled: true
```

## Trading Configuration Examples

### Conservative Trading

```yaml
# Conservative trading configuration
trading:
  max_concurrent_trades: 2
  max_position_size: 0.02  # 2% max position
  min_trade_value: 50.0
  
  enable_stop_loss: true
  enable_take_profit: true
  default_stop_loss_pct: 0.015  # 1.5%
  default_take_profit_pct: 0.03  # 3%

risk:
  max_portfolio_risk: 0.005  # 0.5%
  max_daily_loss: 0.01       # 1%
  confidence_threshold: 0.8
  min_risk_reward_ratio: 2.0
```

### Aggressive Trading

```yaml
# Aggressive trading configuration
trading:
  max_concurrent_trades: 10
  max_position_size: 0.05  # 5% max position
  min_trade_value: 20.0
  
  enable_stop_loss: true
  enable_take_profit: true
  default_stop_loss_pct: 0.03  # 3%
  default_take_profit_pct: 0.06  # 6%

risk:
  max_portfolio_risk: 0.02  # 2%
  max_daily_loss: 0.05      # 5%
  confidence_threshold: 0.6
  min_risk_reward_ratio: 1.2
```

### Scalping Configuration

```yaml
# Scalping strategy configuration
trading:
  max_concurrent_trades: 20
  max_position_size: 0.01  # 1% max position
  min_trade_value: 10.0
  
  default_order_type: "MARKET"
  slippage_tolerance: 0.002  # 0.2%
  order_timeout: 10  # 10 seconds
  
  enable_stop_loss: true
  enable_take_profit: true
  default_stop_loss_pct: 0.005  # 0.5%
  default_take_profit_pct: 0.01   # 1%

data:
  market_data:
    intervals: ["1m", "5m"]
    update_frequency: 10  # 10 seconds
```

## Strategy Configuration Examples

### Multi-Strategy Setup

```yaml
strategies:
  # Primary trend following
  ma_crossover_long:
    enabled: true
    class: "MovingAverageCrossoverStrategy"
    parameters:
      fast_period: 20
      slow_period: 50
      direction: "long_only"
      min_confidence: 0.7
    symbols: ["BTCUSDT", "ETHUSDT"]
    weight: 0.4
  
  # Short-term momentum
  rsi_momentum:
    enabled: true
    class: "RSIStrategy"
    parameters:
      period: 14
      overbought: 75
      oversold: 25
      min_confidence: 0.6
    symbols: ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    weight: 0.3
  
  # Mean reversion
  bollinger_reversion:
    enabled: true
    class: "BollingerBandsStrategy"
    parameters:
      period: 20
      std_dev: 2.0
      min_confidence: 0.65
    symbols: ["SOLUSDT", "AVAXUSDT"]
    weight: 0.3
```

### Symbol-Specific Configuration

```yaml
# Symbol-specific trading parameters
symbol_configs:
  BTCUSDT:
    max_position_size: 0.03
    min_trade_value: 50.0
    stop_loss_pct: 0.02
    take_profit_pct: 0.04
    confidence_threshold: 0.7
    
  ETHUSDT:
    max_position_size: 0.025
    min_trade_value: 30.0
    stop_loss_pct: 0.025
    take_profit_pct: 0.045
    confidence_threshold: 0.65
    
  ADAUSDT:
    max_position_size: 0.02
    min_trade_value: 20.0
    stop_loss_pct: 0.03
    take_profit_pct: 0.05
    confidence_threshold: 0.75
```

## Environment Variables

### Required Environment Variables

```bash
# .env.example
# Copy to .env and fill in your values

# Bybit API Credentials
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here

# Database Configuration (if using database storage)
DATABASE_URL=postgresql://user:password@localhost:5432/trading_bot

# Email Configuration (for alerts)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Slack Configuration (for alerts)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url

# Telegram Configuration (for alerts)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Security
ENCRYPTION_KEY=your_32_character_encryption_key_here

# Monitoring
PROMETHEUS_PORT=8080
GRAFANA_PORT=3000

# Cloud Configuration (if using cloud deployment)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

# Redis Configuration (for caching)
REDIS_URL=redis://localhost:6379/0
```

### Environment Variable Validation

```python
# config/env_validation.py
import os
from typing import Dict, List, Optional

def validate_environment_variables(env: str = "production") -> List[str]:
    """Validate required environment variables for the given environment."""
    
    errors = []
    
    # Base required variables
    required_vars = [
        "BYBIT_API_KEY",
        "BYBIT_API_SECRET",
    ]
    
    # Environment-specific requirements
    if env == "production":
        required_vars.extend([
            "DATABASE_URL",
            "ENCRYPTION_KEY",
            "EMAIL_HOST",
            "EMAIL_USER",
            "EMAIL_PASSWORD",
        ])
    elif env == "staging":
        required_vars.extend([
            "EMAIL_HOST",
            "EMAIL_USER",
            "EMAIL_PASSWORD",
        ])
    
    # Check required variables
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # Validate specific formats
    api_key = os.getenv("BYBIT_API_KEY")
    if api_key and len(api_key) < 10:
        errors.append("BYBIT_API_KEY appears to be too short")
    
    api_secret = os.getenv("BYBIT_API_SECRET")
    if api_secret and len(api_secret) < 10:
        errors.append("BYBIT_API_SECRET appears to be too short")
    
    encryption_key = os.getenv("ENCRYPTION_KEY")
    if env == "production" and encryption_key and len(encryption_key) != 32:
        errors.append("ENCRYPTION_KEY must be exactly 32 characters long")
    
    return errors
```

## Configuration Validation

### YAML Schema Validation

```python
# config/schema.py
from cerberus import Validator

TRADING_BOT_SCHEMA = {
    'app': {
        'type': 'dict',
        'required': True,
        'schema': {
            'name': {'type': 'string', 'required': True},
            'version': {'type': 'string', 'required': True},
            'environment': {'type': 'string', 'allowed': ['development', 'testing', 'staging', 'production']},
            'log_level': {'type': 'string', 'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR']}
        }
    },
    'trading': {
        'type': 'dict',
        'required': True,
        'schema': {
            'enabled': {'type': 'boolean', 'required': True},
            'max_concurrent_trades': {'type': 'integer', 'min': 1, 'max': 50},
            'max_position_size': {'type': 'float', 'min': 0.001, 'max': 1.0},
            'min_trade_value': {'type': 'float', 'min': 1.0},
            'default_order_type': {'type': 'string', 'allowed': ['MARKET', 'LIMIT']},
            'slippage_tolerance': {'type': 'float', 'min': 0.0001, 'max': 0.01},
            'order_timeout': {'type': 'integer', 'min': 5, 'max': 300}
        }
    },
    'risk': {
        'type': 'dict',
        'required': True,
        'schema': {
            'max_portfolio_risk': {'type': 'float', 'min': 0.001, 'max': 0.1},
            'max_daily_loss': {'type': 'float', 'min': 0.001, 'max': 0.5},
            'max_drawdown': {'type': 'float', 'min': 0.01, 'max': 0.5},
            'confidence_threshold': {'type': 'float', 'min': 0.1, 'max': 1.0},
            'min_risk_reward_ratio': {'type': 'float', 'min': 0.5, 'max': 10.0}
        }
    },
    'exchange': {
        'type': 'dict',
        'required': True,
        'schema': {
            'name': {'type': 'string', 'allowed': ['bybit', 'mock']},
            'api': {
                'type': 'dict',
                'schema': {
                    'testnet': {'type': 'boolean'},
                    'base_url': {'type': 'string'},
                    'timeout': {'type': 'integer', 'min': 5, 'max': 300},
                    'max_retries': {'type': 'integer', 'min': 1, 'max': 10}
                }
            },
            'symbols': {'type': 'list', 'schema': {'type': 'string'}}
        }
    }
}

def validate_config(config: dict) -> List[str]:
    """Validate configuration against schema."""
    validator = Validator(TRADING_BOT_SCHEMA)
    
    if validator.validate(config):
        return []
    else:
        return [f"{field}: {error}" for field, error in validator.errors.items()]
```

## Usage Examples

### Loading Configuration

```python
from config.configuration_manager import ConfigurationManager

# Load environment-specific configuration
config_manager = ConfigurationManager("config/production.yaml")

# Validate configuration
errors = config_manager.validate_config()
if errors:
    print(f"Configuration errors: {errors}")
    exit(1)

# Access configuration values
trading_enabled = config_manager.get("trading.enabled")
max_position = config_manager.get("trading.max_position_size")
api_key = config_manager.get_env("BYBIT_API_KEY")
```

### Runtime Configuration Updates

```python
# Update configuration at runtime
config_manager.update_config({
    "trading.max_concurrent_trades": 3,
    "risk.max_portfolio_risk": 0.01
})

# Save configuration changes
config_manager.save_config("config/runtime_updates.yaml")
```

### Environment-Specific Loading

```python
import os

# Determine environment
env = os.getenv("ENVIRONMENT", "development")

# Load appropriate configuration
config_file = f"config/{env}.yaml"
config_manager = ConfigurationManager(config_file)

print(f"Loaded configuration for {env} environment")
```