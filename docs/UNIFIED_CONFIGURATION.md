# Unified Configuration System

Complete guide to the unified configuration system that powers all bot components.

## Overview

The unified configuration system provides a single source of truth for all bot settings. It replaces the previous scattered configuration files with a centralized, validated, and extensible system.

## Key Benefits

- **🎯 Single Configuration**: One file controls all bot components
- **🔐 Secure**: Encrypted API keys and sensitive data
- **✅ Validated**: Comprehensive validation and error checking
- **🔄 Hot Reload**: Update settings without restarting the bot
- **🌍 Multi-Environment**: Support for dev, test, and production environments
- **🛠️ CLI Tools**: Command-line interface for easy management

## Architecture

```
Unified Configuration System
├── Schema (schema.py)           # Configuration structure and validation
├── Manager (manager.py)         # Configuration loading and management
├── CLI (cli.py)                # Command-line interface
└── Integrations (integrations.py) # Component adapters
```

## Configuration Structure

### Core Sections

```json
{
  "environment": "development",
  "trading": { ... },
  "risk_management": { ... },
  "ml_integration": { ... },
  "api": { ... },
  "security": { ... },
  "monitoring": { ... },
  "logging": { ... }
}
```

### Complete Schema

```json
{
  "environment": "development",
  
  "trading": {
    "mode": "paper_only",
    "max_position_size": 0.02,
    "stop_loss_percentage": 0.02,
    "take_profit_percentage": 0.04,
    "max_open_positions": 5,
    "slippage_tolerance": 0.001
  },
  
  "risk_management": {
    "max_drawdown": 0.15,
    "position_sizing_method": "kelly_criterion",
    "volatility_target": 0.20,
    "correlation_threshold": 0.7,
    "concentration_limit": 0.25,
    "risk_free_rate": 0.02,
    "lookback_period": 252,
    "rebalance_frequency": "daily"
  },
  
  "ml_integration": {
    "feature_engineering": {
      "enabled": true,
      "lookback_periods": [5, 10, 20, 50],
      "technical_indicators": ["rsi", "macd", "bollinger_bands"],
      "feature_selection": "auto",
      "normalization": "standard"
    },
    "model_management": {
      "auto_retrain": true,
      "validation_threshold": 0.6,
      "model_persistence": true,
      "ensemble_methods": true,
      "hyperparameter_tuning": true
    },
    "prediction": {
      "confidence_threshold": 0.7,
      "ensemble_methods": true,
      "prediction_horizon": 24,
      "update_frequency": "hourly"
    }
  },
  
  "api": {
    "bybit": {
      "testnet_enabled": true,
      "mainnet_enabled": false,
      "rate_limits": {
        "orders_per_second": 10,
        "requests_per_minute": 120,
        "websocket_connections": 5
      },
      "websocket": {
        "enabled": true,
        "auto_reconnect": true,
        "heartbeat_interval": 30,
        "max_reconnect_attempts": 10
      },
      "retry_policy": {
        "max_retries": 3,
        "backoff_factor": 2,
        "retry_delay": 1
      }
    }
  },
  
  "security": {
    "enable_encryption": false,
    "api_keys": {
      "testnet_key": "your_testnet_key",
      "testnet_secret": "your_testnet_secret",
      "mainnet_key": "your_mainnet_key",
      "mainnet_secret": "your_mainnet_secret"
    },
    "encryption": {
      "algorithm": "AES-256-GCM",
      "key_derivation": "PBKDF2",
      "salt_length": 32,
      "iterations": 100000
    }
  },
  
  "monitoring": {
    "performance_tracking": true,
    "alert_thresholds": {
      "max_drawdown_alert": 0.10,
      "consecutive_losses": 5,
      "daily_loss_limit": 0.05
    },
    "reporting": {
      "daily_summary": true,
      "trade_log": true,
      "performance_metrics": true,
      "email_reports": false
    },
    "metrics": {
      "update_frequency": "real_time",
      "retention_period": 365,
      "export_format": "json"
    }
  },
  
  "logging": {
    "level": "INFO",
    "file_logging": true,
    "console_logging": true,
    "log_rotation": true,
    "max_file_size": "10MB",
    "backup_count": 5,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

## Environment Configurations

### Development Environment
```json
{
  "environment": "development",
  "trading": {
    "mode": "paper_only",
    "max_position_size": 0.02
  },
  "security": {
    "enable_encryption": false
  },
  "logging": {
    "level": "DEBUG"
  }
}
```

### Testing Environment
```json
{
  "environment": "testing",
  "trading": {
    "mode": "paper_only",
    "max_position_size": 0.01
  },
  "security": {
    "enable_encryption": false
  },
  "logging": {
    "level": "INFO"
  }
}
```

### Production Environment
```json
{
  "environment": "production",
  "trading": {
    "mode": "paper_and_live",
    "max_position_size": 0.005
  },
  "security": {
    "enable_encryption": true
  },
  "logging": {
    "level": "WARNING"
  }
}
```

## CLI Commands

### Create Configuration
```bash
# Create default configuration
python -m src.bot.core.config.cli create-config --env development

# Create from template
python -m src.bot.core.config.cli create-config --template conservative

# Create with custom settings
python -m src.bot.core.config.cli create-config \
  --env production \
  --trading.max_position_size 0.01 \
  --risk_management.max_drawdown 0.10
```

### Manage API Keys
```bash
# Set API keys
python -m src.bot.core.config.cli set-api-keys \
  --testnet-key "your_key" \
  --testnet-secret "your_secret"

# Set mainnet keys (production)
python -m src.bot.core.config.cli set-api-keys \
  --mainnet-key "your_key" \
  --mainnet-secret "your_secret" \
  --encrypt

# View current keys (masked)
python -m src.bot.core.config.cli show-api-keys
```

### Update Configuration
```bash
# Update single setting
python -m src.bot.core.config.cli update-config \
  --trading.mode "paper_and_live"

# Update multiple settings
python -m src.bot.core.config.cli update-config \
  --trading.max_position_size 0.01 \
  --risk_management.max_drawdown 0.12 \
  --logging.level "INFO"

# Update from file
python -m src.bot.core.config.cli update-config --from-file updates.json
```

### Validate Configuration
```bash
# Validate current configuration
python -m src.bot.core.config.cli validate

# Validate specific file
python -m src.bot.core.config.cli validate --config-file custom_config.json

# Validate with detailed output
python -m src.bot.core.config.cli validate --verbose
```

### Interactive Setup
```bash
# Full interactive configuration wizard
python -m src.bot.core.config.cli interactive-setup

# Environment-specific setup
python -m src.bot.core.config.cli interactive-setup --env production
```

## Integration with Components

### Risk Management Integration
```python
from src.bot.core.config.manager import UnifiedConfigurationManager
from src.bot.risk.core.unified_risk_manager import UnifiedRiskManager

# Load configuration
config_manager = UnifiedConfigurationManager()
config = config_manager.get_configuration()

# Initialize risk manager with unified config
risk_manager = UnifiedRiskManager(config.risk_management)
```

### ML Integration
```python
from src.bot.integration.ml_integration_controller import MLIntegrationController

# Initialize ML controller with unified config
ml_controller = MLIntegrationController(config.ml_integration)
```

### API Integration
```python
from src.bot.api.unified_client import UnifiedBybitClient

# Initialize API client with unified config
api_client = UnifiedBybitClient(config.api.bybit)
```

## Advanced Features

### Hot Reload
```python
# Configuration automatically reloads when file changes
config_manager.enable_hot_reload()

# Manual reload
config_manager.reload_configuration()
```

### Environment Variables
```bash
# Override config with environment variables
export BYBIT_BOT_TRADING_MODE="live_only"
export BYBIT_BOT_MAX_POSITION_SIZE="0.005"
export BYBIT_BOT_LOG_LEVEL="DEBUG"

# Use environment-specific config file
export BYBIT_BOT_CONFIG_ENV="production"
```

### Configuration Templates
```bash
# Conservative template
python -m src.bot.core.config.cli create-config --template conservative

# Aggressive template  
python -m src.bot.core.config.cli create-config --template aggressive

# Custom template
python -m src.bot.core.config.cli create-config --template-file my_template.json
```

## Security Best Practices

### API Key Management
- **Never commit API keys** to version control
- **Use encryption** for production environments
- **Rotate keys regularly** (monthly for active trading)
- **Use minimum required permissions** (trade-only, no withdrawal)

### Configuration Security
```bash
# Enable encryption for sensitive data
python -m src.bot.core.config.cli encrypt-config

# Set restrictive file permissions
chmod 600 config/unified_config.json
chmod 700 config/secrets/
```

### Environment Isolation
- **Development**: Paper trading only, debug logging
- **Testing**: Automated testing, no real money
- **Production**: Live trading, encrypted keys, minimal logging

## Migration from Legacy Configuration

The unified system automatically detects and migrates from legacy configuration files:

```python
# Automatic migration on first run
config_manager = UnifiedConfigurationManager()
# Will detect legacy configs and offer migration
```

### Manual Migration
```bash
# Migrate from legacy configuration
python -m src.bot.core.config.cli migrate-legacy \
  --from-dir old_config/ \
  --to-file config/unified_config.json
```

## Troubleshooting

### Common Issues

**Configuration file not found**
```bash
# Create default configuration
python -m src.bot.core.config.cli create-config --env development
```

**Validation errors**
```bash
# Check validation details
python -m src.bot.core.config.cli validate --verbose
```

**API key errors**
```bash
# Verify API keys are set correctly
python -m src.bot.core.config.cli show-api-keys
```

**Permission errors**
```bash
# Fix file permissions
chmod 600 config/unified_config.json
chmod -R 700 config/
```

### Debug Mode
```bash
# Enable debug logging for configuration system
export BYBIT_BOT_CONFIG_DEBUG="true"
python -m src.bot.main --unified-config
```

---

For more advanced configuration scenarios, see the [Architecture Guide](ARCHITECTURE.md) and [Production Deployment Guide](PRODUCTION.md).