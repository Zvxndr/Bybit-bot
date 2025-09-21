# API Setup Guide

Complete guide to setting up Bybit API keys for your trading bot.

## Overview

The Bybit trading bot requires API keys to interact with the Bybit exchange. This guide walks you through creating API keys safely and configuring them with the unified configuration system.

## Step 1: Create Bybit Account

1. **Visit Bybit**: Go to [www.bybit.com](https://www.bybit.com/)
2. **Sign Up**: Create a new account or sign in to existing account
3. **Complete Verification**: Complete KYC verification (required for API access)
4. **Enable 2FA**: Set up two-factor authentication for security

## Step 2: Create API Keys

### For Testing (Testnet)

1. **Go to Testnet**: Visit [testnet.bybit.com](https://testnet.bybit.com/)
2. **Sign In**: Use your main account credentials (testnet uses same login)
3. **API Management**: Navigate to Account → API Management
4. **Create API Key**:
   - Click "Create New Key"
   - Choose "System-generated API Key"
   - Set name: "Trading Bot - Testnet"
   - **Permissions**: Enable "Trade" only (do NOT enable "Withdraw")
   - **IP Restriction**: Add your server IP for security (optional for testing)
   - Complete 2FA verification

### For Live Trading (Mainnet)

⚠️ **Only after successful testnet trading!**

1. **Go to Mainnet**: Visit [www.bybit.com](https://www.bybit.com/)
2. **API Management**: Navigate to Account → API Management  
3. **Create API Key**:
   - Click "Create New Key"
   - Choose "System-generated API Key"
   - Set name: "Trading Bot - Production"
   - **Permissions**: Enable "Trade" only (NEVER enable "Withdraw")
   - **IP Restriction**: Strongly recommended for production
   - Complete 2FA verification

## Step 3: Secure Your API Keys

### Important Security Rules

✅ **DO:**
- Store API keys securely
- Use different keys for testnet and mainnet
- Enable IP restrictions for production
- Rotate keys regularly (monthly)
- Keep API secret private

❌ **DON'T:**
- Enable withdrawal permissions
- Share API keys with anyone
- Commit keys to version control
- Use production keys for testing
- Store keys in plain text files

### API Key Permissions

**Required Permissions:**
- ✅ **Trade**: Required for placing orders
- ❌ **Withdraw**: NEVER enable this
- ❌ **Transfer**: Not needed
- ❌ **Options**: Only if trading options

## Step 4: Configure Bot with API Keys

### Method 1: Interactive CLI (Recommended)

```bash
python -m src.bot.core.config.cli interactive-setup
```

The CLI will prompt you for:
- Environment (development/production)
- API keys (testnet and/or mainnet)
- Trading preferences
- Risk management settings

### Method 2: Command Line

```bash
# Set testnet keys (for testing)
python -m src.bot.core.config.cli set-api-keys \
  --testnet-key "YOUR_TESTNET_KEY" \
  --testnet-secret "YOUR_TESTNET_SECRET"

# Set mainnet keys (for live trading)
python -m src.bot.core.config.cli set-api-keys \
  --mainnet-key "YOUR_MAINNET_KEY" \
  --mainnet-secret "YOUR_MAINNET_SECRET" \
  --encrypt
```

### Method 3: Configuration File

Edit `config/unified_config.json`:

```json
{
  "security": {
    "enable_encryption": false,
    "api_keys": {
      "testnet_key": "YOUR_TESTNET_KEY",
      "testnet_secret": "YOUR_TESTNET_SECRET",
      "mainnet_key": "YOUR_MAINNET_KEY",
      "mainnet_secret": "YOUR_MAINNET_SECRET"
    }
  }
}
```

⚠️ **For production, always enable encryption:**

```bash
python -m src.bot.core.config.cli encrypt-config
```

## Step 5: Test API Connection

### Verify Testnet Connection
```bash
# Test with paper trading mode
python -m src.bot.main --unified-config --config-env development
```

**Expected output:**
```
✅ Unified configuration loaded
✅ API connection established (testnet)
✅ Account balance retrieved
🤖 Bot ready for paper trading
```

### Verify Mainnet Connection (Production Only)
```bash
# Test connection without trading
python -m src.bot.core.config.cli test-connection --mainnet
```

## API Rate Limits

Bybit has rate limits to prevent abuse. The bot automatically manages these:

### Default Limits
- **Orders**: 10 per second
- **Requests**: 120 per minute
- **WebSocket**: 5 connections

### Configuration
```json
{
  "api": {
    "bybit": {
      "rate_limits": {
        "orders_per_second": 10,
        "requests_per_minute": 120,
        "websocket_connections": 5
      }
    }
  }
}
```

## Environment Configuration

### Development Environment
```json
{
  "environment": "development",
  "api": {
    "bybit": {
      "testnet_enabled": true,
      "mainnet_enabled": false
    }
  }
}
```

### Production Environment
```json
{
  "environment": "production",
  "api": {
    "bybit": {
      "testnet_enabled": true,
      "mainnet_enabled": true
    }
  },
  "security": {
    "enable_encryption": true
  }
}
```

## Troubleshooting

### Common Issues

**"Invalid API credentials"**
- Double-check API key and secret are correct
- Ensure you're using the right environment (testnet vs mainnet)
- Verify API key permissions include "Trade"
- Check if API key is expired or revoked

**"IP not allowed"**
- Remove IP restrictions for testing
- Add your current IP to allowed list
- Use dynamic DNS for changing IPs

**"Rate limit exceeded"**
- Bot automatically handles rate limits
- Reduce trading frequency if needed
- Check for multiple bot instances

**"Insufficient permissions"**
- Enable "Trade" permission on API key
- Disable "Withdraw" permission (security)
- Recreate API key if permissions can't be changed

### Debug API Issues

```bash
# Enable detailed API logging
export BYBIT_BOT_LOG_LEVEL="DEBUG"
python -m src.bot.main --unified-config
```

### Test API Connection

```bash
# Test testnet connection
python -m src.bot.core.config.cli test-connection --testnet

# Test mainnet connection  
python -m src.bot.core.config.cli test-connection --mainnet

# Test with verbose output
python -m src.bot.core.config.cli test-connection --testnet --verbose
```

## Security Best Practices

### API Key Security
1. **Unique Keys**: Use different keys for each bot instance
2. **Minimal Permissions**: Only enable "Trade" permission
3. **IP Restrictions**: Use IP allowlists for production
4. **Regular Rotation**: Change keys monthly
5. **Secure Storage**: Use encryption for production keys

### Network Security
```bash
# Use HTTPS only (default)
# Enable IP restrictions on Bybit
# Use VPN for additional security
# Monitor API usage regularly
```

### Configuration Security
```bash
# Encrypt production configuration
python -m src.bot.core.config.cli encrypt-config

# Set restrictive file permissions
chmod 600 config/unified_config.json
chmod -R 700 config/secrets/

# Use environment variables for CI/CD
export BYBIT_TESTNET_KEY="your_key"
export BYBIT_TESTNET_SECRET="your_secret"
```

## Production Checklist

Before going live with real money:

- [ ] ✅ Testnet API keys working correctly
- [ ] ✅ Paper trading showing profitable performance  
- [ ] ✅ Mainnet API keys created with minimal permissions
- [ ] ✅ IP restrictions enabled for mainnet keys
- [ ] ✅ Configuration encryption enabled
- [ ] ✅ File permissions secured (600/700)
- [ ] ✅ Bot tested with small position sizes
- [ ] ✅ Monitoring and alerts configured
- [ ] ✅ Stop-loss and risk management verified

---

**Next Steps:** Once your API keys are configured, see the [Trading Strategies Guide](TRADING_STRATEGIES.md) to configure your trading approach.