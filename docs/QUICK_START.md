# Quick Start Guide

Get your Bybit trading bot running in 5 minutes with the unified configuration system.

## Prerequisites

- Python 3.8 or higher
- Bybit account (free to create)
- Basic command line knowledge

## Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/Zvxndr/Bybit-bot.git
cd Bybit-bot

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Get Bybit API Keys

1. **Create Bybit Account**: Go to [Bybit](https://www.bybit.com/) and create an account
2. **Enable API Access**: Go to Account → API Management
3. **Create API Key**: 
   - For testing: Create **Testnet** API key
   - For live trading: Create **Mainnet** API key (after testing)
4. **Save Your Keys**: Copy your API key and secret (keep them secure!)

## Step 3: Configure the Bot

### Option A: Interactive Setup (Recommended for beginners)
```bash
python -m src.bot.core.config.cli interactive-setup
```

### Option B: Command Line Setup
```bash
# Create configuration for development environment
python -m src.bot.core.config.cli create-config --env development

# Set your API keys
python -m src.bot.core.config.cli set-api-keys \
  --testnet-key "YOUR_TESTNET_KEY" \
  --testnet-secret "YOUR_TESTNET_SECRET"
```

### Option C: Manual Configuration
Create `config/unified_config.json`:
```json
{
  "environment": "development",
  "trading": {
    "mode": "paper_only",
    "max_position_size": 0.02,
    "stop_loss_percentage": 0.02,
    "take_profit_percentage": 0.04
  },
  "security": {
    "api_keys": {
      "testnet_key": "YOUR_TESTNET_KEY",
      "testnet_secret": "YOUR_TESTNET_SECRET"
    }
  }
}
```

## Step 4: Run the Bot

```bash
# Start the bot with unified configuration
python -m src.bot.main --unified-config --config-env development
```

**You should see:**
```
🚀 Bybit Trading Bot Starting...
✅ Unified configuration loaded
✅ Risk management system initialized
✅ ML integration layer ready
✅ API connection established
🤖 Bot is now running in paper trading mode
```

## Step 5: Monitor Your Bot

The bot will start trading in **paper mode** (no real money) by default. You can monitor:

- **Console Output**: Real-time trading decisions and performance
- **Log Files**: Detailed logs in `logs/` directory
- **Bybit Dashboard**: Check your testnet account for trades

## Next Steps

### Validate Performance
1. **Let it run for a few days** in paper mode
2. **Check performance metrics** in the logs
3. **Review trading decisions** to understand the strategy

### Move to Live Trading (Optional)
⚠️ **Only after successful paper trading validation!**

1. **Create mainnet API keys** on Bybit
2. **Update configuration** to include mainnet keys
3. **Change trading mode** to `paper_and_live`
4. **Start with small position sizes**

```bash
# Update to live trading mode
python -m src.bot.core.config.cli update-config \
  --trading.mode "paper_and_live" \
  --trading.max_position_size 0.005
```

## Troubleshooting

### Common Issues

**"No module named 'src'"**
```bash
# Make sure you're in the project directory
cd /path/to/Bybit-bot
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**"Invalid API credentials"**
- Double-check your API keys are correct
- Ensure you're using testnet keys for paper trading
- Verify API key permissions include trading

**"Configuration file not found"**
```bash
# Create the config directory
mkdir -p config
# Run the interactive setup again
python -m src.bot.core.config.cli interactive-setup
```

### Getting Help

- **Check logs**: Look in `logs/` directory for detailed error messages
- **Read documentation**: See [Configuration Guide](UNIFIED_CONFIGURATION.md)
- **Report issues**: Create an issue on GitHub

## Configuration Overview

The unified configuration system manages all bot settings:

```
config/
├── unified_config.json          # Main configuration
├── secrets/                     # Encrypted API keys
└── environments/                # Environment-specific settings
    ├── development.json
    ├── testing.json
    └── production.json
```

**Key Settings:**
- **Environment**: `development`, `testing`, `production`
- **Trading Mode**: `paper_only`, `live_only`, `paper_and_live`
- **Risk Management**: Position sizing, stop losses, drawdown limits
- **API Configuration**: Rate limits, websocket settings

## Safety Features

The bot includes multiple safety mechanisms:

✅ **Paper Trading First**: Always test strategies with virtual money  
✅ **Position Size Limits**: Maximum risk per trade  
✅ **Drawdown Protection**: Stop trading if losses exceed threshold  
✅ **Real-time Monitoring**: Continuous performance tracking  
✅ **Automatic Validation**: Performance gates before live trading  

---

**🎉 Congratulations!** Your bot is now running. For advanced configuration and customization, see the [Configuration Guide](UNIFIED_CONFIGURATION.md).