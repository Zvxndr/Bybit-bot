# 🔥 Private Bybit Trading Bot - User Guide

## Quick Start

### 1. Initial Setup
```bash
# 1. Configure your API credentials
cp .env.template .env
# Edit .env with your Bybit API keys

# 2. Edit secrets configuration
# Edit config/secrets.yaml with your API credentials

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the system
python activate_ml_engine.py
```

### 2. Access Dashboard
- Open browser: http://localhost:8501
- Fire Cybersigilism theme with real-time ML predictions
- Monitor strategy graduation and live trading

### 3. Safety First
- **Always start with TESTNET** 
- Test strategies thoroughly before live trading
- Monitor risk parameters closely
- Use emergency stop-loss settings

## Configuration Files

### API Configuration (`config/secrets.yaml`)
- **Testnet**: Safe for learning and testing
- **Mainnet**: Real trading - use with caution
- Keep this file secure and never share

### Risk Management (`config/private_risk_config.yaml`)
- Position sizing: 2% max per trade
- Stop loss: 3% default
- Emergency portfolio stop: 5%
- Customize based on your risk tolerance

### Dashboard (`config/private_dashboard_config.yaml`)
- Fire Cybersigilism theme
- Real-time ML predictions
- Strategy graduation monitoring
- Performance analytics

## Features

### 🤖 ML Integration
- Real-time AI predictions
- Ensemble model insights  
- Strategy auto-graduation
- Performance attribution

### 🛡️ Risk Management
- Conservative position sizing
- Automatic stop-losses
- Emergency portfolio protection
- Daily trading limits

### 📊 Analytics
- Real-time performance tracking
- Strategy graduation progress
- ML model confidence levels
- Profit/loss attribution

### 🔥 Fire Dashboard
- Cyberpunk-themed interface
- Real-time updates
- Mobile-friendly design
- Live trading controls

## Safety Guidelines

### For Beginners
1. **Start with testnet only**
2. **Use small position sizes**
3. **Monitor closely for first week**
4. **Understand each strategy before enabling**

### Risk Management
1. **Never risk more than you can afford to lose**
2. **Set emergency stop-loss limits**
3. **Monitor daily trading performance**
4. **Keep API keys secure**

### Advanced Usage
1. **Customize ML model weights**
2. **Adjust strategy graduation thresholds**
3. **Fine-tune risk parameters**
4. **Enable additional trading pairs**

## Support

### Self-Help
- Check logs in `logs/` directory
- Review configuration files
- Test with mock trading first

### Troubleshooting
- Ensure API credentials are correct
- Verify internet connection for data feeds
- Check Bybit API status
- Review error logs for specific issues

---

🔥 **Happy Trading with your Private AI-Powered Bot!** 🔥

Remember: This is for personal use only. Trade responsibly!
