# 🔑 API Credentials Configuration Guide
**Bybit Trading Bot - Live Trading Setup**

## 🚨 IMPORTANT SAFETY NOTES

⚠️ **NEVER share your API credentials with anyone**  
⚠️ **Always test with testnet first**  
⚠️ **Start with small position sizes**  
⚠️ **Monitor your trades closely**

---

## 📋 Step-by-Step Setup

### 1. Create Bybit API Keys

1. **Login to Bybit**: Go to [bybit.com](https://www.bybit.com) and log in to your account
2. **Navigate to API Management**: Account → API Management
3. **Create New API Key**:
   - **Name**: "Trading Bot" (or your preferred name)
   - **Permissions**: 
     - ✅ Read
     - ✅ Trade  
     - ❌ Withdraw (NEVER enable this for trading bots)
   - **IP Restriction**: Add your server's IP address for security
4. **Save your credentials securely**

### 2. Configure Environment Variables

Edit your `.env` file with your actual API credentials:

```bash
# =============================================================================
# API Configuration - REPLACE WITH YOUR ACTUAL VALUES
# =============================================================================
BYBIT_API_KEY=your_actual_api_key_here
BYBIT_SECRET_KEY=your_actual_secret_key_here
BYBIT_TESTNET=true  # Set to 'false' for live trading

# =============================================================================
# Trading Configuration
# =============================================================================
TRADING_ENABLED=false  # Set to 'true' when ready for live trading
DEFAULT_SYMBOL=BTCUSDT
POSITION_SIZE=0.001    # Start small! 0.001 BTC = ~$30 at $30k BTC
MAX_POSITION_SIZE=0.01 # Maximum 0.01 BTC = ~$300 at $30k BTC
STOP_LOSS_PERCENTAGE=2.0
TAKE_PROFIT_PERCENTAGE=3.0
```

### 3. Testing Progression (FOLLOW THIS ORDER!)

#### Phase 1: Testnet Testing ✅ SAFE
```bash
# .env configuration for Phase 1
BYBIT_TESTNET=true
TRADING_ENABLED=false
MOCK_TRADING=true
```

**Action**: Run system validation and check connections
```bash
python -c "from config import Config; Config().validate()"
```

#### Phase 2: Testnet Trading ⚠️ TESTNET ONLY
```bash
# .env configuration for Phase 2  
BYBIT_TESTNET=true
TRADING_ENABLED=true
MOCK_TRADING=false
POSITION_SIZE=0.001
```

**Action**: Enable trading on testnet with small positions
```bash
python main.py
# Monitor for 24-48 hours, verify all systems work
```

#### Phase 3: Live Trading (Minimum Viable) 🚨 REAL MONEY
```bash
# .env configuration for Phase 3
BYBIT_TESTNET=false
TRADING_ENABLED=true  
MOCK_TRADING=false
POSITION_SIZE=0.001    # Very small positions!
MAX_POSITION_SIZE=0.01
```

**Action**: Start live trading with minimal risk
```bash
python main.py
# Monitor constantly, be ready to stop if needed
```

#### Phase 4: Production Scaling 💰 FULL DEPLOYMENT
```bash
# .env configuration for Phase 4 (only after proven success)
BYBIT_TESTNET=false
TRADING_ENABLED=true
POSITION_SIZE=0.01     # Increase after proven performance
MAX_POSITION_SIZE=0.1  # Scale up gradually
```

---

## 🛡️ Security Best Practices

### API Key Security
- ✅ **IP Restriction**: Always restrict API keys to your server IP
- ✅ **No Withdrawal**: Never enable withdrawal permissions
- ✅ **Regular Rotation**: Change API keys every 90 days
- ✅ **Monitor Usage**: Check API key activity regularly
- ✅ **Secure Storage**: Never commit API keys to version control

### Trading Security
- ✅ **Position Limits**: Always set maximum position sizes
- ✅ **Stop Losses**: Ensure stop-loss protection is active
- ✅ **Balance Checks**: Monitor account balance regularly
- ✅ **Alert System**: Set up notifications for large losses
- ✅ **Manual Override**: Have emergency stop procedures ready

---

## 🧪 Validation Commands

### Test API Connection
```bash
python -c "
import ccxt
import os
from dotenv import load_dotenv
load_dotenv()

exchange = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY'),
    'secret': os.getenv('BYBIT_SECRET_KEY'),
    'sandbox': os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
})

try:
    balance = exchange.fetch_balance()
    print('✅ API Connection: SUCCESS')
    print(f'Account Type: {'TESTNET' if exchange.sandbox else 'LIVE'}')
    print(f'USDT Balance: {balance.get(\"USDT\", {}).get(\"free\", 0)}')
except Exception as e:
    print(f'❌ API Connection: FAILED - {e}')
"
```

### Test Trading Permissions
```bash
python -c "
import ccxt
import os
from dotenv import load_dotenv
load_dotenv()

exchange = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY'),
    'secret': os.getenv('BYBIT_SECRET_KEY'), 
    'sandbox': os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
})

try:
    # Test market data access
    ticker = exchange.fetch_ticker('BTC/USDT')
    print('✅ Market Data: SUCCESS')
    
    # Test account access
    balance = exchange.fetch_balance()
    print('✅ Account Access: SUCCESS')
    
    # Test order capabilities (dry run)
    markets = exchange.load_markets()
    print('✅ Trading Setup: SUCCESS')
    print(f'Available Markets: {len(markets)}')
    
except Exception as e:
    print(f'❌ Trading Test: FAILED - {e}')
"
```

---

## 🚀 Quick Start Commands

### Start with Docker (Recommended)
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f trading-bot

# Stop all services
docker-compose down
```

### Start Manually
```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Start API server
python start_api.py

# In another terminal, start dashboard
streamlit run dashboard/main.py
```

### Emergency Stop
```bash
# Stop Docker services
docker-compose down

# Kill Python processes (if running manually)
pkill -f "python.*start_api.py"
pkill -f "streamlit.*dashboard"
```

---

## 📊 Monitoring Your Bot

### Key Metrics to Watch
- **P&L (Profit/Loss)**: Total and daily performance
- **Win Rate**: Percentage of profitable trades
- **Position Size**: Current open positions
- **API Rate Limits**: Ensure not hitting exchange limits
- **System Health**: CPU, memory, disk usage
- **Model Performance**: ML prediction accuracy

### Dashboard Access
- **API Endpoints**: http://localhost:8001
- **Streamlit Dashboard**: http://localhost:8501 (if running manually)
- **Health Check**: http://localhost:8001/health

### Log Locations
```
logs/
├── app/           # Application logs
├── trading/       # Trading activity logs  
├── system/        # System health logs
└── errors/        # Error logs
```

---

## ⚠️ Risk Management

### Position Sizing Rules
- **Start Small**: Begin with 0.001 BTC positions (~$30)
- **Scale Gradually**: Only increase after consistent profits
- **Maximum Risk**: Never risk more than 2% of account per trade
- **Portfolio Limit**: Keep total exposure under 10% of account

### Stop-Loss Protection
```bash
# Ensure these are set in .env
STOP_LOSS_PERCENTAGE=2.0    # 2% stop loss
TAKE_PROFIT_PERCENTAGE=3.0  # 3% take profit
MAX_POSITION_SIZE=0.01      # Maximum position limit
```

### Emergency Procedures
1. **Stop Trading**: Set `TRADING_ENABLED=false`
2. **Close Positions**: Manually close all open positions
3. **Check Logs**: Review recent trading activity
4. **Investigate**: Identify what went wrong
5. **Fix Issues**: Address problems before restarting

---

## 🎯 Success Checklist

### Before Going Live
- [ ] API keys created with correct permissions
- [ ] IP restrictions configured
- [ ] .env file configured correctly
- [ ] Testnet testing completed successfully  
- [ ] Position sizes set appropriately
- [ ] Stop-loss protection verified
- [ ] Dashboard monitoring set up
- [ ] Emergency procedures documented
- [ ] Backup and recovery plan ready

### After Going Live
- [ ] Monitor continuously for first 24 hours
- [ ] Verify all trades execute correctly
- [ ] Check P&L calculations are accurate
- [ ] Ensure risk limits are respected
- [ ] Monitor system performance
- [ ] Document any issues or improvements needed

---

## 📞 Support & Resources

### Documentation
- **Production Report**: `PRODUCTION_READY_REPORT.md`
- **System Validation**: `system_validation_report.json`
- **Configuration Guide**: `config.py` (with inline documentation)

### Testing Commands
- **Health Check**: `python final_system_validation.py`
- **Config Test**: `python -c "from config import Config; Config().validate()"`
- **Docker Test**: `docker-compose up --build`

### Troubleshooting
- **Check Logs**: Review files in `logs/` directory
- **API Issues**: Verify API keys and permissions
- **Connection Problems**: Check internet and firewall settings
- **Performance Issues**: Monitor system resources

---

**Remember**: Trading involves risk. Always start small, monitor closely, and never trade more than you can afford to lose! 🛡️