# 🚀 Bybit Trading Bot - Complete Beginner Setup Guide

**Welcome to the Bybit Trading Bot!** This guide will walk you through everything you need to know to set up and run the bot, even if you have no programming or trading experience.

## 📋 Table of Contents

1. [What You Need Before Starting](#what-you-need-before-starting)
2. [System Requirements](#system-requirements)
3. [Step-by-Step Installation](#step-by-step-installation)
4. [Configuration Setup](#configuration-setup)
5. [Getting Your API Keys](#getting-your-api-keys)
6. [First Time Setup](#first-time-setup)
7. [Running the Bot](#running-the-bot)
8. [Monitoring and Management](#monitoring-and-management)
9. [Troubleshooting](#troubleshooting)
10. [Safety and Risk Management](#safety-and-risk-management)

---

## 🎯 What You Need Before Starting

### **Essential Requirements**
- A computer with Windows, Mac, or Linux
- Internet connection (stable, always-on preferred)
- At least 2-4 hours to complete setup
- Basic computer skills (installing software, editing text files)

### **Financial Requirements**
- **Bybit account** (free to create)
- **Minimum $100-1000** for initial trading capital (start small!)
- **Test environment** - We'll start with "paper trading" (no real money)

### **Optional but Recommended**
- Dedicated computer or VPS (cloud server) for 24/7 operation
- External monitor for monitoring dashboards
- Backup internet connection

---

## 💻 System Requirements

### **Minimum Requirements**
- **CPU**: 2 cores, 2.4 GHz
- **RAM**: 4 GB 
- **Storage**: 10 GB free space
- **OS**: Windows 10+, macOS 10.15+, or Ubuntu 18.04+
- **Internet**: Stable broadband connection

### **Recommended for Live Trading**
- **CPU**: 4 cores, 3.0 GHz+
- **RAM**: 8 GB+
- **Storage**: 20 GB+ SSD
- **Internet**: Business-grade connection with backup
- **UPS**: Uninterruptible power supply

---

## 🔧 Step-by-Step Installation

### **Step 1: Install Python**

The bot is written in Python, so we need to install it first.

#### **Windows:**
1. Go to [python.org](https://python.org)
2. Download **Python 3.11** (latest stable version)
3. Run the installer
4. ⚠️ **IMPORTANT**: Check "Add Python to PATH" during installation
5. Click "Install Now"

#### **macOS:**
```bash
# Install Homebrew first (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11
```

#### **Linux (Ubuntu/Debian):**
```bash
# Update system
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-pip python3.11-venv

# Verify installation
python3.11 --version
```

### **Step 2: Install Git**

Git helps us download and manage the bot code.

#### **Windows:**
1. Download from [git-scm.com](https://git-scm.com)
2. Install with default settings

#### **macOS:**
```bash
brew install git
```

#### **Linux:**
```bash
sudo apt install git
```

### **Step 3: Download the Bot**

Open Terminal (Mac/Linux) or Command Prompt (Windows) and run:

```bash
# Download the bot code
git clone https://github.com/Zvxndr/Bybit-bot.git

# Go into the bot folder
cd Bybit-bot

# Check that files are there
ls  # Mac/Linux
dir # Windows
```

### **Step 4: Set Up Python Environment**

This creates an isolated space for the bot to avoid conflicts:

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

### **Step 5: Install Dependencies**

This installs all the libraries the bot needs:

```bash
# Install all required packages (this takes 5-10 minutes)
pip install -r requirements.txt

# If you get errors, try:
pip install --upgrade pip
pip install -r requirements.txt
```

### **Step 6: Install PostgreSQL Database**

The bot stores data in a PostgreSQL database.

#### **Windows:**
1. Download PostgreSQL from [postgresql.org](https://postgresql.org)
2. Install with these settings:
   - Username: `postgres`
   - Password: `your_secure_password` (remember this!)
   - Port: `5432`
3. Remember your password!

#### **macOS:**
```bash
# Install PostgreSQL
brew install postgresql

# Start PostgreSQL service
brew services start postgresql

# Create database user
createuser -s postgres
```

#### **Linux:**
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Switch to postgres user and create database
sudo -u postgres psql
CREATE USER trading_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE trading_bot OWNER trading_user;
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;
\q
```

---

## ⚙️ Configuration Setup

### **Step 1: Copy Configuration Template**

```bash
# Copy the example configuration
cp config/config.example.yml config/config.yml

# Copy environment variables template
cp .env.example .env
```

### **Step 2: Basic Configuration**

Edit `config/config.yml` with a text editor. Here's what to change:

```yaml
# Basic Settings
environment: development  # Start in development mode

# Trading Settings
trading:
  initial_capital: 1000.0  # Start with $1000 (virtual money in dev mode)
  max_daily_loss: 0.05     # Stop if we lose 5% in one day
  max_position_size: 0.1   # Use max 10% of capital per trade
  
# Trading Pairs
trading_pairs:
  - BTCUSDT  # Bitcoin
  - ETHUSDT  # Ethereum

# Risk Management (VERY IMPORTANT!)
risk_management:
  max_portfolio_risk: 0.02   # 2% of portfolio per trade
  max_drawdown: 0.15         # Stop if we lose 15% total
  stop_loss_percentage: 0.05 # 5% stop loss per trade
```

### **Step 3: Database Configuration**

In the same `config/config.yml` file:

```yaml
database:
  url: postgresql://trading_user:your_secure_password@localhost:5432/trading_bot
  # Replace 'your_secure_password' with your actual PostgreSQL password
```

---

## 🔑 Getting Your API Keys

### **Step 1: Create Bybit Account**

1. Go to [bybit.com](https://bybit.com)
2. Click "Sign Up"
3. Complete registration and KYC (identity verification)
4. **Enable 2FA (Two-Factor Authentication)** for security

### **Step 2: Get Testnet API Keys (For Practice)**

⚠️ **Start with testnet first - it's fake money for practice!**

1. Go to [testnet.bybit.com](https://testnet.bybit.com)
2. Login with your Bybit account
3. Go to API Management → Create API Key
4. Settings:
   - **Name**: "Trading Bot Test"
   - **Permissions**: 
     - ✅ Read
     - ✅ Trade
     - ❌ Transfer (leave unchecked for safety)
   - **IP Restriction**: Add your IP address for security

### **Step 3: Add API Keys to Configuration**

Edit your `.env` file:

```bash
# Testnet API Keys (for practice)
BYBIT_TESTNET_API_KEY=your_testnet_api_key_here
BYBIT_TESTNET_API_SECRET=your_testnet_api_secret_here

# Live API Keys (add these later when ready for real trading)
BYBIT_LIVE_API_KEY=your_live_api_key_here
BYBIT_LIVE_API_SECRET=your_live_api_secret_here

# Database
DATABASE_URL=postgresql://trading_user:your_secure_password@localhost:5432/trading_bot

# Other settings
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

---

## 🛠️ First Time Setup

### **Step 1: Initialize the Database**

```bash
# Create database tables
python -c "from src.bot.database import DatabaseManager; from src.bot.config_manager import ConfigurationManager; dm = DatabaseManager(); dm.initialize()"
```

### **Step 2: Test Configuration**

```bash
# Test that everything is configured correctly
python -m src.bot.config_manager --validate
```

### **Step 3: Run Initial Tests**

```bash
# Test database connection
python -c "from src.bot.database import DatabaseManager; dm = DatabaseManager(); print('Database connection: OK' if dm.test_connection() else 'Database connection: FAILED')"

# Test API connection (testnet)
python -c "from src.bot.config_manager import ConfigurationManager; cm = ConfigurationManager(); cm.load_config(); print('API connection test - check logs for results')"
```

---

## 🚀 Running the Bot

### **Step 1: Start in Paper Trading Mode (Recommended First)**

Paper trading uses fake money but real market data - perfect for learning!

```bash
# Start the bot in paper trading mode
python -m src.bot.main --paper-trade --debug

# You should see output like:
# INFO - Starting trading bot in PAPER TRADING mode
# INFO - Connected to Bybit testnet
# INFO - Bot initialized successfully
```

### **Step 2: Monitor the Dashboard**

Open a new terminal and start the web dashboard:

```bash
# Start dashboard (keep the bot running in the other terminal)
python -m src.bot.main --dashboard-only

# Open your browser to: http://localhost:8501
```

### **Step 3: Let It Run and Learn**

- **Leave it running for 24-48 hours**
- **Monitor the dashboard regularly**
- **Check the logs in the `logs/` folder**
- **Watch for any errors or unusual behavior**

### **Step 4: Understanding the Output**

You'll see messages like:
```
INFO - Market data updated: BTCUSDT price=$43,250.30
INFO - Signal generated: BUY BTCUSDT, strength=0.75
INFO - Paper trade executed: BUY 0.023 BTC at $43,250.30
INFO - Portfolio value: $1,045.67 (+$45.67, +4.57%)
```

---

## 📊 Monitoring and Management

### **Web Dashboard**

The dashboard shows:
- **Portfolio value and performance**
- **Active trades and positions**
- **Risk metrics and safety status**
- **Bot health and system status**

### **Key Metrics to Watch**

1. **Total Return**: How much profit/loss you've made
2. **Daily Return**: Today's profit/loss
3. **Max Drawdown**: Biggest loss from peak value
4. **Sharpe Ratio**: Risk-adjusted return (higher is better)
5. **Win Rate**: Percentage of profitable trades

### **Log Files**

Check these files regularly:
- `logs/trading_bot.log` - Main bot activity
- `logs/trades.log` - All trades executed
- `logs/errors.log` - Any errors or warnings

### **API Endpoints for Advanced Users**

```bash
# Get bot status
curl http://localhost:8080/api/v1/status

# Get current positions
curl http://localhost:8080/api/v1/positions

# Get performance metrics
curl http://localhost:8080/api/v1/performance
```

---

## 🔧 Troubleshooting

### **Common Issues and Solutions**

#### **1. "No module named..." Error**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### **2. Database Connection Error**
```bash
# Check PostgreSQL is running
# Windows: Check Services
# Mac: brew services list
# Linux: sudo systemctl status postgresql

# Test database connection
psql -U trading_user -d trading_bot -h localhost
```

#### **3. API Authentication Error**
- Double-check API keys in `.env` file
- Verify API key permissions on Bybit
- Check IP restrictions
- Ensure testnet keys for testnet environment

#### **4. Bot Stops Trading**
- Check log files for errors
- Verify internet connection
- Check if daily loss limit was hit
- Restart bot if needed

#### **5. High Memory Usage**
```bash
# Check memory usage
top  # Linux/Mac
# Task Manager on Windows

# Reduce memory by limiting data retention
# Edit config.yml:
monitoring:
  max_memory_usage_mb: 1024  # Reduce from default
```

### **Getting Help**

1. **Check log files first** - most issues are logged
2. **Search GitHub issues** for similar problems
3. **Join the community Discord** (if available)
4. **Create a GitHub issue** with:
   - Your operating system
   - Python version
   - Error messages from logs
   - Steps to reproduce the problem

---

## ⚠️ Safety and Risk Management

### **Before Trading Real Money**

✅ **Checklist:**
- [ ] Bot runs successfully in paper trading for 1+ weeks
- [ ] You understand all configuration options
- [ ] You've read and understood all risk warnings
- [ ] You have emergency stop procedures
- [ ] You never risk more than you can afford to lose
- [ ] You have proper monitoring and alerting set up

### **Risk Management Rules**

1. **Start Small**: Begin with minimum capital ($100-500)
2. **Set Limits**: Configure stop losses and daily loss limits
3. **Monitor Regularly**: Check the bot daily, especially first weeks
4. **Have Backup Plans**: Know how to stop the bot immediately
5. **Keep Records**: The bot logs everything, but keep your own notes

### **Emergency Procedures**

#### **Stop Trading Immediately**
```bash
# Stop all trading (keeps monitoring running)
curl -X POST http://localhost:8080/api/v1/trading/stop

# Or kill the bot entirely
# Find the process ID
ps aux | grep "src.bot.main"  # Linux/Mac
# Kill it
kill [process_id]
```

#### **Close All Positions**
1. Log into Bybit web interface
2. Go to Positions
3. Close all positions manually
4. Check that everything is closed

### **Legal and Tax Considerations**

- **Keep Records**: The bot automatically logs all trades
- **Understand Tax Laws**: Crypto trading may be taxable in your jurisdiction
- **Comply with Regulations**: Ensure algorithmic trading is legal where you live
- **Consider Professional Advice**: Consult financial and legal professionals

---

## 🎓 Next Steps

### **After Successful Paper Trading**

1. **Analyze Performance**: Review 1-2 weeks of paper trading results
2. **Understand the Strategies**: Learn how the bot makes trading decisions
3. **Optimize Configuration**: Adjust risk parameters based on results
4. **Plan Live Trading**: Start with small amounts in live trading

### **Advanced Features to Explore**

- **Strategy Graduation System**: Automatic promotion from paper to live trading
- **Custom Strategies**: Develop your own trading strategies
- **Machine Learning Models**: Use AI for market prediction
- **Multi-Exchange Support**: Trade on multiple exchanges
- **Portfolio Optimization**: Advanced risk management

### **Learning Resources**

- **Read the Documentation**: All files in the `docs/` folder
- **Study the Code**: Learn how strategies work
- **Join Communities**: Connect with other algorithmic traders
- **Take Courses**: Learn about quantitative finance and trading

---

## 📞 Support and Community

### **Documentation**
- `docs/SYSTEM_OVERVIEW.md` - Complete system architecture
- `docs/strategy_graduation_system.md` - Advanced strategy management
- `docs/API_REFERENCE.md` - API documentation

### **Getting Help**
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Community**: Join other users (Discord/Telegram if available)

### **Contributing**
- **Report Issues**: Help improve the bot by reporting problems
- **Suggest Features**: Share ideas for new capabilities
- **Submit Code**: Contribute improvements (advanced users)

---

## ⭐ Important Reminders

### **🔴 Critical Safety Notes**

1. **NEVER share your API keys** with anyone
2. **ALWAYS start with paper trading** 
3. **NEVER trade more than you can afford to lose**
4. **ALWAYS set stop losses and daily limits**
5. **MONITOR the bot regularly**, especially when starting

### **🟡 Best Practices**

1. **Keep the bot updated** with latest releases
2. **Backup your configuration** files regularly  
3. **Monitor system resources** (CPU, memory, disk)
4. **Have emergency procedures** ready
5. **Learn continuously** - markets change constantly

### **🟢 Success Tips**

1. **Be patient** - algorithmic trading is a marathon, not a sprint
2. **Learn from losses** - they're part of the learning process
3. **Stay informed** - follow crypto news and market trends
4. **Network with others** - join trading communities
5. **Keep improving** - optimize strategies based on results

---

**Congratulations!** 🎉 You now have everything you need to set up and run the Bybit Trading Bot. Remember to start with paper trading, learn the system thoroughly, and never risk more than you can afford to lose.

**Happy Trading!** 📈

---

*Last Updated: September 2025*
*Version: 1.0.0*

**Disclaimer**: Trading cryptocurrencies involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results. Always do your own research and consider your financial situation carefully.