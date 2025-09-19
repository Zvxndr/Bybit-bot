# Trading Bot System - Quick Start Guide

Welcome to the Trading Bot System! This guide will help you get up and running quickly.

## Prerequisites

- Python 3.11 or higher
- PostgreSQL database
- Git
- 4GB+ RAM
- Stable internet connection

## Quick Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/trading-bot.git
cd trading-bot

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb trading_bot

# Run migrations
python -m alembic upgrade head
```

### 3. Configuration

```bash
# Copy configuration templates
cp config/config.example.yml config/config.yml
cp .env.example .env

# Edit configuration files
nano config/config.yml  # Update database settings
nano .env               # Add your API keys
```

### 4. Start the System

```bash
# Start the trading bot
python -m src.bot.integrated_trading_bot

# In another terminal, start the API server
python -m src.api.trading_bot_api
```

## Docker Quick Start

### 1. Using Docker Compose

```bash
# Clone repository
git clone https://github.com/your-org/trading-bot.git
cd trading-bot

# Copy environment file
cp .env.example .env
# Edit .env with your settings

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 2. Access the System

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **System Status**: http://localhost:8000/status (requires API key)

## First Steps

### 1. Generate API Keys

```bash
python -c "
from src.api.trading_bot_api import SecurityManager
sm = SecurityManager()
key = sm.generate_api_key('My First Key', ['read_only'])
print(f'Your API key: {key}')
"
```

### 2. Test API Access

```bash
# Test with your API key
export API_KEY="your_generated_api_key"
curl -H "Authorization: Bearer $API_KEY" http://localhost:8000/status
```

### 3. Configure Exchange

Edit `config/config.yml`:

```yaml
exchanges:
  bybit:
    api_key: ${BYBIT_API_KEY}
    api_secret: ${BYBIT_API_SECRET}
    testnet: true  # Start with testnet
```

Add to `.env`:
```bash
BYBIT_API_KEY=your_testnet_api_key
BYBIT_API_SECRET=your_testnet_api_secret
```

### 4. Start Trading (Paper Mode)

```bash
# Start in paper trading mode
curl -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"command": "start", "parameters": {"mode": "paper"}}' \
  http://localhost:8000/system/command
```

## System Overview

```
┌─────────────────────────────────────────┐
│           Trading Bot System            │
├─────────────────────────────────────────┤
│  📊 Phase 1: Core Trading Engine        │
│  🛡️  Phase 2: Risk Management          │
│  📈 Phase 3: Backtesting Framework     │
│  🔍 Phase 4: Monitoring System         │
│  📋 Phase 5: Tax Reporting             │
│  🚀 Phase 6: Advanced Features         │
├─────────────────────────────────────────┤
│  🌐 REST API & WebSocket Interface     │
│  📊 Health Monitoring & Alerts         │
│  🔧 Configuration Management           │
│  🧪 Performance Testing                │
└─────────────────────────────────────────┘
```

## Key Features

- ✅ **Multi-Exchange Support**: Currently supports Bybit
- ✅ **Advanced Risk Management**: Position sizing, stop-loss, take-profit
- ✅ **Real-time Monitoring**: System health and performance tracking
- ✅ **Comprehensive API**: REST and WebSocket interfaces
- ✅ **Backtesting Framework**: Historical strategy validation
- ✅ **Tax Reporting**: Automated trade recording and reporting
- ✅ **Machine Learning**: Strategy optimization and portfolio management

## Configuration

### Basic Configuration (`config/config.yml`)

```yaml
environment: development

database:
  host: localhost
  port: 5432
  name: trading_bot
  user: postgres
  password: ${DB_PASSWORD}

trading:
  max_positions: 5
  base_currency: USDT
  trading_pairs:
    - BTCUSDT
    - ETHUSDT

risk:
  max_portfolio_risk: 0.02
  stop_loss_percent: 0.02
  take_profit_percent: 0.04
```

### Environment Variables (`.env`)

```bash
# Database
DB_PASSWORD=your_password

# Exchange APIs (use testnet keys initially)
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret

# Monitoring (optional)
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
DISCORD_WEBHOOK=your_discord_webhook
```

## Common Commands

### System Control

```bash
# Start trading
curl -X POST -H "Authorization: Bearer $API_KEY" \
  -d '{"command": "start"}' \
  http://localhost:8000/system/command

# Stop trading
curl -X POST -H "Authorization: Bearer $API_KEY" \
  -d '{"command": "stop"}' \
  http://localhost:8000/system/command

# Get system status
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:8000/status
```

### Monitoring

```bash
# Check health
curl http://localhost:8000/health

# Get metrics
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:8000/metrics

# View alerts
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:8000/alerts
```

## Development Workflow

### 1. Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
python -m pytest

# Run specific test module
python -m pytest tests/test_trading.py

# Run with coverage
python -m pytest --cov=src tests/
```

### 2. Code Quality

```bash
# Install development tools
pip install black flake8 mypy

# Format code
black src/

# Check code style
flake8 src/

# Type checking
mypy src/
```

### 3. Performance Testing

```bash
# Run performance tests
python -m src.testing.performance_testing

# Generate performance report
python -c "
import asyncio
from src.testing.performance_testing import ComprehensivePerformanceTester
async def main():
    tester = ComprehensivePerformanceTester()
    await tester.run_comprehensive_performance_tests()
asyncio.run(main())
"
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Error

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U postgres -d trading_bot

# Reset database
python -m alembic downgrade base
python -m alembic upgrade head
```

#### 2. API Key Issues

```bash
# Generate new API key
python -c "
from src.api.trading_bot_api import SecurityManager
sm = SecurityManager()
key = sm.generate_api_key('Debug Key', ['read_only'])
print(key)
"

# Test API key
curl -H "Authorization: Bearer tb_your_key" http://localhost:8000/health
```

#### 3. Exchange Connection Issues

```bash
# Test exchange connectivity
python -c "
import asyncio
from src.trading.exchanges.bybit_client import BybitClient
async def test():
    client = BybitClient()
    status = await client.test_connection()
    print(f'Connection: {status}')
asyncio.run(test())
"
```

### Logs

Check log files for detailed error information:

```bash
# Application logs
tail -f logs/app.log

# Error logs
tail -f logs/error.log

# Trading logs
tail -f logs/trading.log
```

## Next Steps

### 1. Production Deployment

- Review the [Production Guide](docs/PRODUCTION_GUIDE.md)
- Set up monitoring and alerting
- Configure SSL/TLS
- Implement backup strategy

### 2. Strategy Development

- Study existing strategies in `src/strategies/`
- Implement custom trading strategies
- Run backtests to validate performance
- Use paper trading before live deployment

### 3. Advanced Features

- Enable machine learning optimization
- Set up advanced portfolio management
- Configure tax reporting
- Implement custom risk models

## Getting Help

- **Documentation**: Check `docs/` directory
- **API Reference**: `docs/API_REFERENCE.md`
- **Issues**: GitHub Issues section
- **Discord**: Community Discord server

## Security Notes

⚠️ **Important Security Considerations**:

1. **Never commit API keys** to version control
2. **Use testnet** for development and testing
3. **Start with small amounts** in live trading
4. **Enable 2FA** on exchange accounts
5. **Keep software updated** regularly
6. **Monitor system logs** for suspicious activity

## System Status

Once running, you can check system status:

- 🟢 **Healthy**: All systems operational
- 🟡 **Warning**: Minor issues detected
- 🔴 **Critical**: Immediate attention required

Monitor at: http://localhost:8000/status

---

## Quick Reference Card

```bash
# Essential Commands
docker-compose up -d                    # Start all services
curl http://localhost:8000/health       # Health check
curl -H "Auth: Bearer $KEY" /status     # System status
python -m pytest                       # Run tests
tail -f logs/app.log                   # View logs

# API Endpoints
GET  /health          # Health check (no auth)
GET  /status          # System status
GET  /metrics         # Performance metrics  
GET  /alerts          # System alerts
POST /system/command  # Control commands

# Configuration Files
config/config.yml     # Main configuration
.env                  # Environment variables
docker-compose.yml    # Docker setup
requirements.txt      # Python dependencies
```

You're ready to start trading! 🚀

For detailed documentation, see the complete guides in the `docs/` directory.
# Comprehensive Development Plan: Advanced Crypto Trading Bot with Dynamic Risk Management

## Design Philosophy
**Scientific Rigor Over Speed:** This bot prioritizes statistical validation and risk management over low-latency execution. It's designed for strategic edges rather than high-frequency trading. The system incorporates a toggleable aggressive mode that dynamically adjusts risk parameters based on account balance, enabling rapid growth of small accounts while automatically reducing risk as capital increases.

**Core Design Principles:**
1. **Statistical Validation:** Every strategy must pass multiple rigorous tests before earning capital
2. **Robustness:** Handle market anomalies, exchange downtime, and unexpected errors gracefully
3. **Auditability:** Complete transaction history for performance analysis and tax compliance
4. **Automation:** Full lifecycle management from strategy discovery to retirement
5. **Maintainability:** Clean, modular codebase that can be extended over time
6. **Adaptive Risk:** Dynamic risk adjustment based on account balance and market conditions

## Key Features

### Strategy Development & Validation
- **Multiple Strategy Types:** Technical indicators, machine learning models, and hybrid approaches
- **Walk-Forward Optimization (WFO):** Rolling window validation approach
- **Permutation Testing:** Statistical significance testing against random chance
- **Combinatorial Symmetric Cross-Validation (CSCV):** Probability of backtest overfitting (PBO) calculation
- **Purged TimeSeries Cross-Validation:** ML-specific validation avoiding lookahead bias
- **Mode-Specific Validation:** Different validation thresholds for conservative vs. aggressive modes

### Risk Management System
- **Dual-Mode Operation:** Toggle between conservative and aggressive trading modes
- **Dynamic Position Sizing:** Volatility-adjusted position sizing with balance-based scaling
- **Multi-layer Protection:** Strategy-level, asset-level, and portfolio-level risk limits
- **Correlation Monitoring:** Real-time correlation analysis between strategies
- **Circuit Breakers:** Automatic trading halt on technical issues or extreme market events
- **Drawdown Limits:** Hard stops at multiple levels with mode-dependent thresholds
- **Balance-Based Risk Scaling:** Automatic risk reduction as account balance grows
- **Value-at-Risk (VaR) Limits:** Daily loss limits with mode-specific thresholds

### Machine Learning Integration
- **Feature Engineering:** Lagged technical indicators and statistical features
- **Model Training:** LightGBM/XGBoost with purged cross-validation
- **Regime Filtering:** Market condition detection for strategy activation
- **Probability Forecasting:** Confidence-based position sizing
- **Adaptive Learning:** Model retraining based on market regime changes

### Tax Compliance (Australian Focus)
- **AUD Conversion:** Historical rate integration with RBA API
- **CGT Calculation:** FIFO method with automatic event tagging
- **Comprehensive Reporting:** Financial year summaries and tax-ready documents
- **Trade Categorization:** Detailed record keeping for accounting purposes
- **Mode-Aware Tracking:** Special handling for aggressive mode trading activity

### Advanced Features
- **Market Regime Detection:** Volatility and trend-based market state classification
- **Portfolio Optimization:** Mean-variance and risk-parity allocation methods
- **Automated Reporting:** Daily PDF performance reports with email distribution
- **News Sentiment Integration:** News blackout rules based on market events
- **Streamlit Dashboard:** Real-time monitoring and performance analytics with mode control
- **Dynamic Parameter Adjustment:** Continuous risk parameter optimization based on performance

## Technical Specifications

### Supported Markets
- **Primary Exchange:** Bybit
- **Instruments:** Perpetual swaps (BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT)
- **Timeframes:** 1h, 4h, 1d (configurable)

### Technology Stack
- **Language:** Python 3.11+
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, LightGBM, XGBoost
- **Database:** DuckDB (development), PostgreSQL (production)
- **Backtesting:** Custom event-driven engine
- **Visualization:** Plotly, Streamlit
- **Deployment:** Docker, Docker Compose
- **Monitoring:** Custom dashboard with performance metrics

### Deployment Options
1. **Local Development:** Docker on home laptop (DuckDB)
2. **Production Deployment:** DigitalOcean droplet with Docker (PostgreSQL)

### Risk Parameters Configuration
```yaml
# Default Conservative Mode Parameters
conservative_mode:
  portfolio_drawdown_limit: 0.25
  strategy_drawdown_limit: 0.15
  sharpe_ratio_min: 0.8
  var_daily_limit: 0.03
  risk_per_trade: 0.01
  consistency_min: 0.60

# Aggressive Mode Base Parameters
aggressive_mode:
  portfolio_drawdown_limit: 0.40
  strategy_drawdown_limit: 0.25
  sharpe_ratio_min: 0.5
  var_daily_limit: 0.05
  max_risk_ratio: 0.02
  min_risk_ratio: 0.005
  consistency_min: 0.50
  balance_thresholds:
    low: 1000  # Balance below this uses max risk
    high: 10000  # Balance above this uses min risk
```

### Validation Pipeline
1. **Research:** Strategy development in Jupyter notebooks
2. **Validation:** Automated WFO, permutation, and CSCV testing with mode-specific thresholds
3. **Paper Trading:** Live simulation with historical data
4. **Live Trading:** Gradual capital allocation with dynamic risk adjustment
5. **Monitoring:** Continuous performance surveillance with mode-aware checks
6. **Retirement:** Automatic decommissioning on performance decay

## Implementation Plan

### Phase 0: Foundation & Setup (Week 1-2)
1. Create GitHub repository with proper structure
2. Set up Python 3.11+ virtual environment
3. Initialize project structure with modules for data, strategies, ML, backtesting, risk, execution, and database
4. Create Dockerfile and docker-compose.yml for containerization
5. Set up requirements.txt with all necessary packages
6. Implement configuration management with mode settings

### Phase 1: Foundational Infrastructure (Weeks 1-2)
1. Implement database layer with SQLAlchemy and DuckDB/PostgreSQL
2. Define data models including Trade model with tax tracking fields
3. Create data acquisition and sanitization module
4. Implement basic logging and error handling
5. Add configuration structure for dual-mode operation

### Phase 2: Advanced Testing & Validation Engine (Weeks 3-5)
1. Implement Walk-Forward Analysis (WFO) with rolling windows
2. Develop Combinatorial Symmetric Cross-Validation (CSCV) for PBO calculation
3. Create permutation testing framework
4. Implement Purged TimeSeries Cross-Validation for ML models
5. Build validation pipeline with mode-specific thresholds
6. Add dynamic validation criteria based on trading mode

### Phase 3: Machine Learning Integration (Weeks 6-8)
1. Develop feature engineering module with lagged features
2. Create target definition for classification and regression
3. Implement model training with purged cross-validation
4. Build LightGBM/XGBoost strategy class
5. Develop model validation with financial metrics
6. Add regime-aware model selection

### Phase 4: Risk, Execution & Core Loop (Weeks 9-10)
1. Implement dynamic position sizing with balance-based scaling
2. Create strategy correlation monitoring and limits
3. Build portfolio-level drawdown limits with mode variations
4. Develop execution abstraction with PaperTrade and Live clients
5. Implement state machine for core trading loop
6. Add dynamic risk parameter system

### Phase 5: Dynamic Risk System Implementation (Week 11)
1. Create DynamicRiskManager class (`src/bot/risk/dynamic_risk.py`)
2. Implement balance-based risk scaling algorithm
3. Add mode transition logic with smooth parameter changes
4. Develop risk parameter interpolation system
5. Create configuration management for risk parameters
6. Implement real-time risk adjustment triggers

### Phase 6: Validation Pipeline & Automation (Week 11)
1. Create automated strategy validation pipeline
2. Implement strategy lifecycle management with mode awareness
3. Build performance monitoring and strategy retirement rules
4. Add mode-specific validation criteria
5. Implement automatic mode switching based on performance

### Phase 7: Australian Tax Tracking (Week 12)
1. Implement AUD conversion service using RBA API
2. Develop CGT calculator with FIFO matching
3. Create tax report generator
4. Build database models for tax tracking
5. Add special handling for aggressive mode trading activity

### Phase 8: Deployment & Dashboard (Weeks 13-14)
1. Containerize application with Docker
2. Implement Streamlit dashboard with mode control interface
3. Add real-time risk parameter visualization
4. Create performance analytics with mode comparison
5. Deploy to DigitalOcean droplet

### Phase 9: Advanced Features Integration (Week 15)
1. Implement market regime detection and filtering
2. Develop portfolio optimization systems
3. Create automated reporting with mode performance analysis
4. Integrate news sentiment analysis with mode-specific rules
5. Build dynamic parameter optimization system

### Phase 10: Final Integration & Testing (Week 16)
1. End-to-end testing of complete system
2. Performance optimization across both modes
3. Security review and hardening
4. Documentation completion
5. Create mode transition tutorial and risk guidelines

## Dynamic Risk Implementation Details

### Configuration Structure
```yaml
trading:
  mode: aggressive  # Options: 'conservative' or 'aggressive'
  base_balance: 1000  # Base balance in USDT for risk scaling
  max_risk_ratio: 0.02  # Max risk per trade (2%) in aggressive mode
  min_risk_ratio: 0.005  # Min risk per trade (0.5%) in aggressive mode
  balance_thresholds:
    low: 1000  # Balance below this uses max risk
    high: 10000  # Balance above this uses min risk
  risk_decay: exponential  # Options: 'linear' or 'exponential'
```

### Risk Scaling Algorithm
The system uses either linear or exponential decay for risk reduction as balance grows:

1. **Linear Decay:** Risk decreases steadily between balance thresholds
2. **Exponential Decay:** Risk decreases rapidly initially, then slows

The formula for linear risk adjustment:
```
risk_ratio = max_risk_ratio - (max_risk_ratio - min_risk_ratio) * 
             ((current_balance - low_threshold) / 
             (high_threshold - low_threshold))
```

### Mode Transition Handling
- Smooth parameter changes to avoid sudden position size changes
- Graceful strategy deactivation when reducing risk
- Comprehensive logging of all parameter adjustments
- Performance tracking during mode transitions

### Dashboard Controls
The Streamlit dashboard includes:
- Mode selection dropdown with confirmation dialog
- Real-time display of current risk parameters
- Balance-based risk adjustment visualization
- Performance comparison between modes
- One-click mode switching with safety checks

## Risk Management Rules
- **Maximum portfolio drawdown:** 25% (conservative) / 40% (aggressive)
- **Maximum strategy drawdown:** 15% (conservative) / 25% (aggressive)
- **Minimum strategy Sharpe ratio:** 0.8 (conservative) / 0.5 (aggressive)
- **Maximum strategy correlation:** 0.7
- **Daily value-at-risk (VaR) limit:** 3% (conservative) / 5% (aggressive)
- **Minimum OOS performance consistency:** 60% (conservative) / 50% (aggressive)

## Tax Considerations for Aggressive Mode
- Higher trading frequency may increase tax complexity
- Additional reporting for aggressive mode periods
- Separate performance tracking for tax purposes
- Automated CGT calculation for frequent trading
- Integration with Australian tax software APIs

This comprehensive plan creates a sophisticated trading system with adaptive risk management that can aggressively grow small accounts while automatically protecting larger balances. The dual-mode operation provides flexibility for different market conditions and risk appetites while maintaining the core principles of statistical validation and robustness.