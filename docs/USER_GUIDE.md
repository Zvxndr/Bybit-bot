# User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Configuration](#basic-configuration)
3. [Trading Strategies](#trading-strategies)
4. [Risk Management](#risk-management)
5. [Monitoring and Alerts](#monitoring-and-alerts)
6. [Common Use Cases](#common-use-cases)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Getting Started

### Overview

The Bybit Trading Bot is an automated cryptocurrency trading system that executes trades based on configurable strategies while managing risk automatically. The bot supports multiple trading strategies, comprehensive risk management, and real-time monitoring.

### Key Features

- **Automated Trading**: Execute trades 24/7 based on predefined strategies
- **Risk Management**: Built-in risk controls to protect your capital
- **Multiple Strategies**: Support for various trading strategies
- **Real-time Monitoring**: Live dashboard and alerting system
- **Backtesting**: Test strategies with historical data
- **Paper Trading**: Practice with virtual funds before going live

### Quick Start

1. **Sign up for Bybit**: Create account at [bybit.com](https://www.bybit.com)
2. **Generate API Keys**: Create API keys in your Bybit account settings
3. **Install the Bot**: Follow the [deployment guide](DEPLOYMENT.md)
4. **Configure Settings**: Set up your trading preferences
5. **Start Trading**: Begin with paper trading, then move to live trading

## Basic Configuration

### Initial Setup

1. **Copy Configuration Template**
   ```bash
   cp config/default.yaml config/my_config.yaml
   ```

2. **Edit Configuration File**
   ```yaml
   # config/my_config.yaml
   app:
     name: "My Trading Bot"
     environment: "development"  # Change to "production" for live trading
   
   # Trading settings
   trading:
     enabled: true
     max_concurrent_trades: 3
     max_position_size: 0.02  # 2% of portfolio per trade
   
   # Risk management
   risk:
     max_portfolio_risk: 0.01  # 1% of portfolio at risk per trade
     max_daily_loss: 0.03      # 3% maximum daily loss
   
   # Exchange settings
   exchange:
     api:
       testnet: true  # Set to false for live trading
   ```

3. **Set Environment Variables**
   ```bash
   # .env file
   BYBIT_API_KEY=your_api_key_here
   BYBIT_API_SECRET=your_api_secret_here
   ENVIRONMENT=development
   ```

### Configuration Sections

#### App Configuration
```yaml
app:
  name: "Your Bot Name"           # Bot identifier
  environment: "development"      # development/staging/production
  log_level: "INFO"              # DEBUG/INFO/WARNING/ERROR
```

#### Trading Configuration
```yaml
trading:
  enabled: true                   # Enable/disable trading
  max_concurrent_trades: 5        # Maximum simultaneous trades
  max_position_size: 0.05         # Maximum position size (5% of portfolio)
  min_trade_value: 20.0          # Minimum trade value in USDT
  
  # Order settings
  default_order_type: "LIMIT"     # LIMIT or MARKET orders
  slippage_tolerance: 0.001       # 0.1% slippage tolerance
  
  # Stop loss and take profit
  enable_stop_loss: true
  enable_take_profit: true
  default_stop_loss_pct: 0.02     # 2% stop loss
  default_take_profit_pct: 0.04   # 4% take profit
```

#### Risk Management Configuration
```yaml
risk:
  max_portfolio_risk: 0.02        # 2% portfolio risk per trade
  max_daily_loss: 0.05           # 5% maximum daily loss
  max_drawdown: 0.15             # 15% maximum drawdown
  
  # Circuit breakers
  circuit_breakers:
    enabled: true
    daily_loss_threshold: 0.03    # Stop trading at 3% daily loss
    consecutive_losses: 5         # Stop after 5 consecutive losses
```

## Trading Strategies

### Available Strategies

#### 1. Moving Average Crossover
Trades based on moving average crossovers - buy when fast MA crosses above slow MA, sell when it crosses below.

```yaml
strategies:
  ma_crossover:
    enabled: true
    class: "MovingAverageCrossoverStrategy"
    parameters:
      fast_period: 10    # Fast moving average period
      slow_period: 50    # Slow moving average period
      min_confidence: 0.7 # Minimum signal confidence
    symbols:
      - "BTCUSDT"
      - "ETHUSDT"
```

**Best for**: Trending markets, longer-term trades
**Time frames**: 1h, 4h, 1d
**Risk level**: Medium

#### 2. RSI Strategy
Uses RSI (Relative Strength Index) to identify overbought/oversold conditions.

```yaml
strategies:
  rsi_strategy:
    enabled: true
    class: "RSIStrategy"
    parameters:
      period: 14          # RSI calculation period
      overbought: 70      # Overbought threshold
      oversold: 30        # Oversold threshold
      min_confidence: 0.6
    symbols:
      - "BTCUSDT"
      - "ETHUSDT"
```

**Best for**: Range-bound markets, mean reversion
**Time frames**: 15m, 1h, 4h
**Risk level**: Medium

#### 3. Bollinger Bands Strategy
Trades based on Bollinger Bands - buy at lower band, sell at upper band.

```yaml
strategies:
  bollinger_bands:
    enabled: true
    class: "BollingerBandsStrategy"
    parameters:
      period: 20          # Moving average period
      std_dev: 2.0       # Standard deviation multiplier
      min_confidence: 0.65
    symbols:
      - "SOLUSDT"
      - "ADAUSDT"
```

**Best for**: Volatile markets, scalping
**Time frames**: 5m, 15m, 1h
**Risk level**: High

### Strategy Configuration

#### Single Strategy Setup
```yaml
# Simple MA crossover for Bitcoin
strategies:
  btc_ma_crossover:
    enabled: true
    class: "MovingAverageCrossoverStrategy"
    parameters:
      fast_period: 20
      slow_period: 50
      min_confidence: 0.75
    symbols:
      - "BTCUSDT"
    weight: 1.0  # 100% allocation
```

#### Multi-Strategy Setup
```yaml
# Diversified strategy portfolio
strategies:
  # Long-term trend following
  ma_longterm:
    enabled: true
    class: "MovingAverageCrossoverStrategy"
    parameters:
      fast_period: 50
      slow_period: 200
    symbols: ["BTCUSDT", "ETHUSDT"]
    weight: 0.5  # 50% allocation
  
  # Short-term mean reversion
  rsi_shortterm:
    enabled: true
    class: "RSIStrategy"
    parameters:
      period: 14
      overbought: 75
      oversold: 25
    symbols: ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    weight: 0.3  # 30% allocation
  
  # Volatility trading
  bollinger_scalp:
    enabled: true
    class: "BollingerBandsStrategy"
    parameters:
      period: 20
      std_dev: 2.5
    symbols: ["SOLUSDT", "AVAXUSDT"]
    weight: 0.2  # 20% allocation
```

#### Symbol-Specific Settings
```yaml
# Different settings per symbol
symbol_configs:
  BTCUSDT:
    max_position_size: 0.05      # 5% max position for Bitcoin
    min_confidence: 0.8          # Higher confidence required
    stop_loss_pct: 0.015         # 1.5% stop loss
    take_profit_pct: 0.03        # 3% take profit
  
  ETHUSDT:
    max_position_size: 0.04      # 4% max position for Ethereum
    min_confidence: 0.75
    stop_loss_pct: 0.02          # 2% stop loss
    take_profit_pct: 0.04        # 4% take profit
  
  ADAUSDT:
    max_position_size: 0.02      # 2% max position for smaller caps
    min_confidence: 0.85         # Higher confidence for altcoins
    stop_loss_pct: 0.03          # 3% stop loss
    take_profit_pct: 0.06        # 6% take profit
```

## Risk Management

### Risk Levels

#### Conservative (Recommended for beginners)
```yaml
risk:
  max_portfolio_risk: 0.005      # 0.5% per trade
  max_daily_loss: 0.02           # 2% daily limit
  max_drawdown: 0.10             # 10% maximum drawdown
  confidence_threshold: 0.8       # High confidence required
  min_risk_reward_ratio: 2.0     # 2:1 reward to risk
```

#### Moderate (Balanced approach)
```yaml
risk:
  max_portfolio_risk: 0.01       # 1% per trade
  max_daily_loss: 0.03           # 3% daily limit
  max_drawdown: 0.15             # 15% maximum drawdown
  confidence_threshold: 0.7       # Moderate confidence
  min_risk_reward_ratio: 1.5     # 1.5:1 reward to risk
```

#### Aggressive (Higher risk/reward)
```yaml
risk:
  max_portfolio_risk: 0.02       # 2% per trade
  max_daily_loss: 0.05           # 5% daily limit
  max_drawdown: 0.20             # 20% maximum drawdown
  confidence_threshold: 0.6       # Lower confidence acceptable
  min_risk_reward_ratio: 1.2     # 1.2:1 reward to risk
```

### Position Sizing

The bot automatically calculates position sizes based on:

1. **Portfolio Risk**: Maximum % of portfolio to risk per trade
2. **Stop Loss Distance**: Distance to stop loss price
3. **Available Balance**: Current account balance
4. **Symbol Configuration**: Symbol-specific limits

**Example Calculation**:
- Portfolio Value: $10,000
- Max Portfolio Risk: 1% = $100
- Stop Loss Distance: 2% from entry
- Position Size: $100 ÷ 0.02 = $5,000 worth of crypto

### Circuit Breakers

Automatic protections that stop trading when triggered:

```yaml
risk:
  circuit_breakers:
    enabled: true
    
    # Daily loss limit
    daily_loss_threshold: 0.03     # Stop at 3% daily loss
    
    # Consecutive losses
    consecutive_losses: 5          # Stop after 5 losses in a row
    
    # Rapid loss protection
    rapid_loss_threshold: 0.02     # Stop if 2% lost in 1 hour
    rapid_loss_window: 3600        # 1 hour window
    
    # Drawdown protection
    max_drawdown_stop: 0.15        # Emergency stop at 15% drawdown
```

## Monitoring and Alerts

### Dashboard Access

Access the monitoring dashboard at `http://localhost:8080` (or your deployed URL).

**Dashboard Sections**:
- **Portfolio**: Current value, P&L, positions
- **Trading**: Active trades, recent signals, performance
- **Risk**: Current risk metrics, circuit breaker status
- **System**: Bot status, API connectivity, errors

### Setting Up Alerts

#### Email Alerts
```yaml
monitoring:
  alerts:
    email:
      enabled: true
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      from_email: "your-bot@gmail.com"
      to_emails: ["your-email@gmail.com"]
      
      # Alert conditions
      triggers:
        - trade_executed
        - circuit_breaker_triggered
        - daily_loss_threshold
        - api_connection_lost
```

#### Slack Alerts
```yaml
monitoring:
  alerts:
    slack:
      enabled: true
      webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
      channel: "#trading-alerts"
      
      triggers:
        - portfolio_milestone     # Every 5% portfolio change
        - large_trade            # Trades > $1000
        - system_error
```

#### Telegram Alerts
```yaml
monitoring:
  alerts:
    telegram:
      enabled: true
      bot_token: "YOUR_BOT_TOKEN"
      chat_id: "YOUR_CHAT_ID"
      
      triggers:
        - all  # Send all alerts to Telegram
```

### Alert Types

- **Trade Alerts**: New trades, filled orders, cancelled orders
- **Risk Alerts**: Circuit breakers, high drawdown, consecutive losses
- **System Alerts**: API errors, connection issues, bot restarts
- **Performance Alerts**: Daily P&L, portfolio milestones, new highs/lows

## Common Use Cases

### 1. Conservative Long-Term Trading

**Objective**: Steady growth with minimal risk
**Strategy**: Moving average crossover on daily timeframes
**Risk**: 0.5% per trade, 2% daily limit

```yaml
# Conservative setup
trading:
  max_concurrent_trades: 2
  max_position_size: 0.02

risk:
  max_portfolio_risk: 0.005
  max_daily_loss: 0.02
  confidence_threshold: 0.8

strategies:
  ma_conservative:
    enabled: true
    class: "MovingAverageCrossoverStrategy"
    parameters:
      fast_period: 50
      slow_period: 200
      min_confidence: 0.8
    symbols: ["BTCUSDT", "ETHUSDT"]

exchange:
  symbols: ["BTCUSDT", "ETHUSDT"]  # Only major coins
```

### 2. Active Day Trading

**Objective**: Frequent trades for daily profits
**Strategy**: Multiple strategies on shorter timeframes
**Risk**: 1% per trade, 3% daily limit

```yaml
# Day trading setup
trading:
  max_concurrent_trades: 8
  max_position_size: 0.03
  default_stop_loss_pct: 0.015
  default_take_profit_pct: 0.03

risk:
  max_portfolio_risk: 0.01
  max_daily_loss: 0.03
  confidence_threshold: 0.7

strategies:
  rsi_scalp:
    enabled: true
    class: "RSIStrategy"
    parameters:
      period: 14
      overbought: 70
      oversold: 30
    symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
  
  bb_scalp:
    enabled: true
    class: "BollingerBandsStrategy"
    parameters:
      period: 20
      std_dev: 2.0
    symbols: ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

data:
  market_data:
    intervals: ["5m", "15m", "1h"]
    update_frequency: 30  # Update every 30 seconds
```

### 3. Portfolio Diversification

**Objective**: Spread risk across multiple strategies and symbols
**Strategy**: Multiple strategies with different allocations
**Risk**: Balanced approach

```yaml
# Diversified portfolio
strategies:
  # 40% Trend following
  trend_following:
    enabled: true
    class: "MovingAverageCrossoverStrategy"
    weight: 0.4
    symbols: ["BTCUSDT", "ETHUSDT"]
  
  # 30% Mean reversion
  mean_reversion:
    enabled: true
    class: "RSIStrategy" 
    weight: 0.3
    symbols: ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
  
  # 20% Momentum
  momentum:
    enabled: true
    class: "BollingerBandsStrategy"
    weight: 0.2
    symbols: ["SOLUSDT", "AVAXUSDT", "MATICUSDT"]
  
  # 10% Experimental
  experimental:
    enabled: true
    class: "CustomStrategy"
    weight: 0.1
    symbols: ["DOTUSDT", "LINKUSDT"]

# Different risk levels per strategy
risk:
  strategy_risk_limits:
    trend_following: 0.008      # 0.8% per trade
    mean_reversion: 0.012       # 1.2% per trade
    momentum: 0.015             # 1.5% per trade (higher risk)
    experimental: 0.005         # 0.5% per trade (lower risk)
```

### 4. Automated DCA (Dollar Cost Averaging)

**Objective**: Regular purchases regardless of price
**Strategy**: Time-based buying strategy
**Risk**: Very low, just buying regularly

```yaml
# DCA strategy
strategies:
  btc_dca:
    enabled: true
    class: "DCAStrategy"
    parameters:
      interval: "daily"           # Buy every day
      amount: 50                  # $50 per purchase
      symbol: "BTCUSDT"
      
  eth_dca:
    enabled: true
    class: "DCAStrategy"
    parameters:
      interval: "weekly"          # Buy every week
      amount: 100                 # $100 per purchase
      symbol: "ETHUSDT"

# Minimal risk management for DCA
risk:
  max_portfolio_risk: 1.0        # DCA doesn't use stop losses
  circuit_breakers:
    enabled: false               # Don't stop DCA
```

## Troubleshooting

### Common Issues

#### Bot Won't Start

**Symptoms**: Bot crashes on startup or shows connection errors

**Solutions**:
1. Check API credentials:
   ```bash
   # Verify API key is set
   echo $BYBIT_API_KEY
   
   # Test API connection
   curl -H "X-BAPI-API-KEY: $BYBIT_API_KEY" https://api-testnet.bybit.com/v2/public/time
   ```

2. Verify configuration:
   ```bash
   # Validate configuration file
   python scripts/validate_config.py config/your_config.yaml
   ```

3. Check logs:
   ```bash
   # View recent logs
   tail -f logs/trading_bot.log
   
   # Search for errors
   grep ERROR logs/trading_bot.log
   ```

#### No Trades Being Executed

**Symptoms**: Bot runs but doesn't place any trades

**Possible Causes**:
1. **No trading signals**: Strategies aren't generating signals
2. **Risk limits**: Trades blocked by risk management
3. **Insufficient balance**: Not enough funds to trade
4. **Configuration issues**: Trading disabled or misconfigured

**Solutions**:
1. Check strategy signals:
   ```bash
   # View strategy status
   curl http://localhost:8080/api/strategies/status
   
   # Check recent signals
   curl http://localhost:8080/api/signals/recent
   ```

2. Review risk limits:
   ```bash
   # Check risk status
   curl http://localhost:8080/api/risk/status
   
   # View portfolio metrics
   curl http://localhost:8080/api/portfolio/metrics
   ```

3. Verify balance:
   ```bash
   # Check account balance
   curl http://localhost:8080/api/account/balance
   ```

#### High CPU/Memory Usage

**Symptoms**: Bot consuming excessive resources

**Solutions**:
1. Reduce data frequency:
   ```yaml
   data:
     market_data:
       update_frequency: 120  # Increase from 60 to 120 seconds
       max_candles: 500       # Reduce from 1000
   ```

2. Limit concurrent trades:
   ```yaml
   trading:
     max_concurrent_trades: 3  # Reduce from higher number
   ```

3. Optimize strategies:
   ```yaml
   strategies:
     # Disable resource-intensive strategies
     complex_strategy:
       enabled: false
   ```

#### API Rate Limiting

**Symptoms**: API errors about rate limits

**Solutions**:
1. Reduce API call frequency:
   ```yaml
   exchange:
     rate_limits:
       orders_per_second: 5      # Reduce from 10
       requests_per_minute: 60   # Reduce from 120
   ```

2. Implement backoff strategy:
   ```yaml
   exchange:
     api:
       max_retries: 5
       retry_delay: 2            # Increase delay
   ```

### Performance Issues

#### Slow Trade Execution

**Causes**: Network latency, server location, order type

**Solutions**:
1. Use market orders for speed:
   ```yaml
   trading:
     default_order_type: "MARKET"
   ```

2. Reduce slippage tolerance:
   ```yaml
   trading:
     slippage_tolerance: 0.002  # 0.2%
   ```

3. Deploy closer to exchange servers (Singapore/Tokyo for Bybit)

#### Memory Leaks

**Symptoms**: Memory usage increasing over time

**Solutions**:
1. Restart bot regularly:
   ```bash
   # Add to crontab for daily restart
   0 0 * * * docker-compose restart trading-bot
   ```

2. Limit data retention:
   ```yaml
   data:
     storage:
       retention_days: 30    # Reduce from 90
   ```

3. Enable garbage collection logging:
   ```yaml
   logging:
     components:
       memory_manager: "DEBUG"
   ```

### Getting Help

1. **Check Documentation**: Review [API documentation](API.md) and [architecture guide](ARCHITECTURE.md)
2. **View Logs**: Always check logs first for error details
3. **Monitor Dashboard**: Use the web dashboard to diagnose issues
4. **Test Configuration**: Validate configuration files before deployment
5. **Use Paper Trading**: Test changes in paper trading mode first

## Best Practices

### Security

1. **API Key Management**:
   - Use read-only keys for testing
   - Store keys securely (environment variables, not config files)
   - Regularly rotate API keys
   - Enable IP whitelisting on Bybit

2. **Configuration Security**:
   - Don't commit sensitive data to version control
   - Use separate configs for different environments
   - Validate configuration before deployment

3. **System Security**:
   - Keep software updated
   - Use firewalls and secure networks
   - Monitor for suspicious activity
   - Implement access logging

### Risk Management

1. **Start Small**:
   - Begin with paper trading
   - Use small position sizes initially
   - Gradually increase as you gain confidence

2. **Diversify**:
   - Don't put all funds in one strategy
   - Trade multiple symbols
   - Use different timeframes

3. **Monitor Regularly**:
   - Check performance daily
   - Review and adjust strategies
   - Set up proper alerts

4. **Have Exit Plans**:
   - Know when to stop a strategy
   - Set clear loss limits
   - Regular profit taking

### Performance Optimization

1. **Resource Management**:
   - Monitor CPU and memory usage
   - Optimize database queries
   - Use appropriate data retention periods

2. **Strategy Optimization**:
   - Backtest strategies thoroughly
   - Monitor real performance vs. backtest
   - Adjust parameters based on market conditions

3. **System Maintenance**:
   - Regular log cleanup
   - Database maintenance
   - Software updates
   - Performance monitoring

### Trading Psychology

1. **Stay Disciplined**:
   - Follow your strategy rules
   - Don't override the bot impulsively
   - Keep emotions out of trading decisions

2. **Continuous Learning**:
   - Analyze winning and losing trades
   - Study market conditions
   - Adapt strategies to changing markets

3. **Realistic Expectations**:
   - Understand that losses are normal
   - Focus on long-term performance
   - Don't expect to win every trade