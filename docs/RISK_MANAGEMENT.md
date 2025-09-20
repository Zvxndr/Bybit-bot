# 🛡️ Risk Management System Documentation

## Overview

The ML Trading Bot implements a sophisticated, multi-layered risk management system designed to protect your capital while maximizing returns. The system uses **dynamic risk scaling** that automatically adjusts risk parameters based on your account size and market conditions.

## 🎯 Core Risk Management Philosophy

### Dynamic Risk Scaling
The bot operates in **aggressive mode** with exponential risk decay, meaning:
- **Small accounts** take higher risk to maximize growth potential
- **Large accounts** take lower risk to preserve capital
- **Risk scales smoothly** between thresholds

## 📊 Risk Configuration Details

### Current Configuration (`config/config.yaml`)

```yaml
trading:
  mode: aggressive  # Current mode
  base_balance: 10000  # Base balance for risk calculations
  
  aggressive_mode:
    max_risk_ratio: 0.02      # 2% max risk per trade
    min_risk_ratio: 0.005     # 0.5% min risk per trade
    balance_thresholds:
      low: 10000              # $10k threshold
      high: 100000            # $100k threshold
    risk_decay: exponential   # Smooth transition
    
    # Additional safety limits
    portfolio_drawdown_limit: 0.40    # 40% max portfolio drawdown
    strategy_drawdown_limit: 0.25     # 25% max strategy drawdown
    var_daily_limit: 0.05             # 5% daily Value at Risk limit
```

## 🎚️ Risk Scaling Formula

### How Risk is Calculated

For balances between $10k and $100k, risk is calculated using exponential decay:

```python
def calculate_risk_ratio(balance: float) -> float:
    """Calculate risk ratio based on current balance"""
    if balance <= 10000:
        return 0.02  # 2% for small accounts
    elif balance >= 100000:
        return 0.005  # 0.5% for large accounts
    else:
        # Exponential decay between thresholds
        ratio = (balance - 10000) / (100000 - 10000)
        decay_factor = math.exp(-2 * ratio)  # Exponential decay
        return 0.005 + (0.015 * decay_factor)
```

### Risk Scaling Examples

| Account Balance | Risk Per Trade | Position Size (Stop Loss = 2%) |
|-----------------|----------------|--------------------------------|
| $5,000 | 2.00% | $100 → $5,000 position |
| $10,000 | 2.00% | $200 → $10,000 position |
| $25,000 | 1.20% | $300 → $15,000 position |
| $50,000 | 0.80% | $400 → $20,000 position |
| $75,000 | 0.65% | $487 → $24,350 position |
| $100,000 | 0.50% | $500 → $25,000 position |
| $200,000 | 0.50% | $1,000 → $50,000 position |

## 🔒 Multi-Layer Protection System

### Layer 1: Position-Level Risk
- **Stop Loss**: Automatic stop losses on every position
- **Take Profit**: Automatic profit taking
- **Position Sizing**: Risk-based position sizing
- **Maximum Leverage**: Capped at 5x (configurable)

### Layer 2: Strategy-Level Risk
- **Strategy Drawdown Limit**: 25% maximum drawdown per strategy
- **Correlation Monitoring**: Prevents highly correlated positions
- **Performance Tracking**: Disables underperforming strategies

### Layer 3: Portfolio-Level Risk
- **Portfolio Drawdown Limit**: 40% maximum total drawdown
- **Daily Loss Limit**: 5% maximum daily loss
- **Sector Exposure**: Maximum 40% in any single sector
- **Value at Risk (VaR)**: 95% confidence level monitoring

### Layer 4: System-Level Risk
- **Circuit Breakers**: Auto-stop during extreme volatility
- **API Monitoring**: Detects and handles API failures
- **Health Checks**: Continuous system monitoring
- **Emergency Stops**: Manual and automatic emergency stops

## ⚙️ Risk Monitoring Components

### Real-Time Risk Metrics
- **Current Drawdown**: Live portfolio drawdown tracking
- **Daily P&L**: Real-time profit/loss monitoring
- **Position Risk**: Individual position risk assessment
- **Portfolio VaR**: Value at Risk calculations
- **Correlation Matrix**: Inter-asset correlation monitoring

### Automated Risk Responses
```python
# Example automated responses
if current_drawdown > 0.25:  # 25% drawdown
    reduce_position_sizes(factor=0.5)
    
if daily_loss > 0.05:  # 5% daily loss
    halt_new_positions()
    
if market_volatility > 0.80:  # 80th percentile volatility
    activate_circuit_breaker()
```

## 🚨 Alert System

### Risk Alerts Configuration
```yaml
alerts:
  risk_warnings:
    - trigger: "drawdown > 15%"
      action: "email + dashboard notification"
    - trigger: "daily_loss > 3%"
      action: "email notification"
    - trigger: "position_loss > 5%"
      action: "dashboard notification"
      
  critical_alerts:
    - trigger: "drawdown > 25%"
      action: "email + SMS + reduce positions"
    - trigger: "daily_loss > 5%"
      action: "email + SMS + halt trading"
    - trigger: "api_failure"
      action: "email + SMS + emergency stop"
```

## 🎛️ Customizing Risk Parameters

### Conservative Mode (Alternative)
If you prefer fixed risk regardless of account size:

```yaml
trading:
  mode: conservative
  
  conservative_mode:
    risk_ratio: 0.01                    # Fixed 1% risk per trade
    portfolio_drawdown_limit: 0.25      # 25% max drawdown
    strategy_drawdown_limit: 0.15       # 15% max strategy drawdown
    var_daily_limit: 0.03               # 3% daily VaR limit
```

### Custom Risk Thresholds
You can modify the balance thresholds in `config/config.yaml`:

```yaml
aggressive_mode:
  balance_thresholds:
    low: 5000      # Start risk decay at $5k
    high: 50000    # End risk decay at $50k
  max_risk_ratio: 0.03  # 3% max risk (more aggressive)
  min_risk_ratio: 0.003 # 0.3% min risk (more conservative)
```

## 📈 Risk-Adjusted Performance Metrics

### Key Metrics Tracked
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside deviation adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Annual return / Maximum drawdown
- **Value at Risk (VaR)**: Worst expected loss at 95% confidence
- **Conditional VaR**: Expected loss beyond VaR threshold

### Performance Thresholds
```yaml
performance_requirements:
  aggressive_mode:
    min_sharpe: 0.5              # Minimum Sharpe ratio
    max_drawdown: 0.25           # Maximum drawdown
    min_win_rate: 0.40           # Minimum win rate
    min_profit_factor: 1.1       # Minimum profit factor
    
  conservative_mode:
    min_sharpe: 0.8              # Higher Sharpe requirement
    max_drawdown: 0.15           # Lower drawdown tolerance
    min_win_rate: 0.45           # Higher win rate requirement
    min_profit_factor: 1.2       # Higher profit factor requirement
```

## 🔧 Risk Management Best Practices

### 1. Account Size Guidelines
- **Under $1,000**: Consider paper trading first
- **$1,000 - $10,000**: Default aggressive mode works well
- **$10,000 - $100,000**: Monitor risk scaling closely
- **Over $100,000**: Consider conservative mode or custom thresholds

### 2. Market Condition Adjustments
- **Bull Markets**: Can tolerate slightly higher risk
- **Bear Markets**: Reduce risk parameters by 20-30%
- **High Volatility**: Activate circuit breakers
- **Low Liquidity**: Reduce position sizes

### 3. Regular Risk Reviews
- **Daily**: Check drawdown and daily P&L
- **Weekly**: Review strategy performance and correlations
- **Monthly**: Analyze risk-adjusted returns and adjust parameters
- **Quarterly**: Full risk system audit and optimization

## 🆘 Emergency Procedures

### Immediate Risk Response
```bash
# Emergency stop all trading
curl -X POST http://localhost:8501/api/emergency/stop-all

# Close all positions immediately
curl -X POST http://localhost:8501/api/positions/close-all

# Reduce all position sizes by 50%
curl -X POST http://localhost:8501/api/risk/reduce-exposure \
  -d '{"reduction_factor": 0.5}'
```

### Risk Recovery Protocol
1. **Stop all new positions**
2. **Assess current losses**
3. **Close losing positions** (if strategy is broken)
4. **Reduce position sizes** for remaining positions
5. **Review and adjust** risk parameters
6. **Gradually resume** trading with lower risk

## 📚 Additional Resources

- **Risk Management Tutorial**: [docs/RISK_TUTORIAL.md](RISK_TUTORIAL.md)
- **API Risk Endpoints**: [docs/API_RISK.md](API_RISK.md)
- **Backtesting Risk Analysis**: [docs/BACKTEST_RISK.md](BACKTEST_RISK.md)
- **Emergency Procedures**: [docs/EMERGENCY_PROCEDURES.md](EMERGENCY_PROCEDURES.md)

---

## ⚠️ Important Disclaimers

1. **Past performance does not guarantee future results**
2. **Risk management cannot eliminate all losses**
3. **Always monitor your bot's performance actively**
4. **Never risk more than you can afford to lose**
5. **Cryptocurrency trading involves significant risk**
6. **This documentation is for educational purposes only**

---

*Last Updated: September 21, 2025*
*Version: 2.0 - Production Release*