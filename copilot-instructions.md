```markdown
# Copilot Instructions: Crypto Trading Bot Development

## Project Overview
This is a sophisticated algorithmic trading system for cryptocurrency perpetual swaps on Bybit. The bot emphasizes statistical rigor, dynamic risk management, and automated strategy validation with a toggleable aggressive mode for rapid growth of small accounts.

## Architecture Philosophy
- **Modular Design:** Each component (data, strategies, risk, execution) should be independent and replaceable
- **Event-Driven Core:** The main trading loop should be state-based with clear transitions
- **Database-Centric:** All actions, decisions, and results must be persisted for audit trails
- **Configurable Behavior:** All parameters should be configurable without code changes
- **Statistical First:** Every decision must be backed by statistical evidence

## Code Style & Quality Guidelines

### General Principles
- Use Python 3.11+ features (pattern matching, typing extensions, zoneinfo)
- Follow PEP 8 with Black formatting (line length: 100)
- Type hints for all function signatures and major variables
- Descriptive variable names (avoid abbreviations unless widely accepted)
- Docstrings for all classes, methods, and functions using Google style
- Modular architecture with single responsibility principles

### Specific Patterns
```python
# Use dataclasses for configuration objects
@dataclass
class RiskParameters:
    portfolio_drawdown_limit: float
    strategy_drawdown_limit: float
    sharpe_ratio_min: float
    # ...

# Use context managers for resource handling
class DatabaseSession:
    def __enter__(self):
        return self.session
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

# Use abstract base classes for strategy interface
class TradingStrategy(ABC):
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        pass
```

### Error Handling
- Use custom exceptions for domain-specific errors (`RiskLimitExceededError`, `StrategyValidationError`)
- Implement comprehensive logging with structured JSON format
- Use circuit breakers for exchange connectivity issues
- Always include context in error messages

## Database Schema & Models

### Core Tables
```python
# Key models for trading operations
class Trade(Base):
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    exchange = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # 'buy' or 'sell'
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float, nullable=False)
    fee_currency = Column(String, nullable=False)
    strategy_id = Column(String, nullable=False)
    
    # Tax tracking fields (Australia specific)
    cost_base_aud = Column(Float)
    proceeds_aud = Column(Float)
    is_cgt_event = Column(Boolean, default=False)
    aud_conversion_rate = Column(Float)

class StrategyPerformance(Base):
    strategy_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    equity = Column(Float)
    drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    mode = Column(String)  # 'conservative' or 'aggressive'
    risk_parameters = Column(JSON)  # Snapshot of parameters used
```

## Critical Implementation Details

### Data Sanitization (src/bot/data/sanitizer.py)
- Implement checks for: missing data, outliers, volume anomalies, constant prices
- Compare with secondary data source (e.g., Binance) for validation
- Use interpolation for small gaps (< 3 candles), exclude periods with large gaps
- Always work with UTC timestamps and ensure proper timezone handling

### Walk-Forward Optimization (src/bot/backtest/walkforward.py)
- Use expanding window approach for maximum data utilization
- Implement warm_start for efficient parameter optimization
- Store all OOS results to build continuous equity curve
- Include transaction costs and slippage in all calculations

### Machine Learning Implementation (src/bot/ml_engine/)
- Always use purged cross-validation to avoid lookahead bias
- Feature engineering must use only lagged values (min 1-period lag)
- Validate models with financial metrics (Sharpe, Calmar, Sortino) not just accuracy
- Implement feature importance analysis for model interpretability

### Dynamic Risk Management (src/bot/risk/dynamic_risk.py)
```python
def calculate_risk_parameters(self, current_balance: float) -> Dict[str, float]:
    """
    Calculate dynamic risk parameters based on current balance and mode.
    
    Uses exponential decay for risk reduction between threshold boundaries.
    """
    if self.mode == 'conservative':
        return self.conservative_params
    
    # Calculate risk scaling factor
    if current_balance <= self.low_threshold:
        scale = 1.0  # Maximum risk
    elif current_balance >= self.high_threshold:
        scale = 0.0  # Minimum risk
    else:
        # Exponential decay between thresholds
        normalized = (current_balance - self.low_threshold) / (self.high_threshold - self.low_threshold)
        scale = math.exp(-2.5 * normalized)  # Adjust decay rate as needed
    
    # Interpolate parameters
    params = {}
    for key in self.aggressive_base_params:
        if key.endswith('_limit') or key.endswith('_ratio'):
            conservative_val = self.conservative_params[key]
            aggressive_val = self.aggressive_base_params[key]
            params[key] = conservative_val + (aggressive_val - conservative_val) * scale
    
    return params
```

### Tax Calculation (Australia Specific) (src/bot/tax/cgt_calculator.py)
- Use FIFO method for CGT calculations (required by ATO)
- Integrate with RBA API for historical AUD/USD rates
- Calculate CGT events on position closure
- Generate comprehensive reports for financial year
- Track separate performance for aggressive vs conservative modes

## Configuration Management

### config/config.yaml Structure
```yaml
trading:
  mode: aggressive
  base_balance: 1000
  max_risk_ratio: 0.02
  min_risk_ratio: 0.005
  balance_thresholds:
    low: 1000
    high: 10000
  risk_decay: exponential

exchange:
  name: bybit
  sandbox: true
  timeframe: 1h
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT]

database:
  dialect: postgresql
  host: localhost
  port: 5432
  name: trading_bot
```

## Testing Requirements

### Unit Tests
- Test all utility functions and data processing
- Verify statistical calculations (Sharpe ratio, drawdown, etc.)
- Test database models and queries
- Verify configuration loading and validation

### Integration Tests
- Test strategy validation pipeline
- Verify exchange API interactions (use sandbox)
- Test risk management rules enforcement
- Validate tax calculation logic

### Performance Tests
- Benchmark backtesting engine speed
- Test ML model training performance
- Verify database query performance
- Stress test under high volatility scenarios

## Deployment & Monitoring

### Docker Configuration
- Use multi-stage builds for optimized image size
- Include health checks in Docker configuration
- Set up proper volume mounting for database persistence
- Configure resource limits (CPU, memory)

### Monitoring
- Implement structured JSON logging
- Create performance dashboards with Streamlit
- Set up alerting for critical events
- Monitor system health and resource usage

## Important Development Notes

### Strategy Development
1. **Always validate** with multiple testing methodologies (WFO, permutation, CSCV)
2. **Start simple** with technical strategies before implementing ML
3. **Test thoroughly** in paper trading before live deployment
4. **Monitor continuously** for regime changes and performance decay

### Risk Management
1. **Never override** risk limits programmatically
2. **Always preserve** capital as the primary objective
3. **Implement circuit breakers** for extreme market conditions
4. **Maintain audit trails** of all risk decisions

### Tax Compliance
1. **Record everything** required for Australian tax reporting
2. **Use official rates** from RBA for AUD conversions
3. **Generate regular reports** for compliance tracking
4. **Separate tracking** for different trading modes

## Common Patterns & Anti-Patterns

### Do This:
```python
# Use context managers for database sessions
with DatabaseSession() as session:
    result = session.query(Trade).filter_by(strategy_id=strategy_id).all()

# Use type hints and validation
def calculate_position_size(risk_amount: float, volatility: float) -> float:
    if risk_amount <= 0:
        raise ValueError("Risk amount must be positive")
    return risk_amount / volatility

# Use configuration objects instead of global variables
def init_risk_engine(config: RiskConfig) -> RiskEngine:
    return RiskEngine(config)
```

### Avoid This:
```python
# Don't use global state
global current_balance  # ❌ Bad

# Don't ignore errors
try:
    place_order(order)
except Exception:
    pass  # ❌ Very bad

# Don't hardcode parameters
position_size = balance * 0.02  # ❌ Should be configurable
```

## Performance Optimization Guidelines

1. **Vectorize operations** with pandas/numpy instead of loops
2. **Use database indexing** on frequently queried columns
3. **Implement caching** for expensive calculations
4. **Use appropriate data types** (e.g., integer timestamps)
5. **Batch database operations** instead of individual commits

This documentation should be updated regularly as the project evolves. Always refer to these guidelines when implementing new features or modifying existing code.
```