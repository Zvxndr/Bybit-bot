# Bybit Trading Bot - API Documentation

## Table of Contents
1. [Trading Engine API](#trading-engine-api)
2. [Risk Management API](#risk-management-api)
3. [Exchange Client API](#exchange-client-api)
4. [Strategy Framework API](#strategy-framework-api)
5. [Configuration API](#configuration-api)
6. [Data Management API](#data-management-api)
7. [Monitoring API](#monitoring-api)

## Trading Engine API

### Overview
The Trading Engine is the core component that orchestrates all trading activities. It processes signals, manages trades, and maintains system state.

### Class: `TradingEngine`

#### Constructor
```python
def __init__(
    self,
    bybit_client: BybitClient,
    data_manager: DataManager,
    risk_manager: UnifiedRiskManager,
    config_manager: ConfigurationManager
)
```

**Parameters:**
- `bybit_client`: Instance of BybitClient for exchange operations
- `data_manager`: Instance of DataManager for market data
- `risk_manager`: Instance of UnifiedRiskManager for risk assessment
- `config_manager`: Instance of ConfigurationManager for configuration

#### Methods

##### `async start() -> None`
Starts the trading engine and initializes all components.

**Returns:** None

**Raises:**
- `RuntimeError`: If engine is already running
- `ConnectionError`: If unable to connect to exchange

**Example:**
```python
engine = TradingEngine(bybit_client, data_manager, risk_manager, config_manager)
await engine.start()
```

##### `async stop() -> None`
Stops the trading engine and cleans up resources.

**Returns:** None

**Example:**
```python
await engine.stop()
```

##### `async pause() -> None`
Pauses the trading engine, preventing new trade processing.

**Returns:** None

##### `async resume() -> None`
Resumes the trading engine after being paused.

**Returns:** None

##### `async emergency_stop() -> None`
Immediately stops all trading and cancels open orders.

**Returns:** None

**Example:**
```python
await engine.emergency_stop()
```

##### `async process_trade_signal(signal: TradeSignal) -> TradeExecution`
Processes a trading signal and executes the trade if approved by risk management.

**Parameters:**
- `signal`: TradeSignal object containing trade details

**Returns:** TradeExecution object with execution details

**Example:**
```python
signal = TradeSignal(
    strategy_id="ma_crossover",
    symbol="BTCUSDT",
    action="BUY",
    quantity=Decimal('0.1'),
    price=Decimal('50000'),
    confidence=0.8,
    strength=SignalStrength.STRONG,
    timestamp=datetime.now()
)

execution = await engine.process_trade_signal(signal)
print(f"Trade status: {execution.status}")
```

##### `async cancel_trade(execution_id: str) -> bool`
Cancels an active trade by execution ID.

**Parameters:**
- `execution_id`: Unique identifier for the trade execution

**Returns:** True if cancellation successful, False otherwise

**Example:**
```python
success = await engine.cancel_trade("execution-123")
if success:
    print("Trade cancelled successfully")
```

##### `get_active_trades() -> List[TradeExecution]`
Returns a list of all active trades.

**Returns:** List of TradeExecution objects

**Example:**
```python
active_trades = engine.get_active_trades()
for trade in active_trades:
    print(f"Symbol: {trade.signal.symbol}, Status: {trade.status}")
```

##### `get_trade_history(symbol: Optional[str] = None, limit: int = 100) -> List[TradeExecution]`
Returns trade history, optionally filtered by symbol.

**Parameters:**
- `symbol`: Optional symbol to filter by
- `limit`: Maximum number of trades to return

**Returns:** List of TradeExecution objects

**Example:**
```python
# Get all trade history
all_trades = engine.get_trade_history()

# Get Bitcoin trades only
btc_trades = engine.get_trade_history("BTCUSDT")
```

##### `get_engine_status() -> Dict[str, Any]`
Returns current engine status and metrics.

**Returns:** Dictionary containing engine status information

**Example:**
```python
status = engine.get_engine_status()
print(f"Engine state: {status['state']}")
print(f"Active trades: {status['active_trades_count']}")
```

### Data Classes

#### `TradeSignal`
```python
@dataclass
class TradeSignal:
    strategy_id: str
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: Decimal
    price: Decimal
    confidence: float  # 0.0 to 1.0
    strength: SignalStrength
    timestamp: datetime
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### `TradeExecution`
```python
@dataclass
class TradeExecution:
    execution_id: str
    signal: TradeSignal
    order_id: Optional[str]
    status: ExecutionStatus
    timestamp: datetime
    fill_price: Optional[Decimal] = None
    filled_quantity: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    rejection_reason: Optional[str] = None
```

#### Enums

##### `EngineState`
```python
class EngineState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOPPED = "emergency_stopped"
```

##### `ExecutionStatus`
```python
class ExecutionStatus(Enum):
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
```

## Risk Management API

### Class: `UnifiedRiskManager`

#### Constructor
```python
def __init__(self, config_manager: ConfigurationManager)
```

#### Methods

##### `async assess_trade_risk(symbol: str, side: str, size: Decimal, entry_price: Decimal, stop_loss: Optional[Decimal] = None, take_profit: Optional[Decimal] = None) -> TradeRiskAssessment`
Assesses the risk of a proposed trade.

**Parameters:**
- `symbol`: Trading symbol (e.g., "BTCUSDT")
- `side`: Trade direction ("BUY" or "SELL")
- `size`: Position size
- `entry_price`: Entry price for the trade
- `stop_loss`: Optional stop loss price
- `take_profit`: Optional take profit price

**Returns:** TradeRiskAssessment object

**Example:**
```python
assessment = await risk_manager.assess_trade_risk(
    symbol="BTCUSDT",
    side="BUY",
    size=Decimal('0.1'),
    entry_price=Decimal('50000'),
    stop_loss=Decimal('48000'),
    take_profit=Decimal('54000')
)

print(f"Risk level: {assessment.risk_level}")
print(f"Recommended action: {assessment.recommended_action}")
print(f"Position size recommendation: {assessment.position_size_recommendation}")
```

##### `async calculate_portfolio_metrics(positions: Dict[str, Any], portfolio_value: Decimal) -> RiskMetrics`
Calculates comprehensive portfolio risk metrics.

**Parameters:**
- `positions`: Dictionary of current positions
- `portfolio_value`: Total portfolio value

**Returns:** RiskMetrics object

**Example:**
```python
positions = {
    "BTCUSDT": {"value": 5000, "unrealized_pnl": 100},
    "ETHUSDT": {"value": 3000, "unrealized_pnl": -50}
}

metrics = await risk_manager.calculate_portfolio_metrics(
    positions, Decimal('10000')
)

print(f"Portfolio VaR (95%): {metrics.var_95}")
print(f"Current drawdown: {metrics.current_drawdown}")
print(f"Risk score: {metrics.risk_score}")
```

##### `async check_circuit_breakers(portfolio_value: Decimal) -> List[str]`
Checks if any circuit breakers should be triggered.

**Parameters:**
- `portfolio_value`: Current portfolio value

**Returns:** List of triggered circuit breaker messages

**Example:**
```python
breakers = await risk_manager.check_circuit_breakers(Decimal('8000'))
if breakers:
    print("Circuit breakers triggered:")
    for breaker in breakers:
        print(f"- {breaker}")
```

##### `get_risk_summary() -> Dict[str, Any]`
Returns a comprehensive risk summary.

**Returns:** Dictionary containing risk profile and current state

**Example:**
```python
summary = risk_manager.get_risk_summary()
print(f"Max portfolio risk: {summary['risk_profile']['max_portfolio_risk']}")
print(f"Emergency stop active: {summary['current_state']['emergency_stop_active']}")
```

### Data Classes

#### `TradeRiskAssessment`
```python
@dataclass
class TradeRiskAssessment:
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    risk_amount: Decimal
    risk_percentage: float
    position_size_recommendation: Decimal
    risk_reward_ratio: float
    probability_of_success: float
    expected_value: float
    risk_level: RiskLevel
    recommended_action: RiskAction
    risk_factors: List[str]
```

#### `RiskMetrics`
```python
@dataclass
class RiskMetrics:
    portfolio_value: Decimal
    total_exposure: Decimal
    leverage: float
    var_95: float
    expected_shortfall: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    risk_score: float
    risk_level: RiskLevel
```

## Exchange Client API

### Class: `BybitClient`

#### Constructor
```python
def __init__(self, config_manager: ConfigurationManager)
```

#### Methods

##### `async place_order(symbol: str, side: OrderSide, order_type: OrderType, qty: Decimal, price: Optional[Decimal] = None, stop_loss: Optional[Decimal] = None, take_profit: Optional[Decimal] = None, time_in_force: TimeInForce = TimeInForce.GTC) -> str`
Places an order on the exchange.

**Parameters:**
- `symbol`: Trading symbol
- `side`: Order side (BUY/SELL)
- `order_type`: Order type (MARKET/LIMIT)
- `qty`: Order quantity
- `price`: Order price (required for limit orders)
- `stop_loss`: Optional stop loss price
- `take_profit`: Optional take profit price
- `time_in_force`: Time in force setting

**Returns:** Order ID string

**Example:**
```python
order_id = await client.place_order(
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    qty=Decimal('0.1'),
    price=Decimal('50000'),
    stop_loss=Decimal('48000'),
    take_profit=Decimal('54000')
)
print(f"Order placed: {order_id}")
```

##### `async cancel_order(symbol: str, order_id: str) -> bool`
Cancels an existing order.

**Parameters:**
- `symbol`: Trading symbol
- `order_id`: Order ID to cancel

**Returns:** True if successful

**Example:**
```python
success = await client.cancel_order("BTCUSDT", "order-123")
if success:
    print("Order cancelled successfully")
```

##### `async get_order_status(symbol: str, order_id: str) -> Dict[str, Any]`
Gets the status of a specific order.

**Parameters:**
- `symbol`: Trading symbol
- `order_id`: Order ID to check

**Returns:** Dictionary with order details

**Example:**
```python
order = await client.get_order_status("BTCUSDT", "order-123")
print(f"Order status: {order['orderStatus']}")
print(f"Filled quantity: {order['cumExecQty']}")
```

##### `async get_positions(symbol: Optional[str] = None) -> List[Dict[str, Any]]`
Gets current positions.

**Parameters:**
- `symbol`: Optional symbol filter

**Returns:** List of position dictionaries

**Example:**
```python
# Get all positions
all_positions = await client.get_positions()

# Get Bitcoin positions only
btc_positions = await client.get_positions("BTCUSDT")

for position in all_positions:
    print(f"Symbol: {position['symbol']}, Size: {position['size']}")
```

##### `async get_wallet_balance(coin: Optional[str] = None) -> Dict[str, Dict[str, Decimal]]`
Gets wallet balance information.

**Parameters:**
- `coin`: Optional coin filter

**Returns:** Dictionary of balances by coin

**Example:**
```python
balances = await client.get_wallet_balance()
print(f"USDT balance: {balances['USDT']['wallet_balance']}")
print(f"Available USDT: {balances['USDT']['available_balance']}")
```

##### `async get_market_data(symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]`
Gets market ticker data.

**Parameters:**
- `symbol`: Optional symbol (returns all if None)

**Returns:** Market data dictionary or list

**Example:**
```python
# Get Bitcoin ticker
btc_ticker = await client.get_market_data("BTCUSDT")
print(f"Last price: {btc_ticker['lastPrice']}")

# Get all tickers
all_tickers = await client.get_market_data()
```

## Strategy Framework API

### Class: `BaseStrategy` (Abstract)

#### Constructor
```python
def __init__(
    self,
    strategy_id: str,
    name: str,
    data_manager: DataManager,
    risk_manager: UnifiedRiskManager,
    config_manager: ConfigurationManager
)
```

#### Abstract Methods

##### `async generate_signals(symbol: str, data: pd.DataFrame) -> List[TradingSignal]`
Generates trading signals based on market data.

**Parameters:**
- `symbol`: Trading symbol
- `data`: Market data DataFrame

**Returns:** List of TradingSignal objects

**Note:** Must be implemented by concrete strategy classes.

#### Concrete Methods

##### `async activate() -> None`
Activates the strategy.

##### `async deactivate() -> None`
Deactivates the strategy.

##### `add_symbol(symbol: str) -> None`
Adds a symbol to the strategy's watchlist.

##### `remove_symbol(symbol: str) -> None`
Removes a symbol from the strategy's watchlist.

##### `update_performance(symbol: str, pnl: Decimal, win: bool) -> None`
Updates strategy performance metrics.

**Example:**
```python
strategy.update_performance("BTCUSDT", Decimal('100'), True)
```

### Class: `MovingAverageCrossoverStrategy`

#### Constructor
```python
def __init__(
    self,
    strategy_id: str,
    name: str,
    data_manager: DataManager,
    risk_manager: UnifiedRiskManager,
    config_manager: ConfigurationManager,
    fast_period: int = 10,
    slow_period: int = 50
)
```

**Additional Parameters:**
- `fast_period`: Period for fast moving average
- `slow_period`: Period for slow moving average

**Example:**
```python
ma_strategy = MovingAverageCrossoverStrategy(
    strategy_id="ma_5_20",
    name="MA 5-20 Crossover",
    data_manager=data_manager,
    risk_manager=risk_manager,
    config_manager=config_manager,
    fast_period=5,
    slow_period=20
)

ma_strategy.add_symbol("BTCUSDT")
await ma_strategy.activate()
```

### Class: `StrategyManager`

#### Constructor
```python
def __init__(
    self,
    data_manager: DataManager,
    risk_manager: UnifiedRiskManager,
    config_manager: ConfigurationManager
)
```

#### Methods

##### `add_strategy(strategy: BaseStrategy) -> None`
Adds a strategy to the manager.

##### `remove_strategy(strategy_id: str) -> None`
Removes a strategy by ID.

##### `get_strategy(strategy_id: str) -> Optional[BaseStrategy]`
Gets a strategy by ID.

##### `async generate_all_signals(symbols: List[str]) -> List[TradingSignal]`
Generates signals from all active strategies.

**Parameters:**
- `symbols`: List of symbols to analyze

**Returns:** List of all generated signals

**Example:**
```python
manager = StrategyManager(data_manager, risk_manager, config_manager)
manager.add_strategy(ma_strategy)

await manager.activate_all_strategies()
signals = await manager.generate_all_signals(["BTCUSDT", "ETHUSDT"])

for signal in signals:
    print(f"Strategy: {signal.strategy_id}, Symbol: {signal.symbol}, Action: {signal.action}")
```

## Configuration API

### Class: `ConfigurationManager`

#### Constructor
```python
def __init__(self, config_path: str = "config/default.yaml")
```

#### Methods

##### `load_config(config_path: str) -> None`
Loads configuration from file.

##### `get(key: str, default: Any = None) -> Any`
Gets a configuration value.

**Example:**
```python
config_manager = ConfigurationManager("config/production.yaml")

# Get trading configuration
max_position_size = config_manager.get("trading.max_position_size", 0.1)
enable_stop_loss = config_manager.get("trading.enable_stop_loss", True)

# Get risk configuration
max_portfolio_risk = config_manager.get("risk.max_portfolio_risk", 0.02)
```

##### `set(key: str, value: Any) -> None`
Sets a configuration value.

##### `update_config(updates: Dict[str, Any]) -> None`
Updates multiple configuration values.

**Example:**
```python
config_manager.update_config({
    "trading.max_position_size": 0.05,
    "risk.max_daily_loss": 0.03
})
```

##### `validate_config() -> List[str]`
Validates current configuration.

**Returns:** List of validation errors (empty if valid)

## Error Handling

### Exception Classes

#### `BybitAPIError`
Base exception for Bybit API errors.

```python
class BybitAPIError(Exception):
    def __init__(self, message: str, code: int):
        self.message = message
        self.code = code
```

#### `RateLimitError`
Raised when API rate limits are exceeded.

#### `InsufficientBalanceError`
Raised when account has insufficient balance.

#### `OrderNotFoundError`
Raised when trying to operate on non-existent order.

### Error Handling Examples

```python
try:
    order_id = await client.place_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal('0.1')
    )
except InsufficientBalanceError:
    print("Insufficient balance to place order")
except RateLimitError:
    print("Rate limit exceeded, waiting...")
    await asyncio.sleep(1)
except BybitAPIError as e:
    print(f"API error: {e.message} (code: {e.code})")
```

## Best Practices

### Async/Await Usage
Always use async/await for API calls:

```python
# Correct
result = await client.get_positions()

# Incorrect
result = client.get_positions()  # This will return a coroutine
```

### Error Handling
Always handle potential exceptions:

```python
try:
    execution = await engine.process_trade_signal(signal)
    if execution.status == ExecutionStatus.FILLED:
        print("Trade executed successfully")
    elif execution.status == ExecutionStatus.REJECTED:
        print(f"Trade rejected: {execution.rejection_reason}")
except Exception as e:
    print(f"Error processing trade: {e}")
```

### Resource Management
Always clean up resources:

```python
try:
    await engine.start()
    # Trading operations...
finally:
    await engine.stop()
```

### Configuration Validation
Validate configuration before use:

```python
config_manager = ConfigurationManager("config/production.yaml")
errors = config_manager.validate_config()
if errors:
    print(f"Configuration errors: {errors}")
    sys.exit(1)
```