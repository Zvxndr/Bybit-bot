"""
Test Configuration and Utilities

This module provides common test configuration, fixtures, and utilities
for the comprehensive test suite of the trading bot system.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
import pandas as pd
import numpy as np

# Test data and configurations
TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
TEST_PORTFOLIO_VALUE = Decimal('10000')
TEST_PRICE_DATA = {
    "BTCUSDT": Decimal('50000'),
    "ETHUSDT": Decimal('3000'), 
    "ADAUSDT": Decimal('0.50')
}

class MockBybitClient:
    """Mock Bybit client for testing."""
    
    def __init__(self):
        self.connected = False
        self.orders = {}
        self.positions = {}
        self.balance = {"USDT": Decimal('10000')}
        self.next_order_id = 1
        
    async def connect(self):
        self.connected = True
        return True
        
    async def disconnect(self):
        self.connected = False
        
    async def place_order(self, **kwargs):
        if not self.connected:
            return None
            
        order_id = f"test_order_{self.next_order_id}"
        self.next_order_id += 1
        
        self.orders[order_id] = {
            "orderId": order_id,
            "symbol": kwargs.get("symbol"),
            "side": kwargs.get("side"),
            "orderType": kwargs.get("orderType"),
            "qty": kwargs.get("qty"),
            "price": kwargs.get("price"),
            "orderStatus": "Filled",
            "cumExecQty": kwargs.get("qty")
        }
        
        return {"orderId": order_id}
        
    async def cancel_order(self, **kwargs):
        order_id = kwargs.get("order_id")
        if order_id in self.orders:
            self.orders[order_id]["orderStatus"] = "Cancelled"
            return {"orderId": order_id}
        return None
        
    async def get_positions(self, **kwargs):
        return {
            "list": [
                {
                    "symbol": symbol,
                    "side": "Buy",
                    "size": "1.0",
                    "avgPrice": str(TEST_PRICE_DATA.get(symbol, 50000)),
                    "markPrice": str(TEST_PRICE_DATA.get(symbol, 50000)),
                    "unrealisedPnl": "100.0",
                    "leverage": "1",
                    "positionIM": "1000.0",
                    "liqPrice": ""
                }
                for symbol in TEST_SYMBOLS[:1]  # Only one position for testing
            ]
        }
        
    async def get_wallet_balance(self, account_type):
        return {
            "list": [
                {
                    "coin": [
                        {
                            "coin": "USDT",
                            "walletBalance": str(self.balance["USDT"]),
                            "availableToWithdraw": str(self.balance["USDT"] * Decimal('0.9'))
                        }
                    ]
                }
            ]
        }
        
    async def cancel_all_orders(self, **kwargs):
        cancelled_orders = []
        for order_id, order in self.orders.items():
            if order["orderStatus"] != "Cancelled":
                order["orderStatus"] = "Cancelled"
                cancelled_orders.append(order_id)
        return {"cancelled": cancelled_orders}


class MockDataManager:
    """Mock data manager for testing."""
    
    def __init__(self):
        self.price_data = TEST_PRICE_DATA.copy()
        
    async def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        if symbol in self.price_data:
            return {"close": float(self.price_data[symbol])}
        return None
        
    def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Generate mock historical data."""
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1H')
        base_price = float(self.price_data.get(symbol, 50000))
        
        # Generate realistic OHLCV data
        np.random.seed(42)  # For reproducible tests
        returns = np.random.normal(0, 0.02, limit)  # 2% volatility
        prices = base_price * np.exp(returns.cumsum())
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, limit)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, limit))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, limit))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, limit)
        })


class MockConfigurationManager:
    """Mock configuration manager for testing."""
    
    def __init__(self):
        self.config = {
            'risk_management': {
                'max_portfolio_risk': 0.02,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.15,
                'max_leverage': 3.0,
                'volatility_target': 0.15
            },
            'trading': {
                'max_position_size': 0.1,
                'max_open_orders': 10
            }
        }
    
    def get_config(self):
        return type('Config', (), {'risk_management': type('RiskConfig', (), self.config['risk_management'])})()
    
    def get(self, key: str, default: Any = None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


def create_mock_market_data(symbol: str, periods: int = 100, volatility: float = 0.02) -> pd.DataFrame:
    """Create realistic mock market data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
    base_price = float(TEST_PRICE_DATA.get(symbol, 50000))
    
    # Generate price movements
    np.random.seed(hash(symbol) % 2**32)  # Seed based on symbol for consistency
    returns = np.random.normal(0, volatility, periods)
    prices = base_price * np.exp(returns.cumsum())
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, periods)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum.reduce([data['high'], data['open'], data['close']])
    data['low'] = np.minimum.reduce([data['low'], data['open'], data['close']])
    
    return data


def create_test_trade_request(symbol: str = "BTCUSDT", side: str = "Buy", 
                            quantity: Decimal = Decimal('0.1')) -> Dict[str, Any]:
    """Create a test trade request."""
    return {
        "symbol": symbol,
        "side": side,
        "trade_type": "Market",
        "quantity": quantity,
        "price": TEST_PRICE_DATA.get(symbol, Decimal('50000')),
        "stop_loss": None,
        "take_profit": None,
        "custom_id": f"test_trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }


def create_test_position(symbol: str = "BTCUSDT", side: str = "Buy",
                        size: Decimal = Decimal('1.0')) -> Dict[str, Any]:
    """Create a test position."""
    price = TEST_PRICE_DATA.get(symbol, Decimal('50000'))
    return {
        "symbol": symbol,
        "side": side,
        "size": size,
        "entry_price": price,
        "mark_price": price,
        "unrealized_pnl": Decimal('100'),
        "leverage": 1.0,
        "margin": size * price * Decimal('0.1'),
        "liquidation_price": None,
        "entry_time": datetime.now() - timedelta(hours=2)
    }


# Pytest fixtures
@pytest.fixture
def mock_bybit_client():
    """Provide a mock Bybit client."""
    return MockBybitClient()


@pytest.fixture
def mock_data_manager():
    """Provide a mock data manager."""
    return MockDataManager()


@pytest.fixture
def mock_config_manager():
    """Provide a mock configuration manager."""
    return MockConfigurationManager()


@pytest.fixture
def sample_market_data():
    """Provide sample market data for testing."""
    return {
        symbol: create_mock_market_data(symbol)
        for symbol in TEST_SYMBOLS
    }


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Test decorators and utilities
def async_test(func):
    """Decorator for async test functions."""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


def assert_decimal_close(actual: Decimal, expected: Decimal, tolerance: Decimal = Decimal('0.01')):
    """Assert two Decimal values are close within tolerance."""
    assert abs(actual - expected) <= tolerance, f"Expected {expected}, got {actual} (tolerance: {tolerance})"


def assert_risk_metrics_valid(metrics):
    """Assert risk metrics are within valid ranges."""
    assert 0 <= metrics.risk_score <= 1, f"Risk score out of range: {metrics.risk_score}"
    assert metrics.leverage >= 0, f"Negative leverage: {metrics.leverage}"
    assert metrics.portfolio_value >= 0, f"Negative portfolio value: {metrics.portfolio_value}"
    assert -1 <= metrics.current_drawdown <= 0, f"Invalid drawdown: {metrics.current_drawdown}"


# Performance testing utilities
class PerformanceTimer:
    """Context manager for measuring performance."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"{self.name} took {duration:.4f} seconds")
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


# Test data generators
def generate_price_series(initial_price: float, periods: int, volatility: float = 0.02) -> pd.Series:
    """Generate a realistic price series for testing."""
    np.random.seed(42)
    returns = np.random.normal(0, volatility, periods)
    prices = initial_price * np.exp(returns.cumsum())
    return pd.Series(prices)


def generate_trade_history(num_trades: int = 50) -> list:
    """Generate synthetic trade history for testing."""
    trades = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(num_trades):
        # Simulate 60% win rate
        is_winner = np.random.random() < 0.6
        pnl = np.random.uniform(50, 200) if is_winner else np.random.uniform(-150, -25)
        
        trade = {
            "id": f"test_trade_{i}",
            "symbol": np.random.choice(TEST_SYMBOLS),
            "side": np.random.choice(["Buy", "Sell"]),
            "entry_time": base_time + timedelta(hours=i * 12),
            "exit_time": base_time + timedelta(hours=i * 12 + 4),
            "entry_price": np.random.uniform(45000, 55000),
            "exit_price": np.random.uniform(45000, 55000),
            "size": np.random.uniform(0.01, 0.1),
            "pnl": Decimal(str(pnl)),
            "reason": "Test Trade"
        }
        trades.append(trade)
    
    return trades


# Validation helpers
def validate_signal(signal):
    """Validate a trading signal structure."""
    required_fields = ['symbol', 'signal_type', 'strength', 'price', 'timestamp', 'strategy_id', 'confidence']
    for field in required_fields:
        assert hasattr(signal, field), f"Signal missing required field: {field}"
    
    assert 0 <= signal.confidence <= 1, f"Invalid confidence: {signal.confidence}"
    assert signal.price > 0, f"Invalid price: {signal.price}"


def validate_trade_result(result):
    """Validate a trade result structure."""
    required_fields = ['request', 'success', 'status', 'timestamp']
    for field in required_fields:
        assert hasattr(result, field), f"Trade result missing required field: {field}"
    
    if result.success:
        assert result.order_id is not None, "Successful trade should have order_id"
    else:
        assert result.error_message is not None, "Failed trade should have error_message"