"""
Unit Tests for Strategy Framework

This module contains comprehensive unit tests for the strategy framework
including BaseStrategy, StrategyManager, and MovingAverageCrossoverStrategy.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd
import numpy as np

from src.bot.strategies.strategy_framework import (
    BaseStrategy,
    StrategyManager,
    TradingSignal,
    StrategyPerformance,
    StrategyState,
    SignalStrength
)
from src.bot.strategies.ma_crossover_strategy import MovingAverageCrossoverStrategy
from tests.conftest import (
    MockBybitClient,
    MockDataManager,
    MockUnifiedRiskManager,
    MockConfigurationManager,
    create_test_market_data,
    create_test_trade_signal,
    async_test
)


class TestBaseStrategy:
    """Test suite for BaseStrategy abstract class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.data_manager = MockDataManager()
        self.risk_manager = MockUnifiedRiskManager()
        
        # Create a concrete implementation for testing
        class TestStrategy(BaseStrategy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            
            async def generate_signals(self, symbol: str, data: pd.DataFrame) -> list:
                """Test implementation of signal generation."""
                if len(data) < 2:
                    return []
                
                # Simple test logic: buy if price increased, sell if decreased
                current_price = data.iloc[-1]['close']
                previous_price = data.iloc[-2]['close']
                
                if current_price > previous_price * 1.01:  # 1% increase
                    return [TradingSignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        action="BUY",
                        quantity=Decimal('0.1'),
                        price=current_price,
                        confidence=0.7,
                        strength=SignalStrength.MEDIUM,
                        timestamp=datetime.now(),
                        metadata={"test": True}
                    )]
                elif current_price < previous_price * 0.99:  # 1% decrease
                    return [TradingSignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        action="SELL",
                        quantity=Decimal('0.1'),
                        price=current_price,
                        confidence=0.7,
                        strength=SignalStrength.MEDIUM,
                        timestamp=datetime.now(),
                        metadata={"test": True}
                    )]
                
                return []
        
        self.strategy = TestStrategy(
            strategy_id="test-strategy",
            name="Test Strategy",
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager
        )
    
    def test_base_strategy_initialization(self):
        """Test BaseStrategy initialization."""
        assert self.strategy.strategy_id == "test-strategy"
        assert self.strategy.name == "Test Strategy"
        assert self.strategy.data_manager == self.data_manager
        assert self.strategy.risk_manager == self.risk_manager
        assert self.strategy.config_manager == self.config_manager
        
        assert self.strategy.state == StrategyState.INACTIVE
        assert self.strategy.enabled is True
        assert len(self.strategy.symbols) == 0
        assert len(self.strategy.generated_signals) == 0
        assert self.strategy.performance is not None
        assert isinstance(self.strategy.performance, StrategyPerformance)
    
    @async_test
    async def test_strategy_activation(self):
        """Test strategy activation."""
        await self.strategy.activate()
        assert self.strategy.state == StrategyState.ACTIVE
    
    @async_test
    async def test_strategy_deactivation(self):
        """Test strategy deactivation."""
        await self.strategy.activate()
        await self.strategy.deactivate()
        assert self.strategy.state == StrategyState.INACTIVE
    
    def test_strategy_enable_disable(self):
        """Test strategy enable/disable."""
        self.strategy.disable()
        assert self.strategy.enabled is False
        
        self.strategy.enable()
        assert self.strategy.enabled is True
    
    def test_add_remove_symbols(self):
        """Test adding and removing symbols."""
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        for symbol in symbols:
            self.strategy.add_symbol(symbol)
        
        assert len(self.strategy.symbols) == 3
        assert all(symbol in self.strategy.symbols for symbol in symbols)
        
        self.strategy.remove_symbol("ETHUSDT")
        assert len(self.strategy.symbols) == 2
        assert "ETHUSDT" not in self.strategy.symbols
    
    @async_test
    async def test_signal_generation(self):
        """Test signal generation."""
        symbol = "BTCUSDT"
        self.strategy.add_symbol(symbol)
        
        # Create test data with price increase
        data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=2), datetime.now() - timedelta(minutes=1)],
            'open': [49000, 49500],
            'high': [49100, 50100],
            'low': [48900, 49400],
            'close': [49050, 50000],  # ~2% increase
            'volume': [100, 120]
        })
        
        await self.strategy.activate()
        signals = await self.strategy.generate_signals(symbol, data)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.symbol == symbol
        assert signal.action == "BUY"
        assert signal.strategy_id == self.strategy.strategy_id
        assert signal.confidence == 0.7
        assert signal.strength == SignalStrength.MEDIUM
    
    @async_test
    async def test_signal_generation_no_signals(self):
        """Test signal generation with no clear trend."""
        symbol = "ETHUSDT"
        self.strategy.add_symbol(symbol)
        
        # Create test data with minimal price change
        data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=2), datetime.now() - timedelta(minutes=1)],
            'open': [3000, 3005],
            'high': [3010, 3015],
            'low': [2995, 3000],
            'close': [3000, 3005],  # 0.17% increase (below threshold)
            'volume': [500, 520]
        })
        
        await self.strategy.activate()
        signals = await self.strategy.generate_signals(symbol, data)
        
        assert len(signals) == 0
    
    @async_test
    async def test_signal_generation_insufficient_data(self):
        """Test signal generation with insufficient data."""
        symbol = "ADAUSDT"
        self.strategy.add_symbol(symbol)
        
        # Create test data with only one row
        data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [0.45],
            'high': [0.46],
            'low': [0.44],
            'close': [0.45],
            'volume': [1000]
        })
        
        await self.strategy.activate()
        signals = await self.strategy.generate_signals(symbol, data)
        
        assert len(signals) == 0
    
    def test_update_performance(self):
        """Test performance update."""
        initial_trades = self.strategy.performance.total_trades
        initial_pnl = self.strategy.performance.total_pnl
        
        # Update with a winning trade
        self.strategy.update_performance(
            symbol="BTCUSDT",
            pnl=Decimal('100'),
            win=True
        )
        
        assert self.strategy.performance.total_trades == initial_trades + 1
        assert self.strategy.performance.winning_trades == 1
        assert self.strategy.performance.total_pnl == initial_pnl + Decimal('100')
        assert self.strategy.performance.win_rate == 1.0
    
    def test_update_performance_multiple_trades(self):
        """Test performance update with multiple trades."""
        # Add multiple trades
        trades = [
            (Decimal('100'), True),
            (Decimal('-50'), False),
            (Decimal('75'), True),
            (Decimal('-25'), False),
            (Decimal('150'), True)
        ]
        
        for pnl, win in trades:
            self.strategy.update_performance(
                symbol="BTCUSDT",
                pnl=pnl,
                win=win
            )
        
        assert self.strategy.performance.total_trades == 5
        assert self.strategy.performance.winning_trades == 3
        assert self.strategy.performance.losing_trades == 2
        assert self.strategy.performance.win_rate == 0.6
        assert self.strategy.performance.total_pnl == Decimal('250')
    
    def test_get_strategy_info(self):
        """Test getting strategy information."""
        info = self.strategy.get_strategy_info()
        
        assert "strategy_id" in info
        assert "name" in info
        assert "state" in info
        assert "enabled" in info
        assert "symbols" in info
        assert "performance" in info
        assert "signals_generated" in info
        
        assert info["strategy_id"] == "test-strategy"
        assert info["name"] == "Test Strategy"
        assert info["state"] == StrategyState.INACTIVE


class TestMovingAverageCrossoverStrategy:
    """Test suite for MovingAverageCrossoverStrategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.data_manager = MockDataManager()
        self.risk_manager = MockUnifiedRiskManager()
        
        self.strategy = MovingAverageCrossoverStrategy(
            strategy_id="ma-crossover-test",
            name="MA Crossover Test",
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager,
            fast_period=5,
            slow_period=20
        )
    
    def test_ma_strategy_initialization(self):
        """Test MA crossover strategy initialization."""
        assert self.strategy.fast_period == 5
        assert self.strategy.slow_period == 20
        assert self.strategy.strategy_id == "ma-crossover-test"
    
    @async_test
    async def test_ma_signal_generation_bullish_crossover(self):
        """Test bullish crossover signal generation."""
        symbol = "BTCUSDT"
        self.strategy.add_symbol(symbol)
        
        # Create data with bullish crossover (fast MA crosses above slow MA)
        dates = pd.date_range(start='2023-01-01', periods=25, freq='1H')
        
        # Create price data that will result in bullish crossover
        prices = []
        base_price = 45000
        
        # First 20 periods: declining trend (slow MA higher)
        for i in range(20):
            prices.append(base_price - i * 100)
        
        # Last 5 periods: strong uptrend (fast MA crosses above slow MA)
        for i in range(5):
            prices.append(base_price - 2000 + i * 500)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 50 for p in prices],
            'low': [p - 50 for p in prices],
            'close': prices,
            'volume': [100] * 25
        })
        
        await self.strategy.activate()
        signals = await self.strategy.generate_signals(symbol, data)
        
        # Should generate a buy signal due to bullish crossover
        assert len(signals) > 0
        signal = signals[0]
        assert signal.action == "BUY"
        assert signal.symbol == symbol
        assert signal.strength in [SignalStrength.MEDIUM, SignalStrength.STRONG]
    
    @async_test
    async def test_ma_signal_generation_bearish_crossover(self):
        """Test bearish crossover signal generation."""
        symbol = "ETHUSDT"
        self.strategy.add_symbol(symbol)
        
        # Create data with bearish crossover (fast MA crosses below slow MA)
        dates = pd.date_range(start='2023-01-01', periods=25, freq='1H')
        
        # Create price data that will result in bearish crossover
        prices = []
        base_price = 3000
        
        # First 20 periods: rising trend (fast MA higher)
        for i in range(20):
            prices.append(base_price + i * 10)
        
        # Last 5 periods: strong downtrend (fast MA crosses below slow MA)
        for i in range(5):
            prices.append(base_price + 200 - i * 50)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 5 for p in prices],
            'low': [p - 5 for p in prices],
            'close': prices,
            'volume': [500] * 25
        })
        
        await self.strategy.activate()
        signals = await self.strategy.generate_signals(symbol, data)
        
        # Should generate a sell signal due to bearish crossover
        assert len(signals) > 0
        signal = signals[0]
        assert signal.action == "SELL"
        assert signal.symbol == symbol
        assert signal.strength in [SignalStrength.MEDIUM, SignalStrength.STRONG]
    
    @async_test
    async def test_ma_signal_generation_no_crossover(self):
        """Test no signal generation when no crossover occurs."""
        symbol = "ADAUSDT"
        self.strategy.add_symbol(symbol)
        
        # Create data with consistent trend (no crossover)
        dates = pd.date_range(start='2023-01-01', periods=25, freq='1H')
        prices = [0.45 + i * 0.001 for i in range(25)]  # Gentle uptrend
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 0.001 for p in prices],
            'low': [p - 0.001 for p in prices],
            'close': prices,
            'volume': [1000] * 25
        })
        
        await self.strategy.activate()
        signals = await self.strategy.generate_signals(symbol, data)
        
        # Should not generate any signals (no crossover)
        assert len(signals) == 0
    
    @async_test
    async def test_ma_signal_generation_insufficient_data(self):
        """Test signal generation with insufficient data."""
        symbol = "DOTUSDT"
        self.strategy.add_symbol(symbol)
        
        # Create data with fewer periods than required for slow MA
        dates = pd.date_range(start='2023-01-01', periods=15, freq='1H')
        prices = [10.0 + i * 0.1 for i in range(15)]
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 0.05 for p in prices],
            'low': [p - 0.05 for p in prices],
            'close': prices,
            'volume': [200] * 15
        })
        
        await self.strategy.activate()
        signals = await self.strategy.generate_signals(symbol, data)
        
        # Should not generate signals due to insufficient data
        assert len(signals) == 0
    
    def test_calculate_moving_averages(self):
        """Test moving average calculations."""
        # Create test price data
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        
        # Calculate 3-period and 5-period moving averages
        fast_ma = prices.rolling(window=3).mean()
        slow_ma = prices.rolling(window=5).mean()
        
        # Verify calculations
        assert pd.isna(fast_ma.iloc[0])  # First two values should be NaN
        assert pd.isna(fast_ma.iloc[1])
        assert fast_ma.iloc[2] == 101.0  # (100 + 102 + 101) / 3
        
        assert pd.isna(slow_ma.iloc[3])  # First four values should be NaN
        assert slow_ma.iloc[4] == 102.2  # (100 + 102 + 101 + 103 + 105) / 5
    
    def test_detect_crossover(self):
        """Test crossover detection logic."""
        # Test data for crossover detection
        fast_ma_prev, fast_ma_curr = 100.0, 102.0
        slow_ma_prev, slow_ma_curr = 101.0, 101.5
        
        # Bullish crossover: fast was below, now above
        bullish = (fast_ma_prev <= slow_ma_prev and fast_ma_curr > slow_ma_curr)
        assert bullish is True
        
        # Test bearish crossover
        fast_ma_prev, fast_ma_curr = 102.0, 100.0
        slow_ma_prev, slow_ma_curr = 101.0, 101.5
        
        bearish = (fast_ma_prev >= slow_ma_prev and fast_ma_curr < slow_ma_curr)
        assert bearish is True
        
        # Test no crossover
        fast_ma_prev, fast_ma_curr = 100.0, 100.5
        slow_ma_prev, slow_ma_curr = 101.0, 101.5
        
        no_crossover = not (
            (fast_ma_prev <= slow_ma_prev and fast_ma_curr > slow_ma_curr) or
            (fast_ma_prev >= slow_ma_prev and fast_ma_curr < slow_ma_curr)
        )
        assert no_crossover is True


class TestStrategyManager:
    """Test suite for StrategyManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.data_manager = MockDataManager()
        self.risk_manager = MockUnifiedRiskManager()
        
        self.strategy_manager = StrategyManager(
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager
        )
        
        # Create test strategies
        self.test_strategy1 = MovingAverageCrossoverStrategy(
            strategy_id="ma-5-20",
            name="MA 5-20 Crossover",
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager,
            fast_period=5,
            slow_period=20
        )
        
        self.test_strategy2 = MovingAverageCrossoverStrategy(
            strategy_id="ma-10-50",
            name="MA 10-50 Crossover",
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager,
            fast_period=10,
            slow_period=50
        )
    
    def test_strategy_manager_initialization(self):
        """Test StrategyManager initialization."""
        assert len(self.strategy_manager.strategies) == 0
        assert self.strategy_manager.data_manager == self.data_manager
        assert self.strategy_manager.risk_manager == self.risk_manager
        assert self.strategy_manager.config_manager == self.config_manager
    
    def test_add_strategy(self):
        """Test adding strategies to manager."""
        self.strategy_manager.add_strategy(self.test_strategy1)
        self.strategy_manager.add_strategy(self.test_strategy2)
        
        assert len(self.strategy_manager.strategies) == 2
        assert "ma-5-20" in self.strategy_manager.strategies
        assert "ma-10-50" in self.strategy_manager.strategies
    
    def test_remove_strategy(self):
        """Test removing strategies from manager."""
        self.strategy_manager.add_strategy(self.test_strategy1)
        self.strategy_manager.add_strategy(self.test_strategy2)
        
        self.strategy_manager.remove_strategy("ma-5-20")
        
        assert len(self.strategy_manager.strategies) == 1
        assert "ma-5-20" not in self.strategy_manager.strategies
        assert "ma-10-50" in self.strategy_manager.strategies
    
    def test_get_strategy(self):
        """Test getting specific strategy."""
        self.strategy_manager.add_strategy(self.test_strategy1)
        
        strategy = self.strategy_manager.get_strategy("ma-5-20")
        assert strategy == self.test_strategy1
        
        # Test non-existent strategy
        strategy = self.strategy_manager.get_strategy("non-existent")
        assert strategy is None
    
    def test_get_all_strategies(self):
        """Test getting all strategies."""
        self.strategy_manager.add_strategy(self.test_strategy1)
        self.strategy_manager.add_strategy(self.test_strategy2)
        
        strategies = self.strategy_manager.get_all_strategies()
        
        assert len(strategies) == 2
        assert self.test_strategy1 in strategies
        assert self.test_strategy2 in strategies
    
    def test_get_active_strategies(self):
        """Test getting only active strategies."""
        self.strategy_manager.add_strategy(self.test_strategy1)
        self.strategy_manager.add_strategy(self.test_strategy2)
        
        # Activate only one strategy
        asyncio.run(self.test_strategy1.activate())
        
        active_strategies = self.strategy_manager.get_active_strategies()
        
        assert len(active_strategies) == 1
        assert self.test_strategy1 in active_strategies
        assert self.test_strategy2 not in active_strategies
    
    @async_test
    async def test_activate_all_strategies(self):
        """Test activating all strategies."""
        self.strategy_manager.add_strategy(self.test_strategy1)
        self.strategy_manager.add_strategy(self.test_strategy2)
        
        await self.strategy_manager.activate_all_strategies()
        
        assert self.test_strategy1.state == StrategyState.ACTIVE
        assert self.test_strategy2.state == StrategyState.ACTIVE
    
    @async_test
    async def test_deactivate_all_strategies(self):
        """Test deactivating all strategies."""
        self.strategy_manager.add_strategy(self.test_strategy1)
        self.strategy_manager.add_strategy(self.test_strategy2)
        
        # First activate them
        await self.strategy_manager.activate_all_strategies()
        
        # Then deactivate
        await self.strategy_manager.deactivate_all_strategies()
        
        assert self.test_strategy1.state == StrategyState.INACTIVE
        assert self.test_strategy2.state == StrategyState.INACTIVE
    
    @async_test
    async def test_generate_all_signals(self):
        """Test generating signals from all active strategies."""
        self.strategy_manager.add_strategy(self.test_strategy1)
        self.strategy_manager.add_strategy(self.test_strategy2)
        
        # Add symbols to strategies
        self.test_strategy1.add_symbol("BTCUSDT")
        self.test_strategy2.add_symbol("BTCUSDT")
        
        # Activate strategies
        await self.test_strategy1.activate()
        await self.test_strategy2.activate()
        
        # Mock data manager to return test data
        test_data = create_test_market_data("BTCUSDT", 60)  # 60 periods
        self.data_manager.get_market_data.return_value = test_data
        
        # Mock strategies to return test signals
        test_signal1 = create_test_trade_signal(symbol="BTCUSDT", action="BUY")
        test_signal2 = create_test_trade_signal(symbol="BTCUSDT", action="SELL")
        
        with patch.object(self.test_strategy1, 'generate_signals', return_value=[test_signal1]):
            with patch.object(self.test_strategy2, 'generate_signals', return_value=[test_signal2]):
                
                all_signals = await self.strategy_manager.generate_all_signals(["BTCUSDT"])
                
                assert len(all_signals) == 2
                assert test_signal1 in all_signals
                assert test_signal2 in all_signals
    
    def test_get_strategy_performance_summary(self):
        """Test getting strategy performance summary."""
        self.strategy_manager.add_strategy(self.test_strategy1)
        self.strategy_manager.add_strategy(self.test_strategy2)
        
        # Add some performance data
        self.test_strategy1.update_performance("BTCUSDT", Decimal('100'), True)
        self.test_strategy1.update_performance("BTCUSDT", Decimal('-50'), False)
        
        self.test_strategy2.update_performance("ETHUSDT", Decimal('75'), True)
        
        summary = self.strategy_manager.get_strategy_performance_summary()
        
        assert len(summary) == 2
        assert "ma-5-20" in summary
        assert "ma-10-50" in summary
        
        # Check strategy 1 performance
        perf1 = summary["ma-5-20"]
        assert perf1["total_trades"] == 2
        assert perf1["winning_trades"] == 1
        assert perf1["total_pnl"] == Decimal('50')
        
        # Check strategy 2 performance
        perf2 = summary["ma-10-50"]
        assert perf2["total_trades"] == 1
        assert perf2["winning_trades"] == 1
        assert perf2["total_pnl"] == Decimal('75')


class TestTradingSignal:
    """Test suite for TradingSignal dataclass."""
    
    def test_trading_signal_creation(self):
        """Test TradingSignal creation."""
        signal = TradingSignal(
            strategy_id="test-strategy",
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('50000'),
            confidence=0.8,
            strength=SignalStrength.STRONG,
            timestamp=datetime.now(),
            metadata={"test_key": "test_value"}
        )
        
        assert signal.strategy_id == "test-strategy"
        assert signal.symbol == "BTCUSDT"
        assert signal.action == "BUY"
        assert signal.quantity == Decimal('0.1')
        assert signal.price == Decimal('50000')
        assert signal.confidence == 0.8
        assert signal.strength == SignalStrength.STRONG
        assert "test_key" in signal.metadata
        assert signal.metadata["test_key"] == "test_value"
    
    def test_trading_signal_optional_fields(self):
        """Test TradingSignal with optional fields."""
        signal = TradingSignal(
            strategy_id="test-strategy",
            symbol="ETHUSDT",
            action="SELL",
            quantity=Decimal('1.0'),
            price=Decimal('3000'),
            confidence=0.6,
            strength=SignalStrength.MEDIUM,
            timestamp=datetime.now(),
            stop_loss=Decimal('3100'),
            take_profit=Decimal('2800'),
            metadata={}
        )
        
        assert signal.stop_loss == Decimal('3100')
        assert signal.take_profit == Decimal('2800')


class TestStrategyPerformance:
    """Test suite for StrategyPerformance dataclass."""
    
    def test_strategy_performance_initialization(self):
        """Test StrategyPerformance initialization."""
        performance = StrategyPerformance()
        
        assert performance.total_trades == 0
        assert performance.winning_trades == 0
        assert performance.losing_trades == 0
        assert performance.total_pnl == Decimal('0')
        assert performance.win_rate == 0.0
        assert performance.average_win == Decimal('0')
        assert performance.average_loss == Decimal('0')
        assert performance.max_drawdown == Decimal('0')
        assert performance.sharpe_ratio == 0.0
    
    def test_strategy_performance_with_data(self):
        """Test StrategyPerformance with actual data."""
        performance = StrategyPerformance(
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            total_pnl=Decimal('500'),
            win_rate=0.7,
            average_win=Decimal('100'),
            average_loss=Decimal('-50'),
            max_drawdown=Decimal('-100'),
            sharpe_ratio=1.5
        )
        
        assert performance.total_trades == 10
        assert performance.winning_trades == 7
        assert performance.losing_trades == 3
        assert performance.win_rate == 0.7
        assert performance.sharpe_ratio == 1.5


# Performance tests
class TestStrategyFrameworkPerformance:
    """Performance tests for strategy framework."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.data_manager = MockDataManager()
        self.risk_manager = MockUnifiedRiskManager()
        
        self.strategy_manager = StrategyManager(
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager
        )
    
    @async_test
    async def test_multiple_strategies_signal_generation(self):
        """Test signal generation performance with multiple strategies."""
        from tests.conftest import PerformanceTimer
        
        # Create multiple strategies
        strategies = []
        for i in range(10):
            strategy = MovingAverageCrossoverStrategy(
                strategy_id=f"ma-strategy-{i}",
                name=f"MA Strategy {i}",
                data_manager=self.data_manager,
                risk_manager=self.risk_manager,
                config_manager=self.config_manager,
                fast_period=5 + i,
                slow_period=20 + i * 2
            )
            strategy.add_symbol("BTCUSDT")
            strategies.append(strategy)
            self.strategy_manager.add_strategy(strategy)
        
        # Activate all strategies
        await self.strategy_manager.activate_all_strategies()
        
        # Create test data
        test_data = create_test_market_data("BTCUSDT", 100)
        self.data_manager.get_market_data.return_value = test_data
        
        with PerformanceTimer("10 strategies signal generation"):
            all_signals = await self.strategy_manager.generate_all_signals(["BTCUSDT"])
        
        # Verify signals were generated (may be empty if no crossovers)
        assert isinstance(all_signals, list)
    
    @async_test
    async def test_large_dataset_processing(self):
        """Test strategy performance with large datasets."""
        from tests.conftest import PerformanceTimer
        
        strategy = MovingAverageCrossoverStrategy(
            strategy_id="large-data-test",
            name="Large Data Test",
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager,
            fast_period=10,
            slow_period=50
        )
        
        strategy.add_symbol("BTCUSDT")
        await strategy.activate()
        
        # Create large dataset (1 year of hourly data)
        large_data = create_test_market_data("BTCUSDT", 8760)  # 365 * 24 hours
        
        with PerformanceTimer("Process 1 year of hourly data"):
            signals = await strategy.generate_signals("BTCUSDT", large_data)
        
        assert isinstance(signals, list)