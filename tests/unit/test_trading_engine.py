"""
Unit Tests for Trading Engine

This module contains comprehensive unit tests for the TradingEngine
to ensure all trading execution functionality works correctly.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Optional

from src.bot.core.trading_engine import (
    TradingEngine,
    TradeSignal,
    TradeExecution,
    TradingSession,
    ExecutionStatus,
    EngineState
)
from src.bot.exchange.bybit_client import OrderSide, OrderType
from src.bot.risk_management.unified_risk_manager import RiskLevel, RiskAction
from tests.conftest import (
    MockBybitClient,
    MockDataManager,
    MockUnifiedRiskManager,
    MockConfigurationManager,
    create_test_trade_signal,
    create_test_position,
    async_test
)


class TestTradingEngine:
    """Test suite for TradingEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.bybit_client = MockBybitClient()
        self.data_manager = MockDataManager()
        self.risk_manager = MockUnifiedRiskManager()
        
        self.trading_engine = TradingEngine(
            bybit_client=self.bybit_client,
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager
        )
    
    def test_initialization(self):
        """Test trading engine initialization."""
        assert self.trading_engine.bybit_client == self.bybit_client
        assert self.trading_engine.data_manager == self.data_manager
        assert self.trading_engine.risk_manager == self.risk_manager
        assert self.trading_engine.config_manager == self.config_manager
        
        assert self.trading_engine.state == EngineState.STOPPED
        assert len(self.trading_engine.active_trades) == 0
        assert len(self.trading_engine.trade_history) == 0
        assert len(self.trading_engine.pending_signals) == 0
        assert self.trading_engine.trading_session is None
    
    @async_test
    async def test_start_engine(self):
        """Test starting the trading engine."""
        await self.trading_engine.start()
        
        assert self.trading_engine.state == EngineState.RUNNING
        assert self.trading_engine.trading_session is not None
        assert isinstance(self.trading_engine.trading_session, TradingSession)
    
    @async_test
    async def test_stop_engine(self):
        """Test stopping the trading engine."""
        await self.trading_engine.start()
        await self.trading_engine.stop()
        
        assert self.trading_engine.state == EngineState.STOPPED
        assert self.trading_engine.trading_session is None
    
    @async_test
    async def test_pause_resume_engine(self):
        """Test pausing and resuming the trading engine."""
        await self.trading_engine.start()
        
        # Test pause
        await self.trading_engine.pause()
        assert self.trading_engine.state == EngineState.PAUSED
        
        # Test resume
        await self.trading_engine.resume()
        assert self.trading_engine.state == EngineState.RUNNING
    
    @async_test
    async def test_emergency_stop(self):
        """Test emergency stop functionality."""
        await self.trading_engine.start()
        await self.trading_engine.emergency_stop()
        
        assert self.trading_engine.state == EngineState.EMERGENCY_STOPPED
        # Verify all open positions are closed
        self.bybit_client.cancel_all_orders.assert_called_once()
    
    @async_test
    async def test_process_trade_signal_buy(self):
        """Test processing buy trade signal."""
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('50000')
        )
        
        # Mock risk assessment - allow trade
        self.risk_manager.assess_trade_risk.return_value = Mock(
            recommended_action=RiskAction.CONTINUE,
            risk_level=RiskLevel.LOW,
            position_size_recommendation=Decimal('0.1')
        )
        
        # Mock successful order placement
        self.bybit_client.place_order.return_value = "test-order-123"
        
        await self.trading_engine.start()
        execution = await self.trading_engine.process_trade_signal(signal)
        
        assert execution is not None
        assert execution.signal == signal
        assert execution.status == ExecutionStatus.PENDING
        assert execution.order_id == "test-order-123"
        
        # Verify order was placed with correct parameters
        self.bybit_client.place_order.assert_called_once_with(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal('0.1')
        )
    
    @async_test
    async def test_process_trade_signal_sell(self):
        """Test processing sell trade signal."""
        signal = create_test_trade_signal(
            symbol="ETHUSDT",
            action="SELL",
            quantity=Decimal('1.0'),
            price=Decimal('3000')
        )
        
        # Mock risk assessment - allow trade
        self.risk_manager.assess_trade_risk.return_value = Mock(
            recommended_action=RiskAction.CONTINUE,
            risk_level=RiskLevel.LOW,
            position_size_recommendation=Decimal('1.0')
        )
        
        self.bybit_client.place_order.return_value = "test-order-456"
        
        await self.trading_engine.start()
        execution = await self.trading_engine.process_trade_signal(signal)
        
        assert execution.order_id == "test-order-456"
        
        # Verify sell order placement
        self.bybit_client.place_order.assert_called_once_with(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=Decimal('1.0')
        )
    
    @async_test
    async def test_process_trade_signal_limit_order(self):
        """Test processing trade signal with limit order."""
        signal = create_test_trade_signal(
            symbol="ADAUSDT",
            action="BUY",
            quantity=Decimal('1000'),
            price=Decimal('0.45'),
            order_type="LIMIT"
        )
        
        self.risk_manager.assess_trade_risk.return_value = Mock(
            recommended_action=RiskAction.CONTINUE,
            risk_level=RiskLevel.LOW,
            position_size_recommendation=Decimal('1000')
        )
        
        self.bybit_client.place_order.return_value = "limit-order-789"
        
        await self.trading_engine.start()
        execution = await self.trading_engine.process_trade_signal(signal)
        
        # Verify limit order placement
        self.bybit_client.place_order.assert_called_once_with(
            symbol="ADAUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=Decimal('1000'),
            price=Decimal('0.45')
        )
    
    @async_test
    async def test_process_trade_signal_with_stops(self):
        """Test processing trade signal with stop loss and take profit."""
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('50000'),
            stop_loss=Decimal('48000'),
            take_profit=Decimal('54000')
        )
        
        self.risk_manager.assess_trade_risk.return_value = Mock(
            recommended_action=RiskAction.CONTINUE,
            risk_level=RiskLevel.LOW,
            position_size_recommendation=Decimal('0.1')
        )
        
        self.bybit_client.place_order.return_value = "stops-order-123"
        
        await self.trading_engine.start()
        execution = await self.trading_engine.process_trade_signal(signal)
        
        # Verify order with stops
        self.bybit_client.place_order.assert_called_once_with(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal('0.1'),
            stop_loss=Decimal('48000'),
            take_profit=Decimal('54000')
        )
    
    @async_test
    async def test_process_trade_signal_rejected_by_risk(self):
        """Test trade signal rejected by risk management."""
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('10.0'),  # Large position
            price=Decimal('50000')
        )
        
        # Mock risk assessment - reject trade
        self.risk_manager.assess_trade_risk.return_value = Mock(
            recommended_action=RiskAction.HALT_TRADING,
            risk_level=RiskLevel.CRITICAL,
            position_size_recommendation=Decimal('0')
        )
        
        await self.trading_engine.start()
        execution = await self.trading_engine.process_trade_signal(signal)
        
        assert execution.status == ExecutionStatus.REJECTED
        assert "risk management" in execution.rejection_reason.lower()
        
        # Verify no order was placed
        self.bybit_client.place_order.assert_not_called()
    
    @async_test
    async def test_process_trade_signal_engine_stopped(self):
        """Test processing trade signal when engine is stopped."""
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('50000')
        )
        
        # Engine is stopped by default
        execution = await self.trading_engine.process_trade_signal(signal)
        
        assert execution.status == ExecutionStatus.REJECTED
        assert "engine not running" in execution.rejection_reason.lower()
    
    @async_test
    async def test_process_trade_signal_emergency_stopped(self):
        """Test processing trade signal during emergency stop."""
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('50000')
        )
        
        await self.trading_engine.start()
        await self.trading_engine.emergency_stop()
        
        execution = await self.trading_engine.process_trade_signal(signal)
        
        assert execution.status == ExecutionStatus.REJECTED
        assert "emergency stop" in execution.rejection_reason.lower()
    
    @async_test
    async def test_cancel_trade(self):
        """Test canceling an active trade."""
        # First place a trade
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('50000')
        )
        
        self.risk_manager.assess_trade_risk.return_value = Mock(
            recommended_action=RiskAction.CONTINUE,
            risk_level=RiskLevel.LOW,
            position_size_recommendation=Decimal('0.1')
        )
        
        self.bybit_client.place_order.return_value = "cancel-test-order"
        self.bybit_client.cancel_order.return_value = True
        
        await self.trading_engine.start()
        execution = await self.trading_engine.process_trade_signal(signal)
        
        # Now cancel the trade
        success = await self.trading_engine.cancel_trade(execution.execution_id)
        
        assert success is True
        assert execution.status == ExecutionStatus.CANCELLED
        
        # Verify cancel order was called
        self.bybit_client.cancel_order.assert_called_once_with(
            symbol="BTCUSDT",
            order_id="cancel-test-order"
        )
    
    @async_test
    async def test_cancel_nonexistent_trade(self):
        """Test canceling a non-existent trade."""
        await self.trading_engine.start()
        success = await self.trading_engine.cancel_trade("nonexistent-id")
        
        assert success is False
    
    @async_test
    async def test_update_trade_status(self):
        """Test updating trade status based on order status."""
        # Create a pending execution
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('50000')
        )
        
        execution = TradeExecution(
            execution_id="test-execution-123",
            signal=signal,
            order_id="test-order-123",
            status=ExecutionStatus.PENDING,
            timestamp=datetime.now()
        )
        
        self.trading_engine.active_trades["test-execution-123"] = execution
        
        # Mock order status as filled
        self.bybit_client.get_order_status.return_value = {
            "orderId": "test-order-123",
            "orderStatus": "Filled",
            "avgPrice": "49950.00",
            "cumExecQty": "0.1"
        }
        
        await self.trading_engine.start()
        await self.trading_engine._update_trade_status(execution)
        
        assert execution.status == ExecutionStatus.FILLED
        assert execution.fill_price == Decimal('49950.00')
        assert execution.filled_quantity == Decimal('0.1')
    
    @async_test
    async def test_monitor_active_trades(self):
        """Test monitoring active trades."""
        # Create active trades
        executions = []
        for i in range(3):
            signal = create_test_trade_signal(
                symbol=f"TEST{i}USDT",
                action="BUY",
                quantity=Decimal('0.1'),
                price=Decimal('1000')
            )
            
            execution = TradeExecution(
                execution_id=f"test-execution-{i}",
                signal=signal,
                order_id=f"test-order-{i}",
                status=ExecutionStatus.PENDING,
                timestamp=datetime.now()
            )
            
            executions.append(execution)
            self.trading_engine.active_trades[f"test-execution-{i}"] = execution
        
        # Mock order statuses
        def mock_get_order_status(symbol, order_id):
            return {
                "orderId": order_id,
                "orderStatus": "Filled",
                "avgPrice": "999.50",
                "cumExecQty": "0.1"
            }
        
        self.bybit_client.get_order_status.side_effect = mock_get_order_status
        
        await self.trading_engine.start()
        await self.trading_engine._monitor_active_trades()
        
        # Verify all trades were updated
        for execution in executions:
            assert execution.status == ExecutionStatus.FILLED
    
    @async_test
    async def test_position_sizing_adjustment(self):
        """Test position sizing adjustment based on risk management."""
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('1.0'),  # Original size
            price=Decimal('50000')
        )
        
        # Mock risk assessment with reduced position size
        self.risk_manager.assess_trade_risk.return_value = Mock(
            recommended_action=RiskAction.REDUCE_POSITION,
            risk_level=RiskLevel.MEDIUM,
            position_size_recommendation=Decimal('0.5')  # Reduced size
        )
        
        self.bybit_client.place_order.return_value = "size-adjusted-order"
        
        await self.trading_engine.start()
        execution = await self.trading_engine.process_trade_signal(signal)
        
        # Verify adjusted position size was used
        self.bybit_client.place_order.assert_called_once_with(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal('0.5')  # Adjusted size
        )
    
    @async_test
    async def test_get_active_trades(self):
        """Test getting active trades."""
        # Add some active trades
        for i in range(3):
            signal = create_test_trade_signal(
                symbol=f"TEST{i}USDT",
                action="BUY",
                quantity=Decimal('0.1'),
                price=Decimal('1000')
            )
            
            execution = TradeExecution(
                execution_id=f"test-execution-{i}",
                signal=signal,
                order_id=f"test-order-{i}",
                status=ExecutionStatus.PENDING,
                timestamp=datetime.now()
            )
            
            self.trading_engine.active_trades[f"test-execution-{i}"] = execution
        
        active_trades = self.trading_engine.get_active_trades()
        
        assert len(active_trades) == 3
        assert all(isinstance(trade, TradeExecution) for trade in active_trades)
    
    @async_test
    async def test_get_trade_history(self):
        """Test getting trade history."""
        # Add some completed trades to history
        for i in range(5):
            signal = create_test_trade_signal(
                symbol=f"HIST{i}USDT",
                action="BUY",
                quantity=Decimal('0.1'),
                price=Decimal('1000')
            )
            
            execution = TradeExecution(
                execution_id=f"hist-execution-{i}",
                signal=signal,
                order_id=f"hist-order-{i}",
                status=ExecutionStatus.FILLED,
                timestamp=datetime.now() - timedelta(hours=i),
                fill_price=Decimal('999.50'),
                filled_quantity=Decimal('0.1')
            )
            
            self.trading_engine.trade_history.append(execution)
        
        history = self.trading_engine.get_trade_history()
        
        assert len(history) == 5
        assert all(trade.status == ExecutionStatus.FILLED for trade in history)
    
    @async_test
    async def test_get_trade_history_filtered(self):
        """Test getting filtered trade history."""
        # Add trades for different symbols
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        for symbol in symbols:
            for i in range(2):
                signal = create_test_trade_signal(
                    symbol=symbol,
                    action="BUY",
                    quantity=Decimal('0.1'),
                    price=Decimal('1000')
                )
                
                execution = TradeExecution(
                    execution_id=f"{symbol}-{i}",
                    signal=signal,
                    order_id=f"order-{symbol}-{i}",
                    status=ExecutionStatus.FILLED,
                    timestamp=datetime.now() - timedelta(hours=i),
                    fill_price=Decimal('999.50'),
                    filled_quantity=Decimal('0.1')
                )
                
                self.trading_engine.trade_history.append(execution)
        
        # Filter by symbol
        btc_history = self.trading_engine.get_trade_history(symbol="BTCUSDT")
        
        assert len(btc_history) == 2
        assert all(trade.signal.symbol == "BTCUSDT" for trade in btc_history)
    
    def test_get_engine_status(self):
        """Test getting engine status."""
        status = self.trading_engine.get_engine_status()
        
        assert "state" in status
        assert "active_trades_count" in status
        assert "total_trades_today" in status
        assert "session_info" in status
        assert "risk_metrics" in status
        
        assert status["state"] == EngineState.STOPPED
        assert status["active_trades_count"] == 0
    
    @async_test
    async def test_session_management(self):
        """Test trading session management."""
        await self.trading_engine.start()
        
        session = self.trading_engine.trading_session
        assert session is not None
        assert isinstance(session.start_time, datetime)
        assert session.end_time is None
        assert session.total_trades == 0
        assert session.successful_trades == 0
        assert session.total_pnl == Decimal('0')
        
        await self.trading_engine.stop()
        
        assert session.end_time is not None
    
    @async_test
    async def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        # Add some completed trades
        profits = [Decimal('100'), Decimal('-50'), Decimal('150'), Decimal('-25')]
        
        for i, profit in enumerate(profits):
            signal = create_test_trade_signal(
                symbol=f"PERF{i}USDT",
                action="BUY",
                quantity=Decimal('0.1'),
                price=Decimal('1000')
            )
            
            execution = TradeExecution(
                execution_id=f"perf-execution-{i}",
                signal=signal,
                order_id=f"perf-order-{i}",
                status=ExecutionStatus.FILLED,
                timestamp=datetime.now() - timedelta(hours=i),
                fill_price=Decimal('1000'),
                filled_quantity=Decimal('0.1'),
                realized_pnl=profit
            )
            
            self.trading_engine.trade_history.append(execution)
        
        metrics = self.trading_engine._calculate_performance_metrics()
        
        assert metrics["total_trades"] == 4
        assert metrics["successful_trades"] == 2  # Positive PnL trades
        assert metrics["win_rate"] == 0.5
        assert metrics["total_pnl"] == Decimal('175')  # Sum of all PnL
        assert metrics["average_win"] == Decimal('125')  # (100 + 150) / 2
        assert metrics["average_loss"] == Decimal('-37.5')  # (-50 + -25) / 2


class TestTradeExecution:
    """Test suite for TradeExecution dataclass."""
    
    def test_trade_execution_creation(self):
        """Test TradeExecution creation."""
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('50000')
        )
        
        execution = TradeExecution(
            execution_id="test-execution-123",
            signal=signal,
            order_id="test-order-123",
            status=ExecutionStatus.PENDING,
            timestamp=datetime.now()
        )
        
        assert execution.execution_id == "test-execution-123"
        assert execution.signal == signal
        assert execution.order_id == "test-order-123"
        assert execution.status == ExecutionStatus.PENDING
        assert execution.fill_price is None
        assert execution.filled_quantity is None
        assert execution.realized_pnl is None
    
    def test_trade_execution_completion(self):
        """Test TradeExecution with completion data."""
        signal = create_test_trade_signal(
            symbol="ETHUSDT",
            action="SELL",
            quantity=Decimal('1.0'),
            price=Decimal('3000')
        )
        
        execution = TradeExecution(
            execution_id="completed-execution-456",
            signal=signal,
            order_id="completed-order-456",
            status=ExecutionStatus.FILLED,
            timestamp=datetime.now(),
            fill_price=Decimal('2995.50'),
            filled_quantity=Decimal('1.0'),
            realized_pnl=Decimal('25.75')
        )
        
        assert execution.status == ExecutionStatus.FILLED
        assert execution.fill_price == Decimal('2995.50')
        assert execution.filled_quantity == Decimal('1.0')
        assert execution.realized_pnl == Decimal('25.75')


class TestTradingSession:
    """Test suite for TradingSession dataclass."""
    
    def test_trading_session_creation(self):
        """Test TradingSession creation."""
        session = TradingSession(
            session_id="test-session-123",
            start_time=datetime.now()
        )
        
        assert session.session_id == "test-session-123"
        assert isinstance(session.start_time, datetime)
        assert session.end_time is None
        assert session.total_trades == 0
        assert session.successful_trades == 0
        assert session.total_pnl == Decimal('0')
    
    def test_trading_session_completion(self):
        """Test TradingSession with completion data."""
        start_time = datetime.now() - timedelta(hours=2)
        session = TradingSession(
            session_id="completed-session-456",
            start_time=start_time,
            end_time=datetime.now(),
            total_trades=10,
            successful_trades=7,
            total_pnl=Decimal('250.50')
        )
        
        assert session.end_time is not None
        assert session.total_trades == 10
        assert session.successful_trades == 7
        assert session.total_pnl == Decimal('250.50')


# Performance tests
class TestTradingEnginePerformance:
    """Performance tests for trading engine."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.bybit_client = MockBybitClient()
        self.data_manager = MockDataManager()
        self.risk_manager = MockUnifiedRiskManager()
        
        self.trading_engine = TradingEngine(
            bybit_client=self.bybit_client,
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager
        )
    
    @async_test
    async def test_concurrent_signal_processing(self):
        """Test concurrent signal processing performance."""
        from tests.conftest import PerformanceTimer
        
        # Create multiple signals
        signals = []
        for i in range(50):
            signal = create_test_trade_signal(
                symbol=f"PERF{i}USDT",
                action="BUY",
                quantity=Decimal('0.1'),
                price=Decimal('1000')
            )
            signals.append(signal)
        
        # Mock responses
        self.risk_manager.assess_trade_risk.return_value = Mock(
            recommended_action=RiskAction.CONTINUE,
            risk_level=RiskLevel.LOW,
            position_size_recommendation=Decimal('0.1')
        )
        
        self.bybit_client.place_order.return_value = "performance-order"
        
        await self.trading_engine.start()
        
        with PerformanceTimer("50 concurrent signal processing"):
            tasks = []
            for signal in signals:
                task = self.trading_engine.process_trade_signal(signal)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        assert len(results) == 50
        assert all(isinstance(r, TradeExecution) for r in results)
    
    @async_test
    async def test_trade_monitoring_performance(self):
        """Test trade monitoring performance with many active trades."""
        from tests.conftest import PerformanceTimer
        
        # Create many active trades
        for i in range(100):
            signal = create_test_trade_signal(
                symbol=f"MON{i}USDT",
                action="BUY",
                quantity=Decimal('0.1'),
                price=Decimal('1000')
            )
            
            execution = TradeExecution(
                execution_id=f"monitor-execution-{i}",
                signal=signal,
                order_id=f"monitor-order-{i}",
                status=ExecutionStatus.PENDING,
                timestamp=datetime.now()
            )
            
            self.trading_engine.active_trades[f"monitor-execution-{i}"] = execution
        
        # Mock order status responses
        self.bybit_client.get_order_status.return_value = {
            "orderId": "test-order",
            "orderStatus": "Filled",
            "avgPrice": "999.50",
            "cumExecQty": "0.1"
        }
        
        await self.trading_engine.start()
        
        with PerformanceTimer("Monitor 100 active trades"):
            await self.trading_engine._monitor_active_trades()
        
        # Verify all trades were processed
        filled_trades = [t for t in self.trading_engine.active_trades.values() 
                        if t.status == ExecutionStatus.FILLED]
        assert len(filled_trades) == 100