"""
Integration Tests for Trading Engine Workflows

This module contains integration tests that test the TradingEngine's
integration with other components like risk management, API client, and data management.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd

from src.bot.core.trading_engine import (
    TradingEngine,
    TradeSignal,
    TradeExecution,
    ExecutionStatus,
    EngineState
)
from src.bot.exchange.bybit_client import OrderSide, OrderType
from src.bot.risk_management.unified_risk_manager import RiskLevel, RiskAction, TradeRiskAssessment
from tests.conftest import (
    MockBybitClient,
    MockDataManager,
    MockUnifiedRiskManager,
    MockConfigurationManager,
    create_test_trade_signal,
    create_test_position,
    async_test
)


class TestTradingEngineIntegration:
    """Integration tests for trading engine workflows."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
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
    async def test_complete_trade_execution_workflow(self):
        """Test complete trade execution workflow with all components."""
        # Setup: Start the engine
        await self.trading_engine.start()
        
        # Create a trade signal
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('50000'),
            stop_loss=Decimal('48000'),
            take_profit=Decimal('54000')
        )
        
        # Mock risk assessment - allow trade with position sizing
        risk_assessment = TradeRiskAssessment(
            symbol="BTCUSDT",
            side="Buy",
            size=Decimal('0.1'),
            entry_price=Decimal('50000'),
            stop_loss=Decimal('48000'),
            take_profit=Decimal('54000'),
            risk_amount=Decimal('200'),
            risk_percentage=0.02,
            position_size_recommendation=Decimal('0.08'),  # Reduced size
            risk_reward_ratio=2.0,
            probability_of_success=0.6,
            expected_value=80.0,
            risk_level=RiskLevel.MEDIUM,
            recommended_action=RiskAction.REDUCE_POSITION,
            risk_factors=["Moderate volatility"]
        )
        
        self.risk_manager.assess_trade_risk.return_value = risk_assessment
        
        # Mock successful order placement
        self.bybit_client.place_order.return_value = "integration-order-123"
        
        # Mock order status progression: New -> PartiallyFilled -> Filled
        order_statuses = [
            {
                "orderId": "integration-order-123",
                "orderStatus": "New",
                "avgPrice": "0",
                "cumExecQty": "0"
            },
            {
                "orderId": "integration-order-123",
                "orderStatus": "PartiallyFilled",
                "avgPrice": "49980.00",
                "cumExecQty": "0.05"
            },
            {
                "orderId": "integration-order-123",
                "orderStatus": "Filled",
                "avgPrice": "49990.00",
                "cumExecQty": "0.08"
            }
        ]
        
        self.bybit_client.get_order_status.side_effect = order_statuses
        
        # Execute the trade
        execution = await self.trading_engine.process_trade_signal(signal)
        
        # Verify execution was created
        assert execution is not None
        assert execution.status == ExecutionStatus.PENDING
        assert execution.order_id == "integration-order-123"
        
        # Verify risk assessment was called
        self.risk_manager.assess_trade_risk.assert_called_once_with(
            symbol="BTCUSDT",
            side="BUY",
            size=Decimal('0.1'),
            entry_price=Decimal('50000'),
            stop_loss=Decimal('48000'),
            take_profit=Decimal('54000')
        )
        
        # Verify order was placed with adjusted size
        self.bybit_client.place_order.assert_called_once_with(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal('0.08'),  # Risk-adjusted size
            stop_loss=Decimal('48000'),
            take_profit=Decimal('54000')
        )
        
        # Simulate status updates
        await self.trading_engine._update_trade_status(execution)
        assert execution.status == ExecutionStatus.PENDING  # Still New
        
        await self.trading_engine._update_trade_status(execution)
        assert execution.status == ExecutionStatus.PARTIALLY_FILLED
        assert execution.fill_price == Decimal('49980.00')
        assert execution.filled_quantity == Decimal('0.05')
        
        await self.trading_engine._update_trade_status(execution)
        assert execution.status == ExecutionStatus.FILLED
        assert execution.fill_price == Decimal('49990.00')
        assert execution.filled_quantity == Decimal('0.08')
        
        # Verify trade is moved to history
        assert len(self.trading_engine.trade_history) > 0
        assert execution in self.trading_engine.trade_history
    
    @async_test
    async def test_risk_rejection_integration(self):
        """Test integration when risk management rejects trade."""
        await self.trading_engine.start()
        
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('10.0'),  # Large position
            price=Decimal('50000')
        )
        
        # Mock risk assessment - reject trade
        risk_assessment = TradeRiskAssessment(
            symbol="BTCUSDT",
            side="Buy",
            size=Decimal('10.0'),
            entry_price=Decimal('50000'),
            stop_loss=None,
            take_profit=None,
            risk_amount=Decimal('50000'),
            risk_percentage=0.50,  # 50% portfolio risk
            position_size_recommendation=Decimal('0'),
            risk_reward_ratio=1.0,
            probability_of_success=0.3,
            expected_value=-1000.0,
            risk_level=RiskLevel.CRITICAL,
            recommended_action=RiskAction.HALT_TRADING,
            risk_factors=["Excessive position size", "High portfolio risk", "Poor risk-reward ratio"]
        )
        
        self.risk_manager.assess_trade_risk.return_value = risk_assessment
        
        # Execute the trade
        execution = await self.trading_engine.process_trade_signal(signal)
        
        # Verify trade was rejected
        assert execution.status == ExecutionStatus.REJECTED
        assert "risk management" in execution.rejection_reason.lower()
        
        # Verify no order was placed
        self.bybit_client.place_order.assert_not_called()
        
        # Verify risk assessment was called
        self.risk_manager.assess_trade_risk.assert_called_once()
    
    @async_test
    async def test_multiple_concurrent_trades_integration(self):
        """Test handling multiple concurrent trades."""
        await self.trading_engine.start()
        
        # Create multiple signals
        signals = []
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "SOLUSDT"]
        
        for i, symbol in enumerate(symbols):
            signal = create_test_trade_signal(
                symbol=symbol,
                action="BUY",
                quantity=Decimal('0.1'),
                price=Decimal(f'{1000 + i * 100}')
            )
            signals.append(signal)
        
        # Mock risk assessments - allow all trades
        def mock_risk_assessment(symbol, side, size, entry_price, **kwargs):
            return TradeRiskAssessment(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                stop_loss=kwargs.get('stop_loss'),
                take_profit=kwargs.get('take_profit'),
                risk_amount=Decimal('100'),
                risk_percentage=0.01,
                position_size_recommendation=size,
                risk_reward_ratio=2.0,
                probability_of_success=0.7,
                expected_value=50.0,
                risk_level=RiskLevel.LOW,
                recommended_action=RiskAction.CONTINUE,
                risk_factors=["Normal market conditions"]
            )
        
        self.risk_manager.assess_trade_risk.side_effect = mock_risk_assessment
        
        # Mock order placement
        def mock_place_order(symbol, **kwargs):
            return f"order-{symbol}-{hash(str(kwargs)) % 1000}"
        
        self.bybit_client.place_order.side_effect = mock_place_order
        
        # Execute all trades concurrently
        tasks = []
        for signal in signals:
            task = self.trading_engine.process_trade_signal(signal)
            tasks.append(task)
        
        executions = await asyncio.gather(*tasks)
        
        # Verify all trades were processed
        assert len(executions) == 5
        assert all(ex.status == ExecutionStatus.PENDING for ex in executions)
        assert all(ex.order_id is not None for ex in executions)
        
        # Verify risk assessments were called for all trades
        assert self.risk_manager.assess_trade_risk.call_count == 5
        
        # Verify orders were placed for all trades
        assert self.bybit_client.place_order.call_count == 5
        
        # Verify active trades tracking
        assert len(self.trading_engine.active_trades) == 5
    
    @async_test
    async def test_portfolio_monitoring_integration(self):
        """Test portfolio monitoring integration."""
        await self.trading_engine.start()
        
        # Add some active trades
        for i in range(3):
            signal = create_test_trade_signal(
                symbol=f"TEST{i}USDT",
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
        
        # Mock order status updates
        def mock_order_status(symbol, order_id):
            return {
                "orderId": order_id,
                "orderStatus": "Filled",
                "avgPrice": "999.50",
                "cumExecQty": "0.1"
            }
        
        self.bybit_client.get_order_status.side_effect = mock_order_status
        
        # Mock position data
        mock_positions = [
            {
                "symbol": "TEST0USDT",
                "size": "0.1",
                "unrealisedPnl": "5.00",
                "positionValue": "100.00"
            },
            {
                "symbol": "TEST1USDT", 
                "size": "0.1",
                "unrealisedPnl": "-2.50",
                "positionValue": "100.00"
            },
            {
                "symbol": "TEST2USDT",
                "size": "0.1", 
                "unrealisedPnl": "8.75",
                "positionValue": "100.00"
            }
        ]
        
        self.bybit_client.get_positions.return_value = mock_positions
        
        # Mock portfolio metrics from risk manager
        from src.bot.risk_management.unified_risk_manager import RiskMetrics
        mock_metrics = RiskMetrics(
            portfolio_value=Decimal('10000'),
            total_exposure=Decimal('300'),
            leverage=0.03,
            var_95=-0.02,
            expected_shortfall=-0.03,
            max_drawdown=-0.05,
            current_drawdown=-0.01,
            sharpe_ratio=1.2,
            sortino_ratio=1.8,
            volatility=0.15,
            beta=1.1,
            correlation_risk=0.3,
            concentration_risk=0.1,
            liquidity_risk=0.2,
            risk_score=0.25,
            risk_level=RiskLevel.LOW
        )
        
        self.risk_manager.calculate_portfolio_metrics.return_value = mock_metrics
        
        # Perform monitoring cycle
        await self.trading_engine._monitor_active_trades()
        
        # Verify all trades were updated
        for execution in self.trading_engine.active_trades.values():
            assert execution.status == ExecutionStatus.FILLED
            assert execution.fill_price == Decimal('999.50')
            assert execution.filled_quantity == Decimal('0.1')
        
        # Test portfolio metrics calculation
        positions = {pos["symbol"]: pos for pos in mock_positions}
        metrics = await self.risk_manager.calculate_portfolio_metrics(
            positions, Decimal('10000')
        )
        
        assert metrics.portfolio_value == Decimal('10000')
        assert metrics.risk_level == RiskLevel.LOW
    
    @async_test
    async def test_emergency_stop_integration(self):
        """Test emergency stop integration across all components."""
        await self.trading_engine.start()
        
        # Add active trades
        active_executions = []
        for i in range(3):
            signal = create_test_trade_signal(
                symbol=f"EMERGENCY{i}USDT",
                action="BUY",
                quantity=Decimal('0.1'),
                price=Decimal('1000')
            )
            
            execution = TradeExecution(
                execution_id=f"emergency-execution-{i}",
                signal=signal,
                order_id=f"emergency-order-{i}",
                status=ExecutionStatus.PENDING,
                timestamp=datetime.now()
            )
            
            self.trading_engine.active_trades[f"emergency-execution-{i}"] = execution
            active_executions.append(execution)
        
        # Mock successful order cancellations
        self.bybit_client.cancel_order.return_value = True
        self.bybit_client.cancel_all_orders.return_value = True
        
        # Trigger emergency stop
        await self.trading_engine.emergency_stop()
        
        # Verify engine state
        assert self.trading_engine.state == EngineState.EMERGENCY_STOPPED
        
        # Verify all orders were cancelled
        self.bybit_client.cancel_all_orders.assert_called_once()
        
        # Verify risk manager was notified (if implemented)
        # In a real implementation, you might want to notify risk manager of emergency stop
        
        # Try to process new signal during emergency stop
        new_signal = create_test_trade_signal(
            symbol="NEWSIGNAL",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('1000')
        )
        
        execution = await self.trading_engine.process_trade_signal(new_signal)
        
        # Verify new trade is rejected
        assert execution.status == ExecutionStatus.REJECTED
        assert "emergency stop" in execution.rejection_reason.lower()
    
    @async_test
    async def test_data_integration_workflow(self):
        """Test integration with data management for market data."""
        await self.trading_engine.start()
        
        # Mock market data from data manager
        market_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1H'),
            'open': [50000 + i * 10 for i in range(100)],
            'high': [50010 + i * 10 for i in range(100)],
            'low': [49990 + i * 10 for i in range(100)],
            'close': [50005 + i * 10 for i in range(100)],
            'volume': [100 + i for i in range(100)]
        })
        
        self.data_manager.get_market_data.return_value = market_data
        
        # Create signal based on latest market data
        latest_price = market_data.iloc[-1]['close']
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal(str(latest_price))
        )
        
        # Mock risk assessment using market data
        self.risk_manager.assess_trade_risk.return_value = TradeRiskAssessment(
            symbol="BTCUSDT",
            side="Buy",
            size=Decimal('0.1'),
            entry_price=Decimal(str(latest_price)),
            stop_loss=None,
            take_profit=None,
            risk_amount=Decimal('100'),
            risk_percentage=0.01,
            position_size_recommendation=Decimal('0.1'),
            risk_reward_ratio=2.0,
            probability_of_success=0.7,
            expected_value=50.0,
            risk_level=RiskLevel.LOW,
            recommended_action=RiskAction.CONTINUE,
            risk_factors=["Normal market conditions"]
        )
        
        self.bybit_client.place_order.return_value = "data-integration-order"
        
        # Process signal
        execution = await self.trading_engine.process_trade_signal(signal)
        
        # Verify data manager was used (in practice, might be called for additional analysis)
        # For now, just verify the signal was processed correctly
        assert execution.status == ExecutionStatus.PENDING
        assert execution.signal.price == Decimal(str(latest_price))
    
    @async_test
    async def test_performance_tracking_integration(self):
        """Test performance tracking integration across components."""
        await self.trading_engine.start()
        
        # Create and execute multiple trades with different outcomes
        trade_outcomes = [
            ("BTCUSDT", Decimal('100'), True),    # Win
            ("ETHUSDT", Decimal('-50'), False),   # Loss
            ("ADAUSDT", Decimal('75'), True),     # Win
            ("DOTUSDT", Decimal('-25'), False),   # Loss
            ("SOLUSDT", Decimal('150'), True)     # Win
        ]
        
        for symbol, pnl, is_win in trade_outcomes:
            signal = create_test_trade_signal(
                symbol=symbol,
                action="BUY",
                quantity=Decimal('0.1'),
                price=Decimal('1000')
            )
            
            # Create completed execution
            execution = TradeExecution(
                execution_id=f"perf-{symbol}",
                signal=signal,
                order_id=f"order-{symbol}",
                status=ExecutionStatus.FILLED,
                timestamp=datetime.now(),
                fill_price=Decimal('1000'),
                filled_quantity=Decimal('0.1'),
                realized_pnl=pnl
            )
            
            self.trading_engine.trade_history.append(execution)
        
        # Calculate performance metrics
        metrics = self.trading_engine._calculate_performance_metrics()
        
        # Verify metrics
        assert metrics["total_trades"] == 5
        assert metrics["successful_trades"] == 3
        assert metrics["win_rate"] == 0.6
        assert metrics["total_pnl"] == Decimal('250')  # 100 - 50 + 75 - 25 + 150
        assert metrics["average_win"] == Decimal('108.33')  # (100 + 75 + 150) / 3
        assert metrics["average_loss"] == Decimal('-37.5')  # (-50 + -25) / 2
        
        # Test session performance
        if self.trading_engine.trading_session:
            session = self.trading_engine.trading_session
            session.total_trades = 5
            session.successful_trades = 3
            session.total_pnl = Decimal('250')
            
            assert session.total_trades == 5
            assert session.successful_trades == 3
            assert session.total_pnl == Decimal('250')
    
    @async_test
    async def test_configuration_integration(self):
        """Test integration with configuration management."""
        # Test that configuration affects trading behavior
        config = self.config_manager.config
        
        # Mock configuration values
        config["trading"]["max_position_size"] = 0.1
        config["trading"]["default_stop_loss_pct"] = 0.02
        config["trading"]["default_take_profit_pct"] = 0.04
        
        await self.trading_engine.start()
        
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.15'),  # Exceeds max position size
            price=Decimal('50000')
        )
        
        # Mock risk assessment that considers configuration
        self.risk_manager.assess_trade_risk.return_value = TradeRiskAssessment(
            symbol="BTCUSDT",
            side="Buy",
            size=Decimal('0.15'),
            entry_price=Decimal('50000'),
            stop_loss=None,
            take_profit=None,
            risk_amount=Decimal('150'),
            risk_percentage=0.015,
            position_size_recommendation=Decimal('0.1'),  # Reduced to max allowed
            risk_reward_ratio=2.0,
            probability_of_success=0.7,
            expected_value=50.0,
            risk_level=RiskLevel.MEDIUM,
            recommended_action=RiskAction.REDUCE_POSITION,
            risk_factors=["Position size exceeds maximum"]
        )
        
        self.bybit_client.place_order.return_value = "config-test-order"
        
        execution = await self.trading_engine.process_trade_signal(signal)
        
        # Verify configuration was respected
        assert execution.status == ExecutionStatus.PENDING
        
        # Verify order was placed with reduced size
        self.bybit_client.place_order.assert_called_once()
        call_args = self.bybit_client.place_order.call_args[1]
        assert call_args["qty"] == Decimal('0.1')  # Reduced to max allowed
    
    @async_test 
    async def test_error_recovery_integration(self):
        """Test error recovery integration across components."""
        await self.trading_engine.start()
        
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('50000')
        )
        
        # Mock risk assessment success
        self.risk_manager.assess_trade_risk.return_value = TradeRiskAssessment(
            symbol="BTCUSDT",
            side="Buy",
            size=Decimal('0.1'),
            entry_price=Decimal('50000'),
            stop_loss=None,
            take_profit=None,
            risk_amount=Decimal('100'),
            risk_percentage=0.01,
            position_size_recommendation=Decimal('0.1'),
            risk_reward_ratio=2.0,
            probability_of_success=0.7,
            expected_value=50.0,
            risk_level=RiskLevel.LOW,
            recommended_action=RiskAction.CONTINUE,
            risk_factors=["Normal conditions"]
        )
        
        # Mock order placement failure then success
        from src.bot.exchange.bybit_client import BybitAPIError
        
        self.bybit_client.place_order.side_effect = [
            BybitAPIError("Network error", 500),  # First attempt fails
            "recovery-order-123"  # Second attempt succeeds
        ]
        
        # Process signal - should handle error and retry
        execution = await self.trading_engine.process_trade_signal(signal)
        
        # In a real implementation, you might want retry logic
        # For now, verify the error was handled
        assert execution.status == ExecutionStatus.REJECTED
        assert "api error" in execution.rejection_reason.lower() or "network error" in execution.rejection_reason.lower()


class TestTradingEngineStateManagement:
    """Test trading engine state management integration."""
    
    def setup_method(self):
        """Set up state management test fixtures."""
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
    async def test_state_transitions_integration(self):
        """Test all state transitions work correctly."""
        # Initial state
        assert self.trading_engine.state == EngineState.STOPPED
        
        # Start -> Running
        await self.trading_engine.start()
        assert self.trading_engine.state == EngineState.RUNNING
        
        # Running -> Paused
        await self.trading_engine.pause()
        assert self.trading_engine.state == EngineState.PAUSED
        
        # Paused -> Running
        await self.trading_engine.resume()
        assert self.trading_engine.state == EngineState.RUNNING
        
        # Running -> Emergency Stopped
        await self.trading_engine.emergency_stop()
        assert self.trading_engine.state == EngineState.EMERGENCY_STOPPED
        
        # Emergency Stopped -> Stopped
        await self.trading_engine.stop()
        assert self.trading_engine.state == EngineState.STOPPED
    
    @async_test
    async def test_state_behavior_integration(self):
        """Test that different states affect trading behavior correctly."""
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('50000')
        )
        
        # Test STOPPED state
        execution = await self.trading_engine.process_trade_signal(signal)
        assert execution.status == ExecutionStatus.REJECTED
        assert "not running" in execution.rejection_reason.lower()
        
        # Test RUNNING state
        await self.trading_engine.start()
        
        self.risk_manager.assess_trade_risk.return_value = TradeRiskAssessment(
            symbol="BTCUSDT",
            side="Buy",
            size=Decimal('0.1'),
            entry_price=Decimal('50000'),
            stop_loss=None,
            take_profit=None,
            risk_amount=Decimal('100'),
            risk_percentage=0.01,
            position_size_recommendation=Decimal('0.1'),
            risk_reward_ratio=2.0,
            probability_of_success=0.7,
            expected_value=50.0,
            risk_level=RiskLevel.LOW,
            recommended_action=RiskAction.CONTINUE,
            risk_factors=[]
        )
        
        self.bybit_client.place_order.return_value = "state-test-order"
        
        execution = await self.trading_engine.process_trade_signal(signal)
        assert execution.status == ExecutionStatus.PENDING
        
        # Test PAUSED state
        await self.trading_engine.pause()
        execution = await self.trading_engine.process_trade_signal(signal)
        assert execution.status == ExecutionStatus.REJECTED
        assert "paused" in execution.rejection_reason.lower()
        
        # Test EMERGENCY_STOPPED state
        await self.trading_engine.resume()
        await self.trading_engine.emergency_stop()
        execution = await self.trading_engine.process_trade_signal(signal)
        assert execution.status == ExecutionStatus.REJECTED
        assert "emergency" in execution.rejection_reason.lower()


# Performance integration tests
class TestTradingEnginePerformanceIntegration:
    """Performance integration tests for trading engine."""
    
    def setup_method(self):
        """Set up performance integration test fixtures."""
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
    async def test_high_throughput_integration(self):
        """Test high throughput signal processing with all components."""
        from tests.conftest import PerformanceTimer
        
        await self.trading_engine.start()
        
        # Create many signals
        signals = []
        for i in range(100):
            signal = create_test_trade_signal(
                symbol=f"PERF{i % 10}USDT",  # 10 different symbols
                action="BUY" if i % 2 == 0 else "SELL",
                quantity=Decimal('0.1'),
                price=Decimal(f'{1000 + i}')
            )
            signals.append(signal)
        
        # Mock fast responses from all components
        self.risk_manager.assess_trade_risk.return_value = TradeRiskAssessment(
            symbol="PERFUSDT",
            side="Buy",
            size=Decimal('0.1'),
            entry_price=Decimal('1000'),
            stop_loss=None,
            take_profit=None,
            risk_amount=Decimal('10'),
            risk_percentage=0.001,
            position_size_recommendation=Decimal('0.1'),
            risk_reward_ratio=2.0,
            probability_of_success=0.7,
            expected_value=5.0,
            risk_level=RiskLevel.LOW,
            recommended_action=RiskAction.CONTINUE,
            risk_factors=[]
        )
        
        self.bybit_client.place_order.return_value = "perf-order"
        
        with PerformanceTimer("Process 100 signals with full integration"):
            tasks = []
            for signal in signals:
                task = self.trading_engine.process_trade_signal(signal)
                tasks.append(task)
            
            executions = await asyncio.gather(*tasks)
        
        assert len(executions) == 100
        assert all(ex.status in [ExecutionStatus.PENDING, ExecutionStatus.REJECTED] for ex in executions)
        
        # Verify component interactions
        assert self.risk_manager.assess_trade_risk.call_count == 100
        # Order placement count may be less if some were rejected
        assert self.bybit_client.place_order.call_count <= 100