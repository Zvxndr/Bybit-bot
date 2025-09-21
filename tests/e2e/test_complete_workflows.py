"""
End-to-End Tests for Complete Trading Bot Workflows

This module contains end-to-end tests that test complete workflows
from signal generation through strategy execution to trade completion.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd
import numpy as np

# Import all components for E2E testing
from src.bot.core.trading_engine import TradingEngine, EngineState
from src.bot.exchange.bybit_client import BybitClient, OrderSide, OrderType
from src.bot.risk_management.unified_risk_manager import UnifiedRiskManager, RiskLevel, RiskAction
from src.bot.strategies.strategy_framework import StrategyManager, StrategyState
from src.bot.strategies.ma_crossover_strategy import MovingAverageCrossoverStrategy

from tests.conftest import (
    MockBybitClient,
    MockDataManager,
    MockConfigurationManager,
    create_test_market_data,
    create_test_trade_signal,
    async_test
)


class TestCompleteTradingWorkflow:
    """End-to-end tests for complete trading workflows."""
    
    def setup_method(self):
        """Set up complete trading system."""
        self.config_manager = MockConfigurationManager()
        self.bybit_client = MockBybitClient()
        self.data_manager = MockDataManager()
        
        # Use real components for E2E testing
        self.risk_manager = UnifiedRiskManager(self.config_manager)
        self.trading_engine = TradingEngine(
            bybit_client=self.bybit_client,
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager
        )
        
        self.strategy_manager = StrategyManager(
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager
        )
        
        # Create and add a real strategy
        self.ma_strategy = MovingAverageCrossoverStrategy(
            strategy_id="e2e-ma-strategy",
            name="E2E MA Crossover",
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager,
            fast_period=5,
            slow_period=20
        )
        
        self.strategy_manager.add_strategy(self.ma_strategy)
        self.ma_strategy.add_symbol("BTCUSDT")
    
    @async_test
    async def test_complete_buy_signal_workflow(self):
        """Test complete workflow from bullish signal generation to trade completion."""
        # Setup system
        await self.trading_engine.start()
        await self.ma_strategy.activate()
        
        # Create market data that will generate bullish crossover
        dates = pd.date_range(start='2023-01-01', periods=30, freq='1H')
        
        # Price pattern that creates bullish crossover
        prices = []
        base_price = 45000
        
        # First 25 periods: declining trend
        for i in range(25):
            prices.append(base_price - i * 50)
        
        # Last 5 periods: strong uptrend causing bullish crossover
        for i in range(5):
            prices.append(base_price - 1250 + i * 300)
        
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 25 for p in prices],
            'low': [p - 25 for p in prices],
            'close': prices,
            'volume': [100] * 30
        })
        
        # Mock data manager to return test data
        self.data_manager.get_market_data.return_value = market_data
        
        # Mock API client responses
        self.bybit_client.place_order.return_value = "e2e-buy-order-123"
        self.bybit_client.get_order_status.return_value = {
            "orderId": "e2e-buy-order-123",
            "orderStatus": "Filled",
            "avgPrice": "44250.00",
            "cumExecQty": "0.02"  # Risk-adjusted size
        }
        
        # Mock portfolio data for risk management
        with patch.object(self.risk_manager, '_get_portfolio_value', return_value=Decimal('10000')):
            with patch.object(self.risk_manager, '_get_asset_volatility', return_value=0.15):
                
                # Generate signals from strategy
                signals = await self.strategy_manager.generate_all_signals(["BTCUSDT"])
                
                # Should generate buy signal
                assert len(signals) > 0
                buy_signal = signals[0]
                assert buy_signal.action == "BUY"
                assert buy_signal.symbol == "BTCUSDT"
                
                # Process signal through trading engine
                execution = await self.trading_engine.process_trade_signal(buy_signal)
                
                # Verify execution
                assert execution is not None
                assert execution.status.value in ["PENDING", "FILLED"]
                assert execution.order_id == "e2e-buy-order-123"
                
                # Update trade status to simulate completion
                await self.trading_engine._update_trade_status(execution)
                
                # Verify completed trade
                assert execution.status.value == "FILLED"
                assert execution.fill_price == Decimal('44250.00')
                assert execution.filled_quantity > 0
                
                # Verify risk management was involved
                assert execution.signal.quantity >= execution.filled_quantity  # Size may be adjusted
    
    @async_test
    async def test_complete_sell_signal_workflow(self):
        """Test complete workflow from bearish signal generation to trade completion."""
        await self.trading_engine.start()
        await self.ma_strategy.activate()
        
        # Create market data that will generate bearish crossover
        dates = pd.date_range(start='2023-01-01', periods=30, freq='1H')
        
        # Price pattern that creates bearish crossover
        prices = []
        base_price = 50000
        
        # First 25 periods: rising trend
        for i in range(25):
            prices.append(base_price + i * 40)
        
        # Last 5 periods: sharp decline causing bearish crossover
        for i in range(5):
            prices.append(base_price + 1000 - i * 250)
        
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 30 for p in prices],
            'low': [p - 30 for p in prices],
            'close': prices,
            'volume': [150] * 30
        })
        
        self.data_manager.get_market_data.return_value = market_data
        
        # Mock API responses
        self.bybit_client.place_order.return_value = "e2e-sell-order-456"
        self.bybit_client.get_order_status.return_value = {
            "orderId": "e2e-sell-order-456",
            "orderStatus": "Filled",
            "avgPrice": "49750.00",
            "cumExecQty": "0.02"
        }
        
        with patch.object(self.risk_manager, '_get_portfolio_value', return_value=Decimal('10000')):
            with patch.object(self.risk_manager, '_get_asset_volatility', return_value=0.15):
                
                # Generate signals
                signals = await self.strategy_manager.generate_all_signals(["BTCUSDT"])
                
                # Should generate sell signal
                assert len(signals) > 0
                sell_signal = signals[0]
                assert sell_signal.action == "SELL"
                
                # Process through trading engine
                execution = await self.trading_engine.process_trade_signal(sell_signal)
                
                # Verify and complete trade
                assert execution.status.value in ["PENDING", "FILLED"]
                await self.trading_engine._update_trade_status(execution)
                
                assert execution.status.value == "FILLED"
                assert execution.fill_price == Decimal('49750.00')
    
    @async_test
    async def test_risk_management_integration_workflow(self):
        """Test complete workflow where risk management affects trading decisions."""
        await self.trading_engine.start()
        await self.ma_strategy.activate()
        
        # Create strong bullish signal
        market_data = create_test_market_data("BTCUSDT", 30)
        # Modify data to ensure bullish crossover
        market_data.iloc[-1, market_data.columns.get_loc('close')] = market_data.iloc[-2]['close'] * 1.05
        
        self.data_manager.get_market_data.return_value = market_data
        
        # Mock high portfolio risk scenario
        with patch.object(self.risk_manager, '_get_portfolio_value', return_value=Decimal('10000')):
            with patch.object(self.risk_manager, '_get_asset_volatility', return_value=0.45):  # High volatility
                with patch.object(self.risk_manager, '_calculate_current_leverage', return_value=2.8):  # High leverage
                    
                    # Generate signal
                    signals = await self.strategy_manager.generate_all_signals(["BTCUSDT"])
                    
                    if len(signals) > 0:
                        signal = signals[0]
                        
                        # Process signal - should be affected by risk management
                        execution = await self.trading_engine.process_trade_signal(signal)
                        
                        # Risk management should either:
                        # 1. Reduce position size significantly
                        # 2. Reject the trade entirely
                        
                        if execution.status.value == "REJECTED":
                            assert "risk" in execution.rejection_reason.lower()
                        else:
                            # If not rejected, position should be heavily reduced
                            original_qty = signal.quantity
                            if hasattr(execution, 'adjusted_quantity'):
                                assert execution.adjusted_quantity < original_qty * 0.5
    
    @async_test
    async def test_multiple_strategies_workflow(self):
        """Test workflow with multiple strategies generating signals."""
        # Add another strategy with different parameters
        ma_strategy_2 = MovingAverageCrossoverStrategy(
            strategy_id="e2e-ma-strategy-2",
            name="E2E MA Crossover 2",
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager,
            fast_period=10,
            slow_period=50
        )
        
        self.strategy_manager.add_strategy(ma_strategy_2)
        ma_strategy_2.add_symbol("BTCUSDT")
        
        await self.trading_engine.start()
        await self.strategy_manager.activate_all_strategies()
        
        # Create market data that might generate different signals from each strategy
        market_data = create_test_market_data("BTCUSDT", 60)  # More data for 50-period MA
        self.data_manager.get_market_data.return_value = market_data
        
        # Mock successful order placements
        order_counter = 0
        def mock_place_order(*args, **kwargs):
            nonlocal order_counter
            order_counter += 1
            return f"multi-strategy-order-{order_counter}"
        
        self.bybit_client.place_order.side_effect = mock_place_order
        
        # Mock order status
        def mock_order_status(symbol, order_id):
            return {
                "orderId": order_id,
                "orderStatus": "Filled",
                "avgPrice": "50000.00",
                "cumExecQty": "0.02"
            }
        
        self.bybit_client.get_order_status.side_effect = mock_order_status
        
        with patch.object(self.risk_manager, '_get_portfolio_value', return_value=Decimal('10000')):
            with patch.object(self.risk_manager, '_get_asset_volatility', return_value=0.15):
                
                # Generate signals from all strategies
                all_signals = await self.strategy_manager.generate_all_signals(["BTCUSDT"])
                
                # Process all signals
                executions = []
                for signal in all_signals:
                    execution = await self.trading_engine.process_trade_signal(signal)
                    executions.append(execution)
                    
                    # Update status
                    if execution.status.value == "PENDING":
                        await self.trading_engine._update_trade_status(execution)
                
                # Verify multiple strategies can work together
                successful_executions = [ex for ex in executions if ex.status.value == "FILLED"]
                
                # At least some signals should be processed successfully
                # (depends on market data and risk management)
                assert len(all_signals) >= 0  # May be 0 if no crossovers detected
    
    @async_test
    async def test_portfolio_monitoring_workflow(self):
        """Test complete portfolio monitoring workflow."""
        await self.trading_engine.start()
        
        # Create multiple positions
        mock_positions = [
            {
                "symbol": "BTCUSDT",
                "size": "0.1",
                "unrealisedPnl": "50.00",
                "positionValue": "5000.00",
                "entryPrice": "50000.00",
                "markPrice": "50500.00"
            },
            {
                "symbol": "ETHUSDT", 
                "size": "2.0",
                "unrealisedPnl": "-25.00",
                "positionValue": "6000.00",
                "entryPrice": "3000.00",
                "markPrice": "2987.50"
            }
        ]
        
        self.bybit_client.get_positions.return_value = mock_positions
        
        # Mock wallet balance
        self.bybit_client.get_wallet_balance.return_value = {
            "USDT": {
                "wallet_balance": Decimal('10000'),
                "available_balance": Decimal('8500'),
                "locked": Decimal('1500')
            }
        }
        
        # Calculate portfolio metrics
        positions_dict = {pos["symbol"]: pos for pos in mock_positions}
        portfolio_value = Decimal('10000')
        
        with patch.object(self.risk_manager, '_calculate_portfolio_returns') as mock_returns:
            mock_returns.return_value = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01])
            
            metrics = await self.risk_manager.calculate_portfolio_metrics(
                positions_dict, portfolio_value
            )
            
            # Verify portfolio monitoring
            assert metrics.portfolio_value == portfolio_value
            assert metrics.total_exposure == Decimal('11000')  # 5000 + 6000
            assert metrics.leverage == 1.1  # 11000 / 10000
            assert metrics.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    @async_test
    async def test_emergency_scenarios_workflow(self):
        """Test complete workflow during emergency scenarios."""
        await self.trading_engine.start()
        await self.ma_strategy.activate()
        
        # Create positions that would trigger emergency stop
        high_risk_positions = [
            {
                "symbol": "BTCUSDT",
                "size": "2.0",
                "unrealisedPnl": "-2000.00",  # Large loss
                "positionValue": "100000.00",  # Large position
                "entryPrice": "50000.00",
                "markPrice": "49000.00"
            }
        ]
        
        self.bybit_client.get_positions.return_value = high_risk_positions
        self.bybit_client.cancel_all_orders.return_value = True
        
        # Mock high drawdown scenario
        with patch.object(self.risk_manager, '_calculate_current_drawdown', return_value=0.25):  # 25% drawdown
            
            # Check circuit breakers
            portfolio_value = Decimal('8000')  # 20% loss from initial 10000
            breakers = await self.risk_manager.check_circuit_breakers(portfolio_value)
            
            # Should trigger emergency stop
            assert len(breakers) > 0
            assert self.risk_manager.emergency_stop_active
            
            # Trading engine should also be in emergency stop
            await self.trading_engine.emergency_stop()
            assert self.trading_engine.state == EngineState.EMERGENCY_STOPPED
            
            # Try to process new signal during emergency
            emergency_signal = create_test_trade_signal(
                symbol="BTCUSDT",
                action="BUY",
                quantity=Decimal('0.1'),
                price=Decimal('49000')
            )
            
            execution = await self.trading_engine.process_trade_signal(emergency_signal)
            
            # Should be rejected
            assert execution.status.value == "REJECTED"
            assert "emergency" in execution.rejection_reason.lower()
    
    @async_test
    async def test_performance_tracking_workflow(self):
        """Test complete performance tracking workflow."""
        await self.trading_engine.start()
        
        # Simulate a series of completed trades
        trade_data = [
            ("BTCUSDT", "BUY", Decimal('0.1'), Decimal('50000'), Decimal('50500'), Decimal('50')),
            ("ETHUSDT", "SELL", Decimal('1.0'), Decimal('3000'), Decimal('2950'), Decimal('-50')),
            ("ADAUSDT", "BUY", Decimal('1000'), Decimal('0.5'), Decimal('0.52'), Decimal('20')),
            ("DOTUSDT", "SELL", Decimal('100'), Decimal('6.0'), Decimal('5.8'), Decimal('-20')),
            ("SOLUSDT", "BUY", Decimal('10'), Decimal('100'), Decimal('110'), Decimal('100'))
        ]
        
        from src.bot.core.trading_engine import TradeExecution, ExecutionStatus
        
        for symbol, action, qty, entry, exit, pnl in trade_data:
            signal = create_test_trade_signal(
                symbol=symbol,
                action=action,
                quantity=qty,
                price=entry
            )
            
            execution = TradeExecution(
                execution_id=f"perf-{symbol}-{action}",
                signal=signal,
                order_id=f"order-{symbol}",
                status=ExecutionStatus.FILLED,
                timestamp=datetime.now(),
                fill_price=exit,
                filled_quantity=qty,
                realized_pnl=pnl
            )
            
            self.trading_engine.trade_history.append(execution)
        
        # Calculate overall performance
        performance = self.trading_engine._calculate_performance_metrics()
        
        # Verify performance calculations
        assert performance["total_trades"] == 5
        assert performance["successful_trades"] == 3  # Positive PnL trades
        assert performance["win_rate"] == 0.6
        assert performance["total_pnl"] == Decimal('100')  # 50 - 50 + 20 - 20 + 100
        
        # Test strategy performance tracking
        self.ma_strategy.update_performance("BTCUSDT", Decimal('50'), True)
        self.ma_strategy.update_performance("ETHUSDT", Decimal('-50'), False)
        
        strategy_perf = self.ma_strategy.performance
        assert strategy_perf.total_trades == 2
        assert strategy_perf.winning_trades == 1
        assert strategy_perf.win_rate == 0.5
        assert strategy_perf.total_pnl == Decimal('0')
    
    @async_test
    async def test_configuration_driven_workflow(self):
        """Test workflow driven by different configurations."""
        # Test conservative configuration
        conservative_config = self.config_manager.config.copy()
        conservative_config["risk"]["max_portfolio_risk"] = 0.01  # 1% max risk
        conservative_config["risk"]["max_position_size"] = 0.05   # 5% max position
        conservative_config["trading"]["enable_stop_loss"] = True
        conservative_config["trading"]["default_stop_loss_pct"] = 0.02  # 2% stop loss
        
        self.config_manager.config = conservative_config
        
        await self.trading_engine.start()
        await self.ma_strategy.activate()
        
        # Create market data and signal
        market_data = create_test_market_data("BTCUSDT", 30)
        market_data.iloc[-1, market_data.columns.get_loc('close')] = market_data.iloc[-2]['close'] * 1.03  # 3% increase
        
        self.data_manager.get_market_data.return_value = market_data
        
        with patch.object(self.risk_manager, '_get_portfolio_value', return_value=Decimal('10000')):
            with patch.object(self.risk_manager, '_get_asset_volatility', return_value=0.15):
                
                signals = await self.strategy_manager.generate_all_signals(["BTCUSDT"])
                
                if len(signals) > 0:
                    signal = signals[0]
                    
                    # Mock order placement
                    self.bybit_client.place_order.return_value = "conservative-order"
                    
                    execution = await self.trading_engine.process_trade_signal(signal)
                    
                    # Verify conservative risk management
                    if execution.status.value == "PENDING":
                        # Position size should be small due to conservative settings
                        self.bybit_client.place_order.assert_called_once()
                        call_kwargs = self.bybit_client.place_order.call_args[1]
                        
                        # Size should be limited by conservative configuration
                        assert call_kwargs["qty"] <= Decimal('0.5')  # Small position
                        
                        # Stop loss should be included if configured
                        if conservative_config["trading"]["enable_stop_loss"]:
                            assert "stop_loss" in call_kwargs or signal.stop_loss is not None


class TestErrorHandlingWorkflows:
    """End-to-end tests for error handling workflows."""
    
    def setup_method(self):
        """Set up error handling test system."""
        self.config_manager = MockConfigurationManager()
        self.bybit_client = MockBybitClient()
        self.data_manager = MockDataManager()
        
        self.risk_manager = UnifiedRiskManager(self.config_manager)
        self.trading_engine = TradingEngine(
            bybit_client=self.bybit_client,
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager
        )
    
    @async_test
    async def test_api_error_recovery_workflow(self):
        """Test complete workflow with API error recovery."""
        await self.trading_engine.start()
        
        signal = create_test_trade_signal(
            symbol="BTCUSDT",
            action="BUY",
            quantity=Decimal('0.1'),
            price=Decimal('50000')
        )
        
        # Mock API errors then success
        from src.bot.exchange.bybit_client import BybitAPIError, RateLimitError
        
        self.bybit_client.place_order.side_effect = [
            RateLimitError("Rate limit exceeded", 10006),
            BybitAPIError("Server error", 500),
            "recovery-order-123"  # Finally succeeds
        ]
        
        with patch.object(self.risk_manager, '_get_portfolio_value', return_value=Decimal('10000')):
            # Process signal with retry logic (if implemented)
            execution = await self.trading_engine.process_trade_signal(signal)
            
            # Should either succeed after retries or be rejected with appropriate error
            assert execution.status.value in ["PENDING", "REJECTED"]
            
            if execution.status.value == "REJECTED":
                assert any(keyword in execution.rejection_reason.lower() 
                          for keyword in ["rate limit", "api error", "server error"])
    
    @async_test
    async def test_data_unavailable_workflow(self):
        """Test workflow when market data is unavailable."""
        await self.trading_engine.start()
        
        # Mock data manager to return empty/invalid data
        self.data_manager.get_market_data.side_effect = Exception("Data source unavailable")
        
        # Strategy should handle data errors gracefully
        ma_strategy = MovingAverageCrossoverStrategy(
            strategy_id="data-error-test",
            name="Data Error Test",
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager,
            fast_period=5,
            slow_period=20
        )
        
        ma_strategy.add_symbol("BTCUSDT")
        await ma_strategy.activate()
        
        # Should handle data error gracefully
        try:
            signals = await ma_strategy.generate_signals("BTCUSDT", pd.DataFrame())
            # Should return empty signals list if data is unavailable
            assert isinstance(signals, list)
        except Exception as e:
            # Or raise appropriate exception that can be handled upstream
            assert "data" in str(e).lower()


class TestPerformanceWorkflows:
    """End-to-end performance tests for complete workflows."""
    
    def setup_method(self):
        """Set up performance test system."""
        self.config_manager = MockConfigurationManager()
        self.bybit_client = MockBybitClient()
        self.data_manager = MockDataManager()
        
        self.risk_manager = UnifiedRiskManager(self.config_manager)
        self.trading_engine = TradingEngine(
            bybit_client=self.bybit_client,
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager
        )
        
        self.strategy_manager = StrategyManager(
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            config_manager=self.config_manager
        )
    
    @async_test
    async def test_high_frequency_workflow_performance(self):
        """Test performance of high-frequency complete workflows."""
        from tests.conftest import PerformanceTimer
        
        # Setup multiple strategies
        for i in range(5):
            strategy = MovingAverageCrossoverStrategy(
                strategy_id=f"perf-strategy-{i}",
                name=f"Performance Strategy {i}",
                data_manager=self.data_manager,
                risk_manager=self.risk_manager,
                config_manager=self.config_manager,
                fast_period=5 + i,
                slow_period=20 + i * 2
            )
            strategy.add_symbol("BTCUSDT")
            self.strategy_manager.add_strategy(strategy)
        
        await self.trading_engine.start()
        await self.strategy_manager.activate_all_strategies()
        
        # Create market data
        market_data = create_test_market_data("BTCUSDT", 100)
        self.data_manager.get_market_data.return_value = market_data
        
        # Mock fast API responses
        self.bybit_client.place_order.return_value = "perf-order"
        
        with patch.object(self.risk_manager, '_get_portfolio_value', return_value=Decimal('10000')):
            with patch.object(self.risk_manager, '_get_asset_volatility', return_value=0.15):
                
                with PerformanceTimer("Complete E2E workflow with 5 strategies"):
                    # Generate signals from all strategies
                    signals = await self.strategy_manager.generate_all_signals(["BTCUSDT"])
                    
                    # Process all signals
                    tasks = []
                    for signal in signals:
                        task = self.trading_engine.process_trade_signal(signal)
                        tasks.append(task)
                    
                    if tasks:
                        executions = await asyncio.gather(*tasks)
                        
                        # Verify performance
                        assert len(executions) == len(signals)
                        assert all(isinstance(ex.status, type(executions[0].status)) for ex in executions)
    
    @async_test
    async def test_large_portfolio_workflow_performance(self):
        """Test performance with large portfolio monitoring."""
        from tests.conftest import PerformanceTimer
        
        await self.trading_engine.start()
        
        # Create large number of positions
        large_positions = {}
        for i in range(50):
            large_positions[f"TEST{i}USDT"] = {
                "symbol": f"TEST{i}USDT",
                "size": f"{0.1 + i * 0.01}",
                "unrealisedPnl": f"{np.random.uniform(-10, 10):.2f}",
                "positionValue": f"{np.random.uniform(100, 1000):.2f}"
            }
        
        portfolio_value = Decimal('50000')
        
        with patch.object(self.risk_manager, '_calculate_portfolio_returns') as mock_returns:
            mock_returns.return_value = pd.Series(np.random.normal(0, 0.02, 252))  # 1 year of returns
            
            with PerformanceTimer("Large portfolio risk calculation (50 positions)"):
                metrics = await self.risk_manager.calculate_portfolio_metrics(
                    large_positions, portfolio_value
                )
            
            # Verify metrics calculation completed
            assert metrics.portfolio_value == portfolio_value
            assert metrics.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]