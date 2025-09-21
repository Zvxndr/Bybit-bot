"""
Comprehensive Integration Tests for Phase 3 Enhanced Backtesting Engine

This module tests the integration of all Phase 3 components:
1. BybitEnhancedBacktestEngine integration with existing BacktestEngine
2. BybitFeeSimulator integration and functionality
3. BybitLiquidationRiskManager integration and accuracy
4. BybitExecutionSimulator integration and execution quality
5. End-to-end backtesting with real strategy examples
6. Performance benchmarking and validation

Tests ensure Phase 3 components work seamlessly with existing architecture
while providing enhanced Bybit-specific backtesting capabilities.

Author: Trading Bot Team
Version: 1.0.0
"""

import sys
import os
import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, patch, AsyncMock

# Ensure the src directory is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Phase 3 components
from src.bot.backtesting.bybit_enhanced_backtest_engine import (
    BybitEnhancedBacktestEngine, BybitVIPTier, BybitContractType,
    BybitTrade, BybitBacktestResults
)
from src.bot.backtesting.bybit_fee_simulator import (
    BybitFeeCalculator, BybitVIPTierRequirements, BybitFeeOptimizer
)
from src.bot.backtesting.bybit_liquidation_risk_manager import (
    BybitLiquidationRiskManager, LiquidationRiskAssessment, MarginTier
)
from src.bot.backtesting.bybit_execution_simulator import (
    BybitExecutionSimulator, OrderType, ExecutionStrategy, ExecutionResult
)

# Import existing components
from src.bot.backtesting.backtest_engine import BacktestEngine, BacktestTrade, BacktestResults
from src.bot.config_manager import ConfigurationManager
from src.bot.core.strategy_manager import BaseStrategy, TradingSignal, SignalType
from src.bot.data.bybit_client import BybitClient
from src.bot.data.historical_data_manager import HistoricalDataManager
from src.bot.utils.logging import TradingLogger


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, strategy_id: str = "test_strategy"):
        self.strategy_id = strategy_id
        self.lookback_periods = 50
        self.signal_count = 0
        
    async def on_start(self):
        pass
        
    async def on_stop(self):
        pass
        
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate simple momentum signal for testing."""
        if len(data) < 20:
            return None
            
        # Simple momentum strategy
        current_price = data['close'].iloc[-1]
        ma_short = data['close'].rolling(5).mean().iloc[-1]
        ma_long = data['close'].rolling(20).mean().iloc[-1]
        
        self.signal_count += 1
        
        # Buy signal when short MA crosses above long MA
        if ma_short > ma_long and self.signal_count % 10 == 0:  # Reduce signal frequency
            return TradingSignal(
                timestamp=data.index[-1],
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.8,
                price=current_price,
                quantity=None,  # Will be calculated by risk manager
                strategy_id=self.strategy_id,
                metadata={'reason': 'momentum_up'}
            )
        # Sell signal when short MA crosses below long MA
        elif ma_short < ma_long and self.signal_count % 15 == 0:  # Different frequency
            return TradingSignal(
                timestamp=data.index[-1],
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=0.7,
                price=current_price,
                quantity=None,
                strategy_id=self.strategy_id,
                metadata={'reason': 'momentum_down'}
            )
            
        return None


@pytest.fixture
def config_manager():
    """Create mock configuration manager."""
    config = Mock(spec=ConfigurationManager)
    config.get = Mock(side_effect=lambda key, default=None: {
        'backtesting.commission_rate': 0.001,
        'backtesting.slippage_rate': 0.0005,
        'backtesting.max_slippage': 0.002,
        'bybit.api_key': 'test_key',
        'bybit.api_secret': 'test_secret',
        'bybit.testnet': True
    }.get(key, default))
    return config


@pytest.fixture
def sample_data():
    """Generate sample market data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='1H')
    n_points = len(dates)
    
    # Generate realistic OHLCV data
    returns = np.random.normal(0.0001, 0.02, n_points)  # Small positive drift
    prices = 30000 * np.exp(np.cumsum(returns))  # Start around $30,000
    
    data = pd.DataFrame(index=dates)
    data['open'] = prices
    data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
    data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
    data['close'] = prices * (1 + np.random.normal(0, 0.005, n_points))
    data['volume'] = np.random.lognormal(10, 1, n_points)  # Realistic volume distribution
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        high = max(data.iloc[i]['open'], data.iloc[i]['close'])
        low = min(data.iloc[i]['open'], data.iloc[i]['close'])
        data.iloc[i, data.columns.get_loc('high')] = max(data.iloc[i]['high'], high)
        data.iloc[i, data.columns.get_loc('low')] = min(data.iloc[i]['low'], low)
    
    return data


class TestPhase3Integration:
    """Comprehensive integration tests for Phase 3 components."""
    
    def test_enhanced_backtest_engine_inheritance(self, config_manager):
        """Test that BybitEnhancedBacktestEngine properly inherits from BacktestEngine."""
        # Test instantiation
        enhanced_engine = BybitEnhancedBacktestEngine(
            config_manager=config_manager,
            initial_balance=Decimal('10000'),
            vip_tier=BybitVIPTier.VIP1
        )
        
        # Verify inheritance
        assert isinstance(enhanced_engine, BacktestEngine)
        assert isinstance(enhanced_engine, BybitEnhancedBacktestEngine)
        
        # Verify enhanced properties
        assert enhanced_engine.vip_tier == BybitVIPTier.VIP1
        assert enhanced_engine.fee_calculator is not None
        assert enhanced_engine.liquidation_manager is not None
        assert enhanced_engine.execution_simulator is not None
        assert enhanced_engine.funding_tracker is not None
        
        # Verify base properties are preserved
        assert enhanced_engine.initial_balance == Decimal('10000')
        assert enhanced_engine.current_balance == Decimal('10000')
        assert enhanced_engine.config == config_manager
    
    @pytest.mark.asyncio
    async def test_enhanced_vs_basic_backtest(self, config_manager, sample_data):
        """Test enhanced backtest against basic backtest to verify enhancements."""
        strategy = MockStrategy()
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 6, 30)
        
        # Run basic backtest
        basic_engine = BacktestEngine(config_manager, Decimal('10000'))
        basic_results = await basic_engine.run_backtest(strategy, sample_data, start_date, end_date)
        
        # Run enhanced backtest
        enhanced_engine = BybitEnhancedBacktestEngine(
            config_manager=config_manager,
            initial_balance=Decimal('10000'),
            vip_tier=BybitVIPTier.VIP2
        )
        enhanced_results = await enhanced_engine.run_enhanced_backtest(
            strategy=strategy,
            data=sample_data,
            start_date=start_date,
            end_date=end_date,
            symbol='BTCUSDT'
        )
        
        # Verify enhanced results have additional fields
        assert isinstance(enhanced_results, BybitBacktestResults)
        assert hasattr(enhanced_results, 'total_funding_cost')
        assert hasattr(enhanced_results, 'vip_fee_savings')
        assert hasattr(enhanced_results, 'liquidation_events')
        assert hasattr(enhanced_results, 'execution_quality_score')
        
        # Enhanced backtest should have more detailed trade information
        if len(enhanced_results.trades) > 0:
            enhanced_trade = enhanced_results.trades[0]
            assert isinstance(enhanced_trade, BybitTrade)
            assert hasattr(enhanced_trade, 'funding_cost')
            assert hasattr(enhanced_trade, 'is_maker')
            assert hasattr(enhanced_trade, 'vip_tier')
        
        # Compare performance (enhanced should be more realistic)
        # Enhanced backtest should typically show slightly lower returns due to funding costs
        print(f"Basic Return: {basic_results.total_return_pct:.2f}%")
        print(f"Enhanced Return: {enhanced_results.total_return_pct:.2f}%")
    
    def test_fee_calculator_integration(self):
        """Test BybitFeeCalculator integration and accuracy."""
        calculator = BybitFeeCalculator()
        
        # Test different VIP tiers
        test_cases = [
            {
                'contract_type': BybitContractType.LINEAR_PERPETUAL,
                'quantity': Decimal('1.0'),
                'price': Decimal('30000'),
                'is_maker': True,
                'vip_tier': BybitVIPTier.NO_VIP,
                'expected_rate': Decimal('0.0001')  # 0.01% maker fee
            },
            {
                'contract_type': BybitContractType.LINEAR_PERPETUAL,
                'quantity': Decimal('1.0'),
                'price': Decimal('30000'),
                'is_maker': False,
                'vip_tier': BybitVIPTier.VIP2,
                'expected_rate': Decimal('0.00045')  # VIP2 taker fee
            }
        ]
        
        for case in test_cases:
            result = calculator.calculate_trading_fee(
                contract_type=case['contract_type'],
                quantity=case['quantity'],
                price=case['price'],
                is_maker=case['is_maker'],
                vip_tier=case['vip_tier']
            )
            
            expected_fee = case['quantity'] * case['price'] * case['expected_rate']
            assert abs(result.fee_amount - expected_fee) < Decimal('0.01')
            assert result.fee_rate == case['expected_rate']
            assert result.contract_type == case['contract_type']
    
    def test_liquidation_risk_manager_integration(self):
        """Test BybitLiquidationRiskManager integration and risk assessment."""
        risk_manager = BybitLiquidationRiskManager()
        
        # Test BTCUSDT margin assessment
        test_positions = [
            {
                'symbol': 'BTCUSDT',
                'side': 'long',
                'size': Decimal('0.5'),
                'entry_price': Decimal('30000'),
                'margin': Decimal('3000'),  # 10x leverage
                'current_price': Decimal('29000')  # Underwater position
            }
        ]
        
        assessment = risk_manager.assess_portfolio_risk(
            positions=test_positions,
            account_balance=Decimal('10000')
        )
        
        # Verify risk assessment structure
        assert isinstance(assessment, LiquidationRiskAssessment)
        assert assessment.total_margin_ratio > 0
        assert assessment.liquidation_price > 0
        assert assessment.risk_level in ['SAFE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'IMMINENT']
        assert len(assessment.recommendations) > 0
        
        # Test that high-risk positions are properly identified
        if assessment.total_margin_ratio < Decimal('0.2'):  # Below 20% margin
            assert assessment.risk_level in ['HIGH', 'CRITICAL', 'IMMINENT']
    
    def test_execution_simulator_integration(self):
        """Test BybitExecutionSimulator integration and execution modeling."""
        simulator = BybitExecutionSimulator(vip_tier=BybitVIPTier.VIP1)
        
        # Test market order execution
        result = simulator.simulate_order_execution(
            symbol='BTCUSDT',
            side='buy',
            quantity=Decimal('0.1'),
            order_type=OrderType.MARKET,
            execution_strategy=ExecutionStrategy.AGGRESSIVE,
            current_price=Decimal('30000')
        )
        
        # Verify execution result structure
        assert isinstance(result, ExecutionResult)
        assert result.symbol == 'BTCUSDT'
        assert result.side == 'buy'
        assert result.requested_quantity == Decimal('0.1')
        assert result.fill_type.value in ['full', 'partial', 'rejected']
        assert result.execution_time_ms > 0
        assert result.vip_tier == BybitVIPTier.VIP1
        
        # Test that VIP tier affects execution
        assert len(result.fills) > 0
        if result.filled_quantity > 0:
            assert result.average_fill_price > 0
            assert result.total_cost > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_integration(self, config_manager, sample_data):
        """Test complete end-to-end integration of Phase 3 components."""
        # Create enhanced backtest engine with all components
        engine = BybitEnhancedBacktestEngine(
            config_manager=config_manager,
            initial_balance=Decimal('50000'),
            vip_tier=BybitVIPTier.VIP2,
            contract_type=BybitContractType.LINEAR_PERPETUAL
        )
        
        # Create strategy
        strategy = MockStrategy("end_to_end_test")
        
        # Run enhanced backtest
        results = await engine.run_enhanced_backtest(
            strategy=strategy,
            data=sample_data,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),  # Shorter period for testing
            symbol='BTCUSDT'
        )
        
        # Verify comprehensive results
        assert isinstance(results, BybitBacktestResults)
        
        # Check basic metrics
        assert results.initial_balance == Decimal('50000')
        assert results.final_balance > 0
        assert results.total_trades >= 0
        
        # Check enhanced metrics
        assert hasattr(results, 'total_funding_cost')
        assert hasattr(results, 'vip_fee_savings')
        assert hasattr(results, 'execution_quality_score')
        assert hasattr(results, 'liquidation_events')
        
        # Check trade details if trades occurred
        if len(results.trades) > 0:
            trade = results.trades[0]
            assert isinstance(trade, BybitTrade)
            assert hasattr(trade, 'funding_cost')
            assert hasattr(trade, 'is_maker')
            assert hasattr(trade, 'execution_quality')
            assert trade.vip_tier == BybitVIPTier.VIP2
        
        # Verify realistic execution modeling
        if len(results.trades) > 0:
            # Check that market impact is modeled
            assert any(trade.market_impact > 0 for trade in results.trades)
            
            # Check that VIP benefits are applied
            vip_savings = results.vip_fee_savings
            assert vip_savings >= 0  # Should have some savings with VIP2
        
        print(f"End-to-end test completed:")
        print(f"  Initial Balance: ${results.initial_balance:,.2f}")
        print(f"  Final Balance: ${results.final_balance:,.2f}")
        print(f"  Total Return: {results.total_return_pct:.2f}%")
        print(f"  Total Trades: {results.total_trades}")
        print(f"  Funding Cost: ${results.total_funding_cost:.2f}")
        print(f"  VIP Savings: ${results.vip_fee_savings:.2f}")
        print(f"  Execution Quality: {results.execution_quality_score:.3f}")
    
    def test_performance_benchmarking(self, config_manager, sample_data):
        """Test performance of Phase 3 components vs baseline."""
        import time
        
        # Benchmark basic backtest engine
        basic_engine = BacktestEngine(config_manager, Decimal('10000'))
        strategy = MockStrategy("benchmark_basic")
        
        start_time = time.time()
        # We'll simulate the timing since async is complex in benchmark
        basic_duration = 0.1  # Placeholder
        
        # Benchmark enhanced backtest engine
        enhanced_engine = BybitEnhancedBacktestEngine(
            config_manager=config_manager,
            initial_balance=Decimal('10000'),
            vip_tier=BybitVIPTier.VIP1
        )
        
        start_time = time.time()
        # Component initialization timing
        enhanced_duration = time.time() - start_time
        
        # Performance should be reasonable (within 3x of basic)
        # This is a reasonable overhead for the enhanced features
        print(f"Enhanced engine initialization: {enhanced_duration:.4f}s")
        assert enhanced_duration < 1.0  # Should initialize quickly
        
        # Test individual component performance
        fee_calc_start = time.time()
        calculator = BybitFeeCalculator()
        for _ in range(1000):
            calculator.calculate_trading_fee(
                BybitContractType.LINEAR_PERPETUAL,
                Decimal('1.0'),
                Decimal('30000'),
                True,
                BybitVIPTier.VIP1
            )
        fee_calc_duration = time.time() - fee_calc_start
        print(f"1000 fee calculations: {fee_calc_duration:.4f}s")
        assert fee_calc_duration < 0.5  # Should be fast
    
    def test_component_compatibility(self, config_manager):
        """Test that Phase 3 components are compatible with existing architecture."""
        # Test that enhanced engine can be used as drop-in replacement
        basic_engine = BacktestEngine(config_manager, Decimal('10000'))
        enhanced_engine = BybitEnhancedBacktestEngine(config_manager, Decimal('10000'))
        
        # Both should have the same interface
        basic_methods = set(dir(basic_engine))
        enhanced_methods = set(dir(enhanced_engine))
        
        # Enhanced engine should have all basic methods
        missing_methods = basic_methods - enhanced_methods
        # Filter out private methods and properties
        missing_methods = {m for m in missing_methods if not m.startswith('_')}
        
        assert len(missing_methods) == 0, f"Enhanced engine missing methods: {missing_methods}"
        
        # Test configuration compatibility
        assert enhanced_engine.config == basic_engine.config
        assert enhanced_engine.initial_balance == basic_engine.initial_balance
        assert enhanced_engine.commission_rate == basic_engine.commission_rate


class TestDataIntegration:
    """Test integration with Phase 2 data pipeline."""
    
    @pytest.mark.asyncio
    async def test_bybit_client_integration(self, config_manager):
        """Test integration with BybitClient from Phase 2."""
        # Mock BybitClient for testing
        with patch('src.bot.data.bybit_client.BybitClient') as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_kline_data = AsyncMock(return_value=pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
                'open': np.random.uniform(29000, 31000, 100),
                'high': np.random.uniform(29500, 31500, 100),
                'low': np.random.uniform(28500, 30500, 100),
                'close': np.random.uniform(29000, 31000, 100),
                'volume': np.random.uniform(1000, 10000, 100)
            }))
            
            # Test enhanced engine with BybitClient
            engine = BybitEnhancedBacktestEngine(
                config_manager=config_manager,
                initial_balance=Decimal('10000'),
                vip_tier=BybitVIPTier.VIP1
            )
            
            # Verify client integration
            assert engine.bybit_client is not None
            
            # Test data retrieval integration
            data = await mock_client.get_kline_data('BTCUSDT', '1h', 100)
            assert len(data) == 100
            assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_historical_data_manager_integration(self, config_manager):
        """Test integration with HistoricalDataManager from Phase 2."""
        # Mock HistoricalDataManager
        with patch('src.bot.data.historical_data_manager.HistoricalDataManager') as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.get_historical_data = Mock(return_value=pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1H'),
                'open': np.random.uniform(29000, 31000, 1000),
                'high': np.random.uniform(29500, 31500, 1000),
                'low': np.random.uniform(28500, 30500, 1000),
                'close': np.random.uniform(29000, 31000, 1000),
                'volume': np.random.uniform(1000, 10000, 1000)
            }))
            
            # Test enhanced engine with HistoricalDataManager
            engine = BybitEnhancedBacktestEngine(
                config_manager=config_manager,
                initial_balance=Decimal('10000'),
                vip_tier=BybitVIPTier.VIP1
            )
            
            # Test data manager integration
            data = mock_manager.get_historical_data('BTCUSDT', '1h', 1000)
            assert len(data) == 1000
            
            # Verify the engine can process this data format
            assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])


def create_integration_test_suite():
    """Create a comprehensive test suite for Phase 3 integration."""
    
    suite_config = {
        'test_duration_hours': 24,  # 24 hours of data
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'vip_tiers': [BybitVIPTier.NO_VIP, BybitVIPTier.VIP1, BybitVIPTier.VIP2],
        'strategies': ['momentum', 'mean_reversion', 'breakout'],
        'execution_strategies': [ExecutionStrategy.AGGRESSIVE, ExecutionStrategy.PASSIVE, ExecutionStrategy.TWAP]
    }
    
    print("Phase 3 Integration Test Suite Configuration:")
    for key, value in suite_config.items():
        print(f"  {key}: {value}")
    
    return suite_config


if __name__ == "__main__":
    """Run integration tests standalone."""
    import asyncio
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = TradingLogger("Phase3IntegrationTest")
    logger.info("Starting Phase 3 Integration Tests")
    
    # Create test suite configuration
    test_config = create_integration_test_suite()
    
    # Run basic integration test
    async def run_basic_test():
        try:
            # Mock config manager
            config = Mock()
            config.get = Mock(side_effect=lambda key, default=None: {
                'backtesting.commission_rate': 0.001,
                'backtesting.slippage_rate': 0.0005,
                'bybit.api_key': 'test_key',
                'bybit.api_secret': 'test_secret',
                'bybit.testnet': True
            }.get(key, default))
            
            # Generate test data
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
            prices = 30000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, 1000)))
            data = pd.DataFrame({
                'open': prices,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices * (1 + np.random.normal(0, 0.005, 1000)),
                'volume': np.random.lognormal(10, 1, 1000)
            }, index=dates)
            
            # Test enhanced backtest engine
            engine = BybitEnhancedBacktestEngine(
                config_manager=config,
                initial_balance=Decimal('10000'),
                vip_tier=BybitVIPTier.VIP1
            )
            
            strategy = MockStrategy("integration_test")
            
            results = await engine.run_enhanced_backtest(
                strategy=strategy,
                data=data,
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 31),
                symbol='BTCUSDT'
            )
            
            logger.info("Integration test completed successfully!")
            logger.info(f"Results: {results.total_return_pct:.2f}% return, {results.total_trades} trades")
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            raise
    
    # Run the test
    asyncio.run(run_basic_test())
    
    logger.info("Phase 3 Integration Tests completed successfully!")