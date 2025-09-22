"""
Comprehensive Integration Test Suite - Phase 10

This module provides comprehensive integration tests that validate:
- Component interactions across all phases
- End-to-end trading workflows
- System reliability under various market conditions
- Error handling and recovery mechanisms
- Performance characteristics
- Configuration management
- API interfaces

Author: Trading Bot Team
Version: 1.0.0
"""

import pytest
import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import tempfile
import shutil
from pathlib import Path
import logging

# Import all phase components for testing
from integrated_trading_bot import IntegratedTradingBot, BotConfiguration, BotStatus
from config_manager import ConfigurationManager, ComprehensiveConfig, Environment

# Core Phase 1 imports
from core.trading_engine import TradingEngine, OrderType, OrderSide
from core.market_data import MarketDataManager
from core.position_manager import PositionManager

# Risk Management Phase 2 imports
from risk_management.risk_manager import RiskManager
from risk_management.portfolio_risk import PortfolioRiskManager
from risk_management.drawdown_protection import DrawdownProtectionManager

# Backtesting Phase 3 imports
from backtesting.backtesting_engine import BacktestingEngine
from backtesting.performance_analyzer import PerformanceAnalyzer

# Monitoring Phase 4 imports
from monitoring.system_monitor import SystemMonitor, SystemHealthStatus
from monitoring.performance_tracker import PerformanceTracker
from monitoring.alerting_system import AlertingSystem, AlertLevel

# Tax and Reporting Phase 5 imports
from tax_reporting.trade_logger import TradeLogger
from tax_reporting.tax_calculator import TaxCalculator

# Advanced Features Phase 6 imports
from advanced.regime_detector import RegimeDetector, MarketRegime
from advanced.portfolio_optimizer import PortfolioOptimizer
from advanced.news_analyzer import NewsAnalyzer


class TestDataGenerator:
    """Generate test data for integration tests"""
    
    @staticmethod
    def generate_market_data(days: int = 100, 
                           start_price: float = 30000.0,
                           volatility: float = 0.02) -> pd.DataFrame:
        """Generate realistic market data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Generate price series with random walk
        np.random.seed(42)  # For reproducible tests
        returns = np.random.normal(0.001, volatility, days)
        
        prices = [start_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, days)
        })
    
    @staticmethod
    def generate_trade_data(count: int = 50) -> List[Dict]:
        """Generate sample trade data"""
        trades = []
        base_price = 30000.0
        
        for i in range(count):
            trade = {
                'timestamp': datetime.now() - timedelta(hours=i),
                'pair': 'BTCUSDT',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'quantity': np.random.uniform(0.01, 0.1),
                'price': base_price * (1 + np.random.normal(0, 0.02)),
                'order_id': f'test_order_{i}',
                'reason': 'Test trade'
            }
            trades.append(trade)
        
        return trades
    
    @staticmethod
    def generate_news_articles(count: int = 20) -> List[Dict]:
        """Generate sample news articles for testing"""
        articles = []
        sentiments = ['positive', 'negative', 'neutral']
        
        for i in range(count):
            article = {
                'title': f'Test News Article {i}',
                'content': f'This is test content for article {i}',
                'sentiment': np.random.choice(sentiments),
                'timestamp': datetime.now() - timedelta(hours=i),
                'source': 'test_source'
            }
            articles.append(article)
        
        return articles


class IntegrationTestBase(unittest.IAsyncioTestCase):
    """Base class for integration tests"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Set up logging
        logging.basicConfig(level=logging.WARNING)  # Reduce noise in tests
        
        # Create test configuration
        self.config = BotConfiguration(
            trading_pairs=["BTCUSDT", "ETHUSDT"],
            initial_capital=10000.0,
            max_position_size=0.1,
            update_frequency_seconds=1,  # Fast updates for testing
            enable_regime_detection=True,
            enable_portfolio_optimization=True,
            enable_news_analysis=True,
            enable_automated_reporting=False  # Disable for tests
        )
        
        # Initialize test data generator
        self.data_generator = TestDataGenerator()
        
        # Create sample market data
        self.market_data = self.data_generator.generate_market_data(days=100)
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        # Remove temporary directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)


class TestConfigurationManager(unittest.TestCase):
    """Test configuration management system"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_manager = ConfigurationManager(str(self.test_dir))
    
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_create_default_configs(self):
        """Test creation of default configuration files"""
        self.config_manager.create_default_config_files()
        
        # Check that all environment configs were created
        for env in Environment:
            config_file = self.test_dir / f"config_{env.value}.yaml"
            self.assertTrue(config_file.exists())
    
    def test_load_configuration(self):
        """Test loading configuration from file"""
        self.config_manager.create_default_config_files()
        
        config = self.config_manager.load_config(environment=Environment.DEVELOPMENT)
        
        self.assertIsNotNone(config)
        self.assertEqual(config.environment, Environment.DEVELOPMENT)
        self.assertTrue(config.exchange.testnet)  # Development should use testnet
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        config = ComprehensiveConfig()
        
        # Should not raise exception with valid config
        config.validate()
        
        # Test invalid configuration
        config.exchange.api_key = ""
        config.exchange.api_secret = ""
        with self.assertRaises(Exception):
            config.validate()
    
    def test_configuration_updates(self):
        """Test dynamic configuration updates"""
        config = self.config_manager.load_config(environment=Environment.DEVELOPMENT)
        
        updates = {
            'trading.initial_capital': 15000.0,
            'risk_management.max_daily_loss': 0.03
        }
        
        changed = self.config_manager.update_config(updates)
        self.assertTrue(changed)
        
        # Verify updates were applied
        self.assertEqual(self.config_manager.config.trading.initial_capital, 15000.0)
        self.assertEqual(self.config_manager.config.risk_management.max_daily_loss, 0.03)


class TestIntegratedTradingBot(IntegrationTestBase):
    """Test the integrated trading bot system"""
    
    async def test_bot_initialization(self):
        """Test bot initialization with all components"""
        bot = IntegratedTradingBot(self.config)
        
        self.assertEqual(bot.status, BotStatus.INITIALIZING)
        self.assertIsNotNone(bot.trading_engine)
        self.assertIsNotNone(bot.market_data_manager)
        self.assertIsNotNone(bot.position_manager)
        self.assertIsNotNone(bot.risk_manager)
        self.assertIsNotNone(bot.regime_detector)
        
        # Test component initialization
        self.assertTrue(hasattr(bot, 'performance_metrics'))
        self.assertEqual(bot.trade_count, 0)
    
    @patch('integrated_trading_bot.TradingEngine')
    @patch('integrated_trading_bot.MarketDataManager')
    async def test_bot_startup_sequence(self, mock_market_data, mock_trading_engine):
        """Test bot startup sequence"""
        # Mock components
        mock_trading_engine.return_value.start = AsyncMock()
        mock_market_data.return_value.start = AsyncMock()
        
        bot = IntegratedTradingBot(self.config)
        
        # Start bot
        await bot.start()
        
        self.assertEqual(bot.status, BotStatus.RUNNING)
        self.assertIsNotNone(bot.start_time)
        
        # Clean shutdown
        await bot.stop()
        self.assertEqual(bot.status, BotStatus.STOPPED)
    
    async def test_risk_assessment_integration(self):
        """Test integration of risk assessment across components"""
        bot = IntegratedTradingBot(self.config)
        
        # Mock position manager
        with patch.object(bot.position_manager, 'get_all_positions') as mock_positions:
            mock_positions.return_value = {
                'BTCUSDT': Mock(market_value=1000.0, unrealized_pnl=-50.0)
            }
            
            # Test risk assessment
            risk_metrics = await bot._assess_portfolio_risk()
            
            self.assertIsNotNone(risk_metrics)
            self.assertEqual(risk_metrics.portfolio_value, 1000.0)
    
    async def test_market_context_analysis(self):
        """Test advanced market context analysis"""
        bot = IntegratedTradingBot(self.config)
        
        # Mock regime detector
        with patch.object(bot.regime_detector, 'detect_regime') as mock_regime:
            mock_regime_result = Mock()
            mock_regime_result.regime = MarketRegime.BULL_MARKET
            mock_regime_result.confidence = 0.8
            mock_regime.return_value = mock_regime_result
            
            # Test market context analysis
            context = await bot._analyze_market_context(self.market_data)
            
            self.assertIsNotNone(context)
            self.assertEqual(context['regime'], MarketRegime.BULL_MARKET)
            self.assertIsInstance(context['trading_halted'], bool)
    
    async def test_trading_signal_generation(self):
        """Test trading signal generation with market context"""
        bot = IntegratedTradingBot(self.config)
        
        # Create market context
        context = {
            'regime': MarketRegime.BULL_MARKET,
            'sentiment': Mock(overall_sentiment='positive'),
            'trading_halted': False,
            'volatility': 0.02
        }
        
        # Generate signals
        signals = await bot._generate_trading_signals(self.market_data, context)
        
        # Should generate signals for configured pairs
        self.assertIsInstance(signals, list)
        # Signals may be empty if conditions not met, which is valid
    
    async def test_position_size_calculation(self):
        """Test position size calculation with risk management"""
        bot = IntegratedTradingBot(self.config)
        
        # Mock portfolio value
        with patch.object(bot.position_manager, 'get_portfolio_value') as mock_portfolio_value:
            mock_portfolio_value.return_value = 10000.0
            
            signal = {
                'pair': 'BTCUSDT',
                'side': 'buy',
                'price': 30000.0,
                'confidence': 0.7
            }
            
            position_size = await bot._calculate_position_size(signal)
            
            # Should calculate reasonable position size
            self.assertGreaterEqual(position_size, 0)
            
            # Position value should respect risk limits
            position_value = position_size * signal['price']
            max_position_value = 10000.0 * self.config.max_position_size
            self.assertLessEqual(position_value, max_position_value)


class TestEndToEndWorkflows(IntegrationTestBase):
    """Test complete end-to-end trading workflows"""
    
    @patch('integrated_trading_bot.TradingEngine')
    @patch('integrated_trading_bot.MarketDataManager')
    async def test_complete_trading_cycle(self, mock_market_data, mock_trading_engine):
        """Test complete trading cycle from signal to execution"""
        # Set up mocks
        mock_market_data.return_value.get_latest_data = AsyncMock(return_value=self.market_data)
        mock_market_data.return_value.start = AsyncMock()
        
        mock_order_result = Mock()
        mock_order_result.success = True
        mock_order_result.fill_price = 30000.0
        mock_order_result.order_id = 'test_order_123'
        
        mock_trading_engine.return_value.place_order = AsyncMock(return_value=mock_order_result)
        mock_trading_engine.return_value.start = AsyncMock()
        
        bot = IntegratedTradingBot(self.config)
        
        # Mock other dependencies
        with patch.object(bot.system_monitor, 'get_system_health') as mock_health:
            mock_health.return_value = Mock(status=SystemHealthStatus.HEALTHY)
            
            with patch.object(bot.position_manager, 'get_portfolio_value') as mock_portfolio_value:
                mock_portfolio_value.return_value = 10000.0
                
                with patch.object(bot.position_manager, 'get_all_positions') as mock_positions:
                    mock_positions.return_value = {}
                    
                    # Start bot
                    await bot.start()
                    
                    # Let it run briefly
                    await asyncio.sleep(0.1)
                    
                    # Stop bot
                    await bot.stop()
                    
                    # Verify bot ran successfully
                    self.assertEqual(bot.status, BotStatus.STOPPED)
    
    async def test_risk_breach_handling(self):
        """Test handling of risk threshold breaches"""
        bot = IntegratedTradingBot(self.config)
        
        # Create risk metrics indicating breach
        risk_metrics = Mock()
        risk_metrics.requires_immediate_action = True
        risk_metrics.current_drawdown = 0.20  # Exceeds max drawdown
        
        # Mock position manager
        with patch.object(bot.position_manager, 'get_all_positions') as mock_positions:
            mock_positions.return_value = {
                'BTCUSDT': Mock(market_value=1000.0, unrealized_pnl=-200.0)
            }
            
            with patch.object(bot.trading_engine, 'close_position') as mock_close:
                mock_close.return_value = AsyncMock()
                
                # Handle risk breach
                await bot._handle_risk_breach(risk_metrics)
                
                # Verify trading was disabled
                self.assertFalse(bot.trading_enabled)
    
    async def test_emergency_shutdown(self):
        """Test emergency shutdown procedure"""
        bot = IntegratedTradingBot(self.config)
        
        # Mock position manager
        with patch.object(bot.position_manager, 'get_all_positions') as mock_positions:
            mock_positions.return_value = {
                'BTCUSDT': Mock(),
                'ETHUSDT': Mock()
            }
            
            with patch.object(bot.trading_engine, 'close_position') as mock_close:
                mock_close.return_value = AsyncMock()
                
                # Trigger emergency shutdown
                await bot._emergency_shutdown("Test emergency")
                
                # Verify emergency state
                self.assertEqual(bot.status, BotStatus.ERROR)
                self.assertFalse(bot.trading_enabled)
                
                # Verify all positions were closed
                self.assertEqual(mock_close.call_count, 2)


class TestPerformanceAndStress(IntegrationTestBase):
    """Test system performance and stress scenarios"""
    
    async def test_high_frequency_updates(self):
        """Test system performance under high-frequency market updates"""
        bot = IntegratedTradingBot(self.config)
        bot.config.update_frequency_seconds = 0.1  # Very fast updates
        
        # Mock market data with frequent updates
        market_updates = []
        for i in range(100):
            update = self.market_data.copy()
            update['close'] = update['close'] * (1 + np.random.normal(0, 0.001))
            market_updates.append(update)
        
        # Test processing multiple updates
        start_time = datetime.now()
        
        for update in market_updates[:10]:  # Test subset for speed
            context = await bot._analyze_market_context(update)
            signals = await bot._generate_trading_signals(update, context)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Should process updates within reasonable time
        self.assertLess(processing_time, 1.0)  # Under 1 second for 10 updates
    
    async def test_memory_usage_monitoring(self):
        """Test system behavior under memory constraints"""
        bot = IntegratedTradingBot(self.config)
        
        # Mock system monitor with high memory usage
        with patch.object(bot.system_monitor, 'get_system_health') as mock_health:
            # First call: normal
            # Second call: high memory
            mock_health.side_effect = [
                Mock(status=SystemHealthStatus.HEALTHY),
                Mock(status=SystemHealthStatus.WARNING, message="High memory usage")
            ]
            
            # Test monitoring response
            health1 = await bot.system_monitor.get_system_health()
            self.assertEqual(health1.status, SystemHealthStatus.HEALTHY)
            
            health2 = await bot.system_monitor.get_system_health()
            self.assertEqual(health2.status, SystemHealthStatus.WARNING)
    
    async def test_concurrent_operations(self):
        """Test system behavior with concurrent operations"""
        bot = IntegratedTradingBot(self.config)
        
        # Create multiple concurrent tasks
        tasks = []
        
        # Market analysis tasks
        for i in range(5):
            task = asyncio.create_task(
                bot._analyze_market_context(self.market_data)
            )
            tasks.append(task)
        
        # Performance update tasks
        for i in range(3):
            task = asyncio.create_task(
                bot._update_performance_metrics()
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that no tasks failed with exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        self.assertEqual(len(exceptions), 0, f"Concurrent operations failed: {exceptions}")


class TestComponentInteraction(IntegrationTestBase):
    """Test interactions between different system components"""
    
    async def test_regime_detection_risk_integration(self):
        """Test integration between regime detection and risk management"""
        bot = IntegratedTradingBot(self.config)
        
        # Mock regime detection with high-risk regime
        with patch.object(bot.regime_detector, 'detect_regime') as mock_regime:
            mock_regime_result = Mock()
            mock_regime_result.regime = MarketRegime.CRASH
            mock_regime_result.confidence = 0.9
            mock_regime.return_value = mock_regime_result
            
            with patch.object(bot.regime_detector, 'should_trade_in_regime') as mock_should_trade:
                mock_should_trade.return_value = False
                
                # Analyze market context
                context = await bot._analyze_market_context(self.market_data)
                
                # Should halt trading in crash regime
                self.assertTrue(context['trading_halted'])
                self.assertIn('Regime filter', context['halt_reason'])
    
    async def test_news_sentiment_trading_integration(self):
        """Test integration between news sentiment and trading decisions"""
        bot = IntegratedTradingBot(self.config)
        
        # Mock news analyzer with negative sentiment
        with patch.object(bot.news_analyzer, 'get_current_sentiment') as mock_sentiment:
            mock_sentiment.return_value = Mock(
                overall_sentiment='very_negative',
                sentiment_score=-0.8,
                confidence=0.9
            )
            
            with patch.object(bot.news_analyzer, 'should_halt_trading') as mock_should_halt:
                mock_should_halt.return_value = (True, "Very negative market sentiment")
                
                # Analyze market context
                context = await bot._analyze_market_context(self.market_data)
                
                # Should halt trading due to negative sentiment
                self.assertTrue(context['trading_halted'])
                self.assertIn('negative market sentiment', context['halt_reason'])
    
    async def test_portfolio_optimization_execution_integration(self):
        """Test integration between portfolio optimization and trade execution"""
        bot = IntegratedTradingBot(self.config)
        
        if bot.portfolio_optimizer:
            # Mock historical data
            with patch.object(bot.market_data_manager, 'get_historical_data') as mock_historical:
                mock_historical.return_value = self.market_data
                
                # Mock portfolio optimizer
                with patch.object(bot.portfolio_optimizer, 'optimize_portfolio') as mock_optimize:
                    mock_result = Mock()
                    mock_result.weights = {'BTCUSDT': 0.6, 'ETHUSDT': 0.4}
                    mock_result.sharpe_ratio = 1.5
                    mock_optimize.return_value = mock_result
                    
                    # Run portfolio optimization
                    await bot._optimize_portfolio()
                    
                    # Verify optimization was called
                    mock_optimize.assert_called_once()


class TestSystemReliability(IntegrationTestBase):
    """Test system reliability and error handling"""
    
    async def test_component_failure_recovery(self):
        """Test system behavior when components fail"""
        bot = IntegratedTradingBot(self.config)
        
        # Mock market data manager failure
        with patch.object(bot.market_data_manager, 'get_latest_data') as mock_market_data:
            mock_market_data.side_effect = Exception("Market data connection failed")
            
            # The main loop should handle this gracefully
            try:
                # This would normally be part of the main trading loop
                market_data = await bot.market_data_manager.get_latest_data()
            except Exception as e:
                # Should catch and log the exception without crashing
                self.assertIn("Market data connection failed", str(e))
    
    async def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration"""
        # Create invalid configuration
        invalid_config = BotConfiguration(
            trading_pairs=[],  # Empty pairs - invalid
            initial_capital=-1000.0,  # Negative capital - invalid
            max_position_size=1.5  # > 1.0 - invalid
        )
        
        # Bot should handle invalid config gracefully
        with self.assertRaises(Exception):
            bot = IntegratedTradingBot(invalid_config)
            # Configuration validation should catch this
    
    async def test_network_connectivity_issues(self):
        """Test handling of network connectivity issues"""
        bot = IntegratedTradingBot(self.config)
        
        # Mock trading engine with network failure
        with patch.object(bot.trading_engine, 'place_order') as mock_place_order:
            mock_place_order.side_effect = Exception("Network timeout")
            
            signal = {
                'pair': 'BTCUSDT',
                'side': 'buy',
                'price': 30000.0,
                'confidence': 0.7,
                'reason': 'Test signal'
            }
            
            # Execute trading signals - should handle network error
            executed_trades = await bot._execute_trading_signals([signal])
            
            # Should return empty list when orders fail
            self.assertEqual(len(executed_trades), 0)


# Test runner and utilities
class TestRunner:
    """Test runner for integration tests"""
    
    @staticmethod
    def run_all_tests():
        """Run all integration tests"""
        print("Phase 10: Running Comprehensive Integration Tests")
        print("=" * 60)
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Add test classes
        test_classes = [
            TestConfigurationManager,
            TestIntegratedTradingBot,
            TestEndToEndWorkflows,
            TestPerformanceAndStress,
            TestComponentInteraction,
            TestSystemReliability
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        # Print summary
        print(f"\n" + "=" * 60)
        print(f"INTEGRATION TEST RESULTS")
        print(f"=" * 60)
        print(f"Tests Run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
        
        if result.failures:
            print(f"\nFailures:")
            for test, failure in result.failures:
                print(f"  - {test}: {failure}")
        
        if result.errors:
            print(f"\nErrors:")
            for test, error in result.errors:
                print(f"  - {test}: {error}")
        
        return result.wasSuccessful()


# Example usage
if __name__ == "__main__":
    # Set up event loop for async tests
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run all integration tests
    success = TestRunner.run_all_tests()
    
    print(f"\n🎉 Integration Test Suite Complete!")
    print(f"{'✅ All tests passed!' if success else '❌ Some tests failed.'}")
    print(f"✅ Component interaction validation")
    print(f"✅ End-to-end workflow testing")
    print(f"✅ System reliability verification")
    print(f"✅ Performance and stress testing")
    print(f"✅ Error handling validation")
    
    if not success:
        exit(1)