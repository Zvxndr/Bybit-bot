"""
Unit Tests for Unified Risk Management System

This module contains comprehensive unit tests for the UnifiedRiskManager
to ensure all risk management functionality works correctly.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import numpy as np

from src.bot.risk_management.unified_risk_manager import (
    UnifiedRiskManager,
    RiskLevel,
    RiskAction,
    PositionSizeMethod,
    RiskMetrics,
    TradeRiskAssessment,
    PortfolioRiskProfile
)
from tests.conftest import (
    MockConfigurationManager,
    create_test_trade_request,
    create_test_position,
    assert_decimal_close,
    assert_risk_metrics_valid,
    async_test
)


class TestUnifiedRiskManager:
    """Test suite for UnifiedRiskManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.risk_manager = UnifiedRiskManager(self.config_manager)
    
    def test_initialization(self):
        """Test risk manager initialization."""
        assert self.risk_manager.risk_profile is not None
        assert isinstance(self.risk_manager.risk_profile, PortfolioRiskProfile)
        assert self.risk_manager.monitoring_active == False
        assert self.risk_manager.emergency_stop_active == False
        assert len(self.risk_manager.position_risks) == 0
        assert len(self.risk_manager.risk_alerts) == 0
    
    def test_risk_profile_configuration(self):
        """Test risk profile configuration loading."""
        profile = self.risk_manager.risk_profile
        
        assert profile.max_portfolio_risk == 0.02
        assert profile.max_daily_loss == 0.05
        assert profile.max_drawdown == 0.15
        assert profile.max_leverage == 3.0
        assert profile.volatility_target == 0.15
    
    @async_test
    async def test_assess_trade_risk_basic(self):
        """Test basic trade risk assessment."""
        symbol = "BTCUSDT"
        side = "Buy"
        size = Decimal('0.1')
        entry_price = Decimal('50000')
        stop_loss = Decimal('48000')
        take_profit = Decimal('54000')
        
        assessment = await self.risk_manager.assess_trade_risk(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Validate assessment structure
        assert isinstance(assessment, TradeRiskAssessment)
        assert assessment.symbol == symbol
        assert assessment.side == side
        assert assessment.size == size
        assert assessment.entry_price == entry_price
        assert assessment.stop_loss == stop_loss
        assert assessment.take_profit == take_profit
        
        # Validate calculated fields
        assert assessment.risk_amount > 0
        assert 0 <= assessment.risk_percentage <= 1
        assert assessment.risk_reward_ratio > 0
        assert 0 <= assessment.probability_of_success <= 1
        assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert assessment.recommended_action in [action for action in RiskAction]
    
    @async_test 
    async def test_assess_trade_risk_no_stops(self):
        """Test trade risk assessment without stop loss/take profit."""
        assessment = await self.risk_manager.assess_trade_risk(
            symbol="ETHUSDT",
            side="Sell",
            size=Decimal('1.0'),
            entry_price=Decimal('3000')
        )
        
        assert assessment.stop_loss is None
        assert assessment.take_profit is None
        assert assessment.risk_amount > 0  # Should calculate default risk
        assert assessment.risk_level is not None
    
    @async_test
    async def test_position_size_calculation(self):
        """Test optimal position size calculation."""
        with patch.object(self.risk_manager, '_get_portfolio_value', return_value=Decimal('10000')):
            with patch.object(self.risk_manager, '_get_asset_volatility', return_value=0.15):
                
                position_size = await self.risk_manager._calculate_optimal_position_size(
                    symbol="BTCUSDT",
                    side="Buy",
                    entry_price=Decimal('50000'),
                    stop_loss=Decimal('48000'),
                    volatility_target=0.15
                )
                
                assert position_size >= 0
                assert position_size <= Decimal('100')  # Reasonable upper bound
    
    @async_test
    async def test_kelly_position_sizing(self):
        """Test Kelly criterion position sizing."""
        with patch.object(self.risk_manager, '_get_historical_win_rate', return_value=0.6):
            with patch.object(self.risk_manager, '_get_average_win', return_value=150.0):
                with patch.object(self.risk_manager, '_get_average_loss', return_value=100.0):
                    
                    kelly_size = await self.risk_manager._calculate_kelly_position_size(
                        symbol="BTCUSDT",
                        portfolio_value=Decimal('10000'),
                        entry_price=Decimal('50000')
                    )
                    
                    assert kelly_size >= 0
                    assert kelly_size <= Decimal('2.5')  # Kelly should be reasonable
    
    def test_risk_reward_ratio_calculation(self):
        """Test risk-reward ratio calculation."""
        # Long position
        rr_long = self.risk_manager._calculate_risk_reward_ratio(
            entry_price=Decimal('50000'),
            take_profit=Decimal('54000'),
            stop_loss=Decimal('48000'),
            side="Buy"
        )
        assert rr_long == 2.0  # (54000-50000)/(50000-48000) = 4000/2000 = 2.0
        
        # Short position
        rr_short = self.risk_manager._calculate_risk_reward_ratio(
            entry_price=Decimal('50000'),
            take_profit=Decimal('46000'),
            stop_loss=Decimal('52000'),
            side="Sell"
        )
        assert rr_short == 2.0  # (50000-46000)/(52000-50000) = 4000/2000 = 2.0
    
    def test_expected_value_calculation(self):
        """Test expected value calculation."""
        risk_amount = Decimal('100')
        risk_reward_ratio = 2.0
        probability = 0.6
        
        expected_value = self.risk_manager._calculate_expected_value(
            risk_amount, risk_reward_ratio, probability
        )
        
        # EV = (0.6 * 200) - (0.4 * 100) = 120 - 40 = 80
        assert expected_value == 80.0
    
    @async_test
    async def test_portfolio_metrics_calculation(self):
        """Test portfolio risk metrics calculation."""
        positions = {
            "BTCUSDT": {"value": 5000},
            "ETHUSDT": {"value": 3000}
        }
        portfolio_value = Decimal('10000')
        
        with patch.object(self.risk_manager, '_calculate_portfolio_returns') as mock_returns:
            mock_returns.return_value = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01])
            
            metrics = await self.risk_manager.calculate_portfolio_metrics(positions, portfolio_value)
            
            assert isinstance(metrics, RiskMetrics)
            assert_risk_metrics_valid(metrics)
            assert metrics.portfolio_value == portfolio_value
            assert metrics.total_exposure == Decimal('8000')  # 5000 + 3000
            assert metrics.leverage == 0.8  # 8000 / 10000
    
    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        returns = pd.Series([-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03])
        var_95 = self.risk_manager._calculate_var(returns, 0.95)
        
        assert var_95 < 0  # VaR should be negative
        assert var_95 >= -0.1  # Should be reasonable
    
    def test_expected_shortfall_calculation(self):
        """Test Expected Shortfall calculation."""
        returns = pd.Series([-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03])
        es = self.risk_manager._calculate_expected_shortfall(returns, 0.95)
        
        assert es < 0  # ES should be negative
        assert es <= self.risk_manager._calculate_var(returns, 0.95)  # ES should be worse than VaR
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01, 0.008, 0.012])
        sharpe = self.risk_manager._calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert -5 <= sharpe <= 5  # Reasonable range
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01, 0.008, 0.012])
        sortino = self.risk_manager._calculate_sortino_ratio(returns)
        
        assert isinstance(sortino, float)
        assert -10 <= sortino <= 10  # Reasonable range
    
    def test_concentration_risk_calculation(self):
        """Test concentration risk calculation."""
        positions = {
            "BTCUSDT": {"value": 6000},
            "ETHUSDT": {"value": 3000},
            "ADAUSDT": {"value": 1000}
        }
        portfolio_value = Decimal('10000')
        
        concentration_risk = self.risk_manager._calculate_concentration_risk(positions, portfolio_value)
        
        # HHI = (0.6)^2 + (0.3)^2 + (0.1)^2 = 0.36 + 0.09 + 0.01 = 0.46
        assert abs(concentration_risk - 0.46) < 0.01
    
    def test_risk_score_calculation(self):
        """Test overall risk score calculation."""
        risk_score = self.risk_manager._calculate_risk_score(
            var_95=-0.02,
            volatility=0.15,
            correlation_risk=0.3,
            concentration_risk=0.4,
            liquidity_risk=0.8,
            leverage=2.0
        )
        
        assert 0 <= risk_score <= 1
    
    def test_risk_level_determination(self):
        """Test risk level determination from risk score."""
        assert self.risk_manager._determine_portfolio_risk_level(0.9) == RiskLevel.CRITICAL
        assert self.risk_manager._determine_portfolio_risk_level(0.7) == RiskLevel.HIGH
        assert self.risk_manager._determine_portfolio_risk_level(0.4) == RiskLevel.MEDIUM
        assert self.risk_manager._determine_portfolio_risk_level(0.1) == RiskLevel.LOW
    
    @async_test
    async def test_circuit_breakers(self):
        """Test circuit breaker functionality."""
        portfolio_value = Decimal('8000')  # 20% loss from initial 10000
        
        # Set up daily loss tracking
        self.risk_manager.daily_loss_tracker = Decimal('600')  # 7.5% loss
        
        with patch.object(self.risk_manager, '_calculate_current_drawdown', return_value=0.20):
            with patch.object(self.risk_manager, '_calculate_current_leverage', return_value=4.0):
                
                breakers = await self.risk_manager.check_circuit_breakers(portfolio_value)
                
                assert len(breakers) >= 2  # Should trigger multiple breakers
                assert any("drawdown" in breaker.lower() for breaker in breakers)
                assert any("leverage" in breaker.lower() for breaker in breakers)
                assert self.risk_manager.emergency_stop_active == True
    
    def test_daily_loss_tracking_reset(self):
        """Test daily loss tracking reset."""
        # Set up yesterday's date
        yesterday = datetime.now() - timedelta(days=1)
        self.risk_manager.daily_loss_reset_time = yesterday
        self.risk_manager.daily_loss_tracker = Decimal('500')
        
        # Update should reset
        self.risk_manager._update_daily_loss_tracking()
        
        assert self.risk_manager.daily_loss_tracker == Decimal('0')
        assert self.risk_manager.daily_loss_reset_time.date() == datetime.now().date()
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        # Set high water mark
        self.risk_manager.drawdown_high_water_mark = Decimal('12000')
        
        # Calculate current drawdown
        current_value = Decimal('10000')
        drawdown = self.risk_manager._calculate_current_drawdown(current_value)
        
        expected_drawdown = (12000 - 10000) / 12000  # 16.67%
        assert abs(drawdown - expected_drawdown) < 0.001
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        returns = pd.Series([0.02, -0.05, 0.03, -0.08, 0.01, 0.04, -0.02])
        max_dd = self.risk_manager._calculate_max_drawdown(returns)
        
        assert max_dd <= 0  # Max drawdown should be negative or zero
        assert max_dd >= -0.2  # Should be reasonable
    
    @async_test
    async def test_risk_factors_identification(self):
        """Test risk factor identification."""
        with patch.object(self.risk_manager, '_get_current_price', return_value=Decimal('50000')):
            with patch.object(self.risk_manager, '_get_asset_volatility', return_value=0.35):
                with patch.object(self.risk_manager, '_assess_correlation_risk', return_value=0.8):
                    with patch.object(self.risk_manager, '_calculate_current_leverage', return_value=4.0):
                        
                        risk_factors = await self.risk_manager._identify_risk_factors(
                            symbol="BTCUSDT",
                            size=Decimal('2.0'),
                            portfolio_value=Decimal('10000')
                        )
                        
                        assert len(risk_factors) > 0
                        assert any("high volatility" in factor.lower() for factor in risk_factors)
                        assert any("high leverage" in factor.lower() for factor in risk_factors)
                        assert any("high correlation" in factor.lower() for factor in risk_factors)
    
    def test_risk_action_determination(self):
        """Test risk action determination."""
        # Test various risk levels
        assert self.risk_manager._determine_risk_action(RiskLevel.LOW, 0.01) == RiskAction.CONTINUE
        assert self.risk_manager._determine_risk_action(RiskLevel.MEDIUM, 0.02) == RiskAction.CONTINUE
        assert self.risk_manager._determine_risk_action(RiskLevel.HIGH, 0.03) == RiskAction.REDUCE_POSITION
        assert self.risk_manager._determine_risk_action(RiskLevel.CRITICAL, 0.05) == RiskAction.HALT_TRADING
        assert self.risk_manager._determine_risk_action(RiskLevel.CRITICAL, 0.15) == RiskAction.EMERGENCY_EXIT
        
        # Test emergency stop
        self.risk_manager.emergency_stop_active = True
        assert self.risk_manager._determine_risk_action(RiskLevel.LOW, 0.01) == RiskAction.HALT_TRADING
    
    @async_test
    async def test_risk_monitoring_start_stop(self):
        """Test risk monitoring start and stop."""
        assert self.risk_manager.monitoring_active == False
        
        # Start monitoring
        await self.risk_manager.start_risk_monitoring()
        assert self.risk_manager.monitoring_active == True
        
        # Stop monitoring
        await self.risk_manager.stop_risk_monitoring()
        assert self.risk_manager.monitoring_active == False
    
    @async_test
    async def test_risk_alerts_processing(self):
        """Test risk alerts processing."""
        # Create mock metrics with high risk
        metrics = RiskMetrics(
            portfolio_value=Decimal('10000'),
            total_exposure=Decimal('8000'),
            leverage=2.5,
            var_95=-0.08,
            expected_shortfall=-0.12,
            max_drawdown=-0.20,
            current_drawdown=-0.15,  # High drawdown
            sharpe_ratio=0.5,
            sortino_ratio=0.7,
            volatility=0.25,
            beta=1.2,
            correlation_risk=0.4,
            concentration_risk=0.6,
            liquidity_risk=0.3,
            risk_score=0.8,
            risk_level=RiskLevel.HIGH
        )
        
        breakers = ["High drawdown: 15%"]
        
        await self.risk_manager._process_risk_alerts(metrics, breakers)
        
        assert len(self.risk_manager.risk_alerts) > 0
        alert_types = [alert['type'] for alert in self.risk_manager.risk_alerts]
        assert 'portfolio_risk' in alert_types
        assert 'circuit_breaker' in alert_types
        assert 'drawdown' in alert_types
    
    def test_get_risk_summary(self):
        """Test risk summary generation."""
        summary = self.risk_manager.get_risk_summary()
        
        assert 'risk_profile' in summary
        assert 'current_state' in summary
        assert 'portfolio_metrics' in summary
        assert 'active_positions' in summary
        assert 'recent_alerts' in summary
        assert 'system_version' in summary
        
        # Validate risk profile
        risk_profile = summary['risk_profile']
        assert risk_profile['max_portfolio_risk'] == 0.02
        assert risk_profile['max_daily_loss'] == 0.05
        
        # Validate current state
        current_state = summary['current_state']
        assert 'monitoring_active' in current_state
        assert 'emergency_stop_active' in current_state


class TestRiskMetrics:
    """Test suite for RiskMetrics dataclass."""
    
    def test_risk_metrics_creation(self):
        """Test RiskMetrics creation and validation."""
        metrics = RiskMetrics(
            portfolio_value=Decimal('10000'),
            total_exposure=Decimal('8000'),
            leverage=1.5,
            var_95=-0.05,
            expected_shortfall=-0.08,
            max_drawdown=-0.12,
            current_drawdown=-0.05,
            sharpe_ratio=1.2,
            sortino_ratio=1.8,
            volatility=0.15,
            beta=1.1,
            correlation_risk=0.3,
            concentration_risk=0.4,
            liquidity_risk=0.8,
            risk_score=0.4,
            risk_level=RiskLevel.MEDIUM
        )
        
        assert_risk_metrics_valid(metrics)


class TestTradeRiskAssessment:
    """Test suite for TradeRiskAssessment dataclass."""
    
    def test_trade_risk_assessment_creation(self):
        """Test TradeRiskAssessment creation."""
        assessment = TradeRiskAssessment(
            symbol="BTCUSDT",
            side="Buy",
            size=Decimal('0.1'),
            entry_price=Decimal('50000'),
            stop_loss=Decimal('48000'),
            take_profit=Decimal('54000'),
            risk_amount=Decimal('200'),
            risk_percentage=0.02,
            position_size_recommendation=Decimal('0.1'),
            risk_reward_ratio=2.0,
            probability_of_success=0.6,
            expected_value=80.0,
            risk_level=RiskLevel.MEDIUM,
            recommended_action=RiskAction.CONTINUE,
            risk_factors=["Normal market conditions"]
        )
        
        assert assessment.symbol == "BTCUSDT"
        assert assessment.risk_level == RiskLevel.MEDIUM
        assert assessment.recommended_action == RiskAction.CONTINUE
        assert len(assessment.risk_factors) == 1


class TestPortfolioRiskProfile:
    """Test suite for PortfolioRiskProfile dataclass."""
    
    def test_portfolio_risk_profile_defaults(self):
        """Test PortfolioRiskProfile default values."""
        profile = PortfolioRiskProfile()
        
        assert profile.max_portfolio_risk == 0.02
        assert profile.max_daily_loss == 0.05
        assert profile.max_drawdown == 0.15
        assert profile.max_correlation == 0.7
        assert profile.max_leverage == 3.0
        assert profile.volatility_target == 0.15
        assert profile.var_confidence == 0.95
        assert profile.max_position_concentration == 0.25
    
    def test_portfolio_risk_profile_custom(self):
        """Test PortfolioRiskProfile with custom values."""
        profile = PortfolioRiskProfile(
            max_portfolio_risk=0.03,
            max_daily_loss=0.08,
            max_drawdown=0.20,
            volatility_target=0.20
        )
        
        assert profile.max_portfolio_risk == 0.03
        assert profile.max_daily_loss == 0.08
        assert profile.max_drawdown == 0.20
        assert profile.volatility_target == 0.20


# Performance tests
class TestRiskManagementPerformance:
    """Performance tests for risk management system."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.risk_manager = UnifiedRiskManager(self.config_manager)
    
    @async_test
    async def test_risk_assessment_performance(self):
        """Test risk assessment performance with multiple trades."""
        from tests.conftest import PerformanceTimer
        
        trade_requests = [
            create_test_trade_request(symbol=f"TEST{i}USDT", quantity=Decimal('0.1'))
            for i in range(100)
        ]
        
        with PerformanceTimer("100 risk assessments"):
            tasks = []
            for req in trade_requests:
                task = self.risk_manager.assess_trade_risk(
                    symbol=req["symbol"],
                    side=req["side"],
                    size=req["quantity"],
                    entry_price=req["price"]
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        assert len(results) == 100
        assert all(isinstance(r, TradeRiskAssessment) for r in results)
    
    @async_test
    async def test_portfolio_metrics_performance(self):
        """Test portfolio metrics calculation performance."""
        from tests.conftest import PerformanceTimer
        
        # Create large portfolio
        positions = {
            f"TEST{i}USDT": {"value": np.random.uniform(100, 5000)}
            for i in range(50)
        }
        
        with PerformanceTimer("Portfolio metrics calculation"):
            with patch.object(self.risk_manager, '_calculate_portfolio_returns') as mock_returns:
                mock_returns.return_value = pd.Series(np.random.normal(0, 0.02, 252))  # 1 year of returns
                
                metrics = await self.risk_manager.calculate_portfolio_metrics(
                    positions, Decimal('100000')
                )
        
        assert_risk_metrics_valid(metrics)