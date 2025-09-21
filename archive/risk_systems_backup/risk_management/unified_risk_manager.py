"""
Unified Risk Management System - Consolidation of Multiple Risk Systems.

This module consolidates the three separate risk management implementations into a single,
comprehensive system that provides all the features from:
1. src/bot/risk_management/ - Core risk management 
2. src/bot/risk/ - Advanced risk analysis
3. Scattered risk components in other packages

Key Features:
- Unified risk management interface
- Advanced position sizing methods (Kelly, Risk Parity, Volatility Targeting)
- Real-time risk monitoring with circuit breakers
- Dynamic risk adjustment based on market conditions
- Portfolio risk analysis and optimization
- Comprehensive risk metrics and alerting
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskAction(Enum):
    """Risk management actions."""
    CONTINUE = "continue"
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    HALT_TRADING = "halt_trading"
    EMERGENCY_EXIT = "emergency_exit"

class PositionSizeMethod(Enum):
    """Position sizing methods."""
    FIXED_PERCENT = "fixed_percent"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGET = "volatility_target"
    RISK_PARITY = "risk_parity"
    MAX_DRAWDOWN = "max_drawdown"
    DYNAMIC = "dynamic"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
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
    
@dataclass
class TradeRiskAssessment:
    """Individual trade risk assessment."""
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
    
@dataclass
class PortfolioRiskProfile:
    """Portfolio risk profile configuration."""
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    max_daily_loss: float = 0.05      # 5% max daily loss
    max_drawdown: float = 0.15        # 15% max drawdown
    max_correlation: float = 0.7      # Max correlation between positions
    max_leverage: float = 3.0         # Max total leverage
    volatility_target: float = 0.15   # 15% annual volatility target
    var_confidence: float = 0.95      # 95% VaR confidence level
    max_position_concentration: float = 0.25  # 25% max single position size
    
class UnifiedRiskManager:
    """
    Unified Risk Management System.
    
    Consolidates all risk management functionality into a single, comprehensive system
    that handles position sizing, portfolio risk monitoring, and trade execution safety.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = TradingLogger()
        
        # Risk configuration
        self.risk_profile = PortfolioRiskProfile()
        self.load_risk_configuration()
        
        # Risk state tracking
        self.portfolio_metrics: Optional[RiskMetrics] = None
        self.position_risks: Dict[str, TradeRiskAssessment] = {}
        self.risk_alerts: List[Dict[str, Any]] = []
        self.emergency_stop_active = False
        
        # Historical data for risk calculations
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.returns_history: Dict[str, pd.Series] = {}
        self.volatility_cache: Dict[str, float] = {}
        
        # Risk monitoring
        self.monitoring_active = False
        self.last_risk_check = datetime.now()
        self.risk_check_interval = 60  # seconds
        
        # Circuit breakers
        self.daily_loss_tracker = Decimal('0')
        self.daily_loss_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.drawdown_high_water_mark = Decimal('0')
        
        self.logger.info("UnifiedRiskManager initialized")
    
    def load_risk_configuration(self):
        """Load risk configuration from config manager."""
        try:
            config = self.config_manager.get_config()
            
            if hasattr(config, 'risk_management'):
                risk_config = config.risk_management
                
                self.risk_profile.max_portfolio_risk = getattr(risk_config, 'max_portfolio_risk', 0.02)
                self.risk_profile.max_daily_loss = getattr(risk_config, 'max_daily_loss', 0.05)
                self.risk_profile.max_drawdown = getattr(risk_config, 'max_drawdown', 0.15)
                self.risk_profile.max_leverage = getattr(risk_config, 'max_leverage', 3.0)
                self.risk_profile.volatility_target = getattr(risk_config, 'volatility_target', 0.15)
                
                self.logger.info("Risk configuration loaded from config manager")
            else:
                self.logger.warning("No risk configuration found, using defaults")
                
        except Exception as e:
            self.logger.error(f"Failed to load risk configuration: {e}")
    
    async def assess_trade_risk(self, symbol: str, side: str, size: Decimal, 
                               entry_price: Decimal, stop_loss: Optional[Decimal] = None,
                               take_profit: Optional[Decimal] = None) -> TradeRiskAssessment:
        """
        Comprehensive trade risk assessment.
        
        Analyzes individual trade risk considering portfolio context, correlation,
        volatility, and various risk factors.
        """
        try:
            # Calculate basic risk metrics
            risk_amount = self._calculate_trade_risk(size, entry_price, stop_loss)
            portfolio_value = await self._get_portfolio_value()
            risk_percentage = float(risk_amount / portfolio_value) if portfolio_value > 0 else 0
            
            # Get position size recommendation
            recommended_size = await self._calculate_optimal_position_size(
                symbol, side, entry_price, stop_loss, volatility_target=self.risk_profile.volatility_target
            )
            
            # Calculate risk-reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                entry_price, take_profit, stop_loss, side
            )
            
            # Estimate probability of success (simplified model)
            probability_of_success = await self._estimate_success_probability(symbol, side)
            
            # Calculate expected value
            expected_value = self._calculate_expected_value(
                risk_amount, risk_reward_ratio, probability_of_success
            )
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(symbol, size, portfolio_value)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_percentage, risk_factors)
            
            # Determine recommended action
            recommended_action = self._determine_risk_action(risk_level, risk_percentage)
            
            assessment = TradeRiskAssessment(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_amount=risk_amount,
                risk_percentage=risk_percentage,
                position_size_recommendation=recommended_size,
                risk_reward_ratio=risk_reward_ratio,
                probability_of_success=probability_of_success,
                expected_value=expected_value,
                risk_level=risk_level,
                recommended_action=recommended_action,
                risk_factors=risk_factors
            )
            
            # Store assessment
            self.position_risks[f"{symbol}_{side}"] = assessment
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Failed to assess trade risk: {e}")
            # Return conservative assessment on error
            return self._get_conservative_assessment(symbol, side, size, entry_price)
    
    def _calculate_trade_risk(self, size: Decimal, entry_price: Decimal, 
                             stop_loss: Optional[Decimal]) -> Decimal:
        """Calculate the monetary risk of a trade."""
        if stop_loss is None:
            # Use default risk based on volatility if no stop loss
            return size * entry_price * Decimal('0.02')  # 2% default risk
        
        risk_per_unit = abs(entry_price - stop_loss)
        return size * risk_per_unit
    
    async def _calculate_optimal_position_size(self, symbol: str, side: str, 
                                              entry_price: Decimal, stop_loss: Optional[Decimal],
                                              volatility_target: float) -> Decimal:
        """Calculate optimal position size using multiple methods."""
        try:
            portfolio_value = await self._get_portfolio_value()
            
            if portfolio_value <= 0:
                return Decimal('0')
            
            # Method 1: Fixed percentage risk
            max_risk_amount = portfolio_value * Decimal(str(self.risk_profile.max_portfolio_risk))
            
            if stop_loss:
                risk_per_unit = abs(entry_price - stop_loss)
                fixed_percent_size = max_risk_amount / risk_per_unit if risk_per_unit > 0 else Decimal('0')
            else:
                fixed_percent_size = max_risk_amount / (entry_price * Decimal('0.02'))
            
            # Method 2: Volatility targeting
            volatility = await self._get_asset_volatility(symbol)
            if volatility > 0:
                volatility_target_size = (portfolio_value * Decimal(str(volatility_target))) / (entry_price * Decimal(str(volatility)))
            else:
                volatility_target_size = fixed_percent_size
            
            # Method 3: Kelly Criterion (simplified)
            kelly_size = await self._calculate_kelly_position_size(symbol, portfolio_value, entry_price)
            
            # Use the most conservative size
            recommended_size = min(fixed_percent_size, volatility_target_size, kelly_size)
            
            # Apply concentration limits
            max_concentration_size = portfolio_value * Decimal(str(self.risk_profile.max_position_concentration)) / entry_price
            recommended_size = min(recommended_size, max_concentration_size)
            
            return max(recommended_size, Decimal('0'))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate optimal position size: {e}")
            return Decimal('0')
    
    async def _calculate_kelly_position_size(self, symbol: str, portfolio_value: Decimal, 
                                           entry_price: Decimal) -> Decimal:
        """Calculate position size using Kelly Criterion."""
        try:
            # Get historical win rate and average win/loss
            win_rate = await self._get_historical_win_rate(symbol)
            avg_win = await self._get_average_win(symbol)
            avg_loss = await self._get_average_loss(symbol)
            
            if avg_loss == 0:
                return Decimal('0')
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            kelly_amount = portfolio_value * Decimal(str(kelly_fraction))
            kelly_size = kelly_amount / entry_price
            
            return kelly_size
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Kelly position size: {e}")
            return Decimal('0')
    
    def _calculate_risk_reward_ratio(self, entry_price: Decimal, take_profit: Optional[Decimal],
                                   stop_loss: Optional[Decimal], side: str) -> float:
        """Calculate risk-reward ratio for the trade."""
        try:
            if not take_profit or not stop_loss:
                return 1.0  # Default ratio if targets not set
            
            if side.lower() == 'buy':
                potential_profit = take_profit - entry_price
                potential_loss = entry_price - stop_loss
            else:  # sell
                potential_profit = entry_price - take_profit
                potential_loss = stop_loss - entry_price
            
            if potential_loss <= 0:
                return float('inf')
            
            return float(potential_profit / potential_loss)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate risk-reward ratio: {e}")
            return 1.0
    
    def _calculate_expected_value(self, risk_amount: Decimal, risk_reward_ratio: float,
                                probability_of_success: float) -> float:
        """Calculate expected value of the trade."""
        try:
            potential_profit = float(risk_amount) * risk_reward_ratio
            potential_loss = float(risk_amount)
            
            expected_value = (probability_of_success * potential_profit) - ((1 - probability_of_success) * potential_loss)
            
            return expected_value
            
        except Exception as e:
            self.logger.error(f"Failed to calculate expected value: {e}")
            return 0.0
    
    async def _identify_risk_factors(self, symbol: str, size: Decimal, 
                                   portfolio_value: Decimal) -> List[str]:
        """Identify specific risk factors for the trade."""
        risk_factors = []
        
        try:
            # Position size risk
            position_value = size * await self._get_current_price(symbol)
            concentration = float(position_value / portfolio_value) if portfolio_value > 0 else 0
            
            if concentration > self.risk_profile.max_position_concentration:
                risk_factors.append(f"High concentration: {concentration:.1%}")
            
            # Volatility risk
            volatility = await self._get_asset_volatility(symbol)
            if volatility > 0.3:  # 30% volatility threshold
                risk_factors.append(f"High volatility: {volatility:.1%}")
            
            # Correlation risk
            correlation_risk = await self._assess_correlation_risk(symbol)
            if correlation_risk > self.risk_profile.max_correlation:
                risk_factors.append(f"High correlation: {correlation_risk:.2f}")
            
            # Leverage risk
            current_leverage = await self._calculate_current_leverage()
            if current_leverage > self.risk_profile.max_leverage:
                risk_factors.append(f"High leverage: {current_leverage:.1f}x")
            
            # Market conditions risk
            market_condition = await self._assess_market_conditions()
            if market_condition == "volatile":
                risk_factors.append("Volatile market conditions")
            elif market_condition == "trending":
                risk_factors.append("Strong trending market")
            
            # Liquidity risk
            liquidity_score = await self._assess_liquidity_risk(symbol)
            if liquidity_score < 0.5:
                risk_factors.append("Low liquidity")
            
        except Exception as e:
            self.logger.error(f"Failed to identify risk factors: {e}")
            risk_factors.append("Risk assessment error")
        
        return risk_factors
    
    def _determine_risk_level(self, risk_percentage: float, risk_factors: List[str]) -> RiskLevel:
        """Determine overall risk level based on metrics."""
        # Base risk level on risk percentage
        if risk_percentage > 0.05:  # > 5%
            risk_level = RiskLevel.CRITICAL
        elif risk_percentage > 0.03:  # > 3%
            risk_level = RiskLevel.HIGH
        elif risk_percentage > 0.01:  # > 1%
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Adjust based on risk factors
        critical_factors = [f for f in risk_factors if any(keyword in f.lower() 
                          for keyword in ['high', 'volatile', 'low liquidity'])]
        
        if len(critical_factors) >= 3:
            risk_level = RiskLevel.CRITICAL
        elif len(critical_factors) >= 2 and risk_level != RiskLevel.LOW:
            risk_level = RiskLevel.HIGH
        
        return risk_level
    
    def _determine_risk_action(self, risk_level: RiskLevel, risk_percentage: float) -> RiskAction:
        """Determine recommended risk action."""
        if self.emergency_stop_active:
            return RiskAction.HALT_TRADING
        
        if risk_level == RiskLevel.CRITICAL:
            if risk_percentage > 0.1:  # > 10%
                return RiskAction.EMERGENCY_EXIT
            else:
                return RiskAction.HALT_TRADING
        elif risk_level == RiskLevel.HIGH:
            return RiskAction.REDUCE_POSITION
        elif risk_level == RiskLevel.MEDIUM:
            return RiskAction.CONTINUE
        else:
            return RiskAction.CONTINUE
    
    async def calculate_portfolio_metrics(self, positions: Dict[str, Any], 
                                        portfolio_value: Decimal) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        try:
            # Basic portfolio metrics
            total_exposure = sum(Decimal(str(pos.get('value', 0))) for pos in positions.values())
            leverage = float(total_exposure / portfolio_value) if portfolio_value > 0 else 0
            
            # Calculate returns for risk metrics
            portfolio_returns = await self._calculate_portfolio_returns(positions)
            
            # VaR and Expected Shortfall
            var_95 = self._calculate_var(portfolio_returns, 0.95)
            expected_shortfall = self._calculate_expected_shortfall(portfolio_returns, 0.95)
            
            # Drawdown metrics
            current_drawdown = self._calculate_current_drawdown(portfolio_value)
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            # Performance ratios
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
            sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
            
            # Risk measures
            volatility = float(portfolio_returns.std() * np.sqrt(252)) if len(portfolio_returns) > 1 else 0
            beta = await self._calculate_portfolio_beta(positions)
            correlation_risk = await self._calculate_correlation_risk(positions)
            concentration_risk = self._calculate_concentration_risk(positions, portfolio_value)
            liquidity_risk = await self._calculate_liquidity_risk(positions)
            
            # Overall risk score
            risk_score = self._calculate_risk_score(
                var_95, volatility, correlation_risk, concentration_risk, liquidity_risk, leverage
            )
            
            # Risk level
            risk_level = self._determine_portfolio_risk_level(risk_score)
            
            metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=total_exposure,
                leverage=leverage,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                volatility=volatility,
                beta=beta,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                risk_score=risk_score,
                risk_level=risk_level
            )
            
            self.portfolio_metrics = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio metrics: {e}")
            return self._get_default_metrics(portfolio_value)
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        
        return float(returns.quantile(1 - confidence))
    
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence)
        return float(returns[returns <= var].mean())
    
    def _calculate_current_drawdown(self, current_value: Decimal) -> float:
        """Calculate current drawdown from high water mark."""
        if current_value > self.drawdown_high_water_mark:
            self.drawdown_high_water_mark = current_value
        
        if self.drawdown_high_water_mark > 0:
            return float((self.drawdown_high_water_mark - current_value) / self.drawdown_high_water_mark)
        
        return 0.0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        return float(drawdown.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return float(excess_returns.mean() / excess_returns.std() * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std()
        return float(excess_returns.mean() / downside_deviation * np.sqrt(252))
    
    def _calculate_concentration_risk(self, positions: Dict[str, Any], 
                                    portfolio_value: Decimal) -> float:
        """Calculate concentration risk (Herfindahl index)."""
        if not positions or portfolio_value <= 0:
            return 0.0
        
        weights = []
        for pos in positions.values():
            weight = Decimal(str(pos.get('value', 0))) / portfolio_value
            weights.append(float(weight))
        
        # Herfindahl index
        hhi = sum(w**2 for w in weights)
        return hhi
    
    def _calculate_risk_score(self, var_95: float, volatility: float, correlation_risk: float,
                            concentration_risk: float, liquidity_risk: float, leverage: float) -> float:
        """Calculate overall risk score (0-1, higher is riskier)."""
        # Normalize and weight risk components
        var_score = min(abs(var_95) * 10, 1.0)  # Scale VaR
        vol_score = min(volatility / 0.5, 1.0)  # Scale volatility (50% = max)
        corr_score = correlation_risk  # Already 0-1
        conc_score = min(concentration_risk * 2, 1.0)  # Scale concentration
        liq_score = 1 - liquidity_risk  # Invert liquidity (low liquidity = high risk)
        lev_score = min(leverage / 5.0, 1.0)  # Scale leverage (5x = max)
        
        # Weighted average
        risk_score = (
            var_score * 0.25 +
            vol_score * 0.20 +
            corr_score * 0.15 +
            conc_score * 0.15 +
            liq_score * 0.10 +
            lev_score * 0.15
        )
        
        return min(risk_score, 1.0)
    
    def _determine_portfolio_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine portfolio risk level from risk score."""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def check_circuit_breakers(self, portfolio_value: Decimal) -> List[str]:
        """Check all circuit breaker conditions."""
        breakers_triggered = []
        
        try:
            # Daily loss limit
            self._update_daily_loss_tracking()
            daily_loss_pct = float(self.daily_loss_tracker / portfolio_value) if portfolio_value > 0 else 0
            
            if daily_loss_pct >= self.risk_profile.max_daily_loss:
                breakers_triggered.append(f"Daily loss limit exceeded: {daily_loss_pct:.1%}")
                self.emergency_stop_active = True
            
            # Drawdown limit
            current_drawdown = self._calculate_current_drawdown(portfolio_value)
            if current_drawdown >= self.risk_profile.max_drawdown:
                breakers_triggered.append(f"Max drawdown exceeded: {current_drawdown:.1%}")
                self.emergency_stop_active = True
            
            # Leverage limit
            current_leverage = await self._calculate_current_leverage()
            if current_leverage >= self.risk_profile.max_leverage:
                breakers_triggered.append(f"Max leverage exceeded: {current_leverage:.1f}x")
            
            # Portfolio risk score
            if self.portfolio_metrics and self.portfolio_metrics.risk_level == RiskLevel.CRITICAL:
                breakers_triggered.append("Critical portfolio risk level")
            
        except Exception as e:
            self.logger.error(f"Failed to check circuit breakers: {e}")
            breakers_triggered.append("Circuit breaker check error")
        
        return breakers_triggered
    
    def _update_daily_loss_tracking(self):
        """Update daily loss tracking with reset at market open."""
        now = datetime.now()
        
        # Reset daily loss at market open (assuming UTC)
        if now.date() > self.daily_loss_reset_time.date():
            self.daily_loss_tracker = Decimal('0')
            self.daily_loss_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    async def start_risk_monitoring(self):
        """Start continuous risk monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.logger.info("Starting continuous risk monitoring")
        
        asyncio.create_task(self._risk_monitoring_loop())
    
    async def stop_risk_monitoring(self):
        """Stop continuous risk monitoring."""
        self.monitoring_active = False
        self.logger.info("Stopping risk monitoring")
    
    async def _risk_monitoring_loop(self):
        """Main risk monitoring loop."""
        while self.monitoring_active:
            try:
                # Get current portfolio state
                portfolio_value = await self._get_portfolio_value()
                positions = await self._get_current_positions()
                
                # Calculate risk metrics
                metrics = await self.calculate_portfolio_metrics(positions, portfolio_value)
                
                # Check circuit breakers
                breakers = await self.check_circuit_breakers(portfolio_value)
                
                # Generate alerts if needed
                await self._process_risk_alerts(metrics, breakers)
                
                # Update last check time
                self.last_risk_check = datetime.now()
                
                # Wait for next check
                await asyncio.sleep(self.risk_check_interval)
                
            except Exception as e:
                self.logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _process_risk_alerts(self, metrics: RiskMetrics, breakers: List[str]):
        """Process and generate risk alerts."""
        try:
            alerts = []
            
            # High risk level alert
            if metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                alerts.append({
                    'type': 'portfolio_risk',
                    'level': metrics.risk_level.value,
                    'message': f"Portfolio risk level: {metrics.risk_level.value}",
                    'timestamp': datetime.now(),
                    'metrics': {
                        'risk_score': metrics.risk_score,
                        'volatility': metrics.volatility,
                        'leverage': metrics.leverage
                    }
                })
            
            # Circuit breaker alerts
            for breaker in breakers:
                alerts.append({
                    'type': 'circuit_breaker',
                    'level': 'critical',
                    'message': breaker,
                    'timestamp': datetime.now()
                })
            
            # High drawdown alert
            if metrics.current_drawdown > 0.1:  # 10% drawdown
                alerts.append({
                    'type': 'drawdown',
                    'level': 'high',
                    'message': f"High drawdown: {metrics.current_drawdown:.1%}",
                    'timestamp': datetime.now()
                })
            
            # Store alerts
            self.risk_alerts.extend(alerts)
            
            # Keep only recent alerts (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.risk_alerts = [
                alert for alert in self.risk_alerts 
                if alert['timestamp'] > cutoff_time
            ]
            
            # Log critical alerts
            for alert in alerts:
                if alert.get('level') == 'critical':
                    self.logger.warning(f"RISK ALERT: {alert['message']}")
                    
        except Exception as e:
            self.logger.error(f"Failed to process risk alerts: {e}")
    
    # Helper methods (simplified implementations)
    
    async def _get_portfolio_value(self) -> Decimal:
        """Get current portfolio value."""
        # Placeholder implementation
        return Decimal('10000')
    
    async def _get_current_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        # Placeholder implementation
        return {}
    
    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for symbol."""
        # Placeholder implementation
        return Decimal('50000')  # Default BTC price
    
    async def _get_asset_volatility(self, symbol: str) -> float:
        """Get asset volatility."""
        # Check cache first
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol]
        
        # Placeholder implementation
        volatility = 0.15  # 15% default volatility
        self.volatility_cache[symbol] = volatility
        return volatility
    
    async def _calculate_current_leverage(self) -> float:
        """Calculate current portfolio leverage."""
        # Placeholder implementation
        return 1.0
    
    async def _assess_correlation_risk(self, symbol: str) -> float:
        """Assess correlation risk with existing positions."""
        # Placeholder implementation
        return 0.3
    
    async def _assess_market_conditions(self) -> str:
        """Assess current market conditions."""
        # Placeholder implementation
        return "normal"
    
    async def _assess_liquidity_risk(self, symbol: str) -> float:
        """Assess liquidity risk for symbol."""
        # Placeholder implementation
        return 0.8  # 80% liquidity score
    
    async def _get_historical_win_rate(self, symbol: str) -> float:
        """Get historical win rate for symbol."""
        # Placeholder implementation
        return 0.55  # 55% win rate
    
    async def _get_average_win(self, symbol: str) -> float:
        """Get average win amount."""
        # Placeholder implementation
        return 100.0
    
    async def _get_average_loss(self, symbol: str) -> float:
        """Get average loss amount."""
        # Placeholder implementation
        return 80.0
    
    async def _calculate_portfolio_returns(self, positions: Dict[str, Any]) -> pd.Series:
        """Calculate portfolio returns."""
        # Placeholder implementation
        return pd.Series([0.01, -0.005, 0.02, 0.015, -0.01])  # Sample returns
    
    async def _calculate_portfolio_beta(self, positions: Dict[str, Any]) -> float:
        """Calculate portfolio beta."""
        # Placeholder implementation
        return 1.2
    
    async def _calculate_correlation_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate correlation risk."""
        # Placeholder implementation
        return 0.4
    
    async def _calculate_liquidity_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate liquidity risk."""
        # Placeholder implementation
        return 0.7
    
    def _get_conservative_assessment(self, symbol: str, side: str, size: Decimal, 
                                   entry_price: Decimal) -> TradeRiskAssessment:
        """Get conservative risk assessment on error."""
        return TradeRiskAssessment(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            stop_loss=None,
            take_profit=None,
            risk_amount=size * entry_price * Decimal('0.05'),  # 5% risk
            risk_percentage=0.05,
            position_size_recommendation=Decimal('0'),
            risk_reward_ratio=1.0,
            probability_of_success=0.5,
            expected_value=0.0,
            risk_level=RiskLevel.HIGH,
            recommended_action=RiskAction.HALT_TRADING,
            risk_factors=["Assessment error - conservative approach"]
        )
    
    def _get_default_metrics(self, portfolio_value: Decimal) -> RiskMetrics:
        """Get default risk metrics on error."""
        return RiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=Decimal('0'),
            leverage=0.0,
            var_95=0.0,
            expected_shortfall=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            volatility=0.0,
            beta=1.0,
            correlation_risk=0.0,
            concentration_risk=0.0,
            liquidity_risk=1.0,
            risk_score=0.5,
            risk_level=RiskLevel.MEDIUM
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk management summary."""
        try:
            return {
                'risk_profile': {
                    'max_portfolio_risk': self.risk_profile.max_portfolio_risk,
                    'max_daily_loss': self.risk_profile.max_daily_loss,
                    'max_drawdown': self.risk_profile.max_drawdown,
                    'max_leverage': self.risk_profile.max_leverage,
                    'volatility_target': self.risk_profile.volatility_target
                },
                'current_state': {
                    'monitoring_active': self.monitoring_active,
                    'emergency_stop_active': self.emergency_stop_active,
                    'last_risk_check': self.last_risk_check.isoformat(),
                    'daily_loss_tracker': float(self.daily_loss_tracker)
                },
                'portfolio_metrics': {
                    'risk_level': self.portfolio_metrics.risk_level.value if self.portfolio_metrics else 'unknown',
                    'risk_score': self.portfolio_metrics.risk_score if self.portfolio_metrics else 0,
                    'leverage': self.portfolio_metrics.leverage if self.portfolio_metrics else 0
                },
                'active_positions': len(self.position_risks),
                'recent_alerts': len(self.risk_alerts),
                'system_version': '1.0.0 - Unified Risk Management'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate risk summary: {e}")
            return {'error': 'Unable to generate risk summary'}