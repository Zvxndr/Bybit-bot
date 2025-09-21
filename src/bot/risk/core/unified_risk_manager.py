"""
Unified Risk Management System

Consolidates features from the three previous risk management systems:
- Australian-focused risk management with tax-aware position sizing
- Advanced algorithmic position sizing (Kelly, Risk Parity, Volatility Targeting)
- Dynamic risk adjustment with volatility and correlation analysis

This unified system provides:
- Tax-optimized position sizing for Australian traders
- Advanced risk algorithms (Kelly Criterion, Risk Parity, Max Drawdown)
- Real-time risk monitoring with circuit breakers
- Dynamic risk adjustment based on market regimes
- Portfolio risk analysis with correlation monitoring
- Comprehensive risk metrics and alerting
"""

from enum import Enum
from decimal import Decimal
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

# Import unified configuration system
try:
    from src.bot.core.config import (
        UnifiedConfigurationManager, UnifiedConfigurationSchema,
        Environment, TradingMode
    )
    from src.bot.core.config.integrations import RiskManagementConfigAdapter
except ImportError:
    # Fallback if unified config not available
    UnifiedConfigurationManager = None
    RiskManagementConfigAdapter = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class RiskLevel(Enum):
    """Risk level classification"""
    VERY_LOW = "very_low"
    LOW = "low" 
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

class PositionSizeMethod(Enum):
    """Position sizing methods"""
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGETING = "volatility_targeting"
    RISK_PARITY = "risk_parity"
    MAX_DRAWDOWN = "max_drawdown"
    TAX_OPTIMIZED = "tax_optimized"

class MarketRegime(Enum):
    """Market regime classification"""
    LOW_VOLATILITY = "low_volatility"
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class RiskParameters:
    """Unified risk parameters configuration"""
    # Basic risk limits
    max_portfolio_risk: Decimal = Decimal('0.02')  # 2% portfolio risk
    max_position_size: Decimal = Decimal('0.10')   # 10% max position
    max_correlation: float = 0.7                   # Max correlation between positions
    
    # Australian tax optimization
    cgt_discount_threshold: int = 365              # Days to hold for CGT discount
    tax_rate: Decimal = Decimal('0.325')           # Marginal tax rate
    enable_tax_optimization: bool = True
    
    # Dynamic adjustment parameters
    volatility_lookback: int = 30                  # Days for volatility calculation
    correlation_lookback: int = 60                 # Days for correlation analysis
    regime_detection_window: int = 90              # Days for regime detection
    
    # Risk limits by regime
    regime_risk_multipliers: Dict[MarketRegime, float] = None
    
    def __post_init__(self):
        if self.regime_risk_multipliers is None:
            self.regime_risk_multipliers = {
                MarketRegime.LOW_VOLATILITY: 1.2,
                MarketRegime.NORMAL: 1.0,
                MarketRegime.HIGH_VOLATILITY: 0.8,
                MarketRegime.CRISIS: 0.5,
                MarketRegime.TRENDING_UP: 1.1,
                MarketRegime.TRENDING_DOWN: 0.9
            }
    
    @classmethod
    def from_unified_config(cls, config: 'UnifiedConfigurationSchema') -> 'RiskParameters':
        """Create RiskParameters from unified configuration"""
        if not config:
            return cls()
        
        # Get risk-specific configuration via adapter
        if RiskManagementConfigAdapter:
            adapter = RiskManagementConfigAdapter(config)
            risk_config = adapter.get_risk_config()
            compliance_config = adapter.get_australian_compliance_config()
            
            # Map unified config to risk parameters
            return cls(
                max_portfolio_risk=Decimal(str(risk_config.get('portfolio_drawdown_limit', 0.02))), 
                max_position_size=Decimal('0.10'),  # Use default for now
                max_correlation=0.7,  # Use default for now
                cgt_discount_threshold=365,
                tax_rate=Decimal('0.325'),
                enable_tax_optimization=compliance_config.get('enable_tax_reporting', True),
                volatility_lookback=30,
                correlation_lookback=60,
                regime_detection_window=90
            )
        
        return cls()

@dataclass
class PositionRisk:
    """Risk assessment for a specific position"""
    symbol: str
    position_size: Decimal
    risk_amount: Decimal
    risk_percentage: float
    var_1d: float
    expected_shortfall: float
    correlation_risk: float
    volatility: float
    risk_level: RiskLevel
    tax_impact: Optional[Decimal] = None
    cgt_discount_eligible: bool = False
    hold_period: Optional[int] = None

@dataclass
class PortfolioRiskMetrics:
    """Comprehensive portfolio risk metrics"""
    total_var: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_matrix: pd.DataFrame
    concentration_risk: float
    regime: MarketRegime
    risk_score: float
    alerts: List[str]

# ============================================================================
# POSITION SIZING ENGINES
# ============================================================================

class PositionSizer(ABC):
    """Abstract base class for position sizing methods"""
    
    @abstractmethod
    def calculate_size(self, symbol: str, portfolio_value: Decimal, 
                      returns: pd.Series, **kwargs) -> Decimal:
        """Calculate optimal position size"""
        pass

class KellyCriterionSizer(PositionSizer):
    """Kelly Criterion position sizing"""
    
    def calculate_size(self, symbol: str, portfolio_value: Decimal,
                      returns: pd.Series, **kwargs) -> Decimal:
        """Calculate Kelly optimal position size"""
        if len(returns) < 30:
            return portfolio_value * Decimal('0.01')  # Conservative fallback
            
        # Calculate Kelly fraction: f = (bp - q) / b
        mean_return = returns.mean()
        variance = returns.var()
        
        if variance == 0:
            return portfolio_value * Decimal('0.01')
            
        # Kelly fraction
        kelly_fraction = mean_return / variance
        
        # Cap Kelly fraction to prevent excessive leverage
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        return portfolio_value * Decimal(str(kelly_fraction))

class RiskParitySizer(PositionSizer):
    """Risk Parity position sizing"""
    
    def calculate_size(self, symbol: str, portfolio_value: Decimal,
                      returns: pd.Series, **kwargs) -> Decimal:
        """Calculate risk parity position size"""
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        if volatility == 0:
            return portfolio_value * Decimal('0.01')
            
        # Target risk contribution
        target_risk = kwargs.get('target_risk', 0.02)  # 2% risk per position
        
        # Position size = target_risk / volatility
        position_size = target_risk / volatility
        
        return portfolio_value * Decimal(str(min(position_size, 0.10)))

class VolatilityTargetingSizer(PositionSizer):
    """Volatility targeting position sizing"""
    
    def calculate_size(self, symbol: str, portfolio_value: Decimal,
                      returns: pd.Series, **kwargs) -> Decimal:
        """Calculate volatility-targeted position size"""
        target_volatility = kwargs.get('target_volatility', 0.15)  # 15% target vol
        
        asset_volatility = returns.std() * np.sqrt(252)
        
        if asset_volatility == 0:
            return portfolio_value * Decimal('0.01')
            
        # Scale position to achieve target volatility
        vol_scalar = target_volatility / asset_volatility
        
        return portfolio_value * Decimal(str(min(vol_scalar, 0.10)))

class TaxOptimizedSizer(PositionSizer):
    """Australian tax-optimized position sizing"""
    
    def __init__(self, risk_params: RiskParameters):
        self.risk_params = risk_params
    
    def calculate_size(self, symbol: str, portfolio_value: Decimal,
                      returns: pd.Series, **kwargs) -> Decimal:
        """Calculate tax-optimized position size"""
        base_size = portfolio_value * self.risk_params.max_position_size
        
        # Check if position qualifies for CGT discount
        hold_period = kwargs.get('expected_hold_period', 180)  # days
        
        if hold_period >= self.risk_params.cgt_discount_threshold:
            # Increase position size for CGT discount eligible trades
            tax_multiplier = Decimal('1.2')  # 20% larger for tax efficiency
            base_size *= tax_multiplier
            
        return min(base_size, portfolio_value * Decimal('0.15'))  # Cap at 15%

# ============================================================================
# MARKET REGIME DETECTION
# ============================================================================

class MarketRegimeDetector:
    """Detects market regimes for dynamic risk adjustment"""
    
    def __init__(self, lookback_window: int = 90):
        self.lookback_window = lookback_window
    
    def detect_regime(self, returns: pd.Series) -> MarketRegime:
        """Detect current market regime"""
        if len(returns) < self.lookback_window:
            return MarketRegime.NORMAL
            
        recent_returns = returns.tail(self.lookback_window)
        
        # Calculate regime indicators
        volatility = recent_returns.std() * np.sqrt(252)
        trend = recent_returns.mean() * 252
        skewness = recent_returns.skew()
        
        # Regime classification logic
        if volatility > 0.40:  # > 40% annualized volatility
            return MarketRegime.CRISIS
        elif volatility > 0.25:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.10:
            return MarketRegime.LOW_VOLATILITY
        elif trend > 0.20:  # > 20% annual trend
            return MarketRegime.TRENDING_UP
        elif trend < -0.20:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.NORMAL

# ============================================================================
# UNIFIED RISK MANAGER
# ============================================================================

class UnifiedRiskManager:
    """
    Unified Risk Management System
    
    Combines Australian tax-aware risk management with advanced algorithms
    and dynamic risk adjustment capabilities.
    """
    
    def __init__(self, risk_params: RiskParameters = None, unified_config: 'UnifiedConfigurationSchema' = None):
        # Try to use unified configuration if available
        if unified_config and not risk_params:
            try:
                self.risk_params = RiskParameters.from_unified_config(unified_config)
                logger.info("Risk Manager initialized with unified configuration")
            except Exception as e:
                logger.warning(f"Failed to load unified configuration: {e}, using defaults")
                self.risk_params = RiskParameters()
        else:
            self.risk_params = risk_params or RiskParameters()
        
        # Store unified config reference for runtime updates
        self.unified_config = unified_config
        
        # Initialize position sizers
        self.sizers = {
            PositionSizeMethod.KELLY_CRITERION: KellyCriterionSizer(),
            PositionSizeMethod.RISK_PARITY: RiskParitySizer(),
            PositionSizeMethod.VOLATILITY_TARGETING: VolatilityTargetingSizer(),
            PositionSizeMethod.TAX_OPTIMIZED: TaxOptimizedSizer(self.risk_params)
        }
        
        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector(
            self.risk_params.regime_detection_window
        )
        
        # Risk monitoring state
        self.portfolio_metrics: Optional[PortfolioRiskMetrics] = None
        self.position_risks: Dict[str, PositionRisk] = {}
        self.alerts: List[Dict[str, Any]] = []
        
        config_source = "unified configuration" if unified_config else "default parameters"
        logger.info(f"Unified Risk Manager initialized with {config_source}")
    
    def reload_configuration(self, unified_config: 'UnifiedConfigurationSchema' = None):
        """Reload configuration from unified configuration system"""
        if unified_config:
            self.unified_config = unified_config
        
        if self.unified_config:
            try:
                new_params = RiskParameters.from_unified_config(self.unified_config)
                self.risk_params = new_params
                
                # Reinitialize components that depend on config
                self.regime_detector = MarketRegimeDetector(
                    self.risk_params.regime_detection_window
                )
                
                # Update tax-optimized sizer with new parameters
                self.sizers[PositionSizeMethod.TAX_OPTIMIZED] = TaxOptimizedSizer(self.risk_params)
                
                logger.info("Risk Manager configuration reloaded from unified system")
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")
    
    async def calculate_position_size(self, symbol: str, side: str,
                                    portfolio_value: Decimal,
                                    returns: pd.Series,
                                    method: PositionSizeMethod = PositionSizeMethod.TAX_OPTIMIZED,
                                    **kwargs) -> Tuple[Decimal, PositionRisk]:
        """Calculate optimal position size using specified method"""
        
        # Detect current market regime
        current_regime = self.regime_detector.detect_regime(returns)
        
        # Apply regime-based risk adjustment
        regime_multiplier = self.risk_params.regime_risk_multipliers.get(
            current_regime, 1.0
        )
        
        # Calculate base position size
        sizer = self.sizers.get(method)
        if not sizer:
            raise ValueError(f"Unknown position sizing method: {method}")
            
        base_size = sizer.calculate_size(symbol, portfolio_value, returns, **kwargs)
        
        # Apply regime adjustment
        adjusted_size = base_size * Decimal(str(regime_multiplier))
        
        # Calculate risk metrics
        position_risk = await self._calculate_position_risk(
            symbol, adjusted_size, portfolio_value, returns, current_regime
        )
        
        # Apply risk limits
        final_size = self._apply_risk_limits(adjusted_size, position_risk)
        
        # Update position risk with final size
        position_risk.position_size = final_size
        
        # Store position risk
        self.position_risks[symbol] = position_risk
        
        logger.info(f"Position size calculated for {symbol}: {final_size} "
                   f"(method: {method.value}, regime: {current_regime.value})")
        
        return final_size, position_risk
    
    async def _calculate_position_risk(self, symbol: str, position_size: Decimal,
                                     portfolio_value: Decimal, returns: pd.Series,
                                     regime: MarketRegime) -> PositionRisk:
        """Calculate comprehensive position risk metrics"""
        
        # Basic risk calculations
        position_percentage = float(position_size / portfolio_value)
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Value at Risk (1-day, 95% confidence)
        var_1d = returns.quantile(0.05) * float(position_size)
        
        # Expected Shortfall (Conditional VaR)
        tail_returns = returns[returns <= returns.quantile(0.05)]
        expected_shortfall = tail_returns.mean() * float(position_size) if len(tail_returns) > 0 else var_1d
        
        # Risk amount (maximum loss)
        risk_amount = position_size * Decimal(str(abs(var_1d)))
        
        # Correlation risk (placeholder - would need other positions)
        correlation_risk = 0.0
        
        # Determine risk level
        risk_level = self._determine_risk_level(position_percentage, volatility)
        
        # Tax calculations (if applicable)
        tax_impact = None
        cgt_discount_eligible = False
        
        if self.risk_params.enable_tax_optimization:
            expected_hold_period = 365  # Default assumption
            cgt_discount_eligible = expected_hold_period >= self.risk_params.cgt_discount_threshold
            
            # Estimate tax impact on profits
            expected_return = returns.mean() * expected_hold_period
            if expected_return > 0:
                tax_rate = self.risk_params.tax_rate
                if cgt_discount_eligible:
                    tax_rate *= Decimal('0.5')  # 50% CGT discount
                
                tax_impact = position_size * Decimal(str(expected_return)) * tax_rate
        
        return PositionRisk(
            symbol=symbol,
            position_size=position_size,
            risk_amount=risk_amount,
            risk_percentage=position_percentage,
            var_1d=var_1d,
            expected_shortfall=expected_shortfall,
            correlation_risk=correlation_risk,
            volatility=volatility,
            risk_level=risk_level,
            tax_impact=tax_impact,
            cgt_discount_eligible=cgt_discount_eligible
        )
    
    def _determine_risk_level(self, position_percentage: float, volatility: float) -> RiskLevel:
        """Determine risk level based on position size and volatility"""
        
        risk_score = position_percentage * volatility
        
        if risk_score < 0.005:  # < 0.5%
            return RiskLevel.VERY_LOW
        elif risk_score < 0.01:  # < 1%
            return RiskLevel.LOW
        elif risk_score < 0.02:  # < 2%
            return RiskLevel.MODERATE
        elif risk_score < 0.04:  # < 4%
            return RiskLevel.HIGH
        elif risk_score < 0.08:  # < 8%
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME
    
    def _apply_risk_limits(self, position_size: Decimal, position_risk: PositionRisk) -> Decimal:
        """Apply risk limits to position size"""
        
        # Position size limits
        max_position = self.risk_params.max_position_size
        if position_risk.risk_percentage > float(max_position):
            logger.warning(f"Position size exceeds limit, reducing from "
                         f"{position_risk.risk_percentage:.1%} to {max_position:.1%}")
            return position_size * (max_position / Decimal(str(position_risk.risk_percentage)))
        
        # Risk level limits
        if position_risk.risk_level in [RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
            reduction_factor = Decimal('0.5')  # Reduce by 50%
            logger.warning(f"High risk level detected ({position_risk.risk_level.value}), "
                         f"reducing position size by {reduction_factor:.0%}")
            return position_size * reduction_factor
        
        return position_size
    
    async def calculate_portfolio_risk(self, positions: Dict[str, Dict[str, Any]],
                                     market_data: Dict[str, pd.Series]) -> PortfolioRiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        if not positions:
            return self._empty_portfolio_metrics()
        
        # Calculate correlation matrix
        returns_df = pd.DataFrame({
            symbol: data for symbol, data in market_data.items()
            if symbol in positions
        })
        correlation_matrix = returns_df.corr()
        
        # Portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(positions, market_data)
        
        # Risk metrics
        total_var = portfolio_returns.quantile(0.05)
        tail_returns = portfolio_returns[portfolio_returns <= total_var]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else total_var
        
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        
        # Detect current regime
        regime = self.regime_detector.detect_regime(portfolio_returns)
        
        # Concentration risk
        concentration_risk = self._calculate_concentration_risk(positions)
        
        # Overall risk score
        risk_score = self._calculate_portfolio_risk_score(
            total_var, max_drawdown, concentration_risk, regime
        )
        
        # Generate alerts
        alerts = self._generate_portfolio_alerts(
            total_var=total_var,
            max_drawdown=max_drawdown,
            concentration_risk=concentration_risk,
            regime=regime
        )
        
        portfolio_metrics = PortfolioRiskMetrics(
            total_var=total_var,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            beta=1.0,  # Placeholder - would need benchmark
            correlation_matrix=correlation_matrix,
            concentration_risk=concentration_risk,
            regime=regime,
            risk_score=risk_score,
            alerts=alerts
        )
        
        self.portfolio_metrics = portfolio_metrics
        return portfolio_metrics
    
    def _calculate_portfolio_returns(self, positions: Dict[str, Dict[str, Any]], 
                                   market_data: Dict[str, pd.Series]) -> pd.Series:
        """Calculate portfolio returns based on positions and market data"""
        
        total_value = sum(pos['value'] for pos in positions.values())
        if total_value == 0:
            return pd.Series(dtype=float)
        
        # Calculate weighted returns
        portfolio_returns = pd.Series(dtype=float)
        
        for symbol, position in positions.items():
            if symbol in market_data:
                weight = position['value'] / total_value
                asset_returns = market_data[symbol]
                
                if portfolio_returns.empty:
                    portfolio_returns = weight * asset_returns
                else:
                    # Align indices
                    common_index = portfolio_returns.index.intersection(asset_returns.index)
                    portfolio_returns = (portfolio_returns.loc[common_index] + 
                                       weight * asset_returns.loc[common_index])
        
        return portfolio_returns
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if returns.empty:
            return 0.0
            
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.empty or returns.std() == 0:
            return 0.0
            
        excess_return = returns.mean() * 252 - risk_free_rate  # Annualized
        volatility = returns.std() * np.sqrt(252)
        
        return excess_return / volatility
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if returns.empty:
            return 0.0
            
        excess_return = returns.mean() * 252 - risk_free_rate
        negative_returns = returns[returns < 0]
        
        if negative_returns.empty:
            return float('inf') if excess_return > 0 else 0.0
            
        downside_deviation = negative_returns.std() * np.sqrt(252)
        
        return excess_return / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_concentration_risk(self, positions: Dict[str, Dict[str, Any]]) -> float:
        """Calculate portfolio concentration risk (Herfindahl index)"""
        if not positions:
            return 0.0
            
        total_value = sum(pos['value'] for pos in positions.values())
        if total_value == 0:
            return 0.0
            
        # Calculate Herfindahl-Hirschman Index
        weights = [pos['value'] / total_value for pos in positions.values()]
        hhi = sum(w ** 2 for w in weights)
        
        return hhi
    
    def _calculate_portfolio_risk_score(self, var: float, max_drawdown: float,
                                      concentration_risk: float, regime: MarketRegime) -> float:
        """Calculate overall portfolio risk score (0-100)"""
        
        # Base risk score from VaR and drawdown
        var_score = min(abs(var) * 100, 50)  # Cap at 50
        drawdown_score = min(max_drawdown * 100, 30)  # Cap at 30
        concentration_score = min(concentration_risk * 20, 20)  # Cap at 20
        
        base_score = var_score + drawdown_score + concentration_score
        
        # Regime adjustment
        regime_adjustments = {
            MarketRegime.LOW_VOLATILITY: 0.8,
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOLATILITY: 1.3,
            MarketRegime.CRISIS: 1.5,
            MarketRegime.TRENDING_UP: 0.9,
            MarketRegime.TRENDING_DOWN: 1.2
        }
        
        adjusted_score = base_score * regime_adjustments.get(regime, 1.0)
        
        return min(adjusted_score, 100)  # Cap at 100
    
    def _generate_portfolio_alerts(self, total_var: float, max_drawdown: float,
                                 concentration_risk: float, regime: MarketRegime) -> List[str]:
        """Generate portfolio risk alerts"""
        alerts = []
        
        # VaR alerts
        if abs(total_var) > 0.05:  # > 5% daily VaR
            alerts.append(f"HIGH RISK: Daily VaR exceeds 5% ({abs(total_var):.1%})")
        elif abs(total_var) > 0.03:  # > 3% daily VaR
            alerts.append(f"WARNING: Daily VaR elevated ({abs(total_var):.1%})")
        
        # Drawdown alerts
        if max_drawdown > 0.20:  # > 20% drawdown
            alerts.append(f"CRITICAL: Maximum drawdown exceeds 20% ({max_drawdown:.1%})")
        elif max_drawdown > 0.10:  # > 10% drawdown
            alerts.append(f"WARNING: Maximum drawdown elevated ({max_drawdown:.1%})")
        
        # Concentration alerts
        if concentration_risk > 0.5:  # > 50% in single position
            alerts.append(f"HIGH CONCENTRATION: Portfolio concentration risk elevated ({concentration_risk:.1%})")
        
        # Regime alerts
        if regime in [MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY]:
            alerts.append(f"MARKET ALERT: {regime.value.replace('_', ' ').title()} regime detected")
        
        return alerts
    
    def _empty_portfolio_metrics(self) -> PortfolioRiskMetrics:
        """Return empty portfolio metrics for empty portfolio"""
        return PortfolioRiskMetrics(
            total_var=0.0,
            expected_shortfall=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            beta=1.0,
            correlation_matrix=pd.DataFrame(),
            concentration_risk=0.0,
            regime=MarketRegime.NORMAL,
            risk_score=0.0,
            alerts=[]
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'risk_parameters': {
                'max_portfolio_risk': float(self.risk_params.max_portfolio_risk),
                'max_position_size': float(self.risk_params.max_position_size),
                'tax_optimization_enabled': self.risk_params.enable_tax_optimization
            },
            'positions': {
                symbol: {
                    'size': float(risk.position_size),
                    'risk_level': risk.risk_level.value,
                    'risk_percentage': risk.risk_percentage,
                    'var_1d': risk.var_1d,
                    'volatility': risk.volatility,
                    'cgt_discount_eligible': risk.cgt_discount_eligible
                }
                for symbol, risk in self.position_risks.items()
            }
        }
        
        if self.portfolio_metrics:
            summary['portfolio'] = {
                'total_var': self.portfolio_metrics.total_var,
                'max_drawdown': self.portfolio_metrics.max_drawdown,
                'sharpe_ratio': self.portfolio_metrics.sharpe_ratio,
                'risk_score': self.portfolio_metrics.risk_score,
                'regime': self.portfolio_metrics.regime.value,
                'alerts': self.portfolio_metrics.alerts
            }
        
        return summary
    
    def should_reject_trade(self, symbol: str, position_size: Decimal) -> Tuple[bool, str]:
        """Determine if a trade should be rejected based on risk limits"""
        
        if symbol in self.position_risks:
            position_risk = self.position_risks[symbol]
            
            # Check risk level
            if position_risk.risk_level == RiskLevel.EXTREME:
                return True, f"EXTREME risk level for {symbol}"
            
            # Check portfolio risk limits
            if self.portfolio_metrics:
                if self.portfolio_metrics.risk_score > 80:
                    return True, f"Portfolio risk score too high ({self.portfolio_metrics.risk_score:.1f})"
                
                if abs(self.portfolio_metrics.total_var) > 0.05:
                    return True, f"Portfolio VaR exceeds limit ({abs(self.portfolio_metrics.total_var):.1%})"
        
        return False, ""

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'UnifiedRiskManager',
    'RiskParameters', 
    'PositionRisk',
    'PortfolioRiskMetrics',
    'RiskLevel',
    'PositionSizeMethod',
    'MarketRegime',
    'AlertLevel',
    'MarketRegimeDetector',
    'KellyCriterionSizer',
    'RiskParitySizer', 
    'VolatilityTargetingSizer',
    'TaxOptimizedSizer'
]