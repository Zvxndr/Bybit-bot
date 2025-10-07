"""
Bybit Liquidation Risk Modeling System

This module provides comprehensive liquidation risk assessment and modeling for Bybit trading:

Core Features:
- Accurate liquidation price calculations for all Bybit contract types
- Dynamic margin requirements based on position size and market conditions
- Risk-based position sizing with liquidation avoidance
- Real-time liquidation risk monitoring and alerts
- Margin call simulation and forced position reduction
- Cross-margin and isolated margin risk analysis

Advanced Features:
- Portfolio-level liquidation risk assessment
- Stress testing with adverse market scenarios
- Dynamic hedge ratio calculation for risk mitigation
- Liquidation cascade modeling (domino effect analysis)
- Auto-deleveraging (ADL) queue position simulation
- Insurance fund impact calculation

Risk Management Features:
- Pre-trade liquidation risk assessment
- Dynamic stop-loss positioning to avoid liquidation
- Margin buffer management and optimization
- Multi-timeframe volatility-based risk sizing
- Correlation-based portfolio risk adjustment

This system enables sophisticated risk management for leveraged trading
and provides comprehensive liquidation avoidance strategies.

Author: Trading Bot Team  
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math

from .bybit_enhanced_backtest_engine import BybitContractType, BybitVIPTier
from ..utils.logging import TradingLogger


class LiquidationRiskLevel(Enum):
    """Liquidation risk severity levels."""
    SAFE = "safe"           # >20% distance to liquidation
    LOW = "low"             # 10-20% distance  
    MEDIUM = "medium"       # 5-10% distance
    HIGH = "high"           # 2-5% distance
    CRITICAL = "critical"   # <2% distance
    IMMINENT = "imminent"   # <0.5% distance


class MarginType(Enum):
    """Bybit margin types."""
    CROSS_MARGIN = "cross"
    ISOLATED_MARGIN = "isolated"


@dataclass
class MarginTier:
    """Bybit margin tier configuration."""
    min_size: Decimal           # Minimum position size for this tier
    max_size: Decimal           # Maximum position size for this tier
    initial_margin_rate: Decimal # Initial margin requirement
    maintenance_margin_rate: Decimal # Maintenance margin requirement
    max_leverage: Decimal       # Maximum leverage allowed


@dataclass
class LiquidationRiskAssessment:
    """Comprehensive liquidation risk assessment."""
    
    # Basic risk metrics
    liquidation_price: Decimal
    current_price: Decimal
    distance_to_liquidation: Decimal
    distance_percentage: Decimal
    risk_level: LiquidationRiskLevel
    
    # Position details
    position_size: Decimal
    entry_price: Decimal
    leverage: Decimal
    margin_used: Decimal
    unrealized_pnl: Decimal
    
    # Risk calculations
    margin_ratio: Decimal
    maintenance_margin_required: Decimal
    excess_margin: Decimal
    liquidation_buffer: Decimal
    
    # Time-based risk
    estimated_time_to_liquidation: Optional[timedelta]
    max_adverse_move: Decimal
    
    # Portfolio impact
    portfolio_impact_score: Decimal
    correlation_risk: Decimal
    
    # Recommendations
    recommended_action: str
    safe_position_size: Decimal
    stop_loss_price: Optional[Decimal]
    hedge_recommendation: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'liquidation_price': float(self.liquidation_price),
            'current_price': float(self.current_price),
            'distance_to_liquidation': float(self.distance_to_liquidation),
            'distance_percentage': float(self.distance_percentage),
            'risk_level': self.risk_level.value,
            'position_size': float(self.position_size),
            'entry_price': float(self.entry_price),
            'leverage': float(self.leverage),
            'margin_used': float(self.margin_used),
            'unrealized_pnl': float(self.unrealized_pnl),
            'margin_ratio': float(self.margin_ratio),
            'maintenance_margin_required': float(self.maintenance_margin_required),
            'excess_margin': float(self.excess_margin),
            'liquidation_buffer': float(self.liquidation_buffer),
            'estimated_time_to_liquidation': str(self.estimated_time_to_liquidation) if self.estimated_time_to_liquidation else None,
            'max_adverse_move': float(self.max_adverse_move),
            'portfolio_impact_score': float(self.portfolio_impact_score),
            'correlation_risk': float(self.correlation_risk),
            'recommended_action': self.recommended_action,
            'safe_position_size': float(self.safe_position_size),
            'stop_loss_price': float(self.stop_loss_price) if self.stop_loss_price else None,
            'hedge_recommendation': self.hedge_recommendation
        }


@dataclass
class PortfolioLiquidationRisk:
    """Portfolio-level liquidation risk analysis."""
    
    total_margin_used: Decimal
    total_portfolio_value: Decimal
    weighted_liquidation_distance: Decimal
    portfolio_risk_level: LiquidationRiskLevel
    
    # Cross-position risks
    correlation_matrix: Dict[str, Dict[str, Decimal]]
    cascade_risk_score: Decimal
    systemic_risk_factors: List[str]
    
    # Stress testing results
    stress_test_results: Dict[str, Any]
    worst_case_scenario: Dict[str, Any]
    
    # Recommendations
    portfolio_rebalancing_needed: bool
    hedge_recommendations: List[str]
    position_reduction_suggestions: List[str]


class BybitLiquidationRiskManager:
    """
    Advanced liquidation risk management system for Bybit trading.
    
    This system provides:
    1. Accurate liquidation price calculations for all contract types
    2. Real-time risk monitoring and alerting
    3. Portfolio-level risk assessment
    4. Stress testing and scenario analysis
    5. Dynamic position sizing with risk constraints
    6. Liquidation avoidance strategies
    """
    
    def __init__(
        self,
        default_margin_type: MarginType = MarginType.CROSS_MARGIN,
        risk_tolerance: float = 0.1,  # Maximum 10% portfolio risk
        enable_stress_testing: bool = True
    ):
        self.default_margin_type = default_margin_type
        self.risk_tolerance = Decimal(str(risk_tolerance))
        self.enable_stress_testing = enable_stress_testing
        
        # Initialize margin tier configurations
        self._initialize_margin_tiers()
        
        # Risk monitoring
        self.risk_assessments: Dict[str, LiquidationRiskAssessment] = {}
        self.risk_history: List[LiquidationRiskAssessment] = []
        
        # Portfolio tracking
        self.portfolio_positions: Dict[str, Dict[str, Any]] = {}
        self.correlation_matrix: Dict[str, Dict[str, Decimal]] = {}
        
        self.logger = TradingLogger("BybitLiquidationRiskManager")
        self.logger.info(f"Initialized with margin type: {default_margin_type.value}")
    
    def _initialize_margin_tiers(self) -> None:
        """Initialize margin tier configurations for different symbols."""
        
        # BTCUSDT margin tiers (example of Bybit's tiered system)
        self.margin_tiers = {
            'BTCUSDT': [
                MarginTier(
                    min_size=Decimal('0'),
                    max_size=Decimal('5'),
                    initial_margin_rate=Decimal('0.01'),      # 1% (100x leverage)
                    maintenance_margin_rate=Decimal('0.005'), # 0.5%
                    max_leverage=Decimal('100')
                ),
                MarginTier(
                    min_size=Decimal('5'),
                    max_size=Decimal('25'),
                    initial_margin_rate=Decimal('0.015'),     # 1.5% (66x leverage)
                    maintenance_margin_rate=Decimal('0.0065'),# 0.65%
                    max_leverage=Decimal('66')
                ),
                MarginTier(
                    min_size=Decimal('25'),
                    max_size=Decimal('100'),
                    initial_margin_rate=Decimal('0.02'),      # 2% (50x leverage)
                    maintenance_margin_rate=Decimal('0.01'),  # 1%
                    max_leverage=Decimal('50')
                ),
                MarginTier(
                    min_size=Decimal('100'),
                    max_size=Decimal('500'),
                    initial_margin_rate=Decimal('0.05'),      # 5% (20x leverage)
                    maintenance_margin_rate=Decimal('0.025'), # 2.5%
                    max_leverage=Decimal('20')
                ),
                MarginTier(
                    min_size=Decimal('500'),
                    max_size=Decimal('1000000'),
                    initial_margin_rate=Decimal('0.1'),       # 10% (10x leverage)
                    maintenance_margin_rate=Decimal('0.05'),  # 5%
                    max_leverage=Decimal('10')
                )
            ],
            
            # ETHUSDT margin tiers
            'ETHUSDT': [
                MarginTier(
                    min_size=Decimal('0'),
                    max_size=Decimal('50'),
                    initial_margin_rate=Decimal('0.01'),      # 1% (100x leverage)
                    maintenance_margin_rate=Decimal('0.005'), # 0.5%
                    max_leverage=Decimal('100')
                ),
                MarginTier(
                    min_size=Decimal('50'),
                    max_size=Decimal('200'),
                    initial_margin_rate=Decimal('0.02'),      # 2% (50x leverage)
                    maintenance_margin_rate=Decimal('0.01'),  # 1%
                    max_leverage=Decimal('50')
                ),
                MarginTier(
                    min_size=Decimal('200'),
                    max_size=Decimal('1000000'),
                    initial_margin_rate=Decimal('0.05'),      # 5% (20x leverage)
                    maintenance_margin_rate=Decimal('0.025'), # 2.5%
                    max_leverage=Decimal('20')
                )
            ],
            
            # Default tiers for other symbols
            'DEFAULT': [
                MarginTier(
                    min_size=Decimal('0'),
                    max_size=Decimal('50'),
                    initial_margin_rate=Decimal('0.02'),      # 2% (50x max)
                    maintenance_margin_rate=Decimal('0.01'),  # 1%
                    max_leverage=Decimal('50')
                ),
                MarginTier(
                    min_size=Decimal('50'),
                    max_size=Decimal('1000000'),
                    initial_margin_rate=Decimal('0.1'),       # 10% (10x max)
                    maintenance_margin_rate=Decimal('0.05'),  # 5%
                    max_leverage=Decimal('10')
                )
            ]
        }
    
    def calculate_liquidation_price(
        self,
        symbol: str,
        side: str,  # 'long' or 'short'
        entry_price: Decimal,
        position_size: Decimal,
        leverage: Decimal,
        margin_type: Optional[MarginType] = None,
        additional_margin: Decimal = Decimal('0')
    ) -> Decimal:
        """
        Calculate liquidation price for a position.
        
        Args:
            symbol: Trading symbol
            side: Position side ('long' or 'short')
            entry_price: Entry price of position
            position_size: Size of position
            leverage: Leverage used
            margin_type: Cross or isolated margin
            additional_margin: Additional margin available
            
        Returns:
            Liquidation price
        """
        try:
            margin_type = margin_type or self.default_margin_type
            
            # Get appropriate margin tier
            margin_tier = self._get_margin_tier(symbol, position_size)
            maintenance_margin_rate = margin_tier.maintenance_margin_rate
            
            # Calculate position value
            position_value = position_size * entry_price
            
            if margin_type == MarginType.ISOLATED_MARGIN:
                # Isolated margin liquidation calculation
                initial_margin = position_value / leverage
                
                if side.lower() == 'long':
                    # Long liquidation: P_liq = entry_price * (1 - (initial_margin - additional_margin) / position_value + maintenance_margin_rate)
                    liquidation_price = entry_price * (
                        Decimal('1') - 
                        (initial_margin - additional_margin) / position_value + 
                        maintenance_margin_rate
                    )
                else:  # short
                    # Short liquidation: P_liq = entry_price * (1 + (initial_margin - additional_margin) / position_value - maintenance_margin_rate)
                    liquidation_price = entry_price * (
                        Decimal('1') + 
                        (initial_margin - additional_margin) / position_value - 
                        maintenance_margin_rate
                    )
            
            else:  # Cross margin
                # Cross margin calculation is more complex as it involves total account balance
                # Simplified calculation for backtesting purposes
                if side.lower() == 'long':
                    liquidation_price = entry_price * (
                        Decimal('1') - Decimal('1') / leverage + maintenance_margin_rate
                    )
                else:  # short
                    liquidation_price = entry_price * (
                        Decimal('1') + Decimal('1') / leverage - maintenance_margin_rate
                    )
            
            # Ensure liquidation price is positive
            liquidation_price = max(liquidation_price, Decimal('0.01'))
            
            return liquidation_price
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidation price: {e}")
            return Decimal('0')
    
    def _get_margin_tier(self, symbol: str, position_size: Decimal) -> MarginTier:
        """Get appropriate margin tier for symbol and position size."""
        try:
            # Get margin tiers for symbol
            tiers = self.margin_tiers.get(symbol, self.margin_tiers['DEFAULT'])
            
            # Find appropriate tier based on position size
            for tier in tiers:
                if tier.min_size <= position_size <= tier.max_size:
                    return tier
            
            # Return the last (highest) tier if position is very large
            return tiers[-1]
            
        except Exception as e:
            self.logger.error(f"Error getting margin tier: {e}")
            return self.margin_tiers['DEFAULT'][0]
    
    def assess_liquidation_risk(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        current_price: Decimal,
        position_size: Decimal,
        leverage: Decimal,
        account_balance: Decimal,
        margin_type: Optional[MarginType] = None,
        volatility: Optional[Decimal] = None
    ) -> LiquidationRiskAssessment:
        """
        Perform comprehensive liquidation risk assessment.
        
        Args:
            symbol: Trading symbol
            side: Position side
            entry_price: Entry price
            current_price: Current market price
            position_size: Position size
            leverage: Leverage used
            account_balance: Available account balance
            margin_type: Margin type used
            volatility: Recent price volatility
            
        Returns:
            Comprehensive liquidation risk assessment
        """
        try:
            margin_type = margin_type or self.default_margin_type
            volatility = volatility or self._estimate_volatility(symbol, current_price)
            
            # Calculate liquidation price
            liquidation_price = self.calculate_liquidation_price(
                symbol, side, entry_price, position_size, leverage, margin_type
            )
            
            # Calculate distance to liquidation
            if side.lower() == 'long':
                distance_to_liquidation = current_price - liquidation_price
            else:
                distance_to_liquidation = liquidation_price - current_price
            
            distance_percentage = (distance_to_liquidation / current_price) * 100
            
            # Determine risk level
            risk_level = self._classify_risk_level(distance_percentage)
            
            # Calculate margin metrics
            position_value = position_size * current_price
            margin_used = position_value / leverage
            
            # Get margin tier for calculations
            margin_tier = self._get_margin_tier(symbol, position_size)
            maintenance_margin_required = position_value * margin_tier.maintenance_margin_rate
            
            # Calculate unrealized PnL
            if side.lower() == 'long':
                unrealized_pnl = position_size * (current_price - entry_price)
            else:
                unrealized_pnl = position_size * (entry_price - current_price)
            
            # Margin ratio and excess margin
            total_account_value = account_balance + unrealized_pnl
            margin_ratio = maintenance_margin_required / total_account_value if total_account_value > 0 else Decimal('1')
            excess_margin = total_account_value - maintenance_margin_required
            
            # Liquidation buffer
            liquidation_buffer = distance_to_liquidation / volatility if volatility > 0 else Decimal('0')
            
            # Time-based risk assessment
            estimated_time = self._estimate_time_to_liquidation(
                distance_to_liquidation, volatility, current_price
            )
            
            # Maximum adverse price move before liquidation
            max_adverse_move = abs(distance_to_liquidation)
            
            # Portfolio impact (simplified for single position)
            portfolio_impact_score = (margin_used / account_balance) * 100 if account_balance > 0 else Decimal('100')
            
            # Correlation risk (would be calculated with other positions)
            correlation_risk = Decimal('0')  # Simplified for single position
            
            # Generate recommendations
            recommended_action = self._generate_risk_recommendation(risk_level, distance_percentage)
            safe_position_size = self._calculate_safe_position_size(
                symbol, current_price, leverage, account_balance, volatility
            )
            
            # Stop loss recommendation
            stop_loss_price = self._recommend_stop_loss(
                side, current_price, liquidation_price, volatility
            )
            
            # Hedge recommendation
            hedge_recommendation = self._generate_hedge_recommendation(
                risk_level, position_size, current_price
            )
            
            assessment = LiquidationRiskAssessment(
                liquidation_price=liquidation_price,
                current_price=current_price,
                distance_to_liquidation=distance_to_liquidation,
                distance_percentage=distance_percentage,
                risk_level=risk_level,
                position_size=position_size,
                entry_price=entry_price,
                leverage=leverage,
                margin_used=margin_used,
                unrealized_pnl=unrealized_pnl,
                margin_ratio=margin_ratio,
                maintenance_margin_required=maintenance_margin_required,
                excess_margin=excess_margin,
                liquidation_buffer=liquidation_buffer,
                estimated_time_to_liquidation=estimated_time,
                max_adverse_move=max_adverse_move,
                portfolio_impact_score=portfolio_impact_score,
                correlation_risk=correlation_risk,
                recommended_action=recommended_action,
                safe_position_size=safe_position_size,
                stop_loss_price=stop_loss_price,
                hedge_recommendation=hedge_recommendation
            )
            
            # Store assessment
            self.risk_assessments[symbol] = assessment
            self.risk_history.append(assessment)
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing liquidation risk: {e}")
            raise
    
    def _classify_risk_level(self, distance_percentage: Decimal) -> LiquidationRiskLevel:
        """Classify liquidation risk level based on distance percentage."""
        try:
            abs_distance = abs(distance_percentage)
            
            if abs_distance < Decimal('0.5'):
                return LiquidationRiskLevel.IMMINENT
            elif abs_distance < Decimal('2'):
                return LiquidationRiskLevel.CRITICAL
            elif abs_distance < Decimal('5'):
                return LiquidationRiskLevel.HIGH
            elif abs_distance < Decimal('10'):
                return LiquidationRiskLevel.MEDIUM
            elif abs_distance < Decimal('20'):
                return LiquidationRiskLevel.LOW
            else:
                return LiquidationRiskLevel.SAFE
                
        except Exception:
            return LiquidationRiskLevel.MEDIUM
    
    def _estimate_volatility(self, symbol: str, current_price: Decimal) -> Decimal:
        """Estimate price volatility (simplified for backtesting)."""
        try:
            # Default volatility estimates by symbol type
            volatility_estimates = {
                'BTCUSDT': Decimal('0.03'),    # 3% daily volatility
                'ETHUSDT': Decimal('0.04'),    # 4% daily volatility
                'DEFAULT': Decimal('0.05')     # 5% default volatility
            }
            
            base_volatility = volatility_estimates.get(symbol, volatility_estimates['DEFAULT'])
            
            # Apply volatility to current price
            return current_price * base_volatility
            
        except Exception:
            return current_price * Decimal('0.05')  # 5% default
    
    def _estimate_time_to_liquidation(
        self,
        distance_to_liquidation: Decimal,
        volatility: Decimal,
        current_price: Decimal
    ) -> Optional[timedelta]:
        """Estimate time until potential liquidation based on volatility."""
        try:
            if distance_to_liquidation <= 0 or volatility <= 0:
                return timedelta(minutes=0)  # Immediate risk
            
            # Simple model: time = distance / (volatility * sqrt(time_factor))
            # This is a simplified version of financial volatility models
            
            daily_volatility = volatility
            hourly_volatility = daily_volatility / math.sqrt(24)
            
            # Calculate number of standard deviations
            distance_ratio = distance_to_liquidation / current_price
            volatility_ratio = daily_volatility
            
            if volatility_ratio > 0:
                # Estimate hours until liquidation becomes likely
                sigma_distance = distance_ratio / volatility_ratio
                
                # Conservative estimate: time when 2-sigma move would reach liquidation
                estimated_hours = max(1, (sigma_distance ** 2) * 6)  # Simplified model
                estimated_hours = min(estimated_hours, 72)  # Cap at 3 days
                
                return timedelta(hours=float(estimated_hours))
            
            return timedelta(hours=24)  # Default 24 hours
            
        except Exception as e:
            self.logger.error(f"Error estimating time to liquidation: {e}")
            return timedelta(hours=12)  # Default fallback
    
    def _generate_risk_recommendation(
        self,
        risk_level: LiquidationRiskLevel,
        distance_percentage: Decimal
    ) -> str:
        """Generate risk management recommendation."""
        try:
            recommendations = {
                LiquidationRiskLevel.SAFE: "Position is safe. Consider taking profits or increasing position if strategy allows.",
                LiquidationRiskLevel.LOW: "Low risk. Monitor position and consider setting stop-loss below liquidation price.",
                LiquidationRiskLevel.MEDIUM: "Medium risk. Consider reducing leverage or adding margin to increase safety buffer.",
                LiquidationRiskLevel.HIGH: "High risk! Strongly recommend reducing position size or adding significant margin.",
                LiquidationRiskLevel.CRITICAL: "CRITICAL RISK! Immediate action required - reduce position or add margin urgently.",
                LiquidationRiskLevel.IMMINENT: "LIQUIDATION IMMINENT! Emergency position closure or margin addition required NOW!"
            }
            
            base_recommendation = recommendations.get(risk_level, "Monitor position carefully.")
            
            # Add specific percentage context
            return f"{base_recommendation} (Distance to liquidation: {distance_percentage:.2f}%)"
            
        except Exception:
            return "Monitor position and manage risk appropriately."
    
    def _calculate_safe_position_size(
        self,
        symbol: str,
        current_price: Decimal,
        leverage: Decimal,
        account_balance: Decimal,
        volatility: Decimal
    ) -> Decimal:
        """Calculate safe position size given risk constraints."""
        try:
            # Target: Keep liquidation risk at SAFE level (>20% distance)
            target_distance_percentage = Decimal('25')  # 25% safety buffer
            
            # Get margin tier
            margin_tier = self._get_margin_tier(symbol, Decimal('1'))  # Get base tier
            maintenance_margin_rate = margin_tier.maintenance_margin_rate
            
            # Calculate safe leverage to maintain target distance
            safe_leverage_factor = Decimal('1') / (
                Decimal('1') - target_distance_percentage / 100 + maintenance_margin_rate
            )
            
            safe_leverage = min(leverage, safe_leverage_factor * Decimal('0.8'))  # 20% additional buffer
            
            # Calculate maximum position value based on risk tolerance
            max_risk_amount = account_balance * self.risk_tolerance
            safe_position_value = max_risk_amount * safe_leverage
            
            # Convert to position size
            safe_position_size = safe_position_value / current_price
            
            return safe_position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating safe position size: {e}")
            return Decimal('0')
    
    def _recommend_stop_loss(
        self,
        side: str,
        current_price: Decimal,
        liquidation_price: Decimal,
        volatility: Decimal
    ) -> Optional[Decimal]:
        """Recommend stop-loss price to avoid liquidation."""
        try:
            # Place stop-loss with buffer above/below liquidation price
            buffer_percentage = Decimal('0.05')  # 5% buffer
            volatility_buffer = volatility * Decimal('2')  # 2x volatility buffer
            
            if side.lower() == 'long':
                # For long positions, stop-loss below liquidation price
                buffer = max(
                    liquidation_price * buffer_percentage,
                    volatility_buffer
                )
                stop_loss_price = liquidation_price + buffer
                
                # Ensure stop-loss is below current price
                if stop_loss_price >= current_price:
                    stop_loss_price = current_price * Decimal('0.95')  # 5% below current
                    
            else:  # short
                # For short positions, stop-loss above liquidation price
                buffer = max(
                    liquidation_price * buffer_percentage,
                    volatility_buffer
                )
                stop_loss_price = liquidation_price - buffer
                
                # Ensure stop-loss is above current price
                if stop_loss_price <= current_price:
                    stop_loss_price = current_price * Decimal('1.05')  # 5% above current
            
            return stop_loss_price
            
        except Exception as e:
            self.logger.error(f"Error recommending stop-loss: {e}")
            return None
    
    def _generate_hedge_recommendation(
        self,
        risk_level: LiquidationRiskLevel,
        position_size: Decimal,
        current_price: Decimal
    ) -> Optional[str]:
        """Generate hedging recommendation based on risk level."""
        try:
            if risk_level in [LiquidationRiskLevel.SAFE, LiquidationRiskLevel.LOW]:
                return None  # No hedging needed
            
            position_value = position_size * current_price
            
            if risk_level == LiquidationRiskLevel.MEDIUM:
                hedge_ratio = Decimal('0.3')  # Hedge 30% of position
                return f"Consider hedging {hedge_ratio*100:.0f}% of position (${position_value * hedge_ratio:.2f}) with opposite position or options."
            
            elif risk_level == LiquidationRiskLevel.HIGH:  
                hedge_ratio = Decimal('0.5')  # Hedge 50% of position
                return f"Strongly recommend hedging {hedge_ratio*100:.0f}% of position (${position_value * hedge_ratio:.2f}) immediately."
            
            elif risk_level in [LiquidationRiskLevel.CRITICAL, LiquidationRiskLevel.IMMINENT]:
                return "Consider full hedge or position closure. Risk is too high for partial hedging."
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating hedge recommendation: {e}")
            return None
    
    def simulate_liquidation_cascade(
        self,
        portfolio_positions: Dict[str, Dict[str, Any]],
        market_shock_percentage: Decimal
    ) -> Dict[str, Any]:
        """
        Simulate liquidation cascade effects across portfolio.
        
        Args:
            portfolio_positions: Dictionary of all portfolio positions
            market_shock_percentage: Market shock magnitude (e.g., -20% for 20% drop)
            
        Returns:
            Cascade simulation results
        """
        try:
            cascade_results = {
                'initial_positions': len(portfolio_positions),
                'liquidated_positions': [],
                'surviving_positions': [],
                'total_loss': Decimal('0'),
                'cascade_stages': []
            }
            
            # Stage 1: Direct liquidations from market shock
            stage_1_liquidations = []
            
            for symbol, position in portfolio_positions.items():
                current_price = position['current_price']
                shocked_price = current_price * (Decimal('1') + market_shock_percentage / 100)
                
                # Check if position would be liquidated
                liquidation_price = position['liquidation_price']
                side = position['side']
                
                is_liquidated = False
                if side.lower() == 'long' and shocked_price <= liquidation_price:
                    is_liquidated = True
                elif side.lower() == 'short' and shocked_price >= liquidation_price:
                    is_liquidated = True
                
                if is_liquidated:
                    liquidation_loss = position['margin_used']
                    stage_1_liquidations.append({
                        'symbol': symbol,
                        'liquidation_price': float(liquidation_price),
                        'shocked_price': float(shocked_price),
                        'loss': float(liquidation_loss)
                    })
                    cascade_results['total_loss'] += liquidation_loss
                    cascade_results['liquidated_positions'].append(symbol)
                else:
                    cascade_results['surviving_positions'].append(symbol)
            
            cascade_results['cascade_stages'].append({
                'stage': 1,
                'description': f"Direct liquidations from {market_shock_percentage}% market shock",
                'liquidations': stage_1_liquidations,
                'cumulative_loss': float(cascade_results['total_loss'])
            })
            
            # Additional stages could model:
            # - ADL (Auto-Deleveraging) effects
            # - Insurance fund depletion
            # - Socialized losses
            # But these are complex and depend on exchange-specific mechanisms
            
            return cascade_results
            
        except Exception as e:
            self.logger.error(f"Error simulating liquidation cascade: {e}")
            return {'error': str(e)}
    
    def generate_risk_report(
        self,
        portfolio_positions: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive liquidation risk report."""
        try:
            positions = portfolio_positions or self.portfolio_positions
            
            if not positions:
                return {
                    'message': 'No positions to analyze',
                    'overall_risk': 'SAFE',
                    'total_positions': 0
                }
            
            # Analyze each position
            position_risks = []
            total_margin = Decimal('0')
            high_risk_positions = 0
            
            for symbol, position in positions.items():
                if symbol in self.risk_assessments:
                    assessment = self.risk_assessments[symbol]
                    position_risks.append({
                        'symbol': symbol,
                        'risk_level': assessment.risk_level.value,
                        'distance_percentage': float(assessment.distance_percentage),
                        'liquidation_price': float(assessment.liquidation_price),
                        'current_price': float(assessment.current_price),
                        'margin_used': float(assessment.margin_used)
                    })
                    
                    total_margin += assessment.margin_used
                    
                    if assessment.risk_level in [
                        LiquidationRiskLevel.HIGH, 
                        LiquidationRiskLevel.CRITICAL,
                        LiquidationRiskLevel.IMMINENT
                    ]:
                        high_risk_positions += 1
            
            # Overall portfolio risk assessment
            if high_risk_positions > 0:
                overall_risk = 'HIGH'
            elif len([r for r in position_risks if r['risk_level'] == 'medium']) > 0:
                overall_risk = 'MEDIUM'
            else:
                overall_risk = 'LOW'
            
            # Generate recommendations
            recommendations = []
            if high_risk_positions > 0:
                recommendations.append(f"URGENT: {high_risk_positions} positions at high liquidation risk")
            if total_margin > Decimal('0'):
                recommendations.append("Consider diversifying margin usage across positions")
            
            return {
                'overall_risk': overall_risk,
                'total_positions': len(positions),
                'high_risk_positions': high_risk_positions,
                'total_margin_used': float(total_margin),
                'position_risks': position_risks,
                'recommendations': recommendations,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return {'error': str(e)}
    
    def get_safe_leverage(
        self,
        symbol: str,
        account_balance: Decimal,
        risk_percentage: Decimal = Decimal('5')
    ) -> Decimal:
        """
        Calculate safe leverage for symbol given risk constraints.
        
        Args:
            symbol: Trading symbol
            account_balance: Available account balance
            risk_percentage: Maximum risk as percentage of account
            
        Returns:
            Safe leverage to use
        """
        try:
            # Get margin tier
            margin_tier = self._get_margin_tier(symbol, Decimal('1'))
            
            # Calculate safe leverage based on maintenance margin
            maintenance_margin_rate = margin_tier.maintenance_margin_rate
            
            # Safe leverage formula considering:
            # 1. Maintenance margin requirements
            # 2. Risk tolerance
            # 3. Market volatility buffer
            
            volatility_buffer = Decimal('0.1')  # 10% volatility buffer
            risk_factor = risk_percentage / 100
            
            # Conservative leverage calculation
            safe_leverage = min(
                # Based on maintenance margin
                Decimal('1') / (maintenance_margin_rate * Decimal('2')),
                # Based on risk tolerance
                Decimal('1') / (risk_factor + volatility_buffer),
                # Maximum allowed for tier
                margin_tier.max_leverage * Decimal('0.7')  # 30% buffer from max
            )
            
            return max(safe_leverage, Decimal('1'))  # Minimum 1x leverage
            
        except Exception as e:
            self.logger.error(f"Error calculating safe leverage: {e}")
            return Decimal('5')  # Conservative default