"""
Bybit Fee Simulation and Calculation System

This module provides comprehensive fee simulation for Bybit trading including:

Core Features:
- VIP tier-based fee calculation (No VIP through PRO 3)
- Maker/taker fee distinctions with rebates for high-tier VIPs
- Contract-specific fee structures (spot, linear perpetual, inverse)
- Dynamic fee calculation based on trading volume and VIP status
- Fee optimization analysis and cost reduction strategies
- Real-time fee rate updates and tier progression simulation

Advanced Features:
- Volume-based VIP tier progression modeling
- Fee rebate calculation for liquidity providers
- Cross-margin and isolated margin fee differences
- Funding rate impact on overall trading costs
- Fee comparison across different contract types
- Historical fee analysis and cost optimization

This system enables accurate cost modeling for backtesting and
provides insights for fee optimization in live trading.

Author: Trading Bot Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .bybit_enhanced_backtest_engine import BybitVIPTier, BybitContractType
from ..utils.logging import TradingLogger


@dataclass
class VIPTierRequirements:
    """Bybit VIP tier requirements and benefits."""
    
    # VIP tier requirements (30-day trading volume in USDT)
    TIER_REQUIREMENTS = {
        BybitVIPTier.NO_VIP: {
            "volume_requirement": Decimal("0"),
            "asset_requirement": Decimal("0"),
            "description": "No VIP status"
        },
        BybitVIPTier.VIP1: {
            "volume_requirement": Decimal("500000"),      # 500K USDT
            "asset_requirement": Decimal("1000"),         # 1K USDT equivalent
            "description": "VIP 1 - Entry level benefits"
        },
        BybitVIPTier.VIP2: {
            "volume_requirement": Decimal("2500000"),     # 2.5M USDT
            "asset_requirement": Decimal("5000"),         # 5K USDT equivalent
            "description": "VIP 2 - Enhanced benefits"
        },
        BybitVIPTier.VIP3: {
            "volume_requirement": Decimal("12500000"),    # 12.5M USDT
            "asset_requirement": Decimal("25000"),        # 25K USDT equivalent
            "description": "VIP 3 - Premium benefits"
        },
        BybitVIPTier.PRO1: {
            "volume_requirement": Decimal("75000000"),    # 75M USDT
            "asset_requirement": Decimal("200000"),       # 200K USDT equivalent
            "description": "PRO 1 - Professional tier"
        },
        BybitVIPTier.PRO2: {
            "volume_requirement": Decimal("200000000"),   # 200M USDT
            "asset_requirement": Decimal("500000"),       # 500K USDT equivalent
            "description": "PRO 2 - Advanced professional"
        },
        BybitVIPTier.PRO3: {
            "volume_requirement": Decimal("500000000"),   # 500M USDT
            "asset_requirement": Decimal("1000000"),      # 1M USDT equivalent
            "description": "PRO 3 - Institutional level"
        }
    }


@dataclass
class FeeCalculationDetails:
    """Detailed fee calculation breakdown."""
    
    base_fee_rate: Decimal
    volume_discount: Decimal
    vip_discount: Decimal
    final_fee_rate: Decimal
    notional_value: Decimal
    fee_amount: Decimal
    is_maker: bool
    is_rebate: bool
    contract_type: BybitContractType
    vip_tier: BybitVIPTier
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'base_fee_rate': float(self.base_fee_rate),
            'volume_discount': float(self.volume_discount),
            'vip_discount': float(self.vip_discount), 
            'final_fee_rate': float(self.final_fee_rate),
            'notional_value': float(self.notional_value),
            'fee_amount': float(self.fee_amount),
            'is_maker': self.is_maker,
            'is_rebate': self.is_rebate,
            'contract_type': self.contract_type.value,
            'vip_tier': self.vip_tier.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class FeeAnalysisReport:
    """Comprehensive fee analysis and optimization report."""
    
    total_fees_paid: Decimal
    total_rebates_received: Decimal
    net_fees: Decimal
    maker_percentage: Decimal
    taker_percentage: Decimal
    average_fee_rate: Decimal
    fee_as_percentage_of_pnl: Decimal
    
    # VIP tier analysis
    current_vip_tier: BybitVIPTier
    potential_savings_next_tier: Decimal
    volume_needed_for_next_tier: Decimal
    
    # Optimization suggestions
    optimization_suggestions: List[str] = field(default_factory=list)
    estimated_annual_savings: Decimal = Decimal('0')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'total_fees_paid': float(self.total_fees_paid),
            'total_rebates_received': float(self.total_rebates_received),
            'net_fees': float(self.net_fees),
            'maker_percentage': float(self.maker_percentage),
            'taker_percentage': float(self.taker_percentage),
            'average_fee_rate': float(self.average_fee_rate),
            'fee_as_percentage_of_pnl': float(self.fee_as_percentage_of_pnl),
            'current_vip_tier': self.current_vip_tier.value,
            'potential_savings_next_tier': float(self.potential_savings_next_tier),
            'volume_needed_for_next_tier': float(self.volume_needed_for_next_tier),
            'optimization_suggestions': self.optimization_suggestions,
            'estimated_annual_savings': float(self.estimated_annual_savings)
        }


class BybitFeeCalculator:
    """
    Advanced Bybit fee calculation and optimization system.
    
    This class provides:
    1. Accurate fee calculation for all Bybit contract types
    2. VIP tier progression simulation and benefits analysis
    3. Maker/taker optimization strategies
    4. Cost analysis and savings identification
    5. Real-time fee optimization recommendations
    """
    
    def __init__(
        self,
        initial_vip_tier: BybitVIPTier = BybitVIPTier.NO_VIP,
        track_volume_for_vip: bool = True
    ):
        self.current_vip_tier = initial_vip_tier
        self.track_volume_for_vip = track_volume_for_vip
        
        # Volume tracking for VIP tier progression
        self.daily_volumes: Dict[str, Decimal] = {}  # date -> volume
        self.rolling_30day_volume = Decimal('0')
        
        # Fee tracking
        self.fee_history: List[FeeCalculationDetails] = []
        self.total_fees_paid = Decimal('0')
        self.total_rebates_received = Decimal('0')
        
        # Contract-specific fee structures
        self._initialize_fee_structures()
        
        self.logger = TradingLogger("BybitFeeCalculator")
        self.logger.info(f"Initialized with VIP tier: {initial_vip_tier.value}")
    
    def _initialize_fee_structures(self) -> None:
        """Initialize all fee structures for different contract types."""
        
        # Linear Perpetual (USDT-margined) fees
        self.linear_perp_fees = {
            BybitVIPTier.NO_VIP: {"maker": Decimal("0.0001"), "taker": Decimal("0.0006")},
            BybitVIPTier.VIP1: {"maker": Decimal("0.0001"), "taker": Decimal("0.0005")},
            BybitVIPTier.VIP2: {"maker": Decimal("0.0000"), "taker": Decimal("0.0004")},
            BybitVIPTier.VIP3: {"maker": Decimal("0.0000"), "taker": Decimal("0.0003")},
            BybitVIPTier.PRO1: {"maker": Decimal("-0.0001"), "taker": Decimal("0.0002")},
            BybitVIPTier.PRO2: {"maker": Decimal("-0.0002"), "taker": Decimal("0.0001")},
            BybitVIPTier.PRO3: {"maker": Decimal("-0.0002"), "taker": Decimal("0.0001")},
        }
        
        # Inverse Perpetual (coin-margined) fees
        self.inverse_perp_fees = {
            BybitVIPTier.NO_VIP: {"maker": Decimal("0.0001"), "taker": Decimal("0.0006")},
            BybitVIPTier.VIP1: {"maker": Decimal("0.0001"), "taker": Decimal("0.0005")},
            BybitVIPTier.VIP2: {"maker": Decimal("0.0000"), "taker": Decimal("0.0004")},
            BybitVIPTier.VIP3: {"maker": Decimal("0.0000"), "taker": Decimal("0.0003")},
            BybitVIPTier.PRO1: {"maker": Decimal("-0.0001"), "taker": Decimal("0.0002")},  
            BybitVIPTier.PRO2: {"maker": Decimal("-0.0002"), "taker": Decimal("0.0001")},
            BybitVIPTier.PRO3: {"maker": Decimal("-0.0002"), "taker": Decimal("0.0001")},
        }
        
        # Spot trading fees
        self.spot_fees = {
            BybitVIPTier.NO_VIP: {"maker": Decimal("0.001"), "taker": Decimal("0.001")},
            BybitVIPTier.VIP1: {"maker": Decimal("0.0009"), "taker": Decimal("0.0009")},
            BybitVIPTier.VIP2: {"maker": Decimal("0.0008"), "taker": Decimal("0.0008")},
            BybitVIPTier.VIP3: {"maker": Decimal("0.0007"), "taker": Decimal("0.0007")},
            BybitVIPTier.PRO1: {"maker": Decimal("0.0006"), "taker": Decimal("0.0006")},
            BybitVIPTier.PRO2: {"maker": Decimal("0.0005"), "taker": Decimal("0.0005")},
            BybitVIPTier.PRO3: {"maker": Decimal("0.0004"), "taker": Decimal("0.0004")},
        }
        
        # Options fees (if supported in future)
        self.options_fees = {
            BybitVIPTier.NO_VIP: {"maker": Decimal("0.0003"), "taker": Decimal("0.0003")},
            # ... other tiers similar structure
        }
    
    def calculate_fee(
        self,
        quantity: Decimal,
        price: Decimal,
        is_maker: bool,
        contract_type: BybitContractType = BybitContractType.LINEAR_PERPETUAL,
        symbol: str = "BTCUSDT",
        timestamp: Optional[datetime] = None
    ) -> FeeCalculationDetails:
        """
        Calculate fee for a trade with detailed breakdown.
        
        Args:
            quantity: Trade quantity
            price: Trade price
            is_maker: True if maker order, False if taker
            contract_type: Type of contract being traded
            symbol: Trading symbol
            timestamp: Trade timestamp
            
        Returns:
            FeeCalculationDetails with complete breakdown
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Calculate notional value
            notional_value = quantity * price
            
            # Get base fee rate
            base_fee_rate = self._get_base_fee_rate(contract_type, is_maker)
            
            # Apply VIP discounts
            vip_discount = self._calculate_vip_discount(contract_type, is_maker)
            final_fee_rate = base_fee_rate - vip_discount
            
            # Volume discount (additional savings for high volume)
            volume_discount = self._calculate_volume_discount(notional_value)
            final_fee_rate = final_fee_rate - volume_discount
            
            # Calculate fee amount
            fee_amount = notional_value * final_fee_rate
            is_rebate = fee_amount < 0
            
            # Track volume for VIP progression
            if self.track_volume_for_vip:
                self._track_trading_volume(notional_value, timestamp)
            
            # Create detailed calculation
            details = FeeCalculationDetails(
                base_fee_rate=base_fee_rate,
                volume_discount=volume_discount,
                vip_discount=vip_discount,
                final_fee_rate=final_fee_rate,
                notional_value=notional_value,
                fee_amount=fee_amount,
                is_maker=is_maker,
                is_rebate=is_rebate,
                contract_type=contract_type,
                vip_tier=self.current_vip_tier,
                timestamp=timestamp
            )
            
            # Update tracking
            self.fee_history.append(details)
            if is_rebate:
                self.total_rebates_received += abs(fee_amount)
            else:
                self.total_fees_paid += fee_amount
            
            self.logger.debug(
                f"Fee calculated: {symbol} ${notional_value:.2f} -> "
                f"${fee_amount:.4f} ({final_fee_rate*100:.4f}%) "
                f"{'Maker' if is_maker else 'Taker'} {self.current_vip_tier.value}"
            )
            
            return details
            
        except Exception as e:
            self.logger.error(f"Error calculating fee: {e}")
            raise
    
    def _get_base_fee_rate(
        self,
        contract_type: BybitContractType,
        is_maker: bool
    ) -> Decimal:
        """Get base fee rate for contract type."""
        try:
            if contract_type == BybitContractType.LINEAR_PERPETUAL:
                fee_structure = self.linear_perp_fees
            elif contract_type == BybitContractType.INVERSE_PERPETUAL:
                fee_structure = self.inverse_perp_fees
            elif contract_type == BybitContractType.SPOT:
                fee_structure = self.spot_fees
            else:
                # Default to linear perpetual
                fee_structure = self.linear_perp_fees
            
            return fee_structure[self.current_vip_tier]['maker' if is_maker else 'taker']
            
        except Exception as e:
            self.logger.error(f"Error getting base fee rate: {e}")
            return Decimal('0.001')  # Default fallback
    
    def _calculate_vip_discount(
        self,
        contract_type: BybitContractType,
        is_maker: bool
    ) -> Decimal:
        """Calculate VIP tier discount."""
        try:
            # VIP discount is already included in the fee structure
            # This method could be used for additional discounts or promotions
            return Decimal('0')
            
        except Exception:
            return Decimal('0')
    
    def _calculate_volume_discount(self, notional_value: Decimal) -> Decimal:
        """Calculate volume-based discount."""
        try:
            # Additional volume discounts could be implemented here
            # For now, VIP tiers already include volume-based benefits
            return Decimal('0')
            
        except Exception:
            return Decimal('0')
    
    def _track_trading_volume(self, volume: Decimal, timestamp: datetime) -> None:
        """Track trading volume for VIP tier progression."""
        try:
            date_key = timestamp.strftime('%Y-%m-%d')
            
            # Add to daily volume
            if date_key not in self.daily_volumes:
                self.daily_volumes[date_key] = Decimal('0')
            self.daily_volumes[date_key] += volume
            
            # Update 30-day rolling volume
            self._update_rolling_volume(timestamp)
            
            # Check for VIP tier progression
            self._check_vip_tier_progression()
            
        except Exception as e:
            self.logger.error(f"Error tracking volume: {e}")
    
    def _update_rolling_volume(self, current_date: datetime) -> None:
        """Update 30-day rolling volume."""
        try:
            cutoff_date = current_date - timedelta(days=30)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d')
            
            # Remove old entries
            dates_to_remove = [
                date for date in self.daily_volumes.keys()
                if date < cutoff_str
            ]
            
            for date in dates_to_remove:
                del self.daily_volumes[date]
            
            # Calculate current rolling volume
            self.rolling_30day_volume = sum(self.daily_volumes.values())
            
        except Exception as e:
            self.logger.error(f"Error updating rolling volume: {e}")
    
    def _check_vip_tier_progression(self) -> None:
        """Check if current volume qualifies for higher VIP tier."""
        try:
            current_tier_level = list(BybitVIPTier).index(self.current_vip_tier)
            
            # Check each higher tier
            for tier in list(BybitVIPTier)[current_tier_level + 1:]:
                requirements = VIPTierRequirements.TIER_REQUIREMENTS[tier]
                
                if self.rolling_30day_volume >= requirements["volume_requirement"]:
                    old_tier = self.current_vip_tier
                    self.current_vip_tier = tier
                    
                    self.logger.info(
                        f"VIP tier upgraded: {old_tier.value} -> {tier.value} "
                        f"(Volume: ${self.rolling_30day_volume:,.2f})"
                    )
                    break
            
        except Exception as e:
            self.logger.error(f"Error checking VIP progression: {e}")
    
    def get_current_vip_status(self) -> Dict[str, Any]:
        """Get current VIP status and progression information."""
        try:
            current_requirements = VIPTierRequirements.TIER_REQUIREMENTS[self.current_vip_tier]
            
            # Find next tier
            current_tier_level = list(BybitVIPTier).index(self.current_vip_tier)
            next_tier = None
            next_requirements = None
            volume_needed = Decimal('0')
            
            if current_tier_level < len(BybitVIPTier) - 1:
                next_tier = list(BybitVIPTier)[current_tier_level + 1]
                next_requirements = VIPTierRequirements.TIER_REQUIREMENTS[next_tier]
                volume_needed = max(
                    Decimal('0'),
                    next_requirements["volume_requirement"] - self.rolling_30day_volume
                )
            
            return {
                'current_tier': self.current_vip_tier.value,
                'current_30day_volume': float(self.rolling_30day_volume),
                'current_tier_requirement': float(current_requirements["volume_requirement"]),
                'next_tier': next_tier.value if next_tier else None,
                'next_tier_requirement': float(next_requirements["volume_requirement"]) if next_requirements else None,
                'volume_needed_for_next_tier': float(volume_needed),
                'progress_to_next_tier': float(
                    (self.rolling_30day_volume / next_requirements["volume_requirement"]) * 100
                ) if next_requirements and next_requirements["volume_requirement"] > 0 else 100.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting VIP status: {e}")
            return {'error': str(e)}
    
    def simulate_fee_savings(
        self,
        target_tier: BybitVIPTier,
        projected_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Simulate fee savings from upgrading to target VIP tier.
        
        Args:
            target_tier: Target VIP tier to simulate
            projected_trades: List of projected trades with keys:
                - quantity, price, is_maker, contract_type
                
        Returns:
            Dictionary with savings analysis
        """
        try:
            current_fees = Decimal('0')
            target_fees = Decimal('0')
            
            # Save current tier
            original_tier = self.current_vip_tier
            
            for trade_data in projected_trades:
                quantity = Decimal(str(trade_data['quantity']))
                price = Decimal(str(trade_data['price']))
                is_maker = trade_data['is_maker']
                contract_type = trade_data.get('contract_type', BybitContractType.LINEAR_PERPETUAL)
                
                # Calculate fee with current tier
                self.current_vip_tier = original_tier
                current_fee = self.calculate_fee(
                    quantity, price, is_maker, contract_type
                ).fee_amount
                current_fees += max(current_fee, Decimal('0'))  # Only count positive fees
                
                # Calculate fee with target tier
                self.current_vip_tier = target_tier
                target_fee = self.calculate_fee(
                    quantity, price, is_maker, contract_type
                ).fee_amount
                target_fees += max(target_fee, Decimal('0'))  # Only count positive fees
            
            # Restore original tier
            self.current_vip_tier = original_tier
            
            # Calculate savings
            total_savings = current_fees - target_fees
            savings_percentage = (total_savings / current_fees * 100) if current_fees > 0 else Decimal('0')
            
            # Get volume requirement for target tier
            target_requirements = VIPTierRequirements.TIER_REQUIREMENTS[target_tier]
            volume_needed = max(
                Decimal('0'),
                target_requirements["volume_requirement"] - self.rolling_30day_volume
            )
            
            return {
                'current_tier': original_tier.value,
                'target_tier': target_tier.value,
                'current_total_fees': float(current_fees),
                'target_total_fees': float(target_fees),
                'total_savings': float(total_savings),
                'savings_percentage': float(savings_percentage),
                'volume_needed_for_target': float(volume_needed),
                'estimated_trades_analyzed': len(projected_trades),
                'worth_upgrading': total_savings > Decimal('100')  # Worth it if saving >$100
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating fee savings: {e}")
            return {'error': str(e)}
    
    def generate_fee_analysis_report(
        self,
        period_days: int = 30,
        total_pnl: Optional[Decimal] = None
    ) -> FeeAnalysisReport:
        """Generate comprehensive fee analysis report."""
        try:
            # Filter recent fee history
            cutoff_date = datetime.now() - timedelta(days=period_days)
            recent_fees = [
                fee for fee in self.fee_history 
                if fee.timestamp >= cutoff_date
            ]
            
            if not recent_fees:
                self.logger.warning("No recent fee data available for analysis")
                return FeeAnalysisReport(
                    total_fees_paid=Decimal('0'),
                    total_rebates_received=Decimal('0'),
                    net_fees=Decimal('0'),
                    maker_percentage=Decimal('0'),
                    taker_percentage=Decimal('0'),
                    average_fee_rate=Decimal('0'),
                    fee_as_percentage_of_pnl=Decimal('0'),
                    current_vip_tier=self.current_vip_tier,
                    potential_savings_next_tier=Decimal('0'),
                    volume_needed_for_next_tier=Decimal('0')
                )
            
            # Calculate basic metrics
            total_fees = sum(max(fee.fee_amount, Decimal('0')) for fee in recent_fees)
            total_rebates = sum(abs(fee.fee_amount) for fee in recent_fees if fee.fee_amount < 0)
            net_fees = total_fees - total_rebates
            
            # Maker/taker analysis
            maker_trades = [fee for fee in recent_fees if fee.is_maker]
            taker_trades = [fee for fee in recent_fees if not fee.is_maker]
            
            maker_percentage = (len(maker_trades) / len(recent_fees)) * 100 if recent_fees else Decimal('0')
            taker_percentage = (len(taker_trades) / len(recent_fees)) * 100 if recent_fees else Decimal('0')
            
            # Average fee rate
            total_volume = sum(fee.notional_value for fee in recent_fees)
            average_fee_rate = (net_fees / total_volume) if total_volume > 0 else Decimal('0')
            
            # Fee as percentage of PnL
            fee_pnl_percentage = Decimal('0')
            if total_pnl and total_pnl != 0:
                fee_pnl_percentage = (net_fees / abs(total_pnl)) * 100
            
            # Next tier analysis
            vip_status = self.get_current_vip_status()
            potential_savings = self._calculate_next_tier_savings(recent_fees)
            
            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(
                recent_fees, maker_percentage, average_fee_rate
            )
            
            # Estimate annual savings potential
            annual_savings = self._estimate_annual_savings(recent_fees, period_days)
            
            return FeeAnalysisReport(
                total_fees_paid=total_fees,
                total_rebates_received=total_rebates,
                net_fees=net_fees,
                maker_percentage=Decimal(str(maker_percentage)),
                taker_percentage=Decimal(str(taker_percentage)),
                average_fee_rate=average_fee_rate,
                fee_as_percentage_of_pnl=fee_pnl_percentage,
                current_vip_tier=self.current_vip_tier,
                potential_savings_next_tier=potential_savings,
                volume_needed_for_next_tier=Decimal(str(vip_status.get('volume_needed_for_next_tier', 0))),
                optimization_suggestions=suggestions,
                estimated_annual_savings=annual_savings
            )
            
        except Exception as e:
            self.logger.error(f"Error generating fee analysis: {e}")
            raise
    
    def _calculate_next_tier_savings(self, recent_fees: List[FeeCalculationDetails]) -> Decimal:
        """Calculate potential savings from next VIP tier."""
        try:
            current_tier_level = list(BybitVIPTier).index(self.current_vip_tier)
            
            if current_tier_level >= len(BybitVIPTier) - 1:
                return Decimal('0')  # Already at highest tier
            
            next_tier = list(BybitVIPTier)[current_tier_level + 1]
            
            # Simulate fees with next tier
            total_savings = Decimal('0')
            
            for fee_detail in recent_fees:
                # Calculate what fee would be with next tier
                original_tier = self.current_vip_tier
                self.current_vip_tier = next_tier
                
                next_tier_fee_rate = self._get_base_fee_rate(
                    fee_detail.contract_type, fee_detail.is_maker
                )
                next_tier_fee = fee_detail.notional_value * next_tier_fee_rate
                
                # Restore original tier
                self.current_vip_tier = original_tier
                
                # Calculate savings (only for positive fees)
                if fee_detail.fee_amount > 0 and next_tier_fee >= 0:
                    savings = fee_detail.fee_amount - next_tier_fee
                    total_savings += max(savings, Decimal('0'))
            
            return total_savings
            
        except Exception as e:
            self.logger.error(f"Error calculating next tier savings: {e}")
            return Decimal('0')
    
    def _generate_optimization_suggestions(
        self,
        recent_fees: List[FeeCalculationDetails],
        maker_percentage: float,
        average_fee_rate: Decimal
    ) -> List[str]:
        """Generate fee optimization suggestions."""
        suggestions = []
        
        try:
            # Maker/taker ratio optimization
            if maker_percentage < 50:
                suggestions.append(
                    f"Consider using more limit orders to increase maker ratio "
                    f"(currently {maker_percentage:.1f}% maker trades)"
                )
            
            # VIP tier progression
            vip_status = self.get_current_vip_status()
            if vip_status.get('volume_needed_for_next_tier', 0) > 0:
                volume_needed = vip_status['volume_needed_for_next_tier']
                suggestions.append(
                    f"Increase trading volume by ${volume_needed:,.0f} to reach "
                    f"{vip_status.get('next_tier', 'next VIP tier')} for lower fees"
                )
            
            # High fee rate warning
            if average_fee_rate > Decimal('0.0005'):  # 0.05%
                suggestions.append(
                    f"Average fee rate of {average_fee_rate*100:.4f}% is high - "
                    f"consider optimizing order types and VIP tier"
                )
            
            # Contract type optimization
            contract_types = {}
            for fee in recent_fees:
                contract_types[fee.contract_type] = contract_types.get(fee.contract_type, 0) + 1
            
            if len(contract_types) > 1:
                suggestions.append(
                    "Review contract type usage - linear perpetuals typically have lower fees"
                )
            
            # Volume concentration
            if len(recent_fees) > 100:  # High frequency trading
                suggestions.append(
                    "High trading frequency detected - ensure optimal execution timing "
                    "to maximize maker orders"
                )
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            suggestions.append("Error generating optimization suggestions")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _estimate_annual_savings(
        self,
        recent_fees: List[FeeCalculationDetails],
        period_days: int
    ) -> Decimal:
        """Estimate potential annual savings from optimization."""
        try:
            if not recent_fees or period_days <= 0:
                return Decimal('0')
            
            # Annualize recent fees
            daily_fees = sum(fee.fee_amount for fee in recent_fees if fee.fee_amount > 0) / period_days
            annual_fees = daily_fees * 365
            
            # Estimate savings potential
            maker_ratio = len([f for f in recent_fees if f.is_maker]) / len(recent_fees)
            
            # Potential savings from increasing maker ratio to 70%
            if maker_ratio < 0.7:
                target_maker_ratio = 0.7
                maker_savings_rate = Decimal('0.0002')  # Typical maker/taker difference
                maker_savings = annual_fees * (target_maker_ratio - Decimal(str(maker_ratio))) * maker_savings_rate
            else:
                maker_savings = Decimal('0')
            
            # Potential savings from VIP tier upgrade
            tier_savings = self._calculate_next_tier_savings(recent_fees)
            annual_tier_savings = tier_savings * (365 / period_days)
            
            total_annual_savings = maker_savings + annual_tier_savings
            
            return total_annual_savings
            
        except Exception as e:
            self.logger.error(f"Error estimating annual savings: {e}")
            return Decimal('0')
    
    def export_fee_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Export fee history as pandas DataFrame."""
        try:
            filtered_fees = self.fee_history
            
            if start_date:
                filtered_fees = [f for f in filtered_fees if f.timestamp >= start_date]
            if end_date:
                filtered_fees = [f for f in filtered_fees if f.timestamp <= end_date]
            
            if not filtered_fees:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = [fee.to_dict() for fee in filtered_fees]
            df = pd.DataFrame(data)
            
            # Set timestamp as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error exporting fee history: {e}")
            return pd.DataFrame()
    
    def get_fee_summary(self, period_days: int = 30) -> Dict[str, Any]:
        """Get summary of fee performance over specified period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=period_days)
            recent_fees = [
                fee for fee in self.fee_history 
                if fee.timestamp >= cutoff_date
            ]
            
            if not recent_fees:
                return {'error': 'No fee data available for the specified period'}
            
            # Basic statistics
            total_fees = sum(max(fee.fee_amount, Decimal('0')) for fee in recent_fees)
            total_rebates = sum(abs(fee.fee_amount) for fee in recent_fees if fee.fee_amount < 0)
            total_volume = sum(fee.notional_value for fee in recent_fees)
            
            # Breakdown by contract type
            contract_breakdown = {}
            for fee in recent_fees:
                contract_type = fee.contract_type.value
                if contract_type not in contract_breakdown:
                    contract_breakdown[contract_type] = {
                        'count': 0, 'volume': Decimal('0'), 'fees': Decimal('0')
                    }
                
                contract_breakdown[contract_type]['count'] += 1
                contract_breakdown[contract_type]['volume'] += fee.notional_value
                contract_breakdown[contract_type]['fees'] += max(fee.fee_amount, Decimal('0'))
            
            # Convert Decimal to float for JSON serialization
            for contract_type in contract_breakdown:
                contract_breakdown[contract_type]['volume'] = float(contract_breakdown[contract_type]['volume'])
                contract_breakdown[contract_type]['fees'] = float(contract_breakdown[contract_type]['fees'])
            
            return {
                'period_days': period_days,
                'total_trades': len(recent_fees),
                'total_volume': float(total_volume),
                'total_fees_paid': float(total_fees),
                'total_rebates_received': float(total_rebates),
                'net_fees': float(total_fees - total_rebates),
                'average_fee_rate': float((total_fees / total_volume) if total_volume > 0 else 0),
                'maker_trades': len([f for f in recent_fees if f.is_maker]),
                'taker_trades': len([f for f in recent_fees if not f.is_maker]),
                'current_vip_tier': self.current_vip_tier.value,
                'rolling_30day_volume': float(self.rolling_30day_volume),
                'contract_breakdown': contract_breakdown,
                'vip_status': self.get_current_vip_status()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting fee summary: {e}")
            return {'error': str(e)}