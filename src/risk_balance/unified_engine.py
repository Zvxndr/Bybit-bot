"""
Unified Risk-Balance Engine
==========================

Core engine that integrates:
- Dynamic risk scaling (Speed Demon features built-in)
- Real-time balance tracking across all environments
- Market regime-aware risk adjustments
- Portfolio-level risk management

This replaces multiple scattered risk management systems with
a single, unified approach.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import math


class MarketRegime(Enum):
    """Market regime classification"""
    LOW_VOLATILITY = "low_volatility"
    NORMAL = "normal" 
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"


class RiskLevel(Enum):
    """Risk level classification"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    portfolio_risk_score: float
    position_size_limit: Decimal
    current_drawdown: float
    max_positions: int
    risk_level: RiskLevel
    regime_adjustment: float
    balance_tier: str


@dataclass
class BalanceSnapshot:
    """Multi-environment balance snapshot"""
    total_balance: Decimal
    available_balance: Decimal
    unrealized_pnl: Decimal
    margin_used: Decimal
    environments: Dict[str, Dict[str, Any]]
    timestamp: datetime


class UnifiedRiskBalanceEngine:
    """
    Unified engine for risk management and balance tracking.
    
    Features:
    - Built-in Speed Demon dynamic risk scaling
    - Real-time balance tracking across testnet/mainnet
    - Market regime-aware risk adjustments
    - Portfolio-level risk monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger("risk_balance_engine")
        self.config = config or {}
        
        # Core components
        self._balance_cache = {}
        self._risk_cache = {}
        self._regime_history = []
        
        # Speed Demon configuration (built-in)
        self.speed_demon_config = self.config.get('speed_demon', {
            'enabled': True,
            'dynamic_risk_scaling': {
                'enabled': True,
                'small_account_risk': 0.02,      # 2% for accounts < $10k
                'large_account_risk': 0.005,     # 0.5% for accounts > $100k
                'transition_start': 10000,       # $10k
                'transition_end': 100000,        # $100k
                'decay_factor': 0.5              # Exponential decay rate
            }
        })
        
        self.logger.info("[OK] Unified Risk-Balance Engine initialized")
    
    async def get_current_risk_metrics(self, balance_usd: Optional[Decimal] = None) -> RiskMetrics:
        """
        Get current risk metrics with dynamic scaling
        
        Args:
            balance_usd: Account balance in USD (if None, will fetch current)
            
        Returns:
            Complete risk metrics including Speed Demon adjustments
        """
        try:
            # Get current balance if not provided
            if balance_usd is None:
                balance_snapshot = await self.get_balance_snapshot()
                balance_usd = balance_snapshot.total_balance
            
            # Calculate dynamic risk ratio (Speed Demon core feature)
            base_risk_ratio = self._calculate_dynamic_risk_ratio(balance_usd)
            
            # Get market regime adjustment
            regime = await self._detect_market_regime()
            regime_multiplier = self._get_regime_risk_multiplier(regime)
            
            # Final risk ratio with regime adjustment
            final_risk_ratio = base_risk_ratio * regime_multiplier
            
            # Calculate position size limit
            position_size_limit = balance_usd * Decimal(str(final_risk_ratio))
            
            # Determine risk level based on final ratio
            risk_level = self._classify_risk_level(final_risk_ratio)
            
            # Calculate portfolio risk score
            portfolio_risk_score = self._calculate_portfolio_risk_score(
                balance_usd, final_risk_ratio, regime
            )
            
            # Determine balance tier for UI display
            balance_tier = self._get_balance_tier(balance_usd)
            
            metrics = RiskMetrics(
                portfolio_risk_score=portfolio_risk_score,
                position_size_limit=position_size_limit,
                current_drawdown=0.0,  # TODO: Calculate from position history
                max_positions=self._calculate_max_positions(balance_usd, final_risk_ratio),
                risk_level=risk_level,
                regime_adjustment=regime_multiplier,
                balance_tier=balance_tier
            )
            
            self.logger.info(f"[OK] Risk metrics calculated: {risk_level.value}, ${position_size_limit:.2f} limit")
            return metrics
            
        except Exception as e:
            self.logger.error(f"[ERROR] Risk metrics calculation failed: {e}")
            # Return conservative defaults
            return RiskMetrics(
                portfolio_risk_score=80.0,
                position_size_limit=Decimal('100'),
                current_drawdown=0.0,
                max_positions=1,
                risk_level=RiskLevel.CONSERVATIVE,
                regime_adjustment=0.5,
                balance_tier="small"
            )
    
    def _calculate_dynamic_risk_ratio(self, balance_usd: Decimal) -> float:
        """
        Calculate dynamic risk ratio using Speed Demon algorithm.
        
        Core Speed Demon feature: Higher risk for small accounts,
        exponential decay to conservative risk for large accounts.
        """
        if not self.speed_demon_config.get('enabled', True):
            return 0.01  # 1% default if Speed Demon disabled
        
        scaling_config = self.speed_demon_config.get('dynamic_risk_scaling', {})
        if not scaling_config.get('enabled', True):
            return scaling_config.get('fallback_risk', 0.01)
        
        # Speed Demon parameters
        small_risk = scaling_config.get('small_account_risk', 0.02)
        large_risk = scaling_config.get('large_account_risk', 0.005)
        transition_start = scaling_config.get('transition_start', 10000)
        transition_end = scaling_config.get('transition_end', 100000)
        decay_factor = scaling_config.get('decay_factor', 0.5)
        
        balance_float = float(balance_usd)
        
        # Small account: use high risk
        if balance_float <= transition_start:
            return small_risk
        
        # Large account: use low risk
        if balance_float >= transition_end:
            return large_risk
        
        # Transition range: exponential decay
        range_ratio = (balance_float - transition_start) / (transition_end - transition_start)
        decay_multiplier = math.exp(-decay_factor * range_ratio * 5)  # Scale decay
        risk_range = small_risk - large_risk
        
        dynamic_risk = large_risk + (risk_range * decay_multiplier)
        
        self.logger.debug(f"[DEBUG] Dynamic risk: ${balance_float:.0f} -> {dynamic_risk:.3%}")
        return dynamic_risk
    
    async def _detect_market_regime(self) -> MarketRegime:
        """
        Detect current market regime for risk adjustment.
        
        Returns:
            Current market regime classification
        """
        # TODO: Implement actual market regime detection
        # For now, return normal regime
        return MarketRegime.NORMAL
    
    def _get_regime_risk_multiplier(self, regime: MarketRegime) -> float:
        """Get risk multiplier based on market regime"""
        multipliers = {
            MarketRegime.LOW_VOLATILITY: 1.2,    # Slightly higher risk in calm markets
            MarketRegime.NORMAL: 1.0,            # Normal risk
            MarketRegime.HIGH_VOLATILITY: 0.7,   # Reduce risk in volatile markets
            MarketRegime.CRISIS: 0.3             # Very conservative in crisis
        }
        return multipliers.get(regime, 1.0)
    
    def _classify_risk_level(self, risk_ratio: float) -> RiskLevel:
        """Classify risk level based on risk ratio"""
        if risk_ratio >= 0.015:        # 1.5%+
            return RiskLevel.VERY_AGGRESSIVE
        elif risk_ratio >= 0.01:       # 1.0%+
            return RiskLevel.AGGRESSIVE
        elif risk_ratio >= 0.005:      # 0.5%+
            return RiskLevel.MODERATE
        else:                          # < 0.5%
            return RiskLevel.CONSERVATIVE
    
    def _calculate_portfolio_risk_score(self, balance: Decimal, risk_ratio: float, regime: MarketRegime) -> float:
        """Calculate overall portfolio risk score (0-100)"""
        # Base score from risk ratio
        risk_score = min(risk_ratio * 2000, 50)  # Scale to 0-50
        
        # Regime adjustment
        if regime == MarketRegime.HIGH_VOLATILITY:
            risk_score *= 1.3
        elif regime == MarketRegime.CRISIS:
            risk_score *= 1.5
        
        return min(risk_score, 100)
    
    def _calculate_max_positions(self, balance: Decimal, risk_ratio: float) -> int:
        """Calculate maximum number of simultaneous positions"""
        balance_float = float(balance)
        
        if balance_float < 1000:
            return 1
        elif balance_float < 10000:
            return 2
        elif balance_float < 50000:
            return 3
        else:
            return min(5, int(balance_float / 20000))
    
    def _get_balance_tier(self, balance: Decimal) -> str:
        """Get balance tier for UI display"""
        balance_float = float(balance)
        
        if balance_float < 1000:
            return "micro"
        elif balance_float < 10000:
            return "small"
        elif balance_float < 100000:
            return "medium"
        elif balance_float < 1000000:
            return "large"
        else:
            return "whale"
    
    async def get_balance_snapshot(self) -> BalanceSnapshot:
        """
        Get comprehensive balance snapshot across all environments
        
        Returns:
            Current balance state with multi-environment data
        """
        try:
            # TODO: Implement actual balance fetching from APIs
            # For now, return mock data
            
            mock_balance = BalanceSnapshot(
                total_balance=Decimal('5000.00'),
                available_balance=Decimal('4500.00'),
                unrealized_pnl=Decimal('150.00'),
                margin_used=Decimal('500.00'),
                environments={
                    'testnet': {
                        'balance': 5000.00,
                        'status': 'active',
                        'unrealized_pnl': 150.00
                    },
                    'mainnet': {
                        'balance': 0.00,
                        'status': 'inactive',
                        'unrealized_pnl': 0.00
                    }
                },
                timestamp=datetime.now()
            )
            
            self.logger.debug(f"[DEBUG] Balance snapshot: ${mock_balance.total_balance}")
            return mock_balance
            
        except Exception as e:
            self.logger.error(f"[ERROR] Balance snapshot failed: {e}")
            # Return safe defaults
            return BalanceSnapshot(
                total_balance=Decimal('1000.00'),
                available_balance=Decimal('1000.00'),
                unrealized_pnl=Decimal('0.00'),
                margin_used=Decimal('0.00'),
                environments={},
                timestamp=datetime.now()
            )
    
    async def calculate_position_size(self, symbol: str, entry_price: Decimal, 
                                     stop_loss_price: Optional[Decimal] = None) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Calculate optimal position size for a trade
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            entry_price: Expected entry price
            stop_loss_price: Stop loss price (optional)
            
        Returns:
            Tuple of (position_size, calculation_details)
        """
        try:
            # Get current risk metrics
            risk_metrics = await self.get_current_risk_metrics()
            
            # Calculate position size based on risk limit
            max_risk_usd = risk_metrics.position_size_limit
            
            if stop_loss_price:
                # Calculate based on stop loss distance
                risk_per_unit = abs(entry_price - stop_loss_price)
                position_size = max_risk_usd / risk_per_unit if risk_per_unit > 0 else Decimal('0')
            else:
                # Use default 2% stop loss assumption
                risk_per_unit = entry_price * Decimal('0.02')
                position_size = max_risk_usd / risk_per_unit
            
            calculation_details = {
                'risk_limit_usd': float(max_risk_usd),
                'risk_per_unit': float(risk_per_unit),
                'position_size': float(position_size),
                'risk_level': risk_metrics.risk_level.value,
                'regime_adjustment': risk_metrics.regime_adjustment,
                'balance_tier': risk_metrics.balance_tier
            }
            
            self.logger.info(f"[OK] Position size calculated: {position_size:.4f} {symbol}")
            return position_size, calculation_details
            
        except Exception as e:
            self.logger.error(f"[ERROR] Position size calculation failed: {e}")
            return Decimal('0'), {'error': str(e)}


# Global unified engine instance
unified_engine = UnifiedRiskBalanceEngine()