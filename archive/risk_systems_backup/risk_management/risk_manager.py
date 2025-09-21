"""
Risk Management System for Bybit Trading Bot

This module implements comprehensive risk management including:
- Dynamic position sizing based on account balance
- Portfolio drawdown monitoring and limits  
- Strategy-level risk controls
- Real-time risk metrics calculation
- Automatic position reduction on risk limit breaches

Supports both conservative and aggressive trading modes with
configurable risk parameters and decay functions.

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import math

from ..config_manager import ConfigurationManager
from ..utils.logging import setup_logger


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingMode(Enum):
    """Trading mode enumeration."""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskMetrics:
    """Risk metrics data structure."""
    current_risk_ratio: Decimal
    portfolio_drawdown: Decimal
    strategy_drawdown: Decimal
    daily_var: Decimal
    sharpe_ratio: Decimal
    consistency_score: Decimal
    max_position_size: Decimal
    available_balance: Decimal
    risk_level: RiskLevel
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/API."""
        return {
            'current_risk_ratio': float(self.current_risk_ratio),
            'portfolio_drawdown': float(self.portfolio_drawdown),
            'strategy_drawdown': float(self.strategy_drawdown),
            'daily_var': float(self.daily_var),
            'sharpe_ratio': float(self.sharpe_ratio),
            'consistency_score': float(self.consistency_score),
            'max_position_size': float(self.max_position_size),
            'available_balance': float(self.available_balance),
            'risk_level': self.risk_level.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class TradeRiskAssessment:
    """Risk assessment for a potential trade."""
    is_approved: bool
    risk_ratio: Decimal
    position_size: Decimal
    stop_loss_price: Optional[Decimal]
    take_profit_price: Optional[Decimal]
    risk_reason: Optional[str] = None
    max_loss: Decimal = Decimal('0')
    reward_to_risk: Decimal = Decimal('0')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_approved': self.is_approved,
            'risk_ratio': float(self.risk_ratio),
            'position_size': float(self.position_size),
            'stop_loss_price': float(self.stop_loss_price) if self.stop_loss_price else None,
            'take_profit_price': float(self.take_profit_price) if self.take_profit_price else None,
            'risk_reason': self.risk_reason,
            'max_loss': float(self.max_loss),
            'reward_to_risk': float(self.reward_to_risk)
        }


class RiskManager:
    """
    Comprehensive risk management system.
    
    Features:
    - Dynamic position sizing based on account balance and volatility
    - Portfolio and strategy-level drawdown monitoring
    - Real-time risk metrics calculation
    - Configurable risk parameters for different trading modes
    - Automatic position reduction on limit breaches
    - Risk-adjusted position sizing with exponential/linear decay
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager
        self.logger = setup_logger("risk_manager")
        
        # Load risk configuration
        self.trading_mode = TradingMode(config_manager.get('trading.mode', 'conservative'))
        self.base_balance = Decimal(str(config_manager.get('trading.base_balance', 10000)))
        
        # Load mode-specific parameters
        if self.trading_mode == TradingMode.AGGRESSIVE:
            self._load_aggressive_config()
        else:
            self._load_conservative_config()
        
        # Risk tracking
        self.portfolio_peak = Decimal('0')
        self.strategy_peaks: Dict[str, Decimal] = {}
        self.daily_returns: List[Decimal] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Performance tracking for Sharpe ratio calculation
        self.return_history: pd.Series = pd.Series(dtype=float)
        self.last_balance_update = datetime.now()
        
        self.logger.info(f"RiskManager initialized - Mode: {self.trading_mode.value}")
    
    def _load_aggressive_config(self) -> None:
        """Load aggressive mode risk parameters."""
        config_path = 'trading.aggressive_mode'
        
        self.max_risk_ratio = Decimal(str(self.config.get(f'{config_path}.max_risk_ratio', 0.02)))
        self.min_risk_ratio = Decimal(str(self.config.get(f'{config_path}.min_risk_ratio', 0.005)))
        self.portfolio_drawdown_limit = Decimal(str(self.config.get(f'{config_path}.portfolio_drawdown_limit', 0.40)))
        self.strategy_drawdown_limit = Decimal(str(self.config.get(f'{config_path}.strategy_drawdown_limit', 0.25)))
        self.sharpe_ratio_min = Decimal(str(self.config.get(f'{config_path}.sharpe_ratio_min', 0.5)))
        self.var_daily_limit = Decimal(str(self.config.get(f'{config_path}.var_daily_limit', 0.05)))
        self.consistency_min = Decimal(str(self.config.get(f'{config_path}.consistency_min', 0.50)))
        
        # Balance thresholds for dynamic scaling
        balance_config = self.config.get(f'{config_path}.balance_thresholds', {})
        self.balance_low = Decimal(str(balance_config.get('low', 10000)))
        self.balance_high = Decimal(str(balance_config.get('high', 100000)))
        
        # Risk decay function
        self.risk_decay = self.config.get(f'{config_path}.risk_decay', 'exponential')
    
    def _load_conservative_config(self) -> None:
        """Load conservative mode risk parameters."""
        config_path = 'trading.conservative_mode'
        
        self.max_risk_ratio = Decimal(str(self.config.get(f'{config_path}.risk_ratio', 0.01)))
        self.min_risk_ratio = self.max_risk_ratio  # Fixed risk in conservative mode
        self.portfolio_drawdown_limit = Decimal(str(self.config.get(f'{config_path}.portfolio_drawdown_limit', 0.25)))
        self.strategy_drawdown_limit = Decimal(str(self.config.get(f'{config_path}.strategy_drawdown_limit', 0.15)))
        self.sharpe_ratio_min = Decimal(str(self.config.get(f'{config_path}.sharpe_ratio_min', 0.8)))
        self.var_daily_limit = Decimal(str(self.config.get(f'{config_path}.var_daily_limit', 0.03)))
        self.consistency_min = Decimal(str(self.config.get(f'{config_path}.consistency_min', 0.60)))
        
        # Conservative mode uses fixed parameters
        self.balance_low = self.base_balance
        self.balance_high = self.base_balance
        self.risk_decay = 'linear'
    
    async def assess_trade_risk(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        stop_loss: Optional[Decimal],
        take_profit: Optional[Decimal],
        current_balance: Decimal,
        strategy_id: str = "default"
    ) -> TradeRiskAssessment:
        """
        Assess risk for a potential trade.
        
        Args:
            symbol: Trading symbol
            side: Trade side ('buy' or 'sell')
            entry_price: Planned entry price
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            current_balance: Current account balance
            strategy_id: Strategy identifier
            
        Returns:
            TradeRiskAssessment object
        """
        try:
            # Calculate current risk metrics
            risk_metrics = await self.calculate_risk_metrics(current_balance, strategy_id)
            
            # Check if trading is allowed based on current risk level
            if risk_metrics.risk_level == RiskLevel.CRITICAL:
                return TradeRiskAssessment(
                    is_approved=False,
                    risk_ratio=Decimal('0'),
                    position_size=Decimal('0'),
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    risk_reason="Critical risk level - trading suspended"
                )
            
            # Calculate dynamic risk ratio based on balance
            risk_ratio = self._calculate_dynamic_risk_ratio(current_balance)
            
            # Calculate position size based on risk ratio and stop loss
            if stop_loss:
                # Risk-based position sizing
                risk_distance = abs(entry_price - stop_loss) / entry_price
                if risk_distance > 0:
                    max_loss = current_balance * risk_ratio
                    position_size = max_loss / (entry_price * risk_distance)
                else:
                    position_size = Decimal('0')
            else:
                # Default position sizing without stop loss (conservative)
                position_size = (current_balance * risk_ratio) / entry_price
            
            # Apply additional risk constraints
            max_position_value = current_balance * Decimal('0.1')  # Max 10% per position
            if position_size * entry_price > max_position_value:
                position_size = max_position_value / entry_price
            
            # Calculate risk metrics for this trade
            max_loss = Decimal('0')
            reward_to_risk = Decimal('0')
            
            if stop_loss:
                max_loss = position_size * abs(entry_price - stop_loss)
                
                if take_profit and max_loss > 0:
                    potential_profit = position_size * abs(take_profit - entry_price)
                    reward_to_risk = potential_profit / max_loss
            
            # Final approval checks
            is_approved = True
            risk_reason = None
            
            # Check minimum reward-to-risk ratio
            if reward_to_risk > 0 and reward_to_risk < Decimal('1.5'):
                is_approved = False
                risk_reason = f"Insufficient reward-to-risk ratio: {reward_to_risk:.2f} < 1.5"
            
            # Check position size is meaningful
            if position_size <= 0:
                is_approved = False
                risk_reason = "Position size too small or invalid"
            
            # Check maximum loss doesn't exceed risk ratio
            loss_ratio = max_loss / current_balance if current_balance > 0 else Decimal('1')
            if loss_ratio > risk_ratio * Decimal('1.1'):  # 10% buffer
                is_approved = False
                risk_reason = f"Max loss exceeds risk limit: {loss_ratio:.3f} > {risk_ratio:.3f}"
            
            return TradeRiskAssessment(
                is_approved=is_approved,
                risk_ratio=risk_ratio,
                position_size=position_size,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                risk_reason=risk_reason,
                max_loss=max_loss,
                reward_to_risk=reward_to_risk
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing trade risk: {e}")
            return TradeRiskAssessment(
                is_approved=False,
                risk_ratio=Decimal('0'),
                position_size=Decimal('0'),
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                risk_reason=f"Risk assessment error: {str(e)}"
            )
    
    async def calculate_risk_metrics(
        self,
        current_balance: Decimal,
        strategy_id: str = "default"
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            current_balance: Current account balance
            strategy_id: Strategy identifier
            
        Returns:
            RiskMetrics object
        """
        try:
            # Update portfolio peak
            if current_balance > self.portfolio_peak:
                self.portfolio_peak = current_balance
            
            # Calculate portfolio drawdown
            portfolio_drawdown = Decimal('0')
            if self.portfolio_peak > 0:
                portfolio_drawdown = (self.portfolio_peak - current_balance) / self.portfolio_peak
            
            # Calculate strategy drawdown
            strategy_peak = self.strategy_peaks.get(strategy_id, current_balance)
            if current_balance > strategy_peak:
                strategy_peak = current_balance
                self.strategy_peaks[strategy_id] = strategy_peak
            
            strategy_drawdown = Decimal('0')
            if strategy_peak > 0:
                strategy_drawdown = (strategy_peak - current_balance) / strategy_peak
            
            # Calculate current risk ratio
            current_risk_ratio = self._calculate_dynamic_risk_ratio(current_balance)
            
            # Calculate daily VaR (Value at Risk)
            daily_var = self._calculate_daily_var()
            
            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Calculate consistency score
            consistency_score = self._calculate_consistency_score()
            
            # Determine risk level
            risk_level = self._determine_risk_level(
                portfolio_drawdown, strategy_drawdown, daily_var, sharpe_ratio, consistency_score
            )
            
            # Calculate maximum position size
            max_position_size = current_balance * current_risk_ratio
            
            return RiskMetrics(
                current_risk_ratio=current_risk_ratio,
                portfolio_drawdown=portfolio_drawdown,
                strategy_drawdown=strategy_drawdown,
                daily_var=daily_var,
                sharpe_ratio=sharpe_ratio,
                consistency_score=consistency_score,
                max_position_size=max_position_size,
                available_balance=current_balance,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                current_risk_ratio=self.min_risk_ratio,
                portfolio_drawdown=Decimal('0'),
                strategy_drawdown=Decimal('0'),
                daily_var=Decimal('0'),
                sharpe_ratio=Decimal('0'),
                consistency_score=Decimal('0'),
                max_position_size=Decimal('0'),
                available_balance=current_balance,
                risk_level=RiskLevel.HIGH
            )
    
    def _calculate_dynamic_risk_ratio(self, current_balance: Decimal) -> Decimal:
        """Calculate dynamic risk ratio based on balance and decay function."""
        if self.trading_mode == TradingMode.CONSERVATIVE:
            return self.max_risk_ratio
        
        # Aggressive mode - dynamic scaling
        if current_balance <= self.balance_low:
            return self.max_risk_ratio
        elif current_balance >= self.balance_high:
            return self.min_risk_ratio
        
        # Scale between min and max based on decay function
        balance_ratio = (current_balance - self.balance_low) / (self.balance_high - self.balance_low)
        
        if self.risk_decay == 'exponential':
            # Exponential decay
            decay_factor = Decimal(str(math.exp(-2 * float(balance_ratio))))
            risk_ratio = self.min_risk_ratio + (self.max_risk_ratio - self.min_risk_ratio) * decay_factor
        else:
            # Linear decay
            risk_ratio = self.max_risk_ratio - (self.max_risk_ratio - self.min_risk_ratio) * balance_ratio
        
        return max(self.min_risk_ratio, min(self.max_risk_ratio, risk_ratio))
    
    def _calculate_daily_var(self) -> Decimal:
        """Calculate daily Value at Risk (95% confidence)."""
        if len(self.daily_returns) < 30:  # Need at least 30 days
            return Decimal('0')
        
        try:
            returns_array = np.array([float(r) for r in self.daily_returns[-30:]])
            var_95 = np.percentile(returns_array, 5)  # 5th percentile for 95% VaR
            return abs(Decimal(str(var_95)))
        except Exception:
            return Decimal('0')
    
    def _calculate_sharpe_ratio(self) -> Decimal:
        """Calculate Sharpe ratio over the last 30 trading periods."""
        if len(self.return_history) < 30:
            return Decimal('0')
        
        try:
            recent_returns = self.return_history.tail(30)
            if recent_returns.std() == 0:
                return Decimal('0')
            
            excess_return = recent_returns.mean()  # Assuming risk-free rate ≈ 0
            sharpe = excess_return / recent_returns.std() * np.sqrt(252)  # Annualized
            return Decimal(str(sharpe))
        except Exception:
            return Decimal('0')
    
    def _calculate_consistency_score(self) -> Decimal:
        """Calculate trading consistency score (0-1)."""
        if len(self.trade_history) < 10:
            return Decimal('0.5')  # Neutral score for insufficient data
        
        try:
            recent_trades = self.trade_history[-20:]  # Last 20 trades
            winning_trades = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
            win_rate = winning_trades / len(recent_trades)
            
            # Calculate profit factor
            total_profit = sum(trade.get('pnl', 0) for trade in recent_trades if trade.get('pnl', 0) > 0)
            total_loss = abs(sum(trade.get('pnl', 0) for trade in recent_trades if trade.get('pnl', 0) < 0))
            
            profit_factor = total_profit / total_loss if total_loss > 0 else 1.0
            
            # Combine win rate and profit factor for consistency score
            consistency = (win_rate * 0.6) + (min(profit_factor / 2.0, 1.0) * 0.4)
            return Decimal(str(consistency))
            
        except Exception:
            return Decimal('0.5')
    
    def _determine_risk_level(
        self,
        portfolio_drawdown: Decimal,
        strategy_drawdown: Decimal,
        daily_var: Decimal,
        sharpe_ratio: Decimal,
        consistency_score: Decimal
    ) -> RiskLevel:
        """Determine overall risk level based on metrics."""
        risk_flags = 0
        
        # Check drawdown limits
        if portfolio_drawdown > self.portfolio_drawdown_limit:
            risk_flags += 3  # Critical flag
        elif portfolio_drawdown > self.portfolio_drawdown_limit * Decimal('0.8'):
            risk_flags += 2  # High flag
        elif portfolio_drawdown > self.portfolio_drawdown_limit * Decimal('0.6'):
            risk_flags += 1  # Medium flag
        
        # Check strategy drawdown
        if strategy_drawdown > self.strategy_drawdown_limit:
            risk_flags += 2
        elif strategy_drawdown > self.strategy_drawdown_limit * Decimal('0.8'):
            risk_flags += 1
        
        # Check VaR
        if daily_var > self.var_daily_limit:
            risk_flags += 2
        elif daily_var > self.var_daily_limit * Decimal('0.8'):
            risk_flags += 1
        
        # Check Sharpe ratio
        if sharpe_ratio < self.sharpe_ratio_min:
            risk_flags += 1
        
        # Check consistency
        if consistency_score < self.consistency_min:
            risk_flags += 1
        
        # Determine risk level
        if risk_flags >= 4:
            return RiskLevel.CRITICAL
        elif risk_flags >= 3:
            return RiskLevel.HIGH
        elif risk_flags >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def update_performance(
        self,
        current_balance: Decimal,
        trade_pnl: Optional[Decimal] = None,
        strategy_id: str = "default"
    ) -> None:
        """
        Update performance tracking data.
        
        Args:
            current_balance: Current account balance
            trade_pnl: PnL from completed trade (optional)
            strategy_id: Strategy identifier
        """
        try:
            now = datetime.now()
            
            # Update return history for Sharpe calculation
            if len(self.return_history) > 0:
                last_balance = self.return_history.iloc[-1] if len(self.return_history) > 0 else float(current_balance)
                daily_return = (float(current_balance) - last_balance) / last_balance if last_balance > 0 else 0
                self.return_history = pd.concat([
                    self.return_history, 
                    pd.Series([daily_return], index=[now])
                ])
            else:
                self.return_history = pd.Series([0.0], index=[now])
            
            # Keep only last 100 periods
            if len(self.return_history) > 100:
                self.return_history = self.return_history.tail(100)
            
            # Update daily returns for VaR calculation
            if (now - self.last_balance_update).days >= 1:
                if len(self.daily_returns) > 0:
                    last_balance = self.daily_returns[-1] if self.daily_returns else current_balance
                    daily_return = (current_balance - last_balance) / last_balance if last_balance > 0 else Decimal('0')
                    self.daily_returns.append(daily_return)
                else:
                    self.daily_returns.append(Decimal('0'))
                
                # Keep only last 60 days
                if len(self.daily_returns) > 60:
                    self.daily_returns = self.daily_returns[-60:]
                
                self.last_balance_update = now
            
            # Record trade if PnL provided
            if trade_pnl is not None:
                self.trade_history.append({
                    'timestamp': now,
                    'pnl': float(trade_pnl),
                    'strategy_id': strategy_id,
                    'balance': float(current_balance)
                })
                
                # Keep only last 100 trades
                if len(self.trade_history) > 100:
                    self.trade_history = self.trade_history[-100:]
            
        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")
    
    def get_risk_limits(self) -> Dict[str, Any]:
        """Get current risk limits configuration."""
        return {
            'trading_mode': self.trading_mode.value,
            'max_risk_ratio': float(self.max_risk_ratio),
            'min_risk_ratio': float(self.min_risk_ratio),
            'portfolio_drawdown_limit': float(self.portfolio_drawdown_limit),
            'strategy_drawdown_limit': float(self.strategy_drawdown_limit),
            'sharpe_ratio_min': float(self.sharpe_ratio_min),
            'var_daily_limit': float(self.var_daily_limit),
            'consistency_min': float(self.consistency_min),
            'balance_low': float(self.balance_low),
            'balance_high': float(self.balance_high),
            'risk_decay': self.risk_decay
        }