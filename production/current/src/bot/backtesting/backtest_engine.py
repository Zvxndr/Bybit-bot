"""
Backtesting Engine for Bybit Trading Bot

This module implements a comprehensive backtesting system for:
- Historical strategy validation and testing
- Performance analysis and metrics calculation
- Risk-adjusted return evaluation
- Strategy parameter optimization
- Walk-forward analysis and validation

Provides realistic simulation of trading strategies with
proper handling of slippage, commissions, and market impact.

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
from decimal import Decimal
import matplotlib.pyplot as plt
import seaborn as sns

from ..config_manager import ConfigurationManager
from ..utils.logging import TradingLogger
from ..core.strategy_manager import BaseStrategy, TradingSignal, SignalType
from ..risk_management.risk_manager import RiskManager
from ..risk_management.portfolio_manager import PortfolioManager


@dataclass
class BacktestTrade:
    """Individual backtest trade record."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    entry_price: Decimal
    exit_price: Optional[Decimal] = None
    exit_timestamp: Optional[datetime] = None
    pnl: Decimal = Decimal('0')
    commission: Decimal = Decimal('0')
    slippage: Decimal = Decimal('0')
    duration_hours: float = 0.0
    strategy_id: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side,
            'quantity': float(self.quantity),
            'entry_price': float(self.entry_price),
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            'pnl': float(self.pnl),
            'commission': float(self.commission),
            'slippage': float(self.slippage),
            'duration_hours': self.duration_hours,
            'strategy_id': self.strategy_id
        }


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    start_date: datetime
    end_date: datetime
    initial_balance: Decimal
    final_balance: Decimal
    total_return: Decimal
    total_return_pct: Decimal
    annual_return_pct: Decimal
    max_drawdown_pct: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: Decimal
    profit_factor: Decimal
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    average_trade_duration: timedelta
    total_commission: Decimal
    total_slippage: Decimal
    
    # Daily metrics
    daily_returns: pd.Series = field(default_factory=pd.Series)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    
    # Trade list
    trades: List[BacktestTrade] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_balance': float(self.initial_balance),
            'final_balance': float(self.final_balance),
            'total_return': float(self.total_return),
            'total_return_pct': float(self.total_return_pct),
            'annual_return_pct': float(self.annual_return_pct),
            'max_drawdown_pct': float(self.max_drawdown_pct),
            'sharpe_ratio': float(self.sharpe_ratio),
            'sortino_ratio': float(self.sortino_ratio),
            'calmar_ratio': float(self.calmar_ratio),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': float(self.win_rate_pct),
            'profit_factor': float(self.profit_factor),
            'average_win': float(self.average_win),
            'average_loss': float(self.average_loss),
            'largest_win': float(self.largest_win),
            'largest_loss': float(self.largest_loss),
            'average_trade_duration_hours': self.average_trade_duration.total_seconds() / 3600,
            'total_commission': float(self.total_commission),
            'total_slippage': float(self.total_slippage)
        }


class BacktestEngine:
    """
    Comprehensive backtesting engine for strategy validation.
    
    Features:
    - Historical data simulation with realistic execution
    - Commission and slippage modeling
    - Risk management integration
    - Portfolio-level backtesting
    - Performance analytics and reporting
    - Walk-forward analysis support
    """
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        initial_balance: Decimal = Decimal('10000')
    ):
        self.config = config_manager
        self.initial_balance = initial_balance
        self.logger = TradingLogger("backtest_engine")
        
        # Backtest configuration
        self.commission_rate = Decimal(str(config_manager.get('backtesting.commission_rate', 0.001)))
        self.slippage_rate = Decimal(str(config_manager.get('backtesting.slippage_rate', 0.0005)))
        self.max_slippage = Decimal(str(config_manager.get('backtesting.max_slippage', 0.002)))
        
        # Current backtest state
        self.current_balance = initial_balance
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position data
        self.trades: List[BacktestTrade] = []
        self.daily_balances: List[Tuple[datetime, Decimal]] = []
        
        # Components
        self.risk_manager: Optional[RiskManager] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        
        self.logger.info(f"BacktestEngine initialized with balance: {initial_balance}")
    
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResults:
        """
        Run a complete backtest for a strategy.
        
        Args:
            strategy: Strategy to test
            data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            BacktestResults object
        """
        try:
            self.logger.info(f"Starting backtest: {strategy.strategy_id}")
            self.logger.info(f"Period: {start_date} to {end_date}")
            self.logger.info(f"Initial Balance: {self.initial_balance}")
            
            # Reset backtest state
            self._reset_state()
            
            # Initialize risk manager for backtest
            self.risk_manager = RiskManager(self.config)
            
            # Filter data for backtest period
            backtest_data = data[
                (data.index >= start_date) & 
                (data.index <= end_date)
            ].copy()
            
            if len(backtest_data) == 0:
                raise ValueError("No data available for backtest period")
            
            self.logger.info(f"Backtest data: {len(backtest_data)} rows")
            
            # Start strategy
            await strategy.on_start()
            
            # Main backtest loop
            for timestamp, row in backtest_data.iterrows():
                await self._process_timestamp(strategy, timestamp, row, backtest_data)
            
            # Close any remaining positions
            await self._close_all_positions(backtest_data.iloc[-1])
            
            # Stop strategy
            await strategy.on_stop()
            
            # Calculate results
            results = await self._calculate_results(start_date, end_date)
            
            self.logger.info(f"Backtest completed - Final Balance: {results.final_balance}")
            self.logger.info(f"Total Return: {results.total_return_pct:.2f}%")
            self.logger.info(f"Total Trades: {results.total_trades}")
            self.logger.info(f"Win Rate: {results.win_rate_pct:.2f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise
    
    async def _process_timestamp(
        self,
        strategy: BaseStrategy,
        timestamp: datetime,
        current_data: pd.Series,
        full_data: pd.DataFrame
    ) -> None:
        """Process a single timestamp in the backtest."""
        try:
            # Update position values with current prices
            await self._update_positions(current_data)
            
            # Record daily balance
            if len(self.daily_balances) == 0 or timestamp.date() != self.daily_balances[-1][0].date():
                self.daily_balances.append((timestamp, self.current_balance))
            
            # Get lookback data for strategy
            current_idx = full_data.index.get_loc(timestamp)
            lookback_periods = getattr(strategy, 'lookback_periods', 100)
            start_idx = max(0, current_idx - lookback_periods + 1)
            
            strategy_data = full_data.iloc[start_idx:current_idx + 1]
            
            # Generate trading signal
            symbol = current_data.name if hasattr(current_data, 'name') else 'BTCUSDT'  # Default symbol
            signal = await strategy.generate_signal(symbol, strategy_data)
            
            if signal:
                await self._execute_signal(signal, current_data, timestamp)
            
        except Exception as e:
            self.logger.error(f"Error processing timestamp {timestamp}: {e}")
    
    async def _execute_signal(
        self,
        signal: TradingSignal,
        market_data: pd.Series,
        timestamp: datetime
    ) -> None:
        """Execute a trading signal in the backtest."""
        try:
            symbol = signal.symbol
            current_price = Decimal(str(market_data.get('close', market_data.iloc[-1])))
            
            # Check if we have enough balance or existing position
            if signal.signal_type == SignalType.BUY:
                await self._execute_buy_signal(signal, current_price, timestamp)
            elif signal.signal_type == SignalType.SELL:
                await self._execute_sell_signal(signal, current_price, timestamp)
            elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                await self._close_position(symbol, current_price, timestamp)
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
    
    async def _execute_buy_signal(
        self,
        signal: TradingSignal,
        current_price: Decimal,
        timestamp: datetime
    ) -> None:
        """Execute a buy signal."""
        try:
            symbol = signal.symbol
            
            # Calculate position size based on risk management
            if self.risk_manager:
                risk_assessment = await self.risk_manager.assess_trade_risk(
                    symbol=symbol,
                    side="buy",
                    entry_price=current_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    current_balance=self.current_balance,
                    strategy_id=signal.strategy_id
                )
                
                if not risk_assessment.is_approved:
                    self.logger.debug(f"Buy signal rejected: {risk_assessment.risk_reason}")
                    return
                
                quantity = risk_assessment.position_size
            else:
                # Default position sizing (5% of balance)
                position_value = self.current_balance * Decimal('0.05')
                quantity = position_value / current_price
            
            # Apply slippage and commission
            execution_price = self._apply_slippage(current_price, "buy")
            commission = quantity * execution_price * self.commission_rate
            total_cost = quantity * execution_price + commission
            
            if total_cost > self.current_balance:
                self.logger.debug(f"Insufficient balance for buy: {total_cost} > {self.current_balance}")
                return
            
            # Execute trade
            self.current_balance -= total_cost
            
            # Update or create position
            if symbol in self.positions:
                # Average into existing position
                existing = self.positions[symbol]
                total_quantity = existing['quantity'] + quantity
                avg_price = (existing['quantity'] * existing['entry_price'] + 
                           quantity * execution_price) / total_quantity
                
                self.positions[symbol].update({
                    'quantity': total_quantity,
                    'entry_price': avg_price
                })
            else:
                # New position
                self.positions[symbol] = {
                    'side': 'long',
                    'quantity': quantity,
                    'entry_price': execution_price,
                    'entry_timestamp': timestamp,
                    'unrealized_pnl': Decimal('0')
                }
            
            # Record trade
            trade = BacktestTrade(
                timestamp=timestamp,
                symbol=symbol,
                side='buy',
                quantity=quantity,
                entry_price=execution_price,
                commission=commission,
                slippage=execution_price - current_price,
                strategy_id=signal.strategy_id
            )
            self.trades.append(trade)
            
            self.logger.debug(f"Buy executed: {symbol} {quantity:.6f} @ {execution_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error executing buy signal: {e}")
    
    async def _execute_sell_signal(
        self,
        signal: TradingSignal,
        current_price: Decimal,
        timestamp: datetime
    ) -> None:
        """Execute a sell signal (close long position)."""
        try:
            symbol = signal.symbol
            
            if symbol not in self.positions:
                self.logger.debug(f"No position to sell for {symbol}")
                return
            
            position = self.positions[symbol]
            if position['side'] != 'long':
                self.logger.debug(f"Cannot sell short position for {symbol}")
                return
            
            # Close the position
            await self._close_position(symbol, current_price, timestamp)
            
        except Exception as e:
            self.logger.error(f"Error executing sell signal: {e}")
    
    async def _close_position(
        self,
        symbol: str,
        current_price: Decimal,
        timestamp: datetime
    ) -> None:
        """Close an existing position."""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            quantity = position['quantity']
            entry_price = position['entry_price']
            
            # Apply slippage and commission
            execution_price = self._apply_slippage(current_price, "sell")
            commission = quantity * execution_price * self.commission_rate
            gross_proceeds = quantity * execution_price
            net_proceeds = gross_proceeds - commission
            
            # Calculate PnL
            cost_basis = quantity * entry_price
            pnl = net_proceeds - cost_basis
            
            # Update balance
            self.current_balance += net_proceeds
            
            # Calculate trade duration
            duration = timestamp - position['entry_timestamp']
            
            # Update trade record
            for trade in reversed(self.trades):
                if (trade.symbol == symbol and 
                    trade.side == 'buy' and 
                    trade.exit_price is None):
                    trade.exit_price = execution_price
                    trade.exit_timestamp = timestamp
                    trade.pnl = pnl
                    trade.commission += commission
                    trade.slippage += execution_price - current_price
                    trade.duration_hours = duration.total_seconds() / 3600
                    break
            
            # Remove position
            del self.positions[symbol]
            
            self.logger.debug(
                f"Position closed: {symbol} {quantity:.6f} @ {execution_price:.2f}, "
                f"PnL: {pnl:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
    
    async def _update_positions(self, market_data: pd.Series) -> None:
        """Update unrealized PnL for open positions."""
        try:
            for symbol, position in self.positions.items():
                if symbol in market_data.index or 'close' in market_data:
                    current_price = Decimal(str(market_data.get('close', market_data.iloc[-1])))
                    entry_price = position['entry_price']
                    quantity = position['quantity']
                    
                    if position['side'] == 'long':
                        unrealized_pnl = quantity * (current_price - entry_price)
                    else:  # short
                        unrealized_pnl = quantity * (entry_price - current_price)
                    
                    position['unrealized_pnl'] = unrealized_pnl
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _close_all_positions(self, final_data: pd.Series) -> None:
        """Close all remaining positions at the end of backtest."""
        try:
            final_timestamp = final_data.name if hasattr(final_data, 'name') else datetime.now()
            symbols_to_close = list(self.positions.keys())
            
            for symbol in symbols_to_close:
                current_price = Decimal(str(final_data.get('close', final_data.iloc[-1])))
                await self._close_position(symbol, current_price, final_timestamp)
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
    
    def _apply_slippage(self, price: Decimal, side: str) -> Decimal:
        """Apply realistic slippage to execution price."""
        try:
            # Calculate slippage based on volatility and market impact
            base_slippage = self.slippage_rate
            
            # Random component for realism
            import random
            random_factor = Decimal(str(random.uniform(0.5, 1.5)))
            actual_slippage = min(base_slippage * random_factor, self.max_slippage)
            
            if side == "buy":
                return price * (Decimal('1') + actual_slippage)
            else:  # sell
                return price * (Decimal('1') - actual_slippage)
                
        except Exception:
            return price
    
    async def _calculate_results(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        try:
            # Basic metrics
            total_return = self.current_balance - self.initial_balance
            total_return_pct = (total_return / self.initial_balance) * 100
            
            # Annualized return
            duration_years = (end_date - start_date).days / 365.25
            annual_return_pct = ((self.current_balance / self.initial_balance) ** (1 / duration_years) - 1) * 100 if duration_years > 0 else 0
            
            # Trade statistics
            completed_trades = [t for t in self.trades if t.exit_price is not None]
            total_trades = len(completed_trades)
            winning_trades = len([t for t in completed_trades if t.pnl > 0])
            losing_trades = total_trades - winning_trades
            win_rate_pct = (winning_trades / total_trades * 100) if total_trades > 0 else Decimal('0')
            
            # PnL statistics
            winning_pnls = [t.pnl for t in completed_trades if t.pnl > 0]
            losing_pnls = [t.pnl for t in completed_trades if t.pnl <= 0]
            
            average_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else Decimal('0')
            average_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else Decimal('0')
            largest_win = max(winning_pnls) if winning_pnls else Decimal('0')
            largest_loss = min(losing_pnls) if losing_pnls else Decimal('0')
            
            # Profit factor
            total_wins = sum(winning_pnls) if winning_pnls else Decimal('0')
            total_losses = abs(sum(losing_pnls)) if losing_pnls else Decimal('0.01')  # Avoid division by zero
            profit_factor = total_wins / total_losses
            
            # Duration statistics
            durations = [t.duration_hours for t in completed_trades if t.duration_hours > 0]
            average_trade_duration = timedelta(hours=sum(durations) / len(durations)) if durations else timedelta()
            
            # Create equity curve and calculate drawdown
            equity_curve = pd.Series(index=[b[0] for b in self.daily_balances], 
                                   data=[float(b[1]) for b in self.daily_balances])
            
            # Calculate drawdown
            rolling_max = equity_curve.expanding().max()
            drawdown_series = (equity_curve - rolling_max) / rolling_max * 100
            max_drawdown_pct = abs(drawdown_series.min()) if len(drawdown_series) > 0 else Decimal('0')
            
            # Calculate daily returns
            daily_returns = equity_curve.pct_change().dropna()
            
            # Risk metrics
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            sortino_ratio = self._calculate_sortino_ratio(daily_returns)
            calmar_ratio = float(annual_return_pct) / float(max_drawdown_pct) if max_drawdown_pct > 0 else Decimal('0')
            
            # Commission and slippage totals
            total_commission = sum(t.commission for t in self.trades)
            total_slippage = sum(abs(t.slippage) for t in self.trades)
            
            return BacktestResults(
                start_date=start_date,
                end_date=end_date,
                initial_balance=self.initial_balance,
                final_balance=self.current_balance,
                total_return=total_return,
                total_return_pct=Decimal(str(total_return_pct)),
                annual_return_pct=Decimal(str(annual_return_pct)),
                max_drawdown_pct=Decimal(str(max_drawdown_pct)),
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=Decimal(str(calmar_ratio)),
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate_pct=win_rate_pct,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                average_trade_duration=average_trade_duration,
                total_commission=total_commission,
                total_slippage=total_slippage,
                daily_returns=daily_returns,
                equity_curve=equity_curve,
                drawdown_series=drawdown_series,
                trades=self.trades.copy()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating results: {e}")
            raise
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> Decimal:
        """Calculate Sharpe ratio."""
        try:
            if len(returns) < 2 or returns.std() == 0:
                return Decimal('0')
            
            # Assume risk-free rate of 2% annually
            risk_free_daily = 0.02 / 252
            excess_returns = returns - risk_free_daily
            sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
            
            return Decimal(str(sharpe))
            
        except Exception:
            return Decimal('0')
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> Decimal:
        """Calculate Sortino ratio."""
        try:
            if len(returns) < 2:
                return Decimal('0')
            
            # Only use negative returns for downside deviation
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return Decimal('999')  # Perfect - no negative returns
            
            downside_std = negative_returns.std()
            if downside_std == 0:
                return Decimal('0')
            
            # Assume risk-free rate of 2% annually
            risk_free_daily = 0.02 / 252
            excess_return = returns.mean() - risk_free_daily
            sortino = excess_return / downside_std * np.sqrt(252)
            
            return Decimal(str(sortino))
            
        except Exception:
            return Decimal('0')
    
    def _reset_state(self) -> None:
        """Reset backtest state for new run."""
        self.current_balance = self.initial_balance
        self.positions.clear()
        self.trades.clear()
        self.daily_balances.clear()
    
    def generate_report(self, results: BacktestResults, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive backtest report."""
        try:
            report = f"""
# Backtest Results Report

## Summary
- **Strategy**: {results.trades[0].strategy_id if results.trades else 'Unknown'}
- **Period**: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}
- **Initial Balance**: ${results.initial_balance:,.2f}
- **Final Balance**: ${results.final_balance:,.2f}
- **Total Return**: ${results.total_return:,.2f} ({results.total_return_pct:.2f}%)
- **Annualized Return**: {results.annual_return_pct:.2f}%

## Risk Metrics
- **Maximum Drawdown**: {results.max_drawdown_pct:.2f}%
- **Sharpe Ratio**: {results.sharpe_ratio:.3f}
- **Sortino Ratio**: {results.sortino_ratio:.3f}
- **Calmar Ratio**: {results.calmar_ratio:.3f}

## Trading Statistics
- **Total Trades**: {results.total_trades}
- **Winning Trades**: {results.winning_trades}
- **Losing Trades**: {results.losing_trades}
- **Win Rate**: {results.win_rate_pct:.2f}%
- **Profit Factor**: {results.profit_factor:.2f}
- **Average Win**: ${results.average_win:.2f}
- **Average Loss**: ${results.average_loss:.2f}
- **Largest Win**: ${results.largest_win:.2f}
- **Largest Loss**: ${results.largest_loss:.2f}
- **Average Trade Duration**: {results.average_trade_duration}

## Costs
- **Total Commission**: ${results.total_commission:.2f}
- **Total Slippage**: ${results.total_slippage:.2f}
"""
            
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report)
                self.logger.info(f"Report saved to: {save_path}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"