"""
Advanced Multi-Asset Portfolio Manager.
Provides comprehensive portfolio management with optimization, constraints, and real-time monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize, Bounds, LinearConstraint
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class PositionType(Enum):
    """Position types."""
    LONG = "long"
    SHORT = "short"

class PortfolioStatus(Enum):
    """Portfolio status."""
    ACTIVE = "active"
    PAUSED = "paused"
    LIQUIDATING = "liquidating"
    CLOSED = "closed"

@dataclass
class Position:
    """Individual position in the portfolio."""
    symbol: str
    position_type: PositionType
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    weight: float
    target_weight: float
    risk_contribution: float
    last_updated: datetime
    entry_time: datetime
    
    @property
    def total_pnl(self) -> float:
        """Total profit/loss for position."""
        return self.unrealized_pnl + self.realized_pnl
    
    @property
    def pnl_percentage(self) -> float:
        """PnL as percentage of entry value."""
        entry_value = abs(self.quantity * self.entry_price)
        return (self.total_pnl / entry_value * 100) if entry_value > 0 else 0.0

@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    max_weight: float = 0.5  # Maximum weight per asset
    min_weight: float = 0.0  # Minimum weight per asset
    max_positions: int = 20  # Maximum number of positions
    max_turnover: float = 0.5  # Maximum portfolio turnover
    max_leverage: float = 1.0  # Maximum leverage
    max_sector_weight: float = 0.3  # Maximum sector concentration
    max_correlation: float = 0.8  # Maximum correlation between assets
    min_liquidity: float = 1000000  # Minimum daily volume
    risk_budget_constraints: Dict[str, float] = field(default_factory=dict)
    custom_constraints: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Portfolio:
    """Complete portfolio representation."""
    portfolio_id: str
    name: str
    base_currency: str
    total_value: float
    available_cash: float
    positions: Dict[str, Position]
    weights: Dict[str, float]
    target_weights: Dict[str, float]
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    constraints: PortfolioConstraints
    status: PortfolioStatus
    created_at: datetime
    last_updated: datetime
    benchmark: Optional[str] = None
    
    @property
    def invested_value(self) -> float:
        """Total invested value."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Total portfolio PnL."""
        return sum(pos.total_pnl for pos in self.positions.values())
    
    @property
    def num_positions(self) -> int:
        """Number of active positions."""
        return len([pos for pos in self.positions.values() if pos.quantity != 0])

class PortfolioManager:
    """Advanced multi-asset portfolio management system."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Portfolio management configuration
        self.portfolio_config = {
            'rebalance_frequency': 'daily',  # daily, weekly, monthly
            'rebalance_threshold': 0.05,  # 5% weight deviation
            'optimization_method': 'mean_variance',  # mean_variance, risk_parity, black_litterman
            'risk_model': 'sample_covariance',  # sample_covariance, shrinkage, factor_model
            'transaction_costs': 0.001,  # 0.1% transaction cost
            'slippage_model': 'linear',  # linear, sqrt, fixed
            'benchmark': 'BTC',
            'max_iterations': 1000,
            'convergence_tolerance': 1e-6,
            'supported_assets': [
                'BTC', 'ETH', 'ADA', 'SOL', 'AVAX', 'DOT', 'LINK', 'UNI',
                'AAVE', 'SUSHI', 'CRV', 'YFI', 'COMP', 'MKR', 'SNX'
            ]
        }
        
        # Active portfolios
        self.portfolios = {}
        
        # Market data cache
        self.price_data = {}
        self.correlation_cache = {}
        
        # Performance tracking
        self.performance_history = {}
        
        self.logger.info("PortfolioManager initialized")
    
    async def create_portfolio(self, 
                             portfolio_id: str,
                             name: str,
                             initial_capital: float,
                             base_currency: str = 'USDT',
                             constraints: PortfolioConstraints = None,
                             benchmark: str = None) -> Portfolio:
        """Create a new portfolio."""
        try:
            if portfolio_id in self.portfolios:
                raise ValueError(f"Portfolio {portfolio_id} already exists")
            
            # Default constraints
            if constraints is None:
                constraints = PortfolioConstraints()
            
            # Create portfolio
            portfolio = Portfolio(
                portfolio_id=portfolio_id,
                name=name,
                base_currency=base_currency,
                total_value=initial_capital,
                available_cash=initial_capital,
                positions={},
                weights={},
                target_weights={},
                performance_metrics={},
                risk_metrics={},
                constraints=constraints,
                status=PortfolioStatus.ACTIVE,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                benchmark=benchmark or self.portfolio_config['benchmark']
            )
            
            # Store portfolio
            self.portfolios[portfolio_id] = portfolio
            
            # Initialize performance tracking
            self.performance_history[portfolio_id] = []
            
            self.logger.info(f"Created portfolio {portfolio_id} with {initial_capital} {base_currency}")
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Failed to create portfolio {portfolio_id}: {e}")
            raise
    
    async def add_position(self, 
                         portfolio_id: str,
                         symbol: str,
                         quantity: float,
                         entry_price: float,
                         position_type: PositionType = PositionType.LONG) -> bool:
        """Add or update position in portfolio."""
        try:
            if portfolio_id not in self.portfolios:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            portfolio = self.portfolios[portfolio_id]
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            if current_price is None:
                raise ValueError(f"Could not get current price for {symbol}")
            
            # Calculate market value
            market_value = quantity * current_price
            
            # Check if position exists
            if symbol in portfolio.positions:
                # Update existing position
                existing_pos = portfolio.positions[symbol]
                
                # Calculate weighted average entry price
                total_quantity = existing_pos.quantity + quantity
                if total_quantity != 0:
                    weighted_entry_price = (
                        (existing_pos.quantity * existing_pos.entry_price + quantity * entry_price) 
                        / total_quantity
                    )
                else:
                    weighted_entry_price = entry_price
                
                # Update position
                existing_pos.quantity = total_quantity
                existing_pos.entry_price = weighted_entry_price
                existing_pos.current_price = current_price
                existing_pos.market_value = total_quantity * current_price
                existing_pos.unrealized_pnl = (current_price - weighted_entry_price) * total_quantity
                existing_pos.last_updated = datetime.now()
                
            else:
                # Create new position
                position = Position(
                    symbol=symbol,
                    position_type=position_type,
                    quantity=quantity,
                    entry_price=entry_price,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=(current_price - entry_price) * quantity,
                    realized_pnl=0.0,
                    weight=0.0,  # Will be calculated
                    target_weight=0.0,
                    risk_contribution=0.0,
                    last_updated=datetime.now(),
                    entry_time=datetime.now()
                )
                
                portfolio.positions[symbol] = position
            
            # Update portfolio
            await self._update_portfolio_metrics(portfolio_id)
            
            self.logger.info(f"Added position {symbol} to portfolio {portfolio_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add position {symbol} to portfolio {portfolio_id}: {e}")
            return False
    
    async def optimize_portfolio(self, 
                               portfolio_id: str,
                               method: str = 'mean_variance',
                               lookback_days: int = 252,
                               expected_returns: Dict[str, float] = None) -> Dict[str, float]:
        """Optimize portfolio allocation using specified method."""
        try:
            if portfolio_id not in self.portfolios:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            portfolio = self.portfolios[portfolio_id]
            
            if not portfolio.positions:
                self.logger.warning(f"No positions in portfolio {portfolio_id} to optimize")
                return {}
            
            # Get asset symbols
            symbols = list(portfolio.positions.keys())
            
            # Get historical data
            price_data = await self._get_historical_data(symbols, lookback_days)
            
            if price_data.empty:
                raise ValueError("No historical data available for optimization")
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Perform optimization based on method
            if method == 'mean_variance':
                optimal_weights = await self._mean_variance_optimization(
                    returns, portfolio.constraints, expected_returns
                )
            elif method == 'risk_parity':
                optimal_weights = await self._risk_parity_optimization(
                    returns, portfolio.constraints
                )
            elif method == 'minimum_variance':
                optimal_weights = await self._minimum_variance_optimization(
                    returns, portfolio.constraints
                )
            elif method == 'maximum_diversification':
                optimal_weights = await self._maximum_diversification_optimization(
                    returns, portfolio.constraints
                )
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            # Update target weights
            portfolio.target_weights = optimal_weights
            portfolio.last_updated = datetime.now()
            
            self.logger.info(f"Optimized portfolio {portfolio_id} using {method}")
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed for {portfolio_id}: {e}")
            raise
    
    async def _mean_variance_optimization(self, 
                                        returns: pd.DataFrame,
                                        constraints: PortfolioConstraints,
                                        expected_returns: Dict[str, float] = None) -> Dict[str, float]:
        """Perform mean-variance optimization."""
        if not HAS_SCIPY:
            raise ImportError("SciPy required for portfolio optimization")
        
        try:
            n_assets = len(returns.columns)
            
            # Calculate expected returns
            if expected_returns is None:
                mu = returns.mean().values * 252  # Annualized
            else:
                mu = np.array([expected_returns.get(symbol, returns[symbol].mean() * 252) 
                              for symbol in returns.columns])
            
            # Calculate covariance matrix
            cov_matrix = returns.cov().values * 252  # Annualized
            
            # Objective function (negative Sharpe ratio)
            def objective(weights):
                portfolio_return = np.dot(weights, mu)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                # Use risk-free rate from config or default to 2%
                risk_free_rate = self.config.get('risk_free_rate', 0.02)
                sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                return -sharpe  # Negative because we minimize
            
            # Constraints
            constraints_list = []
            
            # Weights sum to 1
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1.0
            })
            
            # Weight bounds
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.portfolio_config['max_iterations']}
            )
            
            if not result.success:
                self.logger.warning(f"Optimization did not converge: {result.message}")
                # Fall back to equal weights
                weights = np.array([1.0 / n_assets] * n_assets)
            else:
                weights = result.x
            
            # Create weights dictionary
            optimal_weights = {
                symbol: float(weight) 
                for symbol, weight in zip(returns.columns, weights)
            }
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Mean-variance optimization failed: {e}")
            raise
    
    async def _risk_parity_optimization(self, 
                                      returns: pd.DataFrame,
                                      constraints: PortfolioConstraints) -> Dict[str, float]:
        """Perform risk parity optimization."""
        if not HAS_SCIPY:
            raise ImportError("SciPy required for portfolio optimization")
        
        try:
            n_assets = len(returns.columns)
            
            # Calculate covariance matrix
            cov_matrix = returns.cov().values * 252
            
            # Risk parity objective: minimize sum of squared risk contribution differences
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                if portfolio_vol == 0:
                    return 1e6
                
                # Risk contributions
                marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol
                risk_contributions = weights * marginal_risk / portfolio_vol
                
                # Target equal risk contribution
                target_risk = 1.0 / n_assets
                
                # Sum of squared deviations from target
                return np.sum((risk_contributions - target_risk) ** 2)
            
            # Constraints
            constraints_list = []
            
            # Weights sum to 1
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1.0
            })
            
            # Weight bounds
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.portfolio_config['max_iterations']}
            )
            
            if not result.success:
                self.logger.warning(f"Risk parity optimization did not converge: {result.message}")
                weights = np.array([1.0 / n_assets] * n_assets)
            else:
                weights = result.x
            
            optimal_weights = {
                symbol: float(weight) 
                for symbol, weight in zip(returns.columns, weights)
            }
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Risk parity optimization failed: {e}")
            raise
    
    async def _minimum_variance_optimization(self, 
                                           returns: pd.DataFrame,
                                           constraints: PortfolioConstraints) -> Dict[str, float]:
        """Perform minimum variance optimization."""
        if not HAS_SCIPY:
            raise ImportError("SciPy required for portfolio optimization")
        
        try:
            n_assets = len(returns.columns)
            
            # Calculate covariance matrix
            cov_matrix = returns.cov().values * 252
            
            # Objective function (portfolio variance)
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints_list = []
            
            # Weights sum to 1
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1.0
            })
            
            # Weight bounds
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.portfolio_config['max_iterations']}
            )
            
            if not result.success:
                self.logger.warning(f"Minimum variance optimization did not converge")
                weights = np.array([1.0 / n_assets] * n_assets)
            else:
                weights = result.x
            
            optimal_weights = {
                symbol: float(weight) 
                for symbol, weight in zip(returns.columns, weights)
            }
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Minimum variance optimization failed: {e}")
            raise
    
    async def _maximum_diversification_optimization(self, 
                                                  returns: pd.DataFrame,
                                                  constraints: PortfolioConstraints) -> Dict[str, float]:
        """Perform maximum diversification optimization."""
        if not HAS_SCIPY:
            raise ImportError("SciPy required for portfolio optimization")
        
        try:
            n_assets = len(returns.columns)
            
            # Calculate volatilities and correlation matrix
            volatilities = returns.std().values * np.sqrt(252)
            correlation_matrix = returns.corr().values
            
            # Objective function (negative diversification ratio)
            def objective(weights):
                weighted_avg_vol = np.dot(weights, volatilities)
                
                # Portfolio volatility
                cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # Diversification ratio
                if portfolio_vol > 0:
                    div_ratio = weighted_avg_vol / portfolio_vol
                    return -div_ratio  # Negative because we minimize
                else:
                    return -1e6
            
            # Constraints
            constraints_list = []
            
            # Weights sum to 1
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1.0
            })
            
            # Weight bounds
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.portfolio_config['max_iterations']}
            )
            
            if not result.success:
                self.logger.warning(f"Maximum diversification optimization did not converge")
                weights = np.array([1.0 / n_assets] * n_assets)
            else:
                weights = result.x
            
            optimal_weights = {
                symbol: float(weight) 
                for symbol, weight in zip(returns.columns, weights)
            }
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Maximum diversification optimization failed: {e}")
            raise
    
    async def calculate_portfolio_metrics(self, portfolio_id: str) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics."""
        try:
            if portfolio_id not in self.portfolios:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            portfolio = self.portfolios[portfolio_id]
            
            # Get performance history
            history = self.performance_history.get(portfolio_id, [])
            
            if len(history) < 2:
                return {
                    'total_return': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'sortino_ratio': 0.0
                }
            
            # Convert to returns series
            values = [h['total_value'] for h in history]
            returns = pd.Series(values).pct_change().dropna()
            
            # Calculate metrics
            total_return = (values[-1] / values[0] - 1) * 100 if values[0] > 0 else 0
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized %
            
            # Sharpe ratio
            risk_free_rate = self.config.get('risk_free_rate', 0.02) / 252  # Daily
            excess_returns = returns - risk_free_rate
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # Sortino ratio
            downside_returns = returns[returns < risk_free_rate]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = excess_returns.mean() * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0
            
            # Update portfolio metrics
            portfolio.performance_metrics = {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'sortino_ratio': sortino_ratio
            }
            
            return portfolio.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio metrics for {portfolio_id}: {e}")
            return {}
    
    async def _update_portfolio_metrics(self, portfolio_id: str):
        """Update portfolio weights and metrics."""
        try:
            portfolio = self.portfolios[portfolio_id]
            
            # Update current prices and market values
            total_value = portfolio.available_cash
            
            for symbol, position in portfolio.positions.items():
                current_price = await self._get_current_price(symbol)
                if current_price:
                    position.current_price = current_price
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    total_value += position.market_value
            
            # Update portfolio total value
            portfolio.total_value = total_value
            
            # Calculate weights
            if total_value > 0:
                for symbol, position in portfolio.positions.items():
                    position.weight = position.market_value / total_value
                    portfolio.weights[symbol] = position.weight
            
            # Update performance history
            self.performance_history[portfolio_id].append({
                'timestamp': datetime.now(),
                'total_value': total_value,
                'positions': len(portfolio.positions),
                'cash': portfolio.available_cash
            })
            
            # Limit history size
            if len(self.performance_history[portfolio_id]) > 10000:
                self.performance_history[portfolio_id] = self.performance_history[portfolio_id][-5000:]
            
            portfolio.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update portfolio metrics for {portfolio_id}: {e}")
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            # This would integrate with actual market data feed
            # For now, return mock price
            base_prices = {
                'BTC': 45000.0,
                'ETH': 3000.0,
                'ADA': 0.5,
                'SOL': 100.0,
                'AVAX': 25.0,
                'DOT': 7.0,
                'LINK': 15.0,
                'UNI': 8.0,
                'AAVE': 150.0,
                'SUSHI': 2.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            # Add some random variation
            import random
            variation = random.uniform(-0.02, 0.02)  # ±2%
            return base_price * (1 + variation)
            
        except Exception as e:
            self.logger.error(f"Failed to get price for {symbol}: {e}")
            return None
    
    async def _get_historical_data(self, symbols: List[str], days: int) -> pd.DataFrame:
        """Get historical price data for symbols."""
        try:
            # Mock historical data generation
            import random
            
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            data = {}
            
            for symbol in symbols:
                # Generate mock price series
                base_price = await self._get_current_price(symbol)
                if base_price is None:
                    continue
                
                prices = [base_price]
                for _ in range(days - 1):
                    # Random walk with drift
                    change = random.normalvariate(0.001, 0.02)  # 0.1% drift, 2% volatility
                    new_price = prices[-1] * (1 + change)
                    prices.append(new_price)
                
                prices.reverse()  # Oldest first
                data[symbol] = prices
            
            return pd.DataFrame(data, index=dates)
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def get_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Get portfolio by ID."""
        return self.portfolios.get(portfolio_id)
    
    def list_portfolios(self) -> List[Portfolio]:
        """List all portfolios."""
        return list(self.portfolios.values())
    
    async def close_position(self, portfolio_id: str, symbol: str) -> bool:
        """Close a position in the portfolio."""
        try:
            if portfolio_id not in self.portfolios:
                return False
            
            portfolio = self.portfolios[portfolio_id]
            
            if symbol not in portfolio.positions:
                return False
            
            position = portfolio.positions[symbol]
            
            # Calculate realized PnL
            realized_pnl = position.unrealized_pnl
            
            # Update cash
            portfolio.available_cash += position.market_value
            
            # Remove position
            del portfolio.positions[symbol]
            if symbol in portfolio.weights:
                del portfolio.weights[symbol]
            if symbol in portfolio.target_weights:
                del portfolio.target_weights[symbol]
            
            # Update portfolio
            await self._update_portfolio_metrics(portfolio_id)
            
            self.logger.info(f"Closed position {symbol} in portfolio {portfolio_id} with PnL: {realized_pnl}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to close position {symbol} in portfolio {portfolio_id}: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of all portfolios."""
        return {
            'total_portfolios': len(self.portfolios),
            'active_portfolios': len([p for p in self.portfolios.values() if p.status == PortfolioStatus.ACTIVE]),
            'total_value': sum(p.total_value for p in self.portfolios.values()),
            'total_positions': sum(len(p.positions) for p in self.portfolios.values()),
            'supported_assets': len(self.portfolio_config['supported_assets'])
        }