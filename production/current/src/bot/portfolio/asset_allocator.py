"""
Advanced Asset Allocation Engine.
Provides sophisticated asset allocation strategies and optimization algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class AllocationStrategy(Enum):
    """Asset allocation strategies."""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    VOLATILITY_PARITY = "volatility_parity"
    RISK_PARITY = "risk_parity"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    ADAPTIVE = "adaptive"
    FACTOR_BASED = "factor_based"

class OptimizationMethod(Enum):
    """Optimization methods for allocation."""
    MEAN_VARIANCE = "mean_variance"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    RISK_BUDGETING = "risk_budgeting"
    CVaR_OPTIMIZATION = "cvar_optimization"

@dataclass
class AllocationResult:
    """Asset allocation result."""
    strategy: AllocationStrategy
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    concentration_index: float
    risk_contributions: Dict[str, float]
    allocation_date: datetime
    confidence_score: float
    rebalance_frequency: str
    next_rebalance_date: datetime
    allocation_rationale: str
    constraints_satisfied: bool
    optimization_details: Dict[str, Any]

class AssetAllocator:
    """Advanced asset allocation engine with multiple strategies."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Allocation configuration
        self.allocation_config = {
            'default_strategy': AllocationStrategy.RISK_PARITY,
            'rebalance_frequencies': {
                'daily': 1,
                'weekly': 7,
                'monthly': 30,
                'quarterly': 90
            },
            'lookback_periods': {
                'short': 60,   # 2 months
                'medium': 180, # 6 months
                'long': 252    # 1 year
            },
            'risk_aversion': 5.0,
            'transaction_costs': 0.001,
            'min_weight': 0.01,  # 1%
            'max_weight': 0.40,  # 40%
            'confidence_threshold': 0.6
        }
        
        # Market regimes for adaptive allocation
        self.market_regimes = {
            'bull': {'momentum_weight': 0.4, 'volatility_weight': 0.3, 'quality_weight': 0.3},
            'bear': {'momentum_weight': 0.2, 'volatility_weight': 0.5, 'quality_weight': 0.3},
            'sideways': {'momentum_weight': 0.3, 'volatility_weight': 0.3, 'quality_weight': 0.4},
            'crisis': {'momentum_weight': 0.1, 'volatility_weight': 0.6, 'quality_weight': 0.3}
        }
        
        # Allocation history
        self.allocation_history = {}
        
        self.logger.info("AssetAllocator initialized")
    
    async def allocate_assets(self, 
                            symbols: List[str],
                            strategy: AllocationStrategy,
                            price_data: pd.DataFrame,
                            market_data: Dict[str, Any] = None,
                            constraints: Dict[str, Any] = None) -> AllocationResult:
        """Perform asset allocation using specified strategy."""
        try:
            # Validate inputs
            self._validate_allocation_inputs(symbols, price_data)
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Apply allocation strategy
            if strategy == AllocationStrategy.EQUAL_WEIGHT:
                weights = await self._equal_weight_allocation(symbols)
            elif strategy == AllocationStrategy.MARKET_CAP_WEIGHT:
                weights = await self._market_cap_allocation(symbols, market_data)
            elif strategy == AllocationStrategy.VOLATILITY_PARITY:
                weights = await self._volatility_parity_allocation(returns)
            elif strategy == AllocationStrategy.RISK_PARITY:
                weights = await self._risk_parity_allocation(returns)
            elif strategy == AllocationStrategy.MOMENTUM:
                weights = await self._momentum_allocation(returns, price_data)
            elif strategy == AllocationStrategy.MEAN_REVERSION:
                weights = await self._mean_reversion_allocation(returns, price_data)
            elif strategy == AllocationStrategy.BLACK_LITTERMAN:
                weights = await self._black_litterman_allocation(returns, market_data)
            elif strategy == AllocationStrategy.HIERARCHICAL_RISK_PARITY:
                weights = await self._hierarchical_risk_parity_allocation(returns)
            elif strategy == AllocationStrategy.ADAPTIVE:
                weights = await self._adaptive_allocation(returns, market_data)
            elif strategy == AllocationStrategy.FACTOR_BASED:
                weights = await self._factor_based_allocation(returns, market_data)
            else:
                raise ValueError(f"Unsupported allocation strategy: {strategy}")
            
            # Apply constraints
            if constraints:
                weights = await self._apply_constraints(weights, constraints)
            
            # Calculate allocation metrics
            metrics = await self._calculate_allocation_metrics(weights, returns)
            
            # Create result
            result = AllocationResult(
                strategy=strategy,
                weights=weights,
                expected_return=metrics['expected_return'],
                expected_volatility=metrics['expected_volatility'],
                sharpe_ratio=metrics['sharpe_ratio'],
                diversification_ratio=metrics['diversification_ratio'],
                concentration_index=metrics['concentration_index'],
                risk_contributions=metrics['risk_contributions'],
                allocation_date=datetime.now(),
                confidence_score=metrics['confidence_score'],
                rebalance_frequency=self._determine_rebalance_frequency(strategy),
                next_rebalance_date=self._calculate_next_rebalance_date(strategy),
                allocation_rationale=self._generate_allocation_rationale(strategy, weights, metrics),
                constraints_satisfied=True,
                optimization_details=metrics.get('optimization_details', {})
            )
            
            # Store allocation history
            self._store_allocation_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Asset allocation failed for strategy {strategy}: {e}")
            raise
    
    async def _equal_weight_allocation(self, symbols: List[str]) -> Dict[str, float]:
        """Equal weight allocation."""
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}
    
    async def _market_cap_allocation(self, symbols: List[str], 
                                   market_data: Dict[str, Any]) -> Dict[str, float]:
        """Market capitalization weighted allocation."""
        try:
            # Get market caps (mock data if not provided)
            market_caps = {}
            
            if market_data and 'market_caps' in market_data:
                market_caps = market_data['market_caps']
            else:
                # Mock market caps
                mock_caps = {
                    'BTC': 850000000000,  # $850B
                    'ETH': 360000000000,  # $360B
                    'ADA': 18000000000,   # $18B
                    'SOL': 15000000000,   # $15B
                    'AVAX': 9000000000,   # $9B
                    'DOT': 8000000000,    # $8B
                    'LINK': 7000000000,   # $7B
                    'UNI': 5000000000     # $5B
                }
                
                for symbol in symbols:
                    market_caps[symbol] = mock_caps.get(symbol, 1000000000)  # Default $1B
            
            # Calculate weights
            total_cap = sum(market_caps.get(symbol, 0) for symbol in symbols)
            
            if total_cap == 0:
                return await self._equal_weight_allocation(symbols)
            
            weights = {
                symbol: market_caps.get(symbol, 0) / total_cap 
                for symbol in symbols
            }
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Market cap allocation failed: {e}")
            return await self._equal_weight_allocation(symbols)
    
    async def _volatility_parity_allocation(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Inverse volatility weighted allocation."""
        try:
            # Calculate volatilities
            volatilities = returns.std() * np.sqrt(252)  # Annualized
            
            # Inverse volatility weights
            inv_vol = 1.0 / volatilities
            total_inv_vol = inv_vol.sum()
            
            weights = (inv_vol / total_inv_vol).to_dict()
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Volatility parity allocation failed: {e}")
            return await self._equal_weight_allocation(returns.columns.tolist())
    
    async def _risk_parity_allocation(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Risk parity allocation (equal risk contribution)."""
        try:
            if not HAS_SCIPY:
                self.logger.warning("SciPy not available, falling back to volatility parity")
                return await self._volatility_parity_allocation(returns)
            
            n_assets = len(returns.columns)
            cov_matrix = returns.cov().values * 252  # Annualized
            
            # Risk parity objective function
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                if portfolio_vol == 0:
                    return 1e6
                
                # Risk contributions
                marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol
                risk_contributions = weights * marginal_risk / portfolio_vol
                
                # Target equal risk contribution
                target_risk = 1.0 / n_assets
                
                # Sum of squared deviations
                return np.sum((risk_contributions - target_risk) ** 2)
            
            # Constraints: weights sum to 1
            constraints = [{
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1.0
            }]
            
            # Bounds
            bounds = [(self.allocation_config['min_weight'], 
                      self.allocation_config['max_weight']) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
            else:
                self.logger.warning("Risk parity optimization failed, using equal weights")
                weights = np.array([1.0 / n_assets] * n_assets)
            
            return {symbol: float(weight) for symbol, weight in zip(returns.columns, weights)}
            
        except Exception as e:
            self.logger.error(f"Risk parity allocation failed: {e}")
            return await self._volatility_parity_allocation(returns)
    
    async def _momentum_allocation(self, returns: pd.DataFrame, 
                                 price_data: pd.DataFrame) -> Dict[str, float]:
        """Momentum-based allocation."""
        try:
            # Calculate momentum scores (12-1 month momentum)
            lookback_long = min(252, len(price_data))  # 12 months or available data
            lookback_short = min(21, len(price_data))   # 1 month
            
            momentum_scores = {}
            
            for symbol in returns.columns:
                if symbol in price_data.columns:
                    prices = price_data[symbol]
                    
                    # Long-term momentum
                    long_return = (prices.iloc[-1] / prices.iloc[-lookback_long] - 1) if len(prices) >= lookback_long else 0
                    
                    # Short-term momentum (to avoid)
                    short_return = (prices.iloc[-1] / prices.iloc[-lookback_short] - 1) if len(prices) >= lookback_short else 0
                    
                    # Momentum score (long-term positive, short-term neutral)
                    momentum_scores[symbol] = long_return - 0.5 * short_return
                else:
                    momentum_scores[symbol] = 0.0
            
            # Convert to weights (softmax transformation)
            scores = np.array(list(momentum_scores.values()))
            
            # Add small constant to avoid extreme weights
            scores = scores + 0.1
            
            # Softmax transformation
            exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
            softmax_weights = exp_scores / np.sum(exp_scores)
            
            # Create weights dictionary
            weights = {
                symbol: float(weight) 
                for symbol, weight in zip(momentum_scores.keys(), softmax_weights)
            }
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Momentum allocation failed: {e}")
            return await self._equal_weight_allocation(returns.columns.tolist())
    
    async def _mean_reversion_allocation(self, returns: pd.DataFrame, 
                                       price_data: pd.DataFrame) -> Dict[str, float]:
        """Mean reversion based allocation."""
        try:
            # Calculate mean reversion scores
            lookback = min(60, len(price_data))  # 2 months
            
            reversion_scores = {}
            
            for symbol in returns.columns:
                if symbol in price_data.columns:
                    prices = price_data[symbol].tail(lookback)
                    
                    # Calculate rolling mean and current position relative to mean
                    rolling_mean = prices.mean()
                    current_price = prices.iloc[-1]
                    
                    # Mean reversion score (higher when price is below mean)
                    reversion_scores[symbol] = (rolling_mean - current_price) / rolling_mean
                else:
                    reversion_scores[symbol] = 0.0
            
            # Convert to positive weights
            scores = np.array(list(reversion_scores.values()))
            
            # Shift to positive range
            min_score = np.min(scores)
            if min_score < 0:
                scores = scores - min_score + 0.1
            else:
                scores = scores + 0.1
            
            # Normalize to weights
            weights_array = scores / np.sum(scores)
            
            weights = {
                symbol: float(weight) 
                for symbol, weight in zip(reversion_scores.keys(), weights_array)
            }
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Mean reversion allocation failed: {e}")
            return await self._equal_weight_allocation(returns.columns.tolist())
    
    async def _black_litterman_allocation(self, returns: pd.DataFrame, 
                                        market_data: Dict[str, Any]) -> Dict[str, float]:
        """Black-Litterman model allocation (simplified)."""
        try:
            # This is a simplified Black-Litterman implementation
            # In practice, would need market equilibrium returns and investor views
            
            # Start with market cap weights as prior
            market_weights = await self._market_cap_allocation(
                returns.columns.tolist(), market_data
            )
            
            # Calculate implied equilibrium returns
            cov_matrix = returns.cov().values * 252
            weights_array = np.array([market_weights[col] for col in returns.columns])
            
            # Risk aversion parameter
            risk_aversion = self.allocation_config['risk_aversion']
            
            # Implied returns
            implied_returns = risk_aversion * np.dot(cov_matrix, weights_array)
            
            # For simplicity, use implied returns for optimization
            # In practice, would incorporate investor views here
            
            # Mean-variance optimization with implied returns
            def objective(w):
                portfolio_return = np.dot(w, implied_returns)
                portfolio_var = np.dot(w.T, np.dot(cov_matrix, w))
                return -(portfolio_return - 0.5 * risk_aversion * portfolio_var)
            
            if HAS_SCIPY:
                n_assets = len(returns.columns)
                
                constraints = [{
                    'type': 'eq',
                    'fun': lambda x: np.sum(x) - 1.0
                }]
                
                bounds = [(self.allocation_config['min_weight'], 
                          self.allocation_config['max_weight']) for _ in range(n_assets)]
                
                result = minimize(
                    objective,
                    weights_array,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    optimal_weights = result.x
                else:
                    optimal_weights = weights_array
            else:
                optimal_weights = weights_array
            
            weights = {
                symbol: float(weight) 
                for symbol, weight in zip(returns.columns, optimal_weights)
            }
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Black-Litterman allocation failed: {e}")
            return await self._market_cap_allocation(returns.columns.tolist(), market_data)
    
    async def _hierarchical_risk_parity_allocation(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Hierarchical Risk Parity allocation (simplified)."""
        try:
            # This is a simplified HRP implementation
            # Full implementation would use hierarchical clustering
            
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            # For simplicity, use risk parity with correlation-based adjustments
            base_weights = await self._risk_parity_allocation(returns)
            
            # Adjust weights based on correlations
            # Assets with lower average correlation get higher weights
            avg_correlations = {}
            
            for symbol in returns.columns:
                correlations = corr_matrix[symbol].drop(symbol)
                avg_correlations[symbol] = correlations.abs().mean()
            
            # Inverse correlation adjustment
            inv_corr = {symbol: 1.0 / (1.0 + corr) for symbol, corr in avg_correlations.items()}
            total_inv_corr = sum(inv_corr.values())
            
            # Combine with base weights
            adjusted_weights = {}
            for symbol in returns.columns:
                base_weight = base_weights[symbol]
                corr_adjustment = inv_corr[symbol] / total_inv_corr
                adjusted_weights[symbol] = 0.7 * base_weight + 0.3 * corr_adjustment
            
            # Normalize
            total_weight = sum(adjusted_weights.values())
            weights = {symbol: weight / total_weight for symbol, weight in adjusted_weights.items()}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Hierarchical risk parity allocation failed: {e}")
            return await self._risk_parity_allocation(returns)
    
    async def _adaptive_allocation(self, returns: pd.DataFrame, 
                                 market_data: Dict[str, Any]) -> Dict[str, float]:
        """Adaptive allocation based on market regime."""
        try:
            # Determine market regime (simplified)
            recent_returns = returns.tail(30)  # Last 30 days
            
            avg_return = recent_returns.mean().mean()
            avg_volatility = recent_returns.std().mean()
            
            # Simple regime classification
            if avg_return > 0.002 and avg_volatility < 0.03:
                regime = 'bull'
            elif avg_return < -0.001 and avg_volatility > 0.04:
                regime = 'bear'
            elif avg_volatility > 0.06:
                regime = 'crisis'
            else:
                regime = 'sideways'
            
            regime_params = self.market_regimes.get(regime, self.market_regimes['sideways'])
            
            # Combine different strategies based on regime
            momentum_weights = await self._momentum_allocation(returns, market_data.get('price_data', pd.DataFrame()))
            risk_parity_weights = await self._risk_parity_allocation(returns)
            equal_weights = await self._equal_weight_allocation(returns.columns.tolist())
            
            # Combine strategies
            adaptive_weights = {}
            for symbol in returns.columns:
                momentum_weight = momentum_weights.get(symbol, 0)
                risk_weight = risk_parity_weights.get(symbol, 0)
                equal_weight = equal_weights.get(symbol, 0)
                
                adaptive_weights[symbol] = (
                    regime_params['momentum_weight'] * momentum_weight +
                    regime_params['volatility_weight'] * risk_weight +
                    regime_params['quality_weight'] * equal_weight
                )
            
            # Normalize
            total_weight = sum(adaptive_weights.values())
            weights = {symbol: weight / total_weight for symbol, weight in adaptive_weights.items()}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Adaptive allocation failed: {e}")
            return await self._risk_parity_allocation(returns)
    
    async def _factor_based_allocation(self, returns: pd.DataFrame, 
                                     market_data: Dict[str, Any]) -> Dict[str, float]:
        """Factor-based allocation using asset characteristics."""
        try:
            # Mock factor scores (in practice, would come from fundamental data)
            factor_scores = {}
            
            for symbol in returns.columns:
                # Calculate simple factors from returns data
                symbol_returns = returns[symbol]
                
                # Momentum factor
                momentum = symbol_returns.tail(60).mean() * 252  # Annualized recent return
                
                # Low volatility factor
                volatility = symbol_returns.std() * np.sqrt(252)
                low_vol_score = 1.0 / (1.0 + volatility)  # Higher score for lower volatility
                
                # Quality factor (Sharpe ratio proxy)
                sharpe = momentum / volatility if volatility > 0 else 0
                
                # Combine factors
                factor_scores[symbol] = {
                    'momentum': momentum,
                    'low_volatility': low_vol_score,
                    'quality': sharpe
                }
            
            # Factor weights (can be adjusted based on market conditions)
            factor_weights = {
                'momentum': 0.4,
                'low_volatility': 0.3,
                'quality': 0.3
            }
            
            # Calculate composite scores
            composite_scores = {}
            for symbol in returns.columns:
                score = 0
                for factor, weight in factor_weights.items():
                    score += weight * factor_scores[symbol][factor]
                composite_scores[symbol] = score
            
            # Convert to allocation weights (rank-based)
            sorted_symbols = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Exponential decay weights
            weights = {}
            total_weight = 0
            
            for i, (symbol, score) in enumerate(sorted_symbols):
                weight = np.exp(-0.2 * i)  # Decay factor
                weights[symbol] = weight
                total_weight += weight
            
            # Normalize
            weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Factor-based allocation failed: {e}")
            return await self._equal_weight_allocation(returns.columns.tolist())
    
    async def _apply_constraints(self, weights: Dict[str, float], 
                               constraints: Dict[str, Any]) -> Dict[str, float]:
        """Apply allocation constraints."""
        try:
            adjusted_weights = weights.copy()
            
            # Min/max weight constraints
            min_weight = constraints.get('min_weight', self.allocation_config['min_weight'])
            max_weight = constraints.get('max_weight', self.allocation_config['max_weight'])
            
            # Apply bounds
            for symbol in adjusted_weights:
                adjusted_weights[symbol] = max(min_weight, min(max_weight, adjusted_weights[symbol]))
            
            # Renormalize
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {symbol: weight / total_weight 
                                  for symbol, weight in adjusted_weights.items()}
            
            return adjusted_weights
            
        except Exception as e:
            self.logger.error(f"Failed to apply constraints: {e}")
            return weights
    
    async def _calculate_allocation_metrics(self, weights: Dict[str, float], 
                                          returns: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for allocation result."""
        try:
            weights_array = np.array([weights[col] for col in returns.columns])
            
            # Expected return
            mean_returns = returns.mean() * 252  # Annualized
            expected_return = np.dot(weights_array, mean_returns.values)
            
            # Expected volatility
            cov_matrix = returns.cov() * 252  # Annualized
            expected_variance = np.dot(weights_array.T, np.dot(cov_matrix.values, weights_array))
            expected_volatility = np.sqrt(expected_variance)
            
            # Sharpe ratio
            risk_free_rate = self.config.get('risk_free_rate', 0.02)
            sharpe_ratio = (expected_return - risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
            
            # Diversification ratio
            individual_vols = returns.std() * np.sqrt(252)
            weighted_avg_vol = np.dot(weights_array, individual_vols.values)
            diversification_ratio = weighted_avg_vol / expected_volatility if expected_volatility > 0 else 1.0
            
            # Concentration index (Herfindahl-Hirschman Index)
            concentration_index = np.sum(weights_array ** 2)
            
            # Risk contributions
            if expected_volatility > 0:
                marginal_risk = np.dot(cov_matrix.values, weights_array) / expected_volatility
                risk_contributions = {
                    symbol: float(weights[symbol] * marginal_risk[i] / expected_volatility)
                    for i, symbol in enumerate(returns.columns)
                }
            else:
                risk_contributions = {symbol: 0.0 for symbol in returns.columns}
            
            # Confidence score (based on historical stability)
            confidence_score = self._calculate_confidence_score(weights, returns)
            
            return {
                'expected_return': float(expected_return),
                'expected_volatility': float(expected_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'diversification_ratio': float(diversification_ratio),
                'concentration_index': float(concentration_index),
                'risk_contributions': risk_contributions,
                'confidence_score': confidence_score
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate allocation metrics: {e}")
            return {
                'expected_return': 0.0,
                'expected_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'diversification_ratio': 1.0,
                'concentration_index': 1.0,
                'risk_contributions': {},
                'confidence_score': 0.5
            }
    
    def _calculate_confidence_score(self, weights: Dict[str, float], 
                                  returns: pd.DataFrame) -> float:
        """Calculate confidence score for allocation."""
        try:
            # Simple confidence based on data quality and weight stability
            
            # Data quality score
            data_completeness = 1.0 - returns.isnull().sum().sum() / (len(returns) * len(returns.columns))
            
            # Weight concentration score (penalize extreme concentrations)
            weight_values = np.array(list(weights.values()))
            weight_entropy = -np.sum(weight_values * np.log(weight_values + 1e-10))
            max_entropy = np.log(len(weight_values))
            diversification_score = weight_entropy / max_entropy if max_entropy > 0 else 0
            
            # Combine scores
            confidence = 0.5 * data_completeness + 0.5 * diversification_score
            
            return float(np.clip(confidence, 0, 1))
            
        except:
            return 0.5  # Default moderate confidence
    
    def _determine_rebalance_frequency(self, strategy: AllocationStrategy) -> str:
        """Determine optimal rebalance frequency for strategy."""
        frequency_map = {
            AllocationStrategy.EQUAL_WEIGHT: 'monthly',
            AllocationStrategy.MARKET_CAP_WEIGHT: 'monthly',
            AllocationStrategy.VOLATILITY_PARITY: 'weekly',
            AllocationStrategy.RISK_PARITY: 'monthly',
            AllocationStrategy.MOMENTUM: 'monthly',
            AllocationStrategy.MEAN_REVERSION: 'weekly',
            AllocationStrategy.BLACK_LITTERMAN: 'quarterly',
            AllocationStrategy.HIERARCHICAL_RISK_PARITY: 'monthly',
            AllocationStrategy.ADAPTIVE: 'weekly',
            AllocationStrategy.FACTOR_BASED: 'monthly'
        }
        
        return frequency_map.get(strategy, 'monthly')
    
    def _calculate_next_rebalance_date(self, strategy: AllocationStrategy) -> datetime:
        """Calculate next rebalance date."""
        frequency = self._determine_rebalance_frequency(strategy)
        days_to_add = self.allocation_config['rebalance_frequencies'][frequency]
        return datetime.now() + timedelta(days=days_to_add)
    
    def _generate_allocation_rationale(self, strategy: AllocationStrategy, 
                                     weights: Dict[str, float],
                                     metrics: Dict[str, Any]) -> str:
        """Generate rationale for allocation decision."""
        top_positions = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        top_symbols = [pos[0] for pos in top_positions]
        
        rationale_templates = {
            AllocationStrategy.EQUAL_WEIGHT: f"Equal weight allocation across all assets for maximum diversification. Top positions: {', '.join(top_symbols)}.",
            AllocationStrategy.RISK_PARITY: f"Risk parity allocation targeting equal risk contribution. Expected Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}.",
            AllocationStrategy.MOMENTUM: f"Momentum-based allocation favoring recent outperformers: {', '.join(top_symbols)}.",
            AllocationStrategy.MEAN_REVERSION: f"Mean reversion strategy targeting oversold assets. Diversification ratio: {metrics.get('diversification_ratio', 1):.2f}.",
            AllocationStrategy.ADAPTIVE: f"Adaptive allocation based on current market regime. Top holdings: {', '.join(top_symbols)}."
        }
        
        return rationale_templates.get(
            strategy, 
            f"Custom allocation strategy with {len(weights)} assets. Expected return: {metrics.get('expected_return', 0)*100:.1f}%."
        )
    
    def _validate_allocation_inputs(self, symbols: List[str], price_data: pd.DataFrame):
        """Validate inputs for allocation."""
        if not symbols:
            raise ValueError("No symbols provided for allocation")
        
        if price_data.empty:
            raise ValueError("No price data provided")
        
        if len(price_data) < 30:
            raise ValueError("Insufficient price data (minimum 30 days required)")
        
        missing_symbols = [s for s in symbols if s not in price_data.columns]
        if missing_symbols:
            raise ValueError(f"Missing price data for symbols: {missing_symbols}")
    
    def _store_allocation_result(self, result: AllocationResult):
        """Store allocation result in history."""
        strategy_key = result.strategy.value
        
        if strategy_key not in self.allocation_history:
            self.allocation_history[strategy_key] = []
        
        self.allocation_history[strategy_key].append(result)
        
        # Limit history size
        if len(self.allocation_history[strategy_key]) > 100:
            self.allocation_history[strategy_key] = self.allocation_history[strategy_key][-50:]
    
    def get_allocation_history(self, strategy: AllocationStrategy = None) -> List[AllocationResult]:
        """Get allocation history for strategy."""
        if strategy:
            return self.allocation_history.get(strategy.value, [])
        else:
            all_results = []
            for results in self.allocation_history.values():
                all_results.extend(results)
            return sorted(all_results, key=lambda x: x.allocation_date, reverse=True)
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of allocation engine."""
        total_allocations = sum(len(results) for results in self.allocation_history.values())
        
        return {
            'total_allocations': total_allocations,
            'strategies_used': list(self.allocation_history.keys()),
            'supported_strategies': [s.value for s in AllocationStrategy],
            'default_strategy': self.allocation_config['default_strategy'].value,
            'last_allocation': max([
                max(results, key=lambda x: x.allocation_date).allocation_date
                for results in self.allocation_history.values()
            ]) if self.allocation_history else None
        }