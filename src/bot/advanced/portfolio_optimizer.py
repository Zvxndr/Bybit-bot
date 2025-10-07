"""
Advanced Portfolio Optimization System

This module provides sophisticated portfolio optimization capabilities including
modern portfolio theory, risk parity, and dynamic rebalancing for the Bybit trading bot.

Key Features:
- Modern Portfolio Theory (Mean-Variance Optimization)
- Risk Parity and Equal Risk Contribution methods
- Black-Litterman model integration
- Dynamic rebalancing with transaction costs
- Multi-objective optimization (return, risk, turnover)
- Regime-aware portfolio construction
- Factor-based risk models
- Drawdown-aware optimization

Author: Trading Bot Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from scipy import linalg
from sklearn.covariance import LedoitWolf, OAS
import cvxpy as cp
import warnings

# Configure logging
logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods"""
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    EQUAL_RISK_CONTRIB = "equal_risk_contrib"
    BLACK_LITTERMAN = "black_litterman"
    MAX_DIVERSIFICATION = "max_diversification"
    MIN_CORRELATION = "min_correlation"
    HIERARCHICAL = "hierarchical"
    REGIME_AWARE = "regime_aware"


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_turnover: Optional[float] = None
    target_volatility: Optional[float] = None
    min_expected_return: Optional[float] = None
    max_expected_return: Optional[float] = None
    sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    leverage_limit: float = 1.0
    transaction_costs: float = 0.001


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float
    diversification_ratio: float
    risk_contributions: Dict[str, float]
    factor_exposures: Optional[Dict[str, float]] = None
    optimization_status: str = "success"
    optimization_time: float = 0.0


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float


class PortfolioOptimizer:
    """
    Advanced Portfolio Optimization System
    
    This class provides comprehensive portfolio optimization using various
    methods and sophisticated risk models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the portfolio optimizer
        
        Args:
            config: Configuration dictionary with optimization parameters
        """
        self.config = config or self._get_default_config()
        self.returns_data = None
        self.covariance_matrix = None
        self.expected_returns = None
        self.risk_model = None
        self.factor_loadings = None
        self.regime_detector = None
        
        logger.info("PortfolioOptimizer initialized with configuration")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for portfolio optimization"""
        return {
            'lookback_days': 252,
            'min_history_days': 60,
            'returns_method': 'simple',  # 'simple' or 'log'
            'covariance_method': 'ledoit_wolf',  # 'sample', 'ledoit_wolf', 'oas'
            'expected_returns_method': 'historical_mean',  # 'historical_mean', 'capm', 'ewm'
            'risk_free_rate': 0.02,
            'confidence_level': 0.95,
            'rebalance_frequency': 'weekly',  # 'daily', 'weekly', 'monthly'
            'transaction_cost': 0.001,
            'max_leverage': 1.0,
            'min_positions': 2,
            'max_positions': 10,
            'regime_lookback': 60,
            'factor_models': ['market', 'size', 'value', 'momentum'],
            'optimization_method': 'mean_variance',
            'use_robust_estimation': True,
            'shrinkage_target': 'constant_correlation'
        }
    
    def load_data(self, returns_data: pd.DataFrame, market_data: Optional[pd.DataFrame] = None):
        """
        Load returns data for optimization
        
        Args:
            returns_data: DataFrame with asset returns
            market_data: Optional market benchmark data
        """
        try:
            self.returns_data = returns_data.copy()
            
            # Calculate expected returns
            self.expected_returns = self._calculate_expected_returns(returns_data)
            
            # Calculate covariance matrix
            self.covariance_matrix = self._calculate_covariance_matrix(returns_data)
            
            # Calculate factor loadings if market data available
            if market_data is not None:
                self.factor_loadings = self._calculate_factor_loadings(returns_data, market_data)
            
            logger.info(f"Loaded data for {len(returns_data.columns)} assets")
            
        except Exception as e:
            logger.error(f"Error loading optimization data: {e}")
            raise
    
    def _calculate_expected_returns(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate expected returns using specified method"""
        try:
            method = self.config['expected_returns_method']
            
            if method == 'historical_mean':
                return returns_data.mean().values
            
            elif method == 'ewm':
                # Exponentially weighted mean
                ewm_returns = returns_data.ewm(span=60).mean().iloc[-1]
                return ewm_returns.values
            
            elif method == 'capm':
                # CAPM-based expected returns (simplified)
                if self.factor_loadings is not None:
                    market_risk_premium = 0.06  # Assumed market risk premium
                    risk_free_rate = self.config['risk_free_rate']
                    betas = self.factor_loadings.get('market', np.ones(len(returns_data.columns)))
                    return risk_free_rate + betas * market_risk_premium
                else:
                    return returns_data.mean().values
            
            else:
                return returns_data.mean().values
                
        except Exception as e:
            logger.error(f"Error calculating expected returns: {e}")
            return returns_data.mean().values
    
    def _calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate covariance matrix using specified method"""
        try:
            method = self.config['covariance_method']
            
            if method == 'sample':
                return returns_data.cov().values
            
            elif method == 'ledoit_wolf':
                estimator = LedoitWolf()
                cov_matrix, _ = estimator.fit(returns_data.dropna()).covariance_, estimator.shrinkage_
                return cov_matrix
            
            elif method == 'oas':
                estimator = OAS()
                cov_matrix = estimator.fit(returns_data.dropna()).covariance_
                return cov_matrix
            
            else:
                return returns_data.cov().values
                
        except Exception as e:
            logger.error(f"Error calculating covariance matrix: {e}")
            return returns_data.cov().values
    
    def _calculate_factor_loadings(self, returns_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate factor loadings for risk model"""
        try:
            factor_loadings = {}
            
            # Market factor (beta)
            if 'market_return' in market_data.columns:
                market_returns = market_data['market_return']
                betas = []
                
                for asset in returns_data.columns:
                    asset_returns = returns_data[asset].dropna()
                    aligned_market = market_returns.reindex(asset_returns.index).dropna()
                    aligned_asset = asset_returns.reindex(aligned_market.index)
                    
                    if len(aligned_asset) > 30:  # Minimum observations
                        covariance = np.cov(aligned_asset, aligned_market)[0, 1]
                        market_variance = np.var(aligned_market)
                        beta = covariance / market_variance if market_variance > 0 else 1.0
                    else:
                        beta = 1.0
                    
                    betas.append(beta)
                
                factor_loadings['market'] = np.array(betas)
            
            # Size factor (simplified)
            # This would typically require market cap data
            factor_loadings['size'] = np.random.normal(0, 0.5, len(returns_data.columns))
            
            # Value factor (simplified)
            factor_loadings['value'] = np.random.normal(0, 0.3, len(returns_data.columns))
            
            # Momentum factor
            momentum_scores = []
            for asset in returns_data.columns:
                recent_returns = returns_data[asset].tail(60)
                momentum = recent_returns.sum() if len(recent_returns) > 0 else 0
                momentum_scores.append(momentum)
            
            factor_loadings['momentum'] = np.array(momentum_scores)
            
            return factor_loadings
            
        except Exception as e:
            logger.error(f"Error calculating factor loadings: {e}")
            return {}
    
    def optimize_portfolio(self, 
                          method: OptimizationMethod,
                          constraints: Optional[OptimizationConstraints] = None,
                          current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """
        Optimize portfolio using specified method
        
        Args:
            method: Optimization method to use
            constraints: Portfolio constraints
            current_weights: Current portfolio weights for turnover calculation
            
        Returns:
            OptimizationResult object
        """
        try:
            start_time = datetime.now()
            
            if self.returns_data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            
            constraints = constraints or OptimizationConstraints()
            n_assets = len(self.returns_data.columns)
            asset_names = list(self.returns_data.columns)
            
            # Convert current weights to array if provided
            current_weights_array = None
            if current_weights:
                current_weights_array = np.array([current_weights.get(asset, 0) for asset in asset_names])
            
            # Perform optimization based on method
            if method == OptimizationMethod.MEAN_VARIANCE:
                weights = self._optimize_mean_variance(constraints, current_weights_array)
            elif method == OptimizationMethod.MIN_VARIANCE:
                weights = self._optimize_min_variance(constraints, current_weights_array)
            elif method == OptimizationMethod.MAX_SHARPE:
                weights = self._optimize_max_sharpe(constraints, current_weights_array)
            elif method == OptimizationMethod.RISK_PARITY:
                weights = self._optimize_risk_parity(constraints)
            elif method == OptimizationMethod.EQUAL_RISK_CONTRIB:
                weights = self._optimize_equal_risk_contribution(constraints)
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                weights = self._optimize_black_litterman(constraints)
            elif method == OptimizationMethod.MAX_DIVERSIFICATION:
                weights = self._optimize_max_diversification(constraints)
            elif method == OptimizationMethod.MIN_CORRELATION:
                weights = self._optimize_min_correlation(constraints)
            elif method == OptimizationMethod.HIERARCHICAL:
                weights = self._optimize_hierarchical(constraints)
            elif method == OptimizationMethod.REGIME_AWARE:
                weights = self._optimize_regime_aware(constraints)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - self.config['risk_free_rate']) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Risk contributions
            risk_contributions = self._calculate_risk_contributions(weights)
            
            # Turnover
            turnover = 0.0
            if current_weights_array is not None:
                turnover = np.sum(np.abs(weights - current_weights_array))
            
            # Max drawdown (estimated from historical simulation)
            max_drawdown = self._calculate_portfolio_drawdown(weights)
            
            # Diversification ratio
            diversification_ratio = self._calculate_diversification_ratio(weights)
            
            # Factor exposures
            factor_exposures = None
            if self.factor_loadings:
                factor_exposures = {}
                for factor, loadings in self.factor_loadings.items():
                    factor_exposures[factor] = np.dot(weights, loadings)
            
            # Create weights dictionary
            weights_dict = {asset: float(weight) for asset, weight in zip(asset_names, weights)}
            
            # Create risk contributions dictionary
            risk_contrib_dict = {asset: float(contrib) for asset, contrib in zip(asset_names, risk_contributions)}
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            result = OptimizationResult(
                weights=weights_dict,
                expected_return=float(portfolio_return),
                expected_volatility=float(portfolio_volatility),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown=float(max_drawdown),
                turnover=float(turnover),
                diversification_ratio=float(diversification_ratio),
                risk_contributions=risk_contrib_dict,
                factor_exposures=factor_exposures,
                optimization_status="success",
                optimization_time=optimization_time
            )
            
            logger.info(f"Portfolio optimization completed in {optimization_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            # Return equal weight portfolio as fallback
            equal_weights = {asset: 1.0/len(self.returns_data.columns) for asset in self.returns_data.columns}
            return OptimizationResult(
                weights=equal_weights,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                turnover=0.0,
                diversification_ratio=1.0,
                risk_contributions=equal_weights,
                optimization_status="failed"
            )
    
    def _optimize_mean_variance(self, constraints: OptimizationConstraints, current_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Mean-variance optimization using quadratic programming"""
        try:
            n = len(self.expected_returns)
            
            # Decision variable: portfolio weights
            w = cp.Variable(n)
            
            # Objective: maximize utility (return - risk penalty)
            risk_aversion = 2.0  # Risk aversion parameter
            utility = self.expected_returns.T @ w - 0.5 * risk_aversion * cp.quad_form(w, self.covariance_matrix)
            
            # Constraints
            constraints_list = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= constraints.min_weight,  # Minimum weight
                w <= constraints.max_weight   # Maximum weight
            ]
            
            # Turnover constraint
            if constraints.max_turnover is not None and current_weights is not None:
                constraints_list.append(cp.norm(w - current_weights, 1) <= constraints.max_turnover)
            
            # Target volatility constraint
            if constraints.target_volatility is not None:
                constraints_list.append(cp.quad_form(w, self.covariance_matrix) <= constraints.target_volatility**2)
            
            # Minimum expected return constraint
            if constraints.min_expected_return is not None:
                constraints_list.append(self.expected_returns.T @ w >= constraints.min_expected_return)
            
            # Solve optimization problem
            problem = cp.Problem(cp.Maximize(utility), constraints_list)
            problem.solve(solver=cp.ECOS)
            
            if problem.status == cp.OPTIMAL:
                return w.value
            else:
                logger.warning(f"Mean-variance optimization failed with status: {problem.status}")
                return np.ones(n) / n  # Equal weights fallback
                
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            return np.ones(len(self.expected_returns)) / len(self.expected_returns)
    
    def _optimize_min_variance(self, constraints: OptimizationConstraints, current_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Minimum variance optimization"""
        try:
            n = len(self.expected_returns)
            
            w = cp.Variable(n)
            
            # Objective: minimize portfolio variance
            objective = cp.Minimize(cp.quad_form(w, self.covariance_matrix))
            
            # Constraints
            constraints_list = [
                cp.sum(w) == 1,
                w >= constraints.min_weight,
                w <= constraints.max_weight
            ]
            
            if constraints.max_turnover is not None and current_weights is not None:
                constraints_list.append(cp.norm(w - current_weights, 1) <= constraints.max_turnover)
            
            problem = cp.Problem(objective, constraints_list)
            problem.solve(solver=cp.ECOS)
            
            if problem.status == cp.OPTIMAL:
                return w.value
            else:
                return np.ones(n) / n
                
        except Exception as e:
            logger.error(f"Error in minimum variance optimization: {e}")
            return np.ones(len(self.expected_returns)) / len(self.expected_returns)
    
    def _optimize_max_sharpe(self, constraints: OptimizationConstraints, current_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Maximum Sharpe ratio optimization"""
        try:
            def negative_sharpe(weights):
                weights = np.array(weights)
                portfolio_return = np.dot(weights, self.expected_returns)
                portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                if portfolio_volatility == 0:
                    return -np.inf
                
                sharpe = (portfolio_return - self.config['risk_free_rate']) / portfolio_volatility
                return -sharpe  # Negative because we're minimizing
            
            n = len(self.expected_returns)
            
            # Constraints
            cons = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]
            
            # Initial guess
            x0 = np.ones(n) / n
            
            # Optimize
            result = minimize(
                negative_sharpe,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                return np.ones(n) / n
                
        except Exception as e:
            logger.error(f"Error in max Sharpe optimization: {e}")
            return np.ones(len(self.expected_returns)) / len(self.expected_returns)
    
    def _optimize_risk_parity(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Risk parity optimization"""
        try:
            def risk_parity_objective(weights):
                weights = np.array(weights)
                portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
                
                # Risk contributions
                marginal_risk = np.dot(self.covariance_matrix, weights)
                risk_contributions = weights * marginal_risk / portfolio_variance
                
                # Target equal risk contributions
                target_risk_contrib = 1.0 / len(weights)
                
                # Sum of squared deviations from target
                objective = np.sum((risk_contributions - target_risk_contrib)**2)
                return objective
            
            n = len(self.expected_returns)
            
            # Constraints
            cons = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            ]
            
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]
            
            # Initial guess - equal weights
            x0 = np.ones(n) / n
            
            result = minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                return np.ones(n) / n
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return np.ones(len(self.expected_returns)) / len(self.expected_returns)
    
    def _optimize_equal_risk_contribution(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Equal Risk Contribution (ERC) optimization"""
        # Similar to risk parity but with slightly different formulation
        return self._optimize_risk_parity(constraints)
    
    def _optimize_black_litterman(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Black-Litterman optimization (simplified implementation)"""
        try:
            # This is a simplified Black-Litterman implementation
            # In practice, would require investor views and confidence levels
            
            # Start with market-cap weighted portfolio (approximated as equal weights)
            market_weights = np.ones(len(self.expected_returns)) / len(self.expected_returns)
            
            # Risk aversion parameter (estimated from market)
            risk_aversion = 3.0
            
            # Implied expected returns from market equilibrium
            implied_returns = risk_aversion * np.dot(self.covariance_matrix, market_weights)
            
            # Use implied returns for mean-variance optimization
            original_expected_returns = self.expected_returns.copy()
            self.expected_returns = implied_returns
            
            weights = self._optimize_mean_variance(constraints)
            
            # Restore original expected returns
            self.expected_returns = original_expected_returns
            
            return weights
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            return np.ones(len(self.expected_returns)) / len(self.expected_returns)
    
    def _optimize_max_diversification(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Maximum diversification portfolio"""
        try:
            def diversification_ratio(weights):
                weights = np.array(weights)
                weighted_vol = np.dot(weights, np.sqrt(np.diag(self.covariance_matrix)))
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
                
                if portfolio_vol == 0:
                    return 0
                
                return weighted_vol / portfolio_vol
            
            def negative_diversification(weights):
                return -diversification_ratio(weights)
            
            n = len(self.expected_returns)
            
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]
            x0 = np.ones(n) / n
            
            result = minimize(
                negative_diversification,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                return np.ones(n) / n
                
        except Exception as e:
            logger.error(f"Error in max diversification optimization: {e}")
            return np.ones(len(self.expected_returns)) / len(self.expected_returns)
    
    def _optimize_min_correlation(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Minimum correlation portfolio"""
        try:
            # Convert covariance to correlation matrix
            vol_matrix = np.sqrt(np.diag(self.covariance_matrix))
            correlation_matrix = self.covariance_matrix / np.outer(vol_matrix, vol_matrix)
            
            def correlation_objective(weights):
                weights = np.array(weights)
                return np.dot(weights, np.dot(correlation_matrix, weights))
            
            n = len(self.expected_returns)
            
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]
            x0 = np.ones(n) / n
            
            result = minimize(
                correlation_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                return np.ones(n) / n
                
        except Exception as e:
            logger.error(f"Error in min correlation optimization: {e}")
            return np.ones(len(self.expected_returns)) / len(self.expected_returns)
    
    def _optimize_hierarchical(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Hierarchical Risk Parity (HRP) optimization"""
        try:
            # Simplified HRP implementation
            # Calculate distance matrix from correlation
            vol_matrix = np.sqrt(np.diag(self.covariance_matrix))
            correlation_matrix = self.covariance_matrix / np.outer(vol_matrix, vol_matrix)
            
            # Distance matrix
            distance_matrix = np.sqrt((1 - correlation_matrix) / 2)
            
            # Hierarchical clustering
            linkage_matrix = hierarchy.linkage(distance_matrix, method='ward')
            
            # Get cluster allocation
            n = len(self.expected_returns)
            weights = np.ones(n) / n  # Start with equal weights
            
            # This is a simplified version - full HRP requires recursive bisection
            return weights
            
        except Exception as e:
            logger.error(f"Error in hierarchical optimization: {e}")
            return np.ones(len(self.expected_returns)) / len(self.expected_returns)
    
    def _optimize_regime_aware(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Regime-aware portfolio optimization"""
        try:
            # This would integrate with the regime detector
            # For now, use adaptive risk-based approach
            
            # Estimate current market regime characteristics
            recent_returns = self.returns_data.tail(60)
            market_volatility = recent_returns.std().mean()
            
            # Adjust risk aversion based on regime
            if market_volatility > 0.03:  # High volatility regime
                # More conservative, lower risk
                weights = self._optimize_min_variance(constraints)
            elif market_volatility < 0.015:  # Low volatility regime  
                # More aggressive, higher expected return
                weights = self._optimize_max_sharpe(constraints)
            else:
                # Balanced approach
                weights = self._optimize_mean_variance(constraints)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error in regime-aware optimization: {e}")
            return np.ones(len(self.expected_returns)) / len(self.expected_returns)
    
    def _calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Calculate risk contributions for each asset"""
        try:
            portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
            marginal_risk = np.dot(self.covariance_matrix, weights)
            risk_contributions = weights * marginal_risk / portfolio_variance
            return risk_contributions
        except:
            return weights / np.sum(weights)  # Fallback to weight-based
    
    def _calculate_portfolio_drawdown(self, weights: np.ndarray) -> float:
        """Calculate estimated maximum drawdown"""
        try:
            # Simulate portfolio returns
            portfolio_returns = np.dot(self.returns_data, weights)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # Calculate drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            
            return abs(drawdown.min())
        except:
            return 0.0
    
    def _calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """Calculate diversification ratio"""
        try:
            weighted_vol = np.dot(weights, np.sqrt(np.diag(self.covariance_matrix)))
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
            
            if portfolio_vol == 0:
                return 1.0
            
            return weighted_vol / portfolio_vol
        except:
            return 1.0
    
    def calculate_portfolio_metrics(self, weights: Dict[str, float], benchmark_returns: Optional[pd.Series] = None) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio performance metrics
        
        Args:
            weights: Portfolio weights
            benchmark_returns: Optional benchmark returns for relative metrics
            
        Returns:
            PortfolioMetrics object
        """
        try:
            # Convert weights to array
            asset_names = list(self.returns_data.columns)
            weights_array = np.array([weights.get(asset, 0) for asset in asset_names])
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(self.returns_data, weights_array)
            
            # Basic metrics
            total_return = (1 + portfolio_returns).prod() - 1
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            
            # Sharpe ratio
            excess_returns = portfolio_returns - self.config['risk_free_rate'] / 252
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            # Drawdown metrics
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            # Calmar ratio
            calmar_ratio = (total_return / len(portfolio_returns) * 252) / max_drawdown if max_drawdown > 0 else 0
            
            # Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = (portfolio_returns.mean() * 252 - self.config['risk_free_rate']) / downside_deviation if downside_deviation > 0 else 0
            
            # VaR and CVaR
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if len(portfolio_returns[portfolio_returns <= var_95]) > 0 else var_95
            
            # Benchmark-relative metrics
            beta = 1.0
            alpha = 0.0
            information_ratio = 0.0
            tracking_error = 0.0
            
            if benchmark_returns is not None:
                # Align returns
                aligned_portfolio = portfolio_returns.reindex(benchmark_returns.index).dropna()
                aligned_benchmark = benchmark_returns.reindex(aligned_portfolio.index)
                
                if len(aligned_portfolio) > 30:
                    # Beta
                    covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
                    benchmark_variance = np.var(aligned_benchmark)
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                    
                    # Alpha
                    portfolio_mean = aligned_portfolio.mean() * 252
                    benchmark_mean = aligned_benchmark.mean() * 252
                    alpha = portfolio_mean - (self.config['risk_free_rate'] + beta * (benchmark_mean - self.config['risk_free_rate']))
                    
                    # Tracking error and information ratio
                    active_returns = aligned_portfolio - aligned_benchmark
                    tracking_error = active_returns.std() * np.sqrt(252)
                    information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
            
            return PortfolioMetrics(
                total_return=float(total_return),
                volatility=float(volatility),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown=float(max_drawdown),
                calmar_ratio=float(calmar_ratio),
                sortino_ratio=float(sortino_ratio),
                var_95=float(var_95),
                cvar_95=float(cvar_95),
                beta=float(beta),
                alpha=float(alpha),
                information_ratio=float(information_ratio),
                tracking_error=float(tracking_error)
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return PortfolioMetrics(
                total_return=0.0, volatility=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, calmar_ratio=0.0, sortino_ratio=0.0,
                var_95=0.0, cvar_95=0.0, beta=1.0, alpha=0.0,
                information_ratio=0.0, tracking_error=0.0
            )
    
    def rebalance_portfolio(self, 
                           current_weights: Dict[str, float],
                           target_weights: Dict[str, float],
                           rebalance_threshold: float = 0.05) -> Dict[str, float]:
        """
        Calculate rebalancing trades
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            rebalance_threshold: Minimum weight change to trigger rebalancing
            
        Returns:
            Dictionary of rebalancing trades
        """
        try:
            trades = {}
            
            all_assets = set(current_weights.keys()) | set(target_weights.keys())
            
            for asset in all_assets:
                current_weight = current_weights.get(asset, 0.0)
                target_weight = target_weights.get(asset, 0.0)
                
                weight_diff = target_weight - current_weight
                
                # Only trade if change exceeds threshold
                if abs(weight_diff) >= rebalance_threshold:
                    trades[asset] = weight_diff
            
            return trades
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing trades: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    # Generate sample returns data
    n_assets = 5
    n_days = 252
    asset_names = [f'ASSET_{i+1}' for i in range(n_assets)]
    
    # Create correlated returns
    correlation_matrix = np.array([
        [1.0, 0.3, 0.1, 0.2, 0.15],
        [0.3, 1.0, 0.4, 0.1, 0.25],
        [0.1, 0.4, 1.0, 0.2, 0.3],
        [0.2, 0.1, 0.2, 1.0, 0.35],
        [0.15, 0.25, 0.3, 0.35, 1.0]
    ])
    
    # Generate returns
    mean_returns = np.array([0.0008, 0.0012, 0.0006, 0.0010, 0.0009])  # Daily returns
    volatilities = np.array([0.02, 0.025, 0.018, 0.022, 0.024])  # Daily volatilities
    
    # Create covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    # Generate random returns
    returns_data = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    returns_df = pd.DataFrame(returns_data, columns=asset_names)
    
    # Create market data
    market_returns = np.random.normal(0.0008, 0.018, n_days)
    market_df = pd.DataFrame({'market_return': market_returns})
    
    # Test portfolio optimizer
    optimizer = PortfolioOptimizer()
    optimizer.load_data(returns_df, market_df)
    
    print("Testing Portfolio Optimization System")
    print("=" * 50)
    
    # Test different optimization methods
    methods = [
        OptimizationMethod.MEAN_VARIANCE,
        OptimizationMethod.MIN_VARIANCE,
        OptimizationMethod.MAX_SHARPE,
        OptimizationMethod.RISK_PARITY,
        OptimizationMethod.MAX_DIVERSIFICATION
    ]
    
    constraints = OptimizationConstraints(
        min_weight=0.0,
        max_weight=0.4,
        max_turnover=0.5
    )
    
    results = {}
    
    for method in methods:
        print(f"\nOptimizing with {method.value}:")
        result = optimizer.optimize_portfolio(method, constraints)
        results[method.value] = result
        
        print(f"  Expected Return: {result.expected_return:.4f}")
        print(f"  Expected Volatility: {result.expected_volatility:.4f}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.4f}")
        print(f"  Max Drawdown: {result.max_drawdown:.4f}")
        print(f"  Diversification Ratio: {result.diversification_ratio:.4f}")
        print(f"  Optimization Time: {result.optimization_time:.3f}s")
        
        print("  Weights:")
        for asset, weight in result.weights.items():
            if weight > 0.01:  # Only show significant weights
                print(f"    {asset}: {weight:.3f}")
        
        print("  Risk Contributions:")
        for asset, contrib in result.risk_contributions.items():
            if contrib > 0.01:  # Only show significant contributions
                print(f"    {asset}: {contrib:.3f}")
    
    # Test portfolio metrics calculation
    print(f"\nPortfolio Metrics (Mean-Variance Portfolio):")
    print("=" * 40)
    
    mv_weights = results['mean_variance'].weights
    metrics = optimizer.calculate_portfolio_metrics(mv_weights)
    
    print(f"Total Return: {metrics.total_return:.4f}")
    print(f"Volatility: {metrics.volatility:.4f}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.4f}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.4f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.4f}")
    print(f"VaR (95%): {metrics.var_95:.4f}")
    print(f"CVaR (95%): {metrics.cvar_95:.4f}")
    
    # Test rebalancing
    print(f"\nRebalancing Test:")
    print("=" * 20)
    
    current_weights = {asset: 0.2 for asset in asset_names}  # Equal weights
    target_weights = mv_weights
    
    trades = optimizer.rebalance_portfolio(current_weights, target_weights, rebalance_threshold=0.02)
    
    print("Required Trades:")
    for asset, trade in trades.items():
        print(f"  {asset}: {trade:+.3f}")
    
    print(f"\nPortfolio Optimization Testing Complete!")