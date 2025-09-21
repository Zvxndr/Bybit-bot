"""
Advanced Position Sizing Methods for Trading Strategies.

This module provides sophisticated position sizing techniques optimized
for cryptocurrency trading:

- Kelly Criterion with safety adjustments
- Risk Parity and Volatility Targeting
- Maximum Drawdown-based sizing
- Dynamic position sizing based on market conditions
- Multi-asset portfolio optimization
- Regime-aware position adjustments
- Correlation-adjusted sizing
- Capital preservation mechanisms

All methods incorporate proper risk management principles and
account for the unique characteristics of cryptocurrency markets.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import matplotlib.pyplot as plt

from ..utils.logging import TradingLogger


@dataclass
class PositionSizeResult:
    """Container for position sizing results."""
    
    asset: str
    recommended_size: float
    max_size: float
    risk_budget: float
    confidence_level: float
    method_used: str
    risk_metrics: Dict[str, float]
    warnings: List[str]
    timestamp: datetime
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.risk_metrics is None:
            self.risk_metrics = {}


class KellyCriterion:
    """
    Kelly Criterion implementation with safety adjustments for crypto trading.
    
    The Kelly Criterion determines optimal position size based on win probability
    and win/loss ratios, with modifications for cryptocurrency market characteristics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("KellyCriterion")
        
    def _default_config(self) -> Dict:
        """Default configuration for Kelly Criterion."""
        return {
            'max_kelly_fraction': 0.25,  # Maximum 25% Kelly allocation
            'safety_factor': 0.5,        # Apply 50% safety reduction
            'min_win_rate': 0.4,         # Minimum win rate to use Kelly
            'min_samples': 30,           # Minimum trade samples
            'lookback_period': 252,      # Trading days for calculation
            'confidence_level': 0.95,    # Confidence level for estimates
            'volatility_adjustment': True,  # Adjust for volatility
            'drawdown_adjustment': True,    # Adjust for drawdown
        }
    
    def calculate_kelly_fraction(
        self,
        returns: pd.Series,
        strategy_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate Kelly fraction based on historical returns.
        
        Args:
            returns: Asset returns
            strategy_returns: Strategy-specific returns (if available)
            
        Returns:
            Dictionary with Kelly metrics
        """
        if strategy_returns is not None:
            analysis_returns = strategy_returns
        else:
            analysis_returns = returns
        
        # Remove NaN values
        analysis_returns = analysis_returns.dropna()
        
        if len(analysis_returns) < self.config['min_samples']:
            self.logger.warning(f"Insufficient data: {len(analysis_returns)} < {self.config['min_samples']}")
            return self._default_kelly_result()
        
        # Calculate win/loss statistics
        positive_returns = analysis_returns[analysis_returns > 0]
        negative_returns = analysis_returns[analysis_returns < 0]
        
        win_rate = len(positive_returns) / len(analysis_returns)
        
        if win_rate < self.config['min_win_rate']:
            self.logger.warning(f"Win rate too low: {win_rate:.2%} < {self.config['min_win_rate']:.2%}")
            return self._default_kelly_result()
        
        # Calculate average win and loss
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0.01
        
        if avg_loss == 0:
            avg_loss = 0.01  # Prevent division by zero
        
        # Kelly formula: f = (bp - q) / b
        # where: b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio
        
        # Apply safety constraints
        kelly_fraction = max(0, kelly_fraction)  # No negative Kelly
        kelly_fraction = min(kelly_fraction, self.config['max_kelly_fraction'])
        
        # Apply safety factor
        safe_kelly = kelly_fraction * self.config['safety_factor']
        
        # Volatility adjustment
        if self.config['volatility_adjustment']:
            volatility = analysis_returns.std() * np.sqrt(252)  # Annualized
            vol_adjustment = min(1.0, 0.2 / volatility)  # Reduce size for high vol
            safe_kelly *= vol_adjustment
        
        # Drawdown adjustment
        if self.config['drawdown_adjustment']:
            cumulative = (1 + analysis_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            if max_drawdown > 0.1:  # 10% drawdown threshold
                dd_adjustment = max(0.5, 1 - max_drawdown)
                safe_kelly *= dd_adjustment
        
        return {
            'raw_kelly': kelly_fraction,
            'safe_kelly': safe_kelly,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sample_size': len(analysis_returns),
            'confidence': self._calculate_confidence(analysis_returns)
        }
    
    def _default_kelly_result(self) -> Dict[str, float]:
        """Return default Kelly result for insufficient data."""
        return {
            'raw_kelly': 0.0,
            'safe_kelly': 0.05,  # Conservative 5% allocation
            'win_rate': 0.5,
            'win_loss_ratio': 1.0,
            'avg_win': 0.01,
            'avg_loss': 0.01,
            'sample_size': 0,
            'confidence': 0.0
        }
    
    def _calculate_confidence(self, returns: pd.Series) -> float:
        """Calculate confidence level in Kelly estimate."""
        n = len(returns)
        if n < 10:
            return 0.0
        
        # Bootstrap confidence intervals
        n_bootstrap = 100
        kelly_estimates = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = returns.sample(n=n, replace=True)
            kelly_result = self.calculate_kelly_fraction(bootstrap_sample)
            kelly_estimates.append(kelly_result['safe_kelly'])
        
        # Calculate confidence based on stability of estimates
        kelly_std = np.std(kelly_estimates)
        confidence = max(0, 1 - (kelly_std * 5))  # Scale to 0-1
        
        return confidence


class VolatilityTargeting:
    """
    Volatility targeting position sizing for consistent risk exposure.
    
    This method adjusts position sizes to maintain a target portfolio volatility,
    automatically scaling up during low volatility periods and down during
    high volatility periods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("VolatilityTargeting")
        
    def _default_config(self) -> Dict:
        """Default configuration for volatility targeting."""
        return {
            'target_volatility': 0.15,   # 15% annual target volatility
            'volatility_window': 20,     # Days for volatility calculation
            'max_leverage': 3.0,         # Maximum leverage allowed
            'min_position': 0.01,        # Minimum position size (1%)
            'max_position': 0.5,         # Maximum position size (50%)
            'vol_floor': 0.05,           # Minimum volatility assumption
            'vol_ceiling': 1.0,          # Maximum volatility assumption
            'rebalance_threshold': 0.2,  # Rebalance when vol changes by 20%
        }
    
    def calculate_position_size(
        self,
        returns: pd.Series,
        current_position: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate position size based on volatility targeting.
        
        Args:
            returns: Historical returns
            current_position: Current position size
            
        Returns:
            Dictionary with position sizing metrics
        """
        # Calculate realized volatility
        realized_vol = self._calculate_realized_volatility(returns)
        
        # Calculate target position size
        target_position = self.config['target_volatility'] / realized_vol
        
        # Apply constraints
        target_position = max(self.config['min_position'], target_position)
        target_position = min(self.config['max_position'], target_position)
        target_position = min(target_position, self.config['max_leverage'])
        
        # Check if rebalancing is needed
        position_change = abs(target_position - current_position) / max(current_position, 0.01)
        needs_rebalancing = position_change > self.config['rebalance_threshold']
        
        return {
            'target_position': target_position,
            'current_position': current_position,
            'position_change': target_position - current_position,
            'realized_volatility': realized_vol,
            'target_volatility': self.config['target_volatility'],
            'vol_ratio': realized_vol / self.config['target_volatility'],
            'needs_rebalancing': needs_rebalancing,
            'confidence': self._calculate_vol_confidence(returns)
        }
    
    def _calculate_realized_volatility(self, returns: pd.Series) -> float:
        """Calculate realized volatility with appropriate adjustments."""
        # Use recent returns for volatility calculation
        recent_returns = returns.tail(self.config['volatility_window'])
        
        if len(recent_returns) < 5:
            self.logger.warning("Insufficient data for volatility calculation")
            return self.config['target_volatility']  # Default to target
        
        # Calculate annualized volatility
        vol = recent_returns.std() * np.sqrt(252)
        
        # Apply floor and ceiling
        vol = max(self.config['vol_floor'], vol)
        vol = min(self.config['vol_ceiling'], vol)
        
        return vol
    
    def _calculate_vol_confidence(self, returns: pd.Series) -> float:
        """Calculate confidence in volatility estimate."""
        n = len(returns)
        
        if n < 10:
            return 0.0
        
        # Confidence based on sample size and stability
        stability_factor = min(1.0, n / 100)  # More confidence with more data
        
        # Check volatility stability over time
        if n >= 40:
            vol1 = returns.iloc[:n//2].std()
            vol2 = returns.iloc[n//2:].std()
            vol_stability = 1 - abs(vol1 - vol2) / max(vol1, vol2, 0.01)
            stability_factor *= vol_stability
        
        return stability_factor


class RiskParity:
    """
    Risk Parity position sizing for multi-asset portfolios.
    
    This method allocates risk equally across assets rather than capital,
    leading to more balanced risk exposure across the portfolio.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("RiskParity")
        
    def _default_config(self) -> Dict:
        """Default configuration for risk parity."""
        return {
            'lookback_period': 60,       # Days for covariance estimation
            'min_weight': 0.01,          # Minimum asset weight
            'max_weight': 0.4,           # Maximum asset weight
            'regularization': 0.01,      # Covariance regularization
            'risk_budget_tolerance': 0.05,  # Tolerance for risk budget deviation
            'rebalance_threshold': 0.1,  # Rebalance when weights change by 10%
            'shrinkage_method': 'ledoit_wolf',  # Covariance shrinkage method
        }
    
    def calculate_risk_parity_weights(
        self,
        returns_matrix: pd.DataFrame,
        risk_budgets: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate risk parity weights for multi-asset portfolio.
        
        Args:
            returns_matrix: DataFrame with asset returns
            risk_budgets: Optional custom risk budgets per asset
            
        Returns:
            Dictionary with weights and risk metrics
        """
        if returns_matrix.empty:
            raise ValueError("Returns matrix cannot be empty")
        
        # Clean data
        returns_clean = returns_matrix.dropna()
        
        if len(returns_clean) < 20:
            self.logger.warning("Insufficient data for risk parity calculation")
            return self._equal_weight_fallback(returns_matrix.columns)
        
        # Calculate covariance matrix with shrinkage
        cov_matrix = self._calculate_covariance_matrix(returns_clean)
        
        # Set equal risk budgets if not provided
        n_assets = len(returns_matrix.columns)
        if risk_budgets is None:
            risk_budgets = {asset: 1.0/n_assets for asset in returns_matrix.columns}
        
        # Optimize for risk parity
        try:
            weights = self._optimize_risk_parity(cov_matrix, risk_budgets)
        except Exception as e:
            self.logger.error(f"Risk parity optimization failed: {e}")
            return self._equal_weight_fallback(returns_matrix.columns)
        
        # Calculate risk contributions
        risk_contributions = self._calculate_risk_contributions(weights, cov_matrix)
        
        return {
            'weights': weights,
            'risk_contributions': risk_contributions,
            'covariance_matrix': cov_matrix,
            'risk_budgets': risk_budgets,
            'portfolio_volatility': self._calculate_portfolio_volatility(weights, cov_matrix),
            'risk_budget_errors': self._calculate_risk_budget_errors(risk_contributions, risk_budgets)
        }
    
    def _calculate_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate shrunk covariance matrix."""
        if self.config['shrinkage_method'] == 'ledoit_wolf':
            cov_estimator = LedoitWolf()
        else:
            cov_estimator = EmpiricalCovariance()
        
        cov_matrix = pd.DataFrame(
            cov_estimator.fit(returns).covariance_,
            index=returns.columns,
            columns=returns.columns
        )
        
        # Apply regularization
        reg_matrix = np.eye(len(cov_matrix)) * self.config['regularization']
        cov_matrix += reg_matrix
        
        return cov_matrix
    
    def _optimize_risk_parity(
        self,
        cov_matrix: pd.DataFrame,
        risk_budgets: Dict[str, float]
    ) -> Dict[str, float]:
        """Optimize weights for risk parity."""
        n_assets = len(cov_matrix)
        assets = cov_matrix.index.tolist()
        
        # Convert risk budgets to array
        b = np.array([risk_budgets[asset] for asset in assets])
        
        def objective(weights):
            """Risk parity objective function."""
            w = np.array(weights)
            portfolio_vol = np.sqrt(w.T @ cov_matrix.values @ w)
            
            # Risk contributions
            risk_contrib = (w * (cov_matrix.values @ w)) / portfolio_vol
            
            # Normalize risk contributions
            risk_contrib = risk_contrib / risk_contrib.sum()
            
            # Sum of squared deviations from risk budgets
            return np.sum((risk_contrib - b) ** 2)
        
        # Constraints: weights sum to 1, non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        
        # Bounds: min and max weights
        bounds = [(self.config['min_weight'], self.config['max_weight']) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            self.logger.warning("Risk parity optimization did not converge")
        
        # Convert to dictionary
        weights = {asset: weight for asset, weight in zip(assets, result.x)}
        
        return weights
    
    def _calculate_risk_contributions(
        self,
        weights: Dict[str, float],
        cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate risk contributions for each asset."""
        w = np.array([weights[asset] for asset in cov_matrix.index])
        portfolio_vol = np.sqrt(w.T @ cov_matrix.values @ w)
        
        # Marginal risk contributions
        marginal_contrib = cov_matrix.values @ w / portfolio_vol
        
        # Risk contributions
        risk_contrib = w * marginal_contrib
        
        # Convert to dictionary and normalize
        total_risk = risk_contrib.sum()
        risk_contributions = {
            asset: contrib / total_risk
            for asset, contrib in zip(cov_matrix.index, risk_contrib)
        }
        
        return risk_contributions
    
    def _calculate_portfolio_volatility(
        self,
        weights: Dict[str, float],
        cov_matrix: pd.DataFrame
    ) -> float:
        """Calculate portfolio volatility."""
        w = np.array([weights[asset] for asset in cov_matrix.index])
        return np.sqrt(w.T @ cov_matrix.values @ w)
    
    def _calculate_risk_budget_errors(
        self,
        risk_contributions: Dict[str, float],
        risk_budgets: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate deviations from target risk budgets."""
        errors = {}
        for asset in risk_contributions:
            target = risk_budgets.get(asset, 0)
            actual = risk_contributions[asset]
            errors[asset] = actual - target
        
        return errors
    
    def _equal_weight_fallback(self, assets: List[str]) -> Dict[str, Any]:
        """Fallback to equal weights when optimization fails."""
        n_assets = len(assets)
        equal_weight = 1.0 / n_assets
        
        weights = {asset: equal_weight for asset in assets}
        risk_contributions = {asset: equal_weight for asset in assets}
        
        return {
            'weights': weights,
            'risk_contributions': risk_contributions,
            'covariance_matrix': None,
            'risk_budgets': risk_contributions,
            'portfolio_volatility': 0.15,  # Default assumption
            'risk_budget_errors': {asset: 0.0 for asset in assets}
        }


class MaxDrawdownSizing:
    """
    Position sizing based on maximum drawdown constraints.
    
    This method calculates position sizes to ensure that the maximum
    expected drawdown stays within acceptable limits.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("MaxDrawdownSizing")
        
    def _default_config(self) -> Dict:
        """Default configuration for max drawdown sizing."""
        return {
            'max_drawdown_limit': 0.15,  # 15% maximum drawdown
            'confidence_level': 0.95,    # 95% confidence level
            'lookback_period': 252,      # Days for calculation
            'drawdown_method': 'historical',  # historical, parametric, monte_carlo
            'safety_margin': 0.8,        # Apply 20% safety margin
            'min_position': 0.01,        # Minimum position size
            'max_position': 0.3,         # Maximum position size
        }
    
    def calculate_max_position_size(
        self,
        returns: pd.Series,
        strategy_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate maximum position size based on drawdown constraint.
        
        Args:
            returns: Asset returns
            strategy_returns: Strategy-specific returns (if available)
            
        Returns:
            Dictionary with position sizing results
        """
        analysis_returns = strategy_returns if strategy_returns is not None else returns
        analysis_returns = analysis_returns.dropna()
        
        if len(analysis_returns) < 30:
            self.logger.warning("Insufficient data for drawdown analysis")
            return {
                'max_position_size': self.config['min_position'],
                'expected_drawdown': 0.0,
                'confidence_level': 0.0,
                'method_used': 'default'
            }
        
        if self.config['drawdown_method'] == 'historical':
            max_size = self._historical_drawdown_sizing(analysis_returns)
        elif self.config['drawdown_method'] == 'parametric':
            max_size = self._parametric_drawdown_sizing(analysis_returns)
        else:  # monte_carlo
            max_size = self._monte_carlo_drawdown_sizing(analysis_returns)
        
        return max_size
    
    def _historical_drawdown_sizing(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate position size based on historical drawdowns."""
        # Calculate historical drawdowns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        
        # Find maximum historical drawdown
        max_historical_dd = abs(drawdowns.min())
        
        if max_historical_dd == 0:
            max_historical_dd = 0.05  # Assume 5% if no drawdowns
        
        # Calculate position size to stay within drawdown limit
        position_size = self.config['max_drawdown_limit'] / max_historical_dd
        position_size *= self.config['safety_margin']
        
        # Apply constraints
        position_size = max(self.config['min_position'], position_size)
        position_size = min(self.config['max_position'], position_size)
        
        return {
            'max_position_size': position_size,
            'expected_drawdown': max_historical_dd * position_size,
            'historical_max_drawdown': max_historical_dd,
            'confidence_level': 0.8,  # Historical confidence
            'method_used': 'historical'
        }
    
    def _parametric_drawdown_sizing(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate position size using parametric drawdown estimation."""
        # Estimate parameters
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Parametric drawdown estimation (simplified)
        # Based on maximum drawdown formula for geometric Brownian motion
        
        # Annualize parameters
        annual_mean = mean_return * 252
        annual_vol = volatility * np.sqrt(252)
        
        # Estimate maximum drawdown using Ornstein-Uhlenbeck approximation
        if annual_mean > 0:
            # Expected maximum drawdown for trending asset
            expected_dd = (annual_vol ** 2) / (2 * annual_mean)
        else:
            # For non-trending asset, use volatility-based estimate
            expected_dd = annual_vol * np.sqrt(np.pi / 2)
        
        # Adjust for confidence level
        confidence_multiplier = stats.norm.ppf(self.config['confidence_level'])
        expected_dd *= confidence_multiplier
        
        # Calculate position size
        position_size = self.config['max_drawdown_limit'] / expected_dd
        position_size *= self.config['safety_margin']
        
        # Apply constraints
        position_size = max(self.config['min_position'], position_size)
        position_size = min(self.config['max_position'], position_size)
        
        return {
            'max_position_size': position_size,
            'expected_drawdown': expected_dd * position_size,
            'parametric_dd_estimate': expected_dd,
            'confidence_level': self.config['confidence_level'],
            'method_used': 'parametric'
        }
    
    def _monte_carlo_drawdown_sizing(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate position size using Monte Carlo simulation."""
        # Estimate distribution parameters
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Monte Carlo parameters
        n_simulations = 1000
        n_periods = len(returns)
        
        max_drawdowns = []
        
        # Run simulations
        for _ in range(n_simulations):
            # Generate random returns
            simulated_returns = np.random.normal(mean_return, volatility, n_periods)
            
            # Calculate drawdowns
            cumulative = np.cumprod(1 + simulated_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            
            max_drawdowns.append(abs(drawdowns.min()))
        
        # Calculate percentile-based drawdown
        expected_dd = np.percentile(max_drawdowns, self.config['confidence_level'] * 100)
        
        # Calculate position size
        position_size = self.config['max_drawdown_limit'] / expected_dd
        position_size *= self.config['safety_margin']
        
        # Apply constraints
        position_size = max(self.config['min_position'], position_size)
        position_size = min(self.config['max_position'], position_size)
        
        return {
            'max_position_size': position_size,
            'expected_drawdown': expected_dd * position_size,
            'monte_carlo_dd_estimate': expected_dd,
            'confidence_level': self.config['confidence_level'],
            'method_used': 'monte_carlo'
        }


class PositionSizer:
    """
    Unified position sizing system that combines multiple methods.
    
    This class provides a unified interface for position sizing using
    multiple techniques and combines them intelligently based on
    market conditions and strategy characteristics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("PositionSizer")
        
        # Initialize individual sizing methods
        self.kelly_criterion = KellyCriterion(self.config.get('kelly', {}))
        self.volatility_targeting = VolatilityTargeting(self.config.get('volatility', {}))
        self.risk_parity = RiskParity(self.config.get('risk_parity', {}))
        self.max_drawdown_sizing = MaxDrawdownSizing(self.config.get('max_drawdown', {}))
        
    def _default_config(self) -> Dict:
        """Default configuration for unified position sizer."""
        return {
            'primary_method': 'volatility_targeting',  # Primary sizing method
            'use_ensemble': True,                      # Use ensemble of methods
            'ensemble_weights': {                      # Weights for ensemble
                'kelly': 0.3,
                'volatility': 0.4,
                'max_drawdown': 0.3
            },
            'max_position_override': 0.2,              # Hard maximum position size
            'min_position_override': 0.01,             # Hard minimum position size
            'regime_adjustment': True,                 # Adjust based on market regime
            'correlation_adjustment': True,            # Adjust for portfolio correlation
            'kelly': {},
            'volatility': {},
            'risk_parity': {},
            'max_drawdown': {},
        }
    
    def calculate_position_size(
        self,
        asset: str,
        returns: pd.Series,
        strategy_returns: Optional[pd.Series] = None,
        current_portfolio: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using ensemble approach.
        
        Args:
            asset: Asset symbol
            returns: Historical asset returns
            strategy_returns: Strategy-specific returns (if available)
            current_portfolio: Current portfolio positions
            market_regime: Current market regime
            
        Returns:
            PositionSizeResult with comprehensive sizing information
        """
        self.logger.info(f"Calculating position size for {asset}")
        
        warnings = []
        risk_metrics = {}
        
        # Calculate individual method recommendations
        method_results = {}
        
        # Kelly Criterion
        try:
            kelly_result = self.kelly_criterion.calculate_kelly_fraction(returns, strategy_returns)
            method_results['kelly'] = kelly_result['safe_kelly']
            risk_metrics.update({f'kelly_{k}': v for k, v in kelly_result.items()})
        except Exception as e:
            self.logger.warning(f"Kelly calculation failed: {e}")
            method_results['kelly'] = 0.05
            warnings.append("Kelly criterion calculation failed")
        
        # Volatility Targeting
        try:
            vol_result = self.volatility_targeting.calculate_position_size(returns)
            method_results['volatility'] = vol_result['target_position']
            risk_metrics.update({f'vol_{k}': v for k, v in vol_result.items()})
        except Exception as e:
            self.logger.warning(f"Volatility targeting failed: {e}")
            method_results['volatility'] = 0.1
            warnings.append("Volatility targeting calculation failed")
        
        # Maximum Drawdown Sizing
        try:
            dd_result = self.max_drawdown_sizing.calculate_max_position_size(returns, strategy_returns)
            method_results['max_drawdown'] = dd_result['max_position_size']
            risk_metrics.update({f'dd_{k}': v for k, v in dd_result.items()})
        except Exception as e:
            self.logger.warning(f"Max drawdown sizing failed: {e}")
            method_results['max_drawdown'] = 0.15
            warnings.append("Max drawdown sizing calculation failed")
        
        # Calculate ensemble recommendation
        if self.config['use_ensemble']:
            recommended_size = self._calculate_ensemble_size(method_results)
            method_used = 'ensemble'
        else:
            recommended_size = method_results.get(self.config['primary_method'], 0.1)
            method_used = self.config['primary_method']
        
        # Apply regime adjustments
        if self.config['regime_adjustment'] and market_regime:
            recommended_size = self._apply_regime_adjustment(recommended_size, market_regime)
            warnings.append(f"Applied {market_regime} regime adjustment")
        
        # Apply correlation adjustments
        if self.config['correlation_adjustment'] and current_portfolio:
            recommended_size = self._apply_correlation_adjustment(
                recommended_size, asset, current_portfolio, returns
            )
            warnings.append("Applied correlation adjustment")
        
        # Apply hard limits
        max_size = min(recommended_size * 1.5, self.config['max_position_override'])
        recommended_size = max(self.config['min_position_override'], recommended_size)
        recommended_size = min(max_size, recommended_size)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(method_results, risk_metrics)
        
        return PositionSizeResult(
            asset=asset,
            recommended_size=recommended_size,
            max_size=max_size,
            risk_budget=recommended_size * 0.15,  # Assume 15% risk per position
            confidence_level=confidence_level,
            method_used=method_used,
            risk_metrics=risk_metrics,
            warnings=warnings,
            timestamp=datetime.now()
        )
    
    def _calculate_ensemble_size(self, method_results: Dict[str, float]) -> float:
        """Calculate ensemble position size using weighted average."""
        total_weight = 0
        weighted_size = 0
        
        for method, size in method_results.items():
            weight = self.config['ensemble_weights'].get(method, 0)
            if weight > 0:
                weighted_size += weight * size
                total_weight += weight
        
        if total_weight > 0:
            return weighted_size / total_weight
        else:
            return np.mean(list(method_results.values()))
    
    def _apply_regime_adjustment(self, size: float, regime: str) -> float:
        """Apply regime-based position size adjustments."""
        regime_adjustments = {
            'bull_market': 1.2,      # Increase size in bull markets
            'bear_market': 0.7,      # Decrease size in bear markets
            'high_volatility': 0.6,  # Reduce size in high vol
            'low_volatility': 1.1,   # Slightly increase in low vol
            'neutral': 1.0           # No adjustment
        }
        
        adjustment = regime_adjustments.get(regime.lower(), 1.0)
        return size * adjustment
    
    def _apply_correlation_adjustment(
        self,
        size: float,
        asset: str,
        portfolio: Dict[str, float],
        returns: pd.Series
    ) -> float:
        """Apply correlation-based position size adjustments."""
        # This is a simplified correlation adjustment
        # In practice, would need returns data for all portfolio assets
        
        portfolio_concentration = sum(portfolio.values())
        
        if portfolio_concentration > 0.8:  # Highly concentrated portfolio
            size *= 0.8  # Reduce new position size
        elif portfolio_concentration < 0.3:  # Low concentration
            size *= 1.1  # Allow slightly larger position
        
        return size
    
    def _calculate_confidence_level(
        self,
        method_results: Dict[str, float],
        risk_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall confidence level in sizing recommendation."""
        confidence_factors = []
        
        # Confidence based on method agreement
        if len(method_results) > 1:
            sizes = list(method_results.values())
            coefficient_of_variation = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 1
            agreement_confidence = max(0, 1 - coefficient_of_variation)
            confidence_factors.append(agreement_confidence)
        
        # Confidence based on individual method confidence
        kelly_confidence = risk_metrics.get('kelly_confidence', 0.5)
        vol_confidence = risk_metrics.get('vol_confidence', 0.5)
        
        confidence_factors.extend([kelly_confidence, vol_confidence])
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def calculate_portfolio_sizes(
        self,
        assets_returns: Dict[str, pd.Series],
        strategy_returns: Optional[Dict[str, pd.Series]] = None,
        market_regime: Optional[str] = None
    ) -> Dict[str, PositionSizeResult]:
        """
        Calculate position sizes for multiple assets simultaneously.
        
        Args:
            assets_returns: Dictionary of asset returns
            strategy_returns: Dictionary of strategy-specific returns
            market_regime: Current market regime
            
        Returns:
            Dictionary of position size results
        """
        results = {}
        current_portfolio = {}  # Build up as we calculate
        
        for asset, returns in assets_returns.items():
            asset_strategy_returns = None
            if strategy_returns and asset in strategy_returns:
                asset_strategy_returns = strategy_returns[asset]
            
            result = self.calculate_position_size(
                asset=asset,
                returns=returns,
                strategy_returns=asset_strategy_returns,
                current_portfolio=current_portfolio,
                market_regime=market_regime
            )
            
            results[asset] = result
            current_portfolio[asset] = result.recommended_size
        
        # Apply portfolio-level constraints
        results = self._apply_portfolio_constraints(results)
        
        return results
    
    def _apply_portfolio_constraints(
        self,
        results: Dict[str, PositionSizeResult]
    ) -> Dict[str, PositionSizeResult]:
        """Apply portfolio-level constraints to position sizes."""
        total_allocation = sum(result.recommended_size for result in results.values())
        
        # Scale down if total allocation exceeds 100%
        if total_allocation > 1.0:
            scale_factor = 0.95 / total_allocation  # Leave 5% cash buffer
            
            for asset, result in results.items():
                result.recommended_size *= scale_factor
                result.warnings.append(f"Scaled down by {scale_factor:.2f} for portfolio constraint")
        
        return results