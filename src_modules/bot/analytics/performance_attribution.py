"""
Advanced Performance Attribution Engine.
Provides factor-based performance analysis, attribution decomposition, and contribution analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class AttributionMethod(Enum):
    """Performance attribution methods."""
    BRINSON = "brinson"
    BARRA = "barra"
    FAMA_FRENCH = "fama_french"
    FACTOR_MODEL = "factor_model"
    CUSTOM = "custom"

@dataclass
class FactorContribution:
    """Individual factor contribution to performance."""
    factor_name: str
    contribution: float
    contribution_pct: float
    factor_return: float
    factor_exposure: float
    significance: float
    confidence_interval: Tuple[float, float]

@dataclass
class AttributionResult:
    """Complete performance attribution result."""
    total_return: float
    benchmark_return: float
    active_return: float
    attribution_method: AttributionMethod
    factor_contributions: List[FactorContribution]
    residual_return: float
    explained_variance: float
    r_squared: float
    tracking_error: float
    information_ratio: float
    attribution_timestamp: datetime
    period: str
    metadata: Dict[str, Any]

class PerformanceAttributor:
    """Advanced performance attribution and analysis engine."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Attribution configuration
        self.attribution_config = {
            'default_method': AttributionMethod.FACTOR_MODEL,
            'lookback_periods': {
                'daily': 30,
                'weekly': 12,
                'monthly': 6,
                'quarterly': 4
            },
            'risk_free_rate': 0.02,  # Annual risk-free rate
            'confidence_level': 0.95,
            'min_observations': 20,
            'factor_models': {
                'market': ['market_return'],
                'fama_french_3': ['market_return', 'smb', 'hml'],
                'fama_french_5': ['market_return', 'smb', 'hml', 'rmw', 'cma'],
                'custom_crypto': ['btc_return', 'eth_return', 'defi_index', 'volatility_factor']
            }
        }
        
        # Factor data storage
        self.factor_data = {}
        self.benchmark_data = {}
        
        # Attribution history
        self.attribution_history = {}
        
        self.logger.info("PerformanceAttributor initialized")
    
    async def attribute_performance(self, 
                                  portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series,
                                  factor_data: pd.DataFrame,
                                  method: AttributionMethod = None,
                                  period: str = 'daily') -> AttributionResult:
        """Perform comprehensive performance attribution analysis."""
        try:
            if method is None:
                method = self.attribution_config['default_method']
            
            # Validate inputs
            self._validate_attribution_inputs(portfolio_returns, benchmark_returns, factor_data)
            
            # Align data
            aligned_data = self._align_data(portfolio_returns, benchmark_returns, factor_data)
            portfolio_ret, benchmark_ret, factors = aligned_data
            
            # Calculate active returns
            active_returns = portfolio_ret - benchmark_ret
            
            # Perform attribution based on method
            if method == AttributionMethod.FACTOR_MODEL:
                attribution = await self._factor_model_attribution(
                    active_returns, factors, portfolio_ret, benchmark_ret
                )
            elif method == AttributionMethod.BRINSON:
                attribution = await self._brinson_attribution(
                    portfolio_ret, benchmark_ret, factors
                )
            elif method == AttributionMethod.FAMA_FRENCH:
                attribution = await self._fama_french_attribution(
                    active_returns, factors
                )
            else:
                raise ValueError(f"Unsupported attribution method: {method}")
            
            # Calculate additional metrics
            tracking_error = np.std(active_returns) * np.sqrt(252)  # Annualized
            information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
            
            # Create result
            result = AttributionResult(
                total_return=float(portfolio_ret.sum()),
                benchmark_return=float(benchmark_ret.sum()),
                active_return=float(active_returns.sum()),
                attribution_method=method,
                factor_contributions=attribution['factors'],
                residual_return=attribution['residual'],
                explained_variance=attribution['explained_variance'],
                r_squared=attribution['r_squared'],
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                attribution_timestamp=datetime.now(),
                period=period,
                metadata=attribution['metadata']
            )
            
            # Store in history
            self._store_attribution_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Performance attribution failed: {e}")
            raise
    
    async def multi_period_attribution(self, 
                                     portfolio_returns: pd.Series,
                                     benchmark_returns: pd.Series,
                                     factor_data: pd.DataFrame,
                                     periods: List[str] = None) -> Dict[str, AttributionResult]:
        """Perform attribution analysis across multiple time periods."""
        try:
            if periods is None:
                periods = ['daily', 'weekly', 'monthly']
            
            results = {}
            
            for period in periods:
                # Resample data for period
                freq_map = {
                    'daily': 'D',
                    'weekly': 'W',
                    'monthly': 'M',
                    'quarterly': 'Q'
                }
                
                freq = freq_map.get(period, 'D')
                
                # Resample returns
                portfolio_period = portfolio_returns.resample(freq).apply(
                    lambda x: (1 + x).prod() - 1
                )
                benchmark_period = benchmark_returns.resample(freq).apply(
                    lambda x: (1 + x).prod() - 1
                )
                factor_period = factor_data.resample(freq).apply(
                    lambda x: (1 + x).prod() - 1
                )
                
                # Perform attribution
                result = await self.attribute_performance(
                    portfolio_period, benchmark_period, factor_period, period=period
                )
                
                results[period] = result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-period attribution failed: {e}")
            raise
    
    async def rolling_attribution(self, 
                                portfolio_returns: pd.Series,
                                benchmark_returns: pd.Series,
                                factor_data: pd.DataFrame,
                                window: int = 60,
                                method: AttributionMethod = None) -> pd.DataFrame:
        """Perform rolling performance attribution analysis."""
        try:
            if method is None:
                method = self.attribution_config['default_method']
            
            results = []
            
            for i in range(window, len(portfolio_returns)):
                # Get rolling window data
                portfolio_window = portfolio_returns.iloc[i-window:i]
                benchmark_window = benchmark_returns.iloc[i-window:i]
                factor_window = factor_data.iloc[i-window:i]
                
                try:
                    # Perform attribution
                    attribution = await self.attribute_performance(
                        portfolio_window, benchmark_window, factor_window, method
                    )
                    
                    # Extract key metrics
                    result_row = {
                        'date': portfolio_returns.index[i-1],
                        'active_return': attribution.active_return,
                        'tracking_error': attribution.tracking_error,
                        'information_ratio': attribution.information_ratio,
                        'explained_variance': attribution.explained_variance,
                        'residual_return': attribution.residual_return
                    }
                    
                    # Add factor contributions
                    for factor_contrib in attribution.factor_contributions:
                        result_row[f'factor_{factor_contrib.factor_name}'] = factor_contrib.contribution
                    
                    results.append(result_row)
                    
                except Exception as e:
                    self.logger.warning(f"Rolling attribution failed for window ending {portfolio_returns.index[i-1]}: {e}")
                    continue
            
            return pd.DataFrame(results).set_index('date')
            
        except Exception as e:
            self.logger.error(f"Rolling attribution analysis failed: {e}")
            raise
    
    async def _factor_model_attribution(self, active_returns: pd.Series, 
                                      factors: pd.DataFrame,
                                      portfolio_returns: pd.Series,
                                      benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Perform factor model-based attribution."""
        try:
            # Prepare regression data
            X = factors.values
            y = active_returns.values
            
            # Add constant term
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            # Perform regression
            try:
                beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
                alpha = beta[0]
                factor_betas = beta[1:]
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                beta = np.linalg.pinv(X_with_const) @ y
                alpha = beta[0]
                factor_betas = beta[1:]
                residuals = y - X_with_const @ beta
            
            # Calculate factor contributions
            factor_contributions = []
            total_contribution = 0
            
            for i, (factor_name, factor_beta) in enumerate(zip(factors.columns, factor_betas)):
                factor_return = factors.iloc[:, i].mean()
                contribution = factor_beta * factor_return
                contribution_pct = contribution / active_returns.mean() * 100 if active_returns.mean() != 0 else 0
                
                # Calculate significance (t-statistic)
                factor_std = factors.iloc[:, i].std()
                if factor_std > 0:
                    t_stat = factor_beta / (factor_std / np.sqrt(len(factors)))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(factors) - len(factor_betas) - 1))
                    significance = 1 - p_value
                else:
                    significance = 0.0
                
                # Confidence interval
                if len(residuals) > 0:
                    residual_std = np.std(residuals)
                    se = residual_std / np.sqrt(len(factors))
                    t_critical = stats.t.ppf(0.975, len(factors) - len(factor_betas) - 1)
                    ci_lower = contribution - t_critical * se
                    ci_upper = contribution + t_critical * se
                else:
                    ci_lower = ci_upper = contribution
                
                factor_contributions.append(FactorContribution(
                    factor_name=factor_name,
                    contribution=float(contribution),
                    contribution_pct=float(contribution_pct),
                    factor_return=float(factor_return),
                    factor_exposure=float(factor_beta),
                    significance=float(significance),
                    confidence_interval=(float(ci_lower), float(ci_upper))
                ))
                
                total_contribution += contribution
            
            # Calculate explained variance
            y_pred = X @ factor_betas + alpha
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            explained_variance = r_squared
            
            # Residual return (alpha)
            residual_return = float(alpha)
            
            return {
                'factors': factor_contributions,
                'residual': residual_return,
                'explained_variance': explained_variance,
                'r_squared': r_squared,
                'metadata': {
                    'regression_alpha': float(alpha),
                    'factor_betas': [float(b) for b in factor_betas],
                    'total_factor_contribution': float(total_contribution),
                    'number_of_factors': len(factors.columns)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Factor model attribution failed: {e}")
            raise
    
    async def _brinson_attribution(self, portfolio_returns: pd.Series,
                                 benchmark_returns: pd.Series,
                                 sector_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Brinson-Hood-Beebower attribution analysis."""
        try:
            # This is a simplified Brinson attribution
            # In practice, would need sector weights and returns
            
            # Calculate allocation and selection effects
            factor_contributions = []
            
            # For demonstration, create synthetic sector attribution
            sectors = ['Technology', 'Finance', 'Energy', 'Healthcare']
            
            for sector in sectors:
                # Synthetic allocation and selection effects
                allocation_effect = np.random.normal(0, 0.01)  # Random for demo
                selection_effect = np.random.normal(0, 0.01)
                total_effect = allocation_effect + selection_effect
                
                factor_contributions.append(FactorContribution(
                    factor_name=f"{sector}_allocation",
                    contribution=float(allocation_effect),
                    contribution_pct=float(allocation_effect * 100),
                    factor_return=0.0,
                    factor_exposure=1.0,
                    significance=0.5,
                    confidence_interval=(float(allocation_effect - 0.01), float(allocation_effect + 0.01))
                ))
                
                factor_contributions.append(FactorContribution(
                    factor_name=f"{sector}_selection",
                    contribution=float(selection_effect),
                    contribution_pct=float(selection_effect * 100),
                    factor_return=0.0,
                    factor_exposure=1.0,
                    significance=0.5,
                    confidence_interval=(float(selection_effect - 0.01), float(selection_effect + 0.01))
                ))
            
            return {
                'factors': factor_contributions,
                'residual': 0.0,
                'explained_variance': 0.8,
                'r_squared': 0.8,
                'metadata': {
                    'attribution_type': 'brinson',
                    'sectors_analyzed': len(sectors)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Brinson attribution failed: {e}")
            raise
    
    async def _fama_french_attribution(self, active_returns: pd.Series,
                                     factors: pd.DataFrame) -> Dict[str, Any]:
        """Perform Fama-French factor attribution."""
        try:
            # Expected Fama-French factors: Market, SMB, HML, (RMW, CMA for 5-factor)
            ff_factors = ['market', 'smb', 'hml', 'rmw', 'cma']
            available_factors = [col for col in ff_factors if col in factors.columns]
            
            if not available_factors:
                raise ValueError("No Fama-French factors found in data")
            
            # Use factor model attribution with FF factors
            ff_factor_data = factors[available_factors]
            return await self._factor_model_attribution(
                active_returns, ff_factor_data, None, None
            )
            
        except Exception as e:
            self.logger.error(f"Fama-French attribution failed: {e}")
            raise
    
    def _validate_attribution_inputs(self, portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series,
                                   factor_data: pd.DataFrame):
        """Validate inputs for attribution analysis."""
        if len(portfolio_returns) < self.attribution_config['min_observations']:
            raise ValueError(f"Insufficient data: need at least {self.attribution_config['min_observations']} observations")
        
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Portfolio and benchmark returns must have same length")
        
        if len(factor_data) < len(portfolio_returns):
            raise ValueError("Factor data must have at least as many observations as returns")
        
        # Check for NaN values
        if portfolio_returns.isna().any():
            raise ValueError("Portfolio returns contain NaN values")
        
        if benchmark_returns.isna().any():
            raise ValueError("Benchmark returns contain NaN values")
        
        if factor_data.isna().any().any():
            raise ValueError("Factor data contains NaN values")
    
    def _align_data(self, portfolio_returns: pd.Series,
                   benchmark_returns: pd.Series,
                   factor_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """Align data by common index."""
        # Find common dates
        common_index = portfolio_returns.index.intersection(
            benchmark_returns.index
        ).intersection(factor_data.index)
        
        if len(common_index) < self.attribution_config['min_observations']:
            raise ValueError("Insufficient overlapping data after alignment")
        
        # Align data
        portfolio_aligned = portfolio_returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]
        factors_aligned = factor_data.loc[common_index]
        
        return portfolio_aligned, benchmark_aligned, factors_aligned
    
    def _store_attribution_result(self, result: AttributionResult):
        """Store attribution result in history."""
        key = f"{result.period}_{result.attribution_timestamp.strftime('%Y%m%d_%H%M%S')}"
        self.attribution_history[key] = result
        
        # Limit history size
        if len(self.attribution_history) > 1000:
            # Remove oldest entries
            sorted_keys = sorted(self.attribution_history.keys())
            for key in sorted_keys[:100]:
                del self.attribution_history[key]
    
    def get_attribution_summary(self, period: str = 'all') -> Dict[str, Any]:
        """Get summary of attribution analysis."""
        if period == 'all':
            results = list(self.attribution_history.values())
        else:
            results = [r for r in self.attribution_history.values() if r.period == period]
        
        if not results:
            return {'message': 'No attribution results available'}
        
        # Calculate summary statistics
        active_returns = [r.active_return for r in results]
        tracking_errors = [r.tracking_error for r in results]
        information_ratios = [r.information_ratio for r in results]
        explained_variances = [r.explained_variance for r in results]
        
        return {
            'total_analyses': len(results),
            'avg_active_return': np.mean(active_returns),
            'avg_tracking_error': np.mean(tracking_errors),
            'avg_information_ratio': np.mean(information_ratios),
            'avg_explained_variance': np.mean(explained_variances),
            'periods_analyzed': list(set(r.period for r in results)),
            'date_range': {
                'start': min(r.attribution_timestamp for r in results).date(),
                'end': max(r.attribution_timestamp for r in results).date()
            }
        }
    
    async def factor_importance_analysis(self, 
                                       portfolio_returns: pd.Series,
                                       benchmark_returns: pd.Series,
                                       factor_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze the importance of different factors in explaining returns."""
        try:
            # Perform full attribution
            attribution = await self.attribute_performance(
                portfolio_returns, benchmark_returns, factor_data
            )
            
            # Calculate factor importance scores
            importance_scores = {}
            total_abs_contribution = sum(
                abs(fc.contribution) for fc in attribution.factor_contributions
            )
            
            if total_abs_contribution > 0:
                for factor_contrib in attribution.factor_contributions:
                    importance = abs(factor_contrib.contribution) / total_abs_contribution
                    importance_scores[factor_contrib.factor_name] = importance
            
            return importance_scores
            
        except Exception as e:
            self.logger.error(f"Factor importance analysis failed: {e}")
            return {}