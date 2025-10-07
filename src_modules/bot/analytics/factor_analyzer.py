"""
Advanced Factor Analysis Engine.
Provides multi-factor model analysis, factor decomposition, and factor attribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class FactorType(Enum):
    """Types of factors in multi-factor models."""
    MARKET = "market"
    SIZE = "size"
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    PROFITABILITY = "profitability"
    INVESTMENT = "investment"
    LIQUIDITY = "liquidity"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    SECTOR = "sector"
    CUSTOM = "custom"

@dataclass
class FactorLoadings:
    """Factor loadings and statistics."""
    factor_name: str
    factor_type: FactorType
    loading: float
    t_statistic: float
    p_value: float
    r_squared: float
    significance_level: str
    confidence_interval: Tuple[float, float]
    factor_return_contribution: float
    factor_risk_contribution: float

@dataclass
class FactorModel:
    """Complete factor model results."""
    model_name: str
    factors: List[str]
    factor_loadings: Dict[str, FactorLoadings]
    alpha: float
    alpha_t_stat: float
    alpha_p_value: float
    r_squared: float
    adjusted_r_squared: float
    residual_std: float
    tracking_error: float
    active_risk: float
    factor_risk: float
    specific_risk: float
    information_ratio: float
    model_timestamp: datetime
    sample_period: Tuple[datetime, datetime]
    observations: int

@dataclass
class FactorDecomposition:
    """Risk and return decomposition by factors."""
    total_return: float
    factor_returns: Dict[str, float]
    specific_return: float
    total_risk: float
    factor_risks: Dict[str, float]
    specific_risk: float
    diversification_ratio: float
    factor_correlations: pd.DataFrame
    risk_attribution: Dict[str, float]
    return_attribution: Dict[str, float]

class FactorAnalyzer:
    """Advanced multi-factor analysis and decomposition."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Factor analysis configuration
        self.factor_config = {
            'common_models': {
                'market_model': ['market'],
                'fama_french_3': ['market', 'smb', 'hml'],
                'fama_french_5': ['market', 'smb', 'hml', 'rmw', 'cma'],
                'carhart_4': ['market', 'smb', 'hml', 'momentum'],
                'crypto_factors': ['btc', 'eth', 'defi', 'volatility']
            },
            'significance_levels': {
                'high': 0.01,
                'medium': 0.05,
                'low': 0.10
            },
            'min_observations': 60,
            'rolling_window': 252,
            'bootstrap_samples': 1000
        }
        
        # Factor data storage
        self.factor_data = {}
        self.factor_models = {}
        
        # Scaler for standardization
        self.scaler = StandardScaler()
        
        self.logger.info("FactorAnalyzer initialized")
    
    async def build_factor_model(self, 
                                returns: pd.Series,
                                factor_data: pd.DataFrame,
                                model_name: str = "custom",
                                factors: List[str] = None) -> FactorModel:
        """Build multi-factor model for given returns and factors."""
        try:
            # Validate inputs
            self._validate_factor_inputs(returns, factor_data)
            
            # Align data
            aligned_returns, aligned_factors = self._align_factor_data(returns, factor_data)
            
            # Select factors
            if factors is None:
                factors = list(aligned_factors.columns)
            else:
                factors = [f for f in factors if f in aligned_factors.columns]
            
            if not factors:
                raise ValueError("No valid factors found")
            
            # Prepare regression data
            X = aligned_factors[factors].values
            y = aligned_returns.values
            
            # Add constant for alpha
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            # Perform regression
            try:
                coefficients, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
                alpha = coefficients[0]
                betas = coefficients[1:]
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                coefficients = np.linalg.pinv(X_with_const) @ y
                alpha = coefficients[0]
                betas = coefficients[1:]
                residuals = y - X_with_const @ coefficients
            
            # Calculate model statistics
            y_pred = X_with_const @ coefficients
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            n = len(y)
            k = len(factors)
            adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))
            
            residual_std = np.sqrt(ss_res / (n - k - 1)) if n > k + 1 else np.sqrt(ss_res / n)
            
            # Calculate factor loadings with statistics
            factor_loadings = {}
            
            for i, factor_name in enumerate(factors):
                beta = betas[i]
                
                # Calculate t-statistic and p-value
                if len(residuals) > 0 and residual_std > 0:
                    # Standard error of beta
                    X_factor = X[:, i]
                    se_beta = residual_std / np.sqrt(np.sum((X_factor - np.mean(X_factor)) ** 2))
                    t_stat = beta / se_beta if se_beta > 0 else 0
                    
                    # Two-tailed p-value
                    df = n - k - 1
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if df > 0 else 0.5
                    
                    # Confidence interval
                    t_critical = stats.t.ppf(0.975, df) if df > 0 else 1.96
                    ci_lower = beta - t_critical * se_beta
                    ci_upper = beta + t_critical * se_beta
                else:
                    t_stat = 0
                    p_value = 0.5
                    ci_lower = ci_upper = beta
                
                # Factor contributions
                factor_return = aligned_factors[factor_name].mean()
                factor_return_contrib = beta * factor_return
                
                # Factor risk contribution (simplified)
                factor_variance = aligned_factors[factor_name].var()
                factor_risk_contrib = (beta ** 2) * factor_variance
                
                # Significance level
                if p_value < self.factor_config['significance_levels']['high']:
                    significance = 'high'
                elif p_value < self.factor_config['significance_levels']['medium']:
                    significance = 'medium'
                elif p_value < self.factor_config['significance_levels']['low']:
                    significance = 'low'
                else:
                    significance = 'not_significant'
                
                factor_loadings[factor_name] = FactorLoadings(
                    factor_name=factor_name,
                    factor_type=self._infer_factor_type(factor_name),
                    loading=float(beta),
                    t_statistic=float(t_stat),
                    p_value=float(p_value),
                    r_squared=float(r_squared),
                    significance_level=significance,
                    confidence_interval=(float(ci_lower), float(ci_upper)),
                    factor_return_contribution=float(factor_return_contrib),
                    factor_risk_contribution=float(factor_risk_contrib)
                )
            
            # Calculate alpha statistics
            if len(residuals) > 0 and residual_std > 0:
                se_alpha = residual_std / np.sqrt(n)
                alpha_t_stat = alpha / se_alpha if se_alpha > 0 else 0
                alpha_p_value = 2 * (1 - stats.t.cdf(abs(alpha_t_stat), n - k - 1))
            else:
                alpha_t_stat = 0
                alpha_p_value = 0.5
            
            # Risk decomposition
            total_var = np.var(aligned_returns)
            factor_var = np.var(y_pred - alpha)  # Variance explained by factors
            specific_var = np.var(residuals) if len(residuals) > 0 else 0
            
            tracking_error = np.sqrt(specific_var) * np.sqrt(252)  # Annualized
            factor_risk = np.sqrt(factor_var) * np.sqrt(252)
            specific_risk = np.sqrt(specific_var) * np.sqrt(252)
            active_risk = np.sqrt(total_var) * np.sqrt(252)
            
            # Information ratio
            information_ratio = (alpha * 252) / tracking_error if tracking_error > 0 else 0
            
            # Create model result
            model = FactorModel(
                model_name=model_name,
                factors=factors,
                factor_loadings=factor_loadings,
                alpha=float(alpha),
                alpha_t_stat=float(alpha_t_stat),
                alpha_p_value=float(alpha_p_value),
                r_squared=float(r_squared),
                adjusted_r_squared=float(adjusted_r_squared),
                residual_std=float(residual_std),
                tracking_error=float(tracking_error),
                active_risk=float(active_risk),
                factor_risk=float(factor_risk),
                specific_risk=float(specific_risk),
                information_ratio=float(information_ratio),
                model_timestamp=datetime.now(),
                sample_period=(aligned_returns.index[0], aligned_returns.index[-1]),
                observations=len(aligned_returns)
            )
            
            # Store model
            self.factor_models[model_name] = model
            
            return model
            
        except Exception as e:
            self.logger.error(f"Factor model building failed: {e}")
            raise
    
    async def decompose_risk_return(self, 
                                  returns: pd.Series,
                                  factor_model: FactorModel,
                                  factor_data: pd.DataFrame) -> FactorDecomposition:
        """Decompose returns and risk by factors."""
        try:
            # Align data
            aligned_returns, aligned_factors = self._align_factor_data(returns, factor_data)
            
            # Calculate factor returns
            factor_returns = {}
            factor_risks = {}
            
            total_factor_return = 0
            total_factor_risk = 0
            
            for factor_name, loading in factor_model.factor_loadings.items():
                if factor_name in aligned_factors.columns:
                    factor_ret = aligned_factors[factor_name].mean()
                    factor_vol = aligned_factors[factor_name].std()
                    
                    # Factor contribution to return
                    factor_contribution = loading.loading * factor_ret
                    factor_returns[factor_name] = factor_contribution
                    total_factor_return += factor_contribution
                    
                    # Factor contribution to risk
                    factor_risk_contrib = (loading.loading ** 2) * (factor_vol ** 2)
                    factor_risks[factor_name] = np.sqrt(factor_risk_contrib) * np.sqrt(252)
                    total_factor_risk += factor_risk_contrib
            
            # Specific return and risk
            total_return = aligned_returns.mean()
            specific_return = total_return - total_factor_return - factor_model.alpha
            
            total_risk = aligned_returns.std() * np.sqrt(252)
            factor_risk_total = np.sqrt(total_factor_risk) * np.sqrt(252)
            specific_risk = factor_model.specific_risk
            
            # Risk attribution (percentage)
            total_variance = total_risk ** 2
            risk_attribution = {}
            
            if total_variance > 0:
                for factor_name, risk in factor_risks.items():
                    risk_attribution[factor_name] = (risk ** 2) / total_variance * 100
                
                risk_attribution['specific'] = (specific_risk ** 2) / total_variance * 100
            
            # Return attribution
            total_return_annualized = total_return * 252
            return_attribution = {}
            
            if total_return_annualized != 0:
                for factor_name, ret in factor_returns.items():
                    return_attribution[factor_name] = (ret * 252) / total_return_annualized * 100
                
                return_attribution['alpha'] = (factor_model.alpha * 252) / total_return_annualized * 100
                return_attribution['specific'] = (specific_return * 252) / total_return_annualized * 100
            
            # Factor correlations
            factor_cols = [f for f in factor_model.factors if f in aligned_factors.columns]
            factor_correlations = aligned_factors[factor_cols].corr()
            
            # Diversification ratio
            # Portfolio volatility / weighted average of individual volatilities
            if len(factor_risks) > 0:
                weighted_avg_vol = sum(abs(loading.loading) * np.sqrt(loading.factor_risk_contribution) 
                                     for loading in factor_model.factor_loadings.values())
                diversification_ratio = total_risk / weighted_avg_vol if weighted_avg_vol > 0 else 1.0
            else:
                diversification_ratio = 1.0
            
            return FactorDecomposition(
                total_return=float(total_return * 252),  # Annualized
                factor_returns={k: v * 252 for k, v in factor_returns.items()},  # Annualized
                specific_return=float(specific_return * 252),  # Annualized
                total_risk=float(total_risk),
                factor_risks=factor_risks,
                specific_risk=float(specific_risk),
                diversification_ratio=float(diversification_ratio),
                factor_correlations=factor_correlations,
                risk_attribution=risk_attribution,
                return_attribution=return_attribution
            )
            
        except Exception as e:
            self.logger.error(f"Risk-return decomposition failed: {e}")
            raise
    
    async def principal_component_analysis(self, 
                                         return_data: pd.DataFrame,
                                         n_components: int = None) -> Dict[str, Any]:
        """Perform PCA on return data to identify principal factors."""
        try:
            # Prepare data
            clean_data = return_data.dropna()
            
            if len(clean_data) < 30:
                raise ValueError("Insufficient data for PCA")
            
            # Standardize data
            scaled_data = self.scaler.fit_transform(clean_data)
            
            # Determine number of components
            if n_components is None:
                n_components = min(10, len(clean_data.columns))
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(scaled_data)
            
            # Create results
            pc_df = pd.DataFrame(
                principal_components,
                index=clean_data.index,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            
            # Component loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                index=clean_data.columns,
                columns=pc_df.columns
            )
            
            # Explained variance
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            return {
                'principal_components': pc_df,
                'loadings': loadings,
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance': cumulative_variance,
                'eigenvalues': pca.explained_variance_,
                'n_components': n_components,
                'total_variance_explained': float(cumulative_variance[-1])
            }
            
        except Exception as e:
            self.logger.error(f"PCA analysis failed: {e}")
            raise
    
    async def factor_analysis(self, 
                            return_data: pd.DataFrame,
                            n_factors: int = None) -> Dict[str, Any]:
        """Perform factor analysis to identify latent factors."""
        try:
            # Prepare data
            clean_data = return_data.dropna()
            scaled_data = self.scaler.fit_transform(clean_data)
            
            # Determine number of factors
            if n_factors is None:
                n_factors = min(5, len(clean_data.columns) // 2)
            
            # Perform factor analysis
            fa = FactorAnalysis(n_components=n_factors, random_state=42)
            factor_scores = fa.fit_transform(scaled_data)
            
            # Create results
            factors_df = pd.DataFrame(
                factor_scores,
                index=clean_data.index,
                columns=[f'Factor{i+1}' for i in range(n_factors)]
            )
            
            # Factor loadings
            loadings = pd.DataFrame(
                fa.components_.T,
                index=clean_data.columns,
                columns=factors_df.columns
            )
            
            # Unique variances (specific to each variable)
            unique_variances = pd.Series(
                fa.noise_variance_,
                index=clean_data.columns,
                name='Unique_Variance'
            )
            
            return {
                'factor_scores': factors_df,
                'factor_loadings': loadings,
                'unique_variances': unique_variances,
                'n_factors': n_factors,
                'log_likelihood': fa.loglike_
            }
            
        except Exception as e:
            self.logger.error(f"Factor analysis failed: {e}")
            raise
    
    async def rolling_factor_analysis(self, 
                                    returns: pd.Series,
                                    factor_data: pd.DataFrame,
                                    window: int = 252,
                                    factors: List[str] = None) -> pd.DataFrame:
        """Perform rolling factor analysis."""
        try:
            results = []
            
            for i in range(window, len(returns)):
                # Get rolling window
                window_returns = returns.iloc[i-window:i]
                window_factors = factor_data.iloc[i-window:i]
                
                try:
                    # Build factor model for window
                    model = await self.build_factor_model(
                        window_returns, window_factors, 
                        f"rolling_{i}", factors
                    )
                    
                    # Extract key metrics
                    result_row = {
                        'date': returns.index[i-1],
                        'alpha': model.alpha,
                        'alpha_t_stat': model.alpha_t_stat,
                        'r_squared': model.r_squared,
                        'tracking_error': model.tracking_error,
                        'information_ratio': model.information_ratio
                    }
                    
                    # Add factor loadings
                    for factor_name, loading in model.factor_loadings.items():
                        result_row[f'beta_{factor_name}'] = loading.loading
                        result_row[f't_stat_{factor_name}'] = loading.t_statistic
                    
                    results.append(result_row)
                    
                except Exception as e:
                    self.logger.warning(f"Rolling analysis failed for window ending {returns.index[i-1]}: {e}")
                    continue
            
            return pd.DataFrame(results).set_index('date')
            
        except Exception as e:
            self.logger.error(f"Rolling factor analysis failed: {e}")
            raise
    
    def _validate_factor_inputs(self, returns: pd.Series, factor_data: pd.DataFrame):
        """Validate inputs for factor analysis."""
        if len(returns) < self.factor_config['min_observations']:
            raise ValueError(f"Insufficient data: need at least {self.factor_config['min_observations']} observations")
        
        if len(factor_data) < len(returns):
            raise ValueError("Factor data must have at least as many observations as returns")
        
        if returns.isna().any():
            raise ValueError("Returns contain NaN values")
        
        if factor_data.isna().any().any():
            raise ValueError("Factor data contains NaN values")
    
    def _align_factor_data(self, returns: pd.Series, 
                          factor_data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Align returns and factor data by index."""
        common_index = returns.index.intersection(factor_data.index)
        
        if len(common_index) < self.factor_config['min_observations']:
            raise ValueError("Insufficient overlapping data after alignment")
        
        return returns.loc[common_index], factor_data.loc[common_index]
    
    def _infer_factor_type(self, factor_name: str) -> FactorType:
        """Infer factor type from factor name."""
        name_lower = factor_name.lower()
        
        if 'market' in name_lower or 'mkt' in name_lower:
            return FactorType.MARKET
        elif 'smb' in name_lower or 'size' in name_lower:
            return FactorType.SIZE
        elif 'hml' in name_lower or 'value' in name_lower:
            return FactorType.VALUE
        elif 'momentum' in name_lower or 'mom' in name_lower:
            return FactorType.MOMENTUM
        elif 'rmw' in name_lower or 'quality' in name_lower:
            return FactorType.QUALITY
        elif 'cma' in name_lower or 'investment' in name_lower:
            return FactorType.INVESTMENT
        elif 'vol' in name_lower or 'volatility' in name_lower:
            return FactorType.VOLATILITY
        elif 'liquidity' in name_lower:
            return FactorType.LIQUIDITY
        elif 'sentiment' in name_lower:
            return FactorType.SENTIMENT
        elif any(x in name_lower for x in ['sector', 'industry', 'tech', 'finance', 'energy']):
            return FactorType.SECTOR
        elif any(x in name_lower for x in ['macro', 'gdp', 'inflation', 'rate']):
            return FactorType.MACRO
        else:
            return FactorType.CUSTOM
    
    def get_model_summary(self, model_name: str = None) -> Dict[str, Any]:
        """Get summary of factor models."""
        if model_name and model_name in self.factor_models:
            model = self.factor_models[model_name]
            return {
                'model_name': model.model_name,
                'factors': model.factors,
                'r_squared': model.r_squared,
                'alpha': model.alpha,
                'alpha_significance': 'significant' if model.alpha_p_value < 0.05 else 'not_significant',
                'tracking_error': model.tracking_error,
                'information_ratio': model.information_ratio,
                'observations': model.observations,
                'significant_factors': [
                    name for name, loading in model.factor_loadings.items()
                    if loading.significance_level in ['high', 'medium']
                ]
            }
        else:
            return {
                'total_models': len(self.factor_models),
                'model_names': list(self.factor_models.keys()),
                'last_updated': max([m.model_timestamp for m in self.factor_models.values()]) if self.factor_models else None
            }