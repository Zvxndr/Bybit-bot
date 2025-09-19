"""
Monte Carlo simulation for model validation and risk assessment.

This module provides comprehensive Monte Carlo simulation capabilities including:
- Bootstrap resampling
- Parametric and non-parametric simulations
- Path-dependent analysis
- Scenario generation
- Risk forecasting
- Model robustness testing
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import json
import warnings
from pathlib import Path

# Statistical libraries
from scipy import stats, optimize
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    # Simulation parameters
    n_simulations: int = 10000
    simulation_horizon: int = 252  # Trading days
    confidence_levels: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])
    
    # Bootstrap settings
    bootstrap_samples: int = 1000
    bootstrap_block_size: int = 20  # For block bootstrap
    replacement: bool = True
    
    # Parametric simulation
    use_parametric: bool = True
    distribution_type: str = "normal"  # normal, t, skewed_t
    estimate_parameters: bool = True
    
    # Non-parametric settings
    use_empirical: bool = True
    kernel_bandwidth: Optional[float] = None
    smoothing_factor: float = 1.0
    
    # Path generation
    use_path_dependency: bool = True
    correlation_model: str = "empirical"  # empirical, dcc, constant
    volatility_clustering: bool = True
    
    # Risk measures
    calculate_var: bool = True
    calculate_cvar: bool = True
    calculate_max_drawdown: bool = True
    stress_test_scenarios: bool = True
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
    seed: Optional[int] = 42
    
    # Advanced features
    regime_switching: bool = False
    jump_diffusion: bool = False
    stochastic_volatility: bool = False
    correlation_breakdown: bool = False

@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    simulation_id: str
    timestamp: datetime
    config: MonteCarloConfig
    
    # Path statistics
    final_values: np.ndarray
    path_statistics: Dict[str, np.ndarray]
    
    # Risk metrics
    var_estimates: Dict[float, float]
    cvar_estimates: Dict[float, float]
    max_drawdown_dist: np.ndarray
    
    # Performance metrics
    returns_dist: np.ndarray
    sharpe_dist: np.ndarray
    sortino_dist: np.ndarray
    
    # Statistical properties
    mean_return: float
    volatility: float
    skewness: float
    kurtosis: float
    
    # Confidence intervals
    confidence_intervals: Dict[str, Dict[float, Tuple[float, float]]]
    
    # Scenario analysis
    stress_scenarios: Dict[str, Dict[str, float]]
    tail_scenarios: List[Dict[str, Any]]
    
    # Model validation
    model_pvalues: Dict[str, float]
    goodness_of_fit: Dict[str, float]
    
    # Execution metrics
    execution_time: float
    memory_usage: float
    convergence_metrics: Dict[str, float]

class DistributionFitter:
    """Fit probability distributions to financial data."""
    
    def __init__(self):
        self.fitted_params = {}
        self.distribution_names = [
            'norm', 't', 'skewnorm', 'genextreme', 'gennorm'
        ]
    
    def fit_distribution(self, data: np.ndarray, 
                        distribution: str = 'auto') -> Dict[str, Any]:
        """Fit probability distribution to data."""
        try:
            data = data[~np.isnan(data)]
            
            if distribution == 'auto':
                best_dist, best_params, best_aic = self._find_best_distribution(data)
            else:
                if hasattr(stats, distribution):
                    dist = getattr(stats, distribution)
                    best_params = dist.fit(data)
                    best_dist = distribution
                    
                    # Calculate AIC
                    log_likelihood = np.sum(dist.logpdf(data, *best_params))
                    k = len(best_params)
                    best_aic = 2 * k - 2 * log_likelihood
                else:
                    raise ValueError(f"Unknown distribution: {distribution}")
            
            # Goodness of fit tests
            ks_stat, ks_pvalue = stats.kstest(data, 
                lambda x: getattr(stats, best_dist).cdf(x, *best_params))
            
            # Anderson-Darling test for normality
            if best_dist == 'norm':
                ad_stat, ad_critical, ad_significance = stats.anderson(data, dist='norm')
                ad_pvalue = self._ad_pvalue(ad_stat, ad_critical, ad_significance)
            else:
                ad_pvalue = np.nan
            
            result = {
                'distribution': best_dist,
                'parameters': best_params,
                'aic': best_aic,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'ad_pvalue': ad_pvalue,
                'data_moments': self._calculate_moments(data)
            }
            
            self.fitted_params[best_dist] = result
            return result
            
        except Exception as e:
            logger.error(f"Error fitting distribution: {e}")
            # Fallback to normal distribution
            return {
                'distribution': 'norm',
                'parameters': (np.mean(data), np.std(data)),
                'aic': np.inf,
                'ks_statistic': np.nan,
                'ks_pvalue': np.nan,
                'ad_pvalue': np.nan,
                'data_moments': self._calculate_moments(data)
            }
    
    def _find_best_distribution(self, data: np.ndarray) -> Tuple[str, tuple, float]:
        """Find best fitting distribution among candidates."""
        best_aic = np.inf
        best_dist = 'norm'
        best_params = (np.mean(data), np.std(data))
        
        for dist_name in self.distribution_names:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                
                # Calculate AIC
                log_likelihood = np.sum(dist.logpdf(data, *params))
                k = len(params)
                aic = 2 * k - 2 * log_likelihood
                
                if aic < best_aic:
                    best_aic = aic
                    best_dist = dist_name
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Could not fit {dist_name}: {e}")
                continue
        
        return best_dist, best_params, best_aic
    
    def _calculate_moments(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate statistical moments of data."""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'min': np.min(data),
            'max': np.max(data)
        }
    
    def _ad_pvalue(self, stat: float, critical: np.ndarray, 
                   significance: np.ndarray) -> float:
        """Approximate p-value for Anderson-Darling test."""
        if stat < critical[0]:
            return 1 - significance[0] / 100
        elif stat < critical[1]:
            return 1 - significance[1] / 100
        elif stat < critical[2]:
            return 1 - significance[2] / 100
        elif stat < critical[3]:
            return 1 - significance[3] / 100
        else:
            return 1 - significance[4] / 100

class PathGenerator:
    """Generate random paths for Monte Carlo simulation."""
    
    def __init__(self, config: MonteCarloConfig):
        self.config = config
        np.random.seed(config.seed)
        
    def generate_parametric_paths(self, initial_value: float,
                                 distribution_params: Dict[str, Any],
                                 n_paths: int,
                                 n_steps: int) -> np.ndarray:
        """Generate paths using parametric distribution."""
        try:
            dist_name = distribution_params['distribution']
            params = distribution_params['parameters']
            
            if dist_name == 'norm':
                mu, sigma = params
                innovations = np.random.normal(mu, sigma, (n_paths, n_steps))
            elif dist_name == 't':
                df, loc, scale = params
                innovations = np.random.standard_t(df, (n_paths, n_steps)) * scale + loc
            elif dist_name == 'skewnorm':
                a, loc, scale = params
                innovations = stats.skewnorm.rvs(a, loc, scale, (n_paths, n_steps))
            else:
                # Generic distribution
                dist = getattr(stats, dist_name)
                innovations = dist.rvs(*params, size=(n_paths, n_steps))
            
            # Generate cumulative paths
            if self.config.use_path_dependency:
                # Geometric Brownian Motion style
                paths = np.zeros((n_paths, n_steps + 1))
                paths[:, 0] = initial_value
                
                for t in range(n_steps):
                    paths[:, t + 1] = paths[:, t] * np.exp(innovations[:, t])
            else:
                # Simple cumulative sum
                paths = np.column_stack([
                    np.full(n_paths, initial_value),
                    initial_value + np.cumsum(innovations, axis=1)
                ])
            
            return paths
            
        except Exception as e:
            logger.error(f"Error generating parametric paths: {e}")
            # Fallback to simple random walk
            innovations = np.random.normal(0, 0.01, (n_paths, n_steps))
            paths = np.column_stack([
                np.full(n_paths, initial_value),
                initial_value + np.cumsum(innovations, axis=1)
            ])
            return paths
    
    def generate_bootstrap_paths(self, historical_data: np.ndarray,
                                initial_value: float,
                                n_paths: int,
                                n_steps: int) -> np.ndarray:
        """Generate paths using bootstrap resampling."""
        try:
            returns = np.diff(np.log(historical_data))
            
            if self.config.bootstrap_block_size > 1:
                # Block bootstrap for preserving time series structure
                sampled_returns = self._block_bootstrap(returns, n_paths, n_steps)
            else:
                # Simple bootstrap
                sampled_returns = np.random.choice(
                    returns, size=(n_paths, n_steps), 
                    replace=self.config.replacement
                )
            
            # Generate paths
            log_paths = np.log(initial_value) + np.column_stack([
                np.zeros(n_paths),
                np.cumsum(sampled_returns, axis=1)
            ])
            
            paths = np.exp(log_paths)
            return paths
            
        except Exception as e:
            logger.error(f"Error generating bootstrap paths: {e}")
            return np.full((n_paths, n_steps + 1), initial_value)
    
    def _block_bootstrap(self, data: np.ndarray, n_paths: int, 
                        n_steps: int) -> np.ndarray:
        """Perform block bootstrap resampling."""
        block_size = self.config.bootstrap_block_size
        n_blocks_needed = int(np.ceil(n_steps / block_size))
        
        sampled_returns = np.zeros((n_paths, n_steps))
        
        for path in range(n_paths):
            path_data = []
            
            for _ in range(n_blocks_needed):
                # Random starting point for block
                start_idx = np.random.randint(0, len(data) - block_size + 1)
                block = data[start_idx:start_idx + block_size]
                path_data.extend(block)
            
            # Trim to exact length needed
            sampled_returns[path, :] = path_data[:n_steps]
        
        return sampled_returns
    
    def generate_multivariate_paths(self, initial_values: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   mean_returns: np.ndarray,
                                   n_paths: int,
                                   n_steps: int) -> np.ndarray:
        """Generate correlated multivariate paths."""
        try:
            n_assets = len(initial_values)
            
            # Generate correlated innovations
            innovations = multivariate_normal.rvs(
                mean=mean_returns,
                cov=covariance_matrix,
                size=(n_paths, n_steps)
            )
            
            # Reshape for broadcasting
            if innovations.ndim == 2:
                innovations = innovations.reshape(1, n_steps, n_assets)
            else:
                innovations = innovations.reshape(n_paths, n_steps, n_assets)
            
            # Generate paths
            paths = np.zeros((n_paths, n_steps + 1, n_assets))
            paths[:, 0, :] = initial_values
            
            for t in range(n_steps):
                if self.config.use_path_dependency:
                    paths[:, t + 1, :] = paths[:, t, :] * np.exp(innovations[:, t, :])
                else:
                    paths[:, t + 1, :] = paths[:, t, :] + innovations[:, t, :]
            
            return paths
            
        except Exception as e:
            logger.error(f"Error generating multivariate paths: {e}")
            # Fallback to uncorrelated paths
            paths = np.zeros((n_paths, n_steps + 1, len(initial_values)))
            for i, initial_val in enumerate(initial_values):
                single_paths = self.generate_parametric_paths(
                    initial_val, 
                    {'distribution': 'norm', 'parameters': (mean_returns[i], np.sqrt(covariance_matrix[i, i]))},
                    n_paths, n_steps
                )
                paths[:, :, i] = single_paths
            
            return paths

class RiskCalculator:
    """Calculate risk metrics from simulation results."""
    
    def __init__(self, config: MonteCarloConfig):
        self.config = config
    
    def calculate_var(self, returns: np.ndarray, 
                     confidence_levels: List[float]) -> Dict[float, float]:
        """Calculate Value at Risk at different confidence levels."""
        var_results = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            var_results[confidence] = np.percentile(returns, alpha * 100)
        
        return var_results
    
    def calculate_cvar(self, returns: np.ndarray,
                      confidence_levels: List[float]) -> Dict[float, float]:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        cvar_results = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            var_threshold = np.percentile(returns, alpha * 100)
            tail_losses = returns[returns <= var_threshold]
            cvar_results[confidence] = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
        
        return cvar_results
    
    def calculate_max_drawdown_distribution(self, paths: np.ndarray) -> np.ndarray:
        """Calculate maximum drawdown distribution."""
        max_drawdowns = []
        
        for path in paths:
            # Calculate running maximum
            running_max = np.maximum.accumulate(path)
            
            # Calculate drawdowns
            drawdowns = (path - running_max) / running_max
            
            # Maximum drawdown for this path
            max_drawdown = np.min(drawdowns)
            max_drawdowns.append(max_drawdown)
        
        return np.array(max_drawdowns)
    
    def calculate_performance_metrics(self, paths: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate performance metrics for each path."""
        results = {}
        
        # Calculate returns for each path
        returns_paths = np.diff(paths, axis=1) / paths[:, :-1]
        
        # Total returns
        total_returns = (paths[:, -1] / paths[:, 0]) - 1
        results['total_returns'] = total_returns
        
        # Annualized returns
        n_periods = paths.shape[1] - 1
        annualized_returns = (1 + total_returns) ** (252 / n_periods) - 1
        results['annualized_returns'] = annualized_returns
        
        # Volatility
        volatilities = np.std(returns_paths, axis=1) * np.sqrt(252)
        results['volatilities'] = volatilities
        
        # Sharpe ratios
        excess_returns = annualized_returns - self.config.risk_free_rate if hasattr(self.config, 'risk_free_rate') else annualized_returns
        sharpe_ratios = excess_returns / volatilities
        results['sharpe_ratios'] = sharpe_ratios
        
        # Sortino ratios
        downside_returns = np.where(returns_paths < 0, returns_paths, 0)
        downside_volatilities = np.std(downside_returns, axis=1) * np.sqrt(252)
        sortino_ratios = np.where(downside_volatilities > 0, 
                                 excess_returns / downside_volatilities, 
                                 np.inf)
        results['sortino_ratios'] = sortino_ratios
        
        return results
    
    def generate_stress_scenarios(self, base_paths: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Generate stress test scenarios."""
        scenarios = {}
        
        # Market crash scenario (-30% shock)
        crash_paths = base_paths * 0.7
        crash_metrics = self.calculate_performance_metrics(crash_paths)
        scenarios['market_crash'] = {
            'mean_return': np.mean(crash_metrics['total_returns']),
            'worst_case': np.min(crash_metrics['total_returns']),
            'volatility': np.mean(crash_metrics['volatilities'])
        }
        
        # High volatility scenario (2x volatility)
        high_vol_returns = np.diff(base_paths, axis=1) / base_paths[:, :-1]
        high_vol_returns *= 2
        high_vol_paths = base_paths[:, 0:1] * np.cumprod(1 + high_vol_returns, axis=1)
        high_vol_paths = np.column_stack([base_paths[:, 0], high_vol_paths])
        high_vol_metrics = self.calculate_performance_metrics(high_vol_paths)
        scenarios['high_volatility'] = {
            'mean_return': np.mean(high_vol_metrics['total_returns']),
            'worst_case': np.min(high_vol_metrics['total_returns']),
            'volatility': np.mean(high_vol_metrics['volatilities'])
        }
        
        # Correlation breakdown (uncorrelated assets)
        # This would be relevant for multi-asset simulations
        scenarios['correlation_breakdown'] = {
            'mean_return': np.mean(base_paths[:, -1] / base_paths[:, 0] - 1),
            'worst_case': np.min(base_paths[:, -1] / base_paths[:, 0] - 1),
            'volatility': np.std((base_paths[:, -1] / base_paths[:, 0] - 1)) * np.sqrt(252)
        }
        
        return scenarios

class MonteCarloSimulator:
    """Main Monte Carlo simulation engine."""
    
    def __init__(self, config: MonteCarloConfig, db_path: Optional[str] = None):
        self.config = config
        self.db_path = db_path or "monte_carlo_results.db"
        self.distribution_fitter = DistributionFitter()
        self.path_generator = PathGenerator(config)
        self.risk_calculator = RiskCalculator(config)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database for storing results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Simulation results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monte_carlo_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    mean_return REAL,
                    volatility REAL,
                    skewness REAL,
                    kurtosis REAL,
                    var_95 REAL,
                    cvar_95 REAL,
                    max_drawdown_mean REAL,
                    max_drawdown_worst REAL,
                    sharpe_mean REAL,
                    sortino_mean REAL,
                    execution_time REAL,
                    n_simulations INTEGER,
                    convergence_achieved INTEGER
                )
            """)
            
            # Path data table (optional, for detailed analysis)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulation_paths (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id TEXT NOT NULL,
                    path_index INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    value REAL NOT NULL
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def run_simulation(self, data: Union[pd.DataFrame, np.ndarray],
                      initial_value: float = 100.0,
                      simulation_id: Optional[str] = None) -> SimulationResult:
        """Run Monte Carlo simulation."""
        start_time = datetime.now()
        simulation_id = simulation_id or f"mc_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting Monte Carlo simulation: {simulation_id}")
        
        try:
            # Prepare data
            if isinstance(data, pd.DataFrame):
                if 'returns' in data.columns:
                    returns_data = data['returns'].dropna().values
                else:
                    # Assume first column is price data
                    price_data = data.iloc[:, 0].dropna().values
                    returns_data = np.diff(np.log(price_data))
            else:
                returns_data = data
            
            # Fit distribution
            dist_params = self.distribution_fitter.fit_distribution(
                returns_data, self.config.distribution_type
            )
            
            # Generate simulation paths
            if self.config.parallel_processing:
                paths = self._run_parallel_simulation(
                    returns_data, dist_params, initial_value
                )
            else:
                paths = self._run_sequential_simulation(
                    returns_data, dist_params, initial_value
                )
            
            # Calculate metrics
            performance_metrics = self.risk_calculator.calculate_performance_metrics(paths)
            
            # Risk measures
            final_returns = (paths[:, -1] / paths[:, 0]) - 1
            var_estimates = self.risk_calculator.calculate_var(
                final_returns, self.config.confidence_levels
            )
            cvar_estimates = self.risk_calculator.calculate_cvar(
                final_returns, self.config.confidence_levels
            )
            
            # Maximum drawdown distribution
            max_drawdown_dist = self.risk_calculator.calculate_max_drawdown_distribution(paths)
            
            # Statistical properties
            mean_return = np.mean(final_returns)
            volatility = np.std(final_returns)
            skewness = stats.skew(final_returns)
            kurtosis = stats.kurtosis(final_returns)
            
            # Confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(performance_metrics)
            
            # Stress scenarios
            stress_scenarios = self.risk_calculator.generate_stress_scenarios(paths)
            
            # Tail scenario analysis
            tail_scenarios = self._analyze_tail_scenarios(paths, final_returns)
            
            # Model validation
            model_pvalues = self._validate_simulation_model(final_returns, dist_params)
            
            # Convergence metrics
            convergence_metrics = self._assess_convergence(paths)
            
            # Create result object
            result = SimulationResult(
                simulation_id=simulation_id,
                timestamp=start_time,
                config=self.config,
                final_values=paths[:, -1],
                path_statistics=performance_metrics,
                var_estimates=var_estimates,
                cvar_estimates=cvar_estimates,
                max_drawdown_dist=max_drawdown_dist,
                returns_dist=final_returns,
                sharpe_dist=performance_metrics['sharpe_ratios'],
                sortino_dist=performance_metrics['sortino_ratios'],
                mean_return=mean_return,
                volatility=volatility,
                skewness=skewness,
                kurtosis=kurtosis,
                confidence_intervals=confidence_intervals,
                stress_scenarios=stress_scenarios,
                tail_scenarios=tail_scenarios,
                model_pvalues=model_pvalues,
                goodness_of_fit=dist_params,
                execution_time=(datetime.now() - start_time).total_seconds(),
                memory_usage=0.0,  # Placeholder
                convergence_metrics=convergence_metrics
            )
            
            # Save results
            self._save_results(result)
            
            logger.info(f"Monte Carlo simulation completed in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            raise
    
    def _run_parallel_simulation(self, returns_data: np.ndarray,
                               dist_params: Dict[str, Any],
                               initial_value: float) -> np.ndarray:
        """Run parallel Monte Carlo simulation."""
        chunk_size = self.config.chunk_size
        n_chunks = int(np.ceil(self.config.n_simulations / chunk_size))
        
        all_paths = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for i in range(n_chunks):
                chunk_start = i * chunk_size
                chunk_end = min((i + 1) * chunk_size, self.config.n_simulations)
                chunk_n_sims = chunk_end - chunk_start
                
                if self.config.use_parametric:
                    future = executor.submit(
                        self.path_generator.generate_parametric_paths,
                        initial_value, dist_params, chunk_n_sims, self.config.simulation_horizon
                    )
                else:
                    future = executor.submit(
                        self.path_generator.generate_bootstrap_paths,
                        np.exp(np.cumsum(np.concatenate([[0], returns_data]))),
                        initial_value, chunk_n_sims, self.config.simulation_horizon
                    )
                
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    chunk_paths = future.result()
                    all_paths.append(chunk_paths)
                except Exception as e:
                    logger.error(f"Error in parallel chunk: {e}")
        
        # Combine all chunks
        if all_paths:
            combined_paths = np.vstack(all_paths)
        else:
            # Fallback
            combined_paths = self._run_sequential_simulation(returns_data, dist_params, initial_value)
        
        return combined_paths
    
    def _run_sequential_simulation(self, returns_data: np.ndarray,
                                 dist_params: Dict[str, Any],
                                 initial_value: float) -> np.ndarray:
        """Run sequential Monte Carlo simulation."""
        if self.config.use_parametric:
            paths = self.path_generator.generate_parametric_paths(
                initial_value, dist_params, 
                self.config.n_simulations, self.config.simulation_horizon
            )
        else:
            # Convert returns to price series for bootstrap
            price_series = np.exp(np.cumsum(np.concatenate([[0], returns_data])))
            paths = self.path_generator.generate_bootstrap_paths(
                price_series, initial_value,
                self.config.n_simulations, self.config.simulation_horizon
            )
        
        return paths
    
    def _calculate_confidence_intervals(self, metrics: Dict[str, np.ndarray]) -> Dict[str, Dict[float, Tuple[float, float]]]:
        """Calculate confidence intervals for performance metrics."""
        confidence_intervals = {}
        
        for metric_name, values in metrics.items():
            metric_cis = {}
            
            for confidence in self.config.confidence_levels:
                alpha = 1 - confidence
                lower = np.percentile(values, (alpha / 2) * 100)
                upper = np.percentile(values, (1 - alpha / 2) * 100)
                metric_cis[confidence] = (lower, upper)
            
            confidence_intervals[metric_name] = metric_cis
        
        return confidence_intervals
    
    def _analyze_tail_scenarios(self, paths: np.ndarray, 
                              returns: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze extreme tail scenarios."""
        tail_scenarios = []
        
        # Find worst performing paths
        worst_indices = np.argsort(returns)[:10]  # Top 10 worst scenarios
        
        for i, idx in enumerate(worst_indices):
            scenario = {
                'rank': i + 1,
                'final_return': returns[idx],
                'path_volatility': np.std(np.diff(paths[idx]) / paths[idx, :-1]) * np.sqrt(252),
                'max_drawdown': np.min((paths[idx] - np.maximum.accumulate(paths[idx])) / np.maximum.accumulate(paths[idx])),
                'path_index': int(idx)
            }
            tail_scenarios.append(scenario)
        
        return tail_scenarios
    
    def _validate_simulation_model(self, simulated_returns: np.ndarray,
                                 dist_params: Dict[str, Any]) -> Dict[str, float]:
        """Validate simulation model against theoretical distribution."""
        try:
            # Kolmogorov-Smirnov test
            dist_name = dist_params['distribution']
            params = dist_params['parameters']
            
            dist = getattr(stats, dist_name)
            ks_stat, ks_pvalue = stats.kstest(simulated_returns, 
                lambda x: dist.cdf(x, *params))
            
            # Shapiro-Wilk test for normality (if applicable)
            if dist_name == 'norm' and len(simulated_returns) <= 5000:
                sw_stat, sw_pvalue = stats.shapiro(simulated_returns[:5000])
            else:
                sw_stat, sw_pvalue = np.nan, np.nan
            
            # Jarque-Bera test
            jb_stat, jb_pvalue = stats.jarque_bera(simulated_returns)
            
            return {
                'ks_pvalue': ks_pvalue,
                'shapiro_wilk_pvalue': sw_pvalue,
                'jarque_bera_pvalue': jb_pvalue
            }
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return {'ks_pvalue': np.nan, 'shapiro_wilk_pvalue': np.nan, 'jarque_bera_pvalue': np.nan}
    
    def _assess_convergence(self, paths: np.ndarray) -> Dict[str, float]:
        """Assess simulation convergence."""
        final_returns = (paths[:, -1] / paths[:, 0]) - 1
        n_sims = len(final_returns)
        
        # Calculate running statistics
        batch_size = max(100, n_sims // 10)
        running_means = []
        running_stds = []
        
        for i in range(batch_size, n_sims + 1, batch_size):
            batch_returns = final_returns[:i]
            running_means.append(np.mean(batch_returns))
            running_stds.append(np.std(batch_returns))
        
        # Convergence metrics
        if len(running_means) > 1:
            mean_stability = 1.0 - np.std(running_means) / (abs(np.mean(running_means)) + 1e-8)
            std_stability = 1.0 - np.std(running_stds) / (np.mean(running_stds) + 1e-8)
        else:
            mean_stability = 1.0
            std_stability = 1.0
        
        # Monte Carlo standard error
        mc_error = np.std(final_returns) / np.sqrt(n_sims)
        
        return {
            'mean_stability': max(0.0, mean_stability),
            'std_stability': max(0.0, std_stability),
            'monte_carlo_error': mc_error,
            'effective_sample_size': n_sims
        }
    
    def _save_results(self, result: SimulationResult):
        """Save simulation results to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO monte_carlo_results (
                    simulation_id, timestamp, config_json, mean_return, volatility,
                    skewness, kurtosis, var_95, cvar_95, max_drawdown_mean,
                    max_drawdown_worst, sharpe_mean, sortino_mean, execution_time,
                    n_simulations, convergence_achieved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.simulation_id,
                result.timestamp.isoformat(),
                json.dumps(result.config.__dict__),
                result.mean_return,
                result.volatility,
                result.skewness,
                result.kurtosis,
                result.var_estimates.get(0.95, np.nan),
                result.cvar_estimates.get(0.95, np.nan),
                np.mean(result.max_drawdown_dist),
                np.min(result.max_drawdown_dist),
                np.mean(result.sharpe_dist),
                np.mean(result.sortino_dist),
                result.execution_time,
                self.config.n_simulations,
                1 if result.convergence_metrics['mean_stability'] > 0.95 else 0
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def plot_results(self, result: SimulationResult, save_path: Optional[str] = None):
        """Plot simulation results."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Monte Carlo Simulation Results - {result.simulation_id}', fontsize=16)
            
            # Returns distribution
            axes[0, 0].hist(result.returns_dist, bins=50, alpha=0.7, density=True)
            axes[0, 0].set_title('Returns Distribution')
            axes[0, 0].set_xlabel('Returns')
            axes[0, 0].set_ylabel('Density')
            
            # Sharpe ratio distribution
            axes[0, 1].hist(result.sharpe_dist, bins=50, alpha=0.7, density=True)
            axes[0, 1].set_title('Sharpe Ratio Distribution')
            axes[0, 1].set_xlabel('Sharpe Ratio')
            axes[0, 1].set_ylabel('Density')
            
            # Maximum drawdown distribution
            axes[0, 2].hist(result.max_drawdown_dist, bins=50, alpha=0.7, density=True)
            axes[0, 2].set_title('Maximum Drawdown Distribution')
            axes[0, 2].set_xlabel('Max Drawdown')
            axes[0, 2].set_ylabel('Density')
            
            # VaR comparison
            conf_levels = list(result.var_estimates.keys())
            var_values = list(result.var_estimates.values())
            cvar_values = list(result.cvar_estimates.values())
            
            x = np.arange(len(conf_levels))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, var_values, width, label='VaR', alpha=0.7)
            axes[1, 0].bar(x + width/2, cvar_values, width, label='CVaR', alpha=0.7)
            axes[1, 0].set_title('Risk Measures')
            axes[1, 0].set_xlabel('Confidence Level')
            axes[1, 0].set_ylabel('Risk Measure')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([f'{c:.1%}' for c in conf_levels])
            axes[1, 0].legend()
            
            # Sample paths
            sample_indices = np.random.choice(len(result.final_values), 
                                            min(100, len(result.final_values)), replace=False)
            
            # This would require storing path data, simplified for now
            axes[1, 1].set_title('Sample Paths (Placeholder)')
            axes[1, 1].set_xlabel('Time Steps')
            axes[1, 1].set_ylabel('Value')
            
            # Performance metrics summary
            metrics_text = f"""
            Mean Return: {result.mean_return:.4f}
            Volatility: {result.volatility:.4f}
            Skewness: {result.skewness:.4f}
            Kurtosis: {result.kurtosis:.4f}
            VaR (95%): {result.var_estimates.get(0.95, 'N/A'):.4f}
            CVaR (95%): {result.cvar_estimates.get(0.95, 'N/A'):.4f}
            """
            
            axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes, 
                           fontsize=12, verticalalignment='center')
            axes[1, 2].set_title('Summary Statistics')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting results: {e}")

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    returns = np.random.normal(0.001, 0.02, n_samples)
    
    # Add some realistic features
    for i in range(1, n_samples):
        returns[i] += 0.05 * returns[i-1] + np.random.normal(0, 0.015)
    
    # Run Monte Carlo simulation
    config = MonteCarloConfig(
        n_simulations=5000,
        simulation_horizon=252,
        confidence_levels=[0.90, 0.95, 0.99],
        parallel_processing=False
    )
    
    simulator = MonteCarloSimulator(config)
    
    try:
        result = simulator.run_simulation(returns, initial_value=100.0)
        
        print("\n=== Monte Carlo Simulation Results ===")
        print(f"Simulation ID: {result.simulation_id}")
        print(f"Number of simulations: {config.n_simulations}")
        print(f"Mean return: {result.mean_return:.4f}")
        print(f"Volatility: {result.volatility:.4f}")
        print(f"Skewness: {result.skewness:.4f}")
        print(f"Kurtosis: {result.kurtosis:.4f}")
        
        print("\n--- Risk Measures ---")
        for conf, var_val in result.var_estimates.items():
            cvar_val = result.cvar_estimates[conf]
            print(f"VaR ({conf:.1%}): {var_val:.4f}")
            print(f"CVaR ({conf:.1%}): {cvar_val:.4f}")
        
        print(f"\nMax Drawdown (mean): {np.mean(result.max_drawdown_dist):.4f}")
        print(f"Max Drawdown (worst): {np.min(result.max_drawdown_dist):.4f}")
        
        print(f"\nExecution time: {result.execution_time:.2f} seconds")
        
        # Plot results
        simulator.plot_results(result)
        
    except Exception as e:
        logger.error(f"Error in example: {e}")