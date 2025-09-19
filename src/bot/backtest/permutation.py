"""
Permutation Testing for strategy validation.

Permutation testing is a non-parametric statistical method to assess whether
a trading strategy's performance is significantly different from random chance.
This implementation provides:

- Multiple permutation methods (shuffle returns, bootstrap, monte carlo)
- Statistical significance testing
- Multiple test correction (Bonferroni, Benjamini-Hochberg)
- Visualization of permutation distributions
- Robust handling of multiple testing scenarios

The process:
1. Calculate strategy performance on actual data
2. Generate many random permutations of the data
3. Calculate performance on each permutation
4. Compare actual performance to permutation distribution
5. Calculate p-value and statistical significance
"""

import warnings
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ..utils.logging import TradingLogger


class PermutationResult:
    """Container for permutation testing results."""
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.actual_value: float = 0.0
        self.permutation_values: List[float] = []
        self.p_value: float = 1.0
        self.significant: bool = False
        self.percentile_rank: float = 0.0
        self.z_score: float = 0.0
        self.confidence_interval: Tuple[float, float] = (0.0, 0.0)
        self.distribution_stats: Dict[str, float] = {}
    
    def calculate_statistics(self, confidence_level: float = 0.95):
        """Calculate statistical measures from permutation results."""
        if not self.permutation_values:
            return
        
        perm_array = np.array(self.permutation_values)
        
        # Calculate p-value (two-tailed)
        n_better = np.sum(perm_array >= self.actual_value)
        n_worse = np.sum(perm_array <= self.actual_value)
        self.p_value = float(2 * min(n_better, n_worse) / len(perm_array))
        
        # Calculate percentile rank
        self.percentile_rank = float(stats.percentileofscore(perm_array, self.actual_value) / 100)
        
        # Calculate z-score
        perm_mean = np.mean(perm_array)
        perm_std = np.std(perm_array)
        if perm_std > 0:
            self.z_score = float((self.actual_value - perm_mean) / perm_std)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        self.confidence_interval = (
            float(np.percentile(perm_array, lower_percentile)),
            float(np.percentile(perm_array, upper_percentile))
        )
        
        # Distribution statistics
        self.distribution_stats = {
            'mean': float(perm_mean),
            'std': float(perm_std),
            'min': float(np.min(perm_array)),
            'max': float(np.max(perm_array)),
            'median': float(np.median(perm_array)),
            'skewness': float(stats.skew(perm_array)),
            'kurtosis': float(stats.kurtosis(perm_array))
        }
        
        # Significance test
        self.significant = self.p_value < 0.05


class PermutationTester:
    """
    Comprehensive permutation testing for trading strategy validation.
    
    This class provides multiple permutation methods to test whether
    a strategy's performance is statistically significant compared to
    random chance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("PermutationTester")
        
        # Set up multiprocessing
        self.n_cores = min(self.config['max_workers'], mp.cpu_count())
    
    def _default_config(self) -> Dict:
        """Default configuration for permutation testing."""
        return {
            'n_permutations': 10000,    # Number of permutations
            'confidence_level': 0.95,   # Confidence level for CI
            'significance_level': 0.05, # Alpha for significance testing
            'method': 'shuffle_returns', # Permutation method
            'max_workers': 4,           # Parallel processing workers
            'bootstrap_block_size': 21, # Block size for block bootstrap
            'seed': 42,                 # Random seed for reproducibility
            'correction_method': 'bonferroni', # Multiple testing correction
        }
    
    def test_strategy_significance(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        strategy_params: Dict,
        metrics_to_test: Optional[List[str]] = None,
        method: Optional[str] = None
    ) -> Dict[str, PermutationResult]:
        """
        Test statistical significance of strategy performance.
        
        Args:
            data: Historical market data
            strategy_func: Strategy function that returns performance metrics
            strategy_params: Parameters for the strategy
            metrics_to_test: List of metrics to test (default: ['sharpe_ratio', 'total_return'])
            method: Permutation method override
            
        Returns:
            Dictionary of PermutationResult objects for each metric
        """
        if metrics_to_test is None:
            metrics_to_test = ['sharpe_ratio', 'total_return', 'max_drawdown']
        
        method = method or self.config['method']
        
        self.logger.info(
            f"Starting permutation testing with {self.config['n_permutations']} permutations "
            f"using {method} method"
        )
        
        # Calculate actual strategy performance
        try:
            actual_metrics = strategy_func(data, **strategy_params)
        except Exception as e:
            self.logger.error(f"Failed to calculate actual strategy metrics: {e}")
            raise
        
        # Validate that required metrics are present
        for metric in metrics_to_test:
            if metric not in actual_metrics:
                raise ValueError(f"Metric '{metric}' not found in strategy results")
        
        # Initialize results
        results = {}
        for metric in metrics_to_test:
            result = PermutationResult(metric)
            result.actual_value = actual_metrics[metric]
            results[metric] = result
        
        # Generate permutations and calculate performance
        permutation_metrics = self._run_permutations(
            data, strategy_func, strategy_params, method
        )
        
        # Process results for each metric
        for metric in metrics_to_test:
            metric_values = [perm_result[metric] for perm_result in permutation_metrics 
                           if metric in perm_result and not np.isnan(perm_result[metric])]
            
            if not metric_values:
                self.logger.warning(f"No valid permutation results for metric '{metric}'")
                continue
            
            results[metric].permutation_values = metric_values
            results[metric].calculate_statistics(self.config['confidence_level'])
            
            self.logger.info(
                f"Metric '{metric}': Actual={results[metric].actual_value:.4f}, "
                f"p-value={results[metric].p_value:.4f}, "
                f"Significant={results[metric].significant}"
            )
        
        # Apply multiple testing correction
        results = self._apply_multiple_testing_correction(results)
        
        return results
    
    def _run_permutations(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        strategy_params: Dict,
        method: str
    ) -> List[Dict]:
        """Run permutation tests using parallel processing."""
        n_permutations = self.config['n_permutations']
        
        # Create permutation tasks
        tasks = []
        np.random.seed(self.config['seed'])
        seeds = np.random.randint(0, 100000, n_permutations)
        
        for i in range(n_permutations):
            tasks.append((data.copy(), strategy_func, strategy_params, method, seeds[i]))
        
        # Run permutations in parallel
        results = []
        completed = 0
        
        if self.n_cores > 1:
            with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                future_to_task = {
                    executor.submit(self._single_permutation, *task): task 
                    for task in tasks
                }
                
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                        completed += 1
                        
                        if completed % 1000 == 0:
                            self.logger.debug(f"Completed {completed}/{n_permutations} permutations")
                            
                    except Exception as e:
                        self.logger.debug(f"Permutation failed: {e}")
                        completed += 1
        else:
            # Single-threaded execution
            for task in tasks:
                try:
                    result = self._single_permutation(*task)
                    if result is not None:
                        results.append(result)
                    completed += 1
                    
                    if completed % 1000 == 0:
                        self.logger.debug(f"Completed {completed}/{n_permutations} permutations")
                        
                except Exception as e:
                    self.logger.debug(f"Permutation failed: {e}")
                    completed += 1
        
        self.logger.info(f"Completed {len(results)}/{n_permutations} successful permutations")
        return results
    
    @staticmethod
    def _single_permutation(
        data: pd.DataFrame,
        strategy_func: Callable,
        strategy_params: Dict,
        method: str,
        seed: int
    ) -> Optional[Dict]:
        """Execute a single permutation test."""
        try:
            np.random.seed(seed)
            
            # Create permuted data based on method
            if method == 'shuffle_returns':
                permuted_data = PermutationTester._shuffle_returns(data)
            elif method == 'shuffle_prices':
                permuted_data = PermutationTester._shuffle_prices(data)
            elif method == 'block_bootstrap':
                permuted_data = PermutationTester._block_bootstrap(data)
            elif method == 'monte_carlo':
                permuted_data = PermutationTester._monte_carlo_simulation(data)
            else:
                raise ValueError(f"Unknown permutation method: {method}")
            
            # Calculate strategy performance on permuted data
            metrics = strategy_func(permuted_data, **strategy_params)
            return metrics
            
        except Exception:
            return None
    
    @staticmethod
    def _shuffle_returns(data: pd.DataFrame) -> pd.DataFrame:
        """Shuffle returns while preserving price structure."""
        data = data.copy()
        
        # Calculate returns
        data['returns'] = data['close'].pct_change().fillna(0)
        
        # Shuffle returns
        returns = data['returns'].values[1:]  # Skip first NaN
        np.random.shuffle(np.array(returns))
        
        # Reconstruct prices from shuffled returns
        new_prices = [data['close'].iloc[0]]  # Start with original first price
        for ret in returns:
            new_prices.append(new_prices[-1] * (1 + ret))
        
        # Update OHLCV with new prices (simplified - same relative structure)
        price_ratio = np.array(new_prices) / data['close'].values
        data['open'] *= price_ratio
        data['high'] *= price_ratio
        data['low'] *= price_ratio
        data['close'] = new_prices
        
        return data.drop('returns', axis=1)
    
    @staticmethod
    def _shuffle_prices(data: pd.DataFrame) -> pd.DataFrame:
        """Shuffle price movements while maintaining OHLCV relationships."""
        data = data.copy()
        
        # Calculate price changes
        price_changes = data['close'].diff().fillna(0).values[1:]
        np.random.shuffle(np.array(price_changes))
        
        # Reconstruct prices
        new_closes = [data['close'].iloc[0]]
        for change in price_changes:
            new_closes.append(max(0.001, new_closes[-1] + change))  # Prevent negative prices
        
        # Update all OHLCV maintaining relationships
        close_ratio = np.array(new_closes) / data['close'].values
        data['open'] *= close_ratio
        data['high'] *= close_ratio
        data['low'] *= close_ratio
        data['close'] = new_closes
        
        return data
    
    @staticmethod
    def _block_bootstrap(data: pd.DataFrame, block_size: int = 21) -> pd.DataFrame:
        """Block bootstrap resampling to preserve temporal dependencies."""
        n_rows = len(data)
        n_blocks = int(np.ceil(n_rows / block_size))
        
        # Create blocks
        blocks = []
        for i in range(n_rows - block_size + 1):
            blocks.append(data.iloc[i:i + block_size])
        
        # Sample blocks with replacement
        sampled_data = []
        for _ in range(n_blocks):
            block = np.random.choice(blocks)
            sampled_data.append(block)
        
        # Concatenate and trim to original length
        result = pd.concat(sampled_data, ignore_index=True)
        return result.iloc[:n_rows]
    
    @staticmethod
    def _monte_carlo_simulation(data: pd.DataFrame) -> pd.DataFrame:
        """Monte Carlo simulation with fitted return distribution."""
        data = data.copy()
        
        # Calculate returns and fit distribution
        returns = data['close'].pct_change().dropna()
        
        # Fit normal distribution to returns
        mu, sigma = returns.mean(), returns.std()
        
        # Generate random returns
        n_periods = len(data)
        random_returns = np.random.normal(mu, sigma, n_periods - 1)
        
        # Reconstruct prices
        new_closes = [data['close'].iloc[0]]
        for ret in random_returns:
            new_closes.append(new_closes[-1] * (1 + ret))
        
        # Update OHLCV
        close_ratio = np.array(new_closes) / data['close'].values
        data['open'] *= close_ratio
        data['high'] *= close_ratio  
        data['low'] *= close_ratio
        data['close'] = new_closes
        
        return data
    
    def _apply_multiple_testing_correction(
        self, 
        results: Dict[str, PermutationResult]
    ) -> Dict[str, PermutationResult]:
        """Apply multiple testing correction to p-values."""
        if len(results) <= 1:
            return results
        
        method = self.config['correction_method']
        p_values = [result.p_value for result in results.values()]
        metric_names = list(results.keys())
        
        if method == 'bonferroni':
            corrected_p_values = np.array(p_values) * len(p_values)
            corrected_p_values = np.minimum(corrected_p_values, 1.0)
        elif method == 'benjamini_hochberg':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            sorted_p_values = np.array(p_values)[sorted_indices]
            
            corrected_p_values = np.zeros_like(sorted_p_values)
            for i in range(len(sorted_p_values) - 1, -1, -1):
                if i == len(sorted_p_values) - 1:
                    corrected_p_values[i] = sorted_p_values[i]
                else:
                    corrected_p_values[i] = min(
                        sorted_p_values[i] * len(sorted_p_values) / (i + 1),
                        corrected_p_values[i + 1]
                    )
            
            # Unsort the corrected p-values
            unsorted_corrected = np.zeros_like(corrected_p_values)
            unsorted_corrected[sorted_indices] = corrected_p_values
            corrected_p_values = unsorted_corrected
        else:
            # No correction
            corrected_p_values = p_values
        
        # Update results with corrected p-values
        for i, metric_name in enumerate(metric_names):
            results[metric_name].p_value = corrected_p_values[i]
            results[metric_name].significant = (
                corrected_p_values[i] < self.config['significance_level']
            )
        
        return results
    
    def generate_report(self, results: Dict[str, PermutationResult]) -> str:
        """Generate a comprehensive permutation testing report."""
        report = f"""
Permutation Testing Report
{'='*50}

Configuration:
- Number of Permutations: {self.config['n_permutations']:,}
- Method: {self.config['method']}
- Significance Level: {self.config['significance_level']}
- Correction Method: {self.config['correction_method']}

Results:
"""
        
        for metric_name, result in results.items():
            report += f"""
{metric_name}:
- Actual Value: {result.actual_value:.6f}
- P-Value: {result.p_value:.6f}
- Significant: {'YES' if result.significant else 'NO'}
- Percentile Rank: {result.percentile_rank:.1%}
- Z-Score: {result.z_score:.3f}
- 95% CI: [{result.confidence_interval[0]:.6f}, {result.confidence_interval[1]:.6f}]
- Permutation Mean: {result.distribution_stats.get('mean', 0):.6f}
- Permutation Std: {result.distribution_stats.get('std', 0):.6f}
"""
        
        return report