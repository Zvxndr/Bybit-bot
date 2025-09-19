"""
Walk-Forward Analysis (WFO) implementation for strategy validation.

Walk-Forward Analysis is a robust method for validating trading strategies by
using rolling windows of historical data. This implementation provides:

- Expanding and rolling window approaches
- Out-of-sample performance tracking
- Parameter optimization with warm start
- Statistical significance testing
- Mode-specific validation criteria

The WFO process:
1. Split data into training and testing periods
2. Optimize strategy parameters on training data
3. Test strategy on out-of-sample data
4. Roll forward and repeat
5. Aggregate results for statistical analysis
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import ParameterGrid
from loguru import logger

from ..utils.logging import TradingLogger


class WalkForwardResult:
    """Container for Walk-Forward Analysis results."""
    
    def __init__(self):
        self.periods: List[Dict] = []
        self.oos_returns: List[float] = []
        self.is_returns: List[float] = []
        self.parameters: List[Dict] = []
        self.metrics: Dict[str, List[float]] = {}
        self.equity_curve: pd.DataFrame = pd.DataFrame()
        self.summary_stats: Dict[str, float] = {}
    
    def add_period(
        self,
        start_date: datetime,
        end_date: datetime,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime, 
        test_end: datetime,
        is_return: float,
        oos_return: float,
        best_params: Dict,
        metrics: Dict[str, float]
    ):
        """Add results from a single WFO period."""
        period_data = {
            'period_start': start_date,
            'period_end': end_date,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'is_return': is_return,
            'oos_return': oos_return,
            'best_params': best_params,
            'metrics': metrics
        }
        
        self.periods.append(period_data)
        self.is_returns.append(is_return)
        self.oos_returns.append(oos_return)
        self.parameters.append(best_params)
        
        # Add metrics to tracking
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(value)
    
    def calculate_summary_stats(self):
        """Calculate summary statistics for the WFO results."""
        if not self.oos_returns:
            return
        
        oos_returns = np.array(self.oos_returns)
        is_returns = np.array(self.is_returns)
        
        # Basic return statistics
        self.summary_stats.update({
            'total_periods': len(self.oos_returns),
            'oos_mean_return': np.mean(oos_returns),
            'oos_std_return': np.std(oos_returns),
            'oos_sharpe_ratio': np.mean(oos_returns) / np.std(oos_returns) if np.std(oos_returns) > 0 else 0,
            'oos_cumulative_return': np.prod(1 + oos_returns) - 1,
            'is_mean_return': np.mean(is_returns),
            'is_std_return': np.std(is_returns),
            'is_sharpe_ratio': np.mean(is_returns) / np.std(is_returns) if np.std(is_returns) > 0 else 0,
        })
        
        # Drawdown calculations
        cumulative_oos = np.cumprod(1 + oos_returns)
        running_max = np.maximum.accumulate(cumulative_oos)
        drawdowns = (cumulative_oos - running_max) / running_max
        
        self.summary_stats.update({
            'max_drawdown': np.min(drawdowns),
            'avg_drawdown': np.mean(drawdowns[drawdowns < 0]) if np.any(drawdowns < 0) else 0,
        })
        
        # Win rate and profit factor
        winning_periods = np.sum(oos_returns > 0)
        total_profits = np.sum(oos_returns[oos_returns > 0])
        total_losses = abs(np.sum(oos_returns[oos_returns < 0]))
        
        self.summary_stats.update({
            'win_rate': winning_periods / len(oos_returns),
            'profit_factor': total_profits / total_losses if total_losses > 0 else np.inf,
        })
        
        # Overfitting detection
        is_oos_correlation = np.corrcoef(is_returns, oos_returns)[0, 1]
        self.summary_stats['is_oos_correlation'] = is_oos_correlation if not np.isnan(is_oos_correlation) else 0
        
        # Statistical significance (t-test against zero)
        t_stat, p_value = stats.ttest_1samp(oos_returns, 0)
        self.summary_stats.update({
            't_statistic': t_stat,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05,
        })


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis implementation for robust strategy validation.
    
    This class provides comprehensive walk-forward testing with parameter
    optimization, statistical analysis, and mode-specific validation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("WalkForwardAnalyzer")
        
        # Validation criteria for different modes
        self.validation_criteria = {
            'conservative': {
                'min_sharpe': 0.8,
                'max_drawdown': 0.15,
                'min_win_rate': 0.45,
                'min_profit_factor': 1.2,
                'min_periods': 20,
                'min_p_value': 0.05
            },
            'aggressive': {
                'min_sharpe': 0.5,
                'max_drawdown': 0.25,
                'min_win_rate': 0.40,
                'min_profit_factor': 1.1,
                'min_periods': 15,
                'min_p_value': 0.10
            }
        }
    
    def _default_config(self) -> Dict:
        """Default configuration for Walk-Forward Analysis."""
        return {
            'initial_window_days': 252,  # 1 year initial training window
            'step_size_days': 21,       # 3 weeks step size
            'oos_period_days': 21,      # 3 weeks out-of-sample period
            'min_train_samples': 100,   # Minimum training samples
            'max_params_combinations': 1000,  # Limit parameter grid size
            'window_type': 'expanding',  # 'expanding' or 'rolling'
            'rolling_window_days': 504,  # 2 years for rolling window
            'optimization_metric': 'sharpe_ratio',  # Metric to optimize
            'n_jobs': -1,               # Parallel processing
        }
    
    def run_walkforward_analysis(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List],
        mode: str = 'conservative',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> WalkForwardResult:
        """
        Run comprehensive Walk-Forward Analysis.
        
        Args:
            data: Historical price data with OHLCV columns
            strategy_func: Strategy function that takes (data, **params) and returns metrics
            param_grid: Parameter grid for optimization
            mode: Trading mode ('conservative' or 'aggressive')
            start_date: Analysis start date (defaults to data start)
            end_date: Analysis end date (defaults to data end)
            
        Returns:
            WalkForwardResult containing all analysis results
        """
        self.logger.info(f"Starting Walk-Forward Analysis in {mode} mode")
        
        # Validate inputs
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        if not callable(strategy_func):
            raise ValueError("strategy_func must be callable")
        
        # Prepare data
        data = data.copy().sort_values('timestamp')
        if start_date:
            data = data[data['timestamp'] >= start_date]
        if end_date:
            data = data[data['timestamp'] <= end_date]
        
        if len(data) < self.config['initial_window_days']:
            raise ValueError(f"Insufficient data: {len(data)} < {self.config['initial_window_days']}")
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        if len(param_combinations) > self.config['max_params_combinations']:
            self.logger.warning(
                f"Parameter grid too large ({len(param_combinations)}), "
                f"sampling {self.config['max_params_combinations']} combinations"
            )
            param_combinations = np.random.choice(
                param_combinations, 
                self.config['max_params_combinations'], 
                replace=False
            ).tolist()
        
        # Generate walk-forward periods
        periods = self._generate_wf_periods(data)
        
        if len(periods) < self.validation_criteria[mode]['min_periods']:
            raise ValueError(
                f"Insufficient periods for {mode} mode: "
                f"{len(periods)} < {self.validation_criteria[mode]['min_periods']}"
            )
        
        self.logger.info(f"Generated {len(periods)} walk-forward periods")
        
        # Run analysis for each period
        results = WalkForwardResult()
        
        for i, period in enumerate(periods):
            self.logger.debug(f"Processing period {i+1}/{len(periods)}")
            
            try:
                # Extract training and testing data
                train_data = data[
                    (data['timestamp'] >= period['train_start']) &
                    (data['timestamp'] <= period['train_end'])
                ]
                
                test_data = data[
                    (data['timestamp'] >= period['test_start']) &
                    (data['timestamp'] <= period['test_end'])
                ]
                
                if len(train_data) < self.config['min_train_samples']:
                    self.logger.warning(f"Insufficient training data in period {i+1}, skipping")
                    continue
                
                if len(test_data) == 0:
                    self.logger.warning(f"No test data in period {i+1}, skipping")
                    continue
                
                # Optimize parameters on training data
                best_params, is_metrics = self._optimize_parameters(
                    train_data, strategy_func, param_combinations
                )
                
                # Test on out-of-sample data
                oos_metrics = strategy_func(test_data, **best_params)
                
                # Extract returns
                is_return = is_metrics.get('total_return', 0)
                oos_return = oos_metrics.get('total_return', 0)
                
                # Add to results
                results.add_period(
                    start_date=period['period_start'],
                    end_date=period['period_end'],
                    train_start=period['train_start'],
                    train_end=period['train_end'],
                    test_start=period['test_start'],
                    test_end=period['test_end'],
                    is_return=is_return,
                    oos_return=oos_return,
                    best_params=best_params,
                    metrics=oos_metrics
                )
                
            except Exception as e:
                self.logger.error(f"Error in period {i+1}: {e}")
                continue
        
        # Calculate summary statistics
        results.calculate_summary_stats()
        
        # Generate equity curve
        results.equity_curve = self._generate_equity_curve(results)
        
        self.logger.info(
            f"Walk-Forward Analysis completed: {len(results.periods)} periods, "
            f"OOS Sharpe: {results.summary_stats.get('oos_sharpe_ratio', 0):.3f}"
        )
        
        return results
    
    def _generate_wf_periods(self, data: pd.DataFrame) -> List[Dict]:
        """Generate walk-forward analysis periods."""
        periods = []
        
        min_date = data['timestamp'].min()
        max_date = data['timestamp'].max()
        
        initial_window = timedelta(days=self.config['initial_window_days'])
        step_size = timedelta(days=self.config['step_size_days'])
        oos_period = timedelta(days=self.config['oos_period_days'])
        
        current_date = min_date + initial_window
        
        while current_date + oos_period <= max_date:
            # Define training period
            if self.config['window_type'] == 'expanding':
                train_start = min_date
            else:  # rolling window
                rolling_window = timedelta(days=self.config['rolling_window_days'])
                train_start = max(min_date, current_date - rolling_window)
            
            train_end = current_date
            test_start = current_date + timedelta(days=1)
            test_end = test_start + oos_period
            
            periods.append({
                'period_start': train_start,
                'period_end': test_end,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            current_date += step_size
        
        return periods
    
    def _optimize_parameters(
        self, 
        data: pd.DataFrame, 
        strategy_func: Callable,
        param_combinations: List[Dict]
    ) -> Tuple[Dict, Dict]:
        """Optimize strategy parameters on training data."""
        best_score = -np.inf
        best_params = {}
        best_metrics = {}
        
        optimization_metric = self.config['optimization_metric']
        
        for params in param_combinations:
            try:
                metrics = strategy_func(data, **params)
                
                # Get optimization score
                score = metrics.get(optimization_metric, -np.inf)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                    
            except Exception as e:
                self.logger.debug(f"Error evaluating parameters {params}: {e}")
                continue
        
        return best_params, best_metrics
    
    def _generate_equity_curve(self, results: WalkForwardResult) -> pd.DataFrame:
        """Generate equity curve from WFO results."""
        if not results.periods:
            return pd.DataFrame()
        
        equity_data = []
        cumulative_equity = 1.0
        
        for period in results.periods:
            oos_return = period['oos_return']
            cumulative_equity *= (1 + oos_return)
            
            equity_data.append({
                'date': period['test_end'],
                'oos_return': oos_return,
                'cumulative_equity': cumulative_equity,
                'period_start': period['period_start'],
                'period_end': period['period_end']
            })
        
        return pd.DataFrame(equity_data)
    
    def validate_results(
        self, 
        results: WalkForwardResult, 
        mode: str = 'conservative'
    ) -> Dict[str, Any]:
        """
        Validate WFO results against mode-specific criteria.
        
        Returns:
            Dictionary with validation results and pass/fail status
        """
        criteria = self.validation_criteria[mode]
        validation_results = {
            'mode': mode,
            'passed': True,
            'criteria': criteria.copy(),
            'actual_values': {},
            'failed_criteria': []
        }
        
        # Check each criterion
        checks = [
            ('min_sharpe', 'oos_sharpe_ratio', '>='),
            ('max_drawdown', 'max_drawdown', '<='),
            ('min_win_rate', 'win_rate', '>='),
            ('min_profit_factor', 'profit_factor', '>='),
            ('min_periods', 'total_periods', '>='),
            ('min_p_value', 'p_value', '<=')
        ]
        
        for criterion, stat_key, operator in checks:
            criterion_value = criteria[criterion]
            actual_value = results.summary_stats.get(stat_key, 0)
            validation_results['actual_values'][criterion] = actual_value
            
            if operator == '>=' and actual_value < criterion_value:
                validation_results['passed'] = False
                validation_results['failed_criteria'].append({
                    'criterion': criterion,
                    'required': criterion_value,
                    'actual': actual_value,
                    'operator': operator
                })
            elif operator == '<=' and actual_value > criterion_value:
                validation_results['passed'] = False
                validation_results['failed_criteria'].append({
                    'criterion': criterion,
                    'required': criterion_value,
                    'actual': actual_value,
                    'operator': operator
                })
        
        # Log validation results
        if validation_results['passed']:
            self.logger.info(f"Strategy passed {mode} mode validation")
        else:
            failed_criteria = [f['criterion'] for f in validation_results['failed_criteria']]
            self.logger.warning(
                f"Strategy failed {mode} mode validation on: {', '.join(failed_criteria)}"
            )
        
        return validation_results
    
    def generate_report(self, results: WalkForwardResult, mode: str = 'conservative') -> str:
        """Generate a comprehensive WFO analysis report."""
        validation = self.validate_results(results, mode)
        
        report = f"""
Walk-Forward Analysis Report
{'='*50}

Configuration:
- Mode: {mode}
- Initial Window: {self.config['initial_window_days']} days
- Step Size: {self.config['step_size_days']} days
- OOS Period: {self.config['oos_period_days']} days
- Window Type: {self.config['window_type']}

Results Summary:
- Total Periods: {results.summary_stats.get('total_periods', 0)}
- OOS Sharpe Ratio: {results.summary_stats.get('oos_sharpe_ratio', 0):.3f}
- OOS Cumulative Return: {results.summary_stats.get('oos_cumulative_return', 0):.2%}
- Maximum Drawdown: {results.summary_stats.get('max_drawdown', 0):.2%}
- Win Rate: {results.summary_stats.get('win_rate', 0):.2%}
- Profit Factor: {results.summary_stats.get('profit_factor', 0):.2f}
- Statistical Significance (p-value): {results.summary_stats.get('p_value', 1):.4f}

Validation Status: {'PASSED' if validation['passed'] else 'FAILED'}
"""
        
        if not validation['passed']:
            report += "\nFailed Criteria:\n"
            for failure in validation['failed_criteria']:
                report += f"- {failure['criterion']}: Required {failure['operator']} {failure['required']}, Got {failure['actual']:.4f}\n"
        
        return report