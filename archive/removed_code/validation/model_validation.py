"""
Model validation framework for trading strategies and risk models.

This module provides comprehensive model validation capabilities including:
- Statistical model validation
- Backtesting validation
- Cross-validation techniques
- Model stability assessment
- Performance attribution
- Robustness testing
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Statistical libraries
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class ValidationType(Enum):
    """Types of model validation."""
    STATISTICAL = "statistical"
    BACKTESTING = "backtesting"
    CROSS_VALIDATION = "cross_validation"
    STABILITY = "stability"
    ROBUSTNESS = "robustness"
    OUT_OF_SAMPLE = "out_of_sample"
    ROLLING_WINDOW = "rolling_window"

class ModelType(Enum):
    """Types of models to validate."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIME_SERIES = "time_series"
    PORTFOLIO = "portfolio"
    RISK_MODEL = "risk_model"
    TRADING_STRATEGY = "trading_strategy"

@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    # Validation parameters
    validation_types: List[ValidationType] = field(default_factory=lambda: [
        ValidationType.STATISTICAL, ValidationType.BACKTESTING, 
        ValidationType.CROSS_VALIDATION, ValidationType.STABILITY
    ])
    
    # Cross-validation settings
    cv_folds: int = 5
    time_series_cv: bool = True
    train_size_ratio: float = 0.7
    
    # Statistical tests
    significance_level: float = 0.05
    confidence_level: float = 0.95
    normality_tests: bool = True
    heteroskedasticity_tests: bool = True
    autocorrelation_tests: bool = True
    
    # Stability testing
    stability_window: int = 252  # Trading days
    stability_step: int = 21    # Rolling step size
    parameter_stability_threshold: float = 0.1
    performance_stability_threshold: float = 0.15
    
    # Robustness testing
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    sample_perturbations: List[float] = field(default_factory=lambda: [0.9, 0.95, 0.99])
    bootstrap_iterations: int = 1000
    
    # Model comparison
    benchmark_models: List[str] = field(default_factory=lambda: ["random", "naive", "linear"])
    performance_metrics: List[str] = field(default_factory=lambda: [
        "mse", "mae", "r2", "sharpe", "sortino", "max_drawdown"
    ])
    
    # Execution settings
    parallel_processing: bool = True
    max_workers: int = 4
    save_intermediate: bool = True
    verbose: bool = True

@dataclass
class ValidationResult:
    """Results from model validation."""
    model_id: str
    model_type: ModelType
    validation_type: ValidationType
    timestamp: datetime
    
    # Statistical test results  
    statistical_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Cross-validation results
    cv_scores: Dict[str, List[float]] = field(default_factory=dict)
    cv_mean_scores: Dict[str, float] = field(default_factory=dict)
    cv_std_scores: Dict[str, float] = field(default_factory=dict)
    
    # Stability metrics
    parameter_stability: Dict[str, float] = field(default_factory=dict)
    performance_stability: Dict[str, float] = field(default_factory=dict)
    stability_windows: List[Dict[str, Any]] = field(default_factory=list)
    
    # Robustness metrics
    noise_robustness: Dict[float, Dict[str, float]] = field(default_factory=dict)
    bootstrap_results: Dict[str, List[float]] = field(default_factory=dict)
    
    # Model comparison
    benchmark_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    relative_performance: Dict[str, float] = field(default_factory=dict)
    
    # Out-of-sample results
    oos_metrics: Dict[str, float] = field(default_factory=dict)
    oos_predictions: Optional[np.ndarray] = None
    oos_actuals: Optional[np.ndarray] = None
    
    # Feature importance and attribution
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_stability: Dict[str, float] = field(default_factory=dict)
    
    # Model diagnostics
    residual_analysis: Dict[str, float] = field(default_factory=dict)
    goodness_of_fit: Dict[str, float] = field(default_factory=dict)
    
    # Validation scores
    overall_score: float = 0.0
    validation_passed: bool = False
    validation_warnings: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    
    # Execution metrics
    validation_time: float = 0.0
    memory_usage: float = 0.0

@dataclass
class ValidationSummary:
    """Summary of all validation results."""
    validation_id: str
    timestamp: datetime
    model_count: int
    
    # Aggregate results
    passed_validations: int
    failed_validations: int
    warning_validations: int
    
    # Best performing models
    best_models: Dict[str, str]  # metric -> model_id
    model_rankings: Dict[str, List[Tuple[str, float]]]
    
    # Statistical summaries
    average_scores: Dict[str, float]
    score_distributions: Dict[str, List[float]]
    
    # Stability analysis
    most_stable_models: List[str]
    least_stable_models: List[str]
    stability_statistics: Dict[str, float]
    
    # Robustness analysis
    most_robust_models: List[str]
    robustness_statistics: Dict[str, float]
    
    # Feature analysis
    important_features: Dict[str, float]
    stable_features: List[str]
    unstable_features: List[str]
    
    # Recommendations
    recommended_models: List[str]
    model_warnings: Dict[str, List[str]]
    improvement_suggestions: Dict[str, List[str]]

class StatisticalValidator:
    """Statistical validation of models."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate_residuals(self, residuals: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Validate model residuals with statistical tests."""
        results = {}
        
        # Remove NaN values
        clean_residuals = residuals[~np.isnan(residuals)]
        
        if len(clean_residuals) < 10:
            logger.warning("Insufficient residuals for statistical validation")
            return results
        
        # Normality tests
        if self.config.normality_tests:
            results['normality'] = self._test_normality(clean_residuals)
        
        # Heteroskedasticity tests
        if self.config.heteroskedasticity_tests:
            results['heteroskedasticity'] = self._test_heteroskedasticity(clean_residuals)
        
        # Autocorrelation tests
        if self.config.autocorrelation_tests:
            results['autocorrelation'] = self._test_autocorrelation(clean_residuals)
        
        # Independence tests
        results['independence'] = self._test_independence(clean_residuals)
        
        return results
    
    def _test_normality(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test normality of residuals."""
        results = {}
        
        try:
            # Shapiro-Wilk test (for smaller samples)
            if len(residuals) <= 5000:
                sw_stat, sw_pvalue = stats.shapiro(residuals)
                results['shapiro_wilk_statistic'] = sw_stat
                results['shapiro_wilk_pvalue'] = sw_pvalue
            
            # Jarque-Bera test
            jb_stat, jb_pvalue = stats.jarque_bera(residuals)
            results['jarque_bera_statistic'] = jb_stat
            results['jarque_bera_pvalue'] = jb_pvalue
            
            # Anderson-Darling test
            ad_stat, ad_critical, ad_significance = stats.anderson(residuals, dist='norm')
            results['anderson_darling_statistic'] = ad_stat
            results['anderson_darling_critical'] = ad_critical[2]  # 5% level
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.kstest(residuals, 'norm', 
                                            args=(np.mean(residuals), np.std(residuals)))
            results['ks_statistic'] = ks_stat
            results['ks_pvalue'] = ks_pvalue
            
        except Exception as e:
            logger.error(f"Error in normality tests: {e}")
            
        return results
    
    def _test_heteroskedasticity(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test for heteroskedasticity in residuals."""
        results = {}
        
        try:
            # Create time index for residuals
            time_index = np.arange(len(residuals))
            
            # Breusch-Pagan test (simplified)
            residuals_squared = residuals ** 2
            
            # Regress squared residuals on time
            if len(time_index) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, residuals_squared)
                results['breusch_pagan_r2'] = r_value ** 2
                results['breusch_pagan_pvalue'] = p_value
            
            # White test (simplified - using squared residuals variance)
            results['white_test_variance'] = np.var(residuals_squared)
            
            # ARCH test (simplified)
            if len(residuals) > 10:
                # Test if squared residuals are correlated with lagged squared residuals
                lagged_residuals_sq = np.roll(residuals_squared, 1)[1:]
                current_residuals_sq = residuals_squared[1:]
                
                if len(current_residuals_sq) > 1:
                    arch_corr, arch_pvalue = stats.pearsonr(current_residuals_sq, lagged_residuals_sq)
                    results['arch_correlation'] = arch_corr
                    results['arch_pvalue'] = arch_pvalue
            
        except Exception as e:
            logger.error(f"Error in heteroskedasticity tests: {e}")
            
        return results
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test for autocorrelation in residuals."""
        results = {}
        
        try:
            # Ljung-Box test (simplified)
            # Test first few lags
            max_lags = min(10, len(residuals) // 4)
            
            autocorrelations = []
            for lag in range(1, max_lags + 1):
                if lag < len(residuals):
                    # Calculate autocorrelation at lag
                    lagged_residuals = np.roll(residuals, lag)[lag:]
                    current_residuals = residuals[lag:]
                    
                    if len(current_residuals) > 1:
                        corr, _ = stats.pearsonr(current_residuals, lagged_residuals)
                        autocorrelations.append(corr)
            
            if autocorrelations:
                results['max_autocorrelation'] = max(abs(ac) for ac in autocorrelations)
                results['mean_autocorrelation'] = np.mean(np.abs(autocorrelations))
                
                # Ljung-Box statistic (simplified)
                n = len(residuals)
                lb_stat = n * (n + 2) * sum(ac**2 / (n - k) for k, ac in enumerate(autocorrelations, 1))
                results['ljung_box_statistic'] = lb_stat
            
            # Durbin-Watson test (for lag-1 autocorrelation)
            if len(residuals) > 2:
                dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
                results['durbin_watson_statistic'] = dw_stat
            
        except Exception as e:
            logger.error(f"Error in autocorrelation tests: {e}")
            
        return results
    
    def _test_independence(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test independence of residuals."""
        results = {}
        
        try:
            # Runs test for randomness
            median_residual = np.median(residuals)
            runs, n1, n2 = self._runs_test(residuals > median_residual)
            
            if n1 > 0 and n2 > 0:
                # Expected runs and variance
                expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
                runs_variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
                
                if runs_variance > 0:
                    z_score = (runs - expected_runs) / np.sqrt(runs_variance)
                    results['runs_test_z_score'] = z_score
                    results['runs_test_pvalue'] = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            # Turn points test (simplified)
            turning_points = sum(1 for i in range(1, len(residuals) - 1)
                               if (residuals[i] > residuals[i-1] and residuals[i] > residuals[i+1]) or
                                  (residuals[i] < residuals[i-1] and residuals[i] < residuals[i+1]))
            
            expected_turns = (2 * (len(residuals) - 2)) / 3
            results['turning_points'] = turning_points
            results['expected_turning_points'] = expected_turns
            
        except Exception as e:
            logger.error(f"Error in independence tests: {e}")
            
        return results
    
    def _runs_test(self, binary_sequence: np.ndarray) -> Tuple[int, int, int]:
        """Perform runs test on binary sequence."""
        n1 = np.sum(binary_sequence)  # Number of True values
        n2 = len(binary_sequence) - n1  # Number of False values
        
        # Count runs
        runs = 1
        for i in range(1, len(binary_sequence)):
            if binary_sequence[i] != binary_sequence[i-1]:
                runs += 1
        
        return runs, n1, n2

class CrossValidator:
    """Cross-validation for model validation."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                      model_type: ModelType) -> Dict[str, Any]:
        """Perform cross-validation on model."""
        results = {}
        
        try:
            # Choose appropriate cross-validation strategy
            if self.config.time_series_cv and model_type in [ModelType.TIME_SERIES, ModelType.TRADING_STRATEGY]:
                cv_splitter = TimeSeriesSplit(n_splits=self.config.cv_folds)
            else:
                # For non-time series, we'll still use TimeSeriesSplit to respect temporal order
                cv_splitter = TimeSeriesSplit(n_splits=self.config.cv_folds)
            
            # Perform cross-validation for different metrics
            for metric in self.config.performance_metrics:
                if metric in ['mse', 'mae', 'r2']:
                    # Regression metrics
                    if hasattr(model, 'predict'):
                        scores = self._cross_validate_regression(model, X, y, cv_splitter, metric)
                        results[f'{metric}_scores'] = scores
                        results[f'{metric}_mean'] = np.mean(scores)
                        results[f'{metric}_std'] = np.std(scores)
                
                elif metric in ['sharpe', 'sortino', 'max_drawdown']:
                    # Financial metrics require special handling
                    scores = self._cross_validate_financial(model, X, y, cv_splitter, metric)
                    results[f'{metric}_scores'] = scores
                    results[f'{metric}_mean'] = np.mean(scores)
                    results[f'{metric}_std'] = np.std(scores)
            
            # Feature importance across folds
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                importance_scores = self._cross_validate_feature_importance(model, X, y, cv_splitter)
                results['feature_importance'] = importance_scores
                
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            results['error'] = str(e)
        
        return results
    
    def _cross_validate_regression(self, model: Any, X: np.ndarray, y: np.ndarray,
                                 cv_splitter: Any, metric: str) -> List[float]:
        """Cross-validate regression metrics."""
        scores = []
        
        for train_idx, val_idx in cv_splitter.split(X):
            try:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                
                # Calculate metric
                if metric == 'mse':
                    score = mean_squared_error(y_val, y_pred)
                elif metric == 'mae':
                    score = mean_absolute_error(y_val, y_pred)
                elif metric == 'r2':
                    score = r2_score(y_val, y_pred)
                else:
                    score = 0.0
                
                scores.append(score)
                
            except Exception as e:
                logger.warning(f"Error in CV fold: {e}")
                scores.append(np.nan)
        
        return [s for s in scores if not np.isnan(s)]
    
    def _cross_validate_financial(self, model: Any, X: np.ndarray, y: np.ndarray,
                                cv_splitter: Any, metric: str) -> List[float]:
        """Cross-validate financial metrics."""
        scores = []
        
        for train_idx, val_idx in cv_splitter.split(X):
            try:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Make predictions (interpret as returns)
                returns_pred = model.predict(X_val)
                
                # Calculate financial metric
                if metric == 'sharpe':
                    if np.std(returns_pred) > 0:
                        score = np.mean(returns_pred) / np.std(returns_pred) * np.sqrt(252)
                    else:
                        score = 0.0
                elif metric == 'sortino':
                    downside_returns = returns_pred[returns_pred < 0]
                    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                        score = np.mean(returns_pred) / np.std(downside_returns) * np.sqrt(252)
                    else:
                        score = 0.0
                elif metric == 'max_drawdown':
                    cumulative = np.cumprod(1 + returns_pred)
                    rolling_max = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - rolling_max) / rolling_max
                    score = np.min(drawdown)
                else:
                    score = 0.0
                
                scores.append(score)
                
            except Exception as e:
                logger.warning(f"Error in financial CV fold: {e}")
                scores.append(np.nan)
        
        return [s for s in scores if not np.isnan(s)]
    
    def _cross_validate_feature_importance(self, model: Any, X: np.ndarray, y: np.ndarray,
                                         cv_splitter: Any) -> Dict[str, List[float]]:
        """Cross-validate feature importance."""
        importance_results = {}
        n_features = X.shape[1]
        
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X)):
            try:
                X_train, y_train = X[train_idx], y[train_idx]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                    if importances.ndim > 1:
                        importances = np.mean(importances, axis=0)
                else:
                    importances = np.ones(n_features) / n_features
                
                # Store importance for each feature
                for i, importance in enumerate(importances):
                    feature_name = f'feature_{i}'
                    if feature_name not in importance_results:
                        importance_results[feature_name] = []
                    importance_results[feature_name].append(importance)
                    
            except Exception as e:
                logger.warning(f"Error getting feature importance in fold {fold}: {e}")
        
        return importance_results

class StabilityValidator:
    """Validate model stability over time."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate_stability(self, model: Any, X: np.ndarray, y: np.ndarray,
                          timestamps: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """Validate model stability over rolling windows."""
        results = {
            'parameter_stability': {},
            'performance_stability': {},
            'stability_windows': []
        }
        
        try:
            window_size = self.config.stability_window
            step_size = self.config.stability_step
            
            if len(X) < window_size:
                logger.warning("Insufficient data for stability validation")
                return results
            
            # Rolling window analysis
            parameter_history = []
            performance_history = []
            
            for start in range(0, len(X) - window_size + 1, step_size):
                end = start + window_size
                
                X_window = X[start:end]
                y_window = y[start:end]
                
                # Fit model on window
                try:
                    model.fit(X_window, y_window)
                    
                    # Get parameters
                    params = self._extract_model_parameters(model)
                    parameter_history.append(params)
                    
                    # Evaluate performance on window
                    y_pred = model.predict(X_window)
                    performance = self._calculate_performance_metrics(y_window, y_pred)
                    performance_history.append(performance)
                    
                    # Store window info
                    window_info = {
                        'start_idx': start,
                        'end_idx': end,
                        'parameters': params,
                        'performance': performance
                    }
                    
                    if timestamps is not None:
                        window_info['start_date'] = timestamps[start]
                        window_info['end_date'] = timestamps[end-1]
                    
                    results['stability_windows'].append(window_info)
                    
                except Exception as e:
                    logger.warning(f"Error in stability window {start}-{end}: {e}")
                    continue
            
            # Analyze parameter stability
            if parameter_history:
                results['parameter_stability'] = self._analyze_parameter_stability(parameter_history)
            
            # Analyze performance stability
            if performance_history:
                results['performance_stability'] = self._analyze_performance_stability(performance_history)
                
        except Exception as e:
            logger.error(f"Error in stability validation: {e}")
            results['error'] = str(e)
        
        return results
    
    def _extract_model_parameters(self, model: Any) -> Dict[str, float]:
        """Extract parameters from fitted model."""
        params = {}
        
        try:
            if hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim == 1:
                    for i, c in enumerate(coef):
                        params[f'coef_{i}'] = float(c)
                else:
                    for i, row in enumerate(coef):
                        for j, c in enumerate(row):
                            params[f'coef_{i}_{j}'] = float(c)
            
            if hasattr(model, 'intercept_'):
                intercept = model.intercept_
                if np.isscalar(intercept):
                    params['intercept'] = float(intercept)
                else:
                    for i, ic in enumerate(intercept):
                        params[f'intercept_{i}'] = float(ic)
            
            if hasattr(model, 'feature_importances_'):
                for i, imp in enumerate(model.feature_importances_):
                    params[f'importance_{i}'] = float(imp)
            
            # For more complex models, extract key hyperparameters
            if hasattr(model, 'get_params'):
                model_params = model.get_params()
                for key, value in model_params.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        params[f'hyperparam_{key}'] = float(value)
                        
        except Exception as e:
            logger.warning(f"Error extracting model parameters: {e}")
        
        return params
    
    def _calculate_performance_metrics(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics for stability analysis."""
        metrics = {}
        
        try:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Financial metrics (assuming returns)
            if np.std(y_pred) > 0:
                metrics['sharpe'] = np.mean(y_pred) / np.std(y_pred) * np.sqrt(252)
            else:
                metrics['sharpe'] = 0.0
            
            # Correlation between actual and predicted
            if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
                metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
            else:
                metrics['correlation'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {e}")
        
        return metrics
    
    def _analyze_parameter_stability(self, parameter_history: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze stability of model parameters."""
        stability_results = {}
        
        try:
            # Get all parameter names
            all_params = set()
            for params in parameter_history:
                all_params.update(params.keys())
            
            # Analyze each parameter
            for param_name in all_params:
                param_values = []
                for params in parameter_history:
                    if param_name in params:
                        param_values.append(params[param_name])
                
                if len(param_values) > 1:
                    # Calculate coefficient of variation as stability measure
                    mean_val = np.mean(param_values)
                    std_val = np.std(param_values)
                    
                    if abs(mean_val) > 1e-8:
                        cv = std_val / abs(mean_val)
                        stability_results[param_name] = 1.0 / (1.0 + cv)  # Higher is more stable
                    else:
                        stability_results[param_name] = 1.0 if std_val < 1e-8 else 0.0
                        
        except Exception as e:
            logger.error(f"Error analyzing parameter stability: {e}")
        
        return stability_results
    
    def _analyze_performance_stability(self, performance_history: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze stability of model performance."""
        stability_results = {}
        
        try:
            # Get all metric names
            all_metrics = set()
            for perf in performance_history:
                all_metrics.update(perf.keys())
            
            # Analyze each metric
            for metric_name in all_metrics:
                metric_values = []
                for perf in performance_history:
                    if metric_name in perf and not np.isnan(perf[metric_name]):
                        metric_values.append(perf[metric_name])
                
                if len(metric_values) > 1:
                    # Calculate stability as inverse of coefficient of variation
                    mean_val = np.mean(metric_values)
                    std_val = np.std(metric_values)
                    
                    if abs(mean_val) > 1e-8:
                        cv = std_val / abs(mean_val)
                        stability_results[f'{metric_name}_stability'] = 1.0 / (1.0 + cv)
                    else:
                        stability_results[f'{metric_name}_stability'] = 1.0 if std_val < 1e-8 else 0.0
                    
                    # Also store trend analysis
                    if len(metric_values) >= 3:
                        # Linear trend test
                        x = np.arange(len(metric_values))
                        slope, _, r_value, p_value, _ = stats.linregress(x, metric_values)
                        stability_results[f'{metric_name}_trend_slope'] = slope
                        stability_results[f'{metric_name}_trend_r2'] = r_value ** 2
                        stability_results[f'{metric_name}_trend_pvalue'] = p_value
                        
        except Exception as e:
            logger.error(f"Error analyzing performance stability: {e}")
        
        return stability_results

class ModelValidator:
    """Main model validation engine."""
    
    def __init__(self, config: ValidationConfig, db_path: Optional[str] = None):
        self.config = config
        self.db_path = db_path or "model_validation_results.db"
        
        # Initialize validators
        self.statistical_validator = StatisticalValidator(config)
        self.cross_validator = CrossValidator(config)
        self.stability_validator = StabilityValidator(config)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database for storing results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Validation results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    validation_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    validation_type TEXT NOT NULL,
                    overall_score REAL,
                    validation_passed INTEGER,
                    cv_mean_scores TEXT,
                    statistical_tests TEXT,
                    stability_metrics TEXT,
                    robustness_metrics TEXT,
                    feature_importance TEXT,
                    validation_warnings TEXT,
                    validation_errors TEXT,
                    validation_time REAL
                )
            """)
            
            # Validation summary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    validation_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    model_count INTEGER,
                    passed_validations INTEGER,
                    failed_validations INTEGER,
                    best_models TEXT,
                    model_rankings TEXT,
                    stability_statistics TEXT,
                    robustness_statistics TEXT,
                    recommendations TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                      model_id: str, model_type: ModelType,
                      timestamps: Optional[pd.DatetimeIndex] = None) -> ValidationResult:
        """Validate a single model comprehensively."""
        start_time = datetime.now()
        
        logger.info(f"Starting validation for model: {model_id}")
        
        result = ValidationResult(
            model_id=model_id,
            model_type=model_type,
            validation_type=ValidationType.STATISTICAL,  # Will be updated
            timestamp=start_time
        )
        
        try:
            # Statistical validation
            if ValidationType.STATISTICAL in self.config.validation_types:
                logger.info("Running statistical validation...")
                
                # Fit model first
                model.fit(X, y)
                y_pred = model.predict(X)
                residuals = y - y_pred
                
                result.statistical_tests = self.statistical_validator.validate_residuals(residuals)
                result.residual_analysis = self._analyze_residuals(residuals)
                result.goodness_of_fit = self._calculate_goodness_of_fit(y, y_pred)
            
            # Cross-validation
            if ValidationType.CROSS_VALIDATION in self.config.validation_types:
                logger.info("Running cross-validation...")
                
                cv_results = self.cross_validator.validate_model(model, X, y, model_type)
                
                # Extract CV scores
                for key, value in cv_results.items():
                    if key.endswith('_scores') and isinstance(value, list):
                        metric_name = key.replace('_scores', '')
                        result.cv_scores[metric_name] = value
                        result.cv_mean_scores[metric_name] = np.mean(value)
                        result.cv_std_scores[metric_name] = np.std(value)
                
                # Feature importance
                if 'feature_importance' in cv_results:
                    importance_dict = cv_results['feature_importance']
                    for feature, importance_list in importance_dict.items():
                        result.feature_importance[feature] = np.mean(importance_list)
                        result.feature_stability[feature] = 1.0 - np.std(importance_list) / (abs(np.mean(importance_list)) + 1e-8)
            
            # Stability validation
            if ValidationType.STABILITY in self.config.validation_types:
                logger.info("Running stability validation...")
                
                stability_results = self.stability_validator.validate_stability(model, X, y, timestamps)
                result.parameter_stability = stability_results.get('parameter_stability', {})
                result.performance_stability = stability_results.get('performance_stability', {})
                result.stability_windows = stability_results.get('stability_windows', [])
            
            # Calculate overall score
            result.overall_score = self._calculate_overall_score(result)
            
            # Determine if validation passed
            result.validation_passed = self._determine_validation_status(result)
            
            # Add warnings and recommendations
            result.validation_warnings = self._generate_warnings(result)
            
            result.validation_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Model validation completed. Score: {result.overall_score:.3f}, "
                       f"Passed: {result.validation_passed}")
            
        except Exception as e:
            error_msg = f"Error validating model {model_id}: {str(e)}"
            logger.error(error_msg)
            result.validation_errors.append(error_msg)
            result.validation_passed = False
            result.validation_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _analyze_residuals(self, residuals: np.ndarray) -> Dict[str, float]:
        """Analyze residual properties."""
        analysis = {}
        
        try:
            clean_residuals = residuals[~np.isnan(residuals)]
            
            if len(clean_residuals) > 0:
                analysis['mean'] = np.mean(clean_residuals)
                analysis['std'] = np.std(clean_residuals)
                analysis['skewness'] = stats.skew(clean_residuals)
                analysis['kurtosis'] = stats.kurtosis(clean_residuals)
                analysis['min'] = np.min(clean_residuals)
                analysis['max'] = np.max(clean_residuals)
                
                # Check for patterns
                analysis['mean_abs_residual'] = np.mean(np.abs(clean_residuals))
                analysis['residual_autocorr'] = self._calculate_autocorrelation(clean_residuals, lag=1)
                
        except Exception as e:
            logger.warning(f"Error analyzing residuals: {e}")
        
        return analysis
    
    def _calculate_autocorrelation(self, data: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at specified lag."""
        try:
            if len(data) <= lag:
                return 0.0
            
            lagged_data = np.roll(data, lag)[lag:]
            current_data = data[lag:]
            
            if len(current_data) > 1:
                corr, _ = stats.pearsonr(current_data, lagged_data)
                return corr if not np.isnan(corr) else 0.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_goodness_of_fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate goodness of fit metrics."""
        metrics = {}
        
        try:
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (len(y_true) - 1) / (len(y_true) - X.shape[1] - 1) if hasattr(self, 'X') else metrics['r2']
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            
            # Additional metrics
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            
            if ss_tot > 0:
                metrics['explained_variance'] = 1 - ss_res / ss_tot
            else:
                metrics['explained_variance'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating goodness of fit: {e}")
        
        return metrics
    
    def _calculate_overall_score(self, result: ValidationResult) -> float:
        """Calculate overall validation score."""
        score_components = []
        
        try:
            # Statistical test scores
            if result.statistical_tests:
                stat_scores = []
                for test_category, tests in result.statistical_tests.items():
                    for test_name, p_value in tests.items():
                        if test_name.endswith('_pvalue') and not np.isnan(p_value):
                            # Higher p-value is better (null hypothesis: model is valid)
                            stat_scores.append(min(1.0, p_value * 20))  # Scale p-values
                
                if stat_scores:
                    score_components.append(('statistical', np.mean(stat_scores), 0.3))
            
            # Cross-validation scores
            if result.cv_mean_scores:
                cv_scores = []
                for metric, score in result.cv_mean_scores.items():
                    if not np.isnan(score):
                        if metric in ['r2']:
                            cv_scores.append(max(0.0, score))  # R2 can be negative
                        elif metric in ['mse', 'mae']:
                            cv_scores.append(1.0 / (1.0 + score))  # Lower is better
                        elif metric in ['sharpe', 'sortino']:
                            cv_scores.append(max(0.0, min(1.0, (score + 2) / 4)))  # Scale to 0-1
                
                if cv_scores:
                    score_components.append(('cross_validation', np.mean(cv_scores), 0.4))
            
            # Stability scores
            if result.parameter_stability or result.performance_stability:
                stability_scores = []
                
                for stability in result.parameter_stability.values():
                    if not np.isnan(stability):
                        stability_scores.append(stability)
                
                for stability in result.performance_stability.values():
                    if not np.isnan(stability):
                        stability_scores.append(stability)
                
                if stability_scores:
                    score_components.append(('stability', np.mean(stability_scores), 0.3))
            
            # Calculate weighted average
            if score_components:
                total_weight = sum(weight for _, _, weight in score_components)
                weighted_score = sum(score * weight for _, score, weight in score_components) / total_weight
                return max(0.0, min(1.0, weighted_score))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0
    
    def _determine_validation_status(self, result: ValidationResult) -> bool:
        """Determine if model passes validation."""
        try:
            # Check overall score threshold
            if result.overall_score < 0.5:
                return False
            
            # Check for critical statistical test failures
            if result.statistical_tests:
                for test_category, tests in result.statistical_tests.items():
                    for test_name, p_value in tests.items():
                        if test_name.endswith('_pvalue') and not np.isnan(p_value):
                            if p_value < self.config.significance_level and test_category in ['normality']:
                                result.validation_warnings.append(f"Failed {test_category} test: {test_name}")
            
            # Check cross-validation performance
            if result.cv_mean_scores:
                if 'r2' in result.cv_mean_scores and result.cv_mean_scores['r2'] < 0.1:
                    result.validation_warnings.append("Low R-squared in cross-validation")
            
            # Check stability
            stability_values = list(result.parameter_stability.values()) + list(result.performance_stability.values())
            if stability_values:
                avg_stability = np.mean([s for s in stability_values if not np.isnan(s)])
                if avg_stability < 0.5:
                    result.validation_warnings.append("Low model stability detected")
            
            # Pass if no critical errors and reasonable score
            return len(result.validation_errors) == 0 and result.overall_score >= 0.4
            
        except Exception as e:
            logger.error(f"Error determining validation status: {e}")
            return False
    
    def _generate_warnings(self, result: ValidationResult) -> List[str]:
        """Generate validation warnings."""
        warnings = list(result.validation_warnings)  # Copy existing warnings
        
        try:
            # Check for overfitting
            if result.cv_std_scores:
                for metric, std_score in result.cv_std_scores.items():
                    if std_score > 0.3:  # High variation across folds
                        warnings.append(f"High variation in {metric} across CV folds (possible overfitting)")
            
            # Check feature importance concentration
            if result.feature_importance:
                importances = list(result.feature_importance.values())
                if importances:
                    max_importance = max(importances)
                    if max_importance > 0.8:  # One feature dominates
                        warnings.append("Model heavily relies on single feature")
            
            # Check residual patterns
            if result.residual_analysis:
                if abs(result.residual_analysis.get('mean', 0)) > 0.1:
                    warnings.append("Residuals have non-zero mean (bias detected)")
                
                if abs(result.residual_analysis.get('residual_autocorr', 0)) > 0.3:
                    warnings.append("Residuals show autocorrelation (model may miss patterns)")
                    
        except Exception as e:
            logger.warning(f"Error generating warnings: {e}")
        
        return warnings
    
    def save_results(self, result: ValidationResult, validation_id: str):
        """Save validation results to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO validation_results (
                    model_id, validation_id, timestamp, model_type, validation_type,
                    overall_score, validation_passed, cv_mean_scores, statistical_tests,
                    stability_metrics, feature_importance, validation_warnings,
                    validation_errors, validation_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.model_id,
                validation_id,
                result.timestamp.isoformat(),
                result.model_type.value,
                result.validation_type.value,
                result.overall_score,
                1 if result.validation_passed else 0,
                json.dumps(result.cv_mean_scores),
                json.dumps(result.statistical_tests),
                json.dumps({
                    'parameter_stability': result.parameter_stability,
                    'performance_stability': result.performance_stability
                }),
                json.dumps(result.feature_importance),
                json.dumps(result.validation_warnings),
                json.dumps(result.validation_errors),
                result.validation_time
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving validation results: {e}")

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
    n_features = 5
    
    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with realistic relationship
    true_coef = np.array([1.5, -0.8, 2.1, 0.3, -1.2])
    y = X @ true_coef + np.random.normal(0, 0.5, n_samples)
    
    # Add some non-linearity and regime changes
    regime_change = n_samples // 2
    y[regime_change:] += X[regime_change:, 0] ** 2 * 0.5
    
    # Create timestamps
    timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create and validate model
    config = ValidationConfig(
        cv_folds=5,
        significance_level=0.05,
        stability_window=100,
        stability_step=20
    )
    
    validator = ModelValidator(config)
    
    # Test with linear regression
    model = LinearRegression()
    
    try:
        result = validator.validate_model(
            model=model,
            X=X, 
            y=y,
            model_id="linear_regression_test",
            model_type=ModelType.REGRESSION,
            timestamps=timestamps
        )
        
        print("\n=== Model Validation Results ===")
        print(f"Model ID: {result.model_id}")
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Validation Passed: {result.validation_passed}")
        print(f"Validation Time: {result.validation_time:.2f}s")
        
        print("\n--- Cross-Validation Scores ---")
        for metric, score in result.cv_mean_scores.items():
            std_score = result.cv_std_scores.get(metric, 0.0)
            print(f"{metric}: {score:.4f} ± {std_score:.4f}")
        
        print("\n--- Feature Importance ---")
        for feature, importance in result.feature_importance.items():
            print(f"{feature}: {importance:.4f}")
        
        print("\n--- Parameter Stability ---")
        for param, stability in result.parameter_stability.items():
            print(f"{param}: {stability:.4f}")
        
        if result.validation_warnings:
            print("\n--- Warnings ---")
            for warning in result.validation_warnings:
                print(f"⚠️  {warning}")
        
        if result.validation_errors:
            print("\n--- Errors ---")
            for error in result.validation_errors:
                print(f"❌ {error}")
        
        # Save results
        validation_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        validator.save_results(result, validation_id)
        print(f"\nResults saved with validation ID: {validation_id}")
        
    except Exception as e:
        logger.error(f"Error in validation example: {e}")