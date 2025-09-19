"""
Walk-forward analysis implementation for model validation.

This module provides comprehensive walk-forward analysis capabilities including:
- Time series cross-validation
- Out-of-sample testing
- Model stability assessment
- Performance degradation analysis
- Regime-aware validation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import sqlite3
from pathlib import Path
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Statistical libraries
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    # Time windows
    train_window: int = 252  # Trading days
    test_window: int = 21    # Trading days
    step_size: int = 21      # Days to advance each iteration
    min_train_size: int = 126  # Minimum training period
    
    # Validation parameters
    n_splits: int = 10       # Number of walk-forward splits
    max_splits: Optional[int] = None  # Maximum splits to process
    overlap_threshold: float = 0.1  # Maximum overlap between train/test
    
    # Performance metrics
    confidence_level: float = 0.95
    benchmark_symbol: str = "BTC"
    risk_free_rate: float = 0.02  # Annual risk-free rate
    
    # Regime awareness
    use_regime_detection: bool = True
    regime_change_threshold: float = 0.05
    min_regime_samples: int = 30
    
    # Stability testing
    stability_window: int = 60  # Days for stability assessment
    stability_threshold: float = 0.15  # Maximum performance deviation
    
    # Advanced options
    parallel_processing: bool = True
    max_workers: int = 4
    cache_results: bool = True
    save_intermediate: bool = True

@dataclass
class WalkForwardResult:
    """Results from a single walk-forward iteration."""
    iteration: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Training metrics
    train_samples: int
    train_returns: float
    train_sharpe: float
    train_max_drawdown: float
    train_volatility: float
    
    # Test metrics
    test_samples: int
    test_returns: float
    test_sharpe: float
    test_max_drawdown: float
    test_volatility: float
    
    # Model metrics
    model_accuracy: float
    prediction_error: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Regime information
    regime_detected: Optional[str] = None
    regime_confidence: float = 0.0
    regime_stability: float = 0.0
    
    # Additional metrics
    information_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    hit_rate: float = 0.0
    
    # Metadata
    execution_time: float = 0.0
    memory_usage: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

@dataclass
class WalkForwardSummary:
    """Summary statistics from walk-forward analysis."""
    total_iterations: int
    successful_iterations: int
    failed_iterations: int
    
    # Aggregate performance
    mean_oos_returns: float
    std_oos_returns: float
    mean_oos_sharpe: float
    std_oos_sharpe: float
    
    # Stability metrics
    performance_stability: float
    regime_consistency: float
    model_reliability: float
    
    # Statistical tests
    is_statistically_significant: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    
    # Regime analysis
    regime_distribution: Dict[str, int]
    regime_performance: Dict[str, float]
    
    # Risk metrics
    worst_case_return: float
    worst_case_drawdown: float
    var_95: float
    cvar_95: float
    
    # Model assessment
    overfitting_score: float
    generalization_ability: float
    feature_stability: Dict[str, float]
    
    # Execution stats
    total_execution_time: float
    average_iteration_time: float
    memory_efficiency: float

class ModelValidator(ABC):
    """Abstract base class for model validation."""
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the model on training data."""
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions on test data."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters."""
        pass

class RegimeDetector:
    """Regime detection for walk-forward analysis."""
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.regimes = {}
        self.regime_history = []
        
    def detect_regime(self, data: pd.DataFrame, 
                     returns_col: str = 'returns') -> Tuple[str, float]:
        """Detect current market regime."""
        try:
            if len(data) < self.config.min_regime_samples:
                return "insufficient_data", 0.0
                
            returns = data[returns_col].dropna()
            
            # Calculate regime indicators
            volatility = returns.rolling(20).std().iloc[-1]
            skewness = returns.rolling(60).skew().iloc[-1]
            kurtosis = returns.rolling(60).kurtosis().iloc[-1]
            trend = returns.rolling(20).mean().iloc[-1]
            
            # Regime classification logic
            regime_score = 0.0
            
            if volatility > returns.std() * 1.5:
                if trend > 0:
                    regime = "bull_volatile"
                    regime_score = 0.8
                else:
                    regime = "bear_volatile"
                    regime_score = 0.8
            elif volatility < returns.std() * 0.5:
                if trend > 0:
                    regime = "bull_stable"
                    regime_score = 0.7
                else:
                    regime = "bear_stable"
                    regime_score = 0.7
            else:
                if abs(trend) < returns.std() * 0.1:
                    regime = "sideways"
                    regime_score = 0.6
                elif trend > 0:
                    regime = "bull_normal"
                    regime_score = 0.7
                else:
                    regime = "bear_normal"
                    regime_score = 0.7
            
            # Adjust confidence based on statistical significance
            if abs(skewness) > 1.0 or kurtosis > 3.0:
                regime_score *= 0.9
                
            return regime, regime_score
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return "unknown", 0.0
    
    def assess_regime_stability(self, regimes: List[str]) -> float:
        """Assess stability of regime classification."""
        if len(regimes) < 2:
            return 0.0
            
        # Count regime changes
        changes = sum(1 for i in range(1, len(regimes)) 
                     if regimes[i] != regimes[i-1])
        
        stability = 1.0 - (changes / len(regimes))
        return max(0.0, stability)

class PerformanceAnalyzer:
    """Performance analysis for walk-forward results."""
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        
    def calculate_metrics(self, returns: pd.Series, 
                         benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            metrics = {}
            
            # Basic return metrics
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            
            metrics['total_return'] = total_return
            metrics['annualized_return'] = annualized_return
            metrics['volatility'] = volatility
            
            # Risk-adjusted metrics
            excess_returns = returns - self.config.risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
            metrics['sharpe_ratio'] = sharpe_ratio
            
            # Drawdown analysis
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            metrics['max_drawdown'] = max_drawdown
            metrics['calmar_ratio'] = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(252)
                sortino_ratio = annualized_return / downside_deviation
                metrics['sortino_ratio'] = sortino_ratio
            else:
                metrics['sortino_ratio'] = float('inf')
            
            # Hit rate
            hit_rate = (returns > 0).mean()
            metrics['hit_rate'] = hit_rate
            
            # Information ratio (vs benchmark)
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
                information_ratio = (annualized_return - benchmark_returns.mean() * 252) / tracking_error
                metrics['information_ratio'] = information_ratio
            else:
                metrics['information_ratio'] = 0.0
            
            # VaR and CVaR
            var_95 = returns.quantile(0.05)
            cvar_95 = returns[returns <= var_95].mean()
            
            metrics['var_95'] = var_95
            metrics['cvar_95'] = cvar_95
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def assess_overfitting(self, train_metrics: Dict[str, float], 
                          test_metrics: Dict[str, float]) -> float:
        """Assess degree of overfitting based on performance degradation."""
        try:
            # Compare key metrics between train and test
            key_metrics = ['sharpe_ratio', 'total_return', 'hit_rate']
            degradations = []
            
            for metric in key_metrics:
                if metric in train_metrics and metric in test_metrics:
                    train_val = train_metrics[metric]
                    test_val = test_metrics[metric]
                    
                    if train_val != 0:
                        degradation = (train_val - test_val) / abs(train_val)
                    else:
                        degradation = 0.0 if test_val == 0 else 1.0
                    
                    degradations.append(max(0.0, degradation))
            
            # Return average degradation as overfitting score
            overfitting_score = np.mean(degradations) if degradations else 0.0
            return min(1.0, overfitting_score)
            
        except Exception as e:
            logger.error(f"Error assessing overfitting: {e}")
            return 0.0

class WalkForwardAnalyzer:
    """Main walk-forward analysis engine."""
    
    def __init__(self, config: WalkForwardConfig, db_path: Optional[str] = None):
        self.config = config
        self.db_path = db_path or "validation_results.db"
        self.regime_detector = RegimeDetector(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.results = []
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database for storing results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Walk-forward results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS walk_forward_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    train_start TEXT NOT NULL,
                    train_end TEXT NOT NULL,
                    test_start TEXT NOT NULL,
                    test_end TEXT NOT NULL,
                    train_samples INTEGER,
                    test_samples INTEGER,
                    train_returns REAL,
                    train_sharpe REAL,
                    train_max_drawdown REAL,
                    test_returns REAL,
                    test_sharpe REAL,
                    test_max_drawdown REAL,
                    model_accuracy REAL,
                    prediction_error REAL,
                    regime_detected TEXT,
                    regime_confidence REAL,
                    overfitting_score REAL,
                    execution_time REAL,
                    feature_importance TEXT,
                    warnings TEXT,
                    errors TEXT
                )
            """)
            
            # Summary statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS walk_forward_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    analysis_id TEXT NOT NULL,
                    total_iterations INTEGER,
                    successful_iterations INTEGER,
                    mean_oos_returns REAL,
                    std_oos_returns REAL,
                    mean_oos_sharpe REAL,
                    std_oos_sharpe REAL,
                    performance_stability REAL,
                    overfitting_score REAL,
                    statistical_significance REAL,
                    regime_distribution TEXT,
                    execution_summary TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def generate_splits(self, data: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        """Generate time-aware train/test splits."""
        try:
            splits = []
            data_sorted = data.sort_index()  # Ensure chronological order
            
            start_idx = 0
            max_idx = len(data_sorted)
            
            iteration = 0
            while start_idx + self.config.train_window + self.config.test_window <= max_idx:
                if self.config.max_splits and iteration >= self.config.max_splits:
                    break
                
                # Define indices
                train_end_idx = start_idx + self.config.train_window
                test_end_idx = min(train_end_idx + self.config.test_window, max_idx)
                
                # Create index slices
                train_indices = data_sorted.index[start_idx:train_end_idx]
                test_indices = data_sorted.index[train_end_idx:test_end_idx]
                
                # Ensure minimum sizes
                if len(train_indices) >= self.config.min_train_size and len(test_indices) > 0:
                    splits.append((train_indices, test_indices))
                
                # Advance window
                start_idx += self.config.step_size
                iteration += 1
            
            logger.info(f"Generated {len(splits)} walk-forward splits")
            return splits
            
        except Exception as e:
            logger.error(f"Error generating splits: {e}")
            return []
    
    def validate_single_split(self, data: pd.DataFrame, 
                            train_indices: pd.Index, 
                            test_indices: pd.Index,
                            model_validator: ModelValidator,
                            iteration: int) -> WalkForwardResult:
        """Validate a single train/test split."""
        start_time = datetime.now()
        result = WalkForwardResult(
            iteration=iteration,
            train_start=train_indices[0],
            train_end=train_indices[-1],
            test_start=test_indices[0],
            test_end=test_indices[-1],
            train_samples=len(train_indices),
            test_samples=len(test_indices),
            train_returns=0.0,
            train_sharpe=0.0,
            train_max_drawdown=0.0,
            train_volatility=0.0,
            test_returns=0.0,
            test_sharpe=0.0,
            test_max_drawdown=0.0,
            test_volatility=0.0,
            model_accuracy=0.0,
            prediction_error=0.0
        )
        
        try:
            # Extract train and test data
            train_data = data.loc[train_indices].copy()
            test_data = data.loc[test_indices].copy()
            
            # Prepare features and targets
            feature_cols = [col for col in data.columns if col.startswith('feature_')]
            target_col = 'returns'  # Assume returns column exists
            
            if not feature_cols:
                result.errors.append("No feature columns found")
                return result
            
            if target_col not in data.columns:
                result.errors.append(f"Target column '{target_col}' not found")
                return result
            
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]
            
            # Handle missing values
            if X_train.isnull().any().any() or y_train.isnull().any():
                result.warnings.append("Missing values in training data")
                X_train = X_train.fillna(method='ffill').fillna(0)
                y_train = y_train.fillna(0)
            
            if X_test.isnull().any().any() or y_test.isnull().any():
                result.warnings.append("Missing values in test data")
                X_test = X_test.fillna(method='ffill').fillna(0)
                y_test = y_test.fillna(0)
            
            # Train model
            model_validator.fit(X_train, y_train)
            
            # Make predictions
            train_predictions = model_validator.predict(X_train)
            test_predictions = model_validator.predict(X_test)
            
            # Calculate training metrics
            train_metrics = self.performance_analyzer.calculate_metrics(y_train)
            result.train_returns = train_metrics.get('total_return', 0.0)
            result.train_sharpe = train_metrics.get('sharpe_ratio', 0.0)
            result.train_max_drawdown = train_metrics.get('max_drawdown', 0.0)
            result.train_volatility = train_metrics.get('volatility', 0.0)
            
            # Calculate test metrics
            test_metrics = self.performance_analyzer.calculate_metrics(y_test)
            result.test_returns = test_metrics.get('total_return', 0.0)
            result.test_sharpe = test_metrics.get('sharpe_ratio', 0.0)
            result.test_max_drawdown = test_metrics.get('max_drawdown', 0.0)
            result.test_volatility = test_metrics.get('volatility', 0.0)
            
            # Additional metrics
            result.information_ratio = test_metrics.get('information_ratio', 0.0)
            result.calmar_ratio = test_metrics.get('calmar_ratio', 0.0)
            result.sortino_ratio = test_metrics.get('sortino_ratio', 0.0)
            result.hit_rate = test_metrics.get('hit_rate', 0.0)
            
            # Model performance
            result.model_accuracy = 1.0 - mean_absolute_error(y_test, test_predictions) / y_test.std()
            result.prediction_error = mean_squared_error(y_test, test_predictions)
            
            # Feature importance
            result.feature_importance = model_validator.get_feature_importance()
            
            # Regime detection
            if self.config.use_regime_detection:
                regime_data = pd.concat([train_data, test_data])
                regime, confidence = self.regime_detector.detect_regime(regime_data, target_col)
                result.regime_detected = regime
                result.regime_confidence = confidence
            
            # Execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            error_msg = f"Error in split validation: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.execution_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def run_analysis(self, data: pd.DataFrame, 
                    model_validator: ModelValidator,
                    analysis_id: Optional[str] = None) -> WalkForwardSummary:
        """Run complete walk-forward analysis."""
        analysis_start = datetime.now()
        analysis_id = analysis_id or f"wfa_{analysis_start.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting walk-forward analysis: {analysis_id}")
        
        try:
            # Generate splits
            splits = self.generate_splits(data)
            if not splits:
                raise ValueError("No valid splits generated")
            
            # Run validation
            results = []
            
            if self.config.parallel_processing:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = []
                    for i, (train_idx, test_idx) in enumerate(splits):
                        future = executor.submit(
                            self.validate_single_split,
                            data, train_idx, test_idx, model_validator, i
                        )
                        futures.append(future)
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                            
                            if self.config.save_intermediate:
                                self._save_result(result, analysis_id)
                                
                        except Exception as e:
                            logger.error(f"Error in parallel execution: {e}")
            else:
                # Sequential execution
                for i, (train_idx, test_idx) in enumerate(splits):
                    result = self.validate_single_split(
                        data, train_idx, test_idx, model_validator, i
                    )
                    results.append(result)
                    
                    if self.config.save_intermediate:
                        self._save_result(result, analysis_id)
            
            # Generate summary
            summary = self._generate_summary(results, analysis_id)
            
            # Save summary
            self._save_summary(summary, analysis_id)
            
            # Store results
            self.results = results
            
            total_time = (datetime.now() - analysis_start).total_seconds()
            logger.info(f"Walk-forward analysis completed in {total_time:.2f}s")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {e}")
            raise
    
    def _generate_summary(self, results: List[WalkForwardResult], 
                         analysis_id: str) -> WalkForwardSummary:
        """Generate summary statistics from results."""
        try:
            successful_results = [r for r in results if not r.errors]
            failed_results = [r for r in results if r.errors]
            
            if not successful_results:
                raise ValueError("No successful validation iterations")
            
            # Extract metrics
            oos_returns = [r.test_returns for r in successful_results]
            oos_sharpe = [r.test_sharpe for r in successful_results]
            train_returns = [r.train_returns for r in successful_results]
            train_sharpe = [r.train_sharpe for r in successful_results]
            
            # Basic statistics
            mean_oos_returns = np.mean(oos_returns)
            std_oos_returns = np.std(oos_returns)
            mean_oos_sharpe = np.mean(oos_sharpe)
            std_oos_sharpe = np.std(oos_sharpe)
            
            # Performance stability
            performance_stability = 1.0 - (std_oos_returns / abs(mean_oos_returns)) if mean_oos_returns != 0 else 0.0
            
            # Statistical significance test
            if len(oos_returns) > 1:
                t_stat, p_value = stats.ttest_1samp(oos_returns, 0)
                is_significant = p_value < (1 - self.config.confidence_level)
                
                # Confidence interval
                confidence_interval = stats.t.interval(
                    self.config.confidence_level,
                    len(oos_returns) - 1,
                    loc=mean_oos_returns,
                    scale=stats.sem(oos_returns)
                )
            else:
                is_significant = False
                p_value = 1.0
                confidence_interval = (mean_oos_returns, mean_oos_returns)
            
            # Regime analysis
            regimes = [r.regime_detected for r in successful_results if r.regime_detected]
            regime_distribution = {}
            regime_performance = {}
            
            if regimes:
                regime_distribution = {regime: regimes.count(regime) for regime in set(regimes)}
                
                for regime in set(regimes):
                    regime_results = [r for r in successful_results if r.regime_detected == regime]
                    regime_returns = [r.test_returns for r in regime_results]
                    regime_performance[regime] = np.mean(regime_returns) if regime_returns else 0.0
            
            # Regime consistency
            regime_stability = self.regime_detector.assess_regime_stability(regimes)
            
            # Risk metrics
            worst_case_return = min(oos_returns) if oos_returns else 0.0
            worst_case_drawdown = min([r.test_max_drawdown for r in successful_results])
            
            # VaR and CVaR
            oos_returns_array = np.array(oos_returns)
            var_95 = np.percentile(oos_returns_array, 5)
            cvar_95 = oos_returns_array[oos_returns_array <= var_95].mean() if len(oos_returns_array[oos_returns_array <= var_95]) > 0 else var_95
            
            # Model assessment
            overfitting_scores = []
            for r in successful_results:
                train_metrics = {'sharpe_ratio': r.train_sharpe, 'total_return': r.train_returns}
                test_metrics = {'sharpe_ratio': r.test_sharpe, 'total_return': r.test_returns}
                overfitting_score = self.performance_analyzer.assess_overfitting(train_metrics, test_metrics)
                overfitting_scores.append(overfitting_score)
            
            mean_overfitting = np.mean(overfitting_scores) if overfitting_scores else 0.0
            
            # Generalization ability
            correlation_train_test = np.corrcoef(train_returns, oos_returns)[0, 1] if len(train_returns) > 1 else 0.0
            generalization_ability = max(0.0, correlation_train_test)
            
            # Feature stability
            feature_stability = {}
            if successful_results and successful_results[0].feature_importance:
                all_features = set()
                for r in successful_results:
                    all_features.update(r.feature_importance.keys())
                
                for feature in all_features:
                    importances = []
                    for r in successful_results:
                        if feature in r.feature_importance:
                            importances.append(r.feature_importance[feature])
                        else:
                            importances.append(0.0)
                    
                    if importances:
                        feature_stability[feature] = 1.0 - (np.std(importances) / (np.mean(importances) + 1e-8))
            
            # Execution statistics
            execution_times = [r.execution_time for r in results]
            total_execution_time = sum(execution_times)
            average_iteration_time = np.mean(execution_times) if execution_times else 0.0
            
            # Create summary
            summary = WalkForwardSummary(
                total_iterations=len(results),
                successful_iterations=len(successful_results),
                failed_iterations=len(failed_results),
                mean_oos_returns=mean_oos_returns,
                std_oos_returns=std_oos_returns,
                mean_oos_sharpe=mean_oos_sharpe,
                std_oos_sharpe=std_oos_sharpe,
                performance_stability=performance_stability,
                regime_consistency=regime_stability,
                model_reliability=1.0 - mean_overfitting,
                is_statistically_significant=is_significant,
                p_value=p_value,
                confidence_interval=confidence_interval,
                regime_distribution=regime_distribution,
                regime_performance=regime_performance,
                worst_case_return=worst_case_return,
                worst_case_drawdown=worst_case_drawdown,
                var_95=var_95,
                cvar_95=cvar_95,
                overfitting_score=mean_overfitting,
                generalization_ability=generalization_ability,
                feature_stability=feature_stability,
                total_execution_time=total_execution_time,
                average_iteration_time=average_iteration_time,
                memory_efficiency=1.0  # Placeholder
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise
    
    def _save_result(self, result: WalkForwardResult, analysis_id: str):
        """Save individual result to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO walk_forward_results (
                    timestamp, iteration, train_start, train_end, test_start, test_end,
                    train_samples, test_samples, train_returns, train_sharpe, train_max_drawdown,
                    test_returns, test_sharpe, test_max_drawdown, model_accuracy, prediction_error,
                    regime_detected, regime_confidence, overfitting_score, execution_time,
                    feature_importance, warnings, errors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                result.iteration,
                result.train_start.isoformat(),
                result.train_end.isoformat(),
                result.test_start.isoformat(),
                result.test_end.isoformat(),
                result.train_samples,
                result.test_samples,
                result.train_returns,
                result.train_sharpe,
                result.train_max_drawdown,
                result.test_returns,
                result.test_sharpe,
                result.test_max_drawdown,
                result.model_accuracy,
                result.prediction_error,
                result.regime_detected,
                result.regime_confidence,
                0.0,  # overfitting_score placeholder
                result.execution_time,
                json.dumps(result.feature_importance),
                json.dumps(result.warnings),
                json.dumps(result.errors)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving result: {e}")
    
    def _save_summary(self, summary: WalkForwardSummary, analysis_id: str):
        """Save summary to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            execution_summary = {
                'total_execution_time': summary.total_execution_time,
                'average_iteration_time': summary.average_iteration_time,
                'memory_efficiency': summary.memory_efficiency
            }
            
            cursor.execute("""
                INSERT INTO walk_forward_summary (
                    timestamp, analysis_id, total_iterations, successful_iterations,
                    mean_oos_returns, std_oos_returns, mean_oos_sharpe, std_oos_sharpe,
                    performance_stability, overfitting_score, statistical_significance,
                    regime_distribution, execution_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                analysis_id,
                summary.total_iterations,
                summary.successful_iterations,
                summary.mean_oos_returns,
                summary.std_oos_returns,
                summary.mean_oos_sharpe,
                summary.std_oos_sharpe,
                summary.performance_stability,
                summary.overfitting_score,
                1.0 if summary.is_statistically_significant else 0.0,
                json.dumps(summary.regime_distribution),
                json.dumps(execution_summary)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
    
    def get_analysis_results(self, analysis_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve analysis results from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if analysis_id:
                query = """
                    SELECT * FROM walk_forward_summary 
                    WHERE analysis_id = ? 
                    ORDER BY timestamp DESC
                """
                df_summary = pd.read_sql_query(query, conn, params=[analysis_id])
            else:
                query = """
                    SELECT * FROM walk_forward_summary 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """
                df_summary = pd.read_sql_query(query, conn)
            
            # Get detailed results
            if not df_summary.empty:
                query = """
                    SELECT * FROM walk_forward_results 
                    WHERE timestamp >= ? 
                    ORDER BY iteration
                """
                timestamp = df_summary.iloc[0]['timestamp']
                df_results = pd.read_sql_query(query, conn, params=[timestamp])
            else:
                df_results = pd.DataFrame()
            
            conn.close()
            
            return {
                'summary': df_summary.to_dict('records'),
                'results': df_results.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            return {'summary': [], 'results': []}

def create_sample_validator() -> ModelValidator:
    """Create a sample model validator for testing."""
    from sklearn.ensemble import RandomForestRegressor
    
    class SampleValidator(ModelValidator):
        def __init__(self):
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.feature_names = []
            
        def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
            self.feature_names = list(X_train.columns)
            self.model.fit(X_train, y_train)
            
        def predict(self, X_test: pd.DataFrame) -> np.ndarray:
            return self.model.predict(X_test)
            
        def get_feature_importance(self) -> Dict[str, float]:
            if hasattr(self.model, 'feature_importances_'):
                return dict(zip(self.feature_names, self.model.feature_importances_))
            return {}
            
        def get_model_params(self) -> Dict[str, Any]:
            return self.model.get_params()
    
    return SampleValidator()

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Generate synthetic features and returns
    data = pd.DataFrame(index=dates)
    data['returns'] = np.random.normal(0.001, 0.02, n_samples)
    data['feature_momentum'] = np.random.normal(0, 1, n_samples)
    data['feature_volatility'] = np.random.exponential(1, n_samples)
    data['feature_volume'] = np.random.lognormal(0, 1, n_samples)
    
    # Add some persistence to make it more realistic
    for i in range(1, n_samples):
        data.iloc[i, 0] += 0.1 * data.iloc[i-1, 0] + np.random.normal(0, 0.01)
    
    # Run walk-forward analysis
    config = WalkForwardConfig(
        train_window=252,
        test_window=21,
        step_size=21,
        n_splits=10,
        parallel_processing=False
    )
    
    analyzer = WalkForwardAnalyzer(config)
    validator = create_sample_validator()
    
    try:
        summary = analyzer.run_analysis(data, validator, "sample_analysis")
        
        print("\n=== Walk-Forward Analysis Summary ===")
        print(f"Total Iterations: {summary.total_iterations}")
        print(f"Successful Iterations: {summary.successful_iterations}")
        print(f"Mean OOS Returns: {summary.mean_oos_returns:.4f}")
        print(f"Mean OOS Sharpe: {summary.mean_oos_sharpe:.4f}")
        print(f"Performance Stability: {summary.performance_stability:.4f}")
        print(f"Overfitting Score: {summary.overfitting_score:.4f}")
        print(f"Statistical Significance: {summary.is_statistically_significant}")
        print(f"P-Value: {summary.p_value:.4f}")
        print(f"Regime Distribution: {summary.regime_distribution}")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        traceback.print_exc()