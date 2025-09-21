"""
Purged TimeSeries Cross-Validation for Machine Learning models.

This module implements time-series cross-validation with purging and embargo
to prevent data leakage in financial ML models. It provides:

- Purged CV splits that remove overlapping observations
- Embargo periods to prevent look-ahead bias
- Group-aware splitting for correlated observations
- Walk-forward time series validation
- Nested CV for hyperparameter tuning

Reference: "Advances in Financial Machine Learning" by Marcos López de Prado

The process:
1. Generate time-ordered train/validation splits
2. Apply purging to remove overlapping observations
3. Apply embargo to prevent information leakage
4. Handle correlated features through grouping
5. Validate model performance on clean test data
"""

import warnings
from typing import Dict, List, Optional, Tuple, Iterator, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, ParameterGrid
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..utils.logging import TradingLogger


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Time Series cross-validator with purging and embargo.
    
    This cross-validator generates train/test splits for time series data
    while preventing data leakage through purging and embargo.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        purge_gap: int = 0,
        embargo_gap: int = 0,
        time_column: str = 'timestamp',
        group_column: Optional[str] = None
    ):
        """
        Initialize Purged TimeSeries Split.
        
        Args:
            n_splits: Number of splits
            max_train_size: Maximum size of training set
            test_size: Size of test set (if None, uses remaining data)
            purge_gap: Number of observations to purge between train/test
            embargo_gap: Number of observations to embargo after test set
            time_column: Name of time column for ordering
            group_column: Name of group column for correlated observations
        """
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
        self.time_column = time_column
        self.group_column = group_column
        
        self.logger = TradingLogger("PurgedTimeSeriesSplit")
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Feature matrix or DataFrame
            y: Target vector (unused)
            groups: Group labels (unused, use group_column instead)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        # Handle DataFrame input
        if hasattr(X, 'index'):
            n_samples = len(X)  # type: ignore
            
            # Sort by time if DataFrame
            if hasattr(X, self.time_column):
                sorted_indices = X.sort_values(self.time_column).index.values  # type: ignore
            else:
                sorted_indices = X.index.values  # type: ignore
        else:
            n_samples = X.shape[0]  # type: ignore
            sorted_indices = np.arange(n_samples)
        
        # Calculate split parameters
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        if test_size <= 0:
            raise ValueError(f"Test size must be positive, got {test_size}")
        
        # Generate splits
        for i in range(self.n_splits):
            # Calculate test set boundaries
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            
            if test_start < 0:
                continue
                
            test_end = min(test_end, n_samples)
            
            # Apply embargo: test set ends earlier
            test_end_embargoed = test_end - self.embargo_gap
            if test_end_embargoed <= test_start:
                continue
            
            # Calculate training set boundaries
            train_end = test_start - self.purge_gap
            if train_end <= 0:
                continue
            
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0
            
            # Get indices
            train_indices = sorted_indices[train_start:train_end]
            test_indices = sorted_indices[test_start:test_end_embargoed]
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
            
            # Apply group purging if specified
            if self.group_column is not None and hasattr(X, self.group_column):
                train_indices, test_indices = self._apply_group_purging(
                    X, train_indices, test_indices
                )
            
            # Convert to positions in original array if needed
            if hasattr(X, 'index'):
                train_pos = np.array([np.where(X.index == idx)[0][0] for idx in train_indices if idx in X.index])  # type: ignore
                test_pos = np.array([np.where(X.index == idx)[0][0] for idx in test_indices if idx in X.index])  # type: ignore
            else:
                train_pos = train_indices  # type: ignore
                test_pos = test_indices  # type: ignore
            
            if len(train_pos) == 0 or len(test_pos) == 0:
                continue
                
            yield train_pos, test_pos
    
    def _apply_group_purging(self, X, train_indices, test_indices):
        """Apply group-based purging to prevent information leakage."""
        if self.group_column is None or not hasattr(X, self.group_column):
            return train_indices, test_indices
        
        # Get groups in test set
        test_groups = set(X.loc[test_indices, self.group_column].unique())  # type: ignore
        
        # Remove training observations from same groups
        train_mask = ~X.loc[train_indices, self.group_column].isin(test_groups)  # type: ignore
        purged_train_indices = train_indices[train_mask.values]  # type: ignore
        
        return purged_train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


class PurgedCVResult:
    """Container for purged cross-validation results."""
    
    def __init__(self):
        self.fold_results: List[Dict] = []
        self.mean_scores: Dict[str, float] = {}
        self.std_scores: Dict[str, float] = {}
        self.feature_importance: Optional[pd.DataFrame] = None
        self.best_params: Optional[Dict] = None
        self.validation_summary: Dict = {}
    
    def calculate_summary_stats(self):
        """Calculate summary statistics across folds."""
        if not self.fold_results:
            return
        
        # Extract scores from all folds
        all_scores = {}
        for fold_result in self.fold_results:
            for metric, score in fold_result.get('scores', {}).items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
        
        # Calculate mean and std
        for metric, scores in all_scores.items():
            self.mean_scores[metric] = np.mean(scores)
            self.std_scores[metric] = float(np.std(scores))
        
        # Calculate feature importance summary
        if all(fold.get('feature_importance') is not None for fold in self.fold_results):
            importance_dfs = [fold['feature_importance'] for fold in self.fold_results]
            self.feature_importance = pd.concat(importance_dfs).groupby('feature').agg({
                'importance': ['mean', 'std', 'min', 'max']
            }).round(4)


class PurgedTimeSeriesCV:
    """
    Purged Time Series Cross-Validation for financial ML models.
    
    This class provides comprehensive time series cross-validation with
    purging, embargo, and proper handling of financial data characteristics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("PurgedTimeSeriesCV")
    
    def _default_config(self) -> Dict:
        """Default configuration for Purged TimeSeries CV."""
        return {
            'n_splits': 5,
            'max_train_size': None,
            'test_size': None,
            'purge_gap': 1,
            'embargo_gap': 1,
            'time_column': 'timestamp',
            'group_column': None,
            'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1'],
            'n_jobs': -1,
            'random_state': 42,
        }
    
    def cross_validate(
        self,
        estimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict] = None,
        scoring: Optional[List[str]] = None
    ) -> PurgedCVResult:
        """
        Perform purged cross-validation with optional hyperparameter tuning.
        
        Args:
            estimator: ML model to validate
            X: Feature matrix
            y: Target vector
            param_grid: Parameters for grid search
            scoring: Scoring metrics to use
            
        Returns:
            PurgedCVResult with validation results
        """
        scoring = scoring or self.config['scoring_metrics']
        
        self.logger.info(
            f"Starting purged CV with {self.config['n_splits']} splits"
        )
        
        # Validate inputs
        if X.empty or y.empty:
            raise ValueError("X and y cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        # Initialize CV splitter
        cv_splitter = PurgedTimeSeriesSplit(
            n_splits=self.config['n_splits'],
            max_train_size=self.config['max_train_size'],
            test_size=self.config['test_size'],
            purge_gap=self.config['purge_gap'],
            embargo_gap=self.config['embargo_gap'],
            time_column=self.config['time_column'],
            group_column=self.config['group_column']
        )
        
        result = PurgedCVResult()
        
        # Perform cross-validation
        fold_num = 0
        for train_idx, test_idx in cv_splitter.split(X, y):
            fold_num += 1
            
            self.logger.debug(f"Processing fold {fold_num}/{self.config['n_splits']}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Perform hyperparameter tuning if param_grid provided
            if param_grid:
                best_estimator = self._grid_search_fold(
                    estimator, X_train, y_train, param_grid
                )
            else:
                best_estimator = clone(estimator)
                best_estimator.fit(X_train, y_train)
            
            # Make predictions
            y_pred = best_estimator.predict(X_test)
            
            # Calculate scores
            fold_scores = self._calculate_scores(y_test, y_pred, scoring)
            
            # Get feature importance if available
            feature_importance = self._get_feature_importance(
                best_estimator, X_train.columns
            )
            
            # Store fold results
            fold_result = {
                'fold': fold_num,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_period': (X_train.index.min(), X_train.index.max()),
                'test_period': (X_test.index.min(), X_test.index.max()),
                'scores': fold_scores,
                'feature_importance': feature_importance,
                'best_params': getattr(best_estimator, 'get_params', lambda: {})()
            }
            
            result.fold_results.append(fold_result)
        
        # Calculate summary statistics
        result.calculate_summary_stats()
        
        self.logger.info(
            f"Purged CV completed: {len(result.fold_results)} folds processed"
        )
        
        return result
    
    def _grid_search_fold(self, estimator, X_train, y_train, param_grid):
        """Perform grid search for a single fold."""
        from sklearn.model_selection import GridSearchCV
        
        # Use inner CV for hyperparameter tuning
        inner_cv = PurgedTimeSeriesSplit(
            n_splits=3,  # Fewer splits for inner CV
            purge_gap=self.config['purge_gap'],
            embargo_gap=self.config['embargo_gap'],
            time_column=self.config['time_column']
        )
        
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='accuracy',  # Primary metric for selection
            n_jobs=1,  # Avoid nested parallelism
            error_score='raise'
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    def _calculate_scores(self, y_true, y_pred, scoring_metrics):
        """Calculate various scoring metrics."""
        scores = {}
        
        try:
            if 'accuracy' in scoring_metrics:
                scores['accuracy'] = accuracy_score(y_true, y_pred)
            
            if 'precision' in scoring_metrics:
                scores['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            
            if 'recall' in scoring_metrics:
                scores['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            
            if 'f1' in scoring_metrics:
                scores['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
        except Exception as e:
            self.logger.warning(f"Error calculating scores: {e}")
            # Return zeros for failed metrics
            for metric in scoring_metrics:
                if metric not in scores:
                    scores[metric] = 0.0
        
        return scores
    
    def _get_feature_importance(self, estimator, feature_names):
        """Extract feature importance from fitted estimator."""
        try:
            if hasattr(estimator, 'feature_importances_'):
                importance = estimator.feature_importances_
            elif hasattr(estimator, 'coef_'):
                importance = np.abs(estimator.coef_).flatten()
            else:
                return None
            
            if len(importance) != len(feature_names):
                return None
            
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    def walk_forward_validation(
        self,
        estimator,
        X: pd.DataFrame,
        y: pd.Series,
        window_size: int,
        step_size: int = 1,
        min_train_size: int = 100
    ) -> PurgedCVResult:
        """
        Perform walk-forward validation.
        
        Args:
            estimator: ML model to validate
            X: Feature matrix
            y: Target vector
            window_size: Size of rolling window
            step_size: Step size for window advancement
            min_train_size: Minimum training set size
            
        Returns:
            PurgedCVResult with walk-forward results
        """
        self.logger.info(
            f"Starting walk-forward validation with window size {window_size}"
        )
        
        result = PurgedCVResult()
        n_samples = len(X)
        
        fold_num = 0
        for start_idx in range(min_train_size, n_samples - window_size, step_size):
            fold_num += 1
            
            # Define train and test periods
            train_start = max(0, start_idx - window_size)
            train_end = start_idx
            test_start = start_idx + self.config['purge_gap']
            test_end = min(test_start + window_size, n_samples)
            
            if test_end <= test_start:
                continue
            
            # Split data
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            if len(X_train) < min_train_size or len(X_test) == 0:
                continue
            
            # Fit model and predict
            model = clone(estimator)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate scores
            fold_scores = self._calculate_scores(
                y_test, y_pred, self.config['scoring_metrics']
            )
            
            # Store results
            fold_result = {
                'fold': fold_num,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_period': (X_train.index.min(), X_train.index.max()),
                'test_period': (X_test.index.min(), X_test.index.max()),
                'scores': fold_scores,
            }
            
            result.fold_results.append(fold_result)
        
        result.calculate_summary_stats()
        
        self.logger.info(
            f"Walk-forward validation completed: {len(result.fold_results)} windows processed"
        )
        
        return result
    
    def generate_report(self, result: PurgedCVResult) -> str:
        """Generate comprehensive validation report."""
        report = f"""
Purged Time Series Cross-Validation Report
{'='*60}

Configuration:
- Number of Splits: {self.config['n_splits']}
- Purge Gap: {self.config['purge_gap']}
- Embargo Gap: {self.config['embargo_gap']}
- Time Column: {self.config['time_column']}

Results Summary:
"""
        
        # Add mean scores
        for metric, mean_score in result.mean_scores.items():
            std_score = result.std_scores.get(metric, 0)
            report += f"- {metric.title()}: {mean_score:.4f} ± {std_score:.4f}\n"
        
        # Add fold details
        report += f"\nFold Details:\n"
        for fold in result.fold_results:
            report += f"Fold {fold['fold']}: "
            report += f"Train={fold['train_size']}, Test={fold['test_size']}, "
            report += f"Accuracy={fold['scores'].get('accuracy', 0):.4f}\n"
        
        # Add feature importance if available
        if result.feature_importance is not None:
            report += f"\nTop 10 Features by Importance:\n"
            top_features = result.feature_importance.head(10)
            for _, row in top_features.iterrows():
                report += f"- {row.name}: {row[('importance', 'mean')]:.4f}\n"
        
        return report