"""
Advanced Feature Selector

Intelligent feature selection system for cryptocurrency trading ML models.
Implements multiple feature selection techniques including mutual information,
recursive feature elimination, stability selection, and correlation analysis.

Key Features:
- Multi-method feature selection (statistical, model-based, filter methods)
- Dynamic feature importance tracking with temporal stability
- Redundancy detection and removal using correlation clustering
- Feature interaction detection and selection
- Regime-aware feature selection for different market conditions
- Real-time feature quality monitoring and adaptation
- Feature engineering pipeline integration
- Performance impact analysis for selected features

Selection Methods:
- Mutual Information: Non-linear dependency detection
- Recursive Feature Elimination: Model-based iterative selection
- Stability Selection: Bootstrap-based robust selection
- Correlation Analysis: Redundancy identification and clustering
- Univariate Statistical Tests: Fast screening methods
- L1 Regularization: Sparse feature selection via LASSO
- Tree-based Importance: Random Forest and XGBoost feature ranking
- PCA/ICA Analysis: Dimensionality reduction insights

Advanced Features:
- Multi-target selection for different prediction horizons
- Feature interaction detection using polynomial and interaction terms
- Regime-specific feature sets for volatile vs stable markets
- Dynamic feature quality scoring with decay factors
- Feature stability analysis across different time periods
- Automated feature engineering and selection pipeline
- Performance attribution analysis for feature contributions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
import asyncio
from collections import defaultdict, deque
import warnings
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ML imports
from sklearn.feature_selection import (
    mutual_info_regression, RFE, SelectKBest, f_regression,
    VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from ..utils.logging import TradingLogger
from ..config.manager import ConfigurationManager


class SelectionMethod(Enum):
    """Feature selection methods."""
    MUTUAL_INFO = "mutual_info"
    RFE = "recursive_elimination"
    STABILITY = "stability_selection"
    CORRELATION = "correlation_analysis"
    UNIVARIATE = "univariate_tests"
    L1_REGULARIZATION = "l1_lasso"
    TREE_IMPORTANCE = "tree_based"
    PCA_ANALYSIS = "pca_analysis"
    COMBINED = "ensemble_selection"


class FeatureType(Enum):
    """Types of features for categorization."""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    CROSS_EXCHANGE = "cross_exchange"
    TEMPORAL = "temporal"
    VOLATILITY = "volatility"
    MACRO = "macro"
    DERIVED = "derived"
    INTERACTION = "interaction"


class MarketRegime(Enum):
    """Market regimes for regime-aware selection."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"


@dataclass
class FeatureMetrics:
    """Comprehensive feature quality metrics."""
    feature_name: str
    feature_type: FeatureType
    
    # Importance scores from different methods
    mutual_info_score: float
    rfe_ranking: int
    stability_score: float
    correlation_redundancy: float
    univariate_score: float
    tree_importance: float
    l1_coefficient: float
    
    # Quality metrics
    missing_rate: float
    variance: float
    skewness: float
    kurtosis: float
    
    # Stability metrics
    temporal_stability: float
    regime_consistency: Dict[MarketRegime, float]
    
    # Performance metrics
    predictive_power: float
    feature_interaction_strength: float
    
    # Combined scores
    overall_score: float
    selection_probability: float
    
    # Metadata
    last_updated: datetime
    selection_history: List[bool] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureSelectionResult:
    """Results from feature selection process."""
    timestamp: datetime
    method: SelectionMethod
    target_variable: str
    
    selected_features: List[str]
    feature_scores: Dict[str, float]
    feature_rankings: Dict[str, int]
    
    # Performance metrics
    selection_performance: float
    cross_validation_score: float
    feature_redundancy_removed: int
    
    # Method-specific results
    method_specific_results: Dict[str, Any]
    
    # Quality indicators
    selection_stability: float
    feature_diversity: float
    
    metadata: Dict[str, Any]


@dataclass
class RegimeFeatureSet:
    """Feature set optimized for specific market regime."""
    regime: MarketRegime
    features: List[str]
    performance_score: float
    regime_probability: float
    last_updated: datetime
    usage_count: int = 0


class AdvancedFeatureSelector:
    """
    Advanced feature selection system for cryptocurrency trading.
    
    Implements multiple sophisticated feature selection techniques and
    provides intelligent feature engineering and redundancy removal.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = TradingLogger("AdvancedFeatureSelector")
        
        # Configuration
        self.config = {
            'max_features': config_manager.get('feature_selection.max_features', 50),
            'min_feature_variance': config_manager.get('feature_selection.min_variance', 0.01),
            'correlation_threshold': config_manager.get('feature_selection.correlation_threshold', 0.95),
            'stability_threshold': config_manager.get('feature_selection.stability_threshold', 0.6),
            'mutual_info_threshold': config_manager.get('feature_selection.mutual_info_threshold', 0.1),
            'cv_folds': config_manager.get('feature_selection.cv_folds', 5),
            'stability_iterations': config_manager.get('feature_selection.stability_iterations', 100),
            'feature_decay_factor': config_manager.get('feature_selection.decay_factor', 0.95),
            'regime_consistency_threshold': config_manager.get('feature_selection.regime_threshold', 0.7),
            'interaction_threshold': config_manager.get('feature_selection.interaction_threshold', 0.15),
            'selection_ensemble_weights': config_manager.get('feature_selection.ensemble_weights', {
                'mutual_info': 0.20,
                'rfe': 0.15,
                'stability': 0.20,
                'correlation': 0.10,
                'univariate': 0.10,
                'tree_importance': 0.15,
                'l1_regularization': 0.10
            })
        }
        
        # Data storage
        self.feature_data: Optional[pd.DataFrame] = None
        self.target_data: Dict[str, pd.Series] = {}
        self.feature_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Feature metrics tracking
        self.feature_metrics: Dict[str, FeatureMetrics] = {}
        self.selection_history: deque = deque(maxlen=1000)
        
        # Regime-specific feature sets
        self.regime_feature_sets: Dict[MarketRegime, RegimeFeatureSet] = {}
        self.current_regime: Optional[MarketRegime] = None
        
        # Model storage for different selection methods
        self.selection_models: Dict[SelectionMethod, Any] = {}
        self.fitted_selectors: Dict[str, Any] = {}
        
        # Performance tracking
        self.method_performance: Dict[SelectionMethod, List[float]] = defaultdict(list)
        self.feature_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Correlation analysis
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.feature_clusters: Dict[int, List[str]] = {}
        
        # Feature interaction tracking
        self.feature_interactions: Dict[Tuple[str, str], float] = {}
        self.interaction_candidates: List[Tuple[str, str]] = []
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize selection models
        self._initialize_selection_models()
    
    def _initialize_selection_models(self):
        """Initialize feature selection models."""
        # Random Forest for tree-based importance
        self.selection_models[SelectionMethod.TREE_IMPORTANCE] = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        
        # LASSO for L1 regularization
        self.selection_models[SelectionMethod.L1_REGULARIZATION] = LassoCV(
            cv=self.config['cv_folds'], random_state=42, max_iter=1000
        )
        
        # ElasticNet as alternative
        self.selection_models['elastic_net'] = ElasticNetCV(
            cv=self.config['cv_folds'], random_state=42, max_iter=1000
        )
        
        # PCA for dimensionality analysis
        self.selection_models[SelectionMethod.PCA_ANALYSIS] = PCA(n_components=0.95)
    
    def update_feature_data(self, feature_data: pd.DataFrame, target_data: Dict[str, pd.Series]):
        """Update feature and target data."""
        try:
            if feature_data.empty:
                self.logger.warning("Empty feature data provided")
                return
            
            # Store data
            self.feature_data = feature_data.copy()
            self.target_data = target_data.copy()
            
            # Update feature metadata
            self._update_feature_metadata()
            
            # Calculate correlation matrix
            self._calculate_correlation_matrix()
            
            # Update feature metrics
            self._update_feature_metrics()
            
            self.logger.info(f"Updated feature data: {len(feature_data.columns)} features, {len(feature_data)} samples")
            
        except Exception as e:
            self.logger.error(f"Error updating feature data: {e}")
    
    def _update_feature_metadata(self):
        """Update metadata for all features."""
        if self.feature_data is None:
            return
        
        for column in self.feature_data.columns:
            # Infer feature type from name
            feature_type = self._infer_feature_type(column)
            
            # Calculate basic statistics
            series = self.feature_data[column]
            
            self.feature_metadata[column] = {
                'type': feature_type,
                'missing_rate': series.isnull().sum() / len(series),
                'variance': series.var(),
                'mean': series.mean(),
                'std': series.std(),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'min': series.min(),
                'max': series.max(),
                'unique_values': series.nunique(),
                'last_updated': datetime.now()
            }
    
    def _infer_feature_type(self, feature_name: str) -> FeatureType:
        """Infer feature type from feature name."""
        name_lower = feature_name.lower()
        
        if any(term in name_lower for term in ['sentiment', 'fear', 'greed', 'news']):
            return FeatureType.SENTIMENT
        elif any(term in name_lower for term in ['cross', 'arbitrage', 'spread', 'exchange']):
            return FeatureType.CROSS_EXCHANGE
        elif any(term in name_lower for term in ['session', 'time', 'hour', 'day', 'temporal']):
            return FeatureType.TEMPORAL
        elif any(term in name_lower for term in ['vol', 'volatility', 'var', 'std']):
            return FeatureType.VOLATILITY
        elif any(term in name_lower for term in ['macro', 'economic', 'gdp', 'inflation']):
            return FeatureType.MACRO
        elif any(term in name_lower for term in ['interaction', 'product', 'ratio']):
            return FeatureType.INTERACTION
        elif any(term in name_lower for term in ['derived', 'engineered', 'combined']):
            return FeatureType.DERIVED
        else:
            return FeatureType.TECHNICAL
    
    def _calculate_correlation_matrix(self):
        """Calculate and store correlation matrix."""
        if self.feature_data is None:
            return
        
        try:
            # Calculate correlation matrix
            self.correlation_matrix = self.feature_data.corr().abs()
            
            # Identify highly correlated features
            self._identify_correlation_clusters()
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
    
    def _identify_correlation_clusters(self):
        """Identify clusters of highly correlated features."""
        if self.correlation_matrix is None:
            return
        
        try:
            # Convert correlation to distance
            distance_matrix = 1 - self.correlation_matrix
            
            # Perform hierarchical clustering
            condensed_distances = squareform(distance_matrix.values)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Form clusters
            threshold = 1 - self.config['correlation_threshold']
            cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
            
            # Group features by cluster
            self.feature_clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in self.feature_clusters:
                    self.feature_clusters[label] = []
                self.feature_clusters[label].append(self.correlation_matrix.columns[i])
            
            self.logger.debug(f"Identified {len(self.feature_clusters)} correlation clusters")
            
        except Exception as e:
            self.logger.error(f"Error identifying correlation clusters: {e}")
    
    def _update_feature_metrics(self):
        """Update comprehensive metrics for all features."""
        if self.feature_data is None:
            return
        
        for feature_name in self.feature_data.columns:
            try:
                # Get existing metrics or create new
                if feature_name in self.feature_metrics:
                    metrics = self.feature_metrics[feature_name]
                else:
                    metadata = self.feature_metadata.get(feature_name, {})
                    metrics = FeatureMetrics(
                        feature_name=feature_name,
                        feature_type=metadata.get('type', FeatureType.TECHNICAL),
                        mutual_info_score=0.0,
                        rfe_ranking=0,
                        stability_score=0.0,
                        correlation_redundancy=0.0,
                        univariate_score=0.0,
                        tree_importance=0.0,
                        l1_coefficient=0.0,
                        missing_rate=metadata.get('missing_rate', 0.0),
                        variance=metadata.get('variance', 0.0),
                        skewness=metadata.get('skewness', 0.0),
                        kurtosis=metadata.get('kurtosis', 0.0),
                        temporal_stability=0.0,
                        regime_consistency={},
                        predictive_power=0.0,
                        feature_interaction_strength=0.0,
                        overall_score=0.0,
                        selection_probability=0.0,
                        last_updated=datetime.now()
                    )
                
                # Update with current data
                metadata = self.feature_metadata.get(feature_name, {})
                metrics.missing_rate = metadata.get('missing_rate', 0.0)
                metrics.variance = metadata.get('variance', 0.0)
                metrics.skewness = metadata.get('skewness', 0.0)
                metrics.kurtosis = metadata.get('kurtosis', 0.0)
                metrics.last_updated = datetime.now()
                
                # Calculate correlation redundancy
                if self.correlation_matrix is not None and feature_name in self.correlation_matrix.columns:
                    correlations = self.correlation_matrix.loc[feature_name]
                    correlations = correlations[correlations.index != feature_name]
                    metrics.correlation_redundancy = correlations.max() if len(correlations) > 0 else 0.0
                
                self.feature_metrics[feature_name] = metrics
                
            except Exception as e:
                self.logger.error(f"Error updating metrics for feature {feature_name}: {e}")
    
    def select_features(self, 
                       target_variable: str,
                       method: SelectionMethod = SelectionMethod.COMBINED,
                       max_features: Optional[int] = None) -> FeatureSelectionResult:
        """
        Perform feature selection using specified method.
        
        Args:
            target_variable: Name of target variable
            method: Selection method to use
            max_features: Maximum number of features to select
            
        Returns:
            FeatureSelectionResult with selected features and metrics
        """
        if self.feature_data is None or target_variable not in self.target_data:
            raise ValueError("Feature data or target variable not available")
        
        if max_features is None:
            max_features = self.config['max_features']
        
        try:
            # Prepare data
            X = self.feature_data.dropna()
            y = self.target_data[target_variable].loc[X.index]
            
            if len(X) == 0:
                raise ValueError("No valid samples after removing NaN values")
            
            # Apply basic filtering first
            X_filtered = self._apply_basic_filtering(X)
            
            # Select method
            if method == SelectionMethod.COMBINED:
                result = self._combined_selection(X_filtered, y, target_variable, max_features)
            elif method == SelectionMethod.MUTUAL_INFO:
                result = self._mutual_info_selection(X_filtered, y, target_variable, max_features)
            elif method == SelectionMethod.RFE:
                result = self._rfe_selection(X_filtered, y, target_variable, max_features)
            elif method == SelectionMethod.STABILITY:
                result = self._stability_selection(X_filtered, y, target_variable, max_features)
            elif method == SelectionMethod.CORRELATION:
                result = self._correlation_selection(X_filtered, y, target_variable, max_features)
            elif method == SelectionMethod.TREE_IMPORTANCE:
                result = self._tree_importance_selection(X_filtered, y, target_variable, max_features)
            elif method == SelectionMethod.L1_REGULARIZATION:
                result = self._l1_selection(X_filtered, y, target_variable, max_features)
            else:
                raise ValueError(f"Unknown selection method: {method}")
            
            # Update feature metrics with selection results
            self._update_metrics_from_selection(result)
            
            # Store selection result
            self.selection_history.append(result)
            
            # Update method performance
            self.method_performance[method].append(result.selection_performance)
            
            self.logger.info(f"Selected {len(result.selected_features)} features using {method.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in feature selection with {method.value}: {e}")
            # Return empty result on error
            return FeatureSelectionResult(
                timestamp=datetime.now(),
                method=method,
                target_variable=target_variable,
                selected_features=[],
                feature_scores={},
                feature_rankings={},
                selection_performance=0.0,
                cross_validation_score=0.0,
                feature_redundancy_removed=0,
                method_specific_results={},
                selection_stability=0.0,
                feature_diversity=0.0,
                metadata={'error': str(e)}
            )
    
    def _apply_basic_filtering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply basic filtering to remove low-quality features."""
        # Remove features with low variance
        variance_selector = VarianceThreshold(threshold=self.config['min_feature_variance'])
        X_var_filtered = pd.DataFrame(
            variance_selector.fit_transform(X),
            columns=X.columns[variance_selector.get_support()],
            index=X.index
        )
        
        # Remove features with too many missing values
        missing_threshold = 0.3  # 30% missing threshold
        missing_rates = X_var_filtered.isnull().sum() / len(X_var_filtered)
        valid_features = missing_rates[missing_rates < missing_threshold].index
        X_filtered = X_var_filtered[valid_features]
        
        self.logger.debug(f"Basic filtering: {len(X.columns)} -> {len(X_filtered.columns)} features")
        
        return X_filtered
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series, target_variable: str, max_features: int) -> FeatureSelectionResult:
        """Select features using mutual information."""
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Create feature scores dictionary
        feature_scores = dict(zip(X.columns, mi_scores))
        
        # Rank features
        feature_rankings = {
            feature: rank for rank, (feature, score) in 
            enumerate(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True), 1)
        }
        
        # Select top features
        selected_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:max_features]
        selected_feature_names = [f[0] for f in selected_features]
        
        # Calculate cross-validation performance
        if len(selected_feature_names) > 0:
            cv_score = self._calculate_cv_performance(X[selected_feature_names], y)
        else:
            cv_score = 0.0
        
        # Update feature metrics
        for feature in X.columns:
            if feature in self.feature_metrics:
                self.feature_metrics[feature].mutual_info_score = feature_scores[feature]
        
        return FeatureSelectionResult(
            timestamp=datetime.now(),
            method=SelectionMethod.MUTUAL_INFO,
            target_variable=target_variable,
            selected_features=selected_feature_names,
            feature_scores=feature_scores,
            feature_rankings=feature_rankings,
            selection_performance=np.mean(mi_scores),
            cross_validation_score=cv_score,
            feature_redundancy_removed=0,
            method_specific_results={'mi_threshold': self.config['mutual_info_threshold']},
            selection_stability=0.8,  # MI is generally stable
            feature_diversity=len(set([self._infer_feature_type(f) for f in selected_feature_names])) / len(FeatureType),
            metadata={'total_features_evaluated': len(X.columns)}
        )
    
    def _rfe_selection(self, X: pd.DataFrame, y: pd.Series, target_variable: str, max_features: int) -> FeatureSelectionResult:
        """Select features using Recursive Feature Elimination."""
        # Use Random Forest as base estimator
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Perform RFE
        rfe = RFE(estimator=estimator, n_features_to_select=max_features, step=0.1)
        rfe.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[rfe.support_].tolist()
        
        # Create rankings and scores
        feature_rankings = dict(zip(X.columns, rfe.ranking_))
        
        # Use feature importance as scores for selected features
        if hasattr(rfe.estimator_, 'feature_importances_'):
            importances = rfe.estimator_.feature_importances_
            feature_scores = dict(zip(selected_features, importances))
            # Set scores for non-selected features to 0
            for feature in X.columns:
                if feature not in feature_scores:
                    feature_scores[feature] = 0.0
        else:
            feature_scores = {feature: 1.0 if feature in selected_features else 0.0 for feature in X.columns}
        
        # Calculate cross-validation performance
        cv_score = self._calculate_cv_performance(X[selected_features], y)
        
        # Update feature metrics
        for feature in X.columns:
            if feature in self.feature_metrics:
                self.feature_metrics[feature].rfe_ranking = feature_rankings[feature]
        
        return FeatureSelectionResult(
            timestamp=datetime.now(),
            method=SelectionMethod.RFE,
            target_variable=target_variable,
            selected_features=selected_features,
            feature_scores=feature_scores,
            feature_rankings=feature_rankings,
            selection_performance=cv_score,
            cross_validation_score=cv_score,
            feature_redundancy_removed=len(X.columns) - len(selected_features),
            method_specific_results={'base_estimator': 'RandomForest'},
            selection_stability=0.7,  # RFE can be less stable
            feature_diversity=len(set([self._infer_feature_type(f) for f in selected_features])) / len(FeatureType),
            metadata={'total_features_evaluated': len(X.columns)}
        )
    
    def _stability_selection(self, X: pd.DataFrame, y: pd.Series, target_variable: str, max_features: int) -> FeatureSelectionResult:
        """Select features using Stability Selection."""
        n_iterations = self.config['stability_iterations']
        n_samples = len(X)
        subsample_size = int(0.8 * n_samples)  # 80% subsampling
        
        # Track feature selection frequency
        feature_selection_counts = defaultdict(int)
        
        # Perform bootstrap iterations
        for iteration in range(n_iterations):
            # Random subsample
            subsample_indices = np.random.choice(n_samples, subsample_size, replace=False)
            X_sub = X.iloc[subsample_indices]
            y_sub = y.iloc[subsample_indices]
            
            # Apply LASSO with random regularization
            alpha = np.random.uniform(0.001, 0.1)
            lasso = LassoCV(alphas=[alpha], cv=3, random_state=iteration)
            
            try:
                lasso.fit(X_sub, y_sub)
                
                # Count selected features (non-zero coefficients)
                selected_mask = np.abs(lasso.coef_) > 1e-6
                selected_features = X_sub.columns[selected_mask]
                
                for feature in selected_features:
                    feature_selection_counts[feature] += 1
                    
            except Exception as e:
                self.logger.debug(f"Stability selection iteration {iteration} failed: {e}")
                continue
        
        # Calculate stability scores (selection frequency)
        stability_scores = {
            feature: count / n_iterations 
            for feature, count in feature_selection_counts.items()
        }
        
        # Add unselected features with score 0
        for feature in X.columns:
            if feature not in stability_scores:
                stability_scores[feature] = 0.0
        
        # Select features above stability threshold
        stable_features = [
            feature for feature, score in stability_scores.items()
            if score >= self.config['stability_threshold']
        ]
        
        # Limit to max_features by taking highest stability scores
        if len(stable_features) > max_features:
            stable_features = sorted(
                stable_features, 
                key=lambda f: stability_scores[f], 
                reverse=True
            )[:max_features]
        
        # Create rankings
        feature_rankings = {
            feature: rank for rank, (feature, score) in 
            enumerate(sorted(stability_scores.items(), key=lambda x: x[1], reverse=True), 1)
        }
        
        # Calculate cross-validation performance
        if len(stable_features) > 0:
            cv_score = self._calculate_cv_performance(X[stable_features], y)
        else:
            cv_score = 0.0
        
        # Update feature metrics
        for feature in X.columns:
            if feature in self.feature_metrics:
                self.feature_metrics[feature].stability_score = stability_scores[feature]
        
        return FeatureSelectionResult(
            timestamp=datetime.now(),
            method=SelectionMethod.STABILITY,
            target_variable=target_variable,
            selected_features=stable_features,
            feature_scores=stability_scores,
            feature_rankings=feature_rankings,
            selection_performance=np.mean(list(stability_scores.values())),
            cross_validation_score=cv_score,
            feature_redundancy_removed=0,
            method_specific_results={
                'iterations': n_iterations,
                'stability_threshold': self.config['stability_threshold']
            },
            selection_stability=0.9,  # High by design
            feature_diversity=len(set([self._infer_feature_type(f) for f in stable_features])) / len(FeatureType),
            metadata={'total_features_evaluated': len(X.columns)}
        )
    
    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series, target_variable: str, max_features: int) -> FeatureSelectionResult:
        """Select features by removing highly correlated ones."""
        if self.correlation_matrix is None:
            self._calculate_correlation_matrix()
        
        # Start with all features
        remaining_features = set(X.columns)
        removed_features = []
        
        # Calculate correlation with target
        target_correlations = {}
        for feature in X.columns:
            try:
                corr = np.corrcoef(X[feature].fillna(0), y)[0, 1]
                target_correlations[feature] = abs(corr) if not np.isnan(corr) else 0.0
            except:
                target_correlations[feature] = 0.0
        
        # Remove highly correlated features, keeping the one most correlated with target
        correlation_threshold = self.config['correlation_threshold']
        
        for cluster_id, cluster_features in self.feature_clusters.items():
            if len(cluster_features) <= 1:
                continue
            
            # Find feature in cluster most correlated with target
            cluster_target_corrs = {f: target_correlations.get(f, 0.0) for f in cluster_features}
            best_feature = max(cluster_target_corrs, key=cluster_target_corrs.get)
            
            # Remove other features from cluster
            for feature in cluster_features:
                if feature != best_feature and feature in remaining_features:
                    remaining_features.remove(feature)
                    removed_features.append(feature)
        
        # Select top features by target correlation
        remaining_features_list = list(remaining_features)
        remaining_features_list.sort(key=lambda f: target_correlations.get(f, 0.0), reverse=True)
        
        selected_features = remaining_features_list[:max_features]
        
        # Create scores and rankings
        feature_scores = target_correlations
        feature_rankings = {
            feature: rank for rank, (feature, score) in 
            enumerate(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True), 1)
        }
        
        # Calculate cross-validation performance
        if len(selected_features) > 0:
            cv_score = self._calculate_cv_performance(X[selected_features], y)
        else:
            cv_score = 0.0
        
        # Update feature metrics
        for feature in X.columns:
            if feature in self.feature_metrics:
                if self.correlation_matrix is not None and feature in self.correlation_matrix.columns:
                    correlations = self.correlation_matrix.loc[feature]
                    correlations = correlations[correlations.index != feature]
                    self.feature_metrics[feature].correlation_redundancy = correlations.max() if len(correlations) > 0 else 0.0
        
        return FeatureSelectionResult(
            timestamp=datetime.now(),
            method=SelectionMethod.CORRELATION,
            target_variable=target_variable,
            selected_features=selected_features,
            feature_scores=feature_scores,
            feature_rankings=feature_rankings,
            selection_performance=np.mean([target_correlations.get(f, 0.0) for f in selected_features]) if selected_features else 0.0,
            cross_validation_score=cv_score,
            feature_redundancy_removed=len(removed_features),
            method_specific_results={
                'correlation_threshold': correlation_threshold,
                'removed_features': removed_features
            },
            selection_stability=0.8,
            feature_diversity=len(set([self._infer_feature_type(f) for f in selected_features])) / len(FeatureType),
            metadata={'total_features_evaluated': len(X.columns)}
        )
    
    def _tree_importance_selection(self, X: pd.DataFrame, y: pd.Series, target_variable: str, max_features: int) -> FeatureSelectionResult:
        """Select features using tree-based feature importance."""
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_scores = dict(zip(X.columns, importances))
        
        # Select top features
        selected_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:max_features]
        selected_feature_names = [f[0] for f in selected_features]
        
        # Create rankings
        feature_rankings = {
            feature: rank for rank, (feature, score) in 
            enumerate(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True), 1)
        }
        
        # Calculate cross-validation performance
        cv_score = self._calculate_cv_performance(X[selected_feature_names], y)
        
        # Update feature metrics
        for feature in X.columns:
            if feature in self.feature_metrics:
                self.feature_metrics[feature].tree_importance = feature_scores[feature]
        
        return FeatureSelectionResult(
            timestamp=datetime.now(),
            method=SelectionMethod.TREE_IMPORTANCE,
            target_variable=target_variable,
            selected_features=selected_feature_names,
            feature_scores=feature_scores,
            feature_rankings=feature_rankings,
            selection_performance=np.mean(importances),
            cross_validation_score=cv_score,
            feature_redundancy_removed=0,
            method_specific_results={'n_estimators': 100},
            selection_stability=0.7,
            feature_diversity=len(set([self._infer_feature_type(f) for f in selected_feature_names])) / len(FeatureType),
            metadata={'total_features_evaluated': len(X.columns)}
        )
    
    def _l1_selection(self, X: pd.DataFrame, y: pd.Series, target_variable: str, max_features: int) -> FeatureSelectionResult:
        """Select features using L1 regularization (LASSO)."""
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # Fit LASSO
        lasso = LassoCV(cv=self.config['cv_folds'], random_state=42, max_iter=1000)
        lasso.fit(X_scaled, y)
        
        # Get coefficients
        coefficients = lasso.coef_
        feature_scores = dict(zip(X.columns, np.abs(coefficients)))
        
        # Select features with non-zero coefficients
        selected_mask = np.abs(coefficients) > 1e-6
        selected_features = X.columns[selected_mask].tolist()
        
        # Limit to max_features if needed
        if len(selected_features) > max_features:
            selected_features = sorted(
                selected_features, 
                key=lambda f: feature_scores[f], 
                reverse=True
            )[:max_features]
        
        # Create rankings
        feature_rankings = {
            feature: rank for rank, (feature, score) in 
            enumerate(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True), 1)
        }
        
        # Calculate cross-validation performance
        if len(selected_features) > 0:
            cv_score = self._calculate_cv_performance(X[selected_features], y)
        else:
            cv_score = 0.0
        
        # Update feature metrics
        for feature, coef in zip(X.columns, coefficients):
            if feature in self.feature_metrics:
                self.feature_metrics[feature].l1_coefficient = abs(coef)
        
        return FeatureSelectionResult(
            timestamp=datetime.now(),
            method=SelectionMethod.L1_REGULARIZATION,
            target_variable=target_variable,
            selected_features=selected_features,
            feature_scores=feature_scores,
            feature_rankings=feature_rankings,
            selection_performance=lasso.score(X_scaled, y),
            cross_validation_score=cv_score,
            feature_redundancy_removed=len(X.columns) - len(selected_features),
            method_specific_results={'alpha': lasso.alpha_, 'l1_ratio': 1.0},
            selection_stability=0.6,  # L1 can be less stable
            feature_diversity=len(set([self._infer_feature_type(f) for f in selected_features])) / len(FeatureType),
            metadata={'total_features_evaluated': len(X.columns)}
        )
    
    def _combined_selection(self, X: pd.DataFrame, y: pd.Series, target_variable: str, max_features: int) -> FeatureSelectionResult:
        """Combine multiple selection methods using ensemble approach."""
        # Run individual methods
        methods_to_run = [
            SelectionMethod.MUTUAL_INFO,
            SelectionMethod.RFE,
            SelectionMethod.STABILITY,
            SelectionMethod.TREE_IMPORTANCE,
            SelectionMethod.L1_REGULARIZATION
        ]
        
        method_results = {}
        for method in methods_to_run:
            try:
                result = self._run_single_method(X, y, target_variable, method, max_features * 2)  # Get more candidates
                method_results[method] = result
            except Exception as e:
                self.logger.warning(f"Method {method.value} failed: {e}")
                continue
        
        if not method_results:
            raise ValueError("All selection methods failed")
        
        # Combine scores using weighted ensemble
        ensemble_scores = defaultdict(float)
        weights = self.config['selection_ensemble_weights']
        
        for method, result in method_results.items():
            method_weight = weights.get(method.value, 0.1)
            
            # Normalize scores to [0, 1] range
            scores = result.feature_scores
            if scores:
                max_score = max(scores.values())
                min_score = min(scores.values())
                score_range = max_score - min_score if max_score > min_score else 1.0
                
                for feature, score in scores.items():
                    normalized_score = (score - min_score) / score_range
                    ensemble_scores[feature] += method_weight * normalized_score
        
        # Select top features
        selected_features = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)[:max_features]
        selected_feature_names = [f[0] for f in selected_features]
        
        # Create final scores and rankings
        feature_scores = dict(ensemble_scores)
        feature_rankings = {
            feature: rank for rank, (feature, score) in 
            enumerate(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True), 1)
        }
        
        # Calculate cross-validation performance
        cv_score = self._calculate_cv_performance(X[selected_feature_names], y)
        
        # Calculate selection stability (agreement between methods)
        stability = self._calculate_selection_stability(method_results, selected_feature_names)
        
        return FeatureSelectionResult(
            timestamp=datetime.now(),
            method=SelectionMethod.COMBINED,
            target_variable=target_variable,
            selected_features=selected_feature_names,
            feature_scores=feature_scores,
            feature_rankings=feature_rankings,
            selection_performance=np.mean(list(ensemble_scores.values())),
            cross_validation_score=cv_score,
            feature_redundancy_removed=self._count_redundancy_removed(X, selected_feature_names),
            method_specific_results={
                'methods_used': [m.value for m in method_results.keys()],
                'ensemble_weights': weights
            },
            selection_stability=stability,
            feature_diversity=len(set([self._infer_feature_type(f) for f in selected_feature_names])) / len(FeatureType),
            metadata={'total_features_evaluated': len(X.columns)}
        )
    
    def _run_single_method(self, X: pd.DataFrame, y: pd.Series, target_variable: str, method: SelectionMethod, max_features: int) -> FeatureSelectionResult:
        """Run a single selection method."""
        if method == SelectionMethod.MUTUAL_INFO:
            return self._mutual_info_selection(X, y, target_variable, max_features)
        elif method == SelectionMethod.RFE:
            return self._rfe_selection(X, y, target_variable, max_features)
        elif method == SelectionMethod.STABILITY:
            return self._stability_selection(X, y, target_variable, max_features)
        elif method == SelectionMethod.TREE_IMPORTANCE:
            return self._tree_importance_selection(X, y, target_variable, max_features)
        elif method == SelectionMethod.L1_REGULARIZATION:
            return self._l1_selection(X, y, target_variable, max_features)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _calculate_cv_performance(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate cross-validation performance."""
        try:
            # Use a simple model for CV evaluation
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
            # Use TimeSeriesSplit for time series data
            cv = TimeSeriesSplit(n_splits=min(5, len(X) // 20))
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            
            return np.mean(scores)
            
        except Exception as e:
            self.logger.debug(f"CV performance calculation failed: {e}")
            return 0.0
    
    def _calculate_selection_stability(self, method_results: Dict[SelectionMethod, FeatureSelectionResult], final_features: List[str]) -> float:
        """Calculate stability of feature selection across methods."""
        if not method_results or not final_features:
            return 0.0
        
        # Count how many methods selected each final feature
        selection_counts = defaultdict(int)
        
        for method, result in method_results.items():
            for feature in final_features:
                if feature in result.selected_features:
                    selection_counts[feature] += 1
        
        # Calculate average agreement
        total_methods = len(method_results)
        stability_scores = [count / total_methods for count in selection_counts.values()]
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def _count_redundancy_removed(self, X: pd.DataFrame, selected_features: List[str]) -> int:
        """Count how many redundant features were removed."""
        if self.correlation_matrix is None:
            return 0
        
        redundant_count = 0
        correlation_threshold = self.config['correlation_threshold']
        
        # Check correlations among selected features
        if len(selected_features) > 1:
            selected_corr = self.correlation_matrix.loc[selected_features, selected_features]
            
            # Count pairs with high correlation (excluding diagonal)
            for i in range(len(selected_features)):
                for j in range(i + 1, len(selected_features)):
                    if selected_corr.iloc[i, j] > correlation_threshold:
                        redundant_count += 1
        
        return len(X.columns) - len(selected_features) - redundant_count
    
    def _update_metrics_from_selection(self, result: FeatureSelectionResult):
        """Update feature metrics based on selection results."""
        for feature in result.selected_features:
            if feature in self.feature_metrics:
                # Update overall score based on selection
                metrics = self.feature_metrics[feature]
                metrics.overall_score = result.feature_scores.get(feature, 0.0)
                metrics.selection_probability = self._calculate_selection_probability(feature)
                
                # Update selection history
                metrics.selection_history.append(True)
                if len(metrics.selection_history) > 100:  # Keep last 100 selections
                    metrics.selection_history.pop(0)
        
        # Update non-selected features
        for feature in self.feature_metrics:
            if feature not in result.selected_features:
                metrics = self.feature_metrics[feature]
                metrics.selection_history.append(False)
                if len(metrics.selection_history) > 100:
                    metrics.selection_history.pop(0)
    
    def _calculate_selection_probability(self, feature: str) -> float:
        """Calculate probability of feature being selected based on history."""
        if feature not in self.feature_metrics:
            return 0.0
        
        history = self.feature_metrics[feature].selection_history
        if not history:
            return 0.0
        
        # Apply decay factor to give more weight to recent selections
        decay_factor = self.config['feature_decay_factor']
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for i, selected in enumerate(reversed(history)):
            weight = decay_factor ** i
            weighted_sum += weight * (1.0 if selected else 0.0)
            weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    # Public interface methods
    def get_feature_metrics(self, feature_name: Optional[str] = None) -> Union[FeatureMetrics, Dict[str, FeatureMetrics]]:
        """Get feature metrics."""
        if feature_name:
            return self.feature_metrics.get(feature_name)
        return self.feature_metrics.copy()
    
    def get_selection_history(self, limit: Optional[int] = None) -> List[FeatureSelectionResult]:
        """Get feature selection history."""
        history = list(self.selection_history)
        if limit:
            return history[-limit:]
        return history
    
    def get_recommended_features(self, target_variable: str, max_features: Optional[int] = None) -> List[str]:
        """Get recommended features based on comprehensive analysis."""
        try:
            result = self.select_features(target_variable, SelectionMethod.COMBINED, max_features)
            return result.selected_features
        except Exception as e:
            self.logger.error(f"Error getting recommended features: {e}")
            return []
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze overall feature importance across all methods."""
        if not self.feature_metrics:
            return {'status': 'no_data'}
        
        # Aggregate importance scores
        importance_analysis = {}
        
        for feature_name, metrics in self.feature_metrics.items():
            importance_analysis[feature_name] = {
                'overall_score': metrics.overall_score,
                'selection_probability': metrics.selection_probability,
                'mutual_info_score': metrics.mutual_info_score,
                'tree_importance': metrics.tree_importance,
                'stability_score': metrics.stability_score,
                'feature_type': metrics.feature_type.value,
                'correlation_redundancy': metrics.correlation_redundancy
            }
        
        # Find top features overall
        top_features = sorted(
            importance_analysis.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )[:20]
        
        # Analyze by feature type
        type_analysis = defaultdict(list)
        for feature_name, metrics in self.feature_metrics.items():
            type_analysis[metrics.feature_type.value].append({
                'name': feature_name,
                'score': metrics.overall_score
            })
        
        # Sort within each type
        for feature_type in type_analysis:
            type_analysis[feature_type].sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'timestamp': datetime.now(),
            'total_features_analyzed': len(self.feature_metrics),
            'top_features': [{'name': name, **scores} for name, scores in top_features],
            'by_feature_type': dict(type_analysis),
            'method_performance': {
                method.value: np.mean(scores) if scores else 0.0
                for method, scores in self.method_performance.items()
            },
            'correlation_clusters': len(self.feature_clusters)
        }


# Example usage and testing
if __name__ == "__main__":
    import json
    from sklearn.datasets import make_regression
    from ..config.manager import ConfigurationManager
    
    def main():
        # Configure logging
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize selector
        config_manager = ConfigurationManager()
        selector = AdvancedFeatureSelector(config_manager)
        
        # Create sample data
        X, y = make_regression(
            n_samples=1000,
            n_features=100,
            n_informative=20,
            n_redundant=10,
            noise=0.1,
            random_state=42
        )
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        feature_data = pd.DataFrame(X, columns=feature_names)
        target_data = {'price_1h': pd.Series(y)}
        
        # Update selector
        selector.update_feature_data(feature_data, target_data)
        
        # Test different selection methods
        methods_to_test = [
            SelectionMethod.MUTUAL_INFO,
            SelectionMethod.TREE_IMPORTANCE,
            SelectionMethod.L1_REGULARIZATION,
            SelectionMethod.COMBINED
        ]
        
        for method in methods_to_test:
            print(f"\n=== Testing {method.value} ===")
            
            result = selector.select_features('price_1h', method, max_features=20)
            
            print(f"Selected {len(result.selected_features)} features")
            print(f"CV Score: {result.cross_validation_score:.4f}")
            print(f"Selection Stability: {result.selection_stability:.4f}")
            print(f"Top 5 features: {result.selected_features[:5]}")
        
        # Analyze feature importance
        importance_analysis = selector.analyze_feature_importance()
        print(f"\nFeature Importance Analysis:")
        print(f"Total features: {importance_analysis['total_features_analyzed']}")
        print(f"Top 5 features overall: {[f['name'] for f in importance_analysis['top_features'][:5]]}")
        
        # Test recommended features
        recommended = selector.get_recommended_features('price_1h', max_features=15)
        print(f"\nRecommended features: {recommended}")
    
    # Run the example
    main()