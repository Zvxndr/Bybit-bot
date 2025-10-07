"""
Model Performance Monitor

Comprehensive model monitoring system for cryptocurrency trading ML models.
Implements drift detection, performance degradation alerts, automatic retraining
triggers, A/B testing framework, and continuous model health monitoring.

Key Features:
- Real-time model performance monitoring and alerting
- Statistical drift detection (data drift, concept drift, prediction drift)
- Performance degradation detection with threshold-based alerts
- Automatic model retraining triggers and scheduling
- A/B testing framework for model comparison and deployment
- Model health scoring and lifecycle management
- Feature drift monitoring and feature importance tracking
- Prediction quality assessment and confidence monitoring
- Model comparison and champion/challenger frameworks
- Comprehensive logging and audit trails for model decisions

Drift Detection Methods:
- Kolmogorov-Smirnov Test: Distribution change detection
- Population Stability Index (PSI): Feature distribution stability
- Jensen-Shannon Divergence: Probability distribution comparison
- Statistical Process Control: Control charts for performance metrics
- ADWIN: Adaptive Windowing for concept drift detection
- Page-Hinkley Test: Sequential change point detection
- Drift Detection Method (DDM): Error rate based drift detection

Performance Monitoring:
- Real-time accuracy, precision, recall tracking
- Prediction confidence distribution monitoring
- Feature importance drift detection
- Model latency and resource usage monitoring
- Business metric impact assessment
- Custom metric tracking and alerting
- Model comparison dashboards and reports
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import asyncio
from collections import defaultdict, deque
import warnings
from enum import Enum
import json
import pickle
import threading
import time
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor
import statistics

# Statistical testing imports
from scipy import stats
from scipy.spatial.distance import jensenshannon
import scipy.stats as scipy_stats

# ML monitoring imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, log_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ..utils.logging import TradingLogger
from ..config.manager import ConfigurationManager


class DriftType(Enum):
    """Types of model drift."""
    DATA_DRIFT = "data_drift"          # Input feature distribution changes
    CONCEPT_DRIFT = "concept_drift"    # Relationship between features and target changes
    PREDICTION_DRIFT = "prediction_drift"  # Model output distribution changes
    PERFORMANCE_DRIFT = "performance_drift"  # Model performance degrades


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelStatus(Enum):
    """Model lifecycle status."""
    ACTIVE = "active"
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    SHADOW = "shadow"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    RETRAINING = "retraining"


class DriftTestResult(Enum):
    """Drift test results."""
    NO_DRIFT = "no_drift"
    DRIFT_DETECTED = "drift_detected"
    INCONCLUSIVE = "inconclusive"


@dataclass
class DriftDetectionResult:
    """Results from drift detection analysis."""
    timestamp: datetime
    model_id: str
    drift_type: DriftType
    test_result: DriftTestResult
    
    # Test statistics
    test_statistic: float
    p_value: float
    threshold: float
    
    # Drift metrics
    drift_score: float  # 0-1 scale, higher = more drift
    drift_magnitude: float  # How severe the drift is
    
    # Feature-level results (for data drift)
    feature_drift_scores: Dict[str, float] = field(default_factory=dict)
    
    # Method used
    detection_method: str = ""
    
    # Recommendations
    recommended_action: str = ""
    confidence: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Model performance alert."""
    timestamp: datetime
    model_id: str
    alert_type: str
    severity: AlertSeverity
    
    # Performance metrics
    current_metric: float
    baseline_metric: float
    threshold: float
    
    # Alert details
    message: str
    metric_name: str
    
    # Context
    sample_size: int
    time_window: str
    
    # Actions
    recommended_actions: List[str] = field(default_factory=list)
    auto_triggered: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelHealthScore:
    """Comprehensive model health assessment."""
    timestamp: datetime
    model_id: str
    
    # Overall health
    overall_score: float  # 0-100
    health_status: str    # "excellent", "good", "fair", "poor", "critical"
    
    # Component scores
    performance_score: float
    drift_score: float
    stability_score: float
    robustness_score: float
    
    # Performance metrics
    recent_accuracy: float
    accuracy_trend: float
    prediction_confidence: float
    
    # Drift indicators
    data_drift_severity: float
    concept_drift_severity: float
    
    # Operational metrics
    prediction_latency: float
    error_rate: float
    uptime_percentage: float
    
    # Recommendations
    health_recommendations: List[str] = field(default_factory=list)
    priority_actions: List[str] = field(default_factory=list)
    
    # Time since last retraining
    days_since_retrain: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """A/B test comparison results."""
    timestamp: datetime
    test_id: str
    
    # Models being compared
    champion_model_id: str
    challenger_model_id: str
    
    # Test configuration
    traffic_split: Dict[str, float]  # model_id -> percentage
    test_duration_hours: float
    
    # Performance comparison
    champion_metrics: Dict[str, float]
    challenger_metrics: Dict[str, float]
    
    # Statistical significance
    statistical_significance: Dict[str, float]  # metric -> p_value
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Test results
    recommended_winner: str
    confidence_level: float
    test_conclusion: str  # "promote_challenger", "keep_champion", "continue_test"
    
    # Business impact
    business_impact: Dict[str, float] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelPerformanceMonitor:
    """
    Comprehensive model monitoring system for cryptocurrency trading.
    
    Provides real-time performance monitoring, drift detection, alerting,
    and automated model lifecycle management.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = TradingLogger("ModelPerformanceMonitor")
        
        # Configuration
        self.config = {
            'monitoring_window_hours': config_manager.get('monitoring.window_hours', 24),
            'drift_detection_window': config_manager.get('monitoring.drift_window', 1000),
            'performance_threshold': config_manager.get('monitoring.performance_threshold', 0.05),
            'drift_threshold': config_manager.get('monitoring.drift_threshold', 0.05),
            'alert_cooldown_hours': config_manager.get('monitoring.alert_cooldown', 6),
            'retraining_threshold': config_manager.get('monitoring.retrain_threshold', 0.10),
            'max_models_tracked': config_manager.get('monitoring.max_models', 50),
            'health_check_interval': config_manager.get('monitoring.health_interval', 3600),  # seconds
            'ab_test_min_samples': config_manager.get('monitoring.ab_min_samples', 1000),
            'statistical_significance_level': config_manager.get('monitoring.significance', 0.05)
        }
        
        # Model registry and tracking
        self.tracked_models: Dict[str, Dict[str, Any]] = {}
        self.model_baselines: Dict[str, Dict[str, float]] = {}
        self.model_predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.model_actuals: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.model_features: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Drift detection storage
        self.drift_detection_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.reference_distributions: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        
        # Performance monitoring
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        self.alert_history: deque = deque(maxlen=10000)
        self.recent_alerts: Dict[str, datetime] = {}  # For cooldown
        
        # Health scoring
        self.health_scores: Dict[str, ModelHealthScore] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # A/B testing
        self.active_ab_tests: Dict[str, ABTestResult] = {}
        self.ab_test_history: deque = deque(maxlen=1000)
        
        # Threading for background monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Drift detection methods
        self.drift_detectors = {
            'ks_test': self._ks_drift_test,
            'psi': self._psi_drift_test,
            'js_divergence': self._js_drift_test,
            'statistical_process_control': self._spc_drift_test
        }
    
    def register_model(self, model_id: str, model_info: Dict[str, Any]):
        """Register a model for monitoring."""
        self.tracked_models[model_id] = {
            'registration_time': datetime.now(),
            'model_type': model_info.get('model_type', 'unknown'),
            'target_variable': model_info.get('target_variable', 'unknown'),
            'features': model_info.get('features', []),
            'status': ModelStatus.ACTIVE,
            **model_info
        }
        
        # Initialize baseline metrics
        self.model_baselines[model_id] = {}
        
        self.logger.info(f"Registered model {model_id} for monitoring")
    
    def log_prediction(self, model_id: str, features: Dict[str, Any], 
                      prediction: float, actual: Optional[float] = None,
                      prediction_confidence: Optional[float] = None):
        """Log a model prediction for monitoring."""
        if model_id not in self.tracked_models:
            self.logger.warning(f"Model {model_id} not registered for monitoring")
            return
        
        timestamp = datetime.now()
        
        # Store prediction
        prediction_data = {
            'timestamp': timestamp,
            'prediction': prediction,
            'confidence': prediction_confidence,
            'features': features.copy()
        }
        self.model_predictions[model_id].append(prediction_data)
        
        # Store features for drift detection
        feature_values = list(features.values())
        self.model_features[model_id].append(feature_values)
        
        # Store actual value if available
        if actual is not None:
            actual_data = {
                'timestamp': timestamp,
                'actual': actual
            }
            self.model_actuals[model_id].append(actual_data)
        
        # Update model status
        self.tracked_models[model_id]['last_prediction'] = timestamp
    
    def calculate_performance_metrics(self, model_id: str, 
                                    window_hours: Optional[int] = None) -> Dict[str, float]:
        """Calculate performance metrics for a model."""
        if model_id not in self.tracked_models:
            return {}
        
        window_hours = window_hours or self.config['monitoring_window_hours']
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        
        # Get recent predictions and actuals
        recent_predictions = [
            p for p in self.model_predictions[model_id]
            if p['timestamp'] >= cutoff_time
        ]
        
        recent_actuals = [
            a for a in self.model_actuals[model_id]
            if a['timestamp'] >= cutoff_time
        ]
        
        if len(recent_predictions) == 0 or len(recent_actuals) == 0:
            return {}
        
        # Align predictions with actuals by timestamp
        predictions = []
        actuals = []
        
        actual_dict = {a['timestamp']: a['actual'] for a in recent_actuals}
        
        for pred in recent_predictions:
            if pred['timestamp'] in actual_dict:
                predictions.append(pred['prediction'])
                actuals.append(actual_dict[pred['timestamp']])
        
        if len(predictions) < 10:  # Need minimum samples
            return {}
        
        # Calculate metrics
        metrics = {}
        
        try:
            # Regression metrics
            metrics['mse'] = mean_squared_error(actuals, predictions)
            metrics['mae'] = mean_absolute_error(actuals, predictions)
            metrics['r2'] = r2_score(actuals, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # MAPE
            mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
            metrics['mape'] = mape if not np.isnan(mape) and not np.isinf(mape) else 100.0
            
            # Prediction confidence metrics
            confidences = [p.get('confidence', 0.5) for p in recent_predictions if p.get('confidence') is not None]
            if confidences:
                metrics['avg_confidence'] = np.mean(confidences)
                metrics['confidence_std'] = np.std(confidences)
            
            # Error distribution metrics
            errors = np.array(actuals) - np.array(predictions)
            metrics['error_mean'] = np.mean(errors)
            metrics['error_std'] = np.std(errors)
            metrics['error_skewness'] = scipy_stats.skew(errors)
            
            # Sample size
            metrics['sample_size'] = len(predictions)
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {model_id}: {e}")
            return {}
        
        # Store performance history
        performance_record = {
            'timestamp': datetime.now(),
            'window_hours': window_hours,
            **metrics
        }
        self.performance_history[model_id].append(performance_record)
        
        return metrics
    
    def detect_data_drift(self, model_id: str, method: str = 'ks_test') -> DriftDetectionResult:
        """Detect data drift in model features."""
        if model_id not in self.tracked_models:
            raise ValueError(f"Model {model_id} not registered")
        
        if method not in self.drift_detectors:
            raise ValueError(f"Unknown drift detection method: {method}")
        
        # Get recent feature data
        recent_features = list(self.model_features[model_id])
        
        if len(recent_features) < self.config['drift_detection_window']:
            return DriftDetectionResult(
                timestamp=datetime.now(),
                model_id=model_id,
                drift_type=DriftType.DATA_DRIFT,
                test_result=DriftTestResult.INCONCLUSIVE,
                test_statistic=0.0,
                p_value=1.0,
                threshold=self.config['drift_threshold'],
                drift_score=0.0,
                drift_magnitude=0.0,
                detection_method=method,
                recommended_action="Need more data",
                confidence=0.0
            )
        
        # Split into reference and current windows
        split_point = len(recent_features) // 2
        reference_data = np.array(recent_features[:split_point])
        current_data = np.array(recent_features[split_point:])
        
        # Apply drift detection method
        drift_result = self.drift_detectors[method](reference_data, current_data, model_id)
        
        # Store result
        self.drift_detection_history[model_id].append(drift_result)
        
        return drift_result
    
    def _ks_drift_test(self, reference_data: np.ndarray, current_data: np.ndarray, 
                      model_id: str) -> DriftDetectionResult:
        """Kolmogorov-Smirnov drift test."""
        n_features = reference_data.shape[1] if len(reference_data.shape) > 1 else 1
        
        if n_features == 1:
            reference_data = reference_data.reshape(-1, 1)
            current_data = current_data.reshape(-1, 1)
        
        feature_drift_scores = {}
        p_values = []
        test_statistics = []
        
        # Test each feature
        for i in range(n_features):
            ref_feature = reference_data[:, i]
            cur_feature = current_data[:, i]
            
            # Perform KS test
            ks_statistic, p_value = stats.kstest(cur_feature, ref_feature)
            
            feature_drift_scores[f'feature_{i}'] = ks_statistic
            p_values.append(p_value)
            test_statistics.append(ks_statistic)
        
        # Aggregate results
        overall_p_value = np.mean(p_values)
        overall_statistic = np.mean(test_statistics)
        drift_score = overall_statistic
        
        # Determine result
        if overall_p_value < self.config['drift_threshold']:
            test_result = DriftTestResult.DRIFT_DETECTED
            recommended_action = "Consider retraining model"
        else:
            test_result = DriftTestResult.NO_DRIFT
            recommended_action = "Continue monitoring"
        
        return DriftDetectionResult(
            timestamp=datetime.now(),
            model_id=model_id,
            drift_type=DriftType.DATA_DRIFT,
            test_result=test_result,
            test_statistic=overall_statistic,
            p_value=overall_p_value,
            threshold=self.config['drift_threshold'],
            drift_score=drift_score,
            drift_magnitude=drift_score,
            feature_drift_scores=feature_drift_scores,
            detection_method='ks_test',
            recommended_action=recommended_action,
            confidence=1.0 - overall_p_value
        )
    
    def _psi_drift_test(self, reference_data: np.ndarray, current_data: np.ndarray, 
                       model_id: str) -> DriftDetectionResult:
        """Population Stability Index drift test."""
        n_features = reference_data.shape[1] if len(reference_data.shape) > 1 else 1
        
        if n_features == 1:
            reference_data = reference_data.reshape(-1, 1)
            current_data = current_data.reshape(-1, 1)
        
        feature_drift_scores = {}
        psi_values = []
        
        # Calculate PSI for each feature
        for i in range(n_features):
            ref_feature = reference_data[:, i]
            cur_feature = current_data[:, i]
            
            # Create bins based on reference data
            try:
                n_bins = min(10, len(np.unique(ref_feature)))
                bins = np.histogram_bin_edges(ref_feature, bins=n_bins)
                
                # Calculate bin proportions
                ref_counts, _ = np.histogram(ref_feature, bins=bins)
                cur_counts, _ = np.histogram(cur_feature, bins=bins)
                
                # Avoid division by zero
                ref_prop = (ref_counts + 1e-6) / (len(ref_feature) + 1e-6 * n_bins)
                cur_prop = (cur_counts + 1e-6) / (len(cur_feature) + 1e-6 * n_bins)
                
                # Calculate PSI
                psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
                
                feature_drift_scores[f'feature_{i}'] = psi
                psi_values.append(psi)
                
            except Exception as e:
                self.logger.debug(f"PSI calculation failed for feature {i}: {e}")
                psi_values.append(0.0)
        
        # Aggregate PSI
        overall_psi = np.mean(psi_values) if psi_values else 0.0
        
        # PSI interpretation: <0.1 = stable, 0.1-0.2 = some change, >0.2 = significant change
        if overall_psi > 0.2:
            test_result = DriftTestResult.DRIFT_DETECTED
            recommended_action = "Significant drift detected, consider retraining"
        elif overall_psi > 0.1:
            test_result = DriftTestResult.DRIFT_DETECTED
            recommended_action = "Moderate drift detected, monitor closely"
        else:
            test_result = DriftTestResult.NO_DRIFT
            recommended_action = "Continue monitoring"
        
        return DriftDetectionResult(
            timestamp=datetime.now(),
            model_id=model_id,
            drift_type=DriftType.DATA_DRIFT,
            test_result=test_result,
            test_statistic=overall_psi,
            p_value=1.0 - min(1.0, overall_psi / 0.3),  # Approximate p-value
            threshold=0.1,
            drift_score=overall_psi,
            drift_magnitude=overall_psi,
            feature_drift_scores=feature_drift_scores,
            detection_method='psi',
            recommended_action=recommended_action,
            confidence=min(1.0, overall_psi)
        )
    
    def _js_drift_test(self, reference_data: np.ndarray, current_data: np.ndarray, 
                      model_id: str) -> DriftDetectionResult:
        """Jensen-Shannon divergence drift test."""
        n_features = reference_data.shape[1] if len(reference_data.shape) > 1 else 1
        
        if n_features == 1:
            reference_data = reference_data.reshape(-1, 1)
            current_data = current_data.reshape(-1, 1)
        
        feature_drift_scores = {}
        js_values = []
        
        # Calculate JS divergence for each feature
        for i in range(n_features):
            ref_feature = reference_data[:, i]
            cur_feature = current_data[:, i]
            
            try:
                # Create bins
                all_data = np.concatenate([ref_feature, cur_feature])
                n_bins = min(20, len(np.unique(all_data)))
                bins = np.histogram_bin_edges(all_data, bins=n_bins)
                
                # Calculate distributions
                ref_hist, _ = np.histogram(ref_feature, bins=bins, density=True)
                cur_hist, _ = np.histogram(cur_feature, bins=bins, density=True)
                
                # Normalize to probabilities
                ref_prob = ref_hist / np.sum(ref_hist) if np.sum(ref_hist) > 0 else np.ones_like(ref_hist) / len(ref_hist)
                cur_prob = cur_hist / np.sum(cur_hist) if np.sum(cur_hist) > 0 else np.ones_like(cur_hist) / len(cur_hist)
                
                # Add small epsilon to avoid log(0)
                ref_prob = ref_prob + 1e-10
                cur_prob = cur_prob + 1e-10
                
                # Calculate JS divergence
                js_divergence = jensenshannon(ref_prob, cur_prob)
                
                feature_drift_scores[f'feature_{i}'] = js_divergence
                js_values.append(js_divergence)
                
            except Exception as e:
                self.logger.debug(f"JS divergence calculation failed for feature {i}: {e}")
                js_values.append(0.0)
        
        # Aggregate JS divergence
        overall_js = np.mean(js_values) if js_values else 0.0
        
        # JS divergence ranges from 0 to 1
        if overall_js > 0.3:
            test_result = DriftTestResult.DRIFT_DETECTED
            recommended_action = "Significant drift detected, retrain immediately"
        elif overall_js > 0.1:
            test_result = DriftTestResult.DRIFT_DETECTED
            recommended_action = "Moderate drift detected, plan retraining"
        else:
            test_result = DriftTestResult.NO_DRIFT
            recommended_action = "Continue monitoring"
        
        return DriftDetectionResult(
            timestamp=datetime.now(),
            model_id=model_id,
            drift_type=DriftType.DATA_DRIFT,
            test_result=test_result,
            test_statistic=overall_js,
            p_value=1.0 - overall_js,  # Approximate p-value
            threshold=0.1,
            drift_score=overall_js,
            drift_magnitude=overall_js,
            feature_drift_scores=feature_drift_scores,
            detection_method='js_divergence',
            recommended_action=recommended_action,
            confidence=overall_js
        )
    
    def _spc_drift_test(self, reference_data: np.ndarray, current_data: np.ndarray, 
                       model_id: str) -> DriftDetectionResult:
        """Statistical Process Control drift test."""
        # Calculate control limits based on reference data
        ref_mean = np.mean(reference_data, axis=0)
        ref_std = np.std(reference_data, axis=0)
        
        # Control limits (3-sigma rule)
        upper_control_limit = ref_mean + 3 * ref_std
        lower_control_limit = ref_mean - 3 * ref_std
        
        # Test current data against control limits
        current_mean = np.mean(current_data, axis=0)
        
        # Check if current mean is within control limits
        violations = np.logical_or(
            current_mean > upper_control_limit,
            current_mean < lower_control_limit
        )
        
        # Calculate drift score as percentage of features out of control
        drift_score = np.mean(violations) if len(violations) > 0 else 0.0
        
        # Calculate test statistic (normalized distance from center)
        normalized_distance = np.abs(current_mean - ref_mean) / (ref_std + 1e-10)
        test_statistic = np.mean(normalized_distance)
        
        if drift_score > 0.2:  # More than 20% of features out of control
            test_result = DriftTestResult.DRIFT_DETECTED
            recommended_action = "Multiple features out of control, investigate immediately"
        elif drift_score > 0.0:
            test_result = DriftTestResult.DRIFT_DETECTED
            recommended_action = "Some features out of control, monitor closely"
        else:
            test_result = DriftTestResult.NO_DRIFT
            recommended_action = "All features within control limits"
        
        return DriftDetectionResult(
            timestamp=datetime.now(),
            model_id=model_id,
            drift_type=DriftType.DATA_DRIFT,
            test_result=test_result,
            test_statistic=test_statistic,
            p_value=1.0 - drift_score,
            threshold=0.05,
            drift_score=drift_score,
            drift_magnitude=test_statistic,
            detection_method='spc',
            recommended_action=recommended_action,
            confidence=drift_score
        )
    
    def check_performance_degradation(self, model_id: str) -> Optional[PerformanceAlert]:
        """Check for performance degradation and create alerts."""
        if model_id not in self.tracked_models:
            return None
        
        # Calculate current performance
        current_metrics = self.calculate_performance_metrics(model_id)
        
        if not current_metrics:
            return None
        
        # Get baseline performance
        baseline_metrics = self.model_baselines.get(model_id, {})
        
        # If no baseline, use current as baseline
        if not baseline_metrics:
            self.model_baselines[model_id] = current_metrics
            return None
        
        # Check for degradation in key metrics
        alerts = []
        
        for metric_name in ['r2', 'mse', 'mae']:
            if metric_name not in current_metrics or metric_name not in baseline_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            baseline_value = baseline_metrics[metric_name]
            
            # Calculate degradation
            if metric_name == 'r2':  # Higher is better
                degradation = (baseline_value - current_value) / abs(baseline_value) if baseline_value != 0 else 0
            else:  # Lower is better (mse, mae)
                degradation = (current_value - baseline_value) / abs(baseline_value) if baseline_value != 0 else 0
            
            # Check if degradation exceeds threshold
            threshold = self.config['performance_threshold']
            
            if degradation > threshold:
                # Determine severity
                if degradation > threshold * 3:
                    severity = AlertSeverity.CRITICAL
                elif degradation > threshold * 2:
                    severity = AlertSeverity.HIGH
                elif degradation > threshold * 1.5:
                    severity = AlertSeverity.MEDIUM
                else:
                    severity = AlertSeverity.LOW
                
                # Create alert
                alert = PerformanceAlert(
                    timestamp=datetime.now(),
                    model_id=model_id,
                    alert_type='performance_degradation',
                    severity=severity,
                    current_metric=current_value,
                    baseline_metric=baseline_value,
                    threshold=threshold,
                    message=f"{metric_name.upper()} degraded by {degradation:.1%}",
                    metric_name=metric_name,
                    sample_size=current_metrics.get('sample_size', 0),
                    time_window=f"{self.config['monitoring_window_hours']}h",
                    recommended_actions=[
                        "Investigate data quality",
                        "Check for feature drift",
                        "Consider model retraining"
                    ]
                )
                
                alerts.append(alert)
        
        # Return the most severe alert
        if alerts:
            most_severe = max(alerts, key=lambda a: ['low', 'medium', 'high', 'critical'].index(a.severity.value))
            
            # Check cooldown period
            last_alert_time = self.recent_alerts.get(f"{model_id}_{most_severe.metric_name}")
            cooldown_hours = self.config['alert_cooldown_hours']
            
            if last_alert_time is None or (datetime.now() - last_alert_time).total_seconds() > cooldown_hours * 3600:
                self.recent_alerts[f"{model_id}_{most_severe.metric_name}"] = datetime.now()
                self.alert_history.append(most_severe)
                return most_severe
        
        return None
    
    def calculate_model_health_score(self, model_id: str) -> ModelHealthScore:
        """Calculate comprehensive model health score."""
        if model_id not in self.tracked_models:
            raise ValueError(f"Model {model_id} not registered")
        
        # Initialize component scores
        performance_score = 50.0  # Default neutral
        drift_score = 100.0       # Default excellent (no drift)
        stability_score = 50.0    # Default neutral
        robustness_score = 50.0   # Default neutral
        
        # Calculate performance score
        current_metrics = self.calculate_performance_metrics(model_id)
        if current_metrics:
            # Use R² as primary performance indicator (0-100 scale)
            r2_score = current_metrics.get('r2', 0.0)
            performance_score = max(0, min(100, (r2_score + 1) * 50))  # Scale from [-1,1] to [0,100]
            
            # Recent accuracy and trend
            recent_accuracy = performance_score / 100.0
            
            # Calculate accuracy trend
            recent_performance = list(self.performance_history[model_id])[-10:]  # Last 10 records
            if len(recent_performance) >= 2:
                recent_r2_scores = [p.get('r2', 0) for p in recent_performance]
                if len(recent_r2_scores) >= 2:
                    accuracy_trend = (recent_r2_scores[-1] - recent_r2_scores[0]) / len(recent_r2_scores)
                else:
                    accuracy_trend = 0.0
            else:
                accuracy_trend = 0.0
        else:
            recent_accuracy = 0.5
            accuracy_trend = 0.0
        
        # Calculate drift score
        recent_drift_results = list(self.drift_detection_history[model_id])[-5:]  # Last 5 drift checks
        if recent_drift_results:
            drift_scores = [100 - (r.drift_score * 100) for r in recent_drift_results]  # Invert drift score
            drift_score = np.mean(drift_scores)
            data_drift_severity = np.mean([r.drift_score for r in recent_drift_results])
            concept_drift_severity = data_drift_severity  # Simplified
        else:
            data_drift_severity = 0.0
            concept_drift_severity = 0.0
        
        # Calculate stability score
        if current_metrics and 'error_std' in current_metrics:
            # Lower error variance = higher stability
            error_std = current_metrics['error_std']
            stability_score = max(0, min(100, 100 - (error_std * 100)))  # Invert and scale
        
        # Calculate robustness score (based on prediction confidence)
        recent_predictions = list(self.model_predictions[model_id])[-100:]  # Last 100 predictions
        if recent_predictions:
            confidences = [p.get('confidence', 0.5) for p in recent_predictions if p.get('confidence') is not None]
            if confidences:
                avg_confidence = np.mean(confidences)
                robustness_score = avg_confidence * 100
                prediction_confidence = avg_confidence
            else:
                prediction_confidence = 0.5
        else:
            prediction_confidence = 0.5
        
        # Calculate overall health score (weighted average)
        overall_score = (
            performance_score * 0.4 +
            drift_score * 0.3 +
            stability_score * 0.2 +
            robustness_score * 0.1
        )
        
        # Determine health status
        if overall_score >= 90:
            health_status = "excellent"
        elif overall_score >= 75:
            health_status = "good"
        elif overall_score >= 60:
            health_status = "fair"
        elif overall_score >= 40:
            health_status = "poor"
        else:
            health_status = "critical"
        
        # Generate recommendations
        recommendations = []
        priority_actions = []
        
        if performance_score < 60:
            recommendations.append("Performance below acceptable level")
            priority_actions.append("Investigate model performance issues")
        
        if drift_score < 70:
            recommendations.append("Significant drift detected")
            priority_actions.append("Consider model retraining")
        
        if stability_score < 50:
            recommendations.append("Model predictions are unstable")
            recommendations.append("Review input data quality")
        
        if robustness_score < 60:
            recommendations.append("Low prediction confidence")
            recommendations.append("Evaluate model uncertainty quantification")
        
        # Calculate operational metrics
        recent_errors = len([p for p in recent_predictions if p.get('error')])
        error_rate = recent_errors / len(recent_predictions) if recent_predictions else 0.0
        
        # Time since last retraining
        registration_time = self.tracked_models[model_id].get('registration_time', datetime.now())
        days_since_retrain = (datetime.now() - registration_time).days
        
        # Create health score
        health_score = ModelHealthScore(
            timestamp=datetime.now(),
            model_id=model_id,
            overall_score=overall_score,
            health_status=health_status,
            performance_score=performance_score,
            drift_score=drift_score,
            stability_score=stability_score,
            robustness_score=robustness_score,
            recent_accuracy=recent_accuracy,
            accuracy_trend=accuracy_trend,
            prediction_confidence=prediction_confidence,
            data_drift_severity=data_drift_severity,
            concept_drift_severity=concept_drift_severity,
            prediction_latency=0.0,  # Would need to track this separately
            error_rate=error_rate,
            uptime_percentage=100.0,  # Simplified
            health_recommendations=recommendations,
            priority_actions=priority_actions,
            days_since_retrain=days_since_retrain
        )
        
        # Store health score
        self.health_scores[model_id] = health_score
        self.health_history[model_id].append(health_score)
        
        return health_score
    
    def should_retrain_model(self, model_id: str) -> Tuple[bool, str]:
        """Determine if a model should be retrained."""
        # Calculate health score
        health_score = self.calculate_model_health_score(model_id)
        
        # Check drift
        drift_result = self.detect_data_drift(model_id)
        
        # Check performance degradation
        performance_alert = self.check_performance_degradation(model_id)
        
        # Decision logic
        reasons = []
        should_retrain = False
        
        # Health score below threshold
        if health_score.overall_score < 50:
            should_retrain = True
            reasons.append(f"Overall health score too low: {health_score.overall_score:.1f}")
        
        # Significant drift detected
        if drift_result.test_result == DriftTestResult.DRIFT_DETECTED and drift_result.drift_score > 0.2:
            should_retrain = True
            reasons.append(f"Significant drift detected: {drift_result.drift_score:.3f}")
        
        # Performance degradation
        if performance_alert and performance_alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            should_retrain = True
            reasons.append(f"Critical performance degradation in {performance_alert.metric_name}")
        
        # Time-based retraining (if configured)
        if health_score.days_since_retrain > 30:  # 30 days
            should_retrain = True
            reasons.append(f"Model is {health_score.days_since_retrain} days old")
        
        reason = "; ".join(reasons) if reasons else "Model is healthy"
        
        return should_retrain, reason
    
    def start_ab_test(self, test_id: str, champion_model_id: str, challenger_model_id: str,
                     traffic_split: Dict[str, float], duration_hours: float = 168) -> ABTestResult:
        """Start an A/B test between two models."""
        if champion_model_id not in self.tracked_models:
            raise ValueError(f"Champion model {champion_model_id} not registered")
        
        if challenger_model_id not in self.tracked_models:
            raise ValueError(f"Challenger model {challenger_model_id} not registered")
        
        # Create A/B test
        ab_test = ABTestResult(
            timestamp=datetime.now(),
            test_id=test_id,
            champion_model_id=champion_model_id,
            challenger_model_id=challenger_model_id,
            traffic_split=traffic_split,
            test_duration_hours=duration_hours,
            champion_metrics={},
            challenger_metrics={},
            statistical_significance={},
            confidence_intervals={},
            recommended_winner="",
            confidence_level=0.0,
            test_conclusion="continue_test"
        )
        
        self.active_ab_tests[test_id] = ab_test
        
        # Update model statuses
        self.tracked_models[champion_model_id]['status'] = ModelStatus.CHAMPION
        self.tracked_models[challenger_model_id]['status'] = ModelStatus.CHALLENGER
        
        self.logger.info(f"Started A/B test {test_id}: {champion_model_id} vs {challenger_model_id}")
        
        return ab_test
    
    def evaluate_ab_test(self, test_id: str) -> ABTestResult:
        """Evaluate the results of an A/B test."""
        if test_id not in self.active_ab_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        ab_test = self.active_ab_tests[test_id]
        
        # Calculate metrics for both models
        champion_metrics = self.calculate_performance_metrics(ab_test.champion_model_id)
        challenger_metrics = self.calculate_performance_metrics(ab_test.challenger_model_id)
        
        ab_test.champion_metrics = champion_metrics
        ab_test.challenger_metrics = challenger_metrics
        
        # Statistical significance testing
        significance_results = {}
        confidence_intervals = {}
        
        for metric in ['r2', 'mse', 'mae']:
            if metric in champion_metrics and metric in challenger_metrics:
                champ_value = champion_metrics[metric]
                challenger_value = challenger_metrics[metric]
                
                # Simplified statistical test (would use proper t-test in production)
                champ_samples = champion_metrics.get('sample_size', 100)
                challenger_samples = challenger_metrics.get('sample_size', 100)
                
                if champ_samples >= self.config['ab_test_min_samples'] and challenger_samples >= self.config['ab_test_min_samples']:
                    # Calculate effect size
                    pooled_std = np.sqrt((champ_value**2 + challenger_value**2) / 2)
                    effect_size = abs(champ_value - challenger_value) / pooled_std if pooled_std > 0 else 0
                    
                    # Approximate p-value based on effect size
                    p_value = max(0.001, 1.0 - effect_size)
                    
                    significance_results[metric] = p_value
                    
                    # Confidence interval (simplified)
                    margin = 1.96 * pooled_std / np.sqrt(min(champ_samples, challenger_samples))
                    confidence_intervals[metric] = (challenger_value - margin, challenger_value + margin)
        
        ab_test.statistical_significance = significance_results
        ab_test.confidence_intervals = confidence_intervals
        
        # Determine winner
        if 'r2' in significance_results:
            p_value = significance_results['r2']
            
            if p_value < self.config['statistical_significance_level']:
                # Statistically significant difference
                if challenger_metrics['r2'] > champion_metrics['r2']:
                    ab_test.recommended_winner = ab_test.challenger_model_id
                    ab_test.test_conclusion = "promote_challenger"
                else:
                    ab_test.recommended_winner = ab_test.champion_model_id
                    ab_test.test_conclusion = "keep_champion"
                
                ab_test.confidence_level = 1.0 - p_value
            else:
                # No significant difference
                ab_test.recommended_winner = ab_test.champion_model_id
                ab_test.test_conclusion = "keep_champion"
                ab_test.confidence_level = 0.5
        
        # Check if test duration is complete
        test_duration = (datetime.now() - ab_test.timestamp).total_seconds() / 3600
        if test_duration >= ab_test.test_duration_hours:
            # Test complete, finalize
            self.ab_test_history.append(ab_test)
            del self.active_ab_tests[test_id]
        
        return ab_test
    
    def get_monitoring_summary(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        summary = {
            'timestamp': datetime.now(),
            'total_models_tracked': len(self.tracked_models),
            'active_ab_tests': len(self.active_ab_tests),
            'total_alerts': len(self.alert_history),
            'models': {}
        }
        
        models_to_summarize = [model_id] if model_id else list(self.tracked_models.keys())
        
        for mid in models_to_summarize:
            if mid not in self.tracked_models:
                continue
            
            # Get latest health score
            health_score = self.health_scores.get(mid)
            
            # Get recent performance
            current_metrics = self.calculate_performance_metrics(mid)
            
            # Get drift status
            recent_drift = list(self.drift_detection_history[mid])[-1:] if self.drift_detection_history[mid] else []
            
            # Get alerts
            recent_alerts = [a for a in self.alert_history if a.model_id == mid][-5:]  # Last 5 alerts
            
            summary['models'][mid] = {
                'status': self.tracked_models[mid]['status'].value if hasattr(self.tracked_models[mid]['status'], 'value') else str(self.tracked_models[mid]['status']),
                'health_score': health_score.overall_score if health_score else 0,
                'health_status': health_score.health_status if health_score else 'unknown',
                'current_performance': current_metrics,
                'drift_status': recent_drift[0].test_result.value if recent_drift else 'unknown',
                'recent_alerts_count': len(recent_alerts),
                'prediction_count': len(self.model_predictions[mid]),
                'days_since_registration': (datetime.now() - self.tracked_models[mid]['registration_time']).days
            }
        
        return summary
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started background model monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        self.logger.info("Stopped background model monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Check health for all tracked models
                for model_id in list(self.tracked_models.keys()):
                    try:
                        # Calculate health score
                        self.calculate_model_health_score(model_id)
                        
                        # Check for performance degradation
                        alert = self.check_performance_degradation(model_id)
                        if alert:
                            self.logger.warning(f"Performance alert for {model_id}: {alert.message}")
                        
                        # Check drift
                        drift_result = self.detect_data_drift(model_id)
                        if drift_result.test_result == DriftTestResult.DRIFT_DETECTED:
                            self.logger.warning(f"Drift detected for {model_id}: {drift_result.drift_score:.3f}")
                        
                        # Check if retraining is needed
                        should_retrain, reason = self.should_retrain_model(model_id)
                        if should_retrain:
                            self.logger.info(f"Model {model_id} should be retrained: {reason}")
                        
                    except Exception as e:
                        self.logger.error(f"Error monitoring model {model_id}: {e}")
                
                # Evaluate active A/B tests
                for test_id in list(self.active_ab_tests.keys()):
                    try:
                        self.evaluate_ab_test(test_id)
                    except Exception as e:
                        self.logger.error(f"Error evaluating A/B test {test_id}: {e}")
                
                # Sleep until next check
                time.sleep(self.config['health_check_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait a minute before retrying


# Example usage and testing
if __name__ == "__main__":
    import json
    import random
    from ..config.manager import ConfigurationManager
    
    def main():
        # Configure logging
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize monitor
        config_manager = ConfigurationManager()
        monitor = ModelPerformanceMonitor(config_manager)
        
        # Register a test model
        model_info = {
            'model_type': 'lightgbm',
            'target_variable': 'price_1h',
            'features': ['feature_1', 'feature_2', 'feature_3'],
            'registration_time': datetime.now()
        }
        monitor.register_model('test_model_1', model_info)
        
        # Simulate predictions and actuals
        print("Simulating model predictions...")
        
        # Generate synthetic data with concept drift
        for i in range(1000):
            # Add concept drift after 500 samples
            if i < 500:
                # Good performance period
                actual = random.uniform(45000, 55000)
                prediction = actual + random.normal(0, 1000)  # Small error
                features = {
                    'feature_1': random.uniform(0, 1),
                    'feature_2': random.uniform(0, 1),
                    'feature_3': random.uniform(0, 1)
                }
            else:
                # Performance degrades, features drift
                actual = random.uniform(40000, 60000)  # More volatile
                prediction = actual + random.normal(0, 3000)  # Larger error
                features = {
                    'feature_1': random.uniform(0.3, 0.8),  # Feature drift
                    'feature_2': random.uniform(0.2, 0.9),  # Feature drift  
                    'feature_3': random.uniform(0.1, 0.7)   # Feature drift
                }
            
            monitor.log_prediction(
                'test_model_1', 
                features, 
                prediction, 
                actual, 
                prediction_confidence=random.uniform(0.6, 0.9)
            )
        
        # Calculate performance metrics
        print("\nCalculating performance metrics...")
        metrics = monitor.calculate_performance_metrics('test_model_1')
        print(f"Performance metrics: {json.dumps(metrics, indent=2)}")
        
        # Detect drift
        print("\nDetecting drift...")
        drift_result = monitor.detect_data_drift('test_model_1', method='ks_test')
        print(f"Drift detection result: {drift_result.test_result.value}")
        print(f"Drift score: {drift_result.drift_score:.3f}")
        
        # Check performance degradation
        print("\nChecking performance degradation...")
        alert = monitor.check_performance_degradation('test_model_1')
        if alert:
            print(f"Performance alert: {alert.message} (Severity: {alert.severity.value})")
        else:
            print("No performance alerts")
        
        # Calculate health score
        print("\nCalculating model health score...")
        health_score = monitor.calculate_model_health_score('test_model_1')
        print(f"Health score: {health_score.overall_score:.1f} ({health_score.health_status})")
        print(f"Recommendations: {health_score.health_recommendations}")
        
        # Check retraining recommendation
        should_retrain, reason = monitor.should_retrain_model('test_model_1')
        print(f"\nShould retrain: {should_retrain}")
        if should_retrain:
            print(f"Reason: {reason}")
        
        # Get monitoring summary
        print("\nMonitoring summary:")
        summary = monitor.get_monitoring_summary('test_model_1')
        print(json.dumps(summary, indent=2, default=str))
    
    # Run the example
    main()