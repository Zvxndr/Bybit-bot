"""
Enhanced Performance Tracker - Phase 1 Integration

Advanced performance monitoring and tracking for optimized components:
- Execution engine performance tracking
- ML inference performance monitoring  
- Slippage and liquidity metrics
- Real-time dashboard integration
- Advanced analytics and alerting

Integration Target: Monitor all Phase 1 optimizations in real-time
Current Status: ✅ IMPLEMENTED
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics"""
    EXECUTION_TIME = "execution_time"
    SLIPPAGE = "slippage"
    FILL_RATE = "fill_rate"
    ML_INFERENCE = "ml_inference"
    LIQUIDITY = "liquidity"
    ACCURACY = "accuracy"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    metric_type: MetricType
    value: float
    timestamp: float
    component: str
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}

@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_level: AlertLevel
    metric_type: MetricType
    message: str
    current_value: float
    threshold_value: float
    component: str
    timestamp: float
    resolved: bool = False

@dataclass
class ComponentStatus:
    """Status of a monitored component"""
    component_name: str
    is_healthy: bool
    performance_score: float  # 0-100
    key_metrics: Dict[str, float]
    last_updated: float
    alerts: List[PerformanceAlert]

class EnhancedPerformanceTracker:
    """
    Enhanced performance tracking for optimized components
    
    Features:
    - Real-time metric collection ✅
    - Performance threshold monitoring ✅
    - Component health tracking ✅
    - Advanced analytics and alerting ✅
    - Dashboard integration ready ✅
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Performance thresholds
        self.thresholds = {
            MetricType.EXECUTION_TIME: {
                'target': 80.0,  # <80ms
                'warning': 100.0,
                'critical': 150.0
            },
            MetricType.SLIPPAGE: {
                'target': 5.0,   # <5bps
                'warning': 8.0,
                'critical': 15.0
            },
            MetricType.FILL_RATE: {
                'target': 98.0,  # >98%
                'warning': 95.0,
                'critical': 90.0
            },
            MetricType.ML_INFERENCE: {
                'target': 20.0,  # <20ms
                'warning': 30.0,
                'critical': 50.0
            },
            MetricType.ACCURACY: {
                'target': 85.0,  # >85%
                'warning': 80.0,
                'critical': 75.0
            }
        }
        
        # Metric storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=10000))
        self.real_time_metrics = {}
        self.component_status = {}
        
        # Alert system
        self.active_alerts = []
        self.alert_history = deque(maxlen=1000)
        self.alert_callbacks = []
        
        # Performance analytics
        self.performance_trends = {}
        self.anomaly_detector = None
        
        # Dashboard data
        self.dashboard_data = {}
        self.dashboard_update_interval = 5.0  # seconds
        
        logger.info("EnhancedPerformanceTracker initialized")

    async def record_execution_performance(self, 
                                         execution_time_ms: float,
                                         slippage_bps: float,
                                         fill_rate: float,
                                         component: str = "execution_engine") -> None:
        """Record execution engine performance metrics"""
        timestamp = time.time()
        
        # Record individual metrics
        await self._record_metric(MetricType.EXECUTION_TIME, execution_time_ms, component, timestamp)
        await self._record_metric(MetricType.SLIPPAGE, slippage_bps, component, timestamp)
        await self._record_metric(MetricType.FILL_RATE, fill_rate, component, timestamp)
        
        # Update component status
        await self._update_component_status(component, {
            'execution_time_ms': execution_time_ms,
            'slippage_bps': slippage_bps,
            'fill_rate': fill_rate
        })
        
        # Check for alerts
        await self._check_thresholds(component)

    async def record_ml_performance(self, 
                                  inference_time_ms: float,
                                  accuracy: float,
                                  memory_usage_mb: float,
                                  component: str = "ml_engine") -> None:
        """Record ML engine performance metrics"""
        timestamp = time.time()
        
        # Record metrics
        await self._record_metric(MetricType.ML_INFERENCE, inference_time_ms, component, timestamp)
        await self._record_metric(MetricType.ACCURACY, accuracy, component, timestamp)
        await self._record_metric(MetricType.MEMORY_USAGE, memory_usage_mb, component, timestamp)
        
        # Update component status
        await self._update_component_status(component, {
            'inference_time_ms': inference_time_ms,
            'accuracy': accuracy,
            'memory_usage_mb': memory_usage_mb
        })

    async def record_liquidity_performance(self, 
                                         discovery_time_ms: float,
                                         sources_found: int,
                                         hidden_liquidity_pct: float,
                                         component: str = "liquidity_seeker") -> None:
        """Record liquidity seeker performance metrics"""
        timestamp = time.time()
        
        # Record metrics
        await self._record_metric(MetricType.EXECUTION_TIME, discovery_time_ms, component, timestamp, {
            'metric_subtype': 'discovery_time'
        })
        await self._record_metric(MetricType.LIQUIDITY, sources_found, component, timestamp, {
            'metric_subtype': 'sources_found'
        })
        await self._record_metric(MetricType.LIQUIDITY, hidden_liquidity_pct, component, timestamp, {
            'metric_subtype': 'hidden_liquidity_pct'
        })

    async def _record_metric(self, 
                           metric_type: MetricType, 
                           value: float, 
                           component: str, 
                           timestamp: float,
                           additional_data: Dict[str, Any] = None) -> None:
        """Record individual metric"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            component=component,
            additional_data=additional_data or {}
        )
        
        # Store in history
        key = f"{component}_{metric_type.value}"
        self.metrics_history[key].append(metric)
        
        # Update real-time metrics
        self.real_time_metrics[key] = metric
        
        # Update performance trends
        await self._update_performance_trends(key, metric)

    async def _update_component_status(self, 
                                     component: str, 
                                     metrics: Dict[str, float]) -> None:
        """Update component health status"""
        # Calculate performance score
        performance_score = await self._calculate_performance_score(component, metrics)
        
        # Determine health status
        is_healthy = performance_score >= 70.0  # 70% threshold for healthy
        
        # Get active alerts for component
        component_alerts = [alert for alert in self.active_alerts 
                          if alert.component == component and not alert.resolved]
        
        # Update status
        self.component_status[component] = ComponentStatus(
            component_name=component,
            is_healthy=is_healthy,
            performance_score=performance_score,
            key_metrics=metrics,
            last_updated=time.time(),
            alerts=component_alerts
        )

    async def _calculate_performance_score(self, 
                                         component: str, 
                                         metrics: Dict[str, float]) -> float:
        """Calculate overall performance score for component"""
        scores = []
        
        for metric_name, value in metrics.items():
            # Map metric name to MetricType
            metric_type = self._get_metric_type_from_name(metric_name)
            if not metric_type:
                continue
            
            threshold = self.thresholds.get(metric_type)
            if not threshold:
                continue
            
            # Calculate score based on thresholds
            target = threshold['target']
            warning = threshold['warning']
            critical = threshold['critical']
            
            if metric_type in [MetricType.FILL_RATE, MetricType.ACCURACY]:
                # Higher is better metrics
                if value >= target:
                    score = 100.0
                elif value >= warning:
                    score = 70.0 + 30.0 * (value - warning) / (target - warning)
                elif value >= critical:
                    score = 30.0 + 40.0 * (value - critical) / (warning - critical)
                else:
                    score = 0.0
            else:
                # Lower is better metrics
                if value <= target:
                    score = 100.0
                elif value <= warning:
                    score = 70.0 + 30.0 * (warning - value) / (warning - target)
                elif value <= critical:
                    score = 30.0 + 40.0 * (critical - value) / (critical - warning)
                else:
                    score = 0.0
            
            scores.append(score)
        
        return np.mean(scores) if scores else 50.0  # Default neutral score

    def _get_metric_type_from_name(self, metric_name: str) -> Optional[MetricType]:
        """Map metric name to MetricType"""
        mapping = {
            'execution_time_ms': MetricType.EXECUTION_TIME,
            'slippage_bps': MetricType.SLIPPAGE,
            'fill_rate': MetricType.FILL_RATE,
            'inference_time_ms': MetricType.ML_INFERENCE,
            'accuracy': MetricType.ACCURACY,
            'memory_usage_mb': MetricType.MEMORY_USAGE
        }
        return mapping.get(metric_name)

    async def _check_thresholds(self, component: str) -> None:
        """Check metrics against thresholds and generate alerts"""
        component_metrics = {k: v for k, v in self.real_time_metrics.items() 
                           if k.startswith(component)}
        
        for key, metric in component_metrics.items():
            threshold = self.thresholds.get(metric.metric_type)
            if not threshold:
                continue
            
            alert_level = None
            threshold_value = None
            
            if metric.metric_type in [MetricType.FILL_RATE, MetricType.ACCURACY]:
                # Higher is better
                if metric.value < threshold['critical']:
                    alert_level = AlertLevel.CRITICAL
                    threshold_value = threshold['critical']
                elif metric.value < threshold['warning']:
                    alert_level = AlertLevel.WARNING
                    threshold_value = threshold['warning']
            else:
                # Lower is better
                if metric.value > threshold['critical']:
                    alert_level = AlertLevel.CRITICAL
                    threshold_value = threshold['critical']
                elif metric.value > threshold['warning']:
                    alert_level = AlertLevel.WARNING
                    threshold_value = threshold['warning']
            
            if alert_level:
                await self._generate_alert(
                    alert_level=alert_level,
                    metric_type=metric.metric_type,
                    current_value=metric.value,
                    threshold_value=threshold_value,
                    component=component
                )

    async def _generate_alert(self, 
                            alert_level: AlertLevel,
                            metric_type: MetricType,
                            current_value: float,
                            threshold_value: float,
                            component: str) -> None:
        """Generate performance alert"""
        message = self._create_alert_message(
            alert_level, metric_type, current_value, threshold_value, component
        )
        
        alert = PerformanceAlert(
            alert_level=alert_level,
            metric_type=metric_type,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            component=component,
            timestamp=time.time()
        )
        
        # Check if similar alert already exists
        existing_alert = None
        for existing in self.active_alerts:
            if (existing.component == component and 
                existing.metric_type == metric_type and 
                not existing.resolved):
                existing_alert = existing
                break
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = current_value
            existing_alert.message = message
            existing_alert.timestamp = time.time()
        else:
            # Add new alert
            self.active_alerts.append(alert)
        
        # Store in history
        self.alert_history.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _create_alert_message(self, 
                            alert_level: AlertLevel,
                            metric_type: MetricType,
                            current_value: float,
                            threshold_value: float,
                            component: str) -> str:
        """Create human-readable alert message"""
        metric_name = metric_type.value.replace('_', ' ').title()
        
        if metric_type in [MetricType.FILL_RATE, MetricType.ACCURACY]:
            comparison = "below"
            unit = "%"
        elif metric_type in [MetricType.EXECUTION_TIME, MetricType.ML_INFERENCE]:
            comparison = "above"
            unit = "ms"
        elif metric_type == MetricType.SLIPPAGE:
            comparison = "above"
            unit = "bps"
        else:
            comparison = "outside threshold"
            unit = ""
        
        return (f"{alert_level.value.upper()}: {component} {metric_name} "
               f"is {comparison} threshold - Current: {current_value:.2f}{unit}, "
               f"Threshold: {threshold_value:.2f}{unit}")

    async def _update_performance_trends(self, key: str, metric: PerformanceMetric) -> None:
        """Update performance trend analysis"""
        if key not in self.performance_trends:
            self.performance_trends[key] = {
                'values': deque(maxlen=100),
                'timestamps': deque(maxlen=100),
                'trend': 'stable',
                'trend_strength': 0.0
            }
        
        trend_data = self.performance_trends[key]
        trend_data['values'].append(metric.value)
        trend_data['timestamps'].append(metric.timestamp)
        
        # Calculate trend if we have enough data
        if len(trend_data['values']) >= 10:
            values = list(trend_data['values'])[-20:]  # Last 20 points
            trend_slope = np.polyfit(range(len(values)), values, 1)[0]
            
            # Determine trend
            if abs(trend_slope) < 0.1:
                trend_data['trend'] = 'stable'
            elif trend_slope > 0:
                trend_data['trend'] = 'increasing'
            else:
                trend_data['trend'] = 'decreasing'
            
            trend_data['trend_strength'] = abs(trend_slope)

    async def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time data for dashboard"""
        dashboard_data = {
            'timestamp': time.time(),
            'component_status': {},
            'key_metrics': {},
            'alerts': {
                'active': len([a for a in self.active_alerts if not a.resolved]),
                'critical': len([a for a in self.active_alerts if a.alert_level == AlertLevel.CRITICAL and not a.resolved]),
                'recent': [asdict(alert) for alert in list(self.alert_history)[-10:]]
            },
            'performance_summary': {}
        }
        
        # Component status
        for component, status in self.component_status.items():
            dashboard_data['component_status'][component] = {
                'healthy': status.is_healthy,
                'score': status.performance_score,
                'last_updated': status.last_updated,
                'metrics': status.key_metrics
            }
        
        # Key metrics
        for key, metric in self.real_time_metrics.items():
            dashboard_data['key_metrics'][key] = {
                'value': metric.value,
                'timestamp': metric.timestamp,
                'component': metric.component,
                'type': metric.metric_type.value
            }
        
        # Performance summary
        execution_times = [m.value for k, m in self.real_time_metrics.items() 
                          if 'execution_time' in k]
        if execution_times:
            dashboard_data['performance_summary']['avg_execution_time'] = np.mean(execution_times)
        
        ml_times = [m.value for k, m in self.real_time_metrics.items() 
                   if 'ml_inference' in k]
        if ml_times:
            dashboard_data['performance_summary']['avg_ml_inference'] = np.mean(ml_times)
        
        return dashboard_data

    async def get_performance_report(self, 
                                   component: Optional[str] = None,
                                   time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        end_time = time.time()
        start_time = end_time - (time_window_hours * 3600)
        
        report = {
            'report_period': {
                'start': datetime.fromtimestamp(start_time).isoformat(),
                'end': datetime.fromtimestamp(end_time).isoformat(),
                'duration_hours': time_window_hours
            },
            'component_performance': {},
            'alerts_summary': {},
            'trends': {},
            'recommendations': []
        }
        
        # Filter metrics by time window and component
        filtered_metrics = {}
        for key, metrics in self.metrics_history.items():
            if component and not key.startswith(component):
                continue
            
            time_filtered = [m for m in metrics if start_time <= m.timestamp <= end_time]
            if time_filtered:
                filtered_metrics[key] = time_filtered
        
        # Analyze each metric
        for key, metrics in filtered_metrics.items():
            values = [m.value for m in metrics]
            component_name = metrics[0].component
            metric_type = metrics[0].metric_type
            
            if component_name not in report['component_performance']:
                report['component_performance'][component_name] = {}
            
            report['component_performance'][component_name][metric_type.value] = {
                'count': len(values),
                'avg': np.mean(values),
                'min': np.min(values),
                'max': np.max(values),
                'p95': np.percentile(values, 95),
                'trend': self.performance_trends.get(key, {}).get('trend', 'unknown')
            }
        
        # Alert summary
        period_alerts = [a for a in self.alert_history 
                        if start_time <= a.timestamp <= end_time]
        if component:
            period_alerts = [a for a in period_alerts if a.component == component]
        
        report['alerts_summary'] = {
            'total': len(period_alerts),
            'by_level': {level.value: len([a for a in period_alerts if a.alert_level == level])
                        for level in AlertLevel},
            'by_component': {comp: len([a for a in period_alerts if a.component == comp])
                           for comp in set(a.component for a in period_alerts)}
        }
        
        # Generate recommendations
        report['recommendations'] = await self._generate_recommendations(filtered_metrics)
        
        return report

    async def _generate_recommendations(self, metrics: Dict[str, List[PerformanceMetric]]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        for key, metric_list in metrics.items():
            if not metric_list:
                continue
                
            values = [m.value for m in metric_list]
            avg_value = np.mean(values)
            metric_type = metric_list[0].metric_type
            component = metric_list[0].component
            
            threshold = self.thresholds.get(metric_type, {})
            target = threshold.get('target')
            
            if not target:
                continue
            
            # Check if performance is below target
            if metric_type in [MetricType.FILL_RATE, MetricType.ACCURACY]:
                if avg_value < target:
                    recommendations.append(
                        f"Improve {component} {metric_type.value}: "
                        f"Current {avg_value:.1f}% vs target {target:.1f}%"
                    )
            else:
                if avg_value > target:
                    recommendations.append(
                        f"Optimize {component} {metric_type.value}: "
                        f"Current {avg_value:.1f}ms vs target {target:.1f}ms"
                    )
        
        return recommendations

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)

    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        for alert in self.active_alerts:
            if str(id(alert)) == alert_id:
                alert.resolved = True
                return True
        return False

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.component_status:
            return {"status": "unknown", "message": "No components monitored"}
        
        all_healthy = all(status.is_healthy for status in self.component_status.values())
        avg_score = np.mean([status.performance_score for status in self.component_status.values()])
        
        critical_alerts = len([a for a in self.active_alerts 
                             if a.alert_level == AlertLevel.CRITICAL and not a.resolved])
        
        if critical_alerts > 0:
            overall_status = "critical"
        elif not all_healthy:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "overall_score": avg_score,
            "components_healthy": sum(1 for s in self.component_status.values() if s.is_healthy),
            "total_components": len(self.component_status),
            "active_alerts": len([a for a in self.active_alerts if not a.resolved]),
            "critical_alerts": critical_alerts
        }

# Example usage and integration
if __name__ == "__main__":
    # Example setup
    pass