"""
Monitoring Stack for Cloud Infrastructure.
Comprehensive monitoring, alerting, and observability system with Prometheus, Grafana, and custom metrics.
"""

import asyncio
import json
import time
import re
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
import hashlib
import warnings
warnings.filterwarnings('ignore')

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, push_to_gateway
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    import elasticsearch
    from elasticsearch import Elasticsearch
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"

class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class MonitoringComponent(Enum):
    """Monitoring components."""
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    ELASTICSEARCH = "elasticsearch"
    ALERTMANAGER = "alertmanager"
    JAEGER = "jaeger"

@dataclass
class MetricDefinition:
    """Metric definition."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: str = ""
    namespace: str = "trading"

@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    query: str
    severity: AlertSeverity
    threshold: Union[float, str]
    duration: str = "5m"
    description: str = ""
    runbook_url: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

@dataclass
class Dashboard:
    """Grafana dashboard configuration."""
    name: str
    uid: str
    panels: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    time_range: str = "1h"
    refresh_interval: str = "30s"
    variables: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: str
    service: str
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    fields: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TraceSpan:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class AlertInstance:
    """Active alert instance."""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    labels: Dict[str, str]
    started_at: datetime
    resolved_at: Optional[datetime] = None
    notification_sent: bool = False

class MonitoringStack:
    """Comprehensive monitoring and observability stack."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Monitoring configuration
        self.monitoring_config = {
            'enabled': True,
            'prometheus': {
                'enabled': True,
                'port': 9090,
                'pushgateway_url': 'http://localhost:9091',
                'scrape_interval': '15s',
                'retention': '30d'
            },
            'grafana': {
                'enabled': True,
                'port': 3000,
                'admin_user': 'admin',
                'admin_password': self._get_secure_grafana_password()
            },
            'alertmanager': {
                'enabled': True,
                'port': 9093,
                'webhook_url': 'http://localhost:8080/alerts',
                'smtp_server': 'smtp.gmail.com',
                'email_from': 'alerts@trading-system.com'
            },
            'elasticsearch': {
                'enabled': True,
                'host': 'localhost',
                'port': 9200,
                'index_pattern': 'trading-logs-*'
            },
            'tracing': {
                'enabled': True,
                'jaeger_endpoint': 'http://localhost:14268/api/traces',
                'sampling_rate': 0.1
            }
        }
        
    def _get_secure_grafana_password(self) -> str:
        """Get secure Grafana password from environment or generate one."""
        import os
        import secrets
        import string
        
        # Try environment variable first
        env_password = os.getenv('GRAFANA_ADMIN_PASSWORD')
        if env_password:
            return env_password
            
        # Generate secure password
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(16))
        
        # Log that password was generated (don't log actual password)
        self.logger.warning("Generated secure Grafana password. Set GRAFANA_ADMIN_PASSWORD env var to customize.")
        return password
        
        # Prometheus metrics registry
        self.metrics_registry = CollectorRegistry()
        self.prometheus_metrics: Dict[str, Any] = {}
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertInstance] = {}
        self.alert_history: List[AlertInstance] = []
        
        # Dashboards
        self.dashboards: Dict[str, Dashboard] = {}
        
        # Log aggregation
        self.log_buffer: List[LogEntry] = []
        self.elasticsearch_client: Optional[Elasticsearch] = None
        
        # Distributed tracing
        self.active_traces: Dict[str, List[TraceSpan]] = {}
        self.completed_traces: List[List[TraceSpan]] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.metrics_collection_task = None
        self.alert_evaluation_task = None
        self.log_shipping_task = None
        
        # Initialize components
        self._setup_default_metrics()
        self._setup_default_alerts()
        self._setup_default_dashboards()
        
        if HAS_ELASTICSEARCH and self.monitoring_config['elasticsearch']['enabled']:
            self._initialize_elasticsearch()
        
        self.logger.info("MonitoringStack initialized")
    
    def _setup_default_metrics(self):
        """Setup default metrics for trading system."""
        try:
            default_metrics = [
                # Trading metrics
                MetricDefinition(
                    name="orders_total",
                    metric_type=MetricType.COUNTER,
                    description="Total number of orders placed",
                    labels=["service", "order_type", "status"]
                ),
                MetricDefinition(
                    name="order_execution_duration",
                    metric_type=MetricType.HISTOGRAM,
                    description="Order execution duration in seconds",
                    labels=["service", "order_type"],
                    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
                ),
                MetricDefinition(
                    name="portfolio_value",
                    metric_type=MetricType.GAUGE,
                    description="Current portfolio value",
                    labels=["service", "currency"],
                    unit="USD"
                ),
                MetricDefinition(
                    name="pnl_realized",
                    metric_type=MetricType.GAUGE,
                    description="Realized PnL",
                    labels=["service", "strategy", "symbol"],
                    unit="USD"
                ),
                MetricDefinition(
                    name="risk_exposure",
                    metric_type=MetricType.GAUGE,
                    description="Current risk exposure",
                    labels=["service", "asset_class"],
                    unit="percent"
                ),
                
                # System metrics
                MetricDefinition(
                    name="service_requests_total",
                    metric_type=MetricType.COUNTER,
                    description="Total HTTP requests",
                    labels=["service", "method", "endpoint", "status"]
                ),
                MetricDefinition(
                    name="service_request_duration",
                    metric_type=MetricType.HISTOGRAM,
                    description="HTTP request duration",
                    labels=["service", "method", "endpoint"],
                    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
                ),
                MetricDefinition(
                    name="service_memory_usage",
                    metric_type=MetricType.GAUGE,
                    description="Memory usage in bytes",
                    labels=["service", "instance"],
                    unit="bytes"
                ),
                MetricDefinition(
                    name="service_cpu_usage",
                    metric_type=MetricType.GAUGE,
                    description="CPU usage percentage",
                    labels=["service", "instance"],
                    unit="percent"
                ),
                
                # ML metrics
                MetricDefinition(
                    name="model_predictions_total",
                    metric_type=MetricType.COUNTER,
                    description="Total model predictions",
                    labels=["service", "model_name", "model_version"]
                ),
                MetricDefinition(
                    name="model_inference_duration",
                    metric_type=MetricType.HISTOGRAM,
                    description="Model inference duration",
                    labels=["service", "model_name"],
                    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
                ),
                MetricDefinition(
                    name="model_accuracy",
                    metric_type=MetricType.GAUGE,
                    description="Model accuracy score",
                    labels=["service", "model_name", "dataset"],
                    unit="percent"
                ),
                
                # Infrastructure metrics
                MetricDefinition(
                    name="container_restarts_total",
                    metric_type=MetricType.COUNTER,
                    description="Container restart count",
                    labels=["service", "container", "reason"]
                ),
                MetricDefinition(
                    name="load_balancer_requests_total",
                    metric_type=MetricType.COUNTER,
                    description="Load balancer requests",
                    labels=["backend", "status"]
                ),
                MetricDefinition(
                    name="circuit_breaker_state",
                    metric_type=MetricType.GAUGE,
                    description="Circuit breaker state (0=closed, 1=open, 2=half-open)",
                    labels=["service", "target"]
                )
            ]
            
            # Create Prometheus metrics
            if HAS_PROMETHEUS:
                for metric_def in default_metrics:
                    self._create_prometheus_metric(metric_def)
                    
        except Exception as e:
            self.logger.error(f"Failed to setup default metrics: {e}")
    
    def _create_prometheus_metric(self, metric_def: MetricDefinition):
        """Create Prometheus metric from definition."""
        try:
            metric_name = f"{metric_def.namespace}_{metric_def.name}"
            
            if metric_def.metric_type == MetricType.COUNTER:
                metric = Counter(
                    metric_name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.metrics_registry
                )
            elif metric_def.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    metric_name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.metrics_registry
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                metric = Histogram(
                    metric_name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    buckets=metric_def.buckets or prometheus_client.DEFAULT_BUCKETS,
                    registry=self.metrics_registry
                )
            elif metric_def.metric_type == MetricType.SUMMARY:
                metric = Summary(
                    metric_name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.metrics_registry
                )
            else:
                return
            
            self.prometheus_metrics[metric_def.name] = metric
            
        except Exception as e:
            self.logger.error(f"Failed to create Prometheus metric {metric_def.name}: {e}")
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        try:
            default_alerts = [
                # Critical trading alerts
                AlertRule(
                    name="high_order_failure_rate",
                    query='rate(trading_orders_total{status="failed"}[5m]) > 0.1',
                    severity=AlertSeverity.CRITICAL,
                    threshold=0.1,
                    duration="2m",
                    description="High order failure rate detected",
                    labels={"team": "trading", "severity": "critical"},
                    annotations={
                        "summary": "Order failure rate is {{ $value }} per second",
                        "description": "More than 10% of orders are failing"
                    }
                ),
                AlertRule(
                    name="portfolio_drawdown",
                    query='(trading_portfolio_value - trading_portfolio_value offset 1d) / trading_portfolio_value offset 1d < -0.05',
                    severity=AlertSeverity.CRITICAL,
                    threshold=-0.05,
                    duration="1m",
                    description="Portfolio drawdown exceeds 5%",
                    labels={"team": "risk", "severity": "critical"}
                ),
                AlertRule(
                    name="risk_exposure_high",
                    query='trading_risk_exposure > 80',
                    severity=AlertSeverity.WARNING,
                    threshold=80,
                    duration="5m",
                    description="Risk exposure is high",
                    labels={"team": "risk", "severity": "warning"}
                ),
                
                # System alerts
                AlertRule(
                    name="service_down",
                    query='up == 0',
                    severity=AlertSeverity.CRITICAL,
                    threshold=0,
                    duration="1m",
                    description="Service is down",
                    labels={"team": "sre", "severity": "critical"}
                ),
                AlertRule(
                    name="high_response_time",
                    query='histogram_quantile(0.95, rate(trading_service_request_duration_bucket[5m])) > 1.0',
                    severity=AlertSeverity.WARNING,
                    threshold=1.0,
                    duration="5m",
                    description="High response time detected",
                    labels={"team": "sre", "severity": "warning"}
                ),
                AlertRule(
                    name="high_memory_usage",
                    query='trading_service_memory_usage / 1024 / 1024 / 1024 > 8',
                    severity=AlertSeverity.WARNING,
                    threshold=8,
                    duration="10m",
                    description="High memory usage",
                    labels={"team": "sre", "severity": "warning"}
                ),
                AlertRule(
                    name="high_cpu_usage",
                    query='trading_service_cpu_usage > 80',
                    severity=AlertSeverity.WARNING,
                    threshold=80,
                    duration="10m",
                    description="High CPU usage",
                    labels={"team": "sre", "severity": "warning"}
                ),
                
                # ML alerts
                AlertRule(
                    name="model_accuracy_degradation",
                    query='trading_model_accuracy < 70',
                    severity=AlertSeverity.WARNING,
                    threshold=70,
                    duration="30m",
                    description="Model accuracy below threshold",
                    labels={"team": "ml", "severity": "warning"}
                ),
                AlertRule(
                    name="model_inference_slow",
                    query='histogram_quantile(0.95, rate(trading_model_inference_duration_bucket[5m])) > 5.0',
                    severity=AlertSeverity.WARNING,
                    threshold=5.0,
                    duration="10m",
                    description="Model inference is slow",
                    labels={"team": "ml", "severity": "warning"}
                ),
                
                # Infrastructure alerts
                AlertRule(
                    name="container_restarts",
                    query='increase(trading_container_restarts_total[1h]) > 5',
                    severity=AlertSeverity.WARNING,
                    threshold=5,
                    duration="1m",
                    description="High container restart rate",
                    labels={"team": "sre", "severity": "warning"}
                ),
                AlertRule(
                    name="circuit_breaker_open",
                    query='trading_circuit_breaker_state == 1',
                    severity=AlertSeverity.WARNING,
                    threshold=1,
                    duration="1m",
                    description="Circuit breaker is open",
                    labels={"team": "sre", "severity": "warning"}
                )
            ]
            
            for alert in default_alerts:
                self.alert_rules[alert.name] = alert
                
        except Exception as e:
            self.logger.error(f"Failed to setup default alerts: {e}")
    
    def _setup_default_dashboards(self):
        """Setup default Grafana dashboards."""
        try:
            # Trading Overview Dashboard
            trading_dashboard = Dashboard(
                name="Trading System Overview",
                uid="trading-overview",
                tags=["trading", "overview"],
                panels=[
                    {
                        "title": "Portfolio Value",
                        "type": "stat",
                        "targets": [{"expr": "trading_portfolio_value"}],
                        "fieldConfig": {"defaults": {"unit": "currencyUSD"}}
                    },
                    {
                        "title": "Orders per Second",
                        "type": "stat",
                        "targets": [{"expr": "rate(trading_orders_total[1m])"}]
                    },
                    {
                        "title": "Order Success Rate",
                        "type": "stat",
                        "targets": [{"expr": "rate(trading_orders_total{status=\"success\"}[5m]) / rate(trading_orders_total[5m]) * 100"}],
                        "fieldConfig": {"defaults": {"unit": "percent"}}
                    },
                    {
                        "title": "PnL by Strategy",
                        "type": "timeseries",
                        "targets": [{"expr": "trading_pnl_realized", "legendFormat": "{{strategy}}"}]
                    }
                ]
            )
            
            # System Performance Dashboard
            system_dashboard = Dashboard(
                name="System Performance",
                uid="system-performance",
                tags=["system", "performance"],
                panels=[
                    {
                        "title": "Response Time (95th percentile)",
                        "type": "timeseries",
                        "targets": [{"expr": "histogram_quantile(0.95, rate(trading_service_request_duration_bucket[5m]))"}]
                    },
                    {
                        "title": "Request Rate",
                        "type": "timeseries",
                        "targets": [{"expr": "rate(trading_service_requests_total[1m])"}]
                    },
                    {
                        "title": "Memory Usage",
                        "type": "timeseries",
                        "targets": [{"expr": "trading_service_memory_usage / 1024 / 1024 / 1024"}],
                        "fieldConfig": {"defaults": {"unit": "gbytes"}}
                    },
                    {
                        "title": "CPU Usage",
                        "type": "timeseries",
                        "targets": [{"expr": "trading_service_cpu_usage"}],
                        "fieldConfig": {"defaults": {"unit": "percent"}}
                    }
                ]
            )
            
            # ML Models Dashboard
            ml_dashboard = Dashboard(
                name="ML Models Performance",
                uid="ml-models",
                tags=["ml", "models"],
                panels=[
                    {
                        "title": "Model Accuracy",
                        "type": "timeseries",
                        "targets": [{"expr": "trading_model_accuracy", "legendFormat": "{{model_name}}"}]
                    },
                    {
                        "title": "Inference Duration",
                        "type": "timeseries",
                        "targets": [{"expr": "histogram_quantile(0.95, rate(trading_model_inference_duration_bucket[5m]))"}]
                    },
                    {
                        "title": "Predictions per Minute",
                        "type": "stat",
                        "targets": [{"expr": "rate(trading_model_predictions_total[1m]) * 60"}]
                    }
                ]
            )
            
            self.dashboards["trading-overview"] = trading_dashboard
            self.dashboards["system-performance"] = system_dashboard
            self.dashboards["ml-models"] = ml_dashboard
            
        except Exception as e:
            self.logger.error(f"Failed to setup default dashboards: {e}")
    
    def _initialize_elasticsearch(self):
        """Initialize Elasticsearch client."""
        try:
            if HAS_ELASTICSEARCH:
                es_config = self.monitoring_config['elasticsearch']
                self.elasticsearch_client = Elasticsearch([{
                    'host': es_config['host'],
                    'port': es_config['port']
                }])
                
                # Test connection
                if self.elasticsearch_client.ping():
                    self.logger.info("Connected to Elasticsearch")
                else:
                    self.logger.warning("Failed to connect to Elasticsearch")
                    self.elasticsearch_client = None
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize Elasticsearch: {e}")
            self.elasticsearch_client = None
    
    async def start_monitoring(self):
        """Start monitoring stack."""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            
            # Start metrics collection
            self.metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
            
            # Start alert evaluation
            self.alert_evaluation_task = asyncio.create_task(self._alert_evaluation_loop())
            
            # Start log shipping
            if self.elasticsearch_client:
                self.log_shipping_task = asyncio.create_task(self._log_shipping_loop())
            
            self.logger.info("Monitoring stack started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring stack: {e}")
    
    async def stop_monitoring(self):
        """Stop monitoring stack."""
        try:
            self.monitoring_active = False
            
            # Cancel tasks
            tasks = [self.metrics_collection_task, self.alert_evaluation_task, self.log_shipping_task]
            for task in tasks:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.logger.info("Monitoring stack stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring stack: {e}")
    
    async def _metrics_collection_loop(self):
        """Metrics collection and export loop."""
        try:
            while self.monitoring_active:
                # Push metrics to Pushgateway
                if HAS_PROMETHEUS and self.monitoring_config['prometheus']['enabled']:
                    try:
                        pushgateway_url = self.monitoring_config['prometheus']['pushgateway_url']
                        push_to_gateway(
                            pushgateway_url,
                            job='trading-system',
                            registry=self.metrics_registry
                        )
                    except Exception as e:
                        self.logger.debug(f"Failed to push metrics: {e}")
                
                await asyncio.sleep(15)  # Push every 15 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Metrics collection loop error: {e}")
    
    async def _alert_evaluation_loop(self):
        """Alert evaluation loop."""
        try:
            while self.monitoring_active:
                await self._evaluate_alerts()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Alert evaluation loop error: {e}")
    
    async def _log_shipping_loop(self):
        """Log shipping to Elasticsearch loop."""
        try:
            while self.monitoring_active:
                if self.log_buffer and self.elasticsearch_client:
                    await self._ship_logs()
                
                await asyncio.sleep(10)  # Ship logs every 10 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Log shipping loop error: {e}")
    
    def record_metric(self, metric_name: str, value: Union[float, int], labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        try:
            if metric_name not in self.prometheus_metrics:
                return
            
            metric = self.prometheus_metrics[metric_name]
            labels = labels or {}
            
            if isinstance(metric, Counter):
                metric.labels(**labels).inc(value)
            elif isinstance(metric, Gauge):
                metric.labels(**labels).set(value)
            elif isinstance(metric, Histogram):
                metric.labels(**labels).observe(value)
            elif isinstance(metric, Summary):
                metric.labels(**labels).observe(value)
                
        except Exception as e:
            self.logger.error(f"Failed to record metric {metric_name}: {e}")
    
    def start_timing(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> Callable:
        """Start timing for a histogram metric."""
        try:
            if metric_name not in self.prometheus_metrics:
                return lambda: None
            
            metric = self.prometheus_metrics[metric_name]
            if isinstance(metric, (Histogram, Summary)):
                labels = labels or {}
                return metric.labels(**labels).time()
            
            return lambda: None
            
        except Exception as e:
            self.logger.error(f"Failed to start timing for {metric_name}: {e}")
            return lambda: None
    
    def add_log_entry(self, level: str, service: str, message: str, 
                     trace_id: Optional[str] = None, **fields):
        """Add log entry to buffer."""
        try:
            log_entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                service=service,
                message=message,
                trace_id=trace_id,
                fields=fields
            )
            
            self.log_buffer.append(log_entry)
            
            # Keep buffer size manageable
            if len(self.log_buffer) > 10000:
                self.log_buffer = self.log_buffer[-5000:]
                
        except Exception as e:
            self.logger.error(f"Failed to add log entry: {e}")
    
    async def _ship_logs(self):
        """Ship logs to Elasticsearch."""
        try:
            if not self.elasticsearch_client or not self.log_buffer:
                return
            
            # Prepare bulk index operations
            actions = []
            for log_entry in self.log_buffer:
                doc = {
                    '@timestamp': log_entry.timestamp.isoformat(),
                    'level': log_entry.level,
                    'service': log_entry.service,
                    'message': log_entry.message,
                    'trace_id': log_entry.trace_id,
                    **log_entry.fields
                }
                
                actions.append({
                    '_index': f"trading-logs-{datetime.now().strftime('%Y.%m.%d')}",
                    '_source': doc
                })
            
            # Bulk index to Elasticsearch
            if actions:
                from elasticsearch.helpers import bulk
                bulk(self.elasticsearch_client, actions)
                self.log_buffer.clear()
                
        except Exception as e:
            self.logger.error(f"Failed to ship logs: {e}")
    
    async def _evaluate_alerts(self):
        """Evaluate alert rules."""
        try:
            for rule_name, rule in self.alert_rules.items():
                # Simulate alert evaluation (would use actual Prometheus queries)
                is_firing = await self._evaluate_alert_rule(rule)
                
                if is_firing and rule_name not in self.active_alerts:
                    # Fire alert
                    alert = AlertInstance(
                        alert_id=f"{rule_name}-{int(time.time())}",
                        rule_name=rule_name,
                        severity=rule.severity,
                        message=rule.description,
                        labels=rule.labels,
                        started_at=datetime.now()
                    )
                    
                    self.active_alerts[rule_name] = alert
                    await self._send_alert_notification(alert)
                    
                elif not is_firing and rule_name in self.active_alerts:
                    # Resolve alert
                    alert = self.active_alerts[rule_name]
                    alert.resolved_at = datetime.now()
                    
                    self.alert_history.append(alert)
                    del self.active_alerts[rule_name]
                    
                    await self._send_alert_resolution(alert)
                    
        except Exception as e:
            self.logger.error(f"Failed to evaluate alerts: {e}")
    
    async def _evaluate_alert_rule(self, rule: AlertRule) -> bool:
        """Evaluate a single alert rule."""
        try:
            # This would normally query Prometheus
            # For now, simulate based on rule name
            
            if "high_order_failure_rate" in rule.name:
                # Simulate order failure rate check
                return False  # No failures for now
            elif "portfolio_drawdown" in rule.name:
                # Simulate portfolio check
                return False  # No drawdown for now
            elif "service_down" in rule.name:
                # Simulate service health check
                return False  # All services up
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to evaluate alert rule {rule.name}: {e}")
            return False
    
    async def _send_alert_notification(self, alert: AlertInstance):
        """Send alert notification."""
        try:
            self.logger.warning(f"ALERT FIRED: {alert.rule_name} - {alert.message}")
            
            # Would integrate with notification systems (email, Slack, PagerDuty)
            alert.notification_sent = True
            
        except Exception as e:
            self.logger.error(f"Failed to send alert notification: {e}")
    
    async def _send_alert_resolution(self, alert: AlertInstance):
        """Send alert resolution notification."""
        try:
            duration = (alert.resolved_at - alert.started_at).total_seconds() if alert.resolved_at else 0
            self.logger.info(f"ALERT RESOLVED: {alert.rule_name} - Duration: {duration:.1f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert resolution: {e}")
    
    def start_trace(self, operation_name: str, service_name: str) -> str:
        """Start a distributed trace."""
        try:
            trace_id = hashlib.md5(f"{time.time()}{operation_name}".encode()).hexdigest()
            span_id = hashlib.md5(f"{trace_id}{service_name}".encode()).hexdigest()[:16]
            
            span = TraceSpan(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=None,
                operation_name=operation_name,
                service_name=service_name,
                start_time=datetime.now()
            )
            
            if trace_id not in self.active_traces:
                self.active_traces[trace_id] = []
            
            self.active_traces[trace_id].append(span)
            
            return trace_id
            
        except Exception as e:
            self.logger.error(f"Failed to start trace: {e}")
            return ""
    
    def finish_trace(self, trace_id: str, tags: Optional[Dict[str, str]] = None):
        """Finish a distributed trace."""
        try:
            if trace_id not in self.active_traces:
                return
            
            spans = self.active_traces[trace_id]
            for span in spans:
                if not span.end_time:
                    span.end_time = datetime.now()
                    span.duration = (span.end_time - span.start_time).total_seconds()
                    if tags:
                        span.tags.update(tags)
            
            # Move to completed traces
            self.completed_traces.append(spans)
            del self.active_traces[trace_id]
            
            # Keep completed traces manageable
            if len(self.completed_traces) > 1000:
                self.completed_traces = self.completed_traces[-500:]
                
        except Exception as e:
            self.logger.error(f"Failed to finish trace: {e}")
    
    def get_dashboard_config(self, dashboard_uid: str) -> Optional[Dict[str, Any]]:
        """Get Grafana dashboard configuration."""
        try:
            if dashboard_uid not in self.dashboards:
                return None
            
            dashboard = self.dashboards[dashboard_uid]
            
            return {
                "dashboard": {
                    "id": None,
                    "uid": dashboard.uid,
                    "title": dashboard.name,
                    "tags": dashboard.tags,
                    "timezone": "browser",
                    "panels": dashboard.panels,
                    "time": {
                        "from": f"now-{dashboard.time_range}",
                        "to": "now"
                    },
                    "refresh": dashboard.refresh_interval,
                    "templating": {
                        "list": dashboard.variables
                    },
                    "version": 1
                },
                "overwrite": True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard config: {e}")
            return None
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring stack summary."""
        try:
            active_alerts_by_severity = {}
            for alert in self.active_alerts.values():
                severity = alert.severity.value
                active_alerts_by_severity[severity] = active_alerts_by_severity.get(severity, 0) + 1
            
            return {
                'monitoring_active': self.monitoring_active,
                'components': {
                    'prometheus': self.monitoring_config['prometheus']['enabled'],
                    'grafana': self.monitoring_config['grafana']['enabled'],
                    'alertmanager': self.monitoring_config['alertmanager']['enabled'],
                    'elasticsearch': self.monitoring_config['elasticsearch']['enabled'] and self.elasticsearch_client is not None,
                    'tracing': self.monitoring_config['tracing']['enabled']
                },
                'metrics': {
                    'total_metrics': len(self.prometheus_metrics),
                    'custom_metrics_available': HAS_PROMETHEUS
                },
                'alerts': {
                    'total_rules': len(self.alert_rules),
                    'active_alerts': len(self.active_alerts),
                    'alerts_by_severity': active_alerts_by_severity,
                    'alert_history_count': len(self.alert_history)
                },
                'dashboards': {
                    'total_dashboards': len(self.dashboards),
                    'available_dashboards': list(self.dashboards.keys())
                },
                'logs': {
                    'buffer_size': len(self.log_buffer),
                    'elasticsearch_connected': self.elasticsearch_client is not None
                },
                'tracing': {
                    'active_traces': len(self.active_traces),
                    'completed_traces': len(self.completed_traces)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate monitoring summary: {e}")
            return {'error': 'Unable to generate summary'}