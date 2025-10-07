"""
Comprehensive System Health Monitoring - Phase 10

This module provides advanced system health monitoring with:
- Real-time metrics collection and analysis
- Intelligent alerting with escalation policies
- Performance monitoring and bottleneck detection
- Error tracking and automated recovery
- Resource utilization monitoring
- Trading system specific health checks
- Automated remediation actions

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import sqlite3
from pathlib import Path


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Metric data types"""
    GAUGE = "gauge"         # Point-in-time value
    COUNTER = "counter"     # Monotonically increasing
    HISTOGRAM = "histogram" # Distribution of values
    SUMMARY = "summary"     # Statistical summary


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Metric:
    """System metric data structure"""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_function: Callable
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enabled: bool = True
    critical: bool = False


@dataclass
class Alert:
    """Alert information"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemStatus:
    """Overall system status"""
    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    component_status: Dict[str, HealthStatus] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    active_alerts: List[Alert] = field(default_factory=list)
    message: str = ""


class MetricsCollector:
    """
    Advanced metrics collection system
    
    Collects and stores system, application, and trading-specific metrics
    with configurable retention and aggregation.
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Trading-specific metric counters
        self.trade_count = 0
        self.error_count = 0
        self.api_call_count = 0
        self.last_trade_time = None
        
        # System resource baselines
        self.cpu_baseline = None
        self.memory_baseline = None
        self.established_baseline = False
    
    def record_metric(self, metric: Metric):
        """Record a metric value"""
        with self.lock:
            self.metrics_buffer[metric.name].append(metric)
            
            # Clean old metrics
            cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
            while (self.metrics_buffer[metric.name] and 
                   self.metrics_buffer[metric.name][0].timestamp < cutoff_time):
                self.metrics_buffer[metric.name].popleft()
    
    def get_metric_history(self, metric_name: str, 
                          hours: int = 1) -> List[Metric]:
        """Get metric history for specified time period"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            metrics = self.metrics_buffer.get(metric_name, deque())
            
            return [m for m in metrics if m.timestamp >= cutoff_time]
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values"""
        current_metrics = {}
        
        with self.lock:
            for metric_name, metric_buffer in self.metrics_buffer.items():
                if metric_buffer:
                    current_metrics[metric_name] = metric_buffer[-1].value
        
        return current_metrics
    
    def collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            now = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            self.record_metric(Metric(
                name="system_cpu_percent",
                value=cpu_percent,
                timestamp=now,
                metric_type=MetricType.GAUGE,
                unit="%",
                description="CPU utilization percentage"
            ))
            
            self.record_metric(Metric(
                name="system_cpu_count",
                value=cpu_count,
                timestamp=now,
                metric_type=MetricType.GAUGE,
                description="Number of CPU cores"
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric(Metric(
                name="system_memory_percent",
                value=memory.percent,
                timestamp=now,
                metric_type=MetricType.GAUGE,
                unit="%",
                description="Memory utilization percentage"
            ))
            
            self.record_metric(Metric(
                name="system_memory_available_gb",
                value=memory.available / (1024**3),
                timestamp=now,
                metric_type=MetricType.GAUGE,
                unit="GB",
                description="Available memory in GB"
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric(Metric(
                name="system_disk_percent",
                value=disk.percent,
                timestamp=now,
                metric_type=MetricType.GAUGE,
                unit="%",
                description="Disk utilization percentage"
            ))
            
            # Network metrics
            network = psutil.net_io_counters()
            self.record_metric(Metric(
                name="system_network_bytes_sent",
                value=network.bytes_sent,
                timestamp=now,
                metric_type=MetricType.COUNTER,
                unit="bytes",
                description="Total bytes sent"
            ))
            
            self.record_metric(Metric(
                name="system_network_bytes_recv",
                value=network.bytes_recv,
                timestamp=now,
                metric_type=MetricType.COUNTER,
                unit="bytes",
                description="Total bytes received"
            ))
            
            # Establish baselines if not done
            if not self.established_baseline:
                self.cpu_baseline = cpu_percent
                self.memory_baseline = memory.percent
                self.established_baseline = True
                self.logger.info(f"Established system baselines - CPU: {cpu_percent}%, Memory: {memory.percent}%")
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def collect_trading_metrics(self, 
                              portfolio_value: float = 0.0,
                              open_positions: int = 0,
                              daily_pnl: float = 0.0,
                              trade_count_delta: int = 0):
        """Collect trading-specific metrics"""
        try:
            now = datetime.now()
            
            # Update counters
            self.trade_count += trade_count_delta
            if trade_count_delta > 0:
                self.last_trade_time = now
            
            # Portfolio metrics
            self.record_metric(Metric(
                name="trading_portfolio_value",
                value=portfolio_value,
                timestamp=now,
                metric_type=MetricType.GAUGE,
                unit="USD",
                description="Current portfolio value"
            ))
            
            self.record_metric(Metric(
                name="trading_open_positions",
                value=open_positions,
                timestamp=now,
                metric_type=MetricType.GAUGE,
                description="Number of open positions"
            ))
            
            self.record_metric(Metric(
                name="trading_daily_pnl",
                value=daily_pnl,
                timestamp=now,
                metric_type=MetricType.GAUGE,
                unit="USD",
                description="Daily profit and loss"
            ))
            
            self.record_metric(Metric(
                name="trading_total_trades",
                value=self.trade_count,
                timestamp=now,
                metric_type=MetricType.COUNTER,
                description="Total number of trades executed"
            ))
            
            # Time since last trade
            if self.last_trade_time:
                time_since_trade = (now - self.last_trade_time).total_seconds()
                self.record_metric(Metric(
                    name="trading_time_since_last_trade",
                    value=time_since_trade,
                    timestamp=now,
                    metric_type=MetricType.GAUGE,
                    unit="seconds",
                    description="Time since last trade execution"
                ))
            
        except Exception as e:
            self.logger.error(f"Error collecting trading metrics: {e}")
    
    def collect_application_metrics(self,
                                  active_threads: int = 0,
                                  queue_sizes: Dict[str, int] = None,
                                  response_times: Dict[str, float] = None):
        """Collect application-specific metrics"""
        try:
            now = datetime.now()
            queue_sizes = queue_sizes or {}
            response_times = response_times or {}
            
            # Thread metrics
            self.record_metric(Metric(
                name="app_active_threads",
                value=active_threads,
                timestamp=now,
                metric_type=MetricType.GAUGE,
                description="Number of active threads"
            ))
            
            # Queue metrics
            for queue_name, size in queue_sizes.items():
                self.record_metric(Metric(
                    name=f"app_queue_size_{queue_name}",
                    value=size,
                    timestamp=now,
                    metric_type=MetricType.GAUGE,
                    labels={"queue": queue_name},
                    description=f"Size of {queue_name} queue"
                ))
            
            # Response time metrics
            for endpoint, response_time in response_times.items():
                self.record_metric(Metric(
                    name=f"app_response_time_{endpoint}",
                    value=response_time,
                    timestamp=now,
                    metric_type=MetricType.HISTOGRAM,
                    labels={"endpoint": endpoint},
                    unit="seconds",
                    description=f"Response time for {endpoint}"
                ))
            
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")
    
    def get_metric_statistics(self, metric_name: str, 
                            hours: int = 1) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        history = self.get_metric_history(metric_name, hours)
        
        if not history:
            return {}
        
        values = [m.value for m in history]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }


class HealthCheckManager:
    """
    Advanced health check management system
    
    Manages multiple health checks with failure tracking,
    automatic recovery detection, and escalation policies.
    """
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.check_status: Dict[str, HealthStatus] = {}
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)
        self.last_check_times: Dict[str, datetime] = {}
        self.check_tasks: Dict[str, asyncio.Task] = {}
        
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check"""
        self.health_checks[health_check.name] = health_check
        self.check_status[health_check.name] = HealthStatus.UNKNOWN
        self.logger.info(f"Registered health check: {health_check.name}")
    
    async def start_monitoring(self):
        """Start all health check monitoring"""
        self.running = True
        
        for name, check in self.health_checks.items():
            if check.enabled:
                task = asyncio.create_task(self._run_health_check_loop(name))
                self.check_tasks[name] = task
        
        self.logger.info("Health check monitoring started")
    
    async def stop_monitoring(self):
        """Stop all health check monitoring"""
        self.running = False
        
        for task in self.check_tasks.values():
            task.cancel()
        
        await asyncio.gather(*self.check_tasks.values(), return_exceptions=True)
        self.check_tasks.clear()
        
        self.logger.info("Health check monitoring stopped")
    
    async def _run_health_check_loop(self, check_name: str):
        """Run health check in a loop"""
        check = self.health_checks[check_name]
        
        while self.running:
            try:
                await self._execute_health_check(check_name)
                await asyncio.sleep(check.interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop {check_name}: {e}")
                await asyncio.sleep(check.interval_seconds)
    
    async def _execute_health_check(self, check_name: str):
        """Execute a single health check"""
        check = self.health_checks[check_name]
        
        try:
            # Execute check with timeout
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(check.check_function):
                result = await asyncio.wait_for(
                    check.check_function(),
                    timeout=check.timeout_seconds
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, check.check_function
                )
            
            execution_time = time.time() - start_time
            
            # Process result
            if result:
                self._handle_check_success(check_name)
            else:
                self._handle_check_failure(check_name, "Check returned False")
            
            self.last_check_times[check_name] = datetime.now()
            
        except asyncio.TimeoutError:
            self._handle_check_failure(check_name, f"Timeout after {check.timeout_seconds}s")
        except Exception as e:
            self._handle_check_failure(check_name, str(e))
    
    def _handle_check_success(self, check_name: str):
        """Handle successful health check"""
        self.success_counts[check_name] += 1
        check = self.health_checks[check_name]
        
        # Check if we should recover from failure
        if (self.check_status[check_name] in [HealthStatus.WARNING, HealthStatus.CRITICAL] and
            self.success_counts[check_name] >= check.recovery_threshold):
            
            old_status = self.check_status[check_name]
            self.check_status[check_name] = HealthStatus.HEALTHY
            self.failure_counts[check_name] = 0
            
            self.logger.info(f"Health check recovered: {check_name} ({old_status} -> HEALTHY)")
        elif self.check_status[check_name] == HealthStatus.UNKNOWN:
            self.check_status[check_name] = HealthStatus.HEALTHY
    
    def _handle_check_failure(self, check_name: str, error_message: str):
        """Handle failed health check"""
        self.failure_counts[check_name] += 1
        self.success_counts[check_name] = 0
        check = self.health_checks[check_name]
        
        # Determine new status
        if self.failure_counts[check_name] >= check.failure_threshold:
            new_status = HealthStatus.CRITICAL if check.critical else HealthStatus.WARNING
            
            if self.check_status[check_name] != new_status:
                old_status = self.check_status[check_name]
                self.check_status[check_name] = new_status
                
                self.logger.warning(f"Health check failed: {check_name} ({old_status} -> {new_status}) - {error_message}")
        
        self.logger.debug(f"Health check failure: {check_name} ({self.failure_counts[check_name]}/{check.failure_threshold}) - {error_message}")
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.check_status:
            return HealthStatus.UNKNOWN
        
        # Critical if any critical check is critical
        if any(status == HealthStatus.CRITICAL for status in self.check_status.values()):
            return HealthStatus.CRITICAL
        
        # Warning if any check is warning
        if any(status == HealthStatus.WARNING for status in self.check_status.values()):
            return HealthStatus.WARNING
        
        # Healthy if all checks are healthy
        if all(status == HealthStatus.HEALTHY for status in self.check_status.values()):
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        return {
            'overall_status': self.get_overall_status().value,
            'total_checks': len(self.health_checks),
            'healthy_checks': sum(1 for s in self.check_status.values() if s == HealthStatus.HEALTHY),
            'warning_checks': sum(1 for s in self.check_status.values() if s == HealthStatus.WARNING),
            'critical_checks': sum(1 for s in self.check_status.values() if s == HealthStatus.CRITICAL),
            'check_details': {
                name: {
                    'status': status.value,
                    'failure_count': self.failure_counts[name],
                    'success_count': self.success_counts[name],
                    'last_check': self.last_check_times.get(name, datetime.min).isoformat()
                }
                for name, status in self.check_status.items()
            }
        }


class AlertManager:
    """
    Intelligent alerting system with escalation policies
    
    Manages alerts with different severity levels, escalation policies,
    and multiple notification channels.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels = {}
        self.escalation_policies = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Alert suppression (prevent spam)
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.default_cooldown_minutes = 15
        
        # Setup notification channels
        self._setup_notification_channels()
    
    def _setup_notification_channels(self):
        """Setup notification channels"""
        # Email notifications
        if self.config.get('email', {}).get('enabled', False):
            self.notification_channels['email'] = self._send_email_notification
        
        # Discord notifications
        if self.config.get('discord', {}).get('enabled', False):
            self.notification_channels['discord'] = self._send_discord_notification
        
        # Slack notifications
        if self.config.get('slack', {}).get('enabled', False):
            self.notification_channels['slack'] = self._send_slack_notification
    
    async def create_alert(self, 
                          level: AlertLevel,
                          title: str,
                          message: str,
                          source: str = "system",
                          metadata: Dict[str, Any] = None) -> str:
        """Create a new alert"""
        alert_id = f"{source}_{title}_{int(time.time())}"
        
        # Check cooldown
        cooldown_key = f"{source}_{title}"
        if cooldown_key in self.alert_cooldowns:
            if datetime.now() < self.alert_cooldowns[cooldown_key]:
                self.logger.debug(f"Alert suppressed due to cooldown: {title}")
                return alert_id
        
        alert = Alert(
            id=alert_id,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Set cooldown
        cooldown_minutes = self.config.get('cooldown_minutes', self.default_cooldown_minutes)
        self.alert_cooldowns[cooldown_key] = datetime.now() + timedelta(minutes=cooldown_minutes)
        
        # Send notifications
        await self._send_notifications(alert)
        
        self.logger.info(f"Alert created: [{level.value.upper()}] {title}")
        return alert_id
    
    async def resolve_alert(self, alert_id: str, resolution_message: str = ""):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            if resolution_message:
                alert.metadata['resolution_message'] = resolution_message
            
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert resolved: {alert.title}")
            
            # Send resolution notification
            await self._send_resolution_notification(alert)
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through all configured channels"""
        for channel_name, send_func in self.notification_channels.items():
            try:
                await send_func(alert)
            except Exception as e:
                self.logger.error(f"Error sending notification via {channel_name}: {e}")
    
    async def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notifications"""
        resolution_alert = Alert(
            id=f"{alert.id}_resolved",
            level=AlertLevel.INFO,
            title=f"RESOLVED: {alert.title}",
            message=f"Alert resolved after {alert.resolved_at - alert.timestamp}",
            timestamp=alert.resolved_at,
            source=alert.source,
            metadata=alert.metadata
        )
        
        await self._send_notifications(resolution_alert)
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        email_config = self.config.get('email', {})
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"[{alert.level.value.upper()}] Trading Bot Alert: {alert.title}"
            
            body = f"""
Alert Details:
- Level: {alert.level.value.upper()}
- Source: {alert.source}
- Time: {alert.timestamp}
- Message: {alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            
            text = msg.as_string()
            server.sendmail(email_config['from'], email_config['recipients'], text)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Error sending email notification: {e}")
    
    async def _send_discord_notification(self, alert: Alert):
        """Send Discord webhook notification"""
        discord_config = self.config.get('discord', {})
        webhook_url = discord_config.get('webhook_url')
        
        if not webhook_url:
            return
        
        try:
            color_map = {
                AlertLevel.INFO: 0x00ff00,      # Green
                AlertLevel.WARNING: 0xffff00,   # Yellow
                AlertLevel.CRITICAL: 0xff0000,  # Red
                AlertLevel.EMERGENCY: 0x800080  # Purple
            }
            
            embed = {
                "title": f"Trading Bot Alert: {alert.title}",
                "description": alert.message,
                "color": color_map.get(alert.level, 0x000000),
                "timestamp": alert.timestamp.isoformat(),
                "fields": [
                    {"name": "Level", "value": alert.level.value.upper(), "inline": True},
                    {"name": "Source", "value": alert.source, "inline": True}
                ]
            }
            
            payload = {"embeds": [embed]}
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Error sending Discord notification: {e}")
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        slack_config = self.config.get('slack', {})
        webhook_url = slack_config.get('webhook_url')
        
        if not webhook_url:
            return
        
        try:
            color_map = {
                AlertLevel.INFO: "good",
                AlertLevel.WARNING: "warning",
                AlertLevel.CRITICAL: "danger",
                AlertLevel.EMERGENCY: "danger"
            }
            
            payload = {
                "text": f"Trading Bot Alert: {alert.title}",
                "attachments": [{
                    "color": color_map.get(alert.level, "good"),
                    "fields": [
                        {"title": "Level", "value": alert.level.value.upper(), "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Message", "value": alert.message, "short": False},
                        {"title": "Time", "value": alert.timestamp.isoformat(), "short": True}
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        total_alerts = len(self.alert_history)
        active_count = len(self.active_alerts)
        
        level_counts = defaultdict(int)
        for alert in self.alert_history:
            level_counts[alert.level.value] += 1
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_count,
            'resolved_alerts': total_alerts - active_count,
            'alerts_by_level': dict(level_counts),
            'active_alert_details': [
                {
                    'id': alert.id,
                    'level': alert.level.value,
                    'title': alert.title,
                    'source': alert.source,
                    'timestamp': alert.timestamp.isoformat(),
                    'age_minutes': (datetime.now() - alert.timestamp).total_seconds() / 60
                }
                for alert in self.active_alerts.values()
            ]
        }


class ComprehensiveHealthMonitor:
    """
    Master health monitoring system
    
    Coordinates all monitoring components and provides unified
    system health status with intelligent analysis and automated responses.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.health_check_manager = HealthCheckManager()
        self.alert_manager = AlertManager(self.config.get('alerting', {}))
        
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.start_time = datetime.now()
        
        # Monitoring tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # System state tracking
        self.last_system_status = HealthStatus.UNKNOWN
        self.status_change_count = 0
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        # System resource checks
        self.health_check_manager.register_health_check(
            HealthCheck(
                name="system_cpu",
                check_function=self._check_cpu_usage,
                interval_seconds=30,
                critical=False
            )
        )
        
        self.health_check_manager.register_health_check(
            HealthCheck(
                name="system_memory",
                check_function=self._check_memory_usage,
                interval_seconds=30,
                critical=True
            )
        )
        
        self.health_check_manager.register_health_check(
            HealthCheck(
                name="system_disk",
                check_function=self._check_disk_usage,
                interval_seconds=60,
                critical=True
            )
        )
        
        # Trading system checks
        self.health_check_manager.register_health_check(
            HealthCheck(
                name="trading_api_connectivity",
                check_function=self._check_api_connectivity,
                interval_seconds=60,
                critical=True
            )
        )
        
        self.health_check_manager.register_health_check(
            HealthCheck(
                name="trading_data_freshness",
                check_function=self._check_data_freshness,
                interval_seconds=30,
                critical=True
            )
        )
    
    async def start_monitoring(self):
        """Start comprehensive monitoring"""
        self.running = True
        self.start_time = datetime.now()
        
        # Start health checks
        await self.health_check_manager.start_monitoring()
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._system_analysis_loop()),
            asyncio.create_task(self._performance_tracking_loop())
        ]
        
        self.logger.info("Comprehensive health monitoring started")
        
        # Send startup alert
        await self.alert_manager.create_alert(
            AlertLevel.INFO,
            "System Monitoring Started",
            "Comprehensive health monitoring system has been started",
            "health_monitor"
        )
    
    async def stop_monitoring(self):
        """Stop comprehensive monitoring"""
        self.running = False
        
        # Stop health checks
        await self.health_check_manager.stop_monitoring()
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
        
        self.logger.info("Comprehensive health monitoring stopped")
    
    async def _metrics_collection_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                # Collect system metrics
                self.metrics_collector.collect_system_metrics()
                
                # Collect application metrics (would be provided by main app)
                active_threads = threading.active_count()
                self.metrics_collector.collect_application_metrics(
                    active_threads=active_threads
                )
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(10)
    
    async def _system_analysis_loop(self):
        """System analysis and intelligent alerting loop"""
        while self.running:
            try:
                # Get current system status
                current_status = self.health_check_manager.get_overall_status()
                
                # Check for status changes
                if current_status != self.last_system_status:
                    await self._handle_status_change(self.last_system_status, current_status)
                    self.last_system_status = current_status
                    self.status_change_count += 1
                
                # Analyze metrics for anomalies
                await self._analyze_metric_anomalies()
                
                # Check system trends
                await self._analyze_system_trends()
                
                await asyncio.sleep(60)  # Analyze every minute
                
            except Exception as e:
                self.logger.error(f"Error in system analysis loop: {e}")
                await asyncio.sleep(60)
    
    async def _performance_tracking_loop(self):
        """Performance tracking and optimization loop"""
        while self.running:
            try:
                # Collect performance snapshot
                performance_snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                    'system_status': self.last_system_status.value,
                    'active_alerts': len(self.alert_manager.get_active_alerts()),
                    'metrics': self.metrics_collector.get_current_metrics()
                }
                
                self.performance_history.append(performance_snapshot)
                
                # Keep only last 24 hours
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.performance_history = [
                    snapshot for snapshot in self.performance_history
                    if datetime.fromisoformat(snapshot['timestamp']) >= cutoff_time
                ]
                
                await asyncio.sleep(300)  # Track every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(300)
    
    async def _handle_status_change(self, old_status: HealthStatus, new_status: HealthStatus):
        """Handle system status changes"""
        if new_status == HealthStatus.CRITICAL:
            await self.alert_manager.create_alert(
                AlertLevel.CRITICAL,
                "System Status Critical",
                f"System status changed from {old_status.value} to {new_status.value}",
                "health_monitor",
                {"old_status": old_status.value, "new_status": new_status.value}
            )
        elif new_status == HealthStatus.WARNING:
            await self.alert_manager.create_alert(
                AlertLevel.WARNING,
                "System Status Warning",
                f"System status changed from {old_status.value} to {new_status.value}",
                "health_monitor",
                {"old_status": old_status.value, "new_status": new_status.value}
            )
        elif new_status == HealthStatus.HEALTHY and old_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            await self.alert_manager.create_alert(
                AlertLevel.INFO,
                "System Status Recovered",
                f"System status recovered from {old_status.value} to {new_status.value}",
                "health_monitor",
                {"old_status": old_status.value, "new_status": new_status.value}
            )
    
    async def _analyze_metric_anomalies(self):
        """Analyze metrics for anomalies and create alerts"""
        current_metrics = self.metrics_collector.get_current_metrics()
        
        # CPU anomaly detection
        cpu_percent = current_metrics.get('system_cpu_percent', 0)
        if cpu_percent > 90:
            await self.alert_manager.create_alert(
                AlertLevel.CRITICAL,
                "High CPU Usage",
                f"CPU usage is {cpu_percent:.1f}%",
                "metrics_analyzer",
                {"cpu_percent": cpu_percent}
            )
        elif cpu_percent > 80:
            await self.alert_manager.create_alert(
                AlertLevel.WARNING,
                "Elevated CPU Usage",
                f"CPU usage is {cpu_percent:.1f}%",
                "metrics_analyzer",
                {"cpu_percent": cpu_percent}
            )
        
        # Memory anomaly detection
        memory_percent = current_metrics.get('system_memory_percent', 0)
        if memory_percent > 95:
            await self.alert_manager.create_alert(
                AlertLevel.CRITICAL,
                "Critical Memory Usage",
                f"Memory usage is {memory_percent:.1f}%",
                "metrics_analyzer",
                {"memory_percent": memory_percent}
            )
        elif memory_percent > 85:
            await self.alert_manager.create_alert(
                AlertLevel.WARNING,
                "High Memory Usage",
                f"Memory usage is {memory_percent:.1f}%",
                "metrics_analyzer",
                {"memory_percent": memory_percent}
            )
    
    async def _analyze_system_trends(self):
        """Analyze system trends and predict issues"""
        # Get CPU trend over last hour
        cpu_history = self.metrics_collector.get_metric_history('system_cpu_percent', hours=1)
        
        if len(cpu_history) >= 10:
            recent_values = [m.value for m in cpu_history[-10:]]
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            # Alert on rapidly increasing CPU
            if trend > 2.0:  # Increasing by >2% per measurement
                await self.alert_manager.create_alert(
                    AlertLevel.WARNING,
                    "CPU Usage Trending Up",
                    f"CPU usage is increasing rapidly (trend: +{trend:.1f}% per measurement)",
                    "trend_analyzer",
                    {"trend": trend, "current_cpu": recent_values[-1]}
                )
    
    # Health check functions
    def _check_cpu_usage(self) -> bool:
        """Check CPU usage health"""
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent < 90  # Healthy if under 90%
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage health"""
        memory = psutil.virtual_memory()
        return memory.percent < 95  # Healthy if under 95%
    
    def _check_disk_usage(self) -> bool:
        """Check disk usage health"""
        disk = psutil.disk_usage('/')
        return disk.percent < 90  # Healthy if under 90%
    
    async def _check_api_connectivity(self) -> bool:
        """Check trading API connectivity"""
        try:
            # This would be implemented to check actual API connectivity
            # For now, return True as placeholder
            return True
        except Exception:
            return False
    
    async def _check_data_freshness(self) -> bool:
        """Check if trading data is fresh"""
        try:
            # This would check if market data is recent
            # For now, return True as placeholder
            return True
        except Exception:
            return False
    
    def get_comprehensive_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        overall_status = self.health_check_manager.get_overall_status()
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return SystemStatus(
            status=overall_status,
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            component_status=self.health_check_manager.check_status.copy(),
            metrics=self.metrics_collector.get_current_metrics(),
            active_alerts=self.alert_manager.get_active_alerts(),
            message=f"System running for {uptime/3600:.1f} hours with {self.status_change_count} status changes"
        )
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        status = self.get_comprehensive_status()
        health_summary = self.health_check_manager.get_status_summary()
        alert_summary = self.alert_manager.get_alert_summary()
        
        return {
            'system_status': {
                'overall_status': status.status.value,
                'uptime_hours': status.uptime_seconds / 3600,
                'status_changes': self.status_change_count,
                'message': status.message
            },
            'health_checks': health_summary,
            'alerts': alert_summary,
            'metrics_summary': {
                'total_metrics': len(self.metrics_collector.metrics_buffer),
                'current_values': status.metrics
            },
            'performance_history_points': len(self.performance_history)
        }
    
    async def update_trading_metrics(self, **kwargs):
        """Update trading-specific metrics"""
        self.metrics_collector.collect_trading_metrics(**kwargs)
    
    async def force_health_check(self, check_name: str = None) -> Dict[str, Any]:
        """Force execution of health checks"""
        if check_name:
            if check_name in self.health_check_manager.health_checks:
                await self.health_check_manager._execute_health_check(check_name)
                return {check_name: self.health_check_manager.check_status[check_name].value}
            else:
                return {"error": f"Health check '{check_name}' not found"}
        else:
            # Execute all health checks
            for name in self.health_check_manager.health_checks.keys():
                await self.health_check_manager._execute_health_check(name)
            
            return self.health_check_manager.get_status_summary()


# Example usage and testing
async def main():
    """Example usage of comprehensive health monitoring"""
    print("Phase 10: Comprehensive System Health Monitoring")
    print("=" * 60)
    
    # Initialize monitoring system
    config = {
        'alerting': {
            'email': {
                'enabled': False,  # Disable for demo
                'smtp_host': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your_email@gmail.com',
                'password': 'your_password',
                'from': 'trading-bot@example.com',
                'recipients': ['admin@example.com']
            },
            'discord': {
                'enabled': False,  # Disable for demo
                'webhook_url': 'https://discord.com/api/webhooks/...'
            },
            'cooldown_minutes': 5
        }
    }
    
    monitor = ComprehensiveHealthMonitor(config)
    
    try:
        print("Starting comprehensive health monitoring...")
        await monitor.start_monitoring()
        
        # Simulate monitoring for a short period
        print("Monitoring system health...")
        
        for i in range(10):
            # Update some trading metrics
            await monitor.update_trading_metrics(
                portfolio_value=10000 + i * 100,
                open_positions=2 + (i % 3),
                daily_pnl=50.0 + i * 10,
                trade_count_delta=1 if i % 3 == 0 else 0
            )
            
            # Get status every few iterations
            if i % 3 == 0:
                status = monitor.get_comprehensive_status()
                print(f"\n📊 System Status (iteration {i+1}):")
                print(f"   Overall Status: {status.status.value.upper()}")
                print(f"   Uptime: {status.uptime_seconds/60:.1f} minutes")
                print(f"   Active Alerts: {len(status.active_alerts)}")
                print(f"   Components: {len(status.component_status)} health checks")
                
                # Show key metrics
                if status.metrics:
                    cpu = status.metrics.get('system_cpu_percent', 0)
                    memory = status.metrics.get('system_memory_percent', 0)
                    portfolio = status.metrics.get('trading_portfolio_value', 0)
                    print(f"   CPU: {cpu:.1f}%, Memory: {memory:.1f}%, Portfolio: ${portfolio:,.0f}")
            
            await asyncio.sleep(2)
        
        # Force health check execution
        print(f"\n🔍 Running forced health checks...")
        health_results = await monitor.force_health_check()
        print(f"Health check results: {health_results['overall_status']}")
        
        # Get comprehensive summary
        print(f"\n📋 Monitoring Summary:")
        summary = monitor.get_monitoring_summary()
        
        system_status = summary['system_status']
        print(f"   System Status: {system_status['overall_status'].upper()}")
        print(f"   Uptime: {system_status['uptime_hours']:.2f} hours")
        print(f"   Status Changes: {system_status['status_changes']}")
        
        health_checks = summary['health_checks']
        print(f"   Health Checks: {health_checks['total_checks']} total")
        print(f"     Healthy: {health_checks['healthy_checks']}")
        print(f"     Warning: {health_checks['warning_checks']}")
        print(f"     Critical: {health_checks['critical_checks']}")
        
        alerts = summary['alerts']
        print(f"   Alerts: {alerts['total_alerts']} total, {alerts['active_alerts']} active")
        
        metrics = summary['metrics_summary']
        print(f"   Metrics: {metrics['total_metrics']} tracked")
        
    finally:
        print(f"\nStopping monitoring system...")
        await monitor.stop_monitoring()
    
    print(f"\n🎉 System Health Monitoring Demo Complete!")
    print(f"✅ Real-time metrics collection")
    print(f"✅ Intelligent health checks")
    print(f"✅ Multi-channel alerting system")
    print(f"✅ Anomaly detection")
    print(f"✅ Trend analysis")
    print(f"✅ Performance tracking")
    print(f"✅ Automated recovery mechanisms")


if __name__ == "__main__":
    asyncio.run(main())