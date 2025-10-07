"""
Alert System for Real-time Notifications and Risk Management

This module provides comprehensive alerting capabilities:
- Risk-based alerts for portfolio and position monitoring
- Trade execution alerts and notifications
- System health alerts and diagnostics
- Customizable alert rules and thresholds
- Multiple notification channels (email, webhook, dashboard)
- Alert escalation and acknowledgment system

Integrates with all system components to provide proactive monitoring
and rapid response to critical events and threshold breaches.

Author: Trading Bot Team
Version: 1.0.0 - Phase 5 Implementation
"""

import asyncio
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ..utils.logging import TradingLogger
from ..config_manager import ConfigurationManager
from .live_execution_engine import ExecutionResult, ExecutionStatus
from .monitoring_dashboard import PerformanceMetrics, SystemHealth


class AlertType(Enum):
    """Types of alerts in the system."""
    RISK_LIMIT_BREACH = "risk_limit_breach"
    POSITION_LOSS = "position_loss"
    STRATEGY_PERFORMANCE = "strategy_performance"
    EXECUTION_ERROR = "execution_error"
    SYSTEM_ERROR = "system_error"
    CONNECTION_LOSS = "connection_loss"
    BALANCE_LOW = "balance_low"
    DRAWDOWN_LIMIT = "drawdown_limit"
    UNUSUAL_ACTIVITY = "unusual_activity"
    HEALTH_CHECK = "health_check"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Available notification channels."""
    DASHBOARD = "dashboard"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    CONSOLE = "console"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: str  # Condition expression
    threshold_value: Optional[Union[Decimal, float, int]] = None
    time_window_minutes: int = 5
    min_trigger_count: int = 1
    cooldown_minutes: int = 15
    enabled: bool = True
    channels: List[NotificationChannel] = field(default_factory=list)
    escalation_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    source_component: str = "unknown"
    affected_symbols: List[str] = field(default_factory=list)
    affected_strategies: List[str] = field(default_factory=list)
    trigger_value: Optional[Union[Decimal, float, int]] = None
    threshold_value: Optional[Union[Decimal, float, int]] = None
    acknowledgments: List[Dict[str, Any]] = field(default_factory=list)
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def acknowledge(self, user: str = "system", note: str = "") -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledgments.append({
            "user": user,
            "timestamp": datetime.now(),
            "note": note
        })
    
    def resolve(self, user: str = "system", note: str = "") -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()
        self.resolution_notes = note
        if not self.acknowledgments:
            self.acknowledge(user, "Auto-acknowledged on resolution")


@dataclass
class NotificationConfig:
    """Notification channel configuration."""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


class AlertSystem:
    """
    Comprehensive alert system for trading bot monitoring.
    
    Features:
    - Configurable alert rules with conditions and thresholds
    - Multiple severity levels and notification channels
    - Alert escalation and acknowledgment workflows
    - Rate limiting and cooldown periods
    - Historical alert tracking and analytics
    - Integration with all system components
    """
    
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.logger = TradingLogger("alert_system")
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Notification channels
        self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
        self.notification_handlers: Dict[NotificationChannel, Callable] = {}
        
        # Alert tracking
        self.trigger_counts: Dict[str, int] = {}  # Rule trigger counts
        self.last_trigger_times: Dict[str, datetime] = {}
        self.suppressed_rules: Dict[str, datetime] = {}  # Rule ID -> suppression end time
        
        # Background tasks
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Configuration
        self.max_active_alerts = config.get('alerts.max_active_alerts', 100)
        self.history_retention_days = config.get('alerts.history_retention_days', 30)
        self.cleanup_interval_minutes = config.get('alerts.cleanup_interval_minutes', 60)
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Setup notification channels
        self._setup_notification_channels()
        
        self.logger.info("AlertSystem initialized")
    
    async def start(self) -> bool:
        """
        Start the alert system.
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.logger.info("Starting alert system...")
            self.running = True
            
            # Start alert monitoring task
            monitor_task = asyncio.create_task(self._alert_monitor_loop())
            self.tasks.append(monitor_task)
            
            # Start cleanup task
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.tasks.append(cleanup_task)
            
            self.logger.info("Alert system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start alert system: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the alert system."""
        try:
            self.logger.info("Stopping alert system...")
            self.running = False
            
            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            self.logger.info("Alert system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping alert system: {e}")
    
    async def trigger_alert(
        self,
        rule_id: str,
        trigger_value: Optional[Union[Decimal, float, int]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Alert]:
        """
        Trigger an alert based on a rule.
        
        Args:
            rule_id: Alert rule identifier
            trigger_value: Value that triggered the alert
            context: Additional context information
            
        Returns:
            Alert: Created alert or None if suppressed
        """
        try:
            if rule_id not in self.alert_rules:
                self.logger.warning(f"Unknown alert rule: {rule_id}")
                return None
            
            rule = self.alert_rules[rule_id]
            
            if not rule.enabled:
                return None
            
            # Check if rule is suppressed
            if rule_id in self.suppressed_rules:
                suppression_end = self.suppressed_rules[rule_id]
                if datetime.now() < suppression_end:
                    return None
                else:
                    del self.suppressed_rules[rule_id]
            
            # Check cooldown period
            if rule_id in self.last_trigger_times:
                last_trigger = self.last_trigger_times[rule_id]
                cooldown_end = last_trigger + timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() < cooldown_end:
                    return None
            
            # Update trigger tracking
            self.trigger_counts[rule_id] = self.trigger_counts.get(rule_id, 0) + 1
            self.last_trigger_times[rule_id] = datetime.now()
            
            # Check minimum trigger count
            if self.trigger_counts[rule_id] < rule.min_trigger_count:
                return None
            
            # Create alert
            alert_id = f"{rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            alert = Alert(
                alert_id=alert_id,
                rule_id=rule_id,
                alert_type=rule.alert_type,
                severity=rule.severity,
                title=rule.name,
                message=self._format_alert_message(rule, trigger_value, context),
                timestamp=datetime.now(),
                source_component=context.get('component', 'unknown') if context else 'unknown',
                trigger_value=trigger_value,
                threshold_value=rule.threshold_value,
                metadata=context or {}
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Send notifications
            await self._send_notifications(alert, rule)
            
            # Reset trigger count after successful alert
            self.trigger_counts[rule_id] = 0
            
            self.logger.info(f"Alert triggered: {alert.title} ({alert.severity.value})")
            return alert
            
        except Exception as e:
            self.logger.error(f"Error triggering alert {rule_id}: {e}")
            return None
    
    async def check_risk_limits(self, metrics: PerformanceMetrics) -> None:
        """Check risk limits and trigger alerts if necessary."""
        try:
            # Portfolio drawdown check
            if metrics.current_drawdown > Decimal('0.15'):  # 15% drawdown
                await self.trigger_alert(
                    "portfolio_drawdown_limit",
                    float(metrics.current_drawdown),
                    {
                        "component": "risk_manager",
                        "portfolio_value": float(metrics.total_value),
                        "max_drawdown": float(metrics.max_drawdown)
                    }
                )
            
            # Balance warning check
            if metrics.cash_balance < Decimal('1000'):  # Low cash balance
                await self.trigger_alert(
                    "low_balance_warning",
                    float(metrics.cash_balance),
                    {
                        "component": "portfolio_manager",
                        "total_value": float(metrics.total_value)
                    }
                )
            
            # Performance degradation check
            if metrics.sharpe_ratio < Decimal('0.5'):  # Poor Sharpe ratio
                await self.trigger_alert(
                    "performance_degradation",
                    float(metrics.sharpe_ratio),
                    {
                        "component": "strategy_manager",
                        "win_rate": float(metrics.win_rate),
                        "profit_factor": float(metrics.profit_factor)
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    async def check_execution_quality(self, execution_result: ExecutionResult) -> None:
        """Check execution quality and trigger alerts if necessary."""
        try:
            # Execution failure check
            if execution_result.status == ExecutionStatus.FAILED:
                await self.trigger_alert(
                    "execution_failure",
                    None,
                    {
                        "component": "execution_engine",
                        "symbol": execution_result.symbol,
                        "strategy": execution_result.strategy_id,
                        "error": execution_result.error_message
                    }
                )
            
            # High slippage check
            if execution_result.slippage_bps > 50:  # More than 50 bps slippage
                await self.trigger_alert(
                    "high_slippage",
                    float(execution_result.slippage_bps),
                    {
                        "component": "execution_engine",
                        "symbol": execution_result.symbol,
                        "requested_quantity": float(execution_result.requested_quantity),
                        "executed_quantity": float(execution_result.executed_quantity)
                    }
                )
            
            # Slow execution check
            if execution_result.execution_time_ms > 5000:  # More than 5 seconds
                await self.trigger_alert(
                    "slow_execution",
                    execution_result.execution_time_ms,
                    {
                        "component": "execution_engine",
                        "symbol": execution_result.symbol,
                        "strategy": execution_result.strategy_id
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Error checking execution quality: {e}")
    
    async def check_system_health(self, health: SystemHealth) -> None:
        """Check system health and trigger alerts if necessary."""
        try:
            # Memory usage check
            if health.memory_usage_mb > 2048:  # More than 2GB memory usage
                await self.trigger_alert(
                    "high_memory_usage",
                    health.memory_usage_mb,
                    {
                        "component": "system_monitor",
                        "cpu_usage": health.cpu_usage_percent
                    }
                )
            
            # Connection health check
            if health.websocket_latency_ms > 1000:  # High WebSocket latency
                await self.trigger_alert(
                    "high_websocket_latency",
                    health.websocket_latency_ms,
                    {
                        "component": "websocket_manager",
                        "api_latency": health.bybit_api_latency_ms
                    }
                )
            
            # Component health check
            unhealthy_components = []
            if health.websocket_status != "connected":
                unhealthy_components.append("websocket")
            if health.execution_engine_status != "healthy":
                unhealthy_components.append("execution_engine")
            
            if unhealthy_components:
                await self.trigger_alert(
                    "component_health_degraded",
                    len(unhealthy_components),
                    {
                        "component": "system_monitor",
                        "unhealthy_components": unhealthy_components,
                        "overall_status": health.overall_status
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="portfolio_drawdown_limit",
                name="Portfolio Drawdown Limit",
                description="Portfolio drawdown exceeds safe threshold",
                alert_type=AlertType.DRAWDOWN_LIMIT,
                severity=AlertSeverity.CRITICAL,
                condition="current_drawdown > 0.15",
                threshold_value=0.15,
                cooldown_minutes=30,
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.EMAIL]
            ),
            AlertRule(
                rule_id="low_balance_warning",
                name="Low Balance Warning",
                description="Cash balance is running low",
                alert_type=AlertType.BALANCE_LOW,
                severity=AlertSeverity.WARNING,
                condition="cash_balance < 1000",
                threshold_value=1000,
                cooldown_minutes=60,
                channels=[NotificationChannel.DASHBOARD]
            ),
            AlertRule(
                rule_id="execution_failure",
                name="Execution Failure",
                description="Trade execution failed",
                alert_type=AlertType.EXECUTION_ERROR,
                severity=AlertSeverity.ERROR,
                condition="execution_status == 'failed'",
                cooldown_minutes=5,
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.EMAIL]
            ),
            AlertRule(
                rule_id="high_slippage",
                name="High Slippage",
                description="Execution slippage exceeds threshold",
                alert_type=AlertType.EXECUTION_ERROR,
                severity=AlertSeverity.WARNING,
                condition="slippage_bps > 50",
                threshold_value=50,
                cooldown_minutes=15,
                channels=[NotificationChannel.DASHBOARD]
            ),
            AlertRule(
                rule_id="performance_degradation",
                name="Performance Degradation",
                description="Strategy performance below acceptable levels",
                alert_type=AlertType.STRATEGY_PERFORMANCE,
                severity=AlertSeverity.WARNING,
                condition="sharpe_ratio < 0.5",
                threshold_value=0.5,
                cooldown_minutes=120,
                channels=[NotificationChannel.DASHBOARD]
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="System memory usage is high",
                alert_type=AlertType.SYSTEM_ERROR,
                severity=AlertSeverity.WARNING,
                condition="memory_usage_mb > 2048",
                threshold_value=2048,
                cooldown_minutes=30,
                channels=[NotificationChannel.DASHBOARD]
            ),
            AlertRule(
                rule_id="high_websocket_latency",
                name="High WebSocket Latency",
                description="WebSocket connection latency is high",
                alert_type=AlertType.CONNECTION_LOSS,
                severity=AlertSeverity.WARNING,
                condition="websocket_latency_ms > 1000",
                threshold_value=1000,
                cooldown_minutes=15,
                channels=[NotificationChannel.DASHBOARD]
            ),
            AlertRule(
                rule_id="component_health_degraded",
                name="Component Health Degraded",
                description="One or more system components are unhealthy",
                alert_type=AlertType.HEALTH_CHECK,
                severity=AlertSeverity.ERROR,
                condition="unhealthy_components > 0",
                cooldown_minutes=10,
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.EMAIL]
            ),
            AlertRule(
                rule_id="slow_execution", 
                name="Slow Execution",
                description="Trade execution taking too long",
                alert_type=AlertType.EXECUTION_ERROR,
                severity=AlertSeverity.WARNING,
                condition="execution_time_ms > 5000",
                threshold_value=5000,
                cooldown_minutes=10,
                channels=[NotificationChannel.DASHBOARD]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
        
        self.logger.info(f"Setup {len(default_rules)} default alert rules")
    
    def _setup_notification_channels(self) -> None:
        """Setup notification channels and handlers."""
        # Dashboard notifications (always enabled)
        self.notification_configs[NotificationChannel.DASHBOARD] = NotificationConfig(
            channel=NotificationChannel.DASHBOARD,
            enabled=True
        )
        self.notification_handlers[NotificationChannel.DASHBOARD] = self._send_dashboard_notification
        
        # Email notifications
        email_config = self.config.get('alerts.email', {})
        if email_config.get('enabled', False):
            self.notification_configs[NotificationChannel.EMAIL] = NotificationConfig(
                channel=NotificationChannel.EMAIL,
                enabled=True,
                config=email_config
            )
            self.notification_handlers[NotificationChannel.EMAIL] = self._send_email_notification
        
        # Webhook notifications
        webhook_config = self.config.get('alerts.webhook', {})
        if webhook_config.get('enabled', False):
            self.notification_configs[NotificationChannel.WEBHOOK] = NotificationConfig(
                channel=NotificationChannel.WEBHOOK,
                enabled=True,
                config=webhook_config
            )
            self.notification_handlers[NotificationChannel.WEBHOOK] = self._send_webhook_notification
        
        # Console notifications (always enabled for development)
        self.notification_configs[NotificationChannel.CONSOLE] = NotificationConfig(
            channel=NotificationChannel.CONSOLE,
            enabled=True
        )
        self.notification_handlers[NotificationChannel.CONSOLE] = self._send_console_notification
    
    def _format_alert_message(
        self, 
        rule: AlertRule, 
        trigger_value: Optional[Union[Decimal, float, int]], 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format alert message with context information."""
        message = rule.description
        
        if trigger_value is not None and rule.threshold_value is not None:
            message += f" (Current: {trigger_value}, Threshold: {rule.threshold_value})"
        elif trigger_value is not None:
            message += f" (Value: {trigger_value})"
        
        if context:
            if "component" in context:
                message += f" [Component: {context['component']}]"
            if "symbol" in context:
                message += f" [Symbol: {context['symbol']}]"
            if "strategy" in context:
                message += f" [Strategy: {context['strategy']}]"
        
        return message
    
    async def _send_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send notifications through configured channels."""
        for channel in rule.channels:
            if channel in self.notification_handlers:
                config = self.notification_configs.get(channel)
                if config and config.enabled:
                    try:
                        await self.notification_handlers[channel](alert, config)
                    except Exception as e:
                        self.logger.error(f"Failed to send {channel.value} notification: {e}")
    
    async def _send_dashboard_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send notification to dashboard (stored in memory)."""
        # Dashboard notifications are handled by storing the alert
        # The dashboard will query active alerts to display them
        pass
    
    async def _send_email_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send email notification."""
        try:
            smtp_config = config.config
            
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = smtp_config['to_email']
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
Alert Details:
- Severity: {alert.severity.value.upper()}
- Type: {alert.alert_type.value}
- Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- Component: {alert.source_component}
- Message: {alert.message}

Additional Information:
{json.dumps(alert.metadata, indent=2)}

This is an automated message from the Trading Bot Alert System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_config['smtp_host'], smtp_config['smtp_port'])
            if smtp_config.get('use_tls', True):
                server.starttls()
            if 'username' in smtp_config and 'password' in smtp_config:
                server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Email notification error: {e}")
    
    async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send webhook notification."""
        try:
            webhook_config = config.config
            url = webhook_config['url']
            
            payload = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "component": alert.source_component,
                "metadata": alert.metadata
            }
            
            headers = {'Content-Type': 'application/json'}
            if 'headers' in webhook_config:
                headers.update(webhook_config['headers'])
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        self.logger.warning(f"Webhook returned status {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Webhook notification error: {e}")
    
    async def _send_console_notification(self, alert: Alert, config: NotificationConfig) -> None:
        """Send console notification (logging)."""
        log_level = {
            AlertSeverity.INFO: self.logger.info,
            AlertSeverity.WARNING: self.logger.warning,
            AlertSeverity.ERROR: self.logger.error,
            AlertSeverity.CRITICAL: self.logger.critical
        }.get(alert.severity, self.logger.info)
        
        log_level(f"ALERT: {alert.title} - {alert.message}")
    
    async def _alert_monitor_loop(self) -> None:
        """Background alert monitoring loop."""
        while self.running:
            try:
                # Check for alerts that should auto-resolve
                current_time = datetime.now()
                auto_resolve_candidates = []
                
                for alert in self.active_alerts.values():
                    if alert.status == AlertStatus.ACTIVE:
                        # Auto-resolve alerts older than 24 hours if not critical
                        if (alert.severity != AlertSeverity.CRITICAL and 
                            current_time - alert.timestamp > timedelta(hours=24)):
                            auto_resolve_candidates.append(alert)
                
                # Auto-resolve candidates
                for alert in auto_resolve_candidates:
                    alert.resolve("system", "Auto-resolved after 24 hours")
                    self.logger.info(f"Auto-resolved alert: {alert.alert_id}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Alert monitor loop error: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Clean up old resolved alerts from active list
                resolved_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if alert.status == AlertStatus.RESOLVED and 
                    alert.resolved_at and 
                    current_time - alert.resolved_at > timedelta(hours=1)
                ]
                
                for alert_id in resolved_alerts:
                    del self.active_alerts[alert_id]
                
                # Clean up old alert history
                cutoff_time = current_time - timedelta(days=self.history_retention_days)
                self.alert_history = [
                    alert for alert in self.alert_history
                    if alert.timestamp >= cutoff_time
                ]
                
                # Limit active alerts
                if len(self.active_alerts) > self.max_active_alerts:
                    # Remove oldest resolved alerts first
                    sorted_alerts = sorted(
                        self.active_alerts.values(),
                        key=lambda a: (a.status != AlertStatus.RESOLVED, a.timestamp)
                    )
                    
                    excess_count = len(self.active_alerts) - self.max_active_alerts
                    for alert in sorted_alerts[:excess_count]:
                        if alert.alert_id in self.active_alerts:
                            del self.active_alerts[alert.alert_id]
                
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = [alert for alert in self.active_alerts.values() if alert.status == AlertStatus.ACTIVE]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics."""
        current_time = datetime.now()
        
        # Count alerts by severity in last 24 hours
        last_24h = current_time - timedelta(hours=24)
        recent_alerts = [a for a in self.alert_history if a.timestamp >= last_24h]
        
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1
        
        # Count alerts by type
        type_counts = {}
        for alert in recent_alerts:
            type_name = alert.alert_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "active_alerts": len(self.active_alerts),
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "alerts_last_24h": len(recent_alerts),
            "severity_counts_24h": severity_counts,
            "type_counts_24h": type_counts,
            "suppressed_rules": len(self.suppressed_rules),
            "total_alerts_all_time": len(self.alert_history)
        }


# Utility functions for alert system integration

async def create_alert_system(config: ConfigurationManager) -> AlertSystem:
    """
    Create and start an alert system.
    
    Args:
        config: Configuration manager
        
    Returns:
        AlertSystem: Started alert system instance
    """
    alert_system = AlertSystem(config)
    await alert_system.start()
    return alert_system