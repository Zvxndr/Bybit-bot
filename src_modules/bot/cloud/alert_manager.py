"""
Alert Manager for Trading System.
Comprehensive alerting, notification, and incident management system.
"""

import asyncio
import json
import smtplib
import time
from typing import Dict, List, Optional, Any, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import hashlib
import re
import aiohttp

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertState(Enum):
    """Alert states."""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"
    SUPPRESSED = "suppressed"

class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"

class EscalationPolicy(Enum):
    """Escalation policies."""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    BUSINESS_HOURS = "business_hours"
    WEEKDAYS_ONLY = "weekdays_only"

@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    query: str
    severity: AlertSeverity
    threshold: Union[float, str]
    duration: str = "5m"
    description: str = ""
    runbook_url: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    group: str = "default"
    eval_interval: str = "1m"
    for_duration: str = "0s"

@dataclass
class AlertInstance:
    """Active alert instance."""
    id: str
    rule_name: str
    severity: AlertSeverity
    state: AlertState
    message: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    started_at: datetime
    resolved_at: Optional[datetime] = None
    last_notified: Optional[datetime] = None
    notification_count: int = 0
    fingerprint: str = ""
    silence_id: Optional[str] = None

@dataclass
class NotificationConfig:
    """Notification configuration."""
    name: str
    channel: NotificationChannel
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=lambda: list(AlertSeverity))
    label_matchers: Dict[str, str] = field(default_factory=dict)
    time_intervals: List[Dict[str, str]] = field(default_factory=list)
    rate_limit: int = 0  # Max notifications per hour, 0 = no limit

@dataclass
class Silence:
    """Alert silence configuration."""
    id: str
    matchers: List[Dict[str, str]]
    starts_at: datetime
    ends_at: datetime
    created_by: str
    comment: str = ""
    active: bool = True

@dataclass
class EscalationRule:
    """Escalation rule configuration."""
    name: str
    policy: EscalationPolicy
    severity_filter: List[AlertSeverity]
    time_to_escalate: timedelta
    escalation_targets: List[str]  # Notification config names
    max_escalations: int = 3
    business_hours: Dict[str, str] = field(default_factory=lambda: {
        'start': '09:00',
        'end': '17:00',
        'timezone': 'UTC',
        'weekdays_only': True
    })

@dataclass
class IncidentTicket:
    """Incident ticket for alert grouping."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: str
    created_at: datetime
    updated_at: datetime
    assignee: Optional[str] = None
    alerts: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)

class AlertManager:
    """Comprehensive alert management system."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Alert management configuration
        self.config = {
            'global': {
                'smtp_smarthost': 'localhost:587',
                'smtp_from': 'alerts@trading-system.com',
                'smtp_require_tls': True,
                'resolve_timeout': '5m',
                'http_config': {
                    'timeout': '10s'
                }
            },
            'route': {
                'group_by': ['alertname', 'cluster', 'service'],
                'group_wait': '30s',
                'group_interval': '5m',
                'repeat_interval': '12h',
                'receiver': 'default'
            },
            'inhibit_rules': [
                {
                    'source_matchers': ['severity=critical'],
                    'target_matchers': ['severity=warning'],
                    'equal': ['alertname', 'instance']
                }
            ],
            'time_intervals': [
                {
                    'name': 'business_hours',
                    'time_intervals': [
                        {
                            'times': [{'start_time': '09:00', 'end_time': '17:00'}],
                            'weekdays': ['monday:friday']
                        }
                    ]
                }
            ]
        }
        
        # Alert state management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertInstance] = {}
        self.alert_history: List[AlertInstance] = []
        self.silences: Dict[str, Silence] = {}
        
        # Notification management
        self.notification_configs: Dict[str, NotificationConfig] = {}
        self.escalation_rules: List[EscalationRule] = []
        self.notification_history: List[Dict[str, Any]] = []
        
        # Incident management
        self.incidents: Dict[str, IncidentTicket] = {}
        self.incident_counter = 1
        
        # Processing state
        self.alert_manager_active = False
        self.evaluation_task = None
        self.notification_task = None
        self.cleanup_task = None
        
        # Rate limiting
        self.notification_rates: Dict[str, List[datetime]] = {}
        
        # Initialize default configurations
        self._setup_default_notifications()
        self._setup_default_escalations()
        
        self.logger.info("AlertManager initialized")
    
    def _setup_default_notifications(self):
        """Setup default notification configurations."""
        try:
            # Email notifications
            email_config = NotificationConfig(
                name="email-critical",
                channel=NotificationChannel.EMAIL,
                config={
                    'to': ['ops@trading-system.com', 'dev@trading-system.com'],
                    'subject': '[ALERT] {{ .GroupLabels.alertname }} - {{ .Status }}',
                    'body': '''
Alert: {{ .GroupLabels.alertname }}
Severity: {{ .CommonLabels.severity }}
Status: {{ .Status }}
Started: {{ .StartsAt }}
{{ if .ResolvedAt }}Resolved: {{ .ResolvedAt }}{{ end }}

Description: {{ .CommonAnnotations.description }}
{{ if .CommonAnnotations.runbook_url }}Runbook: {{ .CommonAnnotations.runbook_url }}{{ end }}

Labels:
{{ range .CommonLabels.SortedPairs }}  {{ .Name }}: {{ .Value }}
{{ end }}

Firing Alerts:
{{ range .Alerts.Firing }}  - {{ .Annotations.summary }}{{ end }}
                    '''
                },
                severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.HIGH],
                rate_limit=10
            )
            
            # Slack notifications
            slack_config = NotificationConfig(
                name="slack-general",
                channel=NotificationChannel.SLACK,
                config={
                    'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
                    'channel': '#alerts',
                    'username': 'AlertManager',
                    'icon_emoji': ':warning:',
                    'title': 'Trading System Alert',
                    'text': '{{ .CommonAnnotations.summary }}',
                    'color': 'danger'
                },
                severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM],
                rate_limit=20
            )
            
            # Webhook notifications
            webhook_config = NotificationConfig(
                name="webhook-incidents",
                channel=NotificationChannel.WEBHOOK,
                config={
                    'url': 'http://localhost:8080/api/alerts',
                    'method': 'POST',
                    'headers': {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer YOUR_TOKEN'
                    }
                },
                severity_filter=list(AlertSeverity)
            )
            
            self.notification_configs["email-critical"] = email_config
            self.notification_configs["slack-general"] = slack_config
            self.notification_configs["webhook-incidents"] = webhook_config
            
        except Exception as e:
            self.logger.error(f"Failed to setup default notifications: {e}")
    
    def _setup_default_escalations(self):
        """Setup default escalation rules."""
        try:
            # Critical alert escalation
            critical_escalation = EscalationRule(
                name="critical-escalation",
                policy=EscalationPolicy.IMMEDIATE,
                severity_filter=[AlertSeverity.CRITICAL],
                time_to_escalate=timedelta(minutes=5),
                escalation_targets=["email-critical", "slack-general"],
                max_escalations=3
            )
            
            # High priority escalation
            high_escalation = EscalationRule(
                name="high-escalation",
                policy=EscalationPolicy.BUSINESS_HOURS,
                severity_filter=[AlertSeverity.HIGH],
                time_to_escalate=timedelta(minutes=15),
                escalation_targets=["slack-general"],
                max_escalations=2
            )
            
            # Medium priority escalation
            medium_escalation = EscalationRule(
                name="medium-escalation",
                policy=EscalationPolicy.WEEKDAYS_ONLY,
                severity_filter=[AlertSeverity.MEDIUM],
                time_to_escalate=timedelta(hours=1),
                escalation_targets=["slack-general"],
                max_escalations=1
            )
            
            self.escalation_rules.extend([
                critical_escalation,
                high_escalation,
                medium_escalation
            ])
            
        except Exception as e:
            self.logger.error(f"Failed to setup default escalations: {e}")
    
    async def start_alert_manager(self):
        """Start alert manager processing."""
        try:
            if self.alert_manager_active:
                return
            
            self.alert_manager_active = True
            
            # Start processing tasks
            self.evaluation_task = asyncio.create_task(self._evaluation_loop())
            self.notification_task = asyncio.create_task(self._notification_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.logger.info("Alert Manager started")
            
        except Exception as e:
            self.logger.error(f"Failed to start Alert Manager: {e}")
    
    async def stop_alert_manager(self):
        """Stop alert manager processing."""
        try:
            self.alert_manager_active = False
            
            # Cancel tasks
            tasks = [self.evaluation_task, self.notification_task, self.cleanup_task]
            for task in tasks:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.logger.info("Alert Manager stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop Alert Manager: {e}")
    
    async def _evaluation_loop(self):
        """Alert evaluation loop."""
        try:
            while self.alert_manager_active:
                await self._evaluate_alerts()
                await asyncio.sleep(60)  # Evaluate every minute
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Alert evaluation loop error: {e}")
    
    async def _notification_loop(self):
        """Notification processing loop."""
        try:
            while self.alert_manager_active:
                await self._process_notifications()
                await asyncio.sleep(30)  # Process notifications every 30 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Notification loop error: {e}")
    
    async def _cleanup_loop(self):
        """Cleanup loop for resolved alerts and old data."""
        try:
            while self.alert_manager_active:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Cleanup loop error: {e}")
    
    async def _evaluate_alerts(self):
        """Evaluate all alert rules."""
        try:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                await self._evaluate_alert_rule(rule)
                
        except Exception as e:
            self.logger.error(f"Failed to evaluate alerts: {e}")
    
    async def _evaluate_alert_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        try:
            # Generate alert fingerprint
            fingerprint = self._generate_fingerprint(rule.name, rule.labels)
            
            # Check if alert should be firing
            is_firing = await self._check_alert_condition(rule)
            
            if is_firing and fingerprint not in self.active_alerts:
                # Fire new alert
                alert = AlertInstance(
                    id=f"{rule.name}-{int(time.time())}",
                    rule_name=rule.name,
                    severity=rule.severity,
                    state=AlertState.FIRING,
                    message=rule.description,
                    labels=rule.labels,
                    annotations=rule.annotations,
                    started_at=datetime.now(),
                    fingerprint=fingerprint
                )
                
                # Check if silenced
                if self._is_silenced(alert):
                    alert.state = AlertState.SILENCED
                
                self.active_alerts[fingerprint] = alert
                
                # Create or update incident
                await self._handle_incident(alert)
                
                self.logger.warning(f"Alert fired: {rule.name}")
                
            elif not is_firing and fingerprint in self.active_alerts:
                # Resolve alert
                alert = self.active_alerts[fingerprint]
                alert.state = AlertState.RESOLVED
                alert.resolved_at = datetime.now()
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[fingerprint]
                
                # Update incident
                await self._update_incident_for_resolved_alert(alert)
                
                self.logger.info(f"Alert resolved: {rule.name}")
                
        except Exception as e:
            self.logger.error(f"Failed to evaluate alert rule {rule.name}: {e}")
    
    async def _check_alert_condition(self, rule: AlertRule) -> bool:
        """Check if alert condition is met."""
        try:
            # This would normally query Prometheus or other metrics systems
            # For now, simulate based on rule characteristics
            
            if "high_order_failure_rate" in rule.name:
                return False  # Simulate no failures
            elif "portfolio_drawdown" in rule.name:
                return False  # Simulate no drawdown
            elif "service_down" in rule.name:
                return False  # Simulate all services up
            elif "high_response_time" in rule.name:
                return False  # Simulate normal response times
            elif "high_memory_usage" in rule.name:
                return False  # Simulate normal memory usage
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to check alert condition: {e}")
            return False
    
    def _generate_fingerprint(self, alert_name: str, labels: Dict[str, str]) -> str:
        """Generate unique fingerprint for alert."""
        try:
            label_string = json.dumps(sorted(labels.items()))
            fingerprint_data = f"{alert_name}:{label_string}"
            return hashlib.md5(fingerprint_data.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Failed to generate fingerprint: {e}")
            return alert_name
    
    def _is_silenced(self, alert: AlertInstance) -> bool:
        """Check if alert is silenced."""
        try:
            current_time = datetime.now()
            
            for silence in self.silences.values():
                if not silence.active:
                    continue
                
                if current_time < silence.starts_at or current_time > silence.ends_at:
                    continue
                
                # Check if alert matches silence matchers
                matches = True
                for matcher in silence.matchers:
                    field = matcher.get('name', '')
                    value = matcher.get('value', '')
                    regex = matcher.get('isRegex', False)
                    
                    alert_value = alert.labels.get(field, '')
                    
                    if regex:
                        if not re.match(value, alert_value):
                            matches = False
                            break
                    else:
                        if alert_value != value:
                            matches = False
                            break
                
                if matches:
                    alert.silence_id = silence.id
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check silence: {e}")
            return False
    
    async def _process_notifications(self):
        """Process notifications for active alerts."""
        try:
            for alert in self.active_alerts.values():
                if alert.state in [AlertState.SILENCED, AlertState.SUPPRESSED]:
                    continue
                
                await self._process_alert_notifications(alert)
                
        except Exception as e:
            self.logger.error(f"Failed to process notifications: {e}")
    
    async def _process_alert_notifications(self, alert: AlertInstance):
        """Process notifications for a single alert."""
        try:
            # Check if notification is needed
            if not self._should_notify(alert):
                return
            
            # Find matching notification configurations
            matching_configs = self._find_matching_notifications(alert)
            
            for config in matching_configs:
                if not self._check_rate_limit(config.name):
                    continue
                
                await self._send_notification(config, alert)
                
            # Update notification timestamp
            alert.last_notified = datetime.now()
            alert.notification_count += 1
            
        except Exception as e:
            self.logger.error(f"Failed to process alert notifications: {e}")
    
    def _should_notify(self, alert: AlertInstance) -> bool:
        """Check if alert should trigger notification."""
        try:
            # Don't notify for silenced or suppressed alerts
            if alert.state in [AlertState.SILENCED, AlertState.SUPPRESSED]:
                return False
            
            # Check repeat interval
            if alert.last_notified:
                repeat_interval = timedelta(hours=12)  # Default repeat interval
                if datetime.now() - alert.last_notified < repeat_interval:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check notification requirement: {e}")
            return False
    
    def _find_matching_notifications(self, alert: AlertInstance) -> List[NotificationConfig]:
        """Find notification configurations matching the alert."""
        try:
            matching_configs = []
            
            for config in self.notification_configs.values():
                if not config.enabled:
                    continue
                
                # Check severity filter
                if alert.severity not in config.severity_filter:
                    continue
                
                # Check label matchers
                if config.label_matchers:
                    matches = True
                    for label, value in config.label_matchers.items():
                        if alert.labels.get(label) != value:
                            matches = False
                            break
                    
                    if not matches:
                        continue
                
                # Check time intervals
                if config.time_intervals and not self._is_in_time_interval(config.time_intervals):
                    continue
                
                matching_configs.append(config)
            
            return matching_configs
            
        except Exception as e:
            self.logger.error(f"Failed to find matching notifications: {e}")
            return []
    
    def _check_rate_limit(self, config_name: str) -> bool:
        """Check rate limit for notification configuration."""
        try:
            if config_name not in self.notification_configs:
                return False
            
            config = self.notification_configs[config_name]
            if config.rate_limit <= 0:
                return True  # No rate limit
            
            current_time = datetime.now()
            hour_ago = current_time - timedelta(hours=1)
            
            # Clean old entries
            if config_name in self.notification_rates:
                self.notification_rates[config_name] = [
                    ts for ts in self.notification_rates[config_name] if ts > hour_ago
                ]
            else:
                self.notification_rates[config_name] = []
            
            # Check if under rate limit
            if len(self.notification_rates[config_name]) < config.rate_limit:
                self.notification_rates[config_name].append(current_time)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check rate limit: {e}")
            return False
    
    def _is_in_time_interval(self, intervals: List[Dict[str, str]]) -> bool:
        """Check if current time is within specified intervals."""
        try:
            current_time = datetime.now().time()
            current_weekday = datetime.now().weekday()  # 0 = Monday
            
            for interval in intervals:
                start_time = datetime.strptime(interval.get('start', '00:00'), '%H:%M').time()
                end_time = datetime.strptime(interval.get('end', '23:59'), '%H:%M').time()
                
                weekdays = interval.get('weekdays', 'all')
                if weekdays != 'all':
                    if 'monday:friday' in weekdays and current_weekday > 4:
                        continue
                    # Add more weekday logic as needed
                
                if start_time <= current_time <= end_time:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check time interval: {e}")
            return True  # Default to allow
    
    async def _send_notification(self, config: NotificationConfig, alert: AlertInstance):
        """Send notification through specified channel."""
        try:
            if config.channel == NotificationChannel.EMAIL:
                await self._send_email_notification(config, alert)
            elif config.channel == NotificationChannel.SLACK:
                await self._send_slack_notification(config, alert)
            elif config.channel == NotificationChannel.WEBHOOK:
                await self._send_webhook_notification(config, alert)
            else:
                self.logger.warning(f"Unsupported notification channel: {config.channel}")
            
            # Record notification
            self.notification_history.append({
                'timestamp': datetime.now(),
                'config_name': config.name,
                'channel': config.channel.value,
                'alert_id': alert.id,
                'severity': alert.severity.value,
                'success': True
            })
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            
            # Record failed notification
            self.notification_history.append({
                'timestamp': datetime.now(),
                'config_name': config.name,
                'channel': config.channel.value,
                'alert_id': alert.id,
                'severity': alert.severity.value,
                'success': False,
                'error': str(e)
            })
    
    async def _send_email_notification(self, config: NotificationConfig, alert: AlertInstance):
        """Send email notification."""
        try:
            email_config = config.config
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['global']['smtp_from']
            msg['To'] = ', '.join(email_config['to'])
            msg['Subject'] = self._template_string(email_config['subject'], alert)
            
            body = self._template_string(email_config['body'], alert)
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            smtp_host, smtp_port = self.config['global']['smtp_smarthost'].split(':')
            server = smtplib.SMTP(smtp_host, int(smtp_port))
            
            if self.config['global']['smtp_require_tls']:
                server.starttls()
            
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
    
    async def _send_slack_notification(self, config: NotificationConfig, alert: AlertInstance):
        """Send Slack notification."""
        try:
            slack_config = config.config
            
            payload = {
                'channel': slack_config['channel'],
                'username': slack_config['username'],
                'icon_emoji': slack_config['icon_emoji'],
                'attachments': [{
                    'color': self._get_slack_color(alert.severity),
                    'title': slack_config['title'],
                    'text': self._template_string(slack_config['text'], alert),
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity.value, 'short': True},
                        {'title': 'Status', 'value': alert.state.value, 'short': True},
                        {'title': 'Started', 'value': alert.started_at.strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
                    ],
                    'ts': int(alert.started_at.timestamp())
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(slack_config['webhook_url'], json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"Slack API error: {response.status}")
            
            self.logger.info(f"Slack notification sent for alert {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
    
    async def _send_webhook_notification(self, config: NotificationConfig, alert: AlertInstance):
        """Send webhook notification."""
        try:
            webhook_config = config.config
            
            payload = {
                'alert_id': alert.id,
                'rule_name': alert.rule_name,
                'severity': alert.severity.value,
                'state': alert.state.value,
                'message': alert.message,
                'labels': alert.labels,
                'annotations': alert.annotations,
                'started_at': alert.started_at.isoformat(),
                'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            
            headers = webhook_config.get('headers', {})
            method = webhook_config.get('method', 'POST').upper()
            
            async with aiohttp.ClientSession() as session:
                if method == 'POST':
                    async with session.post(webhook_config['url'], json=payload, headers=headers) as response:
                        if response.status not in [200, 201, 202]:
                            raise Exception(f"Webhook error: {response.status}")
                elif method == 'PUT':
                    async with session.put(webhook_config['url'], json=payload, headers=headers) as response:
                        if response.status not in [200, 201, 202]:
                            raise Exception(f"Webhook error: {response.status}")
            
            self.logger.info(f"Webhook notification sent for alert {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
    
    def _get_slack_color(self, severity: AlertSeverity) -> str:
        """Get Slack color for alert severity."""
        color_map = {
            AlertSeverity.CRITICAL: 'danger',
            AlertSeverity.HIGH: 'warning',
            AlertSeverity.MEDIUM: 'warning',
            AlertSeverity.LOW: 'good',
            AlertSeverity.INFO: 'good'
        }
        return color_map.get(severity, 'warning')
    
    def _template_string(self, template: str, alert: AlertInstance) -> str:
        """Simple template string replacement."""
        try:
            # Simple template replacement (would use proper templating in production)
            result = template
            result = result.replace('{{ .GroupLabels.alertname }}', alert.rule_name)
            result = result.replace('{{ .CommonLabels.severity }}', alert.severity.value)
            result = result.replace('{{ .Status }}', alert.state.value)
            result = result.replace('{{ .StartsAt }}', alert.started_at.strftime('%Y-%m-%d %H:%M:%S'))
            result = result.replace('{{ .CommonAnnotations.description }}', alert.message)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to template string: {e}")
            return template
    
    async def _handle_incident(self, alert: AlertInstance):
        """Handle incident creation/update for alert."""
        try:
            # Find existing incident for this type of alert
            incident_key = f"{alert.rule_name}-{alert.severity.value}"
            
            existing_incident = None
            for incident in self.incidents.values():
                if incident_key in incident.id or alert.rule_name in incident.title:
                    existing_incident = incident
                    break
            
            if existing_incident:
                # Update existing incident
                if alert.id not in existing_incident.alerts:
                    existing_incident.alerts.append(alert.id)
                    existing_incident.updated_at = datetime.now()
                    existing_incident.timeline.append({
                        'timestamp': datetime.now(),
                        'event': 'alert_added',
                        'alert_id': alert.id,
                        'message': f"Alert {alert.rule_name} added to incident"
                    })
            else:
                # Create new incident
                incident = IncidentTicket(
                    id=f"INC-{self.incident_counter:06d}",
                    title=f"{alert.rule_name} - {alert.severity.value.upper()}",
                    description=alert.message,
                    severity=alert.severity,
                    status="open",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    alerts=[alert.id],
                    timeline=[{
                        'timestamp': datetime.now(),
                        'event': 'incident_created',
                        'alert_id': alert.id,
                        'message': f"Incident created for alert {alert.rule_name}"
                    }]
                )
                
                self.incidents[incident.id] = incident
                self.incident_counter += 1
                
        except Exception as e:
            self.logger.error(f"Failed to handle incident: {e}")
    
    async def _update_incident_for_resolved_alert(self, alert: AlertInstance):
        """Update incident when alert is resolved."""
        try:
            # Find incident containing this alert
            for incident in self.incidents.values():
                if alert.id in incident.alerts:
                    incident.timeline.append({
                        'timestamp': datetime.now(),
                        'event': 'alert_resolved',
                        'alert_id': alert.id,
                        'message': f"Alert {alert.rule_name} resolved"
                    })
                    incident.updated_at = datetime.now()
                    
                    # Check if all alerts in incident are resolved
                    all_resolved = True
                    for alert_id in incident.alerts:
                        if any(a.id == alert_id and a.state != AlertState.RESOLVED 
                               for a in self.active_alerts.values()):
                            all_resolved = False
                            break
                    
                    if all_resolved:
                        incident.status = "resolved"
                        incident.timeline.append({
                            'timestamp': datetime.now(),
                            'event': 'incident_resolved',
                            'message': "All alerts resolved, incident closed"
                        })
                    
                    break
                    
        except Exception as e:
            self.logger.error(f"Failed to update incident for resolved alert: {e}")
    
    async def _cleanup_old_data(self):
        """Cleanup old alert history and notifications."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=30)
            
            # Cleanup alert history
            self.alert_history = [
                alert for alert in self.alert_history 
                if alert.started_at > cutoff_time
            ]
            
            # Cleanup notification history
            self.notification_history = [
                notification for notification in self.notification_history
                if notification['timestamp'] > cutoff_time
            ]
            
            # Cleanup expired silences
            expired_silences = []
            for silence_id, silence in self.silences.items():
                if current_time > silence.ends_at:
                    expired_silences.append(silence_id)
            
            for silence_id in expired_silences:
                del self.silences[silence_id]
            
            # Cleanup resolved incidents older than 7 days
            old_incidents = []
            incident_cutoff = current_time - timedelta(days=7)
            for incident_id, incident in self.incidents.items():
                if (incident.status == "resolved" and 
                    incident.updated_at < incident_cutoff):
                    old_incidents.append(incident_id)
            
            for incident_id in old_incidents:
                del self.incidents[incident_id]
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule."""
        try:
            self.alert_rules[rule.name] = rule
            self.logger.info(f"Alert rule {rule.name} added")
            
        except Exception as e:
            self.logger.error(f"Failed to add alert rule: {e}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule."""
        try:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                self.logger.info(f"Alert rule {rule_name} removed")
            
        except Exception as e:
            self.logger.error(f"Failed to remove alert rule: {e}")
    
    def create_silence(self, matchers: List[Dict[str, str]], duration: timedelta, 
                      created_by: str, comment: str = "") -> str:
        """Create alert silence."""
        try:
            silence_id = f"silence-{int(time.time())}"
            current_time = datetime.now()
            
            silence = Silence(
                id=silence_id,
                matchers=matchers,
                starts_at=current_time,
                ends_at=current_time + duration,
                created_by=created_by,
                comment=comment
            )
            
            self.silences[silence_id] = silence
            self.logger.info(f"Silence {silence_id} created")
            
            return silence_id
            
        except Exception as e:
            self.logger.error(f"Failed to create silence: {e}")
            return ""
    
    def remove_silence(self, silence_id: str):
        """Remove alert silence."""
        try:
            if silence_id in self.silences:
                del self.silences[silence_id]
                self.logger.info(f"Silence {silence_id} removed")
            
        except Exception as e:
            self.logger.error(f"Failed to remove silence: {e}")
    
    def get_alert_manager_summary(self) -> Dict[str, Any]:
        """Get alert manager summary."""
        try:
            active_by_severity = {}
            for alert in self.active_alerts.values():
                severity = alert.severity.value
                active_by_severity[severity] = active_by_severity.get(severity, 0) + 1
            
            return {
                'status': 'active' if self.alert_manager_active else 'inactive',
                'alert_rules': {
                    'total': len(self.alert_rules),
                    'enabled': sum(1 for rule in self.alert_rules.values() if rule.enabled)
                },
                'active_alerts': {
                    'total': len(self.active_alerts),
                    'by_severity': active_by_severity,
                    'firing': sum(1 for alert in self.active_alerts.values() if alert.state == AlertState.FIRING),
                    'silenced': sum(1 for alert in self.active_alerts.values() if alert.state == AlertState.SILENCED)
                },
                'silences': {
                    'total': len(self.silences),
                    'active': sum(1 for s in self.silences.values() if s.active)
                },
                'notifications': {
                    'configurations': len(self.notification_configs),
                    'history_count': len(self.notification_history),
                    'recent_failures': sum(1 for n in self.notification_history[-100:] if not n.get('success', True))
                },
                'incidents': {
                    'total': len(self.incidents),
                    'open': sum(1 for i in self.incidents.values() if i.status == "open"),
                    'resolved': sum(1 for i in self.incidents.values() if i.status == "resolved")
                },
                'escalation_rules': len(self.escalation_rules)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate alert manager summary: {e}")
            return {'error': 'Unable to generate summary'}