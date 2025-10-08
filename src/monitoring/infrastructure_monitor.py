"""
Infrastructure Monitoring System with Email Notifications
========================================================

This module implements comprehensive system monitoring with automated
email reports and real-time alerting capabilities.

Status: HIGH PRIORITY - Production operations requirement
Timezone: Australia/Sydney (AEDT/AEST automatic handling)
Tax Compliance: ATO compliant logging enabled
"""

import asyncio
import logging
import smtplib
import sqlite3
import psutil
import json
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import socket
from pathlib import Path

import aiofiles
from jinja2 import Template

# Import Australian timezone and tax compliance
try:
    from src.compliance.australian_timezone_tax import (
        australian_tz_manager, aus_now, log_system_action
    )
    AUSTRALIAN_COMPLIANCE_ENABLED = True
    aus_now_func = aus_now
    log_system_action_func = log_system_action
except ImportError:
    AUSTRALIAN_COMPLIANCE_ENABLED = False
    aus_now_func = None
    log_system_action_func = None


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """System metric types"""
    SYSTEM = "system"
    API = "api"
    TRADING = "trading"
    DATABASE = "database"
    NETWORK = "network"


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ApiMetrics:
    """API performance metrics"""
    timestamp: datetime
    endpoint: str
    response_time_ms: float
    status_code: int
    error_count: int
    request_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradingMetrics:
    """Trading system metrics"""
    timestamp: datetime
    active_strategies: int
    total_orders: int
    filled_orders: int
    failed_orders: int
    total_pnl: Decimal
    daily_pnl: Decimal
    reconciliation_discrepancies: int
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['total_pnl'] = float(self.total_pnl)
        data['daily_pnl'] = float(self.daily_pnl)
        return data


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    metric_type: MetricType
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['level'] = self.level.value
        data['metric_type'] = self.metric_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class InfrastructureMonitor:
    """
    Comprehensive infrastructure monitoring system.
    
    Features:
    - Real-time system metrics collection
    - API performance monitoring  
    - Trading system monitoring
    - Automated alerting with thresholds
    - Email notifications and daily reports
    - Historical data storage and analysis
    """
    
    def __init__(self, db_path: str = "data/trading_bot.db", 
                 config_path: str = "config/monitoring_config.yaml"):
        
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.config_path = config_path
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Metrics storage
        self.system_metrics: List[SystemMetrics] = []
        self.api_metrics: List[ApiMetrics] = []
        self.trading_metrics: List[TradingMetrics] = []
        self.alerts: List[Alert] = []
        
        # Configuration
        self.config = self._load_config()
        
        # Email settings
        self.email_config = self.config.get('email', {})
        
        # Alert thresholds
        self.thresholds = self.config.get('thresholds', {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'api_response_time_ms': 1000.0,
            'error_rate_percent': 5.0
        })
        
        # Monitoring intervals
        self.intervals = self.config.get('intervals', {
            'system_metrics': 60,  # 1 minute
            'api_metrics': 300,    # 5 minutes  
            'trading_metrics': 300, # 5 minutes
            'daily_report': 86400   # 24 hours
        })
        
        # Australian timezone configuration
        self.timezone = self.config.get('timezone', 'Australia/Sydney')
        
        self._initialize_database()
        
        # Log monitoring system initialization for tax compliance
        if AUSTRALIAN_COMPLIANCE_ENABLED and log_system_action_func:
            log_system_action_func(
                f"Monitoring system initialized - timezone: {self.timezone}, DB: {self.db_path}",
                'monitoring_system'
            )
        
        self.logger.info(f"✅ Infrastructure Monitor initialized with timezone: {self.timezone}")
    
    def _get_australian_now(self) -> datetime:
        """Get current time in Australian timezone"""
        if AUSTRALIAN_COMPLIANCE_ENABLED and aus_now_func:
            return aus_now_func()
        
        # Fallback if Australian compliance module not available
        import pytz
        aus_tz = pytz.timezone('Australia/Sydney')
        return datetime.now(aus_tz)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        try:
            if os.path.exists(self.config_path):
                import yaml
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            else:
                # Return default configuration
                return {
                    'email': {
                        'smtp_server': 'smtp.gmail.com',
                        'smtp_port': 587,
                        'username': os.getenv('SMTP_USERNAME', ''),
                        'password': os.getenv('SMTP_PASSWORD', ''),
                        'from_email': os.getenv('FROM_EMAIL', 'bot@trading.local'),
                        'to_emails': [os.getenv('ALERT_EMAIL', 'admin@trading.local')]
                    },
                    'thresholds': {
                        'cpu_percent': 80.0,
                        'memory_percent': 85.0,
                        'disk_percent': 90.0,
                        'api_response_time_ms': 1000.0,
                        'error_rate_percent': 5.0
                    },
                    'intervals': {
                        'system_metrics': 60,
                        'api_metrics': 300,
                        'trading_metrics': 300,
                        'daily_report': 86400
                    }
                }
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {e}")
            return {}
    
    def _initialize_database(self):
        """Initialize monitoring database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    memory_used_gb REAL NOT NULL,
                    memory_total_gb REAL NOT NULL,
                    disk_percent REAL NOT NULL,
                    disk_used_gb REAL NOT NULL,
                    disk_total_gb REAL NOT NULL,
                    network_bytes_sent INTEGER NOT NULL,
                    network_bytes_recv INTEGER NOT NULL,
                    active_connections INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # API metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    endpoint TEXT NOT NULL,
                    response_time_ms REAL NOT NULL,
                    status_code INTEGER NOT NULL,
                    error_count INTEGER NOT NULL,
                    request_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Trading metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    active_strategies INTEGER NOT NULL,
                    total_orders INTEGER NOT NULL,
                    filled_orders INTEGER NOT NULL,
                    failed_orders INTEGER NOT NULL,
                    total_pnl DECIMAL NOT NULL,
                    daily_pnl DECIMAL NOT NULL,
                    reconciliation_discrepancies INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    level TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL,
                    threshold REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Email logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS email_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    email_type TEXT NOT NULL,
                    recipients TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Monitoring database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization error: {e}")
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_monitoring:
            self.logger.warning("⚠️ Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        # Schedule daily report
        asyncio.create_task(self._daily_report_scheduler())
        
        self.logger.info("✅ Infrastructure monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_monitoring = False
        
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("🔄 Infrastructure monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("🔄 Starting monitoring loop")
        
        while self.is_monitoring:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check for alerts
                await self._check_alert_conditions()
                
                # Wait for next collection
                await asyncio.sleep(self.intervals['system_metrics'])
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _collect_system_metrics(self):
        """Collect current system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Connection count
            connections = len(psutil.net_connections())
            
            # Create metrics object
            metrics = SystemMetrics(
                timestamp=self._get_australian_now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                active_connections=connections
            )
            
            # Store metrics
            self.system_metrics.append(metrics)
            await self._store_system_metrics(metrics)
            
            # Keep only last 1000 metrics in memory
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-1000:]
            
        except Exception as e:
            self.logger.error(f"❌ System metrics collection error: {e}")
    
    async def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_metrics 
                (timestamp, cpu_percent, memory_percent, memory_used_gb, memory_total_gb,
                 disk_percent, disk_used_gb, disk_total_gb, network_bytes_sent, 
                 network_bytes_recv, active_connections)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp,
                metrics.cpu_percent,
                metrics.memory_percent,
                metrics.memory_used_gb,
                metrics.memory_total_gb,
                metrics.disk_percent,
                metrics.disk_used_gb,
                metrics.disk_total_gb,
                metrics.network_bytes_sent,
                metrics.network_bytes_recv,
                metrics.active_connections
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error storing system metrics: {e}")
    
    async def _check_alert_conditions(self):
        """Check for alert conditions based on thresholds"""
        try:
            if not self.system_metrics:
                return
            
            latest_metrics = self.system_metrics[-1]
            
            # Check CPU threshold
            if latest_metrics.cpu_percent > self.thresholds['cpu_percent']:
                await self._create_alert(
                    level=AlertLevel.WARNING,
                    metric_type=MetricType.SYSTEM,
                    message=f"High CPU usage: {latest_metrics.cpu_percent:.1f}%",
                    value=latest_metrics.cpu_percent,
                    threshold=self.thresholds['cpu_percent']
                )
            
            # Check memory threshold
            if latest_metrics.memory_percent > self.thresholds['memory_percent']:
                await self._create_alert(
                    level=AlertLevel.WARNING,
                    metric_type=MetricType.SYSTEM,
                    message=f"High memory usage: {latest_metrics.memory_percent:.1f}%",
                    value=latest_metrics.memory_percent,
                    threshold=self.thresholds['memory_percent']
                )
            
            # Check disk threshold
            if latest_metrics.disk_percent > self.thresholds['disk_percent']:
                await self._create_alert(
                    level=AlertLevel.CRITICAL,
                    metric_type=MetricType.SYSTEM,
                    message=f"High disk usage: {latest_metrics.disk_percent:.1f}%",
                    value=latest_metrics.disk_percent,
                    threshold=self.thresholds['disk_percent']
                )
            
        except Exception as e:
            self.logger.error(f"❌ Alert checking error: {e}")
    
    async def _create_alert(self, 
                          level: AlertLevel,
                          metric_type: MetricType,
                          message: str,
                          value: Optional[float] = None,
                          threshold: Optional[float] = None):
        """Create and process an alert"""
        try:
            alert_id = f"{metric_type.value}_{level.value}_{int(time.time())}"
            
            alert = Alert(
                alert_id=alert_id,
                timestamp=self._get_australian_now(),
                level=level,
                metric_type=metric_type,
                message=message,
                value=value,
                threshold=threshold
            )
            
            # Store alert
            self.alerts.append(alert)
            await self._store_alert(alert)
            
            # Send email notification for critical alerts
            if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                await self._send_alert_email(alert)
            
            self.logger.warning(f"⚠️ Alert created: {message}")
            
        except Exception as e:
            self.logger.error(f"❌ Alert creation error: {e}")
    
    async def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO monitoring_alerts
                (alert_id, timestamp, level, metric_type, message, value, threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.timestamp,
                alert.level.value,
                alert.metric_type.value,
                alert.message,
                alert.value,
                alert.threshold
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error storing alert: {e}")
    
    async def _send_alert_email(self, alert: Alert):
        """Send email notification for alert"""
        try:
            if not self.email_config.get('username') or not self.email_config.get('password'):
                self.logger.warning("⚠️ Email not configured - skipping alert email")
                return
            
            subject = f"🚨 {alert.level.value.upper()} Alert: {alert.metric_type.value}"
            
            body = f"""
            TRADING BOT ALERT
            ================
            
            Alert Level: {alert.level.value.upper()}
            Metric Type: {alert.metric_type.value}
            Message: {alert.message}
            
            Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            Value: {alert.value}
            Threshold: {alert.threshold}
            
            System Information:
            - Server: {socket.gethostname()}
            - Alert ID: {alert.alert_id}
            
            Please investigate immediately if this is a critical alert.
            
            Trading Bot Monitoring System
            """
            
            await self._send_email(
                subject=subject,
                body=body,
                email_type="alert"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Alert email error: {e}")
    
    async def _daily_report_scheduler(self):
        """Schedule daily reports"""
        while self.is_monitoring:
            try:
                # Calculate next report time (daily at 8 AM AEST/AEDT)
                now = self._get_australian_now()
                next_report = now.replace(hour=8, minute=0, second=0, microsecond=0)  # 8 AM Australian time
                
                if next_report <= now:
                    next_report += timedelta(days=1)
                
                # Wait until next report time
                wait_seconds = (next_report - now).total_seconds()
                await asyncio.sleep(wait_seconds)
                
                # Send daily report
                await self._send_daily_report()
                
            except Exception as e:
                self.logger.error(f"❌ Daily report scheduler error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _send_daily_report(self):
        """Generate and send daily performance report"""
        try:
            if not self.email_config.get('username') or not self.email_config.get('password'):
                self.logger.warning("⚠️ Email not configured - skipping daily report")
                return
            
            # Generate report data
            report_data = await self._generate_daily_report_data()
            
            # Create HTML report
            html_report = await self._create_html_report(report_data)
            
            # Send email
            subject = f"📊 Daily Trading Bot Report - {self._get_australian_now().strftime('%Y-%m-%d')}"
            
            await self._send_email(
                subject=subject,
                body=html_report,
                email_type="daily_report",
                is_html=True
            )
            
            self.logger.info("✅ Daily report sent successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Daily report error: {e}")
    
    async def _generate_daily_report_data(self) -> Dict[str, Any]:
        """Generate data for daily report"""
        try:
            # Get 24-hour metrics
            end_time = self._get_australian_now()
            start_time = end_time - timedelta(hours=24)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # System metrics summary
            cursor.execute('''
                SELECT 
                    AVG(cpu_percent) as avg_cpu,
                    MAX(cpu_percent) as max_cpu,
                    AVG(memory_percent) as avg_memory,
                    MAX(memory_percent) as max_memory,
                    AVG(disk_percent) as avg_disk,
                    COUNT(*) as metric_count
                FROM system_metrics 
                WHERE timestamp BETWEEN ? AND ?
            ''', (start_time, end_time))
            
            system_stats = cursor.fetchone()
            
            # Alert summary
            cursor.execute('''
                SELECT level, COUNT(*) as count
                FROM monitoring_alerts 
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY level
            ''', (start_time, end_time))
            
            alert_stats = dict(cursor.fetchall())
            
            # Trading metrics (if available)
            cursor.execute('''
                SELECT 
                    SUM(total_orders) as orders,
                    SUM(filled_orders) as filled,
                    AVG(total_pnl) as avg_pnl,
                    SUM(reconciliation_discrepancies) as discrepancies
                FROM trading_metrics 
                WHERE timestamp BETWEEN ? AND ?
            ''', (start_time, end_time))
            
            trading_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'report_date': end_time.strftime('%Y-%m-%d'),
                'period': '24 hours',
                'system': {
                    'avg_cpu': system_stats[0] or 0,
                    'max_cpu': system_stats[1] or 0,
                    'avg_memory': system_stats[2] or 0,
                    'max_memory': system_stats[3] or 0,
                    'avg_disk': system_stats[4] or 0,
                    'metric_count': system_stats[5] or 0
                },
                'alerts': alert_stats,
                'trading': {
                    'total_orders': trading_stats[0] or 0,
                    'filled_orders': trading_stats[1] or 0,
                    'avg_pnl': trading_stats[2] or 0,
                    'discrepancies': trading_stats[3] or 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Report data generation error: {e}")
            return {}
    
    async def _create_html_report(self, data: Dict[str, Any]) -> str:
        """Create HTML daily report"""
        try:
            template = Template('''
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; }
        .alert-critical { background: #dc3545; color: white; }
        .alert-warning { background: #ffc107; color: black; }
        .good { color: #28a745; }
        .warning { color: #fd7e14; }
        .critical { color: #dc3545; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Trading Bot Daily Report</h1>
        <p>{{ data.report_date }} - {{ data.period }}</p>
    </div>
    
    <div class="section">
        <h2>📊 System Performance</h2>
        <div class="metric">
            <strong>CPU Usage</strong><br>
            Average: {{ "%.1f"|format(data.system.avg_cpu) }}%<br>
            Peak: {{ "%.1f"|format(data.system.max_cpu) }}%
        </div>
        <div class="metric">
            <strong>Memory Usage</strong><br>
            Average: {{ "%.1f"|format(data.system.avg_memory) }}%<br>
            Peak: {{ "%.1f"|format(data.system.max_memory) }}%
        </div>
        <div class="metric">
            <strong>Disk Usage</strong><br>
            Average: {{ "%.1f"|format(data.system.avg_disk) }}%
        </div>
    </div>
    
    <div class="section">
        <h2>🚨 Alerts Summary</h2>
        {% if data.alerts %}
            {% for level, count in data.alerts.items() %}
            <div class="metric alert-{{ level }}">
                <strong>{{ level|title }}</strong><br>
                {{ count }} alerts
            </div>
            {% endfor %}
        {% else %}
            <p class="good">✅ No alerts in the past 24 hours</p>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>💼 Trading Summary</h2>
        <div class="metric">
            <strong>Orders</strong><br>
            Total: {{ data.trading.total_orders }}<br>
            Filled: {{ data.trading.filled_orders }}
        </div>
        <div class="metric">
            <strong>P&L</strong><br>
            Average: ${{ "%.2f"|format(data.trading.avg_pnl) }}
        </div>
        <div class="metric">
            <strong>Data Integrity</strong><br>
            Discrepancies: {{ data.trading.discrepancies }}
        </div>
    </div>
    
    <div class="section">
        <h2>🎯 System Health</h2>
        {% set cpu_status = "good" if data.system.max_cpu < 80 else "warning" if data.system.max_cpu < 95 else "critical" %}
        {% set memory_status = "good" if data.system.max_memory < 85 else "warning" if data.system.max_memory < 95 else "critical" %}
        
        <p class="{{ cpu_status }}">CPU: {{ "✅ Normal" if cpu_status == "good" else "⚠️ High Usage" if cpu_status == "warning" else "🚨 Critical" }}</p>
        <p class="{{ memory_status }}">Memory: {{ "✅ Normal" if memory_status == "good" else "⚠️ High Usage" if memory_status == "warning" else "🚨 Critical" }}</p>
        <p class="good">Database: {{ data.system.metric_count }} metrics collected</p>
    </div>
    
    <div class="section">
        <p><em>Generated by Trading Bot Monitoring System at {{ data.report_date }}</em></p>
    </div>
</body>
</html>
            ''')
            
            return template.render(data=data)
            
        except Exception as e:
            self.logger.error(f"❌ HTML report creation error: {e}")
            return f"Error creating report: {e}"
    
    async def _send_email(self, 
                         subject: str, 
                         body: str, 
                         email_type: str,
                         is_html: bool = False):
        """Send email notification"""
        try:
            if not self.email_config.get('to_emails'):
                self.logger.warning("⚠️ No email recipients configured")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])
            msg['Subject'] = subject
            
            # Add body
            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], self.email_config['to_emails'], text)
            server.quit()
            
            # Log email
            await self._log_email(email_type, subject, "sent", None)
            
            self.logger.info(f"✅ Email sent: {subject}")
            
        except Exception as e:
            await self._log_email(email_type, subject, "failed", str(e))
            self.logger.error(f"❌ Email sending error: {e}")
    
    async def _log_email(self, email_type: str, subject: str, status: str, error: Optional[str]):
        """Log email sending attempt"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO email_logs
                (timestamp, email_type, recipients, subject, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self._get_australian_now(),
                email_type,
                ', '.join(self.email_config.get('to_emails', [])),
                subject,
                status,
                error
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Email logging error: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        if not self.system_metrics:
            return {"status": "no_data"}
        
        latest = self.system_metrics[-1]
        return {
            "timestamp": latest.timestamp.isoformat(),
            "system": latest.to_dict(),
            "alerts": len([a for a in self.alerts if not a.resolved]),
            "monitoring_active": self.is_monitoring
        }
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        recent_alerts = [a for a in self.alerts if not a.resolved and 
                        (self._get_australian_now() - a.timestamp).total_seconds() < 3600]  # Last hour
        
        return {
            "total_alerts": len(self.alerts),
            "recent_alerts": len(recent_alerts),
            "alert_levels": {
                level.value: len([a for a in recent_alerts if a.level == level])
                for level in AlertLevel
            },
            "latest_alerts": [a.to_dict() for a in recent_alerts[-5:]]  # Last 5 alerts
        }


# Factory function for easy integration
def create_infrastructure_monitor(db_path: str = "data/trading_bot.db",
                                config_path: str = "config/monitoring_config.yaml") -> InfrastructureMonitor:
    """Factory function to create infrastructure monitor"""
    return InfrastructureMonitor(db_path=db_path, config_path=config_path)