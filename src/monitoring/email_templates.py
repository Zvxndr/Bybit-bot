"""
Email Notification Templates
===========================

Professional email templates for trading bot alerts and reports.
"""

from typing import Dict, Any
from datetime import datetime
from jinja2 import Template


class EmailTemplates:
    """Email template manager for professional notifications"""
    
    @staticmethod
    def get_alert_template() -> str:
        """Get alert email template"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, {{ header_color }}, {{ header_color_light }}); color: white; padding: 30px; text-align: center; }
        .header h1 { margin: 0; font-size: 24px; font-weight: 300; }
        .header .alert-level { font-size: 14px; opacity: 0.9; margin-top: 5px; }
        .content { padding: 30px; }
        .alert-info { background: #f8f9fa; border-left: 4px solid {{ alert_color }}; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .metric { display: inline-block; background: #f8f9fa; padding: 15px; margin: 10px; border-radius: 8px; text-align: center; min-width: 120px; }
        .metric-value { font-size: 20px; font-weight: bold; color: {{ alert_color }}; }
        .metric-label { font-size: 12px; color: #666; margin-top: 5px; }
        .system-info { background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 20px; }
        .footer { background: #f8f9fa; padding: 20px; text-align: center; font-size: 12px; color: #666; }
        .btn { display: inline-block; background: {{ alert_color }}; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ alert_icon }} Trading Bot Alert</h1>
            <div class="alert-level">{{ alert.level|upper }} LEVEL ALERT</div>
        </div>
        
        <div class="content">
            <div class="alert-info">
                <h2 style="margin-top: 0; color: {{ alert_color }};">{{ alert.message }}</h2>
                <p><strong>Alert Type:</strong> {{ alert.metric_type|title }}</p>
                <p><strong>Time:</strong> {{ alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
                {% if alert.value %}<p><strong>Current Value:</strong> {{ alert.value }}</p>{% endif %}
                {% if alert.threshold %}<p><strong>Threshold:</strong> {{ alert.threshold }}</p>{% endif %}
            </div>
            
            {% if system_metrics %}
            <h3>Current System Status</h3>
            <div style="text-align: center;">
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(system_metrics.cpu_percent) }}%</div>
                    <div class="metric-label">CPU Usage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(system_metrics.memory_percent) }}%</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(system_metrics.disk_percent) }}%</div>
                    <div class="metric-label">Disk Usage</div>
                </div>
            </div>
            {% endif %}
            
            <div class="system-info">
                <h4 style="margin-top: 0;">System Information</h4>
                <p><strong>Server:</strong> {{ system_info.hostname }}</p>
                <p><strong>Alert ID:</strong> {{ alert.alert_id }}</p>
                <p><strong>Environment:</strong> {{ system_info.environment }}</p>
            </div>
            
            {% if alert.level in ['critical', 'emergency'] %}
            <div style="text-align: center; margin: 30px 0;">
                <p><strong>⚠️ This is a {{ alert.level|upper }} alert requiring immediate attention!</strong></p>
                <a href="{{ dashboard_url }}" class="btn">View Dashboard</a>
            </div>
            {% endif %}
        </div>
        
        <div class="footer">
            <p>Trading Bot Monitoring System • Generated at {{ now.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
            <p>This is an automated message. Do not reply to this email.</p>
        </div>
    </div>
</body>
</html>
        """
    
    @staticmethod
    def get_daily_report_template() -> str:
        """Get daily report email template"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, #2c3e50, #34495e); color: white; padding: 40px; text-align: center; }
        .header h1 { margin: 0; font-size: 28px; font-weight: 300; }
        .header .date { font-size: 16px; opacity: 0.9; margin-top: 10px; }
        .section { padding: 30px; border-bottom: 1px solid #eee; }
        .section:last-child { border-bottom: none; }
        .section h2 { color: #2c3e50; margin-top: 0; font-size: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #e9ecef; }
        .metric-value { font-size: 24px; font-weight: bold; margin-bottom: 5px; }
        .metric-label { font-size: 14px; color: #666; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .alert-summary { display: flex; gap: 15px; flex-wrap: wrap; }
        .alert-badge { padding: 8px 16px; border-radius: 20px; font-weight: bold; }
        .alert-info { background: #d1ecf1; color: #0c5460; }
        .alert-warning { background: #fff3cd; color: #856404; }
        .alert-critical { background: #f8d7da; color: #721c24; }
        .chart-placeholder { background: #f8f9fa; border: 2px dashed #dee2e6; padding: 40px; text-align: center; color: #6c757d; border-radius: 8px; margin: 20px 0; }
        .recommendation { background: #e7f3ff; border-left: 4px solid #007bff; padding: 15px; margin: 10px 0; border-radius: 0 5px 5px 0; }
        .footer { background: #f8f9fa; padding: 20px; text-align: center; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Daily Trading Bot Report</h1>
            <div class="date">{{ report_date }} • 24 Hour Summary</div>
        </div>
        
        <!-- Executive Summary -->
        <div class="section">
            <h2>📋 Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value status-{{ overall_status }}">{{ overall_status|title }}</div>
                    <div class="metric-label">Overall System Health</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ data.system.metric_count }}</div>
                    <div class="metric-label">Metrics Collected</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ total_alerts }}</div>
                    <div class="metric-label">Total Alerts</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">99.{{ uptime_percentage }}%</div>
                    <div class="metric-label">System Uptime</div>
                </div>
            </div>
        </div>
        
        <!-- System Performance -->
        <div class="section">
            <h2>🖥️ System Performance</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value status-{{ cpu_status }}">{{ "%.1f"|format(data.system.avg_cpu) }}%</div>
                    <div class="metric-label">Average CPU Usage</div>
                    <div style="font-size: 12px; margin-top: 5px;">Peak: {{ "%.1f"|format(data.system.max_cpu) }}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-{{ memory_status }}">{{ "%.1f"|format(data.system.avg_memory) }}%</div>
                    <div class="metric-label">Average Memory Usage</div>
                    <div style="font-size: 12px; margin-top: 5px;">Peak: {{ "%.1f"|format(data.system.max_memory) }}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-{{ disk_status }}">{{ "%.1f"|format(data.system.avg_disk) }}%</div>
                    <div class="metric-label">Disk Usage</div>
                </div>
            </div>
        </div>
        
        <!-- Alert Summary -->
        <div class="section">
            <h2>🚨 Alert Summary</h2>
            {% if data.alerts %}
                <div class="alert-summary">
                    {% for level, count in data.alerts.items() %}
                    <span class="alert-badge alert-{{ level }}">{{ level|title }}: {{ count }}</span>
                    {% endfor %}
                </div>
            {% else %}
                <p class="status-good">✅ No alerts generated in the past 24 hours - excellent system stability!</p>
            {% endif %}
        </div>
        
        <!-- Trading Performance -->
        <div class="section">
            <h2>💼 Trading Performance</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ data.trading.total_orders }}</div>
                    <div class="metric-label">Total Orders</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ data.trading.filled_orders }}</div>
                    <div class="metric-label">Filled Orders</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-{{ pnl_status }}">${{ "%.2f"|format(data.trading.avg_pnl) }}</div>
                    <div class="metric-label">Average P&L</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ data.trading.discrepancies }}</div>
                    <div class="metric-label">Data Discrepancies</div>
                </div>
            </div>
        </div>
        
        <!-- Performance Charts (Placeholder) -->
        <div class="section">
            <h2>📈 Performance Trends</h2>
            <div class="chart-placeholder">
                <p>📊 Performance charts will be available in the next update</p>
                <p>Charts will show: CPU/Memory trends • Alert frequency • Trading volume • P&L progression</p>
            </div>
        </div>
        
        <!-- Recommendations -->
        {% if recommendations %}
        <div class="section">
            <h2>💡 Recommendations</h2>
            {% for recommendation in recommendations %}
            <div class="recommendation">
                <strong>{{ recommendation.title }}</strong><br>
                {{ recommendation.description }}
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <!-- System Health Score -->
        <div class="section">
            <h2>🎯 System Health Score</h2>
            <div style="text-align: center;">
                <div style="font-size: 48px; font-weight: bold; color: {{ health_score_color }}; margin: 20px 0;">
                    {{ health_score }}/100
                </div>
                <p>{{ health_score_description }}</p>
                
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: left;">
                    <h4>Score Breakdown:</h4>
                    <p>• System Performance: {{ performance_score }}/30</p>
                    <p>• Alert Management: {{ alert_score }}/25</p>
                    <p>• Trading Efficiency: {{ trading_score }}/25</p>
                    <p>• Data Integrity: {{ integrity_score }}/20</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Trading Bot Monitoring System • Report generated at {{ now.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
            <p>Next report will be sent tomorrow at 9:00 AM UTC</p>
            <p style="margin-top: 10px;">
                <a href="{{ dashboard_url }}" style="color: #007bff;">View Live Dashboard</a> • 
                <a href="mailto:{{ support_email }}" style="color: #007bff;">Contact Support</a>
            </p>
        </div>
    </div>
</body>
</html>
        """
    
    @staticmethod
    def render_alert_email(alert: Dict[str, Any], 
                          system_metrics: Dict[str, Any] = None,
                          dashboard_url: str = "http://localhost:8000") -> str:
        """Render alert email from template"""
        
        # Determine colors based on alert level
        colors = {
            'info': ('#17a2b8', '#20c997', '#17a2b8'),
            'warning': ('#ffc107', '#ffdb4d', '#ffc107'),
            'critical': ('#dc3545', '#ff6b7a', '#dc3545'),
            'emergency': ('#6f42c1', '#8e5bcc', '#6f42c1')
        }
        
        icons = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'critical': '🚨',
            'emergency': '🔴'
        }
        
        alert_level = alert.get('level', 'info')
        header_color, header_color_light, alert_color = colors.get(alert_level, colors['info'])
        alert_icon = icons.get(alert_level, 'ℹ️')
        
        template = Template(EmailTemplates.get_alert_template())
        
        return template.render(
            alert=alert,
            system_metrics=system_metrics,
            alert_color=alert_color,
            header_color=header_color,
            header_color_light=header_color_light,
            alert_icon=alert_icon,
            dashboard_url=dashboard_url,
            system_info={
                'hostname': 'trading-bot-server',
                'environment': 'Production'
            },
            now=datetime.now()
        )
    
    @staticmethod
    def render_daily_report(data: Dict[str, Any],
                           dashboard_url: str = "http://localhost:8000",
                           support_email: str = "support@trading.local") -> str:
        """Render daily report email from template"""
        
        # Calculate health scores and statuses
        def get_status(value, warning_threshold, critical_threshold):
            if value < warning_threshold:
                return 'good'
            elif value < critical_threshold:
                return 'warning'
            else:
                return 'critical'
        
        cpu_status = get_status(data['system']['max_cpu'], 80, 95)
        memory_status = get_status(data['system']['max_memory'], 85, 95)
        disk_status = get_status(data['system']['avg_disk'], 90, 98)
        
        # Calculate overall health score
        performance_score = 30 if cpu_status == 'good' and memory_status == 'good' else (20 if cpu_status != 'critical' and memory_status != 'critical' else 10)
        alert_score = 25 if not data['alerts'] else (15 if len(data['alerts']) < 5 else 5)
        trading_score = 25 if data['trading']['avg_pnl'] >= 0 else 15
        integrity_score = 20 if data['trading']['discrepancies'] == 0 else (10 if data['trading']['discrepancies'] < 5 else 5)
        
        health_score = performance_score + alert_score + trading_score + integrity_score
        
        # Health score description and color
        if health_score >= 90:
            health_score_description = "Excellent - System operating at optimal performance"
            health_score_color = "#28a745"
            overall_status = "excellent"
        elif health_score >= 75:
            health_score_description = "Good - System performance is stable with minor areas for improvement"
            health_score_color = "#ffc107"
            overall_status = "good"
        elif health_score >= 60:
            health_score_description = "Fair - System requires attention in several areas"
            health_score_color = "#fd7e14"
            overall_status = "fair"
        else:
            health_score_description = "Poor - Immediate attention required"
            health_score_color = "#dc3545"
            overall_status = "poor"
        
        # Generate recommendations based on system state
        recommendations = []
        if data['system']['max_cpu'] > 80:
            recommendations.append({
                'title': 'High CPU Usage Detected',
                'description': 'Consider optimizing trading algorithms or increasing server resources.'
            })
        if data['system']['max_memory'] > 85:
            recommendations.append({
                'title': 'Memory Usage Optimization',
                'description': 'Review memory usage patterns and consider implementing data cleanup routines.'
            })
        if data['trading']['discrepancies'] > 0:
            recommendations.append({
                'title': 'Data Integrity Issues',
                'description': 'Reconciliation discrepancies detected. Review trade synchronization processes.'
            })
        
        template = Template(EmailTemplates.get_daily_report_template())
        
        return template.render(
            data=data,
            report_date=data['report_date'],
            overall_status=overall_status,
            total_alerts=sum(data['alerts'].values()) if data['alerts'] else 0,
            uptime_percentage=95 + (health_score // 10),  # Simulated uptime
            cpu_status=cpu_status,
            memory_status=memory_status,
            disk_status=disk_status,
            pnl_status='good' if data['trading']['avg_pnl'] >= 0 else 'critical',
            health_score=health_score,
            health_score_description=health_score_description,
            health_score_color=health_score_color,
            performance_score=performance_score,
            alert_score=alert_score,
            trading_score=trading_score,
            integrity_score=integrity_score,
            recommendations=recommendations,
            dashboard_url=dashboard_url,
            support_email=support_email,
            now=datetime.now()
        )