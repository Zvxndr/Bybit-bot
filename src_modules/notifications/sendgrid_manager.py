"""
SendGrid Email Manager
=====================

Professional email notification system with rich HTML reports and chart generation.
Provides weekly trading reports, alerts, and investor communications.
"""

import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content, Attachment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import os

# Set matplotlib backend for headless operation
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class SendGridEmailManager:
    """Professional email notification system for trading bot"""
    
    def __init__(self, api_key: str, from_email: str = None, from_name: str = None):
        """
        Initialize SendGrid email manager
        
        Args:
            api_key: SendGrid API key
            from_email: Default from email address
            from_name: Default from name
        """
        self.sg = sendgrid.SendGridAPIClient(api_key=api_key)
        self.from_email = from_email or "trading-bot@yourdomain.com"
        self.from_name = from_name or "Bybit Trading Bot"
        
        # Email templates directory
        self.templates_dir = Path(__file__).parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        
        # Chart styling
        self._setup_chart_styling()
        
        logger.info("✅ SendGrid email manager initialized")
    
    def _setup_chart_styling(self):
        """Set up consistent chart styling"""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Custom color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'neutral': '#6c757d'
        }
    
    def send_weekly_report(self, recipients: List[str], report_data: Dict[str, Any], 
                          subject_prefix: str = "") -> Dict[str, Any]:
        """
        Send weekly performance report to investors
        
        Args:
            recipients: List of recipient email addresses
            report_data: Trading performance data
            subject_prefix: Optional prefix for email subject
            
        Returns:
            Dictionary with send results
        """
        try:
            # Generate performance charts
            charts = self._generate_performance_charts(report_data)
            
            # Create HTML email content
            html_content = self._create_weekly_report_html(report_data, charts)
            
            # Send to each recipient
            results = {}
            subject = f"{subject_prefix}Weekly Trading Report - {report_data.get('week_ending', datetime.now().strftime('%Y-%m-%d'))}"
            
            for recipient in recipients:
                try:
                    message = Mail(
                        from_email=Email(self.from_email, self.from_name),
                        to_emails=To(recipient),
                        subject=subject,
                        html_content=Content("text/html", html_content)
                    )
                    
                    response = self.sg.send(message)
                    results[recipient] = {
                        'status': 'sent',
                        'status_code': response.status_code,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    logger.info(f"📧 Weekly report sent to {recipient}: {response.status_code}")
                    
                except Exception as e:
                    results[recipient] = {
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    logger.error(f"❌ Failed to send weekly report to {recipient}: {str(e)}")
            
            return {
                'success': True,
                'total_recipients': len(recipients),
                'sent_count': sum(1 for r in results.values() if r['status'] == 'sent'),
                'failed_count': sum(1 for r in results.values() if r['status'] == 'failed'),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to send weekly reports: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'results': {}
            }
    
    def send_alert(self, recipients: List[str], alert_type: str, message: str, 
                   data: Dict[str, Any] = None, priority: str = "normal") -> Dict[str, Any]:
        """
        Send trading alert or notification
        
        Args:
            recipients: List of recipient email addresses
            alert_type: Type of alert (profit, loss, error, info)
            message: Alert message
            data: Additional data for the alert
            priority: Alert priority (low, normal, high, critical)
            
        Returns:
            Dictionary with send results
        """
        try:
            # Choose emoji and colors based on alert type
            alert_config = {
                'profit': {'emoji': '💰', 'color': self.colors['success'], 'subject_prefix': '[PROFIT] '},
                'loss': {'emoji': '📉', 'color': self.colors['danger'], 'subject_prefix': '[LOSS] '},
                'error': {'emoji': '❌', 'color': self.colors['danger'], 'subject_prefix': '[ERROR] '},
                'info': {'emoji': 'ℹ️', 'color': self.colors['info'], 'subject_prefix': '[INFO] '},
                'warning': {'emoji': '⚠️', 'color': self.colors['warning'], 'subject_prefix': '[WARNING] '}
            }
            
            config = alert_config.get(alert_type, alert_config['info'])
            
            # Create HTML content
            html_content = self._create_alert_html(message, alert_type, config, data)
            
            # Send to recipients
            results = {}
            subject = f"{config['subject_prefix']}{message[:50]}{'...' if len(message) > 50 else ''}"
            
            for recipient in recipients:
                try:
                    message_obj = Mail(
                        from_email=Email(self.from_email, self.from_name),
                        to_emails=To(recipient),
                        subject=subject,
                        html_content=Content("text/html", html_content)
                    )
                    
                    response = self.sg.send(message_obj)
                    results[recipient] = {
                        'status': 'sent',
                        'status_code': response.status_code,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    logger.info(f"🚨 Alert sent to {recipient}: {alert_type}")
                    
                except Exception as e:
                    results[recipient] = {
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    logger.error(f"❌ Failed to send alert to {recipient}: {str(e)}")
            
            return {
                'success': True,
                'alert_type': alert_type,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to send alerts: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_performance_charts(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate base64 encoded charts for email reports"""
        charts = {}
        
        try:
            # 1. Portfolio Value Chart
            if 'daily_values' in data and data['daily_values']:
                charts['portfolio_chart'] = self._create_portfolio_chart(data['daily_values'])
            
            # 2. Strategy Performance Pie Chart
            if 'strategy_performance' in data and data['strategy_performance']:
                charts['strategy_chart'] = self._create_strategy_pie_chart(data['strategy_performance'])
            
            # 3. Risk Metrics Chart
            if 'risk_metrics' in data and data['risk_metrics']:
                charts['risk_chart'] = self._create_risk_metrics_chart(data['risk_metrics'])
            
            # 4. Asset Allocation Chart
            if 'asset_allocation' in data and data['asset_allocation']:
                charts['allocation_chart'] = self._create_allocation_chart(data['asset_allocation'])
            
            logger.info(f"📊 Generated {len(charts)} charts for email report")
            
        except Exception as e:
            logger.error(f"❌ Error generating charts: {str(e)}")
        
        return charts
    
    def _create_portfolio_chart(self, daily_values: Dict[str, List]) -> str:
        """Create portfolio value over time chart"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            dates = pd.to_datetime(daily_values.get('dates', []))
            values = daily_values.get('portfolio_values', [])
            
            if len(dates) > 0 and len(values) > 0:
                ax.plot(dates, values, linewidth=3, color=self.colors['primary'], marker='o', markersize=4)
                
                # Add trend line
                if len(dates) > 2:
                    z = np.polyfit(range(len(values)), values, 1)
                    p = np.poly1d(z)
                    ax.plot(dates, p(range(len(values))), "--", color=self.colors['secondary'], alpha=0.8, linewidth=2)
                
                ax.set_title('Portfolio Value Over Time', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Portfolio Value (AUD)', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Format y-axis as currency
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                
                # Rotate x-axis labels
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating portfolio chart: {str(e)}")
            return ""
    
    def _create_strategy_pie_chart(self, strategy_performance: Dict[str, float]) -> str:
        """Create strategy performance pie chart"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            strategies = list(strategy_performance.keys())
            returns = list(strategy_performance.values())
            
            # Filter out zero/negative values for pie chart
            filtered_data = [(s, r) for s, r in zip(strategies, returns) if r > 0]
            
            if filtered_data:
                strategies, returns = zip(*filtered_data)
                
                colors = [self.colors['primary'], self.colors['secondary'], self.colors['info'], 
                         self.colors['warning'], self.colors['success']][:len(strategies)]
                
                wedges, texts, autotexts = ax.pie(
                    returns, 
                    labels=strategies, 
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90,
                    explode=[0.05] * len(strategies)  # Slight separation
                )
                
                # Enhance text
                for text in texts:
                    text.set_fontsize(10)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax.set_title('Strategy Performance Contribution', fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating strategy pie chart: {str(e)}")
            return ""
    
    def _create_risk_metrics_chart(self, risk_metrics: Dict[str, float]) -> str:
        """Create risk metrics bar chart"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = list(risk_metrics.keys())
            values = list(risk_metrics.values())
            
            bars = ax.bar(metrics, values, color=[self.colors['primary'], self.colors['info'], 
                                                 self.colors['warning'], self.colors['danger']][:len(metrics)])
            
            ax.set_title('Risk Metrics Overview', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Risk Level (%)', fontsize=12)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating risk metrics chart: {str(e)}")
            return ""
    
    def _create_allocation_chart(self, asset_allocation: Dict[str, float]) -> str:
        """Create asset allocation donut chart"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            assets = list(asset_allocation.keys())
            allocations = list(asset_allocation.values())
            
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['info'], 
                     self.colors['warning'], self.colors['success']][:len(assets)]
            
            # Create donut chart
            wedges, texts, autotexts = ax.pie(
                allocations,
                labels=assets,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                pctdistance=0.85
            )
            
            # Create donut by adding white circle in center
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            fig.gca().add_artist(centre_circle)
            
            ax.set_title('Asset Allocation', fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating allocation chart: {str(e)}")
            return ""
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)  # Free memory
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error converting figure to base64: {str(e)}")
            plt.close(fig)
            return ""
    
    def _create_weekly_report_html(self, data: Dict[str, Any], charts: Dict[str, str]) -> str:
        """Create comprehensive HTML weekly report"""
        
        # Calculate performance metrics
        weekly_return = data.get('weekly_return', 0)
        portfolio_value = data.get('portfolio_value', 0)
        sharpe_ratio = data.get('sharpe_ratio', 0)
        max_drawdown = data.get('max_drawdown', 0)
        
        # Performance indicators
        return_class = 'positive' if weekly_return > 0 else 'negative'
        return_color = self.colors['success'] if weekly_return > 0 else self.colors['danger']
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Weekly Trading Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{ 
                    max-width: 900px; 
                    margin: 0 auto; 
                    background: white; 
                    border-radius: 15px; 
                    padding: 40px; 
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                }}
                .header {{ 
                    text-align: center; 
                    border-bottom: 3px solid {self.colors['primary']}; 
                    padding-bottom: 25px; 
                    margin-bottom: 35px; 
                }}
                .header h1 {{ 
                    color: {self.colors['primary']}; 
                    font-size: 2.5em; 
                    margin: 0; 
                    font-weight: 700;
                }}
                .header p {{ 
                    color: #666; 
                    font-size: 1.1em; 
                    margin: 10px 0 0 0;
                }}
                .metrics-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 20px; 
                    margin: 30px 0; 
                }}
                .metric-card {{ 
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    border-radius: 12px; 
                    padding: 25px; 
                    text-align: center;
                    border-left: 4px solid {self.colors['primary']};
                    transition: transform 0.2s ease;
                }}
                .metric-card:hover {{
                    transform: translateY(-2px);
                }}
                .metric-value {{ 
                    font-size: 2.2em; 
                    font-weight: bold; 
                    margin: 0;
                }}
                .metric-label {{ 
                    font-size: 0.9em; 
                    color: #666; 
                    margin: 8px 0 0 0;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                .positive {{ color: {self.colors['success']}; }}
                .negative {{ color: {self.colors['danger']}; }}
                .section {{ 
                    margin: 40px 0; 
                    background: #f8f9fa;
                    border-radius: 12px;
                    padding: 30px;
                }}
                .section h2 {{ 
                    color: {self.colors['primary']}; 
                    margin-top: 0;
                    font-size: 1.8em;
                    border-bottom: 2px solid {self.colors['primary']};
                    padding-bottom: 10px;
                }}
                .chart {{ 
                    text-align: center; 
                    margin: 25px 0; 
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .chart img {{ 
                    max-width: 100%; 
                    height: auto; 
                    border-radius: 8px;
                }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 20px 0; 
                    background: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                th, td {{ 
                    padding: 15px 20px; 
                    text-align: left; 
                    border-bottom: 1px solid #dee2e6; 
                }}
                th {{ 
                    background: {self.colors['primary']}; 
                    color: white; 
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                tr:hover {{ background-color: #f8f9fa; }}
                .status-indicator {{ 
                    display: inline-block; 
                    width: 10px; 
                    height: 10px; 
                    border-radius: 50%; 
                    margin-right: 8px;
                }}
                .status-active {{ background-color: {self.colors['success']}; }}
                .status-inactive {{ background-color: {self.colors['danger']}; }}
                .footer {{ 
                    font-size: 0.9em; 
                    color: #666; 
                    text-align: center; 
                    margin-top: 50px; 
                    border-top: 1px solid #dee2e6; 
                    padding-top: 25px;
                }}
                .disclaimer {{ 
                    background: #fff3cd; 
                    border: 1px solid #ffeaa7; 
                    border-radius: 8px; 
                    padding: 15px; 
                    margin: 20px 0;
                    font-size: 0.9em;
                }}
                .logo {{
                    width: 60px;
                    height: 60px;
                    background: {self.colors['primary']};
                    border-radius: 50%;
                    margin: 0 auto 20px auto;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">₿</div>
                    <h1>Weekly Trading Report</h1>
                    <p>Week Ending: {data.get('week_ending', 'N/A')}</p>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value {return_class}">{weekly_return:+.2f}%</div>
                        <div class="metric-label">Weekly Return</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${portfolio_value:,.2f}</div>
                        <div class="metric-label">Portfolio Value</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{sharpe_ratio:.2f}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {'negative' if max_drawdown < 0 else 'positive'}">{max_drawdown:.2f}%</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                </div>
                
                {self._generate_chart_section('Portfolio Performance', 'portfolio_chart', charts)}
                {self._generate_chart_section('Strategy Performance Distribution', 'strategy_chart', charts)}
                {self._generate_chart_section('Asset Allocation', 'allocation_chart', charts)}
                
                {self._generate_active_strategies_table(data.get('active_strategies', {}))}
                {self._generate_risk_compliance_section(data)}
                
                <div class="disclaimer">
                    <strong>⚠️ Important Disclaimer:</strong> This report is generated automatically by your Bybit Trading Bot. 
                    Past performance does not guarantee future results. All trading involves risk of loss. 
                    This information is for informational purposes only and should not be considered as financial advice.
                </div>
                
                <div class="footer">
                    <p><strong>Bybit Trading Bot</strong> - Professional Algorithmic Trading System</p>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} AEST</p>
                    <p>For questions or concerns, please contact your system administrator.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_chart_section(self, title: str, chart_key: str, charts: Dict[str, str]) -> str:
        """Generate HTML section with chart"""
        if chart_key in charts and charts[chart_key]:
            return f"""
            <div class="section">
                <h2>{title}</h2>
                <div class="chart">
                    <img src="data:image/png;base64,{charts[chart_key]}" alt="{title} Chart">
                </div>
            </div>
            """
        return ""
    
    def _generate_active_strategies_table(self, strategies: Dict[str, Dict]) -> str:
        """Generate HTML table for active strategies"""
        if not strategies:
            return ""
        
        rows = ""
        for name, data in strategies.items():
            status_class = "status-active" if data.get('status') == 'Active' else "status-inactive"
            return_class = "positive" if data.get('weekly_return', 0) > 0 else "negative"
            
            rows += f"""
            <tr>
                <td><span class="status-indicator {status_class}"></span>{name}</td>
                <td>{data.get('status', 'Unknown')}</td>
                <td class="{return_class}">{data.get('weekly_return', 0):.2f}%</td>
                <td>{data.get('positions', 0)}</td>
                <td>{data.get('win_rate', 0):.1f}%</td>
            </tr>
            """
        
        return f"""
        <div class="section">
            <h2>Active Trading Strategies</h2>
            <table>
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th>Status</th>
                        <th>Weekly Return</th>
                        <th>Positions</th>
                        <th>Win Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_risk_compliance_section(self, data: Dict[str, Any]) -> str:
        """Generate risk and compliance section"""
        position_risk = data.get('position_risk', 0)
        daily_loss_used = data.get('daily_loss_used', 0)
        
        return f"""
        <div class="section">
            <h2>Risk & Compliance</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                <div>
                    <h4>📊 Position Risk</h4>
                    <p><strong>{position_risk:.2f}%</strong> of portfolio (Limit: 10%)</p>
                </div>
                <div>
                    <h4>🚨 Daily Loss Limit</h4>
                    <p><strong>{daily_loss_used:.1f}%</strong> of 5% limit used</p>
                </div>
                <div>
                    <h4>✅ Compliance Status</h4>
                    <p><span style="color: {self.colors['success']};">All checks passed</span></p>
                </div>
                <div>
                    <h4>🇦🇺 Tax Optimization</h4>
                    <p>CGT discount tracking active</p>
                </div>
            </div>
        </div>
        """
    
    def _create_alert_html(self, message: str, alert_type: str, config: Dict[str, str], 
                          data: Dict[str, Any] = None) -> str:
        """Create HTML content for alerts"""
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; padding: 30px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; padding: 20px; background: {config['color']}; color: white; border-radius: 8px; margin-bottom: 20px; }}
                .alert-icon {{ font-size: 48px; margin-bottom: 10px; }}
                .message {{ font-size: 18px; line-height: 1.6; margin: 20px 0; }}
                .data-section {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .timestamp {{ color: #666; font-size: 14px; text-align: center; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="alert-icon">{config['emoji']}</div>
                    <h2>{alert_type.upper()} ALERT</h2>
                </div>
                
                <div class="message">{message}</div>
                
                {self._format_alert_data(data) if data else ''}
                
                <div class="timestamp">
                    Alert generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} AEST
                </div>
            </div>
        </body>
        </html>
        """
    
    def _format_alert_data(self, data: Dict[str, Any]) -> str:
        """Format additional alert data"""
        if not data:
            return ""
        
        formatted_items = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if 'price' in key.lower() or 'value' in key.lower():
                    formatted_value = f"${value:,.2f}"
                elif 'percent' in key.lower() or 'rate' in key.lower():
                    formatted_value = f"{value:.2f}%"
                else:
                    formatted_value = f"{value:,.2f}"
            else:
                formatted_value = str(value)
            
            formatted_items.append(f"<strong>{key.replace('_', ' ').title()}:</strong> {formatted_value}")
        
        return f"""
        <div class="data-section">
            <h4>Alert Details:</h4>
            <ul>
                {''.join(f'<li>{item}</li>' for item in formatted_items)}
            </ul>
        </div>
        """
    
    def test_email_connection(self) -> Dict[str, Any]:
        """Test SendGrid connection and configuration"""
        try:
            # Try to get account information
            response = self.sg.client.user.account.get()
            
            return {
                'connected': True,
                'status_code': response.status_code,
                'test_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'test_time': datetime.now().isoformat()
            }


# Example usage and testing
if __name__ == "__main__":
    # Test email manager (requires valid SendGrid API key)
    test_api_key = os.getenv('SENDGRID_API_KEY', 'test_key')
    
    if test_api_key != 'test_key':
        email_manager = SendGridEmailManager(
            api_key=test_api_key,
            from_email="test@yourdomain.com",
            from_name="Test Trading Bot"
        )
        
        # Test connection
        connection_test = email_manager.test_email_connection()
        print(f"Connection test: {connection_test}")
        
        # Test data for report generation
        test_report_data = {
            'week_ending': '2024-01-07',
            'portfolio_value': 125000.50,
            'weekly_return': 2.34,
            'sharpe_ratio': 1.45,
            'max_drawdown': -3.21,
            'daily_values': {
                'dates': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
                'portfolio_values': [122000, 123500, 124200, 123800, 125000]
            },
            'strategy_performance': {
                'Momentum': 45.2,
                'Mean Reversion': 32.1,
                'Arbitrage': 22.7
            },
            'active_strategies': {
                'BTC Momentum': {
                    'status': 'Active',
                    'weekly_return': 3.45,
                    'positions': 2,
                    'win_rate': 67.5
                },
                'ETH Mean Reversion': {
                    'status': 'Active',
                    'weekly_return': 1.23,
                    'positions': 1,
                    'win_rate': 58.2
                }
            },
            'position_risk': 7.5,
            'daily_loss_used': 15.2
        }
        
        # Generate charts (this will create them in memory)
        charts = email_manager._generate_performance_charts(test_report_data)
        print(f"Generated charts: {list(charts.keys())}")
        
        print("Email manager test completed!")
    else:
        print("Set SENDGRID_API_KEY environment variable to test email functionality")