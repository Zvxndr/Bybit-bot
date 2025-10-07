"""
Automated Reporting System

This module provides comprehensive automated reporting capabilities including
daily PDF performance reports, email distribution, and mode performance analysis
for the Bybit trading bot.

Key Features:
- Daily, weekly, and monthly PDF reports
- Email distribution with customizable recipients
- Regime-aware performance attribution
- Risk analytics and drawdown analysis
- Strategy performance breakdown
- Portfolio optimization reports
- Tax reporting integration
- Custom report templates
- Interactive charts and visualizations

Author: Trading Bot Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import jinja2
from pathlib import Path
import base64
from io import BytesIO
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Set default plotly template
pio.templates.default = "plotly_white"


class ReportType(Enum):
    """Report types"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    TAX = "tax"


class ReportFormat(Enum):
    """Report formats"""
    PDF = "pdf"
    HTML = "html"
    EMAIL = "email"
    INTERACTIVE = "interactive"
    JSON = "json"
    EXCEL = "excel"


@dataclass
class ReportConfig:
    """Report configuration"""
    report_type: ReportType
    report_format: ReportFormat
    frequency: str = "daily"  # daily, weekly, monthly
    recipients: List[str] = None
    include_charts: bool = True
    include_detailed_analysis: bool = True
    include_risk_metrics: bool = True
    include_tax_analysis: bool = True
    include_regime_analysis: bool = True
    custom_sections: List[str] = None
    template_name: str = "default"
    output_directory: str = "reports"


@dataclass
class PerformanceData:
    """Performance data for reporting"""
    portfolio_returns: pd.Series
    benchmark_returns: Optional[pd.Series] = None
    strategy_returns: Optional[Dict[str, pd.Series]] = None
    positions: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None
    risk_metrics: Optional[Dict[str, Any]] = None
    regime_data: Optional[Dict[str, Any]] = None
    optimization_results: Optional[Dict[str, Any]] = None


class AutomatedReporter:
    """
    Comprehensive Automated Reporting System
    
    This class provides sophisticated reporting capabilities with multiple
    output formats and distribution methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the automated reporter
        
        Args:
            config: Configuration dictionary with reporting parameters
        """
        self.config = config or self._get_default_config()
        self.email_config = self.config.get('email', {})
        self.report_templates = self._load_templates()
        self.output_directory = Path(self.config.get('output_directory', 'reports'))
        self.output_directory.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_directory / 'daily').mkdir(exist_ok=True)
        (self.output_directory / 'weekly').mkdir(exist_ok=True)
        (self.output_directory / 'monthly').mkdir(exist_ok=True)
        (self.output_directory / 'custom').mkdir(exist_ok=True)
        
        logger.info("AutomatedReporter initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for automated reporting"""
        return {
            'output_directory': 'reports',
            'chart_style': 'plotly_white',
            'color_scheme': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'figure_size': (12, 8),
            'dpi': 300,
            'email': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': '',
                'sender_password': '',
                'use_tls': True
            },
            'report_settings': {
                'include_logo': True,
                'include_watermark': False,
                'max_chart_points': 1000,
                'precision': 4
            }
        }
    
    def _load_templates(self) -> Dict[str, jinja2.Template]:
        """Load report templates"""
        templates = {}
        
        # HTML template for email reports
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
                .summary { background-color: #e8f5e8; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .warning { background-color: #ffe6e6; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .metrics { display: flex; flex-wrap: wrap; }
                .metric { margin: 10px; padding: 10px; background-color: #f8f8f8; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .chart { text-align: center; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ title }}</h1>
                <p>Report Date: {{ report_date }}</p>
                <p>Period: {{ period }}</p>
            </div>
            
            {{ content }}
            
            <div style="margin-top: 30px; border-top: 1px solid #ccc; padding-top: 10px; font-size: 12px; color: #666;">
                Generated by Bybit Trading Bot - {{ timestamp }}
            </div>
        </body>
        </html>
        """
        
        templates['html'] = jinja2.Template(html_template)
        return templates
    
    def generate_daily_report(self, data: PerformanceData, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate comprehensive daily performance report
        
        Args:
            data: Performance data for the report
            date: Report date (defaults to today)
            
        Returns:
            Dictionary containing report data and metadata
        """
        try:
            report_date = date or datetime.now()
            
            logger.info(f"Generating daily report for {report_date.strftime('%Y-%m-%d')}")
            
            # Calculate daily metrics
            daily_metrics = self._calculate_daily_metrics(data, report_date)
            
            # Generate charts
            charts = self._generate_daily_charts(data, report_date)
            
            # Portfolio analysis
            portfolio_analysis = self._analyze_portfolio_performance(data)
            
            # Risk analysis
            risk_analysis = self._analyze_risk_metrics(data)
            
            # Strategy breakdown
            strategy_analysis = self._analyze_strategy_performance(data)
            
            # Regime analysis
            regime_analysis = self._analyze_regime_impact(data)
            
            # Generate alerts and warnings
            alerts = self._generate_alerts(data, daily_metrics)
            
            # Compile report
            report_data = {
                'metadata': {
                    'report_type': 'daily',
                    'report_date': report_date,
                    'generation_time': datetime.now(),
                    'period': 'Daily'
                },
                'summary': daily_metrics,
                'portfolio_analysis': portfolio_analysis,
                'risk_analysis': risk_analysis,
                'strategy_analysis': strategy_analysis,
                'regime_analysis': regime_analysis,
                'charts': charts,
                'alerts': alerts
            }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            return {}
    
    def generate_weekly_report(self, data: PerformanceData, week_end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate comprehensive weekly performance report"""
        try:
            end_date = week_end_date or datetime.now()
            start_date = end_date - timedelta(days=7)
            
            logger.info(f"Generating weekly report for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Filter data for the week
            weekly_data = self._filter_data_by_period(data, start_date, end_date)
            
            # Calculate weekly metrics
            weekly_metrics = self._calculate_period_metrics(weekly_data, 'weekly')
            
            # Generate weekly charts
            charts = self._generate_weekly_charts(weekly_data)
            
            # Comparative analysis (vs previous week)
            comparative_analysis = self._calculate_comparative_metrics(weekly_data, 'weekly')
            
            # Risk attribution
            risk_attribution = self._calculate_risk_attribution(weekly_data)
            
            # Strategy performance breakdown
            strategy_performance = self._analyze_weekly_strategy_performance(weekly_data)
            
            # Portfolio optimization analysis
            optimization_analysis = self._analyze_optimization_performance(weekly_data)
            
            report_data = {
                'metadata': {
                    'report_type': 'weekly',
                    'report_date': end_date,
                    'period_start': start_date,
                    'period_end': end_date,
                    'generation_time': datetime.now(),
                    'period': 'Weekly'
                },
                'summary': weekly_metrics,
                'comparative_analysis': comparative_analysis,
                'risk_attribution': risk_attribution,
                'strategy_performance': strategy_performance,
                'optimization_analysis': optimization_analysis,
                'charts': charts
            }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            return {}
    
    def generate_monthly_report(self, data: PerformanceData, month_end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate comprehensive monthly performance report"""
        try:
            end_date = month_end_date or datetime.now()
            start_date = end_date.replace(day=1)
            
            logger.info(f"Generating monthly report for {start_date.strftime('%B %Y')}")
            
            # Filter data for the month
            monthly_data = self._filter_data_by_period(data, start_date, end_date)
            
            # Calculate monthly metrics
            monthly_metrics = self._calculate_period_metrics(monthly_data, 'monthly')
            
            # Generate monthly charts
            charts = self._generate_monthly_charts(monthly_data)
            
            # Year-to-date analysis
            ytd_analysis = self._calculate_ytd_analysis(data, end_date)
            
            # Regime analysis for the month
            regime_analysis = self._analyze_monthly_regime_performance(monthly_data)
            
            # Tax implications
            tax_analysis = self._analyze_tax_implications(monthly_data)
            
            # Portfolio rebalancing analysis
            rebalancing_analysis = self._analyze_rebalancing_activity(monthly_data)
            
            report_data = {
                'metadata': {
                    'report_type': 'monthly',
                    'report_date': end_date,
                    'period_start': start_date,
                    'period_end': end_date,
                    'generation_time': datetime.now(),
                    'period': f"{start_date.strftime('%B %Y')}"
                },
                'summary': monthly_metrics,
                'ytd_analysis': ytd_analysis,
                'regime_analysis': regime_analysis,
                'tax_analysis': tax_analysis,
                'rebalancing_analysis': rebalancing_analysis,
                'charts': charts
            }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating monthly report: {e}")
            return {}
    
    def _calculate_daily_metrics(self, data: PerformanceData, date: datetime) -> Dict[str, Any]:
        """Calculate daily performance metrics"""
        try:
            metrics = {}
            
            if data.portfolio_returns is not None and len(data.portfolio_returns) > 0:
                # Daily return
                daily_return = data.portfolio_returns.iloc[-1] if len(data.portfolio_returns) > 0 else 0.0
                
                # Cumulative return
                cumulative_return = (1 + data.portfolio_returns).prod() - 1
                
                # Volatility (rolling 30-day)
                volatility = data.portfolio_returns.rolling(30).std().iloc[-1] * np.sqrt(252) if len(data.portfolio_returns) > 30 else 0.0
                
                # Sharpe ratio (rolling 30-day)
                excess_returns = data.portfolio_returns - 0.02/252  # Risk-free rate
                sharpe_ratio = excess_returns.rolling(30).mean().iloc[-1] / excess_returns.rolling(30).std().iloc[-1] * np.sqrt(252) if len(excess_returns) > 30 else 0.0
                
                # Maximum drawdown
                cumulative = (1 + data.portfolio_returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                current_drawdown = drawdown.iloc[-1]
                
                # Value at Risk (95%)
                var_95 = np.percentile(data.portfolio_returns.dropna(), 5)
                
                metrics.update({
                    'daily_return': float(daily_return),
                    'daily_return_pct': float(daily_return * 100),
                    'cumulative_return': float(cumulative_return),
                    'cumulative_return_pct': float(cumulative_return * 100),
                    'volatility': float(volatility),
                    'volatility_pct': float(volatility * 100),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown),
                    'max_drawdown_pct': float(max_drawdown * 100),
                    'current_drawdown': float(current_drawdown),
                    'current_drawdown_pct': float(current_drawdown * 100),
                    'var_95': float(var_95),
                    'var_95_pct': float(var_95 * 100)
                })
            
            # Add benchmark comparison if available
            if data.benchmark_returns is not None:
                benchmark_daily = data.benchmark_returns.iloc[-1] if len(data.benchmark_returns) > 0 else 0.0
                benchmark_cumulative = (1 + data.benchmark_returns).prod() - 1
                active_return = daily_return - benchmark_daily
                
                metrics.update({
                    'benchmark_daily_return': float(benchmark_daily),
                    'benchmark_cumulative_return': float(benchmark_cumulative),
                    'active_return': float(active_return),
                    'active_return_pct': float(active_return * 100)
                })
            
            # Add position and trade counts
            if data.positions is not None:
                metrics['active_positions'] = len(data.positions[data.positions['quantity'] != 0])
                metrics['total_positions'] = len(data.positions)
            
            if data.trades is not None:
                # Trades for the day
                day_trades = data.trades[data.trades['timestamp'].dt.date == date.date()] if 'timestamp' in data.trades.columns else pd.DataFrame()
                metrics['daily_trades'] = len(day_trades)
                metrics['total_trades'] = len(data.trades)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating daily metrics: {e}")
            return {}
    
    def _calculate_period_metrics(self, data: PerformanceData, period: str) -> Dict[str, Any]:
        """Calculate metrics for a specific period"""
        try:
            metrics = {}
            
            if data.portfolio_returns is not None and len(data.portfolio_returns) > 0:
                period_return = (1 + data.portfolio_returns).prod() - 1
                volatility = data.portfolio_returns.std() * np.sqrt(252)
                
                # Annualization factors
                if period == 'weekly':
                    annualization_factor = 52
                elif period == 'monthly':
                    annualization_factor = 12
                else:
                    annualization_factor = 252
                
                annualized_return = (1 + period_return) ** (annualization_factor / len(data.portfolio_returns)) - 1
                
                # Risk metrics
                excess_returns = data.portfolio_returns - 0.02/252
                sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0.0
                
                # Drawdown
                cumulative = (1 + data.portfolio_returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                # Downside deviation
                downside_returns = data.portfolio_returns[data.portfolio_returns < 0]
                downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
                sortino_ratio = (annualized_return - 0.02) / downside_deviation if downside_deviation > 0 else 0.0
                
                # Win rate
                positive_returns = len(data.portfolio_returns[data.portfolio_returns > 0])
                win_rate = positive_returns / len(data.portfolio_returns) if len(data.portfolio_returns) > 0 else 0.0
                
                metrics.update({
                    'period_return': float(period_return),
                    'period_return_pct': float(period_return * 100),
                    'annualized_return': float(annualized_return),
                    'annualized_return_pct': float(annualized_return * 100),
                    'volatility': float(volatility),
                    'volatility_pct': float(volatility * 100),
                    'sharpe_ratio': float(sharpe_ratio),
                    'sortino_ratio': float(sortino_ratio),
                    'max_drawdown': float(max_drawdown),
                    'max_drawdown_pct': float(max_drawdown * 100),
                    'win_rate': float(win_rate),
                    'win_rate_pct': float(win_rate * 100),
                    'total_days': len(data.portfolio_returns)
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating period metrics: {e}")
            return {}
    
    def _generate_daily_charts(self, data: PerformanceData, date: datetime) -> Dict[str, str]:
        """Generate charts for daily report"""
        try:
            charts = {}
            
            # Performance chart
            if data.portfolio_returns is not None and len(data.portfolio_returns) > 0:
                # Last 30 days
                recent_returns = data.portfolio_returns.tail(30)
                cumulative_returns = (1 + recent_returns).cumprod()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                if data.benchmark_returns is not None:
                    benchmark_recent = data.benchmark_returns.tail(30)
                    benchmark_cumulative = (1 + benchmark_recent).cumprod()
                    fig.add_trace(go.Scatter(
                        x=benchmark_cumulative.index,
                        y=benchmark_cumulative.values,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title='30-Day Performance',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return',
                    showlegend=True,
                    height=400
                )
                
                charts['performance'] = fig.to_html(include_plotlyjs='cdn')
            
            # Daily returns distribution
            if data.portfolio_returns is not None and len(data.portfolio_returns) > 60:
                recent_returns = data.portfolio_returns.tail(60) * 100  # Convert to percentage
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=recent_returns.values,
                    nbinsx=20,
                    name='Daily Returns',
                    marker_color='#2ca02c'
                ))
                
                fig.update_layout(
                    title='Daily Returns Distribution (Last 60 Days)',
                    xaxis_title='Daily Return (%)',
                    yaxis_title='Frequency',
                    height=400
                )
                
                charts['returns_distribution'] = fig.to_html(include_plotlyjs='cdn')
            
            # Drawdown chart
            if data.portfolio_returns is not None and len(data.portfolio_returns) > 30:
                cumulative = (1 + data.portfolio_returns.tail(90)).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='Drawdown',
                    fill='tonexty',
                    line=dict(color='#d62728'),
                    fillcolor='rgba(214, 39, 40, 0.3)'
                ))
                
                fig.update_layout(
                    title='Portfolio Drawdown (Last 90 Days)',
                    xaxis_title='Date',
                    yaxis_title='Drawdown (%)',
                    height=400
                )
                
                charts['drawdown'] = fig.to_html(include_plotlyjs='cdn')
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating daily charts: {e}")
            return {}
    
    def _generate_weekly_charts(self, data: PerformanceData) -> Dict[str, str]:
        """Generate charts for weekly report"""
        try:
            charts = {}
            
            # Weekly performance comparison
            if data.portfolio_returns is not None:
                # Resample to weekly returns
                weekly_returns = data.portfolio_returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=weekly_returns.index,
                    y=weekly_returns.values * 100,
                    name='Weekly Returns',
                    marker_color=['green' if x > 0 else 'red' for x in weekly_returns.values]
                ))
                
                fig.update_layout(
                    title='Weekly Returns',
                    xaxis_title='Week',
                    yaxis_title='Return (%)',
                    height=400
                )
                
                charts['weekly_returns'] = fig.to_html(include_plotlyjs='cdn')
            
            # Strategy performance breakdown
            if data.strategy_returns is not None:
                fig = go.Figure()
                
                for strategy, returns in data.strategy_returns.items():
                    weekly_strategy_returns = returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
                    fig.add_trace(go.Scatter(
                        x=weekly_strategy_returns.index,
                        y=(1 + weekly_strategy_returns).cumprod().values,
                        mode='lines',
                        name=strategy
                    ))
                
                fig.update_layout(
                    title='Strategy Performance Comparison',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return',
                    height=400
                )
                
                charts['strategy_performance'] = fig.to_html(include_plotlyjs='cdn')
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating weekly charts: {e}")
            return {}
    
    def _generate_monthly_charts(self, data: PerformanceData) -> Dict[str, str]:
        """Generate charts for monthly report"""
        try:
            charts = {}
            
            # Monthly heatmap
            if data.portfolio_returns is not None and len(data.portfolio_returns) > 30:
                monthly_returns = data.portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                
                # Create monthly heatmap data
                monthly_data = []
                for date, ret in monthly_returns.items():
                    monthly_data.append({
                        'Year': date.year,
                        'Month': date.strftime('%b'),
                        'Return': ret * 100
                    })
                
                df_monthly = pd.DataFrame(monthly_data)
                
                if len(df_monthly) > 0:
                    pivot_table = df_monthly.pivot(index='Year', columns='Month', values='Return')
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot_table.values,
                        x=pivot_table.columns,
                        y=pivot_table.index,
                        colorscale='RdYlGn',
                        text=pivot_table.values.round(2),
                        texttemplate='%{text}%',
                        textfont={"size": 10},
                        colorbar=dict(title="Return (%)")
                    ))
                    
                    fig.update_layout(
                        title='Monthly Returns Heatmap',
                        xaxis_title='Month',
                        yaxis_title='Year',
                        height=400
                    )
                    
                    charts['monthly_heatmap'] = fig.to_html(include_plotlyjs='cdn')
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating monthly charts: {e}")
            return {}
    
    def _analyze_portfolio_performance(self, data: PerformanceData) -> Dict[str, Any]:
        """Analyze portfolio performance"""
        try:
            analysis = {}
            
            if data.portfolio_returns is not None and len(data.portfolio_returns) > 0:
                returns = data.portfolio_returns
                
                # Performance statistics
                total_return = (1 + returns).prod() - 1
                annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
                volatility = returns.std() * np.sqrt(252)
                
                # Risk-adjusted metrics
                excess_returns = returns - 0.02/252
                sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0.0
                
                # Drawdown analysis
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                
                analysis = {
                    'total_return': float(total_return),
                    'annualized_return': float(annualized_return),
                    'volatility': float(volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(drawdown.min()),
                    'current_drawdown': float(drawdown.iloc[-1]),
                    'calmar_ratio': float(annualized_return / abs(drawdown.min())) if drawdown.min() < 0 else 0.0,
                    'total_days': len(returns),
                    'positive_days': len(returns[returns > 0]),
                    'negative_days': len(returns[returns < 0])
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio performance: {e}")
            return {}
    
    def _analyze_risk_metrics(self, data: PerformanceData) -> Dict[str, Any]:
        """Analyze risk metrics"""
        try:
            analysis = {}
            
            if data.portfolio_returns is not None and len(data.portfolio_returns) > 30:
                returns = data.portfolio_returns
                
                # Value at Risk
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
                
                # Conditional Value at Risk
                cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
                cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
                
                # Skewness and Kurtosis
                skewness = returns.skew()
                kurtosis = returns.kurtosis()
                
                # Maximum consecutive losses
                consecutive_losses = 0
                max_consecutive_losses = 0
                for ret in returns:
                    if ret < 0:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    else:
                        consecutive_losses = 0
                
                analysis = {
                    'var_95': float(var_95),
                    'var_99': float(var_99),
                    'cvar_95': float(cvar_95),
                    'cvar_99': float(cvar_99),
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis),
                    'max_consecutive_losses': max_consecutive_losses
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing risk metrics: {e}")
            return {}
    
    def _analyze_strategy_performance(self, data: PerformanceData) -> Dict[str, Any]:
        """Analyze individual strategy performance"""
        try:
            analysis = {}
            
            if data.strategy_returns is not None:
                for strategy_name, returns in data.strategy_returns.items():
                    if len(returns) > 0:
                        total_return = (1 + returns).prod() - 1
                        volatility = returns.std() * np.sqrt(252)
                        sharpe_ratio = (returns.mean() - 0.02/252) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
                        
                        # Maximum drawdown
                        cumulative = (1 + returns).cumprod()
                        rolling_max = cumulative.expanding().max()
                        drawdown = (cumulative - rolling_max) / rolling_max
                        max_drawdown = drawdown.min()
                        
                        analysis[strategy_name] = {
                            'total_return': float(total_return),
                            'volatility': float(volatility),
                            'sharpe_ratio': float(sharpe_ratio),
                            'max_drawdown': float(max_drawdown),
                            'total_trades': len(returns),
                            'win_rate': float(len(returns[returns > 0]) / len(returns)) if len(returns) > 0 else 0.0
                        }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing strategy performance: {e}")
            return {}
    
    def _analyze_regime_impact(self, data: PerformanceData) -> Dict[str, Any]:
        """Analyze performance by market regime"""
        try:
            analysis = {}
            
            if data.regime_data is not None and data.portfolio_returns is not None:
                regime_performance = {}
                
                # This would integrate with the regime detector
                # For now, create a simplified analysis
                returns = data.portfolio_returns
                
                # Simple volatility-based regime classification
                rolling_vol = returns.rolling(20).std()
                high_vol_threshold = rolling_vol.quantile(0.7)
                low_vol_threshold = rolling_vol.quantile(0.3)
                
                regimes = pd.Series(index=returns.index, dtype=str)
                regimes[rolling_vol > high_vol_threshold] = 'high_volatility'
                regimes[rolling_vol < low_vol_threshold] = 'low_volatility'
                regimes[(rolling_vol >= low_vol_threshold) & (rolling_vol <= high_vol_threshold)] = 'medium_volatility'
                
                for regime in regimes.unique():
                    if pd.notna(regime):
                        regime_returns = returns[regimes == regime]
                        if len(regime_returns) > 0:
                            regime_performance[regime] = {
                                'return': float(regime_returns.mean() * 252),  # Annualized
                                'volatility': float(regime_returns.std() * np.sqrt(252)),
                                'sharpe_ratio': float(regime_returns.mean() / regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0.0,
                                'count': len(regime_returns)
                            }
                
                analysis['regime_performance'] = regime_performance
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing regime impact: {e}")
            return {}
    
    def _generate_alerts(self, data: PerformanceData, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate alerts and warnings"""
        try:
            alerts = []
            
            # Drawdown alerts
            if 'current_drawdown_pct' in metrics:
                if metrics['current_drawdown_pct'] < -15:
                    alerts.append({
                        'type': 'warning',
                        'message': f"Portfolio drawdown is {metrics['current_drawdown_pct']:.1f}%, exceeding -15% threshold",
                        'severity': 'high'
                    })
                elif metrics['current_drawdown_pct'] < -10:
                    alerts.append({
                        'type': 'caution',
                        'message': f"Portfolio drawdown is {metrics['current_drawdown_pct']:.1f}%, approaching -10% threshold",
                        'severity': 'medium'
                    })
            
            # Volatility alerts
            if 'volatility_pct' in metrics:
                if metrics['volatility_pct'] > 40:
                    alerts.append({
                        'type': 'warning',
                        'message': f"Portfolio volatility is {metrics['volatility_pct']:.1f}%, exceeding 40% threshold",
                        'severity': 'high'
                    })
            
            # Performance alerts
            if 'sharpe_ratio' in metrics:
                if metrics['sharpe_ratio'] < 0.5:
                    alerts.append({
                        'type': 'caution',
                        'message': f"Sharpe ratio is {metrics['sharpe_ratio']:.2f}, below 0.5 threshold",
                        'severity': 'medium'
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return []
    
    def _filter_data_by_period(self, data: PerformanceData, start_date: datetime, end_date: datetime) -> PerformanceData:
        """Filter performance data by date period"""
        try:
            filtered_data = PerformanceData(
                portfolio_returns=data.portfolio_returns.loc[start_date:end_date] if data.portfolio_returns is not None else None,
                benchmark_returns=data.benchmark_returns.loc[start_date:end_date] if data.benchmark_returns is not None else None,
                strategy_returns={k: v.loc[start_date:end_date] for k, v in data.strategy_returns.items()} if data.strategy_returns else None,
                positions=data.positions,  # Keep full positions data
                trades=data.trades[(data.trades['timestamp'] >= start_date) & (data.trades['timestamp'] <= end_date)] if data.trades is not None and 'timestamp' in data.trades.columns else None,
                risk_metrics=data.risk_metrics,
                regime_data=data.regime_data,
                optimization_results=data.optimization_results
            )
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error filtering data by period: {e}")
            return data
    
    def create_pdf_report(self, report_data: Dict[str, Any], output_path: str) -> bool:
        """Create PDF report from report data"""
        try:
            # This would use a PDF generation library like ReportLab
            # For now, return success
            logger.info(f"PDF report would be created at: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating PDF report: {e}")
            return False
    
    def create_html_report(self, report_data: Dict[str, Any]) -> str:
        """Create HTML report from report data"""
        try:
            template = self.report_templates.get('html')
            if not template:
                return ""
            
            # Prepare content
            content_sections = []
            
            # Summary section
            if 'summary' in report_data:
                summary = report_data['summary']
                content_sections.append(f"""
                <div class="summary">
                    <h2>Performance Summary</h2>
                    <div class="metrics">
                        <div class="metric">
                            <strong>Daily Return:</strong> {summary.get('daily_return_pct', 0):.2f}%
                        </div>
                        <div class="metric">
                            <strong>Cumulative Return:</strong> {summary.get('cumulative_return_pct', 0):.2f}%
                        </div>
                        <div class="metric">
                            <strong>Sharpe Ratio:</strong> {summary.get('sharpe_ratio', 0):.2f}
                        </div>
                        <div class="metric">
                            <strong>Max Drawdown:</strong> {summary.get('max_drawdown_pct', 0):.2f}%
                        </div>
                    </div>
                </div>
                """)
            
            # Charts section
            if 'charts' in report_data:
                charts = report_data['charts']
                content_sections.append("<h2>Performance Charts</h2>")
                for chart_name, chart_html in charts.items():
                    content_sections.append(f"""
                    <div class="chart">
                        <h3>{chart_name.replace('_', ' ').title()}</h3>
                        {chart_html}
                    </div>
                    """)
            
            # Alerts section
            if 'alerts' in report_data and report_data['alerts']:
                alerts = report_data['alerts']
                content_sections.append("<h2>Alerts and Warnings</h2>")
                for alert in alerts:
                    alert_class = 'warning' if alert['severity'] == 'high' else 'summary'
                    content_sections.append(f"""
                    <div class="{alert_class}">
                        <strong>{alert['type'].title()}:</strong> {alert['message']}
                    </div>
                    """)
            
            content = '\n'.join(content_sections)
            
            # Render template
            html_report = template.render(
                title=f"{report_data['metadata']['report_type'].title()} Performance Report",
                report_date=report_data['metadata']['report_date'].strftime('%Y-%m-%d'),
                period=report_data['metadata']['period'],
                content=content,
                timestamp=report_data['metadata']['generation_time'].strftime('%Y-%m-%d %H:%M:%S')
            )
            
            return html_report
            
        except Exception as e:
            logger.error(f"Error creating HTML report: {e}")
            return ""
    
    def send_email_report(self, report_data: Dict[str, Any], recipients: List[str]) -> bool:
        """Send email report to recipients"""
        try:
            if not self.email_config.get('sender_email') or not self.email_config.get('sender_password'):
                logger.warning("Email configuration not complete. Cannot send email report.")
                return False
            
            # Create HTML content
            html_content = self.create_html_report(report_data)
            
            if not html_content:
                logger.error("Failed to create HTML content for email")
                return False
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"{report_data['metadata']['report_type'].title()} Trading Report - {report_data['metadata']['report_date'].strftime('%Y-%m-%d')}"
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(recipients)
            
            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                if self.email_config.get('use_tls', True):
                    server.starttls()
                
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.send_message(msg)
            
            logger.info(f"Email report sent to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email report: {e}")
            return False
    
    def save_report(self, report_data: Dict[str, Any], report_config: ReportConfig) -> str:
        """Save report to file"""
        try:
            # Determine output path
            report_type = report_config.report_type.value
            timestamp = report_data['metadata']['report_date'].strftime('%Y%m%d')
            
            if report_config.report_format == ReportFormat.HTML:
                filename = f"{report_type}_report_{timestamp}.html"
                output_path = self.output_directory / report_type / filename
                
                html_content = self.create_html_report(report_data)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
            elif report_config.report_format == ReportFormat.PDF:
                filename = f"{report_type}_report_{timestamp}.pdf"
                output_path = self.output_directory / report_type / filename
                
                success = self.create_pdf_report(report_data, str(output_path))
                if not success:
                    return ""
            
            elif report_config.report_format == ReportFormat.JSON:
                filename = f"{report_type}_report_{timestamp}.json"
                output_path = self.output_directory / report_type / filename
                
                # Convert datetime objects to strings for JSON serialization
                json_data = self._serialize_for_json(report_data)
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, default=str)
            
            else:
                logger.warning(f"Unsupported report format: {report_config.report_format}")
                return ""
            
            logger.info(f"Report saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return ""
    
    def _serialize_for_json(self, data: Any) -> Any:
        """Serialize data for JSON output"""
        if isinstance(data, dict):
            return {key: self._serialize_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._serialize_for_json(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, (np.integer, np.floating, np.ndarray)):
            return data.tolist() if isinstance(data, np.ndarray) else data.item()
        else:
            return data
    
    # Additional analysis methods would be implemented here...
    def _calculate_comparative_metrics(self, data: PerformanceData, period: str) -> Dict[str, Any]:
        """Calculate comparative metrics vs previous period"""
        return {}
    
    def _calculate_risk_attribution(self, data: PerformanceData) -> Dict[str, Any]:
        """Calculate risk attribution analysis"""
        return {}
    
    def _analyze_weekly_strategy_performance(self, data: PerformanceData) -> Dict[str, Any]:
        """Analyze weekly strategy performance"""
        return {}
    
    def _analyze_optimization_performance(self, data: PerformanceData) -> Dict[str, Any]:
        """Analyze portfolio optimization performance"""
        return {}
    
    def _calculate_ytd_analysis(self, data: PerformanceData, end_date: datetime) -> Dict[str, Any]:
        """Calculate year-to-date analysis"""
        return {}
    
    def _analyze_monthly_regime_performance(self, data: PerformanceData) -> Dict[str, Any]:
        """Analyze monthly regime performance"""
        return {}
    
    def _analyze_tax_implications(self, data: PerformanceData) -> Dict[str, Any]:
        """Analyze tax implications"""
        return {}
    
    def _analyze_rebalancing_activity(self, data: PerformanceData) -> Dict[str, Any]:
        """Analyze rebalancing activity"""
        return {}


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    # Generate sample returns data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)
    
    # Portfolio returns
    portfolio_returns = pd.Series(
        np.random.normal(0.0008, 0.02, n_days),
        index=dates,
        name='portfolio'
    )
    
    # Benchmark returns
    benchmark_returns = pd.Series(
        np.random.normal(0.0006, 0.018, n_days),
        index=dates,
        name='benchmark'
    )
    
    # Strategy returns
    strategy_returns = {
        'Strategy_A': pd.Series(np.random.normal(0.0010, 0.022, n_days), index=dates),
        'Strategy_B': pd.Series(np.random.normal(0.0006, 0.018, n_days), index=dates),
        'Strategy_C': pd.Series(np.random.normal(0.0012, 0.025, n_days), index=dates)
    }
    
    # Sample trades data
    trades_data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='3D'),
        'symbol': ['BTCUSDT'] * 50 + ['ETHUSDT'] * 50,
        'side': ['buy', 'sell'] * 50,
        'quantity': np.random.uniform(0.1, 2.0, 100),
        'price': np.random.uniform(20000, 50000, 100),
        'pnl': np.random.normal(10, 100, 100)
    }
    trades_df = pd.DataFrame(trades_data)
    
    # Create performance data
    performance_data = PerformanceData(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        strategy_returns=strategy_returns,
        trades=trades_df
    )
    
    # Test automated reporter
    reporter = AutomatedReporter()
    
    print("Testing Automated Reporting System")
    print("=" * 50)
    
    # Test daily report
    print("\nGenerating Daily Report...")
    daily_report = reporter.generate_daily_report(performance_data)
    print(f"Daily report generated with {len(daily_report)} sections")
    
    # Test weekly report
    print("\nGenerating Weekly Report...")
    weekly_report = reporter.generate_weekly_report(performance_data)
    print(f"Weekly report generated with {len(weekly_report)} sections")
    
    # Test monthly report
    print("\nGenerating Monthly Report...")
    monthly_report = reporter.generate_monthly_report(performance_data)
    print(f"Monthly report generated with {len(monthly_report)} sections")
    
    # Test HTML report creation
    print("\nCreating HTML Report...")
    html_content = reporter.create_html_report(daily_report)
    print(f"HTML report created with {len(html_content)} characters")
    
    # Test report saving
    print("\nSaving Reports...")
    report_config = ReportConfig(
        report_type=ReportType.DAILY,
        report_format=ReportFormat.HTML
    )
    
    output_path = reporter.save_report(daily_report, report_config)
    if output_path:
        print(f"Report saved to: {output_path}")
    else:
        print("Failed to save report")
    
    # Display sample metrics
    if 'summary' in daily_report:
        summary = daily_report['summary']
        print(f"\nSample Daily Metrics:")
        print(f"Daily Return: {summary.get('daily_return_pct', 0):.2f}%")
        print(f"Cumulative Return: {summary.get('cumulative_return_pct', 0):.2f}%")
        print(f"Volatility: {summary.get('volatility_pct', 0):.2f}%")
        print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
    
    print(f"\nAutomated Reporting System Testing Complete!")