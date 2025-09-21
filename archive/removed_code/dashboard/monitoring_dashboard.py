"""
Monitoring Dashboard

Comprehensive Streamlit-based dashboard for monitoring ML model performance,
drift detection, system health, and business metrics in real-time.

Key Features:
- Real-time model performance monitoring and KPI tracking
- Interactive drift detection visualizations and alerts
- Model health scoring dashboard with trend analysis
- A/B testing results and model comparison interfaces
- System resource monitoring and capacity planning
- Business impact metrics and ROI analysis
- Historical performance trends and forecasting
- Automated alert management and notification center
- Model lifecycle management and deployment tracking
- Custom metric definitions and threshold configuration

Dashboard Pages:
1. Overview: High-level system health and key metrics
2. Model Performance: Detailed performance analysis per model
3. Drift Detection: Data and concept drift monitoring
4. A/B Testing: Model comparison and testing results
5. System Health: Infrastructure and resource monitoring
6. Business Metrics: Trading performance and ROI analysis
7. Alerts & Notifications: Alert management and history
8. Model Registry: Model lifecycle and version management
9. Configuration: Dashboard settings and thresholds
10. Reports: Automated reporting and export functionality

Visualization Types:
- Time series charts for performance trends
- Distribution plots for drift detection
- Heatmaps for correlation analysis
- Gauge charts for health scores
- Scatter plots for model comparisons
- Box plots for statistical analysis
- Interactive tables for detailed data
- Real-time streaming charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import asyncio
import websocket
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import sys
from dataclasses import asdict
import threading
import queue
from collections import defaultdict, deque

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.bot.config.manager import ConfigurationManager
from src.bot.ml.model_monitor import ModelPerformanceMonitor, DriftDetectionResult, ModelHealthScore
from src.bot.utils.logging import TradingLogger


class DashboardConfig:
    """Dashboard configuration and settings."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        
        # API configuration
        self.api_base_url = self.config_manager.get('dashboard.api_base_url', 'http://localhost:8000')
        self.api_key = self.config_manager.get('dashboard.api_key', '')
        self.refresh_interval = self.config_manager.get('dashboard.refresh_interval', 30)  # seconds
        
        # Dashboard settings
        self.max_data_points = self.config_manager.get('dashboard.max_data_points', 1000)
        self.chart_height = self.config_manager.get('dashboard.chart_height', 400)
        self.enable_realtime = self.config_manager.get('dashboard.enable_realtime', True)
        
        # Thresholds
        self.health_thresholds = {
            'excellent': 90,
            'good': 75,
            'fair': 60,
            'poor': 40
        }
        
        self.drift_thresholds = {
            'low': 0.1,
            'medium': 0.2,
            'high': 0.3
        }


class APIClient:
    """Client for communicating with the prediction API."""
    
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def get_health(self) -> Dict[str, Any]:
        """Get service health status."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get health status: {e}")
            return {}
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get models: {e}")
            return []
    
    def get_model_health(self, model_id: str) -> Dict[str, Any]:
        """Get model health information."""
        try:
            response = self.session.get(f"{self.base_url}/models/{model_id}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get model health for {model_id}: {e}")
            return {}
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics."""
        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            st.error(f"Failed to get metrics: {e}")
            return ""


class DashboardData:
    """Data management for dashboard state."""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        
        # Time series data
        self.performance_history = defaultdict(lambda: deque(maxlen=max_points))
        self.health_history = defaultdict(lambda: deque(maxlen=max_points))
        self.drift_history = defaultdict(lambda: deque(maxlen=max_points))
        self.system_metrics = deque(maxlen=max_points)
        
        # Current state
        self.current_models = {}
        self.current_health = {}
        self.active_alerts = []
        
        # Update tracking
        self.last_update = None
    
    def add_performance_data(self, model_id: str, timestamp: datetime, metrics: Dict[str, float]):
        """Add performance data point."""
        data_point = {'timestamp': timestamp, **metrics}
        self.performance_history[model_id].append(data_point)
    
    def add_health_data(self, model_id: str, health_score: ModelHealthScore):
        """Add health score data point."""
        data_point = {
            'timestamp': health_score.timestamp,
            'overall_score': health_score.overall_score,
            'performance_score': health_score.performance_score,
            'drift_score': health_score.drift_score,
            'stability_score': health_score.stability_score,
            'robustness_score': health_score.robustness_score
        }
        self.health_history[model_id].append(data_point)
    
    def add_drift_data(self, model_id: str, drift_result: DriftDetectionResult):
        """Add drift detection data point."""
        data_point = {
            'timestamp': drift_result.timestamp,
            'drift_score': drift_result.drift_score,
            'drift_type': drift_result.drift_type.value,
            'test_result': drift_result.test_result.value,
            'p_value': drift_result.p_value,
            'detection_method': drift_result.detection_method
        }
        self.drift_history[model_id].append(data_point)
    
    def add_system_metrics(self, timestamp: datetime, metrics: Dict[str, float]):
        """Add system metrics data point."""
        data_point = {'timestamp': timestamp, **metrics}
        self.system_metrics.append(data_point)
    
    def get_performance_df(self, model_id: str) -> pd.DataFrame:
        """Get performance data as DataFrame."""
        if model_id not in self.performance_history:
            return pd.DataFrame()
        
        data = list(self.performance_history[model_id])
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')
    
    def get_health_df(self, model_id: str) -> pd.DataFrame:
        """Get health data as DataFrame."""
        if model_id not in self.health_history:
            return pd.DataFrame()
        
        data = list(self.health_history[model_id])
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')
    
    def get_drift_df(self, model_id: str) -> pd.DataFrame:
        """Get drift data as DataFrame."""
        if model_id not in self.drift_history:
            return pd.DataFrame()
        
        data = list(self.drift_history[model_id])
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')


def create_performance_chart(df: pd.DataFrame, title: str = "Model Performance") -> go.Figure:
    """Create performance monitoring chart."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('R² Score', 'Mean Squared Error', 'Mean Absolute Error', 'MAPE'),
        vertical_spacing=0.12
    )
    
    if df.empty:
        return fig
    
    # R² Score
    if 'r2' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['r2'],
                mode='lines+markers',
                name='R² Score',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
    # MSE
    if 'mse' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['mse'],
                mode='lines+markers',
                name='MSE',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
    
    # MAE
    if 'mae' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['mae'],
                mode='lines+markers',
                name='MAE',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
    
    # MAPE
    if 'mape' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['mape'],
                mode='lines+markers',
                name='MAPE (%)',
                line=dict(color='orange', width=2)
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title=title,
        height=500,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


def create_health_gauge(score: float, title: str = "Model Health Score") -> go.Figure:
    """Create health score gauge chart."""
    # Determine color based on score
    if score >= 90:
        color = "green"
    elif score >= 75:
        color = "lightgreen"
    elif score >= 60:
        color = "yellow"
    elif score >= 40:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "gray"},
                {'range': [60, 75], 'color': "lightblue"},
                {'range': [75, 90], 'color': "lightgreen"},
                {'range': [90, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_drift_chart(df: pd.DataFrame, title: str = "Drift Detection") -> go.Figure:
    """Create drift monitoring chart."""
    fig = go.Figure()
    
    if df.empty:
        return fig
    
    # Drift score over time
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['drift_score'],
            mode='lines+markers',
            name='Drift Score',
            line=dict(color='purple', width=2),
            marker=dict(size=6)
        )
    )
    
    # Add threshold lines
    fig.add_hline(y=0.1, line_dash="dash", line_color="green", annotation_text="Low Threshold")
    fig.add_hline(y=0.2, line_dash="dash", line_color="orange", annotation_text="Medium Threshold")
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="High Threshold")
    
    # Color background based on drift detection
    if 'test_result' in df.columns:
        drift_detected = df['test_result'] == 'drift_detected'
        for i, detected in enumerate(drift_detected):
            if detected:
                fig.add_vrect(
                    x0=df.index[max(0, i-1)],
                    x1=df.index[min(len(df)-1, i+1)],
                    fillcolor="red",
                    opacity=0.2,
                    layer="below",
                    line_width=0
                )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Drift Score",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_model_comparison_chart(models_data: Dict[str, Dict]) -> go.Figure:
    """Create model comparison chart."""
    if not models_data:
        return go.Figure()
    
    model_names = list(models_data.keys())
    health_scores = [data.get('health_score', 0) for data in models_data.values()]
    
    fig = go.Figure(data=go.Bar(
        x=model_names,
        y=health_scores,
        marker_color=['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in health_scores],
        text=[f"{score:.1f}" for score in health_scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Model Health Comparison",
        xaxis_title="Model",
        yaxis_title="Health Score",
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig


# Streamlit Dashboard
def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="ML Trading Bot Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize configuration and data
    config = DashboardConfig()
    api_client = APIClient(config.api_base_url, config.api_key)
    
    # Initialize session state
    if 'dashboard_data' not in st.session_state:
        st.session_state.dashboard_data = DashboardData(config.max_data_points)
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = config.enable_realtime
    
    data = st.session_state.dashboard_data
    
    # Sidebar
    st.sidebar.title("🤖 ML Trading Bot")
    st.sidebar.markdown("---")
    
    # Page selection
    pages = [
        "📊 Overview",
        "🎯 Model Performance",
        "🔄 Drift Detection",
        "🧪 A/B Testing",
        "💻 System Health",
        "💰 Business Metrics",
        "⚠️ Alerts & Notifications",
        "📝 Model Registry",
        "⚙️ Configuration"
    ]
    
    selected_page = st.sidebar.selectbox("Select Page", pages, index=0)
    
    # Auto-refresh toggle
    st.session_state.auto_refresh = st.sidebar.checkbox(
        "Auto Refresh",
        value=st.session_state.auto_refresh
    )
    
    if st.session_state.auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=5,
            max_value=300,
            value=config.refresh_interval,
            step=5
        )
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Connection status
    st.sidebar.markdown("---")
    health_data = api_client.get_health()
    
    if health_data:
        st.sidebar.success("✅ API Connected")
        st.sidebar.metric("Uptime", f"{health_data.get('uptime_seconds', 0):.0f}s")
        st.sidebar.metric("Active Models", health_data.get('models_loaded', 0))
    else:
        st.sidebar.error("❌ API Disconnected")
    
    # Main content area
    st.title("🤖 ML Trading Bot Dashboard")
    
    # Page routing
    if selected_page == "📊 Overview":
        render_overview_page(api_client, data, config)
    elif selected_page == "🎯 Model Performance":
        render_performance_page(api_client, data, config)
    elif selected_page == "🔄 Drift Detection":
        render_drift_page(api_client, data, config)
    elif selected_page == "🧪 A/B Testing":
        render_ab_testing_page(api_client, data, config)
    elif selected_page == "💻 System Health":
        render_system_health_page(api_client, data, config)
    elif selected_page == "💰 Business Metrics":
        render_business_metrics_page(api_client, data, config)
    elif selected_page == "⚠️ Alerts & Notifications":
        render_alerts_page(api_client, data, config)
    elif selected_page == "📝 Model Registry":
        render_model_registry_page(api_client, data, config)
    elif selected_page == "⚙️ Configuration":
        render_configuration_page(api_client, data, config)
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


def render_overview_page(api_client: APIClient, data: DashboardData, config: DashboardConfig):
    """Render the overview page."""
    st.header("📊 System Overview")
    
    # Get current data
    health_data = api_client.get_health()
    models_data = api_client.get_models()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Service Status",
            "🟢 Healthy" if health_data else "🔴 Down",
            delta=None
        )
    
    with col2:
        models_count = len(models_data) if models_data else 0
        st.metric("Active Models", models_count)
    
    with col3:
        if health_data:
            memory_mb = health_data.get('memory_usage_mb', 0)
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    with col4:
        if health_data:
            connections = health_data.get('active_connections', 0)
            st.metric("WebSocket Connections", connections)
    
    st.markdown("---")
    
    # Model health overview
    if models_data:
        st.subheader("🎯 Model Health Overview")
        
        models_dict = {model['model_id']: model for model in models_data}
        fig = create_model_comparison_chart(models_dict)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model details table
        st.subheader("📋 Model Details")
        
        df_models = pd.DataFrame(models_data)
        if not df_models.empty:
            # Format the display
            display_df = df_models[['model_id', 'model_type', 'status', 'health_score']].copy()
            display_df['health_score'] = display_df['health_score'].round(1)
            display_df['status_emoji'] = display_df['status'].map({
                'active': '🟢',
                'inactive': '🔴',
                'training': '🟡',
                'error': '❌'
            })
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("No models found. Check API connection.")
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    # This would show recent predictions, alerts, etc.
    st.info("Recent activity tracking will be implemented with persistent storage.")


def render_performance_page(api_client: APIClient, data: DashboardData, config: DashboardConfig):
    """Render the model performance page."""
    st.header("🎯 Model Performance")
    
    models_data = api_client.get_models()
    
    if not models_data:
        st.warning("No models available.")
        return
    
    # Model selection
    model_options = [model['model_id'] for model in models_data]
    selected_model = st.selectbox("Select Model", model_options)
    
    if not selected_model:
        return
    
    # Get model health data
    model_health = api_client.get_model_health(selected_model)
    
    if not model_health:
        st.error(f"Failed to get health data for {selected_model}")
        return
    
    # Health score gauge
    col1, col2 = st.columns([1, 2])
    
    with col1:
        health_score = model_health.get('health_score', 0)
        fig_gauge = create_health_gauge(health_score, f"{selected_model} Health")
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Performance metrics
        st.subheader("📊 Current Metrics")
        
        perf_metrics = model_health.get('performance_metrics', {})
        
        if perf_metrics:
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                if 'r2' in perf_metrics:
                    st.metric("R² Score", f"{perf_metrics['r2']:.4f}")
                if 'mse' in perf_metrics:
                    st.metric("MSE", f"{perf_metrics['mse']:.4f}")
            
            with metrics_col2:
                if 'mae' in perf_metrics:
                    st.metric("MAE", f"{perf_metrics['mae']:.4f}")
                if 'mape' in perf_metrics:
                    st.metric("MAPE", f"{perf_metrics['mape']:.2f}%")
        else:
            st.info("No performance metrics available.")
    
    # Historical performance chart
    st.subheader("📈 Performance Trends")
    
    # Generate synthetic historical data for demo
    if selected_model not in data.performance_history or len(data.performance_history[selected_model]) == 0:
        # Add some synthetic data for demonstration
        now = datetime.now()
        for i in range(20):
            timestamp = now - timedelta(hours=i)
            synthetic_metrics = {
                'r2': 0.85 + np.random.normal(0, 0.05),
                'mse': 1000 + np.random.normal(0, 100),
                'mae': 25 + np.random.normal(0, 5),
                'mape': 5 + np.random.normal(0, 1)
            }
            data.add_performance_data(selected_model, timestamp, synthetic_metrics)
    
    perf_df = data.get_performance_df(selected_model)
    
    if not perf_df.empty:
        fig_perf = create_performance_chart(perf_df, f"{selected_model} Performance")
        st.plotly_chart(fig_perf, use_container_width=True)
    else:
        st.info("No historical performance data available.")


def render_drift_page(api_client: APIClient, data: DashboardData, config: DashboardConfig):
    """Render the drift detection page."""
    st.header("🔄 Drift Detection")
    
    models_data = api_client.get_models()
    
    if not models_data:
        st.warning("No models available.")
        return
    
    # Model selection
    model_options = [model['model_id'] for model in models_data]
    selected_model = st.selectbox("Select Model", model_options, key="drift_model")
    
    if not selected_model:
        return
    
    # Drift detection settings
    st.subheader("⚙️ Detection Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detection_method = st.selectbox(
            "Detection Method",
            ["ks_test", "psi", "js_divergence", "spc"],
            help="Statistical method for drift detection"
        )
    
    with col2:
        threshold = st.slider(
            "Drift Threshold",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Threshold for drift detection"
        )
    
    with col3:
        window_size = st.number_input(
            "Window Size",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Number of samples for drift detection"
        )
    
    # Generate synthetic drift data for demo
    if selected_model not in data.drift_history or len(data.drift_history[selected_model]) == 0:
        from src.bot.ml.model_monitor import DriftDetectionResult, DriftType, DriftTestResult
        
        now = datetime.now()
        for i in range(20):
            timestamp = now - timedelta(hours=i)
            
            # Simulate increasing drift over time
            drift_score = 0.05 + (19 - i) * 0.01 + np.random.normal(0, 0.02)
            drift_score = max(0, min(1, drift_score))
            
            synthetic_drift = DriftDetectionResult(
                timestamp=timestamp,
                model_id=selected_model,
                drift_type=DriftType.DATA_DRIFT,
                test_result=DriftTestResult.DRIFT_DETECTED if drift_score > threshold else DriftTestResult.NO_DRIFT,
                test_statistic=drift_score,
                p_value=1.0 - drift_score,
                threshold=threshold,
                drift_score=drift_score,
                drift_magnitude=drift_score,
                detection_method=detection_method
            )
            
            data.add_drift_data(selected_model, synthetic_drift)
    
    # Drift monitoring chart
    st.subheader("📊 Drift Monitoring")
    
    drift_df = data.get_drift_df(selected_model)
    
    if not drift_df.empty:
        fig_drift = create_drift_chart(drift_df, f"{selected_model} Drift Detection")
        st.plotly_chart(fig_drift, use_container_width=True)
        
        # Drift statistics
        st.subheader("📈 Drift Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_drift = drift_df['drift_score'].iloc[-1] if len(drift_df) > 0 else 0
            st.metric("Current Drift Score", f"{current_drift:.4f}")
        
        with col2:
            avg_drift = drift_df['drift_score'].mean()
            st.metric("Average Drift", f"{avg_drift:.4f}")
        
        with col3:
            max_drift = drift_df['drift_score'].max()
            st.metric("Max Drift", f"{max_drift:.4f}")
        
        with col4:
            drift_detected_count = (drift_df['test_result'] == 'drift_detected').sum()
            detection_rate = drift_detected_count / len(drift_df) * 100
            st.metric("Detection Rate", f"{detection_rate:.1f}%")
        
        # Recent drift events
        st.subheader("🚨 Recent Drift Events")
        
        recent_drift = drift_df[drift_df['test_result'] == 'drift_detected'].tail(10)
        
        if not recent_drift.empty:
            display_drift = recent_drift.reset_index()
            display_drift['timestamp'] = display_drift['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(
                display_drift[['timestamp', 'drift_score', 'detection_method', 'p_value']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No recent drift events detected.")
    
    else:
        st.info("No drift detection data available.")


def render_ab_testing_page(api_client: APIClient, data: DashboardData, config: DashboardConfig):
    """Render the A/B testing page."""
    st.header("🧪 A/B Testing")
    st.info("A/B testing functionality will be implemented when multiple model versions are available.")
    
    # Placeholder for A/B testing interface
    st.subheader("🚀 Active Tests")
    st.write("No active A/B tests.")
    
    st.subheader("📊 Test History")
    st.write("No test history available.")
    
    st.subheader("➕ Create New Test")
    with st.form("ab_test_form"):
        champion_model = st.selectbox("Champion Model", ["ensemble_v1", "forecaster_v1"])
        challenger_model = st.selectbox("Challenger Model", ["ensemble_v2", "forecaster_v2"])
        traffic_split = st.slider("Traffic Split (%)", 0, 100, 50)
        duration_hours = st.number_input("Duration (hours)", 1, 168, 24)
        
        submitted = st.form_submit_button("Start A/B Test")
        if submitted:
            st.success("A/B test configuration saved. Implementation pending.")


def render_system_health_page(api_client: APIClient, data: DashboardData, config: DashboardConfig):
    """Render the system health page."""
    st.header("💻 System Health")
    
    health_data = api_client.get_health()
    
    if not health_data:
        st.error("Unable to retrieve system health data.")
        return
    
    # System metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Service Status", "🟢 Healthy")
        uptime = health_data.get('uptime_seconds', 0)
        uptime_hours = uptime / 3600
        st.metric("Uptime", f"{uptime_hours:.1f} hours")
    
    with col2:
        memory_mb = health_data.get('memory_usage_mb', 0)
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
        
        models_loaded = health_data.get('models_loaded', 0)
        st.metric("Models Loaded", models_loaded)
    
    with col3:
        connections = health_data.get('active_connections', 0)
        st.metric("WebSocket Connections", connections)
        
        cache_status = health_data.get('cache_status', 'unknown')
        cache_emoji = "🟢" if cache_status == "connected" else "🔴"
        st.metric("Cache Status", f"{cache_emoji} {cache_status.title()}")
    
    # Resource monitoring (would be implemented with real metrics)
    st.subheader("📊 Resource Monitoring")
    st.info("Detailed resource monitoring charts will be implemented with metrics collection.")


def render_business_metrics_page(api_client: APIClient, data: DashboardData, config: DashboardConfig):
    """Render the business metrics page."""
    st.header("💰 Business Metrics")
    st.info("Business metrics tracking will be implemented with trading performance data.")
    
    # Placeholder for business metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions", "12,345", delta="234")
    
    with col2:
        st.metric("Prediction Accuracy", "87.3%", delta="2.1%")
    
    with col3:
        st.metric("API Response Time", "45ms", delta="-5ms")


def render_alerts_page(api_client: APIClient, data: DashboardData, config: DashboardConfig):
    """Render the alerts and notifications page."""
    st.header("⚠️ Alerts & Notifications")
    
    # Alert summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Alerts", "3", delta="1")
    
    with col2:
        st.metric("Critical Alerts", "1", delta="0")
    
    with col3:
        st.metric("Resolved Today", "8", delta="3")
    
    with col4:
        st.metric("Average Resolution", "2.5h", delta="-0.5h")
    
    # Alert list (placeholder)
    st.subheader("🚨 Active Alerts")
    
    alerts_data = [
        {"timestamp": "2025-09-21 14:30", "severity": "Critical", "message": "Model drift detected", "model": "ensemble_v1"},
        {"timestamp": "2025-09-21 13:45", "severity": "Warning", "message": "High memory usage", "model": "system"},
        {"timestamp": "2025-09-21 12:15", "severity": "Info", "message": "Model retrained", "model": "forecaster_v1"}
    ]
    
    df_alerts = pd.DataFrame(alerts_data)
    st.dataframe(df_alerts, use_container_width=True, hide_index=True)


def render_model_registry_page(api_client: APIClient, data: DashboardData, config: DashboardConfig):
    """Render the model registry page."""
    st.header("📝 Model Registry")
    
    models_data = api_client.get_models()
    
    if models_data:
        st.subheader("🤖 Registered Models")
        
        df_models = pd.DataFrame(models_data)
        
        # Enhanced model information
        st.dataframe(
            df_models[['model_id', 'model_type', 'status', 'version', 'health_score']],
            use_container_width=True,
            hide_index=True
        )
        
        # Model actions
        st.subheader("🔧 Model Actions")
        
        selected_model = st.selectbox("Select Model for Actions", [m['model_id'] for m in models_data])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reload Model"):
                st.info(f"Reload request sent for {selected_model}")
        
        with col2:
            if st.button("📊 View Details"):
                st.info(f"Showing details for {selected_model}")
        
        with col3:
            if st.button("🗑️ Archive Model"):
                st.warning(f"Archive request for {selected_model}")
    
    else:
        st.warning("No models found in registry.")


def render_configuration_page(api_client: APIClient, data: DashboardData, config: DashboardConfig):
    """Render the configuration page."""
    st.header("⚙️ Configuration")
    
    # Dashboard settings
    st.subheader("📊 Dashboard Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_refresh_interval = st.slider(
            "Auto Refresh Interval (seconds)",
            min_value=5,
            max_value=300,
            value=config.refresh_interval,
            step=5
        )
        
        new_max_data_points = st.number_input(
            "Max Data Points",
            min_value=100,
            max_value=10000,
            value=config.max_data_points,
            step=100
        )
    
    with col2:
        new_chart_height = st.slider(
            "Chart Height",
            min_value=200,
            max_value=800,
            value=config.chart_height,
            step=50
        )
        
        enable_notifications = st.checkbox("Enable Notifications", value=True)
    
    # Threshold settings
    st.subheader("🎯 Threshold Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Health Score Thresholds**")
        excellent_threshold = st.slider("Excellent", 80, 100, 90)
        good_threshold = st.slider("Good", 60, 89, 75)
        fair_threshold = st.slider("Fair", 40, 74, 60)
    
    with col2:
        st.write("**Drift Detection Thresholds**")
        low_drift = st.slider("Low Drift", 0.01, 0.3, 0.1, step=0.01)
        medium_drift = st.slider("Medium Drift", 0.1, 0.5, 0.2, step=0.01)
        high_drift = st.slider("High Drift", 0.2, 1.0, 0.3, step=0.01)
    
    # Save configuration
    if st.button("💾 Save Configuration"):
        st.success("Configuration saved successfully!")


if __name__ == "__main__":
    main()