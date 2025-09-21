"""
Grafana Dashboard Manager.
Dynamic dashboard creation, management, and visualization system.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import aiohttp
import base64

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class PanelType(Enum):
    """Grafana panel types."""
    TIMESERIES = "timeseries"
    STAT = "stat"
    GAUGE = "gauge"
    BAR_GAUGE = "bargauge"
    TABLE = "table"
    HEATMAP = "heatmap"
    PIE_CHART = "piechart"
    GRAPH = "graph"
    SINGLESTAT = "singlestat"
    TEXT = "text"
    LOGS = "logs"
    NODE_GRAPH = "nodeGraph"

class VisualizationType(Enum):
    """Visualization types."""
    LINE = "line"
    BARS = "bars"
    POINTS = "points"
    AREA = "area"
    STACKED = "stacked"

class AlertState(Enum):
    """Alert states for panels."""
    NO_DATA = "no_data"
    ALERTING = "alerting"
    OK = "ok"
    PENDING = "pending"
    UNKNOWN = "unknown"

@dataclass
class PanelTarget:
    """Panel query target."""
    expr: str
    legend_format: str = ""
    ref_id: str = "A"
    interval: str = ""
    format: str = "time_series"
    instant: bool = False
    hide: bool = False

@dataclass
class PanelThreshold:
    """Panel threshold configuration."""
    value: float
    color: str
    op: str = "gt"  # gt, lt, eq, ne
    fill: bool = True
    line: bool = True

@dataclass
class PanelAlert:
    """Panel alert configuration."""
    name: str
    message: str
    frequency: str
    conditions: List[Dict[str, Any]]
    executionErrorState: str = "alerting"
    noDataState: str = "no_data"
    for_duration: str = "5m"

@dataclass
class Panel:
    """Grafana panel configuration."""
    id: int
    title: str
    type: PanelType
    x: int = 0
    y: int = 0
    width: int = 12
    height: int = 8
    targets: List[PanelTarget] = field(default_factory=list)
    thresholds: List[PanelThreshold] = field(default_factory=list)
    unit: str = "short"
    decimals: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    legend_show: bool = True
    legend_as_table: bool = False
    legend_to_right: bool = False
    null_point_mode: str = "null"
    fill: int = 1
    line_width: int = 1
    points: bool = False
    pointradius: int = 2
    bars: bool = False
    stack: bool = False
    percentage: bool = False
    alert: Optional[PanelAlert] = None
    description: str = ""
    transparent: bool = False
    repeat_for: Optional[str] = None
    datasource: str = "Prometheus"

@dataclass
class DashboardVariable:
    """Dashboard template variable."""
    name: str
    type: str = "query"
    query: str = ""
    label: str = ""
    description: str = ""
    datasource: str = "Prometheus"
    regex: str = ""
    sort: int = 1
    multi: bool = False
    include_all: bool = False
    all_value: str = ""
    current_value: str = ""
    hide: int = 0  # 0=visible, 1=label, 2=variable
    options: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class DashboardAnnotation:
    """Dashboard annotation."""
    name: str
    datasource: str
    enable: bool = True
    expr: str = ""
    title_format: str = ""
    text_format: str = ""
    tag_format: str = ""
    icon_color: str = "rgba(0, 211, 255, 1)"
    line_color: str = "rgba(128, 0, 128, 1)"
    hide: bool = False

@dataclass
class DashboardTime:
    """Dashboard time range."""
    from_time: str = "now-1h"
    to_time: str = "now"
    refresh_intervals: List[str] = field(default_factory=lambda: ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"])
    time_options: List[str] = field(default_factory=lambda: ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d"])

class GrafanaDashboardManager:
    """Grafana dashboard management system."""
    
    def __init__(self, grafana_url: str = "http://localhost:3000", username: str = "admin", password: str = "admin123"):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        self.grafana_url = grafana_url.rstrip('/')
        self.username = username
        self.password = password
        
        # Dashboard management
        self.dashboards: Dict[str, Dict[str, Any]] = {}
        self.folders: Dict[str, Dict[str, Any]] = {}
        self.datasources: Dict[str, Dict[str, Any]] = {}
        
        # Panel ID counter
        self.next_panel_id = 1
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_headers: Dict[str, str] = {}
        
        self.logger.info("GrafanaDashboardManager initialized")
    
    async def initialize(self):
        """Initialize Grafana connection."""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Setup authentication
            auth_string = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            self.auth_headers = {
                'Authorization': f'Basic {auth_string}',
                'Content-Type': 'application/json'
            }
            
            # Test connection
            await self._test_connection()
            
            # Initialize default datasources
            await self._setup_default_datasources()
            
            # Create default folders
            await self._setup_default_folders()
            
            self.logger.info("Grafana connection initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Grafana connection: {e}")
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.session:
                await self.session.close()
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup Grafana manager: {e}")
    
    async def _test_connection(self):
        """Test Grafana connection."""
        try:
            url = f"{self.grafana_url}/api/health"
            async with self.session.get(url, headers=self.auth_headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(f"Grafana health check: {data}")
                else:
                    raise Exception(f"Health check failed: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Grafana connection test failed: {e}")
            raise
    
    async def _setup_default_datasources(self):
        """Setup default datasources."""
        try:
            # Prometheus datasource
            prometheus_ds = {
                "name": "Prometheus",
                "type": "prometheus",
                "url": "http://localhost:9090",
                "access": "proxy",
                "isDefault": True,
                "jsonData": {
                    "timeInterval": "15s",
                    "queryTimeout": "60s",
                    "httpMethod": "POST"
                }
            }
            
            await self._create_datasource(prometheus_ds)
            
            # Elasticsearch datasource for logs
            elasticsearch_ds = {
                "name": "Elasticsearch",
                "type": "elasticsearch",
                "url": "http://localhost:9200",
                "access": "proxy",
                "database": "trading-logs-*",
                "jsonData": {
                    "timeField": "@timestamp",
                    "esVersion": "7.10.0",
                    "interval": "Daily",
                    "maxConcurrentShardRequests": 5
                }
            }
            
            await self._create_datasource(elasticsearch_ds)
            
        except Exception as e:
            self.logger.error(f"Failed to setup default datasources: {e}")
    
    async def _setup_default_folders(self):
        """Setup default dashboard folders."""
        try:
            folders = [
                {"title": "Trading System", "uid": "trading"},
                {"title": "System Performance", "uid": "system"},
                {"title": "ML Models", "uid": "ml"},
                {"title": "Infrastructure", "uid": "infrastructure"},
                {"title": "Risk Management", "uid": "risk"}
            ]
            
            for folder in folders:
                await self._create_folder(folder)
                
        except Exception as e:
            self.logger.error(f"Failed to setup default folders: {e}")
    
    async def _create_datasource(self, datasource_config: Dict[str, Any]):
        """Create or update datasource."""
        try:
            url = f"{self.grafana_url}/api/datasources"
            
            async with self.session.post(url, headers=self.auth_headers, json=datasource_config) as response:
                if response.status in [200, 409]:  # 409 = already exists
                    data = await response.json()
                    self.datasources[datasource_config['name']] = data
                    self.logger.info(f"Datasource {datasource_config['name']} created/updated")
                else:
                    self.logger.warning(f"Failed to create datasource: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to create datasource: {e}")
    
    async def _create_folder(self, folder_config: Dict[str, str]):
        """Create dashboard folder."""
        try:
            url = f"{self.grafana_url}/api/folders"
            
            async with self.session.post(url, headers=self.auth_headers, json=folder_config) as response:
                if response.status in [200, 412]:  # 412 = already exists
                    data = await response.json()
                    self.folders[folder_config['uid']] = data
                    self.logger.info(f"Folder {folder_config['title']} created")
                else:
                    self.logger.warning(f"Failed to create folder: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to create folder: {e}")
    
    def create_panel(self, title: str, panel_type: PanelType, query: str, 
                    x: int = 0, y: int = 0, width: int = 12, height: int = 8,
                    **kwargs) -> Panel:
        """Create a dashboard panel."""
        try:
            panel = Panel(
                id=self.next_panel_id,
                title=title,
                type=panel_type,
                x=x, y=y, width=width, height=height,
                targets=[PanelTarget(expr=query)],
                **kwargs
            )
            
            self.next_panel_id += 1
            return panel
            
        except Exception as e:
            self.logger.error(f"Failed to create panel: {e}")
            return None
    
    def create_trading_overview_dashboard(self) -> Dict[str, Any]:
        """Create comprehensive trading overview dashboard."""
        try:
            panels = []
            
            # Portfolio metrics row
            panels.extend([
                self.create_panel(
                    "Portfolio Value", PanelType.STAT, 
                    "trading_portfolio_value",
                    x=0, y=0, width=6, height=4,
                    unit="currencyUSD"
                ),
                self.create_panel(
                    "Total PnL", PanelType.STAT,
                    "sum(trading_pnl_realized)",
                    x=6, y=0, width=6, height=4,
                    unit="currencyUSD"
                ),
                self.create_panel(
                    "Daily PnL", PanelType.STAT,
                    "sum(increase(trading_pnl_realized[1d]))",
                    x=12, y=0, width=6, height=4,
                    unit="currencyUSD"
                ),
                self.create_panel(
                    "Win Rate", PanelType.STAT,
                    "rate(trading_orders_total{status=\"success\"}[1h]) / rate(trading_orders_total[1h]) * 100",
                    x=18, y=0, width=6, height=4,
                    unit="percent"
                )
            ])
            
            # Trading activity row
            panels.extend([
                self.create_panel(
                    "Orders per Second", PanelType.TIMESERIES,
                    "rate(trading_orders_total[1m])",
                    x=0, y=4, width=12, height=6
                ),
                self.create_panel(
                    "Order Success Rate", PanelType.TIMESERIES,
                    "rate(trading_orders_total{status=\"success\"}[5m]) / rate(trading_orders_total[5m]) * 100",
                    x=12, y=4, width=12, height=6,
                    unit="percent"
                )
            ])
            
            # Performance metrics row
            panels.extend([
                self.create_panel(
                    "PnL by Strategy", PanelType.TIMESERIES,
                    "trading_pnl_realized",
                    x=0, y=10, width=12, height=6
                ),
                self.create_panel(
                    "Risk Exposure", PanelType.GAUGE,
                    "trading_risk_exposure",
                    x=12, y=10, width=6, height=6,
                    unit="percent",
                    max_value=100
                ),
                self.create_panel(
                    "Position Count", PanelType.STAT,
                    "count(trading_position_size > 0)",
                    x=18, y=10, width=6, height=6
                )
            ])
            
            # Order execution metrics
            panels.extend([
                self.create_panel(
                    "Order Execution Time (95th percentile)", PanelType.TIMESERIES,
                    "histogram_quantile(0.95, rate(trading_order_execution_duration_bucket[5m]))",
                    x=0, y=16, width=12, height=6,
                    unit="s"
                ),
                self.create_panel(
                    "Order Types Distribution", PanelType.PIE_CHART,
                    "sum by (order_type) (rate(trading_orders_total[1h]))",
                    x=12, y=16, width=12, height=6
                )
            ])
            
            # Create dashboard configuration
            dashboard_config = self._create_dashboard_config(
                title="Trading System Overview",
                uid="trading-overview",
                panels=panels,
                folder_uid="trading",
                tags=["trading", "overview", "portfolio"],
                time_range=DashboardTime(from_time="now-6h"),
                variables=[
                    DashboardVariable(
                        name="strategy",
                        query="label_values(trading_pnl_realized, strategy)",
                        label="Strategy",
                        multi=True,
                        include_all=True
                    ),
                    DashboardVariable(
                        name="symbol",
                        query="label_values(trading_pnl_realized, symbol)",
                        label="Symbol",
                        multi=True,
                        include_all=True
                    )
                ]
            )
            
            return dashboard_config
            
        except Exception as e:
            self.logger.error(f"Failed to create trading overview dashboard: {e}")
            return {}
    
    def create_system_performance_dashboard(self) -> Dict[str, Any]:
        """Create system performance monitoring dashboard."""
        try:
            panels = []
            
            # Service health row
            panels.extend([
                self.create_panel(
                    "Service Uptime", PanelType.STAT,
                    "avg(up) * 100",
                    x=0, y=0, width=6, height=4,
                    unit="percent"
                ),
                self.create_panel(
                    "Response Time (95th)", PanelType.STAT,
                    "histogram_quantile(0.95, rate(trading_service_request_duration_bucket[5m]))",
                    x=6, y=0, width=6, height=4,
                    unit="s"
                ),
                self.create_panel(
                    "Request Rate", PanelType.STAT,
                    "sum(rate(trading_service_requests_total[1m]))",
                    x=12, y=0, width=6, height=4,
                    unit="reqps"
                ),
                self.create_panel(
                    "Error Rate", PanelType.STAT,
                    "sum(rate(trading_service_requests_total{status=~\"5..\"}[5m])) / sum(rate(trading_service_requests_total[5m])) * 100",
                    x=18, y=0, width=6, height=4,
                    unit="percent"
                )
            ])
            
            # Response time trends
            panels.extend([
                self.create_panel(
                    "Response Time Percentiles", PanelType.TIMESERIES,
                    "histogram_quantile(0.50, rate(trading_service_request_duration_bucket[5m]))\nhistogram_quantile(0.95, rate(trading_service_request_duration_bucket[5m]))\nhistogram_quantile(0.99, rate(trading_service_request_duration_bucket[5m]))",
                    x=0, y=4, width=12, height=6,
                    unit="s"
                ),
                self.create_panel(
                    "Request Volume", PanelType.TIMESERIES,
                    "sum by (service) (rate(trading_service_requests_total[1m]))",
                    x=12, y=4, width=12, height=6,
                    unit="reqps"
                )
            ])
            
            # Resource utilization row
            panels.extend([
                self.create_panel(
                    "Memory Usage", PanelType.TIMESERIES,
                    "trading_service_memory_usage / 1024 / 1024 / 1024",
                    x=0, y=10, width=8, height=6,
                    unit="gbytes"
                ),
                self.create_panel(
                    "CPU Usage", PanelType.TIMESERIES,
                    "trading_service_cpu_usage",
                    x=8, y=10, width=8, height=6,
                    unit="percent"
                ),
                self.create_panel(
                    "Resource Usage Summary", PanelType.GAUGE,
                    "avg(trading_service_cpu_usage)",
                    x=16, y=10, width=8, height=6,
                    unit="percent",
                    max_value=100
                )
            ])
            
            # Error analysis
            panels.extend([
                self.create_panel(
                    "Error Rate by Service", PanelType.TIMESERIES,
                    "sum by (service) (rate(trading_service_requests_total{status=~\"5..\"}[5m]))",
                    x=0, y=16, width=12, height=6,
                    unit="reqps"
                ),
                self.create_panel(
                    "Top Error Endpoints", PanelType.TABLE,
                    "topk(10, sum by (endpoint) (rate(trading_service_requests_total{status=~\"5..\"}[1h])))",
                    x=12, y=16, width=12, height=6
                )
            ])
            
            dashboard_config = self._create_dashboard_config(
                title="System Performance",
                uid="system-performance",
                panels=panels,
                folder_uid="system",
                tags=["system", "performance", "monitoring"],
                time_range=DashboardTime(from_time="now-1h"),
                variables=[
                    DashboardVariable(
                        name="service",
                        query="label_values(trading_service_requests_total, service)",
                        label="Service",
                        multi=True,
                        include_all=True
                    )
                ]
            )
            
            return dashboard_config
            
        except Exception as e:
            self.logger.error(f"Failed to create system performance dashboard: {e}")
            return {}
    
    def create_ml_models_dashboard(self) -> Dict[str, Any]:
        """Create ML models monitoring dashboard."""
        try:
            panels = []
            
            # Model performance overview
            panels.extend([
                self.create_panel(
                    "Model Accuracy", PanelType.GAUGE,
                    "avg(trading_model_accuracy)",
                    x=0, y=0, width=6, height=6,
                    unit="percent",
                    max_value=100
                ),
                self.create_panel(
                    "Predictions per Minute", PanelType.STAT,
                    "sum(rate(trading_model_predictions_total[1m])) * 60",
                    x=6, y=0, width=6, height=6
                ),
                self.create_panel(
                    "Inference Time (95th)", PanelType.STAT,
                    "histogram_quantile(0.95, rate(trading_model_inference_duration_bucket[5m]))",
                    x=12, y=0, width=6, height=6,
                    unit="s"
                ),
                self.create_panel(
                    "Active Models", PanelType.STAT,
                    "count(count by (model_name) (trading_model_predictions_total))",
                    x=18, y=0, width=6, height=6
                )
            ])
            
            # Model accuracy trends
            panels.extend([
                self.create_panel(
                    "Model Accuracy Over Time", PanelType.TIMESERIES,
                    "trading_model_accuracy",
                    x=0, y=6, width=12, height=6,
                    unit="percent"
                ),
                self.create_panel(
                    "Prediction Volume by Model", PanelType.TIMESERIES,
                    "sum by (model_name) (rate(trading_model_predictions_total[1m]))",
                    x=12, y=6, width=12, height=6
                )
            ])
            
            # Performance metrics
            panels.extend([
                self.create_panel(
                    "Inference Duration Distribution", PanelType.HEATMAP,
                    "rate(trading_model_inference_duration_bucket[5m])",
                    x=0, y=12, width=12, height=6
                ),
                self.create_panel(
                    "Model Performance Comparison", PanelType.BAR_GAUGE,
                    "trading_model_accuracy",
                    x=12, y=12, width=12, height=6,
                    unit="percent"
                )
            ])
            
            # Feature importance and drift
            panels.extend([
                self.create_panel(
                    "Feature Drift Score", PanelType.TIMESERIES,
                    "trading_feature_drift_score",
                    x=0, y=18, width=12, height=6
                ),
                self.create_panel(
                    "Model Retraining Events", PanelType.TIMESERIES,
                    "increase(trading_model_retraining_total[1h])",
                    x=12, y=18, width=12, height=6
                )
            ])
            
            dashboard_config = self._create_dashboard_config(
                title="ML Models Performance",
                uid="ml-models",
                panels=panels,
                folder_uid="ml",
                tags=["ml", "models", "ai"],
                time_range=DashboardTime(from_time="now-2h"),
                variables=[
                    DashboardVariable(
                        name="model",
                        query="label_values(trading_model_accuracy, model_name)",
                        label="Model",
                        multi=True,
                        include_all=True
                    ),
                    DashboardVariable(
                        name="version",
                        query="label_values(trading_model_predictions_total, model_version)",
                        label="Version",
                        multi=True,
                        include_all=True
                    )
                ]
            )
            
            return dashboard_config
            
        except Exception as e:
            self.logger.error(f"Failed to create ML models dashboard: {e}")
            return {}
    
    def _create_dashboard_config(self, title: str, uid: str, panels: List[Panel],
                               folder_uid: str = "", tags: List[str] = None,
                               time_range: DashboardTime = None, 
                               variables: List[DashboardVariable] = None,
                               annotations: List[DashboardAnnotation] = None) -> Dict[str, Any]:
        """Create Grafana dashboard configuration."""
        try:
            tags = tags or []
            variables = variables or []
            annotations = annotations or []
            time_range = time_range or DashboardTime()
            
            # Convert panels to Grafana format
            grafana_panels = []
            for panel in panels:
                if panel:
                    grafana_panel = self._convert_panel_to_grafana(panel)
                    grafana_panels.append(grafana_panel)
            
            # Convert variables to Grafana format
            grafana_variables = []
            for var in variables:
                grafana_var = self._convert_variable_to_grafana(var)
                grafana_variables.append(grafana_var)
            
            # Convert annotations to Grafana format
            grafana_annotations = []
            for annotation in annotations:
                grafana_annotation = self._convert_annotation_to_grafana(annotation)
                grafana_annotations.append(grafana_annotation)
            
            dashboard_config = {
                "dashboard": {
                    "id": None,
                    "uid": uid,
                    "title": title,
                    "tags": tags,
                    "timezone": "browser",
                    "panels": grafana_panels,
                    "time": {
                        "from": time_range.from_time,
                        "to": time_range.to_time
                    },
                    "timepicker": {
                        "refresh_intervals": time_range.refresh_intervals,
                        "time_options": time_range.time_options
                    },
                    "templating": {
                        "list": grafana_variables
                    },
                    "annotations": {
                        "list": grafana_annotations
                    },
                    "refresh": "30s",
                    "schemaVersion": 27,
                    "version": 1,
                    "links": []
                },
                "folderId": self.folders.get(folder_uid, {}).get('id', 0),
                "overwrite": True
            }
            
            return dashboard_config
            
        except Exception as e:
            self.logger.error(f"Failed to create dashboard config: {e}")
            return {}
    
    def _convert_panel_to_grafana(self, panel: Panel) -> Dict[str, Any]:
        """Convert Panel to Grafana panel format."""
        try:
            # Base panel structure
            grafana_panel = {
                "id": panel.id,
                "title": panel.title,
                "type": panel.type.value,
                "gridPos": {
                    "h": panel.height,
                    "w": panel.width,
                    "x": panel.x,
                    "y": panel.y
                },
                "targets": [],
                "datasource": panel.datasource,
                "description": panel.description,
                "transparent": panel.transparent
            }
            
            # Add targets
            for i, target in enumerate(panel.targets):
                grafana_target = {
                    "expr": target.expr,
                    "legendFormat": target.legend_format,
                    "refId": target.ref_id or chr(65 + i),  # A, B, C, ...
                    "interval": target.interval,
                    "format": target.format,
                    "instant": target.instant,
                    "hide": target.hide
                }
                grafana_panel["targets"].append(grafana_target)
            
            # Panel-specific configurations
            if panel.type in [PanelType.TIMESERIES, PanelType.GRAPH]:
                grafana_panel["fieldConfig"] = {
                    "defaults": {
                        "unit": panel.unit,
                        "decimals": panel.decimals,
                        "min": panel.min_value,
                        "max": panel.max_value,
                        "thresholds": {
                            "steps": [{"color": "green", "value": None}] + 
                                   [{"color": t.color, "value": t.value} for t in panel.thresholds]
                        }
                    }
                }
                
                grafana_panel["options"] = {
                    "legend": {
                        "displayMode": "table" if panel.legend_as_table else "list",
                        "placement": "right" if panel.legend_to_right else "bottom",
                        "showLegend": panel.legend_show
                    }
                }
                
            elif panel.type == PanelType.STAT:
                grafana_panel["fieldConfig"] = {
                    "defaults": {
                        "unit": panel.unit,
                        "decimals": panel.decimals,
                        "thresholds": {
                            "steps": [{"color": "green", "value": None}] + 
                                   [{"color": t.color, "value": t.value} for t in panel.thresholds]
                        }
                    }
                }
                
                grafana_panel["options"] = {
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"],
                        "fields": ""
                    },
                    "orientation": "auto",
                    "textMode": "auto",
                    "colorMode": "value",
                    "graphMode": "area"
                }
                
            elif panel.type == PanelType.GAUGE:
                grafana_panel["fieldConfig"] = {
                    "defaults": {
                        "unit": panel.unit,
                        "min": panel.min_value or 0,
                        "max": panel.max_value or 100,
                        "thresholds": {
                            "steps": [{"color": "green", "value": None}] + 
                                   [{"color": t.color, "value": t.value} for t in panel.thresholds]
                        }
                    }
                }
                
                grafana_panel["options"] = {
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"],
                        "fields": ""
                    },
                    "orientation": "auto",
                    "textMode": "auto",
                    "colorMode": "value"
                }
            
            # Add alerts if configured
            if panel.alert:
                grafana_panel["alert"] = self._convert_alert_to_grafana(panel.alert)
            
            return grafana_panel
            
        except Exception as e:
            self.logger.error(f"Failed to convert panel to Grafana format: {e}")
            return {}
    
    def _convert_variable_to_grafana(self, variable: DashboardVariable) -> Dict[str, Any]:
        """Convert DashboardVariable to Grafana format."""
        try:
            return {
                "name": variable.name,
                "type": variable.type,
                "label": variable.label,
                "description": variable.description,
                "query": variable.query,
                "datasource": variable.datasource,
                "regex": variable.regex,
                "sort": variable.sort,
                "multi": variable.multi,
                "includeAll": variable.include_all,
                "allValue": variable.all_value,
                "current": {"value": variable.current_value},
                "hide": variable.hide,
                "options": variable.options,
                "refresh": 1,
                "skipUrlSync": False
            }
            
        except Exception as e:
            self.logger.error(f"Failed to convert variable to Grafana format: {e}")
            return {}
    
    def _convert_annotation_to_grafana(self, annotation: DashboardAnnotation) -> Dict[str, Any]:
        """Convert DashboardAnnotation to Grafana format."""
        try:
            return {
                "name": annotation.name,
                "datasource": annotation.datasource,
                "enable": annotation.enable,
                "expr": annotation.expr,
                "titleFormat": annotation.title_format,
                "textFormat": annotation.text_format,
                "tagsFormat": annotation.tag_format,
                "iconColor": annotation.icon_color,
                "lineColor": annotation.line_color,
                "hide": annotation.hide
            }
            
        except Exception as e:
            self.logger.error(f"Failed to convert annotation to Grafana format: {e}")
            return {}
    
    def _convert_alert_to_grafana(self, alert: PanelAlert) -> Dict[str, Any]:
        """Convert PanelAlert to Grafana format."""
        try:
            return {
                "name": alert.name,
                "message": alert.message,
                "frequency": alert.frequency,
                "conditions": alert.conditions,
                "executionErrorState": alert.executionErrorState,
                "noDataState": alert.noDataState,
                "for": alert.for_duration
            }
            
        except Exception as e:
            self.logger.error(f"Failed to convert alert to Grafana format: {e}")
            return {}
    
    async def create_dashboard(self, dashboard_config: Dict[str, Any]) -> bool:
        """Create dashboard in Grafana."""
        try:
            url = f"{self.grafana_url}/api/dashboards/db"
            
            async with self.session.post(url, headers=self.auth_headers, json=dashboard_config) as response:
                if response.status == 200:
                    data = await response.json()
                    dashboard_uid = dashboard_config['dashboard']['uid']
                    self.dashboards[dashboard_uid] = data
                    self.logger.info(f"Dashboard {dashboard_config['dashboard']['title']} created successfully")
                    return True
                else:
                    error_data = await response.text()
                    self.logger.error(f"Failed to create dashboard: {response.status} - {error_data}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            return False
    
    async def update_dashboard(self, dashboard_uid: str, dashboard_config: Dict[str, Any]) -> bool:
        """Update existing dashboard."""
        try:
            # Get current dashboard version
            current_dashboard = await self.get_dashboard(dashboard_uid)
            if current_dashboard:
                dashboard_config['dashboard']['version'] = current_dashboard.get('version', 0) + 1
            
            return await self.create_dashboard(dashboard_config)
            
        except Exception as e:
            self.logger.error(f"Failed to update dashboard: {e}")
            return False
    
    async def get_dashboard(self, dashboard_uid: str) -> Optional[Dict[str, Any]]:
        """Get dashboard by UID."""
        try:
            url = f"{self.grafana_url}/api/dashboards/uid/{dashboard_uid}"
            
            async with self.session.get(url, headers=self.auth_headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['dashboard']
                else:
                    self.logger.warning(f"Dashboard {dashboard_uid} not found: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to get dashboard: {e}")
            return None
    
    async def delete_dashboard(self, dashboard_uid: str) -> bool:
        """Delete dashboard by UID."""
        try:
            url = f"{self.grafana_url}/api/dashboards/uid/{dashboard_uid}"
            
            async with self.session.delete(url, headers=self.auth_headers) as response:
                if response.status == 200:
                    if dashboard_uid in self.dashboards:
                        del self.dashboards[dashboard_uid]
                    self.logger.info(f"Dashboard {dashboard_uid} deleted")
                    return True
                else:
                    self.logger.error(f"Failed to delete dashboard: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to delete dashboard: {e}")
            return False
    
    async def setup_default_dashboards(self):
        """Setup all default dashboards."""
        try:
            dashboards = [
                self.create_trading_overview_dashboard(),
                self.create_system_performance_dashboard(),
                self.create_ml_models_dashboard()
            ]
            
            for dashboard_config in dashboards:
                if dashboard_config:
                    await self.create_dashboard(dashboard_config)
                    await asyncio.sleep(1)  # Rate limiting
            
            self.logger.info("Default dashboards created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup default dashboards: {e}")
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard management summary."""
        try:
            return {
                'grafana_url': self.grafana_url,
                'connection_status': 'connected' if self.session else 'disconnected',
                'dashboards': {
                    'total': len(self.dashboards),
                    'available': list(self.dashboards.keys())
                },
                'folders': {
                    'total': len(self.folders),
                    'available': list(self.folders.keys())
                },
                'datasources': {
                    'total': len(self.datasources),
                    'available': list(self.datasources.keys())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate dashboard summary: {e}")
            return {'error': 'Unable to generate summary'}