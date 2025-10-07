"""
Live Trading Package - Phase 5 Implementation

This package contains the complete live trading infrastructure:
- WebSocketManager: Real-time data feeds and order execution streams
- LiveExecutionEngine: Live order placement and position management
- MonitoringDashboard: Real-time performance monitoring and visualization
- AlertSystem: Risk alerts and trade notifications
- ProductionDeploymentPipeline: Automated deployment and service management
"""

from .websocket_manager import WebSocketManager, WebSocketConfig
from .live_execution_engine import (
    LiveExecutionEngine, 
    TradingMode, 
    ExecutionResult, 
    ExecutionStatus
)
from .monitoring_dashboard import (
    MonitoringDashboard, 
    PerformanceMetrics, 
    SystemHealth
)
from .alert_system import (
    AlertSystem,
    Alert,
    AlertRule,
    AlertType,
    AlertSeverity,
    NotificationChannel
)
from .production_deployment import (
    ProductionDeploymentPipeline,
    DeploymentEnvironment,
    DeploymentStatus,
    ServiceStatus
)

__all__ = [
    'WebSocketManager',
    'WebSocketConfig', 
    'LiveExecutionEngine',
    'TradingMode',
    'ExecutionResult',
    'ExecutionStatus',
    'MonitoringDashboard',
    'PerformanceMetrics',
    'SystemHealth',
    'AlertSystem',
    'Alert',
    'AlertRule',
    'AlertType',
    'AlertSeverity',
    'NotificationChannel',
    'ProductionDeploymentPipeline',
    'DeploymentEnvironment',
    'DeploymentStatus',
    'ServiceStatus',
]