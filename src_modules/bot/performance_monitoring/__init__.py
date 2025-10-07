"""
Performance Monitoring Package
Comprehensive monitoring dashboard with Australian compliance reporting and performance analysis
"""

from .dashboard import (
    RealTimeDashboard,
    AustralianPerformanceCalculator,
    PerformanceMetric,
    DashboardAlert,
    DashboardMetricType,
    AlertLevel
)

from .compliance_reporter import (
    AustralianComplianceReporter,
    ATOReportGenerator,
    ASICReportGenerator,
    ComplianceReport,
    ReportType,
    ReportStatus
)

from .performance_analyzer import (
    AustralianPerformanceAnalyzer,
    StrategyPerformanceAnalyzer,
    AustralianBenchmarkProvider,
    PerformanceMetrics,
    TradeAnalysis,
    PerformancePeriod,
    BenchmarkType
)

__all__ = [
    # Main monitoring components
    'RealTimeDashboard',
    'AustralianComplianceReporter',
    'AustralianPerformanceAnalyzer',
    
    # Performance calculation components
    'AustralianPerformanceCalculator',
    'StrategyPerformanceAnalyzer',
    'AustralianBenchmarkProvider',
    
    # Report generation components
    'ATOReportGenerator',
    'ASICReportGenerator',
    
    # Data structures
    'PerformanceMetric',
    'DashboardAlert',
    'ComplianceReport',
    'PerformanceMetrics',
    'TradeAnalysis',
    
    # Enums
    'DashboardMetricType',
    'AlertLevel',
    'ReportType',
    'ReportStatus',
    'PerformancePeriod',
    'BenchmarkType'
]

# Package version
__version__ = "1.0.0"

# Package metadata
__author__ = "Australian Trading System"
__description__ = "Comprehensive performance monitoring and Australian compliance reporting"
__license__ = "Private"