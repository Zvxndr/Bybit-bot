"""
API Health Monitoring System

Comprehensive monitoring system for all external APIs used by the trading bot.
Provides health checks, performance monitoring, fallback strategies, and alerting.

Monitored APIs:
- Exchange APIs (Bybit, Binance, OKX)
- Sentiment APIs (CryptoPanic, Alternative.me)
- Data validation and quality monitoring
- System health indicators

Key Features:
- Real-time API health monitoring
- Performance metrics tracking
- Automatic fallback strategies
- Alert system for critical failures
- Historical health data storage
"""

import asyncio
import aiohttp
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import statistics
from collections import deque, defaultdict
import ccxt

from ..utils.logging import TradingLogger
from ..config.manager import ConfigurationManager


class HealthStatus(Enum):
    """API health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetrics:
    """Health metrics for an API endpoint."""
    api_name: str
    status: HealthStatus
    response_time_ms: float
    success_rate: float
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    consecutive_failures: int
    total_requests: int
    total_failures: int
    average_response_time: float
    p95_response_time: float
    error_details: Optional[str] = None


@dataclass
class HealthCheck:
    """Configuration for a health check."""
    name: str
    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    timeout: float = 10.0
    expected_status: int = 200
    check_interval: float = 60.0  # seconds
    custom_validator: Optional[Callable] = None


@dataclass
class Alert:
    """Alert message structure."""
    timestamp: datetime
    api_name: str
    level: AlertLevel
    message: str
    metrics: Dict[str, Any]


class APIHealthMonitor:
    """
    Comprehensive API health monitoring system.
    
    Monitors all external APIs, tracks performance metrics, implements fallback
    strategies, and provides alerting for critical issues.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = TradingLogger("APIHealthMonitor")
        
        # Health check configurations
        self.health_checks = self._initialize_health_checks()
        
        # Metrics storage
        self.health_metrics: Dict[str, HealthMetrics] = {}
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.alerts: List[Alert] = []
        self.max_alerts = 1000
        
        # Monitoring configuration
        self.monitoring_config = {
            'check_interval': config_manager.get('monitoring.check_interval', 60.0),
            'alert_cooldown': config_manager.get('monitoring.alert_cooldown', 300),  # 5 minutes
            'max_consecutive_failures': config_manager.get('monitoring.max_consecutive_failures', 3),
            'response_time_threshold': config_manager.get('monitoring.response_time_threshold', 5000),  # 5 seconds
            'success_rate_threshold': config_manager.get('monitoring.success_rate_threshold', 0.95),
            'enable_alerts': config_manager.get('monitoring.enable_alerts', True)
        }
        
        # Alert tracking
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Fallback strategies
        self.fallback_strategies = self._initialize_fallback_strategies()
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Initialize metrics
        self._initialize_metrics()
    
    def _initialize_health_checks(self) -> Dict[str, HealthCheck]:
        """Initialize health check configurations for all APIs."""
        checks = {}
        
        # Bybit health check
        checks['bybit'] = HealthCheck(
            name='bybit',
            url='https://api.bybit.com/v5/market/time',
            timeout=5.0,
            check_interval=30.0,
            custom_validator=self._validate_bybit_response
        )
        
        # Binance health check
        checks['binance'] = HealthCheck(
            name='binance',
            url='https://api.binance.com/api/v3/ping',
            timeout=5.0,
            check_interval=30.0
        )
        
        # OKX health check
        checks['okx'] = HealthCheck(
            name='okx',
            url='https://www.okx.com/api/v5/public/time',
            timeout=5.0,
            check_interval=30.0
        )
        
        # CryptoPanic health check
        cryptopanic_key = self.config_manager.get('sentiment.cryptopanic.api_key')
        if cryptopanic_key:
            checks['cryptopanic'] = HealthCheck(
                name='cryptopanic',
                url=f'https://cryptopanic.com/api/v1/posts/?auth_token={cryptopanic_key}&public=true',
                timeout=10.0,
                check_interval=120.0,  # Less frequent for news API
                custom_validator=self._validate_cryptopanic_response
            )
        
        # Fear & Greed Index health check
        checks['fear_greed'] = HealthCheck(
            name='fear_greed',
            url='https://api.alternative.me/fng/',
            timeout=10.0,
            check_interval=300.0,  # 5 minutes, data updates daily
            custom_validator=self._validate_fear_greed_response
        )
        
        return checks
    
    def _initialize_fallback_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize fallback strategies for each API."""
        return {
            'binance': {
                'primary_fallback': 'okx',
                'secondary_fallback': 'bybit_only',
                'weight_reduction': 0.5
            },
            'okx': {
                'primary_fallback': 'binance',
                'secondary_fallback': 'bybit_only',
                'weight_reduction': 0.5
            },
            'cryptopanic': {
                'primary_fallback': 'fear_greed_only',
                'weight_reduction': 0.3
            },
            'fear_greed': {
                'primary_fallback': 'news_only',
                'weight_reduction': 0.4
            }
        }
    
    def _initialize_metrics(self):
        """Initialize health metrics for all APIs."""
        for api_name in self.health_checks:
            self.health_metrics[api_name] = HealthMetrics(
                api_name=api_name,
                status=HealthStatus.UNKNOWN,
                response_time_ms=0.0,
                success_rate=1.0,
                last_success=None,
                last_failure=None,
                consecutive_failures=0,
                total_requests=0,
                total_failures=0,
                average_response_time=0.0,
                p95_response_time=0.0
            )
    
    async def start_monitoring(self):
        """Start the continuous monitoring process."""
        if self.is_monitoring:
            self.logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("API health monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring process."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("API health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            while self.is_monitoring:
                try:
                    # Perform health checks for all APIs
                    tasks = []
                    for api_name, health_check in self.health_checks.items():
                        if self._should_check_api(api_name, health_check):
                            task = asyncio.create_task(
                                self._perform_health_check(session, health_check)
                            )
                            tasks.append(task)
                    
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for result in results:
                            if isinstance(result, Exception):
                                self.logger.error(f"Health check error: {result}")
                    
                    # Update overall system health
                    self._update_system_health()
                    
                    # Clean up old alerts
                    self._cleanup_old_alerts()
                    
                    # Wait before next round of checks
                    await asyncio.sleep(self.monitoring_config['check_interval'])
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(10)  # Brief pause before retrying
    
    def _should_check_api(self, api_name: str, health_check: HealthCheck) -> bool:
        """Determine if an API should be checked based on its interval."""
        metrics = self.health_metrics[api_name]
        if not metrics.last_success and not metrics.last_failure:
            return True  # First check
        
        last_check = max(
            metrics.last_success or datetime.min,
            metrics.last_failure or datetime.min
        )
        
        time_since_check = (datetime.now() - last_check).total_seconds()
        return time_since_check >= health_check.check_interval
    
    async def _perform_health_check(self, session: aiohttp.ClientSession, health_check: HealthCheck):
        """Perform a health check for a specific API."""
        api_name = health_check.name
        start_time = time.time()
        
        try:
            # Make the HTTP request
            async with session.request(
                health_check.method,
                health_check.url,
                headers=health_check.headers,
                timeout=aiohttp.ClientTimeout(total=health_check.timeout)
            ) as response:
                response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Check if response is successful
                success = response.status == health_check.expected_status
                
                # Custom validation if provided
                if success and health_check.custom_validator:
                    try:
                        response_data = await response.json()
                        success = health_check.custom_validator(response_data)
                    except Exception as e:
                        success = False
                        self.logger.debug(f"Custom validation failed for {api_name}: {e}")
                
                # Update metrics
                self._update_health_metrics(
                    api_name, 
                    success, 
                    response_time, 
                    None if success else f"HTTP {response.status}"
                )
                
        except asyncio.TimeoutError:
            response_time = health_check.timeout * 1000
            self._update_health_metrics(api_name, False, response_time, "Timeout")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._update_health_metrics(api_name, False, response_time, str(e))
    
    def _update_health_metrics(self, api_name: str, success: bool, response_time_ms: float, error_details: Optional[str]):
        """Update health metrics for an API."""
        metrics = self.health_metrics[api_name]
        now = datetime.now()
        
        # Update basic counters
        metrics.total_requests += 1
        metrics.response_time_ms = response_time_ms
        
        # Track response times
        self.response_times[api_name].append(response_time_ms)
        
        if success:
            metrics.last_success = now
            metrics.consecutive_failures = 0
            metrics.error_details = None
        else:
            metrics.last_failure = now
            metrics.consecutive_failures += 1
            metrics.total_failures += 1
            metrics.error_details = error_details
        
        # Calculate success rate (exponential moving average)
        alpha = 0.1
        current_success = 1.0 if success else 0.0
        metrics.success_rate = (1 - alpha) * metrics.success_rate + alpha * current_success
        
        # Calculate response time statistics
        response_times = list(self.response_times[api_name])
        if response_times:
            metrics.average_response_time = statistics.mean(response_times)
            metrics.p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
        
        # Determine health status
        metrics.status = self._determine_health_status(metrics)
        
        # Generate alerts if necessary
        self._check_for_alerts(metrics)
        
        self.logger.debug(f"Updated metrics for {api_name}: {metrics.status.value}, {response_time_ms:.1f}ms")
    
    def _determine_health_status(self, metrics: HealthMetrics) -> HealthStatus:
        """Determine health status based on metrics."""
        # Critical: Many consecutive failures
        if metrics.consecutive_failures >= self.monitoring_config['max_consecutive_failures']:
            return HealthStatus.CRITICAL
        
        # Unhealthy: Low success rate
        if metrics.success_rate < 0.8:
            return HealthStatus.UNHEALTHY
        
        # Degraded: Success rate below threshold OR high response times
        if (metrics.success_rate < self.monitoring_config['success_rate_threshold'] or
            metrics.average_response_time > self.monitoring_config['response_time_threshold']):
            return HealthStatus.DEGRADED
        
        # Healthy: Everything looks good
        if metrics.total_requests > 0:
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    def _check_for_alerts(self, metrics: HealthMetrics):
        """Check if alerts should be generated for the API."""
        if not self.monitoring_config['enable_alerts']:
            return
        
        api_name = metrics.api_name
        now = datetime.now()
        
        # Check alert cooldown
        last_alert = self.last_alert_time.get(api_name)
        if last_alert and (now - last_alert).total_seconds() < self.monitoring_config['alert_cooldown']:
            return
        
        alert = None
        
        # Critical alerts
        if metrics.status == HealthStatus.CRITICAL:
            alert = Alert(
                timestamp=now,
                api_name=api_name,
                level=AlertLevel.CRITICAL,
                message=f"API {api_name} is critical: {metrics.consecutive_failures} consecutive failures",
                metrics={'consecutive_failures': metrics.consecutive_failures, 'error': metrics.error_details}
            )
        
        # Error alerts
        elif metrics.status == HealthStatus.UNHEALTHY:
            alert = Alert(
                timestamp=now,
                api_name=api_name,
                level=AlertLevel.ERROR,
                message=f"API {api_name} is unhealthy: {metrics.success_rate:.1%} success rate",
                metrics={'success_rate': metrics.success_rate, 'error': metrics.error_details}
            )
        
        # Warning alerts
        elif metrics.status == HealthStatus.DEGRADED:
            if metrics.average_response_time > self.monitoring_config['response_time_threshold']:
                alert = Alert(
                    timestamp=now,
                    api_name=api_name,
                    level=AlertLevel.WARNING,
                    message=f"API {api_name} has high latency: {metrics.average_response_time:.0f}ms",
                    metrics={'response_time': metrics.average_response_time}
                )
            else:
                alert = Alert(
                    timestamp=now,
                    api_name=api_name,
                    level=AlertLevel.WARNING,
                    message=f"API {api_name} is degraded: {metrics.success_rate:.1%} success rate",
                    metrics={'success_rate': metrics.success_rate}
                )
        
        if alert:
            self.alerts.append(alert)
            self.last_alert_time[api_name] = now
            self.logger.warning(f"Alert generated: {alert.message}")
            
            # Keep alerts list manageable
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
    
    def _update_system_health(self):
        """Update overall system health status."""
        critical_apis = [m for m in self.health_metrics.values() if m.status == HealthStatus.CRITICAL]
        unhealthy_apis = [m for m in self.health_metrics.values() if m.status == HealthStatus.UNHEALTHY]
        
        # Log system health status
        if critical_apis:
            self.logger.error(f"System health CRITICAL: {len(critical_apis)} critical APIs")
        elif len(unhealthy_apis) >= 2:
            self.logger.warning(f"System health DEGRADED: {len(unhealthy_apis)} unhealthy APIs")
        elif unhealthy_apis:
            self.logger.info(f"System health GOOD with issues: {len(unhealthy_apis)} unhealthy API")
        else:
            self.logger.debug("System health EXCELLENT: All APIs healthy")
    
    def _cleanup_old_alerts(self):
        """Remove old alerts to prevent memory bloat."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    # Custom validators for specific APIs
    def _validate_bybit_response(self, data: Dict[str, Any]) -> bool:
        """Validate Bybit API response."""
        return data.get('retCode') == 0 and 'time' in data.get('result', {})
    
    def _validate_cryptopanic_response(self, data: Dict[str, Any]) -> bool:
        """Validate CryptoPanic API response."""
        return 'results' in data and isinstance(data['results'], list)
    
    def _validate_fear_greed_response(self, data: Dict[str, Any]) -> bool:
        """Validate Fear & Greed Index API response."""
        return 'data' in data and len(data['data']) > 0 and 'value' in data['data'][0]
    
    # Public interface methods
    def get_api_health_status(self, api_name: str) -> Optional[HealthMetrics]:
        """Get health status for a specific API."""
        return self.health_metrics.get(api_name)
    
    def get_all_health_metrics(self) -> Dict[str, HealthMetrics]:
        """Get health metrics for all APIs."""
        return self.health_metrics.copy()
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        now = datetime.now()
        
        # Count APIs by status
        status_counts = defaultdict(int)
        for metrics in self.health_metrics.values():
            status_counts[metrics.status.value] += 1
        
        # Determine overall system status
        if status_counts['critical'] > 0:
            overall_status = 'critical'
        elif status_counts['unhealthy'] >= 2:
            overall_status = 'degraded'
        elif status_counts['unhealthy'] > 0 or status_counts['degraded'] > 0:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        # Recent alerts
        recent_alerts = [alert for alert in self.alerts if (now - alert.timestamp).total_seconds() < 3600]  # Last hour
        
        return {
            'timestamp': now,
            'overall_status': overall_status,
            'api_count': len(self.health_metrics),
            'status_distribution': dict(status_counts),
            'recent_alerts_count': len(recent_alerts),
            'critical_apis': [name for name, metrics in self.health_metrics.items() if metrics.status == HealthStatus.CRITICAL],
            'monitoring_active': self.is_monitoring,
            'fallback_strategies_available': len(self.fallback_strategies)
        }
    
    def get_fallback_strategy(self, api_name: str) -> Dict[str, Any]:
        """Get fallback strategy for a specific API."""
        return self.fallback_strategies.get(api_name, {})
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get recent alerts within the specified time window."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    async def manual_health_check(self, api_name: str) -> Optional[HealthMetrics]:
        """Perform a manual health check for a specific API."""
        if api_name not in self.health_checks:
            return None
        
        health_check = self.health_checks[api_name]
        
        connector = aiohttp.TCPConnector()
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            await self._perform_health_check(session, health_check)
        
        return self.health_metrics[api_name]


# Example usage and testing
if __name__ == "__main__":
    import json
    from ..config.manager import ConfigurationManager
    
    async def main():
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize with mock config
        config_manager = ConfigurationManager()
        
        # Create and start monitor
        monitor = APIHealthMonitor(config_manager)
        
        try:
            # Start monitoring
            await monitor.start_monitoring()
            
            # Let it run for a bit
            await asyncio.sleep(30)
            
            # Get health report
            report = monitor.get_system_health_report()
            print(f"System health report: {json.dumps(report, indent=2, default=str)}")
            
            # Get detailed metrics
            metrics = monitor.get_all_health_metrics()
            for api_name, health_metrics in metrics.items():
                print(f"{api_name}: {health_metrics.status.value} - {health_metrics.response_time_ms:.1f}ms")
            
        finally:
            await monitor.stop_monitoring()
    
    # Run the example
    asyncio.run(main())