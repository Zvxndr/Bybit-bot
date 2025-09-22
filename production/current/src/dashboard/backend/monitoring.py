"""
Performance Monitoring System for Dashboard Backend
Real-time system health and performance tracking
"""

import asyncio
import psutil
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str
    metadata: Dict[str, Any] = None

@dataclass
class SystemHealth:
    """System health snapshot"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    uptime: float
    status: str
    alerts: List[str] = None

class PerformanceMonitor:
    """Advanced performance monitoring system"""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.monitoring_active = False
        
        # Performance metrics storage
        self.metrics_history: List[PerformanceMetric] = []
        self.health_history: List[SystemHealth] = []
        
        # Alert thresholds
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 85.0  # %
        self.disk_threshold = 90.0  # %
        self.response_time_threshold = 1000.0  # ms
        
        # Performance counters
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.peak_memory_usage = 0.0
        self.peak_cpu_usage = 0.0
        
        # Component response times
        self.component_response_times: Dict[str, List[float]] = {
            "database": [],
            "websocket": [],
            "ml_integration": [],
            "api_endpoints": []
        }
        
        # Active alerts
        self.active_alerts: List[Dict[str, Any]] = []
    
    async def start(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        logger.info("📊 Performance monitoring started")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_system_health())
        asyncio.create_task(self._monitor_performance_metrics())
        asyncio.create_task(self._cleanup_old_data())
    
    async def stop(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("⏹️ Performance monitoring stopped")
    
    async def _monitor_system_health(self):
        """Monitor system health continuously"""
        while self.monitoring_active:
            try:
                health = await self._collect_system_health()
                self.health_history.append(health)
                
                # Check for alerts
                await self._check_health_alerts(health)
                
                # Keep only last 1000 health records
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-1000:]
                
                await asyncio.sleep(5)  # Every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ System health monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_performance_metrics(self):
        """Monitor performance metrics continuously"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_performance_metrics()
                self.metrics_history.extend(metrics)
                
                # Keep only last 5000 metrics
                if len(self.metrics_history) > 5000:
                    self.metrics_history = self.metrics_history[-5000:]
                
                await asyncio.sleep(2)  # Every 2 seconds
                
            except Exception as e:
                logger.error(f"❌ Performance metrics monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_health(self) -> SystemHealth:
        """Collect current system health snapshot"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            self.peak_cpu_usage = max(self.peak_cpu_usage, cpu_usage)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            self.peak_memory_usage = max(self.peak_memory_usage, memory_usage)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": float(network.bytes_sent),
                "bytes_recv": float(network.bytes_recv),
                "packets_sent": float(network.packets_sent),
                "packets_recv": float(network.packets_recv)
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Uptime
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Determine status
            status = "excellent"
            if cpu_usage > self.cpu_threshold or memory_usage > self.memory_threshold:
                status = "warning"
            if cpu_usage > 95 or memory_usage > 95 or disk_usage > self.disk_threshold:
                status = "critical"
            
            return SystemHealth(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                uptime=uptime,
                status=status,
                alerts=[]
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to collect system health: {e}")
            return SystemHealth(
                timestamp=datetime.utcnow(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                process_count=0,
                uptime=0.0,
                status="unknown",
                alerts=["Health collection failed"]
            )
    
    async def _collect_performance_metrics(self) -> List[PerformanceMetric]:
        """Collect performance metrics"""
        metrics = []
        timestamp = datetime.utcnow()
        
        try:
            # Request metrics
            if self.request_count > 0:
                avg_response_time = self.total_response_time / self.request_count
                metrics.append(PerformanceMetric(
                    name="avg_response_time",
                    value=avg_response_time,
                    unit="ms",
                    timestamp=timestamp,
                    category="api_performance"
                ))
            
            # Error rate
            if self.request_count > 0:
                error_rate = (self.error_count / self.request_count) * 100
                metrics.append(PerformanceMetric(
                    name="error_rate",
                    value=error_rate,
                    unit="%",
                    timestamp=timestamp,
                    category="api_performance"
                ))
            
            # Component response times
            for component, times in self.component_response_times.items():
                if times:
                    avg_time = sum(times) / len(times)
                    metrics.append(PerformanceMetric(
                        name=f"{component}_response_time",
                        value=avg_time,
                        unit="ms",
                        timestamp=timestamp,
                        category="component_performance"
                    ))
                    
                    # Clear old response times
                    self.component_response_times[component] = times[-100:]
            
            # System metrics
            system_health = self.health_history[-1] if self.health_history else None
            if system_health:
                metrics.extend([
                    PerformanceMetric(
                        name="cpu_usage",
                        value=system_health.cpu_usage,
                        unit="%",
                        timestamp=timestamp,
                        category="system"
                    ),
                    PerformanceMetric(
                        name="memory_usage",
                        value=system_health.memory_usage,
                        unit="%",
                        timestamp=timestamp,
                        category="system"
                    ),
                    PerformanceMetric(
                        name="disk_usage",
                        value=system_health.disk_usage,
                        unit="%",
                        timestamp=timestamp,
                        category="system"
                    )
                ])
            
            # Peak metrics
            metrics.extend([
                PerformanceMetric(
                    name="peak_cpu_usage",
                    value=self.peak_cpu_usage,
                    unit="%",
                    timestamp=timestamp,
                    category="peaks"
                ),
                PerformanceMetric(
                    name="peak_memory_usage",
                    value=self.peak_memory_usage,
                    unit="%",
                    timestamp=timestamp,
                    category="peaks"
                )
            ])
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to collect performance metrics: {e}")
            return []
    
    async def _check_health_alerts(self, health: SystemHealth):
        """Check for health alerts and manage active alerts"""
        new_alerts = []
        
        # CPU alert
        if health.cpu_usage > self.cpu_threshold:
            alert = {
                "type": "high_cpu_usage",
                "severity": "critical" if health.cpu_usage > 95 else "warning",
                "message": f"CPU usage at {health.cpu_usage:.1f}%",
                "timestamp": health.timestamp,
                "value": health.cpu_usage,
                "threshold": self.cpu_threshold
            }
            new_alerts.append(alert)
        
        # Memory alert
        if health.memory_usage > self.memory_threshold:
            alert = {
                "type": "high_memory_usage",
                "severity": "critical" if health.memory_usage > 95 else "warning",
                "message": f"Memory usage at {health.memory_usage:.1f}%",
                "timestamp": health.timestamp,
                "value": health.memory_usage,
                "threshold": self.memory_threshold
            }
            new_alerts.append(alert)
        
        # Disk alert
        if health.disk_usage > self.disk_threshold:
            alert = {
                "type": "high_disk_usage",
                "severity": "critical",
                "message": f"Disk usage at {health.disk_usage:.1f}%",
                "timestamp": health.timestamp,
                "value": health.disk_usage,
                "threshold": self.disk_threshold
            }
            new_alerts.append(alert)
        
        # Update active alerts
        self.active_alerts = new_alerts
        
        # Log new alerts
        for alert in new_alerts:
            if alert["severity"] == "critical":
                logger.error(f"🚨 CRITICAL ALERT: {alert['message']}")
            else:
                logger.warning(f"⚠️ WARNING: {alert['message']}")
    
    async def _cleanup_old_data(self):
        """Cleanup old monitoring data"""
        while self.monitoring_active:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # Clean old metrics
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                # Clean old health data
                self.health_history = [
                    h for h in self.health_history 
                    if h.timestamp > cutoff_time
                ]
                
                logger.debug("🧹 Cleaned up old monitoring data")
                
                # Run cleanup every hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"❌ Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    # Public API methods
    def record_request(self, response_time: float, success: bool = True):
        """Record API request metrics"""
        self.request_count += 1
        self.total_response_time += response_time
        
        if not success:
            self.error_count += 1
    
    def record_component_response_time(self, component: str, response_time: float):
        """Record component response time"""
        if component in self.component_response_times:
            self.component_response_times[component].append(response_time)
    
    async def get_health_snapshot(self) -> Dict[str, Any]:
        """Get current health snapshot"""
        if not self.health_history:
            return {"status": "no_data", "message": "No health data available"}
        
        latest_health = self.health_history[-1]
        
        return {
            "timestamp": latest_health.timestamp.isoformat(),
            "status": latest_health.status,
            "cpu_usage": latest_health.cpu_usage,
            "memory_usage": latest_health.memory_usage,
            "disk_usage": latest_health.disk_usage,
            "process_count": latest_health.process_count,
            "uptime": latest_health.uptime,
            "uptime_human": str(timedelta(seconds=int(latest_health.uptime))),
            "network_io": latest_health.network_io,
            "active_alerts": self.active_alerts,
            "peak_cpu": self.peak_cpu_usage,
            "peak_memory": self.peak_memory_usage
        }
    
    async def get_performance_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance metrics for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter metrics by time
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {"message": "No metrics data available"}
        
        # Group metrics by category
        metrics_by_category = {}
        for metric in recent_metrics:
            if metric.category not in metrics_by_category:
                metrics_by_category[metric.category] = []
            
            metrics_by_category[metric.category].append({
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat()
            })
        
        # Calculate summary statistics
        summary = {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": (self.error_count / self.request_count * 100) if self.request_count > 0 else 0,
            "avg_response_time": (self.total_response_time / self.request_count) if self.request_count > 0 else 0,
            "uptime": (datetime.utcnow() - self.start_time).total_seconds(),
            "monitoring_active": self.monitoring_active
        }
        
        return {
            "summary": summary,
            "metrics_by_category": metrics_by_category,
            "active_alerts": self.active_alerts,
            "time_period_hours": hours,
            "metrics_count": len(recent_metrics)
        }
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    async def get_component_health(self) -> Dict[str, Any]:
        """Get health status of individual components"""
        health = {}
        
        # API performance
        if self.request_count > 0:
            avg_response = self.total_response_time / self.request_count
            error_rate = (self.error_count / self.request_count) * 100
            
            health["api"] = {
                "status": "healthy" if avg_response < self.response_time_threshold and error_rate < 5 else "degraded",
                "avg_response_time": avg_response,
                "error_rate": error_rate,
                "total_requests": self.request_count
            }
        
        # Component response times
        for component, times in self.component_response_times.items():
            if times:
                avg_time = sum(times) / len(times)
                health[component] = {
                    "status": "healthy" if avg_time < 500 else "slow",
                    "avg_response_time": avg_time,
                    "samples": len(times)
                }
        
        # System health
        if self.health_history:
            latest = self.health_history[-1]
            health["system"] = {
                "status": latest.status,
                "cpu_usage": latest.cpu_usage,
                "memory_usage": latest.memory_usage,
                "disk_usage": latest.disk_usage
            }
        
        return health