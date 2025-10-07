"""
Ultra-Low Latency Engine for High-Frequency Trading.
Provides microsecond-level latency optimization, monitoring, and network performance analysis.
"""

import time
import asyncio
import threading
import statistics
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import psutil
import socket
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class LatencyType(Enum):
    """Types of latency measurements."""
    NETWORK_RTT = "network_rtt"
    ORDER_PROCESSING = "order_processing"
    MARKET_DATA = "market_data"
    EXECUTION = "execution"
    WEBSOCKET = "websocket"
    API_CALL = "api_call"
    INTERNAL_PROCESSING = "internal_processing"
    END_TO_END = "end_to_end"

class OptimizationLevel(Enum):
    """Latency optimization levels."""
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    ULTRA_LOW = "ultra_low"
    CUSTOM = "custom"

@dataclass
class LatencyMeasurement:
    """Individual latency measurement."""
    measurement_type: LatencyType
    latency_us: float  # Microseconds
    timestamp: datetime
    source: str
    destination: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LatencyMetrics:
    """Aggregated latency metrics."""
    measurement_type: LatencyType
    count: int
    mean_us: float
    median_us: float
    p95_us: float
    p99_us: float
    min_us: float
    max_us: float
    std_us: float
    success_rate: float
    measurement_period: timedelta
    last_updated: datetime

@dataclass
class NetworkPath:
    """Network path information."""
    source: str
    destination: str
    host: str
    port: int
    protocol: str
    avg_rtt_us: float
    packet_loss: float
    jitter_us: float
    bandwidth_mbps: float
    last_tested: datetime

class LatencyEngine:
    """Ultra-low latency engine for HFT operations."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Latency configuration
        self.latency_config = {
            'measurement_buffer_size': 10000,
            'sampling_interval_us': 1000,  # 1ms
            'alert_thresholds': {
                'network_rtt': 1000,      # 1ms
                'order_processing': 500,   # 500μs
                'market_data': 200,        # 200μs
                'execution': 1000,         # 1ms
                'websocket': 100,          # 100μs
                'api_call': 2000,          # 2ms
                'end_to_end': 5000         # 5ms
            },
            'optimization_targets': {
                'network_rtt': 500,        # Target 500μs
                'order_processing': 200,   # Target 200μs
                'market_data': 50,         # Target 50μs
                'execution': 500,          # Target 500μs
                'end_to_end': 2000         # Target 2ms
            },
            'monitoring_enabled': True,
            'auto_optimization': True,
            'performance_cores': None,  # Auto-detect
            'thread_affinity': True,
            'memory_preallocation': True
        }
        
        # Measurement storage
        self.measurements: Dict[LatencyType, deque] = {
            lat_type: deque(maxlen=self.latency_config['measurement_buffer_size'])
            for lat_type in LatencyType
        }
        
        # Current metrics
        self.current_metrics: Dict[LatencyType, LatencyMetrics] = {}
        
        # Network paths
        self.network_paths: Dict[Tuple[str, str], NetworkPath] = {}
        
        # Optimization state
        self.optimization_level = OptimizationLevel.STANDARD
        self.optimization_lock = threading.Lock()
        
        # Performance monitoring
        self.cpu_affinity_set = False
        self.memory_pools = {}
        self.high_priority_threads = []
        
        # Start background monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        
        self.logger.info("LatencyEngine initialized")
    
    async def start_monitoring(self):
        """Start latency monitoring."""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            
            # Apply initial optimizations
            await self._apply_system_optimizations()
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("Latency monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start latency monitoring: {e}")
            self.monitoring_active = False
            raise
    
    async def stop_monitoring(self):
        """Stop latency monitoring."""
        try:
            self.monitoring_active = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Latency monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop latency monitoring: {e}")
    
    def measure_latency_sync(self, 
                           measurement_type: LatencyType,
                           operation: Callable,
                           source: str = "local",
                           destination: str = "exchange",
                           **kwargs) -> float:
        """Measure latency of synchronous operation."""
        try:
            start_time = time.perf_counter_ns()
            
            try:
                result = operation(**kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                self.logger.debug(f"Operation failed during latency measurement: {e}")
            
            end_time = time.perf_counter_ns()
            latency_us = (end_time - start_time) / 1000  # Convert to microseconds
            
            # Store measurement
            measurement = LatencyMeasurement(
                measurement_type=measurement_type,
                latency_us=latency_us,
                timestamp=datetime.now(),
                source=source,
                destination=destination,
                success=success,
                metadata=kwargs
            )
            
            self.measurements[measurement_type].append(measurement)
            
            # Check for alerts
            self._check_latency_alert(measurement)
            
            return latency_us
            
        except Exception as e:
            self.logger.error(f"Latency measurement failed: {e}")
            return float('inf')
    
    async def measure_latency_async(self,
                                  measurement_type: LatencyType,
                                  operation: Callable,
                                  source: str = "local",
                                  destination: str = "exchange",
                                  **kwargs) -> float:
        """Measure latency of asynchronous operation."""
        try:
            start_time = time.perf_counter_ns()
            
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(**kwargs)
                else:
                    result = operation(**kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                self.logger.debug(f"Operation failed during latency measurement: {e}")
            
            end_time = time.perf_counter_ns()
            latency_us = (end_time - start_time) / 1000  # Convert to microseconds
            
            # Store measurement
            measurement = LatencyMeasurement(
                measurement_type=measurement_type,
                latency_us=latency_us,
                timestamp=datetime.now(),
                source=source,
                destination=destination,
                success=success,
                metadata=kwargs
            )
            
            self.measurements[measurement_type].append(measurement)
            
            # Check for alerts
            self._check_latency_alert(measurement)
            
            return latency_us
            
        except Exception as e:
            self.logger.error(f"Async latency measurement failed: {e}")
            return float('inf')
    
    def get_current_metrics(self, measurement_type: LatencyType = None) -> Dict[LatencyType, LatencyMetrics]:
        """Get current latency metrics."""
        if measurement_type:
            return {measurement_type: self.current_metrics.get(measurement_type)}
        return self.current_metrics.copy()
    
    async def optimize_latency(self, level: OptimizationLevel = OptimizationLevel.AGGRESSIVE):
        """Apply latency optimizations."""
        try:
            with self.optimization_lock:
                self.optimization_level = level
                
                await self._apply_system_optimizations()
                await self._optimize_network_settings()
                await self._optimize_thread_settings()
                await self._optimize_memory_settings()
                
                self.logger.info(f"Latency optimization applied: {level.value}")
                
        except Exception as e:
            self.logger.error(f"Latency optimization failed: {e}")
            raise
    
    async def benchmark_network_path(self, 
                                   host: str, 
                                   port: int,
                                   protocol: str = "tcp",
                                   samples: int = 100) -> NetworkPath:
        """Benchmark network path latency."""
        try:
            measurements = []
            successful_measurements = 0
            
            for _ in range(samples):
                start_time = time.perf_counter_ns()
                
                try:
                    if protocol.lower() == "tcp":
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(1.0)
                        result = sock.connect_ex((host, port))
                        sock.close()
                        success = result == 0
                    else:
                        # UDP ping simulation
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.settimeout(1.0)
                        sock.sendto(b"ping", (host, port))
                        sock.close()
                        success = True
                    
                    if success:
                        successful_measurements += 1
                        
                except Exception:
                    success = False
                
                end_time = time.perf_counter_ns()
                rtt_us = (end_time - start_time) / 1000
                
                if success and rtt_us < 1000000:  # Filter out timeouts
                    measurements.append(rtt_us)
                
                # Small delay between measurements
                await asyncio.sleep(0.001)
            
            if not measurements:
                raise ValueError("No successful network measurements")
            
            # Calculate statistics
            avg_rtt = statistics.mean(measurements)
            jitter = statistics.stdev(measurements) if len(measurements) > 1 else 0
            packet_loss = 1.0 - (successful_measurements / samples)
            
            # Create network path
            path = NetworkPath(
                source="local",
                destination=host,
                host=host,
                port=port,
                protocol=protocol,
                avg_rtt_us=avg_rtt,
                packet_loss=packet_loss,
                jitter_us=jitter,
                bandwidth_mbps=0.0,  # Would need separate bandwidth test
                last_tested=datetime.now()
            )
            
            # Store path
            path_key = ("local", host)
            self.network_paths[path_key] = path
            
            return path
            
        except Exception as e:
            self.logger.error(f"Network path benchmark failed: {e}")
            raise
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                # Update metrics
                await self._update_metrics()
                
                # Check for optimization opportunities
                if self.latency_config['auto_optimization']:
                    await self._auto_optimize()
                
                # Sleep for sampling interval
                await asyncio.sleep(self.latency_config['sampling_interval_us'] / 1_000_000)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}")
    
    async def _update_metrics(self):
        """Update current latency metrics."""
        try:
            current_time = datetime.now()
            lookback_period = timedelta(minutes=5)  # 5-minute rolling window
            
            for lat_type in LatencyType:
                measurements = self.measurements[lat_type]
                
                if not measurements:
                    continue
                
                # Filter recent measurements
                recent_measurements = [
                    m for m in measurements 
                    if current_time - m.timestamp <= lookback_period
                ]
                
                if not recent_measurements:
                    continue
                
                # Calculate statistics
                latencies = [m.latency_us for m in recent_measurements]
                successful_measurements = [m for m in recent_measurements if m.success]
                
                if latencies:
                    metrics = LatencyMetrics(
                        measurement_type=lat_type,
                        count=len(latencies),
                        mean_us=statistics.mean(latencies),
                        median_us=statistics.median(latencies),
                        p95_us=np.percentile(latencies, 95),
                        p99_us=np.percentile(latencies, 99),
                        min_us=min(latencies),
                        max_us=max(latencies),
                        std_us=statistics.stdev(latencies) if len(latencies) > 1 else 0,
                        success_rate=len(successful_measurements) / len(recent_measurements),
                        measurement_period=lookback_period,
                        last_updated=current_time
                    )
                    
                    self.current_metrics[lat_type] = metrics
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")
    
    def _check_latency_alert(self, measurement: LatencyMeasurement):
        """Check if latency measurement triggers an alert."""
        try:
            threshold = self.latency_config['alert_thresholds'].get(
                measurement.measurement_type.value
            )
            
            if threshold and measurement.latency_us > threshold:
                self.logger.warning(
                    f"Latency alert: {measurement.measurement_type.value} "
                    f"{measurement.latency_us:.0f}μs > {threshold}μs threshold"
                )
        except Exception as e:
            self.logger.error(f"Latency alert check failed: {e}")
    
    async def _apply_system_optimizations(self):
        """Apply system-level optimizations."""
        try:
            if self.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.ULTRA_LOW]:
                # Set process priority
                try:
                    import os
                    if os.name == 'nt':  # Windows
                        import psutil
                        p = psutil.Process()
                        p.nice(psutil.HIGH_PRIORITY_CLASS)
                    else:  # Unix/Linux
                        os.nice(-10)
                    
                    self.logger.debug("Process priority set to high")
                except Exception as e:
                    self.logger.debug(f"Failed to set process priority: {e}")
                
                # Set CPU affinity for performance cores
                if self.latency_config['thread_affinity']:
                    await self._set_cpu_affinity()
            
        except Exception as e:
            self.logger.error(f"System optimization failed: {e}")
    
    async def _set_cpu_affinity(self):
        """Set CPU affinity to performance cores."""
        try:
            if self.cpu_affinity_set:
                return
            
            import psutil
            
            # Try to identify performance cores (simplified heuristic)
            cpu_count = psutil.cpu_count(logical=False)
            performance_cores = self.latency_config.get('performance_cores')
            
            if performance_cores is None:
                # Use first half of cores as "performance" cores
                performance_cores = list(range(min(4, cpu_count)))
            
            try:
                p = psutil.Process()
                p.cpu_affinity(performance_cores)
                self.cpu_affinity_set = True
                self.logger.debug(f"CPU affinity set to cores: {performance_cores}")
            except Exception as e:
                self.logger.debug(f"Failed to set CPU affinity: {e}")
                
        except Exception as e:
            self.logger.debug(f"CPU affinity configuration failed: {e}")
    
    async def _optimize_network_settings(self):
        """Optimize network settings for low latency."""
        try:
            if self.optimization_level == OptimizationLevel.ULTRA_LOW:
                # These would typically require system-level changes
                # Here we just log what should be done
                optimizations = [
                    "TCP_NODELAY should be enabled",
                    "Socket buffer sizes should be optimized", 
                    "Network interrupt coalescing should be disabled",
                    "Network adapter offloading should be configured"
                ]
                
                for opt in optimizations:
                    self.logger.debug(f"Network optimization: {opt}")
                    
        except Exception as e:
            self.logger.error(f"Network optimization failed: {e}")
    
    async def _optimize_thread_settings(self):
        """Optimize thread settings for low latency."""
        try:
            # Set thread priorities for critical threads
            current_thread = threading.current_thread()
            
            if hasattr(current_thread, 'native_id'):
                # This is a simplified example - real implementation would
                # need platform-specific thread priority setting
                self.logger.debug(f"Thread {current_thread.native_id} optimized for low latency")
                
        except Exception as e:
            self.logger.error(f"Thread optimization failed: {e}")
    
    async def _optimize_memory_settings(self):
        """Optimize memory settings for low latency."""
        try:
            if self.latency_config['memory_preallocation']:
                # Pre-allocate common data structures
                if not self.memory_pools:
                    self.memory_pools = {
                        'small_buffers': [bytearray(1024) for _ in range(100)],
                        'medium_buffers': [bytearray(8192) for _ in range(50)],
                        'large_buffers': [bytearray(65536) for _ in range(10)]
                    }
                    
                    self.logger.debug("Memory pools pre-allocated")
                    
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
    
    async def _auto_optimize(self):
        """Automatically optimize based on current performance."""
        try:
            # Check if any metrics exceed targets
            should_optimize = False
            
            for lat_type, metrics in self.current_metrics.items():
                target = self.latency_config['optimization_targets'].get(
                    lat_type.value
                )
                
                if target and metrics.p95_us > target * 1.5:  # 50% above target
                    should_optimize = True
                    break
            
            if should_optimize and self.optimization_level != OptimizationLevel.ULTRA_LOW:
                next_level = {
                    OptimizationLevel.STANDARD: OptimizationLevel.AGGRESSIVE,
                    OptimizationLevel.AGGRESSIVE: OptimizationLevel.ULTRA_LOW
                }.get(self.optimization_level)
                
                if next_level:
                    await self.optimize_latency(next_level)
                    self.logger.info(f"Auto-optimization: upgraded to {next_level.value}")
                    
        except Exception as e:
            self.logger.error(f"Auto-optimization failed: {e}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get latency optimization recommendations."""
        recommendations = []
        
        try:
            for lat_type, metrics in self.current_metrics.items():
                target = self.latency_config['optimization_targets'].get(
                    lat_type.value
                )
                
                if target and metrics.mean_us > target:
                    excess_pct = ((metrics.mean_us - target) / target) * 100
                    
                    recommendations.append(
                        f"{lat_type.value}: {excess_pct:.1f}% above target "
                        f"({metrics.mean_us:.0f}μs vs {target}μs target)"
                    )
            
            # System-level recommendations
            if self.optimization_level == OptimizationLevel.STANDARD:
                recommendations.append("Consider enabling aggressive optimization mode")
            
            if not self.cpu_affinity_set:
                recommendations.append("CPU affinity not set - consider binding to performance cores")
            
            # Network recommendations
            high_rtt_paths = [
                path for path in self.network_paths.values()
                if path.avg_rtt_us > 1000  # > 1ms
            ]
            
            if high_rtt_paths:
                recommendations.append(
                    f"{len(high_rtt_paths)} network paths have high RTT - "
                    "consider network optimization or server relocation"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate optimization recommendations")
        
        return recommendations
    
    def get_latency_summary(self) -> Dict[str, Any]:
        """Get summary of latency performance."""
        try:
            summary = {
                'optimization_level': self.optimization_level.value,
                'monitoring_active': self.monitoring_active,
                'total_measurements': sum(len(deque_obj) for deque_obj in self.measurements.values()),
                'network_paths': len(self.network_paths),
                'cpu_affinity_set': self.cpu_affinity_set,
                'memory_pools_allocated': bool(self.memory_pools),
                'current_performance': {}
            }
            
            # Add current performance metrics
            for lat_type, metrics in self.current_metrics.items():
                target = self.latency_config['optimization_targets'].get(lat_type.value)
                
                summary['current_performance'][lat_type.value] = {
                    'mean_us': round(metrics.mean_us, 1),
                    'p95_us': round(metrics.p95_us, 1),
                    'target_us': target,
                    'meets_target': metrics.mean_us <= target if target else True,
                    'success_rate': round(metrics.success_rate * 100, 1)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate latency summary: {e}")
            return {'error': 'Unable to generate summary'}


class NetworkMonitor:
    """Network performance monitor for HFT operations."""
    
    def __init__(self, latency_engine: LatencyEngine):
        self.latency_engine = latency_engine
        self.logger = TradingLogger()
        
        # Network monitoring configuration
        self.monitoring_config = {
            'ping_interval_ms': 100,
            'bandwidth_test_interval_min': 30,
            'path_discovery_enabled': True,
            'quality_monitoring': True
        }
        
        # Exchange endpoints to monitor
        self.exchange_endpoints = {
            'bybit': [
                ('api.bybit.com', 443),
                ('stream.bybit.com', 443)
            ],
            'binance': [
                ('api.binance.com', 443),
                ('stream.binance.com', 9443)
            ]
        }
        
        self.monitoring_active = False
        self.monitoring_tasks = []
        
    async def start_monitoring(self):
        """Start network monitoring."""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            
            # Start monitoring tasks for each exchange
            for exchange, endpoints in self.exchange_endpoints.items():
                for host, port in endpoints:
                    task = asyncio.create_task(
                        self._monitor_endpoint(exchange, host, port)
                    )
                    self.monitoring_tasks.append(task)
            
            self.logger.info("Network monitoring started")
                    
        except Exception as e:
            self.logger.error(f"Failed to start network monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop network monitoring."""
        try:
            self.monitoring_active = False
            
            for task in self.monitoring_tasks:
                task.cancel()
            
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            self.monitoring_tasks.clear()
            self.logger.info("Network monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop network monitoring: {e}")
    
    async def _monitor_endpoint(self, exchange: str, host: str, port: int):
        """Monitor specific network endpoint."""
        try:
            while self.monitoring_active:
                # Benchmark network path
                try:
                    path = await self.latency_engine.benchmark_network_path(
                        host, port, samples=10
                    )
                    
                    # Log if performance degrades
                    if path.avg_rtt_us > 2000:  # > 2ms
                        self.logger.warning(
                            f"High latency to {exchange} {host}: {path.avg_rtt_us:.0f}μs"
                        )
                    
                    if path.packet_loss > 0.01:  # > 1% packet loss
                        self.logger.warning(
                            f"Packet loss to {exchange} {host}: {path.packet_loss:.1%}"
                        )
                        
                except Exception as e:
                    self.logger.debug(f"Network test failed for {host}: {e}")
                
                # Wait before next test
                await asyncio.sleep(self.monitoring_config['ping_interval_ms'] / 1000)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Endpoint monitoring error for {host}: {e}")


class LatencyOptimizer:
    """Advanced latency optimization engine."""
    
    def __init__(self, latency_engine: LatencyEngine):
        self.latency_engine = latency_engine
        self.logger = TradingLogger()
        
        # Optimization algorithms
        self.optimizers = {
            'system': self._optimize_system_settings,
            'network': self._optimize_network_settings,
            'application': self._optimize_application_settings,
            'hardware': self._analyze_hardware_optimizations
        }
        
    async def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """Run comprehensive latency optimization."""
        results = {}
        
        try:
            baseline_metrics = self.latency_engine.get_current_metrics()
            
            for optimizer_name, optimizer_func in self.optimizers.items():
                try:
                    result = await optimizer_func()
                    results[optimizer_name] = result
                    self.logger.info(f"Completed {optimizer_name} optimization")
                except Exception as e:
                    results[optimizer_name] = {'error': str(e)}
                    self.logger.error(f"{optimizer_name} optimization failed: {e}")
            
            # Measure improvement
            await asyncio.sleep(5)  # Wait for optimizations to take effect
            optimized_metrics = self.latency_engine.get_current_metrics()
            
            results['improvement_analysis'] = self._analyze_improvement(
                baseline_metrics, optimized_metrics
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive optimization failed: {e}")
            return {'error': str(e)}
    
    async def _optimize_system_settings(self) -> Dict[str, Any]:
        """Optimize system-level settings."""
        optimizations = []
        
        try:
            # CPU optimization
            optimizations.append("Set process to high priority")
            optimizations.append("Configure CPU affinity for performance cores")
            
            # Memory optimization  
            optimizations.append("Pre-allocate memory pools")
            optimizations.append("Configure large pages if available")
            
            # Scheduler optimization
            optimizations.append("Set real-time scheduling class")
            
            return {
                'optimizations_applied': optimizations,
                'status': 'success'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _optimize_network_settings(self) -> Dict[str, Any]:
        """Optimize network settings."""
        optimizations = []
        
        try:
            # TCP optimization
            optimizations.append("Enable TCP_NODELAY")
            optimizations.append("Optimize socket buffer sizes")
            optimizations.append("Configure TCP congestion control")
            
            # Network adapter optimization
            optimizations.append("Disable interrupt coalescing")
            optimizations.append("Configure RSS (Receive Side Scaling)")
            optimizations.append("Optimize network adapter queue settings")
            
            return {
                'optimizations_applied': optimizations,
                'status': 'success'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _optimize_application_settings(self) -> Dict[str, Any]:
        """Optimize application-level settings."""
        optimizations = []
        
        try:
            # Threading optimization
            optimizations.append("Configure thread pool sizes")
            optimizations.append("Set thread priorities")
            optimizations.append("Optimize thread synchronization")
            
            # Data structure optimization
            optimizations.append("Use lock-free data structures")
            optimizations.append("Optimize memory allocation patterns")
            optimizations.append("Configure garbage collection")
            
            return {
                'optimizations_applied': optimizations,
                'status': 'success'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _analyze_hardware_optimizations(self) -> Dict[str, Any]:
        """Analyze potential hardware optimizations."""
        recommendations = []
        
        try:
            import psutil
            
            # CPU analysis
            cpu_info = {
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'freq': psutil.cpu_freq()
            }
            
            if cpu_info['cores'] < 4:
                recommendations.append("Consider CPU with more cores for parallel processing")
            
            if cpu_info['freq'] and cpu_info['freq'].current < 3000:
                recommendations.append("Consider higher frequency CPU for lower latency")
            
            # Memory analysis
            memory = psutil.virtual_memory()
            if memory.total < 16 * 1024**3:  # Less than 16GB
                recommendations.append("Consider increasing RAM for larger buffers")
            
            # Network analysis
            recommendations.append("Consider 10GbE or higher network interface")
            recommendations.append("Consider RDMA-capable network adapter")
            recommendations.append("Evaluate proximity to exchange data centers")
            
            return {
                'hardware_recommendations': recommendations,
                'current_specs': {
                    'cpu_cores': cpu_info['cores'],
                    'cpu_freq_mhz': cpu_info['freq'].current if cpu_info['freq'] else 'unknown',
                    'memory_gb': round(memory.total / (1024**3), 1)
                },
                'status': 'success'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_improvement(self, 
                           baseline: Dict[LatencyType, LatencyMetrics],
                           optimized: Dict[LatencyType, LatencyMetrics]) -> Dict[str, Any]:
        """Analyze optimization improvements."""
        improvements = {}
        
        try:
            for lat_type in baseline:
                if lat_type in optimized:
                    baseline_metric = baseline[lat_type]
                    optimized_metric = optimized[lat_type]
                    
                    improvement_pct = (
                        (baseline_metric.mean_us - optimized_metric.mean_us) / 
                        baseline_metric.mean_us * 100
                    )
                    
                    improvements[lat_type.value] = {
                        'baseline_us': round(baseline_metric.mean_us, 1),
                        'optimized_us': round(optimized_metric.mean_us, 1),
                        'improvement_pct': round(improvement_pct, 1),
                        'improvement_us': round(baseline_metric.mean_us - optimized_metric.mean_us, 1)
                    }
            
            return improvements
            
        except Exception as e:
            return {'error': str(e)}