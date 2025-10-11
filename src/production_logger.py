#!/usr/bin/env python3
"""
Production Error Logging & Monitoring System
===========================================

Comprehensive logging system for DigitalOcean App Platform deployment
with structured logging, request correlation, and performance monitoring.

Features:
- Structured logging with correlation IDs
- Performance metrics collection
- Database operation monitoring
- API endpoint response time tracking
- Error aggregation and reporting
- Production deployment debugging
"""

import logging
import structlog
import json
import time
import uuid
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import asyncio
import functools
import traceback


@dataclass
class LogEntry:
    """Structured log entry for production monitoring"""
    timestamp: str
    level: str
    message: str
    correlation_id: str
    component: str
    environment: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    response_time_ms: Optional[float] = None
    error_type: Optional[str] = None
    error_details: Optional[Dict] = None
    database_operation: Optional[str] = None
    data_count: Optional[int] = None
    performance_metrics: Optional[Dict] = None


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    endpoint: str
    method: str
    response_time_ms: float
    timestamp: str
    status_code: int
    database_queries: int = 0
    database_time_ms: float = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage_mb: float = 0


class ProductionLogger:
    """Production-ready logging system with structured logging and monitoring"""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_dir: str = "/app/logs",
                 correlation_id_header: str = "X-Correlation-ID"):
        
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.correlation_id_header = correlation_id_header
        self.environment = os.getenv('TRADING_ENVIRONMENT', 'development')
        self.deployment_id = os.getenv('DO_APP_NAME', 'local')
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_metrics: List[PerformanceMetrics] = []
        self.error_counts = {}
        self.endpoint_stats = {}
        
        # Setup structured logging
        self._setup_structured_logging()
        
        # Create logger instances for different components
        self.app_logger = structlog.get_logger("app")
        self.api_logger = structlog.get_logger("api") 
        self.database_logger = structlog.get_logger("database")
        self.pipeline_logger = structlog.get_logger("pipeline")
        self.deployment_logger = structlog.get_logger("deployment")
        
        # Only log initialization in development or if there's an error
        if self.environment == 'development' or self.log_level <= logging.WARNING:
            self.deployment_logger.info(
                "Production logging system initialized",
                environment=self.environment,
                deployment_id=self.deployment_id,
                log_level=log_level,
                log_directory=str(self.log_dir)
            )
    
    def _setup_structured_logging(self):
        """Setup structured logging configuration"""
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.dev.set_exc_info,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(self.log_level),
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Setup file handlers - only log errors to reduce verbosity
        log_files = {
            'app.log': logging.ERROR,     # Changed from INFO to ERROR
            'api.log': logging.ERROR,     # Changed from INFO to ERROR  
            'database.log': logging.ERROR, # Changed from DEBUG to ERROR
            'errors.log': logging.ERROR,
            'performance.log': logging.ERROR  # Changed from INFO to ERROR
        }
        
        for filename, level in log_files.items():
            file_path = self.log_dir / filename
            handler = logging.FileHandler(file_path)
            handler.setLevel(level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
            # Add handler to root logger - respect the main app's ERROR level
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            # Don't override the root logger level set in main.py
            if root_logger.level == logging.NOTSET:
                root_logger.setLevel(self.log_level)
    
    def generate_correlation_id(self) -> str:
        """Generate unique correlation ID for request tracking"""
        return f"req_{uuid.uuid4().hex[:12]}"
    
    @contextmanager
    def request_context(self, 
                       endpoint: str, 
                       method: str, 
                       correlation_id: Optional[str] = None,
                       user_id: Optional[str] = None):
        """Context manager for request logging with performance tracking"""
        
        if not correlation_id:
            correlation_id = self.generate_correlation_id()
        
        start_time = time.time()
        
        # Set context variables
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            endpoint=endpoint,
            method=method,
            user_id=user_id,
            environment=self.environment
        )
        
        self.api_logger.info(
            "Request started",
            endpoint=endpoint,
            method=method,
            correlation_id=correlation_id
        )
        
        try:
            yield correlation_id
            
        except Exception as e:
            # Log error with full context
            self.log_error(
                f"Request failed: {str(e)}",
                error_type=type(e).__name__,
                endpoint=endpoint,
                method=method,
                correlation_id=correlation_id,
                error_details={"traceback": traceback.format_exc()}
            )
            raise
            
        finally:
            # Calculate and log performance metrics
            response_time_ms = (time.time() - start_time) * 1000
            
            self.api_logger.info(
                "Request completed",
                endpoint=endpoint,
                method=method,
                correlation_id=correlation_id,
                response_time_ms=response_time_ms
            )
            
            # Track performance metrics
            self._track_performance(endpoint, method, response_time_ms)
            
            # Clear context
            structlog.contextvars.clear_contextvars()
    
    def _track_performance(self, endpoint: str, method: str, response_time_ms: float):
        """Track API performance metrics"""
        
        # Update endpoint statistics
        key = f"{method} {endpoint}"
        if key not in self.endpoint_stats:
            self.endpoint_stats[key] = {
                'total_requests': 0,
                'total_time_ms': 0,
                'min_time_ms': float('inf'),
                'max_time_ms': 0,
                'error_count': 0
            }
        
        stats = self.endpoint_stats[key]
        stats['total_requests'] += 1
        stats['total_time_ms'] += response_time_ms
        stats['min_time_ms'] = min(stats['min_time_ms'], response_time_ms)
        stats['max_time_ms'] = max(stats['max_time_ms'], response_time_ms)
        
        # Log performance warning if response time is high
        if response_time_ms > 5000:  # 5 second threshold
            self.api_logger.warning(
                "Slow API response detected",
                endpoint=endpoint,
                method=method,
                response_time_ms=response_time_ms
            )
    
    def log_database_operation(self, 
                              operation: str, 
                              table: str, 
                              record_count: int, 
                              execution_time_ms: float,
                              correlation_id: Optional[str] = None):
        """Log database operations with performance tracking"""
        
        self.database_logger.info(
            "Database operation completed",
            operation=operation,
            table=table,
            record_count=record_count,
            execution_time_ms=execution_time_ms,
            correlation_id=correlation_id
        )
        
        # Log warning for slow database operations
        if execution_time_ms > 1000:  # 1 second threshold
            self.database_logger.warning(
                "Slow database operation detected",
                operation=operation,
                table=table,
                execution_time_ms=execution_time_ms
            )
    
    def log_data_discovery(self, 
                          datasets_found: int, 
                          total_records: int,
                          discovery_time_ms: float,
                          correlation_id: Optional[str] = None):
        """Log data discovery operations"""
        
        self.app_logger.info(
            "Data discovery completed",
            datasets_found=datasets_found,
            total_records=total_records,
            discovery_time_ms=discovery_time_ms,
            correlation_id=correlation_id
        )
    
    def log_backtest_execution(self,
                              pair: str,
                              timeframe: str,
                              period: str,
                              execution_time_ms: float,
                              result_metrics: Dict[str, Any],
                              correlation_id: Optional[str] = None):
        """Log backtest execution with detailed metrics"""
        
        self.pipeline_logger.info(
            "Backtest execution completed",
            pair=pair,
            timeframe=timeframe,
            period=period,
            execution_time_ms=execution_time_ms,
            result_metrics=result_metrics,
            correlation_id=correlation_id
        )
    
    def log_error(self, 
                  message: str, 
                  error_type: Optional[str] = None,
                  endpoint: Optional[str] = None,
                  method: Optional[str] = None,
                  correlation_id: Optional[str] = None,
                  error_details: Optional[Dict] = None):
        """Log errors with full context and tracking"""
        
        # Track error counts
        error_key = error_type or "UnknownError"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Create structured error entry
        error_entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level="ERROR",
            message=message,
            correlation_id=correlation_id or self.generate_correlation_id(),
            component="trading_platform",
            environment=self.environment,
            endpoint=endpoint,
            method=method,
            error_type=error_type,
            error_details=error_details
        )
        
        # Log to structured logger
        self.app_logger.error(
            message,
            error_type=error_type,
            endpoint=endpoint,
            method=method,
            correlation_id=correlation_id,
            error_details=error_details
        )
        
        # Write to error log file
        error_log_path = self.log_dir / "errors.log"
        with open(error_log_path, 'a') as f:
            f.write(json.dumps(asdict(error_entry)) + '\n')
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring dashboard"""
        
        summary = {
            'total_endpoints': len(self.endpoint_stats),
            'total_errors': sum(self.error_counts.values()),
            'error_breakdown': dict(self.error_counts),
            'endpoint_performance': {}
        }
        
        for endpoint, stats in self.endpoint_stats.items():
            avg_time = stats['total_time_ms'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
            summary['endpoint_performance'][endpoint] = {
                'requests': stats['total_requests'],
                'avg_response_time_ms': round(avg_time, 2),
                'min_response_time_ms': round(stats['min_time_ms'], 2),
                'max_response_time_ms': round(stats['max_time_ms'], 2),
                'error_rate': stats['error_count'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
            }
        
        return summary
    
    def log_deployment_status(self, 
                            status: str, 
                            component: str, 
                            details: Dict[str, Any]):
        """Log deployment status for DigitalOcean monitoring"""
        
        self.deployment_logger.info(
            f"Deployment {status}",
            component=component,
            status=status,
            details=details,
            deployment_id=self.deployment_id,
            environment=self.environment
        )


# Global production logger instance - ERROR level only for production
production_logger = ProductionLogger(
    log_level=os.getenv('LOG_LEVEL', 'ERROR'),  # Changed to ERROR for less verbose output
    log_dir=os.getenv('LOG_DIR', '/app/logs')
)


def log_endpoint_performance(func):
    """Decorator for automatic endpoint performance logging"""
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        endpoint = getattr(func, '__name__', 'unknown')
        method = 'POST' if 'post' in endpoint.lower() else 'GET'
        
        with production_logger.request_context(endpoint, method) as correlation_id:
            return await func(*args, **kwargs)
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        endpoint = getattr(func, '__name__', 'unknown')
        method = 'POST' if 'post' in endpoint.lower() else 'GET'
        
        with production_logger.request_context(endpoint, method) as correlation_id:
            return func(*args, **kwargs)
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def log_database_operation_decorator(operation_type: str, table_name: str):
    """Decorator for automatic database operation logging"""
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                # Try to get record count from result
                record_count = 0
                if isinstance(result, list):
                    record_count = len(result)
                elif isinstance(result, dict) and 'count' in result:
                    record_count = result['count']
                
                production_logger.log_database_operation(
                    operation=operation_type,
                    table=table_name,
                    record_count=record_count,
                    execution_time_ms=execution_time
                )
                
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                production_logger.log_error(
                    f"Database operation failed: {str(e)}",
                    error_type=type(e).__name__,
                    error_details={
                        'operation': operation_type,
                        'table': table_name,
                        'execution_time_ms': execution_time
                    }
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                # Try to get record count from result
                record_count = 0
                if isinstance(result, list):
                    record_count = len(result)
                elif isinstance(result, dict) and 'count' in result:
                    record_count = result['count']
                
                production_logger.log_database_operation(
                    operation=operation_type,
                    table=table_name,
                    record_count=record_count,
                    execution_time_ms=execution_time
                )
                
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                production_logger.log_error(
                    f"Database operation failed: {str(e)}",
                    error_type=type(e).__name__,
                    error_details={
                        'operation': operation_type,
                        'table': table_name,
                        'execution_time_ms': execution_time
                    }
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


if __name__ == "__main__":
    # Test the production logging system
    
    # Test basic logging
    production_logger.app_logger.info("Production logging system test")
    
    # Test request context
    with production_logger.request_context("/api/test", "GET") as correlation_id:
        production_logger.app_logger.info("Test request in context")
        
        # Test database logging
        production_logger.log_database_operation(
            operation="SELECT",
            table="historical_data",
            record_count=1000,
            execution_time_ms=150.5,
            correlation_id=correlation_id
        )
        
        # Test error logging
        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            production_logger.log_error(
                "Test error occurred",
                error_type=type(e).__name__,
                endpoint="/api/test",
                method="GET",
                correlation_id=correlation_id,
                error_details={"test_param": "test_value"}
            )
    
    # Test performance summary
    summary = production_logger.get_performance_summary()
    print("Performance Summary:", json.dumps(summary, indent=2))
    
    print("✅ Production logging system test completed")