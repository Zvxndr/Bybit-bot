#!/usr/bin/env python3
"""
Enhanced Health Check System for Production Deployment
===================================================

Comprehensive health check system that validates:
- Database connectivity and data availability  
- API endpoint functionality
- Persistent volume status
- Data discovery pipeline health
- Testnet API connectivity
- System resource utilization
"""

import os
import time
import psutil
import sqlite3
import asyncio
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Import production logger
try:
    from src.production_logger import production_logger
except ImportError:
    import logging
    production_logger = logging.getLogger(__name__)


class ProductionHealthCheck:
    """Comprehensive health check system for production deployment"""
    
    def __init__(self):
        self.environment = os.getenv('TRADING_ENVIRONMENT', 'development')
        self.deployment_id = os.getenv('DO_APP_NAME', 'local')
        
        # Database paths to check
        self.database_paths = [
            "/app/data/trading_bot.db",
            "data/trading_bot.db"
        ]
        
        # Critical directories
        self.critical_directories = [
            "/app/data",
            "/app/logs", 
            "/app/config"
        ]
        
        # API credentials for testnet validation
        self.testnet_api_key = os.getenv('BYBIT_TESTNET_API_KEY')
        self.testnet_api_secret = os.getenv('BYBIT_TESTNET_API_SECRET')
        
        production_logger.deployment_logger.info(
            "Production health check system initialized",
            environment=self.environment,
            deployment_id=self.deployment_id
        )
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check of all system components"""
        
        start_time = time.time()
        
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "environment": self.environment,
            "deployment_id": self.deployment_id,
            "overall_status": "UNKNOWN",
            "components": {},
            "metrics": {},
            "data_availability": {},
            "warnings": [],
            "errors": [],
            "check_duration_ms": 0
        }
        
        try:
            # 1. Check system resources
            health_status["components"]["system"] = await self._check_system_resources()
            
            # 2. Check persistent volumes
            health_status["components"]["storage"] = self._check_persistent_storage()
            
            # 3. Check database connectivity and data
            health_status["components"]["database"] = await self._check_database_health()
            
            # 4. Check data discovery pipeline
            health_status["components"]["data_pipeline"] = await self._check_data_discovery_pipeline()
            
            # 5. Check testnet API connectivity  
            health_status["components"]["testnet_api"] = await self._check_testnet_connectivity()
            
            # 6. Check internal API endpoints
            health_status["components"]["internal_api"] = await self._check_internal_apis()
            
            # 7. Gather data availability metrics
            health_status["data_availability"] = await self._get_data_availability_metrics()
            
            # 8. Calculate overall health status
            health_status["overall_status"] = self._calculate_overall_health(health_status["components"])
            
            # 9. Generate warnings and recommendations
            health_status["warnings"], health_status["errors"] = self._generate_health_warnings(health_status)
            
        except Exception as e:
            health_status["overall_status"] = "ERROR"
            health_status["errors"].append(f"Health check failed: {str(e)}")
            
            production_logger.log_error(
                "Health check system error",
                error_type=type(e).__name__,
                error_details={"message": str(e)}
            )
        
        # Calculate total check duration
        health_status["check_duration_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Log health check results
        production_logger.deployment_logger.info(
            "Health check completed",
            overall_status=health_status["overall_status"],
            duration_ms=health_status["check_duration_ms"],
            warnings_count=len(health_status["warnings"]),
            errors_count=len(health_status["errors"])
        )
        
        return health_status
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization"""
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check for resource constraints
            status = "HEALTHY"
            if cpu_percent > 80:
                status = "WARNING"
            if memory.percent > 85:
                status = "WARNING" 
            if disk.percent > 90:
                status = "CRITICAL"
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / 1024**3, 2),
                "memory_total_gb": round(memory.total / 1024**3, 2),
                "disk_percent": disk.percent,
                "disk_used_gb": round(disk.used / 1024**3, 2),
                "disk_total_gb": round(disk.total / 1024**3, 2)
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    def _check_persistent_storage(self) -> Dict[str, Any]:
        """Check persistent volume mounts and accessibility"""
        
        storage_status = {
            "status": "HEALTHY",
            "volumes": {},
            "total_writable_volumes": 0
        }
        
        for directory in self.critical_directories:
            volume_info = {
                "path": directory,
                "exists": False,
                "writable": False,
                "size_bytes": 0
            }
            
            try:
                path_obj = Path(directory)
                
                # Check existence
                volume_info["exists"] = path_obj.exists()
                
                # Check writability
                if volume_info["exists"]:
                    try:
                        test_file = path_obj / f"health_check_{int(time.time())}.tmp"
                        test_file.write_text("health check")
                        test_file.unlink()
                        volume_info["writable"] = True
                        storage_status["total_writable_volumes"] += 1
                    except:
                        volume_info["writable"] = False
                        storage_status["status"] = "WARNING"
                    
                    # Get directory size if possible
                    try:
                        volume_info["size_bytes"] = sum(
                            f.stat().st_size for f in path_obj.rglob('*') if f.is_file()
                        )
                    except:
                        volume_info["size_bytes"] = 0
                else:
                    storage_status["status"] = "CRITICAL"
                
            except Exception as e:
                volume_info["error"] = str(e)
                storage_status["status"] = "ERROR"
            
            storage_status["volumes"][directory] = volume_info
        
        return storage_status
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and data health"""
        
        database_status = {
            "status": "UNKNOWN",
            "connection": False,
            "tables": [],
            "data_counts": {},
            "latest_data": None,
            "database_path": None
        }
        
        # Find accessible database
        for db_path in self.database_paths:
            if Path(db_path).exists():
                database_status["database_path"] = db_path
                break
        
        if not database_status["database_path"]:
            database_status["status"] = "CRITICAL"
            database_status["error"] = "No database file found"
            return database_status
        
        try:
            with sqlite3.connect(database_status["database_path"], timeout=10) as conn:
                cursor = conn.cursor()
                
                # Test basic connectivity
                cursor.execute("SELECT 1")
                database_status["connection"] = True
                
                # Get table list
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                database_status["tables"] = [row[0] for row in cursor.fetchall()]
                
                # Get data counts for important tables
                important_tables = ['historical_data', 'backtest_results', 'strategies']
                for table in important_tables:
                    if table in database_status["tables"]:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        database_status["data_counts"][table] = cursor.fetchone()[0]
                
                # Get latest historical data timestamp
                if 'historical_data' in database_status["tables"]:
                    cursor.execute("SELECT MAX(timestamp) FROM historical_data")
                    latest_timestamp = cursor.fetchone()[0]
                    if latest_timestamp:
                        database_status["latest_data"] = latest_timestamp
                
                # Determine status based on data availability
                historical_count = database_status["data_counts"].get('historical_data', 0)
                if historical_count >= 1000:
                    database_status["status"] = "HEALTHY"
                elif historical_count > 0:
                    database_status["status"] = "WARNING"
                else:
                    database_status["status"] = "CRITICAL"
                
        except Exception as e:
            database_status["status"] = "ERROR"
            database_status["error"] = str(e)
            
            production_logger.log_error(
                "Database health check failed",
                error_type=type(e).__name__,
                error_details={"database_path": database_status["database_path"]}
            )
        
        return database_status
    
    async def _check_data_discovery_pipeline(self) -> Dict[str, Any]:
        """Check data discovery API pipeline functionality"""
        
        pipeline_status = {
            "status": "UNKNOWN",
            "endpoints_tested": 0,
            "endpoints_healthy": 0,
            "discovery_data_count": 0,
            "response_times": {}
        }
        
        # Test data discovery endpoint
        endpoints_to_test = [
            "http://localhost:8080/api/historical-data/discover",
            "http://127.0.0.1:8080/api/historical-data/discover"
        ]
        
        for base_url in endpoints_to_test:
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(base_url) as response:
                        response_time_ms = (time.time() - start_time) * 1000
                        pipeline_status["response_times"][base_url] = response_time_ms
                        pipeline_status["endpoints_tested"] += 1
                        
                        if response.status == 200:
                            pipeline_status["endpoints_healthy"] += 1
                            
                            try:
                                data = await response.json()
                                if data.get('success'):
                                    datasets = data.get('datasets', [])
                                    pipeline_status["discovery_data_count"] = len(datasets)
                                    
                                    if len(datasets) > 0:
                                        pipeline_status["status"] = "HEALTHY"
                                    else:
                                        pipeline_status["status"] = "WARNING"
                                    break
                                else:
                                    pipeline_status["status"] = "WARNING"
                            except:
                                pipeline_status["status"] = "ERROR"
                        
            except Exception as e:
                pipeline_status["endpoints_tested"] += 1
                pipeline_status["last_error"] = str(e)
        
        # Set final status if not already determined
        if pipeline_status["status"] == "UNKNOWN":
            if pipeline_status["endpoints_healthy"] > 0:
                pipeline_status["status"] = "DEGRADED"
            else:
                pipeline_status["status"] = "CRITICAL"
        
        return pipeline_status
    
    async def _check_testnet_connectivity(self) -> Dict[str, Any]:
        """Check Bybit testnet API connectivity"""
        
        testnet_status = {
            "status": "UNKNOWN",
            "credentials_configured": False,
            "api_accessible": False,
            "response_time_ms": 0
        }
        
        # Check if credentials are configured
        testnet_status["credentials_configured"] = bool(
            self.testnet_api_key and self.testnet_api_secret and 
            len(self.testnet_api_key) > 10 and len(self.testnet_api_secret) > 10
        )
        
        if not testnet_status["credentials_configured"]:
            testnet_status["status"] = "WARNING"
            testnet_status["message"] = "Testnet credentials not configured"
            return testnet_status
        
        # Test API connectivity (without authentication for basic check)
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                # Test public endpoint (server time)
                async with session.get("https://api-testnet.bybit.com/v5/market/time") as response:
                    testnet_status["response_time_ms"] = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        testnet_status["api_accessible"] = True
                        testnet_status["status"] = "HEALTHY"
                    else:
                        testnet_status["status"] = "WARNING"
                        testnet_status["message"] = f"API returned status {response.status}"
        
        except Exception as e:
            testnet_status["status"] = "ERROR"
            testnet_status["error"] = str(e)
            testnet_status["response_time_ms"] = (time.time() - start_time) * 1000
        
        return testnet_status
    
    async def _check_internal_apis(self) -> Dict[str, Any]:
        """Check internal API endpoints"""
        
        api_status = {
            "status": "UNKNOWN",
            "endpoints": {}
        }
        
        # Critical internal endpoints
        endpoints = [
            "/api/backtest/history",
            "/api/historical-data/status"
        ]
        
        healthy_count = 0
        
        for endpoint in endpoints:
            endpoint_status = {
                "accessible": False,
                "response_time_ms": 0,
                "status_code": 0
            }
            
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(f"http://localhost:8080{endpoint}") as response:
                        endpoint_status["response_time_ms"] = (time.time() - start_time) * 1000
                        endpoint_status["status_code"] = response.status
                        endpoint_status["accessible"] = response.status == 200
                        
                        if endpoint_status["accessible"]:
                            healthy_count += 1
            
            except Exception as e:
                endpoint_status["error"] = str(e)
            
            api_status["endpoints"][endpoint] = endpoint_status
        
        # Determine overall API status
        if healthy_count == len(endpoints):
            api_status["status"] = "HEALTHY"
        elif healthy_count > 0:
            api_status["status"] = "DEGRADED"
        else:
            api_status["status"] = "CRITICAL"
        
        return api_status
    
    async def _get_data_availability_metrics(self) -> Dict[str, Any]:
        """Get detailed data availability metrics"""
        
        metrics = {
            "historical_data_symbols": [],
            "timeframes_available": [],
            "data_date_range": None,
            "total_data_points": 0,
            "data_completeness_score": 0.0
        }
        
        # Find database
        db_path = None
        for path in self.database_paths:
            if Path(path).exists():
                db_path = path
                break
        
        if not db_path:
            return metrics
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Get unique symbols
                cursor.execute("SELECT DISTINCT symbol FROM historical_data")
                metrics["historical_data_symbols"] = [row[0] for row in cursor.fetchall()]
                
                # Get unique timeframes
                cursor.execute("SELECT DISTINCT timeframe FROM historical_data")
                metrics["timeframes_available"] = [row[0] for row in cursor.fetchall()]
                
                # Get date range
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM historical_data")
                result = cursor.fetchone()
                if result and result[0] and result[1]:
                    try:
                        min_ts, max_ts = result
                        if isinstance(min_ts, (int, float)) and min_ts > 1e10:
                            min_dt = datetime.fromtimestamp(min_ts / 1000)
                            max_dt = datetime.fromtimestamp(max_ts / 1000)
                        else:
                            min_dt = datetime.fromtimestamp(min_ts)
                            max_dt = datetime.fromtimestamp(max_ts)
                        
                        metrics["data_date_range"] = {
                            "start": min_dt.isoformat(),
                            "end": max_dt.isoformat(),
                            "days": (max_dt - min_dt).days
                        }
                    except:
                        pass
                
                # Get total data points
                cursor.execute("SELECT COUNT(*) FROM historical_data")
                metrics["total_data_points"] = cursor.fetchone()[0]
                
                # Calculate completeness score
                expected_symbols = 4  # BTCUSDT, ETHUSDT, etc.
                expected_timeframes = 6  # 1m, 5m, 15m, 1h, 4h, 1d
                
                completeness_score = 0.0
                if len(metrics["historical_data_symbols"]) >= expected_symbols:
                    completeness_score += 0.5
                if len(metrics["timeframes_available"]) >= expected_timeframes:
                    completeness_score += 0.3
                if metrics["total_data_points"] >= 10000:
                    completeness_score += 0.2
                
                metrics["data_completeness_score"] = completeness_score
        
        except Exception as e:
            production_logger.log_error(
                "Data availability metrics collection failed",
                error_type=type(e).__name__
            )
        
        return metrics
    
    def _calculate_overall_health(self, components: Dict[str, Any]) -> str:
        """Calculate overall system health status"""
        
        component_statuses = []
        for component_name, component_data in components.items():
            if isinstance(component_data, dict) and 'status' in component_data:
                component_statuses.append(component_data['status'])
        
        # Count status types
        critical_count = component_statuses.count('CRITICAL')
        error_count = component_statuses.count('ERROR')
        warning_count = component_statuses.count('WARNING')
        healthy_count = component_statuses.count('HEALTHY')
        
        # Determine overall status
        if critical_count > 0 or error_count > 2:
            return "CRITICAL"
        elif error_count > 0 or warning_count > 2:
            return "DEGRADED"
        elif warning_count > 0:
            return "WARNING"
        elif healthy_count >= len(component_statuses) * 0.8:
            return "HEALTHY"
        else:
            return "DEGRADED"
    
    def _generate_health_warnings(self, health_status: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Generate warnings and errors based on health status"""
        
        warnings = []
        errors = []
        
        # Check component statuses
        for component_name, component_data in health_status["components"].items():
            if isinstance(component_data, dict):
                status = component_data.get('status', 'UNKNOWN')
                
                if status == 'CRITICAL':
                    errors.append(f"{component_name.upper()}: Critical failure - {component_data.get('error', 'Unknown error')}")
                elif status == 'ERROR':
                    errors.append(f"{component_name.upper()}: Error - {component_data.get('error', 'Unknown error')}")
                elif status == 'WARNING':
                    warnings.append(f"{component_name.upper()}: Warning - Performance or functionality degraded")
                elif status == 'DEGRADED':
                    warnings.append(f"{component_name.upper()}: Degraded - Some functionality unavailable")
        
        # Check specific conditions
        data_availability = health_status.get("data_availability", {})
        total_data_points = data_availability.get("total_data_points", 0)
        
        if total_data_points == 0:
            errors.append("DATA: No historical data available - backtesting will not work")
        elif total_data_points < 1000:
            warnings.append("DATA: Limited historical data - may affect backtesting accuracy")
        
        # Check data completeness
        completeness_score = data_availability.get("data_completeness_score", 0.0)
        if completeness_score < 0.5:
            warnings.append(f"DATA: Low data completeness score ({completeness_score:.1f}/1.0)")
        
        return warnings, errors


# Global health check instance
health_checker = ProductionHealthCheck()


async def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status - for use in FastAPI endpoint"""
    return await health_checker.comprehensive_health_check()


async def main():
    """Run standalone health check"""
    
    print("🏥 Running comprehensive production health check...")
    
    health_status = await health_checker.comprehensive_health_check()
    
    print(f"\n📊 HEALTH CHECK RESULTS")
    print(f"{'='*50}")
    print(f"Overall Status: {health_status['overall_status']}")
    print(f"Check Duration: {health_status['check_duration_ms']} ms")
    print(f"Environment: {health_status['environment']}")
    print(f"Deployment ID: {health_status['deployment_id']}")
    
    print(f"\n🔧 COMPONENT STATUS:")
    for component, data in health_status['components'].items():
        status = data.get('status', 'UNKNOWN') if isinstance(data, dict) else 'UNKNOWN'
        status_icon = {
            'HEALTHY': '✅', 
            'WARNING': '⚠️', 
            'DEGRADED': '🟡', 
            'CRITICAL': '❌', 
            'ERROR': '💥'
        }.get(status, '❓')
        print(f"  {component}: {status_icon} {status}")
    
    if health_status['warnings']:
        print(f"\n⚠️ WARNINGS:")
        for warning in health_status['warnings']:
            print(f"  - {warning}")
    
    if health_status['errors']:
        print(f"\n❌ ERRORS:")
        for error in health_status['errors']:
            print(f"  - {error}")
    
    print(f"\n📈 DATA AVAILABILITY:")
    data_metrics = health_status['data_availability']
    print(f"  Symbols: {len(data_metrics.get('historical_data_symbols', []))}")
    print(f"  Timeframes: {len(data_metrics.get('timeframes_available', []))}")
    print(f"  Total Data Points: {data_metrics.get('total_data_points', 0):,}")
    print(f"  Completeness Score: {data_metrics.get('data_completeness_score', 0.0):.1f}/1.0")
    
    # Return exit code based on health
    return 0 if health_status['overall_status'] in ['HEALTHY', 'WARNING'] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)