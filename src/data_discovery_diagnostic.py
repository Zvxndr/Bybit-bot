#!/usr/bin/env python3
"""
Data Discovery & Persistence Diagnostic Tool
==========================================

Comprehensive diagnostic tool to identify and fix data persistence issues
in the DigitalOcean App Platform deployment.

Features:
- Database connectivity validation
- Historical data discovery verification
- Persistent volume mount validation
- API endpoint testing
- Frontend-backend integration testing
- Automated repair functionality
"""

import os
import sqlite3
import json
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict

# Import our production logger
try:
    from src.production_logger import production_logger
except ImportError:
    # Fallback logging if production logger not available
    logging.basicConfig(level=logging.INFO)
    production_logger = logging.getLogger(__name__)

@dataclass 
class DatabaseStatus:
    """Database connectivity and data status"""
    path: str
    exists: bool
    accessible: bool
    tables: List[str]
    historical_data_count: int
    backtest_results_count: int
    latest_data_timestamp: Optional[str]
    data_quality_score: float


@dataclass
class APIEndpointStatus:
    """API endpoint availability and response status"""
    endpoint: str
    method: str
    accessible: bool
    response_time_ms: float
    status_code: int
    response_data: Optional[Dict]
    error_message: Optional[str]


@dataclass
class DataDiscoveryReport:
    """Comprehensive data discovery diagnostic report"""
    timestamp: str
    environment: str
    database_status: DatabaseStatus
    api_endpoints: List[APIEndpointStatus]
    persistent_volumes: Dict[str, bool]
    data_pipeline_health: str
    recommendations: List[str]
    critical_issues: List[str]
    repair_actions_taken: List[str]


class DataDiscoveryDiagnostic:
    """Comprehensive diagnostic tool for data discovery issues"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8080",
                 database_paths: List[str] = None):
        
        self.base_url = base_url
        self.database_paths = database_paths or [
            "/app/data/trading_bot.db",
            "data/trading_bot.db", 
            "src/data/speed_demon_cache/market_data.db",
            "/app/src/data/speed_demon_cache/market_data.db"
        ]
        self.environment = os.getenv('TRADING_ENVIRONMENT', 'development')
        
        # API endpoints to test
        self.critical_endpoints = [
            ("/api/historical-data/discover", "GET"),
            ("/api/historical-data/status", "GET"),
            ("/api/backtest/history", "GET"),
            ("/health", "GET")
        ]
        
        production_logger.deployment_logger.info(
            "Data discovery diagnostic tool initialized",
            base_url=base_url,
            environment=self.environment,
            database_paths=self.database_paths
        )
    
    async def run_comprehensive_diagnostic(self) -> DataDiscoveryReport:
        """Run complete diagnostic of data discovery pipeline"""
        
        production_logger.deployment_logger.info("Starting comprehensive data discovery diagnostic")
        
        # 1. Check database connectivity and data
        database_status = await self.check_database_status()
        
        # 2. Test API endpoints
        api_endpoints = await self.test_api_endpoints()
        
        # 3. Check persistent volumes
        persistent_volumes = self.check_persistent_volumes()
        
        # 4. Assess overall pipeline health
        pipeline_health = self.assess_pipeline_health(database_status, api_endpoints)
        
        # 5. Generate recommendations and repair actions
        recommendations, critical_issues = self.generate_recommendations(
            database_status, api_endpoints, persistent_volumes
        )
        
        # 6. Attempt automated repairs
        repair_actions = await self.attempt_automated_repairs(critical_issues)
        
        # Create comprehensive report
        report = DataDiscoveryReport(
            timestamp=datetime.now().isoformat(),
            environment=self.environment,
            database_status=database_status,
            api_endpoints=api_endpoints,
            persistent_volumes=persistent_volumes,
            data_pipeline_health=pipeline_health,
            recommendations=recommendations,
            critical_issues=critical_issues,
            repair_actions_taken=repair_actions
        )
        
        # Log report summary
        production_logger.deployment_logger.info(
            "Data discovery diagnostic completed",
            pipeline_health=pipeline_health,
            critical_issues_count=len(critical_issues),
            repair_actions_count=len(repair_actions)
        )
        
        return report
    
    async def check_database_status(self) -> DatabaseStatus:
        """Check database connectivity and data availability"""
        
        production_logger.database_logger.info("Checking database status")
        
        # Find the correct database path
        database_path = None
        for path in self.database_paths:
            if Path(path).exists():
                database_path = path
                break
        
        if not database_path:
            return DatabaseStatus(
                path="None found",
                exists=False,
                accessible=False,
                tables=[],
                historical_data_count=0,
                backtest_results_count=0,
                latest_data_timestamp=None,
                data_quality_score=0.0
            )
        
        try:
            with sqlite3.connect(database_path, timeout=10) as conn:
                cursor = conn.cursor()
                
                # Get table list
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Count historical data
                historical_data_count = 0
                latest_timestamp = None
                if 'historical_data' in tables:
                    cursor.execute("SELECT COUNT(*) FROM historical_data")
                    historical_data_count = cursor.fetchone()[0]
                    
                    if historical_data_count > 0:
                        cursor.execute("SELECT MAX(timestamp) FROM historical_data")
                        latest_timestamp = cursor.fetchone()[0]
                
                # Count backtest results
                backtest_results_count = 0
                if 'backtest_results' in tables:
                    cursor.execute("SELECT COUNT(*) FROM backtest_results")
                    backtest_results_count = cursor.fetchone()[0]
                
                # Calculate data quality score
                data_quality_score = self._calculate_data_quality_score(
                    historical_data_count, backtest_results_count, latest_timestamp
                )
                
                status = DatabaseStatus(
                    path=database_path,
                    exists=True,
                    accessible=True,
                    tables=tables,
                    historical_data_count=historical_data_count,
                    backtest_results_count=backtest_results_count,
                    latest_data_timestamp=latest_timestamp,
                    data_quality_score=data_quality_score
                )
                
                production_logger.log_database_operation(
                    operation="DIAGNOSTIC_CHECK",
                    table="multiple",
                    record_count=historical_data_count + backtest_results_count,
                    execution_time_ms=0
                )
                
                return status
                
        except Exception as e:
            production_logger.log_error(
                f"Database access failed: {str(e)}",
                error_type=type(e).__name__,
                error_details={"database_path": database_path}
            )
            
            return DatabaseStatus(
                path=database_path,
                exists=True,
                accessible=False,
                tables=[],
                historical_data_count=0,
                backtest_results_count=0,
                latest_data_timestamp=None,
                data_quality_score=0.0
            )
    
    async def test_api_endpoints(self) -> List[APIEndpointStatus]:
        """Test critical API endpoints for availability and response"""
        
        production_logger.api_logger.info("Testing API endpoints")
        
        endpoint_results = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            
            for endpoint, method in self.critical_endpoints:
                start_time = time.time()
                
                try:
                    url = f"{self.base_url}{endpoint}"
                    
                    if method == "GET":
                        async with session.get(url) as response:
                            response_time_ms = (time.time() - start_time) * 1000
                            response_data = None
                            
                            try:
                                response_data = await response.json()
                            except:
                                response_data = {"text": await response.text()}
                            
                            endpoint_results.append(APIEndpointStatus(
                                endpoint=endpoint,
                                method=method,
                                accessible=True,
                                response_time_ms=response_time_ms,
                                status_code=response.status,
                                response_data=response_data,
                                error_message=None
                            ))
                
                except Exception as e:
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    endpoint_results.append(APIEndpointStatus(
                        endpoint=endpoint,
                        method=method,
                        accessible=False,
                        response_time_ms=response_time_ms,
                        status_code=0,
                        response_data=None,
                        error_message=str(e)
                    ))
                    
                    production_logger.log_error(
                        f"API endpoint test failed: {endpoint}",
                        error_type=type(e).__name__,
                        endpoint=endpoint,
                        method=method,
                        error_details={"url": f"{self.base_url}{endpoint}"}
                    )
        
        return endpoint_results
    
    def check_persistent_volumes(self) -> Dict[str, bool]:
        """Check if persistent volumes are properly mounted and writable"""
        
        production_logger.deployment_logger.info("Checking persistent volumes")
        
        volume_paths = {
            '/app/data': 'data_volume',
            '/app/logs': 'logs_volume', 
            '/app/config': 'config_volume'
        }
        
        volume_status = {}
        
        for path, name in volume_paths.items():
            path_obj = Path(path)
            
            # Check if directory exists
            exists = path_obj.exists()
            
            # Check if writable
            writable = False
            if exists:
                try:
                    test_file = path_obj / f"test_write_{int(time.time())}.tmp"
                    test_file.write_text("test")
                    test_file.unlink()
                    writable = True
                except:
                    writable = False
            
            volume_status[name] = exists and writable
            
            production_logger.deployment_logger.info(
                f"Persistent volume check: {name}",
                path=path,
                exists=exists,
                writable=writable,
                status="OK" if exists and writable else "FAILED"
            )
        
        return volume_status
    
    def assess_pipeline_health(self, 
                              database_status: DatabaseStatus, 
                              api_endpoints: List[APIEndpointStatus]) -> str:
        """Assess overall data pipeline health"""
        
        # Check critical factors
        database_healthy = (database_status.accessible and 
                          database_status.historical_data_count > 0)
        
        api_healthy = all(ep.accessible and ep.status_code == 200 
                         for ep in api_endpoints 
                         if ep.endpoint in ['/api/historical-data/discover', '/health'])
        
        data_discovery_healthy = any(ep.endpoint == '/api/historical-data/discover' and 
                                   ep.accessible and ep.status_code == 200 and
                                   ep.response_data and 
                                   ep.response_data.get('success', False)
                                   for ep in api_endpoints)
        
        if database_healthy and api_healthy and data_discovery_healthy:
            return "HEALTHY"
        elif database_healthy and api_healthy:
            return "DEGRADED" 
        elif database_healthy or api_healthy:
            return "CRITICAL"
        else:
            return "FAILED"
    
    def generate_recommendations(self, 
                               database_status: DatabaseStatus,
                               api_endpoints: List[APIEndpointStatus],
                               persistent_volumes: Dict[str, bool]) -> Tuple[List[str], List[str]]:
        """Generate recommendations and identify critical issues"""
        
        recommendations = []
        critical_issues = []
        
        # Database issues
        if not database_status.exists:
            critical_issues.append("NO_DATABASE_FOUND")
            recommendations.append("Create database and run initial data download")
        elif not database_status.accessible:
            critical_issues.append("DATABASE_ACCESS_FAILED") 
            recommendations.append("Check database permissions and file locks")
        elif database_status.historical_data_count == 0:
            critical_issues.append("NO_HISTORICAL_DATA")
            recommendations.append("Run historical data download process")
        
        # API endpoint issues
        failed_endpoints = [ep for ep in api_endpoints if not ep.accessible]
        if failed_endpoints:
            critical_issues.append("API_ENDPOINTS_FAILED")
            recommendations.append("Check FastAPI server status and port configuration")
        
        # Data discovery specific issue
        discover_endpoint = next((ep for ep in api_endpoints 
                                if ep.endpoint == '/api/historical-data/discover'), None)
        if discover_endpoint and discover_endpoint.accessible:
            if (discover_endpoint.response_data and 
                not discover_endpoint.response_data.get('success', False)):
                critical_issues.append("DATA_DISCOVERY_EMPTY_RESPONSE")
                recommendations.append("Check database query in /api/historical-data/discover endpoint")
        
        # Persistent volume issues
        failed_volumes = [name for name, status in persistent_volumes.items() if not status]
        if failed_volumes:
            critical_issues.append("PERSISTENT_VOLUMES_FAILED")
            recommendations.append(f"Check persistent volume mounts: {', '.join(failed_volumes)}")
        
        return recommendations, critical_issues
    
    async def attempt_automated_repairs(self, critical_issues: List[str]) -> List[str]:
        """Attempt automated repairs for identified issues"""
        
        repair_actions = []
        
        # Repair database table structure
        if "NO_DATABASE_FOUND" in critical_issues or "DATABASE_ACCESS_FAILED" in critical_issues:
            if await self._repair_database_structure():
                repair_actions.append("Created missing database tables")
        
        # Repair data directory structure
        if "PERSISTENT_VOLUMES_FAILED" in critical_issues:
            if self._repair_directory_structure():
                repair_actions.append("Created missing data directories")
        
        # Attempt to populate some sample data if completely empty
        if "NO_HISTORICAL_DATA" in critical_issues:
            if await self._create_sample_data():
                repair_actions.append("Created sample historical data for testing")
        
        return repair_actions
    
    async def _repair_database_structure(self) -> bool:
        """Create database tables if missing"""
        
        try:
            # Find or create database path
            database_path = "/app/data/trading_bot.db"
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(database_path) as conn:
                cursor = conn.cursor()
                
                # Create historical_data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS historical_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create backtest_results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        starting_balance REAL NOT NULL,
                        total_return_pct REAL NOT NULL,
                        sharpe_ratio REAL,
                        status TEXT DEFAULT 'completed',
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        trades_count INTEGER,
                        max_drawdown REAL,
                        win_rate REAL
                    )
                """)
                
                conn.commit()
                
            production_logger.database_logger.info(
                "Database structure repair completed",
                database_path=database_path
            )
            
            return True
            
        except Exception as e:
            production_logger.log_error(
                f"Database structure repair failed: {str(e)}",
                error_type=type(e).__name__
            )
            return False
    
    def _repair_directory_structure(self) -> bool:
        """Create missing data directories"""
        
        try:
            directories = [
                "/app/data",
                "/app/data/models",
                "/app/data/strategies",
                "/app/data/speed_demon_cache",
                "/app/logs",
                "/app/config"
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            production_logger.deployment_logger.info(
                "Directory structure repair completed",
                directories_created=len(directories)
            )
            
            return True
            
        except Exception as e:
            production_logger.log_error(
                f"Directory structure repair failed: {str(e)}",
                error_type=type(e).__name__
            )
            return False
    
    async def _create_sample_data(self) -> bool:
        """Create sample historical data for testing if database is empty"""
        
        try:
            database_path = "/app/data/trading_bot.db"
            
            with sqlite3.connect(database_path) as conn:
                cursor = conn.cursor()
                
                # Check if data already exists
                cursor.execute("SELECT COUNT(*) FROM historical_data")
                existing_count = cursor.fetchone()[0]
                
                if existing_count > 0:
                    return False  # Data already exists
                
                # Create sample data for testing
                sample_data = []
                base_time = int(datetime.now().timestamp()) - (30 * 24 * 60 * 60)  # 30 days ago
                base_price = 50000.0
                
                for i in range(1000):  # 1000 sample records
                    timestamp = base_time + (i * 900)  # 15 minute intervals
                    price_variation = 1 + (i % 100 - 50) * 0.001  # Small price variation
                    price = base_price * price_variation
                    
                    sample_data.append((
                        'BTCUSDT',
                        '15m', 
                        timestamp,
                        price * 0.999,  # open
                        price * 1.001,  # high
                        price * 0.998,  # low
                        price,           # close
                        100.0 + (i % 50)  # volume
                    ))
                
                cursor.executemany("""
                    INSERT INTO historical_data (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, sample_data)
                
                conn.commit()
                
                production_logger.database_logger.info(
                    "Sample data created for testing",
                    records_created=len(sample_data),
                    symbol="BTCUSDT",
                    timeframe="15m"
                )
                
                return True
                
        except Exception as e:
            production_logger.log_error(
                f"Sample data creation failed: {str(e)}",
                error_type=type(e).__name__
            )
            return False
    
    def _calculate_data_quality_score(self, 
                                    historical_count: int, 
                                    backtest_count: int, 
                                    latest_timestamp: Optional[str]) -> float:
        """Calculate data quality score based on various factors"""
        
        score = 0.0
        
        # Historical data volume (40% of score)
        if historical_count >= 10000:
            score += 0.4
        elif historical_count >= 1000:
            score += 0.3
        elif historical_count > 0:
            score += 0.2
        
        # Backtest results existence (20% of score) 
        if backtest_count >= 10:
            score += 0.2
        elif backtest_count > 0:
            score += 0.1
        
        # Data recency (40% of score)
        if latest_timestamp:
            try:
                if isinstance(latest_timestamp, (int, float)):
                    if latest_timestamp > 1e10:
                        latest_dt = datetime.fromtimestamp(latest_timestamp / 1000)
                    else:
                        latest_dt = datetime.fromtimestamp(latest_timestamp)
                else:
                    latest_dt = datetime.fromisoformat(str(latest_timestamp).replace('Z', '+00:00'))
                
                days_old = (datetime.now() - latest_dt).days
                
                if days_old <= 1:
                    score += 0.4  # Very recent
                elif days_old <= 7:
                    score += 0.3  # Recent
                elif days_old <= 30:
                    score += 0.2  # Moderately recent
                else:
                    score += 0.1  # Old data
            except:
                score += 0.1  # Can't parse timestamp
        
        return round(score, 2)
    
    def generate_report_summary(self, report: DataDiscoveryReport) -> str:
        """Generate human-readable report summary"""
        
        summary = f"""
🔍 DATA DISCOVERY DIAGNOSTIC REPORT
====================================
Environment: {report.environment}
Timestamp: {report.timestamp}
Pipeline Health: {report.data_pipeline_health}

📊 DATABASE STATUS:
- Path: {report.database_status.path}
- Accessible: {'✅' if report.database_status.accessible else '❌'}
- Historical Records: {report.database_status.historical_data_count:,}
- Backtest Results: {report.database_status.backtest_results_count:,}
- Data Quality Score: {report.database_status.data_quality_score}/1.0

🌐 API ENDPOINTS:"""
        
        for endpoint in report.api_endpoints:
            status_icon = '✅' if endpoint.accessible and endpoint.status_code == 200 else '❌'
            summary += f"""
- {endpoint.method} {endpoint.endpoint}: {status_icon} ({endpoint.status_code}) - {endpoint.response_time_ms:.1f}ms"""
        
        summary += f"""

💾 PERSISTENT VOLUMES:"""
        
        for volume, status in report.persistent_volumes.items():
            status_icon = '✅' if status else '❌'
            summary += f"""
- {volume}: {status_icon}"""
        
        if report.critical_issues:
            summary += f"""

🚨 CRITICAL ISSUES:"""
            for issue in report.critical_issues:
                summary += f"""
- {issue}"""
        
        if report.recommendations:
            summary += f"""

💡 RECOMMENDATIONS:"""
            for rec in report.recommendations:
                summary += f"""
- {rec}"""
        
        if report.repair_actions_taken:
            summary += f"""

🔧 REPAIR ACTIONS TAKEN:"""
            for action in report.repair_actions_taken:
                summary += f"""
- {action}"""
        
        return summary


async def main():
    """Run comprehensive data discovery diagnostic"""
    
    print("🔍 Starting comprehensive data discovery diagnostic...")
    
    # Create diagnostic tool
    diagnostic = DataDiscoveryDiagnostic()
    
    # Run diagnostic
    report = await diagnostic.run_comprehensive_diagnostic()
    
    # Generate and display report
    summary = diagnostic.generate_report_summary(report)
    print(summary)
    
    # Save detailed report to file
    report_path = Path("/app/logs/data_discovery_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Return exit code based on pipeline health
    if report.data_pipeline_health in ['HEALTHY', 'DEGRADED']:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)