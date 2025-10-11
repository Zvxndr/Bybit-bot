#!/usr/bin/env python3
"""
Production Startup Integration for DigitalOcean App Platform
===========================================================

Integrates comprehensive diagnostic and monitoring systems into the
main application startup process for DigitalOcean deployment.

Features:
- Validates data persistence on startup
- Runs diagnostic checks
- Initializes production logging
- Validates API connectivity
- Sets up monitoring systems
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
from datetime import datetime

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.production_logger import production_logger
    from src.enhanced_health_check import health_checker
    from src.data_discovery_diagnostic import DataDiscoveryDiagnostic
    DIAGNOSTIC_SYSTEMS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Diagnostic systems not available: {e}")
    DIAGNOSTIC_SYSTEMS_AVAILABLE = False
    production_logger = logging.getLogger(__name__)


class ProductionStartupIntegration:
    """Handles production startup diagnostics and system validation"""
    
    def __init__(self):
        self.startup_time = datetime.now()
        self.environment = os.getenv('TRADING_ENVIRONMENT', 'development')
        self.deployment_id = os.getenv('DO_APP_NAME', 'local')
        self.startup_checks_passed = False
        
        # Startup configuration
        self.required_directories = [
            '/app/data',
            '/app/logs',
            '/app/config'
        ]
        
        self.critical_files = [
            '/app/data/trading_bot.db',
            'data/trading_bot.db'
        ]
        
        print(f"🚀 Production Startup Integration - {self.environment}")
        print(f"📅 Startup Time: {self.startup_time}")
        print(f"🆔 Deployment ID: {self.deployment_id}")
    
    async def run_startup_validation(self) -> Dict[str, Any]:
        """Run comprehensive startup validation and diagnostics"""
        
        validation_results = {
            "timestamp": self.startup_time.isoformat(),
            "environment": self.environment,
            "deployment_id": self.deployment_id,
            "startup_checks": {},
            "diagnostics": {},
            "warnings": [],
            "errors": [],
            "startup_duration_ms": 0,
            "ready_for_production": False
        }
        
        start_time = time.time()
        
        try:
            print("🔍 Starting production startup validation...")
            
            # 1. Validate directory structure
            print("📁 Validating directory structure...")
            validation_results["startup_checks"]["directories"] = self._validate_directories()
            
            # 2. Validate database access
            print("🗄️ Validating database access...")
            validation_results["startup_checks"]["database"] = await self._validate_database_access()
            
            # 3. Run data discovery diagnostic
            if DIAGNOSTIC_SYSTEMS_AVAILABLE:
                print("🔍 Running data discovery diagnostic...")
                validation_results["diagnostics"]["data_discovery"] = await self._run_data_diagnostic()
                
                # 4. Run comprehensive health check
                print("🏥 Running comprehensive health check...")
                validation_results["diagnostics"]["health_check"] = await self._run_health_check()
            else:
                validation_results["warnings"].append("Diagnostic systems not available")
            
            # 5. Validate environment variables
            print("⚙️ Validating environment configuration...")
            validation_results["startup_checks"]["environment"] = self._validate_environment()
            
            # 6. Test API endpoints (if server is running)
            print("🌐 Testing internal API connectivity...")
            validation_results["startup_checks"]["api_connectivity"] = await self._test_api_connectivity()
            
            # 7. Assess overall readiness
            validation_results["ready_for_production"] = self._assess_production_readiness(validation_results)
            
            # Calculate startup duration
            validation_results["startup_duration_ms"] = (time.time() - start_time) * 1000
            
            self.startup_checks_passed = validation_results["ready_for_production"]
            
            # Log startup results
            if DIAGNOSTIC_SYSTEMS_AVAILABLE:
                production_logger.deployment_logger.info(
                    "Production startup validation completed",
                    ready_for_production=validation_results["ready_for_production"],
                    startup_duration_ms=validation_results["startup_duration_ms"],
                    warnings_count=len(validation_results["warnings"]),
                    errors_count=len(validation_results["errors"])
                )
            
            print(f"✅ Startup validation completed in {validation_results['startup_duration_ms']:.1f}ms")
            print(f"🎯 Production Ready: {'YES' if validation_results['ready_for_production'] else 'NO'}")
            
            return validation_results
            
        except Exception as e:
            validation_results["errors"].append(f"Startup validation failed: {str(e)}")
            validation_results["startup_duration_ms"] = (time.time() - start_time) * 1000
            
            if DIAGNOSTIC_SYSTEMS_AVAILABLE:
                production_logger.log_error(
                    "Startup validation system error",
                    error_type=type(e).__name__,
                    error_details={"message": str(e)}
                )
            
            print(f"❌ Startup validation failed: {e}")
            return validation_results
    
    def _validate_directories(self) -> Dict[str, Any]:
        """Validate required directory structure"""
        
        directory_status = {
            "status": "HEALTHY",
            "directories": {}
        }
        
        for directory in self.required_directories:
            path_obj = Path(directory)
            
            dir_info = {
                "path": directory,
                "exists": path_obj.exists(),
                "writable": False,
                "created_on_startup": False
            }
            
            # Create directory if it doesn't exist
            if not path_obj.exists():
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    dir_info["exists"] = True
                    dir_info["created_on_startup"] = True
                    print(f"📁 Created directory: {directory}")
                except Exception as e:
                    directory_status["status"] = "ERROR"
                    dir_info["error"] = str(e)
            
            # Test write access
            if dir_info["exists"]:
                try:
                    test_file = path_obj / f"startup_test_{int(time.time())}.tmp"
                    test_file.write_text("startup test")
                    test_file.unlink()
                    dir_info["writable"] = True
                except:
                    directory_status["status"] = "WARNING"
                    dir_info["writable"] = False
            
            directory_status["directories"][directory] = dir_info
        
        return directory_status
    
    async def _validate_database_access(self) -> Dict[str, Any]:
        """Validate database accessibility and basic structure"""
        
        database_status = {
            "status": "UNKNOWN",
            "databases": {},
            "primary_database": None,
            "tables_created": []
        }
        
        # Try to find or create a database
        import sqlite3
        
        for db_path in self.critical_files:
            path_obj = Path(db_path)
            
            db_info = {
                "path": db_path,
                "exists": path_obj.exists(),
                "accessible": False,
                "tables": [],
                "created_on_startup": False
            }
            
            # Create database directory if needed
            if not path_obj.parent.exists():
                path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Try to access or create database
            try:
                with sqlite3.connect(db_path, timeout=10) as conn:
                    cursor = conn.cursor()
                    
                    # Test basic connectivity
                    cursor.execute("SELECT 1")
                    db_info["accessible"] = True
                    
                    # Get existing tables
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    db_info["tables"] = [row[0] for row in cursor.fetchall()]
                    
                    # Create basic tables if they don't exist
                    tables_to_create = []
                    
                    if 'historical_data' not in db_info["tables"]:
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
                        tables_to_create.append("historical_data")
                    
                    if 'backtest_results' not in db_info["tables"]:
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
                        tables_to_create.append("backtest_results")
                    
                    if tables_to_create:
                        conn.commit()
                        db_info["created_on_startup"] = True
                        database_status["tables_created"].extend(tables_to_create)
                        print(f"🗄️ Created tables in {db_path}: {', '.join(tables_to_create)}")
                    
                    # Update table list
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    db_info["tables"] = [row[0] for row in cursor.fetchall()]
                    
                    # Set as primary database if accessible
                    if not database_status["primary_database"]:
                        database_status["primary_database"] = db_path
                        database_status["status"] = "HEALTHY"
                
            except Exception as e:
                db_info["error"] = str(e)
                if database_status["status"] == "UNKNOWN":
                    database_status["status"] = "ERROR"
            
            database_status["databases"][db_path] = db_info
        
        return database_status
    
    async def _run_data_diagnostic(self) -> Dict[str, Any]:
        """Run data discovery diagnostic"""
        
        try:
            diagnostic = DataDiscoveryDiagnostic(base_url="http://localhost:8080")
            report = await diagnostic.run_comprehensive_diagnostic()
            
            from dataclasses import asdict
            return {
                "status": "COMPLETED",
                "pipeline_health": report.data_pipeline_health,
                "critical_issues_count": len(report.critical_issues),
                "repair_actions_count": len(report.repair_actions_taken),
                "datasets_found": report.database_status.historical_data_count if hasattr(report.database_status, 'historical_data_count') else 0
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        
        try:
            health_status = await health_checker.comprehensive_health_check()
            
            return {
                "status": "COMPLETED",
                "overall_health": health_status.get('overall_status', 'UNKNOWN'),
                "warnings_count": len(health_status.get('warnings', [])),
                "errors_count": len(health_status.get('errors', [])),
                "total_data_points": health_status.get('data_availability', {}).get('total_data_points', 0)
            }
            
        except Exception as e:
            return {
                "status": "ERROR", 
                "error": str(e)
            }
    
    def _validate_environment(self) -> Dict[str, Any]:
        """Validate critical environment variables"""
        
        env_status = {
            "status": "HEALTHY",
            "variables": {},
            "missing_critical": []
        }
        
        # Critical environment variables
        critical_vars = [
            "TRADING_ENVIRONMENT",
            "PYTHONPATH"
        ]
        
        # Optional but recommended variables
        optional_vars = [
            "BYBIT_TESTNET_API_KEY",
            "BYBIT_TESTNET_API_SECRET",
            "LOG_LEVEL",
            "DO_APP_NAME"
        ]
        
        # Check critical variables
        for var in critical_vars:
            value = os.getenv(var)
            env_status["variables"][var] = {
                "present": value is not None,
                "value_length": len(value) if value else 0,
                "critical": True
            }
            
            if not value:
                env_status["missing_critical"].append(var)
                env_status["status"] = "WARNING"
        
        # Check optional variables
        for var in optional_vars:
            value = os.getenv(var)
            env_status["variables"][var] = {
                "present": value is not None,
                "value_length": len(value) if value else 0,
                "critical": False
            }
        
        return env_status
    
    async def _test_api_connectivity(self) -> Dict[str, Any]:
        """Test internal API connectivity (if server is running)"""
        
        api_status = {
            "status": "UNKNOWN",
            "server_running": False,
            "endpoints_tested": 0,
            "endpoints_accessible": 0
        }
        
        # Skip API test during startup (server not yet running)
        # This would be used for post-startup validation
        
        try:
            import aiohttp
            
            # Test if server is responding on localhost
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                try:
                    async with session.get("http://localhost:8080/health") as response:
                        api_status["server_running"] = True
                        api_status["endpoints_tested"] = 1
                        if response.status == 200:
                            api_status["endpoints_accessible"] = 1
                            api_status["status"] = "HEALTHY"
                        else:
                            api_status["status"] = "WARNING"
                except:
                    api_status["status"] = "NOT_RUNNING"
                    
        except Exception as e:
            api_status["error"] = str(e)
            api_status["status"] = "ERROR"
        
        return api_status
    
    def _assess_production_readiness(self, validation_results: Dict[str, Any]) -> bool:
        """Assess if system is ready for production based on validation results"""
        
        # Check critical systems
        directories_ok = (validation_results["startup_checks"]["directories"]["status"] in ["HEALTHY", "WARNING"])
        database_ok = (validation_results["startup_checks"]["database"]["status"] in ["HEALTHY", "WARNING"])
        environment_ok = (validation_results["startup_checks"]["environment"]["status"] in ["HEALTHY", "WARNING"])
        
        # Count critical errors
        critical_errors = len([err for err in validation_results["errors"] if "CRITICAL" in err or "ERROR" in err])
        
        # System is ready if basic infrastructure is working and no critical errors
        return directories_ok and database_ok and environment_ok and critical_errors == 0
    
    def generate_startup_summary(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable startup summary"""
        
        summary = f"""
🚀 PRODUCTION STARTUP VALIDATION SUMMARY
=========================================
Environment: {validation_results['environment']}
Deployment ID: {validation_results['deployment_id']}
Startup Time: {validation_results['startup_duration_ms']:.1f}ms
Production Ready: {'✅ YES' if validation_results['ready_for_production'] else '❌ NO'}

📊 SYSTEM CHECKS:
- Directories: {validation_results['startup_checks']['directories']['status']}
- Database: {validation_results['startup_checks']['database']['status']}  
- Environment: {validation_results['startup_checks']['environment']['status']}
- API Connectivity: {validation_results['startup_checks']['api_connectivity']['status']}

🔍 DIAGNOSTICS:"""
        
        if 'data_discovery' in validation_results['diagnostics']:
            dd = validation_results['diagnostics']['data_discovery']
            summary += f"""
- Data Discovery: {dd['status']} (Issues: {dd.get('critical_issues_count', 0)}, Datasets: {dd.get('datasets_found', 0)})"""
        
        if 'health_check' in validation_results['diagnostics']:
            hc = validation_results['diagnostics']['health_check']
            summary += f"""
- Health Check: {hc['status']} (Overall: {hc.get('overall_health', 'UNKNOWN')}, Data: {hc.get('total_data_points', 0):,} points)"""
        
        if validation_results['warnings']:
            summary += f"""

⚠️ WARNINGS ({len(validation_results['warnings'])}):"""
            for warning in validation_results['warnings'][:3]:  # Show first 3
                summary += f"""
- {warning}"""
        
        if validation_results['errors']:
            summary += f"""

❌ ERRORS ({len(validation_results['errors'])}):"""
            for error in validation_results['errors'][:3]:  # Show first 3
                summary += f"""
- {error}"""
        
        return summary


# Global startup integration instance
startup_integration = ProductionStartupIntegration()


async def run_startup_validation() -> Dict[str, Any]:
    """Run startup validation - for use in main application"""
    return await startup_integration.run_startup_validation()


async def main():
    """Run standalone startup validation"""
    
    print("🔥 Production Startup Validation System")
    print("=" * 50)
    
    validation_results = await startup_integration.run_startup_validation()
    
    # Print detailed summary
    summary = startup_integration.generate_startup_summary(validation_results)
    print(summary)
    
    # Return appropriate exit code
    return 0 if validation_results["ready_for_production"] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)