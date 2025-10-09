#!/usr/bin/env python3
"""
🏥 HEALTH CHECK ENDPOINT
=======================

Comprehensive health check for DigitalOcean App Platform deployment.
Validates all critical system components and returns detailed status.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import traceback

class HealthChecker:
    def __init__(self):
        self.checks = {}
        self.start_time = time.time()
        
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "version": "1.0.0",
            "environment": os.getenv("APP_ENV", "development"),
            "checks": {},
            "errors": [],
            "warnings": []
        }
        
        # Run all health checks
        self._check_system_resources(health_status)
        self._check_configuration(health_status)
        self._check_database_connectivity(health_status)
        self._check_api_functionality(health_status)
        self._check_trading_components(health_status)
        
        # Determine overall status
        if health_status["errors"]:
            health_status["status"] = "unhealthy"
        elif health_status["warnings"]:
            health_status["status"] = "degraded"
        
        return health_status
    
    def _check_system_resources(self, status: Dict[str, Any]):
        """Check system resources and environment"""
        try:
            # Check disk space
            data_dir = Path("data")
            if data_dir.exists():
                stat = os.statvfs(str(data_dir))
                free_bytes = stat.f_bavail * stat.f_frsize
                free_gb = free_bytes / (1024**3)
                
                if free_gb < 0.1:  # Less than 100MB
                    status["errors"].append("Low disk space: < 100MB free")
                elif free_gb < 0.5:  # Less than 500MB
                    status["warnings"].append("Low disk space: < 500MB free")
                
                status["checks"]["disk_space_gb"] = round(free_gb, 2)
            
            # Check memory usage (basic)
            status["checks"]["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}"
            status["checks"]["platform"] = sys.platform
            
        except Exception as e:
            status["errors"].append(f"System resource check failed: {e}")
    
    def _check_configuration(self, status: Dict[str, Any]):
        """Check configuration files and environment"""
        try:
            # Check config file exists
            config_path = Path("config/config.yaml")
            if not config_path.exists():
                status["errors"].append("Configuration file missing: config/config.yaml")
                return
            
            # Try to load config
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            required_sections = ["trading", "database", "ml_risk", "api"]
            missing_sections = [s for s in required_sections if s not in config]
            
            if missing_sections:
                status["errors"].append(f"Missing config sections: {missing_sections}")
            else:
                status["checks"]["configuration"] = "valid"
            
            # Check environment variables
            required_env_vars = ["PORT", "APP_ENV"]
            missing_env = [var for var in required_env_vars if not os.getenv(var)]
            
            if missing_env:
                status["warnings"].append(f"Missing environment variables: {missing_env}")
            
        except Exception as e:
            status["errors"].append(f"Configuration check failed: {e}")
    
    def _check_database_connectivity(self, status: Dict[str, Any]):
        """Check database connection and setup"""
        try:
            import sqlite3
            
            db_path = Path("data/trading_bot.db")
            if not db_path.parent.exists():
                db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try database connection
            conn = sqlite3.connect(str(db_path), timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            
            status["checks"]["database"] = "connected"
            
        except Exception as e:
            status["errors"].append(f"Database connectivity failed: {e}")
    
    def _check_api_functionality(self, status: Dict[str, Any]):
        """Check API functionality"""
        try:
            # Test critical imports
            from src.main import TradingAPI
            status["checks"]["main_import"] = "success"
            
            # Check if FastAPI can be imported
            import fastapi
            import uvicorn
            status["checks"]["web_framework"] = "available"
            
        except Exception as e:
            status["errors"].append(f"API functionality check failed: {e}")
    
    def _check_trading_components(self, status: Dict[str, Any]):
        """Check trading system components"""
        try:
            # Check API client import
            from src.bybit_api import BybitAPIClient
            status["checks"]["bybit_client"] = "available"
            
            # Check data providers
            from src.data.multi_exchange_provider import MultiExchangeDataManager
            status["checks"]["data_provider"] = "available"
            
            # Check ML components
            try:
                from src.bot.ml_strategy_discovery.ml_engine import MLStrategyDiscoveryEngine
                status["checks"]["ml_engine"] = "available"
            except ImportError:
                status["warnings"].append("ML engine not available")
            
        except Exception as e:
            status["warnings"].append(f"Some trading components unavailable: {e}")

def create_health_endpoint():
    """Create health check endpoint for FastAPI"""
    
    checker = HealthChecker()
    
    def health_check():
        """Health check endpoint handler"""
        try:
            health_data = checker.perform_health_check()
            
            # Return appropriate HTTP status
            if health_data["status"] == "healthy":
                return health_data
            elif health_data["status"] == "degraded":
                # Still return 200 but with warnings
                return health_data
            else:
                # Return 503 Service Unavailable for unhealthy
                from fastapi import HTTPException
                raise HTTPException(status_code=503, detail=health_data)
                
        except Exception as e:
            error_response = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=error_response)
    
    return health_check

# Standalone health check for command line use
def main():
    """Run standalone health check"""
    checker = HealthChecker()
    health_data = checker.perform_health_check()
    
    print("🏥 SYSTEM HEALTH CHECK")
    print("=" * 30)
    print(f"Status: {health_data['status'].upper()}")
    print(f"Environment: {health_data['environment']}")
    print(f"Uptime: {health_data['uptime_seconds']:.1f}s")
    
    if health_data.get("checks"):
        print("\n✅ Successful Checks:")
        for check, result in health_data["checks"].items():
            print(f"  {check}: {result}")
    
    if health_data.get("warnings"):
        print("\n⚠️ Warnings:")
        for warning in health_data["warnings"]:
            print(f"  {warning}")
    
    if health_data.get("errors"):
        print("\n❌ Errors:")
        for error in health_data["errors"]:
            print(f"  {error}")
    
    # Save detailed results
    with open('health_check_results.json', 'w') as f:
        json.dump(health_data, f, indent=2)
    
    # Exit with appropriate code
    if health_data["status"] == "unhealthy":
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()