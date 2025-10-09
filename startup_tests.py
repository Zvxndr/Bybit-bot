#!/usr/bin/env python3
"""
🧪 STARTUP TEST SUITE
====================

Comprehensive test suite to validate system startup and core functionality.
Designed for DigitalOcean App Platform deployment validation.
"""

import os
import sys
import asyncio
import requests
import time
import json
import signal
import subprocess
from pathlib import Path
from datetime import datetime
import threading
from contextlib import contextmanager

class StartupTests:
    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "failures": [],
            "duration_seconds": 0
        }
        self.server_process = None
        self.start_time = time.time()
    
    def run_all_tests(self):
        """Run complete startup test suite"""
        print("🧪 STARTUP TEST SUITE STARTING...")
        print("=" * 50)
        
        try:
            # Phase 1: Pre-startup tests
            self._test_environment_setup()
            self._test_configuration_loading()
            self._test_module_imports()
            
            # Phase 2: Application startup tests
            self._test_application_startup()
            
            # Phase 3: Runtime tests (if app started successfully)
            if self.server_process:
                self._test_api_endpoints()
                self._test_frontend_accessibility()
                self._test_core_functionality()
            
        except KeyboardInterrupt:
            print("\n⚠️ Tests interrupted by user")
        except Exception as e:
            self._record_failure(f"Test suite error: {e}")
        finally:
            self._cleanup()
            self._generate_test_report()
    
    def _test_environment_setup(self):
        """Test environment and dependencies"""
        print("🌍 Testing Environment Setup...")
        
        # Test 1: Python version
        if self._run_test("Python version >= 3.8", 
                         lambda: sys.version_info >= (3, 8)):
            pass
        
        # Test 2: Required files exist
        required_files = ['config/config.yaml', 'src/main.py', 'frontend/unified_dashboard.html']
        for file_path in required_files:
            self._run_test(f"Required file exists: {file_path}", 
                          lambda f=file_path: os.path.exists(f))
        
        # Test 3: Data directory writable
        self._run_test("Data directory writable", self._test_data_directory_writable)
    
    def _test_configuration_loading(self):
        """Test configuration file loading"""
        print("⚙️ Testing Configuration Loading...")
        
        # Test configuration file syntax
        self._run_test("Config YAML syntax valid", self._test_config_yaml_syntax)
        
        # Test required config sections
        self._run_test("Required config sections present", self._test_required_config_sections)
    
    def _test_module_imports(self):
        """Test critical module imports"""
        print("📦 Testing Module Imports...")
        
        critical_imports = [
            ('src.main', 'Main application module'),
            ('src.bybit_api', 'Bybit API client'),
            ('src.data.multi_exchange_provider', 'Multi-exchange provider'),
            ('src.bot.data', 'Bot data management'),
        ]
        
        for module_path, description in critical_imports:
            self._run_test(f"Import {description}", 
                          lambda m=module_path: self._test_import(m))
    
    def _test_application_startup(self):
        """Test application startup process"""
        print("🚀 Testing Application Startup...")
        
        # Test server startup
        if self._run_test("Server starts successfully", self._test_server_startup):
            # Test server responds to health check
            time.sleep(3)  # Give server time to fully initialize
            self._run_test("Server health check responds", self._test_server_health)
    
    def _test_api_endpoints(self):
        """Test API endpoint accessibility"""
        print("🌐 Testing API Endpoints...")
        
        base_url = "http://localhost:8080"
        
        endpoints = [
            ("/", "Root endpoint"),
            ("/health", "Health check"),
            ("/api/portfolio/status", "Portfolio status API"),
            ("/api/ml-risk-metrics", "ML risk metrics API")
        ]
        
        for endpoint, description in endpoints:
            self._run_test(f"API endpoint {description}", 
                          lambda e=endpoint: self._test_api_endpoint(base_url + e))
    
    def _test_frontend_accessibility(self):
        """Test frontend page accessibility"""
        print("🖥️ Testing Frontend Accessibility...")
        
        self._run_test("Frontend dashboard accessible", 
                      lambda: self._test_api_endpoint("http://localhost:8080/"))
    
    def _test_core_functionality(self):
        """Test core system functionality"""
        print("⚡ Testing Core Functionality...")
        
        # Test configuration endpoints
        self._run_test("Configuration API responds", 
                      lambda: self._test_api_endpoint("http://localhost:8080/api/config"))
    
    # Helper methods for individual tests
    def _run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results"""
        self.test_results["tests_run"] += 1
        
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}")
                self.test_results["tests_passed"] += 1
                return True
            else:
                print(f"❌ {test_name}")
                self._record_failure(test_name)
                return False
        except Exception as e:
            print(f"❌ {test_name} - Exception: {e}")
            self._record_failure(f"{test_name}: {e}")
            return False
    
    def _test_data_directory_writable(self) -> bool:
        """Test if data directory is writable"""
        try:
            test_file = Path("data/startup_test.tmp")
            test_file.parent.mkdir(exist_ok=True)
            test_file.write_text("test")
            test_file.unlink()
            return True
        except:
            return False
    
    def _test_config_yaml_syntax(self) -> bool:
        """Test YAML configuration file syntax"""
        try:
            import yaml
            with open('config/config.yaml', 'r') as f:
                yaml.safe_load(f)
            return True
        except:
            return False
    
    def _test_required_config_sections(self) -> bool:
        """Test required configuration sections exist"""
        try:
            import yaml
            with open('config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            required_sections = ['trading', 'database', 'ml_risk', 'api']
            return all(section in config for section in required_sections)
        except:
            return False
    
    def _test_import(self, module_path: str) -> bool:
        """Test if module can be imported"""
        try:
            import importlib
            importlib.import_module(module_path)
            return True
        except ImportError:
            return False
    
    def _test_server_startup(self) -> bool:
        """Test server startup"""
        try:
            # Start server in background
            self.server_process = subprocess.Popen([
                sys.executable, "-m", "src.main"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Wait a moment for startup
            time.sleep(5)
            
            # Check if process is still running
            if self.server_process.poll() is None:
                return True
            else:
                # Process died, capture output
                stdout, stderr = self.server_process.communicate()
                print(f"Server startup failed. STDERR: {stderr}")
                return False
        except Exception as e:
            print(f"Server startup exception: {e}")
            return False
    
    def _test_server_health(self) -> bool:
        """Test server health endpoint"""
        try:
            response = requests.get("http://localhost:8080/health", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def _test_api_endpoint(self, url: str) -> bool:
        """Test if API endpoint responds"""
        try:
            response = requests.get(url, timeout=10)
            return response.status_code < 500  # Accept any non-server-error response
        except:
            return False
    
    def _record_failure(self, failure_msg: str):
        """Record a test failure"""
        self.test_results["tests_failed"] += 1
        self.test_results["failures"].append(failure_msg)
    
    def _cleanup(self):
        """Clean up test resources"""
        if self.server_process:
            print("\n🧹 Cleaning up server process...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
    
    def _generate_test_report(self):
        """Generate final test report"""
        self.test_results["duration_seconds"] = time.time() - self.start_time
        
        print("\n" + "=" * 50)
        print("📋 STARTUP TEST REPORT")
        print("=" * 50)
        
        passed = self.test_results["tests_passed"]
        total = self.test_results["tests_run"]
        failed = self.test_results["tests_failed"]
        
        print(f"Tests Run: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️ Duration: {self.test_results['duration_seconds']:.2f}s")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! System ready for deployment.")
            status = "READY"
        else:
            print(f"\n⚠️ {failed} TESTS FAILED!")
            print("\nFailures:")
            for failure in self.test_results["failures"]:
                print(f"  - {failure}")
            status = "NOT_READY"
        
        # Save results
        self.test_results["status"] = status
        with open('startup_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n💾 Test results saved to: startup_test_results.json")
        
        # Exit with appropriate code
        return 0 if status == "READY" else 1

def main():
    """Run startup tests"""
    tester = StartupTests()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()