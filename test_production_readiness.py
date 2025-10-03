#!/usr/bin/env python3
"""
Production Readiness API Test Suite
===================================

Tests all frontend API endpoints to ensure backend integration is complete.
Verifies that every frontend API call has a working backend endpoint.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import requests
import json
import time
import threading
from typing import Dict, List, Any
from src.frontend_server import FrontendHandler
import http.server
import socketserver

class APITestSuite:
    """Comprehensive API testing for production readiness"""
    
    def __init__(self, base_url="http://localhost:8081"):
        self.base_url = base_url
        self.test_results = {}
        self.server_process = None
        
    def start_test_server(self):
        """Start test server on port 8081"""
        try:
            handler = FrontendHandler
            httpd = socketserver.TCPServer(("", 8081), handler)
            server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            server_thread.start()
            print("🚀 Test server started on port 8081")
            time.sleep(2)  # Give server time to start
            return True
        except Exception as e:
            print(f"❌ Failed to start test server: {e}")
            return False
    
    def test_get_endpoints(self) -> Dict[str, bool]:
        """Test all GET API endpoints that frontend uses"""
        get_endpoints = [
            "/api/system-stats",
            "/api/debug-status", 
            "/api/multi-balance",
            "/api/trades/testnet",
            "/api/system-status",
            "/api/positions",
            "/health"
        ]
        
        results = {}
        print("\n🔍 Testing GET Endpoints...")
        
        for endpoint in get_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                success = response.status_code == 200
                results[endpoint] = success
                status = "✅" if success else "❌"
                print(f"{status} {endpoint} - {response.status_code}")
                
                if success and endpoint.startswith('/api/'):
                    # Verify JSON response
                    data = response.json()
                    if not isinstance(data, dict):
                        results[endpoint] = False
                        print(f"  ⚠️  Invalid JSON response format")
                        
            except Exception as e:
                results[endpoint] = False
                print(f"❌ {endpoint} - ERROR: {e}")
        
        return results
    
    def test_post_endpoints(self) -> Dict[str, bool]:
        """Test all POST API endpoints that frontend uses"""
        post_endpoints = [
            "/api/emergency-stop",
            "/api/pause",
            "/api/resume", 
            "/api/strategy/promote",
            "/api/admin/close-all-positions",
            "/api/admin/cancel-all-orders",
            "/api/strategy/create",
            "/api/strategy/backtest",
            "/api/risk/limits",
            "/api/risk/scan"
        ]
        
        results = {}
        print("\n🔍 Testing POST Endpoints...")
        
        for endpoint in post_endpoints:
            try:
                # Prepare test data
                test_data = {}
                if endpoint == "/api/strategy/promote":
                    test_data = {"strategyId": "test-strategy", "targetStage": "testnet"}
                elif "strategy" in endpoint:
                    test_data = {"strategyId": "test-strategy"}
                elif "risk" in endpoint:
                    test_data = {"maxPositionSize": 5, "dailyLossLimit": 1000}
                
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=test_data,
                    timeout=5,
                    headers={"Content-Type": "application/json"}
                )
                
                success = response.status_code == 200
                results[endpoint] = success
                status = "✅" if success else "❌"
                print(f"{status} {endpoint} - {response.status_code}")
                
                if success:
                    # Verify JSON response has success field
                    data = response.json()
                    if not data.get("success"):
                        print(f"  ⚠️  Response missing 'success' field")
                    else:
                        print(f"  ✅ {data.get('message', 'OK')}")
                        
            except Exception as e:
                results[endpoint] = False
                print(f"❌ {endpoint} - ERROR: {e}")
        
        return results
    
    def test_frontend_dashboard_load(self) -> bool:
        """Test that the main dashboard loads without errors"""
        print("\n🔍 Testing Dashboard Load...")
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            success = response.status_code == 200 and "AdminLTE" in response.text
            
            if success:
                print("✅ Dashboard loads successfully")
                print(f"  📊 Response size: {len(response.text)} bytes")
                
                # Check for key dashboard elements
                required_elements = [
                    "Strategy Manager",
                    "Live Trading", 
                    "Risk Management",
                    "Advanced Analytics",
                    "emergency-stop",
                    "fetchSystemStats"
                ]
                
                missing_elements = []
                for element in required_elements:
                    if element not in response.text:
                        missing_elements.append(element)
                
                if missing_elements:
                    print(f"  ⚠️  Missing elements: {missing_elements}")
                    return False
                else:
                    print("  ✅ All required dashboard elements present")
                    return True
            else:
                print(f"❌ Dashboard failed to load - {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Dashboard load error: {e}")
            return False
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run complete production readiness test suite"""
        print("🚀 Starting Production Readiness API Test Suite")
        print("=" * 60)
        
        # Start test server
        if not self.start_test_server():
            return {"success": False, "error": "Failed to start test server"}
        
        # Run all tests
        get_results = self.test_get_endpoints()
        post_results = self.test_post_endpoints()  
        dashboard_result = self.test_frontend_dashboard_load()
        
        # Calculate overall results
        total_get = len(get_results)
        passed_get = sum(get_results.values())
        total_post = len(post_results)
        passed_post = sum(post_results.values())
        
        overall_success = (
            passed_get == total_get and 
            passed_post == total_post and 
            dashboard_result
        )
        
        # Generate report
        print("\n" + "=" * 60)
        print("📊 PRODUCTION READINESS TEST RESULTS")
        print("=" * 60)
        
        print(f"📡 GET Endpoints:  {passed_get}/{total_get} ({'✅' if passed_get == total_get else '❌'})")
        print(f"📤 POST Endpoints: {passed_post}/{total_post} ({'✅' if passed_post == total_post else '❌'})")
        print(f"🖥️  Dashboard Load: {'✅' if dashboard_result else '❌'}")
        
        if overall_success:
            print("\n🎉 PRODUCTION READY! All API endpoints working correctly.")
        else:
            print("\n⚠️  NOT PRODUCTION READY - Issues detected:")
            
            failed_get = [ep for ep, success in get_results.items() if not success]
            failed_post = [ep for ep, success in post_results.items() if not success]
            
            if failed_get:
                print(f"   ❌ Failed GET endpoints: {failed_get}")
            if failed_post:
                print(f"   ❌ Failed POST endpoints: {failed_post}")
            if not dashboard_result:
                print(f"   ❌ Dashboard loading issues")
        
        return {
            "success": overall_success,
            "get_endpoints": get_results,
            "post_endpoints": post_results,
            "dashboard_load": dashboard_result,
            "summary": {
                "total_endpoints": total_get + total_post + 1,
                "passed_endpoints": passed_get + passed_post + (1 if dashboard_result else 0),
                "pass_rate": f"{((passed_get + passed_post + (1 if dashboard_result else 0)) / (total_get + total_post + 1)) * 100:.1f}%"
            }
        }

if __name__ == "__main__":
    # Run the test suite
    test_suite = APITestSuite()
    results = test_suite.run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)