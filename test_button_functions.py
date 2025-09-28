#!/usr/bin/env python3
"""
Button Function Tester
=====================

This script tests all the button functions that were reported as failing.
It validates data wipe, pause/resume, and emergency stop functionality.
"""

import sys
import os
import time
import requests
import json
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

def test_api_endpoint(url, method="GET", data=None, timeout=10):
    """Test an API endpoint and return results"""
    try:
        if method.upper() == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
        
        result = {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        }
        
        return result
        
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "response": None
        }

def main():
    """Run comprehensive button function tests"""
    print("🔧 Starting Button Function Tests")
    print("=" * 50)
    
    # Server configuration
    BASE_URL = "http://localhost:8080"
    
    # Test endpoints
    tests = [
        {
            "name": "System Health Check",
            "method": "GET",
            "endpoint": "/health",
            "expected": "Should return system status"
        },
        {
            "name": "System Stats",
            "method": "GET", 
            "endpoint": "/api/system-stats",
            "expected": "Should return system statistics"
        },
        {
            "name": "Bot Pause",
            "method": "POST",
            "endpoint": "/api/bot/pause",
            "expected": "Should pause the trading bot"
        },
        {
            "name": "Bot Resume", 
            "method": "POST",
            "endpoint": "/api/bot/resume",
            "expected": "Should resume the trading bot"
        },
        {
            "name": "Emergency Stop",
            "method": "POST",
            "endpoint": "/api/bot/emergency-stop",
            "expected": "Should trigger emergency stop"
        },
        {
            "name": "Close All Positions",
            "method": "POST", 
            "endpoint": "/api/admin/close-all-positions",
            "expected": "Should close all trading positions"
        },
        {
            "name": "Cancel All Orders",
            "method": "POST",
            "endpoint": "/api/admin/cancel-all-orders", 
            "expected": "Should cancel all pending orders"
        },
        {
            "name": "Data Wipe (CRITICAL TEST)",
            "method": "POST",
            "endpoint": "/api/admin/wipe-data",
            "expected": "Should clear all trading data and files"
        }
    ]
    
    results = []
    
    for i, test in enumerate(tests, 1):
        print(f"\n{i}. Testing: {test['name']}")
        print(f"   Method: {test['method']} {BASE_URL}{test['endpoint']}")
        print(f"   Expected: {test['expected']}")
        
        # Execute test
        url = f"{BASE_URL}{test['endpoint']}"
        result = test_api_endpoint(url, test['method'])
        
        # Display results
        if result['success']:
            print(f"   ✅ SUCCESS (Status: {result['status_code']})")
            if isinstance(result['response'], dict):
                if 'message' in result['response']:
                    print(f"   📝 Message: {result['response']['message']}")
                if 'success' in result['response']:
                    print(f"   🎯 API Success: {result['response']['success']}")
            else:
                print(f"   📄 Response: {str(result['response'])[:100]}...")
        else:
            print(f"   ❌ FAILED (Status: {result.get('status_code', 'N/A')})")
            if 'error' in result:
                print(f"   🔥 Error: {result['error']}")
        
        results.append({
            "test": test['name'],
            "success": result['success'],
            "details": result
        })
        
        # Small delay between tests
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    print("\n🔍 DETAILED RESULTS:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {status} - {result['test']}")
    
    # Special check for critical data wipe function
    data_wipe_result = next((r for r in results if "Data Wipe" in r['test']), None)
    if data_wipe_result:
        print(f"\n🔥 CRITICAL: Data Wipe Test Result")
        if data_wipe_result['success']:
            print("   ✅ Data wipe endpoint is accessible")
            response = data_wipe_result['details'].get('response', {})
            if isinstance(response, dict) and response.get('success'):
                print("   ✅ Data wipe executed successfully")
            else:
                print("   ⚠️ Data wipe endpoint responded but may not have executed properly")
        else:
            print("   ❌ Data wipe endpoint failed - This explains the button failure!")
    
    # Check for common issues
    print(f"\n🛠️ TROUBLESHOOTING:")
    failed_tests = [r for r in results if not r['success']]
    
    if not failed_tests:
        print("   🎉 All tests passed! Button functions should work correctly.")
    else:
        print(f"   🔧 {len(failed_tests)} test(s) failed:")
        for failed in failed_tests:
            error = failed['details'].get('error', 'Unknown error')
            print(f"     - {failed['test']}: {error}")
        
        if any("Connection" in str(f['details'].get('error', '')) for f in failed_tests):
            print("\n   💡 SUGGESTION: Make sure the bot is running on localhost:8080")
            print("      Try running: python src/main.py")
        
        if any("Data Wipe" in f['test'] for f in failed_tests):
            print("\n   🔥 DATA WIPE ISSUE DETECTED:")
            print("      The data wipe button failure is confirmed.")
            print("      Check the frontend server logs for POST /api/admin/wipe-data requests.")

if __name__ == "__main__":
    main()