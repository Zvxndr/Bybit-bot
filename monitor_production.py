#!/usr/bin/env python3
"""
Production Monitoring Script for Bybit Trading Bot
Monitor the fixes deployed for historical data and strategy discovery
"""

import requests
import json
import time
from datetime import datetime

def monitor_production_issues():
    """Monitor the production deployment for the fixes"""
    
    base_url = "https://auto-wealth-j58sx.ondigitalocean.app"
    
    print("🔍 Production Issue Monitoring Script")
    print("=" * 50)
    print(f"Monitoring: {base_url}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test 1: Check if historical data download works (enhanced testing)
    print("📊 Testing Historical Data Download...")
    
    test_cases = [
        {"symbol": "BTCUSDT", "timeframe": "1h", "days": 1, "desc": "BTCUSDT 1 day"},
        {"symbol": "BTC/USDT", "timeframe": "15m", "days": 7, "desc": "BTC/USDT 7 days"},
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"   Test {i}: {test_case['desc']}")
        try:
            response = requests.post(f"{base_url}/api/historical-data/download", 
                                   json={k: v for k, v in test_case.items() if k != 'desc'}, 
                                   timeout=45)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get('data_points', 0) > 0:
                    print(f"      ✅ Success: {data['data_points']} data points")
                    success_count += 1
                else:
                    print(f"      ❌ Failed: {data.get('message', 'No data retrieved')}")
            else:
                print(f"      ❌ HTTP Error {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Connection error: {e}")
    
    if success_count > 0:
        print("✅ Historical data download working")
    else:
        print("❌ Historical data download failed for all test cases")
    
    print()
    
    # Test 2: Check strategy discovery (should be empty now)
    print("🤖 Testing Strategy Discovery...")
    try:
        response = requests.get(f"{base_url}/api/strategies", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            discovery_count = len(data.get("discovery", []))
            paper_count = len(data.get("paper", []))
            live_count = len(data.get("live", []))
            
            total_strategies = discovery_count + paper_count + live_count
            
            if total_strategies == 0:
                print("✅ Strategy discovery fixed - no fake strategies showing")
                print("   Strategies will only appear after backtesting/ML discovery")
            else:
                print(f"⚠️ Found {total_strategies} strategies")
                print(f"   Discovery: {discovery_count}, Paper: {paper_count}, Live: {live_count}")
                print("   Check if these are legitimate or still sample data")
                
        else:
            print(f"❌ HTTP Error {response.status_code}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    print()
    
    # Test 3: Check ML Risk Manager
    print("🛡️ Testing ML Risk Manager...")
    try:
        response = requests.get(f"{base_url}/api/ml-risk-metrics", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "Unknown")
            
            if "ML Risk Engine Active" in status or "ML Engine" in status:
                print("✅ ML Risk Manager working")
                print(f"   Status: {status}")
            else:
                print("⚠️ ML Risk Manager may have issues")
                print(f"   Status: {status}")
                
        else:
            print(f"❌ HTTP Error {response.status_code}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    print()
    
    # Test 4: Check system status
    print("🔧 Testing System Status...")
    try:
        response = requests.get(f"{base_url}/api/system-status", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            overall_status = data.get("overall_status", "Unknown")
            integration_complete = data.get("integration_complete", False)
            
            print(f"   Overall Status: {overall_status}")
            print(f"   Integration Complete: {integration_complete}")
            
            components = data.get("components", {})
            for component, info in components.items():
                if isinstance(info, dict):
                    status = info.get("status", "Unknown")
                    print(f"   {component}: {status}")
                    
        else:
            print(f"❌ HTTP Error {response.status_code}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    print()
    print("🎯 Monitoring complete!")
    print()
    print("📋 Expected Results After Fix:")
    print("- Historical data download should return success with data_points > 0")
    print("- Strategy discovery should show 0 strategies (until you run backtests)")
    print("- ML Risk Manager should show 'Active' status")
    print("- System status should show integration_complete: true")
    print()
    print("🔧 If issues persist:")
    print("1. Check production logs for detailed error messages")
    print("2. Verify all environment variables are set correctly")
    print("3. Ensure database tables were created properly")

if __name__ == "__main__":
    monitor_production_issues()