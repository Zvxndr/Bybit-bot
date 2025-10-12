#!/usr/bin/env python3
"""
🧪 Quick API Test - Verify Historical Data Integration Fix
"""

import requests
import json

def test_api():
    """Test the API endpoints"""
    print("🧪 Testing Historical Data Integration API")
    print("=" * 45)
    
    base_url = "http://localhost:5000"
    
    # Test data discovery endpoint
    print("🔍 Testing Data Discovery API...")
    try:
        response = requests.get(f"{base_url}/api/historical-data/discover", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS: Found {data.get('total_datasets', 0)} datasets")
            
            for dataset in data.get('datasets', [])[:3]:
                print(f"   📊 {dataset['symbol']} {dataset['timeframe']}: {dataset['record_count']:,} records")
                print(f"      📅 Range: {dataset['date_range']}")
        else:
            print(f"❌ FAILED: Status {response.status_code}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    # Test backtest history endpoint
    print("\n📊 Testing Backtest History API...")
    try:
        response = requests.get(f"{base_url}/api/backtest/history", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS: Found {data.get('total_results', 0)} backtest results")
            
            for result in data.get('results', [])[:2]:
                print(f"   📈 {result['pair']} {result['timeframe']}: {result['total_return']:.1f}% return")
                print(f"      🎯 Win Rate: {result['win_rate']:.1f}%, Trades: {result['trades_count']}")
        else:
            print(f"❌ FAILED: Status {response.status_code}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("\n🎯 INTEGRATION TEST COMPLETE!")
    print("If both endpoints returned data, your fix is working! 🎉")

if __name__ == "__main__":
    test_api()