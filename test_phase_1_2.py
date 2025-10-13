#!/usr/bin/env python3
"""
Phase 1 & 2 API Testing Script
==============================

Comprehensive testing of all new Phase 1 & 2 endpoints to ensure
100% functionality before DigitalOcean deployment.
"""

import requests
import json
import sys
from datetime import datetime

BASE_URL = "http://localhost:8080"

def test_endpoint(endpoint, description):
    """Test a single API endpoint"""
    print(f"\n🧪 Testing: {description}")
    print(f"   Endpoint: {endpoint}")
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {response.status_code}")
            print(f"   ✅ Success: {data.get('success', 'N/A')}")
            
            # Show key metrics from response
            if 'strategies' in data:
                print(f"   📊 Strategies: {len(data.get('strategies', []))}")
            if 'sentiment_score' in data:
                print(f"   📈 Sentiment: {data.get('sentiment_score')} ({data.get('sentiment_label')})")
            if 'exchanges' in data:
                exchanges = data.get('exchanges', {})
                connected = sum(1 for ex in exchanges.values() if ex.get('status') == 'connected')
                print(f"   🔗 Exchanges: {connected}/{len(exchanges)} connected")
            if 'datasets' in data:
                print(f"   💾 Datasets: {len(data.get('datasets', []))}")
            if 'headlines' in data:
                print(f"   📰 Headlines: {len(data.get('headlines', []))}")
            
            return True
        else:
            print(f"   ❌ Status: {response.status_code}")
            print(f"   ❌ Error: {response.text[:100]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def test_post_endpoint(endpoint, data, description):
    """Test a POST endpoint"""
    print(f"\n🧪 Testing: {description}")
    print(f"   Endpoint: {endpoint}")
    
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=10)
        
        if response.status_code == 200:
            resp_data = response.json()
            print(f"   ✅ Status: {response.status_code}")
            print(f"   ✅ Success: {resp_data.get('success', 'N/A')}")
            return True
        else:
            print(f"   ❌ Status: {response.status_code}")
            print(f"   ❌ Error: {response.text[:100]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def main():
    """Run comprehensive Phase 1 & 2 API tests"""
    print("🚀 PHASE 1 & 2 API TESTING")
    print("=" * 50)
    print(f"Base URL: {BASE_URL}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Core Strategy APIs
    results.append(test_endpoint("/api/strategies/ranking?period=all", "Strategy Rankings (All Time)"))
    results.append(test_endpoint("/api/strategies/ranking?period=month", "Strategy Rankings (Month)"))
    results.append(test_endpoint("/api/ml/status", "ML Algorithm Status"))
    
    # Historical Data Management
    results.append(test_endpoint("/api/historical-data/discover", "Historical Data Discovery"))
    
    # Multi-Exchange Integration
    results.append(test_endpoint("/api/status/apis", "Exchange API Status"))
    results.append(test_endpoint("/api/correlation/btc", "BTC Cross-Exchange Correlation"))
    results.append(test_endpoint("/api/correlation/matrix", "Correlation Matrix"))
    
    # News Sentiment Analysis
    results.append(test_endpoint("/api/news/sentiment", "Market Sentiment Analysis"))
    results.append(test_endpoint("/api/news/headlines", "News Headlines with Sentiment"))
    
    # Email Reporting System
    results.append(test_endpoint("/api/email/status", "Email System Status"))
    results.append(test_post_endpoint("/api/email/test", {}, "Send Test Email"))
    
    # Future Markets Framework
    results.append(test_endpoint("/api/markets/available", "Available Markets"))
    
    # Advanced Backtesting
    backtest_config = {
        "minimum_requirements": {
            "min_sharpe_ratio": 2.0,
            "min_win_rate": 65.0,
            "min_return_pct": 50.0,
            "max_drawdown": 10.0
        },
        "retirement_metrics": {
            "performance_threshold": "below_25th_percentile_30_days",
            "drawdown_limit": 15.0,
            "consecutive_loss_limit": 10,
            "age_limit_days": 180
        }
    }
    results.append(test_post_endpoint("/api/backtest/run", backtest_config, "ML-Driven Backtest"))
    
    # System Health
    results.append(test_endpoint("/health", "System Health Check"))
    results.append(test_endpoint("/", "Main Dashboard"))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    pass_rate = (passed / total) * 100
    
    print(f"✅ Passed: {passed}/{total} ({pass_rate:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 1 & 2 ready for DigitalOcean deployment!")
    elif pass_rate >= 80:
        print("\n⚠️  Most tests passed. Minor issues may exist but system is functional.")
    else:
        print("\n❌ Multiple test failures. System needs debugging before deployment.")
    
    print(f"\n🔗 Dashboard URL: {BASE_URL}")
    print(f"📚 API Docs: {BASE_URL}/docs")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)