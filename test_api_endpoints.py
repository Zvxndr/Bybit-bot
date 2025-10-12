#!/usr/bin/env python3
"""Test API endpoints to verify the fix"""
import requests
import json

def test_api_endpoints():
    try:
        # Test data discovery endpoint
        response = requests.get('http://localhost:8080/api/historical-data/discover', timeout=5)
        print('📊 Data Discovery API:')
        data = response.json()
        print(f'  Success: {data.get("success", False)}')
        print(f'  Datasets: {len(data.get("datasets", []))}')
        if data.get('datasets'):
            for ds in data['datasets'][:2]:
                print(f'    {ds.get("symbol")} {ds.get("timeframe")}: {ds.get("record_count")} records')
        
        # Test backtest history endpoint
        response2 = requests.get('http://localhost:8080/api/backtest/history', timeout=5)
        print('\n🔍 Backtest History API:')
        data2 = response2.json()
        print(f'  Success: {data2.get("success", False)}')
        print(f'  Results: {len(data2.get("data", []))}')
        if data2.get('data'):
            for result in data2['data'][:1]:
                print(f'    {result.get("pair")} {result.get("timeframe")}: {result.get("total_return", 0):.1f}% return')
        
        print('\n✅ API endpoints are working correctly!')
        return True
        
    except Exception as e:
        print(f'❌ API test failed: {e}')
        print('💡 Start the dashboard server first: python src/main.py')
        return False

if __name__ == "__main__":
    test_api_endpoints()