#!/usr/bin/env python3
"""
PRODUCTION DATA DISCOVERY DEBUG TOOL

This script diagnoses exactly why historical data isn't showing up 
in the backtesting controls after successful download.
"""

import sqlite3
import os
from pathlib import Path
import requests
import json

def check_database_directly():
    """Check the database files directly"""
    print("🔍 DIRECT DATABASE INSPECTION")
    print("=" * 50)
    
    # Check all possible database locations  
    database_paths = [
        "data/trading_bot.db",
        "data/historical_data.db", 
        "/app/data/trading_bot.db",
        "src/data/speed_demon_cache/market_data.db",
        "/app/src/data/speed_demon_cache/market_data.db"
    ]
    
    for db_path in database_paths:
        path_obj = Path(db_path)
        print(f"\n📁 Checking: {db_path}")
        
        if path_obj.exists():
            size = path_obj.stat().st_size
            print(f"   ✅ EXISTS - Size: {size:,} bytes")
            
            try:
                conn = sqlite3.connect(str(path_obj))
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                print(f"   📋 Tables: {tables}")
                
                # Check for historical data specifically
                if 'historical_data' in tables:
                    cursor.execute("SELECT COUNT(*) FROM historical_data")
                    count = cursor.fetchone()[0]
                    print(f"   📊 historical_data records: {count:,}")
                    
                    if count > 0:
                        # Get sample data
                        cursor.execute("""
                            SELECT symbol, timeframe, COUNT(*) as cnt
                            FROM historical_data 
                            GROUP BY symbol, timeframe 
                            ORDER BY cnt DESC
                        """)
                        samples = cursor.fetchall()
                        print(f"   📈 Data breakdown:")
                        for symbol, timeframe, cnt in samples:
                            print(f"      - {symbol} {timeframe}: {cnt:,} records")
                
                # Check other potential data tables
                for table in tables:
                    if any(keyword in table.lower() for keyword in ['market', 'data', 'cache']):
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        if count > 0:
                            print(f"   📊 {table}: {count:,} records")
                
                conn.close()
                
            except Exception as e:
                print(f"   ❌ ERROR reading database: {e}")
        else:
            print(f"   ❌ NOT FOUND")

def test_api_endpoint():
    """Test the API endpoint discovery"""
    print("\n\n🌐 API ENDPOINT TESTING")
    print("=" * 50)
    
    # Test production URL
    base_url = "https://auto-wealth-j58sx.ondigitalocean.app"
    endpoint = f"{base_url}/api/historical-data/discover"
    
    print(f"📡 Testing: {endpoint}")
    
    try:
        response = requests.get(endpoint, timeout=10)
        print(f"   📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Response received")
            print(f"   📋 Success: {data.get('success', False)}")
            print(f"   📊 Datasets: {data.get('total_datasets', 0)}")
            print(f"   💾 Database Path: {data.get('database_path', 'Unknown')}")
            
            if data.get('datasets'):
                print(f"   📈 Found Data:")
                for dataset in data['datasets']:
                    print(f"      - {dataset.get('symbol')} {dataset.get('timeframe')}: {dataset.get('record_count')} records")
            else:
                print(f"   ❌ No datasets found")
                if 'available_tables' in data:
                    print(f"   📋 Available tables: {data['available_tables']}")
                if 'expected_tables' in data:
                    print(f"   🎯 Expected tables: {data['expected_tables']}")
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
            print(f"   📄 Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"   ❌ Connection Error: {e}")

def test_download_endpoint():
    """Test if download endpoint is working"""
    print("\n\n📥 DOWNLOAD ENDPOINT TESTING")  
    print("=" * 50)
    
    base_url = "https://auto-wealth-j58sx.ondigitalocean.app"
    endpoint = f"{base_url}/api/historical-data/status"
    
    print(f"📡 Testing: {endpoint}")
    
    try:
        response = requests.get(endpoint, timeout=10)
        print(f"   📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Download Status Available")
            print(f"   📊 Total Datasets: {data.get('total_datasets', 0)}")
            print(f"   📈 Total Data Points: {data.get('total_data_points', 0)}")
            
            if data.get('data'):
                print(f"   📋 Downloaded Data Summary:")
                for item in data['data'][:5]:  # Show first 5
                    print(f"      - {item.get('symbol')} {item.get('timeframe')}: {item.get('candle_count', 0)} candles")
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Connection Error: {e}")

def main():
    """Run comprehensive data discovery diagnosis"""
    print("🚨 PRODUCTION DATA DISCOVERY DIAGNOSTIC")
    print("=" * 60)
    print("Diagnosing why 2,877 downloaded data points aren't showing in backtesting")
    print()
    
    # Step 1: Check databases directly
    check_database_directly()
    
    # Step 2: Test API endpoints
    test_api_endpoint()
    
    # Step 3: Test download status
    test_download_endpoint()
    
    print("\n\n🎯 DIAGNOSIS COMPLETE")
    print("=" * 60)
    print("If data exists in database but API shows empty:")
    print("  - Database path mismatch between download and discovery")  
    print("  - Table name mismatch")
    print("  - Database connection issues")
    print("\nIf no data in database:")
    print("  - Download didn't actually save to database")
    print("  - Wrong database file being checked")
    print("  - Permission issues")

if __name__ == "__main__":
    main()