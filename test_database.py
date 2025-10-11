#!/usr/bin/env python3
"""Test script to check database contents"""

import sqlite3
import json
from datetime import datetime

# Check database contents
db_path = 'data/historical_data.db'
print(f'Checking database: {db_path}')

try:
    with sqlite3.connect(db_path) as conn:
        # Check table structure
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f'Tables found: {tables}')
        
        # Check if historical_data table exists and has data
        cursor = conn.execute("""
            SELECT symbol, timeframe, COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM historical_data 
            GROUP BY symbol, timeframe
        """)
        
        results = cursor.fetchall()
        print(f'Data summary: {len(results)} symbol-timeframe combinations')
        for row in results:
            symbol, timeframe, count, min_ts, max_ts = row
            min_date = datetime.fromtimestamp(min_ts / 1000) if min_ts else None
            max_date = datetime.fromtimestamp(max_ts / 1000) if max_ts else None
            print(f'  {symbol} {timeframe}: {count} candles, from {min_date} to {max_date}')
            
        # Test specific query for BTCUSDT 15m
        cursor = conn.execute("""
            SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest, COUNT(*) as count
            FROM historical_data 
            WHERE symbol = ? AND timeframe = ?
        """, ('BTCUSDT', '15m'))
        
        result = cursor.fetchone()
        print(f'BTCUSDT 15m query result: {result}')
        
        # Now test the get_available_periods method
        print("\n--- Testing get_available_periods method ---")
        import sys
        sys.path.append('.')
        from historical_data_downloader import HistoricalDataDownloader
        
        downloader = HistoricalDataDownloader()
        periods_result = downloader.get_available_periods('BTCUSDT', '15m')
        print(f'Periods result: {json.dumps(periods_result, indent=2)}')
        
except Exception as e:
    print(f'Database error: {e}')
    import traceback
    traceback.print_exc()