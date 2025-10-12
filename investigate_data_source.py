#!/usr/bin/env python3
"""
Historical Data Forensics
========================

Investigate the source and timing of historical data records
to understand where the 7,998 records came from.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

def investigate_historical_data():
    """Forensic analysis of historical data records"""
    
    print("🔍 HISTORICAL DATA FORENSICS INVESTIGATION")
    print("=" * 50)
    
    db_path = Path("data/historical_data.db")
    if not db_path.exists():
        print("❌ No historical data database found")
        return
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📊 Available tables: {tables}")
        
        if 'historical_data' in tables:
            # Get basic statistics
            cursor.execute("SELECT COUNT(*) FROM historical_data")
            total_count = cursor.fetchone()[0]
            print(f"📈 Total records: {total_count:,}")
            
            # Get unique symbols
            cursor.execute("SELECT DISTINCT symbol FROM historical_data")
            symbols = [row[0] for row in cursor.fetchall()]
            print(f"💰 Symbols: {symbols}")
            
            # Get unique timeframes
            cursor.execute("SELECT DISTINCT timeframe FROM historical_data")
            timeframes = [row[0] for row in cursor.fetchall()]
            print(f"⏰ Timeframes: {timeframes}")
            
            # Get date range of data
            cursor.execute("""
                SELECT 
                    MIN(timestamp) as earliest_timestamp,
                    MAX(timestamp) as latest_timestamp,
                    MIN(created_at) as earliest_created,
                    MAX(created_at) as latest_created
                FROM historical_data
            """)
            
            result = cursor.fetchone()
            if result and result[0]:
                earliest_ts, latest_ts, earliest_created, latest_created = result
                
                # Convert timestamps
                earliest_date = datetime.fromtimestamp(earliest_ts / 1000) if earliest_ts else None
                latest_date = datetime.fromtimestamp(latest_ts / 1000) if latest_ts else None
                
                print(f"📅 Data time range: {earliest_date} to {latest_date}")
                print(f"🕐 Created range: {earliest_created} to {latest_created}")
                
                # Get records by symbol and timeframe
                cursor.execute("""
                    SELECT symbol, timeframe, COUNT(*) as count, 
                           MIN(timestamp) as start_ts, MAX(timestamp) as end_ts,
                           MIN(created_at) as first_created
                    FROM historical_data 
                    GROUP BY symbol, timeframe 
                    ORDER BY symbol, timeframe
                """)
                
                print("\n📊 BREAKDOWN BY SYMBOL/TIMEFRAME:")
                for row in cursor.fetchall():
                    symbol, tf, count, start_ts, end_ts, first_created = row
                    start_date = datetime.fromtimestamp(start_ts / 1000) if start_ts else None
                    end_date = datetime.fromtimestamp(end_ts / 1000) if end_ts else None
                    
                    print(f"  {symbol} {tf}: {count:,} records")
                    print(f"    └── Data: {start_date} to {end_date}")
                    print(f"    └── Downloaded: {first_created}")
            
            # Check for recent activity (last 7 days)
            cursor.execute("""
                SELECT COUNT(*) FROM historical_data 
                WHERE created_at > datetime('now', '-7 days')
            """)
            recent_count = cursor.fetchone()[0]
            print(f"\n🔥 Recent downloads (last 7 days): {recent_count:,} records")
            
            # Sample some records to see the actual data
            cursor.execute("""
                SELECT symbol, timeframe, timestamp, open, high, low, close, volume, created_at
                FROM historical_data 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            
            print("\n📝 SAMPLE RECORDS (most recent):")
            for row in cursor.fetchall():
                symbol, tf, ts, o, h, l, c, v, created = row
                dt = datetime.fromtimestamp(ts / 1000) if ts else None
                print(f"  {symbol} {tf} | {dt} | OHLC: {o:.2f}/{h:.2f}/{l:.2f}/{c:.2f} | Created: {created}")
        
        conn.close()
        
        print("\n🎯 INVESTIGATION SUMMARY:")
        print("This analysis shows exactly when and how the historical data was downloaded.")
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")

if __name__ == "__main__":
    investigate_historical_data()