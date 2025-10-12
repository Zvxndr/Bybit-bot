#!/usr/bin/env python3
import sqlite3
import pandas as pd
from datetime import datetime

# Check the actual historical data
db_path = "data/trading_bot.db"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check total count for BTCUSDT 15m
    cursor.execute('SELECT COUNT(*) FROM historical_data WHERE symbol = ? AND timeframe = ?', ('BTCUSDT', '15m'))
    count = cursor.fetchone()[0]
    print(f"📊 Total BTCUSDT 15m records: {count}")
    
    # Check date range
    cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM historical_data WHERE symbol = ? AND timeframe = ?', ('BTCUSDT', '15m'))
    result = cursor.fetchone()
    print(f"📅 Date range: {result[0]} to {result[1]}")
    
    # Check if data looks real or fake
    cursor.execute('SELECT timestamp, open, high, low, close, volume FROM historical_data WHERE symbol = ? AND timeframe = ? ORDER BY timestamp DESC LIMIT 10', ('BTCUSDT', '15m'))
    recent_data = cursor.fetchall()
    
    print(f"\n🔍 Recent data sample (last 10 records):")
    for i, row in enumerate(recent_data):
        timestamp, open_price, high, low, close, volume = row
        print(f"  {i+1}. {timestamp} | O:{open_price:.2f} H:{high:.2f} L:{low:.2f} C:{close:.2f} V:{volume:.2f}")
    
    # Check for suspicious patterns indicating fake data
    print(f"\n🕵️ Data quality analysis:")
    
    # Check if all prices are too similar (sign of generated data)
    cursor.execute('SELECT AVG(close), MIN(close), MAX(close), COUNT(*) FROM historical_data WHERE symbol = ? AND timeframe = ?', ('BTCUSDT', '15m'))
    avg_price, min_price, max_price, total_count = cursor.fetchone()
    
    price_range = max_price - min_price
    price_variation = (price_range / avg_price) * 100
    
    print(f"  Price range: ${min_price:.2f} - ${max_price:.2f}")
    print(f"  Average price: ${avg_price:.2f}")
    print(f"  Price variation: {price_variation:.2f}%")
    
    if price_variation < 1.0:
        print("  ⚠️  SUSPICIOUS: Very low price variation suggests generated data")
    elif price_variation > 50.0:
        print("  ✅ REALISTIC: Good price variation suggests real market data")
    else:
        print("  🤔 UNCERTAIN: Moderate variation, could be real or generated")
        
    # Check timestamp intervals
    cursor.execute('SELECT timestamp FROM historical_data WHERE symbol = ? AND timeframe = ? ORDER BY timestamp DESC LIMIT 5', ('BTCUSDT', '15m'))
    timestamps = [row[0] for row in cursor.fetchall()]
    
    print(f"\n⏰ Timestamp intervals:")
    for i in range(1, len(timestamps)):
        try:
            t1 = datetime.fromisoformat(timestamps[i-1].replace('Z', '+00:00'))
            t2 = datetime.fromisoformat(timestamps[i].replace('Z', '+00:00'))
            diff_minutes = (t1 - t2).total_seconds() / 60
            print(f"  Gap {i}: {diff_minutes:.1f} minutes")
        except:
            print(f"  Gap {i}: Unable to parse timestamps")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error checking data: {e}")
    import traceback
    traceback.print_exc()