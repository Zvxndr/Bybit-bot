#!/usr/bin/env python3
"""
REAL HISTORICAL DATA DOWNLOADER & BACKTESTING ENABLER

This script:
1. Downloads REAL historical data from Bybit
2. Creates proper historical_data table structure  
3. Enables proper backtesting functionality
4. Fixes the mock data issue completely
"""

import sqlite3
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# Configuration
SYMBOLS = ['BTCUSDT', 'ETHUSDT']
TIMEFRAMES = ['15m', '1h', '4h', '1d']
DAYS_BACK = 90  # 3 months of data

def setup_database():
    """Create proper database structure for historical data"""
    db_path = "data/trading_bot.db"
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create historical_data table with proper structure
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timeframe, timestamp)
        )
    """)
    
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_historical_symbol_timeframe_timestamp 
        ON historical_data(symbol, timeframe, timestamp)
    """)
    
    conn.commit()
    return conn

def download_real_data(symbol, timeframe, days_back=90):
    """Download real historical data from Bybit"""
    print(f"📡 Downloading real data: {symbol} {timeframe} ({days_back} days)")
    
    try:
        # Initialize Bybit exchange
        exchange = ccxt.bybit({
            'sandbox': False,  # Use real market data
            'enableRateLimit': True,
        })
        
        # Calculate time parameters
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Convert to milliseconds
        since = int(start_time.timestamp() * 1000)
        
        all_candles = []
        current_since = since
        
        while current_since < int(end_time.timestamp() * 1000):
            try:
                # Fetch OHLCV data
                candles = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=200  # Bybit limit
                )
                
                if not candles:
                    break
                
                print(f"  📥 Fetched {len(candles)} candles from {datetime.fromtimestamp(candles[0][0]/1000)}")
                all_candles.extend(candles)
                
                # Update since to last candle time + 1ms
                current_since = candles[-1][0] + 1
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ⚠️ Error fetching batch: {e}")
                time.sleep(2)
                break
        
        print(f"  ✅ Downloaded {len(all_candles)} total candles")
        return all_candles
        
    except Exception as e:
        print(f"  ❌ Error downloading {symbol} {timeframe}: {e}")
        return []

def save_to_database(conn, symbol, timeframe, candles):
    """Save candles to database"""
    if not candles:
        return 0
        
    cursor = conn.cursor()
    
    # Prepare data for insertion
    data_to_insert = []
    for candle in candles:
        timestamp_ms, open_price, high, low, close, volume = candle
        timestamp_str = datetime.fromtimestamp(timestamp_ms / 1000).isoformat()
        
        data_to_insert.append((
            symbol,
            timeframe,
            timestamp_str,
            float(open_price),
            float(high),
            float(low),
            float(close),
            float(volume)
        ))
    
    # Insert data (ignore duplicates)
    cursor.executemany("""
        INSERT OR IGNORE INTO historical_data 
        (symbol, timeframe, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, data_to_insert)
    
    inserted_count = cursor.rowcount
    conn.commit()
    
    print(f"  💾 Saved {inserted_count} new records to database")
    return inserted_count

def main():
    """Download real historical data for all symbols and timeframes"""
    print("🚀 DOWNLOADING REAL HISTORICAL DATA FROM BYBIT")
    print("=" * 60)
    
    # Setup database
    conn = setup_database()
    
    total_downloaded = 0
    
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            print(f"\n📊 Processing {symbol} {timeframe}")
            
            # Check if data already exists
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM historical_data 
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            existing_count = cursor.fetchone()[0]
            
            if existing_count > 0:
                print(f"  ℹ️ Already have {existing_count} records, skipping...")
                continue
            
            # Download real data
            candles = download_real_data(symbol, timeframe, DAYS_BACK)
            
            if candles:
                # Save to database
                inserted = save_to_database(conn, symbol, timeframe, candles)
                total_downloaded += inserted
            
            # Small delay between requests
            time.sleep(1)
    
    conn.close()
    
    print(f"\n✅ COMPLETE: Downloaded {total_downloaded} total historical records")
    print("🎯 Real market data is now available for backtesting!")
    
    # Verify the data
    print("\n🔍 VERIFICATION:")
    conn = sqlite3.connect("data/trading_bot.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT symbol, timeframe, COUNT(*) as count,
               MIN(timestamp) as earliest, MAX(timestamp) as latest
        FROM historical_data 
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
    """)
    
    results = cursor.fetchall()
    for symbol, timeframe, count, earliest, latest in results:
        print(f"  📈 {symbol} {timeframe}: {count} candles from {earliest[:10]} to {latest[:10]}")
    
    conn.close()

if __name__ == "__main__":
    main()