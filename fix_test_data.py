import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Connect to historical data database
db_path = 'data/historical_data.db'
conn = sqlite3.connect(db_path)

# Clear existing test data
print("Clearing existing test data...")
conn.execute("DELETE FROM historical_data")
conn.commit()

def create_realistic_ohlcv(base_price, num_candles, timeframe):
    """Create realistic OHLCV data"""
    data = []
    current_price = base_price
    
    for i in range(num_candles):
        # Simulate price movement (small random changes)
        price_change = np.random.normal(0, base_price * 0.002)  # 0.2% volatility
        current_price = max(current_price + price_change, base_price * 0.5)  # Don't go below 50% of base
        
        # Create realistic OHLCV
        high = current_price * np.random.uniform(1.0, 1.005)  # Up to 0.5% higher
        low = current_price * np.random.uniform(0.995, 1.0)   # Up to 0.5% lower
        close = np.random.uniform(low, high)
        open_price = data[-1]['close'] if data else current_price
        volume = np.random.uniform(100, 10000)  # Random volume
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        current_price = close
    
    return data

# Test data configuration
symbols = ['BTCUSDT', 'ETHUSDT']
timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
base_prices = {'BTCUSDT': 65000, 'ETHUSDT': 2500}

# Timeframe intervals in minutes
timeframe_minutes = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '1h': 60,
    '4h': 240,
    '1d': 1440
}

# Calculate how much historical data to create for each timeframe
data_periods = {
    '1m': 7,      # 7 days
    '5m': 30,     # 30 days 
    '15m': 90,    # 3 months
    '1h': 365,    # 1 year
    '4h': 730,    # 2 years
    '1d': 1095    # 3 years
}

print("Creating corrected test data...")
total_records = 0

for symbol in symbols:
    for timeframe in timeframes:
        print(f"Creating data for {symbol} {timeframe}...")
        
        # Calculate number of candles needed
        days = data_periods[timeframe]
        minutes_per_candle = timeframe_minutes[timeframe]
        num_candles = int((days * 24 * 60) / minutes_per_candle)
        
        # Create realistic OHLCV data
        ohlcv_data = create_realistic_ohlcv(base_prices[symbol], num_candles, timeframe)
        
        # Calculate timestamps (proper datetime timestamps, not milliseconds)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Create timestamps for each candle
        timestamps = []
        current_time = start_time
        for i in range(num_candles):
            # Use timestamp() to get seconds since epoch, then convert to milliseconds
            timestamps.append(int(current_time.timestamp() * 1000))
            current_time += timedelta(minutes=minutes_per_candle)
        
        # Insert data into database
        insert_data = []
        for i, candle in enumerate(ohlcv_data):
            if i < len(timestamps):  # Make sure we don't exceed timestamps
                insert_data.append((
                    symbol,
                    timeframe,
                    timestamps[i],  # This is now in seconds, not milliseconds
                    candle['open'],
                    candle['high'],
                    candle['low'],
                    candle['close'],
                    candle['volume']
                ))
        
        # Insert in batches
        conn.executemany('''
            INSERT INTO historical_data (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', insert_data)
        
        total_records += len(insert_data)
        print(f"  Inserted {len(insert_data)} records for {symbol} {timeframe}")

# Commit all changes
conn.commit()

# Verify the data
print("\nVerifying corrected data...")
cursor = conn.execute('''
    SELECT symbol, timeframe, COUNT(*) as count,
           datetime(MIN(timestamp), 'unixepoch') as earliest_date,
           datetime(MAX(timestamp), 'unixepoch') as latest_date
    FROM historical_data 
    GROUP BY symbol, timeframe 
    ORDER BY symbol, timeframe
''')

results = cursor.fetchall()
for row in results:
    symbol, timeframe, count, earliest, latest = row
    print(f"{symbol} {timeframe}: {count} candles from {earliest} to {latest}")

conn.close()
print(f"\n✅ Created {total_records} total records with corrected timestamps!")