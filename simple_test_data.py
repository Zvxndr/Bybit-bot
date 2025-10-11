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

def create_realistic_ohlcv(base_price, num_candles):
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

# Simple test data - just one symbol and timeframe to start
symbol = 'BTCUSDT'
timeframe = '15m'
base_price = 65000

print(f"Creating test data for {symbol} {timeframe}...")

# Create 3 months of 15m data
days = 90
minutes_per_candle = 15
num_candles = int((days * 24 * 60) / minutes_per_candle)
print(f"Will create {num_candles} candles for {days} days")

# Create realistic OHLCV data
ohlcv_data = create_realistic_ohlcv(base_price, num_candles)

# Calculate timestamps
end_time = datetime.now()
start_time = end_time - timedelta(days=days)

# Create timestamps for each candle
timestamps = []
current_time = start_time
for i in range(num_candles):
    timestamps.append(int(current_time.timestamp() * 1000))
    current_time += timedelta(minutes=minutes_per_candle)

print(f"Created {len(timestamps)} timestamps from {datetime.fromtimestamp(timestamps[0]/1000)} to {datetime.fromtimestamp(timestamps[-1]/1000)}")

# Prepare data for insertion
insert_data = []
for i in range(min(len(ohlcv_data), len(timestamps))):
    candle = ohlcv_data[i]
    insert_data.append((
        symbol,
        timeframe,
        timestamps[i],
        candle['open'],
        candle['high'],
        candle['low'],
        candle['close'],
        candle['volume']
    ))

print(f"Inserting {len(insert_data)} records...")

# Insert data one by one to identify any problematic records
for i, record in enumerate(insert_data):
    try:
        conn.execute('''
            INSERT INTO historical_data (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', record)
        
        if i % 1000 == 0:
            print(f"  Inserted {i} records...")
            
    except Exception as e:
        print(f"Error inserting record {i}: {record}")
        print(f"Error: {e}")
        break

# Commit all changes
conn.commit()

# Verify the data
print("\nVerifying data...")
cursor = conn.execute('''
    SELECT symbol, timeframe, COUNT(*) as count,
           datetime(MIN(timestamp)/1000, 'unixepoch') as earliest_date,
           datetime(MAX(timestamp)/1000, 'unixepoch') as latest_date
    FROM historical_data 
    GROUP BY symbol, timeframe 
    ORDER BY symbol, timeframe
''')

results = cursor.fetchall()
for row in results:
    symbol, timeframe, count, earliest, latest = row
    print(f"{symbol} {timeframe}: {count} candles from {earliest} to {latest}")

conn.close()
print(f"\n✅ Successfully created test data!")