#!/usr/bin/env python3
import sqlite3
from datetime import datetime, timedelta
import random

# Create some test historical data
db_path = "data/historical_data.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create test data for BTCUSDT and ETHUSDT with different timeframes
symbols = ['BTCUSDT', 'ETHUSDT']
timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

base_timestamp = int(datetime.now().timestamp())
base_price = 50000  # Starting price for BTC

print("Creating test historical data...")

for symbol in symbols:
    symbol_base_price = base_price if symbol == 'BTCUSDT' else base_price * 0.06  # ETH ~$3000
    
    for timeframe in timeframes:
        # Different periods for different timeframes
        if timeframe == '1m':
            days_back = 7  # 1 week of 1m data
            interval_seconds = 60
        elif timeframe == '5m':
            days_back = 30  # 1 month of 5m data
            interval_seconds = 300
        elif timeframe == '15m':
            days_back = 90  # 3 months of 15m data
            interval_seconds = 900
        elif timeframe == '1h':
            days_back = 365  # 1 year of 1h data
            interval_seconds = 3600
        elif timeframe == '4h':
            days_back = 730  # 2 years of 4h data
            interval_seconds = 14400
        else:  # 1d
            days_back = 1095  # 3 years of daily data
            interval_seconds = 86400
        
        start_time = base_timestamp - (days_back * 86400)
        current_price = symbol_base_price
        
        records_count = 0
        for timestamp in range(start_time, base_timestamp, interval_seconds):
            # Generate realistic OHLCV data
            price_change = random.uniform(-0.02, 0.02)  # ±2% price change
            current_price = current_price * (1 + price_change)
            
            volatility = random.uniform(0.005, 0.015)  # 0.5% to 1.5% volatility
            high = current_price * (1 + volatility)
            low = current_price * (1 - volatility)
            
            open_price = random.uniform(low, high)
            close_price = current_price
            
            volume = random.uniform(1000, 10000)
            
            cursor.execute("""
                INSERT INTO historical_data 
                (symbol, timeframe, timestamp, open, high, low, close, volume, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, timeframe, timestamp, 
                open_price, high, low, close_price, volume,
                datetime.now()
            ))
            records_count += 1
        
        print(f"  {symbol} {timeframe}: {records_count} records ({days_back} days)")

conn.commit()
conn.close()
print("✅ Test historical data created successfully!")

# Now check what we have
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
    SELECT symbol, timeframe, 
           MIN(datetime(timestamp, 'unixepoch')) as start_date,
           MAX(datetime(timestamp, 'unixepoch')) as end_date,
           COUNT(*) as records
    FROM historical_data 
    GROUP BY symbol, timeframe
    ORDER BY symbol, timeframe
""")

data = cursor.fetchall()
print("\n📊 Available data periods:")
for row in data:
    print(f"  {row[0]} {row[1]}: {row[2]} to {row[3]} ({row[4]} records)")

conn.close()