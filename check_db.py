#!/usr/bin/env python3
import sqlite3
import os

# Check historical data database
db_path = "data/historical_data.db"
print(f"DB exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get schema
    cursor.execute("PRAGMA table_info(historical_data)")
    schema = cursor.fetchall()
    print("\nHistorical data schema:")
    for col in schema:
        print(f"  {col[1]} ({col[2]})")
    
    # Get count
    cursor.execute("SELECT COUNT(*) FROM historical_data")
    count = cursor.fetchone()[0]
    print(f"\nTotal records: {count}")
    
    if count > 0:
        # Get summary
        cursor.execute("""
            SELECT symbol, timeframe, 
                   MIN(timestamp), MAX(timestamp), COUNT(*) 
            FROM historical_data 
            GROUP BY symbol, timeframe
        """)
        data = cursor.fetchall()
        print("\nData summary:")
        for row in data:
            print(f"  {row[0]} {row[1]}: {row[2]} to {row[3]} ({row[4]} records)")
    
    conn.close()

# Also check main trading database
trading_db = "data/trading_bot.db"
print(f"\nTrading DB exists: {os.path.exists(trading_db)}")

if os.path.exists(trading_db):
    conn = sqlite3.connect(trading_db)
    cursor = conn.cursor()
    
    # Check if historical data might be in the main db
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Trading DB tables:", [t[0] for t in tables])
    
    conn.close()