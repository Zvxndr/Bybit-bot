#!/usr/bin/env python3
import sqlite3

def inspect_database():
    try:
        conn = sqlite3.connect('data/trading_bot.db')
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cursor.fetchall()
        
        print("📊 DATABASE TABLES:")
        for table in tables:
            print(f"  📋 {table[0]}")
        
        print("\n📊 STRATEGY_PERFORMANCE COLUMNS:")
        cursor.execute("PRAGMA table_info(strategy_performance);")
        columns = cursor.fetchall()
        
        for col in columns:
            print(f"  📋 {col[1]} ({col[2]})")
        
        # Check if there are any records
        cursor.execute("SELECT COUNT(*) FROM strategy_performance;")
        count = cursor.fetchone()[0]
        print(f"\n📊 Records in strategy_performance: {count}")
        
        if count > 0:
            cursor.execute("SELECT * FROM strategy_performance LIMIT 3;")
            records = cursor.fetchall()
            print("\n📊 Sample records:")
            for record in records:
                print(f"  📋 {record}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_database()