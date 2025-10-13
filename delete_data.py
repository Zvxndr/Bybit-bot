#!/usr/bin/env python3
"""
Data Deletion Utility for Trading Bot
Provides direct database operations for deleting historical data
"""

import sqlite3
import sys
import os
from datetime import datetime

def get_database_path():
    """Get the database path"""
    db_path = 'data/trading_bot.db'
    if os.path.exists(db_path):
        return db_path
    else:
        print(f"❌ Database not found at {db_path}")
        return None

def show_available_data():
    """Show all available data in the database"""
    db_path = get_database_path()
    if not db_path:
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, timeframe, COUNT(*) as count, 
                   MIN(timestamp) as start, MAX(timestamp) as end 
            FROM historical_data 
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        ''')
        
        data = cursor.fetchall()
        
        if not data:
            print("📊 No historical data found in database")
            return
        
        print("📊 Available historical data:")
        print("=" * 60)
        for row in data:
            symbol, timeframe, count, start, end = row
            print(f"  {symbol} {timeframe}: {count:,} records")
            print(f"    Range: {start} to {end}")
            print()
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")

def delete_symbol_data(symbol, timeframe):
    """Delete data for a specific symbol and timeframe"""
    db_path = get_database_path()
    if not db_path:
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First check how many records exist
        cursor.execute(
            "SELECT COUNT(*) FROM historical_data WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe)
        )
        count_before = cursor.fetchone()[0]
        
        if count_before == 0:
            print(f"📊 No data found for {symbol} {timeframe}")
            conn.close()
            return True
        
        # Delete the data
        cursor.execute(
            "DELETE FROM historical_data WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe)
        )
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"✅ Deleted {deleted_count:,} records for {symbol} {timeframe}")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting data: {e}")
        return False

def delete_all_data():
    """Delete ALL historical data"""
    db_path = get_database_path()
    if not db_path:
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check total records
        cursor.execute("SELECT COUNT(*) FROM historical_data")
        count_before = cursor.fetchone()[0]
        
        if count_before == 0:
            print("📊 No historical data to delete")
            conn.close()
            return True
        
        # Delete all data
        cursor.execute("DELETE FROM historical_data")
        deleted_count = cursor.rowcount
        conn.commit()
        
        # Reset auto-increment (optional)
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='historical_data'")
        conn.commit()
        conn.close()
        
        print(f"✅ Deleted ALL {deleted_count:,} historical records")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting all data: {e}")
        return False

def vacuum_database():
    """Optimize database after deletions"""
    db_path = get_database_path()
    if not db_path:
        return
    
    try:
        conn = sqlite3.connect(db_path)
        print("🔄 Optimizing database...")
        conn.execute("VACUUM")
        conn.close()
        print("✅ Database optimized")
    except Exception as e:
        print(f"❌ Error optimizing database: {e}")

def main():
    if len(sys.argv) < 2:
        print("🗑️  Trading Bot Data Deletion Utility")
        print("=" * 40)
        print("Usage:")
        print("  python delete_data.py show                     # Show available data")
        print("  python delete_data.py symbol BTCUSDT 15m      # Delete specific symbol/timeframe")
        print("  python delete_data.py all                     # Delete ALL data")
        print("  python delete_data.py vacuum                  # Optimize database")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'show':
        show_available_data()
        
    elif command == 'symbol':
        if len(sys.argv) != 4:
            print("❌ Usage: python delete_data.py symbol SYMBOL TIMEFRAME")
            print("   Example: python delete_data.py symbol BTCUSDT 15m")
            return
        
        symbol = sys.argv[2].upper()
        timeframe = sys.argv[3]
        
        print(f"🗑️  Deleting {symbol} {timeframe} data...")
        if delete_symbol_data(symbol, timeframe):
            vacuum_database()
        
    elif command == 'all':
        confirm = input("⚠️  Are you sure you want to delete ALL historical data? (yes/no): ")
        if confirm.lower() in ['yes', 'y']:
            print("🗑️  Deleting ALL historical data...")
            if delete_all_data():
                vacuum_database()
        else:
            print("❌ Deletion cancelled")
            
    elif command == 'vacuum':
        vacuum_database()
        
    else:
        print(f"❌ Unknown command: {command}")
        print("Valid commands: show, symbol, all, vacuum")

if __name__ == "__main__":
    main()