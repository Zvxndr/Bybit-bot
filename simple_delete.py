#!/usr/bin/env python3
"""
SIMPLE DATA DELETER - WORKING VERSION
Direct database operations, no complex APIs needed
"""

import sqlite3
import os
import sys

DB_PATH = 'data/trading_bot.db'

def check_data():
    """Show what data exists"""
    if not os.path.exists(DB_PATH):
        print("❌ No database found")
        return False
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT symbol, timeframe, COUNT(*) FROM historical_data GROUP BY symbol, timeframe')
        data = cursor.fetchall()
        conn.close()
        
        if not data:
            print("📊 No data in database")
            return False
            
        print("📊 Current data:")
        for symbol, timeframe, count in data:
            print(f"  {symbol} {timeframe}: {count:,} records")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def delete_symbol(symbol, timeframe):
    """Delete specific symbol data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM historical_data WHERE symbol=? AND timeframe=?', (symbol, timeframe))
        count = cursor.fetchone()[0]
        
        if count == 0:
            print(f"📊 No {symbol} {timeframe} data found")
            conn.close()
            return
            
        cursor.execute('DELETE FROM historical_data WHERE symbol=? AND timeframe=?', (symbol, timeframe))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"✅ Deleted {deleted:,} {symbol} {timeframe} records")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def delete_all():
    """Delete ALL data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM historical_data')
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("📊 No data to delete")
            conn.close()
            return
            
        cursor.execute('DELETE FROM historical_data')
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"✅ Deleted ALL {deleted:,} records")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🗑️  SIMPLE DATA DELETER")
    print("=" * 30)
    
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python simple_delete.py check")
        print("  python simple_delete.py delete BTCUSDT 15m")  
        print("  python simple_delete.py delete_all")
        return
        
    cmd = sys.argv[1]
    
    if cmd == 'check':
        check_data()
        
    elif cmd == 'delete' and len(sys.argv) == 4:
        symbol = sys.argv[2]
        timeframe = sys.argv[3]
        print(f"🗑️ Deleting {symbol} {timeframe}...")
        delete_symbol(symbol, timeframe)
        
    elif cmd == 'delete_all':
        confirm = input("⚠️ Delete ALL data? Type 'yes': ")
        if confirm.lower() == 'yes':
            print("🗑️ Deleting ALL data...")
            delete_all()
        else:
            print("❌ Cancelled")
            
    else:
        print("❌ Invalid command")

if __name__ == '__main__':
    main()