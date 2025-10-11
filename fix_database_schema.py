#!/usr/bin/env python3
"""
Database Schema Fix Script
Fixes the backtest_results table schema to include missing columns.
"""

import sqlite3
import sys
import os

def check_and_fix_database_schema():
    """Check and fix the backtest_results table schema."""
    
    db_path = "data/trading_bot.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check current schema
        cursor.execute("PRAGMA table_info(backtest_results)")
        columns = cursor.fetchall()
        
        print("📊 Current backtest_results table schema:")
        column_names = []
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
            column_names.append(col[1])
        
        # Check if the required columns exist
        required_columns = ['winning_trades', 'losing_trades', 'profit_factor']
        missing_columns = [col for col in required_columns if col not in column_names]
        
        if missing_columns:
            print(f"\n🔧 Missing columns: {missing_columns}")
            print("Adding missing columns...")
            
            for col in missing_columns:
                if col in ['winning_trades', 'losing_trades']:
                    cursor.execute(f"ALTER TABLE backtest_results ADD COLUMN {col} INTEGER DEFAULT 0")
                elif col == 'profit_factor':
                    cursor.execute(f"ALTER TABLE backtest_results ADD COLUMN {col} REAL DEFAULT 0.0")
                print(f"✅ Added column: {col}")
            
            conn.commit()
            print("\n🎉 Database schema updated successfully!")
        else:
            print("\n✅ All required columns exist!")
        
        # Verify the update
        cursor.execute("PRAGMA table_info(backtest_results)")
        columns = cursor.fetchall()
        
        print("\n📊 Updated backtest_results table schema:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error fixing database schema: {e}")
        return False

if __name__ == "__main__":
    success = check_and_fix_database_schema()
    sys.exit(0 if success else 1)