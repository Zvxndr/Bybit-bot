#!/usr/bin/env python3
"""
Migrate Historical Data to Unified Database
==========================================

This script migrates existing historical data from the old location 
(data/historical_data.db) to the unified database location used by 
both the downloader and discovery API.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

def migrate_historical_data():
    """Migrate data from old location to unified database"""
    
    # Define paths
    source_db = Path("data/historical_data.db")
    target_db = Path("data/trading_bot.db") 
    
    print("📦 HISTORICAL DATA MIGRATION")
    print("=" * 40)
    print(f"📍 Source: {source_db}")
    print(f"📍 Target: {target_db}")
    
    # Check source exists
    if not source_db.exists():
        print("❌ Source database not found - nothing to migrate")
        return False
        
    # Check target exists
    if not target_db.exists():
        print("❌ Target database not found - run the app first to create it")
        return False
    
    try:
        # Connect to both databases
        source_conn = sqlite3.connect(str(source_db))
        target_conn = sqlite3.connect(str(target_db))
        
        # Check source data
        source_cursor = source_conn.cursor()
        source_cursor.execute("SELECT COUNT(*) FROM historical_data")
        source_count = source_cursor.fetchone()[0]
        
        print(f"📊 Found {source_count:,} records in source database")
        
        if source_count == 0:
            print("❌ No data to migrate")
            return False
        
        # Check target table exists and get current count
        target_cursor = target_conn.cursor()
        target_cursor.execute("""
            SELECT COUNT(*) FROM historical_data
        """)
        target_count_before = target_cursor.fetchone()[0]
        
        print(f"📊 Target database currently has {target_count_before:,} records")
        
        # Migrate the data
        print("🚀 Starting migration...")
        
        # Get all data from source
        source_cursor.execute("""
            SELECT symbol, timeframe, timestamp, open, high, low, close, volume, created_at
            FROM historical_data
            ORDER BY timestamp
        """)
        
        records = source_cursor.fetchall()
        migrated_count = 0
        
        for record in records:
            try:
                target_cursor.execute("""
                    INSERT OR REPLACE INTO historical_data 
                    (symbol, timeframe, timestamp, open, high, low, close, volume, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, record)
                migrated_count += 1
            except sqlite3.Error as e:
                print(f"⚠️ Error migrating record: {e}")
                continue
        
        # Commit changes
        target_conn.commit()
        
        # Verify final count
        target_cursor.execute("SELECT COUNT(*) FROM historical_data")
        target_count_after = target_cursor.fetchone()[0]
        
        print(f"✅ Migration complete!")
        print(f"📊 Records migrated: {migrated_count:,}")
        print(f"📊 Target database now has: {target_count_after:,} records")
        print(f"📊 New records added: {target_count_after - target_count_before:,}")
        
        # Close connections
        source_conn.close()
        target_conn.close()
        
        print("\n🎯 NEXT STEPS:")
        print("1. Test the discovery API - data should now be visible")
        print("2. Test backtesting controls - should show available data")
        print("3. Consider backing up the old database before deleting")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = migrate_historical_data()
    sys.exit(0 if success else 1)