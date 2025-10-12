#!/usr/bin/env python3
"""
DigitalOcean Historical Data Migration
====================================

Migration script specifically designed for DigitalOcean App Platform.
Handles both local development and production deployment scenarios.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime
import os

def get_environment_info():
    """Detect if we're running on DigitalOcean App Platform"""
    is_digitalocean = Path('/app').exists() and Path('/app/data').exists()
    is_local = not is_digitalocean
    
    return {
        'is_digitalocean': is_digitalocean,
        'is_local': is_local,
        'app_path': '/app' if is_digitalocean else '.',
        'data_path': '/app/data' if is_digitalocean else 'data'
    }

def migrate_historical_data():
    """Migrate data with environment-specific paths"""
    
    env = get_environment_info()
    
    print("🌊 DIGITALOCEAN HISTORICAL DATA MIGRATION")
    print("=" * 50)
    print(f"📍 Environment: {'DigitalOcean App Platform' if env['is_digitalocean'] else 'Local Development'}")
    
    # Define paths based on environment
    if env['is_digitalocean']:
        # DigitalOcean App Platform paths
        possible_source_paths = [
            Path('/app/data/historical_data.db'),
            Path('/app/src/data/historical_data.db'),
            Path('/app/data/speed_demon_cache/market_data.db'),
            Path('/app/src/data/speed_demon_cache/market_data.db')
        ]
        target_db = Path('/app/data/trading_bot.db')
    else:
        # Local development paths
        possible_source_paths = [
            Path('data/historical_data.db'),
            Path('src/data/historical_data.db'),
            Path('data/speed_demon_cache/market_data.db'),
            Path('src/data/speed_demon_cache/market_data.db')
        ]
        target_db = Path('data/trading_bot.db')
    
    # Find source database with data
    source_db = None
    source_count = 0
    
    print("🔍 Searching for historical data...")
    for path in possible_source_paths:
        if path.exists():
            try:
                conn = sqlite3.connect(str(path))
                cursor = conn.cursor()
                
                # Check for historical data table
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in ['historical_data', 'market_data', 'data_cache']:
                    if table in tables:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        if count > source_count:
                            source_db = path
                            source_count = count
                            print(f"📊 Found {count:,} records in {path} ({table} table)")
                
                conn.close()
            except Exception as e:
                print(f"⚠️ Error checking {path}: {e}")
    
    if not source_db or source_count == 0:
        print("❌ No historical data found to migrate")
        return False
        
    print(f"📍 Source: {source_db} ({source_count:,} records)")
    print(f"📍 Target: {target_db}")
    
    # Ensure target directory exists
    target_db.parent.mkdir(parents=True, exist_ok=True)
    
    # Create target database if it doesn't exist
    if not target_db.exists():
        print("🔧 Creating target database...")
        target_conn = sqlite3.connect(str(target_db))
        target_conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        """)
        target_conn.commit()
        target_conn.close()
    
    try:
        # Connect to both databases
        source_conn = sqlite3.connect(str(source_db))
        target_conn = sqlite3.connect(str(target_db))
        
        # Get current target count
        target_cursor = target_conn.cursor()
        target_cursor.execute("SELECT COUNT(*) FROM historical_data")
        target_count_before = target_cursor.fetchone()[0]
        
        print(f"📊 Target database currently has {target_count_before:,} records")
        
        # Find the source table with data
        source_cursor = source_conn.cursor()
        source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in source_cursor.fetchall()]
        
        source_table = None
        for table in ['historical_data', 'market_data', 'data_cache']:
            if table in tables:
                source_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                if source_cursor.fetchone()[0] > 0:
                    source_table = table
                    break
        
        if not source_table:
            print("❌ No data found in source database")
            return False
            
        print(f"🚀 Migrating from {source_table} table...")
        
        # Get appropriate columns based on table structure
        if source_table == 'historical_data':
            columns = "symbol, timeframe, timestamp, open, high, low, close, volume, created_at"
            source_query = f"SELECT {columns} FROM {source_table} ORDER BY timestamp"
        else:
            # For market_data or data_cache tables, map columns
            columns = "symbol, timeframe, timestamp, open_price, high_price, low_price, close_price, volume, created_at"
            source_query = f"SELECT {columns} FROM {source_table} ORDER BY timestamp"
        
        source_cursor.execute(source_query)
        records = source_cursor.fetchall()
        
        migrated_count = 0
        for record in records:
            try:
                if source_table == 'historical_data':
                    target_cursor.execute("""
                        INSERT OR REPLACE INTO historical_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, record)
                else:
                    # Map columns from market_data/data_cache format
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
        if env['is_digitalocean']:
            print("1. Restart the DigitalOcean app to pick up the changes")
            print("2. Test the discovery API endpoint")
            print("3. Verify backtesting controls show available data")
        else:
            print("1. Test the discovery API - data should now be visible")
            print("2. Test backtesting controls - should show available data")
            print("3. Deploy to DigitalOcean when ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = migrate_historical_data()
    sys.exit(0 if success else 1)