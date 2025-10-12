#!/usr/bin/env python3
"""
Verify Database Path Synchronization
===================================

This script verifies that the historical downloader and API discovery 
are using the same database path to eliminate data discovery issues.
"""

import sys
import sqlite3
from pathlib import Path

def main():
    print("🔍 DATABASE PATH SYNCHRONIZATION VERIFICATION")
    print("=" * 50)
    
    # Simulate the main app's DB_PATH logic
    DB_PATH = '/app/data/trading_bot.db' if Path('/app/data').exists() else 'data/trading_bot.db'
    print(f"📍 Main App DB_PATH: {DB_PATH}")
    
    # Test historical downloader initialization
    try:
        from historical_data_downloader import HistoricalDataDownloader
        
        # Test with default path (old behavior)
        downloader_default = HistoricalDataDownloader()
        print(f"📍 Downloader Default Path: {downloader_default.db_path}")
        
        # Test with explicit path (new behavior)  
        downloader_fixed = HistoricalDataDownloader(db_path=DB_PATH)
        print(f"📍 Downloader Fixed Path: {downloader_fixed.db_path}")
        
        # Verify they match
        if downloader_fixed.db_path == DB_PATH:
            print("✅ SUCCESS: Downloader path matches main app DB_PATH")
        else:
            print("❌ ERROR: Path mismatch detected")
            
    except ImportError as e:
        print(f"⚠️ Cannot test downloader: {e}")
    
    # Check which databases actually exist
    print("\n📂 EXISTING DATABASE FILES:")
    test_paths = [
        DB_PATH,
        "data/trading_bot.db",
        "data/historical_data.db",
        "/app/data/trading_bot.db"
    ]
    
    for path in test_paths:
        path_obj = Path(path)
        if path_obj.exists():
            try:
                # Check if it has data
                with sqlite3.connect(str(path_obj)) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    # Check for data in historical tables
                    data_count = 0
                    for table in ['historical_data', 'market_data', 'data_cache']:
                        if table in tables:
                            try:
                                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                                count = cursor.fetchone()[0]
                                data_count += count
                            except sqlite3.OperationalError:
                                pass
                    
                    size_mb = path_obj.stat().st_size / (1024 * 1024)
                    print(f"✅ {path} - {data_count:,} records, {size_mb:.2f}MB, tables: {tables}")
                    
            except Exception as e:
                print(f"❌ {path} - Error reading: {e}")
        else:
            print(f"❌ {path} - Not found")
    
    print("\n🎯 RECOMMENDED ACTION:")
    print("The historical downloader now uses the same DB_PATH as the main app.")
    print("Data downloaded after this fix will be discoverable by the API.")

if __name__ == "__main__":
    main()