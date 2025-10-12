#!/usr/bin/env python3
import sqlite3
import os

# Check database structure and content
db_path = "data/trading_bot.db"

if not os.path.exists(db_path):
    print("❌ Database file does not exist!")
    exit(1)

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check what tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"📋 Available tables in {db_path}:")
    for table in tables:
        print(f"  - {table[0]}")
    
    if not tables:
        print("❌ NO TABLES FOUND - Database is empty!")
    else:
        # Check each table's content
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"\n📊 Table '{table_name}': {count} records")
            
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample = cursor.fetchall()
                
                # Get column names
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                
                print(f"  Columns: {', '.join(columns)}")
                print(f"  Sample data:")
                for i, row in enumerate(sample):
                    print(f"    {i+1}. {row}")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()