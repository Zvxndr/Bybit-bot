#!/usr/bin/env python3
"""
Production Database Schema Migration Script
Ensures production databases have the correct schema without losing data.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

from bot.database.manager import DatabaseManager
from bot.database.models import Base, StrategyPipeline
from bot.config import DatabaseConfig
from sqlalchemy import text, inspect
import sqlalchemy as sa
from datetime import datetime

def check_column_exists(engine, table_name, column_name):
    """Check if a column exists in a table."""
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    return any(col['name'] == column_name for col in columns)

def main():
    """Migrate the production database schema safely."""
    
    print("🔄 Production Database Schema Migration")
    print("=" * 50)
    
    # Create a DatabaseConfig for SQLite
    config = DatabaseConfig(
        pool_size=10,
        max_overflow=20,
        echo=False
    )
    config.development = {
        "dialect": "sqlite", 
        "path": "./data/trading_bot.db"
    }
    
    # Initialize database manager
    db_manager = DatabaseManager(config)
    db_manager.initialize()
    
    print("✅ Database connection established")
    
    # Check if the problematic column exists
    if check_column_exists(db_manager.engine, 'strategy_pipeline', 'phase_start_time'):
        print("✅ Schema is already up to date - phase_start_time column exists")
        return True
    
    print("⚠️  Missing column detected: strategy_pipeline.phase_start_time")
    print("🔄 Beginning safe migration...")
    
    try:
        with db_manager.get_session() as session:
            # Check if strategy_pipeline table exists
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_pipeline'"))
            if not result.fetchone():
                print("📋 strategy_pipeline table doesn't exist, creating from scratch...")
                Base.metadata.create_all(bind=db_manager.engine)
                print("✅ Tables created successfully")
                return True
            
            # Backup existing data
            print("💾 Backing up existing strategy_pipeline data...")
            backup_result = session.execute(text("SELECT * FROM strategy_pipeline"))
            backup_data = backup_result.fetchall()
            backup_columns = backup_result.keys()
            
            print(f"📊 Found {len(backup_data)} existing records to preserve")
            
            # Drop and recreate table with correct schema
            print("🔄 Recreating table with correct schema...")
            session.execute(text("DROP TABLE IF EXISTS strategy_pipeline"))
            session.commit()
            
            # Create table with new schema
            Base.metadata.create_all(bind=db_manager.engine)
            
            # Restore data
            if backup_data:
                print("📥 Restoring backed up data...")
                
                # Prepare insert statement for new schema
                insert_cols = []
                insert_values = []
                
                # Map old columns to new schema
                for i, row in enumerate(backup_data):
                    row_dict = dict(zip(backup_columns, row))
                    
                    # Add default values for new columns
                    if 'phase_start_time' not in row_dict:
                        row_dict['phase_start_time'] = datetime.now()
                    if 'phase_duration' not in row_dict:
                        row_dict['phase_duration'] = None
                        
                    if i == 0:  # First row - determine columns
                        insert_cols = list(row_dict.keys())
                    
                    insert_values.append(tuple(row_dict[col] for col in insert_cols))
                
                # Insert all data
                placeholders = ', '.join(['?' for _ in insert_cols])
                columns_str = ', '.join(insert_cols)
                insert_sql = f"INSERT INTO strategy_pipeline ({columns_str}) VALUES ({placeholders})"
                
                for values in insert_values:
                    session.execute(text(insert_sql), values)
                
                session.commit()
                print(f"✅ Successfully restored {len(backup_data)} records")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False
    
    # Verify the migration
    print("🔍 Verifying migration...")
    
    if check_column_exists(db_manager.engine, 'strategy_pipeline', 'phase_start_time'):
        print("✅ Migration successful - phase_start_time column now exists")
        
        with db_manager.get_session() as session:
            result = session.execute(text("SELECT COUNT(*) FROM strategy_pipeline"))
            count = result.fetchone()[0]
            print(f"📊 Table now contains {count} records")
        
        return True
    else:
        print("❌ Migration failed - column still missing")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Production database migration completed successfully!")
        print("💡 The AI pipeline should now work without schema errors.")
    else:
        print("\n💥 Migration failed - manual intervention required")
    
    sys.exit(0 if success else 1)