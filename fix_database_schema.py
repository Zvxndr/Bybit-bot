#!/usr/bin/env python3
"""
Database Schema Fix Script
Forcibly recreates the database with correct StrategyPipeline schema.
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
from sqlalchemy import text
import sqlalchemy as sa

def main():
    """Fix the database schema by recreating it properly."""
    
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
    
    # Initialize the database (this will create the engine)
    db_manager.initialize()
    
    print("🔧 Dropping existing tables...")
    
    # Drop all existing tables
    with db_manager.get_session() as session:
        # Drop strategy_pipeline table if it exists
        try:
            session.execute(text("DROP TABLE IF EXISTS strategy_pipeline"))
            session.commit()
            print("✅ Dropped strategy_pipeline table")
        except Exception as e:
            print(f"⚠️ Error dropping table: {e}")
    
    print("🏗️ Creating tables with correct schema...")
    
    # Recreate all tables
    Base.metadata.drop_all(bind=db_manager.engine)
    Base.metadata.create_all(bind=db_manager.engine)
    
    print("✅ Tables created successfully")
    
    # Verify the schema
    print("🔍 Verifying schema...")
    
    with db_manager.get_session() as session:
        # Check if phase_start_time column exists
        try:
            result = session.execute(text("PRAGMA table_info(strategy_pipeline)"))
            columns = result.fetchall()
            
            print("\n📋 StrategyPipeline table columns:")
            column_names = []
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
                column_names.append(col[1])
            
            required_columns = ['phase_start_time', 'phase_duration', 'current_phase']
            missing_columns = [col for col in required_columns if col not in column_names]
            
            if missing_columns:
                print(f"❌ Missing columns: {missing_columns}")
                return False
            else:
                print("✅ All required columns present")
                
        except Exception as e:
            print(f"❌ Error verifying schema: {e}")
            return False
    
    print("\n🎉 Database schema fixed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)