#!/usr/bin/env python3
"""
Container Startup Database Schema Checker
Ensures the database schema is correct when the container starts.
This script should run before the main application starts.
"""

import sys
import os
from pathlib import Path
import sqlite3
from datetime import datetime

def check_and_fix_schema():
    """Check and fix database schema on container startup."""
    
    print("🔍 Container Startup: Checking database schema...")
    
    db_path = "./data/trading_bot.db"
    
    # Ensure data directory exists
    os.makedirs("./data", exist_ok=True)
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if strategy_pipeline table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_pipeline'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            print("📋 strategy_pipeline table doesn't exist - creating with correct schema")
            
            # Create table with correct schema immediately
            create_table_sql = """
            CREATE TABLE strategy_pipeline (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id VARCHAR(255) NOT NULL UNIQUE,
                strategy_name VARCHAR(255) NOT NULL,
                current_phase VARCHAR(50) NOT NULL DEFAULT 'backtest',
                phase_start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                phase_duration INTEGER,
                asset_pair VARCHAR(20) NOT NULL,
                base_asset VARCHAR(10) NOT NULL,
                strategy_type VARCHAR(50) NOT NULL,
                strategy_description TEXT,
                backtest_score FLOAT,
                backtest_return FLOAT,
                sharpe_ratio FLOAT,
                max_drawdown FLOAT,
                win_rate FLOAT,
                profit_factor FLOAT,
                paper_start_date DATETIME,
                paper_end_date DATETIME,
                paper_pnl FLOAT,
                paper_return_pct FLOAT,
                paper_trade_count INTEGER DEFAULT 0,
                paper_win_count INTEGER DEFAULT 0,
                live_start_date DATETIME,
                live_pnl FLOAT,
                live_return_pct FLOAT,
                live_trade_count INTEGER DEFAULT 0,
                live_win_count INTEGER DEFAULT 0,
                promotion_threshold FLOAT NOT NULL DEFAULT 10.0,
                graduation_threshold FLOAT NOT NULL DEFAULT 10.0,
                rejection_threshold FLOAT NOT NULL DEFAULT -5.0,
                auto_promote BOOLEAN NOT NULL DEFAULT 1,
                auto_graduate BOOLEAN NOT NULL DEFAULT 1,
                max_paper_duration INTEGER NOT NULL DEFAULT 604800,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                rejection_reason TEXT,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                promoted_at DATETIME,
                graduated_at DATETIME,
                rejected_at DATETIME,
                risk_score FLOAT DEFAULT 0.0,
                volatility_score FLOAT,
                correlation_risk FLOAT,
                total_pnl FLOAT,
                best_return FLOAT,
                worst_return FLOAT,
                consistency_score FLOAT
            )
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            print("✅ strategy_pipeline table created with correct schema including phase_start_time")
            return True
        
        # Check if phase_start_time column exists
        cursor.execute("PRAGMA table_info(strategy_pipeline)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if 'phase_start_time' in column_names:
            print("✅ Database schema is correct - phase_start_time column exists")
            return True
        
        print("⚠️  SCHEMA ISSUE DETECTED: phase_start_time column missing")
        print("🔄 Fixing schema automatically...")
        
        # Backup existing data
        cursor.execute("SELECT * FROM strategy_pipeline")
        backup_data = cursor.fetchall()
        
        if backup_data:
            print(f"💾 Backing up {len(backup_data)} existing records...")
        
        # Get column info for backup
        existing_columns = [col[1] for col in columns]
        
        # Drop the old table
        cursor.execute("DROP TABLE strategy_pipeline")
        
        # Create new table with correct schema
        create_table_sql = """
        CREATE TABLE strategy_pipeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id VARCHAR(255) NOT NULL UNIQUE,
            strategy_name VARCHAR(255) NOT NULL,
            current_phase VARCHAR(50) NOT NULL DEFAULT 'backtest',
            phase_start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            phase_duration INTEGER,
            asset_pair VARCHAR(20) NOT NULL,
            base_asset VARCHAR(10) NOT NULL,
            strategy_type VARCHAR(50) NOT NULL,
            strategy_description TEXT,
            backtest_score FLOAT,
            backtest_return FLOAT,
            sharpe_ratio FLOAT,
            max_drawdown FLOAT,
            win_rate FLOAT,
            profit_factor FLOAT,
            paper_start_date DATETIME,
            paper_end_date DATETIME,
            paper_pnl FLOAT,
            paper_return_pct FLOAT,
            paper_trade_count INTEGER DEFAULT 0,
            paper_win_count INTEGER DEFAULT 0,
            live_start_date DATETIME,
            live_pnl FLOAT,
            live_return_pct FLOAT,
            live_trade_count INTEGER DEFAULT 0,
            live_win_count INTEGER DEFAULT 0,
            promotion_threshold FLOAT NOT NULL DEFAULT 10.0,
            graduation_threshold FLOAT NOT NULL DEFAULT 10.0,
            rejection_threshold FLOAT NOT NULL DEFAULT -5.0,
            auto_promote BOOLEAN NOT NULL DEFAULT 1,
            auto_graduate BOOLEAN NOT NULL DEFAULT 1,
            max_paper_duration INTEGER NOT NULL DEFAULT 604800,
            is_active BOOLEAN NOT NULL DEFAULT 1,
            rejection_reason TEXT,
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            promoted_at DATETIME,
            graduated_at DATETIME,
            rejected_at DATETIME,
            risk_score FLOAT DEFAULT 0.0,
            volatility_score FLOAT,
            correlation_risk FLOAT,
            total_pnl FLOAT,
            best_return FLOAT,
            worst_return FLOAT,
            consistency_score FLOAT
        )
        """
        
        cursor.execute(create_table_sql)
        
        # Restore data if any existed
        if backup_data:
            print("📥 Restoring backed up data...")
            
            # Create insert statement with proper column mapping
            insert_columns = [
                'strategy_id', 'strategy_name', 'current_phase', 'asset_pair', 
                'base_asset', 'strategy_type', 'strategy_description',
                'backtest_score', 'backtest_return', 'sharpe_ratio', 'max_drawdown',
                'win_rate', 'profit_factor', 'paper_start_date', 'paper_end_date',
                'paper_pnl', 'paper_return_pct', 'paper_trade_count', 'paper_win_count',
                'live_start_date', 'live_pnl', 'live_return_pct', 'live_trade_count',
                'live_win_count', 'promotion_threshold', 'graduation_threshold',
                'rejection_threshold', 'auto_promote', 'auto_graduate',
                'max_paper_duration', 'is_active', 'rejection_reason', 'notes',
                'created_at', 'updated_at', 'promoted_at', 'graduated_at',
                'rejected_at', 'risk_score', 'volatility_score', 'correlation_risk',
                'total_pnl', 'best_return', 'worst_return', 'consistency_score'
            ]
            
            for row in backup_data:
                # Map old row to new columns with defaults
                values = []
                for i, col in enumerate(insert_columns):
                    if i < len(row) and row[i] is not None:
                        values.append(row[i])
                    elif col == 'phase_start_time':
                        values.append(datetime.now().isoformat())
                    elif col in ['paper_trade_count', 'paper_win_count', 'live_trade_count', 'live_win_count']:
                        values.append(0)
                    elif col in ['promotion_threshold', 'graduation_threshold']:
                        values.append(10.0)
                    elif col == 'rejection_threshold':
                        values.append(-5.0)
                    elif col in ['auto_promote', 'auto_graduate', 'is_active']:
                        values.append(1)
                    elif col == 'max_paper_duration':
                        values.append(604800)
                    elif col == 'risk_score':
                        values.append(0.0)
                    else:
                        values.append(None)
                
                placeholders = ','.join(['?' for _ in values])
                insert_sql = f"INSERT INTO strategy_pipeline ({','.join(insert_columns)}) VALUES ({placeholders})"
                cursor.execute(insert_sql, values)
            
            print(f"✅ Restored {len(backup_data)} records with updated schema")
        
        conn.commit()
        print("✅ Schema fix completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Schema fix failed: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()

if __name__ == "__main__":
    print("🚀 Container Database Schema Validator")
    print("=" * 40)
    
    success = check_and_fix_schema()
    
    if success:
        print("✅ Database ready for application startup")
    else:
        print("❌ Database schema issues - application may fail")
        sys.exit(1)