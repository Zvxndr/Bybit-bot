#!/usr/bin/env python3
"""
Initialize missing database tables for production deployment
"""

import sqlite3
import os
from loguru import logger

def init_missing_tables():
    """Initialize any missing database tables"""
    
    # Database paths
    db_paths = [
        "data/trading_bot.db",
        "data/historical_data.db"
    ]
    
    for db_path in db_paths:
        if not os.path.exists(db_path):
            logger.info(f"Database {db_path} doesn't exist yet, will be created on first use")
            continue
            
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Tax compliance log table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tax_compliance_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        event_data TEXT NOT NULL,
                        financial_year TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        sydney_time TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Pipeline configuration table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_config (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        config_key TEXT NOT NULL UNIQUE,
                        config_value TEXT NOT NULL,
                        description TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Strategy discovery results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_discovery (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL UNIQUE,
                        strategy_name TEXT NOT NULL,
                        discovery_method TEXT NOT NULL,
                        confidence_score REAL DEFAULT 0.0,
                        backtest_results TEXT,
                        phase TEXT DEFAULT 'discovery',
                        status TEXT DEFAULT 'discovered',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Historical backtest results
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        backtest_id TEXT NOT NULL UNIQUE,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        start_date TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        total_return REAL DEFAULT 0.0,
                        sharpe_ratio REAL DEFAULT 0.0,
                        max_drawdown REAL DEFAULT 0.0,
                        win_rate REAL DEFAULT 0.0,
                        total_trades INTEGER DEFAULT 0,
                        results_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # System status monitoring
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_status (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        component TEXT NOT NULL,
                        status TEXT NOT NULL,
                        last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        error_message TEXT,
                        metadata TEXT
                    )
                """)
                
                conn.commit()
                logger.info(f"✅ Initialized missing tables for {db_path}")
                
        except Exception as e:
            logger.error(f"❌ Error initializing tables for {db_path}: {e}")

if __name__ == "__main__":
    init_missing_tables()
    logger.info("Database table initialization complete")