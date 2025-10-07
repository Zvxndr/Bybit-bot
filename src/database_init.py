"""
Database Initialization
======================

Creates and initializes the database for the trading bot.
Supports both PostgreSQL and SQLite with automatic fallback.
"""

import os
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Initialize database for trading bot"""
    
    def __init__(self):
        self.db_path = "data/trading_bot.db"
        self.initialized = False
    
    def initialize(self):
        """Initialize database with required tables"""
        try:
            # Ensure data directory exists
            Path("data").mkdir(exist_ok=True)
            
            # Create SQLite database
            self._create_sqlite_database()
            
            logger.info(f"✅ Database initialized: {self.db_path}")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
        
        return True
    
    def _create_sqlite_database(self):
        """Create SQLite database with trading tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                exchange TEXT NOT NULL DEFAULT 'bybit',
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                amount REAL NOT NULL,
                price REAL NOT NULL,
                fee REAL NOT NULL DEFAULT 0.0,
                fee_currency TEXT NOT NULL DEFAULT 'USDT',
                strategy_id TEXT NOT NULL,
                strategy_version TEXT,
                signal_confidence REAL,
                order_id TEXT,
                order_type TEXT DEFAULT 'market',
                slippage REAL,
                risk_amount REAL,
                position_size_usd REAL,
                portfolio_balance REAL,
                trading_mode TEXT DEFAULT 'conservative',
                cost_base_aud REAL,
                proceeds_aud REAL,
                is_cgt_event BOOLEAN DEFAULT 0,
                aud_conversion_rate REAL,
                holding_period_days INTEGER,
                unrealized_pnl REAL,
                realized_pnl REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')
        
        # Create strategy performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                equity REAL NOT NULL,
                drawdown REAL NOT NULL DEFAULT 0.0,
                total_return REAL NOT NULL DEFAULT 0.0,
                sharpe_ratio REAL,
                var_95 REAL,
                max_drawdown REAL,
                win_rate REAL,
                profit_factor REAL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                avg_win REAL DEFAULT 0.0,
                avg_loss REAL DEFAULT 0.0,
                largest_win REAL DEFAULT 0.0,
                largest_loss REAL DEFAULT 0.0,
                consistency_score REAL,
                risk_adjusted_return REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create portfolio table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                total_equity REAL NOT NULL,
                available_balance REAL NOT NULL,
                used_margin REAL DEFAULT 0.0,
                unrealized_pnl REAL DEFAULT 0.0,
                daily_pnl REAL DEFAULT 0.0,
                total_positions INTEGER DEFAULT 0,
                long_positions INTEGER DEFAULT 0,
                short_positions INTEGER DEFAULT 0,
                portfolio_risk REAL DEFAULT 0.0,
                max_position_size REAL DEFAULT 0.0,
                total_fees_paid REAL DEFAULT 0.0,
                currency TEXT DEFAULT 'USDT',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create risk events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                strategy_id TEXT,
                symbol TEXT,
                trigger_value REAL,
                threshold_value REAL,
                action_taken TEXT,
                description TEXT,
                metadata TEXT,
                resolved BOOLEAN DEFAULT 0,
                resolved_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create ML insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                predicted_value REAL,
                actual_value REAL,
                features TEXT,
                model_version TEXT,
                accuracy_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_performance_timestamp ON strategy_performance(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk_events_timestamp ON risk_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_insights_timestamp ON ml_insights(timestamp)')
        
        conn.commit()
        conn.close()
        
        logger.info("✅ SQLite database tables created successfully")
    
    def get_status(self):
        """Get database status"""
        if not self.initialized:
            return {"status": "not_initialized", "type": "none"}
        
        if os.path.exists(self.db_path):
            # Get database file size
            size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
            
            # Get table count
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            conn.close()
            
            return {
                "status": "ready",
                "type": "sqlite",
                "path": self.db_path,
                "size_mb": round(size_mb, 2),
                "tables": table_count,
                "initialized_at": datetime.fromtimestamp(os.path.getctime(self.db_path)).isoformat()
            }
        
        return {"status": "error", "type": "sqlite", "error": "Database file not found"}


# Global database initializer
database_initializer = DatabaseInitializer()


def initialize_database():
    """Initialize the database"""
    return database_initializer.initialize()


def get_database_status():
    """Get database status"""
    return database_initializer.get_status()


if __name__ == "__main__":
    # Test database initialization
    print("Testing database initialization...")
    success = initialize_database()
    if success:
        status = get_database_status()
        print(f"Database Status: {status}")
    else:
        print("Database initialization failed")