"""
Simplified Dashboard Integration
===============================

Working integration for the unified dashboard that uses existing main.py structure.
This extends the existing TradingAPI class to provide the additional endpoints
needed by the dashboard.

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import logging
import os
import sqlite3
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class SimplifiedDashboardAPI:
    """
    Simplified API endpoints for the dashboard that work with current setup.
    """
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.db_path = "data/trading_bot.db"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Initialize demo database if needed
        self._init_demo_database()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("✅ Simplified Dashboard API initialized")
    
    def _init_demo_database(self):
        """Initialize demo database with sample data if it doesn't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_pipeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT UNIQUE,
                    strategy_name TEXT,
                    current_phase TEXT,
                    asset_pair TEXT,
                    backtest_score REAL,
                    paper_pnl REAL,
                    live_pnl REAL,
                    sharpe_ratio REAL,
                    win_rate REAL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    side TEXT,
                    amount REAL,
                    price REAL,
                    strategy_id TEXT,
                    realized_pnl REAL
                )
            ''')
            
            # Insert sample data if tables are empty
            cursor.execute("SELECT COUNT(*) FROM strategy_pipeline")
            if cursor.fetchone()[0] == 0:
                sample_strategies = [
                    ('BTC_MR_A4F2D', 'BTC Mean Reversion Alpha', 'live', 'BTCUSDT', 82.5, 245.30, 1250.75, 2.1, 68.5),
                    ('ETH_BB_C7E9A', 'ETH Bollinger Bands', 'paper', 'ETHUSDT', 78.2, 156.40, 0, 1.8, 64.2),
                    ('SOL_TF_B3K8M', 'SOL Trend Following', 'backtest', 'SOLUSDT', 85.1, 0, 0, 2.3, 72.1),
                    ('ADA_RSI_F1L9P', 'ADA RSI Strategy', 'live', 'ADAUSDT', 79.8, 89.20, 456.30, 1.9, 69.8),
                    ('MATIC_MA_H6N2Q', 'MATIC Moving Average', 'paper', 'MATICUSDT', 76.4, 123.15, 0, 1.7, 62.3)
                ]
                
                for strategy in sample_strategies:
                    cursor.execute('''
                        INSERT OR IGNORE INTO strategy_pipeline 
                        (strategy_id, strategy_name, current_phase, asset_pair, backtest_score, paper_pnl, live_pnl, sharpe_ratio, win_rate)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', strategy)
                
                # Insert sample trades
                sample_trades = [
                    ('BTCUSDT', 'buy', 0.1, 43250.50, 'BTC_MR_A4F2D', 125.30),
                    ('ETHUSDT', 'sell', 2.5, 2640.75, 'ETH_BB_C7E9A', -45.20),
                    ('ADAUSDT', 'buy', 1000, 0.385, 'ADA_RSI_F1L9P', 89.45),
                ]
                
                for trade in sample_trades:
                    cursor.execute('''
                        INSERT INTO trades (symbol, side, amount, price, strategy_id, realized_pnl)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', trade)
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Demo database initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
    
    def _setup_routes(self):
        """Setup API routes for the dashboard"""
        
        @self.app.get("/api/strategies")
        async def get_strategies():
            """Get strategies from database"""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT strategy_id, strategy_name, current_phase, asset_pair, 
                           backtest_score, paper_pnl, live_pnl, sharpe_ratio, win_rate
                    FROM strategy_pipeline 
                    WHERE is_active = 1
                    ORDER BY created_at DESC
                ''')
                
                strategies = []
                for row in cursor.fetchall():
                    strategy = {
                        "id": row[0],
                        "name": row[1],
                        "asset": row[3],
                        "phase": row[2],
                        "performance": {
                            "backtest_score": row[4],
                            "paper_pnl": row[5],
                            "live_pnl": row[6],
                            "sharpe_ratio": row[7],
                            "win_rate": row[8]
                        },
                        "metrics": {
                            "phase_duration": random.uniform(2, 48),  # Demo data
                            "trade_count": random.randint(5, 25),
                            "ready_for_promotion": row[2] == 'backtest' and row[4] > 75,
                            "ready_for_graduation": row[2] == 'paper' and row[5] > 100
                        }
                    }
                    strategies.append(strategy)
                
                conn.close()
                
                return JSONResponse({
                    "success": True,
                    "data": strategies
                })
                
            except Exception as e:
                logger.error(f"Strategies fetch error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/pipeline-metrics")
        async def get_pipeline_metrics():
            """Get pipeline metrics from database"""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Count strategies by phase
                cursor.execute("SELECT current_phase, COUNT(*) FROM strategy_pipeline WHERE is_active = 1 GROUP BY current_phase")
                phase_counts = dict(cursor.fetchall())
                
                # Calculate metrics
                backtest_count = phase_counts.get('backtest', 0)
                paper_count = phase_counts.get('paper', 0)
                live_count = phase_counts.get('live', 0)
                
                # Get total live P&L
                cursor.execute("SELECT SUM(live_pnl) FROM strategy_pipeline WHERE current_phase = 'live' AND is_active = 1")
                total_live_pnl = cursor.fetchone()[0] or 0
                
                conn.close()
                
                return JSONResponse({
                    "success": True,
                    "data": {
                        "tested_today": random.randint(8, 15),  # Demo data
                        "candidates_found": random.randint(2, 5),
                        "success_rate": round(random.uniform(65, 85), 1),
                        "graduation_rate": round(random.uniform(25, 45), 1),
                        "backtest_count": backtest_count,
                        "paper_count": paper_count,
                        "live_count": live_count,
                        "total_live_pnl": round(total_live_pnl, 2),
                        "last_updated": datetime.now().isoformat()
                    }
                })
                
            except Exception as e:
                logger.error(f"Pipeline metrics error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/performance")
        async def get_performance():
            """Get performance data"""
            try:
                # Generate realistic performance data for demo
                performance_data = []
                base_value = 10000
                
                for i in range(24):  # Last 24 hours
                    timestamp = datetime.now() - timedelta(hours=23-i)
                    change = random.uniform(-0.01, 0.02)  # -1% to +2% per hour
                    base_value *= (1 + change)
                    
                    performance_data.append({
                        "timestamp": timestamp.isoformat(),
                        "strategy_id": "portfolio",
                        "returns": round(change * 100, 2),
                        "sharpe_ratio": round(random.uniform(1.5, 2.5), 2),
                        "drawdown": round(random.uniform(-5, -0.5), 2),
                        "equity": round(base_value, 2)
                    })
                
                return JSONResponse({
                    "success": True,
                    "data": performance_data
                })
                
            except Exception as e:
                logger.error(f"Performance fetch error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/activity")
        async def get_activity():
            """Get recent trading activity"""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT timestamp, symbol, side, amount, price, strategy_id, realized_pnl
                    FROM trades 
                    ORDER BY timestamp DESC 
                    LIMIT 20
                ''')
                
                activity_data = []
                for row in cursor.fetchall():
                    activity_data.append({
                        "timestamp": row[0],
                        "type": "trade",
                        "action": row[2],
                        "symbol": row[1],
                        "amount": row[3],
                        "price": row[4],
                        "strategy_id": row[5],
                        "pnl": row[6]
                    })
                
                conn.close()
                
                return JSONResponse({
                    "success": True,
                    "data": activity_data
                })
                
            except Exception as e:
                logger.error(f"Activity fetch error: {e}")
                raise HTTPException(status_code=500, detail=str(e))