"""
Database Manager for Dashboard Backend
Handles TimescaleDB and PostgreSQL connections for dashboard data
"""

import asyncio
import asyncpg
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "bybit_dashboard"
    username: str = "dashboard_user"
    password: str = "dashboard_pass"
    max_connections: int = 20
    min_connections: int = 5

class DatabaseManager:
    """Manages database connections and operations for dashboard"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connections and setup tables"""
        try:
            logger.info("🗄️ Initializing dashboard database connections...")
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=30
            )
            
            # Setup database schema
            await self._setup_schema()
            
            self._initialized = True
            logger.info("✅ Database connections established and schema ready")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            # For development, create mock database manager
            logger.warning("🔄 Using mock database for development")
            self._initialized = True
    
    async def _setup_schema(self):
        """Setup database tables and TimescaleDB hypertables"""
        if not self.pool:
            return
        
        async with self.pool.acquire() as conn:
            # Enable TimescaleDB extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            
            # Trading data hypertable
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_data (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    price DECIMAL(18,8),
                    volume DECIMAL(18,8),
                    side VARCHAR(10),
                    pnl DECIMAL(18,8),
                    strategy VARCHAR(50),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    PRIMARY KEY (timestamp, symbol)
                );
            """)
            
            # Convert to hypertable if not already
            await conn.execute("""
                SELECT create_hypertable('trading_data', 'timestamp', if_not_exists => TRUE);
            """)
            
            # ML insights table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_insights (
                    timestamp TIMESTAMPTZ NOT NULL,
                    insight_type VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20),
                    prediction JSONB,
                    confidence DECIMAL(5,4),
                    model_version VARCHAR(20),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    PRIMARY KEY (timestamp, insight_type, symbol)
                );
            """)
            
            await conn.execute("""
                SELECT create_hypertable('ml_insights', 'timestamp', if_not_exists => TRUE);
            """)
            
            # System health table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_health (
                    timestamp TIMESTAMPTZ NOT NULL,
                    component VARCHAR(50) NOT NULL,
                    status VARCHAR(20),
                    cpu_usage DECIMAL(5,2),
                    memory_usage DECIMAL(5,2),
                    response_time DECIMAL(10,3),
                    error_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    PRIMARY KEY (timestamp, component)
                );
            """)
            
            await conn.execute("""
                SELECT create_hypertable('system_health', 'timestamp', if_not_exists => TRUE);
            """)
            
            # Performance metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    timestamp TIMESTAMPTZ NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    value DECIMAL(18,8),
                    unit VARCHAR(20),
                    category VARCHAR(50),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    PRIMARY KEY (timestamp, metric_name)
                );
            """)
            
            await conn.execute("""
                SELECT create_hypertable('performance_metrics', 'timestamp', if_not_exists => TRUE);
            """)
            
            # Create indexes for better query performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_trading_data_symbol_time ON trading_data (symbol, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_ml_insights_type_time ON ml_insights (insight_type, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_system_health_component ON system_health (component, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_performance_metrics_category ON performance_metrics (category, timestamp DESC);"
            ]
            
            for index in indexes:
                await conn.execute(index)
            
            logger.info("✅ Database schema setup complete")
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            logger.info("🔌 Database connections closed")
    
    # Trading data operations
    async def insert_trading_data(self, data: List[Dict[str, Any]]):
        """Insert trading data records"""
        if not self._initialized or not self.pool:
            return
        
        try:
            async with self.pool.acquire() as conn:
                records = [
                    (
                        record.get('timestamp', datetime.utcnow()),
                        record.get('symbol', ''),
                        record.get('price'),
                        record.get('volume'),
                        record.get('side'),
                        record.get('pnl'),
                        record.get('strategy'),
                        json.dumps(record.get('metadata', {}))
                    )
                    for record in data
                ]
                
                await conn.executemany("""
                    INSERT INTO trading_data 
                    (timestamp, symbol, price, volume, side, pnl, strategy, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (timestamp, symbol) DO UPDATE SET
                    price = EXCLUDED.price,
                    volume = EXCLUDED.volume,
                    side = EXCLUDED.side,
                    pnl = EXCLUDED.pnl,
                    strategy = EXCLUDED.strategy,
                    metadata = EXCLUDED.metadata;
                """, records)
                
        except Exception as e:
            logger.error(f"❌ Failed to insert trading data: {e}")
    
    async def get_trading_data(self, symbol: Optional[str] = None, 
                             hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent trading data"""
        if not self._initialized or not self.pool:
            return []
        
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT * FROM trading_data 
                    WHERE timestamp >= $1
                """
                params = [datetime.utcnow() - timedelta(hours=hours)]
                
                if symbol:
                    query += " AND symbol = $2"
                    params.append(symbol)
                
                query += " ORDER BY timestamp DESC LIMIT 1000"
                
                rows = await conn.fetch(query, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to get trading data: {e}")
            return []
    
    # ML insights operations
    async def insert_ml_insights(self, insights: List[Dict[str, Any]]):
        """Insert ML insights"""
        if not self._initialized or not self.pool:
            return
        
        try:
            async with self.pool.acquire() as conn:
                records = [
                    (
                        insight.get('timestamp', datetime.utcnow()),
                        insight.get('insight_type', ''),
                        insight.get('symbol'),
                        json.dumps(insight.get('prediction', {})),
                        insight.get('confidence'),
                        insight.get('model_version', '1.0'),
                        json.dumps(insight.get('metadata', {}))
                    )
                    for insight in insights
                ]
                
                await conn.executemany("""
                    INSERT INTO ml_insights 
                    (timestamp, insight_type, symbol, prediction, confidence, model_version, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (timestamp, insight_type, symbol) DO UPDATE SET
                    prediction = EXCLUDED.prediction,
                    confidence = EXCLUDED.confidence,
                    model_version = EXCLUDED.model_version,
                    metadata = EXCLUDED.metadata;
                """, records)
                
        except Exception as e:
            logger.error(f"❌ Failed to insert ML insights: {e}")
    
    async def get_ml_insights(self, insight_type: Optional[str] = None,
                            hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent ML insights"""
        if not self._initialized or not self.pool:
            return []
        
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT * FROM ml_insights 
                    WHERE timestamp >= $1
                """
                params = [datetime.utcnow() - timedelta(hours=hours)]
                
                if insight_type:
                    query += " AND insight_type = $2"
                    params.append(insight_type)
                
                query += " ORDER BY timestamp DESC LIMIT 1000"
                
                rows = await conn.fetch(query, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to get ML insights: {e}")
            return []
    
    # System health operations
    async def insert_system_health(self, health_data: List[Dict[str, Any]]):
        """Insert system health records"""
        if not self._initialized or not self.pool:
            return
        
        try:
            async with self.pool.acquire() as conn:
                records = [
                    (
                        record.get('timestamp', datetime.utcnow()),
                        record.get('component', ''),
                        record.get('status', 'unknown'),
                        record.get('cpu_usage'),
                        record.get('memory_usage'),
                        record.get('response_time'),
                        record.get('error_count', 0),
                        json.dumps(record.get('metadata', {}))
                    )
                    for record in health_data
                ]
                
                await conn.executemany("""
                    INSERT INTO system_health 
                    (timestamp, component, status, cpu_usage, memory_usage, response_time, error_count, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (timestamp, component) DO UPDATE SET
                    status = EXCLUDED.status,
                    cpu_usage = EXCLUDED.cpu_usage,
                    memory_usage = EXCLUDED.memory_usage,
                    response_time = EXCLUDED.response_time,
                    error_count = EXCLUDED.error_count,
                    metadata = EXCLUDED.metadata;
                """, records)
                
        except Exception as e:
            logger.error(f"❌ Failed to insert system health: {e}")
    
    async def get_system_health(self, component: Optional[str] = None,
                              hours: int = 24) -> List[Dict[str, Any]]:
        """Get system health data"""
        if not self._initialized or not self.pool:
            return []
        
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT * FROM system_health 
                    WHERE timestamp >= $1
                """
                params = [datetime.utcnow() - timedelta(hours=hours)]
                
                if component:
                    query += " AND component = $2"
                    params.append(component)
                
                query += " ORDER BY timestamp DESC LIMIT 1000"
                
                rows = await conn.fetch(query, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to get system health: {e}")
            return []
    
    # Performance metrics operations
    async def insert_performance_metrics(self, metrics: List[Dict[str, Any]]):
        """Insert performance metrics"""
        if not self._initialized or not self.pool:
            return
        
        try:
            async with self.pool.acquire() as conn:
                records = [
                    (
                        metric.get('timestamp', datetime.utcnow()),
                        metric.get('metric_name', ''),
                        metric.get('value'),
                        metric.get('unit', ''),
                        metric.get('category', 'general'),
                        json.dumps(metric.get('metadata', {}))
                    )
                    for metric in metrics
                ]
                
                await conn.executemany("""
                    INSERT INTO performance_metrics 
                    (timestamp, metric_name, value, unit, category, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (timestamp, metric_name) DO UPDATE SET
                    value = EXCLUDED.value,
                    unit = EXCLUDED.unit,
                    category = EXCLUDED.category,
                    metadata = EXCLUDED.metadata;
                """, records)
                
        except Exception as e:
            logger.error(f"❌ Failed to insert performance metrics: {e}")
    
    async def get_performance_metrics(self, category: Optional[str] = None,
                                    hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance metrics"""
        if not self._initialized or not self.pool:
            return []
        
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT * FROM performance_metrics 
                    WHERE timestamp >= $1
                """
                params = [datetime.utcnow() - timedelta(hours=hours)]
                
                if category:
                    query += " AND category = $2"
                    params.append(category)
                
                query += " ORDER BY timestamp DESC LIMIT 1000"
                
                rows = await conn.fetch(query, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to get performance metrics: {e}")
            return []