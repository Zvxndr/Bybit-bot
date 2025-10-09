"""
Database manager for the trading bot.

This module provides database connection management, session handling,
and database operations with support for both DuckDB (development)
and PostgreSQL (production).
"""

import os
from contextlib import contextmanager
from typing import Dict, Generator, Optional, Union, Any
from urllib.parse import quote_plus

from loguru import logger
from sqlalchemy import create_engine, event, MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base
from ..config import DatabaseConfig


class DatabaseManager:
    """
    Manages database connections and operations.
    
    Supports both DuckDB for development and PostgreSQL for production
    with automatic connection pooling and session management.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._environment = os.getenv('ENVIRONMENT', 'development')
    
    def initialize(self) -> None:
        """Initialize database connection and create tables."""
        try:
            # Create engine based on environment
            self.engine = self._create_engine()
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create all tables
            self._create_tables()
            
            logger.info(f"Database initialized successfully ({self._environment})")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine based on configuration."""
        if self._environment == 'production':
            return self._create_postgresql_engine()
        else:
            # Check if using SQLite or DuckDB
            config = self.config.development
            dialect = config.get('dialect', 'duckdb')
            
            if dialect == 'sqlite':
                return self._create_sqlite_engine()
            else:
                return self._create_duckdb_engine()
    
    def _create_postgresql_engine(self) -> Engine:
        """Create PostgreSQL engine for production."""
        config = self.config.production
        
        # Build connection URL
        password = quote_plus(str(config.get('password', '')))
        url = (
            f"postgresql://{config['user']}:{password}@"
            f"{config['host']}:{config['port']}/{config['name']}"
        )
        
        # Create engine with connection pooling
        engine = create_engine(
            url,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            echo=self.config.echo,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections every hour
        )
        
        # Add connection event listeners
        self._setup_postgresql_events(engine)
        
        return engine
    
    def _create_duckdb_engine(self) -> Engine:
        """Create DuckDB engine for development."""
        config = self.config.development
        db_path = config.get('path', './data/trading_bot.db')
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(str(db_path)), exist_ok=True)
        
        # Create DuckDB engine
        url = f"duckdb:///{db_path}"
        engine = create_engine(
            url,
            echo=self.config.echo,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,  # Allow multiple threads
            }
        )
        
        # Setup DuckDB-specific configurations
        self._setup_duckdb_events(engine)
        
        return engine
    
    def _create_sqlite_engine(self) -> Engine:
        """Create SQLite engine for development."""
        config = self.config.development
        db_path = config.get('path', './data/trading_bot.db')
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(str(db_path)), exist_ok=True)
        
        # Create SQLite engine
        url = f"sqlite:///{db_path}"
        engine = create_engine(
            url,
            echo=self.config.echo,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,  # Allow multiple threads
            }
        )
        
        # Setup SQLite-specific configurations
        self._setup_sqlite_events(engine)
        
        return engine
    
    def _setup_postgresql_events(self, engine: Engine) -> None:
        """Setup PostgreSQL-specific event listeners."""
        
        @event.listens_for(engine, "connect")
        def set_postgresql_options(dbapi_connection, connection_record):
            """Set PostgreSQL connection options."""
            with dbapi_connection.cursor() as cursor:
                # Set timezone to UTC
                cursor.execute("SET timezone TO 'UTC'")
                
                # Optimize for our workload
                cursor.execute("SET work_mem = '256MB'")
                cursor.execute("SET maintenance_work_mem = '512MB'")
    
    def _setup_duckdb_events(self, engine: Engine) -> None:
        """Setup DuckDB-specific event listeners."""
        
        @event.listens_for(engine, "connect")
        def set_duckdb_options(dbapi_connection, connection_record):
            """Set DuckDB connection options."""
            cursor = dbapi_connection.cursor()
            
            # Set memory limit (adjust based on available RAM)
            cursor.execute("SET memory_limit='2GB'")
            
            # Enable parallel processing
            cursor.execute("SET threads=4")
            
            # Optimize for analytical workloads
            cursor.execute("SET enable_progress_bar=true")
            cursor.execute("SET enable_object_cache=true")
    
    def _setup_sqlite_events(self, engine: Engine) -> None:
        """Setup SQLite-specific event listeners."""
        
        @event.listens_for(engine, "connect")
        def set_sqlite_options(dbapi_connection, connection_record):
            """Set SQLite connection options."""
            cursor = dbapi_connection.cursor()
            
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            
            # Set synchronous mode for better performance
            cursor.execute("PRAGMA synchronous=NORMAL")
            
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys=ON")
            
            # Set cache size (negative value = KB)
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB
    
    def _create_tables(self) -> None:
        """Create all database tables."""
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        
        # Create all tables defined in models
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session with automatic cleanup.
        
        Usage:
            with db_manager.get_session() as session:
                # Use session here
                pass
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute raw SQL query."""
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        
        with self.engine.connect() as connection:
            result = connection.execute(query, params or {})
            return result.fetchall()
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get information about a database table."""
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        
        metadata = MetaData()
        metadata.reflect(bind=self.engine)
        
        if table_name not in metadata.tables:
            raise ValueError(f"Table '{table_name}' not found")
        
        table = metadata.tables[table_name]
        return {
            'columns': [str(col) for col in table.columns],
            'indexes': [str(idx) for idx in table.indexes],
            'constraints': [str(const) for const in table.constraints]
        }
    
    def backup_database(self, backup_path: str) -> None:
        """Create database backup."""
        if self._environment == 'development':
            # For DuckDB, simply copy the file
            import shutil
            config = self.config.development
            db_path = config.get('path', './data/trading_bot.db')
            shutil.copy2(str(db_path), str(backup_path))
            logger.info(f"DuckDB backup created: {backup_path}")
        else:
            # For PostgreSQL, use pg_dump
            config = self.config.production
            import subprocess
            
            cmd = [
                'pg_dump',
                '-h', config['host'],
                '-p', str(config['port']),
                '-U', config['user'],
                '-d', config['name'],
                '-f', backup_path
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = str(config['password'])
            
            subprocess.run(cmd, env=env, check=True)
            logger.info(f"PostgreSQL backup created: {backup_path}")
    
    def reset_database(self, confirm: bool = False) -> None:
        """
        Reset database by dropping and recreating all tables.
        
        WARNING: This will delete all data!
        """
        if not confirm:
            raise ValueError("Must explicitly confirm database reset")
        
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        
        logger.warning("Resetting database - all data will be lost!")
        
        # Drop all tables
        Base.metadata.drop_all(bind=self.engine)
        
        # Recreate tables
        self._create_tables()
        
        logger.info("Database reset completed")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get current database connection information."""
        if not self.engine:
            return {"status": "not_initialized"}
        
        return {
            "status": "connected",
            "environment": self._environment,
            "dialect": self.engine.dialect.name,
            "url": str(self.engine.url).replace(str(self.engine.url.password), "***"),
            "pool_size": getattr(self.engine.pool, 'size', None),
            "checked_in": getattr(self.engine.pool, 'checkedin', None),
            "checked_out": getattr(self.engine.pool, 'checkedout', None),
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            if not self.engine:
                return {"status": "error", "message": "Engine not initialized"}
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute("SELECT 1")
                result.fetchone()
            
            # Get connection info
            info = self.get_connection_info()
            
            return {
                "status": "healthy",
                "connection_info": info,
                "timestamp": "now()"
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": "now()"
            }
    
    def close(self) -> None:
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


class DatabaseSession:
    """
    Context manager for database sessions.
    
    Provides a more convenient interface for database operations
    with automatic session management.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.session: Optional[Session] = None
    
    def __enter__(self) -> Session:
        """Enter context manager and return session."""
        if not self.db_manager.SessionLocal:
            raise RuntimeError("Database not initialized")
        
        self.session = self.db_manager.SessionLocal()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager with proper cleanup."""
        if self.session:
            if exc_type is None:
                # No exception, commit the transaction
                try:
                    self.session.commit()
                except Exception as e:
                    logger.error(f"Failed to commit transaction: {e}")
                    self.session.rollback()
                    raise
            else:
                # Exception occurred, rollback
                self.session.rollback()
            
            self.session.close()