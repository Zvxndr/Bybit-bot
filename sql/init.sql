-- Database initialization script for Bybit Trading Bot
-- Creates TimescaleDB extensions and indexes for optimal performance

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create application user (if not exists)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'trading_user') THEN
        CREATE ROLE trading_user WITH LOGIN PASSWORD 'secure_trading_password_change_this';
    END IF;
END
$$;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;
GRANT ALL ON SCHEMA public TO trading_user;

-- Create optimized indexes for trading queries
-- These will be created by SQLAlchemy models, but we can pre-optimize

-- Performance optimization settings for trading workloads
ALTER SYSTEM SET shared_preload_libraries = 'timescaledb';
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';

-- Enable query performance insights
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

SELECT 'Database initialized successfully for Bybit Trading Bot' as status;