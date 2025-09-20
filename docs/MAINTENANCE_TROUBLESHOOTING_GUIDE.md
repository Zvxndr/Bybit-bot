# 🔧 Maintenance & Troubleshooting Guide

## Overview

This comprehensive guide covers maintenance procedures, troubleshooting steps, and operational best practices for keeping your Bybit Trading Bot running smoothly in production. It's designed for both beginners and advanced users.

## 📋 Table of Contents

1. [Daily Maintenance](#daily-maintenance)
2. [Weekly Maintenance](#weekly-maintenance)
3. [Monthly Maintenance](#monthly-maintenance)
4. [System Monitoring](#system-monitoring)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Error Codes Reference](#error-codes-reference)
8. [Recovery Procedures](#recovery-procedures)
9. [Preventive Measures](#preventive-measures)
10. [Emergency Procedures](#emergency-procedures)

---

## 📅 Daily Maintenance

### **Morning Routine (5-10 minutes)**

#### **1. System Health Check**
```bash
# Check bot status
curl -f http://localhost:8080/health

# Check system resources
free -h
df -h
top -n 1

# Check database status
docker exec postgres pg_isready -U trading_user
```

#### **2. Trading Activity Review**
```bash
# Check overnight trades
curl http://localhost:8080/api/v1/trades/recent?hours=12

# Check current positions
curl http://localhost:8080/api/v1/positions/active

# Check account balance
curl http://localhost:8080/api/v1/account/balance
```

#### **3. Error Log Review**
```bash
# Check for errors in last 24 hours
grep "ERROR\|CRITICAL" logs/trading_bot.log | tail -20

# Check for API connection issues
grep "Connection\|Timeout\|Rate limit" logs/trading_bot.log | tail -10

# Check for strategy errors
grep "Strategy.*error\|Failed to execute" logs/trading_bot.log | tail -10
```

#### **4. Daily Health Report**
```python
# Generate automated daily report
daily_report = {
    "date": "2024-09-15",
    "system_status": "healthy",
    "uptime": "99.8%",
    "trades_executed": 12,
    "total_pnl": 145.50,
    "api_errors": 0,
    "system_errors": 0,
    "balance_change": "+1.5%",
    "active_strategies": 3
}
```

### **Midday Check (2-3 minutes)**

#### **Quick Status Verification**
```bash
# One-liner health check
curl -s http://localhost:8080/health | jq '.status'

# Check if trading is active
curl -s http://localhost:8080/api/v1/bot/status | jq '.trading_active'

# Quick balance check
curl -s http://localhost:8080/api/v1/account/balance | jq '.total_balance'
```

### **Evening Review (10-15 minutes)**

#### **1. Performance Analysis**
```bash
# Daily P&L summary
curl http://localhost:8080/api/v1/analytics/daily-pnl

# Strategy performance
curl http://localhost:8080/api/v1/analytics/strategy-performance

# Trade analysis
curl http://localhost:8080/api/v1/analytics/trade-summary
```

#### **2. Risk Assessment**
```bash
# Check risk limits
curl http://localhost:8080/api/v1/risk/current-exposure

# Review stop losses
curl http://localhost:8080/api/v1/positions/stop-losses

# Check daily loss limits
curl http://localhost:8080/api/v1/risk/daily-limits
```

#### **3. System Cleanup**
```bash
# Rotate logs if needed
if [ $(stat -c%s logs/trading_bot.log) -gt 100000000 ]; then
    mv logs/trading_bot.log logs/trading_bot.log.$(date +%Y%m%d)
    systemctl reload trading-bot
fi

# Clean temporary files
find /tmp -name "*trading*" -mtime +1 -delete
```

---

## 📊 Weekly Maintenance

### **System Updates and Optimization**

#### **1. Software Updates**
```bash
# Update system packages
sudo apt update && sudo apt list --upgradable

# Update Python packages
pip list --outdated
pip install --upgrade pip setuptools wheel

# Update Docker images
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedSince}}"
docker-compose pull
```

#### **2. Database Maintenance**
```bash
# Database statistics
docker exec postgres psql -U trading_user trading_bot_prod -c "
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows
FROM pg_stat_user_tables 
ORDER BY n_live_tup DESC;
"

# Vacuum and analyze
docker exec postgres psql -U trading_user trading_bot_prod -c "VACUUM ANALYZE;"

# Check database size
docker exec postgres psql -U trading_user trading_bot_prod -c "
SELECT 
    pg_size_pretty(pg_database_size('trading_bot_prod')) as db_size;
"
```

#### **3. Log Analysis**
```bash
# Generate weekly log summary
echo "=== Weekly Log Summary ===" > weekly_log_summary.txt
echo "Date Range: $(date -d '7 days ago' +%Y-%m-%d) to $(date +%Y-%m-%d)" >> weekly_log_summary.txt
echo "" >> weekly_log_summary.txt

# Count log levels
echo "Log Level Summary:" >> weekly_log_summary.txt
grep -c "DEBUG" logs/trading_bot.log* | head -1 >> weekly_log_summary.txt
grep -c "INFO" logs/trading_bot.log* | head -1 >> weekly_log_summary.txt
grep -c "WARNING" logs/trading_bot.log* | head -1 >> weekly_log_summary.txt
grep -c "ERROR" logs/trading_bot.log* | head -1 >> weekly_log_summary.txt

# Most common errors
echo "" >> weekly_log_summary.txt
echo "Most Common Errors:" >> weekly_log_summary.txt
grep "ERROR" logs/trading_bot.log* | cut -d':' -f4- | sort | uniq -c | sort -nr | head -10 >> weekly_log_summary.txt
```

#### **4. Performance Metrics**
```bash
# Generate performance report
curl -X POST http://localhost:8080/api/v1/reports/weekly \
  -H "Content-Type: application/json" \
  -d '{
    "include_sections": [
      "portfolio_summary",
      "strategy_performance", 
      "risk_metrics",
      "trade_analysis",
      "system_performance"
    ]
  }'
```

### **Configuration Review**

#### **1. Strategy Performance Analysis**
```python
# Analyze each strategy's performance
strategy_analysis = {
    "timeframe": "7_days",
    "metrics": [
        "total_return",
        "sharpe_ratio", 
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "avg_trade_duration"
    ]
}

# Disable underperforming strategies
underperforming_strategies = [
    strategy for strategy in active_strategies 
    if strategy.sharpe_ratio < 0.5 or strategy.max_drawdown > 15
]
```

#### **2. Risk Parameter Adjustment**
```python
# Review and adjust risk parameters based on performance
risk_adjustments = {
    "position_size": {
        "current": 2.0,
        "recommended": 1.5,  # Reduce if high volatility
        "reason": "Market volatility increased"
    },
    "daily_loss_limit": {
        "current": 5.0,
        "recommended": 4.0,
        "reason": "Recent drawdown events"
    },
    "stop_loss": {
        "current": 2.0,
        "recommended": 2.5,
        "reason": "False stops increased"
    }
}
```

---

## 📈 Monthly Maintenance

### **Comprehensive System Review**

#### **1. Full System Backup**
```bash
#!/bin/bash
# monthly_backup.sh

BACKUP_DATE=$(date +%Y%m%d)
BACKUP_DIR="/backup/monthly/$BACKUP_DATE"
mkdir -p $BACKUP_DIR

# Database backup
docker exec postgres pg_dump -U trading_user trading_bot_prod | gzip > $BACKUP_DIR/database.sql.gz

# Configuration backup
tar -czf $BACKUP_DIR/config.tar.gz config/ .env*

# Application data
tar -czf $BACKUP_DIR/application_data.tar.gz data/ logs/ reports/

# Source code
git archive --format=tar.gz --output=$BACKUP_DIR/source_code.tar.gz HEAD

# Upload to cloud storage
aws s3 sync $BACKUP_DIR s3://trading-bot-backups/monthly/$BACKUP_DATE/

echo "Monthly backup completed: $BACKUP_DIR"
```

#### **2. Security Audit**
```bash
# Check for security updates
sudo apt list --upgradable | grep -i security

# Audit API key usage
curl http://localhost:8080/api/v1/security/api-audit

# Check SSL certificate expiry
echo | openssl s_client -servername your-domain.com -connect your-domain.com:443 2>/dev/null | openssl x509 -noout -dates

# Review user access logs
sudo lastlog
sudo last | head -20
```

#### **3. Performance Optimization**
```bash
# Database optimization
docker exec postgres psql -U trading_user trading_bot_prod -c "
-- Find slow queries
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes 
ORDER BY idx_scan DESC;
"

# System resource analysis
iostat -x 1 5
sar -u 1 5
sar -r 1 5
```

#### **4. Strategy Optimization**
```python
# Monthly strategy review
monthly_optimization = {
    "backtest_period": "90_days",
    "optimization_targets": [
        "sharpe_ratio",
        "calmar_ratio", 
        "max_drawdown",
        "volatility"
    ],
    "parameter_ranges": {
        "sma_fast": [10, 15, 20, 25, 30],
        "sma_slow": [40, 50, 60, 70, 80],
        "rsi_threshold": [25, 30, 35],
        "stop_loss": [1.5, 2.0, 2.5, 3.0]
    }
}
```

---

## 📡 System Monitoring

### **Real-Time Monitoring Setup**

#### **1. Prometheus Metrics**
```yaml
# Key metrics to monitor
monitoring_metrics:
  system:
    - cpu_usage
    - memory_usage
    - disk_usage
    - network_io
  
  application:
    - active_positions
    - daily_pnl
    - api_response_time
    - trade_execution_time
    - error_rate
    
  business:
    - portfolio_value
    - drawdown
    - sharpe_ratio
    - win_rate
```

#### **2. Grafana Dashboards**
```json
{
  "dashboard": {
    "title": "Trading Bot Monitoring",
    "panels": [
      {
        "title": "System Health",
        "type": "stat",
        "targets": [
          "up{job='trading-bot'}",
          "process_resident_memory_bytes",
          "process_cpu_seconds_total"
        ]
      },
      {
        "title": "Trading Metrics", 
        "type": "graph",
        "targets": [
          "trading_bot_portfolio_value",
          "trading_bot_daily_pnl",
          "trading_bot_active_positions"
        ]
      }
    ]
  }
}
```

#### **3. Alert Rules**
```yaml
# Grafana alert rules
alerts:
  - name: "Trading Bot Down"
    condition: "up{job='trading-bot'} == 0"
    duration: "30s"
    severity: "critical"
    
  - name: "High Memory Usage"
    condition: "process_resident_memory_bytes > 2e9"
    duration: "5m"
    severity: "warning"
    
  - name: "Daily Loss Limit Approached"
    condition: "trading_bot_daily_pnl < -0.04 * trading_bot_portfolio_value"
    duration: "1m"
    severity: "critical"
    
  - name: "API Error Rate High"
    condition: "rate(trading_bot_api_errors_total[5m]) > 0.1"
    duration: "2m" 
    severity: "warning"
```

### **Log Monitoring**

#### **1. Centralized Logging**
```yaml
# Filebeat configuration
filebeat.inputs:
- type: log
  paths:
    - /opt/trading-bot/logs/*.log
  fields:
    service: trading-bot
    environment: production
  multiline:
    pattern: '^\d{4}-\d{2}-\d{2}'
    negate: true
    match: after

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "trading-bot-logs-%{+yyyy.MM.dd}"
```

#### **2. Log Analysis Queries**
```json
{
  "elasticsearch_queries": {
    "error_rate": {
      "query": {
        "bool": {
          "must": [
            {"match": {"level": "ERROR"}},
            {"range": {"@timestamp": {"gte": "now-1h"}}}
          ]
        }
      }
    },
    "api_failures": {
      "query": {
        "bool": {
          "must": [
            {"match": {"message": "API"}},
            {"match": {"level": "ERROR"}}
          ]
        }
      }
    }
  }
}
```

---

## ⚡ Performance Optimization

### **Database Optimization**

#### **1. Query Performance**
```sql
-- Enable query statistics
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Find slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements 
WHERE mean_time > 100  -- Queries taking more than 100ms
ORDER BY mean_time DESC 
LIMIT 10;

-- Create necessary indexes
CREATE INDEX CONCURRENTLY idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX CONCURRENTLY idx_trades_symbol ON trades(symbol);
CREATE INDEX CONCURRENTLY idx_price_data_symbol_timestamp ON price_data(symbol, timestamp DESC);
```

#### **2. Connection Pooling**
```python
# PostgreSQL connection pooling configuration
database_config = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_recycle": 3600,  # 1 hour
    "pool_pre_ping": True,
    "echo": False
}

# SQLAlchemy engine configuration
engine = create_engine(
    database_url,
    pool_size=database_config["pool_size"],
    max_overflow=database_config["max_overflow"],
    pool_recycle=database_config["pool_recycle"],
    pool_pre_ping=database_config["pool_pre_ping"]
)
```

### **Application Optimization**

#### **1. Memory Management**
```python
# Memory optimization techniques
import gc
import psutil
import tracemalloc

class MemoryManager:
    def __init__(self, threshold_mb=500):
        self.threshold_mb = threshold_mb
        tracemalloc.start()
    
    def check_memory_usage(self):
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.threshold_mb:
            self.cleanup_memory()
            return memory_mb
    
    def cleanup_memory(self):
        # Force garbage collection
        gc.collect()
        
        # Clear caches
        self.clear_strategy_caches()
        self.clear_data_caches()
    
    def get_memory_profile(self):
        current, peak = tracemalloc.get_traced_memory()
        return {
            "current_mb": current / 1024 / 1024,
            "peak_mb": peak / 1024 / 1024
        }
```

#### **2. Caching Strategy**
```python
# Redis caching for frequently accessed data
import redis
import json
from functools import wraps

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    password='your_redis_password',
    decode_responses=True
)

def cache_result(expiry=300):  # 5 minutes default
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiry, json.dumps(result))
            
            return result
        return wrapper
    return decorator

# Usage example
@cache_result(expiry=60)  # Cache for 1 minute
def get_market_data(symbol):
    return exchange.fetch_ticker(symbol)
```

### **Network Optimization**

#### **1. API Request Optimization**
```python
# Batch API requests and connection pooling
import aiohttp
import asyncio
from aiohttp import TCPConnector

class OptimizedExchangeClient:
    def __init__(self):
        connector = TCPConnector(
            limit=100,  # Total connection limit
            limit_per_host=30,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=10)
        )
    
    async def batch_requests(self, requests):
        """Execute multiple requests concurrently"""
        tasks = [self.make_request(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def make_request(self, request):
        async with self.session.request(**request) as response:
            return await response.json()
```

---

## 🔍 Troubleshooting Guide

### **Common Issues and Solutions**

#### **Issue 1: Bot Stops Trading Unexpectedly**

**Symptoms:**
- Bot status shows running but no new trades
- No error messages in logs
- API connectivity seems fine

**Diagnosis Steps:**
```bash
# Check bot internal state
curl http://localhost:8080/api/v1/debug/internal-state

# Check strategy states
curl http://localhost:8080/api/v1/strategies/debug

# Check risk management state
curl http://localhost:8080/api/v1/risk/debug

# Check market data flow
curl http://localhost:8080/api/v1/data/debug
```

**Common Causes and Solutions:**
1. **Risk limits hit**: Check daily loss limits, position limits
2. **Strategy conditions not met**: Review strategy parameters
3. **Insufficient balance**: Check account balance
4. **Market data issues**: Verify price feed connectivity

**Solution Example:**
```bash
# Reset risk limits if appropriate
curl -X POST http://localhost:8080/api/v1/risk/reset-daily-limits

# Restart specific strategy
curl -X POST http://localhost:8080/api/v1/strategies/restart \
  -d '{"strategy_name": "mean_reversion_btc"}'
```

#### **Issue 2: High Memory Usage**

**Symptoms:**
- System becomes slow or unresponsive
- Out of memory errors
- Memory usage continuously increasing

**Diagnosis Steps:**
```bash
# Check memory usage by process
ps aux --sort=-%mem | head -20

# Check memory usage over time
sar -r 1 10

# Check for memory leaks in bot
curl http://localhost:8080/api/v1/debug/memory-usage
```

**Solutions:**
```python
# 1. Implement memory limits
resource_limits = {
    "max_memory_mb": 1000,
    "cleanup_threshold": 800,
    "gc_frequency": 300  # seconds
}

# 2. Clear data caches regularly
def cleanup_old_data():
    # Remove old price data
    session.execute(
        "DELETE FROM price_data WHERE timestamp < :cutoff",
        {"cutoff": datetime.now() - timedelta(days=30)}
    )
    
    # Clear strategy caches
    for strategy in active_strategies:
        strategy.clear_cache()

# 3. Restart bot if memory usage too high
if memory_usage_mb > resource_limits["max_memory_mb"]:
    restart_bot_gracefully()
```

#### **Issue 3: Database Connection Errors**

**Symptoms:**
- "connection refused" errors
- Timeouts on database queries
- Data not being saved

**Diagnosis Steps:**
```bash
# Check database status
docker exec postgres pg_isready -U trading_user

# Check database connections
docker exec postgres psql -U trading_user trading_bot_prod -c "
SELECT pid, usename, application_name, client_addr, state
FROM pg_stat_activity 
WHERE datname = 'trading_bot_prod';
"

# Check database locks
docker exec postgres psql -U trading_user trading_bot_prod -c "
SELECT blocked_locks.pid AS blocked_pid,
       blocked_activity.usename AS blocked_user,
       blocking_locks.pid AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query AS blocked_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
"
```

**Solutions:**
```bash
# 1. Restart database if needed
docker-compose restart postgres

# 2. Kill long-running queries
docker exec postgres psql -U trading_user trading_bot_prod -c "
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE datname = 'trading_bot_prod' 
AND state = 'active' 
AND query_start < now() - interval '5 minutes';
"

# 3. Increase connection limits
# Edit postgresql.conf:
# max_connections = 200
# shared_buffers = 256MB
```

#### **Issue 4: API Rate Limiting**

**Symptoms:**
- "Rate limit exceeded" errors
- Delayed trade executions
- API timeout errors

**Diagnosis Steps:**
```bash
# Check API call frequency
grep "Rate limit\|429\|Too many requests" logs/trading_bot.log | tail -20

# Check API usage statistics
curl http://localhost:8080/api/v1/debug/api-stats
```

**Solutions:**
```python
# 1. Implement intelligent rate limiting
class RateLimiter:
    def __init__(self, max_requests=1200, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def wait_if_needed(self):
        now = datetime.now()
        
        # Remove old requests
        self.requests = [
            req_time for req_time in self.requests 
            if (now - req_time).seconds < self.time_window
        ]
        
        # Check if we need to wait
        if len(self.requests) >= self.max_requests:
            wait_time = self.time_window - (now - self.requests[0]).seconds
            await asyncio.sleep(wait_time)
        
        self.requests.append(now)

# 2. Batch API calls
async def batch_market_data_requests():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    # Instead of individual calls, use batch endpoint
    return await exchange.fetch_tickers(symbols)

# 3. Cache frequently requested data
@cache_result(expiry=30)  # Cache for 30 seconds
def get_ticker_data(symbol):
    return exchange.fetch_ticker(symbol)
```

### **Performance Issues**

#### **Slow Trade Execution**

**Diagnosis:**
```bash
# Check trade execution times
curl http://localhost:8080/api/v1/debug/execution-times

# Check system latency
ping api.bybit.com
traceroute api.bybit.com
```

**Solutions:**
1. **Optimize order placement logic**
2. **Use market orders for urgent executions**
3. **Implement pre-positioned orders**
4. **Consider VPS closer to exchange servers**

#### **High CPU Usage**

**Diagnosis:**
```bash
# Check CPU usage by function
py-spy top --pid $(pgrep -f "python main.py")

# Profile CPU usage
py-spy record -o profile.svg --duration 60 --pid $(pgrep -f "python main.py")
```

**Solutions:**
```python
# 1. Optimize calculations
import numba

@numba.jit(nopython=True)
def calculate_moving_average(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

# 2. Use vectorized operations
import numpy as np
import pandas as pd

# Instead of loops, use pandas vectorized operations
df['sma'] = df['close'].rolling(window=20).mean()
df['signal'] = np.where(df['close'] > df['sma'], 1, -1)

# 3. Implement calculation caching
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(symbol, timeframe, lookback):
    # Expensive computation here
    return result
```

---

## 📚 Error Codes Reference

### **System Error Codes (10000-19999)**

| Code | Description | Severity | Action Required |
|------|-------------|----------|-----------------|
| 10001 | Configuration file not found | Critical | Check config file path |
| 10002 | Invalid configuration format | Critical | Validate JSON syntax |
| 10003 | Database connection failed | Critical | Check database status |
| 10004 | Redis connection failed | Warning | Check Redis server |
| 10005 | API key validation failed | Critical | Verify API credentials |
| 10006 | Insufficient system resources | Warning | Check memory/CPU usage |
| 10007 | Log file write permission denied | Warning | Check file permissions |
| 10008 | Strategy initialization failed | Warning | Check strategy configuration |
| 10009 | Emergency stop activated | Critical | Manual intervention required |
| 10010 | Daily loss limit exceeded | Warning | Review risk settings |

### **Trading Error Codes (20000-29999)**

| Code | Description | Severity | Action Required |
|------|-------------|----------|-----------------|
| 20001 | Insufficient balance | Warning | Check account balance |
| 20002 | Invalid symbol | Error | Verify symbol format |
| 20003 | Invalid order size | Error | Check minimum order size |
| 20004 | Market closed | Info | Wait for market open |
| 20005 | Position limit reached | Warning | Review position limits |
| 20006 | Stop loss execution failed | Error | Manual position management |
| 20007 | Order placement timeout | Warning | Check API connectivity |
| 20008 | Price outside valid range | Error | Check price parameters |
| 20009 | Risk management violation | Warning | Review risk rules |
| 20010 | Strategy signal conflict | Warning | Review strategy logic |

### **API Error Codes (30000-39999)**

| Code | Description | Severity | Action Required |
|------|-------------|----------|-----------------|
| 30001 | API rate limit exceeded | Warning | Reduce request frequency |
| 30002 | API authentication failed | Critical | Check API credentials |
| 30003 | API server unavailable | Error | Wait and retry |
| 30004 | Invalid API request format | Error | Check request parameters |
| 30005 | API response timeout | Warning | Check network connectivity |
| 30006 | API maintenance mode | Info | Wait for maintenance completion |
| 30007 | API version deprecated | Warning | Update API version |
| 30008 | Account suspended | Critical | Contact exchange support |
| 30009 | IP address blocked | Critical | Check IP whitelist |
| 30010 | API key permissions insufficient | Error | Update API permissions |

### **Data Error Codes (40000-49999)**

| Code | Description | Severity | Action Required |
|------|-------------|----------|-----------------|
| 40001 | Missing price data | Warning | Refresh data feed |
| 40002 | Data timestamp inconsistency | Error | Check data integrity |
| 40003 | Corrupted data detected | Error | Re-download data |
| 40004 | Data source unavailable | Warning | Check data provider |
| 40005 | Historical data incomplete | Warning | Download missing data |
| 40006 | Real-time feed disconnected | Error | Reconnect data feed |
| 40007 | Data validation failed | Error | Check data quality |
| 40008 | Database write failed | Error | Check database status |
| 40009 | Data cache expired | Info | Refresh cache |
| 40010 | Data format conversion error | Error | Check data parser |

---

## 🚨 Recovery Procedures

### **Graceful Bot Restart**

```bash
#!/bin/bash
# graceful_restart.sh

echo "Starting graceful bot restart..."

# 1. Stop accepting new trades
curl -X POST http://localhost:8080/api/v1/bot/pause

# 2. Wait for current operations to complete
sleep 30

# 3. Save current state
curl -X POST http://localhost:8080/api/v1/state/save

# 4. Stop bot
curl -X POST http://localhost:8080/api/v1/bot/stop

# 5. Wait for clean shutdown
sleep 10

# 6. Start bot
docker-compose restart trading-bot

# 7. Wait for initialization
sleep 60

# 8. Verify restart
if curl -f http://localhost:8080/health; then
    echo "✅ Graceful restart successful"
    
    # 9. Resume trading
    curl -X POST http://localhost:8080/api/v1/bot/resume
else
    echo "❌ Restart failed, check logs"
    exit 1
fi
```

### **Database Recovery**

#### **Scenario 1: Database Corruption**
```bash
#!/bin/bash
# database_recovery.sh

echo "Starting database recovery..."

# 1. Stop trading bot
docker-compose stop trading-bot

# 2. Backup corrupted database
docker exec postgres pg_dump -U trading_user trading_bot_prod > corrupted_backup.sql

# 3. Drop corrupted database
docker exec postgres psql -U postgres -c "DROP DATABASE trading_bot_prod;"

# 4. Restore from latest backup
LATEST_BACKUP=$(ls -t /backup/database_*.sql.gz | head -1)
gunzip -c $LATEST_BACKUP | docker exec -i postgres psql -U postgres -c "CREATE DATABASE trading_bot_prod;"
gunzip -c $LATEST_BACKUP | docker exec -i postgres psql -U trading_user trading_bot_prod

# 5. Verify database integrity
docker exec postgres psql -U trading_user trading_bot_prod -c "SELECT COUNT(*) FROM trades;"

# 6. Restart bot
docker-compose start trading-bot

echo "Database recovery completed"
```

#### **Scenario 2: Data Loss Recovery**
```bash
# Recover missing trade data from exchange
python scripts/recover_trade_history.py \
    --start-date 2024-09-01 \
    --end-date 2024-09-15 \
    --symbols BTCUSDT,ETHUSDT

# Reconcile account balance
python scripts/reconcile_balance.py \
    --exchange-balance 10000.50 \
    --database-balance 9950.25
```

### **Emergency Shutdown Procedure**

```bash
#!/bin/bash
# emergency_shutdown.sh

echo "🚨 EMERGENCY SHUTDOWN INITIATED"

# 1. Immediate trading halt
curl -X POST http://localhost:8080/api/v1/emergency/stop-all-trading

# 2. Close all positions (if safe to do so)
read -p "Close all positions? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    curl -X POST http://localhost:8080/api/v1/positions/close-all
fi

# 3. Save current state
curl -X POST http://localhost:8080/api/v1/state/emergency-save

# 4. Stop all services
docker-compose down

# 5. Disable API keys (manual step)
echo "🔴 MANUAL ACTION REQUIRED:"
echo "1. Log into Bybit.com"
echo "2. Go to API Management"
echo "3. Disable all trading bot API keys"
echo "4. Verify no active orders remain"

# 6. Send emergency notification
curl -X POST "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_SID/Messages.json" \
    -u "$TWILIO_SID:$TWILIO_TOKEN" \
    -d "From=$TWILIO_FROM" \
    -d "To=$EMERGENCY_PHONE" \
    -d "Body=TRADING BOT EMERGENCY SHUTDOWN ACTIVATED"

echo "🚨 Emergency shutdown completed"
```

---

## 🛡️ Preventive Measures

### **Automated Health Checks**

```python
# health_monitor.py
import asyncio
import aiohttp
from datetime import datetime, timedelta

class HealthMonitor:
    def __init__(self):
        self.health_checks = [
            self.check_bot_status,
            self.check_database_connection,
            self.check_api_connectivity,
            self.check_system_resources,
            self.check_trading_activity
        ]
    
    async def run_health_checks(self):
        results = {}
        
        for check in self.health_checks:
            try:
                result = await check()
                results[check.__name__] = result
            except Exception as e:
                results[check.__name__] = {"status": "error", "error": str(e)}
        
        # Generate health report
        await self.generate_health_report(results)
        
        # Send alerts if needed
        await self.send_alerts_if_needed(results)
        
        return results
    
    async def check_bot_status(self):
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8080/health") as response:
                data = await response.json()
                return {
                    "status": "healthy" if data.get("status") == "ok" else "unhealthy",
                    "uptime": data.get("uptime"),
                    "last_trade": data.get("last_trade_time")
                }
    
    async def check_system_resources(self):
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy" if cpu_percent < 80 and memory.percent < 80 else "warning",
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent
        }
    
    async def send_alerts_if_needed(self, results):
        unhealthy_checks = [
            name for name, result in results.items() 
            if result.get("status") in ["unhealthy", "error"]
        ]
        
        if unhealthy_checks:
            alert_message = f"Health check failures: {', '.join(unhealthy_checks)}"
            await self.send_alert(alert_message, severity="warning")

# Run health checks every 5 minutes
async def main():
    monitor = HealthMonitor()
    while True:
        await monitor.run_health_checks()
        await asyncio.sleep(300)  # 5 minutes

if __name__ == "__main__":
    asyncio.run(main())
```

### **Predictive Maintenance**

```python
# predictive_maintenance.py
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

class PredictiveMaintenance:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.metrics_history = []
    
    def collect_metrics(self):
        """Collect system and trading metrics"""
        metrics = {
            "timestamp": datetime.now(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "api_response_time": self.measure_api_response_time(),
            "trade_execution_time": self.measure_trade_execution_time(),
            "error_rate": self.calculate_error_rate(),
            "throughput": self.calculate_throughput()
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 data points
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def detect_anomalies(self):
        """Detect anomalies in system behavior"""
        if len(self.metrics_history) < 50:
            return None
        
        # Prepare data for anomaly detection
        features = np.array([
            [m["cpu_usage"], m["memory_usage"], m["api_response_time"], 
             m["error_rate"], m["throughput"]]
            for m in self.metrics_history[-100:]  # Last 100 data points
        ])
        
        # Detect anomalies
        anomalies = self.anomaly_detector.fit_predict(features)
        
        # Return recent anomalies
        recent_anomalies = []
        for i, is_anomaly in enumerate(anomalies[-10:]):  # Last 10 points
            if is_anomaly == -1:
                recent_anomalies.append(self.metrics_history[-(10-i)])
        
        return recent_anomalies
    
    def generate_maintenance_recommendations(self):
        """Generate maintenance recommendations based on metrics"""
        recommendations = []
        
        if len(self.metrics_history) < 10:
            return recommendations
        
        recent_metrics = self.metrics_history[-10:]
        
        # Check for performance degradation
        avg_cpu = np.mean([m["cpu_usage"] for m in recent_metrics])
        avg_memory = np.mean([m["memory_usage"] for m in recent_metrics])
        avg_response_time = np.mean([m["api_response_time"] for m in recent_metrics])
        
        if avg_cpu > 70:
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "message": "High CPU usage detected. Consider optimizing algorithms or scaling up resources."
            })
        
        if avg_memory > 80:
            recommendations.append({
                "type": "memory",
                "priority": "high", 
                "message": "High memory usage detected. Consider memory cleanup or increasing RAM."
            })
        
        if avg_response_time > 1000:  # 1 second
            recommendations.append({
                "type": "network",
                "priority": "medium",
                "message": "Slow API response times. Check network connectivity or consider using VPS closer to exchange."
            })
        
        return recommendations
```

### **Configuration Validation**

```python
# config_validator.py
import json
import jsonschema
from jsonschema import validate

class ConfigValidator:
    def __init__(self):
        self.schema = self.load_config_schema()
    
    def load_config_schema(self):
        """Load configuration schema for validation"""
        return {
            "type": "object",
            "properties": {
                "trading": {
                    "type": "object",
                    "properties": {
                        "max_position_size": {"type": "number", "minimum": 0.01, "maximum": 10000},
                        "daily_loss_limit": {"type": "number", "minimum": 0.1, "maximum": 50},
                        "max_positions": {"type": "integer", "minimum": 1, "maximum": 20},
                        "stop_loss_percent": {"type": "number", "minimum": 0.1, "maximum": 20}
                    },
                    "required": ["max_position_size", "daily_loss_limit"]
                },
                "strategies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "enabled": {"type": "boolean"},
                            "parameters": {"type": "object"}
                        },
                        "required": ["name", "enabled"]
                    }
                }
            },
            "required": ["trading", "strategies"]
        }
    
    def validate_config(self, config_path):
        """Validate configuration file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate against schema
            validate(instance=config, schema=self.schema)
            
            # Additional business logic validation
            validation_errors = self.validate_business_rules(config)
            
            if validation_errors:
                return {
                    "valid": False,
                    "errors": validation_errors
                }
            
            return {
                "valid": True,
                "message": "Configuration is valid"
            }
            
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "errors": [f"Invalid JSON format: {str(e)}"]
            }
        except jsonschema.exceptions.ValidationError as e:
            return {
                "valid": False,
                "errors": [f"Schema validation error: {str(e)}"]
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }
    
    def validate_business_rules(self, config):
        """Validate business-specific rules"""
        errors = []
        
        # Check risk management rules
        if config["trading"]["daily_loss_limit"] > config["trading"]["max_position_size"] * 5:
            errors.append("Daily loss limit should not exceed 5x max position size")
        
        # Check strategy configuration
        enabled_strategies = [s for s in config["strategies"] if s["enabled"]]
        if len(enabled_strategies) == 0:
            errors.append("At least one strategy must be enabled")
        
        # Check for conflicting strategies
        strategy_symbols = []
        for strategy in enabled_strategies:
            symbol = strategy.get("parameters", {}).get("symbol")
            if symbol in strategy_symbols:
                errors.append(f"Multiple strategies targeting same symbol: {symbol}")
            if symbol:
                strategy_symbols.append(symbol)
        
        return errors

# Usage
validator = ConfigValidator()
result = validator.validate_config("config/config.json")
if not result["valid"]:
    print("Configuration errors:", result["errors"])
```

---

**This comprehensive maintenance and troubleshooting guide provides all the tools and procedures needed to keep your trading bot running smoothly in production. Regular maintenance and proactive monitoring are key to successful automated trading operations.**

*Last Updated: September 2025*
*Version: 1.0.0*