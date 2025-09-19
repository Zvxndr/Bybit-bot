# Trading Bot Production Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation Guide](#installation-guide)
4. [Configuration Reference](#configuration-reference)
5. [API Documentation](#api-documentation)
6. [Deployment Guide](#deployment-guide)
7. [Monitoring & Health Checks](#monitoring--health-checks)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)
10. [Operational Procedures](#operational-procedures)
11. [Security Guidelines](#security-guidelines)
12. [Backup & Recovery](#backup--recovery)

---

## System Overview

The Trading Bot System is a comprehensive, enterprise-grade automated trading platform built with Python 3.11+. It integrates multiple components to provide a complete trading solution with advanced risk management, backtesting, monitoring, and reporting capabilities.

### Key Features

- **Advanced Trading Engine**: Multi-exchange support with sophisticated strategy execution
- **Risk Management**: Real-time risk assessment and position sizing
- **Backtesting Framework**: Comprehensive historical strategy validation
- **Monitoring System**: Real-time health monitoring and alerting
- **Tax Reporting**: Automated tax calculation and reporting
- **Advanced Features**: ML-powered optimization and portfolio management
- **API Interface**: REST API and WebSocket for system control
- **Performance Testing**: Comprehensive load testing and optimization

### System Requirements

#### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 11+
- **CPU**: 4 cores, 2.5 GHz
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **Network**: Stable internet connection (low latency preferred)

#### Recommended Requirements
- **OS**: Linux (Ubuntu 22.04 LTS)
- **CPU**: 8 cores, 3.0 GHz+
- **RAM**: 16 GB+
- **Storage**: 100 GB SSD
- **Network**: Dedicated connection with <50ms latency to exchanges

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Trading Bot System                     │
├─────────────────────────────────────────────────────────────┤
│  API Layer (REST/WebSocket)                                │
├─────────────────────────────────────────────────────────────┤
│  Integrated Trading Bot (Core Orchestrator)                │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: Core Trading │ Phase 2: Risk Management          │
│  - Market Data         │ - Position Sizing                  │
│  - Order Management    │ - Stop Loss/Take Profit           │
│  - Strategy Execution  │ - Portfolio Limits                │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: Backtesting  │ Phase 4: Monitoring               │
│  - Historical Analysis │ - Performance Tracking            │
│  - Strategy Validation │ - Real-time Alerts               │
│  - Optimization        │ - System Health                   │
├─────────────────────────────────────────────────────────────┤
│  Phase 5: Tax Reporting│ Phase 6: Advanced Features        │
│  - Trade Recording     │ - ML Optimization                 │
│  - P&L Calculation     │ - Portfolio Management            │
│  - Tax Forms           │ - Advanced Analytics              │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                       │
│  - Configuration Mgmt  │ - Health Monitoring               │
│  - Database            │ - Performance Testing             │
│  - Logging & Metrics   │ - Deployment Automation           │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Market Data Ingestion**: Real-time market data from exchanges
2. **Strategy Processing**: Analysis and signal generation
3. **Risk Assessment**: Position sizing and risk validation
4. **Order Execution**: Trade placement and management
5. **Monitoring**: Real-time system and performance monitoring
6. **Reporting**: Trade recording and tax reporting

---

## Installation Guide

### Prerequisites

1. **Python 3.11+**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.11 python3.11-pip python3.11-venv
   
   # CentOS/RHEL
   sudo dnf install python3.11 python3.11-pip
   
   # macOS (with Homebrew)
   brew install python@3.11
   
   # Windows
   # Download from python.org
   ```

2. **PostgreSQL Database**
   ```bash
   # Ubuntu/Debian
   sudo apt install postgresql postgresql-contrib
   
   # CentOS/RHEL
   sudo dnf install postgresql postgresql-server
   
   # macOS
   brew install postgresql
   ```

3. **Redis (Optional - for caching)**
   ```bash
   # Ubuntu/Debian
   sudo apt install redis-server
   
   # CentOS/RHEL
   sudo dnf install redis
   
   # macOS
   brew install redis
   ```

### Installation Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/trading-bot.git
   cd trading-bot
   ```

2. **Create Virtual Environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Database Setup**
   ```bash
   # Create database
   sudo -u postgres createdb trading_bot
   
   # Run migrations
   python -m alembic upgrade head
   ```

5. **Configuration**
   ```bash
   # Copy configuration template
   cp config/config.example.yml config/config.yml
   
   # Edit configuration
   nano config/config.yml
   ```

6. **Environment Variables**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit environment variables
   nano .env
   ```

### Verification

```bash
# Test installation
python -m pytest tests/test_installation.py

# Check system health
python -m src.monitoring.health_check

# Verify configuration
python -m src.bot.config_manager --validate
```

---

## Configuration Reference

### Main Configuration File (`config/config.yml`)

```yaml
# Environment Configuration
environment: production  # development, testing, production

# Database Configuration
database:
  host: localhost
  port: 5432
  name: trading_bot
  user: trading_bot_user
  password: ${DB_PASSWORD}
  pool_size: 10
  max_overflow: 20

# Exchange Configuration
exchanges:
  bybit:
    api_key: ${BYBIT_API_KEY}
    api_secret: ${BYBIT_API_SECRET}
    testnet: false
    rate_limit: 100
  
# Trading Configuration
trading:
  default_strategy: momentum_strategy
  max_positions: 10
  base_currency: USDT
  trading_pairs:
    - BTCUSDT
    - ETHUSDT
    - ADAUSDT

# Risk Management
risk:
  max_portfolio_risk: 0.02  # 2% of portfolio
  max_position_size: 0.05   # 5% of portfolio per position
  stop_loss_percent: 0.02   # 2% stop loss
  take_profit_percent: 0.04 # 4% take profit

# Monitoring Configuration
monitoring:
  health_check_interval: 30
  metrics_retention_hours: 168  # 7 days
  alert_channels:
    email:
      enabled: true
      smtp_host: smtp.gmail.com
      smtp_port: 587
      username: ${EMAIL_USERNAME}
      password: ${EMAIL_PASSWORD}
    discord:
      enabled: true
      webhook_url: ${DISCORD_WEBHOOK}

# API Configuration
api:
  host: 0.0.0.0
  port: 8000
  cors_origins:
    - http://localhost:3000
    - https://your-frontend.com
  rate_limits:
    default: 100/hour
    authenticated: 1000/hour
```

### Environment Variables (`.env`)

```bash
# Database
DB_PASSWORD=your_secure_password

# Exchange API Keys - ENVIRONMENT SPECIFIC
# TESTNET CREDENTIALS (for development/staging environments)
BYBIT_TESTNET_API_KEY=your_testnet_api_key
BYBIT_TESTNET_API_SECRET=your_testnet_api_secret

# LIVE TRADING CREDENTIALS (for production - REAL MONEY AT RISK!)
BYBIT_LIVE_API_KEY=your_live_api_key
BYBIT_LIVE_API_SECRET=your_live_api_secret

# Monitoring
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
DISCORD_WEBHOOK=https://discord.com/api/webhooks/...

# Security
JWT_SECRET=your_jwt_secret_key
API_ENCRYPTION_KEY=your_encryption_key

# External Services
REDIS_URL=redis://localhost:6379/0
PROMETHEUS_URL=http://localhost:9090
```

## ⚠️ CRITICAL: API Credential Security

### Environment-Specific Credentials

**THE SYSTEM NOW REQUIRES SEPARATE API CREDENTIALS FOR EACH ENVIRONMENT:**

- **DEVELOPMENT/STAGING**: Uses `BYBIT_TESTNET_API_KEY` and `BYBIT_TESTNET_API_SECRET`
- **PRODUCTION**: Uses `BYBIT_LIVE_API_KEY` and `BYBIT_LIVE_API_SECRET`

### Safety Measures

1. **Never mix credentials**: Testnet keys should never be used in production and vice versa
2. **Environment validation**: The system validates credentials match the environment on startup
3. **Clear logging**: Environment and credential type are clearly logged on startup

### Configuration Validation

```bash
# Validate configuration
python -m src.bot.config_manager --validate

# Test configuration in different environments
python -m src.bot.config_manager --environment development --validate
python -m src.bot.config_manager --environment production --validate

# Check current environment credentials
python -c "from src.bot.config_manager import ConfigurationManager; cm = ConfigurationManager(); cm.load_config(); print(f'Environment: {cm.config.environment.value}'); print(f'Is Testnet: {cm.is_testnet}')"
```

---

## API Documentation

### Authentication

The API supports two authentication methods:

1. **API Keys**: For programmatic access
2. **JWT Tokens**: For web applications

#### API Key Authentication

```bash
curl -H "Authorization: Bearer tb_your_api_key_here" \
     http://localhost:8000/status
```

#### JWT Token Authentication

```bash
# Get token (if implemented)
curl -X POST http://localhost:8000/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "password"}'

# Use token
curl -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
     http://localhost:8000/status
```

### REST API Endpoints

#### System Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/health` | Basic health check | No |
| GET | `/status` | System status | Yes |
| GET | `/metrics` | System metrics | Yes |
| GET | `/alerts` | System alerts | Yes |

#### Configuration Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/config` | Get configuration | Yes |
| POST | `/config` | Update configuration | Admin |

#### Trading Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/trading/status` | Trading status | Yes |
| POST | `/system/command` | System commands | Control |

#### System Control Commands

```bash
# Start trading
curl -X POST http://localhost:8000/system/command \
     -H "Authorization: Bearer your_token" \
     -H "Content-Type: application/json" \
     -d '{"command": "start", "reason": "Manual start"}'

# Stop trading
curl -X POST http://localhost:8000/system/command \
     -H "Authorization: Bearer your_token" \
     -H "Content-Type: application/json" \
     -d '{"command": "stop", "reason": "Manual stop"}'
```

### WebSocket API

#### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/your_session_id');

ws.onopen = function(event) {
    console.log('Connected to WebSocket');
    
    // Subscribe to system status
    ws.send(JSON.stringify({
        type: 'subscribe',
        subscription: 'system_status'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

#### Subscription Types

- `system_status`: Real-time system status
- `metrics_update`: Live metrics
- `alert_notification`: Alert notifications
- `trading_update`: Trading updates
- `health_update`: Health status changes

### API Response Examples

#### System Status Response

```json
{
  "status": "healthy",
  "uptime_seconds": 3600.5,
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "trading_engine": "healthy",
    "risk_manager": "healthy",
    "database": "healthy"
  },
  "active_alerts": 0,
  "message": "All systems operational"
}
```

#### Metrics Response

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "metrics": {
    "system_cpu_percent": 45.2,
    "system_memory_percent": 62.1,
    "trading_portfolio_value": 10000.50,
    "trading_open_positions": 3
  },
  "statistics": {
    "system_cpu_percent": {
      "avg": 42.1,
      "min": 15.3,
      "max": 78.9,
      "p95": 65.4
    }
  }
}
```

---

## Deployment Guide

### Docker Deployment

#### Build Image

```bash
# Build production image
docker build -t trading-bot:latest .

# Build with specific tag
docker build -t trading-bot:v1.0.0 .
```

#### Run Container

```bash
# Run with environment file
docker run -d \
  --name trading-bot \
  --env-file .env \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  trading-bot:latest
```

#### Docker Compose

```yaml
version: '3.8'

services:
  trading-bot:
    build: .
    container_name: trading-bot
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15
    container_name: trading-bot-db
    environment:
      POSTGRES_DB: trading_bot
      POSTGRES_USER: trading_bot_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    container_name: trading-bot-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

#### Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: trading-bot
```

#### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trading-bot-config
  namespace: trading-bot
data:
  config.yml: |
    environment: production
    database:
      host: postgres-service
      port: 5432
      name: trading_bot
    # ... rest of configuration
```

#### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-bot
  namespace: trading-bot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trading-bot
  template:
    metadata:
      labels:
        app: trading-bot
    spec:
      containers:
      - name: trading-bot
        image: trading-bot:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        envFrom:
        - secretRef:
            name: trading-bot-secrets
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: logs
          mountPath: /app/logs
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: trading-bot-config
      - name: logs
        emptyDir: {}
```

### Production Deployment Checklist

- [ ] Database configured and migrated
- [ ] Environment variables set
- [ ] SSL certificates installed
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Security hardening applied
- [ ] Performance testing completed
- [ ] Documentation updated
- [ ] Team training completed

---

## Monitoring & Health Checks

### Health Check Endpoints

#### Basic Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Detailed System Status

```bash
curl -H "Authorization: Bearer your_token" \
     http://localhost:8000/status
```

### Monitoring Integration

#### Prometheus Metrics

The system exposes metrics at `/metrics` endpoint:

```bash
curl http://localhost:8000/metrics
```

#### Grafana Dashboard

Import the provided Grafana dashboard (`monitoring/grafana-dashboard.json`):

1. Open Grafana
2. Go to Import
3. Upload `grafana-dashboard.json`
4. Configure data source

#### Alert Configuration

Alerts are configured in `config/alerts.yml`:

```yaml
alerts:
  cpu_high:
    condition: cpu_percent > 80
    duration: 5m
    severity: warning
    
  memory_critical:
    condition: memory_percent > 95
    duration: 2m
    severity: critical
    
  trading_stopped:
    condition: trading_active == false
    duration: 1m
    severity: critical
```

### Log Management

#### Log Locations

- **Application Logs**: `logs/app.log`
- **Error Logs**: `logs/error.log`
- **Trading Logs**: `logs/trading.log`
- **API Logs**: `logs/api.log`

#### Log Rotation

Configure log rotation in `config/logging.yml`:

```yaml
logging:
  version: 1
  handlers:
    file:
      class: logging.handlers.RotatingFileHandler
      filename: logs/app.log
      maxBytes: 100000000  # 100MB
      backupCount: 10
```

---

## Performance Tuning

### System Optimization

#### Database Optimization

1. **Connection Pooling**
   ```yaml
   database:
     pool_size: 20
     max_overflow: 30
     pool_timeout: 30
   ```

2. **Index Optimization**
   ```sql
   -- Create indexes for frequently queried columns
   CREATE INDEX idx_trades_timestamp ON trades(timestamp);
   CREATE INDEX idx_trades_symbol ON trades(symbol);
   ```

#### Memory Optimization

1. **Python Memory Settings**
   ```bash
   export PYTHONHASHSEED=0
   export PYTHONOPTIMIZE=1
   ```

2. **Garbage Collection Tuning**
   ```python
   import gc
   gc.set_threshold(700, 10, 10)
   ```

#### CPU Optimization

1. **Process Priority**
   ```bash
   nice -n -10 python -m src.bot.integrated_trading_bot
   ```

2. **CPU Affinity**
   ```bash
   taskset -c 0,1 python -m src.bot.integrated_trading_bot
   ```

### Application Optimization

#### Async Performance

1. **Event Loop Tuning**
   ```python
   import asyncio
   import uvloop  # Install: pip install uvloop
   
   asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
   ```

2. **Connection Pooling**
   ```python
   # HTTP connection pooling
   connector = aiohttp.TCPConnector(
       limit=100,
       ttl_dns_cache=300,
       use_dns_cache=True
   )
   ```

#### Caching Strategy

1. **Redis Caching**
   ```python
   # Cache market data
   await redis.setex(f"market_data:{symbol}", 30, data)
   ```

2. **Memory Caching**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def calculate_indicator(data):
       # Expensive calculation
       pass
   ```

### Performance Monitoring

#### Performance Testing

```bash
# Run performance tests
python -m src.testing.performance_testing

# Load testing
python -m locust -f tests/load_tests.py --host http://localhost:8000
```

#### Profiling

```bash
# CPU profiling
python -m cProfile -o profile.stats -m src.bot.integrated_trading_bot

# Memory profiling
python -m memory_profiler src/bot/integrated_trading_bot.py
```

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues

**Symptoms**: Connection timeout, database errors

**Solutions**:
```bash
# Check database status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U trading_bot_user -d trading_bot

# Check connection limits
SELECT * FROM pg_stat_activity;
```

#### 2. API Authentication Failures

**Symptoms**: 401 Unauthorized errors

**Solutions**:
```bash
# Verify API key
python -c "from src.api.trading_bot_api import SecurityManager; sm = SecurityManager(); print(sm.validate_api_key('your_key'))"

# Check token expiration
python -c "import jwt; print(jwt.decode('your_token', verify=False))"
```

#### 3. High Memory Usage

**Symptoms**: System slowdown, OOM errors

**Solutions**:
```bash
# Monitor memory usage
top -p $(pgrep -f trading_bot)

# Check for memory leaks
python -m memory_profiler src/bot/integrated_trading_bot.py

# Restart with memory limits
docker run --memory=2g trading-bot:latest
```

#### 4. Exchange API Errors

**Symptoms**: Rate limit errors, API failures

**Solutions**:
```python
# Check rate limits
from src.trading.exchanges.bybit_client import BybitClient
client = BybitClient()
print(client.get_rate_limit_status())

# Implement exponential backoff
import time
import random

def retry_with_backoff(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except RateLimitError:
            time.sleep(2 ** i + random.uniform(0, 1))
```

### Debugging Commands

#### System Diagnostics

```bash
# Check system resources
python -m src.monitoring.system_diagnostics

# Validate configuration
python -m src.bot.config_manager --validate --verbose

# Test database connection
python -m src.database.connection_test

# Check exchange connectivity
python -m src.trading.exchanges.connectivity_test
```

#### Log Analysis

```bash
# Search for errors
grep -i error logs/app.log | tail -20

# Filter by timestamp
grep "2024-01-15 10:" logs/app.log

# Monitor real-time logs
tail -f logs/app.log | grep -i "error\|warning"
```

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| E001 | Database connection failed | Check database status and credentials |
| E002 | Exchange API authentication failed | Verify API keys and permissions |
| E003 | Configuration validation failed | Review configuration syntax |
| E004 | Insufficient funds | Check account balance |
| E005 | Rate limit exceeded | Implement rate limiting |

---

## Operational Procedures

### Daily Operations

#### Morning Checklist

- [ ] Check system status
- [ ] Review overnight alerts
- [ ] Verify trading activity
- [ ] Check portfolio performance
- [ ] Review resource usage

```bash
# Morning health check script
#!/bin/bash
echo "=== Daily Health Check ==="
curl -s http://localhost:8000/health | jq .
curl -s -H "Authorization: Bearer $API_KEY" http://localhost:8000/status | jq .
curl -s -H "Authorization: Bearer $API_KEY" http://localhost:8000/alerts | jq .
```

#### Evening Checklist

- [ ] Review daily performance
- [ ] Check log files for errors
- [ ] Verify backup completion
- [ ] Update configuration if needed
- [ ] Plan next day activities

### Weekly Operations

#### Weekly Maintenance

```bash
#!/bin/bash
# Weekly maintenance script

# Update system packages
sudo apt update && sudo apt upgrade -y

# Clean old logs
find logs/ -name "*.log.*" -mtime +7 -delete

# Database maintenance
python -m src.database.maintenance --vacuum --analyze

# Performance report
python -m src.monitoring.weekly_report

# Backup verification
python -m src.backup.verify_backups
```

### Monthly Operations

#### Monthly Review

- [ ] Performance analysis
- [ ] Security audit
- [ ] Configuration review
- [ ] Documentation updates
- [ ] Team training

#### Capacity Planning

```bash
# Generate capacity report
python -m src.monitoring.capacity_report --period 30d

# Resource utilization analysis
python -m src.monitoring.resource_analysis --detailed
```

### Emergency Procedures

#### System Shutdown

```bash
# Graceful shutdown
curl -X POST -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/system/command \
     -d '{"command": "shutdown", "reason": "Emergency shutdown"}'

# Force shutdown
pkill -f trading_bot
```

#### Emergency Recovery

```bash
#!/bin/bash
# Emergency recovery procedure

# Stop all trading activities
curl -X POST -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/system/command \
     -d '{"command": "stop", "reason": "Emergency stop"}'

# Close all positions (if implemented)
python -m src.trading.emergency_close_positions

# Backup current state
python -m src.backup.emergency_backup

# Restore from backup if needed
python -m src.backup.restore --latest
```

---

## Security Guidelines

### API Security

#### Authentication Best Practices

1. **API Key Management**
   ```bash
   # Generate secure API keys
   python -c "import secrets; print('tb_' + secrets.token_urlsafe(32))"
   
   # Rotate keys regularly
   python -m src.api.rotate_api_keys --key-id your_key_id
   ```

2. **JWT Token Security**
   ```python
   # Use strong secrets
   JWT_SECRET = secrets.token_urlsafe(64)
   
   # Set appropriate expiration
   TOKEN_EXPIRY = timedelta(hours=1)
   ```

#### Rate Limiting

```yaml
rate_limits:
  default: "100/hour"
  authenticated: "1000/hour"
  admin: "5000/hour"
  burst_limit: 10
```

### Network Security

#### SSL/TLS Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Firewall Configuration

```bash
# UFW configuration
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 8000/tcp    # API (if needed)
sudo ufw enable
```

### Data Security

#### Encryption at Rest

```python
# Database encryption
DATABASE_URL = "postgresql://user:pass@localhost/db?sslmode=require"

# File encryption
from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher_suite = Fernet(key)
```

#### Encryption in Transit

```python
# HTTPS only
app.add_middleware(HTTPSRedirectMiddleware)

# Secure headers
app.add_middleware(SecurityHeadersMiddleware)
```

### Access Control

#### Role-Based Access

```yaml
roles:
  read_only:
    permissions:
      - view_status
      - view_metrics
      - view_alerts
  
  operator:
    permissions:
      - all_read_only
      - control_trading
      - update_config
  
  admin:
    permissions:
      - all_operator
      - manage_users
      - system_shutdown
```

---

## Backup & Recovery

### Backup Strategy

#### Database Backups

```bash
#!/bin/bash
# Database backup script

BACKUP_DIR="/backups/database"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="trading_bot_backup_${DATE}.sql"

# Create backup
pg_dump -h localhost -U trading_bot_user trading_bot > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Clean old backups (keep 30 days)
find "${BACKUP_DIR}" -name "*.sql.gz" -mtime +30 -delete

# Upload to cloud storage (optional)
aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" s3://your-backup-bucket/database/
```

#### Configuration Backups

```bash
#!/bin/bash
# Configuration backup script

BACKUP_DIR="/backups/config"
DATE=$(date +%Y%m%d_%H%M%S)

# Create archive
tar -czf "${BACKUP_DIR}/config_backup_${DATE}.tar.gz" \
    config/ \
    .env \
    docker-compose.yml

# Upload to cloud storage
aws s3 cp "${BACKUP_DIR}/config_backup_${DATE}.tar.gz" s3://your-backup-bucket/config/
```

### Recovery Procedures

#### Database Recovery

```bash
#!/bin/bash
# Database recovery script

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application
systemctl stop trading-bot

# Drop and recreate database
sudo -u postgres dropdb trading_bot
sudo -u postgres createdb trading_bot

# Restore from backup
gunzip -c "$BACKUP_FILE" | psql -h localhost -U trading_bot_user trading_bot

# Start application
systemctl start trading-bot
```

#### System Recovery

```bash
#!/bin/bash
# Complete system recovery

# 1. Restore configuration
tar -xzf /backups/config/config_backup_latest.tar.gz -C /

# 2. Restore database
./restore_database.sh /backups/database/trading_bot_backup_latest.sql.gz

# 3. Restart services
docker-compose down
docker-compose up -d

# 4. Verify system
sleep 30
curl http://localhost:8000/health
```

### Disaster Recovery

#### Recovery Time Objectives (RTO)

- **Database Recovery**: 15 minutes
- **Configuration Recovery**: 5 minutes
- **Full System Recovery**: 30 minutes

#### Recovery Point Objectives (RPO)

- **Database**: 1 hour (hourly backups)
- **Configuration**: 24 hours (daily backups)
- **Logs**: 1 hour

#### DR Testing

```bash
#!/bin/bash
# Disaster recovery test

echo "Starting DR test..."

# 1. Backup current state
./backup_all.sh

# 2. Simulate disaster
docker-compose down
docker volume rm trading-bot_postgres_data

# 3. Perform recovery
./full_recovery.sh

# 4. Verify recovery
python -m src.tests.dr_verification

echo "DR test completed"
```

---

This comprehensive documentation provides complete guidance for deploying, operating, and maintaining the Trading Bot System in production environments. Regular updates to this documentation are essential as the system evolves.