# SECURE PRODUCTION DEPLOYMENT GUIDE - DIGITALOCEAN
## Live Trading Security & Infrastructure Setup

### 🚀 **PRODUCTION READINESS STATUS: READY FOR DEPLOYMENT**

This guide provides comprehensive security hardening and deployment strategies for your Bybit trading bot on DigitalOcean with live trading capabilities.

---

## 🛡️ **SECURITY ARCHITECTURE OVERVIEW**

### Multi-Layer Security Strategy
1. **Infrastructure Security** - Firewall, VPC, SSH hardening
2. **Application Security** - API encryption, secret management
3. **Network Security** - SSL/TLS, private networking
4. **Monitoring Security** - Real-time threat detection
5. **Financial Security** - Trading risk controls, emergency stops

---

## 🏗️ **DIGITALOCEAN INFRASTRUCTURE SETUP**

### Recommended Droplet Configuration

```yaml
# Production Droplet Specifications
primary_droplet:
  size: "s-2vcpu-4gb"  # Minimum for live trading
  region: "syd1"       # Sydney for Australian compliance
  image: "ubuntu-22-04-x64"
  
backup_droplet:
  size: "s-1vcpu-2gb"  # Failover instance
  region: "sgp1"       # Singapore backup
  image: "ubuntu-22-04-x64"

database_droplet:
  size: "db-s-1vcpu-1gb"
  engine: "pg"         # PostgreSQL for production
  region: "syd1"
```

### 1. **VPC Network Setup**
```bash
# Create isolated VPC for trading infrastructure
doctl vpcs create \
  --name "trading-vpc-prod" \
  --region "syd1" \
  --ip-range "10.0.0.0/16"

# Create private subnet for databases
doctl vpcs subnets create \
  --vpc-uuid "your-vpc-uuid" \
  --name "database-subnet" \
  --ip-range "10.0.1.0/24"
```

---

## 🔐 **SECURITY HARDENING CHECKLIST**

### SSH & Access Control
```bash
# 1. Create dedicated trading user
sudo useradd -m -s /bin/bash tradingbot
sudo usermod -aG sudo tradingbot

# 2. SSH key-only authentication
sudo nano /etc/ssh/sshd_config
```

```ini
# SSH Security Configuration
Port 2222                    # Change from default 22
PasswordAuthentication no
PubkeyAuthentication yes
PermitRootLogin no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
AllowUsers tradingbot
```

### Firewall Configuration
```bash
# UFW Firewall Setup
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow only essential ports
sudo ufw allow 2222/tcp      # SSH (custom port)
sudo ufw allow 443/tcp       # HTTPS
sudo ufw allow 80/tcp        # HTTP (redirect to HTTPS)

# Internal network access only
sudo ufw allow from 10.0.0.0/16 to any port 5432    # PostgreSQL
sudo ufw allow from 10.0.0.0/16 to any port 6379    # Redis

sudo ufw enable
```

### Fail2Ban Protection
```bash
# Install and configure Fail2Ban
sudo apt install fail2ban -y

sudo nano /etc/fail2ban/jail.local
```

```ini
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = 2222
logpath = /var/log/auth.log

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
action = iptables-multiport[name=ReqLimit, port="http,https", protocol=tcp]
logpath = /var/log/nginx/*error.log
```

---

## 🔑 **SECRET MANAGEMENT & API SECURITY**

### Environment Variables Setup
```bash
# Create secure environment file
sudo nano /opt/trading/.env.production
```

```env
# Bybit API Credentials (LIVE TRADING)
BYBIT_API_KEY="your_live_api_key"
BYBIT_API_SECRET="your_live_api_secret"
BYBIT_TESTNET=false

# Database Configuration
DATABASE_URL="postgresql://trading_user:secure_password@10.0.1.5:5432/trading_prod"
REDIS_URL="redis://10.0.1.6:6379/0"

# Security Configuration
SECRET_KEY="generate_256_bit_secret"
JWT_SECRET="generate_jwt_secret"
API_RATE_LIMIT="100/hour"

# Australian Compliance
TIMEZONE="Australia/Sydney"
TAX_COMPLIANCE_MODE="production"
ATO_REPORTING_ENABLED=true

# Monitoring & Alerts
SMTP_SERVER="smtp.gmail.com"
SMTP_PORT=587
ALERT_EMAIL="your_secure_email@domain.com"
SLACK_WEBHOOK_URL="your_slack_webhook"

# Emergency Controls
MAX_DAILY_LOSS=1000.00
MAX_POSITION_SIZE=0.1
EMERGENCY_STOP_THRESHOLD=0.05
```

### File Permissions Security
```bash
# Secure environment file
sudo chown tradingbot:tradingbot /opt/trading/.env.production
sudo chmod 600 /opt/trading/.env.production

# Secure application directory
sudo chown -R tradingbot:tradingbot /opt/trading/
sudo chmod -R 750 /opt/trading/
```

---

## 🐳 **DOCKER PRODUCTION SETUP**

### Secure Dockerfile for Production
```dockerfile
# Production Dockerfile
FROM python:3.11-slim

# Security: Create non-root user
RUN groupadd -r trading && useradd -r -g trading trading

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Security: Change ownership to non-root user
RUN chown -R trading:trading /app
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/monitoring/health || exit 1

# Expose application port
EXPOSE 8000

# Start application
CMD ["python", "-m", "src.main"]
```

### Docker Compose for Production
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  trading-app:
    build: .
    restart: unless-stopped
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      - ENV=production
    env_file:
      - .env.production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    networks:
      - trading-network

  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: trading_prod
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - trading-network
    ports:
      - "127.0.0.1:5432:5432"

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - trading-network

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - trading-app
    networks:
      - trading-network

volumes:
  postgres_data:
  redis_data:

networks:
  trading-network:
    driver: bridge
```

---

## 🌐 **NGINX SSL CONFIGURATION**

### Nginx Configuration with SSL
```nginx
# /etc/nginx/sites-available/trading-bot
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # API rate limiting for trading endpoints
    location /api/trading/ {
        limit_req zone=api burst=5 nodelay;
        proxy_pass http://127.0.0.1:8000;
    }
}
```

---

## 📊 **PRODUCTION MONITORING & ALERTS**

### System Monitoring Setup
```bash
# Install monitoring tools
sudo apt install htop iotop nethogs fail2ban -y

# Install and configure Prometheus
curl -LO https://github.com/prometheus/prometheus/releases/download/v2.40.0/prometheus-2.40.0.linux-amd64.tar.gz
tar xvf prometheus-2.40.0.linux-amd64.tar.gz
sudo cp prometheus-2.40.0.linux-amd64/prometheus /usr/local/bin/
```

### Trading-Specific Monitoring
```python
# Enhanced monitoring configuration
MONITORING_CONFIG = {
    "alerts": {
        "max_drawdown": 0.05,      # 5% max drawdown
        "daily_loss_limit": 1000,   # $1000 daily loss limit
        "api_error_threshold": 5,   # 5 consecutive API errors
        "connection_timeout": 30,   # 30 second timeout
    },
    "notifications": {
        "email": "your-alerts@domain.com",
        "slack_webhook": "your-slack-webhook",
        "sms": "+61-your-number",   # Australian number
    },
    "backup_triggers": {
        "switch_to_backup": True,
        "emergency_stop": True,
        "position_close": True,
    }
}
```

---

## 💰 **LIVE TRADING SAFETY CONTROLS**

### Risk Management Configuration
```python
# Production risk management settings
LIVE_TRADING_LIMITS = {
    "max_position_size": 0.1,        # 10% of account per position
    "max_daily_trades": 50,          # Limit overtrading
    "max_daily_loss": 1000.00,       # Australian dollars
    "stop_loss_percent": 0.02,       # 2% stop loss
    "take_profit_percent": 0.04,     # 4% take profit
    "cooldown_period": 300,          # 5 minutes between trades
    "emergency_stop_threshold": 0.05, # 5% account drawdown
}
```

### Emergency Stop Implementation
```python
# Add to src/main.py
@app.post("/api/emergency-stop")
async def emergency_stop():
    """Emergency stop all trading activities"""
    try:
        # Close all open positions
        await close_all_positions()
        
        # Cancel all open orders
        await cancel_all_orders()
        
        # Stop all trading algorithms
        stop_all_strategies()
        
        # Send immediate alerts
        send_emergency_alert("EMERGENCY STOP ACTIVATED")
        
        return {"status": "success", "message": "All trading stopped"}
    except Exception as e:
        logger.error(f"Emergency stop failed: {e}")
        return {"status": "error", "message": str(e)}
```

---

## 🚀 **DEPLOYMENT AUTOMATION**

### Deployment Script
```bash
#!/bin/bash
# deploy_production.sh

set -e

echo "🚀 Starting production deployment..."

# Backup current version
docker-compose -f docker-compose.prod.yml down
cp -r /opt/trading /opt/trading.backup.$(date +%Y%m%d_%H%M%S)

# Pull latest code
cd /opt/trading
git pull origin main

# Build and deploy
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d

# Health check
sleep 30
curl -f http://localhost:8000/api/monitoring/health || exit 1

echo "✅ Production deployment completed successfully"
```

### Monitoring Dashboard Deployment
```bash
# Deploy Grafana for monitoring
docker run -d \
  --name=grafana \
  -p 3000:3000 \
  -v grafana-data:/var/lib/grafana \
  grafana/grafana:latest
```

---

## 📋 **PRE-DEPLOYMENT CHECKLIST**

### Security Verification
- [ ] SSH keys configured (no password auth)
- [ ] Firewall properly configured
- [ ] SSL certificates installed and valid
- [ ] Environment variables secured (600 permissions)
- [ ] Database connections encrypted
- [ ] Rate limiting implemented
- [ ] Fail2Ban active and configured

### Trading Safety
- [ ] Risk management limits configured
- [ ] Emergency stop functionality tested
- [ ] Backup systems operational
- [ ] Monitoring alerts configured
- [ ] Australian tax compliance verified
- [ ] API rate limits properly set

### Infrastructure
- [ ] VPC network isolated
- [ ] Database backups automated
- [ ] Log rotation configured
- [ ] Health checks operational
- [ ] Load balancing configured (if needed)
- [ ] Disaster recovery plan tested

---

## 🆘 **EMERGENCY PROCEDURES**

### Immediate Response Actions
1. **Trading Emergency**: POST to `/api/emergency-stop`
2. **System Compromise**: `sudo ufw deny all incoming`
3. **Data Breach**: Rotate all API keys immediately
4. **Server Failure**: Switch to backup droplet
5. **Network Issues**: Check VPC and firewall rules

### Contact Information
- **DigitalOcean Support**: Available 24/7
- **Bybit API Support**: Live chat available
- **Your Alert Channels**: Email, Slack, SMS configured

---

## 📞 **SUPPORT & MAINTENANCE**

### Regular Maintenance Schedule
- **Daily**: Check logs and performance metrics
- **Weekly**: Security updates and system patches  
- **Monthly**: Full security audit and backup verification
- **Quarterly**: Disaster recovery testing

### Log Monitoring Commands
```bash
# Real-time log monitoring
tail -f /opt/trading/logs/app.log
tail -f /var/log/nginx/access.log
tail -f /var/log/auth.log

# System performance
htop
iotop
df -h
free -m
```

---

**🎯 PRODUCTION DEPLOYMENT STATUS: READY**

This guide provides enterprise-grade security for live trading deployment on DigitalOcean. Follow each section carefully for maximum security and reliability.

**⚠️ CRITICAL**: Always test emergency stops and risk controls before going live with real funds!