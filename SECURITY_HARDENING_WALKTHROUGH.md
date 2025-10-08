# 🔒 SECURITY HARDENING WALKTHROUGH
## Complete Security Guide for DigitalOcean Production Deployment

### 📋 SECURITY CHECKLIST OVERVIEW

This guide covers **7 Critical Security Layers** for your Bybit trading bot:

1. **🔐 Environment Variable Security** - API keys, secrets, credentials
2. **🛡️ DigitalOcean Droplet Hardening** - Server security, firewall, SSH
3. **🔑 Application Security** - Authentication, authorization, input validation
4. **🌐 Network Security** - HTTPS, reverse proxy, rate limiting
5. **📊 Database Security** - SQLite protection, backup encryption
6. **🚨 Monitoring & Alerts** - Security monitoring, intrusion detection
7. **📝 Audit & Compliance** - Logging, compliance, incident response

---

## 1. 🔐 ENVIRONMENT VARIABLE SECURITY

### **Current Status**: ✅ GOOD - Dual Environment Setup Complete
### **Risk Level**: 🔴 CRITICAL - API Keys = Real Money Access

#### **A. DigitalOcean Environment Variables Setup**

**Step 1: Create Environment Variables in DigitalOcean**

```bash
# In your DigitalOcean App Platform or Droplet
# Navigate to: Settings > Environment Variables

# REQUIRED: Dual API Environment
BYBIT_TESTNET_API_KEY=your_testnet_key_here
BYBIT_TESTNET_API_SECRET=your_testnet_secret_here
BYBIT_LIVE_API_KEY=your_live_key_here
BYBIT_LIVE_API_SECRET=your_live_secret_here

# SECURITY: Authentication
DASHBOARD_PASSWORD=create_strong_password_here_min_16_chars
SECRET_KEY=generate_random_32_char_string

# DEPLOYMENT: Environment Control
ENVIRONMENT=production
TRADING_MODE=paper  # Start with paper, change to 'live' when ready

# DATABASE: Production Config
DATABASE_PATH=/app/data/trading_bot.db
BACKUP_ENCRYPTION_KEY=generate_32_char_backup_encryption_key

# MONITORING: Security Alerts
ALERT_EMAIL=your_security_alerts@email.com
WEBHOOK_SECRET=generate_webhook_secret_32_chars
```

**Step 2: Validate Environment Security**

```bash
# Test environment variable access (run in DigitalOcean terminal)
echo "Testing environment security..."
echo "Testnet Key Length: ${#BYBIT_TESTNET_API_KEY}"  # Should be >20
echo "Live Key Length: ${#BYBIT_LIVE_API_KEY}"        # Should be >20  
echo "Dashboard Password Length: ${#DASHBOARD_PASSWORD}" # Should be >15
echo "Environment: $ENVIRONMENT"                       # Should be 'production'
```

#### **B. API Key Security Best Practices**

**🔴 CRITICAL: Bybit API Configuration**

1. **Enable IP Restrictions**:
   - Log into Bybit API Management
   - Add your DigitalOcean droplet IP to whitelist
   - Enable "Restrict Access to Trusted IPs Only"

2. **Minimize API Permissions**:
   ```
   ✅ Enable: Spot Trading
   ✅ Enable: Contract Trading (if needed)
   ✅ Enable: Wallet (read-only)
   ❌ Disable: Withdrawal
   ❌ Disable: Transfer
   ❌ Disable: Sub-Account Management
   ```

3. **Set Trading Limits**:
   - Maximum daily trading volume limit
   - Maximum position size limits  
   - Enable automatic logout after inactivity

#### **C. Environment Variable Encryption**

**Add to your deployment script**:

```bash
#!/bin/bash
# encrypt_env.sh - Environment Variable Encryption

# Create encrypted environment file
cat > .env.encrypted << 'EOF'
# Use strong encryption for sensitive variables
export BYBIT_LIVE_API_KEY=$(echo "$BYBIT_LIVE_API_KEY" | openssl enc -aes-256-cbc -a -salt -pass pass:"$MASTER_PASSWORD")
export BYBIT_LIVE_API_SECRET=$(echo "$BYBIT_LIVE_API_SECRET" | openssl enc -aes-256-cbc -a -salt -pass pass:"$MASTER_PASSWORD")
EOF

# Secure the file
chmod 600 .env.encrypted
chown root:root .env.encrypted
```

---

## 2. 🛡️ DIGITALOCEAN DROPLET HARDENING

### **Current Status**: ⚠️ NEEDS HARDENING - Standard Ubuntu Setup
### **Risk Level**: 🟡 HIGH - Server Compromise = Total System Loss

#### **A. Initial Server Hardening**

**Step 1: Create Non-Root User**

```bash
# SSH into your DigitalOcean droplet as root
ssh root@your_droplet_ip

# Create trading bot user
adduser tradingbot
usermod -aG sudo tradingbot

# Set up SSH key authentication
mkdir /home/tradingbot/.ssh
cp ~/.ssh/authorized_keys /home/tradingbot/.ssh/
chown -R tradingbot:tradingbot /home/tradingbot/.ssh
chmod 700 /home/tradingbot/.ssh
chmod 600 /home/tradingbot/.ssh/authorized_keys
```

**Step 2: SSH Security Configuration**

```bash
# Edit SSH configuration
sudo nano /etc/ssh/sshd_config

# Apply these security settings:
Port 2222                          # Change from default port 22
PermitRootLogin no                 # Disable root login
PasswordAuthentication no          # Disable password auth
PubkeyAuthentication yes          # Enable key-based auth only
MaxAuthTries 3                    # Limit login attempts
ClientAliveInterval 300           # Auto logout inactive sessions
ClientAliveCountMax 2             # Connection timeout
AllowUsers tradingbot             # Only allow specific user

# Restart SSH service
sudo systemctl restart sshd
```

**Step 3: Configure Firewall (UFW)**

```bash
# Enable and configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow necessary ports
sudo ufw allow 2222/tcp              # SSH (custom port)
sudo ufw allow 80/tcp                # HTTP
sudo ufw allow 443/tcp               # HTTPS
sudo ufw deny 22/tcp                 # Block default SSH port

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status verbose
```

#### **B. System Security Updates**

**Automated Security Updates**:

```bash
# Install automatic security updates
sudo apt update && sudo apt upgrade -y
sudo apt install unattended-upgrades apt-listchanges -y

# Configure automatic security updates
sudo dpkg-reconfigure -plow unattended-upgrades

# Enable automatic security updates
echo 'Unattended-Upgrade::Automatic-Reboot "false";' | sudo tee -a /etc/apt/apt.conf.d/50unattended-upgrades
echo 'Unattended-Upgrade::Mail "your_email@domain.com";' | sudo tee -a /etc/apt/apt.conf.d/50unattended-upgrades
```

#### **C. Intrusion Detection System**

**Install Fail2Ban**:

```bash
# Install fail2ban for intrusion detection
sudo apt install fail2ban -y

# Configure fail2ban for SSH protection
sudo tee /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 1h
findtime = 10m
maxretry = 5
ignoreip = 127.0.0.1/8 YOUR_HOME_IP_HERE

[sshd]
enabled = true
port = 2222
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 24h

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-limit-req]
enabled = true
filter = nginx-limit-req  
logpath = /var/log/nginx/error.log
maxretry = 10
EOF

# Start fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Check fail2ban status
sudo fail2ban-client status
```

---

## 3. 🔑 APPLICATION SECURITY

### **Current Status**: ⚠️ NEEDS HARDENING - Basic Auth Only
### **Risk Level**: 🟡 HIGH - Dashboard Access = Trading Control

#### **A. Authentication Hardening**

**Step 1: Implement Strong Authentication**

```python
# Add to src/main.py - Enhanced authentication
import hashlib
import hmac
import time
from datetime import datetime, timedelta

class SecurityManager:
    def __init__(self):
        self.failed_attempts = {}
        self.locked_ips = {}
        self.session_tokens = {}
        
    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if IP is rate limited"""
        now = time.time()
        
        # Check if IP is currently locked
        if client_ip in self.locked_ips:
            if now < self.locked_ips[client_ip]:
                return False  # Still locked
            else:
                del self.locked_ips[client_ip]  # Lock expired
                
        # Check failed attempts
        if client_ip in self.failed_attempts:
            attempts = self.failed_attempts[client_ip]
            # Remove attempts older than 15 minutes
            recent_attempts = [t for t in attempts if now - t < 900]
            self.failed_attempts[client_ip] = recent_attempts
            
            # Lock if too many recent attempts
            if len(recent_attempts) >= 5:
                self.locked_ips[client_ip] = now + 3600  # Lock for 1 hour
                return False
                
        return True
    
    def record_failed_attempt(self, client_ip: str):
        """Record a failed authentication attempt"""
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = []
        self.failed_attempts[client_ip].append(time.time())
```

**Step 2: Session Management**

```python
# Add secure session management
class SessionManager:
    def __init__(self):
        self.active_sessions = {}
        self.session_timeout = 3600  # 1 hour
        
    def create_session(self, user_id: str, client_ip: str) -> str:
        """Create a secure session token"""
        token_data = f"{user_id}:{client_ip}:{time.time()}"
        session_token = hashlib.sha256(
            f"{token_data}:{os.getenv('SECRET_KEY')}".encode()
        ).hexdigest()
        
        self.active_sessions[session_token] = {
            'user_id': user_id,
            'client_ip': client_ip,
            'created_at': time.time(),
            'last_activity': time.time()
        }
        
        return session_token
    
    def validate_session(self, token: str, client_ip: str) -> bool:
        """Validate session token"""
        if token not in self.active_sessions:
            return False
            
        session = self.active_sessions[token]
        now = time.time()
        
        # Check if session expired
        if now - session['last_activity'] > self.session_timeout:
            del self.active_sessions[token]
            return False
            
        # Check IP consistency
        if session['client_ip'] != client_ip:
            del self.active_sessions[token]  # Potential session hijacking
            return False
            
        # Update last activity
        session['last_activity'] = now
        return True
```

#### **B. Input Validation & Sanitization**

**Add input validation to API endpoints**:

```python
from pydantic import BaseModel, validator
import re

class TradingRequest(BaseModel):
    symbol: str
    quantity: float
    price: float
    
    @validator('symbol')
    def validate_symbol(cls, v):
        # Only allow valid crypto pairs
        if not re.match(r'^[A-Z]{3,10}USDT$', v):
            raise ValueError('Invalid trading symbol format')
        return v
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0 or v > 1000000:
            raise ValueError('Invalid quantity range')
        return v
    
    @validator('price')
    def validate_price(cls, v):
        if v <= 0 or v > 1000000:
            raise ValueError('Invalid price range')
        return v

# Use in API endpoints
@app.post("/api/trade")
async def place_trade(request: TradingRequest, client_ip: str = Depends(get_client_ip)):
    # Input is automatically validated by Pydantic
    # Additional security checks here
    pass
```

#### **C. API Security Headers**

**Add security headers to FastAPI**:

```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains only
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["yourdomain.com", "www.yourdomain.com"]
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
    
    return response
```

---

## 4. 🌐 NETWORK SECURITY

### **Current Status**: ⚠️ NEEDS SETUP - Direct Application Access
### **Risk Level**: 🟡 MEDIUM - Unencrypted Traffic Exposure

#### **A. HTTPS Setup with Let's Encrypt**

**Step 1: Install Nginx Reverse Proxy**

```bash
# Install Nginx
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx -y

# Configure Nginx for trading bot
sudo tee /etc/nginx/sites-available/trading-bot << 'EOF'
server {
    listen 80;
    server_name your_domain.com www.your_domain.com;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=1r/s;
    
    # Proxy to FastAPI application
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Extra security for authentication endpoints
    location /auth {
        limit_req zone=auth burst=5 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Block access to sensitive files
    location ~ /\. {
        deny all;
    }
    
    location ~ \.(env|config|db)$ {
        deny all;
    }
}
EOF

# Enable the site
sudo ln -s /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

**Step 2: Setup SSL Certificate**

```bash
# Get free SSL certificate from Let's Encrypt
sudo certbot --nginx -d your_domain.com -d www.your_domain.com

# Test automatic renewal
sudo certbot renew --dry-run

# Add automatic renewal to crontab
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo tee -a /etc/crontab
```

#### **B. Advanced Rate Limiting**

**Nginx Rate Limiting Configuration**:

```nginx
# Add to /etc/nginx/nginx.conf in http block
http {
    # Rate limiting zones
    limit_req_zone $binary_remote_addr zone=general:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=api:10m rate=5r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=1r/m;
    limit_req_zone $binary_remote_addr zone=trading:10m rate=1r/s;
    
    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=conn_limit_per_ip:10m;
    
    # Include other configs...
}

# Add to your trading-bot site config
server {
    # General connection limiting
    limit_conn conn_limit_per_ip 10;
    
    location /api/portfolio {
        limit_req zone=api burst=10 nodelay;
        # proxy settings...
    }
    
    location /api/trade {
        limit_req zone=trading burst=3 nodelay;
        # Extra security for trading endpoints
        # proxy settings...
    }
    
    location /auth {
        limit_req zone=auth burst=1 nodelay;
        # proxy settings...
    }
}
```

#### **C. Network Monitoring**

**Install Network Monitoring**:

```bash
# Install network monitoring tools
sudo apt install netstat-nat iftop nethogs -y

# Monitor network connections
sudo netstat -tulpn | grep :8000  # Check FastAPI is running locally only
sudo iftop                        # Monitor network traffic
sudo nethogs                      # Monitor per-process network usage

# Create network monitoring script
cat > /home/tradingbot/monitor_network.sh << 'EOF'
#!/bin/bash

# Network Security Monitoring Script
LOG_FILE="/var/log/network-security.log"

echo "$(date): Starting network security check" >> $LOG_FILE

# Check for suspicious connections
SUSPICIOUS_CONNECTIONS=$(netstat -an | grep :8000 | grep -v 127.0.0.1 | wc -l)
if [ $SUSPICIOUS_CONNECTIONS -gt 0 ]; then
    echo "$(date): WARNING: Direct connections to port 8000 detected" >> $LOG_FILE
    netstat -an | grep :8000 | grep -v 127.0.0.1 >> $LOG_FILE
fi

# Check for too many connections from single IP
netstat -ntu | awk '{print $5}' | cut -d: -f1 | sort | uniq -c | sort -nr | while read COUNT IP; do
    if [ "$COUNT" -gt 50 ] && [ "$IP" != "127.0.0.1" ] && [ "$IP" != "" ]; then
        echo "$(date): WARNING: High connection count from IP $IP: $COUNT connections" >> $LOG_FILE
    fi
done

echo "$(date): Network security check completed" >> $LOG_FILE
EOF

chmod +x /home/tradingbot/monitor_network.sh

# Add to crontab (run every 5 minutes)
echo "*/5 * * * * /home/tradingbot/monitor_network.sh" | crontab -
```

---

## 5. 📊 DATABASE SECURITY

### **Current Status**: ✅ GOOD - SQLite with Proper Permissions
### **Risk Level**: 🟡 MEDIUM - Contains Trading History & Strategies

#### **A. Database File Security**

**Secure SQLite Database**:

```bash
# Create secure database directory
sudo mkdir -p /app/data
sudo chown tradingbot:tradingbot /app/data
sudo chmod 750 /app/data

# Set strict permissions on database file
sudo chmod 640 /app/data/trading_bot.db
sudo chown tradingbot:www-data /app/data/trading_bot.db

# Create database backup directory
sudo mkdir -p /app/backups/database
sudo chown tradingbot:tradingbot /app/backups/database
sudo chmod 700 /app/backups/database
```

#### **B. Encrypted Database Backups**

**Automated Encrypted Backups**:

```bash
# Create backup script
cat > /home/tradingbot/backup_database.sh << 'EOF'
#!/bin/bash

# Database Backup Script with Encryption
BACKUP_DIR="/app/backups/database"
DB_FILE="/app/data/trading_bot.db"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="trading_bot_backup_$DATE.db"
ENCRYPTED_FILE="$BACKUP_FILE.gpg"

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Create database backup
cp $DB_FILE "$BACKUP_DIR/$BACKUP_FILE"

# Encrypt the backup
gpg --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 \
    --s2k-digest-algo SHA512 --s2k-count 65536 --symmetric \
    --output "$BACKUP_DIR/$ENCRYPTED_FILE" \
    "$BACKUP_DIR/$BACKUP_FILE"

# Remove unencrypted backup
rm "$BACKUP_DIR/$BACKUP_FILE"

# Keep only last 30 days of backups
find $BACKUP_DIR -name "*.gpg" -mtime +30 -delete

# Log backup completion
echo "$(date): Database backup completed: $ENCRYPTED_FILE" >> /var/log/backup.log

# Optional: Upload to secure cloud storage
# aws s3 cp "$BACKUP_DIR/$ENCRYPTED_FILE" s3://your-secure-bucket/database-backups/
EOF

chmod +x /home/tradingbot/backup_database.sh

# Add to crontab (daily at 2 AM)
echo "0 2 * * * /home/tradingbot/backup_database.sh" | crontab -
```

#### **C. Database Integrity Monitoring**

**Database Integrity Check**:

```bash
# Create integrity monitoring script
cat > /home/tradingbot/check_db_integrity.sh << 'EOF'
#!/bin/bash

DB_FILE="/app/data/trading_bot.db"
LOG_FILE="/var/log/database-integrity.log"

echo "$(date): Starting database integrity check" >> $LOG_FILE

# Check SQLite database integrity
INTEGRITY_CHECK=$(sqlite3 $DB_FILE "PRAGMA integrity_check;")

if [ "$INTEGRITY_CHECK" != "ok" ]; then
    echo "$(date): ERROR: Database integrity check failed!" >> $LOG_FILE
    echo "$INTEGRITY_CHECK" >> $LOG_FILE
    
    # Send alert email
    echo "Database integrity check failed on $(hostname)" | mail -s "URGENT: Database Integrity Alert" your_email@domain.com
    
    # Create emergency backup
    cp $DB_FILE "/app/backups/emergency_backup_$(date +%Y%m%d_%H%M%S).db"
else
    echo "$(date): Database integrity check passed" >> $LOG_FILE
fi

# Check file permissions
ACTUAL_PERMS=$(stat -c "%a" $DB_FILE)
if [ "$ACTUAL_PERMS" != "640" ]; then
    echo "$(date): WARNING: Database file permissions changed: $ACTUAL_PERMS" >> $LOG_FILE
    chmod 640 $DB_FILE
fi

echo "$(date): Database integrity check completed" >> $LOG_FILE
EOF

chmod +x /home/tradingbot/check_db_integrity.sh

# Run integrity check every hour
echo "0 * * * * /home/tradingbot/check_db_integrity.sh" | crontab -
```

---

## 6. 🚨 MONITORING & ALERTS

### **Current Status**: ✅ PARTIAL - Basic Monitoring Exists
### **Risk Level**: 🟡 MEDIUM - Need Security-Focused Monitoring

#### **A. Security Event Monitoring**

**Enhanced Security Monitoring Script**:

```bash
# Create comprehensive security monitoring
cat > /home/tradingbot/security_monitor.sh << 'EOF'
#!/bin/bash

LOG_FILE="/var/log/security-monitor.log"
ALERT_EMAIL="your_security_alerts@email.com"

# Function to send alerts
send_alert() {
    local message="$1"
    echo "$(date): SECURITY ALERT: $message" >> $LOG_FILE
    echo "$message" | mail -s "Security Alert - Trading Bot" $ALERT_EMAIL
}

# Check for failed SSH attempts
FAILED_SSH=$(grep "Failed password" /var/log/auth.log | grep "$(date '+%b %d')" | wc -l)
if [ $FAILED_SSH -gt 10 ]; then
    send_alert "High number of failed SSH attempts today: $FAILED_SSH"
fi

# Check for new processes listening on ports
NEW_LISTENERS=$(netstat -tulpn | grep LISTEN | grep -v ":22\|:80\|:443\|:8000\|:2222" | wc -l)
if [ $NEW_LISTENERS -gt 0 ]; then
    send_alert "Unexpected services listening on ports: $(netstat -tulpn | grep LISTEN | grep -v ':22\|:80\|:443\|:8000\|:2222')"
fi

# Check for unusual API activity
API_REQUESTS=$(grep "$(date '+%Y-%m-%d')" /var/log/nginx/access.log | wc -l)
if [ $API_REQUESTS -gt 10000 ]; then
    send_alert "Unusually high API activity: $API_REQUESTS requests today"
fi

# Check for file system changes in critical directories
if [ -f /tmp/critical_files_checksum ]; then
    CURRENT_CHECKSUM=$(find /app /etc/nginx /etc/ssh -type f -exec md5sum {} \; | sort | md5sum)
    STORED_CHECKSUM=$(cat /tmp/critical_files_checksum)
    
    if [ "$CURRENT_CHECKSUM" != "$STORED_CHECKSUM" ]; then
        send_alert "Critical system files have been modified"
    fi
else
    # Create initial checksum
    find /app /etc/nginx /etc/ssh -type f -exec md5sum {} \; | sort | md5sum > /tmp/critical_files_checksum
fi

# Check disk usage
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    send_alert "Critical disk usage: ${DISK_USAGE}%"
fi

# Check memory usage
MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
if [ $MEMORY_USAGE -gt 90 ]; then
    send_alert "Critical memory usage: ${MEMORY_USAGE}%"
fi

echo "$(date): Security monitoring check completed" >> $LOG_FILE
EOF

chmod +x /home/tradingbot/security_monitor.sh

# Run security monitoring every 10 minutes
echo "*/10 * * * * /home/tradingbot/security_monitor.sh" | crontab -
```

#### **B. Trading Activity Monitoring**

**Trading Security Monitor**:

```python
# Add to your trading bot application
import logging
from datetime import datetime, timedelta

class TradingSecurityMonitor:
    def __init__(self):
        self.trade_history = []
        self.alert_thresholds = {
            'max_hourly_trades': 50,
            'max_daily_volume': 100000,  # USDT
            'max_position_size': 10000,  # USDT
            'suspicious_profit_rate': 50  # % per day
        }
        
    async def monitor_trade(self, trade_data):
        """Monitor individual trades for security issues"""
        
        # Check trade size
        if trade_data['volume_usdt'] > self.alert_thresholds['max_position_size']:
            await self.send_security_alert(
                f"Large trade detected: {trade_data['volume_usdt']} USDT"
            )
            
        # Check for rapid trading
        recent_trades = [t for t in self.trade_history 
                        if t['timestamp'] > datetime.now() - timedelta(hours=1)]
        
        if len(recent_trades) > self.alert_thresholds['max_hourly_trades']:
            await self.send_security_alert(
                f"High frequency trading detected: {len(recent_trades)} trades in 1 hour"
            )
            
        # Store trade for monitoring
        self.trade_history.append({
            'timestamp': datetime.now(),
            'volume_usdt': trade_data['volume_usdt'],
            'symbol': trade_data['symbol'],
            'side': trade_data['side']
        })
        
        # Keep only recent trades in memory
        cutoff = datetime.now() - timedelta(days=1)
        self.trade_history = [t for t in self.trade_history if t['timestamp'] > cutoff]
        
    async def send_security_alert(self, message):
        """Send security alert"""
        logging.critical(f"TRADING SECURITY ALERT: {message}")
        # Add email/webhook notification here
```

#### **C. Real-Time Alerting**

**Webhook Alert System**:

```python
# Add webhook alerting to your application
import aiohttp
import os

class AlertManager:
    def __init__(self):
        self.webhook_url = os.getenv('SECURITY_WEBHOOK_URL')
        self.email_alerts = os.getenv('ALERT_EMAIL')
        
    async def send_critical_alert(self, alert_type: str, message: str, data: dict = None):
        """Send critical security alert"""
        
        alert_payload = {
            'timestamp': datetime.now().isoformat(),
            'severity': 'CRITICAL',
            'type': alert_type,
            'message': message,
            'data': data or {},
            'server': os.getenv('SERVER_NAME', 'trading-bot'),
        }
        
        # Send to webhook (Slack, Discord, etc.)
        if self.webhook_url:
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.webhook_url,
                        json=alert_payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
            except Exception as e:
                logging.error(f"Failed to send webhook alert: {e}")
                
        # Log critical alert
        logging.critical(f"CRITICAL ALERT [{alert_type}]: {message}")
        
    async def send_api_security_alert(self, client_ip: str, alert_type: str):
        """Send API security specific alerts"""
        
        await self.send_critical_alert(
            alert_type='API_SECURITY',
            message=f"API security event: {alert_type}",
            data={
                'client_ip': client_ip,
                'alert_type': alert_type,
                'user_agent': 'detected_via_request_headers'
            }
        )

# Use in your API endpoints
alert_manager = AlertManager()

@app.middleware("http")
async def security_monitoring_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    # Monitor for suspicious activity
    if request.url.path.startswith('/api/') and request.method in ['POST', 'PUT', 'DELETE']:
        # Log all trading-related API calls
        logging.info(f"API call from {client_ip}: {request.method} {request.url.path}")
        
    response = await call_next(request)
    
    # Alert on error responses that might indicate attacks
    if response.status_code in [401, 403, 429]:
        await alert_manager.send_api_security_alert(client_ip, f"HTTP_{response.status_code}")
    
    return response
```

---

## 7. 📝 AUDIT & COMPLIANCE

### **Current Status**: ✅ PARTIAL - Basic Logging Exists  
### **Risk Level**: 🟡 MEDIUM - Need Comprehensive Audit Trail

#### **A. Comprehensive Logging**

**Enhanced Logging Configuration**:

```python
# Add to your main.py application
import logging.handlers
import json
from datetime import datetime

class SecurityAuditLogger:
    def __init__(self):
        # Create security-specific logger
        self.security_logger = logging.getLogger('security_audit')
        self.security_logger.setLevel(logging.INFO)
        
        # Create rotating file handler for security events
        handler = logging.handlers.RotatingFileHandler(
            '/var/log/trading-security-audit.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        
        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.security_logger.addHandler(handler)
        
    def log_security_event(self, event_type: str, details: dict):
        """Log security event in structured format"""
        
        security_event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'server_id': os.getenv('SERVER_NAME', 'trading-bot')
        }
        
        self.security_logger.info(json.dumps(security_event))
        
    def log_api_access(self, client_ip: str, endpoint: str, method: str, status_code: int, user_id: str = None):
        """Log API access for audit trail"""
        
        self.log_security_event('API_ACCESS', {
            'client_ip': client_ip,
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })
        
    def log_trading_action(self, action_type: str, symbol: str, quantity: float, price: float, user_id: str = None):
        """Log trading actions for compliance"""
        
        self.log_security_event('TRADING_ACTION', {
            'action_type': action_type,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })

# Initialize security audit logger
security_audit = SecurityAuditLogger()

# Add to your API endpoints
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Log all API access
    security_audit.log_api_access(
        client_ip=request.client.host,
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        user_id=getattr(request.state, 'user_id', None)
    )
    
    # Log slow requests (potential DoS)
    duration = time.time() - start_time
    if duration > 5.0:  # 5 seconds
        security_audit.log_security_event('SLOW_REQUEST', {
            'endpoint': request.url.path,
            'duration': duration,
            'client_ip': request.client.host
        })
    
    return response
```

#### **B. Compliance Documentation**

**Create Compliance Documentation**:

```bash
# Create compliance directory
mkdir -p /app/compliance

# Create compliance checklist
cat > /app/compliance/security_checklist.md << 'EOF'
# Security Compliance Checklist

## Daily Checks ✓
- [ ] Review security audit logs
- [ ] Check failed authentication attempts
- [ ] Verify backup completion
- [ ] Monitor disk space usage
- [ ] Check SSL certificate expiry

## Weekly Checks ✓
- [ ] Review trading activity patterns
- [ ] Check system update status
- [ ] Verify firewall rules
- [ ] Test backup restoration
- [ ] Review access logs for anomalies

## Monthly Checks ✓
- [ ] Security patch updates
- [ ] Password rotation (if applicable)
- [ ] SSL certificate renewal check
- [ ] Review and update security policies
- [ ] Penetration testing (if required)

## Quarterly Checks ✓
- [ ] Full security audit
- [ ] Review API key permissions
- [ ] Update incident response plan
- [ ] Security awareness review
- [ ] Compliance documentation update
EOF
```

#### **C. Incident Response Plan**

**Create Incident Response Plan**:

```bash
cat > /app/compliance/incident_response_plan.md << 'EOF'
# Security Incident Response Plan

## Immediate Response (0-15 minutes)

### API Key Compromise
1. **IMMEDIATE**: Disable compromised API keys in Bybit dashboard
2. **IMMEDIATE**: Stop trading bot application
3. **IMMEDIATE**: Change all passwords and generate new API keys
4. **5 min**: Review recent trading activity for unauthorized trades
5. **10 min**: Check account balances and positions
6. **15 min**: Document incident details

### Server Compromise  
1. **IMMEDIATE**: Disconnect server from internet (if safe)
2. **IMMEDIATE**: Stop all trading applications
3. **5 min**: Create system snapshot/backup
4. **10 min**: Review logs for intrusion evidence
5. **15 min**: Contact hosting provider if needed

### Database Breach
1. **IMMEDIATE**: Stop application access to database
2. **5 min**: Create emergency backup
3. **10 min**: Review database logs
4. **15 min**: Assess data exposure scope

## Short-term Response (15 minutes - 1 hour)

### Investigation Phase
- Collect and preserve logs
- Identify attack vectors
- Assess damage scope
- Document evidence
- Notify relevant parties

### Containment Phase  
- Patch security vulnerabilities
- Reset all credentials
- Update firewall rules
- Apply security updates
- Monitor for continued threats

## Recovery Phase (1+ hours)

### System Restoration
- Restore from clean backups
- Apply all security patches
- Reconfigure security settings
- Test all functionality
- Gradual service restoration

### Monitoring Phase
- Enhanced monitoring
- Regular security checks
- Performance monitoring
- User communication
- Documentation updates

## Post-Incident (24+ hours)

### Analysis
- Root cause analysis
- Timeline reconstruction
- Impact assessment
- Response evaluation
- Lessons learned

### Improvements
- Update security policies
- Implement additional controls
- Staff training updates
- Tool improvements
- Communication improvements

## Contact Information

**Emergency Contacts:**
- Bybit Support: support@bybit.com
- DigitalOcean Support: support ticket system
- Security Team: your_security_email@domain.com

**Important Links:**
- Bybit API Management: https://www.bybit.com/app/user/api-management
- DigitalOcean Console: https://cloud.digitalocean.com/
- Server Access: ssh tradingbot@your_server_ip -p 2222
EOF
```

---

## 🎯 SECURITY IMPLEMENTATION CHECKLIST

### **Phase 1: Critical (Deploy First)**
- [ ] ✅ Configure dual environment API keys in DigitalOcean
- [ ] ✅ Set strong DASHBOARD_PASSWORD (16+ characters)  
- [ ] ✅ Enable IP restrictions on Bybit API keys
- [ ] ✅ Configure firewall (UFW) with only necessary ports
- [ ] ✅ Change SSH port from 22 to 2222
- [ ] ✅ Disable root SSH login

### **Phase 2: Essential (Deploy Within 24 hours)**
- [ ] Install and configure Nginx reverse proxy
- [ ] Setup SSL certificate with Let's Encrypt
- [ ] Configure Fail2ban intrusion detection
- [ ] Setup automated security updates
- [ ] Configure encrypted database backups
- [ ] Implement rate limiting

### **Phase 3: Enhanced (Deploy Within 1 Week)**
- [ ] Setup comprehensive security monitoring
- [ ] Implement security audit logging
- [ ] Configure real-time alerting
- [ ] Setup integrity monitoring
- [ ] Create incident response procedures
- [ ] Configure compliance documentation

### **Phase 4: Advanced (Deploy Within 1 Month)**
- [ ] Implement advanced authentication
- [ ] Setup security metrics dashboard
- [ ] Configure automated threat response
- [ ] Setup external security monitoring
- [ ] Implement security testing automation
- [ ] Regular security assessments

---

## ⚠️ CRITICAL SECURITY WARNINGS

### **🔴 NEVER DO THESE:**

1. **Never store API keys in code or config files**
2. **Never use default passwords in production**
3. **Never allow direct database access from internet**
4. **Never run trading bot as root user**
5. **Never disable SSL/HTTPS in production**
6. **Never ignore security alerts or unusual activity**
7. **Never deploy without firewall configuration**
8. **Never use the same API keys for testing and production**

### **🟡 REGULAR MONITORING REQUIRED:**

1. **Daily**: Check security logs and failed login attempts
2. **Weekly**: Review trading activity and system performance  
3. **Monthly**: Update security patches and review configurations
4. **Quarterly**: Full security audit and penetration testing

---

## 🛡️ DEPLOYMENT SECURITY COMMAND SUMMARY

**Quick Deploy Security Essentials:**

```bash
# 1. Initial Security Setup (Run First)
sudo ufw enable
sudo ufw allow 2222/tcp
sudo ufw allow 80/tcp  
sudo ufw allow 443/tcp
sudo sed -i 's/Port 22/Port 2222/' /etc/ssh/sshd_config
sudo sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

# 2. Install Security Tools
sudo apt update && sudo apt upgrade -y
sudo apt install nginx certbot python3-certbot-nginx fail2ban ufw -y

# 3. Setup SSL (Replace with your domain)
sudo certbot --nginx -d your_domain.com

# 4. Start Security Services
sudo systemctl enable fail2ban nginx
sudo systemctl start fail2ban nginx

# 5. Set File Permissions
sudo chmod 750 /app/data
sudo chmod 640 /app/data/trading_bot.db
```

**Your trading bot is now production-ready with enterprise-grade security! 🚀**

---

*Total Implementation Time: 4-8 hours for complete security hardening*
*Security Level: Enterprise Grade 🛡️*
*Cost Impact: $0 additional (all open-source tools)*