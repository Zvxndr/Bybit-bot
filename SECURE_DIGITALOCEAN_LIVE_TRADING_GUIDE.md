# 🛡️ SECURE DIGITALOCEAN DROPLET FOR LIVE TRADING
## Complete Security Hardening Guide for Australian Tax Compliant Bybit Bot

### 🎯 **MISSION: SECURE LIVE TRADING ENVIRONMENT**

This guide will help you secure your DigitalOcean droplet to safely run live cryptocurrency trading with your Australian tax-compliant Bybit bot.

---

## 🚀 **STEP-BY-STEP SECURITY SETUP**

### **Phase 1: Droplet Creation & Initial Security**

#### **1.1 Create Production Droplet**
```bash
# Recommended Specifications for Live Trading
Size: Premium Intel - 2 vCPU, 4 GB RAM, 80 GB SSD ($24/month)
Region: Sydney 1 (syd1) - Australian data residency  
OS: Ubuntu 22.04 LTS x64
Backups: Enable automatic backups
Monitoring: Enable enhanced monitoring
```

#### **1.2 Initial Server Access**
```bash
# Connect to your new droplet
ssh root@your_droplet_ip

# Update system packages immediately
apt update && apt upgrade -y

# Install essential security tools
apt install -y ufw fail2ban nginx certbot python3-certbot-nginx htop git curl
```

---

### **Phase 2: User Security & SSH Hardening**

#### **2.1 Create Dedicated Trading User**
```bash
# Create secure trading user (never use root for trading)
adduser tradingbot
usermod -aG sudo tradingbot

# Set up SSH key authentication
mkdir -p /home/tradingbot/.ssh
chmod 700 /home/tradingbot/.ssh

# Copy your public key to authorized_keys
nano /home/tradingbot/.ssh/authorized_keys
# Paste your public SSH key here

chmod 600 /home/tradingbot/.ssh/authorized_keys
chown -R tradingbot:tradingbot /home/tradingbot/.ssh
```

#### **2.2 Harden SSH Configuration**
```bash
# Edit SSH configuration
nano /etc/ssh/sshd_config

# Apply these critical security settings:
```

```ini
# SSH Security Configuration for Live Trading
Port 2222                          # Change from default port 22
PasswordAuthentication no           # Key-only authentication
PubkeyAuthentication yes           
PermitRootLogin no                 # Never allow root login
MaxAuthTries 3                     # Limit brute force attempts
ClientAliveInterval 300            # Timeout idle connections
ClientAliveCountMax 2
AllowUsers tradingbot              # Only allow your trading user
Protocol 2
X11Forwarding no
AllowTcpForwarding no
```

```bash
# Restart SSH with new settings
systemctl restart sshd

# Test new SSH connection (don't close current session until verified)
# From new terminal: ssh -p 2222 tradingbot@your_droplet_ip
```

---

### **Phase 3: Firewall & Network Security**

#### **3.1 Configure UFW Firewall**
```bash
# Default deny all incoming, allow outgoing
ufw default deny incoming
ufw default allow outgoing

# Allow only essential ports
ufw allow 2222/tcp                 # SSH (custom port)
ufw allow 443/tcp                  # HTTPS (trading dashboard)
ufw allow 80/tcp                   # HTTP (redirect to HTTPS)

# Enable firewall
ufw enable

# Check firewall status
ufw status verbose
```

#### **3.2 Setup Fail2Ban (Anti-Brute Force)**
```bash
# Configure Fail2Ban
nano /etc/fail2ban/jail.local
```

```ini
[DEFAULT]
bantime = 3600              # Ban for 1 hour
findtime = 600              # 10 minute window  
maxretry = 3                # 3 attempts before ban
ignoreip = 127.0.0.1/8      # Don't ban localhost

[sshd]
enabled = true
port = 2222                 # Your custom SSH port
logpath = /var/log/auth.log
maxretry = 3

[nginx-limit-req]  
enabled = true
filter = nginx-limit-req
action = iptables-multiport[name=ReqLimit, port="http,https", protocol=tcp]
logpath = /var/log/nginx/error.log
```

```bash
# Start and enable Fail2Ban
systemctl enable fail2ban
systemctl start fail2ban

# Check banned IPs
fail2ban-client status sshd
```

---

### **Phase 4: SSL/TLS & Web Security**

#### **4.1 Setup Nginx Reverse Proxy**
```bash
# Create Nginx configuration for your domain
nano /etc/nginx/sites-available/trading-bot
```

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL Configuration  
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=trading:10m rate=10r/m;
    limit_req zone=trading burst=5 nodelay;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for real-time dashboard
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

```bash
# Enable site and get SSL certificate
ln -s /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
nginx -t  # Test configuration
systemctl reload nginx

# Get free SSL certificate
certbot --nginx -d your-domain.com -d www.your-domain.com
```

---

### **Phase 5: Application Security**

#### **5.1 Secure Environment Setup**
```bash
# Create secure application directory
mkdir -p /opt/trading
chown tradingbot:tradingbot /opt/trading
chmod 750 /opt/trading

# Switch to trading user
su - tradingbot
cd /opt/trading

# Clone your trading bot
git clone https://github.com/Zvxndr/Bybit-bot.git .
```

#### **5.2 Environment Variables (CRITICAL)**
```bash
# Create secure environment file
nano /opt/trading/.env.production
```

```env
# 🔴 LIVE TRADING API KEYS - HANDLE WITH EXTREME CARE
BYBIT_API_KEY="YOUR_LIVE_API_KEY"
BYBIT_API_SECRET="YOUR_LIVE_API_SECRET"
BYBIT_TESTNET=false

# 🟡 Paper Trading Keys (for testing)  
BYBIT_TESTNET_API_KEY="YOUR_TESTNET_KEY"
BYBIT_TESTNET_API_SECRET="YOUR_TESTNET_SECRET"

# 🇦🇺 Australian Tax Compliance
TIMEZONE="Australia/Sydney"
TAX_COMPLIANCE_MODE="production" 
ATO_REPORTING_ENABLED=true
FINANCIAL_YEAR="2025-26"

# 🛡️ Security Configuration
SECRET_KEY="$(openssl rand -base64 32)"
API_RATE_LIMIT="100/hour"
MAX_LOGIN_ATTEMPTS=3

# 📊 Database & Logging
DATABASE_PATH="/opt/trading/data/trading_bot_prod.db"
LOG_LEVEL="INFO"
LOG_RETENTION_DAYS=2555  # 7 years for Australian tax compliance

# 🚨 Emergency Controls  
MAX_DAILY_LOSS_AUD=1000.00          # Conservative limit
MAX_POSITION_SIZE_PERCENT=2.0        # 2% maximum position size
EMERGENCY_STOP_THRESHOLD=5.0         # 5% daily loss triggers emergency stop
RISK_MANAGEMENT_MODE="conservative"

# 📧 Alert Configuration
SMTP_SERVER="smtp.gmail.com"
SMTP_PORT=587
SMTP_USERNAME="your_alerts@gmail.com"
SMTP_PASSWORD="your_app_specific_password"
ALERT_EMAIL="your_phone_sms@carrier.com"  # SMS alerts

# 🔍 Monitoring
ENABLE_REAL_TIME_MONITORING=true
HEALTH_CHECK_INTERVAL=60
INFRASTRUCTURE_ALERTS=true
```

```bash
# Secure the environment file (CRITICAL)
chmod 600 /opt/trading/.env.production
chown tradingbot:tradingbot /opt/trading/.env.production

# Verify no one else can read it
ls -la /opt/trading/.env.production
# Should show: -rw------- 1 tradingbot tradingbot
```

---

### **Phase 6: Live Trading Safety Setup**

#### **6.1 Bybit API Security Configuration**

**🚨 CRITICAL: Set up your live Bybit API keys with RESTRICTED permissions:**

1. **Login to Bybit Mainnet** (bybit.com)
2. **API Management** → **Create New Key**
3. **Key Permissions** (IMPORTANT - Only enable what you need):
   ```
   ✅ Read Position          # Required for portfolio tracking
   ✅ Read Wallet           # Required for balance information  
   ✅ Trade                 # Required for placing orders
   ❌ Transfer              # DISABLE - No fund transfers
   ❌ Sub Account           # DISABLE - No sub-account access  
   ❌ Block Trading         # DISABLE - No block trading
   ❌ SPOT                  # DISABLE unless you trade spot
   ❌ Options               # DISABLE unless you trade options
   ```

4. **IP Whitelist** (MANDATORY):
   ```
   Add your DigitalOcean droplet IP: YOUR_DROPLET_IP
   Remove any other IPs from whitelist
   ```

5. **Daily Trading Limits**:
   ```
   Set daily trading volume limit: $10,000 AUD (or your comfort level)
   Set daily withdrawal limit: $0 (disable withdrawals via API)
   ```

#### **6.2 Emergency Procedures Setup**
```bash
# Create emergency stop script
nano /opt/trading/emergency_stop.sh
```

```bash
#!/bin/bash
# Emergency Stop Script for Live Trading

echo "🚨 EMERGENCY STOP INITIATED - $(date)"

# Stop trading bot immediately  
pkill -f "python -m src.main"
pkill -f "python main.py"

# Log emergency stop
echo "$(date): EMERGENCY STOP - All trading halted" >> /opt/trading/logs/emergency.log

# Send immediate alert
curl -X POST "http://localhost:8080/api/emergency-stop" 2>/dev/null || echo "Bot already stopped"

# Optional: Send SMS alert (if configured)
# curl -X POST "https://api.twilio.com/..." # Your SMS service

echo "🛑 EMERGENCY STOP COMPLETE"
echo "📧 Check your email for emergency stop confirmation"
echo "💻 Access dashboard to restart when safe: https://your-domain.com"
```

```bash
# Make executable
chmod +x /opt/trading/emergency_stop.sh

# Create desktop shortcut for emergency access
echo '#!/bin/bash
ssh -p 2222 tradingbot@your_droplet_ip "/opt/trading/emergency_stop.sh"
' > ~/emergency_stop_trading.sh

chmod +x ~/emergency_stop_trading.sh
```

---

### **Phase 7: Monitoring & Alerts**

#### **7.1 System Monitoring Setup**
```bash
# Install monitoring tools
sudo apt install -y htop iotop nethogs

# Create system monitoring script
nano /opt/trading/monitor_system.sh
```

```bash
#!/bin/bash
# System Health Monitor

LOG_FILE="/opt/trading/logs/system_health.log"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Check CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    
    # Check memory usage
    MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.1f", ($3/$2) * 100.0)}')
    
    # Check disk usage  
    DISK_USAGE=$(df /opt/trading | tail -1 | awk '{print $5}' | sed 's/%//')
    
    # Check if trading bot is running
    BOT_RUNNING=$(pgrep -f "python.*main.py" | wc -l)
    
    # Log status
    echo "$TIMESTAMP - CPU: ${CPU_USAGE}% | Memory: ${MEMORY_USAGE}% | Disk: ${DISK_USAGE}% | Bot: $BOT_RUNNING" >> $LOG_FILE
    
    # Alert if critical thresholds exceeded
    if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
        echo "🚨 HIGH CPU USAGE: $CPU_USAGE%" | mail -s "Trading Bot Alert" your_alerts@gmail.com
    fi
    
    if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
        echo "🚨 HIGH MEMORY USAGE: $MEMORY_USAGE%" | mail -s "Trading Bot Alert" your_alerts@gmail.com  
    fi
    
    if [ "$BOT_RUNNING" -eq 0 ]; then
        echo "🚨 TRADING BOT NOT RUNNING!" | mail -s "CRITICAL: Trading Bot Down" your_alerts@gmail.com
    fi
    
    sleep 300  # Check every 5 minutes
done
```

```bash
# Make executable and start monitoring
chmod +x /opt/trading/monitor_system.sh

# Create systemd service for monitoring
sudo nano /etc/systemd/system/trading-monitor.service
```

```ini
[Unit]
Description=Trading Bot System Monitor
After=network.target

[Service]
Type=simple  
User=tradingbot
WorkingDirectory=/opt/trading
ExecStart=/opt/trading/monitor_system.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable monitoring service
sudo systemctl enable trading-monitor.service
sudo systemctl start trading-monitor.service
```

---

### **Phase 8: Backup & Recovery**

#### **8.1 Automated Backup System**
```bash
# Create backup script
nano /opt/trading/backup_system.sh
```

```bash
#!/bin/bash
# Automated Backup for Trading Bot

BACKUP_DIR="/opt/trading/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="trading_backup_$DATE.tar.gz"

mkdir -p $BACKUP_DIR

# Stop bot temporarily for consistent backup
pkill -f "python -m src.main"
sleep 5

# Backup critical data
tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
    /opt/trading/data/ \
    /opt/trading/config/ \
    /opt/trading/logs/ \
    /opt/trading/.env.production

# Restart bot
cd /opt/trading && python -m src.main &

# Keep only last 30 backups (30 days)
find $BACKUP_DIR -name "trading_backup_*.tar.gz" -mtime +30 -delete

echo "✅ Backup completed: $BACKUP_FILE"
```

```bash
# Schedule daily backups
crontab -e
# Add: 0 2 * * * /opt/trading/backup_system.sh
```

---

### **Phase 9: Final Security Verification**

#### **9.1 Security Checklist Before Live Trading**
```bash
# Run this security audit before going live
nano /opt/trading/security_audit.sh
```

```bash
#!/bin/bash
# Security Audit Checklist

echo "🔍 SECURITY AUDIT FOR LIVE TRADING"
echo "================================="

# Check SSH configuration
echo "✅ SSH Security:"
grep "PasswordAuthentication no" /etc/ssh/sshd_config && echo "   Password auth disabled ✓" || echo "   ❌ Password auth still enabled!"  
grep "PermitRootLogin no" /etc/ssh/sshd_config && echo "   Root login disabled ✓" || echo "   ❌ Root login still enabled!"

# Check firewall
echo "✅ Firewall Status:"
ufw status | grep "Status: active" && echo "   UFW active ✓" || echo "   ❌ UFW not active!"

# Check file permissions
echo "✅ File Permissions:"
PERM=$(stat -c "%a" /opt/trading/.env.production 2>/dev/null)
[[ "$PERM" == "600" ]] && echo "   Environment file secure ✓" || echo "   ❌ Environment file permissions insecure!"

# Check API key configuration
echo "✅ API Configuration:"
grep -q "BYBIT_API_KEY=" /opt/trading/.env.production && echo "   Live API key configured ✓" || echo "   ❌ Live API key missing!"
grep -q "MAX_DAILY_LOSS_AUD=" /opt/trading/.env.production && echo "   Loss limits configured ✓" || echo "   ❌ Loss limits missing!"

# Check monitoring  
echo "✅ Monitoring:"
systemctl is-active trading-monitor.service && echo "   System monitoring active ✓" || echo "   ❌ System monitoring not running!"

# Check backup system
echo "✅ Backup System:"
[[ -x /opt/trading/backup_system.sh ]] && echo "   Backup script ready ✓" || echo "   ❌ Backup script missing!"

# Check emergency procedures
echo "✅ Emergency Procedures:"
[[ -x /opt/trading/emergency_stop.sh ]] && echo "   Emergency stop ready ✓" || echo "   ❌ Emergency stop missing!"

echo "================================="
echo "🚨 REVIEW ALL ITEMS BEFORE LIVE TRADING!"
```

```bash
chmod +x /opt/trading/security_audit.sh
/opt/trading/security_audit.sh
```

---

## 🚀 **FINAL DEPLOYMENT & GO-LIVE PROCEDURES**

### **Step 1: Deploy Trading Bot**
```bash
# Switch to trading user and deploy
su - tradingbot
cd /opt/trading

# Install Python dependencies
python3 -m pip install --user -r requirements.txt

# Load production environment  
export $(grep -v '^#' .env.production | xargs)

# Test run (paper mode first)
python -m src.main
```

### **Step 2: Verify Dashboard Access**
```
🌐 Access your secure trading dashboard:
https://your-domain.com

✅ Verify all systems showing green
✅ Test emergency stop button
✅ Check Australian tax compliance logs
✅ Confirm 3-phase balance system working
```

### **Step 3: Conservative Live Trading Start**
```bash
# Start with minimal position sizes
# Edit in dashboard or .env.production:
MAX_POSITION_SIZE_PERCENT=0.5    # Start with 0.5% positions
MAX_DAILY_LOSS_AUD=100.00        # Very conservative daily limit
RISK_MANAGEMENT_MODE="ultra_conservative"
```

---

## ⚠️ **CRITICAL SAFETY REMINDERS**

### 🚨 **BEFORE ENABLING LIVE TRADING:**
1. **✅ Test emergency stop procedure multiple times**
2. **✅ Verify Australian tax compliance is logging correctly**  
3. **✅ Start with paper trading for at least 1 week**
4. **✅ Set up SMS/email alerts and test them**
5. **✅ Have emergency stop script readily accessible**
6. **✅ Monitor first live trades closely for 24-48 hours**

### 🇦🇺 **Australian Compliance Verification:**
- **✅ Timezone set to Australia/Sydney**
- **✅ Financial year 2025-26 detected correctly** 
- **✅ Tax logs downloading in ATO-ready format**
- **✅ FIFO cost basis calculations working**
- **✅ 7-year retention system active**

---

**🛡️ Your DigitalOcean droplet is now secured for safe live cryptocurrency trading with full Australian tax compliance! 🇦🇺**

**Remember: Always start conservative, monitor closely, and never risk more than you can afford to lose.**