# 🌊 DIGITALOCEAN SETUP WALKTHROUGH
## Step-by-Step Implementation Guide for Secure Live Trading

### 🎯 **OVERVIEW**
This walkthrough will take you from zero to a fully secured DigitalOcean droplet running your Australian tax-compliant Bybit trading bot.

**Estimated Setup Time: 45-60 minutes**

---

## 📋 **PREREQUISITES CHECKLIST**

Before starting, ensure you have:
- [ ] DigitalOcean account with payment method added  
- [ ] Domain name (optional but recommended) - e.g., from Namecheap, GoDaddy
- [ ] Your Bybit API keys (testnet first, live keys ready)
- [ ] Strong passwords ready for server accounts
- [ ] This trading bot repository ready

## 🛡️ **SECURITY APPROACH: CONSOLE-ONLY ACCESS**

**Why skip SSH keys?**
✅ **Simpler Setup** - No key management or complex authentication
✅ **Always Accessible** - DigitalOcean console works from any browser
✅ **No External Attack Surface** - SSH can be disabled entirely  
✅ **Protected by DigitalOcean Security** - Your account 2FA protects access
✅ **Perfect for Trading Bots** - Once configured, minimal server interaction needed

**This approach is ideal for:**
- Users who prefer simplicity over complex key management
- Trading bots that run autonomously after setup
- Maximum security through minimal external access points

---

## 🚀 **STEP 1: CREATE DIGITALOCEAN DROPLET**

### **1.1 Login to DigitalOcean**
1. Go to [cloud.digitalocean.com](https://cloud.digitalocean.com)
2. Click **"Create"** → **"Droplets"**

### **1.2 Choose Droplet Configuration**
```
📍 Region: Sydney 1 (syd1) - Australian data residency
📦 Image: Ubuntu 22.04 (LTS) x64
💾 Size: Premium Intel
   └── 2 vCPU, 4 GB RAM, 80 GB SSD ($24/month)
🔐 Authentication: SSH Keys (we'll add this)
```

### **1.3 Authentication Method (Password-Based)**
**For simplified security using DigitalOcean console access:**
- We'll skip SSH keys and use strong passwords
- Access will be via DigitalOcean's secure web console
- Additional security through IP whitelisting and firewall rules

### **1.4 Finalize Droplet Creation**
```
🔐 Authentication: Password (we'll use strong passwords + console access)
🏷️ Hostname: trading-bot-prod
🔧 Additional Options:
   ✅ Enable backups (+$4.80/month) - RECOMMENDED
   ✅ Enable monitoring (free)
   ❌ Skip IPv6, VPC, User data for now
```

**Click "Create Droplet" - This takes 1-2 minutes**

---

## 🔐 **STEP 2: INITIAL SERVER SECURITY**

### **2.1 First Connection**
**Using DigitalOcean Console (Secure & Simple):**
1. Go to your DigitalOcean dashboard
2. Click on your droplet name
3. Click **"Console"** in the left sidebar
4. This opens a secure web-based terminal
5. Login as `root` with the password DigitalOcean emailed you

**Alternative - SSH from Windows (if you prefer):**
```powershell
# Connect to your droplet (replace with your droplet IP)
ssh root@YOUR_DROPLET_IP
# Enter the root password when prompted
```

### **2.2 Immediate Security Updates**
```bash
# Update all packages immediately
apt update && apt upgrade -y

# Install essential security tools
apt install -y ufw fail2ban nginx certbot python3-certbot-nginx htop git curl python3-pip python3-venv

# Install mail utilities for alerts
apt install -y mailutils postfix
# When prompted for postfix config: Select "Internet Site"
# System mail name: Use your domain or droplet name
```

### **2.3 Create Trading User Account**
```bash
# Create secure user (NEVER trade as root)
adduser tradingbot
# Enter a STRONG password when prompted (save this password securely!)
# Fill in user information or press Enter to skip

# Add to sudo group
usermod -aG sudo tradingbot
```

### **2.4 Test New User Access**
**In the same DigitalOcean console:**
```bash
# Switch to new user to test
su - tradingbot
# Enter the password you just created

# If successful, you should see: tradingbot@trading-bot-prod:~$
```

**✅ If this works, continue. If not, troubleshoot before proceeding.**

---

## 🛡️ **STEP 3: HARDEN SSH SECURITY**

### **3.1 Configure SSH Security (Password-Based)**
**In your DigitalOcean console as tradingbot:**
```bash
# Edit SSH configuration for secure password access
sudo nano /etc/ssh/sshd_config

# Find and modify these lines (use Ctrl+W to search):
```

**Add/modify these settings for secure password authentication:**
```ini
# Change port from default 22 (security through obscurity)
Port 2222

# Keep password authentication but make it secure
PasswordAuthentication yes
PubkeyAuthentication yes

# Disable root login (CRITICAL)
PermitRootLogin no

# Limit failed attempts (prevent brute force)
MaxAuthTries 3

# Timeout idle connections
ClientAliveInterval 300
ClientAliveCountMax 2

# Only allow our trading user
AllowUsers tradingbot

# Disable unnecessary features
X11Forwarding no
AllowTcpForwarding no

# Additional security for password auth
LoginGraceTime 60
MaxStartups 2
```

**Save and exit:** `Ctrl+X`, then `Y`, then `Enter`

### **3.2 Apply SSH Changes**
```bash
# Test configuration
sudo sshd -t

# If no errors, restart SSH
sudo systemctl restart sshd
```

### **3.3 Test New SSH Configuration**
**Test the new configuration (if you want to use SSH from your machine):**
```powershell
# From Windows PowerShell - test connection on new port
ssh -p 2222 tradingbot@YOUR_DROPLET_IP
# Enter your tradingbot password when prompted
```

**🚨 RECOMMENDED: Continue using DigitalOcean console for maximum security**
- The console access is always available through your DigitalOcean account
- No external network access required
- Protected by your DigitalOcean login credentials

### **3.4 Enhanced Security Without SSH Keys**
```bash
# Disable SSH entirely for maximum security (optional)
sudo systemctl stop ssh
sudo systemctl disable ssh

# This forces all access through DigitalOcean console only
# You can re-enable anytime with: sudo systemctl enable ssh && sudo systemctl start ssh
```

**🛡️ Benefits of Console-Only Access:**
- **No external SSH attack surface**
- **Always accessible through DigitalOcean dashboard**  
- **Protected by 2FA on your DigitalOcean account**
- **No need to manage SSH keys or ports**

---

## 🔥 **STEP 4: CONFIGURE FIREWALL**

### **4.1 Set Up UFW Firewall**
```bash
# Default policies: deny incoming, allow outgoing
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow essential ports only
sudo ufw allow 2222/tcp comment 'SSH'
sudo ufw allow 80/tcp comment 'HTTP'
sudo ufw allow 443/tcp comment 'HTTPS'

# Enable firewall
sudo ufw enable
# Type "y" when prompted

# Verify configuration
sudo ufw status verbose
```

### **4.2 Configure Fail2Ban**
```bash
# Create Fail2Ban configuration
sudo nano /etc/fail2ban/jail.local
```

**Add this configuration:**
```ini
[DEFAULT]
bantime = 3600
findtime = 600  
maxretry = 3
ignoreip = 127.0.0.1/8

[sshd]
enabled = true
port = 2222
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
action = iptables-multiport[name=ReqLimit, port="http,https", protocol=tcp]
logpath = /var/log/nginx/error.log
```

**Save and start Fail2Ban:**
```bash
sudo systemctl enable fail2ban
sudo systemctl restart fail2ban

# Verify it's running
sudo fail2ban-client status
```

---

## 🌐 **STEP 5: DOMAIN & SSL SETUP (Optional but Recommended)**

### **5.1 Configure Domain DNS (if you have one)**
**In your domain registrar (Namecheap, GoDaddy, etc.):**
1. Go to DNS Management
2. Add A Record:
   - **Host**: `@` (or your subdomain like `trading`)
   - **Value**: Your droplet IP address
   - **TTL**: Automatic or 300

**Wait 5-10 minutes for DNS propagation**

### **5.2 Set Up Nginx Reverse Proxy**
```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/trading-bot
```

**If you have a domain, use this config:**
```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL certificates will be added by certbot
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=trading:10m rate=10r/m;
    limit_req zone=trading burst=5 nodelay;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

**If you DON'T have a domain, use this simpler config:**
```nginx
server {
    listen 80;
    server_name _;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=trading:10m rate=10r/m;
    limit_req zone=trading burst=5 nodelay;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### **5.3 Enable Nginx Configuration**
```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/

# Remove default site
sudo rm -f /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# If OK, reload nginx
sudo systemctl reload nginx
```

### **5.4 Get SSL Certificate (if you have domain)**
```bash
# Get free Let's Encrypt certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Follow prompts:
# - Enter email for notifications
# - Agree to terms: Y
# - Share email with EFF: A (your choice)
# - Redirect HTTP to HTTPS: 2 (Yes, recommended)
```

---

## 🤖 **STEP 6: DEPLOY TRADING BOT**

### **6.1 Create Application Directory**
```bash
# Create secure directory for trading bot
sudo mkdir -p /opt/trading
sudo chown tradingbot:tradingbot /opt/trading
sudo chmod 750 /opt/trading

# Switch to app directory
cd /opt/trading
```

### **6.2 Clone Your Trading Bot**
```bash
# Clone the repository
git clone https://github.com/Zvxndr/Bybit-bot.git .

# Make sure we're on the main branch
git checkout main
git pull origin main
```

### **6.3 Set Up Python Environment**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### **6.4 Create Production Environment File**
```bash
# Create production environment configuration
nano /opt/trading/.env.production
```

**Add this configuration (customize the values):**
```env
# 🔴 LIVE TRADING API KEYS - HANDLE WITH EXTREME CARE
BYBIT_API_KEY=""                    # Add your LIVE API key here
BYBIT_API_SECRET=""                 # Add your LIVE API secret here
BYBIT_TESTNET=false

# 🟡 Paper Trading Keys (for initial testing)
BYBIT_TESTNET_API_KEY=""            # Add your testnet key here
BYBIT_TESTNET_API_SECRET=""         # Add your testnet secret here

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
LOG_RETENTION_DAYS=2555

# 🚨 Emergency Controls - START CONSERVATIVE!
MAX_DAILY_LOSS_AUD=100.00           # Very conservative start
MAX_POSITION_SIZE_PERCENT=0.5       # 0.5% position sizes
EMERGENCY_STOP_THRESHOLD=2.0        # 2% daily loss triggers stop
RISK_MANAGEMENT_MODE="ultra_conservative"

# 📧 Alert Configuration (add your email)
SMTP_SERVER="smtp.gmail.com"
SMTP_PORT=587
SMTP_USERNAME="your_alerts@gmail.com"
SMTP_PASSWORD="your_app_specific_password"
ALERT_EMAIL="your_phone_sms@carrier.com"

# 🔍 Monitoring
ENABLE_REAL_TIME_MONITORING=true
HEALTH_CHECK_INTERVAL=60
INFRASTRUCTURE_ALERTS=true
```

**🚨 CRITICAL: Secure this file:**
```bash
# Set secure permissions (only tradingbot can read)
chmod 600 /opt/trading/.env.production

# Verify permissions
ls -la /opt/trading/.env.production
# Should show: -rw------- 1 tradingbot tradingbot
```

### **6.5 Create Required Directories**
```bash
# Create necessary directories
mkdir -p /opt/trading/data
mkdir -p /opt/trading/logs
mkdir -p /opt/trading/backups

# Set proper permissions
chmod 755 /opt/trading/data
chmod 755 /opt/trading/logs
chmod 755 /opt/trading/backups
```

---

## 🧪 **STEP 7: INITIAL TESTING**

### **7.1 Test Bot in Paper Trading Mode**
```bash
# Ensure you're in the trading directory
cd /opt/trading

# Activate virtual environment
source venv/bin/activate

# Set environment to use testnet first
export BYBIT_TESTNET=true

# Test run (should start in paper trading mode)
python -m src.main
```

**🔍 You should see:**
- Bot starting up
- Connection to Bybit testnet
- Dashboard available message
- No errors in startup

**Press `Ctrl+C` to stop the test run**

### **7.2 Test Dashboard Access**
**From your Windows machine:**
```powershell
# Test dashboard access (replace with your IP or domain)
# If using domain:
Start-Process "https://your-domain.com"

# If using IP only:
Start-Process "http://YOUR_DROPLET_IP"
```

**✅ You should see your trading dashboard load successfully**

---

## 🚨 **STEP 8: EMERGENCY PROCEDURES SETUP**

### **8.1 Create Emergency Stop Script**
```bash
# Create emergency stop script
nano /opt/trading/emergency_stop.sh
```

**Add this script:**
```bash
#!/bin/bash
echo "🚨 EMERGENCY STOP INITIATED - $(date)"

# Stop all trading processes
pkill -f "python -m src.main"
pkill -f "python main.py"

# Log emergency stop
echo "$(date): EMERGENCY STOP - All trading halted by user" >> /opt/trading/logs/emergency.log

# Send alert via API if bot is running
curl -X POST "http://localhost:8080/api/emergency-stop" 2>/dev/null || echo "Bot already stopped"

echo "🛑 EMERGENCY STOP COMPLETE"
echo "📧 Check your dashboard for confirmation"
```

```bash
# Make executable
chmod +x /opt/trading/emergency_stop.sh

# Test emergency stop
./emergency_stop.sh
```

### **8.2 Create System Service**
```bash
# Create systemd service for auto-restart
sudo nano /etc/systemd/system/trading-bot.service
```

**Add this service configuration:**
```ini
[Unit]
Description=Australian Tax Compliant Bybit Trading Bot
After=network.target

[Service]
Type=simple
User=tradingbot
Group=tradingbot
WorkingDirectory=/opt/trading
Environment=PATH=/opt/trading/venv/bin
EnvironmentFile=/opt/trading/.env.production
ExecStart=/opt/trading/venv/bin/python -m src.main
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable trading-bot.service

# Don't start it yet - we'll do final testing first
```

---

## 📊 **STEP 9: MONITORING SETUP**

### **9.1 Create System Monitor**
```bash
# Create monitoring script
nano /opt/trading/monitor_system.sh
```

**Add monitoring script:**
```bash
#!/bin/bash
LOG_FILE="/opt/trading/logs/system_health.log"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # System metrics
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.1f", ($3/$2) * 100.0)}')
    DISK_USAGE=$(df /opt/trading | tail -1 | awk '{print $5}' | sed 's/%//')
    BOT_RUNNING=$(pgrep -f "python.*src\.main" | wc -l)
    
    # Log status
    echo "$TIMESTAMP - CPU: ${CPU_USAGE}% | Memory: ${MEMORY_USAGE}% | Disk: ${DISK_USAGE}% | Bot: $BOT_RUNNING" >> $LOG_FILE
    
    # Alert on high usage
    if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
        echo "🚨 HIGH CPU: $CPU_USAGE%" | logger -t TradingBot
    fi
    
    if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
        echo "🚨 HIGH MEMORY: $MEMORY_USAGE%" | logger -t TradingBot
    fi
    
    if [ "$BOT_RUNNING" -eq 0 ]; then
        echo "🚨 BOT NOT RUNNING!" | logger -t TradingBot
    fi
    
    sleep 300  # Check every 5 minutes
done
```

```bash
# Make executable
chmod +x /opt/trading/monitor_system.sh

# Create monitoring service
sudo nano /etc/systemd/system/trading-monitor.service
```

**Add monitoring service:**
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
# Enable and start monitoring
sudo systemctl enable trading-monitor.service
sudo systemctl start trading-monitor.service
```

---

## 🎯 **STEP 10: FINAL VERIFICATION & GO LIVE**

### **10.1 Security Audit**
```bash
# Run final security check
nano /opt/trading/security_check.sh
```

**Add security audit script:**
```bash
#!/bin/bash
echo "🔍 FINAL SECURITY AUDIT"
echo "======================="

# Check SSH config
echo "✅ SSH Security:"
sudo grep "PermitRootLogin no" /etc/ssh/sshd_config >/dev/null && echo "   Root login disabled ✓" || echo "   ❌ Root login enabled!"
sudo grep "AllowUsers tradingbot" /etc/ssh/sshd_config >/dev/null && echo "   User access restricted ✓" || echo "   ❌ User access not restricted!"
systemctl is-active ssh >/dev/null && echo "   SSH service running" || echo "   SSH service disabled (console-only) ✓"

# Check firewall
echo "✅ Firewall:"
sudo ufw status | grep "Status: active" >/dev/null && echo "   UFW active ✓" || echo "   ❌ UFW inactive!"

# Check file permissions
echo "✅ File Security:"
PERM=$(stat -c "%a" /opt/trading/.env.production)
[[ "$PERM" == "600" ]] && echo "   Environment file secure ✓" || echo "   ❌ Environment file permissions wrong!"

# Check services
echo "✅ Services:"
systemctl is-active trading-monitor >/dev/null && echo "   Monitoring active ✓" || echo "   ❌ Monitoring inactive!"

# Check emergency script
echo "✅ Emergency Procedures:"
[[ -x /opt/trading/emergency_stop.sh ]] && echo "   Emergency stop ready ✓" || echo "   ❌ Emergency stop missing!"

echo "======================="
echo "🚨 REVIEW ALL ITEMS BEFORE LIVE TRADING!"
```

```bash
chmod +x /opt/trading/security_check.sh
./security_check.sh
```

### **10.2 Paper Trading Test (MANDATORY)**
```bash
# Start bot in paper trading mode for final test
cd /opt/trading
source venv/bin/activate

# Ensure paper trading mode
export BYBIT_TESTNET=true

# Start bot
python -m src.main
```

**✅ Verify these work:**
- Dashboard loads correctly
- All buttons respond
- Emergency stop works
- Australian tax logs are being created
- Portfolio data displays

**Run paper trading for AT LEAST 24 hours before considering live trading**

### **10.3 Go Live Checklist**

**Before enabling live trading, confirm:**
- [ ] ✅ Paper trading working perfectly for 24+ hours
- [ ] ✅ Emergency stop tested and working
- [ ] ✅ Dashboard fully functional
- [ ] ✅ Australian tax compliance logging properly
- [ ] ✅ All security checks passed
- [ ] ✅ Monitoring system active
- [ ] ✅ Conservative limits set (0.5% positions, $100 daily loss)
- [ ] ✅ Live API keys properly configured with restrictions

### **10.4 Enable Live Trading**
```bash
# Edit production environment to enable live trading
nano /opt/trading/.env.production

# Change this line:
BYBIT_TESTNET=false

# Start with these CONSERVATIVE settings:
MAX_DAILY_LOSS_AUD=100.00
MAX_POSITION_SIZE_PERCENT=0.5
RISK_MANAGEMENT_MODE="ultra_conservative"
```

### **10.5 Start Live Trading**
```bash
# Start the trading bot service
sudo systemctl start trading-bot.service

# Check status
sudo systemctl status trading-bot.service

# Monitor logs in real-time
tail -f /opt/trading/logs/app.log
```

---

## 🎉 **CONGRATULATIONS!**

**Your secure DigitalOcean trading environment is now live! 🚀**

### **🔗 Access Your Dashboard:**
- **With Domain**: https://your-domain.com
- **Without Domain**: http://YOUR_DROPLET_IP

### **🚨 CRITICAL REMINDERS:**
1. **Monitor your first 24-48 hours closely**
2. **Start with tiny position sizes (0.5%)**
3. **Keep daily loss limits very low initially ($100)**
4. **Test emergency stop procedure regularly**
5. **Check Australian tax logs daily**

### **📞 Emergency Access:**
**Option 1: DigitalOcean Console (Recommended)**
1. Login to DigitalOcean dashboard
2. Go to your droplet → Console
3. Login as `tradingbot`
4. Run: `/opt/trading/emergency_stop.sh`

**Option 2: SSH (if enabled)**
```powershell
# From your Windows machine - Emergency stop
ssh -p 2222 tradingbot@YOUR_DROPLET_IP "/opt/trading/emergency_stop.sh"
```

**🛡️ Your trading bot is now secure, compliant, and ready for conservative live trading! 🇦🇺**

---

## 📚 **NEXT STEPS:**
1. **Monitor performance for 1 week with micro positions**
2. **Gradually increase position sizes as confidence builds**
3. **Regular security updates: `sudo apt update && sudo apt upgrade`**
4. **Weekly backup verification**
5. **Monthly review of Australian tax compliance logs**

**Trade safely and responsibly! 💰**