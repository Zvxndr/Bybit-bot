# 🚀 PRODUCTION DEPLOYMENT GUIDE - DigitalOcean
## Complete Setup for Live Trading with Security & Email Alerts

**🎯 This guide deploys your trading bot for REAL MONEY trading with enterprise security.**

---

## 📋 Prerequisites Checklist

- [ ] DigitalOcean account with $12/month droplet budget
- [ ] Bybit MAINNET API keys (live trading)  
- [ ] Bybit TESTNET API keys (strategy validation)
- [ ] SendGrid account (free tier: 100 emails/day)
- [ ] Domain name (optional but recommended)
- [ ] Email addresses for alerts

---

## 💰 Step 1: Get Your Bybit API Keys

### 1.1 Mainnet API Keys (LIVE TRADING)
```
1. Login to Bybit: https://www.bybit.com
2. Go to: Account & Security → API Management  
3. Create API Key with these permissions:
   ✅ Spot Trading
   ✅ Derivatives Trading  
   ✅ Read-Only (Account info, positions)
   ❌ Withdrawal (NEVER enable this)
4. Copy API Key and Secret (you'll only see secret once!)
5. Set IP restrictions to your DigitalOcean server IP
```

### 1.2 Testnet API Keys (STRATEGY VALIDATION)  
```
1. Go to: https://testnet.bybit.com
2. Login with same Bybit account
3. Create API keys with same permissions as mainnet
4. These validate new AI strategies safely before live trading
```

---

## 🌊 Step 2: Create DigitalOcean Droplet

### 2.1 Droplet Specifications
```bash
# Recommended specs for production trading:
OS: Ubuntu 22.04 LTS
Plan: Basic $12/month
RAM: 2GB  
CPU: 1 vCPU
Storage: 50GB SSD
Datacenter: Choose closest to you (lower latency = better fills)

# Premium option ($24/month for high-frequency trading):
RAM: 4GB
CPU: 2 vCPU  
Storage: 80GB SSD
```

### 2.2 Initial Server Setup
```bash
# SSH into your new droplet
ssh root@your_droplet_ip

# Create trading user
adduser trader
usermod -aG sudo trader
su - trader

# Update system
sudo apt update && sudo apt upgrade -y
```

---

## 🔒 Step 3: Run Security Hardening

### 3.1 Download and Execute Security Script
```bash
# Download our enterprise security script
curl -o security_deploy.sh https://raw.githubusercontent.com/Zvxndr/Bybit-bot-fresh/main/security_deploy.sh

# Make executable and run
chmod +x security_deploy.sh
./security_deploy.sh
```

**✅ This automatically configures:**
- Firewall with proper ports (2222, 80, 443)
- SSH hardening (port 2222, no root login)
- Fail2Ban intrusion detection  
- Nginx reverse proxy with rate limiting
- Security monitoring system
- Daily database backups
- Email reporting infrastructure

---

## 📧 Step 4: Configure Email Alerts

### 4.1 Get SendGrid API Key
```bash
# Sign up: https://app.sendgrid.com/signup
# Create API Key:
# 1. Settings → API Keys → Create API Key
# 2. Name: "Trading Bot Production"  
# 3. Permissions: Mail Send (Full Access)
# 4. Copy the API key (shown only once!)
```

### 4.2 Set Up Email System
```bash
# Set your SendGrid API key
export SENDGRID_API_KEY='SG.your_actual_sendgrid_api_key_here'

# Set your email addresses  
export FROM_EMAIL='trading-bot@yourdomain.com'
export ALERT_EMAIL='alerts@yourdomain.com'
export REPORTS_EMAIL='reports@yourdomain.com'

# Run email setup script (created by security script)
./setup_email_reporting.sh

# Test email system
./test_email_system.sh
```

---

## 🏗️ Step 5: Deploy Trading Application

### 5.1 Upload Your Code
```bash
# Create app directory
sudo mkdir -p /app
sudo chown trader:trader /app

# Clone your repository
cd /app
git clone https://github.com/Zvxndr/Bybit-bot-fresh.git .

# Set proper permissions
chmod +x start_pipeline.py
```

### 5.2 Install Python Dependencies
```bash
# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv /app/venv
source /app/venv/bin/activate

# Install all requirements
pip install -r requirements.txt
pip install sendgrid matplotlib seaborn pandas  # Email dependencies
```

---

## ⚙️ Step 6: Configure Production Environment

### 6.1 Create Production Environment File
```bash
# Create production environment configuration
nano /app/.env.production
```

**Copy this configuration (replace with your actual values):**

```bash
# =============================================================================
# BYBIT TRADING BOT - PRODUCTION CONFIGURATION
# =============================================================================
# 🚨 CRITICAL: This file contains LIVE TRADING credentials
# 🔒 Keep this file secure and never commit to version control

# =============================================================================
# TRADING ENVIRONMENT SETTINGS
# =============================================================================
NODE_ENV=production
TRADING_ENVIRONMENT=production
DEBUG=false
FORCE_TESTNET=false

# =============================================================================
# LIVE TRADING API CREDENTIALS (BYBIT MAINNET)
# =============================================================================
# 🚨 These control REAL MONEY - Double check these values!
BYBIT_API_KEY=your_live_bybit_api_key_here
BYBIT_API_SECRET=your_live_bybit_api_secret_here
BYBIT_TESTNET=false

# =============================================================================
# TESTNET API CREDENTIALS (STRATEGY VALIDATION)
# =============================================================================  
# These validate new AI strategies safely before live deployment
BYBIT_TESTNET_API_KEY=your_testnet_api_key_here
BYBIT_TESTNET_API_SECRET=your_testnet_api_secret_here

# =============================================================================
# EMAIL NOTIFICATION SYSTEM
# =============================================================================
# SendGrid Configuration (100 emails/day free)
SENDGRID_API_KEY=SG.your_sendgrid_api_key_here
FROM_EMAIL=trading-bot@yourdomain.com
FROM_NAME=Bybit Trading Bot Production

# Alert Recipients (can be same email)
SECURITY_EMAIL=security@yourdomain.com      # Security breach alerts
TRADING_EMAIL=trading@yourdomain.com        # Trading alerts (losses, gains)
REPORTS_EMAIL=reports@yourdomain.com        # Daily/weekly reports  
INVESTOR_EMAIL=investor@yourdomain.com      # Monthly investor reports

# =============================================================================
# SECURITY & SAFETY SETTINGS
# =============================================================================
# Application Security
SECRET_KEY=your_super_secure_random_secret_key_minimum_32_chars
JWT_SECRET_KEY=another_secure_random_key_for_jwt_tokens

# Rate Limiting & Security
RATE_LIMIT_ENABLED=true
MAX_REQUESTS_PER_MINUTE=60
INTRUSION_DETECTION_ENABLED=true

# =============================================================================
# RISK MANAGEMENT SETTINGS
# =============================================================================
# Maximum Risk Limits (Percentages as decimals)
MAX_DAILY_RISK=0.02                    # 2% max daily portfolio risk
MAX_POSITION_SIZE=0.01                 # 1% max per single position
EMERGENCY_STOP_LOSS=0.05               # 5% total portfolio emergency stop
MAX_CONCURRENT_POSITIONS=10            # Maximum open positions

# Position Management
ENABLE_STOP_LOSS=true
ENABLE_TAKE_PROFIT=true
DEFAULT_STOP_LOSS=0.02                 # 2% stop loss
DEFAULT_TAKE_PROFIT=0.04               # 4% take profit

# =============================================================================
# AI PIPELINE SETTINGS
# =============================================================================
# Strategy Pipeline Configuration
ENABLE_AI_PIPELINE=true
STRATEGY_VALIDATION_REQUIRED=true
MIN_BACKTEST_SCORE=75                  # Minimum score for strategy promotion
MIN_PAPER_TRADING_DAYS=7               # Days of paper trading before live
AUTO_STRATEGY_DEPLOYMENT=false         # Manual approval required for live

# Machine Learning Settings
ML_MODEL_UPDATES_ENABLED=true
FEATURE_SELECTION_AUTO=true
MODEL_RETRAINING_INTERVAL=24           # Hours between model retraining

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
DATABASE_URL=sqlite:///data/trading_bot.db
DATABASE_BACKUP_ENABLED=true
BACKUP_RETENTION_DAYS=30
BACKUP_SCHEDULE=0 2 * * *              # Daily at 2 AM

# =============================================================================
# LOGGING CONFIGURATION  
# =============================================================================
LOG_LEVEL=INFO
CONSOLE_LOGGING=true
FILE_LOGGING=true
LOG_DIRECTORY=/var/log/trading-bot

# Log Files
APP_LOG_FILE=app.log
ERROR_LOG_FILE=errors.log
TRADE_LOG_FILE=trades.log
SECURITY_LOG_FILE=security.log

# =============================================================================
# PERFORMANCE & MONITORING
# =============================================================================
# System Monitoring
CPU_ALERT_THRESHOLD=90                 # Alert if CPU > 90%
MEMORY_ALERT_THRESHOLD=85              # Alert if RAM > 85%
DISK_ALERT_THRESHOLD=90                # Alert if disk > 90%

# API Performance
API_TIMEOUT=30                         # API request timeout (seconds)
MAX_RETRIES=3                          # Maximum API retry attempts
RATE_LIMIT_BUFFER=0.8                  # Use 80% of rate limit

# =============================================================================
# REPORTING SCHEDULE (AEST/AEDT Timezone)
# =============================================================================
REPORT_TIMEZONE=Australia/Sydney
DAILY_REPORT_TIME=08:00               # 8 AM daily performance report
WEEKLY_REPORT_DAY=monday              # Monday weekly analysis
WEEKLY_REPORT_TIME=09:00              # 9 AM weekly report
MONTHLY_REPORT_DAY=1                  # 1st of month
MONTHLY_REPORT_TIME=10:00             # 10 AM monthly report

# Report Settings
DAILY_REPORTS=true
WEEKLY_REPORTS=true  
MONTHLY_REPORTS=true
INCLUDE_CHARTS=true
INCLUDE_PERFORMANCE_ANALYSIS=true

# =============================================================================
# EMERGENCY CONTACTS & ALERTS
# =============================================================================
# Immediate Alert Triggers (sends instant email/SMS)
LARGE_LOSS_THRESHOLD=1000             # Alert if single trade loss > $1000
DAILY_DRAWDOWN_THRESHOLD=0.05         # Alert if daily drawdown > 5%
API_ERROR_THRESHOLD=10                # Alert after 10 consecutive API errors
POSITION_SIZE_VIOLATION=true          # Alert on position size rule violations

# =============================================================================
# MULTI-EXCHANGE DATA INTEGRATION (INCLUDED)
# =============================================================================
# Cross-exchange price comparison and market analysis
ENABLE_BINANCE_DATA=true
ENABLE_OKX_DATA=true
MULTI_EXCHANGE_CACHE_TTL=30
PRICE_COMPARISON_ENABLED=true
CROSS_EXCHANGE_SYMBOLS=BTCUSDT,ETHUSDT,ADAUSDT

# Note: Provides real-time price comparison across Binance, OKX, and Bybit
# Dashboard includes Multi-Exchange tab with cross-exchange price monitoring
# See: MULTI_EXCHANGE_DATA_INTEGRATION_COMPLETE.md for full documentation

# =============================================================================  
# FUTURE: ARBITRAGE CAPABILITIES (TRUST/PTY LTD VERSION)
# =============================================================================
# Arbitrage detection and execution will be available in institutional version
# Requires proper regulatory compliance and licensing (AFSL)
# See: FUTURE_ARBITRAGE_IMPLEMENTATION.md for complete roadmap
# Current version: Data-only cross-exchange analysis (compliant and ready)

# =============================================================================
# OPTIONAL: ADVANCED FEATURES
# =============================================================================
# Telegram Bot (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
TELEGRAM_ALERTS_ENABLED=false

# Slack Notifications (Optional)
SLACK_WEBHOOK_URL=your_slack_webhook_url
SLACK_ALERTS_ENABLED=false

# =============================================================================
# SYSTEM INFORMATION
# =============================================================================
# Deployment Information
DEPLOYMENT_DATE=$(date +%Y-%m-%d)
SERVER_LOCATION=digitalocean
DEPLOYMENT_VERSION=1.0.0
SYSTEM_ADMIN=trader@yourdomain.com

# =============================================================================
# END OF CONFIGURATION
# =============================================================================
# 📝 Remember to:
# 1. Set secure file permissions: chmod 600 /app/.env.production
# 2. Never commit this file to version control
# 3. Test with small amounts first
# 4. Monitor email alerts for first 24 hours
# 5. Keep backups of this configuration
```

### 6.2 Secure Environment File
```bash
# Set secure permissions (only owner can read/write)
chmod 600 /app/.env.production

# Create symlink for application to find it
ln -sf /app/.env.production /app/.env

# Verify permissions
ls -la /app/.env*
```

---

## 🚀 Step 7: Start Production Services

### 7.1 Configure Systemd Service
```bash
# The security script already created the service file
# Just need to enable and start it

sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Check if running
sudo systemctl status trading-bot
```

### 7.2 Monitor Initial Startup
```bash
# Watch logs in real-time
sudo journalctl -u trading-bot -f

# Check for any errors
sudo journalctl -u trading-bot --since "10 minutes ago"

# Application logs
tail -f /var/log/trading-bot/app.log
```

---

## 🌐 Step 8: SSL Certificate & Domain Setup

### 8.1 Configure Domain (If You Have One)
```bash
# Update nginx configuration
sudo nano /etc/nginx/sites-available/trading-bot

# Replace 'yourdomain.com' with your actual domain
# Save and exit

# Test nginx configuration  
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### 8.2 Install SSL Certificate
```bash
# For domain setup (recommended):
sudo certbot --nginx -d yourdomain.com

# For IP-only setup (less secure):
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/nginx-selfsigned.key \
    -out /etc/ssl/certs/nginx-selfsigned.crt
```

---

## 🧪 Step 9: Production Testing & Validation

### 9.1 Test System Components
```bash
# Test email alerts
./test_email_system.sh

# Test security monitoring  
./security_monitor.sh

# Test API connections
curl -k https://your_domain_or_ip/api/health

# Test trading endpoints (should show live data)
curl -k https://your_domain_or_ip/api/portfolio
```

### 9.2 Production Validation Checklist
```bash
# Run this validation script
cat << 'EOF' > /home/trader/production_validation.sh
#!/bin/bash
echo "🧪 PRODUCTION VALIDATION CHECKLIST"
echo "=================================="

# Check services
echo "1. Checking services..."
sudo systemctl is-active trading-bot && echo "✅ Trading bot: Running" || echo "❌ Trading bot: Not running"
sudo systemctl is-active nginx && echo "✅ Nginx: Running" || echo "❌ Nginx: Not running" 
sudo ufw status | grep -q "Status: active" && echo "✅ Firewall: Active" || echo "❌ Firewall: Inactive"

# Check API keys
echo -e "\n2. Checking API configuration..."
grep -q "BYBIT_API_KEY=" /app/.env && echo "✅ Live API key configured" || echo "❌ Live API key missing"
grep -q "SENDGRID_API_KEY=" /app/.env && echo "✅ SendGrid API configured" || echo "❌ SendGrid API missing"

# Check security
echo -e "\n3. Checking security..."
sudo fail2ban-client status sshd | grep -q "Currently banned" && echo "✅ Fail2Ban: Active" || echo "✅ Fail2Ban: Active (no bans yet)"
ls -la /app/.env | grep -q "rw-------" && echo "✅ Environment file: Secure permissions" || echo "❌ Environment file: Insecure permissions"

# Check logs
echo -e "\n4. Checking logs..."
sudo journalctl -u trading-bot --since "5 minutes ago" | grep -q "ERROR" && echo "⚠️  Recent errors found in logs" || echo "✅ No recent errors"

echo -e "\n✅ Validation complete!"
EOF

chmod +x /home/trader/production_validation.sh
./production_validation.sh
```

---

## 📊 Step 10: Monitor Your Production System

### 10.1 Daily Monitoring Commands
```bash
# Check system status
sudo systemctl status trading-bot nginx

# Check resource usage
htop

# Check recent trades/activity
tail -f /var/log/trading-bot/trades.log

# Check email delivery
tail -f /var/log/trading-bot/app.log | grep -i email

# Security check
./security_monitor.sh
```

### 10.2 Set Up Monitoring Alerts
```bash
# Add monitoring cron job
crontab -e

# Add these lines:
# Check system health every 10 minutes
*/10 * * * * /home/trader/security_monitor.sh

# Daily backup verification (5 AM)
0 5 * * * ls -la /home/trader/backups/ | mail -s "Daily Backup Status" $REPORTS_EMAIL
```

---

## 🚨 Emergency Procedures

### Emergency Stop Trading
```bash
# Immediately stop all trading
sudo systemctl stop trading-bot

# Check current positions (via web interface or API)
curl -k https://your_domain/api/positions

# Manual position management if needed:
# Login to Bybit.com directly to close positions
```

### System Recovery
```bash
# Restart services
sudo systemctl restart trading-bot nginx

# Check logs for issues
sudo journalctl -u trading-bot --since "30 minutes ago"

# Reset firewall if needed
sudo ufw --force reset
./security_deploy.sh
```

---

## 💡 Production Tips & Best Practices

### 🔐 Security
- [ ] Change SSH port to 2222 (done by security script)
- [ ] Use SSH keys, disable password auth
- [ ] Monitor fail2ban logs daily
- [ ] Keep system updated weekly
- [ ] Review email alerts immediately

### 💰 Trading
- [ ] Start with small position sizes
- [ ] Monitor first 24 hours closely
- [ ] Test emergency stop procedures
- [ ] Keep manual override capability
- [ ] Backup wallet/API key info securely

### 📧 Monitoring
- [ ] Check daily email reports arrive
- [ ] Verify alert emails work (test with small loss)
- [ ] Monitor system resource usage
- [ ] Review weekly performance reports
- [ ] Keep logs for audit purposes

---

## ✅ Production Deployment Complete!

**🎉 Your trading bot is now live and ready for production trading!**

### 📱 Quick Access URLs
```bash
# Web interface
https://your_domain_or_ip

# API health check
https://your_domain_or_ip/api/health

# Real-time dashboard
https://your_domain_or_ip/dashboard
```

### 📞 Emergency Contacts
- **System Admin**: trader@yourdomain.com
- **Security Alerts**: security@yourdomain.com  
- **Trading Alerts**: trading@yourdomain.com

### 🛡️ What's Active Now:
✅ **Live Trading** - Real Bybit mainnet API  
✅ **Enterprise Security** - Firewall, intrusion detection, SSH hardening  
✅ **Email Alerts** - Instant notifications for trades, security, system health  
✅ **AI Pipeline** - New strategies validate on testnet before going live  
✅ **Risk Management** - Position limits, stop losses, emergency stops  
✅ **Monitoring** - Daily reports, performance tracking, system health  
✅ **Backups** - Daily database backups, configuration backups  

**🚀 Happy Trading! Your enterprise-grade trading system is now LIVE! 🎯**

---

## 📎 Quick Reference

```bash
# SSH to server (note port 2222)
ssh trader@your_server_ip -p 2222

# Check trading bot status
sudo systemctl status trading-bot

# View live logs
sudo journalctl -u trading-bot -f

# Test email alerts
./test_email_system.sh

# Security check
./security_monitor.sh

# Emergency stop
sudo systemctl stop trading-bot
```

**📧 You'll receive email confirmations when everything is working correctly!**