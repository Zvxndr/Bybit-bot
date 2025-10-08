# 🚀 DigitalOcean Deployment Guide with Email Reporting

## Complete Security Hardening + Email Alerts Setup

This guide will deploy your trading bot to DigitalOcean with enterprise-grade security **and** comprehensive email reporting system.

---

## 📋 Prerequisites

**Before you start:**

1. **DigitalOcean Account** - [Sign up here](https://digitalocean.com)
2. **SendGrid Account** - [Get free API key](https://sendgrid.com) (100 emails/day free)
3. **Domain Name** (optional but recommended)
4. **Email Addresses** for alerts and reports

---

## 🚀 Step 1: Create & Connect to Droplet

### 1.1 Create DigitalOcean Droplet

```bash
# Choose these specifications:
# • Ubuntu 22.04 LTS
# • Basic Plan: $12/month (2GB RAM, 1 CPU, 50GB SSD)
# • Datacenter: Choose closest to you
# • Authentication: SSH keys (recommended) or Password
```

### 1.2 Connect to Your Server

```bash
# Connect via SSH (replace with your server IP)
ssh root@your_server_ip

# Create non-root user (replace 'trader' with your username)
adduser trader
usermod -aG sudo trader

# Switch to new user
su - trader
```

---

## 🔒 Step 2: Run Security Hardening Script

### 2.1 Download and Execute Security Script

```bash
# Download the security deployment script
curl -o security_deploy.sh https://raw.githubusercontent.com/Zvxndr/Bybit-bot-fresh/main/security_deploy.sh

# Make executable and run
chmod +x security_deploy.sh
./security_deploy.sh
```

**This script automatically configures:**
- ✅ Firewall (UFW) with proper ports
- ✅ SSH hardening (port 2222, no root login)  
- ✅ Fail2Ban intrusion detection
- ✅ Nginx reverse proxy with rate limiting
- ✅ Security monitoring system
- ✅ Database backup automation
- ✅ Email reporting infrastructure

---

## 📧 Step 3: Configure Email Reporting

### 3.1 Get SendGrid API Key

1. **Sign up for SendGrid**: [https://app.sendgrid.com/signup](https://app.sendgrid.com/signup)
2. **Verify your email address**
3. **Create API Key**: 
   - Go to Settings → API Keys → Create API Key
   - Choose "Restricted Access"
   - Give it a name like "Trading Bot Alerts"
   - Grant permissions: Mail Send (Full Access)
   - **Copy the API key** (you'll only see it once!)

### 3.2 Set Up Email Environment

```bash
# Run the email setup script (created by security script)
export SENDGRID_API_KEY='your_actual_api_key_here'
export FROM_EMAIL='trading-bot@yourdomain.com'
export ALERT_EMAIL='your-alert-email@gmail.com'
export REPORTS_EMAIL='your-reports-email@gmail.com'

# Run email setup
./setup_email_reporting.sh
```

### 3.3 Test Email System

```bash
# Test if email system works
./test_email_system.sh

# You should receive a test email within 1-2 minutes
# Check spam folder if not in inbox
```

---

## 🏗️ Step 4: Deploy Your Trading Application

### 4.1 Upload Your Code

```bash
# Create app directory
sudo mkdir -p /app
sudo chown trader:trader /app

# Clone or upload your code
cd /app
git clone https://github.com/Zvxndr/Bybit-bot-fresh.git .

# Alternative: Upload via SCP from your local machine
# scp -P 2222 -r /path/to/your/code trader@your_server_ip:/app
```

### 4.2 Install Python Dependencies

```bash
# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv /app/venv
source /app/venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install email dependencies
pip install sendgrid matplotlib seaborn pandas
```

### 4.3 Configure Environment Variables

```bash
# Create production environment file
cp /home/trader/email-config/.env.template /app/.env

# Edit the file with your actual values
nano /app/.env
```

**Add these values to `/app/.env`:**

```bash
# Trading Configuration
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret
BYBIT_TESTNET=false

# Email Configuration
SENDGRID_API_KEY=your_sendgrid_api_key_here
FROM_EMAIL=trading-bot@yourdomain.com
FROM_NAME=Bybit Trading Bot

# Alert Recipients
SECURITY_EMAIL=security-alerts@yourdomain.com
TRADING_EMAIL=trading-alerts@yourdomain.com  
REPORTS_EMAIL=daily-reports@yourdomain.com
INVESTOR_EMAIL=investor-reports@yourdomain.com

# Report Settings
DAILY_REPORTS=true
WEEKLY_REPORTS=true
MONTHLY_REPORTS=true
REPORT_TIMEZONE=Australia/Sydney

# Database
DATABASE_URL=sqlite:///data/trading_bot.db

# Security
SECRET_KEY=your_secure_random_secret_key_here
```

### 4.4 Start the Application

```bash
# Start the trading bot service
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Check if it's running
sudo systemctl status trading-bot

# View logs
sudo journalctl -u trading-bot -f
```

---

## 🌐 Step 5: SSL Certificate Setup

### 5.1 Configure Domain (Optional)

If you have a domain name:

```bash
# Update nginx configuration with your domain
sudo nano /etc/nginx/sites-available/trading-bot

# Replace 'yourdomain.com' with your actual domain
# Save and exit

# Test nginx configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### 5.2 Install SSL Certificate

```bash
# For domain setup:
sudo certbot --nginx -d yourdomain.com

# For IP-only setup (less secure but functional):
# Use self-signed certificate for testing
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/nginx-selfsigned.key \
    -out /etc/ssl/certs/nginx-selfsigned.crt
```

---

## 🧪 Step 6: Test Everything

### 6.1 Test Security Features

```bash
# Run security check
./security_monitor.sh

# Check firewall status
sudo ufw status

# Check fail2ban
sudo fail2ban-client status

# Check nginx
sudo nginx -t && echo "✅ Nginx OK"
```

### 6.2 Test Email Alerts

```bash
# Test email system again
./test_email_system.sh

# Manual test: trigger a trading alert (if safe to do)
# Or check application logs for email notifications
tail -f /var/log/trading-bot/app.log | grep -i email
```

### 6.3 Test Web Interface

```bash
# Test API endpoints
curl -k https://your_server_ip/health
curl -k https://your_server_ip/api/status

# Or visit in browser:
# https://your_server_ip (use your actual IP)
```

---

## 📊 Step 7: Configure Email Reports

### 7.1 Scheduled Reports

The system automatically sends:

- **📈 Daily Performance Summary** - Every day at 8 AM AEST
  - Portfolio performance
  - Active trades
  - Risk metrics
  - System health

- **📊 Weekly Detailed Analysis** - Every Monday at 9 AM AEST
  - Full performance breakdown
  - Strategy analysis
  - Risk assessment
  - Charts and graphs

- **💼 Monthly Investor Report** - 1st of each month at 10 AM AEST
  - Executive summary
  - Investment performance
  - Portfolio allocation
  - Compliance statements

### 7.2 Instant Alerts

Automatic notifications for:

- 🚨 **Security Events** (intrusion attempts, failed logins)
- 📉 **Trading Alerts** (large losses, strategy failures)
- ⚠️ **System Issues** (high CPU, memory, disk space)
- 🔌 **API Errors** (exchange connection issues)

### 7.3 Customize Email Settings

```bash
# Edit email configuration
nano /home/trader/email-config/email_settings.yaml

# Restart service to apply changes
sudo systemctl restart trading-bot
```

---

## 🔧 Troubleshooting

### Email Issues

```bash
# Check email logs
sudo journalctl -u trading-bot -f | grep -i email

# Test SendGrid connection
python3 -c "
import os
from src.notifications.sendgrid_manager import SendGridEmailManager
manager = SendGridEmailManager(os.getenv('SENDGRID_API_KEY'))
print(manager.test_email_connection())
"

# Common fixes:
# 1. Verify SENDGRID_API_KEY is correct
# 2. Check SendGrid account status
# 3. Verify FROM_EMAIL domain is authenticated in SendGrid
# 4. Check spam/junk folders
```

### Security Issues

```bash
# Check security logs
sudo tail -f /var/log/fail2ban.log
sudo tail -f /var/log/auth.log

# Reset firewall if needed
sudo ufw --force reset
./security_deploy.sh  # Re-run security setup
```

### Application Issues

```bash
# Check application logs
sudo journalctl -u trading-bot -f

# Check nginx logs
sudo tail -f /var/log/nginx/error.log

# Restart services
sudo systemctl restart trading-bot
sudo systemctl restart nginx
```

---

## 📱 Monitoring & Maintenance

### Daily Checks

1. **Check email reports** - Ensure daily reports arrive
2. **Monitor server health** - Run `./security_monitor.sh`
3. **Review trading logs** - Check for any errors
4. **Verify backups** - Database backups run daily at 2 AM

### Weekly Tasks

1. **Security updates**: `sudo apt update && sudo apt upgrade`
2. **Log rotation**: Check `/var/log/trading-bot/` disk usage
3. **Performance review**: Analyze weekly email reports
4. **Backup verification**: Test database backup restoration

### Monthly Tasks

1. **SSL certificate renewal**: Certbot auto-renews, but verify
2. **Security audit**: Review fail2ban logs and blocked IPs
3. **Email deliverability**: Check SendGrid statistics
4. **Performance optimization**: Review resource usage

---

## 🎉 Success Checklist

**Your deployment is complete when you can check all these:**

- [ ] ✅ SSH works on port 2222 (not 22)
- [ ] ✅ Web interface accessible via HTTPS
- [ ] ✅ Firewall active (`sudo ufw status`)
- [ ] ✅ Fail2Ban protecting SSH (`sudo fail2ban-client status sshd`)
- [ ] ✅ Trading bot service running (`sudo systemctl status trading-bot`)
- [ ] ✅ Daily email reports received
- [ ] ✅ Test security alert email received
- [ ] ✅ Database backups running (`ls -la /home/trader/backups/`)
- [ ] ✅ SSL certificate installed (green padlock in browser)

---

## 🆘 Support

**If you need help:**

1. **Check logs first**: Most issues are visible in logs
2. **Review this guide**: Many solutions are documented above  
3. **Test components individually**: Use the test scripts provided
4. **Start fresh if needed**: DigitalOcean snapshots make this easy

**🛡️ Your trading bot is now enterprise-ready with comprehensive security and email reporting!**

---

## 📎 Quick Reference Commands

```bash
# SSH to server
ssh trader@your_server_ip -p 2222

# Check services
sudo systemctl status trading-bot
sudo systemctl status nginx
sudo ufw status

# View logs
sudo journalctl -u trading-bot -f
tail -f /var/log/trading-bot/app.log

# Test email
./test_email_system.sh

# Security check
./security_monitor.sh

# Restart services
sudo systemctl restart trading-bot
sudo systemctl restart nginx
```

**🚀 Happy Trading! Your bot is now secure and will keep you informed via email! 📧**