# DigitalOcean Security Deployment Guide
*Complete security implementation for Bybit Trading Bot deployment*

## 🎯 **Pre-Deployment Checklist**
- [ ] DigitalOcean Droplet created (Ubuntu 22.04 LTS recommended)
- [ ] Domain name configured and DNS pointed to droplet IP
- [ ] SSH keys generated and ready
- [ ] Backup strategy planned

---

## 🔐 **1. Initial Server Hardening**

### SSH Security Setup
```bash
# Update system packages first
sudo apt update && sudo apt upgrade -y

# Create dedicated user for trading bot
sudo adduser tradingbot
sudo usermod -aG sudo tradingbot

# Switch to new user
sudo su - tradingbot

# Generate SSH key pair (run on LOCAL machine)
ssh-keygen -t ed25519 -C "tradingbot@yourdomain.com" -f ~/.ssh/tradingbot_key

# Copy public key to server (run on LOCAL machine)
ssh-copy-id -i ~/.ssh/tradingbot_key.pub tradingbot@YOUR_DROPLET_IP

# Configure SSH client (add to ~/.ssh/config on LOCAL machine)
Host tradingbot-prod
    HostName YOUR_DROPLET_IP
    User tradingbot
    IdentityFile ~/.ssh/tradingbot_key
    Port 22
    ServerAliveInterval 60
```

### Secure SSH Configuration
```bash
# Edit SSH config on server
sudo nano /etc/ssh/sshd_config

# Apply these settings:
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
AllowUsers tradingbot
Protocol 2
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2

# Restart SSH service
sudo systemctl restart ssh
```

---

## 🛡️ **2. Firewall Configuration**

### UFW Firewall Setup
```bash
# Reset and configure UFW
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow essential services
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 80/tcp comment 'HTTP for Let\'s Encrypt'
sudo ufw allow 443/tcp comment 'HTTPS only'

# Deny common attack ports
sudo ufw deny 3389/tcp comment 'Block RDP'
sudo ufw deny 5900/tcp comment 'Block VNC'
sudo ufw deny 1433/tcp comment 'Block MSSQL'
sudo ufw deny 3306/tcp comment 'Block MySQL'

# Enable firewall
sudo ufw --force enable

# Verify configuration
sudo ufw status verbose
```

### DigitalOcean Cloud Firewall (Additional Layer)
1. Go to DigitalOcean Dashboard → Networking → Firewalls
2. Create new firewall: `tradingbot-firewall`
3. **Inbound Rules:**
   - SSH (22) - Your IP only
   - HTTP (80) - All IPv4/IPv6 (for SSL cert)
   - HTTPS (443) - All IPv4/IPv6
4. **Outbound Rules:**
   - All traffic allowed (for API calls)
5. Apply to your droplet

---

## 🔒 **3. SSL/TLS with Nginx**

### Install Nginx and Certbot
```bash
sudo apt install nginx certbot python3-certbot-nginx -y
```

### Nginx Security Configuration
Create `/etc/nginx/sites-available/tradingbot`:
```nginx
# Rate limiting zones
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=dashboard:10m rate=5r/s;

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

# Main HTTPS server
server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;
    
    # SSL Configuration (Let's Encrypt will populate this)
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' fonts.googleapis.com; font-src fonts.gstatic.com; connect-src 'self' wss: ws:;" always;
    
    # Hide Nginx version
    server_tokens off;
    
    # Main application proxy
    location / {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Dashboard with IP restriction
    location /dashboard {
        # Replace with your actual IPs
        allow YOUR_HOME_IP;
        allow YOUR_OFFICE_IP;
        allow YOUR_VPN_IP;
        deny all;
        
        limit_req zone=dashboard burst=10 nodelay;
        proxy_pass http://127.0.0.1:5000/dashboard;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # API endpoints with stricter limits
    location /api/ {
        limit_req zone=api burst=10 nodelay;
        proxy_pass http://127.0.0.1:5000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Block access to sensitive files
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    location ~* \.(log|sql|conf)$ {
        deny all;
    }
}
```

### Enable Site and Get SSL Certificate
```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/tradingbot /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Get SSL certificate (replace with your domain)
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Enable auto-renewal
sudo systemctl enable nginx
sudo systemctl enable certbot.timer
```

---

## 🔐 **4. Application Security**

### Environment Variables
Create `/home/tradingbot/.env`:
```bash
# API Security (use encrypted values)
BYBIT_API_KEY_ENCRYPTED="your_encrypted_testnet_key"
BYBIT_API_SECRET_ENCRYPTED="your_encrypted_testnet_secret"
ENCRYPTION_KEY="your_32_byte_fernet_key"

# Flask Security
FLASK_SECRET_KEY="$(openssl rand -hex 32)"
FLASK_ENV="production"

# Database
DATABASE_URL="postgresql://tradingbot:secure_password@localhost/tradingbot_db"

# Security Settings
ALLOWED_HOSTS="yourdomain.com,www.yourdomain.com"
CORS_ORIGINS="https://yourdomain.com"
SESSION_COOKIE_SECURE="true"
SESSION_COOKIE_HTTPONLY="true"
SESSION_COOKIE_SAMESITE="Strict"

# Rate Limiting
RATELIMIT_STORAGE_URL="redis://localhost:6379/1"
```

### API Key Encryption Script
Create `/home/tradingbot/encrypt_keys.py`:
```python
from cryptography.fernet import Fernet
import os
import getpass

def generate_encryption_key():
    """Generate and save encryption key"""
    key = Fernet.generate_key()
    with open('/home/tradingbot/.encryption_key', 'wb') as f:
        f.write(key)
    os.chmod('/home/tradingbot/.encryption_key', 0o600)
    return key

def load_encryption_key():
    """Load existing encryption key"""
    with open('/home/tradingbot/.encryption_key', 'rb') as f:
        return f.read()

def encrypt_api_credentials():
    """Encrypt API keys for secure storage"""
    if not os.path.exists('/home/tradingbot/.encryption_key'):
        key = generate_encryption_key()
    else:
        key = load_encryption_key()
    
    f = Fernet(key)
    
    # Get API credentials securely
    api_key = getpass.getpass("Enter Bybit API Key: ")
    api_secret = getpass.getpass("Enter Bybit API Secret: ")
    
    # Encrypt credentials
    encrypted_key = f.encrypt(api_key.encode()).decode()
    encrypted_secret = f.encrypt(api_secret.encode()).decode()
    
    print(f"Add these to your .env file:")
    print(f"BYBIT_API_KEY_ENCRYPTED={encrypted_key}")
    print(f"BYBIT_API_SECRET_ENCRYPTED={encrypted_secret}")

if __name__ == "__main__":
    encrypt_api_credentials()
```

---

## 🗄️ **5. Database Security**

### PostgreSQL Installation and Hardening
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Create database and user
sudo -u postgres createuser tradingbot --pwprompt
sudo -u postgres createdb tradingbot_db -O tradingbot

# Secure PostgreSQL configuration
sudo nano /etc/postgresql/*/main/postgresql.conf
```

PostgreSQL Security Settings:
```conf
# Connection settings
listen_addresses = 'localhost'
port = 5432
max_connections = 100

# Security settings
ssl = on
password_encryption = scram-sha-256
log_statement = 'ddl'
log_min_duration_statement = 1000
```

Update `/etc/postgresql/*/main/pg_hba.conf`:
```conf
# Local connections only
local   all             postgres                                peer
local   all             tradingbot                              scram-sha-256
host    all             tradingbot      127.0.0.1/32            scram-sha-256
```

---

## 🔍 **6. Monitoring and Intrusion Detection**

### Install Fail2Ban
```bash
sudo apt install fail2ban -y

# Configure Fail2Ban
sudo nano /etc/fail2ban/jail.local
```

Fail2Ban Configuration:
```ini
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
backend = systemd

[sshd]
enabled = true
port = ssh
logpath = %(sshd_log)s
maxretry = 3
bantime = 86400

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 5

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
logpath = /var/log/nginx/error.log
maxretry = 10
bantime = 600

[nginx-botsearch]
enabled = true
filter = nginx-botsearch
logpath = /var/log/nginx/access.log
maxretry = 2
```

### System Monitoring
```bash
# Install monitoring tools
sudo apt install htop iotop nethogs logwatch -y

# Setup log monitoring
sudo nano /etc/logwatch/conf/logwatch.conf
```

---

## 💾 **7. Backup Strategy**

### Automated Backup Script
Create `/home/tradingbot/backup.sh`:
```bash
#!/bin/bash
# Automated backup script for trading bot

BACKUP_DIR="/home/tradingbot/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR/{database,application,logs}

# Database backup
pg_dump tradingbot_db > $BACKUP_DIR/database/db_$DATE.sql
gzip $BACKUP_DIR/database/db_$DATE.sql

# Application backup
tar -czf $BACKUP_DIR/application/app_$DATE.tar.gz \
    /home/tradingbot/Bybit-bot-fresh \
    /home/tradingbot/.env \
    --exclude=/home/tradingbot/Bybit-bot-fresh/logs/* \
    --exclude=/home/tradingbot/Bybit-bot-fresh/__pycache__/*

# Log backup
tar -czf $BACKUP_DIR/logs/logs_$DATE.tar.gz /var/log/nginx/ /home/tradingbot/Bybit-bot-fresh/logs/

# Cleanup old backups
find $BACKUP_DIR -name "*.gz" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "*.sql" -mtime +$RETENTION_DAYS -delete

# Log backup completion
echo "$(date): Backup completed successfully" >> /home/tradingbot/backup.log
```

Setup automated backups:
```bash
chmod +x /home/tradingbot/backup.sh

# Add to crontab (daily at 2 AM)
crontab -e
# Add: 0 2 * * * /home/tradingbot/backup.sh
```

---

## 🚨 **8. Emergency Procedures**

### Emergency Stop Script
Create `/home/tradingbot/emergency_stop.sh`:
```bash
#!/bin/bash
# Emergency stop for trading bot

echo "INITIATING EMERGENCY STOP..."

# Stop all trading processes
sudo systemctl stop tradingbot
pkill -f "python.*trading"

# Block web access temporarily
sudo ufw delete allow 443/tcp
sudo systemctl stop nginx

# Kill any remaining Python processes
pkill -f python

# Log emergency stop
echo "$(date): EMERGENCY STOP ACTIVATED" >> /home/tradingbot/emergency.log

# Send notification (configure with your email)
echo "EMERGENCY STOP ACTIVATED on $(hostname) at $(date)" | mail -s "Trading Bot Emergency Stop" your-email@domain.com

echo "EMERGENCY STOP COMPLETED"
```

### Quick Recovery Script
Create `/home/tradingbot/recover.sh`:
```bash
#!/bin/bash
# Quick recovery script

echo "Starting recovery process..."

# Restore firewall rules
sudo ufw allow 443/tcp

# Start services
sudo systemctl start nginx
sudo systemctl start tradingbot

# Verify services
sudo systemctl status nginx
sudo systemctl status tradingbot

echo "Recovery completed. Check service status above."
```

---

## ✅ **9. Pre-Launch Security Checklist**

### Server Security
- [ ] SSH key authentication enabled, passwords disabled
- [ ] Root login disabled
- [ ] Firewall configured (UFW + DigitalOcean Cloud Firewall)
- [ ] Fail2Ban configured and running
- [ ] System packages updated
- [ ] Unnecessary services disabled

### Application Security  
- [ ] SSL certificate installed and auto-renewal configured
- [ ] API keys encrypted and stored securely
- [ ] Environment variables properly configured
- [ ] Database access restricted to localhost
- [ ] Security headers implemented in Nginx
- [ ] Rate limiting configured

### Monitoring & Backup
- [ ] Automated backups scheduled
- [ ] Log monitoring configured
- [ ] Emergency procedures documented and tested
- [ ] Monitoring tools installed
- [ ] Notification system configured

### Network Security
- [ ] Dashboard IP-whitelisted
- [ ] API endpoints rate-limited  
- [ ] Unnecessary ports blocked
- [ ] DDoS protection via DigitalOcean
- [ ] WebSocket connections secured

---

## 🔧 **10. Maintenance Schedule**

### Daily
- Monitor system logs for anomalies
- Check backup completion
- Verify trading bot status

### Weekly  
- Review Fail2Ban logs
- Check SSL certificate status
- Update system packages
- Monitor disk usage

### Monthly
- Rotate API keys
- Review access logs
- Test emergency procedures
- Update security patches
- Review firewall rules

---

## 📞 **11. Emergency Contacts & Procedures**

### When Things Go Wrong
1. **Immediate Response**: Run emergency stop script
2. **Assessment**: Check logs for root cause
3. **Communication**: Notify stakeholders
4. **Recovery**: Use recovery script after issue resolved
5. **Post-mortem**: Document incident and improve procedures

### Important Commands
```bash
# Check service status
sudo systemctl status tradingbot nginx postgresql

# View logs
sudo journalctl -u tradingbot -f
sudo tail -f /var/log/nginx/error.log

# Monitor resources
htop
df -h

# Check security
sudo fail2ban-client status
sudo ufw status
```

---

*This documentation should be reviewed and updated regularly as security requirements evolve.*