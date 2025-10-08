#!/bin/bash
# Server setup script for DigitalOcean droplet
# Run this after SSH into your new droplet

set -e

echo "🔧 Setting up secure production server..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Update system
print_status "Updating system packages..."
apt update && apt upgrade -y

# Install essential packages
print_status "Installing essential packages..."
apt install -y \
    curl \
    wget \
    git \
    htop \
    iotop \
    nethogs \
    ufw \
    fail2ban \
    nginx \
    certbot \
    python3-certbot-nginx \
    docker.io \
    docker-compose \
    postgresql-client \
    redis-tools

# Enable and start Docker
systemctl enable docker
systemctl start docker

# Create trading user
print_status "Creating trading user..."
useradd -m -s /bin/bash tradingbot
usermod -aG sudo tradingbot
usermod -aG docker tradingbot

# Setup SSH for trading user
mkdir -p /home/tradingbot/.ssh
cp /root/.ssh/authorized_keys /home/tradingbot/.ssh/
chown -R tradingbot:tradingbot /home/tradingbot/.ssh
chmod 700 /home/tradingbot/.ssh
chmod 600 /home/tradingbot/.ssh/authorized_keys

# Configure SSH security
print_status "Hardening SSH configuration..."
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

cat > /etc/ssh/sshd_config << 'EOF'
# SSH Security Configuration
Port 2222
Protocol 2

# Authentication
PubkeyAuthentication yes
PasswordAuthentication no
PermitEmptyPasswords no
PermitRootLogin no
MaxAuthTries 3

# Security settings
ClientAliveInterval 300
ClientAliveCountMax 2
MaxSessions 2
AllowUsers tradingbot

# Disable unused features
X11Forwarding no
PermitTunnel no
AllowAgentForwarding no
AllowTcpForwarding no
EOF

# Configure firewall
print_status "Configuring UFW firewall..."
ufw default deny incoming
ufw default allow outgoing

# Allow essential ports
ufw allow 2222/tcp   # SSH
ufw allow 80/tcp     # HTTP
ufw allow 443/tcp    # HTTPS

# Internal network access for databases
ufw allow from 10.0.0.0/16 to any port 5432  # PostgreSQL
ufw allow from 10.0.0.0/16 to any port 6379  # Redis

ufw --force enable

# Configure Fail2Ban
print_status "Configuring Fail2Ban..."
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
backend = systemd

[sshd]
enabled = true
port = 2222
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
action = iptables-multiport[name=ReqLimit, port="http,https", protocol=tcp]
logpath = /var/log/nginx/error.log
maxretry = 10
EOF

systemctl enable fail2ban
systemctl start fail2ban

# Setup application directory
print_status "Creating application directories..."
mkdir -p /opt/trading/{data,logs,config,ssl}
chown -R tradingbot:tradingbot /opt/trading

# Create log rotation
cat > /etc/logrotate.d/trading-bot << 'EOF'
/opt/trading/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    su tradingbot tradingbot
}
EOF

# Setup Nginx basic configuration
print_status "Configuring Nginx..."
cat > /etc/nginx/sites-available/trading-bot << 'EOF'
server {
    listen 80;
    server_name _;

    # Temporary redirect for Let's Encrypt
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name _;

    # Placeholder SSL configuration
    ssl_certificate /etc/ssl/certs/ssl-cert-snakeoil.pem;
    ssl_certificate_key /etc/ssl/private/ssl-cert-snakeoil.key;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /api/emergency-stop {
        limit_req zone=api burst=1 nodelay;
        proxy_pass http://127.0.0.1:8000;
    }
}
EOF

ln -sf /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
nginx -t

# Setup monitoring
print_status "Setting up system monitoring..."
cat > /opt/trading/monitor_system.sh << 'EOF'
#!/bin/bash
# System monitoring script

LOG_FILE="/opt/trading/logs/system_monitor.log"

log_metric() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# CPU usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
log_metric "CPU_USAGE: ${CPU_USAGE}%"

# Memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf("%.1f", ($3/$2) * 100.0)}')
log_metric "MEMORY_USAGE: ${MEM_USAGE}%"

# Disk usage
DISK_USAGE=$(df -h / | awk 'NR==2{printf "%s", $5}')
log_metric "DISK_USAGE: ${DISK_USAGE}"

# Check if trading app is running
if pgrep -f "src.main" > /dev/null; then
    log_metric "TRADING_APP: RUNNING"
else
    log_metric "TRADING_APP: NOT_RUNNING"
fi

# Network connections
CONNECTIONS=$(netstat -an | wc -l)
log_metric "NETWORK_CONNECTIONS: ${CONNECTIONS}"

# Alert if CPU usage > 80%
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "HIGH CPU USAGE: ${CPU_USAGE}%" | mail -s "Server Alert" your-email@domain.com
fi
EOF

chmod +x /opt/trading/monitor_system.sh
chown tradingbot:tradingbot /opt/trading/monitor_system.sh

# Setup cron job for monitoring
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/trading/monitor_system.sh") | crontab -

# Install Python dependencies for trading bot
print_status "Installing Python dependencies..."
apt install -y python3-pip python3-venv
pip3 install --upgrade pip

# Create systemd service for trading bot
cat > /etc/systemd/system/trading-bot.service << 'EOF'
[Unit]
Description=Bybit Trading Bot
After=network.target docker.service
Requires=docker.service

[Service]
Type=forking
User=tradingbot
Group=tradingbot
WorkingDirectory=/opt/trading
ExecStart=/usr/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/bin/docker-compose -f docker-compose.prod.yml down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable trading-bot

# Restart SSH with new configuration
print_status "Restarting SSH service..."
systemctl restart sshd

# Create deployment user setup
print_status "Setting up deployment environment..."
su - tradingbot << 'EOF'
# Generate SSH key for GitHub
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# Setup Git configuration
git config --global user.name "Trading Bot Deploy"
git config --global user.email "deploy@tradingbot.com"

# Clone the repository (user needs to add the SSH key to GitHub)
echo "Add this SSH key to your GitHub account:"
cat ~/.ssh/id_rsa.pub
EOF

print_status "✅ Server setup completed!"
print_warning "⚠️  Important next steps:"
echo ""
echo "1. Add the SSH key displayed above to your GitHub account"
echo "2. Clone your repository: git clone git@github.com:Zvxndr/Bybit-bot.git /opt/trading"
echo "3. Configure your domain name in Nginx"
echo "4. Get SSL certificates: certbot --nginx -d your-domain.com"
echo "5. Set up environment variables in /opt/trading/.env.production"
echo "6. Test the deployment with: docker-compose -f docker-compose.prod.yml up"
echo ""
print_status "🔐 SSH is now on port 2222. Use: ssh -p 2222 tradingbot@your-server-ip"
print_status "🔥 Firewall is active. Only ports 80, 443, and 2222 are open."
print_status "🛡️  Fail2Ban is monitoring for intrusion attempts."

echo ""
print_warning "Don't forget to:"
echo "- Configure your .env.production file with real API keys"
echo "- Test emergency stop procedures"
echo "- Set up monitoring alerts"
echo "- Configure database backups"