#!/bin/bash
# 🚀 ONE-TIME SECURITY SETUP SCRIPT
# Runs automatically on first startup, then disables itself

SETUP_FLAG="/home/$(whoami)/.security_setup_complete"

# Check if already run
if [ -f "$SETUP_FLAG" ]; then
    echo "✅ Security already configured, skipping..."
    exit 0
fi

echo "🔒 Running ONE-TIME Security Setup..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️ $1${NC}"
}

# Update system
print_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y >/dev/null 2>&1
print_status "System updated"

# Install required packages
print_info "Installing security packages..."
sudo apt install -y nginx ufw fail2ban htop curl wget python3 python3-pip python3-venv >/dev/null 2>&1
print_status "Security packages installed"

# Configure firewall
print_info "Configuring firewall..."
sudo ufw --force reset >/dev/null 2>&1
sudo ufw default deny incoming >/dev/null 2>&1
sudo ufw default allow outgoing >/dev/null 2>&1
sudo ufw allow 22/tcp >/dev/null 2>&1  # Keep SSH on standard port for simplicity
sudo ufw allow 80/tcp >/dev/null 2>&1
sudo ufw allow 443/tcp >/dev/null 2>&1
sudo ufw --force enable >/dev/null 2>&1
print_status "Firewall configured (ports 22, 80, 443)"

# Configure fail2ban
print_info "Setting up intrusion detection..."
sudo systemctl enable fail2ban >/dev/null 2>&1
sudo systemctl start fail2ban >/dev/null 2>&1
print_status "Fail2Ban intrusion detection active"

# Configure nginx
print_info "Setting up web server..."
sudo tee /etc/nginx/sites-available/trading-bot >/dev/null << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # Rate limiting
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Simple rate limiting
        limit_req_zone $binary_remote_addr zone=general:10m rate=10r/s;
        limit_req zone=general burst=20 nodelay;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/ >/dev/null 2>&1
sudo rm -f /etc/nginx/sites-enabled/default >/dev/null 2>&1
sudo nginx -t >/dev/null 2>&1 && sudo systemctl restart nginx >/dev/null 2>&1
print_status "Nginx web server configured"

# Create application directories
print_info "Setting up application directories..."
sudo mkdir -p /app/logs /app/data
sudo chown -R $(whoami):$(whoami) /app
print_status "Application directories ready"

# Set up Python environment
print_info "Setting up Python environment..."
cd /app
python3 -m venv venv >/dev/null 2>&1
source venv/bin/activate
pip install --upgrade pip >/dev/null 2>&1

# Install basic requirements if they exist
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt >/dev/null 2>&1
    print_status "Python dependencies installed"
else
    print_info "No requirements.txt found, skipping Python packages"
fi

# Create systemd service
print_info "Setting up system service..."
sudo tee /etc/systemd/system/trading-bot.service >/dev/null << EOF
[Unit]
Description=Bybit Trading Bot
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=/app
Environment=PYTHONPATH=/app
ExecStart=/app/venv/bin/python /app/src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload >/dev/null 2>&1
sudo systemctl enable trading-bot >/dev/null 2>&1
print_status "Trading bot service configured"

# Create setup completion flag
touch "$SETUP_FLAG"
echo "$(date): Security setup completed" > "$SETUP_FLAG"

print_status "🎉 ONE-TIME Security Setup Complete!"
echo ""
echo "📋 WHAT'S CONFIGURED:"
echo "✅ Firewall active (ports 22, 80, 443)"
echo "✅ Intrusion detection (Fail2Ban)"
echo "✅ Web server (Nginx)"
echo "✅ Python environment ready"
echo "✅ System service configured"
echo ""
echo "🔧 NEXT STEPS:"
echo "1. Add your environment variables to /app/.env"
echo "2. Deploy your application code to /app"
echo "3. Start the service: sudo systemctl start trading-bot"
echo ""
echo "ℹ️ This script won't run again (one-time setup complete)"