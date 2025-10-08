#!/bin/bash
# Auto-startup script for DigitalOcean deployment
# This runs once on first startup to configure security

# Create startup service that runs security setup on first boot
sudo tee /etc/systemd/system/auto-setup.service << 'EOF'
[Unit]
Description=Auto Security Setup on First Boot
After=network.target
Wants=network.target

[Service]
Type=oneshot
ExecStart=/app/setup_security.sh
RemainAfterExit=yes
User=root

[Install]
WantedBy=multi-user.target
EOF

# Enable the service to run on startup
sudo systemctl daemon-reload
sudo systemctl enable auto-setup

echo "✅ Auto-setup configured - will run security setup on first boot"