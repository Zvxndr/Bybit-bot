#!/bin/bash
# DigitalOcean Auto-Deploy Startup Script
# Handles security setup and application startup

echo "🚀 Starting DigitalOcean Auto-Deploy..."

# Set working directory
cd /app

# Run security setup if not already done
if [ ! -f "/app/.security_setup_complete" ]; then
    echo "🔒 Running first-time security setup..."
    if [ -f "setup_security.sh" ]; then
        chmod +x setup_security.sh
        ./setup_security.sh
        touch /app/.security_setup_complete
        echo "✅ Security setup completed"
    else
        echo "⚠️ Security script not found, skipping..."
    fi
else
    echo "✅ Security already configured, skipping..."
fi

# Install/update Python dependencies
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p logs data

# Set up environment
export PYTHONPATH=/app

# Start the application
echo "🚀 Starting trading bot..."
exec python src/main.py