#!/bin/bash
# DigitalOcean App Platform startup script
# Ensures persistent data directories exist with proper permissions

echo "🚀 DigitalOcean App Platform - Startup Script"
echo "📅 $(date)"

# Create persistent data directories if they don't exist
echo "📁 Creating persistent data directories..."
mkdir -p /app/data/models
mkdir -p /app/data/strategies  
mkdir -p /app/data/speed_demon_cache
mkdir -p /app/logs
mkdir -p /app/config

# Set proper permissions for SQLite database
echo "🔧 Setting up database permissions..."
touch /app/data/trading_bot.db
chmod 664 /app/data/trading_bot.db

# Create backup directory for tax logs and live trading data  
mkdir -p /app/data/backups
mkdir -p /app/data/tax_logs
mkdir -p /app/data/live_trading

# Log persistent volume status
echo "💾 Persistent volumes status:"
df -h /app/data /app/logs /app/config

# Verify critical files exist
echo "🔍 Checking for existing data..."
if [ -f "/app/data/trading_bot.db" ]; then
    echo "✅ Database exists"
    sqlite3 /app/data/trading_bot.db "SELECT COUNT(*) as records FROM historical_data;" 2>/dev/null || echo "❌ Database needs initialization"
else
    echo "⚠️ Database will be created on first run"
fi

# List data contents
echo "📊 Data directory contents:"
ls -la /app/data/

# Run production startup validation and diagnostics
echo "🔍 Running production startup validation..."
python src/production_startup_integration.py

if [ $? -eq 0 ]; then
    echo "✅ Startup validation passed - launching application"
else
    echo "⚠️ Startup validation issues detected - proceeding with caution"
fi

echo "🎯 Starting main application with enhanced monitoring..."
exec python src/main.py