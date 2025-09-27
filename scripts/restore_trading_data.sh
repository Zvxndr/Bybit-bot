#!/bin/bash
# restore_trading_data.sh
# Restore trading bot data from backup

set -e

if [ $# -eq 0 ]; then
    echo "❌ Usage: $0 <backup_directory>"
    echo "📁 Available backups:"
    ls -la "$HOME/bybit-bot-backups/" 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_DIR="$1"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "❌ Backup directory not found: $BACKUP_DIR"
    exit 1
fi

echo "🔄 Restoring trading bot data from backup..."
echo "📁 Backup source: $BACKUP_DIR"

# Confirm restore
read -p "⚠️ This will overwrite current data. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Restore cancelled"
    exit 1
fi

# 1. Stop services
echo "🛑 Stopping trading bot services..."
docker-compose down

# 2. Restore database
echo "📊 Restoring PostgreSQL database..."
docker-compose up -d postgres
sleep 10  # Wait for PostgreSQL to start
docker exec -i bybit-bot-fresh_postgres_1 psql -U trading_user -d trading_bot < "$BACKUP_DIR/database_backup.sql"

# 3. Restore ML models
echo "🤖 Restoring ML models..."
if [ -f "$BACKUP_DIR/models.tar.gz" ]; then
    docker run --rm -v bybit_models:/data -v "$BACKUP_DIR":/backup alpine tar xzf /backup/models.tar.gz -C /data
fi

# 4. Restore strategy data
echo "📈 Restoring strategy data..."
if [ -f "$BACKUP_DIR/trading_data.tar.gz" ]; then
    docker run --rm -v bybit_trading_data:/data -v "$BACKUP_DIR":/backup alpine tar xzf /backup/trading_data.tar.gz -C /data
fi

# 5. Restore configurations
echo "⚙️ Restoring configurations..."
if [ -f "$BACKUP_DIR/env_config" ]; then
    cp "$BACKUP_DIR/env_config" .env
fi
if [ -d "$BACKUP_DIR/config" ]; then
    cp -r "$BACKUP_DIR/config" ./
fi

# 6. Start all services
echo "🚀 Starting all services..."
docker-compose up -d

echo "✅ Restore completed successfully!"
echo "🔍 Check services status:"
echo "   docker-compose ps"
echo "   docker-compose logs -f bybit-bot"