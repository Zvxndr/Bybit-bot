#!/bin/bash
# backup_trading_data.sh
# Comprehensive backup script for all trading bot data

set -e

BACKUP_DIR="$HOME/bybit-bot-backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "🛡️ Starting comprehensive trading bot backup..."
echo "📁 Backup location: $BACKUP_DIR"

# 1. Database backup (PostgreSQL)
echo "📊 Backing up PostgreSQL database..."
docker exec bybit-bot-fresh_postgres_1 pg_dump -U trading_user trading_bot > "$BACKUP_DIR/database_backup.sql"

# 2. ML Models backup
echo "🤖 Backing up ML models..."
if [ -d "./data/models" ]; then
    cp -r ./data/models "$BACKUP_DIR/ml_models"
fi

# 3. Strategy data backup
echo "📈 Backing up strategy data..."
if [ -d "./data/strategies" ]; then
    cp -r ./data/strategies "$BACKUP_DIR/strategies"
fi

# 4. Configuration backup
echo "⚙️ Backing up configurations..."
cp .env "$BACKUP_DIR/env_config" 2>/dev/null || echo "No .env file found"
cp -r ./config "$BACKUP_DIR/config" 2>/dev/null || echo "No config directory found"

# 5. Application logs (last 7 days)
echo "📝 Backing up recent logs..."
find ./logs -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/" \; 2>/dev/null || echo "No recent logs found"

# 6. Docker volumes backup
echo "🐳 Backing up Docker volumes..."
docker run --rm -v bybit_trading_data:/data -v "$BACKUP_DIR":/backup alpine tar czf /backup/trading_data.tar.gz -C /data .
docker run --rm -v bybit_models:/data -v "$BACKUP_DIR":/backup alpine tar czf /backup/models.tar.gz -C /data .

# 7. Create backup manifest
cat > "$BACKUP_DIR/backup_manifest.json" << EOF
{
    "backup_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "components_backed_up": [
        "postgresql_database",
        "ml_models", 
        "strategy_data",
        "configuration_files",
        "application_logs",
        "docker_volumes"
    ],
    "restore_instructions": "Use restore_trading_data.sh with this backup directory"
}
EOF

echo "✅ Backup completed successfully!"
echo "📁 Backup location: $BACKUP_DIR"
echo "💾 Total size: $(du -sh $BACKUP_DIR | cut -f1)"
echo ""
echo "🔄 To restore this backup, run:"
echo "   ./scripts/restore_trading_data.sh $BACKUP_DIR"