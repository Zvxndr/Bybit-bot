#!/bin/bash
# git_safe_deployment.sh
# Deploy updates while preserving all data

set -e

echo "🚀 Starting Git-safe deployment..."

# 1. Pre-deployment backup
echo "🛡️ Creating pre-deployment backup..."
./scripts/backup_trading_data.sh

# 2. Stop services gracefully
echo "🛑 Stopping services gracefully..."
docker-compose down --timeout 30

# 3. Update code
echo "📥 Pulling latest code..."
git stash  # Stash any local changes
git pull origin main
git stash pop 2>/dev/null || echo "No stash to apply"

# 4. Update dependencies (if needed)
echo "📦 Checking for dependency updates..."
if [ -f "requirements.txt" ]; then
    echo "Rebuilding Python dependencies..."
    docker-compose build --no-cache bybit-bot
fi

# 5. Database migrations (if any)
echo "🗄️ Running database migrations..."
# Add migration commands here when needed
# alembic upgrade head

# 6. Start services with preserved data
echo "🚀 Starting services with preserved data..."
docker-compose up -d

# 7. Health check
echo "🏥 Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🔍 Service status:"
docker-compose ps
echo ""

# Check application health
echo "🏥 Application health check..."
timeout 60 bash -c 'until curl -f http://localhost:8080/health; do echo "Waiting for app..."; sleep 5; done'

echo ""
echo "✅ Git-safe deployment completed successfully!"
echo "📊 Your ML models and trading data are preserved"
echo "🔍 Monitor logs: docker-compose logs -f bybit-bot"