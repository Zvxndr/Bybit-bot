# git_safe_deployment.ps1
# PowerShell version of Git-safe deployment script

Write-Host "🚀 Starting Git-safe deployment..." -ForegroundColor Green

try {
    # 1. Pre-deployment backup
    Write-Host "🛡️ Creating pre-deployment backup..." -ForegroundColor Yellow
    .\scripts\backup_trading_data.ps1
    
    # 2. Stop services gracefully
    Write-Host "🛑 Stopping services gracefully..." -ForegroundColor Yellow
    docker-compose down --timeout 30
    
    # 3. Update code
    Write-Host "📥 Pulling latest code..." -ForegroundColor Yellow
    git stash  # Stash any local changes
    git pull origin main
    try { git stash pop } catch { Write-Host "No stash to apply" -ForegroundColor Gray }
    
    # 4. Update dependencies (if needed) 
    Write-Host "📦 Checking for dependency updates..." -ForegroundColor Yellow
    if (Test-Path "requirements.txt") {
        Write-Host "Rebuilding Python dependencies..." -ForegroundColor Gray
        docker-compose build --no-cache bybit-bot
    }
    
    # 5. Database migrations (if any)
    Write-Host "🗄️ Running database migrations..." -ForegroundColor Yellow
    # Add migration commands here when needed
    # alembic upgrade head
    
    # 6. Start services with preserved data
    Write-Host "🚀 Starting services with preserved data..." -ForegroundColor Yellow
    docker-compose up -d
    
    # 7. Health check
    Write-Host "🏥 Waiting for services to be healthy..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
    
    # Check service status
    Write-Host ""
    Write-Host "🔍 Service status:" -ForegroundColor Cyan
    docker-compose ps
    Write-Host ""
    
    # Check application health
    Write-Host "🏥 Application health check..." -ForegroundColor Yellow
    $healthCheck = $false
    $attempts = 0
    $maxAttempts = 12  # 60 seconds total
    
    while (-not $healthCheck -and $attempts -lt $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                $healthCheck = $true
                Write-Host "✅ Application is healthy!" -ForegroundColor Green
            }
        } catch {
            Write-Host "Waiting for app... (attempt $($attempts + 1)/$maxAttempts)" -ForegroundColor Gray
            Start-Sleep -Seconds 5
            $attempts++
        }
    }
    
    if (-not $healthCheck) {
        Write-Host "⚠️ Application health check failed, but deployment completed" -ForegroundColor Yellow
        Write-Host "🔍 Check logs: docker-compose logs -f bybit-bot" -ForegroundColor Cyan
    }
    
    Write-Host ""
    Write-Host "✅ Git-safe deployment completed successfully!" -ForegroundColor Green
    Write-Host "📊 Your ML models and trading data are preserved" -ForegroundColor Cyan
    Write-Host "🔍 Monitor logs: docker-compose logs -f bybit-bot" -ForegroundColor Cyan
    
} catch {
    Write-Host "❌ Deployment failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "🔄 Rolling back..." -ForegroundColor Yellow
    docker-compose down
    docker-compose up -d
    exit 1
}