# backup_trading_data.ps1
# PowerShell version of backup script for Windows

param(
    [string]$BackupLocation = "$env:USERPROFILE\bybit-bot-backups"
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = "$BackupLocation\$timestamp"

Write-Host "🛡️ Starting comprehensive trading bot backup..." -ForegroundColor Green
Write-Host "📁 Backup location: $backupDir" -ForegroundColor Cyan

# Create backup directory
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

try {
    # 1. Database backup (PostgreSQL)
    Write-Host "📊 Backing up PostgreSQL database..." -ForegroundColor Yellow
    docker exec bybit-bot-fresh_postgres_1 pg_dump -U trading_user trading_bot > "$backupDir\database_backup.sql"
    
    # 2. ML Models backup (Docker volume)
    Write-Host "🤖 Backing up ML models..." -ForegroundColor Yellow
    docker run --rm -v bybit_models:/data -v "${backupDir}:/backup" alpine tar czf /backup/models.tar.gz -C /data .
    
    # 3. Trading data backup (Docker volume)
    Write-Host "📈 Backing up trading data..." -ForegroundColor Yellow
    docker run --rm -v bybit_trading_data:/data -v "${backupDir}:/backup" alpine tar czf /backup/trading_data.tar.gz -C /data .
    
    # 4. Configuration backup
    Write-Host "⚙️ Backing up configurations..." -ForegroundColor Yellow
    if (Test-Path ".env") {
        Copy-Item ".env" "$backupDir\env_config" -ErrorAction SilentlyContinue
    }
    if (Test-Path "config") {
        Copy-Item -Recurse "config" "$backupDir\config" -ErrorAction SilentlyContinue
    }
    
    # 5. Application logs backup
    Write-Host "📝 Backing up logs..." -ForegroundColor Yellow
    docker run --rm -v bybit_logs:/data -v "${backupDir}:/backup" alpine tar czf /backup/logs.tar.gz -C /data .
    
    # 6. Create backup manifest
    $gitCommit = try { git rev-parse HEAD } catch { "unknown" }
    $gitBranch = try { git branch --show-current } catch { "unknown" }
    
    $manifest = @{
        backup_timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        git_commit = $gitCommit
        git_branch = $gitBranch
        components_backed_up = @(
            "postgresql_database",
            "ml_models",
            "trading_data", 
            "configuration_files",
            "application_logs",
            "docker_volumes"
        )
        restore_instructions = "Use restore_trading_data.ps1 with this backup directory"
    } | ConvertTo-Json -Depth 3
    
    $manifest | Out-File "$backupDir\backup_manifest.json" -Encoding UTF8
    
    # Calculate backup size
    $backupSize = (Get-ChildItem -Recurse $backupDir | Measure-Object -Property Length -Sum).Sum
    $backupSizeMB = [math]::Round($backupSize / 1MB, 2)
    
    Write-Host ""
    Write-Host "✅ Backup completed successfully!" -ForegroundColor Green
    Write-Host "📁 Backup location: $backupDir" -ForegroundColor Cyan
    Write-Host "💾 Total size: ${backupSizeMB} MB" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🔄 To restore this backup, run:" -ForegroundColor White
    Write-Host "   .\scripts\restore_trading_data.ps1 '$backupDir'" -ForegroundColor Yellow
    
} catch {
    Write-Host "❌ Backup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}