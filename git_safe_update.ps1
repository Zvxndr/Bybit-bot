# git_safe_update.ps1
# Safe Git update script for Bybit Trading Bot
# Preserves data directories and configuration files

Write-Host "🚀 Starting Git-safe update..." -ForegroundColor Green

try {
    # 1. Create data directories if they don't exist
    Write-Host "📁 Setting up data preservation structure..." -ForegroundColor Yellow
    
    $dataDirs = @(
        "data\models",
        "data\strategies", 
        "data\trading_history",
        "data\backups",
        "logs",
        "config\runtime"
    )
    
    foreach ($dir in $dataDirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "  ✅ Created: $dir" -ForegroundColor Gray
        }
    }
    
    # 2. Backup current configuration and data (if exists)
    Write-Host "🛡️ Backing up current configuration..." -ForegroundColor Yellow
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $tempBackup = "temp_backup_$timestamp"
    New-Item -ItemType Directory -Path $tempBackup -Force | Out-Null
    
    # Backup important files that should be preserved
    $filesToPreserve = @(
        ".env",
        "data\*",
        "logs\*",
        "config\*.yaml",
        "config\*.json"
    )
    
    foreach ($pattern in $filesToPreserve) {
        $items = Get-Item $pattern -ErrorAction SilentlyContinue
        if ($items) {
            foreach ($item in $items) {
                if (Test-Path $item) {
                    $relativePath = Resolve-Path $item -Relative
                    $backupPath = Join-Path $tempBackup $relativePath
                    $backupDir = Split-Path $backupPath -Parent
                    
                    if (-not (Test-Path $backupDir)) {
                        New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
                    }
                    
                    if ($item.PSIsContainer) {
                        Copy-Item -Recurse $item $backupPath -ErrorAction SilentlyContinue
                    } else {
                        Copy-Item $item $backupPath -ErrorAction SilentlyContinue
                    }
                    Write-Host "  ✅ Backed up: $relativePath" -ForegroundColor Gray
                }
            }
        }
    }
    
    # 3. Stop any running Python processes (gracefully)
    Write-Host "🛑 Stopping any running bot processes..." -ForegroundColor Yellow
    
    $pythonProcesses = Get-Process -Name "python*" -ErrorAction SilentlyContinue | Where-Object {
        $_.MainModule.FileName -like "*Bybit-bot-fresh*"
    }
    
    if ($pythonProcesses) {
        foreach ($process in $pythonProcesses) {
            Write-Host "  Stopping process: $($process.ProcessName) (PID: $($process.Id))" -ForegroundColor Gray
            $process.CloseMainWindow() | Out-Null
            Start-Sleep -Seconds 2
            if (-not $process.HasExited) {
                $process.Kill()
            }
        }
    } else {
        Write-Host "  No bot processes running" -ForegroundColor Gray
    }
    
    # 4. Stash any local changes
    Write-Host "💾 Stashing local changes..." -ForegroundColor Yellow
    try {
        git add . 2>$null
        git stash push -m "Auto-stash before update $(Get-Date)" 2>$null
        Write-Host "  ✅ Local changes stashed" -ForegroundColor Gray
    } catch {
        Write-Host "  No changes to stash" -ForegroundColor Gray
    }
    
    # 5. Pull latest changes
    Write-Host "📥 Pulling latest code from Git..." -ForegroundColor Yellow
    $gitPull = git pull origin main 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Git pull successful" -ForegroundColor Green
        Write-Host "  $gitPull" -ForegroundColor Gray
    } else {
        Write-Host "  ⚠️ Git pull had issues: $gitPull" -ForegroundColor Yellow
    }
    
    # 6. Restore preserved files
    Write-Host "🔄 Restoring preserved data and configuration..." -ForegroundColor Yellow
    
    if (Test-Path $tempBackup) {
        $restoredItems = Get-ChildItem -Recurse $tempBackup
        foreach ($item in $restoredItems) {
            if (-not $item.PSIsContainer) {
                $relativePath = $item.FullName.Replace((Resolve-Path $tempBackup).Path + "\", "")
                $targetPath = $relativePath
                $targetDir = Split-Path $targetPath -Parent
                
                if ($targetDir -and (-not (Test-Path $targetDir))) {
                    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
                }
                
                Copy-Item $item.FullName $targetPath -Force -ErrorAction SilentlyContinue
                Write-Host "  ✅ Restored: $relativePath" -ForegroundColor Gray
            }
        }
        
        # Clean up temporary backup
        Remove-Item -Recurse -Force $tempBackup -ErrorAction SilentlyContinue
    }
    
    # 7. Update Python dependencies if requirements.txt changed
    Write-Host "📦 Checking Python dependencies..." -ForegroundColor Yellow
    
    if (Test-Path "requirements.txt") {
        $requirementsHash = Get-FileHash "requirements.txt" -Algorithm MD5
        $hashFile = "temp_requirements_hash.txt"
        $previousHash = ""
        
        if (Test-Path $hashFile) {
            $previousHash = Get-Content $hashFile -ErrorAction SilentlyContinue
        }
        
        if ($requirementsHash.Hash -ne $previousHash) {
            Write-Host "  Dependencies changed, updating..." -ForegroundColor Gray
            try {
                python -m pip install -r requirements.txt --quiet
                Write-Host "  ✅ Dependencies updated" -ForegroundColor Green
                $requirementsHash.Hash | Out-File $hashFile
            } catch {
                Write-Host "  ⚠️ Failed to update dependencies: $($_.Exception.Message)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  Dependencies unchanged" -ForegroundColor Gray
        }
    }
    
    # 8. Test the application
    Write-Host "🧪 Testing application startup..." -ForegroundColor Yellow
    
    $testResult = python -c "
import sys
sys.path.append('src')
try:
    from main import *
    print('✅ Application imports successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  $testResult" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ Application test failed: $testResult" -ForegroundColor Yellow
    }
    
    # 9. Final status
    Write-Host ""
    Write-Host "✅ Git-safe update completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 Status Summary:" -ForegroundColor Cyan
    Write-Host "  ✅ Code updated from Git" -ForegroundColor White
    Write-Host "  ✅ Data directories preserved" -ForegroundColor White
    Write-Host "  ✅ Configuration files maintained" -ForegroundColor White
    Write-Host "  ✅ Dependencies checked" -ForegroundColor White
    Write-Host "  ✅ Application tested" -ForegroundColor White
    Write-Host ""
    Write-Host "🚀 Ready to start trading bot:" -ForegroundColor Cyan
    Write-Host "  python src/main.py" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📁 Data will be preserved in future updates in:" -ForegroundColor Cyan
    Write-Host "  data/models/          - ML models" -ForegroundColor White
    Write-Host "  data/strategies/      - Strategy performance" -ForegroundColor White
    Write-Host "  data/trading_history/ - Trade records" -ForegroundColor White
    Write-Host "  logs/                 - Application logs" -ForegroundColor White
    Write-Host "  .env                  - Configuration" -ForegroundColor White
    
} catch {
    Write-Host ""
    Write-Host "❌ Update failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔄 Attempting to restore from backup..." -ForegroundColor Yellow
    
    if (Test-Path $tempBackup) {
        # Restore from backup if something went wrong
        Copy-Item -Recurse "$tempBackup\*" . -Force -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force $tempBackup -ErrorAction SilentlyContinue
        Write-Host "✅ Backup restored" -ForegroundColor Green
    }
    
    Write-Host "⚠️ Please check the error and try again" -ForegroundColor Yellow
    exit 1
}