# ==============================================================================
# Bybit Trading Bot - Private Use Mode Launcher (PowerShell)
# ==============================================================================
#
# This PowerShell script launches the Bybit Trading Bot in Private Use Mode
# with comprehensive debugging and enhanced safety features.
#
# Features:
# - Ultra-safe private use configuration
# - Comprehensive debugging logs
# - Real-time performance monitoring
# - Conservative risk management
# - Testnet-first approach
# ==============================================================================

param(
    [switch]$Silent,
    [switch]$Help
)

# Function to display help
function Show-Help {
    Write-Host @"

Bybit Trading Bot - Private Use Mode Launcher

USAGE:
    .\start_private_mode.ps1 [OPTIONS]

OPTIONS:
    -Silent     Run without interactive prompts
    -Help       Display this help message

EXAMPLES:
    .\start_private_mode.ps1
    .\start_private_mode.ps1 -Silent

"@ -ForegroundColor Green
    exit 0
}

if ($Help) { Show-Help }

# Clear screen
Clear-Host

Write-Host @"

===============================================================
  BYBIT TRADING BOT - PRIVATE USE MODE LAUNCHER
===============================================================

🛡️ Ultra-Safe Configuration
🔍 Comprehensive Debugging Enabled
📊 Real-Time Monitoring Active  
💰 Conservative Risk Management
🏦 Testnet-First Approach

"@ -ForegroundColor Cyan

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Host "🐍 Python Version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again" -ForegroundColor Yellow
    if (-not $Silent) { Read-Host "Press Enter to exit" }
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "private_mode_launcher.py")) {
    Write-Host "❌ private_mode_launcher.py not found" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory" -ForegroundColor Yellow
    if (-not $Silent) { Read-Host "Press Enter to exit" }
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "❌ .env file not found" -ForegroundColor Red
    Write-Host "Please create .env file with your configuration" -ForegroundColor Yellow
    if (-not $Silent) { Read-Host "Press Enter to exit" }
    exit 1
}

Write-Host @"

🚀 Launching Private Use Mode...

📝 Logs will be saved to the 'logs' directory
🔍 Debug information will be displayed in real-time
🛑 Press Ctrl+C to stop the bot safely

===============================================================

"@ -ForegroundColor Yellow

try {
    # Launch the private mode launcher
    python private_mode_launcher.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Application shut down cleanly" -ForegroundColor Green
    } else {
        Write-Host "`n❌ Application exited with errors" -ForegroundColor Red
        Write-Host "Check the logs in the 'logs' directory for details" -ForegroundColor Yellow
    }
} catch {
    Write-Host "`n❌ Failed to launch the application" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

if (-not $Silent) {
    Read-Host "`nPress Enter to exit"
}