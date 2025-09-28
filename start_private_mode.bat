@echo off
REM =============================================================================
REM Bybit Trading Bot - Private Use Mode Launcher (Windows)
REM =============================================================================
REM
REM This batch file launches the Bybit Trading Bot in Private Use Mode
REM with comprehensive debugging and enhanced safety features.
REM
REM Features:
REM - Ultra-safe private use configuration
REM - Comprehensive debugging logs
REM - Real-time performance monitoring
REM - Conservative risk management
REM - Testnet-first approach
REM =============================================================================

echo.
echo ===============================================================
echo   BYBIT TRADING BOT - PRIVATE USE MODE LAUNCHER
echo ===============================================================
echo.
echo 🛡️ Ultra-Safe Configuration
echo 🔍 Comprehensive Debugging Enabled
echo 📊 Real-Time Monitoring Active  
echo 💰 Conservative Risk Management
echo 🏦 Testnet-First Approach
echo.

REM Set console to UTF-8 for emoji support
chcp 65001 > nul

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Display Python version
echo 🐍 Python Version:
python --version

REM Check if we're in the right directory
if not exist "private_mode_launcher.py" (
    echo ❌ private_mode_launcher.py not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo ❌ .env file not found
    echo Please create .env file with your configuration
    pause
    exit /b 1
)

echo.
echo 🚀 Launching Private Use Mode...
echo.
echo 📝 Logs will be saved to the 'logs' directory
echo 🔍 Debug information will be displayed in real-time
echo 🛑 Press Ctrl+C to stop the bot safely
echo.
echo ===============================================================
echo.

REM Launch the private mode launcher
python private_mode_launcher.py

REM Check exit code
if errorlevel 1 (
    echo.
    echo ❌ Application exited with errors
    echo Check the logs in the 'logs' directory for details
    pause
) else (
    echo.
    echo ✅ Application shut down cleanly
)

pause