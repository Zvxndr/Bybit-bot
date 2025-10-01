@echo off
REM Debug Deployment Startup Script
REM This script starts the application in debug mode with automatic testing

echo ================================
echo   BYBIT BOT - DEBUG DEPLOYMENT
echo ================================

REM Set debug environment variables
set DEBUG_MODE=true
set ENVIRONMENT=debug
set AUTO_RUN_TESTS=true

REM Show current settings
echo Debug Mode: %DEBUG_MODE%
echo Environment: %ENVIRONMENT%
echo Auto Tests: %AUTO_RUN_TESTS%
echo.

REM Start the application with debug logging
echo Starting application in debug mode...
echo Debug scripts will run automatically after initialization.
echo.

python src/main.py --debug

echo.
echo Application stopped.
pause