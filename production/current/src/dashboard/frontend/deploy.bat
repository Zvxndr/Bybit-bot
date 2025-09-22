@echo off
REM Frontend Deployment Script for Bybit Trading Dashboard (Windows)

echo 🚀 Starting Bybit Trading Dashboard Frontend Deployment

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js 18+ first.
    pause
    exit /b 1
)

echo ✅ Node.js found
node --version

REM Check if we're in the correct directory
if not exist "package.json" (
    echo ❌ package.json not found. Please run this script from the frontend directory.
    pause
    exit /b 1
)

REM Install dependencies
echo 📦 Installing dependencies...
call npm install

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Build the application
echo 🔨 Building application...
call npm run build

if %errorlevel% neq 0 (
    echo ❌ Build failed
    pause
    exit /b 1
)

REM Start the application
echo 🌐 Starting dashboard server...
echo Dashboard will be available at: http://localhost:3000
echo Make sure the backend is running at: http://localhost:8001
echo.
echo Press Ctrl+C to stop the server
echo.

call npm start