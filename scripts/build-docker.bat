@echo off
REM Docker Build Script for Windows
REM Builds all Docker images for the ML Trading Bot

echo 🚀 Building ML Trading Bot Docker Images
echo ================================================

REM Configuration
set IMAGE_NAME=ml-trading-bot
if "%VERSION%"=="" set VERSION=latest
for /f "tokens=*" %%i in ('powershell -Command "Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ'"') do set BUILD_DATE=%%i
for /f "tokens=*" %%i in ('git rev-parse --short HEAD 2^>nul') do set GIT_COMMIT=%%i
if "%GIT_COMMIT%"=="" set GIT_COMMIT=unknown

echo Version: %VERSION%
echo Build Date: %BUILD_DATE%
echo Git Commit: %GIT_COMMIT%
echo.

REM Build main application image
echo 📦 Building main application image...
docker build ^
    --file Dockerfile ^
    --tag %IMAGE_NAME%:app-%VERSION% ^
    --tag %IMAGE_NAME%:app-latest ^
    --build-arg BUILD_DATE=%BUILD_DATE% ^
    --build-arg VERSION=%VERSION% ^
    --build-arg GIT_COMMIT=%GIT_COMMIT% ^
    .

if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to build main application image
    exit /b 1
)
echo ✅ Successfully built main application image
echo.

REM Build API service image
echo 📦 Building API service image...
docker build ^
    --file docker\Dockerfile.api ^
    --tag %IMAGE_NAME%:api-%VERSION% ^
    --tag %IMAGE_NAME%:api-latest ^
    --build-arg BUILD_DATE=%BUILD_DATE% ^
    --build-arg VERSION=%VERSION% ^
    --build-arg GIT_COMMIT=%GIT_COMMIT% ^
    .

if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to build API service image
    exit /b 1
)
echo ✅ Successfully built API service image
echo.

REM Build Dashboard service image
echo 📦 Building Dashboard service image...
docker build ^
    --file docker\Dockerfile.dashboard ^
    --tag %IMAGE_NAME%:dashboard-%VERSION% ^
    --tag %IMAGE_NAME%:dashboard-latest ^
    --build-arg BUILD_DATE=%BUILD_DATE% ^
    --build-arg VERSION=%VERSION% ^
    --build-arg GIT_COMMIT=%GIT_COMMIT% ^
    .

if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to build Dashboard service image
    exit /b 1
)
echo ✅ Successfully built Dashboard service image
echo.

REM Show built images
echo 📋 Built Images:
docker images | findstr %IMAGE_NAME%

REM Cleanup dangling images
echo 🧹 Cleaning up dangling images...
docker image prune -f

echo 🎉 Build completed successfully!
echo.
echo 🚀 To run the application:
echo   Development: docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
echo   Production:  docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d