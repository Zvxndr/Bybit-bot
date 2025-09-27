# Digital Ocean Deployment Script with Node.js Integration (PowerShell)
# =====================================================================
# This script deploys the trading bot to Digital Ocean with Node.js support

Write-Host "🚀 Digital Ocean Trading Bot Deployment with Node.js" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

# Configuration
$APP_NAME = "bybit-trading-bot"
$DOCKER_IMAGE = "${APP_NAME}:latest"
$CONTAINER_NAME = "${APP_NAME}-container"
$NODE_VERSION = "22.20.0"
$NPM_VERSION = "10.9.3"

function Log-Info($message) {
    Write-Host "ℹ️  $message" -ForegroundColor Blue
}

function Log-Success($message) {
    Write-Host "✅ $message" -ForegroundColor Green
}

function Log-Warning($message) {
    Write-Host "⚠️  $message" -ForegroundColor Yellow
}

function Log-Error($message) {
    Write-Host "❌ $message" -ForegroundColor Red
}

# Step 1: Verify Docker is installed
Log-Info "Checking Docker installation..."
try {
    $dockerVersion = docker --version
    Log-Success "Docker is installed: $dockerVersion"
} catch {
    Log-Error "Docker is not installed. Please install Docker Desktop first."
    exit 1
}

# Step 2: Pull Node.js Docker image for verification
Log-Info "Pulling Node.js Docker image..."
docker pull node:22-alpine
if ($LASTEXITCODE -eq 0) {
    Log-Success "Node.js Docker image pulled successfully"
} else {
    Log-Error "Failed to pull Node.js Docker image"
    exit 1
}

# Step 3: Verify Node.js version in container
Log-Info "Verifying Node.js version..."
$nodeVer = docker run --rm node:22-alpine node -v
$npmVer = docker run --rm node:22-alpine npm -v
Log-Success "Node.js version: $nodeVer (Expected: v$NODE_VERSION)"
Log-Success "NPM version: $npmVer (Expected: $NPM_VERSION)"

# Step 4: Stop existing container if running
Log-Info "Stopping existing containers..."
docker stop $CONTAINER_NAME 2>$null
if ($LASTEXITCODE -ne 0) {
    Log-Warning "No existing container to stop"
}
docker rm $CONTAINER_NAME 2>$null
if ($LASTEXITCODE -ne 0) {
    Log-Warning "No existing container to remove"
}

# Step 5: Build the application with Node.js support
Log-Info "Building application with Node.js support..."
if (Test-Path "Dockerfile.nodejs") {
    docker build -f Dockerfile.nodejs -t $DOCKER_IMAGE .
    if ($LASTEXITCODE -eq 0) {
        Log-Success "Application built with Node.js integration"
    } else {
        Log-Error "Failed to build application"
        exit 1
    }
} else {
    Log-Warning "Dockerfile.nodejs not found, using standard Dockerfile"
    docker build -t $DOCKER_IMAGE .
    if ($LASTEXITCODE -ne 0) {
        Log-Error "Failed to build application"
        exit 1
    }
}

# Step 6: Create necessary directories on host
Log-Info "Creating host directories..."
New-Item -ItemType Directory -Force -Path ".\data" | Out-Null
New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null
New-Item -ItemType Directory -Force -Path ".\config\secrets" | Out-Null
Log-Success "Host directories created"

# Step 7: Deploy the container
Log-Info "Deploying container with Node.js support..."
$deployCommand = @"
docker run -d ``
    --name $CONTAINER_NAME ``
    --restart unless-stopped ``
    -p 8080:8080 ``
    -p 3000:3000 ``
    -v "$(pwd)/data:/app/data" ``
    -v "$(pwd)/logs:/app/logs" ``
    -v "$(pwd)/config:/app/config" ``
    -e NODE_ENV=production ``
    -e PYTHON_ENV=production ``
    --health-cmd="curl -f http://localhost:8080/health || exit 1" ``
    --health-interval=30s ``
    --health-timeout=10s ``
    --health-retries=3 ``
    $DOCKER_IMAGE
"@

Invoke-Expression $deployCommand
if ($LASTEXITCODE -eq 0) {
    Log-Success "Container deployed successfully"
} else {
    Log-Error "Failed to deploy container"
    exit 1
}

# Step 8: Wait for container to start
Log-Info "Waiting for container to start..."
Start-Sleep -Seconds 10

# Step 9: Verify deployment
Log-Info "Verifying deployment..."
$runningContainers = docker ps
if ($runningContainers -match $CONTAINER_NAME) {
    Log-Success "Container is running"
    
    # Check container health
    try {
        $healthStatus = docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>$null
        Log-Info "Container health status: $healthStatus"
    } catch {
        Log-Info "Container health status: no-health-check"
    }
    
    # Show container logs
    Log-Info "Container startup logs:"
    docker logs --tail 20 $CONTAINER_NAME
    
    # Verify Node.js is available in container
    Log-Info "Verifying Node.js in running container..."
    try {
        $containerNodeVer = docker exec $CONTAINER_NAME node -v 2>$null
        $containerNpmVer = docker exec $CONTAINER_NAME npm -v 2>$null
        
        if ($containerNodeVer) {
            Log-Success "Node.js in container: $containerNodeVer"
            Log-Success "NPM in container: $containerNpmVer"
        } else {
            Log-Warning "Node.js not available in container (this might be expected for Python-only images)"
        }
    } catch {
        Log-Warning "Could not verify Node.js in container"
    }
    
} else {
    Log-Error "Container failed to start"
    Log-Info "Container logs:"
    docker logs $CONTAINER_NAME
    exit 1
}

# Step 10: Display deployment information
Log-Success "Deployment completed successfully!"
Write-Host ""
Write-Host "📊 Deployment Summary:" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan
Write-Host "• Application: $APP_NAME"
Write-Host "• Container: $CONTAINER_NAME"
Write-Host "• Image: $DOCKER_IMAGE"
Write-Host "• Node.js Version: $nodeVer"
Write-Host "• NPM Version: $npmVer"
Write-Host "• Main Port: 8080 (API/Backend)"
Write-Host "• Frontend Port: 3000 (if applicable)"
Write-Host "• Health Check: http://localhost:8080/health"
Write-Host ""
Write-Host "🔧 Management Commands:" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host "• View logs: docker logs -f $CONTAINER_NAME"
Write-Host "• Stop: docker stop $CONTAINER_NAME"
Write-Host "• Start: docker start $CONTAINER_NAME"
Write-Host "• Restart: docker restart $CONTAINER_NAME"
Write-Host "• Shell access: docker exec -it $CONTAINER_NAME /bin/sh"
Write-Host "• Node.js shell: docker exec -it $CONTAINER_NAME node"
Write-Host ""
Write-Host "🌐 Access URLs:" -ForegroundColor Cyan
Write-Host "==============" -ForegroundColor Cyan
Write-Host "• API: http://localhost:8080"
Write-Host "• Health: http://localhost:8080/health"
Write-Host "• Frontend: http://localhost:3000 (if available)"

Log-Success "Trading bot with Node.js support is now running on Digital Ocean!"