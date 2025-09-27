#!/bin/bash
# Digital Ocean Deployment Script with Node.js Integration
# ========================================================
# This script deploys the trading bot to Digital Ocean with Node.js support

set -e  # Exit on any error

echo "🚀 Digital Ocean Trading Bot Deployment with Node.js"
echo "===================================================="

# Configuration
APP_NAME="bybit-trading-bot"
DOCKER_IMAGE="$APP_NAME:latest"
CONTAINER_NAME="${APP_NAME}-container"
NODE_VERSION="22.20.0"
NPM_VERSION="10.9.3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Verify Docker is installed
log_info "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed. Please install Docker first."
    exit 1
fi
log_success "Docker is installed: $(docker --version)"

# Step 2: Pull Node.js Docker image for verification
log_info "Pulling Node.js Docker image..."
docker pull node:22-alpine
log_success "Node.js Docker image pulled successfully"

# Step 3: Verify Node.js version in container
log_info "Verifying Node.js version..."
NODE_VER=$(docker run --rm node:22-alpine node -v)
NPM_VER=$(docker run --rm node:22-alpine npm -v)
log_success "Node.js version: $NODE_VER (Expected: v$NODE_VERSION)"
log_success "NPM version: $NPM_VER (Expected: $NPM_VERSION)"

# Step 4: Stop existing container if running
log_info "Stopping existing containers..."
docker stop $CONTAINER_NAME 2>/dev/null || log_warning "No existing container to stop"
docker rm $CONTAINER_NAME 2>/dev/null || log_warning "No existing container to remove"

# Step 5: Build the application with Node.js support
log_info "Building application with Node.js support..."
if [ -f "Dockerfile.nodejs" ]; then
    docker build -f Dockerfile.nodejs -t $DOCKER_IMAGE .
    log_success "Application built with Node.js integration"
else
    log_warning "Dockerfile.nodejs not found, using standard Dockerfile"
    docker build -t $DOCKER_IMAGE .
fi

# Step 6: Create necessary directories on host
log_info "Creating host directories..."
mkdir -p ./data ./logs ./config/secrets
chmod 755 ./data ./logs
log_success "Host directories created"

# Step 7: Deploy the container
log_info "Deploying container with Node.js support..."
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -p 8080:8080 \
    -p 3000:3000 \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd)/config:/app/config" \
    -e NODE_ENV=production \
    -e PYTHON_ENV=production \
    --health-cmd="curl -f http://localhost:8080/health || exit 1" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-retries=3 \
    $DOCKER_IMAGE

log_success "Container deployed successfully"

# Step 8: Wait for container to start
log_info "Waiting for container to start..."
sleep 10

# Step 9: Verify deployment
log_info "Verifying deployment..."
if docker ps | grep -q $CONTAINER_NAME; then
    log_success "Container is running"
    
    # Check container health
    HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null || echo "no-health-check")
    log_info "Container health status: $HEALTH_STATUS"
    
    # Show container logs
    log_info "Container startup logs:"
    docker logs --tail 20 $CONTAINER_NAME
    
    # Verify Node.js is available in container
    log_info "Verifying Node.js in running container..."
    CONTAINER_NODE_VER=$(docker exec $CONTAINER_NAME node -v 2>/dev/null || echo "not-available")
    CONTAINER_NPM_VER=$(docker exec $CONTAINER_NAME npm -v 2>/dev/null || echo "not-available")
    
    if [ "$CONTAINER_NODE_VER" != "not-available" ]; then
        log_success "Node.js in container: $CONTAINER_NODE_VER"
        log_success "NPM in container: $CONTAINER_NPM_VER"
    else
        log_warning "Node.js not available in container (this might be expected for Python-only images)"
    fi
    
else
    log_error "Container failed to start"
    log_info "Container logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Step 10: Display deployment information
log_success "Deployment completed successfully!"
echo
echo "📊 Deployment Summary:"
echo "====================="
echo "• Application: $APP_NAME"
echo "• Container: $CONTAINER_NAME"
echo "• Image: $DOCKER_IMAGE"
echo "• Node.js Version: $NODE_VER"
echo "• NPM Version: $NPM_VER"
echo "• Main Port: 8080 (API/Backend)"
echo "• Frontend Port: 3000 (if applicable)"
echo "• Health Check: http://localhost:8080/health"
echo
echo "🔧 Management Commands:"
echo "======================"
echo "• View logs: docker logs -f $CONTAINER_NAME"
echo "• Stop: docker stop $CONTAINER_NAME"
echo "• Start: docker start $CONTAINER_NAME"
echo "• Restart: docker restart $CONTAINER_NAME"
echo "• Shell access: docker exec -it $CONTAINER_NAME /bin/sh"
echo "• Node.js shell: docker exec -it $CONTAINER_NAME node"
echo
echo "🌐 Access URLs:"
echo "=============="
echo "• API: http://localhost:8080"
echo "• Health: http://localhost:8080/health"
echo "• Frontend: http://localhost:3000 (if available)"

log_success "Trading bot with Node.js support is now running on Digital Ocean!"