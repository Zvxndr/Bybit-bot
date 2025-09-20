#!/bin/bash
# Docker Build Script
# Builds all Docker images for the ML Trading Bot

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="ml-trading-bot"
VERSION=${VERSION:-"latest"}
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")}

echo -e "${BLUE}🚀 Building ML Trading Bot Docker Images${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "Version: ${GREEN}${VERSION}${NC}"
echo -e "Build Date: ${GREEN}${BUILD_DATE}${NC}"
echo -e "Git Commit: ${GREEN}${GIT_COMMIT}${NC}"
echo ""

# Function to build image
build_image() {
    local dockerfile=$1
    local tag_suffix=$2
    local context=${3:-"."}
    
    local full_tag="${IMAGE_NAME}:${tag_suffix}-${VERSION}"
    local latest_tag="${IMAGE_NAME}:${tag_suffix}-latest"
    
    echo -e "${YELLOW}📦 Building ${tag_suffix} image...${NC}"
    
    docker build \
        --file "${dockerfile}" \
        --tag "${full_tag}" \
        --tag "${latest_tag}" \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg VERSION="${VERSION}" \
        --build-arg GIT_COMMIT="${GIT_COMMIT}" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        "${context}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully built ${full_tag}${NC}"
    else
        echo -e "${RED}❌ Failed to build ${full_tag}${NC}"
        exit 1
    fi
    
    echo ""
}

# Build main application image
echo -e "${YELLOW}🏗️  Building main application image...${NC}"
build_image "Dockerfile" "app"

# Build API service image
echo -e "${YELLOW}🏗️  Building API service image...${NC}"
build_image "docker/Dockerfile.api" "api"

# Build Dashboard service image
echo -e "${YELLOW}🏗️  Building Dashboard service image...${NC}"
build_image "docker/Dockerfile.dashboard" "dashboard"

# Show built images
echo -e "${BLUE}📋 Built Images:${NC}"
docker images | grep "${IMAGE_NAME}" | head -10

# Optional: Push to registry
if [ "${PUSH_TO_REGISTRY}" = "true" ]; then
    echo -e "${YELLOW}📤 Pushing images to registry...${NC}"
    
    REGISTRY=${REGISTRY:-"docker.io"}
    REGISTRY_USER=${REGISTRY_USER:-""}
    
    if [ -n "${REGISTRY_USER}" ]; then
        docker tag "${IMAGE_NAME}:app-${VERSION}" "${REGISTRY}/${REGISTRY_USER}/${IMAGE_NAME}:app-${VERSION}"
        docker tag "${IMAGE_NAME}:api-${VERSION}" "${REGISTRY}/${REGISTRY_USER}/${IMAGE_NAME}:api-${VERSION}"
        docker tag "${IMAGE_NAME}:dashboard-${VERSION}" "${REGISTRY}/${REGISTRY_USER}/${IMAGE_NAME}:dashboard-${VERSION}"
        
        docker push "${REGISTRY}/${REGISTRY_USER}/${IMAGE_NAME}:app-${VERSION}"
        docker push "${REGISTRY}/${REGISTRY_USER}/${IMAGE_NAME}:api-${VERSION}"
        docker push "${REGISTRY}/${REGISTRY_USER}/${IMAGE_NAME}:dashboard-${VERSION}"
        
        echo -e "${GREEN}✅ Images pushed to ${REGISTRY}${NC}"
    else
        echo -e "${YELLOW}⚠️  REGISTRY_USER not set, skipping push${NC}"
    fi
fi

# Cleanup dangling images
echo -e "${YELLOW}🧹 Cleaning up dangling images...${NC}"
docker image prune -f

echo -e "${GREEN}🎉 Build completed successfully!${NC}"
echo ""
echo -e "${BLUE}🚀 To run the application:${NC}"
echo -e "  Development: ${YELLOW}docker-compose -f docker-compose.yml -f docker-compose.dev.yml up${NC}"
echo -e "  Production:  ${YELLOW}docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d${NC}"