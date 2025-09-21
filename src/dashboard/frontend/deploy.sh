#!/bin/bash
# Frontend Deployment Script for Bybit Trading Dashboard

echo "🚀 Starting Bybit Trading Dashboard Frontend Deployment"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed. Please install Node.js 18+ first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Node.js found: $(node --version)${NC}"

# Check if we're in the correct directory
if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ package.json not found. Please run this script from the frontend directory.${NC}"
    exit 1
fi

# Install dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
npm install

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
fi

# Build the application
echo -e "${BLUE}🔨 Building application...${NC}"
npm run build

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

# Start the application
echo -e "${BLUE}🌐 Starting dashboard server...${NC}"
echo -e "${GREEN}Dashboard will be available at: http://localhost:3000${NC}"
echo -e "${GREEN}Make sure the backend is running at: http://localhost:8001${NC}"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop the server${NC}"
echo ""

npm start