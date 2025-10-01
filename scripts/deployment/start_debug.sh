#!/bin/bash
# Debug Deployment Startup Script
# This script starts the application in debug mode with automatic testing

echo "================================"
echo "   BYBIT BOT - DEBUG DEPLOYMENT"
echo "================================"

# Set debug environment variables
export DEBUG_MODE=true
export ENVIRONMENT=debug
export AUTO_RUN_TESTS=true

# Show current settings
echo "Debug Mode: $DEBUG_MODE"
echo "Environment: $ENVIRONMENT"
echo "Auto Tests: $AUTO_RUN_TESTS"
echo ""

# Start the application with debug logging
echo "Starting application in debug mode..."
echo "Debug scripts will run automatically after initialization."
echo ""

python3 src/main.py --debug

echo ""
echo "Application stopped."