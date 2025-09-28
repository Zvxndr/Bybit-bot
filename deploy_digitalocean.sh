#!/bin/bash
# DigitalOcean Debug Deployment Script
# Optimized for DigitalOcean droplet deployment with automatic testing

echo "🌊 ================================"
echo "   BYBIT BOT - DIGITALOCEAN DEBUG"
echo "🌊 ================================"

# Detect if we're on DigitalOcean (common indicators)
if [[ -f /etc/digitalocean-release ]] || [[ $(hostname) == *"digitalocean"* ]] || [[ -n "$DIGITALOCEAN_DROPLET" ]]; then
    echo "✅ DigitalOcean environment detected"
    DIGITALOCEAN_DROPLET=true
else
    echo "🖥️  Local/Other environment detected"
    DIGITALOCEAN_DROPLET=false
fi

# Set debug environment variables
export DEBUG_MODE=true
export ENVIRONMENT=debug
export AUTO_RUN_TESTS=true
export DIGITALOCEAN_DROPLET=$DIGITALOCEAN_DROPLET

# Show environment info
echo "📊 Environment Info:"
echo "   Debug Mode: $DEBUG_MODE"
echo "   Environment: $ENVIRONMENT"
echo "   Auto Tests: $AUTO_RUN_TESTS"
echo "   DigitalOcean: $DIGITALOCEAN_DROPLET"
echo "   Hostname: $(hostname)"
echo "   IP: $(hostname -I | awk '{print $1}')"
echo ""

# Check for required files
echo "🔍 Pre-flight checks..."

if [[ ! -f "src/main.py" ]]; then
    echo "❌ src/main.py not found!"
    exit 1
fi

if [[ ! -f "test_button_functions.py" ]]; then
    echo "❌ test_button_functions.py not found!"
    exit 1  
fi

if [[ ! -f "debug_data_wipe.py" ]]; then
    echo "❌ debug_data_wipe.py not found!"
    exit 1
fi

echo "✅ All required files present"

# Check Python
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found!"
    exit 1
fi

echo "✅ Python found: $PYTHON_CMD"

# Install requirements if needed
if [[ -f "requirements.txt" ]] || [[ -f "requirements_current.txt" ]]; then
    echo "📦 Installing requirements..."
    if [[ -f "requirements_current.txt" ]]; then
        $PYTHON_CMD -m pip install -r requirements_current.txt > /dev/null 2>&1
    else
        $PYTHON_CMD -m pip install -r requirements.txt > /dev/null 2>&1
    fi
    echo "✅ Requirements installed"
fi

# Start the application with debug logging
echo ""
echo "🚀 Starting Bybit Bot in Debug Mode..."
echo "🧪 Debug scripts will run automatically after initialization"
echo "🌐 Web interface will be available on port 8080"
echo "📋 All test results will be logged"
echo ""

if [[ "$DIGITALOCEAN_DROPLET" == "true" ]]; then
    DROPLET_IP=$(hostname -I | awk '{print $1}')
    echo "🌊 DigitalOcean Dashboard will be at: http://$DROPLET_IP:8080"
    echo ""
fi

echo "Press Ctrl+C to stop..."
echo ""

# Run the application
$PYTHON_CMD src/main.py --debug