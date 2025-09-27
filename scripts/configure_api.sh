#!/bin/bash
# Quick API Configuration Script
# Run this after getting your Bybit API credentials

echo "🚀 Bybit Trading Bot - API Configuration"
echo "========================================"

read -p "Enter your Bybit API Key: " API_KEY
read -s -p "Enter your Bybit API Secret: " API_SECRET
echo
read -p "Use Testnet? (y/n, default: y): " TESTNET

# Set testnet default
if [[ -z "$TESTNET" || "$TESTNET" == "y" ]]; then
    TESTNET_BOOL="true"
    echo "✅ Using TESTNET mode (safe for testing)"
else
    TESTNET_BOOL="false"
    echo "⚠️  Using LIVE TRADING mode"
fi

# Update the config file
cat > config/secrets.yaml << EOF
# Live Trading Configuration - Generated $(date)
bybit:
  api_key: "$API_KEY"
  api_secret: "$API_SECRET" 
  testnet: $TESTNET_BOOL

trading:
  max_position_size: 100.00
  confidence_threshold: 0.80
  leverage: 5
  stop_loss_pct: 2.0
  take_profit_pct: 4.0

notifications:
  email_alerts: true
  profit_threshold: 50.00
  loss_threshold: -25.00
EOF

echo "✅ Configuration updated!"
echo "📋 Next steps:"
echo "   1. Restart your deployment"  
echo "   2. Monitor logs for 'Bybit client connected'"
echo "   3. Check your control center for live trading status"
echo
echo "🎯 Your bot will start trading with 0.80+ confidence signals!"