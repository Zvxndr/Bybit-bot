#!/usr/bin/env python3
"""Test Environment Configuration Loading"""

import os

def test_environment_credentials():
    """Test the environment credential loading logic"""
    
    def _load_environment_credentials():
        """Load API credentials based on environment and trading mode with fallbacks"""
        environment = os.getenv('ENVIRONMENT', 'development')
        trading_mode = os.getenv('TRADING_MODE', 'paper')
        
        print(f"🔍 Testing Configuration:")
        print(f"   ENVIRONMENT: {environment}")
        print(f"   TRADING_MODE: {trading_mode}")
        
        # Environment-specific credential loading
        if environment == 'production' and trading_mode == 'live':
            # Production live trading - use live API keys
            api_key = os.getenv('BYBIT_LIVE_API_KEY')
            api_secret = os.getenv('BYBIT_LIVE_API_SECRET')
            testnet = False
            print(f"🔴 LIVE TRADING MODE: Using production API keys")
            
        elif environment in ['development', 'staging'] or trading_mode == 'paper':
            # Development/staging or paper trading - use testnet keys
            api_key = os.getenv('BYBIT_TESTNET_API_KEY')
            api_secret = os.getenv('BYBIT_TESTNET_API_SECRET')
            testnet = True
            print(f"🟡 PAPER/TEST MODE: Using testnet API keys")
            
        else:
            # Fallback to legacy single API key (backward compatibility)
            api_key = os.getenv('BYBIT_API_KEY')
            api_secret = os.getenv('BYBIT_API_SECRET')
            testnet = (trading_mode != 'live')  # Default to testnet unless explicitly live
            print(f"⚠️ LEGACY MODE: Using single API key (testnet: {testnet})")
        
        # Validation
        if not api_key or not api_secret:
            print(f"❌ Missing API credentials for {environment}/{trading_mode}")
            print(f"   Expected: BYBIT_{'LIVE' if not testnet else 'TESTNET'}_API_KEY/SECRET")
        else:
            print(f"✅ API credentials loaded: {api_key[:8]}... (testnet: {testnet})")
            
        return api_key, api_secret, testnet
    
    # Test scenarios
    test_cases = [
        {'ENVIRONMENT': 'development', 'TRADING_MODE': 'paper'},
        {'ENVIRONMENT': 'production', 'TRADING_MODE': 'paper'},
        {'ENVIRONMENT': 'production', 'TRADING_MODE': 'live'},
        {'ENVIRONMENT': 'staging', 'TRADING_MODE': 'live'},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        for key, value in case.items():
            os.environ[key] = value
        
        try:
            api_key, api_secret, testnet = _load_environment_credentials()
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_environment_credentials()