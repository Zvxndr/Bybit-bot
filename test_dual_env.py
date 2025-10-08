#!/usr/bin/env python3
"""Test Dual Environment Architecture"""

import os
import sys

# Set test environment
os.environ['ENVIRONMENT'] = 'development'
os.environ['TRADING_MODE'] = 'paper'

# Add src to path
sys.path.append('src')

try:
    from main import TradingAPI
    
    print('=== Testing Dual Environment Setup ===')
    api = TradingAPI()
    print(f'Testnet enabled: {api.enable_testnet}')
    print(f'Live enabled: {api.enable_live}')
    print(f'Testnet valid: {api.testnet_credentials["valid"]}')  
    print(f'Live valid: {api.live_credentials["valid"]}')
    print('=== Dual Environment Test Complete ===')
    
except Exception as e:
    print(f"Error testing dual environment: {e}")
    import traceback
    traceback.print_exc()