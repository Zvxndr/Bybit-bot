"""
Test Historical Data Integration
===============================

Tests the historical data provider and debug safety integration.
"""

import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_historical_data():
    """Test historical data integration"""
    print("=" * 60)
    print("🧪 Testing Historical Data Integration")
    print("=" * 60)
    
    try:
        # Test 1: Historical Data Provider
        print("\n📊 Test 1: Historical Data Provider")
        from src.historical_data_provider import get_historical_data_provider
        
        provider = get_historical_data_provider()
        
        # Test realistic balances
        balances = provider.get_realistic_balances()
        print(f"✅ Historical Balances: {balances}")
        
        # Test historical positions  
        positions = provider.get_historical_positions(limit=5)
        print(f"✅ Historical Positions ({len(positions)}): {positions[:2] if positions else 'None'}")
        
        # Test historical trades
        trades = provider.get_historical_trades(limit=5) 
        print(f"✅ Historical Trades ({len(trades)}): {trades[:2] if trades else 'None'}")
        
        # Test market data
        market_data = provider.get_market_data_sample()
        print(f"✅ Market Data: {market_data}")
        
    except Exception as e:
        print(f"❌ Historical Data Provider Error: {e}")
        logger.error(f"Historical data provider test failed: {e}")
    
    try:
        # Test 2: Debug Safety Integration
        print("\n🛡️ Test 2: Debug Safety Integration")
        from src.debug_safety import get_debug_manager
        
        debug_manager = get_debug_manager()
        
        # Test debug data retrieval
        debug_balances = debug_manager.get_mock_data('balances')
        print(f"✅ Debug Balances: {debug_balances}")
        
        debug_positions = debug_manager.get_mock_data('positions')
        print(f"✅ Debug Positions: {len(debug_positions) if debug_positions else 0} positions")
        
        debug_trades = debug_manager.get_mock_data('trades')
        print(f"✅ Debug Trades: {len(debug_trades) if debug_trades else 0} trades")
        
        # Test safety features
        print(f"✅ Debug Mode Active: {debug_manager.is_debug_mode()}")
        print(f"✅ Trading Blocked: {debug_manager.block_trading_operation('place_order')}")
        
    except Exception as e:
        print(f"❌ Debug Safety Error: {e}")
        logger.error(f"Debug safety test failed: {e}")
    
    try:
        # Test 3: Bybit API Integration
        print("\n📡 Test 3: Bybit API Integration")
        from src.bybit_api import BybitAPIClient
        
        # Create client in debug mode
        client = BybitAPIClient(api_key=None, api_secret=None, testnet=True)
        
        # Test balance retrieval (should use historical data in debug mode)
        balance_result = await client.get_account_balance()
        print(f"✅ API Balance (Debug): {balance_result.get('success', False)}")
        if balance_result.get('data'):
            print(f"   Total Wallet: {balance_result['data'].get('total_wallet_balance')}")
            print(f"   Coins: {len(balance_result['data'].get('coins', []))} coins")
        
        # Test positions retrieval
        positions_result = await client.get_positions()
        print(f"✅ API Positions (Debug): {positions_result.get('success', False)}")
        if positions_result.get('data'):
            print(f"   Positions: {len(positions_result['data'].get('positions', []))} positions")
        
        # Test trade history
        trades_result = await client.get_trade_history(limit=5)
        print(f"✅ API Trades (Debug): {trades_result.get('success', False)}")
        if trades_result.get('data'):
            print(f"   Trades: {len(trades_result['data'].get('trades', []))} trades")
        
        await client.close()
        
    except Exception as e:
        print(f"❌ Bybit API Error: {e}")
        logger.error(f"Bybit API test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Historical Data Integration Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_historical_data())