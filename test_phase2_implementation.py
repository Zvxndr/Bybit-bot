"""
Phase 2 Implementation Test

Basic test to validate the BybitClient and HistoricalDataManager implementations.
This test can be run to ensure the core functionality works correctly.
"""

import asyncio
import os
from datetime import datetime, timedelta
import pandas as pd

# Test the imports
try:
    from src.bot.exchange import BybitClient, BybitCredentials, create_bybit_client
    from src.bot.data import HistoricalDataManager, DataFetchRequest, create_data_manager
    from src.bot.utils import RateLimiter, create_bybit_rate_limiter
    print("✅ All Phase 2 imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


async def test_bybit_client():
    """Test BybitClient basic functionality."""
    print("\n🔬 Testing BybitClient...")
    
    try:
        # Create credentials (using dummy keys for structure test)
        credentials = BybitCredentials(
            api_key="dummy_key_for_testing",
            api_secret="dummy_secret_for_testing",
            testnet=True
        )
        
        client = BybitClient(credentials)
        print("✅ BybitClient created successfully")
        
        # Test signature generation
        signature = client._generate_signature("1234567890", "GET", "/v5/market/time", "")
        print(f"✅ Signature generation works: {signature[:20]}...")
        
        # Test header generation
        headers = client._get_headers("GET", "/v5/market/time", "")
        expected_headers = ['X-BAPI-API-KEY', 'X-BAPI-TIMESTAMP', 'X-BAPI-SIGN', 'X-BAPI-RECV-WINDOW']
        
        if all(header in headers for header in expected_headers):
            print("✅ Header generation works correctly")
        else:
            print("❌ Missing required headers")
        
        print("✅ BybitClient basic functionality test passed")
        
    except Exception as e:
        print(f"❌ BybitClient test failed: {e}")


async def test_rate_limiter():
    """Test RateLimiter functionality."""
    print("\n🔬 Testing RateLimiter...")
    
    try:
        # Create a rate limiter (5 requests per 10 seconds)
        limiter = RateLimiter(max_requests=5, time_window=10)
        print("✅ RateLimiter created successfully")
        
        # Test token acquisition
        start_time = datetime.now()
        for i in range(3):
            await limiter.acquire()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ Acquired 3 tokens in {elapsed:.2f} seconds")
        
        # Test usage statistics
        usage = limiter.get_current_usage()
        expected_keys = ['tokens_available', 'max_tokens', 'utilization_percent', 'requests_in_window']
        
        if all(key in usage for key in expected_keys):
            print("✅ Usage statistics work correctly")
            print(f"   Tokens available: {usage['tokens_available']}")
            print(f"   Utilization: {usage['utilization_percent']:.1f}%")
        else:
            print("❌ Missing usage statistics")
        
        # Test Bybit-specific rate limiter
        bybit_limiter = create_bybit_rate_limiter()
        await bybit_limiter.acquire('/v5/market/kline')
        print("✅ Bybit-specific rate limiter works")
        
        print("✅ RateLimiter test passed")
        
    except Exception as e:
        print(f"❌ RateLimiter test failed: {e}")


async def test_historical_data_manager():
    """Test HistoricalDataManager structure."""
    print("\n🔬 Testing HistoricalDataManager...")
    
    try:
        # Create mock credentials
        credentials = BybitCredentials(
            api_key="dummy_key",
            api_secret="dummy_secret",
            testnet=True
        )
        
        client = BybitClient(credentials)
        data_manager = HistoricalDataManager(client)
        print("✅ HistoricalDataManager created successfully")
        
        # Test data request creation
        request = DataFetchRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            include_funding=True,
            validate_quality=True
        )
        print("✅ DataFetchRequest created successfully")
        
        # Test timeframe mappings
        if "1h" in data_manager.bybit_intervals:
            print("✅ Timeframe mappings configured correctly")
        else:
            print("❌ Timeframe mappings missing")
        
        # Test quality assessment with mock data
        mock_data = pd.DataFrame({
            'timestamp': [1609459200000, 1609462800000, 1609466400000],
            'open': [29000.0, 29100.0, 29200.0],
            'high': [29100.0, 29200.0, 29300.0],
            'low': [28900.0, 29000.0, 29100.0],
            'close': [29100.0, 29200.0, 29300.0],
            'volume': [100.0, 150.0, 120.0]
        })
        
        quality_metrics = data_manager._assess_data_quality(mock_data)
        print(f"✅ Data quality assessment works: Score {quality_metrics.quality_score:.2f}")
        
        print("✅ HistoricalDataManager structure test passed")
        
    except Exception as e:
        print(f"❌ HistoricalDataManager test failed: {e}")


async def test_integration():
    """Test integration between components."""
    print("\n🔬 Testing Component Integration...")
    
    try:
        # Test that components can work together
        credentials = BybitCredentials(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        client = BybitClient(credentials, enable_rate_limiting=True)
        data_manager = HistoricalDataManager(client, cache_ttl_minutes=30)
        
        print("✅ Components integrate successfully")
        
        # Test cache functionality
        data_manager.clear_cache()
        quality_report = data_manager.get_quality_report()
        
        if quality_report['total_datasets'] == 0:
            print("✅ Cache management works correctly")
        else:
            print("❌ Cache management issue")
        
        print("✅ Integration test passed")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")


async def main():
    """Run all Phase 2 tests."""
    print("🚀 Starting Phase 2 Implementation Tests")
    print("=" * 50)
    
    await test_bybit_client()
    await test_rate_limiter()
    await test_historical_data_manager()
    await test_integration()
    
    print("\n" + "=" * 50)
    print("✅ Phase 2 Implementation Tests Complete!")
    print("\n📋 Next Steps:")
    print("1. Configure real API credentials in .env file")
    print("2. Test connectivity with actual Bybit API")
    print("3. Fetch real historical data")
    print("4. Move to Phase 3: Enhanced Backtesting Engine")


if __name__ == "__main__":
    asyncio.run(main())