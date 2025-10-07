"""
Unified API System Usage Examples

This module demonstrates how to use the consolidated API system for
various trading operations, market data access, and system monitoring.

Examples included:
1. Basic system initialization
2. Market data streaming
3. Trading operations
4. Real-time monitoring
5. Configuration management
6. Error handling and recovery
7. Australian compliance integration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

# Import unified API system
from .unified_bybit_client import OrderSide, OrderType, TimeInForce
from .config import (
    UnifiedAPIConfig, Environment, create_testnet_config, 
    create_production_config, LogLevel
)
from . import (
    UnifiedAPISystem, create_api_system, create_api_system_from_config,
    APISystemContext
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# EXAMPLE 1: BASIC SYSTEM INITIALIZATION
# ============================================================================

async def example_basic_initialization():
    """Example: Basic system initialization and health check"""
    print("\n=== Example 1: Basic System Initialization ===")
    
    # Create testnet configuration (no API keys required for market data)
    config = create_testnet_config()
    config.enable_trading = False  # Market data only
    config.enable_websockets = True
    config.logging.level = LogLevel.INFO
    
    # Symbols to monitor
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    
    try:
        # Create and initialize system
        async with APISystemContext(UnifiedAPISystem(config)) as system:
            await system.initialize(symbols)
            
            # Get system status
            status = system.get_status()
            print(f"System Status: {status['system']['is_healthy']}")
            print(f"Uptime: {status['system']['uptime_readable']}")
            print(f"Components: {status['components']}")
            
            # Keep running for demo
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"Initialization failed: {e}")

# ============================================================================
# EXAMPLE 2: MARKET DATA STREAMING
# ============================================================================

async def example_market_data_streaming():
    """Example: Real-time market data streaming with callbacks"""
    print("\n=== Example 2: Market Data Streaming ===")
    
    config = create_testnet_config()
    config.enable_websockets = True
    config.cache.enable_caching = True
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    # Market data callback
    async def on_ticker_update(update_data):
        data = update_data['data']
        print(f"📊 {data.symbol}: ${data.price} (Bid: ${data.bid}, Ask: ${data.ask})")
    
    # Trade callback
    async def on_trade_update(update_data):
        trade = update_data['data']
        print(f"💹 Trade: {trade.symbol} {trade.side} {trade.quantity} @ ${trade.price}")
    
    try:
        system = UnifiedAPISystem(config)
        await system.initialize(symbols)
        
        # Add callbacks for market data
        pipeline = system.get_market_data_pipeline()
        if pipeline:
            pipeline.add_callback('ticker', on_ticker_update)
            pipeline.add_callback('trade', on_trade_update)
        
        print("📡 Streaming market data... (press Ctrl+C to stop)")
        
        # Stream for 30 seconds
        await asyncio.sleep(30)
        
        # Get some cached data
        for symbol in symbols:
            market_data = await pipeline.get_market_data(symbol)
            if market_data:
                print(f"💰 {symbol} cached: ${market_data.price} (24h: {market_data.change_24h}%)")
        
        await system.shutdown()
        
    except KeyboardInterrupt:
        print("🛑 Streaming stopped by user")
    except Exception as e:
        logger.error(f"Streaming error: {e}")

# ============================================================================
# EXAMPLE 3: TRADING OPERATIONS (WITH API KEYS)
# ============================================================================

async def example_trading_operations():
    """Example: Trading operations (requires API keys)"""
    print("\n=== Example 3: Trading Operations ===")
    
    # NOTE: This example requires real API keys
    api_key = "your_api_key_here"
    api_secret = "your_api_secret_here"
    
    if api_key == "your_api_key_here":
        print("⚠️  Please provide real API keys to test trading operations")
        return
    
    try:
        # Create system with trading enabled
        system = await create_api_system(
            api_key=api_key,
            api_secret=api_secret,
            environment=Environment.TESTNET,
            symbols=['BTCUSDT'],
            enable_trading=True,
            enable_risk_management=True
        )
        
        rest_client = system.get_rest_client()
        if not rest_client:
            raise RuntimeError("REST client not available")
        
        # Get account balance
        balances = await rest_client.get_account_balance()
        for balance in balances:
            if balance.available_balance > 0:
                print(f"💳 {balance.coin}: {balance.available_balance} available")
        
        # Get current market data
        market_data = await rest_client.get_market_data('BTCUSDT')
        print(f"📊 BTCUSDT: ${market_data.price}")
        
        # Place a small test order (be careful!)
        if False:  # Set to True to actually place orders
            order = await rest_client.place_order(
                symbol='BTCUSDT',
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                qty=Decimal('0.001'),  # Very small amount
                price=market_data.price * Decimal('0.95'),  # 5% below market
                time_in_force=TimeInForce.GTC
            )
            print(f"📋 Order placed: {order.order_id}")
            
            # Check open orders
            open_orders = await rest_client.get_open_orders('BTCUSDT')
            print(f"📝 Open orders: {len(open_orders)}")
            
            # Cancel the order
            await rest_client.cancel_order('BTCUSDT', order_id=order.order_id)
            print(f"❌ Order cancelled: {order.order_id}")
        
        await system.shutdown()
        
    except Exception as e:
        logger.error(f"Trading example error: {e}")

# ============================================================================
# EXAMPLE 4: REAL-TIME MONITORING
# ============================================================================

async def example_real_time_monitoring():
    """Example: Real-time system monitoring with health checks"""
    print("\n=== Example 4: Real-Time Monitoring ===")
    
    config = create_testnet_config()
    config.enable_websockets = True
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Health check callback
    async def on_health_check(health_data):
        timestamp = health_data['timestamp']
        components = health_data['components']
        print(f"🔍 Health Check {timestamp}:")
        for component, status in components.items():
            if isinstance(status, dict):
                overall_status = status.get('overall_status', status.get('status', 'unknown'))
                print(f"  {component}: {overall_status}")
    
    # Error callback
    async def on_error(error_data):
        component = error_data.get('component', 'unknown')
        error = error_data.get('error', 'unknown error')
        print(f"❌ Error in {component}: {error}")
    
    try:
        system = UnifiedAPISystem(config)
        
        # Add event callbacks
        system.add_event_callback('health_check', on_health_check)
        system.add_event_callback('error_occurred', on_error)
        
        await system.initialize(symbols)
        
        print("🔍 Monitoring system health...")
        
        # Monitor for 60 seconds
        for i in range(12):  # 12 x 5 seconds = 60 seconds
            await asyncio.sleep(5)
            
            status = system.get_status()
            print(f"📊 System Health: {status['system']['is_healthy']} | "
                  f"Uptime: {status['system']['uptime_readable']} | "
                  f"Success Rate: {status['performance']['success_rate_pct']:.1f}%")
        
        await system.shutdown()
        
    except Exception as e:
        logger.error(f"Monitoring error: {e}")

# ============================================================================
# EXAMPLE 5: CONFIGURATION MANAGEMENT
# ============================================================================

async def example_configuration_management():
    """Example: Advanced configuration management"""
    print("\n=== Example 5: Configuration Management ===")
    
    # Create custom configuration
    config = UnifiedAPIConfig()
    
    # Basic settings
    config.environment = Environment.TESTNET
    config.enable_websockets = True
    config.enable_market_data = True
    config.enable_trading = False
    
    # Rate limiting
    config.rate_limits.market_data = 60  # Reduced for testing
    config.rate_limits.enable_rate_limiting = True
    
    # Caching
    config.cache.enable_caching = True
    config.cache.cache_size = 5000
    config.cache.cache_ttl_seconds = 30
    
    # Australian compliance
    config.australian_compliance.timezone = "Australia/Sydney"
    config.australian_compliance.enable_tax_reporting = True
    config.australian_compliance.reporting_currency = "AUD"
    
    # Logging
    config.logging.level = LogLevel.INFO
    config.logging.enable_console_logging = True
    
    # Performance
    config.performance.enable_batching = True
    config.performance.batch_size = 50
    
    print("⚙️  Configuration Summary:")
    summary = config.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save configuration to file
    config.save_to_file("example_config.json")
    print("💾 Configuration saved to example_config.json")
    
    # Load and use configuration
    try:
        system = await create_api_system_from_config(
            config=config,
            symbols=['BTCUSDT']
        )
        
        print("✅ System created from custom configuration")
        
        # Run briefly
        await asyncio.sleep(5)
        await system.shutdown()
        
    except Exception as e:
        logger.error(f"Configuration example error: {e}")

# ============================================================================
# EXAMPLE 6: ERROR HANDLING AND RECOVERY
# ============================================================================

async def example_error_handling():
    """Example: Error handling and recovery scenarios"""
    print("\n=== Example 6: Error Handling and Recovery ===")
    
    config = create_testnet_config()
    config.connection.max_retries = 2
    config.websocket.max_reconnect_attempts = 3
    
    # Error tracking
    error_count = 0
    reconnect_count = 0
    
    async def on_error(error_data):
        nonlocal error_count
        error_count += 1
        print(f"❌ Error #{error_count}: {error_data}")
    
    async def on_reconnect(reconnect_data):
        nonlocal reconnect_count
        reconnect_count += 1
        print(f"🔄 Reconnection #{reconnect_count}: {reconnect_data}")
    
    try:
        system = UnifiedAPISystem(config)
        system.add_event_callback('error_occurred', on_error)
        system.add_event_callback('component_reconnected', on_reconnect)
        
        await system.initialize(['BTCUSDT'])
        
        print("🧪 Testing error scenarios...")
        
        # Simulate running with potential connection issues
        for i in range(10):
            await asyncio.sleep(2)
            status = system.get_status()
            
            if not status['system']['is_healthy']:
                print(f"⚠️  System unhealthy at iteration {i+1}")
            else:
                print(f"✅ System healthy at iteration {i+1}")
        
        print(f"📊 Final stats - Errors: {error_count}, Reconnects: {reconnect_count}")
        
        await system.shutdown()
        
    except Exception as e:
        logger.error(f"Error handling example error: {e}")

# ============================================================================
# EXAMPLE 7: HISTORICAL DATA AND ANALYSIS
# ============================================================================

async def example_historical_data():
    """Example: Historical data retrieval and analysis"""
    print("\n=== Example 7: Historical Data and Analysis ===")
    
    config = create_testnet_config()
    
    try:
        system = UnifiedAPISystem(config)
        await system.initialize(['BTCUSDT'])
        
        pipeline = system.get_market_data_pipeline()
        if not pipeline:
            raise RuntimeError("Market data pipeline not available")
        
        # Get historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        print(f"📈 Fetching 24h historical data for BTCUSDT...")
        df = await pipeline.get_historical_data(
            symbol='BTCUSDT',
            interval='1h',  # 1 hour intervals
            start_time=start_time,
            end_time=end_time
        )
        
        if not df.empty:
            print(f"📊 Retrieved {len(df)} data points")
            print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            print(f"Volume: {df['volume'].sum():.2f} BTC")
            
            # Simple analysis
            current_price = df['close'].iloc[-1]
            high_24h = df['high'].max()
            low_24h = df['low'].min()
            change_24h = ((current_price - df['open'].iloc[0]) / df['open'].iloc[0]) * 100
            
            print(f"\n📊 24h Summary:")
            print(f"  Current: ${current_price:.2f}")
            print(f"  High: ${high_24h:.2f}")
            print(f"  Low: ${low_24h:.2f}")
            print(f"  Change: {change_24h:.2f}%")
        
        await system.shutdown()
        
    except Exception as e:
        logger.error(f"Historical data example error: {e}")

# ============================================================================
# MAIN EXAMPLE RUNNER
# ============================================================================

async def run_all_examples():
    """Run all examples in sequence"""
    print("\n🚀 Running Unified API System Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Initialization", example_basic_initialization),
        ("Market Data Streaming", example_market_data_streaming),
        ("Configuration Management", example_configuration_management),
        ("Real-time Monitoring", example_real_time_monitoring),
        ("Error Handling", example_error_handling),
        ("Historical Data", example_historical_data),
        # Note: Trading example requires API keys
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n🏃 Running: {name}")
            await example_func()
            print(f"✅ Completed: {name}")
        except Exception as e:
            print(f"❌ Failed: {name} - {e}")
            logger.error(f"Example '{name}' failed", exc_info=True)
        
        # Brief pause between examples
        await asyncio.sleep(2)
    
    print("\n🎉 All examples completed!")

if __name__ == "__main__":
    # Run examples
    asyncio.run(run_all_examples())