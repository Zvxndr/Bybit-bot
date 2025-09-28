#!/usr/bin/env python3
"""
Speed Demon Logic Test
=====================

Test script to validate the Speed Demon virtual trading logic 
vs real testnet order execution separation.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from shared_state import shared_state

class MockTradingBot:
    """Mock trading bot for testing Speed Demon logic"""
    
    def __init__(self):
        pass
    
    async def _execute_virtual_paper_trade(self, signal, symbol, action, confidence):
        """Execute virtual paper trade for Speed Demon backtesting"""
        try:
            # Generate virtual order ID
            import uuid
            virtual_order_id = f"PAPER-{str(uuid.uuid4())[:8]}"
            
            # Calculate virtual order size
            order_qty = "0.001" if symbol == "BTCUSDT" else "0.01"
            
            # Simulate trade execution with virtual prices
            print(f"✅ VIRTUAL PAPER TRADE: {action.upper()} {order_qty} {symbol} (Virtual ID: {virtual_order_id})")
            shared_state.add_log_entry("SUCCESS", f"Paper trade: {action.upper()} {symbol} (Speed Demon)")
            
            # Add virtual position to shared state
            position = {
                "symbol": symbol,
                "side": action.upper(),
                "size": str(order_qty),
                "entry_price": "VIRTUAL",  # Speed Demon uses historical data
                "mark_price": "VIRTUAL",
                "pnl": "+0.00",
                "order_id": virtual_order_id,
                "timestamp": datetime.now().isoformat(),
                "mode": "SPEED_DEMON_BACKTEST"
            }
            
            # Get current positions and add new virtual position
            current_positions = shared_state._state.get("positions", [])
            current_positions.append(position)
            shared_state.update_positions(current_positions)
            
            return {"success": True, "virtual_id": virtual_order_id}
            
        except Exception as e:
            print(f"❌ Virtual paper trade error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _execute_testnet_order(self, signal, symbol, action, confidence):
        """Execute real testnet order for live testing (MOCK)"""
        try:
            # Calculate order size (small testnet amounts)
            order_qty = "0.001" if symbol == "BTCUSDT" else "0.01"
            
            # Mock testnet order (would normally use real API)
            mock_order_id = f"TESTNET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            print(f"✅ MOCK TESTNET ORDER: {action.upper()} {order_qty} {symbol} (Order ID: {mock_order_id})")
            shared_state.add_log_entry("SUCCESS", f"Mock testnet order: {action.upper()} {symbol}")
            
            # Add position to shared state
            position = {
                "symbol": symbol,
                "side": action.upper(),
                "size": str(order_qty),
                "entry_price": "45000.00" if symbol == "BTCUSDT" else "2500.00",
                "mark_price": "45050.00" if symbol == "BTCUSDT" else "2505.00",
                "pnl": "+50.00",
                "order_id": mock_order_id,
                "timestamp": datetime.now().isoformat(),
                "mode": "TESTNET_LIVE"
            }
            
            # Get current positions and add new one
            current_positions = shared_state._state.get("positions", [])
            current_positions.append(position)
            shared_state.update_positions(current_positions)
            
            return {"success": True, "order_id": mock_order_id}
                    
        except Exception as order_error:
            print(f"❌ Mock order placement error: {str(order_error)}")
            return {"success": False, "error": str(order_error)}

async def test_speed_demon_backtesting_lifecycle():
    """Test the complete Speed Demon backtesting lifecycle and phase transitions"""
    print("🧪 Testing Speed Demon Backtesting Lifecycle")
    print("=" * 50)
    
    # Initialize mock bot
    bot = MockTradingBot()
    
    # Mock trading signal
    signal = {
        "action": "buy", 
        "symbol": "BTCUSDT",
        "confidence": 0.85,
        "timestamp": datetime.now().isoformat()
    }
    
    # Test Scenario 1: Speed Demon in 'ready' state - Should use virtual trading
    print("\n🚀 PHASE 1: Speed Demon Ready - Historical Backtesting Phase")
    shared_state.speed_demon_status = {
        "mode": "speed_demon",
        "status": "ready",
        "initialization_complete": True
    }
    
    # Simulate the main app logic
    speed_demon_status = getattr(shared_state, 'speed_demon_status', {})
    is_speed_demon_mode = speed_demon_status.get('mode') == 'speed_demon'  
    speed_demon_phase = speed_demon_status.get('status', 'unknown')
    
    print(f"   Mode: {speed_demon_status.get('mode')}")
    print(f"   Phase: {speed_demon_phase}")
    
    if is_speed_demon_mode:
        if speed_demon_phase in ['ready', 'backtesting_active']:
            print("   Action: Virtual paper trading (historical backtesting)")
            result = await bot._execute_virtual_paper_trade(signal, "BTCUSDT", "buy", 0.85)
            print(f"   Result: {result}")
        elif speed_demon_phase == 'backtesting_complete':
            print("   Action: Testnet trading (backtesting complete)")
            result = await bot._execute_testnet_order(signal, "BTCUSDT", "buy", 0.85)
            print(f"   Result: {result}")
    
    # Test Scenario 2: Speed Demon backtesting complete - Should use testnet
    print("\n✅ PHASE 2: Speed Demon Backtesting Complete - Testnet Phase")  
    shared_state.speed_demon_status = {
        "mode": "speed_demon",
        "status": "backtesting_complete",
        "backtest_completed_at": datetime.now().isoformat()
    }
    
    speed_demon_status = getattr(shared_state, 'speed_demon_status', {})
    is_speed_demon_mode = speed_demon_status.get('mode') == 'speed_demon'
    speed_demon_phase = speed_demon_status.get('status', 'unknown')
    
    print(f"   Mode: {speed_demon_status.get('mode')}")
    print(f"   Phase: {speed_demon_phase}")
    
    if is_speed_demon_mode:
        if speed_demon_phase in ['ready', 'backtesting_active']:
            print("   Action: Virtual paper trading (historical backtesting)")
            result = await bot._execute_virtual_paper_trade(signal, "BTCUSDT", "sell", 0.78)
            print(f"   Result: {result}")
        elif speed_demon_phase == 'backtesting_complete':
            print("   Action: Testnet trading (backtesting complete)")
            result = await bot._execute_testnet_order(signal, "BTCUSDT", "sell", 0.78)
            print(f"   Result: {result}")
    
    # Test Scenario 3: Standard mode - Should go straight to testnet
    print("\n📈 PHASE 3: Standard Mode - Direct Testnet Trading")
    shared_state.speed_demon_status = {
        "mode": "standard",
        "status": "active"
    }
    
    speed_demon_status = getattr(shared_state, 'speed_demon_status', {})
    is_speed_demon_mode = speed_demon_status.get('mode') == 'speed_demon'
    speed_demon_phase = speed_demon_status.get('status', 'unknown')
    
    print(f"   Mode: {speed_demon_status.get('mode')}")
    print(f"   Phase: {speed_demon_phase}")
    print(f"   Is Speed Demon Mode: {is_speed_demon_mode}")
    
    if is_speed_demon_mode:
        print("   Action: Speed Demon routing")
    else:
        print("   Action: Standard testnet trading")
        result = await bot._execute_testnet_order(signal, "ETHUSDT", "buy", 0.82)
        print(f"   Result: {result}")
    
    # Show final positions summary
    print("\n📊 Final Position Summary:")
    positions = shared_state._state.get("positions", [])
    for i, pos in enumerate(positions, 1):
        print(f"   {i}. {pos['mode']}: {pos['side']} {pos['size']} {pos['symbol']} | ID: {pos['order_id']}")
    
    print(f"\n✅ Total positions: {len(positions)}")
    print("🧪 Speed Demon lifecycle test completed!")
    print("\n🔥 Key Insight: Speed Demon completes historical backtesting BEFORE any testnet trades!")

if __name__ == "__main__":
    asyncio.run(test_speed_demon_backtesting_lifecycle())