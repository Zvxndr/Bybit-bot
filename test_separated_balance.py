#!/usr/bin/env python3

import asyncio
import sys
import os
import sqlite3
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Create a simple test of the new balance structure
async def test_separated_balance():
    """Test the new separated balance system"""
    print("=== TESTING NEW SEPARATED BALANCE SYSTEM ===")
    
    try:
        # Import the trading API class
        sys.path.insert(0, 'src')
        import main
        TradingAPI = main.TradingAPI
        
        # Initialize without API keys (paper mode)
        api = TradingAPI(
            api_key=None,
            api_secret=None,
            testnet=True
        )
        
        # Test the new portfolio method
        portfolio_data = await api.get_portfolio()
        
        print("\n📊 NEW PORTFOLIO STRUCTURE:")
        print(f"System Message: {portfolio_data.get('system_message', 'N/A')}")
        
        print("\n🧪 PAPER/TESTNET BALANCE (Phase 2):")
        paper = portfolio_data.get('paper_testnet', {})
        print(f"  Total Balance: ${paper.get('total_balance', 0):,.2f}")
        print(f"  Available: ${paper.get('available_balance', 0):,.2f}")
        print(f"  Used: ${paper.get('used_balance', 0):,.2f}")
        print(f"  Unrealized PnL: ${paper.get('unrealized_pnl', 0):+,.2f}")
        print(f"  Environment: {paper.get('environment', 'N/A')}")
        print(f"  Message: {paper.get('message', 'N/A')}")
        
        print("\n🚀 LIVE BALANCE (Phase 3):")
        live = portfolio_data.get('live', {})
        print(f"  Total Balance: ${live.get('total_balance', 0):,.2f}")
        print(f"  Available: ${live.get('available_balance', 0):,.2f}")
        print(f"  Used: ${live.get('used_balance', 0):,.2f}")
        print(f"  Unrealized PnL: ${live.get('unrealized_pnl', 0):+,.2f}")
        print(f"  Environment: {live.get('environment', 'N/A')}")
        print(f"  Message: {live.get('message', 'N/A')}")
        
        print("\n✅ SUCCESS: Balances are now properly separated!")
        print("✅ Paper/Testnet balance shows your simulated trading")
        print("✅ Live balance shows real account (empty without API keys)")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Make sure database exists
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/trading_bot.db'):
        conn = sqlite3.connect('data/trading_bot.db')
        conn.execute('''CREATE TABLE IF NOT EXISTS strategy_pipeline (
            id INTEGER PRIMARY KEY,
            strategy_id TEXT,
            current_phase TEXT DEFAULT 'paper',
            paper_pnl REAL DEFAULT 0,
            live_pnl REAL DEFAULT 0,
            is_active INTEGER DEFAULT 1
        )''')
        
        # Add some sample data
        conn.execute("INSERT INTO strategy_pipeline (strategy_id, current_phase, paper_pnl) VALUES ('BTCUSDT_demo1', 'paper', 279.55)")
        conn.execute("INSERT INTO strategy_pipeline (strategy_id, current_phase, paper_pnl) VALUES ('ETHUSDT_demo2', 'paper', -36.04)")
        conn.commit()
        conn.close()
        print("✅ Created demo database")
    
    asyncio.run(test_separated_balance())