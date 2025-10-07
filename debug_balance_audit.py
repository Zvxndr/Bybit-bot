#!/usr/bin/env python3

import sqlite3
import os

print("=== DATABASE BALANCE AUDIT ===")

# Check if database exists
if os.path.exists('data/trading_bot.db'):
    print("✅ Database found")
    
    conn = sqlite3.connect('data/trading_bot.db')
    cursor = conn.cursor()
    
    # Check all strategies
    cursor.execute("SELECT strategy_id, current_phase, paper_pnl, live_pnl FROM strategy_pipeline LIMIT 10")
    strategies = cursor.fetchall()
    print(f"\n📊 Found {len(strategies)} strategies:")
    for strategy in strategies:
        print(f"  {strategy[0]} | Phase: {strategy[1]} | Paper PnL: ${strategy[2]} | Live PnL: ${strategy[3]}")
    
    # Check paper PnL sum
    cursor.execute("SELECT SUM(paper_pnl) FROM strategy_pipeline WHERE current_phase = 'paper'")
    total_paper_pnl = cursor.fetchone()[0] or 0
    print(f"\n💰 Total Paper PnL: ${total_paper_pnl}")
    
    # Check environment variables
    paper_balance = float(os.getenv('PAPER_TRADING_BALANCE', '10000'))
    print(f"📋 Paper Trading Base Balance: ${paper_balance}")
    
    # Calculate what the system would show
    current_balance = paper_balance + total_paper_pnl
    print(f"🧮 Calculated Total Balance: ${current_balance}")
    print(f"🧮 Available (85%): ${current_balance * 0.85}")
    print(f"🧮 Used (15%): ${current_balance * 0.15}")
    
    conn.close()
else:
    print("❌ No database found at data/trading_bot.db")

print("\n=== EXPECTED VALUES ===")
print("You reported seeing:")
print("  Total Balance: $10,279.55")
print("  Available: $8,737.62") 
print("  Used: $1,541.93")
print("  Unrealized PnL: $35.27")