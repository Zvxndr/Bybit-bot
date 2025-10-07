import asyncio
import sys
sys.path.append('src')
from src.services.balance_manager import balance_manager

async def test_multi_balance():
    try:
        balances = await balance_manager.get_all_balances()
        print('=== MULTI-ENVIRONMENT BALANCE SYSTEM AUDIT ===')
        
        for env, data in balances.items():
            print(f'\n{env.upper()} ENVIRONMENT:')
            print(f'  Total: ${data.get("total", 0)}')
            print(f'  Available: ${data.get("available", 0)}')
            print(f'  Status: {data.get("status", "unknown")}')
            if 'message' in data:
                print(f'  Message: {data["message"]}')
        
        print('\n=== PAPER BALANCE DETAILED ===')
        paper_balance = balance_manager._get_paper_balance()
        print(f'Initial Balance: ${paper_balance.get("initial_balance", 0)}')
        print(f'Current Balance: ${paper_balance.get("total", 0)}')
        print(f'Total PnL: ${paper_balance.get("total_pnl", 0)}')
        print(f'Total Trades: {paper_balance.get("total_trades", 0)}')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_multi_balance())