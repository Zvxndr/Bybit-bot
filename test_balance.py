import asyncio
import sys
sys.path.append('src')
from src.main import TradingAPI

async def test_balance():
    try:
        api = TradingAPI()
        portfolio = await api.get_portfolio()
        print('=== CURRENT PORTFOLIO SYSTEM AUDIT ===')
        print(f'Environment: {portfolio.get("environment")}')
        print(f'Total Balance: ${portfolio.get("total_balance")}')
        print(f'Available: ${portfolio.get("available_balance")}')
        print(f'Used: ${portfolio.get("used_balance")}')
        print(f'Unrealized PnL: ${portfolio.get("unrealized_pnl")}')
        print(f'Message: {portfolio.get("message")}')
        print('=====================================')
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_balance())