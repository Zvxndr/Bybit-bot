"""
Trading API Router for Dashboard Backend
Handles trading data endpoints and real-time trading information
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Mock trading data for development
class TradingDataService:
    """Service for trading data operations"""
    
    @staticmethod
    async def get_recent_trades(symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trading data"""
        # Mock data - in production this would connect to Phase 1 components
        trades = []
        
        symbols = [symbol] if symbol else ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
        
        for sym in symbols:
            for i in range(min(limit // len(symbols), 25)):
                trade = {
                    "id": f"trade_{sym}_{i}",
                    "symbol": sym,
                    "timestamp": (datetime.utcnow() - timedelta(minutes=i*2)).isoformat(),
                    "side": "buy" if i % 2 == 0 else "sell",
                    "price": 45000 + (i * 100) if sym == "BTCUSDT" else 3000 + (i * 10),
                    "quantity": 0.01 + (i * 0.001),
                    "value": 450 + (i * 10),
                    "pnl": (i - 10) * 2.5,
                    "strategy": ["momentum", "mean_reversion", "arbitrage"][i % 3],
                    "execution_time": 0.023 + (i * 0.001),
                    "status": "filled"
                }
                trades.append(trade)
        
        return sorted(trades, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    @staticmethod
    async def get_trading_summary() -> Dict[str, Any]:
        """Get trading performance summary"""
        return {
            "total_trades": 15247,
            "total_volume": 2847362.45,
            "total_pnl": 45782.33,
            "win_rate": 68.4,
            "avg_trade_duration": 4.2,
            "best_performing_strategy": "arbitrage",
            "active_positions": 12,
            "daily_pnl": 1247.89,
            "weekly_pnl": 8934.22,
            "monthly_pnl": 31456.78,
            "sharpe_ratio": 2.34,
            "max_drawdown": 0.0234,
            "success_rate": 94.7
        }
    
    @staticmethod
    async def get_position_data() -> List[Dict[str, Any]]:
        """Get current position data"""
        positions = [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "size": 0.245,
                "entry_price": 44850.0,
                "current_price": 45120.0,
                "pnl": 66.15,
                "pnl_percentage": 0.6,
                "margin_used": 1126.25,
                "liquidation_price": 42350.0,
                "strategy": "momentum"
            },
            {
                "symbol": "ETHUSDT",
                "side": "short",
                "size": 2.5,
                "entry_price": 3125.0,
                "current_price": 3098.0,
                "pnl": 67.5,
                "pnl_percentage": 0.86,
                "margin_used": 781.25,
                "liquidation_price": 3250.0,
                "strategy": "mean_reversion"
            },
            {
                "symbol": "SOLUSDT",
                "side": "long",
                "size": 15.0,
                "entry_price": 98.45,
                "current_price": 99.23,
                "pnl": 11.7,
                "pnl_percentage": 0.79,
                "margin_used": 369.19,
                "liquidation_price": 89.30,
                "strategy": "arbitrage"
            }
        ]
        return positions

@router.get("/trades", response_model=Dict[str, Any])
async def get_trades(
    symbol: Optional[str] = Query(None, description="Trading symbol filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of trades to return")
):
    """Get recent trading data"""
    try:
        trades = await TradingDataService.get_recent_trades(symbol, limit)
        
        return {
            "success": True,
            "data": trades,
            "count": len(trades),
            "symbol_filter": symbol,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get trades: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trading data")

@router.get("/summary", response_model=Dict[str, Any])
async def get_trading_summary():
    """Get comprehensive trading performance summary"""
    try:
        summary = await TradingDataService.get_trading_summary()
        
        return {
            "success": True,
            "data": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get trading summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trading summary")

@router.get("/positions", response_model=Dict[str, Any])
async def get_positions():
    """Get current trading positions"""
    try:
        positions = await TradingDataService.get_position_data()
        
        # Calculate portfolio summary
        total_pnl = sum(pos["pnl"] for pos in positions)
        total_margin = sum(pos["margin_used"] for pos in positions)
        
        portfolio_summary = {
            "total_positions": len(positions),
            "total_pnl": total_pnl,
            "total_margin_used": total_margin,
            "long_positions": len([p for p in positions if p["side"] == "long"]),
            "short_positions": len([p for p in positions if p["side"] == "short"]),
            "avg_pnl_percentage": sum(pos["pnl_percentage"] for pos in positions) / len(positions) if positions else 0
        }
        
        return {
            "success": True,
            "data": {
                "positions": positions,
                "portfolio_summary": portfolio_summary
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve position data")

@router.get("/symbols", response_model=Dict[str, Any])
async def get_trading_symbols():
    """Get available trading symbols and their current status"""
    try:
        symbols = [
            {
                "symbol": "BTCUSDT",
                "price": 45120.0,
                "change_24h": 2.34,
                "volume_24h": 125847.23,
                "active": True,
                "strategies": ["momentum", "arbitrage", "trend_following"]
            },
            {
                "symbol": "ETHUSDT", 
                "price": 3098.0,
                "change_24h": -1.12,
                "volume_24h": 89234.56,
                "active": True,
                "strategies": ["mean_reversion", "volatility", "ml_ensemble"]
            },
            {
                "symbol": "SOLUSDT",
                "price": 99.23,
                "change_24h": 4.67,
                "volume_24h": 45678.90,
                "active": True,
                "strategies": ["arbitrage", "momentum"]
            },
            {
                "symbol": "ADAUSDT",
                "price": 0.456,
                "change_24h": -0.89,
                "volume_24h": 23456.78,
                "active": False,
                "strategies": ["trend_following"]
            }
        ]
        
        return {
            "success": True,
            "data": symbols,
            "active_symbols": len([s for s in symbols if s["active"]]),
            "total_symbols": len(symbols),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get symbols: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve symbol data")

@router.get("/strategies", response_model=Dict[str, Any])
async def get_strategy_performance():
    """Get performance breakdown by trading strategy"""
    try:
        strategies = [
            {
                "name": "momentum",
                "trades": 3247,
                "win_rate": 72.4,
                "total_pnl": 12456.78,
                "avg_trade_pnl": 3.84,
                "sharpe_ratio": 2.67,
                "max_drawdown": 0.0123,
                "active": True,
                "allocation": 25.0
            },
            {
                "name": "mean_reversion",
                "trades": 2834,
                "win_rate": 68.9,
                "total_pnl": 9876.54,
                "avg_trade_pnl": 3.48,
                "sharpe_ratio": 2.34,
                "max_drawdown": 0.0156,
                "active": True,
                "allocation": 20.0
            },
            {
                "name": "arbitrage",
                "trades": 4523,
                "win_rate": 89.2,
                "total_pnl": 15678.90,
                "avg_trade_pnl": 3.47,
                "sharpe_ratio": 3.12,
                "max_drawdown": 0.0067,
                "active": True,
                "allocation": 15.0
            },
            {
                "name": "trend_following",
                "trades": 2156,
                "win_rate": 64.3,
                "total_pnl": 8234.56,
                "avg_trade_pnl": 3.82,
                "sharpe_ratio": 2.01,
                "max_drawdown": 0.0234,
                "active": True,
                "allocation": 25.0
            },
            {
                "name": "volatility",
                "trades": 1876,
                "win_rate": 61.7,
                "total_pnl": 5432.10,
                "avg_trade_pnl": 2.89,
                "sharpe_ratio": 1.78,
                "max_drawdown": 0.0198,
                "active": True,
                "allocation": 10.0
            },
            {
                "name": "ml_ensemble",
                "trades": 987,
                "win_rate": 84.6,
                "total_pnl": 7890.12,
                "avg_trade_pnl": 7.99,
                "sharpe_ratio": 4.23,
                "max_drawdown": 0.0089,
                "active": True,
                "allocation": 5.0
            }
        ]
        
        # Calculate totals
        totals = {
            "total_trades": sum(s["trades"] for s in strategies),
            "weighted_win_rate": sum(s["win_rate"] * s["allocation"] / 100 for s in strategies),
            "total_pnl": sum(s["total_pnl"] for s in strategies),
            "portfolio_sharpe": sum(s["sharpe_ratio"] * s["allocation"] / 100 for s in strategies),
            "active_strategies": len([s for s in strategies if s["active"]])
        }
        
        return {
            "success": True,
            "data": {
                "strategies": strategies,
                "totals": totals
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get strategy performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve strategy data")

@router.get("/performance/{symbol}", response_model=Dict[str, Any])
async def get_symbol_performance(symbol: str):
    """Get detailed performance data for specific symbol"""
    try:
        # Mock performance data for the symbol
        performance = {
            "symbol": symbol.upper(),
            "timeframes": {
                "1h": {"pnl": 123.45, "trades": 12, "win_rate": 75.0},
                "4h": {"pnl": 456.78, "trades": 34, "win_rate": 70.6},
                "1d": {"pnl": 1247.89, "trades": 87, "win_rate": 68.4},
                "1w": {"pnl": 5678.90, "trades": 234, "win_rate": 71.2},
                "1m": {"pnl": 12345.67, "trades": 756, "win_rate": 69.8}
            },
            "best_strategy": "arbitrage",
            "worst_strategy": "volatility",
            "current_position": {
                "side": "long",
                "size": 0.245,
                "pnl": 66.15,
                "duration": "2h 34m"
            },
            "risk_metrics": {
                "var_95": 0.0234,
                "max_drawdown": 0.0167,
                "volatility": 0.0456,
                "beta": 1.23
            }
        }
        
        return {
            "success": True,
            "data": performance,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get symbol performance for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data for {symbol}")

@router.get("/orderbook/{symbol}", response_model=Dict[str, Any])
async def get_orderbook(symbol: str, depth: int = Query(20, ge=1, le=100)):
    """Get order book data for symbol"""
    try:
        # Mock order book data
        base_price = 45120.0 if symbol.upper() == "BTCUSDT" else 3098.0
        
        bids = []
        asks = []
        
        # Generate mock bid/ask data
        for i in range(depth):
            bid_price = base_price - (i + 1) * 0.5
            ask_price = base_price + (i + 1) * 0.5
            
            bids.append([bid_price, 0.1 + i * 0.02])
            asks.append([ask_price, 0.1 + i * 0.02])
        
        orderbook = {
            "symbol": symbol.upper(),
            "timestamp": datetime.utcnow().isoformat(),
            "bids": bids,
            "asks": asks,
            "spread": asks[0][0] - bids[0][0],
            "spread_percentage": ((asks[0][0] - bids[0][0]) / bids[0][0]) * 100
        }
        
        return {
            "success": True,
            "data": orderbook,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get orderbook for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve orderbook for {symbol}")