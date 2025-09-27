"""
Bybit API Integration
====================

Real-time integration with Bybit API for fetching account balance, positions, 
and trading data. Supports both testnet and mainnet environments.
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import json
from typing import Dict, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BybitAPIClient:
    """Async Bybit API client for real-time data fetching"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # API endpoints
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
            
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, timestamp: str, params: str) -> str:
        """Generate API signature for authenticated requests"""
        if not self.api_secret:
            return ""
            
        param_str = f"{timestamp}{self.api_key}{params}"
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_headers(self, params: str = "") -> Dict[str, str]:
        """Get headers for API requests"""
        timestamp = str(int(time.time() * 1000))
        
        headers = {
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": self.api_key or "",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000"
        }
        
        if self.api_key and self.api_secret:
            signature = self._generate_signature(timestamp, params)
            headers["X-BAPI-SIGN"] = signature
            
        return headers
    
    async def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance for all coins"""
        try:
            if not self.api_key:
                return {
                    "success": False,
                    "message": "API credentials not configured",
                    "data": {
                        "total_wallet_balance": "0.00",
                        "total_available_balance": "0.00",
                        "total_used_margin": "0.00",
                        "coins": []
                    }
                }
            
            endpoint = "/v5/account/wallet-balance"
            params = "accountType=UNIFIED"
            url = f"{self.base_url}{endpoint}?{params}"
            
            headers = self._get_headers(params)
            
            async with self.session.get(url, headers=headers) as response:
                data = await response.json()
                
                if response.status == 200 and data.get("retCode") == 0:
                    wallet_data = data["result"]["list"][0] if data["result"]["list"] else {}
                    
                    # Process balance data
                    total_wallet = float(wallet_data.get("totalWalletBalance", "0"))
                    total_available = float(wallet_data.get("totalAvailableBalance", "0"))
                    total_used = float(wallet_data.get("totalMarginBalance", "0")) - total_available
                    
                    # Get individual coin balances
                    coins = []
                    for coin_data in wallet_data.get("coin", []):
                        wallet_balance = float(coin_data.get("walletBalance", "0"))
                        if wallet_balance > 0:
                            coins.append({
                                "coin": coin_data.get("coin"),
                                "wallet_balance": wallet_balance,
                                "available_balance": float(coin_data.get("availableToWithdraw", "0")),
                                "used_margin": float(coin_data.get("totalOrderIM", "0"))
                            })
                    
                    return {
                        "success": True,
                        "data": {
                            "total_wallet_balance": f"{total_wallet:.2f}",
                            "total_available_balance": f"{total_available:.2f}",
                            "total_used_margin": f"{total_used:.2f}",
                            "coins": coins
                        }
                    }
                else:
                    logger.error(f"Bybit API error: {data}")
                    return {
                        "success": False,
                        "message": f"API Error: {data.get('retMsg', 'Unknown error')}",
                        "data": {
                            "total_wallet_balance": "Error",
                            "total_available_balance": "Error",
                            "total_used_margin": "Error",
                            "coins": []
                        }
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching balance: {str(e)}")
            return {
                "success": False,
                "message": f"Connection error: {str(e)}",
                "data": {
                    "total_wallet_balance": "Connection Error",
                    "total_available_balance": "Connection Error", 
                    "total_used_margin": "Connection Error",
                    "coins": []
                }
            }
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get all active positions"""
        try:
            if not self.api_key:
                return {
                    "success": False,
                    "message": "API credentials not configured",
                    "data": {"positions": []}
                }
            
            endpoint = "/v5/position/list"
            params = "category=linear"
            url = f"{self.base_url}{endpoint}?{params}"
            
            headers = self._get_headers(params)
            
            async with self.session.get(url, headers=headers) as response:
                data = await response.json()
                
                if response.status == 200 and data.get("retCode") == 0:
                    positions = []
                    for pos in data["result"]["list"]:
                        size = float(pos.get("size", "0"))
                        if size > 0:  # Only active positions
                            positions.append({
                                "symbol": pos.get("symbol"),
                                "side": pos.get("side").lower(),
                                "size": pos.get("size"),
                                "entry_price": pos.get("avgPrice"),
                                "mark_price": pos.get("markPrice"),
                                "pnl": pos.get("unrealisedPnl"),
                                "pnl_percentage": f"{float(pos.get('unrealisedPnl', '0')) / float(pos.get('positionValue', '1')) * 100:.2f}%"
                            })
                    
                    return {
                        "success": True,
                        "data": {"positions": positions}
                    }
                else:
                    return {
                        "success": False,
                        "message": f"API Error: {data.get('retMsg', 'Unknown error')}",
                        "data": {"positions": []}
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching positions: {str(e)}")
            return {
                "success": False,
                "message": f"Connection error: {str(e)}",
                "data": {"positions": []}
            }
    
    async def get_active_symbols(self, category: str = "linear") -> Dict[str, Any]:
        """Get all active trading symbols/pairs"""
        try:
            endpoint = "/v5/market/instruments-info"
            params = f"category={category}"
            url = f"{self.base_url}{endpoint}?{params}"
            
            headers = {"Content-Type": "application/json"}
            
            async with self.session.get(url, headers=headers) as response:
                data = await response.json()
                
                if response.status == 200 and data.get("retCode") == 0:
                    active_symbols = []
                    for instrument in data["result"]["list"]:
                        if instrument.get("status") == "Trading":
                            active_symbols.append({
                                "symbol": instrument.get("symbol"),
                                "base_coin": instrument.get("baseCoin"),
                                "quote_coin": instrument.get("quoteCoin"),
                                "status": instrument.get("status"),
                                "min_price": instrument.get("priceFilter", {}).get("minPrice"),
                                "tick_size": instrument.get("priceFilter", {}).get("tickSize"),
                                "min_order_qty": instrument.get("lotSizeFilter", {}).get("minOrderQty")
                            })
                    
                    return {
                        "success": True,
                        "data": {
                            "total_symbols": len(active_symbols),
                            "symbols": active_symbols
                        }
                    }
                else:
                    return {
                        "success": False,
                        "message": f"API Error: {data.get('retMsg', 'Unknown error')}",
                        "data": {"symbols": []}
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching symbols: {str(e)}")
            return {
                "success": False,
                "message": f"Connection error: {str(e)}",
                "data": {"symbols": []}
            }

def load_api_credentials() -> tuple[Optional[str], Optional[str]]:
    """Load API credentials from config or environment"""
    try:
        # Try to load from config file
        config_path = Path("config/secrets.yaml")
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                bybit_config = config.get('bybit', {})
                return (
                    bybit_config.get('api_key'),
                    bybit_config.get('api_secret')
                )
    except Exception as e:
        logger.warning(f"Could not load config file: {e}")
    
    # Try environment variables
    import os
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    
    return api_key, api_secret

# Global client instance
_bybit_client = None

async def get_bybit_client() -> BybitAPIClient:
    """Get or create Bybit API client"""
    global _bybit_client
    
    if _bybit_client is None:
        api_key, api_secret = load_api_credentials()
        _bybit_client = BybitAPIClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True  # Set to False for mainnet
        )
        await _bybit_client.__aenter__()
    
    return _bybit_client