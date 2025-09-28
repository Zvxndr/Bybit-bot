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

# Import debug logger with fallback
try:
    from .debug_logger import log_exception, log_performance
except ImportError:
    try:
        from debug_logger import log_exception, log_performance
    except ImportError:
        # Fallback functions if debug_logger is not available
        def log_exception(exc, context=""):
            import logging
            logging.getLogger(__name__).error(f"Exception in {context}: {exc}")
        def log_performance(operation, start_time, **kwargs):
            import logging
            duration = time.time() - start_time
            logging.getLogger(__name__).debug(f"Performance {operation}: {duration:.2f}s {kwargs}")

# Import debug safety with fallback
try:
    from .debug_safety import get_debug_manager
except ImportError:
    try:
        from debug_safety import get_debug_manager
    except ImportError:
        # Fallback debug manager for deployment safety
        class FallbackDebugManager:
            def is_debug_mode(self): return True
            def block_trading_operation(self, op): return True
            def should_use_testnet(self): return True
            def should_mock_api_calls(self): return True
            def log_debug_action(self, action, details=""): 
                import logging
                logging.getLogger(__name__).debug(f"🔧 {action}: {details}")
        
        def get_debug_manager(): return FallbackDebugManager()

logger = logging.getLogger(__name__)

class BybitAPIClient:
    """Async Bybit API client for real-time data fetching with debug safety"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        logger.info(f"🔧 Initializing BybitAPIClient (testnet={testnet})")
        
        # Initialize debug manager
        self.debug_manager = get_debug_manager()
        
        # Force testnet in debug mode
        if self.debug_manager.should_use_testnet():
            testnet = True
            logger.warning("🚨 DEBUG MODE: Forced testnet usage")
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # API endpoints
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
            
        logger.info(f"🔧 Using API base URL: {self.base_url}")
        logger.debug(f"🔧 API key provided: {'✅' if api_key else '❌'}")
        logger.debug(f"🔧 API secret provided: {'✅' if api_secret else '❌'}")
            
        # Session management - create once, reuse
        self.session = None
        self._session_lock = asyncio.Lock()
        logger.debug("🔧 BybitAPIClient initialization complete")
        
    async def get_session(self):
        """Get or create aiohttp session with proper management"""
        logger.debug("🔧 Getting API session")
        if self.session is None or self.session.closed:
            async with self._session_lock:
                if self.session is None or self.session.closed:
                    logger.debug("🔧 Creating new aiohttp session")
                    self.session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=30)
                    )
                    logger.debug("🔧 New session created successfully")
        return self.session
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, timestamp: str, recv_window: str, params: str, method: str = "GET") -> str:
        """Generate API signature for authenticated requests"""
        logger.debug(f"🔧 Generating signature for {method} request")
        if not self.api_secret:
            logger.warning("⚠️ No API secret provided - signature will be empty")
            return ""
        
        # Bybit signature format varies by method:
        # GET: timestamp + api_key + recv_window + params (query string)
        # POST: timestamp + api_key + recv_window + JSON_body
        if method == "POST":
            # For POST requests, params is the JSON body
            param_str = f"{timestamp}{self.api_key}{recv_window}{params}"
        else:
            # For GET requests, params is the query string
            param_str = f"{timestamp}{self.api_key}{recv_window}{params}"
        
        logger.debug(f"🔧 Param string length: {len(param_str)}")
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        logger.debug(f"🔧 Signature generated: {signature[:16]}...")
        
        return signature
    
    def _get_headers(self, params: str = "", method: str = "GET") -> Dict[str, str]:
        """Get headers for API requests"""
        logger.debug(f"🔧 Preparing headers for {method} request")
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        headers = {
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": self.api_key or "",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window
        }
        
        if self.api_key and self.api_secret:
            logger.debug("🔧 Adding signature to headers")
            signature = self._generate_signature(timestamp, recv_window, params, method)
            headers["X-BAPI-SIGN"] = signature
        else:
            logger.warning("⚠️ No API credentials - using unsigned request")
        
        logger.debug(f"🔧 Headers prepared with timestamp: {timestamp}")
        return headers
    
    async def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance for all coins"""
        start_time = time.time()
        logger.info("🔧 Starting get_account_balance request")
        
        # Check if debug mode is active and return historical data
        if self.debug_manager.is_debug_mode():
            logger.info("🔧 Debug mode active - using historical balance data")
            mock_balances = self.debug_manager.get_mock_data('balances')
            if mock_balances:
                logger.info("✅ Using historical balance data for debugging")
                return {
                    "success": True,
                    "data": {
                        "total_wallet_balance": f"{mock_balances.get('total_usd', 15000.00):.2f}",
                        "total_available_balance": f"{mock_balances.get('USDT', 10000.00):.2f}",
                        "total_used_margin": "0.00",
                        "coins": [
                            {
                                "coin": "USDT",
                                "wallet_balance": mock_balances.get('USDT', 10000.00),
                                "available_balance": mock_balances.get('USDT', 10000.00),
                                "used_margin": 0.00
                            },
                            {
                                "coin": "BTC",
                                "wallet_balance": mock_balances.get('BTC', 0.15),
                                "available_balance": mock_balances.get('BTC', 0.15),
                                "used_margin": 0.00
                            }
                        ]
                    }
                }
        
        try:
            if not self.api_key:
                logger.warning("⚠️ API credentials not configured for balance request")
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
            logger.debug(f"🔧 Request URL: {url}")
            
            headers = self._get_headers(params)
            logger.debug("🔧 Headers prepared for balance request")
            
            session = await self.get_session()
            logger.debug("🔧 Making HTTP GET request to Bybit API")
            
            async with session.get(url, headers=headers) as response:
                logger.debug(f"🔧 Response status: {response.status}")
                
                # Enhanced error handling for None responses in balance method
                try:
                    data = await response.json()
                    logger.debug(f"🔧 Response data size: {len(str(data)) if data else 0} chars")
                except Exception as json_error:
                    logger.error(f"❌ Failed to parse JSON response in get_account_balance: {json_error}")
                    log_exception(json_error, "JSON parsing in get_account_balance")
                    data = None
                
                if data is None:
                    logger.error(f"❌ Bybit API returned None response in balance check. Status: {response.status}")
                    return {
                        "success": False,
                        "message": f"API returned empty response (Status: {response.status})",
                        "data": {
                            "total_wallet_balance": "API Error",
                            "total_available_balance": "API Error",
                            "total_used_margin": "API Error",
                            "coins": []
                        }
                    }
                
                if response.status == 200 and data.get("retCode") == 0:
                    logger.info("✅ Successfully received balance data from Bybit")
                    wallet_data = data["result"]["list"][0] if data["result"]["list"] else {}
                    logger.debug(f"🔧 Wallet data keys: {list(wallet_data.keys())}")
                    
                    # Safe float conversion helper
                    def safe_float(value, default=0.0):
                        try:
                            return float(value) if value and str(value).strip() else default
                        except (ValueError, TypeError):
                            return default
                    
                    # Process balance data with safe conversion
                    total_wallet = safe_float(wallet_data.get("totalWalletBalance"))
                    total_available = safe_float(wallet_data.get("totalAvailableBalance"))
                    total_margin = safe_float(wallet_data.get("totalMarginBalance"))
                    total_used = max(0, total_margin - total_available)
                    
                    logger.debug(f"🔧 Balance summary: Wallet={total_wallet:.2f}, Available={total_available:.2f}, Used={total_used:.2f}")
                    
                    # Get individual coin balances
                    coins = []
                    for coin_data in wallet_data.get("coin", []):
                        wallet_balance = safe_float(coin_data.get("walletBalance"))
                        if wallet_balance > 0:
                            coins.append({
                                "coin": coin_data.get("coin"),
                                "wallet_balance": wallet_balance,
                                "available_balance": safe_float(coin_data.get("availableToWithdraw")),
                                "used_margin": safe_float(coin_data.get("totalOrderIM"))
                            })
                    
                    logger.info(f"✅ Processed {len(coins)} coins with balance > 0")
                    log_performance("get_account_balance", start_time, coins_found=len(coins))
                    
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
                    logger.error(f"❌ Bybit API error: Status={response.status}, Data={data}")
                    return {
                        "success": False,
                        "message": f"API Error: {data.get('retMsg', 'Unknown error') if data else 'No response data'}",
                        "data": {
                            "total_wallet_balance": "Error",
                            "total_available_balance": "Error",
                            "total_used_margin": "Error",
                            "coins": []
                        }
                    }
                    
        except Exception as e:
            logger.error(f"❌ Error fetching balance: {str(e)}")
            log_exception(e, "get_account_balance")
            log_performance("get_account_balance", start_time, error=str(e))
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
        start_time = time.time()
        logger.info("🔧 Starting get_positions request")
        
        # Check if debug mode is active and return historical data
        if self.debug_manager.is_debug_mode():
            logger.info("🔧 Debug mode active - using historical position data")
            mock_positions = self.debug_manager.get_mock_data('positions')
            if mock_positions:
                logger.info(f"✅ Using {len(mock_positions)} historical positions for debugging")
                return {
                    "success": True,
                    "data": {"positions": mock_positions}
                }
        
        try:
            if not self.api_key:
                logger.warning("⚠️ API credentials not configured for positions request")
                return {
                    "success": False,
                    "message": "API credentials not configured",
                    "data": {"positions": []}
                }
            
            endpoint = "/v5/position/list"
            params = "category=linear&settleCoin=USDT"
            url = f"{self.base_url}{endpoint}?{params}"
            logger.debug(f"🔧 Request URL: {url}")
            
            headers = self._get_headers(params)
            logger.debug("🔧 Headers prepared for positions request")
            
            session = await self.get_session()
            logger.debug("🔧 Making HTTP GET request for positions")
            async with session.get(url, headers=headers) as response:
                logger.debug(f"🔧 Response status: {response.status}")
                
                try:
                    data = await response.json()
                    logger.debug(f"🔧 Response data size: {len(str(data)) if data else 0} chars")
                except Exception as json_error:
                    logger.error(f"❌ Failed to parse JSON response in get_positions: {json_error}")
                    log_exception(json_error, "JSON parsing in get_positions")
                    return {
                        "success": False,
                        "message": f"Failed to parse response: {json_error}",
                        "data": {"positions": []}
                    }
                
                if response.status == 200 and data.get("retCode") == 0:
                    logger.info("✅ Successfully received positions data from Bybit")
                    
                    # Safe float conversion helper
                    def safe_float(value, default=0.0):
                        try:
                            return float(value) if value and str(value).strip() else default
                        except (ValueError, TypeError):
                            return default
                    
                    positions = []
                    total_positions = len(data["result"]["list"])
                    logger.debug(f"🔧 Processing {total_positions} total positions from API")
                    
                    for pos in data["result"]["list"]:
                        size = safe_float(pos.get("size"))
                        if size > 0:  # Only active positions
                            unrealised_pnl = safe_float(pos.get("unrealisedPnl"))
                            position_value = safe_float(pos.get("positionValue"), 1.0)
                            pnl_percentage = (unrealised_pnl / position_value * 100) if position_value > 0 else 0
                            
                            positions.append({
                                "symbol": pos.get("symbol"),
                                "side": pos.get("side", "").lower(),
                                "size": str(size),
                                "entry_price": pos.get("avgPrice"),
                                "mark_price": pos.get("markPrice"),
                                "pnl": str(unrealised_pnl),
                                "pnl_percentage": f"{pnl_percentage:.2f}%"
                            })
                    
                    logger.info(f"✅ Found {len(positions)} active positions out of {total_positions} total")
                    log_performance("get_positions", start_time, 
                                  active_positions=len(positions), 
                                  total_positions=total_positions)
                    
                    return {
                        "success": True,
                        "data": {"positions": positions}
                    }
                else:
                    logger.error(f"❌ Bybit API error: Status={response.status}, Data={data}")
                    return {
                        "success": False,
                        "message": f"API Error: {data.get('retMsg', 'Unknown error') if data else 'No response data'}",
                        "data": {"positions": []}
                    }
                    
        except Exception as e:
            logger.error(f"❌ Error fetching positions: {str(e)}")
            log_exception(e, "get_positions")
            log_performance("get_positions", start_time, error=str(e))
            return {
                "success": False,
                "message": f"Connection error: {str(e)}",
                "data": {"positions": []}
            }
    
    async def place_market_order(self, symbol: str, side: str, qty: str) -> Dict[str, Any]:
        """Place a market order"""
        try:
            if not self.api_key:
                return {
                    "success": False,
                    "message": "API credentials not configured"
                }
            
            endpoint = "/v5/order/create"
            
            # Create request body
            request_body = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": qty,
                "timeInForce": "IOC"
            }
            
            # Convert to JSON string for signature
            body_str = json.dumps(request_body, separators=(',', ':'))
            url = f"{self.base_url}{endpoint}"
            
            headers = self._get_headers(body_str, method="POST")
            headers["Content-Type"] = "application/json"
            
            session = await self.get_session()
            async with session.post(url, headers=headers, data=body_str) as response:
                data = await response.json()
                
                if response.status == 200 and data.get("retCode") == 0:
                    return {
                        "success": True,
                        "data": data["result"]
                    }
                else:
                    error_msg = data.get("retMsg", f"HTTP {response.status}")
                    return {
                        "success": False,
                        "message": f"Order failed: {error_msg}"
                    }
                    
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return {
                "success": False,
                "message": f"Connection error: {str(e)}"
            }
    
    async def close(self):
        """Clean shutdown of the client"""
        if self.session and not self.session.closed:
            await self.session.close()
    
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
    
    async def place_order(self, symbol: str, side: str, order_type: str, qty: str, price: str = None) -> Dict[str, Any]:
        """Place a trading order on Bybit with debug safety"""
        try:
            # Check debug mode first - block all orders
            if self.debug_manager.block_trading_operation('place_order'):
                logger.warning(f"🚫 DEBUG MODE: Blocked order placement - {symbol} {side} {qty}")
                return {
                    "success": True,  # Return success for UI testing
                    "message": "DEBUG MODE: Order blocked for safety",
                    "data": {
                        "order_id": f"DEBUG_{int(time.time())}",
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "order_type": order_type,
                        "status": "debug_blocked"
                    }
                }
            
            if not self.api_key or not self.api_secret:
                return {
                    "success": False,
                    "message": "API credentials not configured",
                    "data": {}
                }
            
            endpoint = "/v5/order/create"
            
            # Build order parameters
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side.capitalize(),  # Buy or Sell
                "orderType": order_type.capitalize(),  # Market or Limit
                "qty": qty
            }
            
            # Add price for limit orders
            if order_type.lower() == "limit" and price:
                order_params["price"] = price
            
            # For POST requests, use JSON body directly
            import json
            params_str = json.dumps(order_params, separators=(',', ':'))
            
            url = f"{self.base_url}{endpoint}"
            headers = self._get_headers(params_str, method="POST")
            
            async with self.session.post(url, json=order_params, headers=headers) as response:
                data = await response.json()
                
                if response.status == 200 and data.get("retCode") == 0:
                    return {
                        "success": True,
                        "data": {
                            "order_id": data["result"]["orderId"],
                            "symbol": symbol,
                            "side": side,
                            "qty": qty,
                            "order_type": order_type,
                            "status": "submitted"
                        }
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Order failed: {data.get('retMsg', 'Unknown error')}",
                        "data": {}
                    }
                    
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return {
                "success": False,
                "message": f"Connection error: {str(e)}",
                "data": {}
            }

    async def get_trade_history(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent trade history"""
        start_time = time.time()
        logger.info("🔧 Starting get_trade_history request")
        
        # Check if debug mode is active and return historical data
        if self.debug_manager.is_debug_mode():
            logger.info("🔧 Debug mode active - using historical trade data")
            mock_trades = self.debug_manager.get_mock_data('trades')
            if mock_trades:
                # Limit to the requested number of trades
                limited_trades = mock_trades[:limit] if len(mock_trades) > limit else mock_trades
                logger.info(f"✅ Using {len(limited_trades)} historical trades for debugging")
                return {
                    "success": True,
                    "data": {"trades": limited_trades}
                }
        
        try:
            if not self.api_key:
                logger.warning("⚠️ API credentials not configured for trade history request")
                return {
                    "success": False,
                    "message": "API credentials not configured",
                    "data": {"trades": []}
                }
            
            endpoint = "/v5/execution/list"
            params = f"category=linear&limit={limit}"
            url = f"{self.base_url}{endpoint}?{params}"
            logger.debug(f"🔧 Request URL: {url}")
            
            headers = self._get_headers(params)
            logger.debug("🔧 Headers prepared for trade history request")
            
            session = await self.get_session()
            logger.debug("🔧 Making HTTP GET request for trade history")
            
            async with session.get(url, headers=headers) as response:
                logger.debug(f"🔧 Response status: {response.status}")
                
                try:
                    data = await response.json()
                    logger.debug(f"🔧 Response data size: {len(str(data)) if data else 0} chars")
                except Exception as json_error:
                    logger.error(f"❌ Failed to parse JSON response in get_trade_history: {json_error}")
                    log_exception(json_error, "JSON parsing in get_trade_history")
                    return {
                        "success": False,
                        "message": f"Failed to parse response: {json_error}",
                        "data": {"trades": []}
                    }
                
                if response.status == 200 and data.get("retCode") == 0:
                    logger.info("✅ Successfully received trade history from Bybit")
                    
                    trades = []
                    executions = data.get("result", {}).get("list", [])
                    logger.debug(f"🔧 Processing {len(executions)} trade executions")
                    
                    for execution in executions:
                        try:
                            # Parse execution data
                            trade = {
                                "symbol": execution.get("symbol"),
                                "side": execution.get("side", "").lower(),
                                "size": execution.get("execQty"),
                                "price": execution.get("execPrice"),
                                "value": execution.get("execValue"),
                                "fee": execution.get("execFee"),
                                "timestamp": execution.get("execTime"),
                                "order_id": execution.get("orderId"),
                                "exec_id": execution.get("execId")
                            }
                            trades.append(trade)
                        except Exception as e:
                            logger.warning(f"⚠️ Error parsing trade execution: {e}")
                            continue
                    
                    logger.info(f"✅ Processed {len(trades)} trade executions")
                    log_performance("get_trade_history", start_time, trades_found=len(trades))
                    
                    return {
                        "success": True,
                        "data": {"trades": trades}
                    }
                else:
                    logger.error(f"❌ Bybit API error: Status={response.status}, Data={data}")
                    return {
                        "success": False,
                        "message": f"API Error: {data.get('retMsg', 'Unknown error') if data else 'No response data'}",
                        "data": {"trades": []}
                    }
                    
        except Exception as e:
            logger.error(f"❌ Error fetching trade history: {str(e)}")
            log_exception(e, "get_trade_history")
            log_performance("get_trade_history", start_time, error=str(e))
            return {
                "success": False,
                "message": f"Connection error: {str(e)}",
                "data": {"trades": []}
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
    
    # Try environment variables (support strategy graduation system)
    import os
    
    # For strategy graduation system:
    # - BYBIT_TESTNET_* for paper trading validation
    # - BYBIT_LIVE_* for graduated strategies  
    # - BYBIT_API_* as generic fallback
    
    # Check if we're in testnet mode (default for strategy validation)
    testnet_mode = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
    
    if testnet_mode:
        # Paper trading mode - use testnet credentials
        api_key = os.getenv('BYBIT_TESTNET_API_KEY') or os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_TESTNET_API_SECRET') or os.getenv('BYBIT_API_SECRET')
    else:
        # Live trading mode - use live credentials for graduated strategies
        api_key = os.getenv('BYBIT_LIVE_API_KEY') or os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_LIVE_API_SECRET') or os.getenv('BYBIT_API_SECRET')
    
    return api_key, api_secret

# Global client instance
_bybit_client = None

async def get_bybit_client() -> BybitAPIClient:
    """Get or create Bybit API client with strategy graduation support"""
    global _bybit_client
    
    if _bybit_client is None:
        import os
        api_key, api_secret = load_api_credentials()
        
        # Determine testnet mode for strategy graduation system
        testnet_mode = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
        
        _bybit_client = BybitAPIClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet_mode  # Dynamic based on strategy graduation
        )
        await _bybit_client.__aenter__()
    
    return _bybit_client