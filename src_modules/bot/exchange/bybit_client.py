"""
Bybit API Client - Core Implementation

This module provides a comprehensive Bybit API v5 client implementation with:
- Authentication and signature generation
- Rate limiting compliance
- Historical data fetching (OHLCV, funding rates)
- Real-time market data
- Trading operations (orders, positions, balance)
- WebSocket support for live data streams
- Error handling and retry logic
"""

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlencode
import logging

import aiohttp
import pandas as pd
from dataclasses import dataclass, field

from ..utils.rate_limiter import RateLimiter
from ..utils.logging import TradingLogger


@dataclass
class BybitCredentials:
    """Bybit API credentials configuration."""
    api_key: str
    api_secret: str
    testnet: bool = True
    recv_window: int = 5000
    
    @property
    def base_url(self) -> str:
        """Get base URL for API endpoint."""
        return "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"


@dataclass
class MarketDataResponse:
    """Standardized market data response."""
    symbol: str
    data: pd.DataFrame
    timestamp: datetime
    source: str = "bybit"
    
    def __post_init__(self):
        if self.data is not None and not self.data.empty:
            # Ensure timestamp column is datetime
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ms')


@dataclass
class FundingRateData:
    """Funding rate historical data."""
    symbol: str
    funding_rate: float
    funding_rate_timestamp: datetime
    next_funding_time: datetime


class BybitAPIError(Exception):
    """Custom exception for Bybit API errors."""
    
    def __init__(self, message: str, ret_code: int = None, ret_msg: str = None):
        self.ret_code = ret_code
        self.ret_msg = ret_msg
        super().__init__(message)


class BybitClient:
    """
    Comprehensive Bybit API v5 client implementation.
    
    Features:
    - Full authentication with HMAC SHA256 signatures
    - Automatic rate limiting compliance
    - Historical data fetching with caching
    - Real-time WebSocket connections
    - Error handling with exponential backoff
    - Trading operations for all account types
    """
    
    def __init__(
        self,
        credentials: BybitCredentials,
        enable_rate_limiting: bool = True,
        session_timeout: int = 30,
        max_retries: int = 3
    ):
        self.credentials = credentials
        self.logger = TradingLogger(f"BybitClient-{'testnet' if credentials.testnet else 'mainnet'}")
        
        # HTTP session configuration
        self.session_timeout = aiohttp.ClientTimeout(total=session_timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        self.max_retries = max_retries
        
        # Rate limiting - Bybit API limits
        self.rate_limiter = RateLimiter(
            max_requests=120,  # Conservative limit
            time_window=60,    # Per minute
            burst_limit=10     # Burst allowance
        ) if enable_rate_limiting else None
        
        # Data caching
        self.market_data_cache: Dict[str, MarketDataResponse] = {}
        self.funding_rate_cache: Dict[str, List[FundingRateData]] = {}
        self.cache_ttl = timedelta(minutes=1)
        
        # Connection state
        self.connected = False
        self.last_server_time: Optional[datetime] = None
        
        self.logger.info(f"BybitClient initialized - Testnet: {credentials.testnet}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> bool:
        """
        Initialize connection to Bybit API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.logger.info("Connecting to Bybit API...")
            
            # Create HTTP session
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.session_timeout,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Bybit-Trading-Bot/1.0'
                }
            )
            
            # Test connectivity
            server_time = await self.get_server_time()
            if server_time:
                self.connected = True
                self.logger.info(f"Connected to Bybit API - Server time: {server_time}")
                return True
            else:
                self.logger.error("Failed to get server time")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Bybit API."""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.connected = False
            self.logger.info("Disconnected from Bybit API")
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
    
    def _generate_signature(self, timestamp: str, method: str, endpoint: str, params: str = "") -> str:
        """
        Generate HMAC SHA256 signature for API authentication.
        
        Args:
            timestamp: Current timestamp in milliseconds
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters or request body
            
        Returns:
            str: HMAC signature
        """
        # Bybit v5 signature format: timestamp + api_key + recv_window + params
        param_str = f"{timestamp}{self.credentials.api_key}{self.credentials.recv_window}{params}"
        
        signature = hmac.new(
            self.credentials.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _get_headers(self, method: str, endpoint: str, params: str = "") -> Dict[str, str]:
        """
        Generate headers for authenticated API requests.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Dict[str, str]: Request headers
        """
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, endpoint, params)
        
        return {
            'X-BAPI-API-KEY': self.credentials.api_key,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-SIGN': signature,
            'X-BAPI-RECV-WINDOW': str(self.credentials.recv_window)
        }
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        authenticated: bool = False,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Bybit API with error handling and retries.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters
            authenticated: Whether to include authentication headers
            retry_count: Current retry attempt
            
        Returns:
            Dict[str, Any]: API response
            
        Raises:
            BybitAPIError: If API returns error or max retries exceeded
        """
        if not self.session:
            raise BybitAPIError("Not connected to Bybit API")
        
        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        
        url = f"{self.credentials.base_url}{endpoint}"
        headers = {}
        
        try:
            # Prepare request parameters
            if params is None:
                params = {}
            
            # Add authentication headers if required
            if authenticated:
                if method.upper() == 'GET':
                    query_string = urlencode(params) if params else ""
                    headers.update(self._get_headers(method, endpoint, query_string))
                else:
                    body = json.dumps(params) if params else ""
                    headers.update(self._get_headers(method, endpoint, body))
            
            # Make request
            if method.upper() == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    response_data = await response.json()
            else:
                async with self.session.request(
                    method, url, json=params, headers=headers
                ) as response:
                    response_data = await response.json()
            
            # Check for API errors
            ret_code = response_data.get('retCode', 0)
            if ret_code != 0:
                ret_msg = response_data.get('retMsg', 'Unknown error')
                
                # Handle specific error codes
                if ret_code == 10002:  # Invalid API key
                    raise BybitAPIError(f"Invalid API key: {ret_msg}", ret_code, ret_msg)
                elif ret_code == 10003:  # Invalid signature
                    raise BybitAPIError(f"Invalid signature: {ret_msg}", ret_code, ret_msg)
                elif ret_code == 10004:  # Request timeout
                    if retry_count < self.max_retries:
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                        return await self._make_request(method, endpoint, params, authenticated, retry_count + 1)
                    raise BybitAPIError(f"Request timeout after {self.max_retries} retries", ret_code, ret_msg)
                else:
                    raise BybitAPIError(f"API error: {ret_msg}", ret_code, ret_msg)
            
            return response_data
            
        except aiohttp.ClientError as e:
            if retry_count < self.max_retries:
                await asyncio.sleep(2 ** retry_count)
                return await self._make_request(method, endpoint, params, authenticated, retry_count + 1)
            raise BybitAPIError(f"Network error: {e}")
        
        except Exception as e:
            if retry_count < self.max_retries:
                await asyncio.sleep(2 ** retry_count)
                return await self._make_request(method, endpoint, params, authenticated, retry_count + 1)
            raise BybitAPIError(f"Unexpected error: {e}")
    
    # Public Market Data Methods
    
    async def get_server_time(self) -> Optional[datetime]:
        """
        Get Bybit server time.
        
        Returns:
            Optional[datetime]: Server timestamp or None if failed
        """
        try:
            response = await self._make_request('GET', '/v5/market/time')
            timestamp_ms = int(response['result']['timeSecond']) * 1000
            server_time = datetime.fromtimestamp(timestamp_ms / 1000)
            self.last_server_time = server_time
            return server_time
            
        except Exception as e:
            self.logger.error(f"Failed to get server time: {e}")
            return None
    
    async def get_kline_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 200
    ) -> Optional[MarketDataResponse]:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval ('1', '5', '15', '30', '60', '240', '360', '720', 'D', 'W', 'M')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of data points (max 1000)
            
        Returns:
            Optional[MarketDataResponse]: OHLCV data or None if failed
        """
        try:
            params = {
                'category': 'linear',  # Perpetual futures
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)
            }
            
            if start_time:
                params['start'] = start_time
            if end_time:
                params['end'] = end_time
            
            response = await self._make_request('GET', '/v5/market/kline', params)
            
            # Parse response
            kline_list = response['result']['list']
            if not kline_list:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(kline_list, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Sort by timestamp (ascending)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return MarketDataResponse(
                symbol=symbol,
                data=df,
                timestamp=datetime.now(),
                source="bybit"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to fetch kline data for {symbol}: {e}")
            return None
    
    async def get_funding_rate_history(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 200
    ) -> Optional[List[FundingRateData]]:
        """
        Fetch funding rate history for perpetual contracts.
        
        Args:
            symbol: Trading symbol
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of records (max 200)
            
        Returns:
            Optional[List[FundingRateData]]: Funding rate history or None if failed
        """
        try:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'limit': min(limit, 200)
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            response = await self._make_request('GET', '/v5/market/funding/history', params)
            
            funding_list = response['result']['list']
            if not funding_list:
                return None
            
            funding_data = []
            for item in funding_list:
                funding_data.append(FundingRateData(
                    symbol=item['symbol'],
                    funding_rate=float(item['fundingRate']),
                    funding_rate_timestamp=datetime.fromtimestamp(int(item['fundingRateTimestamp']) / 1000),
                    next_funding_time=datetime.fromtimestamp(int(item.get('nextFundingTime', 0)) / 1000)
                ))
            
            return funding_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch funding rate history for {symbol}: {e}")
            return None
    
    async def get_ticker_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current ticker data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Optional[Dict[str, Any]]: Ticker data or None if failed
        """
        try:
            params = {
                'category': 'linear',
                'symbol': symbol
            }
            
            response = await self._make_request('GET', '/v5/market/tickers', params)
            
            ticker_list = response['result']['list']
            if ticker_list:
                return ticker_list[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to fetch ticker data for {symbol}: {e}")
            return None
    
    # Authenticated Account Methods
    
    async def get_wallet_balance(self, account_type: str = "UNIFIED") -> Optional[Dict[str, Any]]:
        """
        Get wallet balance for specified account type.
        
        Args:
            account_type: Account type ("UNIFIED", "CONTRACT", "SPOT")
            
        Returns:
            Optional[Dict[str, Any]]: Balance data or None if failed
        """
        try:
            params = {
                'accountType': account_type
            }
            
            response = await self._make_request('GET', '/v5/account/wallet-balance', params, authenticated=True)
            return response['result']
            
        except Exception as e:
            self.logger.error(f"Failed to fetch wallet balance: {e}")
            return None
    
    async def get_positions(self, symbol: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Get current positions.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Optional[List[Dict[str, Any]]]: Position data or None if failed
        """
        try:
            params = {
                'category': 'linear'
            }
            
            if symbol:
                params['symbol'] = symbol
            
            response = await self._make_request('GET', '/v5/position/list', params, authenticated=True)
            return response['result']['list']
            
        except Exception as e:
            self.logger.error(f"Failed to fetch positions: {e}")
            return None
    
    # Trading Methods (Placeholder for Phase 3)
    
    async def place_order(
        self,
        category: str,
        symbol: str,
        side: str,
        order_type: str,
        qty: str,
        price: Optional[str] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        close_on_trigger: bool = False,
        order_link_id: Optional[str] = None,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
        tp_trigger_by: Optional[str] = None,
        sl_trigger_by: Optional[str] = None,
        position_idx: int = 0,
        order_filter: str = "Order",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Place a trading order on Bybit.
        
        Args:
            category: Product type (linear, inverse, spot, option)
            symbol: Symbol name (e.g., "BTCUSDT")
            side: Buy or Sell
            order_type: Market, Limit, etc.
            qty: Order quantity
            price: Order price (required for limit orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            reduce_only: Reduce only flag
            close_on_trigger: Close on trigger flag
            order_link_id: Custom order ID
            take_profit: Take profit price
            stop_loss: Stop loss price
            tp_trigger_by: Take profit trigger price type
            sl_trigger_by: Stop loss trigger price type
            position_idx: Position index (0 for one-way mode)
            order_filter: Order filter type
            
        Returns:
            Order response data or None on failure
        """
        try:
            # Build order parameters
            params = {
                "category": category,
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": qty,
                "timeInForce": time_in_force,
                "positionIdx": position_idx,
                "orderFilter": order_filter
            }
            
            # Add optional parameters
            if price is not None:
                params["price"] = price
            if reduce_only:
                params["reduceOnly"] = reduce_only
            if close_on_trigger:
                params["closeOnTrigger"] = close_on_trigger
            if order_link_id:
                params["orderLinkId"] = order_link_id
            if take_profit:
                params["takeProfit"] = take_profit
            if stop_loss:
                params["stopLoss"] = stop_loss
            if tp_trigger_by:
                params["tpTriggerBy"] = tp_trigger_by
            if sl_trigger_by:
                params["slTriggerBy"] = sl_trigger_by
            
            # Add any additional parameters
            params.update(kwargs)
            
            self.logger.info(f"Placing {side} order for {qty} {symbol} at {price if price else 'market'}")
            
            response = await self._make_authenticated_request(
                "POST",
                "/v5/order/create",
                params
            )
            
            if response and response.get("retCode") == 0:
                order_data = response.get("result", {})
                self.logger.info(f"Order placed successfully: {order_data.get('orderId')}")
                return order_data
            else:
                error_msg = response.get("retMsg", "Unknown error") if response else "No response"
                self.logger.error(f"Failed to place order: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception placing order: {e}")
            return None
    
    async def cancel_order(
        self, 
        category: str,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        order_filter: str = "Order"
    ) -> Optional[Dict[str, Any]]:
        """
        Cancel a trading order.
        
        Args:
            category: Product type (linear, inverse, spot, option)
            symbol: Symbol name
            order_id: Order ID (either order_id or order_link_id required)
            order_link_id: Custom order ID
            order_filter: Order filter type
            
        Returns:
            Cancel response data or None on failure
        """
        try:
            if not order_id and not order_link_id:
                self.logger.error("Either order_id or order_link_id must be provided")
                return None
            
            params = {
                "category": category,
                "symbol": symbol,
                "orderFilter": order_filter
            }
            
            if order_id:
                params["orderId"] = order_id
            if order_link_id:
                params["orderLinkId"] = order_link_id
            
            self.logger.info(f"Cancelling order {order_id or order_link_id} for {symbol}")
            
            response = await self._make_authenticated_request(
                "POST",
                "/v5/order/cancel",
                params
            )
            
            if response and response.get("retCode") == 0:
                cancel_data = response.get("result", {})
                self.logger.info(f"Order cancelled successfully: {cancel_data.get('orderId')}")
                return cancel_data
            else:
                error_msg = response.get("retMsg", "Unknown error") if response else "No response"
                self.logger.error(f"Failed to cancel order: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception cancelling order: {e}")
            return None
    
    async def modify_order(
        self,
        category: str,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        qty: Optional[str] = None,
        price: Optional[str] = None,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
        tp_trigger_by: Optional[str] = None,
        sl_trigger_by: Optional[str] = None,
        order_filter: str = "Order"
    ) -> Optional[Dict[str, Any]]:
        """
        Modify an existing order.
        
        Args:
            category: Product type (linear, inverse, spot, option)
            symbol: Symbol name
            order_id: Order ID (either order_id or order_link_id required)
            order_link_id: Custom order ID
            qty: New order quantity
            price: New order price
            take_profit: New take profit price
            stop_loss: New stop loss price
            tp_trigger_by: Take profit trigger price type
            sl_trigger_by: Stop loss trigger price type
            order_filter: Order filter type
            
        Returns:
            Modify response data or None on failure
        """
        try:
            if not order_id and not order_link_id:
                self.logger.error("Either order_id or order_link_id must be provided")
                return None
            
            params = {
                "category": category,
                "symbol": symbol,
                "orderFilter": order_filter
            }
            
            if order_id:
                params["orderId"] = order_id
            if order_link_id:
                params["orderLinkId"] = order_link_id
            if qty:
                params["qty"] = qty
            if price:
                params["price"] = price
            if take_profit:
                params["takeProfit"] = take_profit
            if stop_loss:
                params["stopLoss"] = stop_loss
            if tp_trigger_by:
                params["tpTriggerBy"] = tp_trigger_by
            if sl_trigger_by:
                params["slTriggerBy"] = sl_trigger_by
            
            self.logger.info(f"Modifying order {order_id or order_link_id} for {symbol}")
            
            response = await self._make_authenticated_request(
                "POST",
                "/v5/order/amend",
                params
            )
            
            if response and response.get("retCode") == 0:
                modify_data = response.get("result", {})
                self.logger.info(f"Order modified successfully: {modify_data.get('orderId')}")
                return modify_data
            else:
                error_msg = response.get("retMsg", "Unknown error") if response else "No response"
                self.logger.error(f"Failed to modify order: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception modifying order: {e}")
            return None
    
    # Utility Methods
    
    async def test_connectivity(self) -> bool:
        """
        Test basic API connectivity.
        
        Returns:
            bool: True if connectivity test passes
        """
        try:
            server_time = await self.get_server_time()
            return server_time is not None
            
        except Exception as e:
            self.logger.error(f"Connectivity test failed: {e}")
            return False
    
    async def validate_credentials(self) -> bool:
        """
        Validate API credentials by making an authenticated request.
        
        Returns:
            bool: True if credentials are valid
        """
        try:
            balance = await self.get_wallet_balance()
            return balance is not None
            
        except BybitAPIError as e:
            if e.ret_code in [10002, 10003]:  # Invalid key or signature
                return False
            raise
        
        except Exception as e:
            self.logger.error(f"Credential validation failed: {e}")
            return False
    
    # Additional Trading Methods
    
    async def get_open_orders(
        self,
        category: str,
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        settle_coin: Optional[str] = None,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        order_filter: str = "Order",
        limit: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Get open orders.
        
        Args:
            category: Product type (linear, inverse, spot, option)
            symbol: Symbol name
            base_coin: Base coin (for spot)
            settle_coin: Settle coin  
            order_id: Order ID filter
            order_link_id: Custom order ID filter
            order_filter: Order filter type
            limit: Limit for data size per page
            
        Returns:
            Open orders data or None on failure
        """
        try:
            params = {
                "category": category,
                "orderFilter": order_filter,
                "limit": limit
            }
            
            if symbol:
                params["symbol"] = symbol
            if base_coin:
                params["baseCoin"] = base_coin
            if settle_coin:
                params["settleCoin"] = settle_coin
            if order_id:
                params["orderId"] = order_id
            if order_link_id:
                params["orderLinkId"] = order_link_id
            
            response = await self._make_authenticated_request(
                "GET",
                "/v5/order/realtime",
                params
            )
            
            if response and response.get("retCode") == 0:
                return response.get("result", {})
            else:
                error_msg = response.get("retMsg", "Unknown error") if response else "No response"
                self.logger.error(f"Failed to get open orders: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception getting open orders: {e}")
            return None
    
    async def get_positions(
        self,
        category: str,
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        settle_coin: Optional[str] = None,
        limit: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Get position information.
        
        Args:
            category: Product type (linear, inverse, spot, option)
            symbol: Symbol name
            base_coin: Base coin (for spot)
            settle_coin: Settle coin
            limit: Limit for data size per page
            
        Returns:
            Position data or None on failure
        """
        try:
            params = {
                "category": category,
                "limit": limit
            }
            
            if symbol:
                params["symbol"] = symbol
            if base_coin:
                params["baseCoin"] = base_coin
            if settle_coin:
                params["settleCoin"] = settle_coin
            
            response = await self._make_authenticated_request(
                "GET",
                "/v5/position/list",
                params
            )
            
            if response and response.get("retCode") == 0:
                return response.get("result", {})
            else:
                error_msg = response.get("retMsg", "Unknown error") if response else "No response"
                self.logger.error(f"Failed to get positions: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception getting positions: {e}")
            return None
    
    async def get_wallet_balance(
        self,
        account_type: str,
        coin: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get wallet balance.
        
        Args:
            account_type: Account type (UNIFIED, CONTRACT, SPOT, etc.)
            coin: Coin filter
            
        Returns:
            Wallet balance data or None on failure
        """
        try:
            params = {
                "accountType": account_type
            }
            
            if coin:
                params["coin"] = coin
            
            response = await self._make_authenticated_request(
                "GET",
                "/v5/account/wallet-balance",
                params
            )
            
            if response and response.get("retCode") == 0:
                return response.get("result", {})
            else:
                error_msg = response.get("retMsg", "Unknown error") if response else "No response"
                self.logger.error(f"Failed to get wallet balance: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception getting wallet balance: {e}")
            return None
    
    async def cancel_all_orders(
        self,
        category: str,
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        settle_coin: Optional[str] = None,
        order_filter: str = "Order"
    ) -> Optional[Dict[str, Any]]:
        """
        Cancel all open orders.
        
        Args:
            category: Product type (linear, inverse, spot, option)
            symbol: Symbol name (if specified, cancel orders for this symbol only)
            base_coin: Base coin (for spot)
            settle_coin: Settle coin
            order_filter: Order filter type
            
        Returns:
            Cancel all response data or None on failure
        """
        try:
            params = {
                "category": category,
                "orderFilter": order_filter
            }
            
            if symbol:
                params["symbol"] = symbol
            if base_coin:
                params["baseCoin"] = base_coin
            if settle_coin:
                params["settleCoin"] = settle_coin
            
            self.logger.info(f"Cancelling all orders for category: {category}")
            
            response = await self._make_authenticated_request(
                "POST",
                "/v5/order/cancel-all",
                params
            )
            
            if response and response.get("retCode") == 0:
                cancel_data = response.get("result", {})
                self.logger.info("All orders cancelled successfully")
                return cancel_data
            else:
                error_msg = response.get("retMsg", "Unknown error") if response else "No response"
                self.logger.error(f"Failed to cancel all orders: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception cancelling all orders: {e}")
            return None
    
    async def set_leverage(
        self,
        category: str,
        symbol: str,
        buy_leverage: str,
        sell_leverage: str
    ) -> Optional[Dict[str, Any]]:
        """
        Set leverage for a symbol.
        
        Args:
            category: Product type (linear, inverse)
            symbol: Symbol name
            buy_leverage: Buy leverage
            sell_leverage: Sell leverage
            
        Returns:
            Set leverage response data or None on failure
        """
        try:
            params = {
                "category": category,
                "symbol": symbol,
                "buyLeverage": buy_leverage,
                "sellLeverage": sell_leverage
            }
            
            self.logger.info(f"Setting leverage for {symbol}: buy={buy_leverage}, sell={sell_leverage}")
            
            response = await self._make_authenticated_request(
                "POST",
                "/v5/position/set-leverage",
                params
            )
            
            if response and response.get("retCode") == 0:
                leverage_data = response.get("result", {})
                self.logger.info(f"Leverage set successfully for {symbol}")
                return leverage_data
            else:
                error_msg = response.get("retMsg", "Unknown error") if response else "No response"
                self.logger.error(f"Failed to set leverage: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception setting leverage: {e}")
            return None


# Utility functions for common operations

async def create_bybit_client(api_key: str, api_secret: str, testnet: bool = True) -> BybitClient:
    """
    Create and initialize a BybitClient instance.
    
    Args:
        api_key: Bybit API key
        api_secret: Bybit API secret
        testnet: Whether to use testnet
        
    Returns:
        BybitClient: Initialized client
    """
    credentials = BybitCredentials(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )
    
    client = BybitClient(credentials)
    await client.connect()
    
    return client


async def fetch_historical_data(
    client: BybitClient,
    symbol: str,
    interval: str,
    days_back: int = 30
) -> Optional[MarketDataResponse]:
    """
    Fetch historical data for the specified number of days.
    
    Args:
        client: BybitClient instance
        symbol: Trading symbol
        interval: Kline interval
        days_back: Number of days to fetch
        
    Returns:
        Optional[MarketDataResponse]: Historical data or None if failed
    """
    end_time = int(time.time() * 1000)
    start_time = end_time - (days_back * 24 * 60 * 60 * 1000)
    
    return await client.get_kline_data(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        limit=1000
    )