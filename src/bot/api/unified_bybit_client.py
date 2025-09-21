"""
Unified Bybit API Client - Phase 3 API Consolidation

This module provides a unified, production-ready Bybit API client that consolidates
all scattered API implementations into a single, coherent system. It combines:

- REST API client with full v5 API support
- WebSocket connections for real-time data
- Connection pooling and session management
- Comprehensive error handling and retry logic
- Rate limiting compliance
- Market data, trading, and account operations
- Australian compliance features

Key Features:
- Unified interface for all Bybit operations
- Connection pooling for performance
- Automatic reconnection and failover
- Comprehensive logging and monitoring
- Type-safe responses with proper validation
"""

import asyncio
import hashlib
import hmac
import json
import time
import ssl
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Callable
from urllib.parse import urlencode
import logging
from enum import Enum
from dataclasses import dataclass, field
import uuid

import aiohttp
import websockets
import pandas as pd
from pydantic import BaseModel, Field, validator

# Import unified risk manager
try:
    from ..risk.core.unified_risk_manager import UnifiedRiskManager
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False

# Import unified configuration system
try:
    from ..core.config.manager import UnifiedConfigurationManager
    from ..core.config.schema import UnifiedConfigurationSchema
    from ..core.config.integrations import APISystemConfigAdapter
    UNIFIED_CONFIG_AVAILABLE = True
except ImportError:
    UNIFIED_CONFIG_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class Environment(Enum):
    """Trading environment types"""
    TESTNET = "testnet"
    MAINNET = "mainnet"

class ConnectionStatus(Enum):
    """Connection status types"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class OrderType(Enum):
    """Order types"""
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "StopLimit"

class OrderSide(Enum):
    """Order sides"""
    BUY = "Buy"
    SELL = "Sell"

class TimeInForce(Enum):
    """Time in force options"""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill

@dataclass
class BybitCredentials:
    """Bybit API credentials with validation"""
    api_key: str
    api_secret: str
    environment: Environment = Environment.TESTNET
    recv_window: int = 5000
    
    def __post_init__(self):
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret are required")
        
        if len(self.api_key) < 10 or len(self.api_secret) < 10:
            raise ValueError("Invalid API key or secret format")
    
    @property
    def base_url(self) -> str:
        """Get base URL for REST API"""
        if self.environment == Environment.TESTNET:
            return "https://api-testnet.bybit.com"
        else:
            return "https://api.bybit.com"
    
    @property
    def ws_url(self) -> str:
        """Get WebSocket URL"""
        if self.environment == Environment.TESTNET:
            return "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            return "wss://stream.bybit.com/v5/public/linear"
    
    @property
    def ws_private_url(self) -> str:
        """Get private WebSocket URL"""
        if self.environment == Environment.TESTNET:
            return "wss://stream-testnet.bybit.com/v5/private"
        else:
            return "wss://stream.bybit.com/v5/private"

@dataclass
class MarketData:
    """Standardized market data structure"""
    symbol: str
    price: Decimal
    bid: Decimal
    ask: Decimal
    volume_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal
    change_24h: Decimal
    timestamp: datetime
    source: str = "bybit"

@dataclass
class OrderBookData:
    """Order book data structure"""
    symbol: str
    bids: List[List[str]]  # [price, size]
    asks: List[List[str]]  # [price, size]
    timestamp: datetime
    update_id: int

@dataclass
class TradeData:
    """Trade data structure"""
    symbol: str
    trade_id: str
    price: Decimal
    quantity: Decimal
    side: OrderSide
    timestamp: datetime

@dataclass
class AccountBalance:
    """Account balance information"""
    coin: str
    wallet_balance: Decimal
    available_balance: Decimal
    locked_balance: Decimal

@dataclass
class Position:
    """Position information"""
    symbol: str
    side: str
    size: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal

@dataclass
class OrderResponse:
    """Order response data"""
    order_id: str
    order_link_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: Decimal
    price: Decimal
    status: str
    created_time: datetime

# ============================================================================
# RATE LIMITER
# ============================================================================

class BybitRateLimiter:
    """Rate limiter for Bybit API compliance"""
    
    def __init__(self):
        # Bybit rate limits (per second)
        self.limits = {
            'market_data': 120,  # 120 requests per second
            'trading': 50,       # 50 requests per second
            'account': 20,       # 20 requests per second
            'websocket': 10      # 10 connections per second
        }
        
        # Track request history
        self.request_history: Dict[str, List[float]] = {
            category: [] for category in self.limits.keys()
        }
        
        self._lock = asyncio.Lock()
    
    async def acquire(self, category: str = 'market_data'):
        """Acquire rate limit permission"""
        async with self._lock:
            now = time.time()
            
            # Clean old requests (older than 1 second)
            if category in self.request_history:
                self.request_history[category] = [
                    req_time for req_time in self.request_history[category]
                    if now - req_time < 1.0
                ]
            
            # Check if we're at the limit
            if len(self.request_history[category]) >= self.limits[category]:
                # Calculate wait time
                oldest_request = min(self.request_history[category])
                wait_time = 1.0 - (now - oldest_request)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.request_history[category].append(now)

# ============================================================================
# UNIFIED BYBIT CLIENT
# ============================================================================

class UnifiedBybitClient:
    """
    Unified Bybit API Client
    
    Consolidates all Bybit API functionality into a single, production-ready client
    """
    
    def __init__(
        self,
        credentials: BybitCredentials = None,
        enable_rate_limiting: bool = True,
        connection_pool_size: int = 100,
        request_timeout: int = 30,
        max_retries: int = 3,
        enable_websockets: bool = True,
        unified_config: 'UnifiedConfigurationSchema' = None
    ):
        # Load configuration (unified config takes precedence)
        if unified_config and UNIFIED_CONFIG_AVAILABLE:
            self.unified_config = unified_config
            config_dict = self._load_from_unified_config(unified_config)
            
            # Override with unified config values
            self.credentials = credentials or config_dict.get('credentials')
            self.enable_rate_limiting = config_dict.get('enable_rate_limiting', enable_rate_limiting)
            self.connection_pool_size = config_dict.get('connection_pool_size', connection_pool_size)
            self.request_timeout = config_dict.get('request_timeout', request_timeout)
            self.max_retries = config_dict.get('max_retries', max_retries)
            self.enable_websockets = config_dict.get('enable_websockets', enable_websockets)
            
            logger.info("UnifiedBybitClient initialized with unified configuration")
        else:
            self.unified_config = None
            self.credentials = credentials
            self.enable_rate_limiting = enable_rate_limiting
            self.connection_pool_size = connection_pool_size
            self.request_timeout = request_timeout
            self.max_retries = max_retries
            self.enable_websockets = enable_websockets
            
            logger.info("UnifiedBybitClient initialized with manual configuration")
        
        # Rate limiter
        self.rate_limiter = BybitRateLimiter() if enable_rate_limiting else None
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket connections
        self.ws_connections: Dict[str, Any] = {}
        self.ws_subscriptions: Dict[str, List[str]] = {}
        self.ws_status = ConnectionStatus.DISCONNECTED
        
        # Connection pooling
        self.connector: Optional[aiohttp.TCPConnector] = None
        
        # Callbacks for data streams
        self.market_data_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        self.order_callbacks: List[Callable] = []
        
        # Performance metrics
        self.request_count = 0
        self.error_count = 0
        self.last_request_time: Optional[datetime] = None
        
        # Risk manager integration
        self.risk_manager: Optional[UnifiedRiskManager] = None
        if RISK_MANAGER_AVAILABLE:
            try:
                self.risk_manager = UnifiedRiskManager()
            except Exception as e:
                logger.warning(f"Could not initialize risk manager: {e}")
                
    def _load_from_unified_config(self, unified_config: 'UnifiedConfigurationSchema') -> Dict[str, Any]:
        """Load configuration from unified configuration system"""
        try:
            adapter = APISystemConfigAdapter(unified_config)
            api_config = adapter.get_api_config()
            
            # Extract configuration values
            config = {
                'enable_rate_limiting': api_config.get('rate_limiting', {}).get('enabled', True),
                'connection_pool_size': api_config.get('connection_pool_size', 100),
                'request_timeout': api_config.get('request_timeout', 30),
                'max_retries': api_config.get('max_retries', 3),
                'enable_websockets': api_config.get('websockets', {}).get('enabled', True)
            }
            
            # Handle credentials from unified config if available
            exchange_config = adapter.get_exchange_config()
            if exchange_config and 'bybit' in exchange_config:
                bybit_config = exchange_config['bybit']
                if 'api_key' in bybit_config and 'api_secret' in bybit_config:
                    config['credentials'] = BybitCredentials(
                        api_key=bybit_config['api_key'],
                        api_secret=bybit_config['api_secret'],
                        testnet=bybit_config.get('testnet', False)
                    )
            
            return config
        except Exception as e:
            logger.error(f"Failed to load unified configuration: {e}")
            return {}
    
    def reload_configuration(self, unified_config: 'UnifiedConfigurationSchema' = None):
        """Reload configuration from unified configuration system"""
        if unified_config:
            self.unified_config = unified_config
            
        if self.unified_config:
            try:
                config_dict = self._load_from_unified_config(self.unified_config)
                
                # Update configuration values
                self.enable_rate_limiting = config_dict.get('enable_rate_limiting', self.enable_rate_limiting)
                self.connection_pool_size = config_dict.get('connection_pool_size', self.connection_pool_size)
                self.request_timeout = config_dict.get('request_timeout', self.request_timeout)
                self.max_retries = config_dict.get('max_retries', self.max_retries)
                self.enable_websockets = config_dict.get('enable_websockets', self.enable_websockets)
                
                # Update credentials if provided
                if 'credentials' in config_dict:
                    self.credentials = config_dict['credentials']
                
                logger.info("API client configuration reloaded from unified system")
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Initialize connections"""
        logger.info("Initializing Unified Bybit Client...")
        
        try:
            # Create connection pool
            self.connector = aiohttp.TCPConnector(
                limit=self.connection_pool_size,
                limit_per_host=50,
                keepalive_timeout=300,
                enable_cleanup_closed=True,
                ssl=ssl.create_default_context()
            )
            
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Bybit-TradingBot/1.0',
                    'Content-Type': 'application/json'
                }
            )
            
            # Test connection
            await self._test_connection()
            
            # Initialize WebSocket connections if enabled
            if self.enable_websockets:
                await self._initialize_websockets()
            
            logger.info("Unified Bybit Client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bybit client: {e}")
            await self.disconnect()
            raise
    
    async def disconnect(self):
        """Clean up connections"""
        logger.info("Disconnecting Unified Bybit Client...")
        
        try:
            # Close WebSocket connections
            await self._close_websockets()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
            
            # Close connector
            if self.connector:
                await self.connector.close()
                self.connector = None
            
            logger.info("Unified Bybit Client disconnected")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            response = await self._make_request('GET', '/v5/market/time')
            if response.get('retCode') == 0:
                logger.info("API connection test successful")
                return True
            else:
                raise Exception(f"API test failed: {response}")
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            raise
    
    def _generate_signature(self, timestamp: str, params: str = "") -> str:
        """Generate API signature"""
        param_str = f"{timestamp}{self.credentials.api_key}{self.credentials.recv_window}{params}"
        return hmac.new(
            self.credentials.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        category: str = 'market_data',
        authenticated: bool = False
    ) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and error handling"""
        
        if not self.session:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire(category)
        
        # Prepare request
        url = f"{self.credentials.base_url}{endpoint}"
        headers = {}
        
        if authenticated:
            timestamp = str(int(time.time() * 1000))
            
            if method == 'GET' and params:
                query_string = urlencode(params)
                url += f"?{query_string}"
                signature = self._generate_signature(timestamp, query_string)
            else:
                signature = self._generate_signature(timestamp)
            
            headers.update({
                'X-BAPI-API-KEY': self.credentials.api_key,
                'X-BAPI-TIMESTAMP': timestamp,
                'X-BAPI-RECV-WINDOW': str(self.credentials.recv_window),
                'X-BAPI-SIGN': signature
            })
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                self.request_count += 1
                self.last_request_time = datetime.now()
                
                if method == 'GET':
                    if not authenticated and params:
                        url += f"?{urlencode(params)}"
                    async with self.session.get(url, headers=headers) as response:
                        result = await response.json()
                elif method == 'POST':
                    async with self.session.post(url, json=params, headers=headers) as response:
                        result = await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Check for API errors
                if result.get('retCode') != 0:
                    error_msg = result.get('retMsg', 'Unknown API error')
                    logger.warning(f"API error: {error_msg}")
                    self.error_count += 1
                    
                    # Don't retry certain errors
                    if result.get('retCode') in [10001, 10003]:  # Invalid API key, signature errors
                        raise Exception(f"Authentication error: {error_msg}")
                    
                    if attempt == self.max_retries - 1:
                        raise Exception(f"API error after {self.max_retries} attempts: {error_msg}")
                else:
                    return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    self.error_count += 1
                    raise Exception("Request timeout after multiple attempts")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    self.error_count += 1
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    # ========================================================================
    # MARKET DATA METHODS
    # ========================================================================
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for a symbol"""
        response = await self._make_request(
            'GET',
            '/v5/market/tickers',
            params={'category': 'linear', 'symbol': symbol},
            category='market_data'
        )
        
        if not response.get('result', {}).get('list'):
            raise Exception(f"No market data found for {symbol}")
        
        data = response['result']['list'][0]
        
        return MarketData(
            symbol=data['symbol'],
            price=Decimal(data['lastPrice']),
            bid=Decimal(data['bid1Price']),
            ask=Decimal(data['ask1Price']),
            volume_24h=Decimal(data['volume24h']),
            high_24h=Decimal(data['highPrice24h']),
            low_24h=Decimal(data['lowPrice24h']),
            change_24h=Decimal(data['price24hPcnt']),
            timestamp=datetime.now()
        )
    
    async def get_orderbook(self, symbol: str, limit: int = 25) -> OrderBookData:
        """Get order book data"""
        response = await self._make_request(
            'GET',
            '/v5/market/orderbook',
            params={'category': 'linear', 'symbol': symbol, 'limit': limit},
            category='market_data'
        )
        
        result = response['result']
        
        return OrderBookData(
            symbol=result['s'],
            bids=result['b'],
            asks=result['a'],
            timestamp=datetime.fromtimestamp(int(result['ts']) / 1000),
            update_id=int(result['u'])
        )
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = '1',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get kline/candlestick data"""
        
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['start'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['end'] = int(end_time.timestamp() * 1000)
        
        response = await self._make_request(
            'GET',
            '/v5/market/kline',
            params=params,
            category='market_data'
        )
        
        klines = response['result']['list']
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        return df.sort_values('timestamp').reset_index(drop=True)
    
    # ========================================================================
    # TRADING METHODS
    # ========================================================================
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: Decimal,
        price: Optional[Decimal] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        order_link_id: Optional[str] = None
    ) -> OrderResponse:
        """Place a trading order"""
        
        # Risk management check if available
        if self.risk_manager:
            try:
                risk_check = await self.risk_manager.validate_trade(
                    symbol=symbol,
                    side=side.value,
                    quantity=float(qty),
                    price=float(price) if price else None
                )
                
                if not risk_check.get('approved', False):
                    raise Exception(f"Trade rejected by risk manager: {risk_check.get('reason')}")
                    
            except Exception as e:
                logger.error(f"Risk management check failed: {e}")
                raise
        
        params = {
            'category': 'linear',
            'symbol': symbol,
            'side': side.value,
            'orderType': order_type.value,
            'qty': str(qty),
            'timeInForce': time_in_force.value
        }
        
        if price and order_type != OrderType.MARKET:
            params['price'] = str(price)
        
        if order_link_id:
            params['orderLinkId'] = order_link_id
        else:
            params['orderLinkId'] = f"order_{uuid.uuid4().hex[:12]}"
        
        response = await self._make_request(
            'POST',
            '/v5/order/create',
            params=params,
            category='trading',
            authenticated=True
        )
        
        result = response['result']
        
        return OrderResponse(
            order_id=result['orderId'],
            order_link_id=result['orderLinkId'],
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            price=price or Decimal('0'),
            status='Submitted',
            created_time=datetime.now()
        )
    
    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None
    ) -> bool:
        """Cancel an order"""
        
        if not order_id and not order_link_id:
            raise ValueError("Either order_id or order_link_id must be provided")
        
        params = {
            'category': 'linear',
            'symbol': symbol
        }
        
        if order_id:
            params['orderId'] = order_id
        if order_link_id:
            params['orderLinkId'] = order_link_id
        
        response = await self._make_request(
            'POST',
            '/v5/order/cancel',
            params=params,
            category='trading',
            authenticated=True
        )
        
        return response.get('retCode') == 0
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders"""
        
        params = {'category': 'linear'}
        if symbol:
            params['symbol'] = symbol
        
        response = await self._make_request(
            'GET',
            '/v5/order/realtime',
            params=params,
            category='trading',
            authenticated=True
        )
        
        return response.get('result', {}).get('list', [])
    
    # ========================================================================
    # ACCOUNT METHODS
    # ========================================================================
    
    async def get_account_balance(self) -> List[AccountBalance]:
        """Get account balance"""
        
        response = await self._make_request(
            'GET',
            '/v5/account/wallet-balance',
            params={'accountType': 'UNIFIED'},
            category='account',
            authenticated=True
        )
        
        balances = []
        for account in response.get('result', {}).get('list', []):
            for coin in account.get('coin', []):
                balances.append(AccountBalance(
                    coin=coin['coin'],
                    wallet_balance=Decimal(coin['walletBalance']),
                    available_balance=Decimal(coin['availableToWithdraw']),
                    locked_balance=Decimal(coin['locked'])
                ))
        
        return balances
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get position information"""
        
        params = {'category': 'linear'}
        if symbol:
            params['symbol'] = symbol
        
        response = await self._make_request(
            'GET',
            '/v5/position/list',
            params=params,
            category='account',
            authenticated=True
        )
        
        positions = []
        for pos in response.get('result', {}).get('list', []):
            if float(pos['size']) != 0:  # Only include positions with size > 0
                positions.append(Position(
                    symbol=pos['symbol'],
                    side=pos['side'],
                    size=Decimal(pos['size']),
                    entry_price=Decimal(pos['avgPrice']),
                    mark_price=Decimal(pos['markPrice']),
                    unrealized_pnl=Decimal(pos['unrealisedPnl']),
                    realized_pnl=Decimal(pos['cumRealisedPnl'])
                ))
        
        return positions
    
    # ========================================================================
    # WEBSOCKET METHODS
    # ========================================================================
    
    async def _initialize_websockets(self):
        """Initialize WebSocket connections"""
        try:
            logger.info("Initializing WebSocket connections...")
            
            # Start WebSocket connection tasks
            asyncio.create_task(self._maintain_public_ws())
            if self.credentials.api_key:  # Only if authenticated
                asyncio.create_task(self._maintain_private_ws())
            
            self.ws_status = ConnectionStatus.CONNECTED
            logger.info("WebSocket connections initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket connections: {e}")
            self.ws_status = ConnectionStatus.ERROR
    
    async def _maintain_public_ws(self):
        """Maintain public WebSocket connection"""
        while True:
            try:
                async with websockets.connect(
                    self.credentials.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    
                    logger.info("Public WebSocket connected")
                    self.ws_connections['public'] = websocket
                    
                    # Handle incoming messages
                    async for message in websocket:
                        await self._handle_ws_message(json.loads(message), 'public')
                        
            except Exception as e:
                logger.error(f"Public WebSocket error: {e}")
                self.ws_status = ConnectionStatus.RECONNECTING
                await asyncio.sleep(5)  # Wait before reconnecting
    
    async def _maintain_private_ws(self):
        """Maintain private WebSocket connection"""
        while True:
            try:
                async with websockets.connect(
                    self.credentials.ws_private_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    
                    # Authenticate
                    await self._authenticate_private_ws(websocket)
                    
                    logger.info("Private WebSocket connected and authenticated")
                    self.ws_connections['private'] = websocket
                    
                    # Handle incoming messages
                    async for message in websocket:
                        await self._handle_ws_message(json.loads(message), 'private')
                        
            except Exception as e:
                logger.error(f"Private WebSocket error: {e}")
                self.ws_status = ConnectionStatus.RECONNECTING
                await asyncio.sleep(5)  # Wait before reconnecting
    
    async def _authenticate_private_ws(self, websocket):
        """Authenticate private WebSocket connection"""
        expires = int((time.time() + 60) * 1000)  # 60 seconds from now
        signature = hmac.new(
            self.credentials.api_secret.encode('utf-8'),
            f"GET/realtime{expires}".encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        auth_message = {
            "op": "auth",
            "args": [self.credentials.api_key, expires, signature]
        }
        
        await websocket.send(json.dumps(auth_message))
        
        # Wait for auth response
        response = await websocket.recv()
        auth_result = json.loads(response)
        
        if not auth_result.get('success'):
            raise Exception("WebSocket authentication failed")
    
    async def _handle_ws_message(self, message: Dict[str, Any], connection_type: str):
        """Handle incoming WebSocket messages"""
        try:
            topic = message.get('topic', '')
            
            if 'ticker' in topic:
                # Market data update
                for callback in self.market_data_callbacks:
                    asyncio.create_task(callback(message))
            
            elif 'trade' in topic:
                # Trade update
                for callback in self.trade_callbacks:
                    asyncio.create_task(callback(message))
            
            elif 'order' in topic or 'execution' in topic:
                # Order update
                for callback in self.order_callbacks:
                    asyncio.create_task(callback(message))
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _close_websockets(self):
        """Close all WebSocket connections"""
        for ws_name, ws in self.ws_connections.items():
            try:
                if ws and not ws.closed:
                    await ws.close()
                    logger.info(f"Closed {ws_name} WebSocket")
            except Exception as e:
                logger.error(f"Error closing {ws_name} WebSocket: {e}")
        
        self.ws_connections.clear()
        self.ws_status = ConnectionStatus.DISCONNECTED
    
    def add_market_data_callback(self, callback: Callable):
        """Add callback for market data updates"""
        self.market_data_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable):
        """Add callback for trade updates"""
        self.trade_callbacks.append(callback)
    
    def add_order_callback(self, callback: Callable):
        """Add callback for order updates"""
        self.order_callbacks.append(callback)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status and metrics"""
        return {
            'rest_api_connected': self.session is not None and not self.session.closed,
            'websocket_status': self.ws_status.value,
            'active_ws_connections': list(self.ws_connections.keys()),
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.request_count),
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        try:
            # Test REST API
            await self._test_connection()
            health['components']['rest_api'] = 'healthy'
        except Exception as e:
            health['components']['rest_api'] = f'unhealthy: {str(e)}'
            health['overall_status'] = 'degraded'
        
        # Check WebSocket status
        if self.enable_websockets:
            if self.ws_status == ConnectionStatus.CONNECTED:
                health['components']['websockets'] = 'healthy'
            else:
                health['components']['websockets'] = f'unhealthy: {self.ws_status.value}'
                health['overall_status'] = 'degraded'
        
        # Add metrics
        health.update(self.get_connection_status())
        
        return health

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_unified_bybit_client(
    api_key: str,
    api_secret: str,
    environment: Environment = Environment.TESTNET,
    **kwargs
) -> UnifiedBybitClient:
    """Factory function to create unified Bybit client"""
    
    credentials = BybitCredentials(
        api_key=api_key,
        api_secret=api_secret,
        environment=environment
    )
    
    return UnifiedBybitClient(credentials, **kwargs)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'UnifiedBybitClient',
    'BybitCredentials',
    'Environment',
    'ConnectionStatus',
    'OrderType',
    'OrderSide',
    'TimeInForce',
    'MarketData',
    'OrderBookData',
    'TradeData',
    'AccountBalance',
    'Position',
    'OrderResponse',
    'create_unified_bybit_client',
    'BybitRateLimiter'
]