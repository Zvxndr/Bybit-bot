"""
Unit Tests for Bybit API Client

This module contains comprehensive unit tests for the BybitClient
to ensure all API integration functionality works correctly.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import aiohttp

from src.bot.exchange.bybit_client import (
    BybitClient,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    BybitAPIError,
    RateLimitError,
    InsufficientBalanceError,
    OrderNotFoundError
)
from tests.conftest import (
    MockConfigurationManager,
    create_test_trade_request,
    async_test,
    generate_api_response
)


class TestBybitClient:
    """Test suite for BybitClient."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.client = BybitClient(self.config_manager)
        
        # Mock credentials
        self.client.api_key = "test_api_key"
        self.client.api_secret = "test_api_secret"
    
    def test_initialization(self):
        """Test client initialization."""
        assert self.client.base_url == "https://api.bybit.com"
        assert self.client.api_key == "test_api_key"
        assert self.client.api_secret == "test_api_secret"
        assert self.client.testnet is False
        assert self.client.session is None
        assert self.client.rate_limiter is not None
    
    def test_testnet_initialization(self):
        """Test testnet initialization."""
        config = MockConfigurationManager()
        config.config["exchange"]["testnet"] = True
        
        testnet_client = BybitClient(config)
        testnet_client.api_key = "test_key"
        testnet_client.api_secret = "test_secret"
        
        assert testnet_client.testnet is True
        assert testnet_client.base_url == "https://api-testnet.bybit.com"
    
    def test_signature_generation(self):
        """Test API signature generation."""
        timestamp = "1658983934791"
        params = "category=spot&symbol=BTCUSDT"
        
        signature = self.client._generate_signature(timestamp, params)
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # HMAC-SHA256 produces 64 character hex string
    
    def test_authentication_headers(self):
        """Test authentication headers generation."""
        with patch('time.time', return_value=1658983934.791):
            headers = self.client._get_auth_headers("GET", "/v5/order/list", "category=spot")
            
            assert "X-BAPI-API-KEY" in headers
            assert "X-BAPI-TIMESTAMP" in headers
            assert "X-BAPI-SIGN" in headers
            assert "X-BAPI-RECV-WINDOW" in headers
            assert headers["X-BAPI-API-KEY"] == "test_api_key"
            assert headers["X-BAPI-TIMESTAMP"] == "1658983934791"
    
    @async_test
    async def test_session_creation(self):
        """Test HTTP session creation."""
        await self.client._create_session()
        
        assert self.client.session is not None
        assert isinstance(self.client.session, aiohttp.ClientSession)
        
        # Clean up
        await self.client.close()
    
    @async_test
    async def test_session_cleanup(self):
        """Test HTTP session cleanup."""
        await self.client._create_session()
        session = self.client.session
        
        await self.client.close()
        
        assert session.closed is True
        assert self.client.session is None
    
    @async_test
    async def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Mock rate limiter
        rate_limiter = Mock()
        rate_limiter.acquire = AsyncMock()
        self.client.rate_limiter = rate_limiter
        
        # Mock successful API response
        mock_response = generate_api_response({"result": {"test": "data"}})
        
        with patch.object(self.client, '_make_request', return_value=mock_response):
            result = await self.client._api_request("GET", "/test", {})
            
            rate_limiter.acquire.assert_called_once()
            assert result == {"test": "data"}
    
    @async_test
    async def test_api_request_success(self):
        """Test successful API request."""
        mock_response_data = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "symbol": "BTCUSDT",
                "price": "50000.00"
            }
        }
        
        with patch.object(self.client, '_make_request', return_value=mock_response_data):
            result = await self.client._api_request("GET", "/v5/market/tickers", {"category": "spot"})
            
            assert result == mock_response_data["result"]
    
    @async_test
    async def test_api_request_error_handling(self):
        """Test API request error handling."""
        # Test API error response
        error_response = {
            "retCode": 10001,
            "retMsg": "Invalid API key",
            "result": {}
        }
        
        with patch.object(self.client, '_make_request', return_value=error_response):
            with pytest.raises(BybitAPIError) as exc_info:
                await self.client._api_request("GET", "/v5/account/wallet-balance", {})
            
            assert "Invalid API key" in str(exc_info.value)
            assert exc_info.value.code == 10001
    
    @async_test
    async def test_rate_limit_error_handling(self):
        """Test rate limit error handling."""
        error_response = {
            "retCode": 10006,
            "retMsg": "Too many requests",
            "result": {}
        }
        
        with patch.object(self.client, '_make_request', return_value=error_response):
            with pytest.raises(RateLimitError):
                await self.client._api_request("GET", "/v5/market/tickers", {})
    
    @async_test
    async def test_insufficient_balance_error(self):
        """Test insufficient balance error handling."""
        error_response = {
            "retCode": 110007,
            "retMsg": "Insufficient available balance",
            "result": {}
        }
        
        with patch.object(self.client, '_make_request', return_value=error_response):
            with pytest.raises(InsufficientBalanceError):
                await self.client._api_request("POST", "/v5/order/create", {})
    
    @async_test
    async def test_order_not_found_error(self):
        """Test order not found error handling."""
        error_response = {
            "retCode": 110001,
            "retMsg": "Order does not exist",
            "result": {}
        }
        
        with patch.object(self.client, '_make_request', return_value=error_response):
            with pytest.raises(OrderNotFoundError):
                await self.client._api_request("POST", "/v5/order/cancel", {})
    
    @async_test
    async def test_network_error_retry(self):
        """Test network error retry mechanism."""
        # First call fails, second succeeds
        mock_response = generate_api_response({"result": {"success": True}})
        
        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.side_effect = [
                aiohttp.ClientError("Network error"),
                mock_response
            ]
            
            result = await self.client._api_request("GET", "/v5/market/tickers", {})
            
            assert mock_request.call_count == 2
            assert result == {"success": True}
    
    @async_test
    async def test_place_order_market(self):
        """Test placing market order."""
        mock_response = {
            "orderId": "test-order-123",
            "orderLinkId": "custom-link-456"
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            order_id = await self.client.place_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=Decimal('0.1')
            )
            
            assert order_id == "test-order-123"
    
    @async_test
    async def test_place_order_limit(self):
        """Test placing limit order."""
        mock_response = {
            "orderId": "test-limit-order-789",
            "orderLinkId": "limit-link-123"
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            order_id = await self.client.place_order(
                symbol="ETHUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                qty=Decimal('1.0'),
                price=Decimal('3000.50')
            )
            
            assert order_id == "test-limit-order-789"
    
    @async_test
    async def test_place_order_with_stops(self):
        """Test placing order with stop loss and take profit."""
        mock_response = {
            "orderId": "test-stops-order-456",
            "orderLinkId": "stops-link-789"
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            order_id = await self.client.place_order(
                symbol="ADAUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                qty=Decimal('1000'),
                price=Decimal('0.45'),
                stop_loss=Decimal('0.40'),
                take_profit=Decimal('0.55')
            )
            
            assert order_id == "test-stops-order-456"
    
    @async_test
    async def test_cancel_order(self):
        """Test canceling order."""
        mock_response = {
            "orderId": "cancelled-order-123",
            "orderLinkId": "cancelled-link-456"
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            result = await self.client.cancel_order(
                symbol="BTCUSDT",
                order_id="cancelled-order-123"
            )
            
            assert result is True
    
    @async_test
    async def test_cancel_order_not_found(self):
        """Test canceling non-existent order."""
        with patch.object(self.client, '_api_request', side_effect=OrderNotFoundError("Order not found")):
            with pytest.raises(OrderNotFoundError):
                await self.client.cancel_order(
                    symbol="BTCUSDT",
                    order_id="non-existent-order"
                )
    
    @async_test
    async def test_modify_order(self):
        """Test modifying order."""
        mock_response = {
            "orderId": "modified-order-123",
            "orderLinkId": "modified-link-456"
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            result = await self.client.modify_order(
                symbol="ETHUSDT",
                order_id="modified-order-123",
                qty=Decimal('2.0'),
                price=Decimal('2950.00')
            )
            
            assert result is True
    
    @async_test
    async def test_get_order_status(self):
        """Test getting order status."""
        mock_response = {
            "list": [{
                "orderId": "test-order-123",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "qty": "0.1",
                "price": "50000.00",
                "orderStatus": "Filled",
                "avgPrice": "49950.00",
                "cumExecQty": "0.1",
                "timeInForce": "GTC",
                "createdTime": "1658983934791",
                "updatedTime": "1658984000000"
            }]
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            order = await self.client.get_order_status(
                symbol="BTCUSDT",
                order_id="test-order-123"
            )
            
            assert order["orderId"] == "test-order-123"
            assert order["orderStatus"] == "Filled"
            assert order["avgPrice"] == "49950.00"
    
    @async_test
    async def test_get_open_orders(self):
        """Test getting open orders."""
        mock_response = {
            "list": [
                {
                    "orderId": "open-order-1",
                    "symbol": "BTCUSDT",
                    "orderStatus": "New",
                    "side": "Buy",
                    "qty": "0.1"
                },
                {
                    "orderId": "open-order-2",
                    "symbol": "ETHUSDT",
                    "orderStatus": "PartiallyFilled",
                    "side": "Sell",
                    "qty": "1.0"
                }
            ]
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            orders = await self.client.get_open_orders("BTCUSDT")
            
            assert len(orders) == 2
            assert orders[0]["orderId"] == "open-order-1"
            assert orders[1]["orderStatus"] == "PartiallyFilled"
    
    @async_test
    async def test_get_positions(self):
        """Test getting positions."""
        mock_response = {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "size": "0.1",
                    "positionValue": "5000.00",
                    "entryPrice": "50000.00",
                    "markPrice": "50100.00",
                    "unrealisedPnl": "10.00",
                    "leverage": "2"
                },
                {
                    "symbol": "ETHUSDT",
                    "side": "Sell",
                    "size": "1.0",
                    "positionValue": "3000.00",
                    "entryPrice": "3000.00",
                    "markPrice": "2995.00",
                    "unrealisedPnl": "5.00",
                    "leverage": "1"
                }
            ]
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            positions = await self.client.get_positions()
            
            assert len(positions) == 2
            assert positions[0]["symbol"] == "BTCUSDT"
            assert positions[1]["unrealisedPnl"] == "5.00"
    
    @async_test
    async def test_get_single_position(self):
        """Test getting single position."""
        mock_response = {
            "list": [{
                "symbol": "BTCUSDT",
                "side": "Buy",
                "size": "0.1",
                "positionValue": "5000.00",
                "entryPrice": "50000.00",
                "unrealisedPnl": "10.00"
            }]
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            position = await self.client.get_positions("BTCUSDT")
            
            assert len(position) == 1
            assert position[0]["symbol"] == "BTCUSDT"
    
    @async_test
    async def test_get_wallet_balance(self):
        """Test getting wallet balance."""
        mock_response = {
            "list": [{
                "accountType": "UNIFIED",
                "coin": [{
                    "coin": "USDT",
                    "walletBalance": "10000.00",
                    "availableBalance": "8500.00",
                    "locked": "1500.00"
                }, {
                    "coin": "BTC",
                    "walletBalance": "0.5",
                    "availableBalance": "0.3",
                    "locked": "0.2"
                }]
            }]
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            balance = await self.client.get_wallet_balance()
            
            assert "USDT" in balance
            assert "BTC" in balance
            assert balance["USDT"]["wallet_balance"] == Decimal('10000.00')
            assert balance["BTC"]["available_balance"] == Decimal('0.3')
    
    @async_test
    async def test_get_specific_balance(self):
        """Test getting specific coin balance."""
        mock_response = {
            "list": [{
                "accountType": "UNIFIED",
                "coin": [{
                    "coin": "USDT",
                    "walletBalance": "10000.00",
                    "availableBalance": "8500.00",
                    "locked": "1500.00"
                }]
            }]
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            balance = await self.client.get_wallet_balance("USDT")
            
            assert "USDT" in balance
            assert balance["USDT"]["wallet_balance"] == Decimal('10000.00')
    
    @async_test
    async def test_get_market_data(self):
        """Test getting market data."""
        mock_response = {
            "list": [{
                "symbol": "BTCUSDT",
                "lastPrice": "50000.00",
                "bid1Price": "49999.50",
                "ask1Price": "50000.50",
                "volume24h": "1000.5",
                "price24hPcnt": "0.0245"
            }]
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            ticker = await self.client.get_market_data("BTCUSDT")
            
            assert ticker["symbol"] == "BTCUSDT"
            assert ticker["lastPrice"] == "50000.00"
            assert ticker["price24hPcnt"] == "0.0245"
    
    @async_test
    async def test_get_all_tickers(self):
        """Test getting all tickers."""
        mock_response = {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "lastPrice": "50000.00",
                    "volume24h": "1000.5"
                },
                {
                    "symbol": "ETHUSDT",
                    "lastPrice": "3000.00",
                    "volume24h": "5000.2"
                }
            ]
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            tickers = await self.client.get_market_data()
            
            assert len(tickers) == 2
            assert tickers[0]["symbol"] == "BTCUSDT"
            assert tickers[1]["symbol"] == "ETHUSDT"
    
    @async_test
    async def test_get_klines(self):
        """Test getting kline data."""
        mock_response = {
            "list": [
                ["1658983800000", "50000.00", "50100.00", "49900.00", "50050.00", "10.5", "525000.00"],
                ["1658983740000", "49950.00", "50000.00", "49900.00", "50000.00", "8.2", "409500.00"]
            ]
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            klines = await self.client.get_klines(
                symbol="BTCUSDT",
                interval="1",
                limit=2
            )
            
            assert len(klines) == 2
            assert len(klines[0]) == 7  # timestamp, open, high, low, close, volume, turnover
    
    @async_test
    async def test_get_order_book(self):
        """Test getting order book."""
        mock_response = {
            "s": "BTCUSDT",
            "b": [
                ["49999.50", "0.1"],
                ["49999.00", "0.2"]
            ],
            "a": [
                ["50000.50", "0.15"],
                ["50001.00", "0.25"]
            ]
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            orderbook = await self.client.get_order_book("BTCUSDT", 5)
            
            assert orderbook["s"] == "BTCUSDT"
            assert len(orderbook["b"]) == 2  # bids
            assert len(orderbook["a"]) == 2  # asks
    
    @async_test
    async def test_get_trade_history(self):
        """Test getting trade history."""
        mock_response = {
            "list": [
                {
                    "orderId": "trade-1",
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "execQty": "0.1",
                    "execPrice": "50000.00",
                    "execTime": "1658983934791",
                    "execFee": "2.50"
                },
                {
                    "orderId": "trade-2",
                    "symbol": "ETHUSDT",
                    "side": "Sell",
                    "execQty": "1.0",
                    "execPrice": "3000.00",
                    "execTime": "1658983800000",
                    "execFee": "1.50"
                }
            ]
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            trades = await self.client.get_trade_history("BTCUSDT")
            
            assert len(trades) == 2
            assert trades[0]["symbol"] == "BTCUSDT"
            assert trades[1]["execFee"] == "1.50"
    
    @async_test
    async def test_get_server_time(self):
        """Test getting server time."""
        mock_response = {
            "timeSecond": "1658983934",
            "timeNano": "1658983934791000000"
        }
        
        with patch.object(self.client, '_api_request', return_value=mock_response):
            server_time = await self.client.get_server_time()
            
            assert server_time == 1658983934791
    
    @async_test
    async def test_websocket_connection(self):
        """Test WebSocket connection setup."""
        # Mock WebSocket connection
        mock_ws = AsyncMock()
        mock_ws.send_str = AsyncMock()
        mock_ws.receive = AsyncMock()
        
        with patch('aiohttp.ClientSession.ws_connect', return_value=mock_ws):
            ws = await self.client._connect_websocket()
            
            assert ws is not None
            mock_ws.send_str.assert_not_called()  # No immediate messages
    
    @async_test
    async def test_websocket_subscription(self):
        """Test WebSocket subscription."""
        mock_ws = AsyncMock()
        mock_ws.send_str = AsyncMock()
        
        subscription_message = {
            "op": "subscribe",
            "args": ["orderbook.1.BTCUSDT"]
        }
        
        await self.client._send_websocket_message(mock_ws, subscription_message)
        
        mock_ws.send_str.assert_called_once()
        call_args = json.loads(mock_ws.send_str.call_args[0][0])
        assert call_args["op"] == "subscribe"
        assert call_args["args"] == ["orderbook.1.BTCUSDT"]


class TestBybitErrorHandling:
    """Test suite for Bybit error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.client = BybitClient(self.config_manager)
    
    def test_bybit_api_error_creation(self):
        """Test BybitAPIError creation."""
        error = BybitAPIError("Test error message", 10001)
        
        assert str(error) == "Test error message"
        assert error.code == 10001
    
    def test_rate_limit_error_creation(self):
        """Test RateLimitError creation."""
        error = RateLimitError("Rate limit exceeded", 10006)
        
        assert "Rate limit exceeded" in str(error)
        assert error.code == 10006
    
    def test_insufficient_balance_error_creation(self):
        """Test InsufficientBalanceError creation."""
        error = InsufficientBalanceError("Insufficient balance", 110007)
        
        assert "Insufficient balance" in str(error)
        assert error.code == 110007
    
    def test_order_not_found_error_creation(self):
        """Test OrderNotFoundError creation."""
        error = OrderNotFoundError("Order not found", 110001)
        
        assert "Order not found" in str(error)
        assert error.code == 110001


class TestBybitEnums:
    """Test suite for Bybit enums."""
    
    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET == "Market"
        assert OrderType.LIMIT == "Limit"
        assert OrderType.STOP == "Stop"
        assert OrderType.STOP_LIMIT == "StopLimit"
    
    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY == "Buy"
        assert OrderSide.SELL == "Sell"
    
    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.NEW == "New"
        assert OrderStatus.PARTIALLY_FILLED == "PartiallyFilled"
        assert OrderStatus.FILLED == "Filled"
        assert OrderStatus.CANCELLED == "Cancelled"
        assert OrderStatus.REJECTED == "Rejected"
    
    def test_time_in_force_enum(self):
        """Test TimeInForce enum values."""
        assert TimeInForce.GTC == "GTC"
        assert TimeInForce.IOC == "IOC"
        assert TimeInForce.FOK == "FOK"


# Performance tests
class TestBybitClientPerformance:
    """Performance tests for Bybit client."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.client = BybitClient(self.config_manager)
        self.client.api_key = "test_key"
        self.client.api_secret = "test_secret"
    
    @async_test
    async def test_concurrent_api_requests(self):
        """Test concurrent API requests performance."""
        from tests.conftest import PerformanceTimer
        
        mock_response = generate_api_response({"result": {"success": True}})
        
        with PerformanceTimer("50 concurrent API requests"):
            with patch.object(self.client, '_make_request', return_value=mock_response):
                tasks = []
                for i in range(50):
                    task = self.client._api_request("GET", f"/v5/test/{i}", {})
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
        
        assert len(results) == 50
        assert all(r == {"success": True} for r in results)
    
    @async_test
    async def test_order_operations_performance(self):
        """Test order operations performance."""
        from tests.conftest import PerformanceTimer
        
        mock_order_response = {"orderId": "test-order", "orderLinkId": "test-link"}
        
        with PerformanceTimer("20 order placements"):
            with patch.object(self.client, '_api_request', return_value=mock_order_response):
                tasks = []
                for i in range(20):
                    task = self.client.place_order(
                        symbol=f"TEST{i}USDT",
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        qty=Decimal('0.1')
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
        
        assert len(results) == 20
        assert all(r == "test-order" for r in results)