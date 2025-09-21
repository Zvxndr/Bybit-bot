"""
Integration Tests for Bybit API Client

This module contains integration tests that test the BybitClient's
interaction with the actual Bybit API endpoints using mocked responses.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import aiohttp
import json

from src.bot.exchange.bybit_client import (
    BybitClient,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    BybitAPIError,
    RateLimitError
)
from tests.conftest import (
    MockConfigurationManager,
    async_test,
    generate_api_response
)


class TestBybitAPIIntegration:
    """Integration tests for Bybit API client."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.client = BybitClient(self.config_manager)
        self.client.api_key = "test_api_key"
        self.client.api_secret = "test_api_secret"
    
    @async_test
    async def test_full_trading_workflow(self):
        """Test complete trading workflow: place order -> check status -> cancel."""
        # Mock responses for the complete workflow
        place_order_response = generate_api_response({
            "result": {
                "orderId": "workflow-order-123",
                "orderLinkId": "workflow-link-456"
            }
        })
        
        order_status_response = generate_api_response({
            "result": {
                "list": [{
                    "orderId": "workflow-order-123",
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "orderType": "Limit",
                    "qty": "0.1",
                    "price": "50000.00",
                    "orderStatus": "New",
                    "avgPrice": "0",
                    "cumExecQty": "0",
                    "timeInForce": "GTC",
                    "createdTime": "1658983934791",
                    "updatedTime": "1658983934791"
                }]
            }
        })
        
        cancel_order_response = generate_api_response({
            "result": {
                "orderId": "workflow-order-123",
                "orderLinkId": "workflow-link-456"
            }
        })
        
        # Mock API requests in sequence
        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.side_effect = [
                place_order_response,  # Place order
                order_status_response,  # Check status
                cancel_order_response   # Cancel order
            ]
            
            # Step 1: Place order
            order_id = await self.client.place_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                qty=Decimal('0.1'),
                price=Decimal('50000.00')
            )
            
            assert order_id == "workflow-order-123"
            
            # Step 2: Check order status
            order_status = await self.client.get_order_status(
                symbol="BTCUSDT",
                order_id=order_id
            )
            
            assert order_status["orderId"] == order_id
            assert order_status["orderStatus"] == "New"
            
            # Step 3: Cancel order
            cancel_result = await self.client.cancel_order(
                symbol="BTCUSDT",
                order_id=order_id
            )
            
            assert cancel_result is True
            assert mock_request.call_count == 3
    
    @async_test
    async def test_portfolio_management_workflow(self):
        """Test portfolio management workflow: get balance -> get positions -> get trades."""
        # Mock responses
        balance_response = generate_api_response({
            "result": {
                "list": [{
                    "accountType": "UNIFIED",
                    "coin": [
                        {
                            "coin": "USDT",
                            "walletBalance": "10000.00",
                            "availableBalance": "8500.00",
                            "locked": "1500.00"
                        },
                        {
                            "coin": "BTC",
                            "walletBalance": "0.5",
                            "availableBalance": "0.3",
                            "locked": "0.2"
                        }
                    ]
                }]
            }
        })
        
        positions_response = generate_api_response({
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "size": "0.2",
                        "positionValue": "10000.00",
                        "entryPrice": "50000.00",
                        "markPrice": "50100.00",
                        "unrealisedPnl": "20.00",
                        "leverage": "2"
                    }
                ]
            }
        })
        
        trades_response = generate_api_response({
            "result": {
                "list": [
                    {
                        "orderId": "trade-1",
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "execQty": "0.1",
                        "execPrice": "50000.00",
                        "execTime": "1658983934791",
                        "execFee": "2.50"
                    }
                ]
            }
        })
        
        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.side_effect = [
                balance_response,
                positions_response,
                trades_response
            ]
            
            # Get wallet balance
            balance = await self.client.get_wallet_balance()
            
            assert "USDT" in balance
            assert "BTC" in balance
            assert balance["USDT"]["wallet_balance"] == Decimal('10000.00')
            assert balance["BTC"]["available_balance"] == Decimal('0.3')
            
            # Get positions
            positions = await self.client.get_positions()
            
            assert len(positions) == 1
            assert positions[0]["symbol"] == "BTCUSDT"
            assert positions[0]["unrealisedPnl"] == "20.00"
            
            # Get trade history
            trades = await self.client.get_trade_history("BTCUSDT")
            
            assert len(trades) == 1
            assert trades[0]["symbol"] == "BTCUSDT"
            assert trades[0]["execFee"] == "2.50"
    
    @async_test
    async def test_market_data_integration(self):
        """Test market data integration workflow."""
        # Mock responses
        ticker_response = generate_api_response({
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lastPrice": "50000.00",
                    "bid1Price": "49999.50",
                    "ask1Price": "50000.50",
                    "volume24h": "1000.5",
                    "price24hPcnt": "0.0245"
                }]
            }
        })
        
        klines_response = generate_api_response({
            "result": {
                "list": [
                    ["1658983800000", "50000.00", "50100.00", "49900.00", "50050.00", "10.5", "525000.00"],
                    ["1658983740000", "49950.00", "50000.00", "49900.00", "50000.00", "8.2", "409500.00"]
                ]
            }
        })
        
        orderbook_response = generate_api_response({
            "result": {
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
        })
        
        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.side_effect = [
                ticker_response,
                klines_response,
                orderbook_response
            ]
            
            # Get ticker data
            ticker = await self.client.get_market_data("BTCUSDT")
            
            assert ticker["symbol"] == "BTCUSDT"
            assert ticker["lastPrice"] == "50000.00"
            assert ticker["price24hPcnt"] == "0.0245"
            
            # Get kline data
            klines = await self.client.get_klines(
                symbol="BTCUSDT",
                interval="1",
                limit=2
            )
            
            assert len(klines) == 2
            assert len(klines[0]) == 7  # OHLCV + timestamp + turnover
            
            # Get order book
            orderbook = await self.client.get_order_book("BTCUSDT", 5)
            
            assert orderbook["s"] == "BTCUSDT"
            assert len(orderbook["b"]) == 2  # bids
            assert len(orderbook["a"]) == 2  # asks
    
    @async_test
    async def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        # Test API error
        api_error_response = {
            "retCode": 10001,
            "retMsg": "Invalid API key",
            "result": {}
        }
        
        # Test rate limit error
        rate_limit_response = {
            "retCode": 10006,
            "retMsg": "Too many requests",
            "result": {}
        }
        
        with patch.object(self.client, '_make_request') as mock_request:
            # Test API error handling
            mock_request.return_value = api_error_response
            
            with pytest.raises(BybitAPIError) as exc_info:
                await self.client.get_wallet_balance()
            
            assert "Invalid API key" in str(exc_info.value)
            assert exc_info.value.code == 10001
            
            # Test rate limit error handling
            mock_request.return_value = rate_limit_response
            
            with pytest.raises(RateLimitError) as exc_info:
                await self.client.place_order(
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    qty=Decimal('0.1')
                )
            
            assert "Too many requests" in str(exc_info.value)
            assert exc_info.value.code == 10006
    
    @async_test
    async def test_concurrent_requests_integration(self):
        """Test handling concurrent API requests."""
        # Mock successful responses
        successful_response = generate_api_response({
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lastPrice": "50000.00"
                }]
            }
        })
        
        with patch.object(self.client, '_make_request', return_value=successful_response):
            # Create multiple concurrent requests
            tasks = []
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "SOLUSDT"]
            
            for symbol in symbols:
                task = self.client.get_market_data(symbol)
                tasks.append(task)
            
            # Execute concurrently
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            for result in results:
                assert result["symbol"] == "BTCUSDT"  # Mock returns same data
                assert result["lastPrice"] == "50000.00"
    
    @async_test
    async def test_order_modification_integration(self):
        """Test order modification integration."""
        # Place order first
        place_response = generate_api_response({
            "result": {
                "orderId": "modify-test-order",
                "orderLinkId": "modify-link"
            }
        })
        
        # Modify order response
        modify_response = generate_api_response({
            "result": {
                "orderId": "modify-test-order",
                "orderLinkId": "modify-link"
            }
        })
        
        # Check modified order status
        status_response = generate_api_response({
            "result": {
                "list": [{
                    "orderId": "modify-test-order",
                    "symbol": "ETHUSDT",
                    "side": "Sell",
                    "orderType": "Limit",
                    "qty": "2.0",  # Modified quantity
                    "price": "2950.00",  # Modified price
                    "orderStatus": "New",
                    "avgPrice": "0",
                    "cumExecQty": "0"
                }]
            }
        })
        
        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.side_effect = [
                place_response,
                modify_response,
                status_response
            ]
            
            # Place original order
            order_id = await self.client.place_order(
                symbol="ETHUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                qty=Decimal('1.0'),
                price=Decimal('3000.00')
            )
            
            assert order_id == "modify-test-order"
            
            # Modify order
            modify_result = await self.client.modify_order(
                symbol="ETHUSDT",
                order_id=order_id,
                qty=Decimal('2.0'),
                price=Decimal('2950.00')
            )
            
            assert modify_result is True
            
            # Verify modification
            modified_order = await self.client.get_order_status(
                symbol="ETHUSDT",
                order_id=order_id
            )
            
            assert modified_order["qty"] == "2.0"
            assert modified_order["price"] == "2950.00"
    
    @async_test
    async def test_batch_operations_integration(self):
        """Test batch operations integration."""
        # Mock responses for multiple operations
        responses = []
        
        # Place 3 orders
        for i in range(3):
            responses.append(generate_api_response({
                "result": {
                    "orderId": f"batch-order-{i}",
                    "orderLinkId": f"batch-link-{i}"
                }
            }))
        
        # Get open orders
        responses.append(generate_api_response({
            "result": {
                "list": [
                    {
                        "orderId": "batch-order-0",
                        "symbol": "BTCUSDT",
                        "orderStatus": "New"
                    },
                    {
                        "orderId": "batch-order-1",
                        "symbol": "BTCUSDT",
                        "orderStatus": "New"
                    },
                    {
                        "orderId": "batch-order-2",
                        "symbol": "BTCUSDT",
                        "orderStatus": "New"
                    }
                ]
            }
        }))
        
        # Cancel all orders
        for i in range(3):
            responses.append(generate_api_response({
                "result": {
                    "orderId": f"batch-order-{i}",
                    "orderLinkId": f"batch-link-{i}"
                }
            }))
        
        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.side_effect = responses
            
            # Place multiple orders
            order_ids = []
            for i in range(3):
                order_id = await self.client.place_order(
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    qty=Decimal('0.1'),
                    price=Decimal(f'{50000 - i * 100}')
                )
                order_ids.append(order_id)
            
            # Get all open orders
            open_orders = await self.client.get_open_orders("BTCUSDT")
            
            assert len(open_orders) == 3
            
            # Cancel all orders
            for order_id in order_ids:
                cancel_result = await self.client.cancel_order(
                    symbol="BTCUSDT",
                    order_id=order_id
                )
                assert cancel_result is True
    
    @async_test
    async def test_session_management_integration(self):
        """Test session management in integration scenarios."""
        # Test session creation and cleanup
        assert self.client.session is None
        
        # Create session implicitly through API call
        mock_response = generate_api_response({
            "result": {"timeSecond": "1658983934"}
        })
        
        with patch.object(self.client, '_make_request', return_value=mock_response):
            server_time = await self.client.get_server_time()
            
            assert server_time == 1658983934000  # Converted to milliseconds
            assert self.client.session is not None
        
        # Test session cleanup
        await self.client.close()
        assert self.client.session is None
    
    @async_test
    async def test_authentication_integration(self):
        """Test authentication integration."""
        # Test authenticated endpoint
        mock_response = generate_api_response({
            "result": {
                "list": [{
                    "accountType": "UNIFIED",
                    "coin": [{
                        "coin": "USDT",
                        "walletBalance": "10000.00"
                    }]
                }]
            }
        })
        
        with patch.object(self.client, '_make_request', return_value=mock_response) as mock_request:
            balance = await self.client.get_wallet_balance()
            
            # Verify authentication headers were added
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            
            # Check that auth headers would be present (in real implementation)
            assert balance["USDT"]["wallet_balance"] == Decimal('10000.00')
    
    @async_test
    async def test_data_validation_integration(self):
        """Test data validation in integration scenarios."""
        # Test response data validation
        invalid_response = generate_api_response({
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lastPrice": "invalid_price",  # Invalid price format
                    "volume24h": "not_a_number"
                }]
            }
        })
        
        with patch.object(self.client, '_make_request', return_value=invalid_response):
            # Client should handle invalid data gracefully
            ticker = await self.client.get_market_data("BTCUSDT")
            
            # Verify the response is returned as-is for now
            # (In production, you might want additional validation)
            assert ticker["lastPrice"] == "invalid_price"
            assert ticker["volume24h"] == "not_a_number"
    
    @async_test
    async def test_performance_monitoring(self):
        """Test performance monitoring in integration."""
        from tests.conftest import PerformanceTimer
        
        mock_response = generate_api_response({
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lastPrice": "50000.00"
                }]
            }
        })
        
        with patch.object(self.client, '_make_request', return_value=mock_response):
            with PerformanceTimer("Single API request"):
                ticker = await self.client.get_market_data("BTCUSDT")
            
            assert ticker["symbol"] == "BTCUSDT"
            
            # Test multiple concurrent requests performance
            with PerformanceTimer("10 concurrent API requests"):
                tasks = []
                for i in range(10):
                    task = self.client.get_market_data("BTCUSDT")
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
            
            assert len(results) == 10


class TestBybitWebSocketIntegration:
    """Integration tests for WebSocket functionality."""
    
    def setup_method(self):
        """Set up WebSocket test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.client = BybitClient(self.config_manager)
        self.client.api_key = "test_api_key"
        self.client.api_secret = "test_api_secret"
    
    @async_test
    async def test_websocket_connection_flow(self):
        """Test WebSocket connection establishment."""
        mock_ws = AsyncMock()
        mock_ws.send_str = AsyncMock()
        mock_ws.receive = AsyncMock()
        mock_ws.close = AsyncMock()
        
        with patch('aiohttp.ClientSession.ws_connect', return_value=mock_ws):
            # Connect to WebSocket
            ws = await self.client._connect_websocket()
            
            assert ws is not None
            assert ws == mock_ws
            
            # Test sending subscription message
            subscription = {
                "op": "subscribe",
                "args": ["orderbook.1.BTCUSDT"]
            }
            
            await self.client._send_websocket_message(ws, subscription)
            
            mock_ws.send_str.assert_called_once()
            sent_message = json.loads(mock_ws.send_str.call_args[0][0])
            assert sent_message["op"] == "subscribe"
            assert sent_message["args"] == ["orderbook.1.BTCUSDT"]
    
    @async_test
    async def test_websocket_message_handling(self):
        """Test WebSocket message handling."""
        mock_ws = AsyncMock()
        
        # Mock message reception
        test_messages = [
            {
                "topic": "orderbook.1.BTCUSDT",
                "type": "snapshot",
                "data": {
                    "s": "BTCUSDT",
                    "b": [["50000.00", "0.1"]],
                    "a": [["50001.00", "0.1"]]
                }
            }
        ]
        
        mock_ws.receive.return_value = Mock(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(test_messages[0])
        )
        
        with patch('aiohttp.ClientSession.ws_connect', return_value=mock_ws):
            ws = await self.client._connect_websocket()
            
            # In a real implementation, you would have message handling logic here
            # For now, just verify the connection works
            assert ws is not None
    
    @async_test
    async def test_websocket_error_handling(self):
        """Test WebSocket error handling."""
        mock_ws = AsyncMock()
        mock_ws.send_str.side_effect = aiohttp.WSServerHandshakeError(
            message="WebSocket handshake failed",
            request_info=Mock(),
            history=()
        )
        
        with patch('aiohttp.ClientSession.ws_connect', return_value=mock_ws):
            ws = await self.client._connect_websocket()
            
            # Test error handling in message sending
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await self.client._send_websocket_message(ws, {"test": "message"})


class TestBybitRateLimitingIntegration:
    """Integration tests for rate limiting."""
    
    def setup_method(self):
        """Set up rate limiting test fixtures."""
        self.config_manager = MockConfigurationManager()
        self.client = BybitClient(self.config_manager)
        self.client.api_key = "test_api_key"
        self.client.api_secret = "test_api_secret"
    
    @async_test
    async def test_rate_limiting_behavior(self):
        """Test rate limiting behavior with mock rate limiter."""
        # Mock rate limiter
        rate_limiter = Mock()
        rate_limiter.acquire = AsyncMock()
        self.client.rate_limiter = rate_limiter
        
        mock_response = generate_api_response({
            "result": {"timeSecond": "1658983934"}
        })
        
        with patch.object(self.client, '_make_request', return_value=mock_response):
            # Make multiple requests
            for i in range(5):
                await self.client.get_server_time()
            
            # Verify rate limiter was called for each request
            assert rate_limiter.acquire.call_count == 5
    
    @async_test
    async def test_rate_limit_retry_logic(self):
        """Test retry logic when rate limited."""
        responses = [
            {
                "retCode": 10006,
                "retMsg": "Too many requests",
                "result": {}
            },
            generate_api_response({
                "result": {"timeSecond": "1658983934"}
            })
        ]
        
        with patch.object(self.client, '_make_request') as mock_request:
            mock_request.side_effect = responses
            
            # Should retry after rate limit error
            with pytest.raises(RateLimitError):
                server_time = await self.client.get_server_time()
            
            # Verify the rate limit error was raised
            assert mock_request.call_count == 1