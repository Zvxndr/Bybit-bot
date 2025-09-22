"""
WebSocket Manager for Real-time Market Data and Trading Streams

This module provides comprehensive WebSocket connectivity for Bybit's real-time data feeds:
- Market data streams (tickers, klines, orderbook, trades)
- Private streams (orders, positions, executions, wallet)
- Connection management with automatic reconnection
- Message validation and error handling
- Rate limiting and connection health monitoring

Supports both public market data and authenticated private streams for live trading.

Author: Trading Bot Team
Version: 1.0.0 - Phase 5 Implementation
"""

import asyncio
import json
import time
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from ..utils.logging import TradingLogger
from ..config_manager import ConfigurationManager
from ..exchange.bybit_client import BybitCredentials


class WebSocketStreamType(Enum):
    """WebSocket stream types."""
    PUBLIC = "public"
    PRIVATE = "private"


class ConnectionStatus(Enum):
    """WebSocket connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class WebSocketConfig:
    """WebSocket configuration settings."""
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 5.0
    ping_interval: int = 20
    ping_timeout: int = 10
    message_timeout: int = 30
    max_message_size: int = 1024 * 1024  # 1MB
    heartbeat_interval: int = 20
    
    # Bybit WebSocket URLs
    public_url_testnet: str = "wss://stream-testnet.bybit.com/v5/public/linear"
    private_url_testnet: str = "wss://stream-testnet.bybit.com/v5/private"
    public_url_mainnet: str = "wss://stream.bybit.com/v5/public/linear"
    private_url_mainnet: str = "wss://stream.bybit.com/v5/private"


@dataclass
class WebSocketMessage:
    """Structured WebSocket message."""
    topic: str
    data: Dict[str, Any]
    timestamp: datetime
    stream_type: WebSocketStreamType
    raw_message: str


@dataclass
class ConnectionMetrics:
    """WebSocket connection performance metrics."""
    connected_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    messages_received: int = 0
    messages_sent: int = 0
    reconnections: int = 0
    connection_uptime: timedelta = field(default_factory=lambda: timedelta())
    average_latency_ms: float = 0.0
    last_ping_latency_ms: float = 0.0


class WebSocketManager:
    """
    Comprehensive WebSocket manager for Bybit real-time data streams.
    
    Features:
    - Automatic connection management with reconnection logic
    - Support for both public and private streams
    - Message validation and error handling
    - Performance metrics and health monitoring
    - Subscription management for multiple symbols/topics
    - Event-driven message processing with callbacks
    """
    
    def __init__(
        self,
        config: ConfigurationManager,
        credentials: Optional[BybitCredentials] = None,
        ws_config: Optional[WebSocketConfig] = None
    ):
        self.config = config
        self.credentials = credentials
        self.ws_config = ws_config or WebSocketConfig()
        self.logger = TradingLogger("websocket_manager")
        
        # Connection management
        self.public_ws: Optional[websockets.WebSocketServerProtocol] = None
        self.private_ws: Optional[websockets.WebSocketServerProtocol] = None
        
        self.public_status = ConnectionStatus.DISCONNECTED
        self.private_status = ConnectionStatus.DISCONNECTED
        
        # Metrics tracking
        self.public_metrics = ConnectionMetrics()
        self.private_metrics = ConnectionMetrics()
        
        # Subscription management
        self.public_subscriptions: Dict[str, bool] = {}
        self.private_subscriptions: Dict[str, bool] = {}
        
        # Message handlers
        self.message_handlers: Dict[str, Callable[[WebSocketMessage], None]] = {}
        self.error_handlers: List[Callable[[Exception, WebSocketStreamType], None]] = []
        
        # Internal state
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Rate limiting
        self.last_subscription_time = 0
        self.subscription_rate_limit = 1.0  # 1 subscription per second
        
        self.logger.info("WebSocketManager initialized")
    
    async def start(self) -> bool:
        """
        Start WebSocket connections and begin processing.
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.logger.info("Starting WebSocket manager...")
            self.running = True
            
            # Start public WebSocket connection
            public_task = asyncio.create_task(self._manage_public_connection())
            self.tasks.append(public_task)
            
            # Start private WebSocket connection if credentials provided
            if self.credentials:
                private_task = asyncio.create_task(self._manage_private_connection())
                self.tasks.append(private_task)
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            self.tasks.append(heartbeat_task)
            
            # Wait for initial connections
            await asyncio.sleep(2)
            
            self.logger.info("WebSocket manager started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket manager: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> None:
        """Stop WebSocket connections and cleanup."""
        try:
            self.logger.info("Stopping WebSocket manager...")
            self.running = False
            
            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Close WebSocket connections
            if self.public_ws:
                await self.public_ws.close()
            if self.private_ws:
                await self.private_ws.close()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            self.public_status = ConnectionStatus.DISCONNECTED
            self.private_status = ConnectionStatus.DISCONNECTED
            
            self.logger.info("WebSocket manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket manager: {e}")
    
    async def subscribe_public(self, topics: List[str]) -> bool:
        """
        Subscribe to public market data topics.
        
        Args:
            topics: List of topics to subscribe to
            
        Returns:
            bool: True if subscription successful
        """
        try:
            if self.public_status != ConnectionStatus.CONNECTED:
                self.logger.warning("Public WebSocket not connected")
                return False
            
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_subscription_time < self.subscription_rate_limit:
                await asyncio.sleep(self.subscription_rate_limit)
            
            # Prepare subscription message
            subscription_msg = {
                "op": "subscribe",
                "args": topics
            }
            
            # Send subscription
            await self.public_ws.send(json.dumps(subscription_msg))
            self.public_metrics.messages_sent += 1
            
            # Track subscriptions
            for topic in topics:
                self.public_subscriptions[topic] = True
            
            self.last_subscription_time = time.time()
            self.logger.info(f"Subscribed to public topics: {topics}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to public topics: {e}")
            return False
    
    async def subscribe_private(self, topics: List[str]) -> bool:
        """
        Subscribe to private account data topics.
        
        Args:
            topics: List of private topics to subscribe to
            
        Returns:
            bool: True if subscription successful
        """
        try:
            if not self.credentials:
                self.logger.error("No credentials provided for private subscription")
                return False
            
            if self.private_status != ConnectionStatus.CONNECTED:
                self.logger.warning("Private WebSocket not connected")
                return False
            
            # Private subscriptions require authentication first
            if not await self._authenticate_private_connection():
                return False
            
            # Prepare subscription message
            subscription_msg = {
                "op": "subscribe",
                "args": topics
            }
            
            # Send subscription
            await self.private_ws.send(json.dumps(subscription_msg))
            self.private_metrics.messages_sent += 1
            
            # Track subscriptions
            for topic in topics:
                self.private_subscriptions[topic] = True
            
            self.logger.info(f"Subscribed to private topics: {topics}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to private topics: {e}")
            return False
    
    def add_message_handler(self, topic_pattern: str, handler: Callable[[WebSocketMessage], None]) -> None:
        """
        Add a message handler for specific topics.
        
        Args:
            topic_pattern: Topic pattern to match (supports wildcards)
            handler: Callback function to handle messages
        """
        self.message_handlers[topic_pattern] = handler
        self.logger.info(f"Added message handler for: {topic_pattern}")
    
    def add_error_handler(self, handler: Callable[[Exception, WebSocketStreamType], None]) -> None:
        """
        Add an error handler for connection issues.
        
        Args:
            handler: Callback function to handle errors
        """
        self.error_handlers.append(handler)
        self.logger.info("Added error handler")
    
    async def _manage_public_connection(self) -> None:
        """Manage public WebSocket connection with reconnection logic."""
        reconnect_attempts = 0
        
        while self.running:
            try:
                self.public_status = ConnectionStatus.CONNECTING
                
                # Determine URL based on credentials
                url = (self.ws_config.public_url_testnet if 
                      (self.credentials and self.credentials.testnet) else 
                      self.ws_config.public_url_mainnet)
                
                self.logger.info(f"Connecting to public WebSocket: {url}")
                
                # Connect to WebSocket
                self.public_ws = await websockets.connect(
                    url,
                    ping_interval=self.ws_config.ping_interval,
                    ping_timeout=self.ws_config.ping_timeout,
                    max_size=self.ws_config.max_message_size
                )
                
                self.public_status = ConnectionStatus.CONNECTED
                self.public_metrics.connected_at = datetime.now()
                self.public_metrics.reconnections = reconnect_attempts
                reconnect_attempts = 0
                
                self.logger.info("Public WebSocket connected successfully")
                
                # Start message processing
                await self._process_public_messages()
                
            except Exception as e:
                self.public_status = ConnectionStatus.ERROR
                self.logger.error(f"Public WebSocket error: {e}")
                
                # Notify error handlers
                for handler in self.error_handlers:
                    try:
                        handler(e, WebSocketStreamType.PUBLIC)
                    except Exception as handler_error:
                        self.logger.error(f"Error handler failed: {handler_error}")
                
                # Reconnection logic
                if reconnect_attempts < self.ws_config.max_reconnect_attempts and self.running:
                    reconnect_attempts += 1
                    self.public_status = ConnectionStatus.RECONNECTING
                    
                    delay = min(self.ws_config.reconnect_delay * (2 ** reconnect_attempts), 60)
                    self.logger.info(f"Reconnecting in {delay}s (attempt {reconnect_attempts})")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error("Max reconnection attempts reached for public WebSocket")
                    break
    
    async def _manage_private_connection(self) -> None:
        """Manage private WebSocket connection with authentication."""
        reconnect_attempts = 0
        
        while self.running:
            try:
                self.private_status = ConnectionStatus.CONNECTING
                
                # Determine URL
                url = (self.ws_config.private_url_testnet if self.credentials.testnet else 
                      self.ws_config.private_url_mainnet)
                
                self.logger.info(f"Connecting to private WebSocket: {url}")
                
                # Connect to WebSocket
                self.private_ws = await websockets.connect(
                    url,
                    ping_interval=self.ws_config.ping_interval,
                    ping_timeout=self.ws_config.ping_timeout,
                    max_size=self.ws_config.max_message_size
                )
                
                self.private_status = ConnectionStatus.CONNECTED
                self.private_metrics.connected_at = datetime.now()
                self.private_metrics.reconnections = reconnect_attempts
                reconnect_attempts = 0
                
                self.logger.info("Private WebSocket connected successfully")
                
                # Start message processing
                await self._process_private_messages()
                
            except Exception as e:
                self.private_status = ConnectionStatus.ERROR
                self.logger.error(f"Private WebSocket error: {e}")
                
                # Notify error handlers
                for handler in self.error_handlers:
                    try:
                        handler(e, WebSocketStreamType.PRIVATE)
                    except Exception as handler_error:
                        self.logger.error(f"Error handler failed: {handler_error}")
                
                # Reconnection logic
                if reconnect_attempts < self.ws_config.max_reconnect_attempts and self.running:
                    reconnect_attempts += 1
                    self.private_status = ConnectionStatus.RECONNECTING
                    
                    delay = min(self.ws_config.reconnect_delay * (2 ** reconnect_attempts), 60)
                    self.logger.info(f"Reconnecting in {delay}s (attempt {reconnect_attempts})")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error("Max reconnection attempts reached for private WebSocket")
                    break
    
    async def _process_public_messages(self) -> None:
        """Process incoming public WebSocket messages."""
        try:
            async for message in self.public_ws:
                if not self.running:
                    break
                
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Update metrics
                    self.public_metrics.messages_received += 1
                    self.public_metrics.last_message_at = datetime.now()
                    
                    # Create structured message
                    ws_message = self._create_websocket_message(
                        data, WebSocketStreamType.PUBLIC, message
                    )
                    
                    if ws_message:
                        # Route to appropriate handler
                        await self._route_message(ws_message)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse public message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing public message: {e}")
                    
        except ConnectionClosed:
            self.logger.warning("Public WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"Public message processing error: {e}")
            raise
    
    async def _process_private_messages(self) -> None:
        """Process incoming private WebSocket messages."""
        try:
            async for message in self.private_ws:
                if not self.running:
                    break
                
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Update metrics
                    self.private_metrics.messages_received += 1
                    self.private_metrics.last_message_at = datetime.now()
                    
                    # Create structured message
                    ws_message = self._create_websocket_message(
                        data, WebSocketStreamType.PRIVATE, message
                    )
                    
                    if ws_message:
                        # Route to appropriate handler
                        await self._route_message(ws_message)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse private message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing private message: {e}")
                    
        except ConnectionClosed:
            self.logger.warning("Private WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"Private message processing error: {e}")
            raise
    
    def _create_websocket_message(
        self, 
        data: Dict[str, Any], 
        stream_type: WebSocketStreamType,
        raw_message: str
    ) -> Optional[WebSocketMessage]:
        """Create a structured WebSocket message from raw data."""
        try:
            # Extract topic from message
            topic = data.get('topic', data.get('op', 'unknown'))
            
            return WebSocketMessage(
                topic=topic,
                data=data,
                timestamp=datetime.now(),
                stream_type=stream_type,
                raw_message=raw_message
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create WebSocket message: {e}")
            return None
    
    async def _route_message(self, message: WebSocketMessage) -> None:
        """Route message to appropriate handlers."""
        try:
            # Find matching handlers
            for pattern, handler in self.message_handlers.items():
                if self._topic_matches_pattern(message.topic, pattern):
                    try:
                        # Call handler (can be sync or async)
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except Exception as e:
                        self.logger.error(f"Message handler error for {pattern}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Message routing error: {e}")
    
    def _topic_matches_pattern(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (supports basic wildcards)."""
        if pattern == "*":
            return True
        
        if "*" in pattern:
            # Simple wildcard matching
            parts = pattern.split("*")
            return topic.startswith(parts[0]) and topic.endswith(parts[-1])
        
        return topic == pattern
    
    async def _authenticate_private_connection(self) -> bool:
        """Authenticate private WebSocket connection."""
        try:
            if not self.credentials:
                return False
            
            # Generate authentication signature
            expires = int((time.time() + 5) * 1000)  # 5 seconds from now
            signature_payload = f"GET/realtime{expires}"
            signature = hmac.new(
                self.credentials.api_secret.encode('utf-8'),
                signature_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Send authentication message
            auth_msg = {
                "op": "auth",
                "args": [
                    self.credentials.api_key,
                    expires,
                    signature
                ]
            }
            
            await self.private_ws.send(json.dumps(auth_msg))
            self.private_metrics.messages_sent += 1
            
            # Wait for authentication response
            response = await asyncio.wait_for(
                self.private_ws.recv(), 
                timeout=self.ws_config.message_timeout
            )
            
            auth_response = json.loads(response)
            
            if auth_response.get('op') == 'auth' and auth_response.get('success'):
                self.logger.info("Private WebSocket authenticated successfully")
                return True
            else:
                self.logger.error(f"Authentication failed: {auth_response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor connection health and send heartbeats."""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check public connection health
                if (self.public_metrics.last_message_at and 
                    (current_time - self.public_metrics.last_message_at).seconds > 
                    self.ws_config.heartbeat_interval * 2):
                    self.logger.warning("Public WebSocket: No messages received recently")
                
                # Check private connection health
                if (self.private_metrics.last_message_at and 
                    (current_time - self.private_metrics.last_message_at).seconds > 
                    self.ws_config.heartbeat_interval * 2):
                    self.logger.warning("Private WebSocket: No messages received recently")
                
                # Send ping if connection is idle
                if (self.public_ws and self.public_status == ConnectionStatus.CONNECTED):
                    ping_start = time.time()
                    await self.public_ws.ping()
                    pong_waiter = await self.public_ws.ping()
                    await pong_waiter
                    ping_latency = (time.time() - ping_start) * 1000
                    self.public_metrics.last_ping_latency_ms = ping_latency
                
                await asyncio.sleep(self.ws_config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(self.ws_config.heartbeat_interval)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and metrics."""
        return {
            "public": {
                "status": self.public_status.value,
                "metrics": {
                    "connected_at": self.public_metrics.connected_at.isoformat() if self.public_metrics.connected_at else None,
                    "last_message_at": self.public_metrics.last_message_at.isoformat() if self.public_metrics.last_message_at else None,
                    "messages_received": self.public_metrics.messages_received,
                    "messages_sent": self.public_metrics.messages_sent,
                    "reconnections": self.public_metrics.reconnections,
                    "last_ping_latency_ms": self.public_metrics.last_ping_latency_ms
                },
                "subscriptions": list(self.public_subscriptions.keys())
            },
            "private": {
                "status": self.private_status.value,
                "metrics": {
                    "connected_at": self.private_metrics.connected_at.isoformat() if self.private_metrics.connected_at else None,
                    "last_message_at": self.private_metrics.last_message_at.isoformat() if self.private_metrics.last_message_at else None,
                    "messages_received": self.private_metrics.messages_received,
                    "messages_sent": self.private_metrics.messages_sent,
                    "reconnections": self.private_metrics.reconnections,
                    "last_ping_latency_ms": self.private_metrics.last_ping_latency_ms
                },
                "subscriptions": list(self.private_subscriptions.keys())
            }
        }


# Utility functions for common WebSocket operations

async def create_websocket_manager(
    config: ConfigurationManager,
    credentials: Optional[BybitCredentials] = None
) -> WebSocketManager:
    """
    Create and initialize a WebSocket manager.
    
    Args:
        config: Configuration manager instance
        credentials: Optional Bybit credentials for private streams
        
    Returns:
        WebSocketManager: Initialized WebSocket manager
    """
    ws_manager = WebSocketManager(config, credentials)
    await ws_manager.start()
    return ws_manager


def create_market_data_subscriptions(symbols: List[str]) -> List[str]:
    """
    Create market data subscription topics for given symbols.
    
    Args:
        symbols: List of trading symbols
        
    Returns:
        List[str]: Subscription topics
    """
    topics = []
    for symbol in symbols:
        topics.extend([
            f"orderbook.1.{symbol}",      # Order book updates
            f"publicTrade.{symbol}",      # Public trades
            f"tickers.{symbol}",          # Ticker updates
            f"kline.1.{symbol}"           # 1-minute klines
        ])
    return topics


def create_private_subscriptions() -> List[str]:
    """
    Create private subscription topics for account data.
    
    Returns:
        List[str]: Private subscription topics
    """
    return [
        "position",     # Position updates
        "execution",    # Trade execution updates
        "order",        # Order updates
        "wallet"        # Wallet balance updates
    ]