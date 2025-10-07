"""
Unified WebSocket Manager - Phase 3 API Consolidation

This module provides a comprehensive WebSocket management system for real-time
market data, order updates, and account information from Bybit. It consolidates
all WebSocket functionality into a single, robust system with:

- Connection pooling and management
- Automatic reconnection with exponential backoff
- Subscription management
- Data parsing and validation
- Event-driven architecture
- Performance monitoring
- Error handling and recovery

Key Features:
- Multi-stream WebSocket connections
- Automatic failover and reconnection
- Subscription state management
- Real-time data callbacks
- Connection health monitoring
- Rate limiting compliance
- Memory-efficient data handling
"""

import asyncio
import json
import time
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid
from collections import defaultdict, deque

import websockets
import pandas as pd
from websockets.exceptions import ConnectionClosed, WebSocketException

# Import unified client components
from .unified_bybit_client import BybitCredentials, Environment, ConnectionStatus

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class StreamType(Enum):
    """WebSocket stream types"""
    PUBLIC = "public"
    PRIVATE = "private"
    BOTH = "both"

class SubscriptionStatus(Enum):
    """Subscription status"""
    PENDING = "pending"
    ACTIVE = "active"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WebSocketMetrics:
    """WebSocket connection metrics"""
    connection_time: datetime
    messages_received: int = 0
    messages_sent: int = 0
    reconnection_count: int = 0
    last_message_time: Optional[datetime] = None
    latency_ms: Optional[float] = None
    error_count: int = 0

@dataclass
class Subscription:
    """WebSocket subscription information"""
    id: str
    topic: str
    symbols: List[str]
    callback: Callable
    stream_type: StreamType
    status: SubscriptionStatus = SubscriptionStatus.PENDING
    created_time: datetime = field(default_factory=datetime.now)
    last_update: Optional[datetime] = None

@dataclass
class ConnectionConfig:
    """WebSocket connection configuration"""
    max_reconnect_attempts: int = 10
    base_reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    ping_interval: int = 20
    ping_timeout: int = 10
    close_timeout: int = 10
    message_queue_size: int = 1000
    enable_compression: bool = True

# ============================================================================
# WEBSOCKET MANAGER
# ============================================================================

class UnifiedWebSocketManager:
    """
    Unified WebSocket Manager for Bybit
    
    Manages all WebSocket connections, subscriptions, and real-time data streams
    """
    
    def __init__(
        self,
        credentials: BybitCredentials,
        config: Optional[ConnectionConfig] = None
    ):
        self.credentials = credentials
        self.config = config or ConnectionConfig()
        
        # Connection management
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.connection_status: Dict[str, ConnectionStatus] = {}
        self.connection_tasks: Dict[str, asyncio.Task] = {}
        self.metrics: Dict[str, WebSocketMetrics] = {}
        
        # Subscription management
        self.subscriptions: Dict[str, Subscription] = {}
        self.topic_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.symbol_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Message handling
        self.message_queues: Dict[str, deque] = {}
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Event system
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Control flags
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.total_messages_processed = 0
        self.start_time: Optional[datetime] = None
        
        logger.info("Unified WebSocket Manager initialized")
    
    async def start(self):
        """Start the WebSocket manager"""
        if self.is_running:
            logger.warning("WebSocket manager already running")
            return
        
        logger.info("Starting Unified WebSocket Manager...")
        
        try:
            self.is_running = True
            self.start_time = datetime.now()
            self.shutdown_event.clear()
            
            # Initialize connection tasks
            await self._initialize_connections()
            
            # Start message processing
            asyncio.create_task(self._process_messages())
            
            # Start health monitoring
            asyncio.create_task(self._monitor_health())
            
            logger.info("Unified WebSocket Manager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket manager: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the WebSocket manager"""
        if not self.is_running:
            logger.warning("WebSocket manager not running")
            return
        
        logger.info("Stopping Unified WebSocket Manager...")
        
        try:
            # Signal shutdown
            self.is_running = False
            self.shutdown_event.set()
            
            # Cancel all connection tasks
            for task_name, task in self.connection_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    logger.info(f"Cancelled {task_name} connection task")
            
            # Close all connections
            await self._close_all_connections()
            
            # Clear state
            self.connections.clear()
            self.connection_status.clear()
            self.connection_tasks.clear()
            self.subscriptions.clear()
            self.topic_subscriptions.clear()
            self.symbol_subscriptions.clear()
            
            logger.info("Unified WebSocket Manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket manager: {e}")
    
    async def _initialize_connections(self):
        """Initialize WebSocket connections"""
        # Start public connection
        self.connection_tasks['public'] = asyncio.create_task(
            self._maintain_connection('public')
        )
        
        # Start private connection if credentials available
        if self.credentials.api_key and self.credentials.api_secret:
            self.connection_tasks['private'] = asyncio.create_task(
                self._maintain_connection('private')
            )
        
        # Wait for initial connections
        await asyncio.sleep(2)
    
    async def _maintain_connection(self, connection_type: str):
        """Maintain WebSocket connection with automatic reconnection"""
        reconnect_count = 0
        
        while self.is_running and reconnect_count < self.config.max_reconnect_attempts:
            try:
                # Get WebSocket URL
                if connection_type == 'public':
                    url = self.credentials.ws_url
                else:
                    url = self.credentials.ws_private_url
                
                logger.info(f"Connecting to {connection_type} WebSocket: {url}")
                
                # Set connection status
                self.connection_status[connection_type] = ConnectionStatus.CONNECTING
                
                # Connect with configuration
                async with websockets.connect(
                    url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout,
                    close_timeout=self.config.close_timeout,
                    compression="deflate" if self.config.enable_compression else None
                ) as websocket:
                    
                    # Store connection
                    self.connections[connection_type] = websocket
                    self.metrics[connection_type] = WebSocketMetrics(
                        connection_time=datetime.now(),
                        reconnection_count=reconnect_count
                    )
                    
                    # Authenticate private connection
                    if connection_type == 'private':
                        await self._authenticate_connection(websocket)
                    
                    # Update status
                    self.connection_status[connection_type] = ConnectionStatus.CONNECTED
                    reconnect_count = 0  # Reset on successful connection
                    
                    logger.info(f"{connection_type.title()} WebSocket connected successfully")
                    
                    # Emit connection event
                    await self._emit_event('connection_established', {
                        'connection_type': connection_type,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Resubscribe to topics
                    await self._resubscribe_topics(connection_type)
                    
                    # Handle messages
                    try:
                        async for message in websocket:
                            if not self.is_running:
                                break
                            
                            await self._handle_message(message, connection_type)
                            
                    except ConnectionClosed:
                        logger.warning(f"{connection_type.title()} WebSocket connection closed")
                    except WebSocketException as e:
                        logger.error(f"{connection_type.title()} WebSocket error: {e}")
                    
            except Exception as e:
                logger.error(f"Error in {connection_type} WebSocket connection: {e}")
                reconnect_count += 1
                
                # Update status
                self.connection_status[connection_type] = ConnectionStatus.RECONNECTING
                
                # Emit reconnection event
                await self._emit_event('connection_lost', {
                    'connection_type': connection_type,
                    'error': str(e),
                    'reconnect_attempt': reconnect_count,
                    'timestamp': datetime.now().isoformat()
                })
                
                if reconnect_count < self.config.max_reconnect_attempts:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.config.base_reconnect_delay * (2 ** (reconnect_count - 1)),
                        self.config.max_reconnect_delay
                    )
                    
                    logger.info(f"Reconnecting {connection_type} WebSocket in {delay:.1f}s "
                              f"(attempt {reconnect_count}/{self.config.max_reconnect_attempts})")
                    
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max reconnection attempts reached for {connection_type} WebSocket")
                    self.connection_status[connection_type] = ConnectionStatus.ERROR
                    break
            
            finally:
                # Clean up connection
                if connection_type in self.connections:
                    del self.connections[connection_type]
    
    async def _authenticate_connection(self, websocket):
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
            raise Exception(f"WebSocket authentication failed: {auth_result}")
        
        logger.info("Private WebSocket authenticated successfully")
    
    async def _handle_message(self, message: str, connection_type: str):
        """Handle incoming WebSocket message"""
        try:
            # Update metrics
            if connection_type in self.metrics:
                self.metrics[connection_type].messages_received += 1
                self.metrics[connection_type].last_message_time = datetime.now()
            
            self.total_messages_processed += 1
            
            # Parse message
            data = json.loads(message)
            
            # Add to message queue
            if connection_type not in self.message_queues:
                self.message_queues[connection_type] = deque(maxlen=self.config.message_queue_size)
            
            self.message_queues[connection_type].append({
                'data': data,
                'timestamp': datetime.now(),
                'connection_type': connection_type
            })
            
            # Handle different message types
            if 'topic' in data:
                await self._handle_topic_message(data, connection_type)
            elif 'op' in data:
                await self._handle_operation_message(data, connection_type)
            else:
                logger.debug(f"Unknown message format from {connection_type}: {data}")
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from {connection_type}: {message}")
        except Exception as e:
            logger.error(f"Error handling message from {connection_type}: {e}")
            if connection_type in self.metrics:
                self.metrics[connection_type].error_count += 1
    
    async def _handle_topic_message(self, data: Dict[str, Any], connection_type: str):
        """Handle topic-based message"""
        topic = data.get('topic', '')
        
        # Find matching subscriptions
        matching_subs = []
        for sub_id, subscription in self.subscriptions.items():
            if subscription.topic in topic or any(symbol in topic for symbol in subscription.symbols):
                matching_subs.append(subscription)
        
        # Call subscription callbacks
        for subscription in matching_subs:
            try:
                subscription.last_update = datetime.now()
                await subscription.callback(data)
            except Exception as e:
                logger.error(f"Error in subscription callback {subscription.id}: {e}")
        
        # Call general message handlers
        for handler in self.message_handlers.get(topic, []):
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Error in message handler for {topic}: {e}")
    
    async def _handle_operation_message(self, data: Dict[str, Any], connection_type: str):
        """Handle operation message (subscribe, unsubscribe, etc.)"""
        op = data.get('op', '')
        
        if op == 'subscribe':
            success = data.get('success', False)
            topic = data.get('args', [None])[0] if data.get('args') else None
            
            if success and topic:
                logger.info(f"Successfully subscribed to {topic}")
                await self._emit_event('subscription_success', {
                    'topic': topic,
                    'connection_type': connection_type
                })
            else:
                logger.error(f"Failed to subscribe to {topic}: {data}")
                await self._emit_event('subscription_failed', {
                    'topic': topic,
                    'error': data,
                    'connection_type': connection_type
                })
        
        elif op == 'unsubscribe':
            success = data.get('success', False)
            topic = data.get('args', [None])[0] if data.get('args') else None
            
            if success and topic:
                logger.info(f"Successfully unsubscribed from {topic}")
            else:
                logger.error(f"Failed to unsubscribe from {topic}: {data}")
    
    async def _resubscribe_topics(self, connection_type: str):
        """Resubscribe to topics after reconnection"""
        for subscription in self.subscriptions.values():
            if (subscription.stream_type == StreamType.BOTH or 
                subscription.stream_type.value == connection_type):
                
                await self._send_subscription(subscription, connection_type)
    
    async def _send_subscription(self, subscription: Subscription, connection_type: str):
        """Send subscription message"""
        if connection_type not in self.connections:
            logger.warning(f"No {connection_type} connection available for subscription")
            return
        
        websocket = self.connections[connection_type]
        
        # Build subscription message
        if subscription.symbols:
            # Symbol-specific subscription
            for symbol in subscription.symbols:
                topic_with_symbol = f"{subscription.topic}.{symbol}"
                message = {
                    "op": "subscribe",
                    "args": [topic_with_symbol]
                }
                
                try:
                    await websocket.send(json.dumps(message))
                    if connection_type in self.metrics:
                        self.metrics[connection_type].messages_sent += 1
                    
                    logger.debug(f"Sent subscription for {topic_with_symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to send subscription for {topic_with_symbol}: {e}")
        else:
            # General topic subscription
            message = {
                "op": "subscribe",
                "args": [subscription.topic]
            }
            
            try:
                await websocket.send(json.dumps(message))
                if connection_type in self.metrics:
                    self.metrics[connection_type].messages_sent += 1
                
                logger.debug(f"Sent subscription for {subscription.topic}")
                
            except Exception as e:
                logger.error(f"Failed to send subscription for {subscription.topic}: {e}")
    
    async def _process_messages(self):
        """Process message queues"""
        while self.is_running:
            try:
                # Process messages from all queues
                for connection_type, queue in self.message_queues.items():
                    while queue and self.is_running:
                        message_data = queue.popleft()
                        # Additional message processing can be added here
                        await asyncio.sleep(0)  # Yield control
                
                await asyncio.sleep(0.001)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                logger.error(f"Error in message processing: {e}")
                await asyncio.sleep(1)
    
    async def _monitor_health(self):
        """Monitor connection health"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                for connection_type, metrics in self.metrics.items():
                    # Check for stale connections
                    if metrics.last_message_time:
                        time_since_last = current_time - metrics.last_message_time
                        if time_since_last > timedelta(minutes=2):
                            logger.warning(f"{connection_type} connection appears stale "
                                         f"(last message: {time_since_last.total_seconds():.1f}s ago)")
                            
                            await self._emit_event('connection_stale', {
                                'connection_type': connection_type,
                                'last_message_time': metrics.last_message_time.isoformat(),
                                'time_since_last': time_since_last.total_seconds()
                            })
                
                # Sleep for health check interval
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered callbacks"""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Error in event callback for {event_type}: {e}")
    
    async def _close_all_connections(self):
        """Close all WebSocket connections"""
        for connection_type, websocket in self.connections.items():
            try:
                if not websocket.closed:
                    await websocket.close()
                logger.info(f"Closed {connection_type} WebSocket connection")
            except Exception as e:
                logger.error(f"Error closing {connection_type} WebSocket: {e}")
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable,
        symbols: Optional[List[str]] = None,
        stream_type: StreamType = StreamType.PUBLIC
    ) -> str:
        """Subscribe to a WebSocket topic"""
        
        subscription_id = str(uuid.uuid4())
        
        subscription = Subscription(
            id=subscription_id,
            topic=topic,
            symbols=symbols or [],
            callback=callback,
            stream_type=stream_type
        )
        
        self.subscriptions[subscription_id] = subscription
        
        # Add to topic tracking
        self.topic_subscriptions[topic].add(subscription_id)
        
        if symbols:
            for symbol in symbols:
                self.symbol_subscriptions[symbol].add(subscription_id)
        
        # Send subscription if connections are available
        if stream_type == StreamType.PUBLIC or stream_type == StreamType.BOTH:
            if 'public' in self.connections:
                await self._send_subscription(subscription, 'public')
        
        if stream_type == StreamType.PRIVATE or stream_type == StreamType.BOTH:
            if 'private' in self.connections:
                await self._send_subscription(subscription, 'private')
        
        logger.info(f"Created subscription {subscription_id} for topic {topic}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a topic"""
        
        if subscription_id not in self.subscriptions:
            logger.warning(f"Subscription {subscription_id} not found")
            return False
        
        subscription = self.subscriptions[subscription_id]
        
        # Send unsubscribe message
        for connection_type in ['public', 'private']:
            if connection_type in self.connections:
                websocket = self.connections[connection_type]
                
                if subscription.symbols:
                    for symbol in subscription.symbols:
                        topic_with_symbol = f"{subscription.topic}.{symbol}"
                        message = {
                            "op": "unsubscribe",
                            "args": [topic_with_symbol]
                        }
                        
                        try:
                            await websocket.send(json.dumps(message))
                        except Exception as e:
                            logger.error(f"Failed to unsubscribe from {topic_with_symbol}: {e}")
                else:
                    message = {
                        "op": "unsubscribe",
                        "args": [subscription.topic]
                    }
                    
                    try:
                        await websocket.send(json.dumps(message))
                    except Exception as e:
                        logger.error(f"Failed to unsubscribe from {subscription.topic}: {e}")
        
        # Remove from tracking
        subscription.status = SubscriptionStatus.CANCELLED
        del self.subscriptions[subscription_id]
        
        self.topic_subscriptions[subscription.topic].discard(subscription_id)
        
        if subscription.symbols:
            for symbol in subscription.symbols:
                self.symbol_subscriptions[symbol].discard(subscription_id)
        
        logger.info(f"Unsubscribed from {subscription_id}")
        return True
    
    def add_event_callback(self, event_type: str, callback: Callable):
        """Add callback for WebSocket events"""
        self.event_callbacks[event_type].append(callback)
    
    def add_message_handler(self, topic: str, handler: Callable):
        """Add message handler for specific topic"""
        self.message_handlers[topic].append(handler)
    
    def get_status(self) -> Dict[str, Any]:
        """Get WebSocket manager status"""
        return {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'connection_status': {k: v.value for k, v in self.connection_status.items()},
            'active_connections': list(self.connections.keys()),
            'subscription_count': len(self.subscriptions),
            'total_messages_processed': self.total_messages_processed,
            'metrics': {
                conn_type: {
                    'connection_time': metrics.connection_time.isoformat(),
                    'messages_received': metrics.messages_received,
                    'messages_sent': metrics.messages_sent,
                    'reconnection_count': metrics.reconnection_count,
                    'last_message_time': metrics.last_message_time.isoformat() if metrics.last_message_time else None,
                    'error_count': metrics.error_count
                }
                for conn_type, metrics in self.metrics.items()
            }
        }
    
    def get_subscriptions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active subscriptions"""
        return {
            sub_id: {
                'topic': sub.topic,
                'symbols': sub.symbols,
                'stream_type': sub.stream_type.value,
                'status': sub.status.value,
                'created_time': sub.created_time.isoformat(),
                'last_update': sub.last_update.isoformat() if sub.last_update else None
            }
            for sub_id, sub in self.subscriptions.items()
        }

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def create_market_data_stream(
    credentials: BybitCredentials,
    symbols: List[str],
    callback: Callable,
    topics: List[str] = None
) -> UnifiedWebSocketManager:
    """Create a market data WebSocket stream"""
    
    if topics is None:
        topics = ['ticker', 'orderbook.1', 'publicTrade']
    
    manager = UnifiedWebSocketManager(credentials)
    await manager.start()
    
    # Subscribe to all requested topics for all symbols
    for topic in topics:
        await manager.subscribe(
            topic=topic,
            callback=callback,
            symbols=symbols,
            stream_type=StreamType.PUBLIC
        )
    
    return manager

async def create_trading_stream(
    credentials: BybitCredentials,
    callback: Callable
) -> UnifiedWebSocketManager:
    """Create a trading WebSocket stream for orders and positions"""
    
    manager = UnifiedWebSocketManager(credentials)
    await manager.start()
    
    # Subscribe to trading topics
    await manager.subscribe(
        topic='order',
        callback=callback,
        stream_type=StreamType.PRIVATE
    )
    
    await manager.subscribe(
        topic='position',
        callback=callback,
        stream_type=StreamType.PRIVATE
    )
    
    await manager.subscribe(
        topic='execution',
        callback=callback,
        stream_type=StreamType.PRIVATE
    )
    
    return manager

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'UnifiedWebSocketManager',
    'StreamType',
    'SubscriptionStatus',
    'WebSocketMetrics',
    'Subscription',
    'ConnectionConfig',
    'create_market_data_stream',
    'create_trading_stream'
]