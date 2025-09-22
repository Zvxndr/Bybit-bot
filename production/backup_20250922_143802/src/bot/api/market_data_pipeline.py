"""
Unified Market Data Pipeline - Phase 3 API Consolidation

This module provides a comprehensive market data pipeline that consolidates
all data sources into a unified, high-performance system. Features include:

- Multi-source data aggregation (REST API + WebSocket)
- Real-time data normalization and validation
- Intelligent caching with TTL management
- Data quality monitoring and validation
- Performance optimization with batching
- Historical data integration
- Australian timezone support
- Error recovery and failover

Key Components:
- MarketDataPipeline: Main orchestrator
- DataCache: High-performance caching layer
- DataValidator: Quality assurance system
- DataAggregator: Multi-source aggregation
- PerformanceMonitor: Metrics and monitoring
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import json

import pandas as pd
import numpy as np
from cachetools import TTLCache, LRUCache

# Import unified client components
from .unified_bybit_client import (
    UnifiedBybitClient, MarketData, OrderBookData, TradeData,
    BybitCredentials, Environment
)
from .websocket_manager import (
    UnifiedWebSocketManager, StreamType, create_market_data_stream
)

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class DataSource(Enum):
    """Data source types"""
    REST_API = "rest_api"
    WEBSOCKET = "websocket"
    CACHE = "cache"
    HISTORICAL = "historical"

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"

class CacheStrategy(Enum):
    """Cache strategies"""
    TTL = "ttl"  # Time to live
    LRU = "lru"  # Least recently used
    HYBRID = "hybrid"  # TTL + LRU

@dataclass
class DataPoint:
    """Standardized data point"""
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    source: DataSource
    quality: DataQuality = DataQuality.GOOD
    latency_ms: Optional[float] = None

@dataclass
class PipelineConfig:
    """Market data pipeline configuration"""
    # Cache settings
    cache_size: int = 10000
    cache_ttl_seconds: int = 60
    cache_strategy: CacheStrategy = CacheStrategy.HYBRID
    
    # Update intervals (seconds)
    ticker_update_interval: float = 1.0
    orderbook_update_interval: float = 0.5
    trade_update_interval: float = 0.1
    
    # Data quality thresholds
    max_price_deviation_pct: float = 10.0
    max_latency_ms: float = 5000.0
    min_volume_threshold: float = 0.0
    
    # Performance settings
    batch_size: int = 100
    max_queue_size: int = 1000
    enable_compression: bool = True
    
    # Australian market settings
    timezone: str = "Australia/Sydney"
    market_hours_start: str = "09:00"
    market_hours_end: str = "17:00"

@dataclass
class DataMetrics:
    """Data pipeline metrics"""
    total_updates: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    last_update_time: Optional[datetime] = None
    data_quality_score: float = 100.0

# ============================================================================
# DATA CACHE
# ============================================================================

class UnifiedDataCache:
    """High-performance data caching system"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Initialize caches based on strategy
        if config.cache_strategy == CacheStrategy.TTL:
            self.cache = TTLCache(
                maxsize=config.cache_size,
                ttl=config.cache_ttl_seconds
            )
        elif config.cache_strategy == CacheStrategy.LRU:
            self.cache = LRUCache(maxsize=config.cache_size)
        else:  # HYBRID
            self.cache = TTLCache(
                maxsize=config.cache_size,
                ttl=config.cache_ttl_seconds
            )
            self.lru_cache = LRUCache(maxsize=config.cache_size // 2)
        
        # Specialized caches
        self.ticker_cache: Dict[str, MarketData] = {}
        self.orderbook_cache: Dict[str, OrderBookData] = {}
        self.trade_cache: Dict[str, List[TradeData]] = defaultdict(list)
        
        # Cache metrics
        self.hits = 0
        self.misses = 0
        self.last_cleanup = datetime.now()
        
        logger.info(f"Initialized data cache with {config.cache_strategy.value} strategy")
    
    def get(self, key: str, default=None) -> Any:
        """Get value from cache"""
        # Try main cache first
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        # Try LRU cache if hybrid
        if hasattr(self, 'lru_cache') and key in self.lru_cache:
            self.hits += 1
            value = self.lru_cache[key]
            # Promote to main cache
            self.cache[key] = value
            return value
        
        self.misses += 1
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        self.cache[key] = value
        
        # Also store in LRU cache if hybrid
        if hasattr(self, 'lru_cache'):
            self.lru_cache[key] = value
    
    def get_ticker(self, symbol: str) -> Optional[MarketData]:
        """Get ticker data from cache"""
        return self.ticker_cache.get(symbol)
    
    def set_ticker(self, symbol: str, data: MarketData):
        """Set ticker data in cache"""
        self.ticker_cache[symbol] = data
        self.set(f"ticker:{symbol}", data)
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        """Get orderbook data from cache"""
        return self.orderbook_cache.get(symbol)
    
    def set_orderbook(self, symbol: str, data: OrderBookData):
        """Set orderbook data in cache"""
        self.orderbook_cache[symbol] = data
        self.set(f"orderbook:{symbol}", data)
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """Get recent trades from cache"""
        trades = self.trade_cache.get(symbol, [])
        return sorted(trades, key=lambda t: t.timestamp, reverse=True)[:limit]
    
    def add_trade(self, symbol: str, trade: TradeData):
        """Add trade to cache"""
        if symbol not in self.trade_cache:
            self.trade_cache[symbol] = deque(maxlen=1000)
        
        self.trade_cache[symbol].append(trade)
        
        # Also store in main cache
        key = f"trades:{symbol}:latest"
        recent_trades = list(self.trade_cache[symbol])[-100:]  # Last 100 trades
        self.set(key, recent_trades)
    
    def cleanup(self):
        """Cleanup expired cache entries"""
        current_time = datetime.now()
        
        # Cleanup ticker cache (keep only recent data)
        cutoff_time = current_time - timedelta(seconds=self.config.cache_ttl_seconds)
        
        expired_tickers = [
            symbol for symbol, data in self.ticker_cache.items()
            if data.timestamp < cutoff_time
        ]
        
        for symbol in expired_tickers:
            del self.ticker_cache[symbol]
        
        # Cleanup orderbook cache
        expired_orderbooks = [
            symbol for symbol, data in self.orderbook_cache.items()
            if data.timestamp < cutoff_time
        ]
        
        for symbol in expired_orderbooks:
            del self.orderbook_cache[symbol]
        
        # Cleanup trade cache (keep only recent trades)
        for symbol, trades in self.trade_cache.items():
            # Remove trades older than 1 hour
            trades_cutoff = current_time - timedelta(hours=1)
            while trades and trades[0].timestamp < trades_cutoff:
                trades.popleft()
        
        self.last_cleanup = current_time
        
        if expired_tickers or expired_orderbooks:
            logger.debug(f"Cleaned up {len(expired_tickers)} ticker and "
                        f"{len(expired_orderbooks)} orderbook entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_pct': round(hit_rate, 2),
            'cache_size': len(self.cache),
            'ticker_cache_size': len(self.ticker_cache),
            'orderbook_cache_size': len(self.orderbook_cache),
            'trade_cache_symbols': len(self.trade_cache),
            'last_cleanup': self.last_cleanup.isoformat()
        }

# ============================================================================
# DATA VALIDATOR
# ============================================================================

class DataValidator:
    """Data quality validation and monitoring"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validation_history: Dict[str, List[DataQuality]] = defaultdict(list)
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info("Initialized data validator")
    
    def validate_market_data(self, data: MarketData) -> DataQuality:
        """Validate market data quality"""
        quality_scores = []
        
        # Price validation
        if data.price <= 0 or data.bid <= 0 or data.ask <= 0:
            return DataQuality.INVALID
        
        # Spread validation
        spread_pct = ((data.ask - data.bid) / data.price * 100)
        if spread_pct > 10:  # Spread > 10%
            quality_scores.append(50)
        elif spread_pct > 5:  # Spread > 5%
            quality_scores.append(70)
        else:
            quality_scores.append(100)
        
        # Volume validation
        if data.volume_24h < self.config.min_volume_threshold:
            quality_scores.append(60)
        else:
            quality_scores.append(100)
        
        # Price deviation check
        symbol_prices = self.price_history[data.symbol]
        if len(symbol_prices) > 0:
            avg_price = sum(symbol_prices) / len(symbol_prices)
            deviation_pct = abs((data.price - avg_price) / avg_price * 100)
            
            if deviation_pct > self.config.max_price_deviation_pct:
                quality_scores.append(30)
            elif deviation_pct > self.config.max_price_deviation_pct / 2:
                quality_scores.append(70)
            else:
                quality_scores.append(100)
        
        # Update price history
        symbol_prices.append(data.price)
        
        # Calculate overall quality
        avg_score = sum(quality_scores) / len(quality_scores)
        
        if avg_score >= 90:
            quality = DataQuality.EXCELLENT
        elif avg_score >= 80:
            quality = DataQuality.GOOD
        elif avg_score >= 60:
            quality = DataQuality.FAIR
        elif avg_score >= 40:
            quality = DataQuality.POOR
        else:
            quality = DataQuality.INVALID
        
        # Update validation history
        self.validation_history[data.symbol].append(quality)
        if len(self.validation_history[data.symbol]) > 100:
            self.validation_history[data.symbol].pop(0)
        
        return quality
    
    def validate_orderbook(self, data: OrderBookData) -> DataQuality:
        """Validate orderbook data quality"""
        if not data.bids or not data.asks:
            return DataQuality.INVALID
        
        # Check if bids and asks are properly ordered
        bid_prices = [Decimal(bid[0]) for bid in data.bids]
        ask_prices = [Decimal(ask[0]) for ask in data.asks]
        
        # Bids should be in descending order
        if bid_prices != sorted(bid_prices, reverse=True):
            return DataQuality.POOR
        
        # Asks should be in ascending order
        if ask_prices != sorted(ask_prices):
            return DataQuality.POOR
        
        # Check for crossed market (bid >= ask)
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        
        if best_bid >= best_ask:
            return DataQuality.INVALID
        
        # Check spread reasonableness
        spread_pct = ((best_ask - best_bid) / best_ask * 100)
        if spread_pct > 10:
            return DataQuality.FAIR
        elif spread_pct > 5:
            return DataQuality.GOOD
        else:
            return DataQuality.EXCELLENT
    
    def get_symbol_quality_score(self, symbol: str) -> float:
        """Get quality score for a symbol"""
        history = self.validation_history.get(symbol, [])
        if not history:
            return 100.0
        
        quality_values = {
            DataQuality.EXCELLENT: 100,
            DataQuality.GOOD: 80,
            DataQuality.FAIR: 60,
            DataQuality.POOR: 40,
            DataQuality.INVALID: 0
        }
        
        scores = [quality_values[q] for q in history]
        return sum(scores) / len(scores)
    
    def get_overall_quality_score(self) -> float:
        """Get overall data quality score"""
        all_scores = []
        for symbol in self.validation_history:
            all_scores.append(self.get_symbol_quality_score(symbol))
        
        return sum(all_scores) / len(all_scores) if all_scores else 100.0

# ============================================================================
# MARKET DATA PIPELINE
# ============================================================================

class UnifiedMarketDataPipeline:
    """
    Unified Market Data Pipeline
    
    Orchestrates all market data operations with caching, validation, and monitoring
    """
    
    def __init__(
        self,
        credentials: BybitCredentials,
        config: Optional[PipelineConfig] = None
    ):
        self.credentials = credentials
        self.config = config or PipelineConfig()
        
        # Core components
        self.rest_client: Optional[UnifiedBybitClient] = None
        self.ws_manager: Optional[UnifiedWebSocketManager] = None
        self.cache = UnifiedDataCache(self.config)
        self.validator = DataValidator(self.config)
        
        # Data queues
        self.update_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.processing_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        
        # Subscriptions and callbacks
        self.subscriptions: Dict[str, List[str]] = defaultdict(list)  # symbol -> subscription_ids
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)  # data_type -> callbacks
        
        # Performance metrics
        self.metrics = DataMetrics()
        self.start_time: Optional[datetime] = None
        self.latency_samples: deque = deque(maxlen=1000)
        
        # Control flags
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info("Initialized Unified Market Data Pipeline")
    
    async def start(self, symbols: List[str]):
        """Start the market data pipeline"""
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        logger.info(f"Starting market data pipeline for {len(symbols)} symbols...")
        
        try:
            self.is_running = True
            self.start_time = datetime.now()
            self.shutdown_event.clear()
            
            # Initialize REST client
            self.rest_client = UnifiedBybitClient(self.credentials)
            await self.rest_client.connect()
            
            # Initialize WebSocket manager
            self.ws_manager = UnifiedWebSocketManager(self.credentials)
            await self.ws_manager.start()
            
            # Subscribe to market data streams
            await self._setup_subscriptions(symbols)
            
            # Start background processing tasks
            self.background_tasks = [
                asyncio.create_task(self._process_updates()),
                asyncio.create_task(self._batch_processor()),
                asyncio.create_task(self._periodic_cleanup()),
                asyncio.create_task(self._metrics_collector()),
            ]
            
            logger.info("Market data pipeline started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            self.is_running = False
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the market data pipeline"""
        if not self.is_running:
            logger.warning("Pipeline not running")
            return
        
        logger.info("Stopping market data pipeline...")
        
        try:
            # Signal shutdown
            self.is_running = False
            self.shutdown_event.set()
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.background_tasks.clear()
            
            # Disconnect clients
            if self.ws_manager:
                await self.ws_manager.stop()
                self.ws_manager = None
            
            if self.rest_client:
                await self.rest_client.disconnect()
                self.rest_client = None
            
            logger.info("Market data pipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
    
    async def _setup_subscriptions(self, symbols: List[str]):
        """Setup WebSocket subscriptions for symbols"""
        if not self.ws_manager:
            raise RuntimeError("WebSocket manager not initialized")
        
        # Subscribe to ticker updates
        ticker_sub_id = await self.ws_manager.subscribe(
            topic='ticker',
            callback=self._handle_ticker_update,
            symbols=symbols,
            stream_type=StreamType.PUBLIC
        )
        
        for symbol in symbols:
            self.subscriptions[symbol].append(ticker_sub_id)
        
        # Subscribe to orderbook updates
        orderbook_sub_id = await self.ws_manager.subscribe(
            topic='orderbook.1',
            callback=self._handle_orderbook_update,
            symbols=symbols,
            stream_type=StreamType.PUBLIC
        )
        
        for symbol in symbols:
            self.subscriptions[symbol].append(orderbook_sub_id)
        
        # Subscribe to trade updates
        trade_sub_id = await self.ws_manager.subscribe(
            topic='publicTrade',
            callback=self._handle_trade_update,
            symbols=symbols,
            stream_type=StreamType.PUBLIC
        )
        
        for symbol in symbols:
            self.subscriptions[symbol].append(trade_sub_id)
        
        logger.info(f"Setup subscriptions for {len(symbols)} symbols")
    
    async def _handle_ticker_update(self, data: Dict[str, Any]):
        """Handle ticker update from WebSocket"""
        try:
            ticker_data = data.get('data', {})
            if not ticker_data:
                return
            
            symbol = ticker_data.get('symbol')
            if not symbol:
                return
            
            # Create MarketData object
            market_data = MarketData(
                symbol=symbol,
                price=Decimal(ticker_data.get('lastPrice', '0')),
                bid=Decimal(ticker_data.get('bid1Price', '0')),
                ask=Decimal(ticker_data.get('ask1Price', '0')),
                volume_24h=Decimal(ticker_data.get('volume24h', '0')),
                high_24h=Decimal(ticker_data.get('highPrice24h', '0')),
                low_24h=Decimal(ticker_data.get('lowPrice24h', '0')),
                change_24h=Decimal(ticker_data.get('price24hPcnt', '0')),
                timestamp=datetime.now()
            )
            
            # Validate data quality
            quality = self.validator.validate_market_data(market_data)
            
            if quality != DataQuality.INVALID:
                # Cache the data
                self.cache.set_ticker(symbol, market_data)
                
                # Queue for processing
                await self.update_queue.put({
                    'type': 'ticker',
                    'symbol': symbol,
                    'data': market_data,
                    'quality': quality,
                    'timestamp': datetime.now()
                })
                
                self.metrics.successful_updates += 1
            else:
                self.metrics.failed_updates += 1
                logger.warning(f"Invalid ticker data for {symbol}")
            
            self.metrics.total_updates += 1
            
        except Exception as e:
            logger.error(f"Error handling ticker update: {e}")
            self.metrics.failed_updates += 1
    
    async def _handle_orderbook_update(self, data: Dict[str, Any]):
        """Handle orderbook update from WebSocket"""
        try:
            orderbook_data = data.get('data', {})
            if not orderbook_data:
                return
            
            symbol = orderbook_data.get('s')
            if not symbol:
                return
            
            # Create OrderBookData object
            orderbook = OrderBookData(
                symbol=symbol,
                bids=orderbook_data.get('b', []),
                asks=orderbook_data.get('a', []),
                timestamp=datetime.fromtimestamp(int(orderbook_data.get('ts', 0)) / 1000),
                update_id=int(orderbook_data.get('u', 0))
            )
            
            # Validate data quality
            quality = self.validator.validate_orderbook(orderbook)
            
            if quality != DataQuality.INVALID:
                # Cache the data
                self.cache.set_orderbook(symbol, orderbook)
                
                # Queue for processing
                await self.update_queue.put({
                    'type': 'orderbook',
                    'symbol': symbol,
                    'data': orderbook,
                    'quality': quality,
                    'timestamp': datetime.now()
                })
                
                self.metrics.successful_updates += 1
            else:
                self.metrics.failed_updates += 1
                logger.warning(f"Invalid orderbook data for {symbol}")
            
            self.metrics.total_updates += 1
            
        except Exception as e:
            logger.error(f"Error handling orderbook update: {e}")
            self.metrics.failed_updates += 1
    
    async def _handle_trade_update(self, data: Dict[str, Any]):
        """Handle trade update from WebSocket"""
        try:
            trades_data = data.get('data', [])
            if not trades_data:
                return
            
            for trade_data in trades_data:
                symbol = trade_data.get('s')
                if not symbol:
                    continue
                
                # Create TradeData object
                trade = TradeData(
                    symbol=symbol,
                    trade_id=trade_data.get('i', ''),
                    price=Decimal(trade_data.get('p', '0')),
                    quantity=Decimal(trade_data.get('v', '0')),
                    side=trade_data.get('S', ''),
                    timestamp=datetime.fromtimestamp(int(trade_data.get('T', 0)) / 1000)
                )
                
                # Cache the trade
                self.cache.add_trade(symbol, trade)
                
                # Queue for processing
                await self.update_queue.put({
                    'type': 'trade',
                    'symbol': symbol,
                    'data': trade,
                    'quality': DataQuality.GOOD,
                    'timestamp': datetime.now()
                })
                
                self.metrics.successful_updates += 1
                self.metrics.total_updates += 1
            
        except Exception as e:
            logger.error(f"Error handling trade update: {e}")
            self.metrics.failed_updates += 1
    
    async def _process_updates(self):
        """Process updates from the queue"""
        while self.is_running:
            try:
                # Get update from queue with timeout
                try:
                    update = await asyncio.wait_for(
                        self.update_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Calculate latency
                latency_ms = (datetime.now() - update['timestamp']).total_seconds() * 1000
                self.latency_samples.append(latency_ms)
                
                # Call registered callbacks
                data_type = update['type']
                for callback in self.callbacks.get(data_type, []):
                    try:
                        await callback(update)
                    except Exception as e:
                        logger.error(f"Error in {data_type} callback: {e}")
                
                # Mark task as done
                self.update_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing update: {e}")
                await asyncio.sleep(0.1)
    
    async def _batch_processor(self):
        """Process updates in batches for efficiency"""
        batch = []
        last_process_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Process batch if it's full or enough time has passed
                if (len(batch) >= self.config.batch_size or 
                    (batch and current_time - last_process_time > 1.0)):
                    
                    await self._process_batch(batch)
                    batch.clear()
                    last_process_time = current_time
                
                # Try to get more items for the batch
                try:
                    update = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=0.1
                    )
                    batch.append(update)
                except asyncio.TimeoutError:
                    pass
                
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of updates"""
        if not batch:
            return
        
        try:
            # Group updates by type
            grouped_updates = defaultdict(list)
            for update in batch:
                grouped_updates[update['type']].append(update)
            
            # Process each type
            for data_type, updates in grouped_updates.items():
                # Call batch callbacks if available
                batch_callbacks = self.callbacks.get(f"{data_type}_batch", [])
                for callback in batch_callbacks:
                    try:
                        await callback(updates)
                    except Exception as e:
                        logger.error(f"Error in {data_type} batch callback: {e}")
            
            logger.debug(f"Processed batch of {len(batch)} updates")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
    
    async def _periodic_cleanup(self):
        """Perform periodic cleanup tasks"""
        while self.is_running:
            try:
                # Cache cleanup
                self.cache.cleanup()
                
                # Update metrics
                self.metrics.last_update_time = datetime.now()
                
                if self.latency_samples:
                    self.metrics.avg_latency_ms = sum(self.latency_samples) / len(self.latency_samples)
                
                self.metrics.data_quality_score = self.validator.get_overall_quality_score()
                
                # Sleep for cleanup interval
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collector(self):
        """Collect and log metrics"""
        while self.is_running:
            try:
                cache_stats = self.cache.get_stats()
                
                logger.info(f"Pipeline Metrics - "
                          f"Updates: {self.metrics.total_updates}, "
                          f"Success Rate: {(self.metrics.successful_updates / max(1, self.metrics.total_updates) * 100):.1f}%, "
                          f"Cache Hit Rate: {cache_stats['hit_rate_pct']:.1f}%, "
                          f"Avg Latency: {self.metrics.avg_latency_ms:.1f}ms, "
                          f"Quality Score: {self.metrics.data_quality_score:.1f}")
                
                await asyncio.sleep(300)  # Log every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(300)
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    async def get_market_data(self, symbol: str, use_cache: bool = True) -> Optional[MarketData]:
        """Get current market data for a symbol"""
        # Try cache first if enabled
        if use_cache:
            cached_data = self.cache.get_ticker(symbol)
            if cached_data:
                self.metrics.cache_hits += 1
                return cached_data
            self.metrics.cache_misses += 1
        
        # Fallback to REST API
        if self.rest_client:
            try:
                market_data = await self.rest_client.get_market_data(symbol)
                
                # Cache the result
                self.cache.set_ticker(symbol, market_data)
                
                return market_data
            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {e}")
        
        return None
    
    async def get_orderbook(self, symbol: str, use_cache: bool = True) -> Optional[OrderBookData]:
        """Get orderbook data for a symbol"""
        # Try cache first if enabled
        if use_cache:
            cached_data = self.cache.get_orderbook(symbol)
            if cached_data:
                self.metrics.cache_hits += 1
                return cached_data
            self.metrics.cache_misses += 1
        
        # Fallback to REST API
        if self.rest_client:
            try:
                orderbook_data = await self.rest_client.get_orderbook(symbol)
                
                # Cache the result
                self.cache.set_orderbook(symbol, orderbook_data)
                
                return orderbook_data
            except Exception as e:
                logger.error(f"Error fetching orderbook for {symbol}: {e}")
        
        return None
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """Get recent trades for a symbol"""
        return self.cache.get_recent_trades(symbol, limit)
    
    async def get_historical_data(
        self,
        symbol: str,
        interval: str = '1',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get historical kline data"""
        if not self.rest_client:
            raise RuntimeError("REST client not available")
        
        return await self.rest_client.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    def add_callback(self, data_type: str, callback: Callable):
        """Add callback for data updates"""
        self.callbacks[data_type].append(callback)
        logger.info(f"Added callback for {data_type}")
    
    def remove_callback(self, data_type: str, callback: Callable):
        """Remove callback for data updates"""
        if callback in self.callbacks[data_type]:
            self.callbacks[data_type].remove(callback)
            logger.info(f"Removed callback for {data_type}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status and metrics"""
        return {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'metrics': {
                'total_updates': self.metrics.total_updates,
                'successful_updates': self.metrics.successful_updates,
                'failed_updates': self.metrics.failed_updates,
                'success_rate_pct': (self.metrics.successful_updates / max(1, self.metrics.total_updates) * 100),
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses,
                'cache_hit_rate_pct': (self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses) * 100),
                'avg_latency_ms': self.metrics.avg_latency_ms,
                'data_quality_score': self.metrics.data_quality_score,
                'last_update_time': self.metrics.last_update_time.isoformat() if self.metrics.last_update_time else None
            },
            'cache_stats': self.cache.get_stats(),
            'subscriptions': {symbol: len(subs) for symbol, subs in self.subscriptions.items()},
            'queue_sizes': {
                'update_queue': self.update_queue.qsize(),
                'processing_queue': self.processing_queue.qsize()
            }
        }

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

async def create_market_data_pipeline(
    credentials: BybitCredentials,
    symbols: List[str],
    config: Optional[PipelineConfig] = None
) -> UnifiedMarketDataPipeline:
    """Factory function to create and start market data pipeline"""
    
    pipeline = UnifiedMarketDataPipeline(credentials, config)
    await pipeline.start(symbols)
    
    return pipeline

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'UnifiedMarketDataPipeline',
    'UnifiedDataCache',
    'DataValidator',
    'PipelineConfig',
    'DataSource',
    'DataQuality',
    'CacheStrategy',
    'DataPoint',
    'DataMetrics',
    'create_market_data_pipeline'
]