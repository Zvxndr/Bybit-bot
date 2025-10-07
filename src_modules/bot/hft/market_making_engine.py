"""
Advanced Market Making Engine for High-Frequency Trading.
Provides sophisticated market making algorithms with dynamic spread management and inventory control.
"""

import asyncio
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from collections import deque
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class MarketMakingStrategy(Enum):
    """Market making strategies."""
    SIMPLE_SPREAD = "simple_spread"
    ADAPTIVE_SPREAD = "adaptive_spread"
    INVENTORY_BASED = "inventory_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    MOMENTUM_AWARE = "momentum_aware"
    ORDERBOOK_IMBALANCE = "orderbook_imbalance"
    ADVERSE_SELECTION = "adverse_selection"
    OPTIMAL_EXECUTION = "optimal_execution"

class QuoteStatus(Enum):
    """Quote status."""
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PENDING = "pending"

class InventoryDirection(Enum):
    """Inventory direction."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class MarketQuote:
    """Market making quote."""
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    quote_id: str
    order_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: QuoteStatus = QuoteStatus.PENDING
    spread_bps: float = 0.0
    theoretical_edge: float = 0.0
    risk_adjustment: float = 0.0
    inventory_adjustment: float = 0.0
    priority_level: int = 1

@dataclass
class InventoryPosition:
    """Inventory position tracking."""
    symbol: str
    quantity: float
    average_price: float
    market_value: float
    unrealized_pnl: float
    target_quantity: float
    max_quantity: float
    direction: InventoryDirection
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class SpreadMetrics:
    """Spread performance metrics."""
    symbol: str
    bid_spread_bps: float
    ask_spread_bps: float
    effective_spread_bps: float
    quoted_spread_bps: float
    realized_spread_bps: float
    adverse_selection_cost_bps: float
    inventory_cost_bps: float
    fill_rate: float
    quote_duration_ms: float
    profitability: float
    measurement_period: timedelta
    timestamp: datetime = field(default_factory=datetime.now)

class MarketMakingEngine:
    """Advanced market making engine with multiple strategies."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Market making configuration
        self.mm_config = {
            'default_strategy': MarketMakingStrategy.ADAPTIVE_SPREAD,
            'min_spread_bps': 2,      # Minimum 2 bps spread
            'max_spread_bps': 50,     # Maximum 50 bps spread
            'default_spread_bps': 5,  # Default 5 bps spread
            'max_inventory_ratio': 0.3,  # Max 30% of available capital
            'inventory_target': 0.0,  # Target neutral inventory
            'risk_limit_per_symbol': 100000,  # $100k risk limit per symbol
            'quote_refresh_ms': 100,  # Refresh quotes every 100ms
            'min_quote_size': 0.001,  # Minimum quote size
            'max_quote_size': 1.0,    # Maximum quote size
            'skew_adjustment': True,  # Enable inventory skewing
            'adverse_selection_protection': True,
            'momentum_adjustment': True,
            'volatility_adjustment': True
        }
        
        # Market making state
        self.active_quotes: Dict[str, Dict[str, MarketQuote]] = {}  # symbol -> {side -> quote}
        self.inventory_positions: Dict[str, InventoryPosition] = {}
        self.spread_metrics: Dict[str, SpreadMetrics] = {}
        self.quote_history: Dict[str, deque] = {}
        
        # Market data
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.order_book_data: Dict[str, Dict[str, Any]] = {}
        
        # Strategy components
        self.spread_manager = SpreadManager(self)
        self.inventory_manager = InventoryManager(self)
        self.quote_manager = QuoteManager(self)
        
        # Threading
        self.mm_lock = threading.Lock()
        self.running = False
        self.mm_task = None
        
        self.logger.info("MarketMakingEngine initialized")
    
    async def start_market_making(self, symbols: List[str], strategy: MarketMakingStrategy = None):
        """Start market making for specified symbols."""
        try:
            if self.running:
                return
            
            self.running = True
            strategy = strategy or self.mm_config['default_strategy']
            
            # Initialize positions and quotes for symbols
            for symbol in symbols:
                self.active_quotes[symbol] = {}
                self.inventory_positions[symbol] = InventoryPosition(
                    symbol=symbol,
                    quantity=0.0,
                    average_price=0.0,
                    market_value=0.0,
                    unrealized_pnl=0.0,
                    target_quantity=0.0,
                    max_quantity=self.mm_config['max_quote_size'],
                    direction=InventoryDirection.NEUTRAL
                )
                self.quote_history[symbol] = deque(maxlen=1000)
            
            # Start market making loop
            self.mm_task = asyncio.create_task(
                self._market_making_loop(symbols, strategy)
            )
            
            self.logger.info(f"Market making started for {len(symbols)} symbols with {strategy.value} strategy")
            
        except Exception as e:
            self.logger.error(f"Failed to start market making: {e}")
            self.running = False
            raise
    
    async def stop_market_making(self):
        """Stop market making."""
        try:
            self.running = False
            
            # Cancel all active quotes
            await self._cancel_all_quotes()
            
            # Stop market making task
            if self.mm_task:
                self.mm_task.cancel()
                try:
                    await self.mm_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Market making stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop market making: {e}")
    
    async def update_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """Update market data for symbol."""
        try:
            with self.mm_lock:
                self.market_data[symbol] = market_data
                
                # Update inventory positions
                if symbol in self.inventory_positions:
                    position = self.inventory_positions[symbol]
                    current_price = market_data.get('mid_price', 0)
                    
                    if current_price > 0:
                        position.market_value = position.quantity * current_price
                        if position.quantity != 0:
                            position.unrealized_pnl = (current_price - position.average_price) * position.quantity
                        position.last_updated = datetime.now()
                        
        except Exception as e:
            self.logger.error(f"Failed to update market data for {symbol}: {e}")
    
    async def update_order_book(self, symbol: str, order_book_data: Dict[str, Any]):
        """Update order book data for symbol."""
        try:
            with self.mm_lock:
                self.order_book_data[symbol] = order_book_data
                
        except Exception as e:
            self.logger.error(f"Failed to update order book for {symbol}: {e}")
    
    async def _market_making_loop(self, symbols: List[str], strategy: MarketMakingStrategy):
        """Main market making loop."""
        try:
            while self.running:
                start_time = time.perf_counter()
                
                # Update quotes for all symbols
                for symbol in symbols:
                    if symbol in self.market_data:
                        await self._update_quotes_for_symbol(symbol, strategy)
                
                # Calculate loop duration and wait
                loop_duration_ms = (time.perf_counter() - start_time) * 1000
                refresh_interval_ms = self.mm_config['quote_refresh_ms']
                
                if loop_duration_ms < refresh_interval_ms:
                    await asyncio.sleep((refresh_interval_ms - loop_duration_ms) / 1000)
                else:
                    self.logger.warning(f"Market making loop took {loop_duration_ms:.1f}ms (target: {refresh_interval_ms}ms)")
                    await asyncio.sleep(0.001)  # Small delay to prevent tight loop
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Market making loop error: {e}")
    
    async def _update_quotes_for_symbol(self, symbol: str, strategy: MarketMakingStrategy):
        """Update quotes for a specific symbol."""
        try:
            market_data = self.market_data[symbol]
            current_price = market_data.get('mid_price', 0)
            
            if current_price <= 0:
                return
            
            # Calculate new quotes based on strategy
            if strategy == MarketMakingStrategy.SIMPLE_SPREAD:
                quotes = await self._calculate_simple_spread_quotes(symbol, market_data)
            elif strategy == MarketMakingStrategy.ADAPTIVE_SPREAD:
                quotes = await self._calculate_adaptive_spread_quotes(symbol, market_data)
            elif strategy == MarketMakingStrategy.INVENTORY_BASED:
                quotes = await self._calculate_inventory_based_quotes(symbol, market_data)
            elif strategy == MarketMakingStrategy.VOLATILITY_ADJUSTED:
                quotes = await self._calculate_volatility_adjusted_quotes(symbol, market_data)
            elif strategy == MarketMakingStrategy.MOMENTUM_AWARE:
                quotes = await self._calculate_momentum_aware_quotes(symbol, market_data)
            elif strategy == MarketMakingStrategy.ORDERBOOK_IMBALANCE:
                quotes = await self._calculate_imbalance_quotes(symbol, market_data)
            else:
                quotes = await self._calculate_adaptive_spread_quotes(symbol, market_data)
            
            # Update active quotes
            await self._update_active_quotes(symbol, quotes)
            
        except Exception as e:
            self.logger.error(f"Failed to update quotes for {symbol}: {e}")
    
    async def _calculate_simple_spread_quotes(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, MarketQuote]:
        """Calculate simple spread quotes."""
        try:
            mid_price = market_data['mid_price']
            spread_bps = self.mm_config['default_spread_bps']
            half_spread = mid_price * (spread_bps / 10000) / 2
            
            quote_size = self.mm_config['min_quote_size']
            
            quotes = {
                'buy': MarketQuote(
                    symbol=symbol,
                    side='buy',
                    price=mid_price - half_spread,
                    quantity=quote_size,
                    quote_id=f"{symbol}_buy_{int(time.time() * 1000)}",
                    spread_bps=spread_bps
                ),
                'sell': MarketQuote(
                    symbol=symbol,
                    side='sell',
                    price=mid_price + half_spread,
                    quantity=quote_size,
                    quote_id=f"{symbol}_sell_{int(time.time() * 1000)}",
                    spread_bps=spread_bps
                )
            }
            
            return quotes
            
        except Exception as e:
            self.logger.error(f"Simple spread calculation failed for {symbol}: {e}")
            return {}
    
    async def _calculate_adaptive_spread_quotes(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, MarketQuote]:
        """Calculate adaptive spread quotes based on market conditions."""
        try:
            mid_price = market_data['mid_price']
            
            # Base spread
            base_spread_bps = self.mm_config['default_spread_bps']
            
            # Volatility adjustment
            volatility = market_data.get('volatility', 0.02)
            vol_adjustment = max(0.5, min(2.0, volatility / 0.02))  # Scale by expected 2% vol
            
            # Volume adjustment
            volume = market_data.get('volume', 1000)
            avg_volume = market_data.get('avg_volume', 1000)
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            vol_adjustment_factor = max(0.8, min(1.5, 2.0 - volume_ratio))
            
            # Order book imbalance adjustment
            imbalance = await self._calculate_order_book_imbalance(symbol)
            imbalance_adjustment = 1.0 + (imbalance * 0.2)  # Up to 20% adjustment
            
            # Calculate adaptive spread
            adaptive_spread_bps = base_spread_bps * vol_adjustment * vol_adjustment_factor * imbalance_adjustment
            adaptive_spread_bps = max(self.mm_config['min_spread_bps'], 
                                    min(self.mm_config['max_spread_bps'], adaptive_spread_bps))
            
            half_spread = mid_price * (adaptive_spread_bps / 10000) / 2
            
            # Calculate quote sizes
            quote_size = await self._calculate_optimal_quote_size(symbol, market_data)
            
            # Inventory skew
            inventory_skew = await self._calculate_inventory_skew(symbol)
            
            quotes = {
                'buy': MarketQuote(
                    symbol=symbol,
                    side='buy',
                    price=mid_price - half_spread + inventory_skew,
                    quantity=quote_size,
                    quote_id=f"{symbol}_buy_{int(time.time() * 1000)}",
                    spread_bps=adaptive_spread_bps,
                    theoretical_edge=half_spread / mid_price,
                    inventory_adjustment=inventory_skew
                ),
                'sell': MarketQuote(
                    symbol=symbol,
                    side='sell',
                    price=mid_price + half_spread + inventory_skew,
                    quantity=quote_size,
                    quote_id=f"{symbol}_sell_{int(time.time() * 1000)}",
                    spread_bps=adaptive_spread_bps,
                    theoretical_edge=half_spread / mid_price,
                    inventory_adjustment=inventory_skew
                )
            }
            
            return quotes
            
        except Exception as e:
            self.logger.error(f"Adaptive spread calculation failed for {symbol}: {e}")
            return await self._calculate_simple_spread_quotes(symbol, market_data)
    
    async def _calculate_inventory_based_quotes(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, MarketQuote]:
        """Calculate inventory-based quotes with skewing."""
        try:
            mid_price = market_data['mid_price']
            position = self.inventory_positions.get(symbol)
            
            if not position:
                return await self._calculate_adaptive_spread_quotes(symbol, market_data)
            
            # Calculate inventory imbalance
            max_inventory = position.max_quantity
            current_inventory = position.quantity
            inventory_ratio = current_inventory / max_inventory if max_inventory > 0 else 0
            
            # Base spread
            base_spread_bps = self.mm_config['default_spread_bps']
            
            # Inventory-based spread adjustment
            inventory_spread_multiplier = 1.0 + abs(inventory_ratio) * 0.5  # Up to 50% increase
            adjusted_spread_bps = base_spread_bps * inventory_spread_multiplier
            
            half_spread = mid_price * (adjusted_spread_bps / 10000) / 2
            
            # Inventory skewing - wider spreads on the side we're long
            if inventory_ratio > 0.1:  # Long inventory
                buy_spread_multiplier = 0.8  # Tighter buy spread
                sell_spread_multiplier = 1.4  # Wider sell spread
            elif inventory_ratio < -0.1:  # Short inventory
                buy_spread_multiplier = 1.4  # Wider buy spread
                sell_spread_multiplier = 0.8  # Tighter sell spread
            else:
                buy_spread_multiplier = 1.0
                sell_spread_multiplier = 1.0
            
            # Quote sizes - larger on the side we want to trade
            base_size = await self._calculate_optimal_quote_size(symbol, market_data)
            
            if inventory_ratio > 0.1:  # Long - want to sell more
                buy_size = base_size * 0.7
                sell_size = base_size * 1.3
            elif inventory_ratio < -0.1:  # Short - want to buy more
                buy_size = base_size * 1.3
                sell_size = base_size * 0.7
            else:
                buy_size = sell_size = base_size
            
            quotes = {
                'buy': MarketQuote(
                    symbol=symbol,
                    side='buy',
                    price=mid_price - (half_spread * buy_spread_multiplier),
                    quantity=buy_size,
                    quote_id=f"{symbol}_buy_{int(time.time() * 1000)}",
                    spread_bps=adjusted_spread_bps,
                    inventory_adjustment=inventory_ratio
                ),
                'sell': MarketQuote(
                    symbol=symbol,
                    side='sell',
                    price=mid_price + (half_spread * sell_spread_multiplier),
                    quantity=sell_size,
                    quote_id=f"{symbol}_sell_{int(time.time() * 1000)}",
                    spread_bps=adjusted_spread_bps,
                    inventory_adjustment=inventory_ratio
                )
            }
            
            return quotes
            
        except Exception as e:
            self.logger.error(f"Inventory-based calculation failed for {symbol}: {e}")
            return await self._calculate_adaptive_spread_quotes(symbol, market_data)
    
    async def _calculate_volatility_adjusted_quotes(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, MarketQuote]:
        """Calculate volatility-adjusted quotes."""
        try:
            mid_price = market_data['mid_price']
            
            # Get volatility metrics
            current_vol = market_data.get('volatility', 0.02)
            avg_vol = market_data.get('avg_volatility', 0.02)
            vol_of_vol = market_data.get('vol_of_vol', 0.3)
            
            # Base spread
            base_spread_bps = self.mm_config['default_spread_bps']
            
            # Volatility adjustment
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            vol_adjustment = 0.5 + (vol_ratio * 0.8)  # 0.5x to 1.3x adjustment
            
            # Volatility of volatility adjustment (uncertainty premium)
            vov_adjustment = 1.0 + (vol_of_vol * 0.3)  # Up to 30% increase
            
            # Calculate adjusted spread
            vol_adjusted_spread_bps = base_spread_bps * vol_adjustment * vov_adjustment
            vol_adjusted_spread_bps = max(self.mm_config['min_spread_bps'], 
                                        min(self.mm_config['max_spread_bps'], vol_adjusted_spread_bps))
            
            half_spread = mid_price * (vol_adjusted_spread_bps / 10000) / 2
            
            # Size adjustment - smaller sizes in high volatility
            base_size = await self._calculate_optimal_quote_size(symbol, market_data)
            vol_size_adjustment = max(0.5, min(1.5, 1.0 / vol_ratio))
            adjusted_size = base_size * vol_size_adjustment
            
            quotes = {
                'buy': MarketQuote(
                    symbol=symbol,
                    side='buy',
                    price=mid_price - half_spread,
                    quantity=adjusted_size,
                    quote_id=f"{symbol}_buy_{int(time.time() * 1000)}",
                    spread_bps=vol_adjusted_spread_bps,
                    risk_adjustment=vol_adjustment * vov_adjustment
                ),
                'sell': MarketQuote(
                    symbol=symbol,
                    side='sell',
                    price=mid_price + half_spread,
                    quantity=adjusted_size,
                    quote_id=f"{symbol}_sell_{int(time.time() * 1000)}",
                    spread_bps=vol_adjusted_spread_bps,
                    risk_adjustment=vol_adjustment * vov_adjustment
                )
            }
            
            return quotes
            
        except Exception as e:
            self.logger.error(f"Volatility-adjusted calculation failed for {symbol}: {e}")
            return await self._calculate_adaptive_spread_quotes(symbol, market_data)
    
    async def _calculate_momentum_aware_quotes(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, MarketQuote]:
        """Calculate momentum-aware quotes."""
        try:
            mid_price = market_data['mid_price']
            
            # Get momentum indicators
            short_momentum = market_data.get('momentum_1m', 0.0)
            medium_momentum = market_data.get('momentum_5m', 0.0)
            price_acceleration = market_data.get('price_acceleration', 0.0)
            
            # Base spread
            base_spread_bps = self.mm_config['default_spread_bps']
            
            # Momentum adjustment - wider spreads in strong momentum
            momentum_strength = abs(short_momentum) + abs(medium_momentum) * 0.5
            momentum_adjustment = 1.0 + (momentum_strength * 2.0)  # Up to 3x spread
            
            # Directional skew based on momentum
            if short_momentum > 0.001:  # Upward momentum
                buy_adjustment = 0.9   # Tighter buy spread
                sell_adjustment = 1.2  # Wider sell spread
            elif short_momentum < -0.001:  # Downward momentum
                buy_adjustment = 1.2   # Wider buy spread
                sell_adjustment = 0.9  # Tighter sell spread
            else:
                buy_adjustment = sell_adjustment = 1.0
            
            # Calculate spreads
            momentum_spread_bps = base_spread_bps * momentum_adjustment
            momentum_spread_bps = max(self.mm_config['min_spread_bps'], 
                                    min(self.mm_config['max_spread_bps'], momentum_spread_bps))
            
            half_spread = mid_price * (momentum_spread_bps / 10000) / 2
            
            # Quote size adjustment
            base_size = await self._calculate_optimal_quote_size(symbol, market_data)
            
            quotes = {
                'buy': MarketQuote(
                    symbol=symbol,
                    side='buy',
                    price=mid_price - (half_spread * buy_adjustment),
                    quantity=base_size,
                    quote_id=f"{symbol}_buy_{int(time.time() * 1000)}",
                    spread_bps=momentum_spread_bps,
                    risk_adjustment=momentum_adjustment
                ),
                'sell': MarketQuote(
                    symbol=symbol,
                    side='sell',
                    price=mid_price + (half_spread * sell_adjustment),
                    quantity=base_size,
                    quote_id=f"{symbol}_sell_{int(time.time() * 1000)}",
                    spread_bps=momentum_spread_bps,
                    risk_adjustment=momentum_adjustment
                )
            }
            
            return quotes
            
        except Exception as e:
            self.logger.error(f"Momentum-aware calculation failed for {symbol}: {e}")
            return await self._calculate_adaptive_spread_quotes(symbol, market_data)
    
    async def _calculate_imbalance_quotes(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, MarketQuote]:
        """Calculate quotes based on order book imbalance."""
        try:
            mid_price = market_data['mid_price']
            
            # Get order book imbalance
            imbalance = await self._calculate_order_book_imbalance(symbol)
            imbalance_strength = abs(imbalance)
            
            # Base spread
            base_spread_bps = self.mm_config['default_spread_bps']
            
            # Imbalance adjustment
            imbalance_adjustment = 1.0 + (imbalance_strength * 0.5)  # Up to 50% increase
            
            # Directional skew
            if imbalance > 0.2:  # Buy-side heavy
                buy_adjustment = 1.3   # Wider buy spread
                sell_adjustment = 0.8  # Tighter sell spread
            elif imbalance < -0.2:  # Sell-side heavy
                buy_adjustment = 0.8   # Tighter buy spread
                sell_adjustment = 1.3  # Wider sell spread
            else:
                buy_adjustment = sell_adjustment = 1.0
            
            # Calculate spreads
            imbalance_spread_bps = base_spread_bps * imbalance_adjustment
            imbalance_spread_bps = max(self.mm_config['min_spread_bps'], 
                                     min(self.mm_config['max_spread_bps'], imbalance_spread_bps))
            
            half_spread = mid_price * (imbalance_spread_bps / 10000) / 2
            
            # Quote sizes - larger on imbalanced side
            base_size = await self._calculate_optimal_quote_size(symbol, market_data)
            
            if imbalance > 0.2:  # More selling opportunity
                buy_size = base_size * 0.8
                sell_size = base_size * 1.2
            elif imbalance < -0.2:  # More buying opportunity
                buy_size = base_size * 1.2
                sell_size = base_size * 0.8
            else:
                buy_size = sell_size = base_size
            
            quotes = {
                'buy': MarketQuote(
                    symbol=symbol,
                    side='buy',
                    price=mid_price - (half_spread * buy_adjustment),
                    quantity=buy_size,
                    quote_id=f"{symbol}_buy_{int(time.time() * 1000)}",
                    spread_bps=imbalance_spread_bps,
                    theoretical_edge=imbalance
                ),
                'sell': MarketQuote(
                    symbol=symbol,
                    side='sell',
                    price=mid_price + (half_spread * sell_adjustment),
                    quantity=sell_size,
                    quote_id=f"{symbol}_sell_{int(time.time() * 1000)}",
                    spread_bps=imbalance_spread_bps,
                    theoretical_edge=imbalance
                )
            }
            
            return quotes
            
        except Exception as e:
            self.logger.error(f"Imbalance-based calculation failed for {symbol}: {e}")
            return await self._calculate_adaptive_spread_quotes(symbol, market_data)
    
    async def _calculate_order_book_imbalance(self, symbol: str) -> float:
        """Calculate order book imbalance."""
        try:
            order_book = self.order_book_data.get(symbol, {})
            
            if not order_book:
                return 0.0
            
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return 0.0
            
            # Calculate imbalance using top 5 levels
            bid_volume = sum(bid[1] for bid in bids[:5])
            ask_volume = sum(ask[1] for ask in asks[:5])
            
            total_volume = bid_volume + ask_volume
            
            if total_volume == 0:
                return 0.0
            
            # Imbalance: positive = buy-heavy, negative = sell-heavy
            imbalance = (bid_volume - ask_volume) / total_volume
            
            return imbalance
            
        except Exception as e:
            self.logger.error(f"Order book imbalance calculation failed for {symbol}: {e}")
            return 0.0
    
    async def _calculate_optimal_quote_size(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate optimal quote size."""
        try:
            # Base size
            base_size = self.mm_config['min_quote_size']
            
            # Volume-based adjustment
            volume = market_data.get('volume', 1000)
            avg_volume = market_data.get('avg_volume', 1000)
            
            if avg_volume > 0:
                volume_ratio = volume / avg_volume
                size_multiplier = max(0.5, min(2.0, volume_ratio))
                base_size *= size_multiplier
            
            # Risk-based adjustment
            volatility = market_data.get('volatility', 0.02)
            risk_multiplier = max(0.3, min(1.5, 0.02 / volatility))
            base_size *= risk_multiplier
            
            # Ensure within limits
            base_size = max(self.mm_config['min_quote_size'], 
                          min(self.mm_config['max_quote_size'], base_size))
            
            return base_size
            
        except Exception as e:
            self.logger.error(f"Optimal quote size calculation failed for {symbol}: {e}")
            return self.mm_config['min_quote_size']
    
    async def _calculate_inventory_skew(self, symbol: str) -> float:
        """Calculate inventory-based price skew."""
        try:
            position = self.inventory_positions.get(symbol)
            
            if not position or position.max_quantity == 0:
                return 0.0
            
            inventory_ratio = position.quantity / position.max_quantity
            
            # Skew proportional to inventory imbalance
            # Positive skew pushes prices up (for short positions)
            # Negative skew pushes prices down (for long positions)
            max_skew_bps = 5  # Maximum 5 bps skew
            skew_bps = -inventory_ratio * max_skew_bps
            
            market_data = self.market_data.get(symbol, {})
            mid_price = market_data.get('mid_price', 0)
            
            if mid_price > 0:
                return mid_price * (skew_bps / 10000)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Inventory skew calculation failed for {symbol}: {e}")
            return 0.0
    
    async def _update_active_quotes(self, symbol: str, new_quotes: Dict[str, MarketQuote]):
        """Update active quotes for symbol."""
        try:
            current_quotes = self.active_quotes.get(symbol, {})
            
            # Check if quotes need updating
            for side, new_quote in new_quotes.items():
                current_quote = current_quotes.get(side)
                
                if not current_quote or self._should_update_quote(current_quote, new_quote):
                    # Cancel old quote if exists
                    if current_quote and current_quote.status == QuoteStatus.ACTIVE:
                        await self._cancel_quote(current_quote)
                    
                    # Submit new quote
                    await self._submit_quote(new_quote)
                    current_quotes[side] = new_quote
            
            self.active_quotes[symbol] = current_quotes
            
        except Exception as e:
            self.logger.error(f"Failed to update active quotes for {symbol}: {e}")
    
    def _should_update_quote(self, current_quote: MarketQuote, new_quote: MarketQuote) -> bool:
        """Check if quote should be updated."""
        try:
            # Price difference threshold (0.01%)
            price_threshold = current_quote.price * 0.0001
            
            if abs(current_quote.price - new_quote.price) > price_threshold:
                return True
            
            # Quantity difference threshold (5%)
            if abs(current_quote.quantity - new_quote.quantity) / current_quote.quantity > 0.05:
                return True
            
            # Time-based refresh (every 10 seconds)
            if (datetime.now() - current_quote.timestamp).total_seconds() > 10:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Quote update check failed: {e}")
            return True  # Default to updating on error
    
    async def _submit_quote(self, quote: MarketQuote):
        """Submit quote to exchange (simulation)."""
        try:
            # This would integrate with actual exchange API
            quote.status = QuoteStatus.ACTIVE
            quote.order_id = f"order_{quote.quote_id}"
            
            # Store in quote history
            if quote.symbol in self.quote_history:
                self.quote_history[quote.symbol].append(quote)
            
            self.logger.debug(f"Quote submitted: {quote.symbol} {quote.side} {quote.quantity}@{quote.price}")
            
        except Exception as e:
            self.logger.error(f"Failed to submit quote: {e}")
            quote.status = QuoteStatus.REJECTED
    
    async def _cancel_quote(self, quote: MarketQuote):
        """Cancel active quote."""
        try:
            # This would integrate with actual exchange API
            quote.status = QuoteStatus.CANCELLED
            
            self.logger.debug(f"Quote cancelled: {quote.symbol} {quote.side} {quote.order_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to cancel quote: {e}")
    
    async def _cancel_all_quotes(self):
        """Cancel all active quotes."""
        try:
            for symbol_quotes in self.active_quotes.values():
                for quote in symbol_quotes.values():
                    if quote.status == QuoteStatus.ACTIVE:
                        await self._cancel_quote(quote)
            
            self.logger.info("All quotes cancelled")
            
        except Exception as e:
            self.logger.error(f"Failed to cancel all quotes: {e}")
    
    def get_market_making_summary(self) -> Dict[str, Any]:
        """Get market making performance summary."""
        try:
            total_quotes = sum(len(history) for history in self.quote_history.values())
            active_symbols = len([s for s, quotes in self.active_quotes.items() if quotes])
            
            # Calculate total inventory value
            total_inventory_value = sum(
                pos.market_value for pos in self.inventory_positions.values()
            )
            
            # Calculate total PnL
            total_pnl = sum(
                pos.unrealized_pnl for pos in self.inventory_positions.values()
            )
            
            return {
                'running': self.running,
                'active_symbols': active_symbols,
                'total_quotes_submitted': total_quotes,
                'total_inventory_value': total_inventory_value,
                'total_unrealized_pnl': total_pnl,
                'avg_spread_bps': np.mean([
                    metrics.effective_spread_bps 
                    for metrics in self.spread_metrics.values()
                ]) if self.spread_metrics else 0,
                'inventory_positions': {
                    symbol: {
                        'quantity': pos.quantity,
                        'market_value': pos.market_value,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'direction': pos.direction.value
                    }
                    for symbol, pos in self.inventory_positions.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate market making summary: {e}")
            return {'error': 'Unable to generate summary'}


class SpreadManager:
    """Manages spread calculations and optimizations."""
    
    def __init__(self, mm_engine):
        self.mm_engine = mm_engine
        self.logger = TradingLogger()
    
    async def calculate_optimal_spread(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate optimal spread for symbol."""
        try:
            # This is a simplified optimal spread model
            # In practice, would use more sophisticated models
            
            volatility = market_data.get('volatility', 0.02)
            volume = market_data.get('volume', 1000)
            avg_volume = market_data.get('avg_volume', 1000)
            
            # Base spread from volatility
            vol_spread_bps = volatility * 100  # 2% vol = 2 bps base spread
            
            # Volume adjustment
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            volume_adjustment = max(0.5, min(2.0, 1.0 / volume_ratio))
            
            # Optimal spread
            optimal_spread_bps = vol_spread_bps * volume_adjustment
            
            # Apply limits
            min_spread = self.mm_engine.mm_config['min_spread_bps']
            max_spread = self.mm_engine.mm_config['max_spread_bps']
            
            return max(min_spread, min(max_spread, optimal_spread_bps))
            
        except Exception as e:
            self.logger.error(f"Optimal spread calculation failed for {symbol}: {e}")
            return self.mm_engine.mm_config['default_spread_bps']


class InventoryManager:
    """Manages inventory positions and risk."""
    
    def __init__(self, mm_engine):
        self.mm_engine = mm_engine
        self.logger = TradingLogger()
    
    async def update_position(self, symbol: str, fill_quantity: float, fill_price: float):
        """Update position after fill."""
        try:
            position = self.mm_engine.inventory_positions.get(symbol)
            if not position:
                return
            
            # Update position
            old_quantity = position.quantity
            old_avg_price = position.average_price
            
            new_quantity = old_quantity + fill_quantity
            
            if new_quantity != 0:
                # Update average price
                total_cost = (old_quantity * old_avg_price) + (fill_quantity * fill_price)
                position.average_price = total_cost / new_quantity
            else:
                position.average_price = 0.0
            
            position.quantity = new_quantity
            position.last_updated = datetime.now()
            
            # Update direction
            if new_quantity > 0.01:
                position.direction = InventoryDirection.LONG
            elif new_quantity < -0.01:
                position.direction = InventoryDirection.SHORT
            else:
                position.direction = InventoryDirection.NEUTRAL
            
            self.logger.info(f"Position updated for {symbol}: {new_quantity} @ {position.average_price}")
            
        except Exception as e:
            self.logger.error(f"Failed to update position for {symbol}: {e}")
    
    def check_inventory_limits(self, symbol: str, trade_quantity: float) -> bool:
        """Check if trade would violate inventory limits."""
        try:
            position = self.mm_engine.inventory_positions.get(symbol)
            if not position:
                return True
            
            new_quantity = position.quantity + trade_quantity
            
            return abs(new_quantity) <= position.max_quantity
            
        except Exception as e:
            self.logger.error(f"Inventory limit check failed for {symbol}: {e}")
            return False


class QuoteManager:
    """Manages quote lifecycle and execution."""
    
    def __init__(self, mm_engine):
        self.mm_engine = mm_engine
        self.logger = TradingLogger()
    
    async def handle_fill(self, symbol: str, side: str, fill_quantity: float, fill_price: float):
        """Handle quote fill."""
        try:
            # Update inventory
            signed_quantity = fill_quantity if side == 'buy' else -fill_quantity
            await self.mm_engine.inventory_manager.update_position(symbol, signed_quantity, fill_price)
            
            # Update quote status
            active_quotes = self.mm_engine.active_quotes.get(symbol, {})
            if side in active_quotes:
                quote = active_quotes[side]
                quote.status = QuoteStatus.FILLED
            
            # Generate new quotes
            market_data = self.mm_engine.market_data.get(symbol, {})
            if market_data:
                strategy = self.mm_engine.mm_config['default_strategy']
                await self.mm_engine._update_quotes_for_symbol(symbol, strategy)
            
            self.logger.info(f"Fill handled: {symbol} {side} {fill_quantity}@{fill_price}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle fill for {symbol}: {e}")