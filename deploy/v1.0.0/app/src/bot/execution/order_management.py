"""
Core Order Management System for Cryptocurrency Trading.

This module provides the foundation for sophisticated trade execution including:

- Multiple order types (Market, Limit, Stop, Stop-Limit, OCO, Iceberg)
- Order state management with comprehensive tracking
- Order validation and risk checks
- Real-time order monitoring and updates
- Advanced order routing and execution strategies
- Slippage tracking and optimization
- Fill and partial fill handling
- Order cancellation and modification

The system is designed to work with multiple cryptocurrency exchanges
and provides a unified interface for order management across different
trading venues.
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from ..utils.logging import TradingLogger


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    OCO = "one_cancels_other"
    ICEBERG = "iceberg"
    TWAP = "time_weighted_average_price"
    VWAP = "volume_weighted_average_price"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"           # Order created but not sent
    SUBMITTED = "submitted"       # Order sent to exchange
    OPEN = "open"                 # Order accepted by exchange
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class TimeInForce(Enum):
    """Time in force enumeration."""
    GTC = "good_till_cancelled"   # Good till cancelled
    IOC = "immediate_or_cancel"   # Immediate or cancel
    FOK = "fill_or_kill"          # Fill or kill
    GTD = "good_till_date"        # Good till date
    DAY = "day"                   # Good for day


class OrderPriority(Enum):
    """Order priority enumeration."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class OrderFill:
    """Container for order fill information."""
    
    fill_id: str
    order_id: str
    timestamp: datetime
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_currency: str
    trade_id: Optional[str] = None
    is_maker: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.fill_id is None:
            self.fill_id = str(uuid.uuid4())


@dataclass
class Order:
    """Core order representation with comprehensive tracking."""
    
    # Basic order information
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    
    # Price information
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    trigger_price: Optional[Decimal] = None
    
    # Order parameters
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    post_only: bool = False
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Optional[Decimal] = None
    average_fill_price: Optional[Decimal] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Exchange information
    exchange_order_id: Optional[str] = None
    exchange: Optional[str] = None
    
    # Advanced features
    iceberg_quantity: Optional[Decimal] = None  # For iceberg orders
    parent_order_id: Optional[str] = None       # For OCO orders
    child_orders: List[str] = field(default_factory=list)
    
    # Execution tracking
    fills: List[OrderFill] = field(default_factory=list)
    fees_paid: Dict[str, Decimal] = field(default_factory=dict)
    
    # Metadata
    strategy_id: Optional[str] = None
    priority: OrderPriority = OrderPriority.NORMAL
    tags: Dict[str, str] = field(default_factory=dict)
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
        if self.client_order_id is None:
            self.client_order_id = str(uuid.uuid4())
        if self.order_id is None:
            self.order_id = self.client_order_id
    
    @property
    def is_buy(self) -> bool:
        """Check if order is a buy order."""
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """Check if order is a sell order."""
        return self.side == OrderSide.SELL
    
    @property
    def is_market_order(self) -> bool:
        """Check if order is a market order."""
        return self.order_type == OrderType.MARKET
    
    @property
    def is_limit_order(self) -> bool:
        """Check if order is a limit order."""
        return self.order_type == OrderType.LIMIT
    
    @property
    def is_stop_order(self) -> bool:
        """Check if order is a stop order."""
        return self.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return self.status == OrderStatus.PARTIALLY_FILLED
    
    @property
    def is_open(self) -> bool:
        """Check if order is open (can be filled)."""
        return self.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def is_closed(self) -> bool:
        """Check if order is closed (terminal state)."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                              OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.FAILED]
    
    @property
    def fill_percentage(self) -> float:
        """Get fill percentage (0-1)."""
        if self.quantity <= 0:
            return 0.0
        return float(self.filled_quantity / self.quantity)
    
    @property 
    def total_fees(self) -> Dict[str, Decimal]:
        """Get total fees paid by currency."""
        total_fees = defaultdict(Decimal)
        for fill in self.fills:
            total_fees[fill.fee_currency] += fill.fee
        return dict(total_fees)
    
    def add_fill(self, fill: OrderFill) -> None:
        """Add a fill to the order."""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        
        # Update average fill price
        if self.fills:
            total_value = sum(fill.price * fill.quantity for fill in self.fills)
            self.average_fill_price = total_value / self.filled_quantity
        
        # Update status
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_at = datetime.now()
    
    def update_status(self, new_status: OrderStatus, timestamp: Optional[datetime] = None) -> None:
        """Update order status."""
        self.status = new_status
        self.updated_at = timestamp or datetime.now()
        
        if new_status == OrderStatus.SUBMITTED and not self.submitted_at:
            self.submitted_at = self.updated_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary representation."""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': str(self.quantity),
            'price': str(self.price) if self.price else None,
            'stop_price': str(self.stop_price) if self.stop_price else None,
            'status': self.status.value,
            'filled_quantity': str(self.filled_quantity),
            'remaining_quantity': str(self.remaining_quantity),
            'average_fill_price': str(self.average_fill_price) if self.average_fill_price else None,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'exchange': self.exchange,
            'exchange_order_id': self.exchange_order_id,
            'strategy_id': self.strategy_id,
            'priority': self.priority.value,
            'fills': [fill.__dict__ for fill in self.fills],
            'total_fees': {k: str(v) for k, v in self.total_fees.items()},
            'tags': self.tags,
            'notes': self.notes
        }


class OrderValidator:
    """
    Order validation system with comprehensive checks.
    
    This class provides extensive validation for orders before submission
    including risk checks, balance validation, and market constraints.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("OrderValidator")
        
    def _default_config(self) -> Dict:
        """Default configuration for order validator."""
        return {
            'min_order_size': Decimal('10'),        # Minimum order size in USD
            'max_order_size': Decimal('100000'),    # Maximum order size in USD
            'max_price_deviation': 0.05,            # 5% max price deviation from market
            'min_price_precision': 2,               # Minimum price decimal places
            'min_quantity_precision': 6,            # Minimum quantity decimal places
            'validate_balance': True,               # Validate account balance
            'validate_market_hours': False,         # Crypto markets are 24/7
            'check_position_limits': True,          # Check position size limits
            'symbol_whitelist': None,               # Allowed symbols (None = all)
            'symbol_blacklist': [],                 # Forbidden symbols
        }
    
    def validate_order(
        self,
        order: Order,
        market_price: Optional[Decimal] = None,
        account_balances: Optional[Dict[str, Decimal]] = None,
        current_positions: Optional[Dict[str, Decimal]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive order validation.
        
        Args:
            order: Order to validate
            market_price: Current market price
            account_balances: Available account balances
            current_positions: Current position sizes
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Basic order validation
        basic_errors = self._validate_basic_order(order)
        errors.extend(basic_errors)
        
        # Price validation
        if market_price:
            price_errors = self._validate_price(order, market_price)
            errors.extend(price_errors)
        
        # Size validation
        size_errors = self._validate_size(order, market_price)
        errors.extend(size_errors)
        
        # Symbol validation
        symbol_errors = self._validate_symbol(order)
        errors.extend(symbol_errors)
        
        # Balance validation
        if self.config['validate_balance'] and account_balances:
            balance_errors = self._validate_balance(order, account_balances, market_price)
            errors.extend(balance_errors)
        
        # Position limit validation
        if self.config['check_position_limits'] and current_positions:
            position_errors = self._validate_position_limits(order, current_positions, market_price)
            errors.extend(position_errors)
        
        # Order type specific validation
        type_errors = self._validate_order_type_specific(order)
        errors.extend(type_errors)
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            self.logger.warning(f"Order validation failed for {order.order_id}: {errors}")
        
        return is_valid, errors
    
    def _validate_basic_order(self, order: Order) -> List[str]:
        """Validate basic order properties."""
        errors = []
        
        # Required fields
        if not order.symbol:
            errors.append("Symbol is required")
        
        if not order.side:
            errors.append("Side is required")
        
        if not order.order_type:
            errors.append("Order type is required")
        
        if order.quantity <= 0:
            errors.append("Quantity must be positive")
        
        # Price requirements for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
            if not order.price or order.price <= 0:
                errors.append(f"Price is required for {order.order_type.value} orders")
        
        # Stop price requirements
        if order.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
            if not order.stop_price or order.stop_price <= 0:
                errors.append(f"Stop price is required for {order.order_type.value} orders")
        
        return errors
    
    def _validate_price(self, order: Order, market_price: Decimal) -> List[str]:
        """Validate order price against market conditions."""
        errors = []
        
        if not order.price:
            return errors
        
        # Price deviation check
        max_deviation = self.config['max_price_deviation']
        
        if order.is_buy:
            # Buy orders shouldn't be too far above market
            max_buy_price = market_price * (1 + max_deviation)
            if order.price > max_buy_price:
                errors.append(f"Buy price {order.price} too far above market price {market_price}")
        else:
            # Sell orders shouldn't be too far below market
            min_sell_price = market_price * (1 - max_deviation)
            if order.price < min_sell_price:
                errors.append(f"Sell price {order.price} too far below market price {market_price}")
        
        # Price precision check
        price_str = str(order.price)
        if '.' in price_str:
            decimal_places = len(price_str.split('.')[1])
            if decimal_places > self.config['min_price_precision']:
                errors.append(f"Price precision too high: {decimal_places} decimals")
        
        return errors
    
    def _validate_size(self, order: Order, market_price: Optional[Decimal] = None) -> List[str]:
        """Validate order size constraints."""
        errors = []
        
        # Quantity precision
        quantity_str = str(order.quantity)
        if '.' in quantity_str:
            decimal_places = len(quantity_str.split('.')[1])
            if decimal_places > self.config['min_quantity_precision']:
                errors.append(f"Quantity precision too high: {decimal_places} decimals")
        
        # Notional value validation
        if market_price:
            notional_value = order.quantity * market_price
            
            if notional_value < self.config['min_order_size']:
                errors.append(f"Order size {notional_value} below minimum {self.config['min_order_size']}")
            
            if notional_value > self.config['max_order_size']:
                errors.append(f"Order size {notional_value} above maximum {self.config['max_order_size']}")
        
        return errors
    
    def _validate_symbol(self, order: Order) -> List[str]:
        """Validate trading symbol."""
        errors = []
        
        # Whitelist check
        if self.config['symbol_whitelist']:
            if order.symbol not in self.config['symbol_whitelist']:
                errors.append(f"Symbol {order.symbol} not in whitelist")
        
        # Blacklist check
        if order.symbol in self.config['symbol_blacklist']:
            errors.append(f"Symbol {order.symbol} is blacklisted")
        
        return errors
    
    def _validate_balance(
        self,
        order: Order,
        balances: Dict[str, Decimal],
        market_price: Optional[Decimal] = None
    ) -> List[str]:
        """Validate account balance for order."""
        errors = []
        
        if not market_price:
            return errors
        
        # Extract base and quote currency from symbol (simplified)
        if '/' in order.symbol:
            base_currency, quote_currency = order.symbol.split('/')
        elif 'USDT' in order.symbol:
            base_currency = order.symbol.replace('USDT', '')
            quote_currency = 'USDT'
        else:
            # Can't determine currencies
            return errors
        
        if order.is_buy:
            # Need quote currency for buy orders
            required_balance = order.quantity * market_price
            available_balance = balances.get(quote_currency, Decimal('0'))
            
            if required_balance > available_balance:
                errors.append(
                    f"Insufficient {quote_currency} balance: "
                    f"need {required_balance}, have {available_balance}"
                )
        else:
            # Need base currency for sell orders
            required_balance = order.quantity
            available_balance = balances.get(base_currency, Decimal('0'))
            
            if required_balance > available_balance:
                errors.append(
                    f"Insufficient {base_currency} balance: "
                    f"need {required_balance}, have {available_balance}"
                )
        
        return errors
    
    def _validate_position_limits(
        self,
        order: Order,
        positions: Dict[str, Decimal],
        market_price: Optional[Decimal] = None
    ) -> List[str]:
        """Validate position size limits."""
        errors = []
        
        # This would be implemented based on specific position limit rules
        # For now, we'll just check basic position size constraints
        
        current_position = positions.get(order.symbol, Decimal('0'))
        
        # Calculate new position after order
        if order.is_buy:
            new_position = current_position + order.quantity
        else:
            new_position = current_position - order.quantity
        
        # Example limit check (would be configurable)
        max_position_size = Decimal('1000000')  # $1M position limit
        
        if market_price and abs(new_position * market_price) > max_position_size:
            errors.append(f"Order would exceed position limit of {max_position_size}")
        
        return errors
    
    def _validate_order_type_specific(self, order: Order) -> List[str]:
        """Validate order type specific requirements."""
        errors = []
        
        # Stop order validation
        if order.is_stop_order and order.stop_price and order.price:
            if order.is_buy:
                # Buy stop: stop price should be above current price
                # Stop limit: limit price should be >= stop price
                if order.order_type == OrderType.STOP_LIMIT and order.price < order.stop_price:
                    errors.append("Buy stop-limit: limit price should be >= stop price")
            else:
                # Sell stop: stop price should be below current price
                # Stop limit: limit price should be <= stop price
                if order.order_type == OrderType.STOP_LIMIT and order.price > order.stop_price:
                    errors.append("Sell stop-limit: limit price should be <= stop price")
        
        # Iceberg order validation
        if order.order_type == OrderType.ICEBERG:
            if not order.iceberg_quantity or order.iceberg_quantity <= 0:
                errors.append("Iceberg quantity is required for iceberg orders")
            elif order.iceberg_quantity >= order.quantity:
                errors.append("Iceberg quantity should be less than total quantity")
        
        # Time in force validation
        if order.time_in_force == TimeInForce.IOC and order.order_type != OrderType.MARKET:
            # IOC typically only works with market orders or at competitive prices
            pass
        
        if order.time_in_force == TimeInForce.FOK and order.order_type == OrderType.MARKET:
            errors.append("Fill-or-Kill not supported for market orders")
        
        return errors


class OrderBook:
    """
    Simple order book representation for price calculations.
    
    This class maintains a basic order book for price impact calculations
    and optimal order placement strategies.
    """
    
    def __init__(self):
        self.bids: List[Tuple[Decimal, Decimal]] = []  # (price, quantity)
        self.asks: List[Tuple[Decimal, Decimal]] = []  # (price, quantity)
        self.last_updated = datetime.now()
        
    def update(self, bids: List[Tuple[Decimal, Decimal]], asks: List[Tuple[Decimal, Decimal]]) -> None:
        """Update order book with new data."""
        self.bids = sorted(bids, key=lambda x: x[0], reverse=True)  # Highest bid first
        self.asks = sorted(asks, key=lambda x: x[0])                # Lowest ask first
        self.last_updated = datetime.now()
    
    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price."""
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price."""
        return self.asks[0][0] if self.asks else None
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Get mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Get bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Get spread in basis points."""
        if self.spread and self.mid_price:
            return float(self.spread / self.mid_price * 10000)
        return None
    
    def calculate_slippage(self, side: OrderSide, quantity: Decimal) -> Dict[str, Any]:
        """
        Calculate expected slippage for a market order.
        
        Args:
            side: Order side (buy/sell)
            quantity: Order quantity
            
        Returns:
            Dictionary with slippage analysis
        """
        if side == OrderSide.BUY:
            levels = self.asks
            best_price = self.best_ask
        else:
            levels = self.bids
            best_price = self.best_bid
        
        if not levels or not best_price:
            return {'error': 'No liquidity available'}
        
        remaining_qty = quantity
        total_cost = Decimal('0')
        filled_qty = Decimal('0')
        levels_consumed = 0
        
        for price, available_qty in levels:
            if remaining_qty <= 0:
                break
            
            fill_qty = min(remaining_qty, available_qty)
            total_cost += fill_qty * price
            filled_qty += fill_qty
            remaining_qty -= fill_qty
            levels_consumed += 1
        
        if filled_qty <= 0:
            return {'error': 'Cannot fill order'}
        
        average_price = total_cost / filled_qty
        slippage = (average_price - best_price) / best_price
        
        return {
            'average_price': average_price,
            'slippage_bps': float(slippage * 10000),
            'filled_quantity': filled_qty,
            'unfilled_quantity': remaining_qty,
            'levels_consumed': levels_consumed,
            'total_cost': total_cost
        }
    
    def get_optimal_limit_price(
        self,
        side: OrderSide,
        aggression: float = 0.5
    ) -> Optional[Decimal]:
        """
        Get optimal limit price based on aggression level.
        
        Args:
            side: Order side
            aggression: Aggression level (0=passive, 1=aggressive)
            
        Returns:
            Optimal limit price
        """
        if not self.best_bid or not self.best_ask:
            return None
        
        if side == OrderSide.BUY:
            # 0 = best bid (passive), 1 = best ask (aggressive)
            return self.best_bid + (self.best_ask - self.best_bid) * Decimal(str(aggression))
        else:
            # 0 = best ask (passive), 1 = best bid (aggressive)
            return self.best_ask - (self.best_ask - self.best_bid) * Decimal(str(aggression))


class OrderManager:
    """
    Core order management system with comprehensive tracking and monitoring.
    
    This class provides centralized order management including creation,
    validation, tracking, and lifecycle management.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("OrderManager")
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_strategy: Dict[str, List[str]] = defaultdict(list)
        
        # Order validator
        self.validator = OrderValidator(self.config.get('validator', {}))
        
        # Order books
        self.order_books: Dict[str, OrderBook] = {}
        
        # Event callbacks
        self.order_callbacks: List[Callable[[Order, str], None]] = []
        
        # Statistics
        self.stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_volume': Decimal('0'),
            'total_fees': defaultdict(Decimal)
        }
        
    def _default_config(self) -> Dict:
        """Default configuration for order manager."""
        return {
            'enable_validation': True,
            'auto_cancel_timeout': 3600,     # Auto-cancel orders after 1 hour
            'max_orders_per_symbol': 10,     # Maximum orders per symbol
            'max_total_orders': 100,         # Maximum total orders
            'order_update_interval': 1,      # Seconds between order updates
            'enable_order_book': True,       # Enable order book tracking
            'validator': {}
        }
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Union[Decimal, float, str],
        price: Optional[Union[Decimal, float, str]] = None,
        stop_price: Optional[Union[Decimal, float, str]] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        strategy_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Order type
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            strategy_id: Strategy identifier
            tags: Order tags
            **kwargs: Additional order parameters
            
        Returns:
            Created order
        """
        # Convert to Decimal
        quantity = Decimal(str(quantity))
        price = Decimal(str(price)) if price is not None else None
        stop_price = Decimal(str(stop_price)) if stop_price is not None else None
        
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            client_order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            strategy_id=strategy_id,
            tags=tags or {},
            **kwargs
        )
        
        # Validate order if enabled
        if self.config['enable_validation']:
            is_valid, errors = self._validate_order(order)
            if not is_valid:
                raise ValueError(f"Order validation failed: {errors}")
        
        # Check limits
        self._check_order_limits(order)
        
        # Store order
        self.orders[order.order_id] = order
        self.orders_by_symbol[symbol].append(order.order_id)
        if strategy_id:
            self.orders_by_strategy[strategy_id].append(order.order_id)
        
        # Update statistics
        self.stats['total_orders'] += 1
        
        # Notify callbacks
        self._notify_callbacks(order, 'created')
        
        self.logger.info(f"Created order {order.order_id}: {order.side.value} {order.quantity} {order.symbol}")
        
        return order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        order_ids = self.orders_by_symbol.get(symbol, [])
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_orders_by_strategy(self, strategy_id: str) -> List[Order]:
        """Get all orders for a strategy."""
        order_ids = self.orders_by_strategy.get(strategy_id, [])
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol."""
        orders = []
        for order in self.orders.values():
            if order.is_open and (symbol is None or order.symbol == symbol):
                orders.append(order)
        return orders
    
    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        exchange_order_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Update order status."""
        order = self.orders.get(order_id)
        if not order:
            self.logger.warning(f"Order {order_id} not found for status update")
            return False
        
        old_status = order.status
        order.update_status(status, timestamp)
        
        if exchange_order_id:
            order.exchange_order_id = exchange_order_id
        
        # Update statistics
        if status == OrderStatus.FILLED and old_status != OrderStatus.FILLED:
            self.stats['filled_orders'] += 1
            self.stats['total_volume'] += order.quantity
        elif status == OrderStatus.CANCELLED and old_status != OrderStatus.CANCELLED:
            self.stats['cancelled_orders'] += 1
        elif status == OrderStatus.REJECTED and old_status != OrderStatus.REJECTED:
            self.stats['rejected_orders'] += 1
        
        # Notify callbacks
        self._notify_callbacks(order, 'status_updated')
        
        self.logger.info(f"Updated order {order_id} status: {old_status.value} -> {status.value}")
        
        return True
    
    def add_order_fill(
        self,
        order_id: str,
        fill: OrderFill
    ) -> bool:
        """Add a fill to an order."""
        order = self.orders.get(order_id)
        if not order:
            self.logger.warning(f"Order {order_id} not found for fill")
            return False
        
        order.add_fill(fill)
        
        # Update fee statistics
        self.stats['total_fees'][fill.fee_currency] += fill.fee
        
        # Notify callbacks
        self._notify_callbacks(order, 'filled')
        
        self.logger.info(
            f"Added fill to order {order_id}: "
            f"{fill.quantity} @ {fill.price} "
            f"(total filled: {order.filled_quantity}/{order.quantity})"
        )
        
        return True
    
    def cancel_order(self, order_id: str, reason: Optional[str] = None) -> bool:
        """Cancel an order."""
        order = self.orders.get(order_id)
        if not order:
            self.logger.warning(f"Order {order_id} not found for cancellation")
            return False
        
        if not order.is_open:
            self.logger.warning(f"Order {order_id} is not open, cannot cancel")
            return False
        
        order.update_status(OrderStatus.CANCELLED)
        if reason:
            order.notes = f"{order.notes or ''} Cancelled: {reason}".strip()
        
        # Notify callbacks
        self._notify_callbacks(order, 'cancelled')
        
        self.logger.info(f"Cancelled order {order_id}" + (f": {reason}" if reason else ""))
        
        return True
    
    def cancel_all_orders(self, symbol: Optional[str] = None, strategy_id: Optional[str] = None) -> int:
        """Cancel multiple orders."""
        cancelled_count = 0
        
        for order in self.orders.values():
            if not order.is_open:
                continue
            
            if symbol and order.symbol != symbol:
                continue
            
            if strategy_id and order.strategy_id != strategy_id:
                continue
            
            if self.cancel_order(order.order_id, "Bulk cancellation"):
                cancelled_count += 1
        
        self.logger.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count
    
    def update_order_book(self, symbol: str, bids: List[Tuple[Decimal, Decimal]], 
                         asks: List[Tuple[Decimal, Decimal]]) -> None:
        """Update order book for a symbol."""
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook()
        
        self.order_books[symbol].update(bids, asks)
    
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get order book for a symbol."""
        return self.order_books.get(symbol)
    
    def calculate_order_slippage(self, order: Order) -> Optional[Dict[str, Any]]:
        """Calculate expected slippage for an order."""
        order_book = self.order_books.get(order.symbol)
        if not order_book:
            return None
        
        return order_book.calculate_slippage(order.side, order.quantity)
    
    def get_optimal_price(self, symbol: str, side: OrderSide, aggression: float = 0.5) -> Optional[Decimal]:
        """Get optimal limit price for an order."""
        order_book = self.order_books.get(symbol)
        if not order_book:
            return None
        
        return order_book.get_optimal_limit_price(side, aggression)
    
    def add_order_callback(self, callback: Callable[[Order, str], None]) -> None:
        """Add order event callback."""
        self.order_callbacks.append(callback)
    
    def remove_order_callback(self, callback: Callable[[Order, str], None]) -> None:
        """Remove order event callback."""
        if callback in self.order_callbacks:
            self.order_callbacks.remove(callback)
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order management statistics."""
        open_orders = len(self.get_open_orders())
        
        return {
            'total_orders': self.stats['total_orders'],
            'open_orders': open_orders,
            'filled_orders': self.stats['filled_orders'],
            'cancelled_orders': self.stats['cancelled_orders'],
            'rejected_orders': self.stats['rejected_orders'],
            'fill_rate': (
                self.stats['filled_orders'] / self.stats['total_orders'] 
                if self.stats['total_orders'] > 0 else 0
            ),
            'total_volume': str(self.stats['total_volume']),
            'total_fees': {k: str(v) for k, v in self.stats['total_fees'].items()},
            'orders_by_symbol': {
                symbol: len(order_ids) 
                for symbol, order_ids in self.orders_by_symbol.items()
            },
            'orders_by_strategy': {
                strategy: len(order_ids) 
                for strategy, order_ids in self.orders_by_strategy.items()
            }
        }
    
    def _validate_order(self, order: Order) -> Tuple[bool, List[str]]:
        """Validate order using validator."""
        market_price = None
        order_book = self.order_books.get(order.symbol)
        if order_book and order_book.mid_price:
            market_price = order_book.mid_price
        
        return self.validator.validate_order(order, market_price)
    
    def _check_order_limits(self, order: Order) -> None:
        """Check order limits and constraints."""
        # Check symbol limit
        symbol_orders = len(self.orders_by_symbol.get(order.symbol, []))
        if symbol_orders >= self.config['max_orders_per_symbol']:
            raise ValueError(f"Maximum orders per symbol ({self.config['max_orders_per_symbol']}) exceeded")
        
        # Check total limit
        if len(self.orders) >= self.config['max_total_orders']:
            raise ValueError(f"Maximum total orders ({self.config['max_total_orders']}) exceeded")
    
    def _notify_callbacks(self, order: Order, event_type: str) -> None:
        """Notify order event callbacks."""
        for callback in self.order_callbacks:
            try:
                callback(order, event_type)
            except Exception as e:
                self.logger.error(f"Error in order callback: {e}")
    
    def cleanup_old_orders(self, max_age_hours: int = 24) -> int:
        """Clean up old closed orders."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        orders_to_remove = []
        
        for order_id, order in self.orders.items():
            if order.is_closed and order.updated_at and order.updated_at < cutoff_time:
                orders_to_remove.append(order_id)
        
        for order_id in orders_to_remove:
            order = self.orders.pop(order_id)
            
            # Remove from indexes
            if order.symbol in self.orders_by_symbol:
                self.orders_by_symbol[order.symbol] = [
                    oid for oid in self.orders_by_symbol[order.symbol] if oid != order_id
                ]
            
            if order.strategy_id and order.strategy_id in self.orders_by_strategy:
                self.orders_by_strategy[order.strategy_id] = [
                    oid for oid in self.orders_by_strategy[order.strategy_id] if oid != order_id
                ]
        
        if orders_to_remove:
            self.logger.info(f"Cleaned up {len(orders_to_remove)} old orders")
        
        return len(orders_to_remove)