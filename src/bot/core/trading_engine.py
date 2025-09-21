"""
Core Trading Engine for Bybit Trading Bot

This module implements the main trading engine responsible for:
- Order execution and management
- Position tracking and updates
- Integration with Bybit API
- Risk validation before trades
- Real-time portfolio updates

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from decimal import Decimal, ROUND_DOWN, ROUND_UP

from ..exchange.bybit_client import BybitClient
from ..config_manager import ConfigurationManager
from ..utils.logging import TradingLogger
from ..data.data_manager import DataManager


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class Order:
    """Order data structure."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    average_price: Optional[Decimal] = None
    timestamp: Optional[datetime] = None
    commission: Decimal = Decimal('0')
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    side: str  # 'long' or 'short'
    size: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of position."""
        return self.size * self.mark_price
    
    @property
    def pnl_percentage(self) -> Decimal:
        """Calculate PnL as percentage."""
        if self.entry_price == 0:
            return Decimal('0')
        return (self.unrealized_pnl / (self.size * self.entry_price)) * 100


class TradingEngine:
    """
    Main trading engine for executing trades and managing positions.
    
    Features:
    - Order execution with Bybit API
    - Real-time position tracking
    - Risk validation before trades
    - Portfolio balance management
    - Trade history and reporting
    """
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        bybit_client: BybitClient,
        data_manager: DataManager,
        testnet: bool = True
    ):
        self.config = config_manager
        self.client = bybit_client
        self.data_manager = data_manager
        self.testnet = testnet
        self.logger = TradingLogger("trading_engine")
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.balance: Dict[str, Decimal] = {}
        self.is_running = False
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = Decimal('0')
        self.max_drawdown = Decimal('0')
        self.peak_balance = Decimal('0')
        
        # Risk limits from config
        self.max_position_size = Decimal(str(config_manager.get('trading.max_position_size', 0.1)))
        self.max_open_orders = config_manager.get('trading.max_open_orders', 10)
        
        self.logger.info(f"TradingEngine initialized - Testnet: {testnet}")
    
    async def start(self) -> bool:
        """
        Start the trading engine.
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.logger.info("Starting trading engine...")
            
            # Initialize connection
            if not await self.client.connect():
                self.logger.error("Failed to connect to Bybit API")
                return False
            
            # Load initial balance
            await self.update_balance()
            
            # Load existing positions
            await self.update_positions()
            
            self.is_running = True
            self.logger.info("Trading engine started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start trading engine: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the trading engine."""
        try:
            self.logger.info("Stopping trading engine...")
            
            # Cancel all pending orders
            await self.cancel_all_orders()
            
            # Close WebSocket connections
            await self.client.disconnect()
            
            self.is_running = False
            self.logger.info("Trading engine stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading engine: {e}")
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "GTC"
    ) -> Optional[Order]:
        """
        Place a new order.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT/etc.)
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            
        Returns:
            Order object if successful, None otherwise
        """
        try:
            # Validate order before placing
            if not await self._validate_order(symbol, side, order_type, quantity, price):
                return None
            
            # Create order object
            order = Order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force
            )
            
            # Place order via API
            api_response = await self.client.place_order(
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                qty=str(quantity),
                price=str(price) if price else None,
                stop_px=str(stop_price) if stop_price else None,
                time_in_force=time_in_force
            )
            
            if api_response and 'order_id' in api_response:
                order.order_id = api_response['order_id']
                order.status = OrderStatus.PENDING
                
                # Store order
                self.orders[order.order_id] = order
                
                self.logger.info(
                    f"Order placed: {symbol} {side.value} {quantity} @ {price or 'MARKET'}"
                )
                
                return order
            else:
                self.logger.error(f"Failed to place order: {api_response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if cancelled successfully
        """
        try:
            if order_id not in self.orders:
                self.logger.warning(f"Order {order_id} not found")
                return False
            
            order = self.orders[order_id]
            
            # Cancel via API
            success = await self.client.cancel_order(order.symbol, order_id)
            
            if success:
                order.status = OrderStatus.CANCELLED
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order: {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all pending orders.
        
        Args:
            symbol: Optional symbol to filter orders
            
        Returns:
            int: Number of orders cancelled
        """
        cancelled_count = 0
        
        try:
            orders_to_cancel = [
                order for order in self.orders.values()
                if order.status == OrderStatus.PENDING and 
                (symbol is None or order.symbol == symbol)
            ]
            
            for order in orders_to_cancel:
                if await self.cancel_order(order.order_id):
                    cancelled_count += 1
            
            self.logger.info(f"Cancelled {cancelled_count} orders")
            return cancelled_count
            
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {e}")
            return cancelled_count
    
    async def update_positions(self) -> None:
        """Update positions from API."""
        try:
            api_positions = await self.client.get_positions()
            
            if api_positions:
                for pos_data in api_positions:
                    symbol = pos_data.get('symbol')
                    if not symbol:
                        continue
                    
                    size = Decimal(str(pos_data.get('size', 0)))
                    if size == 0:
                        # Remove closed positions
                        if symbol in self.positions:
                            del self.positions[symbol]
                        continue
                    
                    position = Position(
                        symbol=symbol,
                        side=pos_data.get('side', 'long'),
                        size=size,
                        entry_price=Decimal(str(pos_data.get('entry_price', 0))),
                        mark_price=Decimal(str(pos_data.get('mark_price', 0))),
                        unrealized_pnl=Decimal(str(pos_data.get('unrealized_pnl', 0))),
                        realized_pnl=Decimal(str(pos_data.get('realized_pnl', 0)))
                    )
                    
                    self.positions[symbol] = position
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def update_balance(self) -> None:
        """Update account balance from API."""
        try:
            balance_data = await self.client.get_wallet_balance()
            
            if balance_data:
                for asset_data in balance_data:
                    coin = asset_data.get('coin')
                    available = asset_data.get('available_balance', 0)
                    
                    if coin:
                        self.balance[coin] = Decimal(str(available))
                
                # Update peak balance for drawdown calculation
                usdt_balance = self.balance.get('USDT', Decimal('0'))
                if usdt_balance > self.peak_balance:
                    self.peak_balance = usdt_balance
            
        except Exception as e:
            self.logger.error(f"Error updating balance: {e}")
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position object or None
        """
        return self.positions.get(symbol)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get open orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of open orders
        """
        return [
            order for order in self.orders.values()
            if order.status == OrderStatus.PENDING and
            (symbol is None or order.symbol == symbol)
        ]
    
    def get_portfolio_value(self) -> Decimal:
        """
        Calculate total portfolio value.
        
        Returns:
            Total portfolio value in USDT
        """
        total_value = self.balance.get('USDT', Decimal('0'))
        
        # Add unrealized PnL from positions
        for position in self.positions.values():
            total_value += position.unrealized_pnl
        
        return total_value
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get trading performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        portfolio_value = self.get_portfolio_value()
        
        # Calculate current drawdown
        current_drawdown = Decimal('0')
        if self.peak_balance > 0:
            current_drawdown = ((self.peak_balance - portfolio_value) / self.peak_balance) * 100
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
        
        win_rate = Decimal('0')
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': float(win_rate),
            'total_pnl': float(self.total_pnl),
            'current_drawdown': float(current_drawdown),
            'max_drawdown': float(self.max_drawdown),
            'portfolio_value': float(portfolio_value),
            'peak_balance': float(self.peak_balance),
            'open_positions': len(self.positions),
            'open_orders': len(self.get_open_orders())
        }
    
    async def _validate_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal]
    ) -> bool:
        """
        Validate order before placing.
        
        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Order price
            
        Returns:
            bool: True if order is valid
        """
        try:
            # Check if engine is running
            if not self.is_running:
                self.logger.error("Trading engine is not running")
                return False
            
            # Check quantity is positive
            if quantity <= 0:
                self.logger.error(f"Invalid quantity: {quantity}")
                return False
            
            # Check maximum open orders
            if len(self.get_open_orders()) >= self.max_open_orders:
                self.logger.error(f"Maximum open orders reached: {self.max_open_orders}")
                return False
            
            # Check balance for buy orders
            if side == OrderSide.BUY:
                required_balance = quantity * (price or await self._get_market_price(symbol))
                usdt_balance = self.balance.get('USDT', Decimal('0'))
                
                if required_balance > usdt_balance:
                    self.logger.error(f"Insufficient balance: {required_balance} > {usdt_balance}")
                    return False
            
            # Check position size limits
            current_position = self.positions.get(symbol)
            if current_position:
                new_size = current_position.size
                if side == OrderSide.BUY:
                    new_size += quantity
                else:
                    new_size -= quantity
                
                portfolio_value = self.get_portfolio_value()
                if portfolio_value > 0:
                    position_ratio = abs(new_size) * (price or await self._get_market_price(symbol)) / portfolio_value
                    if position_ratio > self.max_position_size:
                        self.logger.error(f"Position size limit exceeded: {position_ratio} > {self.max_position_size}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            return False
    
    async def _get_market_price(self, symbol: str) -> Decimal:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current market price
        """
        try:
            ticker = await self.client.get_ticker(symbol)
            if ticker and 'last_price' in ticker:
                return Decimal(str(ticker['last_price']))
            else:
                # Fallback to data manager
                latest_data = await self.data_manager.get_latest_price(symbol)
                if latest_data:
                    return Decimal(str(latest_data['close']))
                
            self.logger.warning(f"Could not get market price for {symbol}")
            return Decimal('0')
            
        except Exception as e:
            self.logger.error(f"Error getting market price for {symbol}: {e}")
            return Decimal('0')