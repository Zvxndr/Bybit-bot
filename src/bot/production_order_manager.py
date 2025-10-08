"""
Order Manager - Production Trading Implementation
==============================================

This module implements order management with real Bybit API integration.
Bridges the sophisticated execution framework with actual exchange orders.

Status: HIGH PRIORITY - Required for production trading
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Import Bybit client
try:
    from ..bybit_api import BybitAPIClient
except ImportError:
    # Fallback import
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from bybit_api import BybitAPIClient


class OrderType(Enum):
    """Order types supported by production system"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class OrderFill:
    """Order fill information"""
    fill_id: str
    timestamp: datetime
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_currency: str


@dataclass
class Order:
    """Production order with exchange tracking"""
    order_id: str
    strategy_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    
    # Exchange tracking
    exchange_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    updated_at: datetime = None
    
    # Execution tracking
    filled_quantity: Decimal = Decimal('0')
    average_fill_price: Optional[Decimal] = None
    fills: List[OrderFill] = None
    
    # Fees and costs
    total_fees: Decimal = Decimal('0')
    estimated_cost: Optional[Decimal] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.fills is None:
            self.fills = []
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is active (open or partially filled)"""
        return self.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining unfilled quantity"""
        return self.quantity - self.filled_quantity
    
    @property
    def fill_percentage(self) -> float:
        """Get fill percentage"""
        if self.quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.quantity * 100)


class ProductionOrderManager:
    """
    Production-ready order manager with Bybit integration.
    
    This class handles the CRITICAL gap identified in the project analysis:
    - Real order placement and tracking
    - Exchange integration
    - Fill monitoring
    - Risk controls
    """
    
    def __init__(self, bybit_client: BybitAPIClient, testnet_client: BybitAPIClient):
        self.logger = logging.getLogger(__name__)
        
        # API clients
        self.bybit_client = bybit_client      # Live trading
        self.testnet_client = testnet_client  # Paper trading
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.exchange_order_map: Dict[str, str] = {}  # exchange_id -> our_id
        
        # Order monitoring
        self.monitoring_orders: Dict[str, Order] = {}
        self.monitor_task = None
        
        # Statistics
        self.stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'total_volume': Decimal('0'),
            'total_fees': Decimal('0')
        }
        
        self.logger.info("✅ Production Order Manager initialized")
    
    async def place_order(self,
                         symbol: str,
                         side: OrderSide,
                         order_type: OrderType,
                         quantity: Decimal,
                         price: Optional[Decimal] = None,
                         stop_price: Optional[Decimal] = None,
                         strategy_id: Optional[str] = None,
                         use_testnet: bool = True) -> Optional[Order]:
        """
        Place order on exchange - PRODUCTION IMPLEMENTATION
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            side: Buy or sell
            order_type: Market, limit, etc.
            quantity: Order quantity
            price: Limit price (if applicable)
            stop_price: Stop price (if applicable)
            strategy_id: Strategy that placed the order
            use_testnet: Use testnet (paper trading) or mainnet (live)
            
        Returns:
            Order object if successful, None if failed
        """
        try:
            # Generate unique order ID
            order_id = f"ord_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Create order object
            order = Order(
                order_id=order_id,
                strategy_id=strategy_id or "manual",
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price
            )
            
            # Select appropriate client
            client = self.testnet_client if use_testnet else self.bybit_client
            
            # Prepare order parameters for Bybit API
            order_params = self._prepare_bybit_order(order)
            
            # Place order on exchange
            response = await client.place_order(**order_params)
            
            if response and response.get('success'):
                # Update order with exchange response
                order.exchange_order_id = response.get('order_id')
                order.status = OrderStatus.SUBMITTED
                order.updated_at = datetime.now()
                
                # Store order
                self.orders[order_id] = order
                if order.exchange_order_id:
                    self.exchange_order_map[order.exchange_order_id] = order_id
                
                # Start monitoring
                self.monitoring_orders[order_id] = order
                self._ensure_monitor_running()
                
                # Update statistics
                self.stats['total_orders'] += 1
                
                self.logger.info(f"✅ Order placed: {order_id} -> {order.exchange_order_id}")
                return order
            
            else:
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                order.status = OrderStatus.REJECTED
                self.logger.error(f"❌ Order rejected: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Order placement error: {e}")
            return None
    
    def _prepare_bybit_order(self, order: Order) -> Dict[str, Any]:
        """Prepare order parameters for Bybit API"""
        params = {
            'symbol': order.symbol,
            'side': order.side.value.capitalize(),  # Bybit expects 'Buy'/'Sell'
            'orderType': self._map_order_type(order.order_type),
            'qty': str(order.quantity)
        }
        
        # Add price for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price:
                params['price'] = str(order.price)
        
        # Add stop price for stop orders
        if order.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
            if order.stop_price:
                params['stopPrice'] = str(order.stop_price)
        
        # Add time in force (default to GTC)
        params['timeInForce'] = 'GTC'
        
        return params
    
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map our order types to Bybit order types"""
        mapping = {
            OrderType.MARKET: 'Market',
            OrderType.LIMIT: 'Limit',
            OrderType.STOP_MARKET: 'StopMarket',
            OrderType.STOP_LIMIT: 'StopLimit'
        }
        return mapping.get(order_type, 'Limit')
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            order = self.orders.get(order_id)
            if not order:
                self.logger.warning(f"⚠️ Order not found: {order_id}")
                return False
            
            if not order.is_active:
                self.logger.warning(f"⚠️ Order not active: {order_id}")
                return False
            
            # Determine which client to use
            client = self.testnet_client if order.strategy_id != "live" else self.bybit_client
            
            # Cancel on exchange
            response = await client.cancel_order(
                symbol=order.symbol,
                order_id=order.exchange_order_id
            )
            
            if response and response.get('success'):
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                
                # Remove from monitoring
                self.monitoring_orders.pop(order_id, None)
                
                # Update statistics
                self.stats['cancelled_orders'] += 1
                
                self.logger.info(f"✅ Order cancelled: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Order cancellation error: {e}")
            return False
    
    def _ensure_monitor_running(self):
        """Ensure order monitoring task is running"""
        if not self.monitor_task or self.monitor_task.done():
            self.monitor_task = asyncio.create_task(self._monitor_orders())
    
    async def _monitor_orders(self):
        """Monitor active orders for fills and status changes"""
        self.logger.info("🔄 Starting order monitoring")
        
        while self.monitoring_orders:
            try:
                # Check each monitored order
                for order_id in list(self.monitoring_orders.keys()):
                    order = self.monitoring_orders.get(order_id)
                    if order:
                        await self._check_order_status(order)
                
                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Order monitoring error: {e}")
                await asyncio.sleep(30)  # Longer delay on errors
        
        self.logger.info("🔄 Order monitoring stopped")
    
    async def _check_order_status(self, order: Order):
        """Check order status on exchange"""
        try:
            if not order.exchange_order_id:
                return
            
            # Determine which client to use
            client = self.testnet_client if order.strategy_id != "live" else self.bybit_client
            
            # Get order status from exchange
            response = await client.get_order_status(order.exchange_order_id)
            
            if response and response.get('success'):
                exchange_data = response.get('data', {})
                
                # Update order status
                old_status = order.status
                order.status = self._map_exchange_status(exchange_data.get('orderStatus', ''))
                order.updated_at = datetime.now()
                
                # Update fill information
                if 'cumExecQty' in exchange_data:
                    new_filled = Decimal(str(exchange_data['cumExecQty']))
                    
                    if new_filled > order.filled_quantity:
                        # New fill detected
                        fill_price = Decimal(str(exchange_data.get('avgPrice', 0)))
                        order.filled_quantity = new_filled
                        order.average_fill_price = fill_price
                        
                        # Calculate fees if provided
                        if 'cumExecFee' in exchange_data:
                            order.total_fees = Decimal(str(exchange_data['cumExecFee']))
                        
                        self.logger.info(f"📊 Fill update: {order.order_id} - {order.filled_quantity}/{order.quantity}")
                
                # Remove from monitoring if completed
                if not order.is_active:
                    self.monitoring_orders.pop(order.order_id, None)
                    
                    if order.is_filled:
                        self.stats['filled_orders'] += 1
                        self.stats['total_volume'] += order.filled_quantity
                        self.stats['total_fees'] += order.total_fees
                
                # Log status changes
                if old_status != order.status:
                    self.logger.info(f"📊 Status change: {order.order_id} {old_status.value} -> {order.status.value}")
                    
        except Exception as e:
            self.logger.error(f"❌ Order status check error for {order.order_id}: {e}")
    
    def _map_exchange_status(self, exchange_status: str) -> OrderStatus:
        """Map exchange order status to our status"""
        mapping = {
            'New': OrderStatus.OPEN,
            'PartiallyFilled': OrderStatus.PARTIALLY_FILLED,
            'Filled': OrderStatus.FILLED,
            'Cancelled': OrderStatus.CANCELLED,
            'Rejected': OrderStatus.REJECTED,
            'Pending': OrderStatus.PENDING
        }
        return mapping.get(exchange_status, OrderStatus.PENDING)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_orders_by_strategy(self, strategy_id: str) -> List[Order]:
        """Get all orders for a strategy"""
        return [order for order in self.orders.values() if order.strategy_id == strategy_id]
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return [order for order in self.orders.values() if order.is_active]
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order management statistics"""
        return {
            **self.stats,
            'active_orders': len(self.monitoring_orders),
            'total_stored_orders': len(self.orders),
            'success_rate': (self.stats['filled_orders'] / self.stats['total_orders'] * 100) if self.stats['total_orders'] > 0 else 0
        }
    
    async def cleanup(self):
        """Cleanup order manager"""
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("🔄 Order manager cleanup completed")


# Factory function for easy integration
def create_production_order_manager(bybit_client: BybitAPIClient, 
                                  testnet_client: BybitAPIClient) -> ProductionOrderManager:
    """Factory function to create production order manager"""
    return ProductionOrderManager(
        bybit_client=bybit_client,
        testnet_client=testnet_client
    )