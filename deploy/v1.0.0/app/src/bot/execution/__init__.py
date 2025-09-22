"""
Execution Engine Package - Sophisticated Trade Execution System.

This package provides a comprehensive trade execution system with:

Core Components:
- Order Management: Complete order lifecycle management with validation
- Smart Routing: Intelligent execution strategies (TWAP, VWAP, Iceberg, etc.)
- Position Management: Real-time position tracking and risk monitoring
- Execution Analytics: Performance measurement and optimization insights

Key Features:
- Multiple order types and execution strategies
- Real-time market impact analysis
- Slippage optimization and cost analysis
- Position-based risk controls
- Comprehensive execution reporting
- Cross-exchange position aggregation
- Automated execution plan management

The execution engine integrates with the risk management system to ensure
all trades comply with position limits, drawdown constraints, and other
risk parameters while optimizing for execution quality.
"""

from .order_management import (
    # Core order classes
    Order,
    OrderFill,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    OrderPriority,
    
    # Order validation
    OrderValidator,
    
    # Order book representation
    OrderBook,
    
    # Main order manager
    OrderManager
)

from .smart_routing import (
    # Execution strategies
    ExecutionStrategy,
    ExecutionPlan,
    MarketCondition,
    
    # Market analysis
    MarketMetrics,
    MarketAnalyzer,
    
    # Smart routing system
    SmartRouter
)

from .position_management import (
    # Position representation
    Position,
    PositionSnapshot,
    PositionSide,
    PositionStatus,
    
    # Risk monitoring
    PositionRiskMonitor,
    
    # Main position manager
    PositionManager
)

from .execution_analytics import (
    # Analytics classes
    ExecutionMetrics,
    StrategyMetrics,
    MarketImpactModel,
    BenchmarkType,
    
    # Main analyzer
    ExecutionAnalyzer
)

from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime, timedelta

# Package version
__version__ = "1.0.0"

# Default configuration for execution engine
DEFAULT_EXECUTION_CONFIG = {
    'order_manager': {
        'enable_validation': True,
        'auto_cancel_timeout': 3600,
        'max_orders_per_symbol': 10,
        'max_total_orders': 100,
        'order_update_interval': 1,
        'enable_order_book': True,
        'validator': {
            'min_order_size': Decimal('10'),
            'max_order_size': Decimal('100000'),
            'max_price_deviation': 0.05,
            'validate_balance': True,
            'check_position_limits': True
        }
    },
    'smart_routing': {
        'large_order_threshold': Decimal('50000'),
        'max_market_impact_bps': 20,
        'default_participation_rate': 0.1,
        'min_execution_time': timedelta(minutes=5),
        'max_execution_time': timedelta(hours=4),
        'slice_randomization': 0.2,
        'execution_interval': 10,
        'market_analyzer': {
            'volatility_window': 60,
            'volume_window': 300,
            'min_liquidity_score': 0.3,
            'high_volatility_threshold': 0.02,
            'wide_spread_threshold': 50
        }
    },
    'position_manager': {
        'auto_update_prices': True,
        'snapshot_interval': 300,
        'monitoring_interval': 30,
        'enable_persistence': True,
        'risk_monitor': {
            'risk_limits': {
                'max_position_size_usd': 100000,
                'max_drawdown_pct': 10,
                'max_leverage': 5,
                'var_limit_pct': 5,
                'concentration_limit_pct': 20
            },
            'check_interval': 30,
            'enable_auto_actions': False
        }
    },
    'execution_analytics': {
        'enable_persistence': True,
        'benchmark_window_minutes': 60,
        'max_metrics_history': 50000,
        'outlier_threshold': 3.0,
        'min_sample_size': 50
    }
}


class ExecutionEngine:
    """
    Unified execution engine combining all execution components.
    
    This class provides a high-level interface to the execution system,
    coordinating between order management, smart routing, position tracking,
    and performance analytics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize execution engine with configuration."""
        self.config = {**DEFAULT_EXECUTION_CONFIG, **(config or {})}
        
        # Initialize core components
        self.order_manager = OrderManager(self.config['order_manager'])
        self.smart_router = SmartRouter(self.order_manager, self.config['smart_routing'])
        self.position_manager = PositionManager(self.config['position_manager'])
        self.execution_analyzer = ExecutionAnalyzer(self.config['execution_analytics'])
        
        # Set up inter-component communication
        self._setup_callbacks()
        
        # Start background services
        self.smart_router.start_execution_engine()
        self.position_manager.start_monitoring()
    
    def _setup_callbacks(self) -> None:
        """Set up callbacks between components."""
        # Order callbacks
        def on_order_filled(order: Order, event_type: str) -> None:
            if event_type == 'filled' and order.fills:
                # Update position from fills
                for fill in order.fills:
                    self.position_manager.update_position_from_fill(fill, order)
                
                # Analyze execution if order is complete
                if order.is_filled:
                    self.execution_analyzer.analyze_order_execution(order)
        
        self.order_manager.add_order_callback(on_order_filled)
    
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.LIMIT,
        price: Optional[Decimal] = None,
        execution_strategy: Optional[ExecutionStrategy] = None,
        urgency: float = 0.5,
        **kwargs
    ) -> Any:
        """
        Submit an order with intelligent routing.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            order_type: Order type
            price: Limit price (if applicable)
            execution_strategy: Execution strategy
            urgency: Urgency level (0=patient, 1=urgent)
            **kwargs: Additional parameters
            
        Returns:
            Order or ExecutionPlan depending on complexity
        """
        return self.smart_router.route_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            strategy=execution_strategy,
            urgency=urgency,
            price_limit=price,
            **kwargs
        )
    
    def update_market_data(self, symbol: str, price: Decimal, volume: Optional[Decimal] = None, **kwargs) -> None:
        """Update market data across all components."""
        # Update order manager
        if symbol not in self.order_manager.order_books:
            self.order_manager.order_books[symbol] = OrderBook()
        
        # Update position manager
        self.position_manager.update_market_price(symbol, price)
        
        # Update execution analyzer
        if volume is not None:
            self.execution_analyzer.update_market_data(symbol, price, volume)
    
    def get_positions(self, **kwargs) -> List[Position]:
        """Get current positions."""
        return self.position_manager.get_positions(**kwargs)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders."""
        return self.order_manager.get_open_orders(symbol)
    
    def get_execution_performance(self, symbol: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Get execution performance metrics."""
        if symbol:
            return self.execution_analyzer.get_symbol_performance(symbol, days)
        else:
            # Return overall performance
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
            return self.execution_analyzer.generate_execution_report(start_date, end_date)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        return {
            'positions': self.position_manager.get_position_summary(),
            'orders': self.order_manager.get_order_statistics(),
            'execution_plans': {
                plan_id: self.smart_router.get_execution_status(plan_id)
                for plan_id in self.smart_router.execution_plans.keys()
            }
        }
    
    def cancel_order(self, order_id: str, reason: Optional[str] = None) -> bool:
        """Cancel a specific order."""
        return self.order_manager.cancel_order(order_id, reason)
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all orders for a symbol or all symbols."""
        return self.order_manager.cancel_all_orders(symbol)
    
    def close_position(self, symbol: str, reason: Optional[str] = None) -> bool:
        """Close a position."""
        return self.position_manager.close_position(symbol, reason=reason)
    
    def check_risk_limits(self) -> List[Dict[str, Any]]:
        """Check all risk limits."""
        return self.position_manager.check_risk_limits()
    
    def shutdown(self) -> None:
        """Shutdown execution engine."""
        self.smart_router.stop_execution_engine()
        self.position_manager.stop_monitoring()


# Factory functions for easy component creation
def create_order_manager(config: Optional[Dict] = None) -> OrderManager:
    """Create order manager with default configuration."""
    config = config or DEFAULT_EXECUTION_CONFIG['order_manager']
    return OrderManager(config)


def create_smart_router(order_manager: OrderManager, config: Optional[Dict] = None) -> SmartRouter:
    """Create smart router with default configuration."""
    config = config or DEFAULT_EXECUTION_CONFIG['smart_routing']
    return SmartRouter(order_manager, config)


def create_position_manager(config: Optional[Dict] = None) -> PositionManager:
    """Create position manager with default configuration."""
    config = config or DEFAULT_EXECUTION_CONFIG['position_manager']
    return PositionManager(config)


def create_execution_analyzer(config: Optional[Dict] = None) -> ExecutionAnalyzer:
    """Create execution analyzer with default configuration."""
    config = config or DEFAULT_EXECUTION_CONFIG['execution_analytics']
    return ExecutionAnalyzer(config)


def create_execution_engine(config: Optional[Dict] = None) -> ExecutionEngine:
    """Create complete execution engine with default configuration."""
    return ExecutionEngine(config)


# Utility functions for common operations
def calculate_position_size(
    account_balance: Decimal,
    risk_per_trade: float,
    entry_price: Decimal,
    stop_loss_price: Decimal
) -> Decimal:
    """Calculate position size based on risk management."""
    if entry_price <= 0 or stop_loss_price <= 0:
        return Decimal('0')
    
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit <= 0:
        return Decimal('0')
    
    max_risk_amount = account_balance * Decimal(str(risk_per_trade))
    position_size = max_risk_amount / risk_per_unit
    
    return position_size


def calculate_optimal_order_size(
    total_quantity: Decimal,
    market_impact_threshold: float = 0.002,  # 20 bps
    max_participation_rate: float = 0.1,      # 10%
    estimated_daily_volume: Optional[Decimal] = None
) -> Decimal:
    """Calculate optimal order size to minimize market impact."""
    if estimated_daily_volume and estimated_daily_volume > 0:
        # Limit to participation rate
        max_size_by_volume = estimated_daily_volume * Decimal(str(max_participation_rate))
        optimal_size = min(total_quantity, max_size_by_volume)
    else:
        # Simple heuristic: break large orders into smaller chunks
        if total_quantity > Decimal('10000'):  # Arbitrary threshold
            optimal_size = total_quantity / 10
        else:
            optimal_size = total_quantity
    
    return max(optimal_size, total_quantity / 50)  # Don't make orders too small


def estimate_execution_time(
    quantity: Decimal,
    participation_rate: float = 0.1,
    estimated_volume_per_minute: Optional[Decimal] = None
) -> timedelta:
    """Estimate execution time for a given quantity."""
    if estimated_volume_per_minute and estimated_volume_per_minute > 0:
        # Time based on volume participation
        minutes_needed = float(quantity / (estimated_volume_per_minute * Decimal(str(participation_rate))))
        return timedelta(minutes=max(5, minutes_needed))  # Minimum 5 minutes
    else:
        # Default heuristic based on quantity
        if quantity < Decimal('1000'):
            return timedelta(minutes=5)
        elif quantity < Decimal('10000'):
            return timedelta(minutes=15)
        elif quantity < Decimal('100000'):
            return timedelta(hours=1)
        else:
            return timedelta(hours=4)


# Export all public components
__all__ = [
    # Core classes
    'Order', 'OrderFill', 'OrderType', 'OrderSide', 'OrderStatus', 'TimeInForce', 'OrderPriority',
    'OrderValidator', 'OrderBook', 'OrderManager',
    'ExecutionStrategy', 'ExecutionPlan', 'MarketCondition', 'MarketMetrics', 'MarketAnalyzer', 'SmartRouter',
    'Position', 'PositionSnapshot', 'PositionSide', 'PositionStatus', 'PositionRiskMonitor', 'PositionManager',
    'ExecutionMetrics', 'StrategyMetrics', 'MarketImpactModel', 'BenchmarkType', 'ExecutionAnalyzer',
    
    # Unified engine
    'ExecutionEngine',
    
    # Factory functions
    'create_order_manager', 'create_smart_router', 'create_position_manager',
    'create_execution_analyzer', 'create_execution_engine',
    
    # Utility functions
    'calculate_position_size', 'calculate_optimal_order_size', 'estimate_execution_time',
    
    # Configuration
    'DEFAULT_EXECUTION_CONFIG'
]

# These will be implemented in Phase 4
# from .base import ExecutionEngine
# from .paper_trade import PaperTradeEngine
# from .live_trade import LiveTradeEngine

__all__ = [
    # "ExecutionEngine",
    # "PaperTradeEngine", 
    # "LiveTradeEngine",
]