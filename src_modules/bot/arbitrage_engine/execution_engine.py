"""
Arbitrage Execution Engine
Handles actual execution of arbitrage opportunities
Coordinates with Australian compliance and risk management
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .arbitrage_detector import ArbitrageOpportunity, ArbitrageType
from ..australian_compliance.ato_integration import AustralianTaxCalculator

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """Execution status tracking"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExecutionStage(Enum):
    """Stages of arbitrage execution"""
    VALIDATION = "validation"
    BUY_ORDER = "buy_order"
    TRANSFER = "transfer"
    SELL_ORDER = "sell_order"
    SETTLEMENT = "settlement"

@dataclass
class ExecutionRecord:
    """Record of arbitrage execution"""
    opportunity_id: str
    execution_id: str
    status: ExecutionStatus
    current_stage: ExecutionStage
    
    # Execution details
    planned_amount: Decimal
    actual_amount: Decimal
    start_time: datetime
    completion_time: Optional[datetime]
    
    # Financial results
    planned_profit: Decimal
    actual_profit: Optional[Decimal]
    total_costs: Decimal
    
    # Order details
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    buy_fill_price: Optional[Decimal] = None
    sell_fill_price: Optional[Decimal] = None
    
    # Risk management
    slippage_limit: Decimal = Decimal('0.005')  # 0.5% max slippage
    max_execution_time: int = 300  # 5 minutes max
    
    # Australian compliance
    tax_event_created: bool = False
    compliance_checked: bool = False
    
    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0

class ArbitrageExecutionValidator:
    """
    Validates arbitrage opportunities before execution
    Ensures compliance with Australian regulations and risk limits
    """
    
    def __init__(self, tax_calculator: AustralianTaxCalculator):
        self.tax_calculator = tax_calculator
        self.max_daily_volume = Decimal('100000')  # $100k AUD daily limit
        self.daily_volume_tracked = Decimal('0')
        self.last_reset_date = datetime.now().date()
    
    def _reset_daily_tracking_if_needed(self):
        """Reset daily volume tracking if new day"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_volume_tracked = Decimal('0')
            self.last_reset_date = today
    
    async def validate_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        planned_amount: Decimal,
        current_balance: Decimal
    ) -> Tuple[bool, Optional[str]]:
        """Validate opportunity before execution"""
        
        self._reset_daily_tracking_if_needed()
        
        # Check if opportunity is still valid
        if not opportunity.is_valid():
            return False, "Opportunity has expired"
        
        # Check if still profitable
        if not opportunity.is_profitable():
            return False, "Opportunity is no longer profitable"
        
        # Check balance sufficiency
        required_balance = planned_amount + opportunity.total_costs
        if required_balance > current_balance:
            return False, f"Insufficient balance: need ${required_balance:,.2f}, have ${current_balance:,.2f}"
        
        # Check amount limits
        if planned_amount < opportunity.minimum_amount:
            return False, f"Amount ${planned_amount:,.2f} below minimum ${opportunity.minimum_amount:,.2f}"
        
        if planned_amount > opportunity.maximum_amount:
            return False, f"Amount ${planned_amount:,.2f} exceeds maximum ${opportunity.maximum_amount:,.2f}"
        
        # Check daily volume limits
        if self.daily_volume_tracked + planned_amount > self.max_daily_volume:
            return False, f"Would exceed daily volume limit of ${self.max_daily_volume:,.2f}"
        
        # Check execution time constraints
        estimated_completion = datetime.now() + timedelta(minutes=opportunity.estimated_execution_time)
        if estimated_completion > opportunity.expires_at:
            return False, "Insufficient time remaining for execution"
        
        # Australian tax compliance check
        if opportunity.australian_friendly:
            # Check if this would trigger any tax reporting requirements
            annual_volume = await self._get_annual_trading_volume()
            if annual_volume + planned_amount > Decimal('10000'):  # ATO reporting threshold
                logger.info("Trade will require ATO reporting due to volume threshold")
        
        return True, None
    
    async def _get_annual_trading_volume(self) -> Decimal:
        """Get annual trading volume for tax compliance"""
        # This would integrate with the tax calculator to get actual volume
        # For now, return placeholder
        return Decimal('5000')
    
    def reserve_daily_volume(self, amount: Decimal):
        """Reserve daily volume for execution"""
        self.daily_volume_tracked += amount
    
    def release_daily_volume(self, amount: Decimal):
        """Release reserved daily volume if execution fails"""
        self.daily_volume_tracked = max(Decimal('0'), self.daily_volume_tracked - amount)

class OrderManager:
    """
    Manages individual orders within arbitrage execution
    Handles order placement, monitoring, and cancellation
    """
    
    def __init__(self):
        self.active_orders = {}
        self.order_timeout = 60  # 60 seconds per order
    
    async def place_market_order(
        self,
        exchange: str,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        amount: Decimal,
        max_slippage: Decimal = Decimal('0.005')
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Place market order with slippage protection"""
        
        try:
            # Get current market price for slippage calculation
            current_price = await self._get_current_price(exchange, symbol)
            if not current_price:
                return False, "Could not get current price", None
            
            # Calculate slippage limits
            if side == 'buy':
                max_price = current_price * (1 + max_slippage)
                min_price = None
            else:
                max_price = None
                min_price = current_price * (1 - max_slippage)
            
            # Simulate order placement (in practice, would use exchange API)
            order_id = f"order_{exchange}_{symbol}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Simulate order execution with slight slippage
            import random
            slippage_factor = Decimal(str(random.uniform(-0.002, 0.002)))  # ±0.2% slippage
            fill_price = current_price * (1 + slippage_factor)
            
            # Check if slippage is within acceptable limits
            if side == 'buy' and fill_price > max_price:
                return False, f"Slippage too high: filled at ${fill_price:.2f}, max ${max_price:.2f}", None
            elif side == 'sell' and fill_price < min_price:
                return False, f"Slippage too high: filled at ${fill_price:.2f}, min ${min_price:.2f}", None
            
            # Simulate successful fill
            fill_info = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'fill_price': fill_price,
                'slippage': abs(fill_price - current_price) / current_price,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Order filled: {side} {amount} {symbol} at ${fill_price:.2f} "
                       f"(slippage: {fill_info['slippage']:.4f})")
            
            return True, order_id, fill_info
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False, str(e), None
    
    async def _get_current_price(self, exchange: str, symbol: str) -> Optional[Decimal]:
        """Get current market price for symbol"""
        # Simulate price fetching
        base_prices = {
            'BTC/AUD': Decimal('65000'),
            'ETH/AUD': Decimal('2600'),
            'BTC/USDT': Decimal('43000'),
            'ETH/USDT': Decimal('1730')
        }
        
        base_price = base_prices.get(symbol)
        if base_price:
            # Add some random variation
            import random
            variation = Decimal(str(random.uniform(-0.01, 0.01)))  # ±1% variation
            return base_price * (1 + variation)
        
        return None
    
    async def cancel_order(self, exchange: str, order_id: str) -> bool:
        """Cancel an order"""
        try:
            # Simulate order cancellation
            logger.info(f"Cancelled order {order_id} on {exchange}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

class TransferManager:
    """
    Manages transfers between exchanges during arbitrage execution
    """
    
    def __init__(self):
        self.transfer_timeout = 1800  # 30 minutes max transfer time
        self.active_transfers = {}
    
    async def initiate_transfer(
        self,
        from_exchange: str,
        to_exchange: str,
        currency: str,
        amount: Decimal
    ) -> Tuple[bool, Optional[str]]:
        """Initiate transfer between exchanges"""
        
        try:
            transfer_id = f"transfer_{from_exchange}_{to_exchange}_{currency}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Simulate transfer initiation
            logger.info(f"Initiating transfer: {amount} {currency} from {from_exchange} to {to_exchange}")
            
            # Estimate transfer time based on currency and exchanges
            if currency in ['BTC', 'ETH']:
                estimated_time = 20  # 20 minutes for crypto
            else:
                estimated_time = 60  # 60 minutes for fiat/stablecoins
            
            self.active_transfers[transfer_id] = {
                'from_exchange': from_exchange,
                'to_exchange': to_exchange,
                'currency': currency,
                'amount': amount,
                'start_time': datetime.now(),
                'estimated_completion': datetime.now() + timedelta(minutes=estimated_time),
                'status': 'pending'
            }
            
            return True, transfer_id
            
        except Exception as e:
            logger.error(f"Error initiating transfer: {e}")
            return False, str(e)
    
    async def check_transfer_status(self, transfer_id: str) -> Tuple[str, Optional[datetime]]:
        """Check status of transfer"""
        
        if transfer_id not in self.active_transfers:
            return 'unknown', None
        
        transfer = self.active_transfers[transfer_id]
        
        # Simulate transfer completion based on estimated time
        if datetime.now() >= transfer['estimated_completion']:
            transfer['status'] = 'completed'
            return 'completed', transfer['estimated_completion']
        
        # Check if transfer has timed out
        if datetime.now() > transfer['start_time'] + timedelta(seconds=self.transfer_timeout):
            transfer['status'] = 'failed'
            return 'failed', None
        
        return transfer['status'], transfer['estimated_completion']

class ArbitrageExecutionEngine:
    """
    Main execution engine for arbitrage opportunities
    Coordinates validation, orders, transfers, and compliance
    """
    
    def __init__(self, tax_calculator: AustralianTaxCalculator):
        self.validator = ArbitrageExecutionValidator(tax_calculator)
        self.order_manager = OrderManager()
        self.transfer_manager = TransferManager()
        self.tax_calculator = tax_calculator
        
        self.active_executions = {}
        self.execution_history = []
        
        logger.info("Initialized Arbitrage Execution Engine")
    
    async def execute_arbitrage(
        self,
        opportunity: ArbitrageOpportunity,
        amount: Decimal,
        current_balance: Decimal
    ) -> ExecutionRecord:
        """Execute arbitrage opportunity"""
        
        execution_id = f"exec_{opportunity.opportunity_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create execution record
        execution = ExecutionRecord(
            opportunity_id=opportunity.opportunity_id,
            execution_id=execution_id,
            status=ExecutionStatus.PENDING,
            current_stage=ExecutionStage.VALIDATION,
            planned_amount=amount,
            actual_amount=Decimal('0'),
            start_time=datetime.now(),
            completion_time=None,
            planned_profit=amount * opportunity.net_profit_percentage / 100,
            actual_profit=None,
            total_costs=opportunity.total_costs
        )
        
        self.active_executions[execution_id] = execution
        
        try:
            # Stage 1: Validation
            execution.status = ExecutionStatus.EXECUTING
            execution.current_stage = ExecutionStage.VALIDATION
            
            is_valid, error_msg = await self.validator.validate_opportunity(
                opportunity, amount, current_balance
            )
            
            if not is_valid:
                execution.status = ExecutionStatus.FAILED
                execution.error_message = error_msg
                return execution
            
            # Reserve daily volume
            self.validator.reserve_daily_volume(amount)
            
            # Stage 2: Buy Order
            execution.current_stage = ExecutionStage.BUY_ORDER
            
            buy_success, buy_order_id, buy_fill_info = await self.order_manager.place_market_order(
                exchange=opportunity.buy_exchange,
                symbol=opportunity.symbol,
                side='buy',
                amount=amount,
                max_slippage=execution.slippage_limit
            )
            
            if not buy_success:
                execution.status = ExecutionStatus.FAILED
                execution.error_message = f"Buy order failed: {buy_order_id}"
                self.validator.release_daily_volume(amount)
                return execution
            
            execution.buy_order_id = buy_order_id
            execution.buy_fill_price = buy_fill_info['fill_price']
            
            # Create tax event for buy
            if opportunity.australian_friendly:
                await self._create_tax_event(execution, buy_fill_info, 'buy')
            
            # Stage 3: Transfer (if needed)
            if opportunity.buy_exchange != opportunity.sell_exchange:
                execution.current_stage = ExecutionStage.TRANSFER
                
                # Extract currency from symbol
                currency = opportunity.symbol.split('/')[0]
                
                transfer_success, transfer_id = await self.transfer_manager.initiate_transfer(
                    from_exchange=opportunity.buy_exchange,
                    to_exchange=opportunity.sell_exchange,
                    currency=currency,
                    amount=amount
                )
                
                if not transfer_success:
                    execution.status = ExecutionStatus.FAILED
                    execution.error_message = f"Transfer initiation failed: {transfer_id}"
                    return execution
                
                # Wait for transfer completion
                await self._wait_for_transfer(transfer_id, execution)
                
                if execution.status == ExecutionStatus.FAILED:
                    return execution
            
            # Stage 4: Sell Order
            execution.current_stage = ExecutionStage.SELL_ORDER
            
            sell_success, sell_order_id, sell_fill_info = await self.order_manager.place_market_order(
                exchange=opportunity.sell_exchange,
                symbol=opportunity.symbol,
                side='sell',
                amount=amount,
                max_slippage=execution.slippage_limit
            )
            
            if not sell_success:
                execution.status = ExecutionStatus.FAILED
                execution.error_message = f"Sell order failed: {sell_order_id}"
                return execution
            
            execution.sell_order_id = sell_order_id
            execution.sell_fill_price = sell_fill_info['fill_price']
            
            # Create tax event for sell
            if opportunity.australian_friendly:
                await self._create_tax_event(execution, sell_fill_info, 'sell')
            
            # Stage 5: Settlement
            execution.current_stage = ExecutionStage.SETTLEMENT
            
            # Calculate actual profit
            gross_profit = (execution.sell_fill_price - execution.buy_fill_price) * amount
            execution.actual_profit = gross_profit - execution.total_costs
            execution.actual_amount = amount
            
            # Complete execution
            execution.status = ExecutionStatus.COMPLETED
            execution.completion_time = datetime.now()
            
            logger.info(f"Arbitrage execution completed: {execution_id}, "
                       f"profit: ${execution.actual_profit:.2f}")
            
        except Exception as e:
            logger.error(f"Error during arbitrage execution: {e}")
            execution.status = ExecutionStatus.FAILED
            execution.error_message = str(e)
            self.validator.release_daily_volume(amount)
        
        finally:
            # Move to history
            self.execution_history.append(execution)
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
        
        return execution
    
    async def _wait_for_transfer(
        self,
        transfer_id: str,
        execution: ExecutionRecord,
        max_wait_time: int = 1800  # 30 minutes
    ):
        """Wait for transfer to complete"""
        
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < max_wait_time:
            status, completion_time = await self.transfer_manager.check_transfer_status(transfer_id)
            
            if status == 'completed':
                logger.info(f"Transfer {transfer_id} completed")
                return
            elif status == 'failed':
                execution.status = ExecutionStatus.FAILED
                execution.error_message = f"Transfer {transfer_id} failed"
                return
            
            # Wait before checking again
            await asyncio.sleep(30)  # Check every 30 seconds
        
        # Transfer timed out
        execution.status = ExecutionStatus.FAILED
        execution.error_message = f"Transfer {transfer_id} timed out"
    
    async def _create_tax_event(
        self,
        execution: ExecutionRecord,
        fill_info: Dict,
        side: str
    ):
        """Create tax event for Australian compliance"""
        
        try:
            await self.tax_calculator.record_trade(
                symbol=fill_info['symbol'],
                side=side,
                amount=fill_info['amount'],
                price=fill_info['fill_price'],
                timestamp=fill_info['timestamp'],
                exchange="arbitrage_execution",
                fees=Decimal('0')  # Fees tracked separately
            )
            
            execution.tax_event_created = True
            logger.info(f"Tax event created for {side} order in execution {execution.execution_id}")
            
        except Exception as e:
            logger.error(f"Error creating tax event: {e}")
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionRecord]:
        """Get status of execution"""
        
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]
        
        # Check history
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return execution
        
        return None
    
    def get_performance_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get execution performance summary"""
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter recent executions
        recent_executions = [
            exec for exec in self.execution_history
            if exec.start_time >= cutoff_date
        ]
        
        if not recent_executions:
            return {
                'total_executions': 0,
                'success_rate': 0,
                'total_profit': Decimal('0'),
                'average_profit': Decimal('0'),
                'by_status': {}
            }
        
        # Calculate metrics
        successful = [exec for exec in recent_executions if exec.status == ExecutionStatus.COMPLETED]
        total_profit = sum(exec.actual_profit or Decimal('0') for exec in successful)
        
        status_counts = {}
        for exec in recent_executions:
            status = exec.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_executions': len(recent_executions),
            'successful_executions': len(successful),
            'success_rate': len(successful) / len(recent_executions) if recent_executions else 0,
            'total_profit': total_profit,
            'average_profit': total_profit / len(successful) if successful else Decimal('0'),
            'by_status': status_counts,
            'average_execution_time': np.mean([
                (exec.completion_time - exec.start_time).total_seconds() / 60
                for exec in successful
                if exec.completion_time
            ]) if successful else 0
        }

# Usage example
async def main():
    """Example usage of execution engine"""
    
    from ..australian_compliance.ato_integration import AustralianTaxCalculator
    
    # Initialize components
    tax_calculator = AustralianTaxCalculator()
    execution_engine = ArbitrageExecutionEngine(tax_calculator)
    
    # Create sample opportunity (would come from detector)
    from .arbitrage_detector import ArbitrageOpportunity, ArbitrageType, OpportunityTier
    
    opportunity = ArbitrageOpportunity(
        opportunity_id="test_opp_001",
        arbitrage_type=ArbitrageType.SIMPLE_ARBITRAGE,
        tier=OpportunityTier.SMALL,
        buy_exchange="btcmarkets",
        sell_exchange="bybit",
        symbol="BTC/AUD",
        buy_price=Decimal('64500'),
        sell_price=Decimal('65000'),
        price_difference=Decimal('500'),
        gross_profit_percentage=Decimal('0.77'),
        transfer_costs=Decimal('50'),
        trading_fees=Decimal('100'),
        total_costs=Decimal('150'),
        net_profit_percentage=Decimal('0.54'),
        minimum_amount=Decimal('1000'),
        maximum_amount=Decimal('10000'),
        estimated_execution_time=30,
        liquidity_score=0.8,
        volatility_risk=0.4,
        australian_friendly=True,
        detected_at=datetime.now(),
        expires_at=datetime.now() + timedelta(minutes=10)
    )
    
    # Execute arbitrage
    print("Executing arbitrage opportunity...")
    execution = await execution_engine.execute_arbitrage(
        opportunity=opportunity,
        amount=Decimal('5000'),
        current_balance=Decimal('25000')
    )
    
    print(f"Execution Status: {execution.status.value}")
    print(f"Execution Stage: {execution.current_stage.value}")
    
    if execution.status == ExecutionStatus.COMPLETED:
        print(f"Actual Profit: ${execution.actual_profit:.2f}")
        print(f"Buy Price: ${execution.buy_fill_price:.2f}")
        print(f"Sell Price: ${execution.sell_fill_price:.2f}")
    elif execution.status == ExecutionStatus.FAILED:
        print(f"Error: {execution.error_message}")
    
    # Get performance summary
    performance = execution_engine.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"  Success Rate: {performance['success_rate']:.1%}")
    print(f"  Total Profit: ${performance['total_profit']:.2f}")
    print(f"  Average Profit: ${performance['average_profit']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())