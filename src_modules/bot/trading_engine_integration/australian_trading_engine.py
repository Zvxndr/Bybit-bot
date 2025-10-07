"""
Australian Trading Engine Integration
Integrates ML strategy signals and arbitrage opportunities with existing trading engine
Implements Australian-specific execution logic with compliance monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

# Import existing trading engine components
from ..bot import TradingBot
from ..exchanges.bybit_client import BybitClient

# Import new Australian-specific components
from ..australian_compliance.ato_integration import AustralianTaxCalculator
from ..australian_compliance.regulatory_compliance import AustralianComplianceManager
from ..ml_strategy_discovery.ml_engine import MLStrategyDiscoveryEngine, StrategySignal
from ..arbitrage_engine.arbitrage_detector import OpportunisticArbitrageEngine, ArbitrageOpportunity
from ..risk_management.australian_risk_manager import AustralianRiskCalculator, RiskParameters
from ..risk_management.portfolio_risk_controller import PortfolioRiskController

logger = logging.getLogger(__name__)

class ExecutionPriority(Enum):
    """Execution priority levels"""
    EMERGENCY = "emergency"      # Emergency risk management
    HIGH = "high"               # High-profit arbitrage or strong ML signals
    MEDIUM = "medium"           # Standard ML signals
    LOW = "low"                 # Low-confidence opportunities
    TAX_OPTIMIZATION = "tax"    # Tax-loss harvesting or CGT optimization

class TradeSource(Enum):
    """Source of trading signal"""
    ML_STRATEGY = "ml_strategy"
    ARBITRAGE = "arbitrage"
    RISK_MANAGEMENT = "risk_management"
    TAX_OPTIMIZATION = "tax_optimization"
    MANUAL = "manual"

@dataclass
class AustralianTradeRequest:
    """Enhanced trade request with Australian compliance data"""
    # Basic trade information
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: Decimal
    price: Optional[Decimal] = None  # None for market orders
    order_type: str = 'market'  # 'market' or 'limit'
    
    # Australian-specific information
    source: TradeSource = TradeSource.MANUAL
    priority: ExecutionPriority = ExecutionPriority.MEDIUM
    
    # Compliance and tax information
    tax_lot_method: str = 'FIFO'  # FIFO for Australian compliance
    expected_tax_impact: Optional[Decimal] = None
    cgt_discount_eligible: bool = False
    requires_ato_reporting: bool = False
    
    # Risk management
    risk_score: float = 0.5
    max_slippage: Decimal = Decimal('0.005')  # 0.5% default
    compliance_checked: bool = False
    
    # Strategy context
    signal_strength: Optional[float] = None
    confidence: Optional[float] = None
    strategy_context: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    request_id: str = field(default_factory=lambda: f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

@dataclass
class AustralianExecutionResult:
    """Enhanced execution result with Australian compliance tracking"""
    # Basic execution information
    request_id: str
    order_id: Optional[str]
    status: str  # 'completed', 'failed', 'partial', 'cancelled'
    
    # Execution details
    executed_amount: Decimal
    executed_price: Optional[Decimal]
    total_cost: Decimal
    fees: Decimal
    slippage: Decimal
    
    # Australian compliance
    tax_event_created: bool = False
    ato_reportable: bool = False
    cgt_event_id: Optional[str] = None
    
    # Performance tracking
    execution_time_ms: int = 0
    execution_timestamp: datetime = field(default_factory=datetime.now)
    
    # Error information
    error_message: Optional[str] = None
    compliance_issues: List[str] = field(default_factory=list)

class AustralianOrderRouter:
    """
    Smart order router optimized for Australian traders
    Routes orders to appropriate exchanges considering costs and compliance
    """
    
    def __init__(self):
        self.exchange_preferences = {
            # Australian exchanges - preferred for AUD pairs and compliance
            'btcmarkets': {
                'priority': 1,
                'supports_aud': True,
                'australian_regulated': True,
                'trading_fee_maker': Decimal('0.0085'),
                'trading_fee_taker': Decimal('0.0085'),
                'withdrawal_fee_aud': Decimal('0')
            },
            'coinjar': {
                'priority': 2,
                'supports_aud': True,
                'australian_regulated': True,
                'trading_fee_maker': Decimal('0.001'),
                'trading_fee_taker': Decimal('0.001'),
                'withdrawal_fee_aud': Decimal('0')
            },
            'swyftx': {
                'priority': 3,
                'supports_aud': True,
                'australian_regulated': True,
                'trading_fee_maker': Decimal('0.006'),
                'trading_fee_taker': Decimal('0.006'),
                'withdrawal_fee_aud': Decimal('0')
            },
            
            # International exchanges - for specific opportunities
            'bybit': {
                'priority': 4,
                'supports_aud': False,
                'australian_regulated': False,
                'trading_fee_maker': Decimal('0.001'),
                'trading_fee_taker': Decimal('0.006'),
                'withdrawal_fee_btc': Decimal('0.0005')
            },
            'binance': {
                'priority': 5,
                'supports_aud': False,
                'australian_regulated': False,
                'trading_fee_maker': Decimal('0.001'),
                'trading_fee_taker': Decimal('0.001'),
                'withdrawal_fee_btc': Decimal('0.0005')
            }
        }
        
        self.australian_exchanges = {'btcmarkets', 'coinjar', 'swyftx'}
        
    def select_optimal_exchange(
        self,
        trade_request: AustralianTradeRequest,
        available_exchanges: List[str],
        liquidity_data: Optional[Dict[str, Dict]] = None
    ) -> Tuple[str, str]:
        """Select optimal exchange for trade execution"""
        
        symbol = trade_request.symbol
        amount = trade_request.amount
        
        # Filter available exchanges
        candidate_exchanges = [
            ex for ex in available_exchanges 
            if ex in self.exchange_preferences
        ]
        
        if not candidate_exchanges:
            return None, "No supported exchanges available"
        
        # Prioritize Australian exchanges for AUD pairs and compliance
        if symbol.endswith('AUD') or trade_request.requires_ato_reporting:
            australian_candidates = [
                ex for ex in candidate_exchanges 
                if ex in self.australian_exchanges
            ]
            if australian_candidates:
                candidate_exchanges = australian_candidates
        
        # Score exchanges based on multiple factors
        exchange_scores = {}
        
        for exchange in candidate_exchanges:
            config = self.exchange_preferences[exchange]
            score = 0
            
            # Base priority score
            score += (10 - config['priority']) * 10  # Higher priority = higher score
            
            # Australian regulation bonus
            if config.get('australian_regulated', False):
                score += 20
            
            # Fee consideration (lower fees = higher score)
            if trade_request.side == 'buy':
                fee = config.get('trading_fee_taker', Decimal('0.001'))
            else:
                fee = config.get('trading_fee_maker', Decimal('0.001'))
            
            fee_score = max(0, 10 - float(fee * 1000))  # Convert to basis points
            score += fee_score
            
            # Liquidity consideration
            if liquidity_data and exchange in liquidity_data:
                liquidity_score = liquidity_data[exchange].get('score', 0.5) * 10
                score += liquidity_score
            
            # Volume consideration (higher volume gets better pricing)
            if amount > Decimal('10000'):  # Large orders
                if exchange in self.australian_exchanges:
                    score += 5  # Australian exchanges better for large AUD orders
                else:
                    score += 10  # International exchanges better for large crypto orders
            
            exchange_scores[exchange] = score
        
        # Select exchange with highest score
        best_exchange = max(exchange_scores.items(), key=lambda x: x[1])
        
        return best_exchange[0], f"Selected based on score: {best_exchange[1]:.1f}"
    
    def calculate_execution_costs(
        self,
        exchange: str,
        symbol: str,
        amount: Decimal,
        side: str
    ) -> Dict[str, Decimal]:
        """Calculate total execution costs for exchange"""
        
        if exchange not in self.exchange_preferences:
            return {'total_cost': Decimal('0'), 'trading_fee': Decimal('0')}
        
        config = self.exchange_preferences[exchange]
        
        # Trading fees
        if side == 'buy':
            fee_rate = config.get('trading_fee_taker', Decimal('0.001'))
        else:
            fee_rate = config.get('trading_fee_maker', Decimal('0.001'))
        
        trading_fee = amount * fee_rate
        
        # Withdrawal fees (if applicable)
        withdrawal_fee = Decimal('0')
        if 'withdrawal_fee_aud' in config and symbol.endswith('AUD'):
            withdrawal_fee = config['withdrawal_fee_aud']
        elif 'withdrawal_fee_btc' in config and symbol.startswith('BTC'):
            withdrawal_fee = config['withdrawal_fee_btc']
        
        total_cost = trading_fee + withdrawal_fee
        
        return {
            'total_cost': total_cost,
            'trading_fee': trading_fee,
            'withdrawal_fee': withdrawal_fee,
            'fee_rate': fee_rate
        }

class AustralianComplianceExecutor:
    """
    Execution engine with integrated Australian compliance checking
    Ensures all trades comply with ATO, ASIC, and AUSTRAC requirements
    """
    
    def __init__(
        self,
        tax_calculator: AustralianTaxCalculator,
        compliance_manager: AustralianComplianceManager,
        risk_controller: PortfolioRiskController
    ):
        self.tax_calculator = tax_calculator
        self.compliance_manager = compliance_manager
        self.risk_controller = risk_controller
        self.order_router = AustralianOrderRouter()
        
        # Execution tracking
        self.active_orders = {}
        self.execution_history = []
        self.daily_volume_aud = Decimal('0')
        self.last_volume_reset = datetime.now().date()
        
        # Compliance limits
        self.daily_volume_limit = Decimal('100000')  # $100k AUD ATO threshold
        self.max_single_trade_limit = Decimal('50000')  # $50k single trade limit
        
        logger.info("Initialized Australian Compliance Executor")
    
    async def validate_trade_compliance(
        self,
        trade_request: AustralianTradeRequest
    ) -> Tuple[bool, List[str], List[str]]:
        """Comprehensive compliance validation for trade request"""
        
        errors = []
        warnings = []
        
        # Reset daily volume tracking if new day
        today = datetime.now().date()
        if today > self.last_volume_reset:
            self.daily_volume_aud = Decimal('0')
            self.last_volume_reset = today
        
        # Volume limit checks
        trade_value_aud = await self._estimate_trade_value_aud(trade_request)
        
        if trade_value_aud > self.max_single_trade_limit:
            errors.append(f"Trade value ${trade_value_aud:,.2f} exceeds single trade limit ${self.max_single_trade_limit:,.2f}")
        
        if self.daily_volume_aud + trade_value_aud > self.daily_volume_limit:
            errors.append(f"Trade would exceed daily volume limit: ${self.daily_volume_aud + trade_value_aud:,.2f} > ${self.daily_volume_limit:,.2f}")
        elif self.daily_volume_aud + trade_value_aud > self.daily_volume_limit * Decimal('0.8'):
            warnings.append(f"Approaching daily volume limit: ${self.daily_volume_aud + trade_value_aud:,.2f}")
        
        # ATO reporting requirements
        if trade_value_aud > Decimal('10000'):
            trade_request.requires_ato_reporting = True
            warnings.append("Trade requires ATO reporting (>$10,000 AUD)")
        
        # Professional trader activity monitoring
        annual_trades = await self._get_annual_trade_count()
        if annual_trades > 40:  # Approaching professional threshold
            warnings.append(f"High trading activity ({annual_trades} trades this year) may trigger professional trader status")
        
        # AUSTRAC cash threshold monitoring
        if trade_value_aud > Decimal('10000') and trade_request.symbol.endswith('AUD'):
            warnings.append("Large AUD transaction may require AUSTRAC reporting")
        
        # Tax impact validation
        if trade_request.side == 'sell':
            tax_impact = await self._calculate_tax_impact(trade_request)
            if tax_impact:
                trade_request.expected_tax_impact = tax_impact['total_tax']
                trade_request.cgt_discount_eligible = tax_impact['cgt_discount_eligible']
                
                if tax_impact['total_tax'] > trade_value_aud * Decimal('0.3'):
                    warnings.append(f"High tax impact: ${tax_impact['total_tax']:,.2f} ({tax_impact['total_tax']/trade_value_aud:.1%})")
        
        # Risk management validation
        risk_check = await self.risk_controller.assess_portfolio_risks(
            await self.risk_controller.update_portfolio_state({}, Decimal('100000'))  # Placeholder
        )
        
        if any(alert.level.value == 'emergency' for alert in risk_check):
            errors.append("Emergency risk conditions detected - trading halted")
        
        is_compliant = len(errors) == 0
        trade_request.compliance_checked = True
        
        return is_compliant, errors, warnings
    
    async def _estimate_trade_value_aud(self, trade_request: AustralianTradeRequest) -> Decimal:
        """Estimate trade value in AUD for compliance checking"""
        
        # If already AUD pair, direct calculation
        if trade_request.symbol.endswith('AUD'):
            return trade_request.amount
        
        # For crypto pairs, estimate using current prices (simplified)
        # In practice, would fetch real market prices
        crypto_to_aud_rates = {
            'BTC': Decimal('65000'),
            'ETH': Decimal('2600'),
            'ADA': Decimal('0.45'),
            'DOT': Decimal('6.50')
        }
        
        base_currency = trade_request.symbol.split('/')[0] if '/' in trade_request.symbol else trade_request.symbol[:3]
        rate = crypto_to_aud_rates.get(base_currency, Decimal('1'))
        
        return trade_request.amount * rate
    
    async def _get_annual_trade_count(self) -> int:
        """Get annual trade count for professional trader monitoring"""
        # Would integrate with actual trade history
        return 25  # Placeholder
    
    async def _calculate_tax_impact(self, trade_request: AustralianTradeRequest) -> Optional[Dict]:
        """Calculate tax impact for sell orders"""
        
        if trade_request.side != 'sell':
            return None
        
        try:
            # Get position information from tax calculator
            # This is simplified - would integrate with actual position tracking
            
            return {
                'total_tax': Decimal('500'),  # Placeholder tax calculation
                'cgt_discount_eligible': True,
                'holding_period_days': 400,
                'cost_base': Decimal('60000'),
                'capital_gain': Decimal('5000')
            }
            
        except Exception as e:
            logger.error(f"Error calculating tax impact: {e}")
            return None
    
    async def execute_trade(
        self,
        trade_request: AustralianTradeRequest,
        available_exchanges: List[str]
    ) -> AustralianExecutionResult:
        """Execute trade with full Australian compliance integration"""
        
        start_time = datetime.now()
        
        # Validate compliance first
        is_compliant, errors, warnings = await self.validate_trade_compliance(trade_request)
        
        if not is_compliant:
            return AustralianExecutionResult(
                request_id=trade_request.request_id,
                order_id=None,
                status='failed',
                executed_amount=Decimal('0'),
                executed_price=None,
                total_cost=Decimal('0'),
                fees=Decimal('0'),
                slippage=Decimal('0'),
                error_message=f"Compliance validation failed: {'; '.join(errors)}",
                compliance_issues=errors
            )
        
        # Select optimal exchange
        selected_exchange, selection_reason = self.order_router.select_optimal_exchange(
            trade_request, available_exchanges
        )
        
        if not selected_exchange:
            return AustralianExecutionResult(
                request_id=trade_request.request_id,
                order_id=None,
                status='failed',
                executed_amount=Decimal('0'),
                executed_price=None,
                total_cost=Decimal('0'),
                fees=Decimal('0'),
                slippage=Decimal('0'),
                error_message=selection_reason
            )
        
        # Calculate execution costs
        cost_breakdown = self.order_router.calculate_execution_costs(
            selected_exchange, trade_request.symbol, trade_request.amount, trade_request.side
        )
        
        try:
            # Execute order (simplified - would integrate with actual exchange APIs)
            order_id = f"order_{selected_exchange}_{trade_request.request_id}"
            
            # Simulate order execution
            executed_amount = trade_request.amount
            executed_price = Decimal('65000')  # Placeholder price
            total_cost = executed_amount * executed_price if trade_request.side == 'buy' else executed_amount
            fees = cost_breakdown['trading_fee']
            slippage = Decimal('0.001')  # 0.1% simulated slippage
            
            # Create tax event if required
            tax_event_created = False
            cgt_event_id = None
            
            if trade_request.requires_ato_reporting or await self._estimate_trade_value_aud(trade_request) > Decimal('1000'):
                try:
                    cgt_event_id = await self.tax_calculator.record_trade(
                        symbol=trade_request.symbol,
                        side=trade_request.side,
                        amount=executed_amount,
                        price=executed_price,
                        timestamp=datetime.now(),
                        exchange=selected_exchange,
                        fees=fees
                    )
                    tax_event_created = True
                    logger.info(f"Created tax event {cgt_event_id} for trade {order_id}")
                except Exception as e:
                    logger.error(f"Failed to create tax event: {e}")
            
            # Update daily volume tracking
            trade_value_aud = await self._estimate_trade_value_aud(trade_request)
            self.daily_volume_aud += trade_value_aud
            
            # Create execution result
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = AustralianExecutionResult(
                request_id=trade_request.request_id,
                order_id=order_id,
                status='completed',
                executed_amount=executed_amount,
                executed_price=executed_price,
                total_cost=total_cost,
                fees=fees,
                slippage=slippage,
                tax_event_created=tax_event_created,
                ato_reportable=trade_request.requires_ato_reporting,
                cgt_event_id=cgt_event_id,
                execution_time_ms=int(execution_time),
                compliance_issues=[],
                execution_timestamp=datetime.now()
            )
            
            # Store in execution history
            self.execution_history.append(result)
            
            logger.info(f"Successfully executed trade {trade_request.request_id} on {selected_exchange}: "
                       f"{executed_amount} {trade_request.symbol} at ${executed_price}")
            
            return result
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            
            return AustralianExecutionResult(
                request_id=trade_request.request_id,
                order_id=None,
                status='failed',
                executed_amount=Decimal('0'),
                executed_price=None,
                total_cost=Decimal('0'),
                fees=Decimal('0'),
                slippage=Decimal('0'),
                error_message=str(e),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

class MLStrategyExecutor:
    """
    Executor for ML strategy signals with Australian compliance
    """
    
    def __init__(self, compliance_executor: AustralianComplianceExecutor):
        self.compliance_executor = compliance_executor
        self.active_ml_positions = {}
        
    async def execute_ml_signal(
        self,
        signal: StrategySignal,
        recommended_size_aud: Decimal,
        available_exchanges: List[str]
    ) -> AustralianExecutionResult:
        """Execute ML strategy signal"""
        
        # Create trade request from ML signal
        trade_request = AustralianTradeRequest(
            symbol=signal.symbol,
            side='buy' if signal.signal_strength > 0 else 'sell',
            amount=recommended_size_aud,
            source=TradeSource.ML_STRATEGY,
            priority=ExecutionPriority.HIGH if abs(signal.signal_strength) > 0.7 else ExecutionPriority.MEDIUM,
            signal_strength=signal.signal_strength,
            confidence=signal.confidence,
            strategy_context={
                'strategy_type': signal.strategy_type.value,
                'predicted_return': signal.predicted_return,
                'prediction_horizon': signal.prediction_horizon,
                'features_used': signal.features_used,
                'model_version': signal.model_version
            },
            max_slippage=Decimal('0.01') if abs(signal.signal_strength) > 0.8 else Decimal('0.005'),
            expires_at=datetime.now() + timedelta(minutes=30)  # ML signals expire in 30 minutes
        )
        
        # Execute with compliance checking
        result = await self.compliance_executor.execute_trade(trade_request, available_exchanges)
        
        # Track ML position
        if result.status == 'completed':
            self.active_ml_positions[signal.symbol] = {
                'signal': signal,
                'execution_result': result,
                'entry_time': datetime.now(),
                'position_size': result.executed_amount,
                'entry_price': result.executed_price
            }
        
        return result

class ArbitrageExecutor:
    """
    Executor for arbitrage opportunities with Australian compliance
    """
    
    def __init__(self, compliance_executor: AustralianComplianceExecutor):
        self.compliance_executor = compliance_executor
        self.active_arbitrage_trades = {}
        
    async def execute_arbitrage_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        position_size: Decimal,
        available_exchanges: List[str]
    ) -> Tuple[AustralianExecutionResult, Optional[AustralianExecutionResult]]:
        """Execute arbitrage opportunity (buy and sell sides)"""
        
        # Create buy trade request
        buy_request = AustralianTradeRequest(
            symbol=opportunity.symbol,
            side='buy',
            amount=position_size,
            source=TradeSource.ARBITRAGE,
            priority=ExecutionPriority.HIGH,  # Arbitrage is time-sensitive
            strategy_context={
                'arbitrage_type': opportunity.arbitrage_type.value,
                'buy_exchange': opportunity.buy_exchange,
                'sell_exchange': opportunity.sell_exchange,
                'expected_profit_pct': float(opportunity.net_profit_percentage),
                'opportunity_id': opportunity.opportunity_id
            },
            max_slippage=Decimal('0.002'),  # Tight slippage for arbitrage
            expires_at=opportunity.expires_at
        )
        
        # Execute buy side first
        buy_result = await self.compliance_executor.execute_trade(
            buy_request, [opportunity.buy_exchange]
        )
        
        if buy_result.status != 'completed':
            return buy_result, None
        
        # Create sell trade request
        sell_request = AustralianTradeRequest(
            symbol=opportunity.symbol,
            side='sell',
            amount=buy_result.executed_amount,  # Use actual executed amount
            source=TradeSource.ARBITRAGE,
            priority=ExecutionPriority.HIGH,
            strategy_context=buy_request.strategy_context,
            max_slippage=Decimal('0.002'),
            expires_at=opportunity.expires_at
        )
        
        # Execute sell side
        sell_result = await self.compliance_executor.execute_trade(
            sell_request, [opportunity.sell_exchange]
        )
        
        # Track arbitrage trade
        if buy_result.status == 'completed' and sell_result.status == 'completed':
            self.active_arbitrage_trades[opportunity.opportunity_id] = {
                'opportunity': opportunity,
                'buy_result': buy_result,
                'sell_result': sell_result,
                'completion_time': datetime.now(),
                'actual_profit': (sell_result.executed_price - buy_result.executed_price) * buy_result.executed_amount - buy_result.fees - sell_result.fees
            }
        
        return buy_result, sell_result

class AustralianTradingEngineIntegration:
    """
    Main integration engine coordinating ML strategies, arbitrage, and existing trading engine
    with comprehensive Australian compliance
    """
    
    def __init__(
        self,
        trading_bot: TradingBot,
        ml_engine: MLStrategyDiscoveryEngine,
        arbitrage_engine: OpportunisticArbitrageEngine,
        risk_controller: PortfolioRiskController,
        tax_calculator: AustralianTaxCalculator,
        compliance_manager: AustralianComplianceManager
    ):
        # Core components
        self.trading_bot = trading_bot
        self.ml_engine = ml_engine
        self.arbitrage_engine = arbitrage_engine
        self.risk_controller = risk_controller
        
        # Australian compliance
        self.compliance_executor = AustralianComplianceExecutor(
            tax_calculator, compliance_manager, risk_controller
        )
        
        # Strategy executors
        self.ml_executor = MLStrategyExecutor(self.compliance_executor)
        self.arbitrage_executor = ArbitrageExecutor(self.compliance_executor)
        
        # Available exchanges (would be configured based on actual setup)
        self.available_exchanges = ['btcmarkets', 'bybit', 'binance']
        
        # Execution tracking
        self.execution_queue = []
        self.processing_enabled = True
        
        logger.info("Initialized Australian Trading Engine Integration")
    
    async def process_ml_signals(
        self,
        signals: List[StrategySignal],
        portfolio_value_aud: Decimal
    ) -> List[AustralianExecutionResult]:
        """Process ML strategy signals for execution"""
        
        if not self.processing_enabled:
            logger.warning("Signal processing disabled")
            return []
        
        # Get current portfolio state
        portfolio_state = await self.risk_controller.update_portfolio_state(
            current_prices={'BTC/AUD': Decimal('65000')},  # Placeholder
            account_balance=portfolio_value_aud * Decimal('0.1')  # Assume 10% cash
        )
        
        # Generate position recommendations
        recommendations = await self.risk_controller.generate_ml_position_recommendations(
            signals, portfolio_state
        )
        
        execution_results = []
        
        for rec in recommendations:
            if not rec['compliant']:
                logger.warning(f"Skipping non-compliant ML signal: {rec['compliance_issues']}")
                continue
            
            # Find corresponding signal
            signal = next((s for s in signals if s.symbol == rec['symbol']), None)
            if not signal:
                continue
            
            # Execute signal
            try:
                result = await self.ml_executor.execute_ml_signal(
                    signal=signal,
                    recommended_size_aud=rec['recommended_size_aud'],
                    available_exchanges=self.available_exchanges
                )
                execution_results.append(result)
                
                logger.info(f"ML signal execution: {result.status} for {signal.symbol}")
                
            except Exception as e:
                logger.error(f"Error executing ML signal for {signal.symbol}: {e}")
        
        return execution_results
    
    async def process_arbitrage_opportunities(
        self,
        opportunities: List[ArbitrageOpportunity],
        portfolio_value_aud: Decimal
    ) -> List[Tuple[AustralianExecutionResult, Optional[AustralianExecutionResult]]]:
        """Process arbitrage opportunities for execution"""
        
        if not self.processing_enabled:
            logger.warning("Arbitrage processing disabled")
            return []
        
        # Get current portfolio state
        portfolio_state = await self.risk_controller.update_portfolio_state(
            current_prices={'BTC/AUD': Decimal('65000')},  # Placeholder
            account_balance=portfolio_value_aud * Decimal('0.1')
        )
        
        # Generate arbitrage recommendations
        recommendations = await self.risk_controller.generate_arbitrage_recommendations(
            opportunities, portfolio_state
        )
        
        execution_results = []
        
        for rec in recommendations:
            if not rec['compliant']:
                logger.warning(f"Skipping non-compliant arbitrage: {rec['compliance_issues']}")
                continue
            
            # Find corresponding opportunity
            opportunity = next((op for op in opportunities if op.opportunity_id == rec['opportunity_id']), None)
            if not opportunity:
                continue
            
            # Check if opportunity is still valid
            if not opportunity.is_valid():
                logger.warning(f"Arbitrage opportunity {opportunity.opportunity_id} expired")
                continue
            
            # Execute arbitrage
            try:
                buy_result, sell_result = await self.arbitrage_executor.execute_arbitrage_opportunity(
                    opportunity=opportunity,
                    position_size=rec['recommended_size_aud'],
                    available_exchanges=self.available_exchanges
                )
                execution_results.append((buy_result, sell_result))
                
                logger.info(f"Arbitrage execution: {buy_result.status}/{sell_result.status if sell_result else 'None'} "
                           f"for {opportunity.symbol}")
                
            except Exception as e:
                logger.error(f"Error executing arbitrage for {opportunity.symbol}: {e}")
        
        return execution_results
    
    async def run_trading_cycle(
        self,
        market_data: Dict[str, Any],
        portfolio_value_aud: Decimal
    ) -> Dict[str, Any]:
        """Run complete trading cycle with ML and arbitrage strategies"""
        
        cycle_start = datetime.now()
        cycle_summary = {
            'cycle_start': cycle_start,
            'ml_signals_processed': 0,
            'arbitrage_opportunities_processed': 0,
            'successful_executions': 0,
            'total_volume_aud': Decimal('0'),
            'compliance_issues': []
        }
        
        try:
            # 1. Generate ML signals
            ml_signals = self.ml_engine.generate_signals(
                data={'BTC/AUD': market_data.get('btc_data')},  # Simplified
                macro_data=market_data.get('macro_data')
            )
            
            # 2. Detect arbitrage opportunities
            arbitrage_opportunities = await self.arbitrage_engine.scan_for_opportunities(
                price_data=market_data.get('price_data', {}),
                funding_data=market_data.get('funding_data'),
                balance_aud=portfolio_value_aud,
                symbols=['BTC/AUD', 'ETH/AUD']
            )
            
            # 3. Process ML signals
            if ml_signals:
                ml_results = await self.process_ml_signals(ml_signals, portfolio_value_aud)
                cycle_summary['ml_signals_processed'] = len(ml_signals)
                cycle_summary['successful_executions'] += len([r for r in ml_results if r.status == 'completed'])
                cycle_summary['total_volume_aud'] += sum(r.total_cost for r in ml_results if r.status == 'completed')
            
            # 4. Process arbitrage opportunities
            if arbitrage_opportunities:
                arb_results = await self.process_arbitrage_opportunities(arbitrage_opportunities, portfolio_value_aud)
                cycle_summary['arbitrage_opportunities_processed'] = len(arbitrage_opportunities)
                
                for buy_result, sell_result in arb_results:
                    if buy_result.status == 'completed':
                        cycle_summary['successful_executions'] += 1
                        cycle_summary['total_volume_aud'] += buy_result.total_cost
                    if sell_result and sell_result.status == 'completed':
                        cycle_summary['successful_executions'] += 1
                        cycle_summary['total_volume_aud'] += sell_result.total_cost
            
            # 5. Risk management review
            portfolio_state = await self.risk_controller.update_portfolio_state(
                current_prices=market_data.get('current_prices', {}),
                account_balance=portfolio_value_aud * Decimal('0.1')
            )
            
            risk_alerts = await self.risk_controller.assess_portfolio_risks(portfolio_state)
            if risk_alerts:
                emergency_alerts = [alert for alert in risk_alerts if alert.level.value == 'emergency']
                if emergency_alerts:
                    self.processing_enabled = False
                    cycle_summary['compliance_issues'].append("Trading halted due to emergency risk conditions")
            
            cycle_summary['cycle_duration_ms'] = (datetime.now() - cycle_start).total_seconds() * 1000
            cycle_summary['status'] = 'completed'
            
            logger.info(f"Trading cycle completed: {cycle_summary['successful_executions']} executions, "
                       f"${cycle_summary['total_volume_aud']:,.2f} AUD volume")
            
        except Exception as e:
            cycle_summary['status'] = 'failed'
            cycle_summary['error'] = str(e)
            logger.error(f"Trading cycle failed: {e}")
        
        return cycle_summary
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive status of trading engine integration"""
        
        return {
            'processing_enabled': self.processing_enabled,
            'available_exchanges': self.available_exchanges,
            'daily_volume_aud': float(self.compliance_executor.daily_volume_aud),
            'daily_volume_limit_aud': float(self.compliance_executor.daily_volume_limit),
            'active_ml_positions': len(self.ml_executor.active_ml_positions),
            'active_arbitrage_trades': len(self.arbitrage_executor.active_arbitrage_trades),
            'execution_history_count': len(self.compliance_executor.execution_history),
            'last_cycle_time': datetime.now(),
            'compliance_status': 'operational' if self.processing_enabled else 'halted'
        }

# Usage example
async def main():
    """Example usage of Australian trading engine integration"""
    
    # This would be initialized with actual trading bot components
    print("Australian Trading Engine Integration Example")
    
    # Initialize components (simplified for example)
    from ..australian_compliance.ato_integration import AustralianTaxCalculator
    from ..australian_compliance.regulatory_compliance import AustralianComplianceManager
    from ..ml_strategy_discovery.ml_engine import MLStrategyDiscoveryEngine
    from ..arbitrage_engine.arbitrage_detector import OpportunisticArbitrageEngine
    from ..risk_management.portfolio_risk_controller import PortfolioRiskController
    
    # Create example integration
    # integration = AustralianTradingEngineIntegration(...)
    
    # Example market data
    market_data = {
        'current_prices': {'BTC/AUD': Decimal('65000'), 'ETH/AUD': Decimal('2600')},
        'price_data': {'btcmarkets': {'BTC/AUD': Decimal('65000')}},
        'funding_data': {'BTC/USDT': Decimal('0.0001')},
        'macro_data': {'aud_usd_rate': Decimal('0.67')}
    }
    
    portfolio_value = Decimal('100000')  # $100k AUD
    
    print(f"Example trading cycle with ${portfolio_value:,} AUD portfolio")
    print("Integration would process ML signals and arbitrage opportunities")
    print("All trades would be validated for Australian compliance")
    print("Tax events would be automatically created for ATO reporting")

if __name__ == "__main__":
    asyncio.run(main())