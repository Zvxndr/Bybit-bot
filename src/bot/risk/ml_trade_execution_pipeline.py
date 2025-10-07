"""
ML Trade Execution Pipeline with Integrated Risk Management

This module provides a comprehensive trade execution pipeline that integrates
ML predictions with robust risk management. Every trade passes through multiple
validation layers before execution.

Key Features:
- Pre-execution risk validation for all ML signals
- Dynamic position sizing based on ML confidence
- Circuit breaker integration
- Emergency stop functionality
- Comprehensive audit trail
- Real-time risk monitoring
- Explainable AI integration for risk assessment

Architecture:
    ML Signal → Risk Validation → Position Sizing → Execution Planning → Trade Execution → Monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json

# Import risk management components
from .ml_risk_manager import (
    MLRiskManager, TradeValidationResult, MLTradeRisk, 
    TradeBlockReason, CircuitBreakerType, EmergencyStopStatus
)
from .core.unified_risk_manager import UnifiedRiskManager

# Import ML integration components
try:
    from ..integration.ml_integration_controller import MLIntegrationController
    from ..integration.ml_feature_pipeline import MLFeatures, CombinedSignal
    from ..integration.ml_strategy_orchestrator import MLStrategyOrchestrator
except ImportError:
    logger.warning("ML integration components not available")
    MLIntegrationController = None

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class ExecutionStatus(Enum):
    """Trade execution status"""
    PENDING_VALIDATION = "pending_validation"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    PENDING_EXECUTION = "pending_execution"
    EXECUTING = "executing"
    EXECUTED_SUCCESSFULLY = "executed_successfully"
    EXECUTION_FAILED = "execution_failed"
    CANCELLED = "cancelled"
    EMERGENCY_STOPPED = "emergency_stopped"

class ExecutionPriority(Enum):
    """Trade execution priority levels"""
    LOW = "low"           # Long-term position adjustments
    NORMAL = "normal"     # Standard ML signals
    HIGH = "high"         # High-confidence ML signals
    URGENT = "urgent"     # Risk management actions
    EMERGENCY = "emergency"  # Emergency stop/liquidation

@dataclass
class MLTradeRequest:
    """ML-generated trade request"""
    request_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    signal_data: Dict[str, Any]
    ml_predictions: Dict[str, Any]
    market_data: Dict[str, Any]
    priority: ExecutionPriority
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionPlan:
    """Detailed execution plan for a validated trade"""
    trade_request: MLTradeRequest
    validation_result: TradeValidationResult
    execution_strategy: str  # 'immediate', 'twap', 'vwap', 'iceberg'
    order_params: Dict[str, Any]
    risk_monitoring_params: Dict[str, Any]
    expected_execution_time: timedelta
    backup_plans: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ExecutionResult:
    """Result of trade execution"""
    request_id: str
    symbol: str
    status: ExecutionStatus
    executed_size: Decimal
    executed_price: Optional[Decimal]
    execution_time: Optional[datetime]
    total_fees: Decimal
    slippage: Optional[float]
    execution_details: Dict[str, Any]
    error_message: Optional[str] = None
    risk_metrics: Optional[Dict[str, Any]] = None

@dataclass
class RiskMonitoringAlert:
    """Real-time risk monitoring alert"""
    alert_id: str
    symbol: str
    alert_type: str
    severity: str  # 'info', 'warning', 'critical', 'emergency'
    message: str
    current_value: float
    threshold: float
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)

# ============================================================================
# ML TRADE EXECUTION PIPELINE
# ============================================================================

class MLTradeExecutionPipeline:
    """
    Comprehensive ML Trade Execution Pipeline with Integrated Risk Management
    
    This pipeline ensures that every ML-generated trade signal passes through
    rigorous risk validation before execution, with continuous monitoring and
    circuit breaker protection.
    """
    
    def __init__(self, 
                 ml_risk_manager: MLRiskManager,
                 unified_risk_manager: UnifiedRiskManager,
                 exchange_client = None,  # Will be injected
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the ML Trade Execution Pipeline"""
        
        self.ml_risk_manager = ml_risk_manager
        self.unified_risk_manager = unified_risk_manager
        self.exchange_client = exchange_client
        self.config = config or self._get_default_config()
        
        # Execution state
        self.pending_requests: Dict[str, MLTradeRequest] = {}
        self.active_executions: Dict[str, ExecutionPlan] = {}
        self.execution_history: List[ExecutionResult] = []
        
        # Risk monitoring
        self.risk_alerts: List[RiskMonitoringAlert] = []
        self.position_monitors: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.execution_metrics: Dict[str, Any] = {
            'total_requests': 0,
            'successful_executions': 0,
            'blocked_trades': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'average_slippage': 0.0
        }
        
        # Circuit breaker callbacks
        self.circuit_breaker_callbacks: Dict[CircuitBreakerType, List[Callable]] = {}
        
        logger.info("ML Trade Execution Pipeline initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'execution': {
                'default_timeout_seconds': 30,
                'retry_attempts': 3,
                'retry_delay_seconds': 5,
                'enable_partial_fills': True,
                'max_slippage_tolerance': 0.005  # 0.5%
            },
            'risk_monitoring': {
                'position_check_interval': 10,  # seconds
                'alert_cooldown_seconds': 300,   # 5 minutes
                'max_concurrent_positions': 10
            },
            'circuit_breakers': {
                'enable_auto_recovery': True,
                'recovery_check_interval': 60,  # seconds
                'max_consecutive_failures': 5
            }
        }
    
    async def submit_trade_request(self, trade_request: MLTradeRequest) -> str:
        """
        Submit an ML trade request for processing
        
        This is the main entry point for all ML-generated trade signals.
        """
        logger.info(f"Received ML trade request: {trade_request.request_id} for {trade_request.symbol}")
        
        # Validate request
        if not self._validate_trade_request(trade_request):
            logger.error(f"Invalid trade request: {trade_request.request_id}")
            return ""
        
        # Check if system is operational
        if not await self._is_system_operational():
            logger.warning(f"System not operational, rejecting trade request: {trade_request.request_id}")
            return ""
        
        # Add to pending queue
        self.pending_requests[trade_request.request_id] = trade_request
        self.execution_metrics['total_requests'] += 1
        
        # Process asynchronously
        asyncio.create_task(self._process_trade_request(trade_request))
        
        return trade_request.request_id
    
    async def _process_trade_request(self, trade_request: MLTradeRequest):
        """Process a trade request through the complete pipeline"""
        
        try:
            # Step 1: Risk Validation
            logger.info(f"Validating trade request: {trade_request.request_id}")
            validation_result = await self.ml_risk_manager.validate_trade(
                symbol=trade_request.symbol,
                signal_data=trade_request.signal_data,
                market_data=trade_request.market_data,
                ml_predictions=trade_request.ml_predictions
            )
            
            if not validation_result.is_approved:
                await self._handle_blocked_trade(trade_request, validation_result)
                return
            
            # Step 2: Create Execution Plan
            logger.info(f"Creating execution plan: {trade_request.request_id}")
            execution_plan = await self._create_execution_plan(trade_request, validation_result)
            
            # Step 3: Execute Trade
            logger.info(f"Executing trade: {trade_request.request_id}")
            execution_result = await self._execute_trade(execution_plan)
            
            # Step 4: Post-Execution Monitoring
            if execution_result.status == ExecutionStatus.EXECUTED_SUCCESSFULLY:
                await self._start_position_monitoring(execution_result)
            
        except Exception as e:
            logger.error(f"Error processing trade request {trade_request.request_id}: {e}")
            await self._handle_execution_error(trade_request, str(e))
        
        finally:
            # Cleanup
            if trade_request.request_id in self.pending_requests:
                del self.pending_requests[trade_request.request_id]
    
    async def _create_execution_plan(self, trade_request: MLTradeRequest,
                                   validation_result: TradeValidationResult) -> ExecutionPlan:
        """Create detailed execution plan for validated trade"""
        
        # Determine execution strategy based on trade characteristics
        execution_strategy = self._determine_execution_strategy(
            trade_request, validation_result
        )
        
        # Create order parameters
        order_params = self._create_order_parameters(
            trade_request, validation_result, execution_strategy
        )
        
        # Risk monitoring parameters
        risk_monitoring_params = self._create_risk_monitoring_params(
            trade_request, validation_result
        )
        
        # Estimate execution time
        expected_execution_time = self._estimate_execution_time(
            trade_request, execution_strategy
        )
        
        # Create backup plans
        backup_plans = self._create_backup_plans(trade_request, validation_result)
        
        execution_plan = ExecutionPlan(
            trade_request=trade_request,
            validation_result=validation_result,
            execution_strategy=execution_strategy,
            order_params=order_params,
            risk_monitoring_params=risk_monitoring_params,
            expected_execution_time=expected_execution_time,
            backup_plans=backup_plans
        )
        
        # Store active execution
        self.active_executions[trade_request.request_id] = execution_plan
        
        return execution_plan
    
    def _determine_execution_strategy(self, trade_request: MLTradeRequest,
                                    validation_result: TradeValidationResult) -> str:
        """Determine optimal execution strategy"""
        
        ml_confidence = validation_result.risk_assessment.ml_confidence
        position_size = validation_result.final_position_size
        market_data = trade_request.market_data
        
        # High confidence + large size = VWAP to minimize market impact
        if ml_confidence > 0.8 and float(position_size) > 10000:
            return 'vwap'
        
        # High confidence + normal size = immediate execution
        elif ml_confidence > 0.8:
            return 'immediate'
        
        # Medium confidence = TWAP for gradual execution
        elif ml_confidence > 0.6:
            return 'twap'
        
        # Low confidence = iceberg to hide intentions
        else:
            return 'iceberg'
    
    def _create_order_parameters(self, trade_request: MLTradeRequest,
                               validation_result: TradeValidationResult,
                               execution_strategy: str) -> Dict[str, Any]:
        """Create exchange-specific order parameters"""
        
        base_params = {
            'symbol': trade_request.symbol,
            'side': trade_request.side,
            'quantity': str(validation_result.final_position_size),
            'type': 'limit',  # Default to limit orders for better control
            'timeInForce': 'GTC'
        }
        
        # Add strategy-specific parameters
        if execution_strategy == 'immediate':
            base_params.update({
                'type': 'market',
                'timeInForce': 'IOC'
            })
        elif execution_strategy == 'vwap':
            base_params.update({
                'type': 'limit',
                'postOnly': True,
                'executionInstructions': 'VWAP'
            })
        elif execution_strategy == 'twap':
            base_params.update({
                'type': 'limit',
                'timeInForce': 'GTD',
                'executionInstructions': 'TWAP'
            })
        elif execution_strategy == 'iceberg':
            base_params.update({
                'type': 'limit',
                'timeInForce': 'GTC',
                'icebergQty': str(float(validation_result.final_position_size) * 0.1)  # 10% visible
            })
        
        # Add ML-specific metadata
        base_params['metadata'] = {
            'ml_confidence': validation_result.risk_assessment.ml_confidence,
            'risk_level': validation_result.risk_assessment.overall_ml_risk.value,
            'validation_timestamp': validation_result.validation_timestamp.isoformat(),
            'request_id': trade_request.request_id
        }
        
        return base_params
    
    def _create_risk_monitoring_params(self, trade_request: MLTradeRequest,
                                     validation_result: TradeValidationResult) -> Dict[str, Any]:
        """Create risk monitoring parameters for the position"""
        
        return {
            'stop_loss_percentage': 0.05,  # 5% stop loss
            'take_profit_percentage': 0.15,  # 15% take profit
            'max_holding_period': timedelta(hours=24),  # Maximum holding period
            'position_check_interval': timedelta(minutes=5),
            'risk_alerts': {
                'drawdown_threshold': 0.03,  # 3% drawdown alert
                'volatility_spike_threshold': 2.0,  # 2x normal volatility
                'correlation_break_threshold': 0.3  # Correlation drops below 0.3
            },
            'auto_exit_conditions': {
                'emergency_stop_active': True,
                'circuit_breaker_active': True,
                'confidence_degrades_below': 0.3
            }
        }
    
    def _estimate_execution_time(self, trade_request: MLTradeRequest,
                               execution_strategy: str) -> timedelta:
        """Estimate execution time based on strategy and market conditions"""
        
        base_times = {
            'immediate': timedelta(seconds=5),
            'vwap': timedelta(minutes=30),
            'twap': timedelta(hours=1),
            'iceberg': timedelta(minutes=45)
        }
        
        base_time = base_times.get(execution_strategy, timedelta(minutes=15))
        
        # Adjust for market conditions
        volume = trade_request.market_data.get('volume', 1000000)
        if volume < 100000:  # Low volume market
            base_time *= 2
        
        volatility = trade_request.market_data.get('volatility', 0.02)
        if volatility > 0.05:  # High volatility
            base_time *= 1.5
        
        return base_time
    
    def _create_backup_plans(self, trade_request: MLTradeRequest,
                           validation_result: TradeValidationResult) -> List[Dict[str, Any]]:
        """Create backup execution plans"""
        
        backup_plans = []
        
        # Backup Plan 1: Reduce size if execution fails
        backup_plans.append({
            'name': 'reduce_size',
            'condition': 'execution_timeout',
            'action': 'reduce_position_size',
            'parameters': {
                'size_reduction': 0.5,  # Reduce to 50%
                'max_attempts': 2
            }
        })
        
        # Backup Plan 2: Market order if limit order fails
        backup_plans.append({
            'name': 'market_order_fallback',
            'condition': 'limit_order_timeout',
            'action': 'switch_to_market_order',
            'parameters': {
                'max_slippage_tolerance': 0.01  # 1% max slippage
            }
        })
        
        # Backup Plan 3: Cancel if conditions deteriorate
        backup_plans.append({
            'name': 'cancel_on_risk_increase',
            'condition': 'risk_conditions_deteriorated',
            'action': 'cancel_order',
            'parameters': {
                'risk_threshold_multiplier': 1.5
            }
        })
        
        return backup_plans
    
    async def _execute_trade(self, execution_plan: ExecutionPlan) -> ExecutionResult:
        """Execute the trade according to the execution plan"""
        
        request_id = execution_plan.trade_request.request_id
        symbol = execution_plan.trade_request.symbol
        
        # Initialize execution result
        execution_result = ExecutionResult(
            request_id=request_id,
            symbol=symbol,
            status=ExecutionStatus.PENDING_EXECUTION,
            executed_size=Decimal('0'),
            executed_price=None,
            execution_time=None,
            total_fees=Decimal('0'),
            slippage=None,
            execution_details={}
        )
        
        try:
            # Update status
            execution_result.status = ExecutionStatus.EXECUTING
            
            # Execute based on strategy
            if execution_plan.execution_strategy == 'immediate':
                result = await self._execute_immediate(execution_plan)
            elif execution_plan.execution_strategy == 'vwap':
                result = await self._execute_vwap(execution_plan)
            elif execution_plan.execution_strategy == 'twap':
                result = await self._execute_twap(execution_plan)
            elif execution_plan.execution_strategy == 'iceberg':
                result = await self._execute_iceberg(execution_plan)
            else:
                raise ValueError(f"Unknown execution strategy: {execution_plan.execution_strategy}")
            
            # Update execution result
            execution_result.status = ExecutionStatus.EXECUTED_SUCCESSFULLY
            execution_result.executed_size = result['executed_size']
            execution_result.executed_price = result['executed_price']
            execution_result.execution_time = datetime.now()
            execution_result.total_fees = result['total_fees']
            execution_result.slippage = result['slippage']
            execution_result.execution_details = result['details']
            
            # Update metrics
            self.execution_metrics['successful_executions'] += 1
            
            logger.info(f"Trade executed successfully: {request_id} - "
                       f"Size: {execution_result.executed_size}, "
                       f"Price: {execution_result.executed_price}")
            
        except Exception as e:
            execution_result.status = ExecutionStatus.EXECUTION_FAILED
            execution_result.error_message = str(e)
            self.execution_metrics['failed_executions'] += 1
            
            logger.error(f"Trade execution failed: {request_id} - {e}")
        
        # Store result
        self.execution_history.append(execution_result)
        
        # Cleanup active execution
        if request_id in self.active_executions:
            del self.active_executions[request_id]
        
        return execution_result
    
    async def _execute_immediate(self, execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute trade immediately using market order"""
        
        # This would integrate with the actual exchange client
        # For now, return a mock result
        return {
            'executed_size': execution_plan.validation_result.final_position_size,
            'executed_price': Decimal('50000'),  # Mock price
            'total_fees': Decimal('25'),  # Mock fees
            'slippage': 0.001,  # 0.1% slippage
            'details': {
                'execution_strategy': 'immediate',
                'order_type': 'market',
                'fill_time': datetime.now().isoformat()
            }
        }
    
    async def _execute_vwap(self, execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute trade using VWAP strategy"""
        # Mock implementation
        return {
            'executed_size': execution_plan.validation_result.final_position_size,
            'executed_price': Decimal('49950'),  # Better price due to VWAP
            'total_fees': Decimal('20'),  # Lower fees
            'slippage': 0.0005,  # Lower slippage
            'details': {
                'execution_strategy': 'vwap',
                'execution_time_minutes': 30,
                'average_fill_price': '49950'
            }
        }
    
    async def _execute_twap(self, execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute trade using TWAP strategy"""
        # Mock implementation
        return {
            'executed_size': execution_plan.validation_result.final_position_size,
            'executed_price': Decimal('49975'),
            'total_fees': Decimal('22'),
            'slippage': 0.0007,
            'details': {
                'execution_strategy': 'twap',
                'execution_time_minutes': 60,
                'number_of_fills': 12
            }
        }
    
    async def _execute_iceberg(self, execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute trade using iceberg strategy"""
        # Mock implementation
        return {
            'executed_size': execution_plan.validation_result.final_position_size,
            'executed_price': Decimal('49960'),
            'total_fees': Decimal('18'),  # Lowest fees due to maker orders
            'slippage': 0.0003,  # Lowest slippage
            'details': {
                'execution_strategy': 'iceberg',
                'execution_time_minutes': 45,
                'number_of_tranches': 10,
                'average_visible_size': '10%'
            }
        }
    
    async def _start_position_monitoring(self, execution_result: ExecutionResult):
        """Start real-time monitoring of the executed position"""
        
        symbol = execution_result.symbol
        
        # Create position monitor
        monitor_config = {
            'symbol': symbol,
            'entry_price': execution_result.executed_price,
            'position_size': execution_result.executed_size,
            'entry_time': execution_result.execution_time,
            'stop_loss': None,  # Will be calculated
            'take_profit': None,  # Will be calculated
            'last_check': datetime.now(),
            'alerts_sent': []
        }
        
        self.position_monitors[execution_result.request_id] = monitor_config
        
        # Start monitoring task
        asyncio.create_task(self._monitor_position(execution_result.request_id))
        
        logger.info(f"Started position monitoring for {symbol}: {execution_result.request_id}")
    
    async def _monitor_position(self, request_id: str):
        """Monitor a position for risk and performance"""
        
        monitor_config = self.position_monitors.get(request_id)
        if not monitor_config:
            return
        
        symbol = monitor_config['symbol']
        check_interval = timedelta(minutes=5)
        
        try:
            while request_id in self.position_monitors:
                # Check current market conditions
                # In real implementation, this would fetch live market data
                current_price = Decimal('50100')  # Mock current price
                
                # Calculate P&L
                entry_price = monitor_config['entry_price']
                position_size = monitor_config['position_size']
                unrealized_pnl = (current_price - entry_price) * position_size
                
                # Check risk conditions
                risk_alerts = await self._check_position_risk(
                    request_id, current_price, unrealized_pnl
                )
                
                # Handle alerts
                for alert in risk_alerts:
                    await self._handle_risk_alert(alert)
                
                # Update monitor
                monitor_config['last_check'] = datetime.now()
                
                # Sleep until next check
                await asyncio.sleep(check_interval.total_seconds())
                
        except Exception as e:
            logger.error(f"Error monitoring position {request_id}: {e}")
        
        finally:
            # Cleanup
            if request_id in self.position_monitors:
                del self.position_monitors[request_id]
    
    async def _check_position_risk(self, request_id: str, current_price: Decimal,
                                 unrealized_pnl: Decimal) -> List[RiskMonitoringAlert]:
        """Check position for risk conditions"""
        
        alerts = []
        monitor_config = self.position_monitors[request_id]
        symbol = monitor_config['symbol']
        
        # Check drawdown
        entry_price = monitor_config['entry_price']
        drawdown_pct = float((entry_price - current_price) / entry_price)
        
        if drawdown_pct > 0.03:  # 3% drawdown
            alerts.append(RiskMonitoringAlert(
                alert_id=f"{request_id}_drawdown",
                symbol=symbol,
                alert_type="position_drawdown",
                severity="warning",
                message=f"Position drawdown: {drawdown_pct:.1%}",
                current_value=drawdown_pct,
                threshold=0.03,
                recommended_action="Consider reducing position size"
            ))
        
        # Add more risk checks...
        
        return alerts
    
    async def _handle_risk_alert(self, alert: RiskMonitoringAlert):
        """Handle a risk monitoring alert"""
        
        self.risk_alerts.append(alert)
        
        logger.warning(f"Risk Alert: {alert.alert_type} for {alert.symbol} - {alert.message}")
        
        # Implement automated responses for critical alerts
        if alert.severity == "emergency":
            await self._handle_emergency_alert(alert)
    
    async def _handle_emergency_alert(self, alert: RiskMonitoringAlert):
        """Handle emergency risk alerts"""
        
        logger.critical(f"EMERGENCY ALERT: {alert.message}")
        
        # Trigger emergency stop if configured
        if alert.alert_type in ["position_drawdown", "correlation_breakdown"]:
            await self.ml_risk_manager.activate_emergency_stop(
                reason=f"Emergency alert: {alert.message}",
                manual_override=False
            )
    
    async def _handle_blocked_trade(self, trade_request: MLTradeRequest,
                                  validation_result: TradeValidationResult):
        """Handle a trade that was blocked by risk management"""
        
        self.execution_metrics['blocked_trades'] += 1
        
        # Create execution result for blocked trade
        execution_result = ExecutionResult(
            request_id=trade_request.request_id,
            symbol=trade_request.symbol,
            status=ExecutionStatus.VALIDATION_FAILED,
            executed_size=Decimal('0'),
            executed_price=None,
            execution_time=None,
            total_fees=Decimal('0'),
            slippage=None,
            execution_details={
                'blocked_reasons': [r.value for r in validation_result.blocked_reasons],
                'risk_assessment': validation_result.risk_assessment.__dict__
            }
        )
        
        self.execution_history.append(execution_result)
        
        logger.warning(f"Trade blocked: {trade_request.request_id} - "
                      f"Reasons: {[r.value for r in validation_result.blocked_reasons]}")
    
    async def _handle_execution_error(self, trade_request: MLTradeRequest, error_message: str):
        """Handle execution errors"""
        
        execution_result = ExecutionResult(
            request_id=trade_request.request_id,
            symbol=trade_request.symbol,
            status=ExecutionStatus.EXECUTION_FAILED,
            executed_size=Decimal('0'),
            executed_price=None,
            execution_time=None,
            total_fees=Decimal('0'),
            slippage=None,
            execution_details={},
            error_message=error_message
        )
        
        self.execution_history.append(execution_result)
        self.execution_metrics['failed_executions'] += 1
    
    def _validate_trade_request(self, trade_request: MLTradeRequest) -> bool:
        """Validate incoming trade request"""
        
        required_fields = ['request_id', 'symbol', 'side', 'signal_data', 'ml_predictions']
        
        for field in required_fields:
            if not hasattr(trade_request, field) or getattr(trade_request, field) is None:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate side
        if trade_request.side not in ['buy', 'sell']:
            logger.error(f"Invalid side: {trade_request.side}")
            return False
        
        # Check expiration
        if trade_request.expires_at and datetime.now() > trade_request.expires_at:
            logger.error(f"Trade request expired: {trade_request.request_id}")
            return False
        
        return True
    
    async def _is_system_operational(self) -> bool:
        """Check if the trading system is operational"""
        
        # Check emergency stop
        if self.ml_risk_manager.emergency_stop.is_active:
            return False
        
        # Check critical circuit breakers
        critical_breakers = [
            CircuitBreakerType.DAILY_LOSS_LIMIT,
            CircuitBreakerType.MODEL_PERFORMANCE_DEGRADED
        ]
        
        for breaker_type in critical_breakers:
            if self.ml_risk_manager.circuit_breakers[breaker_type].is_active:
                return False
        
        return True
    
    # ============================================================================
    # PUBLIC API METHODS
    # ============================================================================
    
    def get_execution_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status for a trade request"""
        
        # Check pending requests
        if request_id in self.pending_requests:
            return {
                'status': 'pending_validation',
                'request': self.pending_requests[request_id].__dict__
            }
        
        # Check active executions
        if request_id in self.active_executions:
            return {
                'status': 'executing',
                'execution_plan': self.active_executions[request_id].__dict__
            }
        
        # Check execution history
        for result in self.execution_history:
            if result.request_id == request_id:
                return {
                    'status': result.status.value,
                    'result': result.__dict__
                }
        
        return None
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        
        return {
            'execution_metrics': self.execution_metrics,
            'active_executions': len(self.active_executions),
            'pending_requests': len(self.pending_requests),
            'position_monitors': len(self.position_monitors),
            'recent_alerts': len([a for a in self.risk_alerts 
                                if a.timestamp > datetime.now() - timedelta(hours=1)]),
            'risk_system_status': self.ml_risk_manager.get_system_status()
        }
    
    def get_recent_alerts(self, limit: int = 50) -> List[RiskMonitoringAlert]:
        """Get recent risk monitoring alerts"""
        
        sorted_alerts = sorted(self.risk_alerts, key=lambda x: x.timestamp, reverse=True)
        return sorted_alerts[:limit]
    
    async def emergency_halt_all_trading(self, reason: str) -> bool:
        """Emergency halt all trading activities"""
        
        logger.critical(f"EMERGENCY HALT INITIATED: {reason}")
        
        # Activate emergency stop
        await self.ml_risk_manager.activate_emergency_stop(reason, manual_override=True)
        
        # Cancel all pending requests
        cancelled_count = 0
        for request_id in list(self.pending_requests.keys()):
            del self.pending_requests[request_id]
            cancelled_count += 1
        
        # Cancel active executions (in real implementation, would cancel exchange orders)
        for request_id in list(self.active_executions.keys()):
            del self.active_executions[request_id]
            cancelled_count += 1
        
        logger.critical(f"Emergency halt completed. Cancelled {cancelled_count} trades.")
        
        return True