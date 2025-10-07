"""
Signal Processing Manager
Coordinates between ML strategy signals, arbitrage opportunities, and risk management
Implements priority-based execution queue and signal conflict resolution
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque

from .australian_trading_engine import (
    AustralianTradeRequest, AustralianExecutionResult, ExecutionPriority, 
    TradeSource, AustralianTradingEngineIntegration
)
from ..ml_strategy_discovery.ml_engine import StrategySignal, StrategyType
from ..arbitrage_engine.arbitrage_detector import ArbitrageOpportunity, ArbitrageType
from ..risk_management.portfolio_risk_controller import PortfolioState

logger = logging.getLogger(__name__)

class SignalConflictType(Enum):
    """Types of signal conflicts"""
    OPPOSING_DIRECTIONS = "opposing_directions"  # Buy vs Sell same symbol
    EXCESSIVE_EXPOSURE = "excessive_exposure"    # Too much allocation to same symbol
    TIMING_CONFLICT = "timing_conflict"          # Signals too close in time
    RESOURCE_CONFLICT = "resource_conflict"      # Insufficient balance
    COMPLIANCE_CONFLICT = "compliance_conflict"  # Regulatory violations

class SignalStatus(Enum):
    """Status of signals in processing queue"""
    PENDING = "pending"
    PROCESSING = "processing"
    EXECUTED = "executed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CONFLICTED = "conflicted"

@dataclass
class ProcessingSignal:
    """Wrapper for signals with processing metadata"""
    # Signal data
    signal_id: str
    signal_type: str  # 'ml_strategy' or 'arbitrage'
    symbol: str
    priority: ExecutionPriority
    source: TradeSource
    
    # Signal objects
    ml_signal: Optional[StrategySignal] = None
    arbitrage_opportunity: Optional[ArbitrageOpportunity] = None
    
    # Processing metadata
    status: SignalStatus = SignalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    estimated_size_aud: Optional[Decimal] = None
    
    # Conflict resolution
    conflicts: List[str] = field(default_factory=list)
    conflict_resolution: Optional[str] = None
    
    # Execution tracking
    execution_attempts: int = 0
    last_attempt_at: Optional[datetime] = None
    execution_result: Optional[AustralianExecutionResult] = None
    
    def is_expired(self) -> bool:
        """Check if signal has expired"""
        return self.expires_at and datetime.now() > self.expires_at
    
    def get_signal_strength(self) -> float:
        """Get signal strength for prioritization"""
        if self.ml_signal:
            return abs(self.ml_signal.signal_strength)
        elif self.arbitrage_opportunity:
            return float(self.arbitrage_opportunity.net_profit_percentage) / 100
        return 0.0

class SignalConflictResolver:
    """
    Resolves conflicts between different trading signals
    Prioritizes based on Australian compliance, profitability, and risk
    """
    
    def __init__(self):
        self.resolution_rules = {
            SignalConflictType.OPPOSING_DIRECTIONS: self._resolve_opposing_directions,
            SignalConflictType.EXCESSIVE_EXPOSURE: self._resolve_excessive_exposure,
            SignalConflictType.TIMING_CONFLICT: self._resolve_timing_conflict,
            SignalConflictType.RESOURCE_CONFLICT: self._resolve_resource_conflict,
            SignalConflictType.COMPLIANCE_CONFLICT: self._resolve_compliance_conflict
        }
        
        # Conflict detection thresholds
        self.max_symbol_exposure_pct = Decimal('0.20')  # 20% max per symbol
        self.min_time_between_trades_minutes = 15       # 15 minutes between same symbol trades
        self.min_confidence_threshold = 0.6             # Minimum ML confidence
        
    def detect_conflicts(
        self,
        new_signals: List[ProcessingSignal],
        existing_signals: List[ProcessingSignal],
        portfolio_state: PortfolioState
    ) -> Dict[str, List[SignalConflictType]]:
        """Detect conflicts between new and existing signals"""
        
        conflicts = defaultdict(list)
        
        all_signals = existing_signals + new_signals
        
        # Group signals by symbol
        symbol_signals = defaultdict(list)
        for signal in all_signals:
            symbol_signals[signal.symbol].append(signal)
        
        # Detect conflicts for each symbol
        for symbol, signals in symbol_signals.items():
            signal_conflicts = self._detect_symbol_conflicts(signals, portfolio_state)
            for signal_id, conflict_types in signal_conflicts.items():
                conflicts[signal_id].extend(conflict_types)
        
        # Detect resource conflicts across all signals
        resource_conflicts = self._detect_resource_conflicts(all_signals, portfolio_state)
        for signal_id, conflict_types in resource_conflicts.items():
            conflicts[signal_id].extend(conflict_types)
        
        return dict(conflicts)
    
    def _detect_symbol_conflicts(
        self,
        signals: List[ProcessingSignal],
        portfolio_state: PortfolioState
    ) -> Dict[str, List[SignalConflictType]]:
        """Detect conflicts for a specific symbol"""
        
        conflicts = defaultdict(list)
        
        if len(signals) <= 1:
            return dict(conflicts)
        
        # Check for opposing directions
        buy_signals = [s for s in signals if self._get_signal_direction(s) == 'buy']
        sell_signals = [s for s in signals if self._get_signal_direction(s) == 'sell']
        
        if buy_signals and sell_signals:
            # Conflict between buy and sell signals
            for signal in buy_signals + sell_signals:
                conflicts[signal.signal_id].append(SignalConflictType.OPPOSING_DIRECTIONS)
        
        # Check for timing conflicts
        signals_by_time = sorted(signals, key=lambda s: s.created_at)
        for i in range(1, len(signals_by_time)):
            current = signals_by_time[i]
            previous = signals_by_time[i-1]
            
            time_diff = (current.created_at - previous.created_at).total_seconds() / 60
            if time_diff < self.min_time_between_trades_minutes:
                conflicts[current.signal_id].append(SignalConflictType.TIMING_CONFLICT)
        
        # Check for excessive exposure
        total_exposure_aud = sum(
            s.estimated_size_aud or Decimal('0') for s in signals 
            if s.status in [SignalStatus.PENDING, SignalStatus.PROCESSING]
        )
        
        portfolio_value = portfolio_state.total_value_aud
        exposure_pct = total_exposure_aud / portfolio_value if portfolio_value > 0 else Decimal('1')
        
        if exposure_pct > self.max_symbol_exposure_pct:
            for signal in signals:
                if signal.status in [SignalStatus.PENDING, SignalStatus.PROCESSING]:
                    conflicts[signal.signal_id].append(SignalConflictType.EXCESSIVE_EXPOSURE)
        
        return dict(conflicts)
    
    def _detect_resource_conflicts(
        self,
        signals: List[ProcessingSignal],
        portfolio_state: PortfolioState
    ) -> Dict[str, List[SignalConflictType]]:
        """Detect resource conflicts across all signals"""
        
        conflicts = defaultdict(list)
        
        # Calculate total required capital
        total_required_aud = sum(
            s.estimated_size_aud or Decimal('0') for s in signals
            if s.status in [SignalStatus.PENDING, SignalStatus.PROCESSING]
        )
        
        available_cash = portfolio_state.cash_balance_aud
        
        if total_required_aud > available_cash:
            # Insufficient capital - mark lower priority signals
            sorted_signals = sorted(
                [s for s in signals if s.status in [SignalStatus.PENDING, SignalStatus.PROCESSING]],
                key=lambda s: (s.priority.value, -s.get_signal_strength()),
                reverse=True
            )
            
            cumulative_required = Decimal('0')
            for signal in sorted_signals:
                cumulative_required += signal.estimated_size_aud or Decimal('0')
                if cumulative_required > available_cash:
                    conflicts[signal.signal_id].append(SignalConflictType.RESOURCE_CONFLICT)
        
        return dict(conflicts)
    
    def _get_signal_direction(self, signal: ProcessingSignal) -> str:
        """Get trade direction for signal"""
        if signal.ml_signal:
            return 'buy' if signal.ml_signal.signal_strength > 0 else 'sell'
        elif signal.arbitrage_opportunity:
            return 'buy'  # Arbitrage involves both buy and sell
        return 'unknown'
    
    def resolve_conflicts(
        self,
        conflicts: Dict[str, List[SignalConflictType]],
        signals: List[ProcessingSignal]
    ) -> Dict[str, str]:
        """Resolve detected conflicts and return resolution decisions"""
        
        resolutions = {}
        
        for signal_id, conflict_types in conflicts.items():
            signal = next((s for s in signals if s.signal_id == signal_id), None)
            if not signal:
                continue
            
            # Apply resolution rules in priority order
            resolution = None
            for conflict_type in conflict_types:
                if conflict_type in self.resolution_rules:
                    resolution = self.resolution_rules[conflict_type](signal, conflict_types, signals)
                    if resolution:
                        break
            
            resolutions[signal_id] = resolution or "defer"  # Default to defer
        
        return resolutions
    
    def _resolve_opposing_directions(
        self,
        signal: ProcessingSignal,
        conflicts: List[SignalConflictType],
        all_signals: List[ProcessingSignal]
    ) -> str:
        """Resolve opposing buy/sell signals for same symbol"""
        
        symbol_signals = [s for s in all_signals if s.symbol == signal.symbol]
        
        # Priority order:
        # 1. Arbitrage opportunities (time-sensitive)
        # 2. High-confidence ML signals
        # 3. Emergency risk management signals
        
        arbitrage_signals = [s for s in symbol_signals if s.source == TradeSource.ARBITRAGE]
        if arbitrage_signals and signal in arbitrage_signals:
            return "execute"  # Arbitrage has priority
        elif arbitrage_signals and signal not in arbitrage_signals:
            return "reject"   # Defer to arbitrage
        
        # Compare ML signal strengths
        ml_signals = [s for s in symbol_signals if s.source == TradeSource.ML_STRATEGY]
        if len(ml_signals) > 1:
            # Keep highest confidence signal
            best_signal = max(ml_signals, key=lambda s: s.get_signal_strength())
            return "execute" if signal == best_signal else "reject"
        
        return "defer"
    
    def _resolve_excessive_exposure(
        self,
        signal: ProcessingSignal,
        conflicts: List[SignalConflictType],
        all_signals: List[ProcessingSignal]
    ) -> str:
        """Resolve excessive exposure to single symbol"""
        
        symbol_signals = [
            s for s in all_signals 
            if s.symbol == signal.symbol and s.status in [SignalStatus.PENDING, SignalStatus.PROCESSING]
        ]
        
        # Prioritize by signal strength and priority
        sorted_signals = sorted(
            symbol_signals,
            key=lambda s: (s.priority.value, s.get_signal_strength()),
            reverse=True
        )
        
        # Reduce position sizes to fit within exposure limits
        cumulative_exposure = Decimal('0')
        max_exposure = signal.estimated_size_aud * Decimal('3')  # Allow 3x original allocation
        
        for i, s in enumerate(sorted_signals):
            if cumulative_exposure + (s.estimated_size_aud or Decimal('0')) <= max_exposure:
                cumulative_exposure += s.estimated_size_aud or Decimal('0')
                if s == signal:
                    return "execute"
            else:
                if s == signal:
                    return "reduce_size"  # Reduce position size
        
        return "reject"
    
    def _resolve_timing_conflict(
        self,
        signal: ProcessingSignal,
        conflicts: List[SignalConflictType],
        all_signals: List[ProcessingSignal]
    ) -> str:
        """Resolve timing conflicts between signals"""
        
        # For timing conflicts, prefer:
        # 1. Higher priority signals
        # 2. Stronger signals
        # 3. More recent signals (for ML updates)
        
        symbol_signals = [s for s in all_signals if s.symbol == signal.symbol]
        recent_signals = [
            s for s in symbol_signals 
            if (datetime.now() - s.created_at).total_seconds() < self.min_time_between_trades_minutes * 60
        ]
        
        if len(recent_signals) <= 1:
            return "execute"
        
        # Find the best signal in the timing window
        best_signal = max(
            recent_signals,
            key=lambda s: (s.priority.value, s.get_signal_strength(), s.created_at.timestamp())
        )
        
        return "execute" if signal == best_signal else "defer"
    
    def _resolve_resource_conflict(
        self,
        signal: ProcessingSignal,
        conflicts: List[SignalConflictType],
        all_signals: List[ProcessingSignal]
    ) -> str:
        """Resolve resource conflicts due to insufficient capital"""
        
        # Prioritize by:
        # 1. Execution priority
        # 2. Signal strength
        # 3. Australian compliance considerations
        
        competing_signals = [
            s for s in all_signals 
            if s.status in [SignalStatus.PENDING, SignalStatus.PROCESSING]
        ]
        
        # Sort by priority and strength
        sorted_signals = sorted(
            competing_signals,
            key=lambda s: (
                s.priority.value,
                s.get_signal_strength(),
                1 if s.source == TradeSource.ARBITRAGE else 0  # Arbitrage bonus
            ),
            reverse=True
        )
        
        # Find signal's position in priority queue
        signal_rank = next((i for i, s in enumerate(sorted_signals) if s == signal), len(sorted_signals))
        
        # Allow top 50% of signals by priority
        cutoff = len(sorted_signals) // 2
        
        if signal_rank <= cutoff:
            return "execute"
        else:
            return "defer"
    
    def _resolve_compliance_conflict(
        self,
        signal: ProcessingSignal,
        conflicts: List[SignalConflictType],
        all_signals: List[ProcessingSignal]
    ) -> str:
        """Resolve compliance conflicts (always prioritize compliance)"""
        return "reject"  # Compliance violations are always rejected

class SignalProcessingManager:
    """
    Manages the processing queue and coordination of ML and arbitrage signals
    Handles conflict resolution, priority scheduling, and execution coordination
    """
    
    def __init__(self, trading_engine: AustralianTradingEngineIntegration):
        self.trading_engine = trading_engine
        self.conflict_resolver = SignalConflictResolver()
        
        # Processing queues
        self.processing_queue = deque()  # Main processing queue
        self.executed_signals = []       # History of executed signals
        self.rejected_signals = []       # History of rejected signals
        
        # Signal tracking
        self.active_signals_by_symbol = defaultdict(list)
        self.signal_id_counter = 0
        
        # Processing configuration
        self.max_concurrent_executions = 3
        self.processing_interval_seconds = 10
        self.signal_expiry_minutes = 30
        
        # Processing state
        self.is_processing = False
        self.last_processing_cycle = None
        
        logger.info("Initialized Signal Processing Manager")
    
    def _generate_signal_id(self) -> str:
        """Generate unique signal ID"""
        self.signal_id_counter += 1
        return f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.signal_id_counter:04d}"
    
    def add_ml_signals(
        self,
        ml_signals: List[StrategySignal],
        estimated_sizes: Dict[str, Decimal]
    ) -> List[str]:
        """Add ML strategy signals to processing queue"""
        
        added_signal_ids = []
        
        for ml_signal in ml_signals:
            # Create processing signal
            processing_signal = ProcessingSignal(
                signal_id=self._generate_signal_id(),
                signal_type='ml_strategy',
                symbol=ml_signal.symbol,
                priority=ExecutionPriority.HIGH if abs(ml_signal.signal_strength) > 0.7 else ExecutionPriority.MEDIUM,
                source=TradeSource.ML_STRATEGY,
                ml_signal=ml_signal,
                expires_at=datetime.now() + timedelta(minutes=self.signal_expiry_minutes),
                estimated_size_aud=estimated_sizes.get(ml_signal.symbol, Decimal('1000'))
            )
            
            # Add to queue
            self.processing_queue.append(processing_signal)
            self.active_signals_by_symbol[ml_signal.symbol].append(processing_signal)
            added_signal_ids.append(processing_signal.signal_id)
            
            logger.info(f"Added ML signal {processing_signal.signal_id} for {ml_signal.symbol}: "
                       f"strength={ml_signal.signal_strength:.3f}, confidence={ml_signal.confidence:.3f}")
        
        return added_signal_ids
    
    def add_arbitrage_opportunities(
        self,
        arbitrage_opportunities: List[ArbitrageOpportunity],
        estimated_sizes: Dict[str, Decimal]
    ) -> List[str]:
        """Add arbitrage opportunities to processing queue"""
        
        added_signal_ids = []
        
        for opportunity in arbitrage_opportunities:
            # Create processing signal
            processing_signal = ProcessingSignal(
                signal_id=self._generate_signal_id(),
                signal_type='arbitrage',
                symbol=opportunity.symbol,
                priority=ExecutionPriority.HIGH,  # Arbitrage is always high priority
                source=TradeSource.ARBITRAGE,
                arbitrage_opportunity=opportunity,
                expires_at=opportunity.expires_at,
                estimated_size_aud=estimated_sizes.get(opportunity.symbol, Decimal('5000'))
            )
            
            # Add to queue
            self.processing_queue.append(processing_signal)
            self.active_signals_by_symbol[opportunity.symbol].append(processing_signal)
            added_signal_ids.append(processing_signal.signal_id)
            
            logger.info(f"Added arbitrage signal {processing_signal.signal_id} for {opportunity.symbol}: "
                       f"profit={opportunity.net_profit_percentage:.2f}%, expires in "
                       f"{(opportunity.expires_at - datetime.now()).total_seconds():.0f}s")
        
        return added_signal_ids
    
    async def process_signal_queue(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """Process pending signals in the queue"""
        
        if self.is_processing:
            return {'status': 'already_processing'}
        
        self.is_processing = True
        self.last_processing_cycle = datetime.now()
        
        processing_summary = {
            'cycle_start': self.last_processing_cycle,
            'signals_processed': 0,
            'signals_executed': 0,
            'signals_rejected': 0,
            'signals_deferred': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'execution_errors': 0
        }
        
        try:
            # Clean up expired signals
            self._cleanup_expired_signals()
            
            if not self.processing_queue:
                return {'status': 'no_signals', 'summary': processing_summary}
            
            # Get pending signals
            pending_signals = [s for s in self.processing_queue if s.status == SignalStatus.PENDING]
            
            if not pending_signals:
                return {'status': 'no_pending_signals', 'summary': processing_summary}
            
            processing_summary['signals_processed'] = len(pending_signals)
            
            # Detect conflicts
            existing_signals = [s for s in self.processing_queue if s.status != SignalStatus.PENDING]
            conflicts = self.conflict_resolver.detect_conflicts(
                pending_signals, existing_signals, portfolio_state
            )
            
            processing_summary['conflicts_detected'] = len(conflicts)
            
            # Resolve conflicts
            resolutions = self.conflict_resolver.resolve_conflicts(conflicts, pending_signals)
            processing_summary['conflicts_resolved'] = len(resolutions)
            
            # Process signals based on resolutions
            execution_tasks = []
            
            for signal in pending_signals:
                resolution = resolutions.get(signal.signal_id, "execute")
                
                if resolution == "execute":
                    signal.status = SignalStatus.PROCESSING
                    task = self._execute_signal(signal, portfolio_state)
                    execution_tasks.append(task)
                    
                elif resolution == "reject":
                    signal.status = SignalStatus.REJECTED
                    signal.conflict_resolution = resolution
                    self.rejected_signals.append(signal)
                    processing_summary['signals_rejected'] += 1
                    
                elif resolution == "defer":
                    # Keep as pending for next cycle
                    processing_summary['signals_deferred'] += 1
                    
                elif resolution == "reduce_size":
                    # Reduce position size and execute
                    if signal.estimated_size_aud:
                        signal.estimated_size_aud *= Decimal('0.5')  # Reduce by 50%
                    signal.status = SignalStatus.PROCESSING
                    task = self._execute_signal(signal, portfolio_state)
                    execution_tasks.append(task)
            
            # Execute signals concurrently (up to limit)
            if execution_tasks:
                # Limit concurrent executions
                execution_batches = [
                    execution_tasks[i:i + self.max_concurrent_executions]
                    for i in range(0, len(execution_tasks), self.max_concurrent_executions)
                ]
                
                for batch in execution_batches:
                    results = await asyncio.gather(*batch, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, Exception):
                            processing_summary['execution_errors'] += 1
                            logger.error(f"Signal execution error: {result}")
                        elif result and result.get('status') == 'executed':
                            processing_summary['signals_executed'] += 1
            
            processing_summary['cycle_duration_ms'] = (datetime.now() - self.last_processing_cycle).total_seconds() * 1000
            processing_summary['status'] = 'completed'
            
            logger.info(f"Signal processing cycle completed: {processing_summary['signals_executed']} executed, "
                       f"{processing_summary['signals_rejected']} rejected, {processing_summary['signals_deferred']} deferred")
            
        except Exception as e:
            processing_summary['status'] = 'error'
            processing_summary['error'] = str(e)
            logger.error(f"Signal processing cycle failed: {e}")
        
        finally:
            self.is_processing = False
        
        return {'status': 'completed', 'summary': processing_summary}
    
    def _cleanup_expired_signals(self):
        """Remove expired signals from queue"""
        
        current_time = datetime.now()
        expired_signals = []
        
        # Find expired signals
        for signal in list(self.processing_queue):
            if signal.is_expired():
                signal.status = SignalStatus.EXPIRED
                expired_signals.append(signal)
                self.processing_queue.remove(signal)
                
                # Remove from symbol tracking
                if signal in self.active_signals_by_symbol[signal.symbol]:
                    self.active_signals_by_symbol[signal.symbol].remove(signal)
        
        if expired_signals:
            logger.info(f"Cleaned up {len(expired_signals)} expired signals")
    
    async def _execute_signal(
        self,
        signal: ProcessingSignal,
        portfolio_state: PortfolioState
    ) -> Dict[str, Any]:
        """Execute individual signal"""
        
        try:
            signal.execution_attempts += 1
            signal.last_attempt_at = datetime.now()
            
            # Execute based on signal type
            if signal.ml_signal:
                result = await self.trading_engine.ml_executor.execute_ml_signal(
                    signal=signal.ml_signal,
                    recommended_size_aud=signal.estimated_size_aud,
                    available_exchanges=self.trading_engine.available_exchanges
                )
                
            elif signal.arbitrage_opportunity:
                buy_result, sell_result = await self.trading_engine.arbitrage_executor.execute_arbitrage_opportunity(
                    opportunity=signal.arbitrage_opportunity,
                    position_size=signal.estimated_size_aud,
                    available_exchanges=self.trading_engine.available_exchanges
                )
                # Use buy result as primary result
                result = buy_result
                
            else:
                raise ValueError(f"Unknown signal type: {signal.signal_type}")
            
            # Update signal status
            signal.execution_result = result
            
            if result.status == 'completed':
                signal.status = SignalStatus.EXECUTED
                self.executed_signals.append(signal)
                
                logger.info(f"Signal {signal.signal_id} executed successfully: "
                           f"{result.executed_amount} {signal.symbol} at ${result.executed_price}")
                
                return {'status': 'executed', 'signal_id': signal.signal_id, 'result': result}
                
            else:
                signal.status = SignalStatus.REJECTED
                self.rejected_signals.append(signal)
                
                logger.warning(f"Signal {signal.signal_id} execution failed: {result.error_message}")
                
                return {'status': 'failed', 'signal_id': signal.signal_id, 'error': result.error_message}
        
        except Exception as e:
            signal.status = SignalStatus.REJECTED
            self.rejected_signals.append(signal)
            
            logger.error(f"Signal {signal.signal_id} execution error: {e}")
            
            return {'status': 'error', 'signal_id': signal.signal_id, 'error': str(e)}
        
        finally:
            # Remove from active tracking
            if signal in self.active_signals_by_symbol[signal.symbol]:
                self.active_signals_by_symbol[signal.symbol].remove(signal)
            
            # Remove from processing queue
            if signal in self.processing_queue:
                self.processing_queue.remove(signal)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current status of signal processing queue"""
        
        status_counts = defaultdict(int)
        for signal in self.processing_queue:
            status_counts[signal.status.value] += 1
        
        symbol_counts = defaultdict(int)
        for signal in self.processing_queue:
            symbol_counts[signal.symbol] += 1
        
        return {
            'queue_length': len(self.processing_queue),
            'status_breakdown': dict(status_counts),
            'symbol_breakdown': dict(symbol_counts),
            'is_processing': self.is_processing,
            'last_processing_cycle': self.last_processing_cycle,
            'total_executed': len(self.executed_signals),
            'total_rejected': len(self.rejected_signals),
            'processing_interval_seconds': self.processing_interval_seconds,
            'max_concurrent_executions': self.max_concurrent_executions
        }
    
    def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        
        history = []
        
        # Recent executed signals
        for signal in self.executed_signals[-limit:]:
            history.append({
                'signal_id': signal.signal_id,
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'status': 'executed',
                'execution_time': signal.last_attempt_at,
                'execution_result': {
                    'status': signal.execution_result.status if signal.execution_result else None,
                    'executed_amount': float(signal.execution_result.executed_amount) if signal.execution_result else None,
                    'total_cost': float(signal.execution_result.total_cost) if signal.execution_result else None
                } if signal.execution_result else None
            })
        
        # Recent rejected signals
        for signal in self.rejected_signals[-limit:]:
            history.append({
                'signal_id': signal.signal_id,
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'status': 'rejected',
                'execution_time': signal.last_attempt_at,
                'conflicts': signal.conflicts,
                'resolution': signal.conflict_resolution
            })
        
        # Sort by execution time
        history.sort(key=lambda x: x['execution_time'] or datetime.min, reverse=True)
        
        return history[:limit]

# Usage example
async def main():
    """Example usage of signal processing manager"""
    
    print("Signal Processing Manager Example")
    
    # This would be initialized with actual trading engine
    # signal_manager = SignalProcessingManager(trading_engine)
    
    # Example of adding signals and processing
    print("Would coordinate ML signals and arbitrage opportunities")
    print("Handles conflict resolution and priority-based execution")
    print("Maintains Australian compliance throughout processing")

if __name__ == "__main__":
    asyncio.run(main())