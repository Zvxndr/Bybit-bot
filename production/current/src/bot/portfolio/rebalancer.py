"""
Advanced Portfolio Rebalancing Engine.
Provides intelligent rebalancing strategies with transaction cost optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class RebalanceStrategy(Enum):
    """Rebalancing strategies."""
    CALENDAR = "calendar"                    # Fixed time intervals
    THRESHOLD = "threshold"                  # Drift-based rebalancing
    VOLATILITY_TARGET = "volatility_target"  # Volatility targeting
    RISK_BUDGET = "risk_budget"             # Risk budget maintenance
    MOMENTUM = "momentum"                    # Momentum-based rebalancing
    ADAPTIVE = "adaptive"                    # Adaptive strategy selection
    COST_OPTIMIZED = "cost_optimized"       # Transaction cost optimized
    TACTICAL = "tactical"                    # Tactical asset allocation

class RebalanceTrigger(Enum):
    """Rebalancing triggers."""
    TIME_BASED = "time_based"
    DRIFT_BASED = "drift_based"
    VOLATILITY_BASED = "volatility_based"
    CORRELATION_BASED = "correlation_based"
    DRAWDOWN_BASED = "drawdown_based"
    MOMENTUM_BASED = "momentum_based"
    COMBINED = "combined"

@dataclass
class RebalanceSignal:
    """Rebalancing signal."""
    trigger: RebalanceTrigger
    strength: float  # 0-1 signal strength
    urgency: str     # low, medium, high
    reason: str
    timestamp: datetime
    assets_affected: List[str]
    recommended_action: str

@dataclass
class RebalanceTransaction:
    """Individual rebalancing transaction."""
    symbol: str
    action: str  # buy, sell
    current_weight: float
    target_weight: float
    weight_change: float
    transaction_amount: float
    estimated_cost: float
    priority: int
    execution_price: Optional[float] = None
    actual_cost: Optional[float] = None
    executed_at: Optional[datetime] = None

@dataclass
class RebalanceResult:
    """Rebalancing execution result."""
    strategy: RebalanceStrategy
    trigger: RebalanceTrigger
    execution_date: datetime
    portfolio_value: float
    transactions: List[RebalanceTransaction]
    total_transaction_costs: float
    cost_ratio: float  # costs / portfolio value
    expected_tracking_error_reduction: float
    risk_reduction: float
    diversification_improvement: float
    execution_summary: Dict[str, Any]
    performance_attribution: Dict[str, float]
    next_rebalance_date: Optional[datetime]
    rebalance_effectiveness: float  # 0-1 score
    market_impact_estimate: float

class PortfolioRebalancer:
    """Advanced portfolio rebalancing engine."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Rebalancing configuration
        self.rebalance_config = {
            'calendar_frequencies': {
                'daily': 1,
                'weekly': 7,
                'monthly': 30,
                'quarterly': 90,
                'semi_annual': 180,
                'annual': 365
            },
            'threshold_limits': {
                'minor': 0.05,      # 5% drift
                'moderate': 0.10,    # 10% drift
                'major': 0.20       # 20% drift
            },
            'volatility_targets': {
                'conservative': 0.10,  # 10% annual vol
                'moderate': 0.15,      # 15% annual vol
                'aggressive': 0.25     # 25% annual vol
            },
            'transaction_costs': {
                'fixed_cost': 0.0,      # Fixed cost per trade
                'proportional_cost': 0.001,  # 0.1% of trade value
                'market_impact': 0.0005      # 0.05% market impact
            },
            'min_trade_size': 0.01,     # Minimum 1% allocation change
            'max_positions': 20,        # Maximum positions
            'cash_buffer': 0.02,        # 2% cash buffer
            'rebalance_tolerance': 0.005  # 0.5% tolerance
        }
        
        # Rebalancing history
        self.rebalance_history = []
        self.pending_signals = []
        
        # Strategy weights for adaptive rebalancing
        self.strategy_weights = {
            RebalanceStrategy.CALENDAR: 0.3,
            RebalanceStrategy.THRESHOLD: 0.4,
            RebalanceStrategy.VOLATILITY_TARGET: 0.2,
            RebalanceStrategy.MOMENTUM: 0.1
        }
        
        self.logger.info("PortfolioRebalancer initialized")
    
    async def generate_rebalance_signals(self, 
                                       current_weights: Dict[str, float],
                                       target_weights: Dict[str, float],
                                       market_data: Dict[str, Any],
                                       portfolio_metrics: Dict[str, Any]) -> List[RebalanceSignal]:
        """Generate rebalancing signals based on multiple triggers."""
        signals = []
        
        try:
            # Time-based signals
            time_signals = await self._generate_time_based_signals(market_data)
            signals.extend(time_signals)
            
            # Drift-based signals
            drift_signals = await self._generate_drift_based_signals(
                current_weights, target_weights
            )
            signals.extend(drift_signals)
            
            # Volatility-based signals
            vol_signals = await self._generate_volatility_based_signals(
                portfolio_metrics, market_data
            )
            signals.extend(vol_signals)
            
            # Correlation-based signals
            corr_signals = await self._generate_correlation_based_signals(
                market_data.get('correlation_data')
            )
            signals.extend(corr_signals)
            
            # Drawdown-based signals
            dd_signals = await self._generate_drawdown_based_signals(
                portfolio_metrics
            )
            signals.extend(dd_signals)
            
            # Momentum-based signals
            momentum_signals = await self._generate_momentum_based_signals(
                market_data.get('price_data')
            )
            signals.extend(momentum_signals)
            
            # Filter and prioritize signals
            signals = await self._filter_and_prioritize_signals(signals)
            
            # Store pending signals
            self.pending_signals.extend(signals)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Failed to generate rebalance signals: {e}")
            return []
    
    async def execute_rebalancing(self, 
                                strategy: RebalanceStrategy,
                                current_weights: Dict[str, float],
                                target_weights: Dict[str, float],
                                portfolio_value: float,
                                market_data: Dict[str, Any],
                                signals: List[RebalanceSignal] = None) -> RebalanceResult:
        """Execute portfolio rebalancing using specified strategy."""
        try:
            # Validate inputs
            self._validate_rebalancing_inputs(current_weights, target_weights, portfolio_value)
            
            # Determine primary trigger
            primary_trigger = await self._determine_primary_trigger(signals)
            
            # Calculate optimal transactions
            transactions = await self._calculate_optimal_transactions(
                strategy, current_weights, target_weights, portfolio_value, market_data
            )
            
            # Estimate transaction costs
            total_costs = await self._estimate_transaction_costs(transactions, market_data)
            
            # Execute transactions (simulation)
            executed_transactions = await self._simulate_transaction_execution(
                transactions, market_data
            )
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_rebalance_performance(
                current_weights, target_weights, executed_transactions, portfolio_value
            )
            
            # Create result
            result = RebalanceResult(
                strategy=strategy,
                trigger=primary_trigger,
                execution_date=datetime.now(),
                portfolio_value=portfolio_value,
                transactions=executed_transactions,
                total_transaction_costs=total_costs,
                cost_ratio=total_costs / portfolio_value if portfolio_value > 0 else 0,
                expected_tracking_error_reduction=performance_metrics['tracking_error_reduction'],
                risk_reduction=performance_metrics['risk_reduction'],
                diversification_improvement=performance_metrics['diversification_improvement'],
                execution_summary=performance_metrics['execution_summary'],
                performance_attribution=performance_metrics['performance_attribution'],
                next_rebalance_date=await self._calculate_next_rebalance_date(strategy),
                rebalance_effectiveness=performance_metrics['effectiveness_score'],
                market_impact_estimate=performance_metrics['market_impact']
            )
            
            # Store result
            self._store_rebalance_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Rebalancing execution failed: {e}")
            raise
    
    async def _generate_time_based_signals(self, market_data: Dict[str, Any]) -> List[RebalanceSignal]:
        """Generate time-based rebalancing signals."""
        signals = []
        
        try:
            current_time = datetime.now()
            
            # Check if it's time for scheduled rebalancing
            if not self.rebalance_history:
                # First rebalancing
                signals.append(RebalanceSignal(
                    trigger=RebalanceTrigger.TIME_BASED,
                    strength=0.8,
                    urgency="medium",
                    reason="Initial portfolio rebalancing",
                    timestamp=current_time,
                    assets_affected=[],
                    recommended_action="full_rebalance"
                ))
            else:
                last_rebalance = self.rebalance_history[-1].execution_date
                days_since_rebalance = (current_time - last_rebalance).days
                
                # Monthly rebalancing signal
                if days_since_rebalance >= 30:
                    strength = min(1.0, days_since_rebalance / 30.0)
                    urgency = "high" if days_since_rebalance >= 60 else "medium"
                    
                    signals.append(RebalanceSignal(
                        trigger=RebalanceTrigger.TIME_BASED,
                        strength=strength,
                        urgency=urgency,
                        reason=f"Scheduled rebalancing ({days_since_rebalance} days since last)",
                        timestamp=current_time,
                        assets_affected=[],
                        recommended_action="calendar_rebalance"
                    ))
            
        except Exception as e:
            self.logger.error(f"Failed to generate time-based signals: {e}")
        
        return signals
    
    async def _generate_drift_based_signals(self, 
                                          current_weights: Dict[str, float],
                                          target_weights: Dict[str, float]) -> List[RebalanceSignal]:
        """Generate drift-based rebalancing signals."""
        signals = []
        
        try:
            drift_threshold = self.rebalance_config['threshold_limits']['moderate']
            affected_assets = []
            max_drift = 0.0
            
            for symbol in target_weights:
                current_weight = current_weights.get(symbol, 0.0)
                target_weight = target_weights[symbol]
                
                drift = abs(current_weight - target_weight)
                
                if drift > drift_threshold:
                    affected_assets.append(symbol)
                    max_drift = max(max_drift, drift)
            
            if affected_assets:
                strength = min(1.0, max_drift / self.rebalance_config['threshold_limits']['major'])
                urgency = "high" if max_drift > self.rebalance_config['threshold_limits']['major'] else "medium"
                
                signals.append(RebalanceSignal(
                    trigger=RebalanceTrigger.DRIFT_BASED,
                    strength=strength,
                    urgency=urgency,
                    reason=f"Weight drift detected (max: {max_drift:.1%})",
                    timestamp=datetime.now(),
                    assets_affected=affected_assets,
                    recommended_action="drift_correction"
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to generate drift-based signals: {e}")
        
        return signals
    
    async def _generate_volatility_based_signals(self, 
                                               portfolio_metrics: Dict[str, Any],
                                               market_data: Dict[str, Any]) -> List[RebalanceSignal]:
        """Generate volatility-based rebalancing signals."""
        signals = []
        
        try:
            current_vol = portfolio_metrics.get('volatility', 0.0)
            target_vol = self.rebalance_config['volatility_targets']['moderate']
            
            vol_deviation = abs(current_vol - target_vol) / target_vol if target_vol > 0 else 0
            
            if vol_deviation > 0.2:  # 20% deviation threshold
                strength = min(1.0, vol_deviation)
                urgency = "high" if vol_deviation > 0.5 else "medium"
                
                action = "reduce_risk" if current_vol > target_vol else "increase_risk"
                
                signals.append(RebalanceSignal(
                    trigger=RebalanceTrigger.VOLATILITY_BASED,
                    strength=strength,
                    urgency=urgency,
                    reason=f"Portfolio volatility deviation: {vol_deviation:.1%}",
                    timestamp=datetime.now(),
                    assets_affected=[],
                    recommended_action=action
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to generate volatility-based signals: {e}")
        
        return signals
    
    async def _generate_correlation_based_signals(self, 
                                                correlation_data: Dict[str, Any]) -> List[RebalanceSignal]:
        """Generate correlation-based rebalancing signals."""
        signals = []
        
        try:
            if not correlation_data:
                return signals
            
            avg_correlation = correlation_data.get('average_correlation', 0.0)
            correlation_regime = correlation_data.get('regime', 'moderate')
            
            # High correlation regime requires rebalancing
            if correlation_regime in ['high_correlation', 'crisis_correlation']:
                strength = 0.8 if correlation_regime == 'crisis_correlation' else 0.6
                
                signals.append(RebalanceSignal(
                    trigger=RebalanceTrigger.CORRELATION_BASED,
                    strength=strength,
                    urgency="high",
                    reason=f"High correlation regime detected: {correlation_regime}",
                    timestamp=datetime.now(),
                    assets_affected=[],
                    recommended_action="diversification_enhancement"
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to generate correlation-based signals: {e}")
        
        return signals
    
    async def _generate_drawdown_based_signals(self, 
                                             portfolio_metrics: Dict[str, Any]) -> List[RebalanceSignal]:
        """Generate drawdown-based rebalancing signals."""
        signals = []
        
        try:
            max_drawdown = portfolio_metrics.get('max_drawdown', 0.0)
            current_drawdown = portfolio_metrics.get('current_drawdown', 0.0)
            
            # Significant drawdown threshold
            if abs(current_drawdown) > 0.15:  # 15% drawdown
                strength = min(1.0, abs(current_drawdown) / 0.25)  # Scale to 25% max
                urgency = "high" if abs(current_drawdown) > 0.20 else "medium"
                
                signals.append(RebalanceSignal(
                    trigger=RebalanceTrigger.DRAWDOWN_BASED,
                    strength=strength,
                    urgency=urgency,
                    reason=f"Significant drawdown: {current_drawdown:.1%}",
                    timestamp=datetime.now(),
                    assets_affected=[],
                    recommended_action="risk_reduction"
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to generate drawdown-based signals: {e}")
        
        return signals
    
    async def _generate_momentum_based_signals(self, 
                                             price_data: pd.DataFrame) -> List[RebalanceSignal]:
        """Generate momentum-based rebalancing signals."""
        signals = []
        
        try:
            if price_data is None or price_data.empty:
                return signals
            
            # Calculate momentum for each asset
            momentum_scores = {}
            lookback_periods = [30, 60, 90]  # Short, medium, long-term momentum
            
            for symbol in price_data.columns:
                if symbol in price_data.columns:
                    prices = price_data[symbol].dropna()
                    
                    if len(prices) >= max(lookback_periods):
                        momentum_score = 0
                        for period in lookback_periods:
                            if len(prices) >= period:
                                period_return = (prices.iloc[-1] / prices.iloc[-period] - 1)
                                weight = 1.0 / period  # Recent periods get higher weight
                                momentum_score += period_return * weight
                        
                        momentum_scores[symbol] = momentum_score
            
            if momentum_scores:
                # Check for extreme momentum divergence
                momentum_values = list(momentum_scores.values())
                momentum_std = np.std(momentum_values)
                
                if momentum_std > 0.1:  # High momentum dispersion
                    top_performers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    bottom_performers = sorted(momentum_scores.items(), key=lambda x: x[1])[:3]
                    
                    affected_assets = [asset for asset, _ in top_performers + bottom_performers]
                    
                    signals.append(RebalanceSignal(
                        trigger=RebalanceTrigger.MOMENTUM_BASED,
                        strength=min(1.0, momentum_std / 0.2),
                        urgency="medium",
                        reason=f"Momentum divergence detected (std: {momentum_std:.2f})",
                        timestamp=datetime.now(),
                        assets_affected=affected_assets,
                        recommended_action="momentum_rebalance"
                    ))
            
        except Exception as e:
            self.logger.error(f"Failed to generate momentum-based signals: {e}")
        
        return signals
    
    async def _filter_and_prioritize_signals(self, signals: List[RebalanceSignal]) -> List[RebalanceSignal]:
        """Filter and prioritize rebalancing signals."""
        try:
            if not signals:
                return signals
            
            # Sort by urgency and strength
            urgency_order = {"high": 3, "medium": 2, "low": 1}
            
            sorted_signals = sorted(signals, key=lambda s: (
                urgency_order.get(s.urgency, 0),
                s.strength
            ), reverse=True)
            
            # Remove duplicate triggers (keep strongest)
            seen_triggers = set()
            filtered_signals = []
            
            for signal in sorted_signals:
                if signal.trigger not in seen_triggers:
                    filtered_signals.append(signal)
                    seen_triggers.add(signal.trigger)
            
            # Limit to top 5 signals
            return filtered_signals[:5]
            
        except Exception as e:
            self.logger.error(f"Failed to filter signals: {e}")
            return signals
    
    async def _determine_primary_trigger(self, signals: List[RebalanceSignal]) -> RebalanceTrigger:
        """Determine primary rebalancing trigger."""
        if not signals:
            return RebalanceTrigger.TIME_BASED
        
        # Return highest priority signal
        urgency_order = {"high": 3, "medium": 2, "low": 1}
        
        primary_signal = max(signals, key=lambda s: (
            urgency_order.get(s.urgency, 0),
            s.strength
        ))
        
        return primary_signal.trigger
    
    async def _calculate_optimal_transactions(self, 
                                            strategy: RebalanceStrategy,
                                            current_weights: Dict[str, float],
                                            target_weights: Dict[str, float],
                                            portfolio_value: float,
                                            market_data: Dict[str, Any]) -> List[RebalanceTransaction]:
        """Calculate optimal rebalancing transactions."""
        transactions = []
        
        try:
            tolerance = self.rebalance_config['rebalance_tolerance']
            min_trade_size = self.rebalance_config['min_trade_size']
            
            for symbol in target_weights:
                current_weight = current_weights.get(symbol, 0.0)
                target_weight = target_weights[symbol]
                
                weight_diff = target_weight - current_weight
                
                # Skip small changes
                if abs(weight_diff) < tolerance:
                    continue
                
                # Skip very small trades
                if abs(weight_diff) < min_trade_size:
                    continue
                
                # Calculate transaction details
                transaction_amount = weight_diff * portfolio_value
                action = "buy" if weight_diff > 0 else "sell"
                
                # Estimate transaction cost
                estimated_cost = await self._estimate_individual_transaction_cost(
                    abs(transaction_amount), market_data
                )
                
                # Determine priority (larger weight changes = higher priority)
                priority = int(abs(weight_diff) * 100)
                
                transaction = RebalanceTransaction(
                    symbol=symbol,
                    action=action,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    weight_change=weight_diff,
                    transaction_amount=abs(transaction_amount),
                    estimated_cost=estimated_cost,
                    priority=priority
                )
                
                transactions.append(transaction)
            
            # Sort by priority (highest first)
            transactions.sort(key=lambda t: t.priority, reverse=True)
            
            # Apply cost optimization if strategy requires it
            if strategy == RebalanceStrategy.COST_OPTIMIZED:
                transactions = await self._optimize_transaction_costs(transactions, portfolio_value)
            
            return transactions
            
        except Exception as e:
            self.logger.error(f"Failed to calculate optimal transactions: {e}")
            return []
    
    async def _estimate_transaction_costs(self, 
                                        transactions: List[RebalanceTransaction],
                                        market_data: Dict[str, Any]) -> float:
        """Estimate total transaction costs."""
        total_cost = 0.0
        
        try:
            for transaction in transactions:
                total_cost += transaction.estimated_cost
            
            return total_cost
            
        except Exception as e:
            self.logger.error(f"Failed to estimate transaction costs: {e}")
            return 0.0
    
    async def _estimate_individual_transaction_cost(self, 
                                                  trade_amount: float,
                                                  market_data: Dict[str, Any]) -> float:
        """Estimate cost for individual transaction."""
        try:
            costs = self.rebalance_config['transaction_costs']
            
            # Fixed cost
            fixed_cost = costs['fixed_cost']
            
            # Proportional cost
            proportional_cost = trade_amount * costs['proportional_cost']
            
            # Market impact (for large trades)
            market_impact = trade_amount * costs['market_impact']
            
            total_cost = fixed_cost + proportional_cost + market_impact
            
            return total_cost
            
        except Exception as e:
            self.logger.error(f"Failed to estimate individual transaction cost: {e}")
            return trade_amount * 0.001  # Default 0.1%
    
    async def _optimize_transaction_costs(self, 
                                        transactions: List[RebalanceTransaction],
                                        portfolio_value: float) -> List[RebalanceTransaction]:
        """Optimize transactions to minimize costs."""
        try:
            # Simple cost optimization: filter out very small trades
            min_trade_value = portfolio_value * 0.005  # 0.5% of portfolio
            
            optimized_transactions = [
                t for t in transactions 
                if t.transaction_amount >= min_trade_value
            ]
            
            # Combine related transactions if possible
            # (This is a simplified approach - in practice would be more sophisticated)
            
            return optimized_transactions
            
        except Exception as e:
            self.logger.error(f"Failed to optimize transaction costs: {e}")
            return transactions
    
    async def _simulate_transaction_execution(self, 
                                            transactions: List[RebalanceTransaction],
                                            market_data: Dict[str, Any]) -> List[RebalanceTransaction]:
        """Simulate transaction execution."""
        executed_transactions = []
        
        try:
            for transaction in transactions:
                # Simulate execution
                executed_transaction = RebalanceTransaction(
                    symbol=transaction.symbol,
                    action=transaction.action,
                    current_weight=transaction.current_weight,
                    target_weight=transaction.target_weight,
                    weight_change=transaction.weight_change,
                    transaction_amount=transaction.transaction_amount,
                    estimated_cost=transaction.estimated_cost,
                    priority=transaction.priority,
                    execution_price=1.0,  # Mock execution price
                    actual_cost=transaction.estimated_cost,  # Assume estimate is accurate
                    executed_at=datetime.now()
                )
                
                executed_transactions.append(executed_transaction)
            
            return executed_transactions
            
        except Exception as e:
            self.logger.error(f"Failed to simulate transaction execution: {e}")
            return transactions
    
    async def _calculate_rebalance_performance(self, 
                                             current_weights: Dict[str, float],
                                             target_weights: Dict[str, float],
                                             transactions: List[RebalanceTransaction],
                                             portfolio_value: float) -> Dict[str, Any]:
        """Calculate rebalancing performance metrics."""
        try:
            # Calculate tracking error reduction
            pre_rebalance_error = sum(abs(current_weights.get(symbol, 0) - target_weights[symbol]) 
                                    for symbol in target_weights)
            
            # Simulate post-rebalance weights
            post_rebalance_weights = current_weights.copy()
            for transaction in transactions:
                if transaction.executed_at:  # Only count executed transactions
                    new_weight = transaction.target_weight
                    post_rebalance_weights[transaction.symbol] = new_weight
            
            post_rebalance_error = sum(abs(post_rebalance_weights.get(symbol, 0) - target_weights[symbol])
                                     for symbol in target_weights)
            
            tracking_error_reduction = (pre_rebalance_error - post_rebalance_error) / pre_rebalance_error if pre_rebalance_error > 0 else 0
            
            # Calculate diversification improvement
            pre_hhi = sum(w**2 for w in current_weights.values())
            post_hhi = sum(w**2 for w in post_rebalance_weights.values())
            diversification_improvement = (pre_hhi - post_hhi) / pre_hhi if pre_hhi > 0 else 0
            
            # Calculate total transaction costs
            total_costs = sum(t.actual_cost or t.estimated_cost for t in transactions)
            
            # Risk reduction estimate (simplified)
            risk_reduction = tracking_error_reduction * 0.5  # Simplified estimate
            
            # Effectiveness score
            cost_penalty = total_costs / portfolio_value if portfolio_value > 0 else 0
            effectiveness_score = max(0, tracking_error_reduction - cost_penalty * 10)  # Penalize high costs
            
            # Execution summary
            execution_summary = {
                'total_transactions': len(transactions),
                'executed_transactions': len([t for t in transactions if t.executed_at]),
                'total_transaction_value': sum(t.transaction_amount for t in transactions),
                'largest_transaction': max(t.transaction_amount for t in transactions) if transactions else 0,
                'avg_transaction_size': np.mean([t.transaction_amount for t in transactions]) if transactions else 0
            }
            
            # Performance attribution
            performance_attribution = {}
            for transaction in transactions:
                if transaction.executed_at:
                    performance_attribution[transaction.symbol] = transaction.weight_change
            
            return {
                'tracking_error_reduction': float(tracking_error_reduction),
                'risk_reduction': float(risk_reduction),
                'diversification_improvement': float(diversification_improvement),
                'effectiveness_score': float(effectiveness_score),
                'market_impact': float(total_costs / portfolio_value) if portfolio_value > 0 else 0.0,
                'execution_summary': execution_summary,
                'performance_attribution': performance_attribution
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate rebalance performance: {e}")
            return {
                'tracking_error_reduction': 0.0,
                'risk_reduction': 0.0,
                'diversification_improvement': 0.0,
                'effectiveness_score': 0.0,
                'market_impact': 0.0,
                'execution_summary': {},
                'performance_attribution': {}
            }
    
    async def _calculate_next_rebalance_date(self, strategy: RebalanceStrategy) -> datetime:
        """Calculate next rebalance date based on strategy."""
        try:
            current_time = datetime.now()
            
            if strategy == RebalanceStrategy.CALENDAR:
                return current_time + timedelta(days=30)  # Monthly
            elif strategy == RebalanceStrategy.THRESHOLD:
                return current_time + timedelta(days=7)   # Weekly check
            elif strategy == RebalanceStrategy.VOLATILITY_TARGET:
                return current_time + timedelta(days=14)  # Bi-weekly
            elif strategy == RebalanceStrategy.ADAPTIVE:
                return current_time + timedelta(days=21)  # Every 3 weeks
            else:
                return current_time + timedelta(days=30)  # Default monthly
                
        except Exception as e:
            self.logger.error(f"Failed to calculate next rebalance date: {e}")
            return datetime.now() + timedelta(days=30)
    
    def _validate_rebalancing_inputs(self, 
                                   current_weights: Dict[str, float],
                                   target_weights: Dict[str, float],
                                   portfolio_value: float):
        """Validate rebalancing inputs."""
        if not current_weights:
            raise ValueError("Current weights cannot be empty")
        
        if not target_weights:
            raise ValueError("Target weights cannot be empty")
        
        if portfolio_value <= 0:
            raise ValueError("Portfolio value must be positive")
        
        # Check weight sums
        current_sum = sum(current_weights.values())
        target_sum = sum(target_weights.values())
        
        if abs(current_sum - 1.0) > 0.01:
            self.logger.warning(f"Current weights sum to {current_sum:.3f}, not 1.0")
        
        if abs(target_sum - 1.0) > 0.01:
            raise ValueError(f"Target weights sum to {target_sum:.3f}, not 1.0")
    
    def _store_rebalance_result(self, result: RebalanceResult):
        """Store rebalancing result in history."""
        self.rebalance_history.append(result)
        
        # Limit history size
        if len(self.rebalance_history) > 100:
            self.rebalance_history = self.rebalance_history[-50:]
        
        # Clear executed signals
        self.pending_signals = []
    
    def get_rebalance_history(self, lookback_days: int = None) -> List[RebalanceResult]:
        """Get rebalancing history."""
        if lookback_days is None:
            return self.rebalance_history.copy()
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        return [
            result for result in self.rebalance_history 
            if result.execution_date >= cutoff_date
        ]
    
    def get_pending_signals(self) -> List[RebalanceSignal]:
        """Get pending rebalancing signals."""
        return self.pending_signals.copy()
    
    def get_rebalancer_summary(self) -> Dict[str, Any]:
        """Get summary of rebalancer status."""
        if not self.rebalance_history:
            return {
                'total_rebalances': 0,
                'last_rebalance': None,
                'pending_signals': len(self.pending_signals),
                'avg_effectiveness': None,
                'total_transaction_costs': 0.0
            }
        
        latest = self.rebalance_history[-1]
        
        # Calculate average effectiveness
        effectiveness_scores = [r.rebalance_effectiveness for r in self.rebalance_history]
        avg_effectiveness = np.mean(effectiveness_scores) if effectiveness_scores else 0.0
        
        # Calculate total transaction costs
        total_costs = sum(r.total_transaction_costs for r in self.rebalance_history)
        
        return {
            'total_rebalances': len(self.rebalance_history),
            'last_rebalance': latest.execution_date,
            'pending_signals': len(self.pending_signals),
            'avg_effectiveness': float(avg_effectiveness),
            'total_transaction_costs': float(total_costs),
            'last_strategy': latest.strategy.value,
            'last_trigger': latest.trigger.value,
            'supported_strategies': [s.value for s in RebalanceStrategy],
            'next_rebalance': latest.next_rebalance_date
        }