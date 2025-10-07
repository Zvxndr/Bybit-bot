"""
Strategy Graduation System - Automatic Paper Trading to Live Trading Promotion

This module manages the automatic graduation of strategies from paper trading 
validation to live trading based on performance criteria and risk assessment.

Key Features:
- Real-time performance monitoring in paper trading
- Multi-criteria graduation scoring
- Risk-adjusted capital allocation
- Automatic promotion and demotion
- Continuous validation and adjustment

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import json

from .utils.logging import TradingLogger
from .backtest.validator import StrategyValidator, ValidationResult
from .config_manager import ConfigurationManager


class StrategyStage(Enum):
    """Strategy lifecycle stages"""
    RESEARCH = "research"           # Development phase
    PAPER_VALIDATION = "paper_validation"   # Paper trading validation
    LIVE_CANDIDATE = "live_candidate"       # Passed validation, waiting for graduation
    LIVE_TRADING = "live_trading"          # Active live trading
    UNDER_REVIEW = "under_review"          # Performance degradation detected
    RETIRED = "retired"                    # Permanently disabled


class GraduationDecision(Enum):
    """Graduation decision types"""
    PROMOTE = "promote"       # Graduate to next stage
    MAINTAIN = "maintain"     # Keep in current stage
    DEMOTE = "demote"        # Move back to previous stage
    RETIRE = "retire"        # Permanently disable


@dataclass
class PerformanceMetrics:
    """Strategy performance metrics for graduation assessment"""
    
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0
    
    # Consistency metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0
    
    # Market conditions
    market_correlation: float = 0.0
    regime_stability: float = 0.0
    
    # Operational metrics
    trades_count: int = 0
    avg_slippage: float = 0.0
    execution_success_rate: float = 1.0
    
    # Validation scores
    validation_score: float = 0.0
    confidence_level: str = "LOW"
    
    def calculate_graduation_score(self) -> float:
        """Calculate overall graduation score (0-1)"""
        weights = {
            'returns': 0.25,
            'risk': 0.25, 
            'consistency': 0.20,
            'validation': 0.15,
            'operational': 0.15
        }
        
        # Returns component (0-1)
        returns_score = min(max(self.sharpe_ratio / 2.0, 0), 1)
        
        # Risk component (0-1) - lower risk is better
        risk_score = max(1 - (self.max_drawdown * 5), 0)
        
        # Consistency component (0-1)
        consistency_score = (self.win_rate + min(self.profit_factor / 2.0, 1)) / 2
        
        # Validation component (0-1)
        validation_score = self.validation_score
        
        # Operational component (0-1)
        operational_score = (self.execution_success_rate + (1 - min(self.avg_slippage * 100, 1))) / 2
        
        # Weighted average
        total_score = (
            weights['returns'] * returns_score +
            weights['risk'] * risk_score +
            weights['consistency'] * consistency_score +
            weights['validation'] * validation_score +
            weights['operational'] * operational_score
        )
        
        return min(max(total_score, 0), 1)


@dataclass
class GraduationCriteria:
    """Criteria for strategy graduation between stages"""
    
    # Paper trading to live candidate
    paper_to_candidate: Dict[str, float] = field(default_factory=lambda: {
        'min_trades': 50,
        'min_days': 30,
        'min_sharpe': 1.0,
        'max_drawdown': 0.15,
        'min_win_rate': 0.45,
        'min_profit_factor': 1.2,
        'min_graduation_score': 0.7,
        'min_validation_score': 0.8
    })
    
    # Live candidate to live trading
    candidate_to_live: Dict[str, float] = field(default_factory=lambda: {
        'min_trades': 100,
        'min_days': 60,
        'min_sharpe': 1.2,
        'max_drawdown': 0.12,
        'min_win_rate': 0.48,
        'min_profit_factor': 1.3,
        'min_graduation_score': 0.75,
        'min_validation_score': 0.85,
        'max_correlation': 0.7  # Avoid overly correlated strategies
    })
    
    # Live trading maintenance
    live_maintenance: Dict[str, float] = field(default_factory=lambda: {
        'min_sharpe_30d': 0.5,
        'max_drawdown_30d': 0.20,
        'min_profit_factor_30d': 1.0,
        'min_execution_rate': 0.95
    })
    
    # Demotion triggers
    demotion_triggers: Dict[str, float] = field(default_factory=lambda: {
        'max_consecutive_losses': 10,
        'max_drawdown_period': 0.25,
        'min_sharpe_7d': -1.0,
        'max_correlation_increase': 0.9
    })


@dataclass
class StrategyRecord:
    """Complete record of a strategy's lifecycle"""
    
    strategy_id: str
    name: str
    created_date: datetime
    current_stage: StrategyStage
    
    # Stage history
    stage_history: List[Tuple[datetime, StrategyStage, str]] = field(default_factory=list)
    
    # Performance tracking
    performance_history: List[Tuple[datetime, PerformanceMetrics]] = field(default_factory=list)
    current_metrics: Optional[PerformanceMetrics] = None
    
    # Validation results
    latest_validation: Optional[ValidationResult] = None
    validation_history: List[ValidationResult] = field(default_factory=list)
    
    # Capital allocation (for live strategies)
    allocated_capital: float = 0.0
    max_capital: float = 0.0
    capital_history: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # Configuration
    strategy_config: Dict[str, Any] = field(default_factory=dict)
    risk_params: Dict[str, Any] = field(default_factory=dict)
    
    # Status flags
    is_active: bool = True
    last_updated: datetime = field(default_factory=datetime.now)
    notes: List[str] = field(default_factory=list)
    
    def add_stage_change(self, new_stage: StrategyStage, reason: str):
        """Record stage change"""
        self.stage_history.append((datetime.now(), new_stage, reason))
        self.current_stage = new_stage
        self.last_updated = datetime.now()
    
    def add_performance_snapshot(self, metrics: PerformanceMetrics):
        """Add performance snapshot"""
        self.performance_history.append((datetime.now(), metrics))
        self.current_metrics = metrics
        self.last_updated = datetime.now()
    
    def update_capital_allocation(self, new_allocation: float, reason: str):
        """Update capital allocation"""
        self.capital_history.append((datetime.now(), new_allocation))
        self.allocated_capital = new_allocation
        self.notes.append(f"Capital updated to ${new_allocation:,.2f}: {reason}")
        self.last_updated = datetime.now()


class StrategyGraduationManager:
    """
    Manages automatic strategy graduation from paper trading to live trading
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = TradingLogger("StrategyGraduation")
        
        # Components
        self.validator = StrategyValidator()
        
        # Configuration
        self.criteria = GraduationCriteria()
        self.strategies: Dict[str, StrategyRecord] = {}
        
        # State
        self.is_running = False
        self.last_evaluation = None
        
        # Paths
        self.data_path = Path("data/graduation")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing strategy records
        self._load_strategy_records()
    
    def register_strategy(
        self,
        strategy_id: str,
        name: str,
        config: Dict[str, Any],
        initial_stage: StrategyStage = StrategyStage.PAPER_VALIDATION
    ) -> StrategyRecord:
        """Register a new strategy for graduation tracking"""
        
        if strategy_id in self.strategies:
            self.logger.warning(f"Strategy {strategy_id} already registered")
            return self.strategies[strategy_id]
        
        record = StrategyRecord(
            strategy_id=strategy_id,
            name=name,
            created_date=datetime.now(),
            current_stage=initial_stage,
            strategy_config=config
        )
        
        record.add_stage_change(initial_stage, "Initial registration")
        
        self.strategies[strategy_id] = record
        self._save_strategy_record(record)
        
        self.logger.info(f"Registered strategy {name} ({strategy_id}) in {initial_stage.value} stage")
        return record
    
    async def evaluate_all_strategies(self) -> Dict[str, GraduationDecision]:
        """Evaluate all strategies for potential graduation/demotion"""
        
        self.logger.info("Starting strategy graduation evaluation")
        decisions = {}
        
        for strategy_id, record in self.strategies.items():
            if not record.is_active:
                continue
            
            try:
                decision = await self._evaluate_strategy(record)
                decisions[strategy_id] = decision
                
                if decision != GraduationDecision.MAINTAIN:
                    await self._execute_graduation_decision(record, decision)
                    
            except Exception as e:
                self.logger.error(f"Error evaluating strategy {strategy_id}: {e}")
                decisions[strategy_id] = GraduationDecision.MAINTAIN
        
        self.last_evaluation = datetime.now()
        self._save_evaluation_summary(decisions)
        
        return decisions
    
    async def _evaluate_strategy(self, record: StrategyRecord) -> GraduationDecision:
        """Evaluate a single strategy for graduation"""
        
        if not record.current_metrics:
            return GraduationDecision.MAINTAIN
        
        current_stage = record.current_stage
        metrics = record.current_metrics
        
        # Calculate graduation score
        graduation_score = metrics.calculate_graduation_score()
        
        if current_stage == StrategyStage.PAPER_VALIDATION:
            return self._evaluate_paper_to_candidate(record, graduation_score)
        
        elif current_stage == StrategyStage.LIVE_CANDIDATE:
            return self._evaluate_candidate_to_live(record, graduation_score)
        
        elif current_stage == StrategyStage.LIVE_TRADING:
            return self._evaluate_live_maintenance(record, graduation_score)
        
        elif current_stage == StrategyStage.UNDER_REVIEW:
            return self._evaluate_under_review(record, graduation_score)
        
        return GraduationDecision.MAINTAIN
    
    def _evaluate_paper_to_candidate(self, record: StrategyRecord, score: float) -> GraduationDecision:
        """Evaluate paper trading strategy for candidate promotion"""
        
        criteria = self.criteria.paper_to_candidate
        metrics = record.current_metrics
        
        # Check all criteria
        checks = {
            'trades': metrics.trades_count >= criteria['min_trades'],
            'days': self._get_days_in_stage(record) >= criteria['min_days'],
            'sharpe': metrics.sharpe_ratio >= criteria['min_sharpe'],
            'drawdown': metrics.max_drawdown <= criteria['max_drawdown'],
            'win_rate': metrics.win_rate >= criteria['min_win_rate'],
            'profit_factor': metrics.profit_factor >= criteria['min_profit_factor'],
            'graduation_score': score >= criteria['min_graduation_score'],
            'validation': metrics.validation_score >= criteria['min_validation_score']
        }
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        self.logger.info(f"Paper→Candidate evaluation for {record.name}: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            return GraduationDecision.PROMOTE
        elif passed_checks >= total_checks * 0.6:  # 60% pass rate
            return GraduationDecision.MAINTAIN
        else:
            # Performance degradation - check for demotion
            if self._check_demotion_triggers(record):
                return GraduationDecision.DEMOTE
            return GraduationDecision.MAINTAIN
    
    def _evaluate_candidate_to_live(self, record: StrategyRecord, score: float) -> GraduationDecision:
        """Evaluate candidate strategy for live trading promotion"""
        
        criteria = self.criteria.candidate_to_live
        metrics = record.current_metrics
        
        # Additional check: correlation with existing live strategies
        correlation_check = self._check_strategy_correlation(record) <= criteria['max_correlation']
        
        checks = {
            'trades': metrics.trades_count >= criteria['min_trades'],
            'days': self._get_days_in_stage(record) >= criteria['min_days'],
            'sharpe': metrics.sharpe_ratio >= criteria['min_sharpe'],
            'drawdown': metrics.max_drawdown <= criteria['max_drawdown'],
            'win_rate': metrics.win_rate >= criteria['min_win_rate'],
            'profit_factor': metrics.profit_factor >= criteria['min_profit_factor'],
            'graduation_score': score >= criteria['min_graduation_score'],
            'validation': metrics.validation_score >= criteria['min_validation_score'],
            'correlation': correlation_check
        }
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        self.logger.info(f"Candidate→Live evaluation for {record.name}: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            return GraduationDecision.PROMOTE
        elif self._check_demotion_triggers(record):
            return GraduationDecision.DEMOTE
        else:
            return GraduationDecision.MAINTAIN
    
    def _evaluate_live_maintenance(self, record: StrategyRecord, score: float) -> GraduationDecision:
        """Evaluate live trading strategy maintenance"""
        
        criteria = self.criteria.live_maintenance
        metrics = record.current_metrics
        
        # Get recent performance (30-day window)
        recent_metrics = self._get_recent_metrics(record, days=30)
        
        if not recent_metrics:
            return GraduationDecision.MAINTAIN
        
        checks = {
            'sharpe_30d': recent_metrics.sharpe_ratio >= criteria['min_sharpe_30d'],
            'drawdown_30d': recent_metrics.max_drawdown <= criteria['max_drawdown_30d'],
            'profit_factor_30d': recent_metrics.profit_factor >= criteria['min_profit_factor_30d'],
            'execution_rate': metrics.execution_success_rate >= criteria['min_execution_rate']
        }
        
        # Check for immediate demotion triggers
        if self._check_demotion_triggers(record):
            return GraduationDecision.DEMOTE
        
        passed_checks = sum(checks.values())
        
        if passed_checks >= len(checks) * 0.75:  # 75% pass rate for live strategies
            return GraduationDecision.MAINTAIN
        else:
            # Put under review for potential demotion
            return GraduationDecision.DEMOTE  # Move to UNDER_REVIEW stage
    
    def _evaluate_under_review(self, record: StrategyRecord, score: float) -> GraduationDecision:
        """Evaluate strategy under review"""
        
        days_under_review = self._get_days_in_stage(record)
        recent_metrics = self._get_recent_metrics(record, days=7)
        
        # Give strategy time to recover (up to 14 days)
        if days_under_review < 14:
            if recent_metrics and recent_metrics.sharpe_ratio > 0.5:
                return GraduationDecision.PROMOTE  # Back to live trading
            return GraduationDecision.MAINTAIN
        
        # After 14 days, make final decision
        if recent_metrics and score >= 0.6:
            return GraduationDecision.PROMOTE  # Redemption
        else:
            return GraduationDecision.RETIRE  # Permanent retirement
    
    async def _execute_graduation_decision(self, record: StrategyRecord, decision: GraduationDecision):
        """Execute graduation decision"""
        
        current_stage = record.current_stage
        strategy_name = record.name
        
        if decision == GraduationDecision.PROMOTE:
            new_stage = self._get_next_stage(current_stage)
            reason = f"Promoted from {current_stage.value} to {new_stage.value}"
            
            # Handle capital allocation for live trading promotion
            if new_stage == StrategyStage.LIVE_TRADING:
                initial_capital = self._calculate_initial_capital(record)
                record.update_capital_allocation(initial_capital, "Initial live trading allocation")
            
        elif decision == GraduationDecision.DEMOTE:
            new_stage = self._get_previous_stage(current_stage)
            reason = f"Demoted from {current_stage.value} to {new_stage.value}"
            
            # Reduce capital allocation
            if current_stage == StrategyStage.LIVE_TRADING:
                new_capital = record.allocated_capital * 0.5  # Reduce by 50%
                record.update_capital_allocation(new_capital, "Performance-based capital reduction")
        
        elif decision == GraduationDecision.RETIRE:
            new_stage = StrategyStage.RETIRED
            reason = "Strategy retired due to poor performance"
            record.is_active = False
            record.update_capital_allocation(0.0, "Strategy retirement - capital withdrawn")
        
        else:
            return  # No change
        
        record.add_stage_change(new_stage, reason)
        self._save_strategy_record(record)
        
        self.logger.info(f"Strategy {strategy_name}: {reason}")
        
        # Send notifications for significant changes
        if decision in [GraduationDecision.PROMOTE, GraduationDecision.RETIRE]:
            await self._send_graduation_notification(record, decision, reason)
    
    def _calculate_initial_capital(self, record: StrategyRecord) -> float:
        """Calculate initial capital allocation for newly graduated strategy"""
        
        base_allocation = 1000.0  # Base $1000
        performance_multiplier = min(record.current_metrics.graduation_score * 2, 3.0)  # Max 3x
        
        # Factor in validation confidence
        confidence_multiplier = {
            'HIGH': 1.5,
            'MEDIUM': 1.0,
            'LOW': 0.5
        }.get(record.current_metrics.confidence_level, 1.0)
        
        initial_capital = base_allocation * performance_multiplier * confidence_multiplier
        
        # Cap at reasonable limits
        max_initial = 5000.0  # Maximum $5000 initial allocation
        return min(initial_capital, max_initial)
    
    def _get_days_in_stage(self, record: StrategyRecord) -> int:
        """Get number of days strategy has been in current stage"""
        
        if not record.stage_history:
            return 0
        
        # Find when current stage started
        current_stage_start = None
        for timestamp, stage, _ in reversed(record.stage_history):
            if stage == record.current_stage:
                current_stage_start = timestamp
            else:
                break
        
        if current_stage_start:
            return (datetime.now() - current_stage_start).days
        
        return 0
    
    def _get_recent_metrics(self, record: StrategyRecord, days: int) -> Optional[PerformanceMetrics]:
        """Get performance metrics for recent period"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_snapshots = [
            (timestamp, metrics) for timestamp, metrics in record.performance_history
            if timestamp >= cutoff_date
        ]
        
        if not recent_snapshots:
            return None
        
        # For now, return the most recent metrics
        # In a full implementation, you'd aggregate the metrics over the period
        return recent_snapshots[-1][1]
    
    def _check_strategy_correlation(self, record: StrategyRecord) -> float:
        """Check correlation with existing live strategies"""
        
        # Get all live trading strategies
        live_strategies = [
            r for r in self.strategies.values()
            if r.current_stage == StrategyStage.LIVE_TRADING and r.strategy_id != record.strategy_id
        ]
        
        if not live_strategies:
            return 0.0  # No correlation if no live strategies
        
        # This would require actual return correlation calculation
        # For now, return a placeholder based on strategy type similarity
        max_correlation = 0.0
        
        for live_strategy in live_strategies:
            # Simplified correlation based on strategy configuration similarity
            correlation = self._calculate_config_similarity(record, live_strategy)
            max_correlation = max(max_correlation, correlation)
        
        return max_correlation
    
    def _calculate_config_similarity(self, strategy1: StrategyRecord, strategy2: StrategyRecord) -> float:
        """Calculate similarity between strategy configurations"""
        
        # Simplified similarity calculation
        # In practice, this would compare actual strategy parameters
        
        config1 = strategy1.strategy_config
        config2 = strategy2.strategy_config
        
        # Check strategy type similarity
        type1 = config1.get('strategy_type', 'unknown')
        type2 = config2.get('strategy_type', 'unknown')
        
        if type1 == type2:
            return 0.8  # High similarity for same strategy type
        else:
            return 0.2  # Low similarity for different types
    
    def _check_demotion_triggers(self, record: StrategyRecord) -> bool:
        """Check if strategy should be demoted due to poor performance"""
        
        triggers = self.criteria.demotion_triggers
        metrics = record.current_metrics
        
        if not metrics:
            return False
        
        # Check various demotion triggers
        trigger_flags = {
            'excessive_drawdown': metrics.max_drawdown > triggers['max_drawdown_period'],
            'poor_recent_sharpe': self._get_recent_sharpe(record, 7) < triggers['min_sharpe_7d'],
            'high_correlation': self._check_strategy_correlation(record) > triggers['max_correlation_increase']
        }
        
        # Any critical trigger should cause demotion
        return any(trigger_flags.values())
    
    def _get_recent_sharpe(self, record: StrategyRecord, days: int) -> float:
        """Get Sharpe ratio for recent period"""
        recent_metrics = self._get_recent_metrics(record, days)
        return recent_metrics.sharpe_ratio if recent_metrics else 0.0
    
    def _get_next_stage(self, current_stage: StrategyStage) -> StrategyStage:
        """Get next stage in progression"""
        progression = {
            StrategyStage.PAPER_VALIDATION: StrategyStage.LIVE_CANDIDATE,
            StrategyStage.LIVE_CANDIDATE: StrategyStage.LIVE_TRADING,
            StrategyStage.UNDER_REVIEW: StrategyStage.LIVE_TRADING,
        }
        return progression.get(current_stage, current_stage)
    
    def _get_previous_stage(self, current_stage: StrategyStage) -> StrategyStage:
        """Get previous stage for demotion"""
        regression = {
            StrategyStage.LIVE_TRADING: StrategyStage.UNDER_REVIEW,
            StrategyStage.LIVE_CANDIDATE: StrategyStage.PAPER_VALIDATION,
            StrategyStage.UNDER_REVIEW: StrategyStage.PAPER_VALIDATION,
        }
        return regression.get(current_stage, current_stage)
    
    def _load_strategy_records(self):
        """Load existing strategy records from disk"""
        
        records_file = self.data_path / "strategy_records.json"
        
        if records_file.exists():
            try:
                with open(records_file, 'r') as f:
                    data = json.load(f)
                
                for strategy_id, record_data in data.items():
                    # Reconstruct StrategyRecord from saved data
                    record = self._deserialize_strategy_record(record_data)
                    self.strategies[strategy_id] = record
                
                self.logger.info(f"Loaded {len(self.strategies)} strategy records")
                
            except Exception as e:
                self.logger.error(f"Error loading strategy records: {e}")
    
    def _save_strategy_record(self, record: StrategyRecord):
        """Save individual strategy record"""
        
        # Save to main records file
        records_file = self.data_path / "strategy_records.json"
        
        # Load existing records
        all_records = {}
        if records_file.exists():
            try:
                with open(records_file, 'r') as f:
                    all_records = json.load(f)
            except:
                pass
        
        # Update with current record
        all_records[record.strategy_id] = self._serialize_strategy_record(record)
        
        # Save back
        with open(records_file, 'w') as f:
            json.dump(all_records, f, indent=2, default=str)
    
    def _serialize_strategy_record(self, record: StrategyRecord) -> dict:
        """Convert StrategyRecord to JSON-serializable dict"""
        
        return {
            'strategy_id': record.strategy_id,
            'name': record.name,
            'created_date': record.created_date.isoformat(),
            'current_stage': record.current_stage.value,
            'stage_history': [
                [ts.isoformat(), stage.value, reason]
                for ts, stage, reason in record.stage_history
            ],
            'allocated_capital': record.allocated_capital,
            'max_capital': record.max_capital,
            'strategy_config': record.strategy_config,
            'risk_params': record.risk_params,
            'is_active': record.is_active,
            'last_updated': record.last_updated.isoformat(),
            'notes': record.notes
        }
    
    def _deserialize_strategy_record(self, data: dict) -> StrategyRecord:
        """Convert JSON dict back to StrategyRecord"""
        
        record = StrategyRecord(
            strategy_id=data['strategy_id'],
            name=data['name'],
            created_date=datetime.fromisoformat(data['created_date']),
            current_stage=StrategyStage(data['current_stage']),
            allocated_capital=data.get('allocated_capital', 0.0),
            max_capital=data.get('max_capital', 0.0),
            strategy_config=data.get('strategy_config', {}),
            risk_params=data.get('risk_params', {}),
            is_active=data.get('is_active', True),
            last_updated=datetime.fromisoformat(data['last_updated']),
            notes=data.get('notes', [])
        )
        
        # Reconstruct stage history
        for ts_str, stage_str, reason in data.get('stage_history', []):
            record.stage_history.append((
                datetime.fromisoformat(ts_str),
                StrategyStage(stage_str),
                reason
            ))
        
        return record
    
    def _save_evaluation_summary(self, decisions: Dict[str, GraduationDecision]):
        """Save graduation evaluation summary"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_strategies': len(self.strategies),
            'active_strategies': sum(1 for r in self.strategies.values() if r.is_active),
            'decisions': {k: v.value for k, v in decisions.items()},
            'stage_distribution': self._get_stage_distribution(),
            'total_allocated_capital': sum(r.allocated_capital for r in self.strategies.values())
        }
        
        summary_file = self.data_path / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _get_stage_distribution(self) -> Dict[str, int]:
        """Get distribution of strategies across stages"""
        
        distribution = {}
        for stage in StrategyStage:
            count = sum(1 for r in self.strategies.values() if r.current_stage == stage and r.is_active)
            distribution[stage.value] = count
        
        return distribution
    
    async def _send_graduation_notification(self, record: StrategyRecord, decision: GraduationDecision, reason: str):
        """Send notification about strategy graduation"""
        
        # This would integrate with your notification system
        message = f"Strategy Graduation Alert: {record.name} - {reason}"
        
        if decision == GraduationDecision.PROMOTE and record.current_stage == StrategyStage.LIVE_TRADING:
            message += f" | Allocated Capital: ${record.allocated_capital:,.2f}"
        
        self.logger.info(f"GRADUATION ALERT: {message}")
        
        # TODO: Integrate with email/Discord/Slack notifications
    
    def get_graduation_report(self) -> Dict[str, Any]:
        """Generate comprehensive graduation system report"""
        
        active_strategies = [r for r in self.strategies.values() if r.is_active]
        
        report = {
            'summary': {
                'total_strategies': len(self.strategies),
                'active_strategies': len(active_strategies),
                'total_allocated_capital': sum(r.allocated_capital for r in active_strategies),
                'last_evaluation': self.last_evaluation.isoformat() if self.last_evaluation else None
            },
            'stage_distribution': self._get_stage_distribution(),
            'performance_summary': self._get_performance_summary(),
            'recent_graduations': self._get_recent_graduations(),
            'capital_allocation': self._get_capital_allocation_summary()
        }
        
        return report
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all strategies"""
        
        live_strategies = [
            r for r in self.strategies.values()
            if r.current_stage == StrategyStage.LIVE_TRADING and r.is_active and r.current_metrics
        ]
        
        if not live_strategies:
            return {}
        
        metrics_list = [r.current_metrics for r in live_strategies]
        
        return {
            'avg_sharpe': np.mean([m.sharpe_ratio for m in metrics_list]),
            'avg_drawdown': np.mean([m.max_drawdown for m in metrics_list]),
            'avg_win_rate': np.mean([m.win_rate for m in metrics_list]),
            'total_trades': sum([m.trades_count for m in metrics_list]),
            'avg_graduation_score': np.mean([m.calculate_graduation_score() for m in metrics_list])
        }
    
    def _get_recent_graduations(self) -> List[Dict[str, Any]]:
        """Get recent graduation events"""
        
        recent_events = []
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for record in self.strategies.values():
            for timestamp, stage, reason in record.stage_history:
                if timestamp >= cutoff_date:
                    recent_events.append({
                        'timestamp': timestamp.isoformat(),
                        'strategy': record.name,
                        'stage': stage.value,
                        'reason': reason
                    })
        
        return sorted(recent_events, key=lambda x: x['timestamp'], reverse=True)[:10]
    
    def _get_capital_allocation_summary(self) -> Dict[str, Any]:
        """Get capital allocation summary"""
        
        live_strategies = [
            r for r in self.strategies.values()
            if r.current_stage == StrategyStage.LIVE_TRADING and r.is_active
        ]
        
        if not live_strategies:
            return {}
        
        allocations = [r.allocated_capital for r in live_strategies]
        
        return {
            'total_allocated': sum(allocations),
            'average_allocation': np.mean(allocations),
            'min_allocation': min(allocations),
            'max_allocation': max(allocations),
            'strategies_count': len(live_strategies)
        }


# Example usage and integration
async def example_graduation_workflow():
    """Example of how to use the graduation system"""
    
    # Initialize
    config_manager = ConfigurationManager()
    graduation_manager = StrategyGraduationManager(config_manager)
    
    # Register a new strategy
    strategy_record = graduation_manager.register_strategy(
        strategy_id="momentum_v1",
        name="Momentum Strategy V1",
        config={
            'strategy_type': 'momentum',
            'lookback_period': 20,
            'threshold': 0.02
        }
    )
    
    # Simulate adding performance metrics
    metrics = PerformanceMetrics(
        total_return=0.15,
        sharpe_ratio=1.2,
        max_drawdown=0.08,
        win_rate=0.55,
        profit_factor=1.4,
        trades_count=75,
        validation_score=0.85,
        confidence_level="HIGH"
    )
    
    strategy_record.add_performance_snapshot(metrics)
    
    # Run graduation evaluation
    decisions = await graduation_manager.evaluate_all_strategies()
    
    # Get comprehensive report
    report = graduation_manager.get_graduation_report()
    
    print("Graduation System Report:")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(example_graduation_workflow())