"""
ML Risk Parameter Self-Adjustment System
Safe implementation with human oversight and regulatory compliance
"""

import asyncio
import logging
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from decimal import Decimal
import json
import hashlib

logger = logging.getLogger(__name__)

class AdjustmentType(Enum):
    """Types of parameter adjustments"""
    MINOR = "minor"           # Within pre-approved ranges, can be automatic
    MODERATE = "moderate"     # Requires risk manager approval
    MAJOR = "major"           # Requires full committee approval

class ApprovalStatus(Enum):
    """Approval status for parameter changes"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    ROLLED_BACK = "rolled_back"

@dataclass
class RiskParameterChange:
    """Represents a proposed or implemented risk parameter change"""
    change_id: str
    timestamp: datetime
    parameter_name: str
    current_value: float
    proposed_value: float
    change_percentage: float
    adjustment_type: AdjustmentType
    rationale: str
    supporting_evidence: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    backtesting_results: Dict[str, Any]
    approval_status: ApprovalStatus
    approved_by: Optional[str] = None
    implementation_timestamp: Optional[datetime] = None
    rollback_timestamp: Optional[datetime] = None
    performance_impact: Optional[Dict[str, Any]] = None

class MLRiskParameterAdjuster:
    """
    Safe ML-driven risk parameter adjustment system
    
    Features:
    - Human oversight for all material changes
    - Immutable safety limits
    - Comprehensive audit trails
    - Automatic performance monitoring
    - Rollback capabilities
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.change_history: List[RiskParameterChange] = []
        self.pending_changes: List[RiskParameterChange] = []
        
        # Load current configuration
        self.current_config = self._load_config()
        
        # Define immutable safety limits (cannot be changed by ML)
        self.immutable_limits = {
            'max_portfolio_drawdown': 0.15,        # 15% absolute maximum
            'emergency_stop_threshold': 0.05,      # 5% emergency stop (unchangeable)
            'min_confidence_floor': 0.30,          # Never trade below 30% confidence
            'max_position_size_pct': 0.20,         # Never risk more than 20% on single trade
            'daily_loss_limit_floor': 0.02         # Minimum 2% daily loss limit
        }
        
        # Define adjustment ranges for autonomous changes
        self.autonomous_ranges = {
            'min_confidence_threshold': (0.55, 0.75),      # Can adjust confidence threshold
            'confidence_scaling_factor': (1.5, 2.5),       # Position sizing based on confidence
            'volatility_spike_multiplier': (2.0, 4.0),     # Volatility response range
            'circuit_breaker_recovery_minutes': (10, 45),   # Recovery time ranges
        }
        
        # Governance rules
        self.governance_rules = {
            'max_adjustments_per_day': 5,
            'max_parameter_change_pct': 0.15,               # 15% max change per adjustment
            'mandatory_cooling_period_minutes': 30,         # 30 minutes between changes
            'require_backtest_validation': True,
            'require_stress_test_pass': True,
            'min_performance_history_days': 7               # Need 7 days data for analysis
        }
        
        # Performance tracking
        self.performance_metrics = {}
        self.last_adjustment_time = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load current risk configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save updated configuration with backup"""
        try:
            # Create backup
            backup_path = f"{self.config_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(backup_path, 'w') as f:
                yaml.dump(self.current_config, f, default_flow_style=False)
            
            # Save new configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            logger.info(f"Configuration saved. Backup created: {backup_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
    
    def _generate_change_id(self, parameter_name: str, proposed_value: float) -> str:
        """Generate unique change ID"""
        data = f"{parameter_name}_{proposed_value}_{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:8]
    
    def _calculate_change_percentage(self, current: float, proposed: float) -> float:
        """Calculate percentage change"""
        if current == 0:
            return float('inf') if proposed != 0 else 0
        return abs((proposed - current) / current) * 100
    
    def _determine_adjustment_type(self, parameter_name: str, change_pct: float) -> AdjustmentType:
        """Determine the type of adjustment based on parameter and change size"""
        
        # Major changes always require full approval
        if change_pct > 25:
            return AdjustmentType.MAJOR
        
        # Safety-critical parameters require higher approval
        safety_critical = [
            'emergency_stop_threshold',
            'max_portfolio_drawdown', 
            'circuit_breaker',
            'daily_loss_limit'
        ]
        
        if any(critical in parameter_name.lower() for critical in safety_critical):
            return AdjustmentType.MAJOR if change_pct > 10 else AdjustmentType.MODERATE
        
        # Autonomous parameters with small changes
        if parameter_name in self.autonomous_ranges and change_pct <= 10:
            return AdjustmentType.MINOR
        
        # Everything else is moderate
        return AdjustmentType.MODERATE
    
    def _validate_safety_constraints(self, parameter_name: str, proposed_value: float) -> Tuple[bool, str]:
        """Validate that proposed change doesn't violate safety constraints"""
        
        # Check immutable limits
        for limit_name, limit_value in self.immutable_limits.items():
            if limit_name.replace('_', '') in parameter_name.replace('_', ''):
                if 'max' in limit_name or 'threshold' in limit_name:
                    if proposed_value > limit_value:
                        return False, f"Proposed value {proposed_value} exceeds immutable limit {limit_value}"
                elif 'min' in limit_name or 'floor' in limit_name:
                    if proposed_value < limit_value:
                        return False, f"Proposed value {proposed_value} below immutable floor {limit_value}"
        
        # Check autonomous ranges
        if parameter_name in self.autonomous_ranges:
            min_val, max_val = self.autonomous_ranges[parameter_name]
            if not (min_val <= proposed_value <= max_val):
                return False, f"Proposed value {proposed_value} outside autonomous range [{min_val}, {max_val}]"
        
        return True, "Validation passed"
    
    def _check_cooling_period(self) -> Tuple[bool, str]:
        """Check if enough time has passed since last adjustment"""
        if self.last_adjustment_time is None:
            return True, "No previous adjustments"
        
        time_since_last = datetime.now() - self.last_adjustment_time
        cooling_period = timedelta(minutes=self.governance_rules['mandatory_cooling_period_minutes'])
        
        if time_since_last < cooling_period:
            remaining = cooling_period - time_since_last
            return False, f"Cooling period active. {remaining} remaining"
        
        return True, "Cooling period satisfied"
    
    async def _analyze_parameter_performance(self, parameter_name: str) -> Dict[str, Any]:
        """Analyze current performance of a parameter"""
        
        # Simulate performance analysis (in real implementation, would analyze actual trading data)
        analysis = {
            'current_performance': {
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08,
                'win_rate': 0.65,
                'avg_return': 0.15,
                'volatility': 0.18
            },
            'parameter_effectiveness': {
                'trades_filtered': 150,
                'false_positives': 12,
                'false_negatives': 8,
                'parameter_utilization': 0.75
            },
            'market_conditions': {
                'volatility_regime': 'moderate',
                'trend_strength': 0.6,
                'correlation_breakdown': False,
                'liquidity_stress': False
            },
            'optimization_potential': {
                'estimated_improvement': 0.08,  # 8% performance improvement
                'confidence_interval': (0.04, 0.12),
                'risk_assessment': 'low'
            }
        }
        
        return analysis
    
    async def _run_backtesting(self, parameter_name: str, proposed_value: float) -> Dict[str, Any]:
        """Run backtesting with proposed parameter value"""
        
        # Simulate backtesting results (in real implementation, would run actual backtests)
        backtest_results = {
            'test_period': '2025-06-01 to 2025-09-01',
            'total_trades': 245,
            'current_parameter_results': {
                'total_return': 0.18,
                'sharpe_ratio': 1.25,
                'max_drawdown': 0.09,
                'win_rate': 0.64
            },
            'proposed_parameter_results': {
                'total_return': 0.22,
                'sharpe_ratio': 1.38,
                'max_drawdown': 0.07,
                'win_rate': 0.68
            },
            'improvement_metrics': {
                'return_improvement': 0.04,       # 4% better returns
                'risk_reduction': 0.02,          # 2% less drawdown
                'efficiency_gain': 0.13,         # 13% better Sharpe ratio
                'trade_quality_improvement': 0.04 # 4% better win rate
            },
            'stress_test_results': {
                'high_volatility': 'passed',
                'market_crash': 'passed',
                'low_liquidity': 'passed',
                'correlation_breakdown': 'passed'
            }
        }
        
        return backtest_results
    
    async def _generate_adjustment_recommendation(self, parameter_name: str) -> Optional[RiskParameterChange]:
        """Generate ML-based parameter adjustment recommendation"""
        
        # Get current parameter value
        current_value = self._get_current_parameter_value(parameter_name)
        if current_value is None:
            logger.warning(f"Parameter {parameter_name} not found in configuration")
            return None
        
        # Analyze current performance
        performance_analysis = await self._analyze_parameter_performance(parameter_name)
        
        # Generate optimization recommendation (simplified ML optimization)
        optimization_result = self._optimize_parameter(parameter_name, current_value, performance_analysis)
        proposed_value = optimization_result['optimal_value']
        
        # Calculate change metrics
        change_pct = self._calculate_change_percentage(current_value, proposed_value)
        adjustment_type = self._determine_adjustment_type(parameter_name, change_pct)
        
        # Validate safety constraints
        is_safe, safety_msg = self._validate_safety_constraints(parameter_name, proposed_value)
        if not is_safe:
            logger.warning(f"Parameter adjustment rejected: {safety_msg}")
            return None
        
        # Run backtesting
        backtest_results = await self._run_backtesting(parameter_name, proposed_value)
        
        # Create change recommendation
        change = RiskParameterChange(
            change_id=self._generate_change_id(parameter_name, proposed_value),
            timestamp=datetime.now(),
            parameter_name=parameter_name,
            current_value=current_value,
            proposed_value=proposed_value,
            change_percentage=change_pct,
            adjustment_type=adjustment_type,
            rationale=optimization_result['rationale'],
            supporting_evidence=performance_analysis,
            risk_assessment=optimization_result['risk_assessment'],
            backtesting_results=backtest_results,
            approval_status=ApprovalStatus.PENDING
        )
        
        return change
    
    def _optimize_parameter(self, parameter_name: str, current_value: float, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parameter value based on analysis (simplified ML optimization)"""
        
        # Simulate ML optimization logic
        if 'confidence' in parameter_name.lower():
            # For confidence thresholds, optimize based on false positive/negative rate
            false_pos_rate = analysis['parameter_effectiveness']['false_positives'] / 150
            false_neg_rate = analysis['parameter_effectiveness']['false_negatives'] / 150
            
            if false_pos_rate > 0.10:  # Too many false positives, increase threshold
                optimal_value = min(current_value * 1.08, 0.75)  # Increase by 8%, max 75%
                rationale = f"Reducing false positives (current rate: {false_pos_rate:.1%})"
            elif false_neg_rate > 0.08:  # Too many false negatives, decrease threshold
                optimal_value = max(current_value * 0.95, 0.50)  # Decrease by 5%, min 50%
                rationale = f"Reducing false negatives (current rate: {false_neg_rate:.1%})"
            else:
                optimal_value = current_value  # No change needed
                rationale = "Current parameter performing optimally"
                
        elif 'scaling' in parameter_name.lower():
            # For scaling factors, optimize based on risk-adjusted returns
            current_sharpe = analysis['current_performance']['sharpe_ratio']
            if current_sharpe < 1.0:  # Poor risk-adjusted returns, reduce scaling
                optimal_value = max(current_value * 0.90, 1.0)
                rationale = f"Reducing position scaling due to poor Sharpe ratio ({current_sharpe:.2f})"
            elif current_sharpe > 1.5:  # Excellent returns, can increase scaling
                optimal_value = min(current_value * 1.05, 2.5)
                rationale = f"Increasing position scaling due to excellent Sharpe ratio ({current_sharpe:.2f})"
            else:
                optimal_value = current_value
                rationale = "Current scaling factor is optimal"
                
        else:
            # Default optimization logic
            improvement_potential = analysis['optimization_potential']['estimated_improvement']
            if improvement_potential > 0.05:  # 5% improvement potential
                optimal_value = current_value * (1 + improvement_potential / 4)  # Conservative adjustment
                rationale = f"Optimization suggests {improvement_potential:.1%} improvement potential"
            else:
                optimal_value = current_value
                rationale = "No significant optimization opportunity identified"
        
        return {
            'optimal_value': round(optimal_value, 4),
            'rationale': rationale,
            'risk_assessment': {
                'risk_level': 'low' if abs(optimal_value - current_value) / current_value < 0.1 else 'moderate',
                'impact_assessment': 'minimal market impact expected',
                'rollback_complexity': 'simple'
            }
        }
    
    def _get_current_parameter_value(self, parameter_name: str) -> Optional[float]:
        """Get current value of a parameter from configuration"""
        
        # Navigate through nested configuration structure
        parts = parameter_name.split('.')
        current = self.current_config
        
        try:
            for part in parts:
                current = current[part]
            return float(current)
        except (KeyError, TypeError, ValueError):
            return None
    
    async def analyze_and_recommend(self, parameter_names: List[str] = None) -> List[RiskParameterChange]:
        """Analyze parameters and generate recommendations"""
        
        if parameter_names is None:
            # Default parameters to analyze
            parameter_names = [
                'ml_risk_management.ml_risk_thresholds.min_confidence_threshold',
                'ml_risk_management.ml_risk_thresholds.confidence_scaling_factor',
                'ml_risk_management.circuit_breakers.volatility_spike_multiplier',
            ]
        
        recommendations = []
        
        for param_name in parameter_names:
            try:
                logger.info(f"Analyzing parameter: {param_name}")
                recommendation = await self._generate_adjustment_recommendation(param_name)
                
                if recommendation:
                    recommendations.append(recommendation)
                    self.pending_changes.append(recommendation)
                    logger.info(f"Generated recommendation for {param_name}: "
                              f"{recommendation.current_value} -> {recommendation.proposed_value} "
                              f"({recommendation.change_percentage:.1f}% change)")
                else:
                    logger.info(f"No adjustment recommended for {param_name}")
                    
            except Exception as e:
                logger.error(f"Error analyzing parameter {param_name}: {e}")
        
        return recommendations
    
    def approve_change(self, change_id: str, approver: str, notes: str = "") -> bool:
        """Approve a pending parameter change"""
        
        change = next((c for c in self.pending_changes if c.change_id == change_id), None)
        if not change:
            logger.error(f"Change {change_id} not found in pending changes")
            return False
        
        # Validate approval authority
        if change.adjustment_type == AdjustmentType.MAJOR and 'risk_committee' not in approver.lower():
            logger.error(f"Major changes require risk committee approval")
            return False
        
        change.approval_status = ApprovalStatus.APPROVED
        change.approved_by = approver
        
        logger.info(f"Change {change_id} approved by {approver}. Notes: {notes}")
        return True
    
    def reject_change(self, change_id: str, rejector: str, reason: str) -> bool:
        """Reject a pending parameter change"""
        
        change = next((c for c in self.pending_changes if c.change_id == change_id), None)
        if not change:
            logger.error(f"Change {change_id} not found in pending changes")
            return False
        
        change.approval_status = ApprovalStatus.REJECTED
        change.approved_by = rejector
        
        # Remove from pending
        self.pending_changes.remove(change)
        self.change_history.append(change)
        
        logger.info(f"Change {change_id} rejected by {rejector}. Reason: {reason}")
        return True
    
    async def implement_approved_changes(self) -> List[str]:
        """Implement all approved parameter changes"""
        
        implemented_changes = []
        approved_changes = [c for c in self.pending_changes if c.approval_status == ApprovalStatus.APPROVED]
        
        for change in approved_changes:
            try:
                # Check daily adjustment limit
                today_changes = len([c for c in self.change_history 
                                   if c.implementation_timestamp and 
                                   c.implementation_timestamp.date() == datetime.now().date()])
                
                if today_changes >= self.governance_rules['max_adjustments_per_day']:
                    logger.warning(f"Daily adjustment limit reached. Skipping change {change.change_id}")
                    continue
                
                # Check cooling period
                can_adjust, msg = self._check_cooling_period()
                if not can_adjust:
                    logger.warning(f"Cooling period active. Skipping change {change.change_id}: {msg}")
                    continue
                
                # Implement the change
                success = await self._implement_parameter_change(change)
                
                if success:
                    change.approval_status = ApprovalStatus.IMPLEMENTED
                    change.implementation_timestamp = datetime.now()
                    self.last_adjustment_time = datetime.now()
                    
                    # Move to history
                    self.pending_changes.remove(change)
                    self.change_history.append(change)
                    
                    implemented_changes.append(change.change_id)
                    logger.info(f"Successfully implemented change {change.change_id}")
                    
                    # Start performance monitoring
                    asyncio.create_task(self._monitor_change_performance(change))
                    
                else:
                    logger.error(f"Failed to implement change {change.change_id}")
                    
            except Exception as e:
                logger.error(f"Error implementing change {change.change_id}: {e}")
        
        return implemented_changes
    
    async def _implement_parameter_change(self, change: RiskParameterChange) -> bool:
        """Actually implement a parameter change in the configuration"""
        
        try:
            # Create new configuration with updated parameter
            new_config = self.current_config.copy()
            
            # Navigate to parameter and update value
            parts = change.parameter_name.split('.')
            current = new_config
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = change.proposed_value
            
            # Save updated configuration
            self._save_config(new_config)
            self.current_config = new_config
            
            logger.info(f"Parameter {change.parameter_name} updated: "
                       f"{change.current_value} -> {change.proposed_value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to implement parameter change: {e}")
            return False
    
    async def _monitor_change_performance(self, change: RiskParameterChange) -> None:
        """Monitor performance after implementing a parameter change"""
        
        monitoring_duration = timedelta(hours=24)  # Monitor for 24 hours
        check_interval = timedelta(minutes=30)     # Check every 30 minutes
        
        start_time = datetime.now()
        baseline_performance = await self._get_current_performance_metrics()
        
        while datetime.now() - start_time < monitoring_duration:
            await asyncio.sleep(check_interval.total_seconds())
            
            current_performance = await self._get_current_performance_metrics()
            performance_change = self._calculate_performance_impact(baseline_performance, current_performance)
            
            # Check for performance degradation
            if performance_change['sharpe_ratio_change'] < -0.2:  # 20% degradation in Sharpe ratio
                logger.warning(f"Performance degradation detected for change {change.change_id}")
                await self._trigger_automatic_rollback(change, "Performance degradation")
                break
            
            # Update change record with performance data
            change.performance_impact = performance_change
        
        logger.info(f"Performance monitoring completed for change {change.change_id}")
    
    async def _get_current_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics (simulated)"""
        return {
            'sharpe_ratio': 1.25,
            'max_drawdown': 0.08,
            'win_rate': 0.65,
            'total_return': 0.15
        }
    
    def _calculate_performance_impact(self, baseline: Dict[str, float], current: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance impact of parameter change"""
        return {
            'sharpe_ratio_change': (current['sharpe_ratio'] - baseline['sharpe_ratio']) / baseline['sharpe_ratio'],
            'drawdown_change': (current['max_drawdown'] - baseline['max_drawdown']) / baseline['max_drawdown'],
            'win_rate_change': (current['win_rate'] - baseline['win_rate']) / baseline['win_rate'],
            'return_change': (current['total_return'] - baseline['total_return']) / baseline['total_return']
        }
    
    async def _trigger_automatic_rollback(self, change: RiskParameterChange, reason: str) -> None:
        """Automatically rollback a parameter change due to poor performance"""
        
        try:
            logger.warning(f"Triggering automatic rollback for change {change.change_id}: {reason}")
            
            # Create rollback configuration
            rollback_config = self.current_config.copy()
            parts = change.parameter_name.split('.')
            current = rollback_config
            
            for part in parts[:-1]:
                current = current[part]
            
            current[parts[-1]] = change.current_value  # Revert to original value
            
            # Save rollback configuration
            self._save_config(rollback_config)
            self.current_config = rollback_config
            
            # Update change record
            change.approval_status = ApprovalStatus.ROLLED_BACK
            change.rollback_timestamp = datetime.now()
            
            logger.info(f"Successfully rolled back change {change.change_id}")
            
        except Exception as e:
            logger.error(f"Failed to rollback change {change.change_id}: {e}")
    
    def get_change_summary(self) -> Dict[str, Any]:
        """Get summary of all parameter changes"""
        
        total_changes = len(self.change_history)
        approved_changes = len([c for c in self.change_history if c.approval_status == ApprovalStatus.APPROVED])
        implemented_changes = len([c for c in self.change_history if c.approval_status == ApprovalStatus.IMPLEMENTED])
        rolled_back_changes = len([c for c in self.change_history if c.approval_status == ApprovalStatus.ROLLED_BACK])
        
        return {
            'total_changes_proposed': total_changes,
            'approved_changes': approved_changes,
            'implemented_changes': implemented_changes,
            'rolled_back_changes': rolled_back_changes,
            'pending_changes': len(self.pending_changes),
            'approval_rate': approved_changes / total_changes if total_changes > 0 else 0,
            'success_rate': (implemented_changes - rolled_back_changes) / implemented_changes if implemented_changes > 0 else 0,
            'recent_changes': [asdict(c) for c in self.change_history[-5:]],  # Last 5 changes
            'last_adjustment_time': self.last_adjustment_time.isoformat() if self.last_adjustment_time else None
        }
    
    def generate_audit_report(self) -> str:
        """Generate comprehensive audit report for regulatory compliance"""
        
        report = []
        report.append("=" * 80)
        report.append("ML RISK PARAMETER ADJUSTMENT AUDIT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Configuration File: {self.config_path}")
        report.append("")
        
        # Summary statistics
        summary = self.get_change_summary()
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Changes Proposed: {summary['total_changes_proposed']}")
        report.append(f"Approved Changes: {summary['approved_changes']}")
        report.append(f"Implemented Changes: {summary['implemented_changes']}")
        report.append(f"Rolled Back Changes: {summary['rolled_back_changes']}")
        report.append(f"Pending Changes: {summary['pending_changes']}")
        report.append(f"Approval Rate: {summary['approval_rate']:.1%}")
        report.append(f"Success Rate: {summary['success_rate']:.1%}")
        report.append("")
        
        # Immutable limits compliance
        report.append("SAFETY LIMITS COMPLIANCE")
        report.append("-" * 40)
        for limit_name, limit_value in self.immutable_limits.items():
            current_value = self._get_current_parameter_value(f"ml_risk_management.ml_risk_thresholds.{limit_name}")
            if current_value is not None:
                compliance = "✅ COMPLIANT" if current_value <= limit_value else "❌ VIOLATION"
                report.append(f"{limit_name}: {current_value} (limit: {limit_value}) - {compliance}")
            else:
                report.append(f"{limit_name}: NOT FOUND in configuration")
        report.append("")
        
        # Recent changes detail
        report.append("RECENT CHANGES (LAST 10)")
        report.append("-" * 40)
        recent_changes = self.change_history[-10:] if len(self.change_history) >= 10 else self.change_history
        
        for change in recent_changes:
            report.append(f"Change ID: {change.change_id}")
            report.append(f"  Parameter: {change.parameter_name}")
            report.append(f"  Change: {change.current_value} -> {change.proposed_value} ({change.change_percentage:.1f}%)")
            report.append(f"  Type: {change.adjustment_type.value}")
            report.append(f"  Status: {change.approval_status.value}")
            report.append(f"  Approved By: {change.approved_by or 'N/A'}")
            report.append(f"  Rationale: {change.rationale}")
            report.append(f"  Timestamp: {change.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
        
        return "\n".join(report)

# Example usage and integration
async def main():
    """Example usage of ML Risk Parameter Adjuster"""
    
    # Initialize the adjuster
    adjuster = MLRiskParameterAdjuster('config/ml_risk_config.yaml')
    
    # Analyze parameters and generate recommendations
    print("🔍 Analyzing risk parameters...")
    recommendations = await adjuster.analyze_and_recommend()
    
    print(f"\n📋 Generated {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"  • {rec.parameter_name}: {rec.current_value} -> {rec.proposed_value}")
        print(f"    Rationale: {rec.rationale}")
        print(f"    Type: {rec.adjustment_type.value}")
        print(f"    Status: {rec.approval_status.value}")
        print()
    
    # Simulate approvals (in real system, would be done by authorized personnel)
    for rec in recommendations:
        if rec.adjustment_type == AdjustmentType.MINOR:
            # Auto-approve minor changes
            adjuster.approve_change(rec.change_id, "system_auto_approval", "Minor adjustment within limits")
        elif rec.adjustment_type == AdjustmentType.MODERATE:
            # Simulate risk manager approval
            adjuster.approve_change(rec.change_id, "risk_manager_john_doe", "Approved after review")
    
    # Implement approved changes
    print("\n⚡ Implementing approved changes...")
    implemented = await adjuster.implement_approved_changes()
    print(f"✅ Implemented {len(implemented)} changes")
    
    # Generate summary and audit report
    print("\n📊 Change Summary:")
    summary = adjuster.get_change_summary()
    print(f"  • Total changes: {summary['total_changes_proposed']}")
    print(f"  • Success rate: {summary['success_rate']:.1%}")
    print(f"  • Approval rate: {summary['approval_rate']:.1%}")
    
    print("\n📄 Generating audit report...")
    audit_report = adjuster.generate_audit_report()
    
    # Save audit report
    with open('ml_risk_adjustment_audit.txt', 'w') as f:
        f.write(audit_report)
    
    print("✅ Audit report saved to ml_risk_adjustment_audit.txt")
    print("\n🎯 ML Risk Parameter Adjustment System operational!")

if __name__ == "__main__":
    asyncio.run(main())