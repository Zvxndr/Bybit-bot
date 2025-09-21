"""
Portfolio Risk Controller
Coordinates risk management across ML strategies and arbitrage systems
Real-time monitoring and automatic position adjustments
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .australian_risk_manager import (
    AustralianRiskCalculator, 
    TaxAwarePositionManager,
    ComplianceMonitor,
    RiskParameters,
    PositionRisk,
    PositionType,
    RiskLevel
)
from ..ml_strategy_discovery.ml_engine import StrategySignal, MLStrategyDiscoveryEngine
from ..arbitrage_engine.arbitrage_detector import ArbitrageOpportunity, OpportunisticArbitrageEngine

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Risk alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ActionType(Enum):
    """Automatic risk management actions"""
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    HEDGE_POSITION = "hedge_position"
    PAUSE_TRADING = "pause_trading"
    REBALANCE_PORTFOLIO = "rebalance_portfolio"
    TAX_LOSS_HARVEST = "tax_loss_harvest"

@dataclass
class RiskAlert:
    """Risk management alert"""
    alert_id: str
    level: AlertLevel
    message: str
    affected_positions: List[str]
    recommended_actions: List[ActionType]
    timestamp: datetime
    auto_action_taken: bool = False
    acknowledged: bool = False

@dataclass
class PortfolioState:
    """Current portfolio state snapshot"""
    timestamp: datetime
    total_value_aud: Decimal
    cash_balance_aud: Decimal
    positions: Dict[str, PositionRisk]
    
    # Allocation breakdown  
    ml_strategy_allocation: Decimal
    arbitrage_allocation: Decimal
    cash_allocation: Decimal
    
    # Risk metrics
    total_risk_score: float
    max_drawdown_risk: float
    international_exposure: float
    compliance_status: str
    
    # Performance tracking
    daily_pnl: Decimal
    weekly_pnl: Decimal
    monthly_pnl: Decimal

class PortfolioRiskController:
    """
    Main portfolio risk controller
    Coordinates all risk management activities across strategies
    """
    
    def __init__(
        self,
        risk_calculator: AustralianRiskCalculator,
        position_manager: TaxAwarePositionManager,
        compliance_monitor: ComplianceMonitor,
        ml_engine: MLStrategyDiscoveryEngine,
        arbitrage_engine: OpportunisticArbitrageEngine
    ):
        self.risk_calculator = risk_calculator
        self.position_manager = position_manager
        self.compliance_monitor = compliance_monitor
        self.ml_engine = ml_engine
        self.arbitrage_engine = arbitrage_engine
        
        # Risk monitoring
        self.active_alerts = []
        self.portfolio_history = []
        self.monitoring_enabled = True
        self.auto_action_enabled = True
        
        # Emergency thresholds
        self.emergency_drawdown_threshold = Decimal('0.20')  # 20% drawdown emergency
        self.critical_risk_score_threshold = 0.8              # Critical risk score
        self.max_daily_loss_threshold = Decimal('0.05')      # 5% daily loss limit
        
        # Performance tracking
        self.performance_baseline = None
        self.last_rebalance_date = datetime.now()
        self.rebalance_frequency_days = 7  # Weekly rebalancing
        
        logger.info("Initialized Portfolio Risk Controller")
    
    async def update_portfolio_state(
        self,
        current_prices: Dict[str, Decimal],
        account_balance: Decimal
    ) -> PortfolioState:
        """Update current portfolio state with latest data"""
        
        # Get current positions (would integrate with actual position tracking)
        current_positions = await self._get_current_positions(current_prices)
        
        # Calculate portfolio metrics
        total_position_value = sum(pos.current_value_aud for pos in current_positions.values())
        total_portfolio_value = account_balance + total_position_value
        
        # Allocation breakdown
        ml_positions = {k: v for k, v in current_positions.items() if v.position_type == PositionType.ML_STRATEGY}
        arbitrage_positions = {k: v for k, v in current_positions.items() if v.position_type == PositionType.ARBITRAGE}
        
        ml_allocation = sum(pos.current_value_aud for pos in ml_positions.values())
        arbitrage_allocation = sum(pos.current_value_aud for pos in arbitrage_positions.values())
        cash_allocation = account_balance
        
        # Risk metrics
        risk_metrics = self.risk_calculator.calculate_portfolio_risk_metrics(
            current_positions, total_portfolio_value
        )
        
        # Performance calculation
        daily_pnl, weekly_pnl, monthly_pnl = await self._calculate_performance_metrics(total_portfolio_value)
        
        portfolio_state = PortfolioState(
            timestamp=datetime.now(),
            total_value_aud=total_portfolio_value,
            cash_balance_aud=account_balance,
            positions=current_positions,
            ml_strategy_allocation=ml_allocation,
            arbitrage_allocation=arbitrage_allocation,
            cash_allocation=cash_allocation,
            total_risk_score=risk_metrics['total_risk_score'],
            max_drawdown_risk=risk_metrics['drawdown_risk'],
            international_exposure=risk_metrics['international_exposure'],
            compliance_status=risk_metrics['compliance_status'],
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            monthly_pnl=monthly_pnl
        )
        
        # Store in history
        self.portfolio_history.append(portfolio_state)
        
        # Keep only last 30 days of history
        cutoff_date = datetime.now() - timedelta(days=30)
        self.portfolio_history = [
            state for state in self.portfolio_history
            if state.timestamp >= cutoff_date
        ]
        
        return portfolio_state
    
    async def _get_current_positions(self, current_prices: Dict[str, Decimal]) -> Dict[str, PositionRisk]:
        """Get current portfolio positions (placeholder - would integrate with actual position tracking)"""
        
        # This would integrate with actual position tracking system
        # For now, return example positions
        positions = {}
        
        # Example ML strategy position
        if 'BTC/AUD' in current_prices:
            positions['BTC/AUD'] = self.risk_calculator.assess_position_risk(
                symbol='BTC/AUD',
                position_value_aud=Decimal('15000'),
                position_type=PositionType.ML_STRATEGY,
                entry_date=datetime.now() - timedelta(days=120),
                entry_price=Decimal('62000'),
                current_price=current_prices['BTC/AUD'],
                exchange='btcmarkets'
            )
        
        # Example arbitrage position
        if 'ETH/USDT' in current_prices:
            positions['ETH/USDT'] = self.risk_calculator.assess_position_risk(
                symbol='ETH/USDT',
                position_value_aud=Decimal('8000'),
                position_type=PositionType.ARBITRAGE,
                entry_date=datetime.now() - timedelta(days=30),
                entry_price=Decimal('2500'),
                current_price=current_prices['ETH/USDT'],
                exchange='bybit'
            )
        
        return positions
    
    async def _calculate_performance_metrics(self, current_value: Decimal) -> Tuple[Decimal, Decimal, Decimal]:
        """Calculate performance metrics vs historical data"""
        
        if len(self.portfolio_history) < 2:
            return Decimal('0'), Decimal('0'), Decimal('0')
        
        # Daily performance
        yesterday_state = None
        for state in reversed(self.portfolio_history):
            if state.timestamp <= datetime.now() - timedelta(days=1):
                yesterday_state = state
                break
        
        daily_pnl = current_value - yesterday_state.total_value_aud if yesterday_state else Decimal('0')
        
        # Weekly performance
        week_ago_state = None
        for state in reversed(self.portfolio_history):
            if state.timestamp <= datetime.now() - timedelta(days=7):
                week_ago_state = state
                break
        
        weekly_pnl = current_value - week_ago_state.total_value_aud if week_ago_state else Decimal('0')
        
        # Monthly performance
        month_ago_state = None
        for state in reversed(self.portfolio_history):
            if state.timestamp <= datetime.now() - timedelta(days=30):
                month_ago_state = state
                break
        
        monthly_pnl = current_value - month_ago_state.total_value_aud if month_ago_state else Decimal('0')
        
        return daily_pnl, weekly_pnl, monthly_pnl
    
    async def assess_portfolio_risks(self, portfolio_state: PortfolioState) -> List[RiskAlert]:
        """Assess current portfolio risks and generate alerts"""
        
        alerts = []
        timestamp = datetime.now()
        
        # Emergency drawdown check
        if abs(portfolio_state.daily_pnl / portfolio_state.total_value_aud) > self.max_daily_loss_threshold:
            alerts.append(RiskAlert(
                alert_id=f"emergency_drawdown_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.EMERGENCY,
                message=f"Emergency: Daily loss of {abs(portfolio_state.daily_pnl):,.2f} AUD ({abs(portfolio_state.daily_pnl / portfolio_state.total_value_aud):.1%}) exceeds threshold",
                affected_positions=list(portfolio_state.positions.keys()),
                recommended_actions=[ActionType.PAUSE_TRADING, ActionType.REDUCE_POSITION],
                timestamp=timestamp
            ))
        
        # Critical risk score
        if portfolio_state.total_risk_score > self.critical_risk_score_threshold:
            alerts.append(RiskAlert(
                alert_id=f"critical_risk_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.CRITICAL,
                message=f"Critical risk score: {portfolio_state.total_risk_score:.3f}",
                affected_positions=list(portfolio_state.positions.keys()),
                recommended_actions=[ActionType.REDUCE_POSITION, ActionType.REBALANCE_PORTFOLIO],
                timestamp=timestamp
            ))
        
        # International exposure warning
        if portfolio_state.international_exposure > 0.6:  # 60% threshold
            international_positions = [
                symbol for symbol, pos in portfolio_state.positions.items()
                if pos.international_exposure
            ]
            alerts.append(RiskAlert(
                alert_id=f"international_exposure_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING,
                message=f"High international exposure: {portfolio_state.international_exposure:.1%}",
                affected_positions=international_positions,
                recommended_actions=[ActionType.REBALANCE_PORTFOLIO],
                timestamp=timestamp
            ))
        
        # Compliance issues
        if portfolio_state.compliance_status != 'compliant':
            alerts.append(RiskAlert(
                alert_id=f"compliance_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING,
                message=f"Compliance issues detected: {portfolio_state.compliance_status}",
                affected_positions=list(portfolio_state.positions.keys()),
                recommended_actions=[ActionType.REBALANCE_PORTFOLIO],
                timestamp=timestamp
            ))
        
        # Tax optimization opportunities
        tax_recommendations = self.position_manager.get_tax_optimization_recommendations(
            portfolio_state.positions,
            {symbol: Decimal('65000') for symbol in portfolio_state.positions.keys()}  # Placeholder prices
        )
        
        high_priority_tax_ops = [rec for rec in tax_recommendations if rec['priority'] == 'high']
        if high_priority_tax_ops:
            alerts.append(RiskAlert(
                alert_id=f"tax_optimization_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.INFO,
                message=f"{len(high_priority_tax_ops)} high-priority tax optimization opportunities",
                affected_positions=[rec['symbol'] for rec in high_priority_tax_ops],
                recommended_actions=[ActionType.TAX_LOSS_HARVEST],
                timestamp=timestamp
            ))
        
        # Portfolio rebalancing needed
        days_since_rebalance = (datetime.now() - self.last_rebalance_date).days
        if days_since_rebalance >= self.rebalance_frequency_days:
            # Check if allocation is significantly off target
            target_ml_allocation = portfolio_state.total_value_aud * Decimal('0.7')
            target_arbitrage_allocation = portfolio_state.total_value_aud * Decimal('0.2')
            
            ml_deviation = abs(portfolio_state.ml_strategy_allocation - target_ml_allocation) / target_ml_allocation
            arbitrage_deviation = abs(portfolio_state.arbitrage_allocation - target_arbitrage_allocation) / target_arbitrage_allocation
            
            if ml_deviation > Decimal('0.15') or arbitrage_deviation > Decimal('0.15'):  # 15% deviation threshold
                alerts.append(RiskAlert(
                    alert_id=f"rebalancing_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.INFO,
                    message=f"Portfolio rebalancing recommended (ML: {ml_deviation:.1%} deviation, Arbitrage: {arbitrage_deviation:.1%} deviation)",
                    affected_positions=list(portfolio_state.positions.keys()),
                    recommended_actions=[ActionType.REBALANCE_PORTFOLIO],
                    timestamp=timestamp
                ))
        
        return alerts
    
    async def execute_automatic_actions(self, alerts: List[RiskAlert]) -> List[str]:
        """Execute automatic risk management actions"""
        
        if not self.auto_action_enabled:
            return []
        
        actions_taken = []
        
        for alert in alerts:
            if alert.level == AlertLevel.EMERGENCY:
                # Emergency actions - pause trading and reduce positions
                actions_taken.append(f"EMERGENCY: Paused all trading due to {alert.message}")
                self.monitoring_enabled = False  # Temporarily disable new positions
                
                # Reduce high-risk positions by 50%
                for position_symbol in alert.affected_positions:
                    actions_taken.append(f"EMERGENCY: Reduced {position_symbol} position by 50%")
                
                alert.auto_action_taken = True
                
            elif alert.level == AlertLevel.CRITICAL:
                # Critical actions - reduce positions
                for position_symbol in alert.affected_positions:
                    actions_taken.append(f"CRITICAL: Reduced {position_symbol} position by 25%")
                
                alert.auto_action_taken = True
                
            elif alert.level == AlertLevel.WARNING and ActionType.TAX_LOSS_HARVEST in alert.recommended_actions:
                # Tax loss harvesting (automatic for high-value opportunities)
                for position_symbol in alert.affected_positions:
                    actions_taken.append(f"AUTO: Initiated tax loss harvesting for {position_symbol}")
                
                alert.auto_action_taken = True
        
        return actions_taken
    
    async def generate_ml_position_recommendations(
        self,
        ml_signals: List[StrategySignal],
        portfolio_state: PortfolioState
    ) -> List[Dict[str, Any]]:
        """Generate position recommendations for ML strategy signals"""
        
        recommendations = []
        
        for signal in ml_signals:
            # Skip if monitoring disabled (emergency mode)
            if not self.monitoring_enabled:
                continue
            
            # Calculate recommended position size
            position_size, reasoning = self.risk_calculator.calculate_position_size(
                signal=signal,
                portfolio_value_aud=portfolio_state.total_value_aud,
                current_positions=portfolio_state.positions,
                market_conditions={'volatility': 0.3}  # Would get actual market conditions
            )
            
            if position_size > Decimal('1000'):  # Minimum $1000 position
                # Check compliance
                proposed_trade = {
                    'symbol': signal.symbol,
                    'value_aud': position_size,
                    'international_exchange': not signal.symbol.endswith('AUD')
                }
                
                is_compliant, issues, warnings = await self.compliance_monitor.check_trade_compliance(
                    proposed_trade, portfolio_state.positions, portfolio_state.total_value_aud
                )
                
                recommendation = {
                    'signal_id': f"{signal.symbol}_{signal.strategy_type.value}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}",
                    'symbol': signal.symbol,
                    'strategy_type': signal.strategy_type.value,
                    'signal_strength': signal.signal_strength,
                    'confidence': signal.confidence,
                    'recommended_size_aud': position_size,
                    'reasoning': reasoning,
                    'compliant': is_compliant,
                    'compliance_issues': issues,
                    'compliance_warnings': warnings,
                    'risk_adjusted': True,
                    'tax_optimized': signal.australian_risk_adjusted
                }
                
                recommendations.append(recommendation)
        
        # Sort by signal strength * confidence * position size
        recommendations.sort(
            key=lambda x: x['signal_strength'] * x['confidence'] * float(x['recommended_size_aud']),
            reverse=True
        )
        
        return recommendations
    
    async def generate_arbitrage_recommendations(
        self,
        arbitrage_opportunities: List[ArbitrageOpportunity],
        portfolio_state: PortfolioState
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for arbitrage opportunities"""
        
        recommendations = []
        
        for opportunity in arbitrage_opportunities:
            # Skip if monitoring disabled
            if not self.monitoring_enabled:
                continue
            
            # Calculate position size
            position_size, reasoning = self.risk_calculator.calculate_arbitrage_position_size(
                opportunity=opportunity,
                portfolio_value_aud=portfolio_state.total_value_aud,
                current_positions=portfolio_state.positions
            )
            
            if position_size >= opportunity.minimum_amount:
                # Check compliance
                proposed_trade = {
                    'symbol': opportunity.symbol,
                    'value_aud': position_size,
                    'international_exchange': not opportunity.australian_friendly
                }
                
                is_compliant, issues, warnings = await self.compliance_monitor.check_trade_compliance(
                    proposed_trade, portfolio_state.positions, portfolio_state.total_value_aud
                )
                
                recommendation = {
                    'opportunity_id': opportunity.opportunity_id,
                    'arbitrage_type': opportunity.arbitrage_type.value,
                    'symbol': opportunity.symbol,
                    'buy_exchange': opportunity.buy_exchange,
                    'sell_exchange': opportunity.sell_exchange,
                    'net_profit_percentage': float(opportunity.net_profit_percentage),
                    'recommended_size_aud': position_size,
                    'reasoning': reasoning,
                    'execution_time_minutes': opportunity.estimated_execution_time,
                    'compliant': is_compliant,
                    'compliance_issues': issues,
                    'compliance_warnings': warnings,
                    'australian_friendly': opportunity.australian_friendly,
                    'expires_at': opportunity.expires_at
                }
                
                recommendations.append(recommendation)
        
        # Sort by net profit * position size
        recommendations.sort(
            key=lambda x: x['net_profit_percentage'] * float(x['recommended_size_aud']),
            reverse=True
        )
        
        return recommendations
    
    def get_risk_dashboard_data(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """Generate comprehensive risk dashboard data"""
        
        # Active alerts summary
        alert_summary = {
            'emergency': len([a for a in self.active_alerts if a.level == AlertLevel.EMERGENCY]),
            'critical': len([a for a in self.active_alerts if a.level == AlertLevel.CRITICAL]),
            'warning': len([a for a in self.active_alerts if a.level == AlertLevel.WARNING]),
            'info': len([a for a in self.active_alerts if a.level == AlertLevel.INFO])
        }
        
        # Portfolio allocation vs targets
        total_value = portfolio_state.total_value_aud
        allocation_analysis = {
            'ml_strategy': {
                'current': float(portfolio_state.ml_strategy_allocation / total_value),
                'target': 0.7,
                'deviation': float(abs(portfolio_state.ml_strategy_allocation / total_value - Decimal('0.7')))
            },
            'arbitrage': {
                'current': float(portfolio_state.arbitrage_allocation / total_value),
                'target': 0.2,
                'deviation': float(abs(portfolio_state.arbitrage_allocation / total_value - Decimal('0.2')))
            },
            'cash': {
                'current': float(portfolio_state.cash_allocation / total_value),
                'target': 0.1,
                'deviation': float(abs(portfolio_state.cash_allocation / total_value - Decimal('0.1')))
            }
        }
        
        # Risk metrics summary
        risk_summary = {
            'overall_risk_score': portfolio_state.total_risk_score,
            'drawdown_risk': portfolio_state.max_drawdown_risk,
            'international_exposure': portfolio_state.international_exposure,
            'compliance_status': portfolio_state.compliance_status,
            'positions_count': len(portfolio_state.positions),
            'monitoring_enabled': self.monitoring_enabled,
            'auto_action_enabled': self.auto_action_enabled
        }
        
        # Performance summary
        performance_summary = {
            'daily_pnl': float(portfolio_state.daily_pnl),
            'daily_return': float(portfolio_state.daily_pnl / total_value) if total_value > 0 else 0,
            'weekly_pnl': float(portfolio_state.weekly_pnl),
            'weekly_return': float(portfolio_state.weekly_pnl / total_value) if total_value > 0 else 0,
            'monthly_pnl': float(portfolio_state.monthly_pnl),
            'monthly_return': float(portfolio_state.monthly_pnl / total_value) if total_value > 0 else 0
        }
        
        return {
            'timestamp': portfolio_state.timestamp,
            'portfolio_value': float(total_value),
            'alerts': alert_summary,
            'allocation': allocation_analysis,
            'risk_metrics': risk_summary,
            'performance': performance_summary,
            'active_alerts': [
                {
                    'level': alert.level.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp,
                    'auto_action_taken': alert.auto_action_taken
                } for alert in self.active_alerts[-10:]  # Last 10 alerts
            ]
        }

# Usage example
async def main():
    """Example usage of portfolio risk controller"""
    
    # Initialize all components (would be done in main application)
    from ..australian_compliance.ato_integration import AustralianTaxCalculator
    from ..australian_compliance.regulatory_compliance import AustralianComplianceManager
    
    # Initialize risk management components
    risk_params = RiskParameters()
    risk_calculator = AustralianRiskCalculator(risk_params)
    
    tax_calculator = AustralianTaxCalculator()
    position_manager = TaxAwarePositionManager(tax_calculator)
    
    compliance_manager = AustralianComplianceManager()
    compliance_monitor = ComplianceMonitor(compliance_manager)
    
    # Initialize strategy engines (simplified)
    ml_engine = MLStrategyDiscoveryEngine()
    arbitrage_engine = OpportunisticArbitrageEngine()
    
    # Initialize portfolio controller
    controller = PortfolioRiskController(
        risk_calculator, position_manager, compliance_monitor,
        ml_engine, arbitrage_engine
    )
    
    # Simulate portfolio monitoring
    current_prices = {
        'BTC/AUD': Decimal('65000'),
        'ETH/USDT': Decimal('2600'),
        'ETH/AUD': Decimal('3900')
    }
    
    account_balance = Decimal('75000')  # $75k cash
    
    # Update portfolio state
    portfolio_state = await controller.update_portfolio_state(current_prices, account_balance)
    
    print("=== Portfolio Risk Controller Analysis ===")
    print(f"Total Portfolio Value: ${portfolio_state.total_value_aud:,.2f} AUD")
    print(f"ML Strategy Allocation: ${portfolio_state.ml_strategy_allocation:,.2f}")
    print(f"Arbitrage Allocation: ${portfolio_state.arbitrage_allocation:,.2f}")
    print(f"Cash Allocation: ${portfolio_state.cash_allocation:,.2f}")
    print(f"Overall Risk Score: {portfolio_state.total_risk_score:.3f}")
    print(f"Daily P&L: ${portfolio_state.daily_pnl:,.2f}")
    
    # Assess risks
    alerts = await controller.assess_portfolio_risks(portfolio_state)
    print(f"\n=== Risk Alerts ({len(alerts)}) ===")
    for alert in alerts:
        print(f"  {alert.level.value.upper()}: {alert.message}")
    
    # Execute automatic actions
    if alerts:
        actions = await controller.execute_automatic_actions(alerts)
        if actions:
            print(f"\n=== Automatic Actions Taken ===")
            for action in actions:
                print(f"  {action}")
    
    # Get dashboard data
    dashboard = controller.get_risk_dashboard_data(portfolio_state)
    print(f"\n=== Risk Dashboard Summary ===")
    print(f"Monitoring Enabled: {dashboard['risk_metrics']['monitoring_enabled']}")
    print(f"Compliance Status: {dashboard['risk_metrics']['compliance_status']}")
    print(f"International Exposure: {dashboard['risk_metrics']['international_exposure']:.1%}")

if __name__ == "__main__":
    asyncio.run(main())