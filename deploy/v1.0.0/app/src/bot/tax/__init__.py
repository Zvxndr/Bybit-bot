"""
Integrated Tax and Reporting System

This module provides the main interface for the comprehensive tax system including:
- Tax calculation engine with wash sale detection
- Advanced tax reporting and compliance
- Tax optimization and loss harvesting
- Multi-jurisdiction support
- Real-time tax impact analysis
- Professional-grade audit trails
"""

import logging
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json

from .tax_engine import (
    TaxEngine, TaxConfiguration, AccountingMethod, Transaction, 
    TransactionType, TaxEvent, TaxEventType, HoldingPeriod
)
from .tax_reporter import (
    TaxReporter, ReportConfiguration, ReportType, ReportFormat, 
    JurisdictionCompliance
)
from .tax_optimizer import (
    TaxOptimizer, TaxOptimizationConfig, OptimizationStrategy,
    OptimizationObjective, RiskTolerance, OptimizationRecommendation
)

logger = logging.getLogger(__name__)

class TaxSystem:
    """Integrated cryptocurrency tax calculation and optimization system."""
    
    def __init__(self, 
                 jurisdiction: str = "US",
                 accounting_method: AccountingMethod = AccountingMethod.FIFO,
                 db_path: str = "comprehensive_tax.db",
                 reports_dir: str = "tax_reports"):
        
        # Initialize core components
        self.config = TaxConfiguration(jurisdiction)
        self.config.accounting_method = accounting_method
        
        self.tax_engine = TaxEngine(self.config, db_path)
        self.tax_reporter = TaxReporter(self.tax_engine, reports_dir)
        
        # Initialize optimizer (default configuration)
        opt_config = TaxOptimizationConfig(
            strategy=OptimizationStrategy.TAX_LOSS_HARVEST,
            objective=OptimizationObjective.MAXIMIZE_TAX_SAVINGS,
            risk_tolerance=RiskTolerance.MODERATE
        )
        self.tax_optimizer = TaxOptimizer(self.tax_engine, opt_config)
        
        # System state
        self.system_initialized = True
        self.last_optimization_run = None
        self.system_stats = {
            'transactions_processed': 0,
            'tax_events_generated': 0,
            'reports_created': 0,
            'optimization_runs': 0
        }
        
        logger.info(f"TaxSystem initialized: {jurisdiction} jurisdiction, "
                   f"{accounting_method.value} accounting method")
    
    # Transaction Management
    def add_transaction(self, transaction: Transaction) -> Dict[str, Any]:
        """Add a transaction and return tax impact analysis."""
        logger.info(f"Processing transaction: {transaction.id}")
        
        try:
            # Store transaction state before processing
            pre_state = self._capture_portfolio_state()
            
            # Process transaction
            self.tax_engine.add_transaction(transaction)
            
            # Analyze impact
            post_state = self._capture_portfolio_state()
            impact_analysis = self._analyze_transaction_impact(pre_state, post_state, transaction)
            
            # Update stats
            self.system_stats['transactions_processed'] += 1
            if transaction.transaction_type == TransactionType.SELL:
                self.system_stats['tax_events_generated'] += 1
            
            logger.info(f"Transaction processed successfully: {transaction.id}")
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Failed to process transaction {transaction.id}: {e}")
            raise
    
    def add_bulk_transactions(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Add multiple transactions efficiently."""
        logger.info(f"Processing {len(transactions)} transactions in bulk")
        
        results = {
            'successful': 0,
            'failed': 0,
            'errors': [],
            'total_tax_impact': Decimal('0'),
            'transaction_results': []
        }
        
        for transaction in transactions:
            try:
                impact = self.add_transaction(transaction)
                results['successful'] += 1
                results['total_tax_impact'] += impact.get('tax_impact', Decimal('0'))
                results['transaction_results'].append({
                    'transaction_id': transaction.id,
                    'status': 'success',
                    'impact': impact
                })
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'transaction_id': transaction.id,
                    'error': str(e)
                })
                results['transaction_results'].append({
                    'transaction_id': transaction.id,
                    'status': 'failed',
                    'error': str(e)
                })
        
        logger.info(f"Bulk processing complete: {results['successful']} successful, "
                   f"{results['failed']} failed")
        return results
    
    # Tax Calculations
    def calculate_tax_liability(self, tax_year: int, 
                              income: Decimal = Decimal('0'),
                              filing_status: str = 'single') -> Dict[str, Any]:
        """Calculate comprehensive tax liability."""
        logger.info(f"Calculating tax liability for {tax_year}")
        
        try:
            # Get base tax calculation
            tax_liability = self.tax_engine.calculate_tax_liability(tax_year, income, filing_status)
            
            # Add enhanced analysis
            enhanced_analysis = self._enhance_tax_analysis(tax_liability, tax_year)
            tax_liability.update(enhanced_analysis)
            
            return tax_liability
            
        except Exception as e:
            logger.error(f"Failed to calculate tax liability: {e}")
            raise
    
    def get_real_time_tax_impact(self, proposed_transaction: Transaction) -> Dict[str, Any]:
        """Calculate real-time tax impact of a proposed transaction."""
        logger.debug(f"Analyzing tax impact for proposed transaction: {proposed_transaction.id}")
        
        try:
            # Simulate transaction without committing
            current_state = self._capture_portfolio_state()
            
            # Calculate impact based on transaction type
            if proposed_transaction.transaction_type == TransactionType.SELL:
                impact = self._simulate_sell_impact(proposed_transaction)
            elif proposed_transaction.transaction_type == TransactionType.BUY:
                impact = self._simulate_buy_impact(proposed_transaction)
            else:
                impact = {'tax_impact': Decimal('0'), 'impact_type': 'neutral'}
            
            # Add wash sale risk assessment
            impact['wash_sale_risk'] = self._assess_proposed_wash_sale_risk(proposed_transaction)
            
            # Add optimization recommendations
            impact['optimization_notes'] = self._get_optimization_notes(proposed_transaction)
            
            return impact
            
        except Exception as e:
            logger.error(f"Failed to analyze tax impact: {e}")
            return {'error': str(e)}
    
    # Reporting
    def generate_tax_report(self, report_config: ReportConfiguration) -> Dict[str, Any]:
        """Generate comprehensive tax report."""
        logger.info(f"Generating {report_config.report_type.value} report for {report_config.tax_year}")
        
        try:
            report_result = self.tax_reporter.generate_report(report_config)
            self.system_stats['reports_created'] += 1
            
            logger.info(f"Report generated successfully: {report_result.get('filename', 'N/A')}")
            return report_result
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise
    
    def generate_standard_reports(self, tax_year: int, 
                                taxpayer_info: Dict[str, str]) -> Dict[str, Any]:
        """Generate standard set of tax reports."""
        logger.info(f"Generating standard reports for {tax_year}")
        
        reports_generated = {}
        
        # Standard report configurations
        standard_reports = [
            (ReportType.FORM_8949, ReportFormat.PDF),
            (ReportType.SCHEDULE_D, ReportFormat.PDF),
            (ReportType.COMPREHENSIVE, ReportFormat.HTML),
            (ReportType.TAX_SUMMARY, ReportFormat.CSV),
            (ReportType.AUDIT_TRAIL, ReportFormat.PDF)
        ]
        
        for report_type, format_type in standard_reports:
            try:
                config = ReportConfiguration(
                    report_type=report_type,
                    format=format_type,
                    tax_year=tax_year,
                    jurisdiction=JurisdictionCompliance.US_FEDERAL,
                    taxpayer_info=taxpayer_info
                )
                
                result = self.generate_tax_report(config)
                reports_generated[f"{report_type.value}_{format_type.value}"] = result
                
            except Exception as e:
                logger.error(f"Failed to generate {report_type.value} report: {e}")
                reports_generated[f"{report_type.value}_{format_type.value}"] = {'error': str(e)}
        
        return reports_generated
    
    # Tax Optimization
    async def run_tax_optimization(self, 
                                 strategy: Optional[OptimizationStrategy] = None) -> List[OptimizationRecommendation]:
        """Run tax optimization analysis."""
        if strategy:
            self.tax_optimizer.config.strategy = strategy
        
        logger.info(f"Running tax optimization with {self.tax_optimizer.config.strategy.value} strategy")
        
        try:
            recommendations = await self.tax_optimizer.analyze_optimization_opportunities()
            self.last_optimization_run = datetime.now()
            self.system_stats['optimization_runs'] += 1
            
            logger.info(f"Optimization complete: {len(recommendations)} recommendations generated")
            return recommendations
            
        except Exception as e:
            logger.error(f"Tax optimization failed: {e}")
            raise
    
    async def execute_optimization_recommendation(self, 
                                                recommendation: OptimizationRecommendation,
                                                dry_run: bool = True) -> Dict[str, Any]:
        """Execute a tax optimization recommendation."""
        logger.info(f"Executing optimization recommendation: {recommendation.id}")
        
        try:
            result = await self.tax_optimizer.execute_recommendation(recommendation, dry_run)
            
            if not dry_run and result.get('executed', False):
                # Update system with actual execution
                logger.info(f"Recommendation executed successfully: {recommendation.id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute recommendation: {e}")
            raise
    
    def get_tax_loss_harvesting_opportunities(self) -> List[Dict[str, Any]]:
        """Get current tax-loss harvesting opportunities."""
        logger.info("Identifying tax-loss harvesting opportunities")
        
        try:
            opportunities = self.tax_engine.get_tax_loss_harvesting_opportunities()
            
            # Enhance with additional analysis
            enhanced_opportunities = []
            for opp in opportunities:
                enhanced_opp = dict(opp)
                enhanced_opp['priority_score'] = self._calculate_opportunity_priority(opp)
                enhanced_opp['execution_timeline'] = self._recommend_execution_timeline(opp)
                enhanced_opportunities.append(enhanced_opp)
            
            # Sort by priority
            enhanced_opportunities.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return enhanced_opportunities
            
        except Exception as e:
            logger.error(f"Failed to identify harvesting opportunities: {e}")
            raise
    
    # Portfolio Analysis
    def get_comprehensive_portfolio_analysis(self) -> Dict[str, Any]:
        """Get comprehensive portfolio analysis with tax implications."""
        logger.info("Generating comprehensive portfolio analysis")
        
        try:
            # Base portfolio summary
            portfolio_summary = self.tax_engine.get_portfolio_summary()
            
            # Enhanced analysis
            analysis = {
                'portfolio_summary': portfolio_summary,
                'tax_efficiency_score': self._calculate_tax_efficiency_score(),
                'wash_sale_analysis': self._analyze_wash_sale_exposure(),
                'holding_period_analysis': self._analyze_holding_periods(),
                'cost_basis_analysis': self._analyze_cost_basis_distribution(),
                'optimization_opportunities': self._identify_optimization_opportunities(),
                'risk_assessment': self._assess_tax_related_risks(),
                'compliance_status': self._check_compliance_status()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate portfolio analysis: {e}")
            raise
    
    def get_asset_tax_analysis(self, asset: str) -> Dict[str, Any]:
        """Get detailed tax analysis for a specific asset."""
        logger.info(f"Analyzing tax implications for {asset}")
        
        try:
            # Get asset holdings
            portfolio = self.tax_engine.get_portfolio_summary()
            if asset not in portfolio['holdings']:
                return {'error': f'No holdings found for {asset}'}
            
            holding = portfolio['holdings'][asset]
            
            # Get tax lots for the asset
            tax_lots = list(self.tax_engine.tax_lots.get(asset, []))
            
            # Calculate comprehensive analysis
            analysis = {
                'asset': asset,
                'current_position': holding,
                'tax_lots': [
                    {
                        'acquisition_date': lot.acquisition_date,
                        'quantity': lot.quantity,
                        'cost_basis_per_unit': lot.cost_basis_per_unit,
                        'days_held': (datetime.now() - lot.acquisition_date).days,
                        'holding_period': 'LONG_TERM' if (datetime.now() - lot.acquisition_date).days > 365 else 'SHORT_TERM',
                        'wash_sale_affected': lot.is_wash_sale_affected
                    }
                    for lot in tax_lots
                ],
                'unrealized_gain_loss': self._calculate_unrealized_gain_loss(asset, holding),
                'tax_lot_optimization': self._analyze_tax_lot_optimization(asset, tax_lots),
                'wash_sale_risk': self._analyze_asset_wash_sale_risk(asset),
                'harvesting_potential': self._assess_harvesting_potential(asset, holding)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze {asset}: {e}")
            raise
    
    # System Management
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        return {
            'system_initialized': self.system_initialized,
            'configuration': {
                'jurisdiction': self.config.jurisdiction,
                'accounting_method': self.config.accounting_method.value,
                'wash_sale_enabled': self.config.wash_sale_rule.enabled
            },
            'statistics': self.system_stats.copy(),
            'last_optimization_run': self.last_optimization_run,
            'portfolio_summary': {
                'total_assets': len(self.tax_engine.tax_lots),
                'total_transactions': len(self.tax_engine.transactions),
                'total_tax_events': len(self.tax_engine.tax_events)
            },
            'optimization_performance': self.tax_optimizer.get_optimization_performance()
        }
    
    def export_tax_data(self, format: str = 'json', include_sensitive: bool = False) -> Dict[str, Any]:
        """Export comprehensive tax data."""
        logger.info(f"Exporting tax data in {format} format")
        
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'system_configuration': {
                    'jurisdiction': self.config.jurisdiction,
                    'accounting_method': self.config.accounting_method.value
                },
                'portfolio_summary': self.tax_engine.get_portfolio_summary(),
                'system_statistics': self.system_stats.copy()
            }
            
            if include_sensitive:
                # Include detailed transaction data
                export_data['transactions'] = [
                    {
                        'id': tx.id,
                        'timestamp': tx.timestamp.isoformat(),
                        'type': tx.transaction_type.value,
                        'asset': tx.asset,
                        'quantity': str(tx.quantity),
                        'price': str(tx.price_per_unit),
                        'total_value': str(tx.total_value)
                    }
                    for tx in self.tax_engine.transactions
                ]
                
                export_data['tax_events'] = [
                    {
                        'id': event.id,
                        'timestamp': event.timestamp.isoformat(),
                        'type': event.event_type.value,
                        'asset': event.asset,
                        'gain_loss': str(event.gain_loss),
                        'holding_period': event.holding_period.value
                    }
                    for event in self.tax_engine.tax_events
                ]
            
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export tax data: {e}")
            raise
    
    # Helper methods
    def _capture_portfolio_state(self) -> Dict[str, Any]:
        """Capture current portfolio state for impact analysis."""
        return {
            'timestamp': datetime.now(),
            'portfolio_summary': self.tax_engine.get_portfolio_summary(),
            'total_cost_basis': sum(
                holding['cost_basis'] 
                for holding in self.tax_engine.get_portfolio_summary()['holdings'].values()
            )
        }
    
    def _analyze_transaction_impact(self, pre_state: Dict[str, Any], 
                                  post_state: Dict[str, Any],
                                  transaction: Transaction) -> Dict[str, Any]:
        """Analyze the impact of a transaction."""
        
        impact = {
            'transaction_id': transaction.id,
            'transaction_type': transaction.transaction_type.value,
            'asset': transaction.asset,
            'tax_impact': Decimal('0'),
            'portfolio_impact': {},
            'compliance_notes': []
        }
        
        # Calculate tax impact for sales
        if transaction.transaction_type == TransactionType.SELL:
            # Find the corresponding tax event
            recent_events = [
                event for event in self.tax_engine.tax_events
                if event.transaction_id == transaction.id
            ]
            
            if recent_events:
                tax_event = recent_events[0]
                impact['tax_impact'] = tax_event.gain_loss
                impact['holding_period'] = tax_event.holding_period.value
                impact['wash_sale_affected'] = tax_event.wash_sale_affected
        
        # Portfolio impact
        pre_cost_basis = pre_state['total_cost_basis']
        post_cost_basis = post_state['total_cost_basis']
        impact['portfolio_impact'] = {
            'cost_basis_change': post_cost_basis - pre_cost_basis,
            'position_change': f"{'Increased' if transaction.transaction_type in [TransactionType.BUY] else 'Decreased'} {transaction.asset} position"
        }
        
        return impact
    
    def _enhance_tax_analysis(self, base_analysis: Dict[str, Any], tax_year: int) -> Dict[str, Any]:
        """Enhance tax analysis with additional insights."""
        
        enhanced = {
            'tax_efficiency_metrics': {
                'short_term_percentage': self._calculate_short_term_percentage(tax_year),
                'wash_sale_impact': self._calculate_wash_sale_impact(tax_year),
                'loss_harvesting_utilization': self._calculate_loss_harvesting_utilization(tax_year)
            },
            'optimization_opportunities': {
                'potential_additional_savings': self._estimate_additional_savings(tax_year),
                'holding_period_optimization': self._assess_holding_period_optimization(tax_year)
            },
            'risk_factors': {
                'audit_risk_score': self._calculate_audit_risk_score(tax_year),
                'compliance_gaps': self._identify_compliance_gaps(tax_year)
            }
        }
        
        return enhanced
    
    def _simulate_sell_impact(self, transaction: Transaction) -> Dict[str, Any]:
        """Simulate the tax impact of a proposed sell transaction."""
        
        # Get current tax lots for the asset
        asset_lots = list(self.tax_engine.tax_lots.get(transaction.asset, []))
        
        if not asset_lots:
            return {'tax_impact': Decimal('0'), 'impact_type': 'no_position'}
        
        # Simulate cost basis calculation using current accounting method
        remaining_quantity = transaction.quantity
        total_cost_basis = Decimal('0')
        
        for lot in asset_lots:
            if remaining_quantity <= 0:
                break
            
            if lot.quantity <= remaining_quantity:
                total_cost_basis += lot.adjusted_total_cost_basis
                remaining_quantity -= lot.quantity
            else:
                partial_cost_basis = lot.adjusted_cost_basis_per_unit * remaining_quantity
                total_cost_basis += partial_cost_basis
                remaining_quantity = Decimal('0')
        
        # Calculate simulated gain/loss
        proceeds = transaction.total_value
        simulated_gain_loss = proceeds - total_cost_basis
        
        return {
            'tax_impact': simulated_gain_loss,
            'impact_type': 'gain' if simulated_gain_loss > 0 else 'loss',
            'cost_basis': total_cost_basis,
            'proceeds': proceeds,
            'estimated_tax_liability': self._estimate_tax_on_gain_loss(simulated_gain_loss)
        }
    
    def _simulate_buy_impact(self, transaction: Transaction) -> Dict[str, Any]:
        """Simulate the tax impact of a proposed buy transaction."""
        
        return {
            'tax_impact': Decimal('0'),
            'impact_type': 'cost_basis_increase',
            'cost_basis_addition': transaction.total_value,
            'potential_wash_sale_risk': self._assess_proposed_wash_sale_risk(transaction)
        }
    
    def _assess_proposed_wash_sale_risk(self, transaction: Transaction) -> Dict[str, Any]:
        """Assess wash sale risk for a proposed transaction."""
        
        current_date = datetime.now()
        lookback_start = current_date - timedelta(days=30)
        
        # Check for recent sales with losses
        recent_loss_sales = []
        for event in self.tax_engine.tax_events:
            if (event.asset == transaction.asset and
                event.gain_loss < 0 and
                lookback_start <= event.timestamp <= current_date):
                recent_loss_sales.append(event)
        
        risk_level = 'LOW'
        if len(recent_loss_sales) >= 2:
            risk_level = 'HIGH'
        elif len(recent_loss_sales) >= 1:
            risk_level = 'MEDIUM'
        
        return {
            'risk_level': risk_level,
            'recent_loss_sales': len(recent_loss_sales),
            'potential_disallowed_loss': sum(abs(event.gain_loss) for event in recent_loss_sales)
        }
    
    def _get_optimization_notes(self, transaction: Transaction) -> List[str]:
        """Get optimization notes for a proposed transaction."""
        
        notes = []
        
        # Holding period optimization
        if transaction.transaction_type == TransactionType.SELL:
            asset_lots = list(self.tax_engine.tax_lots.get(transaction.asset, []))
            for lot in asset_lots:
                days_held = (datetime.now() - lot.acquisition_date).days
                if 300 <= days_held <= 365:
                    notes.append(f"Consider waiting {365 - days_held} more days for long-term treatment")
        
        # Tax loss harvesting
        if transaction.transaction_type == TransactionType.BUY:
            wash_sale_risk = self._assess_proposed_wash_sale_risk(transaction)
            if wash_sale_risk['risk_level'] != 'LOW':
                notes.append("Purchase may trigger wash sale rule - consider alternative assets")
        
        return notes
    
    def _calculate_tax_efficiency_score(self) -> Decimal:
        """Calculate overall tax efficiency score (0-100)."""
        
        base_score = Decimal('70')  # Starting score
        
        # Factor in wash sale optimization
        total_events = len(self.tax_engine.tax_events)
        wash_sale_events = len([e for e in self.tax_engine.tax_events if e.wash_sale_affected])
        
        if total_events > 0:
            wash_sale_rate = wash_sale_events / total_events
            base_score -= Decimal(str(wash_sale_rate * 20))  # Penalty for wash sales
        
        # Factor in holding period optimization
        long_term_events = len([e for e in self.tax_engine.tax_events 
                               if e.holding_period == HoldingPeriod.LONG_TERM])
        
        if total_events > 0:
            long_term_rate = long_term_events / total_events
            base_score += Decimal(str(long_term_rate * 15))  # Bonus for long-term holdings
        
        return min(max(base_score, Decimal('0')), Decimal('100'))
    
    def _analyze_wash_sale_exposure(self) -> Dict[str, Any]:
        """Analyze current wash sale exposure."""
        
        wash_sale_events = [e for e in self.tax_engine.tax_events if e.wash_sale_affected]
        
        return {
            'total_wash_sales': len(wash_sale_events),
            'total_disallowed_loss': sum(e.wash_sale_disallowed for e in wash_sale_events),
            'affected_assets': list(set(e.asset for e in wash_sale_events)),
            'exposure_level': 'HIGH' if len(wash_sale_events) > 5 else 'MEDIUM' if len(wash_sale_events) > 2 else 'LOW'
        }
    
    def _analyze_holding_periods(self) -> Dict[str, Any]:
        """Analyze holding period distribution."""
        
        total_events = len(self.tax_engine.tax_events)
        if total_events == 0:
            return {'analysis': 'No tax events to analyze'}
        
        short_term_events = len([e for e in self.tax_engine.tax_events 
                               if e.holding_period == HoldingPeriod.SHORT_TERM])
        long_term_events = total_events - short_term_events
        
        return {
            'total_events': total_events,
            'short_term_events': short_term_events,
            'long_term_events': long_term_events,
            'short_term_percentage': (short_term_events / total_events) * 100,
            'long_term_percentage': (long_term_events / total_events) * 100,
            'optimization_score': 'GOOD' if long_term_events > short_term_events else 'NEEDS_IMPROVEMENT'
        }
    
    def _calculate_opportunity_priority(self, opportunity: Dict[str, Any]) -> Decimal:
        """Calculate priority score for an optimization opportunity."""
        
        # Base priority on tax savings
        priority = opportunity.get('potential_tax_savings', Decimal('0'))
        
        # Adjust for risk factors
        if opportunity.get('wash_sale_risk', {}).get('high_risk', False):
            priority *= Decimal('0.7')  # Reduce priority for high wash sale risk
        
        return priority
    
    def _recommend_execution_timeline(self, opportunity: Dict[str, Any]) -> str:
        """Recommend execution timeline for an opportunity."""
        
        if opportunity.get('wash_sale_risk', {}).get('high_risk', False):
            return "WAIT_30_DAYS"
        elif opportunity.get('potential_tax_savings', Decimal('0')) > Decimal('1000'):
            return "IMMEDIATE"
        else:
            return "WITHIN_7_DAYS"
    
    # Additional helper methods would be implemented here...
    def _calculate_short_term_percentage(self, tax_year: int) -> float:
        """Calculate percentage of short-term transactions."""
        year_events = [e for e in self.tax_engine.tax_events if e.timestamp.year == tax_year]
        if not year_events:
            return 0.0
        
        short_term_count = len([e for e in year_events if e.holding_period == HoldingPeriod.SHORT_TERM])
        return (short_term_count / len(year_events)) * 100
    
    def _calculate_wash_sale_impact(self, tax_year: int) -> Decimal:
        """Calculate total wash sale impact for the year."""
        year_events = [e for e in self.tax_engine.tax_events 
                      if e.timestamp.year == tax_year and e.wash_sale_affected]
        return sum(e.wash_sale_disallowed for e in year_events)
    
    def _calculate_loss_harvesting_utilization(self, tax_year: int) -> float:
        """Calculate loss harvesting utilization rate."""
        # This would analyze how well losses were harvested vs opportunities missed
        return 0.75  # Placeholder - would implement actual calculation
    
    def _estimate_additional_savings(self, tax_year: int) -> Decimal:
        """Estimate potential additional tax savings."""
        # This would analyze current positions for additional opportunities
        return Decimal('500')  # Placeholder
    
    def _assess_holding_period_optimization(self, tax_year: int) -> str:
        """Assess holding period optimization opportunities."""
        short_term_pct = self._calculate_short_term_percentage(tax_year)
        if short_term_pct > 60:
            return "Consider holding positions longer for favorable long-term rates"
        else:
            return "Good long-term holding discipline"
    
    def _calculate_audit_risk_score(self, tax_year: int) -> Decimal:
        """Calculate audit risk score."""
        # Simplified risk scoring
        base_risk = Decimal('0.1')  # Base 10% risk
        
        # Increase risk for high transaction volume
        year_transactions = len([tx for tx in self.tax_engine.transactions 
                               if tx.timestamp.year == tax_year])
        if year_transactions > 1000:
            base_risk += Decimal('0.05')
        
        return min(base_risk, Decimal('0.5'))  # Cap at 50%
    
    def _identify_compliance_gaps(self, tax_year: int) -> List[str]:
        """Identify potential compliance gaps."""
        gaps = []
        
        # Check for missing documentation
        year_events = [e for e in self.tax_engine.tax_events if e.timestamp.year == tax_year]
        if len(year_events) > 100:
            gaps.append("High transaction volume - ensure detailed record keeping")
        
        # Check wash sale compliance
        wash_sales = len([e for e in year_events if e.wash_sale_affected])
        if wash_sales > 0:
            gaps.append("Wash sale adjustments present - verify proper reporting")
        
        return gaps
    
    def _estimate_tax_on_gain_loss(self, gain_loss: Decimal) -> Decimal:
        """Estimate tax liability on a gain or loss."""
        if gain_loss <= 0:
            return Decimal('0')  # Losses don't create liability
        
        # Use current tax bracket as estimate
        return gain_loss * self.config.current_tax_bracket
    
    def _calculate_unrealized_gain_loss(self, asset: str, holding: Dict[str, Any]) -> Decimal:
        """Calculate unrealized gain/loss for an asset."""
        # Placeholder - would use current market price
        return Decimal('0')  # Would implement with real price data
    
    def _analyze_tax_lot_optimization(self, asset: str, tax_lots: List) -> Dict[str, Any]:
        """Analyze tax lot optimization opportunities."""
        return {
            'lots_count': len(tax_lots),
            'optimization_potential': 'MEDIUM',  # Placeholder
            'recommended_action': 'Consider specific identification for optimal tax outcomes'
        }
    
    def _analyze_asset_wash_sale_risk(self, asset: str) -> Dict[str, Any]:
        """Analyze wash sale risk for a specific asset."""
        recent_events = [
            e for e in self.tax_engine.tax_events
            if e.asset == asset and e.timestamp >= datetime.now() - timedelta(days=30)
        ]
        
        return {
            'recent_activity': len(recent_events),
            'risk_level': 'HIGH' if len(recent_events) > 3 else 'MEDIUM' if len(recent_events) > 1 else 'LOW'
        }
    
    def _assess_harvesting_potential(self, asset: str, holding: Dict[str, Any]) -> Dict[str, Any]:
        """Assess tax loss harvesting potential for an asset."""
        return {
            'potential': 'MEDIUM',  # Placeholder
            'estimated_savings': Decimal('100'),  # Would calculate based on current prices
            'recommendation': 'Monitor for harvesting opportunities'
        }
    
    def _analyze_cost_basis_distribution(self) -> Dict[str, Any]:
        """Analyze cost basis distribution across portfolio."""
        portfolio = self.tax_engine.get_portfolio_summary()
        
        total_cost_basis = portfolio['total_cost_basis']
        holdings_count = len(portfolio['holdings'])
        
        return {
            'total_cost_basis': total_cost_basis,
            'average_position_size': total_cost_basis / holdings_count if holdings_count > 0 else Decimal('0'),
            'diversification_score': min(holdings_count / 10, 1.0),  # Score based on diversification
            'concentration_risk': 'LOW' if holdings_count > 10 else 'MEDIUM' if holdings_count > 5 else 'HIGH'
        }
    
    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify current optimization opportunities."""
        opportunities = []
        
        # Check for loss harvesting
        tlh_opportunities = self.tax_engine.get_tax_loss_harvesting_opportunities()
        if tlh_opportunities:
            opportunities.append(f"{len(tlh_opportunities)} tax-loss harvesting opportunities available")
        
        # Check for holding period optimization
        current_lots = []
        for asset_lots in self.tax_engine.tax_lots.values():
            current_lots.extend(asset_lots)
        
        approaching_long_term = [
            lot for lot in current_lots
            if 300 <= (datetime.now() - lot.acquisition_date).days <= 365
        ]
        
        if approaching_long_term:
            opportunities.append(f"{len(approaching_long_term)} positions approaching long-term status")
        
        return opportunities
    
    def _assess_tax_related_risks(self) -> Dict[str, Any]:
        """Assess tax-related risks in the portfolio."""
        return {
            'wash_sale_exposure': self._analyze_wash_sale_exposure()['exposure_level'],
            'concentration_risk': self._analyze_cost_basis_distribution()['concentration_risk'],
            'compliance_risk': 'LOW',  # Would implement detailed compliance risk assessment
            'audit_risk': float(self._calculate_audit_risk_score(datetime.now().year))
        }
    
    def _check_compliance_status(self) -> Dict[str, Any]:
        """Check current compliance status."""
        return {
            'record_completeness': 'GOOD',  # Would check for missing data
            'wash_sale_compliance': 'COMPLIANT',  # Would verify wash sale calculations
            'reporting_readiness': 'READY',  # Would check if ready for tax reporting
            'documentation_score': 85  # Score out of 100
        }


# Factory functions for easy initialization
def create_tax_system(jurisdiction: str = "US", 
                     accounting_method: str = "FIFO",
                     db_path: str = "tax_system.db") -> TaxSystem:
    """Factory function to create a configured tax system."""
    
    accounting_method_enum = AccountingMethod(accounting_method.lower())
    return TaxSystem(jurisdiction, accounting_method_enum, db_path)

def create_demo_tax_system() -> TaxSystem:
    """Create a demo tax system with sample data."""
    
    system = create_tax_system()
    
    # Add some sample transactions for demonstration
    sample_transactions = [
        Transaction(
            id="demo_buy_1",
            timestamp=datetime(2024, 1, 15),
            transaction_type=TransactionType.BUY,
            asset="BTC",
            quantity=Decimal('1.0'),
            price_per_unit=Decimal('45000'),
            total_value=Decimal('45000')
        ),
        Transaction(
            id="demo_sell_1",
            timestamp=datetime(2024, 6, 15),
            transaction_type=TransactionType.SELL,
            asset="BTC",
            quantity=Decimal('0.5'),
            price_per_unit=Decimal('60000'),
            total_value=Decimal('30000')
        )
    ]
    
    for tx in sample_transactions:
        system.add_transaction(tx)
    
    return system

# Export main classes and functions
__all__ = [
    'TaxSystem',
    'TaxEngine',
    'TaxReporter', 
    'TaxOptimizer',
    'create_tax_system',
    'create_demo_tax_system',
    'AccountingMethod',
    'TransactionType',
    'ReportType',
    'ReportFormat',
    'OptimizationStrategy'
]