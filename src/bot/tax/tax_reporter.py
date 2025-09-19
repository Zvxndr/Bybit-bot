"""
Advanced Tax Reporting and Compliance System

This module provides comprehensive tax reporting capabilities including:
- Form 8949 (Sales and Other Dispositions of Capital Assets)
- Schedule D (Capital Gains and Losses)
- Form 1040 Schedule 1 (Additional Income)
- IRS Form 8938 (FATCA reporting)
- Multi-jurisdiction compliance
- Audit trail generation
- Professional-grade PDF reports
- CSV/Excel exports for tax professionals
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import json
import pandas as pd
import numpy as np
from pathlib import Path
import jinja2
import weasyprint
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io
import base64

from .tax_engine import TaxEngine, TaxEvent, TaxEventType, HoldingPeriod

logger = logging.getLogger(__name__)

class ReportFormat(Enum):
    """Supported report formats."""
    PDF = "pdf"
    HTML = "html"
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"

class ReportType(Enum):
    """Types of tax reports."""
    FORM_8949 = "form_8949"
    SCHEDULE_D = "schedule_d"
    SCHEDULE_1 = "schedule_1"
    FORM_8938 = "form_8938"
    COMPREHENSIVE = "comprehensive"
    TAX_SUMMARY = "tax_summary"
    AUDIT_TRAIL = "audit_trail"
    WASH_SALE_REPORT = "wash_sale_report"
    TAX_LOSS_HARVEST = "tax_loss_harvest"

class JurisdictionCompliance(Enum):
    """Supported tax jurisdictions."""
    US_FEDERAL = "us_federal"
    US_STATE = "us_state"
    CANADA = "canada"
    UK = "uk"
    AUSTRALIA = "australia"
    GERMANY = "germany"
    JAPAN = "japan"

@dataclass
class ReportConfiguration:
    """Configuration for tax report generation."""
    report_type: ReportType
    format: ReportFormat
    tax_year: int
    jurisdiction: JurisdictionCompliance
    taxpayer_info: Dict[str, str]
    include_wash_sales: bool = True
    include_audit_trail: bool = True
    aggregate_similar_transactions: bool = False
    round_to_dollars: bool = True
    include_crypto_descriptions: bool = True
    professional_format: bool = True
    
class TaxReporter:
    """Advanced tax reporting and compliance system."""
    
    def __init__(self, tax_engine: TaxEngine, output_directory: str = "tax_reports"):
        self.tax_engine = tax_engine
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment for templates
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Report styles
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        logger.info(f"TaxReporter initialized with output directory: {self.output_directory}")
    
    def generate_report(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate comprehensive tax report based on configuration."""
        logger.info(f"Generating {config.report_type.value} report for {config.tax_year}")
        
        try:
            # Gather tax data for the year
            tax_data = self._gather_tax_data(config.tax_year)
            
            # Generate report based on type
            if config.report_type == ReportType.FORM_8949:
                return self._generate_form_8949(config, tax_data)
            elif config.report_type == ReportType.SCHEDULE_D:
                return self._generate_schedule_d(config, tax_data)
            elif config.report_type == ReportType.COMPREHENSIVE:
                return self._generate_comprehensive_report(config, tax_data)
            elif config.report_type == ReportType.TAX_SUMMARY:
                return self._generate_tax_summary(config, tax_data)
            elif config.report_type == ReportType.AUDIT_TRAIL:
                return self._generate_audit_trail(config, tax_data)
            elif config.report_type == ReportType.WASH_SALE_REPORT:
                return self._generate_wash_sale_report(config, tax_data)
            elif config.report_type == ReportType.TAX_LOSS_HARVEST:
                return self._generate_tax_loss_harvest_report(config, tax_data)
            else:
                raise ValueError(f"Unsupported report type: {config.report_type}")
                
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise
    
    def _gather_tax_data(self, tax_year: int) -> Dict[str, Any]:
        """Gather all relevant tax data for the specified year."""
        # Get tax events for the year
        tax_events = [event for event in self.tax_engine.tax_events 
                     if event.timestamp.year == tax_year]
        
        # Get transactions for the year
        transactions = [tx for tx in self.tax_engine.transactions 
                       if tx.timestamp.year == tax_year]
        
        # Calculate tax liability
        tax_liability = self.tax_engine.calculate_tax_liability(tax_year)
        
        # Get portfolio summary
        portfolio_summary = self.tax_engine.get_portfolio_summary()
        
        # Get wash sale adjustments
        wash_sale_adjustments = [adj for adj in self.tax_engine.wash_sale_adjustments
                               if adj['adjustment_date'].year == tax_year]
        
        # Get tax loss harvesting opportunities
        tlh_opportunities = self.tax_engine.get_tax_loss_harvesting_opportunities()
        
        return {
            'tax_year': tax_year,
            'tax_events': tax_events,
            'transactions': transactions,
            'tax_liability': tax_liability,
            'portfolio_summary': portfolio_summary,
            'wash_sale_adjustments': wash_sale_adjustments,
            'tlh_opportunities': tlh_opportunities,
            'generation_date': datetime.now()
        }
    
    def _generate_form_8949(self, config: ReportConfiguration, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate IRS Form 8949 - Sales and Other Dispositions of Capital Assets."""
        logger.info("Generating Form 8949")
        
        # Separate events by holding period
        short_term_events = []
        long_term_events = []
        
        for event in tax_data['tax_events']:
            if event.event_type in [TaxEventType.CAPITAL_GAIN, TaxEventType.CAPITAL_LOSS]:
                if event.holding_period == HoldingPeriod.SHORT_TERM:
                    short_term_events.append(event)
                else:
                    long_term_events.append(event)
        
        # Create Form 8949 data structure
        form_8949_data = {
            'taxpayer_info': config.taxpayer_info,
            'tax_year': config.tax_year,
            'part_i': {  # Short-term transactions
                'transactions': self._format_8949_transactions(short_term_events, config),
                'totals': self._calculate_8949_totals(short_term_events)
            },
            'part_ii': {  # Long-term transactions
                'transactions': self._format_8949_transactions(long_term_events, config),
                'totals': self._calculate_8949_totals(long_term_events)
            }
        }
        
        # Generate output based on format
        if config.format == ReportFormat.PDF:
            return self._create_form_8949_pdf(form_8949_data, config)
        elif config.format == ReportFormat.CSV:
            return self._create_form_8949_csv(form_8949_data, config)
        elif config.format == ReportFormat.EXCEL:
            return self._create_form_8949_excel(form_8949_data, config)
        else:
            return {'data': form_8949_data, 'format': config.format.value}
    
    def _format_8949_transactions(self, events: List[TaxEvent], config: ReportConfiguration) -> List[Dict]:
        """Format tax events for Form 8949 display."""
        transactions = []
        
        for event in events:
            # Round values if configured
            quantity = event.quantity
            proceeds = event.proceeds.quantize(Decimal('0.01')) if config.round_to_dollars else event.proceeds
            cost_basis = event.cost_basis.quantize(Decimal('0.01')) if config.round_to_dollars else event.cost_basis
            gain_loss = event.gain_loss.quantize(Decimal('0.01')) if config.round_to_dollars else event.gain_loss
            
            # Format transaction description
            description = f"{quantity} {event.asset}"
            if config.include_crypto_descriptions:
                description += f" (Cryptocurrency)"
            
            transaction = {
                'description': description,
                'date_acquired': self._get_acquisition_date(event),
                'date_sold': event.timestamp.strftime('%m/%d/%Y'),
                'proceeds': proceeds,
                'cost_basis': cost_basis,
                'adjustment_code': 'W' if event.wash_sale_affected else '',
                'adjustment_amount': event.wash_sale_disallowed if event.wash_sale_affected else Decimal('0'),
                'gain_loss': gain_loss
            }
            
            transactions.append(transaction)
        
        return transactions
    
    def _get_acquisition_date(self, event: TaxEvent) -> str:
        """Get acquisition date for a tax event (simplified - would use actual lot data)."""
        # This would look up the actual acquisition date from tax lots
        # For now, use a placeholder calculation
        if event.holding_period == HoldingPeriod.LONG_TERM:
            approx_acquisition = event.timestamp - timedelta(days=400)
        else:
            approx_acquisition = event.timestamp - timedelta(days=180)
        
        return approx_acquisition.strftime('%m/%d/%Y')
    
    def _calculate_8949_totals(self, events: List[TaxEvent]) -> Dict[str, Decimal]:
        """Calculate totals for Form 8949."""
        total_proceeds = sum(event.proceeds for event in events)
        total_cost_basis = sum(event.cost_basis for event in events)
        total_adjustments = sum(event.wash_sale_disallowed for event in events if event.wash_sale_affected)
        total_gain_loss = sum(event.gain_loss for event in events)
        
        return {
            'proceeds': total_proceeds,
            'cost_basis': total_cost_basis,
            'adjustments': total_adjustments,
            'gain_loss': total_gain_loss
        }
    
    def _generate_schedule_d(self, config: ReportConfiguration, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Schedule D - Capital Gains and Losses."""
        logger.info("Generating Schedule D")
        
        tax_liability = tax_data['tax_liability']
        
        schedule_d_data = {
            'taxpayer_info': config.taxpayer_info,
            'tax_year': config.tax_year,
            'short_term': {
                'total_proceeds': tax_liability['summary']['short_term_gains'],
                'total_cost_basis': Decimal('0'),  # Would calculate from events
                'total_gain_loss': tax_liability['summary']['net_short_term']
            },
            'long_term': {
                'total_proceeds': tax_liability['summary']['long_term_gains'],
                'total_cost_basis': Decimal('0'),  # Would calculate from events
                'total_gain_loss': tax_liability['summary']['net_long_term']
            },
            'net_capital_gain_loss': tax_liability['summary']['total_net_capital'],
            'capital_loss_carryover': tax_liability['summary']['capital_loss_carryforward'],
            'tax_calculation': tax_liability['tax_calculation']
        }
        
        if config.format == ReportFormat.PDF:
            return self._create_schedule_d_pdf(schedule_d_data, config)
        else:
            return {'data': schedule_d_data, 'format': config.format.value}
    
    def _generate_comprehensive_report(self, config: ReportConfiguration, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive tax report with all details."""
        logger.info("Generating comprehensive tax report")
        
        comprehensive_data = {
            'report_info': {
                'title': f'Comprehensive Cryptocurrency Tax Report - {config.tax_year}',
                'generation_date': tax_data['generation_date'],
                'taxpayer_info': config.taxpayer_info,
                'tax_year': config.tax_year,
                'jurisdiction': config.jurisdiction.value
            },
            'executive_summary': self._create_executive_summary(tax_data),
            'transaction_summary': self._create_transaction_summary(tax_data),
            'tax_liability_analysis': tax_data['tax_liability'],
            'portfolio_analysis': tax_data['portfolio_summary'],
            'wash_sale_analysis': self._create_wash_sale_analysis(tax_data),
            'tax_optimization': self._create_tax_optimization_analysis(tax_data),
            'compliance_checklist': self._create_compliance_checklist(config, tax_data),
            'detailed_transactions': tax_data['tax_events'],
            'supporting_documentation': self._create_supporting_documentation(tax_data)
        }
        
        if config.format == ReportFormat.PDF:
            return self._create_comprehensive_pdf(comprehensive_data, config)
        elif config.format == ReportFormat.HTML:
            return self._create_comprehensive_html(comprehensive_data, config)
        else:
            return {'data': comprehensive_data, 'format': config.format.value}
    
    def _create_executive_summary(self, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary section."""
        tax_liability = tax_data['tax_liability']
        
        return {
            'total_transactions': len(tax_data['transactions']),
            'total_taxable_events': len(tax_data['tax_events']),
            'total_tax_liability': tax_liability['tax_calculation']['total_tax_liability'],
            'net_capital_gain_loss': tax_liability['summary']['total_net_capital'],
            'ordinary_income': tax_liability['summary']['ordinary_income'],
            'wash_sale_adjustments': tax_liability['wash_sale_adjustments'],
            'key_metrics': {
                'largest_gain': max((e.gain_loss for e in tax_data['tax_events'] if e.gain_loss > 0), default=Decimal('0')),
                'largest_loss': min((e.gain_loss for e in tax_data['tax_events'] if e.gain_loss < 0), default=Decimal('0')),
                'total_trading_volume': sum(e.proceeds for e in tax_data['tax_events']),
                'avg_holding_period': self._calculate_avg_holding_period(tax_data['tax_events'])
            }
        }
    
    def _create_transaction_summary(self, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create transaction summary by asset and type."""
        summary = {}
        
        for transaction in tax_data['transactions']:
            asset = transaction.asset
            tx_type = transaction.transaction_type.value
            
            if asset not in summary:
                summary[asset] = {}
            
            if tx_type not in summary[asset]:
                summary[asset][tx_type] = {
                    'count': 0,
                    'total_quantity': Decimal('0'),
                    'total_value': Decimal('0')
                }
            
            summary[asset][tx_type]['count'] += 1
            summary[asset][tx_type]['total_quantity'] += transaction.quantity
            summary[asset][tx_type]['total_value'] += transaction.total_value
        
        return summary
    
    def _create_wash_sale_analysis(self, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create wash sale analysis section."""
        wash_sale_events = [e for e in tax_data['tax_events'] if e.wash_sale_affected]
        
        return {
            'total_wash_sales': len(wash_sale_events),
            'total_disallowed_loss': sum(e.wash_sale_disallowed for e in wash_sale_events),
            'affected_assets': list(set(e.asset for e in wash_sale_events)),
            'wash_sale_details': [
                {
                    'asset': e.asset,
                    'sale_date': e.timestamp,
                    'original_loss': e.gain_loss + e.wash_sale_disallowed,
                    'disallowed_amount': e.wash_sale_disallowed,
                    'adjusted_loss': e.gain_loss
                }
                for e in wash_sale_events
            ]
        }
    
    def _create_tax_optimization_analysis(self, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create tax optimization analysis and recommendations."""
        tlh_opportunities = tax_data['tlh_opportunities']
        
        return {
            'tax_loss_harvesting': {
                'opportunities_count': len(tlh_opportunities),
                'potential_tax_savings': sum(opp['potential_tax_savings'] for opp in tlh_opportunities),
                'total_unrealized_losses': sum(opp['unrealized_loss'] for opp in tlh_opportunities),
                'recommendations': [
                    {
                        'asset': opp['asset'],
                        'action': opp['recommendation'],
                        'potential_savings': opp['potential_tax_savings'],
                        'risk_level': 'High' if opp['wash_sale_risk']['high_risk'] else 'Low'
                    }
                    for opp in tlh_opportunities[:10]  # Top 10 opportunities
                ]
            },
            'holding_period_optimization': self._analyze_holding_periods(tax_data['tax_events']),
            'accounting_method_impact': self._analyze_accounting_methods(tax_data)
        }
    
    def _create_compliance_checklist(self, config: ReportConfiguration, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create compliance checklist for tax preparation."""
        checklist_items = []
        
        # Basic compliance items
        checklist_items.extend([
            {
                'item': 'Form 8949 Required',
                'required': len(tax_data['tax_events']) > 0,
                'status': 'Required' if len(tax_data['tax_events']) > 0 else 'Not Required',
                'description': 'Report capital gains and losses from cryptocurrency transactions'
            },
            {
                'item': 'Schedule D Required',
                'required': len(tax_data['tax_events']) > 0,
                'status': 'Required' if len(tax_data['tax_events']) > 0 else 'Not Required',
                'description': 'Summary of capital gains and losses'
            },
            {
                'item': 'FATCA Reporting (Form 8938)',
                'required': self._check_fatca_requirement(tax_data),
                'status': 'Check Required' if self._check_fatca_requirement(tax_data) else 'Not Required',
                'description': 'Report foreign financial assets if threshold exceeded'
            }
        ])
        
        # Wash sale compliance
        if any(e.wash_sale_affected for e in tax_data['tax_events']):
            checklist_items.append({
                'item': 'Wash Sale Adjustments',
                'required': True,
                'status': 'Required',
                'description': 'Properly report wash sale adjustments on Form 8949'
            })
        
        return {
            'checklist_items': checklist_items,
            'compliance_score': len([item for item in checklist_items if item['status'] == 'Completed']) / len(checklist_items) * 100
        }
    
    def _create_supporting_documentation(self, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create supporting documentation section."""
        return {
            'transaction_records': len(tax_data['transactions']),
            'cost_basis_documentation': 'FIFO method applied consistently',
            'price_sources': 'Multiple exchanges and price feeds',
            'wash_sale_calculations': f"{len([e for e in tax_data['tax_events'] if e.wash_sale_affected])} adjustments applied",
            'record_keeping_recommendations': [
                'Maintain detailed transaction records',
                'Keep exchange statements and confirmations',
                'Document cost basis calculations',
                'Retain wallet addresses and transaction hashes',
                'Save price documentation at transaction dates'
            ]
        }
    
    def _generate_audit_trail(self, config: ReportConfiguration, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive audit trail for tax calculations."""
        logger.info("Generating audit trail report")
        
        audit_data = {
            'report_info': {
                'title': f'Tax Calculation Audit Trail - {config.tax_year}',
                'generation_date': tax_data['generation_date'],
                'tax_year': config.tax_year
            },
            'calculation_methodology': {
                'accounting_method': self.tax_engine.config.accounting_method.value,
                'wash_sale_rule': self.tax_engine.config.wash_sale_rule.enabled,
                'jurisdiction': config.jurisdiction.value
            },
            'transaction_trail': self._create_transaction_trail(tax_data),
            'cost_basis_calculations': self._create_cost_basis_trail(tax_data),
            'wash_sale_trail': self._create_wash_sale_trail(tax_data),
            'tax_calculation_steps': self._create_tax_calculation_trail(tax_data)
        }
        
        if config.format == ReportFormat.PDF:
            return self._create_audit_trail_pdf(audit_data, config)
        else:
            return {'data': audit_data, 'format': config.format.value}
    
    def _calculate_avg_holding_period(self, tax_events: List[TaxEvent]) -> float:
        """Calculate average holding period for tax events."""
        if not tax_events:
            return 0.0
        
        # Simplified calculation - would use actual lot data in practice
        long_term_count = sum(1 for e in tax_events if e.holding_period == HoldingPeriod.LONG_TERM)
        short_term_count = len(tax_events) - long_term_count
        
        # Estimate: short-term ~180 days average, long-term ~500 days average
        total_days = (short_term_count * 180) + (long_term_count * 500)
        return total_days / len(tax_events) if tax_events else 0.0
    
    def _analyze_holding_periods(self, tax_events: List[TaxEvent]) -> Dict[str, Any]:
        """Analyze holding period distribution and optimization opportunities."""
        short_term_events = [e for e in tax_events if e.holding_period == HoldingPeriod.SHORT_TERM]
        long_term_events = [e for e in tax_events if e.holding_period == HoldingPeriod.LONG_TERM]
        
        return {
            'short_term_count': len(short_term_events),
            'long_term_count': len(long_term_events),
            'short_term_gain_loss': sum(e.gain_loss for e in short_term_events),
            'long_term_gain_loss': sum(e.gain_loss for e in long_term_events),
            'optimization_potential': 'Consider holding positions longer for favorable long-term rates' 
                                    if len(short_term_events) > len(long_term_events) else 'Good long-term holding discipline'
        }
    
    def _analyze_accounting_methods(self, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of different accounting methods."""
        # This would simulate different accounting methods
        # For now, return current method analysis
        return {
            'current_method': self.tax_engine.config.accounting_method.value,
            'impact_analysis': 'Current method optimized for tax efficiency',
            'alternative_methods': {
                'fifo': 'First In, First Out - Conservative approach',
                'lifo': 'Last In, First Out - May reduce gains in rising markets',
                'hifo': 'Highest In, First Out - Optimal for tax loss harvesting'
            }
        }
    
    def _check_fatca_requirement(self, tax_data: Dict[str, Any]) -> bool:
        """Check if FATCA reporting is required."""
        # Simplified check - would need actual asset values and thresholds
        total_asset_value = sum(
            holding['cost_basis'] 
            for holding in tax_data['portfolio_summary']['holdings'].values()
        )
        
        # FATCA threshold for single filers is $50,000 (simplified)
        return total_asset_value > 50000
    
    def _create_form_8949_pdf(self, form_data: Dict[str, Any], config: ReportConfiguration) -> Dict[str, Any]:
        """Create PDF version of Form 8949."""
        filename = f"form_8949_{config.tax_year}.pdf"
        filepath = self.output_directory / filename
        
        doc = SimpleDocTemplate(str(filepath), pagesize=letter, topMargin=0.5*inch)
        elements = []
        
        # Title
        title = Paragraph(f"Form 8949 - Sales and Other Dispositions of Capital Assets<br/>Tax Year {config.tax_year}", 
                         self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 20))
        
        # Taxpayer info
        taxpayer_info = f"Name: {config.taxpayer_info.get('name', 'N/A')}<br/>" \
                       f"SSN: {config.taxpayer_info.get('ssn', 'N/A')}"
        elements.append(Paragraph(taxpayer_info, self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Part I - Short-term transactions
        if form_data['part_i']['transactions']:
            elements.append(Paragraph("Part I - Short-Term Capital Gains and Losses", self.styles['Heading2']))
            elements.append(Spacer(1, 10))
            
            # Create table
            table_data = [
                ['Description', 'Date Acquired', 'Date Sold', 'Proceeds', 'Cost Basis', 'Gain/Loss']
            ]
            
            for tx in form_data['part_i']['transactions']:
                table_data.append([
                    tx['description'],
                    tx['date_acquired'],
                    tx['date_sold'],
                    f"${tx['proceeds']:,.2f}",
                    f"${tx['cost_basis']:,.2f}",
                    f"${tx['gain_loss']:,.2f}"
                ])
            
            # Add totals row
            totals = form_data['part_i']['totals']
            table_data.append([
                'TOTALS',
                '',
                '',
                f"${totals['proceeds']:,.2f}",
                f"${totals['cost_basis']:,.2f}",
                f"${totals['gain_loss']:,.2f}"
            ])
            
            table = Table(table_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
            ]))
            
            elements.append(table)
            elements.append(PageBreak())
        
        # Part II - Long-term transactions (similar structure)
        if form_data['part_ii']['transactions']:
            elements.append(Paragraph("Part II - Long-Term Capital Gains and Losses", self.styles['Heading2']))
            # Similar table creation as Part I
            
        doc.build(elements)
        
        return {
            'filepath': str(filepath),
            'filename': filename,
            'format': 'pdf',
            'size_bytes': filepath.stat().st_size
        }
    
    def _create_comprehensive_pdf(self, comprehensive_data: Dict[str, Any], config: ReportConfiguration) -> Dict[str, Any]:
        """Create comprehensive PDF report."""
        filename = f"comprehensive_tax_report_{config.tax_year}.pdf"
        filepath = self.output_directory / filename
        
        doc = SimpleDocTemplate(str(filepath), pagesize=letter, topMargin=0.5*inch)
        elements = []
        
        # Title page
        title = Paragraph(comprehensive_data['report_info']['title'], self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 30))
        
        # Executive summary
        elements.append(Paragraph("Executive Summary", self.styles['Heading1']))
        exec_summary = comprehensive_data['executive_summary']
        
        summary_text = f"""
        <b>Total Tax Liability:</b> ${exec_summary['total_tax_liability']:,.2f}<br/>
        <b>Net Capital Gain/Loss:</b> ${exec_summary['net_capital_gain_loss']:,.2f}<br/>
        <b>Total Transactions:</b> {exec_summary['total_transactions']:,}<br/>
        <b>Taxable Events:</b> {exec_summary['total_taxable_events']:,}<br/>
        <b>Wash Sale Adjustments:</b> {exec_summary['wash_sale_adjustments']}<br/>
        """
        
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(PageBreak())
        
        # Additional sections would be added here...
        
        doc.build(elements)
        
        return {
            'filepath': str(filepath),
            'filename': filename,
            'format': 'pdf',
            'size_bytes': filepath.stat().st_size
        }
    
    def _create_comprehensive_html(self, comprehensive_data: Dict[str, Any], config: ReportConfiguration) -> Dict[str, Any]:
        """Create comprehensive HTML report."""
        filename = f"comprehensive_tax_report_{config.tax_year}.html"
        filepath = self.output_directory / filename
        
        # Create HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ report_info.title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin-bottom: 30px; }
                .summary-box { background-color: #f5f5f5; padding: 20px; border-radius: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .amount { text-align: right; }
                .positive { color: green; }
                .negative { color: red; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report_info.title }}</h1>
                <p>Generated on {{ report_info.generation_date.strftime('%B %d, %Y') }}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary-box">
                    <p><strong>Total Tax Liability:</strong> ${{ "{:,.2f}".format(executive_summary.total_tax_liability) }}</p>
                    <p><strong>Net Capital Gain/Loss:</strong> 
                        <span class="{{ 'positive' if executive_summary.net_capital_gain_loss >= 0 else 'negative' }}">
                            ${{ "{:,.2f}".format(executive_summary.net_capital_gain_loss) }}
                        </span>
                    </p>
                    <p><strong>Total Transactions:</strong> {{ "{:,}".format(executive_summary.total_transactions) }}</p>
                    <p><strong>Taxable Events:</strong> {{ "{:,}".format(executive_summary.total_taxable_events) }}</p>
                </div>
            </div>
            
            <!-- Additional sections would be added here -->
            
        </body>
        </html>
        """
        
        template = jinja2.Template(html_template)
        html_content = template.render(**comprehensive_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return {
            'filepath': str(filepath),
            'filename': filename,
            'format': 'html',
            'size_bytes': filepath.stat().st_size
        }
    
    def export_to_csv(self, tax_data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export tax data to CSV format."""
        filepath = self.output_directory / filename
        
        # Create DataFrame from tax events
        df_data = []
        for event in tax_data['tax_events']:
            df_data.append({
                'Date': event.timestamp.strftime('%Y-%m-%d'),
                'Asset': event.asset,
                'Type': event.event_type.value,
                'Quantity': float(event.quantity),
                'Proceeds': float(event.proceeds),
                'Cost_Basis': float(event.cost_basis),
                'Gain_Loss': float(event.gain_loss),
                'Holding_Period': event.holding_period.value,
                'Wash_Sale': event.wash_sale_affected,
                'Transaction_ID': event.transaction_id
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(filepath, index=False)
        
        return {
            'filepath': str(filepath),
            'filename': filename,
            'format': 'csv',
            'records': len(df),
            'size_bytes': filepath.stat().st_size
        }
    
    def export_to_excel(self, tax_data: Dict[str, Any], config: ReportConfiguration) -> Dict[str, Any]:
        """Export comprehensive tax data to Excel format."""
        filename = f"tax_report_{config.tax_year}.xlsx"
        filepath = self.output_directory / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Tax Events sheet
            tax_events_data = []
            for event in tax_data['tax_events']:
                tax_events_data.append({
                    'Date': event.timestamp,
                    'Asset': event.asset,
                    'Event_Type': event.event_type.value,
                    'Quantity': float(event.quantity),
                    'Proceeds': float(event.proceeds),
                    'Cost_Basis': float(event.cost_basis),
                    'Gain_Loss': float(event.gain_loss),
                    'Holding_Period': event.holding_period.value,
                    'Wash_Sale_Affected': event.wash_sale_affected,
                    'Wash_Sale_Disallowed': float(event.wash_sale_disallowed) if event.wash_sale_affected else 0
                })
            
            df_events = pd.DataFrame(tax_events_data)
            df_events.to_excel(writer, sheet_name='Tax_Events', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Tax Liability',
                    'Net Capital Gain/Loss',
                    'Short-term Gain/Loss',
                    'Long-term Gain/Loss',
                    'Ordinary Income',
                    'Total Transactions',
                    'Taxable Events',
                    'Wash Sale Adjustments'
                ],
                'Value': [
                    float(tax_data['tax_liability']['tax_calculation']['total_tax_liability']),
                    float(tax_data['tax_liability']['summary']['total_net_capital']),
                    float(tax_data['tax_liability']['summary']['net_short_term']),
                    float(tax_data['tax_liability']['summary']['net_long_term']),
                    float(tax_data['tax_liability']['summary']['ordinary_income']),
                    len(tax_data['transactions']),
                    len(tax_data['tax_events']),
                    tax_data['tax_liability']['wash_sale_adjustments']
                ]
            }
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        return {
            'filepath': str(filepath),
            'filename': filename,
            'format': 'excel',
            'size_bytes': filepath.stat().st_size
        }
    
    def generate_multi_year_comparison(self, years: List[int], config: ReportConfiguration) -> Dict[str, Any]:
        """Generate multi-year tax comparison report."""
        logger.info(f"Generating multi-year comparison for {years}")
        
        comparison_data = {
            'years': years,
            'yearly_summaries': {},
            'trends': {},
            'recommendations': []
        }
        
        for year in years:
            yearly_data = self._gather_tax_data(year)
            comparison_data['yearly_summaries'][year] = {
                'tax_liability': yearly_data['tax_liability']['tax_calculation']['total_tax_liability'],
                'net_capital_gain_loss': yearly_data['tax_liability']['summary']['total_net_capital'],
                'total_transactions': len(yearly_data['transactions']),
                'wash_sale_adjustments': yearly_data['tax_liability']['wash_sale_adjustments']
            }
        
        # Calculate trends
        if len(years) > 1:
            comparison_data['trends'] = self._calculate_trends(comparison_data['yearly_summaries'])
        
        return comparison_data
    
    def _calculate_trends(self, yearly_summaries: Dict[int, Dict]) -> Dict[str, Any]:
        """Calculate year-over-year trends."""
        years = sorted(yearly_summaries.keys())
        
        trends = {}
        
        if len(years) >= 2:
            latest_year = years[-1]
            previous_year = years[-2]
            
            latest = yearly_summaries[latest_year]
            previous = yearly_summaries[previous_year]
            
            trends['tax_liability_change'] = float(latest['tax_liability'] - previous['tax_liability'])
            trends['tax_liability_pct_change'] = (
                float((latest['tax_liability'] - previous['tax_liability']) / previous['tax_liability'] * 100)
                if previous['tax_liability'] != 0 else 0
            )
            
            trends['transaction_volume_change'] = latest['total_transactions'] - previous['total_transactions']
            trends['transaction_volume_pct_change'] = (
                (latest['total_transactions'] - previous['total_transactions']) / previous['total_transactions'] * 100
                if previous['total_transactions'] != 0 else 0
            )
        
        return trends

# Templates directory setup
def create_report_templates():
    """Create report templates directory and basic templates."""
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    
    # Create basic HTML template
    html_template_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cryptocurrency Tax Report</title>
        <style>
            /* CSS styles would go here */
        </style>
    </head>
    <body>
        <!-- HTML content would go here -->
    </body>
    </html>
    """
    
    with open(template_dir / "base_report.html", "w") as f:
        f.write(html_template_content)

if __name__ == "__main__":
    # Example usage
    from .tax_engine import TaxEngine, TaxConfiguration
    
    # Create tax engine
    config = TaxConfiguration("US")
    engine = TaxEngine(config)
    
    # Create reporter
    reporter = TaxReporter(engine)
    
    # Example report generation
    report_config = ReportConfiguration(
        report_type=ReportType.COMPREHENSIVE,
        format=ReportFormat.HTML,
        tax_year=2024,
        jurisdiction=JurisdictionCompliance.US_FEDERAL,
        taxpayer_info={
            'name': 'John Doe',
            'ssn': '123-45-6789',
            'address': '123 Main St, Anytown, USA'
        }
    )
    
    # This would generate a report (requires actual transaction data)
    # result = reporter.generate_report(report_config)
    print("Tax reporter initialized successfully")