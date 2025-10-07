"""
Australian Banking Manager
Handles banking relationships, transfer costs, and FX calculations for Australian traders
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Dict, List, Optional, Any
from enum import Enum
import requests
import logging

logger = logging.getLogger(__name__)

class AustralianBank(Enum):
    """Major Australian banks"""
    COMMONWEALTH = "cba"
    NAB = "nab"
    ANZ = "anz"
    WESTPAC = "westpac"
    MACQUARIE = "macquarie"
    BENDIGO = "bendigo"
    ING = "ing"

class TransferType(Enum):
    """Types of bank transfers"""
    DOMESTIC = "domestic"
    INTERNATIONAL = "international"
    SWIFT = "swift"
    RTGS = "rtgs"  # Real Time Gross Settlement
    OSKO = "osko"  # PayID/Osko instant transfers

@dataclass
class BankAccount:
    """Australian bank account details"""
    bank: AustralianBank
    account_name: str
    bsb: str
    account_number: str
    daily_limit: Decimal
    weekly_limit: Decimal
    monthly_limit: Decimal
    crypto_friendly: bool = False
    last_crypto_transfer: Optional[datetime] = None
    risk_rating: str = "LOW"  # LOW, MEDIUM, HIGH
    
class BankTransferCost:
    """Calculate costs for bank transfers"""
    
    def __init__(self, amount: Decimal, bank: AustralianBank, transfer_type: TransferType):
        self.amount = amount
        self.bank = bank
        self.transfer_type = transfer_type
        self.base_fee = self._get_base_fee()
        self.fx_spread = self._get_fx_spread()
        self.processing_days = self._get_processing_days()
        
    def _get_base_fee(self) -> Decimal:
        """Get base transfer fee by bank and type"""
        fees = {
            AustralianBank.COMMONWEALTH: {
                TransferType.DOMESTIC: Decimal('0'),
                TransferType.INTERNATIONAL: Decimal('6'),
                TransferType.SWIFT: Decimal('25'),
                TransferType.RTGS: Decimal('20')
            },
            AustralianBank.NAB: {
                TransferType.DOMESTIC: Decimal('0'),
                TransferType.INTERNATIONAL: Decimal('10'),
                TransferType.SWIFT: Decimal('30'),
                TransferType.RTGS: Decimal('25')
            },
            AustralianBank.ANZ: {
                TransferType.DOMESTIC: Decimal('0'),
                TransferType.INTERNATIONAL: Decimal('9'),
                TransferType.SWIFT: Decimal('32'),
                TransferType.RTGS: Decimal('22')
            },
            AustralianBank.WESTPAC: {
                TransferType.DOMESTIC: Decimal('0'),
                TransferType.INTERNATIONAL: Decimal('12'),
                TransferType.SWIFT: Decimal('35'),
                TransferType.RTGS: Decimal('30')
            },
            AustralianBank.MACQUARIE: {
                TransferType.DOMESTIC: Decimal('0'),
                TransferType.INTERNATIONAL: Decimal('0'),  # Often waived
                TransferType.SWIFT: Decimal('20'),
                TransferType.RTGS: Decimal('15')
            }
        }
        
        return fees.get(self.bank, {}).get(self.transfer_type, Decimal('25'))
    
    def _get_fx_spread(self) -> Decimal:
        """Get FX spread percentage by bank"""
        spreads = {
            AustralianBank.COMMONWEALTH: Decimal('0.015'),  # 1.5%
            AustralianBank.NAB: Decimal('0.020'),           # 2.0%
            AustralianBank.ANZ: Decimal('0.018'),           # 1.8%
            AustralianBank.WESTPAC: Decimal('0.022'),       # 2.2%
            AustralianBank.MACQUARIE: Decimal('0.008')      # 0.8% (better rates)
        }
        
        return spreads.get(self.bank, Decimal('0.020'))
    
    def _get_processing_days(self) -> int:
        """Get processing time in business days"""
        processing_times = {
            TransferType.DOMESTIC: 0,      # Same day
            TransferType.OSKO: 0,          # Instant
            TransferType.INTERNATIONAL: 3,  # 1-3 days
            TransferType.SWIFT: 5,         # 3-5 days
            TransferType.RTGS: 0           # Same day
        }
        
        return processing_times.get(self.transfer_type, 3)
    
    def calculate_total_cost(self, opportunity_cost_rate: Decimal = Decimal('0.05')) -> Dict[str, Decimal]:
        """Calculate total cost including opportunity cost"""
        # FX spread cost
        fx_cost = self.amount * self.fx_spread if self.transfer_type != TransferType.DOMESTIC else Decimal('0')
        
        # Opportunity cost (annual rate applied to transfer days)
        daily_rate = opportunity_cost_rate / Decimal('365')
        opportunity_cost = self.amount * daily_rate * Decimal(str(self.processing_days))
        
        total_cost = self.base_fee + fx_cost + opportunity_cost
        
        return {
            'base_fee': self.base_fee,
            'fx_spread_cost': fx_cost,
            'opportunity_cost': opportunity_cost,
            'total_cost': total_cost,
            'cost_percentage': (total_cost / self.amount * Decimal('100')) if self.amount > 0 else Decimal('0')
        }

class AustralianBankingManager:
    """
    Manages Australian banking relationships and transfer optimization
    """
    
    def __init__(self):
        self.bank_accounts: Dict[AustralianBank, BankAccount] = {}
        self.transfer_history: List[Dict] = []
        self.crypto_friendly_banks = [
            AustralianBank.MACQUARIE,  # Most crypto-friendly
            AustralianBank.ING,        # Generally crypto-tolerant
            AustralianBank.BENDIGO     # Regional bank, more flexible
        ]
        
        # Current FX rate (would be updated from API)
        self.aud_usd_rate = Decimal('0.67')  # Example rate
        
        # Bank transfer limits (conservative estimates)
        self.bank_transfer_limits = {
            AustralianBank.COMMONWEALTH: {
                'daily': Decimal('25000'),
                'weekly': Decimal('100000'),
                'monthly': Decimal('400000')
            },
            AustralianBank.NAB: {
                'daily': Decimal('20000'),
                'weekly': Decimal('50000'),
                'monthly': Decimal('200000')
            },
            AustralianBank.ANZ: {
                'daily': Decimal('50000'),
                'weekly': Decimal('200000'),
                'monthly': Decimal('800000')
            },
            AustralianBank.WESTPAC: {
                'daily': Decimal('15000'),
                'weekly': Decimal('75000'),
                'monthly': Decimal('300000')
            },
            AustralianBank.MACQUARIE: {
                'daily': Decimal('100000'),
                'weekly': Decimal('500000'),
                'monthly': Decimal('2000000')
            }
        }
        
        logger.info("Initialized Australian Banking Manager")
    
    def add_bank_account(self, account: BankAccount) -> None:
        """Add a bank account to the manager"""
        self.bank_accounts[account.bank] = account
        logger.info(f"Added bank account: {account.bank.value}")
    
    def get_current_fx_rate(self) -> Decimal:
        """Get current AUD/USD exchange rate"""
        try:
            # In production, this would call a real FX API
            # For now, return a static rate
            return self.aud_usd_rate
        except Exception as e:
            logger.error(f"Failed to get FX rate: {e}")
            return Decimal('0.67')  # Fallback rate
    
    def calculate_aud_conversion_costs(
        self,
        amount_aud: Decimal,
        bank: AustralianBank = AustralianBank.COMMONWEALTH
    ) -> Dict[str, Any]:
        """
        Calculate costs of moving AUD to exchanges and back
        """
        
        # Bank transfer costs
        transfer_cost = BankTransferCost(
            amount=amount_aud,
            bank=bank,
            transfer_type=TransferType.SWIFT
        )
        
        cost_breakdown = transfer_cost.calculate_total_cost()
        
        # Add regulatory and compliance costs
        aml_cost = Decimal('0')
        if amount_aud > Decimal('10000'):  # AUSTRAC reporting threshold
            aml_cost = Decimal('50')  # Estimated compliance cost
        
        # Risk premium for crypto-related transfers
        crypto_risk_premium = amount_aud * Decimal('0.001')  # 0.1%
        
        total_cost = cost_breakdown['total_cost'] + aml_cost + crypto_risk_premium
        
        return {
            'amount_aud': amount_aud,
            'bank': bank.value,
            'transfer_costs': cost_breakdown,
            'aml_compliance_cost': aml_cost,
            'crypto_risk_premium': crypto_risk_premium,
            'total_cost': total_cost,
            'total_cost_percentage': (total_cost / amount_aud * Decimal('100')),
            'net_amount': amount_aud - total_cost,
            'processing_days': transfer_cost.processing_days,
            'fx_rate_used': self.aud_usd_rate,
            'usd_equivalent': amount_aud * self.aud_usd_rate
        }
    
    def optimize_transfer_route(
        self,
        amount_aud: Decimal,
        urgency: str = "normal"  # "urgent", "normal", "low"
    ) -> Dict[str, Any]:
        """
        Find the optimal transfer route considering cost and speed
        """
        
        routes = []
        
        for bank in self.bank_accounts.keys():
            # Determine transfer type based on urgency
            if urgency == "urgent":
                transfer_type = TransferType.RTGS
            elif urgency == "normal":
                transfer_type = TransferType.SWIFT
            else:
                transfer_type = TransferType.INTERNATIONAL
            
            cost_analysis = self.calculate_aud_conversion_costs(amount_aud, bank)
            
            routes.append({
                'bank': bank,
                'transfer_type': transfer_type,
                'total_cost': cost_analysis['total_cost'],
                'cost_percentage': cost_analysis['total_cost_percentage'],
                'processing_days': cost_analysis['processing_days'],
                'crypto_friendly': bank in self.crypto_friendly_banks,
                'recommended': False
            })
        
        # Sort by total cost
        routes.sort(key=lambda x: x['total_cost'])
        
        # Mark the best route as recommended
        if routes:
            routes[0]['recommended'] = True
        
        return {
            'amount_aud': amount_aud,
            'urgency': urgency,
            'routes': routes,
            'best_route': routes[0] if routes else None,
            'cost_savings': routes[-1]['total_cost'] - routes[0]['total_cost'] if len(routes) > 1 else Decimal('0')
        }
    
    def check_transfer_limits(
        self,
        amount_aud: Decimal,
        bank: AustralianBank,
        period: str = "daily"
    ) -> Dict[str, Any]:
        """
        Check if transfer amount is within bank limits
        """
        
        limits = self.bank_transfer_limits.get(bank, {})
        limit = limits.get(period, Decimal('10000'))  # Default conservative limit
        
        within_limit = amount_aud <= limit
        utilization_percentage = (amount_aud / limit * Decimal('100')) if limit > 0 else Decimal('100')
        
        return {
            'bank': bank.value,
            'amount_aud': amount_aud,
            'period': period,
            'limit': limit,
            'within_limit': within_limit,
            'utilization_percentage': utilization_percentage,
            'remaining_limit': limit - amount_aud if within_limit else Decimal('0'),
            'recommendation': self._get_limit_recommendation(utilization_percentage)
        }
    
    def _get_limit_recommendation(self, utilization: Decimal) -> str:
        """Get recommendation based on limit utilization"""
        if utilization < Decimal('50'):
            return "SAFE - Well within limits"
        elif utilization < Decimal('80'):
            return "CAUTION - Approaching limits"
        elif utilization < Decimal('100'):
            return "WARNING - Near limit, consider spreading across banks"
        else:
            return "BLOCKED - Exceeds bank limits"
    
    def manage_banking_risks(self) -> Dict[str, Any]:
        """
        Implement banking risk management strategies
        """
        
        risk_management = {
            'diversification_strategy': {
                'primary_bank': AustralianBank.MACQUARIE,  # Most crypto-friendly
                'primary_allocation': Decimal('0.4'),
                'secondary_bank': AustralianBank.COMMONWEALTH,  # Largest bank
                'secondary_allocation': Decimal('0.3'),
                'tertiary_bank': AustralianBank.ING,  # Online bank
                'tertiary_allocation': Decimal('0.3')
            },
            
            'relationship_maintenance': {
                'maintain_traditional_banking': True,
                'regular_non_crypto_transactions': True,
                'transparent_communication': True,
                'proper_documentation': True
            },
            
            'otc_desk_relationships': {
                'independent_reserve': {'established': False, 'volume_threshold': Decimal('50000')},
                'coinjar': {'established': False, 'volume_threshold': Decimal('25000')},
                'digital_surge': {'established': False, 'volume_threshold': Decimal('10000')}
            },
            
            'backup_strategies': [
                'Multiple bank relationships',
                'OTC desk partnerships',
                'International bank accounts',
                'Stablecoin bridges'
            ]
        }
        
        return risk_management
    
    def generate_banking_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive banking relationship report
        """
        
        return {
            'account_summary': {
                'total_accounts': len(self.bank_accounts),
                'crypto_friendly_accounts': len([b for b in self.bank_accounts.keys() if b in self.crypto_friendly_banks]),
                'total_daily_limits': sum(self.bank_transfer_limits.get(b, {}).get('daily', Decimal('0')) for b in self.bank_accounts.keys())
            },
            
            'risk_assessment': {
                'banking_relationship_risk': 'MEDIUM',
                'regulatory_compliance': 'COMPLIANT',
                'transfer_capacity': 'ADEQUATE',
                'recommendations': [
                    'Establish OTC desk relationships for large transfers',
                    'Maintain transparent communication with banks',
                    'Consider international banking relationships'
                ]
            },
            
            'cost_analysis': {
                'average_transfer_cost': Decimal('0.025'),  # 2.5%
                'monthly_transfer_savings_potential': Decimal('500'),
                'optimal_bank_utilization': 'Macquarie for large transfers, CBA for regular banking'
            }
        }
    
    def schedule_transfer(
        self,
        amount_aud: Decimal,
        target_date: date,
        bank: AustralianBank = None
    ) -> Dict[str, Any]:
        """
        Schedule a transfer considering Australian business days
        """
        
        # Australian public holidays (simplified)
        australian_holidays = [
            date(2024, 1, 1),   # New Year's Day
            date(2024, 1, 26),  # Australia Day
            date(2024, 3, 29),  # Good Friday
            date(2024, 4, 1),   # Easter Monday
            date(2024, 4, 25),  # ANZAC Day
            date(2024, 6, 10),  # Queen's Birthday
            date(2024, 12, 25), # Christmas Day
            date(2024, 12, 26), # Boxing Day
        ]
        
        # Find next business day
        current_date = target_date
        while current_date.weekday() >= 5 or current_date in australian_holidays:  # Weekend or holiday
            current_date += timedelta(days=1)
        
        # Select optimal bank if not specified
        if not bank:
            optimal_route = self.optimize_transfer_route(amount_aud)
            bank = optimal_route['best_route']['bank']
        
        # Calculate processing time
        transfer_cost = BankTransferCost(amount_aud, bank, TransferType.SWIFT)
        completion_date = current_date + timedelta(days=transfer_cost.processing_days)
        
        return {
            'scheduled_date': current_date,
            'completion_date': completion_date,
            'amount_aud': amount_aud,
            'selected_bank': bank.value,
            'business_days_to_complete': transfer_cost.processing_days,
            'cost_estimate': transfer_cost.calculate_total_cost(),
            'status': 'SCHEDULED'
        }

# Usage example
if __name__ == "__main__":
    banking_manager = AustralianBankingManager()
    
    # Add bank accounts
    cba_account = BankAccount(
        bank=AustralianBank.COMMONWEALTH,
        account_name="Trading Account",
        bsb="062-000",
        account_number="12345678",
        daily_limit=Decimal('25000'),
        weekly_limit=Decimal('100000'),
        monthly_limit=Decimal('400000'),
        crypto_friendly=False
    )
    
    macquarie_account = BankAccount(
        bank=AustralianBank.MACQUARIE,
        account_name="Investment Account",
        bsb="182-512",
        account_number="87654321",
        daily_limit=Decimal('100000'),
        weekly_limit=Decimal('500000'),
        monthly_limit=Decimal('2000000'),
        crypto_friendly=True
    )
    
    banking_manager.add_bank_account(cba_account)
    banking_manager.add_bank_account(macquarie_account)
    
    # Calculate transfer costs
    cost_analysis = banking_manager.calculate_aud_conversion_costs(Decimal('10000'))
    print(f"Transfer cost: ${cost_analysis['total_cost']} AUD ({cost_analysis['total_cost_percentage']:.2f}%)")
    
    # Optimize transfer route
    optimal_route = banking_manager.optimize_transfer_route(Decimal('50000'))
    print(f"Best route: {optimal_route['best_route']['bank']} with cost ${optimal_route['best_route']['total_cost']}")
    
    # Generate banking report
    report = banking_manager.generate_banking_report()
    print(f"Banking relationship risk: {report['risk_assessment']['banking_relationship_risk']}")