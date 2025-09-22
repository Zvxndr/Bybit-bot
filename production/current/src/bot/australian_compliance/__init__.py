"""
Australian Compliance Module
Provides integrated compliance management for Australian cryptocurrency traders
"""

from .ato_integration import AustralianTaxCalculator, Trade, CGTEvent
from .banking_manager import AustralianBankingManager, BankAccount, AustralianBank
from .regulatory_compliance import AustralianComplianceManager, BusinessStructure

__all__ = [
    'AustralianTaxCalculator',
    'AustralianBankingManager', 
    'AustralianComplianceManager',
    'Trade',
    'CGTEvent',
    'BankAccount',
    'BusinessStructure',
    'AustralianBank'
]