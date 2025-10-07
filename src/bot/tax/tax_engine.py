"""
Advanced Tax Calculation Engine for Cryptocurrency Trading

This module provides comprehensive tax calculation capabilities including:
- Multiple accounting methods (FIFO, LIFO, Specific Identification)
- Wash sale detection and adjustments
- Tax-loss harvesting optimization
- Short-term vs long-term capital gains classification
- Staking and DeFi rewards handling
- Multi-jurisdiction support
- Real-time tax impact analysis
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class AccountingMethod(Enum):
    """Supported accounting methods for cost basis calculation."""
    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    SPECIFIC_ID = "specific_id"  # Specific Identification
    AVERAGE_COST = "average_cost"  # Average Cost Basis
    HIFO = "hifo"  # Highest In, First Out (for tax optimization)

class TransactionType(Enum):
    """Types of cryptocurrency transactions."""
    BUY = "buy"
    SELL = "sell"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    STAKE = "stake"
    UNSTAKE = "unstake"
    REWARD = "reward"
    AIRDROP = "airdrop"
    FORK = "fork"
    MINING = "mining"
    DeFi_YIELD = "defi_yield"
    NFT_MINT = "nft_mint"
    NFT_SALE = "nft_sale"

class TaxEventType(Enum):
    """Types of taxable events."""
    CAPITAL_GAIN = "capital_gain"
    CAPITAL_LOSS = "capital_loss"
    ORDINARY_INCOME = "ordinary_income"
    WASH_SALE_ADJUSTMENT = "wash_sale_adjustment"
    TAX_LOSS_HARVEST = "tax_loss_harvest"

class HoldingPeriod(Enum):
    """Classification of holding periods for tax purposes."""
    SHORT_TERM = "short_term"  # <= 1 year
    LONG_TERM = "long_term"    # > 1 year

@dataclass
class Transaction:
    """Represents a cryptocurrency transaction."""
    id: str
    timestamp: datetime
    transaction_type: TransactionType
    asset: str
    quantity: Decimal
    price_per_unit: Decimal
    total_value: Decimal
    fees: Decimal = Decimal('0')
    exchange: str = ""
    wallet_address: str = ""
    transaction_hash: str = ""
    notes: str = ""
    
    def __post_init__(self):
        """Validate transaction data."""
        if self.quantity <= 0:
            raise ValueError(f"Transaction quantity must be positive: {self.quantity}")
        if self.price_per_unit < 0:
            raise ValueError(f"Price per unit cannot be negative: {self.price_per_unit}")
        if self.total_value < 0:
            raise ValueError(f"Total value cannot be negative: {self.total_value}")

@dataclass
class TaxLot:
    """Represents a tax lot (FIFO queue entry) for cost basis tracking."""
    acquisition_date: datetime
    asset: str
    quantity: Decimal
    cost_basis_per_unit: Decimal
    total_cost_basis: Decimal
    transaction_id: str
    is_wash_sale_affected: bool = False
    wash_sale_adjustment: Decimal = Decimal('0')
    
    @property
    def adjusted_cost_basis_per_unit(self) -> Decimal:
        """Cost basis per unit after wash sale adjustments."""
        return self.cost_basis_per_unit + self.wash_sale_adjustment
    
    @property
    def adjusted_total_cost_basis(self) -> Decimal:
        """Total cost basis after wash sale adjustments."""
        return self.quantity * self.adjusted_cost_basis_per_unit

@dataclass
class TaxEvent:
    """Represents a taxable event."""
    id: str
    timestamp: datetime
    event_type: TaxEventType
    asset: str
    quantity: Decimal
    proceeds: Decimal
    cost_basis: Decimal
    gain_loss: Decimal
    holding_period: HoldingPeriod
    transaction_id: str
    wash_sale_affected: bool = False
    wash_sale_disallowed: Decimal = Decimal('0')
    tax_rate: Optional[Decimal] = None
    
    @property
    def is_gain(self) -> bool:
        """Check if this is a capital gain."""
        return self.gain_loss > 0
    
    @property
    def is_loss(self) -> bool:
        """Check if this is a capital loss."""
        return self.gain_loss < 0

@dataclass
class WashSaleRule:
    """Configuration for wash sale rule application."""
    enabled: bool = True
    lookback_days: int = 30
    lookforward_days: int = 30
    substantially_identical_threshold: Decimal = Decimal('0.95')  # Correlation threshold
    
class TaxConfiguration:
    """Tax calculation configuration and parameters."""
    
    def __init__(self, jurisdiction: str = "US"):
        self.jurisdiction = jurisdiction
        self.accounting_method = AccountingMethod.FIFO
        self.wash_sale_rule = WashSaleRule()
        
        # Tax rates (US federal rates for 2024-2025)
        self.short_term_tax_rates = {
            'single': [
                (0, Decimal('0.10')),
                (11000, Decimal('0.12')),
                (44725, Decimal('0.22')),
                (95375, Decimal('0.24')),
                (182050, Decimal('0.32')),
                (231250, Decimal('0.35')),
                (578125, Decimal('0.37'))
            ]
        }
        
        self.long_term_tax_rates = {
            'single': [
                (0, Decimal('0.00')),
                (44625, Decimal('0.15')),
                (492300, Decimal('0.20'))
            ]
        }
        
        # Net investment income tax (NIIT) - 3.8% on investment income
        self.niit_threshold = {'single': 200000, 'married': 250000}
        self.niit_rate = Decimal('0.038')
        
        # Alternative Minimum Tax considerations
        self.amt_exemption = {'single': 81300, 'married': 126500}
        
    def get_tax_rate(self, income: Decimal, holding_period: HoldingPeriod, 
                     filing_status: str = 'single') -> Decimal:
        """Calculate applicable tax rate based on income and holding period."""
        rates = (self.long_term_tax_rates if holding_period == HoldingPeriod.LONG_TERM 
                else self.short_term_tax_rates)
        
        if filing_status not in rates:
            filing_status = 'single'
        
        for threshold, rate in reversed(rates[filing_status]):
            if income >= threshold:
                return rate
        
        return rates[filing_status][0][1]  # Lowest rate

class TaxEngine:
    """Advanced cryptocurrency tax calculation engine."""
    
    def __init__(self, config: TaxConfiguration, db_path: str = "tax_records.db"):
        self.config = config
        self.db_path = db_path
        
        # In-memory data structures
        self.tax_lots: Dict[str, deque] = defaultdict(deque)  # Asset -> deque of TaxLot
        self.transactions: List[Transaction] = []
        self.tax_events: List[TaxEvent] = []
        self.wash_sale_adjustments: List[Dict] = []
        
        # Performance tracking
        self.realized_gains: Dict[str, Decimal] = defaultdict(Decimal)  # Asset -> realized gains
        self.unrealized_gains: Dict[str, Decimal] = defaultdict(Decimal)  # Asset -> unrealized gains
        self.total_cost_basis: Dict[str, Decimal] = defaultdict(Decimal)  # Asset -> total cost basis
        
        # Initialize database
        self._init_database()
        
        logger.info(f"TaxEngine initialized with {config.jurisdiction} jurisdiction, "
                   f"{config.accounting_method.value} accounting method")
    
    def _init_database(self):
        """Initialize SQLite database for tax record persistence."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS transactions (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        transaction_type TEXT NOT NULL,
                        asset TEXT NOT NULL,
                        quantity TEXT NOT NULL,
                        price_per_unit TEXT NOT NULL,
                        total_value TEXT NOT NULL,
                        fees TEXT DEFAULT '0',
                        exchange TEXT DEFAULT '',
                        wallet_address TEXT DEFAULT '',
                        transaction_hash TEXT DEFAULT '',
                        notes TEXT DEFAULT '',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE IF NOT EXISTS tax_lots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        acquisition_date TEXT NOT NULL,
                        asset TEXT NOT NULL,
                        quantity TEXT NOT NULL,
                        cost_basis_per_unit TEXT NOT NULL,
                        total_cost_basis TEXT NOT NULL,
                        transaction_id TEXT NOT NULL,
                        is_wash_sale_affected BOOLEAN DEFAULT FALSE,
                        wash_sale_adjustment TEXT DEFAULT '0',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (transaction_id) REFERENCES transactions (id)
                    );
                    
                    CREATE TABLE IF NOT EXISTS tax_events (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        asset TEXT NOT NULL,
                        quantity TEXT NOT NULL,
                        proceeds TEXT NOT NULL,
                        cost_basis TEXT NOT NULL,
                        gain_loss TEXT NOT NULL,
                        holding_period TEXT NOT NULL,
                        transaction_id TEXT NOT NULL,
                        wash_sale_affected BOOLEAN DEFAULT FALSE,
                        wash_sale_disallowed TEXT DEFAULT '0',
                        tax_rate TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (transaction_id) REFERENCES transactions (id)
                    );
                    
                    CREATE TABLE IF NOT EXISTS wash_sale_adjustments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sale_transaction_id TEXT NOT NULL,
                        purchase_transaction_id TEXT NOT NULL,
                        asset TEXT NOT NULL,
                        disallowed_loss TEXT NOT NULL,
                        adjustment_date TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_transactions_asset_timestamp 
                        ON transactions (asset, timestamp);
                    CREATE INDEX IF NOT EXISTS idx_tax_lots_asset 
                        ON tax_lots (asset, acquisition_date);
                    CREATE INDEX IF NOT EXISTS idx_tax_events_asset_timestamp 
                        ON tax_events (asset, timestamp);
                """)
                
            logger.info("Tax database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tax database: {e}")
            raise
    
    def add_transaction(self, transaction: Transaction):
        """Add a new transaction and process tax implications."""
        logger.debug(f"Adding transaction: {transaction.id} - {transaction.transaction_type.value}")
        
        # Validate transaction
        self._validate_transaction(transaction)
        
        # Store transaction
        self.transactions.append(transaction)
        self._save_transaction_to_db(transaction)
        
        # Process tax implications based on transaction type
        if transaction.transaction_type in [TransactionType.BUY, TransactionType.TRANSFER_IN,
                                          TransactionType.REWARD, TransactionType.AIRDROP,
                                          TransactionType.MINING, TransactionType.DeFi_YIELD]:
            self._process_acquisition(transaction)
            
        elif transaction.transaction_type in [TransactionType.SELL, TransactionType.TRANSFER_OUT]:
            self._process_disposal(transaction)
            
        # Update performance tracking
        self._update_performance_metrics(transaction)
        
        logger.info(f"Transaction {transaction.id} processed successfully")
    
    def _validate_transaction(self, transaction: Transaction):
        """Validate transaction data and business rules."""
        # Check for duplicate transaction IDs
        if any(t.id == transaction.id for t in self.transactions):
            raise ValueError(f"Transaction ID {transaction.id} already exists")
        
        # Validate timestamp
        if transaction.timestamp > datetime.now():
            raise ValueError(f"Transaction timestamp cannot be in the future: {transaction.timestamp}")
        
        # Validate asset symbol
        if not transaction.asset or len(transaction.asset) < 2:
            raise ValueError(f"Invalid asset symbol: {transaction.asset}")
        
        # Validate amounts
        if transaction.quantity <= 0:
            raise ValueError(f"Transaction quantity must be positive: {transaction.quantity}")
        
        # Type-specific validations
        if transaction.transaction_type == TransactionType.SELL:
            available_quantity = self._get_available_quantity(transaction.asset)
            if transaction.quantity > available_quantity:
                logger.warning(f"Selling more than available quantity for {transaction.asset}: "
                             f"{transaction.quantity} > {available_quantity}")
    
    def _process_acquisition(self, transaction: Transaction):
        """Process an acquisition transaction (buy, reward, etc.)."""
        cost_basis_per_unit = transaction.price_per_unit
        
        # For rewards, airdrops, mining - use fair market value at time of receipt
        if transaction.transaction_type in [TransactionType.REWARD, TransactionType.AIRDROP,
                                          TransactionType.MINING, TransactionType.DeFi_YIELD]:
            # Create taxable income event
            tax_event = TaxEvent(
                id=f"income_{transaction.id}",
                timestamp=transaction.timestamp,
                event_type=TaxEventType.ORDINARY_INCOME,
                asset=transaction.asset,
                quantity=transaction.quantity,
                proceeds=transaction.total_value,
                cost_basis=Decimal('0'),
                gain_loss=transaction.total_value,
                holding_period=HoldingPeriod.SHORT_TERM,
                transaction_id=transaction.id
            )
            self.tax_events.append(tax_event)
            self._save_tax_event_to_db(tax_event)
        
        # Create tax lot
        tax_lot = TaxLot(
            acquisition_date=transaction.timestamp,
            asset=transaction.asset,
            quantity=transaction.quantity,
            cost_basis_per_unit=cost_basis_per_unit,
            total_cost_basis=transaction.quantity * cost_basis_per_unit,
            transaction_id=transaction.id
        )
        
        # Add to appropriate position in queue based on accounting method
        if self.config.accounting_method == AccountingMethod.FIFO:
            self.tax_lots[transaction.asset].append(tax_lot)
        elif self.config.accounting_method == AccountingMethod.LIFO:
            self.tax_lots[transaction.asset].appendleft(tax_lot)
        elif self.config.accounting_method == AccountingMethod.HIFO:
            # Insert in descending cost basis order
            lots = list(self.tax_lots[transaction.asset])
            lots.append(tax_lot)
            lots.sort(key=lambda x: x.cost_basis_per_unit, reverse=True)
            self.tax_lots[transaction.asset] = deque(lots)
        else:
            self.tax_lots[transaction.asset].append(tax_lot)
        
        self._save_tax_lot_to_db(tax_lot)
        
        logger.debug(f"Created tax lot for {transaction.asset}: {transaction.quantity} @ {cost_basis_per_unit}")
    
    def _process_disposal(self, transaction: Transaction):
        """Process a disposal transaction (sell, transfer out)."""
        remaining_quantity = transaction.quantity
        total_cost_basis = Decimal('0')
        disposal_lots = []
        
        # Calculate cost basis using specified accounting method
        while remaining_quantity > 0 and self.tax_lots[transaction.asset]:
            if self.config.accounting_method in [AccountingMethod.FIFO, AccountingMethod.HIFO]:
                lot = self.tax_lots[transaction.asset].popleft()
            elif self.config.accounting_method == AccountingMethod.LIFO:
                lot = self.tax_lots[transaction.asset].pop()
            else:  # FIFO as default
                lot = self.tax_lots[transaction.asset].popleft()
            
            if lot.quantity <= remaining_quantity:
                # Use entire lot
                disposal_quantity = lot.quantity
                disposal_cost_basis = lot.adjusted_total_cost_basis
                remaining_quantity -= disposal_quantity
            else:
                # Partial lot usage
                disposal_quantity = remaining_quantity
                disposal_cost_basis = disposal_quantity * lot.adjusted_cost_basis_per_unit
                
                # Return unused portion to queue
                remaining_lot = TaxLot(
                    acquisition_date=lot.acquisition_date,
                    asset=lot.asset,
                    quantity=lot.quantity - disposal_quantity,
                    cost_basis_per_unit=lot.cost_basis_per_unit,
                    total_cost_basis=(lot.quantity - disposal_quantity) * lot.cost_basis_per_unit,
                    transaction_id=lot.transaction_id,
                    is_wash_sale_affected=lot.is_wash_sale_affected,
                    wash_sale_adjustment=lot.wash_sale_adjustment
                )
                
                if self.config.accounting_method in [AccountingMethod.FIFO, AccountingMethod.HIFO]:
                    self.tax_lots[transaction.asset].appendleft(remaining_lot)
                else:
                    self.tax_lots[transaction.asset].append(remaining_lot)
                
                remaining_quantity = Decimal('0')
            
            disposal_lots.append({
                'lot': lot,
                'disposal_quantity': disposal_quantity,
                'disposal_cost_basis': disposal_cost_basis
            })
            
            total_cost_basis += disposal_cost_basis
        
        if remaining_quantity > 0:
            logger.warning(f"Insufficient inventory for {transaction.asset}: "
                         f"trying to dispose {transaction.quantity}, "
                         f"but only have cost basis for {transaction.quantity - remaining_quantity}")
        
        # Calculate gain/loss for taxable disposals
        if transaction.transaction_type == TransactionType.SELL:
            proceeds = transaction.total_value - transaction.fees
            gain_loss = proceeds - total_cost_basis
            
            # Determine holding period (use weighted average)
            total_disposal_quantity = transaction.quantity - remaining_quantity
            weighted_holding_days = Decimal('0')
            
            for disposal in disposal_lots:
                weight = disposal['disposal_quantity'] / total_disposal_quantity
                holding_days = (transaction.timestamp - disposal['lot'].acquisition_date).days
                weighted_holding_days += weight * holding_days
            
            holding_period = (HoldingPeriod.LONG_TERM if weighted_holding_days > 365 
                            else HoldingPeriod.SHORT_TERM)
            
            # Create tax event
            tax_event = TaxEvent(
                id=f"disposal_{transaction.id}",
                timestamp=transaction.timestamp,
                event_type=TaxEventType.CAPITAL_GAIN if gain_loss >= 0 else TaxEventType.CAPITAL_LOSS,
                asset=transaction.asset,
                quantity=total_disposal_quantity,
                proceeds=proceeds,
                cost_basis=total_cost_basis,
                gain_loss=gain_loss,
                holding_period=holding_period,
                transaction_id=transaction.id
            )
            
            # Check for wash sale if it's a loss
            if gain_loss < 0 and self.config.wash_sale_rule.enabled:
                self._check_wash_sale(tax_event, transaction)
            
            self.tax_events.append(tax_event)
            self._save_tax_event_to_db(tax_event)
            
            logger.debug(f"Created tax event for {transaction.asset} disposal: "
                        f"gain/loss = {gain_loss}, holding period = {holding_period.value}")
    
    def _check_wash_sale(self, tax_event: TaxEvent, sale_transaction: Transaction):
        """Check for wash sale rule violations."""
        if tax_event.gain_loss >= 0:
            return  # Wash sale rule only applies to losses
        
        wash_sale_window_start = (sale_transaction.timestamp - 
                                timedelta(days=self.config.wash_sale_rule.lookback_days))
        wash_sale_window_end = (sale_transaction.timestamp + 
                              timedelta(days=self.config.wash_sale_rule.lookforward_days))
        
        # Find substantially identical purchases within wash sale window
        wash_sale_purchases = []
        
        for transaction in self.transactions:
            if (transaction.asset == sale_transaction.asset and
                transaction.transaction_type in [TransactionType.BUY, TransactionType.TRANSFER_IN] and
                wash_sale_window_start <= transaction.timestamp <= wash_sale_window_end and
                transaction.id != sale_transaction.id):
                
                wash_sale_purchases.append(transaction)
        
        if wash_sale_purchases:
            # Calculate disallowed loss
            total_wash_purchase_quantity = sum(t.quantity for t in wash_sale_purchases)
            disallowed_quantity = min(tax_event.quantity, total_wash_purchase_quantity)
            disallowed_loss = abs(tax_event.gain_loss) * (disallowed_quantity / tax_event.quantity)
            
            # Update tax event
            tax_event.wash_sale_affected = True
            tax_event.wash_sale_disallowed = disallowed_loss
            tax_event.gain_loss += disallowed_loss  # Reduce the loss
            
            # Adjust cost basis of replacement shares
            self._apply_wash_sale_adjustments(wash_sale_purchases, disallowed_loss, sale_transaction)
            
            logger.info(f"Wash sale detected for {sale_transaction.asset}: "
                       f"disallowed loss = {disallowed_loss}")
    
    def _apply_wash_sale_adjustments(self, purchase_transactions: List[Transaction], 
                                   disallowed_loss: Decimal, sale_transaction: Transaction):
        """Apply wash sale adjustments to replacement share cost basis."""
        total_purchase_quantity = sum(t.quantity for t in purchase_transactions)
        
        for purchase_transaction in purchase_transactions:
            # Calculate proportional adjustment
            weight = purchase_transaction.quantity / total_purchase_quantity
            adjustment_per_unit = (disallowed_loss * weight) / purchase_transaction.quantity
            
            # Find and adjust corresponding tax lots
            for lot in self.tax_lots[purchase_transaction.asset]:
                if lot.transaction_id == purchase_transaction.id:
                    lot.is_wash_sale_affected = True
                    lot.wash_sale_adjustment += adjustment_per_unit
                    
                    # Update database
                    self._update_tax_lot_wash_sale(lot)
            
            # Record wash sale adjustment
            adjustment_record = {
                'sale_transaction_id': sale_transaction.id,
                'purchase_transaction_id': purchase_transaction.id,
                'asset': purchase_transaction.asset,
                'disallowed_loss': disallowed_loss * weight,
                'adjustment_date': datetime.now()
            }
            
            self.wash_sale_adjustments.append(adjustment_record)
            self._save_wash_sale_adjustment(adjustment_record)
    
    def calculate_tax_liability(self, tax_year: int, income: Decimal = Decimal('0'), 
                              filing_status: str = 'single') -> Dict[str, Any]:
        """Calculate comprehensive tax liability for a given tax year."""
        logger.info(f"Calculating tax liability for {tax_year}")
        
        # Filter tax events for the specified year
        year_events = [event for event in self.tax_events 
                      if event.timestamp.year == tax_year]
        
        # Categorize gains and losses
        short_term_gains = Decimal('0')
        short_term_losses = Decimal('0')
        long_term_gains = Decimal('0')
        long_term_losses = Decimal('0')
        ordinary_income = Decimal('0')
        
        for event in year_events:
            if event.event_type == TaxEventType.ORDINARY_INCOME:
                ordinary_income += event.gain_loss
            elif event.holding_period == HoldingPeriod.SHORT_TERM:
                if event.gain_loss > 0:
                    short_term_gains += event.gain_loss
                else:
                    short_term_losses += abs(event.gain_loss)
            else:  # Long-term
                if event.gain_loss > 0:
                    long_term_gains += event.gain_loss
                else:
                    long_term_losses += abs(event.gain_loss)
        
        # Calculate net capital gains/losses
        net_short_term = short_term_gains - short_term_losses
        net_long_term = long_term_gains - long_term_losses
        
        # Apply capital loss limitations
        total_net_capital = net_short_term + net_long_term
        capital_loss_deduction = min(abs(total_net_capital), Decimal('3000')) if total_net_capital < 0 else Decimal('0')
        capital_loss_carryforward = max(abs(total_net_capital) - Decimal('3000'), Decimal('0')) if total_net_capital < 0 else Decimal('0')
        
        # Calculate taxable amounts
        taxable_short_term = max(net_short_term, Decimal('0'))
        taxable_long_term = max(net_long_term, Decimal('0'))
        
        # Calculate tax liability
        total_income = income + ordinary_income + taxable_short_term
        short_term_tax = taxable_short_term * self.config.get_tax_rate(total_income, HoldingPeriod.SHORT_TERM, filing_status)
        long_term_tax = taxable_long_term * self.config.get_tax_rate(total_income, HoldingPeriod.LONG_TERM, filing_status)
        ordinary_income_tax = ordinary_income * self.config.get_tax_rate(total_income, HoldingPeriod.SHORT_TERM, filing_status)
        
        # Net Investment Income Tax (NIIT)
        niit_threshold = self.config.niit_threshold.get(filing_status, self.config.niit_threshold['single'])
        investment_income = taxable_short_term + taxable_long_term + ordinary_income
        niit_tax = Decimal('0')
        if total_income > niit_threshold:
            niit_tax = min(investment_income, total_income - niit_threshold) * self.config.niit_rate
        
        total_tax_liability = short_term_tax + long_term_tax + ordinary_income_tax + niit_tax
        
        return {
            'tax_year': tax_year,
            'filing_status': filing_status,
            'summary': {
                'short_term_gains': short_term_gains,
                'short_term_losses': short_term_losses,
                'net_short_term': net_short_term,
                'long_term_gains': long_term_gains,
                'long_term_losses': long_term_losses,
                'net_long_term': net_long_term,
                'ordinary_income': ordinary_income,
                'total_net_capital': total_net_capital,
                'capital_loss_deduction': capital_loss_deduction,
                'capital_loss_carryforward': capital_loss_carryforward
            },
            'tax_calculation': {
                'taxable_short_term': taxable_short_term,
                'taxable_long_term': taxable_long_term,
                'short_term_tax': short_term_tax,
                'long_term_tax': long_term_tax,
                'ordinary_income_tax': ordinary_income_tax,
                'niit_tax': niit_tax,
                'total_tax_liability': total_tax_liability
            },
            'events_count': len(year_events),
            'wash_sale_adjustments': len([e for e in year_events if e.wash_sale_affected])
        }
    
    def get_tax_loss_harvesting_opportunities(self, current_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Identify tax-loss harvesting opportunities."""
        if current_date is None:
            current_date = datetime.now()
        
        opportunities = []
        
        for asset, lots in self.tax_lots.items():
            if not lots:
                continue
            
            # Get current market price (placeholder - would integrate with price feed)
            current_price = self._get_current_price(asset)
            if current_price is None:
                continue
            
            total_quantity = sum(lot.quantity for lot in lots)
            total_cost_basis = sum(lot.adjusted_total_cost_basis for lot in lots)
            current_value = total_quantity * current_price
            unrealized_gain_loss = current_value - total_cost_basis
            
            if unrealized_gain_loss < 0:  # Unrealized loss
                # Check if wash sale rule would apply
                wash_sale_risk = self._assess_wash_sale_risk(asset, current_date)
                
                # Calculate potential tax benefit
                avg_cost_basis = total_cost_basis / total_quantity
                potential_tax_savings = self._calculate_tax_savings(abs(unrealized_gain_loss))
                
                opportunity = {
                    'asset': asset,
                    'total_quantity': total_quantity,
                    'avg_cost_basis': avg_cost_basis,
                    'current_price': current_price,
                    'unrealized_loss': abs(unrealized_gain_loss),
                    'potential_tax_savings': potential_tax_savings,
                    'wash_sale_risk': wash_sale_risk,
                    'recommendation': 'HARVEST' if not wash_sale_risk['high_risk'] else 'WAIT',
                    'holding_periods': self._analyze_holding_periods(lots, current_date)
                }
                
                opportunities.append(opportunity)
        
        # Sort by potential tax savings
        opportunities.sort(key=lambda x: x['potential_tax_savings'], reverse=True)
        
        return opportunities
    
    def _get_current_price(self, asset: str) -> Optional[Decimal]:
        """Get current market price for an asset. Placeholder for price feed integration."""
        # This would integrate with a real-time price feed
        # For now, return None to indicate price not available
        return None
    
    def _assess_wash_sale_risk(self, asset: str, current_date: datetime) -> Dict[str, Any]:
        """Assess wash sale risk for a potential sale."""
        lookback_start = current_date - timedelta(days=30)
        lookforward_end = current_date + timedelta(days=30)
        
        recent_purchases = []
        planned_purchases = []  # Would come from trading strategy
        
        for transaction in self.transactions:
            if (transaction.asset == asset and 
                transaction.transaction_type == TransactionType.BUY and
                lookback_start <= transaction.timestamp <= current_date):
                recent_purchases.append(transaction)
        
        high_risk = len(recent_purchases) > 0  # Simplified risk assessment
        
        return {
            'high_risk': high_risk,
            'recent_purchases': len(recent_purchases),
            'risk_factors': [
                'Recent purchases detected' if recent_purchases else None
            ]
        }
    
    def _calculate_tax_savings(self, loss_amount: Decimal) -> Decimal:
        """Calculate potential tax savings from realizing a loss."""
        # Simplified calculation using highest marginal rate
        marginal_rate = Decimal('0.37')  # Highest federal rate
        return loss_amount * marginal_rate
    
    def _analyze_holding_periods(self, lots: deque, current_date: datetime) -> Dict[str, Any]:
        """Analyze holding periods for tax lots."""
        short_term_quantity = Decimal('0')
        long_term_quantity = Decimal('0')
        
        for lot in lots:
            holding_days = (current_date - lot.acquisition_date).days
            if holding_days <= 365:
                short_term_quantity += lot.quantity
            else:
                long_term_quantity += lot.quantity
        
        return {
            'short_term_quantity': short_term_quantity,
            'long_term_quantity': long_term_quantity,
            'total_quantity': short_term_quantity + long_term_quantity
        }
    
    def _get_available_quantity(self, asset: str) -> Decimal:
        """Get available quantity for an asset."""
        return sum(lot.quantity for lot in self.tax_lots.get(asset, []))
    
    def _update_performance_metrics(self, transaction: Transaction):
        """Update performance tracking metrics."""
        asset = transaction.asset
        
        if transaction.transaction_type in [TransactionType.BUY, TransactionType.TRANSFER_IN]:
            self.total_cost_basis[asset] += transaction.total_value
        elif transaction.transaction_type == TransactionType.SELL:
            # Update realized gains when tax events are processed
            pass
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary with tax implications."""
        summary = {
            'holdings': {},
            'total_cost_basis': Decimal('0'),
            'total_unrealized_gain_loss': Decimal('0'),
            'total_realized_gain_loss': Decimal('0'),
            'wash_sale_adjustments_total': Decimal('0')
        }
        
        for asset, lots in self.tax_lots.items():
            if not lots:
                continue
            
            total_quantity = sum(lot.quantity for lot in lots)
            total_cost_basis = sum(lot.adjusted_total_cost_basis for lot in lots)
            wash_sale_adjustments = sum(lot.wash_sale_adjustment * lot.quantity for lot in lots)
            
            summary['holdings'][asset] = {
                'quantity': total_quantity,
                'cost_basis': total_cost_basis,
                'avg_cost_basis': total_cost_basis / total_quantity if total_quantity > 0 else Decimal('0'),
                'wash_sale_adjustments': wash_sale_adjustments,
                'lots_count': len(lots)
            }
            
            summary['total_cost_basis'] += total_cost_basis
            summary['wash_sale_adjustments_total'] += wash_sale_adjustments
        
        # Calculate realized gains/losses
        for event in self.tax_events:
            if event.event_type in [TaxEventType.CAPITAL_GAIN, TaxEventType.CAPITAL_LOSS]:
                summary['total_realized_gain_loss'] += event.gain_loss
        
        return summary
    
    # Database operations
    def _save_transaction_to_db(self, transaction: Transaction):
        """Save transaction to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    transaction.id,
                    transaction.timestamp.isoformat(),
                    transaction.transaction_type.value,
                    transaction.asset,
                    str(transaction.quantity),
                    str(transaction.price_per_unit),
                    str(transaction.total_value),
                    str(transaction.fees),
                    transaction.exchange,
                    transaction.wallet_address,
                    transaction.transaction_hash,
                    transaction.notes,
                    datetime.now().isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to save transaction to database: {e}")
    
    def _save_tax_lot_to_db(self, tax_lot: TaxLot):
        """Save tax lot to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO tax_lots 
                    (acquisition_date, asset, quantity, cost_basis_per_unit, total_cost_basis, 
                     transaction_id, is_wash_sale_affected, wash_sale_adjustment) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tax_lot.acquisition_date.isoformat(),
                    tax_lot.asset,
                    str(tax_lot.quantity),
                    str(tax_lot.cost_basis_per_unit),
                    str(tax_lot.total_cost_basis),
                    tax_lot.transaction_id,
                    tax_lot.is_wash_sale_affected,
                    str(tax_lot.wash_sale_adjustment)
                ))
        except Exception as e:
            logger.error(f"Failed to save tax lot to database: {e}")
    
    def _save_tax_event_to_db(self, tax_event: TaxEvent):
        """Save tax event to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO tax_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tax_event.id,
                    tax_event.timestamp.isoformat(),
                    tax_event.event_type.value,
                    tax_event.asset,
                    str(tax_event.quantity),
                    str(tax_event.proceeds),
                    str(tax_event.cost_basis),
                    str(tax_event.gain_loss),
                    tax_event.holding_period.value,
                    tax_event.transaction_id,
                    tax_event.wash_sale_affected,
                    str(tax_event.wash_sale_disallowed),
                    str(tax_event.tax_rate) if tax_event.tax_rate else None,
                    datetime.now().isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to save tax event to database: {e}")
    
    def _save_wash_sale_adjustment(self, adjustment: Dict):
        """Save wash sale adjustment to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO wash_sale_adjustments 
                    (sale_transaction_id, purchase_transaction_id, asset, disallowed_loss, adjustment_date)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    adjustment['sale_transaction_id'],
                    adjustment['purchase_transaction_id'],
                    adjustment['asset'],
                    str(adjustment['disallowed_loss']),
                    adjustment['adjustment_date'].isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to save wash sale adjustment to database: {e}")
    
    def _update_tax_lot_wash_sale(self, tax_lot: TaxLot):
        """Update tax lot wash sale information in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE tax_lots 
                    SET is_wash_sale_affected = ?, wash_sale_adjustment = ?
                    WHERE transaction_id = ? AND asset = ?
                """, (
                    tax_lot.is_wash_sale_affected,
                    str(tax_lot.wash_sale_adjustment),
                    tax_lot.transaction_id,
                    tax_lot.asset
                ))
        except Exception as e:
            logger.error(f"Failed to update tax lot wash sale information: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Example usage of the tax engine
    config = TaxConfiguration("US")
    engine = TaxEngine(config)
    
    # Example transactions
    transactions = [
        Transaction(
            id="buy_1",
            timestamp=datetime(2024, 1, 15),
            transaction_type=TransactionType.BUY,
            asset="BTC",
            quantity=Decimal('1.0'),
            price_per_unit=Decimal('45000'),
            total_value=Decimal('45000'),
            fees=Decimal('50')
        ),
        Transaction(
            id="sell_1",
            timestamp=datetime(2024, 6, 15),
            transaction_type=TransactionType.SELL,
            asset="BTC",
            quantity=Decimal('0.5'),
            price_per_unit=Decimal('60000'),
            total_value=Decimal('30000'),
            fees=Decimal('30')
        )
    ]
    
    for transaction in transactions:
        engine.add_transaction(transaction)
    
    # Calculate tax liability
    tax_liability = engine.calculate_tax_liability(2024)
    print(f"Tax liability for 2024: {tax_liability}")
    
    # Get portfolio summary
    portfolio = engine.get_portfolio_summary()
    print(f"Portfolio summary: {portfolio}")