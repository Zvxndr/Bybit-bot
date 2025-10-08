"""
Trade Reconciliation System - Data Integrity Engine
==================================================

This module implements comprehensive trade reconciliation to ensure
data consistency between our system and exchange records.

Status: HIGH PRIORITY - Critical for production reliability
"""

import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Import Bybit client
try:
    from ..bybit_api import BybitAPIClient
except ImportError:
    # Fallback import
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from bybit_api import BybitAPIClient


class ReconciliationStatus(Enum):
    """Reconciliation status"""
    PENDING = "pending"
    MATCHED = "matched"
    DISCREPANCY = "discrepancy"
    MISSING_LOCAL = "missing_local"
    MISSING_EXCHANGE = "missing_exchange"
    ERROR = "error"


@dataclass
class TradeRecord:
    """Trade record for reconciliation"""
    trade_id: str
    order_id: str
    exchange_order_id: Optional[str]
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    fee: Decimal
    fee_currency: str
    source: str  # 'local' or 'exchange'


@dataclass
class ReconciliationResult:
    """Result of a reconciliation check"""
    trade_id: str
    status: ReconciliationStatus
    local_record: Optional[TradeRecord]
    exchange_record: Optional[TradeRecord]
    discrepancies: List[str]
    resolution: Optional[str] = None


@dataclass
class PositionSnapshot:
    """Position snapshot for reconciliation"""
    symbol: str
    quantity: Decimal
    average_price: Decimal
    unrealized_pnl: Decimal
    timestamp: datetime
    source: str  # 'local' or 'exchange'


class TradeReconciler:
    """
    Trade reconciliation engine for data integrity.
    
    This addresses the CRITICAL gap identified in the project analysis:
    - Ensures data consistency between local and exchange records
    - Detects and reports discrepancies
    - Maintains audit trail
    - Provides position reconciliation
    """
    
    def __init__(self, 
                 bybit_client: BybitAPIClient,
                 testnet_client: BybitAPIClient,
                 db_path: str = "data/trading_bot.db"):
        
        self.logger = logging.getLogger(__name__)
        
        # API clients
        self.bybit_client = bybit_client
        self.testnet_client = testnet_client
        
        # Database
        self.db_path = db_path
        
        # Reconciliation tracking
        self.reconciliation_results: Dict[str, ReconciliationResult] = {}
        self.last_reconciliation: Optional[datetime] = None
        
        # Discrepancy tracking
        self.unresolved_discrepancies: List[ReconciliationResult] = []
        
        # Reconciliation settings
        self.auto_reconcile_interval = 300  # 5 minutes
        self.max_discrepancy_threshold = Decimal('0.01')  # 1 cent tolerance
        
        # Background task
        self.reconciliation_task: Optional[asyncio.Task] = None
        
        self._initialize_database()
        self.logger.info("✅ Trade Reconciler initialized")
    
    def _initialize_database(self):
        """Initialize reconciliation database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create reconciliation results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reconciliation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    reconciliation_time TIMESTAMP NOT NULL,
                    status TEXT NOT NULL,
                    discrepancies TEXT,
                    resolution TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create position snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity DECIMAL NOT NULL,
                    average_price DECIMAL NOT NULL,
                    unrealized_pnl DECIMAL NOT NULL,
                    snapshot_time TIMESTAMP NOT NULL,
                    source TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create discrepancy log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS discrepancy_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    discrepancy_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Reconciliation database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization error: {e}")
    
    async def start_auto_reconciliation(self):
        """Start automatic reconciliation process"""
        if self.reconciliation_task and not self.reconciliation_task.done():
            self.logger.warning("⚠️ Auto reconciliation already running")
            return
        
        self.reconciliation_task = asyncio.create_task(self._auto_reconciliation_loop())
        self.logger.info("✅ Auto reconciliation started")
    
    async def stop_auto_reconciliation(self):
        """Stop automatic reconciliation process"""
        if self.reconciliation_task and not self.reconciliation_task.done():
            self.reconciliation_task.cancel()
            try:
                await self.reconciliation_task
            except asyncio.CancelledError:
                pass
            
            self.logger.info("🔄 Auto reconciliation stopped")
    
    async def _auto_reconciliation_loop(self):
        """Automatic reconciliation loop"""
        self.logger.info("🔄 Starting auto reconciliation loop")
        
        while True:
            try:
                # Run reconciliation
                await self.reconcile_trades()
                await self.reconcile_positions()
                
                # Wait for next interval
                await asyncio.sleep(self.auto_reconcile_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Auto reconciliation error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
        
        self.logger.info("🔄 Auto reconciliation loop ended")
    
    async def reconcile_trades(self, 
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None,
                             use_testnet: bool = True) -> List[ReconciliationResult]:
        """
        Reconcile trades between local database and exchange
        
        Args:
            start_time: Start time for reconciliation period
            end_time: End time for reconciliation period  
            use_testnet: Use testnet or mainnet for reconciliation
            
        Returns:
            List of reconciliation results
        """
        try:
            # Set default time range (last 24 hours)
            if not end_time:
                end_time = datetime.now()
            if not start_time:
                start_time = end_time - timedelta(hours=24)
            
            self.logger.info(f"🔄 Starting trade reconciliation: {start_time} to {end_time}")
            
            # Get local trade records
            local_trades = await self._get_local_trades(start_time, end_time)
            
            # Get exchange trade records
            exchange_trades = await self._get_exchange_trades(start_time, end_time, use_testnet)
            
            # Perform reconciliation
            results = await self._reconcile_trade_records(local_trades, exchange_trades)
            
            # Store results
            await self._store_reconciliation_results(results)
            
            # Update tracking
            self.last_reconciliation = datetime.now()
            
            # Report summary
            matched = sum(1 for r in results if r.status == ReconciliationStatus.MATCHED)
            discrepancies = sum(1 for r in results if r.status == ReconciliationStatus.DISCREPANCY)
            
            self.logger.info(f"✅ Reconciliation complete: {matched} matched, {discrepancies} discrepancies")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Trade reconciliation error: {e}")
            return []
    
    async def _get_local_trades(self, start_time: datetime, end_time: datetime) -> List[TradeRecord]:
        """Get local trade records from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query trades from orders and fills tables (if they exist)
            cursor.execute('''
                SELECT 
                    o.id as trade_id,
                    o.id as order_id,
                    o.exchange_order_id,
                    o.symbol,
                    o.side,
                    o.filled_quantity as quantity,
                    o.average_fill_price as price,
                    o.updated_at as timestamp,
                    o.total_fees as fee,
                    'USDT' as fee_currency
                FROM orders o 
                WHERE o.status = 'filled'
                AND o.updated_at BETWEEN ? AND ?
            ''', (start_time, end_time))
            
            rows = cursor.fetchall()
            conn.close()
            
            trades = []
            for row in rows:
                trades.append(TradeRecord(
                    trade_id=row[0],
                    order_id=row[1],
                    exchange_order_id=row[2],
                    symbol=row[3],
                    side=row[4],
                    quantity=Decimal(str(row[5])) if row[5] else Decimal('0'),
                    price=Decimal(str(row[6])) if row[6] else Decimal('0'),
                    timestamp=datetime.fromisoformat(row[7]) if row[7] else datetime.now(),
                    fee=Decimal(str(row[8])) if row[8] else Decimal('0'),
                    fee_currency=row[9],
                    source='local'
                ))
            
            self.logger.info(f"📊 Found {len(trades)} local trades")
            return trades
            
        except Exception as e:
            self.logger.error(f"❌ Error getting local trades: {e}")
            return []
    
    async def _get_exchange_trades(self, 
                                 start_time: datetime, 
                                 end_time: datetime,
                                 use_testnet: bool) -> List[TradeRecord]:
        """Get trade records from exchange"""
        try:
            client = self.testnet_client if use_testnet else self.bybit_client
            
            # Get trade history from exchange
            response = await client.get_trade_history(
                start_time=start_time,
                end_time=end_time
            )
            
            trades = []
            if response and response.get('success'):
                exchange_data = response.get('data', [])
                
                for trade_data in exchange_data:
                    trades.append(TradeRecord(
                        trade_id=trade_data.get('execId', ''),
                        order_id=trade_data.get('orderId', ''),
                        exchange_order_id=trade_data.get('orderId', ''),
                        symbol=trade_data.get('symbol', ''),
                        side=trade_data.get('side', '').lower(),
                        quantity=Decimal(str(trade_data.get('execQty', 0))),
                        price=Decimal(str(trade_data.get('execPrice', 0))),
                        timestamp=datetime.fromtimestamp(trade_data.get('execTime', 0) / 1000),
                        fee=Decimal(str(trade_data.get('execFee', 0))),
                        fee_currency=trade_data.get('feeRate', 'USDT'),
                        source='exchange'
                    ))
            
            self.logger.info(f"📊 Found {len(trades)} exchange trades")
            return trades
            
        except Exception as e:
            self.logger.error(f"❌ Error getting exchange trades: {e}")
            return []
    
    async def _reconcile_trade_records(self, 
                                     local_trades: List[TradeRecord],
                                     exchange_trades: List[TradeRecord]) -> List[ReconciliationResult]:
        """Reconcile local and exchange trade records"""
        results = []
        
        # Create lookup maps
        local_map = {trade.exchange_order_id: trade for trade in local_trades if trade.exchange_order_id}
        exchange_map = {trade.order_id: trade for trade in exchange_trades}
        
        # Check local trades against exchange
        for local_trade in local_trades:
            if not local_trade.exchange_order_id:
                continue
                
            exchange_trade = exchange_map.get(local_trade.exchange_order_id)
            
            if exchange_trade:
                # Compare records
                discrepancies = self._compare_trade_records(local_trade, exchange_trade)
                
                status = ReconciliationStatus.MATCHED if not discrepancies else ReconciliationStatus.DISCREPANCY
                
                result = ReconciliationResult(
                    trade_id=local_trade.trade_id,
                    status=status,
                    local_record=local_trade,
                    exchange_record=exchange_trade,
                    discrepancies=discrepancies
                )
                
                results.append(result)
                
                # Remove from exchange map (processed)
                exchange_map.pop(local_trade.exchange_order_id, None)
            else:
                # Missing from exchange
                result = ReconciliationResult(
                    trade_id=local_trade.trade_id,
                    status=ReconciliationStatus.MISSING_EXCHANGE,
                    local_record=local_trade,
                    exchange_record=None,
                    discrepancies=["Trade missing from exchange"]
                )
                results.append(result)
        
        # Check remaining exchange trades (missing locally)
        for exchange_trade in exchange_map.values():
            result = ReconciliationResult(
                trade_id=exchange_trade.trade_id,
                status=ReconciliationStatus.MISSING_LOCAL,
                local_record=None,
                exchange_record=exchange_trade,
                discrepancies=["Trade missing from local database"]
            )
            results.append(result)
        
        return results
    
    def _compare_trade_records(self, local_trade: TradeRecord, exchange_trade: TradeRecord) -> List[str]:
        """Compare local and exchange trade records for discrepancies"""
        discrepancies = []
        
        # Compare quantities
        qty_diff = abs(local_trade.quantity - exchange_trade.quantity)
        if qty_diff > self.max_discrepancy_threshold:
            discrepancies.append(f"Quantity mismatch: local={local_trade.quantity}, exchange={exchange_trade.quantity}")
        
        # Compare prices
        price_diff = abs(local_trade.price - exchange_trade.price)
        if price_diff > self.max_discrepancy_threshold:
            discrepancies.append(f"Price mismatch: local={local_trade.price}, exchange={exchange_trade.price}")
        
        # Compare sides
        if local_trade.side.lower() != exchange_trade.side.lower():
            discrepancies.append(f"Side mismatch: local={local_trade.side}, exchange={exchange_trade.side}")
        
        # Compare symbols
        if local_trade.symbol != exchange_trade.symbol:
            discrepancies.append(f"Symbol mismatch: local={local_trade.symbol}, exchange={exchange_trade.symbol}")
        
        # Compare fees (with tolerance)
        fee_diff = abs(local_trade.fee - exchange_trade.fee)
        if fee_diff > self.max_discrepancy_threshold:
            discrepancies.append(f"Fee mismatch: local={local_trade.fee}, exchange={exchange_trade.fee}")
        
        return discrepancies
    
    async def reconcile_positions(self, use_testnet: bool = True) -> Dict[str, Any]:
        """Reconcile positions between local and exchange"""
        try:
            self.logger.info("🔄 Starting position reconciliation")
            
            # Get local positions
            local_positions = await self._get_local_positions()
            
            # Get exchange positions
            exchange_positions = await self._get_exchange_positions(use_testnet)
            
            # Compare positions
            reconciliation_summary = {
                'reconciliation_time': datetime.now().isoformat(),
                'matched_positions': [],
                'discrepancies': [],
                'missing_local': [],
                'missing_exchange': []
            }
            
            # Create lookup maps
            local_map = {pos.symbol: pos for pos in local_positions}
            exchange_map = {pos.symbol: pos for pos in exchange_positions}
            
            # Check all symbols
            all_symbols = set(local_map.keys()) | set(exchange_map.keys())
            
            for symbol in all_symbols:
                local_pos = local_map.get(symbol)
                exchange_pos = exchange_map.get(symbol)
                
                if local_pos and exchange_pos:
                    # Compare positions
                    qty_diff = abs(local_pos.quantity - exchange_pos.quantity)
                    
                    if qty_diff > self.max_discrepancy_threshold:
                        reconciliation_summary['discrepancies'].append({
                            'symbol': symbol,
                            'local_quantity': float(local_pos.quantity),
                            'exchange_quantity': float(exchange_pos.quantity),
                            'difference': float(qty_diff)
                        })
                    else:
                        reconciliation_summary['matched_positions'].append(symbol)
                        
                elif local_pos:
                    reconciliation_summary['missing_exchange'].append({
                        'symbol': symbol,
                        'local_quantity': float(local_pos.quantity)
                    })
                    
                elif exchange_pos:
                    reconciliation_summary['missing_local'].append({
                        'symbol': symbol,
                        'exchange_quantity': float(exchange_pos.quantity)
                    })
            
            # Store position snapshots
            await self._store_position_snapshots(local_positions + exchange_positions)
            
            self.logger.info(f"✅ Position reconciliation complete: {len(reconciliation_summary['matched_positions'])} matched")
            
            return reconciliation_summary
            
        except Exception as e:
            self.logger.error(f"❌ Position reconciliation error: {e}")
            return {'error': str(e)}
    
    async def _get_local_positions(self) -> List[PositionSnapshot]:
        """Get local position snapshots"""
        try:
            # This would query the local position tracking system
            # For now, return empty list until position tracking is implemented
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Error getting local positions: {e}")
            return []
    
    async def _get_exchange_positions(self, use_testnet: bool) -> List[PositionSnapshot]:
        """Get exchange position snapshots"""
        try:
            client = self.testnet_client if use_testnet else self.bybit_client
            
            response = await client.get_positions()
            
            positions = []
            if response and response.get('success'):
                position_data = response.get('data', [])
                
                for pos_data in position_data:
                    if float(pos_data.get('size', 0)) != 0:  # Only non-zero positions
                        positions.append(PositionSnapshot(
                            symbol=pos_data.get('symbol', ''),
                            quantity=Decimal(str(pos_data.get('size', 0))),
                            average_price=Decimal(str(pos_data.get('avgPrice', 0))),
                            unrealized_pnl=Decimal(str(pos_data.get('unrealisedPnl', 0))),
                            timestamp=datetime.now(),
                            source='exchange'
                        ))
            
            return positions
            
        except Exception as e:
            self.logger.error(f"❌ Error getting exchange positions: {e}")
            return []
    
    async def _store_reconciliation_results(self, results: List[ReconciliationResult]):
        """Store reconciliation results in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for result in results:
                cursor.execute('''
                    INSERT INTO reconciliation_results 
                    (trade_id, reconciliation_time, status, discrepancies, resolution)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    result.trade_id,
                    datetime.now(),
                    result.status.value,
                    '; '.join(result.discrepancies),
                    result.resolution
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error storing reconciliation results: {e}")
    
    async def _store_position_snapshots(self, positions: List[PositionSnapshot]):
        """Store position snapshots in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for position in positions:
                cursor.execute('''
                    INSERT INTO position_snapshots
                    (symbol, quantity, average_price, unrealized_pnl, snapshot_time, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    position.symbol,
                    position.quantity,
                    position.average_price,
                    position.unrealized_pnl,
                    position.timestamp,
                    position.source
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error storing position snapshots: {e}")
    
    def get_reconciliation_summary(self) -> Dict[str, Any]:
        """Get reconciliation summary and statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent reconciliation stats
            cursor.execute('''
                SELECT 
                    status,
                    COUNT(*) as count
                FROM reconciliation_results 
                WHERE reconciliation_time > datetime('now', '-24 hours')
                GROUP BY status
            ''')
            
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get unresolved discrepancies
            cursor.execute('''
                SELECT COUNT(*) 
                FROM reconciliation_results 
                WHERE status = 'discrepancy'
                AND resolution IS NULL
            ''')
            
            unresolved_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'last_reconciliation': self.last_reconciliation.isoformat() if self.last_reconciliation else None,
                'status_counts': status_counts,
                'unresolved_discrepancies': unresolved_count,
                'auto_reconciliation_active': self.reconciliation_task is not None and not self.reconciliation_task.done(),
                'reconciliation_interval_seconds': self.auto_reconcile_interval
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting reconciliation summary: {e}")
            return {'error': str(e)}


# Factory function for easy integration
def create_trade_reconciler(bybit_client: BybitAPIClient,
                          testnet_client: BybitAPIClient,
                          db_path: str = "data/trading_bot.db") -> TradeReconciler:
    """Factory function to create trade reconciler"""
    return TradeReconciler(
        bybit_client=bybit_client,
        testnet_client=testnet_client,
        db_path=db_path
    )