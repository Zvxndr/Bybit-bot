"""
Data Infrastructure for ML Strategy Discovery
Multi-exchange data collection with AUD-focused features
Transfer cost database and Australian market integration
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import logging
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ExchangeName(Enum):
    """Supported exchanges"""
    BYBIT = "bybit"
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BTCMARKETS = "btcmarkets"  # Australian exchange
    COINJAR = "coinjar"       # Australian exchange
    SWYFTX = "swyftx"        # Australian exchange

class DataType(Enum):
    """Types of market data"""
    OHLCV = "ohlcv"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    FUNDING_RATES = "funding_rates"
    OPEN_INTEREST = "open_interest"

@dataclass
class TransferCost:
    """Transfer cost between exchanges"""
    from_exchange: str
    to_exchange: str
    currency: str
    fixed_fee: Decimal
    percentage_fee: Decimal
    min_amount: Decimal
    max_amount: Optional[Decimal]
    estimated_time_minutes: int
    requires_verification: bool
    australian_friendly: bool = False

@dataclass
class ExchangeInfo:
    """Exchange information and capabilities"""
    name: str
    display_name: str
    australian_regulated: bool
    supports_aud: bool
    api_url: str
    rate_limit_per_second: int
    withdrawal_fees: Dict[str, Decimal]
    deposit_fees: Dict[str, Decimal]
    trading_fees: Dict[str, Decimal]
    verification_levels: List[str]

@dataclass
class MarketData:
    """Market data container"""
    symbol: str
    exchange: str
    data_type: DataType
    timestamp: datetime
    data: Dict[str, Any]
    aud_equivalent: Optional[Decimal] = None

class AustralianDataProvider:
    """
    Australian-specific data provider
    RBA, ASX, and macro data relevant to crypto trading
    """
    
    def __init__(self):
        self.rba_api_url = "https://www.rba.gov.au/statistics"
        self.asx_api_url = "https://www.asx.com.au"
        self.cached_data = {}
        self.cache_ttl = 3600  # 1 hour cache
    
    async def get_rba_cash_rate(self) -> Optional[Decimal]:
        """Get current RBA cash rate"""
        
        cache_key = "rba_cash_rate"
        if self._is_cached_valid(cache_key):
            return self.cached_data[cache_key]['data']
        
        try:
            # In practice, would scrape RBA website or use API
            # For now, using placeholder logic
            cash_rate = Decimal('4.35')  # Current rate as of 2024
            
            self.cached_data[cache_key] = {
                'data': cash_rate,
                'timestamp': datetime.now()
            }
            
            return cash_rate
            
        except Exception as e:
            logger.error(f"Error fetching RBA cash rate: {e}")
            return None
    
    async def get_aud_usd_rate(self) -> Optional[Decimal]:
        """Get current AUD/USD exchange rate"""
        
        cache_key = "aud_usd_rate"
        if self._is_cached_valid(cache_key):
            return self.cached_data[cache_key]['data']
        
        try:
            # Use forex API (e.g., exchangerate-api.com)
            async with aiohttp.ClientSession() as session:
                url = "https://api.exchangerate-api.com/v4/latest/AUD"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        usd_rate = Decimal(str(data['rates']['USD']))
                        
                        self.cached_data[cache_key] = {
                            'data': usd_rate,
                            'timestamp': datetime.now()
                        }
                        
                        return usd_rate
                        
        except Exception as e:
            logger.error(f"Error fetching AUD/USD rate: {e}")
            return None
    
    async def get_asx_200_data(self) -> Optional[Dict]:
        """Get ASX 200 index data"""
        
        cache_key = "asx_200"
        if self._is_cached_valid(cache_key):
            return self.cached_data[cache_key]['data']
        
        try:
            # In practice, would use ASX API or financial data provider
            # Placeholder data
            asx_data = {
                'price': Decimal('7850.5'),
                'change': Decimal('-12.3'),
                'change_percent': Decimal('-0.16')
            }
            
            self.cached_data[cache_key] = {
                'data': asx_data,
                'timestamp': datetime.now()
            }
            
            return asx_data
            
        except Exception as e:
            logger.error(f"Error fetching ASX 200 data: {e}")
            return None
    
    async def get_gold_price_aud(self) -> Optional[Decimal]:
        """Get gold price in AUD"""
        
        cache_key = "gold_price_aud"
        if self._is_cached_valid(cache_key):
            return self.cached_data[cache_key]['data']
        
        try:
            # Use precious metals API
            async with aiohttp.ClientSession() as session:
                url = "https://api.metals.live/v1/spot/gold"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        usd_price = Decimal(str(data[0]['price']))
                        
                        # Convert to AUD
                        aud_usd_rate = await self.get_aud_usd_rate()
                        if aud_usd_rate:
                            aud_price = usd_price / aud_usd_rate
                            
                            self.cached_data[cache_key] = {
                                'data': aud_price,
                                'timestamp': datetime.now()
                            }
                            
                            return aud_price
                            
        except Exception as e:
            logger.error(f"Error fetching gold price: {e}")
            return None
    
    async def get_australian_market_hours(self) -> Dict[str, bool]:
        """Get Australian market session information"""
        
        now = datetime.now()
        sydney_time = now.astimezone(tz=None)  # Assume system timezone is Sydney
        hour = sydney_time.hour
        weekday = sydney_time.weekday()
        
        return {
            'asx_open': weekday < 5 and 10 <= hour < 16,  # 10 AM - 4 PM AEST
            'sydney_session': 8 <= hour < 17,  # Broader Sydney trading session
            'is_business_day': weekday < 5,
            'is_weekend': weekday >= 5
        }
    
    def _is_cached_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cached_data:
            return False
        
        timestamp = self.cached_data[key]['timestamp']
        return (datetime.now() - timestamp).seconds < self.cache_ttl

class TransferCostDatabase:
    """
    Database for tracking transfer costs between exchanges
    Critical for Australian arbitrage where bank transfer costs matter
    """
    
    def __init__(self, db_path: str = "data/transfer_costs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        self._populate_australian_costs()
    
    def _initialize_database(self):
        """Initialize the transfer cost database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transfer_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_exchange TEXT NOT NULL,
                to_exchange TEXT NOT NULL,
                currency TEXT NOT NULL,
                fixed_fee DECIMAL(20, 8),
                percentage_fee DECIMAL(10, 6),
                min_amount DECIMAL(20, 8),
                max_amount DECIMAL(20, 8),
                estimated_time_minutes INTEGER,
                requires_verification BOOLEAN,
                australian_friendly BOOLEAN,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(from_exchange, to_exchange, currency)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exchange_info (
                name TEXT PRIMARY KEY,
                display_name TEXT,
                australian_regulated BOOLEAN,
                supports_aud BOOLEAN,
                api_url TEXT,
                rate_limit_per_second INTEGER,
                withdrawal_fees TEXT,  -- JSON
                deposit_fees TEXT,     -- JSON
                trading_fees TEXT,     -- JSON
                verification_levels TEXT,  -- JSON
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _populate_australian_costs(self):
        """Populate database with Australian-specific transfer costs"""
        
        # Australian exchanges
        australian_costs = [
            # BTC Markets (Australian)
            TransferCost(
                from_exchange="bank_aud",
                to_exchange="btcmarkets",
                currency="AUD",
                fixed_fee=Decimal('0'),
                percentage_fee=Decimal('0'),
                min_amount=Decimal('1'),
                max_amount=Decimal('50000'),
                estimated_time_minutes=60,
                requires_verification=True,
                australian_friendly=True
            ),
            
            # CoinJar (Australian)
            TransferCost(
                from_exchange="bank_aud",
                to_exchange="coinjar",
                currency="AUD",
                fixed_fee=Decimal('0'),
                percentage_fee=Decimal('0'),
                min_amount=Decimal('1'),
                max_amount=Decimal('100000'),
                estimated_time_minutes=30,
                requires_verification=True,
                australian_friendly=True
            ),
            
            # Swyftx (Australian)
            TransferCost(
                from_exchange="bank_aud",
                to_exchange="swyftx",
                currency="AUD",
                fixed_fee=Decimal('0'),
                percentage_fee=Decimal('0'),
                min_amount=Decimal('1'),
                max_amount=Decimal('20000'),
                estimated_time_minutes=15,
                requires_verification=True,
                australian_friendly=True
            ),
            
            # International to Australian
            TransferCost(
                from_exchange="bybit",
                to_exchange="btcmarkets",
                currency="USDT",
                fixed_fee=Decimal('1'),
                percentage_fee=Decimal('0'),
                min_amount=Decimal('10'),
                max_amount=Decimal('100000'),
                estimated_time_minutes=10,
                requires_verification=False,
                australian_friendly=True
            ),
            
            # Crypto transfers between major exchanges
            TransferCost(
                from_exchange="binance",
                to_exchange="bybit",
                currency="BTC",
                fixed_fee=Decimal('0.0005'),
                percentage_fee=Decimal('0'),
                min_amount=Decimal('0.001'),
                max_amount=None,
                estimated_time_minutes=30,
                requires_verification=False,
                australian_friendly=False
            )
        ]
        
        for cost in australian_costs:
            self.add_transfer_cost(cost)
        
        # Australian exchange information
        australian_exchanges = [
            ExchangeInfo(
                name="btcmarkets",
                display_name="BTC Markets",
                australian_regulated=True,
                supports_aud=True,
                api_url="https://api.btcmarkets.net",
                rate_limit_per_second=10,
                withdrawal_fees={"BTC": Decimal('0.0005'), "ETH": Decimal('0.005'), "AUD": Decimal('0')},
                deposit_fees={"AUD": Decimal('0')},
                trading_fees={"maker": Decimal('0.0085'), "taker": Decimal('0.0085')},
                verification_levels=["Basic", "Intermediate", "Pro"]
            ),
            
            ExchangeInfo(
                name="coinjar",
                display_name="CoinJar",
                australian_regulated=True,
                supports_aud=True,
                api_url="https://api.coinjar.com",
                rate_limit_per_second=15,
                withdrawal_fees={"BTC": Decimal('0.001'), "ETH": Decimal('0.01'), "AUD": Decimal('0')},
                deposit_fees={"AUD": Decimal('0')},
                trading_fees={"maker": Decimal('0.001'), "taker": Decimal('0.001')},
                verification_levels=["Basic", "Pro"]
            ),
            
            ExchangeInfo(
                name="swyftx",
                display_name="Swyftx",
                australian_regulated=True,
                supports_aud=True,
                api_url="https://api.swyftx.com.au",
                rate_limit_per_second=20,
                withdrawal_fees={"BTC": Decimal('0.0001'), "ETH": Decimal('0.005'), "AUD": Decimal('0')},
                deposit_fees={"AUD": Decimal('0')},
                trading_fees={"maker": Decimal('0.006'), "taker": Decimal('0.006')},
                verification_levels=["Bronze", "Silver", "Gold"]
            )
        ]
        
        for exchange in australian_exchanges:
            self.add_exchange_info(exchange)
    
    def add_transfer_cost(self, cost: TransferCost):
        """Add or update transfer cost"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO transfer_costs 
            (from_exchange, to_exchange, currency, fixed_fee, percentage_fee, 
             min_amount, max_amount, estimated_time_minutes, requires_verification, 
             australian_friendly, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            cost.from_exchange, cost.to_exchange, cost.currency,
            float(cost.fixed_fee), float(cost.percentage_fee),
            float(cost.min_amount), 
            float(cost.max_amount) if cost.max_amount else None,
            cost.estimated_time_minutes, cost.requires_verification,
            cost.australian_friendly
        ))
        
        conn.commit()
        conn.close()
    
    def add_exchange_info(self, exchange: ExchangeInfo):
        """Add or update exchange information"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO exchange_info
            (name, display_name, australian_regulated, supports_aud, api_url,
             rate_limit_per_second, withdrawal_fees, deposit_fees, trading_fees,
             verification_levels, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            exchange.name, exchange.display_name, exchange.australian_regulated,
            exchange.supports_aud, exchange.api_url, exchange.rate_limit_per_second,
            json.dumps({k: float(v) for k, v in exchange.withdrawal_fees.items()}),
            json.dumps({k: float(v) for k, v in exchange.deposit_fees.items()}),
            json.dumps({k: float(v) for k, v in exchange.trading_fees.items()}),
            json.dumps(exchange.verification_levels)
        ))
        
        conn.commit()
        conn.close()
    
    def get_transfer_cost(
        self,
        from_exchange: str,
        to_exchange: str,
        currency: str
    ) -> Optional[TransferCost]:
        """Get transfer cost between exchanges"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM transfer_costs
            WHERE from_exchange = ? AND to_exchange = ? AND currency = ?
        ''', (from_exchange, to_exchange, currency))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return TransferCost(
                from_exchange=row[1],
                to_exchange=row[2],
                currency=row[3],
                fixed_fee=Decimal(str(row[4])),
                percentage_fee=Decimal(str(row[5])),
                min_amount=Decimal(str(row[6])),
                max_amount=Decimal(str(row[7])) if row[7] else None,
                estimated_time_minutes=row[8],
                requires_verification=bool(row[9]),
                australian_friendly=bool(row[10])
            )
        
        return None
    
    def get_australian_friendly_routes(self, currency: str) -> List[TransferCost]:
        """Get all Australian-friendly transfer routes for a currency"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM transfer_costs
            WHERE currency = ? AND australian_friendly = 1
            ORDER BY fixed_fee + (percentage_fee * 1000)  -- Rough cost ordering
        ''', (currency,))
        
        rows = cursor.fetchall()
        conn.close()
        
        costs = []
        for row in rows:
            costs.append(TransferCost(
                from_exchange=row[1],
                to_exchange=row[2],
                currency=row[3],
                fixed_fee=Decimal(str(row[4])),
                percentage_fee=Decimal(str(row[5])),
                min_amount=Decimal(str(row[6])),
                max_amount=Decimal(str(row[7])) if row[7] else None,
                estimated_time_minutes=row[8],
                requires_verification=bool(row[9]),
                australian_friendly=bool(row[10])
            ))
        
        return costs
    
    def calculate_transfer_cost(
        self,
        from_exchange: str,
        to_exchange: str,
        currency: str,
        amount: Decimal
    ) -> Optional[Decimal]:
        """Calculate total transfer cost for given amount"""
        
        cost_info = self.get_transfer_cost(from_exchange, to_exchange, currency)
        
        if not cost_info:
            return None
        
        if amount < cost_info.min_amount:
            return None  # Below minimum
        
        if cost_info.max_amount and amount > cost_info.max_amount:
            return None  # Above maximum
        
        total_cost = cost_info.fixed_fee + (amount * cost_info.percentage_fee)
        
        return total_cost

class MultiExchangeDataCollector:
    """
    Multi-exchange data collection optimized for Australian traders
    """
    
    def __init__(self):
        self.australian_data = AustralianDataProvider()
        self.transfer_db = TransferCostDatabase()
        self.active_connections = {}
        self.data_cache = {}
        
        # Exchange configurations
        self.exchanges = {
            ExchangeName.BYBIT: {
                "api_url": "https://api.bybit.com",
                "rate_limit": 10,
                "australian_priority": False
            },
            ExchangeName.BINANCE: {
                "api_url": "https://api.binance.com",
                "rate_limit": 20,  
                "australian_priority": False
            },
            ExchangeName.BTCMARKETS: {
                "api_url": "https://api.btcmarkets.net",
                "rate_limit": 10,
                "australian_priority": True
            },
            ExchangeName.COINJAR: {
                "api_url": "https://api.coinjar.com",
                "rate_limit": 15,
                "australian_priority": True  
            },
            ExchangeName.SWYFTX: {
                "api_url": "https://api.swyftx.com.au",
                "rate_limit": 20,
                "australian_priority": True
            }
        }
        
        logger.info(f"Initialized data collector for {len(self.exchanges)} exchanges")
    
    async def collect_ohlcv_data(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """Collect OHLCV data from multiple exchanges"""
        
        all_data = {}
        
        # Prioritize Australian exchanges for AUD pairs
        sorted_exchanges = sorted(
            self.exchanges.items(),
            key=lambda x: (x[1]["australian_priority"], x[1]["rate_limit"]),
            reverse=True
        )
        
        for exchange_name, config in sorted_exchanges:
            exchange_data = {}
            
            for symbol in symbols:
                try:
                    data = await self._fetch_ohlcv(
                        exchange_name.value,
                        symbol,
                        timeframe,
                        limit
                    )
                    
                    if data is not None and len(data) > 0:
                        # Add exchange identifier
                        data['exchange'] = exchange_name.value
                        exchange_data[symbol] = data
                        
                        logger.debug(f"Collected {len(data)} {timeframe} candles for {symbol} from {exchange_name.value}")
                        
                except Exception as e:
                    logger.error(f"Error collecting {symbol} from {exchange_name.value}: {e}")
                    continue
            
            if exchange_data:
                all_data[exchange_name.value] = exchange_data
        
        return all_data
    
    async def _fetch_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from specific exchange"""
        
        # This is a simplified implementation
        # In practice, would use exchange-specific APIs
        
        try:
            # Simulate API call with random data for demonstration
            dates = pd.date_range(
                end=datetime.now(),
                periods=limit,
                freq='1H' if timeframe == '1h' else '1D'
            )
            
            # Generate realistic-looking price data
            base_price = 65000 if 'BTC' in symbol else 2500
            price_data = np.random.randn(limit).cumsum() * 100 + base_price
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': price_data,
                'high': price_data * (1 + np.random.rand(limit) * 0.02),
                'low': price_data * (1 - np.random.rand(limit) * 0.02),
                'close': price_data + np.random.randn(limit) * 50,
                'volume': np.random.rand(limit) * 1000
            })
            
            df.set_index('timestamp', inplace=True)
            
            # Add AUD conversion for Australian context
            if not symbol.endswith('AUD'):
                aud_usd_rate = await self.australian_data.get_aud_usd_rate()
                if aud_usd_rate:
                    df['close_aud'] = df['close'] / aud_usd_rate
                    df['volume_aud'] = df['volume'] / aud_usd_rate
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV from {exchange}: {e}")
            return None
    
    async def collect_funding_rates(self, symbols: List[str]) -> Dict[str, Any]:
        """Collect funding rates from perpetual futures"""
        
        funding_data = {}
        
        for exchange_name in [ExchangeName.BYBIT, ExchangeName.BINANCE]:
            exchange_funding = {}
            
            for symbol in symbols:
                try:
                    # Simulate funding rate data
                    funding_rate = (np.random.randn() * 0.0001)  # Realistic funding rate
                    next_funding_time = datetime.now() + timedelta(hours=8)
                    
                    exchange_funding[symbol] = {
                        'funding_rate': funding_rate,
                        'next_funding_time': next_funding_time,
                        'timestamp': datetime.now()
                    }
                    
                except Exception as e:
                    logger.error(f"Error collecting funding rate for {symbol} from {exchange_name.value}: {e}")
            
            if exchange_funding:
                funding_data[exchange_name.value] = exchange_funding
        
        return funding_data
    
    async def collect_australian_macro_data(self) -> Dict[str, Any]:
        """Collect Australian macroeconomic data"""
        
        macro_data = {}
        
        try:
            # RBA cash rate
            cash_rate = await self.australian_data.get_rba_cash_rate()
            if cash_rate:
                macro_data['rba_cash_rate'] = cash_rate
            
            # AUD/USD rate
            aud_usd = await self.australian_data.get_aud_usd_rate()
            if aud_usd:
                macro_data['aud_usd_rate'] = aud_usd
            
            # ASX 200 data
            asx_data = await self.australian_data.get_asx_200_data()
            if asx_data:
                macro_data.update(asx_data)
            
            # Gold price in AUD
            gold_aud = await self.australian_data.get_gold_price_aud()
            if gold_aud:
                macro_data['gold_price_aud'] = gold_aud
            
            # Market hours
            market_hours = await self.australian_data.get_australian_market_hours()
            macro_data.update(market_hours)
            
            logger.info(f"Collected {len(macro_data)} macro data points")
            
        except Exception as e:
            logger.error(f"Error collecting Australian macro data: {e}")
        
        return macro_data
    
    async def get_optimal_arbitrage_routes(
        self,
        symbol: str,
        amount: Decimal
    ) -> List[Dict[str, Any]]:
        """Get optimal arbitrage routes considering transfer costs"""
        
        routes = []
        
        # Get all Australian-friendly transfer routes
        currency = symbol.split('/')[0] if '/' in symbol else symbol[:3]
        transfer_routes = self.transfer_db.get_australian_friendly_routes(currency)
        
        for route in transfer_routes:
            transfer_cost = self.transfer_db.calculate_transfer_cost(
                route.from_exchange,
                route.to_exchange,
                currency,
                amount
            )
            
            if transfer_cost is not None:
                routes.append({
                    'from_exchange': route.from_exchange,
                    'to_exchange': route.to_exchange,
                    'currency': currency,
                    'transfer_cost': transfer_cost,
                    'transfer_cost_percentage': (transfer_cost / amount) * 100,
                    'estimated_time_minutes': route.estimated_time_minutes,
                    'australian_friendly': route.australian_friendly,
                    'net_amount': amount - transfer_cost
                })
        
        # Sort by lowest cost percentage
        routes.sort(key=lambda x: x['transfer_cost_percentage'])
        
        return routes
    
    async def create_comprehensive_dataset(
        self,
        symbols: List[str],
        days_back: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """Create comprehensive dataset for ML training"""
        
        logger.info(f"Creating comprehensive dataset for {len(symbols)} symbols, {days_back} days back")
        
        # Collect OHLCV data
        ohlcv_data = await self.collect_ohlcv_data(symbols, "1h", days_back * 24)
        
        # Collect funding rates
        funding_data = await self.collect_funding_rates(symbols)
        
        # Collect Australian macro data
        macro_data = await self.collect_australian_macro_data()
        
        # Combine data for each symbol
        combined_data = {}
        
        for symbol in symbols:
            symbol_data = None
            
            # Find best exchange data for this symbol
            for exchange, exchange_data in ohlcv_data.items():
                if symbol in exchange_data:
                    symbol_data = exchange_data[symbol].copy()
                    break
            
            if symbol_data is not None:
                # Add funding rate data
                for exchange, exchange_funding in funding_data.items():
                    if symbol in exchange_funding:
                        funding_info = exchange_funding[symbol]
                        symbol_data['funding_rate'] = funding_info['funding_rate']
                
                # Add macro data as constant columns
                for key, value in macro_data.items():
                    if isinstance(value, (int, float, Decimal)):
                        symbol_data[f'macro_{key}'] = float(value)
                    elif isinstance(value, bool):
                        symbol_data[f'macro_{key}'] = int(value)
                
                # Add transfer cost information
                optimal_routes = await self.get_optimal_arbitrage_routes(symbol, Decimal('10000'))
                if optimal_routes:
                    best_route = optimal_routes[0]
                    symbol_data['best_transfer_cost_pct'] = best_route['transfer_cost_percentage']
                    symbol_data['best_transfer_time_min'] = best_route['estimated_time_minutes']
                    symbol_data['australian_route_available'] = best_route['australian_friendly']
                
                combined_data[symbol] = symbol_data
        
        logger.info(f"Created dataset for {len(combined_data)} symbols")
        
        return combined_data

# Usage example
async def main():
    """Example usage of data infrastructure"""
    
    # Initialize data collector
    collector = MultiExchangeDataCollector()
    
    # Symbols to collect (AUD pairs prioritized for Australian traders)
    symbols = ['BTCAUD', 'ETHAUD', 'BTCUSDT', 'ETHUSDT']
    
    # Create comprehensive dataset
    dataset = await collector.create_comprehensive_dataset(symbols, days_back=7)
    
    print(f"Created dataset with {len(dataset)} symbols:")
    for symbol, data in dataset.items():
        print(f"  {symbol}: {len(data)} data points, {len(data.columns)} features")
        print(f"    Sample features: {list(data.columns)[:10]}")
    
    # Get transfer cost information
    transfer_costs = await collector.get_optimal_arbitrage_routes('BTC', Decimal('5000'))
    print(f"\nOptimal transfer routes for 5000 BTC:")
    for route in transfer_costs[:3]:
        print(f"  {route['from_exchange']} -> {route['to_exchange']}: "
              f"{route['transfer_cost_percentage']:.3f}% cost, "
              f"{route['estimated_time_minutes']} minutes")

if __name__ == "__main__":
    asyncio.run(main())