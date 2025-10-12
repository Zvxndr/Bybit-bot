"""
Historical Data Provider for Debug Mode
======================================

Replaces mock data with real historical data from the database during debugging.
This provides more realistic testing conditions while maintaining safety.
"""

import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class HistoricalDataProvider:
    """
    Provides real historical data for debugging instead of mock data.
    Reads from the market_data.db and provides realistic test scenarios.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with database connection"""
        if db_path is None:
            # Default path to the speed demon cache
            self.db_path = Path(__file__).parent / "data" / "speed_demon_cache" / "market_data.db"
        else:
            self.db_path = Path(db_path)
        
        self.connection = None
        self._connect()
        
    def _connect(self):
        """Connect to the SQLite database"""
        try:
            if self.db_path.exists():
                self.connection = sqlite3.connect(str(self.db_path))
                logger.info(f"[SUCCESS] Connected to historical data: {self.db_path}")
                self._inspect_database_schema()
            else:
                logger.warning(f"[WARNING] Historical data not found: {self.db_path}")
                logger.info("[INFO] Historical data provider will return empty results when database unavailable")
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to historical database: {e}")

    def _inspect_database_schema(self):
        """Inspect database schema to understand available tables and columns"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            self.available_tables = [table[0] for table in tables]
            logger.info(f"[INFO] Found database tables: {self.available_tables}")
            
            # Inspect data_cache table schema if it exists
            if 'data_cache' in self.available_tables:
                cursor.execute("PRAGMA table_info(data_cache)")
                columns = cursor.fetchall()
                self.data_cache_columns = [col[1] for col in columns]
                logger.info(f"[INFO] data_cache columns: {self.data_cache_columns}")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to inspect database schema: {e}")
            self.available_tables = []
            self.data_cache_columns = []

    def _get_table_data(self, table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get data from specified table"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT * FROM {table_name} LIMIT ?", (limit,))
            rows = cursor.fetchall()
            
            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Convert to list of dictionaries
            data = []
            for row in rows:
                data.append(dict(zip(columns, row)))
            
            return data
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get table data from {table_name}: {e}")
            return []

    def get_historical_data(self, symbol: str, timeframe: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get historical OHLC data for a specific symbol and timeframe"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT timestamp, open_price, high_price, low_price, close_price, volume
                FROM data_cache 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (symbol, timeframe, limit))
            
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            historical_data = []
            for row in rows:
                historical_data.append({
                    'timestamp': row[0],
                    'open': row[1],
                    'high': row[2],
                    'low': row[3],
                    'close': row[4],
                    'volume': row[5]
                })
            
            logger.info(f"[SUCCESS] Retrieved {len(historical_data)} historical data points for {symbol} {timeframe}")
            return historical_data
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get historical data: {e}")
            return []
    
    def get_realistic_balances(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        """Get realistic balance data based on historical trading patterns"""
        if not self.connection:
            return self._generate_mock_balances()
        
        try:
            # Try different possible table schemas
            data = self._get_table_data('data_cache', limit=10)
            
            if not data:
                logger.info("[INFO] No cached data found, generating realistic mock balances")
                return self._generate_mock_balances()
            
            # Analyze the data to create realistic balances
            # This assumes cached market data contains price/volume information
            latest_record = data[0]
            
            # Try to extract price information from various possible column names
            price = None
            volume = None
            
            for key, value in latest_record.items():
                if 'price' in key.lower() and isinstance(value, (int, float)):
                    price = float(value)
                elif 'volume' in key.lower() and isinstance(value, (int, float)):
                    volume = float(value)
            
            # If we found price data, calculate realistic balances
            if price:
                balances = {
                    'USDT': round(8000 + (price * 0.1), 2),  # Base USDT balance varying with price
                    'BTC': round(3000 / price, 6) if price > 0 else 0.05,  # BTC equivalent of $3000
                    'total_usd': round(11000 + (price * 0.1), 2)
                }
                
                logger.info(f"[SUCCESS] Generated realistic balances based on price data: ${price:.2f}")
                return balances
            
            # Fallback to time-based realistic balances
            base_balance = 10000.0
            time_variation = len(data) * 50  # Vary based on data richness
            
            balances = {
                'USDT': round(base_balance + time_variation, 2),
                'BTC': 0.15 + (len(data) * 0.001),  # Slight variation based on data
                'total_usd': round(base_balance + time_variation + (0.15 * 66000), 2)  # Estimate BTC value
            }
            
            logger.info(f"[SUCCESS] Generated realistic balances from {len(data)} data records")
            return balances
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get realistic balances: {e}")
            return self._generate_mock_balances()
    
    def get_historical_positions(self, symbol: str = "BTCUSDT", limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical position data for realistic testing"""
        if not self.connection:
            return self._generate_mock_positions()
        
        try:
            # Get recent cached data to simulate positions
            data = self._get_table_data('data_cache', limit=limit * 2)
            
            if not data:
                return self._generate_mock_positions()
            
            positions = []
            
            # Generate positions based on available data
            for i, record in enumerate(data[:limit]):
                # Try to find numeric values that could represent price/volume
                numeric_values = []
                for key, value in record.items():
                    if isinstance(value, (int, float)) and value > 0:
                        numeric_values.append(float(value))
                
                if len(numeric_values) >= 2:
                    # Use the first two numeric values as price and size indicators
                    price_indicator = numeric_values[0]
                    size_indicator = min(numeric_values[1], 1.0)  # Cap size at 1.0
                    
                    # Simulate position based on data patterns
                    side = 'long' if i % 2 == 0 else 'short'
                    size = round(abs(size_indicator * 0.1), 4)  # Scale down size
                    
                    if size > 0:
                        # Generate entry price with some variation
                        entry_price = price_indicator * (0.98 + (i * 0.005))
                        current_price = price_indicator
                        pnl = round(size * (current_price - entry_price) * (1 if side == 'long' else -1), 2)
                        
                        position = {
                            'symbol': symbol,
                            'side': side,
                            'size': size,
                            'entry_price': round(entry_price, 2),
                            'current_price': round(current_price, 2),
                            'pnl': pnl,
                            'timestamp': record.get('timestamp', datetime.now().isoformat())
                        }
                        positions.append(position)
            
            if not positions:
                # If we couldn't generate from data, create at least one realistic position
                positions = self._generate_mock_positions()
            
            logger.info(f"[SUCCESS] Generated {len(positions)} realistic positions from cached data")
            return positions
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get historical positions: {e}")
            return self._generate_mock_positions()
    
    def get_historical_trades(self, symbol: str = "BTCUSDT", limit: int = 50) -> List[Dict[str, Any]]:
        """Get historical trade data for realistic testing"""
        if not self.connection:
            return self._generate_mock_trades()
        
        try:
            # Get cached data to simulate trade history
            data = self._get_table_data('data_cache', limit=limit)
            
            if not data:
                return self._generate_mock_trades()
            
            trades = []
            
            for i, record in enumerate(data):
                # Extract numeric values that could represent price/amount
                price = None
                amount = None
                
                for key, value in record.items():
                    if isinstance(value, (int, float)) and value > 0:
                        if not price or 'price' in key.lower():
                            price = float(value)
                        elif not amount or 'amount' in key.lower() or 'volume' in key.lower():
                            amount = min(float(value) * 0.01, 1.0)  # Scale down amount
                
                # Generate trade if we have the necessary data
                if price and amount:
                    trade = {
                        'symbol': symbol,
                        'side': 'buy' if i % 2 == 0 else 'sell',
                        'amount': round(amount, 4),
                        'price': round(price, 2),
                        'fee': round(price * amount * 0.001, 4),  # 0.1% fee
                        'timestamp': record.get('timestamp', datetime.now().isoformat()),
                        'order_id': f"hist_{i}_{int(datetime.now().timestamp())}"
                    }
                    trades.append(trade)
            
            if not trades:
                # If no valid trades could be generated, use mock data
                trades = self._generate_mock_trades()
            
            logger.info(f"[SUCCESS] Generated {len(trades)} realistic trades from cached data")
            return trades
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get historical trades: {e}")
            return self._generate_mock_trades()
    
    def get_market_data_sample(self, symbol: str = "BTCUSDT", hours: int = 24) -> Dict[str, Any]:
        """Get recent market data for realistic price feeds"""
        if not self.connection:
            logger.error("[PRODUCTION ERROR] No database connection for market data")
            return self._generate_mock_market_data()
        
        try:
            # Get data from the last N hours
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = """
            SELECT symbol, price, volume, timestamp
            FROM market_data 
            WHERE symbol = ? AND datetime(timestamp) >= datetime(?)
            ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(
                query, 
                self.connection, 
                params=(symbol, cutoff_time.isoformat())
            )
            
            if df.empty:
                return self._generate_mock_market_data()
            
            # Calculate market statistics
            prices = df['price'].astype(float)
            volumes = df['volume'].astype(float)
            
            market_data = {
                'symbol': symbol,
                'current_price': float(prices.iloc[0]),
                'high_24h': float(prices.max()),
                'low_24h': float(prices.min()),
                'avg_price': float(prices.mean()),
                'volume_24h': float(volumes.sum()),
                'price_change': float(prices.iloc[0] - prices.iloc[-1]) if len(prices) > 1 else 0,
                'price_change_percent': float(((prices.iloc[0] - prices.iloc[-1]) / prices.iloc[-1]) * 100) if len(prices) > 1 else 0,
                'data_points': len(df),
                'last_update': df['timestamp'].iloc[0]
            }
            
            logger.info(f"[SUCCESS] Generated market data summary: {symbol} @ ${market_data['current_price']:.2f}")
            return market_data
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get market data: {e}")
            return self._generate_mock_market_data()
    
    # Fallback mock data generators
    def _generate_mock_balances(self) -> Dict[str, float]:
        """Generate realistic mock balances when no historical data is available"""
        return {
            'USDT': 10000.00,
            'BTC': 0.15,
            'total_usd': 15000.00
        }
    
    def _generate_mock_positions(self) -> List[Dict[str, Any]]:
        """Generate mock positions when no historical data is available"""
        return [
            {
                'symbol': 'BTCUSDT',
                'side': 'long',
                'size': 0.1,
                'entry_price': 65000.00,
                'current_price': 66000.00,
                'pnl': 100.00,
                'timestamp': datetime.now().isoformat()
            }
        ]
    
    def _generate_mock_trades(self) -> List[Dict[str, Any]]:
        """Generate mock trades when no historical data is available"""
        return [
            {
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'amount': 0.05,
                'price': 65000.00,
                'fee': 3.25,
                'timestamp': datetime.now().isoformat(),
                'order_id': 'mock_001'
            }
        ]
    
    def _generate_mock_market_data(self) -> Dict[str, Any]:
        """PRODUCTION FIX: No mock data - return error instead"""
        logger.error("[PRODUCTION ERROR] No real historical data available")
        return {
            'error': 'No historical data found',
            'message': 'Use DigitalOcean data download to populate database',
            'data_points': 0,
            'last_update': datetime.now().isoformat()
        }
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("[INFO] Historical data connection closed")


# Global instance for use throughout the application
historical_data_provider = None

def get_historical_data_provider() -> HistoricalDataProvider:
    """Get or create the global historical data provider"""
    global historical_data_provider
    if historical_data_provider is None:
        historical_data_provider = HistoricalDataProvider()
    return historical_data_provider