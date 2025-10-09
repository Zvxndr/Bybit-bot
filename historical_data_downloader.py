#!/usr/bin/env python3
"""
Historical Data Downloader for Bybit Trading Bot
Supports downloading up to 10 years of historical market data
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import os
from loguru import logger
import time


class HistoricalDataDownloader:
    """Downloads and manages historical market data with support for up to 10 years"""
    
    def __init__(self, db_path: str = "data/historical_data.db"):
        self.db_path = db_path
        self.exchanges = {
            'bybit': ccxt.bybit({
                'sandbox': True,  # Use testnet by default
                'rateLimit': 1200,
                'enableRateLimit': True,
            })
        }
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for historical data storage"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
                ON historical_data(symbol, timeframe, timestamp)
            """)
            
            logger.info(f"✅ Historical data database initialized: {self.db_path}")
    
    async def download_historical_data(
        self, 
        symbol: str, 
        timeframe: str = '1h', 
        days: int = 30,
        exchange: str = 'bybit'
    ) -> Dict:
        """
        Download historical data for specified period
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            days: Number of days to download (up to 3650 for 10 years)
            exchange: Exchange to use
            
        Returns:
            Dict with success status and data information
        """
        try:
            # Validate inputs
            if days > 3650:  # 10 years maximum
                days = 3650
                logger.warning(f"⚠️ Limited download to 10 years maximum")
                
            # Convert symbol format for CCXT
            ccxt_symbol = symbol.replace('USDT', '/USDT').replace('BTC', 'BTC').replace('ETH', 'ETH')
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            logger.info(f"📥 Starting download: {ccxt_symbol} {timeframe} from {start_time} to {end_time}")
            
            # Get exchange instance
            exchange_obj = self.exchanges.get(exchange)
            if not exchange_obj:
                raise ValueError(f"Unsupported exchange: {exchange}")
            
            # Download data in chunks to avoid rate limits
            all_data = []
            chunk_days = min(30, days)  # Download in 30-day chunks
            current_start = start_time
            
            total_chunks = (days // chunk_days) + (1 if days % chunk_days > 0 else 0)
            processed_chunks = 0
            
            while current_start < end_time:
                chunk_end = min(current_start + timedelta(days=chunk_days), end_time)
                
                try:
                    # Fetch OHLCV data
                    since = int(current_start.timestamp() * 1000)
                    limit = self._calculate_limit(timeframe, chunk_days)
                    
                    ohlcv_data = await self._fetch_with_retry(
                        exchange_obj, ccxt_symbol, timeframe, since, limit
                    )
                    
                    if ohlcv_data:
                        all_data.extend(ohlcv_data)
                        logger.info(f"📊 Downloaded {len(ohlcv_data)} candles for chunk {processed_chunks + 1}/{total_chunks}")
                    
                    processed_chunks += 1
                    current_start = chunk_end
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as chunk_error:
                    logger.error(f"❌ Error downloading chunk {processed_chunks + 1}: {chunk_error}")
                    # Continue with next chunk
                    current_start = chunk_end
                    processed_chunks += 1
            
            if not all_data:
                return {
                    "success": False,
                    "message": f"No data retrieved for {symbol}",
                    "data_points": 0
                }
            
            # Sort data by timestamp and remove duplicates
            all_data.sort(key=lambda x: x[0])
            unique_data = []
            seen_timestamps = set()
            
            for candle in all_data:
                if candle[0] not in seen_timestamps:
                    unique_data.append(candle)
                    seen_timestamps.add(candle[0])
            
            # Store in database
            stored_count = await self._store_data(symbol, timeframe, unique_data)
            
            return {
                "success": True,
                "message": f"Successfully downloaded {len(unique_data)} data points",
                "data_points": len(unique_data),
                "stored_count": stored_count,
                "symbol": symbol,
                "timeframe": timeframe,
                "period_days": days,
                "start_date": start_time.isoformat(),
                "end_date": end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Historical data download failed: {e}")
            return {
                "success": False,
                "message": str(e),
                "data_points": 0
            }
    
    def _calculate_limit(self, timeframe: str, days: int) -> int:
        """Calculate the number of candles needed for the timeframe and period"""
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360,
            '8h': 480, '12h': 720, '1d': 1440
        }
        
        minutes_in_period = days * 24 * 60
        minutes_per_candle = timeframe_minutes.get(timeframe, 60)
        
        return min(1000, int(minutes_in_period / minutes_per_candle))  # Limit to 1000 candles per request
    
    async def _fetch_with_retry(
        self, 
        exchange, 
        symbol: str, 
        timeframe: str, 
        since: int, 
        limit: int,
        max_retries: int = 3
    ) -> List:
        """Fetch data with retry logic"""
        
        for attempt in range(max_retries):
            try:
                data = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                return data
                
            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e
    
    async def _store_data(self, symbol: str, timeframe: str, ohlcv_data: List) -> int:
        """Store OHLCV data in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stored_count = 0
                
                for candle in ohlcv_data:
                    try:
                        conn.execute("""
                            INSERT OR REPLACE INTO historical_data
                            (symbol, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            symbol.replace('/', ''),  # Store as BTCUSDT format
                            timeframe,
                            candle[0],  # timestamp
                            candle[1],  # open
                            candle[2],  # high
                            candle[3],  # low
                            candle[4],  # close
                            candle[5] or 0  # volume
                        ))
                        stored_count += 1
                    except sqlite3.IntegrityError:
                        # Duplicate entry, skip
                        pass
                
                conn.commit()
                logger.info(f"💾 Stored {stored_count} new candles in database")
                return stored_count
                
        except Exception as e:
            logger.error(f"❌ Database storage error: {e}")
            return 0
    
    def get_stored_data(
        self, 
        symbol: str, 
        timeframe: str, 
        days: int = 30
    ) -> pd.DataFrame:
        """Retrieve stored historical data as DataFrame"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM historical_data
                    WHERE symbol = ? AND timeframe = ? 
                    AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                """
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[
                        symbol.replace('/', ''),
                        timeframe,
                        int(start_time.timestamp() * 1000),
                        int(end_time.timestamp() * 1000)
                    ]
                )
                
                if not df.empty:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('datetime', inplace=True)
                
                return df
                
        except Exception as e:
            logger.error(f"❌ Error retrieving stored data: {e}")
            return pd.DataFrame()
    
    def clear_data(self, symbol: Optional[str] = None) -> bool:
        """Clear stored historical data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if symbol:
                    conn.execute("DELETE FROM historical_data WHERE symbol = ?", (symbol,))
                    logger.info(f"🗑️ Cleared data for {symbol}")
                else:
                    conn.execute("DELETE FROM historical_data")
                    logger.info("🗑️ Cleared all historical data")
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Error clearing data: {e}")
            return False
    
    def get_data_summary(self) -> Dict:
        """Get summary of stored historical data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        symbol,
                        timeframe,
                        COUNT(*) as candle_count,
                        MIN(timestamp) as earliest_timestamp,
                        MAX(timestamp) as latest_timestamp
                    FROM historical_data
                    GROUP BY symbol, timeframe
                    ORDER BY symbol, timeframe
                """)
                
                summary = []
                for row in cursor.fetchall():
                    summary.append({
                        'symbol': row[0],
                        'timeframe': row[1],
                        'candle_count': row[2],
                        'earliest_date': datetime.fromtimestamp(row[3] / 1000).isoformat() if row[3] else None,
                        'latest_date': datetime.fromtimestamp(row[4] / 1000).isoformat() if row[4] else None
                    })
                
                return {
                    'success': True,
                    'summary': summary,
                    'total_symbols': len(set(item['symbol'] for item in summary))
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting data summary: {e}")
            return {
                'success': False,
                'summary': [],
                'error': str(e)
            }

    # Legacy methods for compatibility
    def get_available_symbols(self):
        """Get available symbols for download"""
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
    
    def get_status(self):
        """Get downloader status"""
        return {"status": "ready", "type": "full_implementation"}


# Global instance
historical_downloader = HistoricalDataDownloader()

# Async function for compatibility
async def download_data(symbol: str, timeframe: str = '1h', days: int = 30) -> Dict:
    """Convenience function for downloading historical data"""
    return await historical_downloader.download_historical_data(symbol, timeframe, days)