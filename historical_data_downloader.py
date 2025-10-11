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
                'sandbox': False,  # Use mainnet for historical data (public data doesn't need credentials)
                'rateLimit': 1200,
                'enableRateLimit': True,
                'timeout': 30000,  # 30 second timeout
                'options': {
                    'adjustForTimeDifference': True,
                }
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
    
    def _check_existing_data_coverage(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> Dict:
        """Check how much data already exists for the requested period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get the date range of existing data
                cursor = conn.execute("""
                    SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest, COUNT(*) as count
                    FROM historical_data
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                
                result = cursor.fetchone()
                if not result or not result[0]:
                    return {"coverage_percent": 0, "existing_count": 0, "message": "No existing data found"}
                
                existing_earliest = datetime.fromtimestamp(result[0] / 1000)
                existing_latest = datetime.fromtimestamp(result[1] / 1000)
                existing_count = result[2]
                
                # Calculate expected total candles for requested period
                timeframe_minutes = {
                    '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                    '1h': 60, '2h': 120, '4h': 240, '6h': 360,
                    '8h': 480, '12h': 720, '1d': 1440
                }
                minutes_per_candle = timeframe_minutes.get(timeframe, 60)
                total_minutes = (end_time - start_time).total_seconds() / 60
                expected_candles = int(total_minutes / minutes_per_candle)
                
                # Check overlap with requested period
                overlap_start = max(start_time, existing_earliest)
                overlap_end = min(end_time, existing_latest)
                
                if overlap_start >= overlap_end:
                    return {"coverage_percent": 0, "existing_count": 0, "message": "No data overlap with requested period"}
                
                # Count actual data points in the requested range
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM historical_data
                    WHERE symbol = ? AND timeframe = ? 
                    AND timestamp >= ? AND timestamp <= ?
                """, (symbol, timeframe, int(start_time.timestamp() * 1000), int(end_time.timestamp() * 1000)))
                
                overlapping_count = cursor.fetchone()[0]
                coverage_percent = min(100, (overlapping_count / expected_candles) * 100) if expected_candles > 0 else 0
                
                return {
                    "coverage_percent": round(coverage_percent, 1),
                    "existing_count": overlapping_count,
                    "total_expected": expected_candles,
                    "existing_range": f"{existing_earliest.strftime('%Y-%m-%d')} to {existing_latest.strftime('%Y-%m-%d')}",
                    "message": f"Found {overlapping_count} existing candles ({coverage_percent:.1f}% coverage)"
                }
                
        except Exception as e:
            logger.warning(f"⚠️ Error checking existing data: {e}")
            return {"coverage_percent": 0, "existing_count": 0, "message": "Error checking existing data"}

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
                
            # Convert symbol format for CCXT (ensure proper format)
            if '/' not in symbol:
                # Convert BTCUSDT to BTC/USDT format
                if symbol.endswith('USDT'):
                    base = symbol.replace('USDT', '')
                    ccxt_symbol = f"{base}/USDT"
                else:
                    ccxt_symbol = symbol
            else:
                ccxt_symbol = symbol
            
            logger.info(f"🔄 Using symbol format: {ccxt_symbol}")
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Define timeframe minutes for calculations
            timeframe_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360,
                '8h': 480, '12h': 720, '1d': 1440
            }
            
            logger.info(f"📥 Starting download: {ccxt_symbol} {timeframe} from {start_time} to {end_time}")
            logger.info(f"📊 Expected total candles: ~{int((days * 24 * 60) / timeframe_minutes.get(timeframe, 60))}")
            
            # Check existing data coverage
            coverage_info = self._check_existing_data_coverage(ccxt_symbol, timeframe, start_time, end_time)
            logger.info(f"🔍 Data coverage check: {coverage_info['message']}")
            
            # If we have high coverage (>90%), offer to download only missing data
            if coverage_info['coverage_percent'] >= 90:
                logger.info(f"📚 Already downloaded ({coverage_info['coverage_percent']:.1f}%), downloading remaining data...")
                # Continue with download to fill gaps, but let user know
            elif coverage_info['coverage_percent'] >= 50:
                logger.info(f"📊 Partial data exists ({coverage_info['coverage_percent']:.1f}%), downloading to complete dataset...")
            
            # Get exchange instance
            exchange_obj = self.exchanges.get(exchange)
            if not exchange_obj:
                raise ValueError(f"Unsupported exchange: {exchange}")
            
            # Download data in chunks to avoid rate limits
            all_data = []
            # Calculate optimal chunk size based on timeframe to stay within API limits
            max_candles_per_chunk = 999  # Bybit's actual limit
            minutes_per_candle = timeframe_minutes.get(timeframe, 60)
            max_minutes_per_chunk = max_candles_per_chunk * minutes_per_candle
            chunk_days = min(days, max_minutes_per_chunk / (24 * 60))  # Dynamic chunk size
            chunk_days = max(1, int(chunk_days))  # Ensure at least 1 day chunks
            
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
            
            # Create success message based on coverage
            if coverage_info['coverage_percent'] >= 90:
                message = f"Already downloaded ({coverage_info['coverage_percent']:.1f}%), added {stored_count} new data points"
            elif coverage_info['coverage_percent'] >= 50:
                message = f"Updated existing dataset ({coverage_info['coverage_percent']:.1f}% → now complete), added {stored_count} new data points"
            else:
                message = f"Successfully downloaded {len(unique_data)} data points"
            
            return {
                "success": True,
                "message": message,
                "data_points": len(unique_data),
                "stored_count": stored_count,
                "existing_coverage": coverage_info['coverage_percent'],
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
        calculated_candles = int(minutes_in_period / minutes_per_candle)
        
        # Use Bybit's actual limit of 999 candles per request
        return min(999, calculated_candles)
    
    async def _fetch_with_retry(
        self, 
        exchange, 
        symbol: str, 
        timeframe: str, 
        since: int, 
        limit: int,
        max_retries: int = 3
    ) -> List[List]:
        """Fetch data with retry logic"""
        
        for attempt in range(max_retries):
            try:
                # Run synchronous CCXT call in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None, 
                    lambda: exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                )
                
                if data and len(data) > 0:
                    logger.info(f"✅ Fetched {len(data)} candles for {symbol} {timeframe}")
                    return data
                else:
                    logger.warning(f"⚠️ No data returned for {symbol} {timeframe}")
                    return []
                
            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"❌ All {max_retries} attempts failed for {symbol}")
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

    def get_available_periods(self, symbol: str, timeframe: str) -> Dict:
        """Get available data periods for backtesting for a specific symbol and timeframe"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest, COUNT(*) as count
                    FROM historical_data 
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                
                result = cursor.fetchone()
                if not result or result[0] is None:
                    return {
                        'success': True,
                        'periods': [],
                        'message': f'No data available for {symbol} {timeframe}'
                    }
                
                earliest_ts, latest_ts, total_candles = result
                earliest_date = datetime.fromtimestamp(earliest_ts / 1000)
                latest_date = datetime.fromtimestamp(latest_ts / 1000)
                
                # Calculate available periods based on actual data
                periods = []
                now = latest_date
                
                # Generate meaningful periods based on available data
                time_ranges = [
                    ("7 Days", 7),
                    ("14 Days", 14), 
                    ("30 Days", 30),
                    ("2 Months", 60),
                    ("3 Months", 90),
                    ("6 Months", 180),
                    ("1 Year", 365),
                    ("2 Years", 730),
                    ("All Available Data", None)
                ]
                
                for label, days in time_ranges:
                    if days is None:
                        # All available data
                        periods.append({
                            'label': f"{label} ({earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')})",
                            'value': 'all',
                            'days': (latest_date - earliest_date).days,
                            'start_date': earliest_date.isoformat(),
                            'end_date': latest_date.isoformat(),
                            'estimated_candles': total_candles
                        })
                    else:
                        start_date = now - timedelta(days=days)
                        if start_date >= earliest_date:
                            # Check if we have enough data for this period
                            cursor.execute("""
                                SELECT COUNT(*) FROM historical_data 
                                WHERE symbol = ? AND timeframe = ?
                                AND timestamp >= ? AND timestamp <= ?
                            """, (symbol, timeframe, int(start_date.timestamp() * 1000), int(now.timestamp() * 1000)))
                            
                            candle_count = cursor.fetchone()[0]
                            if candle_count > 0:
                                periods.append({
                                    'label': f"{label} ({start_date.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')})",
                                    'value': f"{days}d",
                                    'days': days,
                                    'start_date': start_date.isoformat(),
                                    'end_date': now.isoformat(), 
                                    'estimated_candles': candle_count
                                })
                
                return {
                    'success': True,
                    'periods': periods,
                    'total_available_days': (latest_date - earliest_date).days,
                    'data_range': {
                        'earliest': earliest_date.isoformat(),
                        'latest': latest_date.isoformat(),
                        'total_candles': total_candles
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting available periods for {symbol} {timeframe}: {e}")
            return {
                'success': False,
                'periods': [],
                'error': str(e)
            }

    def delete_symbol_data(self, symbol: str, timeframe: str = None) -> Dict:
        """Delete historical data for a specific symbol and optionally timeframe"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if timeframe:
                    # Delete specific symbol-timeframe combination
                    cursor = conn.execute("""
                        DELETE FROM historical_data 
                        WHERE symbol = ? AND timeframe = ?
                    """, (symbol, timeframe))
                    deleted_count = cursor.rowcount
                    logger.info(f"🗑️ Deleted {deleted_count} candles for {symbol} {timeframe}")
                else:
                    # Delete all data for symbol across all timeframes
                    cursor = conn.execute("""
                        DELETE FROM historical_data 
                        WHERE symbol = ?
                    """, (symbol,))
                    deleted_count = cursor.rowcount
                    logger.info(f"🗑️ Deleted {deleted_count} candles for {symbol} (all timeframes)")
                
                conn.commit()
                
                return {
                    'success': True,
                    'message': f'Deleted {deleted_count} candles',
                    'deleted_count': deleted_count
                }
                
        except Exception as e:
            logger.error(f"❌ Error deleting data for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'deleted_count': 0
            }

    def clear_all_data(self) -> Dict:
        """Clear all historical data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM historical_data")
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"🗑️ Cleared all historical data: {deleted_count} candles deleted")
                
                return {
                    'success': True,
                    'message': f'Cleared all data: {deleted_count} candles deleted',
                    'deleted_count': deleted_count
                }
                
        except Exception as e:
            logger.error(f"❌ Error clearing all data: {e}")
            return {
                'success': False,
                'error': str(e),
                'deleted_count': 0
            }

    def validate_data_integrity(self, symbol: str, timeframe: str) -> Dict:
        """Check for data corruption, gaps, and anomalies"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all candles for symbol-timeframe ordered by timestamp
                cursor = conn.execute("""
                    SELECT timestamp, open, high, low, close, volume
                    FROM historical_data 
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp ASC
                """, (symbol, timeframe))
                
                candles = cursor.fetchall()
                if not candles:
                    return {
                        'success': True,
                        'issues': [],
                        'summary': {'total_candles': 0, 'issues_found': 0}
                    }
                
                issues = []
                
                # Check for data corruption and anomalies
                for i, (timestamp, open_price, high, low, close, volume) in enumerate(candles):
                    candle_issues = []
                    
                    # Check OHLC validity
                    if not (low <= open_price <= high and low <= close <= high):
                        candle_issues.append("Invalid OHLC: prices outside high-low range")
                    
                    if high < low:
                        candle_issues.append("Invalid OHLC: high < low")
                    
                    # Check for negative values
                    if any(val < 0 for val in [open_price, high, low, close, volume]):
                        candle_issues.append("Negative price or volume values")
                    
                    # Check for zero values (might be valid for volume but suspicious for prices)
                    if any(val == 0 for val in [open_price, high, low, close]):
                        candle_issues.append("Zero price values detected")
                    
                    # Check for extreme price movements (more than 50% change)
                    if i > 0:
                        prev_close = candles[i-1][4]  # Previous candle's close
                        if prev_close > 0:  # Avoid division by zero
                            price_change_pct = abs(open_price - prev_close) / prev_close * 100
                            if price_change_pct > 50:
                                candle_issues.append(f"Extreme price gap: {price_change_pct:.1f}% change from previous close")
                    
                    if candle_issues:
                        issues.append({
                            'timestamp': datetime.fromtimestamp(timestamp / 1000).isoformat(),
                            'candle_index': i,
                            'issues': candle_issues,
                            'data': {
                                'open': open_price, 'high': high, 'low': low, 
                                'close': close, 'volume': volume
                            }
                        })
                
                # Check for time gaps
                timeframe_minutes = self._get_timeframe_minutes(timeframe)
                for i in range(1, len(candles)):
                    current_time_ms = candles[i][0]
                    prev_time_ms = candles[i-1][0]
                    expected_diff_ms = timeframe_minutes * 60 * 1000
                    actual_diff_ms = current_time_ms - prev_time_ms
                    
                    if actual_diff_ms > expected_diff_ms * 1.5:  # Allow some tolerance
                        gap_minutes = actual_diff_ms / (1000 * 60)
                        issues.append({
                            'timestamp': datetime.fromtimestamp(current_time_ms / 1000).isoformat(),
                            'candle_index': i,
                            'issues': [f"Time gap detected: {gap_minutes:.0f} minutes (expected {timeframe_minutes})"],
                            'data': {'gap_duration_minutes': gap_minutes}
                        })
                
                return {
                    'success': True,
                    'issues': issues,
                    'summary': {
                        'total_candles': len(candles),
                        'issues_found': len(issues),
                        'period_start': datetime.fromtimestamp(candles[0][0] / 1000).isoformat() if candles else None,
                        'period_end': datetime.fromtimestamp(candles[-1][0] / 1000).isoformat() if candles else None,
                        'has_corruption': any('Invalid OHLC' in str(issue['issues']) for issue in issues),
                        'has_gaps': any('Time gap' in str(issue['issues']) for issue in issues),
                        'has_anomalies': any('Extreme price' in str(issue['issues']) for issue in issues)
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Error validating data integrity for {symbol} {timeframe}: {e}")
            return {
                'success': False,
                'error': str(e),
                'issues': []
            }

    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return timeframe_map.get(timeframe, 15)  # Default to 15 minutes

    # Legacy methods for compatibility
    def get_available_symbols(self):
        """Get available symbols for download"""
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
    
    def get_status(self):
        """Get downloader status"""
        return {"status": "ready", "type": "full_implementation"}
    
    def get_historical_performance(self, symbol: str, timeframe: str, limit: int = 30) -> Dict:
        """Get historical performance data for a symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent historical data
                cursor.execute("""
                    SELECT timestamp, open, high, low, close, volume 
                    FROM historical_data 
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (symbol, timeframe, limit))
                
                rows = cursor.fetchall()
                if not rows:
                    return {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "data": [],
                        "performance": {
                            "total_return": 0,
                            "volatility": 0,
                            "max_drawdown": 0
                        }
                    }
                
                # Calculate performance metrics
                prices = [row[4] for row in rows]  # closing prices
                if len(prices) > 1:
                    total_return = ((prices[0] - prices[-1]) / prices[-1]) * 100
                    returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
                    volatility = (sum(r**2 for r in returns) / len(returns))**0.5 * 100 if returns else 0
                    
                    # Calculate max drawdown
                    peak = prices[-1]
                    max_drawdown = 0
                    for price in reversed(prices):
                        if price > peak:
                            peak = price
                        drawdown = ((peak - price) / peak) * 100
                        max_drawdown = max(max_drawdown, drawdown)
                else:
                    total_return = volatility = max_drawdown = 0
                
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data": [
                        {
                            "timestamp": row[0],
                            "open": row[1],
                            "high": row[2], 
                            "low": row[3],
                            "close": row[4],
                            "volume": row[5]
                        }
                        for row in reversed(rows)
                    ],
                    "performance": {
                        "total_return": round(total_return, 2),
                        "volatility": round(volatility, 2),
                        "max_drawdown": round(max_drawdown, 2)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting historical performance for {symbol}: {e}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": [],
                "performance": {"total_return": 0, "volatility": 0, "max_drawdown": 0},
                "error": str(e)
            }


# Global instance
historical_downloader = HistoricalDataDownloader()

# Async function for compatibility
async def download_data(symbol: str, timeframe: str = '1h', days: int = 30) -> Dict:
    """Convenience function for downloading historical data"""
    return await historical_downloader.download_historical_data(symbol, timeframe, days)