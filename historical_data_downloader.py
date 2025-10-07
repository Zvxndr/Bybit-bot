#!/usr/bin/env python3
"""
Historical Data Downloader for Bybit Trading Bot
Downloads and manages real market data for portfolio performance analysis
"""

import requests
import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalDataDownloader:
    """Downloads and manages historical market data from Bybit API"""
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        self.db_path = db_path
        self.base_url = "https://api.bybit.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Bybit-Trading-Bot/1.0'
        })
        self.init_database()
    
    def init_database(self):
        """Initialize database tables for historical data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create historical_data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_historical_symbol_timeframe_timestamp 
                ON historical_data(symbol, timeframe, timestamp DESC)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Historical data database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds"""
        timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400,
            '1w': 604800
        }
        return timeframe_map.get(timeframe, 3600)
    
    def download_klines(self, symbol: str, timeframe: str, days: int) -> Dict[str, Any]:
        """Download historical kline data from Bybit"""
        try:
            # Calculate time range
            end_time = int(time.time() * 1000)  # Current time in milliseconds
            start_time = end_time - (days * 24 * 60 * 60 * 1000)  # X days ago
            
            # Bybit API endpoint for klines
            endpoint = f"{self.base_url}/v5/market/kline"
            
            params = {
                'category': 'spot',  # Use spot market for clean price data
                'symbol': symbol,
                'interval': timeframe,
                'start': start_time,
                'end': end_time,
                'limit': 1000  # Max limit per request
            }
            
            logger.info(f"📡 Downloading {symbol} {timeframe} data for {days} days...")
            
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('retCode') != 0:
                raise Exception(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
            
            klines = data.get('result', {}).get('list', [])
            
            if not klines:
                logger.warning("⚠️ No historical data received from API")
                return {
                    'success': False,
                    'message': 'No data available for the specified period',
                    'data_points': 0
                }
            
            # Process and store data
            stored_count = self.store_historical_data(symbol, timeframe, klines)
            
            logger.info(f"✅ Successfully downloaded {len(klines)} data points, stored {stored_count}")
            
            return {
                'success': True,
                'message': f'Downloaded {len(klines)} data points',
                'data_points': len(klines),
                'stored_count': stored_count,
                'period_start': datetime.fromtimestamp(start_time / 1000).isoformat(),
                'period_end': datetime.fromtimestamp(end_time / 1000).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading data: {e}")
            return {
                'success': False,
                'message': f'Network error: {str(e)}',
                'data_points': 0
            }
        except Exception as e:
            logger.error(f"Error downloading historical data: {e}")
            return {
                'success': False,
                'message': str(e),
                'data_points': 0
            }
    
    def store_historical_data(self, symbol: str, timeframe: str, klines: List) -> int:
        """Store historical data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stored_count = 0
            
            for kline in klines:
                try:
                    # Bybit kline format: [timestamp, open, high, low, close, volume, turnover]
                    timestamp = int(kline[0])  # Already in milliseconds
                    open_price = float(kline[1])
                    high_price = float(kline[2])
                    low_price = float(kline[3])
                    close_price = float(kline[4])
                    volume = float(kline[5])
                    
                    # Insert with IGNORE to handle duplicates
                    cursor.execute('''
                        INSERT OR IGNORE INTO historical_data 
                        (symbol, timeframe, timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, timeframe, timestamp, open_price, high_price, low_price, close_price, volume))
                    
                    if cursor.rowcount > 0:
                        stored_count += 1
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping invalid kline data: {kline}, error: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing historical data: {e}")
            return 0
    
    def get_historical_performance(self, symbol: str = None, timeframe: str = None, limit: int = 90) -> Dict[str, Any]:
        """Get stored historical data for performance analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query based on parameters
            where_conditions = []
            params = []
            
            if symbol:
                where_conditions.append("symbol = ?")
                params.append(symbol)
            
            if timeframe:
                where_conditions.append("timeframe = ?")
                params.append(timeframe)
            
            where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
            params.append(limit)
            
            query = f'''
                SELECT symbol, timeframe, timestamp, open_price, high_price, low_price, close_price, volume
                FROM historical_data 
                {where_clause}
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {
                    'success': False,
                    'message': 'No historical data found',
                    'data': []
                }
            
            # Format data for frontend
            formatted_data = []
            for row in rows:
                formatted_data.append({
                    'symbol': row[0],
                    'timeframe': row[1],
                    'timestamp': row[2],
                    'open': row[3],
                    'high': row[4],
                    'low': row[5],
                    'close': row[6],
                    'volume': row[7]
                })
            
            # Sort by timestamp ascending for proper chart display
            formatted_data.reverse()
            
            return {
                'success': True,
                'message': f'Retrieved {len(formatted_data)} data points',
                'data': formatted_data,
                'data_points': len(formatted_data)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving historical performance: {e}")
            return {
                'success': False,
                'message': str(e),
                'data': []
            }
    
    def clear_historical_data(self, symbol: str = None, timeframe: str = None) -> Dict[str, Any]:
        """Clear historical data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if symbol and timeframe:
                cursor.execute("DELETE FROM historical_data WHERE symbol = ? AND timeframe = ?", (symbol, timeframe))
                message = f"Cleared data for {symbol} {timeframe}"
            elif symbol:
                cursor.execute("DELETE FROM historical_data WHERE symbol = ?", (symbol,))
                message = f"Cleared all data for {symbol}"
            else:
                cursor.execute("DELETE FROM historical_data")
                message = "Cleared all historical data"
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"🗑️ {message} ({deleted_count} records)")
            
            return {
                'success': True,
                'message': message,
                'deleted_count': deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error clearing historical data: {e}")
            return {
                'success': False,
                'message': str(e),
                'deleted_count': 0
            }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available historical data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    symbol, 
                    timeframe, 
                    COUNT(*) as data_points,
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM historical_data 
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            summary = []
            for row in rows:
                summary.append({
                    'symbol': row[0],
                    'timeframe': row[1],
                    'data_points': row[2],
                    'earliest': datetime.fromtimestamp(row[3] / 1000).isoformat() if row[3] else None,
                    'latest': datetime.fromtimestamp(row[4] / 1000).isoformat() if row[4] else None
                })
            
            return {
                'success': True,
                'summary': summary,
                'total_datasets': len(summary)
            }
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {
                'success': False,
                'message': str(e),
                'summary': []
            }

# Global instance for use by main application
historical_downloader = HistoricalDataDownloader()