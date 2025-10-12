#!/usr/bin/env python3
"""
🚀 Simple Test Server - Historical Data Integration
Minimal API server to test the backtesting controls integration
"""

import json
import sqlite3
from datetime import datetime
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DB_PATH = "data/trading_bot.db"

@app.route('/')
def index():
    """Serve the dashboard"""
    return send_from_directory('.', 'frontend/unified_dashboard.html')

@app.route('/api/historical-data/discover')
def discover_historical_data():
    """Enhanced API endpoint to discover available historical data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get available datasets with proper error handling
        cursor.execute("""
            SELECT 
                symbol, 
                timeframe, 
                COUNT(*) as record_count,
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest
            FROM historical_data
            GROUP BY symbol, timeframe
            HAVING COUNT(*) > 0
            ORDER BY record_count DESC
        """)
        
        results = cursor.fetchall()
        datasets = []
        total_records = 0
        
        for row in results:
            symbol, timeframe, count, earliest, latest = row
            total_records += count
            
            try:
                # Handle different timestamp formats
                if isinstance(earliest, (int, float)):
                    if earliest > 1e10:  # Milliseconds
                        start_dt = datetime.fromtimestamp(earliest / 1000)
                        end_dt = datetime.fromtimestamp(latest / 1000)
                    else:  # Seconds
                        start_dt = datetime.fromtimestamp(earliest)
                        end_dt = datetime.fromtimestamp(latest)
                else:
                    # String timestamps
                    start_dt = datetime.fromisoformat(str(earliest).replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(str(latest).replace('Z', '+00:00'))
                
                duration = (end_dt - start_dt).days
                
                datasets.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'record_count': count,
                    'duration_days': duration,
                    'date_range': f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}",
                    'start_date': start_dt.isoformat(),
                    'end_date': end_dt.isoformat()
                })
                
                print(f"✅ Processed {symbol} {timeframe}: {count:,} records, {duration} days")
                
            except Exception as e:
                print(f"⚠️ Error processing {symbol} {timeframe}: {e}")
                # Still add basic info even if date processing fails
                datasets.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'record_count': count,
                    'duration_days': 0,
                    'date_range': 'Unknown date range',
                    'start_date': str(earliest),
                    'end_date': str(latest)
                })
        
        conn.close()
        
        response_data = {
            "success": True,
            "datasets": datasets,
            "total_datasets": len(datasets),
            "total_records": total_records,
            "message": f"Found {len(datasets)} datasets with {total_records:,} total records"
        }
        
        print(f"📊 API Response: {len(datasets)} datasets, {total_records:,} total records")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "datasets": [],
            "total_datasets": 0,
            "total_records": 0
        }), 500

@app.route('/api/backtest/history')
def get_backtest_history():
    """API endpoint to get backtest results history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if backtest_results table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_results'")
        if not cursor.fetchone():
            print("⚠️ backtest_results table doesn't exist, creating sample data...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    starting_balance REAL DEFAULT 10000,
                    total_return_pct REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0,
                    status TEXT DEFAULT 'completed',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    trades_count INTEGER DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0
                )
            """)
            
            # Insert sample result
            cursor.execute("""
                INSERT INTO backtest_results 
                (pair, timeframe, starting_balance, total_return_pct, sharpe_ratio, status, trades_count, max_drawdown, win_rate)
                VALUES ('BTCUSDT', '15m', 10000, 15.5, 1.2, 'completed', 45, -8.3, 62.2)
            """)
            conn.commit()
        
        cursor.execute("""
            SELECT pair, timeframe, starting_balance, total_return_pct, sharpe_ratio, 
                   status, timestamp, trades_count, max_drawdown, win_rate
            FROM backtest_results 
            ORDER BY timestamp DESC 
            LIMIT 50
        """)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "pair": row[0],
                "timeframe": row[1], 
                "starting_balance": row[2],
                "total_return": row[3],
                "sharpe_ratio": row[4],
                "status": row[5],
                "timestamp": row[6],
                "trades_count": row[7],
                "max_drawdown": row[8],
                "win_rate": row[9]
            })
        
        conn.close()
        
        print(f"📊 Backtest API: Found {len(results)} results")
        return jsonify({
            "success": True,
            "results": results,
            "total_results": len(results)
        })
        
    except Exception as e:
        print(f"❌ Backtest API Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "results": [],
            "total_results": 0
        }), 500

@app.route('/api/historical-data/available-periods/<symbol>/<timeframe>')
def get_available_periods(symbol, timeframe):
    """Get available data periods for a specific symbol/timeframe"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM historical_data 
            WHERE symbol = ? AND timeframe = ?
        """, (symbol, timeframe))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] > 0:
            count, start_ts, end_ts = result
            
            # Convert timestamps
            if isinstance(start_ts, (int, float)):
                if start_ts > 1e10:
                    start_dt = datetime.fromtimestamp(start_ts / 1000)
                    end_dt = datetime.fromtimestamp(end_ts / 1000)
                else:
                    start_dt = datetime.fromtimestamp(start_ts)
                    end_dt = datetime.fromtimestamp(end_ts)
            else:
                start_dt = datetime.fromisoformat(str(start_ts))
                end_dt = datetime.fromisoformat(str(end_ts))
            
            periods = [{
                "name": f"Full Dataset ({symbol} {timeframe})",
                "start_date": start_dt.strftime('%Y-%m-%d'),
                "end_date": end_dt.strftime('%Y-%m-%d'),
                "record_count": count,
                "duration_days": (end_dt - start_dt).days
            }]
            
            return jsonify({
                "success": True,
                "periods": periods
            })
        else:
            return jsonify({
                "success": False,
                "periods": [],
                "message": f"No data found for {symbol} {timeframe}"
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "periods": []
        }), 500

if __name__ == '__main__':
    print("🎯 HISTORICAL DATA INTEGRATION TEST SERVER")
    print("=" * 50)
    print("🚀 Server: http://localhost:5000")
    print("📊 Dashboard: http://localhost:5000/")
    print("🔍 Data Discovery: http://localhost:5000/api/historical-data/discover")
    print("📈 Backtest History: http://localhost:5000/api/backtest/history")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)