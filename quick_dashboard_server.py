#!/usr/bin/env python3
"""
🎯 Quick Dashboard Server Launcher
Minimal server to test the historical data integration fix
"""

import sys
import os
from pathlib import Path
import sqlite3
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from flask import Flask, jsonify, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("Installing required packages...")
    os.system("pip install flask flask-cors")
    from flask import Flask, jsonify, send_from_directory
    from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Database path
DB_PATH = "data/trading_bot.db"

@app.route('/')
def index():
    """Serve the main dashboard"""
    return send_from_directory('frontend', 'unified_dashboard.html')

@app.route('/api/historical-data/discover')
def discover_historical_data():
    """API endpoint to discover available historical data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get available datasets
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
        
        for row in results:
            symbol, timeframe, count, earliest, latest = row
            
            try:
                # Handle timestamp conversion
                if isinstance(earliest, (int, float)):
                    if earliest > 1e10:
                        start_dt = datetime.fromtimestamp(earliest / 1000)
                        end_dt = datetime.fromtimestamp(latest / 1000)
                    else:
                        start_dt = datetime.fromtimestamp(earliest)
                        end_dt = datetime.fromtimestamp(latest)
                else:
                    start_dt = datetime.fromisoformat(str(earliest))
                    end_dt = datetime.fromisoformat(str(latest))
                
                duration = (end_dt - start_dt).days
                
                datasets.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'record_count': count,
                    'duration_days': duration,
                    'date_range': f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
                })
                
            except Exception as e:
                print(f"Error processing {symbol} {timeframe}: {e}")
                continue
        
        conn.close()
        
        return jsonify({
            "success": True,
            "datasets": datasets,
            "total_datasets": len(datasets)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/backtest/history')
def get_backtest_history():
    """API endpoint to get backtest results history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
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
        
        return jsonify({
            "success": True,
            "results": results,
            "total_results": len(results)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("🎯 DASHBOARD SERVER - Historical Data Integration Test")
    print("=" * 55)
    print("🚀 Starting server on http://localhost:5000")
    print("📊 Available endpoints:")
    print("   • /                                 - Main Dashboard") 
    print("   • /api/historical-data/discover     - Data Discovery")
    print("   • /api/backtest/history            - Backtest Results")
    print("=" * 55)
    
    # Verify database exists
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        sys.exit(1)
    else:
        print(f"✅ Database found: {DB_PATH}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)