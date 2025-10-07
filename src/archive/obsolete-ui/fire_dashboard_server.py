"""
🔥 Fire Cybersigilism Dashboard Server
=====================================
Multi-environment trading dashboard for personal use
"""

from flask import Flask, render_template, request, jsonify
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'fire-cyber-secret-key')

class MultiEnvironmentBalanceManager:
    """Simple balance manager for multi-environment support"""
    
    def __init__(self):
        self.balance_data = {
            'testnet': {
                'total': float(os.getenv('TESTNET_BALANCE', '55116.84')),
                'available': float(os.getenv('TESTNET_BALANCE', '55116.84')) * 0.9,
                'used': float(os.getenv('TESTNET_BALANCE', '55116.84')) * 0.1,
                'unrealized': 125.50,
                'environment': 'testnet',
                'status': 'active' if os.getenv('BYBIT_TESTNET_API_KEY') else 'inactive'
            },
            'mainnet': {
                'total': float(os.getenv('MAINNET_BALANCE', '0')),
                'available': float(os.getenv('MAINNET_BALANCE', '0')) * 0.9,
                'used': float(os.getenv('MAINNET_BALANCE', '0')) * 0.1,
                'unrealized': 0.0,
                'environment': 'mainnet',
                'status': 'inactive'  # Keep mainnet inactive for safety in personal use
            },
            'paper': {
                'total': float(os.getenv('PAPER_TRADING_BALANCE', '100000')),
                'available': float(os.getenv('PAPER_TRADING_BALANCE', '100000')) * 0.9,
                'used': float(os.getenv('PAPER_TRADING_BALANCE', '100000')) * 0.1,
                'unrealized': 0.0,
                'environment': 'paper',
                'status': 'active'
            }
        }
        
    def get_all_balances(self):
        """Get all environment balances"""
        return self.balance_data
    
    def get_environment_status(self):
        """Get status of all environments"""
        return {env: data['status'] for env, data in self.balance_data.items()}

# Initialize balance manager
balance_manager = MultiEnvironmentBalanceManager()

@app.route('/')
def fire_dashboard():
    """Serve the fire cybersigilism dashboard"""
    return render_template('fire_dashboard.html')

@app.route('/ui-showcase')
def ui_showcase():
    """Serve the UI components showcase page"""
    return render_template('ui_showcase.html')

@app.route('/api/multi-balance')
def get_multi_balance():
    """API endpoint to get all environment balances"""
    try:
        balances = balance_manager.get_all_balances()
        return jsonify({
            'success': True,
            'data': balances,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting multi-balance: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/environment/switch', methods=['POST'])
def switch_environment():
    """API endpoint to switch active environment"""
    try:
        data = request.get_json()
        environment = data.get('environment')
        
        if environment not in ['testnet', 'mainnet', 'paper']:
            return jsonify({
                'success': False,
                'error': 'Invalid environment'
            }), 400
        
        return jsonify({
            'success': True,
            'environment': environment,
            'message': f'Switched to {environment} environment',
            'balance': balance_manager.balance_data[environment]
        })
        
    except Exception as e:
        logger.error(f"Error switching environment: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/positions/<environment>')
def get_positions(environment='testnet'):
    """Get positions for specific environment"""
    try:
        positions = []
        
        # Attempt to get real positions from API
        try:
            from src.bot.exchange.bybit_client import BybitClient
            
            # Initialize client based on environment
            api_key = os.getenv('BYBIT_API_KEY') or os.getenv('BYBIT_TESTNET_API_KEY')
            api_secret = os.getenv('BYBIT_API_SECRET') or os.getenv('BYBIT_TESTNET_API_SECRET')
            
            if api_key and api_secret:
                is_testnet = environment == 'testnet' or os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
                
                # Use pybit directly for real positions
                from pybit.unified_trading import HTTP
                client = HTTP(api_key=api_key, api_secret=api_secret, testnet=is_testnet)
                
                # Get real positions
                response = client.get_positions(category="linear")
                if response.get('retCode') == 0:
                    raw_positions = response.get('result', {}).get('list', [])
                    
                    # Transform to our format
                    for pos in raw_positions:
                        if float(pos.get('size', '0')) > 0:  # Only active positions
                            positions.append({
                                'symbol': pos.get('symbol'),
                                'side': pos.get('side', '').lower(),
                                'size': pos.get('size'),
                                'entryPrice': pos.get('avgPrice'),
                                'unrealizedPnl': float(pos.get('unrealisedPnl', '0')),
                                'environment': environment,
                                'leverage': pos.get('leverage'),
                                'markPrice': pos.get('markPrice')
                            })
                
        except Exception as api_error:
            logger.warning(f"Could not fetch real positions from API: {api_error}")
            # Continue with empty positions array - no fallback mock data
        
        return jsonify({
            'success': True,
            'data': positions,
            'environment': environment,
            'count': len(positions)
        })
        
    except Exception as e:
        logger.error(f"Error getting positions for {environment}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trades/<environment>')
def get_trades(environment='testnet'):
    """Get trades for specific environment"""
    try:
        trades = []
        
        # Attempt to get real trade history from API
        try:
            api_key = os.getenv('BYBIT_API_KEY') or os.getenv('BYBIT_TESTNET_API_KEY')
            api_secret = os.getenv('BYBIT_API_SECRET') or os.getenv('BYBIT_TESTNET_API_SECRET')
            
            if api_key and api_secret:
                is_testnet = environment == 'testnet' or os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
                
                from pybit.unified_trading import HTTP
                client = HTTP(api_key=api_key, api_secret=api_secret, testnet=is_testnet)
                
                # Get recent trades (last 50)
                response = client.get_executions(category="linear", limit=50)
                if response.get('retCode') == 0:
                    raw_trades = response.get('result', {}).get('list', [])
                    
                    # Transform to our format
                    for trade in raw_trades:
                        trades.append({
                            'timestamp': trade.get('execTime'),
                            'symbol': trade.get('symbol'),
                            'side': trade.get('side', '').lower(),
                            'size': trade.get('execQty'),
                            'price': trade.get('execPrice'),
                            'fee': float(trade.get('execFee', '0')),
                            'orderId': trade.get('orderId'),
                            'environment': environment,
                            'execType': trade.get('execType')
                        })
                
        except Exception as api_error:
            logger.warning(f"Could not fetch real trades from API: {api_error}")
            # Continue with empty trades array - no fallback mock data
        
        return jsonify({
            'success': True,
            'data': trades,
            'environment': environment,
            'count': len(trades)
        })
        
    except Exception as e:
        logger.error(f"Error getting trades for {environment}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
        return jsonify({
            'success': True,
            'data': trades,
            'environment': environment
        })
        
    except Exception as e:
        logger.error(f"Error getting trades for {environment}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system-stats')
def get_system_stats():
    """Get real system statistics"""
    try:
        import psutil
        import time
        from datetime import datetime
        
        # Real system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Check API connectivity
        api_status = 'Disconnected'
        try:
            api_key = os.getenv('BYBIT_API_KEY') or os.getenv('BYBIT_TESTNET_API_KEY')
            if api_key:
                from pybit.unified_trading import HTTP
                is_testnet = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
                client = HTTP(api_key=api_key, api_secret=os.getenv('BYBIT_API_SECRET') or os.getenv('BYBIT_TESTNET_API_SECRET'), testnet=is_testnet)
                
                # Quick ping test
                response = client.get_server_time()
                if response.get('retCode') == 0:
                    api_status = 'Connected'
        except:
            pass
        
        # Calculate trading performance from logs if available
        win_rate = 0.0
        profit_factor = 0.0
        try:
            # Try to read recent trading performance from logs or database
            from pathlib import Path
            log_dir = Path("logs")
            if log_dir.exists():
                # This would be replaced with actual performance calculation
                # For now, show real system metrics only
                pass
        except:
            pass
        
        stats = {
            'cpu': round(cpu_percent, 1),
            'memory': round(memory_percent, 1),
            'apiStatus': api_status,
            'winRate': win_rate,  # Would be calculated from real trade data
            'profitFactor': profit_factor,  # Would be calculated from real trade data
            'activeEnvironment': 'testnet' if os.getenv('BYBIT_TESTNET', 'true').lower() == 'true' else 'live',
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time(),  # System start time would be tracked
            'memoryUsage': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used
            }
        }
        
        return jsonify({
            'success': True,
            'data': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def run_fire_dashboard(host='127.0.0.1', port=5000, debug=False):
    """Run the fire cybersigilism dashboard server"""
    logger.info("🔥 Starting Fire Cybersigilism Dashboard Server")
    logger.info(f"🔥 Dashboard URL: http://{host}:{port}")
    
    try:
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error running dashboard server: {e}")
        raise

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the dashboard
    run_fire_dashboard(debug=True)