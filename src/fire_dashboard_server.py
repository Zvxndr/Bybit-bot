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
        # Mock position data for now
        positions = []
        
        if environment == 'testnet' and balance_manager.balance_data['testnet']['status'] == 'active':
            positions = [
                {
                    'symbol': 'BTCUSDT',
                    'side': 'buy',
                    'size': '0.1',
                    'entryPrice': '65000',
                    'unrealizedPnl': 125.50,
                    'environment': environment
                }
            ]
        
        return jsonify({
            'success': True,
            'data': positions,
            'environment': environment
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
        # Mock trade data for now
        trades = []
        
        if environment in ['testnet', 'paper']:
            trades = [
                {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': 'BTCUSDT',
                    'side': 'buy',
                    'size': '0.05',
                    'price': '64500',
                    'pnl': 25.00,
                    'environment': environment
                }
            ]
        
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
    """Get system statistics"""
    try:
        stats = {
            'cpu': 15.2,  # Mock data for personal use
            'memory': 45.8,
            'apiStatus': 'Connected' if os.getenv('BYBIT_TESTNET_API_KEY') else 'Disconnected',
            'winRate': 75.5,
            'profitFactor': 1.8,
            'activeEnvironment': 'testnet',
            'timestamp': datetime.now().isoformat()
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