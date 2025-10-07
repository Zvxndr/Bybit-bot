"""
🔥 Multi-Environment Balance Manager
Handles balance tracking across testnet, mainnet, and paper trading environments
"""

import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger(__name__)

class MultiEnvironmentBalanceManager:
    def __init__(self):
        self.environments = {
            'testnet': {
                'api_key': os.getenv('BYBIT_TESTNET_API_KEY'),
                'api_secret': os.getenv('BYBIT_TESTNET_API_SECRET'),
                'base_url': 'https://api-testnet.bybit.com',
                'active': True
            },
            'mainnet': {
                'api_key': os.getenv('BYBIT_MAINNET_API_KEY'),
                'api_secret': os.getenv('BYBIT_MAINNET_API_SECRET'),
                'base_url': 'https://api.bybit.com',
                'active': False  # Start with testnet only for safety
            },
            'paper': {
                'initial_balance': float(os.getenv('PAPER_TRADING_BALANCE', '10000')),  # Use config base_balance
                'current_balance': float(os.getenv('PAPER_TRADING_BALANCE', '10000')),
                'trades': [],
                'active': True,
                'phase': 'Phase 2: Paper Trading/Testnet Validation'
            }
        }
        
        self.balance_cache = {}
        self.last_update = {}
        self.cache_duration = timedelta(seconds=30)  # Cache for 30 seconds
        
    async def get_all_balances(self) -> Dict[str, Dict[str, Any]]:
        """Get balances from all active environments"""
        balances = {}
        
        # Get testnet balance
        if self.environments['testnet']['active'] and self.environments['testnet']['api_key']:
            try:
                testnet_balance = await self._get_bybit_balance('testnet')
                balances['testnet'] = testnet_balance
            except Exception as e:
                logger.error(f"Failed to get testnet balance: {e}")
                balances['testnet'] = self._get_default_balance()
        
        # Get mainnet balance
        if self.environments['mainnet']['active'] and self.environments['mainnet']['api_key']:
            try:
                mainnet_balance = await self._get_bybit_balance('mainnet')
                balances['mainnet'] = mainnet_balance
            except Exception as e:
                logger.error(f"Failed to get mainnet balance: {e}")
                balances['mainnet'] = self._get_default_balance()
        
        # Get paper trading balance
        paper_balance = self._get_paper_balance()
        balances['paper'] = paper_balance
        
        return balances
    
    async def _get_bybit_balance(self, environment: str) -> Dict[str, Any]:
        """Get balance from Bybit API for specific environment"""
        cache_key = f"{environment}_balance"
        now = datetime.now()
        
        # Check cache first
        if (cache_key in self.balance_cache and 
            cache_key in self.last_update and 
            now - self.last_update[cache_key] < self.cache_duration):
            return self.balance_cache[cache_key]
        
        try:
            # Import here to avoid circular imports
            from src.bybit_api import BybitAPI
            
            env_config = self.environments[environment]
            api = BybitAPI(
                api_key=env_config['api_key'],
                api_secret=env_config['api_secret'],
                base_url=env_config['base_url']
            )
            
            # Get wallet balance
            balance_response = await api.get_wallet_balance()
            
            if balance_response.get('success'):
                wallet_data = balance_response.get('data', {}).get('list', [])
                
                total_balance = 0
                available_balance = 0
                used_balance = 0
                
                for account in wallet_data:
                    for coin in account.get('coin', []):
                        if coin.get('coin') == 'USDT':
                            wallet_balance = float(coin.get('walletBalance', 0))
                            available_balance += float(coin.get('availableToWithdraw', 0))
                            total_balance += wallet_balance
                
                used_balance = total_balance - available_balance
                
                # Get position PnL
                unrealized_pnl = await self._get_unrealized_pnl(api)
                
                balance_data = {
                    'total': round(total_balance, 2),
                    'available': round(available_balance, 2),
                    'used': round(used_balance, 2),
                    'unrealized': round(unrealized_pnl, 2),
                    'last_updated': now.isoformat(),
                    'environment': environment,
                    'status': 'connected'
                }
                
                # Cache the result
                self.balance_cache[cache_key] = balance_data
                self.last_update[cache_key] = now
                
                return balance_data
            
            else:
                raise Exception(f"API request failed: {balance_response}")
                
        except Exception as e:
            logger.error(f"Error getting {environment} balance: {e}")
            return {
                'total': 0,
                'available': 0,
                'used': 0,
                'unrealized': 0,
                'last_updated': now.isoformat(),
                'environment': environment,
                'status': 'error',
                'error': str(e)
            }
    
    async def _get_unrealized_pnl(self, api) -> float:
        """Get unrealized PnL from open positions"""
        try:
            positions_response = await api.get_positions()
            
            if positions_response.get('success'):
                positions = positions_response.get('data', {}).get('list', [])
                total_pnl = 0
                
                for position in positions:
                    if float(position.get('size', 0)) > 0:  # Open position
                        pnl = float(position.get('unrealisedPnl', 0))
                        total_pnl += pnl
                
                return total_pnl
            
        except Exception as e:
            logger.error(f"Error getting unrealized PnL: {e}")
        
        return 0.0
    
    def _get_paper_balance(self) -> Dict[str, Any]:
        """Get paper trading balance"""
        paper_config = self.environments['paper']
        
        # Calculate paper trading metrics
        total_trades = len(paper_config.get('trades', []))
        winning_trades = len([t for t in paper_config.get('trades', []) if t.get('pnl', 0) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        current_balance = paper_config.get('current_balance', 100000)
        initial_balance = paper_config.get('initial_balance', 100000)
        total_pnl = current_balance - initial_balance
        
        return {
            'total': round(current_balance, 2),
            'available': round(current_balance * 0.9, 2),  # Assume 10% margin requirement
            'used': round(current_balance * 0.1, 2),
            'unrealized': 0.0,  # Paper trading doesn't have unrealized PnL
            'total_pnl': round(total_pnl, 2),
            'total_trades': total_trades,
            'win_rate': round(win_rate, 1),
            'last_updated': datetime.now().isoformat(),
            'environment': 'paper',
            'status': 'active'
        }
    
    def _get_default_balance(self) -> Dict[str, Any]:
        """Get default balance structure for inactive/error environments"""
        return {
            'total': 0,
            'available': 0,
            'used': 0,
            'unrealized': 0,
            'last_updated': datetime.now().isoformat(),
            'environment': 'unknown',
            'status': 'inactive'
        }
    
    def update_paper_balance(self, trade_data: Dict[str, Any]):
        """Update paper trading balance after a trade"""
        try:
            paper_config = self.environments['paper']
            
            # Add trade to history
            if 'trades' not in paper_config:
                paper_config['trades'] = []
            
            paper_config['trades'].append({
                'timestamp': datetime.now().isoformat(),
                'symbol': trade_data.get('symbol'),
                'side': trade_data.get('side'),
                'size': trade_data.get('size'),
                'price': trade_data.get('price'),
                'pnl': trade_data.get('pnl', 0)
            })
            
            # Update balance
            pnl = trade_data.get('pnl', 0)
            paper_config['current_balance'] += pnl
            
            logger.info(f"Paper trading balance updated: +${pnl}, Total: ${paper_config['current_balance']}")
            
        except Exception as e:
            logger.error(f"Error updating paper balance: {e}")
    
    def set_environment_active(self, environment: str, active: bool):
        """Enable/disable trading for specific environment"""
        if environment in self.environments:
            self.environments[environment]['active'] = active
            logger.info(f"Environment {environment} set to {'active' if active else 'inactive'}")
    
    def get_environment_status(self) -> Dict[str, bool]:
        """Get status of all environments"""
        return {env: config.get('active', False) for env, config in self.environments.items()}

# Global instance
balance_manager = MultiEnvironmentBalanceManager()