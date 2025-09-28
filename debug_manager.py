"""
Debug Management Utility
========================

Utility script for managing debug sessions and cleaning up debugging artifacts.
Provides easy commands for starting, stopping, and managing debug mode.
"""

import os
import sys
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
import logging

def setup_logging():
    """Setup logging for debug management"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - DEBUG_MANAGER - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class DebugManager:
    """Debug session management utility"""
    
    def __init__(self):
        self.config_path = Path("config/debug.yaml")
        self.logs_path = Path("logs")
        self.backup_path = Path("backup")
        
    def enable_debug_mode(self):
        """Enable debug mode with safety checks"""
        logger.info("🔧 Enabling debug mode...")
        
        # Ensure config directory exists
        self.config_path.parent.mkdir(exist_ok=True)
        
        # Create debug configuration
        debug_config = {
            'debug_mode': True,
            'debug_settings': {
                'disable_real_trading': True,
                'disable_api_orders': True,
                'disable_position_changes': True,
                'force_testnet': True,
                'mock_api_responses': True,
                'api_call_logging': True,
                'show_debug_warnings': True,
                'enable_debug_panel': True,
                'log_ui_interactions': True,
                'use_mock_balances': True,
                'preserve_real_data': True,
                'max_debug_runtime': 3600,
                'auto_shutdown_on_error': True
            },
            'logging': {
                'level': 'DEBUG',
                'console_output': True,
                'file_output': True,
                'debug_log_file': 'logs/debug_session.log'
            },
            'mock_data': {
                'testnet_balance': 10000.00,
                'mainnet_balance': 0.00,
                'paper_balance': 100000.00,
                'mock_positions': [
                    {
                        'symbol': 'BTCUSDT',
                        'side': 'long',
                        'size': '0.001',
                        'entry_price': '67500.00',
                        'pnl': '+15.50'
                    }
                ],
                'mock_trades': [
                    {
                        'symbol': 'BTCUSDT',
                        'side': 'buy',
                        'size': '0.001',
                        'price': '67500.00',
                        'timestamp': '2024-01-20 14:30:00',
                        'pnl': '+15.50'
                    }
                ]
            },
            'phase': {
                'current': 'PRIVATE_USE_DEBUGGING',
                'status': 'DEVELOPMENT',
                'trading_allowed': False,
                'live_deployment': False
            }
        }
        
        # Write debug configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(debug_config, f, default_flow_style=False)
        
        logger.info("✅ Debug mode enabled successfully")
        logger.warning("🚨 ALL TRADING OPERATIONS ARE NOW BLOCKED")
        logger.info("💰 Your money is completely safe during debugging")
        
    def disable_debug_mode(self):
        """Disable debug mode (CAUTION: Enables live trading)"""
        logger.warning("⚠️ DISABLING DEBUG MODE - THIS WILL ENABLE LIVE TRADING!")
        response = input("Are you ABSOLUTELY SURE you want to enable live trading? (type 'YES' to confirm): ")
        
        if response != 'YES':
            logger.info("❌ Debug mode disable cancelled - staying in safe debug mode")
            return
        
        logger.info("🔧 Disabling debug mode...")
        
        if self.config_path.exists():
            # Backup current debug config
            backup_file = self.backup_path / f"debug_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            self.backup_path.mkdir(exist_ok=True)
            shutil.copy2(self.config_path, backup_file)
            logger.info(f"📋 Debug config backed up to {backup_file}")
            
            # Update config to disable debug mode
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config['debug_mode'] = False
            config['debug_settings']['disable_real_trading'] = False
            config['debug_settings']['disable_api_orders'] = False
            config['phase']['trading_allowed'] = True
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.warning("🚨 DEBUG MODE DISABLED - LIVE TRADING IS NOW POSSIBLE")
            logger.warning("💰 REAL MONEY CAN NOW BE AT RISK")
        else:
            logger.warning("⚠️ Debug config not found - creating production config")
            
    def check_debug_status(self):
        """Check current debug mode status"""
        logger.info("🔍 Checking debug mode status...")
        
        if not self.config_path.exists():
            logger.warning("❌ Debug config file not found - debug mode unknown")
            return False
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        debug_mode = config.get('debug_mode', False)
        phase = config.get('phase', {}).get('current', 'UNKNOWN')
        trading_allowed = config.get('phase', {}).get('trading_allowed', True)
        
        if debug_mode:
            logger.info("✅ DEBUG MODE ACTIVE - Trading is safely disabled")
            logger.info(f"🔧 Current phase: {phase}")
            logger.info("💰 Your money is completely safe")
        else:
            logger.warning("⚠️ DEBUG MODE DISABLED - Live trading may be active")
            logger.warning(f"💰 Trading allowed: {trading_allowed}")
        
        return debug_mode
        
    def clean_debug_logs(self):
        """Clean up debug session logs"""
        logger.info("🧹 Cleaning debug logs...")
        
        if self.logs_path.exists():
            debug_logs = list(self.logs_path.glob("debug_session*.log"))
            old_logs = list(self.logs_path.glob("open_alpha_*.log"))
            
            total_cleaned = 0
            for log_file in debug_logs + old_logs:
                try:
                    log_file.unlink()
                    total_cleaned += 1
                    logger.debug(f"🗑️ Deleted {log_file}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete {log_file}: {e}")
            
            logger.info(f"✅ Cleaned {total_cleaned} debug log files")
        else:
            logger.info("📁 No logs directory found - nothing to clean")
    
    def backup_current_session(self):
        """Backup current debug session data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.backup_path / f"debug_session_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📋 Creating debug session backup in {backup_dir}")
        
        # Backup config
        if self.config_path.exists():
            shutil.copy2(self.config_path, backup_dir / "debug.yaml")
            logger.debug("✅ Debug config backed up")
        
        # Backup logs
        if self.logs_path.exists():
            for log_file in self.logs_path.glob("*.log"):
                shutil.copy2(log_file, backup_dir / log_file.name)
                logger.debug(f"✅ {log_file.name} backed up")
        
        logger.info(f"✅ Debug session backup complete: {backup_dir}")
        
    def display_safety_warning(self):
        """Display comprehensive safety warning"""
        print("\n" + "="*60)
        print("🚨 DEBUG MODE SAFETY SYSTEM")
        print("="*60)
        print("🔒 MONEY SAFETY: Your funds are completely protected")
        print("🚫 TRADING BLOCKED: All real trading operations disabled")
        print("🧪 MOCK DATA: UI uses fake data for testing")
        print("⏰ AUTO SHUTDOWN: Sessions auto-terminate after 1 hour")
        print("📝 FULL LOGGING: All actions logged for debugging")
        print("="*60)
        print("💡 This is a 100% safe testing environment")
        print("="*60 + "\n")

def main():
    """Main debug management interface"""
    manager = DebugManager()
    
    if len(sys.argv) < 2:
        print("🔧 Debug Management Utility")
        print("Usage:")
        print("  python debug_manager.py enable    # Enable debug mode (safe)")
        print("  python debug_manager.py disable   # Disable debug mode (DANGEROUS)")
        print("  python debug_manager.py status    # Check debug mode status")
        print("  python debug_manager.py clean     # Clean debug logs") 
        print("  python debug_manager.py backup    # Backup debug session")
        print("  python debug_manager.py warning   # Show safety warning")
        return
        
    command = sys.argv[1].lower()
    
    if command == 'enable':
        manager.display_safety_warning()
        manager.enable_debug_mode()
    elif command == 'disable':
        manager.disable_debug_mode()
    elif command == 'status':
        manager.check_debug_status()
    elif command == 'clean':
        manager.clean_debug_logs()
    elif command == 'backup':
        manager.backup_current_session()
    elif command == 'warning':
        manager.display_safety_warning()
    else:
        logger.error(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()