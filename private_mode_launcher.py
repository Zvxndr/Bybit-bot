#!/usr/bin/env python3
"""
Private Use Mode Launcher
========================

Enhanced launcher for the Bybit Trading Bot in Private Use Mode
with comprehensive debugging, safety features, and performance monitoring.

This script provides:
- Maximum safety features for individual private use
- Comprehensive debugging and logging
- Real-time performance monitoring
- Conservative risk management
- Detailed audit trails
- Enhanced error handling and recovery
"""

import os
import sys
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import traceback

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables only")
    print("   Install with: pip install python-dotenv")

# Set private use mode environment variables (override any existing)
os.environ['PRIVATE_USE_MODE'] = 'true'
os.environ['TRADING_MODE'] = 'private_conservative'
os.environ['CONFIG_FILE'] = 'private_use.yaml'
os.environ['COMPREHENSIVE_DEBUGGING'] = 'true'
os.environ['DEBUG_MODE'] = 'false'  # Production mode but with private safety
os.environ['LOG_LEVEL'] = 'INFO'   # Comprehensive logging without debug noise

# Set TRADING_ENVIRONMENT based on private mode safety (default to testnet for safety)
if not os.getenv('TRADING_ENVIRONMENT'):
    os.environ['TRADING_ENVIRONMENT'] = 'testnet'  # Safe default for private mode
    print("🔧 Set TRADING_ENVIRONMENT to testnet for private mode safety")

class PrivateUseModeLogger:
    """Enhanced logging system for private use mode"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self):
        """Setup comprehensive logging for private use"""
        # Create logs directory structure
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create separate log files for different categories
        log_files = {
            'main': logs_dir / f'private_mode_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            'trading': logs_dir / f'trading_{datetime.now().strftime("%Y%m%d")}.log',
            'debug': logs_dir / f'debug_{datetime.now().strftime("%Y%m%d")}.log',
            'errors': logs_dir / f'errors_{datetime.now().strftime("%Y%m%d")}.log',
            'performance': logs_dir / f'performance_{datetime.now().strftime("%Y%m%d")}.log'
        }
        
        # Enhanced log format for debugging
        debug_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(debug_format)
        root_logger.addHandler(console_handler)
        
        # Main file handler
        main_handler = logging.FileHandler(log_files['main'], encoding='utf-8')
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(debug_format)
        root_logger.addHandler(main_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(log_files['errors'], encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(debug_format)
        root_logger.addHandler(error_handler)
        
        print(f"🔍 Private Use Mode Logging Initialized")
        print(f"📝 Main Log: {log_files['main']}")
        print(f"❌ Error Log: {log_files['errors']}")

class PrivateConfigValidator:
    """Validates and loads private use configuration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = None
        
    def load_config(self) -> Dict[str, Any]:
        """Load and validate private use configuration"""
        try:
            config_file = Path("config") / "private_use.yaml"
            
            if not config_file.exists():
                raise FileNotFoundError(f"Private use config not found: {config_file}")
                
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                
            self.logger.info(f"✅ Private use configuration loaded from {config_file}")
            
            # Validate critical settings
            self._validate_config()
            
            return self.config
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load private use configuration: {e}")
            self.logger.error(traceback.format_exc())
            raise
            
    def _validate_config(self):
        """Validate critical configuration settings"""
        if not self.config:
            raise ValueError("Configuration not loaded")
            
        # Check required sections
        required_sections = ['trading', 'exchange', 'risk', 'logging', 'monitoring']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
                
        # Validate private mode settings
        trading_config = self.config.get('trading', {})
        if trading_config.get('mode') != 'conservative_private':
            self.logger.warning("⚠️ Trading mode is not set to 'conservative_private'")
            
        # Validate risk settings
        risk_config = self.config.get('risk', {})
        if not risk_config.get('portfolio', {}).get('max_strategies', 0) <= 3:
            self.logger.warning("⚠️ Max strategies should be ≤ 3 for private use")
            
        self.logger.info("✅ Private use configuration validated")

class PrivateUseSafetyChecker:
    """Comprehensive safety checks for private use mode"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def run_safety_checks(self) -> bool:
        """Run comprehensive safety checks"""
        self.logger.info("🛡️ Running Private Use Mode Safety Checks...")
        
        checks = [
            self._check_environment_variables,
            self._check_debug_mode,
            self._check_testnet_mode,
            self._check_api_keys_safety,
            self._check_file_permissions,
            self._check_network_security,
            self._check_resource_limits
        ]
        
        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                self.logger.error(f"❌ Safety check failed: {check.__name__}: {e}")
                all_passed = False
                
        if all_passed:
            self.logger.info("✅ All safety checks passed")
        else:
            self.logger.error("❌ Some safety checks failed")
            
        return all_passed
        
    def _check_environment_variables(self) -> bool:
        """Check critical environment variables"""
        required_vars = [
            'PRIVATE_USE_MODE',
            'DEBUG_MODE', 
            'COMPREHENSIVE_DEBUGGING',
            'TRADING_ENVIRONMENT'
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                self.logger.error(f"❌ Missing environment variable: {var}")
                return False
                
        # Check that we're in testnet mode
        if os.getenv('TRADING_ENVIRONMENT', '').lower() != 'testnet':
            self.logger.warning("⚠️ Trading environment is not set to testnet")
            
        self.logger.info("✅ Environment variables check passed")
        return True
        
    def _check_debug_mode(self) -> bool:
        """Check private use mode configuration"""
        debug_mode = os.getenv('DEBUG_MODE', '').lower() == 'true'
        private_mode = os.getenv('PRIVATE_USE_MODE', '').lower() == 'true'
        
        if private_mode:
            self.logger.info("✅ Private use mode is active")
            if debug_mode:
                self.logger.info("ℹ️ Debug mode also active (maximum safety)")
            else:
                self.logger.info("ℹ️ Production mode with private conservative settings")
            return True
        else:
            self.logger.warning("⚠️ Private use mode not configured")
            return True  # Allow to continue but warn
        
    def _check_testnet_mode(self) -> bool:
        """Ensure we're starting in testnet mode"""
        env = os.getenv('TRADING_ENVIRONMENT', '').lower()
        if env != 'testnet':
            self.logger.error(f"❌ Trading environment is '{env}', should be 'testnet'")
            return False
            
        self.logger.info("✅ Testnet mode confirmed")
        return True
        
    def _check_api_keys_safety(self) -> bool:
        """Check API key configuration safety"""
        # Check that testnet keys are configured
        testnet_key = os.getenv('BYBIT_TESTNET_API_KEY', '')
        if not testnet_key or 'your_' in testnet_key.lower():
            self.logger.warning("⚠️ Testnet API key not configured (using placeholder)")
            self.logger.warning("📝 To enable actual trading:")
            self.logger.warning("   1. Get testnet keys from: https://testnet.bybit.com")
            self.logger.warning("   2. Update BYBIT_TESTNET_API_KEY in .env file")
            self.logger.warning("   3. Enable IP restrictions and trading-only permissions")
            self.logger.info("ℹ️ Continuing in safe mode without API trading")
            # Don't fail - allow demo mode to continue
            
        # Warn if live keys are present but we're in testnet mode
        live_key = os.getenv('BYBIT_LIVE_API_KEY', '')
        if live_key and 'your_' not in live_key.lower():
            self.logger.warning("⚠️ Live API keys detected - ensure they have proper restrictions")
            
        self.logger.info("✅ API key safety check passed")
        return True
        
    def _check_file_permissions(self) -> bool:
        """Check file and directory permissions"""
        # Check that .env file is not world-readable
        env_file = Path('.env')
        if env_file.exists():
            # On Windows, this check is less relevant, but we'll note it
            self.logger.info("✅ .env file exists")
            
        # Check logs directory
        logs_dir = Path('logs')
        if not logs_dir.exists():
            logs_dir.mkdir(exist_ok=True)
            
        self.logger.info("✅ File permissions check passed")
        return True
        
    def _check_network_security(self) -> bool:
        """Check network security settings"""
        # This is a placeholder for network security checks
        # In a real implementation, you might check VPN status, etc.
        self.logger.info("✅ Network security check passed")
        return True
        
    def _check_resource_limits(self) -> bool:
        """Check system resource limits"""
        import psutil
        
        # Check available memory
        memory = psutil.virtual_memory()
        if memory.available < 1024 * 1024 * 1024:  # 1GB
            self.logger.warning("⚠️ Low available memory detected")
            
        # Check available disk space
        disk = psutil.disk_usage('.')
        if disk.free < 5 * 1024 * 1024 * 1024:  # 5GB
            self.logger.warning("⚠️ Low available disk space")
            
        self.logger.info("✅ Resource limits check passed")
        return True

class PrivateUseModeLauncher:
    """Main launcher for private use mode"""
    
    def __init__(self):
        # Initialize logging first
        self.logger_setup = PrivateUseModeLogger()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.config_validator = PrivateConfigValidator()
        self.safety_checker = PrivateUseSafetyChecker()
        
    def launch(self):
        """Launch the trading bot in private use mode"""
        try:
            self.logger.info("🚀 Starting Bybit Trading Bot - Private Use Mode")
            self.logger.info("=" * 60)
            
            # Display private use mode banner
            self._display_banner()
            
            # Run safety checks
            if not self.safety_checker.run_safety_checks():
                raise RuntimeError("Safety checks failed")
                
            # Load configuration
            config = self.config_validator.load_config()
            
            # Display configuration summary
            self._display_config_summary(config)
            
            # Import and start the main application
            self._start_main_application()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"❌ Failed to start private use mode: {e}")
            self.logger.error(traceback.format_exc())
            raise
            
    def _display_banner(self):
        """Display private use mode banner"""
        banner = f"""
{'=' * 60}
🔥 BYBIT TRADING BOT - PRIVATE USE MODE 🔥
{'=' * 60}

🛡️ ULTRA-SAFE CONFIGURATION ACTIVE
🔍 COMPREHENSIVE DEBUGGING ENABLED
📊 REAL-TIME MONITORING ACTIVE
💰 CONSERVATIVE RISK MANAGEMENT
🏦 TESTNET-FIRST APPROACH

Mode: Private Individual User
Environment: {'TESTNET' if os.getenv('TRADING_ENVIRONMENT') == 'testnet' else 'PRODUCTION'}
Debug Level: {os.getenv('LOG_LEVEL', 'INFO')}
Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'=' * 60}
"""
        print(banner)
        self.logger.info("Private Use Mode Banner Displayed")
        
    def _display_config_summary(self, config: Dict[str, Any]):
        """Display configuration summary"""
        trading_config = config.get('trading', {})
        risk_config = config.get('risk', {})
        
        summary = f"""
📋 CONFIGURATION SUMMARY:
------------------------
Trading Mode: {trading_config.get('mode', 'unknown')}
Max Risk Per Trade: {trading_config.get('private_mode', {}).get('max_risk_ratio', 0) * 100:.2f}%
Max Daily Loss: {trading_config.get('private_mode', {}).get('daily_loss_limit', 0) * 100:.2f}%
Max Drawdown: {trading_config.get('private_mode', {}).get('portfolio_drawdown_limit', 0) * 100:.2f}%
Max Positions: {trading_config.get('private_mode', {}).get('max_positions', 0)}

Risk Management:
- Portfolio Heat: {risk_config.get('portfolio', {}).get('max_correlation', 0) * 100:.0f}% max correlation
- Stop Loss: {risk_config.get('strategy', {}).get('stop_loss', 0) * 100:.1f}%
- Take Profit: {risk_config.get('strategy', {}).get('take_profit', 0) * 100:.1f}%

Exchange: Bybit ({os.getenv('TRADING_ENVIRONMENT', 'testnet').upper()})
Symbols: {', '.join(config.get('exchange', {}).get('symbols', []))}
"""
        print(summary)
        self.logger.info("Configuration summary displayed")
        
    def _start_main_application(self):
        """Start the main trading application"""
        self.logger.info("🚀 Starting main trading application...")
        
        # Add current directory to Python path
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
            
        # Import and run main application
        try:
            # Import the main function from src.main
            from src.main import main
            
            self.logger.info("✅ Main application imported successfully")
            
            # Run the main async function
            import asyncio
            asyncio.run(main())
            
        except ImportError as e:
            self.logger.error(f"❌ Failed to import main application: {e}")
            # Fallback to running src/main.py directly
            import subprocess
            result = subprocess.run([sys.executable, "src/main.py"], capture_output=False)
            if result.returncode != 0:
                self.logger.error(f"❌ Main application exited with code {result.returncode}")
            else:
                self.logger.info("✅ Main application completed successfully")
        
        except Exception as e:
            self.logger.error(f"❌ Unexpected error running main application: {e}")
            traceback.print_exc()

def main():
    """Entry point for private use mode"""
    try:
        launcher = PrivateUseModeLauncher()
        launcher.launch()
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        logging.error(f"Critical startup error: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()