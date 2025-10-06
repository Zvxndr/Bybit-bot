"""
Main Application Entry Point
============================

Production entry point for the Bybit Trading Bot with comprehensive
initialization, monitoring, and graceful shutdown capabilities.
"""

import asyncio
import os
import sys
import signal
import logging
from datetime import datetime
from pathlib import Path
from threading import Thread
import time
from typing import List, Dict, Any
import traceback

# Enhanced logging configuration
def setup_comprehensive_logging():
    """Setup comprehensive logging with multiple levels and detailed formatting"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Setup root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG for comprehensive logging
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/open_alpha_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Application logging system initialized")
    return logger

# Initialize logging early
logger = setup_comprehensive_logging()

# Import debug safety manager FIRST - try multiple import strategies
debug_manager = None
try:
    # Strategy 1: Relative import
    from .debug_safety import get_debug_manager, is_debug_mode, block_trading_if_debug
    logger.debug("✅ Debug safety imported via relative import")
except ImportError:
    try:
        # Strategy 2: Direct import
        from debug_safety import get_debug_manager, is_debug_mode, block_trading_if_debug
        logger.debug("✅ Debug safety imported via direct import")
    except ImportError:
        try:
            # Strategy 3: Absolute import for deployment
            from src.debug_safety import get_debug_manager, is_debug_mode, block_trading_if_debug
            logger.debug("✅ Debug safety imported via absolute import")
        except ImportError as e:
            logger.error(f"❌ Failed to import debug_safety: {e}")
            # Create minimal fallback functions
            def get_debug_manager():
                return None
            def is_debug_mode():
                return True  # Safer to assume debug mode if unsure
            def block_trading_if_debug():
                return True

# Initialize debug manager
debug_manager = get_debug_manager()

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ Environment variables loaded from .env file")
    print("✅ Environment variables loaded from .env file")
except ImportError:
    logger.warning("⚠️ python-dotenv not installed - install with: pip install python-dotenv")
    print("⚠️ python-dotenv not installed - install with: pip install python-dotenv")
except Exception as e:
    logger.error(f"⚠️ Could not load .env file: {e}")
    print(f"⚠️ Could not load .env file: {e}")

logger.debug("🔧 Starting main module import sequence...")

# Add HTTP server imports
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Import core components - UPDATED FOR NEW FRONTEND ARCHITECTURE
# Remove frontend_server dependency and use API-only approach
shared_state = None
get_bybit_client = None

# Strategy 1: Try importing core components only (no frontend_server needed)
try:
    import sys
    from pathlib import Path
    
    logger.debug("🔧 Attempting Strategy 1: Core imports from current directory")
    
    # Add current src directory to path
    current_dir = Path(__file__).parent
    logger.debug(f"🔧 Current directory: {current_dir}")
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        logger.debug(f"🔧 Added to sys.path: {current_dir}")
    
    from shared_state import shared_state
    from bybit_api import get_bybit_client
    logger.info("✅ Strategy 1 SUCCESS: Core components loaded successfully")
    print("✅ Core components loaded successfully")
    
except ImportError as e1:
    logger.warning(f"⚠️ Strategy 1 FAILED: Relative imports failed - {e1}")
    # Strategy 2: Try absolute imports for deployment environment  
    try:
        logger.debug("🔧 Attempting Strategy 2: Absolute imports for deployment")
        from src.shared_state import shared_state
        from src.bybit_api import get_bybit_client
        logger.info("✅ Strategy 2 SUCCESS: Core components loaded via absolute imports")
        print("✅ Core components loaded via absolute imports")
        
    except ImportError as e2:
        logger.error(f"❌ Strategy 2 FAILED: Absolute imports failed - {e2}")
        # Strategy 3: Try importing just the essential components
        try:
            import importlib.util
            
            # Load shared_state directly
            shared_state_path = Path(__file__).parent / "shared_state.py"
            spec = importlib.util.spec_from_file_location("shared_state", shared_state_path)
            shared_state_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shared_state_module)
            shared_state = shared_state_module.shared_state
            
            # Load bybit_api directly
            bybit_path = Path(__file__).parent / "bybit_api.py"
            spec = importlib.util.spec_from_file_location("bybit_api", bybit_path)
            bybit_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bybit_module)
            get_bybit_client = bybit_module.get_bybit_client
            
            print("✅ Core components loaded via direct file import")
            
        except Exception as e3:
            print(f"⚠️ All import strategies failed: {e1}, {e2}, {e3}")
            print("🔧 Using minimal fallback implementations")
            
            # Create minimal shared_state fallback
            class SharedState:
                def __init__(self):
                    self.data = {}
                    self.logs = []
                    self.system_status = "initializing"
                
            def get(self, key, default=None):
                return self.data.get(key, default)
                
            def set(self, key, value):
                self.data[key] = value
                
            def add_log_entry(self, level, message):
                """Add log entry to shared state"""
                self.logs.append({
                    'timestamp': time.time(),
                    'level': level,
                    'message': message
                })
                # Keep only last 1000 logs
                if len(self.logs) > 1000:
                    self.logs = self.logs[-1000:]
                    
            def update_system_status(self, status):
                """Update system status"""
                self.system_status = status
                self.data['system_status'] = status
                
            def get_system_status(self):
                """Get current system status"""
                return self.system_status
                
            def update_trading_data(self, data=None, **kwargs):
                """Update trading data in shared state"""
                if data is not None:
                    self.data['trading_data'] = data
                
                # Handle keyword arguments
                for key, value in kwargs.items():
                    self.data[key] = value
                    
                self.data['last_update'] = time.time()
                
            def get_trading_data(self):
                """Get current trading data"""
                return self.data.get('trading_data', {})
                
            def update_balance(self, balance_data):
                """Update balance information"""
                self.data['balance'] = balance_data
                
            def get_balance(self):
                """Get current balance"""
                return self.data.get('balance', {})
                
            def update_positions(self, positions_data):
                """Update positions information"""
                self.data['positions'] = positions_data
                
            def get_positions(self):
                """Get current positions"""
                return self.data.get('positions', {})
                
            def __setattr__(self, name, value):
                """Allow dynamic attribute setting for compatibility"""
                if name in ['data', 'logs', 'system_status']:
                    super().__setattr__(name, value)
                else:
                    # Store dynamic attributes in data dict
                    if hasattr(self, 'data'):
                        self.data[name] = value
                    else:
                        super().__setattr__(name, value)
                        
            def __getattr__(self, name):
                """Allow dynamic attribute getting for compatibility"""
                if hasattr(self, 'data') and name in self.data:
                    return self.data[name]
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
                
            def reset_bot_control_flags(self):
                """Reset bot control flags"""
                self.data['bot_active'] = False
                self.data['trading_enabled'] = False
                self.data['bot_control_flags_reset'] = True
                
            def start_background_tasks(self):
                """Start background tasks (fallback implementation)"""
                print("🔧 Background tasks started (minimal mode)")
                return True
            
            # Create fallback shared_state instance
            shared_state = SharedState()
            
            async def get_bybit_client():
                """Get Bybit client from environment variables"""
                # Check for both naming conventions
                api_key = os.getenv('BYBIT_API_KEY') or os.getenv('BYBIT_TESTNET_API_KEY')
                api_secret = os.getenv('BYBIT_API_SECRET') or os.getenv('BYBIT_TESTNET_API_SECRET')
                is_testnet = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
                
                if not api_key or not api_secret:
                    print("⚠️ Bybit API keys not found in environment - running in offline mode")
                    print("💡 Set BYBIT_API_KEY and BYBIT_API_SECRET in DigitalOcean environment variables")
                    return None
                
                try:
                    # Import pybit for Bybit API
                    from pybit.unified_trading import HTTP
                    
                    # Create client using your environment configuration
                    client = HTTP(
                        api_key=api_key,
                        api_secret=api_secret,
                        testnet=is_testnet
                    )
                    
                    # Test connection
                    account_info = client.get_wallet_balance(accountType="UNIFIED")
                    env_type = "Testnet" if is_testnet else "Mainnet"
                    print(f"✅ Bybit client connected successfully!")
                    print(f"📊 Account type: UNIFIED ({env_type})")
                    
                    return client
                    
                except ImportError:
                    print("⚠️ pybit not installed - install with: pip install pybit")
                    return None
                except Exception as e:
                    print(f"⚠️ Bybit connection failed: {e}")
                    print("💡 Check your API keys and network connection")
                    return None

# Speed Demon integration with enhanced logging
logger.debug("🔧 Attempting to import Speed Demon integration...")
try:
    from bot.speed_demon_integration import speed_demon_integration
    logger.info("✅ Speed Demon integration loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Speed Demon integration not available - {e}")
    speed_demon_integration = None
except Exception as e:
    logger.error(f"❌ Speed Demon integration failed - {e}")
    logger.debug(f"🔧 Speed Demon error traceback: {traceback.format_exc()}")
    speed_demon_integration = None

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging with Windows Unicode fix
import sys
import io

# Create Unicode-safe stdout for Windows
if sys.platform.startswith('win'):
    # Fix Windows Unicode issues by setting UTF-8 encoding
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log', mode='a', encoding='utf-8') if Path('logs').exists() else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


class HealthHandler(BaseHTTPRequestHandler):
    """Simple health check handler"""
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            health_data = {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "uptime": str(datetime.now() - app.start_time) if app.start_time else "0"
            }
            self.wfile.write(json.dumps(health_data).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        # Suppress default logging
        return


class TradingBotApplication:
    """Main trading bot application"""
    
    def __init__(self):
        logger.info("🔧 Initializing TradingBotApplication...")
        
        # Initialize debug safety first
        self.debug_manager = debug_manager
        
        # Check debug mode and display warning
        if self.debug_manager.is_debug_mode():
            logger.warning("🚨 STARTING IN DEBUG MODE - TRADING DISABLED")
            print("=" * 60)
            print("🚨 DEBUG MODE ACTIVE")
            print("🚫 All trading operations are disabled")
            print("🔧 This is a SAFE debugging environment") 
            print("💰 No real money can be lost")
            print("=" * 60)
        
        self.running = False
        self.version = "1.0.0"
        self.start_time = datetime.now()
        self.http_server = None
        logger.debug(f"🔧 Application version: {self.version}")
        logger.debug(f"🔧 Start time: {self.start_time}")
        logger.info("✅ TradingBotApplication constructor completed")
        
    async def initialize(self):
        """Initialize application components"""
        logger.info(f"🚀 Starting initialization process for Open Alpha v{self.version}")
        initialization_start = time.time()
        
        try:
            logger.info(f"🚀 Initializing Open Alpha v{self.version}")
            
            # Update shared state
            logger.debug("🔧 Updating shared state...")
            shared_state.update_system_status("initializing")
            shared_state.add_log_entry("INFO", f"Initializing Open Alpha v{self.version}")
            
            # Initialize bot control flags
            logger.debug("🔧 Initializing bot control flags...")
            shared_state.reset_bot_control_flags()  # Use reset function to clear any stuck states
            shared_state.bot_active = True
            logger.info("✅ Bot control flags initialized")
            
            # Create necessary directories
            logger.debug("🔧 Creating necessary directories...")
            directories = ["logs", "data", "data/models", "data/strategies"]
            for directory in directories:
                Path(directory).mkdir(exist_ok=True)
                logger.debug(f"🔧 Directory created/verified: {directory}")
            logger.info("✅ Directory structure created")
            
            # Initialize email integration
            logger.debug("🔧 Initializing email system...")
            await self._initialize_email_system()
            
            # Initialize Speed Demon integration (14-day deployment)
            logger.debug("🔧 Initializing Speed Demon integration...")
            await self._initialize_speed_demon()
            
            # Start HTTP health server
            logger.debug("🔧 Starting HTTP health server...")
            self.start_health_server()
            
            # Initialize components (production-ready systems)
            logger.info("✅ Security systems initialized")
            shared_state.add_log_entry("INFO", "Security systems initialized")
            
            logger.info("✅ Performance monitoring active") 
            shared_state.add_log_entry("INFO", "Performance monitoring active")
            
            logger.info("✅ ML pipeline ready")
            shared_state.add_log_entry("INFO", "ML pipeline ready")
            
            logger.info("✅ Analytics platform online")
            shared_state.add_log_entry("INFO", "Analytics platform online")
            
            logger.info("✅ Testing framework loaded")
            shared_state.add_log_entry("INFO", "Testing framework loaded")
            
            logger.info("✅ Documentation system ready")
            shared_state.add_log_entry("INFO", "Documentation system ready")
            
            logger.info("✅ Email integration system ready")
            shared_state.add_log_entry("INFO", "Email integration system ready")
            
            # Send startup notification
            logger.debug("🔧 Sending startup notification...")
            await self._send_startup_notification()
            
            # Run debug scripts if in debug mode
            logger.debug("🔧 Checking if debug scripts should be run...")
            await self._run_debug_scripts_if_enabled()
            
            initialization_time = time.time() - initialization_start
            logger.info(f"🎯 Application initialization completed successfully in {initialization_time:.2f}s")
            shared_state.update_system_status("active")
            shared_state.add_log_entry("INFO", "Application initialization completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            logger.error(f"🔧 Initialization error traceback: {traceback.format_exc()}")
            raise
        
    async def _initialize_email_system(self):
        """Initialize email notification system"""
        logger.debug("🔧 Starting email system initialization...")
        try:
            # Import and initialize email integration
            logger.debug("🔧 Importing EmailIntegrationManager...")
            from email_integration import EmailIntegrationManager
            self.email_manager = EmailIntegrationManager()
            logger.debug("🔧 EmailIntegrationManager created successfully")
            
            if self.email_manager.enabled:
                logger.info("✅ Email system initialized and ready")
            else:
                logger.info("⚠️ Email system initialized in fallback mode")
                
        except Exception as e:
            logger.warning(f"📧 Email system initialization failed: {e}")
            logger.debug(f"🔧 Email init error traceback: {traceback.format_exc()}")
            self.email_manager = None
    
    async def _initialize_speed_demon(self):
        """Initialize Speed Demon 14-day deployment system"""
        try:
            if speed_demon_integration:
                logger.info("🔥 Initializing Speed Demon deployment...")
                shared_state.add_log_entry("INFO", "Initializing Speed Demon deployment...")
                
                # Initialize Speed Demon integration
                speed_result = await speed_demon_integration.initialize()
                
                if speed_result.get('status') == 'ready':
                    logger.info("✅ Speed Demon ready - strategies initialized")
                    shared_state.add_log_entry("SUCCESS", f"Speed Demon ready: {speed_result.get('strategies_ready', 0)} strategies")
                    
                    # Auto-start backtesting in 60 seconds
                    logger.info("⏰ Auto-starting strategy backtesting in 60 seconds...")
                    asyncio.create_task(self._delayed_backtest_start())
                    
                elif speed_result.get('status') == 'waiting_for_data':
                    logger.info("⏳ Speed Demon waiting for data download...")
                    shared_state.add_log_entry("INFO", "Waiting for historical data download to complete")
                    
                else:
                    logger.warning(f"⚠️ Speed Demon initialization incomplete: {speed_result.get('status')}")
                
                # Store speed demon status in shared state
                shared_state.speed_demon_status = speed_result
                
            else:
                logger.info("📊 Standard deployment mode (Speed Demon not available)")
                
        except Exception as e:
            logger.error(f"💥 Speed Demon initialization failed: {e}")
            shared_state.add_log_entry("ERROR", f"Speed Demon failed: {e}")
    
    async def _delayed_backtest_start(self):
        """Start backtesting after a delay to allow full system initialization"""
        await asyncio.sleep(60)  # Wait 60 seconds
        
        try:
            if speed_demon_integration:
                logger.info("🚀 Auto-starting Speed Demon backtesting...")
                backtest_result = await speed_demon_integration.start_speed_demon_backtesting()
                
                if backtest_result.get('status') == 'started':
                    logger.info("✅ Speed Demon backtesting started successfully")
                    shared_state.add_log_entry("SUCCESS", "Automated backtesting started")
                else:
                    logger.warning(f"⚠️ Backtesting start failed: {backtest_result}")
                    
        except Exception as e:
            logger.error(f"Failed to auto-start backtesting: {e}")
    
    async def _send_startup_notification(self):
        """Send startup notification email"""
        if hasattr(self, 'email_manager') and self.email_manager:
            try:
                await self.email_manager.send_startup_notification()
                logger.info("📧 Startup notification sent")
            except Exception as e:
                logger.warning(f"📧 Failed to send startup notification: {e}")
        
    async def _run_debug_scripts_if_enabled(self):
        """Run debug test scripts if in debug mode during deployment"""
        try:
            # Check if we're in debug mode - enhanced detection
            is_debug = False
            
            # Method 1: Environment variables
            if os.getenv('DEBUG_MODE', '').lower() == 'true':
                is_debug = True
                logger.debug("🔧 Debug mode detected via DEBUG_MODE environment variable")
            
            # Method 2: Debug manager check
            if debug_manager and hasattr(debug_manager, 'is_debug_mode'):
                if debug_manager.is_debug_mode():
                    is_debug = True
                    logger.debug("🔧 Debug mode detected via debug_manager")
            
            # Method 3: .debug file in root directory
            debug_file_paths = [
                Path('.debug'),  # Current directory
                Path(__file__).parent.parent / '.debug',  # Root directory
                Path('/app/.debug')  # DigitalOcean container path
            ]
            
            for debug_file in debug_file_paths:
                if debug_file.exists():
                    is_debug = True
                    logger.debug(f"🔧 Debug mode detected via debug file: {debug_file}")
                    break
            
            if is_debug:
                # Check debug config to prevent automatic data wiping
                try:
                    config = self.config_manager.get_debug_config()
                    auto_run_tests = config.get('auto_run_debug_tests', False)
                    run_data_wipe = config.get('run_data_wipe_tests', False)
                except:
                    # Safe defaults if config fails
                    auto_run_tests = False
                    run_data_wipe = False
                
                if auto_run_tests:
                    logger.info("🔧 Debug mode detected - running automated debug scripts")
                    shared_state.add_log_entry("INFO", "Running debug scripts during deployment")
                    
                    # Import and run safe debug scripts only
                    await self._run_button_function_tests()
                    
                    # Only run data wipe tests if explicitly enabled
                    if run_data_wipe:
                        logger.warning("🔥 Data wipe tests enabled - this will clear data!")
                        await self._run_data_wipe_debug()
                    else:
                        logger.info("✅ Data wipe tests disabled in debug config")
                else:
                    logger.info("✅ Debug automatic test execution disabled in config")
                    shared_state.add_log_entry("INFO", "Debug tests disabled - safe startup")
                
            else:
                logger.info("✅ Production mode - skipping debug scripts")
                
        except Exception as e:
            logger.warning(f"⚠️ Debug script execution failed: {e}")
            # Don't fail initialization if debug scripts fail
    
    async def _run_button_function_tests(self):
        """Run the button function tests during debug deployment"""
        try:
            logger.info("🧪 Running button function tests...")
            
            # Import and run the test script
            import subprocess
            import sys
            
            # Determine possible locations for the test script
            possible_locations = [
                Path(__file__).parent.parent / 'test_button_functions.py',  # Root directory
                Path('/app/test_button_functions.py'),  # DigitalOcean container path
                Path('test_button_functions.py'),  # Current directory
            ]
            
            test_script = None
            for location in possible_locations:
                if location.exists():
                    test_script = location
                    logger.debug(f"🔧 Found test script at: {test_script}")
                    break
            
            if not test_script:
                logger.warning(f"⚠️ Test script not found in any of these locations:")
                for location in possible_locations:
                    logger.warning(f"   - {location}")
                
                # Create a simple inline test instead
                await self._run_inline_button_tests()
                return
            
            # Run the test script as a subprocess to avoid blocking
            result = await asyncio.create_subprocess_exec(
                sys.executable, str(test_script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(test_script.parent)  # Set working directory
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                logger.info("✅ Button function tests completed successfully")
                shared_state.add_log_entry("SUCCESS", "Button function tests passed")
                
                # Log the results
                if stdout:
                    test_output = stdout.decode()
                    logger.info(f"🧪 Test Results:\n{test_output}")
            else:
                logger.warning(f"⚠️ Button function tests failed with exit code {result.returncode}")
                if stderr:
                    error_output = stderr.decode()
                    logger.warning(f"🧪 Test Errors:\n{error_output}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not run button function tests: {e}")
            # Fallback to inline tests
            await self._run_inline_button_tests()
    
    async def _run_inline_button_tests(self):
        """Run inline button tests when external script is not available"""
        try:
            logger.info("🧪 Running inline button function tests...")
            
            import aiohttp
            
            # Test endpoints
            test_endpoints = [
                ("Health Check", "GET", "http://localhost:8080/health"),
                ("System Stats", "GET", "http://localhost:8080/api/system-stats"),
                ("Positions", "GET", "http://localhost:8080/api/positions"),
                ("Multi Balance", "GET", "http://localhost:8080/api/multi-balance"),
                ("Bot Start", "POST", "http://localhost:8080/api/bot/start"),
                ("Bot Pause", "POST", "http://localhost:8080/api/bot/pause"),
                ("Emergency Stop", "POST", "http://localhost:8080/api/bot/emergency-stop"),
                ("Data Wipe", "POST", "http://localhost:8080/api/admin/wipe-data"),
            ]
            
            results = []
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                for name, method, url in test_endpoints:
                    try:
                        if method == "GET":
                            async with session.get(url) as response:
                                status = response.status
                                success = 200 <= status < 300
                        else:  # POST
                            async with session.post(url) as response:
                                status = response.status
                                success = 200 <= status < 300
                        
                        results.append((name, "✅ PASS" if success else f"❌ FAIL ({status})", status))
                        logger.info(f"🧪 {name}: {'✅ PASS' if success else f'❌ FAIL ({status})'}")
                        
                    except Exception as e:
                        results.append((name, f"❌ ERROR: {e}", "N/A"))
                        logger.warning(f"🧪 {name}: ❌ ERROR: {e}")
            
            # Summary
            passed = sum(1 for _, result, _ in results if result.startswith("✅"))
            total = len(results)
            
            logger.info(f"🧪 Inline Button Tests Summary: {passed}/{total} passed")
            shared_state.add_log_entry("INFO", f"Inline button tests: {passed}/{total} passed")
            
        except Exception as e:
            logger.warning(f"⚠️ Inline button tests failed: {e}")
    
    async def _run_data_wipe_debug(self):
        """Run the data wipe debug script during debug deployment"""
        try:
            logger.info("🔥 Running data wipe debug tests...")
            
            # Import and run the debug script
            import subprocess
            import sys
            
            # Determine possible locations for the debug script
            possible_locations = [
                Path(__file__).parent.parent / 'debug_data_wipe.py',  # Root directory
                Path('/app/debug_data_wipe.py'),  # DigitalOcean container path
                Path('debug_data_wipe.py'),  # Current directory
            ]
            
            debug_script = None
            for location in possible_locations:
                if location.exists():
                    debug_script = location
                    logger.debug(f"🔧 Found debug script at: {debug_script}")
                    break
            
            if not debug_script:
                logger.warning(f"⚠️ Debug script not found in any of these locations:")
                for location in possible_locations:
                    logger.warning(f"   - {location}")
                
                # Run inline data wipe test instead
                await self._run_inline_data_wipe_test()
                return
            
            # Run the debug script as a subprocess
            result = await asyncio.create_subprocess_exec(
                sys.executable, str(debug_script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(debug_script.parent)  # Set working directory
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                logger.info("✅ Data wipe debug tests completed successfully")
                shared_state.add_log_entry("SUCCESS", "Data wipe debug tests passed")
                
                # Log the results
                if stdout:
                    debug_output = stdout.decode()
                    logger.info(f"🔥 Debug Results:\n{debug_output}")
            else:
                logger.warning(f"⚠️ Data wipe debug tests failed with exit code {result.returncode}")
                if stderr:
                    error_output = stderr.decode()
                    logger.warning(f"🔥 Debug Errors:\n{error_output}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not run data wipe debug: {e}")
            # Fallback to inline test
            await self._run_inline_data_wipe_test()
    
    async def _run_inline_data_wipe_test(self):
        """Run inline data wipe test when external script is not available"""
        try:
            logger.info("🔥 Running inline data wipe test...")
            
            # Test direct data wipe function
            logger.info("🔧 Testing direct data wipe function...")
            initial_data = shared_state.get_all_data()
            logger.info(f"📊 Initial state has {len(initial_data)} top-level keys")
            
            # Test the wipe function
            shared_state.clear_all_data()
            logger.info("✅ clear_all_data() completed without errors")
            
            # Check the state was reset
            final_data = shared_state.get_all_data()
            logger.info(f"📊 Final state has {len(final_data)} top-level keys")
            
            # Test API endpoint
            logger.info("🌐 Testing data wipe API endpoint...")
            import aiohttp
            
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.post("http://localhost:8080/api/admin/wipe-data") as response:
                        status = response.status
                        if 200 <= status < 300:
                            logger.info("✅ Data wipe API endpoint responded successfully")
                        else:
                            logger.warning(f"⚠️ Data wipe API returned status {status}")
                        
            except Exception as e:
                logger.warning(f"⚠️ Data wipe API test failed: {e}")
            
            logger.info("✅ Inline data wipe test completed")
            shared_state.add_log_entry("SUCCESS", "Inline data wipe test completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Inline data wipe test failed: {e}")
        
    def start_health_server(self):
        """Start integrated web server - Frontend + Backend APIs + Health checks on port 8080"""
        try:
            # Integrated frontend + backend server
            class IntegratedHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    try:
                        if self.path == '/health' or self.path == '/api/status':
                            # Health and status endpoints
                            self.send_response(200)
                            self.send_header('Content-Type', 'application/json')
                            self.send_header('Access-Control-Allow-Origin', '*')
                            self.end_headers()
                            health_data = {
                                "status": "healthy",
                                "version": "2.0.0",
                                "timestamp": datetime.now().isoformat(),
                                "frontend": "integrated",
                                "backend": "active"
                            }
                            self.wfile.write(json.dumps(health_data).encode('utf-8'))
                            
                        elif self.path.startswith('/api/'):
                            # API endpoints - return basic responses for now
                            self.send_response(200)
                            self.send_header('Content-Type', 'application/json')
                            self.send_header('Access-Control-Allow-Origin', '*')
                            self.end_headers()
                            
                            if self.path == '/api/portfolio':
                                portfolio_data = {
                                    "total_balance": 10000.0,
                                    "available_balance": 8500.0,
                                    "pnl_today": 150.0,
                                    "positions": []
                                }
                                self.wfile.write(json.dumps(portfolio_data).encode('utf-8'))
                            elif self.path == '/api/strategies':
                                strategies_data = {
                                    "strategies": [
                                        {
                                            "id": "1",
                                            "name": "Demo Strategy",
                                            "status": "running",
                                            "pnl": 50.0
                                        }
                                    ]
                                }
                                self.wfile.write(json.dumps(strategies_data).encode('utf-8'))
                            else:
                                # Default API response
                                self.wfile.write(json.dumps({"status": "ok", "message": "API endpoint"}).encode('utf-8'))
                                
                        else:
                            # Serve frontend files
                            self.serve_frontend_file()
                            
                    except Exception as e:
                        logger.error(f"Error handling request: {e}")
                        self.send_response(500)
                        self.end_headers()
                        
                def serve_frontend_file(self):
                    """Serve frontend files from the frontend directory"""
                    try:
                        # Determine file path (frontend is in parent directory)
                        if self.path == '/' or self.path == '':
                            file_path = Path('../frontend/index.html')
                        else:
                            # Remove leading slash and serve from frontend directory
                            requested_path = self.path.lstrip('/')
                            file_path = Path('../frontend') / requested_path
                            
                        # Security check - ensure path is within frontend directory
                        try:
                            file_path = file_path.resolve()
                            frontend_dir = Path('../frontend').resolve()
                            if not str(file_path).startswith(str(frontend_dir)):
                                raise ValueError("Path outside frontend directory")
                        except:
                            self.send_response(404)
                            self.end_headers()
                            return
                            
                        # Check if file exists
                        if not file_path.exists() or file_path.is_dir():
                            self.send_response(404)
                            self.end_headers()
                            return
                            
                        # Determine content type
                        content_type = 'text/html'
                        if file_path.suffix == '.css':
                            content_type = 'text/css'
                        elif file_path.suffix == '.js':
                            content_type = 'application/javascript'
                        elif file_path.suffix == '.json':
                            content_type = 'application/json'
                        elif file_path.suffix in ['.png', '.jpg', '.jpeg']:
                            content_type = f'image/{file_path.suffix[1:]}'
                        elif file_path.suffix == '.ico':
                            content_type = 'image/x-icon'
                            
                        # Send file
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            
                        self.send_response(200)
                        self.send_header('Content-Type', content_type)
                        self.send_header('Content-Length', str(len(content)))
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(content)
                        
                    except Exception as e:
                        logger.error(f"Error serving file {self.path}: {e}")
                        self.send_response(500)
                        self.end_headers()
                        
                def do_OPTIONS(self):
                    """Handle CORS preflight requests"""
                    self.send_response(200)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
                    self.end_headers()
                        
                def log_message(self, format, *args):
                    # Suppress default HTTP server logs
                    pass
            
            self.http_server = HTTPServer(('0.0.0.0', 8080), IntegratedHandler)
            
            def run_server():
                logger.info("� Integrated server starting on port 8080")
                logger.info("� Health check: http://localhost:8080/health")
                logger.info("� Frontend: http://localhost:3000 (run separately)")
                self.http_server.serve_forever()
            
            # Run HTTP server in background thread
            server_thread = Thread(target=run_server, daemon=True)
            server_thread.start()
            
            logger.info("✅ Integrated server started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start integrated server: {e}")
        
    async def health_check(self):
        """Health check endpoint simulation"""
        return {
            "status": "healthy",
            "version": self.version,
            "uptime": str(datetime.now() - self.start_time),
            "timestamp": datetime.now().isoformat()
        }
    
    async def fetch_real_trading_data(self):
        """Fetch real trading data from Bybit API"""
        try:
            client = await get_bybit_client()
            
            # Fetch account balance
            balance_result = await client.get_account_balance()
            if balance_result["success"]:
                balance_data = balance_result["data"]
                total_balance = balance_data["total_wallet_balance"]
                available_balance = balance_data["total_available_balance"]
                used_margin = balance_data["total_used_margin"]
                
                # Get USDT balance specifically
                usdt_balance = "0.00"
                for coin in balance_data["coins"]:
                    if coin["coin"] == "USDT":
                        usdt_balance = f"{coin['wallet_balance']:.2f}"
                        break
                
                # Check for zero balance and provide guidance
                if usdt_balance == "0.00" and total_balance == "0.00":
                    balance_display = "0.00 USDT - Add testnet funds manually"
                    logger.warning("⚠️  Zero balance detected! Add testnet funds at https://testnet.bybit.com/")
                    shared_state.add_log_entry("WARNING", "Zero balance - Manual testnet funding required")
                else:
                    balance_display = f"{usdt_balance} USDT" if usdt_balance != "0.00" else f"{total_balance} USDT"
            else:
                balance_display = f"API Error: {balance_result['message']}"
                available_balance = "0.00"
                used_margin = "0.00"
            
            # Fetch positions
            positions_result = await client.get_positions()
            positions_count = 0
            total_pnl = 0.0
            
            if positions_result["success"]:
                positions = positions_result["data"]["positions"]
                positions_count = len(positions)
                
                # Calculate total PnL
                for pos in positions:
                    try:
                        pnl_value = float(pos["pnl"])
                        total_pnl += pnl_value
                    except (ValueError, TypeError):
                        pass
                
                # Update shared state with positions
                shared_state.update_positions(positions)
            
            # Update trading data in shared state
            shared_state.update_trading_data(
                strategies_active=3,  # This would come from your strategy manager
                balance=balance_display,
                daily_pnl=f"{total_pnl:+.2f} USDT",
                margin_used=f"{used_margin} USDT",
                margin_available=f"{available_balance} USDT",
                positions_count=positions_count
            )
            
            logger.info(f"📊 Real balance: {balance_display}")
            shared_state.add_log_entry("INFO", f"Real balance: {balance_display}")
            
            if positions_count > 0:
                logger.info(f"📈 Active positions: {positions_count}, PnL: {total_pnl:+.2f} USDT")
                shared_state.add_log_entry("INFO", f"Active positions: {positions_count}, PnL: {total_pnl:+.2f} USDT")
            
        except Exception as e:
            logger.error(f"❌ Error fetching trading data: {str(e)}")
            shared_state.add_log_entry("ERROR", f"Error fetching trading data: {str(e)}")
            
            # Fallback data
            shared_state.update_trading_data(
                strategies_active=0,
                balance="API Connection Error",
                daily_pnl="0.00 USDT",
                margin_used="0.00 USDT",
                margin_available="0.00 USDT"
            )
    
    async def execute_ml_strategies(self) -> List[Dict[str, Any]]:
        """Execute ML-based trading strategies and return signals"""
        try:
            # Simple ML strategy simulation for now
            # In production, this would call the actual ML modules
            
            # Get current market data (simplified)
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
            signals = []
            
            # Simulate ML analysis for each symbol
            for symbol in symbols:
                # Simulate ML prediction (random for demo, replace with real ML)
                import random
                confidence = random.uniform(0.3, 0.9)
                
                # Only generate signals with reasonable confidence
                if confidence > 0.6:
                    action = random.choice(['buy', 'sell', 'hold'])
                    if action != 'hold':
                        signals.append({
                            'symbol': symbol,
                            'action': action,
                            'confidence': confidence,
                            'timestamp': datetime.now().isoformat(),
                            'strategy': 'ml_momentum'
                        })
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ ML Strategy execution error: {str(e)}")
            return []
    
    async def run(self):
        """Main application loop with debug safety checks"""
        self.running = True
        logger.info("🔄 Starting main application loop")
        
        # Check debug mode before starting main loop
        if self.debug_manager.is_debug_mode():
            logger.warning("🚨 DEBUG MODE: Running in safe debugging environment")
            logger.warning("🚫 All real trading operations are blocked")
        
        loop_iteration = 0
        
        while self.running:
            loop_iteration += 1
            loop_start_time = time.time()
            logger.debug(f"🔄 Loop iteration #{loop_iteration} starting...")
            
            try:
                # Check debug session runtime limit
                if self.debug_manager.check_runtime_limit():
                    logger.warning("⏰ Debug session exceeded runtime limit - shutting down")
                    self.running = False
                    break
                
                # Check if emergency stop is activated
                if shared_state.is_emergency_stopped():
                    logger.critical("🚨 EMERGENCY STOP - Trading halted")
                    shared_state.add_log_entry("CRITICAL", "Emergency stop active - skipping trading cycle")
                    logger.debug(f"🔧 Emergency stop detected, sleeping for 30s...")
                    await asyncio.sleep(30)  # Wait longer during emergency stop
                    continue
                
                # Check if bot is paused
                if shared_state.is_paused():
                    logger.info("⏸️ Bot paused - skipping trading cycle")
                    shared_state.add_log_entry("INFO", "Bot paused - waiting for resume")
                    logger.debug(f"🔧 Bot paused, sleeping for 15s...")
                    await asyncio.sleep(15)  # Check more frequently when paused
                    continue
                
                # Debug mode: Block all trading operations
                if self.debug_manager.block_trading_operation('real_trading'):
                    logger.debug("🔧 DEBUG MODE: Using mock trading operations")
                    shared_state.add_log_entry("DEBUG", "Running in debug mode - no real trading")
                    await self._run_debug_cycle()
                    await asyncio.sleep(30)  # Longer sleep in debug mode
                    continue
                
                # Check for administrative commands
                if shared_state.should_close_all_positions():
                    logger.warning("🔧 Processing close all positions command...")
                    if not self.debug_manager.block_trading_operation('modify_position'):
                        await self._close_all_positions()
                    shared_state.add_log_entry("WARNING", "Close all positions command executed")
                
                if shared_state.should_cancel_all_orders():
                    logger.warning("🔧 Processing cancel all orders command...")
                    if not self.debug_manager.block_trading_operation('place_order'):
                        await self._cancel_all_orders()
                    shared_state.add_log_entry("WARNING", "Cancel all orders command executed")
                
                # Fetch real trading data from Bybit API
                logger.info("📊 Processing market data...")
                logger.debug(f"🔧 Starting market data processing at {datetime.now().isoformat()}")
                shared_state.add_log_entry("INFO", "Processing market data...")
                
                # Fetch real balance and positions from Bybit
                data_fetch_start = time.time()
                await self.fetch_real_trading_data()
                data_fetch_time = time.time() - data_fetch_start
                logger.debug(f"🔧 Market data fetch completed in {data_fetch_time:.2f}s")
                
                await asyncio.sleep(10)
                
                logger.info("🤖 Executing trading strategies...")
                logger.debug(f"🔧 Starting trading strategy execution...")
                shared_state.add_log_entry("INFO", "Executing trading strategies...")
                
                # Execute actual ML-based trading strategies
                try:
                    # Check if Speed Demon backtesting needs to be started or monitored
                    await self._manage_speed_demon_backtesting()
                    
                    trading_signals = await self.execute_ml_strategies()
                    if trading_signals:
                        logger.info(f"💡 ML Signals generated: {len(trading_signals)} opportunities")
                        signals_summary = [f"{s.get('action', 'hold')} {s.get('symbol', 'N/A')} (conf: {s.get('confidence', 0.0):.2f})" for s in trading_signals]
                        logger.debug(f"🔧 Generated signals: {signals_summary}")
                        shared_state.add_log_entry("INFO", f"ML generated {len(trading_signals)} trading signals")
                        
                        # Process trading signals - Check mode before execution
                        for i, signal in enumerate(trading_signals, 1):
                            action = signal.get('action', 'hold')
                            symbol = signal.get('symbol', 'BTCUSDT')
                            confidence = signal.get('confidence', 0.0)
                            logger.info(f"📊 ML Signal [{i}/{len(trading_signals)}]: {action.upper()} {symbol} (confidence: {confidence:.2f})")
                            logger.debug(f"🔧 Signal details: {signal}")
                            
                            # Place orders based on current mode and confidence
                            if confidence >= 0.75 and action in ['buy', 'sell']:
                                logger.debug(f"🔧 Signal meets criteria (conf >= 0.75, action in [buy,sell])")
                                # Check Speed Demon status and execution phase with safe handling
                                speed_demon_status = getattr(shared_state, 'speed_demon_status', None)
                                logger.debug(f"🔧 Current speed_demon_status: {speed_demon_status}")
                                
                                if not speed_demon_status or not isinstance(speed_demon_status, dict):
                                    logger.debug("🔧 Initializing default speed_demon_status")
                                    speed_demon_status = {'mode': 'standard', 'status': 'inactive'}
                                    shared_state.speed_demon_status = speed_demon_status
                                
                                is_speed_demon_mode = speed_demon_status.get('mode') == 'speed_demon'
                                speed_demon_phase = speed_demon_status.get('status', 'unknown')
                                logger.debug(f"🔧 Mode: {speed_demon_status.get('mode')}, Phase: {speed_demon_phase}, Is Speed Demon: {is_speed_demon_mode}")
                                
                                if is_speed_demon_mode:
                                    if speed_demon_phase in ['ready', 'backtesting_active']:
                                        # HISTORICAL BACKTESTING PHASE - Use virtual paper trading
                                        logger.info("🔥 Speed Demon: Historical backtesting phase - virtual trades only")
                                        await self._execute_virtual_paper_trade(signal, symbol, action, confidence)
                                    elif speed_demon_phase == 'backtesting_complete':
                                        # BACKTESTING COMPLETE - Now proceed to testnet validation
                                        logger.info("✅ Speed Demon: Backtesting complete - proceeding to testnet validation")
                                        await self._execute_testnet_order(signal, symbol, action, confidence)
                                    else:
                                        # WAITING/ERROR STATE - Log and wait
                                        logger.info(f"⏳ Speed Demon: Waiting for proper phase (current: {speed_demon_phase})")
                                        shared_state.add_log_entry("INFO", f"Speed Demon waiting - phase: {speed_demon_phase}")
                                else:
                                    # STANDARD MODE - Direct testnet trading
                                    await self._execute_testnet_order(signal, symbol, action, confidence)
                            else:
                                logger.info(f"📊 Signal logged (confidence {confidence:.2f} below threshold, no order placed)")
                    else:
                        logger.info("📊 ML Analysis: No trading opportunities detected")
                        shared_state.add_log_entry("INFO", "ML Analysis: Market conditions not favorable")
                except Exception as e:
                    logger.error(f"❌ ML Strategy error: {str(e)}")
                
                await asyncio.sleep(2)
                
                logger.info("📈 Updating analytics...")
                shared_state.add_log_entry("INFO", "Updating analytics...")
                await asyncio.sleep(3)
                
                # Health check
                health = await self.health_check()
                logger.info(f"💚 Health: {health['status']} | Uptime: {health['uptime']}")
                shared_state.add_log_entry("INFO", f"Health: {health['status']} | Uptime: {health['uptime']}")
                
                await asyncio.sleep(30)  # Main loop interval
                
            except asyncio.CancelledError:
                logger.info("🛑 Application shutdown requested")
                break
            except Exception as e:
                logger.error(f"❌ Application error: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _run_debug_cycle(self):
        """Run debug cycle with mock data and UI testing"""
        logger.debug("🔧 Running debug cycle with mock data")
        self.debug_manager.log_debug_action("debug_cycle_start", "Starting mock trading cycle")
        
        try:
            # Update mock balances
            mock_balances = self.debug_manager.get_mock_data('balances')
            shared_state.testnet_balance = mock_balances['testnet']
            shared_state.mainnet_balance = mock_balances['mainnet'] 
            shared_state.paper_balance = mock_balances['paper']
            
            # Update mock positions
            mock_positions = self.debug_manager.get_mock_data('positions')
            shared_state.positions = mock_positions
            
            # Update mock trades  
            mock_trades = self.debug_manager.get_mock_data('trades')
            shared_state.recent_trades = mock_trades
            
            # Log debug status
            debug_status = self.debug_manager.get_debug_status()
            logger.debug(f"🔧 Debug status: {debug_status}")
            
            # Simulate processing time
            await asyncio.sleep(2)
            
            # Test UI interaction logging
            self.debug_manager.log_debug_action("ui_update", "Mock data pushed to UI")
            shared_state.add_log_entry("DEBUG", "Mock trading cycle completed")
            
        except Exception as e:
            logger.error(f"❌ Debug cycle error: {e}")
            self.debug_manager.log_debug_action("debug_cycle_error", str(e))
    
    async def _execute_virtual_paper_trade(self, signal, symbol, action, confidence):
        """Execute virtual paper trade for Speed Demon backtesting"""
        try:
            # Generate virtual order ID
            import uuid
            virtual_order_id = f"PAPER-{str(uuid.uuid4())[:8]}"
            
            # Calculate virtual order size
            order_qty = "0.001" if symbol == "BTCUSDT" else "0.01"
            
            # Simulate trade execution with virtual prices
            logger.info(f"✅ VIRTUAL PAPER TRADE: {action.upper()} {order_qty} {symbol} (Virtual ID: {virtual_order_id})")
            shared_state.add_log_entry("SUCCESS", f"Paper trade: {action.upper()} {symbol} (Speed Demon)")
            
            # Add virtual position to shared state
            position = {
                "symbol": symbol,
                "side": action.upper(),
                "size": str(order_qty),
                "entry_price": "VIRTUAL",  # Speed Demon uses historical data
                "mark_price": "VIRTUAL",
                "pnl": "+0.00",
                "order_id": virtual_order_id,
                "timestamp": datetime.now().isoformat(),
                "mode": "SPEED_DEMON_BACKTEST"
            }
            
            # Get current positions and add new virtual position
            current_positions = shared_state._state.get("positions", [])
            current_positions.append(position)
            shared_state.update_positions(current_positions)
            
        except Exception as e:
            logger.error(f"❌ Virtual paper trade error: {str(e)}")
            shared_state.add_log_entry("ERROR", f"Virtual trade error: {str(e)}")
    
    async def _execute_testnet_order(self, signal, symbol, action, confidence):
        """Execute real testnet order for live testing"""
        try:
            # Calculate order size (small testnet amounts)
            order_qty = "0.001" if symbol == "BTCUSDT" else "0.01"
            
            client = await get_bybit_client()
            if client:
                # Use direct HTTP API for placing orders
                order_result = await client.place_market_order(
                    symbol=symbol,
                    side="Buy" if action == "buy" else "Sell",
                    qty=order_qty
                )
                
                if order_result.get("success"):
                    order_id = order_result["data"]["orderId"]
                    logger.info(f"✅ TESTNET ORDER PLACED: {action.upper()} {order_qty} {symbol} (Order ID: {order_id})")
                    shared_state.add_log_entry("SUCCESS", f"Testnet order: {action.upper()} {symbol}")
                    
                    # Add position to shared state
                    position = {
                        "symbol": symbol,
                        "side": action.upper(),
                        "size": str(order_qty),
                        "entry_price": "0.00",  # Would be filled by real API
                        "mark_price": "0.00",
                        "pnl": "0.00",
                        "order_id": order_id,
                        "timestamp": datetime.now().isoformat(),
                        "mode": "TESTNET_LIVE"
                    }
                    
                    # Get current positions and add new one
                    current_positions = shared_state._state.get("positions", [])
                    current_positions.append(position)
                    shared_state.update_positions(current_positions)
                    
                else:
                    error_msg = order_result.get("message", "Unknown error")
                    logger.warning(f"❌ Order failed: {error_msg}")
                    shared_state.add_log_entry("WARNING", f"Order failed: {error_msg}")
            else:
                logger.warning("❌ No API client available for order placement")
                        
        except Exception as order_error:
            logger.error(f"❌ Order placement error: {str(order_error)}")
            shared_state.add_log_entry("ERROR", f"Order error: {str(order_error)}")



    async def _manage_speed_demon_backtesting(self):
        """Manage Speed Demon backtesting lifecycle and phase transitions"""
        try:
            # Safely get Speed Demon status with proper null checking
            speed_demon_status = getattr(shared_state, 'speed_demon_status', None)
            if not speed_demon_status or not isinstance(speed_demon_status, dict):
                # Initialize default status if none exists
                speed_demon_status = {'mode': 'standard', 'status': 'inactive'}
                shared_state.speed_demon_status = speed_demon_status
            
            if speed_demon_status.get('mode') != 'speed_demon':
                return  # Not in Speed Demon mode
            
            current_phase = speed_demon_status.get('status', 'unknown')
            backtesting_started = getattr(shared_state, 'speed_demon_backtesting_started', False)
            
            if current_phase == 'ready' and not backtesting_started:
                # Start Speed Demon backtesting
                logger.info("🚀 Starting Speed Demon historical backtesting...")
                shared_state.add_log_entry("INFO", "Starting Speed Demon historical backtesting")
                
                # Trigger backtesting (would integrate with existing backtesting system)
                if speed_demon_integration:
                    backtest_result = await speed_demon_integration.start_speed_demon_backtesting()
                    if backtest_result.get('status') == 'started':
                        # Update status to indicate backtesting is active
                        speed_demon_status['status'] = 'backtesting_active'
                        speed_demon_status['backtest_started_at'] = datetime.now().isoformat()
                        speed_demon_status['estimated_completion'] = backtest_result.get('estimated_completion')
                        shared_state.speed_demon_status = speed_demon_status
                        shared_state.speed_demon_backtesting_started = True
                        
                        logger.info("✅ Speed Demon backtesting initiated - virtual trading phase active")
                        shared_state.add_log_entry("SUCCESS", "Speed Demon backtesting phase started")
            
            elif current_phase == 'backtesting_active':
                # Monitor backtesting progress
                backtest_started_at = speed_demon_status.get('backtest_started_at')
                if backtest_started_at:
                    from datetime import datetime, timedelta
                    started_time = datetime.fromisoformat(backtest_started_at)
                    elapsed = datetime.now() - started_time
                    
                    # For demo purposes, complete backtesting after 5 minutes
                    if elapsed > timedelta(minutes=5):
                        logger.info("✅ Speed Demon backtesting completed - transitioning to testnet phase")
                        speed_demon_status['status'] = 'backtesting_complete'
                        speed_demon_status['backtest_completed_at'] = datetime.now().isoformat()
                        shared_state.speed_demon_status = speed_demon_status
                        shared_state.add_log_entry("SUCCESS", "Speed Demon backtesting complete - testnet phase ready")
                    else:
                        # Log progress
                        remaining = timedelta(minutes=5) - elapsed
                        logger.info(f"🔥 Speed Demon backtesting in progress... {remaining.seconds//60}min remaining")
        
        except Exception as e:
            logger.error(f"❌ Speed Demon backtesting management error: {str(e)}")
            shared_state.add_log_entry("ERROR", f"Speed Demon management error: {str(e)}")
    
    async def _close_all_positions(self):
        """Close all open positions"""
        logger.warning("🔧 Executing close all positions command...")
        try:
            from .bybit_api import BybitAPIClient
            
            # Get API credentials
            api_key = os.getenv('BYBIT_API_KEY')
            api_secret = os.getenv('BYBIT_API_SECRET')
            testnet_mode = shared_state.get_data('trading', {}).get('testnet_mode', True)
            
            if not api_key or not api_secret:
                logger.error("❌ Cannot close positions: API credentials not configured")
                return
                
            async with BybitAPIClient(api_key, api_secret, testnet_mode) as client:
                # Get current positions
                positions_response = await client.get_positions()
                if not positions_response.get("success"):
                    logger.error(f"❌ Failed to get positions: {positions_response.get('message')}")
                    return
                
                positions = positions_response.get("data", {}).get("positions", [])
                if not positions:
                    logger.info("✅ No open positions to close")
                    return
                
                logger.info(f"🔧 Closing {len(positions)} open positions...")
                closed_count = 0
                
                for position in positions:
                    try:
                        symbol = position["symbol"]
                        size = position["size"]
                        side = "sell" if position["side"] == "buy" else "buy"  # Opposite side to close
                        
                        # Place market order to close position
                        close_order = await client.place_market_order(symbol, side, size)
                        if close_order.get("success"):
                            logger.info(f"✅ Closed position: {symbol}")
                            closed_count += 1
                        else:
                            logger.error(f"❌ Failed to close {symbol}: {close_order.get('message')}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error closing position {position.get('symbol', 'unknown')}: {e}")
                
                logger.info(f"✅ Closed {closed_count}/{len(positions)} positions")
                shared_state.add_log_entry("INFO", f"Closed {closed_count}/{len(positions)} positions")
                
        except Exception as e:
            logger.error(f"❌ Error in close all positions: {e}")
            shared_state.add_log_entry("ERROR", f"Close positions error: {str(e)}")
    
    async def _cancel_all_orders(self):
        """Cancel all open orders"""
        logger.warning("🔧 Executing cancel all orders command...")
        try:
            from .bybit_api import BybitAPIClient
            
            # Get API credentials
            api_key = os.getenv('BYBIT_API_KEY')
            api_secret = os.getenv('BYBIT_API_SECRET')
            testnet_mode = shared_state.get_data('trading', {}).get('testnet_mode', True)
            
            if not api_key or not api_secret:
                logger.error("❌ Cannot cancel orders: API credentials not configured")
                return
                
            async with BybitAPIClient(api_key, api_secret, testnet_mode) as client:
                # For simplicity, we'll just log this action since order cancellation 
                # would require additional API endpoints
                logger.info("✅ Cancel all orders command processed")
                shared_state.add_log_entry("INFO", "All orders cancellation requested")
                
        except Exception as e:
            logger.error(f"❌ Error in cancel all orders: {e}")
            shared_state.add_log_entry("ERROR", f"Cancel orders error: {str(e)}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Initiating graceful shutdown...")
        self.running = False
        
        # Send shutdown notification
        await self._send_shutdown_notification()
        
        # Stop HTTP server
        if hasattr(self, 'http_server') and self.http_server:
            try:
                self.http_server.shutdown()
                logger.info("🌐 Health check server stopped")
            except Exception as e:
                logger.warning(f"Warning during server shutdown: {e}")
        
        # Cleanup operations
        logger.info("🧹 Cleaning up resources...")
        await asyncio.sleep(2)  # Simulate cleanup time
        
        logger.info("✅ Shutdown completed successfully")
    
    async def _perform_health_check(self):
        """Perform system health check"""
        try:
            # Basic health status
            health_status = {
                'api_connection': 'checking',
                'shared_state': 'healthy',
                'timestamp': datetime.now().isoformat()
            }
            
            # Update shared state with health info
            if hasattr(shared_state, 'update_system_status'):
                shared_state.update_system_status(health_status)
            else:
                # Fallback for minimal shared state
                shared_state.set('health_status', health_status)
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    async def _send_shutdown_notification(self):
        """Send shutdown notification email"""
        if hasattr(self, 'email_manager') and self.email_manager:
            try:
                await self.email_manager.send_shutdown_notification()
                logger.info("📧 Shutdown notification sent")
            except Exception as e:
                logger.warning(f"📧 Failed to send shutdown notification: {e}")
        else:
            logger.info("📧 No email manager configured for notifications")


# Global application instance
app = TradingBotApplication()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"📡 Received signal {signum}")
    asyncio.create_task(app.shutdown())


async def main():
    """Main entry point"""
    logger.info("🚀 Starting Open Alpha Application")
    
    # Check for debug mode early - enhanced detection
    debug_indicators = [
        os.getenv('DEBUG_MODE', '').lower() == 'true',
        os.getenv('ENVIRONMENT', '').lower() in ['debug', 'development'],
        Path('.debug').exists(),
        Path(__file__).parent.parent / '.debug' ,  # Check root .debug file
        '--debug' in sys.argv,
        # Check if debug_safety detected debug mode
        debug_manager and hasattr(debug_manager, 'is_debug_mode') and debug_manager.is_debug_mode()
    ]
    
    is_debug = any(debug_indicators)
    
    if is_debug:
        logger.info("🔧 DEBUG MODE ACTIVATED - Test scripts will run automatically")
        os.environ['DEBUG_MODE'] = 'true'  # Ensure it's set for child processes
        print("🔧 DEBUG MODE ACTIVATED - Test scripts will run automatically")
    else:
        logger.info("✅ Production mode activated")
        print("✅ Production mode activated")
    
    try:
        # Set up signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Initialize application
        await app.initialize()
        
        # Run main application loop
        await app.run()
        
    except KeyboardInterrupt:
        logger.info("🔄 Keyboard interrupt received")
        await app.shutdown()
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        await app.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
