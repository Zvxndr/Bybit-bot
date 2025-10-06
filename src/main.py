"""
Main Application Entry Point - Clean Architecture
===============================================

Clean, reliable entry point using unified architecture components.
Eliminates import chaos and provides predictable startup behavior.
"""

import asyncio
import signal
import sys
from pathlib import Path
from datetime import datetime

# Initialize core components first
from core.import_manager import import_manager
from core.logging_manager import logging_manager
from core.config_loader import config_loader

# Setup logging first (Windows-safe)
logger = logging_manager.setup_comprehensive_logging()

# Initialize configuration
config = config_loader.get_main_config()
debug_config = config_loader.get_debug_config()
speed_demon_config = config_loader.get_speed_demon_config()

# Initialize unified risk-balance engine (Speed Demon built-in)
from risk_balance.unified_engine import unified_engine

# Get debug safety functions (reliable import)
get_debug_manager, is_debug_mode, block_trading_if_debug = import_manager.get_debug_safety_functions()

logger.info("=" * 60)
logger.info("BYBIT TRADING BOT - UNIFIED ARCHITECTURE")
logger.info("=" * 60)
logger.info(f"[OK] Application started at {datetime.now()}")
logger.info(f"[OK] Debug mode: {config_loader.is_debug_mode()}")
logger.info(f"[OK] Speed Demon enabled: {speed_demon_config.get('enabled', True)}")


class TradingBotApplication:
    """Main trading bot application with unified architecture"""
    
    def __init__(self):
        self.logger = logger
        self.config = config
        self.debug_config = debug_config
        self.speed_demon_config = speed_demon_config
        
        # Initialize components
        self.debug_manager = get_debug_manager()
        self.risk_engine = unified_engine
        self.app = None
        self.shutdown_event = asyncio.Event()
        
        self.logger.info("[OK] TradingBotApplication initialized")
    
    async def initialize_services(self):
        """Initialize all application services"""
        try:
            self.logger.info("[START] Initializing application services...")
            
            # Initialize unified risk-balance engine
            self.logger.info("[OK] Risk-Balance engine ready (Speed Demon built-in)")
            
            # Test risk calculations
            await self._test_speed_demon_features()
            
            # Initialize FastAPI app
            await self._initialize_fastapi()
            
            self.logger.info("[SUCCESS] All services initialized successfully")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Service initialization failed: {e}")
            raise
    
    async def _test_speed_demon_features(self):
        """Test Speed Demon dynamic risk features"""
        try:
            self.logger.info("[START] Testing Speed Demon features...")
            
            # Test different balance levels
            test_balances = [1000, 10000, 50000, 100000]
            
            for balance in test_balances:
                from decimal import Decimal
                risk_metrics = await self.risk_engine.get_current_risk_metrics(Decimal(str(balance)))
                
                self.logger.info(
                    f"[OK] Balance ${balance:,} -> "
                    f"{risk_metrics.risk_level.value} risk, "
                    f"${risk_metrics.position_size_limit:.2f} limit, "
                    f"{risk_metrics.balance_tier} tier"
                )
            
            self.logger.info("[SUCCESS] Speed Demon features working correctly")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Speed Demon test failed: {e}")
    
    async def _initialize_fastapi(self):
        """Initialize FastAPI application"""
        try:
            from fastapi import FastAPI
            from fastapi.staticfiles import StaticFiles
            from fastapi.responses import FileResponse
            import uvicorn
            
            # Create FastAPI app
            self.app = FastAPI(
                title="Bybit Trading Bot - Unified Architecture",
                description="Professional trading bot with built-in Speed Demon features",
                version="2.0.0"
            )
            
            # Add API routes
            await self._setup_api_routes()
            
            # Serve frontend
            frontend_dir = Path(__file__).parent.parent / "frontend"
            if frontend_dir.exists():
                self.app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
                
                @self.app.get("/")
                async def serve_frontend():
                    return FileResponse(frontend_dir / "index.html")
            
            self.logger.info("[OK] FastAPI application configured")
            
        except Exception as e:
            self.logger.error(f"[ERROR] FastAPI initialization failed: {e}")
            raise
    
    async def _setup_api_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/api/status")
        async def get_status():
            """Get application status"""
            try:
                # Get current risk metrics
                risk_metrics = await self.risk_engine.get_current_risk_metrics()
                balance_snapshot = await self.risk_engine.get_balance_snapshot()
                
                return {
                    "status": "running",
                    "debug_mode": config_loader.is_debug_mode(),
                    "speed_demon_enabled": self.speed_demon_config.get('enabled', True),
                    "balance": {
                        "total": float(balance_snapshot.total_balance),
                        "available": float(balance_snapshot.available_balance),
                        "tier": risk_metrics.balance_tier
                    },
                    "risk": {
                        "level": risk_metrics.risk_level.value,
                        "score": risk_metrics.portfolio_risk_score,
                        "position_limit": float(risk_metrics.position_size_limit),
                        "max_positions": risk_metrics.max_positions
                    },
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                self.logger.error(f"[ERROR] Status API failed: {e}")
                return {"status": "error", "error": str(e)}
        
        @self.app.get("/api/risk-metrics")
        async def get_risk_metrics():
            """Get detailed risk metrics"""
            try:
                risk_metrics = await self.risk_engine.get_current_risk_metrics()
                
                return {
                    "risk_level": risk_metrics.risk_level.value,
                    "portfolio_risk_score": risk_metrics.portfolio_risk_score,
                    "position_size_limit": float(risk_metrics.position_size_limit),
                    "max_positions": risk_metrics.max_positions,
                    "regime_adjustment": risk_metrics.regime_adjustment,
                    "balance_tier": risk_metrics.balance_tier
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.app.post("/api/calculate-position")
        async def calculate_position(request_data: dict):
            """Calculate position size for a trade"""
            try:
                from decimal import Decimal
                
                symbol = request_data.get('symbol', 'BTCUSDT')
                entry_price = Decimal(str(request_data.get('entry_price', 50000)))
                stop_loss = request_data.get('stop_loss_price')
                stop_loss_price = Decimal(str(stop_loss)) if stop_loss else None
                
                position_size, details = await self.risk_engine.calculate_position_size(
                    symbol, entry_price, stop_loss_price
                )
                
                return {
                    "symbol": symbol,
                    "position_size": float(position_size),
                    "calculation_details": details
                }
            except Exception as e:
                return {"error": str(e)}
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"[SHUTDOWN] Received signal {signum}")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Run the main application"""
        try:
            # Setup shutdown handlers
            self.setup_signal_handlers()
            
            # Initialize services
            await self.initialize_services()
            
            # Start the server
            import uvicorn
            
            config = uvicorn.Config(
                app=self.app,
                host="0.0.0.0",
                port=8080,
                log_config=None,  # Use our custom logging
                access_log=False  # Disable uvicorn access logs
            )
            
            server = uvicorn.Server(config)
            
            self.logger.info("[START] Starting server on http://0.0.0.0:8080")
            
            # Run server with shutdown event
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"[ERROR] Application run failed: {e}")
            raise
        finally:
            self.logger.info("[SHUTDOWN] Application shutdown complete")


async def main():
    """Main application entry point"""
    app = TradingBotApplication()
    await app.run()


if __name__ == "__main__":
    try:
        # Check if running in debug mode
        if config_loader.is_debug_mode():
            logger.warning("[WARN] Running in DEBUG mode - Trading disabled")
        
        # Run the application
        asyncio.run(main())
        
    except KeyboardInterrupt:
        logger.info("[SHUTDOWN] Application interrupted by user")
    except Exception as e:
        logger.error(f"[CRASH] Application crashed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)