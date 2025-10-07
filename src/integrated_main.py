"""
Integrated Trading Bot Dashboard
===============================

Production-ready dashboard integrating existing sophisticated backend:
- AI Pipeline System (ML Strategy Discovery → Paper Trading → Live Trading)
- Strategy Graduation System with automated promotion
- Comprehensive database models and APIs
- Real-time WebSocket updates
- Risk management and emergency controls

Launch: python -m src.integrated_main
"""

import os
import sys
import asyncio
import logging
import signal
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard_integration import DashboardIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/dashboard.log')
    ]
)
logger = logging.getLogger(__name__)

# Global integration instance
dashboard_integration = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global dashboard_integration
    
    # Startup
    try:
        logger.info("🚀 Starting integrated trading dashboard...")
        
        # Initialize dashboard integration
        dashboard_integration = DashboardIntegration(app)
        
        # Start background tasks
        await dashboard_integration.start_background_tasks()
        
        logger.info("✅ Integrated trading dashboard started successfully")
        logger.info("📊 Dashboard available at http://localhost:8000")
        
        yield
        
    finally:
        # Shutdown
        if dashboard_integration:
            logger.info("🛑 Shutting down integrated trading dashboard...")
            await dashboard_integration.shutdown()
            logger.info("✅ Shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Integrated Trading Bot Dashboard",
    description="Production dashboard with AI pipeline system",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

if __name__ == "__main__":
    # Ensure required directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Check environment
    if not os.getenv("BYBIT_API_KEY"):
        logger.warning("⚠️  No BYBIT_API_KEY found - running in paper trading mode")
    
    logger.info("🔧 Environment:")
    logger.info(f"   Python: {sys.version}")
    logger.info(f"   Working Directory: {os.getcwd()}")
    logger.info(f"   Project Root: {project_root}")
    
    # Run the application
    try:
        uvicorn.run(
            "src.integrated_main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload for production
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("👋 Dashboard stopped by user")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        sys.exit(1)