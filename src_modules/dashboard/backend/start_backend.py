"""
Dashboard Backend Startup Script
Quick startup script for development and testing
"""

import asyncio
import uvicorn
import logging
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def main():
    """Main startup function"""
    try:
        logger.info("🚀 Starting Bybit Trading Bot v2.0 Dashboard Backend...")
        
        # Run the FastAPI application
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️ Backend shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Backend startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()