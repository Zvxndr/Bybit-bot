"""
Simple Working Main Application
=============================

Basic FastAPI application that starts reliably and serves the frontend.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log', encoding='utf-8')
    ]
)

# Create logs directory
Path('logs').mkdir(exist_ok=True)

logger = logging.getLogger("main")

def main():
    """Main entry point - simplified version"""
    try:
        logger.info("=" * 50)
        logger.info("BYBIT TRADING BOT - SIMPLIFIED STARTUP")
        logger.info("=" * 50)
        logger.info(f"Started at: {datetime.now()}")
        
        # Import FastAPI components
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse, JSONResponse
        import uvicorn
        
        # Create FastAPI app
        app = FastAPI(
            title="Bybit Trading Bot",
            description="Professional Trading Bot Application",
            version="1.0.0"
        )
        
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # API status endpoint
        @app.get("/api/status")
        async def get_status():
            return {
                "status": "running",
                "mode": "simplified",
                "timestamp": datetime.now().isoformat(),
                "message": "Application is running in simplified mode"
            }
        
        # Dynamic risk calculation based on account size
        import math
        
        def calculate_dynamic_risk(balance_usd: float):
            """Core feature - dynamic risk scaling based on account size"""
            if balance_usd <= 10000:
                risk_ratio = 0.02  # 2% for small accounts
                tier = "small"
                level = "very_aggressive"
            elif balance_usd >= 100000:
                risk_ratio = 0.005  # 0.5% for large accounts  
                tier = "large"
                level = "conservative"
            else:
                # Exponential decay between 10k-100k
                ratio = (balance_usd - 10000) / 90000
                risk_ratio = 0.005 + (0.015 * math.exp(-2 * ratio))
                tier = "medium"
                level = "moderate"
            
            return {
                "balance_usd": balance_usd,
                "risk_ratio": risk_ratio,
                "risk_percentage": f"{risk_ratio*100:.2f}%",
                "tier": tier,
                "level": level,
                "max_position_usd": balance_usd * risk_ratio,
                "portfolio_risk_score": min(risk_ratio * 2000, 100)
            }
        
        # Risk metrics endpoint with dynamic scaling
        @app.get("/api/risk-metrics")
        async def get_risk_metrics():
            # Default balance for demo
            balance = 5000.0
            risk_data = calculate_dynamic_risk(balance)
            return {
                **risk_data,
                "message": "Dynamic risk scaling active"
            }
        
        # Dynamic risk calculation endpoint
        @app.get("/api/calculate-risk/{balance}")
        async def calculate_risk(balance: float):
            return calculate_dynamic_risk(balance)
        
        # Serve frontend if it exists
        frontend_dir = Path(__file__).parent.parent / "frontend"
        if frontend_dir.exists():
            app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
            
            @app.get("/")
            async def serve_frontend():
                return FileResponse(frontend_dir / "index.html")
        else:
            @app.get("/")
            async def root():
                return {"message": "Bybit Trading Bot API", "docs": "/docs"}
        
        logger.info("[OK] FastAPI application configured")
        logger.info("[OK] Starting server on http://0.0.0.0:8080")
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            log_config=None,  # Use our logging
            access_log=False
        )
        
    except ImportError as e:
        logger.error(f"[ERROR] Import failed: {e}")
        logger.error("Please install missing dependencies: pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Application failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()