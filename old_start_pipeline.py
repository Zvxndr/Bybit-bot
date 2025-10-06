"""
AI Pipeline System Startup Script

This script starts the complete AI Pipeline system with:
- FastAPI server with pipeline endpoints
- WebSocket real-time updates
- Automated strategy discovery
- Three-column pipeline management

Run this to start the complete system.
"""

import asyncio
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import the pipeline integration
from src.bot.pipeline_integration import integrate_pipeline_api, initialize_pipeline_system
from src.bot.pipeline import pipeline_manager
from src.bot.database.manager import DatabaseManager


def create_app():
    """Create FastAPI application with pipeline integration."""
    
    # Create FastAPI app
    app = FastAPI(
        title="Bybit AI Trading Bot - Pipeline System",
        description="Automated AI strategy discovery and trading pipeline",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify your frontend domains
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Serve static files (for the dashboard)
    try:
        app.mount("/static", StaticFiles(directory="src/static"), name="static")
    except Exception:
        logging.warning("Static files directory not found")
    
    # Integrate pipeline API
    pipeline_api = integrate_pipeline_api(app)
    
    # Add dashboard route
    @app.get("/")
    async def get_dashboard():
        """Serve the main dashboard."""
        try:
            return FileResponse("src/templates/adminlte_dashboard.html")
        except Exception:
            return {"message": "Dashboard not found. API endpoints are available."}
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "pipeline_running": pipeline_manager.is_running,
            "timestamp": pipeline_manager.current_metrics.last_updated.isoformat()
        }
    
    # Startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Initialize pipeline system on startup."""
        logging.info("🚀 Starting AI Pipeline System...")
        
        try:
            # Initialize database
            db_manager = DatabaseManager()
            await db_manager.initialize()
            
            # Start pipeline system
            success = await initialize_pipeline_system()
            
            if success:
                logging.info("✅ AI Pipeline System started successfully")
                logging.info("🌐 Dashboard available at: http://localhost:8000")
                logging.info("📊 API documentation at: http://localhost:8000/docs")
            else:
                logging.error("❌ Failed to start AI Pipeline System")
                
        except Exception as e:
            logging.error(f"❌ Startup error: {e}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logging.info("⏹️ Shutting down AI Pipeline System...")
        
        try:
            await pipeline_manager.stop_pipeline()
            logging.info("✅ AI Pipeline System stopped")
        except Exception as e:
            logging.error(f"❌ Shutdown error: {e}")
    
    return app


def main():
    """Main function to start the server."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🤖 Bybit AI Trading Bot - Pipeline System")
    print("=" * 50)
    print("🎯 Starting automated strategy discovery pipeline...")
    print("📊 Three-column pipeline: Backtest → Paper → Live")
    print("🌐 Real-time WebSocket updates")
    print("💹 USDT cryptocurrency trading pairs")
    print()
    
    # Create the application
    app = create_app()
    
    # Run the server
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,  # Set to True for development
        access_log=True
    )
    
    server = uvicorn.Server(config)
    
    print("🚀 Starting server...")
    print("🌐 Dashboard: http://localhost:8000")
    print("📊 API Docs: http://localhost:8000/docs")
    print("🔌 WebSocket: ws://localhost:8000/ws/pipeline")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")


if __name__ == "__main__":
    main()