"""
Bybit Trading Bot v2.0 - Phase 3 Dashboard Backend
Advanced FastAPI application for real-time trading dashboard

Features:
- Real-time WebSocket streaming
- Phase 1-2 ML component integration
- Advanced analytics API
- System health monitoring
- High-performance data serving
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uvicorn
from contextlib import asynccontextmanager

# Dashboard-specific imports
from routers import trading_router, analytics_router, health_router, ml_router
from websocket import WebSocketManager
from database import DatabaseManager
from integration import Phase1Integration, Phase2Integration
from monitoring import PerformanceMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardBackend:
    """Main dashboard backend application"""
    
    def __init__(self):
        self.app = None
        self.websocket_manager = WebSocketManager()
        self.db_manager = DatabaseManager()
        self.phase1_integration = Phase1Integration()
        self.phase2_integration = Phase2Integration()
        self.performance_monitor = PerformanceMonitor()
        self._running = False
    
    async def startup(self):
        """Initialize dashboard backend services"""
        try:
            logger.info("🚀 Starting Phase 3 Dashboard Backend...")
            
            # Initialize database connections
            await self.db_manager.initialize()
            logger.info("✅ Database connections established")
            
            # Initialize Phase 1-2 integrations
            await self.phase1_integration.initialize()
            await self.phase2_integration.initialize()
            logger.info("✅ ML component integrations ready")
            
            # Start performance monitoring
            await self.performance_monitor.start()
            logger.info("✅ Performance monitoring active")
            
            # Start data streaming tasks
            asyncio.create_task(self._start_data_streams())
            logger.info("✅ Real-time data streaming started")
            
            self._running = True
            logger.info("🎯 Dashboard backend fully operational!")
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            raise
    
    async def shutdown(self):
        """Cleanup dashboard backend services"""
        logger.info("🔄 Shutting down dashboard backend...")
        self._running = False
        
        await self.performance_monitor.stop()
        await self.db_manager.close()
        await self.websocket_manager.disconnect_all()
        
        logger.info("✅ Dashboard backend shutdown complete")
    
    async def _start_data_streams(self):
        """Start real-time data streaming tasks"""
        tasks = [
            self._stream_trading_data(),
            self._stream_ml_insights(),
            self._stream_system_health(),
            self._stream_performance_metrics()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _stream_trading_data(self):
        """Stream real-time trading data to connected clients"""
        while self._running:
            try:
                # Get latest trading data from Phase 1 components
                trading_data = await self.phase1_integration.get_real_time_data()
                
                # Broadcast to all connected WebSocket clients
                await self.websocket_manager.broadcast("trading_update", trading_data)
                
                await asyncio.sleep(0.1)  # 10 updates per second
                
            except Exception as e:
                logger.error(f"Trading data stream error: {e}")
                await asyncio.sleep(1)
    
    async def _stream_ml_insights(self):
        """Stream ML insights and predictions"""
        while self._running:
            try:
                # Get ML insights from Phase 2 components
                ml_insights = await self.phase2_integration.get_real_time_insights()
                
                # Broadcast ML insights
                await self.websocket_manager.broadcast("ml_insights", ml_insights)
                
                await asyncio.sleep(0.5)  # 2 updates per second
                
            except Exception as e:
                logger.error(f"ML insights stream error: {e}")
                await asyncio.sleep(2)
    
    async def _stream_system_health(self):
        """Stream system health and monitoring data"""
        while self._running:
            try:
                # Get system health metrics
                health_data = await self.performance_monitor.get_health_snapshot()
                
                # Broadcast health updates
                await self.websocket_manager.broadcast("system_health", health_data)
                
                await asyncio.sleep(2)  # Every 2 seconds
                
            except Exception as e:
                logger.error(f"System health stream error: {e}")
                await asyncio.sleep(5)
    
    async def _stream_performance_metrics(self):
        """Stream performance analytics"""
        while self._running:
            try:
                # Get performance metrics
                performance_data = await self.performance_monitor.get_performance_metrics()
                
                # Broadcast performance updates
                await self.websocket_manager.broadcast("performance_update", performance_data)
                
                await asyncio.sleep(1)  # Every second
                
            except Exception as e:
                logger.error(f"Performance stream error: {e}")
                await asyncio.sleep(3)

# Global dashboard instance
dashboard_backend = DashboardBackend()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management"""
    await dashboard_backend.startup()
    yield
    await dashboard_backend.shutdown()

# Create FastAPI application
app = FastAPI(
    title="Bybit Trading Bot v2.0 - Dashboard API",
    description="Advanced real-time trading dashboard with ML insights",
    version="3.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(trading_router.router, prefix="/api/trading", tags=["Trading"])
app.include_router(analytics_router.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(health_router.router, prefix="/api/health", tags=["Health"])
app.include_router(ml_router.router, prefix="/api/ml", tags=["Machine Learning"])

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Main WebSocket endpoint for real-time updates"""
    await dashboard_backend.websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client requests
            await dashboard_backend.websocket_manager.handle_client_message(
                client_id, message
            )
            
    except WebSocketDisconnect:
        await dashboard_backend.websocket_manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await dashboard_backend.websocket_manager.disconnect(client_id)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Bybit Trading Bot v2.0 Dashboard API",
        "version": "3.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "Real-time trading data streaming",
            "ML insights and predictions",
            "Advanced analytics API",
            "System health monitoring",
            "High-performance WebSocket updates"
        ],
        "endpoints": {
            "trading": "/api/trading",
            "analytics": "/api/analytics", 
            "health": "/api/health",
            "ml": "/api/ml",
            "websocket": "/ws/{client_id}"
        }
    }

@app.get("/api/status")
async def get_status():
    """Get comprehensive system status"""
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": dashboard_backend.performance_monitor.get_uptime(),
        "connected_clients": dashboard_backend.websocket_manager.get_connection_count(),
        "system_health": await dashboard_backend.performance_monitor.get_health_snapshot(),
        "phase1_status": await dashboard_backend.phase1_integration.get_status(),
        "phase2_status": await dashboard_backend.phase2_integration.get_status()
    }

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )