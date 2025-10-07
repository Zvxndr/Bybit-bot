"""
Dashboard Integration Bridge
===========================

Simplified integration for the unified dashboard that works with existing structure.
This provides the API endpoints needed by the dashboard using available components.

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import logging
import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)

class DashboardIntegration:
    """
    Simplified integration layer for the unified dashboard.
    
    This provides the API endpoints needed by the dashboard using a simplified
    approach that works with the current setup.
    """
    
    def __init__(self, app: FastAPI):
        self.app = app
        
        # Database path
        self.db_path = "data/trading_bot.db"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Setup dashboard-specific endpoints
        self.setup_dashboard_routes()
        
        logger.info("🚀 Dashboard integration initialized with existing backend")
    
    def setup_dashboard_routes(self):
        """Setup unified dashboard API routes that bridge to existing systems"""
        
        @self.app.get("/")
        async def serve_dashboard():
            """Serve the unified dashboard HTML"""
            return FileResponse("frontend/unified_dashboard.html")
        
        @self.app.get("/api/portfolio")
        async def get_portfolio():
            """Get portfolio data from trading bot"""
            try:
                portfolio_data = await self.trading_bot.get_portfolio_summary()
                return JSONResponse({
                    "success": True,
                    "data": portfolio_data
                })
            except Exception as e:
                logger.error(f"Portfolio fetch error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/strategies")
        async def get_strategies():
            """Get strategies from pipeline system"""
            try:
                with self.db_manager.get_session() as session:
                    # Get active strategies from pipeline
                    strategies = session.query(StrategyPipeline).filter(
                        StrategyPipeline.is_active == True
                    ).all()
                    
                    strategy_data = []
                    for strategy in strategies:
                        strategy_info = {
                            "id": strategy.strategy_id,
                            "name": strategy.strategy_name,
                            "asset": strategy.asset_pair,
                            "phase": strategy.current_phase,
                            "performance": {
                                "backtest_score": strategy.backtest_score,
                                "paper_pnl": strategy.paper_pnl,
                                "live_pnl": strategy.live_pnl,
                                "sharpe_ratio": strategy.sharpe_ratio,
                                "win_rate": strategy.win_rate
                            },
                            "metrics": {
                                "phase_duration": strategy.current_phase_duration_hours,
                                "trade_count": strategy.paper_trade_count + strategy.live_trade_count,
                                "ready_for_promotion": strategy.ready_for_promotion(),
                                "ready_for_graduation": strategy.ready_for_graduation()
                            }
                        }
                        strategy_data.append(strategy_info)
                
                return JSONResponse({
                    "success": True,
                    "data": strategy_data
                })
            except Exception as e:
                logger.error(f"Strategies fetch error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/pipeline-metrics")
        async def get_pipeline_metrics():
            """Get pipeline metrics - bridge to existing pipeline API"""
            try:
                # Use existing pipeline manager metrics
                metrics = self.pipeline_manager.get_current_metrics()
                
                # Get phase counts from database
                with self.db_manager.get_session() as session:
                    backtest_count = session.query(StrategyPipeline).filter(
                        StrategyPipeline.current_phase == 'backtest',
                        StrategyPipeline.is_active == True
                    ).count()
                    
                    paper_count = session.query(StrategyPipeline).filter(
                        StrategyPipeline.current_phase == 'paper',
                        StrategyPipeline.is_active == True
                    ).count()
                    
                    live_count = session.query(StrategyPipeline).filter(
                        StrategyPipeline.current_phase == 'live',
                        StrategyPipeline.is_active == True
                    ).count()
                
                return JSONResponse({
                    "success": True,
                    "data": {
                        "tested_today": metrics.strategies_tested_today,
                        "candidates_found": metrics.candidates_found_today,
                        "success_rate": round(metrics.success_rate_pct, 1),
                        "graduation_rate": round(getattr(metrics, 'graduation_rate_pct', 0), 1),
                        "backtest_count": backtest_count,
                        "paper_count": paper_count,
                        "live_count": live_count,
                        "total_live_pnl": round(getattr(metrics, 'total_live_pnl', 0), 2),
                        "last_updated": datetime.now().isoformat()
                    }
                })
            except Exception as e:
                logger.error(f"Pipeline metrics error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/performance")
        async def get_performance():
            """Get performance metrics from database"""
            try:
                with self.db_manager.get_session() as session:
                    # Get recent performance data
                    recent_performance = session.query(StrategyPerformance).filter(
                        StrategyPerformance.timestamp >= datetime.now() - timedelta(days=30)
                    ).order_by(StrategyPerformance.timestamp.desc()).limit(100).all()
                    
                    performance_data = []
                    for perf in recent_performance:
                        performance_data.append({
                            "timestamp": perf.timestamp.isoformat(),
                            "strategy_id": perf.strategy_id,
                            "returns": perf.returns,
                            "sharpe_ratio": perf.sharpe_ratio,
                            "drawdown": perf.current_drawdown,
                            "equity": perf.equity
                        })
                
                return JSONResponse({
                    "success": True,
                    "data": performance_data
                })
            except Exception as e:
                logger.error(f"Performance fetch error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/activity")
        async def get_activity():
            """Get recent trading activity"""
            try:
                with self.db_manager.get_session() as session:
                    # Get recent trades
                    recent_trades = session.query(Trade).filter(
                        Trade.timestamp >= datetime.now() - timedelta(days=7)
                    ).order_by(Trade.timestamp.desc()).limit(50).all()
                    
                    activity_data = []
                    for trade in recent_trades:
                        activity_data.append({
                            "timestamp": trade.timestamp.isoformat(),
                            "type": "trade",
                            "action": trade.side,
                            "symbol": trade.symbol,
                            "amount": trade.amount,
                            "price": trade.price,
                            "strategy_id": trade.strategy_id,
                            "pnl": trade.realized_pnl
                        })
                
                return JSONResponse({
                    "success": True,
                    "data": activity_data
                })
            except Exception as e:
                logger.error(f"Activity fetch error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/emergency-stop")
        async def emergency_stop():
            """Emergency stop all trading operations"""
            try:
                # Use existing trading bot emergency stop
                await self.trading_bot.emergency_stop()
                
                # Notify all connected WebSocket clients
                await self.broadcast_message({
                    "type": "emergency_stop",
                    "message": "Emergency stop activated",
                    "timestamp": datetime.now().isoformat()
                })
                
                return JSONResponse({
                    "success": True,
                    "message": "Emergency stop activated"
                })
            except Exception as e:
                logger.error(f"Emergency stop error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time dashboard updates"""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                # Start sending periodic updates
                await self.send_periodic_updates(websocket)
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
    
    async def send_periodic_updates(self, websocket: WebSocket):
        """Send periodic updates to WebSocket client"""
        while True:
            try:
                # Get current metrics
                metrics = self.pipeline_manager.get_current_metrics()
                
                # Send update
                await websocket.send_json({
                    "type": "pipeline_update",
                    "data": {
                        "tested_today": metrics.strategies_tested_today,
                        "candidates_found": metrics.candidates_found_today,
                        "success_rate": round(metrics.success_rate_pct, 1),
                        "backtest_count": metrics.backtest_count,
                        "paper_count": metrics.paper_count,
                        "live_count": metrics.live_count,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
                # Wait 30 seconds before next update
                await asyncio.sleep(30)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket update error: {e}")
                break
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients"""
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except WebSocketDisconnect:
                    disconnected.append(connection)
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)
    
    async def start_background_tasks(self):
        """Start background tasks for pipeline automation"""
        try:
            # Start the automated pipeline manager
            await self.pipeline_manager.start()
            logger.info("✅ Automated pipeline manager started")
            
            # Start ML discovery engine
            await self.ml_engine.start_discovery_loop()
            logger.info("✅ ML strategy discovery engine started")
            
        except Exception as e:
            logger.error(f"Background tasks startup error: {e}")
    
    async def shutdown(self):
        """Shutdown all systems gracefully"""
        try:
            # Stop pipeline manager
            await self.pipeline_manager.stop()
            
            # Close database connections
            await self.db_manager.close_all()
            
            # Close WebSocket connections
            for connection in self.active_connections:
                await connection.close()
            
            logger.info("🛑 Dashboard integration shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")