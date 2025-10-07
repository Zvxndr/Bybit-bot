"""
AI Pipeline API
Provides endpoints for the automated strategy discovery pipeline system
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import json
import asyncio
from pydantic import BaseModel

from ..bot.pipeline import pipeline_manager, AutomatedPipelineManager, PipelineConfig
from ..bot.database.manager import DatabaseManager
from ..bot.database.models import StrategyPipeline, StrategyPerformance

logger = logging.getLogger(__name__)


class PipelineControlRequest(BaseModel):
    """Request model for pipeline control actions."""
    action: str  # 'start', 'stop', 'pause', 'resume'
    config: Optional[Dict[str, Any]] = None


class StrategyActionRequest(BaseModel):
    """Request model for manual strategy actions."""
    strategy_id: str
    action: str  # 'promote', 'reject', 'pause'
    reason: Optional[str] = None


class PipelineAPI:
    """API endpoints for AI pipeline management and monitoring"""
    
    def __init__(self, app: FastAPI, db_manager: DatabaseManager):
        self.app = app
        self.db_manager = db_manager
        self.pipeline_manager = pipeline_manager
        
        # WebSocket connection management
        self.active_connections: List[WebSocket] = []
        
        self.register_routes()
        self.setup_websocket_callbacks()
        
    def register_routes(self):
        """Register all pipeline API routes"""
        
        @self.app.get("/api/pipeline/status")
        async def get_pipeline_status():
            """Get current pipeline status and metrics"""
            try:
                status = await self.pipeline_manager.get_pipeline_status()
                return JSONResponse({
                    "status": "success",
                    "data": status
                })
                
            except Exception as e:
                logger.error(f"Failed to get pipeline status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/pipeline/metrics")
        async def get_pipeline_metrics():
            """Get real-time pipeline metrics"""
            try:
                metrics = self.pipeline_manager.get_current_metrics()
                
                return JSONResponse({
                    "status": "success",
                    "metrics": {
                        "tested_today": metrics.strategies_tested_today,
                        "candidates_found": metrics.candidates_found_today,
                        "success_rate": round(metrics.success_rate_pct, 1),
                        "graduation_rate": round(metrics.graduation_rate_pct, 1),
                        "backtest_count": metrics.backtest_count,
                        "paper_count": metrics.paper_count,
                        "live_count": metrics.live_count,
                        "total_live_pnl": round(metrics.total_live_pnl, 2),
                        "last_updated": metrics.last_updated.isoformat()
                    }
                })
                
            except Exception as e:
                logger.error(f"Failed to get pipeline metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/pipeline/strategies/{phase}")
        async def get_strategies_by_phase(phase: str):
            """Get strategies in a specific pipeline phase"""
            try:
                if phase not in ['backtest', 'paper', 'live', 'rejected']:
                    raise HTTPException(status_code=400, detail="Invalid phase")
                
                with self.db_manager.get_session() as session:
                    strategies = session.query(StrategyPipeline).filter(
                        StrategyPipeline.current_phase == phase,
                        StrategyPipeline.is_active == True if phase != 'rejected' else True
                    ).order_by(StrategyPipeline.created_at.desc()).limit(20).all()
                    
                    strategy_list = []
                    for strategy in strategies:
                        strategy_data = {
                            "strategy_id": strategy.strategy_id,
                            "strategy_name": strategy.strategy_name,
                            "asset_pair": strategy.asset_pair,
                            "base_asset": strategy.base_asset,
                            "strategy_type": strategy.strategy_type,
                            "current_phase": strategy.current_phase,
                            "created_at": strategy.created_at.isoformat(),
                            "phase_duration_hours": strategy.current_phase_duration_hours
                        }
                        
                        # Add phase-specific metrics
                        if phase == 'backtest':
                            strategy_data.update({
                                "backtest_score": strategy.backtest_score,
                                "backtest_return": strategy.backtest_return,
                                "sharpe_ratio": strategy.sharpe_ratio,
                                "max_drawdown": strategy.max_drawdown,
                                "win_rate": strategy.win_rate,
                                "ready_for_promotion": strategy.ready_for_promotion()
                            })
                        elif phase == 'paper':
                            strategy_data.update({
                                "paper_pnl": strategy.paper_pnl,
                                "paper_return_pct": strategy.paper_return_pct,
                                "paper_trade_count": strategy.paper_trade_count,
                                "paper_duration_hours": strategy.paper_duration_hours,
                                "ready_for_graduation": strategy.ready_for_graduation()
                            })
                        elif phase == 'live':
                            strategy_data.update({
                                "live_pnl": strategy.live_pnl,
                                "live_return_pct": strategy.live_return_pct,
                                "live_trade_count": strategy.live_trade_count,
                                "live_duration_hours": strategy.live_duration_hours
                            })
                        elif phase == 'rejected':
                            strategy_data.update({
                                "rejection_reason": strategy.rejection_reason,
                                "rejected_at": strategy.rejected_at.isoformat() if strategy.rejected_at else None
                            })
                        
                        strategy_list.append(strategy_data)
                
                return JSONResponse({
                    "status": "success",
                    "phase": phase,
                    "strategies": strategy_list,
                    "count": len(strategy_list)
                })
                
            except Exception as e:
                logger.error(f"Failed to get {phase} strategies: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/pipeline/control")
        async def control_pipeline(request: PipelineControlRequest):
            """Control pipeline operations (start/stop/pause)"""
            try:
                if request.action == 'start':
                    success = await self.pipeline_manager.start_pipeline()
                    message = "Pipeline started successfully" if success else "Failed to start pipeline"
                elif request.action == 'stop':
                    success = await self.pipeline_manager.stop_pipeline()
                    message = "Pipeline stopped successfully" if success else "Failed to stop pipeline"
                else:
                    raise HTTPException(status_code=400, detail="Invalid action")
                
                return JSONResponse({
                    "status": "success" if success else "error",
                    "message": message,
                    "action": request.action,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to control pipeline: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/pipeline/strategy-action")
        async def strategy_action(request: StrategyActionRequest):
            """Perform manual action on a strategy"""
            try:
                if request.action == 'promote':
                    success = await self.pipeline_manager.manual_promote(request.strategy_id)
                    message = f"Strategy {request.strategy_id} promoted successfully" if success else "Failed to promote strategy"
                elif request.action == 'reject':
                    success = await self.pipeline_manager.manual_reject(
                        request.strategy_id, 
                        request.reason or "Manual rejection"
                    )
                    message = f"Strategy {request.strategy_id} rejected" if success else "Failed to reject strategy"
                else:
                    raise HTTPException(status_code=400, detail="Invalid action")
                
                return JSONResponse({
                    "status": "success" if success else "error",
                    "message": message,
                    "strategy_id": request.strategy_id,
                    "action": request.action,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to perform strategy action: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/pipeline/strategy/{strategy_id}/details")
        async def get_strategy_details(strategy_id: str):
            """Get detailed information about a specific strategy"""
            try:
                with self.db_manager.get_session() as session:
                    strategy = session.query(StrategyPipeline).filter(
                        StrategyPipeline.strategy_id == strategy_id
                    ).first()
                    
                    if not strategy:
                        raise HTTPException(status_code=404, detail="Strategy not found")
                    
                    # Get performance history
                    performance_history = session.query(StrategyPerformance).filter(
                        StrategyPerformance.strategy_id == strategy_id
                    ).order_by(StrategyPerformance.timestamp.desc()).limit(100).all()
                    
                    performance_data = []
                    for perf in performance_history:
                        performance_data.append({
                            "timestamp": perf.timestamp.isoformat(),
                            "equity": perf.equity,
                            "returns": perf.returns,
                            "cumulative_returns": perf.cumulative_returns,
                            "sharpe_ratio": perf.sharpe_ratio,
                            "max_drawdown": perf.max_drawdown,
                            "win_rate": perf.win_rate
                        })
                    
                    strategy_details = {
                        "strategy_id": strategy.strategy_id,
                        "strategy_name": strategy.strategy_name,
                        "asset_pair": strategy.asset_pair,
                        "base_asset": strategy.base_asset,
                        "strategy_type": strategy.strategy_type,
                        "strategy_description": strategy.strategy_description,
                        "current_phase": strategy.current_phase,
                        "created_at": strategy.created_at.isoformat(),
                        
                        # Backtest metrics
                        "backtest_score": strategy.backtest_score,
                        "backtest_return": strategy.backtest_return,
                        "sharpe_ratio": strategy.sharpe_ratio,
                        "max_drawdown": strategy.max_drawdown,
                        "win_rate": strategy.win_rate,
                        "profit_factor": strategy.profit_factor,
                        
                        # Paper trading metrics
                        "paper_pnl": strategy.paper_pnl,
                        "paper_return_pct": strategy.paper_return_pct,
                        "paper_trade_count": strategy.paper_trade_count,
                        "paper_duration_hours": strategy.paper_duration_hours,
                        
                        # Live trading metrics
                        "live_pnl": strategy.live_pnl,
                        "live_return_pct": strategy.live_return_pct,
                        "live_trade_count": strategy.live_trade_count,
                        "live_duration_hours": strategy.live_duration_hours,
                        
                        # Status
                        "is_active": strategy.is_active,
                        "rejection_reason": strategy.rejection_reason,
                        
                        # Performance history
                        "performance_history": performance_data
                    }
                    
                    return JSONResponse({
                        "status": "success",
                        "strategy": strategy_details
                    })
                
            except Exception as e:
                logger.error(f"Failed to get strategy details: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/pipeline/analytics/overview")
        async def get_pipeline_analytics():
            """Get comprehensive pipeline analytics"""
            try:
                with self.db_manager.get_session() as session:
                    # Get asset distribution
                    asset_counts = session.query(
                        StrategyPipeline.base_asset,
                        session.query(StrategyPipeline).filter(
                            StrategyPipeline.base_asset == StrategyPipeline.base_asset
                        ).count().label('count')
                    ).filter(
                        StrategyPipeline.is_active == True
                    ).group_by(StrategyPipeline.base_asset).all()
                    
                    # Get type distribution  
                    type_counts = session.query(
                        StrategyPipeline.strategy_type,
                        session.query(StrategyPipeline).filter(
                            StrategyPipeline.strategy_type == StrategyPipeline.strategy_type
                        ).count().label('count')
                    ).filter(
                        StrategyPipeline.is_active == True
                    ).group_by(StrategyPipeline.strategy_type).all()
                    
                    # Get performance metrics
                    avg_metrics = session.query(
                        session.query(StrategyPipeline.backtest_score).filter(
                            StrategyPipeline.backtest_score.isnot(None)
                        ).func.avg().label('avg_backtest_score'),
                        session.query(StrategyPipeline.paper_return_pct).filter(
                            StrategyPipeline.paper_return_pct.isnot(None)
                        ).func.avg().label('avg_paper_return'),
                        session.query(StrategyPipeline.live_return_pct).filter(
                            StrategyPipeline.live_return_pct.isnot(None)
                        ).func.avg().label('avg_live_return')
                    ).first()
                    
                    analytics = {
                        "asset_distribution": {asset: count for asset, count in asset_counts},
                        "type_distribution": {stype: count for stype, count in type_counts},
                        "performance_averages": {
                            "avg_backtest_score": round(avg_metrics[0] or 0, 2),
                            "avg_paper_return": round(avg_metrics[1] or 0, 2),
                            "avg_live_return": round(avg_metrics[2] or 0, 2)
                        }
                    }
                    
                    return JSONResponse({
                        "status": "success",
                        "analytics": analytics
                    })
                
            except Exception as e:
                logger.error(f"Failed to get pipeline analytics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws/pipeline")
        async def pipeline_websocket(websocket: WebSocket):
            """WebSocket endpoint for real-time pipeline updates"""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(10)
                    
                    # Send current metrics
                    metrics = self.pipeline_manager.get_current_metrics()
                    update_data = {
                        "type": "metrics_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "tested_today": metrics.strategies_tested_today,
                            "candidates_found": metrics.candidates_found_today,
                            "success_rate": metrics.success_rate_pct,
                            "graduation_rate": metrics.graduation_rate_pct,
                            "backtest_count": metrics.backtest_count,
                            "paper_count": metrics.paper_count,
                            "live_count": metrics.live_count,
                            "total_live_pnl": metrics.total_live_pnl
                        }
                    }
                    
                    await websocket.send_text(json.dumps(update_data))
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
    
    def setup_websocket_callbacks(self):
        """Setup callbacks for pipeline events to broadcast via WebSocket"""
        
        async def websocket_callback(event_data):
            """Callback to broadcast pipeline events to WebSocket clients"""
            if not self.active_connections:
                return
            
            # Prepare message
            message = json.dumps(event_data)
            
            # Broadcast to all connected clients
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.warning(f"Failed to send WebSocket message: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for connection in disconnected:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)
        
        # Register callback with pipeline manager
        self.pipeline_manager.register_update_callback(websocket_callback)