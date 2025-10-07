"""
Strategy Graduation API

This module provides REST API endpoints for monitoring and controlling the
strategy graduation system, allowing external management of strategy lifecycle.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from ..bot.strategy_graduation import (
    StrategyStage,
    PerformanceMetrics,
    GraduationCriteria,
    StrategyRecord,
    StrategyGraduationManager
)

# ===== API MODELS =====

class StrategyStageAPI(str, Enum):
    """API representation of strategy stages"""
    RESEARCH = "research"
    PAPER_VALIDATION = "paper_validation"
    LIVE_CANDIDATE = "live_candidate" 
    LIVE_TRADING = "live_trading"
    UNDER_REVIEW = "under_review"
    RETIRED = "retired"

class PerformanceMetricsAPI(BaseModel):
    """API model for performance metrics"""
    total_return: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    volatility: float = Field(..., description="Volatility percentage")
    win_rate: float = Field(..., description="Win rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    trades_count: int = Field(..., description="Number of trades")
    execution_success_rate: float = Field(..., description="Execution success rate")
    validation_score: float = Field(..., description="Validation score (0-1)")
    confidence_level: str = Field(..., description="Confidence level")
    timestamp: datetime = Field(default_factory=datetime.now)

class StrategyRecordAPI(BaseModel):
    """API model for strategy record"""
    strategy_id: str
    name: str
    current_stage: StrategyStageAPI
    created_at: datetime
    last_updated: datetime
    allocated_capital: float
    performance_history: List[PerformanceMetricsAPI]
    graduation_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class StrategyRegistrationRequest(BaseModel):
    """Request model for strategy registration"""
    strategy_id: str = Field(..., description="Unique strategy identifier")
    name: str = Field(..., description="Strategy name")
    config: Dict[str, Any] = Field(..., description="Strategy configuration")
    start_in_paper: bool = Field(True, description="Start in paper trading")
    initial_capital: Optional[float] = Field(None, description="Initial capital allocation")

class ManualGraduationRequest(BaseModel):
    """Request model for manual strategy graduation"""
    strategy_id: str = Field(..., description="Strategy ID to graduate")
    target_stage: StrategyStageAPI = Field(..., description="Target stage")
    reason: str = Field(..., description="Reason for manual graduation")
    force: bool = Field(False, description="Force graduation ignoring criteria")

class GraduationCriteriaUpdateRequest(BaseModel):
    """Request model for updating graduation criteria"""
    stage: StrategyStageAPI = Field(..., description="Strategy stage")
    criteria: Dict[str, Any] = Field(..., description="New criteria values")

# ===== API ROUTER =====

router = APIRouter(prefix="/graduation", tags=["Strategy Graduation"])

# ===== DEPENDENCY INJECTION =====

async def get_graduation_manager() -> StrategyGraduationManager:
    """Dependency to get graduation manager instance"""
    # This would be injected from the main bot instance
    # For now, we'll assume it's available as a singleton
    from ..bot.integrated_trading_bot import IntegratedTradingBot
    
    # You would implement a proper dependency injection pattern here
    bot_instance = IntegratedTradingBot.get_instance()  # hypothetical method
    if not bot_instance or not bot_instance.graduation_manager:
        raise HTTPException(
            status_code=503,
            detail="Strategy graduation system not available"
        )
    
    return bot_instance.graduation_manager

# ===== API ENDPOINTS =====

@router.get("/status", response_model=Dict[str, Any])
async def get_graduation_system_status(
    manager: StrategyGraduationManager = Depends(get_graduation_manager)
):
    """Get overall graduation system status"""
    
    try:
        report = manager.get_graduation_report()
        
        return {
            "status": "active",
            "summary": report["summary"],
            "last_evaluation": report.get("last_evaluation"),
            "system_health": "healthy"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

@router.get("/strategies", response_model=List[StrategyRecordAPI])
async def list_all_strategies(
    stage: Optional[StrategyStageAPI] = None,
    manager: StrategyGraduationManager = Depends(get_graduation_manager)
):
    """List all strategies, optionally filtered by stage"""
    
    try:
        strategies = []
        
        for strategy_id, record in manager.strategies.items():
            # Filter by stage if specified
            if stage and record.current_stage.value != stage.value:
                continue
            
            # Convert to API model
            api_record = StrategyRecordAPI(
                strategy_id=record.strategy_id,
                name=record.name,
                current_stage=StrategyStageAPI(record.current_stage.value),
                created_at=record.created_at,
                last_updated=record.last_updated,
                allocated_capital=record.allocated_capital,
                performance_history=[
                    PerformanceMetricsAPI(**metrics.__dict__)
                    for metrics in record.performance_history[-10:]  # Last 10 snapshots
                ],
                graduation_history=record.graduation_history[-5:],  # Last 5 graduations
                metadata=record.metadata
            )
            
            strategies.append(api_record)
        
        return strategies
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing strategies: {str(e)}")

@router.get("/strategies/{strategy_id}", response_model=StrategyRecordAPI)
async def get_strategy_details(
    strategy_id: str,
    manager: StrategyGraduationManager = Depends(get_graduation_manager)
):
    """Get detailed information about a specific strategy"""
    
    if strategy_id not in manager.strategies:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    
    try:
        record = manager.strategies[strategy_id]
        
        return StrategyRecordAPI(
            strategy_id=record.strategy_id,
            name=record.name,
            current_stage=StrategyStageAPI(record.current_stage.value),
            created_at=record.created_at,
            last_updated=record.last_updated,
            allocated_capital=record.allocated_capital,
            performance_history=[
                PerformanceMetricsAPI(**metrics.__dict__)
                for metrics in record.performance_history
            ],
            graduation_history=record.graduation_history,
            metadata=record.metadata
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting strategy details: {str(e)}")

@router.post("/strategies", response_model=Dict[str, str])
async def register_strategy(
    request: StrategyRegistrationRequest,
    manager: StrategyGraduationManager = Depends(get_graduation_manager)
):
    """Register a new strategy for graduation tracking"""
    
    if request.strategy_id in manager.strategies:
        raise HTTPException(
            status_code=409,
            detail=f"Strategy {request.strategy_id} already exists"
        )
    
    try:
        initial_stage = StrategyStage.PAPER_VALIDATION if request.start_in_paper else StrategyStage.LIVE_TRADING
        
        record = manager.register_strategy(
            strategy_id=request.strategy_id,
            name=request.name,
            config=request.config,
            initial_stage=initial_stage
        )
        
        if request.initial_capital:
            record.allocated_capital = request.initial_capital
        
        return {
            "message": f"Strategy {request.name} registered successfully",
            "strategy_id": request.strategy_id,
            "initial_stage": initial_stage.value
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering strategy: {str(e)}")

@router.post("/strategies/{strategy_id}/graduate", response_model=Dict[str, Any])
async def manual_graduation(
    strategy_id: str,
    request: ManualGraduationRequest,
    manager: StrategyGraduationManager = Depends(get_graduation_manager)
):
    """Manually graduate a strategy to a different stage"""
    
    if strategy_id not in manager.strategies:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    
    try:
        record = manager.strategies[strategy_id]
        
        # Convert API stage to internal stage
        target_stage = StrategyStage(request.target_stage.value)
        
        # Perform manual graduation
        success = manager.manual_graduation(
            strategy_id=strategy_id,
            target_stage=target_stage,
            reason=request.reason,
            force=request.force
        )
        
        if not success:
            raise HTTPException(
                status_code=422,
                detail="Manual graduation failed - criteria not met"
            )
        
        return {
            "message": f"Strategy {strategy_id} graduated to {target_stage.value}",
            "previous_stage": record.current_stage.value,
            "new_stage": target_stage.value,
            "reason": request.reason
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during manual graduation: {str(e)}")

@router.post("/evaluation", response_model=Dict[str, Any])
async def trigger_graduation_evaluation(
    manager: StrategyGraduationManager = Depends(get_graduation_manager)
):
    """Trigger immediate graduation evaluation for all strategies"""
    
    try:
        decisions = await manager.evaluate_all_strategies()
        
        return {
            "message": "Graduation evaluation completed",
            "strategies_evaluated": len(decisions),
            "decisions": {
                strategy_id: {
                    "action": decision.action.value,
                    "new_stage": decision.new_stage.value if decision.new_stage else None,
                    "reason": decision.reason
                }
                for strategy_id, decision in decisions.items()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")

@router.get("/criteria", response_model=Dict[str, Any])
async def get_graduation_criteria(
    manager: StrategyGraduationManager = Depends(get_graduation_manager)
):
    """Get current graduation criteria for all stages"""
    
    try:
        criteria_dict = {}
        
        for stage in StrategyStage:
            criteria = manager.criteria.get(stage)
            if criteria:
                criteria_dict[stage.value] = {
                    "min_trades": criteria.min_trades,
                    "min_sharpe_ratio": criteria.min_sharpe_ratio,
                    "max_drawdown": criteria.max_drawdown,
                    "min_validation_score": criteria.min_validation_score,
                    "min_win_rate": criteria.min_win_rate,
                    "min_profit_factor": criteria.min_profit_factor,
                    "observation_period_days": criteria.observation_period_days,
                    "required_confidence": criteria.required_confidence.value,
                    "capital_allocation_pct": criteria.capital_allocation_pct,
                    "max_risk_per_trade": criteria.max_risk_per_trade
                }
        
        return criteria_dict
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting criteria: {str(e)}")

@router.put("/criteria", response_model=Dict[str, str])
async def update_graduation_criteria(
    request: GraduationCriteriaUpdateRequest,
    manager: StrategyGraduationManager = Depends(get_graduation_manager)
):
    """Update graduation criteria for a specific stage"""
    
    try:
        stage = StrategyStage(request.stage.value)
        
        if stage not in manager.criteria:
            raise HTTPException(status_code=404, detail=f"Criteria for stage {stage.value} not found")
        
        criteria = manager.criteria[stage]
        
        # Update criteria fields
        for field, value in request.criteria.items():
            if hasattr(criteria, field):
                setattr(criteria, field, value)
            else:
                raise HTTPException(status_code=400, detail=f"Invalid criteria field: {field}")
        
        return {
            "message": f"Graduation criteria updated for stage {stage.value}",
            "stage": stage.value
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating criteria: {str(e)}")

@router.get("/report", response_model=Dict[str, Any])
async def get_graduation_report(
    include_details: bool = False,
    manager: StrategyGraduationManager = Depends(get_graduation_manager)
):
    """Get comprehensive graduation system report"""
    
    try:
        report = manager.get_graduation_report()
        
        if not include_details:
            # Return summary only
            return {
                "summary": report["summary"],
                "last_evaluation": report.get("last_evaluation"),
                "system_status": "active"
            }
        
        return report
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@router.delete("/strategies/{strategy_id}", response_model=Dict[str, str])
async def retire_strategy(
    strategy_id: str,
    reason: str = "Manual retirement",
    manager: StrategyGraduationManager = Depends(get_graduation_manager)
):
    """Retire a strategy permanently"""
    
    if strategy_id not in manager.strategies:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    
    try:
        success = manager.retire_strategy(strategy_id, reason)
        
        if not success:
            raise HTTPException(status_code=422, detail="Failed to retire strategy")
        
        return {
            "message": f"Strategy {strategy_id} retired successfully",
            "reason": reason
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retiring strategy: {str(e)}")

# ===== WEBSOCKET ENDPOINTS =====

from fastapi import WebSocket, WebSocketDisconnect

@router.websocket("/ws/status")
async def websocket_graduation_status(websocket: WebSocket):
    """WebSocket endpoint for real-time graduation system status"""
    
    await websocket.accept()
    
    try:
        manager = await get_graduation_manager()
        
        while True:
            # Send current status every 30 seconds
            report = manager.get_graduation_report()
            
            status_update = {
                "timestamp": datetime.now().isoformat(),
                "summary": report["summary"],
                "recent_decisions": report.get("recent_decisions", [])[-5:],  # Last 5 decisions
                "system_status": "active"
            }
            
            await websocket.send_json(status_update)
            await asyncio.sleep(30)
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "error": f"WebSocket error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        await websocket.close()

# ===== HEALTH CHECK =====

@router.get("/health")
async def health_check():
    """Health check endpoint for graduation system"""
    
    try:
        manager = await get_graduation_manager()
        
        return {
            "status": "healthy",
            "service": "strategy_graduation",
            "timestamp": datetime.now().isoformat(),
            "strategies_count": len(manager.strategies)
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "strategy_graduation", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }