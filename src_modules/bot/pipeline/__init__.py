"""
AI Pipeline System Package

This package implements the automated three-column AI strategy discovery pipeline:
Backtest → Paper Trading → Live Trading

Components:
- StrategyNamingEngine: Generates unique strategy IDs (BTC_MR_A4F2D format)
- AutomatedPipelineManager: Orchestrates the complete pipeline
- StrategyPipeline: Database model for pipeline state tracking

The pipeline integrates with existing infrastructure:
- ML Strategy Discovery Engine for strategy generation
- Bybit Enhanced Backtest Engine for validation
- Historical Data Manager for data provisioning
- WebSocket system for real-time frontend updates

Author: Trading Bot Team
Version: 1.0.0
"""

from .strategy_naming_engine import (
    StrategyNamingEngine,
    StrategyName,
    StrategyTypeCode,
    strategy_naming_engine
)

from .automated_pipeline_manager import (
    AutomatedPipelineManager,
    PipelinePhase,
    PipelineConfig,
    PipelineMetrics,
    pipeline_manager
)

__all__ = [
    'StrategyNamingEngine',
    'StrategyName', 
    'StrategyTypeCode',
    'strategy_naming_engine',
    'AutomatedPipelineManager',
    'PipelinePhase',
    'PipelineConfig',
    'PipelineMetrics',
    'pipeline_manager'
]