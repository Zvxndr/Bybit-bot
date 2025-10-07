"""
High-Frequency Trading Module - Phase 6 Implementation

This package provides ultra-low latency trading capabilities:
- Latency Engine: Microsecond-level latency optimization and monitoring
- Market Making Engine: Advanced market making algorithms with dynamic spread management
- Arbitrage Detection: Cross-market and statistical arbitrage identification
- Order Flow Analysis: Real-time order book analysis and flow prediction
- Execution Engine: Ultra-fast order execution with smart routing
- Risk Controls: Real-time risk monitoring and position limits for HFT

Author: Trading Bot Team
Version: 1.0.0 - Phase 6 HFT Module
"""

from .latency_engine import (
    LatencyEngine,
    LatencyMetrics,
    LatencyOptimizer,
    NetworkMonitor
)
from .market_making_engine import (
    MarketMakingEngine,
    MarketMakingStrategy,
    SpreadManager,
    InventoryManager,
    QuoteManager
)
from .arbitrage_detector import (
    ArbitrageDetector,
    ArbitrageOpportunity,
    ArbitrageType,
    CrossExchangeArbitrage,
    StatisticalArbitrage
)
from .order_flow_analyzer import (
    OrderFlowAnalyzer,
    OrderBookAnalyzer,
    FlowPredictor,
    MarketMicrostructure
)
from .execution_engine import (
    HFTExecutionEngine,
    ExecutionAlgorithm,
    SmartOrderRouter,
    FillPredictor
)
from .hft_risk_manager import (
    HFTRiskManager,
    HFTRiskLimits,
    PositionMonitor,
    DrawdownProtection
)

__all__ = [
    # Latency optimization
    'LatencyEngine',
    'LatencyMetrics',
    'LatencyOptimizer',
    'NetworkMonitor',
    
    # Market making
    'MarketMakingEngine',
    'MarketMakingStrategy',
    'SpreadManager',
    'InventoryManager',
    'QuoteManager',
    
    # Arbitrage detection
    'ArbitrageDetector',
    'ArbitrageOpportunity',
    'ArbitrageType',
    'CrossExchangeArbitrage',
    'StatisticalArbitrage',
    
    # Order flow analysis
    'OrderFlowAnalyzer',
    'OrderBookAnalyzer',
    'FlowPredictor',
    'MarketMicrostructure',
    
    # Execution
    'HFTExecutionEngine',
    'ExecutionAlgorithm',
    'SmartOrderRouter',
    'FillPredictor',
    
    # Risk management
    'HFTRiskManager',
    'HFTRiskLimits',
    'PositionMonitor',
    'DrawdownProtection',
]