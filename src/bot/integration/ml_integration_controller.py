"""
ML Integration Controller - Main ML Trading System Controller

This is the main controller that orchestrates all ML components, providing a
unified interface for ML-enhanced trading operations. It coordinates between
the feature pipeline, model manager, strategy orchestrator, execution optimizer,
and performance monitor to create a complete ML trading system.

Key Features:
- Unified ML trading system coordination
- Real-time market analysis and signal generation
- Automated trade execution with ML optimization
- Continuous performance monitoring and adaptation
- Risk management integration
- System health monitoring and alerting
- Configuration management and hot-reloading
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass
import json
from pathlib import Path

# Import ML integration components
from .ml_feature_pipeline import MLFeaturePipeline, MLFeatures
from .ml_model_manager import MLModelManager, EnsemblePrediction
from .ml_strategy_orchestrator import MLStrategyOrchestrator, CombinedSignal
from .ml_execution_optimizer import MLExecutionOptimizer, ExecutionPlan
from .ml_performance_monitor import MLPerformanceMonitor, PerformanceReport

# Import risk management
try:
    from ..risk.core.unified_risk_manager import UnifiedRiskManager
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("Unified risk manager not available")
    RISK_MANAGER_AVAILABLE = False

# Import unified configuration system
try:
    from ..core.config.manager import UnifiedConfigurationManager
    from ..core.config.schema import UnifiedConfigurationSchema
    from ..core.config.integrations import MLIntegrationConfigAdapter
    UNIFIED_CONFIG_AVAILABLE = True
except ImportError:
    logger.warning("Unified configuration system not available")
    UNIFIED_CONFIG_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class SystemStatus(Enum):
    """ML system status"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class TradingMode(Enum):
    """Trading mode options"""
    FULLY_AUTOMATED = "fully_automated"
    SEMI_AUTOMATED = "semi_automated"  # Human approval required
    ADVISORY_ONLY = "advisory_only"    # Signals only, no execution
    BACKTESTING = "backtesting"
    PAPER_TRADING = "paper_trading"

@dataclass
class TradingDecision:
    """Complete trading decision from ML system"""
    decision_id: str
    symbol: str
    signal: CombinedSignal
    execution_plan: Optional[ExecutionPlan]
    risk_assessment: Dict[str, float]
    confidence_score: float
    recommended_action: str
    position_size: Decimal
    expected_outcome: Dict[str, float]
    timestamp: datetime

@dataclass
class SystemHealthMetrics:
    """Overall system health metrics"""
    status: SystemStatus
    uptime: timedelta
    feature_pipeline_health: float  # 0-1 score
    model_manager_health: float
    strategy_orchestrator_health: float
    execution_optimizer_health: float
    performance_monitor_health: float
    overall_health_score: float
    active_alerts: int
    last_decision_time: datetime
    decisions_per_hour: float
    error_rate: float

# ============================================================================
# ML INTEGRATION CONTROLLER
# ============================================================================

class MLIntegrationController:
    """
    Main ML Integration Controller
    
    Coordinates all ML components to provide unified ML-enhanced trading
    """
    
    def __init__(self, config_path: Optional[str] = None, unified_config: 'UnifiedConfigurationSchema' = None):
        # Load configuration (unified config takes precedence)
        if unified_config and UNIFIED_CONFIG_AVAILABLE:
            self.unified_config = unified_config
            self.config = self._load_unified_config(unified_config)
            logger.info("ML Integration Controller initialized with unified configuration")
        else:
            self.unified_config = None
            self.config = self._load_config(config_path)
            logger.info("ML Integration Controller initialized with file/default configuration")
        
        # System status
        self.status = SystemStatus.INITIALIZING
        self.start_time = datetime.now()
        self.last_health_check = datetime.now()
        
        # ML Components
        self.feature_pipeline: Optional[MLFeaturePipeline] = None
        self.model_manager: Optional[MLModelManager] = None
        self.strategy_orchestrator: Optional[MLStrategyOrchestrator] = None
        self.execution_optimizer: Optional[MLExecutionOptimizer] = None
        self.performance_monitor: Optional[MLPerformanceMonitor] = None
        
        # Risk management
        self.risk_manager: Optional[UnifiedRiskManager] = None
        
        # Decision tracking
        self.recent_decisions: List[TradingDecision] = []
        self.decision_history: Dict[str, Any] = {}
        
        # System metrics
        self.health_metrics: Optional[SystemHealthMetrics] = None
        self.error_count = 0
        self.last_error_time: Optional[datetime] = None
        
        # Trading mode
        self.trading_mode = TradingMode(self.config.get('trading_mode', 'paper_trading'))
        
        # Initialize the system
        asyncio.create_task(self._initialize_system())
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        
        # Default configuration
        default_config = {
            'trading_mode': 'paper_trading',
            'system': {
                'max_concurrent_decisions': 5,
                'decision_cooldown_seconds': 30,
                'health_check_interval': 300,  # 5 minutes
                'max_error_rate': 0.05,  # 5% error rate threshold
                'component_timeout_seconds': 30
            },
            'feature_pipeline': {
                'update_frequency_seconds': 60,
                'cache_ttl_seconds': 300,
                'required_data_sources': ['price', 'volume', 'orderbook']
            },
            'model_manager': {
                'prediction_timeout_seconds': 10,
                'ensemble_min_models': 2,
                'confidence_threshold': 0.6
            },
            'strategy_orchestrator': {
                'signal_timeout_seconds': 15,
                'regime_detection_enabled': True,
                'fallback_to_traditional': True
            },
            'execution_optimizer': {
                'optimization_timeout_seconds': 20,
                'max_slippage_tolerance': 0.002,
                'execution_monitoring_enabled': True
            },
            'performance_monitor': {
                'real_time_tracking': True,
                'alert_notifications': True,
                'report_generation_enabled': True
            },
            'risk_management': {
                'enabled': True,
                'max_position_size': 0.1,
                'max_daily_loss': 0.05,
                'correlation_limit': 0.7
            }
        }
        
        # Load from file if provided
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
        
        return default_config
    
    def _load_unified_config(self, unified_config: 'UnifiedConfigurationSchema') -> Dict[str, Any]:
        """Load configuration from unified configuration system"""
        try:
            adapter = MLIntegrationConfigAdapter(unified_config)
            ml_config = adapter.get_ml_config()
            
            # Convert unified config to our expected format
            config = {
                'trading_mode': ml_config.get('trading_mode', 'paper_trading'),
                'system': ml_config.get('system', {}),
                'feature_pipeline': ml_config.get('feature_pipeline', {}),
                'model_manager': ml_config.get('model_manager', {}),
                'strategy_orchestrator': ml_config.get('strategy_orchestrator', {}),
                'execution_optimizer': ml_config.get('execution_optimizer', {}),
                'performance_monitor': ml_config.get('performance_monitor', {}),
                'risk_management': ml_config.get('risk_management', {})
            }
            
            # Fill in any missing defaults
            default_config = self._get_default_config()
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict) and isinstance(config[key], dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load unified configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'trading_mode': 'paper_trading',
            'system': {
                'max_concurrent_decisions': 5,
                'decision_cooldown_seconds': 30,
                'health_check_interval': 300,
                'max_error_rate': 0.05,
                'component_timeout_seconds': 30
            },
            'feature_pipeline': {
                'update_frequency_seconds': 60,
                'cache_ttl_seconds': 300,
                'required_data_sources': ['price', 'volume', 'orderbook']
            },
            'model_manager': {
                'prediction_timeout_seconds': 10,
                'ensemble_min_models': 2,
                'confidence_threshold': 0.6
            },
            'strategy_orchestrator': {
                'signal_timeout_seconds': 15,
                'regime_detection_enabled': True,
                'fallback_to_traditional': True
            },
            'execution_optimizer': {
                'optimization_timeout_seconds': 20,
                'max_slippage_tolerance': 0.002,
                'execution_monitoring_enabled': True
            },
            'performance_monitor': {
                'real_time_tracking': True,
                'alert_notifications': True,
                'report_generation_enabled': True
            },
            'risk_management': {
                'enabled': True,
                'max_position_size': 0.1,
                'max_daily_loss': 0.05,
                'correlation_limit': 0.7
            }
        }
    
    async def _initialize_system(self):
        """Initialize all ML system components"""
        logger.info("Initializing ML Integration System...")
        
        try:
            # Initialize components in order
            await self._initialize_feature_pipeline()
            await self._initialize_model_manager()
            await self._initialize_strategy_orchestrator()
            await self._initialize_execution_optimizer()
            await self._initialize_performance_monitor()
            await self._initialize_risk_manager()
            
            # Start background tasks
            asyncio.create_task(self._health_monitoring_loop())
            asyncio.create_task(self._decision_processing_loop())
            asyncio.create_task(self._performance_tracking_loop())
            
            # System is now active
            self.status = SystemStatus.ACTIVE
            logger.info("ML Integration System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML Integration System: {e}")
            self.status = SystemStatus.ERROR
            raise
    
    async def _initialize_feature_pipeline(self):
        """Initialize ML feature pipeline"""
        try:
            self.feature_pipeline = MLFeaturePipeline(
                config=self.config.get('feature_pipeline', {})
            )
            logger.info("Feature pipeline initialized")
        except Exception as e:
            logger.error(f"Error initializing feature pipeline: {e}")
            raise
    
    async def _initialize_model_manager(self):
        """Initialize ML model manager"""
        try:
            self.model_manager = MLModelManager(
                config=self.config.get('model_manager', {})
            )
            logger.info("Model manager initialized")
        except Exception as e:
            logger.error(f"Error initializing model manager: {e}")
            raise
    
    async def _initialize_strategy_orchestrator(self):
        """Initialize strategy orchestrator"""
        try:
            if not self.model_manager:
                raise RuntimeError("Model manager must be initialized first")
            
            self.strategy_orchestrator = MLStrategyOrchestrator(
                ml_model_manager=self.model_manager,
                config=self.config.get('strategy_orchestrator', {})
            )
            logger.info("Strategy orchestrator initialized")
        except Exception as e:
            logger.error(f"Error initializing strategy orchestrator: {e}")
            raise
    
    async def _initialize_execution_optimizer(self):
        """Initialize execution optimizer"""
        try:
            self.execution_optimizer = MLExecutionOptimizer(
                config=self.config.get('execution_optimizer', {})
            )
            logger.info("Execution optimizer initialized")
        except Exception as e:
            logger.error(f"Error initializing execution optimizer: {e}")
            raise
    
    async def _initialize_performance_monitor(self):
        """Initialize performance monitor"""
        try:
            self.performance_monitor = MLPerformanceMonitor(
                config=self.config.get('performance_monitor', {})
            )
            logger.info("Performance monitor initialized")
        except Exception as e:
            logger.error(f"Error initializing performance monitor: {e}")
            raise
    
    async def _initialize_risk_manager(self):
        """Initialize risk manager"""
        try:
            if RISK_MANAGER_AVAILABLE and self.config.get('risk_management', {}).get('enabled', True):
                self.risk_manager = UnifiedRiskManager()
                logger.info("Risk manager initialized")
            else:
                logger.warning("Risk manager not available or disabled")
        except Exception as e:
            logger.error(f"Error initializing risk manager: {e}")
            # Risk manager is optional, don't fail initialization
    
    async def generate_trading_decision(self, symbol: str, 
                                      market_data: Dict[str, Any]) -> TradingDecision:
        """
        Generate complete trading decision using all ML components
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            
        Returns:
            TradingDecision with complete analysis and recommendations
        """
        
        if self.status != SystemStatus.ACTIVE:
            raise RuntimeError(f"System not active (status: {self.status.value})")
        
        decision_start_time = datetime.now()
        decision_id = f"decision_{symbol}_{decision_start_time.strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            logger.info(f"Generating trading decision for {symbol}")
            
            # Step 1: Generate ML features
            features = await self._generate_features(symbol, market_data)
            
            # Step 2: Get combined signal from strategy orchestrator
            current_price = market_data.get('price', 0)
            combined_signal = await self.strategy_orchestrator.generate_combined_signal(
                features, symbol, current_price
            )
            
            # Step 3: Risk assessment
            risk_assessment = await self._assess_risk(symbol, combined_signal, market_data)
            
            # Step 4: Generate execution plan if signal is actionable
            execution_plan = None
            if combined_signal.final_signal.value != 'hold' and combined_signal.position_size > 0:
                execution_plan = await self._generate_execution_plan(
                    symbol, combined_signal, features, market_data
                )
            
            # Step 5: Calculate confidence and expected outcomes
            confidence_score = self._calculate_overall_confidence(combined_signal, risk_assessment)
            expected_outcome = self._calculate_expected_outcome(combined_signal, execution_plan)
            
            # Step 6: Determine recommended action
            recommended_action = self._determine_recommended_action(
                combined_signal, execution_plan, risk_assessment, confidence_score
            )
            
            # Create trading decision
            decision = TradingDecision(
                decision_id=decision_id,
                symbol=symbol,
                signal=combined_signal,
                execution_plan=execution_plan,
                risk_assessment=risk_assessment,
                confidence_score=confidence_score,
                recommended_action=recommended_action,
                position_size=Decimal(str(combined_signal.position_size)),
                expected_outcome=expected_outcome,
                timestamp=decision_start_time
            )
            
            # Store decision
            self.recent_decisions.append(decision)
            if len(self.recent_decisions) > 100:  # Keep last 100 decisions
                self.recent_decisions = self.recent_decisions[-100:]
            
            self.decision_history[decision_id] = {
                'decision': decision,
                'processing_time': (datetime.now() - decision_start_time).total_seconds(),
                'components_used': ['feature_pipeline', 'strategy_orchestrator', 'execution_optimizer']
            }
            
            # Log decision
            processing_time = (datetime.now() - decision_start_time).total_seconds()
            logger.info(f"Generated decision {decision_id} in {processing_time:.2f}s: "
                       f"{recommended_action} {symbol} (confidence: {confidence_score:.2f})")
            
            return decision
            
        except Exception as e:
            self.error_count += 1
            self.last_error_time = datetime.now()
            logger.error(f"Error generating trading decision for {symbol}: {e}")
            
            # Return safe/neutral decision on error
            return self._create_error_decision(decision_id, symbol, str(e))
    
    async def _generate_features(self, symbol: str, market_data: Dict[str, Any]) -> MLFeatures:
        """Generate ML features using feature pipeline"""
        
        timeout = self.config['system']['component_timeout_seconds']
        
        try:
            features = await asyncio.wait_for(
                self.feature_pipeline.generate_features(symbol, market_data),
                timeout=timeout
            )
            return features
        except asyncio.TimeoutError:
            logger.error(f"Feature generation timeout for {symbol}")
            raise
        except Exception as e:
            logger.error(f"Error generating features for {symbol}: {e}")
            raise
    
    async def _assess_risk(self, symbol: str, signal: CombinedSignal, 
                          market_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess risk for the trading signal"""
        
        risk_assessment = {
            'market_risk': 0.5,  # Default medium risk
            'execution_risk': 0.3,
            'model_risk': 1.0 - signal.final_confidence,
            'position_risk': min(1.0, float(signal.position_size) * 2),  # Higher position = higher risk
            'regime_risk': 0.4,  # Default regime risk
            'overall_risk': 0.5
        }
        
        # Use risk manager if available
        if self.risk_manager:
            try:
                # Get risk metrics from unified risk manager
                risk_metrics = await self.risk_manager.assess_trade_risk(
                    symbol=symbol,
                    position_size=float(signal.position_size),
                    signal_confidence=signal.final_confidence,
                    market_conditions=market_data
                )
                
                # Merge with risk assessment
                if isinstance(risk_metrics, dict):
                    risk_assessment.update(risk_metrics)
                
            except Exception as e:
                logger.error(f"Error getting risk assessment from risk manager: {e}")
        
        # Calculate overall risk
        risk_assessment['overall_risk'] = np.mean([
            risk_assessment['market_risk'],
            risk_assessment['execution_risk'],
            risk_assessment['model_risk'],
            risk_assessment['position_risk']
        ])
        
        return risk_assessment
    
    async def _generate_execution_plan(self, symbol: str, signal: CombinedSignal,
                                     features: MLFeatures, market_data: Dict[str, Any]) -> ExecutionPlan:
        """Generate execution plan using execution optimizer"""
        
        order_request = {
            'symbol': symbol,
            'quantity': float(signal.position_size),
            'side': 'buy' if signal.final_signal.value in ['buy', 'strong_buy'] else 'sell',
            'urgency': signal.final_confidence  # Higher confidence = higher urgency
        }
        
        timeout = self.config['system']['component_timeout_seconds']
        
        try:
            execution_plan = await asyncio.wait_for(
                self.execution_optimizer.create_execution_plan(order_request, features),
                timeout=timeout
            )
            return execution_plan
        except asyncio.TimeoutError:
            logger.error(f"Execution plan generation timeout for {symbol}")
            raise
        except Exception as e:
            logger.error(f"Error generating execution plan for {symbol}: {e}")
            raise
    
    def _calculate_overall_confidence(self, signal: CombinedSignal, 
                                    risk_assessment: Dict[str, float]) -> float:
        """Calculate overall confidence score for the decision"""
        
        # Base confidence from signal
        base_confidence = signal.final_confidence
        
        # Adjust for risk
        risk_penalty = risk_assessment.get('overall_risk', 0.5) * 0.3  # Max 30% penalty
        
        # Adjust for model consensus
        consensus_bonus = signal.ml_signal.confidence * 0.2  # Max 20% bonus
        
        # Calculate final confidence
        overall_confidence = base_confidence - risk_penalty + consensus_bonus
        
        # Clamp to 0-1 range
        return max(0.0, min(1.0, overall_confidence))
    
    def _calculate_expected_outcome(self, signal: CombinedSignal, 
                                  execution_plan: Optional[ExecutionPlan]) -> Dict[str, float]:
        """Calculate expected outcome metrics"""
        
        expected_outcome = {
            'expected_return': 0.0,
            'expected_volatility': 0.02,  # 2% default volatility
            'probability_of_profit': signal.final_confidence,
            'expected_holding_period': 3600.0,  # 1 hour default
            'expected_drawdown': 0.01  # 1% expected drawdown
        }
        
        # Use ML predictions if available
        if signal.ml_signal.expected_return:
            expected_outcome['expected_return'] = signal.ml_signal.expected_return
        
        if signal.ml_signal.expected_volatility:
            expected_outcome['expected_volatility'] = signal.ml_signal.expected_volatility
        
        # Use execution plan timing if available
        if execution_plan:
            completion_time = execution_plan.estimated_completion_time
            expected_outcome['expected_holding_period'] = (
                completion_time - datetime.now()
            ).total_seconds()
        
        return expected_outcome
    
    def _determine_recommended_action(self, signal: CombinedSignal, 
                                    execution_plan: Optional[ExecutionPlan],
                                    risk_assessment: Dict[str, float],
                                    confidence_score: float) -> str:
        """Determine recommended action based on all inputs"""
        
        # Check if signal is actionable
        if signal.final_signal.value == 'hold':
            return 'HOLD - No clear directional signal'
        
        # Check confidence threshold
        min_confidence = self.config.get('model_manager', {}).get('confidence_threshold', 0.6)
        if confidence_score < min_confidence:
            return f'HOLD - Confidence {confidence_score:.2f} below threshold {min_confidence}'
        
        # Check risk limits
        max_risk = self.config.get('risk_management', {}).get('max_position_risk', 0.7)
        if risk_assessment.get('overall_risk', 0.5) > max_risk:
            return f'HOLD - Risk {risk_assessment.get("overall_risk", 0.5):.2f} exceeds limit {max_risk}'
        
        # Check trading mode
        if self.trading_mode == TradingMode.ADVISORY_ONLY:
            action = 'ADVISE' if signal.final_signal.value in ['buy', 'strong_buy'] else 'ADVISE'
            return f'{action} - {signal.final_signal.value.upper()} {signal.position_size:.3f}'
        
        elif self.trading_mode == TradingMode.PAPER_TRADING:
            action = 'PAPER' if signal.final_signal.value in ['buy', 'strong_buy'] else 'PAPER'
            return f'{action} - {signal.final_signal.value.upper()} {signal.position_size:.3f}'
        
        elif self.trading_mode in [TradingMode.FULLY_AUTOMATED, TradingMode.SEMI_AUTOMATED]:
            action = 'EXECUTE' if signal.final_signal.value in ['buy', 'strong_buy'] else 'EXECUTE'
            approval = '' if self.trading_mode == TradingMode.FULLY_AUTOMATED else ' (PENDING APPROVAL)'
            return f'{action} - {signal.final_signal.value.upper()} {signal.position_size:.3f}{approval}'
        
        else:
            return 'HOLD - Unknown trading mode'
    
    def _create_error_decision(self, decision_id: str, symbol: str, error_msg: str) -> TradingDecision:
        """Create a safe decision when errors occur"""
        
        from .ml_feature_pipeline import MLSignalType
        from .ml_strategy_orchestrator import CombinedSignal, MarketRegime, TraditionalSignal
        from .ml_feature_pipeline import MLPrediction
        
        # Create minimal safe decision
        safe_ml_prediction = MLPrediction(
            signal_type=MLSignalType.HOLD,
            confidence=0.0,
            probability_distribution={},
            feature_importance={},
            model_name='error',
            timestamp=datetime.now()
        )
        
        safe_combined_signal = CombinedSignal(
            ml_signal=safe_ml_prediction,
            traditional_signals=[],
            final_signal=MLSignalType.HOLD,
            final_confidence=0.0,
            strategy_weights={},
            position_size=0.0,
            risk_metrics={'overall_risk': 1.0},
            market_regime=MarketRegime.UNCERTAIN,
            timestamp=datetime.now()
        )
        
        return TradingDecision(
            decision_id=decision_id,
            symbol=symbol,
            signal=safe_combined_signal,
            execution_plan=None,
            risk_assessment={'overall_risk': 1.0},
            confidence_score=0.0,
            recommended_action=f'HOLD - System Error: {error_msg}',
            position_size=Decimal('0'),
            expected_outcome={'expected_return': 0.0},
            timestamp=datetime.now()
        )
    
    async def execute_decision(self, decision: TradingDecision) -> Dict[str, Any]:
        """Execute a trading decision"""
        
        if self.trading_mode == TradingMode.ADVISORY_ONLY:
            return {'status': 'advisory_only', 'message': 'System in advisory mode only'}
        
        if not decision.execution_plan:
            return {'status': 'no_execution_plan', 'message': 'No execution plan available'}
        
        try:
            logger.info(f"Executing decision {decision.decision_id} for {decision.symbol}")
            
            # In a real implementation, this would execute the trades
            execution_result = {
                'status': 'success',
                'decision_id': decision.decision_id,
                'symbol': decision.symbol,
                'executed_quantity': float(decision.position_size),
                'execution_plan_id': decision.execution_plan.order_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Record execution for performance monitoring
            if self.performance_monitor:
                # Would record actual execution performance
                pass
            
            logger.info(f"Successfully executed decision {decision.decision_id}")
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing decision {decision.decision_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        
        while True:
            try:
                await asyncio.sleep(self.config['system']['health_check_interval'])
                await self._update_health_metrics()
                await self._check_system_health()
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
    
    async def _update_health_metrics(self):
        """Update system health metrics"""
        
        current_time = datetime.now()
        uptime = current_time - self.start_time
        
        # Calculate component health scores
        feature_pipeline_health = 1.0 if self.feature_pipeline else 0.0
        model_manager_health = 1.0 if self.model_manager else 0.0
        strategy_orchestrator_health = 1.0 if self.strategy_orchestrator else 0.0
        execution_optimizer_health = 1.0 if self.execution_optimizer else 0.0
        performance_monitor_health = 1.0 if self.performance_monitor else 0.0
        
        # Calculate overall health
        component_healths = [
            feature_pipeline_health,
            model_manager_health,
            strategy_orchestrator_health,
            execution_optimizer_health,
            performance_monitor_health
        ]
        overall_health = np.mean(component_healths)
        
        # Calculate error rate
        recent_decisions = len([d for d in self.recent_decisions if d.timestamp > current_time - timedelta(hours=1)])
        error_rate = self.error_count / max(1, recent_decisions) if recent_decisions > 0 else 0.0
        
        # Get active alerts
        active_alerts = 0
        if self.performance_monitor:
            active_alerts = len(self.performance_monitor.get_active_alerts())
        
        # Calculate decisions per hour
        decisions_per_hour = len([d for d in self.recent_decisions if d.timestamp > current_time - timedelta(hours=1)])
        
        # Create health metrics
        self.health_metrics = SystemHealthMetrics(
            status=self.status,
            uptime=uptime,
            feature_pipeline_health=feature_pipeline_health,
            model_manager_health=model_manager_health,
            strategy_orchestrator_health=strategy_orchestrator_health,
            execution_optimizer_health=execution_optimizer_health,
            performance_monitor_health=performance_monitor_health,
            overall_health_score=overall_health,
            active_alerts=active_alerts,
            last_decision_time=self.recent_decisions[-1].timestamp if self.recent_decisions else self.start_time,
            decisions_per_hour=decisions_per_hour,
            error_rate=error_rate
        )
        
        self.last_health_check = current_time
    
    async def _check_system_health(self):
        """Check system health and adjust status if needed"""
        
        if not self.health_metrics:
            return
        
        # Check error rate
        max_error_rate = self.config['system']['max_error_rate']
        if self.health_metrics.error_rate > max_error_rate:
            if self.status == SystemStatus.ACTIVE:
                self.status = SystemStatus.DEGRADED
                logger.warning(f"System degraded due to high error rate: {self.health_metrics.error_rate:.1%}")
        
        # Check overall health
        if self.health_metrics.overall_health_score < 0.7:
            if self.status == SystemStatus.ACTIVE:
                self.status = SystemStatus.DEGRADED
                logger.warning(f"System degraded due to low health score: {self.health_metrics.overall_health_score:.2f}")
        
        # Recovery check
        if self.status == SystemStatus.DEGRADED:
            if (self.health_metrics.error_rate < max_error_rate / 2 and 
                self.health_metrics.overall_health_score > 0.8):
                self.status = SystemStatus.ACTIVE
                logger.info("System recovered to active status")
    
    async def _decision_processing_loop(self):
        """Background decision processing loop"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Process pending decisions if any
                # In a real implementation, this might process scheduled decisions
                
            except Exception as e:
                logger.error(f"Error in decision processing loop: {e}")
    
    async def _performance_tracking_loop(self):
        """Background performance tracking loop"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Update performance tracking
                if self.performance_monitor:
                    # Get real-time metrics
                    metrics = self.performance_monitor.get_real_time_metrics()
                    logger.debug(f"Current performance metrics: {metrics.get('model_accuracies', {})}")
                
            except Exception as e:
                logger.error(f"Error in performance tracking loop: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            'system_status': self.status.value,
            'trading_mode': self.trading_mode.value,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'recent_decisions': len(self.recent_decisions),
            'error_count': self.error_count,
            'last_error': self.last_error_time.isoformat() if self.last_error_time else None,
            'components': {
                'feature_pipeline': self.feature_pipeline is not None,
                'model_manager': self.model_manager is not None,
                'strategy_orchestrator': self.strategy_orchestrator is not None,
                'execution_optimizer': self.execution_optimizer is not None,
                'performance_monitor': self.performance_monitor is not None,
                'risk_manager': self.risk_manager is not None
            }
        }
        
        if self.health_metrics:
            status['health_metrics'] = {
                'overall_health_score': self.health_metrics.overall_health_score,
                'active_alerts': self.health_metrics.active_alerts,
                'decisions_per_hour': self.health_metrics.decisions_per_hour,
                'error_rate': self.health_metrics.error_rate
            }
        
        return status
    
    async def shutdown(self):
        """Gracefully shutdown the ML system"""
        
        logger.info("Shutting down ML Integration System...")
        
        try:
            self.status = SystemStatus.MAINTENANCE
            
            # Stop processing new decisions
            # In a real implementation, this would stop all background tasks
            
            # Save state if needed
            # In a real implementation, this would save important state
            
            logger.info("ML Integration System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'MLIntegrationController',
    'SystemStatus',
    'TradingMode',
    'TradingDecision',
    'SystemHealthMetrics'
]