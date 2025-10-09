"""
ML-Enhanced Risk Management System

This module provides comprehensive risk management specifically designed for ML-driven trading systems.
It acts as a final safety layer before any trade execution, with special considerations for:

- ML model confidence and uncertainty quantification
- Dynamic risk adjustment based on ML predictions
- Circuit breakers for automated trading systems
- Emergency stop mechanisms
- Comprehensive trade pre-validation
- ML explainability and audit trails

Key Features:
- Pre-trade risk validation for all ML-generated signals
- Dynamic position sizing based on ML confidence
- Real-time risk monitoring with circuit breakers
- Emergency stop functionality
- Comprehensive logging and audit trails
- Integration with existing UnifiedRiskManager
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json

# Import existing risk management components
from .core.unified_risk_manager import (
    UnifiedRiskManager, RiskParameters, PositionRisk, 
    PortfolioRiskMetrics, RiskLevel, PositionSizeMethod, AlertLevel
)

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ML-SPECIFIC ENUMS AND DATA STRUCTURES
# ============================================================================

class MLRiskLevel(Enum):
    """ML-specific risk levels"""
    VERY_LOW = "very_low"           # High confidence, low volatility predictions
    LOW = "low"                     # Good confidence, stable models
    MODERATE = "moderate"           # Medium confidence, normal conditions
    HIGH = "high"                   # Low confidence or high uncertainty
    VERY_HIGH = "very_high"         # Very low confidence, model disagreement
    EXTREME = "extreme"             # Model failure, anomalous predictions
    EMERGENCY_STOP = "emergency_stop"  # System-wide emergency

class TradeBlockReason(Enum):
    """Reasons why a trade was blocked"""
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"
    EXCESSIVE_POSITION_SIZE = "excessive_position_size"
    PORTFOLIO_RISK_LIMIT = "portfolio_risk_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    CIRCUIT_BREAKER_ACTIVE = "circuit_breaker_active"
    EMERGENCY_STOP_ACTIVE = "emergency_stop_active"
    MODEL_UNCERTAINTY_HIGH = "model_uncertainty_high"
    CORRELATION_RISK_HIGH = "correlation_risk_high"
    MARKET_CONDITIONS_ADVERSE = "market_conditions_adverse"
    EXPLAINABILITY_FAILED = "explainability_failed"

class CircuitBreakerType(Enum):
    """Types of circuit breakers"""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    MODEL_PERFORMANCE_DEGRADED = "model_performance_degraded"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    EXECUTION_FAILURE_RATE = "execution_failure_rate"
    DATA_QUALITY_ISSUE = "data_quality_issue"

@dataclass
class MLTradeRisk:
    """Comprehensive ML trade risk assessment"""
    symbol: str
    ml_confidence: float
    model_uncertainty: float
    prediction_stability: float
    ensemble_agreement: float
    feature_importance_score: float
    market_regime_risk: float
    execution_risk: float
    portfolio_impact_risk: float
    overall_ml_risk: MLRiskLevel
    risk_score: float
    confidence_adjusted_size: Decimal
    recommended_action: str
    risk_factors: List[str]
    explanation: str
    timestamp: datetime

@dataclass
class CircuitBreakerStatus:
    """Circuit breaker status information"""
    breaker_type: CircuitBreakerType
    is_active: bool
    triggered_at: Optional[datetime]
    trigger_value: float
    threshold: float
    estimated_duration: Optional[timedelta]
    recovery_conditions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmergencyStopStatus:
    """Emergency stop system status"""
    is_active: bool
    activated_at: Optional[datetime]
    activation_reason: str
    manual_override: bool
    auto_recovery_enabled: bool
    recovery_conditions: List[str]
    override_code: Optional[str] = None

@dataclass
class TradeValidationResult:
    """Result of pre-trade validation"""
    is_approved: bool
    final_position_size: Decimal
    risk_assessment: MLTradeRisk
    blocked_reasons: List[TradeBlockReason]
    warnings: List[str]
    execution_params: Dict[str, Any]
    validation_timestamp: datetime

# ============================================================================
# ML RISK MANAGER
# ============================================================================

class MLRiskManager:
    """
    ML-Enhanced Risk Management System
    
    Provides comprehensive risk management for ML-driven trading with:
    - Pre-trade validation of all ML signals
    - Dynamic risk adjustment based on ML confidence
    - Circuit breakers and emergency stops
    - Real-time risk monitoring
    - Comprehensive audit trails
    """
    
    def __init__(self, unified_risk_manager: UnifiedRiskManager, 
                 ml_risk_params: Optional[Dict[str, Any]] = None):
        """Initialize ML Risk Manager"""
        self.unified_risk_manager = unified_risk_manager
        self.ml_risk_params = ml_risk_params or self._get_default_ml_params()
        
        # Circuit breaker states
        self.circuit_breakers: Dict[CircuitBreakerType, CircuitBreakerStatus] = {}
        self._initialize_circuit_breakers()
        
        # Emergency stop system
        self.emergency_stop = EmergencyStopStatus(
            is_active=False,
            activated_at=None,
            activation_reason="",
            manual_override=False,
            auto_recovery_enabled=True,
            recovery_conditions=[]
        )
        
        # Risk monitoring state
        self.daily_loss_tracker: Dict[str, Decimal] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.blocked_trades: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.model_performance_tracker: Dict[str, List[float]] = {}
        self.execution_failure_tracker: List[datetime] = []
        
        logger.info("ML Risk Manager initialized with enhanced safety features")
    
    def _get_default_ml_params(self) -> Dict[str, Any]:
        """Get default ML-specific risk parameters"""
        return {
            'min_confidence_threshold': 0.6,      # Minimum ML confidence for execution
            'max_uncertainty_threshold': 0.4,     # Maximum model uncertainty allowed
            'min_ensemble_agreement': 0.7,        # Minimum agreement between models
            'confidence_scaling_factor': 2.0,     # How much confidence affects position size
            'daily_loss_limit': 0.05,            # 5% daily loss limit
            'circuit_breaker_thresholds': {
                CircuitBreakerType.DAILY_LOSS_LIMIT: 0.03,        # 3% daily loss
                CircuitBreakerType.VOLATILITY_SPIKE: 3.0,         # 3x normal volatility
                CircuitBreakerType.MODEL_PERFORMANCE_DEGRADED: 0.4, # < 40% accuracy
                CircuitBreakerType.CORRELATION_BREAKDOWN: 0.3,     # Correlation drops < 0.3
                CircuitBreakerType.EXECUTION_FAILURE_RATE: 0.2,    # > 20% execution failures
                CircuitBreakerType.DATA_QUALITY_ISSUE: 0.8        # > 80% data quality issues
            },
            'emergency_stop_conditions': {
                'max_consecutive_losses': 5,
                'max_drawdown_percentage': 0.10,  # 10% portfolio drawdown
                'model_complete_failure': True
            }
        }
    
    def _initialize_circuit_breakers(self):
        """Initialize all circuit breakers"""
        for breaker_type in CircuitBreakerType:
            threshold = self.ml_risk_params['circuit_breaker_thresholds'].get(breaker_type.value, 1.0)
            self.circuit_breakers[breaker_type] = CircuitBreakerStatus(
                breaker_type=breaker_type,
                is_active=False,
                triggered_at=None,
                trigger_value=0.0,
                threshold=threshold,
                estimated_duration=None,
                recovery_conditions=self._get_recovery_conditions(breaker_type)
            )
    
    def _get_recovery_conditions(self, breaker_type: CircuitBreakerType) -> List[str]:
        """Get recovery conditions for each circuit breaker type"""
        conditions = {
            CircuitBreakerType.DAILY_LOSS_LIMIT: [
                "Wait until next trading day",
                "Manual override by risk manager"
            ],
            CircuitBreakerType.VOLATILITY_SPIKE: [
                "Market volatility returns to normal levels",
                "Wait minimum 30 minutes after spike"
            ],
            CircuitBreakerType.MODEL_PERFORMANCE_DEGRADED: [
                "Model performance improves above threshold",
                "Model retraining completed successfully"
            ],
            CircuitBreakerType.CORRELATION_BREAKDOWN: [
                "Market correlations return to expected ranges",
                "Risk models updated with new correlation matrix"
            ],
            CircuitBreakerType.EXECUTION_FAILURE_RATE: [
                "Execution success rate improves",
                "Exchange connectivity issues resolved"
            ],
            CircuitBreakerType.DATA_QUALITY_ISSUE: [
                "Data feed quality improves",
                "Data source issues resolved"
            ]
        }
        return conditions.get(breaker_type, ["Manual intervention required"])
    
    async def validate_trade(self, symbol: str, signal_data: Dict[str, Any], 
                           market_data: Dict[str, Any], 
                           ml_predictions: Dict[str, Any]) -> TradeValidationResult:
        """
        Comprehensive pre-trade validation for ML-generated signals
        
        This is the main entry point for validating any trade before execution.
        """
        logger.info(f"Validating ML trade for {symbol}")
        
        try:
            # Step 1: Check emergency stop
            if self.emergency_stop.is_active:
                return self._create_blocked_result(
                    symbol, signal_data, [TradeBlockReason.EMERGENCY_STOP_ACTIVE],
                    "Emergency stop is active - all trading halted"
                )
            
            # Step 2: Check circuit breakers
            active_breakers = self._get_active_circuit_breakers()
            if active_breakers:
                return self._create_blocked_result(
                    symbol, signal_data, [TradeBlockReason.CIRCUIT_BREAKER_ACTIVE],
                    f"Circuit breakers active: {[b.name for b in active_breakers]}"
                )
            
            # Step 3: Assess ML-specific risks
            ml_risk = await self._assess_ml_trade_risk(
                symbol, signal_data, market_data, ml_predictions
            )
            
            # Step 4: Check ML confidence thresholds
            blocked_reasons = []
            if ml_risk.ml_confidence < self.ml_risk_params['min_confidence_threshold']:
                blocked_reasons.append(TradeBlockReason.INSUFFICIENT_CONFIDENCE)
            
            if ml_risk.model_uncertainty > self.ml_risk_params['max_uncertainty_threshold']:
                blocked_reasons.append(TradeBlockReason.MODEL_UNCERTAINTY_HIGH)
            
            if ml_risk.ensemble_agreement < self.ml_risk_params['min_ensemble_agreement']:
                blocked_reasons.append(TradeBlockReason.MODEL_UNCERTAINTY_HIGH)
            
            # Step 5: Calculate risk-adjusted position size
            base_position_size = Decimal(str(signal_data.get('position_size', '0')))
            risk_adjusted_size = await self._calculate_ml_adjusted_position_size(
                symbol, base_position_size, ml_risk, market_data
            )
            
            # Step 6: Validate with unified risk manager
            portfolio_value = Decimal(str(market_data.get('portfolio_value', '100000')))
            returns = market_data.get('returns', pd.Series(dtype=float))
            
            if not returns.empty:
                final_size, position_risk = await self.unified_risk_manager.calculate_position_size(
                    symbol=symbol,
                    side=signal_data.get('side', 'buy'),  
                    portfolio_value=portfolio_value,
                    returns=returns,
                    method=PositionSizeMethod.TAX_OPTIMIZED
                )
                
                # Use the smaller of ML-adjusted and risk manager sizes
                final_size = min(risk_adjusted_size, final_size)
            else:
                final_size = risk_adjusted_size
                
            # Step 7: Final position size checks
            max_position = portfolio_value * self.unified_risk_manager.risk_params.max_position_size
            if final_size > max_position:
                blocked_reasons.append(TradeBlockReason.EXCESSIVE_POSITION_SIZE)
                final_size = max_position
            
            # Step 8: Check daily loss limits
            if await self._would_exceed_daily_loss_limit(symbol, final_size, market_data):
                blocked_reasons.append(TradeBlockReason.DAILY_LOSS_LIMIT)
            
            # Step 9: Portfolio risk checks
            if await self._would_exceed_portfolio_risk(symbol, final_size, market_data):
                blocked_reasons.append(TradeBlockReason.PORTFOLIO_RISK_LIMIT)
            
            # Step 10: Generate result
            is_approved = len(blocked_reasons) == 0
            warnings = self._generate_warnings(ml_risk, final_size, portfolio_value)
            
            result = TradeValidationResult(
                is_approved=is_approved,
                final_position_size=final_size if is_approved else Decimal('0'),
                risk_assessment=ml_risk,
                blocked_reasons=blocked_reasons,
                warnings=warnings,
                execution_params=self._generate_execution_params(ml_risk, final_size),
                validation_timestamp=datetime.now()
            )
            
            # Step 11: Log and track
            await self._log_validation_result(symbol, result)
            
            if not is_approved:
                self.blocked_trades.append({
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'reasons': [r.value for r in blocked_reasons],
                    'original_size': base_position_size,
                    'ml_confidence': ml_risk.ml_confidence
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error during trade validation for {symbol}: {e}")
            return self._create_blocked_result(
                symbol, signal_data, [TradeBlockReason.MODEL_UNCERTAINTY_HIGH],
                f"Validation error: {str(e)}"
            )
    
    async def _assess_ml_trade_risk(self, symbol: str, signal_data: Dict[str, Any],
                                  market_data: Dict[str, Any], 
                                  ml_predictions: Dict[str, Any]) -> MLTradeRisk:
        """Assess ML-specific trade risks"""
        
        # Extract ML metrics
        ml_confidence = ml_predictions.get('confidence', 0.5)
        model_uncertainty = ml_predictions.get('uncertainty', 0.5)
        ensemble_agreement = ml_predictions.get('ensemble_agreement', 0.5)
        prediction_stability = ml_predictions.get('stability', 0.5)
        feature_importance = ml_predictions.get('feature_importance_score', 0.5)
        
        # Market regime risk
        market_volatility = market_data.get('volatility', 0.02)
        normal_volatility = 0.02  # Baseline volatility
        volatility_ratio = market_volatility / normal_volatility
        market_regime_risk = min(1.0, volatility_ratio / 2.0)  # Higher volatility = higher risk
        
        # Execution risk based on market conditions  
        bid_ask_spread = market_data.get('bid_ask_spread', 0.001)
        volume = market_data.get('volume', 1000000)
        execution_risk = min(1.0, bid_ask_spread * 1000 + (1.0 / max(volume, 1000)))
        
        # Portfolio impact risk
        portfolio_value = market_data.get('portfolio_value', 100000)
        position_size = Decimal(str(signal_data.get('position_size', '0')))
        position_ratio = float(position_size / Decimal(str(portfolio_value)))
        portfolio_impact_risk = min(1.0, position_ratio * 5)  # Higher position ratio = higher risk
        
        # Calculate overall ML risk level
        risk_factors = []
        risk_score = 0.0
        
        # Confidence factors
        if ml_confidence < 0.6:
            risk_factors.append("Low ML model confidence")
            risk_score += 0.3
        
        if model_uncertainty > 0.4:
            risk_factors.append("High model uncertainty")
            risk_score += 0.2
            
        if ensemble_agreement < 0.7:
            risk_factors.append("Low ensemble model agreement")
            risk_score += 0.2
            
        if prediction_stability < 0.6:
            risk_factors.append("Unstable predictions")
            risk_score += 0.15
            
        # Market factors
        if market_regime_risk > 0.6:
            risk_factors.append("Adverse market conditions")
            risk_score += 0.1
            
        if execution_risk > 0.3:
            risk_factors.append("Poor execution conditions")
            risk_score += 0.05
        
        # Determine overall risk level
        if risk_score >= 0.8:
            overall_risk = MLRiskLevel.EXTREME
        elif risk_score >= 0.6:
            overall_risk = MLRiskLevel.VERY_HIGH
        elif risk_score >= 0.4:
            overall_risk = MLRiskLevel.HIGH
        elif risk_score >= 0.2:
            overall_risk = MLRiskLevel.MODERATE
        elif risk_score >= 0.1:
            overall_risk = MLRiskLevel.LOW
        else:
            overall_risk = MLRiskLevel.VERY_LOW
        
        # Calculate confidence-adjusted position size
        confidence_multiplier = ml_confidence * self.ml_risk_params['confidence_scaling_factor']
        confidence_adjusted_size = position_size * Decimal(str(min(confidence_multiplier, 1.0)))
        
        # Generate recommendation
        if overall_risk in [MLRiskLevel.EXTREME, MLRiskLevel.VERY_HIGH]:
            recommended_action = "BLOCK_TRADE"
        elif overall_risk == MLRiskLevel.HIGH:
            recommended_action = "REDUCE_SIZE_50%"
        elif overall_risk == MLRiskLevel.MODERATE:
            recommended_action = "REDUCE_SIZE_25%"
        else:
            recommended_action = "PROCEED_WITH_CAUTION"
        
        # Generate explanation
        explanation = self._generate_risk_explanation(
            ml_confidence, model_uncertainty, ensemble_agreement, 
            risk_factors, overall_risk
        )
        
        return MLTradeRisk(
            symbol=symbol,
            ml_confidence=ml_confidence,
            model_uncertainty=model_uncertainty,
            prediction_stability=prediction_stability,
            ensemble_agreement=ensemble_agreement,
            feature_importance_score=feature_importance,
            market_regime_risk=market_regime_risk,
            execution_risk=execution_risk,
            portfolio_impact_risk=portfolio_impact_risk,
            overall_ml_risk=overall_risk,
            risk_score=risk_score,
            confidence_adjusted_size=confidence_adjusted_size,
            recommended_action=recommended_action,
            risk_factors=risk_factors,
            explanation=explanation,
            timestamp=datetime.now()
        )
    
    async def _calculate_ml_adjusted_position_size(self, symbol: str, base_size: Decimal,
                                                 ml_risk: MLTradeRisk, 
                                                 market_data: Dict[str, Any]) -> Decimal:
        """Calculate ML confidence-adjusted position size"""
        
        # Start with confidence-adjusted size
        adjusted_size = ml_risk.confidence_adjusted_size
        
        # Apply risk level adjustments
        risk_multipliers = {
            MLRiskLevel.VERY_LOW: 1.2,    # Can increase size for high confidence
            MLRiskLevel.LOW: 1.0,         # Keep as is
            MLRiskLevel.MODERATE: 0.8,    # Reduce size
            MLRiskLevel.HIGH: 0.5,        # Significantly reduce
            MLRiskLevel.VERY_HIGH: 0.2,   # Minimal size
            MLRiskLevel.EXTREME: 0.0      # Block trade
        }
        
        multiplier = risk_multipliers.get(ml_risk.overall_ml_risk, 0.5)
        final_size = adjusted_size * Decimal(str(multiplier))
        
        # Ensure minimum position size if trade is allowed
        min_trade_size = Decimal('10')  # $10 minimum
        if final_size > 0 and final_size < min_trade_size:
            final_size = min_trade_size
        
        logger.info(f"ML position size adjustment for {symbol}: "
                   f"{base_size} -> {final_size} "
                   f"(confidence: {ml_risk.ml_confidence:.2f}, "
                   f"risk: {ml_risk.overall_ml_risk.value})")
        
        return final_size
    
    async def activate_emergency_stop(self, reason: str, manual_override: bool = True,
                                    override_code: Optional[str] = None) -> bool:
        """Activate emergency stop to halt all trading"""
        
        self.emergency_stop = EmergencyStopStatus(
            is_active=True,
            activated_at=datetime.now(),
            activation_reason=reason,
            manual_override=manual_override,
            auto_recovery_enabled=not manual_override,
            recovery_conditions=[
                "Manual deactivation by authorized personnel",
                "System health checks pass",
                "Risk parameters normalized"
            ],
            override_code=override_code
        )
        
        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
        
        # TODO: Send alerts to administrators
        await self._send_emergency_alert(reason)
        
        return True
    
    async def deactivate_emergency_stop(self, override_code: Optional[str] = None) -> bool:
        """Deactivate emergency stop"""
        
        if self.emergency_stop.manual_override and override_code != self.emergency_stop.override_code:
            logger.warning("Emergency stop deactivation failed: Invalid override code")
            return False
        
        reason = self.emergency_stop.activation_reason
        self.emergency_stop = EmergencyStopStatus(
            is_active=False,
            activated_at=None,
            activation_reason="",
            manual_override=False,
            auto_recovery_enabled=True,
            recovery_conditions=[]
        )
        
        logger.info(f"Emergency stop deactivated. Previous reason: {reason}")
        return True
    
    async def check_and_trigger_circuit_breakers(self, market_data: Dict[str, Any]) -> List[CircuitBreakerType]:
        """Check conditions and trigger circuit breakers if necessary"""
        
        triggered_breakers = []
        
        for breaker_type, breaker in self.circuit_breakers.items():
            if breaker.is_active:
                continue  # Already active
                
            should_trigger = await self._should_trigger_circuit_breaker(
                breaker_type, market_data
            )
            
            if should_trigger:
                await self._trigger_circuit_breaker(breaker_type, market_data)
                triggered_breakers.append(breaker_type)
        
        return triggered_breakers
    
    async def _should_trigger_circuit_breaker(self, breaker_type: CircuitBreakerType,
                                            market_data: Dict[str, Any]) -> bool:
        """Check if a specific circuit breaker should be triggered"""
        
        threshold = self.circuit_breakers[breaker_type].threshold
        
        if breaker_type == CircuitBreakerType.DAILY_LOSS_LIMIT:
            daily_loss = await self._calculate_daily_loss()
            return daily_loss >= threshold
            
        elif breaker_type == CircuitBreakerType.VOLATILITY_SPIKE:
            current_vol = market_data.get('volatility', 0.02)
            normal_vol = 0.02  # Baseline volatility
            vol_ratio = current_vol / normal_vol
            return vol_ratio >= threshold
            
        elif breaker_type == CircuitBreakerType.MODEL_PERFORMANCE_DEGRADED:
            model_accuracy = await self._calculate_recent_model_accuracy()
            return model_accuracy <= threshold
            
        elif breaker_type == CircuitBreakerType.EXECUTION_FAILURE_RATE:
            failure_rate = await self._calculate_execution_failure_rate()
            return failure_rate >= threshold
            
        # Add more circuit breaker logic as needed
        return False
    
    async def _trigger_circuit_breaker(self, breaker_type: CircuitBreakerType,
                                     market_data: Dict[str, Any]):
        """Trigger a specific circuit breaker"""
        
        trigger_value = 0.0  # Extract actual trigger value from market_data
        
        self.circuit_breakers[breaker_type].is_active = True
        self.circuit_breakers[breaker_type].triggered_at = datetime.now()
        self.circuit_breakers[breaker_type].trigger_value = trigger_value
        
        # Set estimated recovery duration
        durations = {
            CircuitBreakerType.DAILY_LOSS_LIMIT: timedelta(hours=24),
            CircuitBreakerType.VOLATILITY_SPIKE: timedelta(minutes=30),
            CircuitBreakerType.MODEL_PERFORMANCE_DEGRADED: timedelta(hours=1),
            CircuitBreakerType.EXECUTION_FAILURE_RATE: timedelta(minutes=15)
        }
        
        self.circuit_breakers[breaker_type].estimated_duration = durations.get(
            breaker_type, timedelta(hours=1)
        )
        
        logger.warning(f"Circuit breaker triggered: {breaker_type.value} "
                      f"(threshold: {self.circuit_breakers[breaker_type].threshold}, "
                      f"value: {trigger_value})")
    
    def _get_active_circuit_breakers(self) -> List[CircuitBreakerType]:
        """Get list of currently active circuit breakers"""
        return [
            breaker_type for breaker_type, breaker in self.circuit_breakers.items()
            if breaker.is_active
        ]
    
    def _create_blocked_result(self, symbol: str, signal_data: Dict[str, Any],
                             blocked_reasons: List[TradeBlockReason],
                             explanation: str) -> TradeValidationResult:
        """Create a blocked trade validation result"""
        
        # Create minimal risk assessment for blocked trade
        ml_risk = MLTradeRisk(
            symbol=symbol,
            ml_confidence=0.0,
            model_uncertainty=1.0,
            prediction_stability=0.0,
            ensemble_agreement=0.0,
            feature_importance_score=0.0,
            market_regime_risk=1.0,
            execution_risk=1.0,
            portfolio_impact_risk=1.0,
            overall_ml_risk=MLRiskLevel.EXTREME,
            risk_score=1.0,
            confidence_adjusted_size=Decimal('0'),
            recommended_action="BLOCK_TRADE",
            risk_factors=["Trade blocked by risk management"],
            explanation=explanation,
            timestamp=datetime.now()
        )
        
        return TradeValidationResult(
            is_approved=False,
            final_position_size=Decimal('0'),
            risk_assessment=ml_risk,
            blocked_reasons=blocked_reasons,
            warnings=[],
            execution_params={},
            validation_timestamp=datetime.now()
        )
    
    def _generate_warnings(self, ml_risk: MLTradeRisk, final_size: Decimal,
                          portfolio_value: Decimal) -> List[str]:
        """Generate warnings for approved trades"""
        warnings = []
        
        if ml_risk.ml_confidence < 0.7:
            warnings.append(f"Low ML confidence: {ml_risk.ml_confidence:.1%}")
        
        if ml_risk.model_uncertainty > 0.3:
            warnings.append(f"High model uncertainty: {ml_risk.model_uncertainty:.1%}")
        
        position_ratio = float(final_size / portfolio_value)
        if position_ratio > 0.05:  # > 5% of portfolio
            warnings.append(f"Large position size: {position_ratio:.1%} of portfolio")
        
        if ml_risk.market_regime_risk > 0.5:
            warnings.append("Adverse market conditions detected")
        
        return warnings
    
    def _generate_execution_params(self, ml_risk: MLTradeRisk, 
                                 final_size: Decimal) -> Dict[str, Any]:
        """Generate execution parameters based on risk assessment"""
        
        # Base execution parameters
        params = {
            'order_type': 'limit',
            'time_in_force': 'GTC',
            'post_only': False
        }
        
        # Adjust based on risk level
        if ml_risk.overall_ml_risk in [MLRiskLevel.HIGH, MLRiskLevel.VERY_HIGH]:
            params.update({
                'order_type': 'limit',  # Force limit orders for high risk
                'post_only': True,      # Reduce execution risk
                'reduce_only': True     # Only reduce positions
            })
        elif ml_risk.overall_ml_risk == MLRiskLevel.MODERATE:
            params.update({
                'order_type': 'limit',
                'post_only': False
            })
        else:  # Low risk
            params.update({
                'order_type': 'market',  # Allow market orders for high confidence
                'post_only': False
            })
        
        # Add ML-specific metadata
        params['ml_metadata'] = {
            'confidence': ml_risk.ml_confidence,
            'risk_level': ml_risk.overall_ml_risk.value,
            'risk_score': ml_risk.risk_score,
            'validation_timestamp': ml_risk.timestamp.isoformat()
        }
        
        return params
    
    def _generate_risk_explanation(self, ml_confidence: float, model_uncertainty: float,
                                 ensemble_agreement: float, risk_factors: List[str],
                                 overall_risk: MLRiskLevel) -> str:
        """Generate human-readable risk explanation"""
        
        explanation = f"ML Risk Assessment: {overall_risk.value.upper()}\n"
        explanation += f"• Model Confidence: {ml_confidence:.1%}\n"
        explanation += f"• Model Uncertainty: {model_uncertainty:.1%}\n"
        explanation += f"• Ensemble Agreement: {ensemble_agreement:.1%}\n"
        
        if risk_factors:
            explanation += f"• Risk Factors: {', '.join(risk_factors)}\n"
        
        if overall_risk in [MLRiskLevel.HIGH, MLRiskLevel.VERY_HIGH, MLRiskLevel.EXTREME]:
            explanation += "\n⚠️  HIGH RISK: Consider manual review or reduced position size"
        elif overall_risk == MLRiskLevel.MODERATE:
            explanation += "\n⚡ MODERATE RISK: Monitor closely and consider smaller position"
        else:
            explanation += "\n✅ LOW RISK: Proceed with normal risk management"
        
        return explanation
    
    async def _log_validation_result(self, symbol: str, result: TradeValidationResult):
        """Log trade validation result for audit trail"""
        
        log_entry = {
            'timestamp': result.validation_timestamp.isoformat(),
            'symbol': symbol,
            'approved': result.is_approved,
            'final_size': str(result.final_position_size),
            'ml_confidence': result.risk_assessment.ml_confidence,
            'risk_level': result.risk_assessment.overall_ml_risk.value,
            'risk_score': result.risk_assessment.risk_score,
            'blocked_reasons': [r.value for r in result.blocked_reasons],
            'warnings': result.warnings,
            'risk_factors': result.risk_assessment.risk_factors
        }
        
        # Add to trade history for analysis
        self.trade_history.append(log_entry)
        
        # Keep only recent history (last 1000 trades)
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
        
        # Log for external systems
        if result.is_approved:
            logger.info(f"Trade APPROVED for {symbol}: "
                       f"size={result.final_position_size}, "
                       f"confidence={result.risk_assessment.ml_confidence:.1%}, "
                       f"risk={result.risk_assessment.overall_ml_risk.value}")
        else:
            logger.warning(f"Trade BLOCKED for {symbol}: "
                          f"reasons={[r.value for r in result.blocked_reasons]}")
    
    async def _send_emergency_alert(self, reason: str):
        """Send emergency alert to administrators"""
        # TODO: Implement actual alerting mechanism (email, SMS, etc.)
        logger.critical(f"EMERGENCY ALERT: {reason}")
    
    async def _calculate_daily_loss(self) -> float:
        """Calculate today's portfolio loss"""
        # TODO: Implement actual daily loss calculation
        return 0.02  # Placeholder
    
    async def _calculate_recent_model_accuracy(self) -> float:
        """Calculate recent ML model accuracy"""
        # TODO: Implement actual model performance tracking
        return 0.75  # Placeholder
    
    async def _calculate_execution_failure_rate(self) -> float:
        """Calculate recent execution failure rate"""
        # TODO: Implement actual execution tracking
        return 0.05  # Placeholder
    
    async def _would_exceed_daily_loss_limit(self, symbol: str, position_size: Decimal,
                                           market_data: Dict[str, Any]) -> bool:
        """Check if trade would exceed daily loss limit"""
        # TODO: Implement actual daily loss tracking
        return False
    
    async def _would_exceed_portfolio_risk(self, symbol: str, position_size: Decimal,
                                         market_data: Dict[str, Any]) -> bool:
        """Check if trade would exceed portfolio risk limits"""
        # TODO: Implement actual portfolio risk calculation
        return False
    
    # ============================================================================
    # PUBLIC API METHODS
    # ============================================================================
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'emergency_stop': {
                'active': self.emergency_stop.is_active,
                'reason': self.emergency_stop.activation_reason,
                'activated_at': self.emergency_stop.activated_at.isoformat() if self.emergency_stop.activated_at else None
            },
            'circuit_breakers': {
                breaker_type.value: {
                    'active': breaker.is_active,
                    'triggered_at': breaker.triggered_at.isoformat() if breaker.triggered_at else None,
                    'threshold': breaker.threshold,
                    'trigger_value': breaker.trigger_value
                }
                for breaker_type, breaker in self.circuit_breakers.items()
            },
            'recent_performance': {
                'trades_validated': len(self.trade_history),
                'trades_blocked': len(self.blocked_trades),
                'block_rate': len(self.blocked_trades) / max(1, len(self.trade_history))
            }
        }
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        return {
            'ml_risk_params': self.ml_risk_params,
            'active_circuit_breakers': len(self._get_active_circuit_breakers()),
            'emergency_stop_active': self.emergency_stop.is_active,
            'recent_block_rate': len(self.blocked_trades) / max(1, len(self.trade_history)),
            'last_validation': self.trade_history[-1]['timestamp'] if self.trade_history else None
        }