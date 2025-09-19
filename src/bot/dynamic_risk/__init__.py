"""
Dynamic Risk Management System.

This is the main integration module for the dynamic risk management system that combines:

- Adaptive volatility monitoring with regime detection
- Dynamic correlation analysis and portfolio risk assessment
- Automated dynamic hedging with real-time rebalancing
- Risk-adjusted position sizing based on market conditions
- Cross-asset risk factor analysis and management
- Automated risk scaling during volatility regime changes
- Portfolio performance-based risk adaptation
- Real-time risk monitoring and alert systems

The system provides sophisticated risk management that adapts to changing market
conditions through continuous monitoring of volatility regimes, correlation
structures, and portfolio performance.
"""

# Import all main components
from .volatility_monitor import (
    AdaptiveVolatilityMonitor,
    VolatilityMetrics,
    VolatilityRegime,
    TrendState,
    VolatilityEstimator,
    VolatilityRegimeDetector,
    GARCHModel,
    EWMAModel
)

from .correlation_analysis import (
    DynamicCorrelationAnalyzer,
    CorrelationMetrics,
    PortfolioCorrelationMetrics,
    CorrelationRegime,
    CorrelationTrend,
    DynamicCorrelationCalculator,
    CorrelationRegimeDetector,
    PortfolioCorrelationAnalyzer
)

from .dynamic_hedging import (
    DynamicHedgingSystem,
    HedgePosition,
    HedgeRatio,
    HedgeType,
    HedgeStatus,
    RebalanceSignal,
    HedgeRatioCalculator,
    HedgePositionManager
)

# Main components exported at package level
__all__ = [
    # Main system
    'DynamicRiskSystem',
    'RiskRegime',
    'AdaptationSignal',
    'RiskAdjustment',
    'PortfolioRiskMetrics',
    
    # Volatility monitoring  
    'AdaptiveVolatilityMonitor',
    'VolatilityMetrics',
    'VolatilityRegime',
    'TrendState',
    'VolatilityEstimator',
    'VolatilityRegimeDetector',
    'GARCHModel',
    'EWMAModel',
    
    # Correlation analysis
    'DynamicCorrelationAnalyzer',
    'CorrelationMetrics', 
    'PortfolioCorrelationMetrics',
    'CorrelationRegime',
    'CorrelationTrend',
    'DynamicCorrelationCalculator',
    'CorrelationRegimeDetector',
    'PortfolioCorrelationAnalyzer',
    
    # Dynamic hedging
    'DynamicHedgingSystem',
    'HedgePosition',
    'HedgeRatio', 
    'HedgeType',
    'HedgeStatus',
    'RebalanceSignal',
    'HedgeRatioCalculator',
    'HedgePositionManager',
    
    # Internal components
    'RiskRegimeDetector',
    'RiskAdjustmentCalculator'
]

import asyncio
import threading
import time
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import statistics
import sqlite3
import json

from .volatility_monitor import (
    AdaptiveVolatilityMonitor,
    VolatilityMetrics,
    VolatilityRegime,
    TrendState
)
from .correlation_analysis import (
    DynamicCorrelationAnalyzer,
    CorrelationMetrics,
    PortfolioCorrelationMetrics,
    CorrelationRegime
)
from .dynamic_hedging import (
    DynamicHedgingSystem,
    HedgePosition,
    HedgeType,
    RebalanceSignal
)
from ..utils.logging import TradingLogger


class RiskRegime(Enum):
    """Overall risk regime classification."""
    LOW_RISK = "low_risk"         # Low volatility, normal correlations
    NORMAL_RISK = "normal_risk"   # Normal volatility and correlations
    HIGH_RISK = "high_risk"       # High volatility or correlations
    CRISIS_RISK = "crisis_risk"   # Extreme volatility and correlations


class AdaptationSignal(Enum):
    """Signals for risk parameter adaptation."""
    NO_CHANGE = "no_change"
    REDUCE_RISK = "reduce_risk"       # Reduce position sizes and exposure
    INCREASE_RISK = "increase_risk"   # Can increase exposure
    HEDGE_MORE = "hedge_more"         # Increase hedging activity
    REBALANCE = "rebalance"          # Rebalance portfolio allocations


@dataclass
class RiskAdjustment:
    """Container for risk adjustment parameters."""
    
    symbol: str
    timestamp: datetime
    
    # Position sizing adjustments
    volatility_scalar: float      # Position size multiplier based on volatility
    correlation_scalar: float     # Adjustment for correlation regime
    regime_scalar: float         # Overall regime-based adjustment
    
    # Combined adjustment
    total_adjustment: float       # Combined position size adjustment
    
    # Recommended actions
    adaptation_signal: AdaptationSignal
    hedge_recommendation: bool    # Whether to create/increase hedging
    
    # Risk metrics
    estimated_var: float         # Estimated Value at Risk
    estimated_volatility: float  # Estimated portfolio volatility
    diversification_benefit: float # Expected diversification benefit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'volatility_scalar': self.volatility_scalar,
            'correlation_scalar': self.correlation_scalar,
            'regime_scalar': self.regime_scalar,
            'total_adjustment': self.total_adjustment,
            'adaptation_signal': self.adaptation_signal.value,
            'hedge_recommendation': self.hedge_recommendation,
            'estimated_var': self.estimated_var,
            'estimated_volatility': self.estimated_volatility,
            'diversification_benefit': self.diversification_benefit
        }


@dataclass
class PortfolioRiskMetrics:
    """Comprehensive portfolio risk metrics."""
    
    timestamp: datetime
    symbols: List[str]
    
    # Risk regime
    risk_regime: RiskRegime
    regime_confidence: float
    regime_change_probability: float
    
    # Volatility metrics
    portfolio_volatility: float       # Overall portfolio volatility
    weighted_avg_volatility: float    # Weighted average of individual vols
    volatility_diversification: float # Volatility diversification benefit
    
    # Correlation metrics
    average_correlation: float        # Average pairwise correlation
    max_correlation: float           # Maximum pairwise correlation
    correlation_concentration: float  # Concentration in correlation structure
    
    # Risk concentration
    concentration_risk: float        # Overall concentration risk
    effective_positions: float       # Effective number of independent positions
    largest_risk_contribution: float # Largest single position risk contribution
    
    # Dynamic adjustments
    risk_adjustments: Dict[str, RiskAdjustment]
    
    # Hedging metrics
    total_hedge_effectiveness: float # Overall hedge effectiveness
    hedged_exposure_pct: float      # Percentage of exposure that's hedged
    hedge_cost_pct: float           # Hedging cost as % of portfolio
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbols': self.symbols,
            'risk_regime': self.risk_regime.value,
            'regime_confidence': self.regime_confidence,
            'regime_change_probability': self.regime_change_probability,
            'portfolio_volatility': self.portfolio_volatility,
            'weighted_avg_volatility': self.weighted_avg_volatility,
            'volatility_diversification': self.volatility_diversification,
            'average_correlation': self.average_correlation,
            'max_correlation': self.max_correlation,
            'correlation_concentration': self.correlation_concentration,
            'concentration_risk': self.concentration_risk,
            'effective_positions': self.effective_positions,
            'largest_risk_contribution': self.largest_risk_contribution,
            'risk_adjustments': {k: v.to_dict() for k, v in self.risk_adjustments.items()},
            'total_hedge_effectiveness': self.total_hedge_effectiveness,
            'hedged_exposure_pct': self.hedged_exposure_pct,
            'hedge_cost_pct': self.hedge_cost_pct
        }


class RiskRegimeDetector:
    """
    Detect overall portfolio risk regime based on volatility and correlation regimes.
    
    This class combines information from volatility and correlation monitoring
    to classify the overall risk environment.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("RiskRegimeDetector")
        
        # Regime history
        self.regime_history: deque = deque(maxlen=100)
        self.current_regime = RiskRegime.NORMAL_RISK
        self.regime_start_time = datetime.now()
        
    def _default_config(self) -> Dict:
        """Default configuration for risk regime detector."""
        return {
            'crisis_vol_threshold': 0.90,    # Vol percentile for crisis
            'high_vol_threshold': 0.75,      # Vol percentile for high risk
            'crisis_corr_threshold': 0.80,   # Correlation for crisis
            'high_corr_threshold': 0.60,     # Correlation for high risk
            'regime_smoothing': 0.8,         # Smoothing factor for regime changes
            'min_regime_duration': 1800,     # Minimum 30 minutes in regime
        }
    
    def detect_risk_regime(
        self,
        volatility_metrics: Dict[str, VolatilityMetrics],
        portfolio_correlation_metrics: Optional[PortfolioCorrelationMetrics] = None
    ) -> Tuple[RiskRegime, float]:
        """Detect current risk regime and confidence."""
        try:
            # Analyze volatility regimes
            vol_regime_scores = defaultdict(int)
            total_vol_metrics = 0
            
            for symbol, vol_metrics in volatility_metrics.items():
                vol_regime_scores[vol_metrics.vol_regime] += 1
                total_vol_metrics += 1
            
            # Calculate volatility regime score (0-1, higher = more risky)
            vol_score = 0.0
            if total_vol_metrics > 0:
                normal_pct = vol_regime_scores.get('normal', 0) / total_vol_metrics
                high_pct = vol_regime_scores.get('high', 0) / total_vol_metrics
                very_high_pct = vol_regime_scores.get('very_high', 0) / total_vol_metrics
                extreme_pct = vol_regime_scores.get('extreme', 0) / total_vol_metrics
                
                vol_score = (0.5 * normal_pct + 0.7 * high_pct + 
                           0.9 * very_high_pct + 1.0 * extreme_pct)
            
            # Analyze correlation regime
            corr_score = 0.5  # Default neutral
            if portfolio_correlation_metrics:
                avg_corr = abs(portfolio_correlation_metrics.average_correlation)
                max_corr = abs(portfolio_correlation_metrics.max_correlation)
                
                # Correlation score based on average and maximum correlations
                if avg_corr > self.config['crisis_corr_threshold']:
                    corr_score = 1.0
                elif avg_corr > self.config['high_corr_threshold']:
                    corr_score = 0.8
                elif max_corr > self.config['crisis_corr_threshold']:
                    corr_score = 0.9
                elif max_corr > self.config['high_corr_threshold']:
                    corr_score = 0.7
                else:
                    corr_score = min(0.6, avg_corr / self.config['high_corr_threshold'])
            
            # Combine volatility and correlation scores
            combined_score = (vol_score + corr_score) / 2
            
            # Determine risk regime
            if combined_score >= 0.85:
                regime = RiskRegime.CRISIS_RISK
                confidence = min(1.0, combined_score)
            elif combined_score >= 0.65:
                regime = RiskRegime.HIGH_RISK
                confidence = combined_score
            elif combined_score <= 0.35:
                regime = RiskRegime.LOW_RISK
                confidence = 1.0 - combined_score
            else:
                regime = RiskRegime.NORMAL_RISK
                confidence = 0.8
            
            # Apply regime smoothing to prevent rapid changes
            if regime != self.current_regime:
                # Check minimum duration
                time_in_regime = datetime.now() - self.regime_start_time
                if time_in_regime.total_seconds() < self.config['min_regime_duration']:
                    # Stay in current regime but reduce confidence
                    regime = self.current_regime
                    confidence *= self.config['regime_smoothing']
                else:
                    # Accept regime change
                    self.current_regime = regime
                    self.regime_start_time = datetime.now()
            
            # Record regime history
            regime_record = {
                'timestamp': datetime.now(),
                'regime': regime,
                'confidence': confidence,
                'vol_score': vol_score,
                'corr_score': corr_score
            }
            self.regime_history.append(regime_record)
            
            return regime, confidence
            
        except Exception as e:
            self.logger.error(f"Error detecting risk regime: {e}")
            return RiskRegime.NORMAL_RISK, 0.5
    
    def get_regime_change_probability(self) -> float:
        """Calculate probability of regime change based on recent history."""
        if len(self.regime_history) < 10:
            return 0.1
        
        try:
            # Look at volatility in regime scores
            recent_scores = [r['vol_score'] + r['corr_score'] for r in list(self.regime_history)[-10:]]
            score_volatility = np.std(recent_scores)
            
            # Higher score volatility = higher probability of regime change
            change_probability = min(1.0, float(score_volatility) * 2)
            
            return float(change_probability)
            
        except Exception:
            return 0.1


class RiskAdjustmentCalculator:
    """
    Calculate risk adjustments based on market conditions.
    
    This class determines position size adjustments and risk management
    actions based on volatility and correlation regimes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("RiskAdjustmentCalculator")
        
    def _default_config(self) -> Dict:
        """Default configuration for risk adjustment calculator."""
        return {
            'base_volatility': 0.20,         # Base annual volatility assumption
            'volatility_target': 0.15,       # Target portfolio volatility
            'max_position_adjustment': 0.5,  # Maximum position size reduction
            'correlation_penalty': 0.8,      # Penalty factor for high correlations
            'regime_adjustments': {          # Regime-based multipliers
                'low_risk': 1.2,
                'normal_risk': 1.0,
                'high_risk': 0.7,
                'crisis_risk': 0.4
            },
            'hedge_thresholds': {            # When to recommend hedging
                'high_risk': 0.3,           # Hedge if 30%+ of portfolio in high vol
                'crisis_risk': 0.1          # Hedge if 10%+ of portfolio in crisis
            }
        }
    
    def calculate_volatility_adjustment(
        self,
        symbol: str,
        volatility_metrics: VolatilityMetrics
    ) -> float:
        """Calculate position size adjustment based on volatility."""
        try:
            current_vol = volatility_metrics.realized_vol_24h
            base_vol = self.config['base_volatility']
            
            if current_vol <= 0:
                return 1.0
            
            # Inverse volatility scaling (higher vol = smaller positions)
            vol_adjustment = math.sqrt(base_vol / current_vol)
            
            # Apply regime-specific adjustment
            regime_multiplier = self.config['regime_adjustments'].get(
                volatility_metrics.vol_regime.value, 1.0
            )
            
            vol_adjustment *= regime_multiplier
            
            # Bound the adjustment
            vol_adjustment = max(
                self.config['max_position_adjustment'],
                min(2.0, vol_adjustment)
            )
            
            return vol_adjustment
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment for {symbol}: {e}")
            return 1.0
    
    def calculate_correlation_adjustment(
        self,
        symbol: str,
        portfolio_correlation_metrics: Optional[PortfolioCorrelationMetrics] = None
    ) -> float:
        """Calculate position size adjustment based on correlations."""
        if not portfolio_correlation_metrics:
            return 1.0
        
        try:
            avg_correlation = abs(portfolio_correlation_metrics.average_correlation)
            
            # Higher correlation = smaller positions (less diversification)
            if avg_correlation > 0.8:
                corr_adjustment = 0.5
            elif avg_correlation > 0.6:
                corr_adjustment = 0.7
            elif avg_correlation > 0.4:
                corr_adjustment = 0.9
            else:
                corr_adjustment = 1.0
            
            # Apply correlation penalty
            corr_adjustment *= self.config['correlation_penalty']
            corr_adjustment = max(corr_adjustment, self.config['max_position_adjustment'])
            
            return corr_adjustment
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation adjustment for {symbol}: {e}")
            return 1.0
    
    def calculate_regime_adjustment(
        self,
        risk_regime: RiskRegime,
        regime_confidence: float
    ) -> float:
        """Calculate position size adjustment based on overall risk regime."""
        regime_multiplier = self.config['regime_adjustments'].get(risk_regime.value, 1.0)
        
        # Apply confidence weighting
        confidence_adjusted = (regime_multiplier * regime_confidence + 
                             1.0 * (1 - regime_confidence))
        
        return confidence_adjusted
    
    def calculate_comprehensive_adjustment(
        self,
        symbol: str,
        volatility_metrics: VolatilityMetrics,
        risk_regime: RiskRegime,
        regime_confidence: float,
        portfolio_correlation_metrics: Optional[PortfolioCorrelationMetrics] = None
    ) -> RiskAdjustment:
        """Calculate comprehensive risk adjustment for a symbol."""
        try:
            # Individual adjustments
            vol_scalar = self.calculate_volatility_adjustment(symbol, volatility_metrics)
            corr_scalar = self.calculate_correlation_adjustment(symbol, portfolio_correlation_metrics)
            regime_scalar = self.calculate_regime_adjustment(risk_regime, regime_confidence)
            
            # Combined adjustment (multiplicative)
            total_adjustment = vol_scalar * corr_scalar * regime_scalar
            
            # Determine adaptation signal
            if total_adjustment < 0.6:
                signal = AdaptationSignal.REDUCE_RISK
            elif total_adjustment > 1.3:
                signal = AdaptationSignal.INCREASE_RISK
            elif risk_regime in [RiskRegime.HIGH_RISK, RiskRegime.CRISIS_RISK]:
                signal = AdaptationSignal.HEDGE_MORE
            else:
                signal = AdaptationSignal.NO_CHANGE
            
            # Hedge recommendation
            hedge_recommendation = (
                risk_regime == RiskRegime.CRISIS_RISK or
                (risk_regime == RiskRegime.HIGH_RISK and regime_confidence > 0.7) or
                volatility_metrics.vol_regime.value in ['very_high', 'extreme']
            )
            
            # Risk metrics estimation
            estimated_vol = volatility_metrics.vol_forecast_24h
            estimated_var = estimated_vol * 2.33  # 99% VaR approximation
            
            # Diversification benefit
            diversification_benefit = 1.0
            if portfolio_correlation_metrics:
                diversification_benefit = portfolio_correlation_metrics.diversification_ratio
            
            return RiskAdjustment(
                symbol=symbol,
                timestamp=datetime.now(),
                volatility_scalar=vol_scalar,
                correlation_scalar=corr_scalar,
                regime_scalar=regime_scalar,
                total_adjustment=total_adjustment,
                adaptation_signal=signal,
                hedge_recommendation=hedge_recommendation,
                estimated_var=estimated_var,
                estimated_volatility=estimated_vol,
                diversification_benefit=diversification_benefit
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive adjustment for {symbol}: {e}")
            return RiskAdjustment(
                symbol=symbol,
                timestamp=datetime.now(),
                volatility_scalar=1.0,
                correlation_scalar=1.0,
                regime_scalar=1.0,
                total_adjustment=1.0,
                adaptation_signal=AdaptationSignal.NO_CHANGE,
                hedge_recommendation=False,
                estimated_var=0.0,
                estimated_volatility=0.0,
                diversification_benefit=1.0
            )


class DynamicRiskSystem:
    """
    Main dynamic risk management system.
    
    This class integrates all components to provide comprehensive
    dynamic risk management with adaptive position sizing and hedging.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = TradingLogger("DynamicRiskSystem")
        
        # Core components
        self.volatility_monitor = AdaptiveVolatilityMonitor(self.config.get('volatility_monitor', {}))
        self.correlation_analyzer = DynamicCorrelationAnalyzer(self.config.get('correlation_analyzer', {}))
        self.hedging_system = DynamicHedgingSystem(self.config.get('hedging_system', {}))
        
        # Risk analysis components
        self.regime_detector = RiskRegimeDetector(self.config.get('regime_detector', {}))
        self.adjustment_calculator = RiskAdjustmentCalculator(self.config.get('adjustment_calculator', {}))
        
        # Current state
        self.current_risk_metrics: Optional[PortfolioRiskMetrics] = None
        self.current_adjustments: Dict[str, RiskAdjustment] = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Callbacks
        self.risk_callbacks: List[Callable[[PortfolioRiskMetrics], None]] = []
        self.adjustment_callbacks: List[Callable[[str, RiskAdjustment], None]] = []
        
        # Database
        self.db_path = self.config.get('database_path', 'dynamic_risk.db')
        self._init_database()
        
        # Setup callbacks
        self._setup_callbacks()
        
    def _default_config(self) -> Dict:
        """Default configuration for dynamic risk system."""
        return {
            'monitoring_interval': 300,    # 5 minutes
            'database_path': 'dynamic_risk.db',
            'enable_persistence': True,
            'portfolio_symbols': [],       # Symbols to monitor
            'auto_hedge_enabled': True,    # Enable automatic hedging
            'auto_adjust_enabled': True,   # Enable automatic adjustments
            'volatility_monitor': {},
            'correlation_analyzer': {},
            'hedging_system': {},
            'regime_detector': {},
            'adjustment_calculator': {}
        }
    
    def _init_database(self) -> None:
        """Initialize SQLite database for dynamic risk data."""
        if not self.config['enable_persistence']:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_adjustments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_time ON risk_metrics (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_adj_symbol_time ON risk_adjustments (symbol, timestamp)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dynamic risk database: {e}")
    
    def _setup_callbacks(self) -> None:
        """Setup callbacks between components."""
        # Volatility regime change callback
        def volatility_callback(symbol: str, metrics: VolatilityMetrics):
            self.logger.info(f"Volatility regime change for {symbol}: {metrics.vol_regime.value}")
        
        self.volatility_monitor.add_regime_callback(volatility_callback)
        
        # Correlation regime change callback  
        def correlation_callback(symbol1: str, symbol2: str, metrics: CorrelationMetrics):
            if metrics.correlation_regime == CorrelationRegime.CRISIS_CORRELATION:
                self.logger.warning(f"Crisis correlation detected: {symbol1}-{symbol2}")
        
        self.correlation_analyzer.add_correlation_callback(correlation_callback)
        
        # Hedge event callback
        def hedge_callback(hedge_id: str, position: HedgePosition):
            self.logger.info(f"Hedge event for {hedge_id}: {position.status.value}")
        
        self.hedging_system.add_hedge_callback(hedge_callback)
    
    def add_market_data(
        self,
        symbol: str,
        price: float,
        position_size: float = 0.0,
        open_price: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
        volume: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add market data to all components."""
        # Update volatility monitor
        self.volatility_monitor.add_market_data(
            symbol, price, open_price, high, low, volume, timestamp
        )
        
        # Update correlation analyzer with returns
        if symbol in self.volatility_monitor.volatility_estimator.price_data:
            price_data = list(self.volatility_monitor.volatility_estimator.price_data[symbol])
            if len(price_data) >= 2:
                old_price = price_data[-2]['price']
                if old_price > 0:
                    return_value = math.log(price / old_price)
                    self.correlation_analyzer.add_return_data(symbol, return_value, timestamp)
        
        # Update hedging system
        self.hedging_system.update_market_data(symbol, price, position_size, timestamp)
    
    def calculate_portfolio_risk_metrics(
        self,
        symbols: Optional[List[str]] = None,
        positions: Optional[Dict[str, float]] = None
    ) -> Optional[PortfolioRiskMetrics]:
        """Calculate comprehensive portfolio risk metrics."""
        symbols = symbols or self.config['portfolio_symbols']
        if not symbols:
            return None
        
        try:
            # Get volatility metrics for all symbols
            volatility_metrics = {}
            for symbol in symbols:
                vol_metrics = self.volatility_monitor.get_current_metrics(symbol)
                if vol_metrics:
                    volatility_metrics[symbol] = vol_metrics
            
            if not volatility_metrics:
                return None
            
            # Get portfolio correlation metrics
            portfolio_corr_metrics = self.correlation_analyzer.calculate_portfolio_metrics(symbols)
            
            # Detect risk regime
            risk_regime, regime_confidence = self.regime_detector.detect_risk_regime(
                volatility_metrics, portfolio_corr_metrics
            )
            
            # Calculate risk adjustments for each symbol
            risk_adjustments = {}
            for symbol in symbols:
                if symbol in volatility_metrics:
                    adjustment = self.adjustment_calculator.calculate_comprehensive_adjustment(
                        symbol,
                        volatility_metrics[symbol],
                        risk_regime,
                        regime_confidence,
                        portfolio_corr_metrics
                    )
                    risk_adjustments[symbol] = adjustment
            
            # Portfolio-level calculations
            portfolio_volatility = 0.0
            weighted_avg_volatility = 0.0
            
            if portfolio_corr_metrics:
                # Use correlation-based portfolio volatility
                portfolio_volatility = math.sqrt(
                    np.mean(portfolio_corr_metrics.correlation_eigenvalues) * 
                    np.mean([vm.realized_vol_24h for vm in volatility_metrics.values()])**2
                )
                
                # Weighted average volatility
                weights = [1.0 / len(symbols)] * len(symbols)  # Equal weights default
                weighted_avg_volatility = float(np.average(
                    [vm.realized_vol_24h for vm in volatility_metrics.values()], 
                    weights=weights
                ))
            
            # Volatility diversification benefit
            volatility_diversification = float(
                weighted_avg_volatility / portfolio_volatility 
                if portfolio_volatility > 0 else 1.0
            )
            
            # Concentration metrics
            concentration_risk = 1.0 / len(symbols)  # Simple equal-weight assumption
            effective_positions = len(symbols)
            largest_risk_contribution = 1.0 / len(symbols)
            
            if portfolio_corr_metrics:
                concentration_risk = portfolio_corr_metrics.concentration_risk
                effective_positions = portfolio_corr_metrics.effective_assets
            
            # Hedging metrics
            hedge_summary = self.hedging_system.get_hedge_summary()
            total_hedge_effectiveness = hedge_summary.get('avg_effectiveness', 0.0)
            hedged_exposure_pct = min(1.0, hedge_summary.get('active_hedges', 0) / len(symbols))
            hedge_cost_pct = 0.01  # Placeholder - would calculate from actual hedge costs
            
            # Regime change probability
            regime_change_probability = self.regime_detector.get_regime_change_probability()
            
            # Create portfolio risk metrics
            portfolio_risk_metrics = PortfolioRiskMetrics(
                timestamp=datetime.now(),
                symbols=symbols,
                risk_regime=risk_regime,
                regime_confidence=regime_confidence,
                regime_change_probability=regime_change_probability,
                portfolio_volatility=portfolio_volatility,
                weighted_avg_volatility=weighted_avg_volatility,
                volatility_diversification=volatility_diversification,
                average_correlation=portfolio_corr_metrics.average_correlation if portfolio_corr_metrics else 0.0,
                max_correlation=portfolio_corr_metrics.max_correlation if portfolio_corr_metrics else 0.0,
                correlation_concentration=concentration_risk,
                concentration_risk=concentration_risk,
                effective_positions=effective_positions,
                largest_risk_contribution=largest_risk_contribution,
                risk_adjustments=risk_adjustments,
                total_hedge_effectiveness=total_hedge_effectiveness,
                hedged_exposure_pct=hedged_exposure_pct,
                hedge_cost_pct=hedge_cost_pct
            )
            
            # Store current metrics
            self.current_risk_metrics = portfolio_risk_metrics
            self.current_adjustments = risk_adjustments
            
            # Save to database
            if self.config['enable_persistence']:
                self._save_risk_metrics(portfolio_risk_metrics)
            
            # Trigger callbacks
            for callback in self.risk_callbacks:
                try:
                    callback(portfolio_risk_metrics)
                except Exception as e:
                    self.logger.error(f"Error in risk callback: {e}")
            
            return portfolio_risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {e}")
            return None
    
    def process_risk_adjustments(self) -> None:
        """Process and execute risk adjustments."""
        if not self.current_adjustments:
            return
        
        try:
            for symbol, adjustment in self.current_adjustments.items():
                # Log adjustment recommendations
                self.logger.info(
                    f"Risk adjustment for {symbol}: "
                    f"scalar={adjustment.total_adjustment:.3f}, "
                    f"signal={adjustment.adaptation_signal.value}, "
                    f"hedge={adjustment.hedge_recommendation}"
                )
                
                # Auto-hedge if recommended and enabled
                if (adjustment.hedge_recommendation and 
                    self.config['auto_hedge_enabled']):
                    
                    # Check if symbol already has active hedge
                    existing_hedges = self.hedging_system.position_manager.get_hedge_positions_for_symbol(symbol)
                    active_hedges = [h for h in existing_hedges if h.status.value == 'active']
                    
                    if not active_hedges:
                        hedge_id = self.hedging_system.create_hedge(symbol)
                        if hedge_id:
                            self.logger.info(f"Auto-created hedge {hedge_id} for {symbol}")
                
                # Trigger adjustment callbacks
                for callback in self.adjustment_callbacks:
                    try:
                        callback(symbol, adjustment)
                    except Exception as e:
                        self.logger.error(f"Error in adjustment callback: {e}")
                
                # Save adjustment to database
                if self.config['enable_persistence']:
                    self._save_risk_adjustment(adjustment)
            
        except Exception as e:
            self.logger.error(f"Error processing risk adjustments: {e}")
    
    def _save_risk_metrics(self, metrics: PortfolioRiskMetrics) -> None:
        """Save portfolio risk metrics to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            data = json.dumps(metrics.to_dict())
            
            cursor.execute("""
                INSERT INTO risk_metrics (timestamp, data)
                VALUES (?, ?)
            """, (metrics.timestamp, data))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save risk metrics: {e}")
    
    def _save_risk_adjustment(self, adjustment: RiskAdjustment) -> None:
        """Save risk adjustment to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            data = json.dumps(adjustment.to_dict())
            
            cursor.execute("""
                INSERT INTO risk_adjustments (symbol, timestamp, data)
                VALUES (?, ?, ?)
            """, (adjustment.symbol, adjustment.timestamp, data))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save risk adjustment: {e}")
    
    def add_risk_callback(self, callback: Callable[[PortfolioRiskMetrics], None]) -> None:
        """Add callback for risk metrics updates."""
        self.risk_callbacks.append(callback)
    
    def add_adjustment_callback(self, callback: Callable[[str, RiskAdjustment], None]) -> None:
        """Add callback for risk adjustments."""
        self.adjustment_callbacks.append(callback)
    
    def start_monitoring(self, symbols: Optional[List[str]] = None) -> None:
        """Start dynamic risk monitoring."""
        if self.is_monitoring:
            return
        
        if symbols:
            self.config['portfolio_symbols'] = symbols
        
        # Start component monitoring
        self.volatility_monitor.start_monitoring()
        self.correlation_analyzer.start_monitoring(symbols)
        self.hedging_system.start_monitoring()
        
        # Start main monitoring loop
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started dynamic risk monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop dynamic risk monitoring."""
        self.is_monitoring = False
        
        # Stop component monitoring
        self.volatility_monitor.stop_monitoring()
        self.correlation_analyzer.stop_monitoring()
        self.hedging_system.stop_monitoring()
        
        # Stop main monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        self.logger.info("Stopped dynamic risk monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Calculate portfolio risk metrics
                risk_metrics = self.calculate_portfolio_risk_metrics()
                
                if risk_metrics:
                    self.logger.debug(
                        f"Updated portfolio risk metrics: "
                        f"regime={risk_metrics.risk_regime.value}, "
                        f"portfolio_vol={risk_metrics.portfolio_volatility:.3f}, "
                        f"avg_corr={risk_metrics.average_correlation:.3f}"
                    )
                    
                    # Process risk adjustments
                    self.process_risk_adjustments()
                
                # Sleep until next update
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in dynamic risk monitoring loop: {e}")
                time.sleep(60)  # Error backoff
    
    def get_current_risk_metrics(self) -> Optional[PortfolioRiskMetrics]:
        """Get current portfolio risk metrics."""
        return self.current_risk_metrics
    
    def get_risk_adjustment(self, symbol: str) -> Optional[RiskAdjustment]:
        """Get current risk adjustment for a symbol."""
        return self.current_adjustments.get(symbol)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        summary = {
            'risk_regime': self.current_risk_metrics.risk_regime.value if self.current_risk_metrics else 'unknown',
            'portfolio_volatility': self.current_risk_metrics.portfolio_volatility if self.current_risk_metrics else 0.0,
            'average_correlation': self.current_risk_metrics.average_correlation if self.current_risk_metrics else 0.0,
            'effective_positions': self.current_risk_metrics.effective_positions if self.current_risk_metrics else 0.0,
            'hedge_effectiveness': self.current_risk_metrics.total_hedge_effectiveness if self.current_risk_metrics else 0.0,
            'volatility_monitoring': self.volatility_monitor.is_monitoring,
            'correlation_monitoring': self.correlation_analyzer.is_monitoring,
            'hedging_monitoring': self.hedging_system.is_monitoring,
            'total_adjustments': len(self.current_adjustments),
            'hedge_summary': self.hedging_system.get_hedge_summary()
        }
        
        return summary