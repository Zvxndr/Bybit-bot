"""
ML Performance Monitor - Comprehensive ML Strategy Performance Tracking

This module provides comprehensive monitoring and analysis of ML-enhanced trading
strategies, tracking prediction accuracy, financial performance, risk metrics,
and providing adaptive feedback for continuous improvement.

Key Features:
- Real-time prediction accuracy tracking
- Financial performance attribution analysis
- Risk-adjusted return metrics (Sharpe, Sortino, Calmar)
- ML model performance comparison and ranking
- Prediction confidence calibration analysis
- Market regime performance breakdown
- Automated performance reporting and alerts
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass, field
import json
from collections import defaultdict, deque
import statistics

# Import ML components
from .ml_feature_pipeline import MLFeatures, MLPrediction, MLSignalType
from .ml_model_manager import MLModelManager, EnsemblePrediction
from .ml_strategy_orchestrator import CombinedSignal, MarketRegime
from .ml_execution_optimizer import ExecutionPerformance

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class PerformanceMetric(Enum):
    """Performance metric types"""
    PREDICTION_ACCURACY = "prediction_accuracy"
    RETURN_ATTRIBUTION = "return_attribution"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"

class AlertLevel(Enum):
    """Performance alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class PredictionOutcome:
    """Outcome of a prediction for performance tracking"""
    prediction_id: str
    symbol: str
    prediction: MLPrediction
    actual_signal: MLSignalType
    actual_return: float
    prediction_horizon: timedelta
    was_accurate: bool
    accuracy_score: float  # 0-1 score
    timestamp: datetime
    market_regime: MarketRegime
    
@dataclass
class TradePerformance:
    """Performance metrics for a completed trade"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    side: str  # 'buy' or 'sell'
    pnl: Decimal
    return_pct: float
    holding_period: timedelta
    ml_signal_confidence: float
    execution_quality: float  # From execution optimizer
    market_regime: MarketRegime

@dataclass
class ModelPerformanceMetrics:
    """Comprehensive performance metrics for an ML model"""
    model_name: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_calibration: float  # How well confidence matches accuracy
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return_per_trade: float
    performance_by_regime: Dict[MarketRegime, Dict[str, float]]
    recent_performance_trend: List[float]  # Last N accuracy scores
    last_updated: datetime

@dataclass
class PerformanceAlert:
    """Performance monitoring alert"""
    alert_id: str
    level: AlertLevel
    metric: PerformanceMetric
    message: str
    current_value: float
    threshold_value: float
    model_name: Optional[str]
    timestamp: datetime
    auto_actions: List[str]  # Suggested automatic actions

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    report_id: str
    period_start: datetime
    period_end: datetime
    total_trades: int
    total_pnl: Decimal
    return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    model_performance: Dict[str, ModelPerformanceMetrics]
    execution_performance: Dict[str, float]
    regime_breakdown: Dict[MarketRegime, Dict[str, float]]
    recommendations: List[str]
    generated_at: datetime

# ============================================================================
# ML PERFORMANCE MONITOR
# ============================================================================

class MLPerformanceMonitor:
    """
    Comprehensive ML Performance Monitor
    
    Tracks and analyzes performance of ML-enhanced trading strategies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Performance data storage
        self.prediction_outcomes: deque = deque(maxlen=10000)
        self.trade_performances: deque = deque(maxlen=5000)
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.performance_alerts: List[PerformanceAlert] = []
        
        # Real-time tracking
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.daily_pnl: Dict[str, Decimal] = defaultdict(lambda: Decimal('0'))
        self.rolling_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance benchmarks
        self.benchmarks: Dict[str, List[float]] = {
            'buy_and_hold': [],
            'random_trading': [],
            'simple_momentum': []
        }
        
        # Initialize monitoring
        asyncio.create_task(self._initialize_monitor())
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for performance monitor"""
        return {
            'tracking': {
                'prediction_accuracy_window': 100,
                'performance_update_frequency': 60,  # seconds
                'report_generation_frequency': 86400,  # daily
                'min_trades_for_metrics': 10
            },
            'alert_thresholds': {
                'accuracy_decline': 0.1,  # 10% decline in accuracy
                'drawdown_limit': 0.15,  # 15% max drawdown
                'sharpe_minimum': 0.5,  # Minimum Sharpe ratio
                'win_rate_minimum': 0.45,  # 45% minimum win rate
                'confidence_calibration': 0.2  # Max deviation from perfect calibration
            },
            'performance_targets': {
                'min_accuracy': 0.55,  # 55% prediction accuracy
                'target_sharpe': 1.5,
                'target_win_rate': 0.6,
                'max_drawdown': 0.1,
                'min_profit_factor': 1.2
            },
            'regime_analysis': {
                'track_by_regime': True,
                'min_samples_per_regime': 20,
                'regime_performance_threshold': 0.05  # 5% performance difference
            },
            'reporting': {
                'include_execution_analysis': True,
                'include_attribution_analysis': True,
                'include_recommendations': True,
                'export_format': 'json'
            }
        }
    
    async def _initialize_monitor(self):
        """Initialize the performance monitor"""
        logger.info("Initializing ML Performance Monitor...")
        
        try:
            # Load historical performance data
            await self._load_historical_data()
            
            # Initialize model metrics
            self._initialize_model_metrics()
            
            # Start background monitoring tasks
            asyncio.create_task(self._background_monitoring_loop())
            asyncio.create_task(self._daily_report_generator())
            
            logger.info("ML Performance Monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML Performance Monitor: {e}")
    
    async def _load_historical_data(self):
        """Load historical performance data"""
        try:
            # In a real implementation, this would load from database
            logger.info("Loaded historical performance data")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def _initialize_model_metrics(self):
        """Initialize performance metrics for all models"""
        # Initialize with default values
        default_models = ['lightgbm_trend', 'lightgbm_mean_reversion', 'xgboost_momentum', 'ensemble']
        
        for model_name in default_models:
            self.model_metrics[model_name] = ModelPerformanceMetrics(
                model_name=model_name,
                total_predictions=0,
                correct_predictions=0,
                accuracy=0.5,
                precision=0.5,
                recall=0.5,
                f1_score=0.5,
                confidence_calibration=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.5,
                avg_return_per_trade=0.0,
                performance_by_regime={regime: {'accuracy': 0.5, 'return': 0.0} for regime in MarketRegime},
                recent_performance_trend=[],
                last_updated=datetime.now()
            )
    
    def record_prediction_outcome(self, prediction: MLPrediction, actual_outcome: Dict[str, Any]):
        """Record the outcome of a prediction for performance tracking"""
        
        try:
            # Determine actual signal based on price movement
            price_change = actual_outcome.get('price_change', 0.0)
            actual_signal = self._price_change_to_signal(price_change)
            
            # Calculate accuracy score
            accuracy_score = self._calculate_prediction_accuracy(prediction, actual_signal, price_change)
            
            # Create prediction outcome record
            outcome = PredictionOutcome(
                prediction_id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                symbol=actual_outcome.get('symbol', 'UNKNOWN'),
                prediction=prediction,
                actual_signal=actual_signal,
                actual_return=price_change,
                prediction_horizon=timedelta(minutes=actual_outcome.get('horizon_minutes', 60)),
                was_accurate=accuracy_score > 0.5,
                accuracy_score=accuracy_score,
                timestamp=datetime.now(),
                market_regime=actual_outcome.get('market_regime', MarketRegime.UNCERTAIN)
            )
            
            # Store outcome
            self.prediction_outcomes.append(outcome)
            
            # Update model metrics
            await self._update_model_metrics(prediction.model_name, outcome)
            
            # Check for alerts
            await self._check_performance_alerts(prediction.model_name)
            
            logger.debug(f"Recorded prediction outcome for {prediction.model_name}: "
                        f"accuracy={accuracy_score:.3f}, signal={actual_signal.value}")
            
        except Exception as e:
            logger.error(f"Error recording prediction outcome: {e}")
    
    def _price_change_to_signal(self, price_change: float) -> MLSignalType:
        """Convert price change to signal type"""
        if price_change > 0.02:  # 2% increase
            return MLSignalType.STRONG_BUY
        elif price_change > 0.005:  # 0.5% increase
            return MLSignalType.BUY
        elif price_change < -0.02:  # 2% decrease
            return MLSignalType.STRONG_SELL
        elif price_change < -0.005:  # 0.5% decrease
            return MLSignalType.SELL
        else:
            return MLSignalType.HOLD
    
    def _calculate_prediction_accuracy(self, prediction: MLPrediction, 
                                     actual_signal: MLSignalType, price_change: float) -> float:
        """Calculate accuracy score for a prediction"""
        
        # Exact match gets full score
        if prediction.signal_type == actual_signal:
            return 1.0
        
        # Directional accuracy gets partial score
        pred_direction = self._signal_to_direction(prediction.signal_type)
        actual_direction = self._signal_to_direction(actual_signal)
        
        if pred_direction == actual_direction and pred_direction != 0:
            return 0.7  # Correct direction
        elif pred_direction == 0 and abs(price_change) < 0.005:
            return 0.8  # Correctly predicted sideways
        elif pred_direction != 0 and actual_direction == 0:
            return 0.3  # Predicted movement but got sideways
        else:
            return 0.0  # Wrong direction
    
    def _signal_to_direction(self, signal: MLSignalType) -> int:
        """Convert signal to direction (-1, 0, 1)"""
        if signal in [MLSignalType.BUY, MLSignalType.STRONG_BUY]:
            return 1
        elif signal in [MLSignalType.SELL, MLSignalType.STRONG_SELL]:
            return -1
        else:
            return 0
    
    async def _update_model_metrics(self, model_name: str, outcome: PredictionOutcome):
        """Update performance metrics for a specific model"""
        
        if model_name not in self.model_metrics:
            return
        
        metrics = self.model_metrics[model_name]
        
        # Update basic counters
        metrics.total_predictions += 1
        if outcome.was_accurate:
            metrics.correct_predictions += 1
        
        # Update accuracy
        metrics.accuracy = metrics.correct_predictions / metrics.total_predictions
        
        # Update recent performance trend
        metrics.recent_performance_trend.append(outcome.accuracy_score)
        if len(metrics.recent_performance_trend) > 50:  # Keep last 50
            metrics.recent_performance_trend = metrics.recent_performance_trend[-50:]
        
        # Update regime-specific performance
        regime = outcome.market_regime
        if regime not in metrics.performance_by_regime:
            metrics.performance_by_regime[regime] = {'accuracy': 0.0, 'return': 0.0, 'count': 0}
        
        regime_metrics = metrics.performance_by_regime[regime]
        regime_metrics['count'] = regime_metrics.get('count', 0) + 1
        regime_metrics['accuracy'] = (
            (regime_metrics['accuracy'] * (regime_metrics['count'] - 1) + outcome.accuracy_score) / 
            regime_metrics['count']
        )
        regime_metrics['return'] = (
            (regime_metrics['return'] * (regime_metrics['count'] - 1) + outcome.actual_return) / 
            regime_metrics['count']
        )
        
        # Update timestamp
        metrics.last_updated = datetime.now()
        
        # Update rolling metrics for real-time monitoring
        self.rolling_metrics[f"{model_name}_accuracy"].append(outcome.accuracy_score)
        self.rolling_metrics[f"{model_name}_returns"].append(outcome.actual_return)
    
    def record_trade_performance(self, trade: TradePerformance):
        """Record completed trade performance"""
        
        try:
            # Store trade performance
            self.trade_performances.append(trade)
            
            # Update daily PnL
            trade_date = trade.exit_time.date().isoformat()
            self.daily_pnl[trade_date] += trade.pnl
            
            # Update model-specific trade metrics
            await self._update_trade_metrics(trade)
            
            # Update rolling metrics
            self.rolling_metrics['trade_returns'].append(trade.return_pct)
            self.rolling_metrics['holding_periods'].append(trade.holding_period.total_seconds() / 3600)  # hours
            
            logger.debug(f"Recorded trade performance: {trade.symbol} PnL={trade.pnl}, Return={trade.return_pct:.2%}")
            
        except Exception as e:
            logger.error(f"Error recording trade performance: {e}")
    
    async def _update_trade_metrics(self, trade: TradePerformance):
        """Update trade-related metrics for models"""
        
        # Update overall trade statistics
        all_trades = list(self.trade_performances)
        if len(all_trades) >= self.config['tracking']['min_trades_for_metrics']:
            
            # Calculate financial metrics
            returns = [float(t.return_pct) for t in all_trades[-100:]]  # Last 100 trades
            
            if returns:
                sharpe_ratio = self._calculate_sharpe_ratio(returns)
                max_drawdown = self._calculate_max_drawdown(returns)
                win_rate = sum(1 for r in returns if r > 0) / len(returns)
                
                # Update all model metrics with trade statistics
                for model_name in self.model_metrics:
                    self.model_metrics[model_name].sharpe_ratio = sharpe_ratio
                    self.model_metrics[model_name].max_drawdown = max_drawdown
                    self.model_metrics[model_name].win_rate = win_rate
                    self.model_metrics[model_name].avg_return_per_trade = np.mean(returns)
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(min(drawdown))
    
    async def _check_performance_alerts(self, model_name: str):
        """Check for performance alerts and trigger if needed"""
        
        if model_name not in self.model_metrics:
            return
        
        metrics = self.model_metrics[model_name]
        thresholds = self.config['alert_thresholds']
        
        alerts = []
        
        # Check accuracy decline
        if len(metrics.recent_performance_trend) >= 20:
            recent_accuracy = np.mean(metrics.recent_performance_trend[-20:])
            historical_accuracy = np.mean(metrics.recent_performance_trend[:-20]) if len(metrics.recent_performance_trend) > 20 else metrics.accuracy
            
            if historical_accuracy - recent_accuracy > thresholds['accuracy_decline']:
                alerts.append(PerformanceAlert(
                    alert_id=f"acc_decline_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.WARNING,
                    metric=PerformanceMetric.PREDICTION_ACCURACY,
                    message=f"Model {model_name} accuracy declined by {(historical_accuracy - recent_accuracy):.1%}",
                    current_value=recent_accuracy,
                    threshold_value=historical_accuracy - thresholds['accuracy_decline'],
                    model_name=model_name,
                    timestamp=datetime.now(),
                    auto_actions=['reduce_model_weight', 'trigger_retraining']
                ))
        
        # Check drawdown limit
        if metrics.max_drawdown > thresholds['drawdown_limit']:
            alerts.append(PerformanceAlert(
                alert_id=f"drawdown_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.CRITICAL,
                metric=PerformanceMetric.MAX_DRAWDOWN,
                message=f"Model {model_name} exceeded drawdown limit: {metrics.max_drawdown:.1%}",
                current_value=metrics.max_drawdown,
                threshold_value=thresholds['drawdown_limit'],
                model_name=model_name,
                timestamp=datetime.now(),
                auto_actions=['pause_model', 'reduce_position_sizes']
            ))
        
        # Check Sharpe ratio
        if metrics.sharpe_ratio < thresholds['sharpe_minimum']:
            alerts.append(PerformanceAlert(
                alert_id=f"sharpe_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=AlertLevel.WARNING,
                metric=PerformanceMetric.SHARPE_RATIO,
                message=f"Model {model_name} Sharpe ratio below minimum: {metrics.sharpe_ratio:.2f}",
                current_value=metrics.sharpe_ratio,
                threshold_value=thresholds['sharpe_minimum'],
                model_name=model_name,
                timestamp=datetime.now(),
                auto_actions=['review_strategy_parameters', 'consider_model_ensemble']
            ))
        
        # Store new alerts
        for alert in alerts:
            self.performance_alerts.append(alert)
            logger.warning(f"Performance Alert: {alert.message}")
            
            # Trigger automatic actions if configured
            await self._execute_auto_actions(alert)
    
    async def _execute_auto_actions(self, alert: PerformanceAlert):
        """Execute automatic actions for performance alerts"""
        
        try:
            for action in alert.auto_actions:
                if action == 'reduce_model_weight':
                    logger.info(f"Auto-action: Reducing weight for model {alert.model_name}")
                    # In a real implementation, this would interface with strategy orchestrator
                    
                elif action == 'trigger_retraining':
                    logger.info(f"Auto-action: Triggering retraining for model {alert.model_name}")
                    # In a real implementation, this would trigger model retraining
                    
                elif action == 'pause_model':
                    logger.warning(f"Auto-action: Pausing model {alert.model_name}")
                    # In a real implementation, this would pause the model
                    
                # Add more auto-actions as needed
                
        except Exception as e:
            logger.error(f"Error executing auto-actions for alert {alert.alert_id}: {e}")
    
    async def _background_monitoring_loop(self):
        """Background loop for continuous performance monitoring"""
        
        while True:
            try:
                await asyncio.sleep(self.config['tracking']['performance_update_frequency'])
                
                # Update real-time metrics
                await self._update_real_time_metrics()
                
                # Check for alerts
                for model_name in self.model_metrics:
                    await self._check_performance_alerts(model_name)
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
            except Exception as e:
                logger.error(f"Error in background monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _update_real_time_metrics(self):
        """Update real-time performance metrics"""
        
        try:
            # Calculate current day performance
            today = datetime.now().date().isoformat()
            today_pnl = self.daily_pnl.get(today, Decimal('0'))
            
            # Update rolling metrics
            if today_pnl != 0:
                self.rolling_metrics['daily_pnl'].append(float(today_pnl))
            
            # Calculate recent performance trends
            for model_name in self.model_metrics:
                model_key = f"{model_name}_accuracy"
                if model_key in self.rolling_metrics and len(self.rolling_metrics[model_key]) > 0:
                    recent_accuracy = np.mean(list(self.rolling_metrics[model_key])[-20:])
                    self.rolling_metrics[f"{model_name}_recent_accuracy"] = deque([recent_accuracy], maxlen=1)
            
        except Exception as e:
            logger.error(f"Error updating real-time metrics: {e}")
    
    def _cleanup_old_alerts(self, max_age_hours: int = 24):
        """Clean up old alerts"""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        self.performance_alerts = [
            alert for alert in self.performance_alerts 
            if alert.timestamp > cutoff_time
        ]
    
    async def _daily_report_generator(self):
        """Generate daily performance reports"""
        
        while True:
            try:
                # Wait for next report time (daily)
                await asyncio.sleep(self.config['tracking']['report_generation_frequency'])
                
                # Generate daily report
                report = await self.generate_performance_report()
                
                # Log report summary
                logger.info(f"Daily Performance Report Generated: "
                           f"Trades={report.total_trades}, "
                           f"Return={report.return_pct:.2%}, "
                           f"Sharpe={report.sharpe_ratio:.2f}")
                
                # In a real implementation, this would save/send the report
                
            except Exception as e:
                logger.error(f"Error generating daily report: {e}")
    
    async def generate_performance_report(self, 
                                        start_date: Optional[datetime] = None,
                                        end_date: Optional[datetime] = None) -> PerformanceReport:
        """Generate comprehensive performance report"""
        
        # Default to last 24 hours if no dates specified
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=1)
        
        # Filter data by date range
        relevant_trades = [
            trade for trade in self.trade_performances
            if start_date <= trade.exit_time <= end_date
        ]
        
        relevant_predictions = [
            outcome for outcome in self.prediction_outcomes
            if start_date <= outcome.timestamp <= end_date
        ]
        
        # Calculate overall metrics
        total_trades = len(relevant_trades)
        total_pnl = sum(trade.pnl for trade in relevant_trades)
        
        if relevant_trades:
            returns = [trade.return_pct for trade in relevant_trades]
            overall_return = sum(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
        else:
            overall_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            win_rate = 0.0
        
        # Calculate model-specific performance
        model_performance = {}
        for model_name, metrics in self.model_metrics.items():
            # Filter predictions for this model in the time period
            model_predictions = [
                outcome for outcome in relevant_predictions
                if outcome.prediction.model_name == model_name
            ]
            
            if model_predictions:
                model_accuracy = np.mean([p.accuracy_score for p in model_predictions])
                model_return = np.mean([p.actual_return for p in model_predictions])
            else:
                model_accuracy = metrics.accuracy
                model_return = 0.0
            
            model_performance[model_name] = ModelPerformanceMetrics(
                model_name=model_name,
                total_predictions=len(model_predictions),
                correct_predictions=sum(1 for p in model_predictions if p.was_accurate),
                accuracy=model_accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                f1_score=metrics.f1_score,
                confidence_calibration=metrics.confidence_calibration,
                sharpe_ratio=metrics.sharpe_ratio,
                max_drawdown=metrics.max_drawdown,
                win_rate=metrics.win_rate,
                avg_return_per_trade=model_return,
                performance_by_regime=metrics.performance_by_regime,
                recent_performance_trend=metrics.recent_performance_trend[-20:],  # Last 20
                last_updated=metrics.last_updated
            )
        
        # Calculate execution performance
        execution_performance = {}
        if relevant_trades:
            avg_slippage = np.mean([getattr(trade, 'execution_quality', 1.0) for trade in relevant_trades])
            execution_performance = {
                'average_execution_quality': avg_slippage,
                'total_execution_cost': sum(getattr(trade, 'execution_cost', 0) for trade in relevant_trades)
            }
        
        # Calculate regime breakdown
        regime_breakdown = {}
        for regime in MarketRegime:
            regime_trades = [trade for trade in relevant_trades if trade.market_regime == regime]
            if regime_trades:
                regime_returns = [trade.return_pct for trade in regime_trades]
                regime_breakdown[regime] = {
                    'trades': len(regime_trades),
                    'return': sum(regime_returns),
                    'win_rate': sum(1 for r in regime_returns if r > 0) / len(regime_returns)
                }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            model_performance, execution_performance, regime_breakdown
        )
        
        # Create report
        report = PerformanceReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            period_start=start_date,
            period_end=end_date,
            total_trades=total_trades,
            total_pnl=total_pnl,
            return_pct=overall_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            model_performance=model_performance,
            execution_performance=execution_performance,
            regime_breakdown=regime_breakdown,
            recommendations=recommendations,
            generated_at=datetime.now()
        )
        
        return report
    
    def _generate_recommendations(self, model_performance: Dict[str, ModelPerformanceMetrics],
                                execution_performance: Dict[str, float],
                                regime_breakdown: Dict[MarketRegime, Dict[str, float]]) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        # Model performance recommendations
        best_model = max(model_performance.items(), key=lambda x: x[1].accuracy) if model_performance else None
        worst_model = min(model_performance.items(), key=lambda x: x[1].accuracy) if model_performance else None
        
        if best_model and worst_model and best_model[1].accuracy - worst_model[1].accuracy > 0.1:
            recommendations.append(
                f"Consider increasing weight for {best_model[0]} (accuracy: {best_model[1].accuracy:.1%}) "
                f"and reducing weight for {worst_model[0]} (accuracy: {worst_model[1].accuracy:.1%})"
            )
        
        # Execution performance recommendations
        if execution_performance.get('average_execution_quality', 1.0) < 0.9:
            recommendations.append("Consider optimizing execution strategies to reduce slippage")
        
        # Regime-specific recommendations
        for regime, metrics in regime_breakdown.items():
            if metrics.get('win_rate', 0) < 0.4:
                recommendations.append(f"Poor performance in {regime.value} regime - consider regime-specific adjustments")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Performance appears stable - continue monitoring for optimization opportunities")
        
        return recommendations
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time performance metrics"""
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_accuracies': {},
            'recent_performance': {},
            'active_alerts': len(self.performance_alerts),
            'daily_pnl': float(self.daily_pnl.get(datetime.now().date().isoformat(), 0))
        }
        
        # Model accuracies
        for model_name in self.model_metrics:
            metrics['model_accuracies'][model_name] = self.model_metrics[model_name].accuracy
        
        # Recent performance trends
        for key, values in self.rolling_metrics.items():
            if values:
                metrics['recent_performance'][key] = {
                    'current': values[-1] if values else 0,
                    'average_10': np.mean(list(values)[-10:]) if len(values) >= 10 else (values[-1] if values else 0),
                    'trend': 'improving' if len(values) >= 2 and values[-1] > values[-2] else 'declining'
                }
        
        return metrics
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active performance alerts"""
        
        return [
            {
                'alert_id': alert.alert_id,
                'level': alert.level.value,
                'metric': alert.metric.value,
                'message': alert.message,
                'model_name': alert.model_name,
                'timestamp': alert.timestamp.isoformat(),
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'auto_actions': alert.auto_actions
            }
            for alert in self.performance_alerts
        ]

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'MLPerformanceMonitor',
    'PerformanceMetric',
    'AlertLevel',
    'PredictionOutcome',
    'TradePerformance',
    'ModelPerformanceMetrics',
    'PerformanceAlert',
    'PerformanceReport'
]