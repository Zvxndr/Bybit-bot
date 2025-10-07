"""
Auto Scaler for Dynamic Resource Scaling.
Monitors metrics and automatically scales containers/pods based on load, performance, and custom metrics.
"""

import asyncio
import json
import time
import math
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
import warnings
warnings.filterwarnings('ignore')

from ..core.configuration_manager import ConfigurationManager
from ..core.trading_logger import TradingLogger

class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"

class MetricType(Enum):
    """Metric types for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"

class ScalingPolicy(Enum):
    """Scaling policies."""
    TARGET_TRACKING = "target_tracking"
    STEP_SCALING = "step_scaling"
    PREDICTIVE = "predictive"
    THRESHOLD = "threshold"

@dataclass
class MetricData:
    """Metric data point."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    source: str
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class ScalingRule:
    """Scaling rule configuration."""
    name: str
    metric_type: MetricType
    threshold_up: float
    threshold_down: float
    scale_up_by: int = 1
    scale_down_by: int = 1
    cooldown_up: int = 300  # seconds
    cooldown_down: int = 600  # seconds
    min_replicas: int = 1
    max_replicas: int = 10
    evaluation_periods: int = 2
    datapoints_to_alarm: int = 2
    enabled: bool = True

@dataclass
class ScalingTarget:
    """Scaling target configuration."""
    name: str
    target_type: str  # 'deployment', 'container', 'service'
    namespace: Optional[str] = None
    current_replicas: int = 1
    desired_replicas: int = 1
    min_replicas: int = 1
    max_replicas: int = 10
    scaling_rules: List[ScalingRule] = field(default_factory=list)
    last_scale_time: Optional[datetime] = None
    last_scale_direction: Optional[ScalingDirection] = None

@dataclass
class ScalingEvent:
    """Scaling event record."""
    target_name: str
    timestamp: datetime
    direction: ScalingDirection
    from_replicas: int
    to_replicas: int
    trigger_metric: str
    trigger_value: float
    reason: str
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class PredictiveModel:
    """Predictive scaling model."""
    target_name: str
    model_type: str  # 'linear', 'polynomial', 'seasonal'
    training_data: List[Tuple[datetime, float]] = field(default_factory=list)
    model_params: Dict[str, Any] = field(default_factory=dict)
    accuracy: float = 0.0
    last_trained: Optional[datetime] = None
    prediction_horizon: int = 1800  # 30 minutes

class AutoScaler:
    """Dynamic auto-scaling system for containers and pods."""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = TradingLogger()
        
        # Auto-scaling configuration
        self.scaling_config = {
            'enabled': True,
            'evaluation_interval': 30,  # seconds
            'metric_collection_interval': 10,  # seconds
            'default_cooldown_up': 300,  # 5 minutes
            'default_cooldown_down': 600,  # 10 minutes
            'safety_margin': 0.1,  # 10% safety margin
            'aggressive_scaling': False,
            'predictive_scaling': True,
            'max_scale_up_percent': 100,  # Double at most
            'max_scale_down_percent': 50,  # Half at most
            'metric_aggregation_window': 300,  # 5 minutes
            'spike_detection_threshold': 3.0,  # 3x standard deviation
        }
        
        # Scaling targets and rules
        self.scaling_targets: Dict[str, ScalingTarget] = {}
        self.scaling_events: List[ScalingEvent] = []
        self.metric_history: Dict[str, List[MetricData]] = {}
        self.predictive_models: Dict[str, PredictiveModel] = {}
        
        # Auto-scaling state
        self.scaling_active = False
        self.scaling_task = None
        self.metric_collection_task = None
        
        # Callbacks for scaling actions
        self.scale_up_callback: Optional[Callable] = None
        self.scale_down_callback: Optional[Callable] = None
        
        # Default scaling rules for trading services
        self._setup_default_scaling_rules()
        
        self.logger.info("AutoScaler initialized")
    
    def _setup_default_scaling_rules(self):
        """Setup default scaling rules for trading services."""
        try:
            # Trading Engine scaling rules
            trading_engine_rules = [
                ScalingRule(
                    name="cpu-scaling",
                    metric_type=MetricType.CPU_UTILIZATION,
                    threshold_up=70.0,
                    threshold_down=30.0,
                    scale_up_by=2,
                    scale_down_by=1,
                    cooldown_up=180,
                    cooldown_down=300,
                    min_replicas=2,
                    max_replicas=20
                ),
                ScalingRule(
                    name="request-rate-scaling",
                    metric_type=MetricType.REQUEST_RATE,
                    threshold_up=1000.0,  # requests per second
                    threshold_down=200.0,
                    scale_up_by=3,
                    scale_down_by=1,
                    cooldown_up=120,
                    cooldown_down=600,
                    min_replicas=2,
                    max_replicas=25
                ),
                ScalingRule(
                    name="response-time-scaling",
                    metric_type=MetricType.RESPONSE_TIME,
                    threshold_up=500.0,  # milliseconds
                    threshold_down=100.0,
                    scale_up_by=2,
                    scale_down_by=1,
                    cooldown_up=60,
                    cooldown_down=300,
                    min_replicas=2,
                    max_replicas=15
                )
            ]
            
            self.scaling_targets['trading-engine'] = ScalingTarget(
                name='trading-engine',
                target_type='deployment',
                current_replicas=3,
                min_replicas=2,
                max_replicas=25,
                scaling_rules=trading_engine_rules
            )
            
            # HFT Module scaling rules (more aggressive)
            hft_rules = [
                ScalingRule(
                    name="latency-scaling",
                    metric_type=MetricType.RESPONSE_TIME,
                    threshold_up=10.0,  # 10ms
                    threshold_down=2.0,
                    scale_up_by=5,
                    scale_down_by=1,
                    cooldown_up=30,
                    cooldown_down=180,
                    min_replicas=5,
                    max_replicas=50
                ),
                ScalingRule(
                    name="throughput-scaling",
                    metric_type=MetricType.THROUGHPUT,
                    threshold_up=10000.0,  # operations per second
                    threshold_down=2000.0,
                    scale_up_by=3,
                    scale_down_by=1,
                    cooldown_up=60,
                    cooldown_down=300,
                    min_replicas=5,
                    max_replicas=40
                )
            ]
            
            self.scaling_targets['hft-module'] = ScalingTarget(
                name='hft-module',
                target_type='deployment',
                current_replicas=5,
                min_replicas=5,
                max_replicas=50,
                scaling_rules=hft_rules
            )
            
            # Market Data scaling rules
            market_data_rules = [
                ScalingRule(
                    name="cpu-scaling",
                    metric_type=MetricType.CPU_UTILIZATION,
                    threshold_up=80.0,
                    threshold_down=40.0,
                    scale_up_by=2,
                    scale_down_by=1,
                    cooldown_up=120,
                    cooldown_down=300,
                    min_replicas=2,
                    max_replicas=15
                ),
                ScalingRule(
                    name="queue-scaling",
                    metric_type=MetricType.QUEUE_LENGTH,
                    threshold_up=1000.0,
                    threshold_down=100.0,
                    scale_up_by=2,
                    scale_down_by=1,
                    cooldown_up=90,
                    cooldown_down=300,
                    min_replicas=2,
                    max_replicas=20
                )
            ]
            
            self.scaling_targets['market-data'] = ScalingTarget(
                name='market-data',
                target_type='deployment',
                current_replicas=2,
                min_replicas=2,
                max_replicas=20,
                scaling_rules=market_data_rules
            )
            
            # Analytics Engine scaling rules
            analytics_rules = [
                ScalingRule(
                    name="memory-scaling",
                    metric_type=MetricType.MEMORY_UTILIZATION,
                    threshold_up=75.0,
                    threshold_down=35.0,
                    scale_up_by=1,
                    scale_down_by=1,
                    cooldown_up=300,
                    cooldown_down=600,
                    min_replicas=1,
                    max_replicas=10
                ),
                ScalingRule(
                    name="cpu-scaling",
                    metric_type=MetricType.CPU_UTILIZATION,
                    threshold_up=70.0,
                    threshold_down=30.0,
                    scale_up_by=1,
                    scale_down_by=1,
                    cooldown_up=180,
                    cooldown_down=400,
                    min_replicas=1,
                    max_replicas=8
                )
            ]
            
            self.scaling_targets['analytics'] = ScalingTarget(
                name='analytics',
                target_type='deployment',
                current_replicas=2,
                min_replicas=1,
                max_replicas=10,
                scaling_rules=analytics_rules
            )
            
            # ML Engine scaling rules
            ml_rules = [
                ScalingRule(
                    name="queue-scaling",
                    metric_type=MetricType.QUEUE_LENGTH,
                    threshold_up=50.0,
                    threshold_down=5.0,
                    scale_up_by=1,
                    scale_down_by=1,
                    cooldown_up=600,  # ML models take time to start
                    cooldown_down=900,
                    min_replicas=1,
                    max_replicas=5
                )
            ]
            
            self.scaling_targets['ml-engine'] = ScalingTarget(
                name='ml-engine',
                target_type='deployment',
                current_replicas=1,
                min_replicas=1,
                max_replicas=5,
                scaling_rules=ml_rules
            )
            
        except Exception as e:
            self.logger.error(f"Failed to setup default scaling rules: {e}")
    
    def set_scaling_callbacks(self, scale_up_callback: Callable, scale_down_callback: Callable):
        """Set callbacks for scaling actions."""
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback
    
    async def start_auto_scaling(self):
        """Start auto-scaling system."""
        try:
            if self.scaling_active:
                self.logger.warning("Auto-scaling is already active")
                return
            
            self.scaling_active = True
            
            # Start metric collection task
            self.metric_collection_task = asyncio.create_task(self._collect_metrics_loop())
            
            # Start scaling evaluation task
            self.scaling_task = asyncio.create_task(self._scaling_loop())
            
            self.logger.info("Auto-scaling system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start auto-scaling: {e}")
            self.scaling_active = False
    
    async def stop_auto_scaling(self):
        """Stop auto-scaling system."""
        try:
            self.scaling_active = False
            
            if self.scaling_task:
                self.scaling_task.cancel()
                try:
                    await self.scaling_task
                except asyncio.CancelledError:
                    pass
            
            if self.metric_collection_task:
                self.metric_collection_task.cancel()
                try:
                    await self.metric_collection_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Auto-scaling system stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop auto-scaling: {e}")
    
    async def _collect_metrics_loop(self):
        """Continuously collect metrics for scaling decisions."""
        try:
            while self.scaling_active:
                # Collect metrics for all targets
                for target_name in self.scaling_targets:
                    await self._collect_target_metrics(target_name)
                
                # Clean old metrics
                await self._cleanup_old_metrics()
                
                # Update predictive models
                if self.scaling_config['predictive_scaling']:
                    await self._update_predictive_models()
                
                await asyncio.sleep(self.scaling_config['metric_collection_interval'])
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Metric collection loop error: {e}")
    
    async def _scaling_loop(self):
        """Main scaling evaluation loop."""
        try:
            while self.scaling_active:
                # Evaluate each scaling target
                for target_name, target in self.scaling_targets.items():
                    if not target.scaling_rules:
                        continue
                    
                    # Check if target is in cooldown
                    if self._is_in_cooldown(target):
                        continue
                    
                    # Evaluate scaling rules
                    scaling_decision = await self._evaluate_scaling_rules(target)
                    
                    if scaling_decision != ScalingDirection.NONE:
                        await self._execute_scaling_decision(target, scaling_decision)
                
                await asyncio.sleep(self.scaling_config['evaluation_interval'])
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Scaling loop error: {e}")
    
    async def _collect_target_metrics(self, target_name: str):
        """Collect metrics for a specific target."""
        try:
            # This would integrate with monitoring systems (Prometheus, etc.)
            # For now, simulate metric collection
            
            current_time = datetime.now()
            
            # Simulate CPU utilization
            import random
            base_cpu = 50.0
            if target_name == 'hft-module':
                base_cpu = 80.0  # HFT runs hot
            elif target_name == 'ml-engine':
                base_cpu = 30.0  # ML is bursty
            
            cpu_metric = MetricData(
                metric_type=MetricType.CPU_UTILIZATION,
                value=base_cpu + random.uniform(-20, 30),
                timestamp=current_time,
                source=target_name,
                tags={'target': target_name}
            )
            
            self._store_metric(target_name, cpu_metric)
            
            # Simulate memory utilization
            memory_metric = MetricData(
                metric_type=MetricType.MEMORY_UTILIZATION,
                value=random.uniform(40, 85),
                timestamp=current_time,
                source=target_name,
                tags={'target': target_name}
            )
            
            self._store_metric(target_name, memory_metric)
            
            # Simulate request rate for appropriate services
            if target_name in ['trading-engine', 'market-data']:
                request_rate = random.uniform(100, 2000)
                if target_name == 'trading-engine':
                    request_rate = random.uniform(500, 3000)
                
                request_metric = MetricData(
                    metric_type=MetricType.REQUEST_RATE,
                    value=request_rate,
                    timestamp=current_time,
                    source=target_name,
                    tags={'target': target_name}
                )
                
                self._store_metric(target_name, request_metric)
            
            # Simulate response time
            base_response_time = 100.0  # ms
            if target_name == 'hft-module':
                base_response_time = 5.0  # Ultra-low latency
            elif target_name == 'ml-engine':
                base_response_time = 2000.0  # ML inference takes time
            
            response_time_metric = MetricData(
                metric_type=MetricType.RESPONSE_TIME,
                value=base_response_time * random.uniform(0.5, 3.0),
                timestamp=current_time,
                source=target_name,
                tags={'target': target_name}
            )
            
            self._store_metric(target_name, response_time_metric)
            
            # Simulate queue length
            if target_name in ['market-data', 'ml-engine']:
                queue_length = random.uniform(0, 200)
                if target_name == 'ml-engine':
                    queue_length = random.uniform(0, 100)
                
                queue_metric = MetricData(
                    metric_type=MetricType.QUEUE_LENGTH,
                    value=queue_length,
                    timestamp=current_time,
                    source=target_name,
                    tags={'target': target_name}
                )
                
                self._store_metric(target_name, queue_metric)
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics for {target_name}: {e}")
    
    def _store_metric(self, target_name: str, metric: MetricData):
        """Store metric in history."""
        metric_key = f"{target_name}:{metric.metric_type.value}"
        
        if metric_key not in self.metric_history:
            self.metric_history[metric_key] = []
        
        self.metric_history[metric_key].append(metric)
        
        # Keep only recent metrics
        max_history = self.scaling_config['metric_aggregation_window'] * 2 // self.scaling_config['metric_collection_interval']
        if len(self.metric_history[metric_key]) > max_history:
            self.metric_history[metric_key] = self.metric_history[metric_key][-max_history:]
    
    async def _evaluate_scaling_rules(self, target: ScalingTarget) -> ScalingDirection:
        """Evaluate scaling rules for a target."""
        try:
            scaling_votes = []
            
            for rule in target.scaling_rules:
                if not rule.enabled:
                    continue
                
                # Get recent metrics for this rule
                metric_key = f"{target.name}:{rule.metric_type.value}"
                recent_metrics = self._get_recent_metrics(metric_key, rule.evaluation_periods * 30)
                
                if len(recent_metrics) < rule.datapoints_to_alarm:
                    continue
                
                # Calculate average metric value
                avg_value = statistics.mean([m.value for m in recent_metrics])
                
                # Check for scaling up
                if avg_value > rule.threshold_up:
                    scaling_votes.append((ScalingDirection.UP, rule.scale_up_by, rule.name, avg_value))
                # Check for scaling down
                elif avg_value < rule.threshold_down:
                    scaling_votes.append((ScalingDirection.DOWN, rule.scale_down_by, rule.name, avg_value))
            
            # Determine final scaling decision
            if not scaling_votes:
                return ScalingDirection.NONE
            
            # Count votes for each direction
            up_votes = [v for v in scaling_votes if v[0] == ScalingDirection.UP]
            down_votes = [v for v in scaling_votes if v[0] == ScalingDirection.DOWN]
            
            # Prioritize scale up if there are conflicting votes
            if up_votes:
                # Calculate scale up amount (max of all up votes)
                max_scale_up = max(v[1] for v in up_votes)
                target.desired_replicas = min(
                    target.current_replicas + max_scale_up,
                    target.max_replicas
                )
                return ScalingDirection.UP
            elif down_votes:
                # Calculate scale down amount (max of all down votes)
                max_scale_down = max(v[1] for v in down_votes)
                target.desired_replicas = max(
                    target.current_replicas - max_scale_down,
                    target.min_replicas
                )
                return ScalingDirection.DOWN
            
            return ScalingDirection.NONE
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate scaling rules for {target.name}: {e}")
            return ScalingDirection.NONE
    
    async def _execute_scaling_decision(self, target: ScalingTarget, direction: ScalingDirection):
        """Execute scaling decision."""
        try:
            if direction == ScalingDirection.NONE:
                return
            
            old_replicas = target.current_replicas
            new_replicas = target.desired_replicas
            
            if old_replicas == new_replicas:
                return
            
            # Apply safety limits
            max_change_up = max(1, int(old_replicas * self.scaling_config['max_scale_up_percent'] / 100))
            max_change_down = max(1, int(old_replicas * self.scaling_config['max_scale_down_percent'] / 100))
            
            if direction == ScalingDirection.UP:
                new_replicas = min(new_replicas, old_replicas + max_change_up)
            else:
                new_replicas = max(new_replicas, old_replicas - max_change_down)
            
            # Execute scaling action
            success = False
            error_message = None
            
            try:
                if direction == ScalingDirection.UP and self.scale_up_callback:
                    success = await self.scale_up_callback(target.name, new_replicas)
                elif direction == ScalingDirection.DOWN and self.scale_down_callback:
                    success = await self.scale_down_callback(target.name, new_replicas)
                else:
                    # Fallback to generic scaling
                    success = True  # Would call generic scaling function
                
                if success:
                    target.current_replicas = new_replicas
                    target.last_scale_time = datetime.now()
                    target.last_scale_direction = direction
                    
            except Exception as e:
                error_message = str(e)
                success = False
            
            # Record scaling event
            scaling_event = ScalingEvent(
                target_name=target.name,
                timestamp=datetime.now(),
                direction=direction,
                from_replicas=old_replicas,
                to_replicas=new_replicas,
                trigger_metric="multiple",  # Would specify exact metric
                trigger_value=0.0,  # Would specify exact value
                reason=f"Auto-scaling {direction.value}",
                success=success,
                error_message=error_message
            )
            
            self.scaling_events.append(scaling_event)
            
            # Keep only recent events
            if len(self.scaling_events) > 1000:
                self.scaling_events = self.scaling_events[-500:]
            
            if success:
                self.logger.info(
                    f"Scaled {target.name} {direction.value}: {old_replicas} -> {new_replicas} replicas"
                )
            else:
                self.logger.error(
                    f"Failed to scale {target.name} {direction.value}: {error_message}"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to execute scaling decision for {target.name}: {e}")
    
    def _is_in_cooldown(self, target: ScalingTarget) -> bool:
        """Check if target is in cooldown period."""
        try:
            if not target.last_scale_time:
                return False
            
            now = datetime.now()
            time_since_last_scale = (now - target.last_scale_time).total_seconds()
            
            # Find the minimum cooldown from active rules
            min_cooldown = float('inf')
            for rule in target.scaling_rules:
                if not rule.enabled:
                    continue
                
                if target.last_scale_direction == ScalingDirection.UP:
                    min_cooldown = min(min_cooldown, rule.cooldown_up)
                elif target.last_scale_direction == ScalingDirection.DOWN:
                    min_cooldown = min(min_cooldown, rule.cooldown_down)
            
            if min_cooldown == float('inf'):
                min_cooldown = self.scaling_config['default_cooldown_up']
            
            return time_since_last_scale < min_cooldown
            
        except Exception as e:
            self.logger.error(f"Failed to check cooldown for {target.name}: {e}")
            return True  # Conservative: assume in cooldown on error
    
    def _get_recent_metrics(self, metric_key: str, seconds: int) -> List[MetricData]:
        """Get recent metrics within specified time window."""
        try:
            if metric_key not in self.metric_history:
                return []
            
            cutoff_time = datetime.now() - timedelta(seconds=seconds)
            return [
                metric for metric in self.metric_history[metric_key]
                if metric.timestamp >= cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get recent metrics for {metric_key}: {e}")
            return []
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory growth."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=2)
            
            for metric_key in list(self.metric_history.keys()):
                self.metric_history[metric_key] = [
                    metric for metric in self.metric_history[metric_key]
                    if metric.timestamp >= cutoff_time
                ]
                
                # Remove empty entries
                if not self.metric_history[metric_key]:
                    del self.metric_history[metric_key]
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old metrics: {e}")
    
    async def _update_predictive_models(self):
        """Update predictive scaling models."""
        try:
            for target_name, target in self.scaling_targets.items():
                # Get CPU utilization history for prediction
                metric_key = f"{target_name}:{MetricType.CPU_UTILIZATION.value}"
                recent_metrics = self._get_recent_metrics(metric_key, 3600)  # 1 hour
                
                if len(recent_metrics) < 10:
                    continue
                
                # Update or create predictive model
                if target_name not in self.predictive_models:
                    self.predictive_models[target_name] = PredictiveModel(
                        target_name=target_name,
                        model_type='linear'
                    )
                
                model = self.predictive_models[target_name]
                
                # Add data points
                for metric in recent_metrics:
                    model.training_data.append((metric.timestamp, metric.value))
                
                # Keep only recent training data
                if len(model.training_data) > 200:
                    model.training_data = model.training_data[-100:]
                
                # Simple linear trend prediction
                if len(model.training_data) >= 20:
                    await self._train_simple_model(model)
                
        except Exception as e:
            self.logger.error(f"Failed to update predictive models: {e}")
    
    async def _train_simple_model(self, model: PredictiveModel):
        """Train a simple predictive model."""
        try:
            if len(model.training_data) < 20:
                return
            
            # Calculate simple linear trend
            values = [point[1] for point in model.training_data]
            n = len(values)
            
            # Calculate moving average and trend
            window_size = min(10, n // 2)
            recent_avg = statistics.mean(values[-window_size:])
            older_avg = statistics.mean(values[-2*window_size:-window_size])
            
            trend = (recent_avg - older_avg) / window_size
            
            model.model_params = {
                'recent_avg': recent_avg,
                'trend': trend,
                'volatility': statistics.stdev(values[-window_size:]) if window_size > 1 else 0
            }
            
            model.last_trained = datetime.now()
            
            # Make prediction for next 30 minutes
            prediction_value = recent_avg + (trend * (model.prediction_horizon / 60))
            
            # If prediction indicates high load, preemptively scale
            if prediction_value > 80.0 and model.target_name in self.scaling_targets:
                target = self.scaling_targets[model.target_name]
                if not self._is_in_cooldown(target):
                    # Predictive scaling up
                    target.desired_replicas = min(
                        target.current_replicas + 1,
                        target.max_replicas
                    )
                    
                    self.logger.info(
                        f"Predictive scaling triggered for {model.target_name}: "
                        f"predicted load {prediction_value:.1f}%"
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to train model for {model.target_name}: {e}")
    
    def add_scaling_target(self, target: ScalingTarget):
        """Add a new scaling target."""
        self.scaling_targets[target.name] = target
        self.logger.info(f"Added scaling target: {target.name}")
    
    def remove_scaling_target(self, target_name: str):
        """Remove a scaling target."""
        if target_name in self.scaling_targets:
            del self.scaling_targets[target_name]
            self.logger.info(f"Removed scaling target: {target_name}")
    
    def add_scaling_rule(self, target_name: str, rule: ScalingRule):
        """Add a scaling rule to a target."""
        if target_name in self.scaling_targets:
            self.scaling_targets[target_name].scaling_rules.append(rule)
            self.logger.info(f"Added scaling rule {rule.name} to {target_name}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        try:
            target_status = {}
            
            for name, target in self.scaling_targets.items():
                in_cooldown = self._is_in_cooldown(target)
                
                # Get recent metrics
                cpu_key = f"{name}:{MetricType.CPU_UTILIZATION.value}"
                recent_cpu = self._get_recent_metrics(cpu_key, 60)
                avg_cpu = statistics.mean([m.value for m in recent_cpu]) if recent_cpu else 0
                
                memory_key = f"{name}:{MetricType.MEMORY_UTILIZATION.value}"
                recent_memory = self._get_recent_metrics(memory_key, 60)
                avg_memory = statistics.mean([m.value for m in recent_memory]) if recent_memory else 0
                
                target_status[name] = {
                    'current_replicas': target.current_replicas,
                    'desired_replicas': target.desired_replicas,
                    'min_replicas': target.min_replicas,
                    'max_replicas': target.max_replicas,
                    'last_scale_time': target.last_scale_time.isoformat() if target.last_scale_time else None,
                    'last_scale_direction': target.last_scale_direction.value if target.last_scale_direction else None,
                    'in_cooldown': in_cooldown,
                    'active_rules': len([r for r in target.scaling_rules if r.enabled]),
                    'metrics': {
                        'avg_cpu': round(avg_cpu, 2),
                        'avg_memory': round(avg_memory, 2)
                    }
                }
            
            return {
                'scaling_active': self.scaling_active,
                'targets': target_status,
                'recent_events': len([e for e in self.scaling_events 
                                    if e.timestamp > datetime.now() - timedelta(hours=1)]),
                'total_events': len(self.scaling_events),
                'predictive_models': len(self.predictive_models)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get scaling status: {e}")
            return {'error': 'Unable to get scaling status'}
    
    def get_scaling_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent scaling events."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_events = [
                {
                    'target_name': event.target_name,
                    'timestamp': event.timestamp.isoformat(),
                    'direction': event.direction.value,
                    'from_replicas': event.from_replicas,
                    'to_replicas': event.to_replicas,
                    'trigger_metric': event.trigger_metric,
                    'trigger_value': event.trigger_value,
                    'reason': event.reason,
                    'success': event.success,
                    'error_message': event.error_message
                }
                for event in self.scaling_events
                if event.timestamp >= cutoff_time
            ]
            
            return sorted(recent_events, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to get scaling events: {e}")
            return []
    
    def get_auto_scaler_summary(self) -> Dict[str, Any]:
        """Get auto-scaler summary."""
        try:
            total_replicas = sum(target.current_replicas for target in self.scaling_targets.values())
            total_max_replicas = sum(target.max_replicas for target in self.scaling_targets.values())
            
            recent_events = len([
                e for e in self.scaling_events
                if e.timestamp > datetime.now() - timedelta(hours=1)
            ])
            
            return {
                'enabled': self.scaling_config['enabled'],
                'active': self.scaling_active,
                'targets': len(self.scaling_targets),
                'total_replicas': total_replicas,
                'max_capacity': total_max_replicas,
                'capacity_utilization': f"{(total_replicas / total_max_replicas * 100):.1f}%" if total_max_replicas > 0 else "0%",
                'recent_scaling_events': recent_events,
                'predictive_scaling': self.scaling_config['predictive_scaling'],
                'evaluation_interval': self.scaling_config['evaluation_interval'],
                'metric_types_monitored': len(set(
                    rule.metric_type.value
                    for target in self.scaling_targets.values()
                    for rule in target.scaling_rules
                ))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate auto-scaler summary: {e}")
            return {'error': 'Unable to generate summary'}