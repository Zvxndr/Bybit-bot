"""
ML Risk Management Configuration

This module provides configuration management for the ML-enhanced risk management system.
It extends the existing configuration to include ML-specific risk parameters, circuit breakers,
and emergency stop conditions.

The configuration is designed to be:
- Environment-aware (development, staging, production)
- Mode-aware (conservative, aggressive, paper trading)
- ML-specific (confidence thresholds, model uncertainty limits)
- Safety-first (comprehensive circuit breakers and emergency stops)
"""

from typing import Dict, Any, Optional, List
from decimal import Decimal
from datetime import timedelta
from enum import Enum
from dataclasses import dataclass, field
import yaml
import json
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION DATA STRUCTURES
# ============================================================================

@dataclass
class MLRiskThresholds:
    """ML-specific risk thresholds"""
    min_confidence_threshold: float = 0.6          # Minimum ML confidence for execution
    max_uncertainty_threshold: float = 0.4         # Maximum model uncertainty allowed
    min_ensemble_agreement: float = 0.7            # Minimum agreement between models
    confidence_scaling_factor: float = 2.0         # How much confidence affects position size
    prediction_stability_threshold: float = 0.6    # Minimum prediction stability required
    feature_importance_threshold: float = 0.3      # Minimum feature importance score
    
    # Dynamic thresholds based on market conditions
    high_volatility_confidence_boost: float = 0.1  # Require higher confidence in volatile markets
    low_volume_confidence_boost: float = 0.05      # Require higher confidence in low volume
    correlation_breakdown_threshold: float = 0.3   # Correlation threshold for risk adjustment

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    daily_loss_limit: float = 0.03                 # 3% daily loss triggers circuit breaker
    volatility_spike_multiplier: float = 3.0       # 3x normal volatility triggers breaker
    model_performance_threshold: float = 0.4       # < 40% accuracy triggers breaker
    execution_failure_rate_threshold: float = 0.2  # > 20% execution failures
    data_quality_threshold: float = 0.8            # > 80% data quality issues
    correlation_breakdown_threshold: float = 0.3   # Correlation drops below 0.3
    
    # Recovery settings
    auto_recovery_enabled: bool = True
    recovery_check_interval_minutes: int = 5
    minimum_recovery_wait_minutes: int = 15
    
    # Escalation settings
    escalate_after_minutes: int = 60               # Escalate to emergency stop after 1 hour
    max_consecutive_triggers: int = 3              # Max consecutive triggers before escalation

@dataclass
class EmergencyStopConfig:
    """Emergency stop configuration"""
    # Automatic triggers
    max_consecutive_losses: int = 5
    max_portfolio_drawdown: float = 0.10           # 10% portfolio drawdown
    max_position_drawdown: float = 0.20            # 20% single position drawdown
    model_complete_failure_threshold: float = 0.1  # < 10% model accuracy
    
    # Manual override settings
    require_override_code: bool = True
    override_code_length: int = 8
    auto_recovery_enabled: bool = False            # Emergency stops require manual recovery
    
    # Alert settings
    send_immediate_alerts: bool = True
    alert_cooldown_minutes: int = 5                # Minimum time between alerts

@dataclass
class PositionMonitoringConfig:
    """Position monitoring configuration"""
    check_interval_seconds: int = 30               # Check positions every 30 seconds
    max_concurrent_positions: int = 10             # Maximum concurrent positions
    
    # Risk monitoring thresholds
    stop_loss_percentage: float = 0.05             # 5% stop loss
    take_profit_percentage: float = 0.15           # 15% take profit
    drawdown_alert_threshold: float = 0.03         # 3% drawdown alert
    
    # Auto-exit conditions
    enable_auto_exit: bool = True
    max_holding_period_hours: int = 24             # Maximum holding period
    confidence_degradation_threshold: float = 0.3  # Exit if confidence drops below 30%
    
    # Correlation monitoring
    correlation_monitoring_enabled: bool = True
    correlation_alert_threshold: float = 0.8      # Alert if correlation > 80%
    max_correlated_positions: int = 3             # Max positions with >80% correlation

@dataclass
class ExecutionConfig:
    """Trade execution configuration"""
    default_timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    
    # Execution strategies
    enable_vwap: bool = True
    enable_twap: bool = True
    enable_iceberg: bool = True
    
    # Risk-based execution parameters
    high_risk_force_limit_orders: bool = True     # Force limit orders for high risk trades
    low_confidence_reduce_size: bool = True       # Reduce size for low confidence trades
    
    # Market impact limits
    max_market_impact_bps: int = 20               # Maximum 20 basis points market impact
    max_adv_percentage: float = 0.05              # Maximum 5% of average daily volume
    
    # Slippage tolerance
    default_slippage_tolerance: float = 0.005     # 0.5% default slippage tolerance
    high_volatility_slippage_tolerance: float = 0.01  # 1% in high volatility
    low_liquidity_slippage_tolerance: float = 0.002   # 0.2% in low liquidity

@dataclass
class MLRiskConfig:
    """Complete ML risk management configuration"""
    # Core components
    ml_risk_thresholds: MLRiskThresholds = field(default_factory=MLRiskThresholds)
    circuit_breakers: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    emergency_stops: EmergencyStopConfig = field(default_factory=EmergencyStopConfig)
    position_monitoring: PositionMonitoringConfig = field(default_factory=PositionMonitoringConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Environment-specific overrides
    environment_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Trading mode adjustments
    mode_adjustments: Dict[str, Dict[str, Any]] = field(default_factory=dict)

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class MLRiskConfigManager:
    """
    Configuration manager for ML-enhanced risk management
    
    Provides environment-aware and mode-aware configuration with safe defaults
    and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 environment: str = "development",
                 trading_mode: str = "paper_trading"):
        """Initialize configuration manager"""
        
        self.config_path = config_path
        self.environment = environment
        self.trading_mode = trading_mode
        
        # Load base configuration
        self.base_config = self._get_default_config()
        
        # Load from file if provided
        if config_path:
            self.file_config = self._load_config_file(config_path)
        else:
            self.file_config = {}
        
        # Build final configuration
        self.config = self._build_final_config()
        
        logger.info(f"ML Risk Configuration loaded for {environment}/{trading_mode}")
    
    def _get_default_config(self) -> MLRiskConfig:
        """Get default ML risk configuration with safe settings"""
        
        config = MLRiskConfig()
        
        # Set conservative defaults
        config.ml_risk_thresholds = MLRiskThresholds(
            min_confidence_threshold=0.7,    # Higher confidence required by default
            max_uncertainty_threshold=0.3,   # Lower uncertainty tolerance
            min_ensemble_agreement=0.8,      # Higher ensemble agreement required
            confidence_scaling_factor=1.5,   # More conservative scaling
            prediction_stability_threshold=0.7,
            feature_importance_threshold=0.4
        )
        
        config.circuit_breakers = CircuitBreakerConfig(
            daily_loss_limit=0.02,           # Tighter daily loss limit (2%)
            volatility_spike_multiplier=2.5, # More sensitive to volatility
            model_performance_threshold=0.5, # Higher performance requirement
            execution_failure_rate_threshold=0.15,  # Lower failure tolerance
            auto_recovery_enabled=False,     # Manual recovery by default
            minimum_recovery_wait_minutes=30
        )
        
        config.emergency_stops = EmergencyStopConfig(
            max_consecutive_losses=3,        # Fewer losses before emergency stop
            max_portfolio_drawdown=0.05,     # Tighter drawdown limit (5%)
            max_position_drawdown=0.10,      # Tighter position drawdown (10%)
            require_override_code=True,
            auto_recovery_enabled=False
        )
        
        return config
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    return json.load(f)
                else:
                    logger.warning(f"Unknown config file format: {config_path}")
                    return {}
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            return {}
    
    def _build_final_config(self) -> MLRiskConfig:
        """Build final configuration by merging defaults, file config, and overrides"""
        
        # Start with base config
        final_config = self.base_config
        
        # Apply file configuration
        if 'ml_risk_management' in self.file_config:
            self._merge_config_section(final_config, self.file_config['ml_risk_management'])
        
        # Apply environment-specific overrides
        if self.environment in final_config.environment_overrides:
            env_overrides = final_config.environment_overrides[self.environment]
            self._merge_config_section(final_config, env_overrides)
        
        # Apply trading mode adjustments
        if self.trading_mode in final_config.mode_adjustments:
            mode_adjustments = final_config.mode_adjustments[self.trading_mode]
            self._merge_config_section(final_config, mode_adjustments)
        
        # Apply specific overrides for different environments/modes
        self._apply_environment_specific_settings(final_config)
        self._apply_trading_mode_specific_settings(final_config)
        
        # Validate final configuration
        self._validate_config(final_config)
        
        return final_config
    
    def _merge_config_section(self, base_config: MLRiskConfig, overrides: Dict[str, Any]):
        """Merge configuration overrides into base config"""
        
        # Merge ML risk thresholds
        if 'ml_risk_thresholds' in overrides:
            thresholds = overrides['ml_risk_thresholds']
            for key, value in thresholds.items():
                if hasattr(base_config.ml_risk_thresholds, key):
                    setattr(base_config.ml_risk_thresholds, key, value)
        
        # Merge circuit breaker config
        if 'circuit_breakers' in overrides:
            breakers = overrides['circuit_breakers']
            for key, value in breakers.items():
                if hasattr(base_config.circuit_breakers, key):
                    setattr(base_config.circuit_breakers, key, value)
        
        # Merge emergency stop config
        if 'emergency_stops' in overrides:
            stops = overrides['emergency_stops']
            for key, value in stops.items():
                if hasattr(base_config.emergency_stops, key):
                    setattr(base_config.emergency_stops, key, value)
        
        # Merge position monitoring config
        if 'position_monitoring' in overrides:
            monitoring = overrides['position_monitoring']
            for key, value in monitoring.items():
                if hasattr(base_config.position_monitoring, key):
                    setattr(base_config.position_monitoring, key, value)
        
        # Merge execution config
        if 'execution' in overrides:
            execution = overrides['execution']
            for key, value in execution.items():
                if hasattr(base_config.execution, key):
                    setattr(base_config.execution, key, value)
    
    def _apply_environment_specific_settings(self, config: MLRiskConfig):
        """Apply environment-specific settings"""
        
        if self.environment == "production":
            # Production: Maximum safety
            config.ml_risk_thresholds.min_confidence_threshold = 0.8
            config.circuit_breakers.daily_loss_limit = 0.02
            config.emergency_stops.max_consecutive_losses = 3
            config.emergency_stops.auto_recovery_enabled = False
            
        elif self.environment == "staging":
            # Staging: Balanced settings for testing
            config.ml_risk_thresholds.min_confidence_threshold = 0.65
            config.circuit_breakers.daily_loss_limit = 0.03
            config.emergency_stops.max_consecutive_losses = 4
            config.circuit_breakers.auto_recovery_enabled = True
            
        elif self.environment == "development":
            # Development: More relaxed for testing
            config.ml_risk_thresholds.min_confidence_threshold = 0.5
            config.circuit_breakers.daily_loss_limit = 0.05
            config.emergency_stops.max_consecutive_losses = 5
            config.circuit_breakers.auto_recovery_enabled = True
    
    def _apply_trading_mode_specific_settings(self, config: MLRiskConfig):
        """Apply trading mode-specific settings"""
        
        if self.trading_mode == "paper_trading":
            # Paper trading: Allow more experimentation
            config.ml_risk_thresholds.min_confidence_threshold *= 0.9  # 10% more relaxed
            config.emergency_stops.auto_recovery_enabled = True
            config.circuit_breakers.auto_recovery_enabled = True
            
        elif self.trading_mode == "conservative":
            # Conservative: Extra safety
            config.ml_risk_thresholds.min_confidence_threshold = 0.8
            config.ml_risk_thresholds.max_uncertainty_threshold = 0.2
            config.circuit_breakers.daily_loss_limit = 0.015  # 1.5%
            config.emergency_stops.max_portfolio_drawdown = 0.03  # 3%
            
        elif self.trading_mode == "aggressive":
            # Aggressive: Higher risk tolerance but still safe
            config.ml_risk_thresholds.min_confidence_threshold = 0.6
            config.ml_risk_thresholds.max_uncertainty_threshold = 0.4
            config.circuit_breakers.daily_loss_limit = 0.04  # 4%
            config.emergency_stops.max_portfolio_drawdown = 0.08  # 8%
    
    def _validate_config(self, config: MLRiskConfig):
        """Validate configuration for consistency and safety"""
        
        errors = []
        warnings = []
        
        # Validate ML risk thresholds
        thresholds = config.ml_risk_thresholds
        
        if thresholds.min_confidence_threshold < 0.3:
            errors.append("Minimum confidence threshold is dangerously low (< 30%)")
        elif thresholds.min_confidence_threshold < 0.5:
            warnings.append("Minimum confidence threshold is low (< 50%)")
        
        if thresholds.max_uncertainty_threshold > 0.7:
            errors.append("Maximum uncertainty threshold is dangerously high (> 70%)")
        
        if thresholds.min_ensemble_agreement < 0.5:
            errors.append("Minimum ensemble agreement is too low (< 50%)")
        
        # Validate circuit breakers
        breakers = config.circuit_breakers
        
        if breakers.daily_loss_limit > 0.1:
            errors.append("Daily loss limit is dangerously high (> 10%)")
        elif breakers.daily_loss_limit > 0.05:
            warnings.append("Daily loss limit is high (> 5%)")
        
        if breakers.model_performance_threshold < 0.3:
            errors.append("Model performance threshold is too low (< 30%)")
        
        # Validate emergency stops
        stops = config.emergency_stops
        
        if stops.max_portfolio_drawdown > 0.2:
            errors.append("Maximum portfolio drawdown is dangerously high (> 20%)")
        elif stops.max_portfolio_drawdown > 0.1:
            warnings.append("Maximum portfolio drawdown is high (> 10%)")
        
        if stops.max_consecutive_losses > 10:
            warnings.append("Maximum consecutive losses is high (> 10)")
        
        # Validate position monitoring
        monitoring = config.position_monitoring
        
        if monitoring.max_concurrent_positions > 20:
            warnings.append("Maximum concurrent positions is high (> 20)")
        
        if monitoring.stop_loss_percentage > 0.1:
            warnings.append("Stop loss percentage is high (> 10%)")
        
        # Log validation results
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            raise ValueError(f"Configuration validation failed: {errors}")
        
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration validation warning: {warning}")
    
    def get_config(self) -> MLRiskConfig:
        """Get the final configuration"""
        return self.config
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        
        def dataclass_to_dict(obj):
            """Convert dataclass to dictionary recursively"""
            if hasattr(obj, '__dataclass_fields__'):
                return {
                    field.name: dataclass_to_dict(getattr(obj, field.name))
                    for field in obj.__dataclass_fields__.values()
                }
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            else:
                return obj
        
        return dataclass_to_dict(self.config)
    
    def save_config(self, output_path: str):
        """Save current configuration to file"""
        
        config_dict = self.get_config_dict()
        
        try:
            with open(output_path, 'w') as f:
                if output_path.endswith('.yaml') or output_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif output_path.endswith('.json'):
                    json.dump(config_dict, f, indent=2, default=str)
                else:
                    logger.warning(f"Unknown output format: {output_path}")
                    return False
            
            logger.info(f"Configuration saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {output_path}: {e}")
            return False
    
    def reload_config(self):
        """Reload configuration from file"""
        
        if self.config_path:
            self.file_config = self._load_config_file(self.config_path)
            self.config = self._build_final_config()
            logger.info("Configuration reloaded successfully")
        else:
            logger.warning("No config file path specified, cannot reload")
    
    def update_environment(self, environment: str):
        """Update environment and rebuild configuration"""
        
        self.environment = environment
        self.config = self._build_final_config()
        logger.info(f"Configuration updated for environment: {environment}")
    
    def update_trading_mode(self, trading_mode: str):
        """Update trading mode and rebuild configuration"""
        
        self.trading_mode = trading_mode
        self.config = self._build_final_config()
        logger.info(f"Configuration updated for trading mode: {trading_mode}")

# ============================================================================
# CONFIGURATION TEMPLATES
# ============================================================================

def create_default_config_template() -> Dict[str, Any]:
    """Create a default configuration template for ML risk management"""
    
    return {
        'ml_risk_management': {
            'ml_risk_thresholds': {
                'min_confidence_threshold': 0.6,
                'max_uncertainty_threshold': 0.4,
                'min_ensemble_agreement': 0.7,
                'confidence_scaling_factor': 2.0,
                'prediction_stability_threshold': 0.6,
                'feature_importance_threshold': 0.3,
                'high_volatility_confidence_boost': 0.1,
                'low_volume_confidence_boost': 0.05,
                'correlation_breakdown_threshold': 0.3
            },
            'circuit_breakers': {
                'daily_loss_limit': 0.03,
                'volatility_spike_multiplier': 3.0,  
                'model_performance_threshold': 0.4,
                'execution_failure_rate_threshold': 0.2,
                'data_quality_threshold': 0.8,
                'correlation_breakdown_threshold': 0.3,
                'auto_recovery_enabled': True,
                'recovery_check_interval_minutes': 5,
                'minimum_recovery_wait_minutes': 15,
                'escalate_after_minutes': 60,
                'max_consecutive_triggers': 3
            },
            'emergency_stops': {
                'max_consecutive_losses': 5,
                'max_portfolio_drawdown': 0.10,
                'max_position_drawdown': 0.20,
                'model_complete_failure_threshold': 0.1,
                'require_override_code': True,
                'override_code_length': 8,
                'auto_recovery_enabled': False,
                'send_immediate_alerts': True,
                'alert_cooldown_minutes': 5
            },
            'position_monitoring': {
                'check_interval_seconds': 30,
                'max_concurrent_positions': 10,
                'stop_loss_percentage': 0.05,
                'take_profit_percentage': 0.15,
                'drawdown_alert_threshold': 0.03,
                'enable_auto_exit': True,
                'max_holding_period_hours': 24,
                'confidence_degradation_threshold': 0.3,
                'correlation_monitoring_enabled': True,
                'correlation_alert_threshold': 0.8,
                'max_correlated_positions': 3
            },
            'execution': {
                'default_timeout_seconds': 30,
                'retry_attempts': 3,
                'retry_delay_seconds': 5,
                'enable_vwap': True,
                'enable_twap': True,
                'enable_iceberg': True,
                'high_risk_force_limit_orders': True,
                'low_confidence_reduce_size': True,
                'max_market_impact_bps': 20,
                'max_adv_percentage': 0.05,
                'default_slippage_tolerance': 0.005,
                'high_volatility_slippage_tolerance': 0.01,
                'low_liquidity_slippage_tolerance': 0.002
            }
        }
    }

def save_config_template(output_path: str = "config/ml_risk_config_template.yaml"):
    """Save a default configuration template to file"""
    
    template = create_default_config_template()
    
    try:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration template saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving configuration template: {e}")
        return False