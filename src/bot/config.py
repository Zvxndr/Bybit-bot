"""
Configuration management for the trading bot.

This module provides a comprehensive configuration system that supports:
- YAML-based configuration files
- Environment variable overrides
- Validation with Pydantic models
- Dual-mode operation (conservative/aggressive)
- Type safety and documentation
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


class TradingModeConfig(BaseModel):
    """Configuration for a specific trading mode."""
    
    portfolio_drawdown_limit: float = Field(..., ge=0, le=1)
    strategy_drawdown_limit: float = Field(..., ge=0, le=1)
    sharpe_ratio_min: float = Field(..., ge=0)
    var_daily_limit: float = Field(..., ge=0, le=1)
    consistency_min: float = Field(..., ge=0, le=1)


class AggressiveModeConfig(TradingModeConfig):
    """Configuration for aggressive trading mode with dynamic risk scaling."""
    
    max_risk_ratio: float = Field(..., ge=0, le=1)
    min_risk_ratio: float = Field(..., ge=0, le=1)
    balance_thresholds: Dict[str, float] = Field(...)
    risk_decay: Literal["linear", "exponential"] = "exponential"
    
    @validator("max_risk_ratio")
    def max_risk_must_be_greater_than_min(cls, v, values):
        if "min_risk_ratio" in values and v <= values["min_risk_ratio"]:
            raise ValueError("max_risk_ratio must be greater than min_risk_ratio")
        return v


class ConservativeModeConfig(TradingModeConfig):
    """Configuration for conservative trading mode with fixed risk."""
    
    risk_ratio: float = Field(..., ge=0, le=1)


class TradingConfig(BaseModel):
    """Main trading configuration."""
    
    mode: Literal["conservative", "aggressive"] = "aggressive"
    base_balance: float = Field(1000, gt=0)
    aggressive_mode: AggressiveModeConfig
    conservative_mode: ConservativeModeConfig


class ExchangeConfig(BaseModel):
    """Exchange connection and trading configuration."""
    
    name: str = "bybit"
    api_key: str = Field(..., min_length=1)
    api_secret: str = Field(..., min_length=1)
    sandbox: bool = True
    testnet_url: str = "https://api-testnet.bybit.com"
    mainnet_url: str = "https://api.bybit.com"
    
    symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframes: Dict[str, str] = Field(default_factory=lambda: {
        "primary": "1h",
        "secondary": "4h", 
        "daily": "1d"
    })
    
    order_type: Literal["market", "limit"] = "limit"
    max_slippage: float = Field(0.001, ge=0, le=1)
    order_timeout: int = Field(30, gt=0)


class DatabaseConfig(BaseModel):
    """Database configuration with development/production profiles."""
    
    development: Dict[str, Union[str, int]] = Field(default_factory=lambda: {
        "dialect": "duckdb",
        "path": "./data/trading_bot.db"
    })
    
    production: Dict[str, Union[str, int]] = Field(default_factory=lambda: {
        "dialect": "postgresql",
        "host": "localhost",
        "port": 5432,
        "name": "trading_bot",
        "user": "trader",
        "password": ""
    })
    
    pool_size: int = Field(10, gt=0)
    max_overflow: int = Field(20, gt=0)
    echo: bool = False


class MLConfig(BaseModel):
    """Machine learning and feature engineering configuration."""
    
    feature_engineering: Dict = Field(default_factory=dict)
    models: Dict = Field(default_factory=dict)
    cross_validation: Dict = Field(default_factory=dict)


class BacktestConfig(BaseModel):
    """Backtesting and validation configuration."""
    
    walk_forward: Dict = Field(default_factory=dict)
    validation: Dict = Field(default_factory=dict)
    costs: Dict[str, float] = Field(default_factory=dict)


class RiskConfig(BaseModel):
    """Risk management configuration."""
    
    portfolio: Dict = Field(default_factory=dict)
    strategy: Dict = Field(default_factory=dict)
    circuit_breakers: Dict = Field(default_factory=dict)


class TaxConfig(BaseModel):
    """Tax reporting configuration (Australia-specific)."""
    
    country: str = "australia"
    financial_year_end: str = "06-30"
    rba_api: Dict = Field(default_factory=dict)
    cgt: Dict = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"
    file: Dict = Field(default_factory=dict)
    console: Dict = Field(default_factory=dict)
    fields: List[str] = Field(default_factory=list)


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    
    host: str = "0.0.0.0"
    port: int = Field(8501, ge=1024, le=65535)
    title: str = "Bybit Trading Bot Dashboard"
    metrics: Dict = Field(default_factory=dict)
    mode_switching: Dict = Field(default_factory=dict)


class Config(BaseModel):
    """Main configuration class containing all settings."""
    
    trading: TradingConfig
    exchange: ExchangeConfig
    database: DatabaseConfig
    ml: MLConfig = Field(default_factory=MLConfig)
    backtesting: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    tax: TaxConfig = Field(default_factory=TaxConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = Field(default_factory=lambda: DashboardConfig(port=8501))
    monitoring: Dict = Field(default_factory=dict)
    development: Dict = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""
        
        extra = "allow"  # Allow additional fields
        validate_assignment = True
        
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """
        Load configuration from YAML file with environment variable substitution.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Validated Config instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValidationError: If configuration validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML content
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
        
        # Substitute environment variables
        yaml_content = cls._substitute_env_vars(yaml_content)
        
        # Parse YAML
        try:
            config_data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML config: {e}")
        
        # Create and validate config
        return cls(**config_data)
    
    @staticmethod
    def _substitute_env_vars(content: str) -> str:
        """
        Substitute environment variables in configuration content.
        
        Supports patterns like:
        - ${VAR_NAME} - Required variable
        - ${VAR_NAME:default_value} - Variable with default
        """
        import re
        
        def replace_var(match):
            var_spec = match.group(1)
            
            if ':' in var_spec:
                var_name, default_value = var_spec.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                var_name = var_spec
                value = os.getenv(var_name)
                if value is None:
                    raise ValueError(f"Required environment variable not set: {var_name}")
                return value
        
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_var, content)
    
    def get_current_mode_config(self) -> Union[AggressiveModeConfig, ConservativeModeConfig]:
        """Get the configuration for the current trading mode."""
        if self.trading.mode == "aggressive":
            return self.trading.aggressive_mode
        else:
            return self.trading.conservative_mode
    
    def get_database_config(self, environment: str = "development") -> Dict:
        """Get database configuration for specified environment."""
        if environment == "production":
            return self.database.production
        else:
            return self.database.development
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.dict()
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)