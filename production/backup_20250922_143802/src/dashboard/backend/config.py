"""
Configuration Settings for Dashboard Backend
Centralized configuration management
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class DatabaseSettings(BaseSettings):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "bybit_dashboard"
    username: str = "dashboard_user"
    password: str = "dashboard_pass"
    max_connections: int = 20
    min_connections: int = 5
    
    class Config:
        env_prefix = "DB_"

class APISettings(BaseSettings):
    """API configuration"""
    title: str = "Bybit Trading Bot v2.0 Dashboard API"
    version: str = "3.0.0"
    description: str = "Advanced real-time trading dashboard with ML insights"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"
    
    # CORS settings
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:3001"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    class Config:
        env_prefix = "API_"

class WebSocketSettings(BaseSettings):
    """WebSocket configuration"""
    max_connections: int = 100
    rate_limit_window: int = 60  # seconds
    max_messages_per_window: int = 1000
    ping_interval: int = 30  # seconds
    ping_timeout: int = 10  # seconds
    
    class Config:
        env_prefix = "WS_"

class MonitoringSettings(BaseSettings):
    """Monitoring and alerting configuration"""
    health_check_interval: int = 5  # seconds
    metrics_collection_interval: int = 2  # seconds
    data_retention_hours: int = 24
    
    # Alert thresholds
    cpu_warning_threshold: float = 80.0
    cpu_critical_threshold: float = 95.0
    memory_warning_threshold: float = 85.0
    memory_critical_threshold: float = 95.0
    disk_warning_threshold: float = 90.0
    disk_critical_threshold: float = 98.0
    response_time_threshold: float = 1000.0  # ms
    
    class Config:
        env_prefix = "MONITOR_"

class MLSettings(BaseSettings):
    """ML component configuration"""
    model_update_interval: int = 300  # seconds (5 minutes)
    prediction_cache_ttl: int = 60  # seconds
    feature_importance_update_interval: int = 1800  # seconds (30 minutes)
    
    # Model paths (relative to project root)
    phase1_components_path: str = "src/bot"
    phase2_components_path: str = "src/bot"
    
    class Config:
        env_prefix = "ML_"

class SecuritySettings(BaseSettings):
    """Security configuration"""
    secret_key: str = "bybit-dashboard-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours
    
    # API key settings (if needed for external APIs)
    bybit_api_key: Optional[str] = None
    bybit_api_secret: Optional[str] = None
    
    class Config:
        env_prefix = "SECURITY_"

class DashboardSettings(BaseSettings):
    """Main dashboard settings container"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.database = DatabaseSettings()
        self.api = APISettings()
        self.websocket = WebSocketSettings()
        self.monitoring = MonitoringSettings()
        self.ml = MLSettings()
        self.security = SecuritySettings()
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = DashboardSettings()

# Helper functions
def get_database_url() -> str:
    """Get database connection URL"""
    db = settings.database
    return f"postgresql://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}"

def get_cors_config() -> dict:
    """Get CORS configuration"""
    api = settings.api
    return {
        "allow_origins": api.cors_origins,
        "allow_credentials": api.cors_allow_credentials,
        "allow_methods": api.cors_allow_methods,
        "allow_headers": api.cors_allow_headers
    }