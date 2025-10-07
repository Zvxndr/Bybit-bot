"""
Bybit Trading Bot v2.0 - Phase 3 Dashboard Backend
Advanced FastAPI backend for real-time trading dashboard
"""

__version__ = "3.0.0"
__author__ = "Bybit Trading Bot Team"
__description__ = "Advanced real-time trading dashboard backend with ML insights"

from .main import app, dashboard_backend
from .config import settings

__all__ = ["app", "dashboard_backend", "settings"]