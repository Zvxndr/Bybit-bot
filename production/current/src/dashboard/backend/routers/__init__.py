"""
API Routers Package
Contains all FastAPI router modules for the dashboard backend
"""

from .trading_router import router as trading_router
from .analytics_router import router as analytics_router
from .health_router import router as health_router
from .ml_router import router as ml_router

__all__ = ["trading_router", "analytics_router", "health_router", "ml_router"]