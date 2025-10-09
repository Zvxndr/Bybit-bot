"""
Main API Module
Centralized API endpoint registration and routing
"""

from fastapi import FastAPI
from .trading_bot_api import TradingBotAPI
from .pipeline_api import PipelineAPI
from .dashboard_analytics_api import DashboardAnalyticsAPI

# API module is available for import
__version__ = "1.0.0"