"""
API Module Initialization

This module provides the production-grade API layer for the cryptocurrency
trading bot, including real-time prediction serving, monitoring endpoints,
and WebSocket streaming capabilities.
"""

from .prediction_service import create_app, run_server, PredictionAPI

__all__ = ['create_app', 'run_server', 'PredictionAPI']