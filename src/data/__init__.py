"""
Data Management Package for AI Trading Pipeline

This package provides data collection, processing, and multi-exchange integration
for the automated trading strategy discovery system.
"""

from .multi_exchange_provider import MultiExchangeDataManager

__all__ = [
    'MultiExchangeDataManager'
]