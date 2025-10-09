"""
Historical Data Downloader Stub
Simple stub to prevent import errors during development.
"""

import logging

logger = logging.getLogger(__name__)

class HistoricalDataDownloader:
    """Stub for historical data downloader functionality"""
    
    def __init__(self):
        self.initialized = True
        logger.info("✅ Historical data downloader stub initialized")
    
    async def download_historical_data(self, symbol: str, timeframe: str = "1h", limit: int = 1000):
        """Placeholder for downloading historical data"""
        logger.info(f"📊 Historical data download requested for {symbol} ({timeframe}, {limit} bars)")
        return {"status": "success", "message": "Historical data downloader not fully implemented"}
    
    def get_available_symbols(self):
        """Get available symbols for download"""
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    def get_status(self):
        """Get downloader status"""
        return {"status": "ready", "type": "development_stub"}

# Export for main.py import
historical_downloader = HistoricalDataDownloader()