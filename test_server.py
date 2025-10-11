from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os

# Add src directory to path
sys.path.append('src')

# Import our historical data downloader
from historical_data_downloader import HistoricalDataDownloader

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the downloader
downloader = HistoricalDataDownloader()

@app.get("/api/historical-data/available-periods/{symbol}/{timeframe}")
async def get_available_periods_endpoint(symbol: str, timeframe: str):
    """Get available historical data periods for a symbol and timeframe"""
    try:
        result = downloader.get_available_periods(symbol, timeframe)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting available periods: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Historical Data API Test Server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)