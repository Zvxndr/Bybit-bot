"""
Simple test version of the dashboard backend
"""

from fastapi import FastAPI
from datetime import datetime

app = FastAPI(
    title="Bybit Trading Bot v2.0 Dashboard API",
    description="Advanced real-time trading dashboard with ML insights",
    version="3.0.0"
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Bybit Trading Bot v2.0 Dashboard API",
        "version": "3.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Phase 3 Dashboard Backend is running!"
    }

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)