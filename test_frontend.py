"""
Simple Frontend Test Server

Just serves the fixed dashboard to test navigation
"""

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

def create_test_app():
    """Create simple test app for frontend."""
    
    app = FastAPI(
        title="Frontend Test Server",
        description="Test the fixed navigation system",
        version="1.0.0"
    )
    
    @app.get("/")
    async def serve_dashboard():
        """Serve the dashboard HTML."""
        return FileResponse("src/templates/adminlte_dashboard.html")
    
    return app

if __name__ == "__main__":
    app = create_test_app()
    
    print("🚀 Starting Frontend Test Server...")
    print("📊 Dashboard: http://localhost:8000")
    print("🔧 Testing Navigation Fixes...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )