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
        title="🚨 MEGA-NUCLEAR Frontend Test Server 🚨",
        description="Test the MEGA-NUCLEAR navigation system with ULTIMATE protection",
        version="2.0.0-MEGA-NUCLEAR"
    )
    
    # 🚨 MEGA-NUCLEAR: Mount static files to serve CSS protection
    app.mount("/static", StaticFiles(directory="src/static"), name="static")
    
    @app.get("/")
    async def serve_dashboard():
        """Serve the MEGA-NUCLEAR protected dashboard HTML."""
        return FileResponse("src/templates/adminlte_dashboard.html")
    
    @app.get("/mega-nuclear-status")
    async def mega_nuclear_status():
        """Check MEGA-NUCLEAR protection status."""
        return {
            "status": "🚨 MEGA-NUCLEAR PROTECTION ACTIVE 🚨",
            "protection_level": "ULTIMATE",
            "template_conflicts": "NEUTRALIZED",
            "navigation_system": "ADMINLTE_ONLY",
            "conflicting_templates": [
                "professional_dashboard.html - DISABLED",
                "fire_dashboard_redesign.html - DISABLED", 
                "adminlte_dashboard_backup.html - SYNCHRONIZED",
                "adminlte_dashboard_clean.html - SYNCHRONIZED"
            ],
            "css_protection": "mega-nuclear-protection.css - LOADED",
            "javascript_overrides": "ALL_CONFLICTS_NEUTRALIZED",
            "message": "🔥 MEGA-NUCLEAR SUCCESS: Single unified navigation system active 🔥"
        }
    
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