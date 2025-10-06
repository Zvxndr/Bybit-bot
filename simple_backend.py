"""
Simple Working Backend - Bypass Complex Imports
===============================================
A minimal FastAPI backend that actually starts and works
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Try FastAPI first, fallback to simple HTTP server
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

def create_simple_backend():
    """Create a working backend that bypasses import issues"""
    
    if FASTAPI_AVAILABLE:
        # Use FastAPI if available
        app = FastAPI(title="Bybit Trading Bot API", version="1.0.0")
        
        # Enable CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mock data
        mock_data = {
            "system_status": "running",
            "trading_enabled": False,
            "api_connected": False,
            "balance": {
                "USDT": 1000.0,
                "BTC": 0.0,
                "ETH": 0.0
            },
            "positions": [],
            "recent_trades": [],
            "strategies": [
                {"name": "momentum", "status": "active", "profit": 12.5},
                {"name": "mean_reversion", "status": "paused", "profit": -3.2}
            ]
        }
        
        @app.get("/")
        async def root():
            return {"message": "Bybit Trading Bot API", "status": "running"}
        
        @app.get("/api/status")
        async def get_status():
            return {
                "status": mock_data["system_status"],
                "timestamp": datetime.now().isoformat(),
                "trading_enabled": mock_data["trading_enabled"],
                "api_connected": mock_data["api_connected"]
            }
        
        @app.get("/api/portfolio")
        async def get_portfolio():
            return {
                "balance": mock_data["balance"],
                "positions": mock_data["positions"],
                "total_value": sum(mock_data["balance"].values()),
                "pnl": 15.75
            }
        
        @app.get("/api/strategies")
        async def get_strategies():
            return {"strategies": mock_data["strategies"]}
        
        @app.get("/api/trades")
        async def get_trades():
            return {"trades": mock_data["recent_trades"]}
        
        return app
    
    else:
        # Fallback to simple HTTP server
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class APIHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_json({"message": "Bybit Trading Bot API", "status": "running"})
                elif self.path == '/api/status':
                    self.send_json({
                        "status": "running",
                        "timestamp": datetime.now().isoformat(),
                        "trading_enabled": False,
                        "api_connected": False
                    })
                elif self.path == '/api/portfolio':
                    self.send_json({
                        "balance": {"USDT": 1000.0, "BTC": 0.0, "ETH": 0.0},
                        "positions": [],
                        "total_value": 1000.0,
                        "pnl": 15.75
                    })
                else:
                    self.send_error(404)
            
            def send_json(self, data):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            
            def log_message(self, format, *args):
                pass  # Suppress logs
        
        return HTTPServer, APIHandler

def main():
    """Start the backend server"""
    print("🚀 Starting Simple Backend Server...")
    
    if FASTAPI_AVAILABLE:
        try:
            print("📡 Using FastAPI backend")
            app = create_simple_backend()
            print("✅ Backend ready at: http://localhost:8000")
            print("📊 API endpoints:")
            print("   - http://localhost:8000/api/status")
            print("   - http://localhost:8000/api/portfolio") 
            print("   - http://localhost:8000/api/strategies")
            print("🛑 Press Ctrl+C to stop")
            
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
        except Exception as e:
            print(f"❌ FastAPI failed: {e}")
            print("📡 Falling back to simple HTTP server")
            FASTAPI_AVAILABLE = False
    
    else:
        print("📡 Using simple HTTP backend")
        server_class, handler_class = create_simple_backend()
        server = server_class(('localhost', 8000), handler_class)
        print("✅ Backend ready at: http://localhost:8000")
        print("📊 API endpoints:")
        print("   - http://localhost:8000/api/status")
        print("   - http://localhost:8000/api/portfolio")
        print("🛑 Press Ctrl+C to stop")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Backend stopped")

if __name__ == "__main__":
    main()