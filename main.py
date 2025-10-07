#!/usr/bin/env python3
"""
Main Entry Point for DigitalOcean Deployment
Explicitly runs the production AI pipeline system
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

if __name__ == "__main__":
    print("🚀 DIGITALOCEAN MAIN ENTRY POINT")
    print("📁 Working Directory:", current_dir)
    print("🐍 Python Path:", sys.path[0])
    print("🎯 Loading production_ai_pipeline.py...")
    
    try:
        # Import the unified dashboard with separated balances  
        sys.path.insert(0, 'src')
        from main import app
        
        # Use port from environment (Docker uses 8080, local can use 8000)
        PORT = int(os.getenv('PORT', 8080))
        
        print(f"✅ Successfully imported unified dashboard with separated balances")
        print(f"🌐 Starting server on port {PORT}")
        print(f"💰 Serving 3-Phase Balance System:")
        print(f"   📊 Phase 1: Historical Backtest ($10,000 starting capital)")
        print(f"   🧪 Phase 2: Paper/Testnet Balance (simulation)")  
        print(f"   🚀 Phase 3: Live Trading Balance (API required)")
        
        # Run the unified dashboard FastAPI app
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
        
    except Exception as e:
        print(f"❌ ERROR loading production pipeline: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: create a simple test server
        from fastapi import FastAPI
        
        fallback_app = FastAPI()
        
        @fallback_app.get("/")
        def fallback_root():
            return {
                "error": "Failed to load production pipeline",
                "message": str(e),
                "working_dir": str(current_dir),
                "python_path": sys.path[0],
                "files": [f.name for f in current_dir.glob("*.py")]
            }
        
        @fallback_app.get("/health")
        def fallback_health():
            return {"status": "fallback", "error": str(e)}
        
        port = int(os.getenv("PORT", 8000))
        uvicorn.run(fallback_app, host="0.0.0.0", port=port)