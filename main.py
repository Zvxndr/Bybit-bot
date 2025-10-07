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
    print("� Python Path:", sys.path[0])
    print("🎯 Loading production_ai_pipeline.py...")
    
    try:
        # Import the production pipeline
        from production_ai_pipeline import app, PORT
        
        print(f"✅ Successfully imported production pipeline app")
        print(f"🌐 Starting server on port {PORT}")
        
        # Run the FastAPI app directly
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
        
        port = int(os.getenv("PORT", 8000))
        uvicorn.run(fallback_app, host="0.0.0.0", port=port)