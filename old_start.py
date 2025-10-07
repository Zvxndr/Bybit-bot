#!/usr/bin/env python3
"""
DigitalOcean Deployment Entry Point
Ensures the production AI pipeline is started correctly
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Set environment variables for deployment
os.environ.setdefault('PYTHONPATH', str(current_dir))
os.environ.setdefault('PYTHONUNBUFFERED', '1')

# Import and run the production AI pipeline
if __name__ == "__main__":
    print("🚀 Starting DigitalOcean Production AI Pipeline...")
    print(f"📁 Working Directory: {current_dir}")
    print(f"🐍 Python Path: {sys.path[0]}")
    
    try:
        # Import the production pipeline module
        import production_ai_pipeline
        print("✅ Production AI Pipeline module loaded successfully")
        
        # The production_ai_pipeline.py file will run when imported due to the
        # if __name__ == "__main__": block, but we need to call it explicitly here
        from production_ai_pipeline import app
        import uvicorn
        
        port = int(os.getenv('PORT', 8000))
        print(f"🌐 Starting server on port {port}")
        
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        
    except Exception as e:
        print(f"❌ Failed to start production pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)