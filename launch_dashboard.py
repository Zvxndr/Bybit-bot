#!/usr/bin/env python3
"""
Launch the unified dashboard with separated balance system
"""

import sys
import os
import uvicorn
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Fix path issues for static files
os.chdir(os.path.dirname(__file__))

if __name__ == "__main__":
    print("🚀 LAUNCHING UNIFIED DASHBOARD WITH SEPARATED BALANCES")
    print("📊 Paper/Testnet vs Live Trading Balances")
    
    try:
        # Import from src directory
        from main import app
        
        print("✅ Successfully loaded unified dashboard")
        print("🌐 Starting server on http://localhost:8080")
        print("💰 Portfolio will show SEPARATED balances:")
        print("  🧪 Paper/Testnet Balance (Phase 2)")
        print("  🚀 Live Trading Balance (Phase 3)")
        
        # Run the server
        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()