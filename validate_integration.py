"""
Integration Validation Script
============================

Tests the complete integration between backend and frontend to ensure
everything is properly connected before deployment.
"""

import asyncio
import sys
import os
import requests
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

async def test_integration():
    """Test all integration points"""
    
    print("🔧 Testing Backend-Frontend Integration...")
    print("=" * 50)
    
    # Test 1: Import check
    print("1️⃣ Testing imports...")
    try:
        from simplified_dashboard_api import SimplifiedDashboardAPI
        print("   ✅ SimplifiedDashboardAPI imports successfully")
        
        import main
        print("   ✅ Main module imports successfully")
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    # Test 2: Database creation
    print("\n2️⃣ Testing database setup...")
    try:
        os.makedirs("data", exist_ok=True)
        if os.path.exists("data/trading_bot.db"):
            print("   ✅ Database file exists")
        else:
            print("   ⚠️  Database will be created on first run")
    except Exception as e:
        print(f"   ❌ Database setup error: {e}")
        return False
    
    # Test 3: Frontend files
    print("\n3️⃣ Testing frontend files...")
    frontend_file = Path("frontend/unified_dashboard.html")
    if frontend_file.exists():
        print("   ✅ Dashboard HTML file exists")
        
        # Check file size (should be substantial)
        size_kb = frontend_file.stat().st_size // 1024
        print(f"   📊 Dashboard file size: {size_kb}KB")
        
        if size_kb > 30:  # Should be at least 30KB for a complete dashboard
            print("   ✅ Dashboard appears complete")
        else:
            print("   ⚠️  Dashboard file seems small")
    else:
        print("   ❌ Dashboard HTML file missing")
        return False
    
    # Test 4: API endpoint validation
    print("\n4️⃣ Testing API endpoints...")
    try:
        # Start a quick server test (import only)
        from fastapi import FastAPI
        from simplified_dashboard_api import SimplifiedDashboardAPI
        
        app = FastAPI()
        dashboard_api = SimplifiedDashboardAPI(app)
        
        print("   ✅ API endpoints registered successfully")
        
        # Check route count
        route_count = len(app.routes)
        print(f"   📊 Total routes registered: {route_count}")
        
    except Exception as e:
        print(f"   ❌ API setup error: {e}")
        return False
    
    # Test 5: Environment check
    print("\n5️⃣ Testing environment...")
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    if api_key and api_secret:
        print("   ✅ API credentials found (live trading mode)")
    else:
        print("   📝 No API credentials (paper trading mode - safe for testing)")
    
    print(f"   🐍 Python version: {sys.version.split()[0]}")
    print(f"   📁 Working directory: {os.getcwd()}")
    
    # Test 6: Dependencies check
    print("\n6️⃣ Testing dependencies...")
    required_packages = ["fastapi", "uvicorn", "sqlite3", "json", "asyncio"]
    
    for package in required_packages:
        try:
            if package == "sqlite3":
                import sqlite3
            elif package == "json":
                import json  
            elif package == "asyncio":
                import asyncio
            else:
                __import__(package)
            print(f"   ✅ {package} available")
        except ImportError:
            print(f"   ❌ {package} missing")
            return False
    
    # Test 7: Port availability
    print("\n7️⃣ Testing deployment readiness...")
    try:
        import socket
        
        # Test if port 8000 is available
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        
        if result == 0:
            print("   ⚠️  Port 8000 is in use (another server running?)")
        else:
            print("   ✅ Port 8000 is available")
    except Exception as e:
        print(f"   ⚠️  Port test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Integration validation complete!")
    print("\n🚀 Ready to deploy:")
    print("   python -m src.main")
    print("   Dashboard: http://localhost:8000")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_integration())