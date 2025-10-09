#!/usr/bin/env python3
"""
Simplified Production Startup
============================

Minimal, robust startup script that avoids complex import manipulation.
Focuses on getting the basic application running first.
"""

import sys
import os
import traceback
from pathlib import Path

# Basic environment setup
os.chdir('/app')
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

print("🚀 Simplified Production Startup")
print(f"📁 Working Directory: {os.getcwd()}")
print(f"🐍 Python Version: {sys.version_info[:2]}")

# Environment check
print("\n📂 File System Check:")
critical_files = [
    "/app/src/main.py",
    "/app/src/data", 
    "/app/src/bot",
    "/app/production_startup.py"
]

for file_path in critical_files:
    if os.path.exists(file_path):
        if os.path.isdir(file_path):
            count = len(os.listdir(file_path)) if os.path.isdir(file_path) else 0
            print(f"   ✅ {file_path} ({count} items)")
        else:
            size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} ({size} bytes)")
    else:
        print(f"   ❌ {file_path}")

# Check if data directory is missing and handle it
if not os.path.exists("/app/src/data"):
    print("\n🚨 CRITICAL: /app/src/data directory missing!")
    print("   Creating minimal data structure...")
    
    try:
        os.makedirs("/app/src/data", exist_ok=True)
        
        # Create minimal multi_exchange_provider.py
        minimal_provider = '''"""
Minimal Multi-Exchange Provider Placeholder
"""

class MultiExchangeDataManager:
    """Minimal placeholder for missing MultiExchangeDataManager"""
    
    def __init__(self):
        self.name = "MultiExchangeDataManager (Placeholder)"
        print("⚠️  Using placeholder MultiExchangeDataManager")
    
    async def get_market_data(self, symbol):
        """Placeholder method"""
        return {"symbol": symbol, "price": 0, "status": "placeholder"}

# Module exports
__all__ = ['MultiExchangeDataManager']
'''
        
        with open("/app/src/data/multi_exchange_provider.py", "w") as f:
            f.write(minimal_provider)
            
        with open("/app/src/data/__init__.py", "w") as f:
            f.write("# Data module\n")
            
        print("   ✅ Created minimal data structure")
        
    except Exception as e:
        print(f"   ❌ Failed to create data structure: {e}")

# Direct main.py execution approach
print("\n🎯 Starting Application...")

try:
    # Change to src directory for relative imports
    os.chdir('/app')
    
    # Add src to path if not already there
    if '/app/src' not in sys.path:
        sys.path.insert(0, '/app/src')
    
    # Try direct import first
    print("   📦 Attempting direct main import...")
    try:
        from src import main as main_module
        app = main_module.app
        print("   ✅ Direct import successful!")
        
    except ImportError as import_err:
        print(f"   ⚠️  Direct import failed: {import_err}")
        print("   📦 Attempting sys.path manipulation...")
        
        # Fallback: manipulate sys.path and try again
        original_path = sys.path.copy()
        try:
            # Clear and rebuild path
            sys.path = ['/app/src', '/app', '/usr/local/lib/python3.11/site-packages'] + sys.path
            
            import main as main_module
            app = main_module.app 
            print("   ✅ Fallback import successful!")
            
        except Exception as fallback_err:
            print(f"   ❌ Fallback import also failed: {fallback_err}")
            
            # Last resort: exec the file directly
            print("   📦 Attempting direct file execution...")
            try:
                main_globals = {}
                with open('/app/src/main.py', 'r') as f:
                    main_code = f.read()
                
                # Execute the main.py file in a controlled environment
                exec(main_code, main_globals)
                
                if 'app' in main_globals:
                    app = main_globals['app']
                    print("   ✅ Direct execution successful!")
                else:
                    raise ValueError("No 'app' object found in main.py")
                    
            except Exception as exec_err:
                print(f"   💥 All import methods failed!")
                print(f"      Last error: {exec_err}")
                traceback.print_exc()
                sys.exit(1)
    
    # Start the server
    print("   🌐 Starting uvicorn server...")
    
    import uvicorn
    
    # Get port from environment
    port = int(os.environ.get("PORT", 8080))
    
    print(f"   📡 Server starting on 0.0.0.0:{port}")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    
except Exception as e:
    print(f"\n💥 Startup failed: {e}")
    print("\n📋 Error Details:")
    traceback.print_exc()
    
    print(f"\n🔍 Debug Information:")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Python path: {sys.path[:3]}")
    print(f"   Files in /app: {os.listdir('/app')[:10] if os.path.exists('/app') else 'N/A'}")
    print(f"   Files in /app/src: {os.listdir('/app/src')[:10] if os.path.exists('/app/src') else 'N/A'}")
    
    sys.exit(1)