#!/usr/bin/env python3
"""
Emergency Docker Import Fix
This script creates a startup wrapper that ensures all modules can be imported correctly
"""

import os
import sys
from pathlib import Path

def fix_docker_imports():
    """Fix import paths for Docker environment"""
    
    # Get the application root directory
    if __file__.endswith('docker_import_fix.py'):
        # We're running the fix script directly
        app_root = Path(__file__).parent.absolute()
    else:
        # We're being imported
        app_root = Path('/app').absolute()
    
    print(f"🔧 Docker Import Fix: App root = {app_root}")
    
    # Add all necessary paths to sys.path
    paths_to_add = [
        str(app_root),                    # /app
        str(app_root / 'src'),            # /app/src  
        str(app_root / 'src' / 'bot'),    # /app/src/bot
        str(app_root / 'src' / 'data'),   # /app/src/data
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"✅ Added to PYTHONPATH: {path}")
    
    # Set PYTHONPATH environment variable
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_pythonpath = ':'.join(paths_to_add + [current_pythonpath] if current_pythonpath else paths_to_add)
    os.environ['PYTHONPATH'] = new_pythonpath
    
    print(f"🔧 Updated PYTHONPATH environment: {new_pythonpath}")
    
    # Verify key modules exist
    key_modules = [
        'src/main.py',
        'src/data/multi_exchange_provider.py',
        'src/bot/pipeline/automated_pipeline_manager.py'
    ]
    
    print(f"\n📦 Verifying key modules:")
    for module in key_modules:
        module_path = app_root / module
        exists = module_path.exists()
        print(f"   {module}: {'✅' if exists else '❌'}")
        
        if not exists:
            print(f"      Expected at: {module_path}")
    
    return True

if __name__ == "__main__":
    print("🚀 Running Docker Import Fix...")
    fix_docker_imports()
    
    # Now run the main application
    print("\n🎯 Starting main application...")
    try:
        # Import and run main
        sys.path.insert(0, '/app')
        from src.main import app
        
        # Start with uvicorn
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
        
    except Exception as e:
        print(f"💥 Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)