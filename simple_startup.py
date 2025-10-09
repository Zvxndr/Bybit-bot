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

# Load AI components directly before starting main app
print("\n🤖 Loading AI Components...")

def load_ai_component_directly(module_name, class_name, file_path):
    """Load AI component directly from file path"""
    try:
        if not os.path.exists(file_path):
            return None
            
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            return None
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        sys.modules[f"src.{module_name}"] = module
        
        spec.loader.exec_module(module)
        
        if hasattr(module, class_name):
            component_class = getattr(module, class_name)
            print(f"   ✅ {class_name} loaded successfully")
            return component_class
        return None
    except Exception as e:
        print(f"   ⚠️  {class_name} load failed: {e}")
        return None

# Pre-load AI components into sys.modules
ai_components = {}

# Load MultiExchangeDataManager
medm_class = load_ai_component_directly(
    'multi_exchange_provider',
    'MultiExchangeDataManager',
    '/app/src/data/multi_exchange_provider.py'
)
if medm_class:
    sys.modules['src.data.multi_exchange_provider'] = sys.modules['multi_exchange_provider']
    ai_components['MultiExchangeDataManager'] = medm_class

# Load AutomatedPipelineManager  
apm_class = load_ai_component_directly(
    'automated_pipeline_manager',
    'AutomatedPipelineManager',
    '/app/src/bot/pipeline/automated_pipeline_manager.py'
)
if apm_class:
    sys.modules['src.bot.pipeline.automated_pipeline_manager'] = sys.modules['automated_pipeline_manager']
    ai_components['AutomatedPipelineManager'] = apm_class

# Load MLStrategyDiscoveryEngine
ml_class = load_ai_component_directly(
    'ml_engine', 
    'MLStrategyDiscoveryEngine',
    '/app/src/bot/ml_strategy_discovery/ml_engine.py'
)
if ml_class:
    sys.modules['src.bot.ml_strategy_discovery.ml_engine'] = sys.modules['ml_engine']
    ai_components['MLStrategyDiscoveryEngine'] = ml_class

print(f"   🎯 AI Components loaded: {len(ai_components)}/3")

# Now start the main application
print("\n🎯 Starting Application...")

try:
    # Change to working directory
    os.chdir('/app')
    
    # Ensure proper Python path
    if '/app/src' not in sys.path:
        sys.path.insert(0, '/app/src')
    
    # Import main module with AI components pre-loaded
    print("   📦 Importing main module with AI components...")
    
    try:
        # Direct import approach
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "/app/src/main.py")
        
        if spec and spec.loader:
            main_module = importlib.util.module_from_spec(spec)
            sys.modules['main'] = main_module
            spec.loader.exec_module(main_module)
            
            app = main_module.app
            print("   ✅ Main application loaded with AI components!")
        else:
            raise ImportError("Could not create module spec for main.py")
        
    except Exception as import_err:
        print(f"   ❌ Main import failed: {import_err}")
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