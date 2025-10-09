#!/usr/bin/env python3
"""
Production Deployment Startup Script
Bypasses Docker import issues with direct module loading
"""

import sys
import os
from pathlib import Path

# Ensure we're in the right directory
os.chdir('/app')

# Add all necessary paths
paths_to_add = [
    '/app',
    '/app/src',
    '/app/src/bot',
    '/app/src/data'
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Set environment variables
os.environ['PYTHONPATH'] = ':'.join(paths_to_add)

print("🚀 Production Deployment Startup")
print(f"📁 Current directory: {os.getcwd()}")
print(f"🐍 Python path: {sys.path[:5]}...")

# Debug: Show file system structure
print("\n📂 Docker File System Debug:")
for root in ['/app', '/app/src']:
    if os.path.exists(root):
        print(f"   {root}: {os.listdir(root)[:10]}")
    else:
        print(f"   {root}: NOT FOUND")

# Debug: Check specific files
critical_files = [
    '/app/src/main.py',
    '/app/src/data/multi_exchange_provider.py',
    '/app/src/bot/pipeline/automated_pipeline_manager.py'
]
print("\n📋 Critical Files Check:")
for file_path in critical_files:
    exists = "✅" if os.path.exists(file_path) else "❌"
    print(f"   {exists} {file_path}")

# Direct module loading approach
def load_module_directly(module_name, file_path):
    """Load a module directly from file path"""
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return None
    except Exception as e:
        print(f"❌ Failed to load {module_name}: {e}")
        return None

# Load critical modules directly
print("📦 Loading critical modules directly...")

# Load MultiExchangeDataManager
medm_path = "/app/src/data/multi_exchange_provider.py"
if os.path.exists(medm_path):
    medm_module = load_module_directly("multi_exchange_provider", medm_path)
    if medm_module and hasattr(medm_module, 'MultiExchangeDataManager'):
        print("✅ MultiExchangeDataManager loaded directly")
        sys.modules['src.data.multi_exchange_provider'] = medm_module
        sys.modules['data.multi_exchange_provider'] = medm_module
    else:
        print("❌ MultiExchangeDataManager not found in module")
else:
    print(f"❌ Module not found: {medm_path}")

# Load AutomatedPipelineManager  
apm_path = "/app/src/bot/pipeline/automated_pipeline_manager.py"
if os.path.exists(apm_path):
    apm_module = load_module_directly("automated_pipeline_manager", apm_path)
    if apm_module and hasattr(apm_module, 'AutomatedPipelineManager'):
        print("✅ AutomatedPipelineManager loaded directly")
        sys.modules['src.bot.pipeline.automated_pipeline_manager'] = apm_module
        sys.modules['bot.pipeline.automated_pipeline_manager'] = apm_module
    else:
        print("❌ AutomatedPipelineManager not found in module")
else:
    print(f"❌ Module not found: {apm_path}")

# Load MLStrategyDiscoveryEngine
ml_path = "/app/src/bot/ml_strategy_discovery/ml_engine.py"
if os.path.exists(ml_path):
    ml_module = load_module_directly("ml_engine", ml_path)
    if ml_module and hasattr(ml_module, 'MLStrategyDiscoveryEngine'):
        print("✅ MLStrategyDiscoveryEngine loaded directly")
        sys.modules['src.bot.ml_strategy_discovery.ml_engine'] = ml_module
        sys.modules['bot.ml_strategy_discovery.ml_engine'] = ml_module
    else:
        print("❌ MLStrategyDiscoveryEngine not found in module")
else:
    print(f"❌ Module not found: {ml_path}")

print("🎯 Starting main application...")

# Now import and run main
try:
    # Import main module
    main_path = "/app/src/main.py"
    main_module = load_module_directly("main", main_path)
    
    if main_module and hasattr(main_module, 'app'):
        print("✅ Main module loaded, starting FastAPI application...")
        
        # Start with uvicorn
        import uvicorn
        uvicorn.run(main_module.app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
        
    else:
        print("❌ Could not load main module or find 'app' attribute")
        sys.exit(1)
        
except Exception as e:
    print(f"💥 Failed to start application: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)