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

# Enhanced module loading with dependency resolution
def load_module_with_dependencies(module_name, file_path, dependencies=None):
    """Load a module directly from file path with dependency injection"""
    import importlib.util
    try:
        # First, try loading dependencies (skip missing ones)
        if dependencies:
            for dep_name, dep_path in dependencies.items():
                if os.path.exists(dep_path) and dep_name not in sys.modules:
                    try:
                        dep_spec = importlib.util.spec_from_file_location(dep_name, dep_path)
                        if dep_spec and dep_spec.loader:
                            dep_module = importlib.util.module_from_spec(dep_spec)
                            sys.modules[dep_name] = dep_module
                            dep_spec.loader.exec_module(dep_module)
                            print(f"   📦 Loaded dependency: {dep_name}")
                    except Exception as e:
                        print(f"   ⚠️  Skipping dependency {dep_name}: {e}")
        
        # Load main module with fallback import handling
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            
            # Create mock imports for missing dependencies
            original_import = __builtins__['__import__']
            
            def mock_import(name, *args, **kwargs):
                try:
                    return original_import(name, *args, **kwargs)
                except ImportError:
                    # Create a mock module for missing imports
                    mock_module = type(sys)('mock_' + name.replace('.', '_'))
                    sys.modules[name] = mock_module
                    return mock_module
            
            # Temporarily replace import function
            __builtins__['__import__'] = mock_import
            try:
                spec.loader.exec_module(module)
            finally:
                __builtins__['__import__'] = original_import
            
            return module
        return None
    except Exception as e:
        print(f"❌ Failed to load {module_name}: {e}")
        return None

# Load critical modules strategically
print("📦 Loading critical modules with dependencies...")

# Load MultiExchangeDataManager first (fewer dependencies)
medm_path = "/app/src/data/multi_exchange_provider.py"  
if os.path.exists(medm_path):
    medm_module = load_module_with_dependencies("multi_exchange_provider", medm_path)
    if medm_module and hasattr(medm_module, 'MultiExchangeDataManager'):
        print("✅ MultiExchangeDataManager loaded directly")
        sys.modules['src.data.multi_exchange_provider'] = medm_module
        sys.modules['data.multi_exchange_provider'] = medm_module
    else:
        print("❌ MultiExchangeDataManager not found in module")
else:
    print(f"❌ Module not found: {medm_path}")

# Load MLStrategyDiscoveryEngine (moderate dependencies)  
ml_path = "/app/src/bot/ml_strategy_discovery/ml_engine.py"
if os.path.exists(ml_path):
    ml_deps = {
        'src.bot.ml_strategy_discovery.data_infrastructure': '/app/src/bot/ml_strategy_discovery/data_infrastructure.py',
        'src.bot.utils.logging': '/app/src/bot/utils/logging.py'
    }
    ml_module = load_module_with_dependencies("ml_engine", ml_path, ml_deps)
    if ml_module and hasattr(ml_module, 'MLStrategyDiscoveryEngine'):
        print("✅ MLStrategyDiscoveryEngine loaded directly") 
        sys.modules['src.bot.ml_strategy_discovery.ml_engine'] = ml_module
        sys.modules['bot.ml_strategy_discovery.ml_engine'] = ml_module
    else:
        print("❌ MLStrategyDiscoveryEngine not found in module")
else:
    print(f"❌ Module not found: {ml_path}")

# Load AutomatedPipelineManager last (most dependencies)
apm_path = "/app/src/bot/pipeline/automated_pipeline_manager.py"
if os.path.exists(apm_path):
    apm_deps = {
        'src.bot.database.manager': '/app/src/bot/database/manager.py',
        'src.bot.utils.logging': '/app/src/bot/utils/logging.py',
        'src.bot.models.strategy': '/app/src/bot/models/strategy.py'
    }
    apm_module = load_module_with_dependencies("automated_pipeline_manager", apm_path, apm_deps)
    if apm_module and hasattr(apm_module, 'AutomatedPipelineManager'):
        print("✅ AutomatedPipelineManager loaded directly")
        sys.modules['src.bot.pipeline.automated_pipeline_manager'] = apm_module
        sys.modules['bot.pipeline.automated_pipeline_manager'] = apm_module
    else:
        print("❌ AutomatedPipelineManager not found in module")
else:
    print(f"❌ Module not found: {apm_path}")

print("🎯 Starting main application...")

# Now import and run main
try:
    # Import main module
    main_path = "/app/src/main.py"
    main_module = load_module_with_dependencies("main", main_path)
    
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