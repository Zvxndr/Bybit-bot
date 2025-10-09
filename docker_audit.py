#!/usr/bin/env python3
"""
Comprehensive Docker Environment Audit
Diagnose exactly what's happening in the Docker container
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def audit_environment():
    """Comprehensive environment audit"""
    print("=" * 80)
    print("🔍 COMPREHENSIVE DOCKER ENVIRONMENT AUDIT")
    print("=" * 80)
    
    # 1. Basic Python Environment
    print("\n📍 PYTHON ENVIRONMENT:")
    print(f"   Python version: {sys.version}")
    print(f"   Python executable: {sys.executable}")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   __file__: {__file__}")
    print(f"   __name__: {__name__}")
    
    # 2. Python Path Analysis
    print(f"\n📍 PYTHON PATH (first 10 entries):")
    for i, path in enumerate(sys.path[:10]):
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"   {i}: {exists} {path}")
    
    # 3. Environment Variables
    print(f"\n📍 RELEVANT ENVIRONMENT VARIABLES:")
    env_vars = ['PYTHONPATH', 'PATH', 'PWD', 'WORKDIR', 'PORT', 'PYTHONUNBUFFERED']
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"   {var}: {value}")
    
    # 4. File System Structure
    print(f"\n📍 FILE SYSTEM STRUCTURE:")
    
    # Check current directory structure
    current_dir = Path(os.getcwd())
    print(f"   Current directory contents ({current_dir}):")
    try:
        for item in sorted(current_dir.iterdir()):
            item_type = "📁" if item.is_dir() else "📄"
            print(f"     {item_type} {item.name}")
    except Exception as e:
        print(f"     ❌ Error reading current directory: {e}")
    
    # Check for src directory
    src_paths = [
        Path('/app/src'),
        Path('./src'),
        Path('../src'),
        current_dir / 'src'
    ]
    
    for src_path in src_paths:
        print(f"\n   Checking src directory: {src_path}")
        if src_path.exists():
            print(f"     ✅ EXISTS")
            try:
                src_contents = list(src_path.iterdir())
                print(f"     Contents ({len(src_contents)} items):")
                for item in sorted(src_contents):
                    if item.is_dir():
                        print(f"       📁 {item.name}/")
                    else:
                        print(f"       📄 {item.name}")
            except Exception as e:
                print(f"     ❌ Error reading src directory: {e}")
        else:
            print(f"     ❌ NOT FOUND")
    
    # 5. Check specific module paths
    print(f"\n📍 MODULE PATH VERIFICATION:")
    
    module_paths = [
        ('/app/src/data/multi_exchange_provider.py', 'Multi-exchange provider'),
        ('/app/src/bot/pipeline/automated_pipeline_manager.py', 'AI Pipeline Manager'),
        ('./src/data/multi_exchange_provider.py', 'Multi-exchange provider (relative)'),
        ('./src/bot/pipeline/automated_pipeline_manager.py', 'AI Pipeline Manager (relative)'),
        ('data/multi_exchange_provider.py', 'Multi-exchange provider (direct)'),
        ('bot/pipeline/automated_pipeline_manager.py', 'AI Pipeline Manager (direct)'),
    ]
    
    for path, description in module_paths:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"   {exists} {description}: {path}")

def test_import_strategies():
    """Test each import strategy individually"""
    print(f"\n📍 TESTING IMPORT STRATEGIES:")
    
    # Add paths like main.py does
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    paths_to_add = [current_dir, parent_dir, '/app', '/app/src']
    for path in paths_to_add:
        if path and os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    print(f"   Added paths to sys.path: {[p for p in paths_to_add if os.path.exists(p)]}")
    
    # Test Multi-exchange provider
    print(f"\n   🧪 TESTING MULTI-EXCHANGE PROVIDER:")
    
    strategies = [
        ('Direct relative', 'from data.multi_exchange_provider import MultiExchangeDataManager'),
        ('Absolute with src', 'from src.data.multi_exchange_provider import MultiExchangeDataManager'),
        ('Import from current', 'import sys; sys.path.insert(0, "."); from data.multi_exchange_provider import MultiExchangeDataManager')
    ]
    
    for strategy_name, import_code in strategies:
        try:
            print(f"     Strategy: {strategy_name}")
            print(f"     Code: {import_code}")
            exec(import_code)
            print(f"     ✅ SUCCESS")
        except Exception as e:
            print(f"     ❌ FAILED: {e}")
            print(f"     Traceback: {traceback.format_exc().split('\\n')[-3:-1]}")
    
    # Test file-based import
    print(f"\n   🧪 TESTING FILE-BASED IMPORT:")
    try:
        import importlib.util
        
        # Try to find the file
        possible_paths = [
            os.path.join(current_dir, 'data', 'multi_exchange_provider.py'),
            '/app/src/data/multi_exchange_provider.py',
            './src/data/multi_exchange_provider.py'
        ]
        
        for path in possible_paths:
            print(f"     Checking path: {path}")
            if os.path.exists(path):
                print(f"     ✅ File found, attempting import...")
                spec = importlib.util.spec_from_file_location("multi_exchange_provider", path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    print(f"     ✅ File-based import SUCCESS")
                    print(f"     Module attributes: {dir(module)[:5]}...")
                    break
                else:
                    print(f"     ❌ Could not create spec")
            else:
                print(f"     ❌ File not found")
    except Exception as e:
        print(f"     ❌ File-based import FAILED: {e}")
        traceback.print_exc()

def main():
    """Run the comprehensive audit"""
    audit_environment()
    test_import_strategies()
    
    print(f"\n" + "=" * 80)
    print("📋 AUDIT COMPLETE")
    print("=" * 80)
    print("Please review the output above to identify the root cause of import failures.")

if __name__ == "__main__":
    main()