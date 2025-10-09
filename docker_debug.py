#!/usr/bin/env python3
"""
Docker Environment Debug Script
Analyzes the Docker environment to understand import path issues
"""

import os
import sys
from pathlib import Path

def debug_docker_environment():
    """Debug the Docker environment for import issues"""
    
    print("🐳 DOCKER ENVIRONMENT DEBUG")
    print("=" * 50)
    
    # Current working directory
    cwd = Path.cwd()
    print(f"📁 Current Working Directory: {cwd}")
    
    # Python path
    print(f"\n🐍 Python Path ({len(sys.path)} entries):")
    for i, path in enumerate(sys.path):
        print(f"   {i}: {path}")
    
    # Environment variables
    print(f"\n🔧 Key Environment Variables:")
    env_vars = ['PYTHONPATH', 'PATH', 'PWD', 'HOME']
    for var in env_vars:
        value = os.getenv(var, 'NOT SET')
        print(f"   {var}: {value}")
    
    # File system structure
    print(f"\n📂 File System Structure:")
    
    # Check /app directory (Docker default)
    app_dir = Path('/app')
    if app_dir.exists():
        print(f"   /app exists: {list(app_dir.iterdir())[:10]}...")
        
        src_dir = app_dir / 'src'
        if src_dir.exists():
            print(f"   /app/src exists: {list(src_dir.iterdir())[:10]}...")
            
            # Check for key modules
            key_modules = [
                'src/main.py',
                'src/data/multi_exchange_provider.py', 
                'src/bot/pipeline/automated_pipeline_manager.py'
            ]
            
            for module in key_modules:
                module_path = app_dir / module
                exists = module_path.exists()
                print(f"   {module}: {'✅ EXISTS' if exists else '❌ MISSING'}")
    else:
        print("   /app directory does not exist")
    
    # Current directory structure
    print(f"\n📂 Current Directory Contents:")
    try:
        for item in cwd.iterdir():
            item_type = "DIR" if item.is_dir() else "FILE"
            print(f"   {item_type}: {item.name}")
    except Exception as e:
        print(f"   Error listing directory: {e}")
    
    # Test import strategies
    print(f"\n🧪 Import Strategy Tests:")
    
    strategies = [
        ("Relative import", "from .data.multi_exchange_provider import MultiExchangeDataManager"),
        ("Absolute import", "from src.data.multi_exchange_provider import MultiExchangeDataManager"),
        ("Direct import", "from data.multi_exchange_provider import MultiExchangeDataManager"),
    ]
    
    for name, import_stmt in strategies:
        try:
            exec(import_stmt)
            print(f"   ✅ {name}: SUCCESS")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    print(f"\n" + "=" * 50)
    print("Debug complete - check logs for import resolution")

if __name__ == "__main__":
    debug_docker_environment()