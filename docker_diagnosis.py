#!/usr/bin/env python3
"""
Docker Debug and Diagnosis Script
=================================

Comprehensive diagnostic tool to identify Docker deployment issues.
"""

import os
import sys
import traceback
from pathlib import Path

def diagnose_docker_environment():
    """Comprehensive Docker environment diagnosis"""
    print("🔍 COMPREHENSIVE DOCKER DIAGNOSIS")
    print("=" * 60)
    
    # Basic environment info
    print(f"📁 Working Directory: {os.getcwd()}")
    print(f"🐍 Python Version: {sys.version}")
    print(f"🎯 Python Executable: {sys.executable}")
    print(f"📦 Python Path: {sys.path[:3]}...")
    print()
    
    # Check file system structure
    print("📂 DOCKER FILE SYSTEM ANALYSIS:")
    
    # Check /app structure
    app_structure = {}
    if os.path.exists("/app"):
        try:
            for root, dirs, files in os.walk("/app"):
                level = root.replace("/app", "").count(os.sep)
                if level < 4:  # Limit depth
                    indent = " " * 2 * level
                    app_structure[root] = {
                        'dirs': dirs[:10],  # Limit to first 10
                        'files': files[:10]  # Limit to first 10
                    }
                    print(f"{indent}{os.path.basename(root)}/")
                    for d in dirs[:5]:
                        print(f"{indent}  📁 {d}/")
                    for f in files[:5]:
                        print(f"{indent}  📄 {f}")
                    if len(files) > 5:
                        print(f"{indent}  📄 ... {len(files)-5} more files")
                    print()
        except Exception as e:
            print(f"❌ Error walking /app: {e}")
    
    # Check critical paths
    print("📋 CRITICAL PATH ANALYSIS:")
    critical_paths = [
        "/app/src",
        "/app/src/data", 
        "/app/src/bot",
        "/app/src/main.py",
        "/app/src/data/multi_exchange_provider.py",
        "/app/src/bot/pipeline/automated_pipeline_manager.py",
        "/app/production_startup.py"
    ]
    
    for path in critical_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                try:
                    contents = os.listdir(path)
                    print(f"✅ {path} → {len(contents)} items: {contents[:3]}{'...' if len(contents) > 3 else ''}")
                except PermissionError:
                    print(f"🔒 {path} → Permission denied")
            else:
                size = os.path.getsize(path)
                print(f"✅ {path} → {size} bytes")
        else:
            print(f"❌ {path} → NOT FOUND")
    
    print()
    
    # Check if data directory exists but is empty
    print("🔍 DATA DIRECTORY INVESTIGATION:")
    data_paths = [
        "/app/src/data",
        "/app/data", 
        "/app/src/data/__init__.py",
        "/app/src/data/multi_exchange_provider.py"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                contents = os.listdir(path) 
                print(f"✅ {path} exists with {len(contents)} items: {contents}")
            else:
                print(f"✅ {path} exists ({os.path.getsize(path)} bytes)")
        else:
            print(f"❌ {path} does not exist")
    
    print()
    
    # Test Python imports
    print("🐍 PYTHON IMPORT TESTING:")
    
    # Test basic imports
    try:
        import importlib.util
        print("✅ importlib.util available")
    except Exception as e:
        print(f"❌ importlib.util failed: {e}")
    
    # Test sys.modules manipulation
    try:
        test_module = type(sys)('test_module')
        sys.modules['test_module'] = test_module  
        print("✅ sys.modules manipulation works")
        del sys.modules['test_module']
    except Exception as e:
        print(f"❌ sys.modules manipulation failed: {e}")
    
    # Test module loading
    try:
        spec = importlib.util.spec_from_file_location("test", "/app/src/main.py")
        if spec:
            print("✅ Module spec creation works")
        else:
            print("❌ Module spec creation returned None")
    except Exception as e:
        print(f"❌ Module spec creation failed: {e}")
    
    print()
    
    # Environment variables check
    print("🌍 ENVIRONMENT VARIABLES:")
    env_vars = ['PYTHONPATH', 'PORT', 'PYTHONUNBUFFERED']
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"   {var}: {value}")
    
    print()
    
    # Git information
    print("📝 GIT INFORMATION:")
    git_files = ['.git/HEAD', '.git/refs/heads/main']
    for git_file in git_files:
        if os.path.exists(git_file):
            try:
                with open(git_file, 'r') as f:
                    content = f.read().strip()
                print(f"✅ {git_file}: {content}")
            except Exception as e:
                print(f"❌ {git_file}: {e}")
        else:
            print(f"❌ {git_file}: not found")

def test_simple_imports():
    """Test simple module imports to isolate the issue"""
    print("\n🧪 SIMPLE IMPORT TESTING:")
    print("-" * 40)
    
    # Test 1: Simple main.py import
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("main_test", "/app/src/main.py")
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            print(f"✅ Module created: {type(module)}")
            
            # Check if we can add to sys.modules
            sys.modules['main_test'] = module
            print("✅ Added to sys.modules")
            
            # Try to execute (this might fail, but we'll see why)
            try:
                spec.loader.exec_module(module)
                print("✅ Module executed successfully")
            except Exception as exec_error:
                print(f"❌ Module execution failed: {exec_error}")
                print(f"   Error type: {type(exec_error)}")
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Simple import test failed: {e}")
        traceback.print_exc()

def suggest_fixes():
    """Analyze issues and suggest fixes"""
    print("\n🔧 SUGGESTED FIXES:")
    print("=" * 40)
    
    # Check if src/data directory missing
    if not os.path.exists("/app/src/data"):
        print("🚨 CRITICAL: /app/src/data directory missing!")
        print("   CAUSE: Docker COPY command not copying src/data directory")
        print("   FIX: Check .dockerignore or add explicit COPY src/data src/data")
        
    # Check for Python version issues
    if sys.version_info < (3, 9):
        print("⚠️  Python version might be causing 'not subscriptable' errors")
        print("   Consider updating type annotations")
    
    # Check for file permissions
    try:
        test_file = "/tmp/test_permissions"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("✅ File write permissions OK")
    except Exception as e:
        print(f"❌ File permission issue: {e}")

if __name__ == "__main__":
    try:
        diagnose_docker_environment()
        test_simple_imports()
        suggest_fixes()
        
        print("\n📋 DIAGNOSIS COMPLETE")
        print("   Check the output above for specific issues and fixes")
        
    except Exception as e:
        print(f"💥 Diagnosis script failed: {e}")
        traceback.print_exc()