#!/usr/bin/env python3
"""
Production Startup with Schema Validation
==========================================

Enhanced production startup that includes database schema checking
before launching the main application.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_schema_check():
    """Run database schema validation before startup."""
    print("🔍 Validating database schema...")
    
    try:
        result = subprocess.run([
            sys.executable, "/app/container_schema_check.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Database schema validated successfully")
            if result.stdout.strip():
                print(result.stdout)
            return True
        else:
            print("⚠️  Schema validation encountered issues:")
            if result.stderr:
                print(result.stderr)
            # Continue anyway - let main app handle it
            return True
            
    except subprocess.TimeoutExpired:
        print("⚠️  Schema check timed out - continuing startup")
        return True
    except Exception as e:
        print(f"⚠️  Schema check error: {e}")
        return True

def main():
    """Enhanced production startup with schema validation."""
    
    print("🚀 Production Startup with Enhanced Validation")
    print("=" * 50)
    
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
    
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path[:3]}...")
    
    # Run schema validation
    schema_ok = run_schema_check()
    
    # Continue with normal startup
    print("\n🚀 Launching main application...")
    
    # Debug: Show file system structure
    print("\n📂 Docker File System Check:")
    for root in ['/app', '/app/src']:
        if os.path.exists(root):
            items = os.listdir(root)[:8]  # Show first 8 items
            print(f"   {root}: {items}")
        else:
            print(f"   {root}: NOT FOUND")
    
    # Debug: Check critical files
    critical_files = [
        '/app/src/main.py',
        '/app/src/bot/pipeline/automated_pipeline_manager.py'
    ]
    print("\n📋 Critical Files:")
    for file_path in critical_files:
        exists = "✅" if os.path.exists(file_path) else "❌"
        print(f"   {exists} {file_path}")
    
    print("\n🎯 Starting main application...")
    
    try:
        # Set __name__ to __main__ to trigger server startup
        import runpy
        
        # Execute main.py as a script to ensure the uvicorn server starts
        runpy.run_path("/app/src/main.py", run_name="__main__")
        
    except Exception as e:
        print(f"💥 Application startup failed: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()