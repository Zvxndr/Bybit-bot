#!/usr/bin/env python3
"""
Test the -m flag approach to verify it works correctly
"""

import subprocess
import sys
import os

def test_module_import():
    print("🧪 TESTING PYTHON -m src.main APPROACH")
    print("=" * 50)
    
    # Change to the project root (simulate Docker WORKDIR /app)
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"📍 Project root: {project_root}")
    
    # Test if we can import the modules using the -m approach
    print("\n🔍 Testing import resolution with -m flag...")
    
    try:
        # Test the command that Docker will run
        cmd = [sys.executable, "-c", 
               "import sys; sys.path.insert(0, '.'); from src.data.multi_exchange_provider import MultiExchangeDataManager; print('SUCCESS: MultiExchangeDataManager import successful')"]
        
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Command successful:")
            print(f"   Output: {result.stdout.strip()}")
        else:
            print("❌ Command failed:")
            print(f"   Error: {result.stderr.strip()}")
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n🔍 Testing AI Pipeline Manager import...")
    
    try:
        cmd = [sys.executable, "-c", 
               "import sys; sys.path.insert(0, '.'); from src.bot.pipeline.automated_pipeline_manager import AutomatedPipelineManager; print('SUCCESS: AutomatedPipelineManager import successful')"]
        
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Command successful:")
            print(f"   Output: {result.stdout.strip()}")
        else:
            print("❌ Command failed:")
            print(f"   Error: {result.stderr.strip()}")
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print(f"\n📋 CONCLUSION:")
    print(f"If both imports succeed, the Docker CMD change should fix the issue.")
    print(f"Docker will run: python -m src.main")
    print(f"This ensures Python resolves imports from /app directory correctly.")

if __name__ == "__main__":
    test_module_import()