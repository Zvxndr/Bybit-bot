#!/usr/bin/env python3
"""
Test the fixed import strategies to verify they work with -m flag
"""

import subprocess
import sys
import os

def test_fixed_imports():
    print("🧪 TESTING FIXED IMPORT STRATEGIES")
    print("=" * 50)
    
    # Test the primary strategy: relative imports with -m flag
    print("\n🔍 Testing relative imports with -m flag...")
    
    try:
        # Test running as a module (like Docker will do)
        cmd = [sys.executable, "-m", "src.main", "--help"]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Module execution successful")
            # Look for our success messages in the output
            if "Multi-exchange provider" in result.stdout or "Multi-exchange provider" in result.stderr:
                print("✅ Multi-exchange provider import detected")
            if "AI Pipeline Manager" in result.stdout or "AI Pipeline Manager" in result.stderr:
                print("✅ AI Pipeline Manager import detected")
        else:
            print("❌ Module execution failed:")
            print(f"   Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ Module execution timed out")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print(f"\n📋 CONCLUSION:")
    print(f"This test simulates exactly how Docker will run the application.")
    print(f"If successful, the Docker deployment should work correctly.")

if __name__ == "__main__":
    test_fixed_imports()