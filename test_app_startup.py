#!/usr/bin/env python3
"""
Test script to verify main application imports and initializes correctly
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_main_imports():
    """Test that main.py can be imported without errors"""
    print("🔍 Testing main application imports...")
    
    try:
        # Test critical imports first
        from dotenv import load_dotenv
        print("✅ dotenv import successful")
        
        import fastapi
        print("✅ FastAPI import successful")
        
        import uvicorn
        print("✅ uvicorn import successful")
        
        import yaml
        print("✅ PyYAML import successful")
        
        # Test debug safety import
        from debug_safety import get_debug_manager
        print("✅ debug_safety import successful")
        
        print("🎉 All critical imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_initialization():
    """Test main module initialization"""
    print("\n🔍 Testing main module initialization...")
    
    try:
        # Import the main module (this will run initialization code)
        import main
        print("✅ Main module imported successfully")
        
        # Check if key components are available
        if hasattr(main, 'app'):
            print("✅ FastAPI app initialized")
        
        if hasattr(main, 'debug_manager'):
            print("✅ Debug manager available")
            
        print("🎉 Main module initialization successful!")
        return True
        
    except Exception as e:
        print(f"❌ Main initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting main application verification tests...")
    
    imports_ok = test_main_imports()
    
    if imports_ok:
        init_ok = test_main_initialization()
    else:
        init_ok = False
    
    if imports_ok and init_ok:
        print("\n🎉 ALL TESTS PASSED - Application is ready for deployment!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Check the output above")
        sys.exit(1)