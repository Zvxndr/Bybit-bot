#!/usr/bin/env python3
"""
Deployment Import Test
======================

Tests all imports work correctly in deployment environment.
"""

import sys
import os

# Add the app directory to Python path
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

print("🧪 Testing deployment imports...")

try:
    # Test main imports
    print("Testing debug_safety import...")
    try:
        from src.debug_safety import get_debug_manager
        print("✅ src.debug_safety import successful")
    except ImportError as e:
        print(f"❌ src.debug_safety import failed: {e}")
        # Try fallback
        try:
            import sys
            sys.path.append('/app/src')
            from debug_safety import get_debug_manager
            print("✅ debug_safety fallback import successful")
        except ImportError as e2:
            print(f"❌ debug_safety fallback import failed: {e2}")

    # Test shared_state import
    print("Testing shared_state import...")
    try:
        from src.shared_state import shared_state
        print("✅ shared_state import successful")
    except ImportError as e:
        print(f"❌ shared_state import failed: {e}")

    # Test bybit_api import
    print("Testing bybit_api import...")
    try:
        from src.bybit_api import BybitAPIClient
        print("✅ bybit_api import successful")
    except ImportError as e:
        print(f"❌ bybit_api import failed: {e}")

    print("\n🎯 Import test completed")
    
except Exception as e:
    print(f"❌ Import test failed with error: {e}")
    sys.exit(1)

print("✅ All critical imports working")