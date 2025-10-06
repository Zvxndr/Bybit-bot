#!/usr/bin/env python3
"""
Test script to verify debug_safety module works correctly
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_debug_safety():
    """Test debug safety module import and configuration"""
    print("🔍 Testing debug_safety module...")
    
    try:
        # Test import
        from debug_safety import get_debug_manager, is_debug_mode, block_trading_if_debug
        print("✅ Debug safety imports successful")
        
        # Test manager creation
        manager = get_debug_manager()
        print(f"✅ Debug manager created: {type(manager)}")
        
        # Test configuration loading
        config = manager.config
        print(f"✅ Config loaded: {type(config)} with {len(config)} keys")
        print(f"   Debug mode: {config.get('debug_mode')}")
        print(f"   Config keys: {list(config.keys())}")
        
        # Test functions
        debug_status = is_debug_mode()
        print(f"✅ Debug mode check: {debug_status}")
        
        block_status = block_trading_if_debug("test_operation")
        print(f"✅ Trading block check: {block_status}")
        
        print("🎉 All debug_safety tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Debug safety test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yaml_import():
    """Test PyYAML import specifically"""
    print("\n🔍 Testing PyYAML import...")
    
    try:
        import yaml
        print("✅ PyYAML imported successfully")
        
        # Test basic YAML functionality
        test_data = {'test': True, 'value': 123}
        yaml_str = yaml.dump(test_data)
        parsed_data = yaml.safe_load(yaml_str)
        
        print(f"✅ YAML round-trip successful: {parsed_data}")
        return True
        
    except ImportError as e:
        print(f"❌ PyYAML import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ YAML functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting debug_safety verification tests...")
    
    yaml_ok = test_yaml_import()
    debug_ok = test_debug_safety()
    
    if yaml_ok and debug_ok:
        print("\n🎉 ALL TESTS PASSED - Debug safety is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Check the output above")
        sys.exit(1)