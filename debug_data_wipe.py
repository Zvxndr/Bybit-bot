#!/usr/bin/env python3
"""
Data Wipe Debug Script
=====================

This script directly tests the data wipe functionality without going through the web interface.
It will help diagnose if the issue is in the button or in the actual data clearing function.
"""

import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

def test_direct_data_wipe():
    """Test the data wipe function directly"""
    print("🔧 Testing Direct Data Wipe Function")
    print("=" * 50)
    
    try:
        # Import the shared state module
        from shared_state import SharedState
        
        print("✅ Successfully imported SharedState")
        
        # Create instance
        shared_state = SharedState()
        print("✅ Successfully created SharedState instance")
        
        # Check initial state
        initial_data = shared_state.get_all_data()
        print(f"📊 Initial state has {len(initial_data)} top-level keys")
        
        # Create some test files/directories to wipe
        test_paths = [
            "data/test_file.txt",
            "logs/test_log.txt", 
            "test.db"
        ]
        
        print(f"\n🔧 Creating test files for wipe test...")
        for test_path in test_paths:
            try:
                # Create directory if needed
                Path(test_path).parent.mkdir(parents=True, exist_ok=True)
                # Create test file
                Path(test_path).write_text(f"Test file created for wipe test: {test_path}")
                print(f"✅ Created: {test_path}")
            except Exception as e:
                print(f"⚠️ Could not create {test_path}: {e}")
        
        # Now test the wipe function
        print(f"\n🔥 Executing clear_all_data()...")
        shared_state.clear_all_data()
        print("✅ clear_all_data() completed without errors")
        
        # Check if files were removed
        print(f"\n🔍 Checking if test files were removed...")
        removed_count = 0
        for test_path in test_paths:
            if not Path(test_path).exists():
                print(f"✅ Removed: {test_path}")
                removed_count += 1
            else:
                print(f"❌ Still exists: {test_path}")
        
        print(f"\n📊 Summary: {removed_count}/{len(test_paths)} test files removed")
        
        # Check the state was reset
        final_data = shared_state.get_all_data()
        print(f"📊 Final state has {len(final_data)} top-level keys")
        
        # Check if logs show the wipe
        logs = final_data.get('logs', [])
        wipe_log = None
        for log in logs:
            if 'SYSTEM RESET' in str(log) or 'cleared' in str(log):
                wipe_log = log
                break
        
        if wipe_log:
            print(f"✅ Found wipe log entry: {wipe_log}")
        else:
            print("⚠️ No wipe log entry found in system logs")
        
        print(f"\n🎯 RESULT: Data wipe function works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing data wipe: {e}")
        import traceback
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return False

def test_api_access():
    """Test if the API endpoint is accessible"""
    print(f"\n🌐 Testing API Endpoint Access")
    print("=" * 30)
    
    try:
        import requests
        
        url = "http://localhost:8080/api/admin/wipe-data"
        print(f"🔧 Testing POST request to: {url}")
        
        response = requests.post(url, timeout=5)
        print(f"📊 Response Status: {response.status_code}")
        
        if response.headers.get('content-type', '').startswith('application/json'):
            data = response.json()
            print(f"📄 Response Data: {data}")
            
            if data.get('success'):
                print("✅ API endpoint responded successfully!")
                return True
            else:
                print("⚠️ API endpoint responded but reported failure")
                return False
        else:
            print(f"📄 Response Text: {response.text}")
            return response.status_code == 200
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to localhost:8080")
        print("💡 Make sure the bot is running: python src/main.py")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def main():
    """Run all data wipe tests"""
    print("🔥 Data Wipe Debug Script")
    print("=" * 50)
    
    # Test 1: Direct function test
    direct_success = test_direct_data_wipe()
    
    # Test 2: API endpoint test  
    api_success = test_api_access()
    
    # Summary
    print(f"\n" + "=" * 50)
    print("🏁 FINAL RESULTS")
    print("=" * 50)
    
    print(f"Direct Function Test: {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"API Endpoint Test: {'✅ PASS' if api_success else '❌ FAIL'}")
    
    if direct_success and api_success:
        print("\n🎉 Both tests passed! The data wipe should work.")
        print("💡 If the button still fails, the issue might be in the web interface JavaScript.")
    elif direct_success and not api_success:
        print("\n⚠️ Function works but API fails.")
        print("💡 Check if the bot is running and the server is accessible.")
    elif not direct_success and api_success:
        print("\n⚠️ API responds but function has issues.")
        print("💡 There might be an issue with the data clearing logic.")
    else:
        print("\n❌ Both tests failed.")
        print("💡 There are fundamental issues with the data wipe functionality.")

if __name__ == "__main__":
    main()