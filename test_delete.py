import requests
import json

# Test the delete endpoints
base_url = "http://localhost:5000"

def test_delete_endpoints():
    print("🗑️  Testing Delete Endpoints")
    print("=" * 40)
    
    # Test 1: Try to delete non-existent data
    print("\n1. Testing delete of non-existent data...")
    try:
        response = requests.delete(f"{base_url}/api/historical-data/symbol/ETHUSDT?timeframe=1h")
        data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Response: {data}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Show current data
    print("\n2. Showing current data...")
    try:
        response = requests.get(f"{base_url}/api/historical-data/discover")
        data = response.json()
        if data['success']:
            for dataset in data['datasets']:
                print(f"   📊 {dataset['symbol']} {dataset['timeframe']}: {dataset['record_count']:,} records")
        else:
            print(f"   ❌ Error: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Delete specific symbol data (we'll use a small test if available)
    print("\n3. Testing specific symbol deletion...")
    try:
        # First check if we have BTCUSDT data to delete
        response = requests.delete(f"{base_url}/api/historical-data/symbol/BTCUSDT?timeframe=15m")
        data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Response: {data}")
        
        if data['success'] and data['deleted_count'] > 0:
            print(f"   ✅ Successfully deleted {data['deleted_count']} records!")
        else:
            print(f"   📊 No data to delete or already empty")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Verify data is gone
    print("\n4. Verifying data after deletion...")
    try:
        response = requests.get(f"{base_url}/api/historical-data/discover")
        data = response.json()
        if data['success']:
            if len(data['datasets']) == 0:
                print("   ✅ All data successfully deleted!")
            else:
                for dataset in data['datasets']:
                    print(f"   📊 Remaining: {dataset['symbol']} {dataset['timeframe']}: {dataset['record_count']:,} records")
        else:
            print(f"   ❌ Error: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 40)
    print("✅ Delete endpoint testing completed!")

if __name__ == "__main__":
    test_delete_endpoints()