"""
Health Check Test Script
========================

Quick test to validate all critical endpoints are working
"""
import requests
import time
import json

def test_endpoints():
    """Test critical endpoints"""
    base_url = "http://localhost:8080"
    
    endpoints = [
        "/health",
        "/api/status", 
        "/api/trades/testnet",
        "/api/positions",
        "/api/multi-balance"
    ]
    
    print("🔍 Testing Critical Endpoints...")
    print("=" * 50)
    
    for endpoint in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ {endpoint}: OK ({response.status_code})")
                if endpoint == "/health":
                    data = response.json()
                    print(f"   Status: {data.get('status', 'Unknown')}")
                    print(f"   Version: {data.get('version', 'Unknown')}")
            else:
                print(f"⚠️  {endpoint}: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint}: Connection refused (server not running)")
        except requests.exceptions.Timeout:
            print(f"⏰ {endpoint}: Timeout")
        except Exception as e:
            print(f"❌ {endpoint}: Error - {str(e)}")
    
    print("=" * 50)
    print("✅ Endpoint testing complete")

if __name__ == "__main__":
    test_endpoints()