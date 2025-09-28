#!/usr/bin/env python3
"""
Quick API Test Script
Tests the API endpoints to see if they're working
"""

import requests
import json

def test_api_endpoints():
    base_url = "http://localhost:8080"
    
    print("🔥 Testing Open Alpha API Endpoints")
    print("=" * 50)
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/api/status")
        print(f"✅ Status API: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Status API failed: {e}")
    
    # Test positions endpoint
    try:
        response = requests.get(f"{base_url}/api/positions")
        print(f"✅ Positions API: {response.status_code}")
        data = response.json()
        print(f"   Positions: {len(data.get('positions', []))}")
    except Exception as e:
        print(f"❌ Positions API failed: {e}")
    
    # Test bot pause
    try:
        response = requests.post(f"{base_url}/api/bot/pause")
        print(f"✅ Bot Pause API: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Bot Pause API failed: {e}")
    
    # Test bot resume
    try:
        response = requests.post(f"{base_url}/api/bot/resume")
        print(f"✅ Bot Resume API: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Bot Resume API failed: {e}")
    
    # Test emergency stop
    try:
        response = requests.post(f"{base_url}/api/bot/emergency-stop")
        print(f"✅ Emergency Stop API: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Emergency Stop API failed: {e}")

if __name__ == "__main__":
    test_api_endpoints()