#!/usr/bin/env python3
"""
Production Deployment Monitor
Monitor the production deployment and check for completion.
"""

import time
import requests
import sys

def check_health(url, timeout=5):
    """Check if the health endpoint is responding."""
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def monitor_deployment(base_url="https://bybit-bot-mm6ub.ondigitalocean.app"):
    """Monitor deployment progress."""
    
    print("🔍 Monitoring Production Deployment")
    print("=" * 40)
    print(f"🌐 Target URL: {base_url}")
    print("")
    
    attempts = 0
    max_attempts = 20
    
    while attempts < max_attempts:
        attempts += 1
        
        print(f"📡 Attempt {attempts}/{max_attempts}: Checking health endpoint...")
        
        if check_health(base_url):
            print("✅ Deployment successful! Server is responding.")
            print(f"🚀 Application available at: {base_url}")
            return True
        
        print("⏳ Server not ready yet, waiting 30 seconds...")
        time.sleep(30)
    
    print("❌ Deployment appears to have issues after maximum attempts.")
    return False

if __name__ == "__main__":
    success = monitor_deployment()
    sys.exit(0 if success else 1)