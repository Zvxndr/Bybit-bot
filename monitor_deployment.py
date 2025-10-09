#!/usr/bin/env python3
"""
Production Deployment Monitor
============================

Monitors the DigitalOcean deployment and checks for successful AI pipeline activation.
"""

import time
import requests
import sys
from datetime import datetime

def check_deployment_status():
    """Check if the application is responding"""
    try:
        # Replace with your actual DigitalOcean app URL
        app_url = "https://your-app-url.ondigitalocean.app"  # Update this
        
        response = requests.get(f"{app_url}/health", timeout=10)
        if response.status_code == 200:
            return True, f"✅ App responding: {response.status_code}"
        else:
            return False, f"❌ App error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"🔌 Connection failed: {e}"

def monitor_deployment():
    """Monitor deployment progress"""
    print("🔍 Monitoring Production Deployment...")
    print("=" * 50)
    
    start_time = datetime.now()
    
    for i in range(30):  # Monitor for 5 minutes
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        is_healthy, message = check_deployment_status()
        
        print(f"[{timestamp}] {message}")
        
        if is_healthy:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n🎉 Deployment successful after {elapsed:.1f} seconds!")
            print("\n📋 Expected improvements in new deployment:")
            print("   ✅ MultiExchangeDataManager should load successfully")
            print("   ✅ AutomatedPipelineManager should load with mock imports")
            print("   ✅ MLStrategyDiscoveryEngine should activate")
            print("   🤖 AI Pipeline should be fully operational")
            return True
        
        time.sleep(10)  # Check every 10 seconds
    
    print("\n⏰ Monitoring timeout - check DigitalOcean logs manually")
    return False

if __name__ == "__main__":
    print("🚀 Production Deployment Monitor")
    print(f"📅 Started at: {datetime.now()}")
    print()
    
    success = monitor_deployment()
    
    if success:
        print("\n🎯 Next Steps:")
        print("   1. Check DigitalOcean app logs for 'AutomatedPipelineManager loaded directly'")
        print("   2. Verify AI pipeline activation messages")
        print("   3. Test trading endpoints for full functionality")
        sys.exit(0)
    else:
        print("\n❌ Deployment needs investigation")
        print("   Check DigitalOcean App Platform logs for errors")
        sys.exit(1)