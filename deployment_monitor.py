#!/usr/bin/env python3
"""
🔍 DEPLOYMENT MONITOR
====================

Quick script to check DigitalOcean App Platform deployment status.
Monitors logs and validates critical components are working.
"""

import requests
import time
import json
import sys
from datetime import datetime

class DeploymentMonitor:
    def __init__(self, app_url: str = "https://bybit-bot-app.ondigitalocean.app"):
        # You'll need to update this with your actual DigitalOcean app URL
        self.app_url = app_url
        self.endpoints_to_check = [
            "/health",
            "/", 
            "/api/portfolio/status",
            "/api/ml-risk-metrics"
        ]
        
    def monitor_deployment(self):
        """Monitor deployment status"""
        print("🔍 DEPLOYMENT STATUS MONITOR")
        print("=" * 50)
        print(f"App URL: {self.app_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()
        
        # Check each endpoint
        results = {}
        for endpoint in self.endpoints_to_check:
            url = f"{self.app_url}{endpoint}"
            status = self._check_endpoint(url)
            results[endpoint] = status
            
        # Generate report
        self._generate_report(results)
        
        # Return overall status
        all_healthy = all(r["status"] == "healthy" for r in results.values())
        return 0 if all_healthy else 1
        
    def _check_endpoint(self, url: str) -> dict:
        """Check single endpoint status"""
        print(f"🌐 Checking: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                print(f"   ✅ Status 200 - Response time: {response.elapsed.total_seconds():.2f}s")
                
                # Try to parse JSON for health endpoint
                if "/health" in url:
                    try:
                        health_data = response.json()
                        status = health_data.get("status", "unknown")
                        print(f"   🏥 Health Status: {status}")
                        return {
                            "status": "healthy" if status == "healthy" else "degraded",
                            "response_time": response.elapsed.total_seconds(),
                            "details": health_data
                        }
                    except:
                        pass
                        
                return {
                    "status": "healthy", 
                    "response_time": response.elapsed.total_seconds(),
                    "content_length": len(response.content)
                }
                
            elif 400 <= response.status_code < 500:
                print(f"   ⚠️ Status {response.status_code} - Client error")
                return {"status": "client_error", "code": response.status_code}
                
            else:
                print(f"   ❌ Status {response.status_code} - Server error")
                return {"status": "server_error", "code": response.status_code}
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout - Endpoint not responding")
            return {"status": "timeout"}
            
        except requests.exceptions.ConnectionError:
            print(f"   🔌 Connection Error - App may not be deployed yet")
            return {"status": "connection_error"}
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {"status": "error", "details": str(e)}
            
    def _generate_report(self, results: dict):
        """Generate deployment status report"""
        print("\n" + "=" * 50)
        print("📊 DEPLOYMENT STATUS REPORT")
        print("=" * 50)
        
        healthy_count = sum(1 for r in results.values() if r["status"] == "healthy")
        total_count = len(results)
        
        if healthy_count == total_count:
            print("🟢 DEPLOYMENT STATUS: FULLY OPERATIONAL")
        elif healthy_count > 0:
            print("🟡 DEPLOYMENT STATUS: PARTIALLY OPERATIONAL") 
        else:
            print("🔴 DEPLOYMENT STATUS: NOT OPERATIONAL")
            
        print(f"\nEndpoints Checked: {total_count}")
        print(f"✅ Healthy: {healthy_count}")
        print(f"❌ Issues: {total_count - healthy_count}")
        
        # Detailed results
        print("\nDetailed Results:")
        for endpoint, result in results.items():
            status_icon = {
                "healthy": "✅", 
                "degraded": "⚠️",
                "client_error": "⚠️", 
                "server_error": "❌",
                "timeout": "⏰",
                "connection_error": "🔌",
                "error": "❌"
            }.get(result["status"], "❓")
            
            print(f"  {status_icon} {endpoint}: {result['status']}")
            
            if "response_time" in result:
                print(f"    Response time: {result['response_time']:.2f}s")
            if "code" in result:
                print(f"    HTTP Code: {result['code']}")
                
        # Next steps
        if healthy_count == total_count:
            print("\n🎉 All systems operational! Your AI trading bot is live.")
            print("   Visit the dashboard to see the unified interface.")
        elif healthy_count > 0:
            print("\n🔧 Some endpoints need attention, but core system is running.")
            print("   Check DigitalOcean logs for specific error details.")
        else:
            print("\n🚨 System not responding. Check deployment status:")
            print("   1. Verify app is deployed and running")
            print("   2. Check DigitalOcean App Platform console")
            print("   3. Review build and runtime logs")

def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor DigitalOcean deployment")
    parser.add_argument("--url", help="App URL to monitor", 
                       default="https://bybit-bot-app.ondigitalocean.app")
    parser.add_argument("--wait", type=int, help="Wait time before checking (seconds)", default=0)
    
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"⏳ Waiting {args.wait} seconds for deployment to complete...")
        time.sleep(args.wait)
    
    monitor = DeploymentMonitor(args.url)
    exit_code = monitor.monitor_deployment()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()