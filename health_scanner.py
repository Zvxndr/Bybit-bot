#!/usr/bin/env python3
"""
🏥 DEPLOYMENT HEALTH SCANNER
============================

Scans for any active DigitalOcean App Platform deployments
and provides deployment status information.
"""

import requests
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_url_fast(url, timeout=5):
    """Fast URL check with minimal overhead"""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return (url, response.status_code, "active")
    except requests.exceptions.Timeout:
        return (url, 0, "timeout")
    except requests.exceptions.ConnectionError:
        return (url, 0, "no_connection")
    except Exception as e:
        return (url, 0, f"error: {str(e)[:30]}")

def scan_deployment_status():
    """Comprehensive deployment status scan"""
    
    print("🏥 DEPLOYMENT HEALTH SCANNER")
    print("=" * 45)
    print("⚡ Fast scanning possible deployment URLs...")
    print()
    
    # Comprehensive URL list
    base_patterns = [
        "bybit-bot", "bybitbot", "bybit", "trading-bot", 
        "ai-trading-bot", "bybit-api", "bybit-dashboard",
        "bybit-trading", "app"
    ]
    
    suffixes = ["", "-app", "-fresh", "-v1", "-v2", "-main", "-prod"]
    
    urls_to_check = []
    for base in base_patterns:
        for suffix in suffixes:
            urls_to_check.append(f"https://{base}{suffix}.ondigitalocean.app")
    
    # Add some common hash patterns
    for i in [12345, 54321, 99999, 11111, 67890]:
        urls_to_check.extend([
            f"https://bybit-bot-{i}.ondigitalocean.app",
            f"https://app-{i}.ondigitalocean.app"
        ])
    
    print(f"🔍 Scanning {len(urls_to_check)} URLs with parallel requests...")
    
    # Parallel scanning for speed
    active_deployments = []
    timeouts = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_url = {executor.submit(check_url_fast, url): url for url in urls_to_check}
        
        for future in as_completed(future_to_url):
            url, status_code, status = future.result()
            
            if status == "active":
                active_deployments.append((url, status_code))
                print(f"✅ FOUND: {url} (Status: {status_code})")
            elif status == "timeout":
                timeouts.append(url)
                print(f"⏰ TIMEOUT: {url} (may be starting)")
            # Skip printing "no_connection" to reduce noise
    
    print()
    print("=" * 45)
    print("📊 SCAN RESULTS")
    print("=" * 45)
    
    if active_deployments:
        print("🎉 ACTIVE DEPLOYMENTS FOUND:")
        for url, status in active_deployments:
            health_status = check_health_endpoint(url)
            print(f"   ✅ {url}")
            print(f"      📊 HTTP Status: {status}")
            print(f"      🏥 Health: {health_status}")
            print()
        
        return active_deployments
        
    elif timeouts:
        print("⏰ POTENTIAL DEPLOYMENTS (timing out):")
        for url in timeouts[:5]:  # Show first 5
            print(f"   🟡 {url}")
        print(f"   ... and {len(timeouts)-5} more" if len(timeouts) > 5 else "")
        print()
        print("💡 These might be starting up or have slow response times.")
        
    else:
        print("❌ NO ACTIVE DEPLOYMENTS DETECTED")
        print()
        print("🔧 NEXT STEPS:")
        print("   1. Check DigitalOcean App Platform console directly")
        print("   2. Verify GitHub repository is properly connected")
        print("   3. Check deployment logs for build issues")
        print("   4. Ensure app is configured to deploy from 'main' branch")
        print("   5. Look for any custom domain configurations")
    
    return active_deployments

def check_health_endpoint(base_url):
    """Check health endpoint for deployment"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            try:
                data = response.json()
                return f"✅ Healthy ({data.get('status', 'ok')})"
            except:
                return "✅ Responding"
        else:
            return f"⚠️ Status {response.status_code}"
    except:
        return "❌ Health check failed"

def main():
    deployments = scan_deployment_status()
    
    if deployments:
        print("🚀 SUCCESS! Your AI Trading Bot is deployed and accessible.")
        print("Visit any of the URLs above to access your dashboard.")
    else:
        print("🔍 No active deployments found. Check DigitalOcean console for:")
        print("   - Build status and logs") 
        print("   - App configuration")
        print("   - Domain settings")

if __name__ == "__main__":
    main()